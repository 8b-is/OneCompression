#!/usr/bin/env python3
"""Counterbalanced fresh-process speed harness: packed HQ vs vLLM 0.20 TQ.

Only two native serving implementations are admitted:

* ``HQ``: the faithful fixed recipe in ``hayate_full_{native,triton}.py``;
* ``TurboQuant``: unmodified upstream vLLM 0.20 Triton store/decode kernels.

The controller never initializes CUDA.  It launches every sample in a fresh
worker process in an ABBA/BAAB order, so Triton compilation, allocator state,
and backend registries cannot leak between methods.  Warmup/compilation is
excluded from CUDA-event timings.  Both methods see identical K/V/Q tensors,
paged-cache topology, sequence/head shapes, split count, input dtype, and GPU;
their slot byte width is intentionally algorithm-specific and reported as the
physical cache footprint.  Fake quantization, public unpacked payload APIs,
and the historical key-HQ/value-TQ hybrid are not accepted by this harness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "KVCOMP_NATIVE_SPEED_RESULT="
FORMAL_CONTRACT = Path("benchmark/kvcomp-codec/hayatequant_native_speed_formal_contract.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor) -> str:
    """Hash exact tensor bytes without relying on NumPy dtype support."""

    raw = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_fingerprint(tensor) -> dict[str, object]:
    value = tensor.detach().contiguous()
    finite = value.float().isfinite()
    return {
        "sha256": _tensor_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "all_finite": bool(finite.all().item()),
        "sum_fp64": float(value.double().sum().item()),
        "l2_fp64": float(value.double().norm().item()),
        "max_abs_fp32": float(value.float().abs().max().item()),
    }


def _correctness_metrics(native, reference, *, rtol: float, atol: float) -> dict[str, object]:
    """Return and enforce an elementwise packed-attention correctness gate."""

    native_f = native.float()
    reference_f = reference.float()
    finite = bool(native_f.isfinite().all().item() and reference_f.isfinite().all().item())
    difference = (native_f - reference_f).abs()
    allowed = atol + rtol * reference_f.abs()
    violations = difference > allowed
    reference_norm = reference_f.double().norm().clamp_min(1e-30)
    passed = finite and not bool(violations.any().item())
    result = {
        "passed": passed,
        "all_finite": finite,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(difference.max().item()),
        "relative_l2_error": float(difference.double().norm().div(reference_norm).item()),
        "violation_count": int(violations.sum().item()),
        "element_count": int(reference.numel()),
        "native_output": _tensor_fingerprint(native),
        "dense_reference_output": _tensor_fingerprint(reference),
    }
    if not passed:
        raise AssertionError(
            "packed attention disagrees with independently dequantized dense attention: "
            f"finite={finite}, violations={result['violation_count']}, "
            f"max_abs={result['max_abs_error']:.8g}, rtol={rtol}, atol={atol}"
        )
    return result


def _dense_gqa_attention(torch, query, keys, values, *, scale: float, output_dtype=None):
    """Materialized fp32 GQA reference for one decode query.

    ``keys`` and ``values`` are [1, Hk, S, D].  They must be reconstructed
    from the packed cache, never from the unquantized source tensors.
    """

    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("dense reference expects matching [1,Hk,S,D] packed reconstructions")
    hq = query.shape[1]
    hk = keys.shape[1]
    if hq % hk:
        raise ValueError("query heads must be divisible by KV heads")
    groups = hq // hk
    key_gqa = keys[0].float().repeat_interleave(groups, dim=0)
    value_gqa = values[0].float().repeat_interleave(groups, dim=0)
    logits = torch.einsum("bhd,hsd->bhs", query.float(), key_gqa) * scale
    weights = logits.softmax(dim=-1)
    output = torch.einsum("bhs,hsd->bhd", weights, value_gqa).to(output_dtype or query.dtype)
    return output, logits


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_worker_result(payload: str) -> dict:
    result = json.loads(payload, parse_constant=_reject_nonfinite_json_constant)
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON object")
    return result


def _selected_nvidia_smi_snapshot(gpu_uuid: str) -> dict[str, object]:
    """Record the selected physical GPU and reject competing compute PIDs."""

    fields = (
        "uuid",
        "driver_version",
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.limit",
        "temperature.gpu",
        "utilization.gpu",
        "memory.used",
    )
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    normalized_uuid = gpu_uuid.removeprefix("GPU-").lower()
    selected = None
    for raw_line in gpu_query.stdout.splitlines():
        values = [item.strip() for item in raw_line.split(",")]
        if len(values) != len(fields):
            raise RuntimeError(f"malformed nvidia-smi GPU row: {raw_line!r}")
        if values[0].removeprefix("GPU-").lower() == normalized_uuid:
            selected = dict(zip(fields, values, strict=True))
            break
    if selected is None:
        raise RuntimeError(f"selected CUDA GPU UUID {gpu_uuid!r} is absent from nvidia-smi")

    apps_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    selected_apps = []
    for raw_line in apps_query.stdout.splitlines():
        values = [item.strip() for item in raw_line.split(",")]
        if len(values) != 3 or values[0].removeprefix("GPU-").lower() != normalized_uuid:
            continue
        selected_apps.append({"pid": int(values[1]), "used_memory_mib": int(values[2].split()[0])})
    competing = [row for row in selected_apps if row["pid"] != os.getpid()]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": selected,
        "this_worker_pid": os.getpid(),
        "selected_gpu_compute_processes": selected_apps,
        "competing_compute_processes": competing,
        "exclusive_compute_process": not competing,
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "n": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": _percentile(samples, 0.10),
        "p90_ms": _percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _event_samples(torch, fn: Callable[[], object], iters: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)]


def _worker(args: argparse.Namespace) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if version("vllm") != "0.20.0":
        raise RuntimeError(f"this comparison pins upstream vLLM 0.20.0, got {version('vllm')}")
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_full_dequant_kv,
        _use_fp8_e4b15,
        triton_turboquant_decode_attention,
    )
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.triton_utils import triton

    from hayatequant_vllm_full_native_smoke import (
        _luts,
        _recipe,
        _scoring_tables,
        _tables,
    )
    from vllm_plugins.kvcomp.hayate_full_triton import (
        triton_full_hayatequant_decode_attention,
        triton_full_hayatequant_dequant_kv,
        triton_full_hayatequant_store,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_uuid = str(getattr(props, "uuid", ""))
    if not gpu_uuid:
        raise RuntimeError("CUDA device did not expose a physical UUID")
    dtype = getattr(torch, args.dtype)
    n, hk, hq, d = args.tokens, args.kv_heads, args.query_heads, args.head_dim
    page_size = args.page_size
    pages = (n + page_size - 1) // page_size
    slots = torch.arange(n, device=device, dtype=torch.int32)
    step_slot = slots[-1:].contiguous()
    key = torch.randn(n, hk, d, device=device, dtype=dtype)
    value = torch.randn_like(key)
    step_key, step_value = key[-1:].contiguous(), value[-1:].contiguous()
    query = torch.randn(1, hq, d, device=device, dtype=dtype)
    block_table = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([n], device=device, dtype=torch.int32)
    scale = d**-0.5

    if args.worker_variant == "HQ":
        recipe = _recipe(args)
        layout = recipe.layout(d)
        cache = torch.full(
            (pages, page_size, hk, layout["slot_size_aligned"]),
            0xA5,
            device=device,
            dtype=torch.uint8,
        )
        _, kr, vr, kcb, vcb = _tables(recipe, d, device)
        key_lut, value_lut = _luts(recipe, d, device)
        key_score, key_csq, value_score, value_csq = _scoring_tables(recipe, kcb, vcb)
        store_aux = {
            "key_lut": key_lut,
            "value_lut": value_lut,
            "key_scoring_codebook": key_score,
            "key_scoring_csq": key_csq,
            "value_scoring_codebook": value_score,
            "value_scoring_csq": value_csq,
        }
        mid = torch.empty(1, hq, args.max_splits, d + 1, device=device, dtype=torch.float32)
        rotated = torch.empty(1, hq, d, device=device, dtype=torch.float32)
        lse = torch.empty(1, hq, device=device, dtype=torch.float32)

        def store_all():
            return triton_full_hayatequant_store(
                key,
                value,
                cache,
                slots,
                kr,
                vr,
                kcb,
                vcb,
                recipe,
                **store_aux,
            )

        def store_step():
            return triton_full_hayatequant_store(
                step_key,
                step_value,
                cache,
                step_slot,
                kr,
                vr,
                kcb,
                vcb,
                recipe,
                **store_aux,
            )

        def decode():
            return triton_full_hayatequant_decode_attention(
                query,
                cache,
                block_table,
                seq_lens,
                kr,
                vr,
                kcb,
                vcb,
                recipe,
                scale,
                mid_o_buf=mid,
                rotated_output_buf=rotated,
                lse_buf=lse,
                max_num_kv_splits=args.max_splits,
            )

        method_contract = {
            "implementation": "faithful_packed_fixed_HQ_native",
            "recipe": recipe.provenance(d),
            "logical_slot_size": layout["slot_size"],
            "slot_size_aligned": layout["slot_size_aligned"],
            "exact_assignment_implementation": ("all_codewords_tensorcore_score_fused_argmax"),
            "approximate_lut": False,
            "persistent_encoder_acceleration_bytes": 0,
            "physical_cache_bytes_includes_encoder_acceleration": False,
            "upstream_turboquant_modified": False,
            "table_sha256": {
                "key_rotation": _tensor_sha256(kr),
                "value_rotation": _tensor_sha256(vr),
                "key_codebook": _tensor_sha256(kcb),
                "value_codebook": _tensor_sha256(vcb),
            },
        }

        def dequantized_reference_inputs():
            k_out = torch.empty(1, hk, pages * page_size, d, device=device, dtype=torch.float32)
            v_out = torch.empty_like(k_out)
            triton_full_hayatequant_dequant_kv(
                cache,
                block_table,
                kcb,
                vcb,
                kr,
                vr,
                recipe,
                k_out,
                v_out,
            )
            return query, k_out[:, :, :n], v_out[:, :, :n]

    elif args.worker_variant == "TurboQuant":
        if args.avg_bits == 3:
            dtype_name = "turboquant_3bit_nc"
        elif args.avg_bits == 4:
            dtype_name = "turboquant_4bit_nc"
        else:
            raise ValueError("upstream vLLM 0.20 has no native 2-bit TurboQuant preset")
        tq = TurboQuantConfig.from_cache_dtype(dtype_name, d)
        cache = torch.full(
            (pages, page_size, hk, tq.slot_size_aligned),
            0xA5,
            device=device,
            dtype=torch.uint8,
        )
        rotation = _build_hadamard(d, str(device)).contiguous()
        centroids = get_centroids(d, tq.centroid_bits).to(device=device, dtype=torch.float32)
        sorted_centroids, _ = centroids.sort()
        midpoints = ((sorted_centroids[:-1] + sorted_centroids[1:]) / 2).contiguous()
        mid = torch.empty(1, hq, args.max_splits, d + 1, device=device, dtype=torch.float32)
        output = torch.empty(1, hq, d, device=device, dtype=torch.float32)
        lse = torch.empty(1, hq, device=device, dtype=torch.float32)

        def store_tensors(k, v, mapping):
            return triton_turboquant_store(
                k,
                v,
                cache,
                mapping,
                rotation,
                midpoints,
                mse_bits=tq.key_mse_bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_fp8=tq.key_fp8,
            )

        def store_all():
            return store_tensors(key, value, slots)

        def store_step():
            return store_tensors(step_key, step_value, step_slot)

        def decode():
            return triton_turboquant_decode_attention(
                query=query,
                kv_cache=cache,
                block_table=block_table,
                seq_lens=seq_lens,
                Pi=rotation,
                PiT=rotation,
                centroids=centroids,
                scale=scale,
                mse_bits=tq.key_mse_bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_fp8=tq.key_fp8,
                norm_correction=tq.norm_correction,
                mid_o_buf=mid,
                output_buf=output,
                lse_buf=lse,
                max_num_kv_splits=args.max_splits,
            )

        method_contract = {
            "implementation": "unmodified_upstream_vllm_0.20_TurboQuant_native",
            "kv_cache_dtype": dtype_name,
            "key_bits": tq.key_quant_bits,
            "value_bits": tq.value_quant_bits,
            "logical_slot_size": tq.slot_size,
            "slot_size_aligned": tq.slot_size_aligned,
            "upstream_turboquant_modified": False,
            "table_sha256": {
                "rotation": _tensor_sha256(rotation),
                "centroids": _tensor_sha256(centroids),
                "midpoints": _tensor_sha256(midpoints),
            },
        }

        def dequantized_reference_inputs():
            k_out = torch.empty(1, hk, pages * page_size, d, device=device, dtype=torch.float32)
            v_out = torch.empty_like(k_out)
            _tq_full_dequant_kv[(pages * page_size, hk)](
                cache,
                block_table,
                centroids,
                k_out,
                v_out,
                k_out.stride(0),
                k_out.stride(1),
                k_out.stride(2),
                v_out.stride(0),
                v_out.stride(1),
                v_out.stride(2),
                cache.stride(0),
                cache.stride(1),
                cache.stride(2),
                block_table.stride(0),
                HEAD_DIM=d,
                BLOCK_SIZE=page_size,
                NUM_KV_HEADS=hk,
                MSE_BYTES=math.ceil(d * tq.key_mse_bits / 8),
                KPS=tq.key_packed_size,
                VQB=tq.effective_value_quant_bits,
                VAL_DATA_BYTES=math.ceil(d * tq.effective_value_quant_bits / 8),
                MSE_BITS=tq.key_mse_bits,
                KEY_FP8=1 if tq.key_fp8 else 0,
                BLOCK_D=triton.next_power_of_2(d),
                NORM_CORRECTION=1 if tq.norm_correction else 0,
                FP8_E4B15=_use_fp8_e4b15(device.index or 0),
                num_warps=4,
            )
            q_reference = (query.float() @ rotation).contiguous()
            return q_reference, k_out[:, :, :n], v_out[:, :, :n]

    else:
        raise ValueError(f"unexpected worker variant {args.worker_variant}")

    # First calls compile Triton and populate all persistent tables/cache.  They
    # are intentionally outside measured samples for both methods.  Before any
    # timing, independently dequantize the physical packed cache and require
    # the fused attention result to match materialized fp32 GQA attention.
    store_all()
    native_output = decode()
    reference_query, reference_keys, reference_values = dequantized_reference_inputs()
    dense_output, dense_logits = _dense_gqa_attention(
        torch,
        reference_query,
        reference_keys,
        reference_values,
        scale=scale,
        output_dtype=dtype,
    )
    torch.cuda.synchronize()
    attention_atol = 1.5e-2 if dtype == torch.bfloat16 else 4e-3
    correctness = _correctness_metrics(
        native_output,
        dense_output,
        rtol=2e-3,
        atol=attention_atol,
    )
    correctness["dense_packed_key_logits"] = _tensor_fingerprint(dense_logits)
    logical_slot_size = int(method_contract["logical_slot_size"])
    logical_cache = cache[..., :logical_slot_size]
    padding = cache[..., logical_slot_size:]
    padding_untouched = bool((padding == 0xA5).all().item()) if padding.numel() else True
    if not padding_untouched:
        raise AssertionError("native store modified aligned slot padding")
    correctness["packed_cache"] = {
        "logical_bytes_sha256": _tensor_sha256(logical_cache),
        "logical_bytes": int(logical_cache.numel()),
        "aligned_bytes": int(cache.numel()),
        "padding_bytes_per_slot": int(cache.shape[-1] - logical_slot_size),
        "padding_sentinel_untouched": padding_untouched,
    }
    environment_before_timing = _selected_nvidia_smi_snapshot(gpu_uuid)
    if not environment_before_timing["exclusive_compute_process"]:
        raise RuntimeError(
            "selected GPU has competing compute processes before timing: "
            f"{environment_before_timing['competing_compute_processes']}"
        )
    for _ in range(args.warmups):
        store_all()
        decode()
        store_step()
        decode()
    torch.cuda.synchronize()

    store_samples = _event_samples(torch, store_all, args.iters)
    decode_samples = _event_samples(torch, decode, args.iters)

    def serving_step():
        store_step()
        return decode()

    step_samples = _event_samples(torch, serving_step, args.iters)
    graph_samples = None
    if args.measure_cudagraph:
        # vLLM serves uniform decode batches under CUDA graphs.  Capture the
        # exact same one-token store + packed attention sequence for both
        # methods after all GEMM/Triton warmups.  Graph construction and its
        # private memory pool are outside the samples.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                serving_step()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = serving_step()
        graph.replay()
        torch.cuda.synchronize()
        graph_correctness = _correctness_metrics(
            graph_output,
            dense_output,
            rtol=2e-3,
            atol=attention_atol,
        )
        graph_samples = _event_samples(torch, graph.replay, args.iters)
    environment_after_timing = _selected_nvidia_smi_snapshot(gpu_uuid)
    if not environment_after_timing["exclusive_compute_process"]:
        raise RuntimeError(
            "selected GPU has competing compute processes after timing: "
            f"{environment_after_timing['competing_compute_processes']}"
        )
    return {
        "worker_variant": args.worker_variant,
        "trial_index": args.trial_index,
        "pid": os.getpid(),
        "gpu": {
            "name": props.name,
            "uuid": (
                str(getattr(props, "uuid")) if getattr(props, "uuid", None) is not None else None
            ),
            "total_memory_bytes": props.total_memory,
            "capability": [props.major, props.minor],
        },
        "gpu_environment": {
            "before_timing": environment_before_timing,
            "after_timing": environment_after_timing,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_executable_sha256": _sha(Path(sys.executable)),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "vllm_version": version("vllm"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "method_contract": method_contract,
        "correctness_gate": {
            "passed": True,
            "reference": "dense fp32 GQA over independently dequantized physical packed cache",
            "eager": correctness,
            **({"cudagraph": graph_correctness} if graph_samples is not None else {}),
        },
        "physical_cache_bytes": int(cache.numel()),
        "shape_contract": {
            "tokens": n,
            "page_size": page_size,
            "pages": pages,
            "kv_heads": hk,
            "query_heads": hq,
            "head_dim": d,
            "dtype": args.dtype,
            "max_kv_splits": args.max_splits,
        },
        "timings": {
            "prefill_store": _summary(store_samples),
            "decode_attention": _summary(decode_samples),
            "serving_step_store1_plus_decode": _summary(step_samples),
            **(
                {"cudagraph_serving_step_store1_plus_decode": _summary(graph_samples)}
                if graph_samples is not None
                else {}
            ),
        },
    }


def _worker_command(args, variant: str, trial: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-variant",
        variant,
        "--trial-index",
        str(trial),
        "--avg-bits",
        str(args.avg_bits),
        "--recipe",
        args.recipe,
        "--head-dim",
        str(args.head_dim),
        "--kv-heads",
        str(args.kv_heads),
        "--query-heads",
        str(args.query_heads),
        "--tokens",
        str(args.tokens),
        "--page-size",
        str(args.page_size),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--iters",
        str(args.iters),
        "--max-splits",
        str(args.max_splits),
        "--dist",
        args.dist,
        "--codebook-init",
        args.codebook_init,
        "--rotation-mode",
        args.rotation_mode,
    ]
    if args.allow_canonical_k10v6_b4:
        command.append("--allow-canonical-k10v6-b4")
    if args.assignment_mode is not None:
        command.extend(("--assignment-mode", args.assignment_mode))
    if args.lut_bins is not None:
        command.extend(("--lut-bins", str(args.lut_bins)))
    if args.lut_candidates is not None:
        command.extend(("--lut-candidates", str(args.lut_candidates)))
    if not args.measure_cudagraph:
        command.append("--no-measure-cudagraph")
    role_values = (
        ("--key-code-bits", args.key_code_bits),
        ("--key-block-size", args.key_block_size),
        ("--value-code-bits", args.value_code_bits),
        ("--value-block-size", args.value_block_size),
    )
    for flag, value in role_values:
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _controller(args: argparse.Namespace) -> dict:
    initial_source_state = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_porcelain": _git("status", "--porcelain"),
    }
    if args.avg_bits not in (3, 4):
        raise ValueError("fair native TQ comparison is defined only at average rates 3 and 4")
    if args.trials < 4 or args.trials % 4:
        raise ValueError("trials must be a positive multiple of four (ABBA blocks)")
    orders: list[str] = []
    for block in range(args.trials // 4):
        orders.extend(
            ("HQ", "TurboQuant", "TurboQuant", "HQ")
            if block % 2 == 0
            else ("TurboQuant", "HQ", "HQ", "TurboQuant")
        )
    rows = []
    for trial, variant in enumerate(orders):
        command = _worker_command(args, variant, trial)
        completed = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            check=False,
            text=True,
            capture_output=True,
        )
        worker_log_root = Path(f"{args.output}.workers") if args.output else None
        if worker_log_root is not None:
            stem = f"trial-{trial:03d}-{variant.lower()}"
            worker_log_root.mkdir(parents=True, exist_ok=True)
            (worker_log_root / f"{stem}.stdout").write_text(completed.stdout, encoding="utf-8")
            (worker_log_root / f"{stem}.stderr").write_text(completed.stderr, encoding="utf-8")
            _write_json(
                worker_log_root / f"{stem}.command.json",
                {
                    "command": command,
                    "cwd": str(REPO),
                    "exit_status": completed.returncode,
                    "variant": variant,
                    "trial": trial,
                },
            )
        if completed.returncode:
            if args.output:
                _write_json(
                    Path(f"{args.output}.failed.json"),
                    {
                        "schema": {
                            "name": "onecomp.hq_vs_upstream_tq_native_fair_speed_failure",
                            "version": 1,
                        },
                        "status": "failed_worker",
                        "variant": variant,
                        "trial": trial,
                        "exit_status": completed.returncode,
                        "command": command,
                        "worker_log_root": str(worker_log_root),
                        "completed_rows": rows,
                        "initial_source_state": initial_source_state,
                    },
                )
            raise RuntimeError(
                f"fresh worker {variant} trial {trial} failed with "
                f"exit={completed.returncode}; stdout={completed.stdout[-4000:]} "
                f"stderr={completed.stderr[-4000:]}"
            )
        line = next(
            (
                item
                for item in reversed(completed.stdout.splitlines())
                if item.startswith(RESULT_PREFIX)
            ),
            None,
        )
        if line is None:
            if args.output:
                _write_json(
                    Path(f"{args.output}.failed.json"),
                    {
                        "schema": {
                            "name": "onecomp.hq_vs_upstream_tq_native_fair_speed_failure",
                            "version": 1,
                        },
                        "status": "missing_worker_result",
                        "variant": variant,
                        "trial": trial,
                        "exit_status": completed.returncode,
                        "command": command,
                        "worker_log_root": str(worker_log_root),
                        "completed_rows": rows,
                        "initial_source_state": initial_source_state,
                    },
                )
            raise RuntimeError(
                f"worker {variant} emitted no result; stdout={completed.stdout[-2000:]} "
                f"stderr={completed.stderr[-2000:]}"
            )
        try:
            row = _load_worker_result(line[len(RESULT_PREFIX) :])
        except (json.JSONDecodeError, ValueError) as exc:
            if args.output:
                _write_json(
                    Path(f"{args.output}.failed.json"),
                    {
                        "schema": {
                            "name": "onecomp.hq_vs_upstream_tq_native_fair_speed_failure",
                            "version": 1,
                        },
                        "status": "invalid_worker_result_json",
                        "variant": variant,
                        "trial": trial,
                        "exit_status": completed.returncode,
                        "command": command,
                        "worker_log_root": str(worker_log_root),
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "completed_rows": rows,
                        "initial_source_state": initial_source_state,
                    },
                )
            raise RuntimeError(
                f"worker {variant} emitted invalid strict JSON in trial {trial}: {exc}"
            ) from exc
        row["worker_stderr_tail"] = completed.stderr[-1000:]
        rows.append(row)

    gpu_ids = {(row["gpu"]["name"], row["gpu"].get("uuid")) for row in rows}
    pids = {row["pid"] for row in rows}
    if len(gpu_ids) != 1:
        raise RuntimeError(f"workers did not use the same physical GPU: {gpu_ids}")
    if len(pids) != len(rows):
        raise RuntimeError("fresh-process contract violated: repeated worker PID")
    if not all(
        row.get("gpu_environment", {}).get(moment, {}).get("exclusive_compute_process") is True
        for row in rows
        for moment in ("before_timing", "after_timing")
    ):
        raise RuntimeError("a fresh worker observed a competing process on the selected GPU")
    gpu_strata = {
        (
            snapshot["gpu"]["driver_version"],
            snapshot["gpu"]["pstate"],
            snapshot["gpu"]["clocks.current.sm"],
            snapshot["gpu"]["clocks.current.memory"],
            snapshot["gpu"]["power.limit"],
        )
        for row in rows
        for snapshot in (
            row["gpu_environment"]["before_timing"],
            row["gpu_environment"]["after_timing"],
        )
    }
    if len(gpu_strata) != 1:
        raise RuntimeError(f"GPU driver/clock/power stratum changed across workers: {gpu_strata}")
    runtime_strata = {
        (
            row["runtime"]["python_executable_sha256"],
            row["runtime"]["torch_version"],
            row["runtime"]["cuda_runtime_version"],
            row["runtime"]["cudnn_version"],
            row["runtime"]["vllm_version"],
        )
        for row in rows
    }
    if len(runtime_strata) != 1:
        raise RuntimeError(f"worker software stratum changed: {runtime_strata}")
    for method in ("HQ", "TurboQuant"):
        method_rows = [row for row in rows if row["worker_variant"] == method]
        if not method_rows or not all(
            row.get("correctness_gate", {}).get("passed") is True for row in method_rows
        ):
            raise RuntimeError(f"{method} has a missing or failed packed correctness gate")
        deterministic_fields = {
            "logical_cache": {
                row["correctness_gate"]["eager"]["packed_cache"]["logical_bytes_sha256"]
                for row in method_rows
            },
            "native_output": {
                row["correctness_gate"]["eager"]["native_output"]["sha256"] for row in method_rows
            },
            "dense_reference": {
                row["correctness_gate"]["eager"]["dense_reference_output"]["sha256"]
                for row in method_rows
            },
            "dense_logits": {
                row["correctness_gate"]["eager"]["dense_packed_key_logits"]["sha256"]
                for row in method_rows
            },
        }
        if any(len(values) != 1 for values in deterministic_fields.values()):
            raise RuntimeError(
                f"{method} packed bytes/output changed across fresh processes: "
                f"{deterministic_fields}"
            )
        if args.measure_cudagraph:
            graph_hashes = {
                row["correctness_gate"]["cudagraph"]["native_output"]["sha256"]
                for row in method_rows
            }
            if len(graph_hashes) != 1 or graph_hashes != deterministic_fields["native_output"]:
                raise RuntimeError(f"{method} CUDA-graph output differs from eager output")
    aggregate = {}
    metrics = ["prefill_store", "decode_attention", "serving_step_store1_plus_decode"]
    if args.measure_cudagraph:
        metrics.append("cudagraph_serving_step_store1_plus_decode")
    for metric in metrics:
        medians = {
            method: [
                row["timings"][metric]["median_ms"]
                for row in rows
                if row["worker_variant"] == method
            ]
            for method in ("HQ", "TurboQuant")
        }
        hq = statistics.median(medians["HQ"])
        tq = statistics.median(medians["TurboQuant"])
        aggregate[metric] = {
            "fresh_process_medians_ms": medians,
            "HQ_median_of_process_medians_ms": hq,
            "TurboQuant_median_of_process_medians_ms": tq,
            "HQ_speedup_over_TurboQuant": tq / hq,
            "HQ_faster_point_estimate": hq < tq,
        }
    source_paths = [
        Path("vllm_plugins/kvcomp/hayate_full_config.py"),
        Path("vllm_plugins/kvcomp/hayate_full_triton.py"),
        Path("vllm_plugins/kvcomp/hayate_full_native.py"),
        Path("onecomp/kvcomp/codec/hayate/recipes.py"),
        Path("onecomp/kvcomp/codec/hayate/vq.py"),
        Path("onecomp/kvcomp/codec/hayate/triton_step.py"),
        Path("benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py"),
        FORMAL_CONTRACT,
    ]
    vllm_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "vllm"
    )
    upstream_paths = [
        vllm_root / "v1/attention/ops/triton_turboquant_store.py",
        vllm_root / "v1/attention/ops/triton_turboquant_decode.py",
        vllm_root / "model_executor/layers/quantization/turboquant/config.py",
    ]
    dirty = bool(initial_source_state["git_status_porcelain"])
    from vllm_plugins.kvcomp.hayate_full_config import fixed_hq_native_recipe

    expected = fixed_hq_native_recipe(args.avg_bits)
    hq_recipe = next(
        row["method_contract"]["recipe"] for row in rows if row["worker_variant"] == "HQ"
    )
    frozen_fields = {
        "key_code_bits_per_block": expected.key_code_bits_per_block,
        "key_block_size": expected.key_block_size,
        "value_code_bits_per_block": expected.value_code_bits_per_block,
        "value_block_size": expected.value_block_size,
        "dist": "gaussian",
        "key_scale_mode": "nc",
        "value_scale_mode": "ls_iter",
        "assignment_mode": "exact",
    }
    frozen_hq = args.recipe == "hq" and all(
        hq_recipe.get(name) == value for name, value in frozen_fields.items()
    )
    ineligible = []
    if dirty:
        ineligible.append("dirty_worktree")
    if not frozen_hq:
        ineligible.append("non_frozen_HQ_ablation_recipe")
    if args.trials < 20:
        ineligible.append("fewer_than_20_counterbalanced_fresh_process_trials")
    if args.warmups < 10:
        ineligible.append("fewer_than_10_warmups")
    if args.iters < 100:
        ineligible.append("fewer_than_100_timed_iterations")
    if not args.measure_cudagraph:
        ineligible.append("cudagraph_endpoint_missing")
    if args.dtype != "bfloat16":
        ineligible.append("formal_dtype_must_be_bfloat16")
    if args.page_size != 16:
        ineligible.append("formal_page_size_must_be_16")
    if args.output is None:
        ineligible.append("output_and_worker_logs_not_retained")
    elif not args.output.is_absolute():
        ineligible.append("formal_output_path_must_be_absolute")
    final_source_status = _git("status", "--porcelain")
    if final_source_status not in (None, ""):
        ineligible.append("source_worktree_changed_during_run")
    return {
        "schema": {"name": "onecomp.hq_vs_upstream_tq_native_fair_speed", "version": 2},
        "formal_eligible": False,
        "formal_cell_eligible": not ineligible,
        "formal_ineligibility_reasons": ["single_cell_requires_complete_matrix_aggregation"],
        "formal_cell_ineligibility_reasons": ineligible,
        "whole_matrix_claim_allowed": False,
        "comparison_contract": {
            "methods": ["HQ", "TurboQuant"],
            "proposal_variants": ["HQ"],
            "excluded": [
                "Adaptive HQ",
                "FibQuant",
                "BlockQuant",
                "hybrid",
                "fake_quant",
                "public_payload_api",
            ],
            "turboquant": "unmodified upstream vLLM 0.20 native kernels",
            "counterbalance_order": orders,
            "fresh_process_per_sample": True,
            "warmup_and_compilation_excluded": True,
            "cudagraph_decode_measured": args.measure_cudagraph,
            "same_gpu": True,
            "same_semantic_shapes_and_paged_layout": True,
            "algorithm_specific_physical_slot_width_reported": True,
            "packed_output_correctness_required_before_timing": True,
            "fresh_process_packed_bytes_and_outputs_deterministic": True,
            "worker_stdout_stderr_retained_when_output_is_set": True,
            "worker_log_root": str(Path(f"{args.output}.workers")) if args.output else None,
            "single_cell_never_authorizes_a_formal_speed_claim": True,
            "formal_matrix_contract": str(FORMAL_CONTRACT),
            "formal_matrix_contract_sha256": _sha(REPO / FORMAL_CONTRACT),
            "timing": "CUDA events with one final synchronization per sample vector",
        },
        "aggregate": aggregate,
        "workers": rows,
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            **initial_source_state,
            "git_status_porcelain_at_finish": final_source_status,
            "vllm_version": version("vllm"),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_executable_sha256": _sha(Path(sys.executable)),
            "source_sha256": {str(path): _sha(REPO / path) for path in source_paths},
            "upstream_vllm_sha256": {str(path): _sha(path) for path in upstream_paths},
            "synthetic_input_contract": {
                "generator": "torch.manual_seed on the worker CUDA device",
                "seed": args.seed,
                "key_value_query_distribution": "independent torch.randn",
                "model_weights_loaded": False,
                "dataset_loaded": False,
                "purpose": "native packed kernel shape benchmark, not model quality",
            },
            "arguments": vars(args),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", choices=("HQ", "TurboQuant"), help=argparse.SUPPRESS)
    parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--avg-bits", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--recipe", choices=("hq", "paper-original"), default="hq")
    parser.add_argument("--key-code-bits", type=int)
    parser.add_argument("--key-block-size", type=int)
    parser.add_argument("--value-code-bits", type=int)
    parser.add_argument("--value-block-size", type=int)
    parser.add_argument("--allow-canonical-k10v6-b4", action="store_true")
    parser.add_argument("--assignment-mode", choices=("exact", "lut"))
    parser.add_argument("--lut-bins", type=int)
    parser.add_argument("--lut-candidates", type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--dist", choices=("gaussian", "beta"), default="gaussian")
    parser.add_argument("--codebook-init", choices=("auto", "fibquant", "forgy"), default="auto")
    parser.add_argument(
        "--rotation-mode", choices=("auto", "dense", "hybrid", "hadamard_gemm"), default="auto"
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--page-size", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-splits", type=int, default=32)
    parser.add_argument("--measure-cudagraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = _parser().parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads must be divisible by kv-heads")
    if args.worker:
        result = _worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, allow_nan=False))
        return
    try:
        result = _controller(args)
    except Exception as exc:
        if args.output:
            failure_path = Path(f"{args.output}.failed.json")
            if not failure_path.exists():
                _write_json(
                    failure_path,
                    {
                        "schema": {
                            "name": "onecomp.hq_vs_upstream_tq_native_fair_speed_failure",
                            "version": 1,
                        },
                        "status": "controller_failure",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "git_commit": _git("rev-parse", "HEAD"),
                        "git_status_porcelain": _git("status", "--porcelain"),
                        "arguments": vars(args),
                    },
                )
        raise
    text = json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False)
    print(text)
    if args.output:
        _write_json(args.output, result)


if __name__ == "__main__":
    main()
