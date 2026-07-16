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
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "KVCOMP_NATIVE_SPEED_RESULT="


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        triton_turboquant_decode_attention,
    )
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

    from hayatequant_vllm_full_native_smoke import (
        _luts,
        _recipe,
        _scoring_tables,
        _tables,
    )
    from vllm_plugins.kvcomp.hayate_full_triton import (
        triton_full_hayatequant_decode_attention,
        triton_full_hayatequant_store,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
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
        cache = torch.empty(
            pages,
            page_size,
            hk,
            layout["slot_size_aligned"],
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
            "slot_size_aligned": layout["slot_size_aligned"],
            "upstream_turboquant_modified": False,
        }
    elif args.worker_variant == "TurboQuant":
        if args.avg_bits == 3:
            dtype_name = "turboquant_3bit_nc"
        elif args.avg_bits == 4:
            dtype_name = "turboquant_4bit_nc"
        else:
            raise ValueError("upstream vLLM 0.20 has no native 2-bit TurboQuant preset")
        tq = TurboQuantConfig.from_cache_dtype(dtype_name, d)
        cache = torch.empty(
            pages,
            page_size,
            hk,
            tq.slot_size_aligned,
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
            "slot_size_aligned": tq.slot_size_aligned,
            "upstream_turboquant_modified": False,
        }
    else:
        raise ValueError(f"unexpected worker variant {args.worker_variant}")

    # First calls compile Triton and populate all persistent tables/cache.  They
    # are intentionally outside measured samples for both methods.
    store_all()
    decode()
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
            serving_step()
        graph_samples = _event_samples(torch, graph.replay, args.iters)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
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
        "method_contract": method_contract,
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
        completed = subprocess.run(
            _worker_command(args, variant, trial),
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
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
            raise RuntimeError(
                f"worker {variant} emitted no result; stdout={completed.stdout[-2000:]} "
                f"stderr={completed.stderr[-2000:]}"
            )
        row = json.loads(line[len(RESULT_PREFIX) :])
        row["worker_stderr_tail"] = completed.stderr[-1000:]
        rows.append(row)

    gpu_ids = {(row["gpu"]["name"], row["gpu"].get("uuid")) for row in rows}
    pids = {row["pid"] for row in rows}
    if len(gpu_ids) != 1:
        raise RuntimeError(f"workers did not use the same physical GPU: {gpu_ids}")
    if len(pids) != len(rows):
        raise RuntimeError("fresh-process contract violated: repeated worker PID")
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
    dirty = bool(_git("status", "--porcelain"))
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
    return {
        "schema": {"name": "onecomp.hq_vs_upstream_tq_native_fair_speed", "version": 1},
        "formal_eligible": not ineligible,
        "formal_ineligibility_reasons": ineligible,
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
            "timing": "CUDA events with one final synchronization per sample vector",
        },
        "aggregate": aggregate,
        "workers": rows,
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "vllm_version": version("vllm"),
            "source_sha256": {str(path): _sha(REPO / path) for path in source_paths},
            "upstream_vllm_sha256": {str(path): _sha(path) for path in upstream_paths},
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
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-splits", type=int, default=32)
    parser.add_argument("--measure-cudagraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser


def main():
    args = _parser().parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads must be divisible by kv-heads")
    if args.worker:
        result = _worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return
    result = _controller(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
