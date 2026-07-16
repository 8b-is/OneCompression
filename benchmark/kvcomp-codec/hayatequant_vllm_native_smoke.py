"""Smoke test for the native vLLM HayateQuant key block-VQ kernels.

This script intentionally avoids model loading.  It allocates the same
TurboQuant-layout KV cache that vLLM uses for ``turboquant_3bit_nc`` and
``turboquant_4bit_nc``, stores random K/V with the OneComp HayateQuant native
kernel, then runs one decode attention step.  It is small enough for a quick
Slurm GPU sanity check but catches the important failures: Triton compilation,
cache layout mismatches, and non-finite decode output.

Example
-------
    sbatch --gres=gpu:1 --wrap \
      "cd /path/to/onecomp-lab && .venv-vllm/bin/python \
       benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from onecomp.kvcomp.codec.hayate.codec import _block_codebook
from onecomp.kvcomp.codec.rotation import hybrid_rotation_matrix
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm_plugins.kvcomp.hayate_triton import (
    triton_hayatequant_decode_attention,
    triton_hayatequant_store,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _command_output(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv,
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nvidia_smi_snapshot() -> list[dict[str, str]]:
    fields = [
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "temperature.gpu",
        "clocks.current.sm",
        "clocks.max.sm",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
        "utilization.gpu",
    ]
    output = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    return [
        dict(zip(fields, (part.strip() for part in line.split(",")), strict=True))
        for line in output.splitlines()
    ]


def _provenance(args: argparse.Namespace) -> dict[str, object]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    source_paths = [
        Path("onecomp/kvcomp/cache/compressed_cache.py"),
        Path("vllm_plugins/kvcomp/hayate_triton.py"),
        Path("benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py"),
    ]
    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    return {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": _package_version("vllm"),
        "triton_version": _package_version("triton"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": {
            "logical_index": torch.cuda.current_device(),
            "name": props.name,
            "capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
            "multiprocessor_count": props.multi_processor_count,
        },
        "nvidia_smi": _nvidia_smi_snapshot(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _command_output(["git", "branch", "--show-current"]),
        "git_status_porcelain": _command_output(["git", "status", "--porcelain"]),
        "source_sha256": {str(path): _sha256(_REPO_ROOT / path) for path in source_paths},
        "arguments": arguments,
        "random_seed": 0,
        "timing_protocol": (
            "5 warmups; Python wall time around CUDA launches; "
            "torch.cuda.synchronize before and after each measured loop"
        ),
        "legacy_store_definition": (
            "value.float() materialization plus forced scratch/atomic/" "key-scale-writeback path"
        ),
        "optimized_store_definition": (
            "native value dtype plus direct key-scale store when n_groups == 1"
        ),
    }


def _dtype_for(bits: int) -> str:
    if bits == 3:
        return "turboquant_3bit_nc"
    if bits == 4:
        return "turboquant_4bit_nc"
    raise ValueError("native smoke supports bits in {3, 4}")


def run_case(
    bits: int,
    *,
    head_dim: int,
    kv_heads: int,
    q_heads: int,
    tokens: int,
    key_scale_mode: str,
    codebook_init: str,
    bench_iters: int,
    verify_optimized_store: bool,
    input_dtype: torch.dtype,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native vLLM HayateQuant smoke test")
    device = torch.device("cuda")
    torch.manual_seed(0)

    dtype_name = _dtype_for(bits)
    tq = TurboQuantConfig.from_cache_dtype(dtype_name, head_dim)
    page_size = 16
    num_blocks = math.ceil(tokens / page_size)
    kv_cache = torch.empty(
        num_blocks,
        page_size,
        kv_heads,
        tq.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int32)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=input_dtype)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=input_dtype)
    query = torch.randn(1, q_heads, head_dim, device=device, dtype=input_dtype)

    block_size = 2
    rotation = hybrid_rotation_matrix(head_dim, block_size, 0, str(device), torch.float32)
    codebook = _block_codebook(bits, block_size, "gaussian", head_dim, codebook_init).to(
        device=device, dtype=torch.float32
    )

    parity: dict[str, object] = {}
    if verify_optimized_store:
        # Reconstruct the exact pre-optimization launcher behaviour (eager
        # value.float() plus scratch/atomic/writeback key-scale path) and
        # compare it against the optimized path byte-for-byte.  Initializing
        # padding identically makes the whole cache comparison meaningful.
        legacy_cache = torch.full_like(kv_cache, 0xA5)
        optimized_cache = torch.full_like(kv_cache, 0xA5)
        env_name = "KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE"
        saved_env = os.environ.get(env_name)
        try:
            os.environ[env_name] = "never"
            triton_hayatequant_store(
                key,
                value.float(),
                legacy_cache,
                slot_mapping,
                rotation,
                codebook,
                block_size,
                bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_scale_mode=key_scale_mode,
            )
            os.environ[env_name] = "always"
            triton_hayatequant_store(
                key,
                value,
                optimized_cache,
                slot_mapping,
                rotation,
                codebook,
                block_size,
                bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_scale_mode=key_scale_mode,
            )
        finally:
            if saved_env is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = saved_env
        torch.cuda.synchronize()
        if not torch.equal(legacy_cache, optimized_cache):
            mismatches = int((legacy_cache != optimized_cache).sum().item())
            raise AssertionError(
                f"optimized store changed {mismatches} cache bytes "
                f"(bits={bits}, key_scale_mode={key_scale_mode})"
            )
        legacy_out = triton_hayatequant_decode_attention(
            query=query,
            kv_cache=legacy_cache,
            block_table=torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1),
            seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
            rotation=rotation,
            codebook=codebook,
            scale=head_dim**-0.5,
            key_bits=bits,
            key_packed_size=tq.key_packed_size,
            value_quant_bits=tq.effective_value_quant_bits,
            hayate_block_size=block_size,
            norm_correction=tq.norm_correction,
            key_scale_mode=key_scale_mode,
            max_num_kv_splits=1,
        )
        optimized_out = triton_hayatequant_decode_attention(
            query=query,
            kv_cache=optimized_cache,
            block_table=torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1),
            seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
            rotation=rotation,
            codebook=codebook,
            scale=head_dim**-0.5,
            key_bits=bits,
            key_packed_size=tq.key_packed_size,
            value_quant_bits=tq.effective_value_quant_bits,
            hayate_block_size=block_size,
            norm_correction=tq.norm_correction,
            key_scale_mode=key_scale_mode,
            max_num_kv_splits=1,
        )
        if not torch.equal(legacy_out, optimized_out):
            raise AssertionError("optimized store changed the decoded attention output")
        parity = {
            "optimized_store_cache_bit_identical": True,
            "optimized_store_decode_bit_identical": True,
        }

    triton_hayatequant_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        rotation,
        codebook,
        block_size,
        bits,
        key_packed_size=tq.key_packed_size,
        value_quant_bits=tq.effective_value_quant_bits,
        key_scale_mode=key_scale_mode,
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([tokens], device=device, dtype=torch.int32)
    out = triton_hayatequant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        rotation=rotation,
        codebook=codebook,
        scale=head_dim**-0.5,
        key_bits=bits,
        key_packed_size=tq.key_packed_size,
        value_quant_bits=tq.effective_value_quant_bits,
        hayate_block_size=block_size,
        norm_correction=tq.norm_correction,
        key_scale_mode=key_scale_mode,
        max_num_kv_splits=1,
    )
    torch.cuda.synchronize()
    if out.shape != (1, q_heads, head_dim):
        raise AssertionError(f"unexpected output shape: {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise AssertionError("decode output contains non-finite values")
    row = {
        "implementation_label": ("native_hybrid_hayate_block_vq_keys_turboquant_scalar_values"),
        "native_hybrid_is_full_hayatequant": False,
        "canonical_hayatequant_frontier_recipe": False,
        "bits": bits,
        "key_bits": bits,
        "value_bits": int(tq.effective_value_quant_bits),
        "key_scale_mode": key_scale_mode,
        "codebook_init": codebook_init,
        "dtype": dtype_name,
        "input_dtype": str(input_dtype).removeprefix("torch."),
        "slot_size_aligned": tq.slot_size_aligned,
        "output_abs_mean": float(out.float().abs().mean().item()),
        "output_abs_max": float(out.float().abs().max().item()),
        **parity,
    }
    if bench_iters > 0:
        for _ in range(5):
            triton_hayatequant_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                rotation,
                codebook,
                block_size,
                bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_scale_mode=key_scale_mode,
            )
            triton_hayatequant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                rotation=rotation,
                codebook=codebook,
                scale=head_dim**-0.5,
                key_bits=bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                hayate_block_size=block_size,
                norm_correction=tq.norm_correction,
                key_scale_mode=key_scale_mode,
                max_num_kv_splits=1,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            triton_hayatequant_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                rotation,
                codebook,
                block_size,
                bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                key_scale_mode=key_scale_mode,
            )
        torch.cuda.synchronize()
        store_ms = (time.perf_counter() - t0) * 1.0e3 / bench_iters

        store_comparison = {}
        if verify_optimized_store:
            env_name = "KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE"
            saved_env = os.environ.get(env_name)
            try:
                os.environ[env_name] = "never"
                for _ in range(5):
                    triton_hayatequant_store(
                        key,
                        value.float(),
                        legacy_cache,
                        slot_mapping,
                        rotation,
                        codebook,
                        block_size,
                        bits,
                        key_packed_size=tq.key_packed_size,
                        value_quant_bits=tq.effective_value_quant_bits,
                        key_scale_mode=key_scale_mode,
                    )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(bench_iters):
                    triton_hayatequant_store(
                        key,
                        value.float(),
                        legacy_cache,
                        slot_mapping,
                        rotation,
                        codebook,
                        block_size,
                        bits,
                        key_packed_size=tq.key_packed_size,
                        value_quant_bits=tq.effective_value_quant_bits,
                        key_scale_mode=key_scale_mode,
                    )
                torch.cuda.synchronize()
                legacy_store_ms = (time.perf_counter() - t0) * 1.0e3 / bench_iters

                os.environ[env_name] = "always"
                for _ in range(5):
                    triton_hayatequant_store(
                        key,
                        value,
                        optimized_cache,
                        slot_mapping,
                        rotation,
                        codebook,
                        block_size,
                        bits,
                        key_packed_size=tq.key_packed_size,
                        value_quant_bits=tq.effective_value_quant_bits,
                        key_scale_mode=key_scale_mode,
                    )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(bench_iters):
                    triton_hayatequant_store(
                        key,
                        value,
                        optimized_cache,
                        slot_mapping,
                        rotation,
                        codebook,
                        block_size,
                        bits,
                        key_packed_size=tq.key_packed_size,
                        value_quant_bits=tq.effective_value_quant_bits,
                        key_scale_mode=key_scale_mode,
                    )
                torch.cuda.synchronize()
                optimized_store_ms = (time.perf_counter() - t0) * 1.0e3 / bench_iters
            finally:
                if saved_env is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = saved_env
            store_comparison = {
                "legacy_store_ms": round(legacy_store_ms, 4),
                "optimized_store_ms": round(optimized_store_ms, 4),
                "optimized_store_speedup": round(
                    legacy_store_ms / max(optimized_store_ms, 1.0e-9), 3
                ),
            }

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            triton_hayatequant_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                rotation=rotation,
                codebook=codebook,
                scale=head_dim**-0.5,
                key_bits=bits,
                key_packed_size=tq.key_packed_size,
                value_quant_bits=tq.effective_value_quant_bits,
                hayate_block_size=block_size,
                norm_correction=tq.norm_correction,
                key_scale_mode=key_scale_mode,
                max_num_kv_splits=1,
            )
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1.0e3 / bench_iters
        row.update(
            {
                "bench_iters": bench_iters,
                "store_ms": round(store_ms, 4),
                "decode_ms": round(decode_ms, 4),
                "store_Mtok_s": round(tokens / max(store_ms, 1.0e-9) / 1.0e3, 3),
                **store_comparison,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--key-scale-mode", choices=["nc", "ub", "ip"], default="nc")
    parser.add_argument("--codebook-init", choices=["auto", "fibquant"], default="auto")
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument(
        "--input-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--verify-optimized-store",
        action="store_true",
        help=(
            "Compare the optimized native store against the previous eager-cast/"
            "scratch path byte-for-byte, including decoded attention output."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the smoke and timing result.",
    )
    args = parser.parse_args()
    if args.head_dim < 1 or args.kv_heads < 1 or args.q_heads < 1 or args.tokens < 1:
        parser.error("head-dim, kv-heads, q-heads, and tokens must be positive")
    if args.q_heads % args.kv_heads:
        parser.error("q-heads must be divisible by kv-heads for ordinary GQA")
    if args.bench_iters < 0:
        parser.error("bench-iters must be non-negative")
    if len(set(args.bits)) != len(args.bits):
        parser.error("bits must not contain duplicate cases")
    rows = [
        run_case(
            b,
            head_dim=args.head_dim,
            kv_heads=args.kv_heads,
            q_heads=args.q_heads,
            tokens=args.tokens,
            key_scale_mode=args.key_scale_mode,
            codebook_init=args.codebook_init,
            bench_iters=args.bench_iters,
            verify_optimized_store=args.verify_optimized_store,
            input_dtype=getattr(torch, args.input_dtype),
        )
        for b in args.bits
    ]
    payload = {
        "schema": {"name": "onecomp.hayatequant_vllm_native_hybrid_smoke", "version": 1},
        "ok": True,
        "implementation_contract": {
            "implementation_label": (
                "native_hybrid_hayate_block_vq_keys_turboquant_scalar_values"
            ),
            "key_path": "HayateQuant block-VQ native store/decode attention",
            "value_path": "TurboQuant scalar native cache layout",
            "symmetric_kv_bit_allocation": True,
            "canonical_frontier_k_r_plus_1_v_r_minus_1": False,
            "native_hybrid_is_full_hayatequant": False,
            "eligible_for_full_hayatequant_serving_claim": False,
            "scope": "kernel correctness/performance smoke; not a model-quality result",
            "counterbalanced_across_bit_cases": False,
            "cross_case_timing_claims_allowed": False,
        },
        "provenance": _provenance(args),
        "cases": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
