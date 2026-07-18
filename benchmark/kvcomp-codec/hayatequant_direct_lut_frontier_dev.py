#!/usr/bin/env python3
"""Map the accuracy/latency frontier of direct centre-winner HQ tables.

This is a frozen development diagnostic, not an HQ implementation or a
promotion gate.  It evaluates every preregistered table resolution, including
unfavourable points, and deliberately keeps approximation error separate from
the canonical all-codeword assignment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import torch

from hayatequant_vllm_full_native_smoke import _scoring_tables, _tables
from vllm.triton_utils import tl, triton
from vllm_plugins.kvcomp.hayate_full_config import fixed_hq_native_recipe
from vllm_plugins.kvcomp.hayate_full_triton import _quantize_exact_b2_tensorcore

REPO = Path(__file__).resolve().parents[2]
FROZEN_BINS = (256, 512, 1024, 2048)


@triton.jit
def _direct_lut_b2(
    Z_ptr,
    LUT_ptr,
    Codes_ptr,
    N: tl.constexpr,
    BINS: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Map each b=2 row to its spatial cell's centre winner."""

    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = row < N
    z0 = tl.load(Z_ptr + row * 2, mask=mask, other=0.0).to(tl.float32)
    z1 = tl.load(Z_ptr + row * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    step = BINS / (2.0 * LIMIT)
    i0 = ((tl.minimum(tl.maximum(z0, -LIMIT), LIMIT) + LIMIT) * step).to(tl.int32)
    i1 = ((tl.minimum(tl.maximum(z1, -LIMIT), LIMIT) + LIMIT) * step).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), BINS - 1)
    i1 = tl.minimum(tl.maximum(i1, 0), BINS - 1)
    code = tl.load(LUT_ptr + i0 * BINS + i1, mask=mask, other=0).to(tl.int64)
    tl.store(Codes_ptr + row, code, mask=mask)


def _run_direct(z: torch.Tensor, lut: torch.Tensor, limit: float) -> torch.Tensor:
    flat = z.reshape(-1, 2).float().contiguous()
    bins = int(round(math.sqrt(lut.numel())))
    if lut.ndim != 1 or bins * bins != lut.numel() or lut.dtype != torch.int16:
        raise ValueError("direct table must be a flat square int16 tensor")
    codes = torch.empty(flat.shape[0], device=flat.device, dtype=torch.int64)
    block_m = 256
    _direct_lut_b2[(triton.cdiv(flat.shape[0], block_m),)](
        flat,
        lut,
        codes,
        N=flat.shape[0],
        BINS=bins,
        LIMIT=float(limit),
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )
    return codes.reshape(z.shape[:-1])


def _build_direct_lut(
    cb: torch.Tensor,
    csq: torch.Tensor,
    bins: int,
    limit: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, float]:
    """Assign cell centres with the accepted all-codeword tensor-core path."""

    total = bins * bins
    table = torch.empty(total, device=cb.device, dtype=torch.int16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        flat = torch.arange(start, stop, device=cb.device, dtype=torch.int64)
        i0 = torch.div(flat, bins, rounding_mode="floor")
        i1 = flat - i0 * bins
        z0 = (i0.float() + 0.5) / bins * (2.0 * limit) - limit
        z1 = (i1.float() + 0.5) / bins * (2.0 * limit) - limit
        centres = torch.stack((z0, z1), dim=1).half()
        table[start:stop] = _quantize_exact_b2_tensorcore(centres, cb, csq).to(torch.int16)
    torch.cuda.synchronize()
    return table.contiguous(), time.perf_counter() - started


def _time_ms(fn, warmups: int, iters: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
        "n": len(ordered),
    }


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mismatch_summary(
    z: torch.Tensor, reference: torch.Tensor, observed: torch.Tensor
) -> dict[str, Any]:
    mismatch = (reference != observed).nonzero(as_tuple=False).flatten()
    examples = []
    for index in mismatch[:16].cpu().tolist():
        examples.append(
            {
                "index": int(index),
                "query_float32": [float(value) for value in z[index].cpu().tolist()],
                "query_float16": [float(value) for value in z[index].half().cpu().tolist()],
                "reference": int(reference[index]),
                "observed": int(observed[index]),
            }
        )
    return {
        "rows": int(reference.numel()),
        "mismatch_count": int(mismatch.numel()),
        "mismatch_fraction": float(mismatch.numel() / reference.numel()),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-rows", type=int, default=1_000_000)
    parser.add_argument("--timing-rows", type=int, default=2_097_152)
    parser.add_argument("--bins", type=int, nargs="+", default=list(FROZEN_BINS))
    parser.add_argument("--limit", type=float, default=16.0)
    parser.add_argument("--table-chunk-rows", type=int, default=262_144)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if tuple(args.bins) != FROZEN_BINS:
        raise ValueError(f"frozen development resolutions are {FROZEN_BINS}")
    if args.limit < math.sqrt(128.0):
        raise ValueError("limit must cover the normalized D=128 coordinate bound")
    if args.random_rows < 1 or args.timing_rows < 1 or args.table_chunk_rows < 1:
        raise ValueError("row counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    known = torch.tensor(
        [
            [-0.07568359375, -1.0361328125],
            [1.40234375, 0.358154296875],
            [0.873046875, 0.007843017578125],
            [1.236328125, 0.0033016204833984375],
            [-0.093994140625, -1.1396484375],
            [1.46875, -0.293701171875],
            [-0.258056640625, -1.4287109375],
            [-0.4072265625, 3.263671875],
            [math.sqrt(128.0), 0.0],
            [-math.sqrt(128.0), 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    gaussian = torch.randn(
        args.random_rows, 2, device=device, dtype=torch.float32, generator=generator
    )
    wide = torch.rand(
        args.random_rows, 2, device=device, dtype=torch.float32, generator=generator
    ) * (2.0 * math.sqrt(128.0)) - math.sqrt(128.0)
    fixtures = {
        "gaussian": torch.cat((known, gaussian)).contiguous(),
        "wide": torch.cat((known, wide)).contiguous(),
    }

    table_cache: dict[tuple[str, int], tuple[torch.Tensor, float]] = {}
    results = []
    for rate in (2, 3, 4):
        recipe = fixed_hq_native_recipe(rate)
        _, _, _, key_cb32, value_cb32 = _tables(recipe, 128, device)
        key_cb, key_csq, value_cb, value_csq = _scoring_tables(recipe, key_cb32, value_cb32)
        for role, cb, csq in (
            ("key", key_cb, key_csq),
            ("value", value_cb, value_csq),
        ):
            identity = _tensor_sha(cb)
            references = {
                name: _quantize_exact_b2_tensorcore(z.half(), cb, csq)
                for name, z in fixtures.items()
            }
            timing_z = torch.randn(
                args.timing_rows,
                2,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            exact_out = torch.empty(0, device=device)

            def exact_call():
                nonlocal exact_out
                exact_out = _quantize_exact_b2_tensorcore(timing_z.half(), cb, csq)

            exact_timing = _time_ms(exact_call, args.warmups, args.iters)
            frontier = []
            for bins in args.bins:
                cache_key = (identity, bins)
                cached = table_cache.get(cache_key)
                table_reused = cached is not None
                if cached is None:
                    cached = _build_direct_lut(cb, csq, bins, args.limit, args.table_chunk_rows)
                    table_cache[cache_key] = cached
                lut, build_seconds = cached
                fixture_results = {}
                for name, z in fixtures.items():
                    observed = _run_direct(z, lut, args.limit)
                    fixture_results[name] = _mismatch_summary(z, references[name], observed)

                direct_out = torch.empty(0, device=device)

                def direct_call():
                    nonlocal direct_out
                    direct_out = _run_direct(timing_z, lut, args.limit)

                direct_timing = _time_ms(direct_call, args.warmups, args.iters)
                timing_mismatch = int((direct_out != exact_out).count_nonzero().item())
                frontier.append(
                    {
                        "bins_per_dimension": bins,
                        "lut_bytes": int(lut.numel() * lut.element_size()),
                        "lut_sha256": _tensor_sha(lut),
                        "build_seconds_first_occurrence": build_seconds,
                        "table_reused_from_identical_codebook": table_reused,
                        "fixtures": fixture_results,
                        "timing_rows": args.timing_rows,
                        "timing_mismatch_count": timing_mismatch,
                        "exact_with_cast_timing": exact_timing,
                        "direct_timing": direct_timing,
                        "exact_over_direct_speedup": (
                            exact_timing["median_ms"] / direct_timing["median_ms"]
                        ),
                    }
                )
            results.append(
                {
                    "rate": rate,
                    "role": role,
                    "codebook_size": int(cb.shape[0]),
                    "codebook_sha256": identity,
                    "csq_sha256": _tensor_sha(csq),
                    "frontier": frontier,
                }
            )

    document = {
        "schema": {"name": "onecomp.hq_direct_lut_frontier_dev", "version": 1},
        "development_only": True,
        "promotion_allowed": False,
        "selection_performed": False,
        "candidate": {
            "resolutions": list(FROZEN_BINS),
            "limit": args.limit,
            "table_rule": "canonical_fp16_all_codeword_winner_at_each_cell_centre",
            "lookup_rule": "clip_then_floor_and_return_centre_winner_without_rescoring",
            "known_to_be_approximate": True,
        },
        "results": results,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
            "torch_version": torch.__version__,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha(Path(__file__).resolve()),
            "reference_source_sha256": _sha(REPO / "vllm_plugins/kvcomp/hayate_full_triton.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
