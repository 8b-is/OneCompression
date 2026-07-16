#!/usr/bin/env python3
"""Development gate for an exact row-specific tensor-core HQ lookup table.

This script keeps candidate coverage, tensor-core score parity, and assignment
latency separate.  It is not a formal benchmark and never promotes a recipe.
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
from typing import Any

import torch

from hayatequant_vllm_full_native_smoke import _scoring_tables, _tables
from vllm.triton_utils import tl, triton
from vllm_plugins.kvcomp.hayate_full_config import fixed_hq_native_recipe
from vllm_plugins.kvcomp.hayate_full_triton import _quantize_exact_b2_tensorcore

REPO = Path(__file__).resolve().parents[2]


@triton.jit
def _exact_lut_tensorcore_b2(
    Z_ptr,
    LUT_ptr,
    CB_ptr,
    CSQ_ptr,
    Codes_ptr,
    N: tl.constexpr,
    BINS: tl.constexpr,
    T: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """Lookup a row-specific list and rank it with the canonical MMA path."""

    row = tl.program_id(0)
    inner = tl.arange(0, 16)
    padded_row = tl.arange(0, 16)
    slot = tl.arange(0, 16)
    z0f = tl.load(Z_ptr + row * 2).to(tl.float32)
    z1f = tl.load(Z_ptr + row * 2 + 1).to(tl.float32)
    step = BINS / (2.0 * LIMIT)
    i0 = ((tl.minimum(tl.maximum(z0f, -LIMIT), LIMIT) + LIMIT) * step).to(tl.int32)
    i1 = ((tl.minimum(tl.maximum(z1f, -LIMIT), LIMIT) + LIMIT) * step).to(tl.int32)
    i0 = tl.minimum(tl.maximum(i0, 0), BINS - 1)
    i1 = tl.minimum(tl.maximum(i1, 0), BINS - 1)
    cell = i0 * BINS + i1
    candidate = tl.load(LUT_ptr + cell * T + slot).to(tl.int32)
    z0 = z0f.to(tl.float16)
    z1 = z1f.to(tl.float16)
    lhs = tl.where(
        padded_row[:, None] == 0,
        tl.where(
            inner[None, :] == 0,
            z0,
            tl.where(inner[None, :] == 1, z1, 0.0),
        ),
        0.0,
    ).to(tl.float16)
    rhs = tl.load(
        CB_ptr + candidate[None, :] * 2 + inner[:, None],
        mask=inner[:, None] < 2,
        other=0.0,
    ).to(tl.float16)
    dot_matrix = tl.dot(lhs, rhs, out_dtype=tl.float32).to(tl.float16)
    dot = tl.sum(
        tl.where(padded_row[:, None] == 0, dot_matrix, 0.0).to(tl.float32),
        axis=0,
    ).to(tl.float16)
    csq = tl.load(CSQ_ptr + candidate).to(tl.float16)
    correction = (0.5 * csq.to(tl.float32)).to(tl.float16)
    score = (dot.to(tl.float32) - correction.to(tl.float32)).to(tl.float16)
    scoref = score.to(tl.float32)
    best = tl.max(scoref, axis=0)
    winner = tl.min(tl.where(scoref == best, candidate, 2**31 - 1), axis=0)
    tl.store(Codes_ptr + row, winner.to(tl.int64))


def _run_lut(
    z: torch.Tensor, lut: torch.Tensor, cb: torch.Tensor, csq: torch.Tensor, limit: float
) -> torch.Tensor:
    flat = z.reshape(-1, 2).float().contiguous()
    if lut.shape[1] != 16 or lut.dtype != torch.int16:
        raise ValueError("development kernel requires a 16-candidate int16 table")
    bins = int(round(math.sqrt(lut.shape[0])))
    if bins * bins != lut.shape[0]:
        raise ValueError("table is not a square b=2 grid")
    codes = torch.empty(flat.shape[0], device=flat.device, dtype=torch.int64)
    _exact_lut_tensorcore_b2[(flat.shape[0],)](
        flat,
        lut,
        cb,
        csq,
        codes,
        N=flat.shape[0],
        BINS=bins,
        T=16,
        LIMIT=float(limit),
        num_warps=4,
        num_stages=1,
    )
    return codes.reshape(z.shape[:-1])


def _canonical_lut(
    cb: torch.Tensor, csq: torch.Tensor, bins: int, top_t: int, limit: float
) -> torch.Tensor:
    """Build a deterministic centre-ranked table in canonical score dtype."""

    axis = (torch.arange(bins, device=cb.device, dtype=torch.float32) + 0.5) / bins
    axis = axis * (2.0 * limit) - limit
    g0, g1 = torch.meshgrid(axis, axis, indexing="ij")
    centers = torch.stack((g0.reshape(-1), g1.reshape(-1)), dim=1).half()
    scores = centers @ cb.t() - 0.5 * csq.unsqueeze(0)
    ranked = scores.argsort(dim=1, descending=True, stable=True)[:, :top_t]
    # Force the independently audited all-codeword winner into every row even
    # if a future backend changes non-winning tie order at the T-th boundary.
    winner = _quantize_exact_b2_tensorcore(centers, cb, csq)
    present = (ranked == winner.unsqueeze(1)).any(dim=1)
    ranked[~present, -1] = winner[~present]
    return ranked.sort(dim=1).values.to(torch.int16).contiguous()


def _cell_indices(z: torch.Tensor, bins: int, limit: float) -> torch.Tensor:
    step = bins / (2.0 * limit)
    indices = ((z.float().clamp(-limit, limit) + limit) * step).long().clamp_(0, bins - 1)
    return indices[:, 0] * bins + indices[:, 1]


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


def _examples(z, reference, observed, mismatch) -> list[dict[str, Any]]:
    rows = []
    for index in mismatch[:16].cpu().tolist():
        rows.append(
            {
                "index": int(index),
                "query_float32": [float(value) for value in z[index].cpu().tolist()],
                "query_float16": [float(value) for value in z[index].half().cpu().tolist()],
                "reference": int(reference[index]),
                "observed": int(observed[index]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-rows", type=int, default=1_000_000)
    parser.add_argument("--timing-rows", type=int, default=2_097_152)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--limit", type=float, default=16.0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidates != 16 or args.bins != 256 or args.limit < math.sqrt(128.0):
        raise ValueError("frozen development point is bins=256, T=16, limit>=sqrt(128)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    gaussian = torch.randn(
        args.random_rows, 2, device=device, dtype=torch.float32, generator=generator
    )
    wide = torch.rand(
        args.random_rows, 2, device=device, dtype=torch.float32, generator=generator
    ) * (2.0 * math.sqrt(128.0)) - math.sqrt(128.0)
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
    fixtures = {"gaussian": torch.cat((known, gaussian)), "wide": torch.cat((known, wide))}

    tables: dict[str, torch.Tensor] = {}
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
            lut = tables.get(identity)
            if lut is None:
                lut = _canonical_lut(cb, csq, args.bins, args.candidates, args.limit)
                tables[identity] = lut
            fixture_rows = {}
            for fixture_name, z in fixtures.items():
                reference = _quantize_exact_b2_tensorcore(z.half(), cb, csq)
                cells = _cell_indices(z, args.bins, args.limit)
                candidates = lut[cells].long()
                absent = (
                    (~(candidates == reference.unsqueeze(1)).any(dim=1))
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                observed = _run_lut(z, lut, cb, csq, args.limit)
                mismatch = (reference != observed).nonzero(as_tuple=False).flatten()
                fixture_rows[fixture_name] = {
                    "rows": int(z.shape[0]),
                    "winner_absent": int(absent.numel()),
                    "mismatch_count": int(mismatch.numel()),
                    "mismatch_examples": _examples(z, reference, observed, mismatch),
                }

            timing_z = torch.randn(
                args.timing_rows,
                2,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            timing_half = timing_z.half()
            exact_kernel_out = torch.empty(0, device=device)
            exact_with_cast_out = torch.empty(0, device=device)
            lut_out = torch.empty(0, device=device)

            def exact_kernel_call():
                nonlocal exact_kernel_out
                exact_kernel_out = _quantize_exact_b2_tensorcore(timing_half, cb, csq)

            def exact_with_cast_call():
                nonlocal exact_with_cast_out
                exact_with_cast_out = _quantize_exact_b2_tensorcore(timing_z.half(), cb, csq)

            def lut_call():
                nonlocal lut_out
                lut_out = _run_lut(timing_z, lut, cb, csq, args.limit)

            exact_kernel_timing = _time_ms(exact_kernel_call, args.warmups, args.iters)
            exact_with_cast_timing = _time_ms(exact_with_cast_call, args.warmups, args.iters)
            lut_timing = _time_ms(lut_call, args.warmups, args.iters)
            timing_kernel_matches = torch.equal(exact_kernel_out, lut_out)
            timing_with_cast_matches = torch.equal(exact_with_cast_out, lut_out)
            results.append(
                {
                    "rate": rate,
                    "role": role,
                    "codebook_size": int(cb.shape[0]),
                    "codebook_sha256": identity,
                    "csq_sha256": _tensor_sha(csq),
                    "lut_sha256": _tensor_sha(lut),
                    "lut_bytes": int(lut.numel() * lut.element_size()),
                    "fixtures": fixture_rows,
                    "timing_rows": args.timing_rows,
                    "exact_kernel_only_timing": exact_kernel_timing,
                    "exact_with_fp32_to_fp16_cast_timing": exact_with_cast_timing,
                    "lut_tensorcore_timing": lut_timing,
                    "timing_outputs_match": {
                        "exact_kernel_only": timing_kernel_matches,
                        "exact_with_fp32_to_fp16_cast": timing_with_cast_matches,
                    },
                    "exact_kernel_only_over_lut_speedup": (
                        exact_kernel_timing["median_ms"] / lut_timing["median_ms"]
                    ),
                    "exact_with_cast_over_lut_speedup": (
                        exact_with_cast_timing["median_ms"] / lut_timing["median_ms"]
                    ),
                }
            )

    empirical_pass = all(
        fixture["winner_absent"] == 0 and fixture["mismatch_count"] == 0
        for row in results
        for fixture in row["fixtures"].values()
    ) and all(all(row["timing_outputs_match"].values()) for row in results)
    document = {
        "schema": {"name": "onecomp.hq_exact_lut_tensorcore_dev", "version": 1},
        "development_only": True,
        "promotion_allowed": False,
        "candidate": {
            "bins": args.bins,
            "candidates": args.candidates,
            "limit": args.limit,
            "table_build": "canonical_fp16_center_score_top16_with_forced_exact_center_winner",
            "score": "row_specific_padded_16x16x16_tensorcore_mma",
        },
        "all_empirical_coverage_and_parity_passed": empirical_pass,
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
    if not empirical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
