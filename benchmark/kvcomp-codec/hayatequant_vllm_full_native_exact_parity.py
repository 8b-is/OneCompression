#!/usr/bin/env python3
"""Adversarial GPU parity for the fused all-codeword b=2 HQ assignment.

This development gate compares the native tensor-core score/argmax directly
with ``quantize_exact_pre`` at realistic seq=1 and seq=4096 flattened block
counts, on wide/grid-boundary inputs, and on rows with fp16 top-score ties.
It is separate from model quality and cross-method speed claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from hayatequant_vllm_full_native_smoke import _recipe, _scoring_tables, _tables
from onecomp.kvcomp.codec.hayate.vq import quantize_exact_pre
from vllm_plugins.kvcomp.hayate_full_triton import _quantize_exact_b2_tensorcore

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare(
    name: str,
    z: torch.Tensor,
    codebook: torch.Tensor,
    csq: torch.Tensor,
) -> dict:
    reference = quantize_exact_pre(z, codebook, csq)
    native = _quantize_exact_b2_tensorcore(z, codebook, csq)
    mismatch = (reference != native).reshape(-1)
    count = int(mismatch.sum().item())
    first = mismatch.nonzero().flatten()[:8]
    result = {
        "name": name,
        "rows": int(z.numel() // 2),
        "mismatch_count": count,
        "exact": count == 0,
        "first_mismatches": [
            {
                "row": int(i),
                "query": [float(v) for v in z.reshape(-1, 2)[i].tolist()],
                "reference_code": int(reference.reshape(-1)[i]),
                "native_code": int(native.reshape(-1)[i]),
            }
            for i in first.cpu().tolist()
        ],
    }
    if count:
        raise AssertionError(f"{name}: {count} native/public exact-code mismatches")
    return result


def _boundary_rows(device: torch.device, head_dim: int) -> torch.Tensor:
    bins = 256
    limit = head_dim**0.5
    axis = torch.linspace(-limit, limit, bins + 1, device=device, dtype=torch.float32).half()
    x = axis.repeat_interleave(axis.numel())
    y = axis.repeat(axis.numel())
    exact = torch.stack((x, y), dim=1)
    lo = torch.nextafter(exact, torch.full_like(exact, -float("inf")))
    hi = torch.nextafter(exact, torch.full_like(exact, float("inf")))
    return torch.cat((exact, lo, hi), dim=0).contiguous()


def _tie_rows(
    z: torch.Tensor,
    codebook: torch.Tensor,
    csq: torch.Tensor,
    max_rows: int = 16384,
) -> tuple[torch.Tensor, int]:
    selected: list[torch.Tensor] = []
    observed = 0
    flat = z.reshape(-1, 2)
    for start in range(0, flat.shape[0], 32768):
        chunk = flat[start : start + 32768]
        scores = chunk @ codebook.t() - 0.5 * csq.unsqueeze(0)
        top = torch.topk(scores, k=2, dim=1).values
        tied = top[:, 0] == top[:, 1]
        observed += int(tied.sum().item())
        if sum(part.shape[0] for part in selected) < max_rows and tied.any():
            selected.append(chunk[tied][:max_rows].clone())
    if not selected:
        raise AssertionError("boundary fixture did not exercise an fp16 top-score tie")
    return torch.cat(selected, dim=0)[:max_rows].contiguous(), observed


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    recipe = _recipe(args)
    if recipe.assignment_mode != "exact" or (
        recipe.key_block_size != 2 or recipe.value_block_size != 2
    ):
        raise ValueError("this parity gate is for canonical exact b=2 HQ only")
    _, _, _, key_cb32, value_cb32 = _tables(recipe, args.head_dim, device)
    key_cb, key_csq, value_cb, value_csq = _scoring_tables(recipe, key_cb32, value_cb32)
    rows_per_vector = args.head_dim // 2
    seq1_rows = args.seq1_tokens * args.kv_heads * rows_per_vector
    seq4096_rows = args.seq4096_tokens * args.kv_heads * rows_per_vector
    boundaries = _boundary_rows(device, args.head_dim)
    explicit_outside = torch.tensor(
        [
            [20.0, 0.0],
            [-20.0, 0.0],
            [0.0, 20.0],
            [0.0, -20.0],
            [30.0, 30.0],
            [-30.0, -30.0],
            [12.0, -12.0],
            [-12.0, 12.0],
        ],
        device=device,
        dtype=torch.float16,
    )
    roles = {}
    for role_index, (role, cb, csq) in enumerate(
        (("key", key_cb, key_csq), ("value", value_cb, value_csq))
    ):
        gen = torch.Generator(device=device).manual_seed(args.seed + 1009 * role_index)
        fixtures = [
            _compare(
                "seq1_standard_normal",
                torch.randn(seq1_rows, 2, generator=gen, device=device, dtype=torch.float16),
                cb,
                csq,
            ),
            _compare(
                "seq4096_standard_normal",
                torch.randn(
                    seq4096_rows,
                    2,
                    generator=gen,
                    device=device,
                    dtype=torch.float16,
                ),
                cb,
                csq,
            ),
            _compare("uniform_grid_boundaries_and_adjacent_fp16", boundaries, cb, csq),
            _compare(
                "wide_and_explicit_outside",
                torch.cat(
                    (
                        torch.randn(
                            args.wide_rows,
                            2,
                            generator=gen,
                            device=device,
                            dtype=torch.float16,
                        )
                        * 8.0,
                        explicit_outside,
                    ),
                    dim=0,
                ),
                cb,
                csq,
            ),
        ]
        ties, observed = _tie_rows(boundaries, cb, csq)
        fixtures.append(_compare("fp16_top_score_ties", ties, cb, csq))
        roles[role] = {
            "codebook_size": int(cb.shape[0]),
            "top_score_tie_rows_observed_in_boundary_grid": observed,
            "fixtures": fixtures,
        }
    sources = [
        Path("benchmark/kvcomp-codec/hayatequant_vllm_full_native_exact_parity.py"),
        Path("benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py"),
        Path("vllm_plugins/kvcomp/hayate_full_triton.py"),
        Path("onecomp/kvcomp/codec/hayate/vq.py"),
    ]
    return {
        "schema": {"name": "onecomp.hq_full_native_exact_assignment_parity", "version": 1},
        "ok": True,
        "development_only": True,
        "formal_quality_benchmark": False,
        "cross_method_speed_claim_allowed": False,
        "algorithm": "all_codewords_tensorcore_score_fused_argmax",
        "approximate_lut": False,
        "persistent_acceleration_table_bytes": 0,
        "recipe": recipe.provenance(args.head_dim),
        "roles": roles,
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "source_sha256": {str(path): _sha(REPO / path) for path in sources},
            "arguments": vars(args),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg-bits", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--recipe", choices=("hq",), default="hq")
    parser.add_argument("--assignment-mode", choices=("exact",), default="exact")
    parser.add_argument("--dist", choices=("gaussian",), default="gaussian")
    parser.add_argument("--codebook-init", choices=("auto",), default="auto")
    parser.add_argument("--rotation-mode", choices=("auto",), default="auto")
    parser.add_argument("--allow-canonical-k10v6-b4", action="store_true")
    parser.add_argument("--key-code-bits", type=int)
    parser.add_argument("--key-block-size", type=int)
    parser.add_argument("--value-code-bits", type=int)
    parser.add_argument("--value-block-size", type=int)
    parser.add_argument("--lut-bins", type=int)
    parser.add_argument("--lut-candidates", type=int)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seq1-tokens", type=int, default=1)
    parser.add_argument("--seq4096-tokens", type=int, default=4096)
    parser.add_argument("--wide-rows", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
