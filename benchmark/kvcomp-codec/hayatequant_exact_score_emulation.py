#!/usr/bin/env python3
"""Development test for scalar emulation of HQ's canonical fp16 GEMM score.

The native candidate-list encoder needs a per-row codeword list, so it cannot
directly use a dense tensor-core GEMM whose right-hand side is shared by every
row.  This diagnostic checks whether two explicit fp32 products, one fp32
sum, and the same fp16 score boundaries reproduce the audited dense-GEMM
assignment exactly for the fixed b=2 HQ codebooks.  It does not promote or
select an implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch

from hayatequant_vllm_full_native_smoke import _scoring_tables, _tables
from vllm.triton_utils import tl, triton
from vllm_plugins.kvcomp.hayate_full_config import fixed_hq_native_recipe
from vllm_plugins.kvcomp.hayate_full_triton import _quantize_exact_b2_tensorcore

REPO = Path(__file__).resolve().parents[2]


@triton.jit
def _manual_fp32_b2_score(
    Z_ptr,
    CB_ptr,
    CSQ_ptr,
    Codes_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Emulate the two-term tensor-core score without a shared RHS matrix."""

    row = tl.program_id(0)
    code = tl.arange(0, BLOCK_K)
    mask = code < K
    z0 = tl.load(Z_ptr + row * 2).to(tl.float16)
    z1 = tl.load(Z_ptr + row * 2 + 1).to(tl.float16)
    c0 = tl.load(CB_ptr + code * 2, mask=mask, other=0.0).to(tl.float16)
    c1 = tl.load(CB_ptr + code * 2 + 1, mask=mask, other=0.0).to(tl.float16)
    # A product of two binary16 values is exactly representable in binary32.
    # The explicit casts prevent Triton from selecting binary16 arithmetic.
    dot = (z0.to(tl.float32) * c0.to(tl.float32) + z1.to(tl.float32) * c1.to(tl.float32)).to(
        tl.float16
    )
    csq = tl.load(CSQ_ptr + code, mask=mask, other=0.0).to(tl.float16)
    correction = (0.5 * csq.to(tl.float32)).to(tl.float16)
    score = (dot.to(tl.float32) - correction.to(tl.float32)).to(tl.float16)
    scoref = tl.where(mask, score, -float("inf")).to(tl.float32)
    best = tl.max(scoref, axis=0)
    winner = tl.min(tl.where(scoref == best, code, K), axis=0)
    tl.store(Codes_ptr + row, winner.to(tl.int64))


def _manual(z: torch.Tensor, cb: torch.Tensor, csq: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, 2).contiguous()
    k = int(cb.shape[0])
    codes = torch.empty(flat.shape[0], device=flat.device, dtype=torch.int64)
    block_k = triton.next_power_of_2(k)
    _manual_fp32_b2_score[(flat.shape[0],)](
        flat,
        cb,
        csq,
        codes,
        N=flat.shape[0],
        K=k,
        BLOCK_K=block_k,
        num_warps=4 if block_k >= 128 else 2,
        num_stages=1,
    )
    return codes.reshape(z.shape[:-1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fixture(rows: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    random = torch.randn(rows, 2, generator=generator, device=device, dtype=torch.float16)
    known = torch.tensor(
        [
            [-0.07568359375, -1.0361328125],
            [1.40234375, 0.358154296875],
            [0.873046875, 0.007843017578125],
            [1.236328125, 0.0033016204833984375],
            [0.0, 0.0],
            [-0.0, 0.0],
            [11.3125, 0.0],
            [-11.3125, 0.0],
        ],
        device=device,
        dtype=torch.float16,
    )
    return torch.cat((known, random), dim=0).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=918273)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("rows must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    z = _fixture(args.rows, args.seed, device)
    results = []
    for rate in (2, 3, 4):
        recipe = fixed_hq_native_recipe(rate)
        if recipe.key_block_size != 2 or recipe.value_block_size != 2:
            raise RuntimeError("diagnostic is restricted to the fixed b=2 recipes")
        _, _, _, key_cb, value_cb = _tables(recipe, 128, device)
        key_score, key_csq, value_score, value_csq = _scoring_tables(recipe, key_cb, value_cb)
        for role, cb, csq in (
            ("key", key_score, key_csq),
            ("value", value_score, value_csq),
        ):
            reference = _quantize_exact_b2_tensorcore(z, cb, csq)
            observed = _manual(z, cb, csq)
            mismatch = (reference != observed).nonzero(as_tuple=False).flatten()
            examples = []
            for index in mismatch[:16].cpu().tolist():
                examples.append(
                    {
                        "index": int(index),
                        "query": [float(v) for v in z[index].cpu().tolist()],
                        "reference": int(reference[index]),
                        "observed": int(observed[index]),
                    }
                )
            results.append(
                {
                    "rate": rate,
                    "role": role,
                    "codebook_size": int(cb.shape[0]),
                    "codebook_sha256": _tensor_sha(cb),
                    "csq_sha256": _tensor_sha(csq),
                    "rows": int(z.shape[0]),
                    "mismatch_count": int(mismatch.numel()),
                    "examples": examples,
                }
            )

    document = {
        "schema": {"name": "onecomp.hq_exact_score_emulation_dev", "version": 1},
        "development_only": True,
        "promotion_allowed": False,
        "all_zero_mismatch": all(row["mismatch_count"] == 0 for row in results),
        "fixture": {
            "random_rows": args.rows,
            "total_rows": int(z.shape[0]),
            "seed": args.seed,
            "dtype": str(z.dtype),
            "sha256": _tensor_sha(z),
            "known_rejected_grid_queries_included": True,
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
            "accepted_reference_source": str(
                (REPO / "vllm_plugins/kvcomp/hayate_full_triton.py").resolve()
            ),
            "accepted_reference_source_sha256": _sha(
                REPO / "vllm_plugins/kvcomp/hayate_full_triton.py"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
