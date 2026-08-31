"""Regenerate HayateQuant's shipped data-free block-VQ codebooks.

The canonical method fixes the vector block size to two and supports average
rates 2, 3, and 4 bits per element. Those rates require five Gaussian
codebooks: K=4, 16, 64, 256, and 1024. This utility fits exactly that audited set
and writes plain fp32 tensors named ``gauss_b2_k{K}.pt``.

Still 100% data-free: the fit sees only synthetic ``N(0,1)`` samples (the
same law as the online fit), never any real KV data.  Committing the
artifact also makes the codebook bit-exactly reproducible across machines
and BLAS builds (GPU K-means kernels are not bitwise deterministic, so two
online fits can differ in the last ulp; a fixed artifact cannot).

Run (GPU strongly recommended for large K):
    python -m onecomp.kvcomp.codec.hayate.codebooks.fit_codebooks

To regenerate selected artifacts into a separate directory::

    python -m onecomp.kvcomp.codec.hayate.codebooks.fit_codebooks \
      --output-dir /path/to/output --target 2:4 --target 2:16

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import os

import torch

from ..vq import _gaussian_block_samples, _kmeans_best_of

# Every (block_size, K) required by the canonical rates 2, 3, and 4.
_TARGETS = (
    (2, 4),
    (2, 16),
    (2, 64),
    (2, 256),
    (2, 1024),
)

# ~8-16x the online budget; n is capped per-K so the K-means score matrix
# stays a few GB.
_N_SAMPLES = 1_600_000
_ITERS = 60
_RESTARTS = 8
_SEED = 0


def fit_all(
    out_dir: str | None = None,
    device: str | None = None,
    *,
    targets: tuple[tuple[int, int], ...] = _TARGETS,
    overwrite: bool = False,
) -> None:
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    from ..vq import lookup_codewords, quantize_to_vq_codebook

    for b, k in targets:
        if b < 1 or k < 2 or k & (k - 1):
            raise ValueError(f"invalid codebook target {(b, k)}")
        path = os.path.join(out_dir, f"gauss_b{b}_k{k}.pt")
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"refusing to replace existing codebook: {path}")
        n = min(_N_SAMPLES, max(200_000, 400_000_000 // k))
        samples = _gaussian_block_samples(b, n, _SEED).to(device)
        cb = _kmeans_best_of(samples, k, _ITERS, _SEED, _RESTARTS)
        cb = cb.contiguous().cpu().to(torch.float32)
        eval_z = _gaussian_block_samples(b, 2_000_000 // b, 123).to(device)
        codes = quantize_to_vq_codebook(eval_z, cb.to(device))
        mse = float((eval_z - lookup_codewords(cb.to(device), codes)).pow(2).mean())
        torch.save(cb, path)
        print(
            f"wrote {path}  (K={k}, b={b}, n_fit={n}, eval MSE/dim={mse:.6f})",
            flush=True,
        )


def _parse_target(value: str) -> tuple[int, int]:
    try:
        block, codewords = value.split(":", maxsplit=1)
        target = int(block), int(codewords)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("target must be BLOCK:CODEWORDS") from exc
    if target not in _TARGETS:
        raise argparse.ArgumentTypeError(
            f"target {target} is not in the audited target set {_TARGETS}"
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", type=_parse_target, dest="targets")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    fit_all(
        args.output_dir,
        args.device,
        targets=tuple(args.targets or _TARGETS),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
