"""Fit and save the high-budget data-free block-VQ codebook artifacts.

The online fit in :func:`.._vq.lloyd_vq_gaussian_block` uses a budget sized
for acceptable first-encode latency (100k samples, 25 iters, 4 restarts).
Fitting once, offline, with ~8-16x that budget measurably improves the
codebook on the exact source law it is fit to (block distortion -1.4% at the
2-bit ``(b=4, K=256)`` point and -2.6% at the 3-bit ``(b=4, K=4096)`` point;
``ichikawa_works/kvcomp_hayatequant/bench_rvq_gain.py``) at zero runtime
cost.  This script fits every ``(block_size, K)`` pair the codec's default
configurations can resolve and saves them as plain fp32 tensors
``gauss_b{b}_k{K}.pt`` next to this file; ``lloyd_vq_gaussian_block``
auto-prefers a matching artifact over the online fit.

Still 100% data-free: the fit sees only synthetic ``N(0,1)`` samples (the
same law as the online fit), never any real KV data.  Committing the
artifact also makes the codebook bit-exactly reproducible across machines
and BLAS builds (GPU K-means kernels are not bitwise deterministic, so two
online fits can differ in the last ulp; a fixed artifact cannot).

Run (GPU strongly recommended for the K=4096 fit):
    python -m onecomp.kvcomp.codec.codebooks.fit_codebooks
"""
from __future__ import annotations

import os

import torch

from ..vq import _kmeans_best_of, _gaussian_block_samples

# Every (block_size, K) the default configs resolve to:
#   1-bit: b=8 K=256 | 2-bit: b=4 K=256, b=2 K=16 | 3-bit: b=2 K=64,
#   b=4 K=4096 (max_codebook_bits=12) | 4-bit: b=2 K=256.
_TARGETS = ((2, 16), (2, 64), (2, 256), (4, 256), (4, 4096), (8, 256))

# ~8-16x the online budget; n is capped per-K so the K-means score matrix
# stays a few GB.
_N_SAMPLES = 1_600_000
_ITERS = 60
_RESTARTS = 8
_SEED = 0


def fit_all(out_dir: str | None = None, device: str | None = None) -> None:
    out_dir = out_dir or os.path.dirname(os.path.abspath(__file__))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    from ..vq import quantize_to_vq_codebook, lookup_codewords

    for b, k in _TARGETS:
        n = min(_N_SAMPLES, max(200_000, 400_000_000 // k))
        samples = _gaussian_block_samples(b, n, _SEED).to(device)
        cb = _kmeans_best_of(samples, k, _ITERS, _SEED, _RESTARTS)
        cb = cb.contiguous().cpu().to(torch.float32)
        eval_z = _gaussian_block_samples(b, 2_000_000 // b, 123).to(device)
        codes = quantize_to_vq_codebook(eval_z, cb.to(device))
        mse = float((eval_z - lookup_codewords(cb.to(device), codes)).pow(2).mean())
        path = os.path.join(out_dir, f"gauss_b{b}_k{k}.pt")
        torch.save(cb, path)
        print(f"wrote {path}  (K={k}, b={b}, n_fit={n}, eval MSE/dim={mse:.6f})",
              flush=True)


if __name__ == "__main__":
    fit_all()
