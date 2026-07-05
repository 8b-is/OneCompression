"""Micro-benchmark of HayateQuant's nearest-codeword encoder paths (GPU).

Measures, per operating point, the wall-clock and the excess distortion of
every candidate encoder implementation, to decide -- with data, not
hand-waving -- which of the following levers are worth strengthening:

  P1  LUT for (b=4, K<=512): the current ``_use_lut`` auto policy excludes
      block_size>=4 with K<=512 (2-bit auto resolves there).  Is the
      exclusion still right after the fp16-scoring work, or does the LUT now
      win there too?
  P2  fp16 candidate-distance inside ``quantize_via_lut``: the candidate
      list is exact; only the argmin over T candidates would see fp16
      rounding (same argument as the fp16 exact-GEMM scoring).  Does it
      measurably speed up the T=8/16 paths?
  P3  block_size=1 sorted-codebook shortcut: 1-D Lloyd-Max codebooks are
      already sorted, yet ``quantize_to_vq_codebook`` re-sorts and pays an
      extra full-N ``order[codes]`` gather every call.  How much does
      skipping the permutation when the codebook is sorted save?

Run (GPU):
    python bench_encoder_paths.py --device cuda --repeats 50

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from onecomp.kvcomp.codec.hayate.vq import (
    lloyd_vq_gaussian_block,
    quantize_to_vq_codebook,
    build_vq_lut,
    quantize_via_lut,
    lut_spec_for,
)
from onecomp.kvcomp.codec.rotation import lloyd_max_gaussian, quantize_to_codebook


def _timeit(fn, repeats, device):
    for _ in range(5):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3  # ms


def _distortion(z, cb, codes):
    return float((cb[codes] - z).pow(2).sum(-1).mean())


def quantize_via_lut_fp16(z, lut, bins, cb, lim=5.0):
    """P2 variant: exact candidate list, fp16 distance over the T candidates."""
    b = z.shape[-1]
    step = bins / (2.0 * lim)
    zi = ((z.clamp(-lim, lim) + lim) * step).long().clamp_(0, bins - 1)
    flat = zi[..., 0]
    for j in range(1, b):
        flat = flat * bins + zi[..., j]
    cands = lut[flat].long()
    if cands.shape[-1] == 1:
        return cands.squeeze(-1)
    zh = z.half()
    cwh = cb.half()[cands]
    dist = (cwh - zh.unsqueeze(-2)).pow(2).sum(-1)
    best = dist.argmin(dim=-1, keepdim=True)
    return cands.gather(-1, best).squeeze(-1)


def bench_block_paths(bits, block_size, n_blocks, device, repeats):
    k = 1 << (bits * block_size)
    cb = lloyd_vq_gaussian_block(bits, block_size).to(device)
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(n_blocks, block_size, generator=gen).to(device)

    out = {"bits": bits, "block_size": block_size, "K": k, "n_blocks": n_blocks}

    codes_exact = quantize_to_vq_codebook(z, cb)
    d_exact = _distortion(z, cb, codes_exact)
    out["exact_ms"] = _timeit(lambda: quantize_to_vq_codebook(z, cb), repeats, device)
    out["exact_dist"] = d_exact

    spec = lut_spec_for(block_size, k)
    if spec is not None:
        bins, top_t = spec
        lut = build_vq_lut(cb, bins, top_t)
        codes_lut = quantize_via_lut(z, lut, bins, cb)
        out["lut_spec"] = [bins, top_t]
        out["lut_ms"] = _timeit(lambda: quantize_via_lut(z, lut, bins, cb), repeats, device)
        out["lut_excess_dist_pct"] = 100.0 * (_distortion(z, cb, codes_lut) / d_exact - 1.0)
        codes_lh = quantize_via_lut_fp16(z, lut, bins, cb)
        out["lut_fp16_ms"] = _timeit(
            lambda: quantize_via_lut_fp16(z, lut, bins, cb), repeats, device)
        out["lut_fp16_excess_dist_pct"] = 100.0 * (_distortion(z, cb, codes_lh) / d_exact - 1.0)
    return out


def bench_bs1_paths(bits, n, device, repeats):
    """P3: current bs1 path (sort + permute) vs direct searchsorted."""
    cb1 = lloyd_max_gaussian(bits).to(device)          # sorted 1-D codebook
    cb = cb1.unsqueeze(-1)
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(n, 1, generator=gen).to(device)

    out = {"bits": bits, "block_size": 1, "K": 1 << bits, "n_blocks": n}
    out["current_ms"] = _timeit(lambda: quantize_to_vq_codebook(z, cb), repeats, device)

    zs = z.squeeze(-1)
    out["direct_ms"] = _timeit(lambda: quantize_to_codebook(zs, cb1), repeats, device)
    assert torch.equal(quantize_to_vq_codebook(z, cb), quantize_to_codebook(zs, cb1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device)

    results = []
    # Realistic block counts: 8 kv-heads x 4096 tokens x (128/b) blocks/vector.
    for bits, b in [(2, 4), (3, 2), (4, 2), (3, 4)]:  # (3,4)=K12 large-codebook point
        n_blocks = 8 * 4096 * (128 // b)
        r = bench_block_paths(bits, b, n_blocks, device, args.repeats)
        results.append(r)
        print(json.dumps(r))
    for bits in (2, 3, 4):
        r = bench_bs1_paths(bits, 8 * 4096 * 128, device, args.repeats)
        results.append(r)
        print(json.dumps(r))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
