"""Sweep (bins, T) for the b=4 candidate-list LUT at the 2-bit operating point.

The 2-bit prefill encode (block_size=4, K=256) is HayateQuant's one remaining
latency weak spot (TTFT 2.6x TurboQuant); its cost is dominated by the
T-candidate gather+distance step, so a smaller T is a direct speed lever --
IF the excess distortion stays negligible. Measure, don't guess.

Run (GPU): python bench_lut_spec_sweep.py --device cuda

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from onecomp.kvcomp.codec._vq import (
    lloyd_vq_gaussian_block,
    quantize_to_vq_codebook,
    build_vq_lut,
    quantize_via_lut,
)


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
    return (time.perf_counter() - t0) / repeats * 1e3


def sweep(bits: int, block_size: int, specs, device, repeats: int):
    cb = lloyd_vq_gaussian_block(bits, block_size).to(device)
    gen = torch.Generator().manual_seed(0)
    n_blocks = 32 * 4096 * (128 // block_size)  # codec_speed's real shape
    z = torch.randn(n_blocks, block_size, generator=gen).to(device)

    codes_exact = quantize_to_vq_codebook(z, cb)
    d_exact = float((cb[codes_exact] - z).pow(2).sum(-1).mean())
    t_exact = _timeit(lambda: quantize_to_vq_codebook(z, cb), repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "exact",
                      "ms": round(t_exact, 4), "dist": d_exact}))

    for bins, top_t in specs:
        lut = build_vq_lut(cb, bins, top_t)
        codes = quantize_via_lut(z, lut, bins, cb)
        d = float((cb[codes] - z).pow(2).sum(-1).mean())
        t = _timeit(lambda: quantize_via_lut(z, lut, bins, cb), repeats, device)
        print(json.dumps({
            "bits": bits, "b": block_size, "path": f"lut_bins{bins}_T{top_t}",
            "ms": round(t, 4),
            "excess_dist_pct": round(100.0 * (d / d_exact - 1.0), 5),
            "lut_mb": round(lut.numel() * 4 / 1e6, 1),
        }))


def bench_decode_gather(bits: int, block_size: int, device, repeats: int):
    """Isolate the decode-side ``cb[codes]`` gather cost (the 2-bit prefill
    decode is ~10x the 3-bit one in codec_speed; find out whether the b=4
    codebook gather itself is the cause or something else in the pipeline),
    and try candidate fixes."""
    k = 1 << (bits * block_size)
    cb = lloyd_vq_gaussian_block(bits, block_size).to(device)
    n_blocks = 32 * 4096 * (128 // block_size)
    codes = torch.randint(0, k, (n_blocks,), device=device)
    t = _timeit(lambda: cb[codes], repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "decode_gather",
                      "ms": round(t, 4)}))

    # Variant A: flattened take() -- one contiguous 1-D gather of b*N scalars.
    def _take():
        idx = codes.unsqueeze(-1) * block_size + torch.arange(
            block_size, device=codes.device)
        return cb.reshape(-1).take(idx)
    r = _take()
    assert torch.equal(r, cb[codes])
    t = _timeit(_take, repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "decode_take",
                      "ms": round(t, 4)}))

    # Variant B: index_select (2-D row gather, same semantics as cb[codes]).
    def _isel():
        return torch.index_select(cb, 0, codes.reshape(-1))
    t = _timeit(_isel, repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "decode_index_select",
                      "ms": round(t, 4)}))

    # Variant C: embedding (dedicated fused row-gather kernel).
    def _emb():
        return torch.nn.functional.embedding(codes, cb)
    t = _timeit(_emb, repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "decode_embedding",
                      "ms": round(t, 4)}))

    # Variant D: int32 indices + index_select.
    codes32 = codes.to(torch.int32)
    def _isel32():
        return torch.index_select(cb, 0, codes32)
    t = _timeit(_isel32, repeats, device)
    print(json.dumps({"bits": bits, "b": block_size, "path": "decode_index_select_i32",
                      "ms": round(t, 4)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    device = torch.device(args.device)

    # 2-bit default point (K=256): can T shrink below the current 8?
    sweep(2, 4, [(32, 8), (32, 4), (48, 4), (64, 4), (48, 2), (64, 2)],
          device, args.repeats)
    # K12 3-bit point (K=4096): can T shrink below the current 16?
    sweep(3, 4, [(32, 16), (32, 8), (48, 8), (64, 8), (64, 4)],
          device, args.repeats)
    # Decode-side gathers for the 2-bit (b=4) vs 3-bit (b=2) operating points.
    bench_decode_gather(2, 4, device, args.repeats)
    bench_decode_gather(3, 2, device, args.repeats)


if __name__ == "__main__":
    main()
