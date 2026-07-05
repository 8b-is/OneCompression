"""Measure whether residual block-VQ (RVQ) beats HayateQuant's current
single-stage block-VQ at equal bits/coordinate, on the exact data-free
source law (i.i.d. N(0,1) after rotation + RMS normalisation).

Motivation: the lever-isolation ablation showed block size is the dominant
accuracy lever (2-bit PPL 10.02 -> 9.51 -> 9.15 for b = 1 -> 2 -> 4), and
classical space-filling theory says the b=4 -> b=8 step still has gain left.
But full VQ at b=8, 2-bit needs K = 2^16 codewords (infeasible to fit or
search). Residual VQ splits the 16-bit/block budget into two 8-bit stages
(K=256 each, storage-neutral): stage 1 quantizes the block, stage 2
quantizes the residual. Greedy RVQ loses part of the joint-VQ gain; an
*encode-time alternating refinement* (re-search stage i holding the other
stages fixed) recovers much of it, at one extra small GEMM per pass.
Codebooks stay data-free: stage 1 is fit on N(0,1)^b samples, stage 2 on
the empirical stage-1 residuals of those samples, optionally followed by a
joint alternating refit (still on synthetic data only).

Grid (equal stored bits per coordinate in every row):
  2-bit: b1 scalar / b2 K16 / b4 K256 (current auto) / RVQ8 (8+8) greedy,
         +refine1, +refine2, joint-refit codebooks +refine1
  3-bit: b2 K64 (auto) / b4 K4096 (K12) / RVQ8 (12+12) greedy, +refine1
  4-bit: b2 K256 (auto) / RVQ4 (8+8) greedy, +refine1

Also times encode at prefill scale (1x32x4096x128 = 4.2M coords) for the
current LUT path vs the RVQ path (greedy / refine1).

Run:  srun -p debug --gres=gpu:1 .venv/bin/python -u bench_rvq_gain.py
"""
from __future__ import annotations

import json
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from onecomp.kvcomp.codec.hayate.vq import (  # noqa: E402
    _kmeans_best_of,
    lloyd_vq_gaussian_block,
    quantize_to_vq_codebook,
    lookup_codewords,
    lut_spec_for,
    build_vq_lut,
    quantize_via_lut,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _gauss(n, b, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, b, generator=gen).to(DEV)


# --- RVQ fit / encode -------------------------------------------------------

def rvq_encode(z, cbs, refine=0):
    """Greedy stage-wise nearest-codeword, then `refine` alternating passes
    (re-search stage i against z minus the other stages' reconstruction)."""
    codes = []
    resid = z
    for cb in cbs:
        c = quantize_to_vq_codebook(resid, cb)
        resid = resid - lookup_codewords(cb, c)
        codes.append(c)
    for _ in range(refine):
        for i in range(len(cbs)):
            recon_others = sum(
                lookup_codewords(cbs[j], codes[j]) for j in range(len(cbs)) if j != i)
            codes[i] = quantize_to_vq_codebook(z - recon_others, cbs[i])
    return codes


def rvq_decode(codes, cbs):
    return sum(lookup_codewords(cbs[i], codes[i]) for i in range(len(cbs)))


def fit_rvq(b, stage_bits, n=800_000, seed=0, iters=25, restarts=2, joint_iters=0):
    """Data-free RVQ codebooks: stage-wise K-means on synthetic N(0,1)^b,
    each stage fit on the previous stages' residuals; optional joint
    alternating refit (re-encode with refinement, then recompute each
    stage's centroids as the mean of its assigned targets)."""
    # _kmeans materialises the full (n, K) score matrix; cap n so it stays
    # a few GB even at the K=4096 stages.
    k_max = 1 << max(stage_bits)
    n = min(n, max(200_000, 400_000_000 // k_max))
    x = _gauss(n, b, seed)
    cbs = []
    resid = x
    for si, sb in enumerate(stage_bits):
        cb = _kmeans_best_of(resid, 1 << sb, iters, seed + 31 * si, restarts)
        c = quantize_to_vq_codebook(resid, cb)
        resid = resid - lookup_codewords(cb, c)
        cbs.append(cb)
    for _ in range(joint_iters):
        codes = rvq_encode(x, cbs, refine=1)
        for i in range(len(cbs)):
            target = x - sum(
                lookup_codewords(cbs[j], codes[j]) for j in range(len(cbs)) if j != i)
            k = cbs[i].shape[0]
            counts = torch.zeros(k, device=DEV).index_add_(
                0, codes[i], torch.ones(x.shape[0], device=DEV))
            sums = torch.zeros(k, b, device=DEV).index_add_(0, codes[i], target)
            keep = counts == 0
            newc = torch.where(keep.unsqueeze(1), cbs[i],
                               sums / counts.clamp_min(1).unsqueeze(1))
            cbs[i] = newc
    return cbs


# --- distortion measurements ------------------------------------------------

def mse_single(bits, b, n_eval=2_000_000, seed=123):
    cb = lloyd_vq_gaussian_block(bits, b).to(DEV)
    z = _gauss(n_eval // b, b, seed)
    c = quantize_to_vq_codebook(z, cb)
    return float((z - lookup_codewords(cb, c)).pow(2).mean())


def mse_single_bigfit(bits, b, n_fit=800_000, restarts=8, n_eval=2_000_000, seed=123):
    """Same single-stage codebook but with an 8x fit budget (free at runtime:
    the codebook is built once and cached) -- checks whether the default
    Lloyd budget (100k samples, 4 restarts) leaves accuracy on the table."""
    k = 1 << (bits * b)
    n_fit = min(n_fit, max(200_000, 400_000_000 // k))
    samples = _gauss(n_fit, b, 0)
    cb = _kmeans_best_of(samples, k, 40, 0, restarts)
    z = _gauss(n_eval // b, b, seed)
    c = quantize_to_vq_codebook(z, cb)
    return float((z - lookup_codewords(cb, c)).pow(2).mean())


def mse_rvq(b, stage_bits, refine, joint_iters, n_eval=2_000_000, seed=123):
    cbs = fit_rvq(b, stage_bits, joint_iters=joint_iters)
    z = _gauss(n_eval // b, b, seed)
    codes = rvq_encode(z, cbs, refine=refine)
    return float((z - rvq_decode(codes, cbs)).pow(2).mean())


# --- encode timing at prefill scale ----------------------------------------

def _time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def timing_2bit():
    n_coords = 1 * 32 * 4096 * 128
    out = {}
    # current path: b4 K256 via LUT (64, 2)
    cb4 = lloyd_vq_gaussian_block(2, 4).to(DEV)
    bins, t = lut_spec_for(4, 256)
    lut = build_vq_lut(cb4, bins, t)
    z4 = torch.randn(n_coords // 4, 4, device=DEV)
    out["b4K256_lut_ms"] = _time(lambda: quantize_via_lut(z4, lut, bins, cb4))
    out["b4K256_exact_ms"] = _time(lambda: quantize_to_vq_codebook(z4, cb4))
    # RVQ path: b8 (8+8)
    cbs = fit_rvq(8, (8, 8))
    z8 = torch.randn(n_coords // 8, 8, device=DEV)
    out["rvq8_greedy_ms"] = _time(lambda: rvq_encode(z8, cbs, refine=0))
    out["rvq8_refine1_ms"] = _time(lambda: rvq_encode(z8, cbs, refine=1))
    out["rvq8_refine2_ms"] = _time(lambda: rvq_encode(z8, cbs, refine=2))
    return out


def main():
    torch.manual_seed(0)
    res = {"device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"}

    print("=== 2 bit/coord ===", flush=True)
    res["2bit"] = {
        "b1_scalar": mse_single(2, 1),
        "b2_K16": mse_single(2, 2),
        "b4_K256_current": mse_single(2, 4),
        "b4_K256_bigfit": mse_single_bigfit(2, 4),
        "rvq8_8+8_greedy": mse_rvq(8, (8, 8), refine=0, joint_iters=0),
        "rvq8_8+8_refine1": mse_rvq(8, (8, 8), refine=1, joint_iters=0),
        "rvq8_8+8_refine2": mse_rvq(8, (8, 8), refine=2, joint_iters=0),
        "rvq8_8+8_joint5_refine1": mse_rvq(8, (8, 8), refine=1, joint_iters=5),
        "rvq8_4x4_refine1": mse_rvq(8, (4, 4, 4, 4), refine=1, joint_iters=0),
    }
    for k, v in res["2bit"].items():
        print(f"  {k}: {v:.6f}", flush=True)

    print("=== 3 bit/coord ===", flush=True)
    res["3bit"] = {
        "b2_K64_current": mse_single(3, 2),
        "b4_K4096_K12": mse_single(3, 4),
        "b4_K4096_bigfit": mse_single_bigfit(3, 4),
        "rvq8_12+12_greedy": mse_rvq(8, (12, 12), refine=0, joint_iters=0),
        "rvq8_12+12_refine1": mse_rvq(8, (12, 12), refine=1, joint_iters=0),
        "rvq8_12+12_joint3_refine1": mse_rvq(8, (12, 12), refine=1, joint_iters=3),
        "rvq8_8x3_refine1": mse_rvq(8, (8, 8, 8), refine=1, joint_iters=0),
        "rvq8_8x3_joint5_refine1": mse_rvq(8, (8, 8, 8), refine=1, joint_iters=5),
        "rvq4_6+6_refine1": mse_rvq(4, (6, 6), refine=1, joint_iters=0),
        "rvq4_6+6_joint5_refine1": mse_rvq(4, (6, 6), refine=1, joint_iters=5),
    }
    for k, v in res["3bit"].items():
        print(f"  {k}: {v:.6f}", flush=True)

    print("=== 4 bit/coord ===", flush=True)
    res["4bit"] = {
        "b2_K256_current": mse_single(4, 2),
        "rvq4_8+8_greedy": mse_rvq(4, (8, 8), refine=0, joint_iters=0),
        "rvq4_8+8_refine1": mse_rvq(4, (8, 8), refine=1, joint_iters=0),
        "rvq4_8+8_joint5_refine1": mse_rvq(4, (8, 8), refine=1, joint_iters=5),
    }
    for k, v in res["4bit"].items():
        print(f"  {k}: {v:.6f}", flush=True)

    if DEV == "cuda":
        print("=== encode timing @4.2M coords (2-bit) ===", flush=True)
        res["timing_2bit"] = timing_2bit()
        for k, v in res["timing_2bit"].items():
            print(f"  {k}: {v:.3f} ms", flush=True)

    out = "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results/rvq_gain.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
