"""KV codec speed-quality micro-benchmark (the HALO Pareto ablation).

The long-context harness (``kv_benchmark.py``) scores *accuracy/PPL/memory*; it
does not measure the codec's own encode/decode latency.  The proposed HALO codec
claims a **speed-quality** win over the shipped TurboQuant default, so the paper
needs the orthogonal *throughput* axis measured under identical conditions.  This
script produces that table directly: for each codec it times encode and decode
(warmup + median over repeats, CUDA-synchronised) **and** measures the
reconstruction quality (KV-MSE and softmax attention-output error) on the *same*
tensors, then reports each lever's effect:

  * ``TurboQuant-hadamard``      -- shipped default (butterfly WHT + RMS scale).
  * ``HALO-hadamard-noLS``       -- HALO with the butterfly + RMS  (== TurboQuant
                                   quality, the internal control).
  * ``HALO-hadamard_gemm-noLS``  -- swap butterfly -> single dense GEMM.  Same
                                   math => *identical quality*, isolates SPEED.
  * ``HALO-hadamard_gemm-LS``    -- + least-squares dequant scale.  Free storage,
                                   isolates the QUALITY lever (full HALO default).
  * ``HALO-dense-LS``            -- Haar-random dense rotation instead of Hadamard:
                                   isolates the rotation choice (quality regresses).

So GEMM buys speed at equal quality, LS buys quality at equal storage, and the
Hadamard rotation is needed (dense regresses) -- a clean three-way ablation that
substantiates "faster *and* no worse than TurboQuant".

Synthetic Gaussian KV by default; ``--model DIR`` captures real KV instead.

Run (GPU node):
    python codec_speed.py --device cuda --heads 32 --seq 4096 --dim 128 \
        --bits 2 3 4 --repeats 30 --out results/codec_speed_cuda.json
Run (CPU smoke):
    python codec_speed.py --device cpu --heads 4 --seq 512 --dim 128 --repeats 5

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.codec.halo import HALOCodec


def _codecs(bits: int) -> dict:
    """The ablation matrix at a fixed bit-width (keys & values both ``bits``)."""
    return {
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, rotation_mode="hadamard", dist="beta"),
        "HALO-hadamard-noLS": HALOCodec(bits=bits, rotation_mode="hadamard", ls_scale=False),
        "HALO-hadamard_gemm-noLS": HALOCodec(bits=bits, rotation_mode="hadamard_gemm", ls_scale=False),
        "HALO-hadamard_gemm-LS": HALOCodec(bits=bits, rotation_mode="hadamard_gemm", ls_scale=True),
        "HALO-dense-LS": HALOCodec(bits=bits, rotation_mode="dense", ls_scale=True),
    }


def _make_kv(args, device):
    """Real KV (``--model``) or synthetic anisotropic Gaussian KV per layer."""
    if args.model:
        return _capture_kv(args.model, args.layers, args.seq, device)
    g = torch.Generator().manual_seed(args.seed)
    keys, vals = [], []
    for _ in range(args.layers):
        # Anisotropic: a few high-variance channels (massive-activation-like) so the
        # rotation choice actually matters, as on real KV.
        scale = torch.ones(args.dim)
        scale[: max(1, args.dim // 16)] = 6.0
        k = torch.randn(1, args.heads, args.seq, args.dim, generator=g) * scale
        v = torch.randn(1, args.heads, args.seq, args.dim, generator=g) * scale
        keys.append(k.to(device)); vals.append(v.to(device))
    return keys, vals


def _capture_kv(model_dir, n_layers, seq, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    ids = tok("The history of cartography and navigation. " * 200,
              return_tensors="pt").input_ids[:, :seq].to(device)
    with torch.no_grad():
        kv = model(ids, use_cache=True).past_key_values
    keys, vals = [], []
    layers = kv.layers if hasattr(kv, "layers") else list(zip(kv.key_cache, kv.value_cache))
    for L in layers:
        k = (L.keys if hasattr(L, "keys") else L[0]).float()
        v = (L.values if hasattr(L, "values") else L[1]).float()
        keys.append(k.to(device)); vals.append(v.to(device))
        if n_layers and len(keys) >= n_layers:
            break
    return keys, vals


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(repeats):
        _sync(device); t0 = time.perf_counter()
        fn()
        _sync(device); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]  # median seconds


def _mse(a, b):
    return float((a - b).pow(2).mean())


def _attn_err(k, v, kq, vq, q):
    d = k.shape[-1]; sc = 1.0 / math.sqrt(d)
    ao = lambda kk, vv: torch.softmax((q @ kk.transpose(-1, -2)) * sc, -1) @ vv
    o, oq = ao(k, v), ao(kq, vq)
    return float((o - oq).norm() / o.norm().clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="capture real KV from this HF dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    keys, vals = _make_kv(args, device)
    d = keys[0].shape[-1]
    g = torch.Generator().manual_seed(1234)
    queries = [torch.randn(k.shape[0], k.shape[1], 32, d, generator=g).to(device) for k in keys]
    n_elem = sum(k.numel() + v.numel() for k, v in zip(keys, vals))
    print(f"device={device} layers={len(keys)} heads={keys[0].shape[1]} "
          f"seq={keys[0].shape[2]} dim={d} model={os.path.basename(args.model) if args.model else 'synthetic'}")

    out = {"meta": vars(args), "results": {}}
    for bits in args.bits:
        codecs = _codecs(bits)
        print(f"\n--- bits={bits} (median latency over {args.repeats} reps; quality mean over layers) ---")
        print(f"{'codec':26} {'enc ms':>8} {'dec ms':>8} {'enc x':>7} {'KV-MSE':>10} "
              f"{'mse x':>7} {'attn-err':>9} {'aerr x':>7} {'bits/el':>8}")
        base = {}
        rows = {}
        for label, codec in codecs.items():
            # pre-encode payloads for decode timing
            payloads = [codec.encode(k, v) for k, v in zip(keys, vals)]

            def enc():
                for k, v in zip(keys, vals):
                    codec.encode(k, v)

            def dec():
                for c in payloads:
                    codec.decode(c)

            enc_ms = _time(enc, device, args.repeats) * 1e3 / len(keys)
            dec_ms = _time(dec, device, args.repeats) * 1e3 / len(keys)
            ms_q, aerr_q, sb = [], [], []
            for (k, v, q, c) in zip(keys, vals, queries, payloads):
                kq, vq = codec.decode(c)
                ms_q.append((_mse(kq, k) + _mse(vq, v)) / 2)
                aerr_q.append(_attn_err(k, v, kq, vq, q))
                sb.append(c.meta["stored_bits"] / (k.numel() + v.numel()))
            mm = sum(ms_q) / len(ms_q); aa = sum(aerr_q) / len(aerr_q); bb = sum(sb) / len(sb)
            rows[label] = dict(enc_ms=enc_ms, dec_ms=dec_ms, kv_mse=mm, attn_err=aa, bits_per_el=bb)
            if not base:
                base = dict(enc=enc_ms, mse=mm, aerr=aa)
            print(f"{label:26} {enc_ms:8.3f} {dec_ms:8.3f} {base['enc']/max(enc_ms,1e-9):6.2f}x "
                  f"{mm:10.5f} {mm/max(base['mse'],1e-12):6.3f}x {aa:9.5f} "
                  f"{aa/max(base['aerr'],1e-12):6.3f}x {bb:8.3f}")
        out["results"][bits] = rows

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
