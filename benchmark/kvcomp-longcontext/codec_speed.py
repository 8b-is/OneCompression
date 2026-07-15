"""KV codec speed-quality micro-benchmark (the HayateQuant Pareto ablation).

The long-context harness (``kv_benchmark.py``) scores *accuracy/PPL/memory*; it
does not measure the codec's own encode/decode latency.  HALO and HayateQuant
both claim a **speed-quality** win over the shipped TurboQuant default, so the
paper needs the orthogonal *throughput* axis measured under identical
conditions.  This script produces that table directly: for each codec it times
encode and decode (warmup + median over repeats, CUDA-synchronised) **and**
measures the reconstruction quality (KV-MSE and softmax attention-output error)
on the *same* tensors, then reports each lever's effect:

  * ``TurboQuant-hadamard``      -- explicit fast WHT deployment approximation.
  * ``TurboQuant-paper``         -- exact Beta codebook + Gaussian-QR rotation.
  * ``HALO-hadamard-noLS``       -- HALO with the butterfly + RMS  (== TurboQuant
                                   quality, the internal control).
  * ``HALO-hadamard_gemm-noLS``  -- swap butterfly -> single dense GEMM.  Same
                                   math => *identical quality*, isolates SPEED.
  * ``HALO-hadamard_gemm-LS``    -- + least-squares dequant scale.  Free storage,
                                   isolates the QUALITY lever (full HALO default).
  * ``HALO-dense-LS``            -- Haar-random dense rotation instead of Hadamard:
                                   isolates the rotation choice (quality regresses).
  * ``HayateQuant-bs1``          -- block_size=1 sanity check: must be within
                                   noise of ``HALO-hadamard_gemm-NCIT`` (the
                                   superset property -- HayateQuant *is* HALO at
                                   ``block_size=1``), isolates the ADDED encode
                                   overhead of the block-VQ machinery itself
                                   (codebook lookup vs. scalar Lloyd-Max lookup)
                                   even when block-VQ is a no-op.
  * ``HayateQuant-auto``         -- the proposed default (block-VQ, auto block
                                   size): the actual speed/quality this codec
                                   ships with, so a reader sees the honest
                                   encode-latency cost of the accuracy gain
                                   (dominated by the ``O(K)`` codebook GEMM,
                                   ``K = 2**(bits*block_size)``) rather than
                                   only the CPU-only distortion numbers.

So GEMM buys speed at equal quality, LS buys quality at equal storage, the
Hadamard rotation is needed (dense regresses), and HayateQuant's block-VQ buys
*more* quality at a measured (not hand-waved) encode-latency cost -- a clean
ablation that substantiates "more accurate than TurboQuant/HALO, with an
honestly reported latency trade-off" rather than "strictly free".

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
from onecomp.kvcomp.codec.hayatequant import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.recipes import build_hayate_frontier, hayate_frontier_bits


def _codecs(bits: int) -> dict:
    """The ablation matrix at a fixed bit-width (keys & values both ``bits``).

    Adds the enhanced HALO scale levers (NC keys / LS values, and + ls_iter on
    values).  These are *quality* levers with essentially the same encode/decode
    cost as plain LS, so the speed table demonstrates the quality gain is free
    (no latency / throughput regression vs the shipped TurboQuant default).
    HayateQuant is added at both ``block_size=1`` (superset sanity check --
    isolates the block-VQ machinery's fixed overhead) and ``block_size=None``
    ("auto", the shipped default -- the actual proposed operating point).
    """
    codecs = {
        "TurboQuant-paper": TurboQuantCodec(bits=bits, rotation_mode="dense", dist="beta"),
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, rotation_mode="hadamard", dist="beta"),
        "HALO-hadamard-noLS": HALOCodec(bits=bits, rotation_mode="hadamard", ls_scale=False),
        "HALO-hadamard_gemm-noLS": HALOCodec(bits=bits, rotation_mode="hadamard_gemm", ls_scale=False),
        # Symmetric LS: explicit modes so the label stays correct regardless of class defaults.
        "HALO-hadamard_gemm-LS": HALOCodec(bits=bits, rotation_mode="hadamard_gemm",
                                            key_scale_mode="ls", value_scale_mode="ls"),
        # Enhanced HALO (proposed): role-asymmetric scale, iso-storage to LS.
        "HALO-hadamard_gemm-NC": HALOCodec(bits=bits, rotation_mode="hadamard_gemm",
                                           key_scale_mode="nc", value_scale_mode="ls"),
        "HALO-hadamard_gemm-NCIT": HALOCodec(bits=bits, rotation_mode="hadamard_gemm",
                                             key_scale_mode="nc", value_scale_mode="ls_iter"),
        # Symmetric LS with dense rotation: explicit to avoid default-ambiguity.
        "HALO-dense-LS": HALOCodec(bits=bits, rotation_mode="dense",
                                    key_scale_mode="ls", value_scale_mode="ls"),
        # HayateQuant (proposed, replaces HALO): superset sanity + shipped default
        # + the paper ablation variants (aligned with codec_benchmark.py).
        "HayateQuant-bs1": HayateQuantCodec(bits=bits, block_size=1, rotation_mode="auto"),
        "HayateQuant-auto": HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto"),
        # Exact-GEMM encoder (LUT ablation: isolates the LUT's speed contribution).
        "HayateQuant-exact": HayateQuantCodec(bits=bits, block_size=None,
                                              rotation_mode="auto", encoder="exact"),
        # Large-codebook quality ceiling (K up to 4096), affordable thanks to the LUT.
        "HayateQuant-K12": HayateQuantCodec(bits=bits, block_size=None,
                                            rotation_mode="auto", max_codebook_bits=12),
        # Dense rotation + symmetric LS (Qwen-family PPL point, iso to HALO-dense-LS).
        "HayateQuant-dense-LS": HayateQuantCodec(bits=bits, block_size=None,
                                                 rotation_mode="dense",
                                                 key_scale_mode="ls", value_scale_mode="ls"),
    }
    if bits >= 2:
        kbits, vbits = hayate_frontier_bits(bits)
        codecs[f"HQ-Frontier-K{kbits}V{vbits}"] = build_hayate_frontier(bits)
    return codecs


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


def _bench_regime(codec, keys, vals, device, repeats):
    """Time encode/decode and derive latency + throughput for one KV stack.

    Returns a dict with per-layer median latency (ms) and throughput
    (million KV-elements / second) for both encode and decode.  Throughput uses
    the total elements (keys + values) processed per layer so prefill (S large)
    and decode (S=1) regimes are directly comparable on the same scale.
    """
    payloads = [codec.encode(k, v) for k, v in zip(keys, vals)]

    def enc():
        for k, v in zip(keys, vals):
            codec.encode(k, v)

    def dec():
        for c in payloads:
            codec.decode(c)

    def rt():
        # The serving-path round trip (what CompressedDynamicCache.update
        # actually calls): fused encode+decode where the codec provides it,
        # plain encode->decode otherwise.
        for k, v in zip(keys, vals):
            codec.fake_quant_roundtrip(k, v)

    n_layers = len(keys)
    elem_per_layer = (keys[0].numel() + vals[0].numel())
    tok_per_layer = keys[0].shape[0] * keys[0].shape[2]  # batch * seq positions
    enc_ms = _time(enc, device, repeats) * 1e3 / n_layers
    dec_ms = _time(dec, device, repeats) * 1e3 / n_layers
    rt_ms = _time(rt, device, repeats) * 1e3 / n_layers
    # throughput: elements (Melem/s) and KV-token positions (Mtok/s) per second
    enc_melem_s = elem_per_layer / max(enc_ms, 1e-9) / 1e3
    dec_melem_s = elem_per_layer / max(dec_ms, 1e-9) / 1e3
    enc_mtok_s = tok_per_layer / max(enc_ms, 1e-9) / 1e3
    return dict(enc_ms=enc_ms, dec_ms=dec_ms, rt_ms=rt_ms,
                enc_melem_s=enc_melem_s, dec_melem_s=dec_melem_s,
                enc_mtok_s=enc_mtok_s)


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
    # Decode-regime (incremental, q_len=1) tensors: the matrix-vector shape every
    # generation step hits, where the GEMM advantage and per-token codec overhead
    # differ from the prefill (S large) regime.  Same layers/heads/dim, S=1.
    keys1 = [k[:, :, :1, :].contiguous() for k in keys]
    vals1 = [v[:, :, :1, :].contiguous() for v in vals]
    print(f"device={device} layers={len(keys)} heads={keys[0].shape[1]} "
          f"seq={keys[0].shape[2]} dim={d} model={os.path.basename(args.model) if args.model else 'synthetic'}")

    out = {"meta": vars(args), "results": {}}
    for bits in args.bits:
        codecs = _codecs(bits)
        print(f"\n--- bits={bits} (median latency over {args.repeats} reps; quality mean over layers) ---")
        print(f"  [prefill S={keys[0].shape[2]}]            "
              f"[decode S=1]")
        print(f"{'codec':26} {'enc ms':>8} {'dec ms':>8} {'enc x':>6} "
              f"{'enc Mel/s':>10} {'d1 enc us':>10} {'d1 dec us':>10} {'d1 rt us':>9} {'tok/s/L':>9} "
              f"{'KV-MSE':>9} {'mse x':>7} {'attn-err':>9} {'bits/el':>8}")
        base = {}
        rows = {}
        for label, codec in codecs.items():
            pf = _bench_regime(codec, keys, vals, device, args.repeats)
            de = _bench_regime(codec, keys1, vals1, device, max(args.repeats, 50))
            payloads = [codec.encode(k, v) for k, v in zip(keys, vals)]
            ms_q, aerr_q, sb = [], [], []
            for (k, v, q, c) in zip(keys, vals, queries, payloads):
                kq, vq = codec.decode(c)
                ms_q.append((_mse(kq, k) + _mse(vq, v)) / 2)
                aerr_q.append(_attn_err(k, v, kq, vq, q))
                sb.append(c.meta["stored_bits"] / (k.numel() + v.numel()))
            mm = sum(ms_q) / len(ms_q); aa = sum(aerr_q) / len(aerr_q); bb = sum(sb) / len(sb)
            # decode step latency in microseconds (per layer); tok/s per layer
            # is 1 token / roundtrip time -- the codec overhead a generated
            # token actually pays on the serving path (the fused round trip
            # where the codec provides one, encode->decode otherwise).
            d1_enc_us = de["enc_ms"] * 1e3
            d1_dec_us = de["dec_ms"] * 1e3
            d1_rt_us = de["rt_ms"] * 1e3
            tok_s_layer = 1.0 / max(de["rt_ms"] / 1e3, 1e-9)
            rows[label] = dict(
                prefill=dict(enc_ms=pf["enc_ms"], dec_ms=pf["dec_ms"], rt_ms=pf["rt_ms"],
                             enc_melem_s=pf["enc_melem_s"], dec_melem_s=pf["dec_melem_s"],
                             enc_mtok_s=pf["enc_mtok_s"]),
                decode=dict(enc_us=d1_enc_us, dec_us=d1_dec_us, rt_us=d1_rt_us,
                            tok_s_per_layer=tok_s_layer),
                kv_mse=mm, attn_err=aa, bits_per_el=bb)
            if not base:
                base = dict(enc=pf["enc_ms"], mse=mm, aerr=aa)
            print(f"{label:26} {pf['enc_ms']:8.3f} {pf['dec_ms']:8.3f} "
                  f"{base['enc']/max(pf['enc_ms'],1e-9):5.2f}x "
                  f"{pf['enc_melem_s']:10.2f} {d1_enc_us:10.2f} {d1_dec_us:10.2f} "
                  f"{d1_rt_us:9.2f} {tok_s_layer:9.0f} "
                  f"{mm:9.5f} {mm/max(base['mse'],1e-12):6.3f}x {aa:9.5f} {bb:8.3f}")
        out["results"][bits] = rows

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
