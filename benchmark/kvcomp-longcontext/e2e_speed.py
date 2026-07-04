"""End-to-end inference latency / throughput for KV codecs (real model).

``codec_speed.py`` times the codec kernels in isolation; this script measures
what they cost *inside* a real generation loop, the number a deployer cares
about: prefill latency (time-to-first-token, TTFT), per-token decode latency,
decode throughput (tokens/s), and peak GPU memory.

The research cache is fake-quant (encode->decode keeps an fp16 tensor), so the
*model* matmuls are unchanged; the measured delta vs FullKV is exactly the
per-step codec overhead each generated token pays.  The enhanced HALO scale
levers (NC keys / LS values, +ls_iter) are quality levers with the same
encode/decode cost as plain LS, so this table shows the quality win is free.
HayateQuant's block-VQ is *not* free -- this table is exactly where that
honest cost (extra ``ms/tok`` and lower peak ``tok/s`` vs. TurboQuant/HALO at
equal bits) is reported, alongside its accuracy gain measured elsewhere
(``codec_benchmark.py`` / ``verify_hayatequant_vq_gain.py``).

Methods (each at the same bit-width, codec applied to *every* layer so the
overhead is the honest worst case -- no outlier-layer skip):

    FullKV            transformers.DynamicCache (fp16 upper bound)
    TurboQuant-Nbit   shipped rotated Lloyd-Max (beta + butterfly Hadamard)
    HALO-NCIT-Nbit    strongest prior baseline (hadamard_gemm + NC/ls_iter;
                      superseded by HayateQuant)
    HayateQuant-bs1-Nbit   HayateQuant, block_size=1 (superset-of-HALO sanity;
                           isolates the block-VQ machinery's fixed overhead)
    HayateQuant-Nbit       HayateQuant, block_size=None ("auto", shipped
                           default) -- the PROPOSED operating point

Run (GPU node):
    python e2e_speed.py --model <DIR> --prompt-len 4096 --gen-len 128 \
        --bits 3 --repeats 3 --out results_codec_speed/e2e_<model>.json
Run (CPU smoke):
    python e2e_speed.py --model <DIR> --prompt-len 256 --gen-len 16 --device cpu

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from onecomp.kvcomp import build_codec
from onecomp.kvcomp.cache import CompressedDynamicCache


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _methods(bits, model_config):
    """Codec factory per method label (cache built fresh for every run)."""
    sink = 4

    def comp(codec):
        return lambda: CompressedDynamicCache(
            codec=codec, config=model_config, sink_tokens=sink, skip_layers=[])

    return {
        "FullKV": lambda: DynamicCache(config=model_config),
        f"TurboQuant-{bits}bit": comp(build_codec(
            "turboquant", bits=bits, rotation_mode="hadamard", dist="beta")),
        f"HALO-NCIT-{bits}bit": comp(build_codec(
            "halo", bits=bits, rotation_mode="hadamard_gemm",
            key_scale_mode="nc", value_scale_mode="ls_iter")),
        f"HayateQuant-bs1-{bits}bit": comp(build_codec(
            "hayatequant", bits=bits, block_size=1, rotation_mode="auto")),
        f"HayateQuant-{bits}bit": comp(build_codec(
            "hayatequant", bits=bits, block_size=None, rotation_mode="auto")),
    }


@torch.no_grad()
def _run_once(model, input_ids, make_cache, gen_len, device):
    """One prefill + greedy decode loop; returns timing + memory for this run."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cache = make_cache()
    n = input_ids.shape[1]
    pos = torch.arange(n, device=device)

    # --- prefill (TTFT) ---
    _sync(device); t0 = time.perf_counter()
    out = model(input_ids, past_key_values=cache, use_cache=True, cache_position=pos)
    _sync(device); ttft = time.perf_counter() - t0
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)

    # --- decode loop (per-step latency) ---
    cur = n
    step_t = []
    for _ in range(gen_len):
        cp = torch.tensor([cur], device=device)
        _sync(device); s0 = time.perf_counter()
        out = model(nxt, past_key_values=cache, use_cache=True,
                    cache_position=cp, position_ids=cp.unsqueeze(0))
        _sync(device); step_t.append(time.perf_counter() - s0)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        cur += 1

    cr = cache.compression_ratio() if hasattr(cache, "compression_ratio") else 1.0
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    return dict(ttft=ttft, step_t=step_t, compression_ratio=float(cr), peak_mb=peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--bits", type=int, nargs="+", default=[3])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16, float32=torch.float32)[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    text = "The history of cartography, navigation and scientific computing. " * 4000
    ids = tok(text, return_tensors="pt").input_ids[:, : args.prompt_len].to(device)
    print(f"model={os.path.basename(args.model)} device={device} dtype={args.dtype} "
          f"prompt_len={ids.shape[1]} gen_len={args.gen_len} repeats={args.repeats}")

    out = {"meta": {**vars(args), "prompt_len_actual": int(ids.shape[1])}, "results": {}}
    for bits in args.bits:
        methods = _methods(bits, model.config)
        print(f"\n--- bits={bits} (median over {args.repeats} reps) ---")
        print(f"{'method':20} {'TTFT ms':>9} {'tok/s':>8} {'ms/tok':>8} "
              f"{'tok/s x':>8} {'peak MB':>9} {'CR':>6}")
        base_tps = None
        rows = {}
        for label, make in methods.items():
            ttfts, tpss, mspt, peaks, cr = [], [], [], [], 1.0
            for _ in range(args.repeats):
                r = _run_once(model, ids, make, args.gen_len, device)
                ttfts.append(r["ttft"] * 1e3)
                # exclude the first decode step (often a one-off allocation) from
                # the steady-state throughput, if we have more than one step.
                steps = r["step_t"][1:] if len(r["step_t"]) > 1 else r["step_t"]
                tot = sum(steps)
                tpss.append(len(steps) / max(tot, 1e-9))
                mspt.append(1e3 * tot / max(len(steps), 1))
                peaks.append(r["peak_mb"]); cr = r["compression_ratio"]
            med = lambda xs: sorted(xs)[len(xs) // 2]
            ttft_ms, tok_s, ms_tok, peak_mb = med(ttfts), med(tpss), med(mspt), med(peaks)
            if base_tps is None:
                base_tps = tok_s
            rows[label] = dict(ttft_ms=ttft_ms, decode_tok_s=tok_s, decode_ms_per_tok=ms_tok,
                               peak_mb=peak_mb, compression_ratio=cr)
            print(f"{label:20} {ttft_ms:9.2f} {tok_s:8.1f} {ms_tok:8.3f} "
                  f"{tok_s/max(base_tps,1e-9):7.2f}x {peak_mb:9.1f} {cr:6.2f}")
        out["results"][bits] = rows

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
