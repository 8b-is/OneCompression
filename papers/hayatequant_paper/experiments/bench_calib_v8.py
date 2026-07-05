"""Measure the optional SpinQuant-style calibration for HayateQuant.

Per model: capture real KV samples on *WikiText-2 train* text (held out from
the PPL eval, which uses the test split), fit the calibration artifact, and
measure distortion + PPL + encode latency for the decomposition

    data-free v7 auto  |  rotation-only  |  codebook-only  |  both

at 2-bit (headline) and 3-bit, plus a TurboQuant-hadamard reference row.
Adoption rule: adopt only levers that improve PPL consistently across models;
'both' must beat each single lever for the combination story to hold.

Usage:  srun --gres=gpu:1 python -u bench_calib_v8.py <model_path>
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers  # noqa: E402
from onecomp.kvcomp.eval import wikitext2_perplexity  # noqa: E402
from onecomp.kvcomp.codec import HayateQuantCodec, TurboQuantCodec  # noqa: E402
from onecomp.kvcomp.codec.hayate.calibration import (  # noqa: E402
    calibrate_hayatequant, capture_kv_samples, HayateQuantCalibration)

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct")
OUT_DIR = "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results"


def calib_texts(n_texts=8, min_chars=6000):
    """Long paragraphs from the WikiText-2 TRAIN split (eval uses test)."""
    for repo in ("wikitext", "Salesforce/wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("wikitext train split unavailable")
    texts, buf = [], []
    for line in ds["text"]:
        buf.append(line)
        if sum(len(s) for s in buf) >= min_chars:
            texts.append("".join(buf))
            buf = []
            if len(texts) >= n_texts:
                break
    return texts


@torch.no_grad()
def capture_eval_kv(model, tok, max_tokens=2048):
    text = ("The history of cartography spans many centuries of patient observation. "
            "Merchants traded spices along routes that crossed deserts and mountains. ") * 64
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=max_tokens).input_ids.to(model.device)
    out = model(ids, use_cache=True)
    layers = out.past_key_values.layers
    n = len(layers)
    pick = [n // 4, n // 2, 3 * n // 4]
    ks = torch.cat([layers[i].keys for i in pick], dim=1).float()
    vs = torch.cat([layers[i].values for i in pick], dim=1).float()
    return ks, vs


@torch.no_grad()
def relmse(codec, x, role):
    kd, vd = codec.decode(codec.encode(x, x))
    d = kd if role == "k" else vd
    return float((d.float() - x).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


@torch.no_grad()
def enc_ms(codec, x, iters=50, warmup=10):
    for _ in range(warmup):
        codec.decode(codec.encode(x, x))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.encode(x, x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda:0").eval()
    ks, vs = capture_eval_kv(model, tok)
    D = ks.shape[-1]
    timing_x = torch.randn(1, 8, 1024, D, device=model.device)
    print(f"model={os.path.basename(MODEL)} head_dim={D}", flush=True)

    calib_ids = tok("The history of cartography spans many centuries. " * 8,
                    return_tensors="pt", truncation=True,
                    max_length=512).input_ids.to(model.device)
    skip = set(detect_outlier_layers(model, calib_ids, ratio_thresh=50.0))
    print(f"outlier layers: {sorted(skip)}", flush=True)

    print("capturing calibration KV (wikitext-2 TRAIN) ...", flush=True)
    kc, vc = capture_kv_samples(model, tok, calib_texts(), max_tokens=2048,
                                layer_stride=4)
    print(f"captured {kc.shape[0]} vectors/role", flush=True)

    results = {}

    def bench(name, codec):
        e = {"distortion_key": round(relmse(codec, ks, "k"), 5),
             "distortion_value": round(relmse(codec, vs, "v"), 5),
             "ms_per_encode": round(enc_ms(codec, timing_x), 4)}
        pp = wikitext2_perplexity(
            model, tok,
            cache_factory=lambda c=codec: CompressedDynamicCache(
                codec=c, config=model.config, sink_tokens=4,
                skip_layers=skip if skip else None),
            max_length=2048, stride=1024, max_windows=32)
        e["ppl"] = round(float(pp.get("ppl", pp.get("perplexity", 0.0))), 4)
        results[name] = e
        print(f"  {name}: {e}", flush=True)

    for bits in (2, 3):
        base = HayateQuantCodec(bits=bits)
        t0 = time.perf_counter()
        cals = {}
        for tag, (lr_, lc_) in {"rot": (True, False), "cb": (False, True),
                                "both": (True, True)}.items():
            cals[tag] = calibrate_hayatequant(
                HayateQuantCodec(bits=bits), kc, vc, learn_rotation=lr_,
                learn_codebook=lc_, steps=300, verbose=False)
        fit_s = time.perf_counter() - t0
        m = cals["both"].meta
        print(f"[{bits}bit] fit {fit_s:.0f}s  holdout z-dist "
              f"k {m['k_dist_before']:.5f}->{m['k_dist_after']:.5f} "
              f"v {m['v_dist_before']:.5f}->{m['v_dist_after']:.5f}", flush=True)
        art = os.path.join(OUT_DIR, f"calib_{os.path.basename(MODEL)}_{bits}bit.pt")
        cals["both"].save(art)

        bench(f"{bits}bit_TurboQuant-hadamard", TurboQuantCodec(bits=bits))
        bench(f"{bits}bit_datafree_v7", base)
        bench(f"{bits}bit_calib_rot", HayateQuantCodec(bits=bits, calibration=cals["rot"]))
        bench(f"{bits}bit_calib_cb", HayateQuantCodec(bits=bits, calibration=cals["cb"]))
        bench(f"{bits}bit_calib_both", HayateQuantCodec(bits=bits, calibration=cals["both"]))

    out = os.path.join(OUT_DIR, f"calib_v8_{os.path.basename(MODEL)}.json")
    with open(out, "w") as f:
        json.dump({"model": os.path.basename(MODEL), "results": results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
