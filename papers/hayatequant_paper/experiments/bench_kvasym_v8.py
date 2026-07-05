"""Candidate: role-asymmetric BIT allocation (K vs V) at equal average bits.

The codec already supports ``value_bits`` (K>V or K<V); the literature
(KIVI, KVQuant) suggests keys are more sensitive than values.  Measure at
matched average bits/coordinate whether asymmetric allocations beat the
uniform default:

  3.0 avg: (K3,V3) uniform  |  (K4,V2)  |  (K2,V4)
  2.5 avg: (K2,V2)+outlier64@3 (current 2.5-bit point)  |  (K3,V2)  |  (K2,V3)

Usage:  srun --gres=gpu:1 python -u bench_kvasym_v8.py <model_path>
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers  # noqa: E402
from onecomp.kvcomp.eval import wikitext2_perplexity  # noqa: E402
from onecomp.kvcomp.codec import HayateQuantCodec  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct")


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


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda:0").eval()
    ks, vs = capture_eval_kv(model, tok)
    print(f"model={os.path.basename(MODEL)}", flush=True)

    calib_ids = tok("The history of cartography spans many centuries. " * 8,
                    return_tensors="pt", truncation=True,
                    max_length=512).input_ids.to(model.device)
    skip = set(detect_outlier_layers(model, calib_ids, ratio_thresh=50.0))
    print(f"outlier layers: {sorted(skip)}", flush=True)

    grid = {
        "3.0avg_K3V3_uniform": HayateQuantCodec(bits=3),
        "3.0avg_K4V2": HayateQuantCodec(bits=4, value_bits=2),
        "3.0avg_K2V4": HayateQuantCodec(bits=2, value_bits=4),
        "2.5avg_outlier64@3_current": HayateQuantCodec(bits=2, n_outlier=64, outlier_bits=3),
        "2.5avg_K3V2": HayateQuantCodec(bits=3, value_bits=2),
        "2.5avg_K2V3": HayateQuantCodec(bits=2, value_bits=3),
    }
    results = {}
    for name, codec in grid.items():
        e = {"distortion_key": round(relmse(codec, ks, "k"), 5),
             "distortion_value": round(relmse(codec, vs, "v"), 5)}
        pp = wikitext2_perplexity(
            model, tok,
            cache_factory=lambda c=codec: CompressedDynamicCache(
                codec=c, config=model.config, sink_tokens=4,
                skip_layers=skip if skip else None),
            max_length=2048, stride=1024, max_windows=32)
        e["ppl"] = round(float(pp.get("ppl", pp.get("perplexity", 0.0))), 4)
        results[name] = e
        print(f"  {name}: {e}", flush=True)

    out = ("/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/"
           f"results/kvasym_v8_{os.path.basename(MODEL)}.json")
    with open(out, "w") as f:
        json.dump({"model": os.path.basename(MODEL), "results": results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
