"""Do the v8 levers stack? Combined = K/V-asymmetric bits (K4V2, the measured
best 3.0-avg operating point) + SpinQuant-style calibration ('both').

Rows per model: uniform K3V3 (v7 default) | K4V2 data-free | K4V2 calibrated.

Usage:  srun --gres=gpu:1 python -u bench_combo_v8.py <model_path>
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers  # noqa: E402
from onecomp.kvcomp.eval import wikitext2_perplexity  # noqa: E402
from onecomp.kvcomp.codec import HayateQuantCodec  # noqa: E402
from onecomp.kvcomp.codec.hayate.calibration import (  # noqa: E402
    calibrate_hayatequant, capture_kv_samples)
from bench_calib_v8 import calib_texts, capture_eval_kv, relmse  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct")
OUT_DIR = "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results"


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
    kc, vc = capture_kv_samples(model, tok, calib_texts(), max_tokens=2048,
                                layer_stride=4)
    print(f"captured {kc.shape[0]} vectors/role", flush=True)

    cal = calibrate_hayatequant(HayateQuantCodec(bits=4, value_bits=2), kc, vc,
                                steps=300)
    art = os.path.join(OUT_DIR, f"calib_{os.path.basename(MODEL)}_k4v2.pt")
    cal.save(art)
    print(f"calib holdout: {cal.meta}", flush=True)

    grid = {
        "K3V3_uniform_v7": HayateQuantCodec(bits=3),
        "K4V2_datafree": HayateQuantCodec(bits=4, value_bits=2),
        "K4V2_calib_both": HayateQuantCodec(bits=4, value_bits=2, calibration=cal),
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

    out = os.path.join(OUT_DIR, f"combo_v8_{os.path.basename(MODEL)}.json")
    with open(out, "w") as f:
        json.dump({"model": os.path.basename(MODEL), "results": results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
