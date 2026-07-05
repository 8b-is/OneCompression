"""Round-2 enhancement candidates for HayateQuant, measured on a real model.

Three independent candidates, each with a clean attribution design:

1. **2-bit operating-point policy.** The v6 ablation showed
   HayateQuant-dense-LS beats HayateQuant-auto at 2-bit on BOTH models
   (Llama 8.955 vs 9.147; R1-Qwen-1.5B 157.4 vs 166.5) while auto wins at
   3/4-bit. dense-LS differs from auto in TWO levers at once (rotation
   hybrid->dense, scale nc/ls_iter->ls/ls); this run measures the missing
   off-diagonal cells (dense+nc/ls_iter, hybrid+ls/ls) to attribute the win
   and pick the right 2-bit default.
2. **LUT candidate count at 2-bit.** encoder=exact is 0.075 PPL better than
   the (64, T=2) LUT at 2-bit (9.072 vs 9.147) -- the only operating point
   where the LUT's excess distortion is visible in PPL. T=4 doubles the
   candidate-gather cost (still ~5x cheaper than exact) and should recover
   most of it.
3. **High-budget codebook artifacts.** 8-16x Lloyd fit budget gives -1.4%
   block distortion at (b=4, K=256) / -2.6% at (b=4, K=4096) on the source
   law (bench_rvq_gain.py) at zero runtime cost. Check it moves real-KV
   distortion / PPL in the right direction.

Usage:  srun --gres=gpu:1 python -u bench_policy_v7.py <model_path>
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
from onecomp.kvcomp.codec import hayatequant as hq_mod  # noqa: E402
from onecomp.kvcomp.codec import _vq  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else (
    "/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct")
ART_DIR = "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results/codebooks_v7"


def clear_codebook_caches():
    _vq.lloyd_vq_gaussian_block.cache_clear()
    _vq._gaussian_codebook_artifact.cache_clear()
    hq_mod._block_codebook.cache_clear()
    hq_mod._block_codebook_on.cache_clear()
    hq_mod._block_lut_on.cache_clear()


class lut_T:
    """Temporarily override the (b=4, K<=512) LUT spec's candidate count."""

    def __init__(self, t):
        self.t = t

    def __enter__(self):
        self.orig = hq_mod.lut_spec_for
        t = self.t
        def patched(block_size, k):
            if block_size == 4 and k <= 512:
                return 64, min(t, k)
            return self.orig(block_size, k)
        hq_mod.lut_spec_for = patched
        hq_mod._block_lut_on.cache_clear()
        return self

    def __exit__(self, *a):
        hq_mod.lut_spec_for = self.orig
        hq_mod._block_lut_on.cache_clear()


@torch.no_grad()
def capture_kv(model, tok, max_tokens=2048):
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
    ks, vs = capture_kv(model, tok)
    D = ks.shape[-1]
    timing_x = torch.randn(1, 8, 1024, D, device=model.device)
    print(f"model={os.path.basename(MODEL)} head_dim={D}", flush=True)

    calib_ids = tok("The history of cartography spans many centuries. " * 8,
                    return_tensors="pt", truncation=True,
                    max_length=512).input_ids.to(model.device)
    skip = set(detect_outlier_layers(model, calib_ids, ratio_thresh=50.0))
    print(f"outlier layers: {sorted(skip)}", flush=True)

    def bench(name, codec, results, lut_t=None):
        ctx = lut_T(lut_t) if lut_t else None
        if ctx:
            ctx.__enter__()
        try:
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
        finally:
            if ctx:
                ctx.__exit__()
        results[name] = e
        print(f"  {name}: {e}", flush=True)

    results = {}

    # --- phase 1: 2-bit rotation x scale attribution + LUT T --------------
    grid2 = {
        "2bit_hybrid_ncls (auto)": HayateQuantCodec(bits=2),
        "2bit_dense_ncls": HayateQuantCodec(bits=2, rotation_mode="dense"),
        "2bit_hybrid_lsls": HayateQuantCodec(bits=2, key_scale_mode="ls",
                                             value_scale_mode="ls"),
        "2bit_dense_lsls": HayateQuantCodec(bits=2, rotation_mode="dense",
                                            key_scale_mode="ls", value_scale_mode="ls"),
        "2bit_dense_lsls_exact": HayateQuantCodec(bits=2, rotation_mode="dense",
                                                  key_scale_mode="ls",
                                                  value_scale_mode="ls",
                                                  encoder="exact"),
    }
    for name, codec in grid2.items():
        bench(name, codec, results)
    bench("2bit_hybrid_ncls_T4", HayateQuantCodec(bits=2), results, lut_t=4)
    bench("2bit_dense_lsls_T4",
          HayateQuantCodec(bits=2, rotation_mode="dense", key_scale_mode="ls",
                           value_scale_mode="ls"), results, lut_t=4)

    # --- phase 2: high-budget codebook artifacts --------------------------
    from onecomp.kvcomp.codec.codebooks.fit_codebooks import fit_all
    if not os.path.isdir(ART_DIR) or not os.listdir(ART_DIR):
        print("fitting high-budget codebooks ...", flush=True)
        fit_all(out_dir=ART_DIR)
    os.environ["KVCOMP_CODEBOOK_DIR"] = ART_DIR
    clear_codebook_caches()
    bench("2bit_hybrid_ncls_bigfit", HayateQuantCodec(bits=2), results)
    bench("2bit_dense_lsls_T4_bigfit",
          HayateQuantCodec(bits=2, rotation_mode="dense", key_scale_mode="ls",
                           value_scale_mode="ls"), results, lut_t=4)
    bench("3bit_auto_bigfit", HayateQuantCodec(bits=3), results)
    bench("3bit_K12_bigfit", HayateQuantCodec(bits=3, max_codebook_bits=12), results)
    bench("4bit_auto_bigfit", HayateQuantCodec(bits=4), results)
    del os.environ["KVCOMP_CODEBOOK_DIR"]
    clear_codebook_caches()
    bench("3bit_auto_ref", HayateQuantCodec(bits=3), results)
    bench("4bit_auto_ref", HayateQuantCodec(bits=4), results)

    out = ("/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/"
           f"results/policy_v7_{os.path.basename(MODEL)}.json")
    with open(out, "w") as f:
        json.dump({"model": os.path.basename(MODEL), "results": results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
