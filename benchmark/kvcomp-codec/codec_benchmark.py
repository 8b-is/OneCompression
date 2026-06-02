"""KV codec benchmark: HALO vs TurboQuant (speed-quality trade-off).

Main experiment for the *data-free rotated-scalar* KV codecs, the analogue of
the weight-side `benchmark/<model>-<recipe>/` runs.  HALO and TurboQuant share
the same quality core (random orthogonal rotation + data-free Gaussian
Lloyd-Max), so the comparison is a **speed-quality trade-off** at *equal*
quality:

* **TurboQuant (hadamard)** -- the shipped default rotation (WHT).
* **TurboQuant (dense)**     -- dense random-orthogonal matmul rotation.
* **HALO (auto)**            -- proposed: picks the cheapest orthogonal transform
  per head dim (dense GEMM at realistic D; WHT only for very large power-of-two
  D) and skips preconditioning math when it is off (the default).  Quality is
  identical to TurboQuant; the win is **overhead**.

Metrics (per bit-width in {2,3,4}):
* distortion -- relative reconstruction MSE on captured real KV;
* WikiText-2 perplexity under `CompressedDynamicCache(codec)`;
* overhead -- ms/encode and throughput (Melem/s) at the model's true head dim.

Copyright 2025-2026 Fujitsu Ltd.

Usage (GPU node):
    srun --gres=gpu:1 .../python -u codec_benchmark.py \
        model_path=/lustre/.../models/llama-3.1-8b-instruct
"""

from __future__ import annotations

import json
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers
from onecomp.kvcomp.eval import wikitext2_perplexity
from onecomp.kvcomp.codec import HALOCodec, TurboQuantCodec


@torch.no_grad()
def _capture_real_kv(model, tok, max_tokens=2048):
    text = ("The history of cartography spans many centuries of patient observation. "
            "Merchants traded spices along routes that crossed deserts and mountains. ") * 64
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(model.device)
    out = model(ids, use_cache=True)
    layers = out.past_key_values.layers
    n = len(layers)
    pick = [n // 4, n // 2, 3 * n // 4]
    ks = torch.cat([layers[i].keys for i in pick], dim=1).float()
    vs = torch.cat([layers[i].values for i in pick], dim=1).float()
    return ks, vs


@torch.no_grad()
def _relmse(codec, x):
    kq, _ = codec.decode(codec.encode(x, x))
    return float((kq.float() - x).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


@torch.no_grad()
def _ms_per_encode(codec, x, iters=50, warmup=10):
    for _ in range(warmup):
        codec.decode(codec.encode(x, x))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.decode(codec.encode(x, x))
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    melem = x.numel() / (ms * 1e-3) / 1e6
    return ms, melem


def _codecs(bits):
    return {
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, rotation_mode="hadamard"),
        "TurboQuant-dense": TurboQuantCodec(bits=bits, rotation_mode="dense"),
        "HALO-auto": HALOCodec(bits=bits, rotation_mode="auto"),
    }


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_codec")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map=cfg.device).eval()

    ks, vs = _capture_real_kv(model, tok, cfg.capture_tokens)
    D = ks.shape[-1]
    timing_x = torch.randn(1, cfg.timing_heads, cfg.timing_seq, D, device=model.device)
    print(f"head_dim={D}; captured KV {tuple(ks.shape)}")

    # Detect outlier layers once and skip them uniformly for ALL codecs in the PPL
    # measurement.  Massive-activation outlier layers (common in small Qwen models)
    # destroy generation quality when quantized at any bit-width; skipping them
    # uniformly keeps the comparison fair -- the same layers are skipped for
    # TurboQuant and HALO.  The distortion metrics (KV-MSE) are measured on
    # representative non-outlier layers so they are unaffected.
    skip_layers: set = set()
    if getattr(cfg, "skip_outliers", True):
        calib_text = ("The history of cartography spans many centuries. " * 8)
        calib_ids = tok(calib_text, return_tensors="pt",
                        truncation=True, max_length=512).input_ids.to(model.device)
        skip_layers = set(detect_outlier_layers(model, calib_ids,
                                               ratio_thresh=getattr(cfg, "outlier_ratio", 50.0)))
        print(f"outlier layers (kept FP16): {sorted(skip_layers)}")

    results = {}
    for bits in list(cfg.bits):
        results[bits] = {}
        for name, codec in _codecs(bits).items():
            entry = {}
            entry["distortion_key"] = round(_relmse(codec, ks), 5)
            entry["distortion_value"] = round(_relmse(codec, vs), 5)
            ms, melem = _ms_per_encode(codec, timing_x, cfg.timing_iters)
            entry["ms_per_encode"] = round(ms, 4)
            entry["throughput_Melem_s"] = round(melem, 1)
            if cfg.run_ppl:
                pp = wikitext2_perplexity(
                    model, tok, cache_factory=lambda c=codec, sl=skip_layers: CompressedDynamicCache(
                        codec=c, config=model.config, sink_tokens=cfg.sink_tokens,
                        skip_layers=sl if sl else None),
                    max_length=cfg.ppl_max_length, stride=cfg.ppl_stride, max_windows=cfg.ppl_windows)
                entry["ppl"] = round(float(pp.get("ppl", pp.get("perplexity", 0.0))), 4)
            results[bits][name] = entry
            print(f"  [{bits}bit] {name}: {entry}", flush=True)

    os.makedirs(cfg.output_dir, exist_ok=True)
    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    with open(out_json, "w") as f:
        json.dump({"model": model_tag, "head_dim": D,
                   "config": OmegaConf.to_container(cfg, resolve=True), "results": results}, f, indent=2)
    print(f"\nWrote {out_json}")
    print(_markdown(model_tag, D, results))


def _markdown(model_tag, D, results):
    lines = [f"\n### {model_tag} (head_dim={D})", ""]
    for bits, methods in results.items():
        lines += [f"\n**{bits}-bit**", "",
                  "| Codec | dist(K) | dist(V) | ms/encode | Melem/s | PPL |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, e in methods.items():
            lines.append(f"| {name} | {e.get('distortion_key','')} | {e.get('distortion_value','')} "
                         f"| {e.get('ms_per_encode','')} | {e.get('throughput_Melem_s','')} | {e.get('ppl','')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
