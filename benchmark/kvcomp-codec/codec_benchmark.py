"""KV codec benchmark: HayateQuant vs HALO vs TurboQuant (quality + speed).

Main experiment for the *data-free rotated* KV codecs, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs.

* **TurboQuant (hadamard/dense)** -- the paper baseline: scalar Lloyd-Max on a
  Hadamard- or dense-rotated vector.
* **TurboQuant-QJL**              -- TurboQuant-prod baseline: scalar
  Lloyd-Max at one fewer bit plus a 1-bit Gaussian QJL residual sketch.  This
  preserves the target payload bits but stores an extra residual norm, which is
  reported explicitly.
* **HALO (auto/RMS/dense-LS)**   -- this repo's prior work: same scalar
  codebook as TurboQuant, adds Hadamard-GEMM (overhead lever) and
  role-asymmetric LS/NC scale (quality lever). Kept here as the baseline
  HayateQuant must beat, not as recommended new work (see
  ``onecomp/kvcomp/codec/hayatequant.py`` module docstring).
* **HayateQuant (bs1)**          -- superset sanity check: block_size=1 with
  HALO's scale modes must reproduce HALO-auto to numerical precision.
* **HayateQuant (auto)**         -- the proposed method: adds data-free
  block-VQ (closes TurboQuant's own stated space-filling-gain slack) and the
  Hadamard+block-dense hybrid rotation on top of HALO's scale levers. This is
  the row that must win distortion/PPL at equal bits.

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
from onecomp.kvcomp.codec import HALOCodec, TurboQuantCodec, HayateQuantCodec


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
def _relmse(codec, x, role="k"):
    """Relative MSE for one KV role.

    ``role='k'`` measures key reconstruction quality; ``role='v'`` measures
    value reconstruction quality.  Passing the same tensor as both key and
    value lets the codec apply role-specific settings (e.g. HALO's NC scale
    for keys and LS scale for values) independently -- the decoded output for
    the requested role is then compared back to the original tensor.
    """
    kd, vd = codec.decode(codec.encode(x, x))
    decoded = kd if role == "k" else vd
    return float((decoded.float() - x).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


@torch.no_grad()
def _ms_per_encode(codec, x, iters=50, warmup=10):
    """Encode-only latency (ms) and throughput (Melem/s).

    Only times ``codec.encode``; ``codec.decode`` is excluded because the
    encode step corresponds to prefill cost (the critical path for large KV
    caches) and the two have different complexity profiles.  Warmup runs
    encode+decode so that any lazy device allocations happen before timing.
    """
    dev = x.device
    for _ in range(warmup):
        codec.decode(codec.encode(x, x))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.encode(x, x)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / iters * 1e3
    melem = x.numel() / (ms * 1e-3) / 1e6
    return ms, melem


@torch.no_grad()
def _stored_bits_per_coord(codec, x):
    enc = codec.encode(x, x)
    stored = enc.meta.get("stored_bits")
    if stored is None:
        return None
    return float(stored) / float(2 * x.numel())


def _kv_asym(bits: int) -> tuple[int, int] | None:
    """Return the K(r+1)V(r-1) matched-average allocation for rate ``bits``."""
    if 2 <= bits <= 7:
        return bits + 1, bits - 1
    return None


def _codecs(bits, ablation=False):
    # Pareto / lever ablation for the HayateQuant paper (HayateQuant-auto is
    # the proposed method; TurboQuant and HALO rows are baselines):
    #   TurboQuant-hadamard : Hadamard incoherence via the WHT *butterfly* (slow).
    #   TurboQuant-dense    : dense random-orthogonal rotation (fast, lower quality).
    #   HALO-RMS            : Hadamard incoherence via a *single GEMM* (folded
    #                         signed-Hadamard) + RMS scale -> matches/beats
    #                         TurboQuant-dense speed at strictly better quality
    #                         (isolates the rotation lever).
    #   HALO-auto           : Hadamard-GEMM + least-squares scale -> lowest
    #                         reconstruction relMSE at every bit (best on the
    #                         incoherence axis; wins PPL on reasoning models).
    #   HALO-dense-LS       : dense random-orthogonal rotation (same GEMM cost as
    #                         folded-Hadamard at head_dim) + LS scale.  Dense gives
    #                         marginally better Gaussianisation than +-1 Hadamard,
    #                         which matters at the coarse low-bit codebook, so this
    #                         fixes the 2-bit PPL edge cases while keeping LS.
    codecs = {
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, rotation_mode="hadamard"),
        "TurboQuant-dense": TurboQuantCodec(bits=bits, rotation_mode="dense"),
        **({"TurboQuant-QJL": TurboQuantCodec(bits=bits - 1, value_bits=bits,
                                              rotation_mode="hadamard", use_qjl=True,
                                              qjl_roles="key")}
           if bits > 1 else {}),
        "HALO-RMS": HALOCodec(bits=bits, rotation_mode="auto", ls_scale=False),
        "HALO-auto": HALOCodec(bits=bits, rotation_mode="auto"),
        # Explicit symmetric LS (key_scale_mode="ls") so the label stays correct
        # regardless of the class default for key_scale_mode / value_scale_mode.
        "HALO-dense-LS": HALOCodec(bits=bits, rotation_mode="dense",
                                    key_scale_mode="ls", value_scale_mode="ls"),
        # Superset sanity check: block_size=1 + HALO's smart scale defaults
        # must equal HALO-auto to numerical precision (see
        # test_hayatequant_block_size_1_matches_halo_scale_modes). Included in
        # every run as a live regression guard, not just a unit test.
        "HayateQuant-bs1": HayateQuantCodec(bits=bits, block_size=1, rotation_mode="auto"),
        # The proposed method: auto block-VQ (largest block that fits the
        # max_codebook_bits budget) + folded hybrid rotation + grid-LUT O(1)
        # encoder (encoder="auto" default) + HALO's scale levers.
        "HayateQuant-auto": HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto"),
        # Direct reproduction proxies for the two closest 2026 block-vector
        # baselines under this repo's shared KV-cache runtime.  FibQuant keeps
        # HayateQuant's payload/rotation path but replaces Lloyd's random
        # initialization with the Fibonacci radial-angular construction.
        # BlockQuant keeps the block-sphere scale choices (UB for keys, BSM
        # for values) without HayateQuant's K/V frontier-rate selection.  The
        # paper reports these rows as external-method baselines, while the
        # HayateQuant-* rows isolate what remains novel after importing them.
        "FibQuant": HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto",
                                     codebook_init="fibquant"),
        "BlockQuant": HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto",
                                       key_scale_mode="ub", value_scale_mode="bsm"),
        # LUT-approximation ablation: identical codec with the exact O(K)
        # nearest-codeword search, to quantify the LUT's excess distortion on
        # real KV (expected ~0; the LUT approximates assignment only).
        "HayateQuant-exact": HayateQuantCodec(bits=bits, block_size=None,
                                              rotation_mode="auto", encoder="exact"),
        # Large-codebook operating point the O(1) LUT encoder newly makes
        # affordable: max_codebook_bits=12 lets 3-bit resolve block_size=4
        # (K=4096, previously latency-prohibitive -- 280ms/encode with the
        # exact search) for extra space-filling gain at the same stored bits.
        "HayateQuant-K12": HayateQuantCodec(bits=bits, block_size=None,
                                            rotation_mode="auto", max_codebook_bits=12),
        # Dense-rotation variant for architectures whose KV distributions
        # interact badly with the +-1 Hadamard mix (observed on Qwen-family:
        # HALO-dense-LS beat every Hadamard-based codec's PPL on R1-Qwen).
        # Same block-VQ + LUT, global Haar-dense rotation, symmetric LS scale.
        "HayateQuant-dense-LS": HayateQuantCodec(bits=bits, block_size=None,
                                                 rotation_mode="dense",
                                                 key_scale_mode="ls", value_scale_mode="ls"),
        # K/V-asymmetric bit allocation at the SAME average bits/coord as the
        # uniform row: K(r+1)V(r-1).  This generalizes the validated K4V2
        # operating point across target rates {2,3,4}.
        **(({"HayateQuant-selected":
             (HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto")
              if bits == 4 else
              HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                                key_scale_mode="ub"))})
           if _kv_asym(bits) is not None else {}),
        **(({"HayateQuant-frontier":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              key_scale_mode="ub")})
           if _kv_asym(bits) is not None else {}),
        **(({f"HayateQuant-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1])})
           if _kv_asym(bits) is not None else {}),
        **(({f"HayateQuant-FibK{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              codebook_init="fibquant")})
           if _kv_asym(bits) is not None else {}),
        # Isolate the two BlockQuant scale choices at the K/V-asymmetric point.
        **(({f"HayateQuant-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}-keyUB":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              key_scale_mode="ub")})
           if _kv_asym(bits) is not None else {}),
        **(({f"HayateQuant-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}-valueBSM":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              value_scale_mode="bsm")})
           if _kv_asym(bits) is not None else {}),
        **(({f"HayateQuant-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}-BQscale":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              key_scale_mode="ub", value_scale_mode="bsm")})
           if _kv_asym(bits) is not None else {}),
        **(({f"HayateQuant-FibK{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}-BQscale":
             HayateQuantCodec(bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1],
                              codebook_init="fibquant",
                              key_scale_mode="ub", value_scale_mode="bsm")})
           if _kv_asym(bits) is not None else {}),
    }
    if not ablation:
        return codecs
    # Lever-isolation ablations (opt-in via `ablation: true`, see
    # conf/paper_hayatequant_ablation.yaml). Each row disables exactly ONE of
    # HayateQuant-auto's levers, so (auto - variant) reads off that lever's
    # marginal contribution on real KV:
    codecs.update({
        # (a) scale lever off: block-VQ + hybrid rotation with the
        # TurboQuant-faithful plain RMS scale on both roles. (auto - noscale)
        # = the NC/ls_iter role-asymmetric scale's contribution;
        # (noscale - TurboQuant-dense) = the pure block-VQ gain with zero
        # confound from the scale.
        "HayateQuant-noscale": HayateQuantCodec(bits=bits, block_size=None,
                                                rotation_mode="auto", ls_scale=False),
        # (b) hybrid-rotation lever off: block-VQ straight on the global
        # signed-Hadamard-GEMM output, NO per-block dense factor. The block
        # codebook is fit to an isotropic law, so this mismatch should cost
        # accuracy exactly where verify_hayatequant_isotropy.py predicts.
        "HayateQuant-hadamard-bvq": HayateQuantCodec(bits=bits, block_size=None,
                                                     rotation_mode="hadamard_gemm"),
        # (c) source-law lever: exact finite-d Beta/Dirichlet block law
        # instead of the d->inf Gaussian limit (head_dim is 64-128 in
        # practice, where the two laws are already close).
        "HayateQuant-beta": HayateQuantCodec(bits=bits, block_size=None,
                                             rotation_mode="auto", dist="beta"),
        # (d) FibQuant import: same HayateQuant objective and payload format,
        # but Lloyd is warm-started from FibQuant's radial-angular codebook
        # (Beta-quantile radii + Fibonacci/Roberts directions).
        "HayateQuant-fibinit": HayateQuantCodec(bits=bits, block_size=None,
                                                rotation_mode="auto",
                                                codebook_init="fibquant"),
        # (e) BlockQuant scale terminology: BQ-BSM is HayateQuant LS; BQ-UB is
        # HayateQuant IP. This row makes the unbiased-inner-product key path
        # explicit without changing storage or the block-VQ codebook.
        "HayateQuant-BQscale": HayateQuantCodec(bits=bits, block_size=None,
                                                rotation_mode="auto",
                                                key_scale_mode="ub",
                                                value_scale_mode="bsm"),
        # (f) BlockQuant-style KV recipe: protect key outlier channels and
        # quantise values more aggressively.  This keeps the average payload
        # below the uniform 3-bit row while testing BlockQuant's core
        # key/value asymmetry in the HayateQuant runtime.
        **({"HayateQuant-BQRecipe-K3V2": HayateQuantCodec(
            bits=3, value_bits=2, block_size=None, rotation_mode="auto",
            key_scale_mode="ub", value_scale_mode="bsm",
            key_n_outlier=32, key_outlier_bits=4, value_n_outlier=0,
        )} if bits == 3 else {}),
        # (g) Combined import candidate: FibQuant codebook initialisation plus
        # the BlockQuant KV recipe.  This is the main "does the union help?"
        # row for real-model PPL/zero-shot sweeps.
        **({"HayateQuant-FibBQ-K3V2": HayateQuantCodec(
            bits=3, value_bits=2, block_size=None, rotation_mode="auto",
            codebook_init="fibquant",
            key_scale_mode="ub", value_scale_mode="bsm",
            key_n_outlier=32, key_outlier_bits=4, value_n_outlier=0,
        )} if bits == 3 else {}),
        # (h) block-size monotonicity on real KV: explicit b=2 between the
        # scalar bs1 row and the auto row (auto resolves b=4 at 2-bit,
        # b=2 at 3/4-bit under the default max_codebook_bits=9).
        "HayateQuant-bs2": HayateQuantCodec(bits=bits, block_size=2,
                                            rotation_mode="auto"),
    })
    return codecs


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
        codecs = _codecs(bits, ablation=bool(getattr(cfg, "ablation", False)))
        selected = list(getattr(cfg, "methods", []) or [])
        if selected:
            codecs = {name: codec for name, codec in codecs.items() if name in selected}
        for name, codec in codecs.items():
            entry = {}
            entry["distortion_key"] = round(_relmse(codec, ks, role="k"), 5)
            entry["distortion_value"] = round(_relmse(codec, vs, role="v"), 5)
            bpc = _stored_bits_per_coord(codec, ks)
            if bpc is not None:
                entry["stored_bits_per_coord"] = round(bpc, 4)
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
                  "| Codec | dist(K) | dist(V) | stored b/coord | ms/encode | Melem/s | PPL |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, e in methods.items():
            lines.append(f"| {name} | {e.get('distortion_key','')} | {e.get('distortion_value','')} "
                         f"| {e.get('stored_bits_per_coord','')} | {e.get('ms_per_encode','')} "
                         f"| {e.get('throughput_Melem_s','')} | {e.get('ppl','')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
