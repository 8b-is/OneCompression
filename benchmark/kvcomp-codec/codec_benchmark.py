"""KV codec benchmark: canonical HayateQuant versus TurboQuant.

Both rows use one global data-free recipe at rates 2, 3, and 4. Generated
results are written only to the caller-selected output directory.

Metrics (per bit-width in {2,3,4}):
* distortion -- relative reconstruction MSE on captured real KV;
* WikiText-2 perplexity under `CompressedDynamicCache(codec)`;
* overhead -- ms/encode and throughput (Melem/s) at the model's true head dim.

Copyright 2025-2026 Fujitsu Ltd.

Usage (GPU node):
    srun --gres=gpu:1 "$PYTHON" -u codec_benchmark.py \
        model_path="$MODELS_ROOT/llama-3.1-8b-instruct" \
        output_dir="$OUTPUT_ROOT/ppl/llama31-8b"
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
from pathlib import Path

import hydra
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from onecomp.kvcomp import CompressedDynamicCache
from onecomp.kvcomp.eval import wikitext2_perplexity
from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    merge_run_provenance,
    run_code_commit,
    run_code_revision,
    validate_model_provenance,
)
from onecomp.kvcomp.codec import TurboQuantCodec
from onecomp.kvcomp.codec.hayate.recipes import (
    HAYATEQUANT_NAME,
    HAYATEQUANT_RATES,
    build_hayatequant,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _capture_real_kv(model, tok, max_tokens=2048):
    text = (
        "The history of cartography spans many centuries of patient observation. "
        "Merchants traded spices along routes that crossed deserts and mountains. "
    ) * 64
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(
        model.device
    )
    out = model(ids, use_cache=True)
    layers = out.past_key_values.layers
    n = len(layers)
    pick = [min(n - 1, int(q * n)) for q in (0.25, 0.5, 0.75)]
    ks = torch.cat([layers[i].keys for i in pick], dim=1).float()
    vs = torch.cat([layers[i].values for i in pick], dim=1).float()
    return ks, vs


@torch.no_grad()
def _ms_per_encode(codec, key, value, iters=50, warmup=10):
    """Encode-only latency (ms) and throughput (Melem/s).

    Only times ``codec.encode``; ``codec.decode`` is excluded because the
    encode step corresponds to prefill cost (the critical path for large KV
    caches) and the two have different complexity profiles.  Warmup runs
    encode+decode so that any lazy device allocations happen before timing.
    """
    dev = key.device
    for _ in range(warmup):
        codec.decode(codec.encode(key, value))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.encode(key, value)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    ms = (time.perf_counter() - t0) / iters * 1e3
    # One encode consumes both K and V tensors. Report total processed KV
    # coordinates, not one role (the old value was exactly 2x too small).
    melem = (key.numel() + value.numel()) / (ms * 1e-3) / 1e6
    return ms, melem


@torch.no_grad()
def _stored_rate_statistics(codec, key, value):
    enc = codec.encode(key, value)
    stored = enc.meta.get("stored_bits")
    if stored is None:
        return None
    coordinates = int(key.numel() + value.numel())
    statistics = {
        "stored_bits": int(stored),
        "stored_coordinates": coordinates,
        "stored_bits_per_coord": float(stored) / float(coordinates),
    }
    return statistics


_RESULT_SCHEMA_VERSION = 2
_NON_SEMANTIC_CONFIG_KEYS = {"bits", "methods", "output_dir"}
_DEFAULT_CODEC_METHODS = frozenset({"turboquant", HAYATEQUANT_NAME})


def hayatequant_provenance() -> dict:
    """Benchmark metadata for the one canonical HayateQuant method."""

    return {
        "method_role": "proposal",
        "external_quality_baseline": False,
        "proposal_method": True,
        "proposal_family": "HayateQuant",
        "supported_rates": list(HAYATEQUANT_RATES),
        "allocation_scope": "globally_fixed_key_heavy_code_bits",
        "adopted_kv_rates_bits_per_coordinate": {
            "2": {"key": 3, "value": 1},
            "3": {"key": 4, "value": 2},
            "4": {"key": 5, "value": 3},
        },
        "role_scales": {"key": "nc", "value": "ls_iter"},
    }


def _semantic_config(cfg: DictConfig) -> dict:
    """Fields that must agree before independently-run bit/method shards merge."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    return {key: value for key, value in raw.items() if key not in _NON_SEMANTIC_CONFIG_KEYS}


def _config_fingerprint(cfg: DictConfig, model_attestation: dict | None) -> str:
    identity = {
        "semantic_config": _semantic_config(cfg),
        "code_revision": run_code_revision(),
        "model_checkpoint_fingerprint": (
            model_attestation.get("checkpoint", {}).get("fingerprint_sha256")
            if model_attestation is not None
            else None
        ),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_existing(doc: dict, expected: str, path: str) -> None:
    actual = doc.get("run_fingerprint")
    if doc.get("schema_version") != _RESULT_SCHEMA_VERSION or actual is None:
        raise RuntimeError(
            f"Refusing to merge legacy result file {path}. Move it aside or use a new output_dir."
        )
    if actual != expected:
        raise RuntimeError(
            f"Refusing to merge a different benchmark configuration in {path}: "
            f"expected {expected[:12]}, found {actual[:12]}."
        )


def _merge_results(on_disk: dict, current: dict) -> dict:
    """Merge bit shards, refusing a same-name collision with a changed codec."""
    merged = {str(bits): dict(methods) for bits, methods in on_disk.items()}
    for bits, methods in current.items():
        bit_key = str(bits)
        dest = merged.setdefault(bit_key, {})
        for name, entry in methods.items():
            old = dest.get(name)
            if old is not None and old.get("codec_config") != entry.get("codec_config"):
                raise RuntimeError(
                    f"Refusing to overwrite {bit_key}bit/{name} with a different codec config."
                )
            dest[name] = entry
    return merged


def _save_results(
    cfg: DictConfig,
    model_tag: str,
    head_dim: int,
    results: dict,
    fullkv_ppl: dict | None,
    model_attestation: dict | None,
) -> str:
    """Lock, compatibility-check, merge, and atomically replace the result JSON."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    fingerprint = _config_fingerprint(cfg, model_attestation)
    with FileLock(out_json + ".lock"):
        try:
            with open(out_json, encoding="utf-8") as stream:
                old_doc = json.load(stream)
            _validate_existing(old_doc, fingerprint, out_json)
            old_results = old_doc.get("results", {})
            old_fullkv = old_doc.get("fullkv_ppl")
            old_provenance = old_doc.get("run_provenance", [])
        except FileNotFoundError:
            old_results, old_fullkv, old_provenance = {}, None, []
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Existing result file is not valid JSON: {out_json}") from exc

        doc = {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "run_fingerprint": fingerprint,
            "model": model_tag,
            "head_dim": int(head_dim),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "semantic_config": _semantic_config(cfg),
            "code_commit": run_code_commit(),
            "code_revision": run_code_revision(),
            "model_config_sha256": _sha256_file(Path(cfg.model_path) / "config.json"),
            "model_checkpoint_fingerprint": (
                model_attestation.get("checkpoint", {}).get("fingerprint_sha256")
                if model_attestation is not None
                else None
            ),
            "cache_contract": {
                "sink_tokens": 0,
                "recent_tokens": 0,
                "skip_layers": [],
            },
            "run_provenance": merge_run_provenance(old_provenance, load_run_provenance()),
            "fullkv_ppl": fullkv_ppl if fullkv_ppl is not None else old_fullkv,
            "results": _merge_results(old_results, results),
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_json)}.", suffix=".tmp", dir=cfg.output_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(doc, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, out_json)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return out_json


def _ppl_fields(result: dict) -> dict:
    """Keep the score together with enough provenance to audit its denominator."""
    return {
        # Keep JSON-double precision. Human-facing rendering rounds only after
        # the aggregate computes within-checkpoint NLL contrasts.
        "ppl": float(result["ppl"]),
        "ppl_nll": float(result["nll"]),
        "ppl_nll_sum": float(result["nll_sum"]),
        "ppl_nll_accumulation": result["nll_accumulation"],
        "ppl_n_tokens": int(result["n_tokens"]),
        "ppl_n_windows": int(result["n_windows"]),
        "ppl_traversed_tokens": int(result["traversed_tokens"]),
        "ppl_scored_target_index_set": result["scored_target_index_set"],
        "ppl_scored_target_ranges": result["scored_target_ranges"],
        "ppl_scored_target_coverage": result["scored_target_coverage"],
        "ppl_n_skipped_nonfinite": int(result["n_skipped_nonfinite"]),
        "ppl_complete_text_eval": bool(result["complete_text_eval"]),
        "ppl_window_cache_lifetime": result["window_cache_lifetime"],
        "ppl_cross_window_tensor_retention": result["cross_window_tensor_retention"],
        "ppl_max_windows": result["max_windows"],
        "ppl_source": result["source"],
        "ppl_dataset_revision": result.get("dataset_revision"),
        "ppl_text_sha256": result.get("text_sha256"),
        "ppl_compression_ratio": float(result["compression_ratio"]),
        "ppl_compression_aggregation": result["compression_aggregation"],
        "ppl_compression_fp16_bits": result["compression_fp16_bits"],
        "ppl_compression_stored_bits": result["compression_stored_bits"],
        "ppl_teacher_forced_scoring_protocol": result["teacher_forced_scoring_protocol"],
        "ppl_teacher_forced_temporal_dependency": result["teacher_forced_temporal_dependency"],
        "ppl_force_incremental_control": bool(result["force_incremental_control"]),
    }


@torch.no_grad()
def _distortion_fields(codec, key, value) -> dict:
    """Measure role-wise relative MSE from one joint payload."""

    encoded = codec.encode(key, value)
    decoded_key, decoded_value = codec.decode(encoded)

    def relative_mse(decoded, target):
        return float(
            (decoded.float() - target).pow(2).sum() / target.float().pow(2).sum().clamp_min(1e-12)
        )

    fields = {
        "distortion_key": round(relative_mse(decoded_key, key), 5),
        "distortion_value": round(relative_mse(decoded_value, value), 5),
        "distortion_measurement": {"protocol": "single_joint_captured_kv_update"},
    }
    return fields


def _codecs(bits):
    """Return the two methods in the HayateQuant paper comparison.

    The baseline is the paper's symmetric TurboQuant-MSE Hadamard recipe,
    rather than the general-purpose class's dense-rotation default.
    """

    return {
        "turboquant": TurboQuantCodec(bits=int(bits), dist="beta", rotation_mode="hadamard"),
        HAYATEQUANT_NAME: build_hayatequant(int(bits)),
    }


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_codec")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model_attestation = validate_model_provenance(Path(cfg.model_path))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map=cfg.device
    ).eval()

    ks, vs = _capture_real_kv(model, tok, cfg.capture_tokens)
    D = ks.shape[-1]
    # Independent K/V samples match the serving API.
    with torch.random.fork_rng(devices=[model.device] if model.device.type == "cuda" else []):
        torch.manual_seed(int(getattr(cfg, "sample_seed", 0)))
        timing_k = torch.randn(1, cfg.timing_heads, cfg.timing_seq, D, device=model.device)
        timing_v = torch.randn_like(timing_k)
    print(f"head_dim={D}; captured KV {tuple(ks.shape)}")

    fullkv_ppl = None
    if cfg.run_ppl:
        fullkv_ppl = _ppl_fields(
            wikitext2_perplexity(
                model,
                tok,
                cache_factory=lambda: DynamicCache(config=model.config),
                max_length=cfg.ppl_max_length,
                stride=cfg.ppl_stride,
                max_windows=cfg.ppl_windows,
            )
        )
        print(f"  [FullKV] {fullkv_ppl}", flush=True)

    results = {}
    for bits in list(cfg.bits):
        results[str(bits)] = {}
        codecs = _codecs(bits)
        requested = list(dict.fromkeys(getattr(cfg, "methods", []) or []))
        selected = requested or list(_DEFAULT_CODEC_METHODS)
        missing = sorted(set(selected) - set(codecs))
        if missing:
            raise ValueError(
                f"Unknown method(s) {missing}; use only 'turboquant' and 'hayatequant'"
            )
        codecs = {name: codecs[name] for name in selected}
        for name, codec in codecs.items():
            entry = {
                "codec_config": codec.get_kv_config(),
                "cache_policy": {
                    "sink_tokens": 0,
                    "recent_tokens": 0,
                    "skip_layers": [],
                },
            }
            if name == HAYATEQUANT_NAME:
                entry.update(hayatequant_provenance())
            resolved_config = getattr(codec, "get_resolved_kv_config", None)
            if callable(resolved_config):
                entry["resolved_codec_config"] = resolved_config(D)
            entry.update(_distortion_fields(codec, ks, vs))
            rate_stats = _stored_rate_statistics(codec, ks, vs)
            if rate_stats is not None:
                # Preserve the exact integer numerator and denominator.  A
                # four-decimal rate can hide metadata/bitmap differences and
                # make a nominally iso-rate comparison pass by rounding.
                entry.update(rate_stats)
            ms, melem = _ms_per_encode(codec, timing_k, timing_v, cfg.timing_iters)
            entry["ms_per_encode"] = round(ms, 4)
            entry["throughput_Melem_s"] = round(melem, 1)
            if cfg.run_ppl:
                pp = wikitext2_perplexity(
                    model,
                    tok,
                    cache_factory=lambda c=codec: CompressedDynamicCache(
                        codec=c,
                        config=model.config,
                        sink_tokens=0,
                        recent_tokens=0,
                    ),
                    max_length=cfg.ppl_max_length,
                    stride=cfg.ppl_stride,
                    max_windows=cfg.ppl_windows,
                    force_incremental_control=False,
                )
                entry.update(_ppl_fields(pp))
                entry["ppl_delta_vs_fullkv"] = round(entry["ppl"] - fullkv_ppl["ppl"], 4)
            results[str(bits)][name] = entry
            print(f"  [{bits}bit] {name}: {entry}", flush=True)

    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = _save_results(cfg, model_tag, D, results, fullkv_ppl, model_attestation)
    print(f"\nWrote {out_json}")
    print(_markdown(model_tag, D, results, fullkv_ppl))


def _markdown(model_tag, D, results, fullkv_ppl=None):
    lines = [f"\n### {model_tag} (head_dim={D})", ""]
    if fullkv_ppl:
        scope = "complete test split" if fullkv_ppl["ppl_complete_text_eval"] else "capped subset"
        lines += [
            f"FullKV WikiText-2 PPL: {fullkv_ppl['ppl']} ({scope}, "
            f"n={fullkv_ppl['ppl_n_tokens']} tokens)",
            "",
        ]
    for bits, methods in results.items():
        lines += [
            f"\n**{bits}-bit**",
            "",
            "| Codec | dist(K) | dist(V) | stored b/coord | ms/encode | Melem/s | PPL | delta FullKV |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, e in methods.items():
            lines.append(
                f"| {name} | {e.get('distortion_key','')} | {e.get('distortion_value','')} "
                f"| {e.get('stored_bits_per_coord','')} | {e.get('ms_per_encode','')} "
                f"| {e.get('throughput_Melem_s','')} | {e.get('ppl','')} "
                f"| {e.get('ppl_delta_vs_fullkv','')} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
