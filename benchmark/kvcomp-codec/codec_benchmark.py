"""KV codec benchmark: HayateQuant vs HALO vs TurboQuant (quality + speed).

Main experiment for the *data-free rotated* KV codecs, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs.

* **TurboQuant-paper** -- the finite-Beta scalar Lloyd-Max MSE baseline under
  the paper's Gaussian-QR dense rotation.  The Hadamard path is named and
  reported separately as a deployment/rotation approximation.
* **TurboQuant-QJL**              -- TurboQuant-prod baseline: scalar
  Lloyd-Max at one fewer bit plus a 1-bit Gaussian QJL residual sketch.  This
  preserves the target payload bits but stores an extra residual norm, which is
  reported explicitly.
* **HALO** -- this repository's historical scalar baseline; retained only as
  an explicitly named ablation.
* **HayateQuant scalar/block matched** -- exact-encoder mechanism controls
  under the same Beta+dense+RMS-equivalent K=V=r contract.  Only block size
  changes between the two controls.
* **HQ** -- the canonical proposed recipe: data-free two-coordinate block VQ,
  fixed K/V rates r+0.5/r-0.5, folded rate-aware rotation, NC key scale, and
  iterative-LS value scale. It is one global recipe, never model-selected.

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
import hashlib
import os
import tempfile
import time

import hydra
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from onecomp.kvcomp import CompressedDynamicCache, detect_outlier_layers
from onecomp.kvcomp.eval import wikitext2_perplexity
from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    merge_run_provenance,
    run_code_commit,
)
from onecomp.kvcomp.codec import HALOCodec, TurboQuantCodec, HayateQuantCodec
from onecomp.kvcomp.codec.quality_baselines import (
    BLOCKQUANT_PAPER_METHOD,
    FIBQUANT_PAPER_METHOD,
    blockquant_quality_provenance,
    build_blockquant_paper,
    build_fibquant_paper,
    fibquant_quality_provenance,
    is_quality_only_baseline,
)
from onecomp.kvcomp.codec.hayate.recipes import (
    HQ_BLOCK_SCALE_PROXY_NAME,
    HQ_FIB_INIT_PROXY_NAME,
    build_hayate_adaptive_rate,
    build_hayate_frontier,
    canonical_hayate_method_name,
    hayate_frontier_kwargs,
    hayate_hq_kwargs,
)
from onecomp.kvcomp.codec.hayate.formal import (
    FORMAL_HAYATE_METHODS,
    hayate_formal_provenance,
    reject_legacy_hayate_result_label,
)


@torch.no_grad()
def _capture_real_kv(model, tok, max_tokens=2048, skip_layers=None):
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
    allowed = [i for i in range(n) if i not in set(skip_layers or [])]
    if not allowed:
        raise ValueError("All layers were marked as outliers; cannot capture representative KV")
    pick = [allowed[min(len(allowed) - 1, int(q * len(allowed)))] for q in (0.25, 0.5, 0.75)]
    ks = torch.cat([layers[i].keys for i in pick], dim=1).float()
    vs = torch.cat([layers[i].values for i in pick], dim=1).float()
    return ks, vs


@torch.no_grad()
def _relmse(codec, key, value, role="k"):
    """Relative MSE for one KV role.

    K/V are always encoded jointly using the real captured pair. Adaptive codecs
    choose their K/V bit allocation from joint statistics, so ``encode(x, x)``
    would measure a different algorithm than the serving path.
    """
    kd, vd = codec.decode(codec.encode(key, value))
    decoded = kd if role == "k" else vd
    target = key if role == "k" else value
    return float(
        (decoded.float() - target).pow(2).sum() / target.float().pow(2).sum().clamp_min(1e-12)
    )


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
    # BlockQuant's dynamic top-32 completion has an update-level identity
    # field in addition to per-vector codes/scales.  Preserve the codec's
    # exact integer decomposition so the formal aggregate can prove that the
    # combinadic identity field was counted rather than trusting one total.
    resolved = enc.meta.get("resolved_codec_config")
    logical_bits = resolved.get("logical_bits") if isinstance(resolved, dict) else None
    if logical_bits is not None:
        statistics["stored_bit_accounting"] = logical_bits
    return statistics


def _kv_asym(bits: int) -> tuple[int, int] | None:
    """Legacy integer K(r+1)/V(r-1) allocation for appendix ablations only.

    Canonical HQ is constructed exclusively through ``build_hayate_frontier``
    (a compatibility spelling for the fixed half-step HQ recipe). Keeping this
    helper integer-valued prevents historical ablation rows from accidentally
    passing fractional nominal ``bits`` to :class:`HayateQuantCodec`.
    """

    rate = int(bits)
    return (rate + 1, rate - 1) if 2 <= rate <= 7 else None


def _env_calibration(*names: str) -> str | None:
    for name in names:
        path = os.getenv(name)
        if path:
            return path
    return None


_RESULT_SCHEMA_VERSION = 2
_NON_SEMANTIC_CONFIG_KEYS = {"bits", "methods", "output_dir"}
HQ_INCREMENTAL_CONTROL_NAME = "HQ-incremental-control"
ADAPTIVE_HQ_FIXED_PAIR_CONTROL_NAME = "AdaptiveHQ-fixed-pair-control"


def _semantic_config(cfg: DictConfig) -> dict:
    """Fields that must agree before independently-run bit/method shards merge."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    return {key: value for key, value in raw.items() if key not in _NON_SEMANTIC_CONFIG_KEYS}


def _config_fingerprint(cfg: DictConfig) -> str:
    identity = {
        "semantic_config": _semantic_config(cfg),
        "code_commit": run_code_commit(),
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
    skip_layers: set[int],
) -> str:
    """Lock, compatibility-check, merge, and atomically replace the result JSON."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    fingerprint = _config_fingerprint(cfg)
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
            "skip_layers": sorted(skip_layers),
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
        "adaptive_hq_runtime": result.get("adaptive_hq_runtime"),
        "adaptive_hq_window_provenance": result.get("adaptive_hq_window_provenance", []),
    }


@torch.no_grad()
def _distortion_fields(codec, key, value) -> dict:
    """Measure both roles from one joint payload and label adaptive scope.

    This diagnostic is a single captured-KV update with no cache-owned causal
    history.  It must not be interpreted as the trajectory actually exercised
    by incremental PPL or generation, whose selector ledger is emitted by the
    evaluation harness separately.
    """

    encoded = codec.encode(key, value)
    decoded_key, decoded_value = codec.decode(encoded)

    def relative_mse(decoded, target):
        return float(
            (decoded.float() - target).pow(2).sum() / target.float().pow(2).sum().clamp_min(1e-12)
        )

    fields = {
        "distortion_key": round(relative_mse(decoded_key, key), 5),
        "distortion_value": round(relative_mse(decoded_value, value), 5),
        "distortion_measurement": {
            "protocol": "single_joint_captured_kv_update",
            "codec_context": None,
            "cache_owned_history": False,
            "adaptive_trajectory": False,
        },
    }
    split = encoded.meta.get("adaptive_bit_split")
    if isinstance(split, dict):
        fields["distortion_measurement"]["adaptive_static_selection"] = {
            "state_scope": split.get("state_scope"),
            "used_stats": split.get("used_stats"),
            "selected_pair": [
                split.get("key_code_bits_per_block"),
                split.get("value_code_bits_per_block"),
            ],
            "decision_reason": split.get("decision_reason"),
            "warning": "single_update_ctx_none_not_cache_trajectory",
        }
    return fields


def _codecs(bits, ablation=False, head_dim=128):
    # Pareto / lever ablation for the HayateQuant paper (HayateQuant-auto is
    # the proposed method; TurboQuant and HALO rows are baselines):
    #   TurboQuant-paper    : Algorithm-1 MSE baseline with exact finite-Beta
    #                         codebook + Gaussian-QR random rotation. This is
    #                         not the under-specified mixed-precision Table 1.
    #   TurboQuant-hadamard : fast WHT deployment approximation, kept separate
    #                         because it does not induce the exact Beta law.
    #   HALO-RMS            : Hadamard incoherence via a *single GEMM* (folded
    #                         signed-Hadamard) + RMS scale (rotation control).
    #   HALO-auto           : Hadamard-GEMM + role-specific scale (combined
    #                         research control; quality remains empirical).
    #   HALO-dense-LS       : dense random-orthogonal rotation (same GEMM cost as
    #                         folded-Hadamard at head_dim) + LS scale.  Dense gives
    #                         marginally better Gaussianisation than +-1 Hadamard,
    #                         which matters at the coarse low-bit codebook, so this
    #                         fixes the 2-bit PPL edge cases while keeping LS.
    codecs = {
        "TurboQuant-paper": TurboQuantCodec(bits=bits, dist="beta", rotation_mode="dense"),
        # Independent external quality baseline.  This exact-search research
        # codec has no packed serving kernel, so the main loop deliberately
        # omits its encode/inference timing fields.
        FIBQUANT_PAPER_METHOD: build_fibquant_paper(bits),
        **(
            {BLOCKQUANT_PAPER_METHOD: build_blockquant_paper(bits, head_dim=int(head_dim))}
            if int(bits) == 3
            else {}
        ),
        # Mechanism controls for the TurboQuant Algorithm-1-MSE source/rotation/
        # scale/rate contract.  ``encoder='exact'`` is essential: leaving the
        # block row on ``auto`` would also introduce HQ's approximate LUT
        # assignment at prefill sizes, confounding the block-VQ ablation.
        # HQ parameterises the Beta scale as RMS with a sqrt(d)-rescaled
        # codebook, whereas TQ stores L2 with a unit-sphere codebook.  They
        # therefore have identical assignments and real-arithmetic
        # reconstruction; signed-fp16 metadata can differ by its final rounding.
        "HayateQuant-scalar-matched": HayateQuantCodec(
            bits=bits,
            value_bits=bits,
            block_size=1,
            dist="beta",
            rotation_mode="dense",
            ls_scale=False,
            encoder="exact",
        ),
        "HayateQuant-blockvq-matched": HayateQuantCodec(
            bits=bits,
            value_bits=bits,
            block_size=None,
            dist="beta",
            rotation_mode="dense",
            ls_scale=False,
            encoder="exact",
        ),
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, dist="beta", rotation_mode="hadamard"),
        **(
            {
                "TurboQuant-QJL": TurboQuantCodec(
                    bits=bits - 1,
                    value_bits=bits,
                    dist="beta",
                    rotation_mode="dense",
                    use_qjl=True,
                    qjl_roles="key",
                )
            }
            if bits > 1
            else {}
        ),
        "HALO-RMS": HALOCodec(bits=bits, rotation_mode="auto", ls_scale=False),
        "HALO-auto": HALOCodec(bits=bits, rotation_mode="auto"),
        # Explicit symmetric LS (key_scale_mode="ls") so the label stays correct
        # regardless of the class default for key_scale_mode / value_scale_mode.
        "HALO-dense-LS": HALOCodec(
            bits=bits, rotation_mode="dense", key_scale_mode="ls", value_scale_mode="ls"
        ),
        # Scalar compatibility control: block_size=1 + matching HALO scale
        # settings must equal HALO-auto to numerical precision (see
        # test_hayatequant_block_size_1_matches_halo_scale_modes). Included in
        # every run as a live regression guard, not just a unit test.
        "HayateQuant-bs1": HayateQuantCodec(bits=bits, block_size=1, rotation_mode="auto"),
        # The proposed method: auto block-VQ (largest block that fits the
        # max_codebook_bits budget) + folded hybrid rotation + grid-LUT O(1)
        # encoder (encoder="auto" default) + HALO's scale levers.
        "HayateQuant-auto": HayateQuantCodec(bits=bits, block_size=None, rotation_mode="auto"),
        # HQ-internal proxies, not independent external-method reproductions.
        # The first changes only the Lloyd warm-start to a FibQuant-inspired
        # radial/angular initializer.  The second changes only HQ's scale modes
        # to the BlockQuant UB/IP and BSM/LS aliases.  New result keys retain
        # ``proxy`` explicitly so they cannot be mistaken for external rows.
        HQ_FIB_INIT_PROXY_NAME: HayateQuantCodec(
            bits=bits, block_size=None, rotation_mode="auto", codebook_init="fibquant"
        ),
        HQ_BLOCK_SCALE_PROXY_NAME: HayateQuantCodec(
            bits=bits,
            block_size=None,
            rotation_mode="auto",
            key_scale_mode="ub",
            value_scale_mode="bsm",
        ),
        # LUT-approximation ablation: identical codec with the exact O(K)
        # nearest-codeword search, to quantify the LUT's excess distortion on
        # real KV (expected ~0; the LUT approximates assignment only).
        "HayateQuant-exact": HayateQuantCodec(
            bits=bits, block_size=None, rotation_mode="auto", encoder="exact"
        ),
        # Large-codebook operating point the O(1) LUT encoder newly makes
        # affordable: max_codebook_bits=12 lets 3-bit resolve block_size=4
        # (K=4096, previously latency-prohibitive -- 280ms/encode with the
        # exact search) for extra space-filling gain at the same stored bits.
        "HayateQuant-K12": HayateQuantCodec(
            bits=bits, block_size=None, rotation_mode="auto", max_codebook_bits=12
        ),
        # Dense-rotation variant for architectures whose KV distributions
        # interact badly with the +-1 Hadamard mix (observed on Qwen-family:
        # HALO-dense-LS beat every Hadamard-based codec's PPL on R1-Qwen).
        # Same block-VQ + LUT, global Haar-dense rotation, symmetric LS scale.
        "HayateQuant-dense-LS": HayateQuantCodec(
            bits=bits,
            block_size=None,
            rotation_mode="dense",
            key_scale_mode="ls",
            value_scale_mode="ls",
        ),
        # The paper has exactly one fixed proposal recipe plus its causal
        # opt-in allocator. Development candidates are archived with their raw
        # results and deliberately absent from the production registry.
        **(({"HQ": build_hayate_frontier(bits)}) if _kv_asym(bits) is not None else {}),
        **(
            ({"AdaptiveHQ": build_hayate_adaptive_rate(bits)})
            if _kv_asym(bits) is not None
            else {}
        ),
        # Explicit development controls.  They are absent from every formal
        # method list and cannot create extra paper variants.  The first uses
        # the exact fixed HQ codec but is scored by the matched incremental PPL
        # path below.  The second retains AdaptiveHQ cache/provenance mechanics
        # while freezing its singleton pair to the matching fixed-HQ split.
        **(
            ({HQ_INCREMENTAL_CONTROL_NAME: build_hayate_frontier(bits)})
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    ADAPTIVE_HQ_FIXED_PAIR_CONTROL_NAME: build_hayate_adaptive_rate(
                        bits, adaptive_bit_margin=0
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    "HayateQuant-calib-K4V2": HayateQuantCodec(
                        bits=4,
                        value_bits=2,
                        calibration=_env_calibration("HQ_CALIB_K4V2", "HQ_CALIB_PATH"),
                    )
                }
            )
            if bits == 3 and _env_calibration("HQ_CALIB_K4V2", "HQ_CALIB_PATH")
            else {}
        ),
        **(
            (
                {
                    "HayateQuant-SBR": HayateQuantCodec(
                        **hayate_frontier_kwargs(
                            bits,
                            calibration=_env_calibration(
                                f"HQ_SBR_CALIB_{bits}BIT", "HQ_SBR_CALIB_PATH"
                            ),
                        )
                    )
                }
            )
            if _kv_asym(bits) is not None
            and _env_calibration(f"HQ_SBR_CALIB_{bits}BIT", "HQ_SBR_CALIB_PATH")
            else {}
        ),
        **(
            (
                {
                    "HayateQuant-AW": HayateQuantCodec(
                        **hayate_frontier_kwargs(
                            bits,
                            calibration=_env_calibration(
                                f"HQ_AW_CALIB_{bits}BIT", "HQ_AW_CALIB_PATH"
                            ),
                        )
                    )
                }
            )
            if _kv_asym(bits) is not None
            and _env_calibration(f"HQ_AW_CALIB_{bits}BIT", "HQ_AW_CALIB_PATH")
            else {}
        ),
        **(
            (
                {
                    "HayateQuant-Pair": HayateQuantCodec(
                        **hayate_frontier_kwargs(
                            bits,
                            calibration=_env_calibration(
                                f"HQ_PAIR_CALIB_{bits}BIT", "HQ_PAIR_CALIB_PATH"
                            ),
                        )
                    )
                }
            )
            if _kv_asym(bits) is not None
            and _env_calibration(f"HQ_PAIR_CALIB_{bits}BIT", "HQ_PAIR_CALIB_PATH")
            else {}
        ),
        # Rotation HQ keeps the fixed HQ codebook, K/V block-code budgets,
        # scales, and exact assignment.  Its artifact may contain learned
        # rotations only; the codec rejects a learned-codebook artifact on
        # this explicit-code-bit path so this row cannot quietly change the
        # fixed HQ payload recipe.
        **(
            (
                {
                    "RotationHQ": HayateQuantCodec(
                        **hayate_hq_kwargs(
                            bits,
                            calibration=_env_calibration(
                                f"HQ_ROTATION_CALIB_{bits}BIT",
                                "HQ_ROTATION_CALIB_PATH",
                            ),
                        )
                    )
                }
            )
            if _kv_asym(bits) is not None
            and _env_calibration(f"HQ_ROTATION_CALIB_{bits}BIT", "HQ_ROTATION_CALIB_PATH")
            else {}
        ),
        **(
            (
                {
                    f"HayateQuant-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1]
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    f"HQ-Fib-init-proxy-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0],
                        value_bits=_kv_asym(bits)[1],
                        codebook_init="fibquant",
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        # Isolate BlockQuant-inspired scale aliases inside the HQ runtime.
        **(
            (
                {
                    f"HQ-Block-key-scale-proxy-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0], value_bits=_kv_asym(bits)[1], key_scale_mode="ub"
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    f"HQ-Block-value-scale-proxy-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0],
                        value_bits=_kv_asym(bits)[1],
                        value_scale_mode="bsm",
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    f"HQ-Block-scale-proxy-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0],
                        value_bits=_kv_asym(bits)[1],
                        key_scale_mode="ub",
                        value_scale_mode="bsm",
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
        **(
            (
                {
                    f"HQ-Fib-Block-proxy-K{_kv_asym(bits)[0]}V{_kv_asym(bits)[1]}": HayateQuantCodec(
                        bits=_kv_asym(bits)[0],
                        value_bits=_kv_asym(bits)[1],
                        codebook_init="fibquant",
                        key_scale_mode="ub",
                        value_scale_mode="bsm",
                    )
                }
            )
            if _kv_asym(bits) is not None
            else {}
        ),
    }
    if not ablation:
        return codecs
    # Lever-isolation ablations (opt-in via `ablation: true`, see
    # conf/paper_hayatequant_ablation.yaml). Each row disables exactly ONE of
    # HayateQuant-auto's levers, so (auto - variant) reads off that lever's
    # marginal contribution on real KV:
    codecs.update(
        {
            # (a) scale lever off: block-VQ + hybrid rotation with the
            # TurboQuant-faithful plain RMS scale on both roles. (auto - noscale)
            # = the NC/ls_iter role-asymmetric scale's contribution;
            # (noscale - TurboQuant-paper) = the pure block-VQ gain with zero
            # confound from the scale.
            "HayateQuant-noscale": HayateQuantCodec(
                bits=bits, block_size=None, rotation_mode="auto", ls_scale=False
            ),
            # (b) hybrid-rotation lever off: block-VQ straight on the global
            # signed-Hadamard-GEMM output, NO per-block dense factor. The block
            # codebook is fit to an isotropic law, so this mismatch should cost
            # accuracy exactly where verify_hayatequant_isotropy.py predicts.
            "HayateQuant-hadamard-bvq": HayateQuantCodec(
                bits=bits, block_size=None, rotation_mode="hadamard_gemm"
            ),
            # (c) source-law lever: exact finite-d Beta/Dirichlet block law
            # instead of the d->inf Gaussian limit (head_dim is 64-128 in
            # practice, where the two laws are already close).
            "HayateQuant-beta": HayateQuantCodec(
                bits=bits, block_size=None, rotation_mode="auto", dist="beta"
            ),
            # (d/e) The Fib-init and Block-scale proxy rows are already present in
            # the base registry under their explicit ``*-proxy`` names.
            # (f) BlockQuant-inspired KV proxy: protect key outlier channels and
            # quantise values more aggressively.  This keeps the average payload
            # below the uniform 3-bit row while testing BlockQuant's core
            # key/value asymmetry in the HayateQuant runtime.
            **(
                {
                    "HQ-Block-recipe-proxy-K3V2": HayateQuantCodec(
                        bits=3,
                        value_bits=2,
                        block_size=None,
                        rotation_mode="auto",
                        key_scale_mode="ub",
                        value_scale_mode="bsm",
                        key_n_outlier=32,
                        key_outlier_bits=4,
                        value_n_outlier=0,
                    )
                }
                if bits == 3
                else {}
            ),
            # (g) Combined HQ proxy: FibQuant-inspired initialization plus the
            # BlockQuant-inspired KV recipe.
            **(
                {
                    "HQ-Fib-Block-recipe-proxy-K3V2": HayateQuantCodec(
                        bits=3,
                        value_bits=2,
                        block_size=None,
                        rotation_mode="auto",
                        codebook_init="fibquant",
                        key_scale_mode="ub",
                        value_scale_mode="bsm",
                        key_n_outlier=32,
                        key_outlier_bits=4,
                        value_n_outlier=0,
                    )
                }
                if bits == 3
                else {}
            ),
            # (h) block-size monotonicity on real KV: explicit b=2 between the
            # scalar bs1 row and the auto row (auto resolves b=4 at 2-bit,
            # b=2 at 3/4-bit under the default max_codebook_bits=9).
            "HayateQuant-bs2": HayateQuantCodec(bits=bits, block_size=2, rotation_mode="auto"),
        }
    )
    return codecs


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_codec")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map=cfg.device
    ).eval()

    # Detect outlier layers once and skip them uniformly for ALL codecs in the PPL
    # measurement.  Massive-activation outlier layers (common in small Qwen models)
    # destroy generation quality when quantized at any bit-width; skipping them
    # uniformly keeps the comparison fair -- the same layers are skipped for
    # TurboQuant and HALO.  The distortion metrics (KV-MSE) are measured on
    # representative non-outlier layers so they are unaffected.
    skip_layers: set = set()
    if getattr(cfg, "skip_outliers", True):
        probe_texts = list(
            getattr(cfg, "outlier_probes", [])
            or [
                "Solve: what is 17 times 23? Reason step by step.",
                "Repeat the following key exactly: QH-4821-ZX.",
                "Write a Python function that returns the n-th Fibonacci number.",
                "Explain briefly why the daytime sky appears blue.",
            ]
        )

        def _probe_ids(text):
            if getattr(tok, "chat_template", None):
                return tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )["input_ids"].to(model.device)
            return tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(
                model.device
            )

        skip_layers = set(
            detect_outlier_layers(
                model,
                [_probe_ids(text) for text in probe_texts],
                ratio_thresh=getattr(cfg, "outlier_ratio", 50.0),
                agg=getattr(cfg, "outlier_agg", "mean"),
            )
        )
        print(f"outlier layers (kept FP16): {sorted(skip_layers)}")
    extra_skip = list(getattr(cfg, "extra_skip_layers", []) or [])
    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    for item in extra_skip:
        if isinstance(item, str):
            if item == "first":
                skip_layers.add(0)
            elif item == "last" and n_layers > 0:
                skip_layers.add(n_layers - 1)
            elif item.startswith("first"):
                for i in range(int(item.removeprefix("first") or "1")):
                    skip_layers.add(i)
            elif item.startswith("last") and n_layers > 0:
                for i in range(int(item.removeprefix("last") or "1")):
                    skip_layers.add(n_layers - 1 - i)
            else:
                skip_layers.add(int(item))
        else:
            skip_layers.add(int(item))
    if extra_skip:
        print(f"extra skip layers requested: {extra_skip}; effective skip={sorted(skip_layers)}")

    ks, vs = _capture_real_kv(model, tok, cfg.capture_tokens, skip_layers=skip_layers)
    D = ks.shape[-1]
    # Independent K/V samples match the serving API and avoid benchmarking an
    # adaptive allocation policy on the artificial, perfectly-correlated (x,x).
    with torch.random.fork_rng(devices=[model.device] if model.device.type == "cuda" else []):
        torch.manual_seed(int(getattr(cfg, "sample_seed", 0)))
        timing_k = torch.randn(1, cfg.timing_heads, cfg.timing_seq, D, device=model.device)
        timing_v = torch.randn_like(timing_k)
    print(f"head_dim={D}; captured non-outlier KV {tuple(ks.shape)}")

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
        codecs = _codecs(
            bits,
            ablation=bool(getattr(cfg, "ablation", False)),
            head_dim=D,
        )
        requested = list(getattr(cfg, "methods", []) or [])
        for requested_name in requested:
            reject_legacy_hayate_result_label(str(requested_name))
        selected = list(dict.fromkeys(canonical_hayate_method_name(name) for name in requested))
        if requested != selected:
            print(
                "legacy proxy method label(s) normalized for new output: "
                f"{requested} -> {selected}",
                flush=True,
            )
        if selected:
            selected = [
                name for name in selected if name != BLOCKQUANT_PAPER_METHOD or int(bits) == 3
            ]
            missing = sorted(set(selected) - set(codecs))
            if missing:
                raise ValueError(f"Unknown codec method(s) for {bits} bit: {missing}")
            codecs = {name: codec for name, codec in codecs.items() if name in selected}
        else:
            # Matched controls are opt-in diagnostics.  Keeping them in the
            # registry makes an explicit development command reproducible,
            # but a legacy/default run must not silently grow two extra HQ
            # variants or spend GPU time on rows that are not paper methods.
            codecs = {
                name: codec
                for name, codec in codecs.items()
                if name
                not in {
                    HQ_INCREMENTAL_CONTROL_NAME,
                    ADAPTIVE_HQ_FIXED_PAIR_CONTROL_NAME,
                }
            }
        for name, codec in codecs.items():
            entry = {
                "codec_config": codec.get_kv_config(),
                "cache_policy": {
                    "sink_tokens": int(cfg.sink_tokens),
                    "recent_tokens": int(cfg.recent_tokens),
                    "skip_layers": sorted(int(layer) for layer in skip_layers),
                },
            }
            if name == HQ_INCREMENTAL_CONTROL_NAME:
                entry["development_control"] = {
                    "reference_method": "HQ",
                    "difference": "force_incremental_teacher_forcing_only",
                    "formal_row": False,
                }
            elif name == ADAPTIVE_HQ_FIXED_PAIR_CONTROL_NAME:
                entry["development_control"] = {
                    "reference_method": "AdaptiveHQ",
                    "difference": "singleton_fixed_hq_pair_no_adaptive_switch",
                    "formal_row": False,
                }
            if name == FIBQUANT_PAPER_METHOD:
                entry.update(fibquant_quality_provenance(codec))
            elif name == BLOCKQUANT_PAPER_METHOD:
                entry.update(blockquant_quality_provenance(codec))
            elif name in FORMAL_HAYATE_METHODS:
                entry.update(hayate_formal_provenance(name))
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
            if is_quality_only_baseline(name):
                entry["speed_metrics_omitted"] = (
                    f"quality-only external baseline; no packed {name.removesuffix('-paper')} "
                    "serving kernel"
                )
            else:
                ms, melem = _ms_per_encode(codec, timing_k, timing_v, cfg.timing_iters)
                entry["ms_per_encode"] = round(ms, 4)
                entry["throughput_Melem_s"] = round(melem, 1)
            if cfg.run_ppl:
                pp = wikitext2_perplexity(
                    model,
                    tok,
                    cache_factory=lambda c=codec, sl=skip_layers: CompressedDynamicCache(
                        codec=c,
                        config=model.config,
                        sink_tokens=cfg.sink_tokens,
                        recent_tokens=cfg.recent_tokens,
                        skip_layers=sl if sl else None,
                    ),
                    max_length=cfg.ppl_max_length,
                    stride=cfg.ppl_stride,
                    max_windows=cfg.ppl_windows,
                    force_incremental_control=(name == HQ_INCREMENTAL_CONTROL_NAME),
                )
                entry.update(_ppl_fields(pp))
                entry["ppl_delta_vs_fullkv"] = round(entry["ppl"] - fullkv_ppl["ppl"], 4)
            results[str(bits)][name] = entry
            print(f"  [{bits}bit] {name}: {entry}", flush=True)

    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = _save_results(cfg, model_tag, D, results, fullkv_ppl, skip_layers)
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
