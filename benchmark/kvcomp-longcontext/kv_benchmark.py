"""KV-cache compression benchmark: long-context retrieval + reasoning.

Main experiment for the KV-compression methods, the analogue of the weight-side
``benchmark/<model>-<recipe>/quant_benchmark.py`` runs (GPTQ / JointQ). It
compares the two *families* the design memo contrasts and the proposed method
on common prompts, recording realised memory for Pareto analysis. Pairwise
iso-memory is a separate, explicitly validated protocol.

* **FullKV**            -- fp16 upper bound.
* **Uniform-INT{4,2}**  -- uniform per-token quantization (the baseline the memo
  argues is less memory-efficient than eviction).
* **StreamingLLM**      -- sink + recent eviction (the "evict" regime).
* **ThinKV-TBQ**        -- thought-adaptive quantization only; the HF path does
  not implement the paper's TBE eviction and is labelled non-comparable.
* **PM-KVQ**            -- progressive *dynamic*-precision quantization (paper baseline).
* **TaperKV**           -- proposed: static age-graded precision, *no eviction*.

Metrics (each method scored with its own ``cache_factory``):
* needle-in-a-haystack retrieval accuracy over a depth x length grid (the axis
  where eviction structurally fails) -- :func:`needle_in_haystack`;
* GSM8K pass@1 (reasoning non-degradation) -- :func:`gsm8k_pass_at_1`;
* WikiText-2 perplexity (sanity) -- :func:`wikitext2_perplexity`;
* realised memory fraction (stored / fp16) per method.

Results are written to ``output_dir`` as JSON + a Markdown table.

Copyright 2025-2026 Fujitsu Ltd.

Usage (GPU node):
    srun --gres=gpu:1 .../python -u kv_benchmark.py \
        model_path=/lustre/.../models/DeepSeek-R1-Distill-Qwen-7B
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from typing import Any

import hydra
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from onecomp.kvcomp import (
    PMKVQPolicy,
    PMKVQCache,
    TaperKVPolicy,
    TaperKVCache,
    TurboTaperPolicy,
    TurboTaperCache,
    ThinKVCache,
    StreamingLLMEvictCache,
    CompressedDynamicCache,
    build_codec,
    auto_configure_thinkv,
    detect_outlier_layers,
)
from onecomp.kvcomp.recipe.thinkv import ThinKVPolicy
from onecomp.kvcomp.eval import (
    needle_in_haystack,
    ruler_multikey,
    gsm8k_pass_at_1,
    math500_pass_at_1,
    aime_pass_at_1,
    wikitext2_perplexity,
    longbench_e,
    livecodebench_pass_at_1,
    ruler_v1,
    longcodeqa,
    aggregate_longcodeqa,
)
from onecomp.kvcomp.eval.longcontext import (
    NEEDLE_PROTOCOL_LABEL,
    NEEDLE_PROTOCOL_SCOPE,
    NEEDLE_PROTOCOL_VERSION,
    aggregate_cache_compression_records,
    proportion_stats,
)
from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    merge_run_provenance,
    run_code_commit,
)
from onecomp.kvcomp.codec.quality_baselines import (
    BLOCKQUANT_PAPER_METHOD,
    FIBQUANT_PAPER_METHOD,
    blockquant_quality_provenance,
    build_blockquant_paper,
    build_fibquant_paper,
    fibquant_quality_provenance,
)
from onecomp.kvcomp.codec.hayate.formal import (
    FORMAL_HAYATE_METHODS,
    hayate_formal_provenance,
    reject_legacy_hayate_result_label,
)
from onecomp.kvcomp.codec.hayate.recipes import (
    build_hayate_adaptive_rate,
    build_hayate_hq,
)


def _strict_json_load(stream, *, label: str):
    def reject_constant(value: str):
        raise RuntimeError(f"Non-standard JSON numeric constant {value!r} in {label}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON object key {key!r} in {label}")
            result[key] = value
        return result

    return json.load(
        stream,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _strict_json_loads(text: str, *, label: str):
    def reject_constant(value: str):
        raise RuntimeError(f"Non-standard JSON numeric constant {value!r} in {label}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON object key {key!r} in {label}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _require_formal_generation_accounting(method: str, entry: dict[str, Any]) -> None:
    """Fail before persistence if a formal generation axis lacks exact ledgers."""

    formal_methods = {
        "FullKV",
        "TurboQuant-paper",
        "FibQuant-paper",
        "BlockQuant-paper",
        "HQ",
        "AdaptiveHQ",
    }
    if method not in formal_methods:
        return
    for axis in ("ruler_v1", "longbench", "math500", "longcodeqa"):
        if axis not in entry:
            continue
        result = entry[axis]
        statistics = (
            result.get("compression_sufficient_statistics") if isinstance(result, dict) else None
        )
        if not isinstance(statistics, dict):
            raise RuntimeError(
                f"Formal {axis}/{method} omitted exact compression sufficient statistics"
            )
        if method == "AdaptiveHQ":
            runtime = statistics.get("adaptive_hq_runtime")
            if (
                not isinstance(runtime, dict)
                or runtime.get("selector_incremental_stored_bits_total") != 0
                or not isinstance(runtime.get("split_histogram"), dict)
                or not runtime["split_histogram"]
                or not isinstance(runtime.get("selector_carrier_version"), str)
            ):
                raise RuntimeError(f"Formal {axis}/AdaptiveHQ carrier provenance is incomplete")


def _model_head_dim(config) -> int:
    """Resolve the text attention head dimension without loading a layer."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    value = getattr(text, "head_dim", None)
    if isinstance(value, int) and value > 0:
        return value
    hidden = getattr(text, "hidden_size", None)
    heads = getattr(text, "num_attention_heads", None)
    if (
        isinstance(hidden, int)
        and isinstance(heads, int)
        and hidden > 0
        and heads > 0
        and hidden % heads == 0
    ):
        return hidden // heads
    raise ValueError("Model config has no unambiguous attention head dimension")


def build_methods(cfg: DictConfig, model, tok):
    """Return ``{name: dict(make, evict, reason, capture)}``.

    ``make(ctx)`` builds the needle cache for a given requested filler budget.
    The runner reports exact realised memory and therefore supports a Pareto
    comparison. A single fixed eviction window is not automatically iso-memory
    with every quantizer; matching requires a separately pre-registered pair.
    ``reason`` builds the
    generate-based GSM8K / PPL evals (the reasoning harness explicitly drives
    eviction after prefill and every decode step). ``capture``
    flags methods (ThinKV) whose decode-time policy needs attention sparsity.
    """
    mc = model.config
    sink = cfg.sink_tokens
    recent = cfg.recent_tokens
    target = cfg.target_mem  # requested stored/fp16 operating point for StreamingLLM only
    methods = {}
    model_head_dim = _model_head_dim(mc)

    # Detect massive-activation (outlier) layers ONCE and skip them in *every*
    # quant method.  Quantizing these layers with low bits breaks generation on
    # reasoning models (K23); skipping them is the standard StreamingLLM/KIVI
    # robustness every KV-quant method relies on, so it must be applied uniformly
    # for a fair comparison (not just to ThinKV).
    if cfg.get("skip_outliers", True):
        # Several diverse probes (math / retrieval / code / narrative) instead of a
        # single short prompt: the massive-activation layer set is then aggregated
        # (mean ratio) across probes, so it is reproducible rather than hostage to
        # one prompt's tokenisation (benchmark review B-6).
        _probe_texts = cfg.get(
            "outlier_probes",
            [
                "Solve: what is 17 times 23? Reason step by step.",
                "Repeat the following key exactly: the passcode is QH-4821-ZX. What is the passcode?",
                "Write a Python function that returns the n-th Fibonacci number.",
                "Summarize in one sentence why the sky appears blue during the day.",
            ],
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

        probes = [_probe_ids(text) for text in _probe_texts]
        skip = detect_outlier_layers(
            model,
            probes,
            ratio_thresh=cfg.get("outlier_ratio", 50.0),
            agg=cfg.get("outlier_agg", "mean"),
        )
    else:
        skip = []
    print(f"  [outlier] skip_layers (all quant methods) = {skip}", flush=True)

    methods["FullKV"] = dict(
        make=lambda ctx: DynamicCache(config=mc),
        evict=False,
        reason=lambda: DynamicCache(config=mc),
    )

    for b in cfg.uniform_bits:
        # Per-token axis for both K and V: a well-defined "uniform per-token"
        # baseline whose online scales are faithful under the incremental cache
        # (per-channel keys would need block granularity; that is the KIVI codec).
        codec = build_codec(f"int{b}", skip_layers=skip, key_axis="token", value_axis="token")
        methods[f"Uniform-INT{b}"] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
        )

    # StreamingLLM's explicit operating point. It is intentionally not called
    # iso-memory with TaperKV/PM-KVQ/etc.: exact payload and metadata fractions
    # are measured from the same prompts and reported as a Pareto frontier.
    def _evict(ctx):
        rec = max(recent, int(target * ctx) - sink)
        return StreamingLLMEvictCache(sink=sink, recent=rec, config=mc)

    methods["StreamingLLM"] = dict(
        make=_evict,
        evict=True,
        reason=lambda: StreamingLLMEvictCache(sink=sink, recent=recent, config=mc),
    )

    # Faithful ThinKV: outlier-layer skip + calibrated L*/thresholds (the 08
    # setup), engaged via attention capture during decode.  Without this ThinKV
    # degenerates to all-Reasoning NVFP4 over *every* layer (incl. massive-
    # activation outliers), which breaks reasoning models -- see lessons.md K23.
    if cfg.thinkv.get("auto_configure", True):
        thinp = auto_configure_thinkv(
            model,
            tok,
            precision=dict(cfg.thinkv.precision),
            sink_tokens=sink,
            max_new_tokens=cfg.thinkv.get("calib_max_new", 200),
            skip_layers=skip,
        )
        thinp.group_size = cfg.thinkv.group_size
        print(
            f"  [ThinKV] skip_layers={thinp.skip_layers} L*={thinp.calib_layers} "
            f"thresholds={tuple(round(t,3) for t in thinp.thresholds)}",
            flush=True,
        )
        capture = True
    else:
        thinp = ThinKVPolicy(
            sink_tokens=sink,
            group_size=cfg.thinkv.group_size,
            precision=dict(cfg.thinkv.precision),
        )
        capture = False
    # ``capture`` (attention sparsity for thought classification) is only used on
    # the *short* GSM8K prompts.  output_attentions at long context retains every
    # layer's full attention matrix and OOMs (lessons.md K25); needle/RULER on an
    # instruct model do not need thought classification (ThinKV quant-only already
    # retrieves), so long-context capture is left off.
    capture_long = capture and cfg.thinkv.get("capture_long", False)
    methods["ThinKV"] = dict(
        make=lambda ctx: ThinKVCache(thinp, config=mc, sink_tokens=sink),
        evict=False,
        reason=lambda: ThinKVCache(thinp, config=mc, sink_tokens=sink),
        capture=capture,
        capture_long=capture_long,
        method_definition="ThinKV-TBQ (HF research path; TBE eviction disabled)",
        paper_comparable=False,
    )

    pmp = PMKVQPolicy(
        key_bits=cfg.pmkvq.key_bits,
        value_bits=cfg.pmkvq.value_bits,
        final_bits=cfg.pmkvq.final_bits,
        key_final_bits=cfg.pmkvq.key_final_bits,
        recent_window=recent,
        ladder_span=cfg.pmkvq.ladder_span,
        skip_layers=skip,
    )
    # Faithful PM-KVQ: run the offline block-wise integer-program allocation
    # (§3.2) and the positional-interpolation channel reparameterization (§3.3).
    # Without these the comparison understates PM-KVQ by ~3-6% on AIME (the
    # "simplified block allocation" caveat); enabling them is paper-faithful.
    if cfg.pmkvq.get("block_allocation", False) or cfg.pmkvq.get("channel_reparam", False):
        from onecomp.kvcomp import calibrate_pmkvq

        calibrate_pmkvq(
            model,
            tok,
            pmp,
            block_allocation=cfg.pmkvq.get("block_allocation", False),
            channel_reparam=cfg.pmkvq.get("channel_reparam", False),
            bit_options=list(cfg.pmkvq.get("bit_options", [2, 4, 8])),
            avg_bit_budget=cfg.pmkvq.get("avg_bit_budget", None),
            pos_scale=cfg.pmkvq.get("pos_scale", 4),
        )
        print(
            f"  [PM-KVQ] block_final_bits set for {len(pmp.block_final_bits)} layers; "
            f"channel_reparam on {len(pmp.key_channel_scale)} layers",
            flush=True,
        )
    pm_paper_comparable = bool(
        cfg.pmkvq.get("block_allocation", False) and cfg.pmkvq.get("channel_reparam", False)
    )
    methods["PM-KVQ"] = dict(
        make=lambda ctx: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride),
        evict=False,
        reason=lambda: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride),
        method_definition=(
            "PM-KVQ (offline allocation + channel reparameterization)"
            if pm_paper_comparable
            else "PM-KVQ online precision path (offline allocation/reparameterization disabled)"
        ),
        paper_comparable=pm_paper_comparable,
    )

    tap = TaperKVPolicy(
        key_bits=cfg.taperkv.key_bits,
        value_bits=cfg.taperkv.value_bits,
        b_min=cfg.taperkv.b_min,
        b_min_value=cfg.taperkv.b_min_value,
        sink_tokens=sink,
        recent_tokens=recent,
        skip_layers=skip,
    )
    methods["TaperKV"] = dict(
        make=lambda ctx: TaperKVCache(tap, config=mc, requant_stride=cfg.requant_stride),
        evict=False,
        reason=lambda: TaperKVCache(tap, config=mc, requant_stride=cfg.requant_stride),
    )

    # TurboTaper (proposed, ThinKV+TurboQuant superset): TaperKV's age ladder
    # quantized with TurboQuant's rotated Lloyd-Max codec (MSE-optimal floor).
    # Defaults track the TaperKV config so the two are iso-schedule comparable.
    tt = cfg.get("turbotaper", {}) or {}
    ttp = TurboTaperPolicy(
        key_bits=tt.get("key_bits", cfg.taperkv.key_bits),
        value_bits=tt.get("value_bits", cfg.taperkv.value_bits),
        b_min=tt.get("b_min", cfg.taperkv.b_min),
        b_min_value=tt.get("b_min_value", cfg.taperkv.b_min_value),
        sink_tokens=sink,
        recent_tokens=recent,
        skip_layers=skip,
        dist=tt.get("dist", "gaussian"),
        rotation_mode=tt.get("rotation_mode", "hadamard"),
        seed=tt.get("seed", 0),
        n_outlier_floor=tt.get("n_outlier_floor", 0),
    )
    methods["TurboTaper"] = dict(
        make=lambda ctx: TurboTaperCache(ttp, config=mc, requant_stride=cfg.requant_stride),
        evict=False,
        reason=lambda: TurboTaperCache(ttp, config=mc, requant_stride=cfg.requant_stride),
    )

    # Optional TurboTaper sweep: same pattern as taperkv_sweep, so multiple
    # TurboTaper operating points (e.g. 3-bit / 4-bit average) can be compared
    # in one run -- needed for the Pareto curve in the recipe paper.
    for variant in cfg.get("turbotaper_sweep", None) or []:
        v = dict(variant)
        label = v.pop("label")
        vp = TurboTaperPolicy(
            key_bits=v.get("key_bits", tt.get("key_bits", cfg.taperkv.key_bits)),
            value_bits=v.get("value_bits", tt.get("value_bits", cfg.taperkv.value_bits)),
            b_min=v.get("b_min", tt.get("b_min", cfg.taperkv.b_min)),
            b_min_value=v.get("b_min_value", tt.get("b_min_value", cfg.taperkv.b_min_value)),
            sink_tokens=sink,
            recent_tokens=v.get("recent_tokens", recent),
            skip_layers=skip,
            dist=v.get("dist", tt.get("dist", "gaussian")),
            rotation_mode=v.get("rotation_mode", tt.get("rotation_mode", "hadamard")),
            seed=v.get("seed", tt.get("seed", 0)),
            n_outlier_floor=v.get("n_outlier_floor", tt.get("n_outlier_floor", 0)),
        )
        methods[label] = dict(
            make=lambda ctx, p=vp: TurboTaperCache(
                p, config=mc, requant_stride=cfg.requant_stride
            ),
            evict=False,
            reason=lambda p=vp: TurboTaperCache(p, config=mc, requant_stride=cfg.requant_stride),
        )

    # Optional TaperKV memory sweep: register one method per (label, overrides)
    # so a single run traces TaperKV's accuracy-vs-memory Pareto curve and we can
    # show it dominates PM-KVQ / eviction *at matched memory*, not just at one
    # operating point.
    for variant in cfg.get("taperkv_sweep", None) or []:
        v = dict(variant)
        label = v.pop("label")
        vp = TaperKVPolicy(
            key_bits=v.get("key_bits", cfg.taperkv.key_bits),
            value_bits=v.get("value_bits", cfg.taperkv.value_bits),
            b_min=v.get("b_min", cfg.taperkv.b_min),
            b_min_value=v.get("b_min_value", cfg.taperkv.b_min_value),
            sink_tokens=sink,
            recent_tokens=v.get("recent_tokens", recent),
            skip_layers=skip,
        )
        methods[label] = dict(
            make=lambda ctx, p=vp: TaperKVCache(p, config=mc, requant_stride=cfg.requant_stride),
            evict=False,
            reason=lambda p=vp: TaperKVCache(p, config=mc, requant_stride=cfg.requant_stride),
        )

    # TurboQuant (arXiv:2504.19874): random rotation + Lloyd-Max scalar quant.
    # Pure 2/3/4-bit plus matched-rate outlier splits. At head_dim=128 the
    # paper's 32 ch @ 3 + 96 @ 2 is 2.25 payload bits plus 0.25 bits for the two
    # fp16 subset norms, hence 2.5 stored bits/coordinate (likewise 3.5).
    # Each variant is a dict {label, bits, [n_outlier, outlier_bits, ...]}.
    for variant in cfg.get("turboquant_variants", None) or []:
        v = dict(variant)
        label = v.pop("label")
        codec = build_codec("turboquant", skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            codec=codec,
        )

    # FibQuant v1 is an independently implemented, exact-search *quality*
    # baseline.  Keep it on a dedicated fail-closed path so a generic codec
    # override cannot silently change the paper recipe while retaining the
    # formal label.  There is deliberately no FibQuant entry in any speed
    # harness.
    for variant in cfg.get("fibquant_variants", None) or []:
        v = dict(variant)
        label = v.pop("label", None)
        bits = v.pop("bits", None)
        if label != FIBQUANT_PAPER_METHOD:
            raise ValueError(
                f"fibquant_variants label must be {FIBQUANT_PAPER_METHOD!r}, got {label!r}"
            )
        if v:
            raise ValueError(
                "FibQuant-paper only accepts the canonical {label,bits} quality recipe; "
                f"unexpected override(s): {sorted(v)}"
            )
        codec = build_fibquant_paper(bits, skip_layers=skip)
        provenance = fibquant_quality_provenance(codec)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            codec=codec,
            **provenance,
        )

    # BlockQuant v1 publishes one Llama-3.1 head_dim=128 KV operating point.
    # The exact-search research codec uses a versioned dynamic top-L2 identity
    # completion and exact identity-bit accounting; it is never a speed row.
    for variant in cfg.get("blockquant_variants", None) or []:
        v = dict(variant)
        label = v.pop("label", None)
        bits = v.pop("bits", None)
        if label != BLOCKQUANT_PAPER_METHOD:
            raise ValueError(
                f"blockquant_variants label must be {BLOCKQUANT_PAPER_METHOD!r}, got {label!r}"
            )
        if v:
            raise ValueError(
                "BlockQuant-paper only accepts the canonical {label,bits} quality recipe; "
                f"unexpected override(s): {sorted(v)}"
            )
        codec = build_blockquant_paper(
            bits,
            head_dim=model_head_dim,
            skip_layers=skip,
        )
        provenance = blockquant_quality_provenance(codec)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            codec=codec,
            **provenance,
        )

    # KIVI (arXiv:2402.02750): per-channel key + per-token value asymmetric quant.
    # The headline baseline TurboQuant compares against; bits via ``kivi_bits``.
    for b in cfg.get("kivi_bits", None) or []:
        codec = build_codec("kivi", skip_layers=skip, bits=b)
        methods[f"KIVI-INT{b}"] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
        )

    # HALO (fast online-whitened randomized-Hadamard Lloyd-Max codec). Each entry
    # is a codec kwargs dict {label, bits, [value_bits, rotation_mode, ...]}.
    for variant in cfg.get("halo_variants", None) or []:
        v = dict(variant)
        label = v.pop("label")
        codec = build_codec("halo", skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            codec=codec,
        )

    # HayateQuant is a distinct block-VQ/hybrid-rotation/role-scale recipe.
    # Its block_size=1 setting is a scalar compatibility control; that does not
    # make the full codec an implementation of every external scalar/vector
    # baseline whose individual design ideas can be expressed as ablations.
    # Each entry is a codec kwargs dict {label, bits, [block_size, ...]}, same
    # shape as ``halo_variants`` -- ``block_size`` defaults to ``None`` (auto)
    # so the paper number is the shipped, recommended operating point.
    for variant in cfg.get("hayatequant_variants", None) or []:
        v = dict(variant)
        label = v.pop("label")
        reject_legacy_hayate_result_label(str(label))
        if label in FORMAL_HAYATE_METHODS:
            bits = v.pop("bits", None)
            if v:
                raise ValueError(
                    f"{label} formal variants accept only {{label,bits}}; "
                    f"unexpected override(s): {sorted(v)}"
                )
            builder = build_hayate_hq if label == "HQ" else build_hayate_adaptive_rate
            codec = builder(bits, skip_layers=skip)
            provenance = hayate_formal_provenance(label)
        else:
            codec = build_codec("hayatequant", skip_layers=skip, **v)
            provenance = {}
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            codec=codec,
            **provenance,
        )

    # Generic codec methods: benchmark ANY registered codec by name so no
    # implemented codec is unreachable from the harness (fp8, nvfp4, ternary,
    # int8, ...).  Each entry is {label, codec, **codec_kwargs}, e.g.
    # ``{label: FP8, codec: fp8}`` or ``{label: NVFP4, codec: nvfp4}``.
    for variant in cfg.get("codec_variants", None) or []:
        v = dict(variant)
        label = v.pop("label")
        name = v.pop("codec")
        codec = build_codec(name, skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(
                codec=c, config=mc, sink_tokens=sink, recent_tokens=recent
            ),
        )

    return methods


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_longcontext")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=dtype,
        device_map=cfg.device,
        attn_implementation=cfg.attn_implementation,
    ).eval()

    methods = build_methods(cfg, model, tok)
    only = set(cfg.methods) if cfg.get("methods") else None
    # Fail-fast on typos / unregistered method names (K43): catch before model
    # load + outlier scan would otherwise be wasted, and prevents mid-run crashes.
    if only is not None:
        unknown = only - set(methods)
        if unknown:
            raise KeyError(
                f"Unknown method(s) in cfg.methods: {sorted(unknown)}. "
                f"Available: {sorted(methods)}.  Add the codec to the registry "
                f"and/or the corresponding sweep section in the YAML."
            )

    run_fingerprint = _config_fingerprint(cfg)

    # Resume: load existing results so we can (a) skip already-done methods in a
    # normal run (if cfg.resume=true), or (b) merge a targeted re-run of specific
    # methods (cfg.methods=[...]) into the existing JSON without losing other results.
    existing_results: dict = {}
    model_head_dim = _model_head_dim(model.config)
    model_tag_pre = os.path.basename(str(cfg.model_path).rstrip("/"))
    existing_json = os.path.join(cfg.output_dir, f"results_{model_tag_pre}.json")
    if os.path.exists(existing_json):
        try:
            with open(existing_json, encoding="utf-8") as stream:
                existing_doc = _strict_json_load(stream, label=existing_json)
            _validate_result_fingerprint(existing_doc, run_fingerprint, existing_json)
            existing_results = existing_doc.get("results", {})
            loaded_n = len(existing_results)
            if only is not None:
                print(
                    f"[resume] loaded {loaded_n} existing method(s); "
                    f"will re-run and merge: {sorted(only)}",
                    flush=True,
                )
            elif cfg.get("resume", False):
                print(
                    f"[resume] loaded {loaded_n} existing method(s); "
                    f"will skip already-done methods",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read existing result file {existing_json}: {exc}"
            ) from exc

    results = dict(existing_results)  # start from existing; overwritten per method below
    for name, spec in methods.items():
        if only is not None and name not in only:
            continue
        # Skip already-done methods when cfg.resume=true and no explicit filter.
        if only is None and cfg.get("resume", False) and name in existing_results:
            print(f"\n===== {name} [SKIP – already in results] =====", flush=True)
            continue
        print(f"\n===== {name} =====", flush=True)
        codec = spec.get("codec")
        codec_skip_layers = (
            sorted(int(layer) for layer in codec.skip_layers) if codec is not None else []
        )
        entry = {
            "cache_policy": {
                "sink_tokens": int(cfg.sink_tokens),
                "recent_tokens": int(cfg.recent_tokens),
                "skip_layers": codec_skip_layers,
            },
        }
        if codec is not None:
            entry["codec_config"] = codec.get_kv_config()
            resolved_config = getattr(codec, "get_resolved_kv_config", None)
            if callable(resolved_config):
                entry["resolved_codec_config"] = resolved_config(model_head_dim)
        if spec.get("method_definition"):
            entry["method_definition"] = spec["method_definition"]
            entry["paper_comparable"] = bool(spec.get("paper_comparable", True))
        for provenance_key in (
            "method_role",
            "proposal_method",
            "speed_comparison",
            "primary_source",
            "nominal_payload_bits_per_coordinate",
            "external_quality_baseline",
            "paper_reproduction",
            "source_tex_sha256",
            "formal_rate_slot",
            "paper_declared_key_bits_per_coordinate",
            "paper_declared_value_payload_bits_per_coordinate",
            "top32_completion",
            "remainder_completion",
            "codebook_completion",
            "random_seed_protocol",
            "proposal_family",
            "proposal_variant",
            "paper_comparable",
            "formal_rate_slots",
            "allocation_scope",
        ):
            if provenance_key in spec:
                entry[provenance_key] = spec[provenance_key]
        cap = spec.get("capture", False)  # short reasoning prompts
        cap_long = spec.get("capture_long", False)  # long context (OOM-prone)

        # Needle: one pass per requested filler budget. All methods receive the
        # same deterministic prompt grid; outputs retain rendered token counts,
        # gold/prediction pairs, prompt hashes, and per-trial cache accounting
        # for regrading and memory verification.
        # Skipped on reasoning models where FullKV itself fails retrieval (K24).
        if cfg.get("run_needle", True):
            by_len, by_depth_counts, n_tot, hit_tot = {}, {}, 0, 0
            needle_compression_records = []
            needle_grid = []
            for ctx in list(cfg.context_lens):
                nr = needle_in_haystack(
                    model,
                    tok,
                    cache_factory=lambda c=ctx: spec["make"](c),
                    context_lens=[ctx],
                    depths=list(cfg.depths),
                    seeds=cfg.seeds,
                    max_new_tokens=cfg.max_new_needle,
                    evict=spec["evict"],
                    attn_capture=cap_long,
                )
                by_len[ctx] = round(nr.accuracy, 4)
                needle_compression_records.extend(nr.compression_records)
                needle_grid.extend(nr.grid)
                for row in nr.grid:
                    depth = str(row["requested_depth"])
                    bucket = by_depth_counts.setdefault(depth, [0, 0])
                    bucket[0] += int(bool(row["hit"]))
                    bucket[1] += 1
                hit_tot += nr.hits
                n_tot += nr.n
            # Pooled accuracy + binomial SE / Wilson 95% CI over all (ctx,depth,seed)
            # trials, so the paper table carries error bars (benchmark review B-4).
            acc_all = hit_tot / max(1, n_tot)
            se, lo, hi = proportion_stats(hit_tot, n_tot)
            needle_compression, needle_mem_fraction, needle_mem_aggregation = (
                aggregate_cache_compression_records(needle_compression_records)
            )
            entry["needle"] = {
                "accuracy": round(acc_all, 4),
                "accuracy_stderr": round(se, 4),
                "accuracy_ci95": [round(lo, 4), round(hi, 4)],
                "hits": hit_tot,
                "mem_fraction": round(needle_mem_fraction, 4),
                "mem_fraction_aggregation": needle_mem_aggregation,
                "compression_sufficient_statistics": needle_compression,
                "by_length": by_len,
                "by_depth": {
                    depth: round(hits / max(1, count), 4)
                    for depth, (hits, count) in by_depth_counts.items()
                },
                "benchmark": NEEDLE_PROTOCOL_LABEL,
                "protocol_version": NEEDLE_PROTOCOL_VERSION,
                "official_reference": False,
                "protocol_scope": NEEDLE_PROTOCOL_SCOPE,
                "comparison_scope": (
                    "common deterministic prompts; Pareto over realised memory; "
                    "not an implicit pairwise iso-memory comparison"
                ),
                "grid": needle_grid,
                "n": n_tot,
            }
            print("  needle:", entry["needle"], flush=True)

        if cfg.get("run_ruler", False):
            by_len_r, n_r, hit_r = {}, 0, 0
            multikey_compression_records = []
            multikey_grid = []
            for ctx in list(cfg.context_lens):
                rr2 = ruler_multikey(
                    model,
                    tok,
                    cache_factory=lambda c=ctx: spec["make"](c),
                    context_lens=[ctx],
                    num_keys=cfg.get("ruler_keys", 4),
                    seeds=cfg.get("ruler_seeds", 5),
                    max_new_tokens=cfg.max_new_needle,
                    evict=spec["evict"],
                    attn_capture=cap_long,
                )
                by_len_r[ctx] = round(rr2.accuracy, 4)
                multikey_compression_records.extend(rr2.compression_records)
                multikey_grid.extend(rr2.grid)
                hit_r += rr2.hits
                n_r += rr2.n
            acc_all_r = hit_r / max(1, n_r)
            se_r, lo_r, hi_r = proportion_stats(hit_r, n_r)
            multikey_compression, multikey_mem_fraction, multikey_mem_aggregation = (
                aggregate_cache_compression_records(multikey_compression_records)
            )
            entry["synthetic_multikey"] = {
                "accuracy": round(acc_all_r, 4),
                "accuracy_stderr": round(se_r, 4),
                "accuracy_ci95": [round(lo_r, 4), round(hi_r, 4)],
                "hits": hit_r,
                "mem_fraction": round(multikey_mem_fraction, 4),
                "mem_fraction_aggregation": multikey_mem_aggregation,
                "compression_sufficient_statistics": multikey_compression,
                "by_length": by_len_r,
                "benchmark": "deterministic synthetic multi-key retrieval v2",
                "protocol_version": 2,
                "official_reference": False,
                "protocol_scope": (
                    "local diagnostic on common deterministic prompts; not NVIDIA RULER"
                ),
                "grid": multikey_grid,
                "n": n_r,
            }
            print("  synthetic_multikey:", entry["synthetic_multikey"], flush=True)

        # Pinned NeMo-Skills RULER-v1 100-sample/task protocol. This is
        # intentionally independent of the
        # historical `run_ruler` flag above, which remains a small local
        # synthetic diagnostic. Generated tokenizer-specific data are consumed
        # verbatim and any prompt truncation/context overflow is fatal. The
        # `official_ruler_*` config names are compatibility keys, not permission
        # to relabel this 100/task adapter as canonical NVIDIA 500/task.
        if cfg.get("run_official_ruler_v1", False):
            if not cfg.get("official_ruler_data_root", None):
                raise ValueError("run_official_ruler_v1=true requires official_ruler_data_root")
            ruler_context = cfg.get("official_ruler_max_context", None) or int(
                os.path.basename(str(cfg.official_ruler_data_root).rstrip("/"))
            )
            rv1 = ruler_v1(
                model,
                tok,
                data_root=cfg.official_ruler_data_root,
                manifest_path=cfg.get("official_ruler_manifest_path", None),
                cache_factory=(
                    spec["reason"] if spec["evict"] else lambda: spec["make"](ruler_context)
                ),
                tasks=list(cfg.get("official_ruler_tasks", [])) or None,
                max_context_tokens=ruler_context,
                start=cfg.get("official_ruler_start", 0),
                limit_per_task=cfg.get("official_ruler_limit", None),
                evict=spec["evict"],
                attn_capture=cap_long,
            )
            entry["ruler_v1"] = rv1.as_dict()
            print(
                "  ruler_v1:",
                {
                    "benchmark": entry["ruler_v1"]["benchmark"],
                    "score": entry["ruler_v1"]["score"],
                    "n": entry["ruler_v1"]["n"],
                    "samples_per_task": entry["ruler_v1"]["samples_per_task"],
                    "protocol_complete": entry["ruler_v1"]["protocol_complete"],
                    "canonical_nvidia_full_eval": entry["ruler_v1"]["canonical_nvidia_full_eval"],
                    "per_task": entry["ruler_v1"]["per_task"],
                },
                flush=True,
            )

        # LongBench-E: end-to-end long-context generation (TurboQuant Table 1).
        # cache_factory ignores ctx (codec/graded caches are length-agnostic);
        # the eviction baseline uses its reason() cache (recent window).
        if cfg.get("run_longbench", False):
            lb = longbench_e(
                model,
                tok,
                cache_factory=(
                    spec["reason"]
                    if spec["evict"]
                    else lambda: spec["make"](cfg.get("longbench_max_length", 32768))
                ),
                tasks=list(cfg.get("longbench_tasks", [])) or None,
                max_length=cfg.get("longbench_max_length", 32768),
                limit_per_task=cfg.get("longbench_limit", None),
                evict=spec["evict"],
                attn_capture=cap_long,
            )
            entry["longbench"] = lb.as_dict()
            print("  longbench:", entry["longbench"], flush=True)

        # LongCodeBench LongCodeQA: final-filtered release, verbatim raw prompt,
        # released A--D parser, and per-bracket accuracy.  The fixed 16-token
        # greedy decode is our explicitly recorded paired-cache adaptation: the
        # upstream wrappers do not define one uniform generation configuration.
        # Unlike the legacy functional-code smoke below, this is the 443-row
        # long-repository dataset and never executes model code.
        if cfg.get("run_longcodeqa", False):
            lqa_results = []
            for bracket in list(cfg.get("longcodeqa_brackets", ["32K"])):
                lqa_results.append(
                    longcodeqa(
                        model,
                        tok,
                        data_root=cfg.longcodeqa_data_root,
                        bracket=str(bracket),
                        cache_factory=(
                            spec["reason"]
                            if spec["evict"]
                            else lambda: spec["make"](
                                cfg.get("longcodeqa_max_context", None)
                                or model.config.max_position_embeddings
                            )
                        ),
                        max_context_tokens=cfg.get("longcodeqa_max_context", None),
                        max_new_tokens=cfg.get("longcodeqa_max_new", 16),
                        start=cfg.get("longcodeqa_start", 0),
                        limit=cfg.get("longcodeqa_limit", None),
                        evict=spec["evict"],
                        attn_capture=cap_long,
                        dataset_revision=cfg.get(
                            "longcodeqa_revision",
                            "989d5eff750d65a72c522e47f8f745ef2e22906b",
                        ),
                    )
                )
            entry["longcodeqa"] = aggregate_longcodeqa(lqa_results)
            print("  longcodeqa:", entry["longcodeqa"], flush=True)

        if cfg.run_gsm8k:
            rr = gsm8k_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                limit=cfg.gsm8k_limit,
                max_new_tokens=cfg.max_new_reason,
                attn_capture=cap,
                sample_seed=cfg.get("sample_seed", 0),
            )
            entry["gsm8k"] = rr.as_dict()
            print("  gsm8k:", rr.as_dict(), flush=True)

        if cfg.get("run_math", False):
            rm = math500_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                start=cfg.get("math_start", 0),
                limit=cfg.get("math_limit", 50),
                max_new_tokens=cfg.get("max_new_math", 1024),
                attn_capture=cap,
                sample_seed=cfg.get("sample_seed", 0),
                grader=cfg.get("math_grader", "math_verify"),
                math_verify_version=cfg.get("math_verify_version", "0.9.0"),
                dataset_revision=cfg.get(
                    "math500_revision", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
                ),
            )
            entry["math500"] = rm.as_dict()
            print("  math500:", rm.as_dict(), flush=True)

        # AIME-2024 / AIME-2025: paper-level competition math evaluation.
        # ThinKV paper uses k=8 samples; PM-KVQ paper uses k=16 samples.
        # Default: k=8 (ThinKV setting) with temperature=0.6, top_p=0.95.
        for aime_year in cfg.get("aime_years", []):
            ra = aime_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                year=int(aime_year),
                limit=cfg.get("aime_limit", 30),
                k=cfg.get("aime_k", 8),
                max_new_tokens=cfg.get("max_new_aime", 8192),
                attn_capture=cap,
                sample_seed=cfg.get("sample_seed", 0),
            )
            entry[f"aime{aime_year}"] = ra.as_dict()
            print(f"  aime{aime_year}:", ra.as_dict(), flush=True)

        # HumanEval-shaped functional-code smoke axis. This is deliberately not
        # named LiveCodeBench: official LCB requires its APPS checker/runner.
        if cfg.get("run_livecodebench", False):
            problems = _load_lcb_problems(cfg)
            rc = livecodebench_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                dataset=cfg.get("lcb_dataset", "livecodebench/code_generation_lite"),
                split=cfg.get("lcb_split", "test"),
                limit=cfg.get("lcb_limit", 50),
                problems=problems,
                k=cfg.get("lcb_k", 1),
                max_new_tokens=cfg.get("max_new_code", 4096),
                timeout=cfg.get("lcb_timeout", 6.0),
                attn_capture=cap,
                sample_seed=cfg.get("sample_seed", 0),
            )
            entry["functional_code_smoke"] = rc.as_dict()
            print("  functional_code_smoke:", rc.as_dict(), flush=True)

        if cfg.run_ppl:
            pp = wikitext2_perplexity(
                model,
                tok,
                cache_factory=spec["reason"],
                max_length=cfg.ppl_max_length,
                stride=cfg.ppl_stride,
                max_windows=cfg.ppl_windows,
            )
            # Preserve provenance/completeness as typed values. The previous
            # numeric-only conversion silently dropped the dataset source and
            # turned booleans into 0.0/1.0, making capped PPL look official.
            entry["ppl"] = {
                key: (round(float(value), 4) if isinstance(value, float) else value)
                for key, value in pp.items()
            }
            print("  ppl:", entry["ppl"], flush=True)

        # Targeted re-run (cfg.methods=[X,Y,...]): merge the newly computed
        # benchmarks INTO the existing method entry so that previously computed
        # benchmarks (e.g. AIME data from a prior AIME-only run) are preserved.
        # Full runs (no methods filter) replace the entry as before.
        if only is not None and name in existing_results:
            merged = dict(existing_results[name])
            merged.update(entry)
            results[name] = merged
        else:
            results[name] = entry
        _require_formal_generation_accounting(name, entry)
        # Persist after every method so a long multi-method sweep is crash-safe
        # (a failure late in the run still leaves the completed methods on disk).
        # Persist only the method touched by this process. Passing a complete
        # stale in-memory snapshot can overwrite fields a concurrent shard added
        # after this process started.
        _save(cfg, {name: entry}, run_fingerprint)

    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = _save(cfg, {}, run_fingerprint)
    with open(out_json, encoding="utf-8") as stream:
        results = _strict_json_load(stream, label=out_json)["results"]
    print(f"\nWrote {out_json}")
    print(_markdown_table(model_tag, results))


def _load_lcb_problems(cfg):
    """Local HumanEval-shaped problems ({prompt, entry_point, test}) or None.

    Returning ``None`` lets ``livecodebench_pass_at_1`` load the HF dataset itself.
    A local file is the reproducible path for the functional subset on an offline
    GPU node.
    """
    path = cfg.get("lcb_problems_path", None)
    if not path:
        return None
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(_strict_json_loads(line, label=str(path)))
    return problems


_RESULT_SCHEMA_VERSION = 2
_NON_SEMANTIC_CONFIG_KEYS = {"methods", "output_dir", "resume"}


def _semantic_config(cfg) -> dict:
    """Return config fields that must match before result shards may be merged."""
    raw = OmegaConf.to_container(cfg, resolve=True)
    return {
        key: value
        for key, value in raw.items()
        if key not in _NON_SEMANTIC_CONFIG_KEYS and not key.startswith("run_")
    }


def _config_fingerprint(cfg) -> str:
    identity = {
        "semantic_config": _semantic_config(cfg),
        "code_commit": run_code_commit(),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_result_fingerprint(doc: dict, expected: str, path: str) -> None:
    actual = doc.get("run_fingerprint")
    if doc.get("schema_version") != _RESULT_SCHEMA_VERSION or actual is None:
        raise RuntimeError(
            f"Refusing to merge legacy result file {path}. Move it aside or choose a new output_dir."
        )
    if actual != expected:
        raise RuntimeError(
            f"Refusing to merge results from a different configuration in {path}: "
            f"expected {expected[:12]}, found {actual[:12]}. Use a new output_dir."
        )


def _save(cfg, results, run_fingerprint=None) -> str:
    """Persist results to disk, merging with any on-disk state written by
    concurrent jobs since this process started.  The in-memory ``results``
    (freshly computed by *this* job) take precedence over on-disk entries for
    the same method key; all other on-disk methods are preserved unchanged.
    This prevents a race condition where two jobs share the same output dir and
    the slower job overwrites the faster job's already-written results.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    fingerprint = run_fingerprint or _config_fingerprint(cfg)
    lock_path = out_json + ".lock"
    with FileLock(lock_path):
        try:
            with open(out_json, encoding="utf-8") as stream:
                on_disk_doc = _strict_json_load(stream, label=out_json)
            _validate_result_fingerprint(on_disk_doc, fingerprint, out_json)
            on_disk = on_disk_doc.get("results", {})
            old_provenance = on_disk_doc.get("run_provenance", [])
        except FileNotFoundError:
            on_disk = {}
            old_provenance = []
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Existing result file is not valid JSON: {out_json}") from exc
        # Merge at benchmark-field granularity. Parallel shards commonly write
        # different axes (PPL vs AIME) for the same method; replacing the whole
        # method entry loses the first shard. Fresh values win only for fields
        # this process actually recomputed.
        merged = {name: dict(entry) for name, entry in on_disk.items()}
        for name, fresh in results.items():
            entry = dict(merged.get(name, {}))
            entry.update(fresh)
            merged[name] = entry
        doc = {
            "schema_version": _RESULT_SCHEMA_VERSION,
            "run_fingerprint": fingerprint,
            "model": model_tag,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "semantic_config": _semantic_config(cfg),
            "code_commit": run_code_commit(),
            "run_provenance": merge_run_provenance(old_provenance, load_run_provenance()),
            "results": merged,
        }
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(out_json)}.", suffix=".tmp", dir=cfg.output_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(doc, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, out_json)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return out_json


def _fmt_ci(d, key="pass@1", ci_key="pass@1_ci95"):
    """Render ``value ±half-width`` from a result dict's metric + its 95% CI."""
    if not d or key not in d:
        return ""
    val = d.get(key, "")
    ci = d.get(ci_key)
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        half = (float(ci[1]) - float(ci[0])) / 2.0
        return f"{val}\u00b1{half:.3f}"
    return str(val)


def _markdown_table(model_tag, results) -> str:
    # Detect which AIME years are present
    aime_years = sorted(
        {k.replace("aime", "") for e in results.values() for k in e if k.startswith("aime")}
    )
    aime_cols = [f"AIME-{y} p@1" for y in aime_years]
    has_lb = any("longbench" in e for e in results.values())
    lb_cols = ["LongBench-E"] if has_lb else []
    has_ruler_v1 = any("ruler_v1" in e for e in results.values())
    ruler_v1_cols = ["NeMo RULER v1 100/task"] if has_ruler_v1 else []
    has_lcb = any("functional_code_smoke" in e for e in results.values())
    lcb_cols = ["Functional-code smoke p@1"] if has_lcb else []
    header = (
        ["Method", "Needle acc", "Synthetic multi-key acc"]
        + ruler_v1_cols
        + lb_cols
        + ["Mem", "GSM8K p@1", "MATH p@1"]
        + aime_cols
        + lcb_cols
        + ["PPL"]
    )
    lines = [
        f"\n### {model_tag}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for name, e in results.items():
        nd = e.get("needle", {})
        ru = e.get("synthetic_multikey", {})
        rv1 = e.get("ruler_v1", {})
        lb = e.get("longbench", {})
        gs = e.get("gsm8k", {})
        ma = e.get("math500", {})
        pp = e.get("ppl", {})
        ppl_val = pp.get("perplexity") or pp.get("ppl") or next(iter(pp.values()), "")
        mem = nd.get(
            "mem_fraction",
            rv1.get("mem_fraction", lb.get("mem_fraction", ru.get("mem_fraction", ""))),
        )
        # Reasoning columns carry their 95% CI half-width; needle carries its CI too.
        aime_vals = [_fmt_ci(e.get(f"aime{y}", {})) for y in aime_years]
        display_name = e.get("method_definition", name)
        row = [
            display_name,
            _fmt_ci(nd, "accuracy", "accuracy_ci95"),
            _fmt_ci(ru, "accuracy", "accuracy_ci95"),
        ]
        if has_ruler_v1:
            row.append(str(rv1.get("score", "")))
        if has_lb:
            row.append(str(lb.get("score", "")))
        row += [str(mem), _fmt_ci(gs), _fmt_ci(ma)]
        row += aime_vals
        if has_lcb:
            row.append(_fmt_ci(e.get("functional_code_smoke", {})))
        row += [str(ppl_val)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
