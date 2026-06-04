"""KV-cache compression benchmark: long-context retrieval + reasoning.

Main experiment for the KV-compression methods, the analogue of the weight-side
``benchmark/<model>-<recipe>/quant_benchmark.py`` runs (GPTQ / JointQ).  It
compares, at a matched memory budget, the two *families* the design memo
contrasts and the proposed method:

* **FullKV**            -- fp16 upper bound.
* **Uniform-INT{4,2}**  -- uniform per-token quantization (the baseline the memo
  argues is less memory-efficient than eviction).
* **StreamingLLM**      -- sink + recent eviction (the "evict" regime).
* **ThinKV**            -- static-precision quant + (TBE) eviction (paper baseline).
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
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from onecomp.kvcomp import (
    PMKVQPolicy, PMKVQCache, TaperKVPolicy, TaperKVCache,
    TurboTaperPolicy, TurboTaperCache,
    ThinKVCache, StreamingLLMEvictCache, CompressedDynamicCache,
    build_codec, auto_configure_thinkv, detect_outlier_layers,
)
from onecomp.kvcomp.recipe.thinkv import ThinKVPolicy
from onecomp.kvcomp.eval import (
    needle_in_haystack, ruler_multikey, gsm8k_pass_at_1, math500_pass_at_1,
    aime_pass_at_1, wikitext2_perplexity, longbench_e, livecodebench_pass_at_1,
)
from onecomp.kvcomp.eval.longcontext import proportion_stats


def build_methods(cfg: DictConfig, model, tok):
    """Return ``{name: dict(make, evict, reason, capture)}``.

    ``make(ctx)`` builds the needle cache for a given context length (so the
    eviction baseline can be sized to the *same* memory budget as the quant
    methods -- a fair iso-memory test); ``reason`` builds the cache for the
    generate-based GSM8K / PPL evals (eviction caches degrade to FullKV under
    HF ``generate``, acceptable since those prompts are short).  ``capture``
    flags methods (ThinKV) whose decode-time policy needs attention sparsity.
    """
    mc = model.config
    sink = cfg.sink_tokens
    recent = cfg.recent_tokens
    target = cfg.target_mem  # iso-memory budget (stored/fp16) for the eviction window
    methods = {}

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
        _probe_texts = cfg.get("outlier_probes", [
            "Solve: what is 17 times 23? Reason step by step.",
            "Repeat the following key exactly: the passcode is QH-4821-ZX. What is the passcode?",
            "Write a Python function that returns the n-th Fibonacci number.",
            "Summarize in one sentence why the sky appears blue during the day.",
        ])
        probes = [
            tok.apply_chat_template(
                [{"role": "user", "content": t}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True)["input_ids"].to(model.device)
            for t in _probe_texts
        ]
        skip = detect_outlier_layers(
            model, probes, ratio_thresh=cfg.get("outlier_ratio", 50.0),
            agg=cfg.get("outlier_agg", "mean"))
    else:
        skip = []
    print(f"  [outlier] skip_layers (all quant methods) = {skip}", flush=True)

    methods["FullKV"] = dict(
        make=lambda ctx: DynamicCache(config=mc), evict=False,
        reason=lambda: DynamicCache(config=mc))

    for b in cfg.uniform_bits:
        # Per-token axis for both K and V: a well-defined "uniform per-token"
        # baseline whose online scales are faithful under the incremental cache
        # (per-channel keys would need block granularity; that is the KIVI codec).
        codec = build_codec(f"int{b}", skip_layers=skip, key_axis="token", value_axis="token")
        methods[f"Uniform-INT{b}"] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink))

    # Eviction window sized to the iso-memory budget at each context length:
    # recent = target * ctx - sink (so kept/total ~= target, the same budget the
    # quant methods use).  The needle survives only if it lands in this window.
    def _evict(ctx):
        rec = max(recent, int(target * ctx) - sink)
        return StreamingLLMEvictCache(sink=sink, recent=rec, config=mc)
    methods["StreamingLLM"] = dict(
        make=_evict, evict=True,
        reason=lambda: StreamingLLMEvictCache(sink=sink, recent=recent, config=mc))

    # Faithful ThinKV: outlier-layer skip + calibrated L*/thresholds (the 08
    # setup), engaged via attention capture during decode.  Without this ThinKV
    # degenerates to all-Reasoning NVFP4 over *every* layer (incl. massive-
    # activation outliers), which breaks reasoning models -- see lessons.md K23.
    if cfg.thinkv.get("auto_configure", True):
        thinp = auto_configure_thinkv(
            model, tok, precision=dict(cfg.thinkv.precision), sink_tokens=sink,
            max_new_tokens=cfg.thinkv.get("calib_max_new", 200), skip_layers=skip)
        thinp.group_size = cfg.thinkv.group_size
        print(f"  [ThinKV] skip_layers={thinp.skip_layers} L*={thinp.calib_layers} "
              f"thresholds={tuple(round(t,3) for t in thinp.thresholds)}", flush=True)
        capture = True
    else:
        thinp = ThinKVPolicy(sink_tokens=sink, group_size=cfg.thinkv.group_size,
                             precision=dict(cfg.thinkv.precision))
        capture = False
    # ``capture`` (attention sparsity for thought classification) is only used on
    # the *short* GSM8K prompts.  output_attentions at long context retains every
    # layer's full attention matrix and OOMs (lessons.md K25); needle/RULER on an
    # instruct model do not need thought classification (ThinKV quant-only already
    # retrieves), so long-context capture is left off.
    capture_long = capture and cfg.thinkv.get("capture_long", False)
    methods["ThinKV"] = dict(
        make=lambda ctx: ThinKVCache(thinp, config=mc, sink_tokens=sink), evict=False,
        reason=lambda: ThinKVCache(thinp, config=mc, sink_tokens=sink),
        capture=capture, capture_long=capture_long)

    pmp = PMKVQPolicy(key_bits=cfg.pmkvq.key_bits, value_bits=cfg.pmkvq.value_bits,
                      final_bits=cfg.pmkvq.final_bits, key_final_bits=cfg.pmkvq.key_final_bits,
                      recent_window=recent, ladder_span=cfg.pmkvq.ladder_span,
                      skip_layers=skip)
    # Faithful PM-KVQ: run the offline block-wise integer-program allocation
    # (§3.2) and the positional-interpolation channel reparameterization (§3.3).
    # Without these the comparison understates PM-KVQ by ~3-6% on AIME (the
    # "simplified block allocation" caveat); enabling them is paper-faithful.
    if cfg.pmkvq.get("block_allocation", False) or cfg.pmkvq.get("channel_reparam", False):
        from onecomp.kvcomp import calibrate_pmkvq
        calibrate_pmkvq(
            model, tok, pmp,
            block_allocation=cfg.pmkvq.get("block_allocation", False),
            channel_reparam=cfg.pmkvq.get("channel_reparam", False),
            bit_options=list(cfg.pmkvq.get("bit_options", [2, 4, 8])),
            avg_bit_budget=cfg.pmkvq.get("avg_bit_budget", None),
            pos_scale=cfg.pmkvq.get("pos_scale", 4))
        print(f"  [PM-KVQ] block_final_bits set for {len(pmp.block_final_bits)} layers; "
              f"channel_reparam on {len(pmp.key_channel_scale)} layers", flush=True)
    methods["PM-KVQ"] = dict(
        make=lambda ctx: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride), evict=False,
        reason=lambda: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride))

    tap = TaperKVPolicy(key_bits=cfg.taperkv.key_bits, value_bits=cfg.taperkv.value_bits,
                        b_min=cfg.taperkv.b_min, b_min_value=cfg.taperkv.b_min_value,
                        sink_tokens=sink, recent_tokens=recent, skip_layers=skip)
    methods["TaperKV"] = dict(
        make=lambda ctx: TaperKVCache(tap, config=mc, requant_stride=cfg.requant_stride), evict=False,
        reason=lambda: TaperKVCache(tap, config=mc, requant_stride=cfg.requant_stride))

    # TurboTaper (proposed, ThinKV+TurboQuant superset): TaperKV's age ladder
    # quantized with TurboQuant's rotated Lloyd-Max codec (MSE-optimal floor).
    # Defaults track the TaperKV config so the two are iso-schedule comparable.
    tt = cfg.get("turbotaper", {}) or {}
    ttp = TurboTaperPolicy(
        key_bits=tt.get("key_bits", cfg.taperkv.key_bits),
        value_bits=tt.get("value_bits", cfg.taperkv.value_bits),
        b_min=tt.get("b_min", cfg.taperkv.b_min),
        b_min_value=tt.get("b_min_value", cfg.taperkv.b_min_value),
        sink_tokens=sink, recent_tokens=recent, skip_layers=skip,
        dist=tt.get("dist", "gaussian"), rotation_mode=tt.get("rotation_mode", "hadamard"),
        seed=tt.get("seed", 0), n_outlier_floor=tt.get("n_outlier_floor", 0))
    methods["TurboTaper"] = dict(
        make=lambda ctx: TurboTaperCache(ttp, config=mc, requant_stride=cfg.requant_stride),
        evict=False,
        reason=lambda: TurboTaperCache(ttp, config=mc, requant_stride=cfg.requant_stride))

    # Optional TurboTaper sweep: same pattern as taperkv_sweep, so multiple
    # TurboTaper operating points (e.g. 3-bit / 4-bit average) can be compared
    # in one run -- needed for the Pareto curve in the recipe paper.
    for variant in (cfg.get("turbotaper_sweep", None) or []):
        v = dict(variant)
        label = v.pop("label")
        vp = TurboTaperPolicy(
            key_bits=v.get("key_bits", tt.get("key_bits", cfg.taperkv.key_bits)),
            value_bits=v.get("value_bits", tt.get("value_bits", cfg.taperkv.value_bits)),
            b_min=v.get("b_min", tt.get("b_min", cfg.taperkv.b_min)),
            b_min_value=v.get("b_min_value", tt.get("b_min_value", cfg.taperkv.b_min_value)),
            sink_tokens=sink, recent_tokens=v.get("recent_tokens", recent), skip_layers=skip,
            dist=v.get("dist", tt.get("dist", "gaussian")),
            rotation_mode=v.get("rotation_mode", tt.get("rotation_mode", "hadamard")),
            seed=v.get("seed", tt.get("seed", 0)),
            n_outlier_floor=v.get("n_outlier_floor", tt.get("n_outlier_floor", 0)))
        methods[label] = dict(
            make=lambda ctx, p=vp: TurboTaperCache(p, config=mc, requant_stride=cfg.requant_stride),
            evict=False,
            reason=lambda p=vp: TurboTaperCache(p, config=mc, requant_stride=cfg.requant_stride))

    # Optional TaperKV memory sweep: register one method per (label, overrides)
    # so a single run traces TaperKV's accuracy-vs-memory Pareto curve and we can
    # show it dominates PM-KVQ / eviction *at matched memory*, not just at one
    # operating point.
    for variant in (cfg.get("taperkv_sweep", None) or []):
        v = dict(variant)
        label = v.pop("label")
        vp = TaperKVPolicy(
            key_bits=v.get("key_bits", cfg.taperkv.key_bits),
            value_bits=v.get("value_bits", cfg.taperkv.value_bits),
            b_min=v.get("b_min", cfg.taperkv.b_min),
            b_min_value=v.get("b_min_value", cfg.taperkv.b_min_value),
            sink_tokens=sink, recent_tokens=v.get("recent_tokens", recent),
            skip_layers=skip)
        methods[label] = dict(
            make=lambda ctx, p=vp: TaperKVCache(p, config=mc, requant_stride=cfg.requant_stride),
            evict=False,
            reason=lambda p=vp: TaperKVCache(p, config=mc, requant_stride=cfg.requant_stride))

    # TurboQuant (arXiv:2504.19874): random rotation + Lloyd-Max scalar quant.
    # Paper bit-widths: pure 2/3/4-bit plus the *outlier-split* 2.5-bit
    # (32 ch @ 3 + 96 @ 2) and 3.5-bit (32 ch @ 4 + 96 @ 3) for head_dim=128.
    # Each variant is a dict {label, bits, [n_outlier, outlier_bits, ...]}.
    for variant in (cfg.get("turboquant_variants", None) or []):
        v = dict(variant)
        label = v.pop("label")
        codec = build_codec("turboquant", skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink))

    # KIVI (arXiv:2402.02750): per-channel key + per-token value asymmetric quant.
    # The headline baseline TurboQuant compares against; bits via ``kivi_bits``.
    for b in (cfg.get("kivi_bits", None) or []):
        codec = build_codec("kivi", skip_layers=skip, bits=b)
        methods[f"KIVI-INT{b}"] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink))

    # HALO (fast online-whitened randomized-Hadamard Lloyd-Max codec). Each entry
    # is a codec kwargs dict {label, bits, [value_bits, rotation_mode, ...]}.
    for variant in (cfg.get("halo_variants", None) or []):
        v = dict(variant)
        label = v.pop("label")
        codec = build_codec("halo", skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink))

    # Generic codec methods: benchmark ANY registered codec by name so no
    # implemented codec is unreachable from the harness (fp8, nvfp4, ternary,
    # int8, ...).  Each entry is {label, codec, **codec_kwargs}, e.g.
    # ``{label: FP8, codec: fp8}`` or ``{label: NVFP4, codec: nvfp4}``.
    for variant in (cfg.get("codec_variants", None) or []):
        v = dict(variant)
        label = v.pop("label")
        name = v.pop("codec")
        codec = build_codec(name, skip_layers=skip, **v)
        methods[label] = dict(
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink))

    return methods


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_longcontext")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=dtype, device_map=cfg.device,
        attn_implementation=cfg.attn_implementation).eval()

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
                f"and/or the corresponding sweep section in the YAML.")

    # Resume: load existing results so we can (a) skip already-done methods in a
    # normal run (if cfg.resume=true), or (b) merge a targeted re-run of specific
    # methods (cfg.methods=[...]) into the existing JSON without losing other results.
    existing_results: dict = {}
    model_tag_pre = os.path.basename(str(cfg.model_path).rstrip("/"))
    existing_json = os.path.join(cfg.output_dir, f"results_{model_tag_pre}.json")
    if os.path.exists(existing_json):
        try:
            existing_results = json.load(open(existing_json)).get("results", {})
            loaded_n = len(existing_results)
            if only is not None:
                print(f"[resume] loaded {loaded_n} existing method(s); "
                      f"will re-run and merge: {sorted(only)}", flush=True)
            elif cfg.get("resume", False):
                print(f"[resume] loaded {loaded_n} existing method(s); "
                      f"will skip already-done methods", flush=True)
        except Exception as exc:
            print(f"[resume] WARNING: could not load {existing_json}: {exc}", flush=True)
            existing_results = {}

    results = dict(existing_results)  # start from existing; overwritten per method below
    for name, spec in methods.items():
        if only is not None and name not in only:
            continue
        # Skip already-done methods when cfg.resume=true and no explicit filter.
        if only is None and cfg.get("resume", False) and name in existing_results:
            print(f"\n===== {name} [SKIP – already in results] =====", flush=True)
            continue
        print(f"\n===== {name} =====", flush=True)
        entry = {}
        cap = spec.get("capture", False)            # short reasoning prompts
        cap_long = spec.get("capture_long", False)  # long context (OOM-prone)

        # Needle: one pass per context length so eviction is sized iso-memory.
        # Skipped on reasoning models where FullKV itself fails retrieval (K24).
        if cfg.get("run_needle", True):
            by_len, mems, accs, n_tot = {}, [], [], 0
            for ctx in list(cfg.context_lens):
                nr = needle_in_haystack(
                    model, tok, cache_factory=lambda c=ctx: spec["make"](c),
                    context_lens=[ctx], depths=list(cfg.depths),
                    seeds=cfg.seeds, max_new_tokens=cfg.max_new_needle,
                    evict=spec["evict"], attn_capture=cap_long)
                by_len[ctx] = round(nr.accuracy, 4)
                mems.append(nr.mem_fraction)
                accs.append(nr.accuracy * nr.n); n_tot += nr.n
            # Pooled accuracy + binomial SE / Wilson 95% CI over all (ctx,depth,seed)
            # trials, so the paper table carries error bars (benchmark review B-4).
            acc_all = sum(accs) / max(1, n_tot)
            se, lo, hi = proportion_stats(int(round(acc_all * n_tot)), n_tot)
            entry["needle"] = {
                "accuracy": round(acc_all, 4),
                "accuracy_stderr": round(se, 4),
                "accuracy_ci95": [round(lo, 4), round(hi, 4)],
                "mem_fraction": round(sum(mems) / max(1, len(mems)), 4),
                "by_length": by_len, "n": n_tot}
            print("  needle:", entry["needle"], flush=True)

        if cfg.get("run_ruler", False):
            by_len_r, mems_r, accs_r, n_r = {}, [], [], 0
            for ctx in list(cfg.context_lens):
                rr2 = ruler_multikey(
                    model, tok, cache_factory=lambda c=ctx: spec["make"](c),
                    context_lens=[ctx], num_keys=cfg.get("ruler_keys", 4),
                    seeds=cfg.get("ruler_seeds", 5), max_new_tokens=cfg.max_new_needle,
                    evict=spec["evict"], attn_capture=cap_long)
                by_len_r[ctx] = round(rr2.accuracy, 4)
                mems_r.append(rr2.mem_fraction)
                accs_r.append(rr2.accuracy * rr2.n); n_r += rr2.n
            acc_all_r = sum(accs_r) / max(1, n_r)
            se_r, lo_r, hi_r = proportion_stats(int(round(acc_all_r * n_r)), n_r)
            entry["ruler"] = {
                "accuracy": round(acc_all_r, 4),
                "accuracy_stderr": round(se_r, 4),
                "accuracy_ci95": [round(lo_r, 4), round(hi_r, 4)],
                "mem_fraction": round(sum(mems_r) / max(1, len(mems_r)), 4),
                "by_length": by_len_r, "n": n_r}
            print("  ruler:", entry["ruler"], flush=True)

        # LongBench-E: end-to-end long-context generation (TurboQuant Table 1).
        # cache_factory ignores ctx (codec/graded caches are length-agnostic);
        # the eviction baseline uses its reason() cache (recent window).
        if cfg.get("run_longbench", False):
            lb = longbench_e(
                model, tok,
                cache_factory=(spec["reason"] if spec["evict"]
                               else lambda: spec["make"](cfg.get("longbench_max_length", 7500))),
                tasks=list(cfg.get("longbench_tasks", [])) or None,
                max_length=cfg.get("longbench_max_length", 7500),
                limit_per_task=cfg.get("longbench_limit", 20),
                evict=spec["evict"])
            entry["longbench"] = lb.as_dict()
            print("  longbench:", entry["longbench"], flush=True)

        if cfg.run_gsm8k:
            rr = gsm8k_pass_at_1(
                model, tok, cache_factory=spec["reason"],
                limit=cfg.gsm8k_limit, max_new_tokens=cfg.max_new_reason,
                attn_capture=cap)
            entry["gsm8k"] = rr.as_dict()
            print("  gsm8k:", rr.as_dict(), flush=True)

        if cfg.get("run_math", False):
            rm = math500_pass_at_1(
                model, tok, cache_factory=spec["reason"],
                limit=cfg.get("math_limit", 50),
                max_new_tokens=cfg.get("max_new_math", 3072),
                attn_capture=cap)
            entry["math500"] = rm.as_dict()
            print("  math500:", rm.as_dict(), flush=True)

        # AIME-2024 / AIME-2025: paper-level competition math evaluation.
        # ThinKV paper uses k=8 samples; PM-KVQ paper uses k=16 samples.
        # Default: k=8 (ThinKV setting) with temperature=0.6, top_p=0.95.
        for aime_year in cfg.get("aime_years", []):
            ra = aime_pass_at_1(
                model, tok, cache_factory=spec["reason"],
                year=int(aime_year),
                limit=cfg.get("aime_limit", 30),
                k=cfg.get("aime_k", 8),
                max_new_tokens=cfg.get("max_new_aime", 8192),
                attn_capture=cap)
            entry[f"aime{aime_year}"] = ra.as_dict()
            print(f"  aime{aime_year}:", ra.as_dict(), flush=True)

        # LiveCodeBench (functional subset): ThinKV's code axis. Either a local
        # JSONL of {prompt, entry_point, test} (lcb_problems_path) or the HF
        # dataset (lcb_dataset). pass@1 with the same CI as the math evals.
        if cfg.get("run_livecodebench", False):
            problems = _load_lcb_problems(cfg)
            rc = livecodebench_pass_at_1(
                model, tok, cache_factory=spec["reason"],
                dataset=cfg.get("lcb_dataset", "livecodebench/code_generation_lite"),
                split=cfg.get("lcb_split", "test"),
                limit=cfg.get("lcb_limit", 50), problems=problems,
                k=cfg.get("lcb_k", 1), max_new_tokens=cfg.get("max_new_code", 4096),
                timeout=cfg.get("lcb_timeout", 6.0), attn_capture=cap)
            entry["livecodebench"] = rc.as_dict()
            print("  livecodebench:", rc.as_dict(), flush=True)

        if cfg.run_ppl:
            pp = wikitext2_perplexity(
                model, tok, cache_factory=spec["reason"],
                max_length=cfg.ppl_max_length, stride=cfg.ppl_stride,
                max_windows=cfg.ppl_windows)
            entry["ppl"] = {k: round(float(v), 4) for k, v in pp.items()
                            if isinstance(v, (int, float))}
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
        # Persist after every method so a long multi-method sweep is crash-safe
        # (a failure late in the run still leaves the completed methods on disk).
        _save(cfg, results)

    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = _save(cfg, results)
    print(f"\nWrote {out_json}")
    print(_markdown_table(model_tag, results))


def _load_lcb_problems(cfg):
    """Local LiveCodeBench problems (JSONL of {prompt, entry_point, test}) or None.

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
                problems.append(json.loads(line))
    return problems


def _save(cfg, results) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    with open(out_json, "w") as f:
        json.dump({"model": model_tag, "config": OmegaConf.to_container(cfg, resolve=True),
                   "results": results}, f, indent=2)
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
    aime_years = sorted({
        k.replace("aime", "") for e in results.values() for k in e
        if k.startswith("aime")
    })
    aime_cols = [f"AIME-{y} p@1" for y in aime_years]
    has_lb = any("longbench" in e for e in results.values())
    lb_cols = ["LongBench-E"] if has_lb else []
    has_lcb = any("livecodebench" in e for e in results.values())
    lcb_cols = ["LiveCodeBench p@1"] if has_lcb else []
    header = (["Method", "Needle acc", "RULER acc"] + lb_cols
              + ["Mem", "GSM8K p@1", "MATH p@1"] + aime_cols + lcb_cols + ["PPL"])
    lines = [f"\n### {model_tag}", "",
             "| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    for name, e in results.items():
        nd = e.get("needle", {})
        ru = e.get("ruler", {})
        lb = e.get("longbench", {})
        gs = e.get("gsm8k", {})
        ma = e.get("math500", {})
        pp = e.get("ppl", {})
        ppl_val = pp.get("perplexity") or pp.get("ppl") or next(iter(pp.values()), "")
        mem = nd.get("mem_fraction", lb.get("mem_fraction", ru.get("mem_fraction", "")))
        # Reasoning columns carry their 95% CI half-width; needle carries its CI too.
        aime_vals = [_fmt_ci(e.get(f"aime{y}", {})) for y in aime_years]
        row = [name, _fmt_ci(nd, "accuracy", "accuracy_ci95"),
               _fmt_ci(ru, "accuracy", "accuracy_ci95")]
        if has_lb:
            row.append(str(lb.get("score", "")))
        row += [str(mem), _fmt_ci(gs), _fmt_ci(ma)]
        row += aime_vals
        if has_lcb:
            row.append(_fmt_ci(e.get("livecodebench", {})))
        row += [str(ppl_val)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
