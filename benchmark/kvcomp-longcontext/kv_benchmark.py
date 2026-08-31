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
    CompressedDynamicCache,
    PMKVQCache,
    PMKVQPolicy,
    StreamingLLMEvictCache,
    TaperKVCache,
    TaperKVPolicy,
    ThinKVCache,
    auto_configure_thinkv,
    build_codec,
    detect_outlier_layers,
)
from onecomp.kvcomp.eval import (
    gsm8k_pass_at_1,
    math500_pass_at_1,
    needle_in_haystack,
    ruler_multikey,
    wikitext2_perplexity,
)
from onecomp.kvcomp.recipe.thinkv import ThinKVPolicy


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
        probe = tok.apply_chat_template(
            [{"role": "user", "content": "Solve: what is 17 times 23? Reason step by step."}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"].to(model.device)
        skip = detect_outlier_layers(model, probe, ratio_thresh=cfg.get("outlier_ratio", 50.0))
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
            make=lambda ctx, c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
            evict=False,
            reason=lambda c=codec: CompressedDynamicCache(codec=c, config=mc, sink_tokens=sink),
        )

    # Eviction window sized to the iso-memory budget at each context length:
    # recent = target * ctx - sink (so kept/total ~= target, the same budget the
    # quant methods use).  The needle survives only if it lands in this window.
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
    methods["PM-KVQ"] = dict(
        make=lambda ctx: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride),
        evict=False,
        reason=lambda: PMKVQCache(pmp, config=mc, requant_stride=cfg.requant_stride),
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

    return methods


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_longcontext")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        dtype=dtype,
        device_map=cfg.device,
        attn_implementation=cfg.attn_implementation,
    ).eval()

    methods = build_methods(cfg, model, tok)
    only = set(cfg.methods) if cfg.get("methods") else None
    results = {}
    for name, spec in methods.items():
        if only is not None and name not in only:
            continue
        print(f"\n===== {name} =====", flush=True)
        entry = {}
        cap = spec.get("capture", False)  # short reasoning prompts
        cap_long = spec.get("capture_long", False)  # long context (OOM-prone)

        # Needle: one pass per context length so eviction is sized iso-memory.
        # Skipped on reasoning models where FullKV itself fails retrieval (K24).
        if cfg.get("run_needle", True):
            by_len, mems, accs, n_tot = {}, [], [], 0
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
                mems.append(nr.mem_fraction)
                accs.append(nr.accuracy * nr.n)
                n_tot += nr.n
            entry["needle"] = {
                "accuracy": round(sum(accs) / max(1, n_tot), 4),
                "mem_fraction": round(sum(mems) / max(1, len(mems)), 4),
                "by_length": by_len,
                "n": n_tot,
            }
            print("  needle:", entry["needle"], flush=True)

        if cfg.get("run_ruler", False):
            by_len_r, mems_r, accs_r, n_r = {}, [], [], 0
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
                mems_r.append(rr2.mem_fraction)
                accs_r.append(rr2.accuracy * rr2.n)
                n_r += rr2.n
            entry["ruler"] = {
                "accuracy": round(sum(accs_r) / max(1, n_r), 4),
                "mem_fraction": round(sum(mems_r) / max(1, len(mems_r)), 4),
                "by_length": by_len_r,
                "n": n_r,
            }
            print("  ruler:", entry["ruler"], flush=True)

        if cfg.run_gsm8k:
            rr = gsm8k_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                limit=cfg.gsm8k_limit,
                max_new_tokens=cfg.max_new_reason,
                attn_capture=cap,
            )
            entry["gsm8k"] = rr.as_dict()
            print("  gsm8k:", rr.as_dict(), flush=True)

        if cfg.get("run_math", False):
            rm = math500_pass_at_1(
                model,
                tok,
                cache_factory=spec["reason"],
                limit=cfg.get("math_limit", 50),
                max_new_tokens=cfg.get("max_new_math", 3072),
                attn_capture=cap,
            )
            entry["math500"] = rm.as_dict()
            print("  math500:", rm.as_dict(), flush=True)

        if cfg.run_ppl:
            pp = wikitext2_perplexity(
                model,
                tok,
                cache_factory=spec["reason"],
                max_length=cfg.ppl_max_length,
                stride=cfg.ppl_stride,
                max_windows=cfg.ppl_windows,
            )
            entry["ppl"] = {
                k: round(float(v), 4) for k, v in pp.items() if isinstance(v, (int, float))
            }
            print("  ppl:", entry["ppl"], flush=True)

        results[name] = entry
        # Persist after every method so a long multi-method sweep is crash-safe
        # (a failure late in the run still leaves the completed methods on disk).
        _save(cfg, results)

    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = _save(cfg, results)
    print(f"\nWrote {out_json}")
    print(_markdown_table(model_tag, results))


def _save(cfg, results) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_tag = os.path.basename(str(cfg.model_path).rstrip("/"))
    out_json = os.path.join(cfg.output_dir, f"results_{model_tag}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "model": model_tag,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "results": results,
            },
            f,
            indent=2,
        )
    return out_json


def _markdown_table(model_tag, results) -> str:
    lines = [
        f"\n### {model_tag}",
        "",
        "| Method | Needle acc | RULER acc | Mem | GSM8K p@1 | MATH p@1 | PPL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in results.items():
        nd = e.get("needle", {})
        ru = e.get("ruler", {})
        gs = e.get("gsm8k", {})
        ma = e.get("math500", {})
        pp = e.get("ppl", {})
        ppl_val = pp.get("perplexity") or pp.get("ppl") or next(iter(pp.values()), "")
        mem = nd.get("mem_fraction", ru.get("mem_fraction", ""))
        lines.append(
            f"| {name} | {nd.get('accuracy','')} | {ru.get('accuracy','')} "
            f"| {mem} | {gs.get('pass@1','')} | {ma.get('pass@1','')} | {ppl_val} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
