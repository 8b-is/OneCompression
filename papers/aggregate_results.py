#!/usr/bin/env python3
"""
aggregate_results.py
====================
全ベンチマーク結果JSONを読み取り、2本の論文用に集約します。

出力:
  papers/halo_paper/results/all_results.json
  papers/taperkv_paper/results/all_results.json

使用法:
  python papers/aggregate_results.py
"""
from __future__ import annotations
import json, os, sys, copy
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent  # onecomp-lab/

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def short_model(model_name: str) -> str:
    MAP = {
        "llama-3.1-8b-instruct":              "llama-8b",
        "Ministral-8B-Instruct-2410":         "ministral-8b",
        "DeepSeek-R1-Distill-Qwen-1.5B":      "r1-qwen-1.5b",
        "DeepSeek-R1-Distill-Qwen-7B":        "r1-qwen-7b",
        "DeepSeek-R1-Distill-Qwen-14B":       "r1-qwen-14b",
        "DeepSeek-R1-Distill-Llama-8B":       "r1-llama-8b",
        "QwQ-32B":                            "qwq-32b",
    }
    for k, v in MAP.items():
        if k in model_name:
            return v
    return model_name


def _valid_cache_accounting(value: object) -> bool:
    """Validate the compact per-generation cache accounting schema."""

    if not isinstance(value, dict):
        return False
    mode = value.get("compression_aggregation")
    if mode == "fullkv_no_cache":
        return (
            value.get("compression_fp16_bits") is None
            and value.get("compression_stored_bits") is None
        )
    if mode == "exact_cache_bit_counters":
        fp16_bits = value.get("compression_fp16_bits")
        stored_bits = value.get("compression_stored_bits")
        return (
            not isinstance(fp16_bits, bool)
            and isinstance(fp16_bits, int)
            and fp16_bits > 0
            and not isinstance(stored_bits, bool)
            and isinstance(stored_bits, int)
            and stored_bits >= 0
        )
    if mode == "legacy_cache_ratio_only":
        # A displayed ratio is useful for diagnostics, but is not sufficient
        # for a citable v2 result. TaperKV now exposes integer counters; reject
        # older rows so they cannot silently re-enter the paper aggregate.
        return False
    return False


def _valid_synthetic_grid_row(row: object, *, require_depth: bool) -> bool:
    """Validate one persisted retrieval trial before it reaches a paper table."""

    if not isinstance(row, dict):
        return False
    integer_fields = (
        "requested_filler_tokens",
        "prompt_seed",
        "rendered_prompt_tokens",
        "reserved_prompt_plus_generation_tokens",
        "model_context_capacity",
    )
    if any(
        isinstance(row.get(key), bool)
        or not isinstance(row.get(key), int)
        or row[key] <= 0
        for key in integer_fields
    ):
        return False
    if not (
        row["rendered_prompt_tokens"]
        < row["reserved_prompt_plus_generation_tokens"]
        <= row["model_context_capacity"]
    ):
        return False
    if require_depth:
        depth = row.get("requested_depth")
        if (
            isinstance(depth, bool)
            or not isinstance(depth, (int, float))
            or not math.isfinite(float(depth))
            or not 0.0 <= float(depth) <= 1.0
        ):
            return False
    prompt_sha256 = row.get("prompt_sha256")
    if (
        not isinstance(prompt_sha256, str)
        or len(prompt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in prompt_sha256)
    ):
        return False
    return (
        isinstance(row.get("gold"), str)
        and bool(row["gold"])
        and isinstance(row.get("prediction"), str)
        and isinstance(row.get("hit"), bool)
        and _valid_cache_accounting(row.get("cache_accounting"))
    )


def audited_needle(bench: dict) -> dict | None:
    """Return a citable synthetic-needle record or reject legacy summaries.

    Needle v1 did not retain the rendered prompt length, depth, prediction, or
    exact hit ledger, and its filler counter injected sentence-level special
    tokens.  Keeping its scalar accuracy in a paper aggregate would let a
    historical development run masquerade as a reproducible retrieval result.
    """

    needle = bench.get("needle", {})
    if not isinstance(needle, dict):
        return None
    if (
        needle.get("benchmark") != "deterministic synthetic needle-in-a-haystack v2"
        or needle.get("protocol_version") != 2
        or needle.get("official_reference") is not False
        or not isinstance(needle.get("grid"), list)
        or needle.get("n") != len(needle["grid"])
    ):
        return None
    n, hits = needle.get("n"), needle.get("hits")
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        or isinstance(hits, bool)
        or not isinstance(hits, int)
        or not 0 <= hits <= n
    ):
        return None
    grid = needle["grid"]
    if any(not _valid_synthetic_grid_row(row, require_depth=True) for row in grid):
        return None
    if sum(int(row["hit"] is True) for row in grid) != hits:
        return None
    accuracy = needle.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or not math.isclose(float(accuracy), hits / n, abs_tol=5e-4)
    ):
        return None
    return needle


def audited_synthetic_multikey(bench: dict) -> dict | None:
    """Return an explicitly-labelled local multi-key diagnostic, if auditable.

    It deliberately has a separate name from RULER: a local synthetic task must
    never become an apparent external RULER score in a paper summary.
    """

    result = bench.get("synthetic_multikey", {})
    if not isinstance(result, dict):
        return None
    if (
        result.get("benchmark") != "deterministic synthetic multi-key retrieval v2"
        or result.get("protocol_version") != 2
        or result.get("official_reference") is not False
        or not isinstance(result.get("grid"), list)
        or result.get("n") != len(result["grid"])
    ):
        return None
    n, hits = result.get("n"), result.get("hits")
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        or isinstance(hits, bool)
        or not isinstance(hits, int)
        or not 0 <= hits <= n
        or any(
            not _valid_synthetic_grid_row(row, require_depth=True)
            for row in result["grid"]
        )
        or sum(int(row["hit"]) for row in result["grid"]) != hits
    ):
        return None
    return result

# ──────────────────────────────────────────────────────────
# 1.  HALO Paper — Codec + Speed + Longcontext retrieval
# ──────────────────────────────────────────────────────────
def aggregate_halo_paper() -> dict:
    out: dict = {
        "paper": "HALO: Hadamard-Rotation and Least-Squares Optimal KV Codec",
        "description": (
            "Speed-quality trade-off paper for data-free rotated KV quantization. "
            "Main claim: HALO Pareto-dominates TurboQuant across PPL, latency, and retrieval."
        ),
        "codec_ppl_latency": {},   # model -> bits -> method -> metrics
        "longcontext_retrieval": {},  # model -> method -> {needle, ruler, ppl}
        "speed": {},               # config -> bits -> method -> {enc_ms, dec_ms}
        "status": {
            "complete": [],
            "running": [],
            "missing": [],
        }
    }

    # ── Codec benchmark (PPL + latency) ──
    codec_dirs = {
        "llama-8b":      ROOT / "benchmark/kvcomp-codec/results_paper_llama",
        "ministral-8b":  ROOT / "benchmark/kvcomp-codec/results_paper_ministral",
        "r1-qwen-1.5b":  ROOT / "benchmark/kvcomp-codec/results_paper_r1q15",
        "r1-qwen-7b":    ROOT / "benchmark/kvcomp-codec/results_paper_r1q7",
    }
    for mkey, dpath in codec_dirs.items():
        files = list(dpath.glob("results_*.json")) if dpath.exists() else []
        if not files:
            out["status"]["missing"].append(f"codec/{mkey}")
            continue
        data = load(files[0])
        results_by_bits: dict = {}
        for bits_str, methods in data.get("results", {}).items():
            results_by_bits[bits_str] = {}
            for method, metrics in methods.items():
                results_by_bits[bits_str][method] = {
                    "ppl": metrics.get("ppl"),
                    "distortion_key": metrics.get("distortion_key"),
                    "distortion_value": metrics.get("distortion_value"),
                    "ms_per_encode": metrics.get("ms_per_encode"),
                    "throughput_Melem_s": metrics.get("throughput_Melem_s"),
                    "mem_fraction": metrics.get("mem_fraction"),
                    "compression_ratio": metrics.get("compression_ratio"),
                }
        out["codec_ppl_latency"][mkey] = results_by_bits
        out["status"]["complete"].append(f"codec/{mkey}")

    # ── Long-context retrieval (Needle + local multi-key diagnostic + PPL) ──
    retr_dirs = {
        "llama-8b":     ROOT / "benchmark/kvcomp-longcontext/results_tq_llama_retr",
        "ministral-8b": ROOT / "benchmark/kvcomp-longcontext/results_tq_minis_retr",
    }
    for mkey, dpath in retr_dirs.items():
        files = list(dpath.glob("results_*.json")) if dpath.exists() else []
        if not files:
            out["status"]["missing"].append(f"retrieval/{mkey}")
            continue
        data = load(files[0])
        methods_out: dict = {}
        for method, bench in data.get("results", {}).items():
            needle = audited_needle(bench)
            multikey = audited_synthetic_multikey(bench)
            methods_out[method] = {
                "needle_acc":       needle.get("accuracy") if needle else None,
                "needle_protocol":  needle.get("benchmark") if needle else "legacy_or_unverified",
                "synthetic_multikey_acc": (
                    multikey.get("accuracy") if multikey else None
                ),
                "synthetic_multikey_protocol": (
                    multikey.get("benchmark") if multikey else "legacy_or_unverified"
                ),
                "ppl":              bench.get("ppl", {}).get("ppl"),
                "mem_fraction":     bench.get("needle", {}).get("mem_fraction")
                                    or bench.get("ppl", {}).get("mem_fraction"),
                "compression_ratio": bench.get("ppl", {}).get("compression_ratio"),
            }
        out["longcontext_retrieval"][mkey] = methods_out
        out["status"]["complete"].append(f"retrieval/{mkey}")

    # ── LongBench (longcontext understanding) ──
    lb_dirs = {
        "llama-8b":     ROOT / "benchmark/kvcomp-longcontext/results_tq_llama_lb",
        "ministral-8b": ROOT / "benchmark/kvcomp-longcontext/results_tq_minis_lb",
    }
    out["longbench"] = {}
    for mkey, dpath in lb_dirs.items():
        files = list(dpath.glob("results_*.json")) if dpath.exists() else []
        if not files:
            out["status"]["missing"].append(f"longbench/{mkey}")
            continue
        data = load(files[0])
        methods_out = {}
        for method, bench in data.get("results", {}).items():
            lb = bench.get("longbench", {})
            methods_out[method] = {
                "score":        lb.get("score"),
                "mem_fraction": lb.get("mem_fraction"),
                "per_task":     lb.get("per_task", {}),
            }
        out["longbench"][mkey] = methods_out
        out["status"]["complete"].append(f"longbench/{mkey}")

    # ── Speed benchmark ──
    speed_files = {
        "llama-setup":  ROOT / "benchmark/kvcomp-longcontext/results_codec_speed/codec_speed_llama.json",
        "cuda-large":   ROOT / "benchmark/kvcomp-longcontext/results_codec_speed/codec_speed_cuda.json",
    }
    for skey, spath in speed_files.items():
        data = load(spath)
        if data:
            speed_out: dict = {}
            for bits_str, methods in data.get("results", {}).items():
                speed_out[bits_str] = {}
                for method, metrics in methods.items():
                    speed_out[bits_str][method] = {
                        "enc_ms":   metrics.get("enc_ms"),
                        "dec_ms":   metrics.get("dec_ms"),
                        "kv_mse":   metrics.get("kv_mse"),
                        "attn_err": metrics.get("attn_err"),
                        "bits_per_el": metrics.get("bits_per_el"),
                    }
            out["speed"][skey] = speed_out
            out["status"]["complete"].append(f"speed/{skey}")
        else:
            out["status"]["missing"].append(f"speed/{skey}")

    return out


# ──────────────────────────────────────────────────────────
# 2.  TaperKV Paper — Policy comparison + Reasoning
# ──────────────────────────────────────────────────────────
def aggregate_taperkv_paper() -> dict:
    out: dict = {
        "paper": "TaperKV: Recency-Graded Precision Scheduling for KV Cache Compression",
        "description": (
            "Policy paper for age-graded precision scheduling without token eviction. "
            "Needle results are reported as an accuracy-vs-realised-memory Pareto "
            "frontier unless an explicit matched-memory validation is present."
        ),
        "policy_comparison": {},  # model -> method -> metrics (needle/ruler/lb/gsm8k/math500)
        "aime_reasoning": {},     # model -> method -> {aime2024, aime2025, gsm8k, math500}
        "livecodebench": {},      # model -> method -> {lcb_pass@1}
        "status": {
            "complete": [],
            "running": [],
            "missing": [],
        }
    }

    # ── Policy comparison (TaperKV suite) ──
    taperkv_dirs = {
        "llama-8b":     ROOT / "benchmark/kvcomp-longcontext/results_taperkv",
        "r1-qwen-7b":   ROOT / "benchmark/kvcomp-longcontext/results_taperkv",
        "r1-qwen-14b":  ROOT / "benchmark/kvcomp-longcontext/results_taperkv",
        "r1-qwen-1.5b": ROOT / "benchmark/kvcomp-longcontext/results_taperkv",
    }
    model_filename_map = {
        "llama-8b":     "results_llama-3.1-8b-instruct.json",
        "r1-qwen-7b":   "results_DeepSeek-R1-Distill-Qwen-7B.json",
        "r1-qwen-14b":  "results_DeepSeek-R1-Distill-Qwen-14B.json",
        "r1-qwen-1.5b": "results_DeepSeek-R1-Distill-Qwen-1.5B.json",
    }
    for mkey, dpath in taperkv_dirs.items():
        fpath = dpath / model_filename_map[mkey]
        data = load(fpath)
        if not data:
            out["status"]["missing"].append(f"policy/{mkey}")
            continue
        methods_out = {}
        n_complete = 0
        for method, bench in data.get("results", {}).items():
            needle = audited_needle(bench)
            multikey = audited_synthetic_multikey(bench)
            methods_out[method] = {
                "needle_acc":    needle.get("accuracy") if needle else None,
                "needle_protocol": needle.get("benchmark") if needle else "legacy_or_unverified",
                "synthetic_multikey_acc": (
                    multikey.get("accuracy") if multikey else None
                ),
                "synthetic_multikey_protocol": (
                    multikey.get("benchmark") if multikey else "legacy_or_unverified"
                ),
                "longbench":     bench.get("longbench", {}).get("score"),
                "gsm8k":         bench.get("gsm8k", {}).get("pass@1"),
                "math500":       bench.get("math500", {}).get("pass@1"),
                "ppl":           bench.get("ppl", {}).get("ppl"),
                "mem_fraction":  (needle or bench.get("ppl", {})).get("mem_fraction"),
                "compression_ratio": bench.get("ppl", {}).get("compression_ratio"),
            }
            n_complete += 1
        out["policy_comparison"][mkey] = methods_out
        # 11 methods expected
        status = "complete" if n_complete >= 10 else "partial"
        out["status"][status if status == "complete" else "running"].append(
            f"policy/{mkey} ({n_complete}/11)"
        )

    # ── AIME / GSM8K / MATH-500 (basic run) ──
    aime_basic_dirs = {
        "qwq-32b":      ROOT / "benchmark/kvcomp-longcontext/results_aime_basic_qwq/results_QwQ-32B.json",
        "r1-llama-8b":  ROOT / "benchmark/kvcomp-longcontext/results_aime_basic_r1l8/results_DeepSeek-R1-Distill-Llama-8B.json",
        "r1-qwen-14b":  ROOT / "benchmark/kvcomp-longcontext/results_aime_basic_r1q14/results_DeepSeek-R1-Distill-Qwen-14B.json",
        "r1-qwen-1.5b": ROOT / "benchmark/kvcomp-longcontext/results_aime_basic_r1q15/results_DeepSeek-R1-Distill-Qwen-1.5B.json",
        "r1-qwen-7b":   ROOT / "benchmark/kvcomp-longcontext/results_aime_basic_r1q7/results_DeepSeek-R1-Distill-Qwen-7B.json",
    }
    for mkey, fpath in aime_basic_dirs.items():
        data = load(fpath)
        if not data:
            out["status"]["missing"].append(f"aime_basic/{mkey}")
            continue
        methods_out = {}
        for method, bench in data.get("results", {}).items():
            methods_out[method] = {
                "gsm8k":  bench.get("gsm8k", {}).get("pass@1"),
                "math500": bench.get("math500", {}).get("pass@1"),
                "ppl":    bench.get("ppl", {}).get("ppl"),
                "mem_fraction": bench.get("ppl", {}).get("mem_fraction"),
            }
        out["aime_reasoning"].setdefault(mkey, {}).update({"basic": methods_out})
        n = len(methods_out)
        out["status"]["complete" if n >= 7 else "running"].append(f"aime_basic/{mkey} ({n} methods)")

    # ── AIME comparison (AIME2024 + LiveCodeBench) ──
    aime_comp_dirs = {
        "qwq-32b":     ROOT / "benchmark/kvcomp-longcontext/results_aime_comp_qwq_24/results_QwQ-32B.json",
        "r1-llama-8b": ROOT / "benchmark/kvcomp-longcontext/results_aime_comp_r1l8/results_DeepSeek-R1-Distill-Llama-8B.json",
        "r1-qwen-7b":  ROOT / "benchmark/kvcomp-longcontext/results_aime_comp_r1q7/results_DeepSeek-R1-Distill-Qwen-7B.json",
        "r1-qwen-14b_24": ROOT / "benchmark/kvcomp-longcontext/results_aime_comp_r1q14_24/results_DeepSeek-R1-Distill-Qwen-14B.json",
        "r1-qwen-1.5b": ROOT / "benchmark/kvcomp-longcontext/results_aime_comp_r1q15/results_DeepSeek-R1-Distill-Qwen-1.5B.json",
    }
    for mkey_raw, fpath in aime_comp_dirs.items():
        mkey = mkey_raw.replace("_24", "")
        data = load(fpath)
        if not data:
            out["status"]["missing"].append(f"aime_comp/{mkey_raw}")
            continue
        comp_methods: dict = {}
        lcb_methods: dict = {}
        for method, bench in data.get("results", {}).items():
            aime24 = bench.get("aime2024", {})
            lcb = bench.get("livecodebench", {})
            if aime24:
                comp_methods[method] = {
                    "aime2024_pass@1": aime24.get("pass@1"),
                    "aime2024_k": aime24.get("k"),
                }
            if lcb:
                lcb_methods[method] = {
                    "lcb_pass@1": lcb.get("pass@1"),
                    "lcb_n": lcb.get("n"),
                }
        if comp_methods:
            out["aime_reasoning"].setdefault(mkey, {})["aime2024"] = comp_methods
        if lcb_methods:
            out["livecodebench"][mkey] = lcb_methods
            n = len(lcb_methods)
            out["status"]["complete" if n >= 7 else "running"].append(
                f"lcb/{mkey_raw} ({n} methods)"
            )

    return out


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    halo = aggregate_halo_paper()
    taper = aggregate_taperkv_paper()

    halo_out  = ROOT / "papers/halo_paper/results/all_results.json"
    taper_out = ROOT / "papers/taperkv_paper/results/all_results.json"

    with open(halo_out, "w") as f:
        json.dump(halo, f, indent=2)
    print(f"[HALO]    wrote {halo_out}")

    with open(taper_out, "w") as f:
        json.dump(taper, f, indent=2)
    print(f"[TaperKV] wrote {taper_out}")

    # ── Print summary tables ──
    def _pct(v):
        return f"{v*100:.1f}" if v is not None else "—"
    def _f(v, d=2):
        return f"{v:.{d}f}" if v is not None else "—"
    def _ms(v):
        return f"{v:.3f}" if v is not None else "—"

    print("\n" + "="*70)
    print("HALO PAPER — Codec PPL @ 3-bit (Llama-8B / Ministral-8B)")
    print("="*70)
    for model in ["llama-8b", "ministral-8b"]:
        print(f"\n  {model}")
        data = halo["codec_ppl_latency"].get(model, {}).get("3", {})
        for m, v in sorted(data.items()):
            print(f"    {m:<30} ppl={_f(v['ppl'])}  enc={_ms(v['ms_per_encode'])}ms  relMSE_k={_f(v['distortion_key'],4)}")

    print("\n" + "="*70)
    print("HALO PAPER — Retrieval Accuracy (Needle / synthetic multi-key) @ realised memory")
    print("="*70)
    for model in ["llama-8b", "ministral-8b"]:
        print(f"\n  {model}")
        data = halo["longcontext_retrieval"].get(model, {})
        key_methods = ["FullKV","Uniform-INT4","HALO-4bit","HALO-3bit","HALO-2bit",
                       "HALO-dense-LS-4bit","HALO-dense-LS-3bit","HALO-dense-LS-2bit",
                       "TurboQuant-4bit","TurboQuant-3bit","TurboQuant-2bit",
                       "FP8","NVFP4","KIVI-INT2"]
        print(f"    {'method':<28} {'needle':>7} {'synthetic':>9} {'ppl':>7} {'mem':>6}")
        for m in key_methods:
            v = data.get(m, {})
            print(f"    {m:<28} {_pct(v.get('needle_acc')):>7} {_pct(v.get('synthetic_multikey_acc')):>9} {_f(v.get('ppl')):>7} {_f(v.get('mem_fraction'),3):>6}")

    print("\n" + "="*70)
    print("HALO PAPER — Speed (encode ms @3-bit, Llama setup)")
    print("="*70)
    speed_3 = halo["speed"].get("llama-setup", {}).get("3", {})
    for m, v in sorted(speed_3.items()):
        print(f"  {m:<36} enc={_ms(v['enc_ms'])}ms  dec={_ms(v['dec_ms'])}ms  kv_mse={_f(v.get('kv_mse'),4)}")

    print("\n" + "="*70)
    print("TaperKV PAPER — Policy Comparison (Llama-8B)")
    print("="*70)
    data = taper["policy_comparison"].get("llama-8b", {})
    print(f"  {'method':<22} {'needle':>7} {'synthetic':>9} {'lb':>7} {'gsm8k':>7} {'math':>7} {'ppl':>7} {'mem':>6}")
    key_methods = ["FullKV","StreamingLLM","ThinKV","PM-KVQ",
                   "TaperKV","TaperKV-3bit","TurboTaper-3bit",
                   "TurboQuant-3bit","KIVI-INT2","HALO-3bit","Uniform-INT3"]
    for m in key_methods:
        v = data.get(m, {})
        print(f"  {m:<22} {_pct(v.get('needle_acc')):>7} {_pct(v.get('synthetic_multikey_acc')):>9} "
              f"{_f(v.get('longbench')):>7} {_pct(v.get('gsm8k')):>7} "
              f"{_pct(v.get('math500')):>7} {_f(v.get('ppl')):>7} {_f(v.get('mem_fraction'),3):>6}")

    print("\n" + "="*70)
    print("TaperKV PAPER — AIME2024 (QwQ-32B, pass@1 k=1)")
    print("="*70)
    aime_q = taper["aime_reasoning"].get("qwq-32b", {}).get("aime2024", {})
    for m, v in sorted(aime_q.items()):
        print(f"  {m:<28} aime2024_pass@1={_f(v.get('aime2024_pass@1'))}")

    print("\n" + "="*70)
    print("STATUS SUMMARY")
    print("="*70)
    for label, items in [("HALO complete", halo["status"]["complete"]),
                          ("HALO missing",  halo["status"]["missing"]),
                          ("TaperKV complete", taper["status"]["complete"]),
                          ("TaperKV running",  taper["status"]["running"]),
                          ("TaperKV missing",  taper["status"]["missing"])]:
        if items:
            print(f"\n  {label}:")
            for x in items:
                print(f"    • {x}")
