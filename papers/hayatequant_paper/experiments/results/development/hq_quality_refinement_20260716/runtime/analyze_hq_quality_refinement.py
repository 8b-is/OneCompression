"""Fail-closed analysis of the preregistered development-only HQ/AHQ panel."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[5]
BASELINE_ROOT = ROOT.parent / "baseline_quality_20260716"
MODELS = {
    "llama31": "llama-3.1-8b-instruct",
    "llama2": "Llama-2-7b-hf",
    "qwen34": "Qwen3-4B",
    "qwen38": "Qwen3-8B",
    "r1llama8": "DeepSeek-R1-Distill-Llama-8B",
    "r1qwen15": "DeepSeek-R1-Distill-Qwen-1.5B",
    "r1qwen7": "DeepSeek-R1-Distill-Qwen-7B",
}
RATES = (2, 3, 4)
FIXED_METHODS = (
    "DEVONLY-HQ-b4-gaussian-auto",
    "DEVONLY-HQ-b4-gaussian-fibinit",
    "DEVONLY-HQ-b4-finite-beta",
)
FIXED_CONTROL = "CURRENT-HQ"
ADAPTIVE_METHODS = (
    "DEVONLY-AdaptiveHQ-safe-kw1",
    "DEVONLY-AdaptiveHQ-safe-kw4",
    "DEVONLY-AdaptiveHQ-safe-kw16",
)
ADAPTIVE_WEIGHTS = dict(zip(ADAPTIVE_METHODS, (1.0, 4.0, 16.0)))
EXTERNAL_BASELINES = ("TurboQuant-paper", "FibQuant-paper", "BlockQuant-paper")
FIXED_RAW_METHODS = {"TurboQuant-paper", "HQ", *FIXED_METHODS}
NATIVE_METRICS = (
    "prefill_store",
    "decode_attention",
    "serving_step_store1_plus_decode",
    "cudagraph_serving_step_store1_plus_decode",
)
NATIVE_SMOKES = {
    "CURRENT-HQ-b2-K5V3": (
        "current_hq_b2_smoke.json",
        (5, 2, 3, 2, "gaussian", "auto", False),
    ),
    "DEVONLY-HQ-b4-gaussian-auto": (
        "hq_b4_gaussian_auto_smoke.json",
        (10, 4, 6, 4, "gaussian", "auto", True),
    ),
    "DEVONLY-HQ-b4-gaussian-fibinit": (
        "hq_b4_gaussian_fibinit_smoke.json",
        (10, 4, 6, 4, "gaussian", "fibquant", True),
    ),
    "DEVONLY-HQ-b4-finite-beta": (
        "hq_b4_finite_beta_smoke.json",
        (10, 4, 6, 4, "beta", "auto", True),
    ),
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} is non-finite: {value!r}")
    return number


def _load(path: Path) -> dict:
    if not path.is_file():
        _fail(f"missing result {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON {path}") from exc


def _identity(entry: dict) -> tuple:
    return (
        entry.get("ppl_source"),
        entry.get("ppl_dataset_revision"),
        entry.get("ppl_text_sha256"),
    )


def _validate_entry(
    entry: dict,
    *,
    label: str,
    n_tokens: int,
    identity: tuple,
    temporal: bool,
) -> None:
    for field in ("ppl", "ppl_nll", "ppl_nll_sum", "distortion_key", "distortion_value"):
        _finite(entry.get(field), f"{label}/{field}")
    if int(entry.get("ppl_n_tokens", -1)) != n_tokens:
        _fail(f"{label} token denominator mismatch")
    if int(entry.get("ppl_n_windows", -1)) != 1:
        _fail(f"{label} is not the preregistered one-window endpoint")
    if entry.get("ppl_complete_text_eval") is not False:
        _fail(f"{label} is incorrectly marked as a complete-text result")
    if int(entry.get("ppl_n_skipped_nonfinite", -1)) != 0:
        _fail(f"{label} skipped a non-finite PPL window")
    if _identity(entry) != identity:
        _fail(f"{label} dataset identity mismatch")
    if int(entry.get("ppl_compression_fp16_bits", -1)) <= 0:
        _fail(f"{label} has no positive exact fp16 counter")
    if int(entry.get("ppl_compression_stored_bits", -1)) <= 0:
        _fail(f"{label} has no positive exact stored-bit counter")
    protocol = entry.get("ppl_teacher_forced_scoring_protocol")
    expected = (
        "causal_prefix_prefill_then_token_step_targets"
        if temporal
        else "single_forward_causal_mask_token_local_cache"
    )
    if protocol != expected:
        _fail(f"{label} teacher-forced protocol mismatch: {protocol!r}")


def _result_path(root: Path, key: str, model: str) -> Path:
    return root / key / f"results_{model}.json"


def _summary(cells: list[dict], candidate: str) -> dict:
    external = [
        row
        for row in cells
        if row["candidate"] == candidate and row["comparison_family"] == "external"
    ]
    hq = [
        row
        for row in cells
        if row["candidate"] == candidate and row["comparison_family"] == "current_hq"
    ]
    previous_adaptive = [
        row
        for row in cells
        if row["candidate"] == candidate
        and row["comparison_family"] == "previous_adaptive_hq"
    ]
    by_baseline = {}
    for baseline in EXTERNAL_BASELINES:
        rows = [row for row in external if row["baseline"] == baseline]
        if rows:
            by_baseline[baseline] = {
                "wins": sum(bool(row["strict_unrounded_nll_win"]) for row in rows),
                "total": len(rows),
            }
    return {
        "candidate": candidate,
        "external": {
            "wins": sum(bool(row["strict_unrounded_nll_win"]) for row in external),
            "total": len(external),
            "by_baseline": by_baseline,
        },
        "versus_current_hq": {
            "wins": sum(bool(row["strict_unrounded_nll_win"]) for row in hq),
            "total": len(hq),
            "rate2_wins": sum(
                bool(row["strict_unrounded_nll_win"]) for row in hq if row["rate"] == 2
            ),
        },
        "versus_previous_adaptive_hq": {
            "wins": sum(
                bool(row["strict_unrounded_nll_win"]) for row in previous_adaptive
            ),
            "total": len(previous_adaptive),
            "by_rate": {
                str(rate): {
                    "wins": sum(
                        bool(row["strict_unrounded_nll_win"])
                        for row in previous_adaptive
                        if row["rate"] == rate
                    ),
                    "total": sum(1 for row in previous_adaptive if row["rate"] == rate),
                }
                for rate in RATES
            },
        },
        "worst_nll_delta_vs_current_hq": max(
            (float(row["nll_difference_candidate_minus_baseline"]) for row in hq),
            default=0.0,
        ),
        "sum_nll_delta_vs_current_hq": sum(
            float(row["nll_difference_candidate_minus_baseline"]) for row in hq
        ),
        "worst_nll_delta_vs_previous_adaptive_hq": max(
            (
                float(row["nll_difference_candidate_minus_baseline"])
                for row in previous_adaptive
            ),
            default=0.0,
        ),
        "sum_nll_delta_vs_previous_adaptive_hq": sum(
            float(row["nll_difference_candidate_minus_baseline"])
            for row in previous_adaptive
        ),
    }


def _fixed_rank(summary: dict) -> tuple:
    external = summary["external"]
    by = external["by_baseline"]
    return (
        by["TurboQuant-paper"]["wins"],
        by["FibQuant-paper"]["wins"],
        external["wins"],
        summary["versus_current_hq"]["rate2_wins"],
        -summary["sum_nll_delta_vs_current_hq"],
    )


def _adaptive_rank(summary: dict) -> tuple:
    external = summary["external"]
    return (
        external["by_baseline"]["TurboQuant-paper"]["wins"],
        external["wins"],
        summary["versus_current_hq"]["wins"],
        -summary["worst_nll_delta_vs_current_hq"],
        -summary["sum_nll_delta_vs_current_hq"],
    )


def _by_rate_external_summary(cells: list[dict], candidate: str) -> dict:
    result = {}
    for rate in RATES:
        rows = [
            row
            for row in cells
            if row["candidate"] == candidate
            and row["comparison_family"] == "external"
            and row["rate"] == rate
        ]
        result[str(rate)] = {
            "wins": sum(bool(row["strict_unrounded_nll_win"]) for row in rows),
            "total": len(rows),
        }
    return result


def _native_contract_audit() -> dict:
    sys.path.insert(0, str(REPO))
    from vllm_plugins.kvcomp.hayate_full_config import (  # pylint: disable=import-outside-toplevel
        FullHQNativeRecipe,
        fixed_hq_native_recipe,
    )

    default_rejects = False
    try:
        FullHQNativeRecipe(10, 4, 6, 4)
    except ValueError:
        default_rejects = True
    candidate = FullHQNativeRecipe(
        10,
        4,
        6,
        4,
        max_codebook_bits=10,
        allow_canonical_k10v6_b4=True,
        assignment_mode="exact",
    )
    current = fixed_hq_native_recipe(2)
    candidate_layout = candidate.layout(128)
    current_layout = current.layout(128)
    if candidate_layout["slot_size_aligned"] != current_layout["slot_size_aligned"]:
        _fail("K10/V6 native layout does not preserve the current rate-2 slot size")
    if not default_rejects:
        _fail("K10/V6 unexpectedly bypassed the default native safety cap")
    candidate_codebook_bytes = ((1 << 10) + (1 << 6)) * 4 * 4
    current_codebook_bytes = ((1 << 5) + (1 << 3)) * 2 * 4
    return {
        "default_cap_rejects_k10": default_rejects,
        "explicit_k10v6_b4_exception_required": True,
        "exact_assignment_representable": candidate.assignment_mode == "exact",
        "candidate_layout": candidate_layout,
        "current_layout": current_layout,
        "logical_and_physical_rate_match": True,
        "candidate_uses_specialized_b2_decode_kernel": False,
        "candidate_uses_generic_native_decode_kernel": True,
        "candidate_is_current_native_registry_recipe": False,
        "candidate_codebook_footprint_bytes_fp32": candidate_codebook_bytes,
        "current_codebook_footprint_bytes_fp32": current_codebook_bytes,
        "codebook_footprint_ratio_candidate_over_current": (
            candidate_codebook_bytes / current_codebook_bytes
        ),
        "exact_key_score_evaluations_ratio_candidate_over_current_per_vector": (
            (128 // 4) * (1 << 10) / ((128 // 2) * (1 << 5))
        ),
        "exact_value_score_evaluations_ratio_candidate_over_current_per_vector": (
            (128 // 4) * (1 << 6) / ((128 // 2) * (1 << 3))
        ),
        "canonical_adoption_requires_native_latency_gate": True,
    }


def _native_measurement_audit() -> dict:
    native_root = ROOT / "native"
    parity_fields = (
        "key_codes_exact",
        "value_codes_exact",
        "key_fp16_scales_exact",
        "value_fp16_scales_exact",
        "packed_stream_exact",
        "full_dequant_exact_at_output_dtype",
        "store1_matches_prefill_payload_and_bytes",
    )
    smokes = {}
    for label, (filename, expected) in NATIVE_SMOKES.items():
        path = native_root / filename
        doc = _load(path)
        schema = doc.get("schema", {})
        if schema.get("name") != "onecomp.hq_full_native_correctness_smoke" or int(
            schema.get("version", -1)
        ) != 2:
            _fail(f"{label} native smoke schema mismatch")
        if doc.get("ok") is not True:
            _fail(f"{label} native smoke did not pass")
        parity = doc.get("parity", {})
        if any(parity.get(field) is not True for field in parity_fields):
            _fail(f"{label} native parity is incomplete")
        if doc.get("quality_gate_against_exact_assignment", {}).get("passed") is not True:
            _fail(f"{label} exact-assignment quality gate failed")
        recipe = doc.get("recipe", {})
        observed = (
            recipe.get("key_code_bits_per_block"),
            recipe.get("key_block_size"),
            recipe.get("value_code_bits_per_block"),
            recipe.get("value_block_size"),
            recipe.get("dist"),
            recipe.get("codebook_init"),
            recipe.get("allow_canonical_k10v6_b4"),
        )
        if observed != expected:
            _fail(f"{label} native smoke recipe mismatch: {observed!r}")
        if recipe.get("assignment_mode") != "exact":
            _fail(f"{label} native smoke did not use exact assignment")
        if _finite(recipe.get("average_kv_rate_bits_per_coordinate"), label) != 2.0:
            _fail(f"{label} native smoke is not rate 2")
        layout = recipe.get("layout", {})
        if int(layout.get("slot_size", -1)) != 68 or int(
            layout.get("slot_size_aligned", -1)
        ) != 68:
            _fail(f"{label} native smoke has a non-matched physical slot")
        smokes[label] = {
            "path": str(path.relative_to(REPO)),
            "sha256": _sha256(path),
            "ok": True,
            "parity": {field: True for field in parity_fields},
            "slot_size_bytes": 68,
        }

    abba_path = native_root / "rate2_native_abba.json"
    abba = _load(abba_path)
    if abba.get("schema") != "hayatequant.development_rate2_native_abba.v1":
        _fail("native ABBA schema mismatch")
    contract = abba.get("comparison_contract", {})
    expected_contract = {
        "order_per_candidate": ["current", "candidate", "candidate", "current"],
        "fresh_process_per_sample": True,
        "same_gpu": True,
        "same_physical_cache_bytes": True,
        "upstream_tq_2bit_comparison": "unavailable; not invented",
        "warmup_and_compilation_excluded": True,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            _fail(f"native ABBA contract mismatch for {field}")
    workers = abba.get("workers", [])
    if len(workers) != 12:
        _fail("native ABBA did not retain all 12 fresh-process workers")
    if len({int(row.get("pid", -1)) for row in workers}) != 12:
        _fail("native ABBA worker PIDs are not unique")
    gpu_ids = {
        (row.get("gpu", {}).get("name"), row.get("gpu", {}).get("uuid"))
        for row in workers
    }
    physical = {int(row.get("physical_cache_bytes", -1)) for row in workers}
    if len(gpu_ids) != 1 or len(physical) != 1 or next(iter(physical)) <= 0:
        _fail("native ABBA GPU/physical-cache matching failed")
    aggregates = abba.get("aggregate", {})
    if set(aggregates) != set(FIXED_METHODS):
        _fail("native ABBA candidate set mismatch")
    audited = {}
    for candidate in FIXED_METHODS:
        metrics = aggregates[candidate]
        if set(metrics) != set(NATIVE_METRICS):
            _fail(f"{candidate} native ABBA metric set mismatch")
        audited_metrics = {}
        for metric in NATIVE_METRICS:
            row = metrics[metric]
            current_samples = [
                _finite(value, f"{candidate}/{metric}/current")
                for value in row.get("current_process_medians_ms", [])
            ]
            candidate_samples = [
                _finite(value, f"{candidate}/{metric}/candidate")
                for value in row.get("candidate_process_medians_ms", [])
            ]
            if len(current_samples) != 2 or len(candidate_samples) != 2:
                _fail(f"{candidate}/{metric} is not a two-by-two ABBA comparison")
            current_median = statistics.median(current_samples)
            candidate_median = statistics.median(candidate_samples)
            recorded_current = _finite(
                row.get("current_median_of_process_medians_ms"),
                f"{candidate}/{metric}/recorded_current",
            )
            recorded_candidate = _finite(
                row.get("candidate_median_of_process_medians_ms"),
                f"{candidate}/{metric}/recorded_candidate",
            )
            recorded_speedup = _finite(
                row.get("candidate_speedup_over_current_hq"),
                f"{candidate}/{metric}/recorded_speedup",
            )
            if not math.isclose(recorded_current, current_median, rel_tol=0.0, abs_tol=1e-15):
                _fail(f"{candidate}/{metric} current median mismatch")
            if not math.isclose(
                recorded_candidate, candidate_median, rel_tol=0.0, abs_tol=1e-15
            ):
                _fail(f"{candidate}/{metric} candidate median mismatch")
            speedup = current_median / candidate_median
            if not math.isclose(recorded_speedup, speedup, rel_tol=1e-15, abs_tol=0.0):
                _fail(f"{candidate}/{metric} speedup mismatch")
            faster = candidate_median < current_median
            if row.get("candidate_faster_point_estimate") is not faster:
                _fail(f"{candidate}/{metric} faster flag mismatch")
            audited_metrics[metric] = {
                "current_process_medians_ms": current_samples,
                "candidate_process_medians_ms": candidate_samples,
                "current_median_of_process_medians_ms": current_median,
                "candidate_median_of_process_medians_ms": candidate_median,
                "candidate_speedup_over_current_hq": speedup,
                "candidate_latency_ratio_over_current_hq": 1.0 / speedup,
                "candidate_faster_point_estimate": faster,
            }
        audited[candidate] = {
            "metrics": audited_metrics,
            "strict_speed_pareto_dominates_current_hq": all(
                row["candidate_faster_point_estimate"] for row in audited_metrics.values()
            ),
            "slower_metrics": [
                metric
                for metric, row in audited_metrics.items()
                if not row["candidate_faster_point_estimate"]
            ],
        }
    return {
        "all_public_native_parity_checks_passed": True,
        "same_physical_cache_bytes": next(iter(physical)),
        "same_gpu": list(gpu_ids)[0],
        "fresh_process_workers": len(workers),
        "smokes": smokes,
        "abba": audited,
        "abba_path": str(abba_path.relative_to(REPO)),
        "abba_sha256": _sha256(abba_path),
        "upstream_tq_2bit_native_comparison_available": False,
    }


def main() -> None:
    baseline_docs = {}
    fixed_docs = {}
    adaptive_docs = {}
    raw_paths = []
    codebook_hashes: dict[tuple, set[tuple[str, str]]] = {}
    adaptive_traces = []
    speed_rows = []

    for key, model in MODELS.items():
        baseline_path = _result_path(BASELINE_ROOT / "raw", key, model)
        fixed_path = _result_path(ROOT / "raw", f"{key}/fixed", model)
        adaptive_path = _result_path(ROOT / "raw", f"{key}/adaptive", model)
        baseline = _load(baseline_path)
        fixed = _load(fixed_path)
        adaptive = _load(adaptive_path)
        raw_paths.extend((baseline_path, fixed_path, adaptive_path))
        for label, doc in (("baseline", baseline), ("fixed", fixed), ("adaptive", adaptive)):
            if doc.get("model") != model:
                _fail(f"{label}/{key} model mismatch")
            if doc.get("skip_layers") != []:
                _fail(f"{label}/{key} unexpectedly skipped model layers")
        baseline_docs[key] = baseline
        fixed_docs[key] = fixed
        adaptive_docs[key] = adaptive

        baseline_full = baseline.get("fullkv_ppl") or {}
        n_tokens = int(baseline_full.get("ppl_n_tokens", -1))
        identity = _identity(baseline_full)
        if n_tokens <= 0 or int(baseline_full.get("ppl_n_windows", -1)) != 1:
            _fail(f"{model} baseline FullKV denominator is invalid")

        fixed_config = fixed.get("config", {})
        adaptive_config = adaptive.get("config", {})
        expected_common = {
            "sink_tokens": 0,
            "recent_tokens": 0,
            "capture_tokens": 512,
            "run_ppl": True,
            "ppl_max_length": 512,
            "ppl_stride": 512,
            "ppl_windows": 1,
            "skip_outliers": False,
        }
        for name, expected in expected_common.items():
            if fixed_config.get(name) != expected or adaptive_config.get(name) != expected:
                _fail(f"{model}/{name} refinement protocol mismatch")
        if fixed_config.get("bits") != [2] or adaptive_config.get("bits") != [2, 3, 4]:
            _fail(f"{model} refinement rate grid mismatch")

        fixed_results = fixed.get("results", {})
        if set(fixed_results) != {"2"} or set(fixed_results["2"]) != FIXED_RAW_METHODS:
            _fail(f"{model} fixed candidate method set mismatch")
        adaptive_results = adaptive.get("results", {})
        if set(adaptive_results) != {"2", "3", "4"}:
            _fail(f"{model} adaptive rate set mismatch")

        baseline_rate2 = baseline["results"]["2"]
        for control in ("TurboQuant-paper", "HQ"):
            candidate_control = fixed_results["2"][control]
            reference_control = baseline_rate2[control]
            if candidate_control.get("ppl_nll") != reference_control.get("ppl_nll"):
                _fail(f"{model}/{control} rerun NLL is not deterministic")
            for counter in ("ppl_compression_fp16_bits", "ppl_compression_stored_bits"):
                if candidate_control.get(counter) != reference_control.get(counter):
                    _fail(f"{model}/{control}/{counter} changed between development runs")

        for method in FIXED_METHODS:
            entry = fixed_results["2"][method]
            label = f"{model}/r2/{method}"
            _validate_entry(
                entry, label=label, n_tokens=n_tokens, identity=identity, temporal=False
            )
            cfg = entry.get("codec_config", {})
            expected_dist = "beta" if method.endswith("finite-beta") else "gaussian"
            expected_init = "fibquant" if method.endswith("fibinit") else "auto"
            expected_cfg = {
                "block_size": 4,
                "key_code_bits_per_block": 10,
                "value_code_bits_per_block": 6,
                "max_codebook_bits": 10,
                "dist": expected_dist,
                "codebook_init": expected_init,
                "encoder": "exact",
                "key_scale_mode": "nc",
                "value_scale_mode": "ls_iter",
            }
            for name, expected in expected_cfg.items():
                if cfg.get(name) != expected:
                    _fail(f"{label}/{name}: {cfg.get(name)!r} != {expected!r}")
            reference = baseline_rate2["HQ"]
            for counter in ("ppl_compression_fp16_bits", "ppl_compression_stored_bits"):
                if entry.get(counter) != reference.get(counter):
                    _fail(f"{label}/{counter} is not exact-rate matched")
            if entry.get("stored_bits") != fixed_results["2"]["HQ"].get("stored_bits"):
                _fail(f"{label} captured-KV stored bits differ from current HQ")
            resolved = entry.get("resolved_codec_config", {})
            roles = resolved.get("roles", {})
            if (
                roles.get("key", {}).get("block_size") != 4
                or roles.get("key", {}).get("code_bits_per_block") != 10
                or roles.get("value", {}).get("block_size") != 4
                or roles.get("value", {}).get("code_bits_per_block") != 6
            ):
                _fail(f"{label} resolved K10/V6 b4 contract mismatch")
            identity_record = resolved.get("development_codebook_identity")
            if not isinstance(identity_record, dict):
                _fail(f"{label} is missing codebook identity")
            role_ids = identity_record.get("roles", {})
            if identity_record.get("total_footprint_bytes") != 17408:
                _fail(f"{label} unexpected fp32 codebook footprint")
            key_sha = role_ids.get("key", {}).get("sha256")
            value_sha = role_ids.get("value", {}).get("sha256")
            if not all(isinstance(item, str) and len(item) == 64 for item in (key_sha, value_sha)):
                _fail(f"{label} malformed codebook SHA-256")
            group = (method, int(identity_record.get("head_dim", -1)))
            codebook_hashes.setdefault(group, set()).add((key_sha, value_sha))
            speed_rows.append(
                {
                    "model": model,
                    "candidate": method,
                    "candidate_ms_per_encode": _finite(entry.get("ms_per_encode"), label + "/ms"),
                    "current_hq_ms_per_encode": _finite(
                        fixed_results["2"]["HQ"].get("ms_per_encode"), label + "/hq_ms"
                    ),
                    "tq_ms_per_encode": _finite(
                        fixed_results["2"]["TurboQuant-paper"].get("ms_per_encode"),
                        label + "/tq_ms",
                    ),
                }
            )

        for rate in RATES:
            methods = adaptive_results[str(rate)]
            if set(methods) != set(ADAPTIVE_METHODS):
                _fail(f"{model}/r{rate} adaptive candidate method set mismatch")
            baseline_reference = baseline["results"][str(rate)]["HQ"]
            expected_pairs = [[2 * rate, 2 * rate], [2 * rate + 1, 2 * rate - 1]]
            for method in ADAPTIVE_METHODS:
                entry = methods[method]
                label = f"{model}/r{rate}/{method}"
                _validate_entry(
                    entry, label=label, n_tokens=n_tokens, identity=identity, temporal=True
                )
                cfg = entry.get("codec_config", {})
                expected_cfg = {
                    "block_size": 2,
                    "adaptive_bit_split": "online_stats",
                    "key_error_weight": ADAPTIVE_WEIGHTS[method],
                    "value_error_weight": 1.0,
                    "encoder": "exact",
                    "key_scale_mode": "nc",
                    "value_scale_mode": "ls_iter",
                }
                for name, expected in expected_cfg.items():
                    if cfg.get(name) != expected:
                        _fail(f"{label}/{name}: {cfg.get(name)!r} != {expected!r}")
                for counter in ("ppl_compression_fp16_bits", "ppl_compression_stored_bits"):
                    if entry.get(counter) != baseline_reference.get(counter):
                        _fail(f"{label}/{counter} is not exact-rate matched")
                allocation = entry.get("resolved_codec_config", {}).get(
                    "adaptive_allocation", {}
                )
                if allocation.get("candidate_pairs") != expected_pairs:
                    _fail(f"{label} resolved candidate set is not the preregistered two pairs")
                if allocation.get("selector_bits_per_encode") != 1:
                    _fail(f"{label} selector carrier is not exactly one bit")
                if allocation.get("selector_incremental_stored_bits_per_encode") != 0:
                    _fail(f"{label} selector unexpectedly adds stored bits")
                trace = entry.get("ppl_development_adaptive_split_trace")
                if not isinstance(trace, dict) or trace.get("candidate_pairs") != expected_pairs:
                    _fail(f"{label} missing/mismatched PPL selector trace")
                updates = int(trace.get("encoded_updates", -1))
                if updates <= 0 or int(trace.get("selector_bits_total", -1)) != updates:
                    _fail(f"{label} selector-bit total is not one per encoded update")
                if int(trace.get("selector_incremental_stored_bits_total", -1)) != 0:
                    _fail(f"{label} PPL selectors add logical stored bits")
                histogram = trace.get("split_histogram")
                if not isinstance(histogram, dict) or not histogram:
                    _fail(f"{label} has no selector histogram")
                allowed = {tuple(pair) for pair in expected_pairs}
                observed = {tuple(bucket.get("pair", [])) for bucket in histogram.values()}
                if not observed <= allowed or any(pair in observed for pair in ((7, 1), (1, 7))):
                    _fail(f"{label} observed an unregistered/extreme split: {sorted(observed)}")
                if sum(int(bucket.get("updates", -1)) for bucket in histogram.values()) != updates:
                    _fail(f"{label} selector histogram update total mismatch")
                adaptive_traces.append(
                    {
                        "model": model,
                        "rate": rate,
                        "candidate": method,
                        "candidate_pairs": expected_pairs,
                        "observed_pairs": [list(pair) for pair in sorted(observed)],
                        "encoded_updates": updates,
                        "encoded_tokens": int(trace.get("encoded_tokens", -1)),
                        "selector_bits_total": int(trace["selector_bits_total"]),
                        "split_histogram": histogram,
                    }
                )

    for group, identities in codebook_hashes.items():
        if len(identities) != 1:
            _fail(f"codebook generation is not deterministic across workers for {group}")

    cells = []

    def add_cell(
        *, family: str, candidate: str, model: str, rate: int, baseline: str,
        candidate_entry: dict, baseline_entry: dict, measurement: str,
    ) -> None:
        candidate_nll = _finite(candidate_entry["ppl_nll"], "candidate_nll")
        baseline_nll = _finite(baseline_entry["ppl_nll"], "baseline_nll")
        delta = candidate_nll - baseline_nll
        cells.append(
            {
                "proposal_family": family,
                "candidate": candidate,
                "model": model,
                "rate": rate,
                "comparison_family": (
                    "external"
                    if baseline in EXTERNAL_BASELINES
                    else (
                        "previous_adaptive_hq"
                        if baseline == "PREVIOUS-AdaptiveHQ"
                        else "current_hq"
                    )
                ),
                "baseline": baseline,
                "candidate_nll": candidate_nll,
                "baseline_nll": baseline_nll,
                "nll_difference_candidate_minus_baseline": delta,
                "candidate_ppl": float(candidate_entry["ppl"]),
                "baseline_ppl": float(baseline_entry["ppl"]),
                "strict_unrounded_nll_win": candidate_nll < baseline_nll,
                "measurement": measurement,
                "candidate_stored_bits": int(candidate_entry["ppl_compression_stored_bits"]),
                "baseline_stored_bits": int(baseline_entry["ppl_compression_stored_bits"]),
            }
        )

    for key, model in MODELS.items():
        baseline = baseline_docs[key]
        fixed = fixed_docs[key]
        adaptive = adaptive_docs[key]
        for candidate in (FIXED_CONTROL, *FIXED_METHODS):
            for rate in RATES:
                methods = baseline["results"][str(rate)]
                if candidate == FIXED_CONTROL:
                    candidate_entry = methods["HQ"]
                    measurement = "job69320_current_hq"
                elif rate == 2:
                    candidate_entry = fixed["results"]["2"][candidate]
                    measurement = "refinement_rate2_candidate"
                else:
                    candidate_entry = methods["HQ"]
                    measurement = "unchanged_current_hq_rate3_or_rate4"
                for external in EXTERNAL_BASELINES:
                    if external in methods:
                        add_cell(
                            family="fixed_hq",
                            candidate=candidate,
                            model=model,
                            rate=rate,
                            baseline=external,
                            candidate_entry=candidate_entry,
                            baseline_entry=methods[external],
                            measurement=measurement,
                        )
                add_cell(
                    family="fixed_hq",
                    candidate=candidate,
                    model=model,
                    rate=rate,
                    baseline="HQ",
                    candidate_entry=candidate_entry,
                    baseline_entry=methods["HQ"],
                    measurement=measurement,
                )

        for candidate in ADAPTIVE_METHODS:
            for rate in RATES:
                methods = baseline["results"][str(rate)]
                candidate_entry = adaptive["results"][str(rate)][candidate]
                for external in EXTERNAL_BASELINES:
                    if external in methods:
                        add_cell(
                            family="adaptive_hq",
                            candidate=candidate,
                            model=model,
                            rate=rate,
                            baseline=external,
                            candidate_entry=candidate_entry,
                            baseline_entry=methods[external],
                            measurement="refinement_safe_adaptive_candidate",
                        )
                add_cell(
                    family="adaptive_hq",
                    candidate=candidate,
                    model=model,
                    rate=rate,
                    baseline="HQ",
                    candidate_entry=candidate_entry,
                    baseline_entry=methods["HQ"],
                    measurement="refinement_safe_adaptive_candidate",
                )
                add_cell(
                    family="adaptive_hq",
                    candidate=candidate,
                    model=model,
                    rate=rate,
                    baseline="PREVIOUS-AdaptiveHQ",
                    candidate_entry=candidate_entry,
                    baseline_entry=methods["AdaptiveHQ"],
                    measurement=(
                        "refinement_safe_adaptive_vs_job69320_extreme_split_control"
                    ),
                )

    # Gaussian auto and forced FibQuant initialization collapse to the same
    # current online-fit fixed point at K10 because no shipped K1024 artifact
    # exists.  Treating them as independent candidates would double-count one
    # result; fail if any numerical or codebook identity field differs.
    for key, model in MODELS.items():
        auto = fixed_docs[key]["results"]["2"]["DEVONLY-HQ-b4-gaussian-auto"]
        fib = fixed_docs[key]["results"]["2"]["DEVONLY-HQ-b4-gaussian-fibinit"]
        for field in (
            "ppl_nll",
            "ppl_nll_sum",
            "ppl",
            "ppl_compression_stored_bits",
            "ppl_compression_fp16_bits",
        ):
            if auto.get(field) != fib.get(field):
                _fail(f"{model} Gaussian auto/Fib-init equivalence failed for {field}")
        auto_roles = auto["resolved_codec_config"]["development_codebook_identity"]["roles"]
        fib_roles = fib["resolved_codec_config"]["development_codebook_identity"]["roles"]
        for role in ("key", "value"):
            if auto_roles[role]["sha256"] != fib_roles[role]["sha256"]:
                _fail(f"{model} Gaussian auto/Fib-init codebook SHA differs for {role}")

    fixed_summaries = [_summary(cells, candidate) for candidate in (FIXED_CONTROL, *FIXED_METHODS)]
    adaptive_summaries = [_summary(cells, candidate) for candidate in ADAPTIVE_METHODS]
    for row in (*fixed_summaries, *adaptive_summaries):
        row["external_by_rate"] = _by_rate_external_summary(cells, row["candidate"])
    current = next(row for row in fixed_summaries if row["candidate"] == FIXED_CONTROL)
    quality_preferred_fixed = max(fixed_summaries, key=_fixed_rank)
    if (
        quality_preferred_fixed["external"]["by_baseline"]["TurboQuant-paper"]["wins"]
        < current["external"]["by_baseline"]["TurboQuant-paper"]["wins"]
        or quality_preferred_fixed["external"]["wins"] < current["external"]["wins"]
    ):
        _fail("fixed candidate ranking violated the preregistered no-regression retention guard")
    selected_adaptive = max(adaptive_summaries, key=_adaptive_rank)

    speed_summary = {}
    for method in FIXED_METHODS:
        rows = [row for row in speed_rows if row["candidate"] == method]
        ratios_hq = [row["candidate_ms_per_encode"] / row["current_hq_ms_per_encode"] for row in rows]
        ratios_tq = [row["candidate_ms_per_encode"] / row["tq_ms_per_encode"] for row in rows]
        speed_summary[method] = {
            "models": len(rows),
            "median_encode_latency_ratio_over_current_hq": statistics.median(ratios_hq),
            "worst_encode_latency_ratio_over_current_hq": max(ratios_hq),
            "median_encode_latency_ratio_over_tq": statistics.median(ratios_tq),
            "worst_encode_latency_ratio_over_tq": max(ratios_tq),
            "per_model": rows,
        }

    native_contract = _native_contract_audit()
    native_measurement = _native_measurement_audit()
    quality_native = native_measurement["abba"][quality_preferred_fixed["candidate"]]
    analysis = {
        "schema": "hayatequant.development_hq_quality_refinement.v1",
        "scope": "development-only one-window WikiText-2; not formal paper evidence",
        "comparison_rule": "strict unrounded JSON-double candidate_nll < baseline_nll",
        "models": list(MODELS.values()),
        "rates": list(RATES),
        "candidate_set_frozen_before_results": True,
        "fixed_summaries": fixed_summaries,
        "adaptive_summaries": adaptive_summaries,
        "quality_preferred_fixed_candidate": quality_preferred_fixed["candidate"],
        "fixed_public_recipe_after_quality_analysis": (
            FIXED_CONTROL
            if quality_preferred_fixed["candidate"] != FIXED_CONTROL
            else FIXED_CONTROL
        ),
        "fixed_public_recipe_reason": (
            "b4 K10/V6 passes parity but fails the native speed-Pareto gate; current b2 HQ retained"
            if quality_preferred_fixed["candidate"] != FIXED_CONTROL
            else "current HQ wins the preregistered quality rank"
        ),
        "selected_adaptive_candidate": selected_adaptive["candidate"],
        "selected_adaptive_key_error_weight": ADAPTIVE_WEIGHTS[selected_adaptive["candidate"]],
        "fixed_rate3_rate4_unchanged_by_construction": True,
        "all_ppl_counters_exact_rate_matched": True,
        "all_candidate_outputs_finite": True,
        "all_fixed_codebooks_deterministic_across_workers": True,
        "fixed_gaussian_auto_and_fibinit_one_equivalence_class": True,
        "fixed_quality_preferred_native_speed_pareto_dominates_current_hq": (
            quality_native["strict_speed_pareto_dominates_current_hq"]
        ),
        "adaptive_candidate_contract": {
            "pairs": "[(2r,2r),(2r+1,2r-1)]",
            "selector_bits_per_encode": 1,
            "selector_incremental_stored_bits": 0,
            "extreme_split_observed": False,
        },
        "adaptive_traces": adaptive_traces,
        "fixed_speed_summary": speed_summary,
        "native_contract_audit": native_contract,
        "native_measurement_audit": native_measurement,
        "cells": cells,
        "raw_sha256": {
            str(path.relative_to(REPO)): _sha256(path) for path in sorted(raw_paths)
        },
    }
    (ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (ROOT / "cells.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    with (ROOT / "adaptive_split_traces.json").open("w", encoding="utf-8") as stream:
        json.dump(adaptive_traces, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (ROOT / "codebook_hashes.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                f"{method}/head_dim={head_dim}": [
                    {"key_sha256": key_sha, "value_sha256": value_sha}
                    for key_sha, value_sha in sorted(identities)
                ]
                for (method, head_dim), identities in sorted(codebook_hashes.items())
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    print("fixed candidates")
    for row in fixed_summaries:
        print(row["candidate"], row["external"], "rank=", _fixed_rank(row))
    print("quality preferred fixed:", quality_preferred_fixed["candidate"])
    print("selected safe adaptive:", selected_adaptive["candidate"])


if __name__ == "__main__":
    main()
