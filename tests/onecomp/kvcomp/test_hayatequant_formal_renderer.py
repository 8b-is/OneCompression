"""Fail-closed tests for the aggregate-to-LaTeX paper boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/render_hayatequant_formal_results.py"
SPEC = importlib.util.spec_from_file_location("hayatequant_formal_renderer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
AGGREGATOR_SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/aggregate_hayatequant_paper_shards.py"
AGGREGATOR_SPEC = importlib.util.spec_from_file_location(
    "hayatequant_formal_aggregator_for_renderer_tests", AGGREGATOR_SCRIPT
)
assert AGGREGATOR_SPEC is not None and AGGREGATOR_SPEC.loader is not None
aggregator = importlib.util.module_from_spec(AGGREGATOR_SPEC)
AGGREGATOR_SPEC.loader.exec_module(aggregator)

COMMIT = "a" * 40
ZERO_PROTOCOL_HASHES = {
    task: hashlib.sha256(task.encode("utf-8")).hexdigest() for task in renderer.ZERO_TASK_PRIMARY
}


def _source(name: str) -> dict:
    return {"path": f"/formal/{name}.json", "bytes": 1, "sha256": "b" * 64}


def _ppl_row(ppl: float, *, compressed: bool = True) -> dict:
    n_tokens = 1000
    nll = math.log(ppl)
    return {
        "ppl": ppl,
        "ppl_nll": nll,
        "ppl_nll_sum": nll * n_tokens,
        "ppl_n_tokens": n_tokens,
        "ppl_n_windows": 10,
        "distortion_key": 0.01,
        "distortion_value": 0.02,
        "stored_bits_per_coord": 3.0,
        "stored_bits": 4000 if compressed else None,
        "stored_coordinates": 1000 if compressed else None,
        "codec_config": {},
        "ppl_compression_ratio": 4.0 if compressed else 1.0,
        "ppl_compression_aggregation": (
            "exact_cache_bit_counters" if compressed else "fullkv_no_cache"
        ),
        "ppl_compression_fp16_bits": 16000 if compressed else None,
        "ppl_compression_stored_bits": 4000 if compressed else None,
        "ppl_teacher_forced_scoring_protocol": renderer.PPL_TEACHER_FORCED_PROTOCOL,
        "ppl_teacher_forced_temporal_dependency": None,
        "ppl_dataset_revision": renderer.WIKITEXT2_REVISION,
        "ppl_text_sha256": renderer.WIKITEXT2_TEXT_SHA256,
        "dataset_revision": renderer.WIKITEXT2_REVISION,
        "text_sha256": renderer.WIKITEXT2_TEXT_SHA256,
    }


def _block_ppl_row(ppl: float) -> dict:
    row = _ppl_row(ppl)
    vectors = 32
    identity_fields = 4
    identity_bits = identity_fields * 101
    key_code = vectors * 416
    key_scales = vectors * 32
    key_total = key_code + identity_bits + key_scales
    value_payload = vectors * 256
    value_scales = vectors * 16
    value_total = value_payload + value_scales
    total = key_total + value_total
    coordinates = 2 * vectors * 128
    row.update(
        {
            "ppl_teacher_forced_scoring_protocol": (
                renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
            ),
            "ppl_teacher_forced_temporal_dependency": (renderer.BLOCKQUANT_TEMPORAL_DEPENDENCY),
            "stored_bits": total,
            "stored_coordinates": coordinates,
            "stored_bits_per_coord": total / coordinates,
            "method_role": "external_quality_baseline",
            "external_quality_baseline": True,
            "paper_comparable": False,
            "paper_reproduction": False,
            "proposal_method": False,
            "speed_comparison": False,
            "random_seed_protocol": (
                "fixed rotation_seed=0 and codebook_seed=0 same-runtime quality row; "
                "not the paper five-random-seed aggregate"
            ),
            "stored_bit_accounting": {
                "scope": "fixed-width-data-plane-growth-only",
                "physical_packing_claim": False,
                "vectors_per_role": vectors,
                "head_dim": 128,
                "key": {
                    "code_bits": key_code,
                    "payload_bits": key_code + identity_bits,
                    "fp16_scale_bits": key_scales,
                    "scale_fields_per_vector": 2,
                    "fixed_channel_identity_bits": 0,
                    "dynamic_channel_identity_bits": identity_bits,
                    "dynamic_channel_identity_fields": identity_fields,
                    "dynamic_channel_identity_coding": "combinadic_rank_fixed_width_v1",
                    "total_bits": key_total,
                    "bits_per_coordinate": key_total / float(vectors * 128),
                },
                "value": {
                    "payload_bits": value_payload,
                    "fp16_scale_bits": value_scales,
                    "scale_fields_per_vector": 1,
                    "total_bits": value_total,
                    "bits_per_coordinate": value_total / float(vectors * 128),
                },
                "total_bits": total,
                "excluded_global_state": [
                    "dense Haar rotations",
                    "data-free codebooks",
                    "Python control/container bytes",
                    "allocator and unpacked research tensor bytes",
                ],
            },
        }
    )
    return row


def _proposal_ppl_row(ppl: float, method: str) -> dict:
    row = _ppl_row(ppl)
    row.update(copy.deepcopy(renderer.HAYATE_FORMAL_PROVENANCE[method]))
    protocol, dependency = renderer._expected_teacher_forced_contract(method)
    row["ppl_teacher_forced_scoring_protocol"] = protocol
    row["ppl_teacher_forced_temporal_dependency"] = dependency
    return row


def _add_ppl_payload_evidence(row: dict) -> dict:
    row["payload_match_vs_turboquant"] = {
        "exact_integer_counters": True,
        "ppl_compression_fp16_bits": row["ppl_compression_fp16_bits"],
        "ppl_compression_stored_bits": row["ppl_compression_stored_bits"],
    }
    return row


def _ppl_axis() -> dict:
    ratios = {
        proposal: {rate: [0.8] * 7 for rate in renderer.RATES}
        for proposal in renderer.PROPOSAL_METHODS
    }
    models = {}
    for model_index, model in enumerate(renderer.PAPER_MODELS):
        rates = {}
        for rate in renderer.RATES:
            tq = 10.0 + model_index / 10.0
            rates[rate] = {
                "TurboQuant-paper": _ppl_row(tq),
                "FibQuant-paper": _add_ppl_payload_evidence(_ppl_row(tq * 1.01)),
                "HQ": _add_ppl_payload_evidence(
                    _proposal_ppl_row(tq * ratios["HQ"][rate][model_index], "HQ")
                ),
                "AdaptiveHQ": _add_ppl_payload_evidence(
                    _proposal_ppl_row(tq * ratios["AdaptiveHQ"][rate][model_index], "AdaptiveHQ")
                ),
            }
            if rate == "3":
                rates[rate]["BlockQuant-paper"] = _block_ppl_row(tq * 1.01)
        models[model] = {
            "fullkv": _ppl_row(9.0 + model_index / 10.0, compressed=False),
            "rates": rates,
            "official_complete": True,
        }
    primary = {}
    for proposal in renderer.PROPOSAL_METHODS:
        primary[proposal] = {}
        for rate in renderer.RATES:
            values = [
                math.exp(
                    models[model]["rates"][rate][proposal]["ppl_nll"]
                    - models[model]["rates"][rate]["TurboQuant-paper"]["ppl_nll"]
                )
                for model in renderer.PAPER_MODELS
            ]
            logs = [math.log(value) for value in values]
            mean = statistics.fmean(logs)
            stderr = statistics.stdev(logs) / math.sqrt(len(logs))
            critical = renderer._student_t_critical(7)
            simultaneous_critical = renderer._bonferroni_three_rate_t_critical(7)
            primary[proposal][rate] = {
                "proposal": proposal,
                "estimand": "geometric_mean_within_model_ppl_ratio_proposal_over_turboquant",
                "student_t_statistical_unit": "fixed_checkpoint",
                "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
                "population_generalization": False,
                "n_models": 7,
                "mean_log_ppl_ratio": round(mean, 12),
                "mean_log_ppl_ratio_standard_error": round(stderr, 12),
                "geometric_mean_ppl_ratio": round(math.exp(mean), 10),
                "relative_ppl_change_percent": round(100.0 * (math.exp(mean) - 1.0), 8),
                "student_t_ci95_geometric_mean_ratio": [
                    round(math.exp(mean - critical * stderr), 10),
                    round(math.exp(mean + critical * stderr), 10),
                ],
                "nominal_interval_status": "secondary_robustness_not_primary_win_gate",
                "bonferroni_simultaneous_familywise95_geometric_mean_ratio": [
                    round(math.exp(mean - simultaneous_critical * stderr), 10),
                    round(math.exp(mean + simultaneous_critical * stderr), 10),
                ],
                "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
                "familywise_alpha": 0.05,
                "per_rate_two_sided_alpha": 0.05 / 3.0,
                "per_model_ratio": {
                    model: round(value, 10) for model, value in zip(renderer.PAPER_MODELS, values)
                },
                "claim_rule": "interval_is_secondary_robustness_only",
            }
    return {
        "benchmark": "WikiText-2 perplexity",
        "commit": COMMIT,
        "models": models,
        "secondary_interval_robustness_vs_turboquant": primary,
        "primary_metric_note": "within-checkpoint ratios",
        "inference_scope": {
            "student_t_statistical_unit": "fixed_checkpoint",
            "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
            "interval_interpretation": (
                "descriptive_uncertainty_across_the_fixed_checkpoint_panel"
            ),
            "population_generalization": False,
        },
        "compression_accounting": dict(renderer.PPL_COMPRESSION_ACCOUNTING),
        "evaluation_config": dict(renderer.PPL_EVALUATION_CONFIG),
        "teacher_forced_scoring_contract": {
            "protocol": renderer.PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": False,
            "method_overrides": {
                "BlockQuant-paper": {
                    "protocol": renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": renderer.BLOCKQUANT_TEMPORAL_DEPENDENCY,
                },
                "AdaptiveHQ": {
                    "protocol": renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": renderer.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY,
                },
            },
        },
        "dataset_revision": renderer.WIKITEXT2_REVISION,
        "text_sha256": renderer.WIKITEXT2_TEXT_SHA256,
        "official_complete": True,
        "sources": [_source("ppl")],
    }


def _zero_task(n: int, correct: int, primary_metric: str) -> dict:
    score = correct / n
    return {
        "n": n,
        "correct": correct,
        "correct_norm": correct,
        "acc": round(score, 8),
        "acc_norm": round(score, 8),
        "primary_metric": primary_metric,
        "primary_score": round(score, 8),
        "dataset_revision": "revision",
        "split": "test",
        "selection_order": "validated_dataset_order",
        "coverage": [0, n],
        "audit_records_validated": n,
        "teacher_forced_scoring_protocol": renderer.PPL_TEACHER_FORCED_PROTOCOL,
        "teacher_forced_temporal_dependency": None,
        "shards": ["/formal/zero.json"],
    }


def _zero_method(correct_by_task: dict[str, int], *, compressed: bool = True) -> dict:
    tasks = {
        task: _zero_task(renderer.ZERO_TASK_SIZES[task], correct_by_task[task], primary_metric)
        for task, primary_metric in renderer.ZERO_TASK_PRIMARY.items()
    }
    for task, row in tasks.items():
        n = renderer.ZERO_TASK_SIZES[task]
        row.update(
            {
                "protocol_input_hash_schema": renderer.ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA,
                "protocol_input_sha256": ZERO_PROTOCOL_HASHES[task],
                "compression_ratio": 4.0 if compressed else 1.0,
                "compression_aggregation": (
                    "exact_cache_bit_counters" if compressed else "fullkv_no_cache"
                ),
                "compression_fp16_bits": n * 1600 if compressed else None,
                "compression_stored_bits": n * 400 if compressed else None,
            }
        )
    total_n = sum(renderer.ZERO_TASK_SIZES.values())
    return {
        "tasks": tasks,
        "compression_ratio": 4.0 if compressed else 1.0,
        "compression_aggregation": (
            "exact_cache_bit_counters" if compressed else "fullkv_no_cache"
        ),
        "compression_fp16_bits": total_n * 1600 if compressed else None,
        "compression_stored_bits": total_n * 400 if compressed else None,
        "avg_acc": round(statistics.fmean(row["acc"] for row in tasks.values()), 8),
        "avg_acc_norm": round(statistics.fmean(row["acc_norm"] for row in tasks.values()), 8),
        "avg_primary_score": round(
            statistics.fmean(row["primary_score"] for row in tasks.values()), 8
        ),
        "n_total": sum(renderer.ZERO_TASK_SIZES.values()),
        "official_complete": True,
    }


def _block_zero_method(source: dict) -> dict:
    row = copy.deepcopy(source)
    for task in row["tasks"].values():
        task["teacher_forced_scoring_protocol"] = (
            renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
        )
        task["teacher_forced_temporal_dependency"] = renderer.BLOCKQUANT_TEMPORAL_DEPENDENCY
    return row


def _paired_zero_task(n: int, tq_correct: int, hq_correct: int, metric: str) -> dict:
    hq_only = max(0, hq_correct - tq_correct)
    tq_only = max(0, tq_correct - hq_correct)
    both_correct = min(tq_correct, hq_correct)
    both_wrong = n - both_correct - hq_only - tq_only
    delta = (hq_only - tq_only) / n
    variance = ((hq_only + tq_only) - n * delta * delta) / (n - 1)
    stderr = math.sqrt(max(0.0, variance / n))
    low = max(-1.0, delta - 1.959963984540054 * stderr)
    high = min(1.0, delta + 1.959963984540054 * stderr)
    return {
        "metric": metric,
        "statistical_unit": "example_within_fixed_task_and_checkpoint",
        "task_and_checkpoint_status": "fixed_not_randomly_sampled",
        "population_generalization": False,
        "n_paired": n,
        "hq_correct": hq_correct,
        "turboquant_correct": tq_correct,
        "delta_accuracy_hq_minus_turboquant": round(delta, 10),
        "delta_percentage_points_hq_minus_turboquant": round(100.0 * delta, 8),
        "paired_standard_error": round(stderr, 10),
        "paired_normal_ci95": [round(low, 10), round(high, 10)],
        "interval_status": "nominal_per_task_not_an_across_rate_familywise_interval",
        "discordant": {
            "hq_only_correct": hq_only,
            "turboquant_only_correct": tq_only,
        },
        "concordant": {"both_correct": both_correct, "both_wrong": both_wrong},
        "mcnemar_exact_two_sided_p": 1.0,
        "claim_rule": "claim_improvement_only_if_ci95_excludes_zero",
    }


def _zero_pair(proposal: str, direction: int) -> tuple[dict, dict, dict]:
    tq_counts = {task: n // 2 for task, n in renderer.ZERO_TASK_SIZES.items()}
    hq_counts = {
        task: count + direction * max(1, renderer.ZERO_TASK_SIZES[task] // 20)
        for task, count in tq_counts.items()
    }
    tq = _zero_method(tq_counts)
    hq = _zero_method(hq_counts)
    hq.update(copy.deepcopy(renderer.HAYATE_FORMAL_PROVENANCE[proposal]))
    if proposal == "AdaptiveHQ":
        for task in hq["tasks"].values():
            task["teacher_forced_scoring_protocol"] = (
                renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
            )
            task["teacher_forced_temporal_dependency"] = renderer.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY
    comparisons = {}
    for task, primary_metric in renderer.ZERO_TASK_PRIMARY.items():
        metric = "correct_norm" if primary_metric == "acc_norm" else "correct"
        comparisons[task] = _paired_zero_task(
            renderer.ZERO_TASK_SIZES[task], tq_counts[task], hq_counts[task], metric
        )
        hq["tasks"][task]["paired_vs_turboquant"] = comparisons[task]
    macro_delta = statistics.fmean(
        row["delta_accuracy_hq_minus_turboquant"] for row in comparisons.values()
    )
    macro_stderr = math.sqrt(
        sum(row["paired_standard_error"] ** 2 for row in comparisons.values())
    ) / len(comparisons)
    paired = {
        "proposal": proposal,
        "estimand": "unweighted five-task macro of each task primary metric",
        "statistical_unit": "paired_example_within_each_fixed_task_and_checkpoint",
        "task_panel": "five_fixed_tasks_not_a_random_task_population",
        "population_generalization": False,
        "delta_macro_proposal_minus_turboquant": round(macro_delta, 10),
        "delta_macro_percentage_points_proposal_minus_turboquant": round(100.0 * macro_delta, 8),
        "stratified_paired_standard_error": round(macro_stderr, 10),
        "stratified_paired_normal_ci95": [
            round(max(-1.0, macro_delta - 1.959963984540054 * macro_stderr), 10),
            round(min(1.0, macro_delta + 1.959963984540054 * macro_stderr), 10),
        ],
        "interval_status": "secondary_robustness_not_primary_win_gate",
        "per_task": comparisons,
        "claim_rule": "interval_is_secondary_robustness_only",
    }
    return tq, hq, paired


def _zeroshot_axis() -> dict:
    models = {}
    for model in renderer.PAPER_MODELS:
        base_counts = {task: n // 2 for task, n in renderer.ZERO_TASK_SIZES.items()}
        rates = {"full": {"FullKV": _zero_method(base_counts, compressed=False)}}
        for rate in renderer.RATES:
            tq, hq, paired_hq = _zero_pair("HQ", 1)
            _, adaptive, paired_adaptive = _zero_pair("AdaptiveHQ", 1)
            fib = copy.deepcopy(tq)
            for matched in (fib, hq, adaptive):
                matched["payload_match_vs_turboquant"] = {
                    "exact_integer_counters": True,
                    "compression_fp16_bits": tq["compression_fp16_bits"],
                    "compression_stored_bits": tq["compression_stored_bits"],
                }
            rates[rate] = {
                "TurboQuant-paper": tq,
                "FibQuant-paper": fib,
                "HQ": hq,
                "AdaptiveHQ": adaptive,
                "paired_proposals_vs_turboquant": {
                    "HQ": paired_hq,
                    "AdaptiveHQ": paired_adaptive,
                },
            }
            if rate == "3":
                rates[rate]["BlockQuant-paper"] = _block_zero_method(tq)
        models[model] = {"rates": rates, "official_complete": True}
    primary = {}
    for proposal in renderer.PROPOSAL_METHODS:
        primary[proposal] = {}
        for rate in renderer.RATES:
            values = [
                models[model]["rates"][rate]["paired_proposals_vs_turboquant"][proposal][
                    "delta_macro_proposal_minus_turboquant"
                ]
                for model in renderer.PAPER_MODELS
            ]
            mean = statistics.fmean(values)
            stderr = statistics.stdev(values) / math.sqrt(len(values))
            nominal_critical = renderer._student_t_critical(7)
            simultaneous_critical = renderer._bonferroni_three_rate_t_critical(7)
            primary[proposal][rate] = {
                "proposal": proposal,
                "estimand": "mean_fixed_checkpoint_five_task_macro_paired_delta",
                "student_t_statistical_unit": "fixed_checkpoint",
                "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
                "population_generalization": False,
                "n_checkpoints": 7,
                "mean_delta": round(mean, 12),
                "mean_delta_standard_error": round(stderr, 12),
                "nominal_student_t_ci95": [
                    round(max(-1.0, mean - nominal_critical * stderr), 12),
                    round(min(1.0, mean + nominal_critical * stderr), 12),
                ],
                "nominal_interval_status": "secondary_robustness_not_primary_win_gate",
                "bonferroni_simultaneous_familywise95_ci": [
                    round(max(-1.0, mean - simultaneous_critical * stderr), 12),
                    round(min(1.0, mean + simultaneous_critical * stderr), 12),
                ],
                "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
                "familywise_alpha": 0.05,
                "per_rate_two_sided_alpha": 0.05 / 3.0,
                "claim_rule": "interval_is_secondary_robustness_only",
            }
    return {
        "benchmark": "corrected_zero_shot_five_task",
        "schema_version": 1,
        "aggregation": "paired exact coverage",
        "sample_seed": 2027,
        "commit": COMMIT,
        "models": models,
        "secondary_interval_robustness_vs_turboquant": primary,
        "inference_scope": {
            "within_checkpoint_paired_unit": ("example_within_each_fixed_task_and_checkpoint"),
            "task_panel": "five_fixed_tasks_not_a_random_task_population",
            "cross_checkpoint_student_t_unit": "fixed_checkpoint",
            "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
            "population_generalization": False,
        },
        "teacher_forced_scoring_contract": {
            "protocol": renderer.PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": False,
            "method_overrides": {
                "BlockQuant-paper": {
                    "protocol": renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": renderer.BLOCKQUANT_TEMPORAL_DEPENDENCY,
                },
                "AdaptiveHQ": {
                    "protocol": renderer.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": renderer.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY,
                },
            },
        },
        "compression_accounting": dict(renderer.ZEROSHOT_COMPRESSION_ACCOUNTING),
        "pinned_source_prompt_audit": {
            "input_manifest": {
                "path": "/formal/paper-inputs.json",
                "sha256": "c" * 64,
            },
            "protocol_input_hash_schema": renderer.ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA,
            "scope": renderer.ZEROSHOT_SOURCE_AUDIT_SCOPE,
            "tasks": {
                task: {
                    "manifest_key": renderer.ZERO_TASK_MANIFEST_KEYS[task],
                    "rows": renderer.ZERO_TASK_SIZES[task],
                    "full_split_protocol_input_sha256": ZERO_PROTOCOL_HASHES[task],
                    "source_files": [
                        {
                            "path": f"/formal/{task}.arrow",
                            "bytes": renderer.ZERO_TASK_SIZES[task],
                            "sha256": hashlib.sha256(f"source-{task}".encode("utf-8")).hexdigest(),
                        }
                    ],
                }
                for task in renderer.ZERO_TASK_PRIMARY
            },
        },
        "official_complete": True,
        "sources": [_source("zeroshot")],
    }


def _mechanism_axis() -> dict:
    ratios = {"2": (0.8, 0.8), "3": (0.8, 1.2), "4": (1.2, 1.2)}
    models = {}
    for model_index, model in enumerate(renderer.LONG_MODELS):
        rates = {}
        for rate in renderer.RATES:
            scalar = _ppl_row(10.0)
            block = _ppl_row(10.0 * ratios[rate][model_index])
            rates[rate] = {
                "methods": {
                    "TurboQuant-paper": _ppl_row(11.0),
                    "HayateQuant-scalar-matched": scalar,
                    "HayateQuant-blockvq-matched": block,
                    "HQ": _proposal_ppl_row(9.0, "HQ"),
                    "AdaptiveHQ": _proposal_ppl_row(8.5, "AdaptiveHQ"),
                },
                "causal_block_vq_ratio_block_over_scalar": round(
                    math.exp(block["ppl_nll"] - scalar["ppl_nll"]), 10
                ),
                "control_contract": "block_size only",
                "applicability": {
                    "single_lever_causal_block_vq_contrast": [
                        "HayateQuant-scalar-matched",
                        "HayateQuant-blockvq-matched",
                    ],
                    "HQ": "contextual formal proposal row; not part of the single-lever contrast",
                    "AdaptiveHQ": (
                        "contextual formal proposal row; causal selector changes allocation "
                        "and is not part of the single-lever contrast"
                    ),
                },
            }
        models[model] = {"rates": rates, "official_complete": True}
    return {
        "benchmark": "HayateQuant block-VQ mechanism ablation",
        "commit": COMMIT,
        "models": models,
        "scope": "two fixed model families, complete WikiText-2 raw test",
        "primary_contrast": ("HayateQuant-blockvq-matched vs HayateQuant-scalar-matched"),
        "proposal_rows_in_scope": list(renderer.PROPOSAL_METHODS),
        "inference_scope": {
            "checkpoint_panel": "two_preselected_family_correlated_checkpoints_not_iid",
            "population_generalization": False,
            "reporting": "descriptive_geometric_mean_only",
            "uncertainty_interval": None,
            "directional_claim_allowed": False,
            "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
        },
        "compression_accounting": dict(renderer.PPL_COMPRESSION_ACCOUNTING),
        "teacher_forced_scoring_contract": {
            "protocol": renderer.PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": True,
        },
        "official_complete": True,
        "sources": [_source("mechanism")],
    }


def _adaptive_runtime() -> dict:
    return {
        "schema_version": 1,
        "policy_version": renderer.ADAPTIVE_HQ_POLICY_VERSION,
        "encoded_updates": 1,
        "encoded_tokens": 1,
        "selector_bits_total": 1,
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": renderer.ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
        "split_histogram": {
            "6:6": {
                "pair": [6, 6],
                "updates": 1,
                "tokens": 1,
                "vectors": 1,
            }
        },
    }


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("policy_version",), "fabricated-policy"),
        (("selector_carrier_version",), "fabricated-carrier"),
        (("encoded_updates",), True),
        (("selector_bits_total",), True),
        (("selector_bits_total",), 2),
        (("split_histogram", "6:6", "updates"), 0),
        (("split_histogram", "6:6", "vectors"), 0),
    ],
)
def test_renderer_rejects_noncurrent_or_inconsistent_adaptive_runtime(path, bad_value):
    runtime = copy.deepcopy(_adaptive_runtime())
    target = runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(renderer.RenderError):
        renderer._validate_adaptive_hq_runtime(  # pylint: disable=protected-access
            runtime, "adversarial runtime"
        )


def _long_accounting(method: str, sample_ids: list[str]) -> dict:
    ledger = []
    for sample_id in sample_ids:
        item = {
            "sample_id": sample_id,
            "input_tokens": 128,
            "max_new_tokens": 16,
            "generated_tokens": 8,
            "compression_aggregation": (
                "fullkv_no_cache" if method == "FullKV" else "exact_cache_bit_counters"
            ),
            "compression_fp16_bits": None if method == "FullKV" else 1600,
            "compression_stored_bits": None if method == "FullKV" else 400,
            "adaptive_hq_runtime": (_adaptive_runtime() if method == "AdaptiveHQ" else None),
        }
        ledger.append(item)
    result = {
        "compression_ledger": ledger,
        "compression_observations": len(sample_ids),
        "compression_aggregation": (
            "fullkv_no_cache"
            if method == "FullKV"
            else "sum_exact_per_generation_cache_bit_counters"
        ),
        "compression_fp16_bits": None if method == "FullKV" else 1600 * len(sample_ids),
        "compression_stored_bits": None if method == "FullKV" else 400 * len(sample_ids),
        "adaptive_hq_runtime": None,
    }
    if method == "AdaptiveHQ":
        result["adaptive_hq_runtime"] = {
            "schema_version": 1,
            "policy_version": renderer.ADAPTIVE_HQ_POLICY_VERSION,
            "encoded_updates": len(sample_ids),
            "encoded_tokens": len(sample_ids),
            "selector_bits_total": len(sample_ids),
            "selector_incremental_stored_bits_total": 0,
            "selector_carrier_version": renderer.ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "split_histogram": {
                "6:6": {
                    "pair": [6, 6],
                    "updates": len(sample_ids),
                    "tokens": len(sample_ids),
                    "vectors": len(sample_ids),
                }
            },
        }
    if method == "TurboQuant-paper":
        result["matched_rate_reference"] = {
            "method": "TurboQuant-paper",
            "n_samples": len(sample_ids),
        }
    elif method in {"FibQuant-paper", *renderer.PROPOSAL_METHODS}:
        result["matched_rate_vs_turboquant"] = {
            "exact_integer_cross_products": True,
            "matched_sample_universe": True,
            "input_tokens_and_max_budget_identical": True,
            "generated_lengths_forced_equal": False,
            "generated_length_difference_count": 0,
            "n_samples": len(sample_ids),
        }
    return result


def _ruler_axis() -> dict:
    models = {}
    generation_stop_protocols = {
        model: {
            "eos_token_ids": [7, 8],
            "eos_token_id_source": "model.generation_config.eos_token_id",
            "stop_conditions": [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
            ],
        }
        for model in renderer.LONG_MODELS
    }
    model_wrapper_protocols = {
        model: {
            "prompt_wrappers": [renderer.RULER_CHAT_WRAPPER],
            "chat_template_kwargs": dict(renderer.RULER_CHAT_TEMPLATE_KWARGS),
            "protocol_scope": renderer.RULER_PROTOCOL_SCOPE,
            "reference_model_wrapper_exact": False,
        }
        for model in renderer.LONG_MODELS
    }
    for model_index, model in enumerate(renderer.LONG_MODELS):
        models[model] = {}
        for context in renderer.RULER_CONTEXTS:
            sample_ids = [f"ruler-{index}" for index in range(renderer.RULER_ROWS_PER_RUN)]
            models[model][context] = {}
            for method_index, method in enumerate(renderer.METHODS):
                score = 60.0 + model_index + 5.0 * method_index
                models[model][context][method] = {
                    "score": score,
                    "score_unrounded": score,
                    "n": renderer.RULER_ROWS_PER_RUN,
                    "combined_dataset_sha256": "d" * 64,
                    "generation_manifest_sha256": str(model_index + 1) * 64,
                    "model_max_position_embeddings": (
                        renderer.RULER_MODEL_MAX_POSITION_EMBEDDINGS[model]
                    ),
                    "max_observed_prompt_plus_max_generation_tokens": int(context) - 1,
                    "generation_stop_protocol": copy.deepcopy(generation_stop_protocols[model]),
                    "model_wrapper_protocol": copy.deepcopy(model_wrapper_protocols[model]),
                    "audit_records_validated": renderer.RULER_ROWS_PER_RUN,
                    "source": "/formal/ruler.json",
                }
                models[model][context][method].update(_long_accounting(method, sample_ids))
    return {
        "schema_version": 1,
        "source_result_schema_version": renderer.RULER_SOURCE_RESULT_SCHEMA_VERSION,
        "benchmark": renderer.RULER_PROTOCOL_LABEL,
        "aggregation": renderer.RULER_AGGREGATION,
        "context_semantics": renderer.RULER_CONTEXT_SEMANTICS,
        "capacity_check": renderer.RULER_CAPACITY_CHECK,
        "protocol_scope": renderer.RULER_PROTOCOL_SCOPE,
        "reference_model_wrapper_exact": False,
        "protocol_complete": True,
        "samples_per_task": renderer.RULER_SAMPLES_PER_TASK,
        "sampling_protocol": renderer.RULER_SAMPLING_PROTOCOL,
        "canonical_nvidia_samples_per_task": (renderer.RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK),
        "canonical_nvidia_full_eval": False,
        "commit": COMMIT,
        "models": models,
        "generation_stop_protocols": generation_stop_protocols,
        "model_wrapper_protocols": model_wrapper_protocols,
        "official_complete": True,
        "audit_records_validated": renderer.RULER_AXIS_AUDIT_RECORDS,
        "sources": [_source("ruler")],
    }


def _first_ruler_row(document: dict) -> dict:
    return document["ruler"]["models"][renderer.LONG_MODELS[0]][renderer.RULER_CONTEXTS[0]][
        renderer.METHODS[0]
    ]


def _longbench_axis() -> dict:
    models = {}
    generation_stop_protocols = {
        model: {
            "eos_token_ids": [7, 8],
            "eos_token_id_source": "model.generation_config.eos_token_id",
            "samsum_newline_token_id": 42,
            "stop_conditions": [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
                "samsum_newline_token_id_for_samsum_only",
            ],
        }
        for model in renderer.LONG_MODELS
    }
    for model_index, model in enumerate(renderer.LONG_MODELS):
        methods = {}
        for method_index, method in enumerate(renderer.METHODS):
            score = 50.0 + model_index + method_index
            all_sample_ids = []
            per_task = {}
            for task, count in renderer.LONGBENCH_COUNTS.items():
                task_sample_ids = [f"{task}:{index}" for index in range(count)]
                all_sample_ids.extend(task_sample_ids)
                task_row = {
                    "score_sum_unrounded": score * count,
                    "score_unrounded": score,
                    "score_displayed": round(score, 2),
                    "n": count,
                    "audit_records_validated": count,
                    "data_zip_sha256": renderer.LONGBENCH_ARCHIVE_SHA256,
                    "truncated_examples": (renderer.LONGBENCH_EXPECTED_TRUNCATIONS[model][task]),
                    "prompt_budget": renderer.LONGBENCH_PROMPT_BUDGETS[model][task],
                    "prompt_audit_signature": "f" * 64,
                    "generation_stop_protocol": copy.deepcopy(generation_stop_protocols[model]),
                    "source": "/formal/longbench.json",
                }
                task_row.update(_long_accounting(method, task_sample_ids))
                per_task[task] = task_row
            methods[method] = {
                "score": round(score, 2),
                "score_unrounded": score,
                "n": renderer.LONGBENCH_TOTAL,
                "per_task": per_task,
                "data_zip_sha256": renderer.LONGBENCH_ARCHIVE_SHA256,
                "released_data_complete": True,
            }
            methods[method].update(_long_accounting(method, all_sample_ids))
        models[model] = methods
    return {
        "benchmark": "LongBench-E",
        "commit": COMMIT,
        "models": models,
        "official_primary_reporting": "per_task_three_length_bucket_scores",
        "reported_scalar": "derived_unweighted_macro_over_task_scores",
        "protocol_scope": renderer.LONGBENCH_PROTOCOL_SCOPE,
        "reference_model_wrapper_exact": False,
        "prompt_budgets": copy.deepcopy(renderer.LONGBENCH_PROMPT_BUDGETS),
        "expected_truncations": copy.deepcopy(renderer.LONGBENCH_EXPECTED_TRUNCATIONS),
        "generation_stop_protocols": generation_stop_protocols,
        "truncation_policy": (
            "official middle truncation at an explicit model-specific raw-prompt "
            "budget; post-wrapper prompt plus task generation is capacity checked"
        ),
        "released_data_complete": True,
        "sources": [_source("longbench")],
    }


def _math_axis() -> dict:
    models = {}
    for model_index, model in enumerate(renderer.LONG_MODELS):
        methods = {}
        for method_index, method in enumerate(renderer.METHODS):
            correct = 240 + model_index * 10 + method_index
            estimate = correct / 500
            low, high = renderer._normal_binomial_ci(estimate, 500)
            methods[method] = {
                "pass@1": round(estimate, 8),
                "pass@1_sum": correct,
                "correct": correct,
                "pass@1_stderr": 0.0,
                "pass@1_ci95": [round(low, 8), round(high, 8)],
                "n": 500,
                "coverage": [0, 500],
                "audit_records_validated": 500,
                "shards": ["/formal/math.json"],
            }
            methods[method].update(
                _long_accounting(method, [f"math:{index}" for index in range(500)])
            )
        models[model] = methods
    return {
        "benchmark": "MATH-500",
        "commit": COMMIT,
        "grader": "Math-Verify 0.9.0",
        "protocol_scope": renderer.MATH500_PROTOCOL_SCOPE,
        "official_reference_gold_serialization_exact": False,
        "gold_serialization": renderer.MATH500_GOLD_SERIALIZATION,
        "aggregation": "exact",
        "few_shot_protocol": {
            "protocol_id": renderer.MATH500_PROTOCOL_ID,
            "protocol_origin": "adapted_composite",
            "math_verify_task_source_commit": renderer.MATH_VERIFY_TASK_SOURCE_COMMIT,
            "lighteval_prompt_manager_reference_version": (
                renderer.MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
            ),
            "lighteval_prompt_manager_source_commit": (
                renderer.MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT
            ),
            "upstream_lighteval_builtin_task_exact": False,
            "upstream_math_verify_custom_task_exact": False,
            "upstream_prompt_generation_components_exact": True,
            "upstream_component_pair_executable": False,
            "balanced_sampler_seed": 0,
            "few_shot_count": 4,
            "few_shot_order_one_based": [4, 3, 2, 1],
            "few_shot_content_sha256": renderer.MATH500_FEWSHOT_CONTENT_SHA256,
            "few_shot_rendering": renderer.MATH500_FEWSHOT_RENDERING,
            "use_chat_template": False,
        },
        "models": models,
        "official_complete": True,
        "sources": [_source("math")],
    }


def _longcode_axis() -> dict:
    models = {}
    for model_index, (model, plan) in enumerate(renderer.LONGCODE_PLANS.items()):
        methods = {}
        for method_index, method in enumerate(renderer.METHODS):
            brackets = {}
            model_correct = 0
            all_sample_ids = []
            for bracket, n in plan.items():
                correct = n // 2 + model_index + method_index
                model_correct += correct
                low, high = renderer._wilson_ci(correct, n)
                sample_ids = [f"{bracket}:{index}" for index in range(n)]
                all_sample_ids.extend(sample_ids)
                brackets[bracket] = {
                    "n": n,
                    "correct": correct,
                    "accuracy": round(correct / n, 8),
                    "accuracy_stderr": 0.0,
                    "accuracy_ci95": [round(low, 8), round(high, 8)],
                    "coverage": [0, n],
                    "released_data_complete_bracket": True,
                    "audit_records_validated": n,
                    "context_fit_manifest_sha256": "d" * 64,
                }
                brackets[bracket].update(_long_accounting(method, sample_ids))
            total = sum(plan.values())
            low, high = renderer._wilson_ci(model_correct, total)
            methods[method] = {
                "primary_reporting": "per_bracket_exact_match_accuracy",
                "derived_summary_metric": "micro_exact_match_accuracy",
                "n": total,
                "correct": model_correct,
                "accuracy": round(model_correct / total, 8),
                "accuracy_stderr": 0.0,
                "accuracy_ci95": [round(low, 8), round(high, 8)],
                "brackets": brackets,
                "released_data_complete_feasible_plan": True,
            }
            methods[method].update(_long_accounting(method, all_sample_ids))
        models[model] = {
            "methods": methods,
            "released_data_complete_feasible_plan": True,
        }
    return {
        "benchmark": "LongCodeBench/LongCodeQA",
        "schema_version": 1,
        "aggregation": "sum_exact_counts_after_exact_interval_coverage",
        "primary_reporting": "per_bracket_exact_match_accuracy",
        "cross_bracket_summary": "derived_micro_exact_match_accuracy",
        "scope": "all released brackets that fit each unmodified model context",
        "protocol_scope": renderer.LONGCODE_PROTOCOL_SCOPE,
        "official_reference_decoding_exact": False,
        "source_row_audit": dict(renderer.LONGCODE_SOURCE_ROW_AUDIT),
        "commit": COMMIT,
        "models": models,
        "released_data_complete_feasible_plan": True,
        "sources": [_source("longcode")],
    }


def _speed_metric(value: float, unit: str) -> dict:
    values = [value - 0.2, value - 0.1, value, value + 0.1, value + 0.2]
    return {
        "unit": unit,
        "replicate_medians": values,
        "median_of_replicate_medians": value,
        "min": min(values),
        "max": max(values),
    }


def _speed_pair(
    values: list[float],
    baseline: str,
    candidate: str,
    repeats: int,
    sources: list[str],
) -> dict:
    logs = [math.log(value) for value in values]
    log_mean = statistics.fmean(logs)
    log_sd = statistics.stdev(logs)
    log_se = log_sd / math.sqrt(5.0)
    critical = 2.7764451052
    return {
        "unit": "candidate_speedup_x",
        "ratio_definition": "baseline_latency / candidate_latency",
        "n_independent_processes": 5,
        "process_median_paired_speedups": values,
        "geometric_mean_process_speedup_x": math.exp(log_mean),
        "median_process_speedup_x": statistics.median(values),
        "min_process_speedup_x": min(values),
        "max_process_speedup_x": max(values),
        "log_ratio_sample_sd": log_sd,
        "log_ratio_standard_error": log_se,
        "student_t_df": 4,
        "student_t_critical_95": critical,
        "student_t_log_geomean_ci95": [
            math.exp(log_mean - critical * log_se),
            math.exp(log_mean + critical * log_se),
        ],
        "baseline_method": baseline,
        "candidate_method": candidate,
        "processes": [
            {
                "seed": 20260715 + index,
                "source": sources[index],
                "run_uuid": (
                    lambda digest: (
                        f"{digest[:8]}-{digest[8:12]}-4{digest[13:16]}-"
                        f"8{digest[17:20]}-{digest[20:32]}"
                    )
                )(hashlib.sha256(sources[index].encode()).hexdigest()),
                "slurm_job_id": hashlib.sha256(sources[index].encode()).hexdigest()[:12],
                "baseline_method": baseline,
                "candidate_method": candidate,
                "n_paired_repeats": repeats,
                "raw_paired_ratios": [{} for _ in range(repeats)],
                "process_median_speedup_x": value,
            }
            for index, value in enumerate(values)
        ],
    }


def _speed_memory_pair(
    values: list[float], baseline: str, candidate: str, sources: list[str]
) -> dict:
    critical = 2.7764451052
    logs = [math.log(value) for value in values]
    log_mean = statistics.fmean(logs)
    log_sd = statistics.stdev(logs)
    log_se = log_sd / math.sqrt(5.0)
    differences = [1000.0 * value - 1000.0 for value in values]
    difference_mean = statistics.fmean(differences)
    difference_sd = statistics.stdev(differences)
    difference_se = difference_sd / math.sqrt(5.0)
    processes = []
    for index, (value, difference) in enumerate(zip(values, differences)):
        digest = hashlib.sha256(sources[index].encode()).hexdigest()
        run_uuid = (
            f"{digest[:8]}-{digest[8:12]}-4{digest[13:16]}-" f"8{digest[17:20]}-{digest[20:32]}"
        )
        raw = [
            {
                "repeat": repeat,
                "baseline_peak_allocated_mib": 1000.0 * value,
                "candidate_peak_allocated_mib": 1000.0,
                "candidate_memory_advantage_x": value,
                "candidate_memory_reduction_mib": difference,
            }
            for repeat in range(5)
        ]
        processes.append(
            {
                "seed": 20260715 + index,
                "source": sources[index],
                "run_uuid": run_uuid,
                "slurm_job_id": digest[:12],
                "baseline_method": baseline,
                "candidate_method": candidate,
                "n_paired_repeats": 5,
                "raw_paired_memory": raw,
                "process_median_memory_advantage_x": value,
                "process_median_memory_reduction_mib": difference,
            }
        )
    return {
        "baseline_method": baseline,
        "candidate_method": candidate,
        "processes": processes,
        "ratio": {
            "unit": "candidate_memory_advantage_x",
            "ratio_definition": ("baseline_peak_allocated_mib / candidate_peak_allocated_mib"),
            "n_independent_processes": 5,
            "process_median_paired_memory_advantages": values,
            "geometric_mean_process_memory_advantage_x": math.exp(log_mean),
            "median_process_memory_advantage_x": statistics.median(values),
            "min_process_memory_advantage_x": min(values),
            "max_process_memory_advantage_x": max(values),
            "student_t_df": 4,
            "student_t_log_geomean_ci95": [
                math.exp(log_mean - critical * log_se),
                math.exp(log_mean + critical * log_se),
            ],
        },
        "difference": {
            "unit": "MiB",
            "difference_definition": (
                "baseline_peak_allocated_mib - candidate_peak_allocated_mib"
            ),
            "n_independent_processes": 5,
            "process_median_paired_memory_reductions_mib": differences,
            "mean_process_memory_reduction_mib": difference_mean,
            "student_t_df": 4,
            "student_t_mean_difference_ci95_mib": [
                difference_mean - critical * difference_se,
                difference_mean + critical * difference_se,
            ],
        },
    }


def _speed_scientific_recipe(rate: int, names: dict[str, str]) -> dict:
    common = {
        "key_axis": "token",
        "value_axis": "token",
        "group_size": -1,
        "symmetric": False,
        "skip_layers": [],
        "residual_window": 128,
    }
    hayate_defaults = {
        **common,
        "name": "hayatequant",
        "max_codebook_bits": 9,
        "codebook_init": "auto",
        "seed": 0,
        "encoder": "exact",
        "hadamard_gemm_max_dim": 4096,
        "ls_scale": True,
        "outlier_bits": None,
        "key_n_outlier": None,
        "value_n_outlier": None,
        "key_outlier_bits": None,
        "value_outlier_bits": None,
        "avg_bits": float(rate),
        "code_bits_per_pair_block": None,
        "key_bit_fraction": None,
        "adaptive_bit_split": None,
        "adaptive_bit_margin": 4,
        "adaptive_stats_min_tokens": 16,
        "key_error_weight": 4.0,
        "value_error_weight": 1.0,
        "key_code_bits_per_block": None,
        "value_code_bits_per_block": None,
        "use_qjl": False,
    }
    declared = {
        "tq": {
            **common,
            "name": "turboquant",
            "bits": rate,
            "dist": "beta",
            "n_outlier": 0,
            "outlier_bits": None,
            "use_qjl": False,
            "qjl_roles": "key",
            "rotation_mode": "dense",
            "seed": 0,
            "value_bits": rate,
        },
        "hq": {
            **hayate_defaults,
            "required_codebook_bundle": renderer.HQ_CODEBOOK_BUNDLE,
            "bits": rate,
            "value_bits": rate,
            "dist": "gaussian",
            "rotation_mode": "auto",
            "block_size": 2,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_code_bits_per_block": 2 * rate + 1,
            "value_code_bits_per_block": 2 * rate - 1,
            "n_outlier": 0,
        },
        "adaptive": {
            **hayate_defaults,
            "required_codebook_bundle": renderer.ADAPTIVE_HQ_CODEBOOK_BUNDLE,
            "bits": rate,
            "value_bits": rate,
            "dist": "gaussian",
            "rotation_mode": "auto",
            "block_size": 2,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_bit_fraction": 0.5,
            "adaptive_bit_split": "online_stats",
            "n_outlier": 0,
        },
    }
    effective = {
        "tq": {},
        "hq": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
        "adaptive": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
    }
    taxonomy = {
        "tq": {
            "semantic_role": "existing TurboQuant class; symmetric scalar Beta/L2 reference",
            "fake_quant_implementation": "inherited unfused encode_then_decode",
            "proposal_method": False,
            "proposal_variant": None,
        },
        "hq": {
            "semantic_role": "canonical globally fixed HQ paper proposal",
            "fake_quant_implementation": "HayateQuant fused research fake_quant_roundtrip",
            "proposal_method": True,
            "proposal_variant": "fixed_global",
        },
        "adaptive": {
            "semantic_role": "causal online-rate-allocation HQ paper proposal",
            "fake_quant_implementation": "HayateQuant fused research fake_quant_roundtrip",
            "proposal_method": True,
            "proposal_variant": "causal_adaptive",
        },
    }
    configs = {
        names[method]: {
            **taxonomy[method],
            "codec": {
                "declared": declared[method],
                "effective": effective[method],
            },
            "resolved_at_model_head_dim": renderer._expected_speed_resolved_recipe(  # pylint: disable=protected-access
                rate, method
            ),
        }
        for method in ("tq", "hq", "adaptive")
    }
    hashes = {
        name: hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        for name, record in configs.items()
    }
    return {"recipe_sha256": hashes, "method_configs": configs}


def _speed_axis() -> dict:
    models = {}
    source_records = []
    for model in renderer.LONG_MODELS:
        models[model] = {}
        for rate in renderer.RATES:
            names = {
                "full": "fullkv_uncompressed",
                "tq": "TurboQuant-paper",
                "hq": "HQ",
                "adaptive": "AdaptiveHQ",
            }
            models[model][rate] = {}
            for mode in ("off", "decode-only"):
                row_sources = [
                    f"/formal/speed-{model}-{rate}-{mode}-{index}.json" for index in range(5)
                ]
                source_records.extend(
                    _source(f"speed-{model}-{rate}-{mode}-{index}") for index in range(5)
                )
                metrics = {}
                for index, name in enumerate(names.values()):
                    metrics[f"e2e.summary.{name}.decode_steady_throughput"] = _speed_metric(
                        100.0 + index, "tokens_per_second"
                    )
                for index, name in enumerate((names["tq"], names["hq"], names["adaptive"])):
                    path = (
                        "micro.regimes.decode.payload_api."
                        f"roundtrip_explicit_encode_then_decode.summary.{name}"
                    )
                    metrics[path] = _speed_metric(1.0 + index, "ms_per_call")
                expected_pairs = {
                    "hq_vs_turboquant": (names["tq"], names["hq"]),
                    "adaptive_hq_vs_turboquant": (names["tq"], names["adaptive"]),
                }
                paired = {}
                for metric_index, metric in enumerate(renderer.SPEED_PAIRED_METRICS):
                    repeats = 5 if metric.startswith("e2e.") else 40
                    paired[metric] = {
                        comparison: _speed_pair(
                            [
                                1.05 + 0.01 * metric_index,
                                1.10 + 0.01 * metric_index,
                                1.15 + 0.01 * metric_index,
                                1.20 + 0.01 * metric_index,
                                1.25 + 0.01 * metric_index,
                            ],
                            baseline,
                            candidate,
                            repeats,
                            row_sources,
                        )
                        for comparison, (baseline, candidate) in expected_pairs.items()
                    }
                paired_memory = {
                    comparison: _speed_memory_pair(
                        [1.01, 1.02, 1.03, 1.04, 1.05],
                        baseline,
                        candidate,
                        row_sources,
                    )
                    for comparison, (baseline, candidate) in expected_pairs.items()
                }
                public_payload_footprints = {}
                for regime, factor in (("prefill", 4096), ("decode", 1)):
                    public_payload_footprints[regime] = {
                        names["tq"]: {
                            "materialized_payload_tensor_bytes": 1024 * factor,
                            "logical_total_stored_bits": 384 * factor,
                        },
                        names["hq"]: {
                            "materialized_payload_tensor_bytes": 640 * factor,
                            "logical_total_stored_bits": 384 * factor,
                        },
                        names["adaptive"]: {
                            "materialized_payload_tensor_bytes": 608 * factor,
                            "logical_total_stored_bits": 384 * factor,
                        },
                    }
                models[model][rate][mode] = {
                    "n_replicates": 5,
                    "seeds": [20260715, 20260716, 20260717, 20260718, 20260719],
                    "scientific_recipe": _speed_scientific_recipe(int(rate), names),
                    "metrics": metrics,
                    "paired_speedups": paired,
                    "paired_peak_allocated_memory": paired_memory,
                    "public_payload_footprints": public_payload_footprints,
                    "sources": row_sources,
                }
    pareto = aggregator._build_speed_pareto_contract(  # pylint: disable=protected-access
        models,
        expected_models=renderer.LONG_MODELS,
        expected_rates=tuple(int(rate) for rate in renderer.RATES),
    )
    return {
        "benchmark": "paired HayateQuant fair speed",
        "model_evaluation_roles": dict(renderer.SPEED_MODEL_EVALUATION_ROLES),
        "commit": COMMIT,
        "primary_compile_mode": "off",
        "secondary_compile_modes": ["decode-only"],
        "primary_mode_predeclared_in_source": True,
        "hardware_stratum": {
            "gpu_name": "Synthetic GPU",
            "compute_capability": [9, 0],
            "total_memory_bytes": 1000000,
            "multi_processor_count": 100,
            "driver_version": "999.0",
        },
        "software_stratum": {"python": "3.12", "package:torch": "2.9"},
        "timing_protocol_stratum": {"micro_repeats": 40, "e2e_repeats": 5},
        "statistical_unit": "independently_launched_fresh_process",
        "independent_process_audit": {
            "n_unique_run_uuid4": 60,
            "n_unique_slurm_job_ids": 60,
            "uuid_and_slurm_job_id_unique_for_every_source": True,
        },
        "five_process_design_note": (
            "five independent processes are predeclared for a process-level "
            "Student-t interval; technical repeats remain paired within process "
            "and are never promoted to inferential replicates"
        ),
        "paired_ratio_estimator": (
            "within-process median of repeatwise baseline/candidate latency ratios, "
            "then geometric mean across independent processes"
        ),
        "confidence_interval": ("two-sided 95% Student-t interval on process-level log ratios"),
        "multiplicity_contract": {
            "interval_scope": "pointwise_nominal_95_percent",
            "multiple_comparison_adjustment": (
                "none for per-endpoint tables; one-sided Bonferroni over every "
                "predeclared primary endpoint for the complete-dominance assessment"
            ),
            "reporting_scope": (
                "per-endpoint ratios remain descriptive; a complete-superiority claim "
                "is admitted only by the simultaneous pareto contract"
            ),
            "directional_speed_claim_allowed": False,
            "blanket_or_global_speed_claim_allowed": pareto[
                "complete_strict_superiority_established"
            ],
        },
        "public_payload_scope": dict(renderer.PUBLIC_LOGICAL_PAYLOAD_SCOPE),
        "public_payload_tensor_scope": renderer.PUBLIC_PAYLOAD_TENSOR_SCOPE,
        "e2e_peak_allocated_memory_scope": dict(renderer.SPEED_PEAK_ALLOCATED_SCOPE),
        "allocation_isolation_contract": {
            **renderer.SPEED_ALLOCATION_CONTRACT,
            "physical_node": "kagura-gpu06",
            "allocation_cuda_visibility_recorded_before_process_narrowing": True,
        },
        "implementation_scope_contract": {
            "packed_native_hybrid_measured": False,
            "packed_native_hybrid_must_not_be_called_full_hayatequant": True,
            "packed_decode_claim_allowed": False,
            "measurement_scope": (
                "public unpacked logical-payload encode/decode plus research fake-quant "
                "and dequantized research-cache E2E"
            ),
            "methods": {
                "TurboQuant-paper": "inherited unfused encode_then_decode",
                "HQ": "HayateQuant fused research fake_quant_roundtrip",
                "AdaptiveHQ": "HayateQuant fused research fake_quant_roundtrip",
            },
        },
        "pareto_contract": pareto,
        "models": models,
        "official_complete": True,
        "sources": source_records,
    }


def _formal_metadata(document: dict) -> tuple[dict, dict]:
    input_path = "/formal/manifests/hayatequant-paper-inputs.json"
    input_sha256 = "c" * 64
    normalized_runtime = {
        "python": "3.12.10",
        "packages": {package: "1.0.0" for package in renderer.QUALITY_SOFTWARE_PACKAGES},
        "onecomp_commit": COMMIT,
        "cuda": {"torch_cuda": "12.8", "cudnn": 91002},
        "gpu_classes": [
            {"name": "nvidia b200", "compute_capability": [10, 0]},
        ],
        "semantic_environment": dict(renderer.QUALITY_SEMANTIC_ENVIRONMENT),
    }
    runtime_fingerprint = hashlib.sha256(
        json.dumps(normalized_runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_paths = {
        axis: {source["path"] for source in document[axis]["sources"]}
        for axis in renderer.REQUIRED_AXES
    }
    formal_provenance = {
        "input_manifest": {"path": input_path, "sha256": input_sha256},
        "models": {
            model: {
                "model_key": model,
                "model_path": f"/formal/models/{model}",
                "identity_sha256": hashlib.sha256(model.encode()).hexdigest(),
            }
            for model in renderer.PAPER_MODELS
        },
        "dataset_files": [
            {
                "path": input_path,
                "bytes": 1,
                "sha256": input_sha256,
            },
        ],
        "axis_source_counts": {axis: len(paths) for axis, paths in source_paths.items()},
        "validated_source_files": len(set().union(*source_paths.values())),
        "quality_runtime": {
            "fingerprint_sha256": runtime_fingerprint,
            "normalized": normalized_runtime,
            "scope": ("all quality axes; speed retains its separate timing-stratum gate"),
            "excluded_nuisance_identity": [
                "GPU UUID",
                "host name",
                "Slurm job id",
            ],
        },
    }
    matrix_status = {
        "required_axes": list(renderer.REQUIRED_AXES),
        "present_axes": list(renderer.REQUIRED_AXES),
        "missing_axes": [],
        "incomplete_axes": [],
        "one_cross_axis_provenance_identity": True,
    }
    return formal_provenance, matrix_status


def _aggregate() -> dict:
    document = {
        "schema_version": 1,
        "created_at_utc": "2026-07-15T00:00:00+00:00",
        "commit": COMMIT,
        "formal_matrix_complete": True,
        "method_scopes": {
            "TurboQuant-paper": dict(renderer.TURBOQUANT_METHOD_SCOPE),
            "FibQuant-paper": dict(renderer.FIBQUANT_METHOD_SCOPE),
            "BlockQuant-paper": dict(renderer.BLOCKQUANT_METHOD_SCOPE),
        },
        "ppl": _ppl_axis(),
        "mechanism": _mechanism_axis(),
        "zeroshot": _zeroshot_axis(),
        "ruler": _ruler_axis(),
        "longbench": _longbench_axis(),
        "math500": _math_axis(),
        "longcodeqa": _longcode_axis(),
        "speed": _speed_axis(),
    }
    for axis in ("ppl", "zeroshot"):
        document[axis]["model_evaluation_roles"] = dict(renderer.PAPER_MODEL_EVALUATION_ROLES)
    for axis in ("mechanism", "ruler", "longbench", "math500", "longcodeqa"):
        document[axis]["model_evaluation_roles"] = dict(renderer.SPEED_MODEL_EVALUATION_ROLES)
    document["formal_provenance"], document["formal_matrix_status"] = _formal_metadata(document)
    document["point_estimate_win_gate"] = aggregator._point_estimate_win_gate(document)
    document["formal_matrix_status"].update(
        {
            "coverage_complete": True,
            "point_estimate_win_gate_passed": document["point_estimate_win_gate"]["passed"],
        }
    )
    document["official_complete"] = True
    return document


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def test_renderer_is_deterministic_and_emits_every_formal_axis(tmp_path):
    aggregate = tmp_path / "coverage_checked_aggregates.json"
    _write(aggregate, _aggregate())

    first = renderer.render(aggregate, tmp_path / "first")
    second = renderer.render(aggregate, tmp_path / "second")
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()

    macros = first[0].read_text(encoding="utf-8")
    tables = first[1].read_text(encoding="utf-8")
    aggregate_sha256 = hashlib.sha256(aggregate.read_bytes()).hexdigest()
    assert rf"\newcommand{{\HayateFormalCommitHex}}{{{COMMIT}}}" in macros
    assert rf"\newcommand{{\HayateFormalAggregateSHAHex}}{{{aggregate_sha256}}}" in macros
    assert rf"\newcommand{{\HayateFormalTablesCommitHex}}{{{COMMIT}}}" in tables
    assert rf"\newcommand{{\HayateFormalTablesAggregateSHAHex}}{{{aggregate_sha256}}}" in tables
    assert (
        rf"\newcommand{{\HayateCodebookManifestSHA}}{{{renderer.HQ_CODEBOOK_MANIFEST_SHA256}}}"
        in macros
    )
    assert r"\newcommand{\HayateHQTurboQuantGateWins}{164}" in macros
    assert r"\newcommand{\HayateHQTurboQuantGateCells}{164}" in macros
    assert r"\newcommand{\HayateHQTurboQuantGateWinFraction}{1.000000}" in macros
    assert r"\newcommand{\HayateHQTurboQuantGatePassed}{true}" in macros
    assert r"\newcommand{\HayateHQBlockQuantGateCells}{80}" in macros
    assert (
        r"\newcommand{\HayateHQBlockQuantGatePassed}"
        r"{not-applicable-unmatched-operating-point}" in macros
    )
    assert r"\newcommand{\HayateHQBlockQuantGateIncludedInGate}{false}" in macros
    assert r"\newcommand{\HayateAdaptiveHQTurboQuantSpeedGateCells}{60}" in macros
    assert r"\newcommand{\HayateAbstractQualitySummary}" in macros
    assert r"\newcommand{\HayateAbstractSpeedSummary}" in macros
    assert r"\newcommand{\HayateIntroResultSummary}" in macros
    assert tables.index(r"\HayateValidateFormalTableIdentity") < tables.index(r"\begin{table*")
    assert r"\newcommand{\HayatePPLWinsTwoBit}{7}" in macros
    assert r"\newcommand{\HayatePPLTiesThreeBit}{0}" in macros
    assert r"\newcommand{\HayatePPLLossesFourBit}{0}" in macros
    assert (
        r"\newcommand{\HayatePPLPointStatementThreeBit}"
        r"{lower point-estimate PPL on this fixed panel}" in macros
    )
    assert (
        r"\newcommand{\HayatePPLRobustnessStatementThreeBit}"
        r"{simultaneous interval supports lower PPL on this fixed panel}" in macros
    )
    assert (
        r"\newcommand{\HayatePPLRobustnessStatementFourBit}"
        r"{simultaneous interval supports lower PPL on this fixed panel}" in macros
    )
    assert r"\newcommand{\HayateZeroShotWinsTwoBit}{7}" in macros
    assert r"\newcommand{\HayateZeroShotTiesThreeBit}{0}" in macros
    assert r"\newcommand{\HayateZeroShotLossesFourBit}{0}" in macros
    assert (
        r"\newcommand{\HayateZeroShotPointStatementTwoBit}"
        r"{higher point-estimate ZS-Mean on this fixed panel}" in macros
    )
    assert (
        r"\newcommand{\HayateZeroShotPointStatementFourBit}"
        r"{higher point-estimate ZS-Mean on this fixed panel}" in macros
    )
    assert (
        r"\newcommand{\HayateZeroShotRobustnessStatementThreeBit}"
        r"{simultaneous interval supports higher ZS-Mean on this fixed panel}" in macros
    )
    assert (
        r"\newcommand{\HayateMechanismClaimThreeBit}"
        r"{descriptive only; no directional mechanism claim}" in macros
    )
    assert (
        r"\newcommand{\HayateResearchEndToEndSpeedLlamaTwoBitClaim}"
        r"{descriptive ratio only; no directional speed claim}" in macros
    )
    assert (
        r"\newcommand{\HayatePublicPayloadSpeedQwenFourBitClaim}"
        r"{descriptive ratio only; no directional speed claim}" in macros
    )
    assert (
        r"\newcommand{\HayateGlobalSpeedClaim}"
        r"{complete speed/measured-memory superiority not established}" in macros
    )
    assert r"\newcommand{\HayateCompleteDominanceEndpointCount}{132}" in macros
    assert r"\newcommand{\HayatePackedDecodeClaimAllowed}{false}" in macros
    assert r"\newcommand{\HayateTurboQuantPaperTableOneReproduction}{false}" in macros
    assert "Algorithm 1 MSE protocol baseline only" in macros
    assert "unpacked int64 index tensors" in macros
    assert "fixed-width KV data-plane fields" in macros
    assert "self-describing control tags" in macros
    assert "Python containers" in macros
    assert "persistent codec state are uncharged" in macros
    assert "not physical tensor memory" in macros
    assert "TQ Alg.-1 MSE" in tables
    assert "unpacked int64 arrays" in tables
    assert "excluding control tags, Python containers" in tables
    assert "not their physical tensor footprint" in tables
    assert "not packed serving storage" in tables
    assert "median of five process medians" in tables
    assert "median of three process medians" not in tables
    assert r"$^{+}$" not in tables
    assert r"$^{-}$" not in tables
    assert r"$^{?}$" not in tables
    assert "pointwise nominal" in tables
    assert "point direction" in tables
    assert "W/T/L" in tables
    assert "AdaptiveHQ" in tables
    assert "primary descriptive report" in tables
    assert "robustness supplement" in tables
    assert "descriptive only" in tables
    assert "directional" in tables
    assert "statistically significant" not in macros + tables
    assert "HQ faster" not in macros + tables
    assert "HQ slower" not in macros + tables
    assert renderer.LONGBENCH_TOTAL == 3668
    assert "3,668 examples per method" in tables

    expected_labels = {
        "formal-ppl-primary",
        "formal-ppl-models",
        "formal-zeroshot-primary",
        "formal-zeroshot-models",
        "formal-mechanism",
        "formal-ruler",
        "formal-longbench",
        "formal-math500",
        "formal-longcodeqa",
        "formal-speed-e2e",
        "formal-speed-micro",
        "formal-speed-paired-e2e",
        "formal-speed-paired-micro",
        "formal-speed-public-operations",
        "formal-speed-research-fused",
        "formal-speed-memory-representations",
    }
    assert all(rf"\label{{tab:{label}}}" in tables for label in expected_labels)
    lowered = (macros + tables).lower()
    assert "placeholder" not in lowered
    assert "fallback" not in lowered
    assert "27.910" not in tables
    assert r"12.4\times" not in tables


def test_complete_negative_result_renders_but_suppresses_superiority_claim(tmp_path):
    document = _aggregate()
    gate = document["point_estimate_win_gate"]
    changed = 0
    for cell in gate["cells"]:
        if (
            cell["proposal"] == "HQ"
            and cell["baseline"] == "TurboQuant-paper"
            and cell["axis"] != "speed"
            and changed < 20
        ):
            baseline = float(cell["baseline_value_unrounded"])
            cell["proposal_value_unrounded"] = (
                baseline + 1.0 if cell["direction"] == "lower_is_better" else baseline - 1.0
            )
            cell["win"] = False
            cell["status"] = "strict_loss"
            changed += 1
    assert changed == 20
    summaries = gate["proposal_baseline_summaries"]
    all_summaries = []
    for proposal in renderer.PROPOSAL_METHODS:
        for baseline in ("TurboQuant-paper", "FibQuant-paper"):
            rows = [
                cell
                for cell in gate["cells"]
                if cell["proposal"] == proposal
                and cell["baseline"] == baseline
                and cell["axis"] != "speed"
            ]
            summary = aggregator._point_estimate_pair_summary(rows, scope="quality")
            summaries[proposal][baseline] = summary
            all_summaries.append(summary)
        rows = [
            cell
            for cell in gate["cells"]
            if cell["proposal"] == proposal and cell["axis"] == "speed"
        ]
        summary = aggregator._point_estimate_pair_summary(rows, scope="speed")
        summaries[proposal]["TurboQuant-paper-speed"] = summary
        all_summaries.append(summary)
    gate["passed"] = all(summary["passed"] for summary in all_summaries)
    assert gate["passed"] is False
    document["formal_matrix_status"]["point_estimate_win_gate_passed"] = False
    assert document["formal_matrix_complete"] is True
    assert document["official_complete"] is True

    aggregate = tmp_path / "complete_negative.json"
    _write(aggregate, document)
    macros_path, _ = renderer.render(aggregate, tmp_path / "rendered")
    macros = macros_path.read_text(encoding="utf-8")
    assert r"\newcommand{\HayatePointEstimateWinGatePassed}{false}" in macros
    assert r"\newcommand{\HayateDirectionalSuperiorityClaimAllowed}{false}" in macros
    assert "directional superiority claims are suppressed" in macros
    assert "passed 5 of 6 matched-operating-point proposal/baseline scopes" in macros
    assert r"\newcommand{\HayateHQTurboQuantGatePassed}{false}" in macros


@pytest.mark.parametrize("mutation", ["inject_into_denominator", "claim_matched"])
def test_renderer_rejects_blockquant_gate_scope_tampering(tmp_path, mutation):
    document = _aggregate()
    gate = document["point_estimate_win_gate"]
    block_cell = copy.deepcopy(gate["unmatched_descriptive_comparisons"]["cells"][0])
    if mutation == "inject_into_denominator":
        descriptive_better = block_cell.pop("descriptive_better")
        block_cell.pop("sample_set_matched")
        block_cell["matched_comparison"] = True
        block_cell["rate_match_required"] = True
        block_cell["operating_point_matched"] = True
        block_cell["included_in_denominator"] = True
        block_cell["win"] = descriptive_better
        block_cell["status"] = block_cell["status"].removeprefix("unmatched_descriptive_")
        gate["cells"][0] = block_cell
    else:
        gate["unmatched_descriptive_comparisons"]["cells"][0]["operating_point_matched"] = True
    aggregate = tmp_path / f"block-{mutation}.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match="gate|BlockQuant|cell|scope"):
        renderer.render(aggregate, tmp_path / f"rendered-{mutation}")


def test_cross_include_identity_is_content_addressed(tmp_path):
    first_document = _aggregate()
    second_document = copy.deepcopy(first_document)
    second_document["created_at_utc"] = "2026-07-15T00:00:01+00:00"

    first_aggregate = tmp_path / "first.json"
    second_aggregate = tmp_path / "second.json"
    _write(first_aggregate, first_document)
    _write(second_aggregate, second_document)
    first_paths = renderer.render(first_aggregate, tmp_path / "first")
    second_paths = renderer.render(second_aggregate, tmp_path / "second")

    first_sha = hashlib.sha256(first_aggregate.read_bytes()).hexdigest()
    second_sha = hashlib.sha256(second_aggregate.read_bytes()).hexdigest()
    assert first_sha != second_sha
    first_macros = first_paths[0].read_text(encoding="utf-8")
    first_tables = first_paths[1].read_text(encoding="utf-8")
    second_macros = second_paths[0].read_text(encoding="utf-8")
    second_tables = second_paths[1].read_text(encoding="utf-8")
    assert rf"\newcommand{{\HayateFormalAggregateSHAHex}}{{{first_sha}}}" in first_macros
    assert rf"\newcommand{{\HayateFormalTablesAggregateSHAHex}}{{{first_sha}}}" in first_tables
    assert rf"\newcommand{{\HayateFormalAggregateSHAHex}}{{{second_sha}}}" in second_macros
    assert rf"\newcommand{{\HayateFormalTablesAggregateSHAHex}}{{{second_sha}}}" in second_tables


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda doc: doc["ppl"]["evaluation_config"].__setitem__("ppl_stride", 512),
            "complete formal WikiText-2 benchmark",
        ),
        (
            lambda doc: doc["ppl"]["models"][renderer.PAPER_MODELS[0]]["fullkv"].__setitem__(
                "ppl_text_sha256", "f" * 64
            ),
            "FullKV source identity changed",
        ),
        (
            lambda doc: doc["ppl"]["models"][renderer.PAPER_MODELS[0]]["rates"]["2"][
                "HQ"
            ].__setitem__("ppl_compression_ratio", 3.5),
            "total-bit compression ratio",
        ),
        (
            lambda doc: doc["mechanism"]["inference_scope"].__setitem__(
                "directional_claim_allowed", True
            ),
            "exact formal causal control",
        ),
        (
            lambda doc: doc["math500"].__setitem__(
                "official_reference_gold_serialization_exact", True
            ),
            "complete pinned Math-Verify",
        ),
    ],
)
def test_renderer_rejects_new_protocol_contract_drift(tmp_path, mutation, message):
    document = _aggregate()
    mutation(document)
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, tmp_path / "generated")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda doc: doc.pop("math500"), "keys differ"),
        (lambda doc: doc.__setitem__("formal_matrix_complete", False), "must be true"),
        (
            lambda doc: doc["method_scopes"]["TurboQuant-paper"].__setitem__(
                "paper_table1_reproduction", True
            ),
            "must not claim a paper Table 1 reproduction",
        ),
        (
            lambda doc: doc["speed"]["public_payload_scope"].__setitem__(
                "index_tensors_bit_packed", True
            ),
            "predeclared complete fair-speed matrix",
        ),
        (
            lambda doc: doc["speed"]["multiplicity_contract"].__setitem__(
                "directional_speed_claim_allowed", True
            ),
            "predeclared complete fair-speed matrix",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("commit", "f" * 40),
            "aggregate commit",
        ),
        (
            lambda doc: doc["ppl"].__setitem__("legacy_values", {"ppl": 27.910}),
            "forbidden",
        ),
        (
            lambda doc: doc["longcodeqa"].__setitem__(
                "protocol_scope", "official_reference_decoding_exact"
            ),
            "complete model-feasible plan",
        ),
        (
            lambda doc: doc["longcodeqa"].__setitem__("official_reference_decoding_exact", True),
            "complete model-feasible plan",
        ),
        (
            lambda doc: doc["longcodeqa"].pop("source_row_audit"),
            "source-row/context-fit audit",
        ),
        (
            lambda doc: doc["longcodeqa"]["models"]["Qwen3-8B"]["methods"]["FullKV"]["brackets"][
                "32K"
            ].__setitem__("audit_records_validated", 0),
            "bracket incomplete",
        ),
        (
            lambda doc: doc["longbench"].__setitem__("reference_model_wrapper_exact", True),
            "complete LongBench-E",
        ),
    ],
)
def test_renderer_rejects_incomplete_or_legacy_matrix_before_write(tmp_path, mutation, message):
    document = _aggregate()
    mutation(document)
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda doc: doc["formal_matrix_status"]["missing_axes"].append("math500"),
            "exact complete eight-axis matrix",
        ),
        (
            lambda doc: doc["formal_provenance"]["quality_runtime"]["normalized"][
                "packages"
            ].__setitem__("torch", "9.9.9"),
            "fingerprint is inconsistent",
        ),
        (
            lambda doc: doc["formal_provenance"]["quality_runtime"]["normalized"]["packages"].pop(
                "math-verify"
            ),
            "keys differ",
        ),
        (
            lambda doc: doc["formal_provenance"]["axis_source_counts"].__setitem__("longbench", 2),
            "source count is stale",
        ),
        (
            lambda doc: doc["formal_provenance"].__setitem__(
                "validated_source_files",
                doc["formal_provenance"]["validated_source_files"] + 1,
            ),
            "source-ledger union",
        ),
        (
            lambda doc: doc["formal_provenance"]["models"][renderer.PAPER_MODELS[0]].__setitem__(
                "model_path", "relative/model"
            ),
            "model identity is malformed",
        ),
    ],
)
def test_renderer_rejects_cross_axis_provenance_tampering(tmp_path, mutation, message):
    document = _aggregate()
    mutation(document)
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda doc: doc["ruler"].__setitem__("schema_version", 2),
            "complete pinned NeMo-Skills protocol",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("schema_version", True),
            "complete pinned NeMo-Skills protocol",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("aggregation", "trust_displayed_score"),
            "complete pinned NeMo-Skills protocol",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("sampling_protocol", "canonical NVIDIA 500/task"),
            "complete pinned NeMo-Skills protocol",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("reference_model_wrapper_exact", True),
            "complete pinned NeMo-Skills protocol",
        ),
        (
            lambda doc: doc["ruler"].__setitem__("audit_records_validated", 1300),
            "axis audit coverage",
        ),
        (
            lambda doc: doc["ruler"].__setitem__(
                "audit_records_validated", float(renderer.RULER_AXIS_AUDIT_RECORDS)
            ),
            "must be an integer",
        ),
        (
            lambda doc: _first_ruler_row(doc).__setitem__("audit_records_validated", 1299),
            "row audit coverage",
        ),
        (
            lambda doc: _first_ruler_row(doc).__setitem__("model_max_position_embeddings", 65536),
            "model capacity changed",
        ),
        (
            lambda doc: _first_ruler_row(doc).__setitem__(
                "max_observed_prompt_plus_max_generation_tokens", 4097
            ),
            "named total-sequence budget violated",
        ),
        (
            lambda doc: _first_ruler_row(doc).__setitem__("generation_manifest_sha256", "f" * 64),
            "generation manifest changed",
        ),
        (
            lambda doc: _first_ruler_row(doc).pop("generation_manifest_sha256"),
            "generation manifest hash missing",
        ),
        (
            lambda doc: doc["ruler"]["generation_stop_protocols"][
                renderer.LONG_MODELS[0]
            ].__setitem__("eos_token_id_source", "tokenizer.eos_token_id"),
            "EOS/stop provenance changed",
        ),
        (
            lambda doc: _first_ruler_row(doc)["generation_stop_protocol"]["eos_token_ids"].pop(),
            "EOS/stop protocol changed",
        ),
        (
            lambda doc: doc["ruler"]["model_wrapper_protocols"][renderer.LONG_MODELS[0]][
                "chat_template_kwargs"
            ].__setitem__("enable_thinking", True),
            "non-thinking wrapper provenance changed",
        ),
    ],
)
def test_renderer_rejects_ruler_audit_contract_tampering(tmp_path, mutation, message):
    document = _aggregate()
    mutation(document)
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda doc: doc["zeroshot"].pop("pinned_source_prompt_audit"),
            "source audit.*object",
        ),
        (
            lambda doc: doc["zeroshot"]["pinned_source_prompt_audit"].__setitem__(
                "protocol_input_hash_schema", "unscoped_hash"
            ),
            "audit contract changed",
        ),
        (
            lambda doc: doc["zeroshot"]["pinned_source_prompt_audit"]["tasks"]["piqa"].__setitem__(
                "rows", 1
            ),
            "audit changed for piqa",
        ),
        (
            lambda doc: doc["zeroshot"]["models"][renderer.PAPER_MODELS[0]]["rates"]["2"]["HQ"][
                "tasks"
            ]["arc_easy"].__setitem__("protocol_input_sha256", "f" * 64),
            "Incomplete zero-shot coverage",
        ),
    ],
)
def test_renderer_rejects_zero_shot_source_prompt_audit_tampering(tmp_path, mutation, message):
    document = _aggregate()
    mutation(document)
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "kind",
    [
        "ppl",
        "zero_metric",
        "longbench",
        "longbench_numerator",
        "longbench_truncation",
        "math",
        "speed",
    ],
)
def test_renderer_rejects_stale_summaries_and_missing_nested_coverage(tmp_path, kind):
    document = _aggregate()
    if kind == "ppl":
        document["ppl"]["secondary_interval_robustness_vs_turboquant"]["HQ"]["2"][
            "geometric_mean_ppl_ratio"
        ] += 0.1
    elif kind == "zero_metric":
        model = renderer.PAPER_MODELS[0]
        document["zeroshot"]["models"][model]["rates"]["2"]["TurboQuant-paper"]["tasks"][
            "arc_easy"
        ]["primary_metric"] = "acc"
    elif kind == "longbench":
        model = renderer.LONG_MODELS[0]
        document["longbench"]["models"][model]["FullKV"]["per_task"].pop("lcc")
    elif kind in {"longbench_numerator", "longbench_truncation"}:
        model = renderer.LONG_MODELS[1]
        task = "multi_news"
        row = document["longbench"]["models"][model]["FullKV"]["per_task"][task]
        if kind == "longbench_numerator":
            row["score_sum_unrounded"] += 1.0
        else:
            row["truncated_examples"] = 0
    elif kind == "math":
        model = renderer.LONG_MODELS[0]
        document["math500"]["models"][model]["FullKV"]["pass@1"] += 0.1
    else:
        model = renderer.LONG_MODELS[0]
        document["speed"]["models"][model]["2"]["off"]["paired_speedups"].pop(
            "e2e.decode_steady_latency"
        )
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("rate", "key_spec", "value_spec"),
    [
        (2, (2, 2, "hybrid"), (2, 2, "dense")),
        (3, (3, 2, "hybrid"), (3, 2, "hybrid")),
        (4, (4, 2, "hybrid"), (4, 2, "hybrid")),
    ],
)
def test_renderer_speed_recipe_is_exactly_rate_and_role_specific(rate, key_spec, value_spec):
    record = renderer._expected_speed_resolved_recipe(  # pylint: disable=protected-access
        rate, "hq"
    )
    key, value = record["roles"]["key"], record["roles"]["value"]
    assert (key["bits"], key["block_size"], key["rotation_mode"]) == key_spec
    assert (value["bits"], value["block_size"], value["rotation_mode"]) == value_spec
    assert key["code_bits_per_block"] == 2 * rate + 1
    assert value["code_bits_per_block"] == 2 * rate - 1
    assert key["scale_mode"] == "nc"
    assert value["scale_mode"] == "ls_iter"
    assert key["source_law"] == value["source_law"] == "gaussian"
    assert key["rotation_seed"] == value["rotation_seed"] == 0


def test_renderer_rejects_rehashed_but_wrong_r4_role_rotation(tmp_path):
    document = _aggregate()
    model = renderer.LONG_MODELS[0]
    row = document["speed"]["models"][model]["4"]["off"]
    name = "HQ"
    record = row["scientific_recipe"]["method_configs"][name]
    record["resolved_at_model_head_dim"]["roles"]["key"]["rotation_mode"] = "dense"
    row["scientific_recipe"]["recipe_sha256"][name] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match="Role-resolved scientific speed recipe"):
        renderer.render(aggregate, tmp_path / "generated")


def test_renderer_rejects_stale_complete_dominance_decision(tmp_path):
    document = _aggregate()
    endpoint = document["speed"]["pareto_contract"]["stochastic_endpoints"][0]
    endpoint["bonferroni_one_sided_lower95"] += 0.1
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match="Pareto endpoint"):
        renderer.render(aggregate, tmp_path / "generated")


def test_renderer_rejects_tampered_paired_peak_memory_and_payload_footprint(tmp_path):
    document = _aggregate()
    model = renderer.LONG_MODELS[0]
    row = document["speed"]["models"][model]["2"]["off"]
    memory = row["paired_peak_allocated_memory"]["hq_vs_turboquant"]
    memory["processes"][0]["raw_paired_memory"][0]["candidate_peak_allocated_mib"] += 1.0
    aggregate = tmp_path / "aggregate-memory.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match="raw ratio"):
        renderer.render(aggregate, tmp_path / "generated-memory")

    document = _aggregate()
    row = document["speed"]["models"][model]["2"]["off"]
    footprint = row["public_payload_footprints"]["decode"]["HQ"]
    footprint["logical_total_stored_bits"] += 1
    aggregate = tmp_path / "aggregate-footprint.json"
    _write(aggregate, document)
    with pytest.raises(renderer.RenderError, match="logical rate differs"):
        renderer.render(aggregate, tmp_path / "generated-footprint")


def test_point_estimates_are_primary_and_robustness_gates_are_strict_at_null_boundary():
    assert (
        renderer._ppl_point_statement(0.999999) == "lower point-estimate PPL on this fixed panel"
    )
    assert renderer._ppl_point_statement(1.0) == "equal point-estimate PPL on this fixed panel"
    assert (
        renderer._ppl_point_statement(1.000001) == "higher point-estimate PPL on this fixed panel"
    )
    assert renderer._ppl_robustness_statement((0.8, 0.999999)) == (
        "simultaneous interval supports lower PPL on this fixed panel"
    )
    assert renderer._ppl_robustness_statement((0.8, 1.0)) == (
        "simultaneous interval does not establish directional robustness on this fixed panel"
    )
    assert renderer._ppl_robustness_statement((1.000001, 1.2)) == (
        "simultaneous interval supports higher PPL on this fixed panel"
    )
    assert renderer._zeroshot_point_statement(0.000001) == (
        "higher point-estimate ZS-Mean on this fixed panel"
    )
    assert renderer._zeroshot_point_statement(0.0) == (
        "equal point-estimate ZS-Mean on this fixed panel"
    )
    assert renderer._zeroshot_point_statement(-0.000001) == (
        "lower point-estimate ZS-Mean on this fixed panel"
    )
    assert renderer._zeroshot_robustness_statement((0.000001, 0.1)) == (
        "simultaneous interval supports higher ZS-Mean on this fixed panel"
    )
    assert renderer._zeroshot_robustness_statement((0.0, 0.1)) == (
        "simultaneous interval does not establish directional robustness on this fixed panel"
    )
    assert renderer._zeroshot_robustness_statement((-0.1, -0.000001)) == (
        "simultaneous interval supports lower ZS-Mean on this fixed panel"
    )
    assert renderer._win_tie_loss_counts((0.8, 1.0, 1.2), null=1.0, lower_is_better=True) == (
        1,
        1,
        1,
    )
    assert renderer._win_tie_loss_counts((-0.1, 0.0, 0.1), null=0.0, lower_is_better=False) == (
        1,
        1,
        1,
    )
    fixed_speed_statement = "descriptive ratio only; no directional speed claim"
    assert renderer._speed_claim((1.000001, 1.2)) == fixed_speed_statement
    assert renderer._speed_claim((1.0, 1.2)) == fixed_speed_statement
    assert renderer._speed_claim((0.8, 0.999999)) == fixed_speed_statement
    assert renderer._speed_claim((0.8, 1.0)) == fixed_speed_statement
    assert renderer._speed_claim((0.9, 1.1)) == fixed_speed_statement


@pytest.mark.parametrize(
    ("axis", "message"),
    [
        ("ppl", "PPL HQ/2 simultaneous low"),
        ("zeroshot", "zero-shot HQ/2 simultaneous low"),
    ],
)
def test_renderer_rejects_tampered_bonferroni_interval(tmp_path, axis, message):
    document = _aggregate()
    if axis == "ppl":
        interval = document["ppl"]["secondary_interval_robustness_vs_turboquant"]["HQ"]["2"][
            "bonferroni_simultaneous_familywise95_geometric_mean_ratio"
        ]
    else:
        interval = document["zeroshot"]["secondary_interval_robustness_vs_turboquant"]["HQ"]["2"][
            "bonferroni_simultaneous_familywise95_ci"
        ]
    interval[0] -= 0.001
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match=message):
        renderer.render(aggregate, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "axis",
    ("ppl", "zeroshot", "mechanism", "ruler", "longbench", "math500", "longcodeqa"),
)
def test_renderer_rejects_model_selection_role_drift_on_every_quality_axis(tmp_path, axis):
    document = _aggregate()
    document[axis]["model_evaluation_roles"]["Qwen3-8B"] = "unexpected_role_drift"
    aggregate = tmp_path / "aggregate.json"
    _write(aggregate, document)
    output = tmp_path / "generated"
    with pytest.raises(renderer.RenderError, match="evaluation/selection roles"):
        renderer.render(aggregate, output)
    assert not output.exists()
