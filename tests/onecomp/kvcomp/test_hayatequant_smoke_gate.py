"""Adversarial tests for the formal-run smoke approval marker."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/validate_hayatequant_smoke_gate.py"
SPEC = importlib.util.spec_from_file_location("hayatequant_smoke_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _blockquant_identity() -> dict:
    config = {
        "name": "blockquant",
        "bits": 3,
        "value_bits": 2,
        "block_size": 3,
        "key_n_outlier": 32,
        "outlier_bits": 4,
        "key_outlier_indices": None,
        "remainder_policy": "lower_dimensional_tail_v1",
        "assignment_mode": "exact-full-codebook",
        "lut": None,
        "paper_kv_recipe": True,
        "independent_baseline": True,
        "reference": "arXiv:2605.19972v1",
    }
    return {
        "codec_config": config,
        "resolved_codec_config": {
            "codec": "blockquant",
            "head_dim": 128,
            "paper_reproduction": False,
            "config": config,
        },
        "method_role": "external_quality_baseline",
        "external_quality_baseline": True,
        "proposal_method": False,
        "paper_comparable": False,
        "paper_reproduction": False,
        "speed_comparison": False,
    }


def _blockquant_accounting() -> dict:
    vectors = 32
    identity_fields = 4
    identity_bits = identity_fields * 101
    key_total = vectors * 416 + identity_bits + vectors * 32
    value_total = vectors * 256 + vectors * 16
    total = key_total + value_total
    coordinates = 2 * vectors * 128
    return {
        "stored_bits": total,
        "stored_coordinates": coordinates,
        "stored_bits_per_coord": total / coordinates,
        "stored_bit_accounting": {
            "physical_packing_claim": False,
            "vectors_per_role": vectors,
            "head_dim": 128,
            "key": {
                "code_bits": vectors * 416,
                "dynamic_channel_identity_bits": identity_bits,
                "dynamic_channel_identity_fields": identity_fields,
                "dynamic_channel_identity_coding": "combinadic_rank_fixed_width_v1",
                "fp16_scale_bits": vectors * 32,
                "total_bits": key_total,
            },
            "value": {
                "payload_bits": vectors * 256,
                "fp16_scale_bits": vectors * 16,
                "total_bits": value_total,
            },
            "total_bits": total,
        },
    }


def _hayate_identity(method: str) -> dict:
    adaptive = method == "AdaptiveHQ"
    config = {
        "name": "hayatequant",
        "bits": 3,
        "value_bits": 3,
        "avg_bits": 3.0,
        "dist": "gaussian",
        "block_size": 2,
        "max_codebook_bits": 9,
        "rotation_mode": "auto",
        "encoder": "exact",
        "key_scale_mode": "nc",
        "value_scale_mode": "ls_iter",
        "n_outlier": 0,
        "key_error_weight": 4.0,
        "value_error_weight": 1.0,
        "required_codebook_bundle": (
            gate.ADAPTIVE_HQ_CODEBOOK_BUNDLE if adaptive else gate.HQ_CODEBOOK_BUNDLE
        ),
        "adaptive_bit_split": "online_stats" if adaptive else None,
        "key_bit_fraction": 7 / 12 if adaptive else None,
        "key_code_bits_per_block": None if adaptive else 7,
        "value_code_bits_per_block": None if adaptive else 5,
    }
    resolved = {
        "adaptive_allocation": None,
        "codebook_artifacts": gate._expected_hayate_codebook_artifacts(  # pylint: disable=protected-access
            gate.ADAPTIVE_HQ_CODEBOOK_BUNDLE if adaptive else gate.HQ_CODEBOOK_BUNDLE
        ),
    }
    if adaptive:
        resolved["adaptive_allocation"] = {
            "pair_code_bits_per_block": 12,
            "candidate_pairs": [[6, 6], [7, 5]],
            "selector_bits_per_encode": 1,
            "policy_version": gate.ADAPTIVE_HQ_POLICY_VERSION,
            "distortion_table_version": gate.ADAPTIVE_HQ_DISTORTION_TABLE_VERSION,
            "objective": gate.ADAPTIVE_HQ_OBJECTIVE,
            "selector_carrier": "first_value_fp16_scale_low_fraction_bits",
            "selector_carrier_version": "value_scale_fp16_fraction_lsb_v1",
            "selector_carrier_distortion_amortization": (gate.ADAPTIVE_HQ_CARRIER_AMORTIZATION),
            "selector_incremental_stored_bits_per_encode": 0,
            "prior_pair": [7, 5],
            "incumbent_pair": [7, 5],
            "switch_margin_fraction": gate.ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION,
            "switch_margin_basis": gate.ADAPTIVE_HQ_SWITCH_MARGIN_BASIS,
            "hysteresis": gate.ADAPTIVE_HQ_HYSTERESIS,
        }
    return {
        "codec_config": config,
        "resolved_codec_config": resolved,
        "method_role": "paper_proposal_adaptive" if adaptive else "paper_proposal_fixed",
        "proposal_method": True,
        "proposal_family": "HayateQuant",
        "proposal_variant": "causal_adaptive" if adaptive else "fixed_global",
        "paper_comparable": True,
        "speed_comparison": True,
        "allocation_scope": (
            "cache_local_layer_local_causal_online_stats"
            if adaptive
            else "globally_fixed_kv_code_bits"
        ),
    }


def _adaptive_runtime() -> dict:
    return {
        "schema_version": 1,
        "policy_version": gate.ADAPTIVE_HQ_POLICY_VERSION,
        "encoded_updates": 1,
        "encoded_tokens": 10,
        "selector_bits_total": 1,
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": "value_scale_fp16_fraction_lsb_v1",
        "split_histogram": {"6:6": {"pair": [6, 6], "updates": 1, "tokens": 10, "vectors": 10}},
    }


def _compression_record(method: str, *, input_tokens: int, max_new: int, generated: int) -> dict:
    if method == "FullKV":
        return {
            "input_tokens": input_tokens,
            "max_new_tokens": max_new,
            "generated_tokens": generated,
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "adaptive_hq_runtime": None,
        }
    fp16_bits = 1600
    stored_bits = 400
    return {
        "input_tokens": input_tokens,
        "max_new_tokens": max_new,
        "generated_tokens": generated,
        "compression_fp16_bits": fp16_bits,
        "compression_stored_bits": stored_bits,
        "compression_aggregation": "exact_cache_bit_counters",
        "adaptive_hq_runtime": _adaptive_runtime() if method == "AdaptiveHQ" else None,
    }


def _compression_statistics(method: str, n: int) -> dict:
    if method == "FullKV":
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": n,
            "adaptive_hq_runtime": None,
        }
    runtime = None
    if method == "AdaptiveHQ":
        runtime = _adaptive_runtime()
        runtime["encoded_updates"] = n
        runtime["encoded_tokens"] = 10 * n
        runtime["selector_bits_total"] = n
        runtime["split_histogram"]["6:6"]["updates"] = n
        runtime["split_histogram"]["6:6"]["tokens"] = 10 * n
        runtime["split_histogram"]["6:6"]["vectors"] = 10 * n
    return {
        "compression_fp16_bits": 1600 * n,
        "compression_stored_bits": 400 * n,
        "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
        "compression_observations": n,
        "adaptive_hq_runtime": runtime,
    }


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("policy_version",), "fabricated-policy"),
        (("selector_carrier_version",), "fabricated-carrier"),
        (("encoded_updates",), True),
        (("encoded_tokens",), True),
        (("selector_bits_total",), True),
        (("selector_bits_total",), 2),
        (("split_histogram", "6:6", "updates"), 0),
        (("split_histogram", "6:6", "tokens"), 0),
        (("split_histogram", "6:6", "vectors"), 0),
        (("split_histogram", "6:6", "vectors"), 9),
    ],
)
def test_adaptive_runtime_summary_rejects_noncurrent_or_inconsistent_ledger(path, bad_value):
    runtime = copy.deepcopy(_adaptive_runtime())
    target = runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(gate.SmokeGateError, match="AdaptiveHQ"):
        gate._validate_adaptive_runtime_summary(  # pylint: disable=protected-access
            runtime, label="adversarial runtime"
        )


def _populate_axis_accounting(axis: dict, *, kind: str, method: str, n: int) -> dict:
    axis = copy.deepcopy(axis)
    axis["example_cache_lifetime"] = gate.PER_EXAMPLE_CACHE_LIFETIME
    axis["cross_iteration_tensor_retention"] = (
        gate.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
    )
    if kind == "ruler_v1":
        records = axis["per_example"]["niah_single_1"]
        for record in records:
            record.update(_compression_record(method, input_tokens=100, max_new=128, generated=4))
    elif kind == "longbench":
        records = axis["per_example"]["passage_retrieval_en"]
        for record in records:
            record.update(_compression_record(method, input_tokens=100, max_new=32, generated=4))
    elif kind == "math500":
        records = []
        axis["per_question"] = []
        for index in range(n):
            sample = _compression_record(method, input_tokens=100, max_new=256, generated=4)
            sample.update(
                {
                    "generated_text": f"reasoning {index} final answer {index}",
                    "raw_generated_text": f"reasoning {index} final answer {index}",
                    "grading_input": "generated_text",
                    "legacy_regex_extracted_answer_non_grading": str(index),
                    "correct": index == 0,
                }
            )
            records.append(sample)
            axis["per_question"].append({"input_tokens": 100, "samples": [sample]})
    else:
        records = axis["brackets"]["32K"]["per_example"]
        for record in records:
            record.update(_compression_record(method, input_tokens=100, max_new=16, generated=4))
        axis["brackets"]["32K"]["compression_sufficient_statistics"] = _compression_statistics(
            method, n
        )
    axis["compression_sufficient_statistics"] = _compression_statistics(method, n)
    if kind == "math500":
        axis["eviction_applicable"] = False
        axis["eviction_fired"] = None
        axis["compression_ratio"] = 1.0 if method == "FullKV" else 4.0
        axis["compression_ratio_aggregation"] = (
            "fullkv_no_cache" if method == "FullKV" else "exact_sum_fp16_bits_over_sum_stored_bits"
        )
    return axis


def _preflight(
    *,
    run_root: Path,
    work_root: Path,
    input_manifest: Path,
    input_sha256: str,
    spec: dict,
    commit: str,
    native: bool = False,
) -> dict:
    pinned_inputs = json.loads(input_manifest.read_text(encoding="utf-8"))
    pinned_model = pinned_inputs["models"][spec["model_key"]]
    hf_home = Path(pinned_inputs["hf_datasets"]["hf_home"])
    manifest_record = {
        "path": str(input_manifest.resolve()),
        "bytes": input_manifest.stat().st_size,
        "sha256": input_sha256,
    }
    return {
        "schema_version": 1,
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "code": {
            "repo": str(gate.REPO_ROOT),
            "commit": commit,
            "tracked_dirty": False,
            "onecomp_module": str((gate.REPO_ROOT / "onecomp/__init__.py").resolve()),
        },
        "model": {
            "key": spec["model_key"],
            "path": f"/models/{spec['model']}",
            "metadata_files": pinned_model["metadata_files"],
            "weights": pinned_model["weights"],
            "weight_verification_mode": "full_sha256",
            "input_manifest": manifest_record,
        },
        "pinned_input_files": gate._expected_full_hash_summary(input_manifest),
        "dataset_files": [manifest_record]
        + gate._expected_dataset_records(input_manifest, spec["dataset_selectors"]),
        "environment": {
            "KVCOMP_CODEBOOK_DIR": None,
            "HF_HOME": str(hf_home.resolve()),
            "HF_DATASETS_CACHE": str((hf_home / "datasets").resolve()),
            "HOME": str((work_root / "runtime/home").resolve()),
            "SLURM_JOB_NAME": "hq-s-native-hybrid" if native else f"hq-s-{spec['key']}",
            **gate.SEMANTIC_ENVIRONMENT,
        },
        "gpu": {
            "available": True,
            "devices": [{"logical_index": 0, "name": "NVIDIA B200"}],
        },
        "host": {
            "hostname": "synthetic-b200",
            "python_executable": "/synthetic/venv/bin/python",
        },
        "packages": {
            "torch": "2.9.0",
            "transformers": "4.57.0",
            "datasets": "4.2.0",
            "huggingface-hub": "0.35.0",
            "pyarrow": "21.0.0",
            "hydra-core": "1.3.2",
            "omegaconf": "2.3.0",
            "filelock": "3.20.0",
            "safetensors": "0.6.0",
            "math-verify": "0.9.0",
            "triton": "3.5.0",
            "vllm": "0.11.0",
        },
    }


def _standard_result(spec: dict, preflight: dict) -> dict:
    record = preflight["record"]
    commit = record["code"]["commit"]
    model_path = record["model"]["path"]
    kind = spec["kind"]
    if kind in {"ruler_v1", "longbench", "math500", "longcodeqa"}:
        config = {
            **gate.LONG_RECIPE_CONFIG,
            **spec["config"],
            "model_path": model_path,
            "output_dir": "/synthetic/output",
            "thinkv": {"auto_configure": False},
            **{flag: False for flag in gate.LONG_RUN_FLAGS},
        }
        active = {
            "ruler_v1": "run_official_ruler_v1",
            "longbench": "run_longbench",
            "math500": "run_math",
            "longcodeqa": "run_longcodeqa",
        }[kind]
        config[active] = True
        if kind == "ruler_v1":
            ruler_manifest = Path(record["dataset_files"][1]["path"])
            config["official_ruler_data_root"] = str(ruler_manifest.parent / "4096")
            config["official_ruler_manifest_path"] = str(ruler_manifest)
        elif kind == "longcodeqa":
            longcode_dataset = Path(record["dataset_files"][1]["path"])
            config["longcodeqa_data_root"] = str(longcode_dataset.parents[1])
    else:
        config = {**spec.get("config", {}), "model_path": model_path}
    document = {
        "schema_version": 2,
        "model": spec["model"],
        "config": config,
        "run_provenance": [preflight],
    }
    if kind == "ppl":
        document["code_commit"] = commit
        document["semantic_config"] = {
            key: value
            for key, value in config.items()
            if key not in {"bits", "methods", "output_dir"}
        }
        def ppl_fields(ppl: float) -> dict:
            mean_nll = math.log(ppl)
            return {
                "ppl": ppl,
                "ppl_nll": mean_nll,
                "ppl_nll_sum": mean_nll * 767,
                "ppl_nll_accumulation": gate.PPL_NLL_ACCUMULATION,
                "ppl_n_tokens": 767,
                "ppl_n_windows": 2,
                "ppl_traversed_tokens": 768,
                "ppl_scored_target_index_set": "[1,768)",
                "ppl_scored_target_ranges": [[1, 512], [512, 768]],
                "ppl_scored_target_coverage": "contiguous_exact_once",
                "ppl_n_skipped_nonfinite": 0,
                "ppl_complete_text_eval": False,
                "ppl_window_cache_lifetime": gate.PPL_WINDOW_CACHE_LIFETIME,
                "ppl_cross_window_tensor_retention": (
                    gate.PPL_CROSS_WINDOW_TENSOR_RETENTION
                ),
                "ppl_max_windows": 2,
                "ppl_source": "wikitext-2-raw-v1/test",
                "ppl_dataset_revision": gate.WIKITEXT2_REVISION,
                "ppl_text_sha256": gate.WIKITEXT2_TEXT_SHA256,
            }

        document["fullkv_ppl"] = {
            **ppl_fields(10.0),
            "ppl_teacher_forced_scoring_protocol": gate.PPL_TEACHER_FORCED_PROTOCOL,
            "ppl_teacher_forced_temporal_dependency": None,
        }
        rate = {}
        for method in gate.COMPRESSED_METHODS:
            rate[method] = {
                **ppl_fields(10.1),
                "ppl_compression_fp16_bits": 16000,
                "ppl_compression_stored_bits": 4000,
                "ppl_teacher_forced_scoring_protocol": (
                    gate.BLOCKQUANT_TEACHER_FORCED_PROTOCOL
                    if method == "BlockQuant-paper"
                    else (
                        gate.ADAPTIVE_HQ_TEACHER_FORCED_PROTOCOL
                        if method == "AdaptiveHQ"
                        else gate.PPL_TEACHER_FORCED_PROTOCOL
                    )
                ),
                "ppl_teacher_forced_temporal_dependency": (
                    gate.BLOCKQUANT_TEMPORAL_DEPENDENCY
                    if method == "BlockQuant-paper"
                    else (gate.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY if method == "AdaptiveHQ" else None)
                ),
            }
            if method == "BlockQuant-paper":
                rate[method].update(_blockquant_identity())
                rate[method].update(_blockquant_accounting())
            elif method in {"HQ", "AdaptiveHQ"}:
                rate[method].update(_hayate_identity(method))
        document["results"] = {"3": rate}
    elif kind == "zeroshot":
        config.update({"schema_version": 2, "code_commit": commit})

        def zero_row(method: str) -> dict:
            task = {
                "n": 4,
                "primary_metric": "acc_norm",
                "primary_score": 0.5,
                "per_example": [{"index": index} for index in range(4)],
                "compression_fp16_bits": None if method == "FullKV" else 6400,
                "compression_stored_bits": None if method == "FullKV" else 1600,
                "teacher_forced_scoring_protocol": (
                    gate.BLOCKQUANT_TEACHER_FORCED_PROTOCOL
                    if method == "BlockQuant-paper"
                    else (
                        gate.ADAPTIVE_HQ_TEACHER_FORCED_PROTOCOL
                        if method == "AdaptiveHQ"
                        else gate.PPL_TEACHER_FORCED_PROTOCOL
                    )
                ),
                "teacher_forced_temporal_dependency": (
                    gate.BLOCKQUANT_TEMPORAL_DEPENDENCY
                    if method == "BlockQuant-paper"
                    else (gate.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY if method == "AdaptiveHQ" else None)
                ),
                "lm_eval_task_metadata_version": gate.LM_EVAL_TASK_METADATA_VERSION,
                "protocol_probe_cache_lifetime": (
                    gate.ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME
                ),
                "choice_cache_lifetime": gate.PER_EXAMPLE_CACHE_LIFETIME,
                "cross_iteration_tensor_retention": (
                    gate.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
                ),
            }
            row = {"tasks": {"piqa": task}}
            if method == "BlockQuant-paper":
                row.update(_blockquant_identity())
            elif method in {"HQ", "AdaptiveHQ"}:
                row.update(_hayate_identity(method))
            return row

        document["results"] = {
            "full": {"FullKV": zero_row("FullKV")},
            "3": {method: zero_row(method) for method in gate.LONG_METHODS},
        }
    elif kind == "speed":
        arguments = {
            **spec["arguments"],
            "model": model_path,
            "output": "/synthetic/speed.json",
            "overwrite": False,
        }
        document["code_commit"] = commit
        document["semantic_config"] = {
            key: value for key, value in arguments.items() if key not in {"output", "overwrite"}
        }
        document["schema"] = {"name": "onecomp.hayatequant_fair_speed", "version": 4}
        document["protocol"] = {
            "arguments": arguments,
            "method_taxonomy": {
                "TurboQuant-paper": {
                    "proposal_method": False,
                    "proposal_variant": None,
                    "fake_quant_implementation": gate.TURBOQUANT_FAKE_QUANT_IMPLEMENTATION,
                },
                "HQ": {
                    "proposal_method": True,
                    "proposal_variant": "fixed_global",
                    "fake_quant_implementation": gate.HQ_FAKE_QUANT_IMPLEMENTATION,
                },
                "AdaptiveHQ": {
                    "proposal_method": True,
                    "proposal_variant": "causal_adaptive",
                    "fake_quant_implementation": gate.ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION,
                },
            },
        }
        method_configs = {"TurboQuant-paper": {"method": "TurboQuant-paper"}}
        for method in ("HQ", "AdaptiveHQ"):
            identity = _hayate_identity(method)
            method_configs[method] = {
                "method": method,
                "codec": {"declared": identity["codec_config"], "effective": {}},
                "resolved_at_model_head_dim": identity["resolved_codec_config"],
            }
        recipes = {
            method: gate._canonical_sha256(method_configs[method]) for method in gate.SPEED_METHODS
        }

        def measurement():
            return {
                "timer": "torch.cuda.Event",
                "raw_paired_repeats": [
                    {
                        "repeat": repeat,
                        "samples": [
                            {"method": method, "elapsed_ms_per_call": 1.0}
                            for method in gate.SPEED_METHODS
                        ],
                    }
                    for repeat in range(5)
                ],
                "summary": {
                    method: {"unit": "ms_per_call", "median": 1.0} for method in gate.SPEED_METHODS
                },
            }

        document["micro"] = {
            "method_configs": method_configs,
            "recipe_sha256": recipes,
            "quality": {
                method: {
                    "key_mse": 0.1,
                    "value_mse": 0.1,
                    "attention_output_relative_l2": 0.1,
                    "stored_bits": 1024,
                    "recipe_sha256": recipes[method],
                }
                for method in gate.SPEED_METHODS
            },
            "recipe_linkage": {
                "payload_quality_and_all_micro_timings_use_same_codec_instances": True
            },
            "rate_contract": {
                "nominal_average_bits": 3,
                "equal_input_shape": True,
                "equal_accounted_payload_bits": True,
                "methods": {method: {} for method in gate.SPEED_METHODS},
            },
            "parity_checks": {
                "proposal_payload_vs_fake_quant_quality": [
                    {
                        "method": method,
                        "role": role,
                        "stored_bits_match": True,
                        "effective_implementation": (
                            {
                                "effective_path": "hayate_fused_roundtrip",
                                "payload_materialized": False,
                                "hayate_override_entered": True,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            }
                            if method == "HQ"
                            else {
                                "effective_path": "inherited_encode_then_decode",
                                "payload_materialized": True,
                                "hayate_override_entered": False,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            }
                        ),
                    }
                    for method in ("HQ", "AdaptiveHQ")
                    for role in ("key", "value")
                ],
            },
            "proposal_contract": {
                "formal_proposals": ["HQ", "AdaptiveHQ"],
                "same_total_pair_code_bit_budget": True,
                "rows": {
                    "HQ": {
                        "proposal_variant": "fixed_global",
                        "pair_code_bits_per_block": 12,
                        "exact_assignment": True,
                        "selector": None,
                    },
                    "AdaptiveHQ": {
                        "proposal_variant": "causal_adaptive",
                        "pair_code_bits_per_block": 12,
                        "exact_assignment": True,
                        "selector": {
                            **_hayate_identity("AdaptiveHQ")["resolved_codec_config"][
                                "adaptive_allocation"
                            ],
                        },
                    },
                },
            },
            "regimes": {
                regime: {
                    "payload_api": {
                        "encode": measurement(),
                        "decode_prebuilt_payload": measurement(),
                        "roundtrip_explicit_encode_then_decode": measurement(),
                    },
                    "research_fake_quant": {
                        "effective_implementations": {
                            "TurboQuant-paper": {
                                "effective_path": "inherited_encode_then_decode",
                                "payload_materialized": True,
                                "hayate_override_entered": False,
                                "runtime_probe": "codec_has_no_hayate_rt_support_probe",
                            },
                            "HQ": {
                                "effective_path": "hayate_fused_roundtrip",
                                "payload_materialized": False,
                                "hayate_override_entered": True,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            },
                            "AdaptiveHQ": {
                                "effective_path": "inherited_encode_then_decode",
                                "payload_materialized": True,
                                "hayate_override_entered": False,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            },
                        },
                        "fake_quant_roundtrip": measurement(),
                    },
                }
                for regime in ("prefill", "decode")
            },
        }
    else:
        document["code_commit"] = commit
        document["semantic_config"] = {
            key: value
            for key, value in config.items()
            if key not in {"methods", "output_dir", "resume"} and not key.startswith("run_")
        }
        if kind == "ruler_v1":
            axis = {
                "schema_version": 3,
                "benchmark": gate.RULER_PROTOCOL_LABEL,
                "metric": "task-subset macro accuracy",
                "n": 2,
                "score": 50.0,
                "accuracy": 0.5,
                "tasks": ["niah_single_1"],
                "start": 0,
                "limit_per_task": 2,
                "samples_per_task": gate.RULER_SAMPLES_PER_TASK,
                "sampling_protocol": gate.RULER_SAMPLING_PROTOCOL,
                "protocol_complete": False,
                "canonical_nvidia_samples_per_task": (
                    gate.RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK
                ),
                "canonical_nvidia_full_eval": False,
                "reference_model_wrapper_exact": False,
                "per_task": {"niah_single_1": {"n": 2, "accuracy": 0.5}},
                "per_example": {"niah_single_1": [{}, {}]},
            }
        elif kind == "longbench":
            axis = {
                "n": 1,
                "score": 50.0,
                "tasks": ["passage_retrieval_en"],
                "limit_per_task": 1,
                "released_data_complete": False,
                "dataset_revision": gate.LONGBENCH_REVISION,
                "max_length": 4096,
                "per_task_n": {"passage_retrieval_en": 1},
                "per_example": {"passage_retrieval_en": [{}]},
            }
        elif kind == "math500":
            axis = {
                "task": "math500",
                "n": 2,
                "k": 1,
                "pass@1": 0.5,
                "correct": 1,
                "grader": "math_verify",
                "grader_version": "0.9.0",
                "dataset_size": 500,
                "complete_dataset_eval": False,
                "selection": {"start": 0, "stop_exclusive": 2, "limit": 2},
                "per_question": [{}, {}],
            }
        else:
            bracket = {
                "metric": "exact_match_accuracy",
                "n": 1,
                "start": 0,
                "limit": 1,
                "protocol_commit": gate.LONGCODEQA_PROTOCOL_COMMIT,
                "dataset": {"revision": gate.LONGCODEQA_REVISION},
                "per_example": [{}],
            }
            axis = {
                "benchmark": "LongCodeBench/LongCodeQA",
                "metric": "micro_exact_match_accuracy",
                "n": 1,
                "correct": 1,
                "accuracy": 1.0,
                "released_data_complete_dataset": False,
                "brackets": {"32K": bracket},
            }
        document["results"] = {}
        for method in gate.LONG_METHODS:
            row = {kind: _populate_axis_accounting(axis, kind=kind, method=method, n=spec["n"])}
            if method == "BlockQuant-paper":
                row.update(_blockquant_identity())
            elif method in {"HQ", "AdaptiveHQ"}:
                row.update(_hayate_identity(method))
            document["results"][method] = row
    if kind == "zeroshot":
        document["run_fingerprint"] = gate._canonical_sha256(config)
    else:
        document["run_fingerprint"] = gate._canonical_sha256(
            {"semantic_config": document["semantic_config"], "code_commit": commit}
        )
    return document


def _build_completed_smoke(tmp_path: Path) -> tuple[Path, Path, str, str]:
    run_root = (
        tmp_path
        / "results-worktree/papers/hayatequant_paper/experiments/results/synthetic-run"
    )
    work_root = tmp_path / "work"
    data_root = tmp_path / "pinned-data"
    (work_root / "runtime/home").mkdir(parents=True)
    (data_root / "hf_home").mkdir(parents=True)
    hf_manifest = data_root / "pinned_hf_datasets_manifest.json"
    ruler_manifest = data_root / "ruler_v1/llama31-8b/manifest.json"
    longcode_dataset = data_root / "LongCodeQA/LQA/32K.json"
    longcode_context = data_root / "longcodeqa_context_llama31-8b.json"
    for path in (hf_manifest, ruler_manifest, longcode_dataset, longcode_context):
        _write(path, {"fixture": str(path.relative_to(data_root))})
    input_manifest = run_root / "manifests/paper_inputs.json"
    _write(
        input_manifest,
        {
            "schema_version": 2,
            "kind": "hayatequant_paper_input_validation",
            "status": "ok",
            "runtime_paths": {
                "work_root": str(work_root.resolve()),
                "home": str((work_root / "runtime/home").resolve()),
            },
            "hf_datasets": {
                "status": "validated",
                "hf_home": str((data_root / "hf_home").resolve()),
                "manifest": gate._file_record(hf_manifest),
                "packages": {
                    "datasets": "4.2.0",
                    "huggingface-hub": "0.35.0",
                    "pyarrow": "21.0.0",
                },
            },
            "ruler_v1": {
                "llama31-8b": {"manifest": gate._file_record(ruler_manifest)}
            },
            "longcodeqa": {
                "brackets": {"32K": gate._file_record(longcode_dataset)},
                "context_fit_manifests": {
                    "llama31-8b": {"manifest": gate._file_record(longcode_context)}
                },
            },
            "models": {
                spec["model_key"]: {
                    "path": f"/models/{spec['model']}",
                    "metadata_files": {},
                    "weights": {
                        "schema_version": 1,
                        "model_key_fixture": spec["model_key"],
                    },
                }
                for spec in gate.SMOKE_PLAN
            },
            "pinned_files": [
                gate._file_record(path)
                for path in (
                    hf_manifest,
                    ruler_manifest,
                    longcode_dataset,
                    longcode_context,
                )
            ],
        },
    )
    input_sha256 = hashlib.sha256(input_manifest.read_bytes()).hexdigest()
    commit = "a" * 40
    native_preflight = _preflight(
        run_root=run_root,
        work_root=work_root,
        input_manifest=input_manifest,
        input_sha256=input_sha256,
        spec=gate.SMOKE_PLAN[-1],
        commit=commit,
        native=True,
    )
    _write(run_root / "provenance/smoke-native-hybrid_123.json", native_preflight)
    for spec in gate.SMOKE_PLAN:
        path = run_root / spec["path"]
        if spec["kind"] == "native":
            arguments = dict(spec["arguments"])
            arguments["output"] = str(path)
            document = {
                "schema": {
                    "name": "onecomp.hayatequant_vllm_native_hybrid_smoke",
                    "version": 1,
                },
                "ok": True,
                "implementation_contract": {
                    "native_hybrid_is_full_hayatequant": False,
                    "eligible_for_full_hayatequant_serving_claim": False,
                    "canonical_frontier_k_r_plus_1_v_r_minus_1": False,
                },
                "provenance": {
                    "hostname": "synthetic-b200",
                    "finished_at_utc": "2026-07-16T01:00:00+00:00",
                    "git_commit": commit,
                    "git_status_porcelain": "",
                    "arguments": arguments,
                },
                "cases": [
                    {
                        "bits": bits,
                        "key_bits": bits,
                        "native_hybrid_is_full_hayatequant": False,
                        "canonical_hayatequant_frontier_recipe": False,
                        "optimized_store_cache_bit_identical": True,
                        "optimized_store_decode_bit_identical": True,
                        "bench_iters": spec["arguments"]["bench_iters"],
                        "store_ms": 1.0,
                        "decode_ms": 1.0,
                        "output_abs_mean": 0.1,
                        "output_abs_max": 1.0,
                    }
                    for bits in (3, 4)
                ],
            }
        else:
            record = _preflight(
                run_root=run_root,
                work_root=work_root,
                input_manifest=input_manifest,
                input_sha256=input_sha256,
                spec=spec,
                commit=commit,
            )
            provenance_path = run_root / f"provenance/smoke-{spec['key']}_123.json"
            _write(provenance_path, record)
            envelope = {
                "path": str(provenance_path.resolve()),
                "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                "record": record,
            }
            document = _standard_result(
                spec,
                envelope,
            )
        _write(path, document)
    return run_root, input_manifest, input_sha256, commit


def test_approval_requires_explicit_statement_and_exact_completed_plan(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    validated = gate.validate_smoke_plan(
        run_root=run_root,
        expected_commit=commit,
        input_manifest=input_manifest,
        input_manifest_sha256=input_sha256,
    )
    marker = run_root / "manifests/formal_smoke_approval.json"
    with pytest.raises(gate.SmokeGateError, match="Explicit approval"):
        gate.approve(marker, validated_plan=validated, approval_statement="")
    gate.approve(
        marker,
        validated_plan=validated,
        approval_statement=gate.APPROVAL_STATEMENT,
    )
    gate.check_marker(marker, validated_plan=validated)
    assert validated["completed_job_keys"] == [spec["key"] for spec in gate.SMOKE_PLAN]


def test_marker_is_invalid_after_any_smoke_result_changes(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    validated = gate.validate_smoke_plan(
        run_root=run_root,
        expected_commit=commit,
        input_manifest=input_manifest,
        input_manifest_sha256=input_sha256,
    )
    marker = run_root / "manifests/formal_smoke_approval.json"
    gate.approve(
        marker,
        validated_plan=validated,
        approval_statement=gate.APPROVAL_STATEMENT,
    )
    result_path = run_root / gate.SMOKE_PLAN[0]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    # Change a non-scientific extension field so the smoke remains valid but
    # its byte identity no longer matches the approved marker.
    document["audit_note"] = "post-approval byte change"
    _write(result_path, document)
    changed = gate.validate_smoke_plan(
        run_root=run_root,
        expected_commit=commit,
        input_manifest=input_manifest,
        input_manifest_sha256=input_sha256,
    )
    with pytest.raises(gate.SmokeGateError, match="stale"):
        gate.check_marker(marker, validated_plan=changed)


def test_tampered_scientific_smoke_config_is_rejected(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[2]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["config"]["official_ruler_limit"] = 1
    document["semantic_config"] = {
        key: value
        for key, value in document["config"].items()
        if key not in {"methods", "output_dir", "resume"} and not key.startswith("run_")
    }
    document["run_fingerprint"] = gate._canonical_sha256(
        {"semantic_config": document["semantic_config"], "code_commit": commit}
    )
    _write(result_path, document)
    with pytest.raises(gate.SmokeGateError, match="exact smoke plan"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_zeroshot_device_tamper_is_rejected_even_with_rehashed_config(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    spec = next(spec for spec in gate.SMOKE_PLAN if spec["key"] == "zeroshot")
    result_path = run_root / spec["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["config"]["device"] = "cuda:1"
    document["run_fingerprint"] = gate._canonical_sha256(document["config"])
    _write(result_path, document)

    with pytest.raises(gate.SmokeGateError, match="exact smoke plan"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_ruler_smoke_rejects_canonical_protocol_relabel(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[2]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    document["results"]["FullKV"]["ruler_v1"][
        "sampling_protocol"
    ] = "canonical NVIDIA/RULER 500/task"
    _write(result_path, document)

    with pytest.raises(gate.SmokeGateError, match="RULER smoke metric/protocol"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("ambiguous_extraction", "grading audit is ambiguous"),
        ("missing_legacy_extraction", "grading audit is ambiguous"),
        ("wrong_grading_input", "grading audit is ambiguous"),
        ("fake_eviction", "non-eviction provenance"),
        ("stale_fullkv_display", "FullKV displayed accounting"),
    ],
)
def test_math_smoke_rejects_misleading_raw_schema(tmp_path, mutation, message):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[4]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    full = document["results"]["FullKV"]["math500"]
    sample = full["per_question"][0]["samples"][0]
    if mutation == "ambiguous_extraction":
        sample["extracted_answer"] = "0"
    elif mutation == "missing_legacy_extraction":
        sample.pop("legacy_regex_extracted_answer_non_grading")
    elif mutation == "wrong_grading_input":
        sample["grading_input"] = "legacy_regex"
    elif mutation == "fake_eviction":
        full["eviction_applicable"] = True
        full["eviction_fired"] = True
    else:
        full["compression_ratio_aggregation"] = (
            "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters"
        )
    _write(result_path, document)
    with pytest.raises(gate.SmokeGateError, match=message):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_long_smoke_rejects_any_additional_enabled_axis(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[3]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    # Long-context fingerprints intentionally omit run_* fields for shard
    # merging, so the smoke gate must bind the one-and-only active axis itself.
    document["config"]["run_ppl"] = True
    _write(result_path, document)
    with pytest.raises(gate.SmokeGateError, match="run_ppl=false"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_speed_smoke_command_model_must_match_preflight_model(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[6]["path"]
    document = json.loads(result_path.read_text(encoding="utf-8"))
    arguments = document["protocol"]["arguments"]
    arguments["model"] = "/different/model-with-same-label"
    document["semantic_config"] = {
        key: value for key, value in arguments.items() if key not in {"output", "overwrite"}
    }
    document["run_fingerprint"] = gate._canonical_sha256(
        {"semantic_config": document["semantic_config"], "code_commit": commit}
    )
    _write(result_path, document)
    with pytest.raises(gate.SmokeGateError, match="approved model"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_unpinned_or_missing_worker_dataset_is_rejected_even_with_rehashed_envelope(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[0]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    envelope = result["run_provenance"][0]
    preflight_path = Path(envelope["path"])
    record = json.loads(preflight_path.read_text(encoding="utf-8"))
    record["dataset_files"] = record["dataset_files"][:1]
    _write(preflight_path, record)
    envelope["record"] = record
    envelope["sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    _write(result_path, result)

    with pytest.raises(gate.SmokeGateError, match="exact pinned dataset/protocol set"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_run_root_home_is_rejected_even_with_rehashed_worker_envelope(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[0]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    envelope = result["run_provenance"][0]
    preflight_path = Path(envelope["path"])
    record = json.loads(preflight_path.read_text(encoding="utf-8"))
    record["environment"]["HOME"] = str((run_root / "runtime/home").resolve())
    _write(preflight_path, record)
    envelope["record"] = record
    envelope["sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    _write(result_path, result)

    with pytest.raises(gate.SmokeGateError, match="stale/dirty model, code, or GPU provenance"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_nonfinite_or_uppercase_result_identity_is_rejected(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    result_path = run_root / gate.SMOKE_PLAN[0]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_fingerprint"] = "F" * 64
    _write(result_path, result)
    with pytest.raises(gate.SmokeGateError, match="wrong result schema"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )

    result["run_fingerprint"] = "f" * 64
    result["fullkv_ppl"]["ppl"] = float("nan")
    _write(result_path, result)
    with pytest.raises(gate.SmokeGateError, match="non-finite JSON"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )


def test_native_result_must_bind_to_exactly_one_worker_preflight(tmp_path):
    run_root, input_manifest, input_sha256, commit = _build_completed_smoke(tmp_path)
    source = run_root / "provenance/smoke-native-hybrid_123.json"
    duplicate = run_root / "provenance/smoke-native-hybrid_456.json"
    duplicate.write_bytes(source.read_bytes())
    with pytest.raises(gate.SmokeGateError, match="exactly one native worker preflight"):
        gate.validate_smoke_plan(
            run_root=run_root,
            expected_commit=commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_sha256,
        )
