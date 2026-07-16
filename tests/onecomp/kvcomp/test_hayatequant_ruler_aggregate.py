"""Fail-closed tests for the NeMo-Skills RULER-v1 100/task aggregation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/aggregate_hayatequant_paper_shards.py"
SPEC = importlib.util.spec_from_file_location("ruler_paper_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
shards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shards)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_formal_ruler_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write(model_path / "config.json", {"max_position_embeddings": 8192})
    _write(model_path / "generation_config.json", {"eos_token_id": [7, 8]})
    _write(model_path / "tokenizer.json", {"version": "synthetic"})
    chat_template = "{{ messages }} synthetic chat template"
    _write(model_path / "tokenizer_config.json", {"chat_template": chat_template})

    manifest_root = tmp_path / "ruler-data"
    context = 4096
    source_rows: dict[str, list[dict]] = {}
    file_group = {}
    for task_index, task in enumerate(shards.RULER_TASKS):
        task_path = manifest_root / str(context) / task / "test.jsonl"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with task_path.open("w", encoding="utf-8") as stream:
            for row_position in range(shards.RULER_ROWS_PER_TASK):
                row = {
                    "index": row_position // 2,  # upstream random IDs need not be unique
                    "input": f"{task} synthetic question {row_position}",
                    "answer_prefix": " Answer:",
                    "outputs": ["alpha", "beta"],
                    "length": 20 + row_position % 2,
                    "length_w_model_temp": 24 + row_position % 2,
                }
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
        source_rows[task] = rows
        file_group[task] = {
            "path": str(task_path.resolve()),
            "bytes": task_path.stat().st_size,
            "sha256": _digest(task_path),
            "rows": shards.RULER_ROWS_PER_TASK,
            "length_w_model_temp_min": 24,
            "length_w_model_temp_max": 25,
        }

    metadata = {}
    for filename in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = model_path / filename
        metadata[filename] = {"bytes": path.stat().st_size, "sha256": _digest(path)}
    manifest = {
        "schema_version": 1,
        "benchmark": shards.RULER_PROTOCOL_LABEL,
        "protocol": {
            "tasks": list(shards.RULER_TASKS),
            "num_samples_per_task": shards.RULER_ROWS_PER_TASK,
            "random_seed": 42,
            "tokenizer_type": "hf",
            "lengths": [context],
            "chat_template_reserve": 50,
            "prepare_for_nemo_skills": True,
            "sampling_protocol": shards.RULER_SAMPLING_PROTOCOL,
        },
        "sources": {
            "ruler_commit": shards.RULER_SOURCE_COMMIT,
            "nemo_skills_commit": shards.NEMO_SKILLS_RULER_SOURCE_COMMIT,
        },
        "model": {
            "key": "synthetic-model",
            "path": str(model_path.resolve()),
            "revision": "a" * 40,
            "snapshot_revisions": ["a" * 40],
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
            "generation_eos_token_ids": [7, 8],
            "generation_eos_token_ids_source": "model.generation_config.eos_token_id",
            "metadata_files": metadata,
        },
        "files": {str(context): file_group},
    }
    manifest_path = manifest_root / "manifest.json"
    _write(manifest_path, manifest)

    per_task = {}
    per_example = {}
    input_summaries = {}
    task_accuracies = []
    all_inputs = []
    all_generated = []
    all_reserved = []
    for task_index, task in enumerate(shards.RULER_TASKS):
        records = []
        credits = []
        inputs = []
        max_new = shards.RULER_MAX_NEW[task]
        for row_position, source in enumerate(source_rows[task]):
            # All-reference tasks get alternating 1/0.5 credit; QA's any-match
            # rule gives every row full credit.  This exercises true macro
            # recomputation rather than an all-100 shortcut.
            prediction = "alpha beta" if row_position % 2 == 0 else "alpha"
            credit = 1.0 if task.startswith("qa_") else (1.0 if row_position % 2 == 0 else 0.5)
            input_tokens = 100 + task_index + row_position % 3
            generated_tokens = 2
            records.append(
                {
                    "row_position": row_position,
                    "index": source["index"],
                    "released_length": source["length"],
                    "released_length_with_template": source["length_w_model_temp"],
                    "input_tokens": input_tokens,
                    "generated_tokens": generated_tokens,
                    "max_new_tokens": max_new,
                    "stop_token_ids": [7, 8],
                    "prompt_truncated": False,
                    "prediction": prediction,
                    "references": list(source["outputs"]),
                    "credit": round(credit, 6),
                    "compression_aggregation": "fullkv_no_cache",
                    "compression_fp16_bits": None,
                    "compression_stored_bits": None,
                    "adaptive_hq_runtime": None,
                }
            )
            credits.append(credit)
            inputs.append(input_tokens)
            all_inputs.append(input_tokens)
            all_generated.append(generated_tokens)
            all_reserved.append(input_tokens + max_new)
        credit_sum = math.fsum(credits)
        accuracy = credit_sum / len(credits)
        per_task[task] = {
            "n": len(credits),
            "credit_sum": round(credit_sum, 10),
            "score": round(100.0 * accuracy, 4),
            "accuracy": round(accuracy, 6),
            "accuracy_stderr": round(statistics.stdev(credits) / math.sqrt(len(credits)), 6),
            "max_new_tokens": max_new,
            "mem_fraction": 1.0,
            "compression_sufficient_statistics": {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "fullkv_no_cache",
                "compression_observations": len(records),
                "adaptive_hq_runtime": None,
            },
        }
        per_example[task] = records
        input_summaries[task] = {
            "min": min(inputs),
            "median": round(float(statistics.median(inputs)), 1),
            "max": max(inputs),
        }
        task_accuracies.append(accuracy)

    task_accuracy_sum = math.fsum(task_accuracies)
    macro_accuracy = task_accuracy_sum / len(shards.RULER_TASKS)
    combined_digest = hashlib.sha256(
        "".join(
            f"{task}:{file_group[task]['sha256']}\n" for task in sorted(shards.RULER_TASKS)
        ).encode("utf-8")
    ).hexdigest()
    ruler = {
        "schema_version": shards.RULER_RESULT_SCHEMA_VERSION,
        "benchmark": shards.RULER_PROTOCOL_LABEL,
        "metric": "13-task macro accuracy",
        "score": round(100.0 * macro_accuracy, 4),
        "accuracy": round(macro_accuracy, 6),
        "score_sufficient_statistics": {
            "task_accuracy_sum": round(task_accuracy_sum, 10),
            "task_count": len(shards.RULER_TASKS),
        },
        "n": len(shards.RULER_TASKS) * shards.RULER_ROWS_PER_TASK,
        "tasks": list(shards.RULER_TASKS),
        "per_task": per_task,
        "mem_fraction": 1.0,
        "start": 0,
        "limit_per_task": None,
        "samples_per_task": shards.RULER_ROWS_PER_TASK,
        "sampling_protocol": shards.RULER_SAMPLING_PROTOCOL,
        "protocol_complete": True,
        "canonical_nvidia_samples_per_task": (shards.RULER_CANONICAL_NVIDIA_ROWS_PER_TASK),
        "canonical_nvidia_full_eval": False,
        "protocol_scope": shards.RULER_PROTOCOL_SCOPE,
        "reference_model_wrapper_exact": False,
        "evaluation_scope": shards.RULER_COMPLETE_EVALUATION_SCOPE,
        "input_tokens": input_summaries,
        "max_context_tokens": context,
        "model_context": {
            "source": "model.config.get_text_config().max_position_embeddings",
            "max_position_embeddings": 8192,
            "named_ruler_sequence_limit": context,
            "max_observed_prompt_tokens": max(all_inputs),
            "max_observed_prompt_plus_max_generation_tokens": max(all_reserved),
            "max_observed_generated_tokens": max(all_generated),
            "capacity_check": shards.RULER_CAPACITY_CHECK,
        },
        "dataset_files": {
            task: {
                "path": file_group[task]["path"],
                "sha256": file_group[task]["sha256"],
                "rows": shards.RULER_ROWS_PER_TASK,
                "row_format": "nvidia_ruler_input_plus_answer_prefix",
            }
            for task in shards.RULER_TASKS
        },
        "combined_dataset_sha256": combined_digest,
        "generation_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _digest(manifest_path),
            "content": manifest,
        },
        "protocol": {
            "ruler_repository": "https://github.com/NVIDIA/RULER",
            "ruler_source_commit": shards.RULER_SOURCE_COMMIT,
            "nemo_skills_repository": "https://github.com/NVIDIA-NeMo/Skills",
            "nemo_skills_source_commit": shards.NEMO_SKILLS_RULER_SOURCE_COMMIT,
            "task_set": "RULER v1 13-task synthetic setup",
            "sampling_protocol": shards.RULER_SAMPLING_PROTOCOL,
            "sequence_length": context,
            "prompt": shards.RULER_PROMPT_DESCRIPTION,
            "prompt_wrappers": [
                "generic user message; tokenizer.apply_chat_template with "
                "enable_thinking=false when supported"
            ],
            "chat_template_kwargs": dict(shards.RULER_CHAT_TEMPLATE_KWARGS),
            "tokenizer_name_or_path": str(model_path.resolve()),
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
            "generation": shards.RULER_GENERATION_DESCRIPTION,
            "max_new_tokens": dict(shards.RULER_MAX_NEW),
            "eos_token_ids": [7, 8],
            "eos_token_id_source": "model.generation_config.eos_token_id",
            "stop_conditions": [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
            ],
            "scoring": shards.RULER_SCORING_DESCRIPTION,
            "truncation": "none; context overflow is fatal",
            "context_semantics": shards.RULER_CONTEXT_SEMANTICS,
        },
        "per_example": per_example,
        "compression_sufficient_statistics": {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": len(shards.RULER_TASKS) * shards.RULER_ROWS_PER_TASK,
            "adaptive_hq_runtime": None,
        },
    }
    config = {
        "model_path": str(model_path.resolve()),
        "skip_outliers": False,
        "sink_tokens": 0,
        "recent_tokens": 0,
        "run_official_ruler_v1": True,
        "official_ruler_data_root": str((manifest_root / str(context)).resolve()),
        "official_ruler_manifest_path": str(manifest_path.resolve()),
        "official_ruler_tasks": [],
        "official_ruler_max_context": context,
        "official_ruler_start": 0,
        "official_ruler_limit": None,
    }
    input_manifest_path = tmp_path / "paper-inputs.json"
    _write(input_manifest_path, {"schema_version": 1, "models": {"model": {}}})
    input_manifest_record = {
        "path": str(input_manifest_path.resolve()),
        "bytes": input_manifest_path.stat().st_size,
        "sha256": _digest(input_manifest_path),
    }
    smoke_keys = list(shards.FORMAL_SMOKE_JOB_KEYS)

    def smoke_records(root: str) -> list[dict]:
        return [
            {
                "key": key,
                "path": str((tmp_path / root / f"{index:02d}-{key}.json").resolve()),
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, key in enumerate(smoke_keys)
        ]

    formal_smoke_approval = {
        "verification_mode": shards.FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE,
        "marker": {
            "path": str((tmp_path / "manifests/formal_smoke_approval.json").resolve()),
            "bytes": 4096,
            "sha256": "d" * 64,
        },
        "validated_plan_sha256": "e" * 64,
        "smoke_plan_identity": {
            "version": shards.FORMAL_SMOKE_PLAN_VERSION,
            "sha256": "f" * 64,
        },
        "code_commit": "test-commit",
        "input_manifest": dict(input_manifest_record),
        "completed_job_keys": smoke_keys,
        "results": smoke_records("results/smoke"),
        "worker_preflights": smoke_records("provenance"),
    }
    weight_path = model_path / "model.safetensors"
    weight_path.write_bytes(b"synthetic-weight-shard")
    preflight_record = {
        "schema_version": 1,
        "code": {"commit": "test-commit", "tracked_dirty": False},
        "host": {"python": "3.12.10 synthetic-test-runtime"},
        "packages": {name: "1.0.0" for name in shards.QUALITY_SOFTWARE_PACKAGES},
        "gpu": {
            "available": True,
            "torch_cuda": "12.8",
            "cudnn": 91002,
            "devices": [
                {
                    "name": "NVIDIA B200",
                    "compute_capability": [10, 0],
                }
            ],
        },
        "model": {
            "key": "model",
            "path": str(model_path.resolve()),
            "weight_verification_mode": "full_sha256",
            "weights": {
                "schema_version": 1,
                "fingerprinted_file_count": 1,
                "fingerprinted_bytes": weight_path.stat().st_size,
                "selected_format": "safetensors",
                "layouts": {
                    "safetensors": {
                        "shards": [
                            {
                                "relative_path": weight_path.name,
                                "bytes": weight_path.stat().st_size,
                                "sha256": _digest(weight_path),
                            }
                        ]
                    }
                },
            },
            "input_manifest": input_manifest_record,
        },
        "dataset_files": [input_manifest_record],
        "formal_smoke_approval": formal_smoke_approval,
        "environment": {
            **shards.QUALITY_SEMANTIC_ENVIRONMENT,
            "KVCOMP_CODEBOOK_DIR": None,
        },
    }
    preflight_path = tmp_path / "preflight.json"
    _write(preflight_path, preflight_record)
    semantic_config = dict(config)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"semantic_config": semantic_config, "code_commit": "test-commit"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    document = {
        "schema_version": 2,
        "run_fingerprint": fingerprint,
        "code_commit": "test-commit",
        "semantic_config": semantic_config,
        "model": "model",
        "config": config,
        "run_provenance": [
            {
                "path": str(preflight_path.resolve()),
                "sha256": _digest(preflight_path),
                "record": preflight_record,
            }
        ],
        "results": {
            "FullKV": {
                "cache_policy": {
                    "sink_tokens": 0,
                    "recent_tokens": 0,
                    "skip_layers": [],
                },
                "ruler_v1": ruler,
            }
        },
    }
    result_path = tmp_path / "results" / "results_model.json"
    _write(result_path, document)
    return result_path.parent, manifest_path, manifest_root / str(context), document


def _verify(root: Path) -> dict:
    return shards.verify_ruler(
        root,
        expected_models=("model",),
        contexts=(4096,),
        methods=("FullKV",),
    )


def test_ruler_aggregate_regrades_all_1300_records(tmp_path):
    root, _, _, document = _build_formal_ruler_fixture(tmp_path)
    result = _verify(root)
    row = result["models"]["model"]["4096"]["FullKV"]
    assert row["score"] == document["results"]["FullKV"]["ruler_v1"]["score"]
    assert row["audit_records_validated"] == 1300
    assert result["audit_records_validated"] == 1300
    assert result["aggregation"] == "regrade_1300_predictions_then_recompute_13_task_macro"
    assert result["benchmark"] == shards.RULER_PROTOCOL_LABEL
    assert result["sampling_protocol"] == shards.RULER_SAMPLING_PROTOCOL


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda ruler: ruler.__setitem__("schema_version", 0), "schema/scope"),
        (
            lambda ruler: ruler.__setitem__("sampling_protocol", "canonical NVIDIA 500/task"),
            "schema/scope",
        ),
        (
            lambda ruler: ruler["protocol"].__setitem__("ruler_source_commit", "bad"),
            "protocol metadata",
        ),
        (
            lambda ruler: ruler["protocol"].__setitem__(
                "sampling_protocol", "canonical NVIDIA 500/task"
            ),
            "protocol metadata",
        ),
        (
            lambda ruler: ruler["protocol"].__setitem__(
                "chat_template_kwargs", {"enable_thinking": False}
            ),
            "protocol metadata",
        ),
        (
            lambda ruler: ruler["protocol"].__setitem__("eos_token_ids", [7]),
            "protocol metadata",
        ),
        (
            lambda ruler: ruler["per_example"]["niah_single_1"].pop(),
            "exactly 100",
        ),
        (
            lambda ruler: ruler["per_example"]["niah_single_1"][0].__setitem__(
                "prediction", "no matching reference"
            ),
            "credit row",
        ),
        (
            lambda ruler: ruler["per_task"]["niah_single_1"].__setitem__("credit_sum", 99.0),
            "credit numerator",
        ),
        (lambda ruler: ruler.__setitem__("score", 99.0), "macro score"),
        (
            lambda ruler: ruler["per_example"]["niah_single_1"][0].__setitem__(
                "input_tokens", 4096 - shards.RULER_MAX_NEW["niah_single_1"] + 1
            ),
            "named RULER sequence budget",
        ),
        (
            lambda ruler: ruler["per_example"]["niah_single_1"][0].__setitem__(
                "prompt_truncated", True
            ),
            "named RULER sequence budget",
        ),
        (
            lambda ruler: ruler["model_context"].__setitem__("max_position_embeddings", 4096),
            "model-context",
        ),
        (
            lambda ruler: ruler["per_example"]["niah_single_1"][0].__setitem__(
                "references", ["tampered"]
            ),
            "source identity",
        ),
    ],
)
def test_ruler_aggregate_rejects_result_tampering(tmp_path, mutation, match):
    root, _, _, document = _build_formal_ruler_fixture(tmp_path)
    ruler = document["results"]["FullKV"]["ruler_v1"]
    mutation(ruler)
    _write(root / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match=match):
        _verify(root)


def test_ruler_aggregate_rejects_manifest_file_tampering(tmp_path):
    root, manifest_path, _, _ = _build_formal_ruler_fixture(tmp_path)
    manifest_path.write_text(manifest_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(shards.AggregateError, match="manifest SHA-256 mismatch"):
        _verify(root)


def test_ruler_aggregate_rejects_hashed_task_file_tampering(tmp_path):
    root, _, context_root, _ = _build_formal_ruler_fixture(tmp_path)
    task_path = context_root / "niah_single_1" / "test.jsonl"
    task_path.write_text(task_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(shards.AggregateError, match="Blank RULER JSONL row"):
        _verify(root)


def test_ruler_aggregate_rejects_generation_config_tampering(tmp_path):
    root, _, _, document = _build_formal_ruler_fixture(tmp_path)
    model_path = Path(document["config"]["model_path"])
    _write(model_path / "generation_config.json", {"eos_token_id": [7]})
    with pytest.raises(shards.AggregateError, match="model metadata changed"):
        _verify(root)
