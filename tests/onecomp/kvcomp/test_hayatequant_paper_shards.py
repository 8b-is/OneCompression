"""Tests for exact-coverage paper shard aggregation."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import zipfile

import pytest

from onecomp.kvcomp.eval import reasoning as reasoning_eval

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/aggregate_hayatequant_paper_shards.py"
SPEC = importlib.util.spec_from_file_location("paper_shards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
shards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shards)

RENDERER_SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/render_hayatequant_formal_results.py"
RENDERER_SPEC = importlib.util.spec_from_file_location(
    "paper_renderer_for_scope_contract", RENDERER_SCRIPT
)
assert RENDERER_SPEC is not None and RENDERER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(renderer)


def test_longbench_regrader_matches_every_saved_pinned_official_metric_fixture():
    fixture_path = REPO_ROOT / "tests/onecomp/fixtures/longbench_official_metric_fixtures.json"
    fixture_document = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture_document["upstream"]["commit"] == shards.LONGBENCH_PROTOCOL_COMMIT
    metric_functions = {
        "qa_f1_score": shards._longbench_qa_f1,
        "rouge_score": shards._longbench_rouge,
        "classification_score": shards._longbench_classification,
        "retrieval_score": shards._longbench_retrieval,
        "count_score": shards._longbench_count,
        "code_sim_score": shards._longbench_code_similarity,
    }
    for fixture in fixture_document["fixtures"]:
        kwargs = (
            {"all_classes": fixture["all_classes"]}
            if fixture["metric"] == "classification_score"
            else {}
        )
        actual = metric_functions[fixture["metric"]](
            fixture["prediction"], fixture["ground_truth"], **kwargs
        )
        assert actual == pytest.approx(fixture["expected"], abs=1e-12), fixture["id"]


def test_formal_point_gate_rejects_899_of_1000_without_rounding():
    cells = [{"complete": True, "win": index < 899} for index in range(1000)]
    summary = shards._point_estimate_pair_summary(cells, scope="quality")
    assert summary["win_fraction_unrounded"] == 0.899
    assert summary["passed"] is False
    cells[899]["win"] = True
    summary = shards._point_estimate_pair_summary(cells, scope="quality")
    assert summary["win_fraction_unrounded"] == 0.9
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("proposal", "baseline", "matched", "expected"),
    [
        (1.0, 1.0, True, "tie_nonwin"),
        (None, 1.0, True, "missing_or_nonfinite"),
        (float("nan"), 1.0, True, "missing_or_nonfinite"),
        (1.1, 1.0, False, "comparison_mismatch"),
    ],
)
def test_formal_point_gate_tie_missing_nonfinite_and_mismatch_are_nonwins(
    proposal, baseline, matched, expected
):
    status, complete, win = shards._strict_point_estimate_status(
        proposal,
        baseline,
        direction="higher_is_better",
        matched=matched,
    )
    assert status == expected
    assert win is False
    assert complete is (expected == "tie_nonwin")


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": 1, "value": 2}\n',
        '{"value": NaN}\n',
        '{"value": Infinity}\n',
    ],
)
def test_aggregate_json_loader_rejects_duplicate_keys_and_nonfinite_numbers(payload, tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(shards.AggregateError):
        shards._load_object(path)
    with pytest.raises(renderer.RenderError):
        renderer._load_json(path)


def test_public_payload_bit_scope_is_identical_in_aggregator_and_renderer():
    scope = shards.PUBLIC_LOGICAL_PAYLOAD_SCOPE
    assert scope == renderer.PUBLIC_LOGICAL_PAYLOAD_SCOPE
    assert scope["stored_bits_scope"] == "fixed_width_per_token_data_plane_growth_only"
    assert scope["self_describing_control_tags_counted"] is False
    assert scope["python_container_bytes_counted"] is False
    assert scope["persistent_codec_state_counted"] is False
    assert "out-of-band/cache-level" in scope["control_state_interpretation"]
    assert "not physical serving memory" in scope["control_state_interpretation"]


def test_longcodeqa_source_row_audit_is_identical_in_aggregator_and_renderer():
    assert shards.LONGCODE_SOURCE_ROW_AUDIT == renderer.LONGCODE_SOURCE_ROW_AUDIT
    audit = shards.LONGCODE_SOURCE_ROW_AUDIT
    assert audit["retained_source_fields_compared"] == [
        "correct_letter_as_gold",
        "repo",
        "question",
    ]
    assert audit["released_prompt_retained_in_result"] is False
    assert "not claimed" in audit["released_prompt_comparison"]


def _fixture_declared_codec(method: str, rate: int) -> dict:
    if method == "TurboQuant-paper":
        return {
            "name": "turboquant",
            "bits": rate,
            "value_bits": None,
            "dist": "beta",
            "rotation_mode": "dense",
            "seed": 0,
            "n_outlier": 0,
            "outlier_bits": None,
            "use_qjl": False,
            "qjl_roles": "key",
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
    if method == "FibQuant-paper":
        return {
            "name": "fibquant",
            "bits": float(rate),
            "block_size": None,
            "code_bits": None,
            "seed": 0,
            "codebook_seed": 0,
            "key_axis": "token",
            "value_axis": "token",
            "group_size": -1,
            "symmetric": False,
            "skip_layers": [],
            "residual_window": 128,
        }
    if method == "BlockQuant-paper":
        assert rate == 3
        return shards._blockquant_declared_config()
    if method in shards.PROPOSAL_METHODS:
        return {
            "name": "hayatequant",
            "bits": rate,
            "value_bits": rate,
            "dist": "gaussian",
            "block_size": 2,
            "max_codebook_bits": 9,
            "codebook_init": "auto",
            "required_codebook_bundle": (
                shards.HQ_CODEBOOK_BUNDLE if method == "HQ" else shards.ADAPTIVE_HQ_CODEBOOK_BUNDLE
            ),
            "rotation_mode": "auto",
            "seed": 0,
            "encoder": "exact",
            "hadamard_gemm_max_dim": 4096,
            "ls_scale": True,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "n_outlier": 0,
            "outlier_bits": None,
            "key_n_outlier": None,
            "value_n_outlier": None,
            "use_qjl": False,
            "calibration": None,
            "adaptive_bit_split": (None if method == "HQ" else "online_stats"),
            "avg_bits": float(rate),
            "key_bit_fraction": (None if method == "HQ" else (2 * rate + 1) / (4 * rate)),
            "key_code_bits_per_block": (2 * rate + 1 if method == "HQ" else None),
            "value_code_bits_per_block": (2 * rate - 1 if method == "HQ" else None),
            "adaptive_bit_margin": 4,
            "adaptive_stats_min_tokens": 16,
            "key_error_weight": 4.0,
            "value_error_weight": 1.0,
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
    if method in {"HayateQuant-scalar-matched", "HayateQuant-blockvq-matched"}:
        return {
            "name": "hayatequant",
            "bits": rate,
            "value_bits": rate,
            "dist": "beta",
            "block_size": (1 if method == "HayateQuant-scalar-matched" else None),
            "max_codebook_bits": 9,
            "codebook_init": "auto",
            "rotation_mode": "dense",
            "seed": 0,
            "encoder": "exact",
            "ls_scale": False,
            "key_scale_mode": None,
            "value_scale_mode": None,
            "n_outlier": 0,
            "key_n_outlier": None,
            "value_n_outlier": None,
            "use_qjl": False,
            "calibration": None,
            "adaptive_bit_split": None,
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
    raise AssertionError(method)


def _add_formal_recipe_fixture(document: dict) -> None:
    recognized = {
        "FullKV",
        "TurboQuant-paper",
        "FibQuant-paper",
        "BlockQuant-paper",
        "HQ",
        "AdaptiveHQ",
        "HayateQuant-scalar-matched",
        "HayateQuant-blockvq-matched",
    }
    model = str(document.get("model", "model"))

    def add(method: str, row: dict, rate: int | None) -> None:
        row.setdefault(
            "cache_policy",
            {
                "sink_tokens": 0,
                "recent_tokens": 0,
                "skip_layers": [],
            },
        )
        if method == "FullKV":
            return
        assert rate is not None
        declared = row.setdefault("codec_config", {})
        for key, value in _fixture_declared_codec(method, rate).items():
            declared.setdefault(key, value)
        row.setdefault(
            "resolved_codec_config",
            shards._expected_resolved_codec_config(method, rate, 128),
        )
        if method == "FibQuant-paper":
            block_size, code_bits, _ = shards.FIBQUANT_PAPER_RECIPES[rate]
            row.setdefault("method_role", "external_quality_baseline")
            row.setdefault("proposal_method", False)
            row.setdefault("paper_comparable", True)
            row.setdefault("speed_comparison", False)
            row.setdefault("primary_source", "arXiv:2605.11478v1")
            row.setdefault("nominal_payload_bits_per_coordinate", rate)
            row.setdefault(
                "method_definition",
                "FibQuant v1 exact spherical-Beta block-VQ quality baseline "
                f"({rate} payload bpc; k={block_size}, code_bits={code_bits})",
            )
        elif method == "BlockQuant-paper":
            for key, value in shards._blockquant_quality_provenance_contract().items():
                row.setdefault(key, value)
        elif method in shards.PROPOSAL_METHODS:
            for key, value in shards.HAYATE_FORMAL_PROVENANCE[method].items():
                row.setdefault(key, value)
            if method == "AdaptiveHQ":
                if "ppl_teacher_forced_scoring_protocol" in row:
                    row["ppl_teacher_forced_scoring_protocol"] = (
                        shards.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
                    )
                    row["ppl_teacher_forced_temporal_dependency"] = (
                        shards.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY
                    )
                for task in row.get("tasks", {}).values():
                    task["teacher_forced_scoring_protocol"] = (
                        shards.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
                    )
                    task["teacher_forced_temporal_dependency"] = (
                        shards.ADAPTIVE_HQ_TEMPORAL_DEPENDENCY
                    )

    results = document.get("results")
    if not isinstance(results, dict):
        return
    for outer_key, outer_value in results.items():
        if outer_key in recognized and isinstance(outer_value, dict):
            add(outer_key, outer_value, None if outer_key == "FullKV" else 3)
            continue
        if not isinstance(outer_value, dict):
            continue
        for method, row in outer_value.items():
            if method not in recognized or not isinstance(row, dict):
                continue
            add(method, row, None if method == "FullKV" else int(outer_key))


def _write(path: Path, document: dict) -> None:
    if document.get("schema_version") == 2:
        _add_formal_recipe_fixture(document)
        document["code_commit"] = "test-commit"
        document.setdefault("semantic_config", dict(document.get("config", {})))
        document["run_fingerprint"] = (
            __import__("hashlib")
            .sha256(
                json.dumps(
                    {
                        "semantic_config": document["semantic_config"],
                        "code_commit": "test-commit",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            .hexdigest()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


def _model_path(model: str) -> str:
    return f"/models/{model}"


def _formal_smoke_approval_fixture(input_manifest: dict) -> dict:
    keys = list(shards.FORMAL_SMOKE_JOB_KEYS)

    def records(root: str) -> list[dict]:
        return [
            {
                "key": key,
                "path": f"/run/{root}/{index:02d}-{key}.json",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, key in enumerate(keys)
        ]

    return {
        "verification_mode": shards.FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE,
        "marker": {
            "path": "/run/manifests/formal_smoke_approval.json",
            "bytes": 4096,
            "sha256": "d" * 64,
        },
        "validated_plan_sha256": "e" * 64,
        "smoke_plan_identity": {
            "version": shards.FORMAL_SMOKE_PLAN_VERSION,
            "sha256": "f" * 64,
        },
        "code_commit": "test-commit",
        "input_manifest": dict(input_manifest),
        "completed_job_keys": keys,
        "results": records("results/smoke"),
        "worker_preflights": records("provenance"),
    }


def _run_provenance(
    model: str,
    *,
    slurm_job_id: str | None = None,
    input_manifest: dict | None = None,
    dataset_files: list[dict] | None = None,
) -> list[dict]:
    if input_manifest is None:
        input_manifest = {
            "path": "/run/manifests/paper-inputs.json",
            "bytes": 123,
            "sha256": "b" * 64,
        }
    if dataset_files is None:
        dataset_files = [input_manifest]
    environment = {
        "KVCOMP_CODEBOOK_DIR": None,
        "SLURM_JOB_ID": slurm_job_id,
        "ONECOMP_TRITON_CODEC": "0",
        "ONECOMP_TRITON_PREFILL": "0",
        "ONECOMP_COMPILE_CODEC": "0",
        "ONECOMP_COMPILE_DECODE_ONLY": "0",
    }
    if slurm_job_id is not None:
        environment.update(
            {
                "SLURM_JOB_NODELIST": "kagura-gpu06",
                "SLURM_JOB_PARTITION": "batch-8gpu",
                "SLURM_GPUS_ON_NODE": "8",
                "SLURM_JOB_GPUS": "0,1,2,3,4,5,6,7",
                "SLURM_CPUS_PER_TASK": "160",
                "SLURM_JOB_CPUS_PER_NODE": "160",
                "SLURM_MEM_PER_NODE": "2457600",
                "CUDA_VISIBLE_DEVICES": "0",
                "PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE": "0",
            }
        )
    return [
        {
            "path": f"/preflight/{model}.json",
            "sha256": "a" * 64,
            "record": {
                "schema_version": 1,
                "code": {
                    "commit": "test-commit",
                    "tracked_dirty": False,
                    "onecomp_module": "/repo/onecomp/__init__.py",
                },
                "host": {
                    "hostname": "kagura-gpu06",
                    "python": "3.11.9 (main, Jan 1 2026, 00:00:00) [GCC]",
                    "python_executable": "/venv/bin/python",
                },
                "packages": {
                    "torch": "2.8.0",
                    "transformers": "4.57.1",
                    "datasets": "4.8.5",
                    "lm-eval": "0.4.11",
                    "triton": "3.4.0",
                    "vllm": None,
                    "math-verify": "0.9.0",
                    "safetensors": "0.6.2",
                    "rouge": "1.0.1",
                    "fuzzywuzzy": "0.18.0",
                    "fuzzywuzzy-backend": "python-difflib",
                },
                "gpu": {
                    "available": True,
                    "torch_cuda": "12.8",
                    "cudnn": 91002,
                    "devices": [
                        {
                            "logical_index": 0,
                            "name": "NVIDIA B200",
                            "uuid": "GPU-test-only",
                            "total_memory": 1,
                            "compute_capability": [10, 0],
                        }
                    ],
                },
                "model": {
                    "key": model,
                    "path": _model_path(model),
                    "weight_verification_mode": "full_sha256",
                    "weights": {
                        "schema_version": 1,
                        "fingerprinted_file_count": 1,
                        "fingerprinted_bytes": 10,
                        "selected_format": "safetensors",
                        "layouts": {
                            "safetensors": {
                                "shards": [
                                    {
                                        "relative_path": "model.safetensors",
                                        "bytes": 10,
                                        "sha256": "c" * 64,
                                    }
                                ],
                            },
                        },
                    },
                    "input_manifest": input_manifest,
                },
                "dataset_files": dataset_files,
                "environment": environment,
                "formal_smoke_approval": _formal_smoke_approval_fixture(input_manifest),
            },
        }
    ]


def test_real_preflight_envelope_is_accepted_and_dirty_record_is_rejected(tmp_path):
    document = {
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "run_provenance": _run_provenance("model"),
    }
    shards._require_run_provenance(document, path=tmp_path / "result.json")
    document["run_provenance"][0]["record"]["code"]["tracked_dirty"] = True
    with pytest.raises(shards.AggregateError, match="clean preflight"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")


def test_formal_provenance_requires_declared_commit_and_reproducible_fingerprint(
    tmp_path,
):
    document = {
        "schema_version": 2,
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "semantic_config": {"model_path": _model_path("model")},
        "run_fingerprint": "d" * 64,
        "run_provenance": _run_provenance("model"),
    }
    with pytest.raises(shards.AggregateError, match="declare its code commit"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")

    document["code_commit"] = "test-commit"
    with pytest.raises(shards.AggregateError, match="fingerprint/preflight commit"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")

    document["run_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "semantic_config": document["semantic_config"],
                "code_commit": "test-commit",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    shards._require_run_provenance(document, path=tmp_path / "result.json")


def test_formal_provenance_accepts_config_fingerprint_without_semantic_config(tmp_path):
    config = {
        "schema_version": 2,
        "model_path": _model_path("model"),
        "device": "cuda:0",
        "code_commit": "test-commit",
    }
    document = {
        "schema_version": 2,
        "model": "model",
        "config": config,
        "run_fingerprint": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "run_provenance": _run_provenance("model"),
    }

    shards._require_run_provenance(document, path=tmp_path / "result.json")


def test_preflight_rejects_model_and_external_codebook_identity_drift(tmp_path):
    document = {
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "run_provenance": _run_provenance("model"),
    }
    document["run_provenance"][0]["record"]["model"]["key"] = "other"
    with pytest.raises(shards.AggregateError, match="model label/config/preflight"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")

    document["run_provenance"] = _run_provenance("model")
    document["run_provenance"][0]["record"]["environment"]["KVCOMP_CODEBOOK_DIR"] = "/external"
    with pytest.raises(shards.AggregateError, match="external KV codebook"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda approval: approval.clear(), "smoke-approval binding"),
        (
            lambda approval: approval.__setitem__("validated_plan_sha256", "0"),
            "approval identity",
        ),
        (
            lambda approval: approval.__setitem__("validated_plan_sha256", "A" * 64),
            "approval identity",
        ),
        (
            lambda approval: approval["results"].reverse(),
            "order/key identity",
        ),
        (
            lambda approval: approval["worker_preflights"][0].__setitem__(
                "path", "/outside/ppl.json"
            ),
            "escapes",
        ),
        (
            lambda approval: approval["results"].__setitem__(
                1, dict(approval["results"][0], key=approval["results"][1]["key"])
            ),
            "duplicate path",
        ),
        (
            lambda approval: approval.__setitem__("code_commit", "other-commit"),
            "approval identity",
        ),
    ],
)
def test_preflight_rejects_missing_or_malformed_worker_smoke_binding(mutation, message, tmp_path):
    document = {
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "run_provenance": _run_provenance("model"),
    }
    approval = document["run_provenance"][0]["record"]["formal_smoke_approval"]
    mutation(approval)
    with pytest.raises(shards.AggregateError, match=message):
        shards._require_run_provenance(document, path=tmp_path / "result.json")


def test_preflight_fails_closed_without_quality_runtime_provenance(tmp_path):
    document = {
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "run_provenance": _run_provenance("model"),
    }
    del document["run_provenance"][0]["record"]["packages"]["transformers"]
    with pytest.raises(shards.AggregateError, match="package version"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")


@pytest.mark.parametrize("package", ["rouge", "fuzzywuzzy", "fuzzywuzzy-backend"])
def test_preflight_fails_closed_without_longbench_scorer_runtime(package, tmp_path):
    document = {
        "model": "model",
        "config": {"model_path": _model_path("model")},
        "run_provenance": _run_provenance("model"),
    }
    del document["run_provenance"][0]["record"]["packages"][package]
    with pytest.raises(shards.AggregateError, match="package version"):
        shards._require_run_provenance(document, path=tmp_path / "result.json")


def _cross_axis_source(
    tmp_path: Path,
    name: str,
    *,
    input_manifest: Path,
    weight_sha256: str = "c" * 64,
    torch_version: str = "2.8.0",
    fuzzy_backend: str = "python-difflib",
    smoke_validated_plan_sha256: str = "e" * 64,
) -> dict:
    manifest_record = {
        "path": str(input_manifest.resolve()),
        "bytes": input_manifest.stat().st_size,
        "sha256": hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
    }
    provenance = _run_provenance("model", input_manifest=manifest_record)
    model = provenance[0]["record"]["model"]
    provenance[0]["record"]["packages"]["torch"] = torch_version
    provenance[0]["record"]["packages"]["fuzzywuzzy-backend"] = fuzzy_backend
    provenance[0]["record"]["formal_smoke_approval"][
        "validated_plan_sha256"
    ] = smoke_validated_plan_sha256
    model["input_manifest"] = manifest_record
    model["weights"]["layouts"]["safetensors"]["shards"][0]["sha256"] = weight_sha256
    provenance[0]["record"]["dataset_files"] = [manifest_record]
    result_path = tmp_path / name / "results_model.json"
    _write(
        result_path,
        {
            "schema_version": 2,
            "model": "model",
            "config": {"model_path": _model_path("model")},
            "run_provenance": provenance,
            "results": {},
        },
    )
    return shards._source(result_path)


def _axis_document(source: dict) -> dict:
    return {
        "commit": "test-commit",
        "official_complete": True,
        "released_data_complete": True,
        "sources": [source],
    }


def test_formal_matrix_complete_requires_all_eight_axes(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    source = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    document = {"ppl": _axis_document(source)}
    shards._finalize_formal_document(document)
    assert document["formal_matrix_complete"] is False
    assert document["formal_matrix_status"]["missing_axes"] == list(shards.FORMAL_AXIS_KEYS[1:])


def test_complete_matrix_preserves_negative_point_estimate_gate_result(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    source = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    document = {axis: _axis_document(source) for axis in shards.FORMAL_AXIS_KEYS}
    shards._finalize_formal_document(document)
    assert document["formal_matrix_status"]["coverage_complete"] is True
    assert document["point_estimate_win_gate"]["passed"] is False
    gate = document["point_estimate_win_gate"]
    assert gate["baseline_applicability"]["BlockQuant-paper"] == {
        "quality": True,
        "speed": False,
        "included_in_denominator": False,
        "rate_match_required": False,
        "operating_point_matched": False,
        "paper_comparable": False,
        "paper_reproduction": False,
        "reason_code": "unmatched_source_operating_point_descriptive_only",
        "completion_specification": shards.BLOCKQUANT_SELECTION_COMPLETION,
    }
    assert all(cell["baseline"] != "BlockQuant-paper" for cell in gate["cells"])
    descriptive = gate["unmatched_descriptive_comparisons"]
    assert descriptive["included_in_point_estimate_gate"] is False
    assert len(descriptive["cells"]) == 160
    assert all(cell["matched_comparison"] is False for cell in descriptive["cells"])
    assert set(gate["proposal_baseline_summaries"]["HQ"]) == {
        "TurboQuant-paper",
        "FibQuant-paper",
        "TurboQuant-paper-speed",
    }
    assert document["formal_matrix_status"]["point_estimate_win_gate_passed"] is False
    assert document["formal_matrix_complete"] is True
    assert document["official_complete"] is True
    assert document["formal_provenance"]["validated_source_files"] == 1


def test_formal_matrix_rejects_mixed_input_manifests_across_axes(tmp_path):
    first_manifest = tmp_path / "paper-inputs-a.json"
    second_manifest = tmp_path / "paper-inputs-b.json"
    first_manifest.write_text('{"schema_version":1,"id":"a"}\n', encoding="utf-8")
    second_manifest.write_text('{"schema_version":1,"id":"b"}\n', encoding="utf-8")
    first = _cross_axis_source(tmp_path, "one", input_manifest=first_manifest)
    second = _cross_axis_source(tmp_path, "two", input_manifest=second_manifest)
    document = {axis: _axis_document(first) for axis in shards.FORMAL_AXIS_KEYS}
    document["speed"] = _axis_document(second)
    with pytest.raises(shards.AggregateError, match="one paper input manifest"):
        shards._finalize_formal_document(document)


def test_formal_matrix_rejects_model_provenance_drift_across_axes(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    first = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    second = _cross_axis_source(tmp_path, "two", input_manifest=manifest, weight_sha256="d" * 64)
    document = {axis: _axis_document(first) for axis in shards.FORMAL_AXIS_KEYS}
    document["speed"] = _axis_document(second)
    with pytest.raises(shards.AggregateError, match="model provenance"):
        shards._finalize_formal_document(document)


def test_formal_matrix_rejects_mixed_quality_runtime_strata(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    first = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    second = _cross_axis_source(tmp_path, "two", input_manifest=manifest, torch_version="2.9.0")
    document = {axis: _axis_document(first) for axis in shards.FORMAL_AXIS_KEYS}
    document["math500"] = _axis_document(second)
    with pytest.raises(shards.AggregateError, match="software/GPU/ONECOMP"):
        shards._finalize_formal_document(document)


def test_formal_matrix_rejects_mixed_worker_validated_smoke_approvals(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    first = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    second = _cross_axis_source(
        tmp_path,
        "two",
        input_manifest=manifest,
        smoke_validated_plan_sha256="9" * 64,
    )
    document = {axis: _axis_document(first) for axis in shards.FORMAL_AXIS_KEYS}
    document["speed"] = _axis_document(second)
    with pytest.raises(shards.AggregateError, match="one worker-validated smoke approval"):
        shards._finalize_formal_document(document)


def test_formal_matrix_rejects_longbench_fuzzy_backend_version_drift(tmp_path):
    manifest = tmp_path / "paper-inputs.json"
    manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    first = _cross_axis_source(tmp_path, "one", input_manifest=manifest)
    second = _cross_axis_source(
        tmp_path,
        "two",
        input_manifest=manifest,
        fuzzy_backend="python-Levenshtein:0.27.1",
    )
    document = {axis: _axis_document(first) for axis in shards.FORMAL_AXIS_KEYS}
    document["longbench"] = _axis_document(second)
    with pytest.raises(shards.AggregateError, match="software/GPU/ONECOMP"):
        shards._finalize_formal_document(document)


def _zero_doc(start: int, stop: int, *, correct: int, overlap_seed: int = 2027) -> dict:
    n = stop - start
    examples = []
    for offset, index in enumerate(range(start, stop)):
        gold = index % 2
        is_correct = offset < correct
        prediction = gold if is_correct else 1 - gold
        examples.append(
            {
                "index": index,
                "gold": gold,
                "prediction": prediction,
                "prediction_norm": prediction,
                "correct": is_correct,
                "correct_norm": is_correct,
            }
        )
    result = {
        "n": n,
        "correct": correct,
        "correct_norm": correct,
        "dataset_size": 4,
        "dataset_revision": "rev",
        "split": "test",
        "evaluator": "lm-eval-compatible@0.4.11",
        "lm_eval_protocol_commit": shards.LM_EVAL_PROTOCOL_COMMIT,
        "lm_eval_task_metadata_version": shards.LM_EVAL_TASK_METADATA_VERSION,
        "num_fewshot": 0,
        "target_delimiter": " ",
        "apply_chat_template": False,
        "compression_ratio": 1.0,
        "compression_aggregation": "fullkv_no_cache",
        "compression_fp16_bits": None,
        "compression_stored_bits": None,
        "teacher_forced_scoring_protocol": shards.PPL_TEACHER_FORCED_PROTOCOL,
        "teacher_forced_temporal_dependency": None,
        "protocol_probe_cache_lifetime": shards.ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME,
        "choice_cache_lifetime": shards.PER_EXAMPLE_CACHE_LIFETIME,
        "cross_iteration_tensor_retention": (
            shards.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ),
        "selection": {
            "start_valid_example": start,
            "stop_valid_example_exclusive": stop,
            "limit": n,
            "sample_seed": overlap_seed,
            "order": "validated_dataset_order",
        },
        "per_example": examples,
    }
    return {
        "schema_version": 2,
        "run_fingerprint": f"fp-{start}",
        "model": "model",
        "tasks": ["task"],
        "start": start,
        "limit": n,
        "sample_seed": 2027,
        "config": {
            "model_path": _model_path("model"),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "bits": [3],
            "tasks": ["task"],
            "methods": ["FullKV"],
            "include_fullkv": False,
            "start": start,
            "limit": n,
            "sample_seed": 2027,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
        "run_provenance": _run_provenance("model"),
        "results": {"3": {"FullKV": {"tasks": {"task": result}}}},
    }


def _pinned_piqa_source_fixture(tmp_path: Path) -> tuple[Path, str, dict, dict]:
    datasets = pytest.importorskip("datasets")
    dataset_dir = tmp_path / "pinned-piqa"
    datasets.Dataset.from_dict(
        {
            "goal": ["goal zero", "goal one", "goal two", "goal three"],
            "sol1": ["first a", "first b", "first c", "first d"],
            "sol2": ["second a", "second b", "second c", "second d"],
            "label": [0, 1, 0, 1],
        }
    ).save_to_disk(dataset_dir)
    arrow = next(dataset_dir.glob("*.arrow"))
    manifest = tmp_path / "paper-inputs.json"
    task_spec = {
        "size": 4,
        "split": "validation",
        "revision": "rev",
        "primary": "acc_norm",
        "manifest_key": "piqa_validation",
        "repo": "baber/piqa",
        "config": None,
    }
    manifest.write_text(
        json.dumps(
            {
                "hf_datasets": {
                    "status": "validated",
                    "datasets": {
                        "piqa_validation": {
                            "repo": "baber/piqa",
                            "config": None,
                            "split": "validation",
                            "revision": "rev",
                            "rows": 4,
                            "cache_files": [
                                {
                                    "path": str(arrow.resolve()),
                                    "bytes": arrow.stat().st_size,
                                    "sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
                                }
                            ],
                        }
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows, audit = shards._load_pinned_zeroshot_source_rows(
        manifest,
        manifest_sha256,
        {"piqa": task_spec},
    )
    return manifest, manifest_sha256, task_spec, {"rows": rows, "audit": audit}


def test_zero_shot_rechecks_prompt_digest_gold_and_choice_range_from_pinned_arrow(tmp_path):
    manifest, manifest_sha256, task_spec, source = _pinned_piqa_source_fixture(tmp_path)
    document = _zero_doc(0, 4, correct=4)
    document["tasks"] = ["piqa"]
    document["config"]["tasks"] = ["piqa"]
    result = document["results"]["3"]["FullKV"]["tasks"].pop("task")
    document["results"]["3"]["FullKV"]["tasks"]["piqa"] = result
    result["split"] = "validation"
    result["protocol_input_hash_schema"] = shards.ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA
    result["protocol_input_sha256"] = source["audit"]["tasks"]["piqa"][
        "full_split_protocol_input_sha256"
    ]
    for example, source_row in zip(result["per_example"], source["rows"]["piqa"]):
        example["gold"] = source_row["gold"]
        example["prediction"] = source_row["gold"]
        example["prediction_norm"] = source_row["gold"]
        example["correct"] = True
        example["correct_norm"] = True
    manifest_record = {
        "path": str(manifest.resolve()),
        "bytes": manifest.stat().st_size,
        "sha256": manifest_sha256,
    }
    document["run_provenance"] = _run_provenance(
        "model",
        input_manifest=manifest_record,
    )
    path = tmp_path / "results" / "zeroshot_piqa.json"
    _write(path, document)
    aggregate = shards.aggregate_zeroshot(
        path.parent,
        expected_models=("model",),
        task_specs={"piqa": task_spec},
        rate_methods={"full": ("FullKV",)},
    )
    assert aggregate["pinned_source_prompt_audit"]["tasks"]["piqa"]["rows"] == 4
    assert "not claimed" in aggregate["pinned_source_prompt_audit"]["scope"]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    example = tampered["results"]["3"]["FullKV"]["tasks"]["piqa"]["per_example"][0]
    example["gold"] = 9
    example["prediction"] = 9
    example["prediction_norm"] = 9
    _write(path, tampered)
    with pytest.raises(shards.AggregateError, match="Pinned source gold mismatch"):
        shards.aggregate_zeroshot(
            path.parent,
            expected_models=("model",),
            task_specs={"piqa": task_spec},
            rate_methods={"full": ("FullKV",)},
        )


def test_zero_shards_sum_counts_after_exact_coverage(tmp_path):
    _write(tmp_path / "zeroshot_a.json", _zero_doc(0, 2, correct=1))
    _write(tmp_path / "zeroshot_b.json", _zero_doc(2, 4, correct=2))
    result = shards.aggregate_zeroshot(
        tmp_path,
        expected_models=("model",),
        task_specs={
            "task": {"size": 4, "split": "test", "revision": "rev", "primary": "acc_norm"}
        },
        rate_methods={"full": ("FullKV",)},
    )
    row = result["models"]["model"]["rates"]["full"]["FullKV"]["tasks"]["task"]
    assert row["n"] == 4
    assert row["correct"] == 3
    assert isinstance(row["correct"], int)
    assert row["acc"] == 0.75
    assert row["correct_norm"] == 3
    assert isinstance(row["correct_norm"], int)
    assert row["acc_norm"] == 0.75
    assert result["official_complete"] is True


def test_zero_shot_rejects_teacher_forcing_protocol_drift(tmp_path):
    document = _zero_doc(0, 4, correct=2)
    task = document["results"]["3"]["FullKV"]["tasks"]["task"]
    task["teacher_forced_temporal_dependency"] = "future_token_coupled_fit"
    _write(tmp_path / "zeroshot_drift.json", document)
    with pytest.raises(shards.AggregateError, match="Wrong/unpinned lm-eval"):
        shards.aggregate_zeroshot(
            tmp_path,
            expected_models=("model",),
            task_specs={
                "task": {
                    "size": 4,
                    "split": "test",
                    "revision": "rev",
                    "primary": "acc_norm",
                }
            },
            rate_methods={"full": ("FullKV",)},
        )


def test_zero_shot_rejects_device_drift_even_with_rehashed_envelope(tmp_path):
    document = _zero_doc(0, 4, correct=2)
    document["config"]["device"] = "cuda:1"
    _write(tmp_path / "zeroshot_device_drift.json", document)

    with pytest.raises(shards.AggregateError, match="Zero-shot device is not cuda:0"):
        shards.aggregate_zeroshot(
            tmp_path,
            expected_models=("model",),
            task_specs={
                "task": {
                    "size": 4,
                    "split": "test",
                    "revision": "rev",
                    "primary": "acc_norm",
                }
            },
            rate_methods={"full": ("FullKV",)},
        )


def _zero_pair_doc(start: int, stop: int, tq_correct: set[int], hq_correct: set[int]) -> dict:
    def method_result(correct_indexes: set[int]) -> dict:
        examples = []
        for index in range(start, stop):
            gold = index % 2
            is_correct = index in correct_indexes
            prediction = gold if is_correct else 1 - gold
            examples.append(
                {
                    "index": index,
                    "gold": gold,
                    "prediction": prediction,
                    "prediction_norm": prediction,
                    "correct": is_correct,
                    "correct_norm": is_correct,
                }
            )
        correct = sum(row["correct"] for row in examples)
        return {
            "tasks": {
                "task": {
                    "n": stop - start,
                    "correct": correct,
                    "correct_norm": correct,
                    "dataset_size": 4,
                    "dataset_revision": "rev",
                    "split": "test",
                    "evaluator": "lm-eval-compatible@0.4.11",
                    "lm_eval_protocol_commit": shards.LM_EVAL_PROTOCOL_COMMIT,
                    "lm_eval_task_metadata_version": shards.LM_EVAL_TASK_METADATA_VERSION,
                    "num_fewshot": 0,
                    "target_delimiter": " ",
                    "apply_chat_template": False,
                    "compression_ratio": 4.0,
                    "compression_aggregation": "exact_cache_bit_counters",
                    "compression_fp16_bits": (stop - start) * 1600,
                    "compression_stored_bits": (stop - start) * 400,
                    "teacher_forced_scoring_protocol": shards.PPL_TEACHER_FORCED_PROTOCOL,
                    "teacher_forced_temporal_dependency": None,
                    "protocol_probe_cache_lifetime": (
                        shards.ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME
                    ),
                    "choice_cache_lifetime": shards.PER_EXAMPLE_CACHE_LIFETIME,
                    "cross_iteration_tensor_retention": (
                        shards.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
                    ),
                    "selection": {
                        "start_valid_example": start,
                        "stop_valid_example_exclusive": stop,
                        "limit": stop - start,
                        "sample_seed": 2027,
                        "order": "validated_dataset_order",
                    },
                    "per_example": examples,
                }
            }
        }

    return {
        "schema_version": 2,
        "run_fingerprint": f"paired-{start}",
        "model": "model",
        "tasks": ["task"],
        "start": start,
        "limit": stop - start,
        "sample_seed": 2027,
        "config": {
            "model_path": _model_path("model"),
            "device": "cuda:0",
            "dtype": "bfloat16",
            "bits": [3],
            "tasks": ["task"],
            "methods": ["TurboQuant-paper", "HQ"],
            "include_fullkv": False,
            "start": start,
            "limit": stop - start,
            "sample_seed": 2027,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
        "run_provenance": _run_provenance("model"),
        "results": {
            "3": {
                "TurboQuant-paper": method_result(tq_correct),
                "HQ": method_result(hq_correct),
            }
        },
    }


def test_zero_shots_report_paired_delta_and_exact_mcnemar(tmp_path):
    _write(tmp_path / "zeroshot_a.json", _zero_pair_doc(0, 2, {0, 1}, {0, 1}))
    _write(tmp_path / "zeroshot_b.json", _zero_pair_doc(2, 4, set(), {2}))
    result = shards.aggregate_zeroshot(
        tmp_path,
        expected_models=("model",),
        task_specs={
            "task": {"size": 4, "split": "test", "revision": "rev", "primary": "acc_norm"}
        },
        rate_methods={"3": ("TurboQuant-paper", "HQ")},
    )
    paired = result["models"]["model"]["rates"]["3"]["paired_proposals_vs_turboquant"]["HQ"]
    assert paired["delta_macro_proposal_minus_turboquant"] == 0.25
    task = paired["per_task"]["task"]
    assert task["discordant"] == {"hq_only_correct": 1, "turboquant_only_correct": 0}
    assert task["mcnemar_exact_two_sided_p"] == 1.0
    assert task["n_paired"] == 4


def test_zero_shots_reject_cross_method_gold_mismatch(tmp_path):
    document = _zero_pair_doc(0, 4, {0}, {0})
    document["results"]["3"]["HQ"]["tasks"]["task"]["per_example"][2]["gold"] = 9
    # Keep the row internally self-consistent so the failure specifically comes
    # from pairing the two methods, not from the per-row audit.
    row = document["results"]["3"]["HQ"]["tasks"]["task"]["per_example"][2]
    row["correct"] = row["prediction"] == row["gold"]
    row["correct_norm"] = row["prediction_norm"] == row["gold"]
    _write(tmp_path / "zeroshot_gold_mismatch.json", document)
    with pytest.raises(shards.AggregateError, match="identity/gold mismatch"):
        shards.aggregate_zeroshot(
            tmp_path,
            expected_models=("model",),
            task_specs={
                "task": {"size": 4, "split": "test", "revision": "rev", "primary": "acc_norm"}
            },
            rate_methods={"3": ("TurboQuant-paper", "HQ")},
        )


@pytest.mark.parametrize("second", [(1, 4), (3, 4)])
def test_zero_shards_reject_overlap_and_gap(tmp_path, second):
    _write(tmp_path / "zeroshot_a.json", _zero_doc(0, 2, correct=1))
    _write(tmp_path / "zeroshot_b.json", _zero_doc(*second, correct=1))
    with pytest.raises(shards.AggregateError, match="overlap|gap"):
        shards.aggregate_zeroshot(
            tmp_path,
            expected_models=("model",),
            task_specs={
                "task": {
                    "size": 4,
                    "split": "test",
                    "revision": "rev",
                    "primary": "acc_norm",
                }
            },
            rate_methods={"full": ("FullKV",)},
        )


def _ppl_doc(
    model: str,
    *,
    turbo_ppl: float,
    hq_ppl: float,
    text_hash: str = shards.WIKITEXT2_TEXT_SHA256,
):
    def row(ppl: float, *, compressed: bool = True) -> dict:
        nll = math.log(ppl)
        fp16_bits = 16000 if compressed else None
        stored_bits = 4000 if compressed else None
        return {
            "ppl": ppl,
            "ppl_nll": nll,
            "ppl_nll_sum": nll * 100,
            "ppl_nll_accumulation": shards.PPL_NLL_ACCUMULATION,
            "ppl_n_tokens": 100,
            "ppl_n_windows": 3,
            "ppl_traversed_tokens": 101,
            "ppl_scored_target_index_set": "[1,101)",
            "ppl_scored_target_ranges": [[1, 34], [34, 67], [67, 101]],
            "ppl_scored_target_coverage": "contiguous_exact_once",
            "ppl_n_skipped_nonfinite": 0,
            "ppl_complete_text_eval": True,
            "ppl_window_cache_lifetime": shards.PPL_WINDOW_CACHE_LIFETIME,
            "ppl_cross_window_tensor_retention": (
                shards.PPL_CROSS_WINDOW_TENSOR_RETENTION
            ),
            "ppl_max_windows": None,
            "ppl_source": "wikitext-2-raw-v1/test",
            "ppl_dataset_revision": shards.WIKITEXT2_REVISION,
            "ppl_text_sha256": text_hash,
            "ppl_compression_ratio": 4.0 if compressed else 1.0,
            "ppl_compression_aggregation": (
                "exact_cache_bit_counters" if compressed else "fullkv_no_cache"
            ),
            "ppl_compression_fp16_bits": fp16_bits,
            "ppl_compression_stored_bits": stored_bits,
            "ppl_teacher_forced_scoring_protocol": shards.PPL_TEACHER_FORCED_PROTOCOL,
            "ppl_teacher_forced_temporal_dependency": None,
        }

    results = {}
    for rate in ("2", "3", "4"):
        rate_i = int(rate)
        turbo = row(turbo_ppl)
        turbo.update(
            {
                "codec_config": {
                    "name": "turboquant",
                    "bits": rate_i,
                    "value_bits": None,
                    "dist": "beta",
                    "rotation_mode": "dense",
                    "n_outlier": 0,
                    "outlier_bits": None,
                    "use_qjl": False,
                    "qjl_roles": "key",
                },
                "distortion_key": 0.1,
                "distortion_value": 0.2,
                "stored_bits": rate_i * 10 + 1,
                "stored_coordinates": 10,
                "stored_bits_per_coord": rate_i + 0.1,
            }
        )
        hq = row(hq_ppl)
        hq.update(
            {
                "codec_config": _fixture_declared_codec("HQ", rate_i),
                "distortion_key": 0.08,
                "distortion_value": 0.18,
                "stored_bits": rate_i * 10 + 1,
                "stored_coordinates": 10,
                "stored_bits_per_coord": rate_i + 0.1,
            }
        )
        adaptive = row(hq_ppl * 0.99)
        adaptive.update(
            {
                "codec_config": _fixture_declared_codec("AdaptiveHQ", rate_i),
                "distortion_key": 0.07,
                "distortion_value": 0.17,
                "stored_bits": rate_i * 10 + 1,
                "stored_coordinates": 10,
                "stored_bits_per_coord": rate_i + 0.1,
            }
        )
        fib = row(turbo_ppl)
        fib.update(
            {
                "codec_config": _fixture_declared_codec("FibQuant-paper", rate_i),
                "distortion_key": 0.09,
                "distortion_value": 0.19,
                "stored_bits": rate_i * 10 + 1,
                "stored_coordinates": 10,
                "stored_bits_per_coord": rate_i + 0.1,
                "speed_metrics_omitted": (
                    "quality-only external baseline; no packed FibQuant serving kernel"
                ),
            }
        )
        results[rate] = {
            "TurboQuant-paper": turbo,
            "FibQuant-paper": fib,
            "HQ": hq,
            "AdaptiveHQ": adaptive,
        }
        if rate == "3":
            block = row(turbo_ppl)
            block.update(
                {
                    "ppl_teacher_forced_scoring_protocol": (
                        shards.CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL
                    ),
                    "ppl_teacher_forced_temporal_dependency": (
                        shards.BLOCKQUANT_TEMPORAL_DEPENDENCY
                    ),
                    "codec_config": _fixture_declared_codec("BlockQuant-paper", 3),
                    "distortion_key": 0.095,
                    "distortion_value": 0.195,
                    "stored_bits": 821,
                    "stored_coordinates": 256,
                    "stored_bits_per_coord": 821 / 256,
                    "speed_metrics_omitted": (
                        "quality-only external baseline; no packed BlockQuant serving kernel"
                    ),
                    "stored_bit_accounting": {
                        "scope": "fixed-width-data-plane-growth-only",
                        "physical_packing_claim": False,
                        "vectors_per_role": 1,
                        "head_dim": 128,
                        "key": {
                            "code_bits": 416,
                            "payload_bits": 517,
                            "fp16_scale_bits": 32,
                            "scale_fields_per_vector": 2,
                            "fixed_channel_identity_bits": 0,
                            "dynamic_channel_identity_bits": 101,
                            "dynamic_channel_identity_fields": 1,
                            "dynamic_channel_identity_coding": ("combinadic_rank_fixed_width_v1"),
                            "total_bits": 549,
                            "bits_per_coordinate": 549 / 128,
                        },
                        "value": {
                            "payload_bits": 256,
                            "fp16_scale_bits": 16,
                            "scale_fields_per_vector": 1,
                            "total_bits": 272,
                            "bits_per_coordinate": 272 / 128,
                        },
                        "total_bits": 821,
                        "excluded_global_state": [
                            "dense Haar rotations",
                            "data-free codebooks",
                            "Python control/container bytes",
                            "allocator and unpacked research tensor bytes",
                        ],
                    },
                }
            )
            results[rate]["BlockQuant-paper"] = block

    return {
        "schema_version": 2,
        "run_fingerprint": f"ppl-{model}",
        "model": model,
        "config": {
            "model_path": _model_path(model),
            "dtype": "bfloat16",
            "bits": [2, 3, 4],
            "methods": [
                "TurboQuant-paper",
                "FibQuant-paper",
                "BlockQuant-paper",
                "HQ",
                "AdaptiveHQ",
            ],
            "run_ppl": True,
            "ppl_windows": None,
            "ppl_max_length": 2048,
            "ppl_stride": 1024,
            "sample_seed": 2027,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
        "run_provenance": _run_provenance(model),
        "fullkv_ppl": row(10.0, compressed=False),
        "results": results,
    }


def _verify_ppl(root: Path, *, expected_models: tuple[str, ...]) -> dict:
    """Run the formal verifier with an explicit synthetic denominator ledger."""
    return shards.verify_ppl(
        root,
        expected_models=expected_models,
        _expected_denominators={
            model: {"n_tokens": 100, "traversed_tokens": 101}
            for model in expected_models
        },
    )


def test_ppl_primary_is_within_model_geometric_mean_ratio(tmp_path):
    _write(tmp_path / "a/results_model-a.json", _ppl_doc("model-a", turbo_ppl=10, hq_ppl=9))
    _write(tmp_path / "b/results_model-b.json", _ppl_doc("model-b", turbo_ppl=20, hq_ppl=22))
    result = _verify_ppl(tmp_path, expected_models=("model-a", "model-b"))
    primary = result["secondary_interval_robustness_vs_turboquant"]["HQ"]["3"]
    assert primary["geometric_mean_ppl_ratio"] == pytest.approx((0.9 * 1.1) ** 0.5)
    assert primary["per_model_ratio"] == {"model-a": 0.9, "model-b": 1.1}
    assert "point estimates" in result["primary_metric_note"]
    fullkv = result["models"]["model-a"]["fullkv"]
    assert fullkv["ppl"] == 10.0
    assert fullkv["ppl_compression_aggregation"] == "fullkv_no_cache"
    assert fullkv["ppl_compression_fp16_bits"] is None
    assert fullkv["ppl_compression_stored_bits"] is None


def test_ppl_rejects_text_hash_drift_across_models(tmp_path):
    _write(tmp_path / "a/results_model-a.json", _ppl_doc("model-a", turbo_ppl=10, hq_ppl=9))
    _write(
        tmp_path / "b/results_model-b.json",
        _ppl_doc("model-b", turbo_ppl=10, hq_ppl=9, text_hash="d" * 64),
    )
    with pytest.raises(shards.AggregateError, match="text hash changed"):
        _verify_ppl(tmp_path, expected_models=("model-a", "model-b"))


def test_ppl_rejects_unpinned_joined_text_and_window_config(tmp_path):
    document = _ppl_doc("model", turbo_ppl=10, hq_ppl=9, text_hash="d" * 64)
    _write(tmp_path / "wrong-text/results_model.json", document)
    with pytest.raises(shards.AggregateError, match="not the pinned 4358-row"):
        _verify_ppl(tmp_path / "wrong-text", expected_models=("model",))

    document = _ppl_doc("model", turbo_ppl=10, hq_ppl=9)
    document["config"]["ppl_stride"] = 512
    _write(tmp_path / "wrong-window/results_model.json", document)
    with pytest.raises(shards.AggregateError, match="formal config contract"):
        _verify_ppl(tmp_path / "wrong-window", expected_models=("model",))


def test_ppl_rejects_rounded_or_tampered_nll_numerator(tmp_path):
    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    document["results"]["3"]["HQ"]["ppl_nll_sum"] += 0.1
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="numerator/denominator"):
        _verify_ppl(tmp_path, expected_models=("model",))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.__setitem__("ppl_compression_ratio", 3.5),
            "not total fp16_bits/stored_bits",
        ),
        (
            lambda row: row.__setitem__(
                "ppl_compression_aggregation", "arithmetic_mean_window_ratio"
            ),
            "not exact total-bit accounting",
        ),
        (
            lambda row: row.__setitem__(
                "ppl_teacher_forced_scoring_protocol",
                "causal_prefix_prefill_then_token_step_targets",
            ),
            "Wrong teacher-forced scoring protocol",
        ),
    ],
)
def test_ppl_rejects_stale_window_ratio_or_scoring_protocol(tmp_path, mutation, message):
    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    mutation(document["results"]["3"]["HQ"])
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match=message):
        _verify_ppl(tmp_path, expected_models=("model",))


def test_ppl_rejects_non_iso_rate_primary_comparison(tmp_path):
    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    row = document["results"]["3"]["HQ"]
    row["stored_bits"] += 1
    row["stored_bits_per_coord"] = row["stored_bits"] / row["stored_coordinates"]
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="exact matched bit counters"):
        _verify_ppl(tmp_path, expected_models=("model",))


def test_ppl_rejects_stored_rate_not_derived_from_exact_integers(tmp_path):
    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    document["results"]["3"]["HQ"]["stored_bits_per_coord"] += 0.0001
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="numerator/denominator"):
        _verify_ppl(tmp_path, expected_models=("model",))


def test_ppl_rejects_role_resolved_rotation_or_cache_policy_drift(tmp_path):
    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    _add_formal_recipe_fixture(document)
    hq = document["results"]["3"]["HQ"]
    hq["resolved_codec_config"]["roles"]["key"]["rotation_mode"] = "dense"
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="Role-resolved codec recipe changed"):
        _verify_ppl(tmp_path, expected_models=("model",))

    document = _ppl_doc("model", turbo_ppl=10.0, hq_ppl=9.0)
    _add_formal_recipe_fixture(document)
    document["results"]["3"]["TurboQuant-paper"]["cache_policy"]["recent_tokens"] = 1
    _write(tmp_path / "other" / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="formal cache policy"):
        _verify_ppl(tmp_path / "other", expected_models=("model",))


def _mechanism_doc(
    model: str,
    *,
    scalar_ppl: float,
    block_ppl: float,
    text_hash: str = shards.WIKITEXT2_TEXT_SHA256,
) -> dict:
    def row(ppl: float, rate: int, block_size: int | None) -> dict:
        nll = math.log(ppl)
        return {
            "ppl": ppl,
            "ppl_nll": nll,
            "ppl_nll_sum": nll * 100,
            "ppl_nll_accumulation": shards.PPL_NLL_ACCUMULATION,
            "ppl_n_tokens": 100,
            "ppl_n_windows": 3,
            "ppl_traversed_tokens": 101,
            "ppl_scored_target_index_set": "[1,101)",
            "ppl_scored_target_ranges": [[1, 34], [34, 67], [67, 101]],
            "ppl_scored_target_coverage": "contiguous_exact_once",
            "ppl_n_skipped_nonfinite": 0,
            "ppl_complete_text_eval": True,
            "ppl_window_cache_lifetime": shards.PPL_WINDOW_CACHE_LIFETIME,
            "ppl_cross_window_tensor_retention": (
                shards.PPL_CROSS_WINDOW_TENSOR_RETENTION
            ),
            "ppl_max_windows": None,
            "ppl_source": "wikitext-2-raw-v1/test",
            "ppl_dataset_revision": shards.WIKITEXT2_REVISION,
            "ppl_text_sha256": text_hash,
            "ppl_compression_ratio": 4.0,
            "ppl_compression_aggregation": "exact_cache_bit_counters",
            "ppl_compression_fp16_bits": 16000,
            "ppl_compression_stored_bits": 4000,
            "ppl_teacher_forced_scoring_protocol": shards.PPL_TEACHER_FORCED_PROTOCOL,
            "ppl_teacher_forced_temporal_dependency": None,
            "codec_config": {
                "name": "hayatequant",
                "bits": rate,
                "value_bits": rate,
                "block_size": block_size,
                "dist": "beta",
                "rotation_mode": "dense",
                "encoder": "exact",
                "ls_scale": False,
                "key_scale_mode": None,
                "value_scale_mode": None,
                "max_codebook_bits": 9,
                "codebook_init": "auto",
                "n_outlier": 0,
                "key_n_outlier": None,
                "value_n_outlier": None,
                "use_qjl": False,
                "calibration": None,
            },
            "distortion_key": 0.09 if block_size == 1 else 0.08,
            "distortion_value": 0.19 if block_size == 1 else 0.18,
            "stored_bits": rate * 10 + 1,
            "stored_coordinates": 10,
            "stored_bits_per_coord": rate + 0.1,
        }

    results = {
        str(rate): {
            "HayateQuant-scalar-matched": row(scalar_ppl, rate, 1),
            "HayateQuant-blockvq-matched": row(block_ppl, rate, None),
        }
        for rate in (2, 3, 4)
    }
    return {
        "schema_version": 2,
        "run_fingerprint": f"mechanism-{model}",
        "model": model,
        "config": {
            "model_path": _model_path(model),
            "methods": [
                "HayateQuant-scalar-matched",
                "HayateQuant-blockvq-matched",
            ],
            "bits": [2, 3, 4],
            "ppl_windows": None,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
        "run_provenance": _run_provenance(model),
        "fullkv_ppl": {
            "ppl": 10.0,
            "ppl_nll": math.log(10.0),
            "ppl_nll_sum": math.log(10.0) * 100,
            "ppl_nll_accumulation": shards.PPL_NLL_ACCUMULATION,
            "ppl_n_tokens": 100,
            "ppl_n_windows": 3,
            "ppl_traversed_tokens": 101,
            "ppl_scored_target_index_set": "[1,101)",
            "ppl_scored_target_ranges": [[1, 34], [34, 67], [67, 101]],
            "ppl_scored_target_coverage": "contiguous_exact_once",
            "ppl_n_skipped_nonfinite": 0,
            "ppl_complete_text_eval": True,
            "ppl_window_cache_lifetime": shards.PPL_WINDOW_CACHE_LIFETIME,
            "ppl_cross_window_tensor_retention": (
                shards.PPL_CROSS_WINDOW_TENSOR_RETENTION
            ),
            "ppl_max_windows": None,
            "ppl_source": "wikitext-2-raw-v1/test",
            "ppl_dataset_revision": shards.WIKITEXT2_REVISION,
            "ppl_text_sha256": text_hash,
            "ppl_compression_ratio": 1.0,
            "ppl_compression_aggregation": "fullkv_no_cache",
            "ppl_compression_fp16_bits": None,
            "ppl_compression_stored_bits": None,
            "ppl_teacher_forced_scoring_protocol": shards.PPL_TEACHER_FORCED_PROTOCOL,
            "ppl_teacher_forced_temporal_dependency": None,
        },
        "results": results,
    }


def test_mechanism_control_joins_primary_curve_and_isolates_block_size(tmp_path):
    ppl_root = tmp_path / "ppl"
    mechanism_root = tmp_path / "mechanism"
    _write(ppl_root / "results_model.json", _ppl_doc("model", turbo_ppl=10, hq_ppl=9))
    reference = _verify_ppl(ppl_root, expected_models=("model",))
    _write(
        mechanism_root / "results_model.json",
        _mechanism_doc("model", scalar_ppl=10.0, block_ppl=9.5),
    )

    result = shards.verify_mechanism(
        mechanism_root,
        reference_ppl=reference,
        expected_models=("model",),
    )
    rate = result["models"]["model"]["rates"]["3"]
    assert set(rate["methods"]) == {
        "TurboQuant-paper",
        "HayateQuant-scalar-matched",
        "HayateQuant-blockvq-matched",
        "HQ",
        "AdaptiveHQ",
    }
    assert rate["causal_block_vq_ratio_block_over_scalar"] == 0.95
    assert rate["causal_block_vq_relative_ppl_change_percent"] == -5.0


def test_mechanism_control_rejects_non_block_codec_difference(tmp_path):
    ppl_root = tmp_path / "ppl"
    mechanism_root = tmp_path / "mechanism"
    _write(ppl_root / "results_model.json", _ppl_doc("model", turbo_ppl=10, hq_ppl=9))
    reference = _verify_ppl(ppl_root, expected_models=("model",))
    document = _mechanism_doc("model", scalar_ppl=10.0, block_ppl=9.5)
    document["results"]["3"]["HayateQuant-blockvq-matched"]["codec_config"][
        "max_codebook_bits"
    ] = 12
    _write(mechanism_root / "results_model.json", document)

    with pytest.raises(
        shards.AggregateError,
        match="Declared codec recipe changed|outside block_size",
    ):
        shards.verify_mechanism(
            mechanism_root,
            reference_ppl=reference,
            expected_models=("model",),
        )


def test_mechanism_control_rejects_rate_mismatch_to_turboquant(tmp_path):
    ppl_root = tmp_path / "ppl"
    mechanism_root = tmp_path / "mechanism"
    _write(ppl_root / "results_model.json", _ppl_doc("model", turbo_ppl=10, hq_ppl=9))
    reference = _verify_ppl(ppl_root, expected_models=("model",))
    document = _mechanism_doc("model", scalar_ppl=10.0, block_ppl=9.5)
    for method in (
        "HayateQuant-scalar-matched",
        "HayateQuant-blockvq-matched",
    ):
        row = document["results"]["3"][method]
        row["stored_bits"] += 1
        row["stored_bits_per_coord"] = row["stored_bits"] / row["stored_coordinates"]
    _write(mechanism_root / "results_model.json", document)

    with pytest.raises(shards.AggregateError, match="TurboQuant rate"):
        shards.verify_mechanism(
            mechanism_root,
            reference_ppl=reference,
            expected_models=("model",),
        )


def _plain_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _add_fullkv_compression(container: dict, records: list[dict]) -> dict:
    for record in records:
        record.update(
            {
                "compression_aggregation": "fullkv_no_cache",
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "adaptive_hq_runtime": None,
            }
        )
    stats = {
        "compression_fp16_bits": None,
        "compression_stored_bits": None,
        "compression_aggregation": "fullkv_no_cache",
        "compression_observations": len(records),
        "adaptive_hq_runtime": None,
    }
    container["compression_sufficient_statistics"] = stats
    return stats


def _adaptive_shard_compression_statistics() -> dict:
    return {
        "compression_fp16_bits": 1600,
        "compression_stored_bits": 400,
        "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
        "compression_observations": 1,
        "adaptive_hq_runtime": {
            "schema_version": 1,
            "policy_version": shards.ADAPTIVE_HQ_POLICY_VERSION,
            "encoded_updates": 1,
            "encoded_tokens": 2,
            "selector_bits_total": 1,
            "selector_incremental_stored_bits_total": 0,
            "selector_carrier_version": shards.ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "split_histogram": {
                "6:6": {
                    "pair": [6, 6],
                    "updates": 1,
                    "tokens": 2,
                    "vectors": 4,
                }
            },
        },
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
        (("split_histogram", "6:6", "vectors"), 1),
        (("split_histogram", "6:6", "pair"), [4, 4]),
    ],
)
def test_adaptive_shard_compression_summary_is_fail_closed(path, bad_value):
    row = _adaptive_shard_compression_statistics()
    runtime = row["adaptive_hq_runtime"]
    target = runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    with pytest.raises(shards.AggregateError, match="AdaptiveHQ"):
        shards._sum_long_compression_statistics(  # pylint: disable=protected-access
            [row], method="AdaptiveHQ", label="adversarial shard"
        )


def test_adaptive_shard_compression_summary_replays_exactly():
    row = _adaptive_shard_compression_statistics()
    combined = shards._sum_long_compression_statistics(  # pylint: disable=protected-access
        [row, copy.deepcopy(row)], method="AdaptiveHQ", label="valid shards"
    )
    runtime = combined["adaptive_hq_runtime"]
    assert runtime["encoded_updates"] == 2
    assert runtime["encoded_tokens"] == 4
    assert runtime["selector_bits_total"] == 2
    assert runtime["split_histogram"]["6:6"]["vectors"] == 8


def _longcode_files(tmp_path: Path, *, size: int = 4) -> dict:
    input_tokens = [2 + index for index in range(size)]
    rows = [
        {
            "prompt": f"released prompt {index}",
            "correct_letter": "A",
            "repo": "owner/repo",
            "question": f"released question {index}",
        }
        for index in range(size)
    ]
    dataset_path = tmp_path / "inputs/LongCodeQA/LQA/32K.json"
    _plain_json(dataset_path, rows)
    dataset_record = {**_file_record(dataset_path), "rows": size}
    context_rows = [
        {
            "index": index,
            "input_tokens": tokens,
            "margin": 32 - tokens - 16,
        }
        for index, tokens in enumerate(input_tokens)
    ]
    input_summary = {
        "min": min(input_tokens),
        "median": float((input_tokens[1] + input_tokens[2]) / 2),
        "max": max(input_tokens),
    }
    protocol = {
        "scope": shards.LONGCODE_INPUT_PROTOCOL_SCOPE,
        "dataset_revision": shards.LONGCODE_REVISION,
        "prompt": "released prompt verbatim as raw causal text",
        "prompt_release_clarification": shards.LONGCODE_DISCUSSION_URL,
        "chat_template": "none",
        "truncation": "none",
        "answer_parser_reference_commit": shards.LONGCODE_PROTOCOL_COMMIT,
        "answer_parser": shards.LONGCODE_ANSWER_PARSER,
        "reference_evaluator_executable_verbatim": False,
        "reference_evaluator_known_issue": (
            "pinned evaluator stores response but reads prediction text"
        ),
        "generation": {
            "protocol": "hayatequant deterministic greedy paired comparison",
            "do_sample": False,
            "max_new_tokens": 16,
            "seed": None,
            "eos_token_ids_source": "model.generation_config.eos_token_id",
            "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
            "official_reference_max_output_length": None,
            "official_reference_uniform_decoding_config": False,
            "official_reference_decoding_exact": False,
        },
    }
    context_bracket = {
        "rows": size,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_record["sha256"],
        "input_tokens": input_summary,
        "minimum_margin_tokens": min(row["margin"] for row in context_rows),
        "overflow_rows": [],
        "per_row": context_rows,
    }
    context_path = tmp_path / "inputs/longcodeqa_context_model.json"
    context_document = {
        "schema_version": 2,
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "benchmark": "LongCodeBench/LongCodeQA context-fit validation",
        "model": {
            "path": _model_path("model"),
            "context_limit": 32,
            "generation_eos_token_ids": [1, 2],
            "generation_eos_token_ids_source": ("model.generation_config.eos_token_id"),
        },
        "protocol": protocol,
        "brackets": {"32K": context_bracket},
        "all_rows_fit": True,
    }
    _plain_json(context_path, context_document)
    context_record = _file_record(context_path)
    context_summary = {
        "manifest": context_record,
        "created_at_utc": context_document["created_at_utc"],
        "model": context_document["model"],
        "protocol": protocol,
        "all_rows_fit": True,
        "brackets": {
            "32K": {key: value for key, value in context_bracket.items() if key != "per_row"}
        },
    }
    input_path = tmp_path / "inputs/paper-inputs.json"
    input_document = {
        "schema_version": 2,
        "kind": "hayatequant_paper_input_validation",
        "status": "ok",
        "longcodeqa": {
            "dataset": "LongCodeBench/LongCodeQA",
            "release_revision": shards.LONGCODE_REVISION,
            "release_clarification": shards.LONGCODE_DISCUSSION_URL,
            "parser_reference_commit": shards.LONGCODE_PROTOCOL_COMMIT,
            "protocol_scope": shards.LONGCODE_INPUT_PROTOCOL_SCOPE,
            "official_reference_decoding_exact": False,
            "brackets": {"32K": dataset_record},
            "total_rows": size,
            "context_fit_manifests": {"model": context_summary},
        },
    }
    _plain_json(input_path, input_document)
    input_record = _file_record(input_path)
    return {
        "rows": rows,
        "input_tokens": input_tokens,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_record["sha256"],
        "context_path": context_path,
        "run_provenance": _run_provenance(
            "model",
            input_manifest=input_record,
            dataset_files=[input_record, dataset_record, context_record],
        ),
    }


def _long_doc(
    start: int,
    stop: int,
    correct_indexes: set[int],
    files: dict,
) -> dict:
    records = [
        {
            "index": index,
            "repo": files["rows"][index]["repo"],
            "question": files["rows"][index]["question"],
            "input_tokens": files["input_tokens"][index],
            "generated_text": (
                "Final Answer: A" if index in correct_indexes else "Final Answer: B"
            ),
            "generated_tokens": 3,
            "prediction": "A" if index in correct_indexes else "B",
            "gold": "A",
            "correct": index in correct_indexes,
        }
        for index in range(start, stop)
    ]
    bracket = {
        "n": stop - start,
        "correct": sum(row["correct"] for row in records),
        "expected_bracket_n": 4,
        "start": start,
        "limit": stop - start,
        "max_context_tokens": 32,
        "max_new_tokens": 16,
        "protocol_commit": shards.LONGCODE_PROTOCOL_COMMIT,
        "reference_evaluator_executable_verbatim": False,
        "reference_evaluator_known_issue": (
            "pinned evaluator stores response but reads prediction text"
        ),
        "protocol_scope": shards.LONGCODE_PROTOCOL_SCOPE,
        "example_cache_lifetime": shards.PER_EXAMPLE_CACHE_LIFETIME,
        "cross_iteration_tensor_retention": (
            shards.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ),
        "prompt_protocol": "released_prompt_verbatim_raw_causal_text",
        "chat_template": "none",
        "truncation": "none",
        "generation": {
            "protocol": "hayatequant_deterministic_greedy_paired_comparison",
            "do_sample": False,
            "max_new_tokens": 16,
            "seed": None,
            "eos_token_ids": [1, 2],
            "eos_token_ids_source": "model.generation_config.eos_token_id",
            "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
            "official_reference_max_output_length": None,
            "official_reference_uniform_decoding_config": False,
            "official_reference_decoding_exact": False,
        },
        "answer_parser": shards.LONGCODE_ANSWER_PARSER,
        "per_example_source_fields": [
            "index",
            "repo",
            "question",
            "correct_letter_as_gold",
        ],
        "per_example_prompt_retained": False,
        "dataset": {
            "repo_id": "Steefano/LCB",
            "revision": shards.LONGCODE_REVISION,
            "path": str(files["dataset_path"].resolve()),
            "sha256": files["dataset_sha256"],
            "final_filtered_release": True,
            "release_clarification": shards.LONGCODE_DISCUSSION_URL,
        },
        "input_tokens": {"min": 1, "median": 2, "max": 3},
        "per_example": records,
    }
    compression = _add_fullkv_compression(bracket, records)
    longcodeqa = {
        "official_primary_reporting": "per_bracket_exact_match_accuracy",
        "derived_cross_bracket_summary": True,
        "protocol_scope": shards.LONGCODE_PROTOCOL_SCOPE,
        "official_reference_decoding_exact": False,
        "brackets": {"32K": bracket},
        "compression_sufficient_statistics": compression,
    }
    return {
        "schema_version": 2,
        "run_fingerprint": f"long-{start}",
        "model": "model",
        "config": {
            "model_path": _model_path("model"),
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
            "longcodeqa_max_new": 16,
        },
        "run_provenance": files["run_provenance"],
        "results": {"FullKV": {"longcodeqa": longcodeqa}},
    }


def test_longcodeqa_shards_require_exact_indexes_and_recompute_micro_score(tmp_path):
    files = _longcode_files(tmp_path)
    _write(tmp_path / "a/results_model.json", _long_doc(0, 2, {0}, files))
    _write(tmp_path / "b/results_model.json", _long_doc(2, 4, {2, 3}, files))
    result = shards.aggregate_longcodeqa(
        tmp_path,
        plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
        methods=("FullKV",),
        expected_sha256={"32K": files["dataset_sha256"]},
        model_eos_token_ids={"model": [1, 2]},
    )
    row = result["models"]["model"]["methods"]["FullKV"]
    assert row["n"] == 4
    assert row["correct"] == 3
    assert row["accuracy"] == 0.75
    assert row["brackets"]["32K"]["released_data_complete_bracket"] is True
    assert row["brackets"]["32K"]["audit_records_validated"] == 4
    assert result["source_row_audit"]["released_prompt_retained_in_result"] is False
    assert "not claimed" in result["source_row_audit"]["released_prompt_comparison"]


def test_longcodeqa_rejects_wrong_per_example_interval(tmp_path):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    document["results"]["FullKV"]["longcodeqa"]["brackets"]["32K"]["per_example"][2]["index"] = 3
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="indexes"):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("generation", "eos_token_ids"), [1], "prompt/parser/generation"),
        (
            ("generation", "official_reference_decoding_exact"),
            True,
            "prompt/parser/generation",
        ),
        (("dataset", "sha256"), "b" * 64, "dataset revision"),
    ],
)
def test_longcodeqa_rejects_protocol_or_release_provenance_drift(tmp_path, path, value, error):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    bracket = document["results"]["FullKV"]["longcodeqa"]["brackets"]["32K"]
    bracket[path[0]][path[1]] = value
    _write(tmp_path / "results_model.json", document)

    with pytest.raises(shards.AggregateError, match=error):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


def test_longcodeqa_rejects_gold_and_prediction_tampered_together(tmp_path):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    row = document["results"]["FullKV"]["longcodeqa"]["brackets"]["32K"]["per_example"][0]
    row.update(
        {
            "generated_text": "Final Answer: B",
            "prediction": "B",
            "gold": "B",
            "correct": True,
        }
    )
    _write(tmp_path / "results_model.json", document)

    with pytest.raises(shards.AggregateError, match="pinned source row"):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("repo", "tampered/repo"), ("question", "tampered question")],
)
def test_longcodeqa_rejects_tampered_retained_source_fields(tmp_path, field, value):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    document["results"]["FullKV"]["longcodeqa"]["brackets"]["32K"]["per_example"][0][field] = value
    _write(tmp_path / "results_model.json", document)

    with pytest.raises(shards.AggregateError, match="pinned source row"):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


def test_longcodeqa_rejects_token_length_not_in_pinned_context_manifest(tmp_path):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    document["results"]["FullKV"]["longcodeqa"]["brackets"]["32K"]["per_example"][0][
        "input_tokens"
    ] += 1
    _write(tmp_path / "results_model.json", document)

    with pytest.raises(shards.AggregateError, match="pinned context-fit row"):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


def test_longcodeqa_rejects_context_manifest_changed_after_preflight(tmp_path):
    files = _longcode_files(tmp_path)
    document = _long_doc(0, 4, {0}, files)
    _write(tmp_path / "results_model.json", document)
    context = json.loads(files["context_path"].read_text(encoding="utf-8"))
    context["brackets"]["32K"]["per_row"][0]["input_tokens"] += 1
    context["brackets"]["32K"]["per_row"][0]["margin"] -= 1
    _plain_json(files["context_path"], context)

    with pytest.raises(shards.AggregateError, match="Pinned file (byte size|SHA-256) changed"):
        shards.aggregate_longcodeqa(
            tmp_path,
            plans={"model": {"32K": {"size": 4, "max_context_tokens": 32}}},
            methods=("FullKV",),
            expected_sha256={"32K": files["dataset_sha256"]},
            model_eos_token_ids={"model": [1, 2]},
        )


def _longbench_doc(task: str, scores: list[float], *, max_new: int) -> dict:
    def texts(score: float, index: int) -> tuple[str, str]:
        if score == pytest.approx(1.0):
            return f"exact answer {index}", f"exact answer {index}"
        if score == pytest.approx(0.5):
            return "alpha beta", "alpha gamma"
        if score == pytest.approx(2.0 / 3.0):
            return "alpha beta gamma", "alpha beta delta"
        if score == pytest.approx(0.0):
            return "alpha", "omega"
        raise AssertionError(f"unsupported synthetic LongBench score {score}")

    score_pct = [100.0 * score for score in scores]
    records = [
        {
            "row_index": index,
            "prediction": texts(score, index)[0],
            "generated_text": texts(score, index)[0],
            "generated_tokens": min(max_new, 4),
            "answers": [texts(score, index)[1]],
            "all_classes": None,
            "original_length": 1000,
            "length_bucket": "0-4k",
            "input_tokens": 900,
            "raw_prompt_tokens": 1000,
            "raw_prompt_token_budget": 32768,
            "middle_truncated": False,
            "chat_wrapper_used": task not in shards.LONGBENCH_NO_CHAT_TASKS,
            "model_input_tokens": 900,
            "max_new_tokens": max_new,
            "stop_token_ids": [7, 8, 42] if task == "samsum" else [7, 8],
            "score": score,
            "score_pct": score_pct[index],
        }
        for index, score in enumerate(scores)
    ]
    mean = sum(score_pct) / len(score_pct)
    value = {
        "n": len(scores),
        "score": round(mean, 2),
        "per_task": {task: round(mean, 2)},
        "per_task_n": {task: len(scores)},
        "per_task_score_sum": {task: sum(score_pct)},
        "by_length": {"0-4k": round(mean, 2), "4-8k": 0.0, "8k+": 0.0},
        "by_length_n": {"0-4k": len(scores), "4-8k": 0, "8k+": 0},
        "limit_per_task": None,
        "tasks": [task],
        "dataset_revision": "longbench-revision",
        "protocol_commit": "protocol-commit",
        "protocol_scope": shards.LONGBENCH_PROTOCOL_SCOPE,
        "example_cache_lifetime": shards.PER_EXAMPLE_CACHE_LIFETIME,
        "cross_iteration_tensor_retention": (
            shards.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ),
        "reference_model_wrapper_exact": False,
        "data_zip_sha256": "b" * 64,
        "max_length": 32768,
        "task_dataset_sizes": {task: len(scores)},
        "score_protocol": "derived_unweighted_macro_over_task_scores",
        "by_length_protocol": ("derived_unweighted_macro_over_official_per_task_bucket_means"),
        "prompt_protocol": {
            "raw_prompt": "official dataset2prompt formatting",
            "truncation": "official middle truncation before chat wrapping",
            "chat_wrapper": shards.LONGBENCH_CHAT_WRAPPER_SCOPE,
            "no_chat_tasks": sorted(shards.LONGBENCH_NO_CHAT_TASKS),
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "sampling_seed": None,
            "official_reference_global_seed": 42,
            "max_new_tokens": "official dataset2maxlen per task",
            "eos_token_ids": [7, 8],
            "eos_token_id_source": "model.generation_config.eos_token_id",
            "samsum_newline_token_id": 42,
            "stop_conditions": [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
                "samsum_newline_token_id_for_samsum_only",
            ],
        },
        "truncated_examples": 0,
        "truncated_per_task": {task: 0},
        "per_example": {task: records},
        "released_data_complete": False,
    }
    compression = _add_fullkv_compression(value, records)
    value["per_task_compression_sufficient_statistics"] = {task: compression}
    return {
        "schema_version": 2,
        "run_fingerprint": f"lb-{task}",
        "model": "model",
        "config": {
            "model_path": _model_path("model"),
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
            "longbench_max_length": 32768,
        },
        "run_provenance": _run_provenance("model"),
        "results": {"FullKV": {"longbench": value}},
    }


def _longbench_archive(tmp_path: Path, documents: list[dict]) -> tuple[Path, str]:
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for document in documents:
            value = document["results"]["FullKV"]["longbench"]
            task = value["tasks"][0]
            lines = []
            for record in value["per_example"][task]:
                lines.append(
                    json.dumps(
                        {
                            "context": f"context {task} {record['row_index']}",
                            "input": f"input {task} {record['row_index']}",
                            "answers": record["answers"],
                            "all_classes": record["all_classes"],
                            "length": record["original_length"],
                        }
                    )
                )
            archive.writestr(f"data/{task}_e.jsonl", "\n".join(lines) + "\n")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    for document in documents:
        document["results"]["FullKV"]["longbench"]["data_zip_sha256"] = digest
    return archive_path, digest


def test_longbench_recomputes_unrounded_macro_from_regradable_examples(tmp_path):
    qasper = _longbench_doc("qasper", [0.5, 2.0 / 3.0], max_new=128)
    multifield = _longbench_doc("multifieldqa_en", [1.0], max_new=64)
    archive, digest = _longbench_archive(tmp_path, [qasper, multifield])
    _write(tmp_path / "a/results_model.json", qasper)
    _write(tmp_path / "b/results_model.json", multifield)
    result = shards.verify_longbench(
        tmp_path,
        expected_models=("model",),
        methods=("FullKV",),
        task_counts={"qasper": 2, "multifieldqa_en": 1},
        task_max_new={"qasper": 128, "multifieldqa_en": 64},
        dataset_revision="longbench-revision",
        protocol_commit="protocol-commit",
        archive_sha256=digest,
        prompt_budgets={"model": 32768},
        model_context_limits={"model": 65536},
        _archive_path_override=archive,
    )
    row = result["models"]["model"]["FullKV"]
    expected = (((0.5 + 2.0 / 3.0) * 100 / 2) + 100.0) / 2
    assert row["score_unrounded"] == pytest.approx(expected)
    assert row["n"] == 3
    assert all(value["audit_records_validated"] for value in row["per_task"].values())


def test_longbench_rejects_result_without_lossless_prediction_audit(tmp_path):
    document = _longbench_doc("qasper", [0.5], max_new=128)
    archive, digest = _longbench_archive(tmp_path, [document])
    del document["results"]["FullKV"]["longbench"]["per_example"]["qasper"][0]["prediction"]
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="prediction"):
        shards.verify_longbench(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            task_counts={"qasper": 1},
            task_max_new={"qasper": 128},
            dataset_revision="longbench-revision",
            protocol_commit="protocol-commit",
            archive_sha256=digest,
            prompt_budgets={"model": 32768},
            model_context_limits={"model": 65536},
            _archive_path_override=archive,
        )


def test_longbench_rejects_tampered_scores_even_when_all_stored_totals_agree(tmp_path):
    document = _longbench_doc("qasper", [0.5], max_new=128)
    archive, digest = _longbench_archive(tmp_path, [document])
    value = document["results"]["FullKV"]["longbench"]
    record = value["per_example"]["qasper"][0]
    record["score"] = 1.0
    record["score_pct"] = 100.0
    value["per_task_score_sum"]["qasper"] = 100.0
    value["per_task"]["qasper"] = 100.0
    value["score"] = 100.0
    value["by_length"]["0-4k"] = 100.0
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="official metric replay"):
        shards.verify_longbench(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            task_counts={"qasper": 1},
            task_max_new={"qasper": 128},
            dataset_revision="longbench-revision",
            protocol_commit="protocol-commit",
            archive_sha256=digest,
            prompt_budgets={"model": 32768},
            model_context_limits={"model": 65536},
            _archive_path_override=archive,
        )


def test_longbench_rejects_tampered_pinned_row_metadata(tmp_path):
    document = _longbench_doc("qasper", [0.5], max_new=128)
    archive, digest = _longbench_archive(tmp_path, [document])
    document["results"]["FullKV"]["longbench"]["per_example"]["qasper"][0]["answers"] = [
        "forged gold"
    ]
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="pinned row"):
        shards.verify_longbench(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            task_counts={"qasper": 1},
            task_max_new={"qasper": 128},
            dataset_revision="longbench-revision",
            protocol_commit="protocol-commit",
            archive_sha256=digest,
            prompt_budgets={"model": 32768},
            model_context_limits={"model": 65536},
            _archive_path_override=archive,
        )


def _math_doc(start: int, stop: int, correct_indexes: set[int]) -> dict:
    questions = []
    for index in range(start, stop):
        correct = index in correct_indexes
        predicted_answer = index if correct else index + 1000
        questions.append(
            {
                "index": index,
                "question": (
                    reasoning_eval._MATH500_FEWSHOT_PREFIX
                    + "\n\n"
                    + shards.MATH500_QUERY_TEMPLATE.format(problem=f"problem {index}")
                ),
                "gold": str(index),
                "input_tokens": 8,
                "pass@1": float(correct),
                "observed_pass@1": float(correct),
                "samples": [
                    {
                        "seed": 2027 + index * 100_003,
                        "generated_text": (
                            f"reasoning trace {index}\\n\\boxed{{{predicted_answer}}}"
                        ),
                        "raw_generated_text": (
                            f"reasoning trace {index}\\n\\boxed{{{predicted_answer}}}"
                        ),
                        "matched_stop_sequence": None,
                        "grading_input": "generated_text",
                        "legacy_regex_extracted_answer_non_grading": str(predicted_answer),
                        "generated_tokens": 12,
                        "correct": correct,
                    }
                ],
                "k": 1,
            }
        )
    n = stop - start
    correct = sum(index in correct_indexes for index in range(start, stop))
    value = {
        "task": "math500",
        "n": n,
        "k": 1,
        "pass@1": round(correct / n, 4),
        "pass@1_sum": float(correct),
        "correct": correct,
        "selection": {"start": start, "stop_exclusive": stop, "limit": n},
        "per_question": questions,
        "dataset_size": 4,
        "complete_dataset_eval": False,
        "compression_ratio": 1.0,
        "compression_ratio_aggregation": "fullkv_no_cache",
        "example_cache_lifetime": shards.PER_EXAMPLE_CACHE_LIFETIME,
        "cross_iteration_tensor_retention": (
            shards.PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ),
        "eviction_applicable": False,
        "eviction_fired": None,
        "grader": "math_verify",
        "grader_version": "0.9.0",
        "grader_config": {
            "dataset_revision": "math-revision",
            "gold_extraction": "latex(boxed_match_priority=0; inline_math_wrapped)",
            "gold_serialization": shards.MATH500_GOLD_SERIALIZATION,
            "official_reference_gold_serialization_exact": False,
            "gold_serialization_adaptation_reason": shards.MATH500_GOLD_ADAPTATION_REASON,
            "protocol_scope": shards.MATH500_PROTOCOL_SCOPE,
            "prediction_extraction": "full_generation:latex+expr",
            "strict": True,
            "parse": {
                "fallback_mode": "first_match",
                "extraction_mode": "any_match",
                "parsing_timeout_seconds": 5,
                "raise_on_error": False,
            },
            "verify": {
                "float_rounding": 6,
                "numeric_precision": 15,
                "strict": True,
                "allow_set_relation_comp": False,
                "timeout_seconds": 5,
                "raise_on_error": False,
            },
        },
        "protocol": "hf_math500_test_adapted_composite_four_shot",
        "prompt_config": {
            "use_chat_template": False,
            "system": None,
            "query_protocol": "Math-Verify math_500 task query",
            "query_template": shards.MATH500_QUERY_TEMPLATE,
            "few_shot": 4,
            "few_shot_order_one_based": list(shards.MATH500_FEWSHOT_ORDER),
            "few_shot_content_sha256": shards.MATH500_FEWSHOT_CONTENT_SHA256,
            "few_shot_rendering": shards.MATH500_FEWSHOT_RENDERING,
            "protocol_id": shards.MATH500_PROTOCOL_ID,
            "protocol_origin": "adapted_composite",
            "math_verify_task_source_commit": shards.MATH_VERIFY_TASK_SOURCE_COMMIT,
            "lighteval_prompt_manager_reference_version": (
                shards.MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
            ),
            "lighteval_prompt_manager_source_commit": (
                shards.MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT
            ),
            "upstream_lighteval_builtin_task_exact": False,
            "upstream_math_verify_custom_task_exact": False,
            "upstream_prompt_generation_components_exact": True,
            "upstream_component_pair_executable": False,
            "lighteval_balanced_sampler_seed": 0,
        },
        "generation_config": {
            "max_new_tokens": 1024,
            "reference_max_new_tokens": 1024,
            "protocol": (
                "adapted_composite_math_verify_ba3d3aa_lighteval_"
                "prompt_manager_7c1cd62_four_shot"
            ),
            "do_sample": False,
            "requested_do_sample": False,
            "temperature": None,
            "top_p": None,
            "stop_sequences": list(shards.MATH500_STOP_SEQUENCES),
            "sample_seed_base": 2027,
            "per_sample_seed_formula": ("base + global_question_index * 100003 + sample_index"),
        },
    }
    _add_fullkv_compression(
        value,
        [question["samples"][0] for question in questions],
    )
    return {
        "schema_version": 2,
        "run_fingerprint": f"math-{start}",
        "model": "model",
        "config": {
            "model_path": _model_path("model"),
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
            "math_start": start,
            "math_limit": n,
            "max_new_math": 1024,
            "sample_seed": 2027,
            "math_grader": "math_verify",
            "math_verify_version": "0.9.0",
            "math500_revision": "math-revision",
        },
        "run_provenance": _run_provenance("model"),
        "results": {"FullKV": {"math500": value}},
    }


def _math500_dataset(tmp_path: Path, size: int = 4) -> tuple[Path, str]:
    path = tmp_path / "math500.jsonl"
    path.write_text(
        "".join(
            json.dumps({"problem": f"problem {index}", "answer": str(index)}) + "\n"
            for index in range(size)
        ),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_math500_exact_coverage_sums_integer_correct_and_preserves_audit(tmp_path):
    dataset, digest = _math500_dataset(tmp_path)
    _write(tmp_path / "a/results_model.json", _math_doc(0, 2, {0}))
    _write(tmp_path / "b/results_model.json", _math_doc(2, 4, {2, 3}))
    result = shards.verify_math500(
        tmp_path,
        expected_models=("model",),
        methods=("FullKV",),
        dataset_size=4,
        shard_size=2,
        dataset_revision="math-revision",
        _dataset_path_override=dataset,
        _dataset_sha256_override=digest,
    )
    row = result["models"]["model"]["FullKV"]
    assert row["correct"] == 3
    assert row["pass@1"] == 0.75
    assert row["coverage"] == [0, 4]
    assert row["audit_records_validated"] == 4
    assert row["eviction_applicable"] is False
    assert row["eviction_fired"] is None
    assert result["grading_input"] == "generated_text"


def test_math500_rejects_missing_full_generation(tmp_path):
    dataset, digest = _math500_dataset(tmp_path)
    document = _math_doc(0, 2, {0})
    del document["results"]["FullKV"]["math500"]["per_question"][0]["samples"][0]["generated_text"]
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="full generation"):
        shards.verify_math500(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            dataset_size=4,
            shard_size=2,
            dataset_revision="math-revision",
            _dataset_path_override=dataset,
            _dataset_sha256_override=digest,
        )


def test_math500_rejects_ambiguous_legacy_extracted_answer(tmp_path):
    dataset, digest = _math500_dataset(tmp_path)
    document = _math_doc(0, 2, {0})
    sample = document["results"]["FullKV"]["math500"]["per_question"][0]["samples"][0]
    sample["extracted_answer"] = "0"
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="ambiguous extracted_answer"):
        shards.verify_math500(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            dataset_size=4,
            shard_size=2,
            dataset_revision="math-revision",
            _dataset_path_override=dataset,
            _dataset_sha256_override=digest,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("grading_input", "legacy_regex", "grading input"),
        ("missing_legacy_extraction", None, "non-grading legacy extraction"),
        ("eviction_applicable", True, "non-eviction cache provenance"),
        (
            "compression_ratio_aggregation",
            "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters",
            "FullKV displayed accounting",
        ),
    ],
)
def test_math500_rejects_misleading_raw_schema(tmp_path, field, value, message):
    dataset, digest = _math500_dataset(tmp_path)
    document = _math_doc(0, 2, {0})
    axis = document["results"]["FullKV"]["math500"]
    if field == "missing_legacy_extraction":
        axis["per_question"][0]["samples"][0].pop("legacy_regex_extracted_answer_non_grading")
    elif field == "grading_input":
        axis["per_question"][0]["samples"][0][field] = value
    else:
        axis[field] = value
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match=message):
        shards.verify_math500(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            dataset_size=4,
            shard_size=2,
            dataset_revision="math-revision",
            _dataset_path_override=dataset,
            _dataset_sha256_override=digest,
        )


def test_math500_rejects_tampered_correct_even_when_stored_totals_agree(tmp_path):
    dataset, digest = _math500_dataset(tmp_path)
    document = _math_doc(0, 2, {0})
    value = document["results"]["FullKV"]["math500"]
    sample = value["per_question"][1]["samples"][0]
    sample["correct"] = True
    value["per_question"][1]["pass@1"] = 1.0
    value["per_question"][1]["observed_pass@1"] = 1.0
    value["correct"] = 2
    value["pass@1_sum"] = 2.0
    value["pass@1"] = 1.0
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="Math-Verify replay"):
        shards.verify_math500(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            dataset_size=4,
            shard_size=2,
            dataset_revision="math-revision",
            _dataset_path_override=dataset,
            _dataset_sha256_override=digest,
        )


def test_math500_rejects_tampered_dataset_row_audit(tmp_path):
    dataset, digest = _math500_dataset(tmp_path)
    document = _math_doc(0, 2, {0})
    document["results"]["FullKV"]["math500"]["per_question"][0]["gold"] = "999"
    _write(tmp_path / "results_model.json", document)
    with pytest.raises(shards.AggregateError, match="pinned dataset row"):
        shards.verify_math500(
            tmp_path,
            expected_models=("model",),
            methods=("FullKV",),
            dataset_size=4,
            shard_size=2,
            dataset_revision="math-revision",
            _dataset_path_override=dataset,
            _dataset_sha256_override=digest,
        )


_SPEED_METHODS = (
    "TurboQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
_SPEED_COMPARISONS = {
    "hq_vs_turboquant": (_SPEED_METHODS[0], _SPEED_METHODS[1]),
    "adaptive_hq_vs_turboquant": (_SPEED_METHODS[0], _SPEED_METHODS[2]),
}


def _speed_raw(repeats: int, methods: tuple[str, ...], values: dict[str, float]):
    raw = []
    for repeat in range(repeats):
        offset = repeat % len(methods)
        order = list(methods[offset:] + methods[:offset])
        drift = 1.0 + repeat / 1000.0
        samples = []
        for position, method in enumerate(order):
            base = values[method] * drift
            samples.append(
                {
                    "method": method,
                    "order_position": position,
                    "elapsed_ms_per_call": base,
                    "ttft_ms": 10.0 * base,
                    "decode_steady_mean_ms_per_token": 0.1 * base,
                    "starting_allocated_mib": 900.0,
                    "peak_allocated_mib": 1000.0 + base,
                    "incremental_peak_allocated_mib": 100.0 + base,
                }
            )
        raw.append({"repeat": repeat, "order": order, "samples": samples})
    return raw


def _recorded_speedup_report(raw, *, value_key: str, value_unit: str):
    recomputed = shards._paired_ratios_from_raw(
        raw,
        comparisons=_SPEED_COMPARISONS,
        value_key=value_key,
        expected_repeats=len(raw),
        label="fixture",
    )
    comparisons = {}
    for name, row in recomputed.items():
        comparisons[name] = {
            "baseline_method": row["baseline_method"],
            "candidate_method": row["candidate_method"],
            "n_paired_repeats": row["n_paired_repeats"],
            "raw_paired_ratios": row["raw_paired_ratios"],
            "within_process_summary": {
                "unit": "candidate_speedup_x",
                "n": row["n_paired_repeats"],
                "median": row["process_median_speedup_x"],
            },
        }
    return {
        "ratio_definition": "baseline_latency / candidate_latency",
        "interpretation": "greater_than_1_means_candidate_is_faster",
        "source_value_key": value_key,
        "source_value_unit": value_unit,
        "pairing_unit": "same_counterbalanced_repeat_within_one_process",
        "inferential_unit": "independently_launched_fresh_process",
        "comparisons": comparisons,
    }


def _timing_summary(raw, value_key: str, unit: str):
    methods = raw[0]["order"]
    return {
        method: {
            "unit": unit,
            "n": len(raw),
            "median": sorted(
                sample[value_key]
                for repeat in raw
                for sample in repeat["samples"]
                if sample["method"] == method
            )[len(raw) // 2],
        }
        for method in methods
    }


def _speed_doc(process_index: int) -> dict:
    seed = 20260715 + process_index
    codec_values = {
        _SPEED_METHODS[0]: 10.0,
        _SPEED_METHODS[1]: 5.0 + 0.25 * process_index,
        _SPEED_METHODS[2]: 4.0 + 0.3 * process_index,
    }
    tq, scalar, frontier = _SPEED_METHODS
    taxonomy = {
        tq: {
            "semantic_role": "existing TurboQuant class; symmetric scalar Beta/L2 reference",
            "fake_quant_implementation": "inherited unfused encode_then_decode",
            "proposal_method": False,
            "proposal_variant": None,
        },
        scalar: {
            "semantic_role": "canonical globally fixed HQ paper proposal",
            "fake_quant_implementation": "HayateQuant fused research fake_quant_roundtrip",
            "proposal_method": True,
            "proposal_variant": "fixed_global",
        },
        frontier: {
            "semantic_role": "causal online-rate-allocation HQ paper proposal",
            "fake_quant_implementation": shards.ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION,
            "proposal_method": True,
            "proposal_variant": "causal_adaptive",
        },
    }
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
        "avg_bits": 3.0,
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
    method_configs = {
        tq: {
            **taxonomy[tq],
            "codec": {
                "declared": {
                    **common,
                    "name": "turboquant",
                    "bits": 3,
                    "dist": "beta",
                    "n_outlier": 0,
                    "outlier_bits": None,
                    "use_qjl": False,
                    "qjl_roles": "key",
                    "rotation_mode": "dense",
                    "seed": 0,
                    "value_bits": 3,
                },
                "effective": {},
            },
            "resolved_at_model_head_dim": {
                "schema_version": 1,
                "head_dim": 128,
                "codec": "turboquant",
                "calibration": None,
                "roles": {
                    role: {
                        "bits": 3,
                        "block_size": 1,
                        "rotation_mode": "dense",
                        "rotation_seed": 0,
                        "scale_mode": "l2",
                        "mathematical_scale_mode": "l2",
                        "source_law": "beta",
                        "quantized_dim": 128,
                        "outlier_channels": 0,
                        "outlier_bits": None,
                        "outlier_selection": "none",
                        "qjl_enabled": False,
                    }
                    for role in ("key", "value")
                },
            },
        },
        scalar: {
            **taxonomy[scalar],
            "codec": {
                "declared": {
                    **hayate_defaults,
                    "bits": 3,
                    "value_bits": 3,
                    "dist": "beta",
                    "rotation_mode": "dense",
                    "block_size": 1,
                    "key_scale_mode": "rms",
                    "value_scale_mode": "rms",
                    "n_outlier": 0,
                },
                "effective": {
                    "key_scale_mode": "rms",
                    "value_scale_mode": "rms",
                },
            },
            "resolved_at_model_head_dim": {
                "schema_version": 1,
                "head_dim": 128,
                "codec": "hayatequant",
                "calibration": None,
                "roles": {
                    role: {
                        "bits": 3,
                        "block_size": 1,
                        "rotation_mode": "dense",
                        "rotation_seed": 0,
                        "scale_mode": "rms",
                        "mathematical_scale_mode": "rms",
                        "source_law": "beta",
                        "quantized_dim": 128,
                        "outlier_channels": 0,
                        "outlier_bits": None,
                        "outlier_selection": "none",
                        "qjl_enabled": False,
                    }
                    for role in ("key", "value")
                },
            },
        },
        frontier: {
            **taxonomy[frontier],
            "codec": {
                "declared": {
                    **hayate_defaults,
                    "bits": 4,
                    "value_bits": 2,
                    "dist": "gaussian",
                    "rotation_mode": "auto",
                    "block_size": None,
                    "key_scale_mode": "ub",
                    "value_scale_mode": "ls_iter",
                    "n_outlier": 0,
                },
                "effective": {
                    "key_scale_mode": "ub",
                    "value_scale_mode": "ls_iter",
                },
            },
            "resolved_at_model_head_dim": {
                "schema_version": 1,
                "head_dim": 128,
                "codec": "hayatequant",
                "calibration": None,
                "roles": {
                    "key": {
                        "bits": 4,
                        "block_size": 2,
                        "rotation_mode": "hybrid",
                        "rotation_seed": 0,
                        "scale_mode": "ub",
                        "mathematical_scale_mode": "ip",
                        "source_law": "gaussian",
                        "quantized_dim": 128,
                        "outlier_channels": 0,
                        "outlier_bits": None,
                        "outlier_selection": "none",
                        "qjl_enabled": False,
                    },
                    "value": {
                        "bits": 2,
                        "block_size": 4,
                        "rotation_mode": "dense",
                        "rotation_seed": 0,
                        "scale_mode": "ls_iter",
                        "mathematical_scale_mode": "ls_iter",
                        "source_law": "gaussian",
                        "quantized_dim": 128,
                        "outlier_channels": 0,
                        "outlier_bits": None,
                        "outlier_selection": "none",
                        "qjl_enabled": False,
                    },
                },
            },
        },
    }
    # Replace the historical control/frontier fixture above with the exact
    # formal TQ/HQ/AdaptiveHQ taxonomy admitted by the current boundary.
    hq_declared = {
        **hayate_defaults,
        "required_codebook_bundle": shards.HQ_CODEBOOK_BUNDLE,
        "bits": 3,
        "value_bits": 3,
        "dist": "gaussian",
        "rotation_mode": "auto",
        "block_size": 2,
        "key_scale_mode": "nc",
        "value_scale_mode": "ls_iter",
        "key_code_bits_per_block": 7,
        "value_code_bits_per_block": 5,
        "n_outlier": 0,
    }
    adaptive_declared = {
        **hayate_defaults,
        "required_codebook_bundle": shards.ADAPTIVE_HQ_CODEBOOK_BUNDLE,
        "bits": 3,
        "value_bits": 3,
        "dist": "gaussian",
        "rotation_mode": "auto",
        "block_size": 2,
        "key_scale_mode": "nc",
        "value_scale_mode": "ls_iter",
        "key_bit_fraction": 7 / 12,
        "adaptive_bit_split": "online_stats",
        "n_outlier": 0,
    }
    method_configs = {
        tq: {
            **taxonomy[tq],
            "codec": {
                "declared": {
                    **common,
                    "name": "turboquant",
                    "bits": 3,
                    "dist": "beta",
                    "n_outlier": 0,
                    "outlier_bits": None,
                    "use_qjl": False,
                    "qjl_roles": "key",
                    "rotation_mode": "dense",
                    "seed": 0,
                    "value_bits": 3,
                },
                "effective": {},
            },
            "resolved_at_model_head_dim": shards._expected_speed_resolved_config(
                codec="turboquant", bits=3, head_dim=128
            ),
        },
        scalar: {
            **taxonomy[scalar],
            "codec": {
                "declared": hq_declared,
                "effective": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
            },
            "resolved_at_model_head_dim": shards._expected_speed_resolved_config(
                codec="hq", bits=3, head_dim=128
            ),
        },
        frontier: {
            **taxonomy[frontier],
            "codec": {
                "declared": adaptive_declared,
                "effective": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
            },
            "resolved_at_model_head_dim": shards._expected_speed_resolved_config(
                codec="adaptive_hq", bits=3, head_dim=128
            ),
        },
    }
    recipe_hashes = {
        label: shards._speed_recipe_digest(record) for label, record in method_configs.items()
    }
    micro_raw = _speed_raw(40, _SPEED_METHODS, codec_values)
    measurement = {
        "timer": "torch.cuda.Event",
        "timer_native_unit": "milliseconds",
        "timer_completion_barrier": "end_event.synchronize before elapsed_time read",
        "normalisation": "event elapsed time / 1 inner loop(s)",
        "raw_paired_repeats": micro_raw,
        "summary": _timing_summary(micro_raw, "elapsed_ms_per_call", "ms_per_call"),
        "paired_speedups": _recorded_speedup_report(
            micro_raw,
            value_key="elapsed_ms_per_call",
            value_unit="ms_per_call",
        ),
    }
    regimes = {}
    payload_runtime_audits = {}
    for regime, seq_len in (("prefill", 4096), ("decode", 1)):
        regime_vectors_per_role = 8 * seq_len
        regime_input_elements = 2 * regime_vectors_per_role * 128
        logical_stored_bits = regime_input_elements * 3 + 2 * regime_vectors_per_role * 16
        payload_runtime_audits[regime] = {}
        for label in _SPEED_METHODS:
            index_elements = (
                regime_input_elements
                if label != frontier
                else regime_vectors_per_role * (128 // 2 + 128 // 4)
            )
            payload_runtime_audits[regime][label] = {
                "index_tensor_count": 2,
                "index_elements": index_elements,
                "materialized_index_tensor_bits": index_elements * 64,
                "scale_tensor_count": 2,
                "scale_elements": 2 * regime_vectors_per_role,
                "materialized_scale_tensor_bits": 2 * regime_vectors_per_role * 16,
                "materialized_payload_tensor_bytes": (
                    index_elements * 8 + 2 * regime_vectors_per_role * 2
                ),
                "materialized_payload_tensor_scope": shards.PUBLIC_PAYLOAD_TENSOR_SCOPE,
                "logical_total_stored_bits": logical_stored_bits,
                "all_index_tensors_int64": True,
                "index_tensors_bit_packed": False,
                "all_scale_tensors_signed_fp16": True,
            }
        regimes[regime] = {
            "tensor": {
                "shape": [1, 8, seq_len, 128],
                "dtype": "float16",
                "generated_on_device": True,
                "elements_per_key_value_pair": 2 * 8 * seq_len * 128,
            },
            "timed_recipe_sha256": recipe_hashes,
            "payload_api": {
                "encode": measurement,
                "decode_prebuilt_payload": measurement,
                "roundtrip_explicit_encode_then_decode": measurement,
            },
            "research_fake_quant": {
                "effective_implementations": {
                    tq: {
                        "effective_path": "inherited_encode_then_decode",
                        "payload_materialized": True,
                        "hayate_override_entered": False,
                        "runtime_probe": "codec_has_no_hayate_rt_support_probe",
                    },
                    scalar: {
                        "effective_path": "hayate_fused_roundtrip",
                        "payload_materialized": False,
                        "hayate_override_entered": True,
                        "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                    },
                    frontier: {
                        "effective_path": "inherited_encode_then_decode",
                        "payload_materialized": True,
                        "hayate_override_entered": False,
                        "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                    },
                },
                "fake_quant_roundtrip": measurement,
            },
        }

    e2e_values = {"fullkv_uncompressed": 12.0, **codec_values}
    e2e_raw = _speed_raw(5, ("fullkv_uncompressed", *_SPEED_METHODS), e2e_values)
    e2e_summary = {
        method: {
            "ttft": {
                "unit": "ms",
                "n": 5,
                "median": sorted(
                    sample["ttft_ms"]
                    for repeat in e2e_raw
                    for sample in repeat["samples"]
                    if sample["method"] == method
                )[2],
            },
            "decode_steady_latency": {
                "unit": "ms_per_token",
                "n": 5,
                "median": sorted(
                    sample["decode_steady_mean_ms_per_token"]
                    for repeat in e2e_raw
                    for sample in repeat["samples"]
                    if sample["method"] == method
                )[2],
            },
            "peak_allocated_memory": {
                "unit": "MiB",
                "n": 5,
                "median": sorted(
                    sample["peak_allocated_mib"]
                    for repeat in e2e_raw
                    for sample in repeat["samples"]
                    if sample["method"] == method
                )[2],
            },
            "incremental_peak_allocated_memory": {
                "unit": "MiB_above_synchronized_start",
                "n": 5,
                "median": sorted(
                    sample["incremental_peak_allocated_mib"]
                    for repeat in e2e_raw
                    for sample in repeat["samples"]
                    if sample["method"] == method
                )[2],
            },
        }
        for method in e2e_values
    }
    args = {
        "mode": "both",
        "local_files_only": True,
        "device": "cuda:0",
        "compile_mode": "off",
        "micro_repeats": 40,
        "e2e_repeats": 5,
        "sink_tokens": 0,
        "recent_tokens": 0,
        "dtype": "float16",
        "fused_backend": "triton",
        "batch_size": 1,
        "micro_seq_len": 4096,
        "decode_seq_len": 1,
        "quality_query_len": 8,
        "micro_warmup": 8,
        "micro_inner_loops": 1,
        "bootstrap_resamples": 10000,
        "e2e_prompt_len": 4096,
        "e2e_gen_len": 64,
        "e2e_warmup": 1,
        "attention_implementation": "sdpa",
        "seed": seed,
    }
    input_elements = 2 * 8 * 4096 * 128
    stored_bits = input_elements * 3 + 2 * (8 * 4096) * 16
    quality = {
        label: {
            "recipe_sha256": recipe_hashes[label],
            "input_kv_elements": input_elements,
            "stored_bits": stored_bits,
            "stored_bits_per_kv_element": stored_bits / input_elements,
            "key_mse": 0.1,
            "value_mse": 0.1,
            "attention_output_relative_l2": 0.1,
        }
        for label in _SPEED_METHODS
    }
    rate_methods = {}
    for label in _SPEED_METHODS:
        rate_methods[label] = {
            "declared_key_bits": 3,
            "declared_value_bits": 3,
            "nominal_average_code_bits_per_coordinate": 3.0,
            "allocation": (
                "symmetric_scalar"
                if label == tq
                else "fixed_global" if label == scalar else "causal_adaptive"
            ),
            "accounted_payload_bits": stored_bits,
            "input_kv_elements": input_elements,
            "accounted_payload_bits_per_kv_element": stored_bits / input_elements,
        }
    job_id = str(1000 + process_index)
    semantic_config = dict(args)
    run_fingerprint = hashlib.sha256(
        json.dumps(
            {"semantic_config": semantic_config, "code_commit": "test-commit"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "schema": {"name": "onecomp.hayatequant_fair_speed", "version": 4},
        "model": "model",
        "config": {"model_path": "/models/model"},
        "code_commit": "test-commit",
        "semantic_config": semantic_config,
        "run_fingerprint": run_fingerprint,
        "run_provenance": _run_provenance("model", slurm_job_id=job_id),
        "protocol": {
            "arguments": args,
            "model_evaluation_role": "custom_evaluation_role_unspecified",
            "average_bits": 3,
            "model": "/models/model",
            "model_kv_shape": {
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "gqa_group_size": 4,
                "num_hidden_layers": 32,
                "kv_heads_derivation": "config:num_key_value_heads",
                "head_dim_derivation": "config:head_dim",
            },
            "model_context_capacity": {
                "tokens": 8192,
                "derivation": "config:max_position_embeddings",
                "original_max_position_embeddings": None,
            },
            "method_taxonomy": taxonomy,
            "compile_runtime_audit": {
                "requested_mode": "off",
                "compile_all_forbidden": True,
                "prefill_compile_forbidden": True,
                "decode_compiled_dispatch_created": False,
                "matches_requested_mode": True,
            },
            "comparison_rules": {
                "baseline_method": "TurboQuant-paper",
                "proposal_methods": ["HQ", "AdaptiveHQ"],
                "method_set_exact": True,
                "real_arithmetic_matched_scalar_control": [tq, scalar],
                "implementation_only_claim_allowed": False,
                "scalar_control_residual_difference": (
                    "TurboQuant stores fp16 L2 scale; Hayate stores equivalent fp16 "
                    "RMS scale with a sqrt(head_dim)-rescaled Beta codebook, so metadata "
                    "rounding and norm-kernel work differ"
                ),
                "canonical_hq_operating_point": frontier,
                "speedup_ratio": "baseline_latency / candidate_latency",
                "formal_inferential_unit": "independently launched fresh process",
                "technical_repeats_must_not_be_pooled_across_processes": True,
                "directional_speed_claim_allowed": False,
                "speed_reporting_scope": "descriptive ratio and pointwise nominal CI only",
                "packed_native_hybrid_measured": False,
                "packed_native_hybrid_must_not_be_called_full_hayatequant": True,
            },
        },
        "provenance": {
            "run_uuid": f"00000000-0000-4000-8000-{process_index + 1:012d}",
            "pid": 2000 + process_index,
            "started_at_utc": f"2026-07-15T00:00:0{process_index}+00:00",
            "hostname": "kagura-gpu06",
            "git": {"dirty": False, "commit": "test-commit"},
            "python": "Python test build",
            "platform": "test-platform",
            "packages": {
                "torch": "2.9",
                "transformers": "4.0",
                "triton": "3.0",
                "cuda_runtime_reported_by_torch": "13.0",
                "cudnn": 999,
            },
            "cuda_device": {
                "name": "Test GPU",
                "compute_capability": [10, 0],
                "total_memory_bytes": 1000,
                "multi_processor_count": 10,
            },
            "selected_gpu_nvidia_smi_match": [{"driver_version": "999.0"}],
            "selected_environment": {
                "SLURM_JOB_ID": job_id,
                "SLURM_NODELIST": "kagura-gpu06",
                "SLURMD_NODENAME": "kagura-gpu06",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            "fresh_process_contract": {
                "required": True,
                "multiple_average_rates_accepted_by_cli": False,
                "compile_all_allowed": False,
                "only_allowed_compile_mode": "decode-only",
            },
            "effective_onecomp_environment": {
                "ONECOMP_COMPILE_CODEC": "0",
                "ONECOMP_COMPILE_DECODE_ONLY": "0",
                "ONECOMP_TRITON_CODEC": "1",
                "ONECOMP_TRITON_PREFILL": "0",
            },
        },
        "micro": {
            "public_payload_scope": dict(shards.PUBLIC_LOGICAL_PAYLOAD_SCOPE),
            "public_payload_runtime_audit": payload_runtime_audits,
            "method_configs": method_configs,
            "recipe_sha256": recipe_hashes,
            "recipe_linkage": {
                "payload_quality_and_all_micro_timings_use_same_codec_instances": True,
                "all_rows_fingerprinted_from_declared_and_role_resolved_codec_config": True,
            },
            "rate_contract": {
                "nominal_average_bits": 3,
                "equal_input_shape": True,
                "equal_accounted_payload_bits": True,
                "methods": rate_methods,
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
                        "selector": method_configs[frontier]["resolved_at_model_head_dim"][
                            "adaptive_allocation"
                        ],
                    },
                },
            },
            "quality": quality,
            "parity_checks": {
                "proposal_payload_vs_fake_quant_quality": [
                    {
                        "method": label,
                        "role": role,
                        "effective_implementation": (
                            {
                                "effective_path": "hayate_fused_roundtrip",
                                "payload_materialized": False,
                                "hayate_override_entered": True,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            }
                            if label == scalar
                            else {
                                "effective_path": "inherited_encode_then_decode",
                                "payload_materialized": True,
                                "hayate_override_entered": False,
                                "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
                            }
                        ),
                        "stored_bits_from_fake_quant": stored_bits,
                        "stored_bits_from_quality_payload": stored_bits,
                        "stored_bits_match": True,
                    }
                    for label in (scalar, frontier)
                    for role in ("key", "value")
                ],
            },
            "regimes": regimes,
        },
        "e2e": {
            "native_hybrid_is_full_hayatequant": False,
            "timer_completion_barrier": "end_event.synchronize before elapsed_time read",
            "peak_allocated_memory_scope": dict(shards.SPEED_PEAK_ALLOCATED_SCOPE),
            "model_context_contract": {
                "prompt_tokens": 4096,
                "decode_steps": 64,
                "required_context_tokens": 4160,
                "model_capacity_tokens": 8192,
                "capacity_derivation": "config:max_position_embeddings",
                "fits": True,
            },
            "input": {"shape": [1, 4096], "decode_steps": 64},
            "model_kv_shape_from_config": {
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "gqa_group_size": 4,
                "num_hidden_layers": 32,
                "kv_heads_derivation": "config:num_key_value_heads",
                "head_dim_derivation": "config:head_dim",
            },
            "method_configs": method_configs,
            "recipe_sha256": recipe_hashes,
            "cache_factory_recipe_revalidated_on_every_timed_construction": True,
            "attention_implementation_requested": "sdpa",
            "attention_implementation_effective": "sdpa",
            "raw_paired_repeats": e2e_raw,
            "summary": e2e_summary,
            "paired_speedups": {
                "ttft": _recorded_speedup_report(e2e_raw, value_key="ttft_ms", value_unit="ms"),
                "decode_steady_latency": _recorded_speedup_report(
                    e2e_raw,
                    value_key="decode_steady_mean_ms_per_token",
                    value_unit="ms_per_token",
                ),
            },
        },
    }


def _write_speed_matrix(root: Path):
    for process_index in range(5):
        _write(root / f"rep{process_index}.json", _speed_doc(process_index))


@pytest.mark.parametrize(
    ("rate", "key_spec", "value_spec"),
    [
        (2, (2, 2, "hybrid"), (2, 2, "dense")),
        (3, (3, 2, "hybrid"), (3, 2, "hybrid")),
        (4, (4, 2, "hybrid"), (4, 2, "hybrid")),
    ],
)
def test_speed_aggregate_expectation_is_rate_and_role_specific(rate, key_spec, value_spec):
    record = shards._expected_speed_resolved_config(  # pylint: disable=protected-access
        codec="hq", bits=rate, head_dim=128
    )
    key = record["roles"]["key"]
    value = record["roles"]["value"]
    assert (key["bits"], key["block_size"], key["rotation_mode"]) == key_spec
    assert (value["bits"], value["block_size"], value["rotation_mode"]) == value_spec
    assert key["code_bits_per_block"] == 2 * rate + 1
    assert value["code_bits_per_block"] == 2 * rate - 1
    assert (key["scale_mode"], key["mathematical_scale_mode"]) == ("nc", "nc")
    assert (value["scale_mode"], value["mathematical_scale_mode"]) == ("ls_iter", "ls_iter")
    assert key["source_law"] == value["source_law"] == "gaussian"
    assert key["rotation_seed"] == value["rotation_seed"] == 0


def test_speed_aggregation_uses_process_level_paired_ratios_and_t_ci(tmp_path):
    _write_speed_matrix(tmp_path)
    result = shards.verify_speed(
        tmp_path,
        expected_replicates=5,
        expected_models=("model",),
        expected_rates=(3,),
        expected_compile_modes=("off",),
    )
    comparison = result["models"]["model"]["3"]["off"]["paired_speedups"][
        "micro.prefill.payload_api.encode"
    ]["hq_vs_turboquant"]
    assert comparison["n_independent_processes"] == 5
    assert comparison["student_t_df"] == 4
    assert len(comparison["processes"]) == 5
    assert len(comparison["processes"][0]["raw_paired_ratios"]) == 40
    expected = math.prod(10.0 / (5.0 + 0.25 * index) for index in range(5)) ** (1.0 / 5.0)
    assert comparison["geometric_mean_process_speedup_x"] == pytest.approx(expected)
    low, high = comparison["student_t_log_geomean_ci95"]
    assert low < expected < high
    assert result["statistical_unit"] == "independently_launched_fresh_process"
    assert result["public_payload_scope"] == shards.PUBLIC_LOGICAL_PAYLOAD_SCOPE
    assert result["model_evaluation_roles"] == {"model": "custom_evaluation_role_unspecified"}
    assert result["multiplicity_contract"] == {
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
        "blanket_or_global_speed_claim_allowed": result["pareto_contract"][
            "complete_strict_superiority_established"
        ],
    }
    memory = result["models"]["model"]["3"]["off"]["paired_peak_allocated_memory"][
        "hq_vs_turboquant"
    ]
    assert memory["ratio"]["n_independent_processes"] == 5
    assert memory["difference"]["n_independent_processes"] == 5
    assert len(memory["processes"][0]["raw_paired_memory"]) == 5
    assert result["pareto_contract"]["all_predeclared_axes_present"] is True
    assert result["pareto_contract"]["n_simultaneous_stochastic_endpoints"] == 22
    assert result["independent_process_audit"] == {
        "n_unique_run_uuid4": 5,
        "n_unique_slurm_job_ids": 5,
        "uuid_and_slurm_job_id_unique_for_every_source": True,
    }
    assert result["allocation_isolation_contract"] == {
        **shards.SPEED_ALLOCATION_CONTRACT,
        "physical_node": "kagura-gpu06",
        "allocation_cuda_visibility_recorded_before_process_narrowing": True,
    }
    resolved = result["models"]["model"]["3"]["off"]["scientific_recipe"]["method_configs"][
        _SPEED_METHODS[1]
    ]["resolved_at_model_head_dim"]
    assert resolved["roles"]["key"]["rotation_mode"] == "hybrid"
    assert resolved["roles"]["value"]["rotation_mode"] == "hybrid"
    adaptive = result["models"]["model"]["3"]["off"]["scientific_recipe"]["method_configs"][
        "AdaptiveHQ"
    ]["resolved_at_model_head_dim"]
    assert adaptive["roles"]["key"]["rotation_mode"] == "payload_dependent_candidate_rate"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SLURM_JOB_PARTITION", "batch-1gpu"),
        ("SLURM_GPUS_ON_NODE", "1"),
        ("SLURM_JOB_GPUS", "0"),
        ("SLURM_CPUS_PER_TASK", "12"),
        ("SLURM_JOB_CPUS_PER_NODE", "12"),
        ("SLURM_MEM_PER_NODE", "225280"),
        ("PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES", "0"),
        ("PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE", "1"),
        ("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
    ],
)
def test_speed_aggregation_rejects_allocation_or_visibility_tampering(tmp_path, field, value):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep2.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["run_provenance"][0]["record"]["environment"][field] = value
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="allocation|GPU|visibility"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_node_identity_tampering(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep2.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["provenance"]["hostname"] = "kagura-gpu07"
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="node identity"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_role_resolved_recipe_tampering(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep3.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    frontier = document["micro"]["method_configs"][_SPEED_METHODS[-1]]
    frontier["resolved_at_model_head_dim"]["roles"]["value"]["rotation_mode"] = "hybrid"
    # Rehashing both micro and E2E proves the rejection comes from the
    # scientific expectation, not merely from a stale checksum.
    digest = shards._speed_recipe_digest(frontier)  # pylint: disable=protected-access
    document["micro"]["recipe_sha256"][_SPEED_METHODS[-1]] = digest
    document["e2e"]["method_configs"] = document["micro"]["method_configs"]
    document["e2e"]["recipe_sha256"] = document["micro"]["recipe_sha256"]
    for regime in document["micro"]["regimes"].values():
        regime["timed_recipe_sha256"] = document["micro"]["recipe_sha256"]
    document["micro"]["quality"][_SPEED_METHODS[-1]]["recipe_sha256"] = digest
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="Role-resolved codec recipe changed"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


@pytest.mark.parametrize("drift", ["software", "hardware", "protocol"])
def test_speed_aggregation_rejects_cross_stratum_pooling(tmp_path, drift):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep2.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if drift == "software":
        document["provenance"]["packages"]["torch"] = "different"
    elif drift == "hardware":
        document["provenance"]["cuda_device"]["name"] = "Different GPU"
    else:
        document["protocol"]["arguments"]["micro_seq_len"] = 2048
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="strata|stratum|protocol drift"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_tampered_paired_ratio(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    row = document["micro"]["regimes"]["prefill"]["payload_api"]["encode"]["paired_speedups"][
        "comparisons"
    ]["hq_vs_turboquant"]["raw_paired_ratios"][0]
    row["candidate_speedup_x"] *= 10.0
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="ratio mismatch"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


@pytest.mark.parametrize("identity", ["run_uuid", "slurm_job_id"])
def test_speed_aggregation_rejects_reused_process_identity(tmp_path, identity):
    _write_speed_matrix(tmp_path)
    first = json.loads((tmp_path / "rep0.json").read_text(encoding="utf-8"))
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if identity == "run_uuid":
        document["provenance"]["run_uuid"] = first["provenance"]["run_uuid"]
    else:
        job_id = first["provenance"]["selected_environment"]["SLURM_JOB_ID"]
        document["provenance"]["selected_environment"]["SLURM_JOB_ID"] = job_id
        document["run_provenance"][0]["record"]["environment"]["SLURM_JOB_ID"] = job_id
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="Reused|reused Slurm"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_recipe_or_rate_tampering(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    method = _SPEED_METHODS[2]
    document["micro"]["quality"][method]["stored_bits"] += 1
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="Quality/rate|equal-rate"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_physical_packing_mislabel(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["micro"]["public_payload_scope"]["index_tensors_bit_packed"] = True
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="unpacked int64"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_recomputes_and_rejects_tampered_peak_allocated_memory(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    sample = document["e2e"]["raw_paired_repeats"][0]["samples"][0]
    sample["peak_allocated_mib"] += 100.0
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="allocated-memory counters"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_logical_physical_payload_byte_conflation(tmp_path):
    _write_speed_matrix(tmp_path)
    path = tmp_path / "rep1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    audit = document["micro"]["public_payload_runtime_audit"]["decode"][_SPEED_METHODS[-1]]
    audit["materialized_payload_tensor_bytes"] = audit["logical_total_stored_bits"] // 8
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="Public payload indices"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=5,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def test_speed_aggregation_rejects_two_process_formal_design(tmp_path):
    _write_speed_matrix(tmp_path)
    with pytest.raises(shards.AggregateError, match="exactly five"):
        shards.verify_speed(
            tmp_path,
            expected_replicates=2,
            expected_models=("model",),
            expected_rates=(3,),
            expected_compile_modes=("off",),
        )


def _write_repo_local_receipt_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "paper-run"
    source = root / "results/formal/ppl/model/results.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"result": "ok"}\n', encoding="utf-8")
    provenance = root / "provenance/formal-ppl_123.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}\n", encoding="utf-8")
    stdout = root / "logs/formal/hq-f-ppl_123.out"
    stderr = root / "logs/formal/hq-f-ppl_123.err"
    stdout.parent.mkdir(parents=True)
    stdout.write_text("completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    attempt_root = root / "job_records/formal/formal-ppl/attempt-1"
    attempt_root.mkdir(parents=True)
    scheduler_stdout = attempt_root / "sbatch.stdout"
    scheduler_stderr = attempt_root / "sbatch.stderr"
    scheduler_stdout.write_text("123\n", encoding="utf-8")
    scheduler_stderr.write_text("", encoding="utf-8")
    expected_relative = source.relative_to(root).as_posix()
    identity = {
        "schema_version": 2,
        "phase": "formal",
        "job_key": "formal-ppl",
        "attempt_id": "attempt-1",
        "source_commit": "c" * 40,
        "run_id": root.name,
        "expected_result": expected_relative,
        "results_worktree": str(root.parents[4].resolve(strict=True)),
        "scheduler_user": "fixture-user",
        "scheduler_job_name": "hq-f-ppl",
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "updated_at_utc": "2026-07-16T00:01:00+00:00",
        "terminal": True,
        "stage": "fixture",
    }
    tag_fields = (
        "hayatequant-paper-submission-v1",
        identity["source_commit"],
        identity["run_id"],
        identity["results_worktree"],
        identity["phase"],
        identity["job_key"],
        identity["attempt_id"],
        identity["expected_result"],
    )
    identity["submission_tag"] = "hayatequant-" + hashlib.sha256(
        b"".join(str(value).encode() + b"\0" for value in tag_fields)
    ).hexdigest()
    submission = {
        **identity,
        "kind": "hayatequant_paper_submission_receipt",
        "status": "submitted",
        "reason_code": "sbatch_accepted",
        "exit_code": 0,
        "slurm_job_id": "123",
        "scheduler_stdout": scheduler_stdout.relative_to(root).as_posix(),
        "scheduler_stderr": scheduler_stderr.relative_to(root).as_posix(),
        "scheduler_submission_evidence": {
            "stdout": {
                "path": scheduler_stdout.relative_to(root).as_posix(),
                "bytes": scheduler_stdout.stat().st_size,
                "sha256": hashlib.sha256(scheduler_stdout.read_bytes()).hexdigest(),
            },
            "stderr": {
                "path": scheduler_stderr.relative_to(root).as_posix(),
                "bytes": scheduler_stderr.stat().st_size,
                "sha256": hashlib.sha256(scheduler_stderr.read_bytes()).hexdigest(),
            },
        },
    }
    worker = {
        **identity,
        "kind": "hayatequant_paper_worker_receipt",
        "status": "succeeded",
        "reason_code": "benchmark_and_result_validated",
        "exit_code": 0,
        "slurm_job_id": "123",
        "stdout_log": stdout.relative_to(root).as_posix(),
        "stderr_log": stderr.relative_to(root).as_posix(),
        "provenance": provenance.relative_to(root).as_posix(),
        "result": {
            "path": expected_relative,
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "json_object_validated": True,
            "duplicate_keys_rejected": True,
            "nonfinite_numbers_rejected": True,
        },
    }
    _write(attempt_root / "submission.json", submission)
    _write(attempt_root / "worker.json", worker)
    return root, source, attempt_root / "worker.json"


def _add_scheduler_failure_attempt(root: Path, source: Path) -> Path:
    attempt_root = root / "job_records/formal/formal-ppl/attempt-0"
    attempt_root.mkdir(parents=True)
    scheduler_stdout = attempt_root / "sbatch.stdout"
    scheduler_stderr = attempt_root / "sbatch.stderr"
    scheduler_stdout.write_text("122\n", encoding="utf-8")
    scheduler_stderr.write_text("", encoding="utf-8")
    evidence_root = attempt_root / "scheduler_evidence"
    evidence_root.mkdir()
    sacct_stdout = evidence_root / "sacct.stdout"
    sacct_stderr = evidence_root / "sacct.stderr"
    sacct_stdout.write_text(
        "122|fixture-user|hq-f-ppl|NODE_FAIL|1:0|2026-07-16T00:00:00|"
        "2026-07-16T00:00:00|2026-07-16T00:01:00|placeholder|\n",
        encoding="utf-8",
    )
    sacct_stderr.write_text("", encoding="utf-8")
    identity = {
        "schema_version": 2,
        "phase": "formal",
        "job_key": "formal-ppl",
        "attempt_id": "attempt-0",
        "source_commit": "c" * 40,
        "run_id": root.name,
        "expected_result": source.relative_to(root).as_posix(),
        "results_worktree": str(root.parents[4].resolve(strict=True)),
        "scheduler_user": "fixture-user",
        "scheduler_job_name": "hq-f-ppl",
        "created_at_utc": "2026-07-16T00:00:00+00:00",
        "updated_at_utc": "2026-07-16T00:01:00+00:00",
        "terminal": True,
        "stage": "fixture",
    }
    fields = (
        "hayatequant-paper-submission-v1",
        identity["source_commit"],
        identity["run_id"],
        identity["results_worktree"],
        identity["phase"],
        identity["job_key"],
        identity["attempt_id"],
        identity["expected_result"],
    )
    identity["submission_tag"] = "hayatequant-" + hashlib.sha256(
        b"".join(str(value).encode() + b"\0" for value in fields)
    ).hexdigest()
    sacct_stdout.write_text(
        "122|fixture-user|hq-f-ppl|NODE_FAIL|1:0|2026-07-16T00:00:00|"
        "2026-07-16T00:00:00|2026-07-16T00:01:00|"
        f"{identity['submission_tag']}|\n",
        encoding="utf-8",
    )
    submission = {
        **identity,
        "kind": "hayatequant_paper_submission_receipt",
        "status": "submitted",
        "reason_code": "sbatch_accepted",
        "exit_code": 0,
        "slurm_job_id": "122",
        "scheduler_stdout": scheduler_stdout.relative_to(root).as_posix(),
        "scheduler_stderr": scheduler_stderr.relative_to(root).as_posix(),
        "scheduler_submission_evidence": {
            "stdout": {
                "path": scheduler_stdout.relative_to(root).as_posix(),
                "bytes": scheduler_stdout.stat().st_size,
                "sha256": hashlib.sha256(scheduler_stdout.read_bytes()).hexdigest(),
            },
            "stderr": {
                "path": scheduler_stderr.relative_to(root).as_posix(),
                "bytes": scheduler_stderr.stat().st_size,
                "sha256": hashlib.sha256(scheduler_stderr.read_bytes()).hexdigest(),
            },
        },
    }
    scheduler = {
        **identity,
        "kind": "hayatequant_paper_scheduler_receipt",
        "status": "scheduler_terminal_failure",
        "reason_code": "slurm_terminal_before_worker",
        "slurm_job_id": "122",
        "slurm": {
            "job_id_raw": "122",
            "user": identity["scheduler_user"],
            "job_name": identity["scheduler_job_name"],
            "state": "NODE_FAIL",
            "state_raw": "NODE_FAIL",
            "exit_code": "1:0",
            "submit_time": "2026-07-16T00:00:00",
            "start_time": "2026-07-16T00:00:00",
            "end_time": "2026-07-16T00:01:00",
            "submission_tag": identity["submission_tag"],
        },
        "scheduler_query_evidence": {
            "stdout": {
                "path": sacct_stdout.relative_to(root).as_posix(),
                "bytes": sacct_stdout.stat().st_size,
                "sha256": hashlib.sha256(sacct_stdout.read_bytes()).hexdigest(),
            },
            "stderr": {
                "path": sacct_stderr.relative_to(root).as_posix(),
                "bytes": sacct_stderr.stat().st_size,
                "sha256": hashlib.sha256(sacct_stderr.read_bytes()).hexdigest(),
            },
        },
    }
    _write(attempt_root / "submission.json", submission)
    scheduler_path = attempt_root / "scheduler.json"
    _write(scheduler_path, scheduler)
    return scheduler_path


def test_formal_aggregate_requires_repo_local_success_receipts(tmp_path):
    root, source, _ = _write_repo_local_receipt_fixture(tmp_path)
    audit = shards._validate_formal_job_receipts(
        root, source_paths=[source], expected_commit="c" * 40
    )
    assert audit["source_results_validated"] == 1
    assert audit["worker_status_counts"] == {"succeeded": 1}


def test_formal_aggregate_rejects_result_changed_after_success_receipt(tmp_path):
    root, source, _ = _write_repo_local_receipt_fixture(tmp_path)
    source.write_text('{"result": "tampered"}\n', encoding="utf-8")
    with pytest.raises(shards.AggregateError, match="changed after receipt"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )


def test_formal_aggregate_rejects_nonterminal_or_missing_worker_success(tmp_path):
    root, source, worker_path = _write_repo_local_receipt_fixture(tmp_path)
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["status"] = "failed"
    worker["exit_code"] = 9
    worker.pop("result")
    _write(worker_path, worker)
    with pytest.raises(shards.AggregateError, match="exactly one successful worker receipt"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )


def test_formal_aggregate_rejects_submitted_attempt_without_terminal_worker(tmp_path):
    root, source, worker_path = _write_repo_local_receipt_fixture(tmp_path)
    worker_path.unlink()
    with pytest.raises(shards.AggregateError, match="exactly one terminal execution receipt"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )


def test_formal_aggregate_rejects_failed_worker_with_mismatched_submission(tmp_path):
    root, source, worker_path = _write_repo_local_receipt_fixture(tmp_path)
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["status"] = "failed"
    worker["exit_code"] = 9
    worker["slurm_job_id"] = "999"
    worker.pop("result")
    _write(worker_path, worker)
    with pytest.raises(shards.AggregateError, match="Slurm ID"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )


def test_formal_aggregate_allows_failed_scheduler_attempt_then_successful_retry(tmp_path):
    root, source, _worker_path = _write_repo_local_receipt_fixture(tmp_path)
    _add_scheduler_failure_attempt(root, source)
    audit = shards._validate_formal_job_receipts(
        root, source_paths=[source], expected_commit="c" * 40
    )
    assert audit["worker_status_counts"] == {"succeeded": 1}
    assert audit["scheduler_status_counts"] == {"scheduler_terminal_failure": 1}
    assert audit["source_results_validated"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "succeeded"), ("slurm_state", "COMPLETED")],
)
def test_formal_aggregate_never_counts_scheduler_receipt_as_success(
    tmp_path, field, value
):
    root, source, _worker_path = _write_repo_local_receipt_fixture(tmp_path)
    scheduler_path = _add_scheduler_failure_attempt(root, source)
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    if field == "slurm_state":
        scheduler["slurm"]["state"] = value
    else:
        scheduler[field] = value
    _write(scheduler_path, scheduler)
    with pytest.raises(shards.AggregateError, match="scheduler"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )


@pytest.mark.parametrize("receipt_name", ["submission.json", "worker.json", "scheduler.json"])
def test_formal_aggregate_binds_every_receipt_to_exact_run_and_results_worktree(
    tmp_path, receipt_name
):
    root, source, worker_path = _write_repo_local_receipt_fixture(tmp_path)
    scheduler_path = _add_scheduler_failure_attempt(root, source)
    if receipt_name == "submission.json":
        path = worker_path.with_name(receipt_name)
    elif receipt_name == "worker.json":
        path = worker_path
    else:
        path = scheduler_path
    document = json.loads(path.read_text(encoding="utf-8"))
    document["run_id"] = "different-run"
    document["results_worktree"] = "/different/results-worktree"
    _write(path, document)
    with pytest.raises(shards.AggregateError, match="immutable identity"):
        shards._validate_formal_job_receipts(
            root, source_paths=[source], expected_commit="c" * 40
        )
