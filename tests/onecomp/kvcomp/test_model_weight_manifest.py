"""Model weights are content-pinned and rechecked before paper jobs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "benchmark/kvcomp-codec"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_weight_manifest as weights

PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "paper_preflight_for_weight_test", SCRIPT_DIR / "paper_preflight.py"
)
assert PREFLIGHT_SPEC is not None and PREFLIGHT_SPEC.loader is not None
preflight = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(preflight)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _approved_smoke(tmp_path: Path, *, commit: str = "a" * 40) -> dict:
    run_root = tmp_path / "run"
    input_manifest = run_root / "manifests/inputs.json"
    _write_json(input_manifest, {"schema_version": 2, "status": "ok"})
    result_records = []
    preflight_paths = {}
    for key in preflight.SMOKE_JOB_KEYS:
        result = run_root / "results/smoke" / f"{key}.json"
        if key == "native-hybrid":
            document = {"kind": "native-smoke", "status": "ok"}
        else:
            worker = run_root / "provenance" / f"smoke-{key}_123.json"
            worker_document = {
                "schema_version": 1,
                "code": {"commit": commit, "tracked_dirty": False},
                "job_key": key,
            }
            _write_json(worker, worker_document)
            preflight_paths[key] = worker
            document = {
                "run_provenance": [
                    {
                        "path": str(worker.resolve()),
                        "sha256": _digest(worker),
                        "record": worker_document,
                    }
                ]
            }
        _write_json(result, document)
        result_records.append({"key": key, **_file_record(result)})
    native_worker = run_root / "provenance/smoke-native-hybrid_123.json"
    _write_json(native_worker, {"schema_version": 1, "job_key": "native-hybrid"})
    preflight_paths["native-hybrid"] = native_worker
    plan = {
        "smoke_plan_version": preflight.SMOKE_PLAN_VERSION,
        "smoke_plan_sha256": "b" * 64,
        "code_commit": commit,
        "input_manifest": _file_record(input_manifest),
        "results": result_records,
        "native_worker_preflights": [_file_record(native_worker)],
        "completed_job_keys": list(preflight.SMOKE_JOB_KEYS),
    }
    marker = run_root / "manifests/formal_smoke_approval.json"
    _write_json(
        marker,
        {
            "schema_version": preflight.SMOKE_APPROVAL_SCHEMA_VERSION,
            "kind": preflight.SMOKE_APPROVAL_KIND,
            "approval_statement": preflight.SMOKE_APPROVAL_STATEMENT,
            "approved_at_utc": "2026-07-16T00:00:00+00:00",
            "approved_by": {"user": "pytest", "host": "localhost"},
            "validated_plan": plan,
        },
    )
    return {
        "run_root": run_root,
        "input_manifest": input_manifest,
        "commit": commit,
        "marker": marker,
        "marker_sha256": _digest(marker),
        "plan_sha256": preflight._canonical_sha256(plan),
        "results": {record["key"]: Path(record["path"]) for record in result_records},
        "preflights": preflight_paths,
    }


def _model(root: Path) -> Path:
    model = root / "model"
    _write_json(model / "config.json", {"model_type": "synthetic"})
    _write_json(model / "tokenizer_config.json", {"tokenizer_class": "Synthetic"})
    _write_json(model / "tokenizer.json", {"version": "1"})
    (model / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    _write_json(
        model / "model.safetensors.index.json",
        {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
    )
    return model


def _input_manifest(path: Path, model: Path, fingerprint: dict) -> Path:
    metadata = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        candidate = model / name
        metadata[name] = {
            "path": str(candidate.resolve()),
            "bytes": candidate.stat().st_size,
            "sha256": _digest(candidate),
        }
    _write_json(
        path,
        {
            "schema_version": 2,
            "kind": "hayatequant_paper_input_validation",
            "status": "ok",
            "models": {
                "synthetic": {
                    "path": str(model.resolve()),
                    "metadata_files": metadata,
                    "weights": fingerprint,
                }
            },
        },
    )
    return path


def test_every_verification_full_hashes_and_catches_same_size_content_change(tmp_path):
    model = _model(tmp_path)
    expected = weights.fingerprint_model_weights(model)

    actual = weights.verify_model_weights(model, expected)
    assert actual == expected

    shard = model / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"FIRST-shard")  # same size, different content
    with pytest.raises(weights.ModelWeightError, match="differ from the pinned"):
        weights.verify_model_weights(model, expected)


def test_verification_rejects_repointed_symlink_even_with_same_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "one").write_bytes(b"same payload")
    (model / "two").write_bytes(b"same payload")
    link = model / "model.safetensors"
    link.symlink_to("one")
    expected = weights.fingerprint_model_weights(model)
    weights.verify_model_weights(model, expected)

    link.unlink()
    link.symlink_to("two")
    with pytest.raises(weights.ModelWeightError, match="differ from the pinned"):
        weights.verify_model_weights(model, expected)


def test_preflight_binds_metadata_and_weights_to_input_manifest(tmp_path):
    model = _model(tmp_path)
    fingerprint = weights.fingerprint_model_weights(model)
    manifest = _input_manifest(tmp_path / "inputs.json", model, fingerprint)

    first = preflight._model_provenance(
        str(model),
        input_manifest=str(manifest),
        expected_input_manifest_sha256=_digest(manifest),
    )
    assert first["key"] == "synthetic"
    assert first["weight_verification_mode"] == "full_sha256"
    second = preflight._model_provenance(
        str(model),
        input_manifest=str(manifest),
        expected_input_manifest_sha256=_digest(manifest),
    )
    assert second["weight_verification_mode"] == "full_sha256"

    _write_json(model / "config.json", {"model_type": "Synthetic"})
    with pytest.raises(RuntimeError, match="metadata.*changed"):
        preflight._model_provenance(
            str(model),
            input_manifest=str(manifest),
            expected_input_manifest_sha256=_digest(manifest),
        )


def test_input_manifest_requires_weight_pinned_schema_and_unique_model_path(tmp_path):
    model = _model(tmp_path)
    fingerprint = weights.fingerprint_model_weights(model)
    manifest = _input_manifest(tmp_path / "inputs.json", model, fingerprint)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["schema_version"] = 1
    _write_json(manifest, document)
    with pytest.raises(weights.ModelWeightError, match="schema_version must be 2"):
        weights.load_input_manifest_model(manifest, model)

    manifest = _input_manifest(tmp_path / "inputs2.json", model, fingerprint)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["models"]["duplicate"] = document["models"]["synthetic"]
    _write_json(manifest, document)
    with pytest.raises(weights.ModelWeightError, match="exactly one model"):
        weights.load_input_manifest_model(manifest, model)


def test_weight_index_rejects_nonfinite_json_numbers(tmp_path):
    model = _model(tmp_path)
    index = model / "model.safetensors.index.json"
    index.write_text(
        '{"metadata":{"total_size":NaN},"weight_map":'
        '{"a":"model-00001-of-00002.safetensors",'
        '"b":"model-00002-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(weights.ModelWeightError, match="non-finite JSON number"):
        weights.fingerprint_model_weights(model)


def test_preflight_rejects_manifest_not_matching_submission_digest(tmp_path):
    model = _model(tmp_path)
    manifest = _input_manifest(
        tmp_path / "inputs.json", model, weights.fingerprint_model_weights(model)
    )
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        preflight._model_provenance(
            str(model),
            input_manifest=str(manifest),
            expected_input_manifest_sha256="0" * 64,
        )


def test_worker_rehashes_every_absolute_manifest_file_and_rejects_unpinned_dataset(tmp_path):
    model = _model(tmp_path)
    manifest = _input_manifest(
        tmp_path / "inputs.json", model, weights.fingerprint_model_weights(model)
    )
    manifest_record, pinned, summary = preflight._verify_pinned_input_files(
        str(manifest), expected_input_manifest_sha256=_digest(manifest)
    )
    assert summary["verification_mode"] == "full_sha256"
    assert summary["file_count"] == 3
    assert summary["total_bytes"] > 0

    unpinned = tmp_path / "different-dataset.json"
    _write_json(unpinned, {"rows": []})
    with pytest.raises(RuntimeError, match="not pinned"):
        preflight._dataset_provenance(
            [str(manifest), str(unpinned)],
            input_manifest_record=manifest_record,
            pinned_files=pinned,
        )

    config = model / "config.json"
    config.write_bytes(config.read_bytes().replace(b"synthetic", b"Synthetic"))
    with pytest.raises(RuntimeError, match="changed since validation"):
        preflight._verify_pinned_input_files(
            str(manifest), expected_input_manifest_sha256=_digest(manifest)
        )


def _verify_approved_smoke(campaign: dict) -> dict:
    return preflight._verify_smoke_approval_binding(
        marker=campaign["marker"],
        run_root=campaign["run_root"],
        expected_commit=campaign["commit"],
        input_manifest=campaign["input_manifest"],
        expected_input_manifest_sha256=_digest(campaign["input_manifest"]),
        expected_marker_sha256=campaign["marker_sha256"],
        expected_validated_plan_sha256=campaign["plan_sha256"],
    )


def test_formal_worker_preflight_records_exact_smoke_approval_binding(tmp_path):
    campaign = _approved_smoke(tmp_path)
    binding = _verify_approved_smoke(campaign)
    assert binding["marker"] == _file_record(campaign["marker"])
    assert binding["validated_plan_sha256"] == campaign["plan_sha256"]
    assert binding["code_commit"] == campaign["commit"]
    assert binding["input_manifest"] == _file_record(campaign["input_manifest"])
    assert [row["key"] for row in binding["results"]] == list(preflight.SMOKE_JOB_KEYS)
    assert [row["key"] for row in binding["worker_preflights"]] == list(preflight.SMOKE_JOB_KEYS)


@pytest.mark.parametrize("kind", ["result", "preflight"])
@pytest.mark.parametrize("action", ["delete", "change"])
def test_formal_worker_rejects_changed_or_deleted_approved_smoke_file(tmp_path, kind, action):
    campaign = _approved_smoke(tmp_path)
    target = campaign["results"]["ppl"] if kind == "result" else campaign["preflights"]["ppl"]
    if action == "delete":
        target.unlink()
    else:
        _write_json(target, {"tampered": True})
    with pytest.raises(RuntimeError, match="missing|changed"):
        _verify_approved_smoke(campaign)


def test_formal_worker_rejects_deleted_or_replaced_approval_marker(tmp_path):
    campaign = _approved_smoke(tmp_path / "deleted")
    campaign["marker"].unlink()
    with pytest.raises(RuntimeError, match="missing"):
        _verify_approved_smoke(campaign)

    campaign = _approved_smoke(tmp_path / "replaced")
    replacement = json.loads(campaign["marker"].read_text(encoding="utf-8"))
    replacement["approved_at_utc"] = "2026-07-16T00:00:01+00:00"
    _write_json(campaign["marker"], replacement)
    with pytest.raises(RuntimeError, match="submit-time SHA-256"):
        _verify_approved_smoke(campaign)


def test_formal_worker_rejects_different_validated_plan_binding(tmp_path):
    campaign = _approved_smoke(tmp_path)
    campaign["plan_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="validated plan differs"):
        _verify_approved_smoke(campaign)
