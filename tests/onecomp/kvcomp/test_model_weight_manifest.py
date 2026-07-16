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
