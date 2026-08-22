"""Paper-job provenance must be explicit and merge-safe."""

import hashlib
import json
import os

import pytest

from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    merge_run_provenance,
    run_code_commit,
    run_code_revision,
    validate_model_provenance,
)


def test_no_paper_environment_keeps_library_use_optional(monkeypatch):
    monkeypatch.delenv("PAPER_PROVENANCE_PATH", raising=False)
    monkeypatch.delenv("PAPER_PROVENANCE_SHA256", raising=False)
    monkeypatch.delenv("CODE_REPO_COMMIT", raising=False)
    monkeypatch.delenv("CODE_REPO_REVISION", raising=False)
    assert load_run_provenance() is None
    assert run_code_commit() is None
    assert run_code_revision() is None


def test_preflight_record_is_hashed_and_deduplicated(tmp_path, monkeypatch):
    path = tmp_path / "preflight.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": {"commit": "abc", "revision": "abc+dirty.1234"},
            }
        )
    )
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(path))
    record = load_run_provenance()
    assert record["record"]["code"]["commit"] == "abc"
    assert len(record["sha256"]) == 64
    assert merge_run_provenance([record], record) == [record]
    assert run_code_commit() == "abc"
    assert run_code_revision() == "abc+dirty.1234"


def test_code_commit_environment_is_used_before_preflight_exists(monkeypatch):
    monkeypatch.delenv("PAPER_PROVENANCE_PATH", raising=False)
    monkeypatch.setenv("CODE_REPO_COMMIT", "deadbeef")
    monkeypatch.setenv("CODE_REPO_REVISION", "deadbeef+dirty.cafe")
    assert run_code_commit() == "deadbeef"
    assert run_code_revision() == "deadbeef+dirty.cafe"


def test_invalid_preflight_is_fatal_when_variable_is_set(tmp_path, monkeypatch):
    path = tmp_path / "preflight.json"
    path.write_text("{}")
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(path))
    with pytest.raises(RuntimeError, match="schema"):
        load_run_provenance()


def test_preflight_expected_hash_detects_post_plan_mutation(tmp_path, monkeypatch):
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"schema_version": 1, "code": {"commit": "abc"}}))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(path))
    monkeypatch.setenv("PAPER_PROVENANCE_SHA256", expected)
    assert load_run_provenance()["sha256"] == expected

    path.write_text(json.dumps({"schema_version": 1, "code": {"commit": "changed"}}))
    with pytest.raises(RuntimeError, match="changed after planning"):
        load_run_provenance()


def test_model_provenance_checks_metadata_content_and_weight_stat(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    weight = model / "model.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weight.write_bytes(b"weights")

    def artifact(path, verify):
        file_stat = path.stat()
        return {
            "name": path.name,
            "bytes": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verify_on_launch": verify,
        }

    provenance = tmp_path / "source_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": {"commit": "abc"},
                "models": [
                    {
                        "path": str(model),
                        "checkpoint": {
                            "artifacts": [artifact(config, True), artifact(weight, False)]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(provenance))
    assert validate_model_provenance(model)["path"] == str(model)

    config.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="content changed"):
        validate_model_provenance(model)

    config.write_text("{}\n", encoding="utf-8")
    weight_stat = weight.stat()
    os.utime(
        weight,
        ns=(weight_stat.st_atime_ns, weight_stat.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(RuntimeError, match="weight changed"):
        validate_model_provenance(model)
