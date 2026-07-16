"""Paper-job provenance must be explicit and merge-safe."""

import json

import pytest

from onecomp.kvcomp.eval.provenance import (
    load_run_provenance,
    merge_run_provenance,
    run_code_commit,
)


def test_no_paper_environment_keeps_library_use_optional(monkeypatch):
    monkeypatch.delenv("PAPER_PROVENANCE_PATH", raising=False)
    monkeypatch.delenv("CODE_REPO_COMMIT", raising=False)
    assert load_run_provenance() is None
    assert run_code_commit() is None


def test_preflight_record_is_hashed_and_deduplicated(tmp_path, monkeypatch):
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"schema_version": 1, "code": {"commit": "abc"}}))
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(path))
    record = load_run_provenance()
    assert record["record"]["code"]["commit"] == "abc"
    assert len(record["sha256"]) == 64
    assert merge_run_provenance([record], record) == [record]
    assert run_code_commit() == "abc"


def test_code_commit_environment_is_used_before_preflight_exists(monkeypatch):
    monkeypatch.delenv("PAPER_PROVENANCE_PATH", raising=False)
    monkeypatch.setenv("CODE_REPO_COMMIT", "deadbeef")
    assert run_code_commit() == "deadbeef"


def test_invalid_preflight_is_fatal_when_variable_is_set(tmp_path, monkeypatch):
    path = tmp_path / "preflight.json"
    path.write_text("{}")
    monkeypatch.setenv("PAPER_PROVENANCE_PATH", str(path))
    with pytest.raises(RuntimeError, match="schema"):
        load_run_provenance()
