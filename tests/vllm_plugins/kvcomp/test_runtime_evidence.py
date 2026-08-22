"""CPU tests for the opt-in vLLM worker evidence channel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_plugins.kvcomp.runtime_evidence import (
    RUNTIME_EVIDENCE_ENV,
    append_runtime_evidence,
    emit_runtime_evidence_once,
)


def test_runtime_evidence_is_disabled_without_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv(RUNTIME_EVIDENCE_ENV, raising=False)
    append_runtime_evidence({"event": "unused"})
    assert not list(tmp_path.iterdir())


def test_runtime_evidence_appends_bounded_jsonl(tmp_path, monkeypatch):
    output = tmp_path / "evidence.jsonl"
    monkeypatch.setenv(RUNTIME_EVIDENCE_ENV, str(output.resolve()))
    append_runtime_evidence({"event": "packed_hq_store_launched", "shape": [1, 2, 3]})
    append_runtime_evidence({"event": "packed_hq_decode_attention_launched"})
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == [
        "packed_hq_store_launched",
        "packed_hq_decode_attention_launched",
    ]
    assert all(
        row["schema"] == {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1}
        for row in rows
    )
    assert output.stat().st_mode & 0o777 == 0o600


def test_runtime_evidence_once_marks_only_after_success(tmp_path, monkeypatch):
    class Owner:
        pass

    owner = Owner()
    relative = Path("relative-evidence.jsonl")
    monkeypatch.setenv(RUNTIME_EVIDENCE_ENV, str(relative))
    with pytest.raises(RuntimeError, match="absolute"):
        emit_runtime_evidence_once(owner, "store", {"event": "store"})

    output = tmp_path / "evidence.jsonl"
    monkeypatch.setenv(RUNTIME_EVIDENCE_ENV, str(output.resolve()))
    emit_runtime_evidence_once(owner, "store", {"event": "store"})
    emit_runtime_evidence_once(owner, "store", {"event": "duplicate"})
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["event"] == "store"


def test_runtime_evidence_rejects_non_json_and_oversized_records(tmp_path, monkeypatch):
    output = tmp_path / "evidence.jsonl"
    monkeypatch.setenv(RUNTIME_EVIDENCE_ENV, str(output.resolve()))
    with pytest.raises(RuntimeError, match="JSON-native"):
        append_runtime_evidence({"event": object()})
    with pytest.raises(RuntimeError, match="exceeds"):
        append_runtime_evidence({"event": "x" * (17 * 1024)})
    assert not output.exists()
