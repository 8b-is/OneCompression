"""PostToolUse replacement contracts for Claude Code and Codex."""

import pytest

from onecomp.compaction.adapters.hooks import process_post_tool_use, recall_main
from onecomp.compaction.formal import SQLiteObservationArchive


def test_claude_bash_hook_replaces_long_output_with_schema_compatible_citation(tmp_path):
    payload = {
        "session_id": "claude-session",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "cat large.log"},
        "tool_response": {
            "stdout": "CLAUDE-NEEDLE\n" + "x" * 5000,
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        },
        "tool_use_id": "toolu_1",
    }
    out = process_post_tool_use(payload, platform="claude", hook_root=tmp_path)
    replacement = out["hookSpecificOutput"]["updatedToolOutput"]
    assert set(replacement) == set(payload["tool_response"])
    assert "CLAUDE-NEEDLE" in replacement["stdout"]
    assert len(replacement["stdout"]) < 1200
    archive_path = next(tmp_path.glob("*.sqlite3"))
    with SQLiteObservationArchive(archive_path) as restored:
        assert "CLAUDE-NEEDLE" in restored.get(restored.known_ids()[0]).full_content


def test_codex_hook_uses_documented_archival_context_and_is_idempotent(tmp_path):
    payload = {
        "session_id": "codex-session",
        "turn_id": "turn-1",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "rg needle ."},
        "tool_response": "CODEX-NEEDLE\n" + "y" * 5000,
        "tool_use_id": "call_1",
    }
    first = process_post_tool_use(payload, platform="codex", hook_root=tmp_path)
    second = process_post_tool_use(payload, platform="codex", hook_root=tmp_path)
    context = first["hookSpecificOutput"]["additionalContext"]
    assert "Tool completed" in context
    assert "onecomp-arc-recall" in context
    assert second == first
    with SQLiteObservationArchive(next(tmp_path.glob("*.sqlite3"))) as archive:
        assert archive.n_records == 1


def test_hook_passes_through_short_and_recall_outputs(tmp_path):
    short = {
        "session_id": "s",
        "tool_name": "Bash",
        "tool_input": {"command": "echo ok"},
        "tool_response": "ok",
    }
    assert process_post_tool_use(short, platform="codex", hook_root=tmp_path) == {}
    recall = {
        **short,
        "tool_input": {"command": "onecomp-arc-recall --session-id s --obs-id deadbeef"},
        "tool_response": "z" * 5000,
    }
    assert process_post_tool_use(recall, platform="codex", hook_root=tmp_path) == {}
    assert not list(tmp_path.glob("*.sqlite3"))


def test_hook_does_not_change_existing_root_permissions(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    root.chmod(0o755)
    payload = {
        "session_id": "permission-session",
        "tool_name": "Bash",
        "tool_input": {"command": "cat large.log"},
        "tool_response": "x" * 5000,
        "tool_use_id": "permission-call",
    }

    process_post_tool_use(payload, platform="codex", hook_root=root)

    assert root.stat().st_mode & 0o777 == 0o755


def test_recall_cli_does_not_create_an_archive_for_an_unknown_session(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ONECOMP_ARC_HOOK_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as raised:
        recall_main(["--session-id", "missing", "--obs-id", "deadbeef"])
    assert raised.value.code == 2
    assert "does not exist" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.sqlite3"))
