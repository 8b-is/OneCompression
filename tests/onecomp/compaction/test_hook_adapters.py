"""PostToolUse replacement contracts for Claude Code and Codex."""

from onecomp.compaction.adapters.hooks import process_post_tool_use
from onecomp.compaction.store import ObsStore


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
    wal = next(tmp_path.glob("*.wal"))
    restored = ObsStore.from_wal(wal)
    assert "CLAUDE-NEEDLE" in restored.to_serializable()[0]["full_content"]


def test_codex_hook_uses_documented_feedback_replacement_and_is_idempotent(tmp_path):
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
    assert first["decision"] == "block"
    assert "tool completed" in first["reason"]
    assert "onecomp-arc-recall" in first["reason"]
    assert second["reason"] == first["reason"]
    assert ObsStore.from_wal(next(tmp_path.glob("*.wal"))).n_records == 1


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
    assert not list(tmp_path.glob("*.wal"))
