"""Tests for onecomp.mcp.compaction_tools (ARC over MCP)."""

import io
import json

import pytest

from onecomp.mcp import compaction_tools as ct
from onecomp.mcp.compaction_tools import build_compaction_tools
from onecomp.mcp.protocol import MCPServer
from onecomp.mcp.tools import build_tools


@pytest.fixture(autouse=True)
def _clean_sessions():
    ct.close_all_sessions()
    yield
    ct.close_all_sessions()


def _start() -> str:
    return ct.compaction_start_session()["session_id"]


def test_start_session_returns_id_and_snippet():
    out = ct.compaction_start_session(recall_per_step=3)
    assert len(out["session_id"]) == 12
    assert "_recall \u00a7<obs_id>" in out["system_prompt_snippet"]
    assert "At most 3 recall(s)" in out["system_prompt_snippet"]


def test_sessions_are_isolated():
    sid_a, sid_b = _start(), _start()
    ct.compaction_record_observation(sid_a, command="ls", content="x" * 1000)
    assert ct.compaction_session_stats(sid_a)["n_observations"] == 1
    assert ct.compaction_session_stats(sid_b)["n_observations"] == 0


def test_close_session_frees_and_unknown_id_raises():
    sid = _start()
    assert ct.compaction_close_session(sid) == {"closed": sid}
    with pytest.raises(ValueError, match="Unknown session_id"):
        ct.compaction_session_stats(sid)


def test_session_limit():
    for _ in range(ct._MAX_SESSIONS):
        _start()
    with pytest.raises(ValueError, match="Session limit"):
        _start()


def test_wal_path_is_confined_to_configured_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    sid = ct.compaction_start_session(wal_path="sessions/agent.wal")["session_id"]
    ct.compaction_record_observation(sid, command="echo ok", content="ok")
    ct.compaction_close_session(sid)
    assert (tmp_path / "sessions" / "agent.wal").is_file()

    with pytest.raises(ValueError, match="ONECOMP_MCP_WAL_ROOT"):
        ct.compaction_start_session(wal_path="../escape.wal")


def test_close_all_sessions_flushes_wals(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    sid = ct.compaction_start_session(wal_path="shutdown.wal")["session_id"]
    ct.compaction_record_observation(sid, command="echo bye", content="bye")
    ct.close_all_sessions()
    assert not ct._SESSIONS
    assert "echo bye" in (tmp_path / "shutdown.wal").read_text()


def test_resume_wal_restores_observations_without_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ONECOMP_MCP_WAL_ROOT", str(tmp_path))
    first = ct.compaction_start_session(wal_path="resume.wal")
    rec = ct.compaction_record_observation(
        first["session_id"], command="cat result", content="persisted result"
    )
    ct.compaction_close_session(first["session_id"])

    resumed = ct.compaction_start_session(wal_path="resume.wal", resume_wal=True)
    assert resumed["restored_observations"] == 1
    recalled = ct.compaction_recall(resumed["session_id"], rec["obs_id"])
    assert recalled["found"] is True
    assert "persisted result" in recalled["content"]
    ct.compaction_close_session(resumed["session_id"])
    assert len((tmp_path / "resume.wal").read_text().splitlines()) == 1


def test_resume_wal_requires_a_path():
    with pytest.raises(ValueError, match="requires wal_path"):
        ct.compaction_start_session(resume_wal=True)


def test_record_flags_long_observations():
    sid = _start()
    short = ct.compaction_record_observation(sid, command="echo hi", content="hi")
    long = ct.compaction_record_observation(sid, command="cat big", content="y" * 5000)
    assert short["will_be_cited"] is False
    assert long["will_be_cited"] is True
    assert long["char_count"] == 5000


def test_compact_messages_produces_citations_and_recall_roundtrip():
    sid = _start()
    big = "SECRET=abc123\n" + "noise\n" * 800
    rec = ct.compaction_record_observation(sid, command="cat data.log", content=big)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "```bash\ncat data.log\n```"},
        {"role": "user", "content": f"<output>\n{big}\n</output>", "_obs_id": rec["obs_id"]},
        {"role": "assistant", "content": "```bash\necho done\n```"},
        {"role": "user", "content": "<output>\ndone\n</output>"},
    ]
    out = ct.compaction_compact_messages(sid, messages)
    summary = out["messages"][2]["content"]
    assert big not in summary
    assert f"\u00a7{rec['obs_id']}" in summary
    assert out["total_chars"] < len(big)

    # MCP-native recall brings back the verbatim body.
    recalled = ct.compaction_recall(sid, rec["obs_id"])
    assert recalled["found"] is True
    assert "SECRET=abc123" in recalled["content"]

    meta = ct.compaction_recall(sid, rec["obs_id"], recall_kind="meta")
    assert meta["found"] is True
    assert len(meta["content"]) < 1200


def test_compact_messages_with_char_budget():
    sid = _start()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for i in range(6):
        rec = ct.compaction_record_observation(sid, command=f"cat p{i}", content=f"c{i} " * 800)
        messages.append({"role": "assistant", "content": f"```bash\ncat p{i}\n```"})
        messages.append(
            {"role": "user", "content": f"c{i} " * 800, "_obs_id": rec["obs_id"]}
        )
    out = ct.compaction_compact_messages(sid, messages, max_chars=6000)
    assert out["total_chars"] <= 6000


def test_compact_messages_rejects_unsupported_content_shape():
    sid = _start()
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    with pytest.raises(ValueError, match=r"messages\[0\]\.content"):
        ct.compaction_compact_messages(sid, messages)


def test_handle_action_passthrough_and_errors():
    sid = _start()
    assert ct.compaction_handle_action(sid, "ls -la") == {"handled": False}

    malformed = ct.compaction_handle_action(sid, "_recall deadbeef")
    assert malformed["handled"] is True and "Malformed" in malformed["error"]

    unknown = ct.compaction_recall(sid, "deadbeef")
    assert unknown["found"] is False and "Unknown observation id" in unknown["error"]


def test_recall_kind_validation():
    sid = _start()
    with pytest.raises(ValueError, match="recall_kind must be"):
        ct.compaction_recall(sid, "deadbeef", recall_kind="huge")


def test_verify_value_guard():
    sid = _start()
    ct.compaction_record_observation(
        sid, command="cat log2", content="noise\ntrace-key: TR-REAL42\nnoise"
    )
    assert ct.compaction_verify_value(sid, "TR-REAL42")["seen"] is True
    assert ct.compaction_verify_value(sid, "TR-REAL42", prefix="trace-key: ")["seen"] is True
    missing = ct.compaction_verify_value(sid, "TR-GUESSED", prefix="trace-key: ")
    assert missing["seen"] is False
    assert missing["candidate_obs_ids"]
    assert "Do not guess" in missing["hint"]


def test_recall_budget_enforced_across_calls_and_reset_by_step_boundary():
    sid = ct.compaction_start_session(recall_per_step=1)["session_id"]
    obs_id = ct.compaction_record_observation(sid, command="cat x", content="z" * 900)["obs_id"]

    # 1st recall within budget; 2nd (no step boundary in between) exceeds it.
    assert ct.compaction_recall(sid, obs_id)["found"] is True
    over = ct.compaction_recall(sid, obs_id)
    assert over["found"] is False and "Recall budget exceeded" in over["error"]

    # An explicit step boundary resets the budget...
    ct.compaction_begin_step(sid)
    assert ct.compaction_recall(sid, obs_id)["found"] is True

    # ...and so does recording the next execution.
    ct.compaction_recall(sid, obs_id)  # exhaust again
    ct.compaction_record_observation(sid, command="echo hi", content="hi")
    assert ct.compaction_recall(sid, obs_id)["found"] is True


def test_tools_registered_in_build_tools():
    names = {t.name for t in build_tools()}
    expected = {t.name for t in build_compaction_tools()}
    assert expected <= names
    assert len(expected) == 9


def _rpc(server_input: str) -> list[dict]:
    reader, writer = io.StringIO(server_input), io.StringIO()
    MCPServer(reader, writer, tools=build_compaction_tools()).serve_forever()
    return [json.loads(line) for line in writer.getvalue().splitlines()]


def test_end_to_end_over_jsonrpc_stdio():
    """Full MCP wire-level flow, exactly as an MCP host would drive it."""
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "compaction_start_session", "arguments": {}}},
    ]
    responses = _rpc("\n".join(json.dumps(l) for l in lines) + "\n")
    assert responses[0]["result"]["serverInfo"]["name"] == "onecomp"
    tool_names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert "compaction_recall" in tool_names

    body = json.loads(responses[2]["result"]["content"][0]["text"])
    sid = body["session_id"]

    lines2 = [
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "compaction_record_observation",
                    "arguments": {"session_id": sid, "command": "cat x",
                                  "content": "NEEDLE=42\n" + "n\n" * 600}}},
    ]
    responses2 = _rpc("\n".join(json.dumps(l) for l in lines2) + "\n")
    rec = json.loads(responses2[0]["result"]["content"][0]["text"])
    assert rec["will_be_cited"] is True

    lines3 = [
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "compaction_recall",
                    "arguments": {"session_id": sid, "obs_id": rec["obs_id"]}}},
    ]
    responses3 = _rpc("\n".join(json.dumps(l) for l in lines3) + "\n")
    recalled = json.loads(responses3[0]["result"]["content"][0]["text"])
    assert recalled["found"] is True and "NEEDLE=42" in recalled["content"]
