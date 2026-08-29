"""Tests for onecomp.compaction.compactor."""

from onecomp.compaction.adapters import total_text_chars
from onecomp.compaction.compactor import (
    SUMMARY_HEADER,
    compact_messages,
    compact_messages_arc,
    compact_messages_max,
)
from onecomp.compaction.store import ObsStore


def _assistant(thought: str, cmd: str) -> dict:
    return {"role": "assistant", "content": f"{thought}\n\n```bash\n{cmd}\n```"}


def _observation(body: str, obs_id: str | None = None) -> dict:
    msg = {"role": "user", "content": f"<output>\n{body}\n</output>"}
    if obs_id is not None:
        msg["_obs_id"] = obs_id
    return msg


def _history(n_turns: int, obs_body: str = "ok", store: ObsStore | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the task"},
    ]
    for i in range(n_turns):
        cmd = f"echo step-{i}"
        msgs.append(_assistant(f"thought {i}", cmd))
        if store is not None:
            rec = store.put(cmd_signature=cmd, full_content=obs_body)
            msgs.append(_observation(obs_body, rec.obs_id))
        else:
            msgs.append(_observation(obs_body))
    return msgs


def test_no_compaction_when_history_is_short():
    msgs = _history(2)
    assert compact_messages(msgs, keep_recent=5) == msgs


def test_baseline_keeps_short_observations_and_drops_long():
    short, long = "short output", "X" * 2000
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        _assistant("t1", "echo a"),
        _observation(short),
        _assistant("t2", "cat big"),
        _observation(long),
        _assistant("t3", "echo b"),
        _observation("recent"),
    ]
    out = compact_messages(msgs, keep_recent=1)
    summary = out[2]["content"]
    assert summary.startswith(SUMMARY_HEADER)
    assert short in summary
    assert long not in summary  # L3: long observation dropped
    assert out[-1]["content"] == "<output>\nrecent\n</output>"


def test_max_drops_prior_summary_but_base_inherits():
    prior = {"role": "user", "content": SUMMARY_HEADER + "\n\nPRIOR-BODY"}
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        prior,
        _assistant("t", "echo x"),
        _observation("obs-x"),
        _assistant("t2", "echo y"),
        _observation("recent"),
    ]
    inherited = compact_messages(msgs, keep_recent=1)[2]["content"]
    dropped = compact_messages_max(msgs, keep_recent=1)[2]["content"]
    assert "PRIOR-BODY" in inherited
    assert "PRIOR-BODY" not in dropped  # L4


def test_arc_replaces_long_observations_with_citations():
    store = ObsStore()
    long_body = "line\n" * 500  # 2500 chars > RESULT_MAX_CHARS
    msgs = _history(3, obs_body=long_body, store=store)
    out = compact_messages_arc(msgs, obs_store=store, keep_recent=1)
    summary = out[2]["content"]
    assert long_body not in summary
    # Two compacted turns -> both citations present (dedup by content:
    # identical body+cmd would collide, but cmds differ per turn).
    cited = [oid for oid in store.known_ids() if f"\u00a7{oid}" in summary]
    assert len(cited) == 2
    assert "_recall \u00a7" in summary


def test_arc_keeps_short_observations_verbatim_with_id():
    store = ObsStore()
    msgs = _history(3, obs_body="tiny", store=store)
    out = compact_messages_arc(msgs, obs_store=store, keep_recent=1)
    summary = out[2]["content"]
    assert "result: tiny" in summary


def test_arc_inherits_prior_summary_and_dedups_ids():
    store = ObsStore()
    rec = store.put(cmd_signature="cat big", full_content="Z" * 5000)
    prior = {
        "role": "user",
        "content": SUMMARY_HEADER + f"\n\n\u00a7{rec.obs_id}  cmd='cat big'  rc=0 old citation",
    }
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        prior,
        _observation("Z" * 5000, rec.obs_id),  # same obs re-observed
        _assistant("t", "echo y"),
        _observation("recent"),
    ]
    out = compact_messages_arc(msgs, obs_store=store, keep_recent=1)
    summary = out[2]["content"]
    # Prior summary body inherited...
    assert "old citation" in summary
    # ...and the id it referenced is NOT cited a second time.
    assert summary.count(f"\u00a7{rec.obs_id}") == 1


def test_arc_degenerates_to_max_without_store_except_summary_inheritance():
    msgs = _history(4, obs_body="X" * 2000)  # long, no store, no _obs_id
    arc_out = compact_messages_arc(msgs, obs_store=None, keep_recent=1)
    max_out = compact_messages_max(msgs, keep_recent=1)
    assert arc_out[2]["content"] == max_out[2]["content"]


def test_arc_summary_budget_collapses_oldest_to_id_only():
    store = ObsStore()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
    ]
    ids = []
    for i in range(6):
        body = f"data-{i}\n" * 200  # ~1400 chars each, unique per turn
        cmd = f"cat part-{i}"
        rec = store.put(cmd_signature=cmd, full_content=body + str(i))
        ids.append(rec.obs_id)
        msgs.append(_assistant(f"t{i}", cmd))
        msgs.append(_observation(body + str(i), rec.obs_id))
    msgs.append(_assistant("t-final", "echo done"))
    msgs.append(_observation("recent"))

    out = compact_messages_arc(msgs, obs_store=store, keep_recent=1, summary_max_chars=1200)
    summary = out[2]["content"]
    # Budget forces the oldest citations to collapse to bare ids...
    assert "elided to keep this summary within the context budget" in summary
    # ...but every id remains recall-able (present in some form).
    for oid in ids:
        assert f"\u00a7{oid}" in summary
    # And the assembled summary respects the order of magnitude of the cap.
    assert len(summary) < 6 * 1600


def test_arc_unbounded_by_default():
    store = ObsStore()
    msgs = _history(5, obs_body="Y" * 3000, store=store)
    out = compact_messages_arc(msgs, obs_store=store, keep_recent=1)
    assert "elided to keep this summary" not in out[2]["content"]


def test_arc_summary_budget_is_a_strict_total_cap_at_scale():
    store = ObsStore()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
    ]
    for i in range(300):
        body = (f"item-{i} " * 100) + str(i)
        rec = store.put(cmd_signature=f"cat item-{i}", full_content=body)
        msgs.extend([_assistant("t", f"cat item-{i}"), _observation(body, rec.obs_id)])
    msgs.extend([_assistant("recent", "echo done"), _observation("done")])

    summary = compact_messages_arc(msgs, obs_store=store, keep_recent=1, summary_max_chars=500)[2][
        "content"
    ]
    assert len(summary) <= 500
    assert "older handle(s) could not fit" in summary


def test_summary_budget_assembles_only_the_final_linear_size_candidate(monkeypatch):
    from onecomp.compaction import compactor  # pylint: disable=import-outside-toplevel

    calls = 0
    original = compactor._assemble_summary

    def counted(parts, *, collapsed, omitted_ids=0):
        nonlocal calls
        calls += 1
        return original(parts, collapsed=collapsed, omitted_ids=omitted_ids)

    monkeypatch.setattr(compactor, "_assemble_summary", counted)
    entries = [
        {"full": f"entry-{index}-" + "x" * 40, "obs_ids": [f"{index:016x}"]}
        for index in range(20_000)
    ]
    summary, collapsed = compactor._apply_summary_budget(entries, 800_000)
    assert collapsed
    assert len(summary) <= 800_000
    assert calls == 1


def test_tiny_summary_budget_counts_only_omitted_observation_handles():
    store = ObsStore()
    rec = store.put(cmd_signature="cat data", full_content="z" * 2000)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        _assistant("assistant-only entry", "echo old"),
        _observation("z" * 2000, rec.obs_id),
        _assistant("recent", "echo recent"),
        _observation("ok"),
    ]

    summary = compact_messages_arc(
        messages,
        obs_store=store,
        keep_recent=1,
        summary_max_chars=260,
    )[2]["content"]
    assert len(summary) <= 260
    # Demoting assistant text must not be reported as an omitted handle.
    assert "2 older handle(s)" not in summary


def test_arc_accepts_anthropic_content_blocks_without_flattening_live_tail():
    store = ObsStore()
    body = "ANTHROPIC-NEEDLE\n" + "x" * 3000
    rec = store.put(cmd_signature="Read large.log", full_content=body)
    recent_blocks = [{"type": "text", "text": "recent answer"}]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": body}],
            "_obs_id": rec.obs_id,
        },
        {"role": "assistant", "content": recent_blocks},
        {"role": "user", "content": [{"type": "text", "text": "recent tool result"}]},
    ]
    out = compact_messages_arc(messages, obs_store=store, keep_recent=1)
    assert f"§{rec.obs_id}" in out[1]["content"]
    assert body not in out[1]["content"]
    assert out[-2]["content"] is recent_blocks


def test_arc_preserves_human_followup_but_treats_anthropic_tool_result_as_tool():
    messages = [
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "Please also preserve THIS-REQUIREMENT"},
        {"role": "assistant", "content": "tool call"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 5000}],
        },
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent result"},
    ]
    out = compact_messages_arc(messages, keep_recent=1, summary_max_chars=2000)
    summary = out[1]["content"]
    assert "THIS-REQUIREMENT" in summary
    assert "x" * 1000 not in summary


def test_arc_preserves_mid_history_developer_instruction():
    messages = [
        {"role": "developer", "content": "initial policy"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old action"},
        {"role": "developer", "content": "POLICY-UPDATE: never remove audit logs"},
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent result"},
    ]
    out = compact_messages_arc(messages, keep_recent=1)
    assert "POLICY-UPDATE" in out[2]["content"]


def test_arc_accepts_openai_responses_function_items():
    store = ObsStore()
    body = "RESPONSES-NEEDLE\n" + "y" * 3000
    rec = store.put(cmd_signature="shell ls", full_content=body)
    messages = [
        {"role": "developer", "content": "instructions"},
        {"role": "user", "content": "task"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": body,
            "_obs_id": rec.obs_id,
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "recent"}],
        },
        {"role": "user", "content": "recent observation"},
    ]
    out = compact_messages_arc(messages, obs_store=store, keep_recent=1)
    assert out[:2] == messages[:2]
    assert f"§{rec.obs_id}" in out[2]["content"]
    assert total_text_chars(out) < total_text_chars(messages)
