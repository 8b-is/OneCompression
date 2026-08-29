"""OneComp-style compaction configuration and runner tests."""

from pathlib import Path

import pytest

from onecomp.compaction import (
    CompactionConfig,
    CompactionRunner,
    InvalidAction,
    RecallRequiresChunking,
    sanitize_messages,
)


def test_config_is_side_effect_free_serializable_and_has_explicit_profiles(tmp_path):
    root = tmp_path / "run"
    config = CompactionConfig.persistent(root, max_recall_units=64)
    assert not root.exists()
    assert config.profile == "formal-v1"
    assert config.to_dict()["archive_path"] == str(root / "archive.sqlite3")

    experimental = CompactionConfig.experimental(tmp_path / "legacy.jsonl")
    assert experimental.profile == "experimental-v1"
    assert experimental.archive_path == tmp_path / "legacy.jsonl"


def test_formal_runner_records_occurrences_compacts_and_recovers_exactly(tmp_path):
    config = CompactionConfig.persistent(
        tmp_path / "run",
        keep_recent=1,
        result_max_chars=10,
        max_recall_units=4096,
    )
    with CompactionRunner(config) as runner:
        runner.begin_step()
        first = runner.record_observation(
            command="cat output.log",
            content="identical long output",
            source_key="tool-call-1",
        )
        second = runner.record_observation(
            command="cat output.log",
            content="identical long output",
            source_key="tool-call-2",
        )
        assert first.obs_id != second.obs_id

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "```bash\ncat output.log\n```"},
            runner.make_observation_message(first),
            {"role": "assistant", "content": "```bash\necho next\n```"},
            runner.make_observation_message(second),
            {"role": "assistant", "content": "recent thought"},
            {"role": "user", "content": "recent result"},
        ]
        compacted = runner.compact(messages)
        summary = compacted[2]["content"]
        assert f"§{first.obs_id}" in summary
        assert f"§{second.obs_id}" in summary
        assert first.content_sha256 in summary
        assert "Use _catalog" in summary

        action_messages = []
        response = runner.handle_action(f"_recall §{first.obs_id}", action_messages)
        assert response is not None
        assert response.body == first.full_content
        assert action_messages[-1]["content"] == first.full_content
        sanitized = sanitize_messages(action_messages)
        assert sanitized == [{"role": "user", "content": first.full_content}]

        assert runner.handle_action("pytest -q", action_messages) is None
        with pytest.raises(InvalidAction):
            runner.handle_action(f"_recall §{first.obs_id} # unsafe suffix")
        status = runner.status()
        assert status["profile"] == "formal-v1"
        assert status["archive"]["observations"] == 2
        assert status["archive"]["recall_count"] == 1

    archive_path = Path(config.archive_path)
    assert archive_path.is_file()


def test_runner_hard_recall_limit_requires_chunking(tmp_path):
    config = CompactionConfig.persistent(tmp_path / "run", max_recall_units=8)
    with CompactionRunner(config) as runner:
        record = runner.record_observation(
            command="cmd",
            content="0123456789abcdef",
            source_key="call",
        )
        with pytest.raises(RecallRequiresChunking):
            runner.handle_action(f"_recall §{record.obs_id}")
        chunk = runner.handle_action(f"_recall_chunk §{record.obs_id} 1")
        assert chunk is not None
        assert chunk.body == "89abcdef"


def test_formal_citation_respects_asymmetric_preview_limits_without_overlap(tmp_path):
    config = CompactionConfig.persistent(
        tmp_path / "run",
        keep_recent=1,
        result_max_chars=4,
        citation_head_chars=0,
        citation_tail_chars=5,
    )
    with CompactionRunner(config) as runner:
        record = runner.record_observation(command="cmd", content="0123456789")
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "run"},
            runner.make_observation_message(record),
            {"role": "assistant", "content": "recent"},
            {"role": "user", "content": "tail"},
        ]
        summary = runner.compact(messages)[2]["content"]

    assert "…\n56789" in summary
    assert "01234" not in summary
    assert summary.count("56789") == 1


def test_runner_provider_guard_uses_configured_final_envelope():
    config = CompactionConfig(
        context_window=20,
        completion_reserve=5,
        serialization_margin=2,
    )
    with CompactionRunner(config) as runner:
        guard = runner.create_budget_guard(lambda request: len(request["serialized_request"]))
        assert (
            guard.dispatch(
                {"serialized_request": "x" * 13},
                lambda request: request["serialized_request"],
            )
            == "x" * 13
        )


def test_experimental_profile_remains_available_without_formal_semantics(tmp_path):
    config = CompactionConfig.experimental(
        tmp_path / "legacy.jsonl",
        max_recall_units=100,
    )
    with CompactionRunner(config) as runner:
        runner.begin_step()
        first = runner.record_observation(command="cmd", content="same")
        second = runner.record_observation(command="cmd", content="same")
        assert first.obs_id == second.obs_id
        assert runner.profile == "experimental-v1"
