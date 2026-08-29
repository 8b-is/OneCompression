"""Provider-boundary dispatch coverage and real transcript replacement."""

import asyncio

import pytest

from onecomp.compaction import CompactionConfig, CompactionRunner
from onecomp.compaction.adapters import GuardedProviderAdapter
from onecomp.compaction.formal import BudgetExceeded
from onecomp.compaction.adapters.messages import message_text


def _count(request):
    return sum(len(message_text(message)) for message in request["messages"])


def _runner(**overrides):
    return CompactionRunner(
        CompactionConfig(
            context_window=1200,
            completion_reserve=100,
            serialization_margin=10,
            **overrides,
        )
    )


def test_dispatch_compacts_owned_transcript_and_checks_final_request():
    with _runner(result_max_chars=80, summary_max_chars=400) as runner:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        for index in range(4):
            record = runner.record_observation(
                command=f"tool-{index}",
                content=f"needle-{index}\n" + "x" * 700,
                source_key=f"call-{index}",
            )
            messages.extend(
                [
                    {"role": "assistant", "content": f"tool-{index}"},
                    runner.make_observation_message(record),
                ]
            )
        request = {"model": "test", "messages": messages}
        seen = []
        adapter = GuardedProviderAdapter(runner, _count)
        result = adapter.dispatch(
            request,
            lambda outbound: seen.append(outbound) or "ok",
            compact_to_chars=1000,
        )
        assert result == "ok"
        assert len(messages) < 10  # host-owned history was actually replaced
        assert all("_obs_id" not in item for item in seen[0]["messages"])
        assert _count(seen[0]) <= 1000
        audit = adapter.audit()
        assert audit.checked_requests == audit.dispatched_requests == 1
        assert audit.rejected_requests == 0


def test_oversized_final_serialization_never_reaches_provider():
    with _runner() as runner:
        called = False

        def invoke(_request):
            nonlocal called
            called = True

        adapter = GuardedProviderAdapter(runner, _count)
        request = {"messages": [{"role": "user", "content": "x" * 2000}]}
        with pytest.raises(BudgetExceeded):
            adapter.dispatch(request, invoke)
        assert called is False
        audit = adapter.audit()
        assert audit.checked_requests == 1
        assert audit.dispatched_requests == 0
        assert audit.rejected_requests == 1


def test_async_dispatch_uses_the_same_guard():
    async def scenario():
        with _runner() as runner:
            adapter = GuardedProviderAdapter(runner, _count)

            async def invoke(request):
                await asyncio.sleep(0)
                return request["messages"][0]["content"]

            assert (
                await adapter.dispatch_async(
                    {"messages": [{"role": "user", "content": "ok"}]}, invoke
                )
                == "ok"
            )
            assert adapter.audit().dispatched_requests == 1

    asyncio.run(scenario())
