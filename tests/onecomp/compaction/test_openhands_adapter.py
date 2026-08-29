"""Integration checks against the optional, real OpenHands SDK classes."""

import json

import pytest

pytest.importorskip("openhands.sdk")

from openhands.sdk.context.view import View
from openhands.sdk.event import Condensation, MessageEvent
from openhands.sdk.llm import Message, TextContent

from onecomp.compaction.adapters.openhands import (
    OpenHandsArcCondenser,
    OpenHandsArcListAction,
    OpenHandsArcListTool,
    OpenHandsArcRecallAction,
    OpenHandsArcRecallTool,
)


def _message_event(role: str, text: str) -> MessageEvent:
    source = "agent" if role == "assistant" else "user"
    return MessageEvent(
        source=source,
        llm_message=Message(role=role, content=[TextContent(text=text)]),
    )


def test_openhands_condenser_emits_native_condensation_and_shortens_view():
    events = [_message_event("user", "task")]
    for index in range(8):
        events.extend(
            [
                _message_event("assistant", f"action-{index}"),
                _message_event("user", f"followup-{index}"),
            ]
        )
    condenser = OpenHandsArcCondenser(max_size=10, keep_first=1, summary_max_chars=800)
    result = condenser.condense(View(events=events))
    assert isinstance(result, Condensation)
    assert len(result.summary or "") <= 800
    assert len(result.apply(events)) < len(events)
    condenser.close()


def test_openhands_recall_tool_shares_the_condenser_store():
    condenser = OpenHandsArcCondenser(max_size=10, keep_first=1)
    record = condenser.engine.record_observation(
        command="cat artifact", content="OPENHANDS-NEEDLE\n" + "x" * 3000
    )
    # A condenser pass is the OpenHands model-response boundary and resets the
    # per-step recall allowance used by the paired tool.
    condenser.condense(View(events=[]))
    tool = OpenHandsArcRecallTool.create(runtime_id=condenser.runtime_id)[0]
    observation = tool(OpenHandsArcRecallAction(obs_id=record.obs_id))
    assert observation.is_error is False
    assert "OPENHANDS-NEEDLE" in observation.text
    condenser.close()


def test_openhands_index_maps_an_old_command_to_its_recall_id():
    condenser = OpenHandsArcCondenser(max_size=10, keep_first=1)
    wanted = condenser.engine.record_observation(
        command="inspect src/special_target.py", content="wanted"
    )
    condenser.engine.record_observation(command="inspect logs/other.log", content="other")
    tool = OpenHandsArcListTool.create(runtime_id=condenser.runtime_id)[0]
    observation = tool(OpenHandsArcListAction(query="SPECIAL_target"))
    assert observation.is_error is False
    assert wanted.obs_id in observation.text
    assert "logs/other.log" not in observation.text
    condenser.close()


def test_openhands_index_is_newest_first_and_uses_stable_sequence_cursor():
    condenser = OpenHandsArcCondenser(max_size=8)
    records = [
        condenser.engine.record_observation(command=f"inspect {index}", content=str(index))
        for index in range(5)
    ]
    first = json.loads(condenser.list_observations(limit=2))
    second = json.loads(condenser.list_observations(cursor=first["next_cursor"], limit=2))

    assert [row["obs_id"] for row in first["observations"]] == [
        records[4].obs_id,
        records[3].obs_id,
    ]
    assert [row["obs_id"] for row in second["observations"]] == [
        records[2].obs_id,
        records[1].obs_id,
    ]
    condenser.close()
