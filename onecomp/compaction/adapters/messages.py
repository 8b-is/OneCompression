"""Provider-neutral helpers for agent message payloads.

The ARC core keeps provider objects opaque whenever they remain in the live
tail, but it still needs a textual view of older objects for compaction and
budget accounting.  These helpers cover the shapes used by OpenAI Chat and
Responses, Anthropic Messages / Claude Code, and OpenHands serialised events
without importing any of those SDKs.

Unknown non-text blocks are represented by a short marker instead of their
full JSON encoding.  This avoids accidentally copying base64 image payloads
into an ARC summary.  Recent messages are returned unchanged by the compactor,
so the marker is used only after a block has moved into compacted history.
"""

from __future__ import annotations

import json
from typing import Any

_ASSISTANT_ITEM_TYPES = {
    "assistant",
    "computer_call",
    "custom_tool_call",
    "function_call",
    "local_shell_call",
    "message",
    "reasoning",
    "shell_call",
}
_TOOL_ITEM_TYPES = {
    "computer_call_output",
    "custom_tool_call_output",
    "function_call_output",
    "local_shell_call_output",
    "shell_call_output",
    "tool_result",
}
_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}


def message_role(message: dict[str, Any]) -> str:
    """Return a canonical role for common provider message/item shapes."""
    role = message.get("role")
    if isinstance(role, str):
        canonical = role.lower()
        # Anthropic represents tool results as content blocks inside a
        # user-role message.  Treat those as observations during compaction;
        # otherwise an untagged multi-kilobyte tool result is mistaken for a
        # human follow-up and copied into the summary.
        if canonical == "user" and _contains_tool_result(message.get("content")):
            return "tool"
        return canonical

    item_type = message.get("type") or message.get("kind")
    if not isinstance(item_type, str):
        return ""
    item_type = item_type.lower()
    if item_type in _TOOL_ITEM_TYPES or item_type.endswith("_call_output"):
        return "tool"
    if item_type in _ASSISTANT_ITEM_TYPES or item_type.endswith("_call"):
        return "assistant"
    if item_type in {"user", "message_event", "observation", "observation_event"}:
        return "user"
    if item_type in {"developer", "system", "system_prompt_event"}:
        return "system" if item_type.startswith("system") else "developer"
    return ""


def content_to_text(content: Any) -> str:
    """Extract text from string or provider content-block payloads.

    This function is deliberately total for JSON-like values: malformed or
    unfamiliar blocks cannot crash context-budget enforcement.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, (list, tuple)):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if not isinstance(content, dict):
        return str(content)

    block_type = content.get("type") or content.get("kind")
    if block_type in _TEXT_BLOCK_TYPES and isinstance(content.get("text"), str):
        return content["text"]
    if block_type == "tool_use":
        name = content.get("name", "tool")
        payload = content.get("input", {})
        return f"{name}({_compact_json(payload)})"
    if block_type in _TOOL_ITEM_TYPES or (
        isinstance(block_type, str) and block_type.endswith("_output")
    ):
        for key in ("content", "output", "text"):
            if key in content:
                return content_to_text(content[key])

    for key in ("text", "content", "output", "result", "summary"):
        if key in content:
            text = content_to_text(content[key])
            if text:
                return text

    label = block_type if isinstance(block_type, str) else "non-text"
    return f"[{label} block omitted from compacted text]"


def message_text(message: dict[str, Any]) -> str:
    """Return the model-visible textual projection of one message/item."""
    item_type = message.get("type") or message.get("kind")
    if isinstance(item_type, str) and (
        item_type in _TOOL_ITEM_TYPES or item_type.endswith("_call_output")
    ):
        for key in ("output", "content", "result"):
            if key in message:
                return content_to_text(message[key])

    if isinstance(item_type, str) and (
        item_type in _ASSISTANT_ITEM_TYPES or item_type.endswith("_call")
    ):
        if item_type.endswith("_call") or item_type in {
            "function_call",
            "custom_tool_call",
            "local_shell_call",
            "shell_call",
        }:
            name = message.get("name") or item_type
            arguments = message.get("arguments", message.get("input", ""))
            return f"{name}({_compact_json(arguments)})"

    parts: list[str] = []
    content = content_to_text(message.get("content"))
    if content:
        parts.append(content)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = function.get("name", "tool")
            arguments = function.get("arguments", function.get("input", ""))
            parts.append(f"{name}({_compact_json(arguments)})")
    return "\n".join(parts)


def protected_prefix_count(messages: list[dict[str, Any]]) -> int:
    """Count the immutable prompt prefix for common agent histories.

    Leading system/developer messages and the first user task are protected.
    This yields two messages for the original ARC ReAct layout, one for an
    Anthropic message list whose system prompt is supplied separately, and any
    number of leading developer/system instructions for OpenAI APIs.
    """
    index = 0
    while index < len(messages) and message_role(messages[index]) in {
        "system",
        "developer",
    }:
        index += 1
    if index < len(messages) and message_role(messages[index]) == "user":
        index += 1
    return index


def recent_turn_start(messages: list[dict[str, Any]], keep_recent: int) -> int:
    """Index where the newest ``keep_recent`` assistant turns begin."""
    if keep_recent <= 0:
        return len(messages)
    assistant_indices = [
        index for index, message in enumerate(messages) if message_role(message) == "assistant"
    ]
    if assistant_indices:
        # If a prior ARC summary precedes the only live assistant turn, its
        # index is > 0.  Returning that index permits progressive re-compaction
        # with a tighter cap while still preserving the complete live turn.
        return assistant_indices[max(0, len(assistant_indices) - keep_recent)]
    # Observation-only/event-only histories still get a bounded live tail.
    return max(0, len(messages) - keep_recent)


def total_text_chars(messages: list[dict[str, Any]]) -> int:
    """Character count of the textual projection sent to a model."""
    return sum(len(message_text(message)) for message in messages)


def _compact_json(value: Any, max_chars: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _contains_tool_result(content: Any) -> bool:
    if isinstance(content, dict):
        block_type = content.get("type") or content.get("kind")
        return isinstance(block_type, str) and (
            block_type in _TOOL_ITEM_TYPES or block_type.endswith("_call_output")
        )
    if isinstance(content, (list, tuple)):
        return any(_contains_tool_result(item) for item in content)
    return False


__all__ = [
    "content_to_text",
    "message_role",
    "message_text",
    "protected_prefix_count",
    "recent_turn_start",
    "total_text_chars",
]
