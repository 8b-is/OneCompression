"""Honest wire helpers for a Codex ``PostToolUse`` hook.

Codex supports additional model context and control decisions, but currently
does not support replacing a native tool result. ``context`` therefore adds a
body-free archive reference without claiming prompt reduction. Use
``onecomp-arc run -- COMMAND`` when the original verbose output must never enter
the model-visible transcript.
"""

from __future__ import annotations

from typing import Any, Literal

CodexResultMode = Literal["context", "stop", "block"]


def post_tool_use_output(
    citation: str,
    *,
    mode: CodexResultMode = "context",
) -> dict[str, Any]:
    """Encode archival context or an explicit post-tool control decision."""
    if not isinstance(citation, str) or not citation:
        raise ValueError("citation must be a non-empty string")
    if mode == "context":
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": citation,
            }
        }
    if mode == "stop":
        return {"continue": False, "stopReason": citation}
    if mode == "block":
        return {"decision": "block", "reason": citation}
    raise ValueError("mode must be 'context', 'stop', or 'block'")


__all__ = ["CodexResultMode", "post_tool_use_output"]
