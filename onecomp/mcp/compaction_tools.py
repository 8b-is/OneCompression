"""MCP tools exposing the ARC compaction engine to agent platforms.

Lets any MCP-capable agent framework (OpenHands, Claude Desktop, custom
MAS orchestrators, ...) use :class:`onecomp.compaction.ArcEngine`
over stdio without importing onecomp.  Two usage styles are supported:

*Framework-driven* (a condenser/memory hook owns the message buffer):
    ``compaction_start_session`` once per agent, then
    ``compaction_record_observation`` after every action execution and
    ``compaction_compact_messages`` whenever the history must shrink.
    The returned buffer contains bounded citations instead of dropped
    observations.

*LLM-driven recall* (the platform exposes MCP tools to the LLM itself):
    the LLM calls ``compaction_recall`` directly with an ``obs_id`` it
    sees in a citation — the MCP-native equivalent of emitting the
    ``_recall §<id>`` magic action.  ``compaction_handle_action`` is the
    text-protocol variant for frameworks that intercept raw actions.

Sessions are independent :class:`~onecomp.compaction.engine.ArcEngine`
instances keyed by an opaque id, so one server process can serve many
concurrent agents of a multi-agent system.  Everything here is pure
Python (no torch, no network); tools are safe on CPU-only hosts.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..compaction import ArcEngine, ArcEngineConfig, RecallError
from .protocol import ToolDefinition

#: Live sessions of this server process: session_id -> ArcEngine.
_SESSIONS: Dict[str, ArcEngine] = {}

#: Guard against unbounded growth if a client keeps opening sessions.
_MAX_SESSIONS = 256


def _get_engine(session_id: str) -> ArcEngine:
    engine = _SESSIONS.get(session_id)
    if engine is None:
        raise ValueError(
            f"Unknown session_id: {session_id!r}. "
            f"Call compaction_start_session first."
        )
    return engine


def _resolve_wal_path(wal_path: Optional[str]) -> Optional[str]:
    """Resolve an MCP-requested WAL beneath the configured safe root.

    MCP tools are callable by the model, so accepting an unrestricted path
    would let a prompt append JSON to arbitrary files writable by the server.
    Relative paths are anchored at ``ONECOMP_MCP_WAL_ROOT`` (default: the
    server working directory), and absolute paths must remain under it.
    """
    if wal_path is None:
        return None
    if not isinstance(wal_path, str) or not wal_path.strip():
        raise ValueError("wal_path must be a non-empty string when provided")
    root = Path(os.environ.get("ONECOMP_MCP_WAL_ROOT", os.getcwd())).expanduser().resolve()
    requested = Path(wal_path).expanduser()
    resolved = (requested if requested.is_absolute() else root / requested).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"wal_path must stay within ONECOMP_MCP_WAL_ROOT ({root})"
        ) from exc
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"wal_path exists but is not a regular file: {resolved}")
    return str(resolved)


def close_all_sessions() -> None:
    """Close every live engine and clear the process-local session registry."""
    for engine in list(_SESSIONS.values()):
        engine.close()
    _SESSIONS.clear()


def compaction_start_session(
    keep_recent: int = 1,
    result_max_chars: int = 500,
    summary_max_chars: int = 24000,
    recall_per_step: int = 2,
    recall_budget_chars: int = 200000,
    recall_max_chars: int = 0,
    wal_path: Optional[str] = None,
    resume_wal: bool = False,
) -> Dict[str, Any]:
    """Create an ARC compaction session and return its id.

    Args:
        keep_recent: Recent turn pairs kept uncompacted.
        result_max_chars: Observations at most this long stay verbatim
            in the summary; longer ones become citations.
        summary_max_chars: Character cap on the compaction summary
            (0 = unbounded).
        recall_per_step: Max recalls served per step.
        recall_budget_chars: Total verbatim-recall chars kept in context
            before LRU eviction (0 = no eviction).
        recall_max_chars: Window cap on a single full recall
            (0 = lossless).
        wal_path: Optional server-side write-ahead log for crash-safe
            observation persistence.
        resume_wal: Restore observations already present in ``wal_path``
            before accepting new records.

    Returns:
        ``session_id`` plus ``system_prompt_snippet`` to append to the
        agent's system prompt.
    """
    if len(_SESSIONS) >= _MAX_SESSIONS:
        raise ValueError(
            f"Session limit reached ({_MAX_SESSIONS}). "
            f"Close unused sessions with compaction_close_session."
        )
    if not isinstance(resume_wal, bool):
        raise ValueError("resume_wal must be a boolean")
    resolved_wal_path = _resolve_wal_path(wal_path)
    if resume_wal and resolved_wal_path is None:
        raise ValueError("resume_wal=true requires wal_path")
    config = ArcEngineConfig(
        keep_recent=keep_recent,
        result_max_chars=result_max_chars,
        summary_max_chars=summary_max_chars,
        recall_per_step=recall_per_step,
        recall_budget_chars=recall_budget_chars,
        recall_max_chars=recall_max_chars,
        wal_path=resolved_wal_path,
    )
    engine = (
        ArcEngine.from_wal(resolved_wal_path, config)
        if resume_wal and resolved_wal_path is not None
        else ArcEngine(config)
    )
    session_id = uuid4().hex[:12]
    while session_id in _SESSIONS:  # defensive against an extremely unlikely collision
        session_id = uuid4().hex[:12]
    _SESSIONS[session_id] = engine
    return {
        "session_id": session_id,
        "system_prompt_snippet": engine.system_prompt_snippet(),
        "restored_observations": engine.store.n_records if resume_wal else 0,
    }


def compaction_close_session(session_id: str) -> Dict[str, Any]:
    """Close a session, flush its WAL, and free its store."""
    engine = _get_engine(session_id)
    engine.close()
    del _SESSIONS[session_id]
    return {"closed": session_id}


def compaction_begin_step(session_id: str) -> Dict[str, Any]:
    """Mark a step boundary: advance the step counter and reset the
    per-step recall budget.

    Call once per LLM response when the integration wants precise
    ``recall_per_step`` semantics.  ``compaction_record_observation``
    also marks a boundary (an execution implies a new step), so
    integrations that record every execution can skip this tool.
    """
    engine = _get_engine(session_id)
    return {"step": engine.begin_step()}


def compaction_record_observation(
    session_id: str,
    command: str,
    content: str,
    returncode: int = 0,
) -> Dict[str, Any]:
    """Store an executed action's output verbatim in the session store.

    Call after every environment execution.  The returned ``obs_id`` is
    the stable handle citations and recalls use.  Recording also marks a
    step boundary (resetting the per-step recall budget), since an
    execution means the previous LLM response is complete.

    Args:
        session_id: Session from ``compaction_start_session``.
        command: The action that produced the observation.
        content: Raw observation body (stored losslessly).
        returncode: Exit status, if applicable.
    """
    if not isinstance(command, str):
        raise ValueError("command must be a string")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise ValueError("returncode must be an integer")
    engine = _get_engine(session_id)
    engine.begin_step()
    rec = engine.record_observation(command=command, content=content, returncode=returncode)
    return {
        "obs_id": rec.obs_id,
        "char_count": rec.char_count,
        "line_count": rec.line_count,
        "will_be_cited": rec.char_count > engine.config.result_max_chars,
    }


def compaction_compact_messages(
    session_id: str,
    messages: List[Dict[str, Any]],
    max_chars: int = 0,
) -> Dict[str, Any]:
    """Compact a message buffer with the session's citation-rich operator.

    Observation messages must carry the ``_obs_id`` returned by
    ``compaction_record_observation`` for their bodies to be preserved
    as recall-able citations; untagged long observations fall back to
    the legacy drop behaviour.

    Args:
        session_id: Session from ``compaction_start_session``.
        messages: OpenAI-style message list (``role``/``content``; the
            first two entries — system and task — are kept intact).
        max_chars: When > 0, shrink until the total content length fits
            this budget (proactive mode).  When 0, apply one reactive
            compaction pass.

    Returns:
        ``messages`` (the compacted buffer) and ``total_chars``.
    """
    # The JSON schema is not enforced by the server; validate the shape
    # here (system boundary) so a malformed client payload gets a clear
    # error instead of an AttributeError from deep inside the compactor.
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a list of objects ({role, content, ...})")
    for index, message in enumerate(messages):
        if not isinstance(message.get("role"), str):
            raise ValueError(f"messages[{index}].role must be a string")
        if message.get("content") is not None and not isinstance(message.get("content"), str):
            raise ValueError(f"messages[{index}].content must be a string or null")
        if "_obs_id" in message and not isinstance(message["_obs_id"], str):
            raise ValueError(f"messages[{index}]._obs_id must be a string")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    engine = _get_engine(session_id)
    if max_chars > 0:
        out = engine.enforce_char_budget(messages, max_chars)
    else:
        out = engine.compact(messages)
    return {
        "messages": out,
        "total_chars": sum(len(m.get("content", "") or "") for m in out),
        "n_messages": len(out),
    }


def compaction_handle_action(session_id: str, action: str) -> Dict[str, Any]:
    """Intercept a raw LLM action; serve it if it is a recall.

    Text-protocol variant for frameworks that pass the LLM's action
    string through before executing it.

    The ``recall_per_step`` budget is charged across consecutive recalls
    and resets at the next step boundary (``compaction_begin_step`` or
    ``compaction_record_observation``) — this tool deliberately does NOT
    reset it, otherwise the budget could never trigger over MCP.

    Returns:
        ``handled=False`` when the action is not a recall (execute it
        normally).  ``handled=True`` with ``content`` (inject as a
        user-role observation) on success, or with ``error`` (feed back
        to the LLM) when the recall could not be served.
    """
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    engine = _get_engine(session_id)
    scratch: List[Dict[str, Any]] = []
    try:
        outcome = engine.handle_action(action, scratch)
    except RecallError as exc:
        return {"handled": True, "error": str(exc)}
    if outcome is None:
        return {"handled": False}
    return {
        "handled": True,
        "kind": outcome.kind,
        "obs_id": outcome.obs_id,
        "content": outcome.message["content"],
    }


def compaction_recall(session_id: str, obs_id: str, recall_kind: str = "full") -> Dict[str, Any]:
    """Recall a stored observation by id (MCP-native recall).

    The tool-call equivalent of the ``_recall §<id>`` /
    ``_recall_meta §<id>`` text protocol, for platforms whose LLM calls
    MCP tools directly.  Inject the returned ``content`` into the
    conversation as a user-role observation.  Shares the per-step recall
    budget with ``compaction_handle_action`` (see there for the step
    boundary semantics).

    Args:
        session_id: Session from ``compaction_start_session``.
        obs_id: The ``§<obs_id>`` handle shown in a citation (without
            the section sign).
        recall_kind: ``"full"`` for the verbatim body, ``"meta"`` for
            the bounded metadata preview.  (Named ``recall_kind``, not
            ``kind``: several agent-framework MCP clients — e.g. the
            OpenHands SDK, which reserves ``kind`` as the discriminator
            field on every dynamically generated tool-call model —
            crash if a tool parameter is literally named ``kind``.)
    """
    if not isinstance(obs_id, str) or not obs_id:
        raise ValueError("obs_id must be a non-empty string")
    if recall_kind not in ("full", "meta"):
        raise ValueError(f'recall_kind must be "full" or "meta", got {recall_kind!r}')
    verb = "_recall" if recall_kind == "full" else "_recall_meta"
    result = compaction_handle_action(session_id, f"{verb} \u00a7{obs_id}")
    if "error" in result:
        return {"found": False, "error": result["error"]}
    return {
        "found": True,
        "obs_id": result["obs_id"],
        "recall_kind": result["kind"],
        "content": result["content"],
    }


def compaction_verify_value(
    session_id: str, value: str, prefix: Optional[str] = None
) -> Dict[str, Any]:
    """Anti-confabulation guard: did ``value`` appear in any observation?

    Before committing an agent's final answer, check that the value
    actually occurred verbatim in the session's lossless store.  When it
    did not, the agent is likely confabulating (e.g. copying a
    similar-looking value from the wrong observation) and should
    ``_recall`` the relevant citation instead.

    Args:
        session_id: Session from ``compaction_start_session``.
        value: The exact value the agent is about to submit.
        prefix: Optional literal prefix that must immediately precede
            the value in the observation (e.g. ``"trace-key: "``),
            eliminating false-passes on coincidental substrings.

    Returns:
        ``seen`` plus, when False, the most recent ``candidate_obs_ids``
        the agent could recall to find the correct value.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("value must be a non-empty string")
    if prefix is not None and not isinstance(prefix, str):
        raise ValueError("prefix must be a string or null")
    engine = _get_engine(session_id)
    if prefix:
        seen = engine.store.contains_value_with_prefix(prefix, value)
    else:
        seen = engine.store.contains_value(value)
    out: Dict[str, Any] = {"seen": seen}
    if not seen:
        out["candidate_obs_ids"] = engine.store.known_ids()[-5:][::-1]
        out["hint"] = (
            f"The value {value!r} does not appear verbatim in any of the "
            f"{engine.store.n_records} stored observations"
            + (f" (checked for the exact phrase {prefix!r} + value)" if prefix else "")
            + ". Do not guess — recall the observation that contains the "
            "answer and copy it exactly."
        )
    return out


def compaction_session_stats(session_id: str) -> Dict[str, Any]:
    """Telemetry for a session: store size, compactions, recalls, evictions."""
    engine = _get_engine(session_id)
    return {
        "step": engine.step,
        "n_observations": engine.store.n_records,
        "n_compactions": len(engine.compaction_log),
        "n_recalls": len(engine.recall_log),
        "n_evictions": len(engine.eviction_log),
        "in_context_obs": list(engine.in_context_obs.keys()),
        "known_obs_ids": engine.store.known_ids(),
    }


_MESSAGES_SCHEMA = {
    "type": "array",
    "description": "OpenAI-style message list ({role, content, [_obs_id]}).",
    "items": {"type": "object"},
}


def build_compaction_tools() -> List[ToolDefinition]:
    """Return the ARC compaction :class:`ToolDefinition` list."""
    return [
        ToolDefinition(
            name="compaction_start_session",
            description=(
                "Start an ARC context-compaction session for one agent. "
                "Returns a session_id and a system-prompt snippet that "
                "teaches the LLM the _recall protocol. ARC preserves long "
                "observations losslessly and replaces them with bounded "
                "citations on compaction instead of dropping them."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keep_recent": {"type": "integer", "default": 1},
                    "result_max_chars": {"type": "integer", "default": 500},
                    "summary_max_chars": {"type": "integer", "default": 24000},
                    "recall_per_step": {"type": "integer", "default": 2},
                    "recall_budget_chars": {"type": "integer", "default": 200000},
                    "recall_max_chars": {"type": "integer", "default": 0},
                    "wal_path": {"type": "string"},
                    "resume_wal": {"type": "boolean", "default": False},
                },
            },
            handler=compaction_start_session,
        ),
        ToolDefinition(
            name="compaction_close_session",
            description="Close an ARC compaction session and free its store.",
            input_schema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
            handler=compaction_close_session,
        ),
        ToolDefinition(
            name="compaction_begin_step",
            description=(
                "Mark a step boundary (one LLM response): advances the step "
                "counter and resets the per-step recall budget. Optional — "
                "compaction_record_observation also marks a boundary."
            ),
            input_schema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
            handler=compaction_begin_step,
        ),
        ToolDefinition(
            name="compaction_record_observation",
            description=(
                "Store an executed action's output verbatim in the session's "
                "lossless store. Call after every environment execution; the "
                "returned obs_id is the handle used by citations and recalls."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "command": {"type": "string"},
                    "content": {"type": "string"},
                    "returncode": {"type": "integer", "default": 0},
                },
                "required": ["session_id", "command", "content"],
            },
            handler=compaction_record_observation,
        ),
        ToolDefinition(
            name="compaction_compact_messages",
            description=(
                "Compact a message buffer with the ARC citation-rich "
                "operator. Long observations tagged with _obs_id become "
                "bounded recall-able citations instead of being dropped. "
                "Set max_chars > 0 to shrink until the buffer fits."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "messages": _MESSAGES_SCHEMA,
                    "max_chars": {"type": "integer", "default": 0},
                },
                "required": ["session_id", "messages"],
            },
            handler=compaction_compact_messages,
        ),
        ToolDefinition(
            name="compaction_handle_action",
            description=(
                "Intercept a raw LLM action string before execution. If it "
                "is a `_recall §<id>` / `_recall_meta §<id>` command, serve "
                "it from the lossless store (handled=true, inject content); "
                "otherwise handled=false and the action should be executed "
                "normally."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "action": {"type": "string"},
                },
                "required": ["session_id", "action"],
            },
            handler=compaction_handle_action,
        ),
        ToolDefinition(
            name="compaction_recall",
            description=(
                "Recall a previously observed output by its §<obs_id> "
                "citation handle. recall_kind='full' re-injects the verbatim "
                "body; recall_kind='meta' returns only the bounded metadata "
                "preview. Use when a citation in the conversation summary "
                "points to content you need to read again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "obs_id": {"type": "string"},
                    "recall_kind": {
                        "type": "string",
                        "enum": ["full", "meta"],
                        "default": "full",
                    },
                },
                "required": ["session_id", "obs_id"],
            },
            handler=compaction_recall,
        ),
        ToolDefinition(
            name="compaction_verify_value",
            description=(
                "Anti-confabulation guard: check that a value the agent is "
                "about to submit appeared verbatim in some stored "
                "observation. Returns seen=false with candidate obs_ids to "
                "recall when it did not. Optionally require a literal "
                "prefix (e.g. 'trace-key: ') to precede the value."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "value": {"type": "string"},
                    "prefix": {"type": "string"},
                },
                "required": ["session_id", "value"],
            },
            handler=compaction_verify_value,
        ),
        ToolDefinition(
            name="compaction_session_stats",
            description=(
                "Telemetry for an ARC compaction session: observations "
                "stored, compactions, recalls, evictions, live obs_ids."
            ),
            input_schema={
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
            handler=compaction_session_stats,
        ),
    ]
