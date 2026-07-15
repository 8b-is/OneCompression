"""Pure message-list compaction operators.

All functions here are pure: they take a message buffer (a
``list[dict]`` of OpenAI-style ``{"role": ..., "content": ...}``
messages) and return a new, shorter buffer.  No I/O, no store mutation,
no framework dependency — the caller decides *when* to compact (e.g. on
a context-window error or proactively before a request).

Three operators are provided, from lossiest to most recoverable:

:func:`compact_messages`
    Baseline: truncate assistant thoughts, keep short observations,
    drop long ones, inherit prior summaries.
:func:`compact_messages_max`
    Same, but prior summaries are *discarded* on re-compaction (the
    "AutoComp" baseline in the ARC paper; loss points L3 + L4).
:func:`compact_messages_arc`
    Citation-rich: long observations become bounded-length *citations*
    pointing into a lossless :class:`~onecomp.compaction.store.ObsStore`
    and prior summaries are carried forward, so any observation cited
    once stays recall-able for the remainder of the trajectory.

Message-buffer convention
-------------------------

``messages[0]`` is the system message and ``messages[1]`` the task; both
are always kept intact.  Observation messages produced through the ARC
flow carry an ``"_obs_id"`` key (see
:meth:`onecomp.compaction.engine.ArcEngine.make_observation_message`);
messages without it fall back to the string-only path.  Use
:func:`onecomp.compaction.engine.sanitize_messages` to strip private
keys before handing the buffer to an LLM API that rejects extra fields.

Ported from ARC (Addressable Recall for Compaction), Apache-2.0.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from __future__ import annotations

import re
from typing import Optional

from .protocol import format_citation
from .store import ObsStore


THOUGHT_MAX_CHARS = 300
CMD_MAX_CHARS = 150
RESULT_MAX_CHARS = 500

#: Default regex extracting the action body from an assistant message.
DEFAULT_ACTION_REGEX = r"```bash\s*\n(.*?)\n```"
#: Default regex extracting the observation body from a user message.
DEFAULT_OBSERVATION_REGEX = r"<output>\s*(.*?)\s*</output>"

SUMMARY_HEADER = (
    "The following is a summary of your previous actions (long observations omitted):"
)

# Separator placed between summary parts; counted against the budget so
# the cap reflects the real assembled size.
_SUMMARY_SEP = "\n\n---\n\n"

# Hard ceiling on the number of obs_ids listed in a single collapsed
# id-only line.  Keeps the collapsed fallback bounded even on very long
# trajectories that accumulate thousands of observations.
_MAX_COLLAPSED_IDS = 256

# Appended to the summary header when one or more entries were collapsed
# to id-only references, so the model knows a bare ``§<id>`` is
# recall-able.
_ELISION_NOTE = (
    "Lines below that show only `\u00a7<id>` (no head/tail) are older "
    "observations elided to keep this summary within the context budget. "
    "They are NOT lost — issue `_recall \u00a7<id>` to restore any of them "
    "verbatim."
)

_OBS_ID_IN_TEXT_RE = re.compile(r"\u00a7([0-9a-fA-F]{4,40})")


def _compact_assistant(
    content: str, action_regex: str, thought_max_chars: int = THOUGHT_MAX_CHARS
) -> str:
    """Compact an assistant message: truncate thought, keep the command signature."""
    actions = re.findall(action_regex, content, re.DOTALL)
    thought_match = re.match(r"(?s)(.*?)```bash", content)
    thought = thought_match.group(1).strip() if thought_match else content.split("```")[0].strip()
    if thought_max_chars == 0:
        thought = ""
    elif len(thought) > thought_max_chars:
        thought = thought[:thought_max_chars] + "..."
    if actions:
        # Split on && / ||, truncate each sub-command individually.
        parts = re.split(r"(\s*(?:&&|\|\|)\s*)", actions[0].strip())
        cmd = "".join(p[:CMD_MAX_CHARS] if i % 2 == 0 else p for i, p in enumerate(parts))
        if thought:
            return f"{thought}\n\n```bash\n{cmd}\n```"
        return f"```bash\n{cmd}\n```"
    return thought


def compact_messages(
    messages: list[dict],
    *,
    keep_recent: int = 10,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
    inherit_prior_summary: bool = True,
) -> list[dict]:
    """Compact older messages to reduce context size (string-only baseline).

    Returns the (possibly compacted) message list.  Does not mutate the
    input.

    Args:
        messages: Full message history.
        keep_recent: Number of recent turn pairs (assistant + user) to
            keep uncompacted.
        action_regex: Regex extracting the action from assistant messages.
        observation_regex: Regex extracting the observation body from
            user messages.
        thought_max_chars: Max chars for thought truncation.
        result_max_chars: Observations longer than this are dropped.
        inherit_prior_summary: When True (default), the body of a prior
            summary is folded into the new summary.  When False, prior
            summaries are dropped (the "max" baseline's L4
            information-loss point).
    """
    body = messages[2:]
    if len(body) <= keep_recent * 2:
        return messages

    to_compact = body[: len(body) - keep_recent * 2]
    to_keep = body[len(body) - keep_recent * 2 :]

    parts: list[str] = []
    for msg in to_compact:
        if msg.get("role") == "assistant":
            parts.append(
                _compact_assistant(msg.get("content", "") or "", action_regex, thought_max_chars)
            )
        elif msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content.startswith(SUMMARY_HEADER):
                if inherit_prior_summary:
                    parts.append(content[len(SUMMARY_HEADER) :].strip())
                continue
            m = re.search(observation_regex, content, re.DOTALL)
            if m and len(m.group(1)) <= result_max_chars:
                parts.append(f"result: {m.group(1)}")

    summary = SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(parts) if parts else SUMMARY_HEADER

    return [
        messages[0],
        messages[1],
        {"role": "user", "content": summary},
        *to_keep,
    ]


def compact_messages_max(
    messages: list[dict],
    *,
    keep_recent: int = 1,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
) -> list[dict]:
    """Compact older messages, dropping any previous summary.

    Unlike :func:`compact_messages`, which inherits the body of prior
    summaries into the new one, this variant discards them entirely
    (the "AutoComp" baseline in the ARC paper).
    """
    return compact_messages(
        messages,
        keep_recent=keep_recent,
        action_regex=action_regex,
        observation_regex=observation_regex,
        thought_max_chars=thought_max_chars,
        result_max_chars=result_max_chars,
        inherit_prior_summary=False,
    )


def _collapsed_id_line(ids: list[str]) -> str:
    """Render the id-only fallback for a group of evicted citations.

    ``ids`` is an ordered, de-duplicated list (oldest first).  Only the
    most recent ``_MAX_COLLAPSED_IDS`` are listed verbatim; any overflow
    is noted as a count so the line stays bounded.
    """
    if not ids:
        return ""
    shown = ids[-_MAX_COLLAPSED_IDS:]
    dropped = len(ids) - len(shown)
    prefix = f"(+{dropped} older elided) " if dropped else ""
    return prefix + " ".join(f"\u00a7{i}" for i in shown)


def _apply_summary_budget(
    entries: list[dict], summary_max_chars: int
) -> tuple[list[str], bool]:
    """Select the rendered form of each summary entry under a char budget.

    The most *recent* entries are kept in their ``full`` form; once the
    running total would exceed ``summary_max_chars`` every remaining
    (older) entry falls back to its ``collapsed`` form (an id-only
    reference for a citation, or ``""`` to drop an assistant thought).
    Every collapsed observation is still in the lossless store and its
    ``§<id>`` stays visible for ``_recall``.

    Returns ``(parts, collapsed_any)``.  ``summary_max_chars <= 0``
    disables the cap.
    """
    if summary_max_chars <= 0:
        return [e["full"] for e in entries if e["full"]], False

    use_full = [False] * len(entries)
    acc = 0
    for i in range(len(entries) - 1, -1, -1):
        cost = len(entries[i]["full"]) + (len(_SUMMARY_SEP) if acc else 0)
        if acc + cost <= summary_max_chars:
            use_full[i] = True
            acc += cost
        else:
            # This entry does not fit; it and every older entry collapse.
            break

    parts: list[str] = []
    collapsed_any = False
    for i, e in enumerate(entries):
        if use_full[i]:
            text = e["full"]
        else:
            text = e["collapsed"]
            if e["full"]:  # something was actually demoted/dropped
                collapsed_any = True
        if text:
            parts.append(text)
    return parts, collapsed_any


def compact_messages_arc(
    messages: list[dict],
    *,
    obs_store: Optional[ObsStore] = None,
    keep_recent: int = 1,
    action_regex: str = DEFAULT_ACTION_REGEX,
    observation_regex: str = DEFAULT_OBSERVATION_REGEX,
    thought_max_chars: int = THOUGHT_MAX_CHARS,
    result_max_chars: int = RESULT_MAX_CHARS,
    citation_head_chars: int = 200,
    citation_tail_chars: int = 200,
    summary_max_chars: int = 0,
) -> list[dict]:
    """Compact older messages into a citation-rich summary (ARC operator).

    Instead of *dropping* observations longer than ``result_max_chars``,
    observations tagged with an ``_obs_id`` present in ``obs_store`` are
    rendered as structured citations (bounded-length pointers into the
    lossless store).  Prior summary bodies are carried forward (header
    stripped), so an obs_id cited once stays accessible for the rest of
    the trajectory.

    Degeneration property: with ``obs_store=None`` and no ``_obs_id``
    tags, the output is identical to :func:`compact_messages_max` except
    that prior summary bodies are preserved.

    Args:
        messages: Full message history (first two entries kept intact).
        obs_store: The lossless observation store.  ``None`` falls back
            to the legacy drop behaviour.
        keep_recent: Number of recent turn pairs to keep uncompacted.
        action_regex: Regex extracting the action from assistant messages.
        observation_regex: Regex extracting the observation body from
            user messages (store-less fallback path).
        thought_max_chars: Max chars of assistant thought retained.
        result_max_chars: Observations at most this long are kept
            verbatim; longer ones become citations (or are dropped when
            not in the store).
        citation_head_chars: Bytes of head preview per citation.
        citation_tail_chars: Bytes of tail preview per citation.
        summary_max_chars: Character cap on the assembled summary body.
            ``0`` (default) disables the cap.  When > 0, the most recent
            entries stay verbatim and older citations collapse to
            id-only ``§<id>`` references, bounding the compacted context
            regardless of trajectory length.

    Returns:
        A new ``list[dict]`` with the same first two messages, the
        summary inserted as a user message at index 2, and the recent
        ``keep_recent`` turn pairs appended verbatim.
    """
    body = messages[2:]
    if len(body) <= keep_recent * 2:
        return messages

    to_compact = body[: len(body) - keep_recent * 2]
    to_keep = body[len(body) - keep_recent * 2 :]

    # Each entry is ``{"full": <verbatim rendering>, "collapsed":
    # <id-only fallback or "">}``; the budget pass chooses which form to
    # emit.  ``emitted_obs_ids`` de-duplicates citations across prior
    # summaries and current messages (ids normalised to lowercase).
    entries: list[dict] = []
    emitted_obs_ids: set[str] = set()

    def _add(full: str, collapsed: str = "") -> None:
        if full:
            entries.append({"full": full, "collapsed": collapsed})

    for msg in to_compact:
        if msg.get("role") == "assistant":
            # An assistant thought has no recall-able body: collapsed
            # form is empty (dropped when outside the budget).
            _add(_compact_assistant(msg.get("content", "") or "", action_regex, thought_max_chars))
            continue

        if msg.get("role") != "user":
            continue

        content = (msg.get("content") or "").strip()
        if content.startswith(SUMMARY_HEADER):
            body_text = content[len(SUMMARY_HEADER) :].strip()
            if body_text:
                inherited_ids: list[str] = []
                for m in _OBS_ID_IN_TEXT_RE.finditer(body_text):
                    oid = m.group(1).lower()
                    emitted_obs_ids.add(oid)
                    if oid not in inherited_ids:
                        inherited_ids.append(oid)
                # An inherited prior summary collapses to the
                # (recency-capped) list of obs_ids it referenced.
                _add(body_text, _collapsed_id_line(inherited_ids))
            continue

        obs_id = msg.get("_obs_id")
        if not isinstance(obs_id, str):
            # Untagged (or foreign-tagged, e.g. over an RPC boundary that
            # mangled the field) message: use the string-only fallback.
            obs_id = None
        if obs_id is not None and obs_id.lower() in emitted_obs_ids:
            # Already cited (e.g. a `_recall` on an id inherited from
            # the prior summary); skip the duplicate.
            continue
        if obs_id is not None and obs_store is not None and obs_store.has(obs_id):
            rec = obs_store.get(obs_id)
            assert rec is not None
            if rec.char_count <= result_max_chars:
                full = f"\u00a7{rec.obs_id} result: {rec.full_content}"
            else:
                full = format_citation(
                    obs_id=rec.obs_id,
                    cmd_signature=rec.cmd_signature,
                    returncode=rec.returncode,
                    line_count=rec.line_count,
                    char_count=rec.char_count,
                    head=rec.head,
                    tail=rec.tail,
                    head_chars=citation_head_chars,
                    tail_chars=citation_tail_chars,
                )
            emitted_obs_ids.add(rec.obs_id.lower())
            _add(full, f"\u00a7{rec.obs_id}")
            continue

        m = re.search(observation_regex, content, re.DOTALL)
        if m and len(m.group(1)) <= result_max_chars:
            # No obs_id (store-less fallback): not recall-able, so drop
            # on budget overflow.
            _add(f"result: {m.group(1)}")

    parts, collapsed_any = _apply_summary_budget(entries, summary_max_chars)

    summary_body = _SUMMARY_SEP.join(parts) if parts else ""
    header = (SUMMARY_HEADER + "\n" + _ELISION_NOTE) if collapsed_any else SUMMARY_HEADER
    summary_text = (header + "\n\n" + summary_body) if summary_body else header

    return [
        messages[0],
        messages[1],
        {"role": "user", "content": summary_text},
        *to_keep,
    ]
