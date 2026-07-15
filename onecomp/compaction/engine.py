"""Framework-agnostic ARC compaction engine.

:class:`ArcEngine` binds the three pure layers (store, protocol,
compactor) into a stateful facade that any agent framework — single
agent or multi-agent system — can integrate through three touchpoints:

1.  **Prompt**: append :meth:`ArcEngine.system_prompt_snippet` to the
    agent's system prompt so the LLM knows the recall protocol.
2.  **Action interception**: pass every LLM-emitted action through
    :meth:`ArcEngine.handle_action` *before* executing it in the
    environment.  A recall action is served from the lossless store
    (returning the message to inject); any other action returns ``None``
    and should be executed normally.  After execution, register the
    result with :meth:`ArcEngine.record_observation` and append the
    message built by :meth:`ArcEngine.make_observation_message`.
3.  **Compaction**: when the framework detects a context overflow (or
    proactively, via :meth:`ArcEngine.enforce_char_budget`), call
    :meth:`ArcEngine.compact` on the message buffer.

The engine holds no reference to the message buffer: every method that
needs it takes ``messages`` explicitly, so the framework remains the
single owner of its conversation state.  The engine owns only the
lossless store, the in-context LRU bookkeeping, and research telemetry
(``compaction_log`` / ``recall_log`` / ``eviction_log``).

Nothing in this module (or in the layers below it) imports an agent
framework, an LLM client, or torch — the whole package is pure Python.

Example:
    >>> from onecomp.compaction import ArcEngine, ArcEngineConfig
    >>> engine = ArcEngine(ArcEngineConfig(recall_per_step=2))
    >>> messages = [{"role": "system", "content": "..."},
    ...             {"role": "user", "content": "task"}]
    >>> # after executing an action:
    >>> rec = engine.record_observation(command="grep -rn foo .", content=big_output)
    >>> messages.append(engine.make_observation_message(rec, rendered=big_output))
    >>> # on context overflow:
    >>> messages = engine.compact(messages)
    >>> # before executing the next LLM action:
    >>> outcome = engine.handle_action(llm_action, messages)
    >>> if outcome is not None:   # it was a recall; message already injected
    ...     pass                  # skip environment execution this turn

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from .compactor import (
    DEFAULT_ACTION_REGEX,
    DEFAULT_OBSERVATION_REGEX,
    RESULT_MAX_CHARS,
    THOUGHT_MAX_CHARS,
    compact_messages_arc,
)
from .protocol import (
    SYSTEM_PROMPT_RECALL_SNIPPET,
    format_meta_response,
    format_recall_response,
    looks_like_recall_intent,
    normalise_cmd_signature,
    parse_any_recall,
    suggest_closest_obs_id,
)
from .store import ObsRecord, ObsStore

logger = logging.getLogger(__name__)

#: Message keys private to the compaction machinery.  Strip them with
#: :func:`sanitize_messages` before calling an LLM API that rejects
#: unknown fields.
PRIVATE_MESSAGE_KEYS = ("_obs_id", "_is_recall", "_is_recall_meta", "_is_evicted")


class RecallError(Exception):
    """A recall action could not be served (unknown id, malformed
    syntax, or per-step budget exhausted).

    The exception message is written *for the LLM*: the integration
    layer should feed it back as the observation for this turn so the
    model can self-correct (mirroring the "non-terminating exception"
    convention of ReAct-style agent loops).
    """


@dataclass
class RecallOutcome:
    """Result of a successfully served recall action.

    Attributes:
        kind: ``"recall"`` (full body) or ``"recall_meta"`` (preview).
        obs_id: Canonical id of the recalled observation.
        message: The user-role message dict that was appended to the
            buffer passed to :meth:`ArcEngine.handle_action`.
    """

    kind: str
    obs_id: str
    message: dict


@dataclass
class ArcEngineConfig:
    """Tunable knobs of :class:`ArcEngine`.

    Args:
        keep_recent: Recent turn pairs kept uncompacted by
            :meth:`ArcEngine.compact`.
        thought_max_chars: Max chars of assistant thought retained in
            the compaction summary.
        result_max_chars: Observations at most this long stay verbatim
            in the summary; longer ones become citations.
        citation_head_chars: Head preview bytes per citation.
        citation_tail_chars: Tail preview bytes per citation.
        summary_max_chars: Character cap on the compaction summary
            (``0`` = unbounded).  Bounds the compacted context to O(1)
            instead of O(n_observations) on long trajectories.
        recall_per_step: Max recalls served per step (``begin_step``
            resets the counter).  ``0`` disables the materialise
            primitive while keeping citations and the store active.
        recall_budget_chars: Total verbatim-recall chars kept in
            context; older recalls are LRU-evicted to placeholders
            (bodies stay in the store).  ``0`` disables eviction.
        recall_max_chars: When > 0, a full recall longer than this is
            re-injected as a head+tail window with an elision marker,
            bounding per-recall context cost.  ``0`` = lossless.
        obs_id_hex_chars: Length of minted observation ids.
        action_regex: Regex extracting the action from assistant
            messages during compaction.
        observation_regex: Regex extracting observation bodies from
            untagged user messages during compaction.
        wal_path: Optional write-ahead log for crash-safe store
            persistence (see :class:`~onecomp.compaction.store.ObsStore`).
    """

    keep_recent: int = 1
    thought_max_chars: int = THOUGHT_MAX_CHARS
    result_max_chars: int = RESULT_MAX_CHARS
    citation_head_chars: int = 200
    citation_tail_chars: int = 200
    summary_max_chars: int = 24_000
    recall_per_step: int = 2
    recall_budget_chars: int = 200_000
    recall_max_chars: int = 0
    obs_id_hex_chars: int = 8
    action_regex: str = DEFAULT_ACTION_REGEX
    observation_regex: str = DEFAULT_OBSERVATION_REGEX
    wal_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Reject invalid limits before they reach slicing and budget logic."""
        non_negative = {
            "keep_recent": self.keep_recent,
            "thought_max_chars": self.thought_max_chars,
            "result_max_chars": self.result_max_chars,
            "citation_head_chars": self.citation_head_chars,
            "citation_tail_chars": self.citation_tail_chars,
            "summary_max_chars": self.summary_max_chars,
            "recall_per_step": self.recall_per_step,
            "recall_budget_chars": self.recall_budget_chars,
            "recall_max_chars": self.recall_max_chars,
        }
        for name, value in non_negative.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.obs_id_hex_chars, int) or isinstance(
            self.obs_id_hex_chars, bool
        ):
            raise ValueError("obs_id_hex_chars must be an integer between 4 and 40")
        if not 4 <= self.obs_id_hex_chars <= 40:
            raise ValueError("obs_id_hex_chars must be between 4 and 40")
        for name, pattern in (
            ("action_regex", self.action_regex),
            ("observation_regex", self.observation_regex),
        ):
            if not isinstance(pattern, str):
                raise ValueError(f"{name} must be a string")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{name} is not a valid regular expression: {exc}") from exc
        if self.wal_path is not None:
            if not isinstance(self.wal_path, str):
                raise ValueError("wal_path must be a string or None")
            if not self.wal_path.strip():
                raise ValueError("wal_path must be a non-empty string when provided")


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with private compaction keys removed.

    Use this immediately before handing the buffer to an LLM API that
    rejects fields other than ``role`` / ``content`` (the original list
    — including the tags the engine relies on — stays untouched).
    """
    return [
        {k: v for k, v in msg.items() if k not in PRIVATE_MESSAGE_KEYS} for msg in messages
    ]


class ArcEngine:
    """Stateful ARC facade: lossless store + recall dispatch + compaction.

    See the module docstring for the three integration touchpoints.  All
    methods that operate on the conversation take the message buffer as
    an explicit argument; the engine never keeps a reference to it.
    """

    def __init__(self, config: Optional[ArcEngineConfig] = None):
        self.config = config or ArcEngineConfig()
        self.store = ObsStore(
            obs_id_hex_chars=self.config.obs_id_hex_chars,
            wal_path=self.config.wal_path,
        )
        #: obs_ids whose *full* recall body currently lives in the
        #: buffer, in LRU order (oldest first).
        self.in_context_obs: OrderedDict[str, None] = OrderedDict()
        self.compaction_log: list[dict] = []
        self.recall_log: list[dict] = []
        self.eviction_log: list[dict] = []
        self._step: int = 0
        self._recalls_in_step: int = 0

    @classmethod
    def from_wal(
        cls, wal_path: str, config: Optional[ArcEngineConfig] = None
    ) -> "ArcEngine":
        """Restore observations from ``wal_path`` and continue appending.

        Recall/compaction telemetry is process-local and starts empty; the step
        counter resumes at the greatest stored ``created_step`` so newly
        recorded observations keep monotonic step metadata.
        """
        if not isinstance(wal_path, str) or not wal_path.strip():
            raise ValueError("wal_path must be a non-empty string")
        resolved_config = config or ArcEngineConfig(wal_path=wal_path)
        if resolved_config.wal_path is None:
            resolved_config = replace(resolved_config, wal_path=wal_path)
        elif Path(resolved_config.wal_path).resolve() != Path(wal_path).resolve():
            raise ValueError("config.wal_path must refer to the WAL being resumed")
        engine = cls(resolved_config)
        engine.store.close()
        engine.store = ObsStore.from_wal(
            wal_path,
            obs_id_hex_chars=resolved_config.obs_id_hex_chars,
            new_wal_path=resolved_config.wal_path,
        )
        engine._step = engine.store.max_created_step
        return engine

    # ------------------------------------------------------------------
    # Touchpoint 1: prompt
    # ------------------------------------------------------------------
    def system_prompt_snippet(self) -> str:
        """The recall-protocol instructions to append to the system prompt."""
        return SYSTEM_PROMPT_RECALL_SNIPPET.format(
            recall_per_step=self.config.recall_per_step
        )

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------
    def begin_step(self) -> int:
        """Advance the step counter and reset the per-step recall budget.

        Call once per LLM response, before :meth:`handle_action`.
        Returns the new step index (1-based).
        """
        self._step += 1
        self._recalls_in_step = 0
        return self._step

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    # Touchpoint 2: observations and recall dispatch
    # ------------------------------------------------------------------
    def record_observation(
        self,
        *,
        command: str,
        content: str,
        returncode: int = 0,
        extra: Optional[dict] = None,
    ) -> ObsRecord:
        """Store the verbatim result of an executed action.

        Args:
            command: The action that produced the observation (any
                string; normalised to a signature internally).
            content: The raw observation body, stored verbatim.
            returncode: Exit status of the action, if applicable.
            extra: Optional JSON-serialisable metadata.

        Returns:
            The (possibly pre-existing, idempotent) store record.
        """
        if not isinstance(command, str):
            raise ValueError("command must be a string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise ValueError("returncode must be an integer")
        return self.store.put(
            cmd_signature=normalise_cmd_signature(command),
            full_content=content,
            returncode=returncode,
            created_step=self._step,
            head_chars=self.config.citation_head_chars,
            tail_chars=self.config.citation_tail_chars,
            extra=extra,
        )

    def make_observation_message(
        self, record: ObsRecord, *, rendered: Optional[str] = None, role: str = "user"
    ) -> dict:
        """Build the message dict for an observation, tagged with its id.

        Args:
            record: The store record from :meth:`record_observation`.
            rendered: The observation as the framework presents it to
                the LLM (templated, wrapped, ...).  Defaults to the raw
                stored content.
            role: Message role (default ``"user"``).
        """
        return {
            "role": role,
            "content": rendered if rendered is not None else record.full_content,
            "_obs_id": record.obs_id,
        }

    def handle_action(self, action: str, messages: list[dict]) -> Optional[RecallOutcome]:
        """Intercept a recall action; return ``None`` for anything else.

        Call with every LLM-emitted action *before* executing it.  On a
        valid recall, the response message is appended to ``messages``
        (mutating it), LRU eviction is applied for full recalls, and a
        :class:`RecallOutcome` is returned — the framework should then
        skip environment execution for this turn.

        Raises:
            RecallError: Unknown obs_id (with "did you mean" hints),
                malformed recall syntax, or per-step budget exhausted.
                Feed ``str(exc)`` back to the LLM as this turn's
                observation.
        """
        if not isinstance(action, str):
            raise ValueError("action must be a string")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        parsed = parse_any_recall(action)
        if parsed is None:
            if looks_like_recall_intent(action):
                raise RecallError(
                    "Malformed recall command. Expected exactly "
                    "`_recall §<obs_id>` or `_recall_meta §<obs_id>` "
                    "(no extra shell, the section sign § is required). "
                    "Got: " + action.strip().splitlines()[0][:120]
                )
            return None

        kind, obs_id = parsed
        # Resolve the id BEFORE charging the budget: a typo should give
        # a helpful error without burning a recall slot.
        rec = self.store.get(obs_id)
        if rec is None:
            raise self._unknown_obs_id_error(obs_id, kind)

        self._recalls_in_step += 1
        if self._recalls_in_step > self.config.recall_per_step:
            raise RecallError(
                f"Recall budget exceeded: at most {self.config.recall_per_step} "
                f"recall(s) per response. Please continue with a normal action "
                f"on your next response."
            )

        rec.last_recalled = self._step
        rec.n_recalls += 1

        if kind == "recall":
            body, elided = self._window_recall_body(rec.full_content)
            content = format_recall_response(
                obs_id=rec.obs_id,
                cmd_signature=rec.cmd_signature,
                returncode=rec.returncode,
                char_count=rec.char_count,
                line_count=rec.line_count,
                body=body,
                elided_chars=elided,
            )
        else:
            content = format_meta_response(
                obs_id=rec.obs_id,
                cmd_signature=rec.cmd_signature,
                returncode=rec.returncode,
                line_count=rec.line_count,
                char_count=rec.char_count,
                head=rec.head,
                tail=rec.tail,
                head_chars=self.config.citation_head_chars,
                tail_chars=self.config.citation_tail_chars,
            )

        message: dict = {"role": "user", "content": content, "_obs_id": rec.obs_id,
                         "_is_recall": True}
        if kind == "recall_meta":
            message["_is_recall_meta"] = True
        messages.append(message)

        self.recall_log.append(
            {
                "step": self._step,
                "obs_id": rec.obs_id,
                "kind": kind,
                "char_count": rec.char_count,
                "cmd_signature": rec.cmd_signature,
                "n_recalls": rec.n_recalls,
                "action": action,
            }
        )

        if kind == "recall":
            # A metadata preview never evicts a full recall the LLM may
            # still need, so only full recalls touch the LRU.
            self._touch_in_context(rec.obs_id)
            self._evict_over_recall_budget(messages)

        return RecallOutcome(kind=kind, obs_id=rec.obs_id, message=message)

    def find_prior_executions(self, command: str) -> list[ObsRecord]:
        """Records previously produced by an equivalent command.

        Re-execution telemetry: lets the integration layer notice when
        the LLM re-runs a command whose result is already stored, and
        e.g. log it or hint at ``_recall`` instead.
        """
        return self.store.find_by_signature(normalise_cmd_signature(command))

    # ------------------------------------------------------------------
    # Touchpoint 3: compaction
    # ------------------------------------------------------------------
    def compact(self, messages: list[dict], *, summary_max_chars: Optional[int] = None) -> list[dict]:
        """Compact ``messages`` with the ARC citation-rich operator.

        Call on a context-window overflow (reactive) or ahead of one
        (proactive).  Returns a new list — or the *same* list object when
        there was nothing to compact (too few messages), in which case
        no log entry is written.  ``summary_max_chars`` overrides the
        configured cap for this pass (used by :meth:`enforce_char_budget`
        to shrink progressively).
        """
        if summary_max_chars is not None and (
            not isinstance(summary_max_chars, int)
            or isinstance(summary_max_chars, bool)
            or summary_max_chars < 0
        ):
            raise ValueError("summary_max_chars must be a non-negative integer or None")
        cap = self.config.summary_max_chars if summary_max_chars is None else summary_max_chars
        before = len(messages)
        compacted = compact_messages_arc(
            messages,
            obs_store=self.store,
            keep_recent=self.config.keep_recent,
            action_regex=self.config.action_regex,
            observation_regex=self.config.observation_regex,
            thought_max_chars=self.config.thought_max_chars,
            result_max_chars=self.config.result_max_chars,
            citation_head_chars=self.config.citation_head_chars,
            citation_tail_chars=self.config.citation_tail_chars,
            summary_max_chars=cap,
        )
        if compacted is messages:
            return messages  # nothing to compact — keep the log clean
        evicted = self._sync_in_context(compacted)
        self.compaction_log.append(
            {
                "step": self._step,
                "before": before,
                "after": len(compacted),
                "summary_max_chars": cap,
                "obs_store_size": self.store.n_records,
                "in_context_obs": list(self.in_context_obs.keys()),
                "evicted_by_compaction": evicted,
            }
        )
        return compacted

    #: Chars reserved for the summary header, elision note, and entry
    #: separators when deriving a summary cap from a total-chars budget.
    _SUMMARY_MARGIN_CHARS = 400

    def enforce_char_budget(self, messages: list[dict], max_chars: int) -> list[dict]:
        """Shrink ``messages`` until its total content length fits ``max_chars``.

        Proactive variant of the reactive compact-on-error path.  Two
        levers, cheapest first: (1) LRU-evict live verbatim recalls
        (bodies stay recall-able from the store), then (2) compact with
        a summary cap derived from the actual deficit — the budget minus
        the chars compaction cannot remove (system, task, and the
        ``keep_recent`` tail) — shrinking further on each retry.
        Character-based on purpose: token counting is framework
        specific, and callers can convert their token budget with their
        own chars-per-token estimate.

        Returns the (possibly new) message list.  Best-effort: when the
        uncompactable overhead alone exceeds ``max_chars`` the result
        may still be over budget.
        """
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
            raise ValueError("max_chars must be a non-negative integer")
        msgs = list(messages)
        if _total_chars(msgs) <= max_chars:
            return msgs

        # Never evict the single most recent recall: the LLM has not had
        # a chance to read it yet, and evicting it here puts the agent in
        # a recall -> evict -> recall loop.  (Same guard as the reactive
        # LRU eviction; a lone oversized recall is bounded by
        # ``recall_max_chars`` instead.)
        while len(self.in_context_obs) > 1 and _total_chars(msgs) > max_chars:
            evict_id = next(iter(self.in_context_obs))
            self._replace_recall_with_placeholder(evict_id, msgs)
            self.in_context_obs.pop(evict_id, None)

        keep = self.config.keep_recent * 2
        for attempt in range(8):
            if _total_chars(msgs) <= max_chars:
                break
            head = msgs[:2]
            tail = msgs[len(msgs) - keep :] if keep else []
            overhead = sum(len(m.get("content", "") or "") for m in (*head, *tail))
            cap = max(200, (max_chars - overhead - self._SUMMARY_MARGIN_CHARS) >> attempt)
            compacted = self.compact(msgs, summary_max_chars=cap)
            if compacted is msgs:
                break  # nothing left to compact
            if _total_chars(compacted) < _total_chars(msgs):
                msgs = compacted
            elif cap <= 200:
                break  # cap already at the floor and nothing shrank
            # else: keep msgs; the next attempt derives a smaller cap.
        return msgs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Lossless JSON-friendly dump of the engine state (for sidecar
        files, resumption, or offline analysis)."""
        return {
            "step": self._step,
            "obs_store": self.store.to_serializable(),
            "in_context_obs": list(self.in_context_obs.keys()),
            "compaction_log": self.compaction_log,
            "recall_log": self.recall_log,
            "eviction_log": self.eviction_log,
        }

    def close(self) -> None:
        """Flush and close the store's WAL, if enabled."""
        self.store.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _unknown_obs_id_error(self, obs_id: str, kind: str) -> RecallError:
        suggestions = suggest_closest_obs_id(obs_id, self.store.known_ids(), n=3)
        hint = (
            " Did you mean one of: " + ", ".join(f"\u00a7{s}" for s in suggestions) + "?"
            if suggestions
            else ""
        )
        return RecallError(
            f"Unknown observation id: \u00a7{obs_id}. Only ids that appear in your "
            f"current context (as a citation \u00a7<id>) can be {kind}-recalled."
            f"{hint} Continue with a normal action."
        )

    def _window_recall_body(self, content: str) -> tuple[str, int]:
        """Apply the ``recall_max_chars`` head+tail window.

        Returns ``(body, elided_chars)``.  When the guard is disabled or
        the body fits, the content is returned unchanged.
        """
        limit = self.config.recall_max_chars
        if limit <= 0 or len(content) <= limit:
            return content, 0
        head_n = limit // 2
        tail_n = limit - head_n
        elided = len(content) - head_n - tail_n
        marker = (
            f"\n\n... [recall windowed: {elided} chars elided from the middle to "
            f"cap context re-expansion. The full body is still stored verbatim — "
            f"re-run the original command if you need the elided region.] ...\n\n"
        )
        return content[:head_n] + marker + content[-tail_n:], elided

    def _touch_in_context(self, obs_id: str) -> None:
        self.in_context_obs.pop(obs_id, None)
        self.in_context_obs[obs_id] = None

    def _evict_over_recall_budget(self, messages: list[dict]) -> None:
        budget = self.config.recall_budget_chars
        if budget <= 0:
            return

        def total_chars() -> int:
            return sum(
                self.store.get(oid).char_count
                for oid in self.in_context_obs
                if self.store.has(oid)
            )

        while len(self.in_context_obs) > 1 and total_chars() > budget:
            evict_id = next(iter(self.in_context_obs))
            self._replace_recall_with_placeholder(evict_id, messages)
            self.in_context_obs.pop(evict_id, None)

    def _replace_recall_with_placeholder(self, obs_id: str, messages: list[dict]) -> None:
        """Collapse every live recall message for ``obs_id`` into a single
        placeholder (mutates ``messages``).

        Collecting all matches in one pass matters for the recommended
        ``_recall_meta`` → ``_recall`` flow: placeholder-ising only the
        first match would leave a stale full body later in the buffer.
        """
        indices = [
            i
            for i, m in enumerate(messages)
            if m.get("_obs_id") == obs_id and m.get("_is_recall") and not m.get("_is_evicted")
        ]
        if not indices:
            return
        placeholder = (
            f"\u00a7{obs_id} (evicted from working set — issue "
            f"`_recall \u00a7{obs_id}` to re-load)"
        )
        keep = indices[0]
        messages[keep] = {
            "role": "user",
            "content": placeholder,
            "_obs_id": obs_id,
            "_is_evicted": True,
        }
        for i in reversed(indices[1:]):
            del messages[i]
        self.eviction_log.append({"step": self._step, "obs_id": obs_id, "reason": "budget"})

    def _sync_in_context(self, messages: list[dict]) -> list[str]:
        """Drop LRU entries whose messages no longer exist after compaction.

        Returns the evicted ids.
        """
        live_ids = {m.get("_obs_id") for m in messages if m.get("_obs_id")}
        evicted: list[str] = []
        for oid in list(self.in_context_obs.keys()):
            if oid not in live_ids:
                self.in_context_obs.pop(oid, None)
                evicted.append(oid)
                self.eviction_log.append(
                    {"step": self._step, "obs_id": oid, "reason": "compaction"}
                )
        return evicted


def _total_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "") or "") for m in messages)
