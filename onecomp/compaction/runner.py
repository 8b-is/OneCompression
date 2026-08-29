"""OneComp-style standalone runner for context compaction.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

from .compactor import SUMMARY_HEADER, compact_messages_arc
from .config import CompactionConfig
from .engine import ArcEngine, ArcEngineConfig, RecallOutcome, sanitize_messages
from .formal import (
    ActionResponse,
    ArcCoreService,
    BudgetExceeded,
    CatalogAction,
    InvocationBudgetGuard,
    OccurrenceIdConfig,
    SQLiteObservationArchive,
    SYSTEM_PROMPT_FORMAL_RECALL,
    looks_like_action_intent,
    parse_action,
)
from .formal.protocol import RecallMetadataAction
from .protocol import normalise_cmd_signature
from .adapters.messages import protected_prefix_count


class CompactionRunner:
    """Run compaction independently or embed it at an agent host boundary.

    The runner owns no model and never dispatches network requests itself.
    Host integrations call ``record_observation`` after a tool completes,
    ``compact`` before a provider request, ``handle_action`` before executing a
    recall-looking action, and a guard from ``create_budget_guard`` around the
    final provider invocation.
    """

    def __init__(
        self,
        config: Optional[CompactionConfig] = None,
        *,
        unit_counter: Callable[[str], int] = len,
    ) -> None:
        self.config = config or CompactionConfig()
        if not callable(unit_counter):
            raise TypeError("unit_counter must be callable")
        self.unit_counter = unit_counter
        self._lock = threading.RLock()
        self._step = 0
        self._actions_in_step = 0
        self._closed = False
        self._events: deque[dict[str, Any]] = deque(
            maxlen=self.config.max_telemetry_events or None
        )
        self._engine: Optional[ArcEngine] = None
        self.archive: Optional[SQLiteObservationArchive] = None
        self.service: Optional[ArcCoreService] = None

        if self.config.profile == "formal-v1":
            archive_path: str | Path = self.config.archive_path or ":memory:"
            self.archive = SQLiteObservationArchive(
                archive_path,
                id_config=OccurrenceIdConfig(
                    max_occurrences=self.config.max_occurrences,
                    fingerprint_bits=self.config.fingerprint_bits,
                ),
                max_archive_bytes=self.config.max_archive_bytes,
                preview_chars=max(
                    self.config.citation_head_chars,
                    self.config.citation_tail_chars,
                ),
                verify_on_read=self.config.verify_archive_reads,
            )
            self.service = ArcCoreService(
                self.archive,
                max_recall_units=self.config.max_recall_units,
                measure=self.unit_counter,
            )
            self._step = self.archive.max_created_step
        else:
            self._engine = ArcEngine(
                ArcEngineConfig(
                    keep_recent=self.config.keep_recent,
                    thought_max_chars=self.config.thought_max_chars,
                    result_max_chars=self.config.result_max_chars,
                    citation_head_chars=self.config.citation_head_chars,
                    citation_tail_chars=self.config.citation_tail_chars,
                    summary_max_chars=self.config.summary_max_chars,
                    recall_per_step=self.config.recalls_per_step,
                    recall_budget_chars=self.config.max_recall_units,
                    recall_max_chars=self.config.max_recall_units,
                    action_regex=self.config.action_regex,
                    observation_regex=self.config.observation_regex,
                    wal_path=(
                        None if self.config.archive_path is None else str(self.config.archive_path)
                    ),
                )
            )

    @property
    def profile(self) -> str:
        return self.config.profile

    @property
    def step(self) -> int:
        return self._engine.step if self._engine is not None else self._step

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Return an immutable telemetry snapshot for analyzers."""
        with self._lock:
            return tuple(dict(event) for event in self._events)

    def begin_step(self) -> int:
        """Advance the model-response step and reset the action allowance."""
        self._require_open()
        with self._lock:
            if self._engine is not None:
                return self._engine.begin_step()
            self._step += 1
            self._actions_in_step = 0
            return self._step

    def system_prompt_snippet(self) -> str:
        self._require_open()
        if self._engine is not None:
            return self._engine.system_prompt_snippet()
        return SYSTEM_PROMPT_FORMAL_RECALL

    def record_observation(
        self,
        *,
        command: str,
        content: str,
        source_key: Optional[str] = None,
        returncode: int = 0,
        extra: Optional[dict[str, Any]] = None,
        tags: tuple[str, ...] | list[str] = (),
    ) -> Any:
        """Preserve a tool-result occurrence before replacing host-visible text."""
        self._require_open()
        if self._engine is not None:
            record = self._engine.record_observation(
                command=command,
                content=content,
                returncode=returncode,
                extra=extra,
            )
        else:
            assert self.service is not None
            record = self.service.record(
                normalise_cmd_signature(command, max_chars=None),
                content,
                source_key=source_key,
                returncode=returncode,
                created_step=self._step,
                extra=extra,
                tags=tags,
            )
        self._event(
            "observation",
            obs_id=record.obs_id,
            chars=record.char_count,
            returncode=record.returncode,
        )
        return record

    @staticmethod
    def make_observation_message(
        record: Any,
        *,
        rendered: Optional[str] = None,
        role: str = "user",
    ) -> dict[str, Any]:
        """Build an ARC-tagged message accepted by the pure compactor."""
        return {
            "role": role,
            "content": record.full_content if rendered is None else rendered,
            "_obs_id": record.obs_id,
        }

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compact old turns while preserving every tagged body in the archive."""
        self._require_open()
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if self._engine is not None:
            return self._engine.compact(messages)
        assert self.archive is not None
        compacted = compact_messages_arc(
            messages,
            obs_store=self.archive,
            keep_recent=self.config.keep_recent,
            action_regex=self.config.action_regex,
            observation_regex=self.config.observation_regex,
            thought_max_chars=self.config.thought_max_chars,
            result_max_chars=self.config.result_max_chars,
            citation_head_chars=self.config.citation_head_chars,
            citation_tail_chars=self.config.citation_tail_chars,
            summary_max_chars=self.config.summary_max_chars,
        )
        if compacted is not messages:
            self._event(
                "compaction",
                before_messages=len(messages),
                after_messages=len(compacted),
                before_chars=_message_chars(messages),
                after_chars=_message_chars(compacted),
            )
        return compacted

    def enforce_char_budget(
        self,
        messages: list[dict[str, Any]],
        max_chars: int,
    ) -> list[dict[str, Any]]:
        """Apply compaction and fail closed when protected content cannot fit."""
        self._require_open()
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        if _message_chars(messages) <= max_chars:
            return messages
        if self._engine is not None:
            return self._engine.enforce_char_budget(messages, max_chars)
        assert self.archive is not None

        def compact_with_cap(cap: int) -> list[dict[str, Any]]:
            return compact_messages_arc(
                messages,
                obs_store=self.archive,
                keep_recent=self.config.keep_recent,
                action_regex=self.config.action_regex,
                observation_regex=self.config.observation_regex,
                thought_max_chars=self.config.thought_max_chars,
                result_max_chars=self.config.result_max_chars,
                citation_head_chars=self.config.citation_head_chars,
                citation_tail_chars=self.config.citation_tail_chars,
                summary_max_chars=cap,
            )

        minimum = compact_with_cap(1)
        minimum_chars = _message_chars(minimum)
        if minimum is messages or minimum_chars > max_chars:
            raise BudgetExceeded(
                "protected and recent message content exceeds the character envelope: "
                f"minimum={minimum_chars}, limit={max_chars}"
            )
        summary_index = _summary_index(minimum, protected_prefix_count(messages))
        fixed_chars = minimum_chars - len(str(minimum[summary_index].get("content", "")))
        summary_cap = max(1, max_chars - fixed_chars)
        if self.config.summary_max_chars:
            summary_cap = min(summary_cap, self.config.summary_max_chars)
        compacted = compact_with_cap(summary_cap)
        after_chars = _message_chars(compacted)
        if after_chars > max_chars:
            raise BudgetExceeded(
                "formal compaction could not satisfy the character envelope: "
                f"after={after_chars}, limit={max_chars}"
            )
        self._event(
            "compaction",
            before_messages=len(messages),
            after_messages=len(compacted),
            before_chars=_message_chars(messages),
            after_chars=after_chars,
            enforced_limit=max_chars,
        )
        return compacted

    def handle_action(
        self,
        action: str,
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[ActionResponse | RecallOutcome]:
        """Intercept a recall action before the host executes it as a tool."""
        self._require_open()
        if messages is not None and not isinstance(messages, list):
            raise TypeError("messages must be a list or None")
        if self._engine is not None:
            return self._engine.handle_action(action, [] if messages is None else messages)
        assert self.service is not None
        if not looks_like_action_intent(action):
            return None
        with self._lock:
            if self._actions_in_step >= self.config.recalls_per_step:
                raise ValueError("formal ARC action allowance exceeded for this model response")
            parsed = parse_action(action)
            response = self.service.execute_action(action, recall_step=self._step)
            self._actions_in_step += 1
        if messages is not None:
            message = {
                "role": "user",
                "content": response.body,
                "_obs_id": response.obs_id,
                "_is_recall": not isinstance(parsed, CatalogAction),
                "_arc_response": {
                    "kind": response.kind,
                    "exact": response.exact,
                    **response.metadata,
                },
            }
            if isinstance(parsed, RecallMetadataAction):
                message["_is_recall_meta"] = True
            messages.append(message)
        self._event(
            "action",
            action_kind=response.kind,
            obs_id=response.obs_id,
            chars=len(response.body),
            exact=response.exact,
        )
        return response

    def find_prior_executions(self, command: str) -> list[Any]:
        """Find records with the complete normalized command signature."""
        signature = normalise_cmd_signature(command, max_chars=None)
        if self._engine is not None:
            return self._engine.find_prior_executions(command)
        assert self.archive is not None
        return self.archive.find_by_signature(signature)

    def create_budget_guard(
        self,
        count_request: Callable[[Any], int],
    ) -> InvocationBudgetGuard:
        """Create a guard for the final provider dispatch boundary."""
        if self.config.context_window is None:
            raise ValueError("config.context_window is required for a provider guard")
        return InvocationBudgetGuard(
            context_window=self.config.context_window,
            completion_reserve=self.config.completion_reserve,
            count_request=count_request,
            serialization_margin=self.config.serialization_margin,
        )

    def status(self) -> dict[str, Any]:
        """Return body-free runtime state suitable for CLI and analyzers."""
        self._require_open()
        if self.archive is not None:
            stats = self.archive.stats()
            archive = {
                "path": self.archive.path,
                "observations": stats.observations,
                "body_bytes": stats.body_bytes,
                "recalled_observations": stats.recalled_observations,
                "recall_count": stats.recall_count,
            }
        else:
            assert self._engine is not None
            archive = {
                "path": self.config.archive_path,
                "observations": self._engine.store.n_records,
            }
        return {
            "profile": self.profile,
            "step": self.step,
            "archive": archive,
            "telemetry_events": len(self._events),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.archive is not None:
                self.archive.close()
            if self._engine is not None:
                self._engine.close()
            self._closed = True

    def _event(self, kind: str, **values: Any) -> None:
        if self.config.max_telemetry_events == 0:
            return
        with self._lock:
            self._events.append(
                {
                    "kind": kind,
                    "step": self.step,
                    "timestamp_ns": time.time_ns(),
                    **values,
                }
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("compaction runner is closed")

    def __enter__(self) -> "CompactionRunner":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def _message_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


def _summary_index(messages: list[dict[str, Any]], expected_index: int) -> int:
    if 0 <= expected_index < len(messages):
        # A hard cap smaller than SUMMARY_HEADER intentionally truncates the
        # header itself. The compactor's structural insertion point remains
        # authoritative in that case.
        return expected_index
    for index, message in enumerate(messages):
        if str(message.get("content", "")).startswith(SUMMARY_HEADER):
            return index
    raise BudgetExceeded("formal compaction did not produce a summary message")


__all__ = ["CompactionRunner", "sanitize_messages"]
