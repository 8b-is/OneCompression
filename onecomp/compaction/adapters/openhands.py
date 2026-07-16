"""Optional OpenHands SDK adapter for ARC compaction and recall.

This module is intentionally not imported by :mod:`onecomp.compaction.adapters`:
OneComp's core has no OpenHands dependency.  Import it explicitly in an
OpenHands environment (``pip install openhands-sdk``), then pass the condenser
to ``Agent`` and add ``OpenHandsArcRecallTool`` and ``OpenHandsArcListTool``
using the same ``runtime_id``.

The condenser emits a native OpenHands :class:`Condensation`; consequently the
forgotten events are genuinely removed from the model view.  The paired recall
tool shares the condenser's in-process lossless store, avoiding the common but
incorrect pattern where an MCP tool returns a shorter copy while the host keeps
its original private history unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
from collections.abc import Sequence
from typing import ClassVar, Literal, Optional
from uuid import uuid4

try:
    from openhands.sdk.context.condenser import RollingCondenser
    from openhands.sdk.context.condenser.base import (
        CondensationRequirement,
        NoCondensationAvailableException,
    )
    from openhands.sdk.context.condenser.utils import (
        get_suffix_length_for_token_reduction,
        get_total_token_count,
    )
    from openhands.sdk.context.view import View
    from openhands.sdk.event import Condensation
    from openhands.sdk.llm import LLM
    from openhands.sdk.tool import (
        Action,
        Observation,
        ToolAnnotations,
        ToolDefinition,
        ToolExecutor,
        register_tool,
    )
    from pydantic import Field, PrivateAttr, model_validator
except ImportError as exc:  # pragma: no cover - exercised only without the optional SDK
    raise ImportError(
        "onecomp.compaction.adapters.openhands requires the optional "
        "'openhands-sdk' package"
    ) from exc

from ..engine import ArcEngine, ArcEngineConfig, RecallError
from .messages import message_role, message_text

_RUNTIMES: weakref.WeakValueDictionary[str, "OpenHandsArcCondenser"] = (
    weakref.WeakValueDictionary()
)
_RECALLED_ID_RE = re.compile(r'obs_id="§([0-9a-fA-F]{4,40})"')
_TEXT_RECALL_HINT_RE = re.compile(
    r"\[to recall full output: issue `_recall §([0-9a-fA-F]{4,40})` "
    r"as your ONLY action — no output follows\]"
)


class OpenHandsArcCondenser(RollingCondenser):
    """Deterministic ARC condenser for OpenHands SDK 1.36+.

    Unlike an LLM summarizing condenser, this stores tool observations verbatim
    and emits bounded citations without making a second model call.  ``max_tokens``
    uses OpenHands' own provider-aware counter when an ``agent_llm`` is supplied.
    """

    runtime_id: str = Field(default_factory=lambda: uuid4().hex)
    max_size: int = Field(default=240, gt=0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    keep_first: int = Field(default=2, ge=0)
    minimum_progress: float = Field(default=0.1, gt=0.0, lt=1.0)
    summary_max_chars: int = Field(default=24_000, ge=0)
    result_max_chars: int = Field(default=500, ge=0)
    recall_per_step: int = Field(default=2, ge=0)
    recall_max_chars: int = Field(default=0, ge=0)
    wal_path: Optional[str] = None

    _engine: ArcEngine = PrivateAttr()
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @model_validator(mode="after")
    def _validate_window(self):
        if self.keep_first >= self.max_size // 2:
            raise ValueError("keep_first must be less than max_size // 2")
        return self

    def model_post_init(self, __context: object) -> None:
        config = ArcEngineConfig(
            keep_recent=0,
            result_max_chars=self.result_max_chars,
            summary_max_chars=self.summary_max_chars,
            recall_per_step=self.recall_per_step,
            # OpenHands owns event eviction through Condensation.  The core LRU
            # cannot mutate that private history, so do not maintain a phantom
            # external-recall budget here; max_tokens and recall_max_chars are
            # the authoritative context guards.
            recall_budget_chars=0,
            recall_max_chars=self.recall_max_chars,
            wal_path=self.wal_path,
        )
        self._engine = (
            ArcEngine.from_wal(self.wal_path, config)
            if self.wal_path is not None
            else ArcEngine(config)
        )
        _RUNTIMES[self.runtime_id] = self

    @property
    def engine(self) -> ArcEngine:
        """The shared engine, useful for telemetry and integration tests."""
        return self._engine

    def close(self) -> None:
        self._engine.close()
        _RUNTIMES.pop(self.runtime_id, None)

    def handles_condensation_requests(self) -> bool:
        return True

    def condense(self, view: View, agent_llm: Optional[LLM] = None):
        # OpenHands calls its condenser once before each model response.  This
        # gives multiple parallel arc_recall calls from that response one shared
        # recall_per_step budget.
        with self._lock:
            self._engine.begin_step()
            return super().condense(view, agent_llm=agent_llm)

    async def acondense(self, view: View, agent_llm: Optional[LLM] = None):
        return self.condense(view, agent_llm=agent_llm)

    def condensation_requirement(
        self, view: View, agent_llm: Optional[LLM] = None
    ) -> Optional[CondensationRequirement]:
        if view.unhandled_condensation_request:
            return CondensationRequirement.HARD
        if self.max_tokens is not None and agent_llm is not None:
            if get_total_token_count(view.events, agent_llm) > self.max_tokens:
                return CondensationRequirement.HARD
        if len(view) > self.max_size:
            return CondensationRequirement.SOFT
        return None

    def get_condensation(
        self, view: View, agent_llm: Optional[LLM] = None
    ) -> Condensation:
        forgotten, offset = self._get_forgotten_events(view, agent_llm)
        if not forgotten:
            raise NoCondensationAvailableException(
                "ARC found no manipulation-safe events"
            )
        if len(forgotten) < len(view) * self.minimum_progress:
            raise NoCondensationAvailableException(
                "ARC condensation would make less than minimum_progress"
            )
        return self._build_condensation(view, forgotten, offset)

    def hard_context_reset(
        self, view: View, agent_llm: Optional[LLM] = None  # noqa: ARG002
    ) -> Optional[Condensation]:
        if not view.events:
            return None
        return self._build_condensation(view, view.events, 0)

    def recall(self, obs_id: str, recall_kind: Literal["full", "meta"] = "full") -> str:
        """Serve one recall for :class:`OpenHandsArcRecallTool`."""
        if recall_kind not in {"full", "meta"}:
            raise ValueError("recall_kind must be 'full' or 'meta'")
        verb = "_recall" if recall_kind == "full" else "_recall_meta"
        scratch: list[dict] = []
        with self._lock:
            outcome = self._engine.handle_action(f"{verb} §{obs_id}", scratch)
        if outcome is None:  # pragma: no cover - constructed command is always a recall
            raise RecallError("ARC recall was not handled")
        return str(outcome.message["content"])

    def list_observations(
        self, query: str = "", cursor: int = 0, limit: int = 20
    ) -> str:
        """Return a bounded command/id index for handles absent from summary."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be an integer between 1 and 100")
        with self._lock:
            ids = self._engine.store.known_ids()
            if query:
                needle = query.casefold()
                ids = [
                    obs_id
                    for obs_id in ids
                    if (record := self._engine.store.get(obs_id)) is not None
                    and needle in record.cmd_signature.casefold()
                ]
            selected = list(reversed(ids))[cursor : cursor + limit]
            observations = []
            for obs_id in selected:
                record = self._engine.store.get(obs_id)
                if record is not None:
                    observations.append(
                        {
                            "obs_id": record.obs_id,
                            "cmd_signature": record.cmd_signature[:200],
                            "returncode": record.returncode,
                            "line_count": record.line_count,
                            "char_count": record.char_count,
                        }
                    )
            next_cursor = cursor + len(selected)
            return json.dumps(
                {
                    "observations": observations,
                    "next_cursor": next_cursor if next_cursor < len(ids) else None,
                    "total": len(ids),
                },
                ensure_ascii=False,
            )

    def _get_forgotten_events(
        self, view: View, agent_llm: Optional[LLM]
    ) -> tuple[Sequence, int]:
        suffix_lengths: list[int] = []
        if view.unhandled_condensation_request:
            suffix_lengths.append(len(view) // 2 - self.keep_first - 1)
        if len(view) > self.max_size:
            suffix_lengths.append(self.max_size // 2 - self.keep_first - 1)
        if self.max_tokens is not None and agent_llm is not None:
            total = get_total_token_count(view.events, agent_llm)
            if total > self.max_tokens:
                suffix_lengths.append(
                    get_suffix_length_for_token_reduction(
                        events=view.events[self.keep_first :],
                        llm=agent_llm,
                        token_reduction=total - self.max_tokens // 2,
                        base_events=view.events[: self.keep_first],
                    )
                )
        if not suffix_lengths:
            raise ValueError("no condensation reason")
        events_from_tail = max(0, min(suffix_lengths))
        naive_end = len(view) - events_from_tail
        start = view.manipulation_indices.find_next(self.keep_first)
        end = view.manipulation_indices.find_next(naive_end)
        return view[start:end], start

    def _build_condensation(
        self, view: View, forgotten_events: Sequence, summary_offset: int
    ) -> Condensation:
        by_id = {getattr(event, "id", ""): event for event in view.events}
        messages: list[dict] = [
            {"role": "system", "content": "OpenHands ARC condenser"},
            {"role": "user", "content": "Preserve the forgotten trajectory"},
        ]
        for event in forgotten_events:
            message = self._event_to_arc_message(event, by_id)
            if message is not None:
                messages.append(message)

        if len(messages) == 2:
            # An empty summary would claim successful condensation while
            # silently deleting every forgotten event.  Let OpenHands choose
            # another condenser/reset path instead.
            raise NoCondensationAvailableException(
                "ARC could not convert any forgotten event to an LLM message"
            )

        compacted = self._engine.compact(
            messages, summary_max_chars=self.summary_max_chars
        )
        summary = compacted[2]["content"] if len(compacted) > 2 else ""
        summary = _TEXT_RECALL_HINT_RE.sub(
            lambda match: (
                "[to recall full output: call the `arc_recall` tool with "
                f'obs_id="{match.group(1)}" and recall_kind="full"]'
            ),
            summary,
        )
        digest = hashlib.sha1(
            "\x1f".join(
                str(getattr(event, "id", "")) for event in forgotten_events
            ).encode()
        ).hexdigest()[:16]
        return Condensation(
            forgotten_event_ids={str(event.id) for event in forgotten_events},
            summary=summary,
            summary_offset=summary_offset,
            llm_response_id=f"arc-{digest}",
        )

    def _event_to_arc_message(self, event, by_id: dict[str, object]) -> Optional[dict]:
        try:
            llm_message = event.to_llm_message()
        except (AttributeError, NotImplementedError):
            return None
        payload = llm_message.model_dump(mode="json", exclude_none=True)
        role = message_role(payload)
        text = message_text(payload)
        if not role:
            return None

        if role == "assistant":
            command = _event_command(event)
            if command:
                payload["content"] = f"{text}\n\n```bash\n{command}\n```"
            return payload

        if role == "tool":
            recalled = _RECALLED_ID_RE.search(text)
            if recalled and self._engine.store.has(recalled.group(1)):
                payload["_obs_id"] = recalled.group(1).lower()
                return payload
            parent = by_id.get(str(getattr(event, "action_id", "")))
            command = _event_command(parent) or str(getattr(event, "tool_name", "tool"))
            observation = getattr(event, "observation", None)
            returncode = 1 if bool(getattr(observation, "is_error", False)) else 0
            record = self._engine.record_observation(
                command=command,
                content=text,
                returncode=returncode,
                extra={
                    "openhands_event_id": str(getattr(event, "id", "")),
                    "tool_name": str(getattr(event, "tool_name", "")),
                },
            )
            payload["_obs_id"] = record.obs_id
        return payload


def _event_command(event: object) -> str:
    if event is None:
        return ""
    tool_call = getattr(event, "tool_call", None)
    if tool_call is not None:
        name = str(getattr(tool_call, "name", getattr(event, "tool_name", "tool")))
        arguments = getattr(tool_call, "arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"{name}({arguments})"
    return str(getattr(event, "tool_name", ""))


class OpenHandsArcRecallAction(Action):
    obs_id: str = Field(description="ARC observation id, without the section sign")
    recall_kind: Literal["full", "meta"] = "full"


class OpenHandsArcRecallObservation(Observation):
    pass


class _OpenHandsArcRecallExecutor(
    ToolExecutor[OpenHandsArcRecallAction, OpenHandsArcRecallObservation]
):
    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id

    def __call__(
        self, action: OpenHandsArcRecallAction, conversation=None
    ):  # noqa: ARG002
        runtime = _RUNTIMES.get(self.runtime_id)
        if runtime is None:
            return OpenHandsArcRecallObservation.from_text(
                "ARC runtime is unavailable; resume the condenser with its WAL first.",
                is_error=True,
            )
        try:
            content = runtime.recall(action.obs_id, action.recall_kind)
        except (RecallError, ValueError) as exc:
            return OpenHandsArcRecallObservation.from_text(str(exc), is_error=True)
        return OpenHandsArcRecallObservation.from_text(content)


class OpenHandsArcRecallTool(
    ToolDefinition[OpenHandsArcRecallAction, OpenHandsArcRecallObservation]
):
    """OpenHands tool sharing the matching condenser's ARC store."""

    name: ClassVar[str] = "arc_recall"

    @classmethod
    def create(cls, conv_state=None, **params):  # noqa: ARG003
        runtime_id = params.pop("runtime_id", None)
        if params:
            raise ValueError(
                f"Unsupported OpenHandsArcRecallTool params: {sorted(params)}"
            )
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("OpenHandsArcRecallTool requires runtime_id")
        return [
            cls(
                description=(
                    "Recall a tool observation preserved by the ARC condenser. Use an "
                    "obs_id shown in a § citation; choose meta for a cheap preview or "
                    "full when the exact body is needed."
                ),
                action_type=OpenHandsArcRecallAction,
                observation_type=OpenHandsArcRecallObservation,
                executor=_OpenHandsArcRecallExecutor(runtime_id),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


class OpenHandsArcListAction(Action):
    query: str = Field(
        default="",
        description="Case-insensitive command/path substring; empty lists newest records.",
    )
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class OpenHandsArcListObservation(Observation):
    pass


class _OpenHandsArcListExecutor(
    ToolExecutor[OpenHandsArcListAction, OpenHandsArcListObservation]
):
    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id

    def __call__(
        self, action: OpenHandsArcListAction, conversation=None
    ):  # noqa: ARG002
        runtime = _RUNTIMES.get(self.runtime_id)
        if runtime is None:
            return OpenHandsArcListObservation.from_text(
                "ARC runtime is unavailable; resume the condenser with its WAL first.",
                is_error=True,
            )
        try:
            content = runtime.list_observations(
                action.query, action.cursor, action.limit
            )
        except ValueError as exc:
            return OpenHandsArcListObservation.from_text(str(exc), is_error=True)
        return OpenHandsArcListObservation.from_text(content)


class OpenHandsArcListTool(
    ToolDefinition[OpenHandsArcListAction, OpenHandsArcListObservation]
):
    """Bounded command/id index sharing the matching condenser's store."""

    name: ClassVar[str] = "arc_list_observations"

    @classmethod
    def create(cls, conv_state=None, **params):  # noqa: ARG003
        runtime_id = params.pop("runtime_id", None)
        if params:
            raise ValueError(
                f"Unsupported OpenHandsArcListTool params: {sorted(params)}"
            )
        if not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("OpenHandsArcListTool requires runtime_id")
        return [
            cls(
                description=(
                    "Search the bounded ARC observation index by original command/path. "
                    "Use when a strict summary cap leaves bare or omitted handles, then "
                    "call arc_recall with the returned obs_id."
                ),
                action_type=OpenHandsArcListAction,
                observation_type=OpenHandsArcListObservation,
                executor=_OpenHandsArcListExecutor(runtime_id),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool("OpenHandsArcRecallTool", OpenHandsArcRecallTool)
register_tool("OpenHandsArcListTool", OpenHandsArcListTool)


__all__ = [
    "OpenHandsArcCondenser",
    "OpenHandsArcListAction",
    "OpenHandsArcListObservation",
    "OpenHandsArcListTool",
    "OpenHandsArcRecallAction",
    "OpenHandsArcRecallObservation",
    "OpenHandsArcRecallTool",
]
