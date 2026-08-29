"""Provider-boundary integration with auditable final-request enforcement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Awaitable, Callable, Optional, TypeVar

from ..formal import InvocationBudgetGuard
from ..runner import CompactionRunner, sanitize_messages

ResultT = TypeVar("ResultT")


class GuardedProviderAdapter:
    """Prepare transcript-bearing requests and guard every model dispatch.

    A provider adapter should be the only object through which an integration
    invokes its model client. This makes the formal guarantee observable:
    ``audit().dispatched_requests`` equals the number of provider calls and no
    oversized final serialized request reaches the client.

    Args:
        runner: Formal compaction runner owning the tagged source transcript.
        count_request: Authoritative counter for the final provider request.
        message_field: Request field containing the provider message list.
        sanitize: Remove OneComp-private message tags from the outbound copy.
    """

    def __init__(
        self,
        runner: CompactionRunner,
        count_request: Callable[[Any], int],
        *,
        message_field: str = "messages",
        sanitize: bool = True,
    ) -> None:
        if not isinstance(runner, CompactionRunner):
            raise TypeError("runner must be a CompactionRunner")
        if runner.config.context_window is None:
            raise ValueError("runner.config.context_window is required")
        if not isinstance(message_field, str) or not message_field:
            raise ValueError("message_field must be a non-empty string")
        if not isinstance(sanitize, bool):
            raise TypeError("sanitize must be a boolean")
        self.runner = runner
        self.message_field = message_field
        self.sanitize = sanitize
        self.guard: InvocationBudgetGuard = runner.create_budget_guard(count_request)

    def prepare_request(
        self,
        request: dict[str, Any],
        *,
        compact_to_chars: Optional[int] = None,
    ) -> dict[str, Any]:
        """Return an outbound copy while preserving the tagged source list."""
        if not isinstance(request, dict):
            raise TypeError("request must be a dictionary")
        source = request.get(self.message_field)
        if not isinstance(source, list) or not all(isinstance(item, dict) for item in source):
            raise ValueError(f"request[{self.message_field!r}] must be a list of objects")
        if compact_to_chars is None:
            prepared = source
        else:
            prepared = self.runner.enforce_char_budget(source, compact_to_chars)
            if prepared is not source:
                # The host-visible transcript must actually forget the replaced
                # items. Mutate the owned list only after successful compaction.
                source[:] = prepared
                prepared = source
        # Copy the request exactly once. The previous implementation deep-copied
        # the complete message payload and then immediately discarded that copy,
        # doubling peak memory for image blocks and long tool observations.
        outbound = deepcopy(request)
        outbound[self.message_field] = (
            sanitize_messages(outbound[self.message_field])
            if self.sanitize
            else outbound[self.message_field]
        )
        return outbound

    def dispatch(
        self,
        request: dict[str, Any],
        invoke: Callable[[dict[str, Any]], ResultT],
        *,
        compact_to_chars: Optional[int] = None,
    ) -> ResultT:
        """Prepare, count, and synchronously dispatch one final request."""
        outbound = self.prepare_request(request, compact_to_chars=compact_to_chars)
        return self.guard.dispatch(outbound, invoke)

    async def dispatch_async(
        self,
        request: dict[str, Any],
        invoke: Callable[[dict[str, Any]], Awaitable[ResultT]],
        *,
        compact_to_chars: Optional[int] = None,
    ) -> ResultT:
        """Prepare, count, and asynchronously dispatch one final request."""
        outbound = self.prepare_request(request, compact_to_chars=compact_to_chars)
        return await self.guard.dispatch_async(outbound, invoke)

    def audit(self):
        """Return immutable provider-boundary coverage counters."""
        return self.guard.audit()


__all__ = ["GuardedProviderAdapter"]
