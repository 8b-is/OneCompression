"""Process-local leases for persistent MCP compaction archives.

SQLite serializes writes, but two live agent sessions must not share one
archive because their recall budgets and message histories are independent.
This module owns that process-local exclusivity rule and releases a lease when
the corresponding compaction resource closes.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import threading
from typing import Dict, Optional

from ..compaction import ArcEngine, CompactionRunner

_ACTIVE_ARCHIVE_LOCK = threading.RLock()
_ACTIVE_ARCHIVES: Dict[str, "ArchiveLease"] = {}


class ArchiveLease:
    """One idempotently releasable in-process archive lease."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._released = False

    def release(self) -> None:
        """Release this lease once, even when close paths race."""
        with _ACTIVE_ARCHIVE_LOCK:
            if self._released:
                return
            if _ACTIVE_ARCHIVES.get(self.path) is self:
                del _ACTIVE_ARCHIVES[self.path]
            self._released = True


class _LeasedArchiveMixin:
    _archive_lease: Optional[ArchiveLease] = None

    def attach_archive_lease(self, lease: Optional[ArchiveLease]) -> None:
        """Attach the lease acquired before this resource was constructed."""
        self._archive_lease = lease

    def close(self) -> None:
        """Close the resource and always release its process-local lease."""
        try:
            super().close()
        finally:
            if self._archive_lease is not None:
                self._archive_lease.release()


class ManagedCompactionRunner(_LeasedArchiveMixin, CompactionRunner):
    """Formal session runner that releases its archive lease on close."""


class ManagedArcEngine(_LeasedArchiveMixin, ArcEngine):
    """Compatibility engine that releases its archive lease on close."""


def reserve_archive(path: Optional[str]) -> Optional[ArchiveLease]:
    """Reserve ``path`` for one active MCP session in this process."""
    if path is None:
        return None
    with _ACTIVE_ARCHIVE_LOCK:
        if path in _ACTIVE_ARCHIVES:
            raise ValueError(
                "wal_path is already active in another compaction session; "
                "close that session before resuming it"
            )
        lease = ArchiveLease(path)
        _ACTIVE_ARCHIVES[path] = lease
        return lease
