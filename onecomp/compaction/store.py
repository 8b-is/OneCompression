"""Lossless, ID-addressable store for agent observations.

The :class:`ObsStore` is a pure, in-memory key-value store that maps a
short stable hash of ``(cmd_signature, full_content)`` to an
:class:`ObsRecord` holding the **verbatim** observation.  It has no
dependency on any agent framework; an agent (or the
:class:`onecomp.compaction.engine.ArcEngine` facade) calls ``put()``
after each action execution and ``get()`` when handling a ``_recall``
request.

Design notes
------------

*Determinism.*  The ``obs_id`` is derived from a SHA-1 of the command
signature plus the full content.  Identical observations from identical
commands collapse to the same id (idempotent re-store).

*No eviction in the store itself.*  The store is append-only by design;
what the LLM *sees* in its working context is managed by the caller
(LRU on the in-context recall set, see ``ArcEngine``).  Decoupling means
even an evicted recall can be re-issued cheaply.

*Write-ahead log (WAL).*  Pass ``wal_path`` to :meth:`__init__` to
enable crash-safe persistence.  Each new record is appended as a JSON
line after it is inserted into the in-memory dict; :meth:`from_wal`
reconstructs a store from a previously written WAL (e.g. to resume an
interrupted agent run).

Ported from ARC (Addressable Recall for Compaction), Apache-2.0.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

logger = logging.getLogger(__name__)


@dataclass
class ObsRecord:
    """A single lossless observation entry.

    All fields are filled at insertion time; the record is immutable
    except for the recall bookkeeping fields (``last_recalled``,
    ``n_recalls``) which are updated by the caller on each ``_recall``
    event.
    """

    obs_id: str
    cmd_signature: str
    full_content: str
    head: str
    tail: str
    line_count: int
    char_count: int
    returncode: int
    created_step: int
    last_recalled: int = 0
    n_recalls: int = 0
    extra: dict = field(default_factory=dict)


def _mint_obs_id(cmd_signature: str, full_content: str, *, hex_chars: int = 8) -> str:
    """Stable 8-hex (default) hash of ``(cmd_signature, full_content)``.

    SHA-1 is used for speed, not security.  With <= 10^4 observations per
    agent instance the collision probability at 8 hex chars (32 bits) is
    ~1e-5 per pair; on collision :meth:`ObsStore.put` falls back to a
    longer prefix.
    """
    digest = hashlib.sha1(
        cmd_signature.encode("utf-8", errors="replace")
        + b"\x1f"
        + full_content.encode("utf-8", errors="replace"),
    ).hexdigest()
    return digest[:hex_chars]


class ObsStore:
    """In-memory ``obs_id -> ObsRecord`` mapping with a secondary index
    by command signature.

    The store is intentionally minimal: ``put``, ``get``, ``has``, and
    ``find_by_signature`` are the core operations.  All policy lives in
    the caller.

    Args:
        obs_id_hex_chars: Length of the SHA-1 hex prefix used as
            ``obs_id`` (4-40).
        wal_path: Optional path to a write-ahead log file.  When set,
            every new record is appended as a newline-delimited JSON
            line immediately after being inserted into the in-memory
            dict.  Call :meth:`close` (or use the store as a context
            manager) when the agent exits cleanly, and pass the same
            path to :meth:`from_wal` to reconstruct the store on resume.
    """

    def __init__(
        self,
        *,
        obs_id_hex_chars: int = 8,
        wal_path: Optional[Path | str] = None,
    ) -> None:
        if not isinstance(obs_id_hex_chars, int) or isinstance(obs_id_hex_chars, bool):
            raise ValueError("obs_id_hex_chars must be an integer between 4 and 40")
        if obs_id_hex_chars < 4 or obs_id_hex_chars > 40:
            raise ValueError("obs_id_hex_chars must be between 4 and 40")
        self._obs_id_hex_chars = obs_id_hex_chars
        self._records: dict[str, ObsRecord] = {}
        self._by_signature: dict[str, list[str]] = {}
        self._wal: Optional[IO[str]] = None
        if wal_path is not None:
            self._open_wal(wal_path)

    def _open_wal(self, wal_path: Path | str) -> None:
        """Attach an append-only WAL after construction or replay."""
        if self._wal is not None:
            raise RuntimeError("ObsStore WAL is already open")
        if not isinstance(wal_path, (str, os.PathLike)) or isinstance(wal_path, bool):
            raise ValueError("wal_path must be a filesystem path")
        if isinstance(wal_path, str) and not wal_path.strip():
            raise ValueError("wal_path must be a non-empty filesystem path")
        path = Path(wal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Observations routinely contain source, logs, and credentials.  Do
        # not let a permissive process umask create a world-readable WAL.
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.chmod(path, 0o600)
            self._wal = os.fdopen(fd, "a", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise

    @property
    def n_records(self) -> int:
        return len(self._records)

    @property
    def max_created_step(self) -> int:
        """Greatest persisted creation step, or zero for an empty store."""
        return max((record.created_step for record in self._records.values()), default=0)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, obs_id: str) -> bool:
        return self.has(obs_id)

    def __enter__(self) -> "ObsStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the WAL file handle, if open.

        Safe to call multiple times.  The in-memory records are
        unaffected; the store remains usable after closing (no further
        WAL appends will occur).
        """
        if self._wal is not None:
            try:
                self._wal.flush()
                self._wal.close()
            except OSError as exc:
                logger.warning("ObsStore.close: error closing WAL: %s", exc)
            finally:
                self._wal = None

    @classmethod
    def from_wal(
        cls,
        wal_path: Path | str,
        *,
        obs_id_hex_chars: int = 8,
        new_wal_path: Optional[Path | str] = None,
    ) -> "ObsStore":
        """Reconstruct a store by replaying a write-ahead log.

        Lines that are empty or malformed (e.g. from a crash mid-write)
        are skipped with a warning.

        Args:
            wal_path: Path to the WAL file produced by a previous store.
            obs_id_hex_chars: ID length for the reconstructed store.
                Should match the original; mismatches make citations
                unresolvable.
            new_wal_path: If set, the reconstructed store appends new
                records to this path (may equal ``wal_path`` to continue
                the same file).
        """
        # Replay with WAL writes disabled.  Opening ``new_wal_path`` before
        # replay duplicates every recovered row when it equals ``wal_path``.
        # Attach the append target only after all historical rows are loaded.
        store = cls(obs_id_hex_chars=obs_id_hex_chars)
        wal_path = Path(wal_path)
        if not wal_path.exists():
            logger.info("ObsStore.from_wal: %s does not exist; returning empty store", wal_path)
            if new_wal_path is not None:
                store._open_wal(new_wal_path)
            return store
        loaded = skipped = 0
        # Stream the replay so a large lossless store is not duplicated in one
        # giant ``read_text`` allocation before records are reconstructed.
        with wal_path.open("r", encoding="utf-8", errors="replace") as source:
            for lineno, line in enumerate(source, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "ObsStore.from_wal: line %d malformed (skipped): %r",
                        lineno,
                        line[:80],
                    )
                    skipped += 1
                    continue
                try:
                    store.put(
                        cmd_signature=entry["cmd_signature"],
                        full_content=entry["full_content"],
                        returncode=int(entry.get("returncode", 0) or 0),
                        created_step=int(entry.get("created_step", 0) or 0),
                        extra=entry.get("extra") or {},
                    )
                    loaded += 1
                except (KeyError, TypeError, ValueError, AttributeError) as exc:
                    logger.warning(
                        "ObsStore.from_wal: line %d bad entry (skipped): %s", lineno, exc
                    )
                    skipped += 1
        logger.info(
            "ObsStore.from_wal: loaded %d records, skipped %d from %s", loaded, skipped, wal_path
        )
        if new_wal_path is not None:
            store._open_wal(new_wal_path)
        return store

    def has(self, obs_id: str) -> bool:
        return obs_id is not None and obs_id.lower() in self._records

    def get(self, obs_id: str) -> Optional[ObsRecord]:
        """Look up by ``obs_id``, case-insensitively.

        Minted ids are always lowercase SHA-1 hex; the input is
        lowercased so an LLM that retypes the id in upper or mixed case
        still resolves to the same record.
        """
        if obs_id is None:
            return None
        return self._records.get(obs_id.lower())

    def known_ids(self) -> list[str]:
        """Canonical obs_ids currently in the store, in insertion order."""
        return list(self._records.keys())

    def contains_value(self, value: str) -> bool:
        """True if any stored observation contains ``value`` verbatim.

        Useful for anti-confabulation guards: refuse an agent answer
        that never appeared in any observation.  The empty string
        trivially appears everywhere; return False to keep the predicate
        sensible at the boundary.
        """
        if not value:
            return False
        return any(value in rec.full_content for rec in self._records.values())

    def contains_value_with_prefix(self, prefix: str, value: str) -> bool:
        """True only if ``prefix + value`` appears verbatim in any observation.

        Stricter than :meth:`contains_value`: eliminates false-passes
        where a short token coincidentally occurs as a substring of
        noisy observation content.
        """
        if not prefix or not value:
            return False
        phrase = prefix + value
        return any(phrase in rec.full_content for rec in self._records.values())

    def find_records_containing(self, value: str, limit: int = 5) -> list[ObsRecord]:
        """Up to ``limit`` records whose ``full_content`` contains ``value``."""
        if not value:
            return []
        out: list[ObsRecord] = []
        for rec in self._records.values():
            if value in rec.full_content:
                out.append(rec)
                if len(out) >= limit:
                    break
        return out

    def find_by_signature(self, cmd_signature: str) -> list[ObsRecord]:
        """All records produced by an equivalent command signature.

        Enables re-execution detection: if the agent is about to run a
        command whose signature already produced a stored observation,
        the caller can opt to recall instead of re-execute.
        """
        return [self._records[oid] for oid in self._by_signature.get(cmd_signature, [])]

    def put(
        self,
        *,
        cmd_signature: str,
        full_content: str,
        returncode: int = 0,
        created_step: int = 0,
        head_chars: int = 200,
        tail_chars: int = 200,
        extra: Optional[dict] = None,
    ) -> ObsRecord:
        """Insert (or return the existing record for) an observation.

        Idempotent on identical ``(cmd_signature, full_content)``.  On a
        hash prefix collision with *different* content, the obs_id is
        extended until uniqueness is guaranteed.
        """
        if not isinstance(cmd_signature, str):
            raise ValueError("cmd_signature must be a string")
        if not isinstance(full_content, str):
            raise ValueError("full_content must be a string")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise ValueError("returncode must be an integer")
        if not isinstance(created_step, int) or isinstance(created_step, bool) or created_step < 0:
            raise ValueError("created_step must be a non-negative integer")
        for name, value in (("head_chars", head_chars), ("tail_chars", tail_chars)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if extra is not None:
            if not isinstance(extra, dict):
                raise ValueError("extra must be a JSON-serialisable object or None")
            try:
                json.dumps(extra, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("extra must be JSON-serialisable") from exc

        obs_id = _mint_obs_id(cmd_signature, full_content, hex_chars=self._obs_id_hex_chars)
        if obs_id in self._records:
            existing = self._records[obs_id]
            if existing.cmd_signature == cmd_signature and existing.full_content == full_content:
                return existing
            extended_len = self._obs_id_hex_chars
            while extended_len < 40:
                extended_len += 4
                obs_id = _mint_obs_id(cmd_signature, full_content, hex_chars=extended_len)
                existing = self._records.get(obs_id)
                if existing is None:
                    break
                # A record that previously collided at the short prefix was
                # minted with this longer id.  Re-storing it must remain
                # idempotent instead of extending again and creating a
                # duplicate record on every replay/retry.
                if (
                    existing.cmd_signature == cmd_signature
                    and existing.full_content == full_content
                ):
                    return existing
            else:  # pragma: no cover - astronomically unlikely
                raise RuntimeError("ObsStore: could not mint unique obs_id at 40 hex chars")

        head = full_content[:head_chars]
        tail = full_content[-tail_chars:] if len(full_content) > head_chars else ""
        if full_content:
            line_count = full_content.count("\n") + (0 if full_content.endswith("\n") else 1)
        else:
            line_count = 0
        record = ObsRecord(
            obs_id=obs_id,
            cmd_signature=cmd_signature,
            full_content=full_content,
            head=head,
            tail=tail,
            line_count=line_count,
            char_count=len(full_content),
            returncode=returncode,
            created_step=created_step,
            extra=dict(extra) if extra else {},
        )
        self._records[obs_id] = record
        self._by_signature.setdefault(cmd_signature, []).append(obs_id)
        self._wal_append(record)
        return record

    def _wal_append(self, record: ObsRecord) -> None:
        """Append a single record to the WAL as one JSON line.

        Called only for genuinely new records (not idempotent
        re-stores).  A ``flush()`` is issued after every write so a
        subsequent crash leaves all prior writes durable against Python
        crashes; ``fsync`` is deliberately omitted for throughput.
        """
        if self._wal is None:
            return
        entry = {
            "obs_id": record.obs_id,
            "cmd_signature": record.cmd_signature,
            "full_content": record.full_content,
            "returncode": record.returncode,
            "created_step": record.created_step,
            "extra": record.extra,
        }
        try:
            self._wal.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._wal.flush()
        except OSError as exc:
            logger.error(
                "ObsStore._wal_append: failed to write WAL (obs_id=%s): %s — "
                "in-memory store is intact but this record will be missing "
                "from any WAL-based resume.",
                record.obs_id,
                exc,
            )

    def to_serializable(self) -> list[dict]:
        """JSON-friendly lossless snapshot (includes verbatim contents)."""
        return [
            {
                "obs_id": r.obs_id,
                "cmd_signature": r.cmd_signature,
                "full_content": r.full_content,
                "line_count": r.line_count,
                "char_count": r.char_count,
                "returncode": r.returncode,
                "created_step": r.created_step,
                "last_recalled": r.last_recalled,
                "n_recalls": r.n_recalls,
                "extra": r.extra,
            }
            for r in self._records.values()
        ]
