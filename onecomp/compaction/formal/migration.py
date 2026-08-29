"""Explicit migration from the experimental JSONL observation WAL.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .archive import SQLiteObservationArchive
from .errors import LegacyMigrationError
from .models import MigrationReport


def migrate_experimental_wal(
    source: str | Path,
    archive: SQLiteObservationArchive,
    *,
    strict: bool = True,
    dry_run: bool = False,
) -> MigrationReport:
    """Import every valid JSONL row and preserve old citations as aliases.

    Legacy stores content-addressed observations, so invocation multiplicity
    that was collapsed before WAL creation cannot be reconstructed. Distinct
    WAL rows are nevertheless retained as distinct formal occurrences. A
    deterministic source key makes interrupted imports safely resumable.
    """
    if not isinstance(archive, SQLiteObservationArchive):
        raise TypeError("archive must be a SQLiteObservationArchive")
    if not isinstance(strict, bool) or not isinstance(dry_run, bool):
        raise TypeError("strict and dry_run must be booleans")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"experimental WAL not found: {source_path}")

    raw = source_path.read_bytes()
    source_digest = hashlib.sha256(raw).hexdigest()
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    skipped = 0
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            text = raw_line.decode("utf-8")
            value = json.loads(text)
            row = _validate_legacy_row(value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            message = f"line {line_number}: {exc}"
            if strict:
                raise LegacyMigrationError(message) from exc
            errors.append(message)
            skipped += 1
            continue
        rows.append((line_number, row))

    if dry_run:
        aliases = len({str(row["obs_id"]).lower() for _, row in rows if row.get("obs_id")})
        return MigrationReport(
            source=str(source_path),
            destination=archive.path,
            loaded=len(rows),
            skipped=skipped,
            aliases_registered=aliases,
            errors=tuple(errors),
        )

    loaded = 0
    aliases_registered = 0
    seen_aliases: set[str] = set()
    for line_number, row in rows:
        source_key = f"legacy-jsonl:{source_digest}:{line_number}"
        try:
            record = archive.append(
                row["cmd_signature"],
                row["full_content"],
                source_key=source_key,
                returncode=row["returncode"],
                created_step=row["created_step"],
                extra={
                    **row["extra"],
                    "migration": {
                        "format": "experimental-jsonl-v1",
                        "source_sha256": source_digest,
                        "line": line_number,
                    },
                },
            )
            loaded += 1
            legacy_id = row.get("obs_id")
            if legacy_id:
                alias = str(legacy_id).lower()
                if alias not in seen_aliases:
                    archive.register_alias(alias, record.obs_id)
                    seen_aliases.add(alias)
                    aliases_registered += 1
        # Lenient migration is an explicit file-boundary recovery mode: one
        # malformed or conflicting row must not prevent later independent rows.
        except Exception as exc:  # pylint: disable=broad-exception-caught
            message = f"line {line_number}: {type(exc).__name__}: {exc}"
            if strict:
                raise LegacyMigrationError(message) from exc
            errors.append(message)
            skipped += 1

    return MigrationReport(
        source=str(source_path),
        destination=archive.path,
        loaded=loaded,
        skipped=skipped,
        aliases_registered=aliases_registered,
        errors=tuple(errors),
    )


def _validate_legacy_row(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("entry must be a JSON object")
    command = value.get("cmd_signature")
    content = value.get("full_content")
    if not isinstance(command, str) or not command:
        raise ValueError("cmd_signature must be a non-empty string")
    if not isinstance(content, str):
        raise TypeError("full_content must be a string")
    returncode = value.get("returncode", 0) or 0
    created_step = value.get("created_step", 0) or 0
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise TypeError("returncode must be an integer")
    if not isinstance(created_step, int) or isinstance(created_step, bool) or created_step < 0:
        raise ValueError("created_step must be a non-negative integer")
    extra = value.get("extra") or {}
    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a JSON object")
    try:
        normalized_extra = json.loads(json.dumps(dict(extra), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("extra must be JSON-serializable") from exc
    legacy_id = value.get("obs_id")
    if legacy_id is not None and (
        not isinstance(legacy_id, str)
        or not 4 <= len(legacy_id) <= 128
        or any(character not in "0123456789abcdefABCDEF" for character in legacy_id)
    ):
        raise ValueError("obs_id must be 4-128 hexadecimal characters when present")
    return {
        "obs_id": legacy_id,
        "cmd_signature": command,
        "full_content": content,
        "returncode": returncode,
        "created_step": created_step,
        "extra": normalized_extra,
    }
