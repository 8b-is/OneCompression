"""
Load the immutable preflight record attached to a paper benchmark job.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_provenance() -> dict | None:
    """Return the preflight JSON named by ``PAPER_PROVENANCE_PATH``.

    Ordinary library/tests need not set the variable.  Once a paper wrapper
    sets it, a missing, malformed, or schema-incompatible record is fatal.
    """
    value = os.environ.get("PAPER_PROVENANCE_PATH")
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Paper preflight record is unreadable: {path}") from exc
    if document.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported paper preflight schema in {path}")
    if not document.get("code", {}).get("commit"):
        raise RuntimeError(f"Paper preflight record has no code commit: {path}")
    actual_sha256 = _sha256(path)
    expected_sha256 = os.environ.get("PAPER_PROVENANCE_SHA256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(f"Paper preflight record changed after planning: {path}")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "record": document,
    }


def run_code_commit() -> str | None:
    """Return the fixed source commit for result identity, when available.

    Paper jobs always have an immutable preflight record.  The environment
    fallback is useful during the short interval before that record is loaded
    and for source-only launcher tests.  Ordinary exploratory runs return
    ``None`` and retain their historical config-only identity.
    """
    provenance = load_run_provenance()
    if provenance is not None:
        commit = provenance.get("record", {}).get("code", {}).get("commit")
        if not isinstance(commit, str) or not commit:
            raise RuntimeError("Paper preflight record has no usable code commit")
        return commit
    commit = os.environ.get("CODE_REPO_COMMIT")
    return commit if commit else None


def run_code_revision() -> str | None:
    """Return the immutable source identity used to fingerprint a run.

    A clean run normally uses its Git commit.  A run from a modified worktree
    uses the content-addressed revision stored in its provenance record.  This
    prevents two different source snapshots at the same commit from sharing a
    result fingerprint.
    """

    provenance = load_run_provenance()
    if provenance is not None:
        code = provenance.get("record", {}).get("code", {})
        revision = code.get("revision")
        if isinstance(revision, str) and revision:
            return revision
        commit = code.get("commit")
        if not isinstance(commit, str) or not commit:
            raise RuntimeError("Paper preflight record has no usable code revision")
        return commit
    revision = os.environ.get("CODE_REPO_REVISION")
    return revision if revision else run_code_commit()


def validate_model_provenance(model_path: str | Path) -> dict | None:
    """Validate a local checkpoint against the active paper provenance.

    Small metadata/tokenizer files are re-hashed. Large weight shards are
    checked by size and nanosecond mtime against the plan-time full SHA record,
    avoiding repeated multi-gigabyte reads in every benchmark job.
    """

    provenance = load_run_provenance()
    if provenance is None:
        return None
    root = Path(model_path).expanduser().resolve()
    records = provenance.get("record", {}).get("models", [])
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and Path(record["path"]).expanduser().resolve() == root
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Model is not uniquely attested by paper provenance: {root}")
    record = matches[0]
    checkpoint = record.get("checkpoint")
    artifacts = checkpoint.get("artifacts") if isinstance(checkpoint, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"Paper provenance has no checkpoint artifacts for {root}")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Malformed checkpoint artifact provenance for {root}")
        name = artifact.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise RuntimeError(f"Unsafe checkpoint artifact name in provenance: {name!r}")
        path = root / name
        try:
            file_stat = path.stat()
        except OSError as exc:
            raise RuntimeError(f"Attested checkpoint artifact is unavailable: {path}") from exc
        if file_stat.st_size != artifact.get("bytes"):
            raise RuntimeError(f"Checkpoint artifact size changed after planning: {path}")
        if artifact.get("verify_on_launch"):
            if _sha256(path) != artifact.get("sha256"):
                raise RuntimeError(f"Checkpoint artifact content changed after planning: {path}")
        elif file_stat.st_mtime_ns != artifact.get("mtime_ns"):
            raise RuntimeError(f"Checkpoint weight changed after planning: {path}")
    return record


def merge_run_provenance(existing, current) -> list[dict]:
    """Append one job record while deduplicating atomic result rewrites."""
    records = list(existing or [])
    if current is None:
        return records
    by_hash = {record.get("sha256"): record for record in records}
    by_hash[current["sha256"]] = current
    return list(by_hash.values())
