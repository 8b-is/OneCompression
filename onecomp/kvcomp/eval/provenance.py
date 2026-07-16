"""Load the immutable preflight record attached to a paper benchmark job."""

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
    return {
        "path": str(path),
        "sha256": _sha256(path),
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


def merge_run_provenance(existing, current) -> list[dict]:
    """Append one job record while deduplicating atomic result rewrites."""
    records = list(existing or [])
    if current is None:
        return records
    by_hash = {record.get("sha256"): record for record in records}
    by_hash[current["sha256"]] = current
    return list(by_hash.values())
