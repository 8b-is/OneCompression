"""External model manifest loading shared by paper benchmark entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load model references without copying or owning model weights."""

    source = Path(path).expanduser()
    payload: Any = json.loads(source.read_text())
    entries = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{source} must contain a non-empty models list")
    normalized: list[dict[str, str]] = []
    for entry in entries:
        if isinstance(entry, str) and entry:
            normalized.append({"key": entry, "ref": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("ref"), str) and entry["ref"]:
            normalized.append({"key": str(entry.get("key") or entry["ref"]), "ref": entry["ref"]})
        else:
            raise ValueError(f"Invalid model entry in {source}: {entry!r}")
    return normalized
