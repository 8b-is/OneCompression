"""Opt-in runtime evidence for isolated vLLM correctness smokes.

The vLLM engine executes attention backends in a spawned worker.  Launcher
logs and registry messages alone therefore do not prove that a custom data
plane ran.  When :data:`RUNTIME_EVIDENCE_ENV` names an absolute JSONL path,
backend implementations append small records immediately after their custom
cache mutation or kernel launch.  Ordinary serving does no I/O because the
environment variable is unset.

This is deliberately an evidence channel, not telemetry.  It accepts no
arbitrary tensors or prompt text and fails closed when explicitly enabled.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Mapping

RUNTIME_EVIDENCE_ENV = "KVCOMP_VLLM_RUNTIME_EVIDENCE"
_MAX_RECORD_BYTES = 16 * 1024


def append_runtime_evidence(event: Mapping[str, Any]) -> None:
    """Append one bounded, JSON-native backend event when explicitly enabled."""

    raw_path = os.environ.get(RUNTIME_EVIDENCE_ENV)
    if raw_path is None:
        return
    path = Path(raw_path)
    if not path.is_absolute():
        raise RuntimeError(f"{RUNTIME_EVIDENCE_ENV} must name an absolute path")
    if not path.parent.is_dir():
        raise RuntimeError(f"runtime evidence parent does not exist: {path.parent}")
    payload = {
        "schema": {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1},
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        **dict(event),
    }
    try:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("runtime evidence must be finite JSON-native data") from exc
    if len(encoded) > _MAX_RECORD_BYTES:
        raise RuntimeError(
            f"runtime evidence record exceeds {_MAX_RECORD_BYTES} bytes: {len(encoded)}"
        )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, encoded)
        if written != len(encoded):
            raise RuntimeError(
                f"short runtime evidence write: wrote {written} of {len(encoded)} bytes"
            )
        os.fsync(fd)
    finally:
        os.close(fd)


def emit_runtime_evidence_once(owner: Any, marker: str, event: Mapping[str, Any]) -> None:
    """Emit an event at most once for one backend implementation instance."""

    attribute = f"_kvcomp_evidence_{marker}"
    if getattr(owner, attribute, False):
        return
    append_runtime_evidence(event)
    setattr(owner, attribute, True)


__all__ = [
    "RUNTIME_EVIDENCE_ENV",
    "append_runtime_evidence",
    "emit_runtime_evidence_once",
]
