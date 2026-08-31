"""Portable path and model resolution for benchmark entry points.

The benchmark repository deliberately does not own model weights.  A model
argument may therefore be a local path, a path relative to ``--models-root``,
or a Hugging Face model id.  Output paths default to a repository-relative
directory and can always be overridden with an argument or environment
variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root(anchor: Path | None = None) -> Path:
    """Return the repository root without relying on a machine-specific path."""

    if anchor is None:
        anchor = Path(__file__)
    # paths.py lives in benchmark/kvcomp_common.
    return anchor.resolve().parents[2]


def _looks_like_path(value: str) -> bool:
    return (
        value.startswith((".", "/", "~")) or os.sep in value or (os.altsep and os.altsep in value)
    )


def resolve_model_ref(value: str | Path, models_root: str | Path | None = None) -> str:
    """Resolve a model reference while keeping external model ownership.

    Existing paths are returned as absolute paths.  A relative path is first
    resolved from the current working directory, then from ``models_root`` if
    supplied.  Non-path values (for example ``meta-llama/Llama-3.1-8B``) are
    returned unchanged so Hugging Face loaders can resolve them.
    """

    raw = os.path.expandvars(os.path.expanduser(str(value)))
    candidate = Path(raw)
    if candidate.exists():
        return str(candidate.resolve())
    if models_root is not None:
        rooted = Path(os.path.expandvars(os.path.expanduser(str(models_root)))) / candidate
        if rooted.exists():
            return str(rooted.resolve())
    if _looks_like_path(raw):
        # Preserve an explicit missing path in the error message; do not invent
        # a repository-internal model directory.
        return str(candidate)
    return raw


def resolve_output_root(
    value: str | Path | None = None,
    *,
    env_name: str = "KVCOMP_OUTPUT_ROOT",
    default_relative: str = "results",
    root: Path | None = None,
) -> Path:
    """Resolve output directory from CLI, environment, or repo-relative default."""

    raw = value if value is not None else os.environ.get(env_name)
    if raw is None or str(raw).strip() == "":
        return (root or repo_root()) / default_relative
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()


def resolve_python(value: str | Path | None = None, *, env_name: str = "KVCOMP_PYTHON") -> Path:
    """Resolve the interpreter, defaulting to the active environment."""

    raw = value if value is not None else os.environ.get(env_name)
    return Path(raw).expanduser() if raw else Path(sys.executable)
