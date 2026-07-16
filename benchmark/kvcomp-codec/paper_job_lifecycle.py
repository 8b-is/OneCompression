#!/usr/bin/env python3
"""Validate the paper-results worktree and write fail-closed job receipts.

Scientific outputs, scheduler logs, provenance, and terminal success/failure
receipts must be born inside a *separate linked Git worktree* of this
repository.  The source worktree can therefore remain clean and fixed while a
campaign is running, without relying on a later manual archive operation to
make failed attempts auditable.

The companion shell launchers use two immutable identities for every attempt:
``phase/job-key/attempt-id/submission.json`` and ``worker.json``.  Receipts are
updated atomically through a small, explicit state machine.  A worker may be
marked successful only after its expected JSON result has been independently
opened with duplicate-key and non-finite-number rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RESULTS_RELATIVE_ROOT = Path("papers/hayatequant_paper/experiments/results")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}\Z")
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
SLURM_ID_RE = re.compile(r"[0-9]+(?:[.;_+][A-Za-z0-9._+-]+)?\Z")
ALLOWED_PHASES = frozenset({"smoke", "formal"})
ALLOWED_KINDS = frozenset({"submission", "worker"})
ALLOWED_STATUSES = {
    "submission": frozenset(
        {"submission_started", "submitted", "submission_failed", "skipped_existing"}
    ),
    "worker": frozenset({"worker_started", "succeeded", "failed"}),
}
ALLOWED_TRANSITIONS = {
    "submission": {
        None: frozenset({"submission_started"}),
        "submission_started": frozenset(
            {"submitted", "submission_failed", "skipped_existing"}
        ),
    },
    "worker": {
        None: frozenset({"worker_started"}),
        "worker_started": frozenset({"succeeded", "failed"}),
    },
}
TERMINAL_STATUSES = frozenset(
    {"submitted", "submission_failed", "skipped_existing", "succeeded", "failed"}
)


class LifecycleError(RuntimeError):
    """Raised when a result root or lifecycle transition is not auditable."""


class _DuplicateKeyError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_identifier(value: str, *, label: str, pattern: re.Pattern[str] = SAFE_ID_RE) -> str:
    if pattern.fullmatch(value) is None:
        raise LifecycleError(f"unsafe {label}: {value!r}")
    return value


def _validate_commit(value: str) -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise LifecycleError("source commit must be exactly 40 lowercase hexadecimal digits")
    return value


def _git(cwd: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ("git", "-C", str(cwd), *arguments),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise LifecycleError(f"cannot execute git: {exc}") from exc
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise LifecycleError(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout.strip()


def _git_path(cwd: Path, argument: str) -> Path:
    raw = Path(_git(cwd, "rev-parse", argument))
    if not raw.is_absolute():
        raw = cwd / raw
    return raw.resolve(strict=True)


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _ensure_existing_directory(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise LifecycleError(f"{label} is not an existing directory: {lexical}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise LifecycleError(f"{label} must be a real directory, not a symlink: {lexical}")
    return lexical.resolve(strict=True)


def _assert_no_existing_symlink(root: Path, destination: Path, *, label: str) -> None:
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise LifecycleError(f"{label} escapes its repository root: {destination}") from exc
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise LifecycleError(f"cannot inspect {label} component {cursor}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LifecycleError(f"symlinked {label} component is forbidden: {cursor}")


def validate_results_root(
    *, source_repo: Path, run_root: Path, expected_commit: str
) -> dict[str, str]:
    """Return the canonical linked-worktree identity for ``run_root``.

    ``run_root`` must be one direct run directory below the repository's
    tracked HayateQuant results location.  The containing worktree must share
    the source repository's Git common directory, be distinct from the source
    worktree, and be checked out at the exact campaign commit.
    """

    commit = _validate_commit(expected_commit)
    source = _ensure_existing_directory(source_repo, label="source repository")
    source_git_root = Path(_git(source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if source_git_root != source:
        raise LifecycleError(f"--source-repo is not a Git worktree root: {source}")

    run = _lexical_absolute(run_root)
    _validate_identifier(run.name, label="paper run ID", pattern=RUN_ID_RE)
    results_directory = run.parent
    result_worktree = results_directory
    for _ in RESULTS_RELATIVE_ROOT.parts:
        result_worktree = result_worktree.parent
    result_worktree = _ensure_existing_directory(result_worktree, label="results worktree")
    result_git_root = Path(
        _git(result_worktree, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    if result_git_root != result_worktree:
        raise LifecycleError(
            "RUN_ROOT is not below the root of a Git worktree at the required repository path"
        )
    expected_results_directory = result_worktree / RESULTS_RELATIVE_ROOT
    expected_results_directory = _ensure_existing_directory(
        expected_results_directory, label="repository HayateQuant results directory"
    )
    if results_directory.resolve(strict=True) != expected_results_directory:
        raise LifecycleError(
            f"RUN_ROOT must be a direct child of {expected_results_directory}, got {run}"
        )
    if result_worktree == source_git_root:
        raise LifecycleError(
            "paper RUN_ROOT must use a separate linked Git worktree, not the clean source worktree"
        )
    source_common = _git_path(source_git_root, "--git-common-dir")
    result_common = _git_path(result_git_root, "--git-common-dir")
    if source_common != result_common:
        raise LifecycleError(
            "results worktree is not linked to the same Git repository as the source worktree"
        )
    source_head = _git(source_git_root, "rev-parse", "HEAD")
    result_head = _git(result_git_root, "rev-parse", "HEAD")
    if source_head != commit:
        raise LifecycleError(
            f"source worktree HEAD {source_head} does not match campaign commit {commit}"
        )
    if result_head != commit:
        raise LifecycleError(
            f"results worktree HEAD {result_head} does not match campaign commit {commit}"
        )
    _git(result_git_root, "cat-file", "-e", f"{commit}^{{commit}}")
    _assert_no_existing_symlink(result_worktree, run, label="RUN_ROOT")
    if run.exists():
        run_metadata = run.lstat()
        if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
            raise LifecycleError(f"RUN_ROOT must be a real directory when present: {run}")
    return {
        "source_worktree": str(source_git_root),
        "results_worktree": str(result_git_root),
        "git_common_directory": str(source_common),
        "run_root": str(run),
        "run_id": run.name,
        "source_commit": commit,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise LifecycleError(f"invalid {label} JSON at {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_repo_path(path: Path, *, run_root: Path, label: str) -> str:
    lexical = _lexical_absolute(path)
    try:
        return lexical.relative_to(run_root).as_posix()
    except ValueError as exc:
        raise LifecycleError(f"{label} must stay below RUN_ROOT: {lexical}") from exc


def _atomic_json(path: Path, document: Mapping[str, Any], *, run_root: Path) -> None:
    payload = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _assert_no_existing_symlink(run_root, path, label="receipt path")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_no_existing_symlink(run_root, path, label="receipt path")
    if path.exists() and path.is_symlink():
        raise LifecycleError(f"refusing symlinked receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt(arguments: argparse.Namespace) -> dict[str, Any]:
    identity = validate_results_root(
        source_repo=arguments.source_repo,
        run_root=arguments.run_root,
        expected_commit=arguments.source_commit,
    )
    run_root = Path(identity["run_root"])
    phase = _validate_identifier(arguments.phase, label="phase")
    if phase not in ALLOWED_PHASES:
        raise LifecycleError(f"unsupported phase: {phase!r}")
    kind = arguments.kind
    if kind not in ALLOWED_KINDS:
        raise LifecycleError(f"unsupported receipt kind: {kind!r}")
    status = arguments.status
    if status not in ALLOWED_STATUSES[kind]:
        raise LifecycleError(f"unsupported {kind} status: {status!r}")
    job_key = _validate_identifier(arguments.job_key, label="job key")
    attempt_id = _validate_identifier(arguments.attempt_id, label="attempt ID")
    expected_path = (
        run_root / "job_records" / phase / job_key / attempt_id / f"{kind}.json"
    )
    receipt_path = _lexical_absolute(arguments.path)
    if receipt_path != expected_path:
        raise LifecycleError(
            f"receipt path must be exactly {expected_path}, got {receipt_path}"
        )
    expected_result = _lexical_absolute(arguments.expected_result)
    result_relative = _relative_repo_path(
        expected_result, run_root=run_root, label="expected result"
    )
    required_result_prefix = f"results/{phase}/"
    if not result_relative.startswith(required_result_prefix):
        raise LifecycleError(
            f"expected result must stay below RUN_ROOT/{required_result_prefix}"
        )

    old: Mapping[str, Any] | None = None
    if receipt_path.exists() or receipt_path.is_symlink():
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise LifecycleError(f"existing receipt is not a regular file: {receipt_path}")
        loaded = _load_json(receipt_path, label="existing lifecycle receipt")
        if not isinstance(loaded, dict):
            raise LifecycleError("existing lifecycle receipt must be a JSON object")
        old = loaded
    previous_status = old.get("status") if old is not None else None
    allowed = ALLOWED_TRANSITIONS[kind].get(previous_status, frozenset())
    if status not in allowed:
        raise LifecycleError(
            f"invalid {kind} lifecycle transition {previous_status!r} -> {status!r}"
        )

    fixed_identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"hayatequant_paper_{kind}_receipt",
        "phase": phase,
        "job_key": job_key,
        "attempt_id": attempt_id,
        "source_commit": identity["source_commit"],
        "run_id": identity["run_id"],
        "expected_result": result_relative,
        "results_worktree": identity["results_worktree"],
    }
    if old is not None:
        for key, value in fixed_identity.items():
            if old.get(key) != value:
                raise LifecycleError(f"existing receipt identity changed at field {key!r}")

    now = _utc_now()
    document: dict[str, Any] = dict(fixed_identity)
    document["created_at_utc"] = old.get("created_at_utc", now) if old else now
    document["updated_at_utc"] = now
    document["status"] = status
    document["terminal"] = status in TERMINAL_STATUSES
    if arguments.stage is not None:
        document["stage"] = _validate_identifier(arguments.stage, label="stage")
    if arguments.reason_code is not None:
        document["reason_code"] = _validate_identifier(
            arguments.reason_code, label="reason code"
        )
    if arguments.exit_code is not None:
        if not 0 <= arguments.exit_code <= 255:
            raise LifecycleError("exit code must be in [0, 255]")
        document["exit_code"] = arguments.exit_code
    if arguments.slurm_job_id is not None:
        if SLURM_ID_RE.fullmatch(arguments.slurm_job_id) is None:
            raise LifecycleError(f"unsafe Slurm job ID: {arguments.slurm_job_id!r}")
        document["slurm_job_id"] = arguments.slurm_job_id

    for field, value in (
        ("provenance", arguments.provenance),
        ("stdout_log", arguments.stdout_log),
        ("stderr_log", arguments.stderr_log),
        ("scheduler_stdout", arguments.scheduler_stdout),
        ("scheduler_stderr", arguments.scheduler_stderr),
    ):
        if value is not None:
            document[field] = _relative_repo_path(
                value, run_root=run_root, label=field.replace("_", " ")
            )

    if kind == "worker" and status == "succeeded":
        try:
            metadata = expected_result.lstat()
        except OSError as exc:
            raise LifecycleError(
                f"worker cannot succeed without its expected result: {expected_result}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise LifecycleError(
                f"successful expected result must be a regular non-symlink: {expected_result}"
            )
        if metadata.st_size <= 0:
            raise LifecycleError(f"successful expected result is empty: {expected_result}")
        parsed = _load_json(expected_result, label="expected benchmark result")
        if not isinstance(parsed, dict):
            raise LifecycleError("successful benchmark result must be a JSON object")
        document["result"] = {
            "path": result_relative,
            "bytes": metadata.st_size,
            "sha256": _sha256(expected_result),
            "json_object_validated": True,
            "duplicate_keys_rejected": True,
            "nonfinite_numbers_rejected": True,
        }
        if arguments.exit_code not in (None, 0):
            raise LifecycleError("a successful worker receipt cannot have a nonzero exit code")
    elif kind == "worker" and status == "failed" and arguments.exit_code == 0:
        raise LifecycleError("a failed worker receipt cannot have exit code zero")

    _atomic_json(receipt_path, document, run_root=run_root)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-root")
    validate.add_argument("--source-repo", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--expected-commit", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--source-repo", type=Path, required=True)
    record.add_argument("--run-root", type=Path, required=True)
    record.add_argument("--source-commit", required=True)
    record.add_argument("--path", type=Path, required=True)
    record.add_argument("--kind", choices=sorted(ALLOWED_KINDS), required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--phase", required=True)
    record.add_argument("--job-key", required=True)
    record.add_argument("--attempt-id", required=True)
    record.add_argument("--expected-result", type=Path, required=True)
    record.add_argument("--stage")
    record.add_argument("--reason-code")
    record.add_argument("--exit-code", type=int)
    record.add_argument("--slurm-job-id")
    record.add_argument("--provenance", type=Path)
    record.add_argument("--stdout-log", type=Path)
    record.add_argument("--stderr-log", type=Path)
    record.add_argument("--scheduler-stdout", type=Path)
    record.add_argument("--scheduler-stderr", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate-root":
            document: Mapping[str, Any] = validate_results_root(
                source_repo=arguments.source_repo,
                run_root=arguments.run_root,
                expected_commit=arguments.expected_commit,
            )
        else:
            document = write_receipt(arguments)
    except LifecycleError as exc:
        print(f"paper lifecycle error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
