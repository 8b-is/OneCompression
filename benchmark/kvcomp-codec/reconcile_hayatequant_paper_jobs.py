#!/usr/bin/env python3
"""Reconcile accepted HayateQuant paper jobs that never started their wrapper.

The worker receipt remains the only path to scientific success.  This command
queries Slurm accounting for an accepted attempt without a worker receipt and
writes ``scheduler.json`` only when one exact, attempt-tagged allocation row
proves a terminal non-success state.  PENDING/RUNNING rows are left untouched;
COMPLETED is deliberately rejected because scheduler success cannot replace a
validated benchmark result and worker receipt.

Raw ``sacct`` stdout/stderr and a reconciliation report are written atomically
below the attempt directory.  The terminal scheduler receipt hashes both raw
streams.  An interrupted ``submission_started`` receipt is recoverable as
``submitted`` only when retained sbatch stdout is one exact numeric job ID and
the authoritative accounting row matches every immutable identity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import paper_job_lifecycle as lifecycle


SACCT_FIELDS = (
    "JobIDRaw",
    "User",
    "JobName",
    "State%64",
    "ExitCode",
    "Submit",
    "Start",
    "End",
    "Comment",
)
NONTERMINAL_STATES = frozenset(
    {
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RUNNING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
)
SBATCH_STDOUT_RE = re.compile(r"([0-9]+)(?:;[A-Za-z0-9._-]+)?\r?\n?\Z")
UTC_SACCT_TIME_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\Z")


class ReconcileError(RuntimeError):
    """Raised when scheduler evidence cannot safely identify one attempt."""


def _atomic_bytes(path: Path, payload: bytes, *, run_root: Path) -> None:
    lifecycle._assert_no_existing_symlink(run_root, path, label="scheduler evidence")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lifecycle._assert_no_existing_symlink(run_root, path, label="scheduler evidence")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ReconcileError(f"scheduler evidence destination is not a regular file: {path}")
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


def _strict_relative_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ReconcileError(f"{label} is not a nonempty repo-relative path")
    lexical = root / value
    lifecycle._assert_no_existing_symlink(root, lexical, label=label)
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReconcileError(f"{label} escapes RUN_ROOT") from exc
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise ReconcileError(f"{label} is not retained: {lexical}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReconcileError(f"{label} is not a regular non-symlink: {lexical}")
    return lexical


def _validate_file_evidence(
    root: Path, evidence: Any, *, expected_path: str, label: str
) -> None:
    if not isinstance(evidence, dict) or evidence.get("path") != expected_path:
        raise ReconcileError(f"{label} evidence identity is malformed")
    path = _strict_relative_file(root, expected_path, label=label)
    if evidence.get("bytes") != path.stat().st_size or evidence.get("sha256") != lifecycle._sha256(path):
        raise ReconcileError(f"{label} changed after its submission receipt")


def _submission_identity(
    path: Path, *, root: Path, identity: Mapping[str, str], phase: str
) -> dict[str, Any]:
    row = lifecycle._load_json(path, label="submission receipt")
    if not isinstance(row, dict):
        raise ReconcileError(f"submission receipt is not a JSON object: {path}")
    relative = path.relative_to(root / f"job_records/{phase}")
    if len(relative.parts) != 3 or relative.name != "submission.json":
        raise ReconcileError(f"malformed submission receipt location: {path}")
    job_key, attempt_id = relative.parts[:2]
    expected_result = row.get("expected_result")
    expected_results_worktree = str(root.parents[4].resolve(strict=True))
    fixed = {
        "schema_version": lifecycle.SCHEMA_VERSION,
        "kind": "hayatequant_paper_submission_receipt",
        "phase": phase,
        "job_key": job_key,
        "attempt_id": attempt_id,
        "source_commit": identity["source_commit"],
        "run_id": root.name,
        "expected_result": expected_result,
        "results_worktree": expected_results_worktree,
        "scheduler_user": lifecycle._scheduler_user(),
        "scheduler_job_name": lifecycle.expected_scheduler_job_name(
            phase=phase, job_key=job_key
        ),
    }
    if not isinstance(expected_result, str) or not expected_result.startswith(
        f"results/{phase}/"
    ):
        raise ReconcileError(f"submission expected result is outside results/{phase}: {path}")
    fixed["submission_tag"] = lifecycle.expected_submission_tag(fixed)
    if any(row.get(field) != value for field, value in fixed.items()):
        raise ReconcileError(f"submission immutable identity mismatch: {path}")
    allowed_statuses = {
        "submission_started",
        "submission_interrupted",
        "submitted",
        "submission_failed",
        "skipped_existing",
    }
    if row.get("status") not in allowed_statuses:
        raise ReconcileError(f"submission has unsupported status: {path}")
    expected_terminal = row["status"] in {
        "submitted",
        "submission_failed",
        "skipped_existing",
    }
    if row.get("terminal") is not expected_terminal:
        raise ReconcileError(f"submission terminal flag disagrees with status: {path}")
    stdout_path = _strict_relative_file(
        root, row.get("scheduler_stdout"), label="retained sbatch stdout"
    )
    stderr_path = _strict_relative_file(
        root, row.get("scheduler_stderr"), label="retained sbatch stderr"
    )
    evidence = row.get("scheduler_submission_evidence")
    if not isinstance(evidence, dict):
        raise ReconcileError(f"submission lacks hashed scheduler evidence: {path}")
    _validate_file_evidence(
        root,
        evidence.get("stdout"),
        expected_path=row["scheduler_stdout"],
        label="retained sbatch stdout",
    )
    _validate_file_evidence(
        root,
        evidence.get("stderr"),
        expected_path=row["scheduler_stderr"],
        label="retained sbatch stderr",
    )
    row["_path"] = path
    row["_stdout_path"] = stdout_path
    row["_stderr_path"] = stderr_path
    row["_expected_path"] = root / expected_result
    return row


def _sbatch_job_id(path: Path) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReconcileError(f"cannot read retained sbatch stdout: {exc}") from exc
    match = SBATCH_STDOUT_RE.fullmatch(raw)
    return match.group(1) if match is not None else None


def _normalized_state(raw: str) -> str:
    if raw in lifecycle.SLURM_TERMINAL_FAILURE_STATES or raw in NONTERMINAL_STATES or raw == "COMPLETED":
        return raw
    cancelled = re.fullmatch(r"CANCELLED(?: by [0-9]+)?", raw)
    if cancelled is not None:
        return "CANCELLED"
    raise ReconcileError(f"unknown or ambiguous Slurm state {raw!r}")


def _parse_sacct(stdout: bytes, *, expected_job_id: str) -> dict[str, str]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise ReconcileError(f"sacct stdout is not UTF-8: {exc}") from exc
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line:
            continue
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != len(SACCT_FIELDS):
            raise ReconcileError(f"sacct row has {len(fields)} fields, expected {len(SACCT_FIELDS)}")
        rows.append(fields)
    if len(rows) != 1:
        raise ReconcileError(f"sacct returned {len(rows)} allocation rows, expected exactly one")
    names = tuple(field.split("%", 1)[0] for field in SACCT_FIELDS)
    row = dict(zip(names, rows[0], strict=True))
    if row["JobIDRaw"] != expected_job_id:
        raise ReconcileError(
            f"sacct JobIDRaw {row['JobIDRaw']!r} does not match {expected_job_id!r}"
        )
    row["StateRaw"] = row["State"]
    row["State"] = _normalized_state(row["State"])
    return row


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ReconcileError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReconcileError(f"{label} is malformed: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ReconcileError(f"{label} has no timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _verify_sacct_identity(row: Mapping[str, str], submission: Mapping[str, Any]) -> None:
    for field, expected in (
        ("User", submission["scheduler_user"]),
        ("JobName", submission["scheduler_job_name"]),
        ("Comment", submission["submission_tag"]),
    ):
        if row.get(field) != expected:
            raise ReconcileError(
                f"sacct {field} {row.get(field)!r} does not match immutable value {expected!r}"
            )
    submit_raw = row.get("Submit")
    if not isinstance(submit_raw, str) or UTC_SACCT_TIME_RE.fullmatch(submit_raw) is None:
        raise ReconcileError(f"sacct Submit time is not an exact UTC timestamp: {submit_raw!r}")
    submit = datetime.strptime(submit_raw, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    created = _parse_utc(submission.get("created_at_utc"), label="receipt creation time")
    updated = _parse_utc(submission.get("updated_at_utc"), label="receipt update time")
    tolerance = timedelta(minutes=5)
    if not created - tolerance <= submit <= updated + tolerance:
        raise ReconcileError(
            "sacct Submit time is outside the immutable submission receipt window"
        )


def _record(
    *,
    source_repo: Path,
    run_root: Path,
    source_commit: str,
    path: Path,
    kind: str,
    status: str,
    submission: Mapping[str, Any],
    reason_code: str,
    exit_code: int | None = None,
    slurm_job_id: str | None = None,
    sacct: Mapping[str, str] | None = None,
    query_stdout: Path | None = None,
    query_stderr: Path | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "source_repo": source_repo,
        "run_root": run_root,
        "source_commit": source_commit,
        "path": path,
        "kind": kind,
        "status": status,
        "phase": submission["phase"],
        "job_key": submission["job_key"],
        "attempt_id": submission["attempt_id"],
        "expected_result": submission["_expected_path"],
        "stage": "scheduler_reconciliation" if kind == "scheduler" else "submitter_reconciliation",
        "reason_code": reason_code,
        "exit_code": exit_code,
        "slurm_job_id": slurm_job_id,
        "provenance": None,
        "stdout_log": None,
        "stderr_log": None,
        "scheduler_stdout": submission["_stdout_path"] if kind == "submission" else None,
        "scheduler_stderr": submission["_stderr_path"] if kind == "submission" else None,
        "scheduler_query_stdout": query_stdout,
        "scheduler_query_stderr": query_stderr,
        "slurm_state": sacct.get("State") if sacct else None,
        "slurm_state_raw": sacct.get("StateRaw") if sacct else None,
        "slurm_user": sacct.get("User") if sacct else None,
        "slurm_job_name": sacct.get("JobName") if sacct else None,
        "slurm_exit_code": sacct.get("ExitCode") if sacct else None,
        "slurm_submit_time": sacct.get("Submit") if sacct else None,
        "slurm_start_time": sacct.get("Start") if sacct else None,
        "slurm_end_time": sacct.get("End") if sacct else None,
    }
    return lifecycle.write_receipt(argparse.Namespace(**values))


def _query_sacct(
    executable: Path, *, job_id: str, stdout_path: Path, stderr_path: Path, run_root: Path
) -> tuple[int, bytes, bytes]:
    command = (
        str(executable),
        "--allocations",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format",
        ",".join(SACCT_FIELDS),
    )
    environment = dict(os.environ)
    environment.update({"TZ": "UTC", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        stdout = b""
        stderr = f"cannot execute sacct: {exc}\n".encode("utf-8", errors="replace")
        returncode = 127
    else:
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    _atomic_bytes(stdout_path, stdout, run_root=run_root)
    _atomic_bytes(stderr_path, stderr, run_root=run_root)
    return returncode, stdout, stderr


def _write_report(
    path: Path,
    *,
    run_root: Path,
    submission: Mapping[str, Any],
    outcome: str,
    detail: str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hayatequant_paper_scheduler_reconciliation",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_root.name,
        "results_worktree": str(run_root.parents[4].resolve(strict=True)),
        "phase": submission["phase"],
        "job_key": submission["job_key"],
        "attempt_id": submission["attempt_id"],
        "source_commit": submission["source_commit"],
        "expected_result": submission["expected_result"],
        "submission_tag": submission["submission_tag"],
        "outcome": outcome,
        "detail": detail,
    }
    if stdout_path is not None and stderr_path is not None:
        report["raw_sacct"] = {
            "stdout": lifecycle._regular_file_evidence(
                stdout_path, run_root=run_root, label="raw sacct stdout"
            ),
            "stderr": lifecycle._regular_file_evidence(
                stderr_path, run_root=run_root, label="raw sacct stderr"
            ),
        }
    lifecycle._atomic_json(path, report, run_root=run_root)


def reconcile(arguments: argparse.Namespace) -> dict[str, Any]:
    identity = lifecycle.validate_results_root(
        source_repo=arguments.source_repo,
        run_root=arguments.run_root,
        expected_commit=arguments.source_commit,
    )
    root = Path(identity["run_root"])
    phases = tuple(arguments.phase or sorted(lifecycle.ALLOWED_PHASES))
    counts = {
        "inspected": 0,
        "already_terminal_submission": 0,
        "worker_present": 0,
        "scheduler_present": 0,
        "submission_recovered": 0,
        "submission_failed_unproven": 0,
        "nonterminal": 0,
        "scheduler_terminal_failure": 0,
        "errors": 0,
    }
    errors: list[str] = []
    for phase in phases:
        records = root / f"job_records/{phase}"
        if not records.exists():
            continue
        for submission_path in sorted(records.glob("*/*/submission.json")):
            counts["inspected"] += 1
            submission: dict[str, Any] | None = None
            try:
                submission = _submission_identity(
                    submission_path, root=root, identity=identity, phase=phase
                )
                attempt_root = submission_path.parent
                worker_path = attempt_root / "worker.json"
                scheduler_path = attempt_root / "scheduler.json"
                if (worker_path.exists() or worker_path.is_symlink()) and (
                    scheduler_path.exists() or scheduler_path.is_symlink()
                ):
                    raise ReconcileError(
                        "worker.json and scheduler.json both exist for one attempt"
                    )
                if submission["status"] in {"submission_failed", "skipped_existing"}:
                    if worker_path.exists() or scheduler_path.exists():
                        raise ReconcileError(
                            "non-accepted submission has an execution receipt"
                        )
                    counts["already_terminal_submission"] += 1
                    continue
                if scheduler_path.exists() or scheduler_path.is_symlink():
                    if scheduler_path.is_symlink() or not scheduler_path.is_file():
                        raise ReconcileError(f"invalid scheduler receipt: {scheduler_path}")
                    counts["scheduler_present"] += 1
                    continue

                result_path = submission["_expected_path"]
                if result_path.exists() or result_path.is_symlink():
                    if not worker_path.exists():
                        raise ReconcileError(
                            "expected result exists without a worker receipt; refusing scheduler attestation"
                        )
                if worker_path.exists() and submission["status"] == "submitted":
                    if worker_path.is_symlink() or not worker_path.is_file():
                        raise ReconcileError(f"invalid worker receipt: {worker_path}")
                    counts["worker_present"] += 1
                    continue

                sbatch_job_id = _sbatch_job_id(submission["_stdout_path"])
                receipt_job_id = submission.get("slurm_job_id")
                if receipt_job_id is not None and receipt_job_id != sbatch_job_id:
                    raise ReconcileError(
                        "submission Slurm ID does not match retained sbatch stdout"
                    )
                job_id = receipt_job_id or sbatch_job_id
                evidence_root = attempt_root / "scheduler_evidence"
                report_path = evidence_root / "reconciliation.json"
                if job_id is None:
                    if submission["status"] not in {
                        "submission_started",
                        "submission_interrupted",
                    }:
                        raise ReconcileError("accepted submission has no numeric Slurm job ID")
                    _record(
                        source_repo=arguments.source_repo,
                        run_root=root,
                        source_commit=arguments.source_commit,
                        path=submission_path,
                        kind="submission",
                        status="submission_failed",
                        submission=submission,
                        reason_code="scheduler_acceptance_unproven",
                        exit_code=125,
                    )
                    _write_report(
                        report_path,
                        run_root=root,
                        submission=submission,
                        outcome="submission_failed_unproven",
                        detail="retained sbatch stdout contains no exact numeric job ID",
                    )
                    counts["submission_failed_unproven"] += 1
                    continue

                query_stdout = evidence_root / "sacct.stdout"
                query_stderr = evidence_root / "sacct.stderr"
                returncode, stdout, _stderr = _query_sacct(
                    arguments.sacct,
                    job_id=job_id,
                    stdout_path=query_stdout,
                    stderr_path=query_stderr,
                    run_root=root,
                )
                if returncode != 0:
                    raise ReconcileError(f"sacct exited with status {returncode}")
                sacct = _parse_sacct(stdout, expected_job_id=job_id)
                _verify_sacct_identity(sacct, submission)

                if submission["status"] in {
                    "submission_started",
                    "submission_interrupted",
                }:
                    recovered = _record(
                        source_repo=arguments.source_repo,
                        run_root=root,
                        source_commit=arguments.source_commit,
                        path=submission_path,
                        kind="submission",
                        status="submitted",
                        submission=submission,
                        reason_code="sacct_recovered_acceptance",
                        exit_code=0,
                        slurm_job_id=job_id,
                    )
                    submission.update(recovered)
                    counts["submission_recovered"] += 1

                if worker_path.exists() or worker_path.is_symlink():
                    if worker_path.is_symlink() or not worker_path.is_file():
                        raise ReconcileError(f"invalid worker receipt: {worker_path}")
                    _write_report(
                        report_path,
                        run_root=root,
                        submission=submission,
                        outcome="worker_present",
                        detail="worker receipt exists; scheduler receipt is forbidden",
                        stdout_path=query_stdout,
                        stderr_path=query_stderr,
                    )
                    counts["worker_present"] += 1
                    continue

                state = sacct["State"]
                if state in NONTERMINAL_STATES:
                    _write_report(
                        report_path,
                        run_root=root,
                        submission=submission,
                        outcome="nonterminal",
                        detail=f"Slurm state {state}; no terminal receipt written",
                        stdout_path=query_stdout,
                        stderr_path=query_stderr,
                    )
                    counts["nonterminal"] += 1
                    continue
                if state == "COMPLETED":
                    raise ReconcileError(
                        "Slurm COMPLETED cannot replace a successful worker/result receipt"
                    )
                if state not in lifecycle.SLURM_TERMINAL_FAILURE_STATES:
                    raise ReconcileError(f"unsupported terminal Slurm state: {state}")
                _record(
                    source_repo=arguments.source_repo,
                    run_root=root,
                    source_commit=arguments.source_commit,
                    path=scheduler_path,
                    kind="scheduler",
                    status="scheduler_terminal_failure",
                    submission=submission,
                    reason_code="slurm_terminal_before_worker",
                    slurm_job_id=job_id,
                    sacct=sacct,
                    query_stdout=query_stdout,
                    query_stderr=query_stderr,
                )
                _write_report(
                    report_path,
                    run_root=root,
                    submission=submission,
                    outcome="scheduler_terminal_failure",
                    detail=f"Slurm proved terminal non-success state {state}",
                    stdout_path=query_stdout,
                    stderr_path=query_stderr,
                )
                counts["scheduler_terminal_failure"] += 1
            except (lifecycle.LifecycleError, ReconcileError) as exc:
                if submission is not None and submission.get("status") in {
                    "submission_started",
                    "submission_interrupted",
                }:
                    try:
                        _record(
                            source_repo=arguments.source_repo,
                            run_root=root,
                            source_commit=arguments.source_commit,
                            path=submission_path,
                            kind="submission",
                            status="submission_failed",
                            submission=submission,
                            reason_code="scheduler_identity_unproven",
                            exit_code=125,
                        )
                        counts["submission_failed_unproven"] += 1
                    except lifecycle.LifecycleError:
                        pass
                counts["errors"] += 1
                message = f"{submission_path}: {exc}"
                errors.append(message)
                if submission is not None:
                    try:
                        evidence_root = submission_path.parent / "scheduler_evidence"
                        query_stdout = evidence_root / "sacct.stdout"
                        query_stderr = evidence_root / "sacct.stderr"
                        _write_report(
                            evidence_root / "reconciliation.json",
                            run_root=root,
                            submission=submission,
                            outcome="rejected",
                            detail=str(exc),
                            stdout_path=query_stdout if query_stdout.is_file() else None,
                            stderr_path=query_stderr if query_stderr.is_file() else None,
                        )
                    except (lifecycle.LifecycleError, ReconcileError, OSError):
                        pass
    return {
        "schema_version": 1,
        "kind": "hayatequant_paper_scheduler_reconciliation_summary",
        "run_id": root.name,
        "results_worktree": str(root.parents[4].resolve(strict=True)),
        "source_commit": identity["source_commit"],
        "counts": counts,
        "errors": errors,
        "complete": counts["errors"] == 0 and counts["nonterminal"] == 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--sacct", type=Path, default=Path("sacct"))
    parser.add_argument("--phase", action="append", choices=sorted(lifecycle.ALLOWED_PHASES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = reconcile(arguments)
    except (lifecycle.LifecycleError, ReconcileError) as exc:
        print(f"paper scheduler reconciliation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["counts"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
