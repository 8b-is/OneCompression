"""Strict repo-local lifecycle receipt validation for paper result files.

This module is deliberately independent of model/evaluator code so both the
eight-job smoke approval gate and the formal aggregate can prove that every
admitted result was produced by exactly one accepted, successful wrapper
attempt.  Scheduler-terminal receipts retain failed attempts but can never
attest a result or satisfy success.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
TERMINAL_FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)


class ReceiptValidationError(RuntimeError):
    """Raised when lifecycle evidence does not prove a paper result."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReceiptValidationError(f"invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptValidationError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retained_file(root: Path, value: Any, *, label: str, receipt: Path) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ReceiptValidationError(f"{label} is not repo-relative in {receipt}")
    lexical = root / value
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise ReceiptValidationError(f"{label} is missing in {receipt}: {lexical}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReceiptValidationError(f"{label} is not a regular non-symlink: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReceiptValidationError(f"{label} escapes RUN_ROOT in {receipt}") from exc
    return resolved


def _evidence_file(
    root: Path, evidence: Any, *, expected_path: Any, label: str, receipt: Path
) -> Path:
    if not isinstance(evidence, dict) or evidence.get("path") != expected_path:
        raise ReceiptValidationError(f"{label} evidence identity is malformed: {receipt}")
    path = _retained_file(root, expected_path, label=label, receipt=receipt)
    if evidence.get("bytes") != path.stat().st_size or evidence.get("sha256") != _sha256(path):
        raise ReceiptValidationError(f"{label} changed after receipt: {receipt}")
    return path


def _job_name(*, phase: str, job_key: str) -> str:
    prefix = f"{phase}-"
    if not job_key.startswith(prefix) or job_key == prefix:
        raise ReceiptValidationError(
            f"job key {job_key!r} is not namespaced by phase {phase!r}"
        )
    return f"hq-{phase[0]}-{job_key[len(prefix):]}"


def _submission_tag(row: Mapping[str, Any]) -> str:
    fields = (
        "hayatequant-paper-submission-v1",
        row["source_commit"],
        row["run_id"],
        row["results_worktree"],
        row["phase"],
        row["job_key"],
        row["attempt_id"],
        row["expected_result"],
    )
    digest = hashlib.sha256(
        b"".join(str(value).encode("utf-8") + b"\0" for value in fields)
    ).hexdigest()
    return f"hayatequant-{digest}"


def validate_successful_result_receipts(
    run_root: Path,
    *,
    phase: str,
    source_paths: Sequence[Path],
    expected_commit: str,
) -> dict[str, Any]:
    """Validate every accepted attempt and exactly one worker success per source."""

    if phase not in {"smoke", "formal"}:
        raise ReceiptValidationError(f"unsupported receipt phase: {phase!r}")
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ReceiptValidationError("expected commit is not lowercase 40-hex")
    root = run_root.expanduser().resolve(strict=True)
    records_root = root / f"job_records/{phase}"
    if not records_root.is_dir() or records_root.is_symlink():
        raise ReceiptValidationError(f"repo-local {phase} receipt tree is missing: {records_root}")
    expected_results_worktree = str(root.parents[4].resolve(strict=True))
    submission_paths = sorted(records_root.glob("*/*/submission.json"))
    worker_paths = sorted(records_root.glob("*/*/worker.json"))
    scheduler_paths = sorted(records_root.glob("*/*/scheduler.json"))
    if not submission_paths:
        raise ReceiptValidationError(f"{phase} receipt tree has no submission records")

    identity_fields = (
        "schema_version",
        "phase",
        "job_key",
        "attempt_id",
        "source_commit",
        "run_id",
        "expected_result",
        "results_worktree",
        "scheduler_user",
        "scheduler_job_name",
        "submission_tag",
    )
    submissions: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    workers: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    schedulers: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    successful_workers: list[tuple[Path, dict[str, Any]]] = []
    status_counts: dict[str, dict[str, int]] = {
        "submission": {},
        "worker": {},
        "scheduler": {},
    }

    def validate_identity(
        row: Mapping[str, Any], path: Path, *, filename: str
    ) -> tuple[str, str]:
        relative = path.relative_to(records_root)
        if len(relative.parts) != 3 or relative.name != filename:
            raise ReceiptValidationError(f"malformed lifecycle receipt location: {path}")
        job_key, attempt_id = relative.parts[:2]
        result = row.get("expected_result")
        result_path = Path(result) if isinstance(result, str) else Path(".")
        if (
            not isinstance(result, str)
            or result_path.is_absolute()
            or ".." in result_path.parts
            or result_path.as_posix() != result
            or not result.startswith(f"results/{phase}/")
        ):
            raise ReceiptValidationError(f"receipt result escapes results/{phase}: {path}")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("phase") != phase
            or row.get("job_key") != job_key
            or row.get("attempt_id") != attempt_id
            or row.get("source_commit") != expected_commit
            or row.get("run_id") != root.name
            or row.get("results_worktree") != expected_results_worktree
            or row.get("scheduler_job_name") != _job_name(phase=phase, job_key=job_key)
            or not isinstance(row.get("scheduler_user"), str)
            or not row["scheduler_user"]
            or row.get("submission_tag") != _submission_tag(row)
        ):
            raise ReceiptValidationError(f"receipt immutable identity mismatch: {path}")
        return job_key, attempt_id

    for path in submission_paths:
        if path.is_symlink():
            raise ReceiptValidationError(f"symlinked submission receipt: {path}")
        row = _load_object(path, label="submission receipt")
        key = validate_identity(row, path, filename="submission.json")
        if (
            row.get("kind") != "hayatequant_paper_submission_receipt"
            or row.get("status") not in {"submitted", "submission_failed", "skipped_existing"}
            or row.get("terminal") is not True
        ):
            raise ReceiptValidationError(f"nonterminal or malformed submission receipt: {path}")
        if key in submissions:
            raise ReceiptValidationError(f"duplicate submission attempt: {key}")
        submissions[key] = (path, row)
        status_counts["submission"][row["status"]] = (
            status_counts["submission"].get(row["status"], 0) + 1
        )
        evidence = row.get("scheduler_submission_evidence")
        if not isinstance(evidence, dict):
            raise ReceiptValidationError(f"submission lacks scheduler evidence: {path}")
        sbatch_stdout = _evidence_file(
            root,
            evidence.get("stdout"),
            expected_path=row.get("scheduler_stdout"),
            label="sbatch stdout",
            receipt=path,
        )
        _evidence_file(
            root,
            evidence.get("stderr"),
            expected_path=row.get("scheduler_stderr"),
            label="sbatch stderr",
            receipt=path,
        )
        if row["status"] == "submitted":
            job_id = row.get("slurm_job_id")
            match = re.fullmatch(
                r"([0-9]+)(?:;[A-Za-z0-9._-]+)?\r?\n?",
                sbatch_stdout.read_text(encoding="utf-8"),
            )
            if not isinstance(job_id, str) or match is None or match.group(1) != job_id:
                raise ReceiptValidationError(f"submitted Slurm identity is malformed: {path}")

    for path in worker_paths:
        if path.is_symlink():
            raise ReceiptValidationError(f"symlinked worker receipt: {path}")
        row = _load_object(path, label="worker receipt")
        key = validate_identity(row, path, filename="worker.json")
        if (
            row.get("kind") != "hayatequant_paper_worker_receipt"
            or row.get("status") not in {"succeeded", "failed"}
            or row.get("terminal") is not True
        ):
            raise ReceiptValidationError(f"nonterminal or malformed worker receipt: {path}")
        if key in workers:
            raise ReceiptValidationError(f"duplicate worker attempt: {key}")
        workers[key] = (path, row)
        status_counts["worker"][row["status"]] = status_counts["worker"].get(row["status"], 0) + 1
        submission_entry = submissions.get(key)
        if submission_entry is None:
            raise ReceiptValidationError(f"worker lacks matching submission: {path}")
        submission_path, submission = submission_entry
        if submission.get("status") != "submitted" or any(
            submission.get(field) != row.get(field) for field in identity_fields
        ):
            raise ReceiptValidationError(
                f"worker/submission immutable identities disagree: {path} / {submission_path}"
            )
        if row.get("slurm_job_id") != submission.get("slurm_job_id"):
            raise ReceiptValidationError(f"worker/submission Slurm IDs disagree: {path}")
        _retained_file(root, row.get("stdout_log"), label="worker stdout", receipt=path)
        _retained_file(root, row.get("stderr_log"), label="worker stderr", receipt=path)
        if row["status"] == "failed":
            if not isinstance(row.get("exit_code"), int) or row["exit_code"] == 0:
                raise ReceiptValidationError(f"failed worker has no nonzero exit code: {path}")
            continue
        if row.get("exit_code") != 0:
            raise ReceiptValidationError(f"successful worker has nonzero exit code: {path}")
        _retained_file(root, row.get("provenance"), label="worker provenance", receipt=path)
        result = row.get("result")
        if not isinstance(result, dict) or result.get("path") != row.get("expected_result"):
            raise ReceiptValidationError(f"successful worker result identity is malformed: {path}")
        result_path = _evidence_file(
            root,
            result,
            expected_path=row["expected_result"],
            label="successful result",
            receipt=path,
        )
        if (
            result.get("json_object_validated") is not True
            or result.get("duplicate_keys_rejected") is not True
            or result.get("nonfinite_numbers_rejected") is not True
        ):
            raise ReceiptValidationError(f"successful result validation flags are missing: {path}")
        _load_object(result_path, label="successful benchmark result")
        successful_workers.append((path, row))

    for path in scheduler_paths:
        if path.is_symlink():
            raise ReceiptValidationError(f"symlinked scheduler receipt: {path}")
        row = _load_object(path, label="scheduler receipt")
        key = validate_identity(row, path, filename="scheduler.json")
        if (
            row.get("kind") != "hayatequant_paper_scheduler_receipt"
            or row.get("status") != "scheduler_terminal_failure"
            or row.get("terminal") is not True
        ):
            raise ReceiptValidationError(f"scheduler receipt is not failure-only: {path}")
        if key in schedulers or key in workers:
            raise ReceiptValidationError(f"attempt does not have exactly one execution receipt: {key}")
        schedulers[key] = (path, row)
        status_counts["scheduler"][row["status"]] = (
            status_counts["scheduler"].get(row["status"], 0) + 1
        )
        submission_entry = submissions.get(key)
        if submission_entry is None:
            raise ReceiptValidationError(f"scheduler lacks matching submission: {path}")
        submission_path, submission = submission_entry
        if submission.get("status") != "submitted" or any(
            submission.get(field) != row.get(field) for field in identity_fields
        ):
            raise ReceiptValidationError(
                f"scheduler/submission immutable identities disagree: {path} / {submission_path}"
            )
        slurm = row.get("slurm")
        if (
            not isinstance(slurm, dict)
            or slurm.get("job_id_raw") != submission.get("slurm_job_id")
            or slurm.get("user") != row.get("scheduler_user")
            or slurm.get("job_name") != row.get("scheduler_job_name")
            or slurm.get("submission_tag") != row.get("submission_tag")
            or slurm.get("state") not in TERMINAL_FAILURE_STATES
            or not (
                slurm.get("state_raw") == slurm.get("state")
                or (
                    slurm.get("state") == "CANCELLED"
                    and isinstance(slurm.get("state_raw"), str)
                    and re.fullmatch(
                        r"CANCELLED(?: by [0-9]+)?", slurm["state_raw"]
                    )
                    is not None
                )
            )
            or not isinstance(slurm.get("exit_code"), str)
            or re.fullmatch(r"[0-9]+:[0-9]+", slurm["exit_code"]) is None
        ):
            raise ReceiptValidationError(f"scheduler receipt is not an exact Slurm failure: {path}")
        evidence = row.get("scheduler_query_evidence")
        if not isinstance(evidence, dict):
            raise ReceiptValidationError(f"scheduler lacks raw sacct evidence: {path}")
        sacct_stdout = _evidence_file(
            root,
            evidence.get("stdout"),
            expected_path=(evidence.get("stdout") or {}).get("path"),
            label="raw sacct stdout",
            receipt=path,
        )
        _evidence_file(
            root,
            evidence.get("stderr"),
            expected_path=(evidence.get("stderr") or {}).get("path"),
            label="raw sacct stderr",
            receipt=path,
        )
        lines = [line for line in sacct_stdout.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != 1:
            raise ReceiptValidationError(f"raw sacct evidence is ambiguous: {path}")
        fields = lines[0].split("|")
        if fields and fields[-1] == "":
            fields.pop()
        expected_fields = [
            slurm.get("job_id_raw"),
            slurm.get("user"),
            slurm.get("job_name"),
            slurm.get("state_raw"),
            slurm.get("exit_code"),
            slurm.get("submit_time"),
            slurm.get("start_time"),
            slurm.get("end_time"),
            slurm.get("submission_tag"),
        ]
        if fields != expected_fields:
            raise ReceiptValidationError(f"scheduler receipt disagrees with raw sacct: {path}")

    for key, (path, submission) in submissions.items():
        execution_count = int(key in workers) + int(key in schedulers)
        if submission["status"] == "submitted" and execution_count != 1:
            raise ReceiptValidationError(
                f"accepted attempt requires exactly one terminal execution receipt: {path}"
            )
        if submission["status"] != "submitted" and execution_count:
            raise ReceiptValidationError(f"non-accepted attempt has execution receipt: {path}")

    source_relatives: list[str] = []
    for source_path in source_paths:
        source = source_path.expanduser().resolve(strict=True)
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ReceiptValidationError(f"source escapes RUN_ROOT: {source}") from exc
        if not relative.startswith(f"results/{phase}/"):
            raise ReceiptValidationError(f"source is outside results/{phase}: {source}")
        matches = [
            (path, row)
            for path, row in successful_workers
            if row.get("expected_result") == relative
        ]
        if len(matches) != 1:
            raise ReceiptValidationError(
                f"source requires exactly one successful worker receipt, got {len(matches)}: {source}"
            )
        source_relatives.append(relative)

    successful_relatives = [row["expected_result"] for _path, row in successful_workers]
    if sorted(successful_relatives) != sorted(source_relatives):
        raise ReceiptValidationError(
            "successful worker result set is not exactly the requested result set"
        )

    return {
        "schema_version": 1,
        "phase": phase,
        "source_results_validated": len(source_relatives),
        "submission_receipts": len(submission_paths),
        "worker_receipts": len(worker_paths),
        "scheduler_receipts": len(scheduler_paths),
        "status_counts": status_counts,
        "scheduler_receipts_are_success": False,
    }
