#!/usr/bin/env python3
"""Fail-fast provenance check for HayateQuant paper jobs.

The cluster has several editable environments that point at different onecomp
checkouts.  Every paper job calls this helper before loading a model so a run
cannot silently import the original dirty checkout while claiming the revision
from the reproduction worktree.  The emitted JSON intentionally contains only
an allow-list of environment metadata; credentials are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_weight_manifest import (
    ModelWeightError,
    load_input_manifest_model,
    verify_model_weights,
)

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "datasets",
    "huggingface-hub",
    "pyarrow",
    "hydra-core",
    "omegaconf",
    "filelock",
    "lm-eval",
    "triton",
    "vllm",
    "math-verify",
    "safetensors",
    "rouge",
    "fuzzywuzzy",
)
ENV_ALLOWLIST = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_MEM_PER_NODE",
    "CUDA_VISIBLE_DEVICES",
    "PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES",
    "PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE",
    "ONECOMP_TRITON_CODEC",
    "ONECOMP_TRITON_PREFILL",
    "ONECOMP_COMPILE_CODEC",
    "ONECOMP_COMPILE_DECODE_ONLY",
    "KVCOMP_CODEBOOK_DIR",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "WANDB_MODE",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HOME",
    "TOKENIZERS_PARALLELISM",
    "PYTHONHASHSEED",
    "PAPER_SMOKE_APPROVAL_MARKER",
    "PAPER_SMOKE_APPROVAL_MARKER_SHA256",
    "PAPER_SMOKE_VALIDATED_PLAN_SHA256",
    "PAPER_SMOKE_VALIDATOR_CHECK_COMPLETED",
)
MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "preprocessor_config.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
)
SMOKE_APPROVAL_KIND = "hayatequant_formal_smoke_approval"
SMOKE_APPROVAL_STATEMENT = "I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN"
SMOKE_APPROVAL_SCHEMA_VERSION = 1
SMOKE_PLAN_VERSION = 3
SMOKE_JOB_KEYS = (
    "ppl",
    "zeroshot",
    "ruler",
    "longbench",
    "math500",
    "longcode",
    "speed",
    "native-hybrid",
)


def _run(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        if check:
            raise RuntimeError(f"Required command not found: {args[0]}") from exc
        return ""
    if check and proc.returncode:
        raise RuntimeError(
            f"Command {args[0]!r} failed with exit {proc.returncode}: " f"{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise RuntimeError(f"non-finite JSON number {value!r}")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except RuntimeError as exc:
        raise RuntimeError(f"{label} contains a {exc}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def _state_tuple(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_file_record(path: Path, *, label: str) -> dict:
    """Full-hash one regular file and reject changes during the read."""
    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{label} is missing or cannot be resolved: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    digest = _sha256(resolved)
    try:
        after = resolved.stat()
    except OSError as exc:
        raise RuntimeError(f"{label} disappeared while hashing: {resolved}") from exc
    if _state_tuple(before) != _state_tuple(after):
        raise RuntimeError(f"{label} changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("value cannot be canonically serialized as finite JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _lower_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _record_below(
    declared: Any,
    *,
    root: Path,
    label: str,
) -> dict:
    if not isinstance(declared, dict):
        raise RuntimeError(f"{label} is not a file record")
    raw_path = declared.get("path")
    byte_count = declared.get("bytes")
    digest = declared.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise RuntimeError(f"{label} has a malformed path/byte count")
    _lower_sha256(digest, label=f"{label} sha256")
    actual = _stable_file_record(Path(raw_path), label=label)
    try:
        Path(actual["path"]).relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} escapes its approved run-root scope") from exc
    expected = {"path": raw_path, "bytes": byte_count, "sha256": digest}
    if actual != expected:
        raise RuntimeError(f"{label} changed after formal smoke approval")
    return actual


def _verify_smoke_approval_binding(
    *,
    marker: Path,
    run_root: Path,
    expected_commit: str,
    input_manifest: Path,
    expected_input_manifest_sha256: str,
    expected_marker_sha256: str | None = None,
    expected_validated_plan_sha256: str | None = None,
) -> dict:
    """Re-hash the submitter-bound smoke marker, results, and worker preflights.

    The semantic smoke validator runs both before submission and in every
    formal worker.  This independent integrity check binds the exact marker
    bytes and canonical validated-plan identity into the formal preflight, so
    a pending job cannot accept a replacement marker or changed smoke file.
    """

    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise RuntimeError("Expected smoke code commit must be lowercase 40-hex")
    _lower_sha256(
        expected_input_manifest_sha256,
        label="expected input-manifest sha256",
    )
    if expected_marker_sha256 is not None:
        _lower_sha256(expected_marker_sha256, label="expected smoke marker sha256")
    if expected_validated_plan_sha256 is not None:
        _lower_sha256(
            expected_validated_plan_sha256,
            label="expected validated-plan sha256",
        )

    run_root = run_root.expanduser().resolve(strict=True)
    marker_argument = marker.expanduser()
    if marker_argument.is_symlink():
        raise RuntimeError("Formal smoke approval marker may not be a symlink")
    marker_record_before = _stable_file_record(
        marker_argument,
        label="formal smoke approval marker",
    )
    marker_path = Path(marker_record_before["path"])
    try:
        marker_path.relative_to((run_root / "manifests").resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Formal smoke approval marker escapes RUN_ROOT/manifests") from exc
    if (
        expected_marker_sha256 is not None
        and marker_record_before["sha256"] != expected_marker_sha256
    ):
        raise RuntimeError("Formal smoke approval marker differs from its submit-time SHA-256")

    document = _load_json(marker_path, label="formal smoke approval marker")
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SMOKE_APPROVAL_SCHEMA_VERSION
        or document.get("kind") != SMOKE_APPROVAL_KIND
        or document.get("approval_statement") != SMOKE_APPROVAL_STATEMENT
        or not isinstance(document.get("approved_at_utc"), str)
        or not isinstance(document.get("approved_by"), dict)
    ):
        raise RuntimeError("Formal smoke approval marker has an invalid identity")
    plan = document.get("validated_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Formal smoke approval marker has no validated plan")
    validated_plan_sha256 = _canonical_sha256(plan)
    if (
        expected_validated_plan_sha256 is not None
        and validated_plan_sha256 != expected_validated_plan_sha256
    ):
        raise RuntimeError("Formal smoke validated plan differs from its submit-time SHA-256")

    input_record = _stable_file_record(input_manifest, label="paper input manifest")
    if input_record["sha256"] != expected_input_manifest_sha256:
        raise RuntimeError("Paper input manifest differs from its submission SHA-256")
    completed = plan.get("completed_job_keys")
    results = plan.get("results")
    smoke_plan_sha256 = _lower_sha256(plan.get("smoke_plan_sha256"), label="smoke plan sha256")
    if (
        plan.get("smoke_plan_version") != SMOKE_PLAN_VERSION
        or plan.get("code_commit") != expected_commit
        or plan.get("input_manifest") != input_record
        or completed != list(SMOKE_JOB_KEYS)
        or not isinstance(results, list)
        or len(results) != len(SMOKE_JOB_KEYS)
    ):
        raise RuntimeError("Formal smoke validated-plan identity changed")

    result_records = []
    result_documents: dict[str, dict] = {}
    observed_keys = []
    for index, declared in enumerate(results):
        if not isinstance(declared, dict):
            raise RuntimeError("Formal smoke validated plan has a malformed result record")
        key = declared.get("key")
        observed_keys.append(key)
        actual = _record_below(
            {name: declared.get(name) for name in ("path", "bytes", "sha256")},
            root=run_root / "results/smoke",
            label=f"approved smoke result {key!r}",
        )
        expected_key = SMOKE_JOB_KEYS[index]
        if key != expected_key:
            raise RuntimeError("Formal smoke result order/key identity changed")
        result_records.append({"key": key, **actual})
        loaded = _load_json(Path(actual["path"]), label=f"approved smoke result {key!r}")
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Approved smoke result {key!r} is not a JSON object")
        result_documents[str(key)] = loaded
    if observed_keys != list(SMOKE_JOB_KEYS) or len(set(observed_keys)) != len(observed_keys):
        raise RuntimeError("Formal smoke result set is not the exact eight-job plan")

    provenance_root = run_root / "provenance"
    preflight_records = []
    preflight_paths: set[str] = set()
    for key in SMOKE_JOB_KEYS:
        if key == "native-hybrid":
            continue
        envelopes = result_documents[key].get("run_provenance")
        if not isinstance(envelopes, list) or len(envelopes) != 1:
            raise RuntimeError(f"Approved smoke result {key!r} lost its worker preflight")
        envelope = envelopes[0]
        if not isinstance(envelope, dict) or not isinstance(envelope.get("record"), dict):
            raise RuntimeError(f"Approved smoke result {key!r} has malformed provenance")
        raw_path = envelope.get("path")
        digest = envelope.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise RuntimeError(f"Approved smoke result {key!r} has no preflight path")
        _lower_sha256(digest, label=f"approved smoke result {key!r} preflight sha256")
        actual = _stable_file_record(
            Path(raw_path),
            label=f"approved smoke result {key!r} worker preflight",
        )
        try:
            Path(actual["path"]).relative_to(provenance_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Approved smoke preflight for {key!r} escapes run root") from exc
        if actual["sha256"] != digest:
            raise RuntimeError(f"Approved smoke preflight for {key!r} changed")
        on_disk = _load_json(
            Path(actual["path"]),
            label=f"approved smoke result {key!r} worker preflight",
        )
        if on_disk != envelope["record"]:
            raise RuntimeError(f"Approved smoke preflight for {key!r} differs from its envelope")
        if actual["path"] in preflight_paths:
            raise RuntimeError("Approved smoke results reuse one worker preflight")
        preflight_paths.add(actual["path"])
        preflight_records.append({"key": key, **actual})

    native_preflights = plan.get("native_worker_preflights")
    if not isinstance(native_preflights, list) or len(native_preflights) != 1:
        raise RuntimeError("Approved native smoke must bind exactly one worker preflight")
    native = _record_below(
        native_preflights[0],
        root=provenance_root,
        label="approved native smoke worker preflight",
    )
    if native["path"] in preflight_paths:
        raise RuntimeError("Approved native smoke reused another worker preflight")
    preflight_records.append({"key": "native-hybrid", **native})

    marker_record_after = _stable_file_record(
        marker_path,
        label="formal smoke approval marker",
    )
    if marker_record_after != marker_record_before:
        raise RuntimeError("Formal smoke approval marker changed during binding verification")
    return {
        "verification_mode": (
            "full_sha256_submitter_binding_plus_worker_semantic_validator_check"
        ),
        "marker": marker_record_after,
        "validated_plan_sha256": validated_plan_sha256,
        "smoke_plan_identity": {
            "version": plan["smoke_plan_version"],
            "sha256": smoke_plan_sha256,
        },
        "code_commit": plan["code_commit"],
        "input_manifest": input_record,
        "completed_job_keys": list(completed),
        "results": result_records,
        "worker_preflights": preflight_records,
    }


def _pinned_absolute_file_records(document: Any) -> dict[str, dict]:
    """Collect every absolute ``path/bytes/sha256`` record in the input manifest."""
    records: dict[str, dict] = {}

    def visit(value: Any, *, label: str) -> None:
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"}.issubset(value):
                raw_path = value["path"]
                byte_count = value["bytes"]
                digest = value["sha256"]
                if not isinstance(raw_path, str) or not raw_path:
                    raise RuntimeError(f"{label}.path must be a non-empty string")
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    raise RuntimeError(f"{label}.path is not absolute: {raw_path!r}")
                if (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 0
                ):
                    raise RuntimeError(f"{label}.bytes must be a non-negative integer")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise RuntimeError(f"{label}.sha256 must be a lowercase SHA-256 digest")
                canonical = str(path.resolve(strict=False))
                if canonical != raw_path:
                    raise RuntimeError(
                        f"{label}.path is not canonical: {raw_path!r} != {canonical!r}"
                    )
                record = {"path": canonical, "bytes": byte_count, "sha256": digest}
                previous = records.get(canonical)
                if previous is not None and previous != record:
                    raise RuntimeError(
                        f"Paper input manifest has conflicting records for {canonical}"
                    )
                records[canonical] = record
            for key, item in value.items():
                visit(item, label=f"{label}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, label=f"{label}[{index}]")

    visit(document, label="paper_input_manifest")
    if not records:
        raise RuntimeError("Paper input manifest contains no pinned absolute files")
    return records


def _verify_pinned_input_files(
    input_manifest: str,
    *,
    expected_input_manifest_sha256: str,
) -> tuple[dict, dict[str, dict], dict]:
    """Re-hash every dataset/protocol/metadata file pinned by the manifest."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_input_manifest_sha256):
        raise RuntimeError("--expected-input-manifest-sha256 must be a lowercase SHA-256 digest")
    input_path = Path(input_manifest).expanduser().resolve()
    manifest_before = _stable_file_record(input_path, label="paper input manifest")
    if manifest_before["sha256"] != expected_input_manifest_sha256:
        raise RuntimeError(
            "Paper input manifest digest mismatch: expected "
            f"{expected_input_manifest_sha256}, got {manifest_before['sha256']}"
        )
    document = _load_json(input_path, label="paper input manifest")
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 2
        or document.get("kind") != "hayatequant_paper_input_validation"
        or document.get("status") != "ok"
    ):
        raise RuntimeError("Paper input manifest is not an accepted schema-2 validation record")
    pinned = _pinned_absolute_file_records(document)
    verified: dict[str, dict] = {}
    for path_string, expected in sorted(pinned.items()):
        actual = _stable_file_record(Path(path_string), label="pinned paper input")
        if actual != expected:
            raise RuntimeError(f"Pinned paper input changed since validation: {path_string}")
        verified[path_string] = actual
    manifest_after = _stable_file_record(input_path, label="paper input manifest")
    if manifest_after != manifest_before:
        raise RuntimeError(f"Paper input manifest changed during verification: {input_path}")
    records_sha256 = hashlib.sha256(
        json.dumps(list(verified.values()), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "verification_mode": "full_sha256",
        "file_count": len(verified),
        "total_bytes": sum(item["bytes"] for item in verified.values()),
        "records_sha256": records_sha256,
    }
    return manifest_before, verified, summary


def _package_versions() -> dict:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # ``fuzzywuzzy`` silently switches its edit-distance implementation when
    # python-Levenshtein is installed.  That backend can affect formal
    # LongBench classification scores, so record the effective dependency,
    # including the deterministic pure-Python fallback.
    try:
        levenshtein_version = importlib.metadata.version("python-Levenshtein")
    except importlib.metadata.PackageNotFoundError:
        try:
            levenshtein_version = importlib.metadata.version("Levenshtein")
        except importlib.metadata.PackageNotFoundError:
            levenshtein_version = None
    versions["fuzzywuzzy-backend"] = (
        f"python-Levenshtein:{levenshtein_version}"
        if levenshtein_version is not None
        else "python-difflib"
    )
    return versions


def _model_provenance(
    model_path: str | None,
    *,
    input_manifest: str | None,
    expected_input_manifest_sha256: str | None,
) -> dict | None:
    if model_path is None:
        return None
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    if input_manifest is None:
        raise RuntimeError(
            "--input-manifest is required with --model; metadata-only model "
            "provenance is not accepted for paper jobs"
        )
    if expected_input_manifest_sha256 is None or not re.fullmatch(
        r"[0-9a-f]{64}", expected_input_manifest_sha256
    ):
        raise RuntimeError(
            "--expected-input-manifest-sha256 is required with --model and must "
            "be a lowercase SHA-256 digest fixed at submission time"
        )
    input_path = Path(input_manifest).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Paper input manifest does not exist: {input_path}")
    input_record_before = _stable_file_record(input_path, label="paper input manifest")
    if input_record_before["sha256"] != expected_input_manifest_sha256:
        raise RuntimeError(
            f"Paper input manifest digest mismatch: expected "
            f"{expected_input_manifest_sha256}, got {input_record_before['sha256']}"
        )
    try:
        model_key, pinned = load_input_manifest_model(input_path, path)
    except ModelWeightError as exc:
        raise RuntimeError(f"Pinned model lookup failed: {exc}") from exc
    files = {}
    for name in MODEL_METADATA_FILES:
        candidate = path / name
        if candidate.is_file():
            files[name] = _stable_file_record(candidate, label=f"model metadata {name}")
    pinned_metadata = pinned.get("metadata_files")
    if files != pinned_metadata:
        raise RuntimeError(
            f"Model metadata for {model_key!r} changed since input validation: {path}"
        )
    try:
        weights = verify_model_weights(path, pinned["weights"])
    except ModelWeightError as exc:
        raise RuntimeError(f"Pinned model weight verification failed: {exc}") from exc
    confirmed_files = {
        name: _stable_file_record(path / name, label=f"model metadata {name}") for name in files
    }
    if confirmed_files != files:
        raise RuntimeError(f"Model metadata changed during weight verification: {path}")
    input_record_after = _stable_file_record(input_path, label="paper input manifest")
    if input_record_after != input_record_before:
        raise RuntimeError(f"Paper input manifest changed during model verification: {input_path}")
    revision = None
    if (path / ".git").exists():
        revision = _run("git", "rev-parse", "HEAD", cwd=path)
    snapshot_revisions = set()
    metadata_dir = path / ".cache" / "huggingface" / "download"
    if metadata_dir.is_dir():
        for metadata in metadata_dir.glob("*.metadata"):
            try:
                first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeDecodeError, IndexError):
                continue
            if re.fullmatch(r"[0-9a-f]{40}", first_line):
                snapshot_revisions.add(first_line)
    if revision is None and len(snapshot_revisions) == 1:
        revision = next(iter(snapshot_revisions))
    return {
        "key": model_key,
        "path": str(path),
        "revision": revision,
        "snapshot_revisions": sorted(snapshot_revisions),
        "metadata_files": files,
        "weights": weights,
        "weight_verification_mode": "full_sha256",
        "input_manifest": input_record_before,
    }


def _dataset_provenance(
    paths: list[str],
    *,
    input_manifest_record: dict,
    pinned_files: dict[str, dict],
) -> list[dict]:
    records = []
    seen: set[str] = set()
    for value in paths:
        record = _stable_file_record(Path(value), label="declared dataset/protocol file")
        path_string = record["path"]
        if path_string in seen:
            raise RuntimeError(f"Dataset/protocol file was declared more than once: {path_string}")
        seen.add(path_string)
        expected = (
            input_manifest_record
            if path_string == input_manifest_record["path"]
            else pinned_files.get(path_string)
        )
        if expected is None:
            raise RuntimeError(
                f"Dataset/protocol file is not pinned by the paper input manifest: {path_string}"
            )
        if record != expected:
            raise RuntimeError(
                f"Dataset/protocol file differs from its pinned record: {path_string}"
            )
        records.append(record)
    return records


def _gpu_provenance(require_cuda: bool) -> dict:
    import torch

    available = bool(torch.cuda.is_available())
    if require_cuda and not available:
        raise RuntimeError(
            "CUDA is required for this paper job but torch.cuda.is_available() is false"
        )
    result = {
        "available": available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if available else None,
        "devices": [],
    }
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "logical_index": index,
                    "name": props.name,
                    "uuid": str(getattr(props, "uuid", "")),
                    "total_memory": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                }
            )
    query = (
        "index,uuid,name,driver_version,memory.total,memory.used,temperature.gpu,"
        "clocks.current.sm,clocks.current.memory,pstate"
    )
    smi = _run(
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        check=False,
    )
    result["nvidia_smi_query"] = smi.splitlines() if smi else []
    return result


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expected-commit", default=os.environ.get("CODE_REPO_COMMIT"))
    parser.add_argument("--run-root")
    parser.add_argument("--model")
    parser.add_argument("--input-manifest")
    parser.add_argument("--expected-input-manifest-sha256")
    parser.add_argument("--smoke-approval-marker")
    parser.add_argument("--expected-smoke-approval-marker-sha256")
    parser.add_argument("--expected-smoke-validated-plan-sha256")
    parser.add_argument("--emit-smoke-approval-binding", action="store_true")
    parser.add_argument("--dataset-file", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    commit = _run("git", "rev-parse", "HEAD", cwd=repo)
    if not isinstance(args.expected_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", args.expected_commit
    ):
        raise RuntimeError(
            "--expected-commit (or CODE_REPO_COMMIT) is required and must be a "
            "lowercase 40-hex git commit"
        )
    if commit != args.expected_commit:
        raise RuntimeError(
            f"Code revision mismatch: expected {args.expected_commit}, checked out {commit}"
        )
    # Jobs may remain pending after submission.  Recheck every visible path at
    # worker start: an untracked module or codebook can affect imports/runtime
    # lookup just as decisively as a modified tracked file.
    dirty = _run("git", "status", "--porcelain", "--untracked-files=all", cwd=repo)
    if dirty and not args.allow_dirty:
        raise RuntimeError("Tracked worktree changes are present; commit them before a paper run")

    smoke_arguments = (
        args.run_root,
        args.smoke_approval_marker,
        args.expected_input_manifest_sha256,
    )
    if args.emit_smoke_approval_binding:
        if any(value is None for value in smoke_arguments) or args.input_manifest is None:
            raise RuntimeError(
                "Smoke binding emission requires --run-root, --smoke-approval-marker, "
                "--input-manifest, and --expected-input-manifest-sha256"
            )
        binding = _verify_smoke_approval_binding(
            marker=Path(args.smoke_approval_marker),
            run_root=Path(args.run_root),
            expected_commit=args.expected_commit,
            input_manifest=Path(args.input_manifest),
            expected_input_manifest_sha256=args.expected_input_manifest_sha256,
        )
        print(
            f"{binding['marker']['sha256']} {binding['validated_plan_sha256']}",
            flush=True,
        )
        return
    if args.output is None:
        raise RuntimeError("--output is required for a paper worker preflight")

    supplied_smoke_arguments = (
        args.run_root,
        args.smoke_approval_marker,
        args.expected_smoke_approval_marker_sha256,
        args.expected_smoke_validated_plan_sha256,
    )
    if any(value is not None for value in supplied_smoke_arguments) and any(
        value is None for value in supplied_smoke_arguments
    ):
        raise RuntimeError("Formal smoke approval preflight arguments must be supplied together")

    # Import only after the caller has installed PYTHONPATH=$REPO.  Resolve both
    # paths to defeat symlink-based false positives.
    import onecomp

    module_path = Path(onecomp.__file__).resolve()
    expected_module_path = (repo / "onecomp/__init__.py").resolve(strict=True)
    if module_path != expected_module_path:
        raise RuntimeError(
            f"Imported onecomp from {module_path}, not {expected_module_path}. "
            "Set PYTHONPATH to the reproduction worktree before launching."
        )

    # Reject an invalid GPU allocation before spending up to minutes hashing a
    # large checkpoint on the compute node.
    gpu_record = _gpu_provenance(require_cuda=not args.no_cuda)
    if args.input_manifest is None or args.expected_input_manifest_sha256 is None:
        raise RuntimeError(
            "--input-manifest and --expected-input-manifest-sha256 are required "
            "for every paper job"
        )
    input_manifest_record, pinned_files, pinned_input_summary = _verify_pinned_input_files(
        args.input_manifest,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
    )
    formal_smoke_approval = None
    if all(value is not None for value in supplied_smoke_arguments):
        expected_validator_sentinel = (
            f"{args.expected_smoke_approval_marker_sha256}:"
            f"{args.expected_smoke_validated_plan_sha256}"
        )
        if os.environ.get("PAPER_SMOKE_VALIDATOR_CHECK_COMPLETED") != (
            expected_validator_sentinel
        ):
            raise RuntimeError("Formal worker has no matching completed smoke-validator check")
        formal_smoke_approval = _verify_smoke_approval_binding(
            marker=Path(args.smoke_approval_marker),
            run_root=Path(args.run_root),
            expected_commit=args.expected_commit,
            input_manifest=Path(args.input_manifest),
            expected_input_manifest_sha256=args.expected_input_manifest_sha256,
            expected_marker_sha256=args.expected_smoke_approval_marker_sha256,
            expected_validated_plan_sha256=args.expected_smoke_validated_plan_sha256,
        )
    model_record = _model_provenance(
        args.model,
        input_manifest=args.input_manifest,
        expected_input_manifest_sha256=args.expected_input_manifest_sha256,
    )
    dataset_records = _dataset_provenance(
        args.dataset_file,
        input_manifest_record=input_manifest_record,
        pinned_files=pinned_files,
    )
    if dataset_records.count(input_manifest_record) != 1:
        raise RuntimeError("--input-manifest must also be supplied exactly once as --dataset-file")
    if model_record is not None:
        if model_record["input_manifest"] != input_manifest_record:
            raise RuntimeError(
                "Paper input manifest changed between dataset and model verification"
            )

    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": {
            "repo": str(repo),
            "commit": commit,
            "tracked_dirty": bool(dirty),
            "onecomp_module": str(module_path),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "packages": _package_versions(),
        "gpu": gpu_record,
        "model": model_record,
        "pinned_input_files": pinned_input_summary,
        "dataset_files": dataset_records,
        "environment": {key: os.environ.get(key) for key in ENV_ALLOWLIST},
    }
    if formal_smoke_approval is not None:
        document["formal_smoke_approval"] = formal_smoke_approval
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, document)
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
