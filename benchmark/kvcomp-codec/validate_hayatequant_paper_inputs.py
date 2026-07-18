#!/usr/bin/env python3
"""Validate every immutable input used by the HayateQuant paper rerun.

This command is intentionally offline and read-only with respect to benchmark
inputs.  It validates and SHA-256 fingerprints model metadata, every supported
weight index and every referenced weight shard, generated NeMo-Skills RULER v1
100-sample/task protocol data (distinct from canonical NVIDIA 500/task data),
the pinned LongCodeQA release, and, when present, the manifest produced by
:mod:`prepare_paper_datasets`.  The sole write is an atomic validation manifest
that GPU jobs pass to ``run_paper_job.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Tests load this script directly via importlib rather than executing it, so its
# sibling helper directory is not guaranteed to be present on sys.path.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from model_weight_manifest import (  # noqa: E402
    ModelWeightError,
    fingerprint_model_weights,
    verify_model_weights as verify_model_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = Path(
    os.environ.get("KVCOMP_RUN_ROOT", str(REPO_ROOT / "results" / "hayatequant" / "formal"))
)
DEFAULT_MODELS_ROOT = (
    Path(os.environ["KVCOMP_MODELS_ROOT"]) if os.environ.get("KVCOMP_MODELS_ROOT") else None
)

DEFAULT_MODEL_PATHS = {
    "llama2-7b": "Llama-2-7b-hf",
    "llama31-8b": "llama-3.1-8b-instruct",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "r1-llama-8b": "DeepSeek-R1-Distill-Llama-8B",
    "r1-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B",
    "r1-qwen-1.5b": "DeepSeek-R1-Distill-Qwen-1.5B",
    # Capped smoke jobs exercise evaluator and generation paths with this model.
    "qwen3-0.6b-smoke": "Qwen3-0.6B",
}

RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_LENGTHS = (4096, 8192, 16384)
RULER_ROWS_PER_TASK = 100
RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK = 500
RULER_PROTOCOL_LABEL = "NeMo-Skills RULER v1 100-sample/task protocol"
RULER_SAMPLING_PROTOCOL = (
    "pinned NeMo-Skills adapter: 100 samples/task; canonical "
    "NVIDIA/RULER v1 config: 500 samples/task"
)
RULER_SOURCE_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
NEMO_SKILLS_SOURCE_COMMIT = "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3"
RULER_MODEL_KEYS = ("llama31-8b", "qwen3-8b")

LONGCODEQA_COUNTS = {"32K": 113, "64K": 76, "128K": 92}
LONGCODEQA_SHA256 = {
    "32K": "a4de950aa887c97ed74d64d50a1202b28e566105b9b94e62b7c1a50a48b46d98",
    "64K": "c4c77b52582187753f94d4e3e78d68b6d2549791496b7305fc93b558166bdc74",
    "128K": "c4b44c40f4a2efbeea6f69c210b80ca9a90d3156f032335fcb1182eb2992e1d8",
}
LONGCODEQA_CONTEXT_SPECS = {
    "llama31-8b": {"context_limit": 131072, "brackets": ("32K", "64K", "128K")},
    "qwen3-8b": {"context_limit": 40960, "brackets": ("32K",)},
}
LONGCODEQA_CONTEXT_MAX_NEW_TOKENS = 16
LONGCODEQA_DATASET_REVISION = "989d5eff750d65a72c522e47f8f745ef2e22906b"
LONGCODEQA_PROTOCOL_COMMIT = "91db30f5144363bb809534ec7e4a84a56ea43987"
LONGCODEQA_DISCUSSION_URL = "https://huggingface.co/datasets/Steefano/LCB/discussions/1"
LONGCODEQA_ANSWER_PARSER = (
    "first /Final Answer:\\s*([ABCD])/i; otherwise trailing " "/\\b([ABCD])\\s*$/i"
)
LONGCODEQA_PROTOCOL_SCOPE = (
    "release exact dataset, prompt, parser, and per-bracket accuracy; "
    "preregistered paired greedy decoding adaptation"
)

# Keep these pins in lockstep with prepare_paper_datasets.py.  The validator
# duplicates the compact protocol table deliberately: validation must not run
# dataset library code or import a downloader on an offline GPU node.
HF_DATASETS = {
    "wikitext2_test": {
        "repo": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "expected_rows": 4358,
    },
    "arc_easy_test": {
        "repo": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "expected_rows": 2376,
    },
    "arc_challenge_test": {
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "expected_rows": 1172,
    },
    "piqa_validation": {
        "repo": "baber/piqa",
        "config": None,
        "split": "validation",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "expected_rows": 1838,
    },
    "hellaswag_validation": {
        "repo": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "expected_rows": 10042,
    },
    "winogrande_xl_validation": {
        "repo": "allenai/winogrande",
        "config": "winogrande_xl",
        "split": "validation",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "expected_rows": 1267,
    },
    "math500_test": {
        "repo": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "expected_rows": 500,
    },
    "gsm8k_main_test": {
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "expected_rows": 1319,
    },
    "aime2024_source": {
        "repo": "AI-MO/aimo-validation-aime",
        "config": None,
        "split": "train",
        "revision": "13f9e12f613e720c2a2b2f345dd04b998a29494d",
        "expected_rows": None,
    },
    "aime2025_train": {
        "repo": "yentinglin/aime_2025",
        "config": None,
        "split": "train",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "expected_rows": 30,
    },
}
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"

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
TOKENIZER_PAYLOADS = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class ValidationError(RuntimeError):
    """An input is absent, mutable, malformed, or not pinned as expected."""


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
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValidationError(f"non-finite JSON number {value!r}")


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except ValidationError as exc:
        raise ValidationError(f"{label} contains a {exc}: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def _object(value: Any, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _eos_token_ids(value: Any, *, label: str) -> list[int]:
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValidationError(f"{label} must contain non-negative integer token IDs")
        if item not in result:
            result.append(item)
    if not result:
        raise ValidationError(f"{label} must not be empty")
    return result


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _existing_canonical_directory(path: str | Path, *, label: str) -> Path:
    """Resolve a directory while rejecting symlinks in any path component."""
    raw = Path(path).expanduser()
    lexical = Path(os.path.abspath(raw))
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"{label} is not an existing directory: {lexical}") from exc
    if lexical != resolved:
        raise ValidationError(f"{label} must not contain symlink path components: {lexical}")
    if not resolved.is_dir():
        raise ValidationError(f"{label} is not a directory: {resolved}")
    return resolved


def _validate_runtime_paths(
    *, run_root: Path, work_root: str | Path, runtime_home: str | Path
) -> dict:
    """Bind mutable runtime state without allowing it inside the evidence tree."""
    work = _existing_canonical_directory(work_root, label="WORK_ROOT")
    home = _existing_canonical_directory(runtime_home, label="runtime HOME")
    expected_home = (work / "runtime" / "home").resolve()
    if home != expected_home:
        raise ValidationError(f"runtime HOME must be exactly {expected_home}")
    if _inside(work, run_root) or _inside(run_root, work):
        raise ValidationError("WORK_ROOT and RUN_ROOT must be disjoint directory trees")
    return {"work_root": str(work), "home": str(home)}


def _state_tuple(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_local_file_record(path: Path, *, label: str) -> dict:
    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise ValidationError(f"{label} is missing or cannot be resolved: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValidationError(f"{label} is not a regular file: {resolved}")
    digest = _sha256(resolved)
    try:
        after = resolved.stat()
    except OSError as exc:
        raise ValidationError(f"{label} disappeared while hashing: {resolved}") from exc
    if _state_tuple(before) != _state_tuple(after):
        raise ValidationError(f"{label} changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def _local_file_record(path: Path) -> dict:
    return _stable_local_file_record(path, label="required file")


def _validate_declared_file(
    record: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    required_root: Path | None = None,
) -> dict:
    item = _object(record, label=label)
    declared_path = item.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValidationError(f"{label}.path must be a non-empty string")
    path = _resolved(declared_path)
    if expected_path is not None and path != expected_path:
        raise ValidationError(f"{label}.path points to {path}, expected {expected_path}")
    if required_root is not None and not _inside(path, required_root):
        raise ValidationError(f"{label}.path escapes the pinned root {required_root}: {path}")
    declared_bytes = _integer(item.get("bytes"), label=f"{label}.bytes")
    actual = _stable_local_file_record(path, label=label)
    if declared_bytes != actual["bytes"]:
        raise ValidationError(
            f"{label}.bytes mismatch for {path}: declared {declared_bytes}, "
            f"actual {actual['bytes']}"
        )
    declared_digest = item.get("sha256")
    if not isinstance(declared_digest, str) or not _SHA256_RE.fullmatch(declared_digest):
        raise ValidationError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if declared_digest != actual["sha256"]:
        raise ValidationError(
            f"{label}.sha256 mismatch for {path}: declared {declared_digest}, "
            f"actual {actual['sha256']}"
        )
    return actual


def validate_models(model_paths: Mapping[str, Path]) -> dict:
    """Validate and fingerprint all metadata and weights in a model matrix."""
    if not model_paths:
        raise ValidationError("At least one model must be declared")
    records = {}
    for key, raw_path in sorted(model_paths.items()):
        if not key or not isinstance(key, str):
            raise ValidationError(f"Invalid model key: {key!r}")
        path = _resolved(raw_path)
        if not path.is_dir():
            raise ValidationError(f"Model directory for {key!r} is missing: {path}")
        for required in ("config.json", "tokenizer_config.json"):
            document = _load_json(path / required, label=f"{key} {required}")
            if not isinstance(document, dict) or not document:
                raise ValidationError(f"{key} {required} must be a non-empty JSON object")
        if not any((path / name).is_file() for name in TOKENIZER_PAYLOADS):
            raise ValidationError(
                f"Model {key!r} has no tokenizer payload; expected one of {TOKENIZER_PAYLOADS}"
            )
        metadata = {
            name: _local_file_record(path / name)
            for name in MODEL_METADATA_FILES
            if (path / name).is_file()
        }
        try:
            weights = fingerprint_model_weights(path)
        except ModelWeightError as exc:
            raise ValidationError(f"Model {key!r} weight validation failed: {exc}") from exc
        confirmed_metadata = {
            name: _local_file_record(path / name)
            for name in MODEL_METADATA_FILES
            if (path / name).is_file()
        }
        if confirmed_metadata != metadata:
            raise ValidationError(
                f"Model {key!r} metadata inventory changed while fingerprinting weights"
            )
        records[key] = {
            "path": str(path),
            "metadata_files": metadata,
            "weights": weights,
        }
    return records


def _validate_ruler_rows(path: Path, *, sequence_length: int) -> dict:
    digest = hashlib.sha256()
    rows = 0
    template_lengths = []
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"RULER task path is not a regular file: {path}")
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                digest.update(line)
                if not line.strip():
                    raise ValidationError(f"RULER file has a blank row: {path}:{line_number}")
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_keys,
                        parse_constant=_reject_nonfinite,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValidationError(
                        f"Invalid RULER JSONL row: {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValidationError(f"RULER row must be an object: {path}:{line_number}")
                required = {
                    "index",
                    "input",
                    "answer_prefix",
                    "outputs",
                    "length",
                    "length_w_model_temp",
                }
                missing = required - set(row)
                if missing:
                    raise ValidationError(
                        f"RULER row is missing {sorted(missing)}: {path}:{line_number}"
                    )
                # NVIDIA/RULER's synthetic generators use this as a source-sample
                # identifier, not as a row number. Some upstream NIAH outputs
                # intentionally contain repeated, non-monotonic identifiers.  The
                # manifest's full-file digest binds both row order and contents.
                _integer(row["index"], label=f"{path}:{line_number} index", minimum=0)
                if not isinstance(row["input"], str) or not row["input"]:
                    raise ValidationError(f"RULER input must be non-empty: {path}:{line_number}")
                if not isinstance(row["answer_prefix"], str):
                    raise ValidationError(
                        f"RULER answer_prefix must be a string: {path}:{line_number}"
                    )
                outputs = row["outputs"]
                if (
                    not isinstance(outputs, list)
                    or not outputs
                    or any(not isinstance(item, str) or not item for item in outputs)
                ):
                    raise ValidationError(
                        f"RULER outputs must contain non-empty strings: {path}:{line_number}"
                    )
                _integer(row["length"], label=f"{path}:{line_number} length", minimum=1)
                templated = _integer(
                    row["length_w_model_temp"],
                    label=f"{path}:{line_number} length_w_model_temp",
                    minimum=1,
                )
                if templated > sequence_length:
                    raise ValidationError(
                        f"RULER templated length {templated} exceeds {sequence_length}: "
                        f"{path}:{line_number}"
                    )
                template_lengths.append(templated)
                rows += 1
        after = path.stat()
    except OSError as exc:
        raise ValidationError(f"Unable to read RULER task file {path}: {exc}") from exc
    if _state_tuple(before) != _state_tuple(after):
        raise ValidationError(f"RULER task file changed while validating: {path}")
    if rows != RULER_ROWS_PER_TASK:
        raise ValidationError(f"RULER file {path} has {rows} rows, expected {RULER_ROWS_PER_TASK}")
    return {
        "path": str(path),
        "bytes": after.st_size,
        "sha256": digest.hexdigest(),
        "rows": rows,
        "length_w_model_temp_min": min(template_lengths),
        "length_w_model_temp_max": max(template_lengths),
    }


def validate_ruler(ruler_root: Path, models: Mapping[str, dict]) -> dict:
    ruler_root = _resolved(ruler_root)
    records = {}
    for model_key in RULER_MODEL_KEYS:
        if model_key not in models:
            raise ValidationError(f"RULER requires model {model_key!r} in the model matrix")
        model_root = ruler_root / model_key
        manifest_path = model_root / "manifest.json"
        manifest_record = _local_file_record(manifest_path)
        manifest = _object(
            _load_json(manifest_path, label=f"{model_key} RULER manifest"),
            label=f"{model_key} RULER manifest",
        )
        if manifest.get("schema_version") != 1:
            raise ValidationError(f"{model_key} RULER manifest schema_version must be 1")
        if manifest.get("benchmark") != RULER_PROTOCOL_LABEL:
            raise ValidationError(f"{model_key} manifest is not {RULER_PROTOCOL_LABEL}")
        sources = _object(manifest.get("sources"), label=f"{model_key} sources")
        if sources.get("ruler_commit") != RULER_SOURCE_COMMIT:
            raise ValidationError(
                f"{model_key} RULER commit is not the audited {RULER_SOURCE_COMMIT}"
            )
        if sources.get("nemo_skills_commit") != NEMO_SKILLS_SOURCE_COMMIT:
            raise ValidationError(
                f"{model_key} NeMo-Skills commit is not the audited "
                f"{NEMO_SKILLS_SOURCE_COMMIT}"
            )

        protocol = _object(manifest.get("protocol"), label=f"{model_key} protocol")
        if protocol.get("num_samples_per_task") != RULER_ROWS_PER_TASK:
            raise ValidationError(f"{model_key} RULER num_samples_per_task must be 100")
        if protocol.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL:
            raise ValidationError(f"{model_key} RULER sampling protocol label drifted")
        if protocol.get("prepare_for_nemo_skills") is not True:
            raise ValidationError(f"{model_key} RULER data must use prepare_for_nemo_skills=true")
        if protocol.get("tokenizer_type") != "hf":
            raise ValidationError(f"{model_key} RULER tokenizer_type must be 'hf'")
        if protocol.get("random_seed") != 42:
            raise ValidationError(f"{model_key} RULER random_seed must be 42")
        if protocol.get("chat_template_reserve") != 50:
            raise ValidationError(f"{model_key} RULER chat_template_reserve must be 50")
        lengths = protocol.get("lengths")
        if (
            not isinstance(lengths, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in lengths)
            or len(lengths) != len(set(lengths))
            or lengths != list(RULER_LENGTHS)
        ):
            raise ValidationError(
                f"{model_key} RULER lengths must be exactly {list(RULER_LENGTHS)}"
            )
        tasks = protocol.get("tasks")
        if (
            not isinstance(tasks, list)
            or any(not isinstance(task, str) for task in tasks)
            or len(tasks) != len(set(tasks))
            or tasks != list(RULER_TASKS)
        ):
            raise ValidationError(f"{model_key} RULER task set must be the pinned 13 tasks")

        model_info = _object(manifest.get("model"), label=f"{model_key} model")
        if model_info.get("key") != model_key:
            raise ValidationError(f"{model_key} RULER manifest model.key mismatch")
        declared_model_path = model_info.get("path")
        if not isinstance(declared_model_path, str) or _resolved(declared_model_path) != Path(
            models[model_key]["path"]
        ):
            raise ValidationError(f"{model_key} RULER manifest model.path mismatch")
        revision = model_info.get("revision")
        if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
            raise ValidationError(f"{model_key} RULER model.revision must be a 40-hex commit")
        snapshots = model_info.get("snapshot_revisions")
        if not isinstance(snapshots, list) or revision not in snapshots:
            raise ValidationError(f"{model_key} RULER snapshot_revisions must contain revision")
        declared_metadata = _object(
            model_info.get("metadata_files"), label=f"{model_key} metadata_files"
        )
        available_metadata = models[model_key]["metadata_files"]
        required_metadata = {"config.json", "tokenizer_config.json"}
        if not required_metadata.issubset(declared_metadata) or not any(
            name in declared_metadata for name in TOKENIZER_PAYLOADS
        ):
            raise ValidationError(
                f"{model_key} RULER metadata_files must pin config.json, "
                "tokenizer_config.json, and a tokenizer payload"
            )
        for filename, declared in declared_metadata.items():
            if filename not in MODEL_METADATA_FILES or filename not in available_metadata:
                raise ValidationError(
                    f"{model_key} RULER manifest declares unsupported/missing metadata {filename!r}"
                )
            local = available_metadata[filename]
            item = _object(declared, label=f"{model_key} metadata {filename}")
            if item.get("bytes") != local["bytes"] or item.get("sha256") != local["sha256"]:
                raise ValidationError(
                    f"{model_key} metadata {filename} changed since RULER generation"
                )

        groups = _object(manifest.get("files"), label=f"{model_key} files")
        if set(groups) != {str(length) for length in RULER_LENGTHS}:
            raise ValidationError(f"{model_key} RULER file groups must cover exactly 4K/8K/16K")
        validated_groups = {}
        for length in RULER_LENGTHS:
            length_key = str(length)
            group = _object(groups[length_key], label=f"{model_key} files[{length_key}]")
            if set(group) != set(RULER_TASKS):
                raise ValidationError(
                    f"{model_key} RULER {length_key} must contain exactly all 13 tasks"
                )
            task_records = {}
            for task in RULER_TASKS:
                expected_path = (model_root / length_key / task / "test.jsonl").resolve()
                entry = _object(group[task], label=f"{model_key}/{length_key}/{task}")
                declared_path = entry.get("path")
                if not isinstance(declared_path, str) or _resolved(declared_path) != expected_path:
                    raise ValidationError(
                        f"{model_key}/{length_key}/{task} path must be {expected_path}"
                    )
                if not expected_path.is_file():
                    raise ValidationError(f"RULER task file is missing: {expected_path}")
                actual = _validate_ruler_rows(expected_path, sequence_length=length)
                for field in (
                    "bytes",
                    "sha256",
                    "rows",
                    "length_w_model_temp_min",
                    "length_w_model_temp_max",
                ):
                    if entry.get(field) != actual[field]:
                        raise ValidationError(
                            f"{model_key}/{length_key}/{task} manifest {field} mismatch: "
                            f"declared {entry.get(field)!r}, actual {actual[field]!r}"
                        )
                task_records[task] = actual
            validated_groups[length_key] = task_records
        confirmed_manifest_record = _local_file_record(manifest_path)
        if confirmed_manifest_record != manifest_record:
            raise ValidationError(f"{model_key} RULER manifest changed during validation")
        records[model_key] = {
            "manifest": manifest_record,
            "benchmark": RULER_PROTOCOL_LABEL,
            "samples_per_task": RULER_ROWS_PER_TASK,
            "sampling_protocol": RULER_SAMPLING_PROTOCOL,
            "canonical_nvidia_samples_per_task": RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK,
            "canonical_nvidia_full_eval": False,
            "source_commits": {
                "ruler": RULER_SOURCE_COMMIT,
                "nemo_skills": NEMO_SKILLS_SOURCE_COMMIT,
            },
            "lengths": validated_groups,
            "total_rows": len(RULER_LENGTHS) * len(RULER_TASKS) * RULER_ROWS_PER_TASK,
        }
    return records


def validate_longcodeqa(
    data_root: Path,
    *,
    expected_sha256: Mapping[str, str] = LONGCODEQA_SHA256,
) -> dict:
    data_root = _resolved(data_root)
    records = {}
    for bracket, expected_count in LONGCODEQA_COUNTS.items():
        path = data_root / "LQA" / f"{bracket}.json"
        file_record = _local_file_record(path)
        rows = _load_json(path, label=f"LongCodeQA {bracket}")
        if not isinstance(rows, list):
            raise ValidationError(f"LongCodeQA {bracket} must be a JSON list")
        if len(rows) != expected_count:
            raise ValidationError(
                f"LongCodeQA {bracket} has {len(rows)} rows, expected {expected_count}"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValidationError(f"LongCodeQA {bracket} row {index} must be an object")
            required = {"prompt", "correct_letter", "repo", "question"}
            missing = required - set(row)
            if missing:
                raise ValidationError(
                    f"LongCodeQA {bracket} row {index} is missing {sorted(missing)}"
                )
            for field in ("prompt", "repo", "question"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValidationError(f"LongCodeQA {bracket} row {index} has invalid {field}")
            if row["correct_letter"] not in {"A", "B", "C", "D"}:
                raise ValidationError(
                    f"LongCodeQA {bracket} row {index} has invalid correct_letter"
                )
        confirmed_file_record = _local_file_record(path)
        if confirmed_file_record != file_record:
            raise ValidationError(f"LongCodeQA {bracket} changed during validation")
        digest = file_record["sha256"]
        pinned_digest = expected_sha256.get(bracket)
        if not isinstance(pinned_digest, str) or not _SHA256_RE.fullmatch(pinned_digest):
            raise ValidationError(f"No valid pinned SHA-256 configured for LongCodeQA {bracket}")
        if digest != pinned_digest:
            raise ValidationError(
                f"LongCodeQA {bracket} SHA-256 mismatch: expected {pinned_digest}, actual {digest}"
            )
        records[bracket] = {
            "path": file_record["path"],
            "bytes": file_record["bytes"],
            "sha256": digest,
            "rows": len(rows),
        }
    return records


def _validate_context_model(
    model_key: str,
    value: Any,
    *,
    expected_context: int,
    model_record: Mapping[str, Any],
) -> dict:
    model = _object(value, label=f"{model_key} context-fit model")
    if "key" in model and model["key"] != model_key:
        raise ValidationError(f"{model_key} context-fit manifest model.key is {model['key']!r}")
    declared_path = model.get("path")
    expected_path = Path(model_record["path"])
    if not isinstance(declared_path, str) or _resolved(declared_path) != expected_path:
        raise ValidationError(f"{model_key} context-fit model.path must be {expected_path}")
    if model.get("context_limit") != expected_context:
        raise ValidationError(f"{model_key} context-fit context_limit must be {expected_context}")

    config = _object(
        _load_json(expected_path / "config.json", label=f"{model_key} config.json"),
        label=f"{model_key} config.json",
    )
    if config.get("max_position_embeddings") != expected_context:
        raise ValidationError(
            f"{model_key} config.json max_position_embeddings must be {expected_context}"
        )
    available_metadata = model_record["metadata_files"]
    required_metadata = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for filename in required_metadata:
        declared = _object(model.get(filename), label=f"{model_key} context-fit model.{filename}")
        local = available_metadata.get(filename)
        if local is None:
            raise ValidationError(f"{model_key} model metadata is missing {filename}")
        if declared.get("bytes") != local["bytes"] or declared.get("sha256") != local["sha256"]:
            raise ValidationError(
                f"{model_key} context-fit tokenizer/model metadata {filename} mismatch"
            )

    revision = model.get("revision")
    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        raise ValidationError(f"{model_key} context-fit model.revision must be a 40-hex commit")
    snapshots = model.get("snapshot_revisions")
    if not isinstance(snapshots, list) or revision not in snapshots:
        raise ValidationError(f"{model_key} context-fit snapshot_revisions must contain revision")

    tokenizer_config = _object(
        _load_json(
            expected_path / "tokenizer_config.json",
            label=f"{model_key} tokenizer_config.json",
        ),
        label=f"{model_key} tokenizer_config.json",
    )
    chat_template = tokenizer_config.get("chat_template")
    expected_chat_digest = (
        hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if isinstance(chat_template, str)
        else None
    )
    if model.get("chat_template_sha256") != expected_chat_digest:
        raise ValidationError(
            f"{model_key} context-fit chat_template_sha256 does not match tokenizer metadata"
        )
    generation_config = _object(
        _load_json(
            expected_path / "generation_config.json",
            label=f"{model_key} generation_config.json",
        ),
        label=f"{model_key} generation_config.json",
    )
    expected_eos_token_ids = _eos_token_ids(
        generation_config.get("eos_token_id"),
        label=f"{model_key} generation_config.eos_token_id",
    )
    if model.get("generation_eos_token_ids") != expected_eos_token_ids:
        raise ValidationError(
            f"{model_key} context-fit generation_eos_token_ids does not match "
            "generation_config.json"
        )
    if model.get("generation_eos_token_ids_source") != "model.generation_config.eos_token_id":
        raise ValidationError(
            f"{model_key} context-fit generation_eos_token_ids_source is not pinned"
        )
    return {
        "key": model_key,
        "path": str(expected_path),
        "context_limit": expected_context,
        "revision": revision,
        "snapshot_revisions": snapshots,
        "chat_template_sha256": expected_chat_digest,
        "generation_eos_token_ids": expected_eos_token_ids,
        "generation_eos_token_ids_source": ("model.generation_config.eos_token_id"),
        "metadata_files": {
            filename: available_metadata[filename] for filename in required_metadata
        },
    }


def validate_longcodeqa_context_manifests(
    manifest_paths: Mapping[str, Path],
    *,
    models: Mapping[str, Mapping[str, Any]],
    longcodeqa: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Validate the tokenizer-specific proof that every scheduled prompt fits."""
    if set(manifest_paths) != set(LONGCODEQA_CONTEXT_SPECS):
        missing = sorted(set(LONGCODEQA_CONTEXT_SPECS) - set(manifest_paths))
        extra = sorted(set(manifest_paths) - set(LONGCODEQA_CONTEXT_SPECS))
        raise ValidationError(
            f"LongCodeQA context-fit manifest set mismatch; missing={missing}, extra={extra}"
        )
    records = {}
    for model_key, spec in LONGCODEQA_CONTEXT_SPECS.items():
        if model_key not in models:
            raise ValidationError(
                f"LongCodeQA context-fit validation requires model {model_key!r}"
            )
        manifest_path = _resolved(manifest_paths[model_key])
        manifest_record = _local_file_record(manifest_path)
        manifest = _object(
            _load_json(manifest_path, label=f"{model_key} LongCodeQA context-fit manifest"),
            label=f"{model_key} LongCodeQA context-fit manifest",
        )
        required_top_level = {
            "schema_version",
            "created_at_utc",
            "benchmark",
            "model",
            "protocol",
            "brackets",
            "all_rows_fit",
        }
        if set(manifest) != required_top_level:
            raise ValidationError(
                f"{model_key} context-fit top-level schema mismatch; "
                f"expected {sorted(required_top_level)}, got {sorted(manifest)}"
            )
        if manifest.get("schema_version") != 2:
            raise ValidationError(f"{model_key} context-fit schema_version must be 2")
        if manifest.get("benchmark") != "LongCodeBench/LongCodeQA context-fit validation":
            raise ValidationError(f"{model_key} manifest is not LongCodeQA context-fit validation")
        created_at = manifest.get("created_at_utc")
        if not isinstance(created_at, str):
            raise ValidationError(f"{model_key} context-fit created_at_utc must be a string")
        try:
            parsed_created_at = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValidationError(
                f"{model_key} context-fit created_at_utc is not ISO-8601"
            ) from exc
        if parsed_created_at.tzinfo is None:
            raise ValidationError(f"{model_key} context-fit created_at_utc needs a timezone")

        expected_context = spec["context_limit"]
        validated_model = _validate_context_model(
            model_key,
            manifest.get("model"),
            expected_context=expected_context,
            model_record=models[model_key],
        )
        protocol = _object(manifest.get("protocol"), label=f"{model_key} context-fit protocol")
        expected_protocol = {
            "scope": LONGCODEQA_PROTOCOL_SCOPE,
            "dataset_revision": LONGCODEQA_DATASET_REVISION,
            "prompt": "released prompt verbatim as raw causal text",
            "prompt_release_clarification": LONGCODEQA_DISCUSSION_URL,
            "chat_template": "none",
            "truncation": "none",
            "answer_parser_reference_commit": LONGCODEQA_PROTOCOL_COMMIT,
            "answer_parser": LONGCODEQA_ANSWER_PARSER,
            "reference_evaluator_executable_verbatim": False,
            "reference_evaluator_known_issue": (
                "pinned evaluator stores response but reads prediction text"
            ),
            "generation": {
                "protocol": "hayatequant deterministic greedy paired comparison",
                "do_sample": False,
                "max_new_tokens": LONGCODEQA_CONTEXT_MAX_NEW_TOKENS,
                "seed": None,
                "eos_token_ids_source": "model.generation_config.eos_token_id",
                "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
                "official_reference_max_output_length": None,
                "official_reference_uniform_decoding_config": False,
                "official_reference_decoding_exact": False,
            },
        }
        if protocol != expected_protocol:
            raise ValidationError(
                f"{model_key} context-fit protocol does not match the audited protocol"
            )
        if manifest.get("all_rows_fit") is not True:
            raise ValidationError(f"{model_key} context-fit manifest reports overflow")

        brackets = _object(manifest.get("brackets"), label=f"{model_key} context-fit brackets")
        expected_brackets = tuple(spec["brackets"])
        if set(brackets) != set(expected_brackets):
            raise ValidationError(
                f"{model_key} context-fit brackets must be exactly {list(expected_brackets)}"
            )
        validated_brackets = {}
        for bracket in expected_brackets:
            expected_dataset = longcodeqa[bracket]
            record = _object(brackets[bracket], label=f"{model_key} context-fit {bracket}")
            required_bracket_fields = {
                "rows",
                "dataset_path",
                "dataset_sha256",
                "input_tokens",
                "minimum_margin_tokens",
                "overflow_rows",
                "per_row",
            }
            if set(record) != required_bracket_fields:
                raise ValidationError(f"{model_key}/{bracket} context-fit schema mismatch")
            expected_rows = LONGCODEQA_COUNTS[bracket]
            if record.get("rows") != expected_rows:
                raise ValidationError(
                    f"{model_key}/{bracket} context-fit rows must be {expected_rows}"
                )
            dataset_path = record.get("dataset_path")
            if not isinstance(dataset_path, str) or _resolved(dataset_path) != Path(
                expected_dataset["path"]
            ):
                raise ValidationError(f"{model_key}/{bracket} context-fit dataset_path mismatch")
            if record.get("dataset_sha256") != expected_dataset["sha256"]:
                raise ValidationError(f"{model_key}/{bracket} context-fit dataset_sha256 mismatch")
            if record.get("overflow_rows") != []:
                raise ValidationError(
                    f"{model_key}/{bracket} context-fit overflow_rows must be empty"
                )
            per_row = record.get("per_row")
            if not isinstance(per_row, list) or len(per_row) != expected_rows:
                raise ValidationError(
                    f"{model_key}/{bracket} context-fit per_row must contain {expected_rows} rows"
                )
            input_lengths = []
            margins = []
            for expected_index, raw_row in enumerate(per_row):
                row = _object(
                    raw_row,
                    label=f"{model_key}/{bracket} context-fit per_row[{expected_index}]",
                )
                if set(row) != {"index", "input_tokens", "margin"}:
                    raise ValidationError(
                        f"{model_key}/{bracket} per_row[{expected_index}] schema mismatch"
                    )
                if row.get("index") != expected_index:
                    raise ValidationError(
                        f"{model_key}/{bracket} per_row index sequence is invalid at "
                        f"position {expected_index}"
                    )
                input_tokens = _integer(
                    row.get("input_tokens"),
                    label=f"{model_key}/{bracket} row {expected_index} input_tokens",
                    minimum=1,
                )
                margin = row.get("margin")
                if isinstance(margin, bool) or not isinstance(margin, int):
                    raise ValidationError(
                        f"{model_key}/{bracket} row {expected_index} margin must be an integer"
                    )
                expected_margin = (
                    expected_context - input_tokens - LONGCODEQA_CONTEXT_MAX_NEW_TOKENS
                )
                if margin != expected_margin:
                    raise ValidationError(
                        f"{model_key}/{bracket} row {expected_index} margin mismatch: "
                        f"expected {expected_margin}, got {margin}"
                    )
                if expected_margin < 0:
                    raise ValidationError(
                        f"{model_key}/{bracket} row {expected_index} overflows the "
                        f"{expected_context}-token model context"
                    )
                input_lengths.append(input_tokens)
                margins.append(margin)

            input_summary = _object(
                record.get("input_tokens"),
                label=f"{model_key}/{bracket} input_tokens summary",
            )
            expected_summary = {
                "min": min(input_lengths),
                "median": statistics.median(input_lengths),
                "max": max(input_lengths),
            }
            if input_summary != expected_summary:
                raise ValidationError(f"{model_key}/{bracket} input_tokens summary mismatch")
            minimum_margin = min(margins)
            if record.get("minimum_margin_tokens") != minimum_margin:
                raise ValidationError(
                    f"{model_key}/{bracket} minimum_margin_tokens must be {minimum_margin}"
                )
            validated_brackets[bracket] = {
                "rows": expected_rows,
                "dataset_path": expected_dataset["path"],
                "dataset_sha256": expected_dataset["sha256"],
                "input_tokens": expected_summary,
                "minimum_margin_tokens": minimum_margin,
                "overflow_rows": [],
            }
        confirmed_manifest_record = _local_file_record(manifest_path)
        if confirmed_manifest_record != manifest_record:
            raise ValidationError(f"{model_key} context-fit manifest changed during validation")
        records[model_key] = {
            "manifest": manifest_record,
            "created_at_utc": created_at,
            "model": validated_model,
            "protocol": expected_protocol,
            "all_rows_fit": True,
            "brackets": validated_brackets,
        }
    return records


def validate_hf_manifest(
    manifest_path: Path,
    hf_home: Path,
    *,
    expected_datasets: Mapping[str, Mapping[str, Any]] = HF_DATASETS,
) -> dict:
    """Validate a pre-produced HF manifest and every file it fingerprints."""
    manifest_path = _resolved(manifest_path)
    hf_home = _resolved(hf_home)
    manifest_record = _local_file_record(manifest_path)
    manifest = _object(
        _load_json(manifest_path, label="pinned HF dataset manifest"),
        label="HF manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValidationError("Pinned HF dataset manifest schema_version must be 1")
    declared_home = manifest.get("hf_home")
    if not isinstance(declared_home, str) or _resolved(declared_home) != hf_home:
        raise ValidationError(f"Pinned HF manifest hf_home must be {hf_home}")
    packages = _object(manifest.get("packages"), label="HF manifest packages")
    for package in ("datasets", "huggingface-hub", "pyarrow"):
        if not isinstance(packages.get(package), str) or not packages[package]:
            raise ValidationError(f"HF manifest packages.{package} must be recorded")

    datasets = _object(manifest.get("datasets"), label="HF manifest datasets")
    if set(datasets) != set(expected_datasets):
        missing = sorted(set(expected_datasets) - set(datasets))
        extra = sorted(set(datasets) - set(expected_datasets))
        raise ValidationError(
            f"HF manifest dataset set mismatch; missing={missing}, extra={extra}"
        )
    validated_datasets = {}
    for key, expected in expected_datasets.items():
        record = _object(datasets[key], label=f"HF dataset {key}")
        for field in ("repo", "config", "split", "revision", "expected_rows"):
            if record.get(field) != expected[field]:
                raise ValidationError(
                    f"HF dataset {key}.{field} mismatch: "
                    f"expected {expected[field]!r}, got {record.get(field)!r}"
                )
        rows = _integer(record.get("rows"), label=f"HF dataset {key}.rows", minimum=1)
        expected_rows = expected["expected_rows"]
        if expected_rows is not None and rows != expected_rows:
            raise ValidationError(f"HF dataset {key} has {rows} rows, expected {expected_rows}")
        if (
            not isinstance(record.get("dataset_fingerprint"), str)
            or not record["dataset_fingerprint"]
        ):
            raise ValidationError(f"HF dataset {key} has no dataset_fingerprint")
        cache_files = record.get("cache_files")
        if not isinstance(cache_files, list) or not cache_files:
            raise ValidationError(f"HF dataset {key} has no cache_files")
        validated_cache = [
            _validate_declared_file(
                item,
                label=f"HF dataset {key} cache_files[{index}]",
                required_root=hf_home,
            )
            for index, item in enumerate(cache_files)
        ]
        cache_paths = [item["path"] for item in validated_cache]
        if len(cache_paths) != len(set(cache_paths)):
            raise ValidationError(f"HF dataset {key} repeats a cache file")
        validated_datasets[key] = {
            "repo": expected["repo"],
            "config": expected["config"],
            "split": expected["split"],
            "revision": expected["revision"],
            "rows": rows,
            "dataset_fingerprint": record["dataset_fingerprint"],
            "cache_files": validated_cache,
        }

    longbench = _object(manifest.get("longbench_e"), label="HF manifest longbench_e")
    expected_longbench = {
        "repo": "THUDM/LongBench",
        "revision": LONGBENCH_REVISION,
        "filename": "data.zip",
    }
    for field, expected in expected_longbench.items():
        if longbench.get(field) != expected:
            raise ValidationError(
                f"HF manifest longbench_e.{field} must be {expected!r}, "
                f"got {longbench.get(field)!r}"
            )
    artifact = _validate_declared_file(
        longbench.get("artifact"),
        label="HF manifest longbench_e.artifact",
        required_root=hf_home,
    )
    confirmed_manifest_record = _local_file_record(manifest_path)
    if confirmed_manifest_record != manifest_record:
        raise ValidationError("Pinned HF dataset manifest changed during validation")
    return {
        "status": "validated",
        "manifest": manifest_record,
        "hf_home": str(hf_home),
        "packages": packages,
        "datasets": validated_datasets,
        "longbench_e": {**expected_longbench, "artifact": artifact},
    }


def validate_inputs(
    *,
    run_root: Path,
    work_root: Path,
    runtime_home: Path,
    models_root: Path,
    model_paths: Mapping[str, Path],
    ruler_root: Path | None = None,
    longcodeqa_root: Path | None = None,
    longcodeqa_context_manifests: Mapping[str, Path] | None = None,
    hf_home: Path | None = None,
    hf_manifest: Path | None = None,
    require_hf_manifest: bool = False,
    expected_longcodeqa_sha256: Mapping[str, str] = LONGCODEQA_SHA256,
    expected_hf_datasets: Mapping[str, Mapping[str, Any]] = HF_DATASETS,
) -> dict:
    run_root = _resolved(run_root)
    runtime_paths = _validate_runtime_paths(
        run_root=run_root, work_root=work_root, runtime_home=runtime_home
    )
    models_root = _resolved(models_root)
    ruler_root = _resolved(ruler_root or run_root / "data" / "ruler_v1")
    longcodeqa_root = _resolved(longcodeqa_root or run_root / "data" / "LongCodeQA")
    if longcodeqa_context_manifests is None:
        longcodeqa_context_manifests = {
            key: run_root / "data" / f"longcodeqa_context_{key}.json"
            for key in LONGCODEQA_CONTEXT_SPECS
        }
    hf_home = _resolved(hf_home or run_root / "data" / "hf_home")
    hf_manifest = _resolved(hf_manifest or run_root / "data" / "pinned_hf_datasets_manifest.json")

    models = validate_models(model_paths)
    ruler = validate_ruler(ruler_root, models)
    longcodeqa = validate_longcodeqa(longcodeqa_root, expected_sha256=expected_longcodeqa_sha256)
    longcodeqa_context = validate_longcodeqa_context_manifests(
        longcodeqa_context_manifests,
        models=models,
        longcodeqa=longcodeqa,
    )
    if hf_manifest.is_file():
        hf = validate_hf_manifest(hf_manifest, hf_home, expected_datasets=expected_hf_datasets)
    elif require_hf_manifest:
        raise ValidationError(f"Required pinned HF dataset manifest is missing: {hf_manifest}")
    else:
        hf = {"status": "not_present", "expected_path": str(hf_manifest)}

    return {
        "schema_version": 2,
        "kind": "hayatequant_paper_input_validation",
        "status": "ok",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "runtime_paths": runtime_paths,
        "models_root": str(models_root),
        "models": models,
        "ruler_v1": ruler,
        "longcodeqa": {
            "dataset": "LongCodeBench/LongCodeQA",
            "release_revision": LONGCODEQA_DATASET_REVISION,
            "release_clarification": LONGCODEQA_DISCUSSION_URL,
            "parser_reference_commit": LONGCODEQA_PROTOCOL_COMMIT,
            "protocol_scope": LONGCODEQA_PROTOCOL_SCOPE,
            "official_reference_decoding_exact": False,
            "brackets": longcodeqa,
            "total_rows": sum(LONGCODEQA_COUNTS.values()),
            "context_fit_manifests": longcodeqa_context,
        },
        "hf_datasets": hf,
    }


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Write one complete manifest without exposing a partially written file."""
    path = _resolved(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = _object(
                _load_json(path, label="existing paper input manifest"),
                label="existing paper input manifest",
            )
        except ValidationError:
            existing = None
        if existing is not None:
            existing_identity = dict(existing)
            candidate_identity = dict(document)
            existing_identity.pop("validated_at_utc", None)
            candidate_identity.pop("validated_at_utc", None)
            if existing_identity == candidate_identity:
                # The manifest SHA is part of every smoke/formal worker identity.
                # Revalidation of unchanged inputs must therefore preserve the
                # exact bytes instead of refreshing a timestamp and invalidating
                # an already completed smoke gate.
                return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_model_arguments(values: Sequence[str] | None, models_root: Path) -> dict[str, Path]:
    specifications = values or [f"{key}={value}" for key, value in DEFAULT_MODEL_PATHS.items()]
    result = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValidationError(
                f"Invalid --model {specification!r}; expected KEY=PATH_OR_DIRECTORY_NAME"
            )
        key, value = specification.split("=", 1)
        if not key or not value:
            raise ValidationError(
                f"Invalid --model {specification!r}; key and path must be non-empty"
            )
        if key in result:
            raise ValidationError(f"Duplicate --model key {key!r}")
        path = Path(value).expanduser()
        result[key] = path if path.is_absolute() else models_root / path
    return result


def _parse_context_manifest_arguments(
    values: Sequence[str] | None,
) -> dict[str, Path] | None:
    if values is None:
        return None
    result = {}
    for specification in values:
        if "=" not in specification:
            raise ValidationError(
                f"Invalid --longcodeqa-context-manifest {specification!r}; "
                "expected MODEL_KEY=PATH"
            )
        key, value = specification.split("=", 1)
        if not key or not value:
            raise ValidationError("--longcodeqa-context-manifest key and path must be non-empty")
        if key in result:
            raise ValidationError(f"Duplicate --longcodeqa-context-manifest key {key!r}")
        result[key] = Path(value).expanduser()
    if set(result) != set(LONGCODEQA_CONTEXT_SPECS):
        raise ValidationError(
            "--longcodeqa-context-manifest overrides must specify exactly "
            f"{sorted(LONGCODEQA_CONTEXT_SPECS)}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="existing external root for mutable caches and runtime state",
    )
    parser.add_argument(
        "--runtime-home",
        type=Path,
        required=True,
        help="must be exactly WORK_ROOT/runtime/home and contain no symlink components",
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument(
        "--model",
        action="append",
        metavar="KEY=PATH",
        help="override the default seven formal models plus Qwen3-0.6B smoke model",
    )
    parser.add_argument("--ruler-root", type=Path)
    parser.add_argument("--longcodeqa-root", type=Path)
    parser.add_argument(
        "--longcodeqa-context-manifest",
        action="append",
        metavar="MODEL_KEY=PATH",
        help=(
            "override both default RUN_ROOT/data/longcodeqa_context_MODEL_KEY.json "
            "manifests (repeat for llama31-8b and qwen3-8b)"
        ),
    )
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument(
        "--hf-manifest",
        type=Path,
        help="default: RUN_ROOT/data/pinned_hf_datasets_manifest.json",
    )
    parser.add_argument(
        "--require-hf-manifest",
        action="store_true",
        help="fail unless the pinned HF manifest and all cached artifacts validate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="default: RUN_ROOT/manifests/hayatequant_paper_inputs.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.models_root is None:
        parser.error("--models-root or KVCOMP_MODELS_ROOT is required; model weights are external")
    run_root = _resolved(args.run_root)
    models_root = _resolved(args.models_root)
    output = args.output or run_root / "manifests" / "hayatequant_paper_inputs.json"
    try:
        model_paths = _parse_model_arguments(args.model, models_root)
        context_manifests = _parse_context_manifest_arguments(args.longcodeqa_context_manifest)
        document = validate_inputs(
            run_root=run_root,
            work_root=args.work_root,
            runtime_home=args.runtime_home,
            models_root=models_root,
            model_paths=model_paths,
            ruler_root=args.ruler_root,
            longcodeqa_root=args.longcodeqa_root,
            longcodeqa_context_manifests=context_manifests,
            hf_home=args.hf_home,
            hf_manifest=args.hf_manifest,
            require_hf_manifest=args.require_hf_manifest,
        )
        atomic_write_json(output, document)
    except (ValidationError, OSError) as exc:
        print(f"input validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"validated HayateQuant paper inputs: {_resolved(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
