#!/usr/bin/env python3
"""Fingerprint and verify local Hugging Face model weight layouts.

The paper runners load models from local directories.  A metadata-only model
record is insufficient: two directories can have identical ``config.json``
files while containing different parameters.  This module provides the small,
dependency-free contract shared by the offline input validator and every GPU
job preflight.

Supported layouts are the two formats used by ``transformers`` for these
checkpoints:

* ``model.safetensors`` or ``model.safetensors.index.json``;
* ``pytorch_model.bin`` or ``pytorch_model.bin.index.json``.

When both formats are present, both complete layouts are fingerprinted and
safetensors is selected, matching the normal Transformers preference.  Within
one format the layout must be unambiguous and exhaustive: an index may not
coexist with a single-file payload, every referenced shard must exist, and no
same-format weight payload may be left unreferenced.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
_FORMAT_SPECS = {
    "safetensors": {
        "index": "model.safetensors.index.json",
        "single": "model.safetensors",
        "payload_re": re.compile(r".+\.safetensors"),
    },
    "pytorch_bin": {
        "index": "pytorch_model.bin.index.json",
        "single": "pytorch_model.bin",
        "payload_re": re.compile(r"pytorch_model(?:[-.].*)?\.bin"),
    },
}
_FORMAT_PREFERENCE = ("safetensors", "pytorch_bin")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ModelWeightError(RuntimeError):
    """A model weight layout or its recorded fingerprint is not trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelWeightError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ModelWeightError(f"non-finite JSON number {value!r}")


def _load_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise ModelWeightError(f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except ModelWeightError as exc:
        raise ModelWeightError(f"{label} contains a {exc}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelWeightError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def _relative_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelWeightError(f"{label} must be a non-empty relative path")
    # HF index paths use POSIX separators, including on Linux hosts.  Rejecting
    # backslashes avoids two textual names resolving to one Linux filename.
    if "\\" in value:
        raise ModelWeightError(f"{label} contains a backslash: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ModelWeightError(f"{label} is not a normalized relative path: {value!r}")
    if any(part.startswith(".") for part in path.parts):
        raise ModelWeightError(f"{label} may not enter a hidden directory: {value!r}")
    return path.as_posix()


def _logical_path(root: Path, relative: str) -> Path:
    # ``relative`` has already been normalized.  Do not call resolve here: a
    # file-level symlink is a legitimate HF layout and is recorded explicitly.
    return root.joinpath(*PurePosixPath(relative).parts)


def _check_parent_directories(root: Path, logical: Path) -> None:
    parent = logical.parent
    while parent != root:
        if parent.is_symlink():
            raise ModelWeightError(
                f"weight path has a symlinked parent directory: {logical} via {parent}"
            )
        if not parent.is_dir():
            raise ModelWeightError(f"weight parent directory is missing: {parent}")
        parent = parent.parent


def _stable_file_state(path: Path) -> tuple[dict[str, Any], os.stat_result, os.stat_result]:
    """Return a race-detecting lstat/stat snapshot for one logical file."""
    try:
        before_lstat = path.lstat()
        before_stat = path.stat()
    except FileNotFoundError as exc:
        raise ModelWeightError(f"weight file is missing or has a broken symlink: {path}") from exc
    if not (stat.S_ISREG(before_stat.st_mode)):
        raise ModelWeightError(f"weight path does not resolve to a regular file: {path}")
    is_symlink = stat.S_ISLNK(before_lstat.st_mode)
    record: dict[str, Any] = {"symlink": is_symlink}
    if is_symlink:
        try:
            record["symlink_target"] = os.readlink(path)
            record["resolved_path"] = str(path.resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise ModelWeightError(f"cannot resolve weight symlink: {path}") from exc
    return record, before_lstat, before_stat


def _state_tuple(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_record(root: Path, relative: str) -> tuple[dict[str, Any], tuple[int, int]]:
    relative = _relative_path(relative, label="weight filename")
    logical = _logical_path(root, relative)
    _check_parent_directories(root, logical)
    link_record, before_lstat, before_stat = _stable_file_state(logical)
    digest = _sha256(logical)
    try:
        after_lstat = logical.lstat()
        after_stat = logical.stat()
    except FileNotFoundError as exc:
        raise ModelWeightError(f"weight file disappeared while hashing: {logical}") from exc
    if _state_tuple(before_lstat) != _state_tuple(after_lstat) or _state_tuple(
        before_stat
    ) != _state_tuple(after_stat):
        raise ModelWeightError(f"weight file changed while hashing: {logical}")
    if link_record.get("symlink"):
        if os.readlink(logical) != link_record["symlink_target"]:
            raise ModelWeightError(f"weight symlink changed while hashing: {logical}")
        if str(logical.resolve(strict=True)) != link_record["resolved_path"]:
            raise ModelWeightError(f"weight symlink target changed while hashing: {logical}")
    record = {
        "relative_path": relative,
        "bytes": after_stat.st_size,
        "sha256": digest,
        **link_record,
    }
    return record, (after_stat.st_dev, after_stat.st_ino)


def _candidate_payloads(root: Path, format_name: str) -> set[str]:
    pattern = _FORMAT_SPECS[format_name]["payload_re"]
    result: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        # HF's .cache contains download metadata and may contain temporary
        # artifacts.  It is not a loadable model layout.  Hidden directories
        # are also forbidden in index references above.
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        directory_path = Path(directory)
        for name in filenames:
            if pattern.fullmatch(name):
                relative = (directory_path / name).relative_to(root).as_posix()
                result.add(_relative_path(relative, label="discovered weight filename"))
    return result


def _fingerprint_layout(root: Path, format_name: str) -> dict[str, Any] | None:
    spec = _FORMAT_SPECS[format_name]
    index_relative = spec["index"]
    single_relative = spec["single"]
    index_exists = _logical_path(root, index_relative).is_file()
    single_exists = _logical_path(root, single_relative).is_file()
    candidates = _candidate_payloads(root, format_name)

    if index_exists and single_exists:
        raise ModelWeightError(
            f"ambiguous {format_name} layout in {root}: both {index_relative} and "
            f"{single_relative} exist"
        )
    if not index_exists and not single_exists:
        if candidates:
            raise ModelWeightError(
                f"{format_name} payloads exist without {index_relative} or "
                f"{single_relative}: {sorted(candidates)}"
            )
        return None

    index_record: dict[str, Any] | None = None
    tensor_count: int | None = None
    if index_exists:
        index_record, _ = _file_record(root, index_relative)
        document = _load_json(
            _logical_path(root, index_relative), label=f"{format_name} weight index"
        )
        confirmed_index_record, _ = _file_record(root, index_relative)
        if confirmed_index_record != index_record:
            raise ModelWeightError(f"{format_name} weight index changed while it was being parsed")
        if not isinstance(document, dict):
            raise ModelWeightError(f"{format_name} weight index root must be an object")
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelWeightError(
                f"{format_name} weight index weight_map must be a non-empty object"
            )
        referenced: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ModelWeightError(
                    f"{format_name} weight index has an invalid tensor name {tensor_name!r}"
                )
            relative = _relative_path(
                shard_name, label=f"{format_name} weight_map[{tensor_name!r}]"
            )
            if not spec["payload_re"].fullmatch(PurePosixPath(relative).name):
                raise ModelWeightError(
                    f"{format_name} index references a wrong-format shard: {relative}"
                )
            referenced.add(relative)
        tensor_count = len(weight_map)
        unreferenced = candidates - referenced
        missing = referenced - candidates
        if missing:
            raise ModelWeightError(
                f"{format_name} index references missing shard(s): {sorted(missing)}"
            )
        if unreferenced:
            raise ModelWeightError(
                f"{format_name} has unreferenced weight shard(s): {sorted(unreferenced)}"
            )
        kind = "sharded_index"
        shard_paths = sorted(referenced)
    else:
        if candidates != {single_relative}:
            extras = sorted(candidates - {single_relative})
            raise ModelWeightError(
                f"{format_name} single-file layout has unreferenced weight shard(s): {extras}"
            )
        kind = "single_file"
        shard_paths = [single_relative]

    shards: list[dict[str, Any]] = []
    physical_files: dict[tuple[int, int], str] = {}
    for relative in shard_paths:
        record, identity = _file_record(root, relative)
        previous = physical_files.get(identity)
        if previous is not None:
            raise ModelWeightError(
                f"{format_name} shards {previous!r} and {relative!r} resolve to the "
                "same physical file"
            )
        physical_files[identity] = relative
        shards.append(record)
    return {
        "format": format_name,
        "kind": kind,
        "index": index_record,
        "shards": shards,
        "shard_count": len(shards),
        "tensor_count": tensor_count,
        "total_shard_bytes": sum(record["bytes"] for record in shards),
    }


def fingerprint_model_weights(model_root: str | Path) -> dict[str, Any]:
    """Read and SHA-256 fingerprint every complete local weight layout."""
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise ModelWeightError(f"model directory is missing: {root}")
    initial_inventory = {
        format_name: {
            "payloads": sorted(_candidate_payloads(root, format_name)),
            "index": _logical_path(root, spec["index"]).is_file(),
            "single": _logical_path(root, spec["single"]).is_file(),
        }
        for format_name, spec in _FORMAT_SPECS.items()
    }
    layouts: dict[str, dict[str, Any]] = {}
    for format_name in _FORMAT_PREFERENCE:
        layout = _fingerprint_layout(root, format_name)
        if layout is not None:
            layouts[format_name] = layout
    if not layouts:
        raise ModelWeightError(
            f"model has no supported safetensors or pytorch_model.bin weights: {root}"
        )
    final_inventory = {
        format_name: {
            "payloads": sorted(_candidate_payloads(root, format_name)),
            "index": _logical_path(root, spec["index"]).is_file(),
            "single": _logical_path(root, spec["single"]).is_file(),
        }
        for format_name, spec in _FORMAT_SPECS.items()
    }
    if final_inventory != initial_inventory:
        raise ModelWeightError(f"model weight inventory changed while fingerprinting: {root}")
    selected = next(name for name in _FORMAT_PREFERENCE if name in layouts)
    total_bytes = 0
    total_files = 0
    for layout in layouts.values():
        total_bytes += layout["total_shard_bytes"]
        total_files += layout["shard_count"]
        if layout["index"] is not None:
            total_bytes += layout["index"]["bytes"]
            total_files += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_format": selected,
        "selected_kind": layouts[selected]["kind"],
        "format_preference": list(_FORMAT_PREFERENCE),
        "layouts": layouts,
        "fingerprinted_file_count": total_files,
        "fingerprinted_bytes": total_bytes,
    }


def _validate_file_record(record: Any, *, label: str) -> None:
    if not isinstance(record, dict):
        raise ModelWeightError(f"{label} must be an object")
    required = {"relative_path", "bytes", "sha256", "symlink"}
    if not required.issubset(record):
        raise ModelWeightError(f"{label} is missing {sorted(required - set(record))}")
    _relative_path(record["relative_path"], label=f"{label}.relative_path")
    if (
        isinstance(record["bytes"], bool)
        or not isinstance(record["bytes"], int)
        or record["bytes"] < 0
    ):
        raise ModelWeightError(f"{label}.bytes must be a non-negative integer")
    digest = record["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ModelWeightError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if not isinstance(record["symlink"], bool):
        raise ModelWeightError(f"{label}.symlink must be boolean")
    if record["symlink"]:
        if not isinstance(record.get("symlink_target"), str) or not isinstance(
            record.get("resolved_path"), str
        ):
            raise ModelWeightError(f"{label} has incomplete symlink metadata")


def validate_weight_manifest_schema(document: Any) -> dict[str, Any]:
    """Validate the closed weight-manifest schema before trusting its paths."""
    if not isinstance(document, dict):
        raise ModelWeightError("model weight manifest must be an object")
    expected_top = {
        "schema_version",
        "selected_format",
        "selected_kind",
        "format_preference",
        "layouts",
        "fingerprinted_file_count",
        "fingerprinted_bytes",
    }
    if set(document) != expected_top:
        raise ModelWeightError(
            "model weight manifest top-level schema mismatch; "
            f"expected {sorted(expected_top)}, got {sorted(document)}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ModelWeightError(f"model weight manifest schema_version must be {SCHEMA_VERSION}")
    if document.get("format_preference") != list(_FORMAT_PREFERENCE):
        raise ModelWeightError("model weight format_preference is not canonical")
    layouts = document.get("layouts")
    if not isinstance(layouts, dict) or not layouts:
        raise ModelWeightError("model weight layouts must be a non-empty object")
    if not set(layouts).issubset(_FORMAT_SPECS):
        raise ModelWeightError(f"unsupported model weight layouts: {sorted(layouts)}")
    selected = document.get("selected_format")
    if selected != next(name for name in _FORMAT_PREFERENCE if name in layouts):
        raise ModelWeightError("model weight selected_format violates format preference")
    file_count = 0
    byte_count = 0
    for format_name, layout in layouts.items():
        label = f"model weight layouts.{format_name}"
        if not isinstance(layout, dict):
            raise ModelWeightError(f"{label} must be an object")
        expected_layout = {
            "format",
            "kind",
            "index",
            "shards",
            "shard_count",
            "tensor_count",
            "total_shard_bytes",
        }
        if set(layout) != expected_layout:
            raise ModelWeightError(f"{label} schema mismatch")
        if layout["format"] != format_name:
            raise ModelWeightError(f"{label}.format mismatch")
        if layout["kind"] not in ("single_file", "sharded_index"):
            raise ModelWeightError(f"{label}.kind is invalid")
        shards = layout["shards"]
        if not isinstance(shards, list) or not shards:
            raise ModelWeightError(f"{label}.shards must be a non-empty array")
        for index, record in enumerate(shards):
            _validate_file_record(record, label=f"{label}.shards[{index}]")
        if len({record["relative_path"] for record in shards}) != len(shards):
            raise ModelWeightError(f"{label} repeats a shard relative_path")
        if (
            isinstance(layout["shard_count"], bool)
            or not isinstance(layout["shard_count"], int)
            or layout["shard_count"] != len(shards)
        ):
            raise ModelWeightError(f"{label}.shard_count mismatch")
        shard_bytes = sum(record["bytes"] for record in shards)
        if layout["total_shard_bytes"] != shard_bytes:
            raise ModelWeightError(f"{label}.total_shard_bytes mismatch")
        index_record = layout["index"]
        if layout["kind"] == "sharded_index":
            _validate_file_record(index_record, label=f"{label}.index")
            if (
                isinstance(layout["tensor_count"], bool)
                or not isinstance(layout["tensor_count"], int)
                or layout["tensor_count"] < 1
            ):
                raise ModelWeightError(f"{label}.tensor_count must be positive")
            file_count += 1
            byte_count += index_record["bytes"]
        elif index_record is not None or layout["tensor_count"] is not None:
            raise ModelWeightError(f"{label} single-file index/tensor_count must be null")
        file_count += len(shards)
        byte_count += shard_bytes
    if document.get("selected_kind") != layouts[selected]["kind"]:
        raise ModelWeightError("model weight selected_kind mismatch")
    if (
        isinstance(document.get("fingerprinted_file_count"), bool)
        or document.get("fingerprinted_file_count") != file_count
    ):
        raise ModelWeightError("model weight fingerprinted_file_count mismatch")
    if (
        isinstance(document.get("fingerprinted_bytes"), bool)
        or document.get("fingerprinted_bytes") != byte_count
    ):
        raise ModelWeightError("model weight fingerprinted_bytes mismatch")
    return document


def load_input_manifest_model(
    input_manifest: str | Path,
    model_root: str | Path,
) -> tuple[str, dict[str, Any]]:
    """Select exactly one path-matching model from a paper input manifest."""
    manifest_path = Path(input_manifest).expanduser().resolve()
    document = _load_json(manifest_path, label="HayateQuant paper input manifest")
    if not isinstance(document, dict):
        raise ModelWeightError("HayateQuant paper input manifest root must be an object")
    if document.get("schema_version") != 2:
        raise ModelWeightError(
            "HayateQuant paper input manifest schema_version must be 2 (weight-pinned)"
        )
    if document.get("kind") != "hayatequant_paper_input_validation":
        raise ModelWeightError("file is not a HayateQuant paper input validation manifest")
    if document.get("status") != "ok":
        raise ModelWeightError("HayateQuant paper input manifest status is not ok")
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelWeightError("HayateQuant paper input manifest models must be non-empty")
    root = Path(model_root).expanduser().resolve()
    matches: list[tuple[str, dict[str, Any]]] = []
    for key, value in models.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise ModelWeightError("HayateQuant paper input manifest has an invalid model entry")
        declared_path = value.get("path")
        if not isinstance(declared_path, str) or not declared_path:
            raise ModelWeightError(f"input manifest model {key!r} has no valid path")
        if Path(declared_path).expanduser().resolve() == root:
            matches.append((key, value))
    if len(matches) != 1:
        raise ModelWeightError(
            f"input manifest must contain exactly one model matching {root}, found {len(matches)}"
        )
    key, record = matches[0]
    metadata = record.get("metadata_files")
    if not isinstance(metadata, dict) or not metadata:
        raise ModelWeightError(f"input manifest model {key!r} has no metadata_files")
    if "weights" not in record:
        raise ModelWeightError(f"input manifest model {key!r} has no weight fingerprint")
    validate_weight_manifest_schema(record["weights"])
    return key, record


def verify_model_weights(model_root: str | Path, expected: Any) -> dict[str, Any]:
    """Fully re-hash a model and require exact agreement with ``expected``."""
    expected_document = validate_weight_manifest_schema(expected)
    actual = fingerprint_model_weights(model_root)
    if actual != expected_document:
        # Avoid printing the potentially huge manifests.  The caller can rerun
        # the input validator for a path-specific diagnostic.
        raise ModelWeightError(
            f"model weights differ from the pinned manifest: "
            f"{Path(model_root).expanduser().resolve()}"
        )
    return actual
