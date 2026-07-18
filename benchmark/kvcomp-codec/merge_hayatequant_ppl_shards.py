#!/usr/bin/env python3
"""Fail-closed merger for independently evaluated formal PPL rate shards.

The expensive WikiText-2 pass is submitted once per nominal rate so no single
Slurm allocation contains the entire seven-method, three-rate matrix.  This
CPU-only step validates that the three outputs are otherwise the same formal
experiment and publishes the one canonical document consumed by the strict
paper aggregator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from onecomp.kvcomp.eval.provenance import load_run_provenance


SCHEMA_VERSION = 2
FORMAL_BITS = (2, 3, 4)
FORMAL_METHODS = (
    "TurboQuant-paper",
    "FibQuant-paper",
    "BlockQuant-paper",
    "HQ",
    "HQ-Frontier",
    "AdaptiveHQ",
    "RotationHQ",
)
EXPECTED_METHODS = {
    "2": tuple(method for method in FORMAL_METHODS if method != "BlockQuant-paper"),
    "3": FORMAL_METHODS,
    "4": tuple(method for method in FORMAL_METHODS if method != "BlockQuant-paper"),
}


class MergeError(RuntimeError):
    """A shard set is unsafe, inconsistent, or incomplete."""


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MergeError(f"PPL shard is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"PPL shard is not readable strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"PPL shard root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MergeError(f"{label} is not a lowercase SHA-256")
    return value


def _provenance_identity(document: dict[str, Any], *, path: Path) -> tuple[str, str, str]:
    records = document.get("run_provenance")
    if not isinstance(records, list) or len(records) != 1:
        raise MergeError(f"Each PPL rate shard must contain exactly one preflight: {path}")
    envelope = records[0]
    if not isinstance(envelope, dict):
        raise MergeError(f"Malformed PPL shard preflight envelope: {path}")
    digest = _require_sha256(envelope.get("sha256"), label=f"{path} preflight digest")
    record = envelope.get("record")
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise MergeError(f"Malformed PPL shard preflight record: {path}")
    code = record.get("code")
    if (
        not isinstance(code, dict)
        or code.get("tracked_dirty") is not False
        or code.get("commit") != document.get("code_commit")
    ):
        raise MergeError(f"PPL shard was not produced from its declared clean commit: {path}")
    source = envelope.get("path")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise MergeError(f"PPL shard preflight path is not absolute: {path}")
    source_path = Path(source)
    if source_path.is_file():
        if _sha256(source_path) != digest:
            raise MergeError(f"PPL shard preflight file changed: {source_path}")
        if _load_object(source_path) != record:
            raise MergeError(f"PPL shard embedded preflight differs from disk: {path}")
    model = record.get("model")
    inputs = model.get("input_manifest") if isinstance(model, dict) else None
    if not isinstance(model, dict) or not isinstance(inputs, dict):
        raise MergeError(f"PPL shard preflight omitted model/input identity: {path}")
    model_path = model.get("path")
    input_path = inputs.get("path")
    input_digest = inputs.get("sha256")
    if not all(isinstance(value, str) and value for value in (model_path, input_path)):
        raise MergeError(f"PPL shard preflight has malformed model/input paths: {path}")
    _require_sha256(input_digest, label=f"{path} input manifest digest")
    return str(code["commit"]), str(model_path), f"{input_path}:{input_digest}"


def merge(shards: list[Path], output: Path) -> dict[str, Any]:
    if len(shards) != len(FORMAL_BITS):
        raise MergeError(f"Expected exactly {len(FORMAL_BITS)} PPL shards, found {len(shards)}")
    resolved = [path.expanduser().resolve() for path in shards]
    if len(set(resolved)) != len(resolved):
        raise MergeError("PPL shard paths must be unique")
    output = output.expanduser().resolve()
    if output.is_symlink() or output.exists():
        raise MergeError(f"Refusing to overwrite canonical PPL result: {output}")

    documents = [(_load_object(path), path) for path in resolved]
    reference, reference_path = documents[0]
    invariant_fields = (
        "schema_version",
        "run_fingerprint",
        "model",
        "head_dim",
        "semantic_config",
        "code_commit",
        "skip_layers",
        "fullkv_ppl",
    )
    for field in invariant_fields:
        if any(
            _canonical(document.get(field)) != _canonical(reference.get(field))
            for document, _ in documents[1:]
        ):
            raise MergeError(f"PPL shard invariant {field!r} differs from {reference_path}")
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise MergeError("PPL shards use an unsupported result schema")
    fingerprint = _require_sha256(reference.get("run_fingerprint"), label="run_fingerprint")
    commit = reference.get("code_commit")
    semantic = reference.get("semantic_config")
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(semantic, dict):
        raise MergeError("PPL shard has malformed commit or semantic config")
    expected_fingerprint = hashlib.sha256(
        _canonical({"semantic_config": semantic, "code_commit": commit}).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected_fingerprint:
        raise MergeError("PPL shard run_fingerprint does not bind its config and commit")

    identities = {_provenance_identity(document, path=path) for document, path in documents}
    if len(identities) != 1:
        raise MergeError("PPL shards span multiple commits, models, or input manifests")

    merged_results: dict[str, Any] = {}
    merged_provenance: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen_provenance: set[str] = set()
    for document, path in documents:
        config = document.get("config")
        results = document.get("results")
        if not isinstance(config, dict) or not isinstance(results, dict) or len(results) != 1:
            raise MergeError(f"PPL rate shard must contain exactly one result rate: {path}")
        rate_key = next(iter(results))
        try:
            rate = int(rate_key)
        except (TypeError, ValueError) as exc:
            raise MergeError(f"PPL shard has a non-integer rate key: {path}") from exc
        if str(rate) != str(rate_key) or rate not in FORMAL_BITS:
            raise MergeError(f"PPL shard has an unexpected rate {rate_key!r}: {path}")
        if config.get("bits") != [rate] or config.get("methods") != list(FORMAL_METHODS):
            raise MergeError(
                f"PPL shard config does not declare the formal rate/method grid: {path}"
            )
        methods = results[rate_key]
        if not isinstance(methods, dict) or tuple(methods) != EXPECTED_METHODS[rate_key]:
            raise MergeError(f"PPL shard method coverage/order changed at rate {rate}: {path}")
        if rate_key in merged_results:
            raise MergeError(f"Duplicate PPL rate shard {rate}: {path}")
        merged_results[rate_key] = methods
        envelope = document["run_provenance"][0]
        provenance_digest = envelope["sha256"]
        if provenance_digest in seen_provenance:
            raise MergeError("PPL rate shards reuse one worker preflight")
        seen_provenance.add(provenance_digest)
        merged_provenance.append(envelope)
        ledger.append(
            {
                "rate": rate,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "run_provenance_sha256": provenance_digest,
            }
        )
    if tuple(sorted(int(key) for key in merged_results)) != FORMAL_BITS:
        raise MergeError("PPL rate shards do not cover exactly rates 2, 3, and 4")

    config = dict(reference["config"])
    config["output_dir"] = str(output.parent)
    config["bits"] = list(FORMAL_BITS)
    config["methods"] = list(FORMAL_METHODS)
    merged = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "model": reference["model"],
        "head_dim": reference["head_dim"],
        "config": config,
        "semantic_config": semantic,
        "code_commit": commit,
        "skip_layers": reference["skip_layers"],
        "run_provenance": merged_provenance,
        "fullkv_ppl": reference["fullkv_ppl"],
        "results": {str(rate): merged_results[str(rate)] for rate in FORMAL_BITS},
        "ppl_shard_merge": {
            "schema": "hayatequant.formal_ppl_rate_merge.v1",
            "merge_preflight": load_run_provenance(),
            "shards": sorted(ledger, key=lambda item: item["rate"]),
        },
    }
    return merged


def _write_atomic(document: dict[str, Any], output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or output.exists():
        raise MergeError(f"Refusing to overwrite canonical PPL result: {output}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists() or output.is_symlink():
            raise MergeError(f"Canonical PPL result appeared during merge: {output}")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = merge(args.shard, args.output)
    _write_atomic(document, args.output)
    print(f"Wrote canonical formal PPL result: {args.output}", flush=True)


if __name__ == "__main__":
    main()
