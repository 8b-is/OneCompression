#!/usr/bin/env python3
"""Materialize a pinned, evaluation-disjoint Rotation-HQ calibration text.

Rotation HQ is a calibrated control, so its text must be recorded separately
from WikiText-2 test PPL.  This helper freezes a deterministic prefix of the
WikiText-2 raw *train* split and writes both the text and a content-addressed
manifest.  It never reads the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from datasets import load_dataset


DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
# Pinned dataset-source revision used by the formal WikiText-2 test protocol.
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--max-characters", type=int, default=120_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_characters <= 0:
        raise ValueError("--max-characters must be positive")
    args.hf_home.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="train",
        revision=DATASET_REVISION,
        cache_dir=str(args.hf_home),
    )
    rows = dataset["text"]
    source_text = "\n\n".join(rows)
    selected_rows: list[int] = []
    selected_parts: list[str] = []
    selected_characters = 0
    for row_index, raw in enumerate(rows):
        text = str(raw).strip()
        if not text:
            continue
        separator = 2 if selected_parts else 0
        if selected_characters + separator + len(text) > args.max_characters:
            remaining = args.max_characters - selected_characters - separator
            if remaining > 0:
                selected_parts.append(text[:remaining])
                selected_rows.append(row_index)
            break
        selected_parts.append(text)
        selected_rows.append(row_index)
        selected_characters += separator + len(text)
    calibration_text = "\n\n".join(selected_parts)
    if not calibration_text:
        raise RuntimeError("no non-empty WikiText-2 train row was selected")
    output_dir = args.output_dir.resolve()
    text_path = output_dir / "calibration_text.txt"
    manifest_path = output_dir / "calibration_manifest.json"
    text_bytes = calibration_text.encode("utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "rotation_hq_wikitext2_train_calibration",
        "source": {
            "repo": DATASET_REPO,
            "config": DATASET_CONFIG,
            "split": "train",
            "revision": DATASET_REVISION,
            "rows": len(rows),
            "joined_text_sha256": _sha256_bytes(source_text.encode("utf-8")),
        },
        "selection": {
            "rule": "first_nonempty_train_rows_until_max_characters",
            "max_characters": args.max_characters,
            "row_indices": selected_rows,
            "text_characters": len(calibration_text),
            "text_sha256": _sha256_bytes(text_bytes),
        },
        "evaluation_exclusion": {
            "forbidden_split": "test",
            "statement": "No WikiText-2 test text is used for Rotation HQ calibration.",
        },
        "files": {"calibration_text": str(text_path)},
    }
    _write_atomic(text_path, text_bytes)
    _write_atomic(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({"text": str(text_path), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
