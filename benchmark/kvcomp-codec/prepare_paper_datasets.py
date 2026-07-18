#!/usr/bin/env python3
"""Prefetch and fingerprint every Hugging Face paper-evaluation dataset.

GPU workers run offline.  This helper resolves immutable revisions once on a
networked login node, materializes their Arrow caches below a run-specific
``HF_HOME``, and writes an atomic manifest containing row counts, dataset
fingerprints, upstream checksums, and hashes of the exact cached files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DATASETS = (
    {
        "key": "wikitext2_test",
        "repo": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "expected_rows": 4358,
    },
    {
        "key": "arc_easy_test",
        "repo": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "expected_rows": 2376,
    },
    {
        "key": "arc_challenge_test",
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "expected_rows": 1172,
    },
    {
        "key": "piqa_validation",
        "repo": "baber/piqa",
        "config": None,
        "split": "validation",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "expected_rows": 1838,
    },
    {
        "key": "hellaswag_validation",
        "repo": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "expected_rows": 10042,
    },
    {
        "key": "winogrande_xl_validation",
        "repo": "allenai/winogrande",
        "config": "winogrande_xl",
        "split": "validation",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "expected_rows": 1267,
    },
    {
        "key": "math500_test",
        "repo": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "expected_rows": 500,
    },
    {
        "key": "gsm8k_main_test",
        "repo": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "expected_rows": 1319,
    },
    {
        "key": "aime2024_source",
        "repo": "AI-MO/aimo-validation-aime",
        "config": None,
        "split": "train",
        "revision": "13f9e12f613e720c2a2b2f345dd04b998a29494d",
        "expected_rows": None,
    },
    {
        "key": "aime2025_train",
        "repo": "yentinglin/aime_2025",
        "config": None,
        "split": "train",
        "revision": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        "expected_rows": 30,
    },
)

LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_tuple(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_record(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"Dataset cache path is not a regular file: {resolved}")
    digest = _sha256(resolved)
    after = resolved.stat()
    if _state_tuple(before) != _state_tuple(after):
        raise RuntimeError(f"Dataset cache file changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": after.st_size,
        "sha256": digest,
    }


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
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    hf_home = args.hf_home.expanduser().resolve()
    datasets_cache = hf_home / "datasets"
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    # All paper datasets are public.  Explicitly suppress bearer-token use so
    # neither requests nor provenance can accidentally depend on login state.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    records = {}
    for spec in DATASETS:
        print(
            f"prefetch {spec['repo']}@{spec['revision']} "
            f"{spec['config'] or ''}/{spec['split']}",
            flush=True,
        )
        kwargs = {
            "split": spec["split"],
            "revision": spec["revision"],
            "cache_dir": str(datasets_cache),
            "token": False,
        }
        dataset = (
            load_dataset(spec["repo"], spec["config"], **kwargs)
            if spec["config"] is not None
            else load_dataset(spec["repo"], **kwargs)
        )
        rows = len(dataset)
        expected = spec["expected_rows"]
        if expected is not None and rows != expected:
            raise RuntimeError(
                f"{spec['key']} expected {expected} rows at the pinned revision, got {rows}"
            )
        cache_files = [_file_record(item["filename"]) for item in dataset.cache_files]
        if len({item["path"] for item in cache_files}) != len(cache_files):
            raise RuntimeError(f"{spec['key']} returned duplicate cache files")
        # Raw download URLs can contain signed query parameters.  The immutable
        # Arrow payloads above are the executable inputs, so provenance retains
        # only upstream content checksums and sizes, never source URLs.
        download_records = sorted(
            (
                {
                    "checksum": value.get("checksum"),
                    "num_bytes": value.get("num_bytes"),
                }
                for value in (dataset.info.download_checksums or {}).values()
                if isinstance(value, dict)
            ),
            key=lambda item: (str(item["checksum"]), str(item["num_bytes"])),
        )
        records[spec["key"]] = {
            **spec,
            "rows": rows,
            "dataset_fingerprint": dataset._fingerprint,
            "features": str(dataset.features),
            "cache_files": cache_files,
            "download_records_without_urls": download_records,
            "download_size": dataset.info.download_size,
            "dataset_size": dataset.info.dataset_size,
        }

    longbench_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
        revision=LONGBENCH_REVISION,
        cache_dir=str(hf_home / "hub"),
        token=False,
    )
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_home": str(hf_home),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "huggingface-hub", "pyarrow")
        },
        "datasets": records,
        "longbench_e": {
            "repo": "THUDM/LongBench",
            "revision": LONGBENCH_REVISION,
            "filename": "data.zip",
            "artifact": _file_record(longbench_path),
        },
    }
    manifest = args.manifest.expanduser().resolve()
    _atomic_json(manifest, document)
    print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
