"""Fail-closed inventory, integrity, and secret scan for this dev bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("llama31", "llama2", "qwen34", "qwen38", "r1llama8", "r1qwen15", "r1qwen7")
OUTPUTS = ("MANIFEST.json", "SHA256SUMS")
SECRET_PATTERNS = {
    "credentialed_https_url": re.compile(rb"https?://[^\s/@:]+:[^\s/@]+@"),
    "common_pat_prefix": re.compile(rb"(?:glpat-|gh[pousr]_|hf_)[A-Za-z0-9_-]{12,}"),
    "three_part_long_token": re.compile(
        rb"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{20,}\.[0-9]{2}\.[A-Za-z0-9_-]{8,}"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    generated_bytecode = sorted(ROOT.rglob("*.pyc"))
    if generated_bytecode:
        raise RuntimeError(
            "remove generated Python bytecode before finalization: "
            f"{[str(path.relative_to(ROOT)) for path in generated_bytecode]}"
        )
    required = [
        ROOT / "README.md",
        ROOT / "COMMAND.txt",
        ROOT / "SOURCE_STATE.txt",
        ROOT / "SLURM_ACCOUNTING.txt",
        ROOT / "analysis.json",
        ROOT / "ppl_cells.csv",
    ]
    required += [ROOT / "manifests" / key / "runtime.txt" for key in KEYS]
    required += [
        ROOT / "logs" / f"hq_dev_baseline_69320_{index}.{suffix}"
        for index in range(7)
        for suffix in ("out", "err")
    ]
    raw = sorted((ROOT / "raw").glob("*/results_*.json"))
    if len(raw) != 7:
        raise RuntimeError(f"expected 7 raw JSON documents, found {len(raw)}")
    required += raw
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing bundle files: {missing}")
    for key in KEYS:
        manifest = (ROOT / "manifests" / key / "runtime.txt").read_text(encoding="utf-8")
        if "exit_status=0\n" not in manifest:
            raise RuntimeError(f"nonzero or absent exit status for {key}")

    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.name not in OUTPUTS and not path.name.startswith(".")
    )
    findings = []
    records = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                findings.append({"file": relative, "pattern": label})
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if findings:
        # Report file/category only and never print the matched credential.
        raise RuntimeError(f"potential secret material detected: {findings}")

    manifest = {
        "schema_version": 1,
        "scope": "development-only one-window WikiText-2 baseline-quality matrix",
        "slurm_array_job_id": 69320,
        "raw_result_count": len(raw),
        "all_tasks_exit_zero": True,
        "secret_scan": {
            "passed": True,
            "patterns": sorted(SECRET_PATTERNS),
            "files_scanned": len(records),
        },
        "files": records,
    }
    _atomic_text(ROOT / "MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    sums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
    _atomic_text(ROOT / "SHA256SUMS", sums)

    # Verify what was written rather than trusting the in-memory inventory.
    for record in records:
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"post-write SHA mismatch for {record['path']}")
    print(
        f"finalized {len(records)} files; 7 raw JSON; "
        f"secret scan passed; manifest outputs excluded from their own hash inventory"
    )


if __name__ == "__main__":
    main()
