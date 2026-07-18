#!/usr/bin/env python3
"""Generate the tokenizer-specific NeMo-Skills RULER-v1 100/task dataset.

This is a provenance-preserving driver around NVIDIA/RULER's unmodified
``scripts/data/prepare.py``.  It follows the current NeMo-Skills RULER recipe:
all 13 v1 tasks, 100 examples per task, HF tokenization, random seed 42, and a
50-token chat-template reserve.  Generated files are validated and hashed into
``manifest.json`` so GPU workers can prove exactly which inputs they evaluated.
NVIDIA/RULER's canonical v1 shell config uses 500 examples per task; this
100/task adapter protocol is named explicitly and is never labelled that run.
The filename and ``official_ruler_*`` launcher keys are retained only for
backward compatibility; manifests and result schemas use the explicit protocol
name below.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

RULER_V1_TASKS = (
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
SOURCE_DATA_FILES = (
    "english_words.json",
    "PaulGrahamEssays.json",
    "squad.json",
    "hotpotqa.json",
)
RULER_PROTOCOL_LABEL = "NeMo-Skills RULER v1 100-sample/task protocol"
RULER_SAMPLING_PROTOCOL = (
    "pinned NeMo-Skills adapter: 100 samples/task; canonical "
    "NVIDIA/RULER v1 config: 500 samples/task"
)


def _run(*args: str, cwd: Path | None = None, env: dict | None = None) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_jsonl(path: Path, task: str, num_samples: int) -> dict:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            missing = {"input", "outputs", "answer_prefix", "length"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {sorted(missing)}")
            if not isinstance(row["outputs"], list) or not row["outputs"]:
                raise ValueError(f"{path}:{line_number} has no reference outputs")
            rows.append(row)
    if len(rows) != num_samples:
        raise RuntimeError(f"Expected {num_samples} {task} rows in {path}, got {len(rows)}")
    lengths = [int(row.get("length_w_model_temp", row["length"])) for row in rows]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": len(rows),
        "length_w_model_temp_min": min(lengths),
        "length_w_model_temp_max": max(lengths),
    }


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
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
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--nemo-skills-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--tasks", nargs="+", default=list(RULER_V1_TASKS))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--template-reserve", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=None,
        help="Pinned NLTK data directory containing tokenizers/punkt_tab.",
    )
    args = parser.parse_args()

    ruler_repo = args.ruler_repo.resolve()
    nemo_repo = args.nemo_skills_repo.resolve()
    model = args.model.resolve()
    output_root = args.output_root.resolve() / args.model_key
    nltk_data = (
        args.nltk_data.resolve()
        if args.nltk_data is not None
        else args.output_root.resolve() / ".dependencies" / "nltk_data"
    )
    prepare = ruler_repo / "scripts" / "data" / "prepare.py"
    data_workdir = prepare.parent
    if not prepare.is_file() or not model.is_dir():
        raise FileNotFoundError(f"Missing RULER prepare.py or model: {prepare}, {model}")
    unknown = set(args.tasks) - set(RULER_V1_TASKS)
    if unknown:
        raise ValueError(f"Not a RULER v1 task: {sorted(unknown)}")
    if args.num_samples != 100:
        raise ValueError("This pinned NeMo-Skills protocol requires exactly 100 samples/task")
    if min(args.lengths) <= args.template_reserve:
        raise ValueError("Every length must exceed the template reserve")
    punkt_tab = nltk_data / "tokenizers" / "punkt_tab" / "english"
    if not punkt_tab.is_dir():
        raise FileNotFoundError(
            f"Pinned NLTK punkt_tab data is missing at {punkt_tab}. Download "
            f"it with nltk.download('punkt_tab', download_dir={str(nltk_data)!r})."
        )

    source_json_dir = data_workdir / "synthetic" / "json"
    source_data = {}
    for name in SOURCE_DATA_FILES:
        path = source_json_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Pinned upstream RULER source data is missing: {path}. Run NVIDIA's "
                "download_paulgraham_essay.py and download_qa_dataset.sh first."
            )
        source_data[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    jobs = []
    for length in args.lengths:
        save_dir = output_root / str(length)
        for task in args.tasks:
            jobs.append((length, task, save_dir))

    def generate(job: tuple[int, str, Path]) -> tuple[int, str, dict]:
        length, task, save_dir = job
        path = save_dir / task / "test.jsonl"
        if path.is_file():
            try:
                metadata = _validate_jsonl(path, task, args.num_samples)
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                print(f"reuse validated RULER {args.model_key}/{length}/{task}", flush=True)
                return length, task, metadata
        command = [
            sys.executable,
            os.fspath(prepare),
            "--save_dir",
            os.fspath(save_dir),
            "--benchmark",
            "synthetic",
            "--subset",
            "test",
            "--task",
            task,
            "--tokenizer_path",
            os.fspath(model),
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(length - args.template_reserve),
            "--num_samples",
            str(args.num_samples),
            "--random_seed",
            str(args.random_seed),
            "--model_template_type",
            "base",
            "--prepare_for_ns",
        ]
        print("+ " + " ".join(command), flush=True)
        child_env = dict(os.environ)
        child_env["PATH"] = os.pathsep.join(
            [os.fspath(Path(sys.executable).parent), child_env.get("PATH", "")]
        )
        child_env["NLTK_DATA"] = os.fspath(nltk_data)
        output = _run(*command, cwd=data_workdir, env=child_env)
        try:
            metadata = _validate_jsonl(path, task, args.num_samples)
        except Exception as exc:
            raise RuntimeError(
                f"RULER prepare.py did not produce a valid {task} file. "
                f"Captured output:\n{output}"
            ) from exc
        return length, task, metadata

    generated: dict[str, dict] = {str(length): {} for length in args.lengths}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            length, task, metadata = future.result()
            generated[str(length)][task] = metadata
            print(f"validated RULER {args.model_key}/{length}/{task}", flush=True)

    ruler_commit = _run("git", "rev-parse", "HEAD", cwd=ruler_repo)
    nemo_commit = _run("git", "rev-parse", "HEAD", cwd=nemo_repo)
    model_files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = model / name
        if path.is_file():
            model_files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    model_revisions = set()
    for metadata in (model / ".cache" / "huggingface" / "download").glob("*.metadata"):
        try:
            first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, UnicodeDecodeError, IndexError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first_line):
            model_revisions.add(first_line)
    nltk_files = {}
    for path in sorted(punkt_tab.rglob("*")):
        if path.is_file():
            nltk_files[str(path.relative_to(nltk_data))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": RULER_PROTOCOL_LABEL,
        "protocol": {
            "tasks": list(args.tasks),
            "num_samples_per_task": args.num_samples,
            "random_seed": args.random_seed,
            "tokenizer_type": "hf",
            "lengths": args.lengths,
            "chat_template_reserve": args.template_reserve,
            "prepare_for_nemo_skills": True,
            "sampling_protocol": RULER_SAMPLING_PROTOCOL,
        },
        "sources": {
            "ruler_repo": str(ruler_repo),
            "ruler_commit": ruler_commit,
            "nemo_skills_repo": str(nemo_repo),
            "nemo_skills_commit": nemo_commit,
            "source_data": source_data,
            "nltk": {
                "version": importlib.metadata.version("nltk"),
                "data_root": str(nltk_data),
                "punkt_tab_files": nltk_files,
            },
        },
        "model": {
            "key": args.model_key,
            "path": str(model),
            "revision": (next(iter(model_revisions)) if len(model_revisions) == 1 else None),
            "snapshot_revisions": sorted(model_revisions),
            "metadata_files": model_files,
        },
        "files": generated,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    print(f"Wrote {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
