#!/usr/bin/env python3
"""Run isolated GPU generation smokes for TaperKV, HQ, and HQ+TaperKV.

Each vLLM engine runs in a fresh interpreter because attention backend
registration is process-global.  The suite retains child stdout/stderr and
emits one aggregate receipt even when a phase fails, which makes Slurm failures
diagnosable without rerunning successful phases.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEC_BENCHMARK = REPO_ROOT / "benchmark" / "kvcomp-codec"
RENDERER = CODEC_BENCHMARK / "render_vllm_inference_smoke_latex.py"
SOURCE_FILES = tuple(
    sorted(
        {
            Path(__file__).resolve(),
            CODEC_BENCHMARK / "taperkv_vllm_reference_smoke.py",
            CODEC_BENCHMARK / "hayatequant_vllm_native_generate.py",
            RENDERER,
            *(REPO_ROOT / "vllm_plugins" / "kvcomp").glob("*.py"),
            *(REPO_ROOT / "onecomp" / "kvcomp").rglob("*.py"),
        }
    )
)


def _command_specs(args: argparse.Namespace) -> list[tuple[str, list[str], Path]]:
    common_taper = [
        "--model",
        str(args.model),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--key-bits",
        "4",
        "--value-bits",
        "4",
        "--key-floor",
        "2",
        "--value-floor",
        "2",
        "--sink-tokens",
        "0",
        "--recent-tokens",
        "0",
        "--requant-stride",
        "2",
    ]
    phases = [
        (
            "taperkv",
            [
                str(CODEC_BENCHMARK / "taperkv_vllm_reference_smoke.py"),
                *common_taper,
                "--output",
                str(args.output_dir / "taperkv.json"),
            ],
            args.output_dir / "taperkv.json",
        ),
        (
            "hayatequant",
            [
                str(CODEC_BENCHMARK / "hayatequant_vllm_native_generate.py"),
                "--model-path",
                str(args.model),
                "--output",
                str(args.output_dir / "hayatequant.json"),
                "--variant",
                "hayate_full",
                "--bits",
                "3",
                "--max-model-len",
                str(args.max_model_len),
                "--max-tokens",
                str(args.max_tokens),
                "--gpu-memory-utilization",
                "0.5",
                "--enforce-eager",
            ],
            args.output_dir / "hayatequant.json",
        ),
        (
            "hayatequant_taperkv",
            [
                str(CODEC_BENCHMARK / "taperkv_vllm_reference_smoke.py"),
                *common_taper,
                "--hayate-bits",
                "3",
                "--hayate-value-bits",
                "3",
                "--hayate-block-size",
                "2",
                "--hayate-encoder",
                "exact",
                "--output",
                str(args.output_dir / "hayatequant_taperkv.json"),
            ],
            args.output_dir / "hayatequant_taperkv.json",
        ),
    ]
    return phases


def _run_capture(command: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["VLLM_PLUGINS"] = "kvcomp"
    # vLLM 0.20's B200 capability probe can find the bundled DeepGEMM module
    # while its wheel lacks a newer required alignment symbol.  Its warmup then
    # aborts even for this BF16 model before attention is exercised.  These
    # smokes do not benchmark weight GEMM, so disable that optional backend and
    # record the choice in the aggregate contract below.
    env["VLLM_USE_DEEP_GEMM"] = "0"
    env["VLLM_MOE_USE_DEEP_GEMM"] = "0"
    env["VLLM_DEEP_GEMM_WARMUP"] = "skip"
    # FlashInfer 0.6 writes absolute source paths into build.ninja.  A shared
    # home cache created by another virtualenv is therefore not relocatable.
    # Keep this suite's optional JIT artifacts beside its receipts.
    env["FLASHINFER_WORKSPACE_BASE"] = str(stdout_path.parent / "runtime-cache")
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return completed.returncode, time.perf_counter() - started


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _text_command(command: list[str]) -> list[str]:
    """Record argv without shell quoting ambiguity."""

    return [sys.executable, *command]


def _provenance() -> dict[str, Any]:
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCE_FILES
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    packages = {}
    for name in ("onecomp", "torch", "transformers", "vllm"):
        packages[name] = importlib.metadata.version(name)
    return {
        "git_head": git_head,
        "git_status_short": git_status,
        "source_sha256": source_hashes,
        "packages": packages,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the three isolated engines and return their aggregate receipt."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError(
            "output-dir must be new or empty; refusing to mix stale receipts with this run"
        )
    phases = []
    for name, command, receipt_path in _command_specs(args):
        stdout_path = args.output_dir / f"{name}.stdout.log"
        stderr_path = args.output_dir / f"{name}.stderr.log"
        returncode, elapsed = _run_capture(command, stdout_path, stderr_path)
        receipt = _read_json(receipt_path)
        phases.append(
            {
                "name": name,
                "returncode": returncode,
                "passed": returncode == 0 and isinstance(receipt, dict),
                "elapsed_seconds": round(elapsed, 4),
                "command_argv": _text_command(command),
                "receipt": str(receipt_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "receipt_schema": (
                    receipt.get("schema", receipt.get("schema_version"))
                    if isinstance(receipt, dict)
                    else None
                ),
            }
        )
    return {
        "schema": {"name": "onecomp.kvcomp_vllm_inference_smoke_suite", "version": 1},
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(REPO_ROOT),
        "python": sys.version,
        "provenance": _provenance(),
        "model": str(args.model),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_environment_contract": {
            "VLLM_PLUGINS": "kvcomp",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_MOE_USE_DEEP_GEMM": "0",
            "VLLM_DEEP_GEMM_WARMUP": "skip",
            "FLASHINFER_WORKSPACE_BASE": "<output-dir>/runtime-cache",
            "reason": (
                "KV-attention correctness smoke; optional weight DeepGEMM is outside scope"
            ),
        },
        "phases": phases,
        "passed": all(phase["passed"] for phase in phases),
    }


def parse_args() -> argparse.Namespace:
    """Parse and validate the standalone smoke-suite arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if args.max_model_len < 1 or args.max_tokens < 1:
        parser.error("max-model-len and max-tokens must be positive")
    if args.max_tokens >= args.max_model_len:
        parser.error("max-tokens must be smaller than max-model-len")
    return args


def _render_validated_suite(suite_path: Path, output_path: Path) -> subprocess.CompletedProcess:
    """Run the fail-closed receipt validator and emit the paper-ready table."""

    return subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--suite",
            str(suite_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    """Write the aggregate receipt and render it only after strict validation."""

    args = parse_args()
    result = run(args)
    aggregate = args.output_dir / "suite.json"
    aggregate.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if result["passed"]:
        rendered_path = args.output_dir / "vllm_inference_smoke.tex"
        validation = _render_validated_suite(aggregate, rendered_path)
        if validation.returncode != 0:
            # A zero-exit child plus a JSON object is insufficient: the shared
            # renderer checks schemas, model identity, generation totals,
            # runtime-event digests, and representation labels.  Rewrite the
            # aggregate as failed if any of those contracts is violated.
            result["passed"] = False
            result["receipt_validation_error"] = (
                validation.stderr.strip()
                or validation.stdout.strip()
                or "receipt validator failed without diagnostic output"
            )
            aggregate.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
