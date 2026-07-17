"""Portable paper-benchmark orchestrator.

This entry point is intentionally small: the codec and zero-shot drivers own
their scoring semantics, while this file only composes them.  Model weights
are supplied by an external manifest, ``--models-root``, local paths, or HF
ids; no model directory is bundled with onecomp.

Examples::

    python benchmark/kvcomp-codec/run_paper_pipeline.py \
      --manifest benchmark/configs/paper_models.example.json --dry-run
    python benchmark/kvcomp-codec/run_paper_pipeline.py \
      --models-root /external/models --stages codec zeroshot
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.kvcomp_common.manifest import load_model_manifest  # noqa: E402
from benchmark.kvcomp_common.paths import resolve_output_root, resolve_python  # noqa: E402


def _models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return list(args.models)
    if args.manifest:
        return [item["ref"] for item in load_model_manifest(args.manifest)]
    raise SystemExit("Provide --models or --manifest; model weights are external to onecomp.")


def _base(args: argparse.Namespace, script: str, output: Path) -> list[str]:
    cmd = [str(args.python), "-u", str(REPO_ROOT / script)]
    cmd += ["--models", *_models(args), "--output-root", str(output)]
    if args.models_root:
        cmd += ["--models-root", str(args.models_root)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.run_local:
        cmd.append("--run-local")
    cmd += ["--partition", args.partition, "--cpus", str(args.cpus), "--mem", args.mem, "--time", args.time]
    if args.gpus is not None and script.endswith("hayatequant_cross_model.py"):
        cmd += ["--gpus", str(args.gpus)]
    return cmd


def _commands(args: argparse.Namespace) -> list[list[str]]:
    root = resolve_output_root(args.output_root, default_relative="results/hayatequant/paper")
    commands: list[list[str]] = []
    stages = set(args.stages)
    if "codec" in stages:
        commands.append(_base(args, "benchmark/kvcomp-codec/hayatequant_cross_model.py", root / "codec"))
    if "zeroshot" in stages:
        commands.append(_base(args, "benchmark/kvcomp-codec/hayatequant_zeroshot.py", root / "zeroshot"))
    if "longcontext" in stages:
        commands.append(_base(args, "benchmark/kvcomp-longcontext/hayatequant_exhaustive_grid.py", root / "longcontext"))
    if "render" in stages:
        if args.aggregate is None:
            raise SystemExit("--aggregate is required for the render stage")
        commands.append([
            str(args.python), "-u",
            str(REPO_ROOT / "benchmark/kvcomp-codec/render_hayatequant_formal_results.py"),
            "--aggregate", str(args.aggregate),
            "--output-dir", str(root / "figures_tables"),
        ])
    if "production-render" in stages:
        commands.append([
            str(args.python), "-u",
            str(REPO_ROOT / "benchmark/kvcomp-codec/render_hayatequant_production.py"),
            "--results-root", str(root),
            "--output-dir", str(root / "production_figures_tables"),
        ])
    if not commands:
        raise SystemExit("No stages selected")
    return commands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, help="External JSON manifest with models:[{key,ref}]")
    ap.add_argument("--models", nargs="*", help="External model paths or Hugging Face ids")
    ap.add_argument("--models-root", type=Path, help="Optional external directory containing model paths")
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--stages", nargs="+", choices=["codec", "zeroshot", "longcontext", "render", "production-render"], default=["codec", "zeroshot", "longcontext"])
    ap.add_argument("--aggregate", type=Path, help="Validated aggregate JSON for the render stage")
    ap.add_argument("--python", type=Path)
    ap.add_argument("--run-local", action="store_true", help="Run child drivers locally; otherwise they submit Slurm jobs")
    ap.add_argument("--dry-run", action="store_true", help="Print child commands without launching them")
    ap.add_argument("--partition", default="batch-1gpu")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="180G")
    ap.add_argument("--time", default="06:00:00")
    args = ap.parse_args()
    args.python = resolve_python(args.python)
    if args.models_root:
        args.models_root = args.models_root.expanduser().resolve()
    for command in _commands(args):
        print("+ " + " ".join(shlex.quote(x) for x in command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
