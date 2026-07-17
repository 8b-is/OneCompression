"""Run the Table-1 PPL and zero-shot benchmark pair.

This wrapper keeps the two existing benchmark entry points as the source of
truth and only coordinates their arguments.  It is useful when adding a new
codec row, because the same model list, bit list, and method list are sent to
both WikiText-2 PPL and five-task zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CROSS_MODEL = REPO_ROOT / "benchmark" / "kvcomp-codec" / "hayatequant_cross_model.py"
ZEROSHOT = REPO_ROOT / "benchmark" / "kvcomp-codec" / "hayatequant_zeroshot.py"
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("KVCOMP_OUTPUT_ROOT", str(REPO_ROOT / "results" / "hayatequant" / "table1"))
)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _hydra_list(values: list[str]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--bits", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--methods", nargs="+", default=["AdaptiveHQ"])
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--partition", default="debug")
    ap.add_argument("--time", default="03:00:00")
    ap.add_argument("--mem", default="180G")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ppl_root = args.output_root / "ppl"
    zs_root = args.output_root / "zeroshot"
    mode = "--run-local" if args.run_local else "--submit"
    if args.summarize:
        _run([
            args.python, str(CROSS_MODEL), "--summarize",
            "--output-root", str(ppl_root),
        ], dry_run=args.dry_run)
        _run([
            args.python, str(ZEROSHOT), "--summarize",
            "--output-root", str(zs_root),
        ], dry_run=args.dry_run)
        return

    if not (args.run_local or args.submit):
        args.submit = True
    model_args = [] if not args.models else ["--models", *args.models]
    _run([
        args.python, str(CROSS_MODEL), mode,
        "--output-root", str(ppl_root),
        "--partition", args.partition,
        "--time", args.time,
        "--mem", args.mem,
        *model_args,
        "--override", f"bits={_hydra_list(args.bits)}",
        "--override", f"methods={_hydra_list(args.methods)}",
    ], dry_run=args.dry_run)
    _run([
        args.python, str(ZEROSHOT), mode,
        "--output-root", str(zs_root),
        "--device", args.device,
        "--partition", args.partition,
        "--time", args.time,
        "--mem", args.mem,
        "--bits", *args.bits,
        "--methods", *args.methods,
        *model_args,
    ], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
