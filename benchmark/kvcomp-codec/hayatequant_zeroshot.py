"""Launch and summarize HayateQuant zero-shot downstream runs.

The driver mirrors ``hayatequant_cross_model.py`` so the paper can report the
same eight LLMs for WikiText-2 PPL and lm-eval-style zero-shot accuracy.

Examples
--------
Submit the default eight-model grid:

    python benchmark/kvcomp-codec/hayatequant_zeroshot.py --submit

Summarize completed runs:

    python benchmark/kvcomp-codec/hayatequant_zeroshot.py --summarize \
        --output-root /lustre/.../ichikawa_works/kvcomp_hayatequant/zeroshot_v1

Smoke test one model with a small per-task sample:

    python benchmark/kvcomp-codec/hayatequant_zeroshot.py --run-local \
        --models qwen3-0.6b --limit 4 --tasks piqa
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from hayatequant_cross_model import (  # noqa: E402
    ALL_SPECS,
    DEFAULT_MODELS,
    DEFAULT_MODELS_ROOT,
    ModelSpec,
)


ZEROSHOT_BENCH = THIS_DIR / "zeroshot_benchmark.py"
DEFAULT_OUTPUT_ROOT = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
    "kvcomp_hayatequant/zeroshot_v1"
)

EXTRA_SMOKE_MODELS = {
    "qwen3-0.6b": ModelSpec("qwen3-0.6b", "Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen3-1.7B", "Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
}
ALL_ZERO_SPECS = dict(ALL_SPECS)
ALL_ZERO_SPECS.update(EXTRA_SMOKE_MODELS)


def _run(cmd: list[str], *, dry_run: bool = False, cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _resolve_specs(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return list(DEFAULT_MODELS)
    specs = []
    for name in names:
        if name in ALL_ZERO_SPECS:
            specs.append(ALL_ZERO_SPECS[name])
        else:
            p = Path(name)
            if p.exists() or "/" in name:
                specs.append(ModelSpec(p.name, str(p), None, p.name))
            else:
                known = ", ".join(sorted(ALL_ZERO_SPECS))
                raise SystemExit(f"Unknown model {name!r}. Known keys: {known}")
    return specs


def _model_path(spec: ModelSpec, models_root: Path) -> Path:
    p = Path(spec.path_name)
    return p if p.is_absolute() else models_root / p


def _bench_cmd(args: argparse.Namespace, model_path: Path) -> list[str]:
    cmd = [
        str(args.python),
        "-u",
        str(ZEROSHOT_BENCH),
        "--model-path",
        str(model_path),
        "--output-root",
        str(args.output_root),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--bits",
        *[str(b) for b in args.bits],
        "--tasks",
        *args.tasks,
        "--methods",
        *args.methods,
        "--sample-seed",
        str(args.sample_seed),
        "--sink-tokens",
        str(args.sink_tokens),
        "--outlier-ratio",
        str(args.outlier_ratio),
    ]
    if args.include_fullkv:
        cmd.append("--include-fullkv")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if not args.skip_outliers:
        cmd.append("--no-skip-outliers")
    return cmd


def _launch(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    for spec in _resolve_specs(args.models):
        model_path = _model_path(spec, args.models_root)
        if not model_path.exists():
            raise SystemExit(f"Missing model directory for {spec.key}: {model_path}")
        if args.run_local:
            _run(_bench_cmd(args, model_path), dry_run=args.dry_run, cwd=REPO_ROOT)
            continue
        logs = args.output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        bench_cmd = " ".join(shlex.quote(x) for x in _bench_cmd(args, model_path))
        wrap = (
            f"cd {shlex.quote(str(REPO_ROOT))}; "
            "export TOKENIZERS_PARALLELISM=false; "
            "export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}; "
            f"{bench_cmd}"
        )
        cmd = [
            "sbatch",
            f"--job-name=hq-zs-{spec.key}",
            f"--output={logs}/%x_%j.log",
            f"--partition={args.partition}",
            f"--gres=gpu:{args.gpus}",
            f"--cpus-per-task={args.cpus}",
            f"--mem={args.mem}",
            f"--time={args.time}",
            "--wrap",
            wrap,
        ]
        _run(cmd, dry_run=args.dry_run, cwd=REPO_ROOT)


def _method(payload: dict, bits: str, name: str) -> dict | None:
    return (payload.get("results") or {}).get(bits, {}).get(name)


def _best_hayate(payload: dict, bits: str) -> tuple[str | None, dict | None]:
    methods = (payload.get("results") or {}).get(bits, {})
    candidates = [(k, v) for k, v in methods.items() if k.startswith("HayateQuant")]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1].get("avg_acc_norm", -1.0))


def _fmt(x: float | None, scale: float = 100.0) -> str:
    return "--" if x is None else f"{scale * x:.1f}"


def _merge_task_payloads(paths: list[Path]) -> list[dict]:
    grouped: dict[str, list[tuple[Path, dict]]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        grouped.setdefault(payload.get("model", path.stem), []).append((path, payload))

    merged_payloads: list[dict] = []
    for model, items in sorted(grouped.items()):
        if len(items) == 1:
            merged_payloads.append(items[0][1])
            continue

        first = items[0][1]
        merged = {
            "model": model,
            "tasks": [],
            "limit": first.get("limit"),
            "sample_seed": first.get("sample_seed"),
            "config": first.get("config", {}),
            "results": {},
            "sources": [str(path) for path, _ in items],
        }
        task_order: list[str] = []
        for _, payload in items:
            for task in payload.get("tasks", []):
                if task not in task_order:
                    task_order.append(task)
            for bits, methods in (payload.get("results") or {}).items():
                out_methods = merged["results"].setdefault(bits, {})
                for method, data in methods.items():
                    out_data = out_methods.setdefault(method, {"tasks": {}})
                    out_data["tasks"].update(data.get("tasks", {}))

        merged["tasks"] = task_order
        for methods in merged["results"].values():
            for data in methods.values():
                tasks = data.get("tasks", {})
                acc = [v["acc"] for v in tasks.values()]
                acc_norm = [v["acc_norm"] for v in tasks.values()]
                data["avg_acc"] = round(mean(acc), 4) if acc else 0.0
                data["avg_acc_norm"] = round(mean(acc_norm), 4) if acc_norm else 0.0
                data["n_total"] = sum(v["n"] for v in tasks.values())
        merged_payloads.append(merged)
    return merged_payloads


def _summarize(args: argparse.Namespace) -> None:
    paths = sorted(args.output_root.rglob("zeroshot_*.json"))
    if not paths:
        raise SystemExit(f"No zero-shot result JSON files found under {args.output_root}")
    payloads = _merge_task_payloads(paths)
    expected_tasks = set(args.tasks)
    if args.require_complete:
        incomplete = []
        for payload in payloads:
            got = set(payload.get("tasks") or [])
            if got != expected_tasks:
                missing = ", ".join(sorted(expected_tasks - got))
                incomplete.append(f"{payload.get('model')}: missing {missing}")
        if incomplete:
            raise SystemExit("Incomplete zero-shot results:\n" + "\n".join(incomplete))
    rows = [
        "| Model | Tasks | 2b TQ acc_norm | 2b HQ acc_norm | delta | 3b TQ acc_norm | 3b HQ acc_norm | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tq2_vals: list[float] = []
    hq2_vals: list[float] = []
    tq3_vals: list[float] = []
    hq3_vals: list[float] = []
    for payload in payloads:
        model = payload.get("model", "unknown")
        task_count = len(payload.get("tasks") or [])
        tq2 = _method(payload, "2", "TurboQuant-hadamard")
        hq2_name, hq2 = _best_hayate(payload, "2")
        tq3 = _method(payload, "3", "TurboQuant-hadamard")
        hq3_name, hq3 = _best_hayate(payload, "3")
        tq2_a = None if tq2 is None else tq2.get("avg_acc_norm")
        hq2_a = None if hq2 is None else hq2.get("avg_acc_norm")
        tq3_a = None if tq3 is None else tq3.get("avg_acc_norm")
        hq3_a = None if hq3 is None else hq3.get("avg_acc_norm")
        if tq2_a is not None and hq2_a is not None:
            tq2_vals.append(tq2_a); hq2_vals.append(hq2_a)
        if tq3_a is not None and hq3_a is not None:
            tq3_vals.append(tq3_a); hq3_vals.append(hq3_a)
        rows.append(
            f"| {model} | {task_count} | {_fmt(tq2_a)} | {_fmt(hq2_a)}"
            f" ({(hq2_name or '').replace('HayateQuant-', 'HQ-')}) | {_fmt(None if tq2_a is None or hq2_a is None else hq2_a - tq2_a, 100.0)} |"
            f" {_fmt(tq3_a)} | {_fmt(hq3_a)}"
            f" ({(hq3_name or '').replace('HayateQuant-', 'HQ-')}) | {_fmt(None if tq3_a is None or hq3_a is None else hq3_a - tq3_a, 100.0)} |"
        )
    if tq2_vals and tq3_vals:
        rows.append(
            f"| **Average** | **{len(args.tasks)}** | **{_fmt(mean(tq2_vals))}** | **{_fmt(mean(hq2_vals))}** | "
            f"**{_fmt(mean(hq2_vals) - mean(tq2_vals), 100.0)}** | "
            f"**{_fmt(mean(tq3_vals))}** | **{_fmt(mean(hq3_vals))}** | "
            f"**{_fmt(mean(hq3_vals) - mean(tq3_vals), 100.0)}** |"
        )
    text = "\n".join(rows)
    print(text)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n")
        print(f"\nWrote {args.summary_out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--python", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bits", nargs="+", default=["2", "3"])
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande"])
    ap.add_argument("--methods", nargs="+", default=["TurboQuant-hadamard", "HayateQuant-auto", "HayateQuant-K12", "HayateQuant-dense-LS"])
    ap.add_argument("--include-fullkv", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-seed", type=int, default=2027)
    ap.add_argument("--sink-tokens", type=int, default=4)
    ap.add_argument("--skip-outliers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--outlier-ratio", type=float, default=50.0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partition", default="batch-1gpu")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="180G")
    ap.add_argument("--time", default="06:00:00")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        _summarize(args)
        return
    if not (args.submit or args.run_local):
        args.submit = True
    _launch(args)


if __name__ == "__main__":
    main()
