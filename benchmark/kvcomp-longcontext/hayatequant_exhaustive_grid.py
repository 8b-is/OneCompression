"""Submit exhaustive HayateQuant long/reasoning grids.

The main paper keeps long-context and reasoning tables compact.  This driver is
for the exhaustive follow-up battery: seven models x bits {2,3,4} x
{TurboQuant, HQ-frontier}.  New single-codec HQ candidates can be added as
ordinary HayateQuant variants, not as model-family selectors.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from onecomp.kvcomp.codec.hayate.recipes import (  # noqa: E402
    hayate_adaptive_rate_variant,
    hayate_frontier_variant,
    hayate_hq_frontier_variant,
)
from benchmark.kvcomp_common.paths import (  # noqa: E402
    resolve_model_ref,
    resolve_output_root,
    resolve_python,
)


KV_BENCH = REPO_ROOT / "benchmark" / "kvcomp-longcontext" / "kv_benchmark.py"
DEFAULT_MODELS_ROOT = None
DEFAULT_OUTPUT_ROOT = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path_name: str

    @property
    def label(self) -> str:
        return self.path_name


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("llama2-7b", "Llama-2-7b-hf"),
    ModelSpec("llama31-8b", "llama-3.1-8b-instruct"),
    ModelSpec("qwen3-4b", "Qwen3-4B"),
    ModelSpec("qwen3-8b", "Qwen3-8B"),
    ModelSpec("r1-llama-8b", "DeepSeek-R1-Distill-Llama-8B"),
    ModelSpec("r1-qwen-7b", "DeepSeek-R1-Distill-Qwen-7B"),
    ModelSpec("r1-qwen-1.5b", "DeepSeek-R1-Distill-Qwen-1.5B"),
)
MODEL_MAP = {m.key: m for m in MODELS}
MODEL_MAP.update({m.path_name: m for m in MODELS})


def _frontier_variant(bits: int, label: str, **overrides: Any) -> dict[str, Any]:
    return hayate_frontier_variant(bits, label=label, **overrides)


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fmt_dict(d: dict[str, Any]) -> str:
    return "{" + ",".join(f"{k}:{_fmt_value(v)}" for k, v in d.items()) + "}"


def _fmt_list(items: list[Any]) -> str:
    parts = [_fmt_dict(x) if isinstance(x, dict) else _fmt_value(x) for x in items]
    return "[" + ",".join(parts) + "]"


def _variants(spec: ModelSpec, bits: list[int]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    methods = ["FullKV"]
    turbo: list[dict[str, Any]] = []
    hayate: list[dict[str, Any]] = []
    for b in bits:
        tq = f"TurboQuant-{b}bit"
        hq = f"HQ-{b}bit"
        frontier = f"HQ-Frontier-{b}bit"
        adaptive = f"AdaptiveHQ-{b}bit"
        rotation = f"RotationHQ-{b}bit"
        methods.extend([tq, hq, frontier, adaptive, rotation])
        turbo.append({"label": tq, "bits": b, "dist": "beta"})
        hayate.extend(
            [
                _frontier_variant(b, hq),
                hayate_hq_frontier_variant(b, frontier),
                hayate_adaptive_rate_variant(b, adaptive),
                _frontier_variant(b, rotation, rotation_mode="dense"),
            ]
        )
    return methods, turbo, hayate


def _model_path(spec: ModelSpec, root: Path | None) -> Path:
    return Path(resolve_model_ref(spec.path_name, root))


def _resolve_models(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return list(MODELS)
    out = []
    for name in names:
        if name not in MODEL_MAP:
            path = Path(name).expanduser()
            if path.exists() or "/" in name or os.sep in name:
                out.append(ModelSpec(path.name, str(path)))
                continue
            known = ", ".join(sorted(MODEL_MAP))
            raise SystemExit(f"Unknown model {name!r}. Known: {known}; or pass a model path")
        out.append(MODEL_MAP[name])
    return out


def _base_overrides(args: argparse.Namespace, spec: ModelSpec, task: str) -> list[str]:
    bits = list(args.bits)
    methods, turbo, hayate = _variants(spec, bits)
    out = [
        "--config-name",
        "hayatequant_exhaustive_grid",
        f"model_path={_model_path(spec, args.models_root)}",
        f"output_dir={args.output_root / task / spec.key}",
        f"methods={_fmt_list(methods)}",
        f"turboquant_variants={_fmt_list(turbo)}",
        f"hayatequant_variants={_fmt_list(hayate)}",
        "halo_variants=[]",
        "codec_variants=[]",
        "uniform_bits=[]",
        "kivi_bits=[]",
    ]
    if args.smoke:
        out.extend([
            "context_lens=[4096]",
            "depths=[0.5]",
            "seeds=1",
            "ruler_seeds=1",
            "longbench_limit=2",
            "ppl_windows=2",
        ])
    return out


def _task_overrides(args: argparse.Namespace, spec: ModelSpec, task: str) -> list[str]:
    out = _base_overrides(args, spec, task)
    if task == "long":
        out.extend([
            "run_needle=true",
            "run_ruler=true",
            f"run_longbench={'false' if args.smoke else 'true'}",
            "run_ppl=true",
            "run_gsm8k=false",
            "run_math=false",
        ])
    elif task == "reason":
        out.extend([
            "run_needle=false",
            "run_ruler=false",
            "run_longbench=false",
            "run_ppl=true",
            "ppl_max_length=2048",
            "ppl_stride=1024",
            "run_gsm8k=true",
            f"gsm8k_limit={1 if args.smoke else args.reason_limit}",
            f"max_new_reason={64 if args.smoke else 4096}",
            f"run_math={'false' if args.smoke else 'true'}",
            f"math_limit={0 if args.smoke else args.reason_limit}",
            f"max_new_math={64 if args.smoke else 4096}",
        ])
    else:
        raise ValueError(task)
    return out


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def _submit(args: argparse.Namespace, spec: ModelSpec, task: str, cmd: list[str]) -> None:
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    wrap = (
        f"cd {shlex.quote(str(REPO_ROOT))}; "
        "export TOKENIZERS_PARALLELISM=false; "
        "export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}; "
        "export PYTHONUNBUFFERED=1; "
        + " ".join(shlex.quote(x) for x in cmd)
    )
    sbatch = [
        "sbatch",
        f"--job-name=hq{task}-{spec.key}",
        f"--output={logs}/%x_%j.log",
        f"--partition={args.partition}",
        "--gres=gpu:1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        "--wrap",
        wrap,
    ]
    _run(sbatch, cwd=REPO_ROOT, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    args.models_root = args.models_root.expanduser().resolve() if args.models_root else None
    args.output_root = resolve_output_root(
        args.output_root, default_relative="results/hayatequant/exhaustive_long_reason"
    )
    args.python = resolve_python(args.python)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = ["long", "reason"] if args.task == "both" else [args.task]
    for spec in _resolve_models(args.models):
        model_path = _model_path(spec, args.models_root)
        is_hf_id = "/" in spec.path_name and not spec.path_name.startswith((".", "/", "~"))
        if not model_path.exists() and not is_hf_id:
            raise SystemExit(f"Missing model directory: {model_path}")
        for task in tasks:
            cmd = [str(args.python), "-u", str(KV_BENCH), *_task_overrides(args, spec, task)]
            if args.run_local:
                _run(cmd, cwd=REPO_ROOT, dry_run=args.dry_run)
            else:
                _submit(args, spec, task, cmd)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["long", "reason", "both"], default="both")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT, help="External model directory.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--run-local", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Use tiny task/sample settings for command validation.")
    ap.add_argument("--reason-limit", type=int, default=100)
    ap.add_argument("--partition", default="debug,batch-1gpu")
    ap.add_argument("--cpus", type=int, default=20)
    ap.add_argument("--mem", default="180G")
    ap.add_argument("--time", default="1-00:00:00")
    return ap.parse_args()


if __name__ == "__main__":
    main()
