"""Launch and summarize the HayateQuant cross-model codec benchmark.

This is a thin orchestration layer around ``codec_benchmark.py`` for the
HayateQuant paper.  It keeps the actual benchmark implementation in one place
and makes the main-paper multi-model table reproducible:

* verify that each requested model directory exists,
* optionally download missing public Hugging Face snapshots,
* submit one Slurm job per model (or print the commands), and
* summarize existing ``results_*.json`` files into a compact Markdown table.

Examples
--------
Submit the default seven-model paper grid:

    python benchmark/kvcomp-codec/hayatequant_cross_model.py --submit

Summarize finished runs:

    python benchmark/kvcomp-codec/hayatequant_cross_model.py \
        --summarize --output-root /lustre/.../ichikawa_works/kvcomp_hayatequant/results_cross_model_v1

Run one model directly inside an interactive GPU allocation:

    srun --gres=gpu:1 python benchmark/kvcomp-codec/hayatequant_cross_model.py \
        --run-local --models qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from onecomp.kvcomp.codec.hayate.recipes import hayate_result_method_candidates  # noqa: E402
from benchmark.kvcomp_common.paths import resolve_model_ref, resolve_output_root, resolve_python  # noqa: E402

CODEC_BENCH = REPO_ROOT / "benchmark" / "kvcomp-codec" / "codec_benchmark.py"
DEFAULT_MODELS_ROOT = None
DEFAULT_OUTPUT_ROOT = None


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path_name: str
    hf_id: str | None = None
    label: str | None = None

    @property
    def display(self) -> str:
        return self.label or self.path_name


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("llama2-7b", "Llama-2-7b-hf", None, "Llama-2-7B"),
    ModelSpec("llama31-8b", "llama-3.1-8b-instruct", None, "Llama-3.1-8B"),
    ModelSpec("qwen3-4b", "Qwen3-4B", "Qwen/Qwen3-4B", "Qwen3-4B"),
    ModelSpec("qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B", "Qwen3-8B"),
    ModelSpec(
        "r1-llama-8b",
        "DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "R1-Distill-Llama-8B",
    ),
    ModelSpec(
        "r1-qwen-7b",
        "DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "R1-Distill-Qwen-7B",
    ),
    ModelSpec(
        "r1-qwen-1.5b",
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "R1-Distill-Qwen-1.5B",
    ),
)

# Extra useful coverage for follow-up grids; pass with ``--models qwen3-4b`` etc.
EXTRA_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("ministral-8b", "Ministral-8B-Instruct-2410", None, "Ministral-8B"),
    ModelSpec("qwen3-0.6b", "Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
    ModelSpec("qwen3-1.7b", "Qwen3-1.7B", "Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
    ModelSpec("qwen3-14b", "Qwen3-14B", "Qwen/Qwen3-14B", "Qwen3-14B"),
    ModelSpec(
        "r1-qwen-14b",
        "DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "R1-Distill-Qwen-14B",
    ),
)

ALL_SPECS = {m.key: m for m in (*DEFAULT_MODELS, *EXTRA_MODELS)}
ALL_SPECS.update({m.path_name: m for m in (*DEFAULT_MODELS, *EXTRA_MODELS)})


def _run(cmd: list[str], *, dry_run: bool = False, cwd: Path | None = None) -> None:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _download_model(spec: ModelSpec, model_dir: Path, dry_run: bool) -> None:
    if spec.hf_id is None:
        raise SystemExit(
            f"Model {spec.key!r} is missing at {model_dir}, and no public hf_id "
            "is registered. Download or symlink the weights manually."
        )
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={spec.hf_id!r}, local_dir={str(model_dir)!r}, "
        "local_dir_use_symlinks=False)"
    )
    _run([sys.executable, "-c", code], dry_run=dry_run)


def _resolve_specs(names: list[str] | None) -> list[ModelSpec]:
    if not names:
        return list(DEFAULT_MODELS)
    specs: list[ModelSpec] = []
    for name in names:
        if os.sep in name or name.startswith("."):
            p = Path(name)
            if p.exists():
                specs.append(ModelSpec(p.name, str(p), None, p.name))
            elif name.startswith(("/", "~", ".")):
                specs.append(ModelSpec(p.name, str(p), None, p.name))
            else:
                # A slash-containing non-local value is a valid Hugging Face
                # repository id; weights remain outside this repository.
                specs.append(ModelSpec(name.replace("/", "-"), name, name, name))
        elif name in ALL_SPECS:
            specs.append(ALL_SPECS[name])
        else:
            known = ", ".join(sorted({m.key for m in (*DEFAULT_MODELS, *EXTRA_MODELS)}))
            raise SystemExit(f"Unknown model {name!r}. Known keys: {known}")
    return specs


def _model_ref(spec: ModelSpec, models_root: Path | None) -> str:
    """Return a local external path when available, otherwise a HF id."""

    local = resolve_model_ref(spec.path_name, models_root)
    if Path(local).exists():
        return local
    if models_root is None and spec.hf_id:
        return spec.hf_id
    return local


def _benchmark_cmd(args: argparse.Namespace, spec: ModelSpec, model_path: Path) -> list[str]:
    out_dir = args.output_root / spec.key
    cmd = [
        str(args.python),
        "-u",
        str(CODEC_BENCH),
        "--config-name",
        args.config_name,
        f"model_path={model_path}",
        f"output_dir={out_dir}",
    ]
    cmd.extend(args.override)
    return cmd


def _submit_sbatch(args: argparse.Namespace, spec: ModelSpec, model_path: Path) -> None:
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    bench_cmd = " ".join(shlex.quote(x) for x in _benchmark_cmd(args, spec, model_path))
    wrap = (
        f"cd {shlex.quote(str(REPO_ROOT))}; "
        "export TOKENIZERS_PARALLELISM=false; "
        "export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}; "
        f"{bench_cmd}"
    )
    cmd = [
        "sbatch",
        f"--job-name=hq-{spec.key}",
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


def _launch(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    for spec in _resolve_specs(args.models):
        model_ref = _model_ref(spec, args.models_root)
        model_path = Path(model_ref)
        if not model_path.exists() and args.download_missing:
            if args.models_root is None or not spec.hf_id:
                raise SystemExit(
                    f"Cannot download {spec.key!r} without --models-root and a registered hf id."
                )
            model_path = args.models_root / spec.path_name
            _download_model(spec, model_path, args.dry_run)
            model_ref = str(model_path)
        elif not model_path.exists() and model_ref != spec.hf_id:
            raise SystemExit(
                f"Missing model directory for {spec.key}: {model_ref}. "
                "Pass --models-root with external weights or use a Hugging Face id."
            )
        if args.run_local:
            _run(_benchmark_cmd(args, spec, Path(model_ref)), dry_run=args.dry_run, cwd=REPO_ROOT)
        else:
            _submit_sbatch(args, spec, Path(model_ref))


def _iter_result_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("results_*.json"))


def _entry(data: dict, bits: str, method: str) -> dict | None:
    methods = (data.get("results") or {}).get(str(bits), {})
    # Canonical proxy key first; legacy aliases are read-only compatibility for
    # historical bundles and are never emitted by the current benchmark.
    for candidate in hayate_result_method_candidates(method):
        if candidate in methods:
            return methods[candidate]
    return None


def _ppl(data: dict, bits: str, method: str) -> float | None:
    ent = _entry(data, bits, method)
    val = None if ent is None else ent.get("ppl")
    return None if val in (None, "") else float(val)


def _dist_k(data: dict, bits: str, method: str) -> float | None:
    ent = _entry(data, bits, method)
    val = None if ent is None else ent.get("distortion_key")
    return None if val in (None, "") else float(val)


def _paper_model_names() -> set[str]:
    names: set[str] = set()
    for spec in DEFAULT_MODELS:
        names.update({spec.key, spec.path_name, spec.display})
    return names


def _fmt(value: float | None, ndigits: int = 2) -> str:
    return "--" if value is None else f"{value:.{ndigits}f}"


def _summarize(args: argparse.Namespace) -> None:
    rows: list[str] = []
    rows.append(
        "| Model | 2b TQ | 2b HQ | 2b HQ-Frontier | 2b AdaptiveHQ | 2b RotationHQ | 3b TQ | 3b HQ | 3b HQ-Frontier | 3b AdaptiveHQ | 3b RotationHQ | 4b TQ | 4b HQ | 4b HQ-Frontier | 4b AdaptiveHQ | 4b RotationHQ | 3b dist(K) gain |"
    )
    rows.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for path in _iter_result_files(args.output_root):
        data = json.loads(path.read_text())
        model = data.get("model") or path.stem.replace("results_", "")
        if not args.include_extra_results and model not in _paper_model_names():
            continue
        tq2 = _ppl(data, "2", "TurboQuant-paper")
        hq2 = _ppl(data, "2", args.hq_method)
        frontier2 = _ppl(data, "2", "HQ-Frontier")
        ahq2 = _ppl(data, "2", args.adaptive_method)
        rot2 = _ppl(data, "2", "RotationHQ")
        tq3 = _ppl(data, "3", "TurboQuant-paper")
        hq3 = _ppl(data, "3", args.hq_method)
        frontier3 = _ppl(data, "3", "HQ-Frontier")
        ahq3 = _ppl(data, "3", args.adaptive_method)
        rot3 = _ppl(data, "3", "RotationHQ")
        tq4 = _ppl(data, "4", "TurboQuant-paper")
        hq4 = _ppl(data, "4", args.hq_method)
        frontier4 = _ppl(data, "4", "HQ-Frontier")
        ahq4 = _ppl(data, "4", args.adaptive_method)
        rot4 = _ppl(data, "4", "RotationHQ")
        tq3_d = _dist_k(data, "3", "TurboQuant-paper")
        hq3_d = _dist_k(data, "3", args.hq_method)
        gain = None if tq3_d in (None, 0) or hq3_d is None else 100.0 * (1.0 - hq3_d / tq3_d)
        rows.append(
            f"| {model} | {_fmt(tq2)} | {_fmt(hq2)} | {_fmt(frontier2)} | {_fmt(ahq2)} | {_fmt(rot2)} "
            f"| {_fmt(tq3)} | {_fmt(hq3)} | {_fmt(frontier3)} | {_fmt(ahq3)} | {_fmt(rot3)} | {_fmt(tq4)} "
            f"| {_fmt(hq4)} | {_fmt(frontier4)} | {_fmt(ahq4)} | {_fmt(rot4)} "
            f"| {_fmt(gain, 1)}% |"
        )
    rows += [
        "",
        "Only HQ, HQ-Frontier, AdaptiveHQ, and RotationHQ are emitted as HayateQuant variants; "
        "development proxies are excluded from paper summaries.",
    ]
    text = "\n".join(rows)
    print(text)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(text + "\n")
        print(f"\nWrote {args.summary_out}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="Model keys, path names, or absolute model paths.")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT, help="External model directory (optional).")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--config-name", default="paper_hayatequant")
    ap.add_argument(
        "--override", action="append", default=[], help="Extra Hydra override, e.g. ppl_windows=16"
    )
    ap.add_argument("--download-missing", action="store_true")
    ap.add_argument(
        "--submit",
        action="store_true",
        help="Submit Slurm jobs. This is the default unless --run-local/--summarize is set.",
    )
    ap.add_argument(
        "--run-local",
        action="store_true",
        help="Run sequentially in the current process allocation.",
    )
    ap.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize existing JSON results instead of launching jobs.",
    )
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument(
        "--hq-method",
        default="HQ",
        help="Fixed implementation method reported as HQ in the paper tables.",
    )
    ap.add_argument(
        "--adaptive-method",
        default="AdaptiveHQ",
        help="Implementation method reported as Adaptive HQ.",
    )
    ap.add_argument(
        "--include-extra-results",
        action="store_true",
        help="Summarize every result JSON instead of the current seven-model paper grid.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partition", default="debug")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="180G")
    ap.add_argument("--time", default="03:00:00")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.models_root = args.models_root.expanduser().resolve() if args.models_root else None
    args.output_root = resolve_output_root(args.output_root, default_relative="results/hayatequant/cross_model")
    args.python = resolve_python(args.python)
    if args.summarize:
        _summarize(args)
        return
    if not (args.submit or args.run_local):
        args.submit = True
    _launch(args)


if __name__ == "__main__":
    main()
