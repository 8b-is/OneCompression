"""Audit and summarize the HayateQuant paper result bundle.

This script is intentionally read-only.  It complements
``submit_hayatequant_paper.sh`` by checking that an existing run root contains
the files needed to reproduce the paper tables, and by recomputing the main
seven-model PPL/zero-shot aggregates from per-model JSON files.

The paper HQ row is fixed to the implementation method
``HayateQuant-frontier`` by default.  Do not use the older summary JSON
``HayateQuant`` column for the main table; it may represent a best-of-family
selection from development sweeps rather than the fixed paper operating rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from hayatequant_cross_model import DEFAULT_MODELS


DEFAULT_RUN_ROOT = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
    "kvcomp_hayatequant/paper_final_20260709"
)

METHODS = {
    "TQ": "TurboQuant-hadamard",
    "FibQ": "FibQuant",
    "BlockQ": "BlockQuant",
}
DEFAULT_TASKS = ("arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _model_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for spec in DEFAULT_MODELS:
        for key in {spec.key, spec.path_name, spec.display}:
            names[key] = spec.display
    return names


def _find_json_by_model(root: Path, pattern: str) -> dict[str, Path]:
    aliases = _model_names()
    out: dict[str, Path] = {}
    for path in sorted(root.rglob(pattern)):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        model = payload.get("model") or path.stem
        display = aliases.get(model)
        if display:
            out[display] = path
    return out


def _ppl(payload: dict[str, Any], bits: int, method: str) -> float | None:
    val = (((payload.get("results") or {}).get(str(bits)) or {}).get(method) or {}).get("ppl")
    return None if val is None else float(val)


def _acc(payload: dict[str, Any], bits: int, method: str) -> float | None:
    val = (((payload.get("results") or {}).get(str(bits)) or {}).get(method) or {}).get("avg_acc_norm")
    return None if val is None else float(val) * 100.0


def _geomean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _main_grid(run_root: Path, hq_method: str) -> dict[str, Any]:
    ppl_root = run_root / "ppl"
    if (ppl_root / "results_fibblock_8model_v1").exists():
        ppl_root = ppl_root / "results_fibblock_8model_v1"
    zs_root = run_root / "zeroshot"
    ppl_files = _find_json_by_model(ppl_root, "results_*.json")
    zs_files = _find_json_by_model(zs_root, "zeroshot_*.json")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    aggregates: dict[str, dict[str, float | None]] = {}
    for spec in DEFAULT_MODELS:
        model = spec.display
        ppath = ppl_files.get(model)
        zpath = zs_files.get(model)
        if not ppath:
            missing.append(f"PPL:{model}")
            continue
        if not zpath:
            missing.append(f"zero-shot:{model}")
            continue
        pp = _load_json(ppath)
        zz = _load_json(zpath)
        for bits in (2, 3, 4):
            row = {"model": model, "bits": bits}
            for label, method in {**METHODS, "HQ": hq_method}.items():
                row[f"{label}_ppl"] = _ppl(pp, bits, method)
                row[f"{label}_acc"] = _acc(zz, bits, method)
                if row[f"{label}_ppl"] is None:
                    missing.append(f"PPL:{model}:{bits}:{method}")
                if row[f"{label}_acc"] is None:
                    missing.append(f"zero-shot:{model}:{bits}:{method}")
            rows.append(row)

    for bits in (2, 3, 4):
        aggregates[str(bits)] = {}
        bit_rows = [r for r in rows if r["bits"] == bits]
        for label in (*METHODS.keys(), "HQ"):
            pvals = [r[f"{label}_ppl"] for r in bit_rows if r[f"{label}_ppl"] is not None]
            avals = [r[f"{label}_acc"] for r in bit_rows if r[f"{label}_acc"] is not None]
            aggregates[str(bits)][f"{label}_ppl_geomean"] = _geomean(pvals)
            aggregates[str(bits)][f"{label}_acc_mean"] = mean(avals) if avals else None

    return {
        "hq_method": hq_method,
        "models": [m.display for m in DEFAULT_MODELS],
        "rows": rows,
        "aggregates": aggregates,
        "missing": sorted(set(missing)),
        "source_roots": {"ppl": str(ppl_root), "zero_shot": str(zs_root)},
    }


def _file_status(run_root: Path) -> dict[str, Any]:
    expected = {
        "speed_e2e_frontier": run_root / "frontier_tuning/frontier_e2e_speed_triton_gpu11.json",
        "speed_codec_frontier": run_root / "frontier_tuning/frontier_codec_speed_triton_gpu11.json",
        "native_smoke": run_root / "native/smoke.json",
        "long_llama31": run_root / "long/llama31-8b/results_llama-3.1-8b-instruct.json",
        "long_qwen3": run_root / "long/qwen3-8b/results_Qwen3-8B.json",
        "reason_r1_llama": run_root / "reason/r1-llama-8b/results_DeepSeek-R1-Distill-Llama-8B.json",
        "reason_r1_qwen7": run_root / "reason/r1-qwen-7b/results_DeepSeek-R1-Distill-Qwen-7B.json",
        "reason_r1_qwen15": run_root / "reason/r1-qwen-1.5b/results_DeepSeek-R1-Distill-Qwen-1.5B.json",
    }
    return {name: {"path": str(path), "exists": path.exists()} for name, path in expected.items()}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HayateQuant paper reproduction audit",
        "",
        f"Run root: `{report['run_root']}`",
        f"HQ method: `{report['main_grid']['hq_method']}`",
        "",
        "## Main Grid Aggregates",
        "",
        "| bits | TQ PPL | FibQ PPL | BlockQ PPL | HQ PPL | TQ ACC | FibQ ACC | BlockQ ACC | HQ ACC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bits, vals in report["main_grid"]["aggregates"].items():
        def f(key: str) -> str:
            v = vals.get(key)
            return "--" if v is None else f"{v:.2f}"
        lines.append(
            f"| {bits} | {f('TQ_ppl_geomean')} | {f('FibQ_ppl_geomean')} | "
            f"{f('BlockQ_ppl_geomean')} | {f('HQ_ppl_geomean')} | "
            f"{f('TQ_acc_mean')} | {f('FibQ_acc_mean')} | "
            f"{f('BlockQ_acc_mean')} | {f('HQ_acc_mean')} |"
        )
    missing = report["main_grid"]["missing"]
    lines += ["", "## Coverage", ""]
    for name, info in report["files"].items():
        mark = "OK" if info["exists"] else "MISSING"
        lines.append(f"- {mark}: {name}: `{info['path']}`")
    lines += ["", "## Missing Main-Grid Entries", ""]
    lines.extend(f"- {item}" for item in missing) if missing else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--hq-method", default="HayateQuant-frontier")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if required files or main entries are missing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "run_root": str(args.run_root),
        "main_grid": _main_grid(args.run_root, args.hq_method),
        "files": _file_status(args.run_root),
        "tasks": list(DEFAULT_TASKS),
    }
    text = _markdown(report)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text)
    missing_files = [k for k, v in report["files"].items() if not v["exists"]]
    missing_entries = report["main_grid"]["missing"]
    if args.strict and (missing_files or missing_entries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
