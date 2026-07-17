"""Render path-portable summaries from completed four-variant shards.

This renderer deliberately accepts raw codec/zero-shot/long-context JSON
shards.  It does not assume the legacy formal aggregator's two-method schema,
so new runs can produce paper tables before the full formal archive is
available.  Missing metrics are represented as null rather than silently
being treated as zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS = ("HQ", "HQ-Frontier", "AdaptiveHQ", "RotationHQ")


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _metric(payload: dict[str, Any], method: str) -> dict[str, Any] | None:
    """Find the first method row while preserving the source metric fields."""
    for node in _walk(payload):
        if node.get("method") == method or node.get("name") == method:
            return node
    methods = payload.get("methods")
    if isinstance(methods, dict) and isinstance(methods.get(method), dict):
        return methods[method]
    return None


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"summary.json", "production_models_20260717.json"}:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        model = payload.get("model") or payload.get("model_name") or path.stem
        stage = next((part for part in ("codec", "zeroshot", "longcontext") if part in path.parts), "unknown")
        methods = {method: _metric(payload, method) for method in METHODS}
        if not any(methods.values()):
            continue
        rows.append({"stage": stage, "model": model, "source": str(path), "methods": methods})
    return rows


def render(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    lines = ["# HayateQuant production results", "", "Raw shard metrics are preserved in `summary.json`.", "", "| Stage | Model | " + " | ".join(METHODS) + " |", "|---|---|" + "---:|" * len(METHODS)]
    tex = ["\\begin{tabular}{ll" + "r" * len(METHODS) + "}", "Stage & Model & " + " & ".join(METHODS) + " \\\\ ", "\\hline"]
    for row in rows:
        values = [row["methods"][method] for method in METHODS]
        def fmt(value: Any) -> str:
            if not isinstance(value, dict):
                return "—"
            for key in ("ppl", "avg_ppl", "avg_primary_score", "score", "accuracy", "tokens_per_second"):
                if value.get(key) is not None:
                    return f"{value[key]:.4g}" if isinstance(value[key], (int, float)) else str(value[key])
            return "—"
        rendered = [fmt(value) for value in values]
        lines.append("| " + str(row["stage"]) + " | " + str(row["model"]) + " | " + " | ".join(rendered) + " |")
        tex.append(str(row["stage"]) + " & " + str(row["model"]) + " & " + " & ".join(rendered) + " \\\\ ")
    tex.append("\\end{tabular}\n")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "summary.tex").write_text("\n".join(tex))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(collect(args.results_root.expanduser().resolve()), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
