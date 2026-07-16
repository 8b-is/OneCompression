#!/usr/bin/env python3
"""Fail-closed summary for the development-only fixed-HQ recipe sweep.

This script never selects a recipe per model, rate, or task.  It requires the
complete seven-checkpoint x three-rate panel and reports each globally named
candidate over all 21 cells.  Formal-paper runners reject every ``HQ-dev-*``
label, so this report cannot be mistaken for formal evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


MODELS = {
    "Llama-2-7b-hf": "llama2",
    "llama-3.1-8b-instruct": "llama31",
    "Qwen3-4B": "qwen34",
    "Qwen3-8B": "qwen38",
    "DeepSeek-R1-Distill-Llama-8B": "r1llama8",
    "DeepSeek-R1-Distill-Qwen-7B": "r1qwen7",
    "DeepSeek-R1-Distill-Qwen-1.5B": "r1qwen15",
}
RATES = ("2", "3", "4")
BASELINE = "TurboQuant-paper"
CANDIDATES = (
    "HQ-dev-paper-NC-LSiter-Gauss",
    "HQ-dev-fixed-delta0-Gauss-NC-LSiter",
    "HQ-dev-fixed-delta05-Gauss-NC-LSiter",
    "HQ-dev-fixed-delta05-Beta-NC-LSiter",
    "HQ-dev-fixed-delta05-Gauss-NC-LS",
    "HQ-dev-fixed-delta05-b2-Gauss-NC-LSiter",
    "HQ-dev-fixed-delta05-b2-Beta-NC-LSiter",
    "HQ-dev-fixed-delta1-Beta-NC-LSiter",
)
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"


class IncompleteSweep(RuntimeError):
    """The panel is missing, malformed, or not scientifically matched."""


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompleteSweep(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IncompleteSweep(f"result root is not an object: {path}")
    return value


def _finite(row: dict, key: str, label: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IncompleteSweep(f"{label}.{key} is not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise IncompleteSweep(f"{label}.{key} is non-finite")
    return value


def _result_file(root: Path, key: str) -> Path:
    directory = root / f"hq_fixed_recipe_v2_{key}_20260716"
    matches = sorted(directory.glob("results_*.json"))
    if len(matches) != 1:
        raise IncompleteSweep(
            f"expected one result below {directory}, found {len(matches)}"
        )
    return matches[0]


def _matched_rows(root: Path) -> dict[tuple[str, str], dict[str, dict]]:
    panel: dict[tuple[str, str], dict[str, dict]] = {}
    expected_methods = {BASELINE, *CANDIDATES}
    for model, key in MODELS.items():
        path = _result_file(root, key)
        document = _load(path)
        if document.get("model") != model:
            raise IncompleteSweep(
                f"model identity mismatch in {path}: {document.get('model')!r}"
            )
        results = document.get("results")
        if not isinstance(results, dict) or set(results) != set(RATES):
            raise IncompleteSweep(f"rate panel mismatch in {path}")
        for rate in RATES:
            methods = results[rate]
            if not isinstance(methods, dict) or not expected_methods.issubset(methods):
                missing = sorted(expected_methods - set(methods or {}))
                raise IncompleteSweep(f"missing methods {missing} in {path}, rate {rate}")
            baseline = methods[BASELINE]
            baseline_bits = int(baseline.get("ppl_compression_stored_bits", -1))
            baseline_fp16 = int(baseline.get("ppl_compression_fp16_bits", -1))
            baseline_tokens = int(baseline.get("ppl_n_tokens", -1))
            if baseline_bits <= 0 or baseline_fp16 <= 0 or baseline_tokens <= 0:
                raise IncompleteSweep(f"invalid baseline accounting in {path}, rate {rate}")
            for method in expected_methods:
                row = methods[method]
                label = f"{model}/{rate}/{method}"
                _finite(row, "ppl", label)
                _finite(row, "ppl_nll_sum", label)
                if row.get("ppl_dataset_revision") != WIKITEXT_REVISION:
                    raise IncompleteSweep(f"dataset revision drift in {label}")
                if row.get("ppl_text_sha256") != WIKITEXT_TEXT_SHA256:
                    raise IncompleteSweep(f"dataset text drift in {label}")
                if int(row.get("ppl_n_tokens", -1)) != baseline_tokens:
                    raise IncompleteSweep(f"token denominator mismatch in {label}")
                if int(row.get("ppl_compression_fp16_bits", -1)) != baseline_fp16:
                    raise IncompleteSweep(f"fp16 reference-bit mismatch in {label}")
                if int(row.get("ppl_compression_stored_bits", -1)) != baseline_bits:
                    raise IncompleteSweep(f"logical stored-bit mismatch in {label}")
            panel[(model, rate)] = methods
    if len(panel) != len(MODELS) * len(RATES):
        raise IncompleteSweep("complete 21-cell panel was not constructed")
    return panel


def analyze(root: Path) -> dict:
    panel = _matched_rows(root)
    summaries = []
    for method in CANDIDATES:
        ratios: list[float] = []
        encode_speedups: list[float] = []
        rate_wins = {rate: 0 for rate in RATES}
        wins = ties = losses = 0
        cells = []
        for (model, rate), methods in panel.items():
            baseline = methods[BASELINE]
            candidate = methods[method]
            tq_ppl = _finite(baseline, "ppl", f"{model}/{rate}/{BASELINE}")
            hq_ppl = _finite(candidate, "ppl", f"{model}/{rate}/{method}")
            ratio = hq_ppl / tq_ppl
            ratios.append(ratio)
            if hq_ppl < tq_ppl:
                wins += 1
                rate_wins[rate] += 1
                outcome = "win"
            elif hq_ppl == tq_ppl:
                ties += 1
                outcome = "tie"
            else:
                losses += 1
                outcome = "loss"
            tq_ms = baseline.get("ms_per_encode")
            hq_ms = candidate.get("ms_per_encode")
            speedup = None
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0
                for value in (tq_ms, hq_ms)
            ):
                speedup = float(tq_ms) / float(hq_ms)
                encode_speedups.append(speedup)
            cells.append(
                {
                    "model": model,
                    "rate": int(rate),
                    "tq_ppl": tq_ppl,
                    "candidate_ppl": hq_ppl,
                    "ppl_ratio_candidate_over_tq": ratio,
                    "outcome": outcome,
                    "synthetic_public_encode_speedup_x": speedup,
                }
            )
        summaries.append(
            {
                "method": method,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_fraction": wins / len(ratios),
                "wins_by_rate": {rate: rate_wins[rate] for rate in RATES},
                "geometric_mean_ppl_ratio_candidate_over_tq": math.exp(
                    statistics.fmean(math.log(value) for value in ratios)
                ),
                "worst_ppl_ratio_candidate_over_tq": max(ratios),
                "median_synthetic_public_encode_speedup_x": (
                    statistics.median(encode_speedups) if encode_speedups else None
                ),
                "synthetic_public_encode_speed_win_fraction": (
                    sum(value > 1.0 for value in encode_speedups) / len(encode_speedups)
                    if encode_speedups
                    else None
                ),
                "cells": cells,
            }
        )
    summaries.sort(
        key=lambda row: (
            -row["wins"],
            row["geometric_mean_ppl_ratio_candidate_over_tq"],
            row["worst_ppl_ratio_candidate_over_tq"],
        )
    )
    return {
        "schema": "hayatequant.development_fixed_recipe_summary.v1",
        "formal_evidence": False,
        "selection_scope": "one global recipe; no per-model/rate/task selection",
        "models": list(MODELS),
        "rates": [int(rate) for rate in RATES],
        "n_cells": len(panel),
        "baseline": BASELINE,
        "same_logical_stored_bits_verified": True,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
            "hayate_iclr2027_refresh_20260715/results/dev"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(analyze(args.root.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
