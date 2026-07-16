"""Validate and aggregate the development-only one-window PPL matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "llama31": "llama-3.1-8b-instruct",
    "llama2": "Llama-2-7b-hf",
    "qwen34": "Qwen3-4B",
    "qwen38": "Qwen3-8B",
    "r1llama8": "DeepSeek-R1-Distill-Llama-8B",
    "r1qwen15": "DeepSeek-R1-Distill-Qwen-1.5B",
    "r1qwen7": "DeepSeek-R1-Distill-Qwen-7B",
}
PROPOSALS = ("HQ", "AdaptiveHQ")
BASELINES = ("TurboQuant-paper", "FibQuant-paper", "BlockQuant-paper")
RATES = (2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return number


def main() -> None:
    documents = {}
    raw_files = []
    for key, model in EXPECTED.items():
        path = ROOT / "raw" / key / f"results_{model}.json"
        if not path.is_file():
            raise RuntimeError(f"missing expected result {path.relative_to(ROOT)}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("model") != model:
            raise RuntimeError(f"model mismatch in {path}: {doc.get('model')!r}")
        if doc.get("skip_layers") != []:
            raise RuntimeError(f"development comparison unexpectedly skipped layers in {path}")
        config = doc.get("config", {})
        expected_config = {
            "bits": [2, 3, 4],
            "sink_tokens": 0,
            "recent_tokens": 0,
            "ppl_max_length": 512,
            "ppl_stride": 512,
            "ppl_windows": 1,
            "run_ppl": True,
            "skip_outliers": False,
        }
        for name, expected in expected_config.items():
            if config.get(name) != expected:
                raise RuntimeError(
                    f"config mismatch for {model}/{name}: {config.get(name)!r} != {expected!r}"
                )
        documents[key] = doc
        raw_files.append(path)

    cells = []
    counts = {
        proposal: {
            baseline: {
                "wins": 0,
                "total": 0,
                "by_rate": {str(rate): {"wins": 0, "total": 0} for rate in RATES},
            }
            for baseline in BASELINES
        }
        for proposal in PROPOSALS
    }
    for key, model in EXPECTED.items():
        doc = documents[key]
        full = doc.get("fullkv_ppl") or {}
        full_tokens = int(full.get("ppl_n_tokens", -1))
        if full_tokens <= 0 or int(full.get("ppl_n_windows", -1)) != 1:
            raise RuntimeError(f"invalid FullKV scoring denominator for {model}")
        if full.get("ppl_complete_text_eval") is not False:
            raise RuntimeError(f"{model} must be explicitly marked as a capped dev subset")
        source_identity = (
            full.get("ppl_source"),
            full.get("ppl_dataset_revision"),
            full.get("ppl_text_sha256"),
        )
        for rate in RATES:
            methods = doc.get("results", {}).get(str(rate), {})
            expected_methods = set(PROPOSALS) | {"TurboQuant-paper", "FibQuant-paper"}
            if rate == 3:
                expected_methods.add("BlockQuant-paper")
            if set(methods) != expected_methods:
                raise RuntimeError(f"method set mismatch for {model}/{rate}: {sorted(methods)}")
            fixed_rate_methods = (
                "TurboQuant-paper",
                "FibQuant-paper",
                "HQ",
                "AdaptiveHQ",
            )
            fp16_counters = {
                int(methods[name].get("ppl_compression_fp16_bits", -1))
                for name in fixed_rate_methods
            }
            stored_counters = {
                int(methods[name].get("ppl_compression_stored_bits", -1))
                for name in fixed_rate_methods
            }
            if len(fp16_counters) != 1 or min(fp16_counters) <= 0:
                raise RuntimeError(f"fp16 counter mismatch for {model}/{rate}")
            if len(stored_counters) != 1 or min(stored_counters) <= 0:
                raise RuntimeError(f"stored-bit counter mismatch for {model}/{rate}")
            for proposal in PROPOSALS:
                if methods[proposal].get("codec_config", {}).get("encoder") != "exact":
                    raise RuntimeError(f"{model}/{rate}/{proposal} is not canonical exact HQ")
            for name, entry in methods.items():
                if int(entry.get("ppl_n_tokens", -1)) != full_tokens:
                    raise RuntimeError(f"token denominator mismatch for {model}/{rate}/{name}")
                if int(entry.get("ppl_n_windows", -1)) != 1:
                    raise RuntimeError(f"window denominator mismatch for {model}/{rate}/{name}")
                identity = (
                    entry.get("ppl_source"),
                    entry.get("ppl_dataset_revision"),
                    entry.get("ppl_text_sha256"),
                )
                if identity != source_identity:
                    raise RuntimeError(f"dataset identity mismatch for {model}/{rate}/{name}")
                if int(entry.get("ppl_n_skipped_nonfinite", -1)) != 0:
                    raise RuntimeError(f"non-finite PPL window for {model}/{rate}/{name}")
                _finite_number(entry.get("ppl_nll"), f"{model}/{rate}/{name}/nll")
                protocol = entry.get("ppl_teacher_forced_scoring_protocol")
                expected_protocol = (
                    "causal_prefix_prefill_then_token_step_targets"
                    if name in {"AdaptiveHQ", "BlockQuant-paper"}
                    else "single_forward_causal_mask_token_local_cache"
                )
                if protocol != expected_protocol:
                    raise RuntimeError(
                        f"teacher-forced protocol mismatch for {model}/{rate}/{name}: "
                        f"{protocol!r}"
                    )

            for proposal in PROPOSALS:
                proposal_entry = methods[proposal]
                proposal_nll = _finite_number(
                    proposal_entry["ppl_nll"], f"{model}/{rate}/{proposal}/nll"
                )
                for baseline in BASELINES:
                    if baseline not in methods:
                        continue
                    baseline_entry = methods[baseline]
                    baseline_nll = _finite_number(
                        baseline_entry["ppl_nll"], f"{model}/{rate}/{baseline}/nll"
                    )
                    win = proposal_nll < baseline_nll
                    counts[proposal][baseline]["total"] += 1
                    counts[proposal][baseline]["wins"] += int(win)
                    rate_count = counts[proposal][baseline]["by_rate"][str(rate)]
                    rate_count["total"] += 1
                    rate_count["wins"] += int(win)
                    cells.append(
                        {
                            "model": model,
                            "rate": rate,
                            "proposal": proposal,
                            "baseline": baseline,
                            "proposal_ppl": proposal_entry["ppl"],
                            "baseline_ppl": baseline_entry["ppl"],
                            "proposal_nll": proposal_nll,
                            "baseline_nll": baseline_nll,
                            "nll_difference_proposal_minus_baseline": (
                                proposal_nll - baseline_nll
                            ),
                            "ppl_ratio_proposal_over_baseline": math.exp(
                                proposal_nll - baseline_nll
                            ),
                            "strict_unrounded_nll_win": win,
                            "n_tokens": full_tokens,
                            "proposal_stored_bits": proposal_entry.get(
                                "ppl_compression_stored_bits"
                            ),
                            "baseline_stored_bits": baseline_entry.get(
                                "ppl_compression_stored_bits"
                            ),
                            "fp16_bits": proposal_entry.get("ppl_compression_fp16_bits"),
                        }
                    )

    summaries = {}
    for proposal in PROPOSALS:
        summaries[proposal] = {}
        aggregate_wins = 0
        aggregate_total = 0
        aggregate_by_rate = {str(rate): {"wins": 0, "total": 0} for rate in RATES}
        for baseline in BASELINES:
            item = counts[proposal][baseline]
            item["strict_unrounded_nll_win_rate"] = item["wins"] / item["total"]
            for rate_item in item["by_rate"].values():
                rate_item["strict_unrounded_nll_win_rate"] = (
                    rate_item["wins"] / rate_item["total"] if rate_item["total"] else None
                )
            summaries[proposal][baseline] = item
            aggregate_wins += item["wins"]
            aggregate_total += item["total"]
            for rate, rate_item in item["by_rate"].items():
                aggregate_by_rate[rate]["wins"] += rate_item["wins"]
                aggregate_by_rate[rate]["total"] += rate_item["total"]
        for rate_item in aggregate_by_rate.values():
            rate_item["strict_unrounded_nll_win_rate"] = rate_item["wins"] / rate_item["total"]
        summaries[proposal]["all_external_quality_cells"] = {
            "wins": aggregate_wins,
            "total": aggregate_total,
            "strict_unrounded_nll_win_rate": aggregate_wins / aggregate_total,
            "by_rate": aggregate_by_rate,
        }

    analysis = {
        "scope": "development-only; one capped WikiText-2 window; not a formal-paper result",
        "comparison_rule": (
            "strict proposal_nll < baseline_nll using JSON double precision; ties, missing, "
            "and non-finite cells are not wins (missing/non-finite cells fail validation)"
        ),
        "models": list(EXPECTED.values()),
        "rates": list(RATES),
        "proposals": list(PROPOSALS),
        "baselines": list(BASELINES),
        "summaries": summaries,
        "cells": cells,
        "raw_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sorted(raw_files)
        },
    }
    (ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (ROOT / "ppl_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    for proposal, baseline_rows in summaries.items():
        print(proposal)
        for baseline, item in baseline_rows.items():
            print(
                f"  {baseline}: {item['wins']}/{item['total']} "
                f"({item['strict_unrounded_nll_win_rate']:.6%})"
            )
            print(
                "    by_rate: "
                + ", ".join(
                    f"r{rate}={rate_item['wins']}/{rate_item['total']}"
                    for rate, rate_item in item["by_rate"].items()
                )
            )


if __name__ == "__main__":
    main()
