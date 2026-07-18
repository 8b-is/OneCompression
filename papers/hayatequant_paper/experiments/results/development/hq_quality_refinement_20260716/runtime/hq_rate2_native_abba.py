"""Fresh-process ABBA speed gate for current b2 HQ versus b4 K10/V6."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


REPO = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
    "onecomp-lab-kvcomp-correctness"
)
WORKER = REPO / "benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py"
PREFIX = "KVCOMP_NATIVE_SPEED_RESULT="
CANDIDATES = {
    "DEVONLY-HQ-b4-gaussian-auto": ("gaussian", "auto"),
    "DEVONLY-HQ-b4-gaussian-fibinit": ("gaussian", "fibquant"),
    "DEVONLY-HQ-b4-finite-beta": ("beta", "auto"),
}
METRICS = (
    "prefill_store",
    "decode_attention",
    "serving_step_store1_plus_decode",
    "cudagraph_serving_step_store1_plus_decode",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(label: str, trial: int) -> list[str]:
    command = [
        sys.executable,
        str(WORKER),
        "--worker",
        "--worker-variant",
        "HQ",
        "--trial-index",
        str(trial),
        "--avg-bits",
        "2",
        "--recipe",
        "hq",
        "--head-dim",
        "128",
        "--kv-heads",
        "8",
        "--query-heads",
        "32",
        "--tokens",
        "4096",
        "--page-size",
        "16",
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--warmups",
        "5",
        "--iters",
        "20",
        "--max-splits",
        "32",
        "--assignment-mode",
        "exact",
    ]
    if label != "CURRENT-HQ-b2-K5V3":
        dist, init = CANDIDATES[label]
        command.extend(
            (
                "--key-code-bits",
                "10",
                "--key-block-size",
                "4",
                "--value-code-bits",
                "6",
                "--value-block-size",
                "4",
                "--allow-canonical-k10v6-b4",
                "--dist",
                dist,
                "--codebook-init",
                init,
            )
        )
    return command


def _run(label: str, trial: int) -> dict:
    completed = subprocess.run(
        _command(label, trial),
        cwd=REPO,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native worker {label}/{trial} failed with {completed.returncode}; "
            f"stdout={completed.stdout[-4000:]} stderr={completed.stderr[-4000:]}"
        )
    line = next(
        (row for row in reversed(completed.stdout.splitlines()) if row.startswith(PREFIX)),
        None,
    )
    if line is None:
        raise RuntimeError(f"native worker {label}/{trial} emitted no result")
    result = json.loads(line[len(PREFIX) :])
    result["development_recipe_id"] = label
    result["stderr_tail"] = completed.stderr[-1000:]
    return result


def main() -> None:
    rows = []
    trial = 0
    for candidate in CANDIDATES:
        for label in ("CURRENT-HQ-b2-K5V3", candidate, candidate, "CURRENT-HQ-b2-K5V3"):
            rows.append(_run(label, trial))
            trial += 1
    gpu_ids = {(row["gpu"]["name"], row["gpu"].get("uuid")) for row in rows}
    pids = {int(row["pid"]) for row in rows}
    if len(gpu_ids) != 1 or len(pids) != len(rows):
        raise RuntimeError("ABBA workers did not use one GPU and unique fresh processes")
    physical = {int(row["physical_cache_bytes"]) for row in rows}
    if len(physical) != 1:
        raise RuntimeError("rate-matched HQ candidates use different physical cache bytes")

    aggregates = {}
    for candidate in CANDIDATES:
        # Current rows are repeated by design; use the two current controls in
        # this candidate's own ABBA block rather than pooling across blocks.
        block_index = list(CANDIDATES).index(candidate)
        block = rows[block_index * 4 : block_index * 4 + 4]
        aggregate = {}
        for metric in METRICS:
            current = [
                row["timings"][metric]["median_ms"]
                for row in block
                if row["development_recipe_id"] == "CURRENT-HQ-b2-K5V3"
            ]
            proposed = [
                row["timings"][metric]["median_ms"]
                for row in block
                if row["development_recipe_id"] == candidate
            ]
            current_median = statistics.median(current)
            candidate_median = statistics.median(proposed)
            aggregate[metric] = {
                "current_process_medians_ms": current,
                "candidate_process_medians_ms": proposed,
                "current_median_of_process_medians_ms": current_median,
                "candidate_median_of_process_medians_ms": candidate_median,
                "candidate_speedup_over_current_hq": current_median / candidate_median,
                "candidate_faster_point_estimate": candidate_median < current_median,
            }
        aggregates[candidate] = aggregate

    result = {
        "schema": "hayatequant.development_rate2_native_abba.v1",
        "scope": "development-only native eligibility; not formal speed evidence",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_contract": {
            "order_per_candidate": ["current", "candidate", "candidate", "current"],
            "fresh_process_per_sample": True,
            "same_gpu": True,
            "same_physical_cache_bytes": True,
            "upstream_tq_2bit_comparison": "unavailable; not invented",
            "warmup_and_compilation_excluded": True,
        },
        "aggregate": aggregates,
        "workers": rows,
        "source_sha256": {
            str(path.relative_to(REPO)): _sha256(path)
            for path in (
                WORKER,
                REPO / "benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py",
                REPO / "vllm_plugins/kvcomp/hayate_full_config.py",
                REPO / "vllm_plugins/kvcomp/hayate_full_native.py",
                REPO / "vllm_plugins/kvcomp/hayate_full_triton.py",
                Path(__file__).resolve(),
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
