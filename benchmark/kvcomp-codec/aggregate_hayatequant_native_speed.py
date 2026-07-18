#!/usr/bin/env python3
"""Fail-closed aggregation for the preregistered packed-native HQ/TQ matrix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO / "benchmark/kvcomp-codec/hayatequant_native_speed_formal_contract.json"
RESULT_SCHEMA = {"name": "onecomp.hq_vs_upstream_tq_native_fair_speed", "version": 2}
FAILURE_SCHEMA_NAME = "onecomp.hq_vs_upstream_tq_native_fair_speed_failure"
SUMMARY_SCHEMA = {"name": "onecomp.hayatequant_native_speed_formal_summary", "version": 1}


class MatrixError(RuntimeError):
    """Raised when the matrix cannot support any formal interpretation."""


class _DuplicateKeyError(ValueError):
    pass


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise MatrixError(f"cannot load unambiguous JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"top-level JSON is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MatrixError(f"{label} is non-finite")
    return result


def _contract_shapes(contract: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {}
    for shape in contract["common_support"]["shape_points"]:
        key = (shape["query_heads"], shape["kv_heads"], shape["head_dim"])
        if key in result:
            raise MatrixError(f"formal contract repeats native shape {key}")
        config = Path(shape["model_config_path"])
        if not config.is_file() or _sha256(config) != shape["model_config_sha256"]:
            raise MatrixError(f"model-derived native shape config changed: {config}")
        result[key] = shape
    return result


def _endpoint_names(contract: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(row["metric"] for row in contract["timed_endpoints"])
    if len(names) != len(set(names)) or not names:
        raise MatrixError("formal contract has duplicate or empty endpoint names")
    return names


def _expected_points(contract: dict[str, Any]) -> set[tuple[int, str, int]]:
    return {
        (int(rate), str(shape["id"]), int(context))
        for rate in contract["common_support"]["average_rates"]
        for shape in contract["common_support"]["shape_points"]
        for context in contract["common_support"]["contexts"]
    }


def _validate_comparison_scope(contract: dict[str, Any]) -> None:
    """Reject a native contract that conflates upstream TQ with TQ-paper."""

    methods = contract.get("methods")
    if not isinstance(methods, dict) or any(
        methods.get(key) != expected
        for key, expected in {
            "candidate": "HQ",
            "candidate_implementation": "faithful_packed_fixed_HQ_native",
            "baseline": "TurboQuant",
            "baseline_display_name": "upstream-vLLM-0.20-TurboQuant-native",
            "baseline_implementation": "unmodified_upstream_vllm_0.20_TurboQuant_native",
        }.items()
    ):
        raise MatrixError("native formal method identities changed")

    expected_scope = {
        "candidate_identity": "HQ",
        "baseline_identity": "upstream-vLLM-0.20-TurboQuant-native",
        "baseline_worker_label": "TurboQuant",
        "baseline_is_turboquant_paper_algorithm_1": False,
        "baseline_algorithm_scope": (
            "upstream_hadamard_lloyd_max_keys_uniform_affine_values_native_presets"
        ),
        "rate_match_basis": "nominal_code_payload_bits_per_coordinate_only",
        "exact_logical_stored_bits_matched": False,
        "exact_physical_slot_bytes_matched": False,
        "algorithm_specific_physical_packed_slot_bytes_reported": True,
        "physical_packed_slot_bytes_head_dim_128": {
            "3": {"HQ": 100, "upstream-vLLM-0.20-TurboQuant-native": 102},
            "4": {"HQ": 132, "upstream-vLLM-0.20-TurboQuant-native": 134},
        },
    }
    if contract.get("comparison_scope") != expected_scope:
        raise MatrixError("native formal comparator/rate/slot scope changed")

    common = contract.get("common_support")
    if (
        not isinstance(common, dict)
        or common.get("vllm_version") != "0.20.0"
        or common.get("vllm_0_24_scope")
        != "historical_plugin_mapping_compatibility_smoke_only_not_in_this_matrix"
        or common.get("average_rates") != [3, 4]
    ):
        raise MatrixError("native formal vLLM version or nominal-rate scope changed")


def _expected_order(trials: int) -> list[str]:
    if trials % 4:
        raise MatrixError("formal fresh-process count is not divisible by four")
    order: list[str] = []
    for block in range(trials // 4):
        order.extend(
            ("HQ", "TurboQuant", "TurboQuant", "HQ")
            if block % 2 == 0
            else ("TurboQuant", "HQ", "HQ", "TurboQuant")
        )
    return order


def _validate_worker_stratum(document: dict[str, Any], *, path: Path) -> dict[str, Any]:
    workers = document.get("workers")
    if not isinstance(workers, list) or not workers:
        raise MatrixError(f"missing native fresh workers in {path}")
    pids = set()
    gpu_ids = set()
    software = set()
    clock = set()
    methods = Counter()
    for worker in workers:
        if not isinstance(worker, dict):
            raise MatrixError(f"malformed native worker in {path}")
        if worker.get("correctness_gate", {}).get("passed") is not True:
            raise MatrixError(f"worker lacks a passing packed correctness gate in {path}")
        for regime in ("eager", "cudagraph"):
            gate = worker["correctness_gate"].get(regime)
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise MatrixError(f"worker lacks passing {regime} correctness in {path}")
            if gate.get("all_finite") is not True or gate.get("violation_count") != 0:
                raise MatrixError(f"worker has invalid {regime} output in {path}")
        packed = worker["correctness_gate"]["eager"].get("packed_cache", {})
        if packed.get("padding_sentinel_untouched") is not True:
            raise MatrixError(f"worker modified aligned packed padding in {path}")
        method = worker.get("worker_variant")
        if method not in {"HQ", "TurboQuant"}:
            raise MatrixError(f"unexpected native worker method {method!r} in {path}")
        methods[method] += 1
        pid = worker.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid in pids:
            raise MatrixError(f"native workers did not use unique positive PIDs in {path}")
        pids.add(pid)
        gpu = worker.get("gpu", {})
        gpu_ids.add((gpu.get("name"), gpu.get("uuid"), tuple(gpu.get("capability", []))))
        runtime = worker.get("runtime", {})
        software.add(
            (
                runtime.get("python_executable_sha256"),
                runtime.get("torch_version"),
                runtime.get("cuda_runtime_version"),
                runtime.get("cudnn_version"),
                runtime.get("vllm_version"),
            )
        )
        environment = worker.get("gpu_environment", {})
        for moment in ("before_timing", "after_timing"):
            snapshot = environment.get(moment, {})
            if snapshot.get("exclusive_compute_process") is not True:
                raise MatrixError(f"worker observed a competing GPU process in {path}")
            row = snapshot.get("gpu", {})
            clock.add(
                (
                    row.get("driver_version"),
                    row.get("pstate"),
                    row.get("clocks.current.sm"),
                    row.get("clocks.current.memory"),
                    row.get("power.limit"),
                )
            )
    if len(gpu_ids) != 1 or len(software) != 1 or len(clock) != 1:
        raise MatrixError(f"GPU/software/clock stratum changed inside {path}")
    if methods != Counter({"HQ": len(workers) // 2, "TurboQuant": len(workers) // 2}):
        raise MatrixError(f"unbalanced native fresh processes in {path}: {methods}")
    return {
        "gpu": next(iter(gpu_ids)),
        "software": next(iter(software)),
        "clock": next(iter(clock)),
        "worker_count": len(workers),
    }


def _validate_result(
    path: Path,
    document: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    shapes: dict[tuple[int, int, int], dict[str, Any]],
    endpoints: tuple[str, ...],
) -> tuple[tuple[int, str, int], list[dict[str, Any]], dict[str, Any]]:
    if document.get("schema") != RESULT_SCHEMA:
        raise MatrixError(f"wrong native result schema in {path}")
    if document.get("formal_cell_eligible") is not True:
        raise MatrixError(f"native result is not a formal-eligible cell: {path}")
    if (
        document.get("formal_eligible") is not False
        or document.get("whole_matrix_claim_allowed") is not False
    ):
        raise MatrixError(f"single native cell crosses the formal claim boundary in {path}")
    if document.get("formal_cell_ineligibility_reasons") != []:
        raise MatrixError(f"native cell records ineligibility reasons in {path}")
    comparison = document.get("comparison_contract", {})
    if comparison.get("formal_matrix_contract_sha256") != contract_sha256:
        raise MatrixError(f"native result used a different formal matrix contract in {path}")
    if comparison.get("packed_output_correctness_required_before_timing") is not True:
        raise MatrixError(f"native result omitted correctness-before-timing in {path}")
    if comparison.get("methods") != ["HQ", "TurboQuant"]:
        raise MatrixError(f"native method set changed in {path}")

    provenance = document.get("provenance", {})
    if provenance.get("git_status_porcelain") not in (None, ""):
        raise MatrixError(f"native formal source was dirty at controller start in {path}")
    if provenance.get("git_status_porcelain_at_finish") not in (None, ""):
        raise MatrixError(f"native formal source worktree changed during the cell in {path}")
    arguments = provenance.get("arguments", {})
    rate = arguments.get("avg_bits")
    context = arguments.get("tokens")
    shape_key = (
        arguments.get("query_heads"),
        arguments.get("kv_heads"),
        arguments.get("head_dim"),
    )
    if shape_key not in shapes:
        raise MatrixError(f"native result has an unregistered model-derived shape in {path}")
    shape = shapes[shape_key]
    identity = (rate, shape["id"], context)
    if identity not in _expected_points(contract):
        raise MatrixError(f"native result has an unregistered matrix point {identity} in {path}")

    timing = contract["timing_protocol"]
    expected_arguments = {
        "trials": timing["fresh_processes_total_per_cell"],
        "warmups": timing["warmups_per_process"],
        "iters": timing["timed_iterations_per_process"],
        "dtype": contract["common_support"]["dtype"],
        "page_size": contract["common_support"]["page_size"],
        "max_splits": contract["common_support"]["max_kv_splits"],
        "measure_cudagraph": True,
        "recipe": "hq",
        "seed": timing["synthetic_input_seed"],
        "dist": "gaussian",
        "codebook_init": "auto",
        "rotation_mode": "auto",
    }
    for name, expected in expected_arguments.items():
        if arguments.get(name) != expected:
            raise MatrixError(f"native argument {name} changed in {path}")
    if any(
        arguments.get(name) is not None
        for name in (
            "key_code_bits",
            "key_block_size",
            "value_code_bits",
            "value_block_size",
            "assignment_mode",
            "lut_bins",
            "lut_candidates",
        )
    ):
        raise MatrixError(f"native formal cell used an explicit ablation override in {path}")
    if arguments.get("allow_canonical_k10v6_b4") is not False:
        raise MatrixError(f"native formal cell enabled a non-frozen rate-4 recipe in {path}")

    expected_order = _expected_order(timing["fresh_processes_total_per_cell"])
    if comparison.get("counterbalance_order") != expected_order:
        raise MatrixError(f"native counterbalance order changed in {path}")
    worker_log_root = Path(f"{path}.workers")
    if Path(str(comparison.get("worker_log_root"))).resolve() != worker_log_root.resolve():
        raise MatrixError(f"native worker-log root does not match its result path in {path}")
    if Path(str(arguments.get("output"))).resolve() != path.resolve():
        raise MatrixError(f"native controller output provenance does not match {path}")
    expected_log_files = set()
    for trial, method in enumerate(expected_order):
        stem = f"trial-{trial:03d}-{method.lower()}"
        expected_log_files.update(
            {
                worker_log_root / f"{stem}.stdout",
                worker_log_root / f"{stem}.stderr",
                worker_log_root / f"{stem}.command.json",
            }
        )
    actual_log_files = (
        {item for item in worker_log_root.iterdir() if item.is_file()}
        if worker_log_root.is_dir()
        else set()
    )
    if actual_log_files != expected_log_files:
        raise MatrixError(
            f"native worker stdout/stderr/commands are incomplete in {worker_log_root}"
        )
    for trial, method in enumerate(expected_order):
        stem = f"trial-{trial:03d}-{method.lower()}"
        command_record = _load_json(worker_log_root / f"{stem}.command.json")
        if (
            command_record.get("exit_status") != 0
            or command_record.get("variant") != method
            or command_record.get("trial") != trial
        ):
            raise MatrixError(f"invalid retained native worker command record {stem} in {path}")
    stratum = _validate_worker_stratum(document, path=path)
    if stratum["worker_count"] != timing["fresh_processes_total_per_cell"]:
        raise MatrixError(f"native fresh-process count changed in {path}")

    aggregate = document.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != set(endpoints):
        raise MatrixError(f"native timed endpoint set changed in {path}")
    cells = []
    for endpoint in endpoints:
        row = aggregate[endpoint]
        hq = _number(row.get("HQ_median_of_process_medians_ms"), label=f"{path} HQ {endpoint}")
        tq = _number(
            row.get("TurboQuant_median_of_process_medians_ms"),
            label=f"{path} TurboQuant {endpoint}",
        )
        speedup = _number(
            row.get("HQ_speedup_over_TurboQuant"), label=f"{path} {endpoint} speedup"
        )
        if hq <= 0 or tq <= 0 or speedup <= 0:
            raise MatrixError(f"native latency/speedup is non-positive in {path}")
        if not math.isclose(speedup, tq / hq, rel_tol=1e-12, abs_tol=0.0):
            raise MatrixError(f"recorded native speedup is arithmetically wrong in {path}")
        win = hq < tq
        if row.get("HQ_faster_point_estimate") is not win:
            raise MatrixError(f"recorded native winner is wrong in {path}")
        cells.append(
            {
                "rate": rate,
                "shape": shape["id"],
                "context": context,
                "endpoint": endpoint,
                "HQ_median_ms": hq,
                "TurboQuant_median_ms": tq,
                "HQ_speedup_over_TurboQuant": speedup,
                "HQ_strict_win": win,
                "tie": hq == tq,
                "source_result": str(path),
                "source_result_sha256": _sha256(path),
            }
        )
    return identity, cells, {"stratum": stratum, "provenance": provenance}


def _matrix_gate(cells: Iterable[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    rows = list(cells)
    gate = contract["matrix_gate"]
    expected_count = gate["cell_count"]
    endpoint_names = _endpoint_names(contract)
    rates = tuple(contract["common_support"]["average_rates"])
    endpoint_counts = defaultdict(lambda: Counter(total=0, wins=0, ties=0))
    rate_counts = defaultdict(lambda: Counter(total=0, wins=0, ties=0))
    wins = 0
    ties = 0
    for row in rows:
        won = bool(row["HQ_strict_win"])
        tied = bool(row["tie"])
        wins += int(won)
        ties += int(tied)
        endpoint_counts[row["endpoint"]].update(total=1, wins=int(won), ties=int(tied))
        rate_counts[row["rate"]].update(total=1, wins=int(won), ties=int(tied))
    complete = len(rows) == expected_count
    per_endpoint = {
        endpoint: {
            **endpoint_counts[endpoint],
            "required_total": gate["cells_per_endpoint"],
            "required_wins": gate["minimum_wins_per_endpoint"],
            "passed": endpoint_counts[endpoint]["total"] == gate["cells_per_endpoint"]
            and endpoint_counts[endpoint]["wins"] >= gate["minimum_wins_per_endpoint"],
        }
        for endpoint in endpoint_names
    }
    per_rate = {
        str(rate): {
            **rate_counts[rate],
            "required_total": gate["cells_per_rate"],
            "required_wins": gate["minimum_wins_per_rate"],
            "passed": rate_counts[rate]["total"] == gate["cells_per_rate"]
            and rate_counts[rate]["wins"] >= gate["minimum_wins_per_rate"],
        }
        for rate in rates
    }
    passed = (
        complete
        and wins >= gate["minimum_overall_wins"]
        and all(row["passed"] for row in per_endpoint.values())
        and all(row["passed"] for row in per_rate.values())
    )
    return {
        "complete": complete,
        "cell_count": len(rows),
        "required_cell_count": expected_count,
        "HQ_strict_wins": wins,
        "ties_counted_as_losses": ties,
        "losses_including_ties": len(rows) - wins,
        "win_fraction": wins / expected_count,
        "required_wins": gate["minimum_overall_wins"],
        "required_win_fraction": gate["minimum_overall_win_fraction"],
        "per_endpoint": per_endpoint,
        "per_rate": per_rate,
        "passed": passed,
        "claim": (
            "HQ establishes the preregistered packed-native speed gate over TurboQuant"
            if passed
            else "HQ does not establish the preregistered packed-native speed gate over TurboQuant"
        ),
    }


def aggregate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = _load_json(contract_path)
    if contract.get("schema") != {
        "name": "onecomp.hayatequant_native_speed_formal_contract",
        "version": 1,
    }:
        raise MatrixError("wrong native formal contract schema")
    _validate_comparison_scope(contract)
    contract_sha256 = _sha256(contract_path)
    shapes = _contract_shapes(contract)
    endpoints = _endpoint_names(contract)
    expected_points = _expected_points(contract)
    if len(expected_points) * len(endpoints) != contract["matrix_gate"]["cell_count"]:
        raise MatrixError("formal contract matrix factorization is arithmetically inconsistent")
    if not root.is_dir():
        raise MatrixError(f"native formal result root does not exist: {root}")
    try:
        result_repo = Path(
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        ).resolve()
        result_relative_root = root.resolve().relative_to(result_repo)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise MatrixError(
            f"native formal result root is not inside a git worktree: {root}"
        ) from exc
    expected_prefix = Path("papers/hayatequant_paper/experiments/results/formal")
    if not result_relative_root.is_relative_to(expected_prefix):
        raise MatrixError(
            f"native formal results are not retained below the repo formal archive: {root}"
        )

    results: dict[tuple[int, str, int], tuple[Path, dict[str, Any]]] = {}
    failures = []
    for path in sorted(root.rglob("*.json")):
        document = _load_json(path)
        schema = document.get("schema")
        if schema == RESULT_SCHEMA:
            arguments = document.get("provenance", {}).get("arguments", {})
            shape_key = (
                arguments.get("query_heads"),
                arguments.get("kv_heads"),
                arguments.get("head_dim"),
            )
            if shape_key not in shapes:
                raise MatrixError(f"unregistered shape in candidate native result {path}")
            identity = (
                arguments.get("avg_bits"),
                shapes[shape_key]["id"],
                arguments.get("tokens"),
            )
            if identity in results:
                raise MatrixError(f"duplicate native formal result for {identity}: {path}")
            results[identity] = (path, document)
        elif isinstance(schema, dict) and schema.get("name") == FAILURE_SCHEMA_NAME:
            failures.append(
                {"path": str(path), "sha256": _sha256(path), "status": document.get("status")}
            )

    missing = sorted(expected_points - set(results))
    extra = sorted(set(results) - expected_points)
    if missing or extra or failures:
        raise MatrixError(
            f"native formal matrix is incomplete or contains failures: missing={missing}, "
            f"extra={extra}, failures={failures}"
        )

    cells = []
    records = []
    commits = set()
    source_strata = set()
    upstream_strata = set()
    software_strata = set()
    hardware_models = set()
    clock_strata = set()
    for identity in sorted(results):
        path, document = results[identity]
        checked_identity, point_cells, metadata = _validate_result(
            path, document, contract, contract_sha256, shapes, endpoints
        )
        if checked_identity != identity:
            raise MatrixError(f"native result identity changed during validation: {path}")
        provenance = metadata["provenance"]
        commits.add(provenance.get("git_commit"))
        source_strata.add(json.dumps(provenance.get("source_sha256"), sort_keys=True))
        upstream_strata.add(json.dumps(provenance.get("upstream_vllm_sha256"), sort_keys=True))
        software_strata.add(metadata["stratum"]["software"])
        hardware_models.add((metadata["stratum"]["gpu"][0], metadata["stratum"]["gpu"][2]))
        clock_strata.add(metadata["stratum"]["clock"])
        cells.extend(point_cells)
        records.append(
            {
                "identity": list(identity),
                "path": str(path),
                "sha256": _sha256(path),
                "gpu": metadata["stratum"]["gpu"],
                "clock": metadata["stratum"]["clock"],
            }
        )
    if len(commits) != 1 or None in commits:
        raise MatrixError(f"native formal matrix spans source commits: {commits}")
    if (
        len(source_strata) != 1
        or len(upstream_strata) != 1
        or len(software_strata) != 1
        or len(clock_strata) != 1
    ):
        raise MatrixError("native formal matrix spans source/upstream/software/clock strata")
    if len(hardware_models) != 1:
        raise MatrixError(
            f"native formal matrix spans GPU model/capability strata: {hardware_models}"
        )

    gate = _matrix_gate(cells, contract)
    return {
        "schema": SUMMARY_SCHEMA,
        "scientifically_valid": True,
        "whole_matrix_claim_allowed": gate["passed"],
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha256,
            "known_development_evidence_seen_before_preregistration": True,
        },
        "source_commit": next(iter(commits)),
        "records": records,
        "cells": sorted(
            cells,
            key=lambda row: (row["rate"], row["shape"], row["context"], row["endpoint"]),
        ),
        "matrix_gate": gate,
        "provenance": {
            "aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "aggregator_path": str(Path(__file__).resolve()),
            "aggregator_sha256": _sha256(Path(__file__).resolve()),
            "result_root": str(root.resolve()),
        },
    }


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = aggregate(args.root, args.contract)
    except Exception as exc:
        failure = {
            "schema": SUMMARY_SCHEMA,
            "scientifically_valid": False,
            "whole_matrix_claim_allowed": False,
            "matrix_gate": {"complete": False, "passed": False},
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "provenance": {
                "aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "aggregator_path": str(Path(__file__).resolve()),
                "aggregator_sha256": _sha256(Path(__file__).resolve()),
                "result_root": str(args.root.resolve()),
                "contract_path": str(args.contract.resolve()),
                "contract_sha256": _sha256(args.contract) if args.contract.is_file() else None,
            },
        }
        _write(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        raise SystemExit(2) from exc
    _write(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
