"""CPU-only tests for the preregistered packed-native formal speed matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
AGGREGATE = REPO / "benchmark" / "kvcomp-codec" / "aggregate_hayatequant_native_speed.py"
CONTRACT = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_native_speed_formal_contract.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("native_speed_aggregate_tested", AGGREGATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _cells(*, losses: set[int] = frozenset(), ties: set[int] = frozenset()):
    contract = _contract()
    rows = []
    index = 0
    for rate in contract["common_support"]["average_rates"]:
        for shape in contract["common_support"]["shape_points"]:
            for context in contract["common_support"]["contexts"]:
                for endpoint in contract["timed_endpoints"]:
                    rows.append(
                        {
                            "rate": rate,
                            "shape": shape["id"],
                            "context": context,
                            "endpoint": endpoint["metric"],
                            "HQ_strict_win": index not in losses and index not in ties,
                            "tie": index in ties,
                        }
                    )
                    index += 1
    return rows


def test_native_formal_contract_is_exactly_48_predeclared_cells():
    module = _load_module()
    contract = _contract()
    assert len(module._expected_points(contract)) == 12
    assert len(module._endpoint_names(contract)) == 4
    assert contract["matrix_gate"]["cell_count"] == 48
    assert contract["matrix_gate"]["minimum_overall_wins"] == 44
    assert contract["matrix_gate"]["minimum_wins_per_endpoint"] == 11
    assert contract["matrix_gate"]["minimum_wins_per_rate"] == 22
    assert contract["timing_protocol"]["synthetic_input_seed"] == 0
    assert contract["known_development_evidence"]["blind_preregistration_claimed"] is False


def test_native_formal_gate_accepts_48_wins_and_rejects_43_wins():
    module = _load_module()
    contract = _contract()
    passed = module._matrix_gate(_cells(), contract)
    assert passed["passed"] is True
    assert passed["HQ_strict_wins"] == 48

    failed = module._matrix_gate(_cells(losses=set(range(5))), contract)
    assert failed["HQ_strict_wins"] == 43
    assert failed["passed"] is False
    assert "does not establish" in failed["claim"]


def test_native_formal_gate_requires_every_endpoint_and_rate_subgate():
    module = _load_module()
    contract = _contract()
    # The first endpoint occurs at indices 0,4,...,44.  Two losses leave ten
    # wins for that endpoint, below its preregistered 11/12 requirement, even
    # though the overall count remains 46/48.
    failed_endpoint = module._matrix_gate(_cells(losses={0, 4}), contract)
    assert failed_endpoint["HQ_strict_wins"] == 46
    assert failed_endpoint["per_endpoint"]["prefill_store"]["wins"] == 10
    assert failed_endpoint["passed"] is False

    # Three losses in one rate leave 21/24, below the required 22/24, while
    # preserving an overall 45/48.
    failed_rate = module._matrix_gate(_cells(losses={0, 1, 2}), contract)
    assert failed_rate["HQ_strict_wins"] == 45
    assert failed_rate["per_rate"]["3"]["wins"] == 21
    assert failed_rate["passed"] is False


def test_native_formal_ties_are_losses_and_missing_cells_never_pass():
    module = _load_module()
    contract = _contract()
    tied = module._matrix_gate(_cells(ties={0}), contract)
    assert tied["ties_counted_as_losses"] == 1
    assert tied["HQ_strict_wins"] == 47

    incomplete = module._matrix_gate(_cells()[:-1], contract)
    assert incomplete["complete"] is False
    assert incomplete["passed"] is False


def test_native_aggregator_enforces_repo_retention_and_clean_source():
    source = AGGREGATE.read_text(encoding="utf-8")
    assert "result root is not inside a git worktree" in source
    assert "papers/hayatequant_paper/experiments/results/formal" in source
    assert "git_status_porcelain_at_finish" in source
    assert "spans source/upstream/software/clock strata" in source
    assert "single native cell crosses the formal claim boundary" in source
    assert "ties_counted_as_losses" in source
    assert "worker stdout/stderr/commands are incomplete" in source
    assert "controller output provenance does not match" in source


def test_native_aggregator_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    module = _load_module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    with pytest.raises(module.MatrixError, match="duplicate JSON key"):
        module._load_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": Infinity}\n', encoding="utf-8")
    with pytest.raises(module.MatrixError, match="non-finite JSON constant"):
        module._load_json(nonfinite)
