"""Fail-closed tests for external-baseline policy identity and accounting."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
HISTORICAL = (
    REPO
    / "papers/hayatequant_paper/experiments/results/development/audits"
    / "external_codec_math_audit_20260716"
)
SUPERSESSION = HISTORICAL.with_name(
    "external_codec_math_audit_20260716_policy_supersession"
)
FORMAL_DOC = REPO / "benchmark/kvcomp-codec/FORMAL_PAPER_RESULTS.md"
BASELINE_DOC = REPO / "papers/hayatequant_paper/FIBQUANT_BLOCKQUANT_DIFF.md"
PAPER_README = REPO / "papers/hayatequant_paper/README.md"
NATIVE_CONTRACT = REPO / "benchmark/kvcomp-codec/hayatequant_native_speed_formal_contract.json"
NATIVE_AGGREGATOR = REPO / "benchmark/kvcomp-codec/aggregate_hayatequant_native_speed.py"
FORMAL_AGGREGATOR = REPO / "benchmark/kvcomp-codec/aggregate_hayatequant_paper_shards.py"
FULL_HQ_CONFIG = REPO / "vllm_plugins/kvcomp/hayate_full_config.py"

SOURCE_COMMIT = "82e176a93ca23cea262bb9ce63ed727afdb5e8a7"
HISTORICAL_SHA256SUMS_SHA256 = (
    "c2c0a474360d316ff9c743df2954e73b51da04215661428bd0731d23cf236125"
)
HISTORICAL_FILES = {
    "README.md": "eaddad5e3b69e0612eb2b0b9684819d6b3ecd952481407308e913a018997c9cd",
    "audit.json": "19a23182ee426278393f1a7ee0cfabbeeecc087ced7556e067b69fcb247cb0a5",
    "commands.txt": "ae224fa6202e71a42ec9bf727512dcbce891d1ada543bc063b8132ea1f2d1c14",
    "security_scan.json": "799b3304a1cb95cf4916b56cadf4456ebf18c9fd541b2e574afb4a0aa122eaff",
    "source_manifest.json": "5826350742b92d730af50f0c62ccfe348a6647b94df528fb0afa9cfeb86e59cd",
    "test.log": "e0a5c5deb1fc403169582f6747a8bf2160c6a874a98e047a054c591369857935",
    "test_results.json": "b98e6c0a04bd9917d6eb68dfae67f3b21fb11d8f921715fe7454472176d48f49",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        result[name] = digest
    return result


def test_historical_external_codec_audit_is_byte_identical_and_separately_superseded():
    assert _sha256(HISTORICAL / "SHA256SUMS") == HISTORICAL_SHA256SUMS_SHA256
    assert _sha256_manifest(HISTORICAL / "SHA256SUMS") == HISTORICAL_FILES
    for name, expected in HISTORICAL_FILES.items():
        assert _sha256(HISTORICAL / name) == expected

    policy = _json(SUPERSESSION / "policy_supersession.json")
    assert policy["schema"] == {
        "name": "onecomp.hayatequant.external_codec_policy_supersession",
        "version": 1,
    }
    assert policy["signed_statement"]["policy_source_commit"] == SOURCE_COMMIT
    assert policy["signed_statement"]["attestation_type"] == (
        "repository_commit_bound_sha256"
    )
    assert policy["signed_statement"]["external_identity_signature_claimed"] is False
    assert policy["historical_evidence"]["files"] == HISTORICAL_FILES
    assert (
        policy["historical_evidence"]["sha256sums_file_sha256"]
        == HISTORICAL_SHA256SUMS_SHA256
    )

    historical_policy = _json(HISTORICAL / "audit.json")["formal_contract"][
        "BlockQuant-paper"
    ]
    current = policy["current_policy"]["formal_point_estimate_gate"]
    assert historical_policy["include_in_predeclared_quality_denominator"] is True
    assert current["blockquant_included_in_denominator"] is False
    assert current["blockquant_threshold"] is None
    assert current["blockquant_passed"] is None

    manifest = _sha256_manifest(SUPERSESSION / "SHA256SUMS")
    assert set(manifest) == {"POLICY_SUPERSESSION.md", "policy_supersession.json"}
    for name, expected in manifest.items():
        assert _sha256(SUPERSESSION / name) == expected


def test_live_docs_point_to_the_same_supersession_without_rewriting_history():
    directory_name = SUPERSESSION.name
    assert directory_name in FORMAL_DOC.read_text(encoding="utf-8")
    assert directory_name in BASELINE_DOC.read_text(encoding="utf-8")
    readme = PAPER_README.read_text(encoding="utf-8")
    assert directory_name in readme
    assert "vLLM 0.24.0 記録" in readme
    assert "vLLM 0.20.0" in readme
    assert "exact stored-bit matched ではない" in readme


def test_formal_gate_cardinalities_derive_from_frozen_axes_and_exclude_blockquant():
    formal = _load_module("baseline_policy_formal_aggregator", FORMAL_AGGREGATOR)
    policy = _json(SUPERSESSION / "policy_supersession.json")["current_policy"][
        "formal_point_estimate_gate"
    ]

    per_axis_matched = {
        "ppl": len(formal.PAPER_MODELS) * 3,
        "zeroshot": len(formal.PAPER_MODELS) * 3 * len(formal.ZERO_TASKS),
        "ruler": len(formal.LONG_MODELS) * len(formal.RULER_LENGTHS),
        "longbench": len(formal.LONG_MODELS) * len(formal.LONGBENCH_COUNTS),
        "math500": len(formal.LONG_MODELS),
        "longcodeqa": sum(len(brackets) for brackets in formal.LONGCODE_PLANS.values()),
    }
    per_axis_block = {
        "ppl": len(formal.PAPER_MODELS),
        "zeroshot": len(formal.PAPER_MODELS) * len(formal.ZERO_TASKS),
        "ruler": len(formal.LONG_MODELS) * len(formal.RULER_LENGTHS),
        "longbench": len(formal.LONG_MODELS) * len(formal.LONGBENCH_COUNTS),
        "math500": len(formal.LONG_MODELS),
        "longcodeqa": sum(len(brackets) for brackets in formal.LONGCODE_PLANS.values()),
    }
    matched_quality = sum(per_axis_matched.values())
    block_descriptive = sum(per_axis_block.values())
    speed = len(formal.LONG_MODELS) * 3 * len(formal.SPEED_PAIRED_TIMING_METRICS)
    total_gate = len(formal.PROPOSAL_METHODS) * (2 * matched_quality + speed)
    block_descriptive_total = len(formal.PROPOSAL_METHODS) * block_descriptive

    assert tuple(policy["proposals"]) == formal.PROPOSAL_METHODS
    assert per_axis_matched == policy["quality_axes"]
    assert per_axis_block == policy["blockquant_descriptive_quality_axes"]
    assert matched_quality == policy["matched_quality_cells_per_proposal_and_baseline"] == 164
    assert block_descriptive == policy["blockquant_descriptive_cells_per_proposal"] == 80
    assert policy["blockquant_descriptive_cells_total"] == block_descriptive_total == 320
    assert speed == policy["turboquant_speed_cells_per_proposal"] == 60
    assert total_gate == policy["total_gate_cells"] == 1552
    assert formal.QUALITY_BASELINE_METHODS == (
        "TurboQuant-paper",
        "FibQuant-paper",
        "BlockQuant-paper",
    )


def test_native_contract_separates_upstream_tq_from_algorithm1_and_discloses_slot_bytes():
    native = _load_module("baseline_policy_native_aggregator", NATIVE_AGGREGATOR)
    config = _load_module("baseline_policy_full_hq_config", FULL_HQ_CONFIG)
    formal = _load_module("baseline_policy_formal_scope", FORMAL_AGGREGATOR)
    contract = _json(NATIVE_CONTRACT)
    native._validate_comparison_scope(contract)  # pylint: disable=protected-access

    scope = contract["comparison_scope"]
    assert scope["baseline_identity"] != formal.TURBOQUANT_METHOD_SCOPE["internal_method_key"]
    assert scope["baseline_is_turboquant_paper_algorithm_1"] is False
    assert formal.TURBOQUANT_METHOD_SCOPE["implementation_scope"] == (
        "finite_beta_lloyd_max_dense_haar_l2_scalar_mse"
    )
    assert contract["common_support"]["vllm_version"] == "0.20.0"
    assert contract["common_support"]["average_rates"] == [3, 4]
    assert scope["rate_match_basis"] == "nominal_code_payload_bits_per_coordinate_only"
    assert scope["exact_logical_stored_bits_matched"] is False
    assert scope["exact_physical_slot_bytes_matched"] is False

    for rate in (3, 4):
        reported = scope["physical_packed_slot_bytes_head_dim_128"][str(rate)]
        hq_slot = config.fixed_hq_native_recipe(rate).layout(128)["slot_size"]
        upstream_tq_slot = 2 * math.ceil(128 * rate / 8) + 2 + 4
        assert reported == {
            "HQ": hq_slot,
            "upstream-vLLM-0.20-TurboQuant-native": upstream_tq_slot,
        }
        assert hq_slot < upstream_tq_slot

    tampered = copy.deepcopy(contract)
    tampered["comparison_scope"]["baseline_is_turboquant_paper_algorithm_1"] = True
    with pytest.raises(native.MatrixError, match="comparator/rate/slot scope"):
        native._validate_comparison_scope(tampered)  # pylint: disable=protected-access
