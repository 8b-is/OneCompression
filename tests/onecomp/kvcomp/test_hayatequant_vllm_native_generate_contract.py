"""Contracts separating canonical HQ serving from LUT development candidates."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "kvcomp-codec"
    / "hayatequant_vllm_native_generate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("native_generation_contract_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(*, assignment_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        variant="hayate_full",
        hq_assignment_mode=assignment_mode,
    )


def test_exact_full_hq_keeps_the_canonical_serving_contract():
    module = _module()
    contract = module._implementation_contract(_args(assignment_mode="exact"))
    assert contract["implementation_label"] == "native_faithful_packed_fixed_HQ"
    assert contract["canonical_hayatequant_frontier_recipe"] is True
    assert contract["eligible_for_full_hayatequant_serving_claim"] is True
    assert contract["development_encoder_candidate"] is False


def test_lut_full_hq_is_explicitly_development_only():
    module = _module()
    contract = module._implementation_contract(_args(assignment_mode="lut"))
    assert contract["implementation_label"] == "native_packed_fixed_HQ_grid_lut_candidate"
    assert contract["canonical_hayatequant_frontier_recipe"] is False
    assert contract["eligible_for_full_hayatequant_serving_claim"] is False
    assert contract["formal_quality_gate_required_before_canonical_use"] is True


def test_runtime_evidence_rejects_the_wrong_assignment_mode(tmp_path: Path):
    module = _module()
    evidence = tmp_path / "runtime.jsonl"
    base = {
        "schema": {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1},
        "backend": "vllm_plugins.kvcomp.hayate_full_native.FullHayateQuantNativeBackend",
        "implementation": "native_faithful_packed_fixed_HQ",
        "recipe": {"assignment_mode": "lut"},
    }
    rows = [
        {**base, "event": "packed_hq_store_launched"},
        {**base, "event": "packed_hq_decode_attention_launched"},
    ]
    evidence.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="assignment mode"):
        module._load_full_hq_runtime_evidence(evidence, expected_assignment_mode="exact")
