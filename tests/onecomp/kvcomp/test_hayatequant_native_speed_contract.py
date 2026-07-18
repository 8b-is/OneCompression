"""CPU-safe label/claim boundary tests for native vLLM smoke harnesses."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
GENERATE = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_vllm_native_generate.py"
SMOKE = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_vllm_native_smoke.py"
FAIR = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_vllm_full_native_fair_speed.py"


def _load_generate():
    spec = importlib.util.spec_from_file_location("native_generate_tested", GENERATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fair():
    spec = importlib.util.spec_from_file_location("full_native_fair_speed_tested", FAIR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_hayate_variant_is_explicitly_a_hybrid_not_full_hq():
    contract = _load_generate()._implementation_contract(
        "hayate"
    )  # pylint: disable=protected-access
    assert contract["implementation_label"] == (
        "native_hybrid_hayate_block_vq_keys_turboquant_scalar_values"
    )
    assert contract["key_path"] == "HayateQuant block-VQ"
    assert contract["value_path"] == "TurboQuant scalar"
    assert contract["native_hybrid_is_full_hayatequant"] is False
    assert contract["canonical_hayatequant_frontier_recipe"] is False
    assert contract["eligible_for_full_hayatequant_serving_claim"] is False


def test_kernel_smoke_source_forbids_quality_and_cross_case_speed_claims():
    source = SMOKE.read_text(encoding="utf-8")
    assert "onecomp.hayatequant_vllm_native_hybrid_smoke" in source
    assert '"native_hybrid_is_full_hayatequant": False' in source
    assert '"eligible_for_full_hayatequant_serving_claim": False' in source
    assert '"cross_case_timing_claims_allowed": False' in source
    assert "torch.cuda.synchronize()" in source


def test_full_native_speed_requires_independent_packed_output_correctness():
    source = FAIR.read_text(encoding="utf-8")
    assert "triton_full_hayatequant_dequant_kv" in source
    assert "_tq_full_dequant_kv" in source
    assert '"packed_output_correctness_required_before_timing": True' in source
    assert '"whole_matrix_claim_allowed": False' in source
    assert '"formal_eligible": False' in source
    assert "single_cell_requires_complete_matrix_aggregation" in source
    assert "padding_sentinel_untouched" in source
    assert "dense_packed_key_logits" in source
    assert "fresh_process_packed_bytes_and_outputs_deterministic" in source


def test_full_native_correctness_metric_is_elementwise_and_fail_closed():
    module = _load_fair()
    reference = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    native = reference.clone()
    result = module._correctness_metrics(native, reference, rtol=0.0, atol=0.0)
    assert result["passed"] is True
    assert result["violation_count"] == 0
    assert result["native_output"]["sha256"] == result["dense_reference_output"]["sha256"]

    with pytest.raises(AssertionError, match="independently dequantized"):
        module._correctness_metrics(torch.tensor([[1.0, -1.9]]), reference, rtol=0.0, atol=0.01)
    with pytest.raises(AssertionError, match="finite=False"):
        module._correctness_metrics(
            torch.tensor([[float("nan"), -2.0]]), reference, rtol=1.0, atol=1.0
        )


def test_dense_gqa_reference_maps_query_groups_to_the_same_kv_head():
    module = _load_fair()
    query = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])
    keys = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]]]])
    values = torch.tensor([[[[10.0], [20.0]], [[30.0], [40.0]]]])
    output, logits = module._dense_gqa_attention(
        torch, query, keys, values, scale=1.0, output_dtype=torch.float32
    )
    assert logits.shape == (1, 4, 2)
    assert torch.equal(logits[0, 0], logits[0, 1])
    assert torch.equal(logits[0, 2], logits[0, 3])
    assert torch.equal(output[0, 0], output[0, 1])
    assert torch.equal(output[0, 2], output[0, 3])


def test_full_native_worker_failures_are_retained_when_output_is_requested():
    source = FAIR.read_text(encoding="utf-8")
    assert "worker_log_root" in source
    assert 'Path(f"{args.output}.failed.json")' in source
    assert 'f"{stem}.stdout"' in source
    assert 'f"{stem}.stderr"' in source
    assert '"exit_status": completed.returncode' in source


def test_full_native_result_json_rejects_nonfinite_values(tmp_path):
    module = _load_fair()
    with pytest.raises(ValueError, match="Out of range float values"):
        module._write_json(tmp_path / "bad.json", {"value": float("nan")})
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        module._load_worker_result('{"value": NaN}')
    assert module._load_worker_result('{"value": 1.0}') == {"value": 1.0}


def test_full_native_formal_cell_requires_retained_output_and_clean_finish():
    source = FAIR.read_text(encoding="utf-8")
    assert 'ineligible.append("output_and_worker_logs_not_retained")' in source
    assert 'ineligible.append("formal_output_path_must_be_absolute")' in source
    assert 'ineligible.append("source_worktree_changed_during_run")' in source
    assert '"worker_log_root":' in source
