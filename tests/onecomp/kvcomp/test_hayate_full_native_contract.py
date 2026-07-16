"""Claim-boundary tests for the faithful full HQ native path."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "vllm_plugins/kvcomp/hayate_full_config.py"
KERNEL = REPO / "vllm_plugins/kvcomp/hayate_full_triton.py"
BACKEND = REPO / "vllm_plugins/kvcomp/hayate_full_native.py"
FAIR = REPO / "benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py"
SMOKE = REPO / "benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py"
GENERATE = REPO / "benchmark/kvcomp-codec/hayatequant_vllm_native_generate.py"


def _load_generate():
    spec = importlib.util.spec_from_file_location("full_native_generate_contract", GENERATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_native_is_not_the_historical_hybrid():
    backend = BACKEND.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    assert "FullHayateQuantNativeBackend" in backend
    assert "triton_full_hayatequant_store" in backend
    assert '"nc"' in kernel
    assert '"ls_iter"' in kernel
    assert "_store_quantized_value" not in kernel
    assert "TurboQuant scalar" not in kernel


def test_cache_contract_has_two_scales_and_no_affine_zero_point():
    source = CONFIG.read_text(encoding="utf-8")
    assert "K fp16 scale" in source
    assert "V fp16 scale" in source
    assert '"zero_points": 0' in source
    assert "slot_padding_bytes" in source


def test_fair_speed_admits_only_hq_and_upstream_tq_native():
    source = FAIR.read_text(encoding="utf-8")
    assert '"methods": ["HQ", "TurboQuant"]' in source
    assert "unmodified upstream vLLM 0.20" in source
    assert "fresh_process_per_sample" in source
    for excluded in ("FibQuant", "BlockQuant", "hybrid", "fake_quant", "public_payload_api"):
        assert f'"{excluded}"' in source
    assert "upstream_vllm_sha256" in source


def test_gpu_smoke_requires_payload_and_attention_parity():
    source = SMOKE.read_text(encoding="utf-8")
    assert '"key_codes_exact": True' in source
    assert '"value_codes_exact": True' in source
    assert '"key_fp16_scales_exact": True' in source
    assert '"value_fp16_scales_exact": True' in source
    assert "rotated_value_linearity_tolerance" in source
    assert 'cross_method_speed_claim_allowed": False' in source


def test_generation_harness_labels_full_hq_separately_from_hybrid():
    module = _load_generate()
    full = module._implementation_contract("hayate_full")  # pylint: disable=protected-access
    hybrid = module._implementation_contract("hayate")  # pylint: disable=protected-access
    assert full["proposal_name"] == "HQ"
    assert full["native_hybrid_is_full_hayatequant"] is True
    assert full["eligible_for_full_hayatequant_serving_claim"] is True
    assert hybrid["native_hybrid_is_full_hayatequant"] is False
    assert hybrid["eligible_for_full_hayatequant_serving_claim"] is False
