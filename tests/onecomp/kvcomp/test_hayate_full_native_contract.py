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
    assert "0xA5" in source
    assert 'cache[..., layout["slot_size"] :]' in source
    assert 'cross_method_speed_claim_allowed": False' in source


def test_b2_two_byte_decode_is_guarded_for_every_supported_shift():
    # A b=2 stream code begins at any residue in 0..7.  At <=9 code bits the
    # inclusive terminal bit is at most 15, so byte0+byte1 is sufficient; the
    # the first width with a shift-7 case extending past bit15 is 11 and must
    # use generic three-byte extraction.  The runtime guard is deliberately
    # conservative at the canonical fixed-recipe maximum of 9.
    for code_bits in range(1, 10):
        shifts = {(block * code_bits) % 8 for block in range(16)}
        assert max(shift + code_bits for shift in shifts) <= 16
    shifts_11 = {(block * 11) % 8 for block in range(16)}
    assert max(shift + 11 for shift in shifts_11) > 16

    source = KERNEL.read_text(encoding="utf-8")
    b2 = source.split("def _full_hq_decode_stage1_b2", 1)[1].split(
        "def _full_hq_decode_stage1", 1
    )[0]
    assert "kr2" not in b2
    assert "vr2" not in b2
    assert "recipe.key_code_bits_per_block <= 9" in source
    assert "recipe.value_code_bits_per_block <= 9" in source


def test_decode_launch_tuning_is_scoped_to_the_b2_fast_path():
    source = KERNEL.read_text(encoding="utf-8")
    assert "use_b2_fast_path = (" in source
    assert "stage1 = _full_hq_decode_stage1_b2 if use_b2_fast_path else" in source
    assert "block_kv = 8 if use_b2_fast_path else 16" in source
    assert "stage1_num_warps = 2 if use_b2_fast_path else 4" in source
    assert "num_warps=stage1_num_warps" in source


def test_generation_harness_labels_full_hq_separately_from_hybrid():
    module = _load_generate()
    full = module._implementation_contract("hayate_full")  # pylint: disable=protected-access
    hybrid = module._implementation_contract("hayate")  # pylint: disable=protected-access
    assert full["proposal_name"] == "HQ"
    assert full["native_hybrid_is_full_hayatequant"] is True
    assert full["eligible_for_full_hayatequant_serving_claim"] is True
    assert hybrid["native_hybrid_is_full_hayatequant"] is False
    assert hybrid["eligible_for_full_hayatequant_serving_claim"] is False
