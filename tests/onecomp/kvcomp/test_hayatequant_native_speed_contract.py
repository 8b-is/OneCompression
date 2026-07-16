"""CPU-safe label/claim boundary tests for native vLLM smoke harnesses."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
GENERATE = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_vllm_native_generate.py"
SMOKE = REPO / "benchmark" / "kvcomp-codec" / "hayatequant_vllm_native_smoke.py"


def _load_generate():
    spec = importlib.util.spec_from_file_location("native_generate_tested", GENERATE)
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
