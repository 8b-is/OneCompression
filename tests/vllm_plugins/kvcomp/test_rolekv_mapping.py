"""Unit tests for the rolekv policy -> vLLM engine kwargs mapping.

Pure-Python: neither vllm, torch nor a GPU is required (the mapping module
is deliberately torch-free so it can run on a CPU-only login node).

Copyright 2025-2026 Fujitsu Ltd.
"""

from vllm_plugins.kvcomp.mapping import plan_from_kv_config


def _cfg(policy, codec=None, selector=None):
    return {"codec": codec, "selector": selector, "policy": policy, "residual_window": 128}


def _default_rolekv():
    return {
        "name": "rolekv",
        "bits_per_role": {"system": 8, "user": 8, "assistant": 4, "reasoning": 4, "tool": 2},
        "default_bits": 4,
        "skip_layers": [],
    }


def test_default_rolekv_maps_to_fp8_with_online_scales():
    """The default policy's most precise role is 8-bit -> fp8_e4m3 + scales."""
    plan = plan_from_kv_config(_cfg(_default_rolekv()))
    assert plan.native_supported
    assert plan.kv_cache_dtype == "fp8_e4m3"
    assert plan.engine_kwargs() == {
        "kv_cache_dtype": "fp8_e4m3",
        "calculate_kv_scales": True,
    }


def test_mixed_rolekv_warns_about_collapsed_roles():
    plan = plan_from_kv_config(_cfg(_default_rolekv()))
    assert any("collapsed" in w for w in plan.warnings)


def test_uniform_rolekv_has_no_collapse_warning():
    policy = {
        "name": "rolekv",
        "bits_per_role": {r: 8 for r in ("system", "user", "assistant", "reasoning", "tool")},
        "default_bits": 8,
        "skip_layers": [],
    }
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.kv_cache_dtype == "fp8_e4m3"
    assert not any("collapsed" in w for w in plan.warnings)


def test_low_bit_only_rolekv_maps_to_fp8_with_fidelity_warning():
    policy = {
        "name": "rolekv",
        "bits_per_role": {"system": 4, "tool": 2},
        "default_bits": 2,
        "skip_layers": [],
    }
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.kv_cache_dtype == "fp8_e4m3"
    assert any("below vLLM's lowest native KV cache dtype" in w for w in plan.warnings)


def test_rolekv_propagates_skip_layers():
    policy = _default_rolekv()
    policy["skip_layers"] = [0, 27]
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.skip_layers == [0, 27]
    assert plan.engine_kwargs()["kv_cache_dtype_skip_layers"] == "0,27"


def test_rolekv_conflicting_requested_dtype_warns():
    plan = plan_from_kv_config(_cfg(_default_rolekv()), requested_kv_cache_dtype="fp8_e5m2")
    assert plan.kv_cache_dtype == "fp8_e4m3"
    assert any("conflicts" in w for w in plan.warnings)


def test_rolekv_with_selector_needs_custom_backend():
    """Eviction still wins: rolekv + a selector cannot be a native dtype."""
    plan = plan_from_kv_config(
        _cfg(_default_rolekv(), selector={"name": "snapkv", "keep_ratio": 0.3})
    )
    assert not plan.native_supported
    assert plan.needs_custom_backend
    assert plan.kv_cache_dtype == "auto"


def test_static_policy_unaffected():
    """Regression guard: the rolekv branch must not change existing recipes."""
    plan = plan_from_kv_config(_cfg({"name": "static"}, codec={"name": "fp8", "skip_layers": []}))
    assert plan.native_supported
    assert plan.kv_cache_dtype == "fp8_e4m3"
