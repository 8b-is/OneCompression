"""Unit tests for the rolekv policy -> vLLM engine kwargs mapping.

Pure-Python: neither vllm, torch nor a GPU is required (the mapping module
is deliberately torch-free so it can run on a CPU-only login node).

Copyright 2025-2026 Fujitsu Ltd.
"""

from vllm_plugins.kvcomp.mapping import plan_from_kv_config


def _cfg(policy, codec=None, selector=None, residual_window=0):
    return {
        "codec": codec,
        "selector": selector,
        "policy": policy,
        "residual_window": residual_window,
    }


def _default_rolekv():
    return {
        "name": "rolekv",
        "bits_per_role": {"system": 8, "user": 8, "assistant": 4, "reasoning": 4, "tool": 2},
        "default_bits": 4,
        "skip_layers": [],
    }


def test_default_rolekv_requires_faithful_custom_backend():
    """A uniform FP8 dtype cannot represent role-specific affine precision."""
    plan = plan_from_kv_config(_cfg(_default_rolekv()))
    assert plan.needs_custom_backend
    assert not plan.native_supported
    assert plan.kv_cache_dtype == "auto"
    assert any("change both the quantizer and realised memory rate" in n for n in plan.notes)


def test_rolekv_positive_residual_window_fails_closed():
    plan = plan_from_kv_config(_cfg(_default_rolekv(), residual_window=128))
    assert plan.needs_custom_backend
    assert not plan.native_supported
    assert plan.kv_cache_dtype == "auto"
    assert any("residual_window=128" in note for note in plan.notes)


def test_mixed_rolekv_never_collapses_roles():
    plan = plan_from_kv_config(_cfg(_default_rolekv()))
    assert plan.needs_custom_backend
    assert any("per-role affine bit widths" in n for n in plan.notes)


def test_uniform_compressed_rolekv_still_requires_affine_custom_path():
    policy = {
        "name": "rolekv",
        "bits_per_role": {r: 8 for r in ("system", "user", "assistant", "reasoning", "tool")},
        "default_bits": 8,
        "skip_layers": [],
    }
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.needs_custom_backend
    assert not plan.native_supported
    assert plan.kv_cache_dtype == "auto"


def test_all_fp16_rolekv_is_the_only_faithful_uniform_native_plan():
    policy = {
        "name": "rolekv",
        "bits_per_role": {
            role: 16 for role in ("system", "user", "assistant", "reasoning", "tool")
        },
        "default_bits": 16,
        "sink_tokens": 4,
        "skip_layers": [],
    }
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.native_supported and not plan.needs_custom_backend
    assert plan.kv_cache_dtype == "auto"


def test_low_bit_only_rolekv_does_not_substitute_fp8():
    policy = {
        "name": "rolekv",
        "bits_per_role": {"system": 4, "tool": 2},
        "default_bits": 2,
        "skip_layers": [],
    }
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.needs_custom_backend
    assert plan.kv_cache_dtype == "auto"


def test_rolekv_propagates_skip_layers():
    policy = _default_rolekv()
    policy["skip_layers"] = [0, 27]
    plan = plan_from_kv_config(_cfg(policy))
    assert plan.skip_layers == [0, 27]
    assert "kv_cache_dtype_skip_layers" not in plan.engine_kwargs()
    assert plan.needs_custom_backend


def test_rolekv_conflicting_requested_dtype_warns():
    plan = plan_from_kv_config(_cfg(_default_rolekv()), requested_kv_cache_dtype="fp8_e5m2")
    assert plan.kv_cache_dtype == "auto"
    assert any("cannot reproduce RoleKV" in w for w in plan.warnings)


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
