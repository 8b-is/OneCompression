from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_plugins.kvcomp import native_turboquant
from vllm_plugins.kvcomp.hayate_config import (
    hayatequant_native_recipe,
    hayatequant_native_recipe_from_mapping,
)
from vllm_plugins.kvcomp.mapping import plan_from_kv_config
from vllm_plugins.kvcomp.taper_config import TaperKVVLLMRecipe
from vllm_plugins.kvcomp.vllm_plugin import _apply_taper_engine_contract


def _config(*, codec=None, policy=None, selector=None, residual_window=0):
    return {
        "codec": codec,
        "selector": selector,
        "policy": policy or {"name": "static"},
        "residual_window": residual_window,
    }


@pytest.mark.parametrize(
    ("rate", "carrier", "slot_bytes"),
    [
        (2, "turboquant_3bit_nc", 68),
        (3, "turboquant_3bit_nc", 100),
        (4, "turboquant_4bit_nc", 132),
    ],
)
def test_hayatequant_mapping_and_packed_layout_are_canonical(rate, carrier, slot_bytes):
    plan = plan_from_kv_config(
        _config(codec={"name": "hayatequant", "bits": rate, "skip_layers": [2]})
    )

    assert plan.native_supported and plan.hayatequant_native
    assert not plan.needs_custom_backend
    assert plan.kv_cache_dtype == carrier
    assert plan.hayatequant_config == {"rate": rate}
    assert plan.skip_layers == [2]
    recipe = hayatequant_native_recipe(rate)
    assert recipe.layout(128)["slot_size_aligned"] == slot_bytes
    assert recipe.key_code_bits_per_block + recipe.value_code_bits_per_block == 4 * rate
    assert (
        recipe.key_code_bits_per_block,
        recipe.value_code_bits_per_block,
    ) == (2 * rate + 2, 2 * rate - 2)


@pytest.mark.parametrize("rate", [1, 5, True, 3.5])
def test_hayatequant_rejects_noncanonical_rates(rate):
    with pytest.raises((TypeError, ValueError)):
        hayatequant_native_recipe(rate)


def test_exact_integral_json_float_keeps_the_same_rate():
    assert hayatequant_native_recipe(3.0).rate == 3


def test_hayatequant_process_payload_has_only_rate():
    assert hayatequant_native_recipe_from_mapping({"rate": 3}).rate == 3
    with pytest.raises(ValueError, match="non-canonical"):
        hayatequant_native_recipe_from_mapping({"rate": 3, "encoder": "exact"})


@pytest.mark.parametrize(
    "extra",
    [
        {"vllm_mode": "full_native"},
        {"block_size": 2},
        {"key_scale_mode": "nc"},
        {"encoder": "exact"},
    ],
)
def test_obsolete_hayatequant_fields_fail_closed(extra):
    codec = {"name": "hayatequant", "bits": 3, **extra}
    plan = plan_from_kv_config(_config(codec=codec))

    assert plan.needs_custom_backend and not plan.native_supported
    assert any("obsolete/non-canonical" in note for note in plan.notes)


def test_hayatequant_calibration_and_residual_require_the_hf_path():
    calibrated = plan_from_kv_config(
        _config(codec={"name": "hayatequant", "bits": 3, "calibration": "artifact.pt"})
    )
    residual = plan_from_kv_config(
        _config(codec={"name": "hayatequant", "bits": 3}, residual_window=1)
    )

    assert calibrated.needs_custom_backend and not calibrated.native_supported
    assert residual.needs_custom_backend and not residual.native_supported


def _taper_policy(**overrides):
    return {
        "name": "taperkv",
        "key_bits": 8,
        "value_bits": 4,
        "b_min": 4,
        "b_min_value": 2,
        "sink_tokens": 4,
        "recent_tokens": 128,
        "key_ladder": [8, 6, 4],
        "value_ladder": [4, 3, 2],
        "skip_layers": [],
        "requant_stride": 8,
        **overrides,
    }


def test_taperkv_mapping_preserves_independent_role_ladders():
    policy = _taper_policy()
    plan = plan_from_kv_config(_config(policy=policy))

    assert plan.native_supported and plan.taper_backend
    assert not plan.needs_custom_backend
    assert plan.kv_cache_dtype == "auto"
    assert plan.taper_config["key_ladder"] == [8, 6, 4]
    assert plan.taper_config["value_ladder"] == [4, 3, 2]
    assert plan.taper_config["implementation"] == "gpu_semantic_model_dtype"


def test_taperkv_rejects_unknown_fields_nonzero_residual_and_composition():
    unknown = plan_from_kv_config(_config(policy=_taper_policy(key_lader=[8, 4])))
    residual = plan_from_kv_config(_config(policy=_taper_policy(), residual_window=128))
    composed = plan_from_kv_config(
        _config(
            codec={"name": "hayatequant", "bits": 3},
            policy=_taper_policy(),
        )
    )

    for plan in (unknown, residual, composed):
        assert plan.needs_custom_backend and not plan.native_supported
    assert any("non-canonical TaperKV" in note for note in unknown.notes)
    assert any("residual_window=0" in note for note in residual.notes)
    assert any("standalone" in note for note in composed.notes)


def test_token_taper_engine_keeps_graph_replay_and_disables_shared_prefix_pages():
    engine_args = SimpleNamespace(
        enforce_eager=False,
        enable_prefix_caching=True,
        attention_backend=None,
        compilation_config=SimpleNamespace(cudagraph_mode="FULL"),
    )
    _apply_taper_engine_contract(engine_args)
    assert engine_args.enforce_eager is False
    assert engine_args.compilation_config.cudagraph_mode == "FULL"
    assert engine_args.enable_prefix_caching is False
    assert engine_args.attention_backend == "FLASH_ATTN"

    no_compile_config = SimpleNamespace(enable_prefix_caching=True, compilation_config=None)
    _apply_taper_engine_contract(no_compile_config)
    assert no_compile_config.enable_prefix_caching is False


def test_block_taper_engine_disables_graph_replay():
    engine_args = SimpleNamespace(
        enable_prefix_caching=True,
        attention_backend=None,
        compilation_config=SimpleNamespace(cudagraph_mode="FULL"),
    )
    _apply_taper_engine_contract(
        engine_args,
        {"policy_name": "block_taperkv", "quant_granularity": "block"},
    )
    assert engine_args.compilation_config.cudagraph_mode.name == "NONE"


def test_block_taper_requires_page_safe_residual_contract():
    mapping = {
        **_taper_policy(name="block_taperkv"),
        "block_size": 16,
        "residual_fp16": True,
        "quant_granularity": "block",
    }
    recipe = TaperKVVLLMRecipe.from_policy(mapping)
    assert recipe.block_size == 16

    mapping["residual_fp16"] = False
    with pytest.raises(ValueError, match="residual_fp16=True"):
        TaperKVVLLMRecipe.from_policy(mapping)


def test_hayatequant_assignment_is_fixed_and_not_workload_named():
    recipe = hayatequant_native_recipe(4)
    assignment = recipe.provenance(128)["assignment"]
    assert assignment["mode"] == "fixed_candidate_lut"
    assert "auto_exact_work_threshold" not in assignment


def test_vllm_turboquant_uses_only_explicit_layer_skips(monkeypatch):
    class FakeTurboQuantConfig:
        @staticmethod
        def get_boundary_skip_layers(_model_config, n=2):
            return [str(index) for index in range(n)]

    monkeypatch.setattr(
        native_turboquant,
        "_turboquant_config_cls",
        lambda: FakeTurboQuantConfig,
    )

    native_turboquant.disable_vllm_implicit_turboquant_boundary_skip()
    patched = FakeTurboQuantConfig.get_boundary_skip_layers

    assert patched(object()) == []
    assert native_turboquant.vllm_implicit_turboquant_boundary_skip_disabled()
    native_turboquant.disable_vllm_implicit_turboquant_boundary_skip()
    assert FakeTurboQuantConfig.get_boundary_skip_layers is patched


def test_native_turboquant_kwargs_disable_hidden_boundary_skips(monkeypatch):
    calls = []
    monkeypatch.setattr(
        native_turboquant,
        "disable_vllm_implicit_turboquant_boundary_skip",
        lambda: calls.append(True),
    )

    assert native_turboquant.turboquant_engine_kwargs(3) == {
        "kv_cache_dtype": "turboquant_3bit_nc"
    }
    assert calls == [True]
