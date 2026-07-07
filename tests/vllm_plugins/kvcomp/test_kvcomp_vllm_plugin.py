"""Unit tests for the OneComp KV-compression vLLM plugin
(``vllm_plugins.kvcomp.vllm_plugin``), covering the ``apply_to_engine_args``
surface end to end with a fake ``EngineArgs``-like object (no real ``vllm``
install required -- ``_vllm_accepts_dtype`` degrades to "optimistic" when
``vllm`` cannot be imported, exactly as it would on an older/absent vLLM).

These specifically exercise HayateQuant (the newest codec, replacing HALO as
the recommended default), including the exact ``get_kv_config()`` payload a
``Runner``-saved model's ``config.json`` would contain -- this is what
caught the ``block_size=None`` -> ``int(None)`` crash in
``vllm_plugins.kvcomp.mapping`` (see ``test_kvcomp.py``); here we confirm
the fix holds through the *plugin* entry point as well, not just the mapper.

Copyright 2025-2026 Fujitsu Ltd.
"""
from __future__ import annotations

import json
import os

import pytest

from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.codec.halo import HALOCodec
from vllm_plugins.kvcomp.vllm_plugin import (
    _HAYATE_NATIVE_CONFIG_ENV,
    _register_hayate_native_backend,
    apply_to_engine_args,
    plan_for_model,
    read_kv_compression_config,
)


class _FakeEngineArgs:
    """Minimal stand-in for ``vllm.engine.arg_utils.EngineArgs`` covering the
    attributes ``apply_to_engine_args`` reads/writes."""

    def __init__(self, model, kv_cache_dtype="auto"):
        self.model = model
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = False
        self.kv_cache_dtype_skip_layers = None


def _write_model_config(tmp_path, kv_compression_config):
    cfg = {"model_type": "llama", "kv_compression_config": kv_compression_config}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def test_read_kv_compression_config_missing_dir_returns_none(tmp_path):
    # No config.json at all (e.g. a bare HF Hub id) -> None, not an error.
    assert read_kv_compression_config(str(tmp_path)) is None


def test_read_kv_compression_config_no_kv_block_returns_none(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert read_kv_compression_config(str(tmp_path)) is None


def test_apply_to_engine_args_noop_without_model_path():
    engine_args = _FakeEngineArgs(model=None)
    engine_args.model = None
    out = apply_to_engine_args(engine_args, model_path=None)
    assert out is engine_args
    assert out.kv_cache_dtype == "auto"


def test_apply_to_engine_args_hayatequant_block_size_1_sets_native_dtype(tmp_path):
    codec = HayateQuantCodec(bits=4, block_size=1)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_apply_to_engine_args_hayatequant_default_auto_serves_when_possible(tmp_path, bits):
    """The exact regression this test guards: a HayateQuant model saved with
    its *default* ``block_size=None`` ("auto", the recommended/shipped
    setting) must not crash ``apply_to_engine_args``.  Supported 3/4-bit
    presets keep the byte-compatible TurboQuant dtype but activate OneComp's
    native HayateQuant key block-VQ backend; unsupported 2-bit remains on the
    faithful custom/HF path.
    """
    codec = HayateQuantCodec(bits=bits)  # block_size left at its None default
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)  # must not raise
    expected = {2: "auto", 3: "turboquant_3bit_nc", 4: "turboquant_4bit_nc"}[bits]
    assert engine_args.kv_cache_dtype == expected


def test_apply_to_engine_args_turboquant_unaffected_by_hayatequant_change(tmp_path):
    # Plain TurboQuant model config still maps to its native preset (no
    # regression from the hayatequant-specific branch added to the mapper).
    codec = TurboQuantCodec(bits=8, value_bits=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_k8v4"


def test_apply_to_engine_args_halo_still_maps_with_warning(tmp_path):
    codec = HALOCodec(bits=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"


def test_apply_to_engine_args_lowbit_leaves_auto_and_warns(tmp_path):
    from onecomp.kvcomp.codec.affine import Int4Codec

    codec = Int4Codec()
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"  # never silently downgraded


def test_plan_for_model_roundtrips_hayatequant_config(tmp_path):
    codec = HayateQuantCodec(bits=3, block_size=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    plan = plan_for_model(model_path)
    assert plan.codec_name == "hayatequant"
    assert plan.native_supported and not plan.needs_custom_backend
    assert plan.hayate_native
    assert plan.kv_cache_dtype == "turboquant_3bit_nc"
    assert plan.hayate_config["bits"] == 3
    assert plan.hayate_config["value_bits"] == 3
    assert not any("NOT reproduced" in w for w in plan.warnings)


def test_plan_for_model_hayatequant_scalar_fallback_is_explicit(tmp_path, monkeypatch):
    codec = HayateQuantCodec(bits=3)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "scalar_fallback")
    plan = plan_for_model(model_path)
    assert plan.codec_name == "hayatequant"
    assert plan.native_supported and not plan.needs_custom_backend
    assert not plan.hayate_native
    assert plan.kv_cache_dtype == "turboquant_3bit_nc"
    assert any("NOT reproduced" in w for w in plan.warnings)


def test_hayate_native_backend_registers_from_env_for_spawn(monkeypatch):
    registry = pytest.importorskip("vllm.v1.attention.backends.registry")
    from vllm_plugins.kvcomp import vllm_plugin

    registry.AttentionBackendEnum.TURBOQUANT.clear_override()
    monkeypatch.setattr(vllm_plugin, "_HAYATE_NATIVE_INSTALLED", False)
    monkeypatch.setenv(
        _HAYATE_NATIVE_CONFIG_ENV,
        json.dumps({
            "bits": 3,
            "value_bits": 3,
            "dist": "gaussian",
            "block_size": None,
            "max_codebook_bits": 9,
            "rotation_mode": "auto",
            "seed": 0,
        }, sort_keys=True),
    )

    assert _register_hayate_native_backend(vllm_plugin._get_logger())
    assert registry.AttentionBackendEnum.TURBOQUANT.is_overridden()
    assert (
        registry.AttentionBackendEnum.TURBOQUANT.get_path()
        == "vllm_plugins.kvcomp.hayate_native.HayateQuantNativeBackend"
    )
    registry.AttentionBackendEnum.TURBOQUANT.clear_override()


def test_plan_for_model_hayatequant_strict_mode_requires_custom_backend(tmp_path, monkeypatch):
    codec = HayateQuantCodec(bits=3, block_size=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "strict")
    plan = plan_for_model(model_path)
    assert plan.codec_name == "hayatequant"
    assert plan.needs_custom_backend and not plan.native_supported
    assert plan.kv_cache_dtype == "auto"


@pytest.mark.parametrize("mode", ["hadamard", "hadamard_gemm"])
def test_apply_to_engine_args_hayatequant_hadamard_rotation_native(tmp_path, mode):
    """A HayateQuant model saved with an explicit Hadamard-family rotation and
    block_size=1 maps to the native TurboQuant preset (vLLM's kernel applies
    the same randomized-Hadamard transform family) with no rotation warning."""
    codec = HayateQuantCodec(bits=4, block_size=1, rotation_mode=mode)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"
    plan = plan_for_model(model_path)
    assert any("Hadamard" in n for n in plan.notes)
    assert not any("does NOT match" in w for w in plan.warnings)


def test_apply_to_engine_args_hayatequant_dense_rotation_warns(tmp_path):
    """Dense rotation still maps to the preset (a valid quantizer) but the plan
    must carry an explicit rotation-mismatch warning -- vLLM's native kernel
    hard-codes Hadamard and would quantize under a different transform."""
    codec = HayateQuantCodec(bits=4, block_size=1, rotation_mode="dense")
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"
    plan = plan_for_model(model_path)
    assert any("does NOT match" in w and "Hadamard" in w for w in plan.warnings)
