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
import sys
from types import ModuleType

import pytest

from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.codec.halo import HALOCodec
from vllm_plugins.kvcomp.vllm_plugin import (
    _HAYATE_NATIVE_CONFIG_ENV,
    _install_engine_args_hook,
    _register_hayate_native_backend,
    apply_to_engine_args,
    plan_for_model,
    read_kv_compression_config,
)
from vllm_plugins.kvcomp.hayate_triton_config import (
    _direct_key_scale,
    _store_group_codes,
)
from vllm_plugins.kvcomp.native_turboquant import (
    resolve_turboquant_preset,
    turboquant_engine_kwargs,
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
    # Native-mapping tests intentionally request uniform quantization.  The
    # production config default is residual_window=128 and now fails closed,
    # because vLLM cannot keep a recent FP suffix.
    kv_compression_config = {"residual_window": 0, **kv_compression_config}
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


def test_engine_args_hook_propagates_mapping_failures(monkeypatch):
    """A bad serialized recipe must stop engine construction, not become FullKV."""
    from vllm_plugins.kvcomp import vllm_plugin

    class EngineArgs:
        def create_engine_config(self):
            return "engine-created"

    vllm_mod = ModuleType("vllm")
    engine_mod = ModuleType("vllm.engine")
    arg_utils_mod = ModuleType("vllm.engine.arg_utils")
    arg_utils_mod.EngineArgs = EngineArgs
    vllm_mod.engine = engine_mod
    engine_mod.arg_utils = arg_utils_mod
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)
    monkeypatch.setattr(vllm_plugin, "_PATCHED", False)

    def fail_closed(_engine_args):
        raise RuntimeError("unrepresentable recipe")

    monkeypatch.setattr(vllm_plugin, "apply_to_engine_args", fail_closed)
    assert _install_engine_args_hook(vllm_plugin._get_logger())
    with pytest.raises(RuntimeError, match="unrepresentable recipe"):
        EngineArgs().create_engine_config()


def test_apply_rejects_native_dtype_missing_from_installed_vllm(tmp_path, monkeypatch):
    model_path = _write_model_config(tmp_path, {"codec": {"name": "fp8"}})
    engine_args = _FakeEngineArgs(model=model_path)
    monkeypatch.setattr(
        "vllm_plugins.kvcomp.vllm_plugin._vllm_accepts_dtype", lambda _dtype: False
    )
    with pytest.raises(RuntimeError, match="does not support the required"):
        apply_to_engine_args(engine_args)


def test_apply_to_engine_args_hayatequant_block_size_1_fails_closed(tmp_path):
    codec = HayateQuantCodec(bits=4, block_size=1)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_apply_to_engine_args_hayatequant_default_auto_fails_closed(tmp_path, bits):
    """Default block-VQ cannot silently become FullKV or key-only native HQ.

    The byte-compatible override block-quantizes keys but not values, so the
    faithful default must stop engine construction and direct users to HF/custom.
    """
    codec = HayateQuantCodec(bits=bits)  # block_size left at its None default
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"


def test_apply_to_engine_args_turboquant_defaults_to_faithful_path(tmp_path):
    # The native preset changes both the rotation and the value quantizer, so it
    # must not be selected silently for a serialized OneComp reference codec.
    codec = TurboQuantCodec(bits=8, value_bits=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"


def test_apply_to_engine_args_turboquant_native_is_explicit(tmp_path, monkeypatch):
    codec = TurboQuantCodec(bits=8, value_bits=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_k8v4"


@pytest.mark.parametrize(
    ("bits", "value_bits", "dtype"),
    [
        (8, 4, "turboquant_k8v4"),
        (4, 4, "turboquant_4bit_nc"),
        (3, 4, "turboquant_k3v4_nc"),
        (3, 3, "turboquant_3bit_nc"),
    ],
)
def test_turboquant_engine_kwargs_exposes_all_vllm_presets(bits, value_bits, dtype):
    preset = resolve_turboquant_preset(bits, value_bits)
    assert preset is not None
    assert preset.kv_cache_dtype == dtype
    assert turboquant_engine_kwargs(bits, value_bits) == {"kv_cache_dtype": dtype}


def test_turboquant_engine_kwargs_rejects_unsupported_preset():
    with pytest.raises(ValueError, match="Unsupported vLLM TurboQuant preset"):
        turboquant_engine_kwargs(2, 2)


def test_apply_to_engine_args_halo_defaults_to_faithful_path(tmp_path):
    codec = HALOCodec(bits=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"


def test_apply_to_engine_args_lowbit_fails_before_fullkv_start(tmp_path):
    from onecomp.kvcomp.codec.affine import Int4Codec

    codec = Int4Codec()
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "auto"  # never silently downgraded


def test_plan_for_model_roundtrips_hayatequant_config(tmp_path):
    codec = HayateQuantCodec(bits=3, block_size=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    plan = plan_for_model(model_path)
    assert plan.codec_name == "hayatequant"
    assert plan.needs_custom_backend and not plan.native_supported
    assert not plan.hayate_native and plan.kv_cache_dtype == "auto"


def test_plan_for_model_hayatequant_key_only_native_is_explicit(tmp_path, monkeypatch):
    codec = HayateQuantCodec(bits=3, block_size=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    plan = plan_for_model(model_path)
    assert plan.native_supported and not plan.needs_custom_backend
    assert plan.hayate_native and plan.kv_cache_dtype == "turboquant_3bit_nc"
    assert any("key-only" in warning for warning in plan.warnings)


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
        json.dumps(
            {
                "bits": 3,
                "value_bits": 3,
                "dist": "gaussian",
                "block_size": None,
                "max_codebook_bits": 9,
                "rotation_mode": "auto",
                "seed": 0,
            },
            sort_keys=True,
        ),
    )

    assert _register_hayate_native_backend(vllm_plugin._get_logger())
    assert registry.AttentionBackendEnum.TURBOQUANT.is_overridden()
    assert (
        registry.AttentionBackendEnum.TURBOQUANT.get_path()
        == "vllm_plugins.kvcomp.hayate_native.HayateQuantNativeBackend"
    )
    registry.AttentionBackendEnum.TURBOQUANT.clear_override()


def test_registered_hayate_backend_without_config_delegates_upstream(monkeypatch):
    """A stale process-global registry override must not invent HQ defaults."""
    from types import SimpleNamespace

    pytest.importorskip("vllm", reason="native backend delegation requires vLLM")
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    from vllm_plugins.kvcomp.hayate_native import HayateQuantNativeImpl

    monkeypatch.delenv(_HAYATE_NATIVE_CONFIG_ENV, raising=False)
    delegated = []

    def fake_upstream(_self, layer, device):
        delegated.append((layer, device))

    monkeypatch.setattr(TurboQuantAttentionImpl, "_ensure_on_device", fake_upstream)
    impl = object.__new__(HayateQuantNativeImpl)
    impl.tq_config = SimpleNamespace(key_fp8=False, key_quant_bits=3)
    layer = SimpleNamespace()

    impl._ensure_on_device(layer, "cuda:0")

    assert layer._hq_native_enabled is False
    assert delegated == [(layer, "cuda:0")]


def test_hayate_store_group_codes_auto_tracks_cache_layout(monkeypatch):
    monkeypatch.delenv("KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES", raising=False)
    assert _store_group_codes(6, 32) == 32
    assert _store_group_codes(6, 64) == 64
    assert _store_group_codes(6, 128) == 128
    assert _store_group_codes(8, 32) == 64
    assert _store_group_codes(8, 128) == 128


def test_hayate_store_group_codes_rejects_unsafe_6bit_override(monkeypatch):
    monkeypatch.setenv("KVCOMP_VLLM_HAYATE_STORE_GROUP_CODES", "64")
    with pytest.raises(ValueError, match="number of key VQ blocks"):
        _store_group_codes(6, 32)


def test_hayate_direct_key_scale_only_for_single_group(monkeypatch):
    monkeypatch.delenv("KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE", raising=False)
    assert _direct_key_scale(1, True)
    assert not _direct_key_scale(2, True)
    assert not _direct_key_scale(1, False)

    monkeypatch.setenv("KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE", "never")
    assert not _direct_key_scale(1, True)
    monkeypatch.setenv("KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE", "always")
    assert _direct_key_scale(1, True)
    monkeypatch.setenv("KVCOMP_VLLM_HAYATE_DIRECT_KEY_SCALE", "invalid")
    with pytest.raises(ValueError, match="DIRECT_KEY_SCALE"):
        _direct_key_scale(1, True)


def test_plan_for_model_hayatequant_strict_mode_requires_custom_backend(tmp_path, monkeypatch):
    codec = HayateQuantCodec(bits=3, block_size=4)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "strict")
    plan = plan_for_model(model_path)
    assert plan.codec_name == "hayatequant"
    assert plan.needs_custom_backend and not plan.native_supported
    assert plan.kv_cache_dtype == "auto"

    engine_args = _FakeEngineArgs(model=model_path)
    with pytest.raises(RuntimeError, match="cannot run through native vLLM"):
        apply_to_engine_args(engine_args)


@pytest.mark.parametrize("mode", ["hadamard", "hadamard_gemm"])
def test_apply_to_engine_args_hayatequant_hadamard_rotation_native(tmp_path, mode, monkeypatch):
    """A HayateQuant model saved with an explicit Hadamard-family rotation and
    block_size=1 maps to the native TurboQuant preset (vLLM's kernel applies
    the same randomized-Hadamard transform family) with no rotation warning."""
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    codec = HayateQuantCodec(bits=4, block_size=1, rotation_mode=mode)
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"
    plan = plan_for_model(model_path)
    assert any("Hadamard" in n for n in plan.notes)
    assert not any("does NOT match" in w for w in plan.warnings)


def test_apply_to_engine_args_hayatequant_dense_rotation_warns(tmp_path, monkeypatch):
    """Dense rotation still maps to the preset (a valid quantizer) but the plan
    must carry an explicit rotation-mismatch warning -- vLLM's native kernel
    hard-codes Hadamard and would quantize under a different transform."""
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    codec = HayateQuantCodec(bits=4, block_size=1, rotation_mode="dense")
    model_path = _write_model_config(tmp_path, {"codec": codec.get_kv_config()})
    engine_args = _FakeEngineArgs(model=model_path)
    apply_to_engine_args(engine_args)
    assert engine_args.kv_cache_dtype == "turboquant_4bit_nc"
    plan = plan_for_model(model_path)
    assert any("does NOT match" in w and "Hadamard" in w for w in plan.warnings)
