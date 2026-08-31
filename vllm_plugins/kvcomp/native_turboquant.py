# Copyright 2026 Fujitsu Ltd.
"""Helpers for vLLM native TurboQuant KV-cache presets.

vLLM exposes TurboQuant through named ``kv_cache_dtype`` strings rather than a
free-form key/value bit configuration.  Keeping that mapping in one small module
prevents serving integrations from duplicating the preset table and
also gives users a direct helper when they want to run plain TQ in vLLM.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_plugins.kvcomp._validation import strict_integer


@dataclass(frozen=True)
class TurboQuantPreset:
    """A resolved vLLM TurboQuant preset."""

    key_bits: int
    value_bits: int
    kv_cache_dtype: str
    custom_backend: bool = False


TURBOQUANT_PRESETS: dict[tuple[int, int], str] = {
    # vLLM validates the cache dtype before selecting a plugin backend.  The
    # OneComp 2-bit implementation therefore uses the 3-bit dtype only as a
    # carrier; its backend and cache-spec hook replace every physical detail.
    (2, 2): "turboquant_3bit_nc",
    (8, 4): "turboquant_k8v4",
    (4, 4): "turboquant_4bit_nc",
    (3, 4): "turboquant_k3v4_nc",
    (3, 3): "turboquant_3bit_nc",
}


def _turboquant_config_cls():
    from vllm.model_executor.layers.quantization.turboquant.config import (  # pylint: disable=import-outside-toplevel
        TurboQuantConfig,
    )

    return TurboQuantConfig


def disable_vllm_implicit_turboquant_boundary_skip() -> None:
    """Keep only the layer skips explicitly serialized by OneComp.

    vLLM 0.24 unconditionally adds the first and last two dense-model layers
    to every ``turboquant_*`` cache dtype. HayateQuant uses one of those dtype
    names as its packed-cache carrier, and native TurboQuant comparisons must
    follow the same all-layer protocol. There is no public vLLM switch for
    disabling this implicit policy, so patch its policy resolver before engine
    configuration is created. ``EngineArgs`` still unions any user-provided
    ``kv_cache_dtype_skip_layers`` with the empty implicit set, preserving
    explicit skips.
    """

    config_cls = _turboquant_config_cls()
    current = config_cls.get_boundary_skip_layers
    if getattr(current, "_onecomp_explicit_layers_only", False):
        return

    def _no_implicit_boundary_skip(_model_config, n: int = 2) -> list[str]:
        del n
        return []

    _no_implicit_boundary_skip._onecomp_explicit_layers_only = True
    _no_implicit_boundary_skip._onecomp_original = current
    config_cls.get_boundary_skip_layers = staticmethod(_no_implicit_boundary_skip)


def vllm_implicit_turboquant_boundary_skip_disabled() -> bool:
    """Return whether this process enforces OneComp's explicit-only contract."""

    current = _turboquant_config_cls().get_boundary_skip_layers
    return bool(getattr(current, "_onecomp_explicit_layers_only", False))


def resolve_turboquant_preset(
    bits: int,
    value_bits: int | None = None,
) -> TurboQuantPreset | None:
    """Resolve a key/value bit pair to vLLM's native TurboQuant dtype."""
    key_bits = strict_integer(bits, "bits", minimum=1)
    val_bits = (
        key_bits if value_bits is None else strict_integer(value_bits, "value_bits", minimum=1)
    )
    dtype = TURBOQUANT_PRESETS.get((key_bits, val_bits))
    if dtype is None:
        return None
    return TurboQuantPreset(
        key_bits=key_bits,
        value_bits=val_bits,
        kv_cache_dtype=dtype,
        custom_backend=(key_bits, val_bits) == (2, 2),
    )


def resolve_turboquant_codec(codec: dict) -> TurboQuantPreset | None:
    """Resolve a serialized OneComp TurboQuant-family codec config."""
    return resolve_turboquant_preset(
        codec.get("bits", 4),
        codec.get("value_bits"),
    )


def turboquant_engine_kwargs(bits: int, value_bits: int | None = None) -> dict:
    """Return vLLM ``LLM`` kwargs for plain native TurboQuant.

    This helper is intentionally strict: unsupported bit pairs raise instead of
    silently falling back to ``kv_cache_dtype='auto'``.
    """
    preset = resolve_turboquant_preset(bits, value_bits)
    if preset is None:
        supported = ", ".join(f"k{k}v{v}" for k, v in sorted(TURBOQUANT_PRESETS))
        raise ValueError(
            f"Unsupported vLLM TurboQuant preset k{bits}v{value_bits or bits}; "
            f"supported presets are {supported}."
        )
    if preset.custom_backend:
        from vllm_plugins.kvcomp.turboquant2_native import (  # pylint: disable=import-outside-toplevel
            configure_turboquant2_native_backend,
        )

        configure_turboquant2_native_backend()
    else:
        disable_vllm_implicit_turboquant_boundary_skip()
    return {"kv_cache_dtype": preset.kv_cache_dtype}
