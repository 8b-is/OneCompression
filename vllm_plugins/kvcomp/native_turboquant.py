# Copyright 2026 Fujitsu Ltd.
"""Helpers for vLLM native TurboQuant KV-cache presets.

vLLM exposes TurboQuant through named ``kv_cache_dtype`` strings rather than a
free-form key/value bit configuration.  Keeping that mapping in one small module
prevents HayateQuant/HALO planning logic from duplicating the preset table and
also gives users a direct helper when they want to run plain TQ in vLLM.
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


TURBOQUANT_PRESETS: dict[tuple[int, int], str] = {
    (8, 4): "turboquant_k8v4",
    (4, 4): "turboquant_4bit_nc",
    (3, 4): "turboquant_k3v4_nc",
    (3, 3): "turboquant_3bit_nc",
}


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
    return TurboQuantPreset(key_bits=key_bits, value_bits=val_bits, kv_cache_dtype=dtype)


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
    return {"kv_cache_dtype": preset.kv_cache_dtype}
