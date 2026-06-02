"""Map an OneComp ``kv_compression_config`` to vLLM engine arguments.

This module is the *pure-Python* bridge between
:class:`onecomp.kvcomp.KVCompressionConfig` (written into ``config.json`` as the
``kv_compression_config`` block by :class:`onecomp.runner.Runner`) and the
arguments that configure vLLM's KV cache.

Design (see ``docs/design/kvcomp-architecture.md`` section 11).  Recent vLLM
ships native KV-cache dtypes well beyond FP8, so the bridge maps onto them:

* **FP8** -> ``"fp8_e4m3"`` / ``"fp8_e5m2"`` (data-free, ``calculate_kv_scales``).
* **TurboQuant** (and the TurboQuant-family **HALO**) -> the native
  ``turboquant_*`` presets (``turboquant_k8v4``, ``turboquant_4bit_nc``,
  ``turboquant_k3v4_nc``, ``turboquant_3bit_nc``) chosen by ``(key_bits,
  value_bits)`` -- vLLM runs Hadamard rotation + Lloyd-Max keys + uniform values
  with automatic boundary-layer protection (FlashAttention-2).
* **NVFP4** -> ``"nvfp4"``; **INT8** -> ``"int8_per_token_head"``.
* Remaining low-bit/asymmetric codecs (INT4/INT3/INT2, KIVI, Ternary) and
  token-eviction selectors (SnapKV, ThinKV TBE) / adaptive per-token policies
  (TaperKV, ThinKV) are not expressible as one ``kv_cache_dtype`` and require the
  custom attention-backend / HF cache path.  For those the module returns
  ``native_supported=False`` with an explicit reason rather than silently
  downgrading.  Native dtypes added only in recent vLLM are validated at
  apply-time so an older vLLM falls back gracefully.

The module deliberately imports neither ``vllm`` nor ``torch`` so it can be
unit-tested on a CPU-only login node.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# OneComp codec name -> vLLM native ``kv_cache_dtype`` value.
_NATIVE_FP8 = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}

# vLLM (recent main) ships a native TurboQuant KV cache (Hadamard rotation +
# Lloyd-Max keys + uniform values, with boundary-layer protection and a
# dedicated attention impl).  Its named presets map onto OneComp's TurboQuant
# (and the TurboQuant-family HALO) by (key_bits, value_bits).  Source of truth:
# ``vllm/model_executor/layers/quantization/turboquant/config.py`` (TQ_PRESETS).
_TURBOQUANT_PRESETS = {
    (8, 4): "turboquant_k8v4",      # FP8 keys + 4-bit values   (~2.6x)
    (4, 4): "turboquant_4bit_nc",   # 4-bit MSE keys + 4-bit V  (~3.8x)
    (3, 4): "turboquant_k3v4_nc",   # 3-bit MSE keys + 4-bit V  (~3.5x)
    (3, 3): "turboquant_3bit_nc",   # 3-bit keys + 3-bit values (~4.9x)
}

# Other native (non-FP8) KV cache dtypes vLLM accepts.
_NATIVE_OTHER = {
    "nvfp4": "nvfp4",
    "int8": "int8_per_token_head",
}

# Codecs implemented in OneComp with no native vLLM kv_cache_dtype; they require
# the (experimental) custom attention-backend path to run *inside* vLLM, but work
# in the HF cache path today.
_NEEDS_CUSTOM_BACKEND = {
    "int4",
    "int3",
    "int2",
    "kivi",
    "ternary",
}


def _turboquant_preset(codec: dict):
    """Return (preset_name or None, key_bits, value_bits) for a TQ-family codec."""
    bits = int(codec.get("bits", 4))
    vbits = codec.get("value_bits")
    vbits = int(vbits) if vbits is not None else bits
    return _TURBOQUANT_PRESETS.get((bits, vbits)), bits, vbits


@dataclass
class KVCompVLLMPlan:
    """Resolved vLLM KV-cache configuration derived from an OneComp config.

    Attributes:
        kv_cache_dtype: Value to pass to vLLM ``LLM(kv_cache_dtype=...)`` /
            ``EngineArgs``.  ``"auto"`` keeps the model dtype (no compression).
        native_supported: Whether the codec maps onto a native vLLM
            ``kv_cache_dtype`` (FP8).  ``False`` means a custom attention
            backend is required and ``kv_cache_dtype`` was left at ``"auto"``.
        codec_name: The OneComp codec name (``None`` for eviction-only recipes).
        calculate_kv_scales: Whether vLLM should compute FP8 scales online
            (mirrors OneComp's data-free, tuning-free FP8 path).
        needs_custom_backend: ``True`` when the recipe needs the experimental
            platform/attention-backend path to run inside vLLM.
        notes: Human-readable explanation of the mapping decision.
        warnings: Non-fatal issues (e.g. ``kv_cache_dtype`` conflict).
    """

    kv_cache_dtype: str = "auto"
    native_supported: bool = False
    codec_name: Optional[str] = None
    calculate_kv_scales: bool = False
    needs_custom_backend: bool = False
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def engine_kwargs(self) -> dict:
        """Return the subset of ``LLM`` / ``EngineArgs`` kwargs to apply."""
        kwargs: dict = {"kv_cache_dtype": self.kv_cache_dtype}
        if self.kv_cache_dtype.startswith("fp8") and self.calculate_kv_scales:
            kwargs["calculate_kv_scales"] = True
        return kwargs


def plan_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> KVCompVLLMPlan:
    """Resolve a :class:`KVCompVLLMPlan` from a ``kv_compression_config`` dict.

    Args:
        kv_config: The ``kv_compression_config`` block (``KVCompressionConfig.
            to_dict()``), or ``None`` (no compression).
        requested_kv_cache_dtype: A ``kv_cache_dtype`` the caller already set
            (e.g. on the command line).  Used for conflict detection.

    Returns:
        A :class:`KVCompVLLMPlan`.  Never raises on an unsupported codec -- the
        caller inspects ``native_supported`` / ``needs_custom_backend``.
    """
    plan = KVCompVLLMPlan()
    if not kv_config:
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        plan.notes.append("No kv_compression_config; KV cache left uncompressed.")
        return plan

    codec = kv_config.get("codec") or None
    selector = kv_config.get("selector") or None
    policy = kv_config.get("policy") or None
    plan.codec_name = codec.get("name") if codec else None

    if selector is not None:
        plan.needs_custom_backend = True
        plan.notes.append(
            f"Selector '{selector.get('name')}' (token eviction) needs the custom "
            "attention backend; native kv_cache_dtype cannot evict tokens."
        )

    # Per-token adaptive recipes (ThinKV graded eviction, TaperKV graded
    # quantization) are not a single kv_cache_dtype and need the custom backend.
    policy_name = policy.get("name") if policy else None
    if policy_name not in (None, "static"):
        plan.needs_custom_backend = True
        plan.notes.append(
            f"Policy '{policy_name}' applies per-token adaptive precision; it needs "
            "the custom attention backend (use the HF cache path today)."
        )

    # If the recipe already needs the custom backend (token eviction or an
    # adaptive per-token policy), no single ``kv_cache_dtype`` can express it --
    # the codec's native dtype is irrelevant.  Finalise as custom-backend.
    if plan.needs_custom_backend:
        plan.native_supported = False
        plan.kv_cache_dtype = "auto"
        if codec is not None:
            plan.notes.append(
                f"Codec '{plan.codec_name}' is part of an eviction/adaptive recipe; "
                "the whole recipe needs the custom attention backend (HF cache path)."
            )
        return plan

    if codec is None:
        # Eviction-only recipe handled above; here: nothing to compress.
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        return plan

    name = plan.codec_name

    def _set_native(dtype: str, note: str):
        plan.kv_cache_dtype = dtype
        plan.native_supported = True
        plan.notes.append(note)
        if requested_kv_cache_dtype not in ("auto", dtype):
            plan.warnings.append(
                f"Requested kv_cache_dtype='{requested_kv_cache_dtype}' conflicts "
                f"with codec '{name}' -> '{dtype}'; using the codec's dtype."
            )

    if name in _NATIVE_FP8:
        dtype = _NATIVE_FP8[name]
        # OneComp FP8 is data-free / online -> let vLLM compute scales online.
        plan.calculate_kv_scales = True
        _set_native(dtype, f"Codec '{name}' -> native vLLM kv_cache_dtype='{dtype}'.")
    elif name in ("turboquant", "halo"):
        preset, kb, vb = _turboquant_preset(codec)
        if preset is not None:
            _set_native(
                preset,
                f"Codec '{name}' (k{kb}v{vb}) -> native vLLM kv_cache_dtype="
                f"'{preset}' (requires a vLLM build with native TurboQuant KV "
                "cache + FlashAttention-2; boundary layers auto-protected).",
            )
            if name == "halo":
                plan.warnings.append(
                    "HALO maps to vLLM's faithful TurboQuant kernel (Lloyd-Max "
                    "keys + uniform values + norm-correction); HALO's "
                    "Hadamard-GEMM / least-squares-scale refinements run only in "
                    "the HF cache path."
                )
        else:
            plan.needs_custom_backend = True
            plan.kv_cache_dtype = "auto"
            plan.notes.append(
                f"Codec '{name}' k{kb}v{vb} has no native TurboQuant preset "
                "(vLLM presets: k8v4, 4bit_nc, k3v4_nc, 3bit_nc); use the HF "
                "cache path or one of those bit-widths."
            )
    elif name in _NATIVE_OTHER:
        dtype = _NATIVE_OTHER[name]
        _set_native(dtype, f"Codec '{name}' -> native vLLM kv_cache_dtype='{dtype}'.")
    elif name in _NEEDS_CUSTOM_BACKEND:
        plan.needs_custom_backend = True
        plan.native_supported = False
        plan.kv_cache_dtype = "auto"
        plan.notes.append(
            f"Codec '{name}' has no native vLLM kv_cache_dtype; it requires the "
            "experimental custom attention backend (use the HF cache path for "
            "faithful results today)."
        )
    else:
        plan.native_supported = False
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.warnings.append(f"Unknown KV codec '{name}'; leaving kv_cache_dtype unchanged.")

    return plan


def engine_kwargs_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> dict:
    """Convenience: return the vLLM engine kwargs for a ``kv_compression_config``.

    Example::

        from vllm import LLM
        from vllm_plugins.kvcomp.mapping import engine_kwargs_from_kv_config

        kwargs = engine_kwargs_from_kv_config(cfg["kv_compression_config"])
        llm = LLM(model=path, **kwargs)
    """
    return plan_from_kv_config(kv_config, requested_kv_cache_dtype).engine_kwargs()
