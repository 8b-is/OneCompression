"""Map an OneComp ``kv_compression_config`` to vLLM engine arguments.

This module is the *pure-Python* bridge between
:class:`onecomp.kvcomp.KVCompressionConfig` (written into ``config.json`` as the
``kv_compression_config`` block by :class:`onecomp.runner.Runner`) and the
arguments that configure vLLM's KV cache.

Design (see ``docs/design/kvcomp-architecture.md`` section 11):

* **FP8** is representable by vLLM *natively* via the ``kv_cache_dtype`` engine
  argument (``"fp8_e4m3"`` / ``"fp8_e5m2"``).  This is the recommended 2026
  baseline and needs no custom attention backend, so it is fully supported here.
* Low-bit / asymmetric / rotated codecs (INT4, KIVI, TurboQuant, NVFP4, ...) and
  token-eviction selectors (SnapKV, ThinKV TBE) are *not* expressible as a
  ``kv_cache_dtype`` and require a custom ``AttentionBackend`` (the experimental
  platform path).  For those this module returns ``native_supported=False`` with
  an explicit reason rather than silently downgrading.

The module deliberately imports neither ``vllm`` nor ``torch`` so it can be
unit-tested on a CPU-only login node.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# OneComp codec name -> vLLM native ``kv_cache_dtype`` value.
# vLLM natively supports only FP8 KV cache dtypes; everything else needs a
# custom attention backend.
_NATIVE_FP8 = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}

# Codecs that are implemented in OneComp but require the (experimental) custom
# attention-backend path to run *inside vLLM*.  They still work in the HF cache.
_NEEDS_CUSTOM_BACKEND = {
    "int8",
    "int4",
    "int2",
    "kivi",
    "turboquant",
    "halo",
    "nvfp4",
    "ternary",
}


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

    if codec is None:
        # Eviction-only recipe: dtype unchanged.
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = not plan.needs_custom_backend
        return plan

    name = plan.codec_name
    if name in _NATIVE_FP8:
        dtype = _NATIVE_FP8[name]
        plan.kv_cache_dtype = dtype
        plan.native_supported = True
        # OneComp FP8 is data-free / online -> let vLLM compute scales online.
        plan.calculate_kv_scales = True
        plan.notes.append(f"Codec '{name}' -> native vLLM kv_cache_dtype='{dtype}'.")
        if requested_kv_cache_dtype not in ("auto", dtype):
            plan.warnings.append(
                f"Requested kv_cache_dtype='{requested_kv_cache_dtype}' conflicts "
                f"with codec '{name}' -> '{dtype}'; using the codec's dtype."
            )
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
