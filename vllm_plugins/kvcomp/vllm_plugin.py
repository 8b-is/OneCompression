"""vLLM general plugin for OneComp KV cache compression.

Registered via the ``vllm.general_plugins`` entry point (``pyproject.toml``),
exactly like the weight-quantization plugins (``dbf``, ``mixed_gptq``).  The two
subsystems coexist: weight quant replaces ``LinearMethod`` while KV comp
configures the KV cache, so a model can be both weight- and KV-compressed.  Only
a ``kv_cache_dtype`` conflict is warned about (see ``mapping``).

Scope of this plugin (agreed scaffold):

* Discover and validate the ``kv_compression_config`` saved by
  :class:`onecomp.runner.Runner`.
* Map the **FP8** codec onto vLLM's *native* ``kv_cache_dtype`` (the recommended
  baseline), via :func:`vllm_plugins.kvcomp.mapping.engine_kwargs_from_kv_config`.
* For low-bit / eviction recipes, surface that the experimental custom attention
  backend is required (not silently downgrade).

``vllm`` is imported lazily inside the registration hook so that importing this
module (e.g. in the smoke test) does not require vLLM to be installed.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from vllm_plugins.kvcomp.mapping import (
    KVCompVLLMPlan,
    engine_kwargs_from_kv_config,
    plan_from_kv_config,
)

_PLUGIN_NAME = "onecomp_kvcomp"


def _get_logger():
    """Return vLLM's logger if available, else the stdlib logger."""
    try:
        from vllm.logger import init_logger  # pylint: disable=import-outside-toplevel

        return init_logger(__name__)
    except Exception:  # pragma: no cover - vLLM not installed
        from logging import getLogger  # pylint: disable=import-outside-toplevel

        return getLogger(__name__)


def read_kv_compression_config(model_path: str) -> Optional[dict]:
    """Read the ``kv_compression_config`` block from a model's ``config.json``.

    Returns ``None`` when the model directory has no KV compression config.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("kv_compression_config")


def plan_for_model(model_path: str, requested_kv_cache_dtype: str = "auto") -> KVCompVLLMPlan:
    """Resolve a :class:`KVCompVLLMPlan` for a model directory."""
    return plan_from_kv_config(read_kv_compression_config(model_path), requested_kv_cache_dtype)


def apply_to_engine_args(engine_args, model_path: Optional[str] = None):
    """Set ``kv_cache_dtype`` on a vLLM ``EngineArgs`` from its model's config.

    Mutates and returns ``engine_args``.  Safe to call when the model has no KV
    compression config (no-op).  Use in custom launchers that build ``EngineArgs``
    explicitly; the plain ``LLM(...)`` path can instead pass
    :func:`engine_kwargs_from_kv_config`.
    """
    logger = _get_logger()
    path = model_path or getattr(engine_args, "model", None)
    if not path:
        return engine_args
    plan = plan_for_model(path, getattr(engine_args, "kv_cache_dtype", "auto") or "auto")
    for msg in plan.notes:
        logger.info("[kvcomp] %s", msg)
    for msg in plan.warnings:
        logger.warning("[kvcomp] %s", msg)
    if plan.native_supported and plan.kv_cache_dtype != "auto":
        engine_args.kv_cache_dtype = plan.kv_cache_dtype
        if plan.calculate_kv_scales and hasattr(engine_args, "calculate_kv_scales"):
            engine_args.calculate_kv_scales = True
        logger.info("[kvcomp] set kv_cache_dtype=%s", plan.kv_cache_dtype)
    elif plan.needs_custom_backend:
        logger.warning(
            "[kvcomp] codec=%s needs the experimental custom attention backend; "
            "kv_cache_dtype left at 'auto'. Use the HF cache path for faithful "
            "low-bit/eviction results today.",
            plan.codec_name,
        )
    return engine_args


def register_vllm_plugin():
    """Entry point for ``vllm.general_plugins``.

    vLLM calls this once per engine process.  We only announce availability and
    validate the environment here; the actual ``kv_cache_dtype`` is applied via
    :func:`engine_kwargs_from_kv_config` / :func:`apply_to_engine_args` when the
    engine args are constructed.  This keeps the plugin non-invasive (it never
    hijacks the platform), matching how the weight plugins register.
    """
    logger = _get_logger()
    logger.info(
        "Registered OneComp KV compression plugin '%s' (FP8 -> native "
        "kv_cache_dtype; low-bit/eviction via experimental backend).",
        _PLUGIN_NAME,
    )


__all__ = [
    "register_vllm_plugin",
    "read_kv_compression_config",
    "plan_for_model",
    "apply_to_engine_args",
    "engine_kwargs_from_kv_config",
]
