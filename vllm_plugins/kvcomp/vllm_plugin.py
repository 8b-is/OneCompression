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
_HAYATE_NATIVE_CONFIG_ENV = "KVCOMP_VLLM_HAYATE_NATIVE_CONFIG"


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
        # Local path only: a bare HF Hub model id (no local config.json) yields a
        # silent no-op.  Log it so a missing KV-comp config on a Hub id is visible
        # rather than mysteriously un-applied.
        _get_logger().info(
            "[kvcomp] no local config.json at '%s' (HF Hub id?); KV-comp config "
            "not read -- pass a local path or apply engine kwargs manually.",
            model_path)
        return None
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg.get("kv_compression_config")


def plan_for_model(model_path: str, requested_kv_cache_dtype: str = "auto") -> KVCompVLLMPlan:
    """Resolve a :class:`KVCompVLLMPlan` for a model directory."""
    return plan_from_kv_config(read_kv_compression_config(model_path), requested_kv_cache_dtype)


def _vllm_accepts_dtype(dtype: Optional[str]) -> bool:
    """Best-effort: does the installed vLLM accept this ``kv_cache_dtype``?

    The native TurboQuant / NVFP4 / int8-per-token-head dtypes only exist on
    recent vLLM.  We introspect the ``CacheDType`` literal so an older vLLM falls
    back gracefully (leave ``auto``) instead of erroring at engine start.  When
    the literal cannot be located we are optimistic (return ``True``) and let
    vLLM validate.
    """
    if dtype in (None, "auto"):
        return True
    import typing  # pylint: disable=import-outside-toplevel
    for modname, attr in (("vllm.config.cache", "CacheDType"),
                          ("vllm.config", "CacheDType")):
        try:
            mod = __import__(modname, fromlist=[attr])
            choices = set(typing.get_args(getattr(mod, attr)))
            if choices:
                return dtype in choices
        except Exception:  # pragma: no cover - version dependent
            continue
    return True


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
        if not _vllm_accepts_dtype(plan.kv_cache_dtype):
            logger.warning(
                "[kvcomp] installed vLLM does not support kv_cache_dtype='%s' "
                "(needs a newer vLLM); leaving 'auto'. Upgrade vLLM or use the HF "
                "cache path.", plan.kv_cache_dtype)
            return engine_args
        if plan.hayate_native:
            _install_hayate_native_backend(plan, logger)
        engine_args.kv_cache_dtype = plan.kv_cache_dtype
        if plan.calculate_kv_scales and hasattr(engine_args, "calculate_kv_scales"):
            engine_args.calculate_kv_scales = True
        if plan.skip_layers and hasattr(engine_args, "kv_cache_dtype_skip_layers"):
            engine_args.kv_cache_dtype_skip_layers = ",".join(str(i) for i in plan.skip_layers)
            logger.info("[kvcomp] set kv_cache_dtype_skip_layers=%s",
                        engine_args.kv_cache_dtype_skip_layers)
        logger.info("[kvcomp] set kv_cache_dtype=%s", plan.kv_cache_dtype)
    elif plan.needs_custom_backend:
        logger.warning(
            "[kvcomp] codec=%s needs the experimental custom attention backend; "
            "kv_cache_dtype left at 'auto'. Use the HF cache path for faithful "
            "low-bit/eviction results today.",
            plan.codec_name,
        )
    return engine_args


_PATCHED = False
_HAYATE_NATIVE_INSTALLED = False


def _register_hayate_native_backend(logger) -> bool:
    """Register OneComp's HayateQuant backend override for vLLM TurboQuant."""
    global _HAYATE_NATIVE_INSTALLED
    if not os.environ.get(_HAYATE_NATIVE_CONFIG_ENV):
        return False
    if _HAYATE_NATIVE_INSTALLED:
        return True
    try:
        from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.TURBOQUANT,
            "vllm_plugins.kvcomp.hayate_native.HayateQuantNativeBackend",
        )
        _HAYATE_NATIVE_INSTALLED = True
        logger.info(
            "[kvcomp] registered native HayateQuant backend override for "
            "vLLM TURBOQUANT attention (config=%s).",
            os.environ[_HAYATE_NATIVE_CONFIG_ENV],
        )
        return True
    except Exception as exc:  # pragma: no cover - vLLM absent/version dependent
        logger.warning(
            "[kvcomp] could not register native HayateQuant vLLM backend (%s); "
            "kv_cache_dtype will still map to the byte-compatible TurboQuant "
            "preset, but block-VQ will not be active unless the backend is "
            "registered before engine creation.",
            exc,
        )
        return False


def _install_hayate_native_backend(plan: KVCompVLLMPlan, logger) -> bool:
    """Install OneComp's native HayateQuant backend from a resolved plan."""
    if not plan.hayate_native:
        return False
    os.environ[_HAYATE_NATIVE_CONFIG_ENV] = json.dumps(
        plan.hayate_config, sort_keys=True
    )
    return _register_hayate_native_backend(logger)


def _install_engine_args_hook(logger) -> bool:
    """Wrap ``EngineArgs.create_engine_config`` to auto-apply the KV-comp plan.

    Unlike weight quantization (which vLLM discovers from
    ``quantization_config`` via ``register_quantization_config``), vLLM has *no*
    hook that reads a ``kv_compression_config`` and configures the KV cache.  So
    to make an OneComp FP8 model "just work" under a plain
    ``LLM(model=...)`` / ``vllm serve``, we idempotently wrap
    ``EngineArgs.create_engine_config`` and call :func:`apply_to_engine_args`
    just before the engine config is built -- this sets ``kv_cache_dtype`` (and
    ``calculate_kv_scales``) from the model's config.json.  Non-FP8 codecs are
    only *warned* about (never silently mis-applied).

    Returns ``True`` if the hook is installed (or already was).
    """
    global _PATCHED
    if _PATCHED:
        return True
    from vllm.engine.arg_utils import EngineArgs  # pylint: disable=import-outside-toplevel

    if getattr(EngineArgs, "_onecomp_kvcomp_patched", False):
        _PATCHED = True
        return True

    _orig_create = EngineArgs.create_engine_config

    def _patched_create(self, *args, **kwargs):
        try:
            apply_to_engine_args(self)
        except Exception as exc:  # pragma: no cover - never block engine start
            _get_logger().warning(
                "[kvcomp] auto-apply skipped (%s); call apply_to_engine_args() "
                "or engine_kwargs_from_kv_config() manually.", exc)
        return _orig_create(self, *args, **kwargs)

    _patched_create._onecomp_orig = _orig_create  # introspection / un-patch
    EngineArgs.create_engine_config = _patched_create
    EngineArgs._onecomp_kvcomp_patched = True
    _PATCHED = True
    return True


def register_vllm_plugin():
    """Entry point for ``vllm.general_plugins`` (called once per engine process).

    Installs a non-invasive ``EngineArgs`` hook that auto-applies the FP8 ->
    ``kv_cache_dtype`` mapping for models carrying a ``kv_compression_config``,
    so the FP8 path needs no manual launcher call.  If the hook cannot be
    installed (unexpected vLLM version), the plugin degrades gracefully: the
    explicit :func:`apply_to_engine_args` / :func:`engine_kwargs_from_kv_config`
    helpers still work.
    """
    logger = _get_logger()
    try:
        # EngineCore is often spawned in a fresh Python process.  An in-process
        # register_backend() call made by a launcher does not cross that spawn
        # boundary, but environment variables do.  If a launcher set the Hayate
        # native config before engine creation, re-install the override here so
        # child workers use Hayate block-VQ instead of default TurboQuant.
        _register_hayate_native_backend(logger)
        _install_engine_args_hook(logger)
        logger.info(
            "Registered OneComp KV compression plugin '%s' (FP8 and supported "
            "HayateQuant native block-VQ auto-applied to kv_cache_dtype; "
            "remaining low-bit/eviction recipes require the HF cache path or "
            "the experimental attention backend).",
            _PLUGIN_NAME,
        )
    except Exception as exc:  # pragma: no cover - vLLM internals changed
        logger.warning(
            "[kvcomp] could not install EngineArgs hook (%s); use "
            "apply_to_engine_args()/engine_kwargs_from_kv_config() manually.", exc)


__all__ = [
    "register_vllm_plugin",
    "read_kv_compression_config",
    "plan_for_model",
    "apply_to_engine_args",
    "engine_kwargs_from_kv_config",
]
