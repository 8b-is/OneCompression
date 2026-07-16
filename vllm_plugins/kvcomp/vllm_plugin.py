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
* For low-bit / eviction recipes, fail closed when the experimental custom
  attention backend is required (never start an uncompressed look-alike run).

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
_HAYATE_FULL_NATIVE_CONFIG_ENV = "KVCOMP_VLLM_HAYATE_FULL_NATIVE_CONFIG"


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
            model_path,
        )
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

    for modname, attr in (("vllm.config.cache", "CacheDType"), ("vllm.config", "CacheDType")):
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
    # The TURBOQUANT registry override is process-global, while the selected
    # codec is per model/engine. Remove a prior Hayate payload when a serialized
    # model explicitly selects any other codec (including scalar fallback), so
    # the override delegates to upstream TurboQuant instead of reusing stale
    # block-VQ settings. A model with no kv_compression_config is left alone:
    # the standalone native smoke/generation harness configures the override
    # manually and intentionally has no serialized block.
    if plan.codec_name is not None and not plan.hayate_native:
        os.environ.pop(_HAYATE_NATIVE_CONFIG_ENV, None)
        # A manually requested full-HQ backend is likewise scoped to its fresh
        # engine process.  Never let its process-global TURBOQUANT override
        # leak into a subsequently serialized non-HQ model.
        os.environ.pop(_HAYATE_FULL_NATIVE_CONFIG_ENV, None)
    elif plan.hayate_native:
        # A serialized historical hybrid is an explicit, different backend;
        # it must not coexist with a manual full-HQ override.
        os.environ.pop(_HAYATE_FULL_NATIVE_CONFIG_ENV, None)
    for msg in plan.notes:
        logger.info("[kvcomp] %s", msg)
    for msg in plan.warnings:
        logger.warning("[kvcomp] %s", msg)
    if plan.needs_custom_backend or not plan.native_supported:
        detail = "; ".join(plan.notes + plan.warnings) or "no faithful native mapping"
        raise RuntimeError(
            "Serialized KV compression recipe cannot run through native vLLM: " f"{detail}"
        )
    if plan.native_supported and plan.kv_cache_dtype != "auto":
        if not _vllm_accepts_dtype(plan.kv_cache_dtype):
            raise RuntimeError(
                "Installed vLLM does not support the required kv_cache_dtype="
                f"{plan.kv_cache_dtype!r}; upgrade vLLM or use the faithful HF "
                "cache path"
            )
        if plan.hayate_native:
            _install_hayate_native_backend(plan, logger)
        engine_args.kv_cache_dtype = plan.kv_cache_dtype
        if plan.calculate_kv_scales and hasattr(engine_args, "calculate_kv_scales"):
            engine_args.calculate_kv_scales = True
        if plan.skip_layers and hasattr(engine_args, "kv_cache_dtype_skip_layers"):
            engine_args.kv_cache_dtype_skip_layers = ",".join(str(i) for i in plan.skip_layers)
            logger.info(
                "[kvcomp] set kv_cache_dtype_skip_layers=%s",
                engine_args.kv_cache_dtype_skip_layers,
            )
        logger.info("[kvcomp] set kv_cache_dtype=%s", plan.kv_cache_dtype)
    return engine_args


_PATCHED = False
_HAYATE_NATIVE_INSTALLED = False
_HAYATE_FULL_NATIVE_INSTALLED = False
_HAYATE_FULL_SPEC_PATCHED = False


def _install_hayate_full_cache_spec_hook() -> bool:
    """Make vLLM budget the same full-HQ slot width it later reshapes.

    vLLM 0.20 computes ``TQFullAttentionSpec.tq_slot_size`` directly from the
    named upstream TurboQuant preset, independently of the registered
    backend's ``get_kv_cache_shape``.  Full HQ deliberately has a different
    physical slot (for example 52 rather than 54 bytes at D=64, rate 3), so
    overriding shape alone would allocate a TQ-sized raw buffer and fail its
    final view.  Patch only the explicit full-HQ process and replace that one
    spec field; normal TQ and hybrid paths call the original method unchanged.
    """

    global _HAYATE_FULL_SPEC_PATCHED
    if not os.environ.get(_HAYATE_FULL_NATIVE_CONFIG_ENV):
        return False
    if _HAYATE_FULL_SPEC_PATCHED:
        return True
    from dataclasses import replace  # pylint: disable=import-outside-toplevel
    from vllm.model_executor.layers.attention.attention import (  # pylint: disable=import-outside-toplevel
        Attention,
    )
    from vllm.v1.kv_cache_interface import (  # pylint: disable=import-outside-toplevel
        TQFullAttentionSpec,
    )
    from vllm_plugins.kvcomp.hayate_full_config import (  # pylint: disable=import-outside-toplevel
        full_hq_native_recipe_from_mapping,
    )

    original = Attention.get_kv_cache_spec

    def _full_hq_spec(self, vllm_config):
        spec = original(self, vllm_config)
        raw = os.environ.get(_HAYATE_FULL_NATIVE_CONFIG_ENV)
        if not raw:
            return spec
        recipe = full_hq_native_recipe_from_mapping(json.loads(raw))
        # vLLM's boundary-layer protection intentionally leaves the first and
        # last layers uncompressed.  Those layers return the ordinary full
        # attention spec and must keep their upstream allocation unchanged;
        # only the layers for which vLLM actually selected its TurboQuant
        # data plane need the HQ physical slot substitution.
        if not isinstance(spec, TQFullAttentionSpec):
            return spec
        slot = recipe.layout(self.head_size)["slot_size_aligned"]
        return replace(spec, tq_slot_size=slot)

    _full_hq_spec._onecomp_full_hq_original = original
    Attention.get_kv_cache_spec = _full_hq_spec
    _HAYATE_FULL_SPEC_PATCHED = True
    return True


def _register_hayate_full_native_backend(logger) -> bool:
    """Re-register explicit full HQ in vLLM spawn children.

    Registry state does not cross vLLM's spawned EngineCore boundary, while
    environment variables do.  This hook is intentionally inert unless the
    dedicated full-HQ environment payload is present; normal upstream
    TurboQuant and serialized mapping behavior are unchanged.
    """

    global _HAYATE_FULL_NATIVE_INSTALLED
    payload = os.environ.get(_HAYATE_FULL_NATIVE_CONFIG_ENV)
    if not payload:
        return False
    if os.environ.get(_HAYATE_NATIVE_CONFIG_ENV):
        raise RuntimeError(
            "both full-HQ and historical hybrid Hayate native configs are set; "
            "refusing ambiguous TURBOQUANT backend registration"
        )
    # Validate before mutating the process-global registry.  A malformed or
    # partial recipe must not silently start upstream TQ under an HQ label.
    try:
        config = json.loads(payload)
        if not isinstance(config, dict):
            raise ValueError("full HQ native payload must be a JSON object")
        from vllm_plugins.kvcomp.hayate_full_config import (  # pylint: disable=import-outside-toplevel
            full_hq_native_recipe_from_mapping,
        )

        full_hq_native_recipe_from_mapping(config)
    except Exception as exc:
        raise RuntimeError(
            "invalid explicit full-HQ native config; refusing registration"
        ) from exc
    if _HAYATE_FULL_NATIVE_INSTALLED:
        return True
    try:
        from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.TURBOQUANT,
            "vllm_plugins.kvcomp.hayate_full_native.FullHayateQuantNativeBackend",
        )
        _HAYATE_FULL_NATIVE_INSTALLED = True
        logger.info(
            "[kvcomp] registered explicit faithful packed full-HQ backend "
            "for vLLM TURBOQUANT attention."
        )
        return True
    except Exception as exc:  # pragma: no cover - vLLM absent/version dependent
        raise RuntimeError(
            "Could not register requested full-HQ native backend; refusing "
            "to fall back to upstream TurboQuant"
        ) from exc


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
        raise RuntimeError(
            "Could not register the requested native HayateQuant backend; "
            "refusing to run the byte-compatible TurboQuant preset without "
            "block-VQ"
        ) from exc


def _install_hayate_native_backend(plan: KVCompVLLMPlan, logger) -> bool:
    """Install OneComp's native HayateQuant backend from a resolved plan."""
    if not plan.hayate_native:
        return False
    os.environ[_HAYATE_NATIVE_CONFIG_ENV] = json.dumps(plan.hayate_config, sort_keys=True)
    if not _register_hayate_native_backend(logger):
        raise RuntimeError("HayateQuant native plan did not install its required backend")
    return True


def _install_engine_args_hook(logger) -> bool:
    """Wrap ``EngineArgs.create_engine_config`` to auto-apply the KV-comp plan.

    Unlike weight quantization (which vLLM discovers from
    ``quantization_config`` via ``register_quantization_config``), vLLM has *no*
    hook that reads a ``kv_compression_config`` and configures the KV cache.  So
    to make an OneComp FP8 model "just work" under a plain
    ``LLM(model=...)`` / ``vllm serve``, we idempotently wrap
    ``EngineArgs.create_engine_config`` and call :func:`apply_to_engine_args`
    just before the engine config is built -- this sets ``kv_cache_dtype`` (and
    ``calculate_kv_scales``) from the model's config.json. Recipes without a
    faithful native representation raise before engine construction.

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
        apply_to_engine_args(self)
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
    so the FP8 path needs no manual launcher call. If the hook cannot be
    installed, registration fails visibly instead of leaving a compressed
    model on an uncompressed or mismatched cache path.
    """
    logger = _get_logger()
    try:
        # EngineCore is often spawned in a fresh Python process.  An in-process
        # register_backend() call made by a launcher does not cross that spawn
        # boundary, but environment variables do.  If a launcher set the Hayate
        # native config before engine creation, re-install the override here so
        # child workers use Hayate block-VQ instead of default TurboQuant.
        if _register_hayate_full_native_backend(logger):
            _install_hayate_full_cache_spec_hook()
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
        logger.error("[kvcomp] could not install required EngineArgs hook: %s", exc)
        raise


__all__ = [
    "register_vllm_plugin",
    "read_kv_compression_config",
    "plan_for_model",
    "apply_to_engine_args",
    "engine_kwargs_from_kv_config",
]
