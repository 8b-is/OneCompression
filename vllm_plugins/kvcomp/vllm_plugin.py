"""vLLM general plugin for OneComp KV-cache compression.

The plugin reads kv_compression_config from a local model, validates it through
vllm_plugins.kvcomp.mapping, and installs the required data plane.
HayateQuant uses a packed K/V backend; TaperKV uses a standalone GPU semantic
backend; recipes without an implemented faithful path fail before engine start.
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
_HAYATEQUANT_CONFIG_ENV = "KVCOMP_VLLM_HAYATEQUANT_CONFIG"
_TAPER_CONFIG_ENV = "KVCOMP_VLLM_TAPER_CONFIG"
_TAPER_EXPLICIT_ENV = "KVCOMP_VLLM_TAPER_CONFIG_EXPLICIT"


def _get_logger():
    try:
        from vllm.logger import init_logger  # pylint: disable=import-outside-toplevel

        return init_logger(__name__)
    except Exception:  # pragma: no cover - vLLM is optional
        from logging import getLogger  # pylint: disable=import-outside-toplevel

        return getLogger(__name__)


def read_kv_compression_config(model_path: str) -> Optional[dict]:
    """Read a local model's serialized KV recipe."""

    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        _get_logger().info(
            "[kvcomp] no local config.json at %r; no serialized KV recipe read",
            model_path,
        )
        return None
    with open(cfg_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("kv_compression_config")


def plan_for_model(
    model_path: str,
    requested_kv_cache_dtype: str = "auto",
) -> KVCompVLLMPlan:
    return plan_from_kv_config(
        read_kv_compression_config(model_path),
        requested_kv_cache_dtype,
    )


def _vllm_accepts_dtype(dtype: Optional[str]) -> bool:
    if dtype in (None, "auto"):
        return True
    import typing  # pylint: disable=import-outside-toplevel

    for module_name, attribute in (
        ("vllm.config.cache", "CacheDType"),
        ("vllm.config", "CacheDType"),
    ):
        try:
            module = __import__(module_name, fromlist=[attribute])
            choices = set(typing.get_args(getattr(module, attribute)))
            if choices:
                return dtype in choices
        except Exception:  # pragma: no cover - version dependent
            continue
    return True


def apply_to_engine_args(engine_args, model_path: Optional[str] = None):
    """Apply a model's validated KV recipe to vLLM EngineArgs."""

    logger = _get_logger()
    path = model_path or getattr(engine_args, "model", None)
    if not path:
        return engine_args
    requested = getattr(engine_args, "kv_cache_dtype", "auto") or "auto"
    plan = plan_for_model(path, requested)

    # Registry overrides are process-global. A concrete non-Hayate model clears
    # stale state, while a missing model recipe leaves an explicit launcher
    # payload intact.
    if plan.codec_name is not None and not plan.hayatequant_native:
        os.environ.pop(_HAYATEQUANT_CONFIG_ENV, None)
    if not plan.taper_backend and not os.environ.get(_TAPER_EXPLICIT_ENV):
        os.environ.pop(_TAPER_CONFIG_ENV, None)

    for message in plan.notes:
        logger.info("[kvcomp] %s", message)
    for message in plan.warnings:
        logger.warning("[kvcomp] %s", message)
    if plan.needs_custom_backend or not plan.native_supported:
        detail = "; ".join(plan.notes + plan.warnings) or "no faithful vLLM mapping"
        raise RuntimeError(f"Serialized KV compression recipe cannot run through vLLM: {detail}")

    if plan.taper_backend:
        _install_taper_backend(plan, logger)
        _apply_taper_engine_contract(engine_args, plan.taper_config)
        logger.info(
            "[kvcomp] enabled standalone TaperKV GPU semantic backend "
            "with graph-safe token decode and prefix caching disabled"
        )
        return engine_args

    if plan.kv_cache_dtype != "auto":
        if plan.kv_cache_dtype.startswith("turboquant_"):
            from vllm_plugins.kvcomp.native_turboquant import (  # pylint: disable=import-outside-toplevel
                disable_vllm_implicit_turboquant_boundary_skip,
            )

            disable_vllm_implicit_turboquant_boundary_skip()
        if not _vllm_accepts_dtype(plan.kv_cache_dtype):
            raise RuntimeError(
                "Installed vLLM does not support required kv_cache_dtype="
                f"{plan.kv_cache_dtype!r}"
            )
        if plan.hayatequant_native:
            _install_hayatequant_backend(plan, logger)
        engine_args.kv_cache_dtype = plan.kv_cache_dtype
        if plan.calculate_kv_scales and hasattr(engine_args, "calculate_kv_scales"):
            engine_args.calculate_kv_scales = True
        if plan.skip_layers and hasattr(engine_args, "kv_cache_dtype_skip_layers"):
            engine_args.kv_cache_dtype_skip_layers = ",".join(
                str(index) for index in plan.skip_layers
            )
        logger.info("[kvcomp] set kv_cache_dtype=%s", plan.kv_cache_dtype)
    return engine_args


_PATCHED = False
_HAYATEQUANT_INSTALLED = False
_HAYATEQUANT_SPEC_PATCHED = False
_TAPER_INSTALLED = False


def _apply_taper_engine_contract(engine_args, taper_config: dict | None = None) -> None:
    """Apply execution constraints required by TaperKV's mutable age state."""

    # Token-granularity TaperKV derives cadence from replay-updated seq_lens and
    # uses fixed-shape masked kernels, so CUDA graphs are safe.  Whole-page
    # block regrading still has a Python cadence and remains eager-only.
    is_block = bool(
        taper_config
        and taper_config.get("policy_name") == "block_taperkv"
        and taper_config.get("quant_granularity", "block") == "block"
    )
    if is_block:
        compilation_config = getattr(engine_args, "compilation_config", None)
        if compilation_config is None:
            raise RuntimeError("installed vLLM cannot disable CUDA graphs for block TaperKV")
        if isinstance(compilation_config, dict):
            compilation_config["cudagraph_mode"] = "NONE"
        else:
            from vllm.config.compilation import (  # pylint: disable=import-outside-toplevel
                CUDAGraphMode,
            )

            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if not hasattr(engine_args, "enable_prefix_caching"):
        raise RuntimeError(
            "installed vLLM cannot disable prefix sharing for mutable TaperKV pages"
        )
    engine_args.enable_prefix_caching = False
    if hasattr(engine_args, "attention_backend"):
        engine_args.attention_backend = "FLASH_ATTN"


def _install_hayatequant_cache_spec_hook() -> bool:
    """Make vLLM allocate HayateQuant's actual packed slot width."""

    global _HAYATEQUANT_SPEC_PATCHED
    if not os.environ.get(_HAYATEQUANT_CONFIG_ENV):
        return False
    if _HAYATEQUANT_SPEC_PATCHED:
        return True

    from dataclasses import replace  # pylint: disable=import-outside-toplevel
    from vllm.model_executor.layers.attention.attention import (  # pylint: disable=import-outside-toplevel
        Attention,
    )
    from vllm.v1.kv_cache_interface import (  # pylint: disable=import-outside-toplevel
        TQFullAttentionSpec,
    )
    from vllm_plugins.kvcomp.hayate_config import (  # pylint: disable=import-outside-toplevel
        hayatequant_native_recipe_from_mapping,
    )

    original = Attention.get_kv_cache_spec

    def _hayatequant_spec(self, vllm_config):
        spec = original(self, vllm_config)
        raw = os.environ.get(_HAYATEQUANT_CONFIG_ENV)
        if not raw or not isinstance(spec, TQFullAttentionSpec):
            return spec
        recipe = hayatequant_native_recipe_from_mapping(json.loads(raw))
        slot_size = recipe.layout(self.head_size)["slot_size_aligned"]
        return replace(spec, tq_slot_size=slot_size)

    _hayatequant_spec._onecomp_hayatequant_original = original
    Attention.get_kv_cache_spec = _hayatequant_spec
    _HAYATEQUANT_SPEC_PATCHED = True
    return True


def _register_hayatequant_backend(logger) -> bool:
    """Validate and register HayateQuant in this process or spawn child."""

    global _HAYATEQUANT_INSTALLED
    payload = os.environ.get(_HAYATEQUANT_CONFIG_ENV)
    if not payload:
        return False
    try:
        config = json.loads(payload)
        if not isinstance(config, dict):
            raise ValueError("HayateQuant payload must be a JSON object")
        from vllm_plugins.kvcomp.hayate_config import (  # pylint: disable=import-outside-toplevel
            hayatequant_native_recipe_from_mapping,
        )

        hayatequant_native_recipe_from_mapping(config)
    except Exception as exc:
        raise RuntimeError("invalid HayateQuant native config") from exc
    if _HAYATEQUANT_INSTALLED:
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
    except Exception as exc:  # pragma: no cover - vLLM version dependent
        raise RuntimeError("could not register HayateQuant vLLM backend") from exc
    _HAYATEQUANT_INSTALLED = True
    logger.info("[kvcomp] registered packed HayateQuant backend")
    return True


def _install_hayatequant_backend(plan: KVCompVLLMPlan, logger) -> bool:
    if not plan.hayatequant_native:
        return False
    from vllm_plugins.kvcomp.native_turboquant import (  # pylint: disable=import-outside-toplevel
        disable_vllm_implicit_turboquant_boundary_skip,
    )

    disable_vllm_implicit_turboquant_boundary_skip()
    os.environ[_HAYATEQUANT_CONFIG_ENV] = json.dumps(
        plan.hayatequant_config,
        sort_keys=True,
    )
    if not _register_hayatequant_backend(logger):
        raise RuntimeError("HayateQuant plan did not register its backend")
    if not _install_hayatequant_cache_spec_hook():
        raise RuntimeError("HayateQuant plan did not install its cache-spec hook")
    return True


def _register_taper_backend(logger) -> bool:
    """Validate and register standalone TaperKV in this process."""

    global _TAPER_INSTALLED
    payload = os.environ.get(_TAPER_CONFIG_ENV)
    if not payload:
        return False
    try:
        config = json.loads(payload)
        from vllm_plugins.kvcomp.taper_config import (  # pylint: disable=import-outside-toplevel
            TaperKVVLLMRecipe,
        )

        TaperKVVLLMRecipe.from_mapping(config)
    except Exception as exc:
        raise RuntimeError("invalid TaperKV vLLM config") from exc
    if _TAPER_INSTALLED:
        return True
    try:
        from vllm.v1.attention.backends.registry import (  # pylint: disable=import-outside-toplevel
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.FLASH_ATTN,
            "vllm_plugins.kvcomp.taper_backend.TaperKVBackend",
        )
    except Exception as exc:  # pragma: no cover - vLLM version dependent
        raise RuntimeError("could not register TaperKV vLLM backend") from exc
    _TAPER_INSTALLED = True
    logger.info("[kvcomp] registered standalone TaperKV GPU semantic backend")
    return True


def _install_taper_backend(plan: KVCompVLLMPlan, logger) -> bool:
    if not plan.taper_backend:
        return False
    os.environ[_TAPER_CONFIG_ENV] = json.dumps(plan.taper_config, sort_keys=True)
    if not _register_taper_backend(logger):
        raise RuntimeError("TaperKV plan did not register its backend")
    return True


def _install_engine_args_hook(logger) -> bool:
    """Auto-apply serialized recipes before vLLM builds its engine config."""

    del logger
    global _PATCHED
    if _PATCHED:
        return True
    from vllm.engine.arg_utils import EngineArgs  # pylint: disable=import-outside-toplevel

    if getattr(EngineArgs, "_onecomp_kvcomp_patched", False):
        _PATCHED = True
        return True
    original = EngineArgs.create_engine_config

    def _patched_create(self, *args, **kwargs):
        apply_to_engine_args(self)
        return original(self, *args, **kwargs)

    _patched_create._onecomp_orig = original
    EngineArgs.create_engine_config = _patched_create
    EngineArgs._onecomp_kvcomp_patched = True
    _PATCHED = True
    return True


def register_vllm_plugin():
    """Entry point registered under vllm.general_plugins."""

    logger = _get_logger()
    try:
        if _register_hayatequant_backend(logger):
            _install_hayatequant_cache_spec_hook()
        _register_taper_backend(logger)
        _install_engine_args_hook(logger)
        logger.info("Registered OneComp KV compression plugin %r", _PLUGIN_NAME)
    except Exception as exc:  # pragma: no cover - vLLM internals changed
        logger.error("[kvcomp] plugin registration failed: %s", exc)
        raise


__all__ = [
    "register_vllm_plugin",
    "read_kv_compression_config",
    "plan_for_model",
    "apply_to_engine_args",
    "engine_kwargs_from_kv_config",
]
