"""OneComp KV cache compression vLLM plugin.

Copyright 2025-2026 Fujitsu Ltd.
"""

from vllm_plugins.kvcomp.mapping import (
    KVCompVLLMPlan,
    engine_kwargs_from_kv_config,
    plan_from_kv_config,
)
from vllm_plugins.kvcomp.native_turboquant import (
    TURBOQUANT_PRESETS,
    disable_vllm_implicit_turboquant_boundary_skip,
    resolve_turboquant_preset,
    turboquant_engine_kwargs,
    vllm_implicit_turboquant_boundary_skip_disabled,
)
from vllm_plugins.kvcomp.vllm_plugin import (
    apply_to_engine_args,
    plan_for_model,
    read_kv_compression_config,
    register_vllm_plugin,
)

__all__ = [
    "KVCompVLLMPlan",
    "TURBOQUANT_PRESETS",
    "disable_vllm_implicit_turboquant_boundary_skip",
    "engine_kwargs_from_kv_config",
    "plan_from_kv_config",
    "resolve_turboquant_preset",
    "turboquant_engine_kwargs",
    "vllm_implicit_turboquant_boundary_skip_disabled",
    "apply_to_engine_args",
    "plan_for_model",
    "read_kv_compression_config",
    "register_vllm_plugin",
]
