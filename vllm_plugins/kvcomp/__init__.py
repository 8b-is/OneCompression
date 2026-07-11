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
    resolve_turboquant_preset,
    turboquant_engine_kwargs,
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
    "engine_kwargs_from_kv_config",
    "plan_from_kv_config",
    "resolve_turboquant_preset",
    "turboquant_engine_kwargs",
    "apply_to_engine_args",
    "plan_for_model",
    "read_kv_compression_config",
    "register_vllm_plugin",
]
