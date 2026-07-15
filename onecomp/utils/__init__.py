"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# Lazily loaded on first attribute access (PEP 562) instead of at package
# import time.  ``onecomp.utils`` is imported transitively by lightweight,
# framework-agnostic code (e.g. the MCP server's memory-estimation tools),
# which must not pay for the heavy eval/training stack (``lm_eval`` ->
# ``transformers``/``datasets``/``hqq`` -> ~15s cold-start) just to reach a
# handful of pure arithmetic helpers. Callers that already do
# ``from onecomp.utils import calculate_accuracy`` (etc.) are unaffected:
# the attribute is imported and cached transparently on first access.
from typing import Any

from .vram_estimator import (
    estimate_target_bitwidth,
    estimate_wbits_from_vram,
    effective_bits_per_param,
    raw_bits_for_quantizer,
    effective_bits_for_quantizer,
    weight_memory_gb,
    VRAMBitwidthEstimation,
)

from .dtype import needs_bfloat16

# calculate_perplexity also pulls in `datasets` (via its own module-level
# `from datasets import load_dataset`) on top of transformers/torch, so it
# is lazy too, same reasoning as the eval/training-only helpers below.
_LAZY_SUBMODULES = {
    "calculate_perplexity": "perplexity",
    "calculate_accuracy": "accuracy",
    "check_activations": "activation_check",
    "capture_input_activations": "activation_capture",
    "add_model_specific_inputs": "model_inputs",
    "get_blocks_and_inputs": "blockwise",
    "forward_input": "blockwise",
    "move_kwargs_to_device": "blockwise",
    "expand_kwargs_batch": "blockwise",
}

__all__ = [
    "calculate_perplexity",
    "calculate_accuracy",
    "check_activations",
    "capture_input_activations",
    "estimate_target_bitwidth",
    "estimate_wbits_from_vram",
    "effective_bits_per_param",
    "raw_bits_for_quantizer",
    "effective_bits_for_quantizer",
    "weight_memory_gb",
    "VRAMBitwidthEstimation",
    "add_model_specific_inputs",
    "get_blocks_and_inputs",
    "forward_input",
    "move_kwargs_to_device",
    "expand_kwargs_batch",
    "needs_bfloat16",
]


def __getattr__(name: str) -> Any:
    submodule_name = _LAZY_SUBMODULES.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
