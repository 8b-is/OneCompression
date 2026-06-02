"""OneComp KV cache compression subpackage.

This subpackage adds *KV cache* compression as a second, orthogonal axis to
OneComp's existing *weight* quantization (:mod:`onecomp.quantizer`).

Unlike weight quantization, which statically transforms ``nn.Linear`` weights
offline, KV cache compression operates on the key/value activations produced at
attention time and is therefore applied dynamically at runtime.  The two axes
are designed to *coexist*: a single :class:`~onecomp.runner.Runner` run can
both weight-quantize a model and attach a KV compression recipe.

Public objects mirror the conventions of :mod:`onecomp.quantizer`:
``KVCodec`` / ``KVSelector`` / ``KVPolicy`` play the role of ``Quantizer``,
and methods are discoverable through a small registry.

Copyright 2025-2026 Fujitsu Ltd.
"""

from ._base import (
    CompressedKV,
    KVCodec,
    KVSelector,
    KVPolicy,
    ComposedPress,
)
from ._registry import (
    register_kv_codec,
    register_kv_selector,
    register_kv_policy,
    build_codec,
    build_selector,
    build_policy,
    list_codecs,
    list_selectors,
    list_policies,
)
from .config import (
    Budget,
    RequestState,
    SystemState,
    CompressionPlan,
    KVCompressionConfig,
)

# Importing the method modules triggers their registry decorators.
from . import codec as codec
from . import select as select
from . import prune as prune
from . import policy as policy
from . import recipe as recipe
from .cache import (
    CompressedDynamicCache,
    build_compressed_cache,
    kv_compressed_generate,
    ThinKVCache,
    thinkv_attention_capture,
    TaperKVCache,
    taper_bits,
    TurboTaperCache,
    PMKVQCache,
    StreamingLLMEvictCache,
    GradedQuantCache,
)
from .recipe import (
    TaperKVPolicy,
    TurboTaperPolicy,
    PMKVQPolicy,
    ThinKVPolicy,
    auto_configure_thinkv,
    detect_outlier_layers,
    calibrate_pmkvq,
    estimate_block_sensitivity,
    solve_block_allocation,
    calibrate_channel_reparam,
)

__all__ = [
    "CompressedKV",
    "KVCodec",
    "KVSelector",
    "KVPolicy",
    "ComposedPress",
    "register_kv_codec",
    "register_kv_selector",
    "register_kv_policy",
    "build_codec",
    "build_selector",
    "build_policy",
    "list_codecs",
    "list_selectors",
    "list_policies",
    "Budget",
    "RequestState",
    "SystemState",
    "CompressionPlan",
    "KVCompressionConfig",
    "CompressedDynamicCache",
    "build_compressed_cache",
    "kv_compressed_generate",
    "ThinKVCache",
    "thinkv_attention_capture",
    "TaperKVCache",
    "taper_bits",
    "TurboTaperCache",
    "TaperKVPolicy",
    "TurboTaperPolicy",
    "PMKVQCache",
    "PMKVQPolicy",
    "ThinKVPolicy",
    "auto_configure_thinkv",
    "detect_outlier_layers",
    "calibrate_pmkvq",
    "estimate_block_sensitivity",
    "solve_block_allocation",
    "calibrate_channel_reparam",
    "StreamingLLMEvictCache",
    "GradedQuantCache",
]
