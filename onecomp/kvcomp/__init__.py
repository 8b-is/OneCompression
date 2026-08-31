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

# Importing the method modules triggers their registry decorators.
from . import codec as codec
from . import policy as policy
from . import prune as prune
from . import recipe as recipe
from . import select as select
from ._base import (
    ComposedPress,
    CompressedKV,
    KVCodec,
    KVPolicy,
    KVSelector,
)
from ._registry import (
    build_codec,
    build_policy,
    build_selector,
    list_codecs,
    list_policies,
    list_selectors,
    register_kv_codec,
    register_kv_policy,
    register_kv_selector,
)
from .cache import (
    BlockTaperKVCache,
    BudgetTaperKVCache,
    CompressedDynamicCache,
    GradedQuantCache,
    PMKVQCache,
    RoleKVCache,
    StreamingLLMEvictCache,
    TaperKVCache,
    ThinKVCache,
    apply_byte_budget,
    block_taper_bits,
    build_compressed_cache,
    fake_quant_blocks,
    kv_compressed_generate,
    step_down_bits,
    taper_bits,
    thinkv_attention_capture,
)
from .checkpoint import (
    KVCheckpointCodec,
    KVCheckpointConfig,
    KVCheckpointInfo,
    extract_kv_layers,
)
from .config import (
    Budget,
    CompressionPlan,
    KVCompressionConfig,
    RequestState,
    SystemState,
)
from .recipe import (
    BlockTaperKVPolicy,
    BudgetTaperKVPolicy,
    PMKVQPolicy,
    RolePrecisionPolicy,
    RoleSpan,
    TaperKVPolicy,
    ThinKVPolicy,
    annotate_spans_from_chat,
    auto_configure_thinkv,
    calibrate_channel_reparam,
    calibrate_pmkvq,
    detect_outlier_layers,
    estimate_block_sensitivity,
    solve_block_allocation,
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
    "BudgetTaperKVCache",
    "apply_byte_budget",
    "step_down_bits",
    "BlockTaperKVCache",
    "block_taper_bits",
    "fake_quant_blocks",
    "TaperKVPolicy",
    "BudgetTaperKVPolicy",
    "BlockTaperKVPolicy",
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
    "RoleKVCache",
    "RolePrecisionPolicy",
    "RoleSpan",
    "annotate_spans_from_chat",
    "KVCheckpointCodec",
    "KVCheckpointConfig",
    "KVCheckpointInfo",
    "extract_kv_layers",
]
