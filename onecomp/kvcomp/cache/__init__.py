"""HuggingFace cache integration for KV compression.

Copyright 2025-2026 Fujitsu Ltd.
"""

from ._graded_base import GradedQuantCache, fake_quant_tokens
from .block_taper_cache import BlockTaperKVCache, block_taper_bits, fake_quant_blocks
from .builders import build_compressed_cache, kv_compressed_generate
from .compressed_cache import CompressedDynamicCache
from .pmkvq_cache import PMKVQCache
from .role_cache import RoleKVCache
from .streaming_evict import StreamingLLMEvictCache
from .taper_cache import (
    BudgetTaperKVCache,
    TaperKVCache,
    apply_byte_budget,
    step_down_bits,
    taper_bits,
)
from .thinkv_cache import ThinKVCache, thinkv_attention_capture

__all__ = [
    "CompressedDynamicCache",
    "build_compressed_cache",
    "kv_compressed_generate",
    "ThinKVCache",
    "thinkv_attention_capture",
    "GradedQuantCache",
    "fake_quant_tokens",
    "TaperKVCache",
    "taper_bits",
    "BudgetTaperKVCache",
    "apply_byte_budget",
    "step_down_bits",
    "BlockTaperKVCache",
    "block_taper_bits",
    "fake_quant_blocks",
    "PMKVQCache",
    "StreamingLLMEvictCache",
    "RoleKVCache",
]
