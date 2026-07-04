"""HuggingFace cache integration for KV compression.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .compressed_cache import CompressedDynamicCache
from .builders import build_compressed_cache, kv_compressed_generate
from .thinkv_cache import ThinKVCache, thinkv_attention_capture
from ._graded_base import GradedQuantCache, fake_quant_tokens
from .taper_cache import TaperKVCache, taper_bits
from .turbotaper_cache import TurboTaperCache
from .pmkvq_cache import PMKVQCache
from .streaming_evict import StreamingLLMEvictCache
from .role_cache import RoleKVCache

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
    "TurboTaperCache",
    "PMKVQCache",
    "StreamingLLMEvictCache",
    "RoleKVCache",
]
