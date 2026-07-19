"""Compatibility aliases for the renamed HayateQuant--TaperKV cache module.

Use :mod:`onecomp.kvcomp.cache.hayatequant_taper_cache` in new code.  The old
module and class names remain importable for saved scripts.
"""

from .hayatequant_taper_cache import (
    HayateQuantBlockTaperKVCache,
    HayateQuantTaperKVCache,
)

HayateTaperKVCache = HayateQuantTaperKVCache
HayateBlockTaperKVCache = HayateQuantBlockTaperKVCache

__all__ = [
    "HayateQuantTaperKVCache",
    "HayateQuantBlockTaperKVCache",
    "HayateTaperKVCache",
    "HayateBlockTaperKVCache",
]
