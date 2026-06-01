"""User-facing helpers to attach KV compression to a HuggingFace model.

The integration is intentionally non-invasive: we never patch the model's
modules (unlike KVCache-Factory's global ``replace_llama``).  We build a
:class:`CompressedDynamicCache` and pass it to
``model.generate(past_key_values=...)``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger
from typing import Optional

from ..config import KVCompressionConfig
from .compressed_cache import CompressedDynamicCache

logger = getLogger(__name__)


def build_compressed_cache(model, kv_config: KVCompressionConfig,
                           sink_tokens: int = 4) -> CompressedDynamicCache:
    """Build a :class:`CompressedDynamicCache` for ``model`` from ``kv_config``.

    ``sink_tokens`` leading tokens are kept in full precision (StreamingLLM /
    KIVI), which is required for stable generation on outlier-heavy models.
    """
    config = getattr(model, "config", None)
    return CompressedDynamicCache(
        codec=kv_config.codec, config=config,
        skip_layers=kv_config.codec.skip_layers if kv_config.codec else None,
        sink_tokens=sink_tokens,
    )


def kv_compressed_generate(model, kv_config: Optional[KVCompressionConfig] = None, **gen_kwargs):
    """``model.generate`` wrapper that installs a compressed KV cache.

    Args:
        model: A HuggingFace causal LM.
        kv_config: The KV compression recipe (``None`` -> plain generate).
        **gen_kwargs: Forwarded to ``model.generate`` (``input_ids``, ...).

    Returns:
        The output of ``model.generate``.
    """
    if kv_config is not None and kv_config.codec is not None:
        if "past_key_values" in gen_kwargs and gen_kwargs["past_key_values"] is not None:
            logger.warning("Overriding provided past_key_values with CompressedDynamicCache")
        gen_kwargs["past_key_values"] = build_compressed_cache(model, kv_config)
    return model.generate(**gen_kwargs)
