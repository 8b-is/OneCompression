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


def build_compressed_cache(
    model,
    kv_config: KVCompressionConfig,
    sink_tokens: int = 4,
    recent_tokens: Optional[int] = None,
    skip_sliding_layers: bool = False,
) -> CompressedDynamicCache:
    """Build a :class:`CompressedDynamicCache` for ``model`` from ``kv_config``.

    ``sink_tokens`` leading tokens are kept in full precision (StreamingLLM /
    KIVI), which is required for stable generation on outlier-heavy models.
    ``recent_tokens`` trailing tokens are kept in full precision for
    prefill-sized updates.
    ``skip_sliding_layers=True`` preserves model sliding-window semantics while
    keeping those layers full precision with exact resident accounting.
    """
    policy = kv_config.policy
    if policy is not None:
        from ..recipe.taperkv import (  # pylint: disable=import-outside-toplevel
            BlockTaperKVPolicy,
            BudgetTaperKVPolicy,
            TaperKVPolicy,
        )
        from ..recipe.turbotaper import (  # pylint: disable=import-outside-toplevel
            TurboTaperPolicy,
        )

        if isinstance(policy, TaperKVPolicy):
            if kv_config.selector is not None:
                raise NotImplementedError(
                    "TapeKV cannot be combined with token eviction in the HF cache path"
                )
            from .block_taper_cache import (  # pylint: disable=import-outside-toplevel
                BlockTaperKVCache,
            )
            from .hayate_taper_cache import (  # pylint: disable=import-outside-toplevel
                HayateBlockTaperKVCache,
                HayateTaperKVCache,
            )
            from .taper_cache import (  # pylint: disable=import-outside-toplevel
                BudgetTaperKVCache,
                TaperKVCache,
            )
            from .turbotaper_cache import (  # pylint: disable=import-outside-toplevel
                TurboTaperCache,
            )

            model_config = getattr(model, "config", None)
            codec = kv_config.codec
            if codec is None:
                if isinstance(policy, TurboTaperPolicy):
                    cache_cls = TurboTaperCache
                elif isinstance(policy, BudgetTaperKVPolicy):
                    cache_cls = BudgetTaperKVCache
                elif isinstance(policy, BlockTaperKVPolicy):
                    cache_cls = BlockTaperKVCache
                else:
                    cache_cls = TaperKVCache
                return cache_cls(policy, config=model_config)
            if isinstance(policy, (BudgetTaperKVPolicy, TurboTaperPolicy)):
                raise NotImplementedError(
                    f"Policy {policy.name!r} cannot be composed with a separate codec"
                )
            if getattr(codec, "name", None) != "hayatequant":
                raise NotImplementedError(
                    f"TapeKV HF composition does not support codec {codec.name!r}; "
                    "only the explicitly defined HayateQuant-then-TapeKV path is available"
                )
            cache_cls = (
                HayateBlockTaperKVCache
                if isinstance(policy, BlockTaperKVPolicy)
                else HayateTaperKVCache
            )
            return cache_cls(policy, codec, config=model_config)

    plan = policy.plan() if policy is not None else None
    codec = plan.codec if plan is not None and plan.codec is not None else kv_config.codec
    selector = (
        plan.selector if plan is not None and plan.selector is not None else kv_config.selector
    )
    if selector is not None:
        # Attention-aware selectors need query/attention integration and eviction
        # must run during decode. A plain DynamicCache passed to HF generate cannot
        # do either. Fail loudly instead of silently producing a mislabeled FullKV
        # or quant-only run.
        raise NotImplementedError(
            f"Selector {selector.name!r} is not supported by kv_compressed_generate; "
            "use an attention-integrated/native cache backend."
        )
    config = getattr(model, "config", None)
    if recent_tokens is None:
        recent_tokens = (
            plan.residual_window
            if plan is not None and plan.residual_window is not None
            else kv_config.residual_window
        )
    return CompressedDynamicCache(
        codec=codec,
        config=config,
        skip_layers=(
            plan.skip_layers
            if plan is not None and plan.skip_layers
            else (codec.skip_layers if codec else None)
        ),
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        skip_sliding_layers=skip_sliding_layers,
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
    if kv_config is not None and any(
        component is not None
        for component in (kv_config.codec, kv_config.selector, kv_config.policy)
    ):
        if "past_key_values" in gen_kwargs and gen_kwargs["past_key_values"] is not None:
            logger.warning("Overriding provided past_key_values with CompressedDynamicCache")
        gen_kwargs["past_key_values"] = build_compressed_cache(model, kv_config)
    return model.generate(**gen_kwargs)
