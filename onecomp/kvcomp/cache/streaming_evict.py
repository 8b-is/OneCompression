"""StreamingLLM-style eviction cache (the canonical eviction baseline).

Keeps the first ``sink`` attention-sink tokens and the most-recent ``recent``
tokens, dropping the middle.  This is the "evict" regime that the design memo
argues is more memory-efficient than uniform quantization: it *permanently
removes* mid-context tokens, so a needle in the dropped region is unrecoverable
-- the structural weakness that the proposed graded-quant TaperKV avoids.

Eviction is applied **after prefill** via :meth:`evict_now` (so the prefill
forward sees the full context and the attention mask stays consistent), then
decoding continues over the trimmed cache.  It is driven by the manual greedy
loop in :mod:`onecomp.kvcomp.eval.longcontext`, because HF ``generate`` cannot
trim the cache mid-forward without breaking its mask/position bookkeeping.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


class StreamingLLMEvictCache(DynamicCache):
    """Keep ``sink`` + last ``recent`` tokens; drop the middle (StreamingLLM).

    Args:
        sink: number of leading (attention-sink) tokens kept in full precision.
        recent: number of most-recent tokens kept in full precision.
        config: model config (preserves sliding-window layer typing).
    """

    def __init__(self, sink: int = 4, recent: int = 256, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        sliding_layers = [
            index for index, layer in enumerate(self.layers) if getattr(layer, "is_sliding", False)
        ]
        if sliding_layers:
            raise NotImplementedError(
                "StreamingLLMEvictCache cannot compose manual eviction with "
                f"model-side sliding-window layers {sliding_layers}"
            )
        if isinstance(sink, bool) or not isinstance(sink, int) or sink < 0:
            raise ValueError("sink must be a non-negative integer")
        if isinstance(recent, bool) or not isinstance(recent, int) or recent < 0:
            raise ValueError("recent must be a non-negative integer")
        self.sink = sink
        self.recent = recent
        self.kept = 0
        self.total = 0

    def reset(self):
        """Reset tensors and eviction accounting together."""
        super().reset()
        self.kept = 0
        self.total = 0

    def evict_now(self):
        """Trim every layer to ``sink`` + last ``recent`` tokens (idempotent)."""
        budget = self.sink + self.recent
        if not self.layers or getattr(self.layers[0], "keys", None) is None:
            return
        current = int(self.layers[0].keys.shape[2])
        # Before the first trim current is the whole prefill. Afterwards it is the
        # previous kept cache plus newly appended decode tokens.
        self.total += current if self.total == 0 else max(0, current - self.kept)
        for layer in self.layers:
            s = layer.keys.shape[2]
            if s > budget:
                idx = torch.cat(
                    [
                        torch.arange(self.sink, device=layer.keys.device),
                        torch.arange(s - self.recent, s, device=layer.keys.device),
                    ]
                )
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
        self.kept = int(self.layers[0].keys.shape[2])

    def teacher_forced_temporal_dependency(self):
        """Likelihood requires eviction between autoregressive target steps."""
        return "explicit_eviction_between_autoregressive_steps"

    @staticmethod
    def teacher_forced_forward_kwargs(cache_positions):
        """Preserve absolute RoPE/cache positions after physical eviction."""
        if cache_positions is None:
            raise ValueError("StreamingLLM teacher forcing requires absolute cache positions")
        return {
            "cache_position": cache_positions,
            "position_ids": cache_positions.unsqueeze(0),
        }

    def teacher_forced_post_forward(self):
        """Mirror generation: trim after prefill and after every decode step."""
        self.evict_now()

    def mem_fraction(self) -> float:
        return self.kept / max(1, self.total)

    def compression_ratio(self) -> float:
        return 1.0 / max(1e-9, self.mem_fraction())
