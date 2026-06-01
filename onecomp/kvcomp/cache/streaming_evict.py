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
        self.sink = int(sink)
        self.recent = int(recent)
        self.kept = 0
        self.total = 0

    def evict_now(self):
        """Trim every layer to ``sink`` + last ``recent`` tokens (idempotent)."""
        budget = self.sink + self.recent
        for layer in self.layers:
            s = layer.keys.shape[2]
            self.total = max(self.total, s)
            if s > budget:
                idx = torch.cat([
                    torch.arange(self.sink, device=layer.keys.device),
                    torch.arange(s - self.recent, s, device=layer.keys.device),
                ])
                layer.keys = layer.keys.index_select(2, idx).contiguous()
                layer.values = layer.values.index_select(2, idx).contiguous()
            self.kept = layer.keys.shape[2]

    def mem_fraction(self) -> float:
        return self.kept / max(1, self.total)

    def compression_ratio(self) -> float:
        return 1.0 / max(1e-9, self.mem_fraction())
