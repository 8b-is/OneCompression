"""Position-based selectors: Recent, Sink, StreamingLLM.

These need no attention scores (``flag_needs_attention=False``) and keep tokens
purely by position.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import KVSelector
from .._registry import register_kv_selector


def _streaming_mask(batch, heads, seq_len, sink, recent, device):
    mask = torch.zeros(batch, heads, seq_len, dtype=torch.bool, device=device)
    s = min(sink, seq_len)
    r = min(recent, seq_len)
    if s > 0:
        mask[:, :, :s] = True
    if r > 0:
        mask[:, :, -r:] = True
    return mask


@register_kv_selector("streamingllm")
@dataclass
class StreamingLLMSelector(KVSelector):
    """StreamingLLM: keep ``sink`` leading tokens + ``recent`` trailing tokens."""

    flag_needs_attention: bool = False

    def score(self, query, key, value, attn_stats, layer_idx):
        batch, heads, seq_len, _ = key.shape
        # Position score: 1.0 inside the sink/recent windows handled by select().
        return torch.zeros(batch, heads, seq_len, device=key.device)

    def select(self, scores, budget, seq_len):
        batch, heads, _ = scores.shape
        return _streaming_mask(
            batch, heads, seq_len, budget.sink_tokens, budget.recent_tokens, scores.device
        )


@register_kv_selector("recent")
@dataclass
class RecentSelector(KVSelector):
    """Keep only the most-recent tokens (sliding window)."""

    flag_needs_attention: bool = False

    def score(self, query, key, value, attn_stats, layer_idx):
        batch, heads, seq_len, _ = key.shape
        return torch.zeros(batch, heads, seq_len, device=key.device)

    def select(self, scores, budget, seq_len):
        batch, heads, _ = scores.shape
        keep = budget.resolve_keep(seq_len)
        mask = torch.zeros(batch, heads, seq_len, dtype=torch.bool, device=scores.device)
        mask[:, :, -keep:] = True
        return mask
