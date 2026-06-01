"""ThinK: query-driven head-dimension pruning.

ThinK prunes unimportant *channels* of the key cache (head dimension) based on
how much energy the recent queries place on each channel, complementary to
token eviction.  Reference: Xu et al., "ThinK: Thinner Key Cache by
Query-Driven Pruning" (arXiv:2407.21018).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import KVSelector
from .._registry import register_kv_selector
from ..select._common import group_query_to_kv


@register_kv_selector("think")
@dataclass
class ThinKChannelPruner(KVSelector):
    """Prune the least-important head-dim channels of the key cache.

    Unlike token selectors, ThinK returns a *channel* keep-mask of shape
    ``(batch, num_kv_heads, head_dim)``.  It is typically composed with a token
    selector (ThinKV uses SnapKV + ThinK).
    """

    flag_needs_attention: bool = True
    prune_ratio: float = 0.4  # fraction of channels to drop
    window_size: int = 32

    def channel_scores(self, query, key):
        """Per-channel importance: |q|*|k| energy over the observation window."""
        num_kv_heads = key.shape[1]
        q = group_query_to_kv(query, num_kv_heads)[:, :, -self.window_size :, :]
        q_energy = q.abs().sum(dim=2)  # (batch, kv_heads, head_dim)
        k_energy = key.abs().sum(dim=2)
        return q_energy * k_energy

    def score(self, query, key, value, attn_stats, layer_idx):
        return self.channel_scores(query, key)

    def select(self, scores, budget, seq_len):
        # scores: (batch, kv_heads, head_dim); returns channel keep-mask.
        batch, heads, head_dim = scores.shape
        keep = max(int(round((1.0 - self.prune_ratio) * head_dim)), 1)
        idx = scores.topk(keep, dim=-1).indices
        mask = torch.zeros(batch, heads, head_dim, dtype=torch.bool, device=scores.device)
        mask.scatter_(-1, idx, True)
        return mask
