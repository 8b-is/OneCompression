"""SnapKV token selector.

SnapKV scores prompt tokens by how much the last ``window_size`` "observation"
queries attend to them, pools the scores (1-D max/avg pool) to keep
contiguous spans, then retains the top-k plus a recent window.

Reference: Li et al., "SnapKV: LLM Knows What You are Looking for Before
Generation" (arXiv:2404.14469).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .._base import KVSelector
from .._registry import register_kv_selector
from ._common import observation_attention, mask_from_keep_indices


@register_kv_selector("snapkv")
@dataclass
class SnapKVSelector(KVSelector):
    """SnapKV prompt-cache compression."""

    flag_needs_attention: bool = True

    keep_ratio: float = 0.5
    keep_tokens: int = None
    window_size: int = 32
    kernel_size: int = 7
    pooling: str = "avgpool"
    sink_tokens: int = 0
    recent_tokens: int = 32

    def validate_params(self):
        if self.pooling not in ("avgpool", "maxpool"):
            raise ValueError("pooling must be 'avgpool' or 'maxpool'")

    def score(self, query, key, value, attn_stats, layer_idx):
        # attn_stats may carry precomputed observation attention; otherwise
        # compute it from query/key directly (research layer).
        if attn_stats is not None and "obs_attn" in attn_stats:
            obs = attn_stats["obs_attn"]
        else:
            obs = observation_attention(query, key, self.window_size)
        # Sum the attention each prompt token receives from the observation window.
        return obs.sum(dim=2)  # (batch, num_kv_heads, seq_len)

    def select(self, scores, budget, seq_len):
        batch, heads, _ = scores.shape
        keep = budget.resolve_keep(seq_len)
        recent = min(max(budget.recent_tokens, self.recent_tokens), seq_len)
        sink = min(max(budget.sink_tokens, self.sink_tokens), seq_len)

        # Do not let the recent window participate in top-k scoring.
        prompt_len = max(seq_len - recent, 0)
        # Reserve the sink + recent windows out of the budget.
        n_topk = max(keep - recent - sink, 0)

        scores = scores.clone()
        if prompt_len > 0 and n_topk > 0:
            pooled = scores[:, :, :prompt_len]
            # 1-D pooling to keep contiguous, informative spans.
            pad = self.kernel_size // 2
            if self.pooling == "avgpool":
                pooled = F.avg_pool1d(pooled, kernel_size=self.kernel_size, padding=pad, stride=1)
            else:
                pooled = F.max_pool1d(pooled, kernel_size=self.kernel_size, padding=pad, stride=1)
            pooled = pooled[:, :, :prompt_len]
            topk = min(n_topk, prompt_len)
            idx = pooled.topk(topk, dim=-1).indices
        else:
            idx = torch.empty(batch, heads, 0, dtype=torch.long, device=scores.device)

        mask = mask_from_keep_indices(idx, seq_len) if idx.numel() else torch.zeros(
            batch, heads, seq_len, dtype=torch.bool, device=scores.device
        )
        if recent > 0:
            mask[:, :, -recent:] = True
        if sink > 0:
            mask[:, :, :sink] = True
        return mask
