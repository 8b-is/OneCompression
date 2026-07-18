"""H2O (Heavy-Hitter Oracle) token selector.

H2O keeps the "heavy hitter" tokens that accumulate the most attention mass,
plus a recent window.  Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for
Efficient Generative Inference of Large Language Models" (arXiv:2306.14048).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .._base import KVSelector
from .._registry import register_kv_selector
from ._common import group_query_to_kv, mask_from_keep_indices


@register_kv_selector("h2o")
@dataclass
class H2OSelector(KVSelector):
    """Heavy-hitter + recent-window token selection."""

    flag_needs_attention: bool = True

    keep_ratio: float = 0.5
    keep_tokens: int = None
    recent_tokens: int = 32
    sink_tokens: int = 0

    def score(self, query, key, value, attn_stats, layer_idx):
        if attn_stats is not None and "cum_attn" in attn_stats:
            return attn_stats["cum_attn"]
        # Accumulated column-attention mass over all query positions.
        num_kv_heads = key.shape[1]
        q = group_query_to_kv(query, num_kv_heads).float()
        k = key.float()
        head_dim = key.shape[-1]
        logits = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        # Causal mask so a key is only attended by later queries.
        q_len, k_len = logits.shape[-2], logits.shape[-1]
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=logits.device).tril(
            diagonal=k_len - q_len
        )
        logits = logits.masked_fill(~causal, float("-inf"))
        probs = F.softmax(logits.float(), dim=-1)
        return probs.sum(dim=2)  # (batch, num_kv_heads, seq_len)

    def select(self, scores, budget, seq_len):
        batch, heads, _ = scores.shape
        keep = budget.resolve_keep(seq_len)
        recent = max(budget.recent_tokens, self.recent_tokens)
        recent = min(recent, seq_len)
        sink = min(max(budget.sink_tokens, self.sink_tokens), seq_len)
        protected = torch.zeros(seq_len, dtype=torch.bool, device=scores.device)
        if sink:
            protected[:sink] = True
        if recent:
            protected[-recent:] = True
        protected_count = int(protected.sum())
        keep = min(seq_len, max(keep, protected_count))
        n_topk = keep - protected_count

        scores = scores.clone()
        scores[:, :, protected] = float("-inf")
        topk = min(n_topk, seq_len - protected_count)
        if topk > 0:
            idx = scores.topk(topk, dim=-1).indices
            mask = mask_from_keep_indices(idx, seq_len)
        else:
            mask = torch.zeros(batch, heads, seq_len, dtype=torch.bool, device=scores.device)
        mask[:, :, protected] = True
        return mask
