"""Shared helpers for KV token selectors.

Tensor layout: ``key/value`` are ``(batch, num_kv_heads, seq_len, head_dim)`` and
``query`` is ``(batch, num_heads, q_len, head_dim)``.  For GQA, query heads are
mean-pooled into key-value groups before scoring.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_query_to_kv(query: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Mean-pool query heads into ``num_kv_heads`` groups (GQA aware).

    Args:
        query: ``(batch, num_heads, q_len, head_dim)``.
        num_kv_heads: Number of key/value heads.

    Returns:
        ``(batch, num_kv_heads, q_len, head_dim)``.
    """
    batch, num_heads, q_len, head_dim = query.shape
    if num_heads == num_kv_heads:
        return query
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads {num_heads} not divisible by num_kv_heads {num_kv_heads}")
    groups = num_heads // num_kv_heads
    query = query.view(batch, num_kv_heads, groups, q_len, head_dim)
    return query.mean(dim=2)


def observation_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Attention probabilities of the last ``window_size`` queries over all keys.

    Returns ``(batch, num_kv_heads, window, seq_len)`` softmax probabilities,
    the standard SnapKV "observation window" used to score prompt tokens.
    """
    num_kv_heads = key.shape[1]
    q = group_query_to_kv(query, num_kv_heads).float()
    q = q[:, :, -window_size:, :]
    k = key.float()
    head_dim = key.shape[-1]
    logits = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
    return F.softmax(logits, dim=-1)


def mask_from_keep_indices(keep_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Build a boolean keep-mask ``(batch, heads, seq_len)`` from kept indices.

    ``keep_idx`` is ``(batch, heads, keep)``.
    """
    batch, heads, _ = keep_idx.shape
    mask = torch.zeros(batch, heads, seq_len, dtype=torch.bool, device=keep_idx.device)
    mask.scatter_(-1, keep_idx, True)
    return mask
