"""PM-KVQ KV cache: progressive, dynamic-precision quantization (baseline).

See :class:`onecomp.kvcomp.recipe.pmkvq.PMKVQPolicy` for the method summary.
Implementation rides on :class:`~onecomp.kvcomp.cache._graded_base.GradedQuantCache`;
the only PM-KVQ-specific piece is the bit schedule :func:`progressive_bits`
(a *fixed absolute* high-precision window + ``high -> final`` ramp) plus the
block-wise per-layer bit bump.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import torch

from ._graded_base import GradedQuantCache


def progressive_bits(recency: torch.Tensor, recent_window: int, ladder_span: int,
                     high_bits: int, final_bits: int) -> torch.Tensor:
    """Per-token bits as a function of *recency* (0 = newest).

    Newest ``recent_window`` tokens stay at ``high_bits``; the next
    ``ladder_span`` tokens ramp ``high_bits -> final_bits``; everything older
    is ``final_bits``.  The window/ramp are **absolute** token counts, so as
    the sequence grows the ``final_bits`` tail grows -- the progressive,
    memory-saturation-driven shrink of PM-KVQ.
    """
    bits = torch.full_like(recency, final_bits, dtype=torch.int64)
    bits[recency < recent_window] = high_bits
    mid = (recency >= recent_window) & (recency < recent_window + ladder_span)
    if mid.any() and ladder_span > 0 and high_bits > final_bits:
        frac = (recency[mid].float() - recent_window) / float(ladder_span)
        b = torch.round(high_bits - frac * (high_bits - final_bits))
        bits[mid] = b.clamp(final_bits, high_bits).to(torch.int64)
    return bits


class PMKVQCache(GradedQuantCache):
    """Graded cache with a *dynamic* progressive precision schedule (PM-KVQ).

    Args:
        policy: a :class:`onecomp.kvcomp.recipe.pmkvq.PMKVQPolicy`.
        config: model config (preserves sliding-window layer typing).
        requant_stride: re-apply the schedule every this many decode steps;
            larger = cheaper.  Re-grading is what makes aging tokens shrink
            progressively (and accumulate the paper's re-quantization error).
    """

    def __init__(self, policy, config=None, requant_stride: int = 16, **kwargs):
        super().__init__(config=config, skip_layers=policy.skip_layers,
                         requant_stride=requant_stride, **kwargs)
        self.policy = policy
        self._sensitive = set(policy.sensitive_layers or [])

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        bump = p.sensitive_bump if layer_idx in self._sensitive else 0
        recency = torch.arange(total - 1, -1, -1, device=device)  # 0 = newest
        kbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                 p.key_bits + bump, p.key_final_bits + bump)
        vbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                 p.value_bits + bump, p.final_bits + bump)
        return kbits, vbits
