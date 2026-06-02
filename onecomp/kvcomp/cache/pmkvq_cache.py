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

from ._graded_base import GradedQuantCache, fake_quant_tokens


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
        self._block_final = dict(getattr(policy, "block_final_bits", {}) or {})
        self._key_scale = dict(getattr(policy, "key_channel_scale", {}) or {})

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        recency = torch.arange(total - 1, -1, -1, device=device)  # 0 = newest
        if layer_idx in self._block_final:
            # Faithful block-wise IP allocation: this layer's *value* floor is the
            # solver's choice; the key floor stays >= value floor (MKLV K>V).
            v_final = int(self._block_final[layer_idx])
            k_final = max(v_final, p.key_final_bits)
            kbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                     p.key_bits, k_final)
            vbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                     p.value_bits, v_final)
            return kbits, vbits
        # Manual shortcut: per-layer additive bump for sensitive blocks.
        bump = p.sensitive_bump if layer_idx in self._sensitive else 0
        kbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                 p.key_bits + bump, p.key_final_bits + bump)
        vbits = progressive_bits(recency, p.recent_window, p.ladder_span,
                                 p.value_bits + bump, p.final_bits + bump)
        return kbits, vbits

    def _grade_layer(self, keys, values, layer_idx):
        """Override: apply per-channel key reparameterization before quantizing.

        With §3.3 calibration the key is reconstructed as ``λ·Q_b(K/λ)`` so the
        outlier channels are migrated into range before per-token quantization
        (the value path is unchanged).  Falls back to the base behaviour when no
        reparam factors are calibrated for this layer.
        """
        lam = self._key_scale.get(layer_idx)
        if lam is None:
            return super()._grade_layer(keys, values, layer_idx)
        kbits, vbits = self.token_bits(keys.shape[2], keys.device, layer_idx)
        lam = lam.to(keys.device, keys.dtype)              # (n_kv_heads, head_dim)
        inv = (1.0 / lam.clamp_min(1e-6)).view(1, lam.shape[0], 1, lam.shape[1])
        scl = lam.view(1, lam.shape[0], 1, lam.shape[1])
        kq = scl.float() * fake_quant_tokens(keys.float() * inv.float(), kbits)
        vq = fake_quant_tokens(values, vbits)
        per = keys.shape[1] * keys.shape[-1]
        bits = int(kbits.sum() * per + vbits.sum() * per)
        return kq, vq, bits
