"""TaperKV: retrieval-preserving graded KV quantization (no eviction).

Motivation (onecomp-ref/260528.md): ThinKV compresses by *progressive
eviction* -- it permanently drops tokens.  That is strong for reasoning but
fragile for **long-context retrieval** (Needle / RULER): if the evicted token
*is* the needle, the answer is unrecoverable.

TaperKV keeps **every** token but assigns each a bit-width that *tapers* with
age: a full-precision sink + recent window, then a precision ladder that decays
toward a floor ``b_min > 0`` for old tokens.  No token is ever dropped, so the
needle survives -- lossily, but recoverably.  This trades ThinKV's hard
information loss (eviction) for graceful precision loss (PM-KVQ-style progressive
quantization), and additionally exploits the KVTuner observation that **keys
deserve more bits than values** (``key_bits >= value_bits``).

Length-preserving and RoPE-safe: like :class:`CompressedDynamicCache` we keep the
dequantized tensor (simulated quantization) and never change positions, so the
cache is simply passed to ``model.generate(past_key_values=...)``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger

import torch

from ._graded_base import GradedQuantCache

logger = getLogger(__name__)


def taper_bits(age: torch.Tensor, total: int, sink_tokens: int, recent_tokens: int,
               hi_bits: int, b_min: int, ladder=None) -> torch.Tensor:
    """Per-token bit-width as a function of token *age* (0 = newest).

    Sink (oldest ``sink_tokens``) and the ``recent_tokens`` newest stay at
    ``hi_bits``.  The middle region steps down ``hi_bits -> b_min`` across
    ``ladder`` equal bands (default: every integer bit-width in between).
    """
    bits = torch.full_like(age, hi_bits, dtype=torch.int64)
    is_sink = age >= (total - sink_tokens)
    is_recent = age < recent_tokens
    mid = ~(is_sink | is_recent)
    if mid.any():
        if ladder is None:
            ladder = list(range(hi_bits, b_min - 1, -1)) or [b_min]
        nb = len(ladder)
        a = age[mid].float()
        lo, hi = recent_tokens, max(recent_tokens + 1, total - sink_tokens)
        frac = ((a - lo) / max(1.0, (hi - lo))).clamp(0, 1 - 1e-6)
        band = (frac * nb).long().clamp(0, nb - 1)
        ladder_t = torch.tensor(ladder, device=age.device, dtype=torch.int64)
        bits[mid] = ladder_t[band]
    # An explicitly-supplied ``ladder`` may hold values outside this role's
    # [b_min, hi_bits] range (e.g. a key ladder reused for values).  Clamp so no
    # token exceeds this role's high bit-width (preserving KVTuner K>=V) or drops
    # below its floor.  With the default ladder this is a no-op.
    return bits.clamp(min=b_min, max=hi_bits)


class TaperKVCache(GradedQuantCache):
    """Graded cache with a *static* age-indexed precision ladder; never evicts.

    The bit schedule (:meth:`token_bits`) is a fixed function of token *age*:
    sink + recent tokens stay at high precision, older tokens step down a
    ladder toward a non-zero floor.  All cache mechanics, re-quantization
    cadence and bit accounting are inherited from
    :class:`~onecomp.kvcomp.cache._graded_base.GradedQuantCache`.

    Args:
        policy: a :class:`onecomp.kvcomp.recipe.taperkv.TaperKVPolicy`.
        config: model config (preserves sliding-window layer typing).
        requant_stride: re-apply the ladder to the full cache every this many
            decode steps (1 = every step; larger = cheaper, staler).
    """

    def __init__(self, policy, config=None, requant_stride: int = 16, **kwargs):
        super().__init__(config=config, skip_layers=policy.skip_layers,
                         requant_stride=requant_stride, **kwargs)
        self.policy = policy

    def token_bits(self, total, device, layer_idx):
        p = self.policy
        age = torch.arange(total - 1, -1, -1, device=device)  # 0 = newest
        kbits = taper_bits(age, total, p.sink_tokens, p.recent_tokens,
                           p.key_bits, p.b_min, p.ladder)
        vbits = taper_bits(age, total, p.sink_tokens, p.recent_tokens,
                           p.value_bits, p.b_min_value, p.ladder)
        return kbits, vbits
