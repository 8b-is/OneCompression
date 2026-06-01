"""PM-KVQ policy: Progressive Mixed-precision KV Cache Quantization (baseline).

Faithful re-implementation of the *mechanism* of PM-KVQ (Su et al.,
arXiv:2505.18610) as an online, length-preserving HF cache, for an apples-to-
apples comparison against ThinKV and the proposed TaperKV.

PM-KVQ's two ideas (paper §3):

1. **Progressive quantization.**  Instead of quantizing every token to the
   final bit-width ``Fbit`` immediately (which accumulates error over a long
   chain-of-thought), store recent KV at a higher precision and *gradually
   shrink* the bit-width of older KV as the memory budget saturates.  Here this
   is realised by a **fixed absolute** high-precision recent window plus a
   transition band; tokens older than that collapse to ``final_bits``.  Because
   a token's bit-width is recomputed from its current recency at each
   re-grade, a token's precision *decreases as it ages* -- exactly the
   progressive "shifting" shrink of the paper (with the same cumulative re-
   quantization error).

2. **Block-wise memory allocation.**  More sensitive transformer blocks
   (layers) get a higher bit-width.  The paper solves an integer program over
   per-block sensitivity; for a fair same-budget comparison we expose the
   resulting assignment directly via ``sensitive_layers`` (+``sensitive_bump``
   bits), which is sufficient to reproduce the *behaviour* (uneven per-layer
   bits) without the offline solver.

Contrast with TaperKV (the proposed method): TaperKV's ladder is indexed by
token *age fraction* relative to the **current length**, so the graded
structure *stretches* with context and mid-context tokens keep intermediate
precision.  PM-KVQ's high-precision region is a **fixed token count**, so at
long context almost everything older than the window collapses to ``final_bits``
(2-bit).  This is the crux of the long-context-retrieval comparison.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .._base import KVPolicy
from .._registry import register_kv_policy


@register_kv_policy("pmkvq")
@dataclass
class PMKVQPolicy(KVPolicy):
    """Progressive mixed-precision schedule (dynamic, by decode progress).

    Attributes:
        key_bits: High bit-width for recent keys (top of the progressive ramp).
        value_bits: High bit-width for recent values (``<= key_bits``, MKLV).
        final_bits: ``Fbit`` -- floor the oldest KV shrinks to (never 0, so no
            eviction; matches PM-KVQ's low-bit final precision, e.g. 2).
        recent_window: newest tokens kept at high precision (fixed *absolute*
            count = the high-precision memory budget per layer).
        ladder_span: number of tokens over which bits ramp ``high -> final``
            just older than the recent window (the progressive transition).
        sensitive_layers: transformer blocks given ``sensitive_bump`` extra bits
            (block-wise memory allocation; the IP solution, supplied directly).
        sensitive_bump: extra bits for sensitive blocks.
        skip_layers: layers kept in full precision (outlier layers).
    """

    key_bits: int = 8
    value_bits: int = 4
    final_bits: int = 2            # value floor (Fbit)
    key_final_bits: int = None     # key floor; defaults to max(final_bits, 4) (MKLV: K>V)
    recent_window: int = 64
    ladder_span: int = 256
    sensitive_layers: List[int] = field(default_factory=list)
    sensitive_bump: int = 1
    skip_layers: List[int] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.key_final_bits is None:
            # Keys are outlier-sensitive; per-token 2-bit keys are catastrophic,
            # so the key floor stays >= value floor (KVTuner / MKLV).
            self.key_final_bits = max(self.final_bits, 4)

    def target_avg_bits(self, seq_len: int) -> float:
        """Rough expected average bits (keys+values)/2 at a given length."""
        import torch  # local import; policy is light
        from ..cache.pmkvq_cache import progressive_bits

        r = torch.arange(seq_len - 1, -1, -1)
        kb = progressive_bits(r, self.recent_window, self.ladder_span,
                              self.key_bits, self.key_final_bits).float().mean()
        vb = progressive_bits(r, self.recent_window, self.ladder_span,
                              self.value_bits, self.final_bits).float().mean()
        return float((kb + vb) / 2)

    def get_kv_config(self) -> dict:
        return {
            "name": "pmkvq", "key_bits": self.key_bits, "value_bits": self.value_bits,
            "final_bits": self.final_bits, "recent_window": self.recent_window,
            "ladder_span": self.ladder_span,
            "sensitive_layers": list(self.sensitive_layers),
            "sensitive_bump": self.sensitive_bump, "skip_layers": list(self.skip_layers),
        }
