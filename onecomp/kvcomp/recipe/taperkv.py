"""TaperKV policy: retrieval-preserving graded KV quantization (no eviction).

The companion of :class:`onecomp.kvcomp.cache.taper_cache.TaperKVCache`.  Where
ThinKV evicts tokens to a budget, TaperKV keeps every token and *tapers* its
precision with age, down to a non-zero floor.  This preserves long-context
retrieval (the needle is never dropped) while still hitting an aggressive
average-bit target.

Key design choices (from onecomp-ref/260528.md):

* **No eviction.**  All tokens retained; old tokens are quantized harder, not
  removed (PM-KVQ-style progressive quantization applied by *age*, not by global
  decode step).
* **K > V asymmetry** (KVTuner Obs.): keys carry attention scores, so they get
  more bits than values (``key_bits >= value_bits``).
* **Sink + recent full precision**: attention-sink and the recent window stay at
  ``key_bits`` / ``value_bits`` (high), matching StreamingLLM / KIVI robustness.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .._base import KVPolicy
from .._registry import register_kv_policy


@register_kv_policy("taperkv")
@dataclass
class TaperKVPolicy(KVPolicy):
    """Recency-graded precision schedule.

    Attributes:
        key_bits: High bit-width for sink/recent keys and the top of the ladder.
        value_bits: High bit-width for sink/recent values (``<= key_bits``).
        b_min: Floor bit-width for the oldest keys (never 0 -> no eviction).
        b_min_value: Floor bit-width for the oldest values.
        sink_tokens: Leading tokens kept at high precision.
        recent_tokens: Most-recent tokens kept at high precision.
        ladder: Explicit descending bit ladder for the middle region
            (default: every integer from ``key_bits`` down to ``b_min``).
        skip_layers: Layers left in full precision (outlier layers).
    """

    key_bits: int = 8
    value_bits: int = 4
    b_min: int = 2
    b_min_value: int = 2
    sink_tokens: int = 4
    recent_tokens: int = 64
    ladder: Optional[list] = None
    skip_layers: list = field(default_factory=list)

    def target_avg_bits(self, seq_len: int) -> float:
        """Rough expected average bits (keys+values)/2 for a given length."""
        import torch  # local import; policy is light
        from ..cache.taper_cache import taper_bits

        age = torch.arange(seq_len - 1, -1, -1)
        kb = taper_bits(age, seq_len, self.sink_tokens, self.recent_tokens,
                        self.key_bits, self.b_min, self.ladder).float().mean()
        vb = taper_bits(age, seq_len, self.sink_tokens, self.recent_tokens,
                        self.value_bits, self.b_min_value, self.ladder).float().mean()
        return float((kb + vb) / 2)

    def get_kv_config(self) -> dict:
        return {
            "name": "taperkv", "key_bits": self.key_bits, "value_bits": self.value_bits,
            "b_min": self.b_min, "b_min_value": self.b_min_value,
            "sink_tokens": self.sink_tokens, "recent_tokens": self.recent_tokens,
            "ladder": self.ladder, "skip_layers": list(self.skip_layers),
        }
