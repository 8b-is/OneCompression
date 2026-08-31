"""TaperKV policy: retrieval-preserving graded KV quantization (no eviction).

The companion of :class:`onecomp.kvcomp.cache.taper_cache.TaperKVCache`.  Where
ThinKV evicts tokens to a budget, TaperKV keeps every token and *tapers* its
precision with age, down to a non-zero floor.  This preserves long-context
retrieval (the needle is never dropped) while still hitting an aggressive
average-bit target.

Key design choices:

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
        key_ladder: Explicit descending key ladder for the middle region.
        value_ladder: Explicit descending value ladder for the middle region.
        skip_layers: Layers left in full precision (outlier layers).
    """

    key_bits: int = 8
    value_bits: int = 4
    b_min: int = 4
    b_min_value: int = 2
    sink_tokens: int = 4
    recent_tokens: int = 128
    key_ladder: Optional[list] = None
    value_ladder: Optional[list] = None
    skip_layers: list = field(default_factory=list)
    requant_stride: Optional[int] = None

    def __post_init__(self):
        """Reject schedules that cannot be represented by the cache codec.

        TaperKV is intentionally a parameter carrier, so these checks need to
        live here rather than relying on a particular cache constructor.  In
        particular, accepting a zero-bit floor would quietly turn a
        length-preserving method into an eviction method.
        """
        super().__post_init__()

        def positive_int(value, name, *, maximum=16):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 1 or value > maximum:
                raise ValueError(f"{name} must be in [1, {maximum}], got {value}")

        positive_int(self.key_bits, "key_bits")
        positive_int(self.value_bits, "value_bits")
        positive_int(self.b_min, "b_min")
        positive_int(self.b_min_value, "b_min_value")
        if self.value_bits > self.key_bits:
            raise ValueError("value_bits must not exceed key_bits")
        if self.b_min > self.key_bits:
            raise ValueError("b_min must not exceed key_bits")
        if self.b_min_value > self.value_bits:
            raise ValueError("b_min_value must not exceed value_bits")
        for value, name in (
            (self.sink_tokens, "sink_tokens"),
            (self.recent_tokens, "recent_tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        def validate_ladder(ladder, name, high, floor):
            if ladder is None:
                return tuple(range(high, floor - 1, -1))
            if not isinstance(ladder, (list, tuple)) or not ladder:
                raise ValueError(f"{name} must be a non-empty list or tuple when supplied")
            if any(isinstance(bit, bool) or not isinstance(bit, int) for bit in ladder):
                raise ValueError(f"{name} must contain integer bit-widths")
            if ladder[0] != high or ladder[-1] != floor:
                raise ValueError(f"{name} must start at {high} and end at {floor}")
            if any(bit < floor or bit > high for bit in ladder):
                raise ValueError(f"{name} entries must lie in [{floor}, {high}]")
            if any(left <= right for left, right in zip(ladder, ladder[1:])):
                raise ValueError(f"{name} must be strictly descending")
            return tuple(ladder)

        key_ladder = validate_ladder(self.key_ladder, "key_ladder", self.key_bits, self.b_min)
        value_ladder = validate_ladder(
            self.value_ladder, "value_ladder", self.value_bits, self.b_min_value
        )
        # The two roles may use different numbers of bands.  Check every joint
        # piecewise-constant interval so K >= V holds throughout the schedule.
        boundaries = sorted(
            {0.0, 1.0}
            | {index / len(key_ladder) for index in range(1, len(key_ladder))}
            | {index / len(value_ladder) for index in range(1, len(value_ladder))}
        )
        probes = boundaries[:-1] + [
            (left + right) / 2.0 for left, right in zip(boundaries, boundaries[1:])
        ]
        for fraction in probes:
            key_bit = key_ladder[min(int(fraction * len(key_ladder)), len(key_ladder) - 1)]
            value_bit = value_ladder[min(int(fraction * len(value_ladder)), len(value_ladder) - 1)]
            if key_bit < value_bit:
                raise ValueError("key_ladder must preserve key precision >= value precision")
        if not isinstance(self.skip_layers, (list, tuple)):
            raise ValueError("skip_layers must be a list or tuple of integer layer indices")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in self.skip_layers
        ):
            raise ValueError("skip_layers must contain integer layer indices")
        if self.requant_stride is not None and (
            isinstance(self.requant_stride, bool)
            or not isinstance(self.requant_stride, int)
            or self.requant_stride < 1
        ):
            raise ValueError("requant_stride must be a positive integer or None")

    def target_avg_bits(self, seq_len: int) -> float:
        """Rough expected average bits (keys+values)/2 for a given length."""
        import torch  # local import; policy is light

        from ..cache.taper_cache import taper_bits

        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
            raise ValueError("seq_len must be a non-negative integer")
        if seq_len == 0:
            return 0.0

        age = torch.arange(seq_len - 1, -1, -1)
        kb = (
            taper_bits(
                age,
                seq_len,
                self.sink_tokens,
                self.recent_tokens,
                self.key_bits,
                self.b_min,
                self.key_ladder,
            )
            .float()
            .mean()
        )
        vb = (
            taper_bits(
                age,
                seq_len,
                self.sink_tokens,
                self.recent_tokens,
                self.value_bits,
                self.b_min_value,
                self.value_ladder,
            )
            .float()
            .mean()
        )
        return float((kb + vb) / 2)

    def get_kv_config(self) -> dict:
        return {
            "name": "taperkv",
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "b_min": self.b_min,
            "b_min_value": self.b_min_value,
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
            "key_ladder": self.key_ladder,
            "value_ladder": self.value_ladder,
            "skip_layers": list(self.skip_layers),
            "requant_stride": self.requant_stride,
        }


@register_kv_policy("block_taperkv")
@dataclass
class BlockTaperKVPolicy(TaperKVPolicy):
    """TaperKV constrained to a uniform packed layout per cache block.

    ``quant_granularity='block'`` stores one affine scale/zero pair per
    batch-head-block.  A trailing write block can remain fp16 until full,
    matching the usual paged-cache residual-buffer contract.
    """

    block_size: int = 16
    residual_fp16: bool = True
    quant_granularity: str = "block"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ValueError("block_size must be an integer")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not isinstance(self.residual_fp16, bool):
            raise ValueError("residual_fp16 must be a bool")
        if self.quant_granularity not in ("block", "token"):
            raise ValueError(
                "quant_granularity must be 'block' or 'token', " f"got {self.quant_granularity!r}"
            )
        if self.quant_granularity == "block" and not self.residual_fp16:
            raise ValueError(
                "block-granularity TaperKV requires residual_fp16=True so a "
                "partially written block is never repeatedly requantized"
            )

    def target_avg_bits(self, seq_len: int) -> float:
        """Expected payload bits under the block-uniform schedule.

        This intentionally includes the fp16 residual block when enabled.  It
        is a payload-only rate; affine metadata is reported by the live cache
        separately and is not hidden in this convenience estimate.
        """
        from ..cache.block_taper_cache import block_taper_bits

        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
            raise ValueError("seq_len must be a non-negative integer")
        if seq_len == 0:
            return 0.0
        kb = (
            block_taper_bits(
                seq_len,
                self.block_size,
                self.sink_tokens,
                self.recent_tokens,
                self.key_bits,
                self.b_min,
                self.key_ladder,
                self.residual_fp16,
            )
            .float()
            .mean()
        )
        vb = (
            block_taper_bits(
                seq_len,
                self.block_size,
                self.sink_tokens,
                self.recent_tokens,
                self.value_bits,
                self.b_min_value,
                self.value_ladder,
                self.residual_fp16,
            )
            .float()
            .mean()
        )
        return float((kb + vb) / 2)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "name": "block_taperkv",
                "block_size": self.block_size,
                "residual_fp16": self.residual_fp16,
                "quant_granularity": self.quant_granularity,
            }
        )
        return cfg


@register_kv_policy("budget_taperkv")
@dataclass
class BudgetTaperKVPolicy(TaperKVPolicy):
    """Payload-budgeted no-eviction TaperKV policy.

    ``max_bytes_per_layer`` is a per-layer cap on K/V *payload* only.  Affine scale/zero metadata
    remains explicitly counted in ``GradedQuantCache`` reporting, but cannot
    be folded into this policy without the runtime batch/head/dimension shape.
    """

    max_bytes_per_layer: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_bytes_per_layer is not None and (
            isinstance(self.max_bytes_per_layer, bool)
            or not isinstance(self.max_bytes_per_layer, int)
            or self.max_bytes_per_layer < 1
        ):
            raise ValueError("max_bytes_per_layer must be a positive integer or None")

    def target_avg_bits(self, seq_len: int) -> float:
        """Return a rate only when no budget cap is active.

        A byte cap cannot be converted to bits/token from sequence length
        alone: batch size, KV heads, and head dimension change the result.
        Refuse the ambiguous estimate rather than reporting TaperKV's static
        age schedule for this different algorithm.
        """
        if self.max_bytes_per_layer is not None:
            raise ValueError(
                "BudgetTaperKV target_avg_bits is shape-dependent; use live "
                "cache accounting (avg_bits/mem_fraction) instead"
            )
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
            raise ValueError("seq_len must be a non-negative integer")
        return 0.0 if seq_len == 0 else float((self.key_bits + self.value_bits) / 2)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "name": "budget_taperkv",
                "max_bytes_per_layer": self.max_bytes_per_layer,
                "budget_scope": "per_layer_key_value_payload_only_excludes_affine_metadata",
            }
        )
        return cfg
