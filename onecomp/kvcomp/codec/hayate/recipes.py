"""Named HayateQuant operating points.

The codec implementation deliberately stays parameterized: ``bits``,
``value_bits`` and scale modes are primitive knobs.  Paper/benchmark recipes
should live here instead of being re-spelled in every runner.  That keeps the
definition of HQ-frontier auditable and makes future single-codec candidates
compare against the same baseline.
"""

from __future__ import annotations

from typing import Any

from .codec import HayateQuantCodec

HQ_FRONTIER_NAME = "HayateQuant-frontier"
HQ_FRONTIER_KEY_SCALE_MODE = "ub"
HQ_FRACTIONAL_2BIT_NAME = "HayateQuant-frac2"
HQ_ADAPTIVE_2BIT_NAME = "HayateQuant-adaptive2"


def hayate_frontier_bits(avg_bits: int) -> tuple[int, int] | None:
    """Return the HQ-frontier K/V allocation for an average bit budget.

    HQ-frontier uses ``K(r+1)V(r-1)`` at average rate ``r``.  The helper returns
    ``None`` outside the meaningful low-bit range so benchmark registries can
    omit the row cleanly.
    """

    bits = int(avg_bits)
    if 2 <= bits <= 7:
        return bits + 1, bits - 1
    return None


def hayate_frontier_kwargs(avg_bits: int, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for :class:`HayateQuantCodec` at HQ-frontier."""

    kv_bits = hayate_frontier_bits(avg_bits)
    if kv_bits is None:
        raise ValueError(f"HQ-frontier is only defined for average bits 2..7, got {avg_bits!r}")
    key_bits, value_bits = kv_bits
    kwargs: dict[str, Any] = {
        "bits": key_bits,
        "value_bits": value_bits,
        "key_scale_mode": HQ_FRONTIER_KEY_SCALE_MODE,
    }
    kwargs.update(overrides)
    return kwargs


def build_hayate_frontier(avg_bits: int, **overrides: Any) -> HayateQuantCodec:
    """Construct the canonical HQ-frontier codec."""

    return HayateQuantCodec(**hayate_frontier_kwargs(avg_bits, **overrides))


def hayate_frontier_variant(avg_bits: int, label: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for long-context benchmark runners."""

    variant = {"label": label or f"HQ-frontier-{int(avg_bits)}bit"}
    variant.update(hayate_frontier_kwargs(avg_bits, **overrides))
    return variant


def hayate_fractional2_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for the experimental 2-bit fractional block-rate HQ.

    This keeps the same average payload rate as 2-bit HQ-frontier, but uses
    block-size-four fixed-length VQ codes with K10/V6 bits per block, i.e.
    2.5 key bits/dim and 1.5 value bits/dim before the shared fp16 scale.
    Short probes on Llama-3.1-8B, Qwen3-4B, and R1-Qwen-1.5B showed lower
    WikiText-2 PPL than HQ-frontier at the same stored bits.  It remains an
    opt-in research operating point rather than the canonical HQ definition.
    """

    kwargs = hayate_fractional_rate_kwargs(avg_bits=2, block_size=4, key_bit_fraction=0.625)
    kwargs.update(overrides)
    return kwargs


def hayate_fractional_rate_kwargs(
    avg_bits: float,
    *,
    block_size: int = 4,
    key_bit_fraction: float = 0.625,
    adaptive: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Human-facing fixed-budget fractional block-rate HQ kwargs.

    ``avg_bits`` is the average K/V payload rate per coordinate.  The helper
    converts it to the integer number of fixed-length bits stored for one
    key/value block pair, avoiding manual K10/V6-style bookkeeping.  With the
    defaults, ``avg_bits=2`` and ``block_size=4`` give a 16-bit K/V pair
    budget; ``key_bit_fraction=0.625`` resolves to K10/V6.  Set
    ``adaptive=True`` to let the codec choose the K/V split from cheap online
    statistics under the same total budget, then quantize once.
    """

    nominal = max(1, min(8, int(round(float(avg_bits)))))
    kwargs: dict[str, Any] = {
        "bits": nominal,
        "value_bits": nominal,
        "avg_bits": float(avg_bits),
        "block_size": int(block_size),
        "key_bit_fraction": float(key_bit_fraction),
        "key_scale_mode": HQ_FRONTIER_KEY_SCALE_MODE,
    }
    if adaptive:
        kwargs["adaptive_bit_split"] = "online_stats"
    kwargs.update(overrides)
    return kwargs


def build_hayate_fractional_rate(avg_bits: float, **overrides: Any) -> HayateQuantCodec:
    """Construct a fractional block-rate HQ codec from average-bit UI knobs."""

    return HayateQuantCodec(**hayate_fractional_rate_kwargs(avg_bits, **overrides))


def hayate_fractional_rate_variant(
    avg_bits: float, label: str | None = None, **overrides: Any,
) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for fractional block-rate HQ."""

    variant = {"label": label or f"HayateQuant-frac-{float(avg_bits):g}bit"}
    variant.update(hayate_fractional_rate_kwargs(avg_bits, **overrides))
    return variant


def build_hayate_fractional2(**overrides: Any) -> HayateQuantCodec:
    """Construct the experimental 2-bit fractional block-rate HQ codec."""

    return HayateQuantCodec(**hayate_fractional2_kwargs(**overrides))


def hayate_fractional2_variant(label: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for the fractional 2-bit HQ candidate."""

    variant = {"label": label or f"{HQ_FRACTIONAL_2BIT_NAME}-2bit"}
    variant.update(hayate_fractional2_kwargs(**overrides))
    return variant


def hayate_adaptive2_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for opt-in stats-adaptive 2-bit K/V bit splitting."""

    kwargs = hayate_fractional_rate_kwargs(
        avg_bits=2, block_size=4, key_bit_fraction=0.625, adaptive=True)
    kwargs.update(overrides)
    return kwargs


def build_hayate_adaptive2(**overrides: Any) -> HayateQuantCodec:
    """Construct the opt-in stats-adaptive 2-bit HQ codec."""

    return HayateQuantCodec(**hayate_adaptive2_kwargs(**overrides))


def hayate_adaptive2_variant(label: str | None = None, **overrides: Any) -> dict[str, Any]:
    """Hydra/YAML variant dictionary for opt-in stats-adaptive 2-bit HQ."""

    variant = {"label": label or f"{HQ_ADAPTIVE_2BIT_NAME}-2bit"}
    variant.update(hayate_adaptive2_kwargs(**overrides))
    return variant


__all__ = [
    "HQ_FRONTIER_NAME",
    "HQ_FRONTIER_KEY_SCALE_MODE",
    "HQ_FRACTIONAL_2BIT_NAME",
    "HQ_ADAPTIVE_2BIT_NAME",
    "hayate_frontier_bits",
    "hayate_frontier_kwargs",
    "build_hayate_frontier",
    "hayate_frontier_variant",
    "hayate_fractional_rate_kwargs",
    "build_hayate_fractional_rate",
    "hayate_fractional_rate_variant",
    "hayate_fractional2_kwargs",
    "build_hayate_fractional2",
    "hayate_fractional2_variant",
    "hayate_adaptive2_kwargs",
    "build_hayate_adaptive2",
    "hayate_adaptive2_variant",
]
