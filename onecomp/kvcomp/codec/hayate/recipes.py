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


__all__ = [
    "HQ_FRONTIER_NAME",
    "HQ_FRONTIER_KEY_SCALE_MODE",
    "hayate_frontier_bits",
    "hayate_frontier_kwargs",
    "build_hayate_frontier",
    "hayate_frontier_variant",
]
