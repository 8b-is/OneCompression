"""Canonical HayateQuant recipe.

HayateQuant has one public identity and one adopted configuration per average
rate. Component ablations are not accepted by the serving/user constructor.

The adopted data-free operating points use a two-coordinate block and assign
one extra bit per key coordinate at the expense of values. At average rates
2, 3, and 4 this is K3/V1, K4/V2, and K5/V3 bits per coordinate, or K6/V2,
K8/V4, and K10/V6 code bits per two-coordinate block.
"""

from __future__ import annotations

from .codec import HayateQuantCodec

HAYATEQUANT_NAME = "hayatequant"
HAYATEQUANT_RATES = (2, 3, 4)
HAYATEQUANT_BLOCK_SIZE = 2


def _rate(avg_bits: int | float) -> int:
    if isinstance(avg_bits, bool) or not isinstance(avg_bits, (int, float)):
        raise ValueError(f"hayatequant rate must be one of {HAYATEQUANT_RATES}, got {avg_bits!r}")
    value = float(avg_bits)
    if not value.is_integer() or int(value) not in HAYATEQUANT_RATES:
        raise ValueError(f"hayatequant rate must be one of {HAYATEQUANT_RATES}, got {avg_bits!r}")
    return int(value)


def hayatequant_role_rates(avg_bits: int | float) -> tuple[float, float]:
    """Return key/value code rates in bits per coordinate."""

    rate = _rate(avg_bits)
    return float(rate + 1), float(rate - 1)


def hayatequant_code_bits(avg_bits: int | float) -> tuple[int, int]:
    """Return fixed-width key/value code bits per two-coordinate block."""

    rate = _rate(avg_bits)
    return 2 * rate + 2, 2 * rate - 2


def build_hayatequant(
    avg_bits: int | float,
    *,
    calibration=None,
    skip_layers=None,
) -> HayateQuantCodec:
    """Construct the adopted HayateQuant codec with no variant overrides."""

    return HayateQuantCodec(
        bits=_rate(avg_bits),
        calibration=calibration,
        skip_layers=skip_layers,
    )


__all__ = [
    "HAYATEQUANT_NAME",
    "HAYATEQUANT_RATES",
    "HAYATEQUANT_BLOCK_SIZE",
    "hayatequant_role_rates",
    "hayatequant_code_bits",
    "build_hayatequant",
]
