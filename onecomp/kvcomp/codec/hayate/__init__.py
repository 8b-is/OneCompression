"""HayateQuant data-free block-VQ KV-cache compression."""

from .calibration import HayateQuantCalibration, calibrate_hayatequant, capture_kv_samples
from .codec import HayateQuantCodec
from .recipes import (
    HAYATEQUANT_BLOCK_SIZE,
    HAYATEQUANT_NAME,
    HAYATEQUANT_RATES,
    build_hayatequant,
    hayatequant_code_bits,
    hayatequant_role_rates,
)

__all__ = [
    "HayateQuantCodec",
    "HayateQuantCalibration",
    "calibrate_hayatequant",
    "capture_kv_samples",
    "HAYATEQUANT_NAME",
    "HAYATEQUANT_RATES",
    "HAYATEQUANT_BLOCK_SIZE",
    "hayatequant_role_rates",
    "hayatequant_code_bits",
    "build_hayatequant",
]
