"""Backward-compatibility shim.

Calibration moved to :mod:`onecomp.kvcomp.codec.hayate.calibration`.
Import from there in new code.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .hayate.calibration import (  # noqa: F401
    HayateQuantCalibration,
    calibrate_hayatequant,
    capture_kv_samples,
)
