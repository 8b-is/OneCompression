"""HayateQuant: data-free block-VQ KV-cache quantization (strict superset of
TurboQuant and HALO).

Package layout (one module per concern):

- :mod:`.codec` -- :class:`HayateQuantCodec`: the KV codec itself (rotation
  dispatch, role-asymmetric scales, K/V-asymmetric bits, outlier split,
  encode/decode).
- :mod:`.vq` -- data-free block-VQ machinery: Lloyd (K-means) codebook fits on
  the exact source law, committed high-budget codebook artifacts, exact
  GEMM+argmax encoder and the candidate-list grid-LUT encoder.
- :mod:`.calibration` -- optional SpinQuant-style calibration (learned
  per-role orthogonal rotations + real-data codebook refits); off by default,
  never required.
- :mod:`.codebooks` -- committed ``gauss_b{B}_k{K}.pt`` codebook artifacts and
  the :mod:`.codebooks.fit_codebooks` refit script.

Shared, method-agnostic primitives (Hadamard/dense/hybrid rotations in
:mod:`..rotation`, role-asymmetric scale fits in :mod:`.._scale`) stay at the
``codec`` package level because TurboQuant and HALO use them too.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .codec import HayateQuantCodec
from .calibration import (
    HayateQuantCalibration,
    calibrate_hayatequant,
    capture_kv_samples,
)

__all__ = [
    "HayateQuantCodec",
    "HayateQuantCalibration",
    "calibrate_hayatequant",
    "capture_kv_samples",
]
