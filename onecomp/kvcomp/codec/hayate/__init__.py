"""HayateQuant: data-free block-VQ KV-cache quantization.

Explicit scalar compatibility settings reproduce selected TurboQuant/HALO
quantization components; the default block-VQ recipe is a distinct algorithm
and is evaluated independently rather than described as a strict superset.

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
- :mod:`.recipes` -- the fixed HQ and opt-in Adaptive HQ paper recipes.
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
from .recipes import (
    HQ_NAME,
    HQ_ADAPTIVE_NAME,
    HQ_ROTATION_NAME,
    HQ_PUBLIC_METHODS,
    HQ_PAPER_AVERAGE_RATES,
    HQ_BLOCK_SIZE,
    HQ_KEY_SCALE_MODE,
    HQ_VALUE_SCALE_MODE,
    HQ_FRONTIER_NAME,
    HQ_FRONTIER_KEY_SCALE_MODE,
    HQ_FRONTIER_VALUE_SCALE_MODE,
    HQ_FRONTIER_SCALE_KWARGS,
    HQ_FIB_INIT_PROXY_NAME,
    HQ_BLOCK_SCALE_PROXY_NAME,
    HQ_PROXY_LEGACY_ALIASES,
    canonical_hayate_method_name,
    hayate_result_method_candidates,
    hayate_hq_role_rates,
    hayate_hq_code_bits,
    hayate_hq_kwargs,
    hayate_frontier_bits,
    hayate_frontier_kwargs,
    build_hayate_frontier,
    build_hayate_hq,
    hayate_frontier_variant,
)

__all__ = [
    "HayateQuantCodec",
    "HayateQuantCalibration",
    "calibrate_hayatequant",
    "capture_kv_samples",
    "HQ_NAME",
    "HQ_ADAPTIVE_NAME",
    "HQ_ROTATION_NAME",
    "HQ_PUBLIC_METHODS",
    "HQ_PAPER_AVERAGE_RATES",
    "HQ_BLOCK_SIZE",
    "HQ_KEY_SCALE_MODE",
    "HQ_VALUE_SCALE_MODE",
    "HQ_FRONTIER_NAME",
    "HQ_FRONTIER_KEY_SCALE_MODE",
    "HQ_FRONTIER_VALUE_SCALE_MODE",
    "HQ_FRONTIER_SCALE_KWARGS",
    "HQ_FIB_INIT_PROXY_NAME",
    "HQ_BLOCK_SCALE_PROXY_NAME",
    "HQ_PROXY_LEGACY_ALIASES",
    "canonical_hayate_method_name",
    "hayate_result_method_candidates",
    "hayate_hq_role_rates",
    "hayate_hq_code_bits",
    "hayate_hq_kwargs",
    "hayate_frontier_bits",
    "hayate_frontier_kwargs",
    "build_hayate_frontier",
    "build_hayate_hq",
    "hayate_frontier_variant",
]
