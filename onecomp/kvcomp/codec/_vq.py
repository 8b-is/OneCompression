"""Backward-compatibility shim.

The block-VQ machinery moved to :mod:`onecomp.kvcomp.codec.hayate.vq`.
Import from there in new code.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .hayate.vq import (  # noqa: F401
    resolve_block_size,
    lloyd_vq_gaussian_block,
    lloyd_vq_beta_block,
    quantize_to_vq_codebook,
    quantize_and_lookup,
    lookup_codewords,
    lut_spec_for,
    build_vq_lut,
    quantize_via_lut,
    _CODEBOOK_DIR,
    _kmeans,
    _kmeans_best_of,
    _beta_block_samples,
    _gaussian_block_samples,
    _gaussian_codebook_artifact,
)
