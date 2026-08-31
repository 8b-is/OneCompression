"""Canonical quality-only external codec baselines.

This registry deliberately contains no HayateQuant recipe.  Proposal methods
are owned by :mod:`onecomp.kvcomp.codec.hayate.recipes`; the definitions here
exist only to make independently implemented external quality baselines hard
to mislabel in benchmark configurations.

FibQuant and BlockQuant are intentionally excluded from codec/inference speed
comparisons in this repository.  Their exact nearest-codeword Python research
implementations are quality baselines, not packed serving kernels.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Iterable

from .blockquant import (
    BLOCKQUANT_CODEBOOK_SPEC,
    BLOCKQUANT_DYNAMIC_OUTLIER_POLICY,
    BLOCKQUANT_REFERENCE,
    BLOCKQUANT_REMAINDER_POLICY,
    BLOCKQUANT_SOURCE_TEX_SHA256,
    BlockQuantCodec,
)
from .fibquant import FIBQUANT_REFERENCE, FIBQUANT_SOURCE_TEX_SHA256, FibQuantCodec

FIBQUANT_PAPER_METHOD = "FibQuant-paper"
FIBQUANT_PAPER_REFERENCE = FIBQUANT_REFERENCE
FIBQUANT_PAPER_RECIPES = MappingProxyType(
    {
        2: (4, 8),
        3: (2, 6),
        4: (2, 8),
    }
)

BLOCKQUANT_PAPER_METHOD = "BlockQuant-paper"
BLOCKQUANT_PAPER_REFERENCE = BLOCKQUANT_REFERENCE
BLOCKQUANT_PAPER_RATE_SLOT = 3
BLOCKQUANT_PAPER_HEAD_DIM = 128

QUALITY_ONLY_BASELINE_METHODS = frozenset({FIBQUANT_PAPER_METHOD, BLOCKQUANT_PAPER_METHOD})


def _paper_rate(bits: int | float) -> int:
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise ValueError(f"FibQuant paper rate must be one of 2/3/4, got {bits!r}")
    value = float(bits)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"FibQuant paper rate must be one of 2/3/4, got {bits!r}")
    rate = int(value)
    if rate not in FIBQUANT_PAPER_RECIPES:
        raise ValueError(f"FibQuant paper rate must be one of 2/3/4, got {bits!r}")
    return rate


def _validated_skip_layers(skip_layers: Iterable[int] | None) -> list[int]:
    if skip_layers is None:
        return []
    if isinstance(skip_layers, (str, bytes)):
        raise ValueError("skip_layers must be an iterable of non-negative layer indices")
    try:
        layers = list(skip_layers)
    except TypeError as exc:
        raise ValueError("skip_layers must be an iterable of non-negative layer indices") from exc
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers):
        raise ValueError("skip_layers must contain only non-negative integer layer indices")
    if len(set(layers)) != len(layers):
        raise ValueError("skip_layers must not contain duplicate layer indices")
    return sorted(layers)


def build_fibquant_paper(
    bits: int | float,
    *,
    skip_layers: Iterable[int] | None = None,
) -> FibQuantCodec:
    """Build the method-faithful FibQuant v1 integer-rate quality baseline."""

    rate = _paper_rate(bits)
    codec = FibQuantCodec(bits=rate, skip_layers=_validated_skip_layers(skip_layers))
    block_size, code_bits = codec._resolved_recipe()  # pylint: disable=protected-access
    if (block_size, code_bits) != FIBQUANT_PAPER_RECIPES[rate]:
        raise RuntimeError("FibQuant paper recipe registry drifted from the codec")
    return codec


def build_blockquant_paper(
    bits: int | float,
    *,
    head_dim: int = BLOCKQUANT_PAPER_HEAD_DIM,
    skip_layers: Iterable[int] | None = None,
) -> BlockQuantCodec:
    """Build the sole paper-declared BlockQuant v1 KV operating point.

    ``bits=3`` is a formal matrix slot, not a claim that every BlockQuant role
    stores exactly three bits per coordinate.  The source specifies a 3.5-bpc
    key point (conditional on channel identities) and a 2-bit value
    approximation.  The versioned dynamic top-L2 completion counts the missing
    identity field, so observed exact stored rate depends on update length.
    """
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise ValueError("BlockQuant-paper is defined only in formal rate slot 3")
    numeric = float(bits)
    if not math.isfinite(numeric) or numeric != float(BLOCKQUANT_PAPER_RATE_SLOT):
        raise ValueError("BlockQuant-paper is defined only in formal rate slot 3")
    if head_dim != BLOCKQUANT_PAPER_HEAD_DIM:
        raise ValueError("BlockQuant-paper v1 KV recipe is defined only for head_dim=128")
    return BlockQuantCodec.llama31_dynamic_top_l2_completion(
        rotation_seed=0,
        codebook_seed=0,
        skip_layers=_validated_skip_layers(skip_layers),
    )


def fibquant_quality_provenance(codec: FibQuantCodec) -> dict:
    """Return fail-closed benchmark metadata for a canonical FibQuant row."""

    if not isinstance(codec, FibQuantCodec):
        raise TypeError("FibQuant-paper must be backed by FibQuantCodec")
    rate = _paper_rate(codec.bits)
    expected = FIBQUANT_PAPER_RECIPES[rate]
    actual = codec._resolved_recipe()  # pylint: disable=protected-access
    if actual != expected or codec.seed != 0 or codec.codebook_seed != 0:
        raise ValueError(
            "FibQuant-paper requires the v1 integer recipe and fixed seed=codebook_seed=0"
        )
    return {
        "method_role": "external_quality_baseline",
        "proposal_method": False,
        "paper_comparable": True,
        "paper_reproduction": False,
        "speed_comparison": False,
        "primary_source": FIBQUANT_PAPER_REFERENCE,
        "source_tex_sha256": FIBQUANT_SOURCE_TEX_SHA256,
        "random_seed_protocol": (
            "fixed rotation seed=0 and deterministic codebook completion seed=0; "
            "the v1 source does not publish its seeds or centroid artifact"
        ),
        "method_definition": (
            "FibQuant v1 exact spherical-Beta block-VQ quality baseline "
            f"({rate} payload bpc; k={actual[0]}, code_bits={actual[1]})"
        ),
        "nominal_payload_bits_per_coordinate": rate,
    }


def blockquant_quality_provenance(codec: BlockQuantCodec) -> dict:
    """Return fail-closed scope metadata for the fixed-seed BlockQuant row."""
    if not isinstance(codec, BlockQuantCodec):
        raise TypeError("BlockQuant-paper must be backed by BlockQuantCodec")
    expected = build_blockquant_paper(BLOCKQUANT_PAPER_RATE_SLOT, skip_layers=codec.skip_layers)
    if codec.get_kv_config() != expected.get_kv_config():
        raise ValueError("BlockQuant-paper codec differs from the versioned formal completion")
    return {
        "method_role": "external_quality_baseline",
        "external_quality_baseline": True,
        "proposal_method": False,
        "paper_comparable": False,
        "paper_reproduction": False,
        "speed_comparison": False,
        "primary_source": BLOCKQUANT_PAPER_REFERENCE,
        "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
        "formal_rate_slot": BLOCKQUANT_PAPER_RATE_SLOT,
        "paper_declared_key_bits_per_coordinate": 3.5,
        "paper_declared_value_payload_bits_per_coordinate": 2,
        "top32_completion": BLOCKQUANT_DYNAMIC_OUTLIER_POLICY,
        "remainder_completion": BLOCKQUANT_REMAINDER_POLICY,
        "codebook_completion": BLOCKQUANT_CODEBOOK_SPEC,
        "random_seed_protocol": (
            "fixed rotation_seed=0 and codebook_seed=0 same-runtime quality row; "
            "not the paper five-random-seed aggregate"
        ),
        "method_definition": (
            "BlockQuant v1 exact-search fixed-seed quality baseline at the paper-declared "
            "Llama-3.1 KV point, with versioned dynamic per-head top-L2 identities and "
            "lower-dimensional p=3 tails; identity bits are counted"
        ),
        "nominal_payload_bits_per_coordinate": None,
    }


def is_quality_only_baseline(method: str) -> bool:
    """Return whether ``method`` is intentionally excluded from speed rows."""

    return method in QUALITY_ONLY_BASELINE_METHODS


__all__ = [
    "BLOCKQUANT_PAPER_HEAD_DIM",
    "BLOCKQUANT_PAPER_METHOD",
    "BLOCKQUANT_PAPER_RATE_SLOT",
    "BLOCKQUANT_PAPER_REFERENCE",
    "FIBQUANT_PAPER_METHOD",
    "FIBQUANT_PAPER_RECIPES",
    "FIBQUANT_PAPER_REFERENCE",
    "QUALITY_ONLY_BASELINE_METHODS",
    "blockquant_quality_provenance",
    "build_blockquant_paper",
    "build_fibquant_paper",
    "fibquant_quality_provenance",
    "is_quality_only_baseline",
]
