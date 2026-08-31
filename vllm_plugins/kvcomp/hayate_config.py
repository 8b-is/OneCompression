"""
Dependency-free contract for the packed HayateQuant vLLM backend.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from vllm_plugins.kvcomp._validation import strict_integer

# Canonical two-coordinate recipes:
# (key code bits/block, key block, value code bits/block, value block).
HAYATEQUANT_NATIVE_RECIPES: dict[int, tuple[int, int, int, int]] = {
    2: (6, 2, 2, 2),
    3: (8, 2, 4, 2),
    4: (10, 2, 6, 2),
}

HAYATEQUANT_LUT_BINS = 256
HAYATEQUANT_LUT_CANDIDATES = 2
HAYATEQUANT_LUT_LIMIT = 5.0


@dataclass(frozen=True)
class HayateQuantNativeRecipe:
    """Resolved fixed-rate HayateQuant native data-plane recipe."""

    rate: int

    def __post_init__(self) -> None:
        rate = strict_integer(self.rate, "HayateQuant native rate")
        if rate not in HAYATEQUANT_NATIVE_RECIPES:
            raise ValueError(
                f"HayateQuant native rate must be one of {sorted(HAYATEQUANT_NATIVE_RECIPES)}"
            )
        object.__setattr__(self, "rate", rate)

    @property
    def key_code_bits_per_block(self) -> int:
        return HAYATEQUANT_NATIVE_RECIPES[self.rate][0]

    @property
    def key_block_size(self) -> int:
        return HAYATEQUANT_NATIVE_RECIPES[self.rate][1]

    @property
    def value_code_bits_per_block(self) -> int:
        return HAYATEQUANT_NATIVE_RECIPES[self.rate][2]

    @property
    def value_block_size(self) -> int:
        return HAYATEQUANT_NATIVE_RECIPES[self.rate][3]

    @property
    def key_rate(self) -> float:
        return self.key_code_bits_per_block / self.key_block_size

    @property
    def value_rate(self) -> float:
        return self.value_code_bits_per_block / self.value_block_size

    @property
    def average_rate(self) -> float:
        return float(self.rate)

    def validate_head_dim(self, head_dim: int) -> None:
        head_dim = strict_integer(head_dim, "head_dim", minimum=1)
        for role, block in (
            ("key", self.key_block_size),
            ("value", self.value_block_size),
        ):
            if head_dim % block:
                raise ValueError(f"{role}_block_size={block} must divide head_dim={head_dim}")

    @staticmethod
    def _role_code_bytes(head_dim: int, code_bits: int, block_size: int) -> int:
        return math.ceil((head_dim // block_size) * code_bits / 8)

    def layout(self, head_dim: int) -> dict[str, int]:
        self.validate_head_dim(head_dim)
        key_bytes = self._role_code_bytes(
            head_dim, self.key_code_bits_per_block, self.key_block_size
        )
        value_bytes = self._role_code_bytes(
            head_dim, self.value_code_bits_per_block, self.value_block_size
        )
        key_scale_offset = key_bytes
        value_codes_offset = key_scale_offset + 2
        value_scale_offset = value_codes_offset + value_bytes
        slot_size = value_scale_offset + 2
        slot_size_aligned = slot_size + (slot_size % 2)
        return {
            "key_code_bytes": key_bytes,
            "key_scale_offset": key_scale_offset,
            "value_codes_offset": value_codes_offset,
            "value_code_bytes": value_bytes,
            "value_scale_offset": value_scale_offset,
            "slot_size": slot_size,
            "slot_padding_bytes": slot_size_aligned - slot_size,
            "slot_size_aligned": slot_size_aligned,
        }

    def provenance(self, head_dim: int) -> dict[str, Any]:
        return {
            "method": "hayatequant",
            "rate": self.rate,
            "key_code_bits_per_block": self.key_code_bits_per_block,
            "value_code_bits_per_block": self.value_code_bits_per_block,
            "block_size": 2,
            "key_rate_bits_per_coordinate": self.key_rate,
            "value_rate_bits_per_coordinate": self.value_rate,
            "average_kv_rate_bits_per_coordinate": self.average_rate,
            "rotation_mode": "auto_rate_dependent",
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "layout": self.layout(head_dim),
            "scale_contract": "one_signed_fp16_scale_per_role_vector",
            "zero_points": 0,
            "assignment": {
                "mode": "fixed_candidate_lut",
                "cell_domain": [-HAYATEQUANT_LUT_LIMIT, HAYATEQUANT_LUT_LIMIT],
                "bins_per_dimension": HAYATEQUANT_LUT_BINS,
                "candidates_per_cell": HAYATEQUANT_LUT_CANDIDATES,
                "tie_break": "smallest_global_codebook_index",
            },
        }


def hayatequant_native_runtime_fields(
    recipe: HayateQuantNativeRecipe, head_dim: int
) -> dict[str, Any]:
    return {
        "method": "hayatequant",
        "recipe": recipe.provenance(head_dim),
        "average_bits": recipe.average_rate,
        "key_code_bits_per_block": recipe.key_code_bits_per_block,
        "value_code_bits_per_block": recipe.value_code_bits_per_block,
    }


def hayatequant_native_recipe(rate: int) -> HayateQuantNativeRecipe:
    return HayateQuantNativeRecipe(rate=rate)


def hayatequant_native_recipe_from_mapping(
    config: Mapping[str, Any],
) -> HayateQuantNativeRecipe:
    """Resolve the minimal process payload ``{"rate": 2|3|4}``."""

    if not isinstance(config, Mapping):
        raise ValueError("HayateQuant native config must be a mapping")
    unknown = sorted(set(config) - {"rate"})
    if unknown:
        raise ValueError(f"non-canonical HayateQuant native fields: {unknown}")
    if "rate" not in config:
        raise ValueError("HayateQuant native config requires rate")
    return hayatequant_native_recipe(config["rate"])


__all__ = [
    "HAYATEQUANT_NATIVE_RECIPES",
    "HAYATEQUANT_LUT_BINS",
    "HAYATEQUANT_LUT_CANDIDATES",
    "HAYATEQUANT_LUT_LIMIT",
    "HayateQuantNativeRecipe",
    "hayatequant_native_runtime_fields",
    "hayatequant_native_recipe",
    "hayatequant_native_recipe_from_mapping",
]
