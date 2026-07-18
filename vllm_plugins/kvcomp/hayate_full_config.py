# Copyright 2026 Fujitsu Ltd.
"""Pure-Python contract for the faithful packed HayateQuant vLLM backend.

This module deliberately imports neither torch nor vLLM.  Cache allocation is
decided before an attention implementation exists, so the packed layout and
its validation must be usable from vLLM's backend class method as well as from
CPU-only tests and benchmark provenance code.

The fixed paper recipe stores one fixed-width block-code index per block and
one signed fp16 scale per vector *for each role*::

    [K code stream | K fp16 scale | V code stream | V fp16 scale | padding]

There is no affine zero point.  The role rates may be integer (the original
K3/V1, K4/V2, K5/V3 operating points) or fixed fractional rates expressed as
an integer number of code bits per block (for example b=2, K5/V3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from vllm_plugins.kvcomp._validation import strict_integer

# Original paper operating points.  Entries are
# (key_code_bits, key_block_size, value_code_bits, value_block_size).
PAPER_HQ_NATIVE_RECIPES: dict[int, tuple[int, int, int, int]] = {
    2: (6, 2, 8, 8),  # K3 / V1
    3: (8, 2, 8, 4),  # K4 / V2
    4: (5, 1, 6, 2),  # K5 / V3
}

# Frozen strengthened HQ recipe selected by the preregistered 21-cell
# development sweep.  This remains the paper method (data-free rotation,
# short block-VQ, fixed role-aware allocation, NC/LS-refit); only its fixed
# rate allocation/block shape is strengthened.  Formal native runs resolve
# ``avg_bits`` through this table unless they explicitly request the
# ``paper_original`` ablation.
FIXED_HQ_NATIVE_RECIPES: dict[int, tuple[int, int, int, int]] = {
    2: (5, 2, 3, 2),
    3: (7, 2, 5, 2),
    4: (9, 2, 7, 2),
}

# Fixed table candidates under the formal quality/parity gate.  The accepted
# HQ recipe remains exact until one candidate passes; neither T is selected
# per model or workload.
HQ_LUT_BINS = 256
HQ_LUT_CANDIDATE_OPTIONS = (2, 4, 8, 16)
HQ_LUT_LIMIT = 5.0


@dataclass(frozen=True)
class FullHQNativeRecipe:
    """Resolved fixed-rate HQ native data-plane recipe."""

    key_code_bits_per_block: int
    key_block_size: int
    value_code_bits_per_block: int
    value_block_size: int
    dist: str = "gaussian"
    codebook_init: str = "auto"
    rotation_mode: str = "auto"
    key_scale_mode: str = "nc"
    value_scale_mode: str = "ls_iter"
    # ``exact`` is the canonical frozen HQ assignment.  ``lut`` is an
    # explicitly recorded development implementation candidate and cannot
    # become canonical until it passes the pinned all-rate gates.
    assignment_mode: str = "exact"
    lut_bins: int = HQ_LUT_BINS
    lut_candidates: int = HQ_LUT_CANDIDATE_OPTIONS[0]
    lut_limit: float = HQ_LUT_LIMIT
    seed: int = 0
    max_codebook_bits: int = 9
    # The safety cap stays at 9 globally.  The sole opt-in exception is the
    # preregistered fixed rate-2 candidate b=4 K10/V6; arbitrary 10-bit
    # codebooks remain rejected.
    allow_canonical_k10v6_b4: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_canonical_k10v6_b4, bool):
            raise ValueError("allow_canonical_k10v6_b4 must be a bool")
        max_codebook_bits = strict_integer(
            self.max_codebook_bits, "max_codebook_bits", minimum=1, maximum=12
        )
        object.__setattr__(self, "max_codebook_bits", max_codebook_bits)
        for name in ("key_block_size", "value_block_size"):
            value = strict_integer(getattr(self, name), name, minimum=1)
            if value not in (1, 2, 4, 8):
                raise ValueError(f"{name} must be one of 1,2,4,8, got {value!r}")
            object.__setattr__(self, name, value)
        for name in ("key_code_bits_per_block", "value_code_bits_per_block"):
            object.__setattr__(self, name, strict_integer(getattr(self, name), name, minimum=1))
        canonical_k10 = (
            self.allow_canonical_k10v6_b4
            and self.key_code_bits_per_block == 10
            and self.key_block_size == 4
            and self.value_code_bits_per_block == 6
            and self.value_block_size == 4
        )
        for name in (
            "key_code_bits_per_block",
            "value_code_bits_per_block",
        ):
            value = getattr(self, name)
            allowed_above_cap = canonical_k10 and name == "key_code_bits_per_block"
            if value > self.max_codebook_bits and not allowed_above_cap:
                raise ValueError(
                    f"{name} must be in 1..max_codebook_bits "
                    f"({self.max_codebook_bits}), except explicit canonical "
                    f"b4-K10/V6, got {value!r}"
                )
        if self.allow_canonical_k10v6_b4 and not canonical_k10:
            raise ValueError(
                "allow_canonical_k10v6_b4 is valid only for the exact fixed "
                "key=(10 bits,b4), value=(6 bits,b4) recipe"
            )
        if self.dist not in {"gaussian", "beta"}:
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if self.codebook_init not in {"auto", "forgy", "fibquant"}:
            raise ValueError("codebook_init must be 'auto', 'forgy', or 'fibquant'")
        if self.rotation_mode not in {"auto", "hybrid", "hadamard_gemm", "dense"}:
            raise ValueError("unsupported rotation_mode")
        if self.key_scale_mode != "nc":
            raise ValueError("faithful fixed HQ native keys require key_scale_mode='nc'")
        if self.value_scale_mode not in {"ls_iter", "ls_refit"}:
            raise ValueError("faithful fixed HQ native values require value_scale_mode='ls_iter'")
        if self.assignment_mode not in {"exact", "lut"}:
            raise ValueError("assignment_mode must be 'exact' or 'lut'")
        for name in ("lut_bins", "lut_candidates"):
            object.__setattr__(self, name, strict_integer(getattr(self, name), name, minimum=1))
        object.__setattr__(self, "seed", strict_integer(self.seed, "seed"))
        if isinstance(self.lut_limit, bool):
            raise ValueError("lut_limit must be a positive finite number")
        try:
            lut_limit = float(self.lut_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("lut_limit must be a positive finite number") from exc
        if not math.isfinite(lut_limit) or lut_limit <= 0.0:
            raise ValueError("lut_limit must be a positive finite number")
        object.__setattr__(self, "lut_limit", lut_limit)
        if self.assignment_mode == "lut":
            if self.key_block_size != 2 or self.value_block_size != 2:
                raise ValueError(
                    "formal HQ LUT assignment is defined only for the frozen "
                    "short b=2 K/V recipe"
                )
            if self.lut_bins != HQ_LUT_BINS:
                raise ValueError(f"HQ LUT candidates require lut_bins={HQ_LUT_BINS}")
            if self.lut_candidates not in HQ_LUT_CANDIDATE_OPTIONS:
                raise ValueError(
                    "HQ LUT candidate count must be one of " f"{HQ_LUT_CANDIDATE_OPTIONS}"
                )
            if float(self.lut_limit) != HQ_LUT_LIMIT:
                raise ValueError(f"HQ LUT candidates require lut_limit={HQ_LUT_LIMIT}")

    @property
    def key_rate(self) -> float:
        return self.key_code_bits_per_block / self.key_block_size

    @property
    def value_rate(self) -> float:
        return self.value_code_bits_per_block / self.value_block_size

    @property
    def average_rate(self) -> float:
        return (self.key_rate + self.value_rate) / 2.0

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
        total_bits = (head_dim // block_size) * code_bits
        return math.ceil(total_bits / 8)

    def layout(self, head_dim: int) -> dict[str, int]:
        """Return exact byte offsets and physical aligned slot size."""

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
        # vLLM derives an effective fp16 head size from slot bytes, therefore
        # the backend follows upstream TurboQuant's even-byte alignment.
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
        result = asdict(self)
        assignment = (
            f"grid_lut_t{self.lut_candidates}_exact_candidate_distance_then_value_ls_refit"
            if self.assignment_mode == "lut"
            else "exact_nearest_codeword_then_value_ls_refit"
        )
        result.update(
            {
                "key_rate_bits_per_coordinate": self.key_rate,
                "value_rate_bits_per_coordinate": self.value_rate,
                "average_kv_rate_bits_per_coordinate": self.average_rate,
                "layout": self.layout(head_dim),
                "scale_contract": "one_signed_fp16_scale_per_role_vector",
                "zero_points": 0,
                "assignment": assignment,
                "assignment_contract": (
                    {
                        "cell_domain": [
                            -HQ_LUT_LIMIT,
                            HQ_LUT_LIMIT,
                        ],
                        "bins_per_dimension": self.lut_bins,
                        "candidates_per_cell": self.lut_candidates,
                        "cell_rule": (
                            "clip_then_floor; interior boundaries belong to "
                            "the upper cell; +limit clips to the last cell"
                        ),
                        "candidate_order": "ascending_global_codebook_index",
                        "distance": "canonical_fp16_gemm_score_restricted_to_candidates",
                        "tie_break": "smallest_global_codebook_index",
                        "shape_independent": True,
                    }
                    if self.assignment_mode == "lut"
                    else {"search": "all_codewords", "tie_break": "smallest_index"}
                ),
            }
        )
        return result


def full_hq_native_runtime_fields(
    recipe: FullHQNativeRecipe, head_dim: int
) -> dict[str, Any]:
    """Return bounded JSON-native fields shared by native runtime events."""

    return {
        "recipe": recipe.provenance(head_dim),
        "average_bits": recipe.average_rate,
        "key_code_bits_per_block": recipe.key_code_bits_per_block,
        "value_code_bits_per_block": recipe.value_code_bits_per_block,
    }


def paper_hq_native_recipe(avg_bits: int, **overrides: Any) -> FullHQNativeRecipe:
    """Build one of the original paper's fixed role-aware operating points."""

    rate = strict_integer(avg_bits, "paper HQ native avg_bits")
    if rate not in PAPER_HQ_NATIVE_RECIPES:
        raise ValueError(
            f"paper HQ native avg_bits must be one of {sorted(PAPER_HQ_NATIVE_RECIPES)}"
        )
    kcb, kb, vcb, vb = PAPER_HQ_NATIVE_RECIPES[rate]
    values: dict[str, Any] = {
        "key_code_bits_per_block": kcb,
        "key_block_size": kb,
        "value_code_bits_per_block": vcb,
        "value_block_size": vb,
    }
    values.update(overrides)
    return FullHQNativeRecipe(**values)


def fixed_hq_native_recipe(avg_bits: int, **overrides: Any) -> FullHQNativeRecipe:
    """Build the frozen strengthened fixed HQ operating point."""

    rate = strict_integer(avg_bits, "fixed HQ native avg_bits")
    if rate not in FIXED_HQ_NATIVE_RECIPES:
        raise ValueError(
            f"fixed HQ native avg_bits must be one of {sorted(FIXED_HQ_NATIVE_RECIPES)}"
        )
    kcb, kb, vcb, vb = FIXED_HQ_NATIVE_RECIPES[rate]
    values: dict[str, Any] = {
        "key_code_bits_per_block": kcb,
        "key_block_size": kb,
        "value_code_bits_per_block": vcb,
        "value_block_size": vb,
    }
    values.update(overrides)
    return FullHQNativeRecipe(**values)


def full_hq_native_recipe_from_mapping(config: Mapping[str, Any]) -> FullHQNativeRecipe:
    """Resolve a strict environment/serialized mapping into a fixed recipe.

    Supplying all four role fields takes precedence and supports preregistered
    fixed fractional candidates.  Otherwise ``avg_bits`` selects the original
    paper recipe.  Partial role declarations fail closed.
    """

    if config.get("calibration"):
        raise ValueError("full HQ native currently supports data-free rotations only")
    if config.get("adaptive_bit_split") not in (None, "", "off", "none", False):
        raise ValueError(
            "full HQ native is fixed-rate; Adaptive HQ requires a cache-local selector"
        )
    for name in ("n_outlier", "key_n_outlier", "value_n_outlier"):
        raw_value = config.get(name, 0)
        value = strict_integer(0 if raw_value is None else raw_value, name, minimum=0)
        if value:
            raise ValueError("full HQ native fixed recipe does not support dynamic outlier masks")

    fields = (
        "key_code_bits_per_block",
        "key_block_size",
        "value_code_bits_per_block",
        "value_block_size",
    )
    present = [name for name in fields if config.get(name) is not None]
    common = {
        name: config[name]
        for name in (
            "dist",
            "codebook_init",
            "rotation_mode",
            "key_scale_mode",
            "value_scale_mode",
            "assignment_mode",
            "lut_bins",
            "lut_candidates",
            "lut_limit",
            "seed",
            "max_codebook_bits",
            "allow_canonical_k10v6_b4",
        )
        if config.get(name) is not None
    }
    if present:
        if len(present) != len(fields):
            missing = sorted(set(fields) - set(present))
            raise ValueError(f"partial full-HQ native role recipe; missing {missing}")
        values = {name: strict_integer(config[name], name) for name in fields}
        values.update(common)
        return FullHQNativeRecipe(**values)
    if config.get("avg_bits") is None:
        raise ValueError("full-HQ native config requires avg_bits or all four role block fields")
    recipe_name = str(config.get("recipe") or "hq").strip().lower().replace("-", "_")
    if recipe_name in {"hq", "fixed_hq"}:
        return fixed_hq_native_recipe(config["avg_bits"], **common)
    if recipe_name in {"paper_original", "original_paper"}:
        return paper_hq_native_recipe(config["avg_bits"], **common)
    raise ValueError("full-HQ native recipe must be 'hq' or 'paper_original'")


__all__ = [
    "PAPER_HQ_NATIVE_RECIPES",
    "FIXED_HQ_NATIVE_RECIPES",
    "HQ_LUT_BINS",
    "HQ_LUT_CANDIDATE_OPTIONS",
    "HQ_LUT_LIMIT",
    "FullHQNativeRecipe",
    "full_hq_native_runtime_fields",
    "paper_hq_native_recipe",
    "fixed_hq_native_recipe",
    "full_hq_native_recipe_from_mapping",
]
