"""CPU-only data-plane contract tests for faithful fixed HQ native serving."""

from __future__ import annotations

import pytest

from vllm_plugins.kvcomp.hayate_full_config import (
    HQ_LUT_BINS,
    FullHQNativeRecipe,
    fixed_hq_native_recipe,
    full_hq_native_runtime_fields,
    full_hq_native_recipe_from_mapping,
    paper_hq_native_recipe,
)


@pytest.mark.parametrize(
    ("rate", "role_fields", "slot_bytes"),
    [
        (2, (6, 2, 8, 8), 68),
        (3, (8, 2, 8, 4), 100),
        (4, (5, 1, 6, 2), 132),
    ],
)
def test_original_paper_recipes_and_physical_slots(rate, role_fields, slot_bytes):
    recipe = paper_hq_native_recipe(rate)
    assert (
        recipe.key_code_bits_per_block,
        recipe.key_block_size,
        recipe.value_code_bits_per_block,
        recipe.value_block_size,
    ) == role_fields
    assert recipe.average_rate == rate
    layout = recipe.layout(128)
    assert layout["slot_size"] == slot_bytes
    assert layout["slot_size_aligned"] == slot_bytes
    assert layout["slot_padding_bytes"] == 0
    assert layout["key_scale_offset"] == layout["key_code_bytes"]
    assert layout["value_scale_offset"] == (
        layout["value_codes_offset"] + layout["value_code_bytes"]
    )


@pytest.mark.parametrize(
    ("rate", "role_fields"),
    [(2, (5, 2, 3, 2)), (3, (7, 2, 5, 2)), (4, (9, 2, 7, 2))],
)
def test_frozen_fixed_hq_native_recipe(rate, role_fields):
    recipe = fixed_hq_native_recipe(rate)
    assert (
        recipe.key_code_bits_per_block,
        recipe.key_block_size,
        recipe.value_code_bits_per_block,
        recipe.value_block_size,
    ) == role_fields
    assert recipe.average_rate == rate
    assert recipe.assignment_mode == "exact"
    assert recipe.lut_bins == HQ_LUT_BINS
    contract = recipe.provenance(128)["assignment_contract"]
    assert contract["search"] == "all_codewords"
    assert contract["tie_break"] == "smallest_index"


def test_native_runtime_fields_use_canonical_recipe_serialization():
    recipe = fixed_hq_native_recipe(3)
    fields = full_hq_native_runtime_fields(recipe, 128)
    assert fields["recipe"] == recipe.provenance(128)
    assert fields["average_bits"] == recipe.average_rate == 3
    assert fields["key_code_bits_per_block"] == 7
    assert fields["value_code_bits_per_block"] == 5


@pytest.mark.parametrize(
    ("key_bits", "value_bits", "expected_rate", "slot_bytes"),
    [(5, 3, 2.0, 68), (7, 5, 3.0, 100), (9, 7, 4.0, 132)],
)
def test_fixed_half_bit_delta_b2_candidates_have_same_physical_rate(
    key_bits, value_bits, expected_rate, slot_bytes
):
    recipe = FullHQNativeRecipe(
        key_code_bits_per_block=key_bits,
        key_block_size=2,
        value_code_bits_per_block=value_bits,
        value_block_size=2,
    )
    assert recipe.key_rate == key_bits / 2
    assert recipe.value_rate == value_bits / 2
    assert recipe.average_rate == expected_rate
    assert recipe.layout(128)["slot_size_aligned"] == slot_bytes


def test_rate2_b4_k10v6_is_the_only_opt_in_above_default_safety_cap():
    with pytest.raises(ValueError, match="canonical"):
        FullHQNativeRecipe(10, 4, 6, 4)
    canonical = FullHQNativeRecipe(10, 4, 6, 4, dist="beta", allow_canonical_k10v6_b4=True)
    assert canonical.average_rate == 2.0
    assert canonical.layout(128)["slot_size_aligned"] == 68
    with pytest.raises(ValueError, match="canonical"):
        FullHQNativeRecipe(10, 2, 6, 2, allow_canonical_k10v6_b4=True)


def test_mapping_requires_complete_explicit_role_recipe():
    with pytest.raises(ValueError, match="partial"):
        full_hq_native_recipe_from_mapping({"key_code_bits_per_block": 7, "key_block_size": 2})
    resolved = full_hq_native_recipe_from_mapping(
        {
            "key_code_bits_per_block": 7,
            "key_block_size": 2,
            "value_code_bits_per_block": 5,
            "value_block_size": 2,
        }
    )
    assert resolved.average_rate == 3.0
    assert full_hq_native_recipe_from_mapping({"avg_bits": 3}) == fixed_hq_native_recipe(3)
    assert full_hq_native_recipe_from_mapping(
        {"avg_bits": 3, "recipe": "paper_original"}
    ) == paper_hq_native_recipe(3)


def test_faithful_scale_modes_fail_closed():
    with pytest.raises(ValueError, match="key_scale_mode"):
        paper_hq_native_recipe(3, key_scale_mode="ub")
    with pytest.raises(ValueError, match="value_scale_mode"):
        paper_hq_native_recipe(3, value_scale_mode="rms")


def test_formal_lut_contract_rejects_shape_dependent_or_unfrozen_tables():
    with pytest.raises(ValueError, match="b=2"):
        FullHQNativeRecipe(10, 4, 6, 4, assignment_mode="lut", max_codebook_bits=10)
    with pytest.raises(ValueError, match="lut_bins"):
        FullHQNativeRecipe(7, 2, 5, 2, assignment_mode="lut", lut_bins=128)
    with pytest.raises(ValueError, match="must be one of"):
        FullHQNativeRecipe(7, 2, 5, 2, assignment_mode="lut", lut_candidates=32)
    candidate = FullHQNativeRecipe(7, 2, 5, 2, assignment_mode="lut", lut_candidates=4)
    assert candidate.provenance(128)["assignment_contract"]["candidates_per_cell"] == 4


def test_block_size_must_divide_head_dim_and_provenance_is_explicit():
    recipe = FullHQNativeRecipe(7, 2, 5, 2)
    with pytest.raises(ValueError, match="divide"):
        recipe.layout(127)
    provenance = recipe.provenance(128)
    assert provenance["scale_contract"] == "one_signed_fp16_scale_per_role_vector"
    assert provenance["zero_points"] == 0
    assert provenance["assignment"] == "exact_nearest_codeword_then_value_ls_refit"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_code_bits_per_block", 7.9),
        ("key_block_size", 2.9),
        ("value_code_bits_per_block", "5"),
        ("value_block_size", True),
        ("max_codebook_bits", 9.5),
        ("seed", 1.5),
    ],
)
def test_full_hq_recipe_rejects_lossy_integer_coercion(field, value):
    values = {
        "key_code_bits_per_block": 7,
        "key_block_size": 2,
        "value_code_bits_per_block": 5,
        "value_block_size": 2,
        field: value,
    }
    with pytest.raises(ValueError, match="integer"):
        FullHQNativeRecipe(**values)


def test_full_hq_mapping_accepts_exact_integral_json_float():
    recipe = full_hq_native_recipe_from_mapping({"avg_bits": 3.0})
    assert recipe == fixed_hq_native_recipe(3)


@pytest.mark.parametrize("value", [3.5, "3", True])
def test_full_hq_mapping_rejects_non_integer_average_rate(value):
    with pytest.raises(ValueError, match="integer"):
        full_hq_native_recipe_from_mapping({"avg_bits": value})


@pytest.mark.parametrize("field", ["n_outlier", "key_n_outlier", "value_n_outlier"])
def test_full_native_mapping_rejects_fractional_outlier_counts(field):
    with pytest.raises(ValueError, match=field):
        full_hq_native_recipe_from_mapping({"avg_bits": 3, field: 0.5})


@pytest.mark.parametrize(
    "extra",
    [
        {"calibration": "digest"},
        {"adaptive_bit_split": "online_stats"},
        {"n_outlier": 1},
    ],
)
def test_full_native_mapping_rejects_unsupported_recipe_features(extra):
    with pytest.raises(ValueError):
        full_hq_native_recipe_from_mapping({"avg_bits": 3, **extra})
