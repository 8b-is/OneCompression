"""CPU semantic gate for the shared-rotation full-HQ store fast path."""

from __future__ import annotations

import torch

from onecomp.kvcomp.codec.hayate.codec import HayateQuantCodec, _block_codebook_by_code_bits
from vllm_plugins.kvcomp.hayate_full_config import fixed_hq_native_recipe
from vllm_plugins.kvcomp.hayate_full_triton import (
    _encode_role,
    _encode_roles_with_shared_rotation,
)


def test_shared_rotation_keeps_exact_codes_and_fp16_scales():
    """One merged GEMM must be bit-identical to two role-local rotations."""

    recipe = fixed_hq_native_recipe(3)
    resolver = HayateQuantCodec(
        bits=4,
        value_bits=2,
        dist=recipe.dist,
        block_size=None,
        max_codebook_bits=recipe.max_codebook_bits,
        codebook_init=recipe.codebook_init,
        rotation_mode=recipe.rotation_mode,
        encoder="exact",
        key_scale_mode="nc",
        value_scale_mode="ls_iter",
        seed=recipe.seed,
    )
    rotation = resolver._rt_rotation(  # pylint: disable=protected-access
        128, recipe.key_block_size, recipe.key_rate, torch.device("cpu"), role="k"
    ).contiguous()
    # This is deliberately a b=3 recipe property.  The runtime uses aliasing,
    # not numerical equality, before selecting the fast path.
    value_rotation = resolver._rt_rotation(  # pylint: disable=protected-access
        128, recipe.value_block_size, recipe.value_rate, torch.device("cpu"), role="v"
    ).contiguous()
    assert rotation.data_ptr() == value_rotation.data_ptr()
    key_codebook = _block_codebook_by_code_bits(
        recipe.key_code_bits_per_block,
        recipe.key_block_size,
        recipe.dist,
        128,
        recipe.codebook_init,
    ).float()
    value_codebook = _block_codebook_by_code_bits(
        recipe.value_code_bits_per_block,
        recipe.value_block_size,
        recipe.dist,
        128,
        recipe.codebook_init,
    ).float()
    torch.manual_seed(7)
    key = torch.randn(5, 2, 128, dtype=torch.bfloat16)
    value = torch.randn_like(key)

    shared = _encode_roles_with_shared_rotation(
        key,
        value,
        rotation,
        key_codebook,
        value_codebook,
        recipe,
        key_lut=None,
        value_lut=None,
        key_scoring_codebook=None,
        key_scoring_csq=None,
        value_scoring_codebook=None,
        value_scoring_csq=None,
    )
    key_expected = _encode_role(
        key,
        rotation,
        key_codebook,
        recipe.key_block_size,
        "nc",
        lut=None,
        lut_bins=recipe.lut_bins,
        lut_limit=recipe.lut_limit,
        scoring_codebook=None,
        scoring_csq=None,
    )
    value_expected = _encode_role(
        value,
        value_rotation,
        value_codebook,
        recipe.value_block_size,
        "ls_iter",
        lut=None,
        lut_bins=recipe.lut_bins,
        lut_limit=recipe.lut_limit,
        scoring_codebook=None,
        scoring_csq=None,
    )

    assert torch.equal(shared[0], key_expected[0])
    assert torch.equal(shared[1], key_expected[1])
    assert torch.equal(shared[2], value_expected[0])
    assert torch.equal(shared[3], value_expected[1])
