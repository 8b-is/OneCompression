"""Correctness contracts for TaperKV and the Fujimoto-derived variants."""

import pytest
import torch

from onecomp.kvcomp import (
    BlockTaperKVCache,
    BlockTaperKVPolicy,
    BudgetTaperKVCache,
    BudgetTaperKVPolicy,
    TaperKVPolicy,
    build_policy,
    list_policies,
)
from onecomp.kvcomp.cache.block_taper_cache import block_taper_bits, fake_quant_blocks
from onecomp.kvcomp.cache.taper_cache import apply_byte_budget
from onecomp.kvcomp.codec._quant_primitives import fake_quantize_affine


def test_fujimoto_policy_variants_are_registered_and_constructible():
    assert {"taperkv", "block_taperkv", "budget_taperkv"} <= set(list_policies())
    assert isinstance(build_policy("block_taperkv"), BlockTaperKVPolicy)
    assert isinstance(build_policy("budget_taperkv"), BudgetTaperKVPolicy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key_bits": 0},
        {"value_bits": 9, "key_bits": 8},
        {"b_min": 9, "key_bits": 8},
        {"b_min_value": 5, "value_bits": 4},
        {"sink_tokens": -1},
        {"ladder": [8, 8, 4]},
        {"ladder": [8, 10, 4]},
        {"skip_layers": [True]},
    ],
)
def test_taper_policy_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        TaperKVPolicy(**kwargs)


def test_taper_policy_zero_length_has_defined_rate():
    assert TaperKVPolicy().target_avg_bits(0) == 0.0
    assert BlockTaperKVPolicy().target_avg_bits(0) == 0.0


def test_block_schedule_handles_empty_sequence_and_validates_block_size():
    result = block_taper_bits(0, 16, 4, 64, 8, 2)
    assert result.dtype == torch.int64
    assert result.numel() == 0
    with pytest.raises(ValueError, match="block_size"):
        block_taper_bits(4, 0, 0, 0, 8, 2)


def test_block_quantization_uses_one_affine_pair_per_batch_head_block():
    # Two two-token blocks with a two-coordinate head dimension.  Reducing the
    # final flattened dimension is the only formulation that gives a separate
    # affine pair to each block rather than sharing each coordinate across
    # different blocks.
    x = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0], [10.0, 14.0], [18.0, 22.0]]]], dtype=torch.float32
    )
    bits = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
    actual = fake_quant_blocks(x, bits, block_size=2)
    expected = fake_quantize_affine(x.reshape(1, 1, 2, 4), 2, axis="channel").reshape_as(x)
    assert torch.equal(actual, expected)


def test_block_quantization_refuses_nonuniform_page_layout():
    x = torch.randn(1, 2, 4, 3)
    with pytest.raises(ValueError, match="uniform"):
        fake_quant_blocks(x, torch.tensor([2, 4, 2, 2]), block_size=2)


def test_block_cache_counts_payload_and_metadata_for_every_batch_head():
    policy = BlockTaperKVPolicy(
        key_bits=4,
        value_bits=2,
        b_min=4,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        block_size=2,
        residual_fp16=False,
        quant_granularity="block",
    )
    cache = BlockTaperKVCache(policy)
    keys = torch.randn(2, 3, 4, 5)
    values = torch.randn(2, 3, 4, 5)
    _, _, stored = cache._grade_layer(keys, values, 0)  # pylint: disable=protected-access
    # B=2, H=3 => six independent metadata records per role/page.  There are
    # two pages: K payload 4*4*6*5, V payload 2*4*6*5; K metadata 2*6*(16+4),
    # V metadata 2*6*(16+2).
    expected = 4 * 4 * 6 * 5 + 2 * 4 * 6 * 5 + 2 * 6 * (16 + 4) + 2 * 6 * (16 + 2)
    assert stored == expected


def test_budget_step_uses_whole_batch_payload_not_heads_only():
    kbits = torch.tensor([4, 4], dtype=torch.int64)
    vbits = torch.tensor([4, 4], dtype=torch.int64)
    actual_k, actual_v = apply_byte_budget(
        kbits,
        vbits,
        total=2,
        sink_tokens=0,
        recent_tokens=0,
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        ladder=[4, 2],
        batch_heads=2,
        dim=1,
        max_bits=24,
    )
    assert torch.equal(actual_k, torch.tensor([2, 4]))
    assert torch.equal(actual_v, torch.tensor([2, 4]))


def test_budget_cache_requires_immediate_regrading_and_clears_state_on_reset():
    policy = BudgetTaperKVPolicy(
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        max_bytes=4,
    )
    with pytest.raises(ValueError, match="requant_stride=1"):
        BudgetTaperKVCache(policy, requant_stride=2)
    cache = BudgetTaperKVCache(policy)
    keys = torch.randn(2, 1, 4, 2)
    values = torch.randn(2, 1, 4, 2)
    cache._grade_layer(keys, values, 0)  # pylint: disable=protected-access
    assert 0 in cache._prev_bits  # pylint: disable=protected-access
    cache.reset()
    assert cache._prev_bits == {}  # pylint: disable=protected-access


def test_budget_cache_handles_cache_length_regression_without_shape_error():
    policy = BudgetTaperKVPolicy(
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        max_bytes=8,
    )
    cache = BudgetTaperKVCache(policy)
    cache._grade_layer(torch.randn(1, 1, 5, 2), torch.randn(1, 1, 5, 2), 0)
    cache._grade_layer(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2), 0)


def test_budget_policy_refuses_shape_ambiguous_static_rate():
    with pytest.raises(ValueError, match="shape-dependent"):
        BudgetTaperKVPolicy(max_bytes=1024).target_avg_bits(128)
    assert BudgetTaperKVPolicy(max_bytes=None).target_avg_bits(128) == 6.0
