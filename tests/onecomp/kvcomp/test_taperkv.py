"""Focused correctness contracts for standalone TaperKV caches."""

from types import SimpleNamespace

import pytest
import torch

from onecomp.kvcomp import KVCompressionConfig
from onecomp.kvcomp.cache.block_taper_cache import (
    BlockTaperKVCache,
    block_taper_bits,
    fake_quant_blocks,
)
from onecomp.kvcomp.cache.builders import build_compressed_cache
from onecomp.kvcomp.cache.taper_cache import (
    BudgetTaperKVCache,
    TaperKVCache,
    apply_byte_budget,
    taper_bits,
)
from onecomp.kvcomp.codec._quant_primitives import fake_quantize_affine
from onecomp.kvcomp.codec.hayate.recipes import build_hayatequant
from onecomp.kvcomp.recipe.taperkv import (
    BlockTaperKVPolicy,
    BudgetTaperKVPolicy,
    TaperKVPolicy,
)


def test_taper_schedule_has_independent_key_value_ladders_and_protection():
    policy = TaperKVPolicy(
        key_bits=8,
        value_bits=4,
        b_min=4,
        b_min_value=2,
        sink_tokens=1,
        recent_tokens=2,
        key_ladder=[8, 6, 4],
        value_ladder=[4, 3, 2],
    )
    key_bits, value_bits = TaperKVCache(policy).token_bits(10, torch.device("cpu"), 0)

    assert key_bits[0] == 8 and value_bits[0] == 4
    assert torch.equal(key_bits[-2:], torch.tensor([8, 8]))
    assert torch.equal(value_bits[-2:], torch.tensor([4, 4]))
    assert key_bits.min() == 4
    assert value_bits.min() == 2
    assert torch.all(key_bits >= value_bits)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key_bits": 0},
        {"key_bits": 4, "value_bits": 5},
        {"key_bits": 4, "b_min": 5},
        {"value_bits": 4, "b_min_value": 5},
        {"sink_tokens": -1},
        {"key_ladder": [8, 8, 4]},
        {"key_ladder": [7, 4]},
        {"key_ladder": [8, 4], "value_ladder": [4, 3, 2, 1]},
        {"skip_layers": [True]},
        {"requant_stride": 0},
    ],
)
def test_taper_policy_rejects_invalid_schedules(kwargs):
    with pytest.raises(ValueError):
        TaperKVPolicy(**kwargs)


def test_taper_schedule_and_block_schedule_validate_empty_inputs():
    empty = torch.empty(0, dtype=torch.int64)
    assert taper_bits(empty, 0, 0, 0, 8, 4).numel() == 0
    assert block_taper_bits(0, 16, 0, 0, 8, 4).numel() == 0
    with pytest.raises(ValueError, match="block_size"):
        block_taper_bits(4, 0, 0, 0, 8, 4)


def test_block_quantization_uses_one_affine_pair_per_head_and_page():
    values = torch.tensor([[[[0.0, 1.0], [2.0, 3.0], [10.0, 14.0], [18.0, 22.0]]]])
    bits = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
    actual = fake_quant_blocks(values, bits, block_size=2)
    expected = fake_quantize_affine(values.reshape(1, 1, 2, 4), 2, axis="channel").reshape_as(
        values
    )
    assert torch.equal(actual, expected)

    with pytest.raises(ValueError, match="uniform"):
        fake_quant_blocks(
            torch.randn(1, 1, 4, 2),
            torch.tensor([2, 4, 2, 2]),
            block_size=2,
        )


def test_block_cache_regrades_when_an_unaligned_prefill_page_becomes_full():
    policy = BlockTaperKVPolicy(
        key_bits=4,
        value_bits=2,
        b_min=4,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        block_size=4,
        residual_fp16=True,
        requant_stride=16,
    )
    cache = BlockTaperKVCache(policy)

    cache.update(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2), 0)
    assert torch.equal(cache._layer_token_bits[0][0], torch.full((3,), 16))
    cache.update(torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2), 0)

    key_bits, value_bits = cache._layer_token_bits[0]
    assert torch.equal(key_bits, torch.full((4,), 4))
    assert torch.equal(value_bits, torch.full((4,), 2))
    # Payload plus one scale/zero record per role for the single page/head.
    assert cache.stored_bits == (4 * 4 * 2) + (2 * 4 * 2) + (16 + 4) + (16 + 2)
    assert cache.fp16_bits == 4 * 2 * 2 * 16


def test_regrading_at_the_same_width_is_bit_exact_and_does_not_accumulate_drift():
    policy = TaperKVPolicy(
        key_bits=3,
        value_bits=2,
        b_min=3,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        requant_stride=1,
    )
    cache = TaperKVCache(policy)
    key = torch.randn(1, 2, 5, 8)
    value = torch.randn_like(key)
    first_key, first_value = cache.update(key, value, 0)
    first_key = first_key.clone()
    first_value = first_value.clone()

    second_key, second_value, _ = cache._grade_layer(first_key, first_value, 0)
    assert torch.equal(second_key, first_key)
    assert torch.equal(second_value, first_value)


def test_lifecycle_operations_keep_the_precision_and_memory_ledgers_exact():
    policy = TaperKVPolicy(
        key_bits=4,
        value_bits=2,
        b_min=2,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
    )
    cache = TaperKVCache(policy)
    cache.update(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4), 0)
    cache.crop(4)
    assert cache.get_seq_length() == 4
    assert cache._layer_token_bits[0][0].numel() == 4
    assert cache.fp16_bits == 1 * 2 * 4 * 4 * 2 * 16

    cache.batch_repeat_interleave(3)
    assert cache.fp16_bits == 3 * 2 * 4 * 4 * 2 * 16
    cache.batch_select_indices(torch.tensor([2, 0]))
    assert cache.fp16_bits == 2 * 2 * 4 * 4 * 2 * 16
    assert cache.stored_bits > 0


def test_budget_lowers_oldest_tokens_and_enforces_the_payload_cap():
    key = torch.full((4,), 4, dtype=torch.int64)
    value = torch.full((4,), 4, dtype=torch.int64)
    lowered_key, lowered_value = apply_byte_budget(
        key,
        value,
        total=4,
        sink_tokens=0,
        recent_tokens=0,
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        key_ladder=[4, 3, 2],
        value_ladder=[4, 3, 2],
        batch_heads=1,
        dim=2,
        max_bits=60,
    )
    assert torch.equal(lowered_key, torch.tensor([3, 4, 4, 4]))
    assert torch.equal(lowered_value, torch.tensor([3, 4, 4, 4]))

    policy = BudgetTaperKVPolicy(
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        max_bytes_per_layer=5,
    )
    cache = BudgetTaperKVCache(policy)
    cache.update(torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2), 0)
    key_bits, value_bits = cache._layer_token_bits[0]
    payload_bits = int((key_bits.sum() + value_bits.sum()) * 2)
    assert payload_bits <= 5 * 8
    assert cache.stored_bits > payload_bits  # affine metadata is explicit.


def test_budget_fails_when_protected_tokens_make_the_cap_impossible():
    policy = BudgetTaperKVPolicy(
        key_bits=4,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        sink_tokens=1,
        recent_tokens=1,
        max_bytes_per_layer=1,
    )
    with pytest.raises(ValueError, match="infeasible"):
        BudgetTaperKVCache(policy).update(
            torch.randn(1, 1, 2, 2),
            torch.randn(1, 1, 2, 2),
            0,
        )


def test_taper_is_standalone_in_the_huggingface_cache_builder():
    config = KVCompressionConfig(
        codec=build_hayatequant(3),
        policy=TaperKVPolicy(),
    )
    with pytest.raises(NotImplementedError, match="standalone"):
        build_compressed_cache(SimpleNamespace(config=None), config)
