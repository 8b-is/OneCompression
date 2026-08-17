from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

pytest.importorskip("vllm")

from onecomp.kvcomp.cache.block_taper_cache import block_taper_bits  # noqa: E402
from onecomp.kvcomp.cache.taper_cache import taper_bits  # noqa: E402
from vllm_plugins.kvcomp.taper_backend import (  # noqa: E402
    _UPSTREAM_KV_AXIS,
    _assert_unique_indices,
    _batched_taper_bits,
    _fake_quantize_variable_pages,
    _fake_quantize_variable_tokens,
    _transition_candidate_positions,
    TaperKVImpl,
)
from vllm_plugins.kvcomp.taper_config import TaperKVVLLMRecipe  # noqa: E402
from vllm_plugins.kvcomp.hayate_native import _all_first_chunk_prefills  # noqa: E402


def _kv_cache(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.stack((key, value), dim=_UPSTREAM_KV_AXIS)


def test_hayate_prefill_fast_path_checks_every_request():
    assert _all_first_chunk_prefills([0, 100, 125], [100, 25])
    # The maxima are both 100, but request 1 has 40 cached tokens.
    assert not _all_first_chunk_prefills([0, 100, 110], [100, 50])
    with pytest.raises(ValueError, match="incompatible"):
        _all_first_chunk_prefills([0, 10], [10, 10])


@pytest.mark.parametrize(
    ("high", "floor", "sink", "recent", "ladder"),
    [
        (8, 4, 4, 3, (8, 7, 6, 5, 4)),
        (4, 2, 0, 2, (4, 3, 2)),
        (8, 4, 1, 0, (8, 6, 4)),
    ],
)
def test_batched_schedule_matches_huggingface_reference(high, floor, sink, recent, ladder):
    lengths = torch.tensor([17, 9], dtype=torch.long)
    positions = torch.arange(17).expand(2, -1)
    actual = _batched_taper_bits(
        positions,
        lengths,
        high,
        floor,
        sink,
        recent,
        ladder,
    )

    for row, length in enumerate(lengths.tolist()):
        age = torch.arange(length - 1, -1, -1)
        expected = taper_bits(age, length, sink, recent, high, floor, ladder)
        assert torch.equal(actual[row, :length], expected)


@pytest.mark.parametrize(
    ("high", "floor", "sink", "recent", "ladder"),
    [
        (8, 4, 4, 128, (8, 7, 6, 5, 4)),
        (4, 2, 4, 128, (4, 3, 2)),
        (8, 4, 1, 0, (8, 6, 4)),
    ],
)
def test_transition_candidates_cover_every_width_drop(
    high,
    floor,
    sink,
    recent,
    ladder,
):
    stride = 16
    for previous_length in range(1, 513, 7):
        for delta in (1, 3, stride):
            current_length = previous_length + delta
            previous_positions = torch.arange(previous_length).unsqueeze(0)
            current_positions = torch.arange(current_length).unsqueeze(0)
            previous = _batched_taper_bits(
                previous_positions,
                torch.tensor([previous_length]),
                high,
                floor,
                sink,
                recent,
                ladder,
            )[0]
            current = _batched_taper_bits(
                current_positions,
                torch.tensor([current_length]),
                high,
                floor,
                sink,
                recent,
                ladder,
            )[0]
            lowered = set(torch.nonzero(current[:previous_length] < previous).flatten().tolist())
            positions, valid = _transition_candidate_positions(
                torch.tensor([current_length]),
                ladder_count=len(ladder),
                sink_tokens=sink,
                recent_tokens=recent,
                stride=stride,
            )
            candidates = set(positions[valid].tolist())
            assert lowered <= candidates


def test_variable_token_and_page_quantizers_have_expected_granularity():
    rows = torch.tensor(
        [
            [[-1.0, -0.2, 0.4, 1.0], [2.0, 2.0, 2.0, 2.0]],
            [[-3.0, -0.1, 0.2, 2.0], [1.0, 2.0, 4.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    token_result = _fake_quantize_variable_tokens(rows, torch.tensor([2, 4]))
    assert torch.equal(token_result[0, 1], rows[0, 1])
    assert not torch.equal(token_result[0, 0], rows[0, 0])

    pages = rows.reshape(1, 2, 2, 4)
    page_result = _fake_quantize_variable_pages(pages, torch.tensor([2]))
    # Each head shares one scale across both page positions.
    assert page_result.shape == pages.shape
    assert torch.equal(page_result[0, :, 1], pages[0, :, 1]) is False


def test_monotonic_token_ledger_skips_same_width_requantization(monkeypatch):
    torch.manual_seed(1)
    cache = torch.randn(2, 4, 2, 8)
    ledger = torch.full((8,), 16, dtype=torch.int8)
    slots = torch.tensor([0, 3, 6])
    requested = torch.tensor([8, 6, 4])

    TaperKVImpl._lower_tokens(cache, ledger, slots, requested)
    first = cache.clone()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("same-width data was quantized again")

    monkeypatch.setattr(
        "vllm_plugins.kvcomp.taper_backend._fake_quantize_variable_tokens",
        forbidden,
    )
    TaperKVImpl._lower_tokens(cache, ledger, slots, requested)

    assert torch.equal(cache, first)
    assert ledger.tolist() == [8, 16, 16, 6, 16, 16, 4, 16]


def test_token_lowering_writes_through_strided_cache_view():
    source_key = torch.tensor(
        [
            [
                [[-1.0, -0.7, 0.1, 1.0]],
                [[-2.0, -0.1, 0.3, 2.0]],
            ],
            [
                [[-3.0, -0.5, 0.2, 3.0]],
                [[-4.0, -0.2, 0.7, 4.0]],
            ],
        ],
        dtype=torch.float32,
    )
    source_value = source_key.add(10)
    # vLLM 0.24 interleaves K/V on axis 1, while older supported versions use
    # axis 0. Construct the problematic interleaved view explicitly so this
    # regression remains effective with either test dependency version.
    kv_cache = torch.stack((source_key.clone(), source_value.clone()), dim=1)
    key_cache = kv_cache[:, 0]
    assert not key_cache.is_contiguous()
    ledger = torch.full((4,), 16, dtype=torch.int8)
    slots = torch.tensor([2])
    widths = torch.tensor([2])
    expected = _fake_quantize_variable_tokens(source_key[1, :1], widths)[0]

    TaperKVImpl._lower_tokens(key_cache, ledger, slots, widths)
    actual_key, actual_value = kv_cache[:, 0], kv_cache[:, 1]

    assert torch.equal(actual_key[1, 0], expected)
    assert torch.equal(actual_value, source_value)
    assert ledger.tolist() == [16, 16, 2, 16]


def test_forced_token_lowering_resets_reused_slot_width():
    cache = torch.tensor([[[[-1.0, -0.6, 0.2, 1.0]]]])
    ledger = torch.tensor([2], dtype=torch.int8)
    slots = torch.tensor([0])
    widths = torch.tensor([8])
    expected = _fake_quantize_variable_tokens(cache[0], widths)[0]

    TaperKVImpl._lower_tokens(cache, ledger, slots, widths, force=True)

    assert torch.equal(cache[0, 0], expected)
    assert ledger.tolist() == [8]


def test_paired_token_lowering_matches_independent_cpu_roles():
    torch.manual_seed(17)
    key = torch.randn(2, 4, 2, 8)
    value = torch.randn_like(key)
    paired = _kv_cache(key.clone(), value.clone())
    separate = paired.clone()
    slots = torch.tensor([0, 5, 7])
    key_widths = torch.tensor([8, 6, 4])
    value_widths = torch.tensor([4, 3, 2])
    paired_key_ledger = torch.full((8,), 16, dtype=torch.int8)
    paired_value_ledger = paired_key_ledger.clone()
    separate_key_ledger = paired_key_ledger.clone()
    separate_value_ledger = paired_value_ledger.clone()

    TaperKVImpl._lower_kv_tokens(
        paired,
        paired_key_ledger,
        paired_value_ledger,
        slots,
        key_widths,
        value_widths,
    )
    separate_key, separate_value = TaperKVImpl._split_kv_cache(separate)
    TaperKVImpl._lower_tokens(separate_key, separate_key_ledger, slots, key_widths)
    TaperKVImpl._lower_tokens(separate_value, separate_value_ledger, slots, value_widths)

    assert torch.equal(paired, separate)
    assert torch.equal(paired_key_ledger, separate_key_ledger)
    assert torch.equal(paired_value_ledger, separate_value_ledger)


def test_shared_physical_entries_fail_closed():
    with pytest.raises(RuntimeError, match="shared"):
        _assert_unique_indices(torch.tensor([0, 0]), 2, "shared physical page")


def test_dummy_warmup_slots_do_not_mutate_taper_state(monkeypatch):
    key = torch.randn(1, 4, 1, 8)
    kv_cache = _kv_cache(key, key.clone())
    recipe = TaperKVVLLMRecipe(policy_name="taperkv")
    implementation = object.__new__(TaperKVImpl)
    implementation._taper_pending_slots = torch.full((8,), -1, dtype=torch.long)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dummy warm-up mutated TaperKV cache state")

    monkeypatch.setattr(TaperKVImpl, "_reset_pending_precision", forbidden)
    monkeypatch.setattr(TaperKVImpl, "_requantize_cache", forbidden)
    monkeypatch.setattr(TaperKVImpl, "_quantize_pending_tokens", forbidden)
    implementation._process_cache_update(
        kv_cache,
        SimpleNamespace(max_query_len=128),
        object(),
        recipe,
    )

    assert not hasattr(implementation, "_taper_since")
    assert not hasattr(implementation, "_taper_key_bits")
    assert not hasattr(implementation, "_taper_value_bits")


def test_block_regrade_keeps_partial_page_fp16_then_quantizes_when_full():
    torch.manual_seed(2)
    page_size = 4
    key = torch.randn(2, page_size, 2, 8)
    value = torch.randn_like(key)
    kv_cache = _kv_cache(key, value)
    recipe = TaperKVVLLMRecipe(
        policy_name="block_taperkv",
        key_bits=8,
        value_bits=4,
        b_min=4,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        requant_stride=4,
        block_size=page_size,
        residual_fp16=True,
        quant_granularity="block",
    )
    implementation = object.__new__(TaperKVImpl)
    block_table = torch.tensor([[0, 1]])
    partial_before = TaperKVImpl._split_kv_cache(kv_cache)[0][1].clone()

    implementation._regrade_pages(
        *implementation._split_kv_cache(kv_cache),
        block_table,
        torch.tensor([5]),
        kv_cache,
        recipe,
    )

    assert implementation._taper_key_bits[1].item() == 16
    assert implementation._taper_value_bits[1].item() == 16
    assert torch.equal(implementation._split_kv_cache(kv_cache)[0][1], partial_before)

    implementation._regrade_pages(
        *implementation._split_kv_cache(kv_cache),
        block_table,
        torch.tensor([8]),
        kv_cache,
        recipe,
    )

    assert implementation._taper_key_bits[1].item() < 16
    assert implementation._taper_value_bits[1].item() < 16
    assert not torch.equal(implementation._split_kv_cache(kv_cache)[0][1], partial_before)


def test_block_pending_write_quantizes_exactly_when_page_becomes_full():
    torch.manual_seed(3)
    page_size = 4
    key = torch.randn(2, page_size, 1, 8)
    value = torch.randn_like(key)
    kv_cache = _kv_cache(key, value)
    recipe = TaperKVVLLMRecipe(
        policy_name="block_taperkv",
        key_bits=8,
        value_bits=4,
        sink_tokens=0,
        recent_tokens=0,
        requant_stride=16,
        block_size=page_size,
    )
    implementation = object.__new__(TaperKVImpl)

    partial_slot = torch.tensor([page_size])
    implementation._reset_pending_precision(kv_cache, recipe)
    implementation._taper_pending_slots = partial_slot
    implementation._reset_pending_precision(kv_cache, recipe)
    partial_before = TaperKVImpl._split_kv_cache(kv_cache)[0][1].clone()
    implementation._quantize_pending_tokens(kv_cache, recipe, partial_slot)
    assert implementation._taper_key_bits[1].item() == 16
    assert torch.equal(TaperKVImpl._split_kv_cache(kv_cache)[0][1], partial_before)

    final_slot = torch.tensor([2 * page_size - 1])
    implementation._taper_pending_slots = final_slot
    implementation._reset_pending_precision(kv_cache, recipe)
    implementation._quantize_pending_tokens(kv_cache, recipe, final_slot)
    assert implementation._taper_key_bits[1].item() == recipe.key_bits
    assert implementation._taper_value_bits[1].item() == recipe.value_bits
    assert not torch.equal(TaperKVImpl._split_kv_cache(kv_cache)[0][1], partial_before)


def test_block_regrade_rejects_shared_pages_across_requests():
    page_size = 4
    key = torch.randn(1, page_size, 1, 4)
    kv_cache = _kv_cache(key, key.clone())
    recipe = TaperKVVLLMRecipe(
        policy_name="block_taperkv",
        sink_tokens=0,
        recent_tokens=0,
        block_size=page_size,
    )
    implementation = object.__new__(TaperKVImpl)

    with pytest.raises(RuntimeError, match="shared"):
        implementation._regrade_pages(
            *implementation._split_kv_cache(kv_cache),
            torch.tensor([[0], [0]]),
            torch.tensor([4, 4]),
            kv_cache,
            recipe,
        )


def test_block_page_ledgers_match_huggingface_schedule_for_batched_requests():
    page_size = 4
    key = torch.randn(6, page_size, 1, 4)
    kv_cache = _kv_cache(key, key.clone())
    recipe = TaperKVVLLMRecipe(
        policy_name="block_taperkv",
        key_bits=8,
        value_bits=4,
        b_min=4,
        b_min_value=2,
        sink_tokens=1,
        recent_tokens=2,
        key_ladder=(8, 6, 4),
        value_ladder=(4, 3, 2),
        block_size=page_size,
    )
    implementation = object.__new__(TaperKVImpl)
    block_table = torch.tensor([[4, 1, 5], [0, 3, 2]])
    lengths = torch.tensor([10, 7])

    implementation._regrade_pages(
        *implementation._split_kv_cache(kv_cache),
        block_table,
        lengths,
        kv_cache,
        recipe,
    )

    for request, length in enumerate(lengths.tolist()):
        key_expected = block_taper_bits(
            length,
            page_size,
            recipe.sink_tokens,
            recipe.recent_tokens,
            recipe.key_bits,
            recipe.b_min,
            recipe.key_ladder,
        )
        value_expected = block_taper_bits(
            length,
            page_size,
            recipe.sink_tokens,
            recipe.recent_tokens,
            recipe.value_bits,
            recipe.b_min_value,
            recipe.value_ladder,
        )
        for logical_page in range((length + page_size - 1) // page_size):
            physical_page = int(block_table[request, logical_page])
            offset = logical_page * page_size
            assert implementation._taper_key_bits[physical_page] == key_expected[offset]
            assert implementation._taper_value_bits[physical_page] == value_expected[offset]


def test_taper_process_payload_rejects_unknown_or_missing_fields():
    with pytest.raises(ValueError, match="non-canonical"):
        TaperKVVLLMRecipe.from_mapping({"policy_name": "taperkv", "recent_toknes": 8})
    with pytest.raises(ValueError, match="requires policy_name"):
        TaperKVVLLMRecipe.from_mapping({})
