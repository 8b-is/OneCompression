from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")

from vllm_plugins.kvcomp.taper_backend import (  # noqa: E402
    _UPSTREAM_KV_AXIS,
    TaperKVImpl,
    _fake_quantize_variable_tokens,
)
from vllm_plugins.kvcomp.taper_triton import (  # noqa: E402
    triton_taper_quantize_and_store_kv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _kv_cache(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return torch.stack((key, value), dim=_UPSTREAM_KV_AXIS)


@pytest.mark.parametrize("widths", [(2, 4, 8), (8, 6, 4)])
def test_fused_token_quantizer_matches_reference_on_strided_cache(widths):
    torch.manual_seed(11)
    key = torch.randn(3, 4, 2, 32, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    # Construct the interleaved layout explicitly. vLLM 0.20 uses K/V axis 0,
    # whose unbound tensors are contiguous, while vLLM 0.24 uses axis 1. This
    # regression specifically targets write-through on a strided cache view and
    # must therefore not inherit the installed vLLM version's preferred axis.
    kv_cache = torch.stack((key.clone(), value.clone()), dim=1)
    key_cache = kv_cache[:, 0]
    assert not key_cache.is_contiguous()
    source = key_cache.clone()
    slots = torch.tensor([0, 5, 11], device="cuda", dtype=torch.long)
    requested = torch.tensor(widths, device="cuda", dtype=torch.long)
    ledger = torch.full((12,), 16, device="cuda", dtype=torch.int8)
    pages = torch.div(slots, 4, rounding_mode="floor")
    positions = slots.remainder(4)
    expected = source.clone()
    expected[pages, positions] = _fake_quantize_variable_tokens(
        source[pages, positions],
        requested,
    )

    TaperKVImpl._lower_tokens(key_cache, ledger, slots, requested)
    torch.cuda.synchronize()
    actual_key, actual_value = kv_cache[:, 0], kv_cache[:, 1]

    torch.testing.assert_close(actual_key, expected, rtol=0, atol=2e-3)
    assert torch.equal(actual_value, value)
    assert ledger.cpu().tolist() == [
        widths[0],
        16,
        16,
        16,
        16,
        widths[1],
        16,
        16,
        16,
        16,
        16,
        widths[2],
    ]


def test_fused_token_quantizer_is_monotonic_without_requantization_drift():
    torch.manual_seed(12)
    cache = torch.randn(2, 4, 2, 32, device="cuda", dtype=torch.float16)
    ledger = torch.full((8,), 16, device="cuda", dtype=torch.int8)
    slots = torch.tensor([1, 6], device="cuda")
    widths = torch.tensor([4, 2], device="cuda")

    TaperKVImpl._lower_tokens(cache, ledger, slots, widths)
    torch.cuda.synchronize()
    first = cache.clone()
    TaperKVImpl._lower_tokens(cache, ledger, slots, torch.tensor([8, 8], device="cuda"))
    torch.cuda.synchronize()

    assert torch.equal(cache, first)
    assert ledger.cpu().tolist() == [16, 4, 16, 16, 16, 16, 2, 16]


def test_fused_token_quantizer_forces_new_data_in_reused_slot():
    torch.manual_seed(13)
    cache = torch.randn(1, 2, 2, 32, device="cuda", dtype=torch.float16)
    source = cache.clone()
    ledger = torch.tensor([2, 16], device="cuda", dtype=torch.int8)
    slots = torch.tensor([0], device="cuda")
    widths = torch.tensor([8], device="cuda")
    expected = _fake_quantize_variable_tokens(source[:, :1].reshape(1, 2, 32), widths)[0]

    TaperKVImpl._lower_tokens(cache, ledger, slots, widths, force=True)
    torch.cuda.synchronize()

    torch.testing.assert_close(cache[0, 0], expected, rtol=0, atol=2e-3)
    assert ledger.cpu().tolist() == [8, 16]


def test_fused_kv_token_quantizer_matches_independent_gpu_roles():
    torch.manual_seed(18)
    key = torch.randn(3, 4, 2, 32, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    paired = _kv_cache(key.clone(), value.clone())
    separate = paired.clone()
    slots = torch.tensor([0, 5, 11], device="cuda", dtype=torch.long)
    key_widths = torch.tensor([8, 6, 4], device="cuda", dtype=torch.long)
    value_widths = torch.tensor([4, 3, 2], device="cuda", dtype=torch.long)
    paired_key_ledger = torch.full((12,), 16, device="cuda", dtype=torch.int8)
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
    torch.cuda.synchronize()

    torch.testing.assert_close(paired, separate, rtol=0, atol=2e-3)
    assert torch.equal(paired_key_ledger, separate_key_ledger)
    assert torch.equal(paired_value_ledger, separate_value_ledger)


def test_fused_quantize_and_store_matches_model_dtype_reference():
    torch.manual_seed(19)
    incoming_key = torch.randn(5, 2, 32, device="cuda", dtype=torch.float16)
    incoming_value = torch.randn_like(incoming_key)
    key_cache = torch.randn(3, 4, 2, 32, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    kv_cache = _kv_cache(key_cache.clone(), value_cache.clone())
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    slots = torch.tensor([0, 5, 11], device="cuda", dtype=torch.long)
    pages = torch.div(slots, 4, rounding_mode="floor")
    positions = slots.remainder(4)
    expected_key[pages, positions] = _fake_quantize_variable_tokens(
        incoming_key[: slots.numel()],
        torch.full((slots.numel(),), 8, device="cuda", dtype=torch.long),
    )
    expected_value[pages, positions] = _fake_quantize_variable_tokens(
        incoming_value[: slots.numel()],
        torch.full((slots.numel(),), 4, device="cuda", dtype=torch.long),
    )
    key_ledger = torch.full((12,), 2, device="cuda", dtype=torch.int8)
    value_ledger = torch.full((12,), 2, device="cuda", dtype=torch.int8)

    triton_taper_quantize_and_store_kv(
        incoming_key,
        incoming_value,
        kv_cache,
        key_ledger,
        value_ledger,
        slots,
        key_bits=8,
        value_bits=4,
        kv_axis=_UPSTREAM_KV_AXIS,
    )
    torch.cuda.synchronize()
    actual_key, actual_value = TaperKVImpl._split_kv_cache(kv_cache)

    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=2e-3)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=2e-3)
    assert key_ledger[slots].cpu().tolist() == [8, 8, 8]
    assert value_ledger[slots].cpu().tolist() == [4, 4, 4]


@pytest.mark.parametrize("kv_axis", [0, 1])
def test_fused_quantize_and_store_masks_pad_slots_for_both_cache_layouts(kv_axis):
    torch.manual_seed(20)
    incoming_key = torch.randn(3, 2, 32, device="cuda", dtype=torch.float16)
    incoming_value = torch.randn_like(incoming_key)
    key_cache = torch.randn(2, 4, 2, 32, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    kv_cache = torch.stack((key_cache.clone(), value_cache.clone()), dim=kv_axis)
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    slots = torch.tensor([0, -1, 5], device="cuda", dtype=torch.long)
    live_sources = torch.tensor([0, 2], device="cuda")
    live_slots = slots[live_sources]
    pages = torch.div(live_slots, 4, rounding_mode="floor")
    positions = live_slots.remainder(4)
    expected_key[pages, positions] = _fake_quantize_variable_tokens(
        incoming_key[live_sources],
        torch.full((2,), 8, device="cuda", dtype=torch.long),
    )
    expected_value[pages, positions] = _fake_quantize_variable_tokens(
        incoming_value[live_sources],
        torch.full((2,), 4, device="cuda", dtype=torch.long),
    )
    key_ledger = torch.full((8,), 16, device="cuda", dtype=torch.int8)
    value_ledger = key_ledger.clone()

    triton_taper_quantize_and_store_kv(
        incoming_key,
        incoming_value,
        kv_cache,
        key_ledger,
        value_ledger,
        slots,
        key_bits=8,
        value_bits=4,
        kv_axis=kv_axis,
    )
    torch.cuda.synchronize()
    actual_key, actual_value = kv_cache.unbind(kv_axis)

    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=2e-3)
    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=2e-3)
    assert key_ledger.cpu().tolist() == [8, 16, 16, 16, 16, 8, 16, 16]
    assert value_ledger.cpu().tolist() == [4, 16, 16, 16, 16, 4, 16, 16]
