# Copyright 2026 Fujitsu Ltd.
"""Stride-aware fused fake-quantization kernels for the TaperKV cache."""

from __future__ import annotations

import torch
from triton.language.extra import libdevice
from vllm.triton_utils import tl, triton


@triton.jit
def _taper_quantize_and_store_kv(
    Key_ptr,
    Value_ptr,
    KV_cache_ptr,
    Key_ledger_ptr,
    Value_ledger_ptr,
    Slots_ptr,
    N_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_page: tl.constexpr,
    stride_role: tl.constexpr,
    stride_position: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    KEY_BITS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Quantize incoming K/V and scatter them directly into the paged cache."""

    token = tl.program_id(0)
    slot = tl.load(Slots_ptr + token, mask=token < N_TOKENS, other=-1).to(tl.int64)
    valid_slot = (token < N_TOKENS) & (slot >= 0)
    heads = tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, BLOCK_D)[None, :]
    element_mask = valid_slot & (heads < N_HEADS) & (dims < HEAD_DIM)
    key_source = tl.load(
        Key_ptr + token * stride_kt + heads * stride_kh + dims * stride_kd,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    value_source = tl.load(
        Value_ptr + token * stride_vt + heads * stride_vh + dims * stride_vd,
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    key_minimum = tl.min(
        tl.where(element_mask, key_source, float("inf")),
        axis=1,
    )[:, None]
    key_maximum = tl.max(
        tl.where(element_mask, key_source, -float("inf")),
        axis=1,
    )[:, None]
    value_minimum = tl.min(
        tl.where(element_mask, value_source, float("inf")),
        axis=1,
    )[:, None]
    value_maximum = tl.max(
        tl.where(element_mask, value_source, -float("inf")),
        axis=1,
    )[:, None]
    key_qmax = float((1 << KEY_BITS) - 1)
    value_qmax = float((1 << VALUE_BITS) - 1)
    key_scale = tl.maximum((key_maximum - key_minimum) / key_qmax, 1e-8)
    value_scale = tl.maximum((value_maximum - value_minimum) / value_qmax, 1e-8)
    key_codes = libdevice.rint((key_source - key_minimum) / key_scale)
    value_codes = libdevice.rint((value_source - value_minimum) / value_scale)
    key_quantized = tl.maximum(0.0, tl.minimum(key_codes, key_qmax)) * key_scale + key_minimum
    value_quantized = (
        tl.maximum(0.0, tl.minimum(value_codes, value_qmax)) * value_scale + value_minimum
    )

    page = slot // PAGE_SIZE
    position = slot - page * PAGE_SIZE
    output = (
        KV_cache_ptr
        + page * stride_page
        + position * stride_position
        + heads * stride_head
        + dims * stride_dim
    )
    tl.store(output, key_quantized, mask=element_mask)
    tl.store(output + stride_role, value_quantized, mask=element_mask)
    tl.store(Key_ledger_ptr + slot, KEY_BITS, mask=valid_slot)
    tl.store(Value_ledger_ptr + slot, VALUE_BITS, mask=valid_slot)


def triton_taper_quantize_and_store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    key_ledger: torch.Tensor,
    value_ledger: torch.Tensor,
    slots: torch.Tensor,
    *,
    key_bits: int,
    value_bits: int,
    kv_axis: int,
) -> None:
    """Fused token-local affine quantization and vLLM paged-cache update."""

    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key/value must have matching [tokens, heads, dim] shapes")
    if kv_cache.ndim != 5 or kv_axis not in {0, 1} or kv_cache.shape[kv_axis] != 2:
        raise ValueError("kv_cache must have one two-entry K/V axis in position 0 or 1")
    tensors = (key, value, kv_cache, key_ledger, value_ledger, slots)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused TaperKV cache update requires CUDA tensors")
    if slots.ndim != 1 or slots.numel() > key.shape[0]:
        raise ValueError("slot mapping must fit the incoming key/value token dimension")
    data_axes = [axis for axis in range(5) if axis != kv_axis]
    page_axis, position_axis, head_axis, dim_axis = data_axes
    pages = int(kv_cache.shape[page_axis])
    page_size = int(kv_cache.shape[position_axis])
    heads = int(kv_cache.shape[head_axis])
    dim = int(kv_cache.shape[dim_axis])
    if tuple(key.shape[1:]) != (heads, dim):
        raise ValueError("incoming key/value geometry does not match the paged cache")
    capacity = pages * page_size
    if key_ledger.shape != (capacity,) or value_ledger.shape != (capacity,):
        raise ValueError("K/V ledgers do not match the token capacity")
    if slots.numel() == 0:
        return
    block_h = triton.next_power_of_2(heads)
    block_d = triton.next_power_of_2(dim)
    if block_h * block_d > 4096:
        raise ValueError("TaperKV Triton token tile is too large")
    _taper_quantize_and_store_kv[(slots.numel(),)](
        key,
        value,
        kv_cache,
        key_ledger,
        value_ledger,
        slots,
        N_TOKENS=slots.numel(),
        PAGE_SIZE=page_size,
        N_HEADS=heads,
        HEAD_DIM=dim,
        stride_kt=key.stride(0),
        stride_kh=key.stride(1),
        stride_kd=key.stride(2),
        stride_vt=value.stride(0),
        stride_vh=value.stride(1),
        stride_vd=value.stride(2),
        stride_page=kv_cache.stride(page_axis),
        stride_role=kv_cache.stride(kv_axis),
        stride_position=kv_cache.stride(position_axis),
        stride_head=kv_cache.stride(head_axis),
        stride_dim=kv_cache.stride(dim_axis),
        KEY_BITS=int(key_bits),
        VALUE_BITS=int(value_bits),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=8 if block_h * block_d >= 1024 else 4,
        num_stages=1,
    )


@triton.jit
def _taper_fake_quantize_tokens(
    Cache_ptr,
    Ledger_ptr,
    Slots_ptr,
    Requested_ptr,
    N_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_page: tl.constexpr,
    stride_position: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    FORCE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply one affine scale per token/head and update its width ledger."""

    token = tl.program_id(0)
    valid_token = token < N_TOKENS
    slot = tl.load(Slots_ptr + token, mask=valid_token, other=-1).to(tl.int64)
    requested = tl.load(Requested_ptr + token, mask=valid_token, other=16).to(tl.int32)
    current = tl.load(Ledger_ptr + slot, mask=valid_token & (slot >= 0), other=16).to(tl.int32)
    if FORCE:
        effective = requested
        changed = valid_token & (slot >= 0)
    else:
        effective = tl.minimum(current, requested)
        changed = valid_token & (slot >= 0) & (effective < current)

    page = slot // PAGE_SIZE
    position = slot - page * PAGE_SIZE
    heads = tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, BLOCK_D)[None, :]
    element_mask = changed & (heads < N_HEADS) & (dims < HEAD_DIM)
    pointers = (
        Cache_ptr
        + page * stride_page
        + position * stride_position
        + heads * stride_head
        + dims * stride_dim
    )
    source = tl.load(pointers, mask=element_mask, other=0.0).to(tl.float32)
    minimum = tl.min(
        tl.where(element_mask, source, float("inf")),
        axis=1,
    )[:, None]
    maximum = tl.max(
        tl.where(element_mask, source, -float("inf")),
        axis=1,
    )[:, None]
    qmax = tl.exp2(effective.to(tl.float32)) - 1.0
    scale = tl.maximum((maximum - minimum) / qmax, 1e-8)
    codes = libdevice.rint((source - minimum) / scale)
    codes = tl.maximum(0.0, tl.minimum(codes, qmax))
    quantized = codes * scale + minimum
    tl.store(pointers, quantized, mask=element_mask)
    tl.store(Ledger_ptr + slot, effective, mask=changed)


def triton_taper_fake_quantize_tokens(
    cache: torch.Tensor,
    ledger: torch.Tensor,
    slots: torch.Tensor,
    requested: torch.Tensor,
    *,
    force: bool = False,
) -> None:
    """Quantize selected token rows without flattening a strided K/V view."""

    if cache.ndim != 4:
        raise ValueError("cache must be [pages, page_size, heads, dim]")
    if not cache.is_cuda or not ledger.is_cuda or not slots.is_cuda or not requested.is_cuda:
        raise ValueError("TaperKV Triton quantization requires CUDA tensors")
    if slots.ndim != 1 or requested.ndim != 1 or slots.shape != requested.shape:
        raise ValueError("slots and requested widths must be same-length vectors")
    if ledger.ndim != 1 or ledger.numel() != cache.shape[0] * cache.shape[1]:
        raise ValueError("ledger does not match the token capacity")
    if slots.numel() == 0:
        return
    page_size, heads, dim = map(int, cache.shape[1:])
    block_h = triton.next_power_of_2(heads)
    block_d = triton.next_power_of_2(dim)
    if block_h * block_d > 4096:
        raise ValueError("TaperKV Triton token tile is too large")
    _taper_fake_quantize_tokens[(slots.numel(),)](
        cache,
        ledger,
        slots,
        requested,
        N_TOKENS=slots.numel(),
        PAGE_SIZE=page_size,
        N_HEADS=heads,
        HEAD_DIM=dim,
        stride_page=cache.stride(0),
        stride_position=cache.stride(1),
        stride_head=cache.stride(2),
        stride_dim=cache.stride(3),
        FORCE=force,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=8 if block_h * block_d >= 1024 else 4,
        num_stages=1,
    )


@triton.jit
def _taper_fake_quantize_kv_tokens(
    KV_cache_ptr,
    Key_ledger_ptr,
    Value_ledger_ptr,
    Slots_ptr,
    Key_requested_ptr,
    Value_requested_ptr,
    N_TOKENS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    stride_page: tl.constexpr,
    stride_role: tl.constexpr,
    stride_position: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    FORCE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Quantize K/V in one launch while retaining independent affine scales."""

    item = tl.program_id(0)
    token = item // 2
    role = item - token * 2
    valid_token = token < N_TOKENS
    slot = tl.load(Slots_ptr + token, mask=valid_token, other=-1).to(tl.int64)
    key_requested = tl.load(Key_requested_ptr + token, mask=valid_token, other=16).to(tl.int32)
    value_requested = tl.load(Value_requested_ptr + token, mask=valid_token, other=16).to(tl.int32)
    requested = tl.where(role == 0, key_requested, value_requested)
    valid_slot = valid_token & (slot >= 0)
    key_current = tl.load(Key_ledger_ptr + slot, mask=valid_slot, other=16).to(tl.int32)
    value_current = tl.load(Value_ledger_ptr + slot, mask=valid_slot, other=16).to(tl.int32)
    current = tl.where(role == 0, key_current, value_current)
    if FORCE:
        effective = requested
        changed = valid_slot
    else:
        effective = tl.minimum(current, requested)
        changed = valid_slot & (effective < current)

    page = slot // PAGE_SIZE
    position = slot - page * PAGE_SIZE
    heads = tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, BLOCK_D)[None, :]
    element_mask = changed & (heads < N_HEADS) & (dims < HEAD_DIM)
    pointers = (
        KV_cache_ptr
        + page * stride_page
        + role * stride_role
        + position * stride_position
        + heads * stride_head
        + dims * stride_dim
    )
    source = tl.load(pointers, mask=element_mask, other=0.0).to(tl.float32)
    minimum = tl.min(
        tl.where(element_mask, source, float("inf")),
        axis=1,
    )[:, None]
    maximum = tl.max(
        tl.where(element_mask, source, -float("inf")),
        axis=1,
    )[:, None]
    qmax = tl.exp2(effective.to(tl.float32)) - 1.0
    scale = tl.maximum((maximum - minimum) / qmax, 1e-8)
    codes = libdevice.rint((source - minimum) / scale)
    codes = tl.maximum(0.0, tl.minimum(codes, qmax))
    quantized = codes * scale + minimum
    tl.store(pointers, quantized, mask=element_mask)
    tl.store(Key_ledger_ptr + slot, effective, mask=changed & (role == 0))
    tl.store(Value_ledger_ptr + slot, effective, mask=changed & (role == 1))


def triton_taper_fake_quantize_kv_tokens(
    kv_cache: torch.Tensor,
    key_ledger: torch.Tensor,
    value_ledger: torch.Tensor,
    slots: torch.Tensor,
    key_requested: torch.Tensor,
    value_requested: torch.Tensor,
    *,
    kv_axis: int,
    force: bool = False,
) -> None:
    """Quantize strided K/V cache rows with one GPU kernel launch."""

    if kv_cache.ndim != 5 or kv_axis not in {0, 1} or kv_cache.shape[kv_axis] != 2:
        raise ValueError("kv_cache must have one two-entry K/V axis in position 0 or 1")
    tensors = (kv_cache, key_ledger, value_ledger, slots, key_requested, value_requested)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused TaperKV K/V quantization requires CUDA tensors")
    if (
        slots.ndim != 1
        or key_requested.ndim != 1
        or value_requested.ndim != 1
        or slots.shape != key_requested.shape
        or slots.shape != value_requested.shape
    ):
        raise ValueError("slots and K/V requested widths must be same-length vectors")
    data_axes = [axis for axis in range(5) if axis != kv_axis]
    page_axis, position_axis, head_axis, dim_axis = data_axes
    pages = int(kv_cache.shape[page_axis])
    page_size = int(kv_cache.shape[position_axis])
    heads = int(kv_cache.shape[head_axis])
    dim = int(kv_cache.shape[dim_axis])
    capacity = pages * page_size
    if (
        key_ledger.ndim != 1
        or value_ledger.ndim != 1
        or key_ledger.numel() != capacity
        or value_ledger.numel() != capacity
    ):
        raise ValueError("K/V ledgers do not match the token capacity")
    if slots.numel() == 0:
        return
    block_h = triton.next_power_of_2(heads)
    block_d = triton.next_power_of_2(dim)
    if block_h * block_d > 4096:
        raise ValueError("TaperKV Triton token tile is too large")
    _taper_fake_quantize_kv_tokens[(slots.numel() * 2,)](
        kv_cache,
        key_ledger,
        value_ledger,
        slots,
        key_requested,
        value_requested,
        N_TOKENS=slots.numel(),
        PAGE_SIZE=page_size,
        N_HEADS=heads,
        HEAD_DIM=dim,
        stride_page=kv_cache.stride(page_axis),
        stride_role=kv_cache.stride(kv_axis),
        stride_position=kv_cache.stride(position_axis),
        stride_head=kv_cache.stride(head_axis),
        stride_dim=kv_cache.stride(dim_axis),
        FORCE=force,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=8 if block_h * block_d >= 1024 else 4,
        num_stages=1,
    )


__all__ = [
    "triton_taper_fake_quantize_kv_tokens",
    "triton_taper_fake_quantize_tokens",
    "triton_taper_quantize_and_store_kv",
]
