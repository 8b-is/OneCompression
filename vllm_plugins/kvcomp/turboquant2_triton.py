# Copyright 2026 Fujitsu Ltd.
"""Specialized packed 2-bit TurboQuant kernels for vLLM.

Upstream vLLM's TurboQuant data plane currently has explicit 3-bit and 4-bit
packing branches.  These kernels extend the same scalar method to K2/V2:
Hadamard-rotated Lloyd--Max keys with norm correction and per-vector affine
values.  Four codes are packed into every byte.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

TURBOQUANT2_BITS = 2


def turboquant2_layout(head_dim: int) -> dict[str, int]:
    """Return the packed K2/V2 byte layout for one KV head and token."""

    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    if head_dim % 4:
        raise ValueError("2-bit TurboQuant requires head_dim divisible by four")
    code_bytes = head_dim // 4
    key_packed_size = code_bytes + 2  # fp16 key norm
    slot_size = key_packed_size + code_bytes + 4  # fp16 value scale and zero
    return {
        "key_code_bytes": code_bytes,
        "key_norm_offset": code_bytes,
        "key_packed_size": key_packed_size,
        "value_codes_offset": key_packed_size,
        "value_code_bytes": code_bytes,
        "value_scale_offset": key_packed_size + code_bytes,
        "slot_size": slot_size,
        "slot_size_aligned": slot_size + (slot_size % 2),
    }


@triton.jit
def _tq2_store_kernel(
    rotated_key_ptr,
    key_norm_ptr,
    value_ptr,
    midpoint_ptr,
    cache_ptr,
    slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    KEY_PACKED_SIZE: tl.constexpr,
):
    """Quantize and pack one K/V vector per program."""

    pid = tl.program_id(0)
    token = pid // H
    head = pid % H
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return
    page = (slot // PAGE_SIZE).to(tl.int64)
    position = (slot % PAGE_SIZE).to(tl.int64)
    slot_base = (
        page * stride_cache_block
        + position * stride_cache_pos
        + tl.cast(head, tl.int64) * stride_cache_head
    )

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < D
    base = pid * D

    # Lloyd--Max key assignment.  Three threshold comparisons are cheaper
    # than a generic bucketize and compile to a small branch-free expression.
    key_vec = tl.load(rotated_key_ptr + base + dims, mask=dim_mask, other=0.0)
    midpoint0 = tl.load(midpoint_ptr)
    midpoint1 = tl.load(midpoint_ptr + 1)
    midpoint2 = tl.load(midpoint_ptr + 2)
    key_code = (
        (key_vec >= midpoint0).to(tl.int32)
        + (key_vec >= midpoint1).to(tl.int32)
        + (key_vec >= midpoint2).to(tl.int32)
    )
    key_groups = tl.reshape(key_code, [BLOCK_D // 4, 4])
    shifts = tl.arange(0, 4) * 2
    packed_key = tl.sum(key_groups << shifts[None, :], axis=1).to(tl.uint8)
    byte_offsets = tl.arange(0, BLOCK_D // 4)
    byte_mask = byte_offsets < CODE_BYTES
    tl.store(cache_ptr + slot_base + byte_offsets, packed_key, mask=byte_mask)

    norm_f16 = tl.load(key_norm_ptr + pid).to(tl.float16)
    norm_u16 = norm_f16.to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + slot_base + CODE_BYTES, (norm_u16 & 0xFF).to(tl.uint8))
    tl.store(
        cache_ptr + slot_base + CODE_BYTES + 1,
        ((norm_u16 >> 8) & 0xFF).to(tl.uint8),
    )

    value_vec = tl.load(value_ptr + base + dims, mask=dim_mask, other=0.0).to(tl.float32)
    value_min = tl.min(tl.where(dim_mask, value_vec, float("inf")), axis=0)
    value_max = tl.max(tl.where(dim_mask, value_vec, -float("inf")), axis=0)
    value_scale = (value_max - value_min) / 3.0
    value_scale = tl.where(value_scale > 1e-8, value_scale, 1e-8)
    value_code = tl.minimum(
        tl.maximum(((value_vec - value_min) / value_scale + 0.5).to(tl.int32), 0),
        3,
    )
    value_groups = tl.reshape(value_code, [BLOCK_D // 4, 4])
    packed_value = tl.sum(value_groups << shifts[None, :], axis=1).to(tl.uint8)
    tl.store(
        cache_ptr + slot_base + KEY_PACKED_SIZE + byte_offsets,
        packed_value,
        mask=byte_mask,
    )

    scale_offset = KEY_PACKED_SIZE + CODE_BYTES
    scale_u16 = value_scale.to(tl.float16).to(tl.uint16, bitcast=True)
    zero_u16 = value_min.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + slot_base + scale_offset, (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(
        cache_ptr + slot_base + scale_offset + 1,
        ((scale_u16 >> 8) & 0xFF).to(tl.uint8),
    )
    tl.store(cache_ptr + slot_base + scale_offset + 2, (zero_u16 & 0xFF).to(tl.uint8))
    tl.store(
        cache_ptr + slot_base + scale_offset + 3,
        ((zero_u16 >> 8) & 0xFF).to(tl.uint8),
    )


def triton_turboquant2_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation_t: torch.Tensor,
    midpoints: torch.Tensor,
) -> None:
    """Store K2/V2 in one fused packing kernel after the rotation GEMM."""

    tokens, heads, head_dim = key.shape
    layout = turboquant2_layout(head_dim)
    rows = tokens * heads
    key_flat = key.float().reshape(rows, head_dim)
    norms = key_flat.norm(dim=1, keepdim=True)
    rotated = (key_flat / (norms + 1e-8)) @ rotation_t
    value_flat = value.float().reshape(rows, head_dim)
    block_dim = triton.next_power_of_2(head_dim)
    _tq2_store_kernel[(rows,)](
        rotated,
        norms.squeeze(1),
        value_flat,
        midpoints,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=head_dim,
        H=heads,
        PAGE_SIZE=kv_cache.shape[1],
        BLOCK_D=block_dim,
        CODE_BYTES=layout["key_code_bytes"],
        KEY_PACKED_SIZE=layout["key_packed_size"],
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _tq2_decode_stage1(
    rotated_query_ptr,
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    centroids_ptr,
    mid_output_ptr,
    stride_query_batch,
    stride_query_head,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_block_table_batch,
    stride_mid_batch,
    stride_mid_head,
    stride_mid_split,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    KEY_PACKED_SIZE: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Fused unpack, score, online softmax, and V accumulation for K2/V2."""

    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = query_head // KV_GROUP_SIZE
    seq_len = tl.load(seq_lens_ptr + batch)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * split
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    byte_index = dims // 4
    bit_shift = (dims % 4) * 2
    query_base = batch * stride_query_batch + query_head * stride_query_head
    rotated_query = tl.load(rotated_query_ptr + query_base + dims, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    token_offsets = tl.arange(0, BLOCK_KV)
    block_table_base = batch * stride_block_table_batch

    max_score = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start in range(split_start, split_end, BLOCK_KV):
        positions = start + token_offsets
        token_mask = positions < split_end
        page_indices = positions // PAGE_SIZE
        page_offsets = positions % PAGE_SIZE
        page_numbers = tl.load(
            block_table_ptr + block_table_base + page_indices,
            mask=token_mask,
            other=0,
        ).to(tl.int64)
        slot_bases = (
            page_numbers * stride_cache_block
            + page_offsets.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        key_raw = tl.load(
            cache_ptr + slot_bases[:, None] + byte_index[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0,
        ).to(tl.int32)
        key_indices = (key_raw >> bit_shift[None, :]) & 0x3
        key_values = tl.load(
            centroids_ptr + key_indices,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # Match vLLM's *_nc presets: normalize the selected centroid vector
        # before restoring the original key norm.
        centroid_norm_sq = tl.sum(
            tl.where(dim_mask[None, :], key_values * key_values, 0.0), axis=1
        )
        key_values = key_values * (1.0 / tl.sqrt(centroid_norm_sq + 1e-16))[:, None]
        key_dot = tl.sum(
            tl.where(
                dim_mask[None, :],
                rotated_query[None, :] * key_values,
                0.0,
            ),
            axis=1,
        )
        norm_bases = slot_bases + CODE_BYTES
        norm_low = tl.load(cache_ptr + norm_bases, mask=token_mask, other=0).to(tl.uint16)
        norm_high = tl.load(cache_ptr + norm_bases + 1, mask=token_mask, other=0).to(tl.uint16)
        key_norms = (norm_low | (norm_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        scores = tl.where(token_mask, key_norms * key_dot * ATTN_SCALE, -float("inf"))

        next_max = tl.maximum(tl.max(scores, axis=0), max_score)
        previous_scale = tl.exp(max_score - next_max)
        probabilities = tl.exp(scores - next_max)

        value_bases = slot_bases + KEY_PACKED_SIZE
        value_raw = tl.load(
            cache_ptr + value_bases[:, None] + byte_index[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0,
        ).to(tl.int32)
        value_indices = ((value_raw >> bit_shift[None, :]) & 0x3).to(tl.float32)
        scale_bases = value_bases + CODE_BYTES
        scale_low = tl.load(cache_ptr + scale_bases, mask=token_mask, other=0).to(tl.uint16)
        scale_high = tl.load(cache_ptr + scale_bases + 1, mask=token_mask, other=0).to(tl.uint16)
        value_scales = (scale_low | (scale_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zero_low = tl.load(cache_ptr + scale_bases + 2, mask=token_mask, other=0).to(tl.uint16)
        zero_high = tl.load(cache_ptr + scale_bases + 3, mask=token_mask, other=0).to(tl.uint16)
        value_zeros = (zero_low | (zero_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = value_indices * value_scales[:, None] + value_zeros[:, None]

        accumulator = accumulator * previous_scale + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        normalizer = normalizer * previous_scale + tl.sum(probabilities, axis=0)
        max_score = next_max

    output_base = (
        batch * stride_mid_batch + query_head * stride_mid_head + split * stride_mid_split
    )
    safe_normalizer = tl.where(normalizer > 0.0, normalizer, 1.0)
    tl.store(
        mid_output_ptr + output_base + dims,
        accumulator / safe_normalizer,
        mask=dim_mask,
    )
    tl.store(
        mid_output_ptr + output_base + HEAD_DIM,
        max_score + tl.log(safe_normalizer),
    )


def triton_turboquant2_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rotation_t: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    *,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Run the specialized split-KV TurboQuant K2/V2 decode kernel."""

    batch, query_heads, head_dim = query.shape
    kv_heads = kv_cache.shape[2]
    layout = turboquant2_layout(head_dim)
    block_dim = triton.next_power_of_2(head_dim)
    rotated_query = (query.float() @ rotation_t).contiguous()
    splits = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= batch
        and mid_o_buf.shape[1] >= query_heads
        and mid_o_buf.shape[2] >= splits
    ):
        mid_output = mid_o_buf[:batch, :query_heads, :splits, : head_dim + 1]
    else:
        mid_output = torch.empty(
            batch,
            query_heads,
            splits,
            head_dim + 1,
            dtype=torch.float32,
            device=query.device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_output

    _tq2_decode_stage1[(batch, query_heads, splits)](
        rotated_query,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_output,
        rotated_query.stride(0),
        rotated_query.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=kv_cache.shape[1],
        NUM_KV_SPLITS=splits,
        KV_GROUP_SIZE=query_heads // kv_heads,
        CODE_BYTES=layout["key_code_bytes"],
        KEY_PACKED_SIZE=layout["key_packed_size"],
        ATTN_SCALE=scale,
        BLOCK_D=block_dim,
        BLOCK_KV=4,
        num_warps=1,
        num_stages=1,
    )

    if (
        output_buf is not None
        and output_buf.shape[0] >= batch
        and output_buf.shape[1] >= query_heads
    ):
        output = output_buf[:batch, :query_heads, :head_dim]
    else:
        output = torch.empty(batch, query_heads, head_dim, dtype=query.dtype, device=query.device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= batch and lse_buf.shape[1] >= query_heads:
        lse = lse_buf[:batch, :query_heads]
    else:
        lse = torch.empty(batch, query_heads, dtype=torch.float32, device=query.device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    stage2_options = {}
    if "OUTPUT_FP16" in getattr(_fwd_kernel_stage2, "arg_names", ()):
        stage2_options["OUTPUT_FP16"] = 1 if output.dtype == torch.float16 else 0
    _fwd_kernel_stage2[(batch, query_heads)](
        mid_output,
        output,
        lse,
        seq_lens,
        mid_output.stride(0),
        mid_output.stride(1),
        mid_output.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=splits,
        BLOCK_DV=block_dim,
        Lv=head_dim,
        num_warps=4,
        num_stages=2,
        **stage2_options,
    )
    return output


@triton.jit
def _tq2_dequant_kernel(
    cache_ptr,
    block_table_ptr,
    centroids_ptr,
    key_output_ptr,
    value_output_ptr,
    stride_key_batch,
    stride_key_head,
    stride_key_position,
    stride_value_batch,
    stride_value_head,
    stride_value_position,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_block_table_batch,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    KEY_PACKED_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Materialize rotated keys and affine values for continuation prefill."""

    position = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // NUM_KV_HEADS
    head = batch_head % NUM_KV_HEADS
    page_index = position // PAGE_SIZE
    page_offset = position % PAGE_SIZE
    page_number = tl.load(block_table_ptr + batch * stride_block_table_batch + page_index).to(
        tl.int64
    )
    slot_base = (
        page_number * stride_cache_block
        + tl.cast(page_offset, tl.int64) * stride_cache_pos
        + tl.cast(head, tl.int64) * stride_cache_head
    )
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    byte_index = dims // 4
    bit_shift = (dims % 4) * 2

    key_raw = tl.load(cache_ptr + slot_base + byte_index, mask=dim_mask, other=0).to(tl.int32)
    key_indices = (key_raw >> bit_shift) & 0x3
    key_values = tl.load(centroids_ptr + key_indices, mask=dim_mask, other=0.0).to(tl.float32)
    centroid_norm_sq = tl.sum(tl.where(dim_mask, key_values * key_values, 0.0), axis=0)
    key_values = key_values * (1.0 / tl.sqrt(centroid_norm_sq + 1e-16))
    norm_low = tl.load(cache_ptr + slot_base + CODE_BYTES).to(tl.uint16)
    norm_high = tl.load(cache_ptr + slot_base + CODE_BYTES + 1).to(tl.uint16)
    key_norm = (norm_low | (norm_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    key_base = batch * stride_key_batch + head * stride_key_head + position * stride_key_position
    tl.store(
        key_output_ptr + key_base + dims,
        (key_values * key_norm).to(tl.float16),
        mask=dim_mask,
    )

    value_base = slot_base + KEY_PACKED_SIZE
    value_raw = tl.load(cache_ptr + value_base + byte_index, mask=dim_mask, other=0).to(tl.int32)
    value_indices = ((value_raw >> bit_shift) & 0x3).to(tl.float32)
    scale_offset = value_base + CODE_BYTES
    scale_low = tl.load(cache_ptr + scale_offset).to(tl.uint16)
    scale_high = tl.load(cache_ptr + scale_offset + 1).to(tl.uint16)
    value_scale = (scale_low | (scale_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    zero_low = tl.load(cache_ptr + scale_offset + 2).to(tl.uint16)
    zero_high = tl.load(cache_ptr + scale_offset + 3).to(tl.uint16)
    value_zero = (zero_low | (zero_high << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    value_output_base = (
        batch * stride_value_batch + head * stride_value_head + position * stride_value_position
    )
    tl.store(
        value_output_ptr + value_output_base + dims,
        (value_indices * value_scale + value_zero).to(tl.float16),
        mask=dim_mask,
    )


def triton_turboquant2_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    centroids: torch.Tensor,
    key_output: torch.Tensor,
    value_output: torch.Tensor,
) -> None:
    """Dequantize all allocated positions into caller-owned fp16 buffers."""

    _, kv_heads, positions, head_dim = key_output.shape
    layout = turboquant2_layout(head_dim)
    _tq2_dequant_kernel[(positions, key_output.shape[0] * kv_heads)](
        kv_cache,
        block_table,
        centroids,
        key_output,
        value_output,
        key_output.stride(0),
        key_output.stride(1),
        key_output.stride(2),
        value_output.stride(0),
        value_output.stride(1),
        value_output.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=head_dim,
        PAGE_SIZE=kv_cache.shape[1],
        NUM_KV_HEADS=kv_heads,
        CODE_BYTES=layout["key_code_bytes"],
        KEY_PACKED_SIZE=layout["key_packed_size"],
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )


__all__ = [
    "TURBOQUANT2_BITS",
    "triton_turboquant2_decode_attention",
    "triton_turboquant2_dequant_kv",
    "triton_turboquant2_store",
    "turboquant2_layout",
]
