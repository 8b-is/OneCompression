# Copyright 2026 Fujitsu Ltd.
"""Triton kernels for the vLLM HayateQuant native key block-VQ path.

The cache layout intentionally matches vLLM's native TurboQuant layout.  This
lets vLLM keep its existing cache-capacity accounting and paged-cache
allocation while replacing the key quantizer:

* TurboQuant stores one scalar Lloyd-Max index per key coordinate.
* HayateQuant stores one vector-codebook index per small coordinate block.

The shipped native path supports the production HayateQuant presets that have
the same byte budget as upstream TurboQuant: 3-bit and 4-bit keys with resolved
``block_size == 2``.  Values remain vLLM-compatible uniform quantized values;
that keeps the kernel practical inside upstream paged attention while preserving
HayateQuant's main block-VQ gain on attention logits.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2
from vllm.v1.attention.ops.triton_turboquant_store import _store_quantized_value


@triton.jit
def _hq_store_key_block_vq(
    # Post-rotation inputs
    Y_ptr,  # [NH, D] float32, rotated normalized keys
    Norms_ptr,  # [NH] float32, original key L2 norms
    Scale_scratch_ptr,  # [NH] float32, optional accumulated code-vector norm
    Value_ptr,  # [NH, D] float32, raw values
    Codebook_ptr,  # [K, B] float32
    # Cache and indexing
    KV_cache_ptr,
    Slot_mapping_ptr,
    # Cache strides
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    # Dimensions
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Layout
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    # Hayate key block-VQ params
    HAYATE_BLOCK_SIZE: tl.constexpr,
    HAYATE_CODE_BITS: tl.constexpr,
    NUM_CODEWORDS: tl.constexpr,
    NUM_KEY_BLOCKS: tl.constexpr,
    GROUP_CODES: tl.constexpr,
    PRECOMPUTE_KEY_SCALE: tl.constexpr = 0,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Quantize key blocks to Hayate code indices and store values uniformly."""
    vec_pid = tl.program_id(0)
    group_pid = tl.program_id(1)
    token_idx = vec_pid // H
    head_idx = vec_pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    block_ids = group_pid * GROUP_CODES + tl.arange(0, GROUP_CODES)
    group_mask = block_ids < NUM_KEY_BLOCKS

    # Production native path: block_size=2, code bits are 6 (3-bit rate) or
    # 8 (4-bit rate).  Keeping this specialised avoids a giant score matrix.
    y0 = tl.load(
        Y_ptr + vec_pid * D + block_ids * HAYATE_BLOCK_SIZE,
        mask=group_mask,
        other=0.0,
    )
    y1 = tl.load(
        Y_ptr + vec_pid * D + block_ids * HAYATE_BLOCK_SIZE + 1,
        mask=group_mask,
        other=0.0,
    )

    best_score = tl.full([GROUP_CODES], -float("inf"), dtype=tl.float32)
    best_idx = tl.zeros([GROUP_CODES], dtype=tl.int32)
    best_norm = tl.zeros([GROUP_CODES], dtype=tl.float32)
    for cid in range(NUM_CODEWORDS):
        c0 = tl.load(Codebook_ptr + cid * HAYATE_BLOCK_SIZE).to(tl.float32)
        c1 = tl.load(Codebook_ptr + cid * HAYATE_BLOCK_SIZE + 1).to(tl.float32)
        cw_norm = c0 * c0 + c1 * c1
        score = y0 * c0 + y1 * c1 - 0.5 * cw_norm
        take = score > best_score
        best_score = tl.where(take, score, best_score)
        best_idx = tl.where(take, cid, best_idx)
        best_norm = tl.where(take, cw_norm, best_norm)

    if HAYATE_CODE_BITS == 8:
        tl.store(
            KV_cache_ptr + slot_base + block_ids,
            best_idx.to(tl.uint8),
            mask=group_mask,
        )
    else:
        shifts = tl.arange(0, GROUP_CODES) * HAYATE_CODE_BITS
        packed = tl.sum((best_idx & 0x3F) << shifts, axis=0)
        byte_base = group_pid * 3
        tl.store(KV_cache_ptr + slot_base + byte_base, (packed & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + byte_base + 1,
            ((packed >> 8) & 0xFF).to(tl.uint8),
        )
        tl.store(
            KV_cache_ptr + slot_base + byte_base + 2,
            ((packed >> 16) & 0xFF).to(tl.uint8),
        )

    if PRECOMPUTE_KEY_SCALE:
        group_norm_sq = tl.sum(tl.where(group_mask, best_norm, 0.0), axis=0)
        tl.atomic_add(Scale_scratch_ptr + vec_pid, group_norm_sq, sem="relaxed")

    # Norm + values are per vector, so one group owns them.
    if group_pid == 0:
        vn_f16 = tl.load(Norms_ptr + vec_pid).to(tl.float16)
        vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + MSE_BYTES, (vn_u16 & 0xFF).to(tl.uint8))
        tl.store(
            KV_cache_ptr + slot_base + MSE_BYTES + 1,
            ((vn_u16 >> 8) & 0xFF).to(tl.uint8),
        )

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        _store_quantized_value(
            Value_ptr,
            KV_cache_ptr,
            vec_pid * D,
            slot_base,
            d_offs,
            d_mask,
            D=D,
            KPS=KPS,
            VQB=VQB,
            VAL_DATA_BYTES=VAL_DATA_BYTES,
            BLOCK_D=BLOCK_D,
            BLOCK_VAL=BLOCK_VAL,
            BLOCK_GRP=BLOCK_GRP,
        )


@triton.jit
def _hq_write_key_scales(
    Norms_ptr,
    Scale_scratch_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MSE_BYTES: tl.constexpr,
):
    vec_pid = tl.program_id(0)
    token_idx = vec_pid // H
    head_idx = vec_pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + tl.cast(head_idx, tl.int64) * stride_cache_head
    )

    norm = tl.load(Norms_ptr + vec_pid).to(tl.float32)
    chat_norm_sq = tl.load(Scale_scratch_ptr + vec_pid).to(tl.float32)
    key_scale = norm * (1.0 / tl.sqrt(chat_norm_sq + 1e-16))
    scale_f16 = key_scale.to(tl.float16)
    scale_u16 = scale_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + MSE_BYTES, (scale_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + slot_base + MSE_BYTES + 1,
        ((scale_u16 >> 8) & 0xFF).to(tl.uint8),
    )


@triton.jit
def _hq_decode_stage1(
    Q_rot_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Codebook_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HAYATE_BLOCK_SIZE: tl.constexpr,
    HAYATE_CODE_BITS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 1,
    PRECOMPUTE_KEY_SCALE: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(
        tl.float32
    )

    key_block_offs = tl.arange(0, BLOCK_D // HAYATE_BLOCK_SIZE)
    key_block_mask = key_block_offs < (HEAD_DIM // HAYATE_BLOCK_SIZE)
    q0 = tl.load(
        Q_rot_ptr + q_base + key_block_offs * HAYATE_BLOCK_SIZE,
        mask=key_block_mask,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        Q_rot_ptr + q_base + key_block_offs * HAYATE_BLOCK_SIZE + 1,
        mask=key_block_mask,
        other=0.0,
    ).to(tl.float32)
    if HAYATE_CODE_BITS == 8:
        key_code_byte_idx = key_block_offs
    else:
        key_code_group_idx = key_block_offs // 4
        key_code_shift = (key_block_offs - key_code_group_idx * 4) * 6
        key_code_byte_idx = key_code_group_idx * 3

    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        if HAYATE_CODE_BITS == 8:
            key_code_raw = tl.load(
                KV_cache_ptr + slot_bases[:, None] + key_code_byte_idx[None, :],
                mask=kv_mask[:, None] & key_block_mask[None, :],
                other=0,
            ).to(tl.int32)
            key_code_idx = key_code_raw & 0xFF
        else:
            raw0 = tl.load(
                KV_cache_ptr + slot_bases[:, None] + key_code_byte_idx[None, :],
                mask=kv_mask[:, None] & key_block_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw1 = tl.load(
                KV_cache_ptr + slot_bases[:, None] + key_code_byte_idx[None, :] + 1,
                mask=kv_mask[:, None] & key_block_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw2 = tl.load(
                KV_cache_ptr + slot_bases[:, None] + key_code_byte_idx[None, :] + 2,
                mask=kv_mask[:, None] & key_block_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw24 = raw0 | (raw1 << 8) | (raw2 << 16)
            key_code_idx = (raw24 >> key_code_shift[None, :]) & 0x3F

        c0 = tl.load(
            Codebook_ptr + key_code_idx * HAYATE_BLOCK_SIZE,
            mask=kv_mask[:, None] & key_block_mask[None, :],
            other=0.0,
        )
        c1 = tl.load(
            Codebook_ptr + key_code_idx * HAYATE_BLOCK_SIZE + 1,
            mask=kv_mask[:, None] & key_block_mask[None, :],
            other=0.0,
        )

        block_norm_sq = c0 * c0 + c1 * c1
        term1 = tl.sum(
            tl.where(key_block_mask[None, :], q0[None, :] * c0 + q1[None, :] * c1, 0.0),
            axis=1,
        )
        if NORM_CORRECTION and not PRECOMPUTE_KEY_SCALE:
            c_norm_sq = tl.sum(
                tl.where(key_block_mask[None, :], block_norm_sq, 0.0),
                axis=1,
            )
            term1 = term1 * (1.0 / tl.sqrt(c_norm_sq + 1e-16))

        norm_bases = slot_bases + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
            tl.uint16
        )
        key_scales = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        scores = key_scales * term1 * ATTN_SCALE
        scores = tl.where(kv_mask, scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        val_bases = slot_bases + KPS
        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(
                tl.float32
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(
                tl.float32
            )
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_raw = tl.load(
                KV_cache_ptr + val_bases[:, None] + vb_idx[None, :],
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)
            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(
                tl.float32
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(
                tl.float32
            )
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


@triton.jit
def _hq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Codebook_ptr,
    K_out_ptr,
    V_out_ptr,
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAYATE_BLOCK_SIZE: tl.constexpr,
    HAYATE_CODE_BITS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 1,
    PRECOMPUTE_KEY_SCALE: tl.constexpr = 0,
):
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    block_idx = d_offs // HAYATE_BLOCK_SIZE
    elem_idx = d_offs - block_idx * HAYATE_BLOCK_SIZE

    if HAYATE_CODE_BITS == 8:
        code_raw = tl.load(
            KV_cache_ptr + slot_base + block_idx,
            mask=d_mask,
            other=0,
        ).to(tl.int32)
        code_idx = code_raw & 0xFF
    else:
        code_group_idx = block_idx // 4
        code_shift = (block_idx - code_group_idx * 4) * 6
        code_byte_idx = code_group_idx * 3
        raw0 = tl.load(KV_cache_ptr + slot_base + code_byte_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        raw1 = tl.load(
            KV_cache_ptr + slot_base + code_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw2 = tl.load(
            KV_cache_ptr + slot_base + code_byte_idx + 2, mask=d_mask, other=0
        ).to(tl.int32)
        raw24 = raw0 | (raw1 << 8) | (raw2 << 16)
        code_idx = (raw24 >> code_shift) & 0x3F

    k_mse = tl.load(
        Codebook_ptr + code_idx * HAYATE_BLOCK_SIZE + elem_idx,
        mask=d_mask,
        other=0.0,
    )
    if NORM_CORRECTION and not PRECOMPUTE_KEY_SCALE:
        c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
        k_mse = k_mse * (1.0 / tl.sqrt(c_norm_sq + 1e-16))

    n_lo = tl.load(KV_cache_ptr + slot_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(KV_cache_ptr + slot_base + MSE_BYTES + 1).to(tl.uint16)
    key_scale = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, (key_scale * k_mse).to(tl.float16), mask=d_mask)

    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)
    else:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        raw0 = tl.load(KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        v_idx = (((raw0 | (raw1 << 8)) >> val_bit_shift) & 0x7).to(tl.float32)

    sc_base = val_base + VAL_DATA_BYTES
    sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
    sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
    v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
    zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
    v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, (v_idx * v_scale + v_zero).to(tl.float16), mask=d_mask)


def triton_hayatequant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    hayate_block_size: int,
    key_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
) -> None:
    """Store K/V into a TurboQuant-layout cache using Hayate key block-VQ."""
    if hayate_block_size != 2:
        raise ValueError("native vLLM HayateQuant currently supports block_size=2")
    if key_bits not in (3, 4):
        raise ValueError("native vLLM HayateQuant currently supports 3/4-bit keys")

    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    block_d = triton.next_power_of_2(D)
    mse_bytes = math.ceil(D * key_bits / 8)
    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    block_val = triton.next_power_of_2(val_data_bytes)
    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1
    n_key_blocks = D // hayate_block_size
    code_bits = key_bits * hayate_block_size
    group_codes = 1 if code_bits == 8 else 4
    n_groups = math.ceil(n_key_blocks / group_codes)

    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    # HayateQuant codebooks are fit for unit-RMS rotated vectors.  The decode
    # kernel applies NC scaling by normalizing the gathered code vector and
    # multiplying by the original L2 norm, matching the HF cache path's key
    # scale mode.  Unit-L2 normalization here would shrink the assignment input
    # by sqrt(D) and collapses the code usage.
    rms = k_flat.pow(2).mean(dim=1, keepdim=True).sqrt()
    y = (k_flat @ rotation) / (rms + 1e-8)
    v_flat = value.float().reshape(NH, D)
    norms_flat = norms.squeeze(1).contiguous()
    precompute_key_scale = _precompute_key_scale(key_bits)
    scale_scratch = (
        torch.zeros(NH, device=key.device, dtype=torch.float32)
        if precompute_key_scale
        else norms_flat
    )

    _hq_store_key_block_vq[(NH, n_groups)](
        y.contiguous(),
        norms_flat,
        scale_scratch,
        v_flat.contiguous(),
        codebook.contiguous(),
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=block_d,
        MSE_BYTES=mse_bytes,
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=block_val,
        HAYATE_BLOCK_SIZE=hayate_block_size,
        HAYATE_CODE_BITS=code_bits,
        NUM_CODEWORDS=codebook.shape[0],
        NUM_KEY_BLOCKS=n_key_blocks,
        GROUP_CODES=group_codes,
        PRECOMPUTE_KEY_SCALE=1 if precompute_key_scale else 0,
        BLOCK_GRP=block_grp,
        num_warps=4,
        num_stages=1,
    )
    if precompute_key_scale:
        _hq_write_key_scales[(NH,)](
            norms_flat,
            scale_scratch,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=kv_cache.stride(0),
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
            H=H,
            BLOCK_SIZE=block_size,
            MSE_BYTES=mse_bytes,
            num_warps=1,
            num_stages=1,
        )


_layout_cache: dict[tuple[int, int, int, int, int], dict[str, int]] = {}


def _get_layout(
    dim: int,
    key_bits: int,
    value_quant_bits: int,
    key_packed_size: int,
    hayate_block_size: int,
) -> dict[str, int]:
    cache_key = (dim, key_bits, value_quant_bits, key_packed_size, hayate_block_size)
    cfg = _layout_cache.get(cache_key)
    if cfg is None:
        cfg = {
            "mse_bytes": math.ceil(dim * key_bits / 8),
            "val_data_bytes": math.ceil(dim * value_quant_bits / 8),
            "block_d": triton.next_power_of_2(dim),
            "code_bits": key_bits * hayate_block_size,
        }
        _layout_cache[cache_key] = cfg
    return cfg


def _precompute_key_scale(key_bits: int) -> bool:
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_PRECOMPUTE_SCALE", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    return key_bits == 3


def _decode_block_kv(key_bits: int) -> int:
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_BLOCK_KV", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode == "auto":
        return 4
    block_kv = int(mode)
    if block_kv not in {1, 2, 4, 8, 16}:
        raise ValueError("KVCOMP_VLLM_HAYATE_BLOCK_KV must be one of 1,2,4,8,16")
    return block_kv


def _decode_num_warps(key_bits: int) -> int:
    mode = os.environ.get("KVCOMP_VLLM_HAYATE_NUM_WARPS", "auto")
    mode = mode.strip().lower().replace("-", "_")
    if mode == "auto":
        return 1
    num_warps = int(mode)
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("KVCOMP_VLLM_HAYATE_NUM_WARPS must be one of 1,2,4,8")
    return num_warps


def triton_hayatequant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    scale: float,
    key_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    hayate_block_size: int,
    norm_correction: bool = True,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Decode attention from a Hayate key-block-VQ cache."""
    if hayate_block_size != 2:
        raise ValueError("native vLLM HayateQuant currently supports block_size=2")
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    cfg = _get_layout(D, key_bits, value_quant_bits, key_packed_size, hayate_block_size)
    precompute_key_scale = _precompute_key_scale(key_bits)
    block_kv = _decode_block_kv(key_bits)
    num_warps = _decode_num_warps(key_bits)

    q_rot = (query.float() @ rotation).contiguous()
    num_splits = max_num_kv_splits
    if mid_o_buf is not None and mid_o_buf.shape[0] >= B and mid_o_buf.shape[2] >= num_splits:
        mid_o = mid_o_buf[:B, :Hq, :num_splits, :]
    else:
        mid_o = torch.empty(B, Hq, num_splits, D + 1, dtype=torch.float32, device=query.device)
        if buf_holder is not None:
            buf_holder._hq_mid_o_buf = mid_o

    _hq_decode_stage1[(B, Hq, num_splits)](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        codebook.contiguous(),
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=num_splits,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=cfg["block_d"],
        BLOCK_KV=block_kv,
        HAYATE_BLOCK_SIZE=hayate_block_size,
        HAYATE_CODE_BITS=cfg["code_bits"],
        NORM_CORRECTION=1 if norm_correction else 0,
        PRECOMPUTE_KEY_SCALE=1 if precompute_key_scale else 0,
        num_warps=num_warps,
        num_stages=1,
    )

    out_dtype = query.dtype
    if output_buf is not None and output_buf.shape[0] >= B and output_buf.dtype == out_dtype:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=query.device)
        if buf_holder is not None:
            buf_holder._hq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=query.device)
        if buf_holder is not None:
            buf_holder._hq_lse_buf = lse

    _fwd_kernel_stage2[(B, Hq)](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_splits,
        BLOCK_DV=cfg["block_d"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )
    return output


def triton_hayatequant_full_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    codebook: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    key_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    hayate_block_size: int,
    norm_correction: bool = True,
) -> None:
    """Bulk dequantize cached K/V for large continuation prefill."""
    B, Hk, alloc_len, D = k_out.shape
    block_size = kv_cache.shape[1]
    cfg = _get_layout(D, key_bits, value_quant_bits, key_packed_size, hayate_block_size)
    precompute_key_scale = _precompute_key_scale(key_bits)
    _hq_full_dequant_kv[(alloc_len, B * Hk)](
        kv_cache,
        block_table,
        codebook.contiguous(),
        k_out,
        v_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=Hk,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        BLOCK_D=cfg["block_d"],
        HAYATE_BLOCK_SIZE=hayate_block_size,
        HAYATE_CODE_BITS=cfg["code_bits"],
        NORM_CORRECTION=1 if norm_correction else 0,
        PRECOMPUTE_KEY_SCALE=1 if precompute_key_scale else 0,
        num_warps=4,
    )
