# Copyright 2026 Fujitsu Ltd.
"""Faithful packed fixed-recipe HayateQuant store and attention kernels.

Unlike :mod:`vllm_plugins.kvcomp.hayate_triton` (an explicitly labelled
key-only hybrid), this module quantizes *both* roles with HayateQuant's
data-free rotated block codebooks.  Keys use norm correction (NC); values use
one LS fit, re-assignment, and a final LS refit.  Formal HQ uses the
all-codeword exact assignment.  Fixed grid tables remain development
implementation candidates and may replace that search only after passing the
pinned all-rate quality and payload gates.  Each role stores one signed
fp16 scale and no zero point.

For the fixed b=2 recipes, native Triton evaluates every codeword with the
same fp16 tensor-core score as the audited public codec and fuses the argmax
without materialising an ``N x K`` score matrix.  Other block sizes keep the
public tensor primitive.  Code indices, scales, and packed bytes remain under
the public-payload parity gate.  Packed-cache decode and attention are native
Triton.
"""

from __future__ import annotations

from typing import Any

import torch

from onecomp.kvcomp.codec._scale import fit_scale
from onecomp.kvcomp.codec.hayate.vq import (
    lookup_codewords,
    quantize_and_lookup,
    quantize_exact_pre,
    quantize_via_lut,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

from vllm_plugins.kvcomp.hayate_full_config import FullHQNativeRecipe


def _assignment_lut(
    recipe: FullHQNativeRecipe,
    role: str,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve the public codec's immutable cached table for one HQ role."""

    if recipe.assignment_mode == "exact":
        return None
    if recipe.assignment_mode != "lut":
        raise ValueError(f"unsupported full HQ assignment {recipe.assignment_mode!r}")
    from onecomp.kvcomp.codec.hayate.codec import (  # pylint: disable=import-outside-toplevel
        _block_lut_code_bits_on,
    )

    code_bits = recipe.key_code_bits_per_block if role == "k" else recipe.value_code_bits_per_block
    block_size = recipe.key_block_size if role == "k" else recipe.value_block_size
    return _block_lut_code_bits_on(
        code_bits,
        block_size,
        recipe.dist,
        head_dim,
        recipe.codebook_init,
        recipe.lut_bins,
        recipe.lut_candidates,
        str(device),
    )


@triton.jit
def _full_hq_exact_b2_tensorcore(
    Z_ptr,
    CB_ptr,
    CSQ_ptr,
    Codes_ptr,
    N_BLOCKS: tl.constexpr,
    NUM_CODEWORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """All-codeword b=2 score and argmax without a materialized score matrix."""

    blocks = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    bmask = blocks < N_BLOCKS
    inner = tl.arange(0, 16)
    z0 = tl.load(Z_ptr + blocks * 2, mask=bmask, other=0.0).to(tl.float16)
    z1 = tl.load(Z_ptr + blocks * 2 + 1, mask=bmask, other=0.0).to(tl.float16)
    lhs = tl.where(
        inner[None, :] == 0,
        z0[:, None],
        tl.where(inner[None, :] == 1, z1[:, None], 0.0),
    ).to(tl.float16)
    best = tl.full([BLOCK_M], -float("inf"), tl.float32)
    winner = tl.full([BLOCK_M], NUM_CODEWORDS, tl.int32)
    for start in range(0, NUM_CODEWORDS, BLOCK_K):
        code = start + tl.arange(0, BLOCK_K)
        kmask = code < NUM_CODEWORDS
        rhs = tl.load(
            CB_ptr + code[None, :] * 2 + inner[:, None],
            mask=kmask[None, :] & (inner[:, None] < 2),
            other=0.0,
        ).to(tl.float16)
        dot = tl.dot(lhs, rhs, out_dtype=tl.float32).to(tl.float16)
        csq = tl.load(CSQ_ptr + code, mask=kmask, other=0.0).to(tl.float16)
        correction = (0.5 * csq.to(tl.float32)).to(tl.float16)
        score = (dot.to(tl.float32) - correction[None, :].to(tl.float32)).to(tl.float16)
        scoref = tl.where(bmask[:, None] & kmask[None, :], score, -float("inf")).to(tl.float32)
        tile_best = tl.max(scoref, axis=1)
        tile_winner = tl.min(
            tl.where(scoref == tile_best[:, None], code[None, :], NUM_CODEWORDS),
            axis=1,
        )
        replace = (tile_best > best) | ((tile_best == best) & (tile_winner < winner))
        best = tl.where(replace, tile_best, best)
        winner = tl.where(replace, tile_winner, winner)
    tl.store(Codes_ptr + blocks, winner.to(tl.int64), mask=bmask)


def _quantize_exact_b2_tensorcore(
    z: torch.Tensor,
    scoring_codebook: torch.Tensor,
    scoring_csq: torch.Tensor,
) -> torch.Tensor:
    """Canonical public-GEMM parity with fused tiled all-codeword argmax."""

    flat = z.reshape(-1, 2).contiguous()
    num_codewords = int(scoring_codebook.shape[0])
    if num_codewords < 1 or num_codewords > 512 or num_codewords & (num_codewords - 1):
        raise ValueError("native b=2 tensor-core assignment requires power-of-two K <= 512")
    codes = torch.empty(flat.shape[0], device=flat.device, dtype=torch.int64)
    block_m = 32
    block_k = max(16, num_codewords)
    _full_hq_exact_b2_tensorcore[(triton.cdiv(flat.shape[0], block_m),)](
        flat,
        scoring_codebook,
        scoring_csq,
        codes,
        N_BLOCKS=flat.shape[0],
        NUM_CODEWORDS=scoring_codebook.shape[0],
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return codes.reshape(z.shape[:-1])


def _encode_role(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int,
    scale_mode: str,
    *,
    lut: torch.Tensor | None,
    lut_bins: int,
    lut_limit: float,
    scoring_codebook: torch.Tensor | None,
    scoring_csq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return public-codec-parity integer codes and signed fp16 scales."""

    if x.shape[-1] % block_size:
        raise ValueError(f"block_size={block_size} must divide D={x.shape[-1]}")
    xr = x.float() @ rotation
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    n_blocks = x.shape[-1] // block_size

    def lookup(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = vector.reshape(*vector.shape[:-1], n_blocks, block_size)
        if lut is None:
            if scoring_codebook is None or scoring_csq is None:
                codes, chat_blocks = quantize_and_lookup(blocks, codebook)
            else:
                score_blocks = (
                    blocks.half() if blocks.is_cuda and blocks.dtype == torch.float32 else blocks
                )
                if block_size == 2 and score_blocks.is_cuda and scoring_codebook.shape[0] <= 512:
                    codes = _quantize_exact_b2_tensorcore(
                        score_blocks,
                        scoring_codebook,
                        scoring_csq,
                    )
                else:
                    codes = quantize_exact_pre(
                        score_blocks,
                        scoring_codebook,
                        scoring_csq,
                    )
                chat_blocks = lookup_codewords(codebook, codes)
        else:
            codes = quantize_via_lut(
                blocks,
                lut,
                lut_bins,
                codebook,
                lim=lut_limit,
                scoring_cb=scoring_codebook,
                scoring_csq=scoring_csq,
            )
            chat_blocks = lookup_codewords(codebook, codes)
        return codes, chat_blocks.reshape(*vector.shape[:-1], x.shape[-1])

    codes, chat = lookup(xr / rms)

    def requantize(vector: torch.Tensor):
        codes2, chat2 = lookup(vector)
        return chat2, codes2

    scale, new_codes = fit_scale(
        xr,
        chat,
        scale_mode,
        rms,
        requantize_fn=requantize if scale_mode == "ls_iter" else None,
    )
    if new_codes is not None:
        codes = new_codes
    return (
        codes.reshape(-1, n_blocks).to(torch.int32).contiguous(),
        scale.reshape(-1).to(torch.float16).contiguous(),
    )


@triton.jit
def _full_hq_pack_store(
    K_codes_ptr,
    K_scale_ptr,
    V_codes_ptr,
    V_scale_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_NUM_BLOCKS: tl.constexpr,
    V_NUM_BLOCKS: tl.constexpr,
    K_CODE_BITS: tl.constexpr,
    V_CODE_BITS: tl.constexpr,
    K_CODE_BYTES: tl.constexpr,
    V_CODE_BYTES: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
    BLOCK_K_BYTES: tl.constexpr,
    BLOCK_V_BYTES: tl.constexpr,
):
    """Bit-pack two fixed-width code streams and scatter one cache slot."""

    vec = tl.program_id(0)
    token = vec // H
    head = vec % H
    slot = tl.load(Slot_mapping_ptr + token)
    if slot < 0:
        return
    page = (slot // PAGE_SIZE).to(tl.int64)
    pos = (slot % PAGE_SIZE).to(tl.int64)
    base = (
        page * stride_cache_block
        + pos * stride_cache_pos
        + tl.cast(head, tl.int64) * stride_cache_head
    )

    # Construct the packed streams directly from their fixed-width code words.
    # For code widths >= 7, an eight-bit byte can cross at most one code-word
    # boundary.  Loading those two adjacent words replaces the previous eight
    # one-bit gathers per output byte.  The fixed HQ rate-3/rate-4 b=2 recipes
    # are partly/fully in this fast path (HQ-3 K7; HQ-4 K9/V7).  Keep the
    # bitwise fallback for narrower recipes: a byte can cross three or more
    # code words for widths <= 6, and a two-word shortcut would silently
    # corrupt the packed cache.
    kb = tl.arange(0, BLOCK_K_BYTES)
    kmask = kb < K_CODE_BYTES
    if K_CODE_BITS >= 7:
        kstream_bit = kb * 8
        kcode_idx = kstream_bit // K_CODE_BITS
        kcode_shift = kstream_bit - kcode_idx * K_CODE_BITS
        kcode0 = tl.load(
            K_codes_ptr + vec * K_NUM_BLOCKS + kcode_idx,
            mask=kmask & (kcode_idx < K_NUM_BLOCKS),
            other=0,
        ).to(tl.int32)
        kcode1 = tl.load(
            K_codes_ptr + vec * K_NUM_BLOCKS + kcode_idx + 1,
            mask=kmask & ((kcode_idx + 1) < K_NUM_BLOCKS),
            other=0,
        ).to(tl.int32)
        kout = (kcode0 >> kcode_shift) | (kcode1 << (K_CODE_BITS - kcode_shift))
    else:
        kout = tl.zeros([BLOCK_K_BYTES], tl.int32)
        for bit in range(8):
            stream_bit = kb * 8 + bit
            code_idx = stream_bit // K_CODE_BITS
            code_bit = stream_bit - code_idx * K_CODE_BITS
            code = tl.load(
                K_codes_ptr + vec * K_NUM_BLOCKS + code_idx,
                mask=kmask & (code_idx < K_NUM_BLOCKS),
                other=0,
            ).to(tl.int32)
            kout |= ((code >> code_bit) & 1) << bit
    tl.store(KV_cache_ptr + base + kb, kout.to(tl.uint8), mask=kmask)

    vb = tl.arange(0, BLOCK_V_BYTES)
    vmask = vb < V_CODE_BYTES
    if V_CODE_BITS >= 7:
        vstream_bit = vb * 8
        vcode_idx = vstream_bit // V_CODE_BITS
        vcode_shift = vstream_bit - vcode_idx * V_CODE_BITS
        vcode0 = tl.load(
            V_codes_ptr + vec * V_NUM_BLOCKS + vcode_idx,
            mask=vmask & (vcode_idx < V_NUM_BLOCKS),
            other=0,
        ).to(tl.int32)
        vcode1 = tl.load(
            V_codes_ptr + vec * V_NUM_BLOCKS + vcode_idx + 1,
            mask=vmask & ((vcode_idx + 1) < V_NUM_BLOCKS),
            other=0,
        ).to(tl.int32)
        vout = (vcode0 >> vcode_shift) | (vcode1 << (V_CODE_BITS - vcode_shift))
    else:
        vout = tl.zeros([BLOCK_V_BYTES], tl.int32)
        for bit in range(8):
            stream_bit = vb * 8 + bit
            code_idx = stream_bit // V_CODE_BITS
            code_bit = stream_bit - code_idx * V_CODE_BITS
            code = tl.load(
                V_codes_ptr + vec * V_NUM_BLOCKS + code_idx,
                mask=vmask & (code_idx < V_NUM_BLOCKS),
                other=0,
            ).to(tl.int32)
            vout |= ((code >> code_bit) & 1) << bit
    tl.store(KV_cache_ptr + base + V_CODES_OFFSET + vb, vout.to(tl.uint8), mask=vmask)

    ks = tl.load(K_scale_ptr + vec).to(tl.float16).to(tl.uint16, bitcast=True)
    vs = tl.load(V_scale_ptr + vec).to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + base + K_SCALE_OFFSET, (ks & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + base + K_SCALE_OFFSET + 1, ((ks >> 8) & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + base + V_SCALE_OFFSET, (vs & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + base + V_SCALE_OFFSET + 1, ((vs >> 8) & 0xFF).to(tl.uint8))


def _pack_encoded_hq(
    key_codes: torch.Tensor,
    key_scales: torch.Tensor,
    value_codes: torch.Tensor,
    value_scales: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    recipe: FullHQNativeRecipe,
    head_dim: int,
) -> None:
    """Pack pre-encoded HQ payloads; separated for stage-level profiling."""

    n = int(slot_mapping.numel())
    if n < 1 or key_codes.shape[0] % n:
        raise ValueError("encoded HQ rows must be a positive multiple of token slots")
    h = key_codes.shape[0] // n
    layout = recipe.layout(head_dim)
    nh = n * h
    _full_hq_pack_store[(nh,)](
        key_codes,
        key_scales,
        value_codes,
        value_scales,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        H=h,
        PAGE_SIZE=kv_cache.shape[1],
        K_NUM_BLOCKS=head_dim // recipe.key_block_size,
        V_NUM_BLOCKS=head_dim // recipe.value_block_size,
        K_CODE_BITS=recipe.key_code_bits_per_block,
        V_CODE_BITS=recipe.value_code_bits_per_block,
        K_CODE_BYTES=layout["key_code_bytes"],
        V_CODE_BYTES=layout["value_code_bytes"],
        K_SCALE_OFFSET=layout["key_scale_offset"],
        V_CODES_OFFSET=layout["value_codes_offset"],
        V_SCALE_OFFSET=layout["value_scale_offset"],
        BLOCK_K_BYTES=triton.next_power_of_2(layout["key_code_bytes"]),
        BLOCK_V_BYTES=triton.next_power_of_2(layout["value_code_bytes"]),
        num_warps=4,
        num_stages=1,
    )


def triton_full_hayatequant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    recipe: FullHQNativeRecipe,
    key_lut: torch.Tensor | None = None,
    value_lut: torch.Tensor | None = None,
    key_scoring_codebook: torch.Tensor | None = None,
    key_scoring_csq: torch.Tensor | None = None,
    value_scoring_codebook: torch.Tensor | None = None,
    value_scoring_csq: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Encode and pack faithful fixed-recipe HQ K/V into a paged byte cache.

    The returned code/scale tensors are test/provenance aids.  vLLM callers may
    ignore them; the cache owns the only persistent per-token representation.
    """

    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must have matching [N,H,D] shapes")
    n, h, d = key.shape
    recipe.validate_head_dim(d)
    layout = recipe.layout(d)
    if kv_cache.shape[-1] != layout["slot_size_aligned"]:
        raise ValueError(
            f"HQ cache slot is {kv_cache.shape[-1]} bytes, expected "
            f"{layout['slot_size_aligned']} for D={d}"
        )
    if recipe.assignment_mode == "lut":
        if key_lut is None:
            key_lut = _assignment_lut(recipe, "k", d, key.device)
        if value_lut is None:
            value_lut = _assignment_lut(recipe, "v", d, value.device)
    elif key_lut is not None or value_lut is not None:
        raise ValueError("exact full-HQ assignment must not receive LUT tensors")
    if recipe.assignment_mode == "exact":
        if key_scoring_codebook is None:
            key_scoring_codebook = key_codebook.half() if key.is_cuda else key_codebook
        if value_scoring_codebook is None:
            value_scoring_codebook = value_codebook.half() if value.is_cuda else value_codebook
        if key_scoring_csq is None:
            key_scoring_csq = key_scoring_codebook.pow(2).sum(dim=1)
        if value_scoring_csq is None:
            value_scoring_csq = value_scoring_codebook.pow(2).sum(dim=1)
    key_codes, key_scales = _encode_role(
        key,
        key_rotation,
        key_codebook,
        recipe.key_block_size,
        "nc",
        lut=key_lut,
        lut_bins=recipe.lut_bins,
        lut_limit=recipe.lut_limit,
        scoring_codebook=key_scoring_codebook,
        scoring_csq=key_scoring_csq,
    )
    value_codes, value_scales = _encode_role(
        value,
        value_rotation,
        value_codebook,
        recipe.value_block_size,
        "ls_iter",
        lut=value_lut,
        lut_bins=recipe.lut_bins,
        lut_limit=recipe.lut_limit,
        scoring_codebook=value_scoring_codebook,
        scoring_csq=value_scoring_csq,
    )
    _pack_encoded_hq(
        key_codes,
        key_scales,
        value_codes,
        value_scales,
        kv_cache,
        slot_mapping,
        recipe,
        d,
    )
    return {
        "key_codes": key_codes,
        "key_scales": key_scales,
        "value_codes": value_codes,
        "value_scales": value_scales,
    }


@triton.jit
def _full_hq_decode_stage1_b2(
    Q_rot_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    K_codebook_ptr,
    V_codebook_ptr,
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
    PAGE_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    K_CODE_BITS: tl.constexpr,
    V_CODE_BITS: tl.constexpr,
    K_CODE_BYTES: tl.constexpr,
    V_CODE_BYTES: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_NB: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Formal fixed-HQ b=2 decode without duplicate per-coordinate loads."""

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

    blocks = tl.arange(0, BLOCK_NB)
    bmask = blocks < (HEAD_DIM // 2)
    qbase = bid * stride_qb + hid * stride_qh
    q0 = tl.load(Q_rot_ptr + qbase + blocks * 2, mask=bmask, other=0.0).to(tl.float32)
    q1 = tl.load(Q_rot_ptr + qbase + blocks * 2 + 1, mask=bmask, other=0.0).to(tl.float32)
    kbit = blocks * K_CODE_BITS
    kbyte = kbit // 8
    kshift = kbit - kbyte * 8
    vbit = blocks * V_CODE_BITS
    vbyte = vbit // 8
    vshift = vbit - vbyte * 8
    krange = tl.arange(0, BLOCK_KV)
    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], tl.float32)
    btbase = bid * stride_bt_b

    for start in range(split_start, split_end, BLOCK_KV):
        pos = start + krange
        pmask = pos < split_end
        pidx = pos // PAGE_SIZE
        poff = pos % PAGE_SIZE
        pages = tl.load(Block_table_ptr + btbase + pidx, mask=pmask, other=0).to(tl.int64)
        bases = (
            pages * stride_cache_block
            + poff.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        kr0 = tl.load(
            KV_cache_ptr + bases[:, None] + kbyte[None, :],
            mask=pmask[:, None] & bmask[None, :] & (kbyte[None, :] < K_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        kr1 = tl.load(
            KV_cache_ptr + bases[:, None] + kbyte[None, :] + 1,
            mask=pmask[:, None] & bmask[None, :] & ((kbyte[None, :] + 1) < K_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        kcode = ((kr0 | (kr1 << 8)) >> kshift[None, :]) & ((1 << K_CODE_BITS) - 1)
        kc0 = tl.load(
            K_codebook_ptr + kcode * 2,
            mask=pmask[:, None] & bmask[None, :],
            other=0.0,
        ).to(tl.float32)
        kc1 = tl.load(
            K_codebook_ptr + kcode * 2 + 1,
            mask=pmask[:, None] & bmask[None, :],
            other=0.0,
        ).to(tl.float32)
        ks0 = tl.load(KV_cache_ptr + bases + K_SCALE_OFFSET, mask=pmask, other=0).to(tl.uint16)
        ks1 = tl.load(KV_cache_ptr + bases + K_SCALE_OFFSET + 1, mask=pmask, other=0).to(tl.uint16)
        kscale = (ks0 | (ks1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        dot = tl.sum(
            tl.where(bmask[None, :], q0[None, :] * kc0 + q1[None, :] * kc1, 0.0),
            axis=1,
        )
        scores = tl.where(pmask, dot * kscale * ATTN_SCALE, -float("inf"))

        vr0 = tl.load(
            KV_cache_ptr + bases[:, None] + V_CODES_OFFSET + vbyte[None, :],
            mask=pmask[:, None] & bmask[None, :] & (vbyte[None, :] < V_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        vr1 = tl.load(
            KV_cache_ptr + bases[:, None] + V_CODES_OFFSET + vbyte[None, :] + 1,
            mask=pmask[:, None] & bmask[None, :] & ((vbyte[None, :] + 1) < V_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        vcode = ((vr0 | (vr1 << 8)) >> vshift[None, :]) & ((1 << V_CODE_BITS) - 1)
        vc0 = tl.load(
            V_codebook_ptr + vcode * 2,
            mask=pmask[:, None] & bmask[None, :],
            other=0.0,
        ).to(tl.float32)
        vc1 = tl.load(
            V_codebook_ptr + vcode * 2 + 1,
            mask=pmask[:, None] & bmask[None, :],
            other=0.0,
        ).to(tl.float32)
        vs0 = tl.load(KV_cache_ptr + bases + V_SCALE_OFFSET, mask=pmask, other=0).to(tl.uint16)
        vs1 = tl.load(KV_cache_ptr + bases + V_SCALE_OFFSET + 1, mask=pmask, other=0).to(tl.uint16)
        vscale = (vs0 | (vs1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = tl.interleave(vc0, vc1) * vscale[:, None]

        new_m = tl.maximum(tl.max(scores, 0), m_prev)
        rescale = tl.exp(m_prev - new_m)
        p = tl.exp(scores - new_m)
        acc = acc * rescale + tl.sum(p[:, None] * values, axis=0)
        l_prev = l_prev * rescale + tl.sum(p, axis=0)
        m_prev = new_m

    obase = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + obase + d, acc / safe_l, mask=dmask)
    tl.store(Mid_o_ptr + obase + HEAD_DIM, m_prev + tl.log(safe_l))


@triton.jit
def _full_hq_decode_stage1(
    Q_rot_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    K_codebook_ptr,
    V_codebook_ptr,
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
    PAGE_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    K_CODE_BITS: tl.constexpr,
    V_CODE_BITS: tl.constexpr,
    K_CODE_BYTES: tl.constexpr,
    V_CODE_BYTES: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
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

    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    krange = tl.arange(0, BLOCK_KV)
    qbase = bid * stride_qb + hid * stride_qh
    qrot = tl.load(Q_rot_ptr + qbase + d, mask=dmask, other=0.0).to(tl.float32)

    kbidx = d // K_BLOCK_SIZE
    kelem = d - kbidx * K_BLOCK_SIZE
    kbit = kbidx * K_CODE_BITS
    kbyte = kbit // 8
    kshift = kbit - kbyte * 8
    vbidx = d // V_BLOCK_SIZE
    velem = d - vbidx * V_BLOCK_SIZE
    vbit = vbidx * V_CODE_BITS
    vbyte = vbit // 8
    vshift = vbit - vbyte * 8

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], tl.float32)
    btbase = bid * stride_bt_b
    for start in range(split_start, split_end, BLOCK_KV):
        pos = start + krange
        pmask = pos < split_end
        pidx = pos // PAGE_SIZE
        poff = pos % PAGE_SIZE
        pages = tl.load(Block_table_ptr + btbase + pidx, mask=pmask, other=0).to(tl.int64)
        bases = (
            pages * stride_cache_block
            + poff.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        kraw0 = tl.load(
            KV_cache_ptr + bases[:, None] + kbyte[None, :],
            mask=pmask[:, None] & dmask[None, :] & (kbyte[None, :] < K_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        kraw1 = tl.load(
            KV_cache_ptr + bases[:, None] + kbyte[None, :] + 1,
            mask=pmask[:, None] & dmask[None, :] & ((kbyte[None, :] + 1) < K_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        kraw2 = tl.load(
            KV_cache_ptr + bases[:, None] + kbyte[None, :] + 2,
            mask=pmask[:, None] & dmask[None, :] & ((kbyte[None, :] + 2) < K_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        kcode = ((kraw0 | (kraw1 << 8) | (kraw2 << 16)) >> kshift[None, :]) & (
            (1 << K_CODE_BITS) - 1
        )
        kval = tl.load(
            K_codebook_ptr + kcode * K_BLOCK_SIZE + kelem[None, :],
            mask=pmask[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        kslo = tl.load(KV_cache_ptr + bases + K_SCALE_OFFSET, mask=pmask, other=0).to(tl.uint16)
        kshi = tl.load(KV_cache_ptr + bases + K_SCALE_OFFSET + 1, mask=pmask, other=0).to(
            tl.uint16
        )
        kscale = (kslo | (kshi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        scores = tl.sum(qrot[None, :] * kval, axis=1) * kscale * ATTN_SCALE
        scores = tl.where(pmask, scores, -float("inf"))

        vraw0 = tl.load(
            KV_cache_ptr + bases[:, None] + V_CODES_OFFSET + vbyte[None, :],
            mask=pmask[:, None] & dmask[None, :] & (vbyte[None, :] < V_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        vraw1 = tl.load(
            KV_cache_ptr + bases[:, None] + V_CODES_OFFSET + vbyte[None, :] + 1,
            mask=pmask[:, None] & dmask[None, :] & ((vbyte[None, :] + 1) < V_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        vraw2 = tl.load(
            KV_cache_ptr + bases[:, None] + V_CODES_OFFSET + vbyte[None, :] + 2,
            mask=pmask[:, None] & dmask[None, :] & ((vbyte[None, :] + 2) < V_CODE_BYTES),
            other=0,
        ).to(tl.int32)
        vcode = ((vraw0 | (vraw1 << 8) | (vraw2 << 16)) >> vshift[None, :]) & (
            (1 << V_CODE_BITS) - 1
        )
        vval = tl.load(
            V_codebook_ptr + vcode * V_BLOCK_SIZE + velem[None, :],
            mask=pmask[:, None] & dmask[None, :],
            other=0.0,
        ).to(tl.float32)
        vslo = tl.load(KV_cache_ptr + bases + V_SCALE_OFFSET, mask=pmask, other=0).to(tl.uint16)
        vshi = tl.load(KV_cache_ptr + bases + V_SCALE_OFFSET + 1, mask=pmask, other=0).to(
            tl.uint16
        )
        vscale = (vslo | (vshi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = vval * vscale[:, None]

        new_m = tl.maximum(tl.max(scores, 0), m_prev)
        rescale = tl.exp(m_prev - new_m)
        p = tl.exp(scores - new_m)
        acc = acc * rescale + tl.sum(p[:, None] * values, axis=0)
        l_prev = l_prev * rescale + tl.sum(p, axis=0)
        m_prev = new_m

    obase = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + obase + d, acc / safe_l, mask=dmask)
    tl.store(Mid_o_ptr + obase + HEAD_DIM, m_prev + tl.log(safe_l))


@triton.jit
def _full_hq_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    K_codebook_ptr,
    V_codebook_ptr,
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
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
    K_CODE_BITS: tl.constexpr,
    V_CODE_BITS: tl.constexpr,
    K_CODE_BYTES: tl.constexpr,
    V_CODE_BYTES: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS
    pidx = pos // PAGE_SIZE
    poff = pos % PAGE_SIZE
    page = tl.load(Block_table_ptr + bid * stride_bt_b + pidx).to(tl.int64)
    base = (
        page * stride_cache_block
        + tl.cast(poff, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )
    d = tl.arange(0, BLOCK_D)
    mask = d < HEAD_DIM

    kbidx = d // K_BLOCK_SIZE
    kelem = d - kbidx * K_BLOCK_SIZE
    kbit = kbidx * K_CODE_BITS
    kbyte = kbit // 8
    kshift = kbit - kbyte * 8
    kr0 = tl.load(KV_cache_ptr + base + kbyte, mask=mask & (kbyte < K_CODE_BYTES), other=0).to(
        tl.int32
    )
    kr1 = tl.load(
        KV_cache_ptr + base + kbyte + 1,
        mask=mask & ((kbyte + 1) < K_CODE_BYTES),
        other=0,
    ).to(tl.int32)
    kr2 = tl.load(
        KV_cache_ptr + base + kbyte + 2,
        mask=mask & ((kbyte + 2) < K_CODE_BYTES),
        other=0,
    ).to(tl.int32)
    kcode = ((kr0 | (kr1 << 8) | (kr2 << 16)) >> kshift) & ((1 << K_CODE_BITS) - 1)
    kval = tl.load(K_codebook_ptr + kcode * K_BLOCK_SIZE + kelem, mask=mask, other=0.0)
    ks0 = tl.load(KV_cache_ptr + base + K_SCALE_OFFSET).to(tl.uint16)
    ks1 = tl.load(KV_cache_ptr + base + K_SCALE_OFFSET + 1).to(tl.uint16)
    kscale = (ks0 | (ks1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    vbidx = d // V_BLOCK_SIZE
    velem = d - vbidx * V_BLOCK_SIZE
    vbit = vbidx * V_CODE_BITS
    vbyte = vbit // 8
    vshift = vbit - vbyte * 8
    vr0 = tl.load(
        KV_cache_ptr + base + V_CODES_OFFSET + vbyte,
        mask=mask & (vbyte < V_CODE_BYTES),
        other=0,
    ).to(tl.int32)
    vr1 = tl.load(
        KV_cache_ptr + base + V_CODES_OFFSET + vbyte + 1,
        mask=mask & ((vbyte + 1) < V_CODE_BYTES),
        other=0,
    ).to(tl.int32)
    vr2 = tl.load(
        KV_cache_ptr + base + V_CODES_OFFSET + vbyte + 2,
        mask=mask & ((vbyte + 2) < V_CODE_BYTES),
        other=0,
    ).to(tl.int32)
    vcode = ((vr0 | (vr1 << 8) | (vr2 << 16)) >> vshift) & ((1 << V_CODE_BITS) - 1)
    vval = tl.load(V_codebook_ptr + vcode * V_BLOCK_SIZE + velem, mask=mask, other=0.0)
    vs0 = tl.load(KV_cache_ptr + base + V_SCALE_OFFSET).to(tl.uint16)
    vs1 = tl.load(KV_cache_ptr + base + V_SCALE_OFFSET + 1).to(tl.uint16)
    vscale = (vs0 | (vs1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)

    kobase = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    vobase = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(K_out_ptr + kobase + d, kval.to(tl.float32) * kscale, mask=mask)
    tl.store(V_out_ptr + vobase + d, vval.to(tl.float32) * vscale, mask=mask)


def triton_full_hayatequant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    recipe: FullHQNativeRecipe,
    scale: float,
    mid_o_buf: torch.Tensor | None = None,
    rotated_output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """Attend directly over packed HQ K/V and invert V rotation once.

    If ``v_hat_i = (s_i c_i) W_v^T``, linearity gives
    ``sum_i a_i v_hat_i = (sum_i a_i s_i c_i) W_v^T``.  The kernel therefore
    accumulates values in rotated space in fp32, performs the split reduction
    in fp32, applies exactly one fp32 inverse-rotation GEMM per output vector,
    and only then rounds to the query dtype.
    """

    bsz, hq, d = query.shape
    hk = kv_cache.shape[2]
    if hq % hk:
        raise ValueError("query heads must be divisible by KV heads")
    recipe.validate_head_dim(d)
    layout = recipe.layout(d)
    if kv_cache.shape[-1] != layout["slot_size_aligned"]:
        raise ValueError("packed HQ cache layout mismatch")
    qrot = (query.float() @ key_rotation).contiguous()
    splits = int(max_num_kv_splits)
    if (
        mid_o_buf is not None
        and mid_o_buf.dtype == torch.float32
        and mid_o_buf.shape[0] >= bsz
        and mid_o_buf.shape[1] >= hq
        and mid_o_buf.shape[2] >= splits
    ):
        mid_o = mid_o_buf[:bsz, :hq, :splits, : d + 1]
    else:
        mid_o = torch.empty(bsz, hq, splits, d + 1, device=query.device, dtype=torch.float32)
        if buf_holder is not None:
            buf_holder._hq_full_mid_o = mid_o
    block_d = triton.next_power_of_2(d)
    use_b2_fast_path = (
        recipe.key_block_size == 2
        and recipe.value_block_size == 2
        and recipe.key_code_bits_per_block <= 9
        and recipe.value_code_bits_per_block <= 9
    )
    stage1 = _full_hq_decode_stage1_b2 if use_b2_fast_path else _full_hq_decode_stage1
    block_kv = 8 if use_b2_fast_path else 16
    stage1_num_warps = 2 if use_b2_fast_path else 4
    common = dict(
        NUM_KV_HEADS=hk,
        HEAD_DIM=d,
        PAGE_SIZE=kv_cache.shape[1],
        NUM_KV_SPLITS=splits,
        KV_GROUP_SIZE=hq // hk,
        K_CODE_BITS=recipe.key_code_bits_per_block,
        V_CODE_BITS=recipe.value_code_bits_per_block,
        K_CODE_BYTES=layout["key_code_bytes"],
        V_CODE_BYTES=layout["value_code_bytes"],
        K_SCALE_OFFSET=layout["key_scale_offset"],
        V_CODES_OFFSET=layout["value_codes_offset"],
        V_SCALE_OFFSET=layout["value_scale_offset"],
        ATTN_SCALE=scale,
        BLOCK_D=block_d,
        BLOCK_KV=block_kv,
        num_warps=stage1_num_warps,
        num_stages=1,
    )
    if stage1 is _full_hq_decode_stage1_b2:
        common["BLOCK_NB"] = triton.next_power_of_2(d // 2)
    else:
        common["K_BLOCK_SIZE"] = recipe.key_block_size
        common["V_BLOCK_SIZE"] = recipe.value_block_size
    stage1[(bsz, hq, splits)](
        qrot,
        kv_cache,
        block_table,
        seq_lens,
        key_codebook,
        value_codebook,
        mid_o,
        qrot.stride(0),
        qrot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        **common,
    )
    if (
        rotated_output_buf is not None
        and rotated_output_buf.dtype == torch.float32
        and rotated_output_buf.shape[0] >= bsz
        and rotated_output_buf.shape[1] >= hq
    ):
        rotated = rotated_output_buf[:bsz, :hq, :d]
    else:
        rotated = torch.empty(bsz, hq, d, device=query.device, dtype=torch.float32)
        if buf_holder is not None:
            buf_holder._hq_full_rotated_output = rotated
    if lse_buf is not None and lse_buf.shape[0] >= bsz and lse_buf.shape[1] >= hq:
        lse = lse_buf[:bsz, :hq]
    else:
        lse = torch.empty(bsz, hq, device=query.device, dtype=torch.float32)
        if buf_holder is not None:
            buf_holder._hq_full_lse = lse
    _fwd_kernel_stage2[(bsz, hq)](
        mid_o,
        rotated,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        rotated.stride(0),
        rotated.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=splits,
        BLOCK_DV=block_d,
        Lv=d,
        num_warps=4,
        num_stages=2,
    )
    return (rotated @ value_rotation.t()).to(query.dtype)


def triton_full_hayatequant_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    recipe: FullHQNativeRecipe,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> None:
    """Bulk dequantize a packed cache, inverse rotating in fp32 before cast."""

    bsz, hk, alloc_len, d = k_out.shape
    if v_out.shape != k_out.shape:
        raise ValueError("k_out and v_out must have matching [B,H,S,D] shapes")
    layout = recipe.layout(d)
    krot = torch.empty_like(k_out, dtype=torch.float32)
    vrot = torch.empty_like(v_out, dtype=torch.float32)
    _full_hq_dequant_kv[(alloc_len, bsz * hk)](
        kv_cache,
        block_table,
        key_codebook,
        value_codebook,
        krot,
        vrot,
        krot.stride(0),
        krot.stride(1),
        krot.stride(2),
        vrot.stride(0),
        vrot.stride(1),
        vrot.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=d,
        PAGE_SIZE=kv_cache.shape[1],
        NUM_KV_HEADS=hk,
        K_BLOCK_SIZE=recipe.key_block_size,
        V_BLOCK_SIZE=recipe.value_block_size,
        K_CODE_BITS=recipe.key_code_bits_per_block,
        V_CODE_BITS=recipe.value_code_bits_per_block,
        K_CODE_BYTES=layout["key_code_bytes"],
        V_CODE_BYTES=layout["value_code_bytes"],
        K_SCALE_OFFSET=layout["key_scale_offset"],
        V_CODES_OFFSET=layout["value_codes_offset"],
        V_SCALE_OFFSET=layout["value_scale_offset"],
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )
    k_out.copy_((krot @ key_rotation.t()).to(k_out.dtype))
    v_out.copy_((vrot @ value_rotation.t()).to(v_out.dtype))


__all__ = [
    "triton_full_hayatequant_store",
    "triton_full_hayatequant_decode_attention",
    "triton_full_hayatequant_dequant_kv",
]
