"""
Packed canonical HayateQuant store and attention kernels.

This module quantizes both roles with HayateQuant's data-free rotated block
codebooks. Keys use norm correction (NC); values use one LS fit,
re-assignment, and a final LS refit. CUDA assignment always uses the fixed
candidate table. Each role stores one signed fp16 scale and no zero point.

The fixed CUDA candidate table makes assignment independent of request shape,
so prefill and one-token decode use identical arithmetic. Code indices,
scales, and packed bytes remain under the public-payload parity gate.
Packed-cache decode and attention are native Triton.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

from onecomp.kvcomp.codec._scale import fit_scale
from onecomp.kvcomp.codec.hayate.vq import (
    lookup_codewords,
    quantize_and_lookup,
    quantize_exact_pre,
    quantize_via_lut,
)
from vllm_plugins.kvcomp.hayate_config import (
    HAYATEQUANT_LUT_BINS,
    HAYATEQUANT_LUT_CANDIDATES,
    HAYATEQUANT_LUT_LIMIT,
    HayateQuantNativeRecipe,
)


@triton.jit
def _hayatequant_dense_rotation(
    X_ptr,
    Rotation_ptr,
    Output_ptr,
    rows,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """IEEE-accurate dense rotation for the tiny decode/store matrices."""

    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start in range(0, BLOCK_D, BLOCK_K):
        inner = start + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + row[:, None] * DIM + inner[None, :],
            mask=(row[:, None] < rows) & (inner[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        rotation = tl.load(
            Rotation_ptr + inner[:, None] * DIM + col[None, :],
            mask=(inner[:, None] < DIM) & (col[None, :] < DIM),
            other=0.0,
        )
        acc = tl.dot(x, rotation, acc, input_precision="ieee")
    tl.store(
        Output_ptr + row[:, None] * DIM + col[None, :],
        acc,
        mask=(row[:, None] < rows) & (col[None, :] < DIM),
    )


def _rotate_dense_small_rows(
    x: torch.Tensor,
    rotation: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``x @ rotation`` with a low-launch-overhead decode fast path.

    The dedicated kernel is faster for the 8--32 rows used by one-token
    Llama/Qwen decode.  Larger or strided inputs retain PyTorch GEMM, which is
    preferable for prompt processing and uncommon layouts.
    """

    if x.ndim < 2:
        raise ValueError("rotation input must have at least two dimensions")
    dim = x.shape[-1]
    if rotation.shape != (dim, dim):
        raise ValueError(f"rotation must have shape ({dim}, {dim})")
    if rotation.dtype != torch.float32:
        raise ValueError("rotation must use float32")
    rows = x.numel() // dim
    if output is None:
        output = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    elif output.shape != x.shape or output.dtype != torch.float32:
        raise ValueError("rotation output must match the input shape and use float32")
    if output.device != x.device or rotation.device != x.device:
        raise ValueError("rotation input, matrix, and output must share a device")
    if not output.is_contiguous():
        raise ValueError("rotation output must be contiguous")

    if x.is_cuda and rows <= 32 and x.is_contiguous() and rotation.is_contiguous():
        block_d = triton.next_power_of_2(dim)
        _hayatequant_dense_rotation[(triton.cdiv(rows, 32),)](
            x,
            rotation,
            output,
            rows,
            DIM=dim,
            BLOCK_D=block_d,
            BLOCK_M=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
    else:
        output.copy_(x.float() @ rotation)
    return output


@triton.jit
def _sub_rn_f16x2(a, b):
    return tl.inline_asm_elementwise(
        asm="sub.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _mul_rn_f16x2(a, b):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


def _assignment_lut(
    recipe: HayateQuantNativeRecipe,
    role: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Resolve the public codec's immutable cached table for one role."""

    if role not in {"k", "v"}:
        raise ValueError("role must be 'k' or 'v'")
    from onecomp.kvcomp.codec.hayate.codec import (  # pylint: disable=import-outside-toplevel
        _block_lut_code_bits_on,
    )

    code_bits = recipe.key_code_bits_per_block if role == "k" else recipe.value_code_bits_per_block
    return _block_lut_code_bits_on(
        code_bits,
        HAYATEQUANT_LUT_BINS,
        HAYATEQUANT_LUT_CANDIDATES,
        str(device),
    )


def _encode_rotated_role(
    xr: torch.Tensor,
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
    """Encode a pre-rotated HQ role with public-codec parity.

    Keeping this body separate from the rotation lets the store path share
    one *identical* fixed rotation between K and V.  It deliberately does not
    share any assignment or scale work: K NC and V LS-refit remain the exact
    role-specific HQ recipe.
    """

    if xr.shape[-1] % block_size:
        raise ValueError(f"block_size={block_size} must divide D={xr.shape[-1]}")
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    n_blocks = xr.shape[-1] // block_size

    def lookup(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = vector.reshape(*vector.shape[:-1], n_blocks, block_size)
        if lut is None:
            if scoring_codebook is None or scoring_csq is None:
                codes, chat_blocks = quantize_and_lookup(blocks, codebook)
            else:
                score_blocks = (
                    blocks.half() if blocks.is_cuda and blocks.dtype == torch.float32 else blocks
                )
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
        return codes, chat_blocks.reshape(*vector.shape[:-1], vector.shape[-1])

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
    """Rotate then encode one role; retained for unequal K/V rotations."""

    return _encode_rotated_role(
        x.float() @ rotation,
        codebook,
        block_size,
        scale_mode,
        lut=lut,
        lut_bins=lut_bins,
        lut_limit=lut_limit,
        scoring_codebook=scoring_codebook,
        scoring_csq=scoring_csq,
    )


def _encode_roles_with_shared_rotation(
    key: torch.Tensor,
    value: torch.Tensor,
    rotation: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    recipe: HayateQuantNativeRecipe,
    *,
    key_lut: torch.Tensor | None,
    value_lut: torch.Tensor | None,
    key_scoring_codebook: torch.Tensor | None,
    key_scoring_csq: torch.Tensor | None,
    value_scoring_codebook: torch.Tensor | None,
    value_scoring_csq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode K/V after one exact shared-rotation GEMM.

    The frozen b=3/b=4 HQ recipes resolve both roles to the same cached
    data-free rotation.  The old path launched two under-filled FP32 GEMMs;
    concatenating their independent rows makes one larger GEMM without
    changing any row's mathematical operation.  Pointer identity is checked
    by the caller, so this routine can never collapse distinct role rotations.
    """

    if key.shape != value.shape:
        raise ValueError("shared HayateQuant rotation requires matching K/V shapes")
    rotated = torch.cat((key, value), dim=0).float() @ rotation
    key_xr, value_xr = rotated.split(key.shape[0], dim=0)
    key_codes, key_scales = _encode_rotated_role(
        key_xr,
        key_codebook,
        recipe.key_block_size,
        "nc",
        lut=key_lut,
        lut_bins=HAYATEQUANT_LUT_BINS,
        lut_limit=HAYATEQUANT_LUT_LIMIT,
        scoring_codebook=key_scoring_codebook,
        scoring_csq=key_scoring_csq,
    )
    value_codes, value_scales = _encode_rotated_role(
        value_xr,
        value_codebook,
        recipe.value_block_size,
        "ls_iter",
        lut=value_lut,
        lut_bins=HAYATEQUANT_LUT_BINS,
        lut_limit=HAYATEQUANT_LUT_LIMIT,
        scoring_codebook=value_scoring_codebook,
        scoring_csq=value_scoring_csq,
    )
    return key_codes, key_scales, value_codes, value_scales


@triton.jit
def _hayatequant_encode_roles_b2(
    K_xr_ptr,
    V_xr_ptr,
    K_lut_ptr,
    V_lut_ptr,
    K_cb_ptr,
    V_cb_ptr,
    K_cbh_ptr,
    V_cbh_ptr,
    K_csq_ptr,
    V_csq_ptr,
    K_codes_ptr,
    V_codes_ptr,
    K_scale_ptr,
    V_scale_ptr,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BINS: tl.constexpr,
    LIM: tl.constexpr,
    STEP: tl.constexpr,
):
    """Fuse canonical b=2 K-NC/V-LS-iter encoding into one launch."""

    row = tl.program_id(0)
    blocks = tl.arange(0, NUM_BLOCKS)
    base = row * HEAD_DIM + blocks * 2

    k0 = tl.load(K_xr_ptr + base)
    k1 = tl.load(K_xr_ptr + base + 1)
    ksum = tl.sum(k0 * k0 + k1 * k1, axis=0)
    krms = tl.maximum(tl.sqrt(ksum / HEAD_DIM), 1.0e-8)
    kz0 = k0 / krms
    kz1 = k1 / krms
    kzi0 = ((tl.maximum(tl.minimum(kz0, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    kzi1 = ((tl.maximum(tl.minimum(kz1, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    kzi0 = tl.maximum(0, tl.minimum(kzi0, BINS - 1))
    kzi1 = tl.maximum(0, tl.minimum(kzi1, BINS - 1))
    kflat = kzi0 * BINS + kzi1
    kc0 = tl.load(K_lut_ptr + kflat * 2).to(tl.int32)
    kc1 = tl.load(K_lut_ptr + kflat * 2 + 1).to(tl.int32)
    kzh0 = kz0.to(tl.float16)
    kzh1 = kz1.to(tl.float16)
    kcb00 = tl.load(K_cbh_ptr + kc0 * 2)
    kcb01 = tl.load(K_cbh_ptr + kc0 * 2 + 1)
    kcb10 = tl.load(K_cbh_ptr + kc1 * 2)
    kcb11 = tl.load(K_cbh_ptr + kc1 * 2 + 1)
    kdot0 = (
        kzh0.to(tl.float32) * kcb00.to(tl.float32) + kzh1.to(tl.float32) * kcb01.to(tl.float32)
    ).to(tl.float16)
    kdot1 = (
        kzh0.to(tl.float32) * kcb10.to(tl.float32) + kzh1.to(tl.float32) * kcb11.to(tl.float32)
    ).to(tl.float16)
    kpen0 = _mul_rn_f16x2(tl.load(K_csq_ptr + kc0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    kpen1 = _mul_rn_f16x2(tl.load(K_csq_ptr + kc1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    kscore0 = _sub_rn_f16x2(kdot0, kpen0)
    kscore1 = _sub_rn_f16x2(kdot1, kpen1)
    kcode = tl.where(kscore1 > kscore0, kc1, kc0)
    kchat0 = tl.load(K_cb_ptr + kcode * 2)
    kchat1 = tl.load(K_cb_ptr + kcode * 2 + 1)
    kchat_norm = tl.maximum(
        tl.sqrt(tl.sum(kchat0 * kchat0 + kchat1 * kchat1, axis=0)),
        1.0e-8,
    )
    kscale = tl.sqrt(ksum) / kchat_norm

    v0 = tl.load(V_xr_ptr + base)
    v1 = tl.load(V_xr_ptr + base + 1)
    vsum = tl.sum(v0 * v0 + v1 * v1, axis=0)
    vrms = tl.maximum(tl.sqrt(vsum / HEAD_DIM), 1.0e-8)
    vz0 = v0 / vrms
    vz1 = v1 / vrms
    vzi0 = ((tl.maximum(tl.minimum(vz0, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    vzi1 = ((tl.maximum(tl.minimum(vz1, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    vzi0 = tl.maximum(0, tl.minimum(vzi0, BINS - 1))
    vzi1 = tl.maximum(0, tl.minimum(vzi1, BINS - 1))
    vflat = vzi0 * BINS + vzi1
    vc0 = tl.load(V_lut_ptr + vflat * 2).to(tl.int32)
    vc1 = tl.load(V_lut_ptr + vflat * 2 + 1).to(tl.int32)
    vzh0 = vz0.to(tl.float16)
    vzh1 = vz1.to(tl.float16)
    vcb00 = tl.load(V_cbh_ptr + vc0 * 2)
    vcb01 = tl.load(V_cbh_ptr + vc0 * 2 + 1)
    vcb10 = tl.load(V_cbh_ptr + vc1 * 2)
    vcb11 = tl.load(V_cbh_ptr + vc1 * 2 + 1)
    vdot0 = (
        vzh0.to(tl.float32) * vcb00.to(tl.float32) + vzh1.to(tl.float32) * vcb01.to(tl.float32)
    ).to(tl.float16)
    vdot1 = (
        vzh0.to(tl.float32) * vcb10.to(tl.float32) + vzh1.to(tl.float32) * vcb11.to(tl.float32)
    ).to(tl.float16)
    vpen0 = _mul_rn_f16x2(tl.load(V_csq_ptr + vc0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    vpen1 = _mul_rn_f16x2(tl.load(V_csq_ptr + vc1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    vscore0 = _sub_rn_f16x2(vdot0, vpen0)
    vscore1 = _sub_rn_f16x2(vdot1, vpen1)
    vcode = tl.where(vscore1 > vscore0, vc1, vc0)
    vchat0 = tl.load(V_cb_ptr + vcode * 2)
    vchat1 = tl.load(V_cb_ptr + vcode * 2 + 1)
    vnum = tl.sum(v0 * vchat0 + v1 * vchat1, axis=0)
    vden = tl.maximum(
        tl.sum(vchat0 * vchat0 + vchat1 * vchat1, axis=0),
        1.0e-8,
    )
    vscale_initial = vnum / vden
    vscale_safe = tl.where(
        vscale_initial < 0,
        -tl.maximum(tl.abs(vscale_initial), 1.0e-8),
        tl.maximum(tl.abs(vscale_initial), 1.0e-8),
    )
    vz20 = v0 / vscale_safe
    vz21 = v1 / vscale_safe
    v2zi0 = ((tl.maximum(tl.minimum(vz20, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    v2zi1 = ((tl.maximum(tl.minimum(vz21, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    v2zi0 = tl.maximum(0, tl.minimum(v2zi0, BINS - 1))
    v2zi1 = tl.maximum(0, tl.minimum(v2zi1, BINS - 1))
    v2flat = v2zi0 * BINS + v2zi1
    v2c0 = tl.load(V_lut_ptr + v2flat * 2).to(tl.int32)
    v2c1 = tl.load(V_lut_ptr + v2flat * 2 + 1).to(tl.int32)
    v2zh0 = vz20.to(tl.float16)
    v2zh1 = vz21.to(tl.float16)
    v2cb00 = tl.load(V_cbh_ptr + v2c0 * 2)
    v2cb01 = tl.load(V_cbh_ptr + v2c0 * 2 + 1)
    v2cb10 = tl.load(V_cbh_ptr + v2c1 * 2)
    v2cb11 = tl.load(V_cbh_ptr + v2c1 * 2 + 1)
    v2dot0 = (
        v2zh0.to(tl.float32) * v2cb00.to(tl.float32) + v2zh1.to(tl.float32) * v2cb01.to(tl.float32)
    ).to(tl.float16)
    v2dot1 = (
        v2zh0.to(tl.float32) * v2cb10.to(tl.float32) + v2zh1.to(tl.float32) * v2cb11.to(tl.float32)
    ).to(tl.float16)
    v2pen0 = _mul_rn_f16x2(tl.load(V_csq_ptr + v2c0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    v2pen1 = _mul_rn_f16x2(tl.load(V_csq_ptr + v2c1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    v2score0 = _sub_rn_f16x2(v2dot0, v2pen0)
    v2score1 = _sub_rn_f16x2(v2dot1, v2pen1)
    v2code = tl.where(v2score1 > v2score0, v2c1, v2c0)
    v2chat0 = tl.load(V_cb_ptr + v2code * 2)
    v2chat1 = tl.load(V_cb_ptr + v2code * 2 + 1)
    v2num = tl.sum(v0 * v2chat0 + v1 * v2chat1, axis=0)
    v2den = tl.maximum(
        tl.sum(v2chat0 * v2chat0 + v2chat1 * v2chat1, axis=0),
        1.0e-8,
    )
    vscale = v2num / v2den

    tl.store(K_codes_ptr + row * NUM_BLOCKS + blocks, kcode.to(tl.int32))
    tl.store(V_codes_ptr + row * NUM_BLOCKS + blocks, v2code.to(tl.int32))
    tl.store(K_scale_ptr + row, kscale.to(tl.float16))
    tl.store(V_scale_ptr + row, vscale.to(tl.float16))


def _encode_rotated_roles_b2(
    key_xr: torch.Tensor,
    value_xr: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    key_lut: torch.Tensor,
    value_lut: torch.Tensor,
    key_scoring_codebook: torch.Tensor,
    key_scoring_csq: torch.Tensor,
    value_scoring_codebook: torch.Tensor,
    value_scoring_csq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the serving-only fused canonical b=2 encoder."""

    if key_xr.shape != value_xr.shape or key_xr.shape[-1] % 2:
        raise ValueError("fused HayateQuant encoding requires matching even-width roles")
    rows = key_xr.numel() // key_xr.shape[-1]
    head_dim = int(key_xr.shape[-1])
    num_blocks = head_dim // 2
    key_codes = torch.empty(rows, num_blocks, dtype=torch.int32, device=key_xr.device)
    value_codes = torch.empty_like(key_codes)
    key_scales = torch.empty(rows, dtype=torch.float16, device=key_xr.device)
    value_scales = torch.empty_like(key_scales)
    _hayatequant_encode_roles_b2[(rows,)](
        key_xr,
        value_xr,
        key_lut,
        value_lut,
        key_codebook,
        value_codebook,
        key_scoring_codebook,
        value_scoring_codebook,
        key_scoring_csq,
        value_scoring_csq,
        key_codes,
        value_codes,
        key_scales,
        value_scales,
        HEAD_DIM=head_dim,
        NUM_BLOCKS=num_blocks,
        BINS=HAYATEQUANT_LUT_BINS,
        LIM=HAYATEQUANT_LUT_LIMIT,
        STEP=HAYATEQUANT_LUT_BINS / (2.0 * HAYATEQUANT_LUT_LIMIT),
        num_warps=4,
        num_stages=1,
    )
    return key_codes, key_scales, value_codes, value_scales


@triton.jit
def _hayatequant_encode_pack_store_b2(
    K_xr_ptr,
    V_xr_ptr,
    K_lut_ptr,
    V_lut_ptr,
    K_cb_ptr,
    V_cb_ptr,
    K_cbh_ptr,
    V_cbh_ptr,
    K_csq_ptr,
    V_csq_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BINS: tl.constexpr,
    LIM: tl.constexpr,
    STEP: tl.constexpr,
    K_CODE_BITS: tl.constexpr,
    V_CODE_BITS: tl.constexpr,
    K_CODE_BYTES: tl.constexpr,
    V_CODE_BYTES: tl.constexpr,
    K_SCALE_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALE_OFFSET: tl.constexpr,
):
    """Canonical b=2 encode, bit-pack, and paged scatter in one launch."""

    row = tl.program_id(0)
    token = row // H
    head = row % H
    slot = tl.load(Slot_mapping_ptr + token)
    if slot < 0:
        return

    blocks = tl.arange(0, NUM_BLOCKS)
    vector_base = row * HEAD_DIM + blocks * 2

    k0 = tl.load(K_xr_ptr + vector_base)
    k1 = tl.load(K_xr_ptr + vector_base + 1)
    ksum = tl.sum(k0 * k0 + k1 * k1, axis=0)
    krms = tl.maximum(tl.sqrt(ksum / HEAD_DIM), 1.0e-8)
    kz0 = k0 / krms
    kz1 = k1 / krms
    kzi0 = ((tl.maximum(tl.minimum(kz0, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    kzi1 = ((tl.maximum(tl.minimum(kz1, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    kzi0 = tl.maximum(0, tl.minimum(kzi0, BINS - 1))
    kzi1 = tl.maximum(0, tl.minimum(kzi1, BINS - 1))
    kflat = kzi0 * BINS + kzi1
    kc0 = tl.load(K_lut_ptr + kflat * 2).to(tl.int32)
    kc1 = tl.load(K_lut_ptr + kflat * 2 + 1).to(tl.int32)
    kzh0 = kz0.to(tl.float16)
    kzh1 = kz1.to(tl.float16)
    kcb00 = tl.load(K_cbh_ptr + kc0 * 2)
    kcb01 = tl.load(K_cbh_ptr + kc0 * 2 + 1)
    kcb10 = tl.load(K_cbh_ptr + kc1 * 2)
    kcb11 = tl.load(K_cbh_ptr + kc1 * 2 + 1)
    kdot0 = (
        kzh0.to(tl.float32) * kcb00.to(tl.float32) + kzh1.to(tl.float32) * kcb01.to(tl.float32)
    ).to(tl.float16)
    kdot1 = (
        kzh0.to(tl.float32) * kcb10.to(tl.float32) + kzh1.to(tl.float32) * kcb11.to(tl.float32)
    ).to(tl.float16)
    kpen0 = _mul_rn_f16x2(tl.load(K_csq_ptr + kc0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    kpen1 = _mul_rn_f16x2(tl.load(K_csq_ptr + kc1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    kscore0 = _sub_rn_f16x2(kdot0, kpen0)
    kscore1 = _sub_rn_f16x2(kdot1, kpen1)
    kcode = tl.where(kscore1 > kscore0, kc1, kc0)
    kchat0 = tl.load(K_cb_ptr + kcode * 2)
    kchat1 = tl.load(K_cb_ptr + kcode * 2 + 1)
    kchat_norm = tl.maximum(
        tl.sqrt(tl.sum(kchat0 * kchat0 + kchat1 * kchat1, axis=0)),
        1.0e-8,
    )
    kscale = tl.sqrt(ksum) / kchat_norm

    v0 = tl.load(V_xr_ptr + vector_base)
    v1 = tl.load(V_xr_ptr + vector_base + 1)
    vsum = tl.sum(v0 * v0 + v1 * v1, axis=0)
    vrms = tl.maximum(tl.sqrt(vsum / HEAD_DIM), 1.0e-8)
    vz0 = v0 / vrms
    vz1 = v1 / vrms
    vzi0 = ((tl.maximum(tl.minimum(vz0, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    vzi1 = ((tl.maximum(tl.minimum(vz1, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    vzi0 = tl.maximum(0, tl.minimum(vzi0, BINS - 1))
    vzi1 = tl.maximum(0, tl.minimum(vzi1, BINS - 1))
    vflat = vzi0 * BINS + vzi1
    vc0 = tl.load(V_lut_ptr + vflat * 2).to(tl.int32)
    vc1 = tl.load(V_lut_ptr + vflat * 2 + 1).to(tl.int32)
    vzh0 = vz0.to(tl.float16)
    vzh1 = vz1.to(tl.float16)
    vcb00 = tl.load(V_cbh_ptr + vc0 * 2)
    vcb01 = tl.load(V_cbh_ptr + vc0 * 2 + 1)
    vcb10 = tl.load(V_cbh_ptr + vc1 * 2)
    vcb11 = tl.load(V_cbh_ptr + vc1 * 2 + 1)
    vdot0 = (
        vzh0.to(tl.float32) * vcb00.to(tl.float32) + vzh1.to(tl.float32) * vcb01.to(tl.float32)
    ).to(tl.float16)
    vdot1 = (
        vzh0.to(tl.float32) * vcb10.to(tl.float32) + vzh1.to(tl.float32) * vcb11.to(tl.float32)
    ).to(tl.float16)
    vpen0 = _mul_rn_f16x2(tl.load(V_csq_ptr + vc0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    vpen1 = _mul_rn_f16x2(tl.load(V_csq_ptr + vc1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    vscore0 = _sub_rn_f16x2(vdot0, vpen0)
    vscore1 = _sub_rn_f16x2(vdot1, vpen1)
    vcode = tl.where(vscore1 > vscore0, vc1, vc0)
    vchat0 = tl.load(V_cb_ptr + vcode * 2)
    vchat1 = tl.load(V_cb_ptr + vcode * 2 + 1)
    vnum = tl.sum(v0 * vchat0 + v1 * vchat1, axis=0)
    vden = tl.maximum(
        tl.sum(vchat0 * vchat0 + vchat1 * vchat1, axis=0),
        1.0e-8,
    )
    vscale_initial = vnum / vden
    vscale_safe = tl.where(
        vscale_initial < 0,
        -tl.maximum(tl.abs(vscale_initial), 1.0e-8),
        tl.maximum(tl.abs(vscale_initial), 1.0e-8),
    )
    vz20 = v0 / vscale_safe
    vz21 = v1 / vscale_safe
    v2zi0 = ((tl.maximum(tl.minimum(vz20, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    v2zi1 = ((tl.maximum(tl.minimum(vz21, LIM), -LIM) + LIM) * STEP).to(tl.int32)
    v2zi0 = tl.maximum(0, tl.minimum(v2zi0, BINS - 1))
    v2zi1 = tl.maximum(0, tl.minimum(v2zi1, BINS - 1))
    v2flat = v2zi0 * BINS + v2zi1
    v2c0 = tl.load(V_lut_ptr + v2flat * 2).to(tl.int32)
    v2c1 = tl.load(V_lut_ptr + v2flat * 2 + 1).to(tl.int32)
    v2zh0 = vz20.to(tl.float16)
    v2zh1 = vz21.to(tl.float16)
    v2cb00 = tl.load(V_cbh_ptr + v2c0 * 2)
    v2cb01 = tl.load(V_cbh_ptr + v2c0 * 2 + 1)
    v2cb10 = tl.load(V_cbh_ptr + v2c1 * 2)
    v2cb11 = tl.load(V_cbh_ptr + v2c1 * 2 + 1)
    v2dot0 = (
        v2zh0.to(tl.float32) * v2cb00.to(tl.float32) + v2zh1.to(tl.float32) * v2cb01.to(tl.float32)
    ).to(tl.float16)
    v2dot1 = (
        v2zh0.to(tl.float32) * v2cb10.to(tl.float32) + v2zh1.to(tl.float32) * v2cb11.to(tl.float32)
    ).to(tl.float16)
    v2pen0 = _mul_rn_f16x2(tl.load(V_csq_ptr + v2c0), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    v2pen1 = _mul_rn_f16x2(tl.load(V_csq_ptr + v2c1), tl.full((NUM_BLOCKS,), 0.5, tl.float16))
    v2score0 = _sub_rn_f16x2(v2dot0, v2pen0)
    v2score1 = _sub_rn_f16x2(v2dot1, v2pen1)
    vcode = tl.where(v2score1 > v2score0, v2c1, v2c0)
    v2chat0 = tl.load(V_cb_ptr + vcode * 2)
    v2chat1 = tl.load(V_cb_ptr + vcode * 2 + 1)
    vscale = tl.sum(v0 * v2chat0 + v1 * v2chat1, axis=0) / tl.maximum(
        tl.sum(v2chat0 * v2chat0 + v2chat1 * v2chat1, axis=0),
        1.0e-8,
    )

    page = (slot // PAGE_SIZE).to(tl.int64)
    pos = (slot % PAGE_SIZE).to(tl.int64)
    cache_base = (
        page * stride_cache_block
        + pos * stride_cache_pos
        + tl.cast(head, tl.int64) * stride_cache_head
    )
    # Reuse the 64 code lanes for packing so all local gathers have identical
    # extents.  The sole stream wider than 64 bytes is K10; four ten-bit codes
    # form exactly five bytes and are emitted explicitly per four-code group.
    pack_lane = blocks
    kcode_pack = kcode.to(tl.int32)
    vcode_pack = vcode.to(tl.int32)

    if K_CODE_BITS == 10:
        group_mask = pack_lane < NUM_BLOCKS // 4
        group_base = pack_lane * 4
        kc0 = tl.gather(kcode_pack, tl.minimum(group_base, NUM_BLOCKS - 1), axis=0)
        kc1 = tl.gather(kcode_pack, tl.minimum(group_base + 1, NUM_BLOCKS - 1), axis=0)
        kc2 = tl.gather(kcode_pack, tl.minimum(group_base + 2, NUM_BLOCKS - 1), axis=0)
        kc3 = tl.gather(kcode_pack, tl.minimum(group_base + 3, NUM_BLOCKS - 1), axis=0)
        byte_base = pack_lane * 5
        tl.store(
            KV_cache_ptr + cache_base + byte_base,
            kc0.to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + cache_base + byte_base + 1,
            ((kc0 >> 8) | (kc1 << 2)).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + cache_base + byte_base + 2,
            ((kc1 >> 6) | (kc2 << 4)).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + cache_base + byte_base + 3,
            ((kc2 >> 4) | (kc3 << 6)).to(tl.uint8),
            mask=group_mask,
        )
        tl.store(
            KV_cache_ptr + cache_base + byte_base + 4,
            (kc3 >> 2).to(tl.uint8),
            mask=group_mask,
        )
    elif K_CODE_BITS >= 7:
        kmask = pack_lane < K_CODE_BYTES
        kstream_bit = pack_lane * 8
        kidx = kstream_bit // K_CODE_BITS
        kshift = kstream_bit - kidx * K_CODE_BITS
        kidx0 = tl.minimum(kidx, NUM_BLOCKS - 1)
        kidx1 = tl.minimum(kidx + 1, NUM_BLOCKS - 1)
        kc_lo = tl.gather(kcode_pack, kidx0, axis=0)
        kc_hi = tl.gather(kcode_pack, kidx1, axis=0)
        kc_lo = tl.where(kmask & (kidx < NUM_BLOCKS), kc_lo, 0)
        kc_hi = tl.where(kmask & ((kidx + 1) < NUM_BLOCKS), kc_hi, 0)
        kout = (kc_lo >> kshift) | (kc_hi << (K_CODE_BITS - kshift))
        tl.store(KV_cache_ptr + cache_base + pack_lane, kout.to(tl.uint8), mask=kmask)
    else:
        kmask = pack_lane < K_CODE_BYTES
        kout = tl.zeros((NUM_BLOCKS,), tl.int32)
        for bit in range(8):
            stream_bit = pack_lane * 8 + bit
            code_idx = stream_bit // K_CODE_BITS
            code_bit = stream_bit - code_idx * K_CODE_BITS
            safe_idx = tl.minimum(code_idx, NUM_BLOCKS - 1)
            code = tl.gather(kcode_pack, safe_idx, axis=0)
            code = tl.where(kmask & (code_idx < NUM_BLOCKS), code, 0)
            kout |= ((code >> code_bit) & 1) << bit
        tl.store(KV_cache_ptr + cache_base + pack_lane, kout.to(tl.uint8), mask=kmask)

    vmask = pack_lane < V_CODE_BYTES
    if V_CODE_BITS >= 7:
        vstream_bit = pack_lane * 8
        vidx = vstream_bit // V_CODE_BITS
        vshift = vstream_bit - vidx * V_CODE_BITS
        vidx0 = tl.minimum(vidx, NUM_BLOCKS - 1)
        vidx1 = tl.minimum(vidx + 1, NUM_BLOCKS - 1)
        vc_lo = tl.gather(vcode_pack, vidx0, axis=0)
        vc_hi = tl.gather(vcode_pack, vidx1, axis=0)
        vc_lo = tl.where(vmask & (vidx < NUM_BLOCKS), vc_lo, 0)
        vc_hi = tl.where(vmask & ((vidx + 1) < NUM_BLOCKS), vc_hi, 0)
        vout = (vc_lo >> vshift) | (vc_hi << (V_CODE_BITS - vshift))
    else:
        vout = tl.zeros((NUM_BLOCKS,), tl.int32)
        for bit in range(8):
            stream_bit = pack_lane * 8 + bit
            code_idx = stream_bit // V_CODE_BITS
            code_bit = stream_bit - code_idx * V_CODE_BITS
            safe_idx = tl.minimum(code_idx, NUM_BLOCKS - 1)
            code = tl.gather(vcode_pack, safe_idx, axis=0)
            code = tl.where(vmask & (code_idx < NUM_BLOCKS), code, 0)
            vout |= ((code >> code_bit) & 1) << bit
    tl.store(
        KV_cache_ptr + cache_base + V_CODES_OFFSET + pack_lane,
        vout.to(tl.uint8),
        mask=vmask,
    )

    ks = kscale.to(tl.float16).to(tl.uint16, bitcast=True)
    vs = vscale.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + cache_base + K_SCALE_OFFSET, (ks & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + cache_base + K_SCALE_OFFSET + 1,
        ((ks >> 8) & 0xFF).to(tl.uint8),
    )
    tl.store(KV_cache_ptr + cache_base + V_SCALE_OFFSET, (vs & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + cache_base + V_SCALE_OFFSET + 1,
        ((vs >> 8) & 0xFF).to(tl.uint8),
    )


def _encode_pack_rotated_roles_b2(
    key_xr: torch.Tensor,
    value_xr: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    key_lut: torch.Tensor,
    value_lut: torch.Tensor,
    key_scoring_codebook: torch.Tensor,
    key_scoring_csq: torch.Tensor,
    value_scoring_codebook: torch.Tensor,
    value_scoring_csq: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    recipe: HayateQuantNativeRecipe,
    *,
    num_warps: int = 4,
) -> None:
    """Launch the allocation-free fused serving store for canonical b=2."""

    n, h, head_dim = key_xr.shape
    layout = recipe.layout(head_dim)
    _hayatequant_encode_pack_store_b2[(n * h,)](
        key_xr,
        value_xr,
        key_lut,
        value_lut,
        key_codebook,
        value_codebook,
        key_scoring_codebook,
        value_scoring_codebook,
        key_scoring_csq,
        value_scoring_csq,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        H=h,
        PAGE_SIZE=kv_cache.shape[1],
        HEAD_DIM=head_dim,
        NUM_BLOCKS=head_dim // 2,
        BINS=HAYATEQUANT_LUT_BINS,
        LIM=HAYATEQUANT_LUT_LIMIT,
        STEP=HAYATEQUANT_LUT_BINS / (2.0 * HAYATEQUANT_LUT_LIMIT),
        K_CODE_BITS=recipe.key_code_bits_per_block,
        V_CODE_BITS=recipe.value_code_bits_per_block,
        K_CODE_BYTES=layout["key_code_bytes"],
        V_CODE_BYTES=layout["value_code_bytes"],
        K_SCALE_OFFSET=layout["key_scale_offset"],
        V_CODES_OFFSET=layout["value_codes_offset"],
        V_SCALE_OFFSET=layout["value_scale_offset"],
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _hayatequant_pack_store(
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
    # one-bit gathers per output byte.  The fixed HayateQuant rate-3/rate-4
    # b=2 recipes use this fast path for their wider role streams. Keep the
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
    recipe: HayateQuantNativeRecipe,
    head_dim: int,
) -> None:
    """Pack pre-encoded HQ payloads; separated for stage-level profiling."""

    n = int(slot_mapping.numel())
    if n < 1 or key_codes.shape[0] % n:
        raise ValueError("encoded HQ rows must be a positive multiple of token slots")
    h = key_codes.shape[0] // n
    layout = recipe.layout(head_dim)
    nh = n * h
    _hayatequant_pack_store[(nh,)](
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


def triton_hayatequant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    recipe: HayateQuantNativeRecipe,
    key_lut: torch.Tensor | None = None,
    value_lut: torch.Tensor | None = None,
    key_scoring_codebook: torch.Tensor | None = None,
    key_scoring_csq: torch.Tensor | None = None,
    value_scoring_codebook: torch.Tensor | None = None,
    value_scoring_csq: torch.Tensor | None = None,
    shared_rotation: bool | None = None,
    value_in_rotated_basis: bool = False,
    return_payload: bool = True,
    require_fused: bool = False,
) -> dict[str, torch.Tensor] | None:
    """Encode and pack canonical HayateQuant K/V into a paged byte cache.

    ``return_payload`` exists for parity tests and diagnostic tooling.  vLLM
    serving sets it false because the cache owns the only persistent
    representation and constructing a per-layer Python dictionary is needless
    request-thread work.
    """

    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must have matching [N,H,D] shapes")
    n, h, d = key.shape
    recipe.validate_head_dim(d)
    layout = recipe.layout(d)
    if kv_cache.shape[-1] != layout["slot_size_aligned"]:
        raise ValueError(
            f"HayateQuant cache slot is {kv_cache.shape[-1]} bytes, expected "
            f"{layout['slot_size_aligned']} for D={d}"
        )
    if key_lut is None:
        key_lut = _assignment_lut(recipe, "k", key.device)
    if value_lut is None:
        value_lut = _assignment_lut(recipe, "v", value.device)
    if key_scoring_codebook is None:
        key_scoring_codebook = key_codebook.half() if key.is_cuda else key_codebook
    if value_scoring_codebook is None:
        value_scoring_codebook = value_codebook.half() if value.is_cuda else value_codebook
    if key_scoring_csq is None:
        key_scoring_csq = key_scoring_codebook.pow(2).sum(dim=1)
    if value_scoring_csq is None:
        value_scoring_csq = value_scoring_codebook.pow(2).sum(dim=1)
    # Both roles may use the same immutable rotation matrix (the frozen b=3
    # and b=4 recipes do).  Only aliasing is accepted: comparing contents at
    # request time would synchronise the host, while assuming same-looking
    # matrices could silently violate the role-specific recipe.
    if shared_rotation is None:
        shared_rotation = (
            key_rotation.device == value_rotation.device
            and key_rotation.dtype == value_rotation.dtype
            and key_rotation.shape == value_rotation.shape
            and key_rotation.data_ptr() == value_rotation.data_ptr()
        )
    elif not isinstance(shared_rotation, bool):
        raise TypeError("shared_rotation must be bool or None")
    if not isinstance(value_in_rotated_basis, bool):
        raise TypeError("value_in_rotated_basis must be bool")
    # A projection-folded value already has its role rotation.  It therefore
    # cannot join key in the shared-rotation GEMM even when the immutable K/V
    # matrices alias.
    if value_in_rotated_basis:
        shared_rotation = False
    fused_b2 = (
        key.is_cuda
        and recipe.key_block_size == 2
        and recipe.value_block_size == 2
        and key_lut is not None
        and value_lut is not None
        and key_scoring_codebook is not None
        and key_scoring_csq is not None
        and value_scoring_codebook is not None
        and value_scoring_csq is not None
        and key_scoring_codebook.dtype == torch.float16
        and value_scoring_codebook.dtype == torch.float16
        and key_scoring_csq.dtype == torch.float16
        and value_scoring_csq.dtype == torch.float16
    )
    if require_fused and not fused_b2:
        raise RuntimeError("requested HayateQuant fused b=2 CUDA store is unavailable")
    if fused_b2:
        if shared_rotation:
            rotated = _rotate_dense_small_rows(torch.cat((key, value), dim=0), key_rotation)
            key_xr, value_xr = rotated.split(key.shape[0], dim=0)
        else:
            key_xr = _rotate_dense_small_rows(key, key_rotation)
            value_xr = (
                value.float()
                if value_in_rotated_basis
                else _rotate_dense_small_rows(value, value_rotation)
            )
        if not return_payload:
            _encode_pack_rotated_roles_b2(
                key_xr,
                value_xr,
                key_codebook,
                value_codebook,
                key_lut,
                value_lut,
                key_scoring_codebook,
                key_scoring_csq,
                value_scoring_codebook,
                value_scoring_csq,
                kv_cache,
                slot_mapping,
                recipe,
                num_warps=(2 if recipe.key_code_bits_per_block == 8 else 4),
            )
            return None
        key_codes, key_scales, value_codes, value_scales = _encode_rotated_roles_b2(
            key_xr,
            value_xr,
            key_codebook,
            value_codebook,
            key_lut,
            value_lut,
            key_scoring_codebook,
            key_scoring_csq,
            value_scoring_codebook,
            value_scoring_csq,
        )
    elif shared_rotation:
        key_codes, key_scales, value_codes, value_scales = _encode_roles_with_shared_rotation(
            key,
            value,
            key_rotation,
            key_codebook,
            value_codebook,
            recipe,
            key_lut=key_lut,
            value_lut=value_lut,
            key_scoring_codebook=key_scoring_codebook,
            key_scoring_csq=key_scoring_csq,
            value_scoring_codebook=value_scoring_codebook,
            value_scoring_csq=value_scoring_csq,
        )
    else:
        key_codes, key_scales = _encode_role(
            key,
            key_rotation,
            key_codebook,
            recipe.key_block_size,
            "nc",
            lut=key_lut,
            lut_bins=HAYATEQUANT_LUT_BINS,
            lut_limit=HAYATEQUANT_LUT_LIMIT,
            scoring_codebook=key_scoring_codebook,
            scoring_csq=key_scoring_csq,
        )
        value_encoder = _encode_rotated_role if value_in_rotated_basis else _encode_role
        value_args = (
            (value.float(), value_codebook, recipe.value_block_size, "ls_iter")
            if value_in_rotated_basis
            else (
                value,
                value_rotation,
                value_codebook,
                recipe.value_block_size,
                "ls_iter",
            )
        )
        value_codes, value_scales = value_encoder(
            *value_args,
            lut=value_lut,
            lut_bins=HAYATEQUANT_LUT_BINS,
            lut_limit=HAYATEQUANT_LUT_LIMIT,
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
    if return_payload:
        return {
            "key_codes": key_codes,
            "key_scales": key_scales,
            "value_codes": value_codes,
            "value_scales": value_scales,
        }
    return None


@triton.jit
def _interleave_four(c0, c1, c2, c3):
    """Interleave four equal trailing dimensions as c0,c1,c2,c3 per group."""

    return tl.interleave(tl.interleave(c0, c2), tl.interleave(c1, c3))


@triton.jit
def _interleave_eight(c0, c1, c2, c3, c4, c5, c6, c7):
    """Interleave eight equal trailing dimensions in coordinate order."""

    even = _interleave_four(c0, c2, c4, c6)
    odd = _interleave_four(c1, c3, c5, c7)
    return tl.interleave(even, odd)


@triton.jit
def _interleave_sixteen(
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
    c12,
    c13,
    c14,
    c15,
):
    """Interleave sixteen equal trailing dimensions in coordinate order."""

    even = _interleave_eight(c0, c2, c4, c6, c8, c10, c12, c14)
    odd = _interleave_eight(c1, c3, c5, c7, c9, c11, c13, c15)
    return tl.interleave(even, odd)


@triton.jit
def _load_hayate_b2_codes(
    KV_cache_ptr,
    bases,
    pmask,
    STREAM_OFFSET: tl.constexpr,
    CODE_BITS: tl.constexpr,
    CODE_BYTES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_NB: tl.constexpr,
):
    """Load one packed b=2 code stream without duplicate byte gathers.

    Every canonical HQ width has a short byte-aligned group.  The grouped
    loaders use four 2-bit codes, two 4-bit codes, eight 6-bit codes, one
    8-bit code, or sixteen 10-bit codes.  Decoding those groups directly
    avoids issuing overlapping byte loads for every block coordinate.
    """

    if CODE_BITS == 2:
        groups = tl.arange(0, BLOCK_NB // 4)
        gmask = groups * 4 < NUM_BLOCKS
        raw = tl.load(
            KV_cache_ptr + bases[:, None] + STREAM_OFFSET + groups[None, :],
            mask=pmask[:, None] & gmask[None, :] & (groups[None, :] < CODE_BYTES),
            other=0,
        ).to(tl.int32)
        return _interleave_four(raw & 0x3, (raw >> 2) & 0x3, (raw >> 4) & 0x3, raw >> 6)

    if CODE_BITS == 4:
        groups = tl.arange(0, BLOCK_NB // 2)
        gmask = groups * 2 < NUM_BLOCKS
        raw = tl.load(
            KV_cache_ptr + bases[:, None] + STREAM_OFFSET + groups[None, :],
            mask=pmask[:, None] & gmask[None, :] & (groups[None, :] < CODE_BYTES),
            other=0,
        ).to(tl.int32)
        return tl.interleave(raw & 0xF, raw >> 4)

    if CODE_BITS == 6:
        # Eight 6-bit codes occupy six bytes.  The canonical stream starts at
        # an even offset, so three aligned uint16 loads replace 24 byte loads
        # across each eight-group tile.
        groups = tl.arange(0, BLOCK_NB // 8)
        offsets = groups * 6
        gmask = groups * 8 < NUM_BLOCKS
        mask = pmask[:, None] & gmask[None, :]
        byte_ptr = KV_cache_ptr + bases[:, None] + STREAM_OFFSET + offsets[None, :]
        w0 = tl.load(
            byte_ptr.to(tl.pointer_type(tl.uint16)),
            mask=mask & ((offsets[None, :] + 1) < CODE_BYTES),
            other=0,
        ).to(tl.uint32)
        w1 = tl.load(
            (byte_ptr + 2).to(tl.pointer_type(tl.uint16)),
            mask=mask & ((offsets[None, :] + 3) < CODE_BYTES),
            other=0,
        ).to(tl.uint32)
        w2 = tl.load(
            (byte_ptr + 4).to(tl.pointer_type(tl.uint16)),
            mask=mask & ((offsets[None, :] + 5) < CODE_BYTES),
            other=0,
        ).to(tl.uint32)
        c0 = w0 & 0x3F
        c1 = (w0 >> 6) & 0x3F
        c2 = ((w0 >> 12) | (w1 << 4)) & 0x3F
        c3 = (w1 >> 2) & 0x3F
        c4 = (w1 >> 8) & 0x3F
        c5 = ((w1 >> 14) | (w2 << 2)) & 0x3F
        c6 = (w2 >> 4) & 0x3F
        c7 = (w2 >> 10) & 0x3F
        return _interleave_eight(
            c0.to(tl.int32),
            c1.to(tl.int32),
            c2.to(tl.int32),
            c3.to(tl.int32),
            c4.to(tl.int32),
            c5.to(tl.int32),
            c6.to(tl.int32),
            c7.to(tl.int32),
        )

    if CODE_BITS == 8:
        blocks = tl.arange(0, BLOCK_NB)
        bmask = blocks < NUM_BLOCKS
        return tl.load(
            KV_cache_ptr + bases[:, None] + STREAM_OFFSET + blocks[None, :],
            mask=pmask[:, None] & bmask[None, :] & (blocks[None, :] < CODE_BYTES),
            other=0,
        ).to(tl.int32)

    if CODE_BITS == 10:
        # Sixteen 10-bit codes occupy 20 bytes.  Five aligned uint32 loads
        # cover the group exactly and reduce address/unpack issue pressure.
        groups = tl.arange(0, BLOCK_NB // 16)
        offsets = groups * 20
        gmask = groups * 16 < NUM_BLOCKS
        mask = pmask[:, None] & gmask[None, :]
        byte_ptr = KV_cache_ptr + bases[:, None] + STREAM_OFFSET + offsets[None, :]
        w0 = tl.load(
            byte_ptr.to(tl.pointer_type(tl.uint32)),
            mask=mask & ((offsets[None, :] + 3) < CODE_BYTES),
            other=0,
        )
        w1 = tl.load(
            (byte_ptr + 4).to(tl.pointer_type(tl.uint32)),
            mask=mask & ((offsets[None, :] + 7) < CODE_BYTES),
            other=0,
        )
        w2 = tl.load(
            (byte_ptr + 8).to(tl.pointer_type(tl.uint32)),
            mask=mask & ((offsets[None, :] + 11) < CODE_BYTES),
            other=0,
        )
        w3 = tl.load(
            (byte_ptr + 12).to(tl.pointer_type(tl.uint32)),
            mask=mask & ((offsets[None, :] + 15) < CODE_BYTES),
            other=0,
        )
        w4 = tl.load(
            (byte_ptr + 16).to(tl.pointer_type(tl.uint32)),
            mask=mask & ((offsets[None, :] + 19) < CODE_BYTES),
            other=0,
        )
        c0 = w0 & 0x3FF
        c1 = (w0 >> 10) & 0x3FF
        c2 = (w0 >> 20) & 0x3FF
        c3 = ((w0 >> 30) | (w1 << 2)) & 0x3FF
        c4 = (w1 >> 8) & 0x3FF
        c5 = (w1 >> 18) & 0x3FF
        c6 = ((w1 >> 28) | (w2 << 4)) & 0x3FF
        c7 = (w2 >> 6) & 0x3FF
        c8 = (w2 >> 16) & 0x3FF
        c9 = ((w2 >> 26) | (w3 << 6)) & 0x3FF
        c10 = (w3 >> 4) & 0x3FF
        c11 = (w3 >> 14) & 0x3FF
        c12 = ((w3 >> 24) | (w4 << 8)) & 0x3FF
        c13 = (w4 >> 2) & 0x3FF
        c14 = (w4 >> 12) & 0x3FF
        c15 = (w4 >> 22) & 0x3FF
        return _interleave_sixteen(
            c0.to(tl.int32),
            c1.to(tl.int32),
            c2.to(tl.int32),
            c3.to(tl.int32),
            c4.to(tl.int32),
            c5.to(tl.int32),
            c6.to(tl.int32),
            c7.to(tl.int32),
            c8.to(tl.int32),
            c9.to(tl.int32),
            c10.to(tl.int32),
            c11.to(tl.int32),
            c12.to(tl.int32),
            c13.to(tl.int32),
            c14.to(tl.int32),
            c15.to(tl.int32),
        )

    # This path is unreachable for canonical rates but preserves the helper's
    # behavior if a future recipe introduces a different width.
    blocks = tl.arange(0, BLOCK_NB)
    bmask = blocks < NUM_BLOCKS
    bit = blocks * CODE_BITS
    byte = bit // 8
    shift = bit - byte * 8
    raw0 = tl.load(
        KV_cache_ptr + bases[:, None] + STREAM_OFFSET + byte[None, :],
        mask=pmask[:, None] & bmask[None, :] & (byte[None, :] < CODE_BYTES),
        other=0,
    ).to(tl.int32)
    raw1 = tl.load(
        KV_cache_ptr + bases[:, None] + STREAM_OFFSET + byte[None, :] + 1,
        mask=pmask[:, None] & bmask[None, :] & ((byte[None, :] + 1) < CODE_BYTES),
        other=0,
    ).to(tl.int32)
    return ((raw0 | (raw1 << 8)) >> shift[None, :]) & ((1 << CODE_BITS) - 1)


@triton.jit
def _hayatequant_decode_stage1_b2(
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

        kcode = _load_hayate_b2_codes(
            KV_cache_ptr,
            bases,
            pmask,
            STREAM_OFFSET=0,
            CODE_BITS=K_CODE_BITS,
            CODE_BYTES=K_CODE_BYTES,
            NUM_BLOCKS=HEAD_DIM // 2,
            BLOCK_NB=BLOCK_NB,
        )
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
        kscale_ptr = (KV_cache_ptr + bases + K_SCALE_OFFSET).to(tl.pointer_type(tl.uint16))
        kscale = (
            tl.load(kscale_ptr, mask=pmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
        )
        dot = tl.sum(
            tl.where(bmask[None, :], q0[None, :] * kc0 + q1[None, :] * kc1, 0.0),
            axis=1,
        )
        scores = tl.where(pmask, dot * kscale * ATTN_SCALE, -float("inf"))

        vcode = _load_hayate_b2_codes(
            KV_cache_ptr,
            bases,
            pmask,
            STREAM_OFFSET=V_CODES_OFFSET,
            CODE_BITS=V_CODE_BITS,
            CODE_BYTES=V_CODE_BYTES,
            NUM_BLOCKS=HEAD_DIM // 2,
            BLOCK_NB=BLOCK_NB,
        )
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
        vscale_ptr = (KV_cache_ptr + bases + V_SCALE_OFFSET).to(tl.pointer_type(tl.uint16))
        vscale = (
            tl.load(vscale_ptr, mask=pmask, other=0).to(tl.float16, bitcast=True).to(tl.float32)
        )
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
def _hayatequant_decode_stage1(
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
def _hayatequant_dequant_kv(
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


def triton_hayatequant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    recipe: HayateQuantNativeRecipe,
    scale: float,
    mid_o_buf: torch.Tensor | None = None,
    query_rotation_buf: torch.Tensor | None = None,
    rotated_output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    output_in_rotated_basis: bool = False,
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
    qrot = _rotate_dense_small_rows(query, key_rotation, query_rotation_buf)
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
            buf_holder._hayate_mid_o = mid_o
    block_d = triton.next_power_of_2(d)
    use_b2_fast_path = (
        recipe.key_block_size == 2
        and recipe.value_block_size == 2
        and recipe.key_code_bits_per_block <= 10
        and recipe.value_code_bits_per_block <= 10
    )
    stage1 = _hayatequant_decode_stage1_b2 if use_b2_fast_path else _hayatequant_decode_stage1
    if use_b2_fast_path and recipe.rate == 4:
        # Aligned wide-load K10/V6 unpack lowers register/address pressure
        # enough for a sixteen-token tile.  One warp was fastest on B200 in
        # the exhaustive 4K serving sweep while preserving the same math.
        block_kv, stage1_num_warps = 16, 1
    else:
        block_kv = 8 if use_b2_fast_path else 16
        # Wide fp16-scale loads make one warp fastest for both K6/V2 and
        # byte/nibble-aligned K8/V4 at the formal 4K decode context.
        stage1_num_warps = 1 if use_b2_fast_path else 4
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
    if stage1 is _hayatequant_decode_stage1_b2:
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
    rotated_dtype = query.dtype if output_in_rotated_basis else torch.float32
    if (
        rotated_output_buf is not None
        and rotated_output_buf.dtype == rotated_dtype
        and rotated_output_buf.shape[0] >= bsz
        and rotated_output_buf.shape[1] >= hq
    ):
        rotated = rotated_output_buf[:bsz, :hq, :d]
    else:
        rotated = torch.empty(bsz, hq, d, device=query.device, dtype=rotated_dtype)
        if buf_holder is not None:
            buf_holder._hayate_rotated_output = rotated
    if lse_buf is not None and lse_buf.shape[0] >= bsz and lse_buf.shape[1] >= hq:
        lse = lse_buf[:bsz, :hq]
    else:
        lse = torch.empty(bsz, hq, device=query.device, dtype=torch.float32)
        if buf_holder is not None:
            buf_holder._hayate_lse = lse
    stage2_options = {}
    if "OUTPUT_FP16" in getattr(_fwd_kernel_stage2, "arg_names", ()):
        stage2_options["OUTPUT_FP16"] = 1 if rotated.dtype == torch.float16 else 0
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
        **stage2_options,
    )
    if output_in_rotated_basis:
        return rotated
    return (rotated @ value_rotation.t()).to(query.dtype)


def triton_hayatequant_dequant_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    key_codebook: torch.Tensor,
    value_codebook: torch.Tensor,
    key_rotation: torch.Tensor,
    value_rotation: torch.Tensor,
    recipe: HayateQuantNativeRecipe,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    value_in_rotated_basis: bool = False,
) -> None:
    """Bulk dequantize a packed cache into ordinary or folded value basis."""

    bsz, hk, alloc_len, d = k_out.shape
    if v_out.shape != k_out.shape:
        raise ValueError("k_out and v_out must have matching [B,H,S,D] shapes")
    layout = recipe.layout(d)
    krot = torch.empty_like(k_out, dtype=torch.float32)
    vrot = torch.empty_like(v_out, dtype=torch.float32)
    _hayatequant_dequant_kv[(alloc_len, bsz * hk)](
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
    if value_in_rotated_basis:
        v_out.copy_(vrot.to(v_out.dtype))
    else:
        v_out.copy_((vrot @ value_rotation.t()).to(v_out.dtype))


__all__ = [
    "triton_hayatequant_store",
    "triton_hayatequant_decode_attention",
    "triton_hayatequant_dequant_kv",
]
