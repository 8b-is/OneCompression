"""Bit-packed affine quantization primitives (real low-bit KV storage).

:mod:`onecomp.kvcomp.codec._quant_primitives` implements *simulated* (fake)
quantization: codes stay float and ``decode`` keeps the dequantized tensor,
which is ideal for the correctness-first research layer but stores nothing
compactly.  This module is its storage-layer counterpart: integer codes are
actually packed into ``uint8`` (one byte per 8-bit code, two 4-bit codes per
byte, four 2-bit codes per byte) with compact per-group ``float16`` scales
and zero points, so the result can be serialized byte-for-byte.

It is consumed by the session checkpoint engine
(:mod:`onecomp.kvcomp.checkpoint`, the ``.okv`` container) and by
:class:`onecomp.kvcomp.codec.packed_affine.PackedAffineCodec`, the registered
codec wrapper.

Axis conventions match the rest of :mod:`onecomp.kvcomp` for tensors of shape
``(batch, heads, seq, head_dim)``:

* ``"token"``  : groups run along ``head_dim`` (one group set per token).
* ``"channel"``: groups run along ``seq`` (one group set per channel),
  following the KIVI observation that keys are best quantized per-channel
  and values per-token.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

SUPPORTED_PACKED_BITS = (2, 4, 8)

# Scales are stored as float16; a per-group range larger than
# _FP16_MAX * q_max would overflow to inf when rounded to storage precision,
# silently turning the reconstruction into NaN.  Clamping the scale keeps the
# codes finite (the clamp only activates for pathological magnitudes ~1e6+).
_FP16_MAX = 65504.0


@dataclass
class PackedQuantizedKV:
    """Bit-packed quantized representation of a single KV-cache tensor.

    Attributes:
        codes: Bit-packed quantized codes (``uint8``, 1-D).
        scale: Per-group scale coefficients (``float16``, shape ``(num_groups, 1)``).
        zero: Per-group zero points (``float16``, shape ``(num_groups, 1)``).
        shape: Original tensor shape ``(batch, heads, seq, head_dim)``.
        bits: Number of quantization bits (2, 4 or 8).
        group_size: Number of elements per quantization group.
        axis: Quantization axis, ``"token"`` (groups along ``head_dim``)
            or ``"channel"`` (groups along ``seq``).
        num_padded: Number of zero elements appended along the quantization
            axis so that its length is divisible by ``group_size``.
        dtype: Name of the original torch dtype (e.g. ``"float16"``).
    """

    codes: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    shape: tuple
    bits: int
    group_size: int
    axis: str
    num_padded: int
    dtype: str

    def num_bytes(self) -> int:
        """Total storage bytes of codes, scales and zero points."""
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero.numel() * self.zero.element_size()
        )


def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes into ``uint8`` storage.

    Args:
        q: 1-D tensor of integer codes in ``[0, 2**bits - 1]``.
        bits: Number of bits per code (2, 4 or 8).

    Returns:
        1-D ``uint8`` tensor holding ``8 // bits`` codes per byte.
    """
    if bits == 8:
        return q.to(torch.uint8)
    vals_per_byte = 8 // bits
    pad = (-q.numel()) % vals_per_byte
    if pad:
        q = torch.cat([q, q.new_zeros(pad)])
    q = q.to(torch.int32).reshape(-1, vals_per_byte)
    packed = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    for j in range(vals_per_byte):
        packed |= q[:, j] << (j * bits)
    return packed.to(torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Unpack ``uint8`` storage back to integer codes.

    Args:
        packed: 1-D ``uint8`` tensor produced by :func:`pack_codes`.
        bits: Number of bits per code (2, 4 or 8).
        num_elements: Number of valid codes to return.

    Returns:
        1-D ``int32`` tensor of codes in ``[0, 2**bits - 1]``.
    """
    if bits == 8:
        return packed.to(torch.int32)[:num_elements]
    vals_per_byte = 8 // bits
    mask = (1 << bits) - 1
    packed = packed.to(torch.int32)
    cols = [(packed >> (j * bits)) & mask for j in range(vals_per_byte)]
    q = torch.stack(cols, dim=-1).reshape(-1)
    return q[:num_elements]


def quantize_packed(
    tensor: torch.Tensor,
    bits: int,
    group_size: int = 64,
    axis: str = "token",
) -> PackedQuantizedKV:
    """Quantize a KV-cache tensor with per-group asymmetric quantization.

    The tensor is grouped along the quantization axis (``head_dim`` for
    ``axis="token"``, ``seq`` for ``axis="channel"``) and each group is
    quantized independently to unsigned integers in ``[0, 2**bits - 1]``.
    The quantization range always includes zero, mirroring
    :func:`onecomp.quantizer.rtn.quantizer.pseudo_quantize_tensor`.

    Args:
        tensor: KV tensor of shape ``(batch, heads, seq, head_dim)``.
        bits: Number of quantization bits (2, 4 or 8).
        group_size: Number of elements per quantization group.  The
            quantization axis is zero-padded to a multiple of this value.
        axis: ``"token"`` to group along ``head_dim`` (per-token groups),
            ``"channel"`` to group along ``seq`` (per-channel groups).

    Returns:
        A :class:`PackedQuantizedKV` holding bit-packed codes and
        quantization parameters sufficient to reconstruct the tensor.

    Raises:
        ValueError: If ``bits``, ``axis`` or the tensor shape is invalid,
            or if the tensor contains non-finite values (``inf``/``NaN``),
            which would silently corrupt the per-group scales.
    """
    if bits not in SUPPORTED_PACKED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_PACKED_BITS}, got {bits}")
    if axis not in ("token", "channel"):
        raise ValueError(f'axis must be "token" or "channel", got {axis!r}')
    if tensor.dim() != 4:
        raise ValueError(
            f"Expected a 4-D (batch, heads, seq, head_dim) tensor, got {tensor.shape}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not torch.isfinite(tensor).all():
        raise ValueError("KV tensor contains non-finite values (inf or NaN); refusing to quantize")

    orig_shape = tuple(tensor.shape)
    orig_dtype = str(tensor.dtype).removeprefix("torch.")

    x = tensor.detach()
    if axis == "channel":
        # (batch, heads, seq, head_dim) -> (batch, heads, head_dim, seq)
        x = x.transpose(-1, -2)
    x = x.float()

    axis_len = x.shape[-1]
    # Short spans (role-aware mode) or small head dims would otherwise be
    # zero-padded up to a full group -- the padding is stored in the codes,
    # which can inflate a short span far beyond its original size.  Shrinking
    # the group to the axis length keeps the format identical (the effective
    # group size is recorded in the payload metadata) with no fidelity loss.
    group_size = min(group_size, axis_len) if axis_len > 0 else group_size
    num_padded = (-axis_len) % group_size
    if num_padded:
        x = F.pad(x, (0, num_padded))

    groups = x.reshape(-1, group_size)

    # Per-group range with zero included, as in the RTN quantizer.
    zeros = torch.zeros(1, device=groups.device)
    g_max = torch.maximum(groups.amax(dim=-1, keepdim=True), zeros)
    g_min = torch.minimum(groups.amin(dim=-1, keepdim=True), zeros)

    # Handle all-zero groups.
    dead = (g_min == 0) & (g_max == 0)
    g_min[dead] = -1
    g_max[dead] = +1

    q_max = 2**bits - 1
    # Clamp to the float16 storage range *before* the fp16 round-trip: a
    # finite fp32/bf16 input with a per-group range above ~1e6 would round
    # its scale to inf and silently reconstruct as NaN otherwise.
    scale = ((g_max - g_min) / q_max).clamp(min=1e-5, max=_FP16_MAX)
    # Round the scale to its fp16 storage precision *before* encoding so
    # that encode and decode use bit-identical scale/zero values.
    scale = scale.to(torch.float16).float().clamp(min=1e-5)
    zero = torch.round(-g_min / scale)

    q = torch.clamp(torch.round(groups / scale) + zero, 0, q_max)
    codes = pack_codes(q.reshape(-1), bits)

    return PackedQuantizedKV(
        codes=codes,
        scale=scale.to(torch.float16),
        zero=zero.to(torch.float16),
        shape=orig_shape,
        bits=bits,
        group_size=group_size,
        axis=axis,
        num_padded=num_padded,
        dtype=orig_dtype,
    )


def dequantize_packed(pq: PackedQuantizedKV) -> torch.Tensor:
    """Reconstruct a KV-cache tensor from its bit-packed representation.

    Args:
        pq: Quantized tensor produced by :func:`quantize_packed`.

    Returns:
        Dequantized tensor with the original shape and dtype.
    """
    batch, heads, seq, head_dim = pq.shape
    if pq.axis == "channel":
        work_shape = (batch, heads, head_dim, seq)
    else:
        work_shape = (batch, heads, seq, head_dim)

    axis_len = work_shape[-1]
    padded_len = axis_len + pq.num_padded
    num_elements = batch * heads * work_shape[-2] * padded_len

    q = unpack_codes(pq.codes, pq.bits, num_elements).reshape(-1, pq.group_size)
    deq = (q.float() - pq.zero.float()) * pq.scale.float()

    x = deq.reshape(work_shape[:-1] + (padded_len,))
    if pq.num_padded:
        x = x[..., :axis_len]
    if pq.axis == "channel":
        x = x.transpose(-1, -2)

    dtype = getattr(torch, pq.dtype)
    return x.reshape(pq.shape).to(dtype)
