"""Low-level quantization primitives shared by KV codecs.

All helpers operate on tensors of shape ``(..., seq_len, head_dim)`` and use
*simulated* quantization (quantize then keep the dequantized float), matching
the correctness-first research layer described in
``docs/design/kvcomp-architecture.md``.

Quantization axis conventions:

* ``"token"``  : one (scale, zero) per token  -> reduce over ``head_dim`` (last)
* ``"channel"``: one (scale, zero) per channel -> reduce over ``seq_len`` (dim -2)

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Reduction dim for each axis, given the (..., seq_len, head_dim) layout.
_AXIS_TO_REDUCE_DIM = {"token": -1, "channel": -2}


def _reduce_dim(axis: str) -> int:
    if axis not in _AXIS_TO_REDUCE_DIM:
        raise ValueError(f"axis must be 'token' or 'channel', got {axis!r}")
    return _AXIS_TO_REDUCE_DIM[axis]


def _groupwise_minmax(
    x: torch.Tensor, reduce_dim: int, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group min/max along ``reduce_dim``, expanded back to ``x``'s shape.

    The reduce axis is partitioned into contiguous groups of ``group_size``
    (KIVI groups keys along tokens and values along channels).  A non-divisible
    tail forms a final smaller group.  Returns ``(xmin, xmax)`` broadcastable
    elementwise to ``x`` -- this matches the simulated-quant path, which keeps
    the dequantized float rather than packed codes, so per-element scale/zero
    carries no extra storage cost (bit accounting is modelled separately by
    :func:`effective_bits`).
    """
    xm = x.movedim(reduce_dim, -1)
    *lead, length = xm.shape
    n_full = length // group_size
    mins, maxs = [], []
    if n_full > 0:
        head = xm[..., : n_full * group_size].reshape(*lead, n_full, group_size)
        hmin = head.amin(-1, keepdim=True).expand(*lead, n_full, group_size)
        hmax = head.amax(-1, keepdim=True).expand(*lead, n_full, group_size)
        mins.append(hmin.reshape(*lead, n_full * group_size))
        maxs.append(hmax.reshape(*lead, n_full * group_size))
    if length - n_full * group_size > 0:
        tail = xm[..., n_full * group_size :]
        mins.append(tail.amin(-1, keepdim=True).expand_as(tail))
        maxs.append(tail.amax(-1, keepdim=True).expand_as(tail))
    xmin = torch.cat(mins, dim=-1).movedim(-1, reduce_dim)
    xmax = torch.cat(maxs, dim=-1).movedim(-1, reduce_dim)
    return xmin, xmax


def quantize_affine(
    x: torch.Tensor,
    bits: int,
    axis: str = "token",
    symmetric: bool = False,
    group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine (asymmetric/symmetric) integer quantization along ``axis``.

    Args:
        x: Input tensor ``(..., seq_len, head_dim)``.
        bits: Number of integer bits (>= 1).
        axis: ``"token"`` or ``"channel"``.
        symmetric: If True, use a symmetric range centered at zero.
        group_size: If > 0, quantize in contiguous groups of this size along
            the reduce axis (KIVI-style group-wise quantization).  ``<= 0``
            uses a single group spanning the whole axis (legacy behaviour).

    Returns:
        (q, scale, zero): integer codes (float-typed), and broadcastable
        ``scale`` / ``zero`` so that ``dequant = q * scale + zero``.
    """
    reduce_dim = _reduce_dim(axis)
    qmax = (1 << bits) - 1
    grouped = group_size is not None and group_size > 0 and group_size < x.shape[reduce_dim]

    if symmetric:
        if grouped:
            xmin, xmax = _groupwise_minmax(x, reduce_dim, group_size)
            amax = torch.maximum(xmax.abs(), xmin.abs())
        else:
            amax = x.abs().amax(dim=reduce_dim, keepdim=True)
        scale = ((2.0 * amax) / qmax).clamp_min(1e-8)
        zero = -(qmax // 2) * scale  # so that code (qmax//2) maps to ~0
        q = torch.round((x - zero) / scale).clamp_(0, qmax)
    else:
        if grouped:
            xmin, xmax = _groupwise_minmax(x, reduce_dim, group_size)
        else:
            xmin = x.amin(dim=reduce_dim, keepdim=True)
            xmax = x.amax(dim=reduce_dim, keepdim=True)
        scale = ((xmax - xmin) / qmax).clamp_min(1e-8)
        zero = xmin
        q = torch.round((x - zero) / scale).clamp_(0, qmax)
    return q, scale, zero


def dequantize_affine(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_affine`."""
    return q * scale + zero


def fake_quantize_affine(
    x: torch.Tensor,
    bits: int,
    axis: str = "token",
    symmetric: bool = False,
    group_size: int = -1,
) -> torch.Tensor:
    """Quantize then dequantize ``x`` (simulated quantization)."""
    q, scale, zero = quantize_affine(
        x, bits, axis=axis, symmetric=symmetric, group_size=group_size
    )
    return dequantize_affine(q, scale, zero).to(x.dtype)


# --- FP8 (E4M3) -----------------------------------------------------------

# Max representable magnitude of float8_e4m3fn.
_E4M3_MAX = 448.0


def _has_fp8() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def fake_quantize_fp8(x: torch.Tensor, axis: str = "token") -> torch.Tensor:
    """Per-axis FP8 (E4M3) simulated quantization.

    Falls back to 8-bit affine quantization on builds without ``float8_e4m3fn``.
    """
    if not _has_fp8():
        return fake_quantize_affine(x, bits=8, axis=axis, symmetric=True)

    reduce_dim = _reduce_dim(axis)
    amax = x.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(1e-8)
    scale = amax / _E4M3_MAX
    q = (x / scale).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * scale).to(x.dtype)


def effective_bits(bits: int, group_size: int, head_dim: int, fp_scale_bits: int = 16) -> float:
    """Effective bits-per-element including scale/zero overhead.

    Mirrors :func:`onecomp.utils.effective_bits_per_param` on the weight side so
    that KV memory budgets are comparable to weight budgets.
    """
    if group_size is None or group_size <= 0:
        group_size = head_dim
    overhead = (fp_scale_bits + bits) / group_size  # scale (fp) + zero (bits)
    return bits + overhead
