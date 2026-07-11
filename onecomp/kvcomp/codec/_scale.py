"""Role-asymmetric (attention-aligned) dequantization scale.

Used by every rotated-codebook KV codec in this package (currently
:class:`onecomp.kvcomp.codec.hayatequant.HayateQuantCodec`, which subsumes
TurboQuant and the historical HALO codec as special cases -- see that
module's docstring): for a rotated, per-vector-normalised vector ``x`` and
its current reconstructed **direction** ``chat`` (the concatenation of
per-coordinate Lloyd-Max centroids for a scalar codec, or of block-VQ
codewords for HayateQuant -- either way, a single ``(..., m)`` tensor with
the same shape as ``x``), the choice of dequantization *scale* ``s`` in the
reconstruction ``x_hat = s * chat`` should depend on how the reconstructed
vector is consumed downstream:

* **Keys** enter the attention logit through an inner product ``q . k_hat``.
  The norm-correcting (``"nc"``) scale preserves ``||k_hat|| = ||k||``
  exactly, so the logit is shrunk only by ``cos(theta)`` (``theta`` = the
  angle between ``x`` and ``chat``) instead of ``cos^2(theta)`` for the naive
  least-squares scale -- a systematic, per-token-varying softmax-temperature
  and ranking perturbation that the least-squares scale does not correct.
* **Values** enter the attention output *linearly* (``sum_i a_i v_i``); the
  least-squares (``"ls"``) scale is the exact MSE minimiser for that linear
  consumption, which is the correct objective. ``"ls_iter"`` adds one Lloyd
  alternating-minimisation refinement (re-quantize at the fitted scale, then
  re-fit the scale), which is monotone non-increasing in MSE.

Identity: ``s_nc = sqrt(s_ls * s_ip)`` (NC is the geometric mean of the
MSE-optimal and inner-product-unbiased scales). Small-angle expansion
(``2(1-cos theta) = theta^2 + O(theta^4)``, ``sin^2 theta = theta^2 -
O(theta^4)``) shows the NC MSE penalty over LS is ``O(theta^4)`` (negligible
at 3-4 bit) while the systematic logit shrink halves (``cos^2`` -> ``cos``).
Every mode stores exactly one scalar (fp16) per encoded vector, so the choice
of mode is **storage-neutral** and independent of ``block_size``.

BlockQuant terminology maps directly onto the existing modes: ``"bsm"`` is
the best-scalar MSE scale (HayateQuant ``"ls"``), and ``"ub"`` is the
inner-product-unbiased ratio estimator (HayateQuant ``"ip"``).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

SCALE_MODES = ("rms", "nc", "ls", "ip", "ls_iter", "bsm", "ub")


def resolve_scale_mode(explicit: Optional[str], ls_scale: bool, role: str) -> str:
    """Resolve the effective scale mode for ``role`` (``"k"`` or ``"v"``).

    ``explicit`` (the codec's ``key_scale_mode`` / ``value_scale_mode``) wins
    when set; otherwise falls back to the ``ls_scale`` smart default: NC for
    keys / ``ls_iter`` for values when ``True`` (the validated best
    operating point), plain per-vector RMS (TurboQuant-faithful, no scale
    refinement) for both roles when ``False``.
    """
    if explicit is not None:
        if explicit not in SCALE_MODES:
            raise ValueError(f"unknown scale mode {explicit!r}")
        return explicit
    if not ls_scale:
        return "rms"
    return "nc" if role == "k" else "ls_iter"


def fit_scale(
    xr: torch.Tensor,
    chat: torch.Tensor,
    mode: str,
    rms: torch.Tensor,
    requantize_fn: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    """Return ``(scale, new_codes_or_None)`` for the chosen role-asymmetric ``mode``.

    Dimension-agnostic: ``xr`` and ``chat`` need only share their trailing
    (reduction) dimension, so the same function serves a scalar codebook's
    concatenated per-coordinate centroids and HayateQuant's concatenated
    block-VQ codewords identically.

    Args:
        xr: Rotated, pre-quantization vector(s), ``(..., m)``.
        chat: Current reconstructed direction (codebook lookup already
            performed by the caller), ``(..., m)``.
        mode: One of :data:`SCALE_MODES`.
        rms: Precomputed per-vector RMS scale, ``(..., 1)``; returned as-is
            for ``mode="rms"``.
        requantize_fn: Required for ``mode="ls_iter"``. Callable
            ``z -> (chat2, codes2)`` that re-quantizes the rescaled vector
            ``z = xr / s_ls`` and returns both the new reconstructed
            direction and the new codes (so the caller can update its stored
            codes to match the refined scale).

    Returns:
        ``(scale, new_codes)``: ``scale`` broadcasts against ``xr`` along its
        last dim; ``new_codes`` is ``None`` unless ``mode="ls_iter"`` updated
        the codes.
    """
    if mode == "bsm":
        mode = "ls"
    elif mode == "ub":
        mode = "ip"
    if mode == "rms":
        return rms, None
    if mode == "nc":
        cn = chat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return xr.norm(dim=-1, keepdim=True) / cn, None
    num = (xr * chat).sum(dim=-1, keepdim=True)
    den = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    s_ls = num / den
    if mode == "ls":
        return s_ls, None
    if mode == "ip":
        # Inner-product-unbiased: <x_hat, x> = ||x||^2 exactly. Preserve the
        # sign of num (clamp_min alone would flip a negative inner product to
        # a large positive scale).
        num_safe = num.sign() * num.abs().clamp_min(1e-8)
        return (xr * xr).sum(dim=-1, keepdim=True) / num_safe, None
    if mode == "ls_iter":
        if requantize_fn is None:
            raise ValueError("mode='ls_iter' requires a requantize_fn")
        chat2, codes2 = requantize_fn(xr / s_ls.clamp_min(1e-8))
        num2 = (xr * chat2).sum(dim=-1, keepdim=True)
        den2 = (chat2 * chat2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return num2 / den2, codes2
    raise ValueError(f"unknown scale mode {mode!r}")
