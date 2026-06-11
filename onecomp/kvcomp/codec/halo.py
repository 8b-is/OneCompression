"""HALO: HAdamard-gemm + Least-squares-scale + Online preconditioning KV codec.

HALO is the proposed data-free KV codec.  It shares TurboQuant's skeleton
(orthogonal rotation -> per-vector scale -> Gaussian Lloyd-Max scalar quant) but
wins on the **speed-quality** Pareto against the shipped TurboQuant default
(``rotation_mode="hadamard"`` = butterfly WHT + RMS scale) through two
verified, hardware-robust, storage-neutral levers (see
``ichikawa_works/kvcomp_review_0602/halo_research/`` for the measurements):

1. **Hadamard-GEMM rotation (speed, quality-neutral wrt WHT).**  The randomized
   Hadamard rotation ``(x * sign) @ H`` (``H`` the orthonormal Sylvester matrix)
   is *mathematically identical* to TurboQuant's sequential Walsh-Hadamard
   butterfly, but runs as a **single dense GEMM** -- saturating BLAS / tensor
   cores instead of ``log2(D)`` memory-bound butterfly stages.  Measured ~1.2-1.8x
   faster than the butterfly on CPU (and larger on GPU, where the butterfly's
   sequential stages utilise the device poorly: 0.24ms GEMM vs 0.81ms WHT at
   D=128).  Crucially this keeps the **Hadamard transform's incoherence**, which
   on *real, anisotropic* KV gives lower distortion than a single Haar-random
   *dense* rotation (the earlier "auto->dense" speed trick traded ~15-30% KV-MSE /
   up to ~10% attention-output error away -- a regression this fixes).

2. **Role-asymmetric reconstruction scale (quality, free).**  TurboQuant stores
   the per-vector RMS as the dequant scale.  HALO instead stores a scale chosen
   per *role* (key vs value), because the two roles have *different* quality
   objectives once the codes are fixed.  Write ``x`` for the rotated vector,
   ``c_hat`` for the chosen code direction and ``theta = angle(x, c_hat)``; then

   =========  =========================  ==================  ==============  ====================
   scale       formula                    <x_hat, x>          ||x_hat||       MSE ||x - s c||^2
   =========  =========================  ==================  ==============  ====================
   LS (rms+)  ``<x,c>/<c,c>``            ``||x||^2 cos^2 t``  ``||x|| cos t``  ``||x||^2 sin^2 t`` (min)
   NC         ``||x||/||c||``            ``||x||^2 cos t``    ``||x||`` (exact) ``2||x||^2(1-cos t)``
   IP         ``||x||^2/<x,c>``          ``||x||^2`` (exact)  ``||x||/cos t``   (largest)
   =========  =========================  ==================  ==============  ====================

   with the identity ``s_NC = sqrt(s_LS * s_IP)`` (NC is the geometric mean of
   the MSE-optimal and inner-product-unbiased scales).  Small-angle:
   ``2(1-cos t) = t^2 + O(t^4)`` and ``sin^2 t = t^2 - O(t^4)``, so at 3-4 bit
   (``cos t ~ 0.99``) NC's MSE penalty over LS is ``O(t^4)`` (negligible), while
   the *systematic attention-logit shrink* goes from ``cos^2 t`` (LS) to
   ``cos t`` (NC) -- halving it.  Softmax is sensitive to a per-token logit
   shrink (it raises the temperature and, because ``cos t`` jitters per token,
   perturbs the ranking), so:

   * **keys use NC** (``key_scale_mode="nc"``): preserve the rotated norm
     (``scale = ||x_rot||``, ``c_hat`` normalised), keeping logits faithful;
   * **values use LS** (``value_scale_mode="ls"``): values enter the output
     *linearly* (``sum_i a_i v_i``), so MSE is the correct objective.

   Both store the same single fp16 per vector (storage-neutral).  The
   ``"ls_iter"`` value mode adds one Lloyd alternating-minimisation step
   (re-quantise at the fitted ``s``, then re-fit ``s``) which is monotone
   non-increasing in MSE for a few % extra MSE reduction at 2-3 bit.

This makes HALO dominate the shipped TurboQuant default on *both* the MSE axis
(values) and the attention axis (keys) at identical bit-width and metadata
footprint -- matching upstream vLLM's choice of the ``_nc`` (norm-corrected)
TurboQuant presets as the standard.  The legacy symmetric least-squares scale
(``ls_scale=True`` -> LS on both roles) is retained for backwards comparability.

Optional research levers (OFF by default):

* **Stateless block-wise affine preconditioning (``precond_roles``).**  Real KV
  has massive-activation channels with a large persistent per-channel mean.  HALO
  can remove a per-channel mean ``mu_c`` (and optionally a scale ``w_c``) before
  rotation.  The per-block ``mu``/``w`` are computed *from this block alone* and
  stored in the payload (no cross-block / cross-layer EMA: an earlier EMA keyed
  only on ``(role, head_dim)`` silently mixed the statistics of layer 0 and layer
  31 -- and of successive requests -- through a shared codec instance, and was not
  thread-safe; centering is now exactly stateless).  Mean-centering is PPL-safe
  (attention weights sum to 1, so the added-back mean is exact); whitening lowers
  raw MSE but can amplify attention-output error via the ``1/w_c`` un-whiten on
  high-variance channels, so it stays opt-in.  A multi-architecture sweep (Llama,
  Qwen2, Cohere) did not show a *robust* PPL win from preconditioning, hence OFF
  by default; the scale/rotation levers above are the validated win.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import KVCodec, CompressedKV
from .._registry import register_kv_codec
from .rotation import (
    is_power_of_two,
    random_sign,
    random_orthogonal,
    rotate,
    unrotate,
    signed_hadamard_matrix,
    lloyd_max_gaussian,
    quantize_to_codebook,
)
from ._quant_primitives import fake_quantize_affine
from functools import lru_cache


@lru_cache(maxsize=16)
def _gaussian_codebook(bits: int) -> torch.Tensor:
    return lloyd_max_gaussian(bits)


@lru_cache(maxsize=16)
def _dense_rotation(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _dense_rotation_on(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident orthogonal matrix, cached per ``(dim, seed, device, dtype)``.

    The dense rotation ``Q`` is reused on every encode/decode; materialising it on
    the GPU once (instead of ``_dense_rotation(...).to(device)`` per call, a host->
    device copy each time) removes pure online overhead -- the dominant HALO cost
    at small head dims where ``rotation_mode='auto'`` resolves to dense (see
    tasks/lessons.md K39).
    """
    return _dense_rotation(dim, seed).to(device=torch.device(device_str), dtype=dtype)


@register_kv_codec("halo")
@dataclass
class HALOCodec(KVCodec):
    """Hadamard-GEMM + least-squares-scale + Lloyd-Max scalar KV quantization.

    Attributes:
        bits: Bits per coordinate for keys (and values unless ``value_bits``).
        value_bits: Optional separate bit-width for values (K>V asymmetry).
        ls_scale: Legacy master switch for the dequant scale.  ``True`` -> the
            MSE-optimal least-squares scale on *both* roles (the original HALO
            quality lever); ``False`` -> the per-vector RMS on both roles
            (faithful TurboQuant).  Ignored for a role when its explicit
            ``key_scale_mode`` / ``value_scale_mode`` is set.
        key_scale_mode: Per-role dequant scale for keys (overrides ``ls_scale``):
            ``"nc"`` (norm-correct, recommended -- preserves the attention
            logit), ``"ls"`` (least-squares / MSE-optimal), ``"rms"`` (per-vector
            RMS), ``"ip"`` (inner-product-unbiased) or ``"ls_iter"`` (LS + one
            Lloyd alternating-minimisation step).  ``None`` -> derive from
            ``ls_scale``.
        value_scale_mode: Per-role dequant scale for values (overrides
            ``ls_scale``); same choices.  ``"ls"`` / ``"ls_iter"`` recommended
            (values enter the output linearly, so MSE is the right objective).
        center: Remove a per-block per-channel mean before rotation (research
            lever against massive-activation bias).  Only active when
            ``precond_roles`` is non-empty; has no effect otherwise.  Stateless:
            ``mu`` is computed from the current block and stored in the payload.
        whiten: Divide by a per-block per-channel std before rotation.
        momentum: Deprecated / unused (kept for config back-compat).  Block-wise
            statistics are now exactly stateless (no EMA).
        rotation_mode: ``"auto"`` (Hadamard-GEMM for power-of-two head dims, dense
            QR otherwise), ``"hadamard_gemm"`` (materialised Hadamard, one GEMM),
            ``"hadamard"`` (sequential WHT butterfly) or ``"dense"`` (QR random).
        seed: Seed for the deterministic randomized-Hadamard sign / rotation.
        eps: Whitening floor.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    ls_scale: bool = True
    # Per-role dequant-scale modes (override ``ls_scale`` when set).  Leaving them
    # ``None`` reproduces the legacy symmetric behaviour exactly (so every
    # existing config / result is unchanged); the recommended HALO operating
    # point is the asymmetric ``key_scale_mode="nc"`` + ``value_scale_mode="ls"``.
    key_scale_mode: str = None
    value_scale_mode: str = None
    center: bool = False
    # NB: whitening is OFF by default.  For *values*, un-whitening at decode
    # divides channel c by w_c, which *amplifies* the attention-output error on
    # high-variance channels (the softmax mixes quantization noise, then 1/w_c
    # scales it up) -- this lowers MSE but raises PPL at low bits.  Mean-centering
    # has no such effect (the added-back mean is exact because attention weights
    # sum to 1), so it is the PPL-safe lever.  Enable whiten only to minimise raw
    # reconstruction MSE.
    whiten: bool = False
    # Which roles get the online preconditioner.  IMPORTANT (see module docstring
    # and tasks/lessons.md K18): a multi-architecture sweep (Llama, Qwen2, Cohere)
    # showed preconditioning does NOT robustly improve real-model PPL -- it ties on
    # Llama and *regresses* Cohere at 2-3 bit, because lower value MSE is not lower
    # softmax-attention-output error.  So preconditioning is OFF by default; HALO
    # then equals a faithful (orthogonal, PPL-neutral) TurboQuant.  Opt in
    # (``precond_roles=("v",)``) only for the massive-activation synthetic regime.
    precond_roles: tuple = ()
    momentum: float = 0.9
    # Rotation transform.  "auto" (default) uses the materialised Hadamard GEMM
    # for power-of-two head dims -- the *Hadamard* transform (best incoherence /
    # quality on real anisotropic KV) applied as a single GEMM (best speed) -- and
    # the dense QR random orthogonal for non-power-of-two dims.  "hadamard_gemm"
    # and "dense" force a GEMM transform; "hadamard" forces the (slower, identical
    # quality to hadamard_gemm) sequential WHT butterfly, retained for very large D
    # where materialising the D x D matrix is undesirable.
    rotation_mode: str = "auto"
    # For "auto": fall back from the materialised Hadamard GEMM to the sequential
    # WHT butterfly above this head dim (where the D x D matrix grows large).  No
    # realistic head dim (<=512) reaches it; the matrix is at most ~1 MB at 512.
    hadamard_gemm_max_dim: int = 4096
    seed: int = 0
    eps: float = 1e-6
    key_axis: str = "token"
    value_axis: str = "token"

    _SCALE_MODES = ("rms", "ls", "nc", "ip", "ls_iter")

    def validate_params(self):
        if self.rotation_mode not in ("hadamard", "hadamard_gemm", "dense", "auto"):
            raise ValueError(
                "rotation_mode must be 'auto', 'hadamard_gemm', 'hadamard' or 'dense'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        for mode in (self.key_scale_mode, self.value_scale_mode):
            if mode is not None and mode not in self._SCALE_MODES:
                raise ValueError(
                    f"scale_mode must be one of {self._SCALE_MODES} or None, got {mode!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _scale_mode_for(self, role: str) -> str:
        """Resolve the per-role dequant-scale mode (explicit overrides legacy)."""
        explicit = self.key_scale_mode if role == "k" else self.value_scale_mode
        if explicit is not None:
            return explicit
        return "ls" if self.ls_scale else "rms"

    def _resolved_mode(self, d: int) -> str:
        """Resolve ``"auto"`` to the best speed-quality orthogonal transform.

        Power-of-two dims use the materialised Hadamard GEMM (Hadamard quality at
        GEMM speed); above ``hadamard_gemm_max_dim`` fall back to the butterfly WHT
        (same quality, no D x D matrix); non-power-of-two dims use dense QR.
        """
        if self.rotation_mode != "auto":
            return self.rotation_mode
        if is_power_of_two(d):
            return "hadamard_gemm" if d <= self.hadamard_gemm_max_dim else "hadamard"
        return "dense"

    # --- stateless block-wise affine preconditioner (mean, scale) --------
    def _precondition(self, x: torch.Tensor, role: str):
        """Return per-channel ``(mu, w)`` for *this block* (each ``(d,)`` or None).

        Stateless: ``mu`` / ``w`` are computed from ``x`` alone and stored in the
        payload, so decoding is exact and there is no cross-block / cross-layer /
        cross-request contamination (the previous EMA, keyed only on
        ``(role, head_dim)`` on a codec shared across all layers, mixed those
        statistics and was not thread-safe).  ``None`` is returned for a lever
        that is off so the caller skips the corresponding (identity) op.
        """
        d = x.shape[-1]
        flat = x.float().reshape(-1, d)
        mu = flat.mean(dim=0) if self.center else None
        if self.whiten:
            var = flat.var(dim=0, unbiased=False)
            w = (var + self.eps).rsqrt()
        else:
            w = None
        return mu, w

    # --- rotation helpers -------------------------------------------------
    def _rotate(self, x, m):
        mode = self._resolved_mode(m)
        x = x.float()
        if mode == "hadamard_gemm" and is_power_of_two(m):
            # Folded randomized-Hadamard: (x*sign) @ H == x @ (diag(sign) @ H).
            # Precomputing HS = diag(sign) @ H makes the rotation a single GEMM
            # with no separate elementwise sign pass (~1.15-1.2x faster encode,
            # bit-identical -- verified in kvcomp_halo_research/01).
            hs = signed_hadamard_matrix(m, self.seed, str(x.device), torch.float32)
            return x @ hs, "hg"
        if mode == "hadamard" and is_power_of_two(m):
            sign = random_sign(m, self.seed, device=x.device, dtype=torch.float32)
            return rotate(x, sign), "h"
        q = _dense_rotation_on(m, self.seed, str(x.device), torch.float32)
        return x @ q, "d"

    def _unrotate(self, xr, m, kind):
        if kind == "hg":
            # HS = diag(sign) @ H is orthonormal -> inverse is the transpose.
            hs = signed_hadamard_matrix(m, self.seed, str(xr.device), torch.float32)
            return xr @ hs.t()
        if kind == "h":
            sign = random_sign(m, self.seed, device=xr.device, dtype=torch.float32)
            return unrotate(xr, sign)
        q = _dense_rotation_on(m, self.seed, str(xr.device), torch.float32)
        return xr @ q.t()

    # --- per-role dequant scale ------------------------------------------
    @staticmethod
    def _fit_scale(xr, cb, codes, rms, mode):
        """Return ``(scale, codes)`` for the chosen ``mode`` (codes may update).

        All modes store the same single fp16 per vector, so storage is identical
        regardless of the choice (verified by the storage-neutral tests).
        """
        if mode == "rms":
            return rms, codes
        chat = cb[codes]
        if mode == "nc":
            # Norm-correcting scale: ``||x_hat|| = ||x_rot||`` exactly, so the
            # attention logit ``q.k`` is shrunk only by ``cos(theta)`` (vs
            # ``cos^2`` for LS).  c_hat is the (unnormalised) code direction.
            cn = chat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return xr.norm(dim=-1, keepdim=True) / cn, codes
        # least-squares family: s* = argmin_s ||xr - s*c_hat||^2 = <xr,c>/<c,c>.
        num = (xr * chat).sum(dim=-1, keepdim=True)
        den = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        s_ls = num / den
        if mode == "ls":
            return s_ls, codes
        if mode == "ip":
            # Inner-product-unbiased: <x_hat, x> = ||x||^2 exactly.
            return (xr * xr).sum(dim=-1, keepdim=True) / num.clamp_min(1e-8), codes
        if mode == "ls_iter":
            # One Lloyd alternating-minimisation step: re-quantise at the fitted
            # scale, then re-fit the LS scale.  ||xr - s*c||^2 is monotone
            # non-increasing (the codes-update and the scale-update each minimise
            # the same objective along their coordinate).
            codes2 = quantize_to_codebook(xr / s_ls.clamp_min(1e-8), cb)
            chat2 = cb[codes2]
            num2 = (xr * chat2).sum(dim=-1, keepdim=True)
            den2 = (chat2 * chat2).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return num2 / den2, codes2
        raise ValueError(f"unknown scale mode {mode!r}")

    # --- per-tensor encode / decode --------------------------------------
    def _encode_tensor(self, x, bits, role):
        d = x.shape[-1]
        if self._resolved_mode(d) in ("hadamard", "hadamard_gemm") and not is_power_of_two(d):
            return None  # caller falls back to affine
        # Stateless block-wise affine preconditioning -- only when this role opts
        # in.  When it is off (the default), the (x-mu)*w step is the identity, so
        # we skip it entirely instead of paying a full-tensor subtract-0 /
        # multiply-1; this makes HALO's overhead equal to the bare rotated
        # Lloyd-Max path.
        do_precond = role in self.precond_roles and (self.center or self.whiten)
        if do_precond:
            mu, w = self._precondition(x, role)      # per-channel (mean, scale)
            xw = x.float()
            if mu is not None:
                xw = xw - mu
            if w is not None:
                xw = xw * w
        else:
            mu, w, xw = None, None, x
        xr, kind = self._rotate(xw, d)               # orthogonal rotation
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)  # per-vector RMS
        cb = _gaussian_codebook(bits).to(x.device)
        codes = quantize_to_codebook(xr / rms, cb)
        scale, codes = self._fit_scale(xr, cb, codes, rms, self._scale_mode_for(role))
        return {"codes": codes, "scale": scale, "mu": mu, "w": w,
                "bits": bits, "m": d, "kind": kind}

    def _decode_tensor(self, p, dtype):
        cb = _gaussian_codebook(p["bits"]).to(p["codes"].device)
        xr = cb[p["codes"]] * p["scale"]
        xw = self._unrotate(xr, p["m"], p["kind"])
        if p["w"] is not None:
            xw = xw / p["w"]                          # un-whiten
        if p["mu"] is not None:
            xw = xw + p["mu"]                         # un-center
        return xw.to(dtype)

    def _stored_bits(self, x, bits, precond=False):
        b, h, s, d = x.shape
        vecs = b * h * s
        # codes + per-vec scale (+ per-block (mu,w) only when preconditioning on)
        return x.numel() * bits + vecs * 16 + (d * 32 if precond else 0)

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kp = self._encode_tensor(key, self.bits, "k")
        vp = self._encode_tensor(value, self._value_bits(), "v")
        if kp is None or vp is None:
            kq = fake_quantize_affine(key, self.bits, axis="token")
            vq = fake_quantize_affine(value, self._value_bits(), axis="token")
            stored = key.numel() * self.bits + value.numel() * self._value_bits()
            return CompressedKV(key_q=kq, value_q=vq,
                                meta={"dtype": key.dtype, "fallback": True,
                                      "stored_bits": int(stored), "passthrough": True})
        kpre = "k" in self.precond_roles and (self.center or self.whiten)
        vpre = "v" in self.precond_roles and (self.center or self.whiten)
        stored = (self._stored_bits(key, self.bits, kpre)
                  + self._stored_bits(value, self._value_bits(), vpre))
        return CompressedKV(key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)})

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        if compressed.meta.get("fallback") or compressed.meta.get("passthrough"):
            return compressed.key_q.to(dtype), compressed.value_q.to(dtype)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(compressed.value_q, dtype)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update({"value_bits": self.value_bits, "ls_scale": self.ls_scale,
                    "key_scale_mode": self.key_scale_mode,
                    "value_scale_mode": self.value_scale_mode,
                    "center": self.center, "whiten": self.whiten,
                    "precond_roles": list(self.precond_roles),
                    "momentum": self.momentum, "rotation_mode": self.rotation_mode,
                    "hadamard_gemm_max_dim": self.hadamard_gemm_max_dim})
        return cfg
