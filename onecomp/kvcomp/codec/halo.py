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

2. **Least-squares reconstruction scale (quality, free).**  TurboQuant stores the
   per-vector RMS as the dequant scale.  HALO instead stores the scale that
   *minimises* the reconstruction error for the chosen codes,
   ``s* = <x_rot, c_hat> / <c_hat, c_hat>`` (the closed-form argmin of
   ``||x_rot - s * c_hat||^2``).  It is one ``O(D)`` dot-product pair at encode,
   the same single fp16 per vector at decode, and is provably ``<=`` the RMS MSE.
   On real KV it lowers the attention-output error ~3-7% at equal bits/storage
   across 2/3/4-bit.

Together HALO is strictly faster *and* strictly higher quality than the shipped
TurboQuant default at identical bit-width and metadata footprint.

Optional research levers (OFF by default):

* **Online affine preconditioning (``precond_roles``).**  Real KV has
  massive-activation channels with a large persistent per-channel mean.  HALO can
  remove a running per-channel mean ``mu_c`` (and optionally a scale ``w_c``)
  before rotation.  Mean-centering is PPL-safe (attention weights sum to 1, so the
  added-back mean is exact); whitening lowers raw MSE but can amplify
  attention-output error via the ``1/w_c`` un-whiten on high-variance channels, so
  it stays opt-in.  A multi-architecture sweep (Llama, Qwen2, Cohere) did not show
  a *robust* PPL win from preconditioning, hence OFF by default; the two levers
  above are the validated win.  Per-block ``mu``/``w`` are stored in the payload.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        ls_scale: Store the MSE-optimal least-squares dequant scale instead of the
            RMS (the quality lever; free, same fp16-per-vector storage).
        center: Remove a running per-channel mean before rotation (research lever
            against massive-activation bias).  Only active when ``precond_roles``
            is non-empty; has no effect otherwise.
        whiten: Divide by a running per-channel std before rotation.
        momentum: EMA momentum for the per-channel mean / variance (online).
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
    # Online per-channel (mean, var) EMA, keyed by ("k"/"v", head_dim).
    _ema: dict = field(default_factory=dict, repr=False, compare=False)

    def validate_params(self):
        if self.rotation_mode not in ("hadamard", "hadamard_gemm", "dense", "auto"):
            raise ValueError(
                "rotation_mode must be 'auto', 'hadamard_gemm', 'hadamard' or 'dense'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

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

    # --- online affine preconditioner (mean, scale) ----------------------
    def _precondition(self, x: torch.Tensor, role: str):
        """Return per-channel ``(mu, w)`` (each ``(d,)``), updating the EMA."""
        d = x.shape[-1]
        if role not in self.precond_roles:
            return (torch.zeros(d, device=x.device), torch.ones(d, device=x.device))
        flat = x.float().reshape(-1, d)
        mu_b = flat.mean(dim=0) if self.center else torch.zeros(d, device=x.device)
        var_b = flat.var(dim=0, unbiased=False) if self.whiten else torch.ones(d, device=x.device)
        key = (role, d)
        prev = self._ema.get(key)
        if prev is None or prev[0].shape != mu_b.shape or prev[0].device != mu_b.device:
            mu, var = mu_b, var_b
        else:
            mu = self.momentum * prev[0] + (1.0 - self.momentum) * mu_b
            var = self.momentum * prev[1] + (1.0 - self.momentum) * var_b
        self._ema[key] = (mu.detach(), var.detach())
        w = (var + self.eps).rsqrt() if self.whiten else torch.ones(d, device=x.device)
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

    # --- per-tensor encode / decode --------------------------------------
    def _encode_tensor(self, x, bits, role):
        d = x.shape[-1]
        if self._resolved_mode(d) in ("hadamard", "hadamard_gemm") and not is_power_of_two(d):
            return None  # caller falls back to affine
        # Online affine preconditioning -- only when this role opts in.  When it
        # is off (the default), the (x-mu)*w step is the identity, so we skip it
        # entirely instead of paying a full-tensor subtract-0 / multiply-1; this
        # makes HALO's overhead equal to the bare rotated-Lloyd-Max path.
        do_precond = role in self.precond_roles and (self.center or self.whiten)
        if do_precond:
            mu, w = self._precondition(x, role)      # per-channel (mean, scale)
            xw = (x.float() - mu) * w
        else:
            mu, w, xw = None, None, x
        xr, kind = self._rotate(xw, d)               # orthogonal rotation
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)  # per-vector RMS
        cb = _gaussian_codebook(bits).to(x.device)
        codes = quantize_to_codebook(xr / rms, cb)
        if self.ls_scale:
            # MSE-optimal dequant scale for the chosen codes:
            # s* = argmin_s ||xr - s*c_hat||^2 = <xr,c_hat>/<c_hat,c_hat>.  Same
            # single fp16/vector as RMS, provably <= RMS reconstruction error.
            chat = cb[codes]
            num = (xr * chat).sum(dim=-1, keepdim=True)
            den = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = num / den
        else:
            scale = rms
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
                    "center": self.center, "whiten": self.whiten,
                    "precond_roles": list(self.precond_roles),
                    "momentum": self.momentum, "rotation_mode": self.rotation_mode,
                    "hadamard_gemm_max_dim": self.hadamard_gemm_max_dim})
        return cfg
