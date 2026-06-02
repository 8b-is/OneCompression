"""HALO: HAdamard + onLine-whitening data-free KV quantization.

STATUS (honest, multi-architecture verdict -- see tasks/lessons.md K18).  HALO
was proposed as a TurboQuant improvement via (1) a faster Hadamard rotation and
(2) online affine preconditioning.  A multi-architecture sweep (Llama-3.1-8B,
DeepSeek-Qwen2-1.5B, Cohere/takane-32B) did NOT confirm a win:

* **Overhead**: at realistic head dims (D=128-256) the dense matmul rotation is
  *faster* than the WHT on GPU (0.24ms vs 0.81ms/encode at D=128); the Hadamard
  advantage only appears at very large D or on non-matmul hardware.
* **Real-KV distortion**: HALO's value MSE is within ~1% of TurboQuant -- the
  massive-activation premise that motivated preconditioning does not hold for
  real post-attention values on these models.
* **PPL**: HALO ties TurboQuant on Llama but *regresses* on Cohere at 2-3 bit,
  because lower value MSE is not the same as lower softmax-attention-output error.

Therefore the shipped default has preconditioning OFF (HALO == a faithful,
PPL-neutral TurboQuant) and HALO is retained as a *research* codec for ablations.
The clearly-validated proposed method in this repo is **TaperKV** (graded KV
quantization), which beats eviction at iso-memory on long-context retrieval
across Llama and Cohere.  The mechanism below is kept as documented opt-in levers.

Original design intent (improve TurboQuant on overhead and anisotropic KV):

1. **Fast rotation.**  TurboQuant's faithful rotation is a *dense* random
   orthogonal matrix (``O(D^2)`` matmul per token).  HALO uses the *randomized
   Walsh-Hadamard transform* (``diag(+/-1)`` then WHT, ``O(D log D)``), which is
   provably as incoherent as a Haar rotation but far cheaper.

2. **Online affine preconditioning (mean-center + whiten).**  Real KV has
   *massive-activation* channels: a few channels carry a large, persistent
   *non-zero mean* (a constant bias) and large variance.  TurboQuant rotates
   assuming a zero-mean vector, so a per-token unit-norm rotation spends most of
   its dynamic range encoding that constant bias instead of the informative
   fluctuations.  HALO first removes a running per-channel **mean** ``mu_c`` and
   divides by a running per-channel **scale** ``w_c = 1/sqrt(Var[x_c] + eps)``
   (both tracked by a cheap ``O(D)`` EMA across decode steps) *before* the
   randomized Hadamard rotation.  The residual ``(x - mu) * w`` is centred and
   isotropic, so the rotated coordinates are close to ``N(0, 1)`` and the
   data-free Gaussian Lloyd-Max codebook is near-optimal.

   The mean removal is the key lever: for a channel with mean ``mu_c`` and
   fluctuation std ``s_c << |mu_c|`` (the massive-activation regime), a zero-mean
   rotated quantizer has relative error ``~ const`` while HALO encodes only the
   fluctuation -> error ``~ s_c / |mu_c|`` smaller.  On zero-mean isotropic data
   the preconditioner is the identity, so ``HALO >= TurboQuant`` always.

The per-block ``mu`` and ``w`` are stored in that block's payload (``2D`` floats,
negligible), so ``decode`` is self-contained and needs no state.

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
    """Hadamard + online-whitening + Lloyd-Max scalar KV quantization.

    Attributes:
        bits: Bits per coordinate for keys (and values unless ``value_bits``).
        value_bits: Optional separate bit-width for values (K>V asymmetry).
        center: Remove a running per-channel mean before rotation (the key lever
            against massive-activation bias).
        whiten: Divide by a running per-channel std before rotation.
            ``center=False, whiten=False`` recovers a TurboQuant-style fixed
            randomized-Hadamard codec.
        momentum: EMA momentum for the per-channel mean / variance (online).
        rotation_mode: ``"hadamard"`` (fast WHT) or ``"dense"`` (QR random).
        seed: Seed for the deterministic randomized-Hadamard sign / rotation.
        eps: Whitening floor.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    center: bool = True
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
    # Rotation transform (all orthogonal -> identical quality; this is purely a
    # *speed* lever).  "auto" (default) picks the cheapest orthogonal transform
    # for the given head dim: a dense O(D^2) cuBLAS matmul for small D (where a
    # single GEMM beats the log2(D) sequential Hadamard butterfly stages), and
    # the cached fast Walsh-Hadamard O(D log D) transform for large power-of-two
    # D (e.g. 256), where it overtakes the matmul.  "dense"/"hadamard" force one.
    rotation_mode: str = "auto"
    # Benchmarks (DGX GPU, D up to 512) show a single dense cuBLAS GEMM beats the
    # log2(D) sequential Hadamard butterfly at every realistic head dim, so
    # "auto" uses dense unless D is a power of two AND >= this (large) threshold.
    hadamard_min_dim: int = 1024
    seed: int = 0
    eps: float = 1e-6
    key_axis: str = "token"
    value_axis: str = "token"
    # Online per-channel (mean, var) EMA, keyed by ("k"/"v", head_dim).
    _ema: dict = field(default_factory=dict, repr=False, compare=False)

    def validate_params(self):
        if self.rotation_mode not in ("hadamard", "dense", "auto"):
            raise ValueError("rotation_mode must be 'hadamard', 'dense' or 'auto'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _resolved_mode(self, d: int) -> str:
        """Resolve ``"auto"`` to the cheapest orthogonal transform for dim ``d``."""
        if self.rotation_mode != "auto":
            return self.rotation_mode
        return "hadamard" if (is_power_of_two(d) and d >= self.hadamard_min_dim) else "dense"

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
        if self._resolved_mode(m) == "hadamard" and is_power_of_two(m):
            sign = random_sign(m, self.seed, device=x.device, dtype=torch.float32)
            return rotate(x.float(), sign), "h"
        q = _dense_rotation_on(m, self.seed, str(x.device), torch.float32)
        return x.float() @ q, "d"

    def _unrotate(self, xr, m, kind):
        if kind == "h":
            sign = random_sign(m, self.seed, device=xr.device, dtype=torch.float32)
            return unrotate(xr, sign)
        q = _dense_rotation_on(m, self.seed, str(xr.device), torch.float32)
        return xr @ q.t()

    # --- per-tensor encode / decode --------------------------------------
    def _encode_tensor(self, x, bits, role):
        d = x.shape[-1]
        if self._resolved_mode(d) == "hadamard" and not is_power_of_two(d):
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
        scale = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)  # per-vector RMS
        z = xr / scale
        cb = _gaussian_codebook(bits).to(x.device)
        codes = quantize_to_codebook(z, cb)
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
        cfg.update({"value_bits": self.value_bits, "center": self.center,
                    "whiten": self.whiten, "precond_roles": list(self.precond_roles),
                    "momentum": self.momentum, "rotation_mode": self.rotation_mode,
                    "hadamard_min_dim": self.hadamard_min_dim})
        return cfg
