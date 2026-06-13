"""TurboQuant: data-free rotated scalar KV quantization (arXiv:2504.19874).

TurboQuant rotates each KV vector by a random orthogonal transform so its
coordinates follow a concentrated ``Beta((d-1)/2, (d-1)/2)`` law (Lemma 1),
then quantizes each coordinate with a scalar **Lloyd-Max** quantizer whose
codebook is computed *analytically* from that law -- i.e. **data-free** (no
calibration).  It is an *online* codec: each new KV vector is rotated and
quantized on the fly at decode time, so it lives in ``codec/`` with
``flag_calibration=False`` (OneComp's RTN-vs-GPTQ convention: the flag, not the
folder, marks calibration).

Faithful options
----------------
* ``dist``: ``"gaussian"`` (the high-dimensional limit, normalise by RMS, fast
  default) or ``"beta"`` (the exact ``Beta((d-1)/2)`` codebook, normalise by the
  per-vector L2 norm).
* ``n_outlier`` / ``outlier_bits``: the paper's KV mixed-precision -- split the
  ``n_outlier`` highest-energy channels off and quantize them at
  ``outlier_bits`` while the rest use ``bits``.  The effective payload bit-width
  is ``(n_outlier*outlier_bits + (d-n_outlier)*bits) / d``; for ``head_dim=128``,
  ``64 ch @ 3 + 64 @ 2`` = ``2.5`` bit and ``64 @ 4 + 64 @ 3`` = ``3.5`` bit
  (``32 @ 3 + 96 @ 2`` = ``2.25``, ``32 @ 4 + 96 @ 3`` = ``3.25``).  The
  outlier channel set is chosen *per encoded block* from that block's energy, so
  under incremental decode (``q_len=1``) the channel mask is dynamic per step.
  This is exact for the fake-quant research layer but is a layout a real kernel
  cannot express statically -- the paper must state this per-block granularity
  (memo A-8).
* ``use_qjl``: a 1-bit residual (QJL-style) that refines the reconstruction
  toward the unbiased inner-product estimator (off by default; community
  evaluations agree it can harm softmax attention).

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

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
    lloyd_max_beta,
    quantize_to_codebook,
)
from ._quant_primitives import fake_quantize_affine


@lru_cache(maxsize=16)
def _codebook(bits: int, dist: str, d: int) -> torch.Tensor:
    if dist == "beta":
        return lloyd_max_beta(bits, d)
    return lloyd_max_gaussian(bits)


@lru_cache(maxsize=16)
def _ortho(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _ortho_on(dim: int, seed: int, device_str: str) -> torch.Tensor:
    """Device-resident orthogonal matrix (avoids a host->device copy per call)."""
    return _ortho(dim, seed).to(torch.device(device_str))


@register_kv_codec("turboquant")
@dataclass
class TurboQuantCodec(KVCodec):
    """Random-rotation + Lloyd-Max scalar quantization of the KV cache.

    Attributes:
        bits: Bits per coordinate (key; value shares unless ``value_bits`` set).
        value_bits: Optional separate bit-width for values.
        dist: ``"gaussian"`` (RMS-normalised, high-d limit) or ``"beta"``
            (L2-normalised, exact Beta codebook).
        n_outlier: Number of highest-energy channels quantized at ``outlier_bits``.
        outlier_bits: Bit-width for the outlier channels.
        use_qjl: Enable the 1-bit QJL-style residual refinement.
        rotation_mode: ``"hadamard"`` (fast WHT, power-of-two dims) or ``"dense"``
            (QR random orthogonal, any dim).
        seed: Seed for the deterministic rotations.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    dist: str = "gaussian"
    n_outlier: int = 0
    outlier_bits: int = None
    use_qjl: bool = False
    rotation_mode: str = "hadamard"
    seed: int = 0
    key_axis: str = "token"
    value_axis: str = "token"
    def validate_params(self):
        if self.rotation_mode not in ("hadamard", "dense"):
            raise ValueError("rotation_mode must be 'hadamard' or 'dense'")
        if self.dist not in ("gaussian", "beta"):
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        if not (isinstance(self.n_outlier, int) and self.n_outlier >= 0):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _outlier_bits(self) -> int:
        return self.outlier_bits if self.outlier_bits is not None else self.bits + 1

    # --- single-subset rotate + Lloyd-Max ---------------------------------
    def _rotate(self, x, m, seed):
        if self.rotation_mode == "hadamard" and is_power_of_two(m):
            sign = random_sign(m, seed, device=x.device, dtype=torch.float32)
            return rotate(x.float(), sign), ("h", seed)
        q = _ortho_on(m, seed, str(x.device))
        return x.float() @ q, ("d", seed)

    def _unrotate(self, xr, m, rot):
        kind, seed = rot
        if kind == "h":
            sign = random_sign(m, seed, device=xr.device, dtype=torch.float32)
            return unrotate(xr, sign)
        q = _ortho_on(m, seed, str(xr.device))
        return xr @ q.t()

    def _quant_subset(self, x, bits, seed):
        """Rotate ``(..., m)``, normalise per-vector, quantize to the codebook."""
        m = x.shape[-1]
        xr, rot = self._rotate(x, m, seed)
        if self.dist == "beta":
            scale = xr.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # L2 norm gamma
        else:
            scale = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)  # RMS
        z = xr / scale
        cb = _codebook(bits, self.dist, m).to(x.device)
        codes = quantize_to_codebook(z, cb)
        payload = {"codes": codes, "scale": scale, "bits": bits, "m": m,
                   "rot": rot, "dist": self.dist}
        if self.use_qjl:
            recon = cb.to(x.device)[codes] * scale
            resid = xr - recon
            payload["qjl_sign"] = (resid >= 0)
            payload["qjl_rms"] = resid.abs().mean(dim=-1, keepdim=True)
        return payload

    def _dequant_subset(self, p, device, dtype):
        cb = _codebook(p["bits"], p["dist"], p["m"]).to(device)
        xr = cb[p["codes"]] * p["scale"]
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(xr.dtype)
            xr = xr + sign * p["qjl_rms"]  # 1-bit residual refinement
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode (handles outlier split) ----------------------
    def _encode_tensor(self, x, bits):
        head_dim = x.shape[-1]
        if self.rotation_mode == "hadamard" and not is_power_of_two(head_dim) and self.n_outlier == 0:
            return None  # graceful affine fallback handled by caller
        if self.n_outlier <= 0 or self.n_outlier >= head_dim:
            return {"mode": "single", "main": self._quant_subset(x, bits, self.seed),
                    "head_dim": head_dim}
        # Outlier mixed-precision: top-n_outlier channels by energy over tokens.
        energy = x.float().abs().amax(dim=-2)  # (..., head_dim) per (b,h)
        energy = energy.reshape(-1, head_dim).mean(dim=0)
        out_idx = torch.argsort(energy, descending=True)[: self.n_outlier]
        out_idx, _ = torch.sort(out_idx)
        mask = torch.zeros(head_dim, dtype=torch.bool, device=x.device)
        mask[out_idx] = True
        x_out = x[..., mask]
        x_in = x[..., ~mask]
        return {
            "mode": "outlier", "head_dim": head_dim, "mask": mask,
            "out": self._quant_subset(x_out, self._outlier_bits(), self.seed + 1),
            "in": self._quant_subset(x_in, bits, self.seed + 2),
        }

    def _decode_tensor(self, payload, dtype):
        device = payload["main"]["codes"].device if payload["mode"] == "single" \
            else payload["out"]["codes"].device
        if payload["mode"] == "single":
            return self._dequant_subset(payload["main"], device, dtype).to(dtype)
        head_dim = payload["head_dim"]
        mask = payload["mask"]
        x_out = self._dequant_subset(payload["out"], device, dtype)
        x_in = self._dequant_subset(payload["in"], device, dtype)
        shape = list(x_out.shape[:-1]) + [head_dim]
        x = torch.empty(shape, device=device, dtype=x_out.dtype)
        x[..., mask] = x_out
        x[..., ~mask] = x_in
        return x.to(dtype)

    def _stored_bits(self, x, bits):
        b, h, s, d = x.shape
        vecs = b * h * s
        if self.n_outlier <= 0 or self.n_outlier >= d:
            payload_bits = x.numel() * bits
            scale_bits = vecs * 16
        else:
            no = self.n_outlier
            payload_bits = vecs * (no * self._outlier_bits() + (d - no) * bits)
            scale_bits = vecs * 2 * 16  # two per-vector norms
        # QJL stores a 1-bit sign per element AND a per-vector fp16 residual RMS
        # (``qjl_rms``); count both for honest whole-cache accounting (memo A-7).
        qjl_bits = (x.numel() * 1 + vecs * 16) if self.use_qjl else 0
        return payload_bits + scale_bits + qjl_bits

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kp = self._encode_tensor(key, self.bits)
        vp = self._encode_tensor(value, self._value_bits())
        if kp is None or vp is None:
            kq = fake_quantize_affine(key, self.bits, axis="token")
            vq = fake_quantize_affine(value, self._value_bits(), axis="token")
            # Count the per-token affine scale + zero (two fp16 per vector), like
            # the AffineKVCodec path, so the fallback's compression ratio is not
            # over-stated (memo A-6).
            kvecs = key.shape[0] * key.shape[1] * key.shape[2]
            vvecs = value.shape[0] * value.shape[1] * value.shape[2]
            stored = (key.numel() * self.bits + value.numel() * self._value_bits()
                      + (kvecs + vvecs) * 2 * 16)
            return CompressedKV(key_q=kq, value_q=vq,
                                meta={"dtype": key.dtype, "fallback": True,
                                      "stored_bits": int(stored), "passthrough": True})
        stored = self._stored_bits(key, self.bits) + self._stored_bits(value, self._value_bits())
        return CompressedKV(key_q=kp, value_q=vp,
                            meta={"dtype": key.dtype, "stored_bits": int(stored)})

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        if compressed.meta.get("fallback") or compressed.meta.get("passthrough"):
            return compressed.key_q.to(dtype), compressed.value_q.to(dtype)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(compressed.value_q, dtype)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update({"dist": self.dist, "n_outlier": self.n_outlier,
                    "outlier_bits": self.outlier_bits, "use_qjl": self.use_qjl,
                    "rotation_mode": self.rotation_mode, "value_bits": self.value_bits})
        return cfg
