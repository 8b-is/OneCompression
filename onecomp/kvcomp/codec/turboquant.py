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
* ``use_qjl``: TurboQuant-prod's 1-bit QJL residual sketch for keys by default.
  The scalar MSE reconstruction is augmented with
  ``sqrt(pi/2)/m * ||r||_2 * S^T sign(S r)``, which makes inner products with
  the reconstructed residual unbiased over the Gaussian sketch.  To compare at
  an equal payload budget, instantiate it with ``bits=target_bits-1``; the
  residual norm is still counted explicitly in ``stored_bits``.

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


@lru_cache(maxsize=32)
def _qjl_projection(dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(int(dim), int(dim), generator=gen, dtype=torch.float32)


@lru_cache(maxsize=64)
def _qjl_projection_on(dim: int, seed: int, device_str: str) -> torch.Tensor:
    return _qjl_projection(int(dim), int(seed)).to(torch.device(device_str))


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
        use_qjl: Enable TurboQuant-prod's 1-bit Gaussian QJL residual sketch.
        qjl_roles: Which roles receive QJL when ``use_qjl`` is true:
            ``"key"`` (TurboQuant-prod default), ``"value"``, or ``"both"``.
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
    qjl_roles: str = "key"
    rotation_mode: str = "hadamard"
    seed: int = 0
    key_axis: str = "token"
    value_axis: str = "token"
    def validate_params(self):
        if self.rotation_mode not in ("hadamard", "dense"):
            raise ValueError("rotation_mode must be 'hadamard' or 'dense'")
        if self.dist not in ("gaussian", "beta"):
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if self.qjl_roles not in ("key", "value", "both"):
            raise ValueError("qjl_roles must be 'key', 'value', or 'both'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        if not (isinstance(self.n_outlier, int) and self.n_outlier >= 0):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _outlier_bits(self) -> int:
        return self.outlier_bits if self.outlier_bits is not None else self.bits + 1

    def _qjl_seed(self, seed: int) -> int:
        return int(seed) + 104729

    def _qjl_enabled(self, role: str) -> bool:
        return bool(self.use_qjl and self.qjl_roles in (role, "both"))

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

    def _quant_subset(self, x, bits, seed, use_qjl=False):
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
        if use_qjl:
            recon = cb.to(x.device)[codes] * scale
            resid = xr - recon
            qjl_seed = self._qjl_seed(seed)
            proj = _qjl_projection_on(m, qjl_seed, str(x.device))
            z = resid.float().reshape(-1, m) @ proj.t()
            payload["qjl_sign"] = (z >= 0).reshape_as(resid)
            payload["qjl_norm"] = resid.float().norm(dim=-1, keepdim=True)
            payload["qjl_seed"] = qjl_seed
        return payload

    def _dequant_subset(self, p, device, dtype):
        cb = _codebook(p["bits"], p["dist"], p["m"]).to(device)
        xr = cb[p["codes"]] * p["scale"]
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(torch.float32)
            if "qjl_norm" in p:
                proj = _qjl_projection_on(p["m"], p["qjl_seed"], str(device))
                qjl = (sign.reshape(-1, p["m"]) @ proj).reshape_as(xr)
                qjl = qjl * (math.sqrt(math.pi / 2.0) / float(p["m"]))
                xr = xr.float() + qjl * p["qjl_norm"].float()
            else:  # legacy payloads written by the pre-QJL-sketch prototype
                xr = xr + sign.to(xr.dtype) * p["qjl_rms"]
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode (handles outlier split) ----------------------
    def _encode_tensor(self, x, bits, use_qjl=False):
        head_dim = x.shape[-1]
        if self.rotation_mode == "hadamard" and not is_power_of_two(head_dim) and self.n_outlier == 0:
            return None  # graceful affine fallback handled by caller
        if self.n_outlier <= 0 or self.n_outlier >= head_dim:
            return {"mode": "single", "main": self._quant_subset(x, bits, self.seed, use_qjl),
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
            "out": self._quant_subset(x_out, self._outlier_bits(), self.seed + 1, use_qjl),
            "in": self._quant_subset(x_in, bits, self.seed + 2, use_qjl),
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

    def _stored_bits(self, x, bits, use_qjl=False):
        b, h, s, d = x.shape
        vecs = b * h * s
        split = 0 < self.n_outlier < d
        if not split:
            payload_bits = x.numel() * bits
            scale_bits = vecs * 16
        else:
            no = self.n_outlier
            payload_bits = vecs * (no * self._outlier_bits() + (d - no) * bits)
            scale_bits = vecs * 2 * 16  # two per-vector norms
        # QJL stores a 1-bit sign per element AND a per-vector fp16 residual RMS
        # (``qjl_rms``) for every encoded subset vector -- two subsets when the
        # outlier split is active; count all of it for honest whole-cache
        # accounting (memo A-7).
        qjl_bits = (x.numel() * 1 + vecs * (2 if split else 1) * 16) if use_qjl else 0
        return payload_bits + scale_bits + qjl_bits

    def encode(self, key, value, ctx=None) -> CompressedKV:
        key_qjl = self._qjl_enabled("key")
        value_qjl = self._qjl_enabled("value")
        kp = self._encode_tensor(key, self.bits, key_qjl)
        vp = self._encode_tensor(value, self._value_bits(), value_qjl)
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
        stored = (self._stored_bits(key, self.bits, key_qjl)
                  + self._stored_bits(value, self._value_bits(), value_qjl))
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
                    "qjl_roles": self.qjl_roles,
                    "rotation_mode": self.rotation_mode, "value_bits": self.value_bits})
        return cfg
