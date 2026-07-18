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
* ``dist``: ``"beta"`` (Algorithm-1-MSE default: exact finite-dimensional law,
  normalised by the per-vector L2 norm) or ``"gaussian"`` (explicit
  high-dimensional approximation, normalised by RMS).
* ``rotation_mode``: ``"dense"`` is the Algorithm-1 Gaussian-QR random
  rotation. ``"hadamard"`` is an explicit fast/deployment approximation; it is
  orthogonal but does not make every fixed vector uniform on the sphere.
* ``n_outlier`` / ``outlier_bits``: the paper's under-specified KV
  mixed-precision idea -- split ``n_outlier`` channels off and quantize them at
  ``outlier_bits`` while the rest use ``bits``.  This implementation completes
  the missing selection protocol by ranking one global channel set per role and
  public ``encode`` call by ``mean_(batch,head) max_token |x|``.  It is not the
  per-head top-L2 protocol later stated by BlockQuant.  The effective code
  payload bit-width
  is ``(n_outlier*outlier_bits + (d-n_outlier)*bits) / d``. Each of the two
  subsets also stores one fp16 norm. For ``head_dim=128``, the paper's
  ``32 ch @ 3 + 96 @ 2`` has 2.25 payload bits plus 0.25 norm bits = 2.5
  bits/coordinate before channel identities; ``32 @ 4 + 96 @ 3`` similarly
  totals 3.5 before channel identities. The paper text
  prints the payload-only arithmetic as 2.5 even though it is 2.25; this
  implementation keeps the stated 32/96 split and reports payload plus the two
  required norms and a packed ``d``-bit channel bitmap explicitly rather than
  hiding that inconsistency. Consequently 2.5/3.5 are nominal operating-point
  labels, not exact realised stored rates for this dynamic-mask research path. The
  outlier channel set is chosen *once per encoded role tensor*, so under
  incremental decode (``q_len=1``) the global role mask is dynamic per step.
  This is exact for the fake-quant research layer but is a layout a real kernel
  cannot express statically -- the paper must state this per-update granularity
  (memo A-8).
* ``use_qjl``: TurboQuant-prod's 1-bit QJL residual sketch for keys by default.
  The scalar MSE reconstruction is augmented with
  ``sqrt(pi/2)/m * ||r||_2 * S^T sign(S r)``, which makes inner products with
  the reconstructed residual unbiased over the Gaussian sketch.  To compare at
  an equal payload budget, instantiate it with ``bits=target_bits-1``; the
  residual norm is still counted explicitly in ``stored_bits``.  The paper's
  exact unbiasedness statement assumes a real-valued residual norm; the fp16
  data-plane norm used here introduces only header-rounding error and must not
  be described as exactly unbiased after serialization.

``stored_bits`` has a deliberately narrower meaning than serialized-object or
resident-memory size.  It counts only fixed-width encoded-KV data-plane fields:
codes, signed-fp16 scales, and enabled outlier bitmap/QJL fields.  The Python
payload's self-describing ``bits``/``m``/rotation/distribution/seed tags and
container bytes are uncharged out-of-band/cache-level control state, as are the
global codebook, projection, and rotation caches.  The ordinary payload is not
bit packed, so this logical count is not physical serving memory.

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
    if dist == "gaussian":
        return lloyd_max_gaussian(bits)
    raise ValueError(f"unknown TurboQuant source law {dist!r}")


@lru_cache(maxsize=64)
def _codebook_on(bits: int, dist: str, d: int, device_str: str) -> torch.Tensor:
    """Return the immutable scalar codebook cached on its target device.

    TurboQuant's codebook is a global parameter (Algorithms 1/2), not payload
    metadata.  Reusing one device copy avoids a CPU-to-device transfer and an
    allocation on every public ``encode`` and ``decode`` call without changing
    the codes, reconstruction, or stored-bit accounting.
    """
    return _codebook(int(bits), str(dist), int(d)).to(torch.device(device_str))


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
        dist: ``"beta"`` (L2-normalised, exact finite-Beta codebook used by
            the paper's Algorithm 1) or ``"gaussian"``
            (RMS-normalised high-dimensional approximation).
        n_outlier: Number of highest-energy channels quantized at ``outlier_bits``.
        outlier_bits: Bit-width for the outlier channels.
        use_qjl: Enable TurboQuant-prod's 1-bit Gaussian QJL residual sketch.
        qjl_roles: Which roles receive QJL when ``use_qjl`` is true:
            ``"key"`` (TurboQuant-prod default), ``"value"``, or ``"both"``.
        rotation_mode: ``"dense"`` (paper Gaussian-QR random orthogonal) or
            ``"hadamard"`` (fast WHT approximation, power-of-two dims).
        seed: Seed for the deterministic rotations.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    dist: str = "beta"
    n_outlier: int = 0
    outlier_bits: int = None
    use_qjl: bool = False
    qjl_roles: str = "key"
    rotation_mode: str = "dense"
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
        if not isinstance(self.use_qjl, bool):
            raise ValueError(f"use_qjl must be bool, got {self.use_qjl!r}")
        if not (
            isinstance(self.bits, int) and not isinstance(self.bits, bool) and 1 <= self.bits <= 8
        ):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        for label, value in (
            ("value_bits", self.value_bits),
            ("outlier_bits", self.outlier_bits),
        ):
            if value is not None and not (
                isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 8
            ):
                raise ValueError(f"{label} must be None or int in 1..8, got {value!r}")
        if not (
            isinstance(self.n_outlier, int)
            and not isinstance(self.n_outlier, bool)
            and self.n_outlier >= 0
        ):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if (
            self.n_outlier > 0
            and self.outlier_bits is None
            and max(self.bits, self._value_bits()) == 8
        ):
            raise ValueError(
                "8-bit TurboQuant role cannot default outlier_bits to that role's bits+1"
            )

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def teacher_forced_temporal_dependency(self):
        dependency = super().teacher_forced_temporal_dependency()
        if dependency is not None:
            return dependency
        if self.n_outlier > 0:
            return "dynamic_outlier_channels_over_update_tokens"
        return None

    def _outlier_bits(self, base_bits: int) -> int:
        """Resolve the shared override or default relative to this K/V role."""
        return self.outlier_bits if self.outlier_bits is not None else int(base_bits) + 1

    def _qjl_seed(self, seed: int) -> int:
        return int(seed) + 104729

    def _qjl_enabled(self, role: str) -> bool:
        return bool(self.use_qjl and self.qjl_roles in (role, "both"))

    @staticmethod
    def _validate_tensor_pair(key: torch.Tensor, value: torch.Tensor) -> None:
        """Reject inputs that would silently corrupt the fixed-width payload.

        In particular, ``torch.searchsorted`` accepts NaNs and fp16 conversion
        accepts infinities.  Without an explicit boundary check, a finite-bit
        TurboQuant payload could therefore decode to NaN/Inf while still being
        reported as a successful benchmark sample.
        """
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("TurboQuant key and value must be torch tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("TurboQuant expects K/V layout (batch, heads, sequence, head_dim)")
        if key.shape != value.shape:
            raise ValueError(
                f"TurboQuant key/value shapes must match, got {tuple(key.shape)} and "
                f"{tuple(value.shape)}"
            )
        if key.device != value.device:
            raise ValueError("TurboQuant key and value must be on the same device")
        if key.dtype != value.dtype:
            raise ValueError("TurboQuant key and value must have the same dtype")
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError("TurboQuant key and value must have floating-point dtypes")
        if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
            raise ValueError("TurboQuant rejects non-finite K/V inputs")

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
        if kind == "d":
            q = _ortho_on(m, seed, str(xr.device))
            return xr @ q.t()
        raise ValueError(f"unknown TurboQuant rotation payload tag {kind!r}")

    def _quant_subset(self, x, bits, seed, use_qjl=False):
        """Rotate ``(..., m)``, normalise per-vector, quantize to the codebook."""
        m = x.shape[-1]
        xr, rot = self._rotate(x, m, seed)
        if self.dist == "beta":
            scale = xr.norm(dim=-1, keepdim=True)  # L2 norm gamma
        else:
            scale = xr.pow(2).mean(dim=-1, keepdim=True).sqrt()  # RMS
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("TurboQuant normalization scale is not finite in float32")
        # Preserve the exact normalization of every nonzero vector.  A fixed
        # epsilon would change the source law for tiny finite vectors; only the
        # mathematically directionless zero vector needs a safe divisor.
        safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        z = xr / safe_scale
        cb = _codebook_on(bits, self.dist, m, str(x.device))
        codes = quantize_to_codebook(z, cb)
        # ``stored_bits`` charges one signed fp16 scale per vector.  Round this
        # data-plane field here (rather than only at a packed-kernel boundary) so the
        # fake/payload path has exactly the precision that it reports and that
        # the native cache stores.
        stored_scale = scale.to(dtype=torch.float16)
        if not bool(torch.isfinite(stored_scale).all()):
            raise ValueError("TurboQuant scale is not representable by its fp16 data-plane field")
        payload = {
            "codes": codes,
            "scale": stored_scale,
            "bits": bits,
            "m": m,
            "rot": rot,
            "dist": self.dist,
        }
        if use_qjl:
            recon = cb[codes] * stored_scale.to(dtype=cb.dtype)
            resid = xr - recon
            qjl_seed = self._qjl_seed(seed)
            proj = _qjl_projection_on(m, qjl_seed, str(x.device))
            z = resid.float().reshape(-1, m) @ proj.t()
            payload["qjl_sign"] = (z >= 0).reshape_as(resid)
            # The residual norm is an enabled data-plane field and is likewise
            # charged as a signed fp16 scalar in ``_stored_bits``.
            payload["qjl_norm"] = resid.float().norm(dim=-1, keepdim=True).to(dtype=torch.float16)
            if not bool(torch.isfinite(payload["qjl_norm"]).all()):
                raise ValueError(
                    "TurboQuant QJL residual norm is not representable by its fp16 "
                    "data-plane field"
                )
            payload["qjl_seed"] = qjl_seed
        return payload

    def _dequant_subset(self, p, device, dtype):
        cb = _codebook_on(p["bits"], p["dist"], p["m"], str(device))
        # Accept legacy fp32 payloads, but always make the arithmetic dtype
        # explicit.  New payloads contain fp16 metadata; computation remains
        # fp32 because the Lloyd-Max codebook is fp32.
        scale = p["scale"].to(device=device, dtype=cb.dtype)
        xr = cb[p["codes"]] * scale
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(torch.float32)
            if "qjl_norm" in p:
                proj = _qjl_projection_on(p["m"], p["qjl_seed"], str(device))
                qjl = (sign.reshape(-1, p["m"]) @ proj).reshape_as(xr)
                qjl = qjl * (math.sqrt(math.pi / 2.0) / float(p["m"]))
                qjl_norm = p["qjl_norm"].to(device=device, dtype=torch.float32)
                xr = xr.float() + qjl * qjl_norm
            else:  # legacy payloads written by the pre-QJL-sketch prototype
                xr = xr + sign.to(xr.dtype) * p["qjl_rms"]
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode (handles outlier split) ----------------------
    def _encode_tensor(self, x, bits, use_qjl=False):
        head_dim = x.shape[-1]
        if self.n_outlier >= head_dim and self.n_outlier > 0:
            raise ValueError(
                f"n_outlier={self.n_outlier} must be smaller than head_dim={head_dim}"
            )
        if self.n_outlier <= 0:
            return {
                "mode": "single",
                "main": self._quant_subset(x, bits, self.seed, use_qjl),
                "head_dim": head_dim,
            }
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
            "mode": "outlier",
            "head_dim": head_dim,
            "mask": mask,
            "out": self._quant_subset(x_out, self._outlier_bits(bits), self.seed + 1, use_qjl),
            "in": self._quant_subset(x_in, bits, self.seed + 2, use_qjl),
        }

    def _decode_tensor(self, payload, dtype):
        device = (
            payload["main"]["codes"].device
            if payload["mode"] == "single"
            else payload["out"]["codes"].device
        )
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
        """Count fixed-width data-plane fields, never Python control bytes.

        Rotation/distribution/seed tags make the materialized payload
        self-describing, while codebooks, projections, and transforms are
        cache-level state.  None of those control objects is charged here.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        split = 0 < self.n_outlier < d
        if not split:
            payload_bits = x.numel() * bits
            scale_bits = vecs * 16
            mask_bits = 0
        else:
            no = self.n_outlier
            payload_bits = vecs * (no * self._outlier_bits(bits) + (d - no) * bits)
            scale_bits = vecs * 2 * 16  # two per-vector norms
            # The selected channels are data-dependent for every encoded
            # update.  A decoder therefore needs their identity; account a
            # logically packed bitmap once per role payload.
            mask_bits = d
        # QJL stores a 1-bit projected sign per element AND a per-vector fp16
        # residual L2 norm for every encoded subset vector -- two subsets when the
        # outlier split is active; count all of it for honest whole-cache
        # accounting (memo A-7).
        qjl_bits = (x.numel() * 1 + vecs * (2 if split else 1) * 16) if use_qjl else 0
        return payload_bits + scale_bits + mask_bits + qjl_bits

    def encode(self, key, value, ctx=None) -> CompressedKV:
        self._validate_tensor_pair(key, value)
        key_qjl = self._qjl_enabled("key")
        value_qjl = self._qjl_enabled("value")
        kp = self._encode_tensor(key, self.bits, key_qjl)
        vp = self._encode_tensor(value, self._value_bits(), value_qjl)
        if kp is None or vp is None:
            kq = fake_quantize_affine(key, self.bits, axis="token")
            vq = fake_quantize_affine(value, self._value_bits(), axis="token")
            # Count the per-token affine scale + integer zero, like
            # the AffineKVCodec path, so the fallback's compression ratio is not
            # over-stated. This is retained only for legacy/custom subclasses;
            # the built-in rotation now falls back to dense QR for non-power-of-two dims.
            kvecs = key.shape[0] * key.shape[1] * key.shape[2]
            vvecs = value.shape[0] * value.shape[1] * value.shape[2]
            stored = (
                key.numel() * self.bits
                + kvecs * (16 + self.bits)
                + value.numel() * self._value_bits()
                + vvecs * (16 + self._value_bits())
            )
            return CompressedKV(
                key_q=kq,
                value_q=vq,
                meta={
                    "dtype": key.dtype,
                    "fallback": True,
                    "stored_bits": int(stored),
                    "passthrough": True,
                },
            )
        stored = self._stored_bits(key, self.bits, key_qjl) + self._stored_bits(
            value, self._value_bits(), value_qjl
        )
        return CompressedKV(
            key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)}
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        if compressed.meta.get("fallback") or compressed.meta.get("passthrough"):
            return compressed.key_q.to(dtype), compressed.value_q.to(dtype)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(
            compressed.value_q, dtype
        )

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "dist": self.dist,
                "n_outlier": self.n_outlier,
                "outlier_bits": self.outlier_bits,
                "use_qjl": self.use_qjl,
                "qjl_roles": self.qjl_roles,
                "rotation_mode": self.rotation_mode,
                "seed": self.seed,
                "value_bits": self.value_bits,
            }
        )
        return cfg

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        """Return the role-resolved scientific recipe at ``head_dim``.

        ``rotation_mode`` alone is not sufficient provenance: Hadamard falls
        back to dense for a non-power-of-two main subset, and K/V may use
        different bit widths.  Formal benchmark outputs persist this record so
        aggregation can reject a silently changed transform or scale law.
        """
        d = int(head_dim)
        if d <= 0 or self.n_outlier >= d:
            raise ValueError("head_dim must exceed TurboQuant n_outlier")
        main_dim = d - int(self.n_outlier)

        def _rotation_for_dim(dim: int) -> str:
            return (
                "hadamard"
                if self.rotation_mode == "hadamard" and is_power_of_two(dim)
                else "dense"
            )

        def _role(bits: int, role: str) -> dict:
            main_seed = int(self.seed + 2) if self.n_outlier else int(self.seed)
            record = {
                "bits": int(bits),
                "block_size": 1,
                "rotation_mode": _rotation_for_dim(main_dim),
                "rotation_seed": main_seed,
                "scale_mode": "l2" if self.dist == "beta" else "rms",
                "mathematical_scale_mode": ("l2" if self.dist == "beta" else "rms"),
                "source_law": self.dist,
                "quantized_dim": main_dim,
                "outlier_channels": int(self.n_outlier),
                "outlier_bits": (int(self._outlier_bits(bits)) if self.n_outlier else None),
                "outlier_selection": (
                    "top_n_dynamic_max_abs_over_tokens" if self.n_outlier else "none"
                ),
                "qjl_enabled": self._qjl_enabled(role),
            }
            if self.n_outlier:
                out_bits = int(self._outlier_bits(bits))
                common = {
                    "block_size": 1,
                    "scale_mode": "l2" if self.dist == "beta" else "rms",
                    "mathematical_scale_mode": "l2" if self.dist == "beta" else "rms",
                    "source_law": self.dist,
                    "qjl_enabled": self._qjl_enabled(role),
                }
                record["subsets"] = {
                    "inlier": {
                        **common,
                        "bits": int(bits),
                        "quantized_dim": int(main_dim),
                        "rotation_mode": _rotation_for_dim(main_dim),
                        "rotation_seed": int(self.seed + 2),
                    },
                    "outlier": {
                        **common,
                        "bits": out_bits,
                        "quantized_dim": int(self.n_outlier),
                        "rotation_mode": _rotation_for_dim(int(self.n_outlier)),
                        "rotation_seed": int(self.seed + 1),
                    },
                }
            return record

        return {
            "schema_version": 1,
            "head_dim": d,
            "codec": "turboquant",
            "calibration": None,
            "roles": {
                "key": _role(int(self.bits), "key"),
                "value": _role(int(self._value_bits()), "value"),
            },
        }
