"""HayateQuant: the fixed, data-free block-VQ KV-cache codec.

HayateQuant applies a deterministic orthogonal rotation, short source-law
vector codebooks, a fixed key-heavy K/V rate split, and role-aware scales.
The canonical public recipes live in the hayate.recipes module and are named
only hayatequant. Non-adopted component knobs are not part of its public API.

Logical stored_bits counts packed code bits and signed-fp16 scales. It excludes
Python containers and global codebooks/rotations, and is not a claim about
materialized research payload memory. The vLLM plugin provides the separate
packed serving path.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import torch

from ..._base import KVCodec, CompressedKV
from ..._registry import register_kv_codec
from ..rotation import (
    is_power_of_two,
    random_orthogonal,
    hybrid_rotation_matrix,
)
from .vq import (
    load_hayatequant_codebook,
    quantize_exact_pre,
    lookup_codewords,
    lut_spec_for,
    build_vq_lut,
    quantize_via_lut,
    require_hayatequant_codebooks,
)
from .._scale import fit_scale


@lru_cache(maxsize=64)
def _block_codebook_by_code_bits(
    code_bits: int,
) -> torch.Tensor:
    """Load one canonical two-coordinate Gaussian role codebook."""

    return load_hayatequant_codebook(code_bits)


@lru_cache(maxsize=16)
def _dense_rotation(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _dense_rotation_on(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident dense orthogonal matrix, cached per (dim, seed, device, dtype)."""
    return _dense_rotation(dim, seed).to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def _block_codebook_code_bits_on(
    code_bits: int,
    device_str: str,
) -> torch.Tensor:
    """Device-resident canonical role codebook."""
    return _block_codebook_by_code_bits(code_bits).to(torch.device(device_str))


@lru_cache(maxsize=64)
def _block_lut_code_bits_on(
    code_bits: int,
    bins: int,
    top_t: int,
    device_str: str,
) -> torch.Tensor:
    """Device-resident candidate-list LUT for a canonical role codebook."""
    cb = _block_codebook_code_bits_on(code_bits, device_str)
    return build_vq_lut(cb, bins, top_t)


@lru_cache(maxsize=64)
def _codebook_aux_code_bits_on(
    code_bits: int,
    device_str: str,
):
    """Precomputed constants for one canonical role codebook."""
    cb32 = _block_codebook_code_bits_on(code_bits, device_str)
    cb_score = cb32.half() if torch.device(device_str).type == "cuda" else cb32
    csq = cb_score.pow(2).sum(dim=1)
    return cb32, cb_score, csq, None


def _codebook_aux_from_tensor(cb32: torch.Tensor, device) -> tuple:
    """Per-codebook constants for an already materialised learned codebook."""
    cb32 = cb32.to(device=device, dtype=torch.float32).contiguous()
    if cb32.ndim != 2 or cb32.shape[1] != 2:
        raise ValueError("calibrated HayateQuant codebooks must have block_size=2")
    cb_score = cb32.half() if torch.device(device).type == "cuda" else cb32
    csq = cb_score.pow(2).sum(dim=1)
    return cb32, cb_score, csq, None


def _roundtrip_tensors(x, w, cb32, cb_score, csq, bounds, lut, bins, block_size, modes):
    """Fused encode+decode over ``R`` stacked role tensors (pure tensor ops).

    Args:
        x: ``(R, batch, heads, seq, m)`` fp32 input -- key and value stacked
            along dim 0 when their bit-widths (hence codebook and rotation)
            agree, else a single role at a time.
        w: ``(m, m)`` fp32 orthogonal rotation (the folded hybrid / signed-
            Hadamard / dense GEMM matrix).
        cb32 / cb_score / csq / bounds: Cached assignment constants.
        lut / bins: Candidate-list LUT, or ``None`` for exact assignment.
        block_size: Resolved VQ block size.
        modes: Per-role scale modes, e.g. ``("nc", "ls_iter")`` -- a static
            Python tuple, so ``torch.compile`` specialises on it.

    The body reproduces ``_quant_rotor`` followed by ``_decode_rotor``
    *without* materialising a payload: one batched rotation GEMM for all
    roles, one RMS, the nearest-codeword assignment, the role-asymmetric
    scale fit, and one batched inverse-rotation GEMM -- and the reconstruction
    reuses the codeword gather already performed at encode time instead of
    repeating it at decode time.  Every branch below depends only on config
    and shapes (never on tensor *values*), so there is no host-device
    synchronisation and the whole function is capturable by
    ``torch.compile(mode="reduce-overhead")`` into a single CUDA graph --
    which is what eliminates the launch-bound overhead that dominates the
    per-token cost at decode time.
    """
    m = x.shape[-1]
    n_blocks = m // block_size
    xr = x @ w
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

    def _assign(vec):
        """Nearest-codeword reconstruction of ``vec`` (``(..., m)`` -> fp32)."""
        vb = vec.reshape(*vec.shape[:-1], n_blocks, block_size)
        if block_size == 1:
            codes = torch.searchsorted(bounds, vb.squeeze(-1).contiguous())
        elif lut is not None:
            codes = quantize_via_lut(
                vb,
                lut,
                bins,
                cb32,
                scoring_cb=cb_score,
                scoring_csq=csq,
            )
        else:
            zb = vb
            if zb.is_cuda and zb.dtype == torch.float32 and cb_score.dtype == torch.float16:
                zb = zb.half()
            codes = quantize_exact_pre(zb, cb_score, csq)
        return lookup_codewords(cb32, codes).reshape(vec.shape)

    outs = []
    for i, mode in enumerate(modes):
        if mode == "bsm":
            mode = "ls"
        elif mode == "ub":
            mode = "ip"
        xi, ri = xr[i], rms[i]
        chat = _assign(xi / ri)
        if mode == "rms":
            scale = ri
        elif mode == "nc":
            cn = chat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = xi.norm(dim=-1, keepdim=True) / cn
        else:
            num = (xi * chat).sum(dim=-1, keepdim=True)
            den = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            if mode == "ip":
                num_abs = num.abs().clamp_min(1e-8)
                num_safe = torch.where(num < 0, -num_abs, num_abs)
                scale = (xi * xi).sum(dim=-1, keepdim=True) / num_safe
            else:
                scale = num / den
                if mode == "ls_iter":
                    scale_abs = scale.abs().clamp_min(1e-8)
                    scale_safe = torch.where(scale < 0, -scale_abs, scale_abs)
                    chat = _assign(xi / scale_safe)
                    num2 = (xi * chat).sum(dim=-1, keepdim=True)
                    den2 = (chat * chat).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    scale = num2 / den2
        # Match the payload/native cache contract: a single signed fp16 scale
        # is stored per vector.  The fused fake path skips the payload, so it
        # must explicitly round through fp16 before reconstructing or it would
        # silently evaluate a higher-precision codec than ``stored_bits``.
        stored_scale = scale.to(dtype=torch.float16)
        outs.append(chat * stored_scale.to(dtype=chat.dtype))
    return torch.stack(outs, dim=0) @ w.t()


@register_kv_codec("hayatequant")
@dataclass(init=False)
class HayateQuantCodec(KVCodec):
    """The single adopted HayateQuant codec at average rate 2, 3, or 4.

    All algorithmic choices are fixed by the paper operating point: a
    two-coordinate Gaussian block codebook, K/V code widths ``2r+2`` and
    ``2r-2``, rate-dependent rotation, fixed CUDA candidate-LUT assignment,
    NC key scaling, and one-step LS value scaling.  Optional calibration replaces
    persistent rotations only; it does not create another method name or
    alter the per-token layout.

    Use :func:`onecomp.kvcomp.codec.hayate.build_hayatequant` in user code.
    The deliberately small constructor prevents obsolete ablation knobs from
    leaking into saved configs or silently changing what ``hayatequant`` means.
    """

    dist: ClassVar[str] = "gaussian"
    block_size: ClassVar[int] = 2
    key_scale_mode: ClassVar[str] = "nc"
    value_scale_mode: ClassVar[str] = "ls_iter"
    seed: ClassVar[int] = 0

    calibration: object

    def __init__(
        self,
        bits: int = 4,
        *,
        calibration=None,
        skip_layers=None,
    ) -> None:
        self.name = "hayatequant"
        self.flag_calibration = calibration is not None
        self.bits = bits
        self.key_axis = "token"
        self.value_axis = "token"
        self.group_size = -1
        self.symmetric = False
        self.skip_layers = list(skip_layers or [])
        self.residual_window = 0
        self.calibration = calibration
        KVCodec.__post_init__(self)

    def _calib(self):
        """The loaded :class:`HayateQuantCalibration`, or None (data-free)."""
        if self.calibration is None:
            return None
        if isinstance(self.calibration, str):
            from .calibration import HayateQuantCalibration

            # Load once; replace the path with the object (idempotent).
            self.calibration = HayateQuantCalibration.load(self.calibration)
        return self.calibration

    def validate_params(self):
        if (
            isinstance(self.bits, bool)
            or not isinstance(self.bits, int)
            or self.bits not in (2, 3, 4)
        ):
            raise ValueError("HayateQuant rate must be one of 2, 3, or 4")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in self.skip_layers
        ):
            raise ValueError("skip_layers must contain non-negative integer layer indices")
        if self.calibration is not None:
            from .calibration import HayateQuantCalibration

            if not isinstance(self.calibration, (str, HayateQuantCalibration)):
                raise ValueError(
                    "calibration must be None, a path, or a HayateQuantCalibration, "
                    f"got {type(self.calibration).__name__}"
                )
        # Never perform an online fit in the inference path.  All five
        # canonical role codebooks must be present and hash-verified.
        require_hayatequant_codebooks()
        # OneComp convention: the flag (not the folder) marks calibrated codecs.
        self.flag_calibration = self.calibration is not None

    def _role_code_bits(self, role: str) -> int:
        if role not in {"k", "v"}:
            raise ValueError("role must be 'k' or 'v'")
        return 2 * self.bits + (2 if role == "k" else -2)

    def _code_bits_for(self, role: str, bits: int, block_size: int) -> int:
        if bits != self.bits or block_size != self.block_size:
            raise ValueError("HayateQuant role widths are fixed by its average rate")
        return self._role_code_bits(role)

    def _resolve_block_size_for(
        self,
        role: str,
        bits: int,
        m: int,
    ) -> int:
        """Validate the canonical two-coordinate role layout."""
        self._role_code_bits(role)
        if bits != self.bits:
            raise ValueError("role bits must equal the configured average rate")
        if m % self.block_size:
            raise ValueError(
                f"HayateQuant block_size={self.block_size} must divide head dimension {m}"
            )
        return self.block_size

    def _scale_mode_for(self, role: str) -> str:
        return self.key_scale_mode if role == "k" else self.value_scale_mode

    def _resolved_mode(self, d: int, block_size: int, bits: float) -> str:
        """Resolve ``"auto"`` to a concrete rotation given block size and rate.

        Block-VQ (``block_size>1``) needs each rotated block to be close to
        isotropic (the codebook is fit to an isotropic law); both the hybrid
        (Hadamard + per-block Haar) and the plain Haar-dense rotation provide
        that at identical runtime cost (one folded GEMM). The adopted
        rate-dependent rule uses hybrid rotation at role rates three and
        above, and dense rotation at role rates one and two. Non-power-of-two
        head dimensions use the dense fallback.
        """
        if block_size != self.block_size:
            raise ValueError("HayateQuant rotation requires block_size=2")
        if bits <= 2:
            return "dense"
        return "hybrid" if is_power_of_two(d) else "dense"

    def _rotation_rate_for(
        self,
        role: str,
        nominal_bits: int,
        block_size: int,
    ) -> float:
        """Role rate that controls the fixed rate-dependent rotation."""
        code_bits = self._code_bits_for(role, nominal_bits, block_size)
        return float(code_bits) / float(block_size)

    # --- rotation ----------------------------------------------------------
    def _rotate(
        self,
        x: torch.Tensor,
        m: int,
        block_size: int,
        bits: float,
        role: str = "k",
        layer_idx=None,
    ):
        """Rotate ``x`` (``..., m``); returns ``(x_rotated, rot_tag)``.

        Hybrid rotation requires a power-of-two head dimension and otherwise
        resolves to the dense QR rotation.

        The rot tag ``(kind, block_size, seed)`` is stored in the payload so
        decode reconstructs the exact rotation used at encode time.
        """
        cal = self._calib()
        if cal is not None:
            w = cal.rotation_on(role, m, x.device)
            if w is not None:
                # Learned (SpinQuant-style) rotation: same single GEMM as the
                # data-free path; the tag carries the artifact digest so
                # decode can verify the matching artifact is loaded.
                return x.float() @ w, ("c", role, cal.digest, layer_idx)
        mode = self._resolved_mode(m, block_size, bits)
        x = x.float()
        if mode == "hybrid" and is_power_of_two(m):
            # Folded hybrid rotation: the global signed-Hadamard and the
            # per-block dense rotations are pre-composed into ONE orthogonal
            # GEMM (see rotation.hybrid_rotation_matrix), so the hybrid
            # rotation costs exactly as much as a plain dense rotation at
            # runtime -- the Hadamard/Dense interpolation is free.
            w = hybrid_rotation_matrix(m, block_size, self.seed, str(x.device), torch.float32)
            return x @ w, ("y", block_size, self.seed)
        q = _dense_rotation_on(m, self.seed, str(x.device), torch.float32)
        return x @ q, ("d", None, self.seed)

    def _unrotate(self, xr: torch.Tensor, m: int, rot) -> torch.Tensor:
        kind, bs = rot[0], rot[1]
        if kind == "c":  # calibrated: rot == ("c", role, digest[, layer_idx])
            role, digest = rot[1], rot[2]
            layer_idx = rot[3] if len(rot) > 3 else None
            cal = self._calib()
            if cal is None or cal.digest != digest:
                raise ValueError(
                    "payload was encoded with a calibrated rotation "
                    f"(digest {digest}) but the codec has "
                    f"{'no' if cal is None else 'a different'} calibration loaded"
                )
            return xr @ cal.rotation_on(role, m, xr.device).t()
        seed = rot[2]
        if kind == "y":
            w = hybrid_rotation_matrix(m, bs, seed, str(xr.device), torch.float32)
            return xr @ w.t()
        if kind == "d":
            q = _dense_rotation_on(m, seed, str(xr.device), torch.float32)
            return xr @ q.t()
        raise ValueError(f"unknown HayateQuant rotation payload tag {kind!r}")

    # Below this many (blocks * codewords) score products the exact GEMM is a
    # single trivially small kernel launch (decode-step encodes: one new
    # token), so the LUT's multi-kernel gather pipeline only adds overhead.
    def _lut_spec(self, block_size: int, code_bits: int, is_cuda: bool) -> tuple[int, int] | None:
        """Return the one fixed CUDA assignment path; CPU uses an exact fallback."""
        if not is_cuda:
            return None
        k = 1 << int(code_bits)
        spec = lut_spec_for(block_size, k)
        if spec is None:
            raise RuntimeError("canonical HayateQuant LUT specification is missing")
        return spec

    # --- rotate + block-VQ + role-asymmetric scale for one channel subset --
    def _quant_rotor(
        self,
        xsub: torch.Tensor,
        bits: int,
        role: str,
        layer_idx=None,
    ) -> dict:
        m = xsub.shape[-1]
        block_size = self._resolve_block_size_for(role, bits, m)
        code_bits = self._code_bits_for(role, bits, block_size)
        rotation_rate = self._rotation_rate_for(
            role,
            bits,
            block_size,
        )
        xr, rot = self._rotate(xsub, m, block_size, rotation_rate, role, layer_idx=layer_idx)
        device_str = str(xsub.device)
        cal = self._calib()
        cb_cal = (
            cal.codebook_on(role, bits, block_size, xsub.device)
            if cal is not None and m == cal.head_dim
            else None
        )
        if cb_cal is not None:
            cb, exact_cb_score, exact_csq, _ = _codebook_aux_from_tensor(cb_cal, xsub.device)
        else:
            cb, exact_cb_score, exact_csq, _ = _codebook_aux_code_bits_on(code_bits, device_str)
        n_blocks = m // block_size
        lut_spec = self._lut_spec(block_size, code_bits, xr.is_cuda)
        if lut_spec is None:
            lut = None
        elif cb_cal is not None:
            lut = cal.lut_on(role, bits, block_size, lut_spec[0], lut_spec[1], xsub.device)
        else:
            lut = _block_lut_code_bits_on(
                code_bits,
                lut_spec[0],
                lut_spec[1],
                device_str,
            )

        def _payload(codes_: torch.Tensor, stored_scale_: torch.Tensor) -> dict:
            result = {
                "codes": codes_,
                "scale": stored_scale_,
                "bits": bits,
                "m": m,
                "block_size": block_size,
                "rot": rot,
                "code_bits": code_bits,
            }
            if cb_cal is not None:
                # Decode must gather from the SAME learned codebook; record the
                # role + artifact digest (same convention as the "c" rot tag).
                result["cb_cal"] = (role, cal.digest)
            return result

        # Do not dispatch the monolithic Triton round-trip kernel as a payload
        # encoder. Its reduction order changed codes on a canonical B200
        # fixture, so payload creation keeps the audited exact/LUT arithmetic.

        # One RMS scale per whole subset vector (not per block) -- matches
        # TurboQuant's storage convention exactly (a single fp16 scale
        # per vector regardless of block_size).
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

        def _lookup(vec_full):
            vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
            if lut is not None:
                codes_ = quantize_via_lut(
                    vb,
                    lut,
                    lut_spec[0],
                    cb,
                    scoring_cb=exact_cb_score,
                    scoring_csq=exact_csq,
                )
                chat_b = lookup_codewords(cb, codes_)
            else:
                score_vb = (
                    vb.half()
                    if vb.is_cuda and vb.dtype == torch.float32
                    else vb.to(dtype=exact_cb_score.dtype)
                )
                codes_ = quantize_exact_pre(score_vb, exact_cb_score, exact_csq)
                chat_b = lookup_codewords(cb, codes_)
            return codes_, chat_b.reshape(*vec_full.shape[:-1], m)

        def _requant(z_full):
            codes2, chat2 = _lookup(z_full)
            return chat2, codes2

        codes, chat = _lookup(xr / rms)
        scale, codes2 = fit_scale(
            xr, chat, self._scale_mode_for(role), rms, requantize_fn=_requant
        )
        if codes2 is not None:
            codes = codes2
        # The logical accounting and native backend both store one signed
        # fp16 scale per vector.  Materialise that precision in the research
        # payload as well so quality and logical data-plane rate describe the
        # same codec.
        stored_scale = scale.to(dtype=torch.float16)
        return _payload(codes, stored_scale)

    def _decode_rotor(self, p: dict, device, dtype) -> torch.Tensor:
        if "cb_cal" in p:
            role, digest = p["cb_cal"]
            cal = self._calib()
            if cal is None or cal.digest != digest:
                raise ValueError(
                    "payload was encoded with a calibrated codebook "
                    f"(digest {digest}) but the codec has "
                    f"{'no' if cal is None else 'a different'} calibration loaded"
                )
            cb = cal.codebook_on(role, p["bits"], p["block_size"], device)
        else:
            cb = _block_codebook_code_bits_on(p["code_bits"], str(device))
        if p["codes"].is_cuda:
            from .triton_step import (
                decode_payload_triton,
                triton_payload_decode_supported,
            )

            if triton_payload_decode_supported(p["m"], p["block_size"], cb.shape[0]):
                xr = decode_payload_triton(p["codes"], p["scale"], cb, p["m"], p["block_size"])
            else:
                chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
                scale = p["scale"].to(device=device, dtype=chat.dtype)
                xr = chat * scale
        else:
            chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
            # New payloads contain fp16 metadata; legacy fp32 payloads remain
            # readable.  Restore explicitly to the codebook arithmetic dtype.
            scale = p["scale"].to(device=device, dtype=chat.dtype)
            xr = chat * scale
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode / decode ---------------------------------------
    def _encode_tensor(self, x: torch.Tensor, bits: int, role: str, layer_idx=None) -> dict:
        return {
            "main": self._quant_rotor(x, bits, role, layer_idx=layer_idx),
            "head_dim": x.shape[-1],
        }

    def _decode_tensor(self, payload: dict, dtype) -> torch.Tensor:
        device = payload["main"]["codes"].device
        return self._decode_rotor(payload["main"], device, dtype).to(dtype)

    def _stored_bits(self, x: torch.Tensor, bits: int, role: str) -> int:
        """Fixed-width data-plane accounting at the requested logical rate.

        Counts fixed block indices and one signed fp16 scale per vector.
        Global codebooks, rotations and LUTs are persistent model state and
        are deliberately excluded from this per-token data plane.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        block_size = self._resolve_block_size_for(role, bits, d)
        code_bits = self._code_bits_for(role, bits, block_size)
        return vecs * (d // block_size) * code_bits + vecs * 16

    def calibrate(self, kv_stats, ctx=None):
        """Fit and load a SpinQuant-style calibration (learned per-role
        rotation + real-data codebook refit) from captured KV samples.

        Args:
            kv_stats: ``{"key": (N, head_dim), "value": (N, head_dim)}`` real
                KV vectors (e.g. from
                :func:`.calibration.capture_kv_samples`).
            ctx: Optional kwargs forwarded to
                :func:`.calibration.calibrate_hayatequant`
                (``steps``, ``lr``, ``learn_rotation``, ``learn_codebook``...).

        Returns:
            The fitted :class:`.calibration.HayateQuantCalibration`
            (also installed on this codec; save it with ``.save(path)`` and
            reload via ``HayateQuantCodec(calibration=path)``).
        """
        from .calibration import calibrate_hayatequant

        self.calibration = calibrate_hayatequant(
            self, kv_stats["key"], kv_stats["value"], **(ctx or {})
        )
        self.flag_calibration = True
        return self.calibration

    def encode(self, key, value, ctx=None) -> CompressedKV:
        layer_idx = (ctx or {}).get("layer_idx")
        kp = self._encode_tensor(key, self.bits, "k", layer_idx=layer_idx)
        vp = self._encode_tensor(value, self.bits, "v", layer_idx=layer_idx)
        stored = self._stored_bits(key, self.bits, "k") + self._stored_bits(value, self.bits, "v")
        return CompressedKV(
            key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)}
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(
            compressed.value_q, dtype
        )

    # --- fused fake-quant round-trip (the serving hot path) -----------------
    def _rt_supported(self, m: int) -> bool:
        """Validate that both canonical role layouts support this head size."""
        for role, bits in (("k", self.bits), ("v", self.bits)):
            self._resolve_block_size_for(role, bits, m)
        return True

    def _rt_rotation(
        self,
        m: int,
        block_size: int,
        bits: float,
        device,
        role: str | None = None,
        layer_idx=None,
    ) -> torch.Tensor:
        """The (m, m) fp32 rotation matrix ``_rotate`` would apply (cached)."""
        cal = self._calib()
        if cal is not None and role is not None:
            w = cal.rotation_on(role, m, device)
            if w is not None:
                return w
        mode = self._resolved_mode(m, block_size, bits)
        dev = str(device)
        if mode == "hybrid" and is_power_of_two(m):
            return hybrid_rotation_matrix(m, block_size, self.seed, dev, torch.float32)
        return _dense_rotation_on(m, self.seed, dev, torch.float32)

    def _rt_call(
        self, x: torch.Tensor, bits: int, modes: tuple, role: str | None = None, layer_idx=None
    ) -> torch.Tensor:
        """Run the fused round-trip core on stacked roles ``x`` (R, b, h, s, m)."""
        m = x.shape[-1]
        block_size = self._resolve_block_size_for(role or "k", bits, m)
        code_bits = self._code_bits_for(role or "k", bits, block_size)
        rotation_rate = self._rotation_rate_for(
            role or "k",
            bits,
            block_size,
        )
        w = self._rt_rotation(m, block_size, rotation_rate, x.device, role, layer_idx=layer_idx)
        cal = self._calib()
        cb_cal = (
            cal.codebook_on(role, bits, block_size, x.device)
            if cal is not None and role is not None and m == cal.head_dim
            else None
        )
        if cb_cal is not None:
            cb32, cb_score, csq, bounds = _codebook_aux_from_tensor(cb_cal, x.device)
        else:
            cb32, cb_score, csq, bounds = _codebook_aux_code_bits_on(code_bits, str(x.device))
        spec = self._lut_spec(block_size, code_bits, x.is_cuda)
        if spec is None:
            lut, bins = None, 0
        elif cb_cal is not None:
            lut = cal.lut_on(role, bits, block_size, spec[0], spec[1], x.device)
            bins = spec[0]
        else:
            lut = _block_lut_code_bits_on(
                code_bits,
                spec[0],
                spec[1],
                str(x.device),
            )
            bins = spec[0]
        return _roundtrip_tensors(
            x,
            w,
            cb32,
            cb_score,
            csq,
            bounds,
            lut,
            bins,
            block_size,
            modes,
        )

    def fake_quant_roundtrip(self, key, value, ctx=None):
        """Fused encode+decode: ``(key_hat, value_hat, stored_bits)``.

        Numerically equivalent to ``decode(encode(key, value))`` (same
        rotation, codebook, assignment and scale-fit operations in the same
        dtypes -- see ``test_hayatequant_roundtrip_matches_encode_decode``)
        but skips payload construction and the decode-time codeword gather,
        and reuses cached per-codebook constants.  K/V retain separate rotation
        GEMM shapes so the result also preserves public-encoder fp16 scale
        rounding at the rare boundary cases.
        This research path always uses the canonical tensor arithmetic. Fast
        packed serving is provided by the vLLM backend rather than hidden
        environment-variable variants of this method.
        """
        m = key.shape[-1]
        if not self._rt_supported(m):
            return super().fake_quant_roundtrip(key, value, ctx=ctx)
        kb = vb = self.bits
        stored = int(self._stored_bits(key, kb, "k") + self._stored_bits(value, vb, "v"))
        k_mode, v_mode = self._scale_mode_for("k"), self._scale_mode_for("v")
        dtype = key.dtype
        layer_idx = (ctx or {}).get("layer_idx")
        # Keep each role's GEMM shape identical to the public payload encoder.
        # Stacking K/V changes the GEMM's row count and can select a different
        # reduction kernel; the resulting last-bit fp32 difference occasionally
        # crosses an fp16 scale-rounding boundary.  That made the advertised
        # fused round trip differ from ``decode(encode(...))`` by one adjacent
        # fp16 scale even though K/V shared a rotation and codebook.  Separate
        # role calls preserve the audited payload arithmetic exactly.
        khat = self._rt_call(
            key.float().unsqueeze(0), kb, (k_mode,), role="k", layer_idx=layer_idx
        )[0]
        vhat = self._rt_call(
            value.float().unsqueeze(0), vb, (v_mode,), role="v", layer_idx=layer_idx
        )[0]
        return khat.to(dtype), vhat.to(dtype), stored

    def get_kv_config(self) -> dict:
        """Serialize only user choices; all method constants are implicit."""

        cfg = {
            "name": "hayatequant",
            "bits": self.bits,
            "skip_layers": list(self.skip_layers),
        }
        cal = self._calib()
        if cal is not None:
            cfg["calibration"] = cal.digest
        return cfg

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        """Return the fully resolved immutable method contract."""

        d = int(head_dim)
        if d <= 0:
            raise ValueError("head_dim must be positive")
        if d % self.block_size:
            raise ValueError(
                f"HayateQuant block_size={self.block_size} must divide head dimension {d}"
            )
        cal = self._calib()

        def _role(role: str) -> dict:
            code_bits = self._role_code_bits(role)
            role_rate = code_bits / self.block_size
            scale = self._scale_mode_for(role)
            rotation = (
                "calibrated_artifact"
                if cal is not None
                else self._resolved_mode(
                    d,
                    self.block_size,
                    role_rate,
                )
            )
            return {
                "code_bits_per_block": code_bits,
                "payload_bits_per_coordinate": role_rate,
                "block_size": self.block_size,
                "rotation_mode": rotation,
                "rotation_seed": self.seed,
                "scale_mode": scale,
                "source_law": self.dist,
                "quantized_dim": d,
                "scale_storage_bits_per_vector": 16,
            }

        return {
            "schema_version": 1,
            "head_dim": d,
            "codec": "hayatequant",
            "average_rate_bits_per_coordinate": self.bits,
            "calibration": cal.digest if cal is not None else None,
            "assignment": {
                "cuda_encoder": "fixed_candidate_lut",
                "cpu_fallback": "exact_full_codebook",
                "lut_bins": 256,
                "lut_candidates": 2,
                "lut_limit": 5.0,
                "tie_break": "smallest_global_codebook_index",
            },
            "roles": {
                "key": _role("k"),
                "value": _role("v"),
            },
            "codebook_artifacts": require_hayatequant_codebooks(),
        }
