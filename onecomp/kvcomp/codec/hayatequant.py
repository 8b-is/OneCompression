"""HayateQuant: block-vector-quantized, hybrid-rotation KV codec.

HayateQuant is a **strict, data-free generalization of TurboQuant**
(arXiv:2504.19874) that closes TurboQuant's own stated distortion slack and
absorbs HALO's validated, storage-neutral levers into one codec, replacing
both ``turboquant`` and the (now removed) ``halo`` codec.

Recap: TurboQuant rotates a KV vector by a random orthogonal transform so its
coordinates are (exactly, for finite ``d``, or asymptotically as ``d -> inf``)
i.i.d. draws from a scalar law -- ``Beta((d-1)/2, (d-1)/2)`` or its Gaussian
limit -- then fits a **scalar** Lloyd-Max quantizer to that law, *entirely
data-free* (the codebook only depends on the bit-width, not on any real KV
data). The paper proves this is within a constant factor (``<= sqrt(3*pi/2)
~= 2.7x``) of the information-theoretic distortion-rate optimum. Classical
high-rate quantization theory (Gersho & Gray, *Vector Quantization and Signal
Compression*, 1992; Zamir, *Lattice Coding for Signals and Networks*, 2014)
identifies exactly where that residual factor lives: **independent scalar
quantization of a memoryless Gaussian source loses the "space-filling" and
"shape" gains available to any joint (vector/lattice) quantizer of dimension
> 1** -- a classical ``1.53`` dB (``~0.255`` bit/dim) gap at high rate that
only closes as the joint quantization dimension grows (``E8``: ``~0.65`` dB
gap remaining; this repo's own HALO analysis flags exactly this as unexplored
"Lever 5", see ``benchmark/kvcomp-longcontext/paper/HALO_ENHANCED_REPORT.md``).

HayateQuant closes this gap with two data-free, storage-neutral additions on
top of TurboQuant's own skeleton:

1. **Block vector quantization (the core contribution).** Instead of an
   independent scalar Lloyd-Max codebook, HayateQuant fits a
   ``block_size``-dimensional **vector** codebook (``2**(bits*block_size)``
   codewords) via Lloyd's algorithm (batched K-means, :mod:`._vq`) to
   *synthetic samples drawn exactly from TurboQuant's own source law* --
   i.i.d. ``N(0,1)^{block_size}`` (the ``d -> inf`` limit) or the exact
   finite-``d`` ``block_size``-dim marginal of a random point on ``S^{d-1}``.
   This is data-free in exactly the sense TurboQuant's own scalar codebook is
   data-free (fit offline to the theoretical law, not to real KV) -- just
   generalized from dimension 1 to ``block_size``. Nearest-codeword search is
   a single GEMM (``z @ cb.T``) + ``argmax`` (:func:`._vq.quantize_to_vq_codebook`),
   so encode/decode remain O(1) matmuls per block; no explicit lattice
   geometry (e.g. ``E8``) or per-lattice decoding algorithm is required, and
   because the codebook is *learned* on the exact source law it is always at
   least as good as any fixed lattice at the same rate. The total number of
   stored bits is unchanged (``bits`` per coordinate, same as TurboQuant --
   see ``_stored_bits``); only *how* those bits are allocated changes (jointly
   per block instead of independently per coordinate), which is exactly the
   controlled comparison needed to isolate the vector-quantization gain.
   ``block_size=1`` reduces the block codebook to the ordinary scalar
   Lloyd-Max codebook, bit-for-bit -- HayateQuant is therefore **always able
   to fall back to exactly TurboQuant's quantizer**, never worse.

2. **Hybrid rotation (resolves the Hadamard-vs-Dense trade-off).**
   TurboQuant/HALO's own benchmarks (``benchmark/kvcomp-codec/README.md``)
   found the *sequential* Hadamard transform gives marginally better
   Gaussianisation on real, anisotropic KV than a single Haar-random dense
   rotation (better incoherence => better codebook fit), while dense/folded-
   Hadamard-GEMM is 3-4x faster (one BLAS GEMM vs. a memory-bound butterfly).
   Block-VQ raises the stakes on this trade-off: the codebook is fit to an
   *isotropic* ``block_size``-dim law, so each rotated block must itself be
   close to isotropic for the codebook to fit well -- a fixed Hadamard
   transform alone (a deterministic matrix, randomized only by a global sign
   flip) is not Haar-random at the block level, so directly block-quantizing
   its output would mismatch the codebook. HayateQuant's ``rotation_mode=
   "hybrid"`` resolves this with a **two-level rotation**:

       R = HS_D  (global signed-Hadamard-GEMM, O(D log D) or one O(D^2) GEMM)
           followed by
           block-diagonal(Q_1, ..., Q_{D/b})  (independent dense Haar-random
           rotations, one per block, O(D * b), b << D)

   Composing two orthogonal transforms is orthogonal (norm-preserving,
   invertible via the transposes in reverse order, :mod:`.rotation`
   ``hybrid_rotate``/``hybrid_unrotate``), and the combined cost
   ``O(D log D + D*b)`` stays far below a single full ``D x D`` dense
   rotation (``O(D^2)``) while giving each block genuine Haar-random local
   isotropy -- keeping the Hadamard step's proven-good global incoherence
   against real anisotropic KV (the robustness lever) *and* the exact local
   isotropy the block-VQ codebook needs (the accuracy lever). The block size
   ``block_size`` is therefore literally the interpolation knob between "pure
   Hadamard" (``block_size=1``, cheapest, most robust, no vector-quantization
   gain) and "pure dense" (``block_size=D``, most expensive, most isotropic)
   that was raised as an open question for HALO -- HayateQuant answers it by
   only paying the dense-rotation cost on small blocks.

3. **HALO's validated, storage-neutral levers (inherited, not reinvented).**
   Role-asymmetric reconstruction scale (:mod:`._scale`): keys use the
   norm-correcting (``"nc"``) scale (preserves the attention logit, shrunk by
   ``cos(theta)`` instead of ``cos^2(theta)``), values use the least-squares
   (``"ls"``/``"ls_iter"``) scale (the correct MSE objective for values, which
   are consumed linearly). Outlier mixed-precision (``n_outlier``/
   ``outlier_bits``) for true 2.5/3.5-bit operating points. Optional QJL-style
   1-bit residual (``use_qjl``, off by default -- community evaluations agree
   it is unbiased but high-variance through softmax and can hurt attention).
   These levers are dimension-agnostic (:func:`._scale.fit_scale` reduces
   over the concatenated per-block codewords exactly as it would over
   concatenated scalar centroids), so they compose for free with block-VQ.

**HayateQuant is a strict superset of TurboQuant and HALO**: setting
``block_size=1`` and ``rotation_mode`` to ``"hadamard"``/``"hadamard_gemm"``/
``"dense"`` with ``key_scale_mode=value_scale_mode="rms"`` reproduces
TurboQuant bit-for-bit; adding the role-asymmetric scale modes reproduces
HALO bit-for-bit. The only genuinely new, previously-unexplored lever is
block-VQ (default on, via ``block_size=None`` auto-resolution), which is
exactly the "Lever 5 / future work" gap this repository's own HALO analysis
identified and left unimplemented.

See ``docs/design/kvcomp-architecture.md`` (HayateQuant section) and
``papers/hayatequant_paper/README.md`` for the full derivation, and
``benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py`` /
``verify_hayatequant_isotropy.py`` for the CPU-only, data-free, in-repo proofs
that (a) block-VQ strictly reduces distortion at equal bit-rate as
``block_size`` grows from 1, matching the classical space-filling-gain curve,
and (b) the hybrid rotation's block marginals are measurably closer to
isotropic than the Hadamard-only rotation's.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    signed_hadamard_matrix,
    lloyd_max_gaussian,
    lloyd_max_beta,
    hybrid_rotation_matrix,
)
from ._vq import (
    resolve_block_size,
    lloyd_vq_gaussian_block,
    lloyd_vq_beta_block,
    quantize_and_lookup,
    lookup_codewords,
    lut_spec_for,
    build_vq_lut,
    quantize_via_lut,
)
from ._scale import resolve_scale_mode, fit_scale


@lru_cache(maxsize=16)
def _scalar_codebook(bits: int, dist: str, m: int) -> torch.Tensor:
    if dist == "beta":
        # ``lloyd_max_beta`` lives on the *unit-sphere* coordinate scale
        # ([-1, 1], TurboQuant's L2-normalisation convention); HayateQuant
        # normalises by the RMS (= L2 / sqrt(m)) everywhere, so rescale the
        # codebook by sqrt(m) to match.  Decision boundaries scale with the
        # codewords, so the assigned codes -- and, after the per-vector scale,
        # the reconstruction -- are bit-for-bit identical to TurboQuant's
        # L2-normalised beta path (and consistent with the sqrt(d)-rescaled
        # block codebook of ``lloyd_vq_beta_block``).
        return lloyd_max_beta(bits, m) * math.sqrt(m)
    return lloyd_max_gaussian(bits)


@lru_cache(maxsize=32)
def _block_codebook(bits: int, block_size: int, dist: str, m: int) -> torch.Tensor:
    """``(2**(bits*block_size), block_size)`` data-free block-VQ codebook.

    ``block_size=1`` returns the ordinary scalar Lloyd-Max codebook reshaped
    to ``(2**bits, 1)`` -- the exact TurboQuant/HALO codebook, so every
    downstream computation (nearest-codeword search, role-asymmetric scale)
    reduces to the scalar case bit-for-bit.
    """
    if block_size == 1:
        return _scalar_codebook(bits, dist, m).unsqueeze(-1)
    if dist == "beta":
        return lloyd_vq_beta_block(bits, block_size, m)
    return lloyd_vq_gaussian_block(bits, block_size)


@lru_cache(maxsize=16)
def _dense_rotation(dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(dim, seed)


@lru_cache(maxsize=64)
def _dense_rotation_on(dim: int, seed: int, device_str: str, dtype) -> torch.Tensor:
    """Device-resident dense orthogonal matrix, cached per (dim, seed, device, dtype)."""
    return _dense_rotation(dim, seed).to(device=torch.device(device_str), dtype=dtype)


@lru_cache(maxsize=64)
def _block_codebook_on(bits: int, block_size: int, dist: str, m: int, device_str: str) -> torch.Tensor:
    """Device-resident block codebook, cached so the host->device copy happens once."""
    return _block_codebook(bits, block_size, dist, m).to(torch.device(device_str))


@lru_cache(maxsize=64)
def _block_lut_on(
    bits: int, block_size: int, dist: str, m: int, bins: int, top_t: int,
    device_str: str,
) -> torch.Tensor:
    """Device-resident candidate-list LUT for the block codebook.

    The build pays the exact O(cells * K) scoring a single time per
    (codebook, resolution, device); every subsequent encode is an O(T) exact
    argmin over the cell's candidate list (see :func:`._vq.build_vq_lut`).
    """
    cb = _block_codebook_on(bits, block_size, dist, m, device_str)
    return build_vq_lut(cb, bins, top_t)


@register_kv_codec("hayatequant")
@dataclass
class HayateQuantCodec(KVCodec):
    """Block-VQ + hybrid-rotation + role-asymmetric-scale KV quantization.

    Attributes:
        bits: Bits per coordinate for keys (and values unless ``value_bits``).
        value_bits: Optional separate bit-width for values (K>V asymmetry).
        dist: ``"gaussian"`` (RMS-normalised, high-d limit, fast default) or
            ``"beta"`` (the exact finite-``d`` law).
        block_size: Vector-quantization block size in ``{None, 1, 2, 4, 8}``.
            ``None`` (default) auto-resolves to the largest of ``(8,4,2,1)``
            whose codebook (``2**(bits*block_size)`` entries) fits under
            ``max_codebook_bits`` and evenly divides the subset dimension.
            ``block_size=1`` is the scalar TurboQuant/HALO quantizer exactly.
        max_codebook_bits: Log2 cap on the block-VQ codebook size (``K =
            2**max_codebook_bits`` worst case). This bounds three costs at
            once: the Lloyd-fit budget, the nearest-codeword search's
            ``O(n_blocks * K)`` GEMM (measured to dominate encode latency at
            ``K>~1000`` on real ``batch*heads*seq`` block counts -- see
            ``codec_speed.py`` / ``e2e_speed.py``), and Lloyd-fit sample
            efficiency (``K`` codewords need enough samples to fit well).
            Default ``9`` (``K<=512``) keeps every paper bit-width's encode
            latency within a small constant factor of TurboQuant/HALO while
            still using ``block_size>1`` (block-VQ) at 2/3/4-bit -- raise it
            (e.g. 12+) to trade encode latency for a larger, more accurate
            block codebook at higher bit-widths.
        rotation_mode: ``"auto"`` (default; picks ``"hybrid"`` when
            ``block_size>1`` and the subset dim is a power of two, else
            ``"hadamard_gemm"``/``"hadamard"`` when ``block_size==1``, else
            ``"dense"``), ``"hybrid"`` (global Hadamard-GEMM + per-block dense,
            requires power-of-two dim; folded offline into ONE orthogonal GEMM
            via :func:`rotation.hybrid_rotation_matrix`, so it costs exactly
            as much as ``"dense"`` at runtime), ``"hadamard_gemm"`` (single
            folded Hadamard GEMM), ``"hadamard"`` (sequential WHT butterfly),
            or ``"dense"`` (full QR random orthogonal, any dim).
        encoder: Nearest-codeword assignment algorithm for ``block_size>1``.
            ``"auto"`` (default) picks *per call* between the exact O(K)
            chunked-GEMM search (a tensor-core matmul -- fastest while the
            ``n_blocks x K`` score matrix stays small) and the O(T)-per-block
            candidate-list LUT encoder (:func:`._vq.quantize_via_lut`: grid
            cell -> top-``T`` candidate codewords -> exact argmin over the
            ``T``; ~10x faster once ``n_blocks * K`` crosses the measured
            crossover, e.g. long-prefill encodes against a large codebook).
            ``"lut"`` forces the LUT (error if infeasible for the block
            size); ``"exact"`` forces the exact search. The LUT restricts
            only the *candidate set* (distances over the candidates, and
            reconstruction, are exact); its excess distortion at the default
            specs is measured to be negligible (see
            ``test_hayatequant_lut_encoder_matches_exact_distortion``) while
            making encode latency independent of the codebook size ``K``.
        hadamard_gemm_max_dim: Above this head dim, ``"auto"`` falls back from
            the materialised Hadamard GEMM to the sequential WHT butterfly.
        ls_scale: Master switch for the role-asymmetric scale (``True``:
            smart per-role defaults below; ``False``: per-vector RMS on both
            roles, the TurboQuant-faithful baseline).
        key_scale_mode / value_scale_mode: Explicit override of the per-role
            scale (``"rms"``/``"nc"``/``"ls"``/``"ip"``/``"ls_iter"``, see
            ``._scale``); ``None`` defers to the ``ls_scale`` smart default
            (``"nc"`` for keys, ``"ls_iter"`` for values).
        n_outlier / outlier_bits: Outlier mixed-precision (ported from
            TurboQuant/HALO): split the ``n_outlier`` highest-energy channels
            off and quantize them at ``outlier_bits`` (default ``bits+1``)
            for true fractional-bit operating points (e.g. 2.5/3.5-bit).
        use_qjl: Optional 1-bit QJL-style residual refinement (off by default).
        seed: Seed for the deterministic rotations.
    """

    flag_calibration: bool = False
    bits: int = 4
    value_bits: int = None
    dist: str = "gaussian"
    block_size: int = None
    max_codebook_bits: int = 9
    rotation_mode: str = "auto"
    encoder: str = "auto"
    hadamard_gemm_max_dim: int = 4096
    ls_scale: bool = True
    key_scale_mode: str = None
    value_scale_mode: str = None
    n_outlier: int = 0
    outlier_bits: int = None
    use_qjl: bool = False
    seed: int = 0
    key_axis: str = "token"
    value_axis: str = "token"

    _SCALE_MODES = ("rms", "ls", "nc", "ip", "ls_iter")
    _BLOCK_SIZES = (None, 1, 2, 4, 8)

    def validate_params(self):
        if self.rotation_mode not in ("auto", "hybrid", "hadamard_gemm", "hadamard", "dense"):
            raise ValueError(
                "rotation_mode must be 'auto', 'hybrid', 'hadamard_gemm', 'hadamard' or 'dense'")
        if self.encoder not in ("auto", "lut", "exact"):
            raise ValueError(f"encoder must be 'auto', 'lut' or 'exact', got {self.encoder!r}")
        if self.dist not in ("gaussian", "beta"):
            raise ValueError("dist must be 'gaussian' or 'beta'")
        if not (isinstance(self.bits, int) and 1 <= self.bits <= 8):
            raise ValueError(f"bits must be int in 1..8, got {self.bits!r}")
        if self.block_size not in self._BLOCK_SIZES:
            raise ValueError(f"block_size must be one of {self._BLOCK_SIZES}, got {self.block_size!r}")
        for mode in (self.key_scale_mode, self.value_scale_mode):
            if mode is not None and mode not in self._SCALE_MODES:
                raise ValueError(f"scale_mode must be one of {self._SCALE_MODES} or None, got {mode!r}")
        if not (isinstance(self.n_outlier, int) and self.n_outlier >= 0):
            raise ValueError(f"n_outlier must be a non-negative int, got {self.n_outlier!r}")

    def _value_bits(self) -> int:
        return self.value_bits if self.value_bits is not None else self.bits

    def _outlier_bits(self) -> int:
        return self.outlier_bits if self.outlier_bits is not None else self.bits + 1

    def _scale_mode_for(self, role: str) -> str:
        explicit = self.key_scale_mode if role == "k" else self.value_scale_mode
        return resolve_scale_mode(explicit, self.ls_scale, role)

    def _resolved_mode(self, d: int, block_size: int) -> str:
        """Resolve ``"auto"`` to a concrete rotation given the block size."""
        if self.rotation_mode != "auto":
            return self.rotation_mode
        if block_size > 1:
            return "hybrid" if is_power_of_two(d) else "dense"
        if is_power_of_two(d):
            return "hadamard_gemm" if d <= self.hadamard_gemm_max_dim else "hadamard"
        return "dense"

    # --- rotation ----------------------------------------------------------
    def _rotate(self, x: torch.Tensor, m: int, block_size: int):
        """Rotate ``x`` (``..., m``); returns ``(x_rotated, rot_tag)``.

        Any rotation that structurally requires a power-of-two ``m``
        (``"hybrid"``, ``"hadamard"``, ``"hadamard_gemm"``) transparently
        falls back to the dense QR rotation when ``m`` is not a power of two
        (e.g. an outlier-split subset) -- correctness is never compromised,
        only the specific transform requested.
        """
        mode = self._resolved_mode(m, block_size)
        x = x.float()
        if mode == "hybrid" and is_power_of_two(m):
            # Folded hybrid rotation: the global signed-Hadamard and the
            # per-block dense rotations are pre-composed into ONE orthogonal
            # GEMM (see rotation.hybrid_rotation_matrix), so the hybrid
            # rotation costs exactly as much as a plain dense rotation at
            # runtime -- the Hadamard/Dense interpolation is free.
            w = hybrid_rotation_matrix(m, block_size, self.seed, str(x.device), torch.float32)
            return x @ w, ("y", block_size)
        if mode == "hadamard_gemm" and is_power_of_two(m):
            hs = signed_hadamard_matrix(m, self.seed, str(x.device), torch.float32)
            return x @ hs, ("hg", None)
        if mode == "hadamard" and is_power_of_two(m):
            sign = random_sign(m, self.seed, device=x.device, dtype=torch.float32)
            return rotate(x, sign), ("h", None)
        q = _dense_rotation_on(m, self.seed, str(x.device), torch.float32)
        return x @ q, ("d", None)

    def _unrotate(self, xr: torch.Tensor, m: int, rot) -> torch.Tensor:
        kind, bs = rot
        if kind == "y":
            w = hybrid_rotation_matrix(m, bs, self.seed, str(xr.device), torch.float32)
            return xr @ w.t()
        if kind == "hg":
            hs = signed_hadamard_matrix(m, self.seed, str(xr.device), torch.float32)
            return xr @ hs.t()
        if kind == "h":
            sign = random_sign(m, self.seed, device=xr.device, dtype=torch.float32)
            return unrotate(xr, sign)
        q = _dense_rotation_on(m, self.seed, str(xr.device), torch.float32)
        return xr @ q.t()

    # Below this many (blocks * codewords) score products the exact GEMM is a
    # single trivially small kernel launch (decode-step encodes: one new
    # token), so the LUT's multi-kernel gather pipeline only adds overhead.
    _LUT_MIN_WORK = 2_000_000

    def _use_lut(self, block_size: int, bits: int, n_total_blocks: int) -> "tuple[int, int] | None":
        """``(bins, T)`` LUT spec to use for this call, or None for exact search.

        Measured policy (A100, real ``batch*heads*seq`` block counts,
        ``paper/tests/bench_encoder_paths.py``, fp16 candidate distances):

        * Tiny calls (``n_blocks * K`` below :data:`_LUT_MIN_WORK`, i.e. the
          per-decode-step encode of one new token): the exact GEMM is one
          small kernel; the LUT's several-kernel gather pipeline only adds
          launch overhead -- use exact.
        * ``b=2`` (``T=2``) prefill: the LUT wins outright -- 0.26ms vs
          1.5-3.1ms for the exact chunked GEMM at K=64-256 (5.8-11.7x); this
          is what puts 3/4-bit prefill on par with TurboQuant's scalar encode.
        * ``b=4`` prefill: with the fp16 candidate-distance step the LUT now
          wins at *every* codebook size -- 1.21ms vs 1.56ms at the default
          2-bit K=256 (1.3x; it used to lose at 5.4ms with fp32 gathers,
          which is why older revisions excluded ``b>=4, K<=512``), and
          1.80ms vs 19.4ms at the ``max_codebook_bits=12`` 3-bit K=4096
          point (10.8x) -- so the small-K exclusion is gone.
        """
        if block_size <= 1 or self.encoder == "exact":
            return None
        k = 1 << (bits * block_size)
        spec = lut_spec_for(block_size, k)
        if spec is None:
            if self.encoder == "lut":
                raise ValueError(f"encoder='lut' infeasible for block_size={block_size}")
            return None
        if self.encoder == "auto" and n_total_blocks * k < self._LUT_MIN_WORK:
            return None
        return spec

    # --- rotate + block-VQ + role-asymmetric scale for one channel subset --
    def _quant_rotor(self, xsub: torch.Tensor, bits: int, role: str) -> dict:
        m = xsub.shape[-1]
        block_size = resolve_block_size(bits, self.block_size, self.max_codebook_bits, m)
        xr, rot = self._rotate(xsub, m, block_size)
        # One RMS scale per whole subset vector (not per block) -- matches
        # TurboQuant/HALO's storage convention exactly (a single fp16 scale
        # per vector regardless of block_size).
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        device_str = str(xsub.device)
        cb = _block_codebook_on(bits, block_size, self.dist, m, device_str)
        n_blocks = m // block_size
        n_total_blocks = max(1, xr.numel() // block_size)
        lut_spec = self._use_lut(block_size, bits, n_total_blocks)
        lut = (_block_lut_on(bits, block_size, self.dist, m, lut_spec[0], lut_spec[1], device_str)
               if lut_spec is not None else None)

        def _lookup(vec_full):
            vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
            if lut is not None:
                codes_ = quantize_via_lut(vb, lut, lut_spec[0], cb)
                chat_b = lookup_codewords(cb, codes_)
            else:
                codes_, chat_b = quantize_and_lookup(vb, cb)
            return codes_, chat_b.reshape(*vec_full.shape[:-1], m)

        def _requant(z_full):
            codes2, chat2 = _lookup(z_full)
            return chat2, codes2

        codes, chat = _lookup(xr / rms)
        scale, codes2 = fit_scale(xr, chat, self._scale_mode_for(role), rms, requantize_fn=_requant)
        if codes2 is not None:
            codes = codes2
        # ``dist`` is stored (as in TurboQuant) so decode picks the same
        # codebook even if the codec instance is reconfigured between
        # encode and decode.
        payload = {"codes": codes, "scale": scale, "bits": bits, "m": m,
                   "block_size": block_size, "rot": rot, "dist": self.dist}
        if self.use_qjl:
            recon = lookup_codewords(cb, codes).reshape(*codes.shape[:-1], m) * scale
            resid = xr - recon
            payload["qjl_sign"] = (resid >= 0)
            payload["qjl_rms"] = resid.abs().mean(dim=-1, keepdim=True)
        return payload

    def _decode_rotor(self, p: dict, device, dtype) -> torch.Tensor:
        dist = p.get("dist", self.dist)  # pre-fix payloads lack "dist"
        cb = _block_codebook_on(p["bits"], p["block_size"], dist, p["m"], str(device))
        chat = lookup_codewords(cb, p["codes"]).reshape(*p["codes"].shape[:-1], p["m"])
        xr = chat * p["scale"]
        if "qjl_sign" in p:
            sign = torch.where(p["qjl_sign"], 1.0, -1.0).to(xr.dtype)
            xr = xr + sign * p["qjl_rms"]
        return self._unrotate(xr, p["m"], p["rot"])

    # --- full-tensor encode / decode (handles the outlier split) -----------
    def _encode_tensor(self, x: torch.Tensor, bits: int, role: str) -> dict:
        head_dim = x.shape[-1]
        if self.n_outlier <= 0 or self.n_outlier >= head_dim:
            return {"mode": "single", "main": self._quant_rotor(x, bits, role), "head_dim": head_dim}
        # Outlier mixed-precision: top-n_outlier channels by energy over tokens.
        energy = x.float().abs().amax(dim=-2)
        energy = energy.reshape(-1, head_dim).mean(dim=0)
        out_idx = torch.argsort(energy, descending=True)[: self.n_outlier]
        out_idx, _ = torch.sort(out_idx)
        mask = torch.zeros(head_dim, dtype=torch.bool, device=x.device)
        mask[out_idx] = True
        return {
            "mode": "outlier", "head_dim": head_dim, "mask": mask,
            "out": self._quant_rotor(x[..., mask], self._outlier_bits(), role),
            "in": self._quant_rotor(x[..., ~mask], bits, role),
        }

    def _decode_tensor(self, payload: dict, dtype) -> torch.Tensor:
        if payload["mode"] == "single":
            device = payload["main"]["codes"].device
            return self._decode_rotor(payload["main"], device, dtype).to(dtype)
        device = payload["out"]["codes"].device
        head_dim, mask = payload["head_dim"], payload["mask"]
        x_out = self._decode_rotor(payload["out"], device, dtype)
        x_in = self._decode_rotor(payload["in"], device, dtype)
        shape = list(x_out.shape[:-1]) + [head_dim]
        x = torch.empty(shape, device=device, dtype=x_out.dtype)
        x[..., mask] = x_out
        x[..., ~mask] = x_in
        return x.to(dtype)

    def _stored_bits(self, x: torch.Tensor, bits: int) -> int:
        """Bit accounting -- identical to TurboQuant/HALO at the same ``bits``.

        ``block_size`` only changes how the ``bits``-per-coordinate budget is
        allocated (jointly per block vs. independently per coordinate); the
        total number of payload bits is unchanged (``n_blocks * bits *
        block_size == d * bits``), which is what makes the block-VQ
        comparison to TurboQuant an honest equal-storage comparison.
        """
        b, h, s, d = x.shape
        vecs = b * h * s
        if self.n_outlier <= 0 or self.n_outlier >= d:
            payload_bits = x.numel() * bits
            scale_bits = vecs * 16
        else:
            no = self.n_outlier
            payload_bits = vecs * (no * self._outlier_bits() + (d - no) * bits)
            scale_bits = vecs * 2 * 16
        qjl_bits = (x.numel() * 1 + vecs * 16) if self.use_qjl else 0
        return payload_bits + scale_bits + qjl_bits

    def encode(self, key, value, ctx=None) -> CompressedKV:
        kp = self._encode_tensor(key, self.bits, "k")
        vp = self._encode_tensor(value, self._value_bits(), "v")
        stored = self._stored_bits(key, self.bits) + self._stored_bits(value, self._value_bits())
        return CompressedKV(key_q=kp, value_q=vp, meta={"dtype": key.dtype, "stored_bits": int(stored)})

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return self._decode_tensor(compressed.key_q, dtype), self._decode_tensor(compressed.value_q, dtype)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update({
            "dist": self.dist, "value_bits": self.value_bits,
            "block_size": self.block_size, "max_codebook_bits": self.max_codebook_bits,
            "rotation_mode": self.rotation_mode, "encoder": self.encoder,
            "hadamard_gemm_max_dim": self.hadamard_gemm_max_dim,
            "ls_scale": self.ls_scale, "key_scale_mode": self.key_scale_mode,
            "value_scale_mode": self.value_scale_mode, "n_outlier": self.n_outlier,
            "outlier_bits": self.outlier_bits, "use_qjl": self.use_qjl,
        })
        return cfg
