"""TurboTaper: age-graded *rotated* KV quantization (no eviction).

TurboTaper is a research composition of ThinKV/TaperKV-style age grading and
TurboQuant-style rotated quantization. It is intended to test whether their
strengths combine; it does not imply dominance over either baseline:

* **Keep every token** (like :class:`TaperKVPolicy`; unlike ThinKV which evicts)
  -> the answer token is not deleted, although quantization can still make it
  unrecoverable.
* **Age-graded precision** -- a full-precision sink + recent window, then a bit
  ladder decaying to a floor ``b_min > 0`` for old tokens (unlike TurboQuant's
  *uniform* per-token bits) -> at an identical average memory budget, recent
  tokens (which attention weights concentrate on) keep more bits.
* **Rotated Lloyd-Max at every band** -- each age band is quantized by
  TurboQuant's random-rotation + data-free scalar Lloyd-Max codec, which is
  source-matched, instead of TaperKV's per-token *affine* INT quantizer.
  Source-law distortion bounds do not order errors on arbitrary real KV or
  downstream attention, so iso-bit and iso-memory comparisons are empirical.
* **Outlier-channel protection at the floor** (optional) -- the lowest band may
  split off ``n_outlier_floor`` highest-energy channels at ``+1`` bit
  (TurboQuant's 2.5/3.5-bit trick), where quantization error hurts most.

The schedule (``token_bits``) is inherited unchanged from
:class:`TaperKVPolicy`; only the *quantizer* changes (affine -> rotated
Lloyd-Max), so TurboTaper is a drop-in, directly comparable ablation of TaperKV.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from .._registry import register_kv_policy
from .taperkv import TaperKVPolicy


@register_kv_policy("turbotaper")
@dataclass
class TurboTaperPolicy(TaperKVPolicy):
    """Age-graded precision schedule quantized with a rotated Lloyd-Max codec.

    Adds the TurboQuant rotation knobs on top of :class:`TaperKVPolicy`:

    Attributes:
        dist: ``"gaussian"`` (RMS-normalised, the high-dim limit; default for
            real KV which is not unit-norm) or ``"beta"`` (exact Beta codebook).
        rotation_mode: ``"hadamard"`` (fast WHT, power-of-two head_dim) or
            ``"dense"`` (QR random orthogonal, any dim).
        seed: deterministic rotation seed.
        n_outlier_floor: highest-energy channels protected at ``+1`` bit in the
            *floor* band only (0 = uniform floor, exact bit accounting).
    """

    dist: str = "gaussian"
    rotation_mode: str = "hadamard"
    seed: int = 0
    n_outlier_floor: int = 0

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "name": "turbotaper",
                "dist": self.dist,
                "rotation_mode": self.rotation_mode,
                "seed": self.seed,
                "n_outlier_floor": self.n_outlier_floor,
            }
        )
        return cfg
