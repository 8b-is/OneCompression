"""TurboTaper: age-graded *rotated* KV quantization (no eviction).

TurboTaper is the proposed method designed to dominate **both** ThinKV
(arXiv:2510.01290, thought-adaptive eviction+quant) and TurboQuant
(arXiv:2504.19874, uniform rotated Lloyd-Max) by combining their strengths and
removing their respective weaknesses:

* **Keep every token** (like :class:`TaperKVPolicy`; unlike ThinKV which evicts)
  -> long-context retrieval (Needle / RULER) never loses the answer token.
* **Age-graded precision** -- a full-precision sink + recent window, then a bit
  ladder decaying to a floor ``b_min > 0`` for old tokens (unlike TurboQuant's
  *uniform* per-token bits) -> at an identical average memory budget, recent
  tokens (which attention weights concentrate on) keep more bits.
* **Rotated Lloyd-Max at every band** -- each age band is quantized by
  TurboQuant's random-rotation + data-free scalar Lloyd-Max codec, which is
  MSE-optimal, instead of TaperKV's per-token *affine* INT quantizer.  At an
  identical bit-width the rotated reconstruction error is strictly lower (the
  Theorem-1 bound verified in ``13_turboquant_longbench_verify.py`` T2), so
  TurboTaper >= TaperKV at iso-bit and >= TurboQuant at iso-memory.
* **Outlier-channel protection at the floor** (optional) -- the lowest band may
  split off ``n_outlier_floor`` highest-energy channels at ``+1`` bit
  (TurboQuant's 2.5/3.5-bit trick), where quantization error hurts most.

The schedule (``token_bits``) is inherited unchanged from
:class:`TaperKVPolicy`; only the *quantizer* changes (affine -> rotated
Lloyd-Max), so TurboTaper is a drop-in, directly-comparable upgrade of TaperKV.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        cfg.update({
            "name": "turbotaper", "dist": self.dist,
            "rotation_mode": self.rotation_mode, "seed": self.seed,
            "n_outlier_floor": self.n_outlier_floor,
        })
        return cfg
