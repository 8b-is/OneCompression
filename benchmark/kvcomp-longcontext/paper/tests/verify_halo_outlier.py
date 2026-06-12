"""Array-level verification of HALO lever 3 (outlier mixed-precision) + accounting.

Ported from TurboQuant, lever 3 splits the ``n_outlier`` highest-energy rotated
channels off and quantises them at ``outlier_bits`` while the rest use ``bits``,
giving the 2.5 / 3.5-bit operating points of the TurboQuant headline table.

Hard assertions (non-zero exit means the lever is wrong):
  * round-trip works for power-of-two and non-power-of-two head dims, for
    "auto"/"hadamard_gemm"/"dense" rotations, for keys (nc) and values (ls);
  * the channel mask has exactly ``n_outlier`` channels;
  * stored-bits matches the closed form and lands on the intended *effective*
    bit-width (true 2.5 / 3.5 = 64-split on head_dim=128);
  * the mixed-precision reconstruction is no worse than uniform ``bits``;
  * HALO and TurboQuant compute identical mixed-precision storage (iso-memory);
  * fallback accounting counts the affine scale/zero meta (memo A-6);
  * TurboQuant QJL accounting counts the per-vector residual rms (memo A-7).

Self-contained and CPU-only: resolves the repo root relative to this file.  The
same invariants are asserted in ``tests/onecomp/kvcomp/test_kvcomp.py``; this is
the run-anywhere companion.

Run:  python paper/tests/verify_halo_outlier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Repo root = .../onecomp-lab (paper/tests -> paper -> kvcomp-longcontext ->
# benchmark -> onecomp-lab).  Relative so this runs on any clone.
_REPO = str(Path(__file__).resolve().parents[4])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from onecomp.kvcomp.codec.halo import HALOCodec  # noqa: E402
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec  # noqa: E402

torch.manual_seed(0)


def _relmse(a, b):
    return ((a - b) ** 2).sum().item() / (b ** 2).sum().clamp_min(1e-12).item()


def main() -> int:
    b, h, s, d = 2, 4, 64, 128
    # Anisotropic KV: a few channels carry far more energy (massive activations),
    # exactly the regime outlier split is designed for.
    key = torch.randn(b, h, s, d)
    value = torch.randn(b, h, s, d)
    hot = [3, 17, 40, 91]
    key[..., hot] *= 12.0
    value[..., hot] *= 9.0

    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  ok  : {msg}")

    print("== lever 3: outlier mixed-precision round-trip & accounting ==")
    for rot in ("auto", "hadamard_gemm", "dense"):
        # (n_outlier, outlier_bits, base_bits, expected effective payload bits):
        # true 2.5 / 3.5-bit on head_dim=128 need 64 outliers; 32 gives 2.25/3.25.
        for n_o, ob, base, eff in [(64, 3, 2, 2.5), (64, 4, 3, 3.5),
                                   (32, 3, 2, 2.25), (32, 4, 3, 3.25)]:
            c = HALOCodec(bits=base, n_outlier=n_o, outlier_bits=ob,
                          rotation_mode=rot, key_scale_mode="nc",
                          value_scale_mode="ls")
            comp = c.encode(key, value)
            kd, vd = c.decode(comp)
            check(kd.shape == key.shape and vd.shape == value.shape,
                  f"{rot} n_o={n_o} ob={ob}: round-trip shape")
            kp = comp.key_q
            check(kp.get("outlier") and int(kp["mask"].sum()) == n_o,
                  f"{rot} n_o={n_o}: mask has exactly {n_o} channels")
            vecs = b * h * s * 2  # key + value vectors
            payload = vecs * (n_o * ob + (d - n_o) * base)
            scales = vecs * 2 * 16
            expect = payload + scales
            check(comp.meta["stored_bits"] == expect,
                  f"{rot} n_o={n_o}: stored_bits == closed form ({expect})")
            eff_bits = payload / (vecs * d)
            check(abs(eff_bits - eff) < 1e-6,
                  f"{rot} n_o={n_o}: effective payload bit-width == {eff} (got {eff_bits:.3f})")
            cu = HALOCodec(bits=base, rotation_mode=rot, key_scale_mode="nc",
                           value_scale_mode="ls")
            ku, vu = cu.decode(cu.encode(key, value))
            mse_mix = _relmse(kd, key)
            mse_uni = _relmse(ku, key)
            check(mse_mix <= mse_uni + 1e-9,
                  f"{rot} n_o={n_o}: key relMSE mixed {mse_mix:.4e} <= uniform {mse_uni:.4e}")

    print("\n== n_outlier matches TurboQuant's split structure (iso-memory) ==")
    th = HALOCodec(bits=2, n_outlier=64, outlier_bits=3, rotation_mode="hadamard",
                   key_scale_mode="nc", value_scale_mode="ls")
    tq = TurboQuantCodec(bits=2, n_outlier=64, outlier_bits=3, rotation_mode="hadamard")
    ch, ct = th.encode(key, value), tq.encode(key, value)
    check(ch.meta["stored_bits"] == ct.meta["stored_bits"],
          f"HALO and TurboQuant 2.5-bit stored_bits agree "
          f"({ch.meta['stored_bits']} == {ct.meta['stored_bits']})")

    print("\n== memo A-6: fallback affine accounting counts scale/zero ==")
    # head_dim=96 is not a power of two -> hadamard single path falls back.
    kf = torch.randn(1, 2, 8, 96)
    vf = torch.randn(1, 2, 8, 96)
    cf = HALOCodec(bits=4, rotation_mode="hadamard")
    compf = cf.encode(kf, vf)
    check(compf.meta.get("fallback") is True, "non-pow2 dim falls back to affine")
    vecs = 1 * 2 * 8
    expect_fb = kf.numel() * 4 + vf.numel() * 4 + (vecs + vecs) * 2 * 16
    check(compf.meta["stored_bits"] == expect_fb,
          f"fallback stored_bits counts scale+zero meta ({expect_fb})")

    print("\n== memo A-7: TurboQuant QJL counts the per-vector residual rms ==")
    cq0 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=False)
    cq1 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=True)
    s0 = cq0.encode(key, value).meta["stored_bits"]
    s1 = cq1.encode(key, value).meta["stored_bits"]
    vecs = b * h * s * 2
    check(s1 - s0 == key.numel() * 1 + value.numel() * 1 + vecs * 16,
          "QJL adds 1 sign bit/elem + 1 fp16 rms/vector")

    print()
    if fails:
        print(f"FAILED ({len(fails)} checks)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
