"""Array-level verification of the HALO role-asymmetric scale + stateless precond.

This exhaustively checks, *element by element* on real tensors, the maths behind
the HALO levers in ``onecomp/kvcomp/codec/halo.py``:

  Lever 1 (role-asymmetric scale).  For a rotated vector ``x`` and the chosen code
    direction ``c`` (``theta = angle(x, c)``):
      LS : s = <x,c>/<c,c>     ||x_hat|| = ||x|| cos t   <x_hat,x> = ||x||^2 cos^2 t   MSE = ||x||^2 sin^2 t
      NC : s = ||x||/||c||     ||x_hat|| = ||x||  (exact) <x_hat,x> = ||x||^2 cos t     MSE = 2||x||^2 (1-cos t)
      IP : s = ||x||^2/<x,c>   <x_hat,x> = ||x||^2 (exact)
    Identities checked: s_NC == sqrt(s_LS * s_IP); MSE_NC - MSE_LS == ||x||^2 (1-cos t)^2
    (=> O(theta^4), negligible at 3-4 bit); LS is the elementwise MSE minimiser.

  Lever 2 (ls_iter).  One Lloyd alternating-minimisation step is monotone
    non-increasing in ||x - s c||^2.

  Lever 4 (stateless block-wise centering).  Removing the EMA makes encoding a
    block independent of any previously-encoded block (no cross-layer / request
    contamination); the stored per-block mu equals the block mean exactly and the
    centre/whiten round-trip is exact.

Plus: storage neutrality across all scale modes, legacy-default back-compat,
and full encode/decode round-trip on every (rotation, bits, mode).

Self-contained and CPU-only: it resolves the repo root relative to this file (no
external checkout required) and every check is a hard assertion; a non-zero exit
means a lever's maths is wrong.  The same invariants are also asserted in the
pytest suite (``tests/onecomp/kvcomp/test_kvcomp.py``); this script is the
human-readable, run-anywhere companion.

Run:  python paper/tests/verify_halo_asym_scale.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Repo root = .../onecomp-lab (paper/tests -> paper -> kvcomp-longcontext ->
# benchmark -> onecomp-lab).  Relative so this runs on any clone.
_REPO = str(Path(__file__).resolve().parents[4])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from onecomp.kvcomp.codec.halo import HALOCodec, _gaussian_codebook  # noqa: E402
from onecomp.kvcomp.codec.rotation import quantize_to_codebook  # noqa: E402

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

_FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# Helpers that reproduce the maths independently of the library.
# ---------------------------------------------------------------------------
def manual_codes(xr, cb):
    rms = xr.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    return quantize_to_codebook(xr / rms, cb), rms


def scales(xr, chat):
    n_x = xr.norm(dim=-1, keepdim=True)
    n_c = chat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ip = (xr * chat).sum(-1, keepdim=True)
    s_ls = ip / (chat * chat).sum(-1, keepdim=True).clamp_min(1e-12)
    s_nc = n_x / n_c
    s_ip = n_x.pow(2) / ip.clamp_min(1e-12)
    cos = (ip / (n_x * n_c).clamp_min(1e-12)).clamp(-1, 1)
    return s_ls, s_nc, s_ip, n_x, cos


# ===========================================================================
def section_1_closed_form_identities():
    print("\n[1] Closed-form scale identities (rotated Gaussian KV, dim=128)")
    cdc = HALOCodec(bits=4, rotation_mode="hadamard_gemm")
    x = torch.randn(1, 4, 512, 128) * (1 + 5 * (torch.arange(128) < 8).float())  # anisotropic
    for bits in (2, 3, 4):
        cb = _gaussian_codebook(bits)
        xr, _ = cdc._rotate(x, 128)              # exact library rotation
        codes, rms = manual_codes(xr, cb)
        chat = cb[codes]
        s_ls, s_nc, s_ip, n_x, cos = scales(xr, chat)

        # (a) geometric-mean identity s_NC = sqrt(s_LS * s_IP)
        gm = (s_ls * s_ip).clamp_min(0).sqrt()
        check(f"bits={bits}: s_NC == sqrt(s_LS*s_IP)", rel(s_nc, gm) < 1e-5,
              f"max rel err {rel(s_nc, gm):.2e}")

        # (b) library _fit_scale matches the closed forms
        ls_lib, _ = cdc._fit_scale(xr, cb, codes, rms, "ls")
        nc_lib, _ = cdc._fit_scale(xr, cb, codes, rms, "nc")
        ip_lib, _ = cdc._fit_scale(xr, cb, codes, rms, "ip")
        rms_lib, _ = cdc._fit_scale(xr, cb, codes, rms, "rms")
        check(f"bits={bits}: _fit_scale('ls')==<x,c>/<c,c>", rel(ls_lib, s_ls) < 1e-6)
        check(f"bits={bits}: _fit_scale('nc')==||x||/||c||", rel(nc_lib, s_nc) < 1e-6)
        check(f"bits={bits}: _fit_scale('ip')==||x||^2/<x,c>", rel(ip_lib, s_ip) < 1e-6)
        check(f"bits={bits}: _fit_scale('rms')==rms", rel(rms_lib, rms) < 1e-12)

        # (c) norm + inner-product behaviour
        xh_ls, xh_nc, xh_ip = s_ls * chat, s_nc * chat, s_ip * chat
        nrm = lambda t: t.norm(dim=-1, keepdim=True)
        ipf = lambda t: (t * xr).sum(-1, keepdim=True)
        check(f"bits={bits}: ||x_hat_NC|| == ||x|| (exact)", rel(nrm(xh_nc), n_x) < 1e-5)
        check(f"bits={bits}: ||x_hat_LS|| == ||x|| cos t", rel(nrm(xh_ls), n_x * cos) < 1e-5)
        check(f"bits={bits}: <x_hat_LS,x> == ||x||^2 cos^2", rel(ipf(xh_ls), n_x.pow(2) * cos.pow(2)) < 1e-5)
        check(f"bits={bits}: <x_hat_NC,x> == ||x||^2 cos",   rel(ipf(xh_nc), n_x.pow(2) * cos) < 1e-5)
        check(f"bits={bits}: <x_hat_IP,x> == ||x||^2 (unbiased)", rel(ipf(xh_ip), n_x.pow(2)) < 1e-4)

        # (d) MSE ordering and the (1-cos)^2 gap
        mse = lambda xh: (xr - xh).pow(2).sum(-1, keepdim=True)
        m_ls, m_nc, m_ip = mse(xh_ls), mse(xh_nc), mse(xh_ip)
        check(f"bits={bits}: LS minimises MSE (<= NC, IP elementwise)",
              bool((m_ls <= m_nc + 1e-6).all() and (m_ls <= m_ip + 1e-6).all()))
        gap_pred = n_x.pow(2) * (1 - cos).pow(2)
        check(f"bits={bits}: MSE_NC - MSE_LS == ||x||^2 (1-cos)^2", rel(m_nc - m_ls, gap_pred) < 1e-3,
              f"mean cos={float(cos.mean()):.4f}, mean rel-MSE gap={float(((m_nc-m_ls)/m_ls.clamp_min(1e-9)).mean()):.4%}")


def section_2_ls_iter_monotone():
    print("\n[2] ls_iter is monotone non-increasing in reconstruction MSE")
    cdc = HALOCodec(bits=2, rotation_mode="dense")
    x = torch.randn(1, 2, 1024, 128)
    for bits in (2, 3, 4):
        cb = _gaussian_codebook(bits)
        xr, _ = cdc._rotate(x, 128)
        codes, rms = manual_codes(xr, cb)
        s_ls, c_ls = cdc._fit_scale(xr, cb, codes, rms, "ls")
        s_it, c_it = cdc._fit_scale(xr, cb, codes, rms, "ls_iter")
        m_ls = (xr - s_ls * cb[c_ls]).pow(2).sum()
        m_it = (xr - s_it * cb[c_it]).pow(2).sum()
        check(f"bits={bits}: MSE(ls_iter) <= MSE(ls)", float(m_it) <= float(m_ls) + 1e-4,
              f"ls={float(m_ls):.3f} -> ls_iter={float(m_it):.3f} "
              f"({(float(m_ls)-float(m_it))/float(m_ls):.3%} lower)")


def section_3_storage_neutral_and_roundtrip():
    print("\n[3] Storage neutrality + full encode/decode round-trip across modes")
    x = torch.randn(1, 4, 256, 128)
    modes = ["rms", "ls", "nc", "ip", "ls_iter"]
    for rot in ("hadamard_gemm", "dense"):
        for bits in (2, 3, 4):
            stored = []
            for km in modes:
                for vm in modes:
                    c = HALOCodec(bits=bits, rotation_mode=rot,
                                  key_scale_mode=km, value_scale_mode=vm)
                    enc = c.encode(x, x)
                    kq, vq = c.decode(enc)
                    stored.append(enc.meta["stored_bits"])
                    if (km, vm) == ("nc", "ls"):
                        check(f"{rot} b{bits} K={km}/V={vm}: shapes preserved",
                              kq.shape == x.shape and vq.shape == x.shape)
            check(f"{rot} b{bits}: stored_bits identical across all 25 mode combos",
                  len(set(stored)) == 1, f"distinct={set(stored)}")


def section_4_legacy_backcompat():
    print("\n[4] Legacy back-compat: ls_scale derives modes, smart defaults")
    x = torch.randn(1, 2, 128, 128)
    # Default (ls_scale=True, explicit modes=None) -> smart NC+ls_iter.
    d = HALOCodec(bits=4)
    check("default resolves K=nc, V=ls_iter",
          d._scale_mode_for("k") == "nc" and d._scale_mode_for("v") == "ls_iter")
    # ls_scale=False, no explicit modes -> RMS on both (TurboQuant-faithful).
    r = HALOCodec(bits=4, ls_scale=False)
    check("ls_scale=False resolves K=rms, V=rms",
          r._scale_mode_for("k") == "rms" and r._scale_mode_for("v") == "rms")
    # Explicit mode overrides ls_scale for that role only.
    o = HALOCodec(bits=4, ls_scale=False, key_scale_mode="nc")
    check("explicit key_scale_mode overrides ls_scale",
          o._scale_mode_for("k") == "nc" and o._scale_mode_for("v") == "rms")
    # Symmetric LS is available via explicit modes (legacy behaviour).
    a = HALOCodec(bits=3, rotation_mode="hadamard_gemm",
                  key_scale_mode="ls", value_scale_mode="ls")
    b = HALOCodec(bits=3, rotation_mode="hadamard_gemm",
                  key_scale_mode="ls", value_scale_mode="ls")
    ka, _ = a.decode(a.encode(x, x)); kb, _ = b.decode(b.encode(x, x))
    check("explicit K=ls/V=ls is deterministic (bit-identical)", rel(ka, kb) < 1e-12)


def section_5_stateless_precond():
    print("\n[5] Stateless block-wise centering (EMA removed)")
    cdc = HALOCodec(bits=4, rotation_mode="dense", precond_roles=("k", "v"),
                    center=True, whiten=True)
    check("no _ema state attribute on codec", not hasattr(cdc, "_ema"))

    d = 128
    blockA = torch.randn(1, 4, 300, d)
    means = torch.zeros(d); means[:6] = torch.tensor([50., -40., 30., -20., 60., -10.])
    blockB = torch.randn(1, 4, 300, d) * 3.0 + means

    encB_after_A = cdc.encode(blockA, blockA) and cdc.encode(blockB, blockB)
    fresh = HALOCodec(bits=4, rotation_mode="dense", precond_roles=("k", "v"),
                      center=True, whiten=True)
    encB_fresh = fresh.encode(blockB, blockB)

    same_codes = torch.equal(encB_after_A.key_q["codes"], encB_fresh.key_q["codes"])
    same_scale = rel(encB_after_A.key_q["scale"], encB_fresh.key_q["scale"]) < 1e-10
    check("encode(B) independent of prior encode(A) -- codes", same_codes)
    check("encode(B) independent of prior encode(A) -- scale", same_scale)

    mu_stored = encB_fresh.key_q["mu"]
    mu_true = blockB.float().reshape(-1, d).mean(0)
    check("stored mu == block B channel mean (no EMA mixing)", rel(mu_stored, mu_true) < 1e-4,
          f"max|mu_stored-mu_true|={float((mu_stored-mu_true).abs().max()):.2e}")

    kq, vq = fresh.decode(encB_fresh)
    err = float((kq - blockB).pow(2).mean() / blockB.pow(2).mean())
    check("center+whiten round-trip relMSE reasonable", math.isfinite(err) and err < 0.2,
          f"relMSE={err:.4f}")

    co = HALOCodec(bits=4, rotation_mode="dense", precond_roles=("k",), center=True)
    e = co.encode(blockB, blockB)
    kq2, _ = co.decode(e)
    rec_mean = kq2.float().reshape(-1, d).mean(0)[:6]
    check("un-center restores massive-activation channel means", rel(rec_mean, mu_true[:6]) < 1e-2,
          f"rec={rec_mean.tolist()}")


def section_6_attention_logit_fidelity():
    print("\n[6] Attention-logit fidelity: NC keys vs LS keys (same storage)")
    torch.manual_seed(3)
    H, S, D = 8, 1024, 128
    K = torch.randn(1, H, S, D) * (1 + 4 * (torch.arange(D) < 8).float())
    V = torch.randn(1, H, S, D)
    Q = torch.randn(1, H, 64, D)
    sc = 1.0 / math.sqrt(D)

    def attn(k, v):
        return torch.softmax((Q @ k.transpose(-1, -2)) * sc, -1) @ v

    o_fp = attn(K, V)
    for bits in (2, 3, 4):
        c_nc = HALOCodec(bits=bits, rotation_mode="hadamard_gemm", key_scale_mode="nc", value_scale_mode="ls")
        c_ls = HALOCodec(bits=bits, rotation_mode="hadamard_gemm", key_scale_mode="ls", value_scale_mode="ls")
        knc, vnc = c_nc.decode(c_nc.encode(K, V))
        kls, vls = c_ls.decode(c_ls.encode(K, V))
        check(f"bits={bits}: V identical for K=nc vs K=ls", rel(vnc, vls) < 1e-12)
        e_nc = float((attn(knc, vnc) - o_fp).norm() / o_fp.norm())
        e_ls = float((attn(kls, vls) - o_fp).norm() / o_fp.norm())
        mse_k_nc = float((knc - K).pow(2).mean())
        mse_k_ls = float((kls - K).pow(2).mean())
        check(f"bits={bits}: key MSE(LS) <= key MSE(NC) (MSE optimality holds)",
              mse_k_ls <= mse_k_nc + 1e-6, f"LS={mse_k_ls:.5f} NC={mse_k_nc:.5f}")
        print(f"      bits={bits}: attn-err  NC={e_nc:.5f}  LS={e_ls:.5f}  "
              f"({'NC better' if e_nc < e_ls else 'LS better'} by {abs(e_nc-e_ls)/max(e_ls,1e-9):.2%})")


def main():
    print("=" * 78)
    print("HALO role-asymmetric scale + stateless precond: array-level verification")
    print("=" * 78)
    section_1_closed_form_identities()
    section_2_ls_iter_monotone()
    section_3_storage_neutral_and_roundtrip()
    section_4_legacy_backcompat()
    section_5_stateless_precond()
    section_6_attention_logit_fidelity()
    print("\n" + "=" * 78)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): " + "; ".join(_FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
