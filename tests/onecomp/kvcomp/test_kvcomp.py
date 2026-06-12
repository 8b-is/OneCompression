"""Unit tests for the KV-cache compression subpackage (``onecomp.kvcomp``).

These are CPU-only, model-free correctness tests covering the codec math
(rotation invertibility, Lloyd-Max nearest-centroid, affine error bound), the
graded-cache *bit accounting* (which must be a whole-cache figure consistent
with the per-token codec baseline), the age/recency precision schedules, the
ThinKV TBQ/TBE state machine, the token selectors, and the vLLM dtype mapping.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp.kvcomp import list_codecs, list_selectors, list_policies, build_codec
from onecomp.kvcomp.codec._quant_primitives import (
    fake_quantize_affine,
    effective_bits,
)
from onecomp.kvcomp.codec.rotation import (
    walsh_hadamard_transform,
    random_sign,
    rotate,
    unrotate,
    random_orthogonal,
    hadamard_matrix,
    lloyd_max_gaussian,
    lloyd_max_beta,
    quantize_to_codebook,
)
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.codec.halo import HALOCodec
from onecomp.kvcomp.codec.affine import AffineKVCodec, Int4Codec, Int3Codec, KIVICodec
from onecomp.kvcomp.codec.fp8 import FP8Codec
from onecomp.kvcomp.codec.lowbit_fp import NVFP4Codec, TernaryCodec
from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.cache.taper_cache import TaperKVCache, taper_bits
from onecomp.kvcomp.cache.pmkvq_cache import PMKVQCache, progressive_bits
from onecomp.kvcomp.cache.turbotaper_cache import TurboTaperCache
from onecomp.kvcomp.recipe.taperkv import TaperKVPolicy
from onecomp.kvcomp.recipe.pmkvq import PMKVQPolicy
from onecomp.kvcomp.recipe.turbotaper import TurboTaperPolicy
from onecomp.kvcomp.recipe.thinkv import ThinKVEngine, ThoughtClassifier, T_REASONING, T_TRANSITION
from onecomp.kvcomp.select.snapkv import SnapKVSelector
from onecomp.kvcomp.select.h2o import H2OSelector
from onecomp.kvcomp.config import Budget


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- registry -------------------------------------------------------------
def test_registry_lists():
    assert {
        "fp8",
        "int4",
        "int3",
        "int2",
        "kivi",
        "turboquant",
        "halo",
        "nvfp4",
        "ternary",
    }.issubset(set(list_codecs()))
    assert {"snapkv", "h2o", "streamingllm", "recent", "think"}.issubset(set(list_selectors()))
    assert {"static", "taperkv", "turbotaper", "pmkvq", "thinkv"}.issubset(set(list_policies()))


# --- rotation primitives --------------------------------------------------
def test_wht_is_orthonormal_involution():
    x = torch.randn(5, 16)
    w = walsh_hadamard_transform(x)
    assert torch.allclose(walsh_hadamard_transform(w), x, atol=1e-5)
    assert torch.allclose(x.norm(), w.norm(), atol=1e-4)


def test_rotate_unrotate_roundtrip():
    x = torch.randn(3, 8)
    sign = random_sign(8, 0, device=x.device)
    assert torch.allclose(unrotate(rotate(x, sign), sign), x, atol=1e-5)


def test_random_orthogonal():
    q = random_orthogonal(16, 0)
    assert torch.allclose(q @ q.t(), torch.eye(16), atol=1e-5)


def test_codebook_nearest_matches_bruteforce():
    cb = lloyd_max_gaussian(3)
    z = torch.linspace(-3, 3, 100)
    idx = quantize_to_codebook(z, cb)
    brute = (z[:, None] - cb[None, :]).abs().argmin(1)
    assert torch.equal(idx, brute)


def test_lloyd_max_sizes_and_sorted():
    assert lloyd_max_gaussian(4).numel() == 16
    assert bool((lloyd_max_gaussian(4).diff() > 0).all())
    assert lloyd_max_beta(3, 128).numel() == 8


# --- affine primitive -----------------------------------------------------
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_affine_error_within_half_step(bits):
    x = torch.randn(2, 4, 32, 64)
    xq = fake_quantize_affine(x, bits, axis="token")
    rng = x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)
    step = (rng / ((1 << bits) - 1)).squeeze(-1)
    maxerr = (xq - x).abs().amax(-1)
    assert bool((maxerr <= step * 0.5 + 1e-3).all())


def test_effective_bits_overhead():
    assert abs(effective_bits(4, -1, 128) - (4 + (16 + 4) / 128)) < 1e-6


# --- codec round-trips ----------------------------------------------------
def _kv(b=1, h=8, s=128, d=128):
    x = torch.randn(b, h, s, d)
    return x, x.clone()


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_turboquant_mse_decreases_with_bits(bits):
    x, _ = _kv()
    c = TurboQuantCodec(bits=bits, rotation_mode="hadamard")
    mse = float((c.decode(c.encode(x, x))[0] - x).pow(2).mean())
    assert 0.0 < mse < 1.0


def test_turboquant_bits_monotone():
    x, _ = _kv()
    mses = []
    for b in (2, 3, 4):
        c = TurboQuantCodec(bits=b, rotation_mode="hadamard")
        mses.append(float((c.decode(c.encode(x, x))[0] - x).pow(2).mean()))
    assert mses[0] > mses[1] > mses[2]


def test_turboquant_dense_matches_hadamard_quality():
    x, _ = _kv()
    mh = float(
        (
            TurboQuantCodec(bits=4, rotation_mode="hadamard").decode(
                TurboQuantCodec(bits=4, rotation_mode="hadamard").encode(x, x)
            )[0]
            - x
        )
        .pow(2)
        .mean()
    )
    md = float(
        (
            TurboQuantCodec(bits=4, rotation_mode="dense").decode(
                TurboQuantCodec(bits=4, rotation_mode="dense").encode(x, x)
            )[0]
            - x
        )
        .pow(2)
        .mean()
    )
    assert abs(mh - md) / max(mh, md) < 0.2


def test_turboquant_outlier_improves_over_uniform():
    x, _ = _kv()
    co = TurboQuantCodec(bits=2, n_outlier=32, outlier_bits=4, rotation_mode="hadamard")
    mo = float((co.decode(co.encode(x, x))[0] - x).pow(2).mean())
    cu = TurboQuantCodec(bits=2, rotation_mode="hadamard")
    mu = float((cu.decode(cu.encode(x, x))[0] - x).pow(2).mean())
    assert mo < mu


def test_halo_precond_off_matches_turboquant_quality():
    x, _ = _kv()
    h = HALOCodec(bits=4, rotation_mode="dense")
    assert float((h.decode(h.encode(x, x))[0] - x).pow(2).mean()) < 1.0


def test_halo_non_pow2_hadamard_falls_back():
    x = torch.randn(1, 2, 16, 96)  # 96 not power of two
    h = HALOCodec(bits=4, rotation_mode="hadamard")
    enc = h.encode(x, x)
    assert enc.meta.get("fallback") or enc.meta.get("passthrough")
    assert h.decode(enc)[0].shape == x.shape


def test_hadamard_matrix_orthonormal():
    for d in (8, 64, 128, 256):
        h = hadamard_matrix(d, "cpu", torch.float32)
        assert torch.allclose(h @ h.t(), torch.eye(d), atol=1e-5)


def test_hadamard_gemm_equals_wht():
    # (x*sign) @ H is mathematically the randomized Walsh-Hadamard transform.
    for d in (64, 128, 256):
        x = torch.randn(50, d)
        sign = random_sign(d, 0, dtype=torch.float32)
        h = hadamard_matrix(d, "cpu", torch.float32)
        assert torch.allclose((x * sign) @ h, rotate(x, sign), atol=1e-4)


def test_halo_auto_resolves_hadamard_gemm_for_pow2():
    h = HALOCodec(bits=4, rotation_mode="auto")
    assert h._resolved_mode(128) == "hadamard_gemm"
    assert h._resolved_mode(256) == "hadamard_gemm"
    assert h._resolved_mode(96) == "dense"  # non power of two
    # Above the matrix-size guard it falls back to the butterfly WHT.
    assert HALOCodec(bits=4, hadamard_gemm_max_dim=64)._resolved_mode(128) == "hadamard"


def test_halo_hadamard_gemm_matches_wht_quality():
    # The GEMM Hadamard must reconstruct identically to the butterfly WHT and to
    # TurboQuant's hadamard mode (it is the SAME transform, speed lever only).
    # Use RMS scale on both HALO variants to isolate the rotation effect.
    x, _ = _kv()
    rms_kw = dict(key_scale_mode="rms", value_scale_mode="rms")
    hg = HALOCodec(bits=4, rotation_mode="hadamard_gemm", **rms_kw)
    hw = HALOCodec(bits=4, rotation_mode="hadamard", **rms_kw)
    tq = TurboQuantCodec(bits=4, rotation_mode="hadamard")
    m_hg = float((hg.decode(hg.encode(x, x))[0] - x).pow(2).mean())
    m_hw = float((hw.decode(hw.encode(x, x))[0] - x).pow(2).mean())
    m_tq = float((tq.decode(tq.encode(x, x))[0] - x).pow(2).mean())
    assert abs(m_hg - m_hw) / max(m_hg, m_hw) < 1e-3
    assert abs(m_hg - m_tq) / max(m_hg, m_tq) < 1e-3


def test_halo_ls_scale_reduces_mse_and_is_storage_neutral():
    # The least-squares dequant scale minimises ||xr - s*c_hat||^2, so it can only
    # lower reconstruction MSE; storage (one fp16/vector) is unchanged.
    x, _ = _kv()
    for bits in (2, 3, 4):
        ls = HALOCodec(bits=bits, rotation_mode="hadamard_gemm",
                       key_scale_mode="ls", value_scale_mode="ls")
        rms = HALOCodec(bits=bits, rotation_mode="hadamard_gemm",
                        key_scale_mode="rms", value_scale_mode="rms")
        e_ls, e_rms = ls.encode(x, x), rms.encode(x, x)
        m_ls = float((ls.decode(e_ls)[0] - x).pow(2).mean())
        m_rms = float((rms.decode(e_rms)[0] - x).pow(2).mean())
        assert m_ls <= m_rms + 1e-9
        assert e_ls.meta["stored_bits"] == e_rms.meta["stored_bits"]


# --- HALO role-asymmetric scale modes (lever 1) + ls_iter (lever 2) -------
def test_halo_scale_modes_storage_neutral_and_roundtrip():
    # Every key/value scale-mode combo stores the SAME bits and round-trips.
    x, _ = _kv(h=4, s=128)
    modes = ["rms", "ls", "nc", "ip", "ls_iter"]
    stored = set()
    for km in modes:
        for vm in modes:
            c = HALOCodec(bits=3, rotation_mode="hadamard_gemm",
                          key_scale_mode=km, value_scale_mode=vm)
            enc = c.encode(x, x)
            k, v = c.decode(enc)
            assert k.shape == x.shape and v.shape == x.shape
            stored.add(enc.meta["stored_bits"])
    assert len(stored) == 1  # scale choice is storage-neutral


def test_halo_nc_preserves_rotated_norm():
    # The norm-correcting scale reconstructs each rotated vector at its true norm
    # (||x_hat|| == ||x_rot||), unlike the least-squares scale which shrinks it.
    x, _ = _kv(h=2, s=256)
    c = HALOCodec(bits=4, rotation_mode="hadamard_gemm", key_scale_mode="nc")
    p = c._encode_tensor(x, 4, "k")
    from onecomp.kvcomp.codec.halo import _gaussian_codebook
    cb = _gaussian_codebook(4)
    xr, _ = c._rotate(x, x.shape[-1])
    xhat = cb[p["codes"]] * p["scale"]
    assert torch.allclose(xhat.norm(dim=-1), xr.norm(dim=-1), rtol=1e-4)


def test_halo_ls_iter_monotone_mse():
    # One alternating-minimisation step never raises reconstruction MSE vs LS.
    from onecomp.kvcomp.codec.halo import _gaussian_codebook
    from onecomp.kvcomp.codec.rotation import quantize_to_codebook
    x, _ = _kv(h=2, s=512)
    c = HALOCodec(bits=2, rotation_mode="dense")
    for bits in (2, 3, 4):
        cb = _gaussian_codebook(bits)
        xr, _ = c._rotate(x, x.shape[-1])
        rms = xr.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        codes = quantize_to_codebook(xr / rms, cb)
        s_ls, c_ls = c._fit_scale(xr, cb, codes, rms, "ls")
        s_it, c_it = c._fit_scale(xr, cb, codes, rms, "ls_iter")
        m_ls = (xr - s_ls * cb[c_ls]).pow(2).sum()
        m_it = (xr - s_it * cb[c_it]).pow(2).sum()
        assert float(m_it) <= float(m_ls) + 1e-3


def test_halo_legacy_ls_scale_backcompat():
    # New defaults: key_scale_mode="nc", value_scale_mode="ls_iter" (validated best).
    x, _ = _kv(h=2, s=128)
    assert HALOCodec(bits=4)._scale_mode_for("k") == "nc"        # new default
    assert HALOCodec(bits=4)._scale_mode_for("v") == "ls_iter"   # new default
    # Explicit modes always override ls_scale.
    o = HALOCodec(bits=4, key_scale_mode="nc", value_scale_mode="rms")
    assert o._scale_mode_for("k") == "nc" and o._scale_mode_for("v") == "rms"
    # ls_scale fallback (only active when explicit modes are None).
    o2 = HALOCodec(bits=4, key_scale_mode=None, value_scale_mode=None, ls_scale=False)
    assert o2._scale_mode_for("k") == "rms" and o2._scale_mode_for("v") == "rms"
    # Explicit ls on both is the old symmetric-LS behaviour.
    a = HALOCodec(bits=3, rotation_mode="hadamard_gemm", key_scale_mode="ls", value_scale_mode="ls")
    b = HALOCodec(bits=3, rotation_mode="hadamard_gemm", key_scale_mode="ls", value_scale_mode="ls")
    assert torch.allclose(a.decode(a.encode(x, x))[0], b.decode(b.encode(x, x))[0])


def test_halo_invalid_scale_mode_raises():
    with pytest.raises(Exception):
        HALOCodec(bits=4, key_scale_mode="bogus")


def test_halo_stateless_precond_no_ema():
    # Lever 4: no EMA state; encoding a block is independent of any prior block.
    x_a = torch.randn(1, 4, 200, 128)
    means = torch.zeros(128); means[:4] = torch.tensor([40.0, -30.0, 20.0, -10.0])
    x_b = torch.randn(1, 4, 200, 128) + means
    c = HALOCodec(bits=4, rotation_mode="dense", precond_roles=("k", "v"), center=True)
    assert not hasattr(c, "_ema")
    _ = c.encode(x_a, x_a)              # would have polluted an EMA
    enc_b_after = c.encode(x_b, x_b)
    fresh = HALOCodec(bits=4, rotation_mode="dense", precond_roles=("k", "v"), center=True)
    enc_b_fresh = fresh.encode(x_b, x_b)
    assert torch.equal(enc_b_after.key_q["codes"], enc_b_fresh.key_q["codes"])
    # stored mu is exactly this block's channel mean (no cross-block mixing)
    mu_true = x_b.float().reshape(-1, 128).mean(0)
    assert torch.allclose(enc_b_fresh.key_q["mu"], mu_true, atol=1e-4)


# --- HALO outlier mixed-precision (lever 3) -------------------------------
@pytest.mark.parametrize("rot", ["auto", "hadamard_gemm", "dense"])
@pytest.mark.parametrize("n_o,ob,base,eff", [(64, 3, 2, 2.5), (64, 4, 3, 3.5),
                                             (32, 3, 2, 2.25)])
def test_halo_outlier_roundtrip_and_accounting(rot, n_o, ob, base, eff):
    x, _ = _kv(h=4, s=64)  # head_dim 128
    b, h, s, d = x.shape
    c = HALOCodec(bits=base, n_outlier=n_o, outlier_bits=ob, rotation_mode=rot,
                  key_scale_mode="nc", value_scale_mode="ls")
    enc = c.encode(x, x)
    k, v = c.decode(enc)
    assert k.shape == x.shape and v.shape == x.shape
    assert enc.key_q.get("outlier") and int(enc.key_q["mask"].sum()) == n_o
    vecs = b * h * s * 2
    payload = vecs * (n_o * ob + (d - n_o) * base)
    assert enc.meta["stored_bits"] == payload + vecs * 2 * 16
    assert abs(payload / (vecs * d) - eff) < 1e-6


def test_halo_outlier_matches_turboquant_storage():
    # HALO and TurboQuant compute identical mixed-precision storage at the same
    # split, so the 2.5/3.5-bit comparison table is iso-memory.
    x, _ = _kv(h=4, s=64)
    hh = HALOCodec(bits=2, n_outlier=64, outlier_bits=3, rotation_mode="hadamard",
                   key_scale_mode="nc", value_scale_mode="ls")
    tq = TurboQuantCodec(bits=2, n_outlier=64, outlier_bits=3, rotation_mode="hadamard")
    assert hh.encode(x, x).meta["stored_bits"] == tq.encode(x, x).meta["stored_bits"]


def test_halo_outlier_beats_uniform_on_anisotropic_kv():
    # With a few massive-activation channels, spending more bits on them lowers
    # reconstruction MSE vs uniform base bits at the SAME nominal base.
    x = torch.randn(2, 4, 64, 128)
    x[..., [3, 17, 40, 91]] *= 12.0
    mix = HALOCodec(bits=2, n_outlier=64, outlier_bits=3, rotation_mode="dense",
                    key_scale_mode="nc")
    uni = HALOCodec(bits=2, rotation_mode="dense", key_scale_mode="nc")
    mse_mix = (mix.decode(mix.encode(x, x))[0] - x).pow(2).sum()
    mse_uni = (uni.decode(uni.encode(x, x))[0] - x).pow(2).sum()
    assert float(mse_mix) <= float(mse_uni)


# --- honest fallback / QJL accounting (memo A-6, A-7) ---------------------
def test_fallback_affine_accounting_counts_scale_zero():
    # Non-power-of-two head dim falls back to per-token affine; the stored-bits
    # must count the scale + zero metadata (two fp16 per vector), not just codes.
    kf = torch.randn(1, 2, 8, 96)
    for codec in (HALOCodec(bits=4, rotation_mode="hadamard"),
                  TurboQuantCodec(bits=4, rotation_mode="hadamard")):
        enc = codec.encode(kf, kf)
        assert enc.meta.get("fallback")
        vecs = 1 * 2 * 8
        assert enc.meta["stored_bits"] == kf.numel() * 4 * 2 + (vecs * 2) * 2 * 16


def test_turboquant_qjl_counts_residual_rms():
    x, _ = _kv(h=4, s=64)
    b, h, s, d = x.shape
    s0 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=False).encode(x, x).meta["stored_bits"]
    s1 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=True).encode(x, x).meta["stored_bits"]
    vecs = b * h * s * 2
    assert s1 - s0 == x.numel() * 2 * 1 + vecs * 16  # sign bits + per-vector fp16 rms


@pytest.mark.parametrize(
    "codec",
    [KIVICodec(bits=2), FP8Codec(), NVFP4Codec(), TernaryCodec(), Int3Codec(), Int4Codec()],
)
def test_codec_roundtrip_shapes_and_bits(codec):
    x, _ = _kv(h=4, s=64)
    enc = codec.encode(x, x)
    k, v = codec.decode(enc)
    assert k.shape == x.shape and v.shape == x.shape
    assert enc.numel_bits() and enc.numel_bits() > 0


# --- validate_params fires at construction (L-3) --------------------------
def test_invalid_params_raise_on_construction():
    with pytest.raises(Exception):
        TurboQuantCodec(rotation_mode="bogus")
    with pytest.raises(Exception):
        AffineKVCodec(bits=99)
    with pytest.raises(Exception):
        FP8Codec(key_axis="bogus")
    assert TurboQuantCodec(bits=4).bits == 4  # valid still constructs


# --- precision schedules --------------------------------------------------
def test_taper_bits_windows_and_floor():
    total = 1000
    age = torch.arange(total - 1, -1, -1)
    tb = taper_bits(age, total, sink_tokens=4, recent_tokens=64, hi_bits=8, b_min=2)
    assert bool((tb[-64:] == 8).all())  # recent
    assert bool((tb[:4] == 8).all())  # sink
    assert int(tb.min()) >= 2 and int(tb.max()) <= 8


def test_taper_bits_explicit_ladder_clamped_to_role():
    # M-1: a key-style ladder reused for the value role must not exceed value_bits.
    age = torch.arange(199, -1, -1)
    vb = taper_bits(age, 200, 4, 8, hi_bits=4, b_min=2, ladder=[8, 7, 6, 5, 4, 3, 2])
    assert int(vb.max()) <= 4 and int(vb.min()) >= 2
    # default ladder is unchanged by the clamp
    a = taper_bits(age, 200, 4, 8, 8, 2)
    b = taper_bits(age, 200, 4, 8, 8, 2, ladder=None)
    assert torch.equal(a, b)


def test_progressive_bits_windows():
    total = 1000
    r = torch.arange(total - 1, -1, -1)
    pb = progressive_bits(r, recent_window=64, ladder_span=256, high_bits=8, final_bits=2)
    assert bool((pb[-64:] == 8).all())
    assert bool((pb[: total - 64 - 256] == 2).all())


# --- graded cache bit accounting (H-1) ------------------------------------
def _meta_affine(bv, heads):
    q = bv < 16
    return float(heads * (16 * int(q.sum()) + int(bv[q].sum())))


@pytest.mark.parametrize("d", [64, 128])
def test_graded_ratio_counts_metadata(d):
    b, h, s = 1, 4, 40
    pol = TaperKVPolicy(
        key_bits=8, value_bits=4, b_min=2, b_min_value=2, sink_tokens=4, recent_tokens=8
    )
    tc = TaperKVCache(pol, requant_stride=1)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    tc.update(k, v, 0)
    age = torch.arange(s - 1, -1, -1)
    kb = taper_bits(age, s, 4, 8, 8, 2)
    vb = taper_bits(age, s, 4, 8, 4, 2)
    payload = float(kb.sum() * h * d + vb.sum() * h * d)
    honest = payload + _meta_affine(kb, h) + _meta_affine(vb, h)
    assert abs(tc.compression_ratio() - (s * h * d * 2 * 16) / honest) < 1e-3
    # strictly below the metadata-free (optimistic) figure
    assert tc.compression_ratio() < (s * h * d * 2 * 16) / payload


def test_uniform_taperkv_matches_compressed_int4_accounting():
    # A uniform 4-bit TaperKV must report the SAME whole-cache ratio as the
    # per-token INT4 codec baseline -- the cross-path accounting consistency
    # that the H-1 fix guarantees.
    b, h, s, d = 1, 4, 48, 128
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    upol = TaperKVPolicy(
        key_bits=4, value_bits=4, b_min=4, b_min_value=4, sink_tokens=0, recent_tokens=0
    )
    ut = TaperKVCache(upol, requant_stride=1)
    ut.update(k.clone(), v.clone(), 0)
    cc = CompressedDynamicCache(
        codec=Int4Codec(key_axis="token", value_axis="token"), sink_tokens=0
    )
    cc.update(k.clone(), v.clone(), 0)
    assert abs(ut.compression_ratio() - cc.compression_ratio()) / cc.compression_ratio() < 0.01


def test_turbotaper_ratio_counts_norm_metadata():
    b, h, s, d = 1, 4, 40, 128
    ttp = TurboTaperPolicy(
        key_bits=8,
        value_bits=4,
        b_min=2,
        b_min_value=2,
        sink_tokens=4,
        recent_tokens=8,
        rotation_mode="hadamard",
    )
    ttc = TurboTaperCache(ttp, requant_stride=1)
    ttc.update(torch.randn(b, h, s, d), torch.randn(b, h, s, d), 0)
    age = torch.arange(s - 1, -1, -1)
    kb = taper_bits(age, s, 4, 8, 8, 2)
    vb = taper_bits(age, s, 4, 8, 4, 2)
    payload = float(kb.sum() * h * d + vb.sum() * h * d)
    norm_meta = float(h * 16 * int((kb < 16).sum()) + h * 16 * int((vb < 16).sum()))
    honest = payload + norm_meta
    assert abs(ttc.compression_ratio() - (s * h * d * 2 * 16) / honest) < 1e-3


# --- CompressedDynamicCache integration -----------------------------------
def test_compressed_cache_prefill_decode_and_sink():
    b, h, s, d = 1, 4, 40, 64
    c = CompressedDynamicCache(
        codec=Int4Codec(key_axis="token", value_axis="token"), sink_tokens=4
    )
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    ok, _ = c.update(k, v, 0)
    assert ok.shape == (b, h, s, d)
    assert torch.allclose(ok[:, :, :4], k[:, :, :4], atol=1e-3)  # sinks exact
    ok, _ = c.update(torch.randn(b, h, 1, d), torch.randn(b, h, 1, d), 0)
    assert ok.shape == (b, h, s + 1, d)
    assert c.compression_ratio() > 1.0


def test_pmkvq_cache_runs():
    b, h, s, d = 1, 4, 48, 128
    pol = PMKVQPolicy(key_bits=8, value_bits=4, final_bits=2, recent_window=8, ladder_span=16)
    pc = PMKVQCache(pol, requant_stride=1)
    ok, _ = pc.update(torch.randn(b, h, s, d), torch.randn(b, h, s, d), 0)
    assert ok.shape == (b, h, s, d)
    assert 1.0 < pc.compression_ratio() < 8.0


# --- ThinKV engine --------------------------------------------------------
def test_thinkv_engine_budget_and_segments():
    h, d = 4, 64
    eng = ThinKVEngine(
        classifier=ThoughtClassifier(),
        group_size=8,
        token_budget=64,
        retention_schedule=(16, 8, 4),
        precision={"R": 4, "E": 4, "T": 2},
    )
    eng.add_prefill(torch.randn(h, 30, d), torch.randn(h, 30, d))
    assert eng.committed_tokens == 30
    for step in range(1, 200):
        thought = T_TRANSITION if (step // 8) % 5 == 0 else T_REASONING
        eng.add_decode_step(
            torch.randn(h, d), torch.randn(h, d), attn_probs=None, step=step, thought=thought
        )
    keys, _ = eng.materialize()
    assert keys.shape[0] == h and keys.shape[-1] == d
    assert eng.committed_tokens <= eng.token_budget + eng.group_size
    assert sum(len(s.indices) for s in eng.segments) == eng.committed_tokens


# --- selectors ------------------------------------------------------------
def test_snapkv_keeps_sink_and_recent():
    h, d = 4, 64
    q = torch.randn(1, h, 40, d)
    k = torch.randn(1, h, 40, d)
    v = torch.randn(1, h, 40, d)
    sel = SnapKVSelector(keep_ratio=0.5, window_size=8, recent_tokens=8, sink_tokens=2)
    scores = sel.score(q, k, v, None, 0)
    assert scores.shape == (1, h, 40)
    mask = sel.select(scores, Budget(keep_ratio=0.5, recent_tokens=8, sink_tokens=2), 40)
    assert bool(mask[:, :, -8:].all()) and bool(mask[:, :, :2].all())


def test_h2o_scores_nonneg_and_keeps_recent():
    h, d = 4, 64
    q = torch.randn(1, h, 40, d)
    k = torch.randn(1, h, 40, d)
    v = torch.randn(1, h, 40, d)
    sel = H2OSelector(keep_ratio=0.5, recent_tokens=8)
    scores = sel.score(q, k, v, None, 0)
    assert bool((scores >= 0).all())
    mask = sel.select(scores, Budget(keep_ratio=0.5, recent_tokens=8), 40)
    assert bool(mask[:, :, -8:].all())


# --- vLLM mapping (L-2) ---------------------------------------------------
def test_vllm_mapping_fp8_native_and_int3_custom():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p_fp8 = plan_from_kv_config({"codec": {"name": "fp8"}})
    assert p_fp8.native_supported and p_fp8.kv_cache_dtype.startswith("fp8")
    p_int3 = plan_from_kv_config({"codec": {"name": "int3"}})
    assert p_int3.needs_custom_backend and not p_int3.warnings


@pytest.mark.parametrize("name,dtype", [
    ("fp8", "fp8_e4m3"), ("fp8_e4m3", "fp8_e4m3"), ("fp8_e5m2", "fp8_e5m2"),
])
def test_vllm_mapping_fp8_variants(name, dtype):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config, engine_kwargs_from_kv_config

    p = plan_from_kv_config({"codec": {"name": name}})
    assert p.native_supported and not p.needs_custom_backend
    assert p.kv_cache_dtype == dtype and p.calculate_kv_scales
    kw = engine_kwargs_from_kv_config({"codec": {"name": name}})
    assert kw == {"kv_cache_dtype": dtype, "calculate_kv_scales": True}


@pytest.mark.parametrize("name", ["int4", "int3", "int2", "kivi", "ternary"])
def test_vllm_mapping_lowbit_needs_custom_backend(name):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {"name": name}})
    assert p.needs_custom_backend and not p.native_supported
    assert p.kv_cache_dtype == "auto" and not p.warnings  # warned via notes, no silent downgrade


@pytest.mark.parametrize("bits,vbits,preset", [
    (8, 4, "turboquant_k8v4"),
    (4, None, "turboquant_4bit_nc"),
    (4, 4, "turboquant_4bit_nc"),
    (3, 4, "turboquant_k3v4_nc"),
    (3, None, "turboquant_3bit_nc"),
    (3, 3, "turboquant_3bit_nc"),
])
def test_vllm_mapping_turboquant_native_presets(bits, vbits, preset):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    codec = {"name": "turboquant", "bits": bits}
    if vbits is not None:
        codec["value_bits"] = vbits
    p = plan_from_kv_config({"codec": codec})
    assert p.native_supported and not p.needs_custom_backend
    assert p.kv_cache_dtype == preset


def test_vllm_mapping_turboquant_unsupported_bits_custom():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    # 2-bit has no native preset -> custom backend.
    p = plan_from_kv_config({"codec": {"name": "turboquant", "bits": 2}})
    assert p.needs_custom_backend and p.kv_cache_dtype == "auto"


def test_vllm_mapping_halo_maps_to_turboquant_with_warning():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {"name": "halo", "bits": 4}})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    # honestly warns that vLLM runs faithful TurboQuant (no HALO LS/hadamard-gemm)
    assert any("HALO" in w for w in p.warnings)


@pytest.mark.parametrize("name,dtype", [("nvfp4", "nvfp4"),
                                        ("int8", "int8_per_token_head")])
def test_vllm_mapping_other_native(name, dtype):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {"name": name}})
    assert p.native_supported and p.kv_cache_dtype == dtype


def test_vllm_mapping_eviction_only_and_policy():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    # Eviction selector, no codec -> custom backend, dtype unchanged.
    p = plan_from_kv_config({"selector": {"name": "snapkv"}})
    assert p.needs_custom_backend and p.kv_cache_dtype == "auto" and p.codec_name is None
    # Non-static adaptive policy needs the custom backend even with FP8 codec.
    p2 = plan_from_kv_config({"codec": {"name": "fp8"}, "policy": {"name": "taperkv"}})
    assert p2.needs_custom_backend
    # static policy + fp8 stays native.
    p3 = plan_from_kv_config({"codec": {"name": "fp8"}, "policy": {"name": "static"}})
    assert p3.native_supported and not p3.needs_custom_backend


def test_vllm_mapping_conflict_and_unknown():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {"name": "fp8"}}, requested_kv_cache_dtype="fp8_e5m2")
    assert p.kv_cache_dtype == "fp8_e4m3" and p.warnings  # codec dtype wins, warns
    p2 = plan_from_kv_config({"codec": {"name": "mystery"}})
    assert not p2.native_supported and p2.warnings


def test_vllm_apply_to_engine_args_fp8_and_lowbit(tmp_path):
    """apply_to_engine_args mutates a (mock) EngineArgs from a model config.json.

    Uses a SimpleNamespace mock -- apply_to_engine_args never imports vLLM, so
    the FP8 auto-apply logic is fully testable on a CPU-only node.
    """
    import json
    from types import SimpleNamespace
    from vllm_plugins.kvcomp.vllm_plugin import apply_to_engine_args

    def model_dir(kv_cfg):
        d = tmp_path / ("m" + str(abs(hash(str(kv_cfg)))))
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"kv_compression_config": kv_cfg}))
        return str(d)

    fp8_dir = model_dir({"codec": {"name": "fp8"}, "policy": {"name": "static"}})
    ea = SimpleNamespace(model=fp8_dir, kv_cache_dtype="auto", calculate_kv_scales=False)
    apply_to_engine_args(ea)
    assert ea.kv_cache_dtype == "fp8_e4m3" and ea.calculate_kv_scales is True

    int4_dir = model_dir({"codec": {"name": "int4"}, "policy": {"name": "static"}})
    ea2 = SimpleNamespace(model=int4_dir, kv_cache_dtype="auto", calculate_kv_scales=False)
    apply_to_engine_args(ea2)
    assert ea2.kv_cache_dtype == "auto"  # low-bit not auto-applied (needs custom backend)

    # No kv_compression_config -> no-op.
    plain = tmp_path / "plain"; plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "llama"}))
    ea3 = SimpleNamespace(model=str(plain), kv_cache_dtype="auto")
    apply_to_engine_args(ea3)
    assert ea3.kv_cache_dtype == "auto"


# --- PPL harness non-finite guard (B-2) -----------------------------------
def test_ppl_skips_nonfinite_windows():
    from types import SimpleNamespace
    from onecomp.kvcomp.eval.ppl import perplexity_over_text

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=(torch.arange(60) % 7).unsqueeze(0))

    class _Model:
        device = "cpu"
        def __init__(self, bad):
            self.bad = bad
        def __call__(self, ids, past_key_values=None, use_cache=False):
            v, L = 7, ids.shape[1]
            logits = torch.zeros(1, L, v)
            logits[..., 0] = 2.0
            if self.bad:
                logits = logits + float("inf")
            return SimpleNamespace(logits=logits)

    out = perplexity_over_text(_Model(False), _Tok(), "x", max_length=20, stride=10, max_windows=3)
    assert out["n_skipped_nonfinite"] == 0 and out["ppl"] > 0
    # every window non-finite -> perplexity undefined -> raises (no silent NaN).
    with pytest.raises(ValueError):
        perplexity_over_text(_Model(True), _Tok(), "x", max_length=20, stride=10, max_windows=3)


# --- outlier-layer detection: multi-probe aggregation (B-6) ----------------
def _outlier_fake_model(schedule):
    """Fake model whose per-call key tensors mark the layers flagged in ``schedule``.

    ``schedule`` is a list over probes; each entry is a list over layers of bools
    (True => inject a massive-activation channel => range/median ratio ~ 1000).
    """
    from types import SimpleNamespace

    class _FakeModel:
        device = "cpu"

        def __init__(self, sched):
            self.sched = sched
            self.i = 0

        def __call__(self, ids, use_cache=True):
            flags = self.sched[self.i]
            self.i += 1
            layers = []
            for is_out in flags:
                k = torch.ones(1, 4, 8)
                if is_out:
                    k[..., 0] = 1000.0
                layers.append(SimpleNamespace(keys=k))
            return SimpleNamespace(past_key_values=SimpleNamespace(layers=layers))

    return _FakeModel([list(p) for p in schedule])


def test_detect_outlier_layers_multiprobe_agg():
    from onecomp.kvcomp.recipe.thinkv import detect_outlier_layers

    ids = torch.zeros(1, 4, dtype=torch.long)
    # 3 probes, 2 layers: layer0 outlier always, layer1 outlier in only 1/3 probes.
    sched = [[True, True], [True, False], [True, False]]

    # single-tensor arg reproduces single-probe behaviour (uses probe 0 -> [0, 1]).
    assert detect_outlier_layers(_outlier_fake_model([sched[0]]), ids) == [0, 1]

    # mean: layer1's one huge probe dominates the average -> both skipped.
    assert detect_outlier_layers(
        _outlier_fake_model(sched), [ids, ids, ids], agg="mean") == [0, 1]

    # vote: layer1 is a minority outlier (1/3) -> only layer0 skipped (stable).
    assert detect_outlier_layers(
        _outlier_fake_model(sched), [ids, ids, ids], agg="vote") == [0]


# --- benchmark harness: codec speed-quality ablation matrix & CI formatting ---
def _import_benchmark_module(name):
    import importlib
    import os
    import sys

    bench_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "benchmark", "kvcomp-longcontext")
    bench_dir = os.path.abspath(bench_dir)
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    return importlib.import_module(name)


def test_codec_speed_ablation_matrix():
    """The HALO lever ablation: GEMM is quality-neutral, LS lowers MSE, iso-storage."""
    cs = _import_benchmark_module("codec_speed")
    codecs = cs._codecs(4)
    # The legacy speed/quality levers plus the role-asymmetric enhanced variants
    # (NC keys, NC keys + ls_iter values) that the paper benchmarks.
    assert {"TurboQuant-hadamard", "HALO-hadamard-noLS", "HALO-hadamard_gemm-noLS",
            "HALO-hadamard_gemm-LS", "HALO-dense-LS"} <= set(codecs)
    assert {"HALO-hadamard_gemm-NC", "HALO-hadamard_gemm-NCIT"} <= set(codecs)

    torch.manual_seed(0)
    k = torch.randn(1, 2, 64, 128)
    v = torch.randn(1, 2, 64, 128)

    def rt(c):
        comp = c.encode(k, v)
        kq, vq = c.decode(comp)
        mse = ((kq - k).pow(2).mean() + (vq - v).pow(2).mean()) / 2
        return float(mse), comp.meta["stored_bits"]

    mse_h, bits_h = rt(codecs["HALO-hadamard-noLS"])
    mse_hg, bits_hg = rt(codecs["HALO-hadamard_gemm-noLS"])
    mse_hg_ls, bits_ls = rt(codecs["HALO-hadamard_gemm-LS"])
    # GEMM is mathematically identical to the butterfly => same MSE & storage.
    assert mse_hg == pytest.approx(mse_h, rel=1e-4)
    assert bits_hg == bits_h == bits_ls          # LS is storage-neutral
    # Least-squares dequant scale provably lowers (<=) reconstruction MSE.
    assert mse_hg_ls <= mse_hg + 1e-9


def test_markdown_ci_formatting():
    kb = _import_benchmark_module("kv_benchmark")
    # value ± half-width of the 95% CI
    assert kb._fmt_ci({"pass@1": 0.5, "pass@1_ci95": [0.4, 0.6]}) == "0.5\u00b10.100"
    assert kb._fmt_ci({"accuracy": 0.9, "accuracy_ci95": [0.85, 0.97]},
                      "accuracy", "accuracy_ci95") == "0.9\u00b10.060"
    assert kb._fmt_ci({}) == ""               # missing metric -> blank cell
    assert kb._fmt_ci({"pass@1": 0.3}) == "0.3"  # no CI -> bare value


# --- LiveCodeBench functional pass@1 sandbox (ThinKV code axis) -------------
def test_code_sandbox_executor():
    from onecomp.kvcomp.eval.code import run_code, _build_program

    test = "def check(c):\n    assert c(2, 3) == 5\n    assert c(-1, 1) == 0"
    ok, _ = run_code(_build_program("def add(a, b):\n    return a + b", test, "add"))
    assert ok is True
    bad, detail = run_code(_build_program("def add(a, b):\n    return a - b", test, "add"))
    assert bad is False and "AssertionError" in detail
    loop, det2 = run_code(
        _build_program("def add(a, b):\n    while True:\n        pass", test, "add"),
        timeout=2)
    assert loop is False and det2 == "timeout"


def test_extract_code_variants():
    from onecomp.kvcomp.eval.code import extract_code

    fenced = "Reasoning...\n```python\ndef f(x):\n    return x\n```\nok"
    assert extract_code(fenced, "f") == "def f(x):\n    return x"
    # picks the block that defines the requested entry point
    multi = "```python\ndef g():\n    pass\n```\n```python\ndef f(x):\n    return x*2\n```"
    assert "return x*2" in extract_code(multi, "f")
    # bare code (no fences): from first def to end
    bare = "Here:\ndef f(x):\n    return x + 1"
    assert extract_code(bare, "f").startswith("def f(x):")


def test_code_pass_at_1_fake_model():
    """End-to-end code_pass_at_1 loop with a stub model that emits a fixed solution."""
    from types import SimpleNamespace
    from onecomp.kvcomp.eval.code import code_pass_at_1

    solution = "```python\ndef add(a, b):\n    return a + b\n```"

    class _Tok:
        eos_token_id = 0
        pad_token_id = 0

        def apply_chat_template(self, msgs, **kw):
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.zeros(1, 4, dtype=torch.long))

        def decode(self, ids, skip_special_tokens=True):
            return solution  # model "wrote" the right function

    class _Model:
        device = "cpu"

        def generate(self, input_ids, **kw):
            # return prompt (4 toks) + 3 "generated" toks
            return torch.zeros(1, 7, dtype=torch.long)

    problems = [{
        "prompt": "Write add(a, b) returning the sum.",
        "entry_point": "add",
        "test": "def check(c):\n    assert c(2, 3) == 5",
    }]
    res = code_pass_at_1(_Model(), _Tok(), problems, cache_factory=None,
                         k=1, max_new_tokens=8)
    d = res.as_dict()
    assert d["task"] == "livecodebench" and d["pass@1"] == 1.0
    assert "pass@1_ci95" in d and d["n"] == 1


# --- vLLM mapping: TurboQuant semantic warning + skip_layers propagation ---
def test_vllm_mapping_turboquant_warns_semantics_and_propagates_skip_layers():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {
        "name": "turboquant", "bits": 4, "value_bits": 4,
        "skip_layers": [0, 1, 34, 35]}})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    # semantic-swap warning (Lloyd-Max keys + uniform values) now fires for TQ too
    assert any("uniform" in w for w in p.warnings)
    # boundary-layer protection is propagated, not dropped
    assert p.skip_layers == [0, 1, 34, 35]
    assert p.engine_kwargs()["kv_cache_dtype_skip_layers"] == "0,1,34,35"


def test_vllm_mapping_turboquant_outlier_and_qjl_warn():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {
        "name": "turboquant", "bits": 4, "value_bits": 4,
        "n_outlier": 32, "use_qjl": True}})
    assert any("n_outlier" in w for w in p.warnings)
    assert any("use_qjl" in w or "QJL" in w for w in p.warnings)
