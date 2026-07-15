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
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec, _codebook, _qjl_projection_on
from onecomp.kvcomp.codec.halo import HALOCodec
from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.codec import _block_codebook_on
from onecomp.kvcomp.codec.hayate.vq import lookup_codewords
from onecomp.kvcomp.codec.hayate.recipes import (
    HQ_FRONTIER_KEY_SCALE_MODE,
    build_hayate_adaptive2,
    build_hayate_frontier,
    build_hayate_fractional_rate,
    build_hayate_fractional2,
    hayate_adaptive2_kwargs,
    hayate_fractional_rate_kwargs,
    hayate_fractional2_kwargs,
    hayate_fractional2_variant,
    hayate_frontier_bits,
    hayate_frontier_kwargs,
    hayate_frontier_variant,
)
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
        "hayatequant",
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


def test_halo_non_pow2_hadamard_uses_dense_rotated_quantizer():
    x = torch.randn(1, 2, 16, 96)  # 96 not power of two
    h = HALOCodec(bits=4, rotation_mode="hadamard")
    enc = h.encode(x, x)
    assert not enc.meta.get("fallback")
    assert enc.key_q["kind"] == "d"
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
    tq = TurboQuantCodec(bits=4, rotation_mode="hadamard", dist="gaussian")
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
    # Default (key/value modes=None, ls_scale=True) -> smart NC+ls_iter.
    x, _ = _kv(h=2, s=128)
    assert HALOCodec(bits=4)._scale_mode_for("k") == "nc"        # smart default
    assert HALOCodec(bits=4)._scale_mode_for("v") == "ls_iter"   # smart default
    # Explicit modes always override ls_scale.
    o = HALOCodec(bits=4, key_scale_mode="nc", value_scale_mode="rms")
    assert o._scale_mode_for("k") == "nc" and o._scale_mode_for("v") == "rms"
    # ls_scale=False -> RMS on both (TurboQuant-faithful), regardless of explicit modes.
    o2 = HALOCodec(bits=4, ls_scale=False)
    assert o2._scale_mode_for("k") == "rms" and o2._scale_mode_for("v") == "rms"
    # ls_scale=False with explicit key_scale_mode -> explicit wins for that role.
    o3 = HALOCodec(bits=4, ls_scale=False, key_scale_mode="nc")
    assert o3._scale_mode_for("k") == "nc" and o3._scale_mode_for("v") == "rms"
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
@pytest.mark.parametrize("n_o,ob,base,eff", [(32, 3, 2, 2.5), (32, 4, 3, 3.5),
                                             (64, 3, 2, 2.75)])
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
    assert abs(enc.meta["stored_bits"] / (vecs * d) - eff) < 1e-6


def test_halo_outlier_matches_turboquant_storage():
    # HALO and TurboQuant compute identical mixed-precision storage at the same
    # split, so the 2.5/3.5-bit comparison table is iso-memory.
    x, _ = _kv(h=4, s=64)
    hh = HALOCodec(bits=2, n_outlier=32, outlier_bits=3, rotation_mode="hadamard",
                   key_scale_mode="nc", value_scale_mode="ls")
    tq = TurboQuantCodec(bits=2, n_outlier=32, outlier_bits=3, rotation_mode="hadamard")
    assert hh.encode(x, x).meta["stored_bits"] == tq.encode(x, x).meta["stored_bits"]


def test_halo_outlier_beats_uniform_on_anisotropic_kv():
    # With a few massive-activation channels, spending more bits on them lowers
    # reconstruction MSE vs uniform base bits at the SAME nominal base.
    x = torch.randn(2, 4, 64, 128)
    x[..., [3, 17, 40, 91]] *= 12.0
    mix = HALOCodec(bits=2, n_outlier=32, outlier_bits=3, rotation_mode="dense",
                    key_scale_mode="nc")
    uni = HALOCodec(bits=2, rotation_mode="dense", key_scale_mode="nc")
    mse_mix = (mix.decode(mix.encode(x, x))[0] - x).pow(2).sum()
    mse_uni = (uni.decode(uni.encode(x, x))[0] - x).pow(2).sum()
    assert float(mse_mix) <= float(mse_uni)


# --- HayateQuant: superset of TurboQuant + HALO, block-VQ, hybrid rotation
def test_hayatequant_block_size_1_matches_turboquant():
    # block_size=1, RMS scale on both roles -> bit-identical to TurboQuant
    # (HayateQuant's scalar codebook falls back to rotation.lloyd_max_*).
    # TurboQuantCodec itself only exposes "hadamard"/"dense" (no accelerated
    # "hadamard_gemm" alias); HayateQuant's hadamard_gemm-vs-hadamard
    # equivalence is already covered by the HALO tests above (same transform).
    x, _ = _kv(h=4, s=64)
    for rot in ("hadamard", "dense"):
        hq = HayateQuantCodec(bits=4, block_size=1, rotation_mode=rot,
                               key_scale_mode="rms", value_scale_mode="rms")
        tq = TurboQuantCodec(bits=4, rotation_mode=rot, dist="gaussian")
        mse_hq = float((hq.decode(hq.encode(x, x))[0] - x).pow(2).mean())
        mse_tq = float((tq.decode(tq.encode(x, x))[0] - x).pow(2).mean())
        assert abs(mse_hq - mse_tq) < 1e-6


def test_hayatequant_block_size_1_matches_turboquant_beta():
    # The beta (exact finite-d law) path must ALSO degenerate to TurboQuant at
    # block_size=1.  TurboQuant normalises by the L2 norm and uses the [-1,1]
    # beta codebook; HayateQuant normalises by the RMS (= L2/sqrt(m)) and a
    # sqrt(m)-rescaled beta codebook -- the codes and reconstruction are
    # identical (regression test for the missing rescale, which silently
    # broke the superset claim for dist="beta").
    x, _ = _kv(h=4, s=64)
    for rot in ("hadamard", "dense"):
        hq = HayateQuantCodec(bits=4, block_size=1, rotation_mode=rot,
                              dist="beta", key_scale_mode="rms",
                              value_scale_mode="rms")
        tq = TurboQuantCodec(bits=4, rotation_mode=rot, dist="beta")
        mse_hq = float((hq.decode(hq.encode(x, x))[0] - x).pow(2).mean())
        mse_tq = float((tq.decode(tq.encode(x, x))[0] - x).pow(2).mean())
        assert abs(mse_hq - mse_tq) < 1e-6


def test_hayatequant_payload_dist_roundtrip():
    # ``dist`` is stored in the payload (as in TurboQuant) so decode picks the
    # same codebook even if the codec instance is reconfigured in between.
    x, _ = _kv(h=2, s=32)
    enc = HayateQuantCodec(bits=4, block_size=1, dist="beta",
                           key_scale_mode="rms", value_scale_mode="rms")
    comp = enc.encode(x, x)
    dec = HayateQuantCodec(bits=4, block_size=1, dist="gaussian",
                           key_scale_mode="rms", value_scale_mode="rms")
    k1, _ = enc.decode(comp)
    k2, _ = dec.decode(comp)
    assert torch.allclose(k1, k2)


def test_hayatequant_payload_codebook_init_roundtrip():
    # ``codebook_init`` can change the block codebook. Store it in the payload
    # so decode is self-describing even if the codec object is reconfigured.
    x, _ = _kv(h=2, s=16, d=32)
    enc = HayateQuantCodec(bits=2, block_size=2, codebook_init="fibquant",
                           encoder="exact")
    comp = enc.encode(x, x)
    dec = HayateQuantCodec(bits=2, block_size=2, codebook_init="forgy",
                           encoder="exact")
    k1, _ = enc.decode(comp)
    k2, _ = dec.decode(comp)
    assert torch.allclose(k1, k2)


def test_hayatequant_fibquant_init_shape_and_finite():
    # FibQuant's radial-angular design is used as a deterministic warm start
    # for HayateQuant's Lloyd block-VQ fit.
    from onecomp.kvcomp.codec.hayate.vq import fibquant_initial_codebook

    cb2 = fibquant_initial_codebook(bits=2, block_size=2, ambient_dim=128)
    cb4 = fibquant_initial_codebook(bits=1, block_size=4, ambient_dim=128)
    assert cb2.shape == (16, 2)
    assert cb4.shape == (16, 4)
    assert torch.isfinite(cb2).all() and torch.isfinite(cb4).all()
    assert bool((cb2.norm(dim=1) > 0).all())


def test_hayatequant_block_size_1_matches_halo_scale_modes():
    # block_size=1 with HALO's role-asymmetric scale modes -> bit-identical to
    # HALOCodec (HayateQuant absorbs _scale.fit_scale, the shared module both
    # codecs now call).
    x, _ = _kv(h=4, s=64)
    hq = HayateQuantCodec(bits=4, block_size=1, rotation_mode="hadamard_gemm",
                          key_scale_mode="nc", value_scale_mode="ls_iter")
    ha = HALOCodec(bits=4, rotation_mode="hadamard_gemm",
                   key_scale_mode="nc", value_scale_mode="ls_iter")
    mse_hq = float((hq.decode(hq.encode(x, x))[0] - x).pow(2).mean())
    mse_ha = float((ha.decode(ha.encode(x, x))[0] - x).pow(2).mean())
    assert abs(mse_hq - mse_ha) < 1e-6


@pytest.mark.parametrize("block_size", [2, 4])
def test_hayatequant_block_vq_beats_scalar_at_equal_bits(block_size):
    # Space-filling gain: at the SAME bits/coordinate, a block codebook must
    # not do worse than the scalar (block_size=1) codebook on the source law
    # it was fit to (see paper/tests/verify_hayatequant_vq_gain.py for the
    # exhaustive, CPU-only version of this claim).
    torch.manual_seed(0)
    x, _ = _kv(h=2, s=32, d=64)
    hq1 = HayateQuantCodec(bits=2, block_size=1, rotation_mode="dense", dist="gaussian")
    hqb = HayateQuantCodec(bits=2, block_size=block_size, rotation_mode="dense", dist="gaussian")
    mse1 = float((hq1.decode(hq1.encode(x, x))[0] - x).pow(2).mean())
    mseb = float((hqb.decode(hqb.encode(x, x))[0] - x).pow(2).mean())
    assert mseb <= mse1 * 1.05  # small slack: single random KV sample, not the idealised source


def test_hayatequant_storage_neutral_across_block_size():
    # Same nominal bits-per-coordinate must report the same stored_bits
    # regardless of block_size (block-VQ trades codebook shape for accuracy,
    # not rate).
    x, _ = _kv(h=2, s=32, d=64)
    s1 = HayateQuantCodec(bits=2, block_size=1).encode(x, x).meta["stored_bits"]
    s4 = HayateQuantCodec(bits=2, block_size=4).encode(x, x).meta["stored_bits"]
    assert s1 == s4


def test_hayatequant_frontier_recipe_is_canonical():
    assert hayate_frontier_bits(2) == (3, 1)
    assert hayate_frontier_bits(3) == (4, 2)
    assert hayate_frontier_bits(4) == (5, 3)
    assert hayate_frontier_bits(1) is None

    kwargs = hayate_frontier_kwargs(2)
    assert kwargs == {
        "bits": 3,
        "value_bits": 1,
        "key_scale_mode": HQ_FRONTIER_KEY_SCALE_MODE,
    }
    codec = build_hayate_frontier(2)
    assert codec.bits == 3
    assert codec.value_bits == 1
    assert codec.key_scale_mode == "ub"

    variant = hayate_frontier_variant(3, label="HQ")
    assert variant == {
        "label": "HQ",
        "bits": 4,
        "value_bits": 2,
        "key_scale_mode": "ub",
    }


def test_hayatequant_fractional2_recipe_is_opt_in():
    kwargs = hayate_fractional2_kwargs()
    assert kwargs == {
        "bits": 2,
        "value_bits": 2,
        "avg_bits": 2.0,
        "block_size": 4,
        "key_bit_fraction": 0.625,
        "key_scale_mode": "ub",
    }
    codec = build_hayate_fractional2()
    assert codec.bits == 2
    assert codec.value_bits == 2
    assert codec.block_size == 4
    assert codec.key_code_bits_per_block == 10
    assert codec.value_code_bits_per_block == 6
    assert codec.max_codebook_bits == 10

    variant = hayate_fractional2_variant(label="HQ-frac")
    assert variant["label"] == "HQ-frac"
    assert variant["avg_bits"] == 2.0


def test_hayatequant_fractional_rate_ui_resolves_code_bits():
    codec = build_hayate_fractional_rate(avg_bits=1.25, block_size=4, key_bit_fraction=0.6)
    assert codec.bits == 1
    assert codec.value_bits == 1
    assert codec.key_code_bits_per_block == 6
    assert codec.value_code_bits_per_block == 4
    assert codec.max_codebook_bits == 9


def test_hayatequant_adaptive2_recipe_is_opt_in():
    kwargs = hayate_adaptive2_kwargs()
    assert kwargs["avg_bits"] == 2.0
    assert kwargs["block_size"] == 4
    assert kwargs["adaptive_bit_split"] == "online_stats"
    assert kwargs.get("adaptive_stats_min_tokens", 16) == 16
    codec = build_hayate_adaptive2()
    assert codec.key_code_bits_per_block is None
    assert codec.value_code_bits_per_block is None
    assert codec.max_codebook_bits == 10


def test_hayatequant_invalid_block_size_raises():
    with pytest.raises(Exception):
        HayateQuantCodec(bits=2, block_size=3)  # not in {None,1,2,4,8}
    with pytest.raises(Exception):
        HayateQuantCodec(bits=2, block_size=0)


def test_hayatequant_invalid_codebook_init_raises():
    with pytest.raises(Exception):
        HayateQuantCodec(bits=2, codebook_init="random_magic")


def test_blockquant_scale_aliases_match_existing_modes():
    from onecomp.kvcomp.codec._scale import fit_scale

    xr = torch.randn(8, 16)
    chat = xr + 0.05 * torch.randn_like(xr)
    rms = xr.pow(2).mean(-1, keepdim=True).sqrt()
    s_ls, _ = fit_scale(xr, chat, "ls", rms)
    s_bsm, _ = fit_scale(xr, chat, "bsm", rms)
    s_ip, _ = fit_scale(xr, chat, "ip", rms)
    s_ub, _ = fit_scale(xr, chat, "ub", rms)
    assert torch.allclose(s_bsm, s_ls)
    assert torch.allclose(s_ub, s_ip)


def test_hayatequant_calibrated_fast_roundtrip_matches_encode_decode():
    from onecomp.kvcomp.codec.hayate.calibration import HayateQuantCalibration
    from onecomp.kvcomp.codec.hayate.codec import _block_codebook

    x, _ = _kv(h=2, s=64, d=32)
    cb = _block_codebook(2, 2, "gaussian", 32, "auto")
    cb = cb + 0.01 * torch.randn_like(cb)
    cal = HayateQuantCalibration(
        head_dim=32,
        rot={"k": torch.eye(32), "v": torch.eye(32)},
        codebook={"k": cb, "v": cb.clone()},
        cb_key={"k": (2, 2), "v": (2, 2)},
    )
    c = HayateQuantCodec(bits=2, value_bits=2, block_size=2, encoder="lut",
                         calibration=cal, key_scale_mode="ub",
                         value_scale_mode="ls_iter")
    comp = c.encode(x, x)
    k_ref, v_ref = c.decode(comp)
    k_rt, v_rt, stored = c.fake_quant_roundtrip(x, x)
    assert stored == comp.meta["stored_bits"]
    assert torch.allclose(k_rt, k_ref, atol=1e-6)
    assert torch.allclose(v_rt, v_ref, atol=1e-6)


def test_hayatequant_block_size_falls_back_when_codebook_too_big():
    # An unreachable max_codebook_bits budget (even block_size=1 doesn't fit)
    # must fall back to block_size=1, never raise at encode time -- HayateQuant
    # is always at least able to run TurboQuant's scalar quantizer.
    x, _ = _kv(h=2, s=16, d=64)
    hq = HayateQuantCodec(bits=4, block_size=4, max_codebook_bits=1)
    k, v = hq.decode(hq.encode(x, x))
    assert k.shape == x.shape


def test_hayatequant_outlier_mixed_precision_matches_halo_accounting():
    # HayateQuant inherits HALO/TurboQuant's outlier mixed-precision accounting
    # verbatim (same n_outlier/outlier_bits split logic) at block_size=1.
    x, _ = _kv(h=4, s=64)
    hq = HayateQuantCodec(bits=2, block_size=1, n_outlier=32, outlier_bits=3,
                          rotation_mode="hadamard")
    ha = HALOCodec(bits=2, n_outlier=32, outlier_bits=3, rotation_mode="hadamard")
    assert hq.encode(x, x).meta["stored_bits"] == ha.encode(x, x).meta["stored_bits"]


def test_hayatequant_role_specific_outlier_accounting_blockquant_recipe():
    # BlockQuant's KV-cache recipe protects key outlier channels while keeping
    # values at a lower uniform bit-width.
    x, _ = _kv(h=2, s=16, d=128)
    c = HayateQuantCodec(bits=3, value_bits=2,
                         key_n_outlier=32, key_outlier_bits=4,
                         value_n_outlier=0)
    enc = c.encode(x, x)
    vecs = x.shape[0] * x.shape[1] * x.shape[2]
    key_bits = vecs * (32 * 4 + 96 * 3) + vecs * 2 * 16
    value_bits = x.numel() * 2 + vecs * 16
    assert enc.meta["stored_bits"] == key_bits + value_bits
    assert enc.key_q["mode"] == "outlier"
    assert enc.value_q["mode"] == "single"


def test_hayatequant_explicit_code_bits_per_block_fractional_rate_roundtrip():
    # Fixed-length block codes make fractional bits/dim explicit:
    # key: 5 bits / 4 dims = 1.25 bpc, value: 3 / 4 = 0.75 bpc,
    # average payload rate = 1.0 bpc before the shared fp16 scale overhead.
    x, _ = _kv(h=2, s=16, d=128)
    c = HayateQuantCodec(
        bits=1,
        value_bits=1,
        block_size=4,
        key_code_bits_per_block=5,
        value_code_bits_per_block=3,
        encoder="exact",
    )
    enc = c.encode(x, x)
    kd, vd = c.decode(enc)
    assert kd.shape == x.shape
    assert vd.shape == x.shape
    vecs = x.shape[0] * x.shape[1] * x.shape[2]
    blocks = x.shape[-1] // 4
    expected = vecs * blocks * (5 + 3) + vecs * 2 * 16
    assert enc.meta["stored_bits"] == expected
    assert enc.key_q["main"]["code_bits"] == 5
    assert enc.value_q["main"]["code_bits"] == 3
    kr, vr, stored = c.fake_quant_roundtrip(x, x)
    assert stored == expected
    assert torch.allclose(kr, kd, atol=1e-5)
    assert torch.allclose(vr, vd, atol=1e-5)


def test_hayatequant_adaptive_bit_split_roundtrip_and_accounting():
    x, _ = _kv(h=2, s=32, d=32)
    c = build_hayate_adaptive2(encoder="exact", adaptive_bit_margin=1)
    enc = c.encode(x, x * 0.25)
    kd, vd = c.decode(enc)
    assert kd.shape == x.shape
    assert vd.shape == x.shape
    chosen = enc.meta["adaptive_bit_split"]
    assert chosen["policy"] == "online_stats"
    assert chosen["used_stats"]
    assert chosen["key_code_bits_per_block"] + chosen["value_code_bits_per_block"] == 16
    assert enc.key_q["main"]["code_bits"] == chosen["key_code_bits_per_block"]
    assert enc.value_q["main"]["code_bits"] == chosen["value_code_bits_per_block"]
    assert chosen["key_power"] > 0
    assert chosen["value_power"] > 0
    assert chosen["predicted_key_error"] > 0
    assert chosen["predicted_value_error"] > 0
    vecs = x.shape[0] * x.shape[1] * x.shape[2]
    blocks = x.shape[-1] // 4
    expected = vecs * blocks * 16 + vecs * 2 * 16
    assert enc.meta["stored_bits"] == expected


def test_hybrid_rotation_matrix_matches_two_stage():
    # The folded single-GEMM hybrid rotation (rotation.hybrid_rotation_matrix)
    # must be mathematically identical to the two-stage reference
    # (signed-Hadamard butterfly then per-block dense einsum) AND orthogonal
    # (exact inverse via the transpose) -- this is what makes the hybrid
    # rotation cost the same as a plain dense rotation at runtime.
    from onecomp.kvcomp.codec.rotation import (
        random_sign, block_dense_matrices, hybrid_rotate,
        hybrid_rotation_matrix,
    )
    torch.manual_seed(0)
    for d, bsz in ((128, 2), (128, 4), (64, 2)):
        x = torch.randn(2, 3, 5, d)
        sign = random_sign(d, 0, dtype=torch.float32)
        bm = block_dense_matrices(d // bsz, bsz, 0, "cpu", torch.float32)
        ref = hybrid_rotate(x, sign, bm)
        w = hybrid_rotation_matrix(d, bsz, 0, "cpu", torch.float32)
        assert torch.allclose(ref, x @ w, atol=1e-4)
        assert torch.allclose((x @ w) @ w.t(), x, atol=1e-4)


def test_hayatequant_lut_encoder_matches_exact_distortion():
    # The candidate-list LUT encoder restricts the nearest-codeword search to
    # the top-T candidates of the query's grid cell; distances over the
    # candidates are exact, so the only possible error is the true nearest
    # codeword missing from the list. Verify the resulting excess distortion
    # is negligible (<0.5%) at every default (block_size, K) spec, on the
    # exact source law the codebook was fit to.
    from onecomp.kvcomp.codec.hayate.vq import (
        lloyd_vq_gaussian_block, lut_spec_for, build_vq_lut,
        quantize_via_lut, quantize_to_vq_codebook,
    )
    torch.manual_seed(0)
    for bits, bsz in ((2, 4), (3, 2), (4, 2)):
        k = 1 << (bits * bsz)
        cb = lloyd_vq_gaussian_block(bits, bsz)
        bins, top_t = lut_spec_for(bsz, k)
        lut = build_vq_lut(cb, bins, top_t)
        z = torch.randn(50_000, bsz)
        mse_exact = (z - cb[quantize_to_vq_codebook(z, cb)]).pow(2).mean()
        mse_lut = (z - cb[quantize_via_lut(z, lut, bins, cb)]).pow(2).mean()
        assert float(mse_lut) <= float(mse_exact) * 1.005


def test_hayatequant_lut_and_exact_encoders_agree_end_to_end():
    # Full-codec check: the forced LUT path (encoder="lut") and the exact
    # search must give near-identical reconstruction error on real-shaped KV
    # at every paper bit-width (the LUT restricts the candidate set only,
    # never the reconstruction). encoder="lut" is forced because "auto" only
    # engages the LUT above the measured GEMM/LUT crossover (K >= 1024).
    torch.manual_seed(0)
    x, _ = _kv(h=4, s=64)
    for bits in (2, 3, 4):
        lut_c = HayateQuantCodec(bits=bits, encoder="lut")
        ex_c = HayateQuantCodec(bits=bits, encoder="exact")
        mse_lut = float((lut_c.decode(lut_c.encode(x, x))[0] - x).pow(2).mean())
        mse_ex = float((ex_c.decode(ex_c.encode(x, x))[0] - x).pow(2).mean())
        assert abs(mse_lut - mse_ex) <= mse_ex * 0.02


def test_hayatequant_invalid_encoder_raises():
    with pytest.raises(Exception):
        HayateQuantCodec(bits=4, encoder="magic")


def test_hayatequant_auto_rotation_is_rate_aware():
    # Measured policy (v7 rotation-x-scale attribution, Llama-3.1-8B): at
    # coarse rates (bits<=2) the fixed +-1 Hadamard factor correlates the
    # quantization error and dense wins PPL (-0.21); at bits>=3 the hybrid's
    # Hadamard incoherence wins. auto must implement exactly that.
    hq = HayateQuantCodec(bits=2)
    assert hq._resolved_mode(128, 4, 2) == "dense"
    assert hq._resolved_mode(128, 4, 1) == "dense"
    assert hq._resolved_mode(128, 2, 3) == "hybrid"
    assert hq._resolved_mode(128, 2, 4) == "hybrid"
    assert hq._resolved_mode(96, 2, 4) == "dense"     # non power-of-two dim
    assert hq._resolved_mode(128, 1, 4) == "hadamard_gemm"  # scalar path
    # Explicit rotation_mode always wins over the rate-aware default.
    assert HayateQuantCodec(bits=2, rotation_mode="hybrid")._resolved_mode(128, 4, 2) == "hybrid"
    # End-to-end: the payload rot tag records the resolved rotation.
    x, _ = _kv(h=2, s=16, d=64)
    enc2 = HayateQuantCodec(bits=2).encode(x, x)
    assert enc2.key_q["main"]["rot"][0] == "d"
    enc3 = HayateQuantCodec(bits=3).encode(x, x)
    assert enc3.key_q["main"]["rot"][0] == "y"


def test_hayatequant_codebook_artifacts_loaded_when_present():
    # The committed high-budget Lloyd artifacts (codebooks/gauss_b*_k*.pt)
    # must be preferred over the online fit for the default fit params, and
    # ignored for non-default fit params (those callers asked for a specific
    # online fit).
    import os
    from onecomp.kvcomp.codec.hayate import vq as _vq
    art_dir = os.path.join(os.path.dirname(_vq.__file__), "codebooks")
    if not os.path.isfile(os.path.join(art_dir, "gauss_b4_k256.pt")):
        pytest.skip("codebook artifacts not present")
    _vq.lloyd_vq_gaussian_block.cache_clear()
    _vq._gaussian_codebook_artifact.cache_clear()
    cb = _vq.lloyd_vq_gaussian_block(2, 4)
    art = torch.load(os.path.join(art_dir, "gauss_b4_k256.pt"),
                     map_location="cpu", weights_only=True)
    assert torch.equal(cb, art) and cb.shape == (256, 4)
    # Non-default fit params bypass the artifact (different tensor).
    online = _vq.lloyd_vq_gaussian_block(2, 4, n_samples=50_000)
    assert not torch.equal(online, art)


def _tiny_calibration(codec, x, steps=8):
    from onecomp.kvcomp.codec.hayate.calibration import calibrate_hayatequant
    flat = x.reshape(-1, x.shape[-1])
    return calibrate_hayatequant(codec, flat, flat, steps=steps, refit_every=4,
                                 batch_vectors=256, device="cpu")


def test_hayatequant_spectral_balanced_rotation_flattens_covariance():
    from onecomp.kvcomp.codec.hayate.calibration import spectral_balanced_rotation

    torch.manual_seed(0)
    d = 32
    scales = torch.linspace(0.2, 8.0, d)
    x = torch.randn(4096, d) * scales
    cov0 = x.t() @ x / x.shape[0]
    w = spectral_balanced_rotation(x, seed=3)
    cov1 = w.t() @ cov0 @ w
    spread0 = float(cov0.diag().std() / cov0.diag().mean())
    spread1 = float(cov1.diag().std() / cov1.diag().mean())
    assert spread1 < spread0 * 0.05
    assert torch.allclose(w.t() @ w, torch.eye(d), atol=1e-5)


def test_hayatequant_spectral_calibration_roundtrip_and_guard():
    from onecomp.kvcomp.codec.hayate.calibration import calibrate_hayatequant_spectral

    x, _ = _kv(h=2, s=32, d=32)
    flat = x.reshape(-1, x.shape[-1])
    codec = HayateQuantCodec(bits=2, block_size=2, encoder="exact")
    cal = calibrate_hayatequant_spectral(
        codec,
        flat,
        flat,
        learn_codebook=True,
        refit_iters=1,
        max_vectors=512,
        device="cpu",
    )
    assert cal.meta["method"] == "spectral_balanced_rotation"
    for role in ("k", "v"):
        assert cal.meta[f"{role}_role_error_after"] <= cal.meta[f"{role}_role_error_before"] + 1e-9
    c = HayateQuantCodec(bits=2, block_size=2, encoder="exact", calibration=cal)
    comp = c.encode(x, x)
    if "k" in cal.rot:
        assert comp.key_q["main"]["rot"][0] == "c"
    kd, vd = c.decode(comp)
    assert kd.shape == x.shape
    assert vd.shape == x.shape


def test_hayatequant_attention_weighted_calibration_roundtrip():
    from onecomp.kvcomp.codec.hayate.calibration import calibrate_hayatequant_attention_weighted

    torch.manual_seed(4)
    x, _ = _kv(h=2, s=32, d=32)
    flat = x.reshape(-1, x.shape[-1])
    q = flat + 0.1 * torch.randn_like(flat)
    codec = HayateQuantCodec(bits=2, block_size=2, encoder="exact", key_scale_mode="ub")
    cal = calibrate_hayatequant_attention_weighted(
        codec,
        flat,
        q,
        steps=2,
        batch_vectors=128,
        max_vectors=512,
        device="cpu",
    )
    assert cal.meta["method"] == "attention_weighted_key_rotation"
    assert cal.meta["k_aw_after"] <= cal.meta["k_aw_before"] + 1e-9
    c = HayateQuantCodec(bits=2, block_size=2, encoder="exact",
                         key_scale_mode="ub", calibration=cal)
    kd, vd = c.decode(c.encode(x, x))
    assert kd.shape == x.shape
    assert vd.shape == x.shape


def test_hayatequant_pair_weighted_calibration_roundtrip():
    from onecomp.kvcomp.codec.hayate.calibration import calibrate_hayatequant_pair_weighted

    torch.manual_seed(5)
    x, _ = _kv(h=2, s=32, d=32)
    flat = x.reshape(-1, x.shape[-1])
    q = torch.nn.functional.normalize(flat + 0.05 * torch.randn_like(flat), dim=-1)
    weights = torch.linspace(0.5, 1.5, flat.shape[0])
    codec = HayateQuantCodec(bits=2, block_size=2, encoder="exact", key_scale_mode="ub")
    cal = calibrate_hayatequant_pair_weighted(
        codec,
        flat,
        q,
        weights,
        steps=2,
        batch_pairs=128,
        max_pairs=512,
        device="cpu",
    )
    assert cal.meta["method"] == "attention_pair_key_rotation"
    assert cal.meta["k_pair_after"] <= cal.meta["k_pair_before"] + 1e-9
    c = HayateQuantCodec(bits=2, block_size=2, encoder="exact",
                         key_scale_mode="ub", calibration=cal)
    kd, vd = c.decode(c.encode(x, x))
    assert kd.shape == x.shape
    assert vd.shape == x.shape


def test_hayatequant_layer_rotation_uses_ctx():
    from onecomp.kvcomp.codec.hayate.calibration import HayateQuantCalibration
    from onecomp.kvcomp.codec.rotation import random_orthogonal

    torch.manual_seed(6)
    x, _ = _kv(h=2, s=16, d=32)
    cal = HayateQuantCalibration(head_dim=32)
    cal.layer_rot["k"] = {
        0: torch.eye(32),
        1: random_orthogonal(32, seed=99),
    }
    codec = HayateQuantCodec(bits=2, block_size=2, encoder="exact",
                             key_scale_mode="ub", calibration=cal)
    out0 = codec.fake_quant_roundtrip(x, x, ctx={"layer_idx": 0})[0]
    out1 = codec.fake_quant_roundtrip(x, x, ctx={"layer_idx": 1})[0]
    assert out0.shape == x.shape
    assert out1.shape == x.shape
    assert not torch.allclose(out0, out1)
    p = codec.encode(x, x, ctx={"layer_idx": 1})
    kd, vd = codec.decode(p)
    assert kd.shape == x.shape
    assert vd.shape == x.shape


def test_hayatequant_calibration_roundtrip_and_digest_guard(tmp_path):
    # Calibrated encode/decode roundtrips; payloads carry the artifact digest
    # and decoding with a missing/different artifact fails fast instead of
    # silently unrotating with the wrong matrix.
    torch.manual_seed(0)
    x, _ = _kv(h=2, s=48, d=32)
    codec = HayateQuantCodec(bits=3)
    cal = codec.calibrate({"key": x.reshape(-1, 32), "value": x.reshape(-1, 32)},
                          ctx={"steps": 8, "refit_every": 4, "batch_vectors": 256,
                               "device": "cpu"})
    assert codec.flag_calibration
    comp = codec.encode(x, x)
    if "k" in cal.rot:
        assert comp.key_q["main"]["rot"][0] == "c"
    if "v" in cal.rot:
        assert comp.value_q["main"]["rot"][0] == "c"
    if "k" in cal.codebook:
        assert "cb_cal" in comp.key_q["main"]
    if "v" in cal.codebook:
        assert "cb_cal" in comp.value_q["main"]
    kd, vd = codec.decode(comp)
    assert kd.shape == x.shape
    rel = float((kd.float() - x).pow(2).sum() / x.pow(2).sum())
    assert rel < 0.05
    # Save/load: a fresh codec pointing at the saved artifact decodes the
    # same payload bit-identically.
    p = str(tmp_path / "cal.pt")
    cal.save(p)
    codec2 = HayateQuantCodec(bits=3, calibration=p)
    kd2, _ = codec2.decode(comp)
    assert torch.equal(kd, kd2)
    # Digest guard: decoding without the artifact must raise when the payload
    # actually contains calibrated side information.
    has_cal_payload = (
        comp.key_q["main"]["rot"][0] == "c"
        or comp.value_q["main"]["rot"][0] == "c"
        or "cb_cal" in comp.key_q["main"]
        or "cb_cal" in comp.value_q["main"]
    )
    if has_cal_payload:
        with pytest.raises(ValueError, match="calibrat"):
            HayateQuantCodec(bits=3).decode(comp)


def test_hayatequant_calibration_never_worse_on_holdout():
    # The best-on-holdout selection guarantees the artifact's holdout
    # distortion is <= the data-free starting point (step 0 is the data-free
    # rotation/codebook, so 'after' can only improve on 'before').
    torch.manual_seed(1)
    x, _ = _kv(h=2, s=64, d=32)
    codec = HayateQuantCodec(bits=2)
    cal = _tiny_calibration(codec, x)
    for role in ("k", "v"):
        assert cal.meta[f"{role}_dist_after"] <= cal.meta[f"{role}_dist_before"] + 1e-9
        if role in cal.rot:
            assert cal.rot[role].shape == (32, 32)
            # Learned rotation stays orthogonal (Cayley retraction).
            eye = cal.rot[role] @ cal.rot[role].t()
            assert torch.allclose(eye, torch.eye(32), atol=1e-4)


def test_hayatequant_calibration_config_and_outlier_guard():
    x, _ = _kv(h=2, s=32, d=32)
    codec = HayateQuantCodec(bits=2)
    cal = _tiny_calibration(codec, x, steps=4)
    c = HayateQuantCodec(bits=2, calibration=cal)
    assert c.get_kv_config()["calibration"] == cal.digest
    with pytest.raises(ValueError, match="n_outlier"):
        HayateQuantCodec(bits=2, calibration=cal, n_outlier=4)
    with pytest.raises(ValueError, match="n_outlier"):
        HayateQuantCodec(bits=2, calibration=cal, key_n_outlier=4)
    with pytest.raises(ValueError, match="calibration must be"):
        HayateQuantCodec(bits=2, calibration=123)


def test_hayatequant_invalid_value_or_outlier_bits_raise():
    with pytest.raises(Exception):
        HayateQuantCodec(bits=4, value_bits=99)
    with pytest.raises(Exception):
        HayateQuantCodec(bits=4, outlier_bits=0)
    with pytest.raises(Exception):
        HayateQuantCodec(bits=4, key_outlier_bits=0)
    with pytest.raises(Exception):
        HayateQuantCodec(bits=4, value_n_outlier=-1)
    assert HayateQuantCodec(bits=4, value_bits=2, outlier_bits=5).value_bits == 2


def test_hayatequant_decode_uses_payload_seed():
    # The rotation seed is stored in the payload (rot tag), so decode must
    # reconstruct the encode-time rotation even if the decoding instance was
    # configured with a different seed -- same self-describing-payload
    # convention as the stored "dist".
    torch.manual_seed(0)
    x, _ = _kv(h=2, s=32, d=64)
    for rotation_mode in ("hybrid", "hadamard_gemm", "hadamard", "dense"):
        enc_c = HayateQuantCodec(bits=4, rotation_mode=rotation_mode, seed=7)
        dec_c = HayateQuantCodec(bits=4, rotation_mode=rotation_mode, seed=123)
        enc = enc_c.encode(x, x)
        ref_k, ref_v = enc_c.decode(enc)
        got_k, got_v = dec_c.decode(enc)
        assert torch.equal(got_k, ref_k) and torch.equal(got_v, ref_v)


def test_hayatequant_decode_accepts_legacy_rot_tag():
    # Payloads written before the seed was stored carried 2-tuple rot tags;
    # decode must fall back to the instance seed for those.
    torch.manual_seed(0)
    x, _ = _kv(h=2, s=16, d=64)
    c = HayateQuantCodec(bits=4, rotation_mode="hybrid", seed=7)
    enc = c.encode(x, x)
    ref_k, _ = c.decode(enc)
    for payload in (enc.key_q, enc.value_q):
        p = payload["main"]
        p["rot"] = tuple(p["rot"][:2])
    got_k, _ = c.decode(enc)
    assert torch.equal(got_k, ref_k)


def test_hayatequant_qjl_outlier_accounting_counts_two_rms():
    # With the outlier split active, QJL stores one residual-RMS per subset
    # vector (2 per full vector), plus 1 sign bit per coordinate.
    x, _ = _kv(h=4, s=64)
    b, h, s, d = x.shape
    kw = dict(bits=4, block_size=1, n_outlier=16, rotation_mode="hadamard")
    s0 = HayateQuantCodec(use_qjl=False, **kw).encode(x, x).meta["stored_bits"]
    s1 = HayateQuantCodec(use_qjl=True, **kw).encode(x, x).meta["stored_bits"]
    vecs = b * h * s * 2  # key + value
    assert s1 - s0 == x.numel() * 2 * 1 + vecs * 2 * 16


def test_hayatequant_qjl_uses_gaussian_projection_formula():
    """HayateQuant QJL must match TurboQuant's Gaussian residual sketch."""
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 32)
    c = HayateQuantCodec(
        bits=2, block_size=2, rotation_mode="dense", use_qjl=True, seed=13
    )
    enc = c.encode(x, x)
    p = enc.key_q["main"]
    cb = _block_codebook_on(
        p["bits"], p["block_size"], p["dist"], p["m"],
        p["codebook_init"], str(x.device),
    )
    xr, _ = c._rotate(x, p["m"], p["block_size"], p["bits"], "k")
    recon = lookup_codewords(cb, p["codes"]).reshape_as(xr) * p["scale"]
    resid = xr - recon
    proj = _qjl_projection_on(p["m"], p["qjl_seed"], str(x.device))
    projected = (resid.reshape(-1, p["m"]) @ proj.t()).reshape_as(resid)

    assert torch.equal(p["qjl_sign"], projected >= 0)
    assert torch.allclose(p["qjl_norm"], resid.norm(dim=-1, keepdim=True), atol=1e-6)

    sign = torch.where(p["qjl_sign"], 1.0, -1.0).float()
    expected_rot = recon + (
        sign.reshape(-1, p["m"]) @ proj
    ).reshape_as(resid) * (torch.sqrt(torch.tensor(torch.pi / 2.0)) / p["m"]) * p["qjl_norm"]
    got_rot, _ = c._rotate(c.decode(enc)[0], p["m"], p["block_size"], p["bits"], "k")
    assert torch.allclose(got_rot, expected_rot, atol=1e-5)


def test_hayatequant_large_codebook_not_worse_at_3bit():
    # max_codebook_bits=12 lets 3-bit resolve block_size=4 (K=4096), which the
    # O(T) LUT encoder makes affordable; the bigger joint codebook must not do
    # worse than the default (block_size=2, K=64) at the same stored bits.
    torch.manual_seed(0)
    x, _ = _kv(h=4, s=64)
    small = HayateQuantCodec(bits=3)
    big = HayateQuantCodec(bits=3, max_codebook_bits=12)
    enc_s, enc_b = small.encode(x, x), big.encode(x, x)
    assert enc_s.meta["stored_bits"] == enc_b.meta["stored_bits"]
    mse_s = float((small.decode(enc_s)[0] - x).pow(2).mean())
    mse_b = float((big.decode(enc_b)[0] - x).pow(2).mean())
    assert mse_b <= mse_s * 1.02


# --- non-power-of-two rotation / QJL accounting (memo A-6, A-7) -----------
def test_non_power_of_two_hadamard_preserves_rotated_codec():
    # A Hadamard is undefined at d=96. The codec records a dense QR rotation
    # instead of silently changing the requested algorithm to affine INT.
    kf = torch.randn(1, 2, 8, 96)
    for codec in (HALOCodec(bits=4, rotation_mode="hadamard"),
                  TurboQuantCodec(bits=4, rotation_mode="hadamard")):
        enc = codec.encode(kf, kf)
        assert not enc.meta.get("fallback")
        vecs = 1 * 2 * 8
        assert enc.meta["stored_bits"] == kf.numel() * 4 * 2 + (vecs * 2) * 16
        if isinstance(codec, HALOCodec):
            assert enc.key_q["kind"] == "d"
        else:
            assert enc.key_q["main"]["rot"][0] == "d"


def test_turboquant_qjl_counts_residual_norm():
    x, _ = _kv(h=4, s=64)
    b, h, s, d = x.shape
    s0 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=False).encode(x, x).meta["stored_bits"]
    s1 = TurboQuantCodec(bits=4, rotation_mode="hadamard", use_qjl=True).encode(x, x).meta["stored_bits"]
    vecs = b * h * s
    assert s1 - s0 == x.numel() * 1 + vecs * 16  # key signs + key residual norms


def test_turboquant_qjl_outlier_accounting_counts_two_rms():
    # With the outlier split active, QJL stores one residual-RMS per subset
    # vector (2 per full vector), plus 1 sign bit per coordinate -- mirrors
    # the HayateQuant accounting (test above) for the shared convention.
    x, _ = _kv(h=4, s=64)
    b, h, s, d = x.shape
    kw = dict(bits=4, n_outlier=16, rotation_mode="hadamard")
    s0 = TurboQuantCodec(use_qjl=False, **kw).encode(x, x).meta["stored_bits"]
    s1 = TurboQuantCodec(use_qjl=True, **kw).encode(x, x).meta["stored_bits"]
    vecs = b * h * s
    assert s1 - s0 == x.numel() * 1 + vecs * 2 * 16


def test_turboquant_qjl_uses_gaussian_projection_formula():
    # TurboQuant-prod QJL is not a coordinate sign residual.  It stores
    # sign(S r) and reconstructs sqrt(pi/2)/d * ||r||_2 * S^T sign(S r), which
    # is the vector whose inner product with a query is the QJL estimator.
    torch.manual_seed(0)
    x = torch.randn(1, 2, 8, 32)
    c = TurboQuantCodec(bits=2, rotation_mode="dense", use_qjl=True, seed=13)
    enc = c.encode(x, x)
    p = enc.key_q["main"]
    cb = _codebook(p["bits"], p["dist"], p["m"]).to(x.device)

    xr, _ = c._rotate(x, p["m"], 13)
    mse_recon = cb[p["codes"]] * p["scale"]
    resid = xr - mse_recon
    proj = _qjl_projection_on(p["m"], p["qjl_seed"], str(x.device))
    sign = torch.where(p["qjl_sign"], 1.0, -1.0).float()

    assert torch.equal(p["qjl_sign"], (resid.reshape(-1, p["m"]) @ proj.t() >= 0).reshape_as(resid))
    assert torch.allclose(p["qjl_norm"], resid.norm(dim=-1, keepdim=True), atol=1e-6)

    expected_rot = mse_recon + (
        sign.reshape(-1, p["m"]) @ proj
    ).reshape_as(resid) * (torch.sqrt(torch.tensor(torch.pi / 2.0)) / p["m"]) * p["qjl_norm"]
    got_rot, _ = c._rotate(c.decode(enc)[0], p["m"], 13)
    assert torch.allclose(got_rot, expected_rot, atol=1e-5)


# --- fused fake-quant round trip (the serving hot path) --------------------
@pytest.mark.parametrize("kw", [
    dict(bits=2), dict(bits=3), dict(bits=4),
    dict(bits=3, block_size=1),            # scalar (TurboQuant-equivalent) path
    dict(bits=4, value_bits=2),            # K/V bit asymmetry (separate cores)
    dict(bits=3, dist="beta"),             # exact finite-d law
    dict(bits=3, ls_scale=False),          # rms scale on both roles
    dict(bits=3, key_scale_mode="ip", value_scale_mode="ls"),
    dict(bits=2, block_size=2, encoder="lut"),  # forced LUT assignment path
    dict(bits=3, rotation_mode="dense"),
    dict(bits=3, rotation_mode="hadamard_gemm"),
])
def test_hayatequant_roundtrip_matches_encode_decode(kw):
    # The fused round trip must reproduce decode(encode(x)) -- same rotation,
    # codebook, assignment and scale fit -- while skipping the payload.  Any
    # divergence would silently change every serving benchmark and eval.
    torch.manual_seed(0)
    k, v = torch.randn(1, 4, 64, 128), torch.randn(1, 4, 64, 128)
    c = HayateQuantCodec(**kw)
    enc = c.encode(k, v)
    k1, v1 = c.decode(enc)
    k2, v2, stored = c.fake_quant_roundtrip(k, v)
    assert stored == enc.meta["stored_bits"]
    assert k2.dtype == k.dtype and v2.dtype == v.dtype
    assert torch.allclose(k1, k2, atol=1e-5), kw
    assert torch.allclose(v1, v2, atol=1e-5), kw


def test_hayatequant_roundtrip_decode_step_shape():
    # The serving decode step (S=1) is the launch-bound hot path the fused
    # round trip exists for; make sure the stacked-KV core handles it.
    torch.manual_seed(1)
    k, v = torch.randn(1, 8, 1, 128), torch.randn(1, 8, 1, 128)
    c = HayateQuantCodec(bits=3)
    k1, v1 = c.decode(c.encode(k, v))
    k2, v2, _ = c.fake_quant_roundtrip(k, v)
    assert torch.allclose(k1, k2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


@pytest.mark.parametrize("kw", [
    dict(bits=3, n_outlier=16),   # outlier split -> base-class fallback
    dict(bits=3, use_qjl=True),   # QJL residual -> base-class fallback
])
def test_hayatequant_roundtrip_fallback_matches(kw):
    torch.manual_seed(2)
    k, v = torch.randn(1, 4, 64, 128), torch.randn(1, 4, 64, 128)
    c = HayateQuantCodec(**kw)
    enc = c.encode(k, v)
    k1, v1 = c.decode(enc)
    k2, v2, stored = c.fake_quant_roundtrip(k, v)
    assert stored == enc.meta["stored_bits"]
    assert torch.equal(k1, k2) and torch.equal(v1, v2)


def _triton_reference_roundtrip(x, w, cb32, block_size, modes):
    """fp32 torch mirror of the fused Triton step kernel's exact semantics."""
    r, d = x.shape[0], x.shape[-1]
    nb = d // block_size
    xr = x.reshape(-1, d) @ w

    def assign(z):
        zb = z.reshape(-1, nb, block_size)
        scores = zb @ cb32.t() - 0.5 * cb32.pow(2).sum(1)
        return cb32[scores.argmax(-1)].reshape(z.shape)

    rows = xr.shape[0]
    split = rows // r if r == 2 else rows
    out = torch.empty_like(xr)
    for lo, hi, mode in ((0, split, modes[0]), (split, rows, modes[-1])):
        if lo >= hi:
            continue
        if mode == "bsm":
            mode = "ls"
        elif mode == "ub":
            mode = "ip"
        xi = xr[lo:hi]
        rms = xi.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        chat = assign(xi / rms)
        if mode == "rms":
            s = rms
        elif mode == "nc":
            s = xi.norm(dim=-1, keepdim=True) / chat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            num = (xi * chat).sum(-1, keepdim=True)
            den = (chat * chat).sum(-1, keepdim=True).clamp_min(1e-8)
            if mode == "ip":
                num_safe = torch.where(num >= 0, num.clamp_min(1e-8), num.clamp_max(-1e-8))
                s = (xi * xi).sum(-1, keepdim=True) / num_safe
            else:
                s = num / den
                if mode == "ls_iter":
                    chat = assign(xi / s.clamp_min(1e-8))
                    num2 = (xi * chat).sum(-1, keepdim=True)
                    den2 = (chat * chat).sum(-1, keepdim=True).clamp_min(1e-8)
                    s = num2 / den2
        out[lo:hi] = chat * s
    return (out @ w.t()).reshape(x.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kw", [
    dict(bits=2), dict(bits=3), dict(bits=4),
    dict(bits=4, value_bits=2), dict(bits=3, block_size=1),
    dict(bits=3, ls_scale=False),
])
def test_hayatequant_triton_step_matches_eager_roundtrip(kw, monkeypatch):
    # Two-level parity for the fused Triton decode-step kernel:
    #  1. exact agreement with a fp32 torch mirror of the kernel semantics
    #     (same scoring dtype, so only measure-zero argmax ties can differ);
    #  2. reconstruction quality within a small band of the eager fused round
    #     trip, whose exact search scores in fp16 on CUDA and therefore
    #     resolves near-ties differently (a flipped block changes that row's
    #     fitted scale, so elementwise comparisons are meaningless).
    from onecomp.kvcomp.codec.hayate.triton_step import HAS_TRITON, roundtrip_step_triton
    from onecomp.kvcomp.codec.hayate.codec import _codebook_aux_on, resolve_block_size
    if not HAS_TRITON:
        pytest.skip("triton not installed")
    torch.manual_seed(0)
    c = HayateQuantCodec(**kw)
    for s in (1, 8):  # decode-step and small-block shapes (exact-search regime)
        k = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
        monkeypatch.delenv("ONECOMP_TRITON_CODEC", raising=False)
        k1, v1, s1 = c.fake_quant_roundtrip(k, v)
        monkeypatch.setenv("ONECOMP_TRITON_CODEC", "1")
        k2, v2, s2 = c.fake_quant_roundtrip(k, v)
        assert s1 == s2

        # (1) exact parity with the fp32 reference, per role
        kb, vb = c.bits, c._value_bits()
        for x, bits, modes, got in (
            (k.float().unsqueeze(0), kb, (c._scale_mode_for("k"),), k2),
            (v.float().unsqueeze(0), vb, (c._scale_mode_for("v"),), v2),
        ):
            m = x.shape[-1]
            bs = resolve_block_size(bits, c.block_size, c.max_codebook_bits, m)
            w = c._rt_rotation(m, bs, bits, x.device)
            cb32 = _codebook_aux_on(bits, bs, c.dist, m, c.codebook_init, str(x.device))[0]
            ref = _triton_reference_roundtrip(x, w, cb32, bs, modes)[0]
            assert torch.allclose(got.float(), ref, rtol=1e-3, atol=2e-3), (kw, s, modes)

        # (2) quality parity with the eager fused path
        for a, b, x in ((k1, k2, k), (v1, v2, v)):
            m1 = float((a - x).float().pow(2).mean())
            m2 = float((b - x).float().pow(2).mean())
            assert abs(m1 - m2) / max(m1, 1e-12) < 5e-2, (kw, s, m1, m2)


def test_base_codec_roundtrip_default_matches_encode_decode():
    # Codecs without a fused override (e.g. TurboQuant) must behave exactly
    # as before through the cache's new fake_quant_roundtrip entry point.
    x, y = _kv(h=4, s=64)
    c = TurboQuantCodec(bits=4, rotation_mode="hadamard")
    enc = c.encode(x, y)
    k1, v1 = c.decode(enc)
    k2, v2, stored = c.fake_quant_roundtrip(x, y)
    assert stored == enc.meta["stored_bits"]
    assert torch.equal(k1, k2) and torch.equal(v1, v2)


def test_compressed_cache_matches_manual_roundtrip():
    # The cache's update() must store exactly the codec's round-trip output
    # (per-block, sink excluded) -- guards the encode/decode -> roundtrip swap.
    torch.manual_seed(3)
    codec = HayateQuantCodec(bits=3)
    cache = CompressedDynamicCache(codec=codec, sink_tokens=0)
    k, v = torch.randn(1, 4, 32, 128), torch.randn(1, 4, 32, 128)
    ck, cv = cache.update(k, v, 0)
    k2, v2, stored = codec.fake_quant_roundtrip(k, v)
    assert torch.allclose(ck, k2, atol=1e-5) and torch.allclose(cv, v2, atol=1e-5)
    assert cache.stored_bits == stored


@pytest.mark.parametrize(
    "codec",
    [KIVICodec(bits=2), FP8Codec(), NVFP4Codec(), TernaryCodec(), Int3Codec(), Int4Codec(),
     HayateQuantCodec(bits=2, block_size=1), HayateQuantCodec(bits=2, block_size=4)],
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


def test_compressed_cache_online_recent_retires_one_token():
    b, h, s, d = 1, 2, 8, 64
    c = CompressedDynamicCache(
        codec=Int4Codec(key_axis="token", value_axis="token"),
        sink_tokens=2,
        recent_tokens=2,
    )
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    c.update(k, v, 0)
    assert torch.equal(c.layers[0].keys[:, :, :2], k[:, :, :2])
    assert torch.equal(c.layers[0].keys[:, :, -2:], k[:, :, -2:])

    knew = torch.randn(b, h, 1, d)
    vnew = torch.randn(b, h, 1, d)
    c.update(knew, vnew, 0)
    assert torch.equal(c.layers[0].keys[:, :, -1:], knew)
    assert torch.equal(c.layers[0].keys[:, :, -2:-1], k[:, :, -1:])
    assert not torch.allclose(c.layers[0].keys[:, :, -3:-2], k[:, :, -2:-1])
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
def test_vllm_mapping_turboquant_native_presets(bits, vbits, preset, monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
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


def test_vllm_mapping_halo_maps_to_turboquant_with_warning(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {"name": "halo", "bits": 4}})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    # honestly warns that vLLM runs faithful TurboQuant (no HALO LS/hadamard-gemm)
    assert any("HALO" in w for w in p.warnings)


def test_vllm_mapping_hayatequant_block_size_1_native(monkeypatch):
    # block_size=1 is TurboQuant/HALO-equivalent -> same native preset path.
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {"name": "hayatequant", "bits": 4, "block_size": 1}})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    assert any("HayateQuant" in w for w in p.warnings)


def test_vllm_mapping_hayatequant_block_vq_no_preset_needs_custom_backend():
    # block_size>1 (block-VQ) at 2-bit has no native scalar vLLM preset either.
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {"name": "hayatequant", "bits": 2, "block_size": 4}})
    assert p.needs_custom_backend and not p.native_supported and p.kv_cache_dtype == "auto"
    assert any("no native scalar TurboQuant preset" in n for n in p.notes)


def test_vllm_mapping_hayatequant_fractional_rate_needs_custom_backend():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": build_hayate_fractional2().get_kv_config()})
    notes = "\n".join(p.notes)
    assert p.needs_custom_backend and not p.native_supported
    assert "key_code_bits_per_block" in notes
    assert "value_code_bits_per_block" in notes


def test_vllm_mapping_hayatequant_adaptive_rate_needs_custom_backend():
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    p = plan_from_kv_config({"codec": {**hayate_adaptive2_kwargs(), "name": "hayatequant"}})
    notes = "\n".join(p.notes)
    assert p.needs_custom_backend and not p.native_supported
    assert "avg_bits" in notes
    assert "adaptive_bit_split" in notes


def test_vllm_mapping_hayatequant_block_vq_is_strict_by_default(monkeypatch):
    """The key-only vLLM hybrid must never be mistaken for full HayateQuant."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.delenv("KVCOMP_VLLM_HAYATEQUANT_MODE", raising=False)
    monkeypatch.delenv("KVCOMP_VLLM_SCALAR_FALLBACK", raising=False)
    codec = {"name": "hayatequant", "bits": 3, "block_size": 4}
    p = plan_from_kv_config({"codec": codec})
    assert p.needs_custom_backend and not p.native_supported
    assert p.kv_cache_dtype == "auto" and not p.hayate_native

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    p_native = plan_from_kv_config({"codec": codec})
    assert p_native.native_supported and p_native.kv_cache_dtype == "turboquant_3bit_nc"
    assert not p_native.needs_custom_backend and p_native.hayate_native
    assert p_native.hayate_config["bits"] == 3
    assert p_native.hayate_config["key_scale_mode"] == "nc"
    assert any("EXPERIMENTAL APPROXIMATION" in n for n in p_native.notes)
    assert any("key-only" in w for w in p_native.warnings)

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "scalar_fallback")
    p_fb = plan_from_kv_config({"codec": codec})
    assert p_fb.native_supported and p_fb.kv_cache_dtype == "turboquant_3bit_nc"
    assert not p_fb.needs_custom_backend
    assert not p_fb.hayate_native
    assert any("block-VQ accuracy" in w for w in p_fb.warnings)

    # Bit-widths with no scalar preset still fall through to custom-backend.
    p2 = plan_from_kv_config({"codec": {"name": "hayatequant", "bits": 2, "block_size": 4}})
    assert p2.needs_custom_backend and p2.kv_cache_dtype == "auto"


def test_vllm_mapping_hayatequant_block_vq_native_key_ub():
    """BlockQuant's UB/IP key scale is byte-compatible with the native
    HayateQuant scale slot, so it can stay on the fast path."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    codec = {
        "name": "hayatequant",
        "bits": 3,
        "block_size": 4,
        "vllm_mode": "native",
        "key_scale_mode": "ub",
        "codebook_init": "fibquant",
    }
    p = plan_from_kv_config({"codec": codec})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_3bit_nc"
    assert p.hayate_native
    assert p.hayate_config["key_scale_mode"] == "ip"
    assert p.hayate_config["codebook_init"] == "fibquant"


def test_vllm_mapping_hayatequant_block_vq_strict_mode(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "strict")
    codec = {"name": "hayatequant", "bits": 3, "block_size": 4}
    p = plan_from_kv_config({"codec": codec})
    assert p.needs_custom_backend and not p.native_supported
    assert p.kv_cache_dtype == "auto"
    assert any("strict/custom mode" in n for n in p.notes)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_vllm_mapping_hayatequant_default_get_kv_config_does_not_crash(bits):
    """Regression test: ``HayateQuantCodec``'s real ``get_kv_config()`` output
    (default ``block_size=None``, "auto") must not crash the mapper.

    ``codec.get("block_size", 1)`` silently returns ``None`` (not the ``1``
    default) when the key is present with value ``None`` -- exactly what a
    default-constructed ``HayateQuantCodec().get_kv_config()`` produces --
    so a naive ``int(codec.get("block_size", 1))`` raises ``TypeError``. This
    feeds the *actual* codec's config dict through the mapper (not a
    hand-built dict) so it exercises the real integration surface a
    ``Runner``-saved model's ``config.json`` would hit.
    """
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    codec = HayateQuantCodec(bits=bits)
    assert codec.block_size is None  # sanity: the default really is "auto"
    p = plan_from_kv_config({"codec": codec.get_kv_config()})
    # Whichever branch it resolves to, it must not have thrown, and must be
    # internally consistent (native XOR custom-backend, never neither/both).
    assert p.native_supported != p.needs_custom_backend


@pytest.mark.parametrize("bits,expect_block_vq", [
    (2, True), (3, True), (4, True), (6, False), (8, False),
])
def test_vllm_mapping_hayatequant_block_vq_detection_matches_resolve_block_size(bits, expect_block_vq):
    """The mapper's torch-free ``block_size>1`` guess must agree with the real,
    torch-backed :func:`onecomp.kvcomp.codec._vq.resolve_block_size` for the
    codec's own default ``max_codebook_bits`` -- otherwise a config that
    actually uses block-VQ (accuracy gain over TurboQuant) could be silently
    routed onto the native scalar preset (silently losing the improvement).

    Only the ``True`` (block-VQ in play) direction implies a specific
    ``needs_custom_backend``/``native_supported`` outcome unconditionally;
    when block-VQ resolves to off (``block_size==1``), whether the plan ends
    up native still additionally depends on the (bits, bits) preset table
    (covered separately by ``test_vllm_mapping_hayatequant_block_size_1_native``
    et al.), so here we only assert the block-VQ branch was/was not taken.
    """
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config
    from onecomp.kvcomp.codec.hayate.vq import resolve_block_size

    codec = HayateQuantCodec(bits=bits)
    real_block_size = resolve_block_size(bits, None, codec.max_codebook_bits, m=128)
    assert (real_block_size > 1) == expect_block_vq
    p = plan_from_kv_config({"codec": codec.get_kv_config()})
    if expect_block_vq:
        assert p.needs_custom_backend and not p.native_supported
        assert any("block-VQ" in n or "block_size" in n for n in p.notes)
    else:
        assert not any("block-VQ" in n for n in p.notes)


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
    assert d["task"] == "functional_code_smoke" and d["pass@1"] == 1.0
    assert "pass@1_ci95" in d and d["n"] == 1


# --- vLLM mapping: TurboQuant semantic warning + skip_layers propagation ---
def test_vllm_mapping_turboquant_warns_semantics_and_propagates_skip_layers(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {
        "name": "turboquant", "bits": 4, "value_bits": 4,
        "skip_layers": [0, 1, 34, 35]}})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    # semantic-swap warning (Lloyd-Max keys + uniform values) now fires for TQ too
    assert any("uniform" in w for w in p.warnings)
    # boundary-layer protection is propagated, not dropped
    assert p.skip_layers == [0, 1, 34, 35]
    assert p.engine_kwargs()["kv_cache_dtype_skip_layers"] == "0,1,34,35"


def test_vllm_mapping_turboquant_outlier_and_qjl_warn(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {
        "name": "turboquant", "bits": 4, "value_bits": 4,
        "n_outlier": 32, "use_qjl": True}})
    assert any("n_outlier" in w for w in p.warnings)
    assert any("use_qjl" in w or "QJL" in w for w in p.warnings)


def test_vllm_mapping_hayatequant_role_specific_outlier_warns(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {
        "name": "hayatequant", "bits": 4, "block_size": 1,
        "key_n_outlier": 32, "value_n_outlier": 0}})
    assert any("role-specific outlier" in w for w in p.warnings)


# --- vLLM mapping: rotation-family check against the native Hadamard kernel ---
@pytest.mark.parametrize("mode", ["hadamard", "hadamard_gemm", "auto", None])
def test_vllm_mapping_hadamard_family_rotation_is_native_compatible(mode, monkeypatch):
    """Hadamard-family rotations (incl. the folded hadamard_gemm and the
    'auto' default that resolves to Hadamard on power-of-two head dims) map
    onto vLLM's native TurboQuant kernel with a confirming note and *no*
    rotation-mismatch warning."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    codec = {"name": "hayatequant", "bits": 4, "block_size": 1}
    if mode is not None:
        codec["rotation_mode"] = mode
    p = plan_from_kv_config({"codec": codec})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    assert not any("does NOT match" in w for w in p.warnings)
    assert any("Hadamard" in n for n in p.notes)


@pytest.mark.parametrize("name", ["turboquant", "halo", "hayatequant"])
def test_vllm_mapping_dense_rotation_warns(name, monkeypatch):
    """A dense rotation cannot be reproduced by vLLM's native kernel
    (it hard-codes a randomized Hadamard rotation); the mapping must warn
    loudly instead of silently quantizing under a different transform."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    if name == "hayatequant":
        monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    else:
        monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    codec = {"name": name, "bits": 4, "block_size": 1, "rotation_mode": "dense"}
    p = plan_from_kv_config({"codec": codec})
    # Still mapped (the preset is a valid quantizer) but with an explicit
    # rotation-mismatch warning so nothing is swapped silently.
    assert p.native_supported and p.kv_cache_dtype == "turboquant_4bit_nc"
    assert any("does NOT match" in w and "Hadamard" in w for w in p.warnings)


def test_vllm_mapping_hybrid_rotation_at_block_size_1_is_hadamard(monkeypatch):
    """rotation_mode='hybrid' with block_size=1 degenerates to signed-Hadamard
    (1x1 block rotations are sign flips), so it is native-compatible: a
    confirming note, no rotation-mismatch warning."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    p = plan_from_kv_config({"codec": {
        "name": "hayatequant", "bits": 4, "block_size": 1,
        "rotation_mode": "hybrid"}})
    assert p.native_supported
    assert any("degenerates to a signed-Hadamard" in n for n in p.notes)
    assert not any("does NOT match" in w for w in p.warnings)


def test_vllm_mapping_hayatequant_real_codec_hadamard_gemm_bs1(tmp_path, monkeypatch):
    """End-to-end: a real HayateQuantCodec(block_size=1, rotation_mode=
    'hadamard_gemm') get_kv_config() payload maps to the native preset with
    the Hadamard-compatibility note."""
    from onecomp.kvcomp.codec.hayate import HayateQuantCodec
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    codec = HayateQuantCodec(bits=3, value_bits=4, block_size=1,
                             rotation_mode="hadamard_gemm")
    p = plan_from_kv_config({"codec": codec.get_kv_config()})
    assert p.native_supported and p.kv_cache_dtype == "turboquant_k3v4_nc"
    assert any("Hadamard" in n for n in p.notes)
    assert not any("does NOT match" in w for w in p.warnings)
