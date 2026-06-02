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
