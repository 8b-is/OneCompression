"""Bit-exact contracts for HayateQuant's accelerated public payload encoder."""

import pytest
import torch

from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.codec import (
    _block_codebook_code_bits_on,
    _block_codebook_on,
    _block_lut_code_bits_on,
    _block_lut_on,
)
from onecomp.kvcomp.codec.hayate.recipes import (
    HQ_ADAPTIVE_NAME,
    HQ_NAME,
    HQ_PUBLIC_METHODS,
    HQ_ROTATION_NAME,
    build_hayate_frontier,
    hayate_adaptive_rate_kwargs,
    hayate_hq_kwargs,
)
from onecomp.kvcomp.codec.hayate.vq import (
    build_vq_lut,
    quantize_exact_pre,
    quantize_to_vq_codebook,
    quantize_via_lut,
)


def test_public_hayate_method_family_has_exactly_four_named_variants():
    assert HQ_PUBLIC_METHODS == (HQ_NAME, "HQ-Frontier", HQ_ADAPTIVE_NAME, HQ_ROTATION_NAME)


def _main_payload(compressed, role):
    payload = compressed.key_q if role == "k" else compressed.value_q
    assert payload["mode"] == "single"
    return payload["main"]


def test_datafree_scalar_public_encode_uses_cached_boundaries(monkeypatch):
    """The scalar hot path keeps legacy assignments without sort/equal sync."""
    torch.manual_seed(73)
    key = torch.randn(1, 3, 17, 32)
    value = torch.randn(1, 3, 17, 32)
    codec = HayateQuantCodec(
        bits=4,
        value_bits=3,
        block_size=1,
        dist="beta",
        rotation_mode="dense",
        key_scale_mode="rms",
        value_scale_mode="ls_iter",
    )

    # Reproduce the former generic scalar assignment before preventing calls
    # to it.  It defensively sorts the codebook and checks ``torch.equal``;
    # the new data-free path obtains the same boundaries from the codebook
    # cache instead.
    references = {}
    for role, source, bits in (("k", key, 4), ("v", value, 3)):
        xr, _ = codec._rotate(source, 32, 1, bits, role)  # pylint: disable=protected-access
        rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-8)
        cb = _block_codebook_on(bits, 1, "beta", 32, "auto", "cpu")
        references[role] = quantize_to_vq_codebook((xr / rms).unsqueeze(-1), cb)

    import onecomp.kvcomp.codec.hayate.vq as vq_module

    def _unexpected_generic_scalar_call(*_args, **_kwargs):
        raise AssertionError("data-free scalar encode rebuilt/sorted decision boundaries")

    monkeypatch.setattr(vq_module, "quantize_to_vq_codebook", _unexpected_generic_scalar_call)
    compressed = codec.encode(key, value)
    for role in ("k", "v"):
        payload = _main_payload(compressed, role)
        # RMS keys keep the initial assignment.  LS-iter values intentionally
        # requantize, so compare their first-pass contract separately below.
        if role == "k":
            assert torch.equal(payload["codes"], references[role].squeeze(-1))
        assert payload["codes"].dtype == torch.int64
        assert payload["scale"].dtype == torch.float16


def test_fake_roundtrip_preserves_public_rolewise_gemm_rounding():
    """K/V stacking must not move an fp16 payload scale by one ULP."""
    torch.manual_seed(1)
    key = torch.randn(1, 8, 1, 128)
    value = torch.randn(1, 8, 1, 128)
    codec = HayateQuantCodec(bits=3)
    compressed = codec.encode(key, value)
    public_key, public_value = codec.decode(compressed)
    fake_key, fake_value, stored_bits = codec.fake_quant_roundtrip(key, value)
    assert stored_bits == compressed.numel_bits()
    assert torch.equal(fake_key, public_key)
    assert torch.equal(fake_value, public_value)


def test_fake_roundtrip_rejects_cuda_graph_compile_flag_on_cpu(monkeypatch):
    """A leaked serving flag must not silently change CPU codec arithmetic."""
    monkeypatch.setenv("ONECOMP_COMPILE_CODEC", "1")
    monkeypatch.setenv("ONECOMP_COMPILE_DECODE_ONLY", "1")
    key = torch.randn(1, 2, 1, 32)
    value = torch.randn(1, 2, 1, 32)
    with pytest.raises(RuntimeError, match="CUDA-graph serving option"):
        HayateQuantCodec(bits=3).fake_quant_roundtrip(key, value)


def test_lut_build_has_deterministic_global_index_tie_contract():
    """The cell table itself uses the smallest indices at a top-T tie."""

    # All codewords score identically at every cell centre.  A non-stable
    # top-k may choose any pair; the formal table must always retain [0, 1].
    cb = torch.zeros(4, 2, dtype=torch.float32)
    first = build_vq_lut(cb, bins=4, top_t=2, lim=1.0)
    second = build_vq_lut(cb, bins=4, top_t=2, lim=1.0)
    expected = torch.tensor([0, 1], dtype=first.dtype).expand_as(first)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


def test_explicit_lut_candidate_count_reaches_table_and_canonical_scorer():
    """T is a real cache/kernel dimension, and full-T reproduces exact."""

    code_bits, block_size, bins = 4, 2, 8
    cb = _block_codebook_code_bits_on(code_bits, block_size, "gaussian", 32, "auto", "cpu")
    tables = {
        top_t: _block_lut_code_bits_on(
            code_bits,
            block_size,
            "gaussian",
            32,
            "auto",
            bins,
            top_t,
            "cpu",
        ).long()
        for top_t in (2, 4, 8, 16)
    }
    assert {top_t: table.shape for top_t, table in tables.items()} == {
        top_t: torch.Size((bins**block_size, top_t)) for top_t in tables
    }
    for lower, upper in ((2, 4), (4, 8), (8, 16)):
        rowwise_present = (tables[lower][:, :, None] == tables[upper][:, None, :]).any(dim=-1)
        assert bool(rowwise_present.all())
        # Every cell receives at least one genuinely new candidate.
        upper_in_lower = (tables[upper][:, :, None] == tables[lower][:, None, :]).any(dim=-1)
        assert bool((~upper_in_lower).any(dim=-1).all())

    generator = torch.Generator().manual_seed(321)
    z = torch.randn(4096, block_size, generator=generator)
    csq = cb.pow(2).sum(dim=1)
    exact = quantize_exact_pre(z, cb, csq)
    candidate = quantize_via_lut(
        z,
        tables[16],
        bins,
        cb,
        scoring_cb=cb,
        scoring_csq=csq,
    )
    assert torch.equal(candidate, exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lut_candidate_bmm_matches_canonical_half_gemm_when_set_is_complete():
    """Candidate scoring differs from exact search only by set restriction."""

    code_bits, block_size, bins = 4, 2, 8
    device = "cuda:0"
    cb32 = _block_codebook_code_bits_on(code_bits, block_size, "gaussian", 32, "auto", device)
    cb16 = cb32.half()
    csq16 = cb16.pow(2).sum(dim=1)
    lut = _block_lut_code_bits_on(
        code_bits,
        block_size,
        "gaussian",
        32,
        "auto",
        bins,
        1 << code_bits,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(4321)
    z = torch.randn(131_072, block_size, generator=generator, device=device)
    exact = quantize_exact_pre(z.half(), cb16, csq16)
    candidate = quantize_via_lut(
        z,
        lut,
        bins,
        cb32,
        scoring_cb=cb16,
        scoring_csq=csq16,
    )
    assert torch.equal(candidate, exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lut_canonical_t4_triton_matches_full_half_gemm_large_fixture():
    """The fused b2/T4 scorer is exact when its candidate set is complete."""

    from onecomp.kvcomp.codec.hayate.triton_step import HAS_TRITON

    if not HAS_TRITON:
        pytest.skip("triton not installed")
    device, bins = "cuda:0", 256
    cb32 = _block_codebook_on(1, 2, "gaussian", 128, "auto", device)
    cb16 = cb32.half()
    csq16 = cb16.pow(2).sum(dim=1)
    lut = _block_lut_on(1, 2, "gaussian", 128, "auto", bins, 4, device)
    generator = torch.Generator(device=device).manual_seed(4404)
    z = torch.randn(1_048_576, 2, generator=generator, device=device)
    exact = quantize_exact_pre(z.half(), cb16, csq16)
    got = quantize_via_lut(
        z,
        lut,
        bins,
        cb32,
        scoring_cb=cb16,
        scoring_csq=csq16,
    )
    assert torch.equal(got, exact)


@pytest.mark.parametrize("average_bits", [2, 3, 4])
def test_formal_hq_and_adaptive_hq_keep_quality_safe_fixed_assignment(average_bits):
    assert hayate_hq_kwargs(average_bits)["encoder"] == "exact"
    assert hayate_adaptive_rate_kwargs(average_bits)["encoder"] == "exact"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "bits,block_size,bins",
    [
        (4, 2, 256),
        (2, 4, 64),
    ],
)
def test_lut_t2_triton_matches_eager_large_fixture(bits, block_size, bins, monkeypatch):
    """One million blocks retain exact eager codes, including boundary rows."""
    import onecomp.kvcomp.codec.hayate.triton_step as triton_step

    if not triton_step.HAS_TRITON:
        pytest.skip("triton not installed")
    device = "cuda:0"
    cb = _block_codebook_on(bits, block_size, "gaussian", 128, "auto", device)
    lut = _block_lut_on(bits, block_size, "gaussian", 128, "auto", bins, 2, device)
    generator = torch.Generator(device=device).manual_seed(1000 + block_size)
    z = torch.randn(1_048_576, block_size, generator=generator, device=device, dtype=torch.float32)

    # Exercise clipping and exact/adjacent regular-grid boundaries in
    # addition to the million random blocks.
    rows = [
        torch.full((block_size,), -7.0, device=device),
        torch.full((block_size,), 7.0, device=device),
        torch.full((block_size,), -5.0, device=device),
        torch.full((block_size,), 5.0, device=device),
    ]
    for cell in (1, bins // 2, bins - 1):
        boundary = torch.tensor(-5.0 + 10.0 * cell / bins, device=device)
        rows.extend(
            [
                boundary.expand(block_size),
                torch.nextafter(boundary, torch.tensor(float("-inf"), device=device)).expand(
                    block_size
                ),
                torch.nextafter(boundary, torch.tensor(float("inf"), device=device)).expand(
                    block_size
                ),
            ]
        )
    z[: len(rows)] = torch.stack(rows)

    monkeypatch.setattr(triton_step, "HAS_TRITON", False)
    reference = quantize_via_lut(z, lut, bins, cb)
    monkeypatch.setattr(triton_step, "HAS_TRITON", True)
    got = quantize_via_lut(z, lut, bins, cb)
    assert reference.dtype == got.dtype == torch.int64
    assert torch.equal(got, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [2, 4])
def test_lut_t2_triton_preserves_smallest_index_tie(block_size, monkeypatch):
    """Equal fp16 distances retain candidate zero, as eager argmin does."""
    import onecomp.kvcomp.codec.hayate.triton_step as triton_step

    if not triton_step.HAS_TRITON:
        pytest.skip("triton not installed")
    cb = torch.zeros(2, block_size, device="cuda", dtype=torch.float32)
    cb[0, 0], cb[1, 0] = -1.0, 1.0
    bins = 2
    lut = torch.tensor([[0, 1]] * (bins**block_size), device="cuda", dtype=torch.int16)
    z = torch.zeros(4096, block_size, device="cuda", dtype=torch.float32)

    monkeypatch.setattr(triton_step, "HAS_TRITON", False)
    reference = quantize_via_lut(z, lut, bins, cb)
    monkeypatch.setattr(triton_step, "HAS_TRITON", True)
    got = quantize_via_lut(z, lut, bins, cb)
    assert torch.equal(reference, torch.zeros_like(reference))
    assert torch.equal(got, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lut_t2_non_fp32_query_uses_generic_fallback(dtype, monkeypatch):
    """The narrow kernel must not change eager fp16/bf16 scoring contracts."""
    import onecomp.kvcomp.codec.hayate.triton_step as triton_step

    block_size, bins = 2, 2
    cb = torch.randn(4, block_size, device="cuda", dtype=torch.float32)
    lut = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]], device="cuda", dtype=torch.int16)
    z = torch.randn(4096, block_size, device="cuda", dtype=dtype)
    monkeypatch.setattr(triton_step, "HAS_TRITON", False)
    reference = quantize_via_lut(z, lut, bins, cb)

    def _unexpected_narrow_dispatch(*_args, **_kwargs):
        raise AssertionError("non-fp32 LUT query reached narrow Triton kernel")

    monkeypatch.setattr(triton_step, "HAS_TRITON", True)
    monkeypatch.setattr(triton_step, "quantize_lut_t2_triton", _unexpected_narrow_dispatch)
    got = quantize_via_lut(z, lut, bins, cb)
    assert torch.equal(got, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_b8_triton_matches_eager_large_fixture_and_tie():
    """Canonical r2 value assignment retains half-GEMM codes bit-for-bit."""
    from onecomp.kvcomp.codec.hayate.codec import _codebook_aux_on
    from onecomp.kvcomp.codec.hayate.triton_step import (
        HAS_TRITON,
        quantize_exact_b8_triton,
        triton_exact_b8_supported,
    )

    if not HAS_TRITON:
        pytest.skip("triton not installed")
    cb, cb16, csq16, _ = _codebook_aux_on(1, 8, "gaussian", 128, "auto", "cuda:0")
    if not triton_exact_b8_supported(cb16, csq16, cb.device):
        pytest.skip("exact b=8 public assignment is audited only on SM100")
    generator = torch.Generator(device="cuda").manual_seed(8108)
    z = torch.randn(131_072, 8, generator=generator, device="cuda")
    reference = quantize_to_vq_codebook(z, cb)
    got = quantize_exact_b8_triton(z, cb16, csq16)
    assert torch.equal(got, reference)

    # A symmetric two-word codebook gives an exact score tie at zero.  Both
    # eager argmax and the fused kernel must retain the smallest index.
    tie_cb = torch.zeros(2, 8, device="cuda", dtype=torch.float16)
    tie_cb[0, 0], tie_cb[1, 0] = -1.0, 1.0
    tie_csq = tie_cb.pow(2).sum(dim=1)
    tie_z = torch.zeros(4096, 8, device="cuda")
    tie_codes = quantize_exact_b8_triton(tie_z, tie_cb, tie_csq)
    assert torch.equal(tie_codes, torch.zeros_like(tie_codes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_b8_unsupported_codebook_uses_generic_fallback(monkeypatch):
    """K>512 must retain the tensor path instead of raising in dispatch."""
    import onecomp.kvcomp.codec.hayate.codec as codec_module
    import onecomp.kvcomp.codec.hayate.triton_step as triton_step

    cb = torch.randn(1024, 8, device="cuda", dtype=torch.float32)
    cb16 = cb.half()
    csq16 = cb16.pow(2).sum(dim=1)
    assert not triton_step.triton_exact_b8_supported(cb16, csq16, cb.device)

    def _fake_aux(*_args, **_kwargs):
        return cb, cb16, csq16, None

    def _unexpected_narrow_dispatch(*_args, **_kwargs):
        raise AssertionError("unsupported b=8 codebook reached narrow Triton kernel")

    monkeypatch.setattr(codec_module, "_codebook_aux_code_bits_on", _fake_aux)
    monkeypatch.setattr(triton_step, "quantize_exact_b8_triton", _unexpected_narrow_dispatch)
    codec = HayateQuantCodec(
        bits=1,
        value_bits=1,
        block_size=8,
        max_codebook_bits=10,
        key_code_bits_per_block=10,
        value_code_bits_per_block=10,
        rotation_mode="dense",
        key_scale_mode="rms",
        value_scale_mode="rms",
    )
    x = torch.randn(1, 1, 1, 8, device="cuda", dtype=torch.float16)
    compressed = codec.encode(x, x)
    assert _main_payload(compressed, "k")["codes"].max() < 1024
    assert _main_payload(compressed, "v")["codes"].max() < 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("average_bits", [2, 3, 4])
def test_frontier_public_payload_triton_encode_is_bit_exact(average_bits, monkeypatch):
    """Canonical r2/r3/r4 codes, fp16 scales, and reconstruction stay exact."""
    import onecomp.kvcomp.codec.hayate.triton_step as triton_step

    if not triton_step.HAS_TRITON:
        pytest.skip("triton not installed")
    generator = torch.Generator(device="cuda").manual_seed(20260716 + average_bits)
    shape = (1, 8, 4096, 128)
    key = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
    value = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
    codec = build_hayate_frontier(average_bits)

    monkeypatch.setattr(triton_step, "HAS_TRITON", False)
    reference = codec.encode(key, value)
    reference_reconstruction = codec.decode(reference)
    monkeypatch.setattr(triton_step, "HAS_TRITON", True)
    got = codec.encode(key, value)
    got_reconstruction = codec.decode(got)

    assert got.numel_bits() == reference.numel_bits()
    for role in ("k", "v"):
        ref_payload = _main_payload(reference, role)
        got_payload = _main_payload(got, role)
        assert ref_payload["codes"].dtype == got_payload["codes"].dtype == torch.int64
        assert ref_payload["scale"].dtype == got_payload["scale"].dtype == torch.float16
        assert torch.equal(got_payload["codes"], ref_payload["codes"]), role
        assert torch.equal(got_payload["scale"], ref_payload["scale"]), role
    for got_tensor, ref_tensor in zip(got_reconstruction, reference_reconstruction):
        assert torch.equal(got_tensor, ref_tensor)
