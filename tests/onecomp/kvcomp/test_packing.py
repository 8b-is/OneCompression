"""Unit tests for the bit-packed affine primitives and PackedAffineCodec.

CPU-only, model-free.  Covers pack/unpack exactness, roundtrip error bounds,
padding/short-axis behaviour, the fp16 scale-overflow guard, and the
registered codec wrapper.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp.kvcomp import build_codec, list_codecs
from onecomp.kvcomp.codec import PackedAffineCodec
from onecomp.kvcomp.codec._packing import (
    PackedQuantizedKV,
    dequantize_packed,
    pack_codes,
    quantize_packed,
    unpack_codes,
)

SHAPE = (2, 4, 32, 64)  # (batch, heads, seq, head_dim)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _roundtrip_error(tensor, bits, axis, group_size=64):
    pq = quantize_packed(tensor, bits, group_size=group_size, axis=axis)
    recon = dequantize_packed(pq)
    return (recon.float() - tensor.float()).abs().mean().item()


@pytest.mark.parametrize("axis", ["token", "channel"])
def test_roundtrip_error_monotone_in_bits(axis):
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    err8 = _roundtrip_error(tensor, 8, axis)
    err4 = _roundtrip_error(tensor, 4, axis)
    err2 = _roundtrip_error(tensor, 2, axis)
    assert err8 < err4 < err2
    assert err8 < 0.01


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("axis", ["token", "channel"])
def test_shape_and_dtype_restored(bits, axis):
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    recon = dequantize_packed(quantize_packed(tensor, bits, axis=axis))
    assert recon.shape == tensor.shape
    assert recon.dtype == tensor.dtype


@pytest.mark.parametrize("bits,vals_per_byte", [(2, 4), (4, 2), (8, 1)])
def test_packed_code_bytes(bits, vals_per_byte):
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    pq = quantize_packed(tensor, bits, group_size=64, axis="token")
    num_elements = tensor.numel()  # head_dim=64 is divisible, no padding
    expected_bytes = -(-num_elements // vals_per_byte)
    assert pq.num_padded == 0
    assert pq.codes.dtype == torch.uint8
    assert pq.codes.numel() == expected_bytes


def test_group_size_shrinks_to_short_axis():
    # axis_len (48) < group_size (64): the group shrinks to the axis length
    # instead of zero-padding the codes (which would inflate short spans).
    tensor = torch.randn(1, 2, 8, 48, dtype=torch.float16)
    pq = quantize_packed(tensor, 8, group_size=64, axis="token")
    assert pq.group_size == 48
    assert pq.num_padded == 0
    recon = dequantize_packed(pq)
    assert recon.shape == tensor.shape
    assert (recon.float() - tensor.float()).abs().mean().item() < 0.02


def test_padding_for_non_divisible_axis():
    # axis_len (100) > group_size (64): only the remainder group is padded.
    tensor = torch.randn(1, 2, 8, 100, dtype=torch.float16)
    pq = quantize_packed(tensor, 8, group_size=64, axis="token")
    assert pq.group_size == 64
    assert pq.num_padded == 28
    recon = dequantize_packed(pq)
    assert recon.shape == tensor.shape
    assert (recon.float() - tensor.float()).abs().mean().item() < 0.02


def test_zero_and_constant_tensors_safe():
    zero = torch.zeros(SHAPE, dtype=torch.float16)
    const = torch.full(SHAPE, 0.5, dtype=torch.float16)
    for bits in (2, 4, 8):
        recon = dequantize_packed(quantize_packed(zero, bits, axis="token"))
        assert torch.allclose(recon.float(), zero.float(), atol=1e-3)
    for bits in (4, 8):
        recon = dequantize_packed(quantize_packed(const, bits, axis="token"))
        assert (recon.float() - const.float()).abs().max().item() < 0.1


def test_large_magnitude_finite_input_stays_finite():
    """Regression: a finite fp32 input with a huge per-group range used to
    overflow the fp16 scale to inf and silently reconstruct as NaN."""
    for mag in (1e4, 1e6, 1e8):
        tensor = torch.randn(1, 2, 8, 64, dtype=torch.float32) * mag
        assert torch.isfinite(tensor).all()
        pq = quantize_packed(tensor, 4, axis="token")
        assert torch.isfinite(pq.scale.float()).all()
        assert torch.isfinite(dequantize_packed(pq)).all()


def test_non_finite_input_raises():
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        quantize_packed(tensor, 4, axis="token")


def test_invalid_arguments_raise():
    tensor = torch.randn(SHAPE)
    with pytest.raises(ValueError):
        quantize_packed(tensor, 3)
    with pytest.raises(ValueError):
        quantize_packed(tensor, 4, axis="row")
    with pytest.raises(ValueError):
        quantize_packed(torch.randn(4, 4), 4)
    with pytest.raises(ValueError):
        quantize_packed(tensor, 4, group_size=0)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("axis", ["token", "channel"])
@pytest.mark.parametrize("seq", [1, 5, 7])
@pytest.mark.parametrize("head_dim", [5, 17, 48])
def test_packing_roundtrip_awkward_shapes(bits, axis, seq, head_dim):
    """Codes survive pack/unpack exactly for shapes not divisible by the
    packing unit or the group size, and padding never pollutes the scales."""
    torch.manual_seed(seq * 100 + head_dim)
    tensor = torch.randn(1, 2, seq, head_dim, dtype=torch.float16)
    pq = quantize_packed(tensor, bits, group_size=17, axis=axis)

    unpacked = unpack_codes(pq.codes, bits, pq.codes.numel() * (8 // bits))
    repacked = pack_codes(unpacked, bits)
    assert torch.equal(repacked, pq.codes)

    recon = dequantize_packed(pq)
    assert recon.shape == tensor.shape
    tol = {8: 0.05, 4: 0.6, 2: 2.5}[bits]
    assert (recon.float() - tensor.float()).abs().max().item() < tol


def test_num_bytes_reflects_bits():
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    sizes = {bits: quantize_packed(tensor, bits, axis="token").num_bytes() for bits in (2, 4, 8)}
    assert sizes[2] < sizes[4] < sizes[8]


def test_metadata_dataclass():
    tensor = torch.randn(SHAPE, dtype=torch.float16)
    pq = quantize_packed(tensor, 4, group_size=32, axis="channel")
    assert isinstance(pq, PackedQuantizedKV)
    assert pq.shape == SHAPE
    assert pq.bits == 4
    assert pq.group_size == 32
    assert pq.axis == "channel"
    assert pq.dtype == "float16"


# --- registered codec wrapper ----------------------------------------------
def test_packed_affine_registered():
    assert "packed_affine" in list_codecs()
    codec = build_codec("packed_affine", bits=4, group_size=32)
    assert isinstance(codec, PackedAffineCodec)


def test_packed_affine_encode_decode_matches_primitives():
    key = torch.randn(SHAPE, dtype=torch.float16)
    value = torch.randn(SHAPE, dtype=torch.float16)
    codec = PackedAffineCodec(bits=4, group_size=64)
    compressed = codec.encode(key, value)
    dk, dv = codec.decode(compressed)
    ref_k = dequantize_packed(quantize_packed(key, 4, group_size=64, axis="channel"))
    ref_v = dequantize_packed(quantize_packed(value, 4, group_size=64, axis="token"))
    assert torch.equal(dk, ref_k)
    assert torch.equal(dv, ref_v)
    # Real storage accounting: stored bits equal the packed payload exactly.
    assert (
        compressed.numel_bits()
        == (compressed.key_q.num_bytes() + compressed.value_q.num_bytes()) * 8
    )


def test_packed_affine_validates_params():
    with pytest.raises(ValueError):
        PackedAffineCodec(bits=3)
    with pytest.raises(ValueError):
        PackedAffineCodec(key_axis="row")
    with pytest.raises(ValueError):
        PackedAffineCodec(group_size=0)
