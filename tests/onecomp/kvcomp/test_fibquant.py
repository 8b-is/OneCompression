"""Focused correctness tests for the independent FibQuant paper baseline."""

import copy
import math

import pytest
import torch

from onecomp.kvcomp import build_codec, list_codecs
from onecomp.kvcomp.codec import FibQuantCodec
from onecomp.kvcomp.codec.fibquant import (
    _fibquant_codebook_cpu,
    _paper_directions,
    _paper_radii,
    fibquant_codebook,
)
from onecomp.kvcomp.codec.quality_baselines import (
    FIBQUANT_PAPER_METHOD,
    build_fibquant_paper,
    fibquant_quality_provenance,
    is_quality_only_baseline,
)
from onecomp.kvcomp.codec.rotation import lloyd_max_beta
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec


def _small_codec(**kwargs):
    defaults = {
        "bits": 1.5,
        "block_size": 2,
        "code_bits": 3,
        "seed": 7,
        "codebook_seed": 11,
    }
    defaults.update(kwargs)
    return FibQuantCodec(**defaults)


def test_fibquant_is_registered_and_exported():
    assert "fibquant" in list_codecs()
    codec = build_codec("fibquant", bits=3)
    assert isinstance(codec, FibQuantCodec)
    assert codec.name == "fibquant"


@pytest.mark.parametrize(
    "bits,expected",
    [
        (2, (4, 8)),
        (3, (2, 6)),
        (4, (2, 8)),
    ],
)
def test_fibquant_integer_recipes_match_paper(bits, expected):
    assert FibQuantCodec(bits=bits)._resolved_recipe() == expected


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_canonical_fibquant_benchmark_baseline_is_quality_only(bits):
    codec = build_fibquant_paper(bits, skip_layers=[3, 1])
    provenance = fibquant_quality_provenance(codec)
    assert isinstance(codec, FibQuantCodec)
    assert codec.skip_layers == [1, 3]
    assert provenance["method_role"] == "external_quality_baseline"
    assert provenance["proposal_method"] is False
    assert provenance["speed_comparison"] is False
    assert provenance["nominal_payload_bits_per_coordinate"] == bits
    assert is_quality_only_baseline(FIBQUANT_PAPER_METHOD)


@pytest.mark.parametrize("bits", [True, 1, 2.5, 5, math.inf, "3"])
def test_canonical_fibquant_benchmark_baseline_rejects_mislabeled_rates(bits):
    with pytest.raises(ValueError, match="one of 2/3/4"):
        build_fibquant_paper(bits)


def test_canonical_fibquant_benchmark_provenance_rejects_recipe_drift():
    codec = build_fibquant_paper(3)
    codec.codebook_seed = 1
    with pytest.raises(ValueError, match="fixed seed"):
        fibquant_quality_provenance(codec)


def test_fibquant_custom_fractional_rate_is_explicit():
    codec = FibQuantCodec(bits=2.5, block_size=4, code_bits=10)
    assert codec._resolved_recipe() == (4, 10)
    config = codec.get_kv_config()
    assert config["bits"] == 2.5
    assert config["block_size"] == 4
    assert config["code_bits"] == 10


def test_fibquant_paper_initialization_equations():
    radii = _paper_radii(64, 2, 8)
    q = (torch.arange(8, dtype=torch.float64) + 0.5) / 8.0
    expected = (1.0 - (1.0 - q).pow(4.0 / 64.0)).sqrt()
    assert torch.allclose(radii, expected, atol=0.0, rtol=2.0e-15)

    directions_2d = _paper_directions(2, 8)
    assert torch.equal(directions_2d[0], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(
        directions_2d.norm(dim=1), torch.ones(8, dtype=torch.float64), atol=1.0e-15
    )
    directions_4d = _paper_directions(4, 17)
    assert torch.allclose(
        directions_4d.norm(dim=1), torch.ones(17, dtype=torch.float64), atol=1.0e-15
    )


def test_fibquant_codebook_is_deterministic_and_stays_in_unit_ball():
    first = fibquant_codebook(8, 2, 3, codebook_seed=19)
    _fibquant_codebook_cpu.cache_clear()
    second = fibquant_codebook(8, 2, 3, codebook_seed=19)
    assert torch.equal(first, second)
    assert first.shape == (8, 2)
    assert first.dtype == torch.float32
    assert bool(torch.isfinite(first).all())
    assert float(first.norm(dim=1).max()) <= 1.0


def test_fibquant_paper_integer_recipes_beat_scalar_control_on_holdout_source():
    """Scientific sanity check on unseen exact f_(64,k) samples, not KV data."""
    generator = torch.Generator(device="cpu").manual_seed(991)
    sphere = torch.randn(32768, 64, generator=generator)
    sphere /= sphere.norm(dim=1, keepdim=True)

    for bits, block_size, code_bits in ((2, 4, 8), (3, 2, 6), (4, 2, 8)):
        source = sphere[:, :block_size]
        vector_codebook = fibquant_codebook(64, block_size, code_bits)
        vector_scores = source @ vector_codebook.t() - 0.5 * vector_codebook.square().sum(dim=1)
        vector_reconstruction = vector_codebook[vector_scores.argmax(dim=1)]
        vector_mse = (source - vector_reconstruction).square().mean()

        scalar_codebook = lloyd_max_beta(bits, 64)
        boundaries = 0.5 * (scalar_codebook[:-1] + scalar_codebook[1:])
        scalar_codes = torch.searchsorted(boundaries, source.contiguous())
        scalar_mse = (source - scalar_codebook[scalar_codes]).square().mean()
        assert vector_mse < scalar_mse, (bits, vector_mse.item(), scalar_mse.item())


def test_fibquant_encode_decode_and_exact_logical_bit_accounting():
    torch.manual_seed(3)
    codec = _small_codec()
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn_like(key)
    key[0, 0, 0] = 0.0
    compressed = codec.encode(key, value)
    key_hat, value_hat = codec.decode(compressed)

    # Per role: 6 vectors * 4 blocks * 3 bits = 72 payload bits and
    # 6 vectors * 16 bits = 96 norm-header bits.
    assert compressed.meta["stored_payload_bits"] == 2 * 72
    assert compressed.meta["stored_norm_bits"] == 2 * 96
    assert compressed.numel_bits() == 336
    assert compressed.key_q["codes"].shape == (1, 2, 3, 4)
    assert compressed.key_q["norm"].dtype == torch.float16
    assert compressed.key_q["norm"].shape == (1, 2, 3, 1)
    assert key_hat.shape == key.shape
    assert value_hat.shape == value.shape
    assert key_hat.dtype == key.dtype
    assert value_hat.dtype == value.dtype
    assert torch.equal(key_hat[0, 0, 0], torch.zeros(8))
    assert bool(torch.isfinite(key_hat).all())
    assert bool(torch.isfinite(value_hat).all())


def test_fibquant_repeated_instances_are_bit_exact_for_fixed_seeds():
    torch.manual_seed(5)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn_like(key)
    left = _small_codec().encode(key, value)
    right = _small_codec().encode(key, value)
    assert torch.equal(left.key_q["codes"], right.key_q["codes"])
    assert torch.equal(left.value_q["codes"], right.value_q["codes"])
    assert torch.equal(left.key_q["norm"], right.key_q["norm"])
    assert left.key_q["codebook_sha256"] == right.key_q["codebook_sha256"]
    assert left.meta["resolved_codec_config"] == right.meta["resolved_codec_config"]


def test_fibquant_k1_is_the_turboquant_scalar_control():
    assert torch.equal(fibquant_codebook(16, 1, 2), lloyd_max_beta(2, 16).unsqueeze(1))

    torch.manual_seed(23)
    key = torch.randn(2, 3, 5, 16)
    value = torch.randn_like(key)
    fibquant = FibQuantCodec(bits=2, block_size=1, code_bits=2, seed=9)
    turboquant = TurboQuantCodec(bits=2, dist="beta", rotation_mode="dense", seed=9)
    scalar_provenance = fibquant.get_resolved_kv_config(16)["codebook_training"]
    assert scalar_provenance["source"] == "analytic-finite-beta-scalar-control"
    assert scalar_provenance["sample_count"] == 0
    assert scalar_provenance["restarts"] == 0
    fq_payload = fibquant.encode(key, value)
    tq_payload = turboquant.encode(key, value)
    assert torch.equal(fq_payload.key_q["codes"], tq_payload.key_q["main"]["codes"])
    assert torch.equal(fq_payload.value_q["codes"], tq_payload.value_q["main"]["codes"])
    assert torch.equal(fq_payload.key_q["norm"], tq_payload.key_q["main"]["scale"])
    assert fq_payload.numel_bits() == tq_payload.numel_bits()

    fq_key, fq_value = fibquant.decode(fq_payload)
    tq_key, tq_value = turboquant.decode(tq_payload)
    # FibQuant normalizes before rotation as written in its paper, while the
    # TQ implementation computes the algebraically identical norm afterward.
    assert torch.allclose(fq_key, tq_key, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(fq_value, tq_value, atol=2.0e-6, rtol=0.0)


def test_fibquant_resolved_provenance_records_paper_fit_and_actual_rate():
    resolved = _small_codec().get_resolved_kv_config(8)
    assert resolved["codec"] == "fibquant"
    assert resolved["reference"] == "arXiv:2605.11478v1"
    assert resolved["fixed_rate"] is True
    assert resolved["calibration"] is None
    assert resolved["codebook_training"] == {
        "source": "exact-spherical-beta-via-normalized-gaussian",
        "samples_per_codeword": 30,
        "sample_count": 240,
        "restarts": 4,
        "lloyd_iterations": 25,
        "initializer": "beta-quantile-fibonacci-roberts",
        "empty_cell_repair": "max-total-sse-donor-farthest-unclaimed-sample",
    }
    assert resolved["roles"]["key"]["payload_bits_per_coordinate"] == 1.5
    assert resolved["roles"]["key"] == resolved["roles"]["value"]
    assert len(resolved["roles"]["key"]["codebook_sha256"]) == 64
    assert len(resolved["rotation_sha256"]) == 64


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bits": 2.5}, "implicit paper recipe"),
        ({"bits": 2, "code_bits": 8}, "block_size is required"),
        (
            {"bits": 2, "block_size": 4, "code_bits": 7},
            "must equal code_bits/block_size",
        ),
        ({"bits": 1.25, "block_size": 3}, "must be an integer fixed block width"),
        ({"bits": 2, "block_size": True, "code_bits": 2}, "positive int"),
        ({"bits": 9, "block_size": 1, "code_bits": 9}, "scalar TurboQuant"),
        ({"bits": 15, "block_size": 1, "code_bits": 15}, "largest paper"),
        ({"bits": math.nan, "block_size": 2, "code_bits": 4}, "positive finite"),
        ({"bits": 2, "seed": True}, "seed must be an int"),
        ({"bits": 2, "codebook_seed": False}, "codebook_seed must be an int"),
    ],
)
def test_fibquant_invalid_configuration_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FibQuantCodec(**kwargs)


def test_fibquant_runtime_shape_and_input_validation():
    codec = _small_codec()
    good = torch.randn(1, 1, 2, 8)
    with pytest.raises(ValueError, match="shapes must match"):
        codec.encode(good, torch.randn(1, 1, 3, 8))
    with pytest.raises(ValueError, match="layout"):
        codec.encode(good[0], good[0])
    with pytest.raises(TypeError, match="floating-point"):
        codec.encode(good.to(torch.int64), good.to(torch.int64))
    bad = good.clone()
    bad[0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        codec.encode(bad, good)
    too_large = torch.full_like(good, 65504.0)
    with pytest.raises(ValueError, match="fp16 header"):
        codec.encode(too_large, good)

    nondividing = FibQuantCodec(bits=1.0, block_size=3, code_bits=3)
    with pytest.raises(ValueError, match="must divide"):
        nondividing.encode(good, good)


def test_fibquant_decode_rejects_corrupt_code_and_codebook_identity():
    torch.manual_seed(13)
    codec = _small_codec()
    source = torch.randn(1, 1, 2, 8)
    compressed = codec.encode(source, source)

    bad_code = copy.deepcopy(compressed)
    bad_code.key_q["codes"][0, 0, 0, 0] = 8
    with pytest.raises(ValueError, match="outside the resolved codebook"):
        codec.decode(bad_code)

    bad_digest = copy.deepcopy(compressed)
    bad_digest.key_q["codebook_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        codec.decode(bad_digest)

    bad_rotation = copy.deepcopy(compressed)
    bad_rotation.key_q["rotation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rotation digest"):
        codec.decode(bad_rotation)

    bad_norm = copy.deepcopy(compressed)
    bad_norm.key_q["norm"][0, 0, 0, 0] = math.inf
    with pytest.raises(ValueError, match="finite and non-negative"):
        codec.decode(bad_norm)

    wrong_rate_decoder = FibQuantCodec(bits=1.0, block_size=2, code_bits=2, seed=7)
    with pytest.raises(ValueError, match="rate recipe"):
        wrong_rate_decoder.decode(compressed)
