"""Tests for the independent paper-defined BlockQuant quality baseline."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from onecomp.kvcomp._registry import build_codec, list_codecs
from onecomp.kvcomp.codec.blockquant import (
    BLOCKQUANT_CODEBOOK_SPEC,
    BLOCKQUANT_REFERENCE,
    BLOCKQUANT_REMAINDER_POLICY,
    BlockQuantCodec,
    blockquant_codebook,
)
from onecomp.kvcomp.codec.quality_baselines import (
    BLOCKQUANT_PAPER_METHOD,
    blockquant_quality_provenance,
    build_blockquant_paper,
    is_quality_only_baseline,
)
from onecomp.kvcomp.codec.rotation import random_orthogonal


def _inputs(head_dim: int = 6, *, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(20260716 + head_dim)
    key = torch.randn(1, 1, 2, head_dim, generator=generator, dtype=dtype)
    value = torch.randn(1, 1, 2, head_dim, generator=generator, dtype=dtype)
    return key, value


def _small_codec(**overrides) -> BlockQuantCodec:
    arguments = {
        "bits": 1,
        "value_bits": 1,
        "block_size": 3,
        "key_scale_mode": "raw",
        "value_scale_mode": "raw",
        "rotation_seed": 7,
        "codebook_seed": 11,
    }
    arguments.update(overrides)
    return BlockQuantCodec(**arguments)


def _part_chat_and_source(tensor: torch.Tensor, part: dict):
    ambient_dim = part["ambient_dim"]
    bits = part["bits_per_coordinate"]
    seed = part["rotation"]["seed"]
    rotation = random_orthogonal(ambient_dim, seed)
    pieces = []
    if part["full_blocks"]:
        codebook = blockquant_codebook(
            ambient_dim,
            part["block_size"],
            bits,
            codebook_seed=11,
        )
        pieces.append(codebook[part["full_codes"]].reshape(*tensor.shape[:-1], -1))
    if part["remainder"]:
        codebook = blockquant_codebook(
            ambient_dim,
            part["remainder"],
            bits,
            codebook_seed=11,
        )
        pieces.append(codebook[part["tail_code"]])
    return torch.cat(pieces, dim=-1), tensor.float() @ rotation, rotation


def test_registry_identity_is_independent_blockquant():
    assert "blockquant" in list_codecs()
    codec = build_codec(
        "blockquant",
        bits=1,
        value_bits=1,
        block_size=3,
        key_scale_mode="raw",
        value_scale_mode="raw",
    )
    assert type(codec) is BlockQuantCodec
    assert codec.name == "blockquant"
    assert codec.flag_calibration is False
    assert codec.get_kv_config()["independent_baseline"] is True


def test_formal_blockquant_baseline_is_fixed_seed_quality_only_not_reproduction():
    codec = build_blockquant_paper(3, head_dim=128)
    provenance = blockquant_quality_provenance(codec)
    assert BLOCKQUANT_PAPER_METHOD == "BlockQuant-paper"
    assert is_quality_only_baseline(BLOCKQUANT_PAPER_METHOD)
    assert provenance["method_role"] == "external_quality_baseline"
    assert provenance["external_quality_baseline"] is True
    assert provenance["proposal_method"] is False
    assert provenance["paper_comparable"] is False
    assert provenance["paper_reproduction"] is False
    assert provenance["speed_comparison"] is False
    assert provenance["formal_rate_slot"] == 3
    assert provenance["nominal_payload_bits_per_coordinate"] is None
    assert provenance["top32_completion"] == "per_batch_head_update_top_l2_v1"
    assert "not the paper five-random-seed aggregate" in provenance["random_seed_protocol"]


def test_formal_blockquant_builder_fails_closed_outside_sole_operating_point():
    with pytest.raises(ValueError, match="formal rate slot 3"):
        build_blockquant_paper(2, head_dim=128)
    with pytest.raises(ValueError, match="head_dim=128"):
        build_blockquant_paper(3, head_dim=64)


def test_unpublished_lut_and_unversioned_codebooks_fail_closed():
    with pytest.raises(ValueError, match="assignment_mode='exact'"):
        BlockQuantCodec(assignment_mode="lut")
    with pytest.raises(ValueError, match="unversioned centroid fitting"):
        BlockQuantCodec(codebook_spec="latest")
    assert BLOCKQUANT_CODEBOOK_SPEC == "onecomp.blockquant.finite_beta_lloyd.v1"


def test_codebook_is_deterministic_finite_spherical_and_data_free():
    first = blockquant_codebook(6, 3, 1, codebook_seed=11)
    second = blockquant_codebook(6, 3, 1, codebook_seed=11)
    assert first.shape == (8, 3)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first.norm(dim=1) <= 1.0 + 1.0e-6)


@pytest.mark.parametrize(
    "bits,distortion_ceiling",
    [(1, 0.360), (2, 0.104), (3, 0.028)],
)
def test_p3_codebook_quality_tracks_paper_finite_rate_constants(bits, distortion_ceiling):
    """Guard against a nominal BlockQuant row backed by an underfit codebook."""
    ambient_dim = 128
    codebook = blockquant_codebook(ambient_dim, 3, bits, codebook_seed=0).double()
    generator = torch.Generator(device="cpu").manual_seed(818_271 + bits)
    gaussian = torch.randn(50_000, ambient_dim, generator=generator, dtype=torch.float64)
    samples = (gaussian / gaussian.norm(dim=1, keepdim=True))[:, :3]
    errors = []
    for chunk in samples.split(5_000):
        squared_distance = (chunk[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
        errors.append(squared_distance.min(dim=-1).values)
    # A d-vector contains d/p blocks, hence this is the unit-vector distortion
    # corresponding to the per-block validation MSE.  The ceilings bracket the
    # p=3 finite-rate values in Theorem 2 / Appendix C while allowing d=128 and
    # deterministic numerical-integration error.
    unit_vector_distortion = torch.cat(errors).mean() * (ambient_dim / 3.0)
    assert float(unit_vector_distortion) < distortion_ceiling


def test_raw_encode_is_exact_nearest_codebook_and_decode_formula():
    codec = _small_codec()
    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    part = compressed.key_q["body"]["part"]
    codebook = blockquant_codebook(6, 3, 1, codebook_seed=11)
    rotation = random_orthogonal(6, part["rotation"]["seed"])
    unit = key.float() / key.float().norm(dim=-1, keepdim=True)
    blocks = (unit @ rotation).reshape(1, 1, 2, 2, 3)
    distances = (blocks.unsqueeze(-2) - codebook).square().sum(dim=-1)
    assert torch.equal(part["full_codes"], distances.argmin(dim=-1))

    chat = codebook[part["full_codes"]].reshape_as(key)
    expected = (chat @ rotation.t()) * part["scale"].float()
    decoded_key, decoded_value = codec.decode(compressed)
    torch.testing.assert_close(decoded_key, expected, rtol=0.0, atol=0.0)
    assert decoded_key.dtype == key.dtype
    assert decoded_value.dtype == value.dtype

    # Two vectors per role: 2*6 code bits + 2 fp16 scales = 44 bits/role.
    assert compressed.meta["stored_payload_bits"] == 24
    assert compressed.meta["stored_scale_bits"] == 64
    assert compressed.numel_bits() == 88


@pytest.mark.parametrize("canonical,alias", [("bsm", "ls"), ("ub", "ip")])
def test_scale_aliases_are_exactly_identical(canonical, alias):
    key, value = _inputs(6)
    first = _small_codec(key_scale_mode=canonical, value_scale_mode=canonical).encode(key, value)
    second = _small_codec(key_scale_mode=alias, value_scale_mode=alias).encode(key, value)
    first_key = first.key_q["body"]["part"]
    second_key = second.key_q["body"]["part"]
    assert torch.equal(first_key["full_codes"], second_key["full_codes"])
    assert torch.equal(first_key["scale"], second_key["scale"])
    first_decoded = _small_codec(key_scale_mode=canonical, value_scale_mode=canonical).decode(
        first
    )
    second_decoded = _small_codec(key_scale_mode=alias, value_scale_mode=alias).decode(second)
    torch.testing.assert_close(first_decoded[0], second_decoded[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_decoded[1], second_decoded[1], rtol=0.0, atol=0.0)


def test_raw_bsm_and_ub_scales_follow_paper_equations():
    key, value = _inputs(6)
    for mode in ("raw", "bsm", "ub"):
        codec = _small_codec(key_scale_mode=mode, value_scale_mode=mode)
        compressed = codec.encode(key, value)
        part = compressed.key_q["body"]["part"]
        chat, source_rotated, _ = _part_chat_and_source(key, part)
        if mode == "raw":
            expected = source_rotated.norm(dim=-1, keepdim=True)
        elif mode == "bsm":
            expected = (source_rotated * chat).sum(dim=-1, keepdim=True) / chat.square().sum(
                dim=-1, keepdim=True
            )
        else:
            expected = source_rotated.square().sum(dim=-1, keepdim=True) / (
                source_rotated * chat
            ).sum(dim=-1, keepdim=True)
        assert torch.equal(part["scale"], expected.to(torch.float16))

    raw = _small_codec(key_scale_mode="raw", value_scale_mode="raw").fake_quantize(key, value)[0]
    bsm = _small_codec(key_scale_mode="bsm", value_scale_mode="bsm").fake_quantize(key, value)[0]
    raw_mse = (raw.float() - key.float()).square().sum()
    bsm_mse = (bsm.float() - key.float()).square().sum()
    # fp16 scalar rounding can perturb the analytic optimum only at roundoff scale.
    assert float(bsm_mse) <= float(raw_mse) + 1.0e-5


def test_ub_ip_rejects_zero_vector_instead_of_inventing_a_scale():
    codec = _small_codec(key_scale_mode="ub", value_scale_mode="bsm")
    key, value = _inputs(6)
    key.zero_()
    with pytest.raises(ValueError, match="undefined for a zero input"):
        codec.encode(key, value)


@pytest.mark.parametrize("head_dim,remainder", [(8, 2), (7, 1)])
def test_p3_remainder_is_fail_closed_then_explicitly_encoded(head_dim, remainder):
    key, value = _inputs(head_dim)
    with pytest.raises(ValueError, match="requires block_size=3 to divide"):
        _small_codec().encode(key, value)

    codec = _small_codec(remainder_policy=BLOCKQUANT_REMAINDER_POLICY)
    compressed = codec.encode(key, value)
    part = compressed.key_q["body"]["part"]
    assert part["remainder"] == remainder
    assert tuple(part["tail_code"].shape) == tuple(key.shape[:-1])
    assert part["tail_codebook"]["block_dim"] == remainder
    assert part["tail_codebook"]["codebook_size"] == 1 << remainder
    decoded = codec.decode(compressed)
    assert decoded[0].shape == key.shape and decoded[1].shape == value.shape
    expected_bits = 2 * (key.numel() + (key.numel() // head_dim) * 16)
    assert compressed.numel_bits() == expected_bits


def test_llama31_recipe_has_exact_declared_logical_rates_without_dynamic_mask():
    codec = BlockQuantCodec.llama31_paper_recipe(tuple(range(32)))
    accounting = codec.logical_bits_for_shape((1, 1, 1, 128))
    assert accounting["key"] == {
        "code_bits": 32 * 4 + 96 * 3,
        "payload_bits": 32 * 4 + 96 * 3,
        "fp16_scale_bits": 32,
        "scale_fields_per_vector": 2,
        "fixed_channel_identity_bits": 0,
        "dynamic_channel_identity_bits": 0,
        "dynamic_channel_identity_fields": 0,
        "dynamic_channel_identity_coding": None,
        "total_bits": 448,
        "bits_per_coordinate": 3.5,
    }
    assert accounting["value"]["payload_bits"] == 128 * 2
    assert accounting["value"]["fp16_scale_bits"] == 16
    assert accounting["value"]["total_bits"] == 272
    assert accounting["value"]["bits_per_coordinate"] == 2.125
    assert accounting["total_bits"] == 720
    config = codec.get_kv_config()
    assert config["paper_kv_recipe"] is True
    assert config["key_scale_mode"] == "ub"
    assert config["value_scale_mode"] == "bsm"
    assert config["assignment_mode"] == "exact-full-codebook"
    assert config["lut"] is None


def test_llama31_recipe_requires_32_unique_explicit_channels_and_128d_shape():
    with pytest.raises(ValueError, match="requires p=3"):
        BlockQuantCodec.llama31_paper_recipe(tuple(range(31)))
    with pytest.raises(ValueError, match="unique"):
        BlockQuantCodec.llama31_paper_recipe(tuple(range(31)) + (0,))
    codec = BlockQuantCodec.llama31_paper_recipe(tuple(range(32)))
    with pytest.raises(ValueError, match="head_dim=128"):
        codec.logical_bits_for_shape((1, 1, 1, 64))
    with pytest.raises(ValueError, match="cannot be overridden"):
        BlockQuantCodec.llama31_paper_recipe(tuple(range(32)), bits=2)


def test_payload_records_source_completion_exact_search_and_codebook_hashes():
    codec = _small_codec(key_scale_mode="bsm", value_scale_mode="bsm")
    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    resolved = compressed.meta["resolved_codec_config"]
    provenance = resolved["provenance"]
    assert provenance["reference"] == BLOCKQUANT_REFERENCE
    assert provenance["independent_baseline"] is True
    assert provenance["proxy_for_another_codec"] is False
    assert provenance["assignment"]["mode"] == "exact-full-codebook-nearest-squared-l2"
    assert provenance["assignment"]["lut"] is None
    assert provenance["codebook"]["published_centroids_or_training_code"] is False
    assert provenance["codebook"]["reproducibility_completion"] == BLOCKQUANT_CODEBOOK_SPEC
    record = compressed.key_q["body"]["part"]["full_codebook"]
    assert record["sha256"]
    assert record["codebook_seed"] == 11
    assert record["codebook_spec"] == BLOCKQUANT_CODEBOOK_SPEC


def test_decode_rejects_tampered_codebook_and_accounting_provenance():
    codec = _small_codec(key_scale_mode="bsm", value_scale_mode="bsm")
    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    tampered = copy.deepcopy(compressed)
    tampered.key_q["body"]["part"]["full_codebook"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="codebook provenance/hash mismatch"):
        codec.decode(tampered)

    tampered = copy.deepcopy(compressed)
    tampered.meta["stored_bits"] += 1
    with pytest.raises(ValueError, match="stored-bit/provenance"):
        codec.decode(tampered)


def test_fixed_channel_split_counts_two_scales_and_no_unreported_identity_field():
    codec = BlockQuantCodec(
        bits=1,
        value_bits=1,
        block_size=3,
        key_scale_mode="bsm",
        value_scale_mode="bsm",
        key_outlier_indices=(0, 2),
        outlier_bits=2,
        remainder_policy=BLOCKQUANT_REMAINDER_POLICY,
        rotation_seed=7,
        codebook_seed=11,
    )
    accounting = codec.logical_bits_for_shape((1, 1, 2, 6))
    # Per key vector: 2*2 + 4*1 code bits and two fp16 scales = 40 bits.
    assert accounting["key"]["payload_bits"] == 16
    assert accounting["key"]["fp16_scale_bits"] == 64
    assert accounting["key"]["total_bits"] == 80
    assert accounting["key"]["fixed_channel_identity_bits"] == 0
    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    decoded_key, decoded_value = codec.decode(compressed)
    assert decoded_key.shape == key.shape and decoded_value.shape == value.shape


def test_dynamic_top_l2_completion_is_per_head_and_charges_identity_bits():
    codec = _small_codec(
        key_n_outlier=2,
        outlier_bits=2,
        key_scale_mode="bsm",
        value_scale_mode="bsm",
        remainder_policy=BLOCKQUANT_REMAINDER_POLICY,
    )
    key, value = _inputs(6)
    key = key.expand(1, 2, 2, 6).clone()
    value = value.expand_as(key).clone()
    key[0, 0, :, 0] *= 20.0
    key[0, 0, :, 3] *= 20.0
    key[0, 1, :, 1] *= 30.0
    key[0, 1, :, 5] *= 30.0
    compressed = codec.encode(key, value)
    indices = compressed.key_q["body"]["outlier_indices"]
    energy = key.float().square().sum(dim=-2)
    expected = torch.sort(
        torch.argsort(energy, dim=-1, descending=True, stable=True)[..., :2], dim=-1
    ).values
    assert torch.equal(indices, expected)

    accounting = codec.logical_bits_for_shape(tuple(key.shape))
    # C(6,2)=15 fixed subsets -> 4 bits, once for each of two batch/head updates.
    assert accounting["key"]["code_bits"] == 32
    assert accounting["key"]["dynamic_channel_identity_bits"] == 8
    assert accounting["key"]["payload_bits"] == 40
    assert accounting["key"]["fp16_scale_bits"] == 128
    assert compressed.numel_bits() == 256
    decoded_key, decoded_value = codec.decode(compressed)
    assert decoded_key.shape == key.shape and decoded_value.shape == value.shape
    assert codec.teacher_forced_temporal_dependency() == (
        "dynamic_outlier_channels_over_update_tokens"
    )


def test_dynamic_llama31_completion_separates_conditional_paper_rate_from_stored_rate():
    codec = BlockQuantCodec.llama31_dynamic_top_l2_completion()
    accounting = codec.logical_bits_for_shape((1, 1, 1, 128))
    assert accounting["key"]["code_bits"] == 416
    assert accounting["key"]["fp16_scale_bits"] == 32
    assert accounting["key"]["dynamic_channel_identity_bits"] == 101
    assert accounting["key"]["total_bits"] == 549
    assert accounting["value"]["total_bits"] == 272
    assert accounting["total_bits"] == 821
    selection = codec.get_kv_config()["key_outlier_selection"]
    assert selection["completion"] == "per_batch_head_update_top_l2_v1"
    assert selection["identity_bits_counted"] is True


def test_dynamic_identity_payload_rejects_duplicates():
    codec = _small_codec(
        key_n_outlier=2,
        outlier_bits=2,
        remainder_policy=BLOCKQUANT_REMAINDER_POLICY,
    )
    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    indices = compressed.key_q["body"]["outlier_indices"]
    indices[..., 1] = indices[..., 0]
    with pytest.raises(ValueError, match="identities are invalid or repeated"):
        codec.decode(compressed)


def test_inputs_and_payloads_fail_closed_on_nonfinite_or_out_of_range_codes():
    codec = _small_codec()
    key, value = _inputs(6)
    key[..., 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        codec.encode(key, value)

    key, value = _inputs(6)
    compressed = codec.encode(key, value)
    compressed.key_q["body"]["part"]["full_codes"][0, 0, 0, 0] = 8
    with pytest.raises(ValueError, match="outside its codebook"):
        codec.decode(compressed)


def test_codec_is_token_local_and_introduces_no_future_token_statistic():
    assert _small_codec().teacher_forced_temporal_dependency() is None
