"""Canonical public and arithmetic contracts for HayateQuant."""

import inspect

import pytest
import torch

from onecomp.kvcomp import KVCompressionConfig
from onecomp.kvcomp.codec.hayate import (
    HAYATEQUANT_BLOCK_SIZE,
    HAYATEQUANT_NAME,
    HAYATEQUANT_RATES,
    HayateQuantCodec,
    build_hayatequant,
    hayatequant_code_bits,
    hayatequant_role_rates,
)


def test_public_identity_and_constructor_are_canonical_only():
    assert HAYATEQUANT_NAME == "hayatequant"
    assert HAYATEQUANT_RATES == (2, 3, 4)
    assert HAYATEQUANT_BLOCK_SIZE == 2
    assert tuple(inspect.signature(HayateQuantCodec).parameters) == (
        "bits",
        "calibration",
        "skip_layers",
    )

    for obsolete in (
        {"block_size": 4},
        {"dist": "beta"},
        {"encoder": "exact"},
        {"n_outlier": 1},
        {"use_qjl": True},
        {"rotation_mode": "dense"},
        {"value_bits": 2},
    ):
        with pytest.raises(TypeError):
            HayateQuantCodec(**obsolete)


@pytest.mark.parametrize("rate", [True, 1, 2.5, 5, "3"])
def test_noncanonical_rates_fail_closed(rate):
    with pytest.raises(ValueError):
        build_hayatequant(rate)


@pytest.mark.parametrize(
    ("rate", "role_rates", "code_bits"),
    [
        (2, (3.0, 1.0), (6, 2)),
        (3, (4.0, 2.0), (8, 4)),
        (4, (5.0, 3.0), (10, 6)),
    ],
)
def test_fixed_role_allocation(rate, role_rates, code_bits):
    assert hayatequant_role_rates(rate) == role_rates
    assert hayatequant_code_bits(rate) == code_bits

    codec = build_hayatequant(rate, skip_layers=[1, 3])
    assert codec.get_kv_config() == {
        "name": "hayatequant",
        "bits": rate,
        "skip_layers": [1, 3],
    }
    restored = KVCompressionConfig.from_dict(
        {
            "codec": codec.get_kv_config(),
            "selector": None,
            "policy": {"name": "static"},
            "residual_window": 0,
        }
    )
    assert restored.codec.get_kv_config() == codec.get_kv_config()


@pytest.mark.parametrize("rate", HAYATEQUANT_RATES)
def test_payload_accounting_and_fused_roundtrip_match_reference_path(rate):
    torch.manual_seed(7)
    codec = build_hayatequant(rate)
    key = torch.randn(2, 3, 5, 8)
    value = torch.randn_like(key)
    payload = codec.encode(key, value)
    decoded_key, decoded_value = codec.decode(payload)
    fused_key, fused_value, fused_bits = codec.fake_quant_roundtrip(key, value)

    key_code_bits, value_code_bits = hayatequant_code_bits(rate)
    vectors = 2 * 3 * 5
    expected = vectors * ((8 // 2) * (key_code_bits + value_code_bits) + 2 * 16)
    assert payload.numel_bits() == expected
    assert fused_bits == expected
    assert torch.equal(fused_key, decoded_key)
    assert torch.equal(fused_value, decoded_value)
    assert torch.isfinite(fused_key).all()
    assert torch.isfinite(fused_value).all()


def test_resolved_contract_records_only_the_adopted_method():
    rate2 = build_hayatequant(2).get_resolved_kv_config(128)
    assert rate2["roles"]["key"]["rotation_mode"] == "hybrid"
    assert rate2["roles"]["value"]["rotation_mode"] == "dense"
    assert rate2["roles"]["key"]["scale_mode"] == "nc"
    assert rate2["roles"]["value"]["scale_mode"] == "ls_iter"
    assert rate2["assignment"]["cuda_encoder"] == "fixed_candidate_lut"
    assert rate2["assignment"]["lut_bins"] == 256
    artifacts = rate2["codebook_artifacts"]["artifacts"]
    assert {(item["block_size"], item["k"]) for item in artifacts} == {
        (2, 4),
        (2, 16),
        (2, 64),
        (2, 256),
        (2, 1024),
    }

    rate3 = build_hayatequant(3).get_resolved_kv_config(128)
    assert rate3["roles"]["key"]["rotation_mode"] == "hybrid"
    assert rate3["roles"]["value"]["rotation_mode"] == "dense"


def test_odd_head_dimension_fails_instead_of_changing_the_rate():
    codec = build_hayatequant(3)
    with pytest.raises(ValueError, match="must divide"):
        codec.encode(torch.randn(1, 1, 2, 7), torch.randn(1, 1, 2, 7))


def test_removed_experiment_environment_knobs_cannot_change_hayatequant(monkeypatch):
    torch.manual_seed(11)
    key = torch.randn(1, 2, 7, 8)
    value = torch.randn_like(key)
    expected = build_hayatequant(3).fake_quant_roundtrip(key, value)

    for name in (
        "ONECOMP_COMPILE_CODEC",
        "ONECOMP_COMPILE_DECODE_ONLY",
        "ONECOMP_TRITON_CODEC",
        "ONECOMP_TRITON_PREFILL",
    ):
        monkeypatch.setenv(name, "1")

    actual = build_hayatequant(3).fake_quant_roundtrip(key, value)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]
