"""Independent audit guards for the published external codec baselines."""

from __future__ import annotations

import math

import pytest
import torch

from onecomp.kvcomp.codec.blockquant import BlockQuantCodec
from onecomp.kvcomp.codec.fibquant import (
    FIBQUANT_SOURCE_TEX_SHA256,
    FibQuantCodec,
)
from onecomp.kvcomp.codec.quality_baselines import (
    blockquant_quality_provenance,
    build_blockquant_paper,
    build_fibquant_paper,
    fibquant_quality_provenance,
)
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec


def test_turboquant_rejects_nonfinite_inputs_before_code_assignment():
    codec = TurboQuantCodec(bits=2)
    good = torch.zeros(1, 1, 1, 8)
    for bad_value in (math.nan, math.inf, -math.inf):
        bad = good.clone()
        bad[..., 0] = bad_value
        with pytest.raises(ValueError, match="non-finite"):
            codec.encode(bad, good)


def test_turboquant_rejects_an_fp16_scale_overflow_instead_of_emitting_inf():
    codec = TurboQuantCodec(bits=2)
    source = torch.full((1, 1, 1, 8), 65_504.0)
    with pytest.raises(ValueError, match="fp16 data-plane field"):
        codec.encode(source, source)


def test_turboquant_code_assignment_is_scale_invariant_below_the_old_epsilon():
    generator = torch.Generator(device="cpu").manual_seed(71)
    source = torch.randn(1, 1, 1, 8, generator=generator)
    codec = TurboQuantCodec(bits=2, dist="beta", rotation_mode="dense", seed=3)
    ordinary = codec.encode(source, source).key_q["main"]["codes"]
    tiny = codec.encode(source * 1.0e-12, source * 1.0e-12).key_q["main"]["codes"]
    assert torch.equal(tiny, ordinary)


@pytest.mark.parametrize(
    "key,value,exception,match",
    [
        (
            torch.zeros(1, 1, 1, 8),
            torch.zeros(1, 1, 2, 8),
            ValueError,
            "shapes must match",
        ),
        (
            torch.zeros(1, 1, 1, 8),
            torch.zeros(1, 1, 1, 8, dtype=torch.float64),
            ValueError,
            "same dtype",
        ),
        (
            torch.zeros(1, 1, 1, 8, dtype=torch.int64),
            torch.zeros(1, 1, 1, 8, dtype=torch.int64),
            TypeError,
            "floating-point",
        ),
    ],
)
def test_turboquant_rejects_payload_contract_drift(key, value, exception, match):
    with pytest.raises(exception, match=match):
        TurboQuantCodec(bits=2).encode(key, value)


def test_turboquant_qjl_switch_is_boolean_not_an_implicit_rate_change():
    with pytest.raises(ValueError, match="use_qjl must be bool"):
        TurboQuantCodec(bits=2, use_qjl=1)


@pytest.mark.parametrize(
    "base_bits,outlier_bits,nominal_without_identities",
    [(2, 3, 2.5), (3, 4, 3.5)],
)
def test_turboquant_mixed_precision_ledger_separates_identity_bits(
    base_bits, outlier_bits, nominal_without_identities
):
    # The source's 2.5-bit equation omits the two 16-bit norms.  For d=128,
    # the stated 32/96 code split plus those norms reaches the nominal label.
    # This dynamic completion additionally needs one 128-bit role mask.
    shape = (1, 2, 4, 128)
    source = torch.randn(shape)
    codec = TurboQuantCodec(
        bits=base_bits,
        n_outlier=32,
        outlier_bits=outlier_bits,
        rotation_mode="dense",
    )
    vectors = math.prod(shape[:-1])
    code_bits = vectors * (32 * outlier_bits + 96 * base_bits)
    scale_bits = vectors * 2 * 16
    identity_bits = 128
    role_bits = codec._stored_bits(source, base_bits, False)  # pylint: disable=protected-access
    assert role_bits == code_bits + scale_bits + identity_bits
    assert (code_bits + scale_bits) / float(vectors * 128) == nominal_without_identities
    assert role_bits / float(vectors * 128) == nominal_without_identities + 1.0 / vectors


def test_turboquant_prod_ledger_counts_b_minus_one_codes_qjl_and_two_norms():
    source = torch.randn(1, 2, 4, 128)
    target_bits = 3
    codec = TurboQuantCodec(
        bits=target_bits - 1,
        use_qjl=True,
        qjl_roles="both",
        rotation_mode="dense",
    )
    vectors = source.numel() // source.shape[-1]
    expected_per_role = source.numel() * target_bits + vectors * 2 * 16
    assert codec._stored_bits(source, target_bits - 1, True) == expected_per_role
    compressed = codec.encode(source, source)
    assert compressed.meta["stored_bits"] == 2 * expected_per_role


def test_fibquant_is_method_comparable_but_not_a_centroid_reproduction():
    provenance = fibquant_quality_provenance(build_fibquant_paper(2))
    assert provenance["paper_comparable"] is True
    assert provenance["paper_reproduction"] is False
    assert provenance["source_tex_sha256"] == FIBQUANT_SOURCE_TEX_SHA256
    resolved = FibQuantCodec(bits=2).get_resolved_kv_config(8)
    assert resolved["paper_reproduction"] is False
    assert resolved["source_tex_sha256"] == FIBQUANT_SOURCE_TEX_SHA256


def test_blockquant_provenance_exposes_every_material_paper_completion():
    codec = build_blockquant_paper(3)
    provenance = blockquant_quality_provenance(codec)
    assert provenance["paper_comparable"] is False
    assert provenance["paper_reproduction"] is False
    resolved = codec.get_resolved_kv_config(128)
    missing = resolved["provenance"]["kv_recipe_under_specification"]
    assert missing == {
        "objective_scale_mode_per_key_value_published": False,
        "completion": "key-ub-value-bsm-v1",
        "p3_tail_for_32_or_128_dimensions_published": False,
        "channel_identity_lifecycle_or_coding_published": False,
        "rotation_sharing_or_role_seed_derivation_published": False,
        "centroids_or_training_code_published": False,
        "lut_grid_or_candidate_hyperparameters_published": False,
    }
    assert isinstance(codec, BlockQuantCodec)
