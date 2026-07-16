"""CPU-only contract tests for the fair HayateQuant speed harness."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "benchmark" / "kvcomp-codec" / "hayatequant_fair_speed.py"
_SPEC = importlib.util.spec_from_file_location("hayatequant_fair_speed_tested", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
fair = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fair
_SPEC.loader.exec_module(fair)


def test_parser_exposes_single_rate_and_safe_compile_modes():
    args = fair._build_parser().parse_args(
        [  # pylint: disable=protected-access
            "--model",
            "/models/example",
            "--avg-bits",
            "3",
            "--dtype",
            "bfloat16",
            "--compile-mode",
            "decode-only",
            "--output",
            "/tmp/fair.json",
        ]
    )
    assert args.avg_bits == 3
    assert args.dtype == "bfloat16"
    assert args.compile_mode == "decode-only"
    assert not hasattr(args, "bits")  # no multi-rate option in one process


def test_compile_all_environment_is_rejected(monkeypatch):
    args = fair._build_parser().parse_args(
        [  # pylint: disable=protected-access
            "--model",
            "/models/example",
            "--compile-mode",
            "off",
            "--output",
            "/tmp/fair.json",
        ]
    )
    monkeypatch.setenv("ONECOMP_COMPILE_CODEC", "1")
    monkeypatch.setenv("ONECOMP_COMPILE_DECODE_ONLY", "0")
    with pytest.raises(RuntimeError, match="compile-all"):
        fair._configure_effective_environment(args)  # pylint: disable=protected-access


def test_decode_only_sets_both_compile_guards(monkeypatch):
    args = fair._build_parser().parse_args(
        [  # pylint: disable=protected-access
            "--model",
            "/models/example",
            "--compile-mode",
            "decode-only",
            "--fused-backend",
            "eager",
            "--output",
            "/tmp/fair.json",
        ]
    )
    names = (
        "ONECOMP_COMPILE_CODEC",
        "ONECOMP_COMPILE_DECODE_ONLY",
        "ONECOMP_TRITON_CODEC",
        "ONECOMP_TRITON_PREFILL",
    )
    original = {name: os.environ.get(name) for name in names}
    # ``_configure_effective_environment`` intentionally writes os.environ
    # directly because the benchmark runs in a fresh process.  Seed then clear
    # every key inside a nested MonkeyPatch lifetime so writes made by the
    # function cannot leak into later in-process tests.
    with monkeypatch.context() as environment:
        for name in names:
            environment.setenv(name, "__pytest_restore_sentinel__")
            environment.delenv(name)
        effective = fair._configure_effective_environment(args)  # pylint: disable=protected-access
    assert effective == {
        "ONECOMP_COMPILE_CODEC": "1",
        "ONECOMP_COMPILE_DECODE_ONLY": "1",
        "ONECOMP_TRITON_CODEC": "0",
        "ONECOMP_TRITON_PREFILL": "0",
    }
    assert {name: os.environ.get(name) for name in names} == original


def test_balanced_orders_are_position_balanced_and_deterministic():
    labels = ["a", "b", "c", "d"]
    orders = fair._balanced_orders(labels, 8, seed=17)  # pylint: disable=protected-access
    assert orders == fair._balanced_orders(labels, 8, seed=17)  # pylint: disable=protected-access
    for block in (orders[:4], orders[4:]):
        for position in range(4):
            assert sorted(order[position] for order in block) == labels
        assert all(sorted(order) == labels for order in block)


def test_summary_has_requested_tail_iqr_and_bootstrap_ci():
    row = fair._summary(  # pylint: disable=protected-access
        [1.0, 2.0, 3.0, 4.0, 5.0],
        unit="ms_per_call",
        bootstrap_resamples=200,
        seed=9,
    )
    assert row["unit"] == "ms_per_call"
    assert row["median"] == 3.0
    assert row["p5"] == pytest.approx(1.2)
    assert row["p95"] == pytest.approx(4.8)
    assert row["iqr"] == pytest.approx(2.0)
    lo, hi = row["bootstrap_median_95ci"]
    assert lo <= row["median"] <= hi


def _paired_repeat(index, ordered_values):
    return {
        "repeat": index,
        "order": [method for method, _ in ordered_values],
        "samples": [
            {
                "method": method,
                "order_position": position,
                "elapsed_ms_per_call": value,
            }
            for position, (method, value) in enumerate(ordered_values)
        ],
    }


def test_paired_speedups_are_repeatwise_ratios_and_keep_raw_values():
    raw = [
        _paired_repeat(0, [("tq", 10.0), ("scalar", 5.0), ("hq", 4.0)]),
        _paired_repeat(1, [("hq", 50.0), ("tq", 100.0), ("scalar", 25.0)]),
        _paired_repeat(2, [("scalar", 20.0), ("hq", 5.0), ("tq", 10.0)]),
    ]
    report = fair._paired_speedup_report(  # pylint: disable=protected-access
        raw,
        {
            "scalar_vs_tq": ("tq", "scalar"),
            "hq_vs_tq": ("tq", "hq"),
            "hq_vs_scalar": ("scalar", "hq"),
        },
        value_key="elapsed_ms_per_call",
        value_unit="ms_per_call",
        bootstrap_resamples=200,
        seed=31,
    )
    scalar = report["comparisons"]["scalar_vs_tq"]
    assert [row["candidate_speedup_x"] for row in scalar["raw_paired_ratios"]] == [
        2.0,
        4.0,
        0.5,
    ]
    # Median of paired ratios is 2.0. A ratio of unpaired method medians would
    # be 10/20=0.5 and would reverse the conclusion.
    assert scalar["within_process_summary"]["median"] == 2.0
    assert scalar["raw_paired_ratios"][1]["baseline_value"] == 100.0
    assert scalar["raw_paired_ratios"][1]["candidate_order_position"] == 2
    assert report["inferential_unit"] == "independently_launched_fresh_process"


def test_paired_speedups_reject_duplicate_or_missing_method():
    raw = [_paired_repeat(0, [("tq", 2.0), ("tq", 1.0)])]
    with pytest.raises(ValueError, match="duplicate"):
        fair._paired_speedup_report(  # pylint: disable=protected-access
            raw,
            {"scalar_vs_tq": ("tq", "scalar")},
            value_key="elapsed_ms_per_call",
            value_unit="ms_per_call",
            bootstrap_resamples=10,
            seed=1,
        )


def test_cuda_event_unit_normalisation_is_milliseconds_per_call():
    assert fair._normalise_elapsed_ms(9.0, 3) == 3.0  # pylint: disable=protected-access
    with pytest.raises(ValueError, match="positive"):
        fair._normalise_elapsed_ms(9.0, 0)  # pylint: disable=protected-access


class _DummyTextConfig:
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 4096
    num_hidden_layers = 32
    max_position_embeddings = 8192


class _DummyConfig:
    text_config = _DummyTextConfig()


def test_model_kv_shape_is_derived_from_config_not_hard_coded():
    shape = fair._resolve_model_shape(_DummyConfig())  # pylint: disable=protected-access
    assert shape["num_attention_heads"] == 32
    assert shape["num_key_value_heads"] == 8
    assert shape["head_dim"] == 128
    assert shape["gqa_group_size"] == 4
    assert shape["head_dim_derivation"] == "config:hidden_size/num_attention_heads"


def test_model_evaluation_role_does_not_claim_independent_confirmation():
    assert (
        fair._model_evaluation_role(  # pylint: disable=protected-access
            "/models/llama-3.1-8b-instruct"
        )
        == "development_and_recipe_selection"
    )
    assert (
        fair._model_evaluation_role("/models/Qwen3-8B")  # pylint: disable=protected-access
        == "additional_development_evaluation_independence_not_claimed"
    )
    assert (
        fair._model_evaluation_role("/models/custom")  # pylint: disable=protected-access
        == "custom_evaluation_role_unspecified"
    )


def test_e2e_context_capacity_is_checked_before_model_load():
    capacity = fair._resolve_model_capacity(_DummyConfig())  # pylint: disable=protected-access
    assert capacity == {
        "tokens": 8192,
        "derivation": "config:max_position_embeddings",
        "original_max_position_embeddings": None,
    }
    contract = fair._validate_e2e_context_capacity(
        4096, 64, capacity
    )  # pylint: disable=protected-access
    assert contract["required_context_tokens"] == 4160
    assert contract["fits"] is True
    with pytest.raises(ValueError, match="exceeds"):
        fair._validate_e2e_context_capacity(8192, 1, capacity)  # pylint: disable=protected-access


def test_recipe_digest_is_canonical_and_sensitive_to_config():
    first = {"codec": {"bits": 3, "dist": "beta"}, "shape": [1, 8, 1, 128]}
    reordered = {"shape": [1, 8, 1, 128], "codec": {"dist": "beta", "bits": 3}}
    changed = {"codec": {"bits": 4, "dist": "beta"}, "shape": [1, 8, 1, 128]}
    assert fair._recipe_sha256(first) == fair._recipe_sha256(
        reordered
    )  # pylint: disable=protected-access
    assert fair._recipe_sha256(first) != fair._recipe_sha256(
        changed
    )  # pylint: disable=protected-access


@pytest.mark.parametrize(
    ("rate", "expected_key", "expected_value"),
    [
        (2, (2, 2, "hybrid", "nc", "nc"), (2, 2, "dense", "ls_iter", "ls_iter")),
        (3, (3, 2, "hybrid", "nc", "nc"), (3, 2, "hybrid", "ls_iter", "ls_iter")),
        (4, (4, 2, "hybrid", "nc", "nc"), (4, 2, "hybrid", "ls_iter", "ls_iter")),
    ],
)
def test_proposal_recipes_record_exact_rate_specific_role_resolution(
    rate, expected_key, expected_value
):
    specs = fair._method_specs(rate)  # pylint: disable=protected-access
    codecs = {spec.label: spec.codec_factory() for spec in specs}
    proposal_contract = fair._validate_proposal_contract(  # pylint: disable=protected-access
        specs, codecs, rate
    )
    assert proposal_contract["formal_proposals"] == ["HQ", "AdaptiveHQ"]
    assert proposal_contract["same_total_pair_code_bit_budget"] is True
    selector = proposal_contract["rows"]["AdaptiveHQ"]["selector"]
    assert selector["policy_version"] == fair.ADAPTIVE_HQ_POLICY_VERSION
    assert selector["distortion_table_version"] == fair.ADAPTIVE_HQ_DISTORTION_TABLE_VERSION
    assert selector["objective"] == fair.ADAPTIVE_HQ_OBJECTIVE
    assert (
        selector["selector_carrier_distortion_amortization"]
        == fair.ADAPTIVE_HQ_CARRIER_AMORTIZATION
    )
    assert selector["candidate_pairs"] == [
        [2 * rate, 2 * rate],
        [2 * rate + 1, 2 * rate - 1],
    ]
    assert selector["selector_bits_per_encode"] == 1
    records, hashes = fair._method_config_records(  # pylint: disable=protected-access
        specs, codecs, 128
    )
    fixed = records[specs[1].label]["resolved_at_model_head_dim"]
    key = fixed["roles"]["key"]
    value = fixed["roles"]["value"]
    assert (
        key["bits"],
        key["block_size"],
        key["rotation_mode"],
        key["scale_mode"],
        key["mathematical_scale_mode"],
    ) == expected_key
    assert (
        value["bits"],
        value["block_size"],
        value["rotation_mode"],
        value["scale_mode"],
        value["mathematical_scale_mode"],
    ) == expected_value
    assert key["source_law"] == value["source_law"] == "gaussian"
    assert key["rotation_seed"] == value["rotation_seed"] == 0
    assert key["code_bits_per_block"] == 2 * rate + 1
    assert value["code_bits_per_block"] == 2 * rate - 1

    adaptive = records[specs[-1].label]["resolved_at_model_head_dim"]
    assert {adaptive["roles"][role]["rotation_mode"] for role in ("key", "value")} == {
        "payload_dependent_candidate_rate"
    }
    assert adaptive["adaptive_allocation"]["selector_incremental_stored_bits_per_encode"] == 0
    assert hashes[specs[-1].label] == fair._recipe_sha256(  # pylint: disable=protected-access
        records[specs[-1].label]
    )

    tq = records[specs[0].label]["resolved_at_model_head_dim"]["roles"]
    hq = records[specs[1].label]["resolved_at_model_head_dim"]["roles"]
    assert {role["scale_mode"] for role in tq.values()} == {"l2"}
    assert {role["scale_mode"] for role in hq.values()} == {"nc", "ls_iter"}
    assert {role["source_law"] for role in tq.values()} == {"beta"}
    assert {role["source_law"] for role in hq.values()} == {"gaussian"}


def test_run_fingerprint_excludes_only_output_mechanics():
    args_a = fair._build_parser().parse_args(
        [  # pylint: disable=protected-access
            "--model",
            "/models/example",
            "--avg-bits",
            "3",
            "--output",
            "/tmp/a.json",
        ]
    )
    args_b = fair._build_parser().parse_args(
        [  # pylint: disable=protected-access
            "--model",
            "/models/example",
            "--avg-bits",
            "3",
            "--output",
            "/tmp/b.json",
            "--overwrite",
        ]
    )
    semantic_a = fair._semantic_config(args_a)  # pylint: disable=protected-access
    semantic_b = fair._semantic_config(args_b)  # pylint: disable=protected-access
    assert semantic_a == semantic_b
    assert "output" not in semantic_a and "overwrite" not in semantic_a
    assert fair._run_fingerprint(
        semantic_a, "abc"
    ) == fair._run_fingerprint(  # pylint: disable=protected-access
        semantic_b, "abc"
    )
    changed = dict(semantic_b, avg_bits=4)
    assert fair._run_fingerprint(
        semantic_a, "abc"
    ) != fair._run_fingerprint(  # pylint: disable=protected-access
        changed, "abc"
    )


def test_public_payload_scope_and_runtime_audit_reject_packed_serving_claim():
    assert fair.PUBLIC_LOGICAL_PAYLOAD_SCOPE == {
        "api": "public KVCodec.encode/decode",
        "representation": "materialized_logical_payload",
        "index_tensor_dtype": "int64",
        "index_tensors_bit_packed": False,
        "stored_bits_scope": "fixed_width_per_token_data_plane_growth_only",
        "stored_bits_semantics": (
            "codes plus signed-fp16 scales and enabled mask/QJL/outlier data-plane fields"
        ),
        "excluded_from_stored_bits": (
            "self-describing control tags, Python/container bytes, allocator-resident tensor "
            "bytes, and global codebook/LUT/rotation state"
        ),
        "control_state_interpretation": (
            "out-of-band/cache-level configuration, not physical serving memory"
        ),
        "physical_memory_interpretation": (
            "not the actual tensor-memory footprint and not a packed serving-cache result"
        ),
        "self_describing_control_tags_counted": False,
        "python_container_bytes_counted": False,
        "persistent_codec_state_counted": False,
        "packed_serving_claim_allowed": False,
    }
    codec = fair._method_specs(3)[0].codec_factory()  # pylint: disable=protected-access
    key = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    payload = codec.encode(key, key)
    audit = fair._audit_public_payload(torch, payload)  # pylint: disable=protected-access
    assert audit["all_index_tensors_int64"] is True
    assert audit["index_tensors_bit_packed"] is False
    assert audit["materialized_index_tensor_bits"] == audit["index_elements"] * 64
    assert audit["materialized_index_tensor_bits"] > payload.numel_bits()
    assert audit["all_scale_tensors_signed_fp16"] is True
    assert audit["materialized_payload_tensor_bytes"] * 8 == (
        audit["materialized_index_tensor_bits"] + audit["materialized_scale_tensor_bits"]
    )
    assert audit["materialized_payload_tensor_bytes"] * 8 != audit["logical_total_stored_bits"]
    assert "excludes Python containers" in audit["materialized_payload_tensor_scope"]
