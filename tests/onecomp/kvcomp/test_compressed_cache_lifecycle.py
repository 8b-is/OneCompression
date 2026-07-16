"""Focused lifecycle/accounting regressions for compressed KV caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedConfig

from onecomp.kvcomp._base import CompressedKV, KVCodec
from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.codec.affine import AffineKVCodec, KIVICodec
from onecomp.kvcomp.codec.halo import HALOCodec
from onecomp.kvcomp.codec.hayate.codec import HayateQuantCodec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec


@dataclass
class _MarkerCodec(KVCodec):
    """Token codec whose visible offset detects missed/double quantization."""

    bits: int = 2
    key_axis: str = "token"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None):
        stored = (key.numel() + value.numel()) * self.bits
        return CompressedKV(
            key_q=key + 100.0,
            value_q=value + 200.0,
            meta={"dtype": key.dtype, "stored_bits": stored, "passthrough": True},
        )

    def decode(self, compressed, ctx=None):
        return compressed.key_q, compressed.value_q


@dataclass
class _ZeroBitConstantCodec(KVCodec):
    """Lossy constant reconstruction with a genuinely zero-bit payload."""

    key_axis: str = "token"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None):
        return CompressedKV(
            key_q=torch.zeros_like(key),
            value_q=torch.zeros_like(value),
            meta={"stored_bits": 0},
        )

    def decode(self, compressed, ctx=None):
        return compressed.key_q, compressed.value_q


def _sliding_config(window: int = 5) -> PreTrainedConfig:
    config = PreTrainedConfig()
    config.num_hidden_layers = 1
    config.sliding_window = window
    config.layer_types = ["sliding_attention"]
    return config


def _token_rows(n: int, d: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(n * d, dtype=torch.float32).reshape(1, 1, n, d) / 10.0
    return key, key + 0.25


def _adaptive_hq_codec(min_tokens: int = 2) -> HayateQuantCodec:
    """Small scalar geometry keeps causal-state tests fast and unambiguous."""
    return HayateQuantCodec(
        bits=2,
        value_bits=2,
        block_size=1,
        max_codebook_bits=3,
        code_bits_per_pair_block=4,
        key_bit_fraction=0.5,
        adaptive_bit_split="online_stats",
        adaptive_bit_margin=1,
        adaptive_stats_min_tokens=min_tokens,
        key_error_weight=1.0,
        value_error_weight=1.0,
        key_scale_mode="rms",
        value_scale_mode="rms",
        encoder="exact",
    )


def _constant_update(key_amplitude: float, value_amplitude: float, seq: int = 1, device="cpu"):
    key = torch.full((1, 1, seq, 8), key_amplitude, device=device)
    value = torch.full_like(key, value_amplitude)
    return key, value


@pytest.mark.parametrize(
    "codec,skip_layers",
    [
        (_MarkerCodec(), None),
        (KIVICodec(group_size=2), None),
    ],
    ids=["token", "channel"],
)
def test_sliding_cache_fails_closed_before_updates(codec, skip_layers):
    """Never expose cumulative history bits as the resident token-set rate."""
    with pytest.raises(NotImplementedError, match="exact logical data-plane stored_bits"):
        CompressedDynamicCache(
            codec=codec,
            config=_sliding_config(window=5),
            skip_layers=skip_layers,
            recent_tokens=2,
        )


@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
def test_skipped_sliding_layer_is_full_precision_and_resident_accounted(automatic):
    """Skipping preserves model sliding semantics and counts resident fp16 only."""
    key, value = _token_rows(9)

    def run(chunks):
        cache = CompressedDynamicCache(
            codec=_MarkerCodec(),
            config=_sliding_config(window=5),
            skip_layers=None if automatic else {0},
            skip_sliding_layers=automatic,
        )
        offset = 0
        for length in chunks:
            cache.update(
                key[:, :, offset : offset + length],
                value[:, :, offset : offset + length],
                0,
            )
            offset += length
        return cache

    whole = run([9])
    chunked = run([3, 3, 3])
    tokenwise = run([1] * 9)
    for cache in (whole, chunked, tokenwise):
        # HF SlidingWindowLayer retains window-1 past positions; the current
        # update is concatenated transiently for attention and is not resident.
        assert torch.equal(cache.layers[0].keys, key[:, :, -4:])
        assert torch.equal(cache.layers[0].values, value[:, :, -4:])
        expected = (key[:, :, -4:].numel() + value[:, :, -4:].numel()) * 16
        assert cache.stored_bits == cache.fp16_bits == expected
        layout = cache.get_codec_runtime_provenance()["cache_layout"]
        assert layout["sliding_window_layers"] == [0]
        assert layout["effective_skip_layers"] == [0]
        assert layout["auto_skipped_sliding_layers"] == ([0] if automatic else [])


def test_skipped_sliding_channel_codec_uses_resident_not_arrival_accounting():
    """The channel-codec dispatcher must not bypass the sliding skip path."""
    key, value = _token_rows(9)
    cache = CompressedDynamicCache(
        codec=KIVICodec(group_size=2),
        config=_sliding_config(window=5),
        skip_sliding_layers=True,
    )

    for offset in range(0, 9, 3):
        cache.update(key[:, :, offset : offset + 3], value[:, :, offset : offset + 3], 0)

    resident_key = key[:, :, -4:]
    resident_value = value[:, :, -4:]
    assert torch.equal(cache.layers[0].keys, resident_key)
    assert torch.equal(cache.layers[0].values, resident_value)
    expected = (resident_key.numel() + resident_value.numel()) * 16
    assert cache.stored_bits == cache.fp16_bits == expected


def test_partial_sliding_skip_still_fails_closed():
    config = PreTrainedConfig()
    config.num_hidden_layers = 2
    config.sliding_window = 5
    config.layer_types = ["sliding_attention", "sliding_attention"]
    with pytest.raises(NotImplementedError, match=r"layers \[1\]"):
        CompressedDynamicCache(codec=_MarkerCodec(), config=config, skip_layers={0})


def test_unused_sliding_metadata_with_full_layer_type_remains_supported():
    """Qwen-style disabled sliding metadata must not reject a full layer."""
    config = _sliding_config(window=5)
    config.use_sliding_window = False
    config.layer_types = ["full_attention"]
    cache = CompressedDynamicCache(codec=_MarkerCodec(), config=config)
    key, value = _token_rows(2)

    returned_key, returned_value = cache.update(key, value, 0)

    assert torch.equal(returned_key, key + 100.0)
    assert torch.equal(returned_value, value + 200.0)


def test_value_channel_codec_is_chunk_independent():
    """A value-channel axis must not take the per-arriving-token fast path."""
    torch.manual_seed(12)
    key = torch.randn(1, 2, 8, 8)
    value = torch.randn_like(key)

    def run(chunks):
        codec = AffineKVCodec(
            bits=2,
            key_axis="token",
            value_axis="channel",
            group_size=4,
        )
        cache = CompressedDynamicCache(codec=codec)
        offset = 0
        for length in chunks:
            cache.update(
                key[:, :, offset : offset + length],
                value[:, :, offset : offset + length],
                0,
            )
            offset += length
        return cache

    prefill = run([8])
    decode = run([1] * 8)
    uneven = run([3, 2, 3])

    assert torch.equal(prefill.layers[0].keys, decode.layers[0].keys)
    assert torch.equal(prefill.layers[0].values, decode.layers[0].values)
    assert torch.equal(prefill.layers[0].keys, uneven.layers[0].keys)
    assert torch.equal(prefill.layers[0].values, uneven.layers[0].values)
    assert prefill.stored_bits == decode.stored_bits == uneven.stored_bits
    assert prefill.fp16_bits == decode.fp16_bits == uneven.fp16_bits


def test_skipped_channel_layer_does_not_require_a_finite_group_size():
    codec = AffineKVCodec(
        bits=2,
        key_axis="channel",
        value_axis="token",
        group_size=-1,
        skip_layers=[0],
    )
    cache = CompressedDynamicCache(codec=codec)
    key, value = _token_rows(3)

    returned_key, returned_value = cache.update(key, value, 0)

    assert torch.equal(returned_key, key)
    assert torch.equal(returned_value, value)
    expected_fp_bits = (key.numel() + value.numel()) * 16
    assert cache.stored_bits == cache.fp16_bits == expected_fp_bits


def test_negative_skip_layer_protects_the_actual_last_cache_layer():
    config = PreTrainedConfig()
    config.num_hidden_layers = 3
    codec = _MarkerCodec(skip_layers=[-1])
    cache = CompressedDynamicCache(codec=codec, config=config)
    key, value = _token_rows(1)

    quant_key, _ = cache.update(key, value, 1)
    skipped_key, skipped_value = cache.update(key, value, 2)

    assert cache.skip_layers == {2}
    assert torch.equal(quant_key, key + 100.0)
    assert torch.equal(skipped_key, key)
    assert torch.equal(skipped_value, value)
    fp_bits = (key.numel() + value.numel()) * 16
    quant_bits = (key.numel() + value.numel()) * codec.bits
    assert cache.fp16_bits == 2 * fp_bits
    assert cache.stored_bits == quant_bits + fp_bits


def test_negative_skip_layer_without_model_depth_fails_closed():
    with pytest.raises(ValueError, match="negative skip_layers require a model config"):
        CompressedDynamicCache(codec=_MarkerCodec(skip_layers=[-1]))


def test_explicit_empty_skip_layers_overrides_codec_defaults():
    config = PreTrainedConfig()
    config.num_hidden_layers = 1
    cache = CompressedDynamicCache(
        codec=_MarkerCodec(skip_layers=[0]),
        skip_layers=[],
        config=config,
    )
    key, value = _token_rows(1)
    returned_key, returned_value = cache.update(key, value, 0)

    assert cache.skip_layers == set()
    assert torch.equal(returned_key, key + 100.0)
    assert torch.equal(returned_value, value + 200.0)


def test_known_zero_stored_bits_is_not_replaced_by_fp16_fallback():
    key, value = _token_rows(3)
    cache = CompressedDynamicCache(codec=_ZeroBitConstantCodec())
    returned_key, returned_value = cache.update(key, value, 0)

    assert torch.count_nonzero(returned_key) == 0
    assert torch.count_nonzero(returned_value) == 0
    assert cache.fp16_bits == (key.numel() + value.numel()) * 16
    assert cache.stored_bits == 0
    assert cache.compression_ratio() == float("inf")


def test_adaptive_hq_uses_cumulative_cache_local_stats_and_current_update():
    codec = _adaptive_hq_codec(min_tokens=2)
    cache = CompressedDynamicCache(codec=codec)

    key, value = _constant_update(10.0, 0.1)
    cache.update(key, value, 0)
    first = cache.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    assert first["last_selection"]["pair"] == [2, 2]
    assert not first["last_selection"]["used_stats"]
    assert first["cumulative"]["token_count"] == 1

    cache.update(key, value, 0)
    mature = cache.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    # The second *current* update crosses the cumulative gate and is included
    # before selecting the payload that will be read by future queries.
    assert mature["last_selection"]["used_stats"]
    assert mature["last_selection"]["pair"] == [3, 1]
    assert mature["cumulative"]["token_count"] == 2
    assert mature["cumulative"]["key_power"] == pytest.approx(100.0)
    assert mature["cumulative"]["value_power"] == pytest.approx(0.01)


def test_adaptive_hq_zero_power_is_recorded_without_surrogate_floor():
    cache = CompressedDynamicCache(codec=_adaptive_hq_codec(min_tokens=1))
    zeros = torch.zeros(1, 1, 1, 8)

    cache.update(zeros, zeros, 0)
    layer = cache.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]

    assert layer["cumulative"]["key_power"] == 0.0
    assert layer["cumulative"]["value_power"] == 0.0
    assert layer["last_selection"]["pair"] == [2, 2]


def test_adaptive_hq_state_is_per_cache_and_per_layer_without_leakage():
    codec = _adaptive_hq_codec(min_tokens=1)
    cache_a = CompressedDynamicCache(codec=codec)
    cache_b = CompressedDynamicCache(codec=codec)

    high_key = _constant_update(10.0, 0.1)
    high_value = _constant_update(0.1, 10.0)
    cache_a.update(high_key[0], high_key[1], 0)
    cache_a.update(high_value[0], high_value[1], 1)
    cache_b.update(high_value[0], high_value[1], 0)

    prov_a = cache_a.get_adaptive_hq_provenance()["runtime"]
    prov_b = cache_b.get_adaptive_hq_provenance()["runtime"]
    assert prov_a["layers"]["0"]["last_selection"]["pair"] == [3, 1]
    # The frozen safe policy has only neutral and fixed-HQ key-heavy pairs;
    # a value-heavy update therefore stays neutral rather than inventing a
    # third, unvalidated candidate.
    assert prov_a["layers"]["1"]["last_selection"]["pair"] == [2, 2]
    assert prov_b["layers"]["0"]["last_selection"]["pair"] == [2, 2]
    assert prov_a["layers"]["0"]["cumulative"]["token_count"] == 1
    assert prov_b["layers"]["0"]["cumulative"]["token_count"] == 1
    assert "adaptive_hq" not in codec.__dict__

    # Returned provenance is detached; callers cannot mutate live cache state.
    prov_a["layers"]["0"]["cumulative"]["token_count"] = 999
    prov_a["layers"]["0"]["last_selection"]["candidate_objectives"][0]["weighted_objective"] = -1.0
    fresh = cache_a.get_adaptive_hq_provenance()["runtime"]
    assert fresh["layers"]["0"]["cumulative"]["token_count"] == 1
    assert (
        fresh["layers"]["0"]["last_selection"]["candidate_objectives"][0]["weighted_objective"]
        >= 0.0
    )


def test_adaptive_hq_payload_metadata_cannot_mutate_cache_owned_policy():
    codec = _adaptive_hq_codec(min_tokens=1)
    owner = {}
    key, value = _constant_update(4.0, 1.0)

    encoded = codec.encode(key, value, ctx={"layer_idx": 0, "cache_state": owner})
    policy_table = owner["adaptive_hq"]["policy"]["datafree_distortion_table"]
    expected = policy_table[0]["key_relative_distortion"]
    encoded.meta["adaptive_bit_split"]["datafree_distortion_table"][0][
        "key_relative_distortion"
    ] = -1.0

    assert policy_table[0]["key_relative_distortion"] == expected


def test_adaptive_hq_chunked_prefix_has_same_next_token_causal_state():
    codec = _adaptive_hq_codec(min_tokens=1)
    one_chunk = CompressedDynamicCache(codec=codec)
    token_chunks = CompressedDynamicCache(codec=codec)
    prefix = _constant_update(10.0, 0.1, seq=4)
    next_token = _constant_update(0.1, 100.0)

    one_chunk.update(prefix[0], prefix[1], 0)
    for index in range(4):
        token_chunks.update(
            prefix[0][:, :, index : index + 1], prefix[1][:, :, index : index + 1], 0
        )
    one_chunk.update(next_token[0], next_token[1], 0)
    token_chunks.update(next_token[0], next_token[1], 0)

    one = one_chunk.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    many = token_chunks.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    assert one["last_selection"]["pair"] == many["last_selection"]["pair"] == [2, 2]
    assert one["cumulative"]["token_count"] == many["cumulative"]["token_count"] == 5
    assert one["cumulative"]["key_power"] == pytest.approx(many["cumulative"]["key_power"])
    assert one["cumulative"]["value_power"] == pytest.approx(many["cumulative"]["value_power"])
    assert one["last_selection"]["carrier_amortization_vector_count"] == 1
    assert many["last_selection"]["carrier_amortization_vector_count"] == 1
    assert one["last_selection"]["value_relative_distortion_effective"] == pytest.approx(
        many["last_selection"]["value_relative_distortion_effective"]
    )
    assert one["last_selection"]["weighted_objective"] == pytest.approx(
        many["last_selection"]["weighted_objective"]
    )
    # Selector information is carried once per encode update, so deliberately
    # different chunkings have different information totals.  The carrier is
    # inside an existing fp16 scale field, hence neither total adds rate.
    assert one["selector_bits_total"] == 2
    assert many["selector_bits_total"] == 5
    assert one["selector_incremental_stored_bits_total"] == 0
    assert many["selector_incremental_stored_bits_total"] == 0


def test_adaptive_hq_selector_accounting_histogram_json_and_reset():
    codec = _adaptive_hq_codec(min_tokens=2)
    cache = CompressedDynamicCache(codec=codec)
    key, value = _constant_update(10.0, 0.1)
    cache.update(key, value, 0)
    cache.update(key, value, 0)
    provenance = cache.get_adaptive_hq_provenance()
    runtime = provenance["runtime"]

    # Per update: 8 coordinates * (K+V=4 code bits) + two fp16 scales.
    # The frozen two-pair policy needs one selector information bit in the
    # first value scale, so it adds no data-plane bits.
    assert cache.stored_bits == 2 * (8 * 4 + 2 * 16)
    assert runtime["aggregate"]["selector_bits_total"] == 2
    assert runtime["aggregate"]["selector_incremental_stored_bits_total"] == 0
    assert runtime["aggregate"]["encoded_updates"] == 2
    assert runtime["aggregate"]["split_histogram"] == {
        "2:2": {"pair": [2, 2], "updates": 1, "tokens": 1, "vectors": 1},
        "3:1": {"pair": [3, 1], "updates": 1, "tokens": 1, "vectors": 1},
    }
    assert runtime["policy_version"].startswith("adaptive_hq_cache_causal")
    assert json.loads(json.dumps(provenance)) == provenance
    layer = runtime["layers"]["0"]
    assert layer["policy"]["current_update_included"] is True
    assert layer["policy"]["maturity_gate"] == "cumulative_batch_tokens"
    assert layer["last_selection"]["key_rotation_mode"] in {
        "dense",
        "hybrid",
        "hadamard",
        "hadamard_gemm",
    }
    last = layer["last_selection"]
    assert len(last["pair"]) == 2
    assert all(len(run["pair"]) == 2 for run in layer["selection_runs"])
    assert all(run["pair"] in ([2, 2], [3, 1]) for run in layer["selection_runs"])
    assert last["update_tokens"] == 1
    assert last["carrier_amortization_vector_count"] == 1
    assert last["cumulative_vector_count"] == layer["cumulative"]["vector_count"]
    assert len(last["candidate_objectives"]) == 2
    selected = last["candidate_objectives"][last["selected_pair_index"]]
    assert selected["pair"] == last["pair"]
    assert selected["carrier_amortization_vector_count"] == 1
    assert selected["value_relative_distortion_effective"] == pytest.approx(
        last["value_relative_distortion_effective"]
    )
    assert selected["predicted_key_error"] == pytest.approx(last["predicted_key_error"])
    assert selected["predicted_value_error"] == pytest.approx(last["predicted_value_error"])
    assert selected["weighted_objective"] == pytest.approx(last["weighted_objective"])
    assert (
        sum(run["vectors"] for run in layer["selection_runs"])
        == layer["cumulative"]["vector_count"]
    )
    assert (
        layer["selection_runs"][-1]["end_cumulative_vectors"]
        == layer["cumulative"]["vector_count"]
    )

    cache.reset()
    reset = cache.get_adaptive_hq_provenance()["runtime"]
    assert cache.stored_bits == cache.fp16_bits == 0
    assert reset["layers"] == {}
    assert reset["aggregate"]["encoded_updates"] == 0
    cache.update(key, value, 0)
    restarted = cache.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    assert restarted["last_selection"]["pair"] == [2, 2]
    assert not restarted["last_selection"]["used_stats"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adaptive_hq_cache_causal_state_cuda():
    codec = _adaptive_hq_codec(min_tokens=2)
    cache = CompressedDynamicCache(codec=codec)
    key, value = _constant_update(10.0, 0.1, device="cuda")
    cache.update(key, value, 0)
    cache.update(key, value, 0)
    layer = cache.get_adaptive_hq_provenance()["runtime"]["layers"]["0"]
    assert layer["last_selection"]["pair"] == [3, 1]
    assert layer["cumulative"]["token_count"] == 2
    assert cache.stored_bits == 2 * (8 * 4 + 2 * 16)


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"sink_tokens": -1}, "sink_tokens"),
        ({"recent_tokens": -1}, "recent_tokens"),
    ],
)
def test_negative_protected_window_fails_closed(kwargs, field):
    with pytest.raises(ValueError, match=field):
        CompressedDynamicCache(codec=_MarkerCodec(), **kwargs)


def _codec_cases():
    return (
        (
            "turboquant-single",
            TurboQuantCodec(bits=3, value_bits=2, dist="gaussian", rotation_mode="dense"),
        ),
        (
            "turboquant-outlier",
            TurboQuantCodec(
                bits=3,
                value_bits=2,
                n_outlier=4,
                outlier_bits=4,
                dist="gaussian",
                rotation_mode="dense",
            ),
        ),
        (
            "turboquant-qjl",
            TurboQuantCodec(
                bits=2,
                value_bits=2,
                use_qjl=True,
                qjl_roles="both",
                dist="gaussian",
                rotation_mode="dense",
            ),
        ),
        (
            "hayate-single",
            HayateQuantCodec(
                bits=2,
                value_bits=2,
                block_size=2,
                dist="gaussian",
                rotation_mode="dense",
                encoder="exact",
            ),
        ),
        (
            "hayate-outlier",
            HayateQuantCodec(
                bits=2,
                value_bits=2,
                block_size=1,
                n_outlier=4,
                outlier_bits=3,
                dist="gaussian",
                rotation_mode="dense",
                encoder="exact",
            ),
        ),
        (
            "hayate-qjl",
            HayateQuantCodec(
                bits=2,
                value_bits=2,
                block_size=2,
                use_qjl=True,
                dist="gaussian",
                rotation_mode="dense",
                encoder="exact",
            ),
        ),
        (
            "halo-single",
            HALOCodec(bits=3, value_bits=2, rotation_mode="dense"),
        ),
        (
            "halo-outlier",
            HALOCodec(
                bits=3,
                value_bits=2,
                n_outlier=4,
                outlier_bits=4,
                rotation_mode="dense",
            ),
        ),
    )


@pytest.mark.parametrize("_name,codec", _codec_cases(), ids=[case[0] for case in _codec_cases()])
def test_rotated_codec_cache_lifecycle_matches_explicit_roundtrips(_name, codec):
    """Prefill/decode/sink/recent accounting must use each payload exactly once."""
    torch.manual_seed(44)
    key = torch.randn(1, 2, 8, 16, dtype=torch.float16)
    value = torch.randn_like(key)
    cache = CompressedDynamicCache(codec=codec, sink_tokens=1, recent_tokens=2)
    cache.update(key[:, :, :5], value[:, :, :5], 0)
    for index in range(5, 8):
        cache.update(key[:, :, index : index + 1], value[:, :, index : index + 1], 0)

    expected_key = [key[:, :, :1]]
    expected_value = [value[:, :, :1]]
    fp_bits = (key[:, :, :1].numel() + value[:, :, :1].numel()) * 16
    expected_bits = fp_bits
    for start, stop in ((1, 3), (3, 4), (4, 5), (5, 6)):
        encoded = codec.encode(key[:, :, start:stop], value[:, :, start:stop])
        decoded_key, decoded_value = codec.decode(encoded)
        round_key, round_value, round_bits = codec.fake_quant_roundtrip(
            key[:, :, start:stop], value[:, :, start:stop]
        )
        assert round_bits == encoded.numel_bits()
        assert torch.allclose(round_key, decoded_key, atol=2e-3, rtol=1e-3)
        assert torch.allclose(round_value, decoded_value, atol=2e-3, rtol=1e-3)
        expected_key.append(round_key)
        expected_value.append(round_value)
        expected_bits += int(round_bits)
    expected_key.append(key[:, :, 6:])
    expected_value.append(value[:, :, 6:])
    expected_bits += (key[:, :, 6:].numel() + value[:, :, 6:].numel()) * 16

    assert cache.stored_bits == expected_bits
    assert cache.fp16_bits == (key.numel() + value.numel()) * 16
    assert torch.equal(cache.layers[0].keys, torch.cat(expected_key, dim=2))
    assert torch.equal(cache.layers[0].values, torch.cat(expected_value, dim=2))


@pytest.mark.parametrize("role_field", ["key_n_outlier", "value_n_outlier"])
def test_hayate_native_rejects_role_specific_outlier_split(monkeypatch, role_field):
    """The experimental native backend cannot silently drop a role split."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    codec = {
        "name": "hayatequant",
        "bits": 4,
        "value_bits": 4,
        "block_size": 2,
        role_field: 4,
    }
    plan = plan_from_kv_config({"codec": codec, "residual_window": 0})

    assert not plan.native_supported
    assert plan.needs_custom_backend
    assert plan.kv_cache_dtype == "auto"
    assert any("role-specific n_outlier" in note for note in plan.notes)


@pytest.mark.parametrize("residual_window", [1, 128])
def test_vllm_native_codec_rejects_full_precision_residual_window(monkeypatch, residual_window):
    """A native uniform dtype cannot silently discard the HF recent window."""
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    plan = plan_from_kv_config(
        {
            "codec": {"name": "turboquant", "bits": 4, "value_bits": 4},
            "residual_window": residual_window,
        }
    )

    assert plan.needs_custom_backend
    assert not plan.native_supported
    assert plan.kv_cache_dtype == "auto"
    assert any(f"residual_window={residual_window}" in note for note in plan.notes)


def test_vllm_missing_residual_window_uses_safe_historical_default(monkeypatch):
    from vllm_plugins.kvcomp.mapping import plan_from_kv_config

    monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    plan = plan_from_kv_config({"codec": {"name": "turboquant", "bits": 4, "value_bits": 4}})

    assert plan.needs_custom_backend
    assert not plan.native_supported
    assert any("Missing fields default to 128" in note for note in plan.notes)


@pytest.mark.parametrize("next_codec", ["turboquant", "scalar_fallback"])
def test_hayate_native_config_does_not_leak_to_next_engine(tmp_path, monkeypatch, next_codec):
    """A process-global backend override must not reuse stale Hayate settings."""
    from vllm_plugins.kvcomp import vllm_plugin

    def write_model(name, codec):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "kv_compression_config": {
                        "codec": codec,
                        "policy": {"name": "static"},
                        "residual_window": 0,
                    }
                }
            )
        )
        return str(model_dir)

    # Avoid importing/registering vLLM in this CPU-only lifecycle test while
    # reproducing the exact process-global environment side effect.
    def fake_install(plan, _logger):
        os.environ[vllm_plugin._HAYATE_NATIVE_CONFIG_ENV] = json.dumps(
            plan.hayate_config, sort_keys=True
        )
        return True

    monkeypatch.setattr(vllm_plugin, "_install_hayate_native_backend", fake_install)
    hayate_model = write_model(
        "hayate",
        {"name": "hayatequant", "bits": 3, "value_bits": 3, "block_size": 2},
    )
    monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "native")
    hayate_args = SimpleNamespace(model=hayate_model, kv_cache_dtype="auto")
    vllm_plugin.apply_to_engine_args(hayate_args)
    assert hayate_args.kv_cache_dtype == "turboquant_3bit_nc"
    assert vllm_plugin._HAYATE_NATIVE_CONFIG_ENV in os.environ

    if next_codec == "turboquant":
        next_model = write_model(
            "turboquant",
            {"name": "turboquant", "bits": 3, "value_bits": 3},
        )
        monkeypatch.setenv("KVCOMP_VLLM_TURBOQUANT_MODE", "native")
    else:
        next_model = hayate_model
        monkeypatch.setenv("KVCOMP_VLLM_HAYATEQUANT_MODE", "scalar_fallback")

    next_args = SimpleNamespace(model=next_model, kv_cache_dtype="auto")
    vllm_plugin.apply_to_engine_args(next_args)
    assert next_args.kv_cache_dtype == "turboquant_3bit_nc"
    assert vllm_plugin._HAYATE_NATIVE_CONFIG_ENV not in os.environ


def test_manual_hayate_native_config_survives_model_without_serialized_recipe(
    tmp_path, monkeypatch
):
    """The standalone native harness configures HQ without a model KV block."""
    from vllm_plugins.kvcomp import vllm_plugin

    model_dir = tmp_path / "plain-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    payload = '{"bits": 3, "block_size": 2}'
    monkeypatch.setenv(vllm_plugin._HAYATE_NATIVE_CONFIG_ENV, payload)
    engine_args = SimpleNamespace(
        model=str(model_dir),
        kv_cache_dtype="turboquant_3bit_nc",
    )

    vllm_plugin.apply_to_engine_args(engine_args)

    assert engine_args.kv_cache_dtype == "turboquant_3bit_nc"
    assert os.environ[vllm_plugin._HAYATE_NATIVE_CONFIG_ENV] == payload
