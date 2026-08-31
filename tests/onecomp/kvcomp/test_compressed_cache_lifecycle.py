"""Focused lifecycle/accounting regressions for compressed KV caches."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedConfig

from onecomp.kvcomp._base import CompressedKV, KVCodec
from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.codec.affine import AffineKVCodec, KIVICodec
from onecomp.kvcomp.codec.hayate.recipes import build_hayatequant
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
        ("turboquant", TurboQuantCodec(bits=3)),
        ("hayatequant-r2", build_hayatequant(2)),
        ("hayatequant-r3", build_hayatequant(3)),
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
