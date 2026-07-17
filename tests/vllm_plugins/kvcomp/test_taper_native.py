"""CPU contracts for the vLLM TapeKV reference data plane."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from onecomp.kvcomp.cache.hayate_taper_cache import HayateTaperKVCache
from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.recipe.taperkv import TaperKVPolicy
from vllm_plugins.kvcomp import taper_native
from vllm_plugins.kvcomp.taper_config import TaperVLLMRecipe
from vllm_plugins.kvcomp.taper_native import TaperKVNativeImpl


def _set_recipe(monkeypatch, **overrides):
    payload = {
        "policy_name": "taperkv",
        "key_bits": 16,
        "value_bits": 16,
        "b_min": 16,
        "b_min_value": 16,
        "requant_stride": 16,
    }
    payload.update(overrides)
    monkeypatch.setenv(taper_native.TAPER_CONFIG_ENV, json.dumps(payload))


def test_layer_index_parses_vllm_layer_name():
    layer = SimpleNamespace(layer_name="model.layers.17.self_attn")
    assert TaperKVNativeImpl._layer_index(layer) == 17


def test_skip_layers_fail_closed_when_layer_index_is_unknown(monkeypatch):
    _set_recipe(monkeypatch, skip_layers=[1])
    impl = object.__new__(TaperKVNativeImpl)
    with pytest.raises(RuntimeError, match="cannot honor skip_layers"):
        impl.do_kv_cache_update(
            SimpleNamespace(),
            torch.empty(1, 1, 1),
            torch.empty(1, 1, 1),
            torch.empty(0),
            torch.tensor([0]),
        )


def test_hayate_is_applied_only_to_arriving_kv_before_upstream_scatter(monkeypatch):
    _set_recipe(monkeypatch, hayate_config={"name": "hayatequant", "encoder": "exact"})
    captured = []

    def fake_upstream(_self, _layer, key, value, _cache, _slots):
        captured.append((key.clone(), value.clone()))

    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.do_kv_cache_update",
        fake_upstream,
    )
    impl = object.__new__(TaperKVNativeImpl)
    impl._hayate_roundtrip = lambda key, value, _recipe, _layer_idx: (key + 1, value + 2)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn")
    slots = torch.tensor([0])
    for value in (0.0, 5.0):
        key = torch.full((1, 1, 2), value)
        impl.do_kv_cache_update(layer, key, key, torch.empty(0), slots)
    assert torch.equal(captured[0][0], torch.ones(1, 1, 2))
    assert torch.equal(captured[1][0], torch.full((1, 1, 2), 6.0))
    assert torch.equal(impl._taper_pending_slots, slots)


def test_batched_decode_uses_max_query_len_not_total_slot_count(monkeypatch):
    _set_recipe(monkeypatch, requant_stride=16)
    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl.forward",
        lambda *_args, **_kwargs: "upstream",
    )
    impl = object.__new__(TaperKVNativeImpl)
    impl._taper_pending_slots = torch.tensor([1, 7])
    calls = []
    impl._quantize_pending_slots = lambda *_args: calls.append("new")
    impl._requantize_cache = lambda *_args: calls.append("all")
    metadata = SimpleNamespace(use_cascade=False, max_query_len=1)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn")
    result = impl.forward(layer, None, None, None, torch.empty(0), metadata, None)
    assert result == "upstream"
    assert calls == ["new"]
    assert impl._taper_pending_slots is None


def test_hayate_taper_prefill_matches_hf_reference_cache(monkeypatch):
    torch.manual_seed(7)
    policy = TaperKVPolicy(
        key_bits=4,
        value_bits=3,
        b_min=2,
        b_min_value=2,
        sink_tokens=0,
        recent_tokens=0,
        requant_stride=2,
    )
    codec = HayateQuantCodec(
        bits=2,
        value_bits=2,
        block_size=2,
        encoder="exact",
        seed=11,
    )
    keys = torch.randn(1, 2, 4, 4)
    values = torch.randn_like(keys)
    hf_cache = HayateTaperKVCache(policy, codec)
    expected_k, expected_v = hf_cache.update(keys, values, 0)

    recipe = TaperVLLMRecipe.from_policy(
        policy.get_kv_config(), hayate_config=codec.get_kv_config()
    )
    monkeypatch.setenv(taper_native.TAPER_CONFIG_ENV, json.dumps(recipe.to_mapping()))
    impl = object.__new__(TaperKVNativeImpl)
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn")
    incoming_k, incoming_v = impl._hayate_roundtrip(
        keys.squeeze(0).transpose(0, 1),
        values.squeeze(0).transpose(0, 1),
        recipe,
        0,
    )
    paged = torch.empty(1, 2, 4, 2, 4)
    paged[0, 0] = incoming_k
    paged[0, 1] = incoming_v
    metadata = SimpleNamespace(
        use_cascade=False,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([4]),
    )
    impl._requantize_cache(paged, metadata, layer)
    actual_k = paged[0, 0].transpose(0, 1).unsqueeze(0)
    actual_v = paged[0, 1].transpose(0, 1).unsqueeze(0)
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)
