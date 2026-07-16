"""Regressions found by the 2026-07-16 core cache/metric audit."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedConfig

import onecomp.kvcomp.eval.reasoning as reasoning_eval
from onecomp.kvcomp._base import CompressedKV, KVCodec
from onecomp.kvcomp.cache._graded_base import GradedQuantCache
from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.cache.streaming_evict import StreamingLLMEvictCache
from onecomp.kvcomp.cache.thinkv_cache import ThinKVCache
from onecomp.kvcomp.codec.hayate.codec import ADAPTIVE_HQ_POLICY_VERSION
from onecomp.kvcomp.eval._teacher_forcing import causal_teacher_forced_logits
from onecomp.kvcomp.eval.compression_accounting import (
    aggregate_compression_sufficient_statistics,
    aggregate_mem_fraction,
    cache_compression_sufficient_statistics,
    cache_mem_fraction,
)
from onecomp.kvcomp.eval.longcodeqa import (
    LONGCODEQA_BRACKET_COUNTS,
    LONGCODEQA_DATASET_REVISION,
    LONGCODEQA_DATASET_SHA256,
    LongCodeQAResult,
)
from onecomp.kvcomp.eval.metrics import KVMemoryReport, effective_bits, kv_cache_bytes
from onecomp.kvcomp.eval.ppl import perplexity_over_text
from onecomp.kvcomp.eval.reasoning import _eval_dataset
from onecomp.kvcomp.eval.zeroshot import zeroshot_accuracy


class _TwoBitGradedCache(GradedQuantCache):
    def token_bits(self, total, device, layer_idx):
        del layer_idx
        bits = torch.full((total,), 2, dtype=torch.int64, device=device)
        return bits, bits


@dataclass
class _BadRoundtripCodec(KVCodec):
    key_axis: str = "token"
    value_axis: str = "token"

    def encode(self, key, value, ctx=None):
        del ctx
        return CompressedKV(
            key_q=key[..., :-1],
            value_q=value[..., :-1],
            meta={"stored_bits": 1},
        )

    def decode(self, compressed, ctx=None):
        del ctx
        return compressed.key_q, compressed.value_q


def _sliding_config() -> PreTrainedConfig:
    config = PreTrainedConfig()
    config.num_hidden_layers = 1
    config.sliding_window = 8
    config.layer_types = ["sliding_attention"]
    return config


def test_graded_bit_accounting_includes_batch_dimension():
    cache = _TwoBitGradedCache(requant_stride=1)
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn_like(key)
    cache.update(key, value, 0)

    batch_heads = 2 * 3
    payload = 2 * (4 * 2) * (batch_heads * 5)
    metadata = 2 * batch_heads * (4 * 16 + 4 * 2)
    assert cache._layer_bits[0] == payload + metadata
    assert cache._layer_fp16[0] == (key.numel() + value.numel()) * 16


def test_graded_and_thinkv_reject_model_sliding_layers_before_scoring():
    with pytest.raises(NotImplementedError, match="sliding-window"):
        _TwoBitGradedCache(config=_sliding_config())
    with pytest.raises(NotImplementedError, match="sliding-window"):
        ThinKVCache(config=_sliding_config())


def test_thinkv_batch_never_silently_falls_back_to_fullkv():
    cache = ThinKVCache(sink_tokens=0)
    key = torch.randn(2, 1, 1, 8)
    with pytest.raises(NotImplementedError, match="batch size >1"):
        cache.update(key, key, 0)
    assert cache.get_seq_length() == 0


def test_cache_reset_clears_algorithm_and_accounting_state():
    graded = _TwoBitGradedCache(requant_stride=1)
    key = torch.randn(1, 1, 3, 4)
    graded.update(key, key, 0)
    assert graded._layer_bits
    graded.reset()
    assert graded._layer_bits == {}
    assert graded._layer_fp16 == {}
    assert graded._since == 0

    streaming = StreamingLLMEvictCache(sink=1, recent=1)
    streaming.update(key, key, 0)
    streaming.evict_now()
    assert streaming.total == 3
    streaming.reset()
    assert (streaming.kept, streaming.total) == (0, 0)


def test_compressed_cache_rejects_malformed_codec_roundtrip_and_skip_index():
    cache = CompressedDynamicCache(codec=_BadRoundtripCodec(), sink_tokens=0, recent_tokens=0)
    key = torch.randn(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match="decoded key shape"):
        cache.update(key, key, 0)

    config = PreTrainedConfig()
    config.num_hidden_layers = 1
    config.layer_types = ["full_attention"]
    with pytest.raises(ValueError, match="out of range"):
        CompressedDynamicCache(codec=_BadRoundtripCodec(), config=config, skip_layers=[1])


@pytest.mark.parametrize("stored_bits", [True, -1, 1.5, "8"])
def test_compressed_payload_bit_contract_is_fail_closed(stored_bits):
    payload = CompressedKV(meta={"stored_bits": stored_bits})
    with pytest.raises(ValueError, match="stored_bits"):
        payload.numel_bits()


def test_zero_bit_memory_report_is_infinite_compression_not_fullkv():
    report = KVMemoryReport(1, 1, 1, 4, fp16_bytes=16, stored_bytes=0)
    assert math.isinf(report.compression_ratio)
    assert report.kv_fraction == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: kv_cache_bytes(-1, 1, 1, 1),
        lambda: kv_cache_bytes(1, 1, 1, 1, bits=float("nan")),
        lambda: effective_bits(1, 0),
    ],
)
def test_memory_helpers_reject_impossible_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_partial_exact_counters_and_legacy_ratio_are_not_mislabeled_fullkv():
    partial = SimpleNamespace(fp16_bits=10)
    with pytest.raises(RuntimeError, match="only one"):
        cache_compression_sufficient_statistics(partial)

    class Legacy:
        def mem_fraction(self):
            return 0.25

    record = cache_compression_sufficient_statistics(Legacy())
    assert record["compression_aggregation"] == "legacy_cache_ratio_only"
    assert record["compression_ratio"] == 4.0
    aggregate = aggregate_compression_sufficient_statistics([record, record])
    assert aggregate["compression_aggregation"].startswith("legacy_arithmetic_mean")
    assert aggregate["compression_ratio_arithmetic_mean"] == 4.0


def test_exact_memory_fraction_is_ratio_of_bit_sums_not_mean_of_ratios():
    records = [
        {
            "compression_aggregation": "exact_cache_bit_counters",
            "compression_fp16_bits": 100,
            "compression_stored_bits": 50,
            "adaptive_hq_runtime": None,
        },
        {
            "compression_aggregation": "exact_cache_bit_counters",
            "compression_fp16_bits": 900,
            "compression_stored_bits": 90,
            "adaptive_hq_runtime": None,
        },
    ]
    statistics = aggregate_compression_sufficient_statistics(records)
    fraction, mode = aggregate_mem_fraction(statistics, [0.5, 0.1])
    assert fraction == pytest.approx(140 / 1000)
    assert fraction != pytest.approx((0.5 + 0.1) / 2)
    assert mode == "sum_stored_bits_over_sum_fp16_bits"


def _valid_adaptive_cache_runtime() -> dict:
    return {
        "kind": "adaptive_hq",
        "schema_version": 1,
        "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
        "aggregate": {
            "encoded_updates": 2,
            "encoded_tokens": 3,
            "selector_bits_total": 2,
            "selector_incremental_stored_bits_total": 0,
            "selector_carrier_version": "value_scale_fp16_fraction_lsb_v1",
            "split_histogram": {
                "5:3": {
                    "pair": [5, 3],
                    "updates": 2,
                    "tokens": 3,
                    "vectors": 7,
                }
            },
        },
    }


def test_adaptive_cache_runtime_accepts_only_exact_current_ledger():
    runtime = _valid_adaptive_cache_runtime()
    cache = SimpleNamespace(
        fp16_bits=1024,
        stored_bits=512,
        get_adaptive_hq_provenance=lambda: {"runtime": runtime},
    )
    record = cache_compression_sufficient_statistics(cache)
    assert record["adaptive_hq_runtime"] == {
        "schema_version": 1,
        "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
        **runtime["aggregate"],
    }
    combined = aggregate_compression_sufficient_statistics([record, record])
    assert combined["adaptive_hq_runtime"]["encoded_updates"] == 4
    assert combined["adaptive_hq_runtime"]["selector_bits_total"] == 4
    assert combined["adaptive_hq_runtime"]["split_histogram"]["5:3"] == {
        "pair": [5, 3],
        "updates": 4,
        "tokens": 6,
        "vectors": 14,
    }


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("policy_version",), "fabricated-policy"),
        (("aggregate", "selector_carrier_version"), "fabricated-carrier"),
        (("aggregate", "selector_bits_total"), 1),
        (("aggregate", "encoded_updates"), True),
        (("aggregate", "encoded_tokens"), True),
        (("aggregate", "split_histogram", "5:3", "updates"), 0),
        (("aggregate", "split_histogram", "5:3", "tokens"), 0),
        (("aggregate", "split_histogram", "5:3", "vectors"), 0),
        (("aggregate", "split_histogram", "5:3", "updates"), 1),
        (("aggregate", "split_histogram", "5:3", "tokens"), 2),
    ],
)
def test_adaptive_cache_runtime_rejects_fabricated_or_inconsistent_ledger(path, bad_value):
    runtime = copy.deepcopy(_valid_adaptive_cache_runtime())
    target = runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    cache = SimpleNamespace(
        fp16_bits=1024,
        stored_bits=512,
        get_adaptive_hq_provenance=lambda: {"runtime": runtime},
    )
    with pytest.raises(RuntimeError, match="AdaptiveHQ"):
        cache_compression_sufficient_statistics(cache)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, float("inf")])
def test_display_memory_fraction_rejects_nonfinite_or_negative_values(bad):
    cache = SimpleNamespace(mem_fraction=lambda: bad)
    with pytest.raises(RuntimeError, match="finite non-negative"):
        cache_mem_fraction(cache)


def test_longcodeqa_complete_flag_requires_pinned_release_provenance():
    kwargs = dict(
        bracket="32K",
        n=LONGCODEQA_BRACKET_COUNTS["32K"],
        correct=1,
        accuracy=1 / LONGCODEQA_BRACKET_COUNTS["32K"],
        accuracy_stderr=0.0,
        accuracy_ci95=(0.0, 1.0),
        input_tokens=[1],
        dataset_path="32K.json",
        dataset_sha256=LONGCODEQA_DATASET_SHA256["32K"],
        dataset_revision=LONGCODEQA_DATASET_REVISION,
        max_context_tokens=32768,
        max_new_tokens=16,
    )
    assert LongCodeQAResult(**kwargs).as_dict()["released_data_complete_bracket"] is True
    kwargs["dataset_sha256"] = "0" * 64
    assert LongCodeQAResult(**kwargs).as_dict()["released_data_complete_bracket"] is False


def test_selection_and_teacher_forcing_bool_inputs_fail_before_work():
    with pytest.raises(ValueError, match="max_length"):
        perplexity_over_text(None, None, "x", max_length=True)
    with pytest.raises(ValueError, match="Unknown zero-shot task"):
        zeroshot_accuracy(None, None, task="not-a-task")
    with pytest.raises(ValueError, match="selection is empty"):
        _eval_dataset(
            None,
            None,
            [],
            str,
            "empty",
            None,
            1,
            1,
            False,
            0.6,
            0.95,
            None,
            False,
        )

    class BadModel:
        def __call__(self, input_ids, **kwargs):
            del kwargs
            return SimpleNamespace(logits=torch.zeros(1, 1, 4))

    with pytest.raises(RuntimeError, match="teacher-forced model"):
        causal_teacher_forced_logits(
            BadModel(), torch.tensor([[1, 2, 3]]), first_target_index=1, cache=None
        )


def test_reasoning_zero_bit_payload_reports_infinite_compression(monkeypatch):
    class Model:
        device = torch.device("cpu")

    class Tokenizer:
        def decode(self, _tokens, **_kwargs):
            return "1"

    monkeypatch.setattr(
        reasoning_eval, "_build_prompt", lambda *_args, **_kwargs: torch.tensor([[1]])
    )
    monkeypatch.setattr(
        reasoning_eval,
        "_generate",
        lambda *_args, **_kwargs: (
            torch.tensor([2]),
            float("inf"),
            True,
            {
                "compression_fp16_bits": 100,
                "compression_stored_bits": 0,
                "compression_aggregation": "exact_cache_bit_counters",
                "adaptive_hq_runtime": None,
            },
        ),
    )
    result = reasoning_eval._eval_dataset(
        Model(),
        Tokenizer(),
        [("question", "1")],
        str,
        "zero-bit",
        object,
        1,
        1,
        False,
        0.6,
        0.95,
        None,
        False,
    )
    assert math.isinf(result.compression_ratio)
