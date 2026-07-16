"""Fail-closed protocol tests for the synthetic long-context evaluators."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

import onecomp.kvcomp.eval.longcontext as longcontext


@pytest.mark.parametrize("evaluator", ["needle", "multikey"])
def test_synthetic_evaluators_release_cache_before_every_next_factory(
    evaluator, monkeypatch
):
    class Cache:
        live = 0
        peak_live = 0

        def __init__(self):
            type(self).live += 1
            type(self).peak_live = max(type(self).peak_live, type(self).live)

        def __del__(self):
            type(self).live -= 1

    references = []

    def cache_factory():
        gc.collect()
        assert not references or references[-1]() is None
        assert Cache.live == 0
        cache = Cache()
        references.append(weakref.ref(cache))
        return cache

    class Tokenizer:
        eos_token_id = 2

        def decode(self, *_args, **_kwargs):
            return "1234"

    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=16, text_config=None)
    )
    monkeypatch.setattr(longcontext, "_encode", lambda *_args: torch.tensor([[1, 2]]))
    monkeypatch.setattr(
        longcontext, "greedy_answer", lambda *_args, **_kwargs: torch.tensor([[3]])
    )
    if evaluator == "needle":
        monkeypatch.setattr(
            longcontext,
            "make_needle_prompt",
            lambda *_args, **_kwargs: ([{"role": "user", "content": "x"}], "1234"),
        )
        result = longcontext.needle_in_haystack(
            model,
            Tokenizer(),
            cache_factory=cache_factory,
            context_lens=[4],
            depths=[0.5],
            seeds=3,
            max_new_tokens=1,
        )
    else:
        monkeypatch.setattr(
            longcontext,
            "make_multikey_prompt",
            lambda *_args, **_kwargs: ([{"role": "user", "content": "x"}], "1234"),
        )
        result = longcontext.ruler_multikey(
            model,
            Tokenizer(),
            cache_factory=cache_factory,
            context_lens=[4],
            seeds=3,
            max_new_tokens=1,
        )
    gc.collect()
    assert len(references) == 3
    assert all(reference() is None for reference in references)
    assert Cache.live == 0
    assert Cache.peak_live == 1
    assert (
        result.as_dict()["example_cache_lifetime"]
        == longcontext.PER_EXAMPLE_CACHE_LIFETIME
    )


def test_chat_template_failure_never_falls_back_to_raw_tokenization():
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("broken template")

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("raw-tokenization fallback is forbidden")

    with pytest.raises(RuntimeError, match="chat-template rendering failed"):
        longcontext._encode(Tokenizer(), [{"role": "user", "content": "x"}])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", True),
        ("The answer is 1234.", True),
        ("1234; I repeat 1234.", True),
        ("1234, or perhaps 5678", False),
        ("91234", False),
        ("no numeric answer", False),
    ],
)
def test_needle_scoring_rejects_ambiguous_or_partial_numbers(text, expected):
    assert longcontext._number_hit("1234", text) is expected


def test_needle_filler_counts_content_without_sentence_level_special_tokens():
    class Tokenizer:
        def __init__(self):
            self.add_special_tokens_values = []

        def __call__(self, text, *, add_special_tokens=True):
            self.add_special_tokens_values.append(add_special_tokens)
            tokens = text.split()
            return {"input_ids": list(range(len(tokens) + int(add_special_tokens)))}

    tokenizer = Tokenizer()
    _messages, _gold, metadata = longcontext.make_needle_prompt(
        tokenizer, 32, depth=0.5, seed=17, return_metadata=True
    )
    assert tokenizer.add_special_tokens_values
    assert set(tokenizer.add_special_tokens_values) == {False}
    assert metadata["requested_filler_tokens"] == 32
    assert metadata["actual_filler_tokens"] >= 32
    assert 0.0 <= metadata["realized_filler_depth"] <= 1.0


def test_old_fixed_chat_signature_retries_without_thinking_kwarg_only():
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(
            self, messages, *, add_generation_prompt, return_tensors, return_dict
        ):
            self.calls.append(messages)
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            assert return_dict is True
            return {"input_ids": torch.tensor([[1, 2]])}

    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "x"}]
    assert longcontext._encode(tokenizer, messages).tolist() == [[1, 2]]
    assert tokenizer.calls == [messages]


def test_prompt_capacity_uses_rendered_prompt_plus_generation():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8, text_config=None))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    assert longcontext._validate_prompt_capacity(model, ids, 3) == 8
    with pytest.raises(ValueError, match=r"5 \+ 4 > 8"):
        longcontext._validate_prompt_capacity(model, ids, 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_lens": []}, "context_lens must be non-empty"),
        ({"context_lens": [8, 8]}, "must not contain duplicates"),
        ({"context_lens": [8], "seeds": 0}, "seeds must be"),
        ({"context_lens": [8], "max_new_tokens": 0}, "max_new_tokens must be"),
        ({"context_lens": [8], "depths": [-0.1]}, "depths must contain"),
    ],
)
def test_grid_validation_rejects_ambiguous_or_empty_experiments(kwargs, message):
    defaults = {"context_lens": [8], "seeds": 1, "max_new_tokens": 1}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        longcontext._validate_grid(**defaults)


def test_fullkv_dynamic_cache_receives_model_config(monkeypatch):
    created_with = []

    class FakeDynamicCache:
        def __init__(self, *, config):
            created_with.append(config)

    class Tokenizer:
        eos_token_id = 2

        def decode(self, *_args, **_kwargs):
            return "1234"

    config = SimpleNamespace(max_position_embeddings=16, text_config=None)
    model = SimpleNamespace(config=config)

    monkeypatch.setattr("transformers.DynamicCache", FakeDynamicCache)
    monkeypatch.setattr(
        longcontext,
        "make_needle_prompt",
        lambda *_args, **_kwargs: ([{"role": "user", "content": "x"}], "1234"),
    )
    monkeypatch.setattr(longcontext, "_encode", lambda *_args: torch.tensor([[1, 2]]))
    monkeypatch.setattr(
        longcontext,
        "greedy_answer",
        lambda *_args, **_kwargs: torch.tensor([[3]]),
    )

    result = longcontext.needle_in_haystack(
        model,
        Tokenizer(),
        context_lens=[4],
        depths=[0.5],
        seeds=1,
        max_new_tokens=1,
    )
    assert result.accuracy == 1.0
    assert created_with == [config]
    assert result.mem_fraction == 1.0
    assert result.mem_fraction_aggregation == "fullkv_no_cache"
    assert result.compression_sufficient_statistics["compression_observations"] == 1
    record = result.as_dict()["grid"][0]
    assert result.hits == 1
    assert record["rendered_prompt_tokens"] == 2
    assert record["reserved_prompt_plus_generation_tokens"] == 3
    assert record["gold"] == "1234"
    assert record["prediction"] == "1234"
    assert record["cache_accounting"]["compression_aggregation"] == "fullkv_no_cache"


def test_synthetic_grid_memory_fraction_uses_exact_ratio_of_sums(monkeypatch):
    class Tokenizer:
        eos_token_id = 2

        def decode(self, *_args, **_kwargs):
            return "1234"

    counters = iter([(100, 50), (900, 90)])

    def cache_factory():
        fp16_bits, stored_bits = next(counters)
        return SimpleNamespace(fp16_bits=fp16_bits, stored_bits=stored_bits)

    config = SimpleNamespace(max_position_embeddings=16, text_config=None)
    model = SimpleNamespace(config=config)
    monkeypatch.setattr(
        longcontext,
        "make_needle_prompt",
        lambda *_args, **_kwargs: ([{"role": "user", "content": "x"}], "1234"),
    )
    monkeypatch.setattr(longcontext, "_encode", lambda *_args: torch.tensor([[1, 2]]))
    monkeypatch.setattr(
        longcontext,
        "greedy_answer",
        lambda *_args, **_kwargs: torch.tensor([[3]]),
    )

    result = longcontext.needle_in_haystack(
        model,
        Tokenizer(),
        cache_factory=cache_factory,
        context_lens=[4],
        depths=[0.25, 0.75],
        seeds=1,
        max_new_tokens=1,
    )

    assert result.mem_fraction == pytest.approx(140 / 1000)
    assert result.mem_fraction != pytest.approx((0.5 + 0.1) / 2)
    assert result.mem_fraction_aggregation == "sum_stored_bits_over_sum_fp16_bits"
    statistics = result.compression_sufficient_statistics
    assert statistics["compression_aggregation"] == ("sum_exact_per_generation_cache_bit_counters")
    assert statistics["compression_fp16_bits"] == 1000
    assert statistics["compression_stored_bits"] == 140
    assert statistics["compression_observations"] == 2
    assert result.as_dict()["compression_sufficient_statistics"] == statistics


def test_synthetic_grid_legacy_memory_fraction_is_explicitly_labeled():
    records = [
        {
            "compression_aggregation": "legacy_cache_ratio_only",
            "compression_ratio": 2.0,
            "adaptive_hq_runtime": None,
        },
        {
            "compression_aggregation": "legacy_cache_ratio_only",
            "compression_ratio": 10.0,
            "adaptive_hq_runtime": None,
        },
    ]

    statistics, fraction, aggregation = longcontext.aggregate_cache_compression_records(records)

    assert fraction == pytest.approx((0.5 + 0.1) / 2)
    assert aggregation == ("legacy_arithmetic_mean_per_generation_mem_fraction_no_exact_counters")
    assert statistics["compression_aggregation"] == (
        "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters"
    )
    assert statistics["compression_fp16_bits"] is None
    assert statistics["compression_stored_bits"] is None
    assert statistics["compression_observations"] == 2


def test_synthetic_grid_rejects_mixed_exact_and_legacy_accounting():
    records = [
        {
            "compression_aggregation": "exact_cache_bit_counters",
            "compression_fp16_bits": 100,
            "compression_stored_bits": 50,
            "adaptive_hq_runtime": None,
        },
        {
            "compression_aggregation": "legacy_cache_ratio_only",
            "compression_ratio": 2.0,
            "adaptive_hq_runtime": None,
        },
    ]

    with pytest.raises(RuntimeError, match="mixed or missing compression accounting modes"):
        longcontext.aggregate_cache_compression_records(records)
