"""Regression tests for functional-code generation accounting."""

import math
from types import SimpleNamespace

import pytest
import torch

from onecomp.kvcomp.eval import code as code_module


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, _messages, **_kwargs):
        return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    def __call__(self, _text, return_tensors=None):
        del return_tensors
        return SimpleNamespace(input_ids=torch.zeros(1, 2, dtype=torch.long))

    def decode(self, _ids, skip_special_tokens=True):
        del skip_special_tokens
        return "def answer():\n    return 1"


class _Model:
    device = "cpu"


@pytest.mark.parametrize(
    ("problems", "k", "max_new_tokens", "message"),
    [
        ([], 1, 1, "at least one problem"),
        ([{}], 0, 1, "k must be positive"),
        ([{}], 1, 0, "max_new_tokens"),
    ],
)
def test_code_pass_at_1_rejects_undefined_evaluations(problems, k, max_new_tokens, message):
    with pytest.raises(ValueError, match=message):
        code_module.code_pass_at_1(
            _Model(),
            _Tokenizer(),
            problems,
            k=k,
            max_new_tokens=max_new_tokens,
        )


def test_code_pass_at_1_uses_exact_summed_compression_counters(monkeypatch):
    """The displayed ratio is a ratio of bit totals, not a mean of ratios."""

    records = iter(((100, 50), (100, 25)))

    def fake_generate(*_args, **_kwargs):
        fp16_bits, stored_bits = next(records)
        return (
            torch.tensor([1]),
            fp16_bits / stored_bits,
            True,
            {
                "compression_fp16_bits": fp16_bits,
                "compression_stored_bits": stored_bits,
                "compression_aggregation": "exact_cache_bit_counters",
            },
        )

    monkeypatch.setattr(code_module, "_generate", fake_generate)
    monkeypatch.setattr(code_module, "run_code", lambda *_args, **_kwargs: (True, "ok"))
    problems = [
        {"prompt": "p0", "entry_point": "answer", "test": "def check(c): pass"},
        {"prompt": "p1", "entry_point": "answer", "test": "def check(c): pass"},
    ]

    result = code_module.code_pass_at_1(
        _Model(),
        _Tokenizer(),
        problems,
        cache_factory=object,
        max_new_tokens=1,
    )

    assert result.compression_ratio == pytest.approx(200 / 75)
    assert result.compression_ratio_aggregation == ("exact_sum_fp16_bits_over_sum_stored_bits")
    assert result.eviction_applicable is False
    assert result.eviction_fired is None
    assert result.compression_sufficient_statistics == {
        "compression_fp16_bits": 200,
        "compression_stored_bits": 75,
        "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
        "compression_observations": 2,
    }


def test_code_zero_bit_payload_reports_infinite_compression(monkeypatch):
    monkeypatch.setattr(
        code_module,
        "_generate",
        lambda *_args, **_kwargs: (
            torch.tensor([1]),
            float("inf"),
            True,
            {
                "compression_fp16_bits": 100,
                "compression_stored_bits": 0,
                "compression_aggregation": "exact_cache_bit_counters",
            },
        ),
    )
    monkeypatch.setattr(code_module, "run_code", lambda *_args, **_kwargs: (True, "ok"))
    result = code_module.code_pass_at_1(
        _Model(),
        _Tokenizer(),
        [{"prompt": "p", "entry_point": "answer", "test": "def check(c): pass"}],
        cache_factory=object,
        max_new_tokens=1,
    )
    assert math.isinf(result.compression_ratio)


def test_code_fullkv_factory_uses_non_eviction_fullkv_schema(monkeypatch):
    monkeypatch.setattr(
        code_module,
        "_generate",
        lambda *_args, **_kwargs: (
            torch.tensor([1]),
            1.0,
            None,
            {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "fullkv_no_cache",
            },
        ),
    )
    monkeypatch.setattr(code_module, "run_code", lambda *_args, **_kwargs: (True, "ok"))
    result = code_module.code_pass_at_1(
        _Model(),
        _Tokenizer(),
        [{"prompt": "p", "entry_point": "answer", "test": "def check(c): pass"}],
        cache_factory=object,
        max_new_tokens=1,
    )
    document = result.as_dict()
    assert document["compression_ratio"] == 1.0
    assert document["compression_ratio_aggregation"] == "fullkv_no_cache"
    assert document["eviction_applicable"] is False
    assert document["eviction_fired"] is None
