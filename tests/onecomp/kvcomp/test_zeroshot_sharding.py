"""Deterministic, exactly aggregatable zero-shot row shards."""

from __future__ import annotations

import pytest
import torch

import onecomp.kvcomp.eval.zeroshot as zero


class _Rows(list):
    def shuffle(self, seed):
        assert seed == 17
        return _Rows(reversed(self))


def _fake_loader(_task):
    rows = _Rows({"id": index, "gold": index % 2} for index in range(6))

    def examples(dataset):
        for row in dataset:
            yield f"ctx-{row['id']}", [" A", " B"], row["gold"]

    return rows, examples


def _fake_flush(_model, _tokenizer, examples, _cache_factory, **_protocol):
    # Predict option zero for every row. This makes exact correct-count sums
    # sensitive to which rows each shard actually selected.
    accounting = {
        "mode": "fullkv_no_cache",
        "numerator": 2.0,
        "denominator": 2.0,
        "compression_ratio": 1.0,
        "compression_aggregation": "fullkv_no_cache",
        "compression_fp16_bits": None,
        "compression_stored_bits": None,
    }
    return [(0, 0, gold, accounting, 2) for _ctx, _cont, gold, _lengths in examples]


def test_zero_shot_shards_cover_the_same_examples_as_full(monkeypatch):
    monkeypatch.setattr(zero, "_load_task", _fake_loader)
    monkeypatch.setattr(
        zero,
        "_encode_pair",
        lambda *_args: (torch.tensor([[1]]), torch.tensor([[2]])),
    )
    monkeypatch.setattr(zero, "_flush_examples", _fake_flush)

    full = zero.zeroshot_accuracy(None, None, task="arc_easy", sample_seed=17)
    first = zero.zeroshot_accuracy(None, None, task="arc_easy", start=0, limit=3, sample_seed=17)
    second = zero.zeroshot_accuracy(None, None, task="arc_easy", start=3, limit=3, sample_seed=17)

    assert first.n + second.n == full.n == 6
    assert first.correct + second.correct == full.correct == 3
    assert first.correct_norm + second.correct_norm == full.correct_norm == 3
    assert first.as_dict()["selection"]["stop_valid_example_exclusive"] == 3
    assert second.as_dict()["selection"]["start_valid_example"] == 3
    assert second.as_dict()["selection"]["stop_valid_example_exclusive"] == 6
    assert first.as_dict()["protocol_input_hash_schema"] == (zero.PROTOCOL_INPUT_HASH_SCHEMA)
    assert len(first.as_dict()["protocol_input_sha256"]) == 64
    assert full.as_dict()["complete_dataset_eval"] is True
    assert first.as_dict()["complete_dataset_eval"] is False
    assert [row["index"] for row in first.per_example] == [0, 1, 2]
    assert [row["index"] for row in second.per_example] == [3, 4, 5]
    assert all(
        set(row) == {"index", "gold", "prediction", "prediction_norm", "correct", "correct_norm"}
        for row in first.per_example + second.per_example
    )
    assert sum(row["correct"] for row in first.per_example + second.per_example) == 3
    assert sum(row["correct_norm"] for row in first.per_example + second.per_example) == 3


def test_zero_shot_shard_rejects_invalid_or_empty_selection(monkeypatch):
    monkeypatch.setattr(zero, "_load_task", _fake_loader)
    with pytest.raises(ValueError, match="non-negative"):
        zero.zeroshot_accuracy(None, None, start=-1)
    with pytest.raises(ValueError, match="positive"):
        zero.zeroshot_accuracy(None, None, limit=0)
    with pytest.raises(ValueError, match="selection is empty"):
        zero.zeroshot_accuracy(None, None, task="arc_easy", start=6, limit=1)
