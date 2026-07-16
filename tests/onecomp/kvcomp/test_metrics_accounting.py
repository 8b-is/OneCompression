"""Regression tests for exact KV memory-accounting units."""

from types import SimpleNamespace

import pytest

from onecomp.kvcomp.eval.metrics import kv_cache_bytes, summarize_compression
from onecomp.kvcomp.eval.longbench import longbench_e


def _tiny_config():
    return SimpleNamespace(
        num_hidden_layers=1,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=1,
        head_dim=1,
    )


@pytest.mark.parametrize("logical_bits,expected_bytes", [(0, 0), (1, 1), (7, 1), (8, 1), (9, 2)])
def test_kv_cache_bytes_rounds_partial_packed_byte_up(logical_bits, expected_bytes):
    # Two K/V elements make ``bits=logical_bits/2`` yield the requested total.
    assert (
        kv_cache_bytes(
            seq_len=1,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            bits=logical_bits / 2,
        )
        == expected_bytes
    )


def test_summarize_compression_rounds_codec_ledger_up_without_zeroing_it():
    report = summarize_compression(seq_len=1, model_config=_tiny_config(), stored_bits=1)
    assert report.stored_bytes == 1
    assert report.fp16_bytes == 4
    assert report.compression_ratio == 4.0


@pytest.mark.parametrize("stored_bits", [-1, 1.5, True])
def test_summarize_compression_rejects_invalid_codec_ledger(stored_bits):
    with pytest.raises(ValueError, match="stored_bits must be a non-negative integer"):
        summarize_compression(seq_len=1, model_config=_tiny_config(), stored_bits=stored_bits)


@pytest.mark.parametrize(
    "tasks,match",
    [
        ([], "At least one LongBench-E task"),
        (["qasper", "qasper"], "Duplicate LongBench-E tasks"),
    ],
)
def test_longbench_rejects_empty_or_duplicate_task_matrix_before_evaluation(tasks, match):
    with pytest.raises(ValueError, match=match):
        longbench_e(model=object(), tokenizer=object(), tasks=tasks)
