"""Correctness tests for the pinned NeMo-Skills RULER-v1 evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import onecomp.kvcomp.eval.ruler as ruler_module
from onecomp.kvcomp.eval.ruler import (
    NEMO_SKILLS_RULER_SOURCE_COMMIT,
    RULER_SOURCE_COMMIT,
    RULER_V1_MAX_NEW_TOKENS,
    RULER_V1_PROTOCOL_LABEL,
    RULER_V1_SAMPLING_PROTOCOL,
    RULER_V1_TASKS,
    load_ruler_v1_task,
    ruler_v1,
    ruler_v1_sample_score,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_task(root: Path, task: str, *, rows: int = 100, converted: bool = False) -> Path:
    path = root / task / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index in range(rows):
            if converted:
                row = {
                    "index": index,
                    "question": f"question {index} PREFIX",
                    "expected_answer": ["alpha", "beta"],
                    "length": 20,
                }
            else:
                row = {
                    "index": index,
                    "input": f"question {index}",
                    "answer_prefix": " PREFIX",
                    "outputs": ["alpha", "beta"],
                    "length": 20,
                    "length_w_model_temp": 24,
                }
            stream.write(json.dumps(row) + "\n")
    return path


def _write_manifest(
    model_root: Path,
    data_root: Path,
    model_path: Path,
    tasks,
) -> Path:
    files = {}
    for task in tasks:
        path = data_root / task / "test.jsonl"
        files[task] = {
            "path": str(path),
            "rows": 100,
            "sha256": _digest(path),
            "length_w_model_temp_min": 24,
            "length_w_model_temp_max": 24,
        }
    manifest = {
        "schema_version": 1,
        "benchmark": RULER_V1_PROTOCOL_LABEL,
        "model": {
            "key": "fake",
            "path": str(model_path),
            "metadata_files": {},
        },
        "protocol": {
            "num_samples_per_task": 100,
            "prepare_for_nemo_skills": True,
            "sampling_protocol": RULER_V1_SAMPLING_PROTOCOL,
            "tasks": list(tasks),
            "lengths": [int(data_root.name)],
            "tokenizer_type": "hf",
        },
        "sources": {
            "ruler_commit": RULER_SOURCE_COMMIT,
            "nemo_skills_commit": NEMO_SKILLS_RULER_SOURCE_COMMIT,
        },
        "files": {data_root.name: files},
    }
    path = model_root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _FakeTokenizer:
    eos_token_id = 2
    chat_template = "fake-chat-template"

    def __init__(
        self,
        model_path: Path,
        input_length: int = 4,
        *,
        supports_thinking_kwarg: bool = True,
    ):
        self.name_or_path = str(model_path)
        self.input_length = input_length
        self.supports_thinking_kwarg = supports_thinking_kwarg
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        assert kwargs["add_generation_prompt"] is True
        if not self.supports_thinking_kwarg and "enable_thinking" in kwargs:
            raise TypeError("got an unexpected keyword argument 'enable_thinking'")
        if self.supports_thinking_kwarg:
            assert kwargs["enable_thinking"] is False
        else:
            assert "enable_thinking" not in kwargs
        return {"input_ids": torch.zeros((1, self.input_length), dtype=torch.long)}

    def decode(self, _tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "The answers are ALPHA and beta."


class _FakeModel:
    device = torch.device("cpu")

    def __init__(self, context: int = 256):
        self.config = SimpleNamespace(max_position_embeddings=context)
        self.generation_config = SimpleNamespace(eos_token_id=[2, 3, 2])


class _FakeCache:
    def mem_fraction(self):
        return 0.25


def test_substring_scoring_matches_pinned_nvidia_protocol():
    assert ruler_v1_sample_score("niah_multivalue", "ALPHA only", ["alpha", "beta"]) == 0.5
    assert ruler_v1_sample_score("vt", "prefix BeTa suffix", ["alpha", "beta"]) == 0.5
    assert ruler_v1_sample_score("qa_1", "prefix BeTa suffix", ["alpha", "beta"]) == 1.0
    assert ruler_v1_sample_score("qa_2", "neither", ["alpha", "beta"]) == 0.0


def test_loader_uses_verbatim_input_plus_answer_prefix(tmp_path):
    raw_root = tmp_path / "raw"
    _write_task(raw_root, "niah_single_1", rows=2)
    samples, _, _, row_format = load_ruler_v1_task(raw_root, "niah_single_1", expected_samples=2)
    assert samples[0].prompt == "question 0 PREFIX"
    assert row_format == "nvidia_ruler_input_plus_answer_prefix"

    converted_root = tmp_path / "converted"
    _write_task(converted_root, "qa_1", rows=2, converted=True)
    converted, _, _, row_format = load_ruler_v1_task(converted_root, "qa_1", expected_samples=2)
    assert converted[0].prompt == "question 0 PREFIX"
    assert row_format == "nemo_skills_lossless_conversion"


def test_loader_rejects_incomplete_protocol_file(tmp_path):
    _write_task(tmp_path, "fwe", rows=99)
    with pytest.raises(RuntimeError, match="Expected 100 pinned NeMo-Skills RULER v1 rows"):
        load_ruler_v1_task(tmp_path, "fwe")


def test_loader_accepts_upstream_generator_duplicate_random_ids(tmp_path):
    path = _write_task(tmp_path, "niah_single_1", rows=2)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["index"] = rows[0]["index"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    samples, _, _, _ = load_ruler_v1_task(tmp_path, "niah_single_1", expected_samples=2)
    assert [sample.index for sample in samples] == [0, 0]


def test_full_13_task_macro_and_provenance(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_root = tmp_path / "fake-model"
    data_root = model_root / "256"
    for task in RULER_V1_TASKS:
        _write_task(data_root, task)
    manifest_path = _write_manifest(model_root, data_root, model_path, RULER_V1_TASKS)

    seen_max_new = []
    seen_eos = []

    def fake_greedy(_model, _ids, _cache, *, max_new_tokens, **_kwargs):
        seen_max_new.append(max_new_tokens)
        seen_eos.append(_kwargs["eos"])
        return torch.tensor([[7]])

    monkeypatch.setattr(ruler_module, "greedy_answer", fake_greedy)
    tokenizer = _FakeTokenizer(model_path)
    result = ruler_v1(
        _FakeModel(),
        tokenizer,
        data_root=data_root,
        cache_factory=_FakeCache,
    )
    doc = result.as_dict()

    assert result.protocol_complete is True
    assert result.n == 1300
    assert result.score == 100.0
    assert doc["schema_version"] == 3
    assert doc["evaluation_scope"] == (
        "complete_nemo_skills_ruler_v1_100_sample_per_task_data_tasks_and_scorer_with_"
        "paired_decoder_adaptation"
    )
    assert doc["benchmark"] == RULER_V1_PROTOCOL_LABEL
    assert doc["samples_per_task"] == 100
    assert doc["sampling_protocol"] == RULER_V1_SAMPLING_PROTOCOL
    assert doc["protocol_complete"] is True
    assert doc["canonical_nvidia_samples_per_task"] == 500
    assert doc["canonical_nvidia_full_eval"] is False
    assert doc["reference_model_wrapper_exact"] is False
    assert doc["score_sufficient_statistics"] == {
        "task_accuracy_sum": 13.0,
        "task_count": 13,
    }
    assert doc["per_task"]["niah_single_1"]["credit_sum"] == 100.0
    assert doc["per_example"]["niah_single_1"][0]["generated_tokens"] == 1
    assert doc["per_example"]["niah_single_1"][0]["max_new_tokens"] == 128
    assert doc["per_example"]["niah_single_1"][0]["stop_token_ids"] == [2, 3]
    assert doc["per_example"]["niah_single_1"][0]["prompt_truncated"] is False
    assert doc["model_context"] == {
        "source": "model.config.max_position_embeddings",
        "max_position_embeddings": 256,
        "named_ruler_sequence_limit": 256,
        "max_observed_prompt_tokens": 4,
        "max_observed_prompt_plus_max_generation_tokens": 132,
        "max_observed_generated_tokens": 1,
        "capacity_check": (
            "input_tokens + pinned upstream task max_new_tokens <= named RULER sequence "
            "length <= max_position_embeddings"
        ),
    }
    assert doc["generation_manifest"]["path"] == str(manifest_path)
    assert doc["generation_manifest"]["sha256"] == _digest(manifest_path)
    assert doc["protocol"]["ruler_source_commit"] == RULER_SOURCE_COMMIT
    assert doc["protocol"]["nemo_skills_source_commit"] == NEMO_SKILLS_RULER_SOURCE_COMMIT
    assert doc["protocol"]["sampling_protocol"] == RULER_V1_SAMPLING_PROTOCOL
    assert doc["protocol"]["truncation"] == "none; context overflow is fatal"
    assert doc["protocol"]["context_semantics"].startswith(
        "named RULER sequence length is the total prompt-plus-generation limit"
    )
    assert doc["protocol"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "unsupported_keyword_policy": "omit_only_enable_thinking",
    }
    assert doc["protocol"]["eos_token_ids"] == [2, 3]
    assert doc["protocol"]["eos_token_id_source"] == "model.generation_config.eos_token_id"
    assert doc["protocol"]["stop_conditions"] == [
        "any_generation_config_eos_token_id",
        "max_new_tokens",
    ]
    assert doc["protocol"]["prompt_wrappers"] == [
        "generic user message; tokenizer.apply_chat_template with "
        "enable_thinking=false when supported"
    ]
    assert set(seen_max_new) == set(RULER_V1_MAX_NEW_TOKENS.values())
    assert len(seen_max_new) == 1300
    assert seen_eos == [[2, 3]] * 1300
    assert tokenizer.prompts[0] == "question 0 PREFIX"


def test_subset_is_never_labelled_protocol_complete_and_overflow_is_fatal(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_root = tmp_path / "fake-model"
    data_root = model_root / "256"
    _write_task(data_root, "niah_single_1")
    _write_manifest(model_root, data_root, model_path, ["niah_single_1"])
    monkeypatch.setattr(
        ruler_module,
        "greedy_answer",
        lambda *_args, **_kwargs: torch.tensor([[7]]),
    )

    subset = ruler_v1(
        _FakeModel(),
        _FakeTokenizer(model_path),
        data_root=data_root,
        cache_factory=_FakeCache,
        tasks=["niah_single_1"],
        limit_per_task=1,
    )
    assert subset.protocol_complete is False
    assert subset.as_dict()["evaluation_scope"] == (
        "subset_or_smoke_with_paired_decoder_adaptation"
    )

    with pytest.raises(RuntimeError, match="Refusing benchmark-changing truncation"):
        ruler_v1(
            _FakeModel(),
            _FakeTokenizer(model_path, input_length=200),
            data_root=data_root,
            cache_factory=_FakeCache,
            tasks=["niah_single_1"],
            limit_per_task=1,
        )


def test_non_thinking_keyword_fallback_is_only_for_unsupported_signature(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_root = tmp_path / "fake-model"
    data_root = model_root / "256"
    _write_task(data_root, "qa_1")
    _write_manifest(model_root, data_root, model_path, ["qa_1"])
    monkeypatch.setattr(
        ruler_module,
        "greedy_answer",
        lambda *_args, **_kwargs: torch.tensor([[7]]),
    )
    tokenizer = _FakeTokenizer(model_path, supports_thinking_kwarg=False)
    result = ruler_v1(
        _FakeModel(),
        tokenizer,
        data_root=data_root,
        cache_factory=_FakeCache,
        tasks=["qa_1"],
        limit_per_task=1,
    )
    assert result.n == 1
    assert tokenizer.prompts == ["question 0 PREFIX", "question 0 PREFIX"]


def test_ruler_length_is_total_prompt_plus_generation_limit(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    model_root = tmp_path / "fake-model"
    data_root = model_root / "256"
    _write_task(data_root, "niah_single_1")
    _write_manifest(model_root, data_root, model_path, ["niah_single_1"])
    monkeypatch.setattr(
        ruler_module,
        "greedy_answer",
        lambda *_args, **_kwargs: torch.tensor([[7]]),
    )

    # NVIDIA's max_seq_length includes input and the task generation allowance.
    # A larger model context must not silently turn the named 256-token test into
    # a 328-token evaluation.
    result = ruler_v1(
        _FakeModel(context=512),
        _FakeTokenizer(model_path, input_length=128),
        data_root=data_root,
        cache_factory=_FakeCache,
        tasks=["niah_single_1"],
        limit_per_task=1,
    )
    assert result.input_tokens["niah_single_1"]["max"] == 128
    assert result.as_dict()["model_context"]["max_position_embeddings"] == 512
    assert (
        result.as_dict()["model_context"]["max_observed_prompt_plus_max_generation_tokens"] == 256
    )

    with pytest.raises(RuntimeError, match="named RULER sequence limit"):
        ruler_v1(
            _FakeModel(context=512),
            _FakeTokenizer(model_path, input_length=129),
            data_root=data_root,
            cache_factory=_FakeCache,
            tasks=["niah_single_1"],
            limit_per_task=1,
        )
