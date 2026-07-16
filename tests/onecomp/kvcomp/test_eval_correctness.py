"""Regression tests for benchmark definitions and cache-accounting invariants."""

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch

import onecomp.kvcomp.eval.longbench as longbench_eval
import onecomp.kvcomp.eval.ppl as ppl_eval
import onecomp.kvcomp.eval.reasoning as reasoning_eval
import onecomp.kvcomp.eval.zeroshot as zeroshot_eval
from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.cache._graded_base import GradedQuantCache
from onecomp.kvcomp.cache.streaming_evict import StreamingLLMEvictCache
from onecomp.kvcomp.cache.thinkv_cache import ThinKVCache
from onecomp.kvcomp.codec.affine import Int2Codec, KIVICodec
from onecomp.kvcomp.codec.halo import HALOCodec
from onecomp.kvcomp.codec.hayate import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.recipes import hayate_frontier_kwargs
from onecomp.kvcomp.codec.lowbit_fp import NVFP4Codec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.config import Budget
from onecomp.kvcomp.eval._teacher_forcing import (
    CAUSAL_INCREMENTAL_PROTOCOL,
    SINGLE_FORWARD_PROTOCOL,
    causal_teacher_forced_logits,
)
from onecomp.kvcomp.eval.longbench import (
    LongBenchResult,
    _bucket,
    _postprocess_prediction,
    _prompt_ids,
    code_sim_score,
    count_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
)
from onecomp.kvcomp.eval.longcodeqa import (
    LONGCODEQA_BRACKET_COUNTS,
    LONGCODEQA_DATASET_REVISION,
    LONGCODEQA_DATASET_SHA256,
    LongCodeQAResult,
    _effective_eos_token_ids,
    _encode_prompt,
    aggregate_longcodeqa,
    extract_longcodeqa_letter,
    load_longcodeqa,
)
from onecomp.kvcomp.eval.longcontext import greedy_answer
from onecomp.kvcomp.eval.reasoning import (
    ReasoningResult,
    extract_answer,
    math_equivalent,
    math_verify_equivalent,
)
from onecomp.kvcomp.eval.ppl import perplexity_over_text
from onecomp.kvcomp.eval.zeroshot import (
    ZeroShotResult,
    _aggregate_compression_accounting,
    _cache_compression_accounting,
    _encode_pair,
    _hellaswag,
    _score_choices,
    _winogrande,
)
from onecomp.kvcomp.select.h2o import H2OSelector
from onecomp.kvcomp.recipe.thinkv import ThinKVPolicy


@pytest.mark.parametrize(
    ("gold", "generated"),
    [
        (r"\frac{1}{2}", r"therefore \boxed{\frac{1}{2}}"),
        ("0.5", r"\boxed{\frac{1}{2}}"),
        (r"\sqrt{4}", r"\boxed{2}"),
        ("1/3", r"\boxed{\frac{1}{3}}"),
        (r"\left(-\frac{5}{2},-\frac{5}{2}\right)", r"\boxed{(-2.5,-2.5)}"),
        ("12x-34", r"\boxed{-34+12x}"),
    ],
)
def test_math_grading_handles_structured_answers(gold, generated):
    assert math_equivalent(extract_answer(generated), gold)


def test_math_grading_does_not_match_first_number_in_gold():
    assert not math_equivalent(extract_answer("The final answer is 1"), r"\frac{1}{3}")


def test_math_verify_grades_complete_generation_and_symbolic_gold():
    pytest.importorskip("math_verify")
    assert math_verify_equivalent(
        r"A discarded guess was 17. Therefore the final answer is $\boxed{0.5}$.",
        r"\frac{1}{2}",
    )
    assert math_verify_equivalent(
        r"The double sum simplifies to $\boxed{p-q}$.",
        "p - q",
    )
    assert not math_verify_equivalent(
        r"A discarded guess was $\frac{1}{2}$. Final answer: $\boxed{\frac{2}{3}}$.",
        r"\frac{1}{2}",
    )
    with pytest.raises(RuntimeError, match="pinned to math-verify"):
        math_verify_equivalent("1", "1", required_version="0.0.0")


def test_math_verify_missing_dependency_never_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(RuntimeError, match="requires math-verify"):
        reasoning_eval._build_math_verify_scorer()


def test_reasoning_full_generation_scorer_receives_untruncated_text(monkeypatch):
    generated_text = r"Intermediate values: 7 and 9. Final: $\boxed{\frac{1}{2}}$."
    seen = []

    class Model:
        device = torch.device("cpu")

    class Tokenizer:
        def decode(self, _tokens, **_kwargs):
            return generated_text

    monkeypatch.setattr(
        reasoning_eval, "_build_prompt", lambda *_args, **_kwargs: torch.tensor([[1]])
    )
    monkeypatch.setattr(
        reasoning_eval,
        "_generate",
        lambda *_args, **_kwargs: (
            torch.tensor([2]),
            1.0,
            True,
            {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "fullkv_no_cache",
                "adaptive_hq_runtime": None,
            },
        ),
    )

    result = reasoning_eval._eval_dataset(
        Model(),
        Tokenizer(),
        [("question", r"\frac{1}{2}")],
        str,
        "math500",
        None,
        1,
        8,
        False,
        0.6,
        0.95,
        None,
        False,
        answer_eq=lambda prediction, gold: seen.append((prediction, gold)) or True,
        score_full_generation=True,
    )
    assert seen == [(generated_text, r"\frac{1}{2}")]
    assert result.pass_at_1 == 1.0
    sample = result.per_question[0]["samples"][0]
    assert "extracted_answer" not in sample
    assert sample["grading_input"] == "generated_text"
    assert sample["legacy_regex_extracted_answer_non_grading"] == r"\frac{1}{2}"
    assert sample["generated_text"] == generated_text
    document = result.as_dict()
    assert document["generation_config"]["max_new_tokens"] == 8
    assert document["eviction_applicable"] is False
    assert document["eviction_fired"] is None


def test_math_verify_polar_answer_never_confuses_legacy_regex_with_grading_input(monkeypatch):
    pytest.importorskip("math_verify")
    generated_text = (
        r"Therefore the polar coordinates are $(3, \frac{\pi}{2})$. "
        r"The final answer is $(3, \frac{\pi}{2})$."
    )

    class Model:
        device = torch.device("cpu")

    class Tokenizer:
        def decode(self, _tokens, **_kwargs):
            return generated_text

    monkeypatch.setattr(
        reasoning_eval, "_build_prompt", lambda *_args, **_kwargs: torch.tensor([[1]])
    )
    monkeypatch.setattr(
        reasoning_eval,
        "_generate",
        lambda *_args, **_kwargs: (
            torch.tensor([2]),
            1.0,
            None,
            {
                "compression_fp16_bits": None,
                "compression_stored_bits": None,
                "compression_aggregation": "fullkv_no_cache",
                "adaptive_hq_runtime": None,
            },
        ),
    )
    scorer, _ = reasoning_eval._build_math_verify_scorer()
    result = reasoning_eval._eval_dataset(
        Model(),
        Tokenizer(),
        [("question", r"\left( 3, \frac{\pi}{2} \right)")],
        str,
        "math500",
        lambda: object(),
        1,
        8,
        False,
        0.6,
        0.95,
        None,
        False,
        answer_eq=scorer,
        score_full_generation=True,
    )
    sample = result.per_question[0]["samples"][0]
    assert sample["correct"] is True
    assert sample["legacy_regex_extracted_answer_non_grading"] == "2"
    assert sample["grading_input"] == "generated_text"
    assert "extracted_answer" not in sample
    assert result.compression_ratio_aggregation == "fullkv_no_cache"
    assert result.eviction_applicable is False
    assert result.eviction_fired is None


def test_reasoning_result_rejects_nonapplicable_eviction_event():
    result = ReasoningResult("math500", 1, 1, 1.0, 1.0, eviction_fired=True)
    with pytest.raises(ValueError, match="Non-applicable eviction"):
        result.as_dict()


def test_reasoning_compression_ratio_uses_summed_exact_bit_counters(monkeypatch):
    class Model:
        device = torch.device("cpu")

    class Tokenizer:
        def decode(self, _tokens, **_kwargs):
            return "1"

    monkeypatch.setattr(
        reasoning_eval, "_build_prompt", lambda *_args, **_kwargs: torch.tensor([[1]])
    )
    generations = iter(
        [
            (
                torch.tensor([2]),
                10.0,
                True,
                {
                    "compression_fp16_bits": 160,
                    "compression_stored_bits": 80,
                    "compression_aggregation": "exact_cache_bit_counters",
                    "adaptive_hq_runtime": None,
                },
            ),
            (
                torch.tensor([2]),
                20.0,
                True,
                {
                    "compression_fp16_bits": 320,
                    "compression_stored_bits": 80,
                    "compression_aggregation": "exact_cache_bit_counters",
                    "adaptive_hq_runtime": None,
                },
            ),
        ]
    )
    monkeypatch.setattr(reasoning_eval, "_generate", lambda *_args, **_kwargs: next(generations))

    result = reasoning_eval._eval_dataset(
        Model(),
        Tokenizer(),
        [("q1", "1"), ("q2", "1")],
        str,
        "math500",
        lambda: object(),
        1,
        8,
        False,
        0.6,
        0.95,
        None,
        False,
        answer_eq=lambda *_args: True,
    )

    assert result.compression_ratio == 3.0
    assert result.compression_ratio_aggregation == "exact_sum_fp16_bits_over_sum_stored_bits"
    assert result.compression_sufficient_statistics["compression_fp16_bits"] == 480
    assert result.compression_sufficient_statistics["compression_stored_bits"] == 160


def test_reasoning_chat_template_failure_never_drops_prompt_roles():
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("bad template")

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("raw fallback must not be attempted")

    with pytest.raises(RuntimeError, match="chat-template rendering failed"):
        reasoning_eval._build_prompt(Tokenizer(), "question", "system")


def test_math_verify_raw_query_never_uses_a_chat_template():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Question: problem\nStep-by-Step Answer:"
            assert kwargs == {"return_tensors": "pt", "truncation": False}
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("Math-Verify public task is raw causal text")

    ids = reasoning_eval._build_prompt(
        Tokenizer(),
        "Question: problem\nStep-by-Step Answer:",
        None,
        use_chat_template=False,
    )
    assert ids.tolist() == [[1, 2, 3]]


def test_reasoning_eviction_decode_stops_on_first_multi_eos_token():
    class Cache:
        def evict_now(self):
            return None

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **_kwargs):
            self.calls += 1
            logits = torch.full((*input_ids.shape, 5), -10.0)
            logits[..., 3] = 10.0
            return type("Out", (), {"logits": logits})()

    model = Model()
    generated = reasoning_eval._eviction_decode(
        model,
        torch.tensor([[1, 2]]),
        Cache(),
        max_new_tokens=8,
        eos=[2, 3],
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    assert generated.tolist() == [3]
    assert model.calls == 1


def test_reasoning_eviction_stop_criterion_sees_prompt_plus_generation():
    class Cache:
        def evict_now(self):
            return None

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, **_kwargs):
            logits = torch.full((*input_ids.shape, 5), -10.0)
            logits[..., 3] = 10.0
            return type("Out", (), {"logits": logits})()

    seen = []

    def stop_criterion(ids, _logits):
        seen.append(ids.tolist())
        return torch.tensor([True])

    generated = reasoning_eval._eviction_decode(
        Model(),
        torch.tensor([[1, 2]]),
        Cache(),
        max_new_tokens=8,
        eos=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        stop_criterion=stop_criterion,
    )
    assert generated.tolist() == [3]
    assert seen == [[[1, 2, 3]]]


def test_reasoning_result_always_reports_mean_pass_at_1():
    result = ReasoningResult("aime2024", 1, 8, 0.125, 10, observed_pass_at_k=1.0)
    doc = result.as_dict()
    assert doc["pass@1"] == 0.125
    assert doc["observed_pass@8"] == 1.0
    assert "pass@8" not in doc


def test_reasoning_result_records_grader_version_and_protocol():
    result = ReasoningResult(
        "math500",
        500,
        1,
        0.5,
        10,
        dataset_size=500,
        grader="math_verify",
        grader_version="0.9.0",
        grader_config={"prediction_extraction": "full_generation:latex+expr"},
        protocol="hf_math500_test_math_verify_full_generation",
    )
    doc = result.as_dict()
    assert doc["grader"] == "math_verify"
    assert doc["grader_version"] == "0.9.0"
    assert doc["grader_config"]["prediction_extraction"].startswith("full_generation")
    assert doc["protocol"] == "hf_math500_test_math_verify_full_generation"


def test_math500_shard_selects_global_interval(monkeypatch):
    class Dataset:
        def __len__(self):
            return 500

        def __getitem__(self, index):
            return {"problem": f"question-{index}", "answer": str(index)}

    monkeypatch.setattr("datasets.load_dataset", lambda *_args, **_kwargs: Dataset())
    monkeypatch.setattr(
        reasoning_eval,
        "_build_math_verify_scorer",
        lambda _version: (lambda _prediction, _gold: True, "0.9.0"),
    )
    captured = {}

    def fake_eval(*_args, **kwargs):
        captured.update(kwargs)
        captured["items"] = _args[2]
        return ReasoningResult("math500", 50, 1, 1.0, 1.0)

    monkeypatch.setattr(reasoning_eval, "_eval_dataset", fake_eval)
    reasoning_eval.math500_pass_at_1(object(), object(), start=50, limit=50, grader="math_verify")
    assert captured["items"][0] == (
        reasoning_eval._MATH500_FEWSHOT_PREFIX + "\n\nQuestion: question-50\nStep-by-Step Answer:",
        "50",
    )
    assert captured["items"][-1] == (
        reasoning_eval._MATH500_FEWSHOT_PREFIX + "\n\nQuestion: question-99\nStep-by-Step Answer:",
        "99",
    )
    assert captured["start_index"] == 50
    assert captured["selection_limit"] == 50
    assert captured["stop_sequences"] == reasoning_eval._MATH500_STOP_SEQUENCES
    assert captured["prompt_config"]["few_shot"] == 4
    assert captured["prompt_config"]["few_shot_order_one_based"] == [4, 3, 2, 1]
    assert captured["prompt_config"]["few_shot_content_sha256"] == (
        "1360154eb365a22beb1a201dd6ff159e829b6d64d59f69df29530700034c07a4"
    )
    assert captured["prompt_config"]["protocol_id"] == (
        "math500_adapted_composite_math_verify_ba3d3aa_" "lighteval_prompt_manager_7c1cd62_v1"
    )
    assert captured["prompt_config"]["protocol_origin"] == "adapted_composite"
    assert captured["prompt_config"]["lighteval_prompt_manager_reference_version"] == ("0.13.0")
    assert captured["prompt_config"]["lighteval_prompt_manager_source_commit"] == (
        "7c1cd62716b0a198a630c26d781430c54726cd02"
    )
    assert captured["prompt_config"]["upstream_component_pair_executable"] is False
    assert captured["prompt_config"]["upstream_lighteval_builtin_task_exact"] is False
    assert captured["prompt_config"]["upstream_math_verify_custom_task_exact"] is False
    assert captured["prompt_config"]["upstream_prompt_generation_components_exact"] is True
    assert captured["prompt_config"]["lighteval_balanced_sampler_seed"] == 0
    assert captured["prompt_config"]["math_verify_task_source_commit"] == (
        "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b"
    )
    assert captured["use_chat_template"] is False
    assert captured["generation_config"]["reference_max_new_tokens"] == 1024
    assert captured["protocol"] == ("hf_math500_test_adapted_composite_four_shot")


def test_math500_four_shot_fixture_is_content_addressed_and_reversed_balanced_order():
    assert reasoning_eval._MATH500_FEWSHOT_ORDER == (4, 3, 2, 1)
    assert hashlib.sha256(reasoning_eval._MATH500_FEWSHOT_PREFIX.encode("utf-8")).hexdigest() == (
        "1360154eb365a22beb1a201dd6ff159e829b6d64d59f69df29530700034c07a4"
    )
    demonstrations = reasoning_eval._MATH500_FEWSHOT_DEMONSTRATIONS
    assert demonstrations[0].startswith("Question: If the system of equations")
    assert demonstrations[1].startswith("Question: Terrell usually lifts")
    assert demonstrations[2].startswith("Question: If $\\det")
    assert demonstrations[3].startswith("Question: Find the domain")


def test_longbench_official_fractional_count_and_retrieval_metrics():
    assert count_score("7 and 8", "7") == 0.5
    assert retrieval_score("Paragraph 7, then Paragraph 8", "Paragraph 7") == 0.5


def test_longbench_qa_empty_answer_edge_case():
    assert qa_f1_score("the", "an") == 0.0
    assert qa_f1_score("the", "answer") == 0.0


def test_longbench_released_data_complete_requires_all_tasks():
    partial = LongBenchResult(n=1, score=0.0, tasks=["qasper"])
    assert not partial.as_dict()["released_data_complete"]


def test_longbench_chat_wrapper_respects_official_raw_task_split():
    class Tok:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[9, 8]])}

        def __call__(self, *args, **kwargs):
            return type("Encoded", (), {"input_ids": torch.tensor([[1]])})()

    assert _prompt_ids(Tok(), "x", "qasper").tolist() == [[9, 8]]
    assert _prompt_ids(Tok(), "x", "lcc").tolist() == [[1]]


def test_longbench_short_answer_tasks_score_only_first_line():
    assert _postprocess_prediction("trec", "\nlabel\nexplanation") == "label"
    assert _postprocess_prediction("triviaqa", "answer\nmore") == "answer"
    assert _postprocess_prediction("samsum", "summary\nmore") == "summary"
    assert _postprocess_prediction("qasper", "answer\nmore") == "answer\nmore"


def test_longbench_truncates_before_chat_wrapping():
    class Enc:
        def __init__(self, ids):
            self.input_ids = ids

    class Tok:
        def __init__(self):
            self.decoded = []

        def __call__(self, *args, **kwargs):
            return Enc(torch.tensor([[1, 2, 3, 4, 5, 6]]))

        def decode(self, ids, **kwargs):
            self.decoded.append(ids.tolist())
            return "head" if ids.tolist() == [1, 2] else "tail"

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "headtail"
            return {"input_ids": torch.tensor([[9]])}

    tok = Tok()
    assert _prompt_ids(tok, "raw", "qasper", max_length=4).tolist() == [[9]]
    assert tok.decoded == [[1, 2], [5, 6]]


def test_longbench_chat_template_failure_never_falls_back_to_raw():
    class Enc:
        input_ids = torch.tensor([[1, 2, 3]])

    class Tok:
        def __call__(self, *_args, **_kwargs):
            return Enc()

        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("invalid model template")

    with pytest.raises(RuntimeError, match="chat template failed"):
        _prompt_ids(Tok(), "raw", "qasper", max_length=32)


def test_longbench_unrelated_chat_template_typeerror_is_not_retried():
    class Enc:
        input_ids = torch.tensor([[1, 2, 3]])

    class Tok:
        calls = 0

        def __call__(self, *_args, **_kwargs):
            return Enc()

        def apply_chat_template(self, *_args, **_kwargs):
            self.calls += 1
            raise TypeError("message content must be a string")

    tok = Tok()
    with pytest.raises(RuntimeError, match="chat template failed"):
        _prompt_ids(tok, "raw", "qasper", max_length=32)
    assert tok.calls == 1


def test_longbench_retries_only_unsupported_enable_thinking_kwarg():
    class Enc:
        input_ids = torch.tensor([[1, 2, 3]])

    class Tok:
        calls = 0

        def __call__(self, *_args, **_kwargs):
            return Enc()

        def apply_chat_template(self, *_args, **kwargs):
            self.calls += 1
            if "enable_thinking" in kwargs:
                raise TypeError("got an unexpected keyword argument 'enable_thinking'")
            return {"input_ids": torch.tensor([[9]])}

    tok = Tok()
    assert _prompt_ids(tok, "raw", "qasper", max_length=32).tolist() == [[9]]
    assert tok.calls == 2


def test_longbench_uses_all_model_generation_eos_and_samsum_newline(monkeypatch):
    rows = [
        {
            "context": "dialogue",
            "input": "summarize",
            "answers": ["summary"],
            "all_classes": None,
            "length": 100,
        }
        for _ in range(longbench_eval.LONGBENCH_E_TASK_COUNTS["samsum"])
    ]
    monkeypatch.setattr(longbench_eval, "_load_task", lambda _task: rows)
    monkeypatch.setattr(longbench_eval, "_longbench_archive", lambda: ("data.zip", "a" * 64))
    monkeypatch.setitem(longbench_eval._METRIC, "samsum", lambda pred, gold: pred == gold)

    seen = []

    def fake_greedy(_model, _ids, _cache, **kwargs):
        seen.append(kwargs["eos"])
        return torch.tensor([[7]])

    monkeypatch.setattr("onecomp.kvcomp.eval.longcontext.greedy_answer", fake_greedy)

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, _text, **_kwargs):
            return type("Encoded", (), {"input_ids": torch.tensor([[1, 2, 3]])})()

        def encode(self, text, **_kwargs):
            assert text == "\n"
            return [41, 42]

        def decode(self, _tokens, **_kwargs):
            return "summary"

    class Model:
        device = torch.device("cpu")
        config = type("Config", (), {"max_position_embeddings": 4096})()
        generation_config = type("GenerationConfig", (), {"eos_token_id": [7, 8, 7]})()

    result = longbench_eval.longbench_e(
        Model(),
        Tokenizer(),
        cache_factory=object,
        tasks=["samsum"],
        max_length=2048,
        limit_per_task=1,
    ).as_dict()
    assert seen == [[7, 8, 42]]
    assert result["generation"]["eos_token_ids"] == [7, 8]
    assert result["generation"]["eos_token_id_source"] == "model.generation_config.eos_token_id"
    assert result["generation"]["samsum_newline_token_id"] == 42
    assert result["per_example"]["samsum"][0]["stop_token_ids"] == [7, 8, 42]


def test_longbench_prompt_metadata_makes_middle_truncation_auditable():
    class Enc:
        def __init__(self, ids):
            self.input_ids = ids

    class Tok:
        def __call__(self, *_args, **_kwargs):
            return Enc(torch.tensor([[1, 2, 3, 4, 5, 6]]))

        def decode(self, ids, **_kwargs):
            return "head" if ids.tolist() == [1, 2] else "tail"

    ids, metadata = _prompt_ids(Tok(), "raw", "lcc", max_length=4, return_metadata=True)
    assert ids.shape == (1, 6)
    assert metadata == {
        "raw_prompt_tokens": 6,
        "raw_prompt_token_budget": 4,
        "middle_truncated": True,
        "chat_wrapper_used": False,
        "model_input_tokens": 6,
    }


def test_greedy_answer_stops_when_first_token_is_any_eos_id():
    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **_kwargs):
            self.calls += 1
            logits = torch.full((*input_ids.shape, 5), -10.0)
            logits[..., 3] = 10.0
            return type("Out", (), {"logits": logits})()

    model = Model()
    generated = greedy_answer(
        model, torch.tensor([[1, 2]]), object(), max_new_tokens=8, eos=[2, 3]
    )
    assert generated.tolist() == [[3]]
    assert model.calls == 1


@pytest.mark.parametrize("bad_call", [1, 2])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_greedy_answer_rejects_nonfinite_next_token_logits(bad_call, bad_value):
    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, **_kwargs):
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 5), dtype=torch.float32)
            logits[..., 3] = 1.0
            if self.calls == bad_call:
                logits[:, -1, 1] = bad_value
            return type("Out", (), {"logits": logits})()

    with pytest.raises(FloatingPointError, match="non-finite next-token logits"):
        greedy_answer(Model(), torch.tensor([[1, 2]]), object(), max_new_tokens=2)


def test_longbench_code_metric_skips_comments_and_fences():
    assert code_sim_score("# explanation\ncorrect_call(x)", "correct_call(x)") == 1.0
    assert code_sim_score("```python\ncorrect_call(x)", "correct_call(x)") == 1.0


def test_longbench_rouge_matches_official_case_sensitive_package():
    assert rouge_score("The Cat", "the cat") == pytest.approx(0.0)
    assert rouge_score("the cat", "the cat") == pytest.approx(1.0)


def test_longbench_buckets_use_official_decimal_boundaries():
    assert _bucket(3999) == "0-4k"
    assert _bucket(4000) == "4-8k"
    assert _bucket(7999) == "4-8k"
    assert _bucket(8000) == "8k+"


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        ("B", "B"),
        ("(c)", None),
        ("Final Answer: D", "D"),
        ("After checking the code, the answer is A.", None),
        ("After checking the code, I choose B", "B"),
        ("Final Answer: A; reconsidered. Final Answer: D", "A"),
        ("A function may return B in one branch.", None),
        ("", None),
    ],
)
def test_longcodeqa_letter_extraction_matches_upstream(text, letter):
    assert extract_longcodeqa_letter(text) == letter


def test_longcodeqa_self_hosted_prompt_is_raw_and_never_chat_wrapped():
    class Tokenizer:
        chat_template = "must not be used"

        def __call__(self, text, **kwargs):
            assert text == "released prompt"
            assert kwargs == {"return_tensors": "pt", "truncation": False}
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def apply_chat_template(self, *_args, **_kwargs):
            raise AssertionError("LongCodeQA self-hosted protocol must stay raw")

    assert _encode_prompt(Tokenizer(), "released prompt").tolist() == [[1, 2, 3]]


def test_longcodeqa_prefers_model_generation_eos_list_over_tokenizer_scalar():
    generation_config = type("GenerationConfig", (), {"eos_token_id": [7, 8, 7]})()
    model = type("Model", (), {"generation_config": generation_config})()
    tokenizer = type("Tokenizer", (), {"eos_token_id": 9})()

    token_ids, source = _effective_eos_token_ids(model, tokenizer)

    assert token_ids == [7, 8]
    assert source == "model.generation_config.eos_token_id"


def test_longcodeqa_rejects_invalid_generation_eos_ids():
    generation_config = type("GenerationConfig", (), {"eos_token_id": [7, True]})()
    model = type("Model", (), {"generation_config": generation_config})()
    tokenizer = type("Tokenizer", (), {"eos_token_id": 9})()

    with pytest.raises(ValueError, match="non-negative integer"):
        _effective_eos_token_ids(model, tokenizer)


def test_longcodeqa_loader_rejects_count_correct_but_unpinned_data(tmp_path):
    rows = [
        {
            "prompt": f"synthetic prompt {index}",
            "correct_letter": "A",
            "repo": "owner/repo",
            "question": "question",
        }
        for index in range(LONGCODEQA_BRACKET_COUNTS["32K"])
    ]
    path = tmp_path / "LQA/32K.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_longcodeqa(str(tmp_path), "32K")


def test_longcodeqa_pins_every_final_filtered_bracket_hash():
    assert set(LONGCODEQA_DATASET_SHA256) == set(LONGCODEQA_BRACKET_COUNTS)
    assert all(len(digest) == 64 for digest in LONGCODEQA_DATASET_SHA256.values())


def test_longcodeqa_aggregate_does_not_claim_full_dataset_for_one_bracket():
    result = LongCodeQAResult(
        bracket="32K",
        n=113,
        correct=56,
        accuracy=56 / 113,
        accuracy_stderr=0.1,
        accuracy_ci95=(0.4, 0.6),
        input_tokens=[32000] * 113,
        dataset_path="32K.json",
        dataset_sha256=LONGCODEQA_DATASET_SHA256["32K"],
        dataset_revision=LONGCODEQA_DATASET_REVISION,
        max_context_tokens=131072,
        max_new_tokens=16,
    )
    doc = aggregate_longcodeqa([result])
    assert doc["n"] == 113
    assert doc["brackets"]["32K"]["released_data_complete_bracket"]
    assert not doc["released_data_complete_dataset"]
    assert doc["official_primary_reporting"] == "per_bracket_exact_match_accuracy"
    bracket_doc = doc["brackets"]["32K"]
    assert bracket_doc["prompt_protocol"] == "released_prompt_verbatim_raw_causal_text"
    assert bracket_doc["chat_template"] == "none"
    assert bracket_doc["generation"]["do_sample"] is False
    assert bracket_doc["generation"]["official_reference_decoding_exact"] is False
    assert bracket_doc["reference_evaluator_executable_verbatim"] is False
    assert "stores response" in bracket_doc["reference_evaluator_known_issue"]
    assert bracket_doc["generation"]["stop_conditions"] == [
        "any_eos_token_id",
        "max_new_tokens",
    ]
    assert bracket_doc["per_example_source_fields"] == [
        "index",
        "repo",
        "question",
        "correct_letter_as_gold",
    ]
    assert bracket_doc["per_example_prompt_retained"] is False
    assert doc["official_reference_decoding_exact"] is False


def test_hellaswag_matches_lm_eval_preprocessing():
    rows = [
        {
            "activity_label": "Making tea",
            "ctx_a": "A person boils water.",
            "ctx_b": "they add [title] leaves",
            "endings": [" [noise] and wait", "leave"],
            "label": "0",
        }
    ]
    ctx, choices, gold = next(_hellaswag(rows))
    assert ctx == "Making tea: A person boils water. They add. leaves"
    # preprocess() leaves one space after removing the bracket artefact, and
    # lm-eval's request builder prepends its own continuation space.
    assert choices[0] == "  and wait"
    assert gold == 0


def test_hellaswag_preserves_lm_eval_whitespace_semantics():
    rows = [
        {
            "activity_label": "Act",
            "ctx_a": "line one\nline two",
            "ctx_b": "has   three spaces",
            "endings": ["keeps\na newline", "other"],
            "label": "1",
        }
    ]
    ctx, choices, _ = next(_hellaswag(rows))
    # lm-eval 0.4.11 runs one literal double-space replacement; it does not
    # normalize arbitrary whitespace with a regular expression.
    assert ctx == "Act: line one\nline two Has  three spaces"
    assert choices[0] == " keeps\na newline"


def test_winogrande_matches_lm_eval_multiple_input_protocol():
    row = {
        "sentence": "The trophy does not fit because _ is too big.",
        "option1": "it",
        "option2": "the case",
        "answer": "1",
    }
    contexts, continuations, gold = next(_winogrande([row]))
    assert contexts == [
        "The trophy does not fit because it",
        "The trophy does not fit because the case",
    ]
    assert continuations == [" is too big.", " is too big."]
    assert gold == 0


def test_zeroshot_pair_tokenization_matches_lm_eval_joined_boundary():
    class Tok:
        def __call__(self, text, **_kwargs):
            mapping = {"ctx": [1], "ctx choice": [1, 9], " choice": [2, 3]}
            return type("Encoded", (), {"input_ids": torch.tensor([mapping[text]])})()

    ctx, cont = _encode_pair(Tok(), "ctx ", "choice")
    assert ctx.tolist() == [[1]]
    assert cont.tolist() == [[9]]  # not independently-tokenized [2, 3]


def test_zeroshot_choice_decision_preserves_double_precision(monkeypatch):
    """A float32 down-cast must not manufacture an argmax tie."""

    def close_scores(*_args, **_kwargs):
        return (
            [-100.00000001, -100.0],
            [1, 1],
            {
                "compression_ratio": 1.0,
                "compression_aggregation": "fullkv_no_cache",
            },
        )

    monkeypatch.setattr(zeroshot_eval, "_score_choices", close_scores)
    rows = zeroshot_eval._flush_examples(
        object(),
        object(),
        [(torch.tensor([[1]]), [torch.tensor([[2]]), torch.tensor([[3]])], 1, [1, 1])],
        None,
    )
    assert rows[0][:3] == (1, 1, 1)


def test_zeroshot_primary_metric_uses_official_winogrande_acc():
    wino = ZeroShotResult("winogrande", 1, acc=1.0, acc_norm=0.0).as_dict()
    hella = ZeroShotResult("hellaswag", 1, acc=0.0, acc_norm=1.0).as_dict()
    assert (wino["primary_metric"], wino["primary_score"]) == ("acc", 1.0)
    assert (hella["primary_metric"], hella["primary_score"]) == ("acc_norm", 1.0)


def test_zeroshot_compression_aggregates_exact_bits_not_choice_ratio_mean():
    result = _aggregate_compression_accounting(
        [
            {"mode": "exact_cache_bit_counters", "numerator": 400, "denominator": 100},
            {"mode": "exact_cache_bit_counters", "numerator": 300, "denominator": 150},
        ]
    )
    assert result["compression_ratio"] == pytest.approx(700 / 250)
    assert result["compression_ratio"] != pytest.approx((4.0 + 2.0) / 2.0)
    assert result["compression_aggregation"] == "exact_cache_bit_counters"
    assert result["compression_fp16_bits"] == 700
    assert result["compression_stored_bits"] == 250


def test_zeroshot_compression_uses_input_weighted_harmonic_ratio_fallback():
    class Cache:
        def __init__(self, ratio):
            self.ratio = ratio

        def compression_ratio(self):
            return self.ratio

    short = _cache_compression_accounting(Cache(4.0), reference_tokens=10)
    long = _cache_compression_accounting(Cache(2.0), reference_tokens=30)
    result = _aggregate_compression_accounting([short, long])
    assert result["compression_ratio"] == pytest.approx(40 / (10 / 4 + 30 / 2))
    assert result["compression_ratio"] != pytest.approx(3.0)
    assert result["compression_aggregation"] == "token_weighted_cache_ratio"
    assert result["compression_fp16_bits"] is None
    assert result["compression_stored_bits"] is None


def test_zeroshot_choice_scorer_sums_exact_cache_counters_across_unequal_choices():
    class Cache:
        fp16_bits = 0
        stored_bits = 0

        def teacher_forced_temporal_dependency(self):
            return None

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, *, past_key_values, **_kwargs):
            tokens = int(input_ids.numel())
            past_key_values.fp16_bits = tokens * 100
            past_key_values.stored_bits = tokens * tokens * 10
            logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32)
            return type("Output", (), {"logits": logits})()

    _sums, lengths, accounting = _score_choices(
        Model(),
        None,
        torch.tensor([[1, 2]]),
        [torch.tensor([[3]]), torch.tensor([[4, 5, 6]])],
        Cache,
    )
    assert lengths == [1, 3]
    assert accounting["compression_fp16_bits"] == 3 * 100 + 5 * 100
    assert accounting["compression_stored_bits"] == 3 * 3 * 10 + 5 * 5 * 10
    assert accounting["compression_ratio"] == pytest.approx(800 / 340)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_zeroshot_choice_scorer_rejects_nonfinite_logits(bad_value):
    class Cache:
        fp16_bits = 0
        stored_bits = 0

        def teacher_forced_temporal_dependency(self):
            return None

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, *, past_key_values, **_kwargs):
            past_key_values.fp16_bits = int(input_ids.numel()) * 100
            past_key_values.stored_bits = int(input_ids.numel()) * 50
            logits = torch.zeros((*input_ids.shape, 16), dtype=torch.float32)
            logits[..., 0] = bad_value
            return type("Output", (), {"logits": logits})()

    with pytest.raises(FloatingPointError, match="non-finite logits"):
        _score_choices(
            Model(),
            None,
            torch.tensor([[1, 2]]),
            [torch.tensor([[3]])],
            Cache,
        )


@pytest.mark.parametrize("do_sample", [False, True])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_reasoning_next_token_selection_rejects_nonfinite_logits(do_sample, bad_value):
    logits = torch.tensor([[0.0, bad_value, 1.0]])
    with pytest.raises(FloatingPointError, match="non-finite next-token logits"):
        reasoning_eval._sample_next(logits, do_sample, temperature=0.6, top_p=0.95)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_reasoning_model_generate_logits_processor_rejects_nonfinite_scores(bad_value):
    processor = reasoning_eval._FailClosedFiniteLogitsProcessor()
    with pytest.raises(FloatingPointError, match="non-finite next-token logits"):
        processor(torch.tensor([[1]]), torch.tensor([[0.0, bad_value, 1.0]]))


def test_reasoning_sampling_rejects_invalid_probability_normalization(monkeypatch):
    monkeypatch.setattr(
        reasoning_eval.torch,
        "softmax",
        lambda logits, dim: torch.full_like(logits, 0.1),
    )
    with pytest.raises(FloatingPointError, match="do not sum to one"):
        reasoning_eval._sample_next(
            torch.tensor([[0.0, 1.0, 2.0]]),
            True,
            temperature=0.6,
            top_p=0.95,
        )


def test_zeroshot_compression_accounting_fails_closed_on_representation_mixing():
    exact = {"mode": "exact_cache_bit_counters", "numerator": 400, "denominator": 100}
    fallback = {
        "mode": "token_weighted_cache_ratio",
        "numerator": 10.0,
        "denominator": 5.0,
    }
    with pytest.raises(RuntimeError, match="changed representation"):
        _aggregate_compression_accounting([exact, fallback])

    partial = type("Partial", (), {"fp16_bits": 100, "stored_bits": 50.0})()
    with pytest.raises(ValueError, match="non-integer"):
        _cache_compression_accounting(partial, reference_tokens=8)


def test_ppl_rejects_gapped_or_empty_window_settings():
    with pytest.raises(ValueError, match="stride"):
        perplexity_over_text(None, None, "x", max_length=4, stride=5)
    with pytest.raises(ValueError, match="max_windows"):
        perplexity_over_text(None, None, "x", max_windows=0)


def test_wikitext_loader_uses_only_pinned_salesforce_source_and_join_hash(monkeypatch):
    expected_text = "first\n\nsecond"
    expected_hash = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    calls = []

    def load_dataset(repo, config, *, split, revision):
        calls.append((repo, config, split, revision))
        return {"text": ["first", "second"]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(ppl_eval, "WIKITEXT2_TEXT_SHA256", expected_hash)
    assert ppl_eval._load_wikitext2_text() == expected_text
    assert calls == [
        (
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            "test",
            ppl_eval.WIKITEXT2_REVISION,
        )
    ]


def test_ppl_preserves_unrounded_fp64_nll_numerator_across_windows():
    class Tok:
        def __call__(self, _text, **_kwargs):
            return type(
                "Encoded", (), {"input_ids": torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long)}
            )()

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, **_kwargs):
            vocab = 4
            logits = torch.full((*input_ids.shape, vocab), -1.25, dtype=torch.float32)
            next_ids = (input_ids + 1) % vocab
            logits.scatter_(-1, next_ids.unsqueeze(-1), 2.75)
            return type("Output", (), {"logits": logits})()

    result = perplexity_over_text(Model(), Tok(), "ignored", max_length=4, stride=2)
    logits = torch.full((4, 4), -1.25, dtype=torch.float32)
    logits.scatter_(1, torch.tensor([[1], [2], [3], [0]]), 2.75)
    expected_losses = torch.nn.functional.cross_entropy(
        logits, torch.tensor([1, 2, 3, 0]), reduction="none"
    )
    expected_sum = float(expected_losses.to(torch.float64).sum())
    assert result["n_tokens"] == 4
    assert result["n_windows"] == 2
    assert result["complete_text_eval"] is True
    assert result["nll_sum"] == expected_sum
    assert result["nll"] == expected_sum / 4
    assert result["ppl"] == pytest.approx(math.exp(expected_sum / 4), rel=1e-12)


def test_ppl_aggregates_compression_from_total_bits_not_mean_window_ratio():
    class Tok:
        def __call__(self, _text, **_kwargs):
            return type(
                "Encoded",
                (),
                {"input_ids": torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long)},
            )()

    bit_pairs = iter(((400, 100), (300, 150)))

    class Cache:
        fp16_bits = 0
        stored_bits = 0

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, *, past_key_values, **_kwargs):
            past_key_values.fp16_bits, past_key_values.stored_bits = next(bit_pairs)
            logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
            return type("Output", (), {"logits": logits})()

    result = perplexity_over_text(
        Model(), Tok(), "ignored", cache_factory=Cache, max_length=4, stride=2
    )
    assert result["compression_ratio"] == pytest.approx(700 / 250)
    assert result["compression_ratio"] != pytest.approx((4.0 + 2.0) / 2.0)
    assert result["compression_aggregation"] == "exact_cache_bit_counters"
    assert result["compression_fp16_bits"] == 700
    assert result["compression_stored_bits"] == 250


def test_teacher_forcing_dependency_contract_separates_formal_hq_tq_from_global_stats():
    # Sequence-axis scales are temporally coupled even though attention itself
    # remains causally masked.
    assert KIVICodec().teacher_forced_temporal_dependency() == ("sequence_axis_group_statistics")
    assert NVFP4Codec().teacher_forced_temporal_dependency() == ("sequence_axis_group_statistics")

    # Formal TurboQuant/HayateQuant recipes have token-local scales and no
    # dynamic outlier/adaptive statistic, so the original one-forward metric is
    # both causal and fast.
    assert TurboQuantCodec(bits=3).teacher_forced_temporal_dependency() is None
    assert all(
        HayateQuantCodec(**hayate_frontier_kwargs(rate)).teacher_forced_temporal_dependency()
        is None
        for rate in (2, 3, 4)
    )

    assert (
        TurboQuantCodec(bits=3, n_outlier=1).teacher_forced_temporal_dependency()
        == "dynamic_outlier_channels_over_update_tokens"
    )
    assert (
        HayateQuantCodec(bits=3, n_outlier=1).teacher_forced_temporal_dependency()
        == "dynamic_outlier_channels_over_update_tokens"
    )
    assert (
        HayateQuantCodec(
            avg_bits=3.0,
            block_size=2,
            adaptive_bit_split="online_stats",
        ).teacher_forced_temporal_dependency()
        == "cache_local_causal_bit_allocation_statistics"
    )
    assert HALOCodec().teacher_forced_temporal_dependency() is None
    assert (
        HALOCodec(center=True, precond_roles=("v",)).teacher_forced_temporal_dependency()
        == "block_global_preconditioner_statistics"
    )

    token_local = Int2Codec(key_axis="token", value_axis="token")
    assert (
        CompressedDynamicCache(
            codec=token_local, recent_tokens=4
        ).teacher_forced_temporal_dependency()
        == "recent_window_status_changes_over_update_tokens"
    )
    assert (
        CompressedDynamicCache(
            codec=token_local, recent_tokens=0
        ).teacher_forced_temporal_dependency()
        is None
    )
    assert (
        GradedQuantCache().teacher_forced_temporal_dependency()
        == "age_dependent_precision_schedule_over_update_tokens"
    )
    assert (
        ThinKVCache().teacher_forced_temporal_dependency()
        == "thin_kv_group_statistics_over_update_tokens"
    )
    assert (
        StreamingLLMEvictCache().teacher_forced_temporal_dependency()
        == "explicit_eviction_between_autoregressive_steps"
    )


def test_causal_teacher_forcing_blocks_future_codec_statistic_leakage():
    class Cache:
        def teacher_forced_temporal_dependency(self):
            return "test_block_global_statistic"

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.call_lengths = []

        def __call__(self, input_ids, **_kwargs):
            self.call_lengths.append(int(input_ids.shape[1]))
            # Stand in for a cache scale/mask fitted to the complete update.
            # In one full forward, the first logit therefore sees future IDs.
            update_stat = input_ids.float().mean()
            logits = torch.zeros((*input_ids.shape, 12), dtype=torch.float32)
            logits[..., 0] = update_stat
            return type("Output", (), {"logits": logits})()

    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 9, 8, 7]])
    model_a, model_b = Model(), Model()
    logits_a, protocol_a, dependency_a = causal_teacher_forced_logits(model_a, first, 1, Cache())
    logits_b, protocol_b, dependency_b = causal_teacher_forced_logits(
        model_b, changed_future, 1, Cache()
    )

    assert logits_a[0, 0, 0] == logits_b[0, 0, 0] == 1.0
    assert model_a.call_lengths == model_b.call_lengths == [1, 1, 1, 1]
    assert protocol_a == protocol_b == CAUSAL_INCREMENTAL_PROTOCOL
    assert dependency_a == dependency_b == "test_block_global_statistic"

    # This is the rejected old protocol: a later target changes the first
    # conditional logit before the causal attention mask can help.
    unsafe_a = Model()(first, past_key_values=Cache()).logits[0, 0, 0]
    unsafe_b = Model()(changed_future, past_key_values=Cache()).logits[0, 0, 0]
    assert unsafe_a != unsafe_b


def test_incremental_teacher_forcing_runs_eviction_lifecycle_with_absolute_positions():
    class Cache:
        def __init__(self):
            self.positions = []
            self.post_forwards = 0

        def teacher_forced_temporal_dependency(self):
            return "explicit_eviction_between_autoregressive_steps"

        def teacher_forced_forward_kwargs(self, positions):
            self.positions.append(positions.tolist())
            return {
                "cache_position": positions,
                "position_ids": positions.unsqueeze(0),
            }

        def teacher_forced_post_forward(self):
            self.post_forwards += 1

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.forward_positions = []

        def __call__(self, input_ids, **kwargs):
            assert kwargs["position_ids"].shape == input_ids.shape
            self.forward_positions.append(kwargs["cache_position"].tolist())
            logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            return type("Output", (), {"logits": logits})()

    cache = Cache()
    model = Model()
    logits, protocol, dependency = causal_teacher_forced_logits(
        model,
        torch.tensor([[1, 2, 3, 4]]),
        first_target_index=2,
        cache=cache,
    )
    assert logits.shape == (1, 2, 8)
    assert cache.positions == model.forward_positions == [[0, 1], [2], [3]]
    assert cache.post_forwards == 3
    assert protocol == CAUSAL_INCREMENTAL_PROTOCOL
    assert dependency == "explicit_eviction_between_autoregressive_steps"


def test_ppl_switches_only_temporally_coupled_caches_to_incremental_targets():
    class Tok:
        def __call__(self, _text, **_kwargs):
            return type(
                "Encoded",
                (),
                {"input_ids": torch.tensor([[0, 1, 2, 3, 0]], dtype=torch.long)},
            )()

    class Cache:
        fp16_bits = 100
        stored_bits = 50

        def __init__(self, dependency):
            self.dependency = dependency

        def teacher_forced_temporal_dependency(self):
            return self.dependency

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.call_lengths = []

        def __call__(self, input_ids, **_kwargs):
            self.call_lengths.append(int(input_ids.shape[1]))
            logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
            return type("Output", (), {"logits": logits})()

    incremental_model = Model()
    incremental = perplexity_over_text(
        incremental_model,
        Tok(),
        "ignored",
        cache_factory=lambda: Cache("sequence_axis_group_statistics"),
        max_length=5,
        stride=5,
    )
    assert incremental_model.call_lengths == [1, 1, 1, 1, 1]
    assert incremental["teacher_forced_scoring_protocol"] == CAUSAL_INCREMENTAL_PROTOCOL
    assert incremental["teacher_forced_temporal_dependency"] == ("sequence_axis_group_statistics")

    token_local_model = Model()
    token_local = perplexity_over_text(
        token_local_model,
        Tok(),
        "ignored",
        cache_factory=lambda: Cache(None),
        max_length=5,
        stride=5,
    )
    assert token_local_model.call_lengths == [5]
    assert token_local["teacher_forced_scoring_protocol"] == SINGLE_FORWARD_PROTOCOL
    assert token_local["teacher_forced_temporal_dependency"] is None


def test_h2o_respects_total_budget_including_sink_and_recent():
    scores = torch.rand(1, 2, 20)
    selector = H2OSelector(keep_ratio=0.25, recent_tokens=2, sink_tokens=2)
    mask = selector.select(
        scores, Budget(keep_tokens=5, sink_tokens=2, recent_tokens=2), seq_len=20
    )
    assert mask.sum(dim=-1).tolist() == [[5, 5]]
    assert mask[..., :2].all() and mask[..., -2:].all()


def test_kivi_channel_axis_has_nonzero_decode_error_after_residual_retires():
    codec = KIVICodec(bits=2, group_size=4, residual_window=1)
    cache = CompressedDynamicCache(codec=codec, sink_tokens=0, recent_tokens=1)
    keys = torch.arange(16, dtype=torch.float32).square().reshape(1, 1, 4, 4)
    values = keys.flip(-1)
    cache.update(keys, values, 0)
    new_key = torch.tensor([[[[17.0, 5.0, -2.0, 9.0]]]])
    new_value = new_key.flip(-1)
    out_k, _ = cache.update(new_key, new_value, 0)
    assert (out_k[:, :, :4] - keys).abs().max() > 0
    assert torch.equal(out_k[:, :, -1], new_key[:, :, 0])


def test_short_prefill_inside_recent_window_stays_full_precision():
    codec = Int2Codec(key_axis="token", value_axis="token", group_size=4)
    cache = CompressedDynamicCache(codec=codec, sink_tokens=1, recent_tokens=4)
    keys = torch.randn(1, 1, 4, 8)
    values = torch.randn_like(keys)
    out_k, out_v = cache.update(keys, values, 0)
    assert torch.equal(out_k, keys) and torch.equal(out_v, values)
    assert cache.compression_ratio() == 1.0


def test_large_decode_chunk_retires_previous_recent_window_once():
    codec = Int2Codec(key_axis="token", value_axis="token", group_size=4)
    cache = CompressedDynamicCache(codec=codec, sink_tokens=0, recent_tokens=2)
    first = torch.tensor([[[[0.1, 1.7, -2.3, 4.9], [0.2, 1.9, -2.7, 5.2]]]])
    cache.update(first, first, 0)
    assert torch.equal(cache.layers[0].keys, first)

    incoming = torch.randn(1, 1, 5, 4)
    cache.update(incoming, incoming, 0)
    # All old recent tokens retired because the appended chunk is larger than
    # the recent window. The last two incoming tokens alone remain exact.
    assert not torch.equal(cache.layers[0].keys[:, :, :2], first)
    assert torch.equal(cache.layers[0].keys[:, :, -2:], incoming[:, :, -2:])


def test_streaming_memory_tracks_generated_tokens():
    cache = StreamingLLMEvictCache(sink=1, recent=2)
    keys = torch.randn(1, 1, 10, 4)
    cache.update(keys, keys, 0)
    cache.evict_now()
    cache.update(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), 0)
    cache.evict_now()
    assert cache.kept == 3
    assert cache.total == 11
    assert cache.mem_fraction() == pytest.approx(3 / 11)


def test_thinkv_lowbit_keys_are_grouped_per_channel():
    # Every token is constant across head_dim, so a per-token quantizer would be
    # exact. Per-channel grouping across the varying sequence must introduce error.
    key = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1).expand(-1, -1, -1, 4)
    value = torch.randn_like(key)
    codec = NVFP4Codec(group_size=16)
    packed = codec.encode(key, value)
    key_q, _ = codec.decode(packed)
    assert (key_q - key).abs().max() > 0
    expected_groups = 4 + 16  # key: 4 channels; value: 16 token vectors
    assert packed.numel_bits() == (key.numel() + value.numel()) * 4 + expected_groups * 8


def test_thinkv_cache_uses_policy_group_size_and_exact_metadata_accounting():
    policy = ThinKVPolicy(group_size=4, precision={"R": 4, "E": 4, "T": 2})
    cache = ThinKVCache(policy=policy, sink_tokens=0)
    key = torch.randn(1, 1, 5, 8)
    value = torch.randn_like(key)
    cache.update(key, value, 0)
    expected = NVFP4Codec(group_size=4).encode(key, value).numel_bits()
    assert cache._codecs[4].group_size == 4
    assert cache._states[0].committed_bits == expected
