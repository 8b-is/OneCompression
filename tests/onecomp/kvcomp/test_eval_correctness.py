"""Regression tests for benchmark definitions and cache-accounting invariants."""

import pytest
import torch

from onecomp.kvcomp.cache.compressed_cache import CompressedDynamicCache
from onecomp.kvcomp.cache.streaming_evict import StreamingLLMEvictCache
from onecomp.kvcomp.cache.thinkv_cache import ThinKVCache
from onecomp.kvcomp.codec.affine import Int2Codec, KIVICodec
from onecomp.kvcomp.codec.lowbit_fp import NVFP4Codec
from onecomp.kvcomp.config import Budget
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
from onecomp.kvcomp.eval.reasoning import (
    ReasoningResult,
    extract_answer,
    math_equivalent,
)
from onecomp.kvcomp.eval.ppl import perplexity_over_text
from onecomp.kvcomp.eval.zeroshot import (
    ZeroShotResult,
    _encode_pair,
    _hellaswag,
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
        (r"\left(-\frac{5}{2},-\frac{5}{2}\right)",
         r"\boxed{(-2.5,-2.5)}"),
        ("12x-34", r"\boxed{-34+12x}"),
    ],
)
def test_math_grading_handles_structured_answers(gold, generated):
    assert math_equivalent(extract_answer(generated), gold)


def test_math_grading_does_not_match_first_number_in_gold():
    assert not math_equivalent(extract_answer("The final answer is 1"), r"\frac{1}{3}")


def test_reasoning_result_always_reports_mean_pass_at_1():
    result = ReasoningResult("aime2024", 1, 8, 0.125, 10, observed_pass_at_k=1.0)
    doc = result.as_dict()
    assert doc["pass@1"] == 0.125
    assert doc["observed_pass@8"] == 1.0
    assert "pass@8" not in doc


def test_longbench_official_fractional_count_and_retrieval_metrics():
    assert count_score("7 and 8", "7") == 0.5
    assert retrieval_score("Paragraph 7, then Paragraph 8", "Paragraph 7") == 0.5


def test_longbench_qa_empty_answer_edge_case():
    assert qa_f1_score("the", "an") == 0.0
    assert qa_f1_score("the", "answer") == 0.0


def test_longbench_full_eval_label_requires_all_tasks():
    partial = LongBenchResult(n=1, score=0.0, tasks=["qasper"])
    assert not partial.as_dict()["official_full_eval"]


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
        def __call__(self, *args, **kwargs):
            return Enc(torch.tensor([[1, 2, 3, 4, 5, 6]]))

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [1, 2, 5, 6]
            return "truncated"

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "truncated"
            return {"input_ids": torch.tensor([[9]])}

    assert _prompt_ids(Tok(), "raw", "qasper", max_length=4).tolist() == [[9]]


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


def test_hellaswag_matches_lm_eval_preprocessing():
    rows = [{
        "activity_label": "Making tea",
        "ctx_a": "A person boils water.",
        "ctx_b": "they add [title] leaves",
        "endings": [" [noise] and wait", "leave"],
        "label": "0",
    }]
    ctx, choices, gold = next(_hellaswag(rows))
    assert ctx == "Making tea: A person boils water. They add. leaves"
    assert choices[0] == " and wait"
    assert gold == 0


def test_winogrande_matches_lm_eval_multiple_input_protocol():
    row = {"sentence": "The trophy does not fit because _ is too big.",
           "option1": "it", "option2": "the case", "answer": "1"}
    contexts, continuations, gold = next(_winogrande([row]))
    assert contexts == ["The trophy does not fit because it",
                        "The trophy does not fit because the case"]
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


def test_zeroshot_primary_metric_uses_official_winogrande_acc():
    wino = ZeroShotResult("winogrande", 1, acc=1.0, acc_norm=0.0).as_dict()
    hella = ZeroShotResult("hellaswag", 1, acc=0.0, acc_norm=1.0).as_dict()
    assert (wino["primary_metric"], wino["primary_score"]) == ("acc", 1.0)
    assert (hella["primary_metric"], hella["primary_score"]) == ("acc_norm", 1.0)


def test_ppl_rejects_gapped_or_empty_window_settings():
    with pytest.raises(ValueError, match="stride"):
        perplexity_over_text(None, None, "x", max_length=4, stride=5)
    with pytest.raises(ValueError, match="max_windows"):
        perplexity_over_text(None, None, "x", max_windows=0)


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
