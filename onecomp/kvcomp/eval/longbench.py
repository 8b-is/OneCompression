"""LongBench-E evaluation for KV-cache compression (TurboQuant downstream metric).

LongBench-E (Bai et al., arXiv:2308.14508) is the end-to-end long-context
generation benchmark used by TurboQuant (arXiv:2504.19874, Table 1) to show a
KV-cache codec is "quality neutral".  The ``-E`` variant has a uniform length
distribution so scores can be bucketed by context length (0-4k / 4-8k / 8k+).

Like the rest of :mod:`onecomp.kvcomp.eval`, the harness takes a ``cache_factory``
``() -> Cache`` so the same code scores FullKV and any compressed cache, and it
reuses the explicit greedy loop in :mod:`.longcontext` (robust to eviction and
length-preserving graded caches alike).

Faithful to the official LongBench protocol:
* middle-truncation of the prompt to ``max_length`` (keep head + tail),
* the official per-task prompt format, ``max_gen`` and metric,
* English LongBench-E tasks only.

The metrics follow THUDM/LongBench's English evaluation definitions.  Length
buckets always use the dataset's original ``length`` field, never a model
tokenizer count after truncation.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from logging import getLogger
from typing import Callable, List, Optional

import torch

from .compression_accounting import (
    PER_EXAMPLE_CACHE_LIFETIME,
    PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION,
    aggregate_compression_sufficient_statistics,
    aggregate_mem_fraction,
    cache_compression_sufficient_statistics,
    cache_mem_fraction,
)

logger = getLogger(__name__)

# Official LongBench English prompt formats (THUDM/LongBench config/dataset2prompt).
_PROMPT = {
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official max generation length per task (config/dataset2maxlen).
_MAXLEN = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# All 13 English LongBench-E tasks.
LONGBENCH_E_TASKS = list(_MAXLEN.keys())
LONGBENCH_E_TASK_COUNTS = {
    "qasper": 224,
    "multifieldqa_en": 150,
    "hotpotqa": 300,
    "2wikimqa": 300,
    "gov_report": 300,
    "multi_news": 294,
    "trec": 300,
    "triviaqa": 300,
    "samsum": 300,
    "passage_count": 300,
    "passage_retrieval_en": 300,
    "lcc": 300,
    "repobench-p": 300,
}
LONGBENCH_E_TOTAL = sum(LONGBENCH_E_TASK_COUNTS.values())
LONGBENCH_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_PROTOCOL_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONGBENCH_PROTOCOL_SCOPE = (
    "release_exact_dataset_prompts_middle_truncation_task_generation_budgets_and_scorers; "
    "model_native_chat_template_adaptation_for_new_checkpoints"
)
LONGBENCH_CHAT_WRAPPER_SCOPE = (
    "Hayate model-native adaptation of official model-specific build_chat: "
    "tokenizer.apply_chat_template(add_generation_prompt=true; "
    "enable_thinking=false when supported)"
)


# --- official metric re-implementations (stdlib) -------------------------------
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1_score(pred: str, gold: str) -> float:
    pt, gt = _normalize(pred).split(), _normalize(gold).split()
    # THUDM's official f1_score returns zero when no tokens overlap, including
    # the empty/empty normalized edge case.
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def rouge_score(pred: str, gold: str) -> float:
    """ROUGE-L F1 from the same ``rouge`` package used by LongBench."""
    from rouge import Rouge  # dependency declared by onecomp

    try:
        return float(Rouge().get_scores([pred], [gold], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def classification_score(pred: str, gold: str, all_classes=None) -> float:
    em = [c for c in (all_classes or []) if c in pred]
    # Preserve even the released evaluator's in-place iteration semantics.
    # The pinned TREC class vocabulary has no substring pairs, so the quirk has
    # no effect there, but byte-for-byte algorithm parity avoids a hidden
    # divergence if a future official archive changes the class list.
    for match in em:
        if match in gold and match != gold:
            em.remove(match)
    return (1.0 / len(em)) if gold in em else 0.0


def retrieval_score(pred: str, gold: str) -> float:
    pattern = r"Paragraph (\d+)"
    # Released LongBench indexes the first match directly.  The pinned dataset
    # guarantees this shape; retaining the exact fail-closed behaviour prevents
    # a malformed reference from being assigned an invented fallback target.
    target = re.findall(pattern, gold)[0]
    # Official LongBench accepts any emitted integer, not only one preceded by
    # the word "Paragraph", and fractionally penalises multiple guesses.
    predicted = re.findall(r"\d+", pred)
    if not predicted:
        return 0.0
    return sum(p == target for p in predicted) / len(predicted)


def count_score(pred: str, gold: str) -> float:
    nums = re.findall(r"\d+", pred)
    if not nums:
        return 0.0
    return sum(n == str(gold) for n in nums) / len(nums)


def code_sim_score(pred: str, gold: str) -> float:
    from fuzzywuzzy import fuzz  # same package/function as THUDM/LongBench

    line = ""
    for candidate in pred.lstrip("\n").split("\n"):
        if "`" not in candidate and "#" not in candidate and "//" not in candidate:
            line = candidate
            break
    return fuzz.ratio(line, gold) / 100.0


_METRIC = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "triviaqa": qa_f1_score,
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "samsum": rouge_score,
    "trec": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Official LongBench prediction code deliberately leaves few-shot/classification
# and code-completion prompts raw; all other tasks use the model's chat wrapper.
_NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def _postprocess_prediction(task: str, prediction: str) -> str:
    """Apply LongBench's task-specific prediction post-processing exactly."""
    # The official evaluator scores only the first non-leading-newline line for
    # these short-answer / classification tasks.  Scoring the whole generation
    # can turn an otherwise correct first line into a false positive or negative
    # when the model continues with an explanation.
    if task in {"trec", "triviaqa", "samsum"}:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


@dataclass
class LongBenchResult:
    n: int
    score: float  # macro-average over tasks (x100)
    per_task: dict = field(default_factory=dict)  # task -> score (x100)
    per_task_n: dict = field(default_factory=dict)
    # Unrounded sums in percentage-point units.  Task-sharded paper runs use
    # these sufficient statistics rather than averaging two-decimal display
    # values, which can change the 13-task macro.
    per_task_score_sum: dict = field(default_factory=dict)
    by_length: dict = field(default_factory=dict)  # bucket -> score (x100)
    by_length_n: dict = field(default_factory=dict)
    mem_fraction: float = 1.0
    mem_fraction_aggregation: str = "fullkv_no_cache"
    limit_per_task: Optional[int] = None
    tasks: List[str] = field(default_factory=list)
    dataset_revision: str = LONGBENCH_DATASET_REVISION
    protocol_commit: str = LONGBENCH_PROTOCOL_COMMIT
    data_zip_sha256: Optional[str] = None
    max_length: Optional[int] = None
    task_dataset_sizes: dict = field(default_factory=dict)
    per_example: dict = field(default_factory=dict)
    truncated_examples: int = 0
    truncated_per_task: dict = field(default_factory=dict)
    generation_eos_token_ids: List[int] = field(default_factory=list)
    generation_eos_token_id_source: Optional[str] = None
    samsum_newline_token_id: Optional[int] = None
    prompt_protocol: dict = field(
        default_factory=lambda: {
            "raw_prompt": "official dataset2prompt formatting",
            "truncation": "official middle truncation before chat wrapping",
            "chat_wrapper": (LONGBENCH_CHAT_WRAPPER_SCOPE),
            "no_chat_tasks": sorted(_NO_CHAT_TASKS),
        }
    )
    per_task_compression_sufficient_statistics: dict = field(default_factory=dict)
    compression_sufficient_statistics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "score": round(self.score, 2),
            "per_task": {k: round(v, 2) for k, v in self.per_task.items()},
            "per_task_n": {k: int(v) for k, v in self.per_task_n.items()},
            "per_task_score_sum": {k: float(v) for k, v in self.per_task_score_sum.items()},
            "by_length": {k: round(v, 2) for k, v in self.by_length.items()},
            "by_length_n": dict(self.by_length_n),
            "mem_fraction": round(self.mem_fraction, 4),
            "mem_fraction_aggregation": self.mem_fraction_aggregation,
            "example_cache_lifetime": PER_EXAMPLE_CACHE_LIFETIME,
            "cross_iteration_tensor_retention": (
                PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
            ),
            "limit_per_task": self.limit_per_task,
            "tasks": list(self.tasks),
            "dataset_revision": self.dataset_revision,
            "protocol_commit": self.protocol_commit,
            "protocol_scope": LONGBENCH_PROTOCOL_SCOPE,
            "reference_model_wrapper_exact": False,
            "data_zip_sha256": self.data_zip_sha256,
            "max_length": self.max_length,
            "task_dataset_sizes": dict(self.task_dataset_sizes),
            "truncated_examples": int(self.truncated_examples),
            "truncated_per_task": {
                key: int(value) for key, value in self.truncated_per_task.items()
            },
            # LongBench-E's official evaluator reports a three-bucket dictionary
            # for each task.  The scalar and cross-task bucket means below are
            # explicit derived summaries, not additional official metrics.
            "score_protocol": "derived_unweighted_macro_over_task_scores",
            "by_length_protocol": ("derived_unweighted_macro_over_official_per_task_bucket_means"),
            "prompt_protocol": dict(self.prompt_protocol),
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "sampling_seed": None,
                "official_reference_global_seed": 42,
                "max_new_tokens": "official dataset2maxlen per task",
                "eos_token_ids": list(self.generation_eos_token_ids),
                "eos_token_id_source": self.generation_eos_token_id_source,
                "samsum_newline_token_id": self.samsum_newline_token_id,
                "stop_conditions": [
                    "any_generation_config_eos_token_id",
                    "max_new_tokens",
                    "samsum_newline_token_id_for_samsum_only",
                ],
            },
            "per_example": self.per_example,
            "per_task_compression_sufficient_statistics": (
                self.per_task_compression_sufficient_statistics
            ),
            "compression_sufficient_statistics": self.compression_sufficient_statistics,
            "released_data_complete": (
                self.limit_per_task is None
                and self.tasks == LONGBENCH_E_TASKS
                and self.n == LONGBENCH_E_TOTAL
                and self.per_task_n == LONGBENCH_E_TASK_COUNTS
            ),
        }


def _bucket(original_length: int) -> str:
    """Official LongBench-E bucket from the row's language-normalised length."""
    if original_length < 4000:
        return "0-4k"
    if original_length < 8000:
        return "4-8k"
    return "8k+"


@lru_cache(maxsize=1)
def _longbench_archive() -> tuple[str, str]:
    """Return the pinned LongBench archive and its content digest."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
        revision=LONGBENCH_DATASET_REVISION,
    )
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return path, digest.hexdigest()


def _load_task(task: str) -> list:
    """Load a LongBench-E task as a list of dict rows.

    ``datasets`` 4.x dropped script-based datasets (THUDM/LongBench ships a
    ``LongBench.py`` loader), so we read the official ``data.zip`` jsonl directly
    via ``huggingface_hub`` (cached after the first download).
    """
    import json
    import zipfile

    path, _ = _longbench_archive()
    rows = []
    with zipfile.ZipFile(path) as z:
        with z.open(f"data/{task}_e.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _prompt_ids(
    tokenizer,
    prompt: str,
    task: str,
    max_length: Optional[int] = None,
    return_metadata: bool = False,
):
    """Middle-truncate raw prompt first, then apply the official chat split.

    LongBench truncates the unwrapped prompt and only then adds the chat
    template. Truncating after wrapping consumes part of the scientific context
    budget with system/control tokens and is not leaderboard-equivalent.
    """
    raw_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    raw_prompt_tokens = int(raw_ids.shape[1])
    middle_truncated = max_length is not None and raw_prompt_tokens > max_length
    if middle_truncated:
        if max_length < 2:
            raise ValueError("LongBench max_length must be at least 2")
        head = max_length // 2
        # Match THUDM/LongBench's two independent decode calls exactly.  A
        # single decode over concatenated token IDs can introduce a different
        # space at the artificial head/tail join for some tokenizers.
        prompt = tokenizer.decode(raw_ids[0, :head], skip_special_tokens=True) + tokenizer.decode(
            raw_ids[0, -head:], skip_special_tokens=True
        )
    if task not in _NO_CHAT_TASKS:
        messages = [{"role": "user", "content": prompt}]
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            raise RuntimeError(
                f"LongBench-E task {task!r} requires the audited model chat "
                "wrapper, but this tokenizer has no apply_chat_template"
            )
        try:
            encoded = apply_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            # Older/non-thinking templates may not accept the Qwen-specific
            # option.  Only that signature mismatch is allowed to fall back;
            # template/rendering errors must invalidate the benchmark instead
            # of silently switching to raw text.
            message = str(exc).lower()
            unsupported_thinking_kwarg = "enable_thinking" in message and (
                "unexpected keyword" in message
                or "unexpected argument" in message
                or "keyword argument" in message
            )
            if not unsupported_thinking_kwarg:
                raise RuntimeError(f"LongBench-E chat template failed for task {task!r}") from exc
            try:
                encoded = apply_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except Exception as exc:
                raise RuntimeError(f"LongBench-E chat template failed for task {task!r}") from exc
        except Exception as exc:
            raise RuntimeError(f"LongBench-E chat template failed for task {task!r}") from exc
        ids = (
            encoded.get("input_ids")
            if isinstance(encoded, dict)
            else getattr(encoded, "input_ids", encoded)
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise TypeError(f"LongBench-E tokenizer returned invalid input_ids for task {task!r}")
    if not return_metadata:
        return ids
    return ids, {
        "raw_prompt_tokens": raw_prompt_tokens,
        "raw_prompt_token_budget": max_length,
        "middle_truncated": bool(middle_truncated),
        "chat_wrapper_used": task not in _NO_CHAT_TASKS,
        "model_input_tokens": int(ids.shape[1]),
    }


def _model_context_limit(model) -> int:
    """Return a fail-closed declared text context limit."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("LongBench-E model has no config with a declared context limit")
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    limit = getattr(config, "max_position_embeddings", None)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("LongBench-E requires a positive model.config.max_position_embeddings")
    return limit


@torch.no_grad()
def longbench_e(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    tasks: Optional[List[str]] = None,
    max_length: int = 32768,
    limit_per_task: Optional[int] = None,
    evict: bool = False,
    attn_capture: bool = False,
) -> LongBenchResult:
    """LongBench-E score under ``cache_factory`` (macro-avg over tasks, x100).

    Args:
        cache_factory: ``() -> Cache`` (``None`` = FullKV ``DynamicCache``).
        tasks: subset of :data:`LONGBENCH_E_TASKS` (default: all 13 English tasks).
        max_length: prompt budget; longer prompts are middle-truncated (keep
            head + tail) exactly as the official LongBench harness does.
        limit_per_task: samples per task. ``None`` runs the complete official
            split; a finite value is a smoke/debug subset and is marked as such.
        evict: call ``cache.evict_now()`` after prefill (for eviction caches).
        attn_capture: materialise ``attn_weights`` via ``output_attentions=True``
            and feed them into a :class:`ThinKVCache`'s thought classifier.
            Requires ``attn_implementation='eager'`` and has an OOM risk on
            large models; disabled by default.
    """
    from transformers import DynamicCache
    from .longcontext import effective_generation_eos_token_ids, greedy_answer

    tasks = list(LONGBENCH_E_TASKS if tasks is None else tasks)
    if not tasks:
        raise ValueError("At least one LongBench-E task is required")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Duplicate LongBench-E tasks: {tasks}")
    unknown = set(tasks) - set(LONGBENCH_E_TASKS)
    if unknown:
        raise ValueError(f"Unknown LongBench-E task(s): {sorted(unknown)}")
    # Keep the released task order stable even for caller-provided subsets.
    tasks = [task for task in LONGBENCH_E_TASKS if task in tasks]
    if limit_per_task is not None and (
        isinstance(limit_per_task, bool)
        or not isinstance(limit_per_task, int)
        or limit_per_task <= 0
    ):
        raise ValueError("limit_per_task must be a positive integer or None")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 2:
        raise ValueError("max_length must be an integer >= 2")
    if not isinstance(evict, bool) or not isinstance(attn_capture, bool):
        raise ValueError("evict and attn_capture must be bools")
    eos_token_ids, eos_token_id_source = effective_generation_eos_token_ids(model, tokenizer)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if (
        not isinstance(newline_ids, (list, tuple))
        or not newline_ids
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in newline_ids
        )
    ):
        raise ValueError("Tokenizer must encode the SamSum newline stop as non-empty token IDs")
    samsum_newline_token_id = newline_ids[-1]
    model_context_limit = _model_context_limit(model)
    per_task: dict = {}
    per_task_n: dict = {}
    per_task_score_sum: dict = {}
    bucket_task_scores: dict = {"0-4k": [], "4-8k": [], "8k+": []}
    bucket_counts: dict = {"0-4k": 0, "4-8k": 0, "8k+": 0}
    mem: List[float] = []
    n_total = 0
    task_dataset_sizes: dict = {}
    per_example: dict = {}
    truncated_per_task: dict = {}
    per_task_compression: dict = {}

    for task in tasks:
        try:
            ds = _load_task(task)
        except Exception as e:  # task missing / offline
            raise RuntimeError(
                f"LongBench-E task {task!r} is unavailable; refusing to report "
                "a silently incomplete macro-average"
            ) from e
        metric = _METRIC[task]
        prompt_fmt = _PROMPT[task]
        max_gen = _MAXLEN[task]
        scores: List[float] = []
        task_records: List[dict] = []
        task_buckets: dict = {"0-4k": [], "4-8k": [], "8k+": []}
        task_dataset_sizes[task] = len(ds)
        truncated_per_task[task] = 0
        expected_rows = LONGBENCH_E_TASK_COUNTS[task]
        if len(ds) != expected_rows:
            raise RuntimeError(
                f"Pinned LongBench-E task {task!r} has {len(ds)} rows; "
                f"expected {expected_rows}"
            )
        rows = ds if limit_per_task is None else ds[:limit_per_task]
        for row_index, row in enumerate(rows):
            prompt = prompt_fmt.format(context=row["context"], input=row.get("input", ""))
            ids, prompt_metadata = _prompt_ids(
                tokenizer,
                prompt,
                task,
                max_length=max_length,
                return_metadata=True,
            )
            truncated_per_task[task] += int(prompt_metadata["middle_truncated"])
            if int(ids.shape[1]) + max_gen > model_context_limit:
                raise RuntimeError(
                    f"LongBench-E {task} row {row_index} needs {ids.shape[1]} "
                    f"input + {max_gen} generation tokens, exceeding model limit "
                    f"{model_context_limit}; refusing implicit truncation"
                )
            cache = (
                cache_factory() if cache_factory is not None else DynamicCache(config=model.config)
            )
            pp = (lambda c: c.evict_now()) if evict else None
            stop_ids = list(eos_token_ids)
            if task == "samsum":
                if samsum_newline_token_id is not None and samsum_newline_token_id not in stop_ids:
                    stop_ids.append(samsum_newline_token_id)
            gen = greedy_answer(
                model,
                ids,
                cache,
                max_new_tokens=max_gen,
                post_prefill=pp,
                eos=stop_ids,
                attn_capture=attn_capture,
            )
            compression = cache_compression_sufficient_statistics(cache)
            generated_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            pred = _postprocess_prediction(task, generated_text)
            golds = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            if not golds or any(not isinstance(gold, str) or not gold for gold in golds):
                raise ValueError(
                    f"LongBench-E {task} row {row_index} has no valid reference answers"
                )
            s = max(
                metric(
                    pred,
                    gold,
                    **(
                        {"all_classes": row.get("all_classes")}
                        if metric is classification_score
                        else {}
                    ),
                )
                for gold in golds
            )
            if not math.isfinite(float(s)) or not 0.0 <= float(s) <= 1.0:
                raise RuntimeError(
                    f"LongBench-E {task} row {row_index} metric returned invalid score {s!r}"
                )
            scores.append(s * 100.0)
            if "length" not in row:
                raise ValueError(f"LongBench-E row for {task!r} is missing official length")
            b = _bucket(int(row["length"]))
            task_buckets[b].append(s * 100.0)
            bucket_counts[b] += 1
            n_total += 1
            fraction = cache_mem_fraction(cache)
            if fraction is not None:
                mem.append(fraction)
            task_records.append(
                {
                    "row_index": row_index,
                    "prediction": pred,
                    "answers": list(golds),
                    "all_classes": row.get("all_classes"),
                    "original_length": int(row["length"]),
                    "length_bucket": b,
                    "input_tokens": int(ids.shape[1]),
                    **prompt_metadata,
                    "generated_text": generated_text,
                    "generated_tokens": int(gen.shape[1]),
                    "max_new_tokens": max_gen,
                    "stop_token_ids": list(stop_ids),
                    "score": float(s),
                    "score_pct": float(s * 100.0),
                    **compression,
                }
            )
            # Per-example records are detached host values; release the cache
            # and generation tensors before constructing the next example's
            # cache to avoid RHS-before-rebind's transient two-cache peak.
            del cache, gen, ids
        if scores:
            per_task[task] = sum(scores) / len(scores)
            per_task_n[task] = len(scores)
            per_task_score_sum[task] = float(sum(scores))
            per_example[task] = task_records
            per_task_compression[task] = aggregate_compression_sufficient_statistics(task_records)
            for bucket, values in task_buckets.items():
                if values:
                    bucket_task_scores[bucket].append(sum(values) / len(values))
            print(f"    [LongBench-E] {task}: {per_task[task]:.2f} (n={len(scores)})", flush=True)
        else:
            raise RuntimeError(f"LongBench-E task {task!r} produced no scored rows")

    macro = sum(per_task.values()) / max(1, len(per_task))
    # Macro-average task-level bucket scores, matching LongBench's per-task
    # scorer_e outputs. Pooling rows would overweight tasks with more examples.
    by_len = {
        bucket: (sum(values) / len(values) if values else 0.0)
        for bucket, values in bucket_task_scores.items()
    }
    _, archive_sha256 = _longbench_archive()
    all_records = [record for task in tasks for record in per_example[task]]
    compression_statistics = aggregate_compression_sufficient_statistics(all_records)
    aggregate_fraction, fraction_aggregation = aggregate_mem_fraction(compression_statistics, mem)
    return LongBenchResult(
        n=n_total,
        score=macro,
        per_task=per_task,
        per_task_n=per_task_n,
        per_task_score_sum=per_task_score_sum,
        by_length=by_len,
        by_length_n=bucket_counts,
        mem_fraction=aggregate_fraction,
        mem_fraction_aggregation=fraction_aggregation,
        limit_per_task=limit_per_task,
        tasks=tasks,
        data_zip_sha256=archive_sha256,
        max_length=max_length,
        task_dataset_sizes=task_dataset_sizes,
        per_example=per_example,
        truncated_examples=sum(truncated_per_task.values()),
        truncated_per_task=truncated_per_task,
        generation_eos_token_ids=eos_token_ids,
        generation_eos_token_id_source=eos_token_id_source,
        samsum_newline_token_id=samsum_newline_token_id,
        per_task_compression_sufficient_statistics=per_task_compression,
        compression_sufficient_statistics=compression_statistics,
    )
