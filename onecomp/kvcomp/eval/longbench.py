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

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

# Official LongBench English prompt formats (THUDM/LongBench config/dataset2prompt).
_PROMPT = {
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official max generation length per task (config/dataset2maxlen).
_MAXLEN = {
    "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32, "2wikimqa": 32,
    "gov_report": 512, "multi_news": 512, "trec": 64, "triviaqa": 32,
    "samsum": 128, "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}

# All 13 English LongBench-E tasks.
LONGBENCH_E_TASKS = list(_MAXLEN.keys())


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
    pred = pred.strip()
    em = [c for c in (all_classes or []) if c in pred]
    # Preserve the official evaluator's exact class disambiguation behaviour.
    for match in list(em):
        if match in gold and match != gold:
            em.remove(match)
    return (1.0 / len(em)) if gold in em else 0.0


def retrieval_score(pred: str, gold: str) -> float:
    pattern = r"Paragraph (\d+)"
    m = re.search(pattern, gold)
    target = m.group(1) if m else gold.strip()
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
    return sum(n == str(gold).strip() for n in nums) / len(nums)


def code_sim_score(pred: str, gold: str) -> float:
    from fuzzywuzzy import fuzz  # same package/function as THUDM/LongBench

    line = ""
    for candidate in pred.lstrip("\n").splitlines():
        if "`" not in candidate and "#" not in candidate and "//" not in candidate:
            line = candidate
            break
    return fuzz.ratio(line, gold) / 100.0


_METRIC = {
    "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score, "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score, "triviaqa": qa_f1_score,
    "gov_report": rouge_score, "multi_news": rouge_score, "samsum": rouge_score,
    "trec": classification_score, "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
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
    score: float                                  # macro-average over tasks (x100)
    per_task: dict = field(default_factory=dict)  # task -> score (x100)
    by_length: dict = field(default_factory=dict)  # bucket -> score (x100)
    by_length_n: dict = field(default_factory=dict)
    mem_fraction: float = 1.0
    limit_per_task: Optional[int] = None
    tasks: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n, "score": round(self.score, 2),
            "per_task": {k: round(v, 2) for k, v in self.per_task.items()},
            "by_length": {k: round(v, 2) for k, v in self.by_length.items()},
            "by_length_n": dict(self.by_length_n),
            "mem_fraction": round(self.mem_fraction, 4),
            "limit_per_task": self.limit_per_task,
            "tasks": list(self.tasks),
            "official_full_eval": (
                self.limit_per_task is None and set(self.tasks) == set(LONGBENCH_E_TASKS)
            ),
        }


def _bucket(original_length: int) -> str:
    """Official LongBench-E bucket from the row's language-normalised length."""
    if original_length < 4000:
        return "0-4k"
    if original_length < 8000:
        return "4-8k"
    return "8k+"


def _load_task(task: str) -> list:
    """Load a LongBench-E task as a list of dict rows.

    ``datasets`` 4.x dropped script-based datasets (THUDM/LongBench ships a
    ``LongBench.py`` loader), so we read the official ``data.zip`` jsonl directly
    via ``huggingface_hub`` (cached after the first download).
    """
    import json
    import zipfile
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip",
                           repo_type="dataset")
    rows = []
    with zipfile.ZipFile(path) as z:
        with z.open(f"data/{task}_e.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _prompt_ids(tokenizer, prompt: str, task: str,
                max_length: Optional[int] = None) -> torch.Tensor:
    """Middle-truncate raw prompt first, then apply the official chat split.

    LongBench truncates the unwrapped prompt and only then adds the chat
    template. Truncating after wrapping consumes part of the scientific context
    budget with system/control tokens and is not leaderboard-equivalent.
    """
    raw_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
    if max_length is not None and raw_ids.shape[1] > max_length:
        head = max_length // 2
        tail = max_length - head
        prompt = tokenizer.decode(
            torch.cat([raw_ids[0, :head], raw_ids[0, -tail:]]),
            skip_special_tokens=True,
        )
    if task not in _NO_CHAT_TASKS:
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
            return encoded["input_ids"]
        except Exception:
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                return encoded["input_ids"]
            except Exception:
                pass
    return tokenizer(prompt, return_tensors="pt", truncation=False).input_ids


@torch.no_grad()
def longbench_e(
    model, tokenizer,
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
    from .longcontext import greedy_answer

    tasks = list(tasks or LONGBENCH_E_TASKS)
    unknown = set(tasks) - set(LONGBENCH_E_TASKS)
    if unknown:
        raise ValueError(f"Unknown LongBench-E task(s): {sorted(unknown)}")
    if limit_per_task is not None and limit_per_task <= 0:
        raise ValueError("limit_per_task must be positive or None")
    eos = tokenizer.eos_token_id
    per_task: dict = {}
    bucket_task_scores: dict = {"0-4k": [], "4-8k": [], "8k+": []}
    bucket_counts: dict = {"0-4k": 0, "4-8k": 0, "8k+": 0}
    mem: List[float] = []
    n_total = 0

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
        task_buckets: dict = {"0-4k": [], "4-8k": [], "8k+": []}
        rows = ds if limit_per_task is None else ds[:limit_per_task]
        for row in rows:
            prompt = prompt_fmt.format(context=row["context"], input=row.get("input", ""))
            ids = _prompt_ids(tokenizer, prompt, task, max_length=max_length)
            cache = cache_factory() if cache_factory is not None else DynamicCache()
            pp = (lambda c: c.evict_now()) if evict else None
            gen = greedy_answer(model, ids, cache, max_new_tokens=max_gen,
                                post_prefill=pp, eos=eos, attn_capture=attn_capture)
            pred = _postprocess_prediction(
                task, tokenizer.decode(gen[0], skip_special_tokens=True)
            )
            golds = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            s = max(metric(pred, g, **({"all_classes": row.get("all_classes")}
                                       if metric is classification_score else {}))
                    for g in golds) if golds else 0.0
            scores.append(s * 100.0)
            if "length" not in row:
                raise ValueError(f"LongBench-E row for {task!r} is missing official length")
            b = _bucket(int(row["length"]))
            task_buckets[b].append(s * 100.0)
            bucket_counts[b] += 1
            n_total += 1
            if hasattr(cache, "mem_fraction"):
                mem.append(cache.mem_fraction())
            elif hasattr(cache, "compression_ratio"):
                mem.append(1.0 / max(1e-9, cache.compression_ratio()))
        if scores:
            per_task[task] = sum(scores) / len(scores)
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
    return LongBenchResult(
        n=n_total, score=macro, per_task=per_task, by_length=by_len,
        by_length_n=bucket_counts,
        mem_fraction=(sum(mem) / len(mem) if mem else 1.0),
        limit_per_task=limit_per_task, tasks=tasks)
