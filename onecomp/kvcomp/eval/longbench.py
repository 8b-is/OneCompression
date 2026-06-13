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

The metrics are re-implemented with the standard library (token-F1, LCS ROUGE-L,
``difflib`` code similarity) so no ``rouge``/``fuzzywuzzy``/``jieba`` dependency
is needed; they match the official definitions on English text.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

# Official LongBench English prompt formats (THUDM/LongBench config/dataset2prompt).
_PROMPT = {
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nQuestion: {input}\n\nAnswer:",
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
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(pt), ns / len(gt)
    return 2 * prec * rec / (prec + rec)


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_score(pred: str, gold: str) -> float:
    """ROUGE-L F1 (LCS-based), the LongBench summary metric."""
    pt, gt = pred.split(), gold.split()
    if not pt or not gt:
        return 0.0
    lcs = _lcs(pt, gt)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(pt), lcs / len(gt)
    return 2 * prec * rec / (prec + rec)


def classification_score(pred: str, gold: str, all_classes=None) -> float:
    pred = pred.strip()
    em = [c for c in (all_classes or []) if c in pred]
    matched = [c for c in em if c == gold]
    return (len(matched) / len(em)) if em else (1.0 if gold.strip() in pred else 0.0)


def retrieval_score(pred: str, gold: str) -> float:
    m = re.search(r"Paragraph (\d+)", gold)
    target = m.group(1) if m else gold.strip()
    return 1.0 if re.search(rf"Paragraph {re.escape(target)}\b", pred) else 0.0


def count_score(pred: str, gold: str) -> float:
    nums = re.findall(r"\d+", pred)
    return 1.0 if (nums and nums[0] == str(gold).strip()) else 0.0


def code_sim_score(pred: str, gold: str) -> float:
    line = pred.lstrip("\n").split("\n", 1)[0]
    return SequenceMatcher(None, line, gold).ratio()


_METRIC = {
    "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score, "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score, "triviaqa": qa_f1_score,
    "gov_report": rouge_score, "multi_news": rouge_score, "samsum": rouge_score,
    "trec": classification_score, "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


@dataclass
class LongBenchResult:
    n: int
    score: float                                  # macro-average over tasks (x100)
    per_task: dict = field(default_factory=dict)  # task -> score (x100)
    by_length: dict = field(default_factory=dict)  # bucket -> score (x100)
    mem_fraction: float = 1.0

    def as_dict(self) -> dict:
        return {
            "n": self.n, "score": round(self.score, 2),
            "per_task": {k: round(v, 2) for k, v in self.per_task.items()},
            "by_length": {k: round(v, 2) for k, v in self.by_length.items()},
            "mem_fraction": round(self.mem_fraction, 4),
        }


def _bucket(n_tokens: int) -> str:
    if n_tokens < 4096:
        return "0-4k"
    if n_tokens < 8192:
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


@torch.no_grad()
def longbench_e(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    tasks: Optional[List[str]] = None,
    max_length: int = 7500,
    limit_per_task: int = 20,
    evict: bool = False,
    attn_capture: bool = False,
) -> LongBenchResult:
    """LongBench-E score under ``cache_factory`` (macro-avg over tasks, x100).

    Args:
        cache_factory: ``() -> Cache`` (``None`` = FullKV ``DynamicCache``).
        tasks: subset of :data:`LONGBENCH_E_TASKS` (default: all 13 English tasks).
        max_length: prompt budget; longer prompts are middle-truncated (keep
            head + tail) exactly as the official LongBench harness does.
        limit_per_task: samples per task (the -E split has ~150-300 each).
        evict: call ``cache.evict_now()`` after prefill (for eviction caches).
        attn_capture: materialise ``attn_weights`` via ``output_attentions=True``
            and feed them into a :class:`ThinKVCache`'s thought classifier.
            Requires ``attn_implementation='eager'`` and has an OOM risk on
            large models; disabled by default.
    """
    from transformers import DynamicCache
    from .longcontext import greedy_answer

    tasks = tasks or LONGBENCH_E_TASKS
    eos = tokenizer.eos_token_id
    per_task: dict = {}
    bucket_acc: dict = {"0-4k": [0.0, 0], "4-8k": [0.0, 0], "8k+": [0.0, 0]}
    mem: List[float] = []
    n_total = 0

    for task in tasks:
        try:
            ds = _load_task(task)
        except Exception as e:  # task missing / offline
            logger.warning("LongBench-E task %s unavailable: %s", task, e)
            continue
        metric = _METRIC[task]
        prompt_fmt = _PROMPT[task]
        max_gen = _MAXLEN[task]
        scores: List[float] = []
        for row in ds[:limit_per_task]:
            prompt = prompt_fmt.format(context=row["context"], input=row.get("input", ""))
            ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
            if ids.shape[1] > max_length:
                half = max_length // 2
                ids = torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
            n_tok = ids.shape[1]
            cache = cache_factory() if cache_factory is not None else DynamicCache()
            pp = (lambda c: c.evict_now()) if evict else None
            gen = greedy_answer(model, ids, cache, max_new_tokens=max_gen,
                                post_prefill=pp, eos=eos, attn_capture=attn_capture)
            pred = tokenizer.decode(gen[0], skip_special_tokens=True)
            golds = row["answers"] if isinstance(row["answers"], list) else [row["answers"]]
            s = max(metric(pred, g, **({"all_classes": row.get("all_classes")}
                                       if metric is classification_score else {}))
                    for g in golds) if golds else 0.0
            scores.append(s * 100.0)
            b = _bucket(n_tok)
            bucket_acc[b][0] += s * 100.0
            bucket_acc[b][1] += 1
            n_total += 1
            if hasattr(cache, "mem_fraction"):
                mem.append(cache.mem_fraction())
            elif hasattr(cache, "compression_ratio"):
                mem.append(1.0 / max(1e-9, cache.compression_ratio()))
        if scores:
            per_task[task] = sum(scores) / len(scores)
            print(f"    [LongBench-E] {task}: {per_task[task]:.2f} (n={len(scores)})", flush=True)

    macro = sum(per_task.values()) / max(1, len(per_task))
    by_len = {b: (acc / cnt if cnt else 0.0) for b, (acc, cnt) in bucket_acc.items()}
    return LongBenchResult(
        n=n_total, score=macro, per_task=per_task, by_length=by_len,
        mem_fraction=(sum(mem) / len(mem) if mem else 1.0))
