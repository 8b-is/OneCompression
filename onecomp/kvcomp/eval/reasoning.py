"""Reasoning pass@1 evaluation (GSM8K / MATH-500) under KV compression.

This is the benchmark family used by the ThinKV paper: long chain-of-thought
generation from a reasoning model, scored by whether the extracted final answer
matches the gold answer.  ``pass@1`` is averaged over ``k`` independent samples
per question (k=1 -> greedy).

Each question is generated with a *fresh* cache from ``cache_factory`` so the
KV compression policy (e.g. ThinKV) drives the autoregressive decode exactly as
in deployment.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH = re.compile(r"####\s*(.+)")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class ReasoningResult:
    task: str
    n: int
    k: int
    pass_at_1: float
    avg_gen_len: float
    compression_ratio: float = 1.0
    per_question: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "n": self.n,
            "k": self.k,
            "pass@1": round(self.pass_at_1, 4),
            "avg_gen_len": round(self.avg_gen_len, 1),
            "compression_ratio": round(self.compression_ratio, 3),
        }


def _normalize_number(s: str) -> Optional[str]:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    m = _NUMBER.search(s)
    if not m:
        return s.strip() or None
    val = m.group(0).replace(",", "")
    try:
        f = float(val)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return val


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer (\\boxed{...}, ####, or last number)."""
    boxed = _BOXED.findall(text)
    if boxed:
        return _normalize_number(boxed[-1])
    hashm = _HASH.findall(text)
    if hashm:
        return _normalize_number(hashm[-1])
    nums = _NUMBER.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def gsm8k_gold(answer_field: str) -> Optional[str]:
    """GSM8K gold answers are after ``#### ``."""
    m = _HASH.findall(answer_field)
    return _normalize_number(m[-1]) if m else _normalize_number(answer_field)


def _build_prompt(tokenizer, question: str, system: Optional[str]) -> torch.Tensor:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    try:
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"]
    except Exception:  # tokenizer without chat template
        ids = tokenizer(question, return_tensors="pt").input_ids
    return ids


@torch.no_grad()
def _generate(model, tokenizer, input_ids, cache_factory, max_new_tokens, do_sample,
              temperature, top_p, attn_capture=False):
    past = cache_factory() if cache_factory is not None else None
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    if past is not None:
        gen_kwargs["past_key_values"] = past
        gen_kwargs["use_cache"] = True
    if attn_capture and past is not None:
        # ThinKV thought classification reads attention sparsity; materialize
        # attn_weights (eager + output_attentions) and feed them into the cache.
        from ..cache import thinkv_attention_capture  # local import
        gen_kwargs["output_attentions"] = True
        with thinkv_attention_capture(model, past):
            out = model.generate(input_ids.to(model.device), **gen_kwargs)
    else:
        out = model.generate(input_ids.to(model.device), **gen_kwargs)
    gen = out[0, input_ids.shape[1]:]
    ratio = past.compression_ratio() if (past is not None and hasattr(past, "compression_ratio")) else 1.0
    return gen, ratio


@torch.no_grad()
def _eval_dataset(
    model, tokenizer, items, gold_fn, task, cache_factory,
    k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
    attn_capture=False,
):
    n = 0
    pass_sum = 0.0
    gen_lens: List[int] = []
    ratios: List[float] = []
    per_q: List[dict] = []
    for question, gold_raw in items:
        gold = gold_fn(gold_raw)
        ids = _build_prompt(tokenizer, question, system)
        correct = 0
        for _ in range(k):
            gen, ratio = _generate(
                model, tokenizer, ids, cache_factory,
                max_new_tokens, do_sample or k > 1, temperature, top_p,
                attn_capture=attn_capture,
            )
            gen_lens.append(int(gen.shape[0]))
            ratios.append(ratio)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(text)
            correct += int(pred is not None and gold is not None and pred == gold)
        p1 = correct / k
        pass_sum += p1
        n += 1
        per_q.append({"gold": gold, "pass@1": p1})
        if verbose:
            logger.info("[%s] q%d gold=%s pass@1=%.2f", task, n, gold, p1)
    return ReasoningResult(
        task=task, n=n, k=k,
        pass_at_1=pass_sum / max(n, 1),
        avg_gen_len=sum(gen_lens) / max(len(gen_lens), 1),
        compression_ratio=float(sum(ratios) / len(ratios)) if ratios else 1.0,
        per_question=per_q,
    )


def gsm8k_pass_at_1(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    limit: int = 50,
    k: int = 1,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[str] = "Please reason step by step, and put your final answer within \\boxed{}.",
    verbose: bool = False,
    attn_capture: bool = False,
) -> ReasoningResult:
    """GSM8K (main, test) pass@1 under ``cache_factory``.

    Set ``attn_capture=True`` for ThinKV (engages attention-sparsity thought
    classification; requires the model loaded with ``attn_implementation='eager'``).
    """
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset("gsm8k", "main", split="test")
    items = [(ds[i]["question"], ds[i]["answer"]) for i in range(min(limit, len(ds)))]
    return _eval_dataset(
        model, tokenizer, items, gsm8k_gold, "gsm8k", cache_factory,
        k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
        attn_capture=attn_capture,
    )


def math500_pass_at_1(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    limit: int = 50,
    k: int = 1,
    max_new_tokens: int = 3072,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[str] = "Please reason step by step, and put your final answer within \\boxed{}.",
    verbose: bool = False,
    attn_capture: bool = False,
) -> ReasoningResult:
    """MATH-500 pass@1 under ``cache_factory`` (HuggingFaceH4/MATH-500).

    Set ``attn_capture=True`` for ThinKV (engages thought classification)."""
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = [(ds[i]["problem"], ds[i]["answer"]) for i in range(min(limit, len(ds)))]
    return _eval_dataset(
        model, tokenizer, items, lambda a: _normalize_number(str(a)), "math500", cache_factory,
        k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
        attn_capture=attn_capture,
    )
