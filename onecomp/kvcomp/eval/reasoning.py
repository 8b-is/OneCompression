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

import math
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

_Z95 = 1.959963984540054

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
    pass_at_1_stderr: float = 0.0          # SE of the mean over questions
    pass_at_1_ci95: tuple = (0.0, 0.0)     # normal 95% interval
    eviction_fired: bool = True            # False if an eviction cache never trimmed

    def as_dict(self) -> dict:
        # Use "pass@k" as the metric key when k>1 (any-correct over k samples),
        # and "pass@1" for k=1 where both definitions coincide.  The _stderr
        # and _ci95 suffixes follow the same key so downstream parsers stay
        # consistent regardless of k.
        metric_key = f"pass@{self.k}"
        return {
            "task": self.task,
            "n": self.n,
            "k": self.k,
            metric_key: round(self.pass_at_1, 4),
            f"{metric_key}_stderr": round(self.pass_at_1_stderr, 4),
            f"{metric_key}_ci95": [round(self.pass_at_1_ci95[0], 4),
                                   round(self.pass_at_1_ci95[1], 4)],
            "avg_gen_len": round(self.avg_gen_len, 1),
            "compression_ratio": round(self.compression_ratio, 3),
            "eviction_fired": self.eviction_fired,
        }


def _mean_stderr_ci(values: List[float]):
    """Mean, standard error of the mean, and normal 95% CI of ``values``."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, (mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, (mean - _Z95 * se, mean + _Z95 * se)


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


def _sample_next(logits, do_sample, temperature, top_p):
    """Greedy or nucleus (top-p) sample of the next token from ``logits`` (1, V)."""
    if not do_sample:
        return logits.argmax(-1, keepdim=True)
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf - sorted_probs <= top_p  # keep tokens up to and incl. the one crossing top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
    choice = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, choice)


@torch.no_grad()
def _eviction_decode(model, input_ids, cache, max_new_tokens, eos,
                     do_sample, temperature, top_p):
    """Manual decode that fires ``cache.evict_now()`` after prefill AND each step.

    HF ``model.generate`` cannot trim a cache mid-forward (mask/position
    bookkeeping breaks), so an eviction cache passed to ``generate`` would *never*
    evict during a long reasoning rollout -- it would silently behave like FullKV
    (benchmark bug B-5).  This loop drives ``cache_position`` explicitly and
    re-applies eviction every step, so the StreamingLLM sink+recent window is
    maintained throughout the (long) generation, exactly as deployed.
    """
    dev = model.device
    n = input_ids.shape[1]
    pos = torch.arange(n, device=dev)
    out = model(input_ids.to(dev), past_key_values=cache, use_cache=True, cache_position=pos)
    cache.evict_now()
    nxt = _sample_next(out.logits[:, -1], do_sample, temperature, top_p)
    toks, cur = [nxt], n
    for _ in range(max_new_tokens - 1):
        cp = torch.tensor([cur], device=dev)
        out = model(nxt, past_key_values=cache, use_cache=True,
                    cache_position=cp, position_ids=cp.unsqueeze(0))
        cache.evict_now()
        nxt = _sample_next(out.logits[:, -1], do_sample, temperature, top_p)
        toks.append(nxt)
        cur += 1
        if eos is not None and nxt.item() == eos:
            break
    return torch.cat(toks, dim=1)[0]  # (T,) to match the generate path


@torch.no_grad()
def _generate(model, tokenizer, input_ids, cache_factory, max_new_tokens, do_sample,
              temperature, top_p, attn_capture=False):
    past = cache_factory() if cache_factory is not None else None
    # Eviction caches must be trimmed via evict_now(); HF generate never calls it,
    # so route them through the explicit eviction-aware loop (B-5).  This keeps the
    # StreamingLLM window enforced for the whole rollout instead of silently
    # degenerating to FullKV.
    if past is not None and hasattr(past, "evict_now"):
        eos = tokenizer.eos_token_id
        gen = _eviction_decode(model, input_ids, past, max_new_tokens, eos,
                               do_sample, temperature, top_p)
        ratio = past.compression_ratio() if hasattr(past, "compression_ratio") else 1.0
        fired = getattr(past, "kept", 0) and getattr(past, "total", 0) and past.kept < past.total
        return gen, ratio, bool(fired)
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
    return gen, ratio, True


@torch.no_grad()
def _eval_dataset(
    model, tokenizer, items, gold_fn, task, cache_factory,
    k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
    attn_capture=False,
):
    n = 0
    p1_values: List[float] = []
    gen_lens: List[int] = []
    ratios: List[float] = []
    per_q: List[dict] = []
    fired_all = True
    saw_eviction = False
    for question, gold_raw in items:
        gold = gold_fn(gold_raw)
        ids = _build_prompt(tokenizer, question, system)
        correct = 0
        any_correct = False
        for _ in range(k):
            gen, ratio, fired = _generate(
                model, tokenizer, ids, cache_factory,
                max_new_tokens, do_sample or k > 1, temperature, top_p,
                attn_capture=attn_capture,
            )
            gen_lens.append(int(gen.shape[0]))
            ratios.append(ratio)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(text)
            is_correct = pred is not None and gold is not None and pred == gold
            correct += int(is_correct)
            any_correct = any_correct or is_correct
        # pass@k: 1 if any of the k samples is correct (standard competition-math
        # metric used by ThinKV and PM-KVQ); stored alongside avg_sample_acc for
        # reference. For k=1 the two are identical.
        avg_sample_acc = correct / k
        pass_at_k = float(any_correct)
        # Metric reported: pass@k when k>1 (matches ThinKV/PM-KVQ protocol),
        # avg_sample_acc for k=1 (identical to pass@1 in that case).
        p1 = pass_at_k if k > 1 else avg_sample_acc
        p1_values.append(p1)
        n += 1
        per_q.append({"gold": gold, "pass@k": pass_at_k,
                      "avg_sample_acc": avg_sample_acc, "k": k})
        if verbose:
            logger.info("[%s] q%d gold=%s pass@%d=%.2f", task, n, gold, k, p1)
    # Detect whether an eviction cache actually trimmed (else its result is FullKV).
    probe = cache_factory() if cache_factory is not None else None
    if probe is not None and hasattr(probe, "evict_now"):
        saw_eviction = True
        fired_all = ratios and (sum(ratios) / len(ratios)) > 1.0 + 1e-6
        if not fired_all:
            logger.warning(
                "[%s] eviction cache did not trim (compression_ratio~1): the "
                "context is shorter than sink+recent, so this row equals FullKV.",
                task,
            )
    mean_p1, se, ci = _mean_stderr_ci(p1_values)
    return ReasoningResult(
        task=task, n=n, k=k,
        pass_at_1=mean_p1,
        avg_gen_len=sum(gen_lens) / max(len(gen_lens), 1),
        compression_ratio=float(sum(ratios) / len(ratios)) if ratios else 1.0,
        per_question=per_q,
        pass_at_1_stderr=se,
        pass_at_1_ci95=ci,
        eviction_fired=(fired_all if saw_eviction else True),
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

    # "gsm8k" (legacy script id) raises HfUriError on datasets >= 4; the canonical
    # repo id is "openai/gsm8k".
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception:  # noqa: BLE001 - legacy fallback
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


def aime_pass_at_1(
    model, tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    year: int = 2024,
    limit: int = 30,
    k: int = 8,
    max_new_tokens: int = 8192,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[str] = "Please reason step by step, and put your final answer within \\boxed{}.",
    verbose: bool = False,
    attn_capture: bool = False,
) -> ReasoningResult:
    """AIME pass@1 under ``cache_factory``.

    Evaluates on AIME competition problems.  Default settings follow the
    ThinKV paper (Ramachandran et al., ICLR 2026): k=8 independent samples per
    question, temperature=0.6, top_p=0.95, max_new_tokens=8192.  The PM-KVQ
    paper (arXiv:2505.18610) uses k=16.  AIME answers are integers in [0, 999].

    Datasets used:
      - year=2024: AI-MO/aimo-validation-aime (filtered to 2024 problems)
      - year=2025: yentinglin/aime_2025
    """
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    if year == 2024:
        ds_all = load_dataset("AI-MO/aimo-validation-aime", split="train")
        items = [
            (row["problem"], str(row["answer"]))
            for row in ds_all
            if "2024" in row.get("url", "")
        ]
    elif year == 2025:
        ds = load_dataset("yentinglin/aime_2025", split="train")
        items = [(row["problem"], str(row["answer"])) for row in ds]
    else:
        raise ValueError(f"Unsupported AIME year={year}; use 2024 or 2025")

    items = items[:limit]

    def _aime_gold(answer_str: str) -> Optional[str]:
        """AIME answers are integers 000-999; normalise to plain int string."""
        s = answer_str.strip().lstrip("0") or "0"
        try:
            return str(int(float(s)))
        except ValueError:
            return s

    return _eval_dataset(
        model, tokenizer, items, _aime_gold, f"aime{year}", cache_factory,
        k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
        attn_capture=attn_capture,
    )
