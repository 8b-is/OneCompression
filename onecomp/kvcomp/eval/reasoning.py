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
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from typing import Callable, List, Optional

import torch

logger = getLogger(__name__)

_Z95 = 1.959963984540054

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
    observed_pass_at_k: Optional[float] = None
    dataset_size: Optional[int] = None
    grader: Optional[str] = None
    protocol: Optional[str] = None

    def as_dict(self) -> dict:
        # ThinKV reports pass@1 as the mean correctness of k independent samples.
        # "At least one of k" is a different metric and is exposed under an
        # explicit diagnostic key so it cannot silently replace pass@1 again.
        out = {
            "task": self.task,
            "n": self.n,
            "k": self.k,
            "pass@1": round(self.pass_at_1, 4),
            "pass@1_stderr": round(self.pass_at_1_stderr, 4),
            "pass@1_ci95": [round(self.pass_at_1_ci95[0], 4),
                             round(self.pass_at_1_ci95[1], 4)],
            "avg_gen_len": round(self.avg_gen_len, 1),
            "compression_ratio": round(self.compression_ratio, 3),
            "eviction_fired": self.eviction_fired,
        }
        if self.k > 1 and self.observed_pass_at_k is not None:
            out[f"observed_pass@{self.k}"] = round(self.observed_pass_at_k, 4)
        if self.dataset_size is not None:
            out["dataset_size"] = int(self.dataset_size)
            out["complete_dataset_eval"] = self.n == self.dataset_size
        if self.grader is not None:
            out["grader"] = self.grader
        if self.protocol is not None:
            out["protocol"] = self.protocol
        return out


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
    return mean, se, (max(0.0, mean - _Z95 * se), min(1.0, mean + _Z95 * se))


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


def _last_boxed_content(text: str) -> Optional[str]:
    """Return the last balanced ``\\boxed{...}``/``\\fbox{...}`` payload.

    A regular expression cannot parse nested LaTeX braces (for example
    ``\\boxed{\\frac{1}{2}}``), which is common in MATH-500.
    """
    starts = [text.rfind("\\boxed"), text.rfind("\\fbox")]
    start = max(starts)
    if start < 0:
        return None
    left = text.find("{", start)
    if left < 0:
        return None
    depth = 0
    for i in range(left, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[left + 1:i].strip()
    return None


def _fix_simple_latex(s: str) -> str:
    """Canonicalise the answer forms used by the official Hendrycks MATH grader."""
    s = str(s).strip().replace("\n", "")
    for old, new in (("\\tfrac", "\\frac"), ("\\dfrac", "\\frac")):
        s = s.replace(old, new)
    for token in ("\\left", "\\right", "\\!", "$", " "):
        s = s.replace(token, "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\text\{(?:or|and)\}", ",", s)
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    # Strip a short variable assignment (x=..., k=...) without accepting an
    # arbitrary number merely because it occurs somewhere in a long answer.
    if s.count("=") == 1 and len(s.split("=", 1)[0]) <= 2:
        s = s.split("=", 1)[1]
    s = re.sub(r"\\sqrt(?!\{)(.)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac(?!\{)(.)(.)", r"\\frac{\1}{\2}", s)
    # Properly formatted thousands separators only; tuple commas are retained.
    while re.search(r"(?<=\d),(?=\d{3}(?:\D|$))", s):
        s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    # Units following an answer are not part of the mathematical value.
    s = re.sub(r"\\(?:text|mbox)\{[^{}]*\}$", "", s)
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def _latex_to_safe_expr(s: str) -> Optional[str]:
    """Convert a small, safe LaTeX arithmetic subset to a SymPy expression."""
    s = _fix_simple_latex(s)
    # Resolve balanced frac/sqrt constructs from the inside out.
    frac = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    root = re.compile(r"\\sqrt\{([^{}]+)\}")
    previous = None
    while previous != s:
        previous = s
        s = frac.sub(r"((\1)/(\2))", s)
        s = root.sub(r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    # SymPy does not enable implicit multiplication by default. Insert only the
    # unambiguous forms common in MATH answers after the grammar is constrained.
    s = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", s)
    s = re.sub(r"(?<=\))(?=[A-Za-z0-9(])", "*", s)
    # parse_expr/sympify are not sandboxes. Reject generated text outside a
    # deliberately tiny arithmetic grammar before invoking SymPy.
    if "__" in s or not re.fullmatch(r"[0-9A-Za-z_+\-*/().,=]*", s):
        return None
    names = re.findall(r"[A-Za-z_]+", s)
    if any(name not in {"sqrt", "pi", "oo"} and len(name) != 1 for name in names):
        return None
    return s


def math_equivalent(prediction: Optional[str], reference: Optional[str]) -> bool:
    """MATH answer equality with canonical-string, numeric and symbolic checks."""
    if prediction is None or reference is None:
        return False
    pred = _fix_simple_latex(prediction)
    ref = _fix_simple_latex(reference)
    if pred.lower() == ref.lower():
        return True
    # Ordered pairs, intervals, and comma-separated multi-answers are common in
    # MATH-500. SymPy tuples cannot be subtracted, so compare their components
    # recursively while preserving interval endpoint brackets.
    for left, right in (("(", ")"), ("[", "]")):
        if pred.startswith(left) and pred.endswith(right) and ref.startswith(left) and ref.endswith(right):
            pred_parts = pred[1:-1].split(",")
            ref_parts = ref[1:-1].split(",")
            if len(pred_parts) > 1 or len(ref_parts) > 1:
                return len(pred_parts) == len(ref_parts) and all(
                    math_equivalent(p, r) for p, r in zip(pred_parts, ref_parts)
                )
    if "," in pred and "," in ref:
        pred_parts, ref_parts = pred.split(","), ref.split(",")
        return len(pred_parts) == len(ref_parts) and all(
            math_equivalent(p, r) for p, r in zip(pred_parts, ref_parts)
        )
    pred_expr, ref_expr = _latex_to_safe_expr(pred), _latex_to_safe_expr(ref)
    if pred_expr is None or ref_expr is None:
        return False
    try:
        import sympy  # dependency of torch/lm-eval

        locals_ = {"sqrt": sympy.sqrt, "pi": sympy.pi, "oo": sympy.oo}
        # Single-letter variables are admitted explicitly after the grammar check.
        for name in set(re.findall(r"[A-Za-z]", pred_expr + ref_expr)):
            if name not in {"p", "i", "o"}:  # pi/oo were already tokenised above
                locals_[name] = sympy.Symbol(name)
        p = sympy.sympify(pred_expr, locals=locals_)
        r = sympy.sympify(ref_expr, locals=locals_)
        if bool(sympy.simplify(p - r) == 0):
            return True
        try:
            return math.isclose(float(sympy.N(p)), float(sympy.N(r)),
                                rel_tol=1e-4, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False
    except Exception:  # malformed/non-scalar expressions fall back to canonical exact match
        return False


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer (\\boxed{...}, ####, or last number)."""
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return boxed
    hashm = _HASH.findall(text)
    if hashm:
        return hashm[-1].strip()
    nums = _NUMBER.findall(text)
    if nums:
        return nums[-1].strip()
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


@contextmanager
def _seeded_rng(seed: Optional[int], device):
    """Isolate and deterministically seed CPU/CUDA sampling for one response."""
    if seed is None:
        yield
        return
    devices = []
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed(int(seed))
        yield


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
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
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
    attn_capture=False, answer_eq=None, sample_seed: Optional[int] = 0,
    dataset_size: Optional[int] = None,
    grader: Optional[str] = None, protocol: Optional[str] = None,
):
    n = 0
    p1_values: List[float] = []
    gen_lens: List[int] = []
    ratios: List[float] = []
    per_q: List[dict] = []
    fired_all = True
    saw_eviction = False
    pass_k_values: List[float] = []
    answer_eq = answer_eq or (
        lambda pred, gold: _normalize_number(pred) == _normalize_number(gold)
    )
    for question_idx, (question, gold_raw) in enumerate(items):
        gold = gold_fn(gold_raw)
        ids = _build_prompt(tokenizer, question, system)
        correct = 0
        any_correct = False
        sample_records = []
        for sample_idx in range(k):
            seed = None if sample_seed is None else int(sample_seed) + question_idx * 100_003 + sample_idx
            with _seeded_rng(seed, model.device):
                gen, ratio, fired = _generate(
                    model, tokenizer, ids, cache_factory,
                    max_new_tokens, do_sample or k > 1, temperature, top_p,
                    attn_capture=attn_capture,
                )
            gen_lens.append(int(gen.shape[0]))
            ratios.append(ratio)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(text)
            is_correct = pred is not None and gold is not None and answer_eq(pred, gold)
            correct += int(is_correct)
            any_correct = any_correct or is_correct
            sample_records.append({"seed": seed, "prediction": pred, "correct": is_correct})
        # ThinKV's reported pass@1 is the average correctness across k independent
        # responses. Keep observed any-correct only as a separately named diagnostic.
        p1 = correct / k
        pass_at_k = float(any_correct)
        p1_values.append(p1)
        pass_k_values.append(pass_at_k)
        n += 1
        per_q.append({"gold": gold, "pass@1": p1, f"observed_pass@{k}": pass_at_k,
                      "samples": sample_records, "k": k})
        if verbose:
            logger.info("[%s] q%d gold=%s pass@1=%.3f (k=%d)", task, n, gold, p1, k)
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
        observed_pass_at_k=(sum(pass_k_values) / len(pass_k_values) if pass_k_values else None),
        dataset_size=dataset_size,
        grader=grader,
        protocol=protocol,
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
    sample_seed: Optional[int] = 0,
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
    count = len(ds) if limit is None else min(limit, len(ds))
    items = [(ds[i]["question"], ds[i]["answer"]) for i in range(count)]
    return _eval_dataset(
        model, tokenizer, items, gsm8k_gold, "gsm8k", cache_factory,
        k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
        attn_capture=attn_capture, sample_seed=sample_seed, dataset_size=len(ds),
        grader="exact_numeric_answer", protocol="openai_gsm8k_test",
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
    sample_seed: Optional[int] = 0,
) -> ReasoningResult:
    """MATH-500 pass@1 under ``cache_factory`` (HuggingFaceH4/MATH-500).

    This evaluator uses a deterministic local canonical/numeric/SymPy grader.
    It intentionally records that provenance because it is not the dataset
    repository's optional four-epoch model-graded ``inspect-ai`` protocol.

    Set ``attn_capture=True`` for ThinKV (engages thought classification)."""
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if len(ds) != 500:
        raise RuntimeError(f"Expected the complete 500-row MATH-500 test set, got {len(ds)}")
    count = len(ds) if limit is None else min(limit, len(ds))
    items = [(ds[i]["problem"], ds[i]["answer"]) for i in range(count)]
    return _eval_dataset(
        model, tokenizer, items, lambda a: str(a).strip(), "math500", cache_factory,
        k, max_new_tokens, do_sample, temperature, top_p, system, verbose,
        attn_capture=attn_capture, answer_eq=math_equivalent, sample_seed=sample_seed,
        dataset_size=len(ds),
        grader="local_symbolic_equivalence",
        protocol="single_model_generation",
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
    sample_seed: Optional[int] = 0,
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

    if len(items) != 30:
        raise RuntimeError(
            f"Expected the complete 30-problem AIME {year} set, got {len(items)}; "
            "dataset schema/filter may have changed"
        )
    dataset_size = len(items)
    items = items if limit is None else items[:limit]

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
        attn_capture=attn_capture, sample_seed=sample_seed, dataset_size=dataset_size,
        grader="exact_integer_answer", protocol=f"aime{year}_competition_set",
    )
