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
import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as package_version
from logging import getLogger
from typing import Callable, List, Optional

import torch

from .compression_accounting import (
    aggregate_compression_sufficient_statistics,
    cache_compression_sufficient_statistics,
)

logger = getLogger(__name__)

_Z95 = 1.959963984540054

_HASH = re.compile(r"####\s*(.+)")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_MATH_VERIFY_VERSION = "0.9.0"
_MATH500_DATASET = "HuggingFaceH4/MATH-500"
_MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
_MATH_VERIFY_TASK_SOURCE_COMMIT = "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b"
_MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION = "0.13.0"
_MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT = "7c1cd62716b0a198a630c26d781430c54726cd02"
_MATH500_PROTOCOL_ID = (
    "math500_adapted_composite_math_verify_ba3d3aa_" "lighteval_prompt_manager_7c1cd62_v1"
)
_MATH500_QUERY_TEMPLATE = "Question: {problem}\nStep-by-Step Answer:"
_MATH500_FEWSHOT_ORDER = (4, 3, 2, 1)
# Vendored composite: Math-Verify @ ba3d3aa supplies the custom task and
# demonstrations; we apply the LightEval v0.13.0 @ 7c1cd62 prompt-manager's
# balanced sampler/rendering rules manually.  With seed 0, math_500's test-split
# demonstration indices are 4,3,2,1 in one-based MATH_HARD_FEW_SHOTS order.
# These strings are the exact plain rendering: query + " " + answer.strip().
_MATH500_FEWSHOT_DEMONSTRATIONS = (
    r"""Question: If the system of equations

\begin{align*}
6x-4y&=a,\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.
Step-by-Step Answer: If we multiply the first equation by $-\frac{3}{2}$, we obtain

$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
The final answer is $-\frac{2}{3}$. I hope it is correct.""",
    r"""Question: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Step-by-Step Answer: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
The final answer is $16$. I hope it is correct.""",
    r"""Question: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$
Step-by-Step Answer: We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$
The final answer is $24$. I hope it is correct.""",
    r"""Question: Find the domain of the expression  $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.
Step-by-Step Answer: The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\boxed{[2,5)}$.
The final answer is $[2,5)$. I hope it is correct.""",
)
_MATH500_FEWSHOT_PREFIX = "\n\n".join(_MATH500_FEWSHOT_DEMONSTRATIONS)
_MATH500_FEWSHOT_CONTENT_SHA256 = (
    "1360154eb365a22beb1a201dd6ff159e829b6d64d59f69df29530700034c07a4"
)
if hashlib.sha256(_MATH500_FEWSHOT_PREFIX.encode("utf-8")).hexdigest() != (
    _MATH500_FEWSHOT_CONTENT_SHA256
):  # pragma: no cover - import-time immutable fixture guard
    raise RuntimeError("pinned Math-Verify four-shot prompt fixture changed")
_MATH500_MAX_NEW_TOKENS = 1024
_MATH500_STOP_SEQUENCES = (
    "\nQuestion:",
    "\nProblem:",
    "\nquestion:",
    "\nproblem:",
)
_MATH500_PROTOCOL_SCOPE = (
    "adapted_composite_math_verify_ba3d3aa_lighteval_prompt_manager_7c1cd62_"
    "four_shot_query_generation_"
    "prediction_extraction_and_verify; "
    "corrected_inline_math_delimiters_for_bare_gold_answer_fragments"
)
_MATH500_GOLD_SERIALIZATION = "wrap_bare_answer_as_inline_math_$...$"
_MATH500_GOLD_ADAPTATION_REASON = (
    "the pinned reference task passes bare dataset answer fragments to the LaTeX "
    "extractor; explicit inline-math delimiters make those gold fragments parseable"
)
_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
_AIME2024_REVISION = "13f9e12f613e720c2a2b2f345dd04b998a29494d"
_AIME2025_REVISION = "6f71d77b0b89b9dabe07ab466c51df33f514df7f"


@dataclass
class ReasoningResult:
    task: str
    n: int
    k: int
    pass_at_1: float
    avg_gen_len: float
    compression_ratio: float = 1.0
    compression_ratio_aggregation: str = "fullkv_no_cache"
    per_question: List[dict] = field(default_factory=list)
    pass_at_1_stderr: float = 0.0  # SE of the mean over questions
    pass_at_1_ci95: tuple = (0.0, 0.0)  # normal 95% interval
    eviction_fired: bool = True  # False if an eviction cache never trimmed
    observed_pass_at_k: Optional[float] = None
    dataset_size: Optional[int] = None
    grader: Optional[str] = None
    grader_version: Optional[str] = None
    grader_config: Optional[dict] = None
    protocol: Optional[str] = None
    prompt_config: Optional[dict] = None
    generation_config: Optional[dict] = None
    start: int = 0
    limit: Optional[int] = None
    # Unrounded sufficient statistic for exact shard aggregation.  For k=1 it
    # is also the integer number of correct questions.
    pass_at_1_sum: Optional[float] = None
    compression_sufficient_statistics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        # ThinKV reports pass@1 as the mean correctness of k independent samples.
        # "At least one of k" is a different metric and is exposed under an
        # explicit diagnostic key so it cannot silently replace pass@1 again.
        pass_at_1_sum = (
            float(self.pass_at_1_sum)
            if self.pass_at_1_sum is not None
            else float(self.pass_at_1 * self.n)
        )
        out = {
            "task": self.task,
            "n": self.n,
            "k": self.k,
            "pass@1": round(self.pass_at_1, 4),
            "pass@1_stderr": round(self.pass_at_1_stderr, 4),
            "pass@1_ci95": [round(self.pass_at_1_ci95[0], 4), round(self.pass_at_1_ci95[1], 4)],
            "avg_gen_len": round(self.avg_gen_len, 1),
            "compression_ratio": round(self.compression_ratio, 3),
            "compression_ratio_aggregation": self.compression_ratio_aggregation,
            "eviction_fired": self.eviction_fired,
            "pass@1_sum": pass_at_1_sum,
            "selection": {
                "start": int(self.start),
                "stop_exclusive": int(self.start + self.n),
                "limit": self.limit,
            },
            # Full generations are required to reproduce Math-Verify grading;
            # keeping only an extracted answer makes a paper score unauditable.
            "per_question": self.per_question,
            "compression_sufficient_statistics": self.compression_sufficient_statistics,
        }
        if self.k > 1 and self.observed_pass_at_k is not None:
            out[f"observed_pass@{self.k}"] = round(self.observed_pass_at_k, 4)
        if self.dataset_size is not None:
            out["dataset_size"] = int(self.dataset_size)
            out["complete_dataset_eval"] = (
                self.start == 0 and self.limit is None and self.n == self.dataset_size
            )
        if self.k == 1:
            out["correct"] = int(round(pass_at_1_sum))
        if self.grader is not None:
            out["grader"] = self.grader
        if self.grader_version is not None:
            out["grader_version"] = self.grader_version
        if self.grader_config is not None:
            out["grader_config"] = self.grader_config
        if self.protocol is not None:
            out["protocol"] = self.protocol
        if self.prompt_config is not None:
            out["prompt_config"] = self.prompt_config
        if self.generation_config is not None:
            out["generation_config"] = self.generation_config
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
                return text[left + 1 : i].strip()
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
    """Legacy local MATH equality, retained only for explicit compatibility runs.

    Paper MATH-500 runs use :func:`math_verify_equivalent`, which delegates
    extraction and equivalence to Hugging Face Math-Verify.  This small local
    grader remains useful for reproducing historical rows, but must not be
    selected implicitly for a paper result.
    """
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
        if (
            pred.startswith(left)
            and pred.endswith(right)
            and ref.startswith(left)
            and ref.endswith(right)
        ):
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
            return math.isclose(float(sympy.N(p)), float(sympy.N(r)), rel_tol=1e-4, abs_tol=1e-8)
        except (TypeError, ValueError):
            return False
    except Exception:  # malformed/non-scalar expressions fall back to canonical exact match
        return False


def _build_math_verify_scorer(required_version: str = _MATH_VERIFY_VERSION):
    """Build the pinned official Math-Verify full-generation scorer.

    Math-Verify's LaTeX extractor expects a LaTeX environment.  MATH-500's
    ``answer`` column is a LaTeX *fragment* without delimiters, so gold answers
    are wrapped in ``$...$`` before parsing.  Predictions are passed as the
    complete generated response and use the package's recommended combined
    LaTeX/expression extraction.  There is deliberately no local fallback: a
    missing or mismatched dependency invalidates a paper benchmark run.
    """
    try:
        from math_verify import (  # pylint: disable=import-outside-toplevel
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MATH-500 paper evaluation requires "
            f"math-verify[antlr4_9_3]=={required_version}; install the "
            "'math-eval' project extra. The legacy local grader is available "
            "only via math_grader=local_symbolic_equivalence."
        ) from exc

    try:
        installed_version = package_version("math-verify")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "math_verify is importable but its package version cannot be "
            "verified; refusing to produce an unversioned MATH-500 result."
        ) from exc
    if installed_version != required_version:
        raise RuntimeError(
            "MATH-500 paper evaluation is pinned to "
            f"math-verify=={required_version}, found {installed_version}."
        )

    gold_config = (LatexExtractionConfig(boxed_match_priority=0),)
    prediction_config = (LatexExtractionConfig(), ExprExtractionConfig())

    @lru_cache(maxsize=512)
    def parse_gold(reference: str):
        # The HF MATH-500 answer field is a bare LaTeX fragment.  Delimit it so
        # Math-Verify parses symbolic answers (e.g. ``p-q`` and ordered pairs),
        # rather than accidentally falling back to the first numeric token.
        gold = parse(
            f"${str(reference).strip()}$",
            extraction_config=gold_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        if not gold:
            raise RuntimeError(
                "Math-Verify could not parse a MATH-500 gold answer: " f"{reference!r}"
            )
        return gold

    def score(prediction_text: str, reference: str) -> bool:
        gold = parse_gold(str(reference))
        prediction = parse(
            str(prediction_text),
            extraction_config=prediction_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        return bool(prediction) and bool(
            verify(
                gold,
                prediction,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                allow_set_relation_comp=False,
                timeout_seconds=5,
                raise_on_error=False,
            )
        )

    return score, installed_version


def math_verify_equivalent(
    prediction_text: str,
    reference: str,
    required_version: str = _MATH_VERIFY_VERSION,
) -> bool:
    """Grade a complete model response with pinned Math-Verify."""
    scorer, _ = _build_math_verify_scorer(required_version)
    return scorer(prediction_text, reference)


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


def _extract_input_ids(encoded, *, benchmark: str) -> torch.Tensor:
    """Return a validated single-example token tensor from tokenizer output."""
    if isinstance(encoded, torch.Tensor):
        ids = encoded
    elif isinstance(encoded, dict):
        ids = encoded.get("input_ids")
    else:
        ids = getattr(encoded, "input_ids", None)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise TypeError(f"{benchmark} tokenizer must return input_ids with shape (1, sequence)")
    if ids.shape[1] == 0:
        raise ValueError(f"{benchmark} tokenizer returned an empty prompt")
    return ids


def _build_prompt(
    tokenizer,
    question: str,
    system: Optional[str],
    *,
    use_chat_template: bool = True,
) -> torch.Tensor:
    """Render one reasoning query without silently changing prompt protocols."""
    if not use_chat_template:
        if system:
            raise ValueError("A system message cannot be represented in raw-text prompt mode")
        return _extract_input_ids(
            tokenizer(question, return_tensors="pt", truncation=False),
            benchmark="reasoning",
        )

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise RuntimeError(
            "The audited reasoning protocol requires the model's chat template, "
            "but this tokenizer has no apply_chat_template"
        )
    try:
        encoded = apply_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception as exc:
        # Falling back to raw question text would discard role markers and, when
        # present, the system message while leaving a plausible-looking score.
        raise RuntimeError("Reasoning chat-template rendering failed") from exc
    return _extract_input_ids(encoded, benchmark="reasoning")


def _sample_next(logits, do_sample, temperature, top_p):
    """Greedy or nucleus (top-p) sample of the next token from ``logits`` (1, V)."""
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        raise TypeError("next-token logits must be a non-empty torch.Tensor")
    if not torch.is_floating_point(logits):
        raise TypeError("next-token logits must have a floating-point dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError(
            "Reasoning decode produced non-finite next-token logits; "
            "refusing to sample or argmax NaN/inf"
        )
    if not do_sample:
        return logits.argmax(-1, keepdim=True)

    try:
        temperature = float(temperature)
        top_p = float(top_p)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature and top_p must be finite real numbers") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sampling temperature must be finite and > 0")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("sampling top_p must be finite and in (0, 1]")

    logits = logits / temperature
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("Temperature scaling produced non-finite logits")
    probs = torch.softmax(logits, dim=-1)
    if not bool(torch.isfinite(probs).all().item()):
        raise FloatingPointError("Reasoning decode produced non-finite sampling probabilities")
    probability_mass = probs.sum(-1, keepdim=True)
    if not bool((probability_mass > 0).all().item()) or not bool(
        torch.isclose(
            probability_mass,
            torch.ones_like(probability_mass),
            rtol=1e-3,
            atol=1e-6,
        )
        .all()
        .item()
    ):
        raise FloatingPointError("Reasoning sampling probabilities do not sum to one")
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf - sorted_probs <= top_p  # keep tokens up to and incl. the one crossing top_p
    sorted_probs = sorted_probs * keep
    retained_mass = sorted_probs.sum(-1, keepdim=True)
    if not bool(torch.isfinite(retained_mass).all().item()) or not bool(
        (retained_mass > 0).all().item()
    ):
        raise FloatingPointError("Top-p filtering retained zero or non-finite probability mass")
    sorted_probs = sorted_probs / retained_mass
    if not bool(torch.isfinite(sorted_probs).all().item()) or not bool(
        torch.isclose(
            sorted_probs.sum(-1, keepdim=True),
            torch.ones_like(retained_mass),
            rtol=1e-3,
            atol=1e-6,
        )
        .all()
        .item()
    ):
        raise FloatingPointError("Filtered top-p probabilities failed normalization")
    choice = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, choice)


class _FailClosedFiniteLogitsProcessor:
    """Reject non-finite model.generate scores before token selection."""

    def __call__(self, input_ids, scores):
        del input_ids
        if not bool(torch.isfinite(scores).all().item()):
            raise FloatingPointError(
                "Reasoning model.generate produced non-finite next-token logits"
            )
        return scores


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
def _eviction_decode(
    model,
    input_ids,
    cache,
    max_new_tokens,
    eos,
    do_sample,
    temperature,
    top_p,
    stop_criterion=None,
):
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

    def should_stop(logits) -> bool:
        token_id = int(toks[-1].item())
        if eos is not None:
            if isinstance(eos, (list, tuple, set, frozenset)):
                if token_id in eos:
                    return True
            elif token_id == int(eos):
                return True
        if stop_criterion is not None:
            # Match Transformers.generate: StopStringCriteria receives the
            # complete prompt+generation token sequence, not generated tokens
            # alone.  Some tokenizers form stop-string overlaps at that
            # boundary, so the two decode paths must expose identical input.
            complete_sequence = torch.cat([input_ids.to(dev), *toks], dim=1)
            if bool(stop_criterion(complete_sequence, logits).item()):
                return True
        return False

    # The previous loop checked EOS only after the *second* sampled token and
    # compared only against a scalar ID.  That could append thousands of tokens
    # after an immediate EOS and mishandled models with multiple EOS IDs.
    if should_stop(out.logits[:, -1]):
        return torch.cat(toks, dim=1)[0]
    for _ in range(max_new_tokens - 1):
        cp = torch.tensor([cur], device=dev)
        out = model(
            nxt,
            past_key_values=cache,
            use_cache=True,
            cache_position=cp,
            position_ids=cp.unsqueeze(0),
        )
        cache.evict_now()
        nxt = _sample_next(out.logits[:, -1], do_sample, temperature, top_p)
        toks.append(nxt)
        cur += 1
        if should_stop(out.logits[:, -1]):
            break
    return torch.cat(toks, dim=1)[0]  # (T,) to match the generate path


@torch.no_grad()
def _generate(
    model,
    tokenizer,
    input_ids,
    cache_factory,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    attn_capture=False,
    stop_sequences=(),
):
    past = cache_factory() if cache_factory is not None else None
    stop_sequences = tuple(stop_sequences or ())
    stop_criterion = None
    if stop_sequences:
        from transformers import StopStringCriteria  # local import

        stop_criterion = StopStringCriteria(tokenizer, list(stop_sequences))
    # Eviction caches must be trimmed via evict_now(); HF generate never calls it,
    # so route them through the explicit eviction-aware loop (B-5).  This keeps the
    # StreamingLLM window enforced for the whole rollout instead of silently
    # degenerating to FullKV.
    if past is not None and hasattr(past, "evict_now"):
        generation_config = getattr(model, "generation_config", None)
        eos = getattr(generation_config, "eos_token_id", None)
        if eos is None:
            eos = tokenizer.eos_token_id
        gen = _eviction_decode(
            model,
            input_ids,
            past,
            max_new_tokens,
            eos,
            do_sample,
            temperature,
            top_p,
            stop_criterion,
        )
        ratio = past.compression_ratio() if hasattr(past, "compression_ratio") else 1.0
        fired = getattr(past, "kept", 0) and getattr(past, "total", 0) and past.kept < past.total
        return gen, float(ratio), bool(fired), cache_compression_sufficient_statistics(past)
    model_input_ids = input_ids.to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        # Every reasoning example is an unpadded single sequence.  Supplying
        # the all-ones mask explicitly is still required when pad_token_id and
        # eos_token_id coincide; Transformers otherwise warns that it cannot
        # infer the mask and model-specific generation code may mask genuine
        # prompt tokens.
        attention_mask=torch.ones_like(model_input_ids),
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    from transformers import LogitsProcessorList  # local import

    gen_kwargs["logits_processor"] = LogitsProcessorList([_FailClosedFiniteLogitsProcessor()])
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    if stop_criterion is not None:
        from transformers import StoppingCriteriaList  # local import

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criterion])
    if past is not None:
        gen_kwargs["past_key_values"] = past
        gen_kwargs["use_cache"] = True
    if attn_capture and past is not None:
        # ThinKV thought classification reads attention sparsity; materialize
        # attn_weights (eager + output_attentions) and feed them into the cache.
        from ..cache import thinkv_attention_capture  # local import

        gen_kwargs["output_attentions"] = True
        with thinkv_attention_capture(model, past):
            out = model.generate(model_input_ids, **gen_kwargs)
    else:
        out = model.generate(model_input_ids, **gen_kwargs)
    if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != 1:
        raise TypeError("Reasoning model.generate must return a rank-2 single-sequence tensor")
    if out.shape[1] <= input_ids.shape[1]:
        raise RuntimeError("Reasoning model.generate returned no generated tokens")
    gen = out[0, input_ids.shape[1] :]
    if gen.shape[0] > max_new_tokens:
        raise RuntimeError("Reasoning model.generate exceeded max_new_tokens")
    ratio = (
        past.compression_ratio()
        if (past is not None and hasattr(past, "compression_ratio"))
        else 1.0
    )
    return gen, float(ratio), True, cache_compression_sufficient_statistics(past)


def _truncate_at_stop(text: str, stop_sequences) -> tuple[str, Optional[str]]:
    """Remove the earliest official stop marker from a decoded generation."""
    matches = [
        (text.find(sequence), sequence)
        for sequence in stop_sequences
        if sequence and text.find(sequence) >= 0
    ]
    if not matches:
        return text, None
    index, sequence = min(matches, key=lambda item: item[0])
    return text[:index], sequence


@torch.no_grad()
def _eval_dataset(
    model,
    tokenizer,
    items,
    gold_fn,
    task,
    cache_factory,
    k,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    system,
    verbose,
    attn_capture=False,
    answer_eq=None,
    sample_seed: Optional[int] = 0,
    dataset_size: Optional[int] = None,
    grader: Optional[str] = None,
    grader_version: Optional[str] = None,
    grader_config: Optional[dict] = None,
    protocol: Optional[str] = None,
    score_full_generation: bool = False,
    start_index: int = 0,
    selection_limit: Optional[int] = None,
    prompt_config: Optional[dict] = None,
    generation_config: Optional[dict] = None,
    stop_sequences=(),
    use_chat_template: bool = True,
):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("Reasoning k must be a positive integer")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("Reasoning max_new_tokens must be a positive integer")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("Reasoning start_index must be a non-negative integer")
    if sample_seed is not None and (
        isinstance(sample_seed, bool) or not isinstance(sample_seed, int)
    ):
        raise ValueError("Reasoning sample_seed must be an integer or None")
    if not isinstance(do_sample, bool) or not isinstance(attn_capture, bool):
        raise ValueError("Reasoning do_sample and attn_capture must be bools")
    if not items:
        raise ValueError("Reasoning evaluation selection is empty")
    stop_sequences = tuple(stop_sequences or ())
    effective_sampling = bool(do_sample or k > 1)
    if effective_sampling:
        try:
            temperature_value = float(temperature)
            top_p_value = float(top_p)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Reasoning sampling parameters must be finite real numbers") from exc
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError("Reasoning sampling temperature must be finite and > 0")
        if not math.isfinite(top_p_value) or not 0.0 < top_p_value <= 1.0:
            raise ValueError("Reasoning sampling top_p must be finite and in (0, 1]")
    result_prompt_config = {
        "use_chat_template": bool(use_chat_template),
        "system": system,
        "query_protocol": "dataset query verbatim",
    }
    if prompt_config:
        result_prompt_config.update(prompt_config)
    result_generation_config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": effective_sampling,
        "requested_do_sample": bool(do_sample),
        "temperature": float(temperature) if effective_sampling else None,
        "top_p": float(top_p) if effective_sampling else None,
        "stop_sequences": list(stop_sequences),
        "sample_seed_base": sample_seed,
        "per_sample_seed_formula": (
            "base + global_question_index * 100003 + sample_index"
            if sample_seed is not None
            else None
        ),
    }
    if generation_config:
        result_generation_config.update(generation_config)
    n = 0
    p1_values: List[float] = []
    gen_lens: List[int] = []
    ratios: List[float] = []
    per_q: List[dict] = []
    fired_all = True
    saw_eviction = False
    fired_values: List[bool] = []
    pass_k_values: List[float] = []
    answer_eq = answer_eq or (
        lambda pred, gold: _normalize_number(pred) == _normalize_number(gold)
    )
    for local_question_idx, (question, gold_raw) in enumerate(items):
        question_idx = start_index + local_question_idx
        gold = gold_fn(gold_raw)
        ids = _build_prompt(
            tokenizer,
            question,
            system,
            use_chat_template=use_chat_template,
        )
        correct = 0
        any_correct = False
        sample_records = []
        for sample_idx in range(k):
            seed = (
                None
                if sample_seed is None
                else int(sample_seed) + question_idx * 100_003 + sample_idx
            )
            with _seeded_rng(seed, model.device):
                gen, ratio, fired, compression = _generate(
                    model,
                    tokenizer,
                    ids,
                    cache_factory,
                    max_new_tokens,
                    do_sample or k > 1,
                    temperature,
                    top_p,
                    attn_capture=attn_capture,
                    stop_sequences=stop_sequences,
                )
            if (
                isinstance(ratio, bool)
                or not isinstance(ratio, (int, float))
                or math.isnan(float(ratio))
                or float(ratio) <= 0.0
            ):
                raise RuntimeError(f"Reasoning cache returned invalid compression ratio {ratio!r}")
            if not isinstance(fired, bool):
                raise TypeError("Reasoning generator eviction flag must be a bool")
            if (
                not isinstance(gen, torch.Tensor)
                or gen.ndim != 1
                or gen.shape[0] == 0
                or gen.shape[0] > max_new_tokens
            ):
                raise RuntimeError("Reasoning generator must return 1..max_new_tokens token IDs")
            gen_lens.append(int(gen.shape[0]))
            ratios.append(float(ratio))
            fired_values.append(fired)
            raw_text = tokenizer.decode(gen, skip_special_tokens=True)
            text, matched_stop = _truncate_at_stop(raw_text, stop_sequences)
            pred = extract_answer(text)
            grader_prediction = text if score_full_generation else pred
            is_correct = (
                grader_prediction is not None
                and gold is not None
                and answer_eq(grader_prediction, gold)
            )
            correct += int(is_correct)
            any_correct = any_correct or is_correct
            sample_records.append(
                {
                    "seed": seed,
                    "generated_text": text,
                    "raw_generated_text": raw_text,
                    "extracted_answer": pred,
                    "generated_tokens": int(gen.shape[0]),
                    "matched_stop_sequence": matched_stop,
                    "correct": is_correct,
                    **compression,
                }
            )
        # ThinKV's reported pass@1 is the average correctness across k independent
        # responses. Keep observed any-correct only as a separately named diagnostic.
        p1 = correct / k
        pass_at_k = float(any_correct)
        p1_values.append(p1)
        pass_k_values.append(pass_at_k)
        n += 1
        per_q.append(
            {
                "index": question_idx,
                "question": question,
                "gold": gold,
                "input_tokens": int(ids.shape[1]),
                "pass@1": p1,
                f"observed_pass@{k}": pass_at_k,
                "samples": sample_records,
                "k": k,
            }
        )
        if verbose:
            logger.info("[%s] q%d gold=%s pass@1=%.3f (k=%d)", task, n, gold, p1, k)
    # Detect whether an eviction cache actually trimmed (else its result is FullKV).
    probe = cache_factory() if cache_factory is not None else None
    if probe is not None and hasattr(probe, "evict_now"):
        saw_eviction = True
        fired_all = bool(fired_values) and any(fired_values)
        if not fired_all:
            logger.warning(
                "[%s] eviction cache did not trim (compression_ratio~1): the "
                "context is shorter than sink+recent, so this row equals FullKV.",
                task,
            )
    mean_p1, se, ci = _mean_stderr_ci(p1_values)
    generation_records = [sample for question in per_q for sample in question["samples"]]
    compression_statistics = aggregate_compression_sufficient_statistics(generation_records)
    if compression_statistics["compression_aggregation"] == (
        "sum_exact_per_generation_cache_bit_counters"
    ):
        exact_fp16_bits = int(compression_statistics["compression_fp16_bits"])
        exact_stored_bits = int(compression_statistics["compression_stored_bits"])
        displayed_compression_ratio = (
            float("inf") if exact_stored_bits == 0 else exact_fp16_bits / exact_stored_bits
        )
        compression_ratio_aggregation = "exact_sum_fp16_bits_over_sum_stored_bits"
    elif cache_factory is None:
        displayed_compression_ratio = 1.0
        compression_ratio_aggregation = "fullkv_no_cache"
    else:
        # Compatibility only for caches predating exact integer bit counters.
        # Formal codecs are rejected by downstream gates unless the exact branch
        # above is used; this explicitly named arithmetic fallback must never be
        # interpreted as an aggregate payload ratio when generation lengths vary.
        displayed_compression_ratio = float(math.fsum(ratios) / len(ratios)) if ratios else 1.0
        compression_ratio_aggregation = (
            "legacy_arithmetic_mean_per_generation_ratio_no_exact_counters"
        )
    return ReasoningResult(
        task=task,
        n=n,
        k=k,
        pass_at_1=mean_p1,
        avg_gen_len=sum(gen_lens) / max(len(gen_lens), 1),
        compression_ratio=displayed_compression_ratio,
        compression_ratio_aggregation=compression_ratio_aggregation,
        per_question=per_q,
        pass_at_1_stderr=se,
        pass_at_1_ci95=ci,
        eviction_fired=(fired_all if saw_eviction else True),
        observed_pass_at_k=(sum(pass_k_values) / len(pass_k_values) if pass_k_values else None),
        dataset_size=dataset_size,
        grader=grader,
        grader_version=grader_version,
        grader_config=grader_config,
        protocol=protocol,
        prompt_config=result_prompt_config,
        generation_config=result_generation_config,
        start=start_index,
        limit=selection_limit,
        pass_at_1_sum=float(sum(p1_values)),
        compression_sufficient_statistics=compression_statistics,
    )


def gsm8k_pass_at_1(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    limit: int = 50,
    k: int = 1,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[
        str
    ] = "Please reason step by step, and put your final answer within \\boxed{}.",
    verbose: bool = False,
    attn_capture: bool = False,
    sample_seed: Optional[int] = 0,
) -> ReasoningResult:
    """GSM8K (main, test) pass@1 under ``cache_factory``.

    Set ``attn_capture=True`` for ThinKV (engages attention-sparsity thought
    classification; requires the model loaded with ``attn_implementation='eager'``).
    """
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset("openai/gsm8k", "main", split="test", revision=_GSM8K_REVISION)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("GSM8K limit must be a positive integer or None")
    count = len(ds) if limit is None else min(limit, len(ds))
    items = [(ds[i]["question"], ds[i]["answer"]) for i in range(count)]
    effective_limit = None if count == len(ds) else limit
    return _eval_dataset(
        model,
        tokenizer,
        items,
        gsm8k_gold,
        "gsm8k",
        cache_factory,
        k,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        system,
        verbose,
        attn_capture=attn_capture,
        sample_seed=sample_seed,
        dataset_size=len(ds),
        grader="exact_numeric_answer",
        grader_config={
            "dataset": "openai/gsm8k",
            "dataset_revision": _GSM8K_REVISION,
        },
        protocol="openai_gsm8k_test",
        selection_limit=effective_limit,
    )


def math500_pass_at_1(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    limit: int = 50,
    k: int = 1,
    max_new_tokens: int = _MATH500_MAX_NEW_TOKENS,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[str] = None,
    verbose: bool = False,
    attn_capture: bool = False,
    sample_seed: Optional[int] = 0,
    grader: str = "math_verify",
    math_verify_version: str = _MATH_VERIFY_VERSION,
    dataset_revision: str = _MATH500_REVISION,
    start: int = 0,
) -> ReasoningResult:
    """MATH-500 pass@1 under ``cache_factory`` (HuggingFaceH4/MATH-500).

    The default paper protocol is a fixed composite adaptation: the pinned
    Math-Verify custom task supplies query/generation/scoring components, and
    the pinned LightEval prompt-manager algorithm supplies the manually applied
    four-shot selection/rendering semantics.  It is neither LightEval's built-in
    MATH-500 task nor an unchanged executable upstream package pair.  It explicitly
    corrects the reference task's bare-gold serialization by wrapping each answer
    fragment in inline-math delimiters before LaTeX extraction; this adaptation is
    recorded and is not claimed to be a byte-identical reference-task scorer.
    ``grader='local_symbolic_equivalence'`` is available only for explicitly
    reproducing legacy rows; it is never a silent fallback.

    Set ``attn_capture=True`` for ThinKV (engages thought classification)."""
    if grader == "math_verify":
        pinned_generation = system is None and max_new_tokens == _MATH500_MAX_NEW_TOKENS
        answer_eq, installed_grader_version = _build_math_verify_scorer(math_verify_version)
        grader_config = {
            "dataset": _MATH500_DATASET,
            "dataset_revision": dataset_revision,
            "gold_extraction": "latex(boxed_match_priority=0; inline_math_wrapped)",
            "gold_serialization": _MATH500_GOLD_SERIALIZATION,
            "official_reference_gold_serialization_exact": False,
            "gold_serialization_adaptation_reason": _MATH500_GOLD_ADAPTATION_REASON,
            "protocol_scope": _MATH500_PROTOCOL_SCOPE,
            "prediction_extraction": "full_generation:latex+expr",
            "parse": {
                "fallback_mode": "first_match",
                "extraction_mode": "any_match",
                "parsing_timeout_seconds": 5,
                "raise_on_error": False,
            },
            "verify": {
                "float_rounding": 6,
                "numeric_precision": 15,
                "strict": True,
                "allow_set_relation_comp": False,
                "timeout_seconds": 5,
                "raise_on_error": False,
            },
            "strict": True,
        }
        protocol = (
            "hf_math500_test_adapted_composite_four_shot"
            if pinned_generation
            else "exploratory_math500_math_verify_custom_generation"
        )
        score_full_generation = True
        fewshot_prefix = _MATH500_FEWSHOT_PREFIX
    elif grader == "local_symbolic_equivalence":
        answer_eq = math_equivalent
        installed_grader_version = None
        grader_config = {
            "dataset": _MATH500_DATASET,
            "dataset_revision": dataset_revision,
            "prediction_extraction": "legacy_extract_answer",
        }
        protocol = "legacy_local_single_model_generation"
        score_full_generation = False
        fewshot_prefix = ""
    else:
        raise ValueError(
            f"Unsupported MATH-500 grader {grader!r}; use 'math_verify' "
            "(paper protocol) or 'local_symbolic_equivalence' (legacy only)."
        )

    from datasets import load_dataset  # pylint: disable=import-outside-toplevel

    ds = load_dataset(_MATH500_DATASET, split="test", revision=dataset_revision)
    if len(ds) != 500:
        raise RuntimeError(f"Expected the complete 500-row MATH-500 test set, got {len(ds)}")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0 or start >= len(ds):
        raise ValueError(f"MATH-500 start must be in [0, {len(ds) - 1}], got {start}")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("MATH-500 limit must be a positive integer or None")
    stop = len(ds) if limit is None else min(start + limit, len(ds))
    effective_limit = None if start == 0 and stop == len(ds) else limit
    items = [
        (
            (f"{fewshot_prefix}\n\n" if fewshot_prefix else "")
            + _MATH500_QUERY_TEMPLATE.format(problem=ds[i]["problem"]),
            ds[i]["answer"],
        )
        for i in range(start, stop)
    ]
    return _eval_dataset(
        model,
        tokenizer,
        items,
        lambda a: str(a).strip(),
        "math500",
        cache_factory,
        k,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        system,
        verbose,
        attn_capture=attn_capture,
        answer_eq=answer_eq,
        sample_seed=sample_seed,
        dataset_size=len(ds),
        grader=grader,
        grader_version=installed_grader_version,
        grader_config=grader_config,
        protocol=protocol,
        score_full_generation=score_full_generation,
        start_index=start,
        selection_limit=effective_limit,
        prompt_config={
            "query_protocol": "Math-Verify math_500 task query",
            "query_template": _MATH500_QUERY_TEMPLATE,
            "few_shot": (4 if grader == "math_verify" else 0),
            "few_shot_order_one_based": (
                list(_MATH500_FEWSHOT_ORDER) if grader == "math_verify" else []
            ),
            "few_shot_content_sha256": (
                _MATH500_FEWSHOT_CONTENT_SHA256 if grader == "math_verify" else None
            ),
            "few_shot_rendering": (
                'join(query + " " + answer.strip(), separator="\\n\\n") then main query'
                if grader == "math_verify"
                else None
            ),
            "system": system,
            "protocol_id": _MATH500_PROTOCOL_ID,
            "protocol_origin": "adapted_composite",
            "math_verify_task_source_commit": _MATH_VERIFY_TASK_SOURCE_COMMIT,
            "lighteval_prompt_manager_reference_version": (
                _MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
                if grader == "math_verify"
                else None
            ),
            "lighteval_prompt_manager_source_commit": (
                _MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT
                if grader == "math_verify"
                else None
            ),
            "upstream_lighteval_builtin_task_exact": (False if grader == "math_verify" else None),
            "upstream_math_verify_custom_task_exact": (False if grader == "math_verify" else None),
            "upstream_prompt_generation_components_exact": (
                True if grader == "math_verify" else None
            ),
            "upstream_component_pair_executable": (False if grader == "math_verify" else None),
            "lighteval_balanced_sampler_seed": (0 if grader == "math_verify" else None),
            "use_chat_template": False,
        },
        generation_config={
            "protocol": (
                (
                    "adapted_composite_math_verify_ba3d3aa_lighteval_"
                    "prompt_manager_7c1cd62_four_shot"
                    if pinned_generation
                    else "exploratory_nonreference_generation"
                )
                if grader == "math_verify"
                else "legacy_local_grader_generation"
            ),
            "reference_max_new_tokens": _MATH500_MAX_NEW_TOKENS,
        },
        stop_sequences=_MATH500_STOP_SEQUENCES,
        use_chat_template=False,
    )


def aime_pass_at_1(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    year: int = 2024,
    limit: int = 30,
    k: int = 8,
    max_new_tokens: int = 8192,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[
        str
    ] = "Please reason step by step, and put your final answer within \\boxed{}.",
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
        ds_all = load_dataset(
            "AI-MO/aimo-validation-aime",
            split="train",
            revision=_AIME2024_REVISION,
        )
        items = [
            (row["problem"], str(row["answer"])) for row in ds_all if "2024" in row.get("url", "")
        ]
    elif year == 2025:
        ds = load_dataset("yentinglin/aime_2025", split="train", revision=_AIME2025_REVISION)
        items = [(row["problem"], str(row["answer"])) for row in ds]
    else:
        raise ValueError(f"Unsupported AIME year={year}; use 2024 or 2025")

    if len(items) != 30:
        raise RuntimeError(
            f"Expected the complete 30-problem AIME {year} set, got {len(items)}; "
            "dataset schema/filter may have changed"
        )
    dataset_size = len(items)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("AIME limit must be a positive integer or None")
    items = items if limit is None else items[:limit]
    effective_limit = None if len(items) == dataset_size else limit

    def _aime_gold(answer_str: str) -> Optional[str]:
        """AIME answers are integers 000-999; normalise to plain int string."""
        s = answer_str.strip().lstrip("0") or "0"
        try:
            return str(int(float(s)))
        except ValueError:
            return s

    return _eval_dataset(
        model,
        tokenizer,
        items,
        _aime_gold,
        f"aime{year}",
        cache_factory,
        k,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        system,
        verbose,
        attn_capture=attn_capture,
        sample_seed=sample_seed,
        dataset_size=dataset_size,
        grader="exact_integer_answer",
        grader_config={
            "dataset": ("AI-MO/aimo-validation-aime" if year == 2024 else "yentinglin/aime_2025"),
            "dataset_revision": (_AIME2024_REVISION if year == 2024 else _AIME2025_REVISION),
        },
        protocol=f"aime{year}_competition_set",
        selection_limit=effective_limit,
    )
