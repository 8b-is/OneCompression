"""LiveCodeBench-style code-generation pass@1 under a KV ``cache_factory``.

ThinKV (Ramachandran et al., ICLR 2026) reports **LiveCodeBench** alongside
MATH-500 / AIME / GSM8K, so a paper comparable to ThinKV needs the code axis too.
This module adds the functional-correctness ``pass@1`` protocol shared by
HumanEval / MBPP / LiveCodeBench (functional subset): the model writes a solution
from a problem statement, and the extracted code is executed against held-out
assertions in a **sandboxed subprocess** (CPU-time + address-space rlimits, a hard
wall-clock timeout, and a throw-away working directory).  It returns the same
:class:`~onecomp.kvcomp.eval.reasoning.ReasoningResult` (pass@1 + stderr + 95% CI)
as the math evals, so the paper table stays uniform and every quality number
carries an error bar.

A *problem* is a dict ``{"prompt": str, "entry_point": str, "test": str}`` where
``test`` defines ``def check(candidate): ...`` (the HumanEval convention).  The
program executed is ``<extracted code>\n<test>\ncheck(<entry_point>)``.  Pass the
real LiveCodeBench (functional subset) via :func:`livecodebench_pass_at_1`, or any
local list of problems via :func:`code_pass_at_1` (used by the unit tests).

Security note: generated code is untrusted.  It runs in a child ``python -c``
process with ``RLIMIT_CPU`` / ``RLIMIT_AS`` and a timeout; it never executes in
the benchmark process.  This is a research sandbox, not a hardened jail -- run the
harness on an isolated node, as with any code-execution benchmark.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from logging import getLogger
from typing import Callable, List, Optional

from .reasoning import ReasoningResult, _build_prompt, _generate, _mean_stderr_ci

logger = getLogger(__name__)

_CODE_SYSTEM = (
    "You are an expert Python programmer. Read the problem and write a correct, "
    "self-contained solution. Put the final solution in a single ```python ... ``` "
    "code block and define exactly the requested function."
)

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, entry_point: Optional[str] = None) -> str:
    """Pull the Python solution out of a model generation.

    Prefers the **last** fenced ```python``` block (reasoning models emit the
    final answer last); falls back to the largest fenced block, then -- if the
    model emitted bare code -- to the text from the first ``def``/``import`` on.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        if entry_point:
            named = [b for b in blocks if f"def {entry_point}" in b]
            if named:
                return named[-1].strip()
        return blocks[-1].strip()
    # No fences: take from the first top-level def/import/class to the end.
    m = re.search(r"^(?:from |import |def |class )", text, re.MULTILINE)
    return text[m.start() :].strip() if m else text.strip()


def _build_program(code: str, test: str, entry_point: str) -> str:
    """Assemble the standalone program: candidate code + checker + invocation."""
    return f"{code}\n\n{test}\n\ncheck({entry_point})\n"


def _limit_resources(cpu_s: int, mem_mb: int):
    """Return a ``preexec_fn`` that caps CPU time and address space (POSIX)."""

    def _set():
        try:
            import resource  # POSIX only

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            nbytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except Exception:
            pass

    return _set


def run_code(program: str, timeout: float = 6.0, mem_mb: int = 2048) -> tuple:
    """Execute ``program`` in a sandboxed subprocess. Returns ``(passed, detail)``."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=(
                    _limit_resources(int(timeout) + 1, mem_mb) if os.name == "posix" else None
                ),
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # pragma: no cover - spawn failure
            return False, f"spawn-error: {e}"
    if proc.returncode == 0:
        return True, "ok"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (lines[-1] if lines else "nonzero-exit")


def code_pass_at_1(
    model,
    tokenizer,
    problems: List[dict],
    cache_factory: Optional[Callable[[], object]] = None,
    k: int = 1,
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    system: Optional[str] = _CODE_SYSTEM,
    timeout: float = 6.0,
    mem_mb: int = 2048,
    task: str = "livecodebench",
    verbose: bool = False,
    attn_capture: bool = False,
) -> ReasoningResult:
    """Functional-correctness pass@1 over ``problems`` under ``cache_factory``.

    Each problem: ``{"prompt", "entry_point", "test"}`` (HumanEval convention,
    ``test`` defines ``check(candidate)``).  Mirrors the math evals' aggregation
    so the result carries pass@1, its standard error, and a 95% CI.
    """
    n = 0
    p1_values: List[float] = []
    gen_lens: List[int] = []
    ratios: List[float] = []
    per_q: List[dict] = []
    fired_all, saw_eviction = True, False
    for prob in problems:
        prompt = prob["prompt"]
        entry = prob["entry_point"]
        test = prob["test"]
        ids = _build_prompt(tokenizer, prompt, system)
        passed = 0
        for _ in range(k):
            gen, ratio, _fired = _generate(
                model,
                tokenizer,
                ids,
                cache_factory,
                max_new_tokens,
                do_sample or k > 1,
                temperature,
                top_p,
                attn_capture=attn_capture,
            )
            gen_lens.append(int(gen.shape[0]))
            ratios.append(ratio)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            code = extract_code(text, entry)
            ok, detail = run_code(_build_program(code, test, entry), timeout, mem_mb)
            passed += int(ok)
            if verbose:
                logger.info("[%s] %s k-sample -> %s (%s)", task, entry, ok, detail)
        p1 = passed / k
        p1_values.append(p1)
        n += 1
        per_q.append({"entry_point": entry, "pass@1": p1})
    probe = cache_factory() if cache_factory is not None else None
    if probe is not None and hasattr(probe, "evict_now"):
        saw_eviction = True
        fired_all = bool(ratios) and (sum(ratios) / len(ratios)) > 1.0 + 1e-6
    mean_p1, se, ci = _mean_stderr_ci(p1_values)
    return ReasoningResult(
        task=task,
        n=n,
        k=k,
        pass_at_1=mean_p1,
        avg_gen_len=sum(gen_lens) / max(len(gen_lens), 1),
        compression_ratio=float(sum(ratios) / len(ratios)) if ratios else 1.0,
        per_question=per_q,
        pass_at_1_stderr=se,
        pass_at_1_ci95=ci,
        eviction_fired=(fired_all if saw_eviction else True),
    )


def _adapt_livecodebench(row: dict) -> Optional[dict]:
    """Map a LiveCodeBench/HumanEval-style row to ``{prompt, entry_point, test}``.

    Supports (a) HumanEval/MBPP fields (``prompt``/``text``, ``entry_point``,
    ``test``) and (b) a LiveCodeBench functional row exposing a question body plus
    a ``check(candidate)`` test.  Rows that are not functional (stdin/stdout I/O
    problems) return ``None`` and are skipped -- supply your own adapter for those.
    """
    entry = row.get("entry_point") or row.get("func_name")
    test = row.get("test") or row.get("test_code")
    prompt = (
        row.get("prompt") or row.get("question_content") or row.get("text") or row.get("problem")
    )
    starter = row.get("starter_code") or row.get("starter") or ""
    if not (entry and test and prompt):
        return None
    full_prompt = (
        prompt
        if not starter
        else f"{prompt}\n\nComplete this starter code:\n```python\n{starter}\n```"
    )
    full_prompt += f"\n\nDefine the function `{entry}`."
    if "def check" not in test:
        return None  # only the functional (assert-via-check) protocol is supported here
    return {"prompt": full_prompt, "entry_point": entry, "test": test}


def livecodebench_pass_at_1(
    model,
    tokenizer,
    cache_factory: Optional[Callable[[], object]] = None,
    dataset: str = "livecodebench/code_generation_lite",
    split: str = "test",
    limit: int = 50,
    problems: Optional[List[dict]] = None,
    adapter: Callable[[dict], Optional[dict]] = _adapt_livecodebench,
    **kwargs,
) -> ReasoningResult:
    """LiveCodeBench (functional subset) pass@1 under ``cache_factory``.

    Pass an explicit ``problems`` list (``{prompt, entry_point, test}``) to bypass
    dataset loading, e.g. a local LiveCodeBench JSONL; otherwise the HF ``dataset``
    is loaded and adapted via ``adapter`` (functional rows only).  Extra kwargs go
    to :func:`code_pass_at_1` (``k``, ``max_new_tokens``, ``timeout`` ...).
    """
    if problems is None:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel

        ds = load_dataset(dataset, split=split)
        problems = []
        for row in ds:
            p = adapter(row)
            if p is not None:
                problems.append(p)
            if len(problems) >= limit:
                break
        if not problems:
            raise ValueError(
                f"No functional problems adapted from {dataset!r}; supply a "
                "`problems` list or a custom `adapter` for stdin/stdout tasks."
            )
    else:
        problems = problems[:limit]
    return code_pass_at_1(model, tokenizer, problems, cache_factory, **kwargs)
