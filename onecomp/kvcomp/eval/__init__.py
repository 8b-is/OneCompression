"""Evaluation harnesses for KV cache compression.

This subpackage provides the *quality* side of the (memory, latency, quality)
triple that the design doc mandates: a codec/selector/policy must never be
judged by compression ratio alone.  It mirrors the role of the weight side's
``model_validation`` Phase-2/3 perplexity + downstream checks.

Modules
-------
* :mod:`onecomp.kvcomp.eval.metrics` -- KV memory / effective-bits accounting.
* :mod:`onecomp.kvcomp.eval.ppl`     -- WikiText-2 sliding-window perplexity.
* :mod:`onecomp.kvcomp.eval.zeroshot`-- zero-shot accuracy via log-likelihood
  (ARC-Easy / PIQA / HellaSwag), lm-eval-harness compatible scoring.
* :mod:`onecomp.kvcomp.eval.reasoning` -- reasoning pass@1 (GSM8K / MATH-500),
  the benchmark family used by the ThinKV paper.

Every harness accepts a ``cache_factory`` callable ``() -> transformers.Cache``
so the *same* code measures the FullKV baseline and any compressed cache
(``CompressedDynamicCache`` for static codecs, ``ThinKVCache`` for ThinKV).

Copyright 2025-2026 Fujitsu Ltd.
"""

from .code import code_pass_at_1, extract_code, livecodebench_pass_at_1
from .longbench import (
    LONGBENCH_E_TASK_COUNTS,
    LONGBENCH_E_TASKS,
    LONGBENCH_E_TOTAL,
    LongBenchResult,
    longbench_e,
)
from .longcodeqa import (
    LONGCODEQA_BRACKET_COUNTS,
    LongCodeQAResult,
    aggregate_longcodeqa,
    extract_longcodeqa_letter,
    longcodeqa,
)
from .longcontext import (
    NeedleResult,
    greedy_answer,
    make_multikey_prompt,
    make_needle_prompt,
    needle_in_haystack,
    ruler_multikey,
)
from .metrics import (
    KVMemoryReport,
    effective_bits,
    kv_cache_bytes,
    summarize_compression,
)
from .ppl import perplexity_over_text, wikitext2_perplexity
from .reasoning import (
    ReasoningResult,
    aime_pass_at_1,
    gsm8k_pass_at_1,
    math500_pass_at_1,
    math_verify_equivalent,
)
from .ruler import (
    RULER_V1_MAX_NEW_TOKENS,
    RULER_V1_SAMPLES_PER_TASK,
    RULER_V1_TASKS,
    RulerV1Result,
    load_ruler_v1_task,
    ruler_v1,
    ruler_v1_sample_score,
)
from .zeroshot import ZeroShotResult, zeroshot_accuracy

__all__ = [
    "KVMemoryReport",
    "effective_bits",
    "kv_cache_bytes",
    "summarize_compression",
    "wikitext2_perplexity",
    "perplexity_over_text",
    "zeroshot_accuracy",
    "ZeroShotResult",
    "gsm8k_pass_at_1",
    "math500_pass_at_1",
    "aime_pass_at_1",
    "math_verify_equivalent",
    "ReasoningResult",
    "code_pass_at_1",
    "livecodebench_pass_at_1",
    "extract_code",
    "needle_in_haystack",
    "ruler_multikey",
    "make_needle_prompt",
    "make_multikey_prompt",
    "greedy_answer",
    "NeedleResult",
    "longbench_e",
    "LongBenchResult",
    "LONGBENCH_E_TASKS",
    "LONGBENCH_E_TASK_COUNTS",
    "LONGBENCH_E_TOTAL",
    "ruler_v1",
    "ruler_v1_sample_score",
    "load_ruler_v1_task",
    "RulerV1Result",
    "RULER_V1_TASKS",
    "RULER_V1_MAX_NEW_TOKENS",
    "RULER_V1_SAMPLES_PER_TASK",
    "longcodeqa",
    "aggregate_longcodeqa",
    "extract_longcodeqa_letter",
    "LongCodeQAResult",
    "LONGCODEQA_BRACKET_COUNTS",
]
