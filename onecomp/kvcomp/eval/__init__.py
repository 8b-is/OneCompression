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

from .metrics import (
    KVMemoryReport,
    effective_bits,
    kv_cache_bytes,
    summarize_compression,
)
from .ppl import wikitext2_perplexity, perplexity_over_text
from .zeroshot import zeroshot_accuracy, ZeroShotResult
from .reasoning import gsm8k_pass_at_1, math500_pass_at_1, aime_pass_at_1, ReasoningResult
from .longcontext import (
    needle_in_haystack,
    ruler_multikey,
    make_needle_prompt,
    make_multikey_prompt,
    greedy_answer,
    NeedleResult,
)
from .longbench import longbench_e, LongBenchResult, LONGBENCH_E_TASKS

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
    "ReasoningResult",
    "needle_in_haystack",
    "ruler_multikey",
    "make_needle_prompt",
    "make_multikey_prompt",
    "greedy_answer",
    "NeedleResult",
    "longbench_e",
    "LongBenchResult",
    "LONGBENCH_E_TASKS",
]
