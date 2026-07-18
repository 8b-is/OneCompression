# HayateQuant paper later-section read-only audit (2026-07-16)

This audit inspected
`/lustre/home/frj/yuma-ichikawa/research/ICLR2027/Hayate/main.tex`
without editing it.  It fixes the protected text identities before any future
paper refresh:

| Scope | Bytes | SHA-256 |
|---|---:|---|
| full `main.tex` | 174781 | `16147ff59dada17ae65bfda3d7c5b5bbf4c0f9ff41845f62b0319ad56aed6457` |
| abstract block, preserving source CRLF | 1797 | `ec4cd174665ce92d1366ca3bcfc86878862612df86d7faba5e35aec5d9144f29` |
| Introduction through immediately before Preliminaries | 6024 | `9e9647ea1f5bfc6d56d99737637a389c23db161e9bea2bbfc4a17e20389ebc90` |

The abstract and introduction are protected invariants.  No edit is needed or
permitted for the current refresh unless the author explicitly changes that
constraint.

## Current result boundary

At audit time, both the old external formal directory and the repository
formal directory contained zero formal result files.  The smoke approval,
coverage-checked aggregate, and rendered formal LaTeX includes did not exist.
Consequently no formal PPL, zero-shot, long-context, speed direction, or 90%
dominance statement can be refreshed yet.  Numeric paper content must be
produced only by `render_hayatequant_formal_results.py` from a complete
fail-closed aggregate; development-native values must not be copied into the
P-formal macros.

## Minimal later-section changes after complete evidence

1. Around source lines 597 and 1833, retain the legacy key-only hybrid label,
   but separately state that a full packed fixed-HQ K/V native implementation
   now exists.  Bound it as development-only until formal validation; Adaptive
   native and a native speed-superiority claim remain unavailable.
2. Around source line 646, replace only the obsolete statement that all full
   packed HQ attention is future work.  The implementation exists; deployment
   validation, Adaptive native support, and a passing formal speed result are
   pending.
3. Around source line 1825, record the actual isolated speed protocol only
   after it is instantiated: one explicit all-8-B200 node allocation per fresh
   process, `exclusive=user`, afterok serialization, one visible measurement
   GPU inside the worker, and exclusion of that node from quality jobs.
4. Around source lines 702--704, state baseline identity limits concisely:
   TurboQuant is its integer-rate Algorithm-1 MSE row rather than a complete
   Table-1 mixed-recipe reproduction; FibQuant is method-faithful exact-search
   with a pinned deterministic completion, not byte-identical published
   centroids; BlockQuant is an unmatched fixed-seed descriptive completion,
   not a paper reproduction or dominance-denominator row.

The current PPL/zero-shot/MATH/LongBench/RULER/LongCodeQA descriptions around
lines 516 and 525--536, and the baseline limitations around lines 636 and
1273--1279, already use the required bounded language and should not be
rewritten semantically.

## Native evidence boundary

The repository development archives establish packed correctness and a large
improvement over an earlier HQ implementation, but every observed exact-HQ
speed endpoint still loses to the unmodified TQ native path.  They therefore
support implementation-status text only, not superiority:

* `native_full_clean_8aee55b_gpu11_20260716/`
* `native_tensorcore_exact_accepted_20260716/`
* `native_formal_integrated_smoke_20260716/`
* `hayatequant_native_speed_formal_contract.json`

The preregistered 48-cell packed-native formal matrix has not yet been
generated.  This unfavorable development boundary must remain visible until a
complete replacement result exists.
