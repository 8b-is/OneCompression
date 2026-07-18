# Core cache and evaluation audit (2026-07-16)

This directory is the repo-local evidence bundle for the independent audit of
`onecomp/kvcomp/cache/**`, the compressed-payload contract, and the evaluation
paths used for PPL, zero-shot accuracy, LongBench-E, RULER, LongCodeQA, code,
and reasoning benchmarks.  It records source hashes, regression results,
machine-readable findings, and a credential-shape scan.  Tests ran against the
shared uncommitted working tree rooted at commit
`e50379c225259b3e5d5df1facafa7133b36915d5`.

No abstract, introduction, paper claim, or benchmark score was edited as part
of this audit.  This bundle establishes implementation and metric contracts;
GPU benchmark results are archived by the corresponding benchmark runs.

## Decision summary

| Area | Status | Decisive correction |
|---|---|---|
| Cache bit accounting | fixed | batch multiplicity is included for graded, PMKVQ, and TurboTaper payloads |
| Cache lifecycle | fixed | reset clears cadence, delta, classifier, and accounting state |
| Unsupported semantics | fail closed | model-side sliding composition and ThinkV batch greater than one can no longer silently produce a mislabeled result |
| Compression accounting | fixed | exact counters, FullKV, and legacy ratio-only observations are distinct representations |
| Aggregate memory | fixed | exact memory fraction is `sum(stored_bits) / sum(fp16_bits)`, not a mean of per-example ratios |
| Zero-bit payload | fixed | positive fp16 baseline with zero stored bits reports infinite compression consistently |
| Official provenance | fixed | LongCodeQA completeness requires the pinned revision, per-bracket SHA, protocol commit, and paired generation budget |
| Metric inputs | fixed | malformed logits, empty prompts/references, invalid selections, non-finite values, and impossible counters fail closed |

## Metric contracts verified

- WikiText PPL uses token-weighted negative log likelihood.  A codec declaring
  sequence-coupled statistics is evaluated with causal prefix/incremental
  teacher forcing, so a future token cannot alter an earlier quantized state.
- Zero-shot multiple-choice accuracy retains both raw and length-normalized
  likelihood decisions, validates choice/context alignment, and records exact
  sample selection.
- LongBench-E is the unweighted macro over per-task scores.  Its exact cache
  memory field now pools counters before division.
- RULER uses the official per-example substring scorer and the unweighted
  13-task macro only when all official tasks and 100 rows per task are present;
  subsets remain explicitly non-official.
- LongCodeQA reports micro exact-match accuracy when disjoint bracket results
  are pooled.  An arbitrary revision/hash can no longer claim an official
  complete bracket.
- Code and reasoning evaluators use the same zero-bit and exact-counter
  semantics as the common accounting layer.

## Important remaining scope labels

- ThinkV evaluation supports batch size one.  Larger batches now raise instead
  of silently falling back to an uncompressed cache.
- The audited compressed-cache wrappers reject composition with Hugging Face
  model-side sliding layers where their update contract cannot return the
  transient full state required by the current attention call.
- A cache exposing only a historical ratio can still be shown diagnostically,
  but it is labelled `legacy_cache_ratio_only`; it is not exact accounting and
  is never relabelled FullKV.
- Synthetic long-context diagnostics that have no exact bit ledger keep an
  explicitly labelled legacy display aggregation.  Formal result axes use
  exact pooled counters.

## Verification

- Final focused correctness suite: 146 passed.
- Full `test_kvcomp.py` regression: 291 passed, 12 skipped.
- Formal RULER aggregate and renderer compatibility: 82 passed.
- Syntax compilation, scoped `git diff --check`, and focused Black check pass.
- Credential-shape scan: zero matches.

See `audit.json` for the findings, `test_results.json` and `test.log` for exact
commands/results, `source_manifest.json` for source hashes, and `SHA256SUMS`
for artifact integrity.
