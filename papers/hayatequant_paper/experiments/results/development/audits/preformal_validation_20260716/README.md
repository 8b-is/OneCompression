# HayateQuant pre-formal validation (2026-07-16)

This repo-local bundle records the final CPU regression and static validation
performed before rerunning the HayateQuant paper smoke and formal matrices.  It
applies to the working tree based on
`3d6f264363d7291c1fd07fe40d310ae041e53425`; the bundle and validated source are
committed together so that the next smoke run can bind to a clean immutable
commit.

No paper abstract or introduction text was changed in this validation phase.
No benchmark point estimate in this bundle is a formal paper result.

## Outcome

- Final focused regression: **407 passed, 0 failed**.
- Final complete `tests/onecomp/kvcomp` regression: **1071 passed, 40 skipped,
  0 failed**.
- Black, Python compilation, shell parsing, `git diff --check`, and JUnit XML
  parsing all pass.
- An intermediate full-suite run is preserved rather than deleted.  Its 18
  failures all came from the synthetic RULER aggregate fixture lacking the new
  mandatory eight-job smoke-approval record.  After updating only that fixture,
  its focused suite passed 18/18 and the complete suite passed.

## Contracts added or tightened

- Formal workers, aggregation, and rendering are bound to one exact smoke
  approval, plan digest, code commit, input manifest, eight smoke results, and
  eight worker preflights.
- Approval file records require lowercase SHA-256, canonical absolute paths,
  positive byte counts, and unique paths for every smoke job.
- The renderer cross-checks the approved input-manifest byte count against the
  aggregate dataset ledger.
- Structured JSON archives reject credential-named fields even when a quoted
  secret is too short for entropy heuristics; explicit redaction placeholders
  remain allowed.
- MATH grading records the complete generated text consumed by Math-Verify and
  labels the legacy regex extraction as non-grading diagnostics.
- Non-eviction runs distinguish `eviction_applicable=false` and
  `eviction_fired=null`; exact FullKV aggregation uses sufficient statistics.
- Zero-shot planning retains BlockQuant at its applicable three-bit operating
  point, records the execution device, and rejects formal results not run on
  `cuda:0`.

## Files

- `focused-junit.xml`: final focused regression.
- `full-kvcomp-junit.xml`: final complete regression.
- `full-kvcomp-failed-missing-ruler-fixture-junit.xml`: preserved intermediate
  fixture-only failure.
- `static-checks.log`: exact static-validation invocations and output.
- `SOURCE_SHA256SUMS`: SHA-256 identities of all changed Python and shell files.
- `SOURCE_DIFF.sha256`: SHA-256 of their binary Git diff from the starting
  commit.
- `audit.json` and `test_results.json`: machine-readable scope and outcomes.
- `SHA256SUMS`: integrity ledger for this evidence bundle.

