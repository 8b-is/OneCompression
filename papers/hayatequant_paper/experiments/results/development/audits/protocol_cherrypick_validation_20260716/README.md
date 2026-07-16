# Protocol hardening integration validation (2026-07-16)

This repo-local record validates the conflict-free integration of benchmark
protocol hardening commit `1fa7607b2e85636d342cff7b50fcfcc9ef5c1666`
as canonical commit `8d2c073595de29a63dda4161792fad3b82c054f0`.

The validation ran `git diff --check`, shell syntax checks for the submitter
and worker, Python compilation for the lifecycle helper and all changed
evaluation loops, and the following focused regression files:

* `test_eval_correctness.py`
* `test_longcontext_protocol.py`
* `test_ruler_eval.py`
* `test_submit_hayatequant_paper.py`
* `test_hayatequant_paper_shards.py`

Outcome: **283 passed, 0 failed**, with 17 dependency warnings, in 67.90
seconds.  `stdout.log`, `stderr.log`, `exit_code.txt`, and `junit.xml` retain
the complete machine-readable and textual result.  No GPU measurement and no
paper edit occurred in this integration check.

The originating audit additionally passed 483 tests and is recorded in
`benchmark/kvcomp-codec/benchmark_protocol_audit_20260716.json` and
`benchmark_protocol_audit_tests_20260716.json`.
