# Formal runtime-path and dataset-binding validation (2026-07-16)

This repo-local audit records the correction of a formal-smoke integration
failure found by independent review of source commit
`8d2c073595de29a63dda4161792fad3b82c054f0`.

The worker correctly uses an external mutable `WORK_ROOT`, including
`HOME=$WORK_ROOT/runtime/home` and externally pinned datasets.  The old smoke
gate incorrectly reconstructed both under `RUN_ROOT`, so a scientifically
valid worker result could never receive formal approval.  The fix:

* binds canonical, non-symlinked `WORK_ROOT` and runtime `HOME` in the validated
  paper-input manifest;
* resolves HF, RULER, LongCodeQA, and context-fit files through logical
  selectors into the manifest's already validated absolute file records;
* re-hashes every selected file during smoke approval;
* advances the exact smoke plan contract from version 3 to version 4;
* requires write-producing preflight to use a validated linked results
  worktree and makes source-only `formal-plan` read-only.

The regression fixture deliberately places `RUN_ROOT`, `WORK_ROOT`, and pinned
dataset data in three disjoint directory trees.  It also rejects a historical
`RUN_ROOT/runtime/home` worker record, a wrong runtime home, symlink path
components, missing dataset records, and dry-run writes outside a results
repository.

Validation command:

```text
.venv/bin/python -m pytest -q --junitxml=.../junit.xml \
  tests/onecomp/kvcomp/test_hayatequant_paper_inputs.py \
  tests/onecomp/kvcomp/test_hayatequant_smoke_gate.py \
  tests/onecomp/kvcomp/test_submit_hayatequant_paper.py \
  tests/onecomp/kvcomp/test_hayatequant_paper_shards.py
```

Outcome: **197 passed, 0 failed**, 15 dependency warnings, in 50.21 seconds.
The machine-readable JUnit file is 33,104 bytes with SHA-256
`34098677daa9d1a5e6bbe197d58044b5ecc066d7fdceaa45a7940df8dbe0eeee`.
No GPU measurement and no paper edit occurred.
