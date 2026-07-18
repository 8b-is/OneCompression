# Baseline-policy reconciliation verification (2026-07-16)

This bundle records the CPU-only verification of the external-baseline policy
supersession rooted at commit
`8d2c073595de29a63dda4161792fad3b82c054f0`.

Scope:

- preserve every byte and the original manifest of
  `external_codec_math_audit_20260716`;
- bind the new Markdown/JSON policy attestation with its own SHA-256 manifest;
- derive the 164 matched-quality, 80 unmatched-BlockQuant, 60 TQ-speed, and
  776 total gate cardinalities from frozen code constants;
- distinguish public `TurboQuant-paper` Algorithm 1 from the upstream vLLM
  0.20 native comparator;
- verify the disclosed head-dimension-128 packed slot widths at rates 3 and 4.

No benchmark result was generated, inspected, or changed. No GPU, Slurm job,
remote host, or network service was used.

Verification result: 21 passed, 0 failed, 0 errors, and 0 skipped. The 16
warnings are dependency deprecations recorded verbatim in `pytest.log`.

Files:

- `commands.txt`: exact verification commands;
- `pytest.log`, `pytest-junit.xml`: focused regression-test output;
- `historical_bundle_sha256.log`: verification of the immutable old bundle;
- `supersession_sha256.log`: verification of the new policy bundle;
- `source_diff_check.log`: whitespace/error check for the patch;
- `test_result.json`: machine-readable test summary;
- `SOURCE_SHA256SUMS`: hashes of changed policy/code/docs/tests;
- `SHA256SUMS`: integrity manifest for this evidence bundle.

