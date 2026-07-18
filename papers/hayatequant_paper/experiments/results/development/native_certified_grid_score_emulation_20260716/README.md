# Rejected scalar score emulation (development only)

Status: **rejected**.

The experiment asked whether two explicit fp32 products plus one fp32 sum,
followed by the canonical fp16 score boundaries, reproduce the accepted
tensor-core assignment.  They do not.  Across 1,000,008 fp16 queries per
cell, mismatch counts for rate 2 K/V, rate 3 K/V, and rate 4 K/V were
respectively `1/0`, `2/1`, and `5/2`.  The four rare queries retained from the
earlier rejected grid implementation all reproduced their original
mismatches.

Consequently, scalar candidate scoring is not an exact replacement for the
public fp16 GEMM and must not be promoted.  `result.json` retains all observed
mismatch examples and source/table hashes.  `controller.stdout` is
byte-identical to the result, `controller.stderr` is empty, and the launcher
exit status was 0.

The next admissible diagnostic must use the same tensor-core MMA path for a
per-row candidate set; no quality or speed result is inferred from this run.

