# Per-row tensor-core candidate-score diagnostic (development only)

Status: **numeric score-path diagnostic passed; no encoder is promoted**.

A row-specific 16-codeword candidate set was evaluated through a padded
16x16x16 tensor-core MMA, preserving the same fp16 dot and correction
boundaries as canonical all-codeword HQ.  The accepted all-codeword winner was
inserted into each diagnostic list solely to isolate scoring parity from
candidate coverage.

Across 1,000,008 fp16 queries for each of the six fixed rate/role codebooks,
including every known near-tie from the rejected scalar grid attempt, the
candidate tensor-core path had zero mismatches.  The scalar fp32 emulation
again had 11 total mismatches, so it remains rejected.

This establishes only that a candidate list *can* be scored with the canonical
numeric path.  It does not establish that a deployable table contains the
winner, that the path is faster, or that it preserves full codes/scales/cache
bytes.  Those are separate fail-closed gates.  `result.json` and the
byte-identical controller stdout retain raw evidence; stderr is empty and the
launcher exit status was 0.

