# HQ fixed-recipe development panel (2026-07-16)

This directory is the immutable, repository-local evidence used to freeze one
HQ recipe before the formal paper run.  It is development evidence, not a
formal test-set result and not a per-model selector.

- Slurm array: `69257_[0-6]`, one B200 per checkpoint.
- Panel: seven predeclared checkpoints at average payload rates 2, 3, and 4.
- Primary development endpoint: WikiText-2 point-estimate PPL over eight
  overlapping windows (`max_length=512`, `stride=256`).
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Concatenated test-text SHA-256:
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`.
- Logical stored bits, fp16 reference bits, evaluated token denominator, and
  dataset provenance were required to match TurboQuant in every cell.
- Repository HEAD at launch: `e50379c225259b3e5d5df1facafa7133b36915d5`.
  The run intentionally preceded the clean formal freeze; the exact
  development wrapper, Slurm script, analyzer, raw JSON, and stdout/stderr are
  therefore stored here and content-addressed by `SHA256SUMS`.
  `PROVENANCE_LIMITATIONS.md` records the historical artifacts that were not
  captured and prevents this dirty-tree panel from being mistaken for formal
  evidence; `SLURM_ACCOUNTING.txt` records the recovered scheduler accounting.

The frozen speed-conscious HQ recipe is one global rule:

- two-coordinate blocks at every rate;
- Gaussian data-free source codebooks and rate-aware/folded rotation;
- fixed K/V payload rates `r+0.5` and `r-0.5` bits per coordinate, i.e. exact
  block-code lengths K5/V3, K7/V5, and K9/V7 at average rates 2, 3, and 4;
- key norm-correcting scale and value one-step LS refit;
- one fp16 scale per vector and no calibration data.

This candidate had lower unrounded PPL than `TurboQuant-paper` in 21/21 cells.
The K10/V6 rate-2 candidate also won 21/21 but was rejected because its public
encoder was materially slower.  `analysis.json` records every candidate and
cell; no result was discarded from the stored panel.

Formal full-text PPL, zero-shot, long-context, reasoning, code, and native
speed results are stored in separate repository-local directories after their
clean-commit runs and must not be conflated with this development panel.
