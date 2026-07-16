# HQ quality refinement preregistration

Frozen before any refinement-candidate score was produced on 2026-07-16 JST.
This is a development-only selection panel.  Formal paper inputs, formal
outputs, renderers, submission scripts, and held-out/task evaluation are out of
scope and must not be read by this search.

## Fixed HQ candidates

The public method remains `HQ`.  The identifiers below are archive-only IDs,
not additional paper methods.  Rates 3 and 4 remain byte-for-byte the current
canonical two-coordinate Gaussian/exact HQ recipe.  Only rate 2 is varied:

1. `DEVONLY-HQ-b4-gaussian-auto`: block size 4, K10/V6, Gaussian source,
   shipped high-budget artifact when present, exact assignment.
2. `DEVONLY-HQ-b4-gaussian-fibinit`: block size 4, K10/V6, Gaussian source,
   forced FibQuant spherical/radial warm start followed by the existing HQ
   Lloyd refinement, exact assignment.
3. `DEVONLY-HQ-b4-finite-beta`: block size 4, K10/V6, exact finite-head-dimension
   spherical Beta source, existing HQ Lloyd codebook, exact assignment.

The current b=2 K5/V3 Gaussian/exact recipe is the retention control.  No
candidate depends on model identity, layer identity, task, or observed score.
TurboQuant-paper and the current public HQ are rerun beside the rate-2
candidates so latency and deterministic quality can be compared within the
same worker process; the finalized job-69320 FibQuant raw remains the
independent quality-only reference.

## Adaptive HQ candidates

The public method remains `AdaptiveHQ`.  Every archive-only candidate has the
same predeclared two-coordinate candidate set at average rate `r`:

- symmetric K/V pair `(2r, 2r)`;
- fixed-HQ pair `(2r+1, 2r-1)`.

Thus rate 2 can select only K4/V4 or K5/V3; K7/V1 and every other extreme split
are impossible.  Exactly one selector bit is embedded in the existing first
fp16 value-scale fraction, adding zero logical stored bits.  The only searched
global scalar is the key/value error-weight ratio, predeclared as `1:1`, `4:1`,
or `16:1`.  It is shared by all seven checkpoints, all layers, all prompts, and
all three rates.

## Development panel and decision rule

- Checkpoints: Llama-2-7B, Llama-3.1-8B-Instruct, Qwen3-4B, Qwen3-8B,
  DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-1.5B, and
  DeepSeek-R1-Distill-Qwen-7B from the pinned local model root.
- Data: the same pinned WikiText-2 raw test concatenation already used by
  development job 69320, one 512-token window, stride 512.  This capped window
  is a development endpoint, not a formal-paper result.
- Rates: 2/3/4 for AdaptiveHQ; rate 2 for fixed-candidate measurement, combined
  with the unchanged job-69320 HQ measurements at rates 3/4.
- Comparison: strict unrounded JSON-double NLL (`candidate_nll < baseline_nll`).
  Ties and non-finite/missing cells are not wins and fail validation.
- Every token denominator, dataset identity, exact integer fp16/stored-bit
  counter, encoder contract, candidate set, carrier width, and split histogram
  is validated before ranking.
- Fixed-HQ rank is lexicographic: TQ wins, FibQuant wins, all external-baseline
  wins, rate-2 wins over current HQ, then lower summed NLL delta.  The current
  HQ retention control participates; a measured candidate is adopted only if
  it does not reduce the current TQ or total-external win count.  Rates 3/4 are
  unchanged by construction.
- AdaptiveHQ rank is lexicographic: TQ wins, all-external wins, wins over fixed
  HQ, then lower worst and summed NLL deltas.  All candidates must satisfy the
  two-pair/one-carrier-bit/no-rate-overrun contract.

All candidates and all negative speed/quality results are retained.  There is
no per-model, per-rate, per-task, or post-hoc cell selector.

## Pre-result native-eligibility amendment

The b=4 K10/V6 rows exceed the default nine-code-bit safety cap and do not use
the specialized b=2 native attention path.  A quality win alone is therefore
insufficient for canonical adoption.  The analyzer must also establish exact
K10/V6 stored-rate/layout representability, deterministic codebook hashes
across workers, finite encode/decode output, exact assignment, and same-run
latency versus current HQ/TQ.  It records that the native generic path is used.
If the cap exception/native path is not explicit and tested, or if latency is
a fatal regression, the row remains quality evidence only and public HQ stays
on the current b=2 recipe.  No quality result may silently change the native
recipe.
