# External codec mathematical and accounting audit (2026-07-16)

This directory records an independent, source-pinned audit of the external
quality baselines used by the HayateQuant paper.  It covers only TurboQuant,
FibQuant and BlockQuant; it does not redefine HQ or AdaptiveHQ.  All rates
below are logical fixed-width data-plane rates, not the physical size of the
unpacked Python research tensors.

## Decision summary

| Row | Scientific status | Allowed formal use | Main limitation |
|---|---|---|---|
| `TurboQuant-paper` | Algorithm 1 MSE implementation matches the source when `dist=beta`, `rotation_mode=dense`, `use_qjl=false` | quality and TQ serving-speed comparisons at integer rates | the paper's mixed 2.5-bit arithmetic and outlier-selection protocol are under-specified |
| `FibQuant-paper` | method-faithful exact-search implementation of the v1 integer recipes | quality comparison only; `paper_comparable=true`, `paper_reproduction=false` | no official centroid artifact/code or seed/tie protocol was published |
| `BlockQuant-paper` | deterministic method-level completion of the v1 objective | include its predeclared quality cells in the comparison denominator; no speed row; `paper_comparable=false`, `paper_reproduction=false` | centroids, LUT, tails, K/V scale mapping, outlier lifecycle and seed protocol are not reproducible from the paper |

No author-linked official implementation is present in any of the three arXiv
records or source archives.  Independent/community implementations were not
used as normative evidence.

## Pinned primary sources

- TurboQuant: [arXiv:2504.19874v1](https://arxiv.org/abs/2504.19874) and the
  [ICLR 2026 OpenReview record](https://openreview.net/forum?id=tO3ASKZlok).
- FibQuant: [arXiv:2605.11478v1](https://arxiv.org/abs/2605.11478).
- Block-Sphere Vector Quantization:
  [arXiv:2605.19972v1](https://arxiv.org/abs/2605.19972).

Exact source/archive digests are in `source_manifest.json`.  The arXiv TeX,
not a search-result summary or third-party implementation, was used for every
equation and protocol decision.

## TurboQuant

### Algorithm 1 MSE path

The canonical formal row is exact at the method level:

1. a Gaussian matrix is QR-factorized and the diagonal signs of `R` are
   folded into `Q`, producing a Haar orthogonal matrix;
2. each vector is L2-normalized;
3. the exact finite-dimensional marginal
   `f(x) proportional to (1-x^2)^((d-3)/2)` is quantized by a converged scalar
   Lloyd-Max codebook;
4. encoding uses nearest scalar centroids and decoding uses the transpose
   rotation.

`dist=gaussian` and `rotation_mode=hadamard` are explicit deployment
approximations, not the formal paper row.  The measured d=128 1-bit centroid
is `0.0706615746` in magnitude, consistent with the paper's asymptotic
`sqrt(2/pi)/sqrt(d)` value; the d=128 2-bit magnitudes are `0.0399915949` and
`0.1330415159`, consistent with the paper's quoted Gaussian-limit constants.

### Algorithm 2 QJL path

The implementation uses

`sign(S r)` and `sqrt(pi/2)/d * ||r||_2 * S^T sign(S r)`

with an i.i.d. standard-normal `d x d` projection.  To obtain a nominal target
of `b` payload bits per coordinate, the scalar MSE stage must be constructed
with `bits=b-1`; the remaining bit is the QJL sign.  The logical ledger also
contains the original fp16 norm and one fp16 residual norm per vector.  The
paper's unbiasedness theorem assumes an exact real residual norm.  A stored
fp16 residual norm introduces header-rounding error, so serialized output is
not labelled exactly unbiased.  QJL is never substituted into the formal
Algorithm 1 MSE baseline.

### 2.5/3.5-bit mixed precision

The v1 source prints

`(32*3 + 96*2)/128 = 2.5`,

but the left side is `2.25`.  Two fp16 subvector norms contribute another
`32/128=0.25` bits per coordinate, which is the only reconciliation using the
fields stated by the paper.  The analogous repository completion
`32@4 + 96@3 + two fp16 norms` is 3.5 bpc, but the source only says that the
3.5-bit row uses a different ratio and does not publish that ratio.

The source also does not define how channel identities are selected or sent
to the decoder.  This research implementation selects one global role mask
per public `encode` call by
`mean_(batch,head) max_token(abs(x))` and charges one `d`-bit bitmap.  Thus, for
`d=128`, `V=B*H*S` vectors in one role payload, the realized rates are
`2.5 + 1/V` and `3.5 + 1/V` bpc respectively.  This is deliberately not
described as BlockQuant's per-head top-L2 rule.

### Correctness fixes from this audit

- reject NaN/Inf K/V before `searchsorted` can silently assign a code;
- reject fp16 scale or QJL-norm overflow instead of emitting Inf payloads;
- normalize every nonzero vector by its exact computed scale instead of a
  fixed `1e-8` clamp, restoring scale-invariant code assignment;
- reject non-boolean `use_qjl` and mismatched K/V shape, device or dtype.

## FibQuant

The implementation matches the v1 method where the source is complete:

- `2 bpc -> (k=4,N=256)`, `3 bpc -> (k=2,N=64)`, and
  `4 bpc -> (k=2,N=256)`;
- Beta-quantile radii, Fibonacci directions at `k=2`, and the
  Roberts-Kronecker construction at `k>=4`;
- `M=30N` spherical-Beta training samples, four randomly rotated restarts,
  25 Lloyd iterations, exact nearest-codeword encoding, one shared Haar
  rotation, and one fp16 vector norm;
- per-role stored bits are exactly
  `num_vectors * (d/k * log2(N) + 16)`.

The paper publishes neither centroids nor code, and leaves training/restart
seeds and the exact empty-cell split/tie rule unspecified.  The repository
therefore pins a deterministic seed-0 completion and records codebook hashes:

| Rate | Recipe | d=128 codebook SHA256 |
|---|---|---|
| 2 | k=4, N=256 | `097a7fe987ca4092b653480bec23f07ed1470694bd911e1544f67966bf5bb2e1` |
| 3 | k=2, N=64 | `27f2806eaf4a4b7d7c05654fca7e4cb93dbcfad5a96b4820cccb3cdfbd1b8b75` |
| 4 | k=2, N=256 | `3b74a174a348e46a6e82c931b65c9adbc3739c504a37549b7d7755b72a6bfdc2` |

These rows are method-comparable but not byte-for-byte or score-level paper
reproductions.

## BlockQuant

The source-defined mathematics is implemented correctly: the finite-sphere
block marginal, `2^(b*p)` centroids, exact nearest-centroid assignment, and
the raw, BSM and UB scale equations all match Algorithm 1.  Exact assignment
is appropriate for a quality-only baseline, but it is not the paper's
experimental approximate LUT encoder.

The source explicitly declares, for Llama-3.1 head dimension 128, keys with
32 channels at 4 bits and 96 at 3 bits plus two fp16 scales (`3.5` conditional
bpc), and values with a 2-bit approximation.  It also explicitly repeats the
LLM experiments over five random seeds and reports mean with standard
deviation.

The following material choices are not published:

- centroid values or fitting code/sample count/restarts/stopping/seeds;
- LUT grid size and candidate-list size;
- the p=3 treatment of 32- and 128-dimensional remainders;
- which of raw/BSM/UB is assigned to K and V;
- channel-identity lifecycle/coding and rotation-sharing/role-seed derivation.

The repository pins these as `onecomp.blockquant.finite_beta_lloyd.v1`, exact
search, `lower_dimensional_tail_v1`, key UB/value BSM, and dynamic
per-(batch, head, encode-update) top-L2 identities encoded by a fixed-width
combinadic rank.  At `(B,H,S,d)=(1,1,1,128)` this gives 549 key bits
(`416 code + 32 scale + 101 identity`, 4.2890625 bpc) and 272 value bits
(`256 code + 16 scale`, 2.125 bpc).  At `(1,8,128,128)` the key rate
amortizes to 3.50616455078125 bpc; the value rate remains 2.125 bpc.

The formal row is a fixed seed-0 point estimate, not the paper's five-seed
aggregate.  It must remain in the predeclared quality comparison denominator,
but must remain `paper_comparable=false`, `paper_reproduction=false`, and be
excluded from speed comparisons.

## Verification

- External-codec focused suite: 71 passed.
- Existing TurboQuant/Lloyd regression subset: 59 passed, 244 deselected.
- All commands, exit status and wall times are recorded in `test_results.json`.
- Machine-readable findings and exact stored-bit examples are in `audit.json`.
- `SHA256SUMS` covers every audit artifact except itself.

