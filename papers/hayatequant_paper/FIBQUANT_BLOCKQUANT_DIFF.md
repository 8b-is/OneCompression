# HayateQuant, FibQuant, and BlockQuant: implementation scope

This note distinguishes the independent quality baselines from ingredient
controls inside HayateQuant. The registry now contains independent
`FibQuant-paper` and `BlockQuant-paper` codecs. The separate
`HQ-Fib-init-proxy` and `HQ-Block-scale-proxy` keys remain HQ-only component
ablations; historical JSON labels `FibQuant`, `HayateQuant-fibinit`,
`BlockQuant`, and `HayateQuant-BQscale` are accepted only as legacy read aliases
for those old proxies and can never populate an external-baseline slot.

## Executive summary

`FibQuant-paper` follows the published spherical-Beta/codebook method with exact
assignment, but the paper does not release official fitted centroids or every
seed, so this is a method-faithful comparison rather than byte-identical artifact
reproduction. `BlockQuant-paper` implements the fixed Llama-3.1 source operating
point with explicitly versioned deterministic completions for unpublished
identity/top-channel lifecycle details. It is reported descriptively because
its identity-bit-counted payload is not exactly rate matched to HQ.

This is the authoritative live denominator policy. It supersedes only the
historical BlockQuant-denominator assertion in
`experiments/results/development/audits/external_codec_math_audit_20260716`;
the immutable audit evidence itself is preserved. The machine-readable and
content-hashed attestation is in the sibling
`external_codec_math_audit_20260716_policy_supersession` directory. Labels
2/3/4 are code-payload rate slots, not a promise that all metadata-inclusive
stored rates are integers. At head dimension 128, matched TQ/FibQuant/HQ rows
include the same `0.125` bpc fp16-scale overhead; BlockQuant's dynamic identity
field makes its exact slot-3 stored rate update-length dependent.

The Fib-inspired proxy changes only HayateQuant's deterministic radial-angular
codebook warm start (Beta-companded radii plus low-discrepancy directions,
followed by HQ's Lloyd refinement).  `codebook_init="fibquant"` retains the
HayateQuant block-VQ payload, scale stack, rotation, and encoder; it must not be
reported as FibQuant.

## Fib-inspired initialization inside HQ

FibQuant normalises by L2, samples a Haar-rotated unit vector, splits into
blocks, and designs a radial-angular codebook for the exact projected
spherical-Beta marginal.  HayateQuant normalises by RMS and learns a block
codebook by Lloyd/K-means on synthetic samples.

Imported now:

* k=2: Fibonacci spiral directions on S1 plus the closed-form FibQuant radius.
* k=3: Fibonacci sphere directions on S2.
* k>=4: Roberts-Kronecker directions mapped through the normal inverse and
  projected to the sphere.
* radii: FibQuant Bennett-Gersho Beta companding, scaled by `sqrt(d)` to match
  HayateQuant's RMS convention.

Not yet imported:

* arbitrary `(block_dim, codebook_size)` profiles;
* sub-one-bit / dense fractional-rate axis;
* radius-first or tree/list encoders for very large codebooks.

## Block-inspired scale controls inside HQ

BlockQuant and HayateQuant share the same mathematical spine: rotate,
normalise, block-quantise, then apply a scalar correction.

Naming map:

* BlockQuant BSM scale = HayateQuant LS scale.
* BlockQuant UB scale = HayateQuant IP scale.
* HayateQuant NC scale is a KV-specific compromise between LS and IP:
  `s_nc = sqrt(s_ls * s_ip)`.

The HQ scale implementation exposes terminology aliases:

* `key_scale_mode="ub"` is equivalent to `key_scale_mode="ip"`.
* `value_scale_mode="bsm"` is equivalent to `value_scale_mode="ls"`.

BlockQuant's KV experiment protects key outlier channels while quantising
values more aggressively.  HayateQuant previously had only a shared
`n_outlier` for K and V.  The codec now supports:

* `key_n_outlier`, `key_outlier_bits`
* `value_n_outlier`, `value_outlier_bits`

Example HQ Block-recipe proxy profile:

```yaml
{bits: 3, value_bits: 2,
 key_n_outlier: 32, key_outlier_bits: 4, value_n_outlier: 0,
 key_scale_mode: ub, value_scale_mode: bsm}
```

The fixed HQ proposal uses `block_size=2`, Gaussian source codebooks, exact
full-codebook assignment, key NC, value LS-reassign/refit, and block-code pairs
K5/V3, K7/V5, and K9/V7 at average rates 2, 3, and 4. These settings come from
`hayate_hq_kwargs`; benchmark configs must not depend on constructor defaults.
The old `HayateQuant-frontier` K=`r+1`/V=`r-1`, key-UB recipe remains only a
legacy appendix ablation. AdaptiveHQ is the sole opt-in proposal and selects
causally between the predeclared symmetric and fixed-HQ pairs at the same rate.

## Is optional calibration a delta?

Yes.  Optional calibration with a never-worse selection rule is a HayateQuant
extra relative to both FibQuant and BlockQuant's core data-free algorithms.
The guarantee is intentionally narrow: the calibration loop keeps the best
artifact on a held-out calibration shard, initialised from the data-free
rotation/codebook.  It means the saved artifact is never worse than the
data-free starting point on that holdout objective.  It is not a universal
guarantee over all prompts, all layers, or downstream softmax behavior.

## New reproduction hooks

Formal proposal and independent-baseline keys:

* `HQ`
* `AdaptiveHQ`
* `FibQuant-paper` (quality only)
* `BlockQuant-paper` (quality only, descriptive unmatched source point)

Component/legacy ablations:

* `HayateQuant-frontier`
* `HayateQuant-selected`
* `HQ-Fib-init-proxy`
* `HQ-Block-scale-proxy`

Fixed `HQ` resolves to:

* 2-bit target: K5/V3 block-code bits + key NC + value LS-reassign/refit.
* 3-bit target: K7/V5 block-code bits + key NC + value LS-reassign/refit.
* 4-bit target: K9/V7 block-code bits + key NC + value LS-reassign/refit.

Long-context config:

* `benchmark/kvcomp-longcontext/conf/hayatequant_fib_blockquant.yaml`

## Further improvement candidates

1. Add `index_bits` independent of `bits` so HayateQuant can express
   FibQuant-style fractional rates such as 6 index bits per 8-D block.
2. Add shell/tree encoders for large codebooks.  FibQuant's radius-first
   search is a better scaling path than a dense grid LUT when `K` grows.
3. Add clamped UB/IP scale for keys.  BlockQuant's UB estimator reduces
   expected inner-product bias but can be high-variance when alignment is
   small; a clipped ratio may dominate raw UB through softmax.
4. Add fixed model-level outlier masks so calibration can compose with
   outlier splits.  Current calibration rejects outliers because subset dims
   change per tensor.
5. Compute HayateQuant block-size constants beside BlockQuant's p=2/p=3
   constants, both on the spherical-Beta law and on RMS-Gaussian samples.
