# HayateQuant vs FibQuant / BlockQuant: current delta and import plan

This note records the design delta after reading the FibQuant paper/code and
the BlockQuant paper, then comparing them to the current HayateQuant code.

## Executive summary

HayateQuant was already closest to BlockQuant: it has block vector
quantisation, a rotated spherical/Gaussian source model, best-scalar style
scales, a LUT encoder, outlier mixed precision, and KV-specific role
asymmetry.  The missing BlockQuant surface was mostly terminology and the
paper's key/value-asymmetric KV recipe.

FibQuant is more different.  Its strongest import for HayateQuant is the
deterministic radial-angular codebook construction: Beta-companded radii plus
low-discrepancy directions, then Lloyd refinement.  HayateQuant now exposes
that through `codebook_init="fibquant"` while keeping the existing
HayateQuant block-VQ payload format and scale/rotation stack.

## Delta vs FibQuant

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

## Delta vs BlockQuant

BlockQuant and HayateQuant share the same mathematical spine: rotate,
normalise, block-quantise, then apply a scalar correction.

Naming map:

* BlockQuant BSM scale = HayateQuant LS scale.
* BlockQuant UB scale = HayateQuant IP scale.
* HayateQuant NC scale is a KV-specific compromise between LS and IP:
  `s_nc = sqrt(s_ls * s_ip)`.

The code now exposes BlockQuant aliases:

* `key_scale_mode="ub"` is equivalent to `key_scale_mode="ip"`.
* `value_scale_mode="bsm"` is equivalent to `value_scale_mode="ls"`.

BlockQuant's KV experiment protects key outlier channels while quantising
values more aggressively.  HayateQuant previously had only a shared
`n_outlier` for K and V.  The codec now supports:

* `key_n_outlier`, `key_outlier_bits`
* `value_n_outlier`, `value_outlier_bits`

Example BlockQuant-style profile:

```yaml
{bits: 3, value_bits: 2,
 key_n_outlier: 32, key_outlier_bits: 4, value_n_outlier: 0,
 key_scale_mode: ub, value_scale_mode: bsm}
```

## Is optional calibration a delta?

Yes.  Optional calibration with a never-worse selection rule is a HayateQuant
extra relative to both FibQuant and BlockQuant's core data-free algorithms.
The guarantee is intentionally narrow: the calibration loop keeps the best
artifact on a held-out calibration shard, initialised from the data-free
rotation/codebook.  It means the saved artifact is never worse than the
data-free starting point on that holdout objective.  It is not a universal
guarantee over all prompts, all layers, or downstream softmax behavior.

## New reproduction hooks

Codec-level ablations:

* `HayateQuant-fibinit`
* `HayateQuant-BQscale`

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
