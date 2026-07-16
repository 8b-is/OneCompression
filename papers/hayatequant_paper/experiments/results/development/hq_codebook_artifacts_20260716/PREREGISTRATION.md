# HQ fixed Gaussian artifact round: frozen amendment

Frozen on 2026-07-16 at 06:00 JST, before any artifact in this round was
generated or scored. This is a separate development round and does not modify
the completed `hq_quality_refinement_20260716` archive or any formal result.

## Purpose and immutable bundle

The canonical fixed HQ story remains the paper's short b=2, role-aware,
data-free block VQ. Its asymmetric codebooks were previously missing shipped
artifacts and therefore fell back to an online fit. This round creates one
globally shared, model-independent Gaussian bundle:

- b=2: K = 8, 32, 128, and 512 (fixed-HQ V3/K5, V5/K7, V7/K9);
- b=4: K = 64 and 1024 (the already measured rate-2 V6/K10 candidate).

The same six tensors must be used for every model, layer, rate, prompt, and
task. There is no per-model, per-role, per-rate, or per-cell fallback/choice.
Existing symmetric b=2 K16/K64/K256 artifacts remain part of AdaptiveHQ's
bundle and are not refit in this round.

## Frozen fitting and source-law audit

Every target uses only synthetic iid N(0,1) fp32 samples. Fit seed is 0;
held-out seed is 123. Lloyd K-means uses at most 60 iterations and exactly 8
predeclared restarts (the first is not selected post hoc across target
budgets). The target-specific sample count is the existing repository rule
`min(1,600,000, max(200,000, floor(400,000,000/K)))`:

| Target | Fit samples | Held-out samples |
|---|---:|---:|
| b2 K8 | 1,600,000 | 1,000,000 |
| b2 K32 | 1,600,000 | 1,000,000 |
| b2 K128 | 1,600,000 | 1,000,000 |
| b2 K512 | 781,250 | 1,000,000 |
| b4 K64 | 1,600,000 | 500,000 |
| b4 K1024 | 390,625 | 500,000 |

The existing `_kmeans_best_of` implementation, fp32 CUDA matmul/update path,
TF32 disabled, and no activation data or learned model weights are used. Early
termination by the implementation's frozen `allclose(atol=1e-6)` condition is
allowed. GPU fitting need not be rerun bitwise; the resulting tensor is the
immutable artifact and both tensor-content and file SHA-256 are recorded.

Each artifact must be finite fp32 with exact `(K,b)` shape, K unique rows, and
strictly lower held-out Gaussian MSE than the explicit current online-fallback
fit on the same held-out panel. Any target failure rejects the whole six-file
bundle; failed files and metadata remain archived.

## Frozen seven-model quality panel

The exact pinned WikiText-2 identity, seven local checkpoints, rates 2/3/4,
one 512-token window, exact encoder, payload counters, and strict unrounded NLL
rule are inherited from `hq_quality_refinement_20260716`. Three proposals are
measured with no additional search:

1. artifact b=2 fixed HQ at every rate;
2. artifact b=4 K10/V6 at rate 2 plus artifact b=2 fixed HQ at rates 3/4;
3. artifact safe AdaptiveHQ with only symmetric/fixed-HQ pairs and the already
   selected global key:value error weight 4:1.

Current online HQ, current safe AdaptiveHQ 4:1, TQ, FibQuant, and BlockQuant
are immutable controls from the completed archive; current HQ and TQ are also
rerun without the staging artifact directory as a determinism check.

The b=2 artifact HQ bundle is eligible only if all fit gates pass, no external
family has fewer wins than current HQ (TQ 18/21, FibQuant 16/21, BlockQuant
4/7), total external wins are at least 38/49, and summed NLL delta versus
current online HQ across 21 model-rate cells is non-positive. If it fails,
this round does not silently return to an online-fit formal result: production
and formal adoption remain unchanged/withheld and the negative bundle is
retained.

Artifact AdaptiveHQ is eligible only if all fit gates pass, its external
family counts are no lower than the selected pre-artifact safe policy (TQ
20/21, FibQuant 17/21, BlockQuant 3/7; total 40/49), and its summed NLL delta
versus that safe policy is non-positive. Otherwise no artifact-backed
AdaptiveHQ is adopted in this round; there is no per-cell online fallback.

The b=4 candidate can supersede eligible b=2 only if it is lexicographically
better on the already frozen quality rank and then strictly speed-Pareto
dominates b=2 on all native ABBA operations. The requested 90% target is
reported as 45/49; it is not manufactured by changing denominators or ties.

## Native and later assignment-speed gates

The final immutable bundle must rerun public/native payload parity, 68-byte
physical-rate parity, and fresh-process ABBA. The prior b=4 generic kernel
regression is retained, not erased by codebook refitting.

Only after the artifact quality decision may a separate assignment-speed
candidate be preregistered. Its allowed ideas are explicit fp32 squared
distance with smallest-index ties, exact fallback for near-margin rows, or a
distance lower-bound completeness certificate. It must first pass all Qwen
rate-2 and all-rate bit-exact gates, then the TQ speed gate. No such candidate
is part of this artifact fit or quality selection.

Frozen source identity: git HEAD
`e50379c225259b3e5d5df1facafa7133b36915d5`; dirty source SHA-256:
`vq.py=f8a0602ea4c852c2c765c9dc59a07d3c79a209b93a17c854dc51ad4791765c2d`,
`codec.py=c5608752399a605a730996df6876a4f6b613bd167ddde45c21d579db224b043b`,
`recipes.py=04b28eb73af881e0f4369aa84ed8af2258629c347f2f5bd46f07ce705be0436d`.
