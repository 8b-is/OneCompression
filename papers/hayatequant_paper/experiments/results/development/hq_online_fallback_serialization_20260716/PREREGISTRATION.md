# Exact current-HQ online-fallback serialization: frozen protocol

Frozen on 2026-07-16 before any reference tensor in this round was written.
This is not a quality search and is disjoint from the rejected high-budget
bundle in `hq_codebook_artifacts_20260716`.

## Immutable target and constructor

Serialize exactly the current canonical b=2 asymmetric Gaussian codebooks:
K8, K32, K128, and K512. For each target, call the existing
`lloyd_vq_gaussian_codebook` fallback with its unchanged defaults: 100,000 iid
N(0,1) fp32 samples, at most 25 Lloyd iterations, 4 restarts, seed 0,
`init="auto"`, and ambient/head dimension 128. The first restart therefore
uses the existing FibQuant geometric warm start and the remaining restarts use
the existing seeded initialization. All seven development checkpoints have
head dimension 128.

The first generated tensor is the designated reference. There is no MSE/PPL
ranking and no restart/budget/target cherry-picking. An independent
fresh-process audit regenerates the same constructor output without loading
the reference and must match its raw fp32 tensor SHA-256 exactly. Any mismatch
rejects serialization; it is not resolved by selecting the better tensor.

## Exact-output and installation gate

Before installation, isolated online and staging-artifact processes must match
for all four codebook tensor hashes and public fixed-HQ payloads at rates 2/3/4:
key/value codes, fp16 scales, stored-bit counters, decoded output at its exact
dtype, and resolved K/V recipe. Safe AdaptiveHQ with only symmetric/fixed-HQ
pairs and global weight 4:1 must receive the same check, including selected
pair, selector carrier, candidate set, payload, and decode.

The staging tensors may be installed only as one four-file bundle after both
independent regeneration and payload parity pass. Installation must preserve
the reference file/content SHA and create a pinned shipped-artifact manifest.
No WikiText result is rerun or selected: exact payload parity preserves the
already archived current-HQ and safe-AdaptiveHQ measurements.

## Formal fail-closed boundary

The public codec remains capable of an online fallback for explicit
development/custom configurations. Canonical paper `HQ` and `AdaptiveHQ`
recipes, however, must require the shipped manifest. Missing file, wrong
original dtype/shape, non-finite or duplicate rows, manifest/content SHA
mismatch, or an unapproved environment override must fail closed instead of
silently refitting. The `KVCOMP_CODEBOOK_DIR` override remains development-only
and must be exposed in resolved provenance; formal preflight continues to
reject it.

Corruption, missing-file, wrong-dtype, non-finite, duplicate-row, wrong-SHA,
development-fallback compatibility, and online/artifact payload-parity tests
are required before production adoption. No `__pycache__` is an artifact.

If the bundle passes, the valid fixed formal method remains unchanged b=2 HQ.
The selected opt-in adaptive policy is separately the safe two-pair policy
`[(2r,2r),(2r+1,2r-1)]`, exactly one in-budget selector bit, and global
`wK:wV=4:1`; its policy version, formal config, tests, and paper text must be
updated consistently. If any gate fails, production/formal recipes remain
unchanged and no artifact-backed claim is valid.

Frozen source: git HEAD `e50379c225259b3e5d5df1facafa7133b36915d5`;
`vq.py` SHA-256
`f8a0602ea4c852c2c765c9dc59a07d3c79a209b93a17c854dc51ad4791765c2d`.
