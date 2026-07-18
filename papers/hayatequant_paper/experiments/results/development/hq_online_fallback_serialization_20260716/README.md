# Exact current-HQ online-fallback serialization

Status: **complete; every preregistered gate passed; bundle adopted**.

This was a serialization/parity round, not a quality search. It freezes the
exact tensors produced by the historical default online constructor for the
four codebooks used by fixed b=2 HQ. No MSE, PPL, task score, model, rate, or
restart was used to choose among tensors. The protocol was frozen first in
`PREREGISTRATION.md` (SHA-256
`73f9aecea156efa8f44e6933fa73931e2ed0af2a9f47f2679c0974035140b359`).

## Outcome

| target | file SHA-256 | raw fp32 tensor SHA-256 | fresh-process regeneration |
|---|---|---|---|
| b2/K8 | `d0b4402fa70317c6a72a85915becc9fd2923d8815793039349f7844c3000947a` | `55d09c31d8bcfe4d597dff7021b85a6c9736b900dd3a298d214de9ec78333390` | bit-exact |
| b2/K32 | `2e81e59d367252758d4ffbbf151d798c379732ecc43a3f21fefedd7b6466ced3` | `5a6383c6742243ef42f6009e724b5a5bb55fa5b13c1ee93f7a9ce59b0d583926` | bit-exact |
| b2/K128 | `67b131bdea9045a8414ec43a81ecce806f87339a7a092ed19a2e5722506c7b51` | `39e5d9b4958c4935180bba5844dcc41a8f2dfa012e2d44cd4611da40491f43fb` | bit-exact |
| b2/K512 | `e931ace83ba25147a8dcdce9a822c5318650fc392e1fa84026c89cc89baabfed` | `97fd30ff4347c60a90e81560afb7833d98bb9c537004d9ba57f83592991bcce2` | bit-exact |

All reference tensors are original `torch.float32`, finite, have shape
`(K,2)`, and have exactly K unique rows. Independent jobs regenerated all four
raw tensors bit-for-bit.

`public_payload_parity.json` records `online_artifact_bit_exact=true` in fresh
processes. Inputs, all seven b2 codebooks reachable by the compared methods,
fixed HQ at rates 2/3/4, and safe AdaptiveHQ at rates 2/3/4 all match. The
comparison includes role codes, fp16 scales, stored-bit counters, decoded
outputs at their exact dtype, resolved recipes, adaptive candidate pairs,
selected pair, and one-bit selector carrier.

The three native staging checks in `native/` also report `ok=true`. For every
fixed-HQ rate, public/native key codes, value codes, fp16 scales, packed
stream, and full dequantization at output dtype are exact; store-one-token and
prefill payload bytes match. These are correctness/parity checks, not formal
cross-method speed results.

## Adopted immutable contract

The four tensors were copied without modification into
`onecomp/kvcomp/codec/hayate/codebooks/`. The shipped manifest and this
archive's `candidate_manifest.json` are byte-identical, with SHA-256
`df02ba5fd41fd0cc3fd2d39474fcfe7a628c15eb9ea012e8c70b1ffb3dabecde`.
It pins ten shipped Gaussian codebooks and defines:

- `fixed_hq_b2_v1`: K8, K32, K128, K512;
- `adaptive_hq_b2_safe_v1`: K8, K16, K32, K64, K128, K256, K512.

Canonical HQ and AdaptiveHQ now require those bundles at construction. The
manifest SHA is also hard-coded as the shipped root of trust, so changing the
manifest and tensor together does not bypass verification. A missing
manifest/artifact, manifest/file/tensor SHA mismatch, wrong original dtype or
shape, non-finite value, duplicate codeword, unpinned shipped file, or
development override fails closed. Explicit custom development points retain
the online fallback; `KVCOMP_CODEBOOK_DIR` remains development-only, receives
the same tensor validation, is exposed in provenance, and is rejected by the
canonical recipes.

Fixed HQ remains the paper's unchanged b=2 Gaussian/NC-key/LS-refit-value
recipe with K5/V3, K7/V5, and K9/V7 block-code budgets. The sole opt-in
AdaptiveHQ policy uses only `[(2r,2r),(2r+1,2r-1)]`, global K:V sensitivity
weight 4:1, and exactly one in-scale selector bit per update. Serialization
does not alter any measured payload or quality result.

## Slurm execution

- 69360[0-3]: reference serialization, all `COMPLETED 0:0`;
- 69364[0-3]: independent fresh-process regeneration, all `COMPLETED 0:0`;
- 69368: public online/artifact payload parity, `COMPLETED 0:0`;
- 69369[0-2]: native parity at rates 2/3/4, all `COMPLETED 0:0`.

Exact scheduler records are in `SACCT.txt`; stdout/stderr are retained in
`logs/`.

## Wheel and test verification

An `uv build` wheel was inspected and executed from an unpacked/isolated
wheel environment. It contained the `.pt` files and `manifest.json`; both
required-bundle checks passed (`fixed=4`, `adaptive=7`), every artifact source
was `shipped_manifest`, and the manifest SHA was the pinned value above. The
checked wheel SHA-256 was
`0ab857bf4c0b5dae3123bdf29e7bdd6f2dd09fd2c593355ef51984c9debaf774`.
See `WHEEL_VERIFICATION.txt`.

Targeted artifact tests cover the root manifest hash, missing/corrupt files,
file and raw-tensor hashes, dtype, shape, finite values, unique rows,
development override provenance/rejection, explicit online fallback, exact
safe candidate pairs, bundle completeness, and setuptools package data. The
artifact suite passed 12/12; combined artifact/cache-policy targets passed
18 with one CUDA skip; final recipe/public targets passed 15 with three CUDA
skips.

`SHA256SUMS` covers every retained archive file except itself. Credential-like
secret scanning returned zero matches (`SECRET_SCAN.txt`). Generated bytecode
and `__pycache__` directories are intentionally absent.
