# Rejected native exact-grid development candidate (2026-07-16)

Status: **rejected; no candidate-index code is retained in the implementation**.

This bundle preserves a development attempt to accelerate canonical b=2 HQ
assignment with a conservative spatial candidate table.  The table itself did
not omit the reference winner in the observed failures.  The rejection was
caused by the manually evaluated Triton fp16 score differing from the public
CUDA fp16 GEMM on rare near-ties (2 mismatches per 1,000,000 rate-4 rows for
both key and value in the recorded stress fixture).  The exact queries and
codes are transcribed in `REJECTED_DIAGNOSTIC.json`; the raw large-smoke
assertion is `raw/failures/smoke_rate4.log`.

The candidate was fast but therefore inadmissible: the 4096-token stage
profile fell from 4.730 to 0.737 ms at rate 3 and from 9.489 to 0.951 ms at
rate 4.  Those numbers are diagnostic only and must not be used in a paper or
formal speed claim.

The separate BLOCK_KV=4/one-warp decode trial was also rejected: its retained
rate-3 smoke measured 1.312 ms decode latency.  BLOCK_KV=8/two warps was kept
for the accepted exact implementation.

Representative commands (the rejected implementation is intentionally no
longer present):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_profile.py \
  --avg-bits 3 --tokens 4096 --warmups 5 --iters 10 \
  --output profile_rate3.json

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py \
  --avg-bits 4 --assignment-mode exact --tokens 4096
```

Every JSON records the then-current commit, dirty status, arguments, GPU, and
source SHA-256.  The candidate profile's native-kernel SHA is also copied into
`REJECTED_DIAGNOSTIC.json`.  `SHA256SUMS` covers this complete bundle.

