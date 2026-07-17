# HQ native speed audit (2026-07-17)

The full native HQ path was profiled on kagura-gpu11 (NVIDIA B200,
`bfloat16`, head dimension 128, 8 KV heads, 32 query heads).  The profile is
stage-level and is not a TQ-vs-HQ claim.

At 1024 captured tokens and 3-bit HQ, the median decode time was 0.135 ms.
The largest non-attention pieces were the query rotation (0.033 ms) and the
inverse value rotation (0.027 ms); packed stage-1 attention and reduction were
about 0.012 and 0.013 ms.  On the store path, exact assignment was already
fused into a tiled Triton argmax; the remaining cost is primarily the two
rotations and value LS-refit.

For a 128-token decode smoke, varying the reduction split count gave:

| max splits | median decode ms |
|---:|---:|
| 1 | 0.1102 |
| 4 | 0.1317 |
| 8 | 0.2321 |
| 16 | 0.1164 |
| 32 | 0.1204 |

The measurements are noisy at this scale, but they expose a clear fixed-32
over-partitioning hazard for short contexts.  The native HQ backend now uses
the CPU sequence-length mirror to choose `ceil(max_seq_len/256)`, bounded by
the original maximum split count; if that mirror is unavailable it preserves
the old fixed count.  This avoids a device synchronisation and leaves long
contexts capped as before.

`KVCOMP_HQ_DYNAMIC_SPLITS=0` restores the fixed split count for deployments
that pin a single CUDA graph across heterogeneous sequence-length batches.

At 1024 tokens, the same profile gave 0.1234 ms with 4 splits versus
0.1329 ms with 32 splits (about 7% lower in this run), so the policy is not
limited to the 128-token case.

The two rotation GEMMs remain mathematically necessary for the current HQ
recipe and are also present in the quality contract.  Converting them to
BF16 did not improve B200 latency in a direct microbenchmark, so no
precision-reducing fast path was enabled.  The next safe optimization target
is a fused inverse-rotation output kernel, but it requires a separate parity
gate and is not silently substituted into the formal method.
