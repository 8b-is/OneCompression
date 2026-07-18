# Integrated packed-native runner smoke (development only)

Status: **correctness passed; speed claim not permitted; HQ lost every observed point estimate**.

This archive checks the integrated exact b=2 HQ kernel and the preregistered
native runner from detached, clean source commit
`7096e8b374fefb430878e433cc0b18fb3bdca03b` on `kagura-gpu11` B200 GPUs.
It is deliberately a four-process, one-warmup, two-iteration smoke.  Each
result therefore records all three formal-cell ineligibility reasons and
cannot authorize a paper claim.

Both methods wrote their physical packed cache, were independently
dequantized, and matched the dense fp32 GQA reference in eager and CUDA-graph
execution.  Packed bytes and outputs were deterministic across the two fresh
processes per method; padding sentinels were intact.  All worker commands,
exit statuses, stdout, and stderr are retained next to the result JSON.  The
controller stdout/stderr streams are retained as `rate*.controller.*`.

The unrounded development point estimates were:

| rate | endpoint | HQ ms | upstream TQ ms | TQ/HQ | HQ faster |
|---:|---|---:|---:|---:|:---:|
| 3 | prefill store | 0.8920719922 | 0.2442960013 | 0.2738523386 | no |
| 3 | decode attention | 0.1707679965 | 0.1373119988 | 0.8040850839 | no |
| 3 | eager store1 + decode | 0.9335359931 | 0.2513439991 | 0.2692386806 | no |
| 3 | graph store1 + decode | 0.1322399992 | 0.0548639996 | 0.4148820320 | no |
| 4 | prefill store | 0.9690879732 | 0.1999760009 | 0.2063548475 | no |
| 4 | decode attention | 0.2107599936 | 0.1197600011 | 0.5682292878 | no |
| 4 | eager store1 + decode | 1.0648719966 | 0.2185519971 | 0.2052378105 | no |
| 4 | graph store1 + decode | 0.1372720003 | 0.0510240002 | 0.3716999828 | no |

These short-run values are diagnostics only.  In particular, they do not
replace the frozen 20-process/10-warmup/100-iteration 48-cell matrix.  The
outer asynchronous SSH launcher did not retain its own shell exit-code
receipt; this limitation is recorded here rather than inferred.  Both
controllers produced schema-valid final JSON, empty stderr, and complete
per-worker exit receipts.
