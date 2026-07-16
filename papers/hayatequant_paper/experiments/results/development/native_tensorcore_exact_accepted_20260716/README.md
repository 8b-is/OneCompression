# Native all-codeword tensor-core development result (2026-07-16)

Status: **accepted as a correctness-preserving implementation candidate;
development-only until repeated from a clean committed source tree**.

The accepted store kernel evaluates every canonical b=2 codeword with a
tensor-core score and fuses argmax, avoiding the public encoder's materialized
`N x K` score matrix.  It is not a LUT or shortlist, changes no codebook,
scale, payload, or quality algorithm, and allocates zero persistent
acceleration-table bytes.  Decode removes the provably unnecessary third byte
load for canonical code widths <=9, guarded so wider recipes use the generic
three-byte path.  BLOCK_KV=8/two warps is likewise scoped to that validated
b=2 fast path; the generic b=1/4/8 path retains BLOCK_KV=16/four warps.

Correctness results:

- Rate 2, 3, and 4, key and value: zero code mismatches against
  `quantize_exact_pre` for seq=1 (512 block rows), seq=4096 (2,097,152 rows),
  198,147 grid-boundary/adjacent-fp16 rows, 250,008 wide/out-of-range rows,
  and 71--1,104 observed fp16 top-score ties.
- Rate 2, 3, and 4 at seq=1 and seq=4096: public codes, signed-fp16 scales,
  packed bytes, one-token/prefill identity, terminal padding sentinel, bulk
  dequantization, and packed attention checks all pass.
- CPU native contracts: 23 passed (`raw/cpu_native_contract_tests.xml`).

Stage profile (B200, tokens=4096, KV heads=8):

| Rate | Baseline exact store | Accepted store | Store improvement | Native exact assignment (K + 2V) |
|---:|---:|---:|---:|---:|
| 3 | 4.730 ms | 0.758 ms | 6.24x | 0.343 ms |
| 4 | 9.489 ms | 1.269 ms | 7.48x | 0.849 ms |

The fresh-process ABBA comparison remains unfavorable to upstream vLLM 0.20
TurboQuant and is preserved without filtering:

| Rate | Prefill-store HQ/TQ speedup | Decode HQ/TQ | Eager step HQ/TQ | CUDA-graph step HQ/TQ |
|---:|---:|---:|---:|---:|
| 3 | 0.1730x | 0.7612x | 0.2452x | 0.5620x |
| 4 | 0.0984x | 0.9239x | 0.2355x | 0.4650x |

Thus this bundle supports a large implementation improvement, **not** a
native-speed dominance claim over TurboQuant.  The ABBA JSON files are marked
ineligible only because the implementation was measured from a dirty tree;
they must be repeated after commit before formal use.

Commands:

```bash
# Adversarial exact-code parity
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_exact_parity.py \
  --avg-bits 2 --output exact_parity_rate2.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_exact_parity.py \
  --avg-bits 3 --output exact_parity_rate3.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_exact_parity.py \
  --avg-bits 4 --output exact_parity_rate4.json

# Packed payload/dequant/attention, repeated with rate={2,3,4}, tokens={1,4096}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_smoke.py \
  --avg-bits 3 --assignment-mode exact --tokens 4096 \
  --warmups 3 --iters 20 --output smoke_rate3_seq4096.json

# Stage profile, repeated for both rates
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_profile.py \
  --avg-bits 3 --tokens 4096 --warmups 5 --iters 20 \
  --output profile_rate3.json

# Fresh-process ABBA, repeated for both rates
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
  benchmark/kvcomp-codec/hayatequant_vllm_full_native_fair_speed.py \
  --avg-bits 3 --trials 4 --warmups 10 --iters 50 \
  --output abba_rate3.json
```

Raw JSON/log files contain the source commit (`e13ee547...` at measurement
time), full dirty status, arguments, GPU identity, package versions, and
per-source SHA-256.  `SHA256SUMS` covers the complete repo-local bundle.
