# Job/command manifest

All jobs used partition `debug`, one GPU, eight CPUs, 64 GiB RAM, and a
one-hour limit unless a raw log states otherwise.  Commands ran from the repo
root with `PYTHONPATH` set to the repo.  Native jobs used
`/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python`;
PPL jobs used the repo `.venv/bin/python`.

For result-producing jobs, the exact resolved flags are embedded in
`results/*.json` under `provenance.arguments`.  Slurm did not retain the full
submission string for early jobs, so entries labelled *reconstructed* record
the command family and scientifically relevant configuration without
pretending to be byte-for-byte shell history.

| Jobs | Command/config record | Status/purpose |
|---|---|---|
| 69256, 69264, 69265 | *reconstructed*: `hayatequant_vllm_full_native_smoke.py`, original paper/fixed HQ rate-3 development payloads | first two harness/kernel fixes; 69265 pass |
| 69266, 69268, 69269 | *reconstructed*: `hayatequant_vllm_full_native_fair_speed.py --avg-bits 3 --trials 4` | fair-harness fixes; 69269 pass |
| 69267 | *reconstructed*: full-native smoke matrix for paper/fixed rates and K10/V6 ablation | pass; produced matrix JSON files |
| 69270 | fair speed, exact HQ rate 3, 4096 tokens, Hkv=8, Hq=32, D=128, bf16, 50 iterations, four ABBA trials | pass |
| 69271 | same as 69270 at rate 4 | pass |
| 69272 | fair speed exact rate 3 with CUDA-graph serving step | pass |
| 69273, 69275, 69280, 69287 | `hayatequant_vllm_native_generate.py --variant hayate_full --bits 3 --model TinyLlama-1.1B-Chat-v1.0 --max-model-len 512 --max-tokens 8 --enforce-eager` | cache-spec/DeepGEMM/FlashInfer environment diagnostics |
| 69274 | fixed HQ rate-3 b=2 direct-decode smoke | pass |
| 69276, 69277, 69278 | full-native smoke rates 2/3/4, `--assignment-mode lut --lut-bins 256 --lut-candidates 2` | pass |
| 69279 | `hayatequant_lut_quality_gate.py --model Qwen3-0.6B --rates 2 3 4 --lut-bins 256 --lut-candidates 2 --max-length 512 --stride 512 --max-windows 1 --max-relative-ppl-degradation-pct 0.25` | intended exit 2, quality fail |
| 69281, 69282 | fair speed rates 3/4, 4096 tokens, LUT T=2 | pass, but slower than TQ |
| 69283, 69284, 69285 | full-native smoke rates 2/3/4, LUT T=4 | failed because public reference still defaulted to T=2; retained harness-bug logs |
| 69286 | Qwen quality gate all rates, LUT T=4, historical subtract/square scoring | intended exit 2 |
| 69288, 69289 | same quality gate, T=8 and T=16 | intended exit 2 |
| 69290 | exact HQ rate-3 smoke after scoring-codebook/norm hoist | pass |
| 69291, 69292 | exact fair speed rates 3/4, 4096 tokens, graph enabled | pass; canonical final exact timing |
| 69293 | Qwen all-rate T=4 gate with candidate dot-minus-norm scorer | intended exit 2, rate-2 fail |
| 69294, 69296, 69295 | T=4 canonical-score native smoke rates 2/3/4 | pass |
| 69297, 69298 | Qwen all-rate canonical-score gate T=16/T=8 | intended exit 2 |
| 69299, 69300 | `hayatequant_vllm_full_native_profile.py --avg-bits 4/3 --tokens 4096 --kv-heads 8 --query-heads 32 --head-dim 128 --dtype bfloat16 --warmups 5 --iters 20` | pass; stage profiles |
| 69301, 69302 | fair speed rates 4/3, generic canonical-score T=4, 4096 tokens | pass, severe bmm regression |
| 69303 | Qwen rate-2 T=16 gate plus `--audit-assignment-mismatches` | intended exit 2; assignment audit saved |
| 69304--69310 | fused T=4 smoke/PPL/kernel-test/fair suite | compile failure retained verbatim |
| 69311 | CUDA pytest `-k lut_canonical_t4_triton_matches_full_half_gemm_large_fixture` | pass, 1,048,576 rows |
| 69312, 69313, 69314 | fused-v2 T=4 native smoke rates 4/3/2, 32 tokens, fp16 | pass |
| 69315, 69317 | fused-v2 T=4 fair speed rates 4/3, 4096 tokens, bf16, graph, four ABBA trials | pass, all speed gates fail |
| 69316 | fused-v2 Qwen all-rate quality gate, T=4, same pinned protocol | intended exit 2, rate-2 fail |
| 69318 | TinyLlama exact full-native generation smoke, rate 3, `VLLM_USE_DEEP_GEMM=0`; virtualenv-bin PATH fix active | external stale FlashInfer cache failure |

Recent exact command templates:

```text
hayatequant_vllm_full_native_fair_speed.py --avg-bits RATE --recipe hq \
  --assignment-mode exact --tokens 4096 --kv-heads 8 --query-heads 32 \
  --head-dim 128 --dtype bfloat16 --page-size 16 --max-splits 32 \
  --warmups 10 --iters 50 --trials 4 --measure-cudagraph

hayatequant_lut_quality_gate.py --model-path .../models/Qwen3-0.6B \
  --rates 2 3 4 --lut-bins 256 --lut-candidates T --max-length 512 \
  --stride 512 --max-windows 1 --max-relative-ppl-degradation-pct 0.25

hayatequant_vllm_full_native_smoke.py --avg-bits RATE --recipe hq \
  --assignment-mode lut --lut-bins 256 --lut-candidates 4 \
  --tokens 32 --kv-heads 2 --query-heads 4 --head-dim 128 --dtype float16
```

