# KV codec: HayateQuant vs TurboQuant / HALO (speed-quality)

Main experiment for the data-free rotated KV codecs, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs.

**HayateQuant** (`onecomp/kvcomp/codec/hayate/codec.py`) is the proposed
method: block-VQ (data-free K-means codebook, high-budget Lloyd
artifacts in `onecomp/kvcomp/codec/hayate/codebooks/`) + folded rate-aware rotation
(single GEMM; auto resolves to Haar-dense at bits<=2 and hybrid at bits>=3,
per the measured v7 rotation-x-scale attribution) + role-asymmetric scales
(NC keys / ls_iter values), with a
candidate-list grid-LUT encoder that keeps encode latency independent of the
codebook size. It is a strict superset of both baselines: `block_size=1`
reproduces TurboQuant/HALO bit-for-bit (enforced by unit tests and by the
live `HayateQuant-bs1` row in every run).

| Codec | Quantizer | Rotation | Scale | Role |
|---|---|---|---|---|
| TurboQuant (hadamard) | scalar Lloyd-Max | WHT butterfly | RMS | paper baseline |
| TurboQuant-QJL | scalar Lloyd-Max + 1-bit QJL residual | WHT butterfly + Gaussian sketch | RMS + residual norm | TurboQuant-prod appendix baseline |
| TurboQuant (dense) | scalar Lloyd-Max | dense GEMM | RMS | rotation ablation |
| HALO (auto) | scalar Lloyd-Max | cheapest GEMM | NC / ls_iter | prior best (superseded) |
| **HayateQuant (auto)** | **block-VQ** | **folded hybrid (1 GEMM)** | NC / ls_iter | **proposed** |

`TurboQuant-QJL` follows the TurboQuant-prod construction for keys: for a target
payload of `b` bits per coordinate it uses a `b-1` bit scalar MSE key code plus
a 1-bit Gaussian QJL sketch of the key residual, while values use the regular
`b` bit scalar MSE code.  The residual norm is counted in
`stored_bits_per_coord`, so this row should be read as an appendix diagnostic
for the inner-product estimator rather than as an equal-stored-bits replacement
for the main TurboQuant-MSE baseline.

Ablation rows included in every run: `HayateQuant-bs1` (superset guard),
`HayateQuant-exact` (LUT off), `HayateQuant-K12` (large-codebook quality
ceiling), `HayateQuant-dense-LS` (Qwen-family PPL operating point).

**Lever-isolation ablation** (`--config-name paper_hayatequant_ablation`, sets
`ablation: true`) adds four more rows, each disabling exactly one of
HayateQuant-auto's levers: `HayateQuant-noscale` (RMS both roles),
`HayateQuant-hadamard-bvq` (no per-block dense rotation factor),
`HayateQuant-beta` (exact finite-d source law), `HayateQuant-bs2` (explicit
block size). Measured on Llama-3.1-8B (job 59050): every lever contributes
positively at 2/3-bit (block-VQ is dominant: 2-bit PPL 10.02 -> 9.51 -> 9.15
for bs1 -> bs2 -> bs4; scale -0.05, hybrid rotation -0.03), and none hurts --
see `papers/hayatequant_paper/README.md` (v6) for the full table. Results ->
`results_ablation/`.

**v7 round** (paper README v7): candidates measured and *rejected* with
evidence: residual VQ (loses to single-stage block-VQ at every rate on the
data-free source law) and LUT T=4 (no PPL gain). Adopted: committed
high-budget Lloyd codebook artifacts (distortion down at every point, 2-bit
PPL -0.12) and the rate-aware auto rotation (2-bit PPL -0.21 on Llama).
Final 2-bit Llama PPL: 8.825 vs TurboQuant-hadamard 10.161 / -dense 9.942.

**v8 round** (paper README v8): adds optional SpinQuant-style calibration
(`HayateQuantCodec(calibration=...)`, learned per-role Cayley rotation +
real-data codebook refit; opt-in, holdout-distortion guarded and -0.18 on Qwen3-8B
2-bit) and the K/V-asymmetric `HayateQuant-K4V2` row at the 3.0-avg-bit
point (keys are the sensitive role: -0.09/-0.15 PPL vs uniform 3-bit on
Llama/Qwen3-8B, local synthetic multi-key +0.067 on Llama, same storage).

## Run

**Standard run:**
```bash
srun --gres=gpu:1 .venv/bin/python -u codec_benchmark.py \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

**HayateQuant paper preset** (`conf/paper_hayatequant.yaml`, headline 4/3/2-bit,
multi-model):
```bash
# one model at a time (override model_path); ~30 min/model on one GPU
srun --gres=gpu:1 .venv/bin/python codec_benchmark.py --config-name paper_hayatequant \
    model_path=/.../models/llama-3.1-8b-instruct
srun --gres=gpu:1 .venv/bin/python codec_benchmark.py --config-name paper_hayatequant \
    model_path=/.../models/DeepSeek-R1-Distill-Qwen-7B
```

**Cross-model launcher / summarizer** (`hayatequant_cross_model.py`):
```bash
# submit the default paper grid (Llama2/3.1, Qwen3-4B/8B, Ministral,
# R1-Llama-8B, R1-Qwen-7B/1.5B)
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py --submit

# add extra coverage or run inside an interactive GPU allocation
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py \
    --models qwen3-14b r1-qwen-14b --submit

# summarize finished results_*.json files as a compact Markdown table
.venv/bin/python benchmark/kvcomp-codec/hayatequant_cross_model.py \
    --summarize --output-root /lustre/.../kvcomp_hayatequant/results_cross_model_v1
```

**Zero-shot downstream grid** (`hayatequant_zeroshot.py`):
```bash
# same eight models as the main PPL table; ARC-Easy, ARC-Challenge, PIQA,
# HellaSwag, and WinoGrande; reports mean acc/acc_norm for TQ and HQ variants
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --submit \
    --output-root /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/zeroshot_v1

# quick smoke test only; do not use --limit for final paper numbers
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --run-local \
    --models qwen3-0.6b --tasks piqa --limit 4

# summarize completed zeroshot_*.json files
.venv/bin/python benchmark/kvcomp-codec/hayatequant_zeroshot.py --summarize \
    --output-root /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/zeroshot_v1
```

**Native vLLM kernel smoke** (`hayatequant_vllm_native_smoke.py`):
```bash
# compiles and runs the byte-compatible TurboQuant-layout HayateQuant key
# block-VQ store/decode kernels for 3-bit and 4-bit presets; no model load
.venv-vllm/bin/python benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py \
    --output /lustre/.../kvcomp_hayatequant/native_vllm/smoke.json
```

**Full HayateQuant paper battery** (`submit_hayatequant_paper.sh`):
```bash
# PPL/distortion, zero-shot ACC, long-context retrieval, reasoning decode,
# codec speed, and native vLLM smoke. The method set includes TurboQuant,
# FibQuant, BlockQuant, HayateQuant, and the Fib/Block import candidates.
RUN_ROOT=/lustre/.../kvcomp_hayatequant/paper_final_YYYYMMDD \
PARTITIONS=debug,batch-1gpu \
  bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh all

# individual slices use the same non-duplicated partition-list submission
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh ppl
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh zeroshot
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh long
bash benchmark/kvcomp-codec/submit_hayatequant_paper.sh speed
```

(`conf/paper_halo.yaml` is kept only to reproduce the historical HALO runs.)

Config: `conf/benchmark_codec.yaml`. Metrics per bit-width {2,3,4}: distortion
(relMSE on captured real KV), WikiText-2 PPL under `CompressedDynamicCache`,
and overhead (ms/encode, Melem/s) at the model's true head dim. Results →
`output_dir` (JSON + Markdown).

## Headline results (v3, see papers/hayatequant_paper/README.md for full tables)

Llama-3.1-8B-Instruct (head_dim=128), real-KV distortion + WikiText-2 PPL:
HayateQuant-auto strictly improves distortion and PPL over both TurboQuant
rotations and HALO-auto at every bit-width (largest gains at 2-3 bit, e.g.
2-bit PPL −9.5% vs TurboQuant-hadamard), at equal stored bits.  On
DeepSeek-R1-Distill-Qwen-1.5B the `HayateQuant-dense-LS` variant is the best
PPL point (Qwen-family KV distributions interact badly with the ±1 Hadamard
mix).  Encode/decode speed: the folded rotation + adaptive LUT/fp16 encoder
give decode throughput 1.4× TurboQuant and TTFT parity at 3/4-bit.

## Historical HALO results (superseded)

The tables below are the original HALO-paper measurements, kept for
reproducibility.  HALO's contribution was **lower online overhead at
TurboQuant-dense quality** (cheapest-transform selection); it did not improve
quality over TurboQuant.  HayateQuant subsumes HALO (same scale levers)
and adds the quality win via block-VQ.

### Rotation overhead (ms/encode, S=1024, H=8)

| head_dim | HALO(auto) | TurboQuant(hadamard) | speedup | quality (relMSE 3-bit) |
|---:|---:|---:|---:|---:|
| 64  | 0.308 | 0.903 | **2.9×** | identical |
| 128 | 0.346 | 1.004 | **2.9×** | identical |
| 256 | 0.441 | 1.103 | **2.5×** | HALO 0.0344 = Turbo 0.0343 |
| 512 | 0.936 | 1.297 | **1.4×** | identical |

### llama-3.1-8b-instruct (head_dim=128)

| bits | Codec | dist(K) | dist(V) | ms/encode | Melem/s | PPL |
|---|---|---:|---:|---:|---:|---:|
| 4 | TurboQuant-hadamard | 0.00884 | 0.00935 | 1.020 | 1028 | 7.7096 |
| 4 | TurboQuant-dense | 0.00952 | 0.00920 | 0.354 | 2960 | 7.7061 |
| 4 | HALO-auto | 0.00952 | 0.00920 | 0.356 | 2944 | 7.7061 |
| 3 | TurboQuant-hadamard | 0.03205 | 0.03413 | 1.027 | 1022 | 8.0454 |
| 3 | TurboQuant-dense | 0.03441 | 0.03392 | 0.358 | 2931 | 8.0973 |
| 3 | HALO-auto | 0.03441 | 0.03392 | 0.351 | 2990 | 8.0973 |
| 2 | TurboQuant-hadamard | 0.10945 | 0.11708 | 1.023 | 1025 | 11.3663 |
| 2 | TurboQuant-dense | 0.11740 | 0.11592 | 0.353 | 2974 | 11.2002 |
| 2 | HALO-auto | 0.11740 | 0.11592 | 0.349 | 3006 | 11.2002 |

### DeepSeek-R1-Distill-Qwen-1.5B (head_dim=128)

| bits | Codec | dist(K) | dist(V) | ms/encode | Melem/s |
|---|---|---:|---:|---:|---:|
| 4 | TurboQuant-hadamard | 0.00827 | 0.00915 | 1.024 | 1024 |
| 4 | HALO-auto | 0.00899 | 0.00925 | 0.354 | 2958 |
| 3 | TurboQuant-hadamard | 0.03043 | 0.03323 | 1.029 | 1019 |
| 3 | HALO-auto | 0.03229 | 0.03352 | 0.354 | 2960 |
| 2 | TurboQuant-hadamard | 0.09668 | 0.11300 | 1.031 | 1018 |
| 2 | HALO-auto | 0.11263 | 0.11480 | 0.355 | 2954 |
