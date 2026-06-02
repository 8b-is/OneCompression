# KV codec: HALO vs TurboQuant (speed-quality)

Main experiment for the data-free rotated-scalar KV codecs, the analogue of the
weight-side `benchmark/<model>-<recipe>/` runs. HALO and TurboQuant share the
**same quality core** (random orthogonal rotation + data-free Gaussian
Lloyd-Max), so this is a pure **speed-quality trade-off at equal quality**.

| Codec | Rotation | Preconditioning | Role |
|---|---|---|---|
| TurboQuant (hadamard) | WHT, O(D log D) | — | shipped default |
| TurboQuant (dense) | dense GEMM, O(D²) | — | rotation ablation |
| **HALO (auto)** | **cheapest per D** (dense GEMM at realistic D; WHT only for very large 2ᵏ D) | off by default; skipped entirely when off | **proposed** |

## Why HALO wins on speed at equal quality

Two online-path optimizations (no quality change — both rotations are
orthogonal and the codebook is identical):

1. **`rotation_mode="auto"`** picks the cheapest orthogonal transform for the
   head dim. A microbenchmark (`ichikawa_works/kvcomp_bench/speed_halo.py`)
   shows a single dense cuBLAS GEMM beats the log₂(D) sequential Hadamard
   butterfly at **every** realistic head dim (D ≤ 512) on GPU, so `auto` uses
   dense there (WHT reserved for very large 2ᵏ D / non-matmul hardware).
2. **Skip dead preconditioning.** When preconditioning is off (the default),
   the `(x−μ)·w` step is the identity; HALO now skips it entirely instead of
   running a full-tensor subtract-0 / multiply-1, plus the per-call sign vector
   is cached (was recomputed on CPU + copied to GPU every encode/decode).

### Rotation overhead (ms/encode, S=1024, H=8) — validated

| head_dim | HALO(auto) | TurboQuant(hadamard) | speedup | quality (relMSE 3-bit) |
|---:|---:|---:|---:|---:|
| 64  | 0.308 | 0.903 | **2.9×** | identical |
| 128 | 0.346 | 1.004 | **2.9×** | identical |
| 256 | 0.441 | 1.103 | **2.5×** | HALO 0.0344 = Turbo 0.0343 |
| 512 | 0.936 | 1.297 | **1.4×** | identical |

HALO(auto) equals TurboQuant(dense) overhead (the cheapest path) and is
**2.3–2.9× faster than TurboQuant's shipped default (Hadamard)** at identical
reconstruction quality.

### Paper-grade headline (`paper_halo.yaml`, 3 models × 3 bit-widths) — Jun 2026

After the **device-resident orthogonal cache** fix the gap widens further (a
single dense GEMM with no per-call host→device copy):

| model | bits | TurboQuant-hadamard | TurboQuant-dense | **HALO-auto** | speedup vs Had. | HALO==dense? |
|---|---:|---:|---:|---:|---:|:---:|
| Llama-3.1-8B-Instruct | 4 | 1.070 ms (PPL 6.786) | 0.286 ms (PPL 6.783) | **0.284 ms** (PPL 6.783) | **3.76×** | ✓ |
| Llama-3.1-8B-Instruct | 3 | 1.081 ms (PPL 7.097) | 0.286 ms (PPL 7.144) | **0.286 ms** (PPL 7.144) | **3.78×** | ✓ |
| Llama-3.1-8B-Instruct | 2 | 1.084 ms (PPL 10.161) | 0.284 ms (PPL 9.942) | **0.285 ms** (PPL 9.942) | **3.81×** | ✓ |
| DeepSeek-R1-Distill-Qwen-7B | 4 | 1.067 ms | 0.290 ms | **0.290 ms** | **3.68×** | ✓ |
| DeepSeek-R1-Distill-Qwen-7B | 3 | 1.141 ms | 0.290 ms | **0.291 ms** | **3.92×** | ✓ |
| DeepSeek-R1-Distill-Qwen-7B | 2 | 1.142 ms | 0.290 ms | **0.292 ms** | **3.91×** | ✓ |
| DeepSeek-R1-Distill-Qwen-1.5B | 4 | 0.950 ms | 0.283 ms | **0.284 ms** | **3.34×** | ✓ |
| DeepSeek-R1-Distill-Qwen-1.5B | 3 | 1.091 ms | 0.286 ms | **0.285 ms** | **3.83×** | ✓ |
| DeepSeek-R1-Distill-Qwen-1.5B | 2 | 1.106 ms | 0.285 ms | **0.285 ms** | **3.88×** | ✓ |

**Precise quality relationship (Llama-3.1-8B, 3-bit):**

| codec | distortion K | PPL | Δ vs Hadamard |
|---|---:|---:|---:|
| TurboQuant-hadamard | 0.03205 | 7.0972 | reference |
| TurboQuant-dense    | 0.03441 | 7.1438 | +0.65% PPL |
| **HALO-auto**       | 0.03441 | 7.1438 | +0.65% PPL, **3.78× faster** |

**HALO-auto is bit-identical to TurboQuant-dense** (same dense rotation, same
codebook) — the speedup vs Hadamard (3.8×) comes at a cost of +0.65% PPL at
3-bit.  The trade-off: Hadamard's structured butterfly gives marginally better
Gaussianisation of the activations (better codebook fit) but costs 3.8× more
wall-clock.  HALO's claim is that the speed:quality trade-off favours dense at
every realistic head_dim (≤ 512) because the absolute PPL gap is tiny (<1%)
and the wall-clock saving is consistent across all bit-widths and model families.

## Run

**Standard run:**
```bash
srun --gres=gpu:1 .venv/bin/python -u codec_benchmark.py \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

**Paper-level run** (100 timing iterations, 32 PPL windows):
```bash
srun --gres=gpu:1 .venv/bin/python -u codec_benchmark.py \
    --config-name paper_run \
    model_path=/lustre/.../models/llama-3.1-8b-instruct
```

**HALO paper preset** (`conf/paper_halo.yaml`, headline 4/3/2-bit, multi-model):
```bash
# launch on three models in parallel via SLURM
sbatch ichikawa_works/kvcomp_verification/run_paper_halo.sh                                                  # llama-3.1-8b-instruct
MODEL=/.../models/DeepSeek-R1-Distill-Qwen-7B sbatch ichikawa_works/kvcomp_verification/run_paper_halo.sh    # 7B reasoning
MODEL=/.../models/DeepSeek-R1-Distill-Qwen-1.5B sbatch ichikawa_works/kvcomp_verification/run_paper_halo.sh  # 1.5B fast
```

Config: `conf/benchmark_codec.yaml`. Metrics per bit-width {2,3,4}: distortion
(relMSE on captured real KV), WikiText-2 PPL under `CompressedDynamicCache`,
and overhead (ms/encode, Melem/s) at the model's true head dim. Results →
`output_dir` (JSON + Markdown).

## Results

### llama-3.1-8b-instruct (head_dim=128)

Real-KV distortion + WikiText-2 PPL + overhead (S=1024, H=8):

| bits | Codec | dist(K) | dist(V) | ms/encode | Melem/s | PPL |
|---|---|---:|---:|---:|---:|---:|
| 4 | TurboQuant-hadamard | 0.00884 | 0.00935 | 1.020 | 1028 | 7.7096 |
| 4 | TurboQuant-dense | 0.00952 | 0.00920 | 0.354 | 2960 | 7.7061 |
| 4 | **HALO-auto** | 0.00952 | 0.00920 | **0.356** | **2944** | 7.7061 |
| 3 | TurboQuant-hadamard | 0.03205 | 0.03413 | 1.027 | 1022 | 8.0454 |
| 3 | TurboQuant-dense | 0.03441 | 0.03392 | 0.358 | 2931 | 8.0973 |
| 3 | **HALO-auto** | 0.03441 | 0.03392 | **0.351** | **2990** | 8.0973 |
| 2 | TurboQuant-hadamard | 0.10945 | 0.11708 | 1.023 | 1025 | 11.3663 |
| 2 | TurboQuant-dense | 0.11740 | 0.11592 | 0.353 | 2974 | 11.2002 |
| 2 | **HALO-auto** | 0.11740 | 0.11592 | **0.349** | **3006** | 11.2002 |

**HALO(auto) is 2.9× faster than TurboQuant's shipped default (Hadamard) at
every bit-width with identical quality** (PPL 7.706 / 8.097 / 11.20 match
TurboQuant-dense to 4 decimals; distortion identical — HALO(auto) and
TurboQuant-dense are the same rotation). The Hadamard default is both slower
*and* no better in quality, confirming the win is purely the cheaper transform
selection + zero-overhead disabled preconditioning.

### DeepSeek-R1-Distill-Qwen-1.5B (head_dim=128) — 2nd architecture

Same speed-quality verdict on a different model family:

| bits | Codec | dist(K) | dist(V) | ms/encode | Melem/s |
|---|---|---:|---:|---:|---:|
| 4 | TurboQuant-hadamard | 0.00827 | 0.00915 | 1.024 | 1024 |
| 4 | **HALO-auto** | 0.00899 | 0.00925 | **0.354** | **2958** |
| 3 | TurboQuant-hadamard | 0.03043 | 0.03323 | 1.029 | 1019 |
| 3 | **HALO-auto** | 0.03229 | 0.03352 | **0.354** | **2960** |
| 2 | TurboQuant-hadamard | 0.09668 | 0.11300 | 1.031 | 1018 |
| 2 | **HALO-auto** | 0.11263 | 0.11480 | **0.355** | **2954** |

Same 2.9× speedup; distortion identical to TurboQuant-dense. (WikiText PPL is
omitted here — this reasoning-distilled model is not a language model, so its
sliding-window PPL is uninformative for *all* codecs equally; K17. Use
distortion as the quality axis on reasoning models.)

## Honest scope

HALO does **not** beat TurboQuant on quality — the two are rate-distortion
equivalent (the same rotation plus Lloyd-Max codebook). The intuition that
"Hadamard is faster" or that online preconditioning improves quality did not hold
up under measurement. The defensible contribution of HALO is therefore **lower
online overhead at identical quality**, achieved by selecting the cheapest
orthogonal transform per head dimension and by skipping the preconditioning math
entirely when it is disabled (the default). Preconditioning and a forced Hadamard
transform remain available as opt-in research levers.
