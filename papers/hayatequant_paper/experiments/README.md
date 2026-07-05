# HayateQuant paper experiments

Self-contained measurement scripts behind the design decisions and tables in
the HayateQuant paper. Every number in the paper can be reproduced from this
repository alone:

| Paper item | Script / harness | What it measures |
|---|---|---|
| Distortion + PPL + speed grid (main tables) | `benchmark/kvcomp-codec/codec_benchmark.py` | TurboQuant / HALO / HayateQuant (+K4V2, ablation rows) across bit-widths on real models |
| Long-context (Needle, RULER, LongBench-E) | `benchmark/kvcomp-longcontext/` with `conf/hayatequant_paper.yaml`, `conf/calib_check_*.yaml`, `conf/asym_check_*.yaml` | downstream accuracy at 8k-64k contexts |
| Rotation x scale attribution, LUT candidate depth, codebook artifacts | `bench_policy_v7.py` | isolates each lever's PPL contribution at 2-4 bits |
| Residual VQ (rejected) | `bench_rvq_gain.py` | RVQ vs single-stage block-VQ distortion + encode cost |
| SpinQuant-style calibration decomposition | `bench_calib_v8.py` | rotation-only / codebook-only / both vs data-free PPL |
| K/V-asymmetric bit allocation | `bench_kvasym_v8.py` | (K4,V2) / (K2,V4) / uniform at matched average bits |
| Combined operating point (K4V2 + calibration) | `bench_combo_v8.py` | additivity of the two adopted v8 levers |
| vLLM end-to-end (plugin + native mapping) | `vllm_hadamard_e2e.py` | HayateQuant config -> vLLM serving parity check |
| High-budget codebook refit | `python -m onecomp.kvcomp.codec.hayate.codebooks.fit_codebooks` | regenerates the committed `gauss_b{B}_k{K}.pt` artifacts |

Run any script on Slurm with the generic wrapper (uses the repo venv, writes
JSON results to `results/` and logs to `logs/`):

```bash
cd papers/hayatequant_paper/experiments
sbatch run_experiment.sh bench_kvasym_v8.py /path/to/llama-3.1-8b-instruct
```

or directly:

```bash
../../../.venv/bin/python bench_policy_v7.py /path/to/model
```

All scripts take the model path as the first positional argument and default
to Llama-3.1-8B-Instruct. See `papers/hayatequant_paper/README.md` for the
measured results and the adopt/reject decision log (v1-v8).
