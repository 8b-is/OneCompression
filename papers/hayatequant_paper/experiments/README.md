# HayateQuant paper experiments

Legacy measurement scripts behind early HayateQuant design decisions.  They
are retained for audit history, but their capped datasets, old codec contracts,
and historical output roots are not admissible sources for the refreshed
paper.  Formal values must come from `submit_hayatequant_paper.sh`, pass
`aggregate_hayatequant_paper_shards.py`, and then be rendered without fallback
constants.  The table below is an index of historical experiments, not a claim
that the old paper numbers reproduce under the current code:

## Archiving an external paper run

Formal and smoke campaigns use an external `RUN_ROOT` while jobs are active.
After the campaign is quiescent, archive its scientific outputs inside the
repository with the fail-closed archiver:

```bash
PY=${PY:-/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python}
"$PY" benchmark/kvcomp-codec/archive_hayatequant_run.py \
  --run-root "$RUN_ROOT" \
  --run-id hayate-formal-20260716 \
  --source-commit "$(git rev-parse HEAD)" \
  --require-scope results/smoke \
  --require-scope results/formal \
  --require-scope summaries \
  --require-scope manifests \
  --require-scope provenance \
  --require-scope logs
```

`--source-commit` must be the full commit recorded for the run, which can
differ from the current `HEAD` when archiving an older campaign. The command
copies every regular scientific file under the following allow-list rather
than selecting only JSON or a hand-maintained filename set. Security-policy
violations reject the whole archive:

* `results/smoke`, `results/formal`, and `results/dev`
* `summaries`, `manifests`, `provenance`, and `logs`

`manifests`, `provenance`, and at least one non-empty result scope are always
required. This retains worker preflights that are referenced by the smoke
approval marker but are not embedded in the native result; archived marker
paths can be relocated below the archive and checked against the manifest's
byte counts and SHA-256 inventory without the external run tree. Other
missing scopes are recorded explicitly; `--require-scope` makes any scope
non-empty and mandatory for a particular campaign. The separately committed
development panel does not need to be duplicated into each formal archive.
Data, `HF_HOME`, model weights, runtime homes, locks, and temporary working
state are outside the allow-list and are never copied.

The destination is fixed at
`papers/hayatequant_paper/experiments/results/<run-id>/`. It contains the
copied hierarchy plus:

* `ARCHIVE_MANIFEST.json`: sorted relative paths, byte sizes, SHA-256 hashes,
  modes, scope coverage, source commit, and an inventory digest.
* `ARCHIVE_COMMAND.json`: the normalized archive command, source and archiver
  commits, source root, and present/missing/required scopes.
* `SHA256SUMS` and `ARCHIVE_MANIFEST.sha256`: independent integrity checks.

The source tree is hashed before and during copying and must remain unchanged.
Symlinks, special files, path traversal, model/cache artifacts inside an
allowed scope, credential-like filenames or contents, and an existing run ID
all abort without leaving a partial archive. The tool never overwrites an
existing archive; choose a new run ID for a scientifically distinct run.

| Paper item | Script / harness | What it measures |
|---|---|---|
| Distortion + PPL + speed grid (main tables) | `benchmark/kvcomp-codec/codec_benchmark.py` | TurboQuant / HALO / HayateQuant (+K4V2, ablation rows) across bit-widths on real models |
| Legacy long-context (Needle, local synthetic multi-key, LongBench-E) | `benchmark/kvcomp-longcontext/` with `conf/hayatequant_paper.yaml`, `conf/calib_check_*.yaml`, `conf/asym_check_*.yaml` | historical downstream diagnostics at 8k-64k contexts; not a RULER benchmark |
| Codec microbenchmarks (encode/decode/round-trip latency table) | `benchmark/kvcomp-longcontext/codec_speed.py` (`ONECOMP_DISABLE_FHT=1` for the butterfly A/B row; `ONECOMP_TRITON_CODEC=1` / `ONECOMP_COMPILE_CODEC=1` for the fused serving-path rows) | per-call encode/decode cost plus the fused `fake_quant_roundtrip` serving path at prefill and decode-step shapes |
| End-to-end speed (TTFT, decode tok/s) | `benchmark/kvcomp-longcontext/e2e_speed.py` (same env-flag A/B; one-shot Slurm campaigns: `run_rt_speed_v9_triton.sh`, `run_rt_speed_v10_final.sh` for the paper tables) | full generation loop against FullKV / TurboQuant / HayateQuant |
| Served-text verification of the fused serving path | `gen_quality_check.py` (Slurm wrapper: `run_gen_quality_v12.sh`) | greedy-generation agreement, serving-cache PPL parity, and in-session speedup of `ONECOMP_TRITON_CODEC` / `+COMPILE` vs the eager fused path |
| FWHT kernel vs folded GEMM vs butterfly | `bench_fht_micro.py` (Slurm wrapper: `run_fht_micro.sh`) | Dao-AILab CUDA kernel / dense GEMM crossover at d=128 vs 4096 |
| Per-step encode profile (launch-bound analysis) | `profile_hq_encode.py` | kernel-launch count and GPU-busy time of the decode-step encode |
| Rotation x scale attribution, LUT candidate depth, codebook artifacts | `bench_policy_v7.py` | isolates each lever's PPL contribution at 2-4 bits |
| Residual VQ (rejected) | `bench_rvq_gain.py` | RVQ vs single-stage block-VQ distortion + encode cost |
| SpinQuant-style calibration decomposition | `bench_calib_v8.py` | rotation-only / codebook-only / both vs data-free PPL |
| K/V-asymmetric bit allocation | `bench_kvasym_v8.py` | (K4,V2) / (K2,V4) / uniform at matched average bits |
| Combined operating point (K4V2 + calibration) | `bench_combo_v8.py` | additivity of the two adopted v8 levers |
| vLLM end-to-end (plugin + native mapping) | `vllm_hadamard_e2e.py` | HayateQuant config -> vLLM serving parity check |
| High-budget codebook refit | `"$PY" -m onecomp.kvcomp.codec.hayate.codebooks.fit_codebooks` | regenerates the committed `gauss_b{B}_k{K}.pt` artifacts |
| Unit / parity tests (incl. FWHT kernel parity) | `"$PY" -m pytest tests/onecomp/kvcomp tests/vllm_plugins/kvcomp` | scalar compatibility reductions, LUT distortion bound, FWHT-vs-butterfly parity |

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

## Serving-path environment flags

The fake-quant serving cache (`CompressedDynamicCache`) calls
`codec.fake_quant_roundtrip(key, value)` -- a fused encode+decode that, for
HayateQuant, batches key and value through one rotation/quantization pipeline
and reuses the encode-time codeword gather for reconstruction. Two flags
control the speed A/B rows:

* `ONECOMP_DISABLE_FHT=1` -- disable the Dao-AILab FWHT CUDA kernel (the
  butterfly "published TurboQuant" row); default grants the kernel to the
  baseline.
* `ONECOMP_COMPILE_CODEC=1` -- wrap HayateQuant's fused round trip in
  `torch.compile(mode="reduce-overhead")` so each call replays as a single
  CUDA graph. This is the big lever for the *prefill* LUT path; adds a
  one-time compilation warmup per (config, shape); leave off for unit tests.
* `ONECOMP_TRITON_CODEC=1` -- dispatch the decode-step (exact-search) regime
  of the fused round trip to a single fused Triton kernel
  (`onecomp/kvcomp/codec/hayate/triton_step.py`: RMS normalize, nearest-
  codeword assign, role-asymmetric scale fit incl. `ls_iter` requantization,
  reconstruct -- one launch; the two rotations stay cuBLAS GEMMs). Parity is
  enforced by `test_hayatequant_triton_step_matches_eager_roundtrip`.
  Composes with `ONECOMP_COMPILE_CODEC` (Triton serves decode steps, the
  CUDA graph serves prefill): that combination is the paper's serving mode
  (`run_rt_speed_v10_final.sh`, session B).
