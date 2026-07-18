# KVComp vLLM backends

This directory contains vLLM KV-cache integration for HayateQuant and TapeKV.
Model weights are never stored here: pass a local model path or Hugging Face
identifier to the benchmark drivers.

## Backend split

* `hayate_full_native.py` and `hayate_full_triton.py` implement the faithful
  packed fixed-rate HQ key and value layout. `vllm_mode="full_native"` selects
  it from a saved config. Rates 3 and 4 are supported by the current upstream
  vLLM TurboQuant cache specifications; rate 2 fails closed.
* `hayate_native.py`, `hayate_triton.py`, and `hayate_triton_config.py` are the
  historical key-block-VQ/scalar-value hybrid. It is not faithful full HQ.
* `taper_native.py` reproduces TapeKV fake-quant semantics in a model-dtype
  paged cache. It is suitable for quality/parity checks, not packed-memory or
  throughput claims. Prefix caching, CUDA graphs, cascade attention, and DCP
  are disabled or rejected because this backend mutates resident pages.
* A saved HayateQuant codec plus a `taperkv` or `block_taperkv` policy composes
  the methods in one defined order: HayateQuant is applied once to arriving KV,
  then TapeKV manages age-dependent resident precision.

`mapping.py` fails closed when a recipe cannot be represented. It never falls
back from faithful full HQ to the hybrid, or from TapeKV to an uncompressed
cache. Full HQ requires an explicit `exact` or frozen `lut` assignment mode;
the latter must serialize `lut_bins` and `lut_candidates`. Shape-dependent
`auto` assignment is rejected for cross-backend parity.

## Smoke checks

From the repository root, run:

```bash
.venv/bin/python -m pytest -q tests/vllm_plugins/kvcomp
```

GPU-dependent tests are skipped when CUDA is absent. CPU tests cover config,
registration, page mutation, and one-time HayateQuant composition contracts.

Run all three generation paths in isolated fresh processes with:

```bash
.venv/bin/python benchmark/kvcomp-codec/run_vllm_inference_smoke_suite.py \
  --model /path/to/local/model \
  --output-dir /path/to/new/receipt-directory
```

The suite writes child logs, one receipt per path, source/package provenance,
and a fail-closed `suite.json`. It also isolates FlashInfer's generated cache
inside the output directory, so stale absolute paths from another environment
cannot contaminate the smoke. The suite disables optional DeepGEMM warmup;
these checks validate KV-cache integration and generation, not weight-GEMM
performance. The receipt directory must be new or empty; the runner refuses to
mix a new run with stale artifacts.

Only a passing three-path suite can be rendered for the paper:

```bash
.venv/bin/python benchmark/kvcomp-codec/render_vllm_inference_smoke_latex.py \
  --suite /path/to/receipt-directory/suite.json \
  --output /path/to/generated/kvcomp_vllm_smoke.tex
```

The renderer rejects missing/failed phases, mismatched model identities,
non-faithful HQ contracts, or TapeKV rows presented as packed evidence. The
result is explicitly implementation-smoke evidence, never a quality, physical
memory, or throughput claim.
