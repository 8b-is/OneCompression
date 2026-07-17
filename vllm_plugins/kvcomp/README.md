# KVComp vLLM backends

This directory contains the vLLM integration for the four public HayateQuant
recipes (`HQ`, `HQ-Frontier`, `AdaptiveHQ`, and `RotationHQ`).  Model weights
are never stored here: pass a local model path or a Hugging Face identifier to
the benchmark drivers, optionally resolving relative names with
`--models-root`.

## Backend split

* `hayate_full_native.py` and `hayate_full_triton.py` are the canonical full
  KV-cache backends used for paper measurements.  Their configuration is in
  `hayate_full_config.py`.
* `hayate_native.py`, `hayate_triton.py`, and `hayate_triton_config.py` are
  compatibility adapters for older vLLM plugin entry points.  They remain
  import-compatible but are not separate paper methods.
* `mapping.py` owns layer/attention mapping; `native_turboquant.py` is the
  TurboQuant compatibility path.

The package-level installer in `vllm_plugin.py` selects the full backend when
the vLLM version supports it and falls back to the compatibility adapter only
when necessary.  This keeps backend selection separate from codec recipes and
from benchmark scoring.

## Smoke checks

From the repository root, run the vLLM plugin tests with the project Python
environment.  GPU-dependent tests are skipped when vLLM or CUDA is absent;
the codec and benchmark tests remain runnable on CPU for path and schema
validation.
