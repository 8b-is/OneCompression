# Model Validation (GPTQ 4-bit, groupsize=128)

Validates OneComp's `GPTQ` quantizer on multiple models with a fixed
`wbits=4`, `groupsize=128` configuration and `qep=False`. Configuration
is managed with [Hydra](https://hydra.cc/).

## Purpose

- Confirm that pure GPTQ (4-bit, gs=128, no QEP) runs end-to-end on a
  variety of model architectures and sizes.
- Save the quantized model and compare original vs quantized
  perplexity for each model.

## Requirements

Hydra is not part of OneComp's runtime dependencies. Install it via the
`hydra` extra:

```bash
# uv
uv sync --extra <cuXXX> --extra hydra

# pip
pip install "onecomp[hydra]"
```

Replace `<cuXXX>` with the CUDA variant matching your environment
(`cpu`, `cu118`, `cu121`, `cu124`, `cu126`, `cu128`, `cu130`).

## Usage

Specify a model via either `model_path` (local directory) or
`model_id` (Hugging Face Hub). Exactly one of the two is required.

```bash
# Local model
python validate_gptq.py model_path=/path/to/model

# Hugging Face Hub
python validate_gptq.py model_id=<HF Hub ID>
```

### Hydra Overrides

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_gptq.py model_path=/path/to/model output_dir=outputs/custom
```

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- `quantized/`  - quantized model saved via `runner.save_quantized_model(...)`
- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: original / quantized perplexity

## Validated Models

The validation set covers the following models:

- TinyLlama-1.1B (`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`)
- gemma-4-E2B
- Llama-2-7B
- Llama-3-8B
- Qwen3-8B

## Results

Perplexity is measured on `wikitext-2-raw-v1` (OneComp default).
Quantizer is `GPTQ(wbits=4, groupsize=128)` with `qep=False` and
`CalibrationConfig(max_length=512, num_calibration_samples=128)`.

| Model | Original PPL | Quantized PPL | Notes |
|---|---:|---:|---|
| TinyLlama-1.1B | 7.77 | 8.69 | OK |
| gemma-4-E2B (base) | 25.99 | 35.03 | warn (see below) |
| Llama-2-7B | 5.47 | 6.59 | OK |
| Llama-3-8B | 6.14 | 27.74 | warn (see below) |
| Qwen3-8B | 9.72 | 10.72 | OK |

### Notes on gemma-4-E2B and Llama-3-8B

Both models show a larger original-vs-quantized PPL gap than the other
entries on this validation set:

- gemma-4-E2B: `25.99 -> 35.03` (≈35% relative increase)
- Llama-3-8B: `6.14 -> 27.74` (≈4.5x increase)

Llama-2-7B (`+20%`), TinyLlama-1.1B (`+12%`), and Qwen3-8B (`+10%`) all
land in a healthier range under the same setting. A likely contributor
is the compact calibration config (`max_length=512`,
`num_calibration_samples=128`), which may be a tight fit for these
architectures (e.g. Llama-3-8B's 128k-vocab tokenizer).

Worth trying if the result needs to be improved:

- increase calibration to defaults (`max_length=2048`,
  `num_calibration_samples=512`),
- enable QEP (`qep=True`) to leverage quantization-error propagation.
