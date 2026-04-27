# Basic Model Validation (AutoBit)

Validates OneComp's `AutoBitQuantizer` on multiple models using a fixed
`target_bit=4` budget with QEP enabled. Configuration is managed with
[Hydra](https://hydra.cc/).

## Purpose

- Confirm that AutoBit (4-bit target, QEP on) runs end-to-end on a
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
python validate_autobit.py model_path=/path/to/model

# Hugging Face Hub
python validate_autobit.py model_id=<HF Hub ID>
```

### Hydra Overrides

Any field in [conf/validate.yaml](conf/validate.yaml) can be overridden
on the command line, for example:

```bash
python validate_autobit.py model_path=/path/to/model output_dir=outputs/custom
```

### Outputs

For each run, Hydra changes into `output_dir` and the following are
produced:

- `quantized/`  - quantized model saved via `runner.save_quantized_model(...)`
- standard Hydra logs (`.hydra/`, `*.log`)
- stdout: original / quantized perplexity

## Validated Models

The validation set covers the following models:

- TinyLlama-1.1B
- Llama-2-7B
- Llama-3-8B
- Qwen3-8B
- gemma-4-E2B
