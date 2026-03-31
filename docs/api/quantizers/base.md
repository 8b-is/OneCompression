# Base Classes

## Quantizer

Abstract base class for all quantizers. Defines the common interface and shared functionality.

### Quantizer Feature Support

`Runner.save_quantized_model()`, `Runner.create_quantized_model()`, and quantized-model
PPL/ACC evaluation internally call `get_quant_config()` and `create_inference_layer()` on
the quantizer. These methods raise `NotImplementedError` by default and must be overridden
by each quantizer to enable these features.

| Quantizer          | `get_quant_config` | `create_inference_layer` | Save | Quantized PPL/ACC |
|--------------------|:------------------:|:------------------------:|:----:|:-----------------:|
| `GPTQ`             | Yes                | Yes                      | Yes  | Yes               |
| `DBF`              | Yes                | Yes                      | Yes  | Yes               |
| `AutoBitQuantizer` | Yes                | Yes                      | Yes  | Yes               |
| `RTN`              | —                  | —                        | No   | No (fallback)     |
| `JointQ`           | —                  | —                        | No   | No (fallback)     |
| `QUIP`             | —                  | —                        | No   | No (fallback)     |
| `CQ`               | —                  | —                        | No   | No (fallback)     |
| `ARB`              | —                  | —                        | No   | No (fallback)     |
| `QBB`              | —                  | —                        | No   | No (fallback)     |
| `Onebit`           | —                  | —                        | No   | No (fallback)     |

For quantizers without support:

- **PPL/ACC evaluation**: `calculate_perplexity()` / `calculate_accuracy()` with
  `quantized_model=True` automatically falls back to the dequantized (FP16) model.
  No error is raised.
- **Saving**: use `save_dequantized_model()` (FP16) or `save_quantization_results()`
  to persist results.

::: onecomp.quantizer._quantizer.Quantizer
    options:
      show_source: false
      members:
        - __post_init__
        - setup
        - quantize
        - quantize_layer
        - quantize_with_qep
        - save_results
        - load_results
        - apply_results_to_model
        - get_quant_config
        - create_inference_layer

## QuantizationResult

Base dataclass for quantization results returned by each layer.

::: onecomp.quantizer._quantizer.QuantizationResult
    options:
      show_source: false

## ResultLoader

Loader for reading saved quantization results without performing quantization.

::: onecomp.quantizer._quantizer.ResultLoader
    options:
      show_source: false
