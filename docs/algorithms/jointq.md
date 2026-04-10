# JointQ

JointQ is a post-training quantization method that jointly optimizes integer weight assignments
and scale parameters to minimize the layer-wise reconstruction error.

## Algorithm

For each linear layer, JointQ minimizes:

\[
\min_{\hat{W}} \| Y - \hat{W} X^T \|_F^2
\]

where \(Y = WX^T\) is the full-precision output. Unlike GPTQ which quantizes column-by-column,
JointQ optimizes weight assignments and scale/zero-point parameters simultaneously using
local search.

The weight is decomposed as:

\[
\hat{W}_{i, g} = s_{i,g} \cdot (a_{i,g} - z_{i,g})
\]

where \(s\) is the scale, \(z\) is the zero-point, and \(a\) is the integer assignment,
with group index \(g\) for group-wise quantization.

### Initialization Strategies

JointQ supports multiple initialization strategies for the local search:

1. **Clip-Optimize**: Finds optimal clipping range, then quantizes
2. **Clip-Optimize with Error Propagation**: Adds GPTQ-style error propagation to initialization
3. **GPTQ**: Uses GPTQ solution as the starting point for joint optimization

### Regularization

To prevent overfitting to calibration data, JointQ applies Tikhonov regularization:

\[
X^T X + n \lambda I
\]

where \(\lambda\) controls the regularization strength (default: 0.2). This stabilizes
the optimization, especially when the number of calibration samples is small relative
to the model dimension.

## Parameters

| Parameter               | Type    | Description                                         | Default  |
|-------------------------|---------|-----------------------------------------------------|----------|
| `bits`                  | `int`   | Quantization bit-width (1--4)                       | `4`      |
| `symmetric`             | `bool`  | Symmetric quantization                              | `False`  |
| `group_size`            | `int`   | Group size for group-wise quantization              | `128`    |
| `batch_size`            | `int`   | Batch size for processing rows (None = all at once) | `None`   |
| `regularization_lambda` | `float` | Tikhonov regularization strength                    | `0.2`    |
| `actorder`              | `bool`  | Reorder columns by activation magnitude             | `False`  |
| `device`                | `torch.device` | Device for computation                       | `None`   |

## Usage

### Basic 4-bit quantization

```python
from onecomp import JointQ, ModelConfig, Runner

model_config = ModelConfig(
    model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    device="cuda:0",
)
jointq = JointQ(bits=4, group_size=128)

runner = Runner(model_config=model_config, quantizer=jointq, qep=False)
runner.run()
```

### With activation ordering

```python
jointq = JointQ(bits=4, group_size=128, actorder=True)
```

### Symmetric quantization

```python
jointq = JointQ(bits=4, symmetric=True, group_size=128)
```

## Notes

- JointQ requires GPU for computation (CUDA-based local search).
- Group-wise quantization (`group_size > 0`) is recommended for accuracy.
- The `batch_size` parameter controls memory usage: smaller values reduce peak GPU memory
  at the cost of slower processing.
- JointQ currently supports dequantized-model evaluation only (not packed quantized inference).
