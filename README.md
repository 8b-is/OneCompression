# OneComp

**OneComp** is a research library for post-training compression of large language models (LLMs).
It provides weight quantization (GPTQ, QEP, LPCD, JointQ, AutoBit, DBF) and
KV-cache compression (HayateQuant, TurboQuant, TaperKV, ThinKV, PM-KVQ) under a unified Python API,
with built-in vLLM plugin integration for deployment.

Full documentation: **[https://FujitsuResearch.github.io/OneCompression/](https://FujitsuResearch.github.io/OneCompression/)**

## Algorithms

### Weight Quantization

| Algorithm | Description | Reference |
|---|---|---|
| **QEP** | Quantization Error Propagation: corrects inter-layer quantization error by propagating residuals to subsequent layers. | [Arai & Ichikawa, NeurIPS 2025](https://openreview.net/forum?id=a3l3K9khbL) |
| **LPCD** | Layer-Projected Coordinate Descent: extends layer-wise PTQ to arbitrary submodules via relaxed objectives and projection. | [Ichikawa et al., arXiv:2512.01546](https://arxiv.org/abs/2512.01546) |
| **GPTQ** | Second-order Hessian-based PTQ. | — |
| **JointQ** | Joint optimisation of weight assignments and scale parameters. | — |
| **AutoBit** | ILP-based mixed-precision bitwidth assignment under a VRAM budget. | — |
| **DBF / Block-wise PTQ** | Phase-1 greedy and CBQ cross-block sliding-window distillation. | — |

### KV-Cache Compression

| Algorithm | Description | Reference |
|---|---|---|
| **TaperKV** | Static age-graded precision ladder; no eviction; length-proportional allocation prevents collapse at long context. | (proposed) |
| **ThinKV** | Per-thought static precision (R/E=4-bit, T=2-bit); TBQ buffer + KDE-calibrated L\* threshold + attention-sparsity thought classification. | [Ramachandran et al., ICLR 2026](https://openreview.net/...) |
| **PM-KVQ** | Progressive dynamic quantization: fixed absolute high-precision window, older KV shrinks to a floor; block-wise layer allocation; K≥V. | [arXiv:2505.18610](https://arxiv.org/abs/2505.18610) |
| **HayateQuant** | Data-free block-VQ with fixed K/V-asymmetric rates, rate-selected rotation, and role-aware scales. | (proposed) |
| **TurboQuant** | Data-free rotated scalar Lloyd-Max quantization; Algorithm-1 quality and public-API speed baseline. | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| **FibQuant / BlockQuant** | Independent method-faithful quality baselines; BlockQuant remains descriptive where its public recipe cannot be fully matched. | — |
| **StreamingLLM / SnapKV / H2O** | Eviction baselines. | — |

### Agent Context Compaction

`onecomp.compaction` implements ARC (Addressable Recall for Compaction): long
tool observations are stored losslessly and replaced in the model-visible
history by bounded citations that can be recalled by stable observation ID.
The `onecomp-mcp` stdio server exposes the same session-based operations to
MCP-capable hosts. OneComp also provides an OpenHands condenser and PostToolUse
adapters for Claude Code and Codex. See the [ARC compaction guide](docs/user-guide/arc-compaction.md)
and [MCP server guide](docs/user-guide/mcp-server.md).

## Installation

```bash
# GPU (recommended)
uv sync --extra cu128 --extra dev

# With vLLM (requires CUDA 13.0)
uv sync --extra cu130 --extra vllm
```

Or with pip:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

See [installation guide](docs/getting-started/installation.md) for details.

## Quick Start

**Weight quantization (one line):**
```python
from onecomp import Runner
Runner.auto_run(model_id="meta-llama/Llama-3.1-8B")
```

**GPTQ with QEP:**
```python
from onecomp import Runner
runner = Runner(model_id="meta-llama/Llama-3.1-8B", quantizer="gptq", qep=True)
runner.run()
runner.save_quantized_model("./quantized")
```

**KV-cache compression:**
```python
from onecomp.kvcomp import KVCompressionConfig, TaperKVPolicy, kv_compressed_generate
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("...", device_map="cuda")
kv_cfg = KVCompressionConfig(policy=TaperKVPolicy(), residual_window=0)
out = kv_compressed_generate(model, kv_cfg, input_ids=ids, max_new_tokens=512)
```

**Joint weight + KV compression:**
```python
from onecomp import Runner
from onecomp.kvcomp import KVCompressionConfig

runner = Runner(model_id="...", quantizer="gptq",
                kv_config=KVCompressionConfig.from_recipe("fp8"))
runner.run()
# config.json carries both quantization_config and kv_compression_config
runner.save_quantized_model("./joint")
```

**HayateQuant:**
```python
from onecomp.kvcomp import KVCompressionConfig
from onecomp.kvcomp.codec.hayate import build_hayatequant

kv_cfg = KVCompressionConfig(codec=build_hayatequant(3), residual_window=0)
```

**vLLM serving:**
```bash
# The OneComp plugin reads kv_compression_config from the saved config.json.
vllm serve ./quantized
```

## Supported Models

| Architecture | Verified Models |
|---|---|
| Llama | Llama-2, Llama-3 |
| Qwen3 | Qwen3-0.6B – 32B |
| Gemma | Gemma 2, 3, 4 |

## Benchmarks

The reproducible HayateQuant/TurboQuant quality and vLLM speed grid is in
[`benchmark/kvcomp-codec`](benchmark/kvcomp-codec/README.md). The standalone
TaperKV long-context correctness grid is in
[`benchmark/kvcomp-longcontext`](benchmark/kvcomp-longcontext/README.md).
Both require caller-selected model and output roots; generated results and
Slurm logs are intentionally not versioned.

## License

See [LICENSE](./LICENSE) for details.

## Citation

OneComp technical report:
```bibtex
@misc{ichikawa2026onecomponelinerevolutiongenerative,
      title={OneComp: One-Line Revolution for Generative AI Model Compression},
      author={Yuma Ichikawa and Keiji Kimura and Akihiro Yoshida and Yudai Fujimoto and Hiroki Tokura and Yamato Arai and Yoshiyuki Ishii and Yusei Kawakami and Genki Shikada and Achille Jacquemond and Yoshihiko Fujisawa and Katsuki Fujisawa and Takumi Honda and Akira Sakai},
      year={2026},
      eprint={2603.28845},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.28845},
}
```

QEP (Quantization Error Propagation):
```bibtex
@inproceedings{arai2025quantization,
  title={Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization},
  author={Yamato Arai and Yuma Ichikawa},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=a3l3K9khbL}
}
```

LPCD (Layer-Projected Coordinate Descent):
```bibtex
@article{ichikawa2025lpcd,
  title={LPCD: Unified Framework from Layer-Wise to Submodule Quantization},
  author={Yuma Ichikawa and Yudai Fujimoto and Akira Sakai},
  journal={arXiv preprint arXiv:2512.01546},
  year={2025},
  url={https://arxiv.org/abs/2512.01546}
}
```
