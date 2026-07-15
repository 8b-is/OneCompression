# OneComp

**OneComp** is a research library for post-training compression of large language models (LLMs).
It provides weight quantization (GPTQ, QEP, LPCD, JointQ, AutoBit, DBF), KV-cache compression
(TaperKV, ThinKV, PM-KVQ, HayateQuant/TurboQuant), and lossless addressable context compaction
under a unified Python API, with built-in MCP and vLLM integrations for deployment.

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
| **HayateQuant** | Hybrid rotated low-bit KV codec with rate-aware K/V allocation and native vLLM integration. | (proposed) |
| **TaperKV** | Static age-graded precision ladder; no eviction; length-proportional allocation prevents collapse at long context. | (proposed) |
| **ThinKV** | Per-thought static precision (R/E=4-bit, T=2-bit); TBQ buffer, KDE-calibrated threshold, and attention-sparsity classification. | Ramachandran et al., ICLR 2026 |
| **PM-KVQ** | Progressive dynamic quantization: fixed absolute high-precision window, older KV shrinks to a floor; block-wise layer allocation; K≥V. | [arXiv:2505.18610](https://arxiv.org/abs/2505.18610) |
| **HALO / TurboQuant** | Data-free rotated scalar quantization using Lloyd-Max codebooks and orthogonal rotations. | — |
| **StreamingLLM / SnapKV / H2O** | Eviction baselines. | — |

### Agent Context Compaction

`onecomp.compaction` implements ARC (Addressable Recall for Compaction): long observations are
stored losslessly and compacted into bounded citations, then restored by stable observation ID.
The `onecomp-mcp` stdio server exposes the same session-based operations to MCP-capable single-
and multi-agent systems such as OpenHands. See the [MCP server guide](docs/user-guide/mcp-server.md).

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

See the [installation guide](docs/getting-started/installation.md) for details.

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
import onecomp.kvcomp as kv
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("...", device_map="cuda")
kv_cfg = kv.KVCompressionConfig.from_recipe("taperkv")
out = kv.kv_compressed_generate(model, kv_cfg, input_ids=ids, max_new_tokens=512)
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

**MCP server:**

```bash
onecomp-mcp
# equivalent: python -m onecomp.mcp
```

**vLLM serving:**

```bash
vllm serve ./quantized --kv-cache-dtype fp8
```

## Supported Models

| Architecture | Verified Models |
|---|---|
| Llama | Llama-2, Llama-3 |
| Qwen3 | Qwen3-0.6B – 32B |
| Gemma | Gemma 2, 3, 4 |

## Benchmarks

### KV-cache compression — long-context retrieval and reasoning

Results on **llama-3.1-8b-instruct**, iso-memory (stored/fp16 ≈ 0.28),
context ∈ {2048, 4096, 8192}. Needle = 63 cells; RULER = 15 multi-key cells;
GSM8K = 40 samples; PPL = WikiText-2 (16 windows).

| Method | Needle | RULER | GSM8K p@1 | PPL |
|---|---:|---:|---:|---:|
| FullKV | 1.00 | 1.00 | 0.825 | 6.10 |
| StreamingLLM (evict) | 0.30 | 0.60 | 0.825 | 6.10 |
| ThinKV (faithful) | 1.00 | 1.00 | 0.575 | 6.21 |
| PM-KVQ | 1.00 | 1.00 | 0.800 | 7.42 |
| **TaperKV** (proposed) | **1.00** | **1.00** | **0.825** | 6.24 |

Full results, including the DeepSeek-R1-Distill reasoning models and the Ministral-8B
retrieval model: [benchmark/kvcomp-longcontext/README.md](benchmark/kvcomp-longcontext/README.md).

### KV codec speed-quality — HayateQuant vs TurboQuant

HayateQuant/HALO operating points and TurboQuant baselines are documented with their
quality, memory-accounting, and runtime methodology in
[benchmark/kvcomp-codec/README.md](benchmark/kvcomp-codec/README.md).

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
