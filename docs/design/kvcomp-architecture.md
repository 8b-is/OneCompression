# KV-cache compression architecture

OneComp treats KV compression as an independent runtime axis. A
`KVCompressionConfig` can contain a codec, a selector, or a policy, but every
combination is validated before a cache or vLLM backend is created. Unsupported
compositions fail rather than silently falling back to an uncompressed cache.

## Public layers

- `onecomp.kvcomp.codec`: fixed-representation codecs such as TurboQuant,
  HayateQuant, affine integer quantization, and FP8.
- `onecomp.kvcomp.recipe`: runtime policies such as TaperKV.
- `onecomp.kvcomp.cache`: Hugging Face cache implementations and accounting.
- `vllm_plugins.kvcomp`: serialized-recipe validation and native serving
  backends.
- `onecomp.kvcomp.eval`: causal PPL, zero-shot, retrieval, reasoning, and exact
  logical-memory accounting.

The top-level serialized block is:

```json
{
  "kv_compression_config": {
    "codec": null,
    "selector": null,
    "policy": {"name": "taperkv"},
    "residual_window": 0
  }
}
```

Model and dataset locations are never part of a recipe. They belong to the
calling application or benchmark manifest.

## HayateQuant

HayateQuant has one public method name, `hayatequant`, and three adopted rates:

| average rate | key code bits / 2-vector | value code bits / 2-vector |
|---:|---:|---:|
| 2 | 5 | 3 |
| 3 | 7 | 5 |
| 4 | 9 | 7 |

The recipe uses shipped Gaussian codebooks, rate-selected orthogonal rotation,
NC key scaling, iterative least-squares value scaling, exact assignment for
small/decode work, and the fixed candidate table for sufficiently large CUDA
prefills. The public builder is `build_hayatequant(rate)`.

The Hugging Face path exposes a logical payload for quality evaluation. The
vLLM path stores packed K/V codes and fp16 role scales, then computes attention
directly from that packed layout with Triton kernels. Backend runtime evidence
is available during isolated validation runs through
`KVCOMP_VLLM_RUNTIME_EVIDENCE`.

## TaperKV

TaperKV is a separate no-eviction policy. Key and value precision follow
independent descending age ladders while preserving key precision greater than
or equal to value precision. Sink and recent windows stay at the high endpoint.

The cache records the precision already applied to every resident token/page.
Regrading is monotonic: a token is transformed only when its requested width
decreases. This prevents repeated same-width fake quantization from accumulating
drift. Block mode leaves a partially written block in model precision and
requires `residual_fp16=True`.

The Hugging Face cache reports exact payload and affine-metadata ledgers. The
current vLLM backend vectorizes regrading on GPU but retains model-dtype physical
pages; its evidence therefore calls itself a semantic backend, not a packed
memory implementation. Prefix caching is disabled because shared pages cannot
carry request-relative ages safely. Eager execution is also required because
CUDA-graph replay would bypass the Python precision-ledger cadence.

## vLLM mapping contract

The plugin reads `kv_compression_config` from a local model's `config.json`.

- Canonical HayateQuant rates 2/3/4 install the packed HayateQuant backend.
- Standalone `taperkv` and `block_taperkv` install the TaperKV backend.
- TaperKV plus a codec or selector is rejected.
- A positive residual window is rejected for uniform native codec layouts.
- OneComp TurboQuant maps to a vLLM deployment preset only after explicit
  opt-in because the two implementations are not bit-identical.
- Unknown codecs, selectors, and unsupported policies fail closed.

## Benchmark and artifact policy

`benchmark/kvcomp-codec/reproduce_hayatequant_paper.py` is the single
HayateQuant/TurboQuant paper-grid orchestrator. TaperKV retrieval checks are in
`benchmark/kvcomp-longcontext/taperkv_benchmark.py`. Both require an external
output root. Results, model weights, Hydra outputs, and Slurm logs are ignored;
only source, manifests, tests, and the small canonical HayateQuant codebooks are
versioned.
