# OneComp KV Cache Compression (`onecomp.kvcomp`) — Design Document

> Status: **RFC + v0 skeleton implemented** (branch `lab/kvcomp`)
> Scope: design of a KV-cache compression sub-package that coexists with the
> existing weight quantizer (`onecomp/quantizer`).
> References: `onecomp-ref/{260528.md, KVCache-Factory, ThinKV (paper), turboquant-repos}`, NVIDIA KVPress, vLLM.

> ## Implementation status (2026-06-01, `lab/kvcomp`)
>
> | Area | Implemented | Verified |
> |---|---|---|
> | Core abstractions `KVCodec/KVSelector/KVPolicy/ComposedPress` + registry + config (to/from_dict) | done | `01`, `03` |
> | codec: fp8 / int8 / int4 / int2 / kivi (group-wise) / turboquant (data-free) / nvfp4 / ternary | done | `01`, `02`, `05`, `09` |
> | select: streamingllm / recent / snapkv / h2o, prune: think | done | `01`, `04` |
> | recipe: ThinKV (ThoughtClassifier + ThinKVEngine [TBQ buffer / TBE] + ThinKVPolicy + KDE calibration + L\*) | done | `01`, `04`, `06` |
> | **baseline: PM-KVQ** (recipe `pmkvq` + `PMKVQCache`, progressive dynamic precision + block-wise layer allocation + K>V) | done | `01`, `smoke_pmkvq`, `kv_benchmark` |
> | **shared base `GradedQuantCache`** (per-token variable bits, sink/skip, accounting; shared by TaperKV/PM-KVQ) | done | `01` |
> | **proposed: TaperKV** (recipe `taperkv` + `TaperKVCache`, age-graded precision, no eviction, K>V) | done (verified, wins, multi-arch) | `01`, `kv_benchmark` |
> | **proposed: HALO** (codec `halo`, preconditioning off by default; `auto` rotation + skip-preconditioning is 2.9× faster than TurboQuant at equal quality; role-asymmetric NC/LS scale is a validated *quality* lever) | done (wins on speed-quality); **superseded by HayateQuant, kept as the baseline it must beat** | `01`, `speed_halo`, `codec_benchmark` |
> | **proposed: HayateQuant** (codec `hayatequant` — strict superset of TurboQuant + HALO: data-free block-VQ + Hadamard/block-dense hybrid rotation + HALO's role-asymmetric scale, all storage-neutral) | done (`block_size=1` reproduces TurboQuant/HALO bit-for-bit; block-VQ strictly reduces distortion at equal bits) | `verify_hayatequant_vq_gain`, `test_kvcomp.py`, `codec_benchmark` |
> | HF integration `CompressedDynamicCache` + `ThinKVCache` (live TBQ) + `StreamingLLMEvictCache` (eviction baseline) | done | `02`, `05`, `07`, `08`, `kv_benchmark` |
> | eval: `eval/{metrics,ppl,zeroshot,reasoning,longcontext}` (PPL / 0-shot / GSM8K · MATH500 / **Needle/RULER**) | done | `08`, `09`, `kv_benchmark` |
> | **main experiment `benchmark/kvcomp-longcontext/`** (Full/uniform/evict/ThinKV/PM-KVQ/TaperKV, iso-memory) | done | llama-3.1-8b, R1-Distill-Qwen-7B/1.5B |
> | **main experiment `benchmark/kvcomp-codec/`** (HayateQuant vs TurboQuant/HALO: distortion/PPL/overhead) | done | llama-8b / Qwen-1.5B |
> | Runner `kv_config` -> `config.json` dual-config | done | `03` |
> | vLLM plugin: FP8 -> native `kv_cache_dtype` mapping + entry-point (`vllm.general_plugins`) | done | `10` |
> | Custom AttentionBackend for low-bit/eviction in vLLM, LongBench, Triton kernels, ILP bit allocation | todo | roadmap |
>
> Verification is in-repo and self-contained: the unit suite `tests/onecomp/kvcomp/test_kvcomp.py` plus the CPU-only array-level proofs under `benchmark/kvcomp-longcontext/paper/tests/` (`verify_halo_asym_scale.py`, `verify_halo_outlier.py`, `taper_schedule.py`). Benchmark models follow the competing papers (TurboQuant: Llama-3.1-8B-Instruct + Ministral-8B; ThinKV/PM-KVQ: DeepSeek-R1-Distill family). Gemma is intentionally excluded — no KV-cache-compression baseline reports Gemma KV-cache results.
>
> ThinKV is mapped directly onto the paper (Ramachandran et al., ICLR 2026 submission):
> its generation-loop code listing, problem formulation (TBQ/TBE), and Algorithm 1
> (KDE calibration). Concretely we implement the full-precision group buffer for TBQ
> (`g=16`), per-thought precision R4E4T2 (NVFP4/ternary, with R8=FP8 optional), threshold
> extraction from KDE modes/minima and L\* selection from the three-mode layers, the TBE
> Case-1 retention schedule `R={64,32,16,8,4}` annealing (keep=R[n]), Case-2 oldest /
> lowest-importance eviction, post-RoPE key K-means eviction, and L\*-averaged thought
> sharing during decode.
> `06_thinkv_repro.py` reproduces, on a same-family reasoning model, Obs1a (multimodal
> sparsity), Obs1b (T>R>E), Algorithm 1 (L\* + thresholds), keyword correspondence, and
> the TBQ+TBE budget adherence (compression ratio < 5% KV) with Pareto dominance.
>
> The live HF cache path is verified for per-token codecs. ThinKV's TBQ (per-thought
> precision) is integrated into live `model.generate` via `ThinKVCache` (a `DynamicCache`
> subclass) plus an attention forward-hook for thought classification; `08_thinkv_gsm8k.py`
> measures GSM8K pass@1 on hardware for FullKV vs ThinKV-R4E4T2/R8E4T2 (R8 matches or
> exceeds R4 and reproduces near-FullKV accuracy). KIVI's group-wise (group=32) quantization
> is implemented in `_quant_primitives.quantize_affine`; WikiText PPL at 2-bit improves from
> 35.7 to 8.9 (FullKV 7.9), reproducing the paper's near-lossless 2-bit (`09`). TurboQuant's
> distortion-rate (Theorem 1) is reproduced to D_mse={0.36, 0.117, 0.03, 0.009}, along with
> outlier mixed-precision / QJL / WikiText PPL (4-bit near-lossless, rotation improving
> compression over naive INT4) (`09`). The vLLM plugin maps the FP8 codec onto the native
> `kv_cache_dtype='fp8_e4m3'`; its mapping/serialization is verified in `10`.
>
> ### Proposed methods (2026-06-01, OneComp original; benches in `benchmark/kvcomp-longcontext/` and `benchmark/kvcomp-codec/`)
>
> **TaperKV** (`recipe/taperkv.py` + `cache/taper_cache.py`) — **the verified proposed
> method (wins)**: it replaces ThinKV's progressive *eviction* with progressive
> *quantization*. Every token is retained, but its precision *tapers* with age (sink +
> recent tokens stay at high bits; older tokens step down to a non-zero floor `b_min>0`,
> e.g. ladder=[8,4,3,2]). Because nothing is evicted, the needle survives long-context
> retrieval. Following the KVTuner observation, keys get more bits than values
> (key_bits >= value_bits). The method is length-preserving and RoPE-safe (simulated
> quantization).
> **Multi-architecture, iso-memory results (2026-06-01):**
> - **llama-3.1-8b-instruct** (Needle depth7 × seed3 × ctx{2048,4096,8192}, iso-mem ~0.28):
>   FullKV=1.00, StreamingLLM (evict, same mem)=0.30, **TaperKV=1.00**; RULER multi-key:
>   eviction=0.60 vs **TaperKV=1.00**; GSM8K p@1: FullKV=0.825 = **TaperKV=0.825**; PPL 6.24 vs 6.10.
> - **DeepSeek-R1-Distill-Qwen-7B** (long chain-of-thought, max_new=2048): with outlier
>   layers held in FP16, every method tracks FullKV within evaluation noise; **TaperKV reaches
>   FullKV-level PPL (53.6 vs 50.8) at 3.4× compression**.
> At equal memory, eviction drops the needle whenever it lands in the discarded middle, whereas
> TaperKV preserves retrieval exactly. The structural advantage is robust across all three
> architectures. The main experiment lives in `benchmark/kvcomp-longcontext/`.
>
> **PM-KVQ** (`recipe/pmkvq.py` + `cache/pmkvq_cache.py`, registered as `"pmkvq"`) —
> **baseline (arXiv:2505.18610)**: progressive, dynamic-precision quantization for long
> chain-of-thought. (1) **Progressive quantization**: a *fixed absolute* recent window is
> kept at high bits, and older blocks shrink toward a floor as memory saturates (precision
> falls dynamically with decode progress; a token's bit-width drops as it ages).
> (2) **Block-wise memory allocation**: sensitive layers get more bits (the IP solution is
> supplied directly via `sensitive_layers`). K>V is enforced (key floor >= 4, since per-token
> 2-bit keys are catastrophic). The contrast with TaperKV: PM-KVQ uses an **absolute window**,
> so the longer the context the more tokens collapse to the floor, whereas TaperKV's
> **length-proportional ladder** keeps mid-context tokens at intermediate precision — the
> axis along which long-context retrieval separates the two. PM-KVQ shares `GradedQuantCache`
> with TaperKV.
>
> **HALO** (`codec/halo.py`, registered as `"halo"`) — **a codec that beats TurboQuant on
> speed-quality**: it extends TurboQuant's rotation + data-free Lloyd-Max with an optional
> online per-channel affine preconditioning step (mean-center / whiten, EMA O(D)) applied
> before rotation. However, **a multi-architecture sweep (Llama / Qwen2 / Cohere) did not
> confirm a quality win** from preconditioning:
> - Overhead: at D=128 a dense matmul rotation is faster than the WHT (0.24 vs 0.81 ms/encode);
>   Hadamard only helps at very large D or on non-matmul hardware.
> - Real-KV distortion: HALO (preconditioning off) is identical to TurboQuant; value
>   preconditioning improves MSE by less than 1%.
> - PPL: parity on Llama (2-bit 11.58 vs 11.89, 3-bit 8.46 vs 8.39); on Cohere, preconditioning
>   ON *worsens* 2–3 bit (lower value MSE does not imply lower softmax output error). Only on
>   synthetic massive-activation distributions does preconditioning help ~3×, and real values
>   are not in that regime.
> We therefore **default to preconditioning OFF (a faithful, PPL-neutral TurboQuant), and treat
> preconditioning / forced Hadamard as opt-in research levers**, correcting the earlier
> "HALO wins on quality" claim. The math: mean-centering is neutral to attention output because
> the weights sum to 1, while whitening amplifies output error on high-variance channels through
> the 1/w un-whitening.
> **HALO does win on speed-quality, however** (main experiment `benchmark/kvcomp-codec/`): as a
> quality-neutral online optimization, (1) `rotation_mode="auto"` picks the cheapest orthogonal
> transform per head dimension (measured: dense GEMM beats WHT across all D <= 512), (2) the
> `(x-mu)*w` step is skipped entirely when preconditioning is off, and (3) the sign vector is
> cached. As a result HALO(auto) is **2.9× faster than TurboQuant's shipped default (Hadamard)
> at identical quality** (llama-8b: 4-bit PPL 7.706 == 7.706, 0.356 vs 1.020 ms/encode; 2/3-bit
> similar).
> Faithful live per-channel-key (KIVI), token eviction (TBE), and low-bit codecs running inside
> vLLM all require a custom AttentionBackend and are scheduled for v0.2–0.3 (see § HF integration
> / § vLLM integration).
>
> **HayateQuant** (`codec/hayatequant.py`, registered as `"hayatequant"`) — **closes the gap this
> document flags above as "Lever 5 / future work"** and replaces `halo` as the recommended codec.
> HALO's own honest finding above is that *rotation choice and preconditioning* are a
> speed lever, not a quality lever — the one HALO addition that *did* survive multi-architecture
> validation is the **role-asymmetric reconstruction scale** (NC for keys — preserves the attention
> logit via `cos(theta)` instead of `cos^2(theta)` — and LS/LS-iter for values, the correct
> objective for a linearly-consumed tensor). HayateQuant inherits that lever verbatim
> (`codec/_scale.py`, shared by both codecs) and adds the genuinely new, previously-unexplored one:
> - **Data-free block vector quantization** (`codec/_vq.py`): TurboQuant's own paper proves its
>   *scalar* Lloyd-Max codebook is within a constant factor (`<= sqrt(3*pi/2) ~= 2.7x`) of the
>   distortion-rate optimum; classical high-rate quantization theory (Gersho & Gray 1992) identifies
>   the residual as exactly the **space-filling / shape gain** any *joint* (vector) quantizer of
>   dimension `>1` can capture and an independent scalar quantizer cannot. HayateQuant fits a
>   `block_size`-dimensional codebook by batched Lloyd's algorithm (multi-restart K-means) to
>   synthetic samples from TurboQuant's own source law (i.i.d. Gaussian, or the exact finite-`d`
>   Beta/Dirichlet marginal) — still fully data-free, just generalized from dimension 1 to
>   `block_size`. `block_size=1` reduces to the scalar codebook exactly, so HayateQuant can never do
>   worse than TurboQuant/HALO at the same bit-rate.
> - **Hybrid rotation** (`codec/rotation.py`: `hybrid_rotate`/`unrotate`): block-VQ needs each
>   rotated block to be close to *isotropic* for the codebook to fit well, but a fixed Hadamard
>   transform (randomized only by a global sign flip) is not Haar-random at the block level.
>   HayateQuant composes the cheap global signed-Hadamard-GEMM with independent small dense
>   Haar-random rotations per block (`O(D log D + D*b)` vs. `O(D^2)` for a single full dense
>   rotation), which is the concrete answer to the "can we interpolate between Hadamard's speed and
>   Dense's isotropy?" question raised during HayateQuant's design — `block_size` is literally that
>   interpolation knob (`b=1` recovers pure Hadamard, `b=D` recovers pure dense).
> Verified CPU-only and data-free in `benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py`:
> block-VQ distortion decreases as `block_size` grows `1->2->4->8` at equal bits (both the Gaussian
> and exact Beta source laws), the hybrid rotation's block marginals are measurably more isotropic
> than Hadamard-only on structured (non-i.i.d.) input, and outlier mixed-precision (`n_outlier`,
> inherited from HALO/TurboQuant) keeps the guarantee under real massive-activation-style
> anisotropy. `HALOCodec` is kept in the codebase (not deleted) purely as the baseline
> `codec_benchmark.py` must show HayateQuant beating — new work should use `hayatequant`.

---

## 1. Purpose and scope

OneComp is currently a **weight-only Post-Training Quantization framework**. The abstraction
`Quantizer.quantize_layer(module) -> QuantizationResult` (`onecomp/quantizer/_quantizer.py`)
acts only on `nn.Linear` weight tensors, runs once during a forward-hook calibration pass, and
is saved as packed safetensors plus a `quantization_config`.

KV-cache compression is an **orthogonal second axis**:

| Aspect | weight PTQ (existing) | KV-cache compression (this design) |
|---|---|---|
| Target | `nn.Linear.weight` (static) | K/V activations produced by attention at runtime (dynamic) |
| Decision time | once, offline | per inference step / per request |
| Core abstraction | `quantize_layer` | `encode/decode` (codec), `score/select` (selector), `plan` (policy) |
| Artifact | packed weights + `quantization_config` | recipe + calibration stats (scale, thresholds, head classes, segment budget) |
| vLLM integration | `register_quantization_config` → `LinearMethod` swap | Platform plugin → `AttentionBackend`/`AttentionImpl` |

It therefore **cannot be solved by subclassing `Quantizer`**. At the same time, the development
conventions OneComp has established (dataclass config, `flag_*`, registry, Runner orchestration,
calibration reuse, eval integration, vLLM plugin entry-point convention) are **assets to reuse
directly**.

**Design direction:**
> Introduce a second-axis sub-package `onecomp/kvcomp/` and reuse the Runner, calibration,
> registry, config conventions, vLLM entry point, and eval as a shared layer. The key
> differentiator (the killer feature) is being able to **compose weight quantization and KV
> compression in a single `Runner.run()`**.

---

## 2. Design principles (consistency with OneComp)

1. **`Quantizer` is untouched.** KV uses separate bases (`KVCodec`/`KVSelector`/`KVPolicy`).
2. **Static vs dynamic is distinguished by `flag_calibration`, not by folder.** OneComp keeps
   RTN (no calibration) and GPTQ (calibration) together in `quantizer/` and distinguishes them
   by a flag (`quantizer/rtn/_rtn.py`). KV codecs follow the same convention (§7).
3. **Separate codec (quantization) from selector (eviction).** KVCache-Factory split these into
   two incompatible paths and could not compose them; we compose them explicitly via
   `ComposedPress`/`KVPolicy`.
4. **Avoid monkeypatching.** KVCache-Factory's `replace_llama()` if/elif plus global class
   overrides has no registry, duplicates code, and is hard to test in isolation. We use a
   **registry + `transformers.Cache` subclass + local hooks** instead.
5. **Never declare victory on compression ratio alone.** Eval always returns
   (memory, latency, quality) together.
6. **Separate the research layer (torch, correctness first) from the kernel layer (Triton,
   performance).**
7. **Minimal intrusion.** Changes to existing files are limited to adding `kv_config` to
   `Runner.__init__` and a few lines of wiring in `create_quantized_model`/`save_quantized_model`.

---

## 3. Taxonomy (three axes)

KV-compression methods can be classified uniquely along three axes:

- **target**: what is compressed → `codec` (K/V dtype) / `select` (token) / `prune` (head-dim) / `sparse` (attention)
- **mechanism**: the folder hierarchy (matches target)
- **calibration**: `flag_calibration` (whether offline calibration is required). Online/data-free is False; calibration-based is True.

| Method | target (folder) | flag_calibration | Notes |
|---|---|---|---|
| FP8 / INT4 / INT2 | codec | False | baseline |
| KIVI | codec | False | online per-token/channel scale |
| NVFP4 / Ternary | codec | False | for ThinKV |
| **TurboQuant** | codec | False | **data-free** (codebook precomputed from a Beta distribution; application is online) |
| KVQuant | codec | True | Pre-RoPE K, non-uniform |
| **KVTuner** | codec | True | layer/head bits chosen by sensitivity + ILP |
| **PM-KVQ** | codec | True | block bits by ILP; precision steps 16→8→4→2 during inference |
| StreamingLLM / Recent / Sink | select | False | |
| H2O | select | True (light) | heavy hitter (attention stats) |
| SnapKV / PyramidKV | select | False-ish | prefill attention score |
| Expected Attention | select | True | estimates future query distribution |
| ThinK | prune | False | head-dim, query-driven pruning |
| Quest / SparQ / RocketKV-stage2 | sparse | False | dynamic selection per decode step |
| **ThinKV** | recipe (composed) | True | codec (NVFP4/Ternary) + select (segment K-means) + policy (thought classification) |

---

## 4. Architecture (three layers)

```
Layer 3 Runtime (serving)   vllm_plugins/kvcomp/     policy engine / online controller / AttentionBackend
Layer 2 Kernel              onecomp/kvcomp/kernels/   Triton dequant (int4/int2/fp8) / paged layout / sparse attn
Layer 1 Research (HF)       onecomp/kvcomp/{codec,select,prune,sparse,recipe,policy,cache}/
                                                      + reuse: onecomp/{calibration,model_config,utils,runner}
```

Researchers add methods at Layer 1; operators use Layers 2/3 (the same division of labor as the
existing OneComp).

### Folder layout

```text
onecomp/kvcomp/
├── __init__.py            # registry re-exports (mirrors quantizer/__init__.py)
├── _base.py               # KVCodec / KVSelector / KVPolicy / ComposedPress
├── _registry.py           # @register_kv_codec / _selector / _policy + build_*
├── config.py              # KVCompressionConfig, Budget, RequestState, SystemState, CompressionPlan
├── runner.py              # delegated target from Runner (thin wiring, runner_methods/ style)
├── cache/
│   ├── block.py           # KVBlock / CompressedKV / metadata
│   ├── compressed_cache.py# transformers.Cache subclass
│   ├── attention.py       # attention forward-hook (collects attention stats)
│   └── builders.py        # build_compressed_cache / kv_compressed_generate (no monkeypatch)
├── codec/                 # includes dynamic quantization (TurboQuant); distinguished by flag_calibration
│   ├── _base.py fp8.py int4.py int2.py kivi.py kvquant.py kvtuner.py pm_kvq.py
│   ├── nvfp4.py ternary.py turboquant.py qjl.py rotation.py
│   └── calibration_kv.py
├── select/  _base.py recent.py sink.py h2o.py snapkv.py expected_attention.py pyramid.py razor.py
├── prune/   think.py
├── sparse/  topk.py quest.py sparq.py rocketkv.py
├── recipe/  thinkv.py     # composed methods
├── policy/  _base.py static.py memory_budget.py latency_budget.py service_aware.py
├── budget/  ilp.py        # KVTuner/PM-KVQ/ThinKV bit allocation (reuses AutoBit ortools)
├── kernels/ triton/ torch_fallback.py
└── eval/    needle.py ruler.py longbench.py reasoning.py kv_metrics.py
```

---

## 5. Core abstractions (skeleton)

### 5.1 `_base.py`

```python
from dataclasses import dataclass, field
from logging import getLogger
import torch

@dataclass
class CompressedKV:
    """Compressed K/V plus metadata returned by a codec."""
    data: dict                       # quantization indices / packed bits, etc.
    scale: torch.Tensor | None = None
    zero: torch.Tensor | None = None
    outlier: dict | None = None
    residual: torch.Tensor | None = None   # most-recent FP window
    meta: dict = field(default_factory=dict)

@dataclass
class KVCodec:
    """Low-bit encode/decode of K/V tensors. The KV analogue of Quantizer."""
    name: str = None
    flag_calibration: bool = False
    bits: int = 8
    key_axis: str = "channel"        # KIVI: K=per-channel
    value_axis: str = "token"        # KIVI: V=per-token
    group_size: int = 64
    skip_layers: list[int] = field(default_factory=list)
    residual_window: int = 128

    def __post_init__(self):
        if self.name is None: self.name = type(self).__name__
        self.logger = getLogger(__name__)
    def validate_params(self): ...
    def calibrate(self, kv_stats, ctx): ...                  # only when flag_calibration
    def encode(self, k, v, ctx) -> CompressedKV: ...         # required
    def decode(self, c: CompressedKV, ctx) -> tuple[torch.Tensor, torch.Tensor]: ...  # required
    def score(self, q, c: CompressedKV, ctx) -> torch.Tensor | None: return None  # optional (LUT, etc.)
    def get_kv_config(self) -> dict:
        return {"name": self.name, "bits": self.bits, "group_size": self.group_size,
                "key_axis": self.key_axis, "value_axis": self.value_axis,
                "skip_layers": self.skip_layers, "residual_window": self.residual_window}

@dataclass
class KVSelector:
    """Retain important tokens and evict the rest (fires at the prefill boundary)."""
    name: str = None
    flag_needs_attention: bool = False    # True for SnapKV/H2O
    def __post_init__(self):
        if self.name is None: self.name = type(self).__name__
        self.logger = getLogger(__name__)
    def validate_params(self): ...
    def score(self, q, k, v, attn_stats, layer_idx) -> torch.Tensor: ...
    def select(self, scores, budget: "Budget") -> torch.Tensor: ...   # -> keep mask
    def get_kv_config(self) -> dict: ...

@dataclass
class KVPolicy:
    """Choose codec/selector/budget based on state. ThinKV switches per thought."""
    name: str = None
    def __post_init__(self):
        if self.name is None: self.name = type(self).__name__
    def plan(self, req: "RequestState", sys: "SystemState") -> "CompressionPlan": ...

@dataclass
class ComposedPress:
    """Bundles codec + selector + policy; the single integration point for cache/attention."""
    codec: KVCodec | None = None
    selector: KVSelector | None = None
    policy: KVPolicy | None = None
    residual_window: int = 128
    flag_calibration: bool = field(init=False, default=False)
    flag_needs_attention: bool = field(init=False, default=False)
    def __post_init__(self):
        self.flag_calibration = bool(self.codec and self.codec.flag_calibration)
        self.flag_needs_attention = bool(self.selector and self.selector.flag_needs_attention)
```

### 5.2 `_registry.py`

```python
_KV_CODECS: dict[str, type] = {}
_KV_SELECTORS: dict[str, type] = {}
_KV_POLICIES: dict[str, type] = {}

def register_kv_codec(name):
    def deco(cls): _KV_CODECS[name] = cls; return cls
    return deco
# register_kv_selector / register_kv_policy follow the same pattern

def build_codec(name, **kw): return _KV_CODECS[name](**kw)
def build_selector(name, **kw): return _KV_SELECTORS[name](**kw)
def build_policy(name, **kw): return _KV_POLICIES[name](**kw)
```

### 5.3 `config.py`

```python
@dataclass
class Budget:
    keep_ratio: float | None = None
    keep_tokens: int | None = None
    sink_tokens: int = 4
    recent_tokens: int = 256
    per_layer: dict[int, int] | None = None
    per_head: dict[tuple[int, int], int] | None = None

@dataclass
class RequestState:
    prompt_len: int = 0
    generated_len: int = 0
    max_new_tokens: int = 0
    task_type: str = "chat"        # chat/rag/summarization/code/math/reasoning/agent
    quality_slo: str = "balanced"  # strict/balanced/best_effort
    latency_slo_ms: int | None = None

@dataclass
class SystemState:
    free_gpu_mem_gb: float = 0.0
    kv_mem_gb: float = 0.0
    decode_tokens_per_sec: float = 0.0
    network_bandwidth_gbps: float | None = None
    gpu_arch: str = ""

@dataclass
class CompressionPlan:
    codec: "KVCodec | None" = None
    selector: "KVSelector | None" = None
    budget: Budget = field(default_factory=Budget)
    skip_layers: list[int] = field(default_factory=list)
    residual_window: int = 128

@dataclass
class KVCompressionConfig:
    codec: "KVCodec | None" = None
    selector: "KVSelector | None" = None
    policy: "KVPolicy" = None
    residual_window: int = 128
    def to_press(self) -> "ComposedPress":
        from ._base import ComposedPress
        from .policy.static import StaticPolicy
        return ComposedPress(self.codec, self.selector,
                             self.policy or StaticPolicy(), self.residual_window)
    def to_dict(self) -> dict:
        return {"codec": self.codec.get_kv_config() if self.codec else None,
                "selector": self.selector.get_kv_config() if self.selector else None,
                "policy": getattr(self.policy, "name", "static"),
                "residual_window": self.residual_window}
    @classmethod
    def from_dict(cls, d: dict) -> "KVCompressionConfig": ...   # for vLLM/HF loading
```

---

## 6. HF Transformers integration (`cache/`)

The **proven invariants** adopted from KVCache-Factory (without the monkeypatching):

- Eviction runs **once at the prefill boundary** (new K/V length == logical sequence length).
- Decode appends to the cache (quantizing into the buffer via the codec when needed).
- State such as `kv_seq_len` is reset in the `prepare_inputs_for_generation` equivalent.
- Per-layer budgets are controlled by passing the layer index (PyramidKV/ThinKV).

```python
# cache/compressed_cache.py
from transformers.cache_utils import Cache

class CompressedCache(Cache):
    """A thin wrapper over transformers.Cache that applies codec/selector in update()."""
    def __init__(self, press: "ComposedPress", model_config): ...
    def update(self, key, value, layer_idx, cache_kwargs=None):
        # 1) decide prefill boundary vs decode from key.shape[-2] and the logical length
        # 2) prefill: evict via selector.select(...) (attn_stats come from the hook)
        # 3) decode: keep within residual_window in FP, encode the overflow into the
        #            quantized buffer via the codec (KIVI style)
        # 4) on return, restore via codec.decode, or score without dequant via codec.score
        ...

# cache/builders.py  (no monkeypatch: just hand the cache to generate)
def build_compressed_cache(model, kv_config: "KVCompressionConfig", sink_tokens=4):
    return CompressedDynamicCache(
        codec=kv_config.codec, config=model.config,
        skip_layers=kv_config.codec.skip_layers if kv_config.codec else None,
        sink_tokens=sink_tokens,
    )                                              # passed to model.generate(past_key_values=...)
    return model
```

> **Note (transformers >= 5.5):** OneComp depends on `transformers>=5.5.0` (`pyproject.toml`).
> 5.5 uses a layered Cache (`CacheLayerMixin` family). `cache/` stays a thin adapter so API
> changes are confined to a single file.

---

## 7. Placement of "dynamic quantization (TurboQuant etc.)" — settled decision

**Place it in `onecomp/kvcomp/codec/turboquant.py` and declare `flag_calibration=False`
(data-free). Do not split into `static/` and `dynamic/` folders.**

Rationale:
- OneComp itself distinguishes static/dynamic by a flag, not a folder (RTN and GPTQ live
  together in `quantizer/`).
- TurboQuant's codebook is precomputed from a Beta distribution (data-free), but **its
  application is sequential at decode time**. The accurate distinction is "is the *parameter
  computation* an offline calibration?" — which is exactly what `flag_calibration` expresses —
  not "is the application dynamic?" (`turboquant-repos/OnlyTerp/src/cache.py`,
  `tonbistudio/turboquant/turboquant.py`).
- A folder split would fragment the registry and leave no home for mixed methods such as
  `KVTuner` (with calibration) plus online outlier detection.

```python
@register_kv_codec("turboquant")
@dataclass
class TurboQuantCodec(KVCodec):
    flag_calibration: bool = False
    bits: int = 4
    rotation_mode: str = "hadamard"   # hadamard / dense
    use_qjl: bool = False             # off by default (agreed harmful under softmax attention)
    mixed_precision: bool = True
    n_outlier: int = 32
    residual_window: int = 128
    def encode(self, k, v, ctx) -> CompressedKV: ...   # rotation -> Lloyd-Max -> pack
    def decode(self, c, ctx): ...                      # unpack -> inverse rotation
```

---

## 8. Runner integration and coexistence with weight quantization

### 8.1 Runner extension (minimal intrusion)

```python
# onecomp/runner.py (sketch of the diff)
def __init__(self, model_config, quantizer=None, ..., kv_config=None):
    ...
    self.kv_config = kv_config

def create_quantized_model(self, ...):
    model, tokenizer = ...   # existing: weights packed
    if self.kv_config is not None:
        # no monkeypatch; bake into the config (HF/vLLM read it to build the cache)
        model.config.kv_compression_config = self.kv_config.to_dict()  # dual-config
    return model, tokenizer
```

When a KV codec has `flag_calibration=True` (KVQuant/KVTuner/ThinKV), the `Runner`'s calibration
forward pass **piggybacks a K/V-distribution-collection hook** (completing in the same forward as
weight calibration) and calls `codec.calibrate()`.

### 8.2 Save schema (`config.json`)

```jsonc
{
  "quantization_config":   { "quant_method": "mixed_gptq", "quantization_bits": [...] },  // weight (existing)
  "kv_compression_config": {                                                              // KV (new)
    "codec":   { "name": "fp8", "skip_layers": [0,1,-1], "residual_window": 128 },
    "selector":{ "name": "snapkv", "keep_ratio": 0.3 },
    "policy":  "static",
    "per_layer_bits": [ ... ]
  }
}
```

- Because KV compression does not change the weights, KV-only runs need no re-packing (the config
  rides on the original or weight-quantized model).
- Calibration stats (scale/thresholds/head classes/segment budget) are shipped in
  `kv_calibration.safetensors`.
- HF/vLLM read `kv_compression_config` to build the cache / attention backend.

---

## 9. Mapping ThinKV (the flagship recipe, `recipe/thinkv.py`)

ThinKV is expressed precisely as a **composition of policy × codec × selector** in this
abstraction (which is itself evidence that the separation is well-founded). The structure of the
paper (`onecomp-ref/ThinKV`):

- **Thought classification (policy)**: the attention sparsity \(u\) (per row, count entries below
  1% of the row max as inactive) is averaged over a layer subset \(\mathcal{L}^*\) and classified
  into R/E/T by thresholds \(\Theta=\{\theta_1,\theta_2\}\), refreshed every \(\tau=128\) steps.
- **TBQ (codec)**: per-thought bits (default R4E4T2 = NVFP4/Ternary), group size 16,
  K=per-channel / V=per-token.
- **TBE (selector)**: on a transition, shrink the preceding segment by the retention schedule
  \(\{64,32,16,8,4\}\); when the budget \(k\) is exceeded, shrink the oldest, lowest-importance
  segment by K-means (on post-RoPE keys).
- **Continuous Thinking**: a PagedAttention extension (soft eviction + in-place reuse).
- **Calibration**: collect sparsity over 100 s1K prompts → determine \(\mathcal{L}^*, \Theta\)
  via KDE (reusing `onecomp/calibration`).

```python
@register_kv_policy("thinkv")
@dataclass
class ThinKVPolicy(KVPolicy):
    refresh_interval: int = 128
    retention_schedule: tuple = (64, 32, 16, 8, 4)
    token_budget: int = 1024
    precision: dict = field(default_factory=lambda: {"R": 4, "E": 4, "T": 2})
    calib_layers: list[int] | None = None     # L* (set by calibrate)
    thresholds: tuple | None = None           # Θ (set by calibrate)
    def calibrate(self, sparsity_stats, ctx): ...   # KDE -> L*, Θ
    def classify(self, sparsity_mean) -> str: ...    # -> "R"/"E"/"T"
    def plan(self, req, sys) -> CompressionPlan: ...  # per-thought NVFP4/Ternary + segment evict
```

---

## 10. Reusing the ILP assets (KVTuner / PM-KVQ)

The bit-budget allocation in KVTuner (layer/head bits), PM-KVQ (block bits), and ThinKV (thought
bits) is **the same integer-program form** as AutoBit's weight bit allocation ILP
(`onecomp/quantizer/autobit/ilp.py`, `ortools`).

Proposal (two stages to reduce risk):
1. First reuse the AutoBit ILP pattern in `onecomp/kvcomp/budget/ilp.py`.
2. Later extract the ILP core into `onecomp/utils/bit_allocation.py` and share it between weight
   and KV. `AutoBit._convert_raw_to_effective_target` (effective-bpw conversion) is also reusable
   for the KV memory budget.

→ This makes OneComp **the only package that optimizes weight and KV bit budgets uniformly**.

---

## 11. vLLM integration (two coexisting subsystems)

| | weight (existing) | KV (new) |
|---|---|---|
| entry point | `vllm.general_plugins` (`pyproject.toml`) | `vllm.platform_plugins` (TurboQuant style) |
| mechanism | `register_quantization_config` → `get_quant_method` → Linear swap | `Platform.get_attn_backend_cls` → `AttentionBackend`/`AttentionImpl` |
| touches | `LinearBase` | `Attention.forward` / KV cache layout |
| combined use | — | **possible simultaneously** (separate subsystems; only a `kv_cache_dtype` conflict warns) |

```toml
# proposed pyproject.toml addition
[project.entry-points."vllm.platform_plugins"]
kvcomp = "vllm_plugins.kvcomp.platform:KVCompPlatform"
```

Direction: **measure correctly in HF first, then vLLM** (as the existing weight path also goes
HF→vLLM). Simple FP8 leverages vLLM's native `kv_cache_dtype` as a baseline; INT4/eviction/ThinKV
are offered experimentally via an `AttentionBackend` swap.

---

## 12. Evaluation (`eval/`)

OneComp currently has only WikiText PPL + four lm-eval tasks. We import the LongBench / RULER /
Needle data and scorers shipped with KVCache-Factory
(`onecomp-ref/KVCache-Factory/{data,eval.py,metrics.py}`) into `onecomp/kvcomp/eval/`.

| Axis | Metric |
|---|---|
| Memory | peak GPU / KV mem / fragmentation |
| Speed | TTFT, ITL, tokens/s, batch capacity |
| Quality (long input) | LongBench, RULER, Needle, PPL |
| Quality (long output) | AIME, GPQA, MATH500, LiveCodeBench |
| Stability | multi-turn retention |

The eval API **always returns (memory, latency, quality) together** (never declares victory on
compression ratio alone).

---

## 13. Testing / benchmark / docs conventions

- **Tests**: create `BaseKVPressSpec` in `tests/onecomp/kvcomp/` (the analogue of `BaseQuantizeSpec`
  in `tests/.../quantizer/test_module.py`), inherited by each codec/selector. Regressions use
  `fixtures/` + JSON.
- **Benchmarks**: `benchmark/llama3-8b-kvcomp/` (Hydra YAML + the existing `quant_benchmark.py`
  conventions).
- **Validation**: `model_validation/kvcomp/validate_kvcomp.py` (end-to-end breakage detection).
- **Docs**: `docs/algorithms/kvcomp/*.md` + `docs/api/kvcomp/*.md` + two `mkdocs.yml` nav additions.

---

## 14. Roadmap

| Phase | Content | Acceptance criteria |
|---|---|---|
| v0.1 MVP | HF `CompressedCache`, FP8/INT4 codec, StreamingLLM/SnapKV selector, residual window, Runner integration, Needle/RULER/PPL, `BaseKVPressSpec` | FP8 matches BF16 PPL on TinyLlama; SnapKV reduces budget on Needle |
| v0.2 | KIVI, KVQuant (calib), Expected Attention, Pyramid/Razor, Triton int4 dequant | per-layer/head budget visualization |
| v0.3 | ThinK dim-prune, RocketKV 2-stage, vLLM experimental (platform + attention) | FP8 KV serving in vLLM |
| v0.4 | **ThinKV** (TBQ+TBE+CT), **KVTuner** (ILP reuse), **TurboQuant** (data-free codec) | ThinKV reproduces <5% KV mem, near-lossless, on reasoning |
| v0.5 | PM-KVQ progressive, service-aware policy, disaggregated serving | SLO-aware method switching |

---

## 15. Risks and pitfalls

1. Never declare victory on compression ratio alone (TurboQuant 3-bit degrades on reasoning /
   long-context; FP8 is a strong baseline).
2. Separate prefill-heavy (SnapKV) from decode-heavy / long-output (reasoning) via
   `RequestState.task_type`.
3. KV removed in multi-turn cannot be recovered → provide a RocketKV-MT-style retain mode in the
   selector.
4. Pure Python without kernels/layout will not be fast → separate Layer 1 (torch) from Layer 2
   (Triton).
5. transformers Cache API changes → keep `cache/` a thin adapter.
6. Avoid monkeypatching (KVCache-Factory's weakness) → registry + Cache subclass + local hooks.

---

## 16. Open questions (need discussion)

- Hook-conflict control when piggybacking KV codec calibration onto the same forward as weight
  calibration.
- How to auto-apply `kv_compression_config` through the HF `from_pretrained` path (custom code vs
  loader).
- The range of attention implementations supported by a vLLM AttentionBackend swap
  (FlashAttention / paged).
- Whether to extract the ILP core (`onecomp/utils/bit_allocation.py`) before or after v0.4.
