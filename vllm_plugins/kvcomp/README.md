# KVComp vLLM backends

This directory contains the vLLM integration for the two independent proposal
methods, HayateQuant and TaperKV. Model paths and result paths are always
provided by the caller.

## HayateQuant

`hayate_config.py` defines the fixed average-rate recipes: K3/V1, K4/V2, and
K5/V3 bits per coordinate at rates 2, 3, and 4. `hayate_native.py` owns the
packed slot layout; `hayate_triton.py` contains the fused K/V store and packed
attention kernels. There is no secondary mode name or hybrid compatibility
recipe.

`mapping.py` accepts only `name`, `bits`, optional `skip_layers`, and an absent
calibration artifact. `residual_window` must be zero. Unsupported fields fail
before engine construction. The upstream TurboQuant dtype is only a registry
carrier: the cache-spec hook replaces the slot width, and the HayateQuant
backend replaces storage and attention.

Only serialized `skip_layers` are honored. OneComp disables vLLM's implicit
TurboQuant first/last-two-layer protection before engine configuration so an
empty list means all model layers for both HayateQuant and TurboQuant. The
benchmark fails unless its custom-backend evidence covers every model layer.

## TaperKV

`taper_config.py` validates independent key and value ladders.
`taper_backend.py` regrades request pages on GPU and applies the resulting
values before attention. It currently retains model-dtype physical pages, so
generation and semantic quality are valid but packed-memory savings are not.
Prefix caching is disabled because sharing mutable aged pages would violate
request isolation. Token TaperKV uses a fused quantize-and-scatter update and
a fixed-shape, sequence-length-aligned boundary regrade, so vLLM's CUDA-graph
and `torch.compile` paths remain enabled. Block TaperKV retains its eager
whole-page cadence. Cascade attention and decode context parallelism fail
closed.

TaperKV is standalone. A serialized TaperKV policy together with any codec or
selector is rejected rather than being reported as either proposal.

## Verification

Run CPU/config tests with the vLLM environment:

```bash
.venv-vllm/bin/python -m pytest -q tests/vllm_plugins/kvcomp
```

Run each GPU path in a fresh process and store all evidence outside Git:

```bash
.venv-vllm/bin/python benchmark/kvcomp-codec/vllm_benchmark.py \
  --model /path/to/local/model --method hayatequant --rate 3 \
  --output /path/to/results/hayatequant-r3.json

.venv-vllm/bin/python benchmark/kvcomp-codec/vllm_benchmark.py \
  --model /path/to/local/model --method taperkv \
  --output /path/to/results/taperkv.json
```

The driver requires backend-emitted runtime evidence for both custom paths.
HayateQuant evidence must identify the fused CUDA store, packed `uint8` cache,
packed decode, and every model layer. TaperKV results must continue to identify
their physical cache as model dtype until a variable-width page allocator is
implemented and measured.
