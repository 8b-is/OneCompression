# TODO: HQ/TQ codec refactor and QA

## Scope

- Repo: `/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab`, branch `lab/kvcomp`.
- Target: HQ/HayateQuant and TurboQuant-facing benchmark/config paths.
- Goal: keep public codec imports stable while separating named HQ operating points from codec internals.

## Plan

- [x] Read current HQ/TQ implementation and benchmark call sites.
- [x] Avoid risky file moves for public modules such as `turboquant.py` and `hayatequant.py`.
- [x] Add a dedicated HayateQuant recipe module for HQ-frontier.
- [x] Replace duplicated K/V frontier allocation logic in benchmark runners.
- [x] Add unit coverage for the canonical frontier recipe.
- [x] Run py_compile and focused pytest.
- [x] Commit and push to `lab/kvcomp`.

## Review Notes

- `onecomp.kvcomp.codec.hayate.recipes` now owns the canonical HQ-frontier definition:
  `K(r+1)V(r-1)` plus key scale mode `ub`.
- Existing public imports remain valid:
  `onecomp.kvcomp.codec.hayatequant.HayateQuantCodec`,
  `onecomp.kvcomp.codec.hayate.HayateQuantCodec`,
  and `onecomp.kvcomp.codec.turboquant.TurboQuantCodec`.
- Large TurboQuant file splitting was intentionally deferred because current tests and downstream imports rely on that module name. The new recipe boundary gives future HQ variants a safer place to evolve first.
- Verification passed:
  `python3 -m py_compile ...` and
  `pytest -q tests/onecomp/kvcomp/test_kvcomp.py -k "turboquant or hayatequant_frontier_recipe or (hayatequant and not vllm and not pair_weighted and not layer_rotation and not attention_weighted)"`
  with `67 passed, 6 skipped`.
