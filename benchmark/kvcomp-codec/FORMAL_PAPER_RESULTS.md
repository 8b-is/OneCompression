# Formal HayateQuant paper result boundary

`render_hayatequant_formal_results.py` is the only supported bridge from the
coverage-checked benchmark aggregate to LaTeX. It does not search result
directories, read historical paper constants, or edit a paper source file.

## Generate the includes

First produce the complete aggregate with
`aggregate_hayatequant_paper_shards.py --only all`. Then run:

```bash
PY=${PY:-${REPO_ROOT:-.}/.venv/bin/python}
"$PY" benchmark/kvcomp-codec/render_hayatequant_formal_results.py \
  --aggregate /absolute/path/coverage_checked_aggregates.json \
  --output-dir /absolute/path/generated
```

This writes:

- `hayatequant_formal_macros.tex`, containing provenance, primary estimates,
  confidence bounds, and CI-gated claim text;
- `hayatequant_formal_tables.tex`, containing the complete formal PPL,
  corrected zero-shot, mechanism-control, the NeMo-Skills RULER-v1
  100-sample/task protocol (not canonical NVIDIA 500/task), LongBench-E, MATH-500,
  LongCodeQA code-comprehension, and fair-speed matrices.

The table include requires the LaTeX `booktabs` package. Inclusion in a paper
is an explicit author action; the renderer intentionally never modifies
`main.tex` and never writes placeholder rows.

## Fail-closed contract

Rendering is refused before either output is written unless all eight formal
axes are present and `formal_matrix_complete` is exactly `true`. Every axis
must carry a nonempty content-hashed source ledger and the same full 40-hex
commit. The aggregate additionally requires one paper-input manifest across
all axes, consistent model and dataset-file provenance, and one normalized
Python/package/CUDA/B200/semantic-`ONECOMP_*` stratum across all quality rows
(GPU UUID, host, and Slurm job identity are excluded nuisance identities).
Partial `--only` aggregates are explicitly incomplete. The renderer accepts
only this complete producer schema.

The frozen five-replicate campaign contains 1206 unique result paths: 1139
single-GPU quality shards, seven dependency-bound CPU PPL merges, and 60
fresh-process timing jobs, after an exact eight-job smoke gate. For each
proposal/baseline pair the matched TQ and FibQuant quality ledger has 164 cells;
the unmatched BlockQuant descriptive ledger has 80 cells per proposal. Each
proposal's TQ speed point ledger has 60 cells. The simultaneous timing/memory
gate has 264 endpoints across all four proposals.

The renderer also recomputes paper-facing statistics from the aggregate's
sufficient statistics. This catches a manually copied or stale summary even
when its surrounding JSON is otherwise well formed:

- the `TurboQuant-paper` internal key must carry the exact protocol scope
  **TurboQuant Algorithm 1 (MSE)** and
  `paper_table1_reproduction=false`; generated captions and macros preserve
  that distinction because the paper's mixed-precision LongBench recipe is
  not fully specified;

- PPL and mechanism ratios are rederived from NLL numerator/denominator data;
- corrected zero-shot task means and paired intervals are rederived from exact
  counts and discordant paired outcomes;
- every NeMo-Skills RULER-v1 100/task score is regraded over the complete pinned 13-task x 100
  sample/task NeMo-Skills adapter matrix; manifests and aggregates explicitly
  record that this is not the canonical NVIDIA 500-sample/task run;
- every LongBench-E prediction is regraded with its official task scorer
  against the corresponding row of the pinned `data.zip`; stored scores are
  cross-checks only;
- every MATH-500 generation is regraded with Math-Verify 0.9.0 against the
  corresponding row of the pinned dataset; stored `correct` fields are
  cross-checks only;
- LongCodeQA bracket/micro values are rederived from complete count records;
  every stored generation is reparsed, retained gold/repository/question fields
  are matched to the indexed content-hashed source row, and every input-token
  count is matched to the content-hashed checkpoint-specific context-fit
  manifest. The very large released prompts are not duplicated in result
  shards, so no comparison against a stored prompt copy is claimed;
- this is only LongCodeBench's LongCodeQA code-comprehension subtask. It is not
  a whole-LongCodeBench score and excludes LongSWE-Bench;
- timing medians and every process-level paired log-ratio interval are
  rederived within one fixed hardware, software, and protocol stratum. Public
  prefill/decode encode, prebuilt-decode, and explicit-roundtrip operations;
  research fused prefill/decode roundtrip; E2E TTFT/steady decode; and paired
  allocated-memory ratio/difference remain separate categories.

Formal timing additionally requires one explicit idle `batch-8gpu` node. Each
fresh process reserves all eight B200 GRES, 160 CPUs, and 2400G, uses
`exclusive=user`, and is chained `afterok`; quality jobs exclude that node. The
worker records the full Slurm allocation and then narrows
`CUDA_VISIBLE_DEVICES` to its first assigned B200 before importing torch. Both
the aggregator and renderer reject a changed allocation, multiple visible
measurement GPUs, or mixed physical nodes.

BlockQuant v1 remains in every applicable quality table at its versioned
identity-bit-counted source operating point. That payload is not matched to HQ,
so its 80 cells per proposal are a separate descriptive ledger with no pass
decision and are excluded from the 90% dominance denominator. Only matched
TurboQuant/FibQuant quality scopes and the TurboQuant speed scope enter that
gate; the renderer rejects relabeling or injecting a BlockQuant cell into it.

This live policy supersedes only the obsolete BlockQuant-denominator field in
the immutable `external_codec_math_audit_20260716` bundle; the audit's
mathematical and accounting evidence remains unchanged. The signed,
content-hashed repository attestation is
`papers/hayatequant_paper/experiments/results/development/audits/`
`external_codec_math_audit_20260716_policy_supersession/`. Integer rate labels
2/3/4 denote code-payload slots. Exact logical stored rates additionally count
fp16 scales and required identity fields: at head dimension 128, matched
TQ/FibQuant/HQ rows at slot `r` store `r + 0.125` bpc, whereas BlockQuant's
slot-3 exact rate depends on encode-update length.

The public payload microbenchmark is deliberately not described as packed
serving. Its codecs materialize code indices as unpacked `int64` tensors;
`stored_bits` counts only fixed-width encoded-KV data-plane fields (logical
code widths, signed-fp16 scales, and enabled mask/QJL/outlier fields).
Self-describing control tags, Python/container and allocator bytes, and global
codebook/LUT/rotation state are uncharged out-of-band/cache-level
configuration. It is therefore not a physical tensor-memory or packed-serving
measurement. Raw jobs also record the exact bytes of the public code-index and
scale tensors; that separate number still excludes container/control/global
state. E2E peak allocated memory is separately scoped to the resident model and
dequantized research cache and excludes reserved-but-unallocated CUDA memory;
it is not called packed-cache memory. Raw jobs record and the aggregator
revalidates these representations before the renderer emits speed/memory
tables. Formal data-free codebooks, LUTs, and
rotations are determined by the pinned source commit, resolved codec config,
fixed seed, and immutable codebook artifact manifest. The renderer records the
validated manifest SHA-256; resident global-state bytes remain outside the
per-token data-plane measurement.

Fields whose names indicate fallback, placeholder, hardcoded, or legacy values
are rejected. Missing models, rates, contexts, tasks, feasible brackets,
replicates, paired comparisons, or audit coverage are also rejected.

## Claim policy

The generated claim macros separate primary point-estimate direction from the
simultaneous-interval robustness supplement:

- PPL direction and checkpoint win/tie/loss counts use the predeclared geometric
  mean of within-checkpoint proposal/baseline point-estimate ratios. A separate
  robustness macro says that the direction is established only when the
  simultaneous 95% upper ratio bound is below one.
- ZS-Mean direction and checkpoint win/tie/loss counts use the signed paired
  point-estimate delta. A separate robustness macro requires the simultaneous
  95% lower bound to exceed zero.
- If an interval touches, crosses, or lies on the adverse side of parity, it is
  never rewritten as an improvement: the point-estimate direction stays
  visible, while interval-based directional robustness is reported as not
  established.
- Per-endpoint speed and allocated-memory ratios retain descriptive pointwise
  nominal intervals; every per-row speed claim macro remains fixed to
  `descriptive ratio only; no directional speed claim`.
- The only admissible complete speed/measured-memory statement is a separate
  predeclared primary-mode gate over all 264 stochastic endpoints (four proposals,
  two models, three rates, ten timing endpoints plus one allocated-memory endpoint). It uses
  one-sided 95% Bonferroni Student-t lower bounds over five fresh processes,
  additionally requires public unpacked payload tensor bytes to be no larger
  and logical `stored_bits` to remain equal, and fails closed if any axis is
  absent. This scope explicitly excludes downstream quality and packed serving.

`\newcommand` is used deliberately. If a paper still defines one of these
formal macros by hand, LaTeX fails instead of silently choosing a stale value.

## Reproducibility and tests

The includes embed only the aggregate SHA-256 and source commit, not a host
path or generation time. Identical aggregate bytes therefore produce
identical include bytes.

Run the focused boundary tests with:

```bash
"$PY" -m pytest -q tests/onecomp/kvcomp/test_hayatequant_formal_renderer.py
```

The tests cover deterministic output, all formal axes, strict CI gates,
incomplete/mixed-commit rejection, legacy-field rejection, stale sufficient
statistics, corrected zero-shot metric selection, and missing nested coverage.
