# Benchmark primary-source web audit (2026-07-16)

This repo-local record documents the current primary sources used to check the
formal benchmark protocol.  It contains no benchmark scores and does not
replace the immutable local dataset manifests or their file hashes.

## Sources and protocol decisions

* OpenAI `simple-evals`: <https://github.com/openai/simple-evals> and
  <https://github.com/openai/simple-evals/blob/main/math_eval.py>.  The project
  describes MATH-500 as the newer IID MATH set used for newer models and uses a
  zero-shot chain-of-thought evaluation.  This repository pins the released
  MATH-500 rows at revision
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, evaluates all 500 rows, and
  explicitly labels its deterministic greedy paired-compression protocol and
  Math-Verify 0.9.0 grader rather than claiming byte-identical API decoding.
* LongCodeBench: <https://github.com/Zteefano/long-code-bench>.  Its README says
  LongCodeQA uses the ordinary evaluation task and stores final accuracy
  directly, with no SWE-bench harness step.  This repository therefore reports
  the released LongCodeQA multiple-choice accuracy only (not a whole
  LongCodeBench or LongSWE-Bench score), pins release revision
  `989d5eff750d65a72c522e47f8f745ef2e22906b`, and records the parser reference
  commit `91db30f5144363bb809534ec7e4a84a56ea43987` plus the known upstream
  evaluator limitation.
* THUDM LongBench: <https://github.com/THUDM/LongBench>.  The official README
  identifies LongBench-E as uniformly sampled across 0--4K, 4--8K, and 8K+
  length groups, lists 13 `*_e` tasks, and directs users to task-specific
  official evaluation.  This repository pins revision
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, regrades all retained predictions
  with pinned official metric fixtures, and never substitutes a generic exact
  match for task-specific scorers.
* NVIDIA RULER: <https://github.com/NVIDIA/RULER>.  The official source exposes
  configurable synthetic task complexity, tokenizer/sequence-length/seed
  reproducibility, and task-specific automatic metrics.  The paper protocol
  here deliberately uses the separately labelled NeMo-Skills adapter with 100
  samples per task at RULER commit
  `38da79d79519ef87aa46ae804f838e1eab7f86d7` and NeMo-Skills commit
  `f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3`; it is not relabelled as the
  canonical NVIDIA 500-sample-per-task evaluation.
* EleutherAI lm-evaluation-harness:
  <https://github.com/EleutherAI/lm-evaluation-harness>.  The project documents
  a configurable standard-task harness and continues to evolve, so this repo
  freezes the compatible multiple-choice definitions at commit
  `27988a293647d5853e48edea291640b4af54740c`, records metadata schema `1.0`,
  exact dataset revisions, selection intervals, prompt hashes, and per-example
  predictions instead of following an unpinned moving `main` branch.

## Audit conclusion

The current benchmark names and metrics are appropriately bounded provided the
formal fail-closed aggregator completes: MATH-500 accuracy, LongCodeQA
micro-accuracy by bracket, official task-specific LongBench-E scores, explicitly
non-canonical 100-sample RULER-v1 adapter accuracy, and pinned lm-eval-compatible
zero-shot task metrics.  None may be promoted from a partial shard, smoke run,
or development candidate result.

Access date: 2026-07-16 (Asia/Tokyo).  Abstract, introduction, and paper result
values were not edited during this audit.
