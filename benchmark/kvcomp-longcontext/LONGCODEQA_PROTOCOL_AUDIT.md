# LongCodeQA protocol audit

Audited on 2026-07-15 against:

- final-filtered dataset archive `Steefano/LCB` at
  `989d5eff750d65a72c522e47f8f745ef2e22906b`;
- reference code `Zteefano/long-code-bench` at
  [`91db30f5144363bb809534ec7e4a84a56ea43987`](https://github.com/Zteefano/long-code-bench/commit/91db30f5144363bb809534ec7e4a84a56ea43987);
- the dataset owner's clarification in
  [Hugging Face discussion #1](https://huggingface.co/datasets/Steefano/LCB/discussions/1);
- the released COLM 2025 paper/source.

## Release-faithful parts

The pinned archive is the post-filter release described by the dataset owner.
Its six bracket counts are exactly `113, 76, 92, 65, 47, 50`, totaling 443.
The earlier counts reported by the discussion author (`327, 170, 286, 185,
128, 145`) came from the accidentally uploaded pre-filter dataset and must not
be used.

Every row's `prompt` is passed verbatim to the tokenizer as raw causal text,
with no chat template, reconstruction, or truncation.  This is the self-hosted
HF input path in `src/long_code_bench/models/open.py`.  Discussion #1 confirms
that the released prompt beginning “You are going to be provided...” is the
prompt used for the paper's final analysis; the older appendix wording was an
oversight.  The current paper source has since been corrected to the released
wording.

The answer parser is copied from
[`src/long_code_bench/inference/codeqa_eval.py`](https://github.com/Zteefano/long-code-bench/blob/91db30f5144363bb809534ec7e4a84a56ea43987/src/long_code_bench/inference/codeqa_eval.py): first accept the first
case-insensitive `Final Answer: X` match, otherwise accept a single A--D letter
at the end of the stripped response.  Broader patterns such as “answer is X”
are intentionally not accepted.  Accuracy is an integer correct count divided
by the number of questions.  The paper reports this accuracy separately for
each length bracket; our cross-bracket micro accuracy is explicitly labelled a
derived summary, not an official primary metric.

The evaluator byte-pins the extracted JSON files:

| Bracket | Rows | SHA-256 |
|---|---:|---|
| 32K | 113 | `a4de950aa887c97ed74d64d50a1202b28e566105b9b94e62b7c1a50a48b46d98` |
| 64K | 76 | `c4c77b52582187753f94d4e3e78d68b6d2549791496b7305fc93b558166bdc74` |
| 128K | 92 | `c4b44c40f4a2efbeea6f69c210b80ca9a90d3156f032335fcb1182eb2992e1d8` |
| 256K | 65 | `3c28f1aa8bd6502c70b298e7f4390d5c9b7a0263b94c06141779ef1cb353aad9` |
| 512K | 47 | `1f4bd0339d68d11cae27af80abf6bb618ac5847ab41cb0a1b68c3d55a21c9fc2` |
| 1M | 50 | `eeaf9b1ac950d43341153e47a806c37b104572c9c1be5591700abecf94275680` |

The pinned zip itself has SHA-256
`0b31cf93a744d659e059056e6cd03bc7bb8d2601bad11e28d6b97dbfddf8d41a`.

## Decoding is a declared paired-comparison adaptation

The upstream release does not specify one reproducible generation protocol.
At the pinned commit:

- [`eval.py`](https://github.com/Zteefano/long-code-bench/blob/91db30f5144363bb809534ec7e4a84a56ea43987/eval.py) passes `max_output_length=None`;
- the [HF wrapper](https://github.com/Zteefano/long-code-bench/blob/91db30f5144363bb809534ec7e4a84a56ea43987/src/long_code_bench/models/open.py) forwards `max_new_tokens=None`, delegating length, sampling,
  and EOS behavior to the checkpoint/Transformers defaults;
- the Together wrapper independently hard-codes `max_tokens=512`;
- other API wrappers use provider defaults or different request settings;
- `LongCodeQAEvaluator.run()` stores generated output under `response`, while
  `_evaluate_predictions()` reads `pred["text"]`, so the pinned evaluator is
  not executable verbatim without correcting that key mismatch.

Consequently, a universal claim such as “official max_new_tokens=16” would be
false.  For a paired KV-cache experiment, all compared caches instead use the
same deterministic greedy decoder, a pre-registered 16-new-token cap, and stop
on either the cap or any EOS ID in the loaded model's pinned
`generation_config.json`.  The result JSON and context-fit manifest both state
`official_reference_decoding_exact: false`.  This controls randomness and
makes cache comparisons paired, but it does not reproduce the heterogeneous
decoding behind the original paper table.

Parsing only newly generated text also corrects the HF wrapper's behavior of
decoding prompt plus continuation.  No pinned prompt contains a matching
`Final Answer: [A-D]` or ends in a standalone A--D letter, so this correction
does not introduce a known prompt-side parser hit in the released 443 rows.

## Context and reporting limits

Prompts are never silently truncated.  Every scheduled row is tokenized with
the exact local tokenizer, and `input_tokens + 16` must fit the advertised
model context.  The formal feasible plan is therefore:

- Llama-3.1-8B-Instruct: complete 32K, 64K, and 128K release brackets
  (281 rows);
- Qwen3-8B: complete 32K release bracket (113 rows).

This is not a complete 443-row evaluation for either checkpoint, and Qwen3-8B
was not one of the checkpoints in the LongCodeBench paper's original table.
Outputs must be described as complete for the explicitly feasible bracket
plan, not as a reproduction of the paper's full LongCodeQA model results.
