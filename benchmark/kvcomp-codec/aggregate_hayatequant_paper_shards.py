#!/usr/bin/env python3
"""Coverage-checked aggregation for HayateQuant paper row shards.

The formal launcher splits corrected zero-shot tasks and LongCodeQA brackets
into deterministic intervals.  This command refuses to average already-rounded
shard accuracies.  Instead it proves that intervals form exactly ``[0, N)``
with no overlap or gap, sums integer correct counts, and recomputes every
reported task/bracket and macro/micro score from those counts.

It is intentionally an offline result validator: no model or network client is
used.  Pinned dataset readers and metric packages are imported lazily only to
regrade exact source rows.  Source result files are content-hashed into the
atomic aggregate so the paper values remain traceable to immutable shards.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
import re
import statistics
import string
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from onecomp.kvcomp.eval.paper_job_receipts import (
    ReceiptValidationError,
    validate_successful_result_receipts,
)

DEFAULT_RUN_ROOT = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/" "hayate_iclr2027_refresh_20260715"
)
REPO_ROOT = Path(__file__).resolve().parents[2]

PAPER_MODELS = (
    "Llama-2-7b-hf",
    "llama-3.1-8b-instruct",
    "Qwen3-4B",
    "Qwen3-8B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B",
)
MODEL_MANIFEST_KEYS = {
    "Llama-2-7b-hf": "llama2-7b",
    "llama-3.1-8b-instruct": "llama31-8b",
    "Qwen3-4B": "qwen3-4b",
    "Qwen3-8B": "qwen3-8b",
    "DeepSeek-R1-Distill-Llama-8B": "r1-llama-8b",
    "DeepSeek-R1-Distill-Qwen-7B": "r1-qwen-7b",
    "DeepSeek-R1-Distill-Qwen-1.5B": "r1-qwen-1.5b",
}
# Every checkpoint in the formal matrix resolves to 128 dimensions per
# attention head (either an explicit ``head_dim`` or hidden_size / heads).
# Keeping this model contract explicit makes role-wise ``auto`` codec choices
# independently re-checkable without importing model code during aggregation.
PAPER_MODEL_HEAD_DIMS = {model: 128 for model in PAPER_MODELS}
PAPER_MODEL_EVALUATION_ROLES = {
    model: (
        "development_and_recipe_selection"
        if model == "llama-3.1-8b-instruct"
        else "additional_development_evaluation_independence_not_claimed"
    )
    for model in PAPER_MODELS
}
ZERO_TASKS = {
    "arc_easy": {
        "size": 2376,
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "primary": "acc_norm",
        "manifest_key": "arc_easy_test",
        "repo": "allenai/ai2_arc",
        "config": "ARC-Easy",
    },
    "arc_challenge": {
        "size": 1172,
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "primary": "acc_norm",
        "manifest_key": "arc_challenge_test",
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
    },
    "piqa": {
        "size": 1838,
        "split": "validation",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "primary": "acc_norm",
        "manifest_key": "piqa_validation",
        "repo": "baber/piqa",
        "config": None,
    },
    "hellaswag": {
        "size": 10042,
        "split": "validation",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "primary": "acc_norm",
        "manifest_key": "hellaswag_validation",
        "repo": "Rowan/hellaswag",
        "config": None,
    },
    "winogrande": {
        "size": 1267,
        "split": "validation",
        "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        "primary": "acc",
        "manifest_key": "winogrande_xl_validation",
        "repo": "allenai/winogrande",
        "config": "winogrande_xl",
    },
}
ZERO_RATE_METHODS = {
    "full": ("FullKV",),
    "2": ("TurboQuant-paper", "FibQuant-paper", "HQ", "AdaptiveHQ"),
    "3": (
        "TurboQuant-paper",
        "FibQuant-paper",
        "BlockQuant-paper",
        "HQ",
        "AdaptiveHQ",
    ),
    "4": ("TurboQuant-paper", "FibQuant-paper", "HQ", "AdaptiveHQ"),
}

LONG_METHODS = (
    "FullKV",
    "TurboQuant-paper",
    "FibQuant-paper",
    "BlockQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
PROPOSAL_METHODS = ("HQ", "AdaptiveHQ")
QUALITY_BASELINE_METHODS = ("TurboQuant-paper", "FibQuant-paper", "BlockQuant-paper")
POINT_ESTIMATE_WIN_THRESHOLD = 0.90
ADAPTIVE_HQ_POLICY_VERSION = "adaptive_hq_cache_causal_fixed_incumbent_hysteresis_v6"
ADAPTIVE_HQ_DISTORTION_TABLE_VERSION = (
    "source_qmc128_exact_codebook_fp16scale_safe_two_pair_update_carrier_v4"
)
ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION = "value_scale_fp16_fraction_lsb_v1"
ADAPTIVE_HQ_OBJECTIVE = "causal_raw_power_times_qmc_base_plus_one_carrier_per_update"
ADAPTIVE_HQ_CARRIER_AMORTIZATION = "signed_qmc_all_scales_delta_divided_by_current_update_vectors"
ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION = 1.0 / math.sqrt(128)
ADAPTIVE_HQ_SWITCH_MARGIN_BASIS = "one_over_sqrt_fixed_qmc_vectors_128"
ADAPTIVE_HQ_HYSTERESIS = (
    "enter_challenger_only_beyond_margin_retain_while_strictly_better_"
    "fallback_to_fixed_incumbent_on_tie_or_loss"
)
TURBOQUANT_FAKE_QUANT_IMPLEMENTATION = "inherited unfused encode_then_decode"
HQ_FAKE_QUANT_IMPLEMENTATION = "HayateQuant fused research fake_quant_roundtrip"
ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION = (
    "inherited encode_then_decode fallback; AdaptiveHQ fused path unsupported"
)
HQ_CODEBOOK_BUNDLE = "fixed_hq_b2_v1"
ADAPTIVE_HQ_CODEBOOK_BUNDLE = "adaptive_hq_b2_safe_v1"
HQ_CODEBOOK_MANIFEST_SHA256 = "df02ba5fd41fd0cc3fd2d39474fcfe7a628c15eb9ea012e8c70b1ffb3dabecde"
ADAPTIVE_HQ_CANDIDATE_PAIRS = {
    rate: ((2 * rate, 2 * rate), (2 * rate + 1, 2 * rate - 1)) for rate in (2, 3, 4)
}
HAYATE_FORMAL_PROVENANCE = {
    "HQ": {
        "method_role": "paper_proposal_fixed",
        "external_quality_baseline": False,
        "proposal_method": True,
        "proposal_family": "HayateQuant",
        "proposal_variant": "fixed_global",
        "paper_comparable": True,
        "speed_comparison": True,
        "formal_rate_slots": [2, 3, 4],
        "allocation_scope": "globally_fixed_kv_code_bits",
    },
    "AdaptiveHQ": {
        "method_role": "paper_proposal_adaptive",
        "external_quality_baseline": False,
        "proposal_method": True,
        "proposal_family": "HayateQuant",
        "proposal_variant": "causal_adaptive",
        "paper_comparable": True,
        "speed_comparison": True,
        "formal_rate_slots": [2, 3, 4],
        "allocation_scope": "cache_local_layer_local_causal_online_stats",
    },
}
# ``TurboQuant-paper`` is retained as the stable internal row key.  The raw
# result validators below require its finite-Beta, dense-Haar, L2-normalized,
# symmetric Algorithm-1 recipe.  TurboQuant Section 4.3 does not disclose
# enough of the mixed-precision LongBench recipe to identify that row as a
# reproduction of the paper table (and its printed 32*3 + 96*2 arithmetic is
# 2.25 payload bits, not 2.5).  Carry this scientific scope into the aggregate
# so a renderer cannot silently promote the baseline to a stronger claim.
TURBOQUANT_METHOD_SCOPE = {
    "internal_method_key": "TurboQuant-paper",
    "display_name": "TurboQuant Algorithm 1 (MSE)",
    "implementation_scope": "finite_beta_lloyd_max_dense_haar_l2_scalar_mse",
    "metadata_precision": "signed_fp16_l2_norm_counted_in_stored_bits",
    "paper_table1_reproduction": False,
    "scope_note": (
        "Algorithm 1 MSE protocol baseline only; not a complete reproduction "
        "of the paper's under-specified mixed-precision LongBench/Table 1 recipe"
    ),
    "primary_source": "https://arxiv.org/abs/2504.19874",
}
FIBQUANT_METHOD_SCOPE = {
    "internal_method_key": "FibQuant-paper",
    "display_name": "FibQuant v1 (exact quality baseline)",
    "implementation_scope": "exact_spherical_beta_block_vq_dense_haar_l2",
    "metadata_precision": "signed_fp16_l2_norm_counted_in_stored_bits",
    "paper_recipe_rates": [2, 3, 4],
    "paper_comparable": True,
    "paper_reproduction": False,
    "proposal_method": False,
    "speed_comparison": False,
    "scope_note": (
        "Independent arXiv v1 quality baseline; exact nearest-codeword research path "
        "with no packed serving-kernel speed claim. The paper recipe is comparable, "
        "but unpublished centroids/seeds prevent a paper-result reproduction claim."
    ),
    "primary_source": "https://arxiv.org/abs/2605.11478v1",
}
FIBQUANT_PAPER_RECIPES = {
    2: (4, 8, "097a7fe987ca4092b653480bec23f07ed1470694bd911e1544f67966bf5bb2e1"),
    3: (2, 6, "27f2806eaf4a4b7d7c05654fca7e4cb93dbcfad5a96b4820cccb3cdfbd1b8b75"),
    4: (2, 8, "3b74a174a348e46a6e82c931b65c9adbc3739c504a37549b7d7755b72a6bfdc2"),
}
FIBQUANT_D128_ROTATION_SHA256 = "499184023e9e9ff994bca12eca897f0ddce26af33afc913fb66d5f58b019c55e"
BLOCKQUANT_SOURCE_TEX_SHA256 = "dc87211edfb3e83285bc19b636d1f6ebe8b054f8c5b69ae0d5c2894b7dac721f"
BLOCKQUANT_SELECTION_COMPLETION = {
    "paper_statement": "per-attention-head 32 key channels with largest L2 norm",
    "paper_lifecycle_or_identity_coding_published": False,
    "mode": "versioned-reproducibility-completion",
    "completion": "per_batch_head_update_top_l2_v1",
    "energy_statistic": "sum_squares_over_current_encode_update_token_axis",
    "scope": "independent_per_batch_and_attention_head",
    "tie_break": "lowest_channel_index",
    "identity_coding": "combinadic_rank_fixed_width_v1",
    "identity_bits_counted": True,
}
BLOCKQUANT_METHOD_SCOPE = {
    "internal_method_key": "BlockQuant-paper",
    "display_name": "BlockQuant v1 (fixed-seed exact quality baseline)",
    "implementation_scope": (
        "exact_finite_sphere_block_vq_dynamic_top32_completion_p3_tail_completion"
    ),
    "metadata_precision": ("signed_fp16_scales_plus_combinadic_top32_identity_bits_counted"),
    "paper_recipe_rate_slots": [3],
    "paper_declared_key_bits_per_coordinate_conditional_on_identities": 3.5,
    "paper_declared_value_payload_bits_per_coordinate": 2,
    "paper_comparable": False,
    "paper_reproduction": False,
    "proposal_method": False,
    "external_quality_baseline": True,
    "speed_comparison": False,
    "scope_note": (
        "Independent BlockQuant v1 Algorithm-1 fixed-seed same-runtime quality row; "
        "not the paper five-random-seed aggregate. Unpublished LUT and top-32 lifecycle "
        "are not guessed: exact search is used, dynamic per-head/update L2 selection is "
        "versioned, and identity bits plus p=3 tails are counted explicitly."
    ),
    "primary_source": "https://arxiv.org/abs/2605.19972v1",
    "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
}
LONGCODE_PLANS = {
    "llama-3.1-8b-instruct": {
        "32K": {"size": 113, "max_context_tokens": 131072},
        "64K": {"size": 76, "max_context_tokens": 131072},
        "128K": {"size": 92, "max_context_tokens": 131072},
    },
    "Qwen3-8B": {
        "32K": {"size": 113, "max_context_tokens": 40960},
    },
}
LONGCODE_REVISION = "989d5eff750d65a72c522e47f8f745ef2e22906b"
LONGCODE_PROTOCOL_COMMIT = "91db30f5144363bb809534ec7e4a84a56ea43987"
LONGCODE_DISCUSSION_URL = "https://huggingface.co/datasets/Steefano/LCB/discussions/1"
LONGCODE_PROTOCOL_SCOPE = (
    "release_exact_dataset_prompt_parser_and_per_bracket_accuracy; "
    "preregistered_paired_greedy_decoding_adaptation"
)
LONGCODE_INPUT_PROTOCOL_SCOPE = (
    "release exact dataset, prompt, parser, and per-bracket accuracy; "
    "preregistered paired greedy decoding adaptation"
)
LONGCODE_SHA256 = {
    "32K": "a4de950aa887c97ed74d64d50a1202b28e566105b9b94e62b7c1a50a48b46d98",
    "64K": "c4c77b52582187753f94d4e3e78d68b6d2549791496b7305fc93b558166bdc74",
    "128K": "c4b44c40f4a2efbeea6f69c210b80ca9a90d3156f032335fcb1182eb2992e1d8",
}
LONGCODE_MODEL_EOS_TOKEN_IDS = {
    "llama-3.1-8b-instruct": [128001, 128008, 128009],
    "Qwen3-8B": [151645, 151643],
}
LONGCODE_ANSWER_PARSER = (
    "first /Final Answer:\\s*([ABCD])/i; otherwise trailing " "/\\b([ABCD])\\s*$/i"
)
_LONGCODE_FINAL_ANSWER = re.compile(r"Final Answer:\s*([ABCD])", re.IGNORECASE)
_LONGCODE_TRAILING_LETTER = re.compile(r"\b([ABCD])\s*$", re.IGNORECASE)
LONGCODE_SOURCE_ROW_AUDIT = {
    "prediction": "reparsed from every stored generated_text",
    "retained_source_fields_compared": [
        "correct_letter_as_gold",
        "repo",
        "question",
    ],
    "input_tokens_compared_to": (
        "every indexed row of the pinned tokenizer-specific context-fit manifest"
    ),
    "released_prompt_retained_in_result": False,
    "released_prompt_comparison": (
        "not claimed: result shards do not duplicate the released prompt; the "
        "source dataset and tokenizer-specific context-fit manifest are instead "
        "independently content-hashed and reopened"
    ),
}

LONG_MODELS = ("llama-3.1-8b-instruct", "Qwen3-8B")
RULER_LENGTHS = (4096, 8192, 16384)
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_ROWS_PER_TASK = 100
RULER_CANONICAL_NVIDIA_ROWS_PER_TASK = 500
RULER_PROTOCOL_LABEL = "NeMo-Skills RULER v1 100-sample/task protocol"
RULER_SAMPLING_PROTOCOL = (
    "pinned NeMo-Skills adapter: 100 samples/task; canonical "
    "NVIDIA/RULER v1 config: 500 samples/task"
)
RULER_RESULT_SCHEMA_VERSION = 3
RULER_SOURCE_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
NEMO_SKILLS_RULER_SOURCE_COMMIT = "f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3"
RULER_MAX_NEW = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}
RULER_PROMPT_DESCRIPTION = "released input + answer_prefix, verbatim, as sole generic user message"
RULER_SCORING_DESCRIPTION = (
    "case-insensitive substring; all-reference fraction for niah/vt/cwe/fwe, "
    "max-reference match for qa"
)
RULER_CONTEXT_SEMANTICS = (
    "named RULER sequence length is the total prompt-plus-generation limit; actual "
    "prompt + pinned upstream task max_new_tokens is checked against both the named limit "
    "and model capacity"
)
RULER_CAPACITY_CHECK = (
    "input_tokens + pinned upstream task max_new_tokens <= named RULER sequence length <= "
    "max_position_embeddings"
)
RULER_PROTOCOL_SCOPE = (
    "release_exact_tokenizer_specific_data_tasks_generation_budgets_and_scorer; "
    "paired_model_native_non_thinking_chat_wrapper_adaptation"
)
RULER_CHAT_TEMPLATE_KWARGS = {
    "enable_thinking": False,
    "unsupported_keyword_policy": "omit_only_enable_thinking",
}
RULER_GENERATION_DESCRIPTION = (
    "greedy; pinned upstream task-specific max_new_tokens; model-aware non-thinking "
    "chat-wrapper adaptation"
)
RULER_COMPLETE_EVALUATION_SCOPE = (
    "complete_nemo_skills_ruler_v1_100_sample_per_task_data_tasks_and_scorer_with_"
    "paired_decoder_adaptation"
)
LONGBENCH_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)
LONGBENCH_COUNTS = {
    "qasper": 224,
    "multifieldqa_en": 150,
    "hotpotqa": 300,
    "2wikimqa": 300,
    "gov_report": 300,
    "multi_news": 294,
    "trec": 300,
    "triviaqa": 300,
    "samsum": 300,
    "passage_count": 300,
    "passage_retrieval_en": 300,
    "lcc": 300,
    "repobench-p": 300,
}
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_PROTOCOL_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONGBENCH_PROTOCOL_SCOPE = (
    "release_exact_dataset_prompts_middle_truncation_task_generation_budgets_and_scorers; "
    "model_native_chat_template_adaptation_for_new_checkpoints"
)
LONGBENCH_CHAT_WRAPPER_SCOPE = (
    "Hayate model-native adaptation of official model-specific build_chat: "
    "tokenizer.apply_chat_template(add_generation_prompt=true; "
    "enable_thinking=false when supported)"
)
LONGBENCH_ARCHIVE_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
LONGBENCH_MAX_NEW = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}
LONGBENCH_PROMPT_BUDGETS = {
    "llama-3.1-8b-instruct": {task: 65536 for task in LONGBENCH_TASKS},
    # Qwen3-8B advertises 40,960 positions. These budgets reserve each task's
    # official generation allowance plus 16 tokens for chat-wrapper/seam
    # retokenization. The final wrapped capacity is independently checked.
    "Qwen3-8B": {task: 40960 - LONGBENCH_MAX_NEW[task] - 16 for task in LONGBENCH_TASKS},
}
LONGBENCH_MODEL_CONTEXT_LIMITS = {
    "llama-3.1-8b-instruct": 131072,
    "Qwen3-8B": 40960,
}
LONGBENCH_EXPECTED_TRUNCATIONS = {
    "llama-3.1-8b-instruct": {task: 0 for task in LONGBENCH_TASKS},
    "Qwen3-8B": {
        task: ({"multi_news": 1, "lcc": 5, "repobench-p": 1}.get(task, 0))
        for task in LONGBENCH_TASKS
    },
}
LONGBENCH_NO_CHAT_TASKS = {
    "trec",
    "triviaqa",
    "samsum",
    "lcc",
    "repobench-p",
}
LONGBENCH_DATASET_REPO = "THUDM/LongBench"
LONGBENCH_DATASET_FILENAME = "data.zip"
MATH500_DATASET_REPO = "HuggingFaceH4/MATH-500"
MATH500_DATASET_CONFIG = None
MATH500_DATASET_SPLIT = "test"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_VERIFY_TASK_SOURCE_COMMIT = "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b"
MATH500_QUERY_TEMPLATE = "Question: {problem}\nStep-by-Step Answer:"
MATH500_FEWSHOT_ORDER = [4, 3, 2, 1]
MATH500_FEWSHOT_CONTENT_SHA256 = "1360154eb365a22beb1a201dd6ff159e829b6d64d59f69df29530700034c07a4"
MATH500_FEWSHOT_RENDERING = (
    'join(query + " " + answer.strip(), separator="\\n\\n") then main query'
)
MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION = "0.13.0"
MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT = "7c1cd62716b0a198a630c26d781430c54726cd02"
MATH500_PROTOCOL_ID = (
    "math500_adapted_composite_math_verify_ba3d3aa_" "lighteval_prompt_manager_7c1cd62_v1"
)
MATH500_STOP_SEQUENCES = (
    "\nQuestion:",
    "\nProblem:",
    "\nquestion:",
    "\nproblem:",
)
MATH500_PROTOCOL_SCOPE = (
    "adapted_composite_math_verify_ba3d3aa_lighteval_prompt_manager_7c1cd62_"
    "four_shot_query_generation_"
    "prediction_extraction_and_verify; "
    "corrected_inline_math_delimiters_for_bare_gold_answer_fragments"
)
MATH500_GOLD_SERIALIZATION = "wrap_bare_answer_as_inline_math_$...$"
MATH500_GOLD_ADAPTATION_REASON = (
    "the pinned reference task passes bare dataset answer fragments to the LaTeX "
    "extractor; explicit inline-math delimiters make those gold fragments parseable"
)
WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT2_TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
PPL_MAX_LENGTH = 2048
PPL_STRIDE = 1024
PPL_NLL_ACCUMULATION = (
    "fp32_token_cross_entropy_per_window_fp64_sum_host_binary64_math_fsum"
)
PPL_WINDOW_CACHE_LIFETIME = "released_before_next_cache_factory"
PPL_CROSS_WINDOW_TENSOR_RETENTION = (
    "tokenized_input_ids_only; window_cache_logits_targets_losses_released; "
    "nll_host_binary64_scalars"
)
PPL_DENOMINATOR_CONTRACT_PATH = Path(__file__).with_name(
    "hayatequant_formal_ppl_denominators.json"
)
PER_EXAMPLE_CACHE_LIFETIME = "released_after_scalar_accounting_before_next_cache_factory"
PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION = (
    "none; retained_records_are_host_scalar_text_and_mappings_only"
)
ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME = (
    "released_after_protocol_dependency_snapshot_before_first_choice_factory"
)
QUALITY_DTYPE = "bfloat16"
QUALITY_DEVICE = "cuda:0"
LM_EVAL_PROTOCOL_COMMIT = "27988a293647d5853e48edea291640b4af54740c"
LM_EVAL_TASK_METADATA_VERSION = "1.0"
ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA = "lm_eval_multiple_choice_prompt_v1"
PPL_TEACHER_FORCED_PROTOCOL = "single_forward_causal_mask_token_local_cache"
CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL = "causal_prefix_prefill_then_token_step_targets"
BLOCKQUANT_TEMPORAL_DEPENDENCY = "dynamic_outlier_channels_over_update_tokens"
ADAPTIVE_HQ_TEMPORAL_DEPENDENCY = "cache_local_causal_bit_allocation_statistics"
PPL_COMPRESSION_ACCOUNTING = {
    "compressed_rows": "exact_cache_bit_counters",
    "ratio_definition": "sum_window_fp16_bits / sum_window_stored_bits",
    "window_ratios_averaged": False,
    "fullkv_rows": "fullkv_no_cache",
}
ZEROSHOT_COMPRESSION_ACCOUNTING = {
    "compressed_rows": "exact_cache_bit_counters",
    "ratio_definition": "sum_choice_and_example_fp16_bits / sum_choice_and_example_stored_bits",
    "shard_aggregation": "sum_exact_integer_bit_counters_before_division",
    "window_or_example_ratios_averaged": False,
    "fullkv_rows": "fullkv_no_cache",
}
SPEED_PRIMARY_COMPILE_MODE = "off"
PUBLIC_LOGICAL_PAYLOAD_SCOPE = {
    "api": "public KVCodec.encode/decode",
    "representation": "materialized_logical_payload",
    "index_tensor_dtype": "int64",
    "index_tensors_bit_packed": False,
    "stored_bits_scope": "fixed_width_per_token_data_plane_growth_only",
    "stored_bits_semantics": (
        "codes plus signed-fp16 scales and enabled mask/QJL/outlier data-plane fields"
    ),
    "excluded_from_stored_bits": (
        "self-describing control tags, Python/container bytes, allocator-resident tensor "
        "bytes, and global codebook/LUT/rotation state"
    ),
    "control_state_interpretation": (
        "out-of-band/cache-level configuration, not physical serving memory"
    ),
    "physical_memory_interpretation": (
        "not the actual tensor-memory footprint and not a packed serving-cache result"
    ),
    "self_describing_control_tags_counted": False,
    "python_container_bytes_counted": False,
    "persistent_codec_state_counted": False,
    "packed_serving_claim_allowed": False,
}

_SPEED_REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "triton",
    "cuda_runtime_reported_by_torch",
    "cudnn",
)
_SPEED_PROTOCOL_STRATUM_KEYS = (
    "dtype",
    "fused_backend",
    "batch_size",
    "micro_seq_len",
    "decode_seq_len",
    "quality_query_len",
    "micro_warmup",
    "micro_repeats",
    "micro_inner_loops",
    "bootstrap_resamples",
    "e2e_prompt_len",
    "e2e_gen_len",
    "e2e_warmup",
    "e2e_repeats",
    "attention_implementation",
    "sink_tokens",
    "recent_tokens",
)

SPEED_PAIRED_TIMING_METRICS = (
    "micro.prefill.payload_api.encode",
    "micro.prefill.payload_api.decode_prebuilt_payload",
    "micro.prefill.payload_api.roundtrip_explicit_encode_then_decode",
    "micro.prefill.research_fake_quant.fake_quant_roundtrip",
    "micro.decode.payload_api.encode",
    "micro.decode.payload_api.decode_prebuilt_payload",
    "micro.decode.payload_api.roundtrip_explicit_encode_then_decode",
    "micro.decode.research_fake_quant.fake_quant_roundtrip",
    "e2e.ttft",
    "e2e.decode_steady_latency",
)
SPEED_PRIMARY_MEMORY_METRIC = "e2e.peak_allocated_memory"
SPEED_ALLOCATION_CONTRACT = {
    "partition": "batch-8gpu",
    "allocated_b200_gpus": 8,
    "cpus_per_task": 160,
    "job_cpus_per_node": "160",
    "memory_per_node_mib": 2457600,
    "measured_visible_gpus": 1,
    "other_users_excluded": True,
    "speed_jobs_afterok_serialized": True,
    "quality_jobs_exclude_speed_node": True,
}
SPEED_PEAK_ALLOCATED_SCOPE = {
    "counter": "torch.cuda.max_memory_allocated",
    "unit": "MiB",
    "reset": "once immediately before each timed prefill/decode generation",
    "includes": "resident model plus live tensors and research fake-quant cache",
    "incremental_definition": "peak_allocated_mib minus synchronized starting_allocated_mib",
    "excludes": "CUDA reserved-but-unallocated bytes",
    "packed_kv_cache_interpretation_allowed": False,
}
PUBLIC_PAYLOAD_TENSOR_SCOPE = (
    "exact storage of code-index and scale tensors in the public payload; "
    "excludes Python containers, control tags, allocator fragmentation, and "
    "persistent codec/codebook/LUT/rotation state"
)

# Two-sided Student-t 0.975 quantiles. Formal speed uses five independent
# process replicates (df=4). For larger ad-hoc sets, df=30 is a conservative
# cap rather than silently substituting the narrower normal interval.
_T975 = (
    None,
    12.7062047364,
    4.3026527299,
    3.1824463053,
    2.7764451052,
    2.5705818356,
    2.4469118488,
    2.3646242510,
    2.3060041352,
    2.2621571629,
    2.2281388520,
    2.2009851601,
    2.1788128297,
    2.1603686565,
    2.1447866879,
    2.1314495456,
    2.1199052992,
    2.1098155778,
    2.1009220402,
    2.0930240544,
    2.0859634473,
    2.0796138447,
    2.0738730679,
    2.0686576104,
    2.0638985616,
    2.0595385528,
    2.0555294386,
    2.0518305165,
    2.0484071418,
    2.0452296421,
    2.0422724563,
)


class AggregateError(RuntimeError):
    """A shard set is malformed, inconsistent, overlapping, or incomplete."""


FORMAL_AXIS_KEYS = (
    "ppl",
    "mechanism",
    "zeroshot",
    "ruler",
    "longbench",
    "math500",
    "longcodeqa",
    "speed",
)
QUALITY_AXIS_KEYS = tuple(axis for axis in FORMAL_AXIS_KEYS if axis != "speed")
QUALITY_SOFTWARE_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "lm-eval",
    "triton",
    "math-verify",
    "safetensors",
    "rouge",
    "fuzzywuzzy",
    "fuzzywuzzy-backend",
)
QUALITY_SEMANTIC_ENVIRONMENT = {
    "ONECOMP_TRITON_CODEC": "0",
    "ONECOMP_TRITON_PREFILL": "0",
    "ONECOMP_COMPILE_CODEC": "0",
    "ONECOMP_COMPILE_DECODE_ONLY": "0",
}
FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE = (
    "full_sha256_submitter_binding_plus_worker_semantic_validator_check"
)
FORMAL_SMOKE_PLAN_VERSION = 4
FORMAL_SMOKE_JOB_KEYS = (
    "ppl",
    "zeroshot",
    "ruler",
    "longbench",
    "math500",
    "longcode",
    "speed",
    "native-hybrid",
)


def _load_object(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise AggregateError(f"Non-standard JSON numeric constant {value!r} in {path}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AggregateError(f"Duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateError(f"Cannot read result JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AggregateError(f"Result root must be a JSON object: {path}")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _expected_hayate_codebook_artifacts(bundle: str) -> dict[str, Any]:
    """Resolve and hash the immutable formal HQ codebook bundle.

    Formal results carry absolute source paths for diagnostics, but scientific
    identity comes from the pinned manifest and artifact hashes.  Rebuilding
    the exact codec-owned record here makes a missing, replaced, or
    development-only codebook fail before any paper value is aggregated.
    """

    if bundle not in {HQ_CODEBOOK_BUNDLE, ADAPTIVE_HQ_CODEBOOK_BUNDLE}:
        raise AggregateError(f"Unexpected formal HayateQuant codebook bundle {bundle!r}")
    codebook_dir = REPO_ROOT / "onecomp/kvcomp/codec/hayate/codebooks"
    manifest_path = (codebook_dir / "manifest.json").resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != HQ_CODEBOOK_MANIFEST_SHA256:
        raise AggregateError("Formal HayateQuant codebook manifest is missing or changed")
    manifest = _load_object(manifest_path)
    if manifest.get("schema") != "hayatequant.shipped_gaussian_codebooks.v1":
        raise AggregateError("Formal HayateQuant codebook manifest schema changed")
    required = manifest.get("required_bundles")
    artifacts_by_name = manifest.get("artifacts")
    filenames = required.get(bundle) if isinstance(required, Mapping) else None
    if (
        not isinstance(filenames, list)
        or not filenames
        or not isinstance(artifacts_by_name, Mapping)
        or len(set(filenames)) != len(filenames)
    ):
        raise AggregateError(f"Malformed formal HayateQuant codebook bundle {bundle!r}")
    artifacts = []
    for filename in filenames:
        spec = artifacts_by_name.get(filename)
        if not isinstance(filename, str) or not isinstance(spec, Mapping):
            raise AggregateError(f"Malformed artifact entry in codebook bundle {bundle!r}")
        path = (codebook_dir / filename).resolve()
        file_sha256 = spec.get("file_sha256")
        if (
            path.parent != codebook_dir.resolve()
            or not path.is_file()
            or not isinstance(file_sha256, str)
            or _sha256(path) != file_sha256
        ):
            raise AggregateError(f"Formal HayateQuant artifact is missing or changed: {filename}")
        block_size = spec.get("block_size")
        k = spec.get("k")
        tensor_sha256 = spec.get("tensor_sha256")
        origin = spec.get("origin")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size < 1
            or isinstance(k, bool)
            or not isinstance(k, int)
            or k < 2
            or not _sha256_string(tensor_sha256)
            or not isinstance(origin, str)
            or not origin
        ):
            raise AggregateError(f"Malformed formal HayateQuant artifact metadata: {filename}")
        artifacts.append(
            {
                "source": "shipped_manifest",
                "path": str(path),
                "block_size": block_size,
                "k": k,
                "file_sha256": file_sha256,
                "tensor_sha256": tensor_sha256,
                "origin": origin,
                "manifest_pinned": True,
                "manifest_path": str(manifest_path),
                "manifest_sha256": HQ_CODEBOOK_MANIFEST_SHA256,
            }
        )
    return {
        "required": True,
        "bundle": bundle,
        "manifest_path": str(manifest_path),
        "manifest_sha256": HQ_CODEBOOK_MANIFEST_SHA256,
        "development_override_dir": None,
        "artifacts": artifacts,
    }


def _validated_declared_file(value: Any, *, label: str) -> Path:
    """Validate one immutable file record and return its resolved path."""
    if not isinstance(value, dict):
        raise AggregateError(f"{label} must be a file provenance object")
    raw_path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
        or not _sha256_string(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise AggregateError(f"Malformed immutable file provenance for {label}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise AggregateError(f"Pinned file is missing for {label}: {path}")
    if path.stat().st_size != size:
        raise AggregateError(f"Pinned file byte size changed for {label}: {path}")
    actual = _sha256(path)
    if actual.lower() != str(digest).lower():
        raise AggregateError(f"Pinned file SHA-256 changed for {label}: {path}")
    return path


def _validated_hf_input_provenance(document: dict, *, result_path: Path) -> dict:
    """Load the exact HF dataset records committed to a result preflight.

    The immutable worker envelope fingerprints the run-local paper input
    manifest.  Re-hashing that manifest here, and then re-hashing its external
    pinned-HF manifest and data files, closes the gap between a declared
    revision string and the rows that are actually regraded.
    """
    identity = _require_run_provenance(document, path=result_path, return_identity=True)
    input_path = Path(identity["input_manifest_path"]).expanduser().resolve()
    if not input_path.is_file():
        raise AggregateError(f"Pinned paper input manifest is missing: {input_path}")
    if _sha256(input_path).lower() != identity["input_manifest_sha256"]:
        raise AggregateError(f"Pinned paper input manifest SHA-256 changed: {input_path}")
    paper_inputs = _load_object(input_path)
    if (
        paper_inputs.get("schema_version") != 2
        or paper_inputs.get("kind") != "hayatequant_paper_input_validation"
        or paper_inputs.get("status") != "ok"
    ):
        raise AggregateError(f"Paper input manifest is not a completed validation: {input_path}")
    hf = paper_inputs.get("hf_datasets")
    if not isinstance(hf, dict) or hf.get("status") != "validated":
        raise AggregateError(f"Paper input manifest has no validated HF provenance: {input_path}")
    manifest_path = _validated_declared_file(
        hf.get("manifest"), label="paper input pinned HF manifest"
    )
    pinned = _load_object(manifest_path)
    if pinned.get("schema_version") != 1:
        raise AggregateError(f"Pinned HF dataset manifest schema changed: {manifest_path}")
    if pinned.get("hf_home") != hf.get("hf_home"):
        raise AggregateError("Pinned HF manifest and paper input manifest disagree on HF_HOME")
    embedded_datasets = hf.get("datasets")
    pinned_datasets = pinned.get("datasets")
    if not isinstance(embedded_datasets, dict) or not isinstance(pinned_datasets, dict):
        raise AggregateError("Pinned HF dataset maps are missing")
    for key, embedded in embedded_datasets.items():
        raw = pinned_datasets.get(key)
        if not isinstance(embedded, dict) or not isinstance(raw, dict):
            raise AggregateError(f"Pinned HF dataset record is missing for {key}")
        for field in ("repo", "config", "split", "revision", "rows", "dataset_fingerprint"):
            if embedded.get(field) != raw.get(field):
                raise AggregateError(f"Pinned HF dataset provenance differs for {key}.{field}")
        if embedded.get("cache_files") != raw.get("cache_files"):
            raise AggregateError(f"Pinned HF cache-file provenance differs for {key}")
    embedded_longbench = hf.get("longbench_e")
    pinned_longbench = pinned.get("longbench_e")
    if embedded_longbench != pinned_longbench:
        raise AggregateError("Pinned LongBench provenance differs between validated manifests")
    return {
        "input_manifest": {
            "path": str(input_path),
            "sha256": identity["input_manifest_sha256"],
        },
        "hf_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "datasets": embedded_datasets,
        "longbench_e": embedded_longbench,
    }


def _load_pinned_longbench_rows(
    rows: Sequence[tuple[Path, dict, str, dict]],
    *,
    tasks: Sequence[str],
    task_counts: Mapping[str, int],
    dataset_revision: str,
    archive_sha256: str,
    archive_override: Path | None,
) -> tuple[dict[str, list[dict]], dict]:
    """Read the exact pinned LongBench-E rows used to recompute metrics."""
    if archive_override is None:
        provenance = _validated_hf_input_provenance(rows[0][1], result_path=rows[0][0])
        record = provenance.get("longbench_e")
        if (
            not isinstance(record, dict)
            or record.get("repo") != LONGBENCH_DATASET_REPO
            or record.get("revision") != dataset_revision
            or record.get("filename") != LONGBENCH_DATASET_FILENAME
        ):
            raise AggregateError("Pinned LongBench release provenance changed")
        archive_path = _validated_declared_file(record.get("artifact"), label="LongBench data.zip")
        dataset_provenance = {
            "repo": LONGBENCH_DATASET_REPO,
            "revision": dataset_revision,
            "filename": LONGBENCH_DATASET_FILENAME,
            "path": str(archive_path),
            "sha256": archive_sha256,
            "input_manifest": provenance["input_manifest"],
            "hf_manifest": provenance["hf_manifest"],
        }
    else:
        archive_path = Path(archive_override).expanduser().resolve()
        dataset_provenance = {
            "repo": LONGBENCH_DATASET_REPO,
            "revision": dataset_revision,
            "filename": LONGBENCH_DATASET_FILENAME,
            "path": str(archive_path),
            "sha256": archive_sha256,
            "fixture_override": True,
        }
    if not archive_path.is_file() or _sha256(archive_path).lower() != archive_sha256.lower():
        raise AggregateError(f"LongBench data.zip does not match pinned SHA-256: {archive_path}")
    task_rows: dict[str, list[dict]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for task in tasks:
                member = f"data/{task}_e.jsonl"
                try:
                    stream = archive.open(member)
                except KeyError as exc:
                    raise AggregateError(f"Pinned LongBench archive omitted {member}") from exc
                parsed = []
                with stream:
                    for line_number, raw_line in enumerate(stream, start=1):
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise AggregateError(
                                f"Malformed pinned LongBench row {member}:{line_number}"
                            ) from exc
                        if not isinstance(row, dict):
                            raise AggregateError(
                                f"Pinned LongBench row is not an object: {member}:{line_number}"
                            )
                        parsed.append(row)
                if len(parsed) != task_counts[task]:
                    raise AggregateError(
                        f"Pinned LongBench {task} has {len(parsed)} rows, "
                        f"expected {task_counts[task]}"
                    )
                task_rows[task] = parsed
    except (OSError, zipfile.BadZipFile) as exc:
        raise AggregateError(f"Cannot read pinned LongBench data.zip: {archive_path}") from exc
    return task_rows, dataset_provenance


def _longbench_normalize(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _longbench_qa_f1(prediction: str, gold: str) -> float:
    predicted = _longbench_normalize(prediction).split()
    reference = _longbench_normalize(gold).split()
    if not predicted or not reference:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def _longbench_rouge(prediction: str, gold: str) -> float:
    try:
        from rouge import Rouge  # pylint: disable=import-outside-toplevel

        return float(Rouge().get_scores([prediction], [gold], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def _longbench_classification(prediction: str, gold: str, all_classes: Any) -> float:
    matches = [candidate for candidate in (all_classes or []) if candidate in prediction]
    for match in matches:
        if match in gold and match != gold:
            matches.remove(match)
    return 1.0 / len(matches) if gold in matches else 0.0


def _longbench_retrieval(prediction: str, gold: str) -> float:
    target = re.findall(r"Paragraph (\d+)", gold)[0]
    predicted = re.findall(r"\d+", prediction)
    return sum(value == target for value in predicted) / len(predicted) if predicted else 0.0


def _longbench_count(prediction: str, gold: str) -> float:
    predicted = re.findall(r"\d+", prediction)
    return (
        sum(value == str(gold) for value in predicted) / len(predicted)
        if predicted
        else 0.0
    )


def _longbench_code_similarity(prediction: str, gold: str) -> float:
    from fuzzywuzzy import fuzz  # pylint: disable=import-outside-toplevel

    line = ""
    for candidate in prediction.lstrip("\n").split("\n"):
        if "`" not in candidate and "#" not in candidate and "//" not in candidate:
            line = candidate
            break
    return fuzz.ratio(line, gold) / 100.0


def _recompute_longbench_score(
    task: str, prediction: str, gold: str, *, all_classes: Any
) -> float:
    if task in {"qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "triviaqa"}:
        return _longbench_qa_f1(prediction, gold)
    if task in {"gov_report", "multi_news", "samsum"}:
        return _longbench_rouge(prediction, gold)
    if task == "trec":
        return _longbench_classification(prediction, gold, all_classes)
    if task == "passage_count":
        return _longbench_count(prediction, gold)
    if task == "passage_retrieval_en":
        return _longbench_retrieval(prediction, gold)
    if task in {"lcc", "repobench-p"}:
        return _longbench_code_similarity(prediction, gold)
    raise AggregateError(f"No pinned LongBench scorer for task {task!r}")


def _load_pinned_math500_rows(
    rows: Sequence[tuple[Path, dict, str, dict]],
    *,
    dataset_size: int,
    dataset_revision: str,
    dataset_path_override: Path | None,
    dataset_sha256_override: str | None,
) -> tuple[list[dict], dict]:
    """Load the exact MATH-500 rows committed to the paper input manifest."""
    if dataset_path_override is None:
        provenance = _validated_hf_input_provenance(rows[0][1], result_path=rows[0][0])
        record = provenance["datasets"].get("math500_test")
        if (
            not isinstance(record, dict)
            or record.get("repo") != MATH500_DATASET_REPO
            or record.get("config") is not MATH500_DATASET_CONFIG
            or record.get("split") != MATH500_DATASET_SPLIT
            or record.get("revision") != dataset_revision
            or record.get("rows") != dataset_size
            or not isinstance(record.get("dataset_fingerprint"), str)
            or not record["dataset_fingerprint"]
            or not isinstance(record.get("cache_files"), list)
            or len(record["cache_files"]) != 1
        ):
            raise AggregateError("Pinned MATH-500 dataset provenance changed")
        dataset_path = _validated_declared_file(
            record["cache_files"][0], label="MATH-500 test Arrow file"
        )
        dataset_digest = record["cache_files"][0]["sha256"].lower()
        dataset_provenance = {
            "repo": MATH500_DATASET_REPO,
            "config": MATH500_DATASET_CONFIG,
            "split": MATH500_DATASET_SPLIT,
            "revision": dataset_revision,
            "dataset_fingerprint": record["dataset_fingerprint"],
            "path": str(dataset_path),
            "sha256": dataset_digest,
            "input_manifest": provenance["input_manifest"],
            "hf_manifest": provenance["hf_manifest"],
        }
    else:
        dataset_path = Path(dataset_path_override).expanduser().resolve()
        if not _sha256_string(dataset_sha256_override):
            raise AggregateError("MATH-500 fixture override requires a pinned SHA-256")
        dataset_digest = str(dataset_sha256_override).lower()
        dataset_provenance = {
            "repo": MATH500_DATASET_REPO,
            "config": MATH500_DATASET_CONFIG,
            "split": MATH500_DATASET_SPLIT,
            "revision": dataset_revision,
            "path": str(dataset_path),
            "sha256": dataset_digest,
            "fixture_override": True,
        }
    if not dataset_path.is_file() or _sha256(dataset_path).lower() != dataset_digest:
        raise AggregateError(
            f"MATH-500 dataset file does not match pinned SHA-256: {dataset_path}"
        )
    try:
        if dataset_path.suffix == ".jsonl":
            parsed = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif dataset_path.suffix == ".json":
            parsed = json.loads(dataset_path.read_text(encoding="utf-8"))
        else:
            from datasets import Dataset  # pylint: disable=import-outside-toplevel

            dataset = Dataset.from_file(str(dataset_path))
            parsed = [dict(dataset[index]) for index in range(len(dataset))]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AggregateError(f"Cannot load pinned MATH-500 rows: {dataset_path}") from exc
    if not isinstance(parsed, list) or len(parsed) != dataset_size:
        raise AggregateError(
            f"Pinned MATH-500 has {len(parsed) if isinstance(parsed, list) else 'invalid'} "
            f"rows, expected {dataset_size}"
        )
    if not all(isinstance(row, dict) for row in parsed):
        raise AggregateError("Pinned MATH-500 rows must be JSON objects")
    return parsed, dataset_provenance


def _build_pinned_math500_scorer():
    """Build the same pinned Math-Verify 0.9.0 scorer as the evaluator."""
    try:
        from math_verify import (  # pylint: disable=import-outside-toplevel
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError as exc:
        raise AggregateError(
            "MATH-500 aggregation requires math-verify[antlr4_9_3]==0.9.0"
        ) from exc
    try:
        installed = package_version("math-verify")
    except PackageNotFoundError as exc:
        raise AggregateError("Cannot verify the installed Math-Verify package version") from exc
    if installed != "0.9.0":
        raise AggregateError(
            f"MATH-500 aggregation requires math-verify==0.9.0, found {installed}"
        )
    gold_config = (LatexExtractionConfig(boxed_match_priority=0),)
    prediction_config = (LatexExtractionConfig(), ExprExtractionConfig())

    @lru_cache(maxsize=512)
    def parse_gold(reference: str):
        parsed = parse(
            f"${reference.strip()}$",
            extraction_config=gold_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        if not parsed:
            raise AggregateError(
                f"Math-Verify could not parse pinned MATH-500 gold: {reference!r}"
            )
        return parsed

    def score(prediction: str, reference: str) -> bool:
        gold = parse_gold(str(reference))
        parsed_prediction = parse(
            str(prediction),
            extraction_config=prediction_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        return bool(parsed_prediction) and bool(
            verify(
                gold,
                parsed_prediction,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                allow_set_relation_comp=False,
                timeout_seconds=5,
                raise_on_error=False,
            )
        )

    return score


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AggregateError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregateError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise AggregateError(f"{label} must be finite and >= {minimum}, got {value!r}")
    return result


def _ppl_target_coverage_contract(
    row: Mapping[str, Any], *, label: str, expected_tokens: int | None = None
) -> tuple[int, int]:
    """Validate the exact global next-token set, not just its scalar size."""
    n_tokens = _integer(row.get("ppl_n_tokens"), label=f"{label} n_tokens", minimum=1)
    traversed = _integer(
        row.get("ppl_traversed_tokens"), label=f"{label} traversed_tokens", minimum=2
    )
    if traversed != n_tokens + 1:
        raise AggregateError(
            f"PPL traversed/scored off-by-one contract changed for {label}: "
            f"traversed={traversed}, scored={n_tokens}"
        )
    if expected_tokens is not None and n_tokens != expected_tokens:
        raise AggregateError(f"PPL frozen denominator changed for {label}")
    if row.get("ppl_scored_target_index_set") != f"[1,{traversed})":
        raise AggregateError(f"PPL scored target-index set changed for {label}")
    if row.get("ppl_scored_target_coverage") != "contiguous_exact_once":
        raise AggregateError(f"PPL target coverage is not contiguous_exact_once for {label}")
    if row.get("ppl_n_skipped_nonfinite") != 0:
        raise AggregateError(f"PPL skipped a non-finite window for {label}")
    n_windows = _integer(
        row.get("ppl_n_windows"), label=f"{label} n_windows", minimum=1
    )
    ranges = row.get("ppl_scored_target_ranges")
    if not isinstance(ranges, list) or len(ranges) != n_windows:
        raise AggregateError(f"PPL target-range ledger/window count changed for {label}")
    expected_start = 1
    for index, target_range in enumerate(ranges):
        if not isinstance(target_range, list) or len(target_range) != 2:
            raise AggregateError(f"PPL target range {index} is malformed for {label}")
        start = _integer(target_range[0], label=f"{label} range {index} start", minimum=1)
        stop = _integer(target_range[1], label=f"{label} range {index} stop", minimum=2)
        if start != expected_start or stop <= start:
            raise AggregateError(
                f"PPL target ranges are gapped, overlapping, or empty for {label}"
            )
        expected_start = stop
    if expected_start != traversed:
        raise AggregateError(f"PPL target ranges do not reach traversed_tokens for {label}")
    return n_tokens, traversed


def _load_frozen_ppl_denominator_contract() -> dict:
    """Load and validate the repo-pinned tokenizer-specific WikiText contract."""
    document = _load_object(PPL_DENOMINATOR_CONTRACT_PATH)
    source = document.get("source")
    models = document.get("models")
    tokenization = document.get("tokenization")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "hayatequant_formal_wikitext2_denominator_contract"
        or not isinstance(source, dict)
        or source.get("repo") != "Salesforce/wikitext"
        or source.get("config") != "wikitext-2-raw-v1"
        or source.get("split") != "test"
        or source.get("revision") != WIKITEXT2_REVISION
        or source.get("rows") != 4358
        or source.get("joined_text_sha256") != WIKITEXT2_TEXT_SHA256
        or not _sha256_string(source.get("cache_file_sha256"))
        or not isinstance(source.get("cache_file_bytes"), int)
        or source["cache_file_bytes"] <= 0
        or not isinstance(tokenization, dict)
        or tokenization.get("call") != "tokenizer(text, return_tensors='pt')"
        or tokenization.get("add_special_tokens") != "tokenizer_default"
        or tokenization.get("joined_row_separator") != "\\n\\n"
        or tokenization.get("scored_target_index_set") != "[1,traversed_tokens)"
        or tokenization.get("coverage") != "contiguous_exact_once"
        or not isinstance(models, dict)
        or set(models) != set(PAPER_MODELS)
    ):
        raise AggregateError(
            f"Frozen formal PPL denominator contract is malformed: "
            f"{PPL_DENOMINATOR_CONTRACT_PATH}"
        )
    for model, record in models.items():
        if (
            not isinstance(record, dict)
            or record.get("model_manifest_key") != MODEL_MANIFEST_KEYS[model]
            or not isinstance(record.get("tokenizer_files"), dict)
            or not record["tokenizer_files"]
            or any(
                not isinstance(name, str) or not _sha256_string(digest)
                for name, digest in record["tokenizer_files"].items()
            )
        ):
            raise AggregateError(f"Frozen formal PPL model contract is malformed for {model}")
        n_tokens = _integer(
            record.get("n_tokens"), label=f"frozen {model} n_tokens", minimum=1
        )
        traversed = _integer(
            record.get("traversed_tokens"),
            label=f"frozen {model} traversed_tokens",
            minimum=2,
        )
        if traversed != n_tokens + 1:
            raise AggregateError(f"Frozen formal PPL off-by-one contract changed for {model}")
    return document


def _validated_frozen_ppl_denominator(
    model: str, *, input_manifest_path: str, input_manifest_sha256: str
) -> dict:
    """Bind one frozen denominator to the exact audited dataset and tokenizer."""
    contract = _load_frozen_ppl_denominator_contract()
    expected = contract["models"].get(model)
    if not isinstance(expected, dict):
        raise AggregateError(f"No frozen formal PPL denominator exists for {model}")
    manifest_path = Path(input_manifest_path).expanduser().resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != input_manifest_sha256:
        raise AggregateError(f"PPL input manifest is missing or changed for {model}")
    manifest = _load_object(manifest_path)
    models = manifest.get("models")
    hf = manifest.get("hf_datasets")
    datasets = hf.get("datasets") if isinstance(hf, dict) else None
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "hayatequant_paper_input_validation"
        or manifest.get("status") != "ok"
        or not isinstance(models, dict)
        or not isinstance(datasets, dict)
        or hf.get("status") != "validated"
    ):
        raise AggregateError(f"PPL input manifest is not a validated paper manifest for {model}")
    model_record = models.get(expected["model_manifest_key"])
    metadata = model_record.get("metadata_files") if isinstance(model_record, dict) else None
    if not isinstance(metadata, dict):
        raise AggregateError(f"PPL input manifest has no tokenizer ledger for {model}")
    for filename, digest in expected["tokenizer_files"].items():
        file_record = metadata.get(filename)
        _validated_declared_file(file_record, label=f"{model} frozen tokenizer {filename}")
        if file_record.get("sha256") != digest:
            raise AggregateError(f"Frozen tokenizer digest changed for {model}/{filename}")
    source = contract["source"]
    dataset = datasets.get("wikitext2_test")
    cache_files = dataset.get("cache_files") if isinstance(dataset, dict) else None
    if (
        not isinstance(dataset, dict)
        or dataset.get("repo") != source["repo"]
        or dataset.get("config") != source["config"]
        or dataset.get("split") != source["split"]
        or dataset.get("revision") != source["revision"]
        or dataset.get("rows") != source["rows"]
        or not isinstance(cache_files, list)
        or len(cache_files) != 1
    ):
        raise AggregateError(f"Frozen WikiText-2 source ledger changed for {model}")
    cache_record = cache_files[0]
    _validated_declared_file(cache_record, label=f"{model} frozen WikiText-2 Arrow")
    if (
        cache_record.get("bytes") != source["cache_file_bytes"]
        or cache_record.get("sha256") != source["cache_file_sha256"]
    ):
        raise AggregateError(f"Frozen WikiText-2 Arrow identity changed for {model}")
    return {
        "n_tokens": int(expected["n_tokens"]),
        "traversed_tokens": int(expected["traversed_tokens"]),
    }


def _validated_ppl_numbers(
    row: Mapping[str, Any], *, label: str, expected_tokens: int | None = None
) -> tuple[float, float, float, int]:
    """Validate unrounded PPL, mean NLL, numerator, and token denominator."""
    if (
        row.get("ppl_nll_accumulation") != PPL_NLL_ACCUMULATION
        or row.get("ppl_window_cache_lifetime") != PPL_WINDOW_CACHE_LIFETIME
        or row.get("ppl_cross_window_tensor_retention")
        != PPL_CROSS_WINDOW_TENSOR_RETENTION
    ):
        raise AggregateError(f"PPL accumulation/cache-lifetime provenance changed for {label}")
    n_tokens, _ = _ppl_target_coverage_contract(
        row, label=label, expected_tokens=expected_tokens
    )
    ppl = _number(row.get("ppl"), label=f"{label} ppl")
    mean_nll = _number(row.get("ppl_nll"), label=f"{label} mean NLL")
    nll_sum = _number(row.get("ppl_nll_sum"), label=f"{label} NLL numerator")
    if ppl <= 0.0:
        raise AggregateError(f"Non-positive PPL for {label}")
    if not math.isclose(nll_sum / n_tokens, mean_nll, rel_tol=0.0, abs_tol=1e-10):
        raise AggregateError(f"NLL numerator/denominator mismatch for {label}")
    if not math.isclose(math.log(ppl), mean_nll, rel_tol=0.0, abs_tol=1e-10):
        raise AggregateError(f"PPL is inconsistent with mean NLL for {label}")
    return ppl, mean_nll, nll_sum, n_tokens


def _expected_teacher_forced_contract(method: str | None) -> tuple[str, str | None]:
    if method == "BlockQuant-paper":
        return (
            CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
            BLOCKQUANT_TEMPORAL_DEPENDENCY,
        )
    if method == "AdaptiveHQ":
        return (
            CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
            ADAPTIVE_HQ_TEMPORAL_DEPENDENCY,
        )
    return PPL_TEACHER_FORCED_PROTOCOL, None


def _validated_ppl_protocol_and_compression(
    row: Mapping[str, Any], *, label: str, compressed: bool, method: str | None = None
) -> dict:
    """Validate causal scoring and exact across-window cache-bit accounting."""
    expected_protocol, expected_dependency = _expected_teacher_forced_contract(method)
    if (
        row.get("ppl_teacher_forced_scoring_protocol") != expected_protocol
        or row.get("ppl_teacher_forced_temporal_dependency") != expected_dependency
    ):
        raise AggregateError(f"Wrong teacher-forced scoring protocol for {label}")

    ratio = _number(
        row.get("ppl_compression_ratio"),
        label=f"{label} PPL compression ratio",
    )
    aggregation = row.get("ppl_compression_aggregation")
    fp16_bits = row.get("ppl_compression_fp16_bits")
    stored_bits = row.get("ppl_compression_stored_bits")
    if compressed:
        if aggregation != "exact_cache_bit_counters":
            raise AggregateError(f"PPL compression is not exact total-bit accounting for {label}")
        fp16_bits = _integer(fp16_bits, label=f"{label} PPL fp16 bits", minimum=1)
        stored_bits = _integer(stored_bits, label=f"{label} PPL stored bits", minimum=1)
        expected_ratio = fp16_bits / stored_bits
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-15, abs_tol=0.0):
            raise AggregateError(
                f"PPL compression ratio is not total fp16_bits/stored_bits for {label}"
            )
    else:
        if (
            aggregation != "fullkv_no_cache"
            or fp16_bits is not None
            or stored_bits is not None
            or ratio != 1.0
        ):
            raise AggregateError(f"FullKV PPL cache-accounting provenance changed for {label}")
    return {
        "ppl_compression_ratio": ratio,
        "ppl_compression_aggregation": aggregation,
        "ppl_compression_fp16_bits": fp16_bits,
        "ppl_compression_stored_bits": stored_bits,
        "ppl_teacher_forced_scoring_protocol": expected_protocol,
        "ppl_teacher_forced_temporal_dependency": expected_dependency,
    }


def _validated_zeroshot_compression(
    result: Mapping[str, Any], *, label: str, compressed: bool
) -> tuple[int | None, int | None]:
    """Validate one zero-shot shard's exact choice/example bit totals."""
    ratio = _number(result.get("compression_ratio"), label=f"{label} compression ratio")
    aggregation = result.get("compression_aggregation")
    fp16_bits = result.get("compression_fp16_bits")
    stored_bits = result.get("compression_stored_bits")
    if compressed:
        if aggregation != "exact_cache_bit_counters":
            raise AggregateError(f"Zero-shot compression is not exact bit totals for {label}")
        fp16_bits = _integer(fp16_bits, label=f"{label} fp16 bits", minimum=1)
        stored_bits = _integer(stored_bits, label=f"{label} stored bits", minimum=1)
        if not math.isclose(
            ratio,
            round(fp16_bits / stored_bits, 3),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AggregateError(f"Zero-shot displayed compression ratio is stale for {label}")
    return fp16_bits, stored_bits


def _validated_long_compression_sufficient_statistics(
    result: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    label: str,
) -> dict[str, Any]:
    """Replay exact per-generation cache counters and AdaptiveHQ provenance."""

    if not records:
        raise AggregateError(f"{label} has no generation records for compression accounting")
    compressed = method != "FullKV"
    fp16_total = 0
    stored_total = 0
    adaptive_rows = []
    allowed_pairs: set[tuple[int, int]] = set()
    if method == "AdaptiveHQ":
        resolved = result.get("resolved_codec_config")
        allocation = resolved.get("adaptive_allocation") if isinstance(resolved, Mapping) else None
        raw_pairs = allocation.get("candidate_pairs") if isinstance(allocation, Mapping) else None
        if not isinstance(raw_pairs, list):
            raise AggregateError(f"{label} lacks the resolved AdaptiveHQ candidate set")
        allowed_pairs = {
            tuple(pair)
            for pair in raw_pairs
            if isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(bit, int) and not isinstance(bit, bool) for bit in pair)
        }
        if allowed_pairs not in [set(pairs) for pairs in ADAPTIVE_HQ_CANDIDATE_PAIRS.values()]:
            raise AggregateError(f"{label} AdaptiveHQ candidate set changed")
    for index, row in enumerate(records):
        prefix = f"{label} generation {index}"
        mode = row.get("compression_aggregation")
        fp16 = row.get("compression_fp16_bits")
        stored = row.get("compression_stored_bits")
        runtime = row.get("adaptive_hq_runtime")
        if not compressed:
            if (
                mode != "fullkv_no_cache"
                or fp16 is not None
                or stored is not None
                or runtime is not None
            ):
                raise AggregateError(f"{prefix} FullKV compression accounting changed")
            continue
        if mode != "exact_cache_bit_counters":
            raise AggregateError(f"{prefix} lacks exact cache-bit counters")
        fp16_total += _integer(fp16, label=f"{prefix} fp16 bits", minimum=1)
        stored_total += _integer(stored, label=f"{prefix} stored bits", minimum=1)
        if method == "AdaptiveHQ":
            if not isinstance(runtime, Mapping):
                raise AggregateError(f"{prefix} lacks AdaptiveHQ runtime provenance")
            if (
                runtime.get("schema_version") != 1
                or runtime.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
                or runtime.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
                or runtime.get("selector_incremental_stored_bits_total") != 0
            ):
                raise AggregateError(f"{prefix} AdaptiveHQ policy/carrier contract changed")
            encoded_updates = _integer(
                runtime.get("encoded_updates"), label=f"{prefix} encoded_updates", minimum=1
            )
            encoded_tokens = _integer(
                runtime.get("encoded_tokens"), label=f"{prefix} encoded_tokens", minimum=1
            )
            selector_bits = _integer(
                runtime.get("selector_bits_total"),
                label=f"{prefix} selector_bits_total",
                minimum=1,
            )
            if selector_bits != encoded_updates:
                raise AggregateError(f"{prefix} AdaptiveHQ must store one selector bit/update")
            histogram = runtime.get("split_histogram")
            if not isinstance(histogram, Mapping) or not histogram:
                raise AggregateError(f"{prefix} AdaptiveHQ selector histogram is missing")
            histogram_vectors = 0
            for pair_key, bucket in histogram.items():
                if (
                    not isinstance(pair_key, str)
                    or not isinstance(bucket, Mapping)
                    or not isinstance(bucket.get("pair"), list)
                    or len(bucket["pair"]) != 2
                ):
                    raise AggregateError(f"{prefix} AdaptiveHQ selector histogram is malformed")
                pair = tuple(bucket["pair"])
                if pair not in allowed_pairs or pair_key != f"{pair[0]}:{pair[1]}":
                    raise AggregateError(f"{prefix} AdaptiveHQ selected an undeclared pair")
                bucket_updates = _integer(
                    bucket.get("updates"),
                    label=f"{prefix} histogram {pair_key}/updates",
                    minimum=1,
                )
                bucket_tokens = _integer(
                    bucket.get("tokens"),
                    label=f"{prefix} histogram {pair_key}/tokens",
                    minimum=1,
                )
                bucket_vectors = _integer(
                    bucket.get("vectors"),
                    label=f"{prefix} histogram {pair_key}/vectors",
                    minimum=1,
                )
                if not bucket_updates <= bucket_tokens <= bucket_vectors:
                    raise AggregateError(
                        f"{prefix} AdaptiveHQ histogram must satisfy updates<=tokens<=vectors"
                    )
                histogram_vectors += bucket_vectors
            if (
                sum(int(bucket["updates"]) for bucket in histogram.values()) != encoded_updates
                or sum(int(bucket["tokens"]) for bucket in histogram.values()) != encoded_tokens
                or histogram_vectors < encoded_tokens
            ):
                raise AggregateError(f"{prefix} AdaptiveHQ histogram totals do not replay")
            adaptive_rows.append(runtime)
        elif runtime is not None:
            raise AggregateError(f"{prefix} non-adaptive method has AdaptiveHQ provenance")

    if not compressed:
        expected = {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": len(records),
            "adaptive_hq_runtime": None,
        }
    else:
        adaptive = None
        if method == "AdaptiveHQ":
            histogram: dict[str, dict[str, Any]] = {}
            for runtime in adaptive_rows:
                for pair_key, bucket in runtime["split_histogram"].items():
                    merged = histogram.setdefault(
                        pair_key,
                        {
                            "pair": list(bucket["pair"]),
                            "updates": 0,
                            "tokens": 0,
                            "vectors": 0,
                        },
                    )
                    if merged["pair"] != bucket["pair"]:
                        raise AggregateError(f"{label} AdaptiveHQ histogram pair changed")
                    for field in ("updates", "tokens", "vectors"):
                        merged[field] += int(bucket[field])
            adaptive = {
                "schema_version": 1,
                "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
                "encoded_updates": sum(int(row["encoded_updates"]) for row in adaptive_rows),
                "encoded_tokens": sum(int(row["encoded_tokens"]) for row in adaptive_rows),
                "selector_bits_total": sum(
                    int(row["selector_bits_total"]) for row in adaptive_rows
                ),
                "selector_incremental_stored_bits_total": 0,
                "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
                "split_histogram": histogram,
            }
        expected = {
            "compression_fp16_bits": fp16_total,
            "compression_stored_bits": stored_total,
            "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
            "compression_observations": len(records),
            "adaptive_hq_runtime": adaptive,
        }
    observed = result.get("compression_sufficient_statistics")
    if observed != expected:
        raise AggregateError(f"{label} compression sufficient statistics do not replay")
    return expected


def _require_long_exact_payload_match(
    method_rows: Mapping[str, dict[str, Any]], *, label: str
) -> None:
    """Require exact per-example matched rates without forcing equal decode length."""

    matched_methods = ("TurboQuant-paper", "FibQuant-paper", *PROPOSAL_METHODS)
    if set(method_rows) == {"FullKV"}:
        # Focused unit/custom baselines have no compressed comparison to match.
        # Formal callers still admit their complete method universe upstream.
        return
    if not set(matched_methods).issubset(method_rows):
        raise AggregateError(f"{label} lacks the formal exact-rate method set")
    reference_ledger = method_rows["TurboQuant-paper"].get("compression_ledger")
    if not isinstance(reference_ledger, list) or not reference_ledger:
        raise AggregateError(f"{label} TurboQuant per-example compression ledger is missing")
    reference_by_id = {}
    for row in reference_ledger:
        if not isinstance(row, Mapping) or not isinstance(row.get("sample_id"), str):
            raise AggregateError(f"{label} TurboQuant compression ledger is malformed")
        if row["sample_id"] in reference_by_id:
            raise AggregateError(f"{label} duplicate TurboQuant sample id {row['sample_id']}")
        reference_by_id[row["sample_id"]] = row
    method_rows["TurboQuant-paper"]["matched_rate_reference"] = {
        "method": "TurboQuant-paper",
        "n_samples": len(reference_by_id),
    }
    for method in matched_methods[1:]:
        ledger = method_rows[method].get("compression_ledger")
        if not isinstance(ledger, list) or not ledger:
            raise AggregateError(f"{label}/{method} per-example compression ledger is missing")
        by_id = {row.get("sample_id"): row for row in ledger if isinstance(row, Mapping)}
        if len(by_id) != len(ledger) or set(by_id) != set(reference_by_id):
            raise AggregateError(f"{label}/{method} compression sample universe changed")
        generated_length_differences = 0
        for sample_id, reference in reference_by_id.items():
            candidate = by_id[sample_id]
            for field in ("input_tokens", "max_new_tokens"):
                if candidate.get(field) != reference.get(field):
                    raise AggregateError(
                        f"{label}/{method}/{sample_id} input/budget contract differs"
                    )
            tq_fp16 = _integer(
                reference.get("compression_fp16_bits"),
                label=f"{label}/{sample_id} TQ fp16 bits",
                minimum=1,
            )
            tq_stored = _integer(
                reference.get("compression_stored_bits"),
                label=f"{label}/{sample_id} TQ stored bits",
                minimum=1,
            )
            candidate_fp16 = _integer(
                candidate.get("compression_fp16_bits"),
                label=f"{label}/{method}/{sample_id} fp16 bits",
                minimum=1,
            )
            candidate_stored = _integer(
                candidate.get("compression_stored_bits"),
                label=f"{label}/{method}/{sample_id} stored bits",
                minimum=1,
            )
            if candidate_stored * tq_fp16 != tq_stored * candidate_fp16:
                raise AggregateError(
                    f"{label}/{method}/{sample_id} exact integer stored/fp16 rate "
                    "differs from TurboQuant-paper"
                )
            generated_length_differences += int(
                candidate.get("generated_tokens") != reference.get("generated_tokens")
            )
            if method == "AdaptiveHQ":
                runtime = candidate.get("adaptive_hq_runtime")
                if (
                    not isinstance(runtime, Mapping)
                    or runtime.get("selector_incremental_stored_bits_total") != 0
                    or runtime.get("selector_carrier_version")
                    != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
                ):
                    raise AggregateError(
                        f"{label}/{method}/{sample_id} AdaptiveHQ carrier audit failed"
                    )
        method_rows[method]["matched_rate_vs_turboquant"] = {
            "exact_integer_cross_products": True,
            "matched_sample_universe": True,
            "input_tokens_and_max_budget_identical": True,
            "generated_lengths_forced_equal": False,
            "generated_length_difference_count": generated_length_differences,
            "n_samples": len(reference_by_id),
        }


def _sum_long_compression_statistics(
    rows: Sequence[Mapping[str, Any]], *, method: str, label: str
) -> dict[str, Any]:
    """Sum already replayed shard/task statistics without averaging ratios."""

    if not rows:
        raise AggregateError(f"{label} has no compression statistics to sum")
    if method == "FullKV":
        if any(
            row.get("compression_aggregation") != "fullkv_no_cache"
            or row.get("compression_fp16_bits") is not None
            or row.get("compression_stored_bits") is not None
            or row.get("adaptive_hq_runtime") is not None
            for row in rows
        ):
            raise AggregateError(f"{label} FullKV shard accounting changed")
        return {
            "compression_fp16_bits": None,
            "compression_stored_bits": None,
            "compression_aggregation": "fullkv_no_cache",
            "compression_observations": sum(
                _integer(
                    row.get("compression_observations"),
                    label=f"{label} observations",
                    minimum=1,
                )
                for row in rows
            ),
            "adaptive_hq_runtime": None,
        }
    for row in rows:
        if row.get("compression_aggregation") != ("sum_exact_per_generation_cache_bit_counters"):
            raise AggregateError(f"{label} contains non-exact shard accounting")
    adaptive = None
    if method == "AdaptiveHQ":
        runtimes = [row.get("adaptive_hq_runtime") for row in rows]
        if not all(isinstance(runtime, Mapping) for runtime in runtimes):
            raise AggregateError(f"{label} lacks AdaptiveHQ shard provenance")
        histogram: dict[str, dict[str, Any]] = {}
        for runtime in runtimes:
            assert isinstance(runtime, Mapping)
            if (
                runtime.get("schema_version") != 1
                or runtime.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
                or runtime.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
                or runtime.get("selector_incremental_stored_bits_total") != 0
            ):
                raise AggregateError(f"{label} AdaptiveHQ shard policy/carrier changed")
            encoded_updates = _integer(
                runtime.get("encoded_updates"),
                label=f"{label} AdaptiveHQ shard encoded_updates",
                minimum=1,
            )
            encoded_tokens = _integer(
                runtime.get("encoded_tokens"),
                label=f"{label} AdaptiveHQ shard encoded_tokens",
                minimum=1,
            )
            selector_bits = _integer(
                runtime.get("selector_bits_total"),
                label=f"{label} AdaptiveHQ shard selector_bits_total",
                minimum=1,
            )
            if selector_bits != encoded_updates:
                raise AggregateError(f"{label} AdaptiveHQ must store one selector bit/update")
            raw_histogram = runtime.get("split_histogram")
            if not isinstance(raw_histogram, Mapping) or not raw_histogram:
                raise AggregateError(f"{label} AdaptiveHQ shard histogram is missing")
            runtime_updates = runtime_tokens = runtime_vectors = 0
            observed_pairs: set[tuple[int, int]] = set()
            for pair_key, bucket in raw_histogram.items():
                if (
                    not isinstance(pair_key, str)
                    or not isinstance(bucket, Mapping)
                    or not isinstance(bucket.get("pair"), list)
                    or len(bucket["pair"]) != 2
                ):
                    raise AggregateError(f"{label} AdaptiveHQ shard histogram is malformed")
                pair = tuple(bucket["pair"])
                if (
                    any(
                        isinstance(bit, bool) or not isinstance(bit, int) or bit < 1
                        for bit in pair
                    )
                    or pair_key != f"{pair[0]}:{pair[1]}"
                ):
                    raise AggregateError(f"{label} AdaptiveHQ shard pair is malformed")
                observed_pairs.add(pair)
                bucket_updates = _integer(
                    bucket.get("updates"),
                    label=f"{label} AdaptiveHQ shard {pair_key}/updates",
                    minimum=1,
                )
                bucket_tokens = _integer(
                    bucket.get("tokens"),
                    label=f"{label} AdaptiveHQ shard {pair_key}/tokens",
                    minimum=1,
                )
                bucket_vectors = _integer(
                    bucket.get("vectors"),
                    label=f"{label} AdaptiveHQ shard {pair_key}/vectors",
                    minimum=1,
                )
                if not bucket_updates <= bucket_tokens <= bucket_vectors:
                    raise AggregateError(
                        f"{label} AdaptiveHQ shard must satisfy updates<=tokens<=vectors"
                    )
                runtime_updates += bucket_updates
                runtime_tokens += bucket_tokens
                runtime_vectors += bucket_vectors
                merged = histogram.setdefault(
                    pair_key,
                    {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
                )
                if merged["pair"] != bucket["pair"]:
                    raise AggregateError(f"{label} AdaptiveHQ selector pair changed")
                for field in ("updates", "tokens", "vectors"):
                    merged[field] += int(bucket[field])
            if (
                runtime_updates != encoded_updates
                or runtime_tokens != encoded_tokens
                or runtime_vectors < encoded_tokens
                or not any(
                    observed_pairs <= set(candidate_pairs)
                    for candidate_pairs in ADAPTIVE_HQ_CANDIDATE_PAIRS.values()
                )
            ):
                raise AggregateError(f"{label} AdaptiveHQ shard histogram does not replay")
        adaptive = {
            "schema_version": 1,
            "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
            "encoded_updates": sum(int(row["encoded_updates"]) for row in runtimes),
            "encoded_tokens": sum(int(row["encoded_tokens"]) for row in runtimes),
            "selector_bits_total": sum(int(row["selector_bits_total"]) for row in runtimes),
            "selector_incremental_stored_bits_total": 0,
            "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "split_histogram": histogram,
        }
    elif any(row.get("adaptive_hq_runtime") is not None for row in rows):
        raise AggregateError(f"{label} non-adaptive shard has AdaptiveHQ provenance")
    return {
        "compression_fp16_bits": sum(
            _integer(
                row.get("compression_fp16_bits"),
                label=f"{label} fp16 bits",
                minimum=1,
            )
            for row in rows
        ),
        "compression_stored_bits": sum(
            _integer(
                row.get("compression_stored_bits"),
                label=f"{label} stored bits",
                minimum=1,
            )
            for row in rows
        ),
        "compression_aggregation": "sum_exact_per_generation_cache_bit_counters",
        "compression_observations": sum(
            _integer(
                row.get("compression_observations"),
                label=f"{label} observations",
                minimum=1,
            )
            for row in rows
        ),
        "adaptive_hq_runtime": adaptive,
    }


def _long_compression_ledger_row(
    record: Mapping[str, Any],
    *,
    sample_id: str,
    input_tokens: int,
    max_new_tokens: int,
    generated_tokens: int,
) -> dict[str, Any]:
    """Retain the minimal exact per-generation evidence needed cross-method."""

    return {
        "sample_id": sample_id,
        "input_tokens": int(input_tokens),
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": int(generated_tokens),
        "compression_fp16_bits": record.get("compression_fp16_bits"),
        "compression_stored_bits": record.get("compression_stored_bits"),
        "compression_aggregation": record.get("compression_aggregation"),
        "adaptive_hq_runtime": record.get("adaptive_hq_runtime"),
    }
    if (
        aggregation != "fullkv_no_cache"
        or fp16_bits is not None
        or stored_bits is not None
        or ratio != 1.0
    ):
        raise AggregateError(f"FullKV zero-shot compression provenance changed for {label}")
    return None, None


def _validated_stored_rate(row: Mapping[str, Any], *, label: str) -> tuple[int, int, float]:
    """Recompute the unrounded stored rate from exact integer statistics."""
    stored_bits = _integer(row.get("stored_bits"), label=f"{label} stored_bits", minimum=1)
    stored_coordinates = _integer(
        row.get("stored_coordinates"),
        label=f"{label} stored_coordinates",
        minimum=1,
    )
    reported = _number(row.get("stored_bits_per_coord"), label=f"{label} stored_bits_per_coord")
    recomputed = stored_bits / stored_coordinates
    if not math.isclose(reported, recomputed, rel_tol=0.0, abs_tol=1e-15):
        raise AggregateError(
            f"Stored-rate numerator/denominator mismatch for {label}: "
            f"{stored_bits}/{stored_coordinates} != {reported}"
        )
    return stored_bits, stored_coordinates, recomputed


def _validated_blockquant_stored_accounting(
    row: Mapping[str, Any],
    *,
    stored_bits: int,
    stored_coordinates: int,
    label: str,
) -> dict:
    """Prove every logical bit in the versioned dynamic-top32 completion."""
    accounting = row.get("stored_bit_accounting")
    if not isinstance(accounting, dict):
        raise AggregateError(f"Missing BlockQuant stored-bit decomposition for {label}")
    if (
        accounting.get("scope") != "fixed-width-data-plane-growth-only"
        or accounting.get("physical_packing_claim") is not False
        or accounting.get("head_dim") != 128
    ):
        raise AggregateError(f"BlockQuant stored-bit scope changed for {label}")
    vectors = _integer(
        accounting.get("vectors_per_role"), label=f"{label} vectors_per_role", minimum=1
    )
    if stored_coordinates != 2 * vectors * 128:
        raise AggregateError(f"BlockQuant coordinate denominator changed for {label}")
    key = accounting.get("key")
    value = accounting.get("value")
    if not isinstance(key, dict) or not isinstance(value, dict):
        raise AggregateError(f"Malformed BlockQuant role bit accounting for {label}")
    identity_fields = _integer(
        key.get("dynamic_channel_identity_fields"),
        label=f"{label} dynamic identity fields",
        minimum=1,
    )
    if identity_fields > vectors or vectors % identity_fields:
        raise AggregateError(f"Invalid BlockQuant update-level identity count for {label}")
    identity_bits = identity_fields * 101  # ceil(log2(C(128, 32)))
    key_code_bits = vectors * (32 * 4 + 96 * 3)
    key_scale_bits = vectors * 2 * 16
    key_payload_bits = key_code_bits + identity_bits
    key_total_bits = key_payload_bits + key_scale_bits
    value_payload_bits = vectors * 128 * 2
    value_scale_bits = vectors * 16
    value_total_bits = value_payload_bits + value_scale_bits
    expected_key = {
        "code_bits": key_code_bits,
        "payload_bits": key_payload_bits,
        "fp16_scale_bits": key_scale_bits,
        "scale_fields_per_vector": 2,
        "fixed_channel_identity_bits": 0,
        "dynamic_channel_identity_bits": identity_bits,
        "dynamic_channel_identity_fields": identity_fields,
        "dynamic_channel_identity_coding": "combinadic_rank_fixed_width_v1",
        "total_bits": key_total_bits,
        "bits_per_coordinate": key_total_bits / float(vectors * 128),
    }
    expected_value = {
        "payload_bits": value_payload_bits,
        "fp16_scale_bits": value_scale_bits,
        "scale_fields_per_vector": 1,
        "total_bits": value_total_bits,
        "bits_per_coordinate": value_total_bits / float(vectors * 128),
    }
    if key != expected_key or value != expected_value:
        raise AggregateError(f"BlockQuant role bit arithmetic changed for {label}")
    expected_total = key_total_bits + value_total_bits
    if accounting.get("total_bits") != expected_total or stored_bits != expected_total:
        raise AggregateError(f"BlockQuant total bit arithmetic changed for {label}")
    if accounting.get("excluded_global_state") != [
        "dense Haar rotations",
        "data-free codebooks",
        "Python control/container bytes",
        "allocator and unpacked research tensor bytes",
    ]:
        raise AggregateError(f"BlockQuant excluded-state contract changed for {label}")
    return accounting


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise AggregateError(f"{label} must be boolean, got {value!r}")
    return value


def _mcnemar_exact_two_sided(hq_only: int, turbo_only: int) -> float:
    """Exact two-sided McNemar p-value under a Binomial(n, 1/2) null."""
    discordant = hq_only + turbo_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(hq_only, turbo_only) + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def _paired_binary_comparison(
    hq_rows: Sequence[dict], turbo_rows: Sequence[dict], *, metric: str, label: str
) -> dict:
    """Compare two methods on identical examples, retaining pairing.

    The confidence interval is the normal interval for the mean of the paired
    {-1, 0, +1} difference vector.  The exact McNemar test is also reported so
    a small paper delta cannot be presented as an independent-sample gain.
    """
    if metric not in ("correct", "correct_norm"):
        raise AggregateError(f"Unsupported paired metric {metric!r} for {label}")
    if len(hq_rows) != len(turbo_rows) or not hq_rows:
        raise AggregateError(f"Paired example count mismatch for {label}")
    differences: list[int] = []
    hq_only = turbo_only = both_correct = both_wrong = 0
    for hq, turbo in zip(hq_rows, turbo_rows):
        if hq["index"] != turbo["index"] or hq["gold"] != turbo["gold"]:
            raise AggregateError(f"Example identity/gold mismatch for paired {label}")
        hq_correct = int(hq[metric])
        turbo_correct = int(turbo[metric])
        differences.append(hq_correct - turbo_correct)
        if hq_correct and turbo_correct:
            both_correct += 1
        elif hq_correct:
            hq_only += 1
        elif turbo_correct:
            turbo_only += 1
        else:
            both_wrong += 1
    n = len(differences)
    delta = math.fsum(differences) / n
    if n > 1:
        variance = math.fsum((value - delta) ** 2 for value in differences) / (n - 1)
        stderr = math.sqrt(variance / n)
    else:
        stderr = 0.0
    low = max(-1.0, delta - 1.959963984540054 * stderr)
    high = min(1.0, delta + 1.959963984540054 * stderr)
    return {
        "metric": metric,
        "statistical_unit": "example_within_fixed_task_and_checkpoint",
        "task_and_checkpoint_status": "fixed_not_randomly_sampled",
        "population_generalization": False,
        "n_paired": n,
        "hq_correct": sum(int(row[metric]) for row in hq_rows),
        "turboquant_correct": sum(int(row[metric]) for row in turbo_rows),
        "delta_accuracy_hq_minus_turboquant": round(delta, 10),
        "delta_percentage_points_hq_minus_turboquant": round(100.0 * delta, 8),
        "paired_standard_error": round(stderr, 10),
        "paired_normal_ci95": [round(low, 10), round(high, 10)],
        "interval_status": "nominal_per_task_not_an_across_rate_familywise_interval",
        "discordant": {"hq_only_correct": hq_only, "turboquant_only_correct": turbo_only},
        "concordant": {"both_correct": both_correct, "both_wrong": both_wrong},
        "mcnemar_exact_two_sided_p": round(_mcnemar_exact_two_sided(hq_only, turbo_only), 12),
        "claim_rule": "claim_improvement_only_if_ci95_excludes_zero",
    }


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_coverage(shards: Sequence[dict], *, expected_size: int, label: str) -> list[dict]:
    """Return shards sorted by start after proving exact [0, expected_size)."""
    ordered = sorted(shards, key=lambda item: (item["start"], item["stop"]))
    if not ordered:
        raise AggregateError(f"Missing all shards for {label}")
    cursor = 0
    for shard in ordered:
        start = _integer(shard["start"], label=f"{label} start")
        stop = _integer(shard["stop"], label=f"{label} stop", minimum=1)
        n = _integer(shard["n"], label=f"{label} n", minimum=1)
        if stop <= start:
            raise AggregateError(f"Empty/reversed interval [{start}, {stop}) for {label}")
        if n != stop - start:
            raise AggregateError(
                f"Interval/count mismatch for {label}: [{start},{stop}) but n={n}"
            )
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise AggregateError(
                f"Shard coverage {relation} for {label}: expected next start {cursor}, got {start}"
            )
        cursor = stop
    if cursor != expected_size:
        raise AggregateError(
            f"Incomplete shard coverage for {label}: covered [0,{cursor}), expected [0,{expected_size})"
        )
    return ordered


def _zeroshot_protocol_source_record(task: str, row: Mapping[str, Any], index: int) -> dict | None:
    """Independently reconstruct one pinned lm-eval multiple-choice request.

    This intentionally does not import the evaluator under test.  Its transforms
    mirror the task definitions at ``LM_EVAL_PROTOCOL_COMMIT`` and operate on
    the pinned Arrow row read by :func:`_load_pinned_zeroshot_source_rows`.
    The digest scope ends at the exact pre-tokenization strings and lm-eval's
    character-length denominator; tokenizer-specific boundary handling remains
    source/test-audited separately.
    """

    def hellaswag_clean(text: str) -> str:
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    try:
        if task in ("arc_easy", "arc_challenge"):
            choices = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            try:
                gold = labels.index(row["answerKey"])
            except ValueError:
                return None
            context: str | list[str] = f"Question: {row['question']}\nAnswer:"
            continuations = [f" {choice}" for choice in choices]
        elif task == "piqa":
            context = f"Question: {row['goal']}\nAnswer:"
            continuations = [f" {row['sol1']}", f" {row['sol2']}"]
            gold = int(row["label"])
        elif task == "hellaswag":
            body = f"{row['ctx_a']} {str(row['ctx_b']).capitalize()}"
            context = hellaswag_clean(f"{row['activity_label']}: {body}")
            continuations = [f" {hellaswag_clean(value)}" for value in row["endings"]]
            try:
                gold = int(row["label"])
            except (TypeError, ValueError):
                return None
        elif task == "winogrande":
            split = str(row["sentence"]).index("_")
            prefix = str(row["sentence"])[:split]
            suffix = str(row["sentence"])[split + 1 :].strip()
            context = [prefix + str(row["option1"]), prefix + str(row["option2"])]
            continuations = [f" {suffix}", f" {suffix}"]
            try:
                gold = int(row["answer"]) - 1
            except (TypeError, ValueError):
                return None
        else:
            raise AggregateError(f"No pinned lm-eval source adapter for zero-shot task {task!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise AggregateError(f"Malformed pinned source row {index} for {task}") from exc

    if isinstance(context, list):
        choice_char_lengths = [max(1, len(choice)) for choice in context]
    else:
        choice_char_lengths = [
            max(1, len(choice[1:] if choice.startswith(" ") else choice))
            for choice in continuations
        ]
    if not continuations or not 0 <= gold < len(continuations):
        raise AggregateError(
            f"Pinned source row {index} has invalid gold/choice count for {task}: "
            f"gold={gold}, choices={len(continuations)}"
        )
    canonical = {
        "schema": ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA,
        "task": task,
        "index": int(index),
        "context": context,
        "continuations": continuations,
        "gold": int(gold),
        "choice_char_lengths": choice_char_lengths,
    }
    line = (
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return {
        "index": int(index),
        "gold": int(gold),
        "n_choices": len(continuations),
        "choice_char_lengths": choice_char_lengths,
        "canonical_line": line,
    }


def _load_pinned_zeroshot_source_rows(
    input_manifest_path: Path,
    input_manifest_sha256: str,
    task_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict]], dict]:
    """Read exact Arrow files from the paper manifest and rebuild source rows."""
    input_manifest_path = input_manifest_path.expanduser().resolve()
    if not input_manifest_path.is_file():
        raise AggregateError(f"Paper input manifest disappeared: {input_manifest_path}")
    if _sha256(input_manifest_path) != input_manifest_sha256:
        raise AggregateError("Paper input manifest content changed before zero-shot aggregation")
    document = _load_object(input_manifest_path)
    hf_datasets = document.get("hf_datasets")
    dataset_records = hf_datasets.get("datasets") if isinstance(hf_datasets, dict) else None
    if not isinstance(dataset_records, dict) or hf_datasets.get("status") != "validated":
        raise AggregateError("Paper input manifest has no validated pinned HF dataset ledger")

    try:
        from datasets import Dataset, concatenate_datasets
    except ImportError as exc:  # pragma: no cover - formal environment contract
        raise AggregateError(
            "datasets is required to independently audit zero-shot sources"
        ) from exc

    by_task: dict[str, list[dict]] = {}
    audit_tasks = {}
    for task, spec in task_specs.items():
        manifest_key = spec.get("manifest_key")
        if not isinstance(manifest_key, str) or not manifest_key:
            raise AggregateError(f"Zero-shot task {task} has no pinned dataset manifest key")
        record = dataset_records.get(manifest_key)
        if not isinstance(record, dict):
            raise AggregateError(f"Pinned dataset record {manifest_key!r} is missing for {task}")
        expected_metadata = {
            "repo": spec.get("repo"),
            "config": spec.get("config"),
            "split": spec.get("split"),
            "revision": spec.get("revision"),
            "rows": int(spec["size"]),
        }
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise AggregateError(
                f"Pinned dataset metadata changed for {task}/{manifest_key}: {record}"
            )
        cache_files = record.get("cache_files")
        if not isinstance(cache_files, list) or not cache_files:
            raise AggregateError(f"Pinned dataset has no exact Arrow files for {task}")
        datasets = []
        source_files = []
        for file_record in cache_files:
            if not isinstance(file_record, dict):
                raise AggregateError(f"Malformed Arrow ledger entry for {task}")
            raw_path = file_record.get("path")
            if not isinstance(raw_path, str):
                raise AggregateError(f"Pinned Arrow path is missing for {task}")
            path = Path(raw_path).expanduser().resolve()
            if (
                not path.is_file()
                or path.stat().st_size != file_record.get("bytes")
                or _sha256(path) != file_record.get("sha256")
            ):
                raise AggregateError(f"Pinned Arrow file content changed for {task}: {path}")
            datasets.append(Dataset.from_file(str(path)))
            source_files.append(_source(path))
        dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
        if len(dataset) != int(spec["size"]):
            raise AggregateError(
                f"Pinned Arrow row count changed for {task}: {len(dataset)} != {spec['size']}"
            )
        # Every formal shard has a finite limit and therefore uses this exact
        # deterministic shuffled order before applying [start, stop).
        dataset = dataset.shuffle(seed=2027)
        rows = []
        for source_row in dataset:
            rebuilt = _zeroshot_protocol_source_record(task, source_row, len(rows))
            if rebuilt is not None:
                rows.append(rebuilt)
        if len(rows) != int(spec["size"]):
            raise AggregateError(
                f"Pinned {task} produced {len(rows)} valid lm-eval rows, expected {spec['size']}"
            )
        task_digest = hashlib.sha256()
        for row in rows:
            task_digest.update(row["canonical_line"])
        by_task[task] = rows
        audit_tasks[task] = {
            "manifest_key": manifest_key,
            "rows": len(rows),
            "full_split_protocol_input_sha256": task_digest.hexdigest(),
            "source_files": source_files,
        }
    return by_task, {
        "input_manifest": {
            "path": str(input_manifest_path),
            "sha256": input_manifest_sha256,
        },
        "protocol_input_hash_schema": ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA,
        "scope": (
            "pinned source row, exact pre-tokenization context/continuation strings, gold, "
            "choice count, and lm-eval character-length normalization denominator; "
            "tokenizer-specific encoded boundary is not claimed by this digest"
        ),
        "tasks": audit_tasks,
    }


def aggregate_zeroshot(
    root: Path,
    *,
    expected_models: Sequence[str] = PAPER_MODELS,
    task_specs: Mapping[str, Mapping[str, Any]] = ZERO_TASKS,
    rate_methods: Mapping[str, Sequence[str]] = ZERO_RATE_METHODS,
) -> dict:
    root = root.expanduser().resolve()
    paths = sorted(root.rglob("zeroshot_*.json")) if root.is_dir() else []
    if not paths:
        raise AggregateError(f"No zero-shot shard JSON files found below {root}")

    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    seen_sources: list[dict] = []
    code_commits = set()
    input_manifests = set()
    source_rows: dict[str, list[dict]] | None = None
    source_audit: dict | None = None
    source_fields = ("manifest_key", "repo", "config")
    source_metadata = [
        all(field in spec for field in source_fields) for spec in task_specs.values()
    ]
    if any(source_metadata) and not all(source_metadata):
        raise AggregateError("Zero-shot task specs only partially declare pinned source metadata")
    source_audit_required = bool(source_metadata) and all(source_metadata)
    allowed_models = set(expected_models)
    allowed_tasks = set(task_specs)
    for path in paths:
        document = _load_object(path)
        if document.get("schema_version") != 2 or not document.get("run_fingerprint"):
            raise AggregateError(f"Legacy/unidentified zero-shot result: {path}")
        provenance = _require_run_provenance(document, path=path, return_identity=True)
        code_commits.add(provenance["commit"])
        input_manifests.add(
            (provenance["input_manifest_path"], provenance["input_manifest_sha256"])
        )
        if source_audit_required and source_rows is None:
            source_rows, source_audit = _load_pinned_zeroshot_source_rows(
                Path(provenance["input_manifest_path"]),
                provenance["input_manifest_sha256"],
                task_specs,
            )
        model = document.get("model")
        if model not in allowed_models:
            raise AggregateError(f"Unexpected zero-shot model {model!r}: {path}")
        if document.get("sample_seed") != 2027:
            raise AggregateError(f"Wrong zero-shot sample seed in {path}")
        config = document.get("config")
        if not isinstance(config, dict):
            raise AggregateError(f"Missing zero-shot scientific config in {path}")
        if config.get("device") != QUALITY_DEVICE:
            raise AggregateError(f"Zero-shot device is not {QUALITY_DEVICE} in {path}")
        if config.get("dtype") != QUALITY_DTYPE:
            raise AggregateError(f"Zero-shot dtype is not {QUALITY_DTYPE} in {path}")
        if config.get("sample_seed") != 2027:
            raise AggregateError(f"Wrong zero-shot config sample seed in {path}")
        if (
            document.get("start") != config.get("start")
            or document.get("limit") != config.get("limit")
            or document.get("sample_seed") != config.get("sample_seed")
            or document.get("tasks") != config.get("tasks")
        ):
            raise AggregateError(f"Zero-shot result/config selection identity drift in {path}")
        if config.get("skip_outliers") is not False:
            raise AggregateError(
                f"Zero-shot formal row used model-dependent layer skipping: {path}"
            )
        if config.get("sink_tokens") != 0 or config.get("recent_tokens") != 0:
            raise AggregateError(f"Zero-shot formal row retained FP16 token windows: {path}")
        results = document.get("results")
        if not isinstance(results, dict) or not results:
            raise AggregateError(f"Missing zero-shot results in {path}")
        declared_bits = config.get("bits")
        declared_methods = config.get("methods")
        declared_tasks = config.get("tasks")
        if (
            not isinstance(declared_bits, list)
            or {str(value) for value in declared_bits} != set(results)
            or not isinstance(declared_methods, list)
            or not declared_methods
            or not isinstance(declared_tasks, list)
            or not declared_tasks
        ):
            raise AggregateError(f"Zero-shot bits/methods/tasks config is malformed in {path}")
        actual_methods = {
            method
            for methods in results.values()
            if isinstance(methods, dict)
            for method in methods
        }
        actual_tasks = {
            task
            for methods in results.values()
            if isinstance(methods, dict)
            for method_result in methods.values()
            if isinstance(method_result, dict)
            for task in (
                method_result.get("tasks", {})
                if isinstance(method_result.get("tasks"), dict)
                else {}
            )
        }
        if set(declared_methods) != actual_methods or set(declared_tasks) != actual_tasks:
            raise AggregateError(f"Zero-shot declared/actual method or task drift in {path}")
        seen_sources.append(_source(path))
        for raw_rate, methods in results.items():
            if not isinstance(methods, dict):
                raise AggregateError(f"Malformed method map in {path}")
            for method, method_result in methods.items():
                rate = "full" if method == "FullKV" else str(raw_rate)
                if method not in set(rate_methods.get(rate, ())):
                    raise AggregateError(
                        f"Unexpected zero-shot matrix point rate={rate}, method={method}: {path}"
                    )
                if not isinstance(method_result, dict):
                    raise AggregateError(f"Malformed zero-shot method result in {path}")
                recipe_sha256 = _require_formal_codec_recipe(
                    method_result,
                    model=str(model),
                    method=str(method),
                    rate=None if method == "FullKV" else int(raw_rate),
                    path=path,
                )
                task_results = (
                    method_result.get("tasks") if isinstance(method_result, dict) else None
                )
                if not isinstance(task_results, dict) or not task_results:
                    raise AggregateError(f"Missing per-task counts in {path} for {method}")
                for task, result in task_results.items():
                    if task not in allowed_tasks or not isinstance(result, dict):
                        raise AggregateError(
                            f"Unexpected/malformed zero-shot task {task!r}: {path}"
                        )
                    selection = result.get("selection")
                    if not isinstance(selection, dict):
                        raise AggregateError(f"Missing selection metadata for {task}: {path}")
                    start = _integer(
                        selection.get("start_valid_example"),
                        label=f"{path} {task} selection.start_valid_example",
                    )
                    stop = _integer(
                        selection.get("stop_valid_example_exclusive"),
                        label=f"{path} {task} selection.stop_valid_example_exclusive",
                        minimum=1,
                    )
                    n = _integer(result.get("n"), label=f"{path} {task} n", minimum=1)
                    dataset_size = _integer(
                        result.get("dataset_size"), label=f"{path} {task} dataset_size", minimum=1
                    )
                    expected = int(task_specs[task]["size"])
                    if dataset_size != expected:
                        raise AggregateError(
                            f"Dataset size mismatch for {task} in {path}: {dataset_size} != {expected}"
                        )
                    if selection.get("sample_seed") != 2027:
                        raise AggregateError(f"Wrong selection seed for {task} in {path}")
                    if not selection.get("order"):
                        raise AggregateError(
                            f"Missing deterministic selection order for {task} in {path}"
                        )
                    if result.get("split") != task_specs[task]["split"]:
                        raise AggregateError(f"Wrong split for {task} in {path}")
                    if result.get("dataset_revision") != task_specs[task]["revision"]:
                        raise AggregateError(f"Wrong dataset revision for {task} in {path}")
                    expected_protocol, expected_dependency = _expected_teacher_forced_contract(
                        str(method)
                    )
                    if (
                        result.get("evaluator") != "lm-eval-compatible@0.4.11"
                        or result.get("lm_eval_protocol_commit") != LM_EVAL_PROTOCOL_COMMIT
                        or result.get("lm_eval_task_metadata_version")
                        != LM_EVAL_TASK_METADATA_VERSION
                        or result.get("num_fewshot") != 0
                        or result.get("target_delimiter") != " "
                        or result.get("apply_chat_template") is not False
                        or result.get("teacher_forced_scoring_protocol") != expected_protocol
                        or result.get("teacher_forced_temporal_dependency") != expected_dependency
                        or result.get("protocol_probe_cache_lifetime")
                        != ZEROSHOT_PROTOCOL_PROBE_CACHE_LIFETIME
                        or result.get("choice_cache_lifetime")
                        != PER_EXAMPLE_CACHE_LIFETIME
                        or result.get("cross_iteration_tensor_retention")
                        != PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
                    ):
                        raise AggregateError(
                            f"Wrong/unpinned lm-eval zero-shot protocol for {task} in {path}"
                        )
                    protocol_input_sha256 = result.get("protocol_input_sha256")
                    if source_rows is not None:
                        if result.get(
                            "protocol_input_hash_schema"
                        ) != ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA or not _sha256_string(
                            protocol_input_sha256
                        ):
                            raise AggregateError(
                                f"Missing pinned prompt/source digest for {task} in {path}"
                            )
                        expected_digest = hashlib.sha256()
                        for source_row in source_rows[task][start:stop]:
                            expected_digest.update(source_row["canonical_line"])
                        if protocol_input_sha256 != expected_digest.hexdigest():
                            raise AggregateError(
                                f"Pinned prompt/gold/normalization digest mismatch for {task} "
                                f"interval [{start},{stop}) in {path}"
                            )
                    compression_fp16_bits, compression_stored_bits = (
                        _validated_zeroshot_compression(
                            result,
                            label=f"{path} {task}/{method}",
                            compressed=method != "FullKV",
                        )
                    )
                    correct = _integer(result.get("correct"), label=f"{path} {task} correct")
                    correct_norm = _integer(
                        result.get("correct_norm"), label=f"{path} {task} correct_norm"
                    )
                    if correct > n or correct_norm > n:
                        raise AggregateError(f"Correct count exceeds n for {task} in {path}")
                    examples = result.get("per_example")
                    if not isinstance(examples, list) or len(examples) != n:
                        raise AggregateError(
                            f"Missing exact zero-shot per-example predictions for {task}: {path}"
                        )
                    validated_examples = []
                    for expected_index, example in zip(range(start, stop), examples):
                        if not isinstance(example, dict) or example.get("index") != expected_index:
                            raise AggregateError(
                                f"Zero-shot example indexes are not exact [{start},{stop}) "
                                f"for {task}: {path}"
                            )
                        gold = _integer(
                            example.get("gold"),
                            label=f"{path} {task} example {expected_index} gold",
                        )
                        prediction = _integer(
                            example.get("prediction"),
                            label=f"{path} {task} example {expected_index} prediction",
                        )
                        prediction_norm = _integer(
                            example.get("prediction_norm"),
                            label=f"{path} {task} example {expected_index} prediction_norm",
                        )
                        if source_rows is not None:
                            source_row = source_rows[task][expected_index]
                            if gold != source_row["gold"]:
                                raise AggregateError(
                                    f"Pinned source gold mismatch for {task} example "
                                    f"{expected_index}: {path}"
                                )
                            if not 0 <= prediction < source_row["n_choices"]:
                                raise AggregateError(
                                    f"Zero-shot prediction is outside pinned choices for {task} "
                                    f"example {expected_index}: {path}"
                                )
                            if not 0 <= prediction_norm < source_row["n_choices"]:
                                raise AggregateError(
                                    f"Normalized zero-shot prediction is outside pinned choices "
                                    f"for {task} example {expected_index}: {path}"
                                )
                        is_correct = _boolean(
                            example.get("correct"),
                            label=f"{path} {task} example {expected_index} correct",
                        )
                        is_correct_norm = _boolean(
                            example.get("correct_norm"),
                            label=f"{path} {task} example {expected_index} correct_norm",
                        )
                        if is_correct != (prediction == gold) or is_correct_norm != (
                            prediction_norm == gold
                        ):
                            raise AggregateError(
                                f"Zero-shot prediction/correctness mismatch for {task} "
                                f"example {expected_index}: {path}"
                            )
                        validated_examples.append(
                            {
                                "index": expected_index,
                                "gold": gold,
                                "prediction": prediction,
                                "prediction_norm": prediction_norm,
                                "correct": is_correct,
                                "correct_norm": is_correct_norm,
                            }
                        )
                    if sum(int(row["correct"]) for row in validated_examples) != correct:
                        raise AggregateError(
                            f"Zero-shot correct count/audit mismatch for {task}: {path}"
                        )
                    if sum(int(row["correct_norm"]) for row in validated_examples) != correct_norm:
                        raise AggregateError(
                            f"Zero-shot normalized count/audit mismatch for {task}: {path}"
                        )
                    groups.setdefault((model, rate, method, task), []).append(
                        {
                            "start": start,
                            "stop": stop,
                            "n": n,
                            "correct": correct,
                            "correct_norm": correct_norm,
                            "examples": validated_examples,
                            "order": selection["order"],
                            "recipe_sha256": recipe_sha256,
                            "protocol_input_sha256": protocol_input_sha256,
                            "compression_fp16_bits": compression_fp16_bits,
                            "compression_stored_bits": compression_stored_bits,
                            "teacher_forced_scoring_protocol": expected_protocol,
                            "teacher_forced_temporal_dependency": expected_dependency,
                            "source": str(path.resolve()),
                        }
                    )

    aggregate_models: dict[str, dict] = {}
    expected_keys = {
        (model, rate, method, task)
        for model in expected_models
        for rate, methods in rate_methods.items()
        for method in methods
        for task in task_specs
    }
    actual_keys = set(groups)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise AggregateError(
            f"Zero-shot matrix mismatch: missing={missing[:8]} ({len(missing)} total), "
            f"extra={extra[:8]} ({len(extra)} total)"
        )
    if len(code_commits) != 1:
        raise AggregateError(f"Zero-shot shards span multiple commits: {sorted(code_commits)}")
    if len(input_manifests) != 1:
        raise AggregateError("Zero-shot shards span multiple paper input manifests")

    example_vectors: dict[tuple[str, str, str, str], list[dict]] = {}
    # Keep the unrounded checkpoint-level estimands for the across-checkpoint
    # interval.  The rounded values written below are presentation fields and
    # must never become inputs to a later significance calculation.
    checkpoint_macro_deltas: dict[tuple[str, str, str], float] = {}
    for model in expected_models:
        model_doc = {"rates": {}}
        for rate, methods in rate_methods.items():
            rate_doc = {}
            for method in methods:
                task_docs = {}
                for task, spec in task_specs.items():
                    label = f"zero-shot/{model}/{rate}/{method}/{task}"
                    ordered = _validate_coverage(
                        groups[(model, rate, method, task)],
                        expected_size=int(spec["size"]),
                        label=label,
                    )
                    orders = {str(item["order"]) for item in ordered}
                    recipe_hashes = {item["recipe_sha256"] for item in ordered}
                    protocols = {
                        (
                            item["teacher_forced_scoring_protocol"],
                            item["teacher_forced_temporal_dependency"],
                        )
                        for item in ordered
                    }
                    if len(orders) != 1:
                        raise AggregateError(f"Selection order changed across {label} shards")
                    if len(recipe_hashes) != 1:
                        raise AggregateError(f"Codec recipe changed across {label} shards")
                    if len(protocols) != 1:
                        raise AggregateError(
                            f"Teacher-forced protocol changed across {label} shards"
                        )
                    n = sum(int(item["n"]) for item in ordered)
                    correct = sum(int(item["correct"]) for item in ordered)
                    correct_norm = sum(int(item["correct_norm"]) for item in ordered)
                    examples = [example for item in ordered for example in item["examples"]]
                    if [row["index"] for row in examples] != list(range(int(spec["size"]))):
                        raise AggregateError(
                            f"Zero-shot audit indexes changed across {label} shards"
                        )
                    example_vectors[(model, rate, method, task)] = examples
                    acc = correct / n
                    acc_norm = correct_norm / n
                    if method == "FullKV":
                        compression_fp16_bits = None
                        compression_stored_bits = None
                        compression_ratio = 1.0
                        compression_aggregation = "fullkv_no_cache"
                    else:
                        compression_fp16_bits = sum(
                            int(item["compression_fp16_bits"]) for item in ordered
                        )
                        compression_stored_bits = sum(
                            int(item["compression_stored_bits"]) for item in ordered
                        )
                        compression_ratio = compression_fp16_bits / compression_stored_bits
                        compression_aggregation = "exact_cache_bit_counters"
                    primary_name = str(spec["primary"])
                    task_docs[task] = {
                        "n": n,
                        "correct": correct,
                        "correct_norm": correct_norm,
                        "acc": round(acc, 8),
                        "acc_norm": round(acc_norm, 8),
                        "primary_metric": primary_name,
                        "primary_score": round(acc_norm if primary_name == "acc_norm" else acc, 8),
                        "teacher_forced_scoring_protocol": next(iter(protocols))[0],
                        "teacher_forced_temporal_dependency": next(iter(protocols))[1],
                        "compression_ratio": compression_ratio,
                        "compression_aggregation": compression_aggregation,
                        "compression_fp16_bits": compression_fp16_bits,
                        "compression_stored_bits": compression_stored_bits,
                        "dataset_revision": spec["revision"],
                        "split": spec["split"],
                        "selection_order": next(iter(orders)),
                        "recipe_sha256": next(iter(recipe_hashes)),
                        "protocol_input_hash_schema": (
                            ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA
                            if source_rows is not None
                            else None
                        ),
                        "protocol_input_sha256": (
                            source_audit["tasks"][task]["full_split_protocol_input_sha256"]
                            if source_audit is not None
                            else None
                        ),
                        "coverage": [0, n],
                        "audit_records_validated": len(examples),
                        "shards": [item["source"] for item in ordered],
                    }
                if method == "FullKV":
                    method_fp16_bits = None
                    method_stored_bits = None
                    method_compression_ratio = 1.0
                    method_compression_aggregation = "fullkv_no_cache"
                else:
                    method_fp16_bits = sum(
                        int(row["compression_fp16_bits"]) for row in task_docs.values()
                    )
                    method_stored_bits = sum(
                        int(row["compression_stored_bits"]) for row in task_docs.values()
                    )
                    method_compression_ratio = method_fp16_bits / method_stored_bits
                    method_compression_aggregation = "exact_cache_bit_counters"
                rate_doc[method] = {
                    "tasks": task_docs,
                    "compression_ratio": method_compression_ratio,
                    "compression_aggregation": method_compression_aggregation,
                    "compression_fp16_bits": method_fp16_bits,
                    "compression_stored_bits": method_stored_bits,
                    "avg_acc": round(
                        math.fsum(row["correct"] / row["n"] for row in task_docs.values())
                        / len(task_docs),
                        8,
                    ),
                    "avg_acc_norm": round(
                        math.fsum(row["correct_norm"] / row["n"] for row in task_docs.values())
                        / len(task_docs),
                        8,
                    ),
                    "avg_primary_score": round(
                        math.fsum(
                            (
                                row["correct_norm"] / row["n"]
                                if row["primary_metric"] == "acc_norm"
                                else row["correct"] / row["n"]
                            )
                            for row in task_docs.values()
                        )
                        / len(task_docs),
                        8,
                    ),
                    "n_total": sum(row["n"] for row in task_docs.values()),
                    "official_complete": True,
                }
                if rate in {"2", "3", "4"}:
                    tq_doc = rate_doc["TurboQuant-paper"]
                    for matched_method in ("FibQuant-paper", *PROPOSAL_METHODS):
                        if matched_method not in rate_doc:
                            # Custom/unit-test matrices may deliberately select a
                            # strict subset.  The default formal matrix still
                            # requires every method through ``expected_keys``.
                            continue
                        matched_doc = rate_doc[matched_method]
                        for task in task_specs:
                            tq_task = tq_doc["tasks"][task]
                            matched_task = matched_doc["tasks"][task]
                            if (
                                matched_task["compression_fp16_bits"]
                                != tq_task["compression_fp16_bits"]
                                or matched_task["compression_stored_bits"]
                                != tq_task["compression_stored_bits"]
                            ):
                                raise AggregateError(
                                    "Zero-shot exact payload totals are not matched for "
                                    f"{model}/{rate}/{task}/{matched_method} versus "
                                    "TurboQuant-paper"
                                )
                        if (
                            matched_doc["compression_fp16_bits"] != tq_doc["compression_fp16_bits"]
                            or matched_doc["compression_stored_bits"]
                            != tq_doc["compression_stored_bits"]
                        ):
                            raise AggregateError(
                                "Zero-shot five-task exact payload totals are not matched for "
                                f"{model}/{rate}/{matched_method} versus TurboQuant-paper"
                            )
                        matched_doc["payload_match_vs_turboquant"] = {
                            "exact_integer_counters": True,
                            "compression_fp16_bits": matched_doc["compression_fp16_bits"],
                            "compression_stored_bits": matched_doc["compression_stored_bits"],
                        }
            model_doc["rates"][rate] = rate_doc
        model_doc["official_complete"] = True
        aggregate_models[model] = model_doc

    # Paper comparisons are paired within every released example.  This is
    # deliberately added only after exact coverage has been established for
    # the complete matrix and the gold labels have been cross-checked below.
    comparison_rates_by_proposal = {
        proposal: tuple(
            rate
            for rate in ("2", "3", "4")
            if "TurboQuant-paper" in set(rate_methods.get(rate, ()))
            and proposal in set(rate_methods.get(rate, ()))
        )
        for proposal in PROPOSAL_METHODS
    }
    for proposal in PROPOSAL_METHODS:
        for model in expected_models:
            for rate in comparison_rates_by_proposal[proposal]:
                task_comparisons = {}
                for task, spec in task_specs.items():
                    metric = "correct_norm" if spec["primary"] == "acc_norm" else "correct"
                    comparison = _paired_binary_comparison(
                        example_vectors[(model, rate, proposal, task)],
                        example_vectors[(model, rate, "TurboQuant-paper", task)],
                        metric=metric,
                        label=f"zero-shot/{model}/{rate}/{proposal}/{task}",
                    )
                    task_comparisons[task] = comparison
                    aggregate_models[model]["rates"][rate][proposal]["tasks"][task][
                        "paired_vs_turboquant"
                    ] = comparison
                exact_task_deltas = []
                exact_task_stderr = []
                for comparison in task_comparisons.values():
                    n = int(comparison["n_paired"])
                    proposal_only = int(comparison["discordant"]["hq_only_correct"])
                    turbo_only = int(comparison["discordant"]["turboquant_only_correct"])
                    delta = (proposal_only - turbo_only) / n
                    variance = (
                        ((proposal_only + turbo_only) - n * delta * delta) / (n - 1)
                        if n > 1
                        else 0.0
                    )
                    exact_task_deltas.append(delta)
                    exact_task_stderr.append(math.sqrt(max(0.0, variance / n)))
                macro_delta = math.fsum(exact_task_deltas) / len(exact_task_deltas)
                macro_stderr = math.sqrt(
                    math.fsum(value**2 for value in exact_task_stderr)
                ) / len(exact_task_stderr)
                checkpoint_macro_deltas[(proposal, model, rate)] = macro_delta
                aggregate_models[model]["rates"][rate].setdefault(
                    "paired_proposals_vs_turboquant", {}
                )[proposal] = {
                    "proposal": proposal,
                    "estimand": "unweighted five-task macro of each task primary metric",
                    "statistical_unit": "paired_example_within_each_fixed_task_and_checkpoint",
                    "task_panel": "five_fixed_tasks_not_a_random_task_population",
                    "population_generalization": False,
                    "delta_macro_proposal_minus_turboquant": round(macro_delta, 10),
                    "delta_macro_percentage_points_proposal_minus_turboquant": round(
                        100.0 * macro_delta, 8
                    ),
                    "stratified_paired_standard_error": round(macro_stderr, 10),
                    "stratified_paired_normal_ci95": [
                        round(max(-1.0, macro_delta - 1.959963984540054 * macro_stderr), 10),
                        round(min(1.0, macro_delta + 1.959963984540054 * macro_stderr), 10),
                    ],
                    "interval_status": "secondary_robustness_not_primary_win_gate",
                    "per_task": task_comparisons,
                    "claim_rule": "interval_is_secondary_robustness_only",
                }

    secondary_intervals = {}
    for proposal in PROPOSAL_METHODS:
        rate_rows = {}
        for rate in comparison_rates_by_proposal[proposal]:
            values = [
                checkpoint_macro_deltas[(proposal, model, rate)] for model in expected_models
            ]
            n_checkpoints = len(values)
            mean_delta = math.fsum(values) / n_checkpoints
            stderr = (
                math.sqrt(
                    math.fsum((value - mean_delta) ** 2 for value in values)
                    / (n_checkpoints - 1)
                    / n_checkpoints
                )
                if n_checkpoints > 1
                else 0.0
            )
            nominal_critical = 2.4469118511449692 if n_checkpoints == 7 else 1.959963984540054
            simultaneous_critical = 3.287455157088189 if n_checkpoints == 7 else 1.959963984540054
            rate_rows[rate] = {
                "proposal": proposal,
                "estimand": "mean_fixed_checkpoint_five_task_macro_paired_delta",
                "student_t_statistical_unit": "fixed_checkpoint",
                "checkpoint_panel": "seven_preselected_family_correlated_checkpoints_not_iid",
                "population_generalization": False,
                "n_checkpoints": n_checkpoints,
                "mean_delta": round(mean_delta, 12),
                "mean_delta_standard_error": round(stderr, 12),
                "nominal_student_t_ci95": [
                    round(max(-1.0, mean_delta - nominal_critical * stderr), 12),
                    round(min(1.0, mean_delta + nominal_critical * stderr), 12),
                ],
                "nominal_interval_status": "secondary_robustness_not_primary_win_gate",
                "bonferroni_simultaneous_familywise95_ci": [
                    round(max(-1.0, mean_delta - simultaneous_critical * stderr), 12),
                    round(min(1.0, mean_delta + simultaneous_critical * stderr), 12),
                ],
                "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
                "familywise_alpha": 0.05,
                "per_rate_two_sided_alpha": 0.05 / 3.0,
                "claim_rule": "interval_is_secondary_robustness_only",
            }
        secondary_intervals[proposal] = rate_rows

    return {
        "benchmark": "corrected_zero_shot_five_task",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "schema_version": 1,
        "aggregation": (
            "sum_exact_counts_after_exact_interval_coverage_with_paired_"
            "example_audit_and_exact_mcnemar"
        ),
        "inference_scope": {
            "within_checkpoint_paired_unit": ("example_within_each_fixed_task_and_checkpoint"),
            "task_panel": "five_fixed_tasks_not_a_random_task_population",
            "cross_checkpoint_student_t_unit": "fixed_checkpoint",
            "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
            "population_generalization": False,
        },
        "teacher_forced_scoring_contract": {
            "protocol": PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": False,
            "method_overrides": {
                "BlockQuant-paper": {
                    "protocol": CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": BLOCKQUANT_TEMPORAL_DEPENDENCY,
                },
                "AdaptiveHQ": {
                    "protocol": CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": ADAPTIVE_HQ_TEMPORAL_DEPENDENCY,
                },
            },
        },
        "compression_accounting": dict(ZEROSHOT_COMPRESSION_ACCOUNTING),
        "pinned_source_prompt_audit": source_audit,
        "sample_seed": 2027,
        "commit": next(iter(code_commits)),
        "models": aggregate_models,
        "secondary_interval_robustness_vs_turboquant": secondary_intervals,
        "official_complete": True,
        "sources": seen_sources,
    }


def _wilson(correct: int, n: int) -> tuple[float, float, float]:
    value = correct / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    centre = (value + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(value * (1.0 - value) / n + z * z / (4.0 * n * n)) / denominator
    return math.sqrt(value * (1.0 - value) / n), max(0.0, centre - half), min(1.0, centre + half)


def _longcodeqa_letter(text: str) -> str | None:
    match = _LONGCODE_FINAL_ANSWER.search(text)
    if match:
        return match.group(1).upper()
    match = _LONGCODE_TRAILING_LETTER.search(text.strip())
    return match.group(1).upper() if match else None


def _longcodeqa_preflight_file(
    document: Mapping[str, Any],
    *,
    expected_path: Path,
    label: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict:
    """Require every embedded preflight to name the same immutable file."""
    expected_path = expected_path.expanduser().resolve()
    identities = set()
    for envelope in document["run_provenance"]:
        record = envelope["record"]
        matches = []
        for raw in record["dataset_files"]:
            if Path(raw["path"]).expanduser().resolve() == expected_path:
                matches.append(raw)
        if len(matches) != 1:
            raise AggregateError(
                f"LongCodeQA preflight must contain exactly one {label} record: "
                f"{expected_path}"
            )
        raw = matches[0]
        identities.add((str(expected_path), raw["bytes"], str(raw["sha256"]).lower()))
    if len(identities) != 1:
        raise AggregateError(f"LongCodeQA preflights disagree on {label}: {expected_path}")
    path, size, digest = next(iter(identities))
    if expected_bytes is not None and size != expected_bytes:
        raise AggregateError(f"LongCodeQA {label} byte size differs from the input manifest")
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise AggregateError(f"LongCodeQA {label} SHA-256 differs from the input manifest")
    return {"path": path, "bytes": size, "sha256": digest}


def _longcodeqa_json_file(
    record: Mapping[str, Any],
    *,
    label: str,
    cache: dict[tuple[str, int, str], tuple[Path, Any]],
) -> tuple[Path, Any]:
    """Hash and parse one pinned LongCodeQA artifact at most once per aggregate."""
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    size = record.get("bytes")
    digest = str(record.get("sha256", "")).lower()
    key = (str(path), size, digest)
    if key in cache:
        return cache[key]
    validated = _validated_declared_file(dict(record), label=label)
    try:
        with validated.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateError(f"Cannot parse pinned {label} JSON {validated}: {exc}") from exc
    cache[key] = (validated, value)
    return cache[key]


def _longcodeqa_source_contract(
    document: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    model: str,
    bracket: str,
    expected_size: int,
    expected_context_tokens: int,
    expected_dataset_sha256: str,
    expected_brackets: Sequence[str],
    expected_eos_token_ids: Sequence[int],
    result_dataset: Mapping[str, Any],
    json_cache: dict[tuple[str, int, str], tuple[Path, Any]],
) -> dict:
    """Reopen the pinned source rows and tokenizer context-fit proof.

    Result shards intentionally do not copy the released, potentially enormous
    prompt.  Therefore this audit compares only source fields actually retained
    by the shard (gold, repository, and question), while the prompt's token
    length is checked against the separately content-hashed context manifest.
    """
    input_path = Path(provenance["input_manifest_path"]).expanduser().resolve()
    input_record = _longcodeqa_preflight_file(
        document,
        expected_path=input_path,
        label="paper input manifest",
        expected_sha256=str(provenance["input_manifest_sha256"]),
    )
    _, paper_inputs = _longcodeqa_json_file(
        input_record,
        label="paper input manifest",
        cache=json_cache,
    )
    if (
        not isinstance(paper_inputs, dict)
        or paper_inputs.get("schema_version") != 2
        or paper_inputs.get("kind") != "hayatequant_paper_input_validation"
        or paper_inputs.get("status") != "ok"
    ):
        raise AggregateError(f"LongCodeQA paper input manifest is not validated: {input_path}")
    longcode = paper_inputs.get("longcodeqa")
    if (
        not isinstance(longcode, dict)
        or longcode.get("dataset") != "LongCodeBench/LongCodeQA"
        or longcode.get("release_revision") != LONGCODE_REVISION
        or longcode.get("release_clarification") != LONGCODE_DISCUSSION_URL
        or longcode.get("parser_reference_commit") != LONGCODE_PROTOCOL_COMMIT
        or longcode.get("protocol_scope") != LONGCODE_INPUT_PROTOCOL_SCOPE
        or longcode.get("official_reference_decoding_exact") is not False
    ):
        raise AggregateError(f"Pinned LongCodeQA input contract drifted in {input_path}")

    bracket_records = longcode.get("brackets")
    source_record = bracket_records.get(bracket) if isinstance(bracket_records, dict) else None
    if (
        not isinstance(source_record, dict)
        or source_record.get("rows") != expected_size
        or source_record.get("sha256") != expected_dataset_sha256
        or isinstance(source_record.get("bytes"), bool)
        or not isinstance(source_record.get("bytes"), int)
        or source_record["bytes"] <= 0
    ):
        raise AggregateError(
            f"Pinned LongCodeQA source record changed for {model}/{bracket} in {input_path}"
        )
    source_path_raw = source_record.get("path")
    result_source_path = result_dataset.get("path")
    if (
        not isinstance(source_path_raw, str)
        or not Path(source_path_raw).is_absolute()
        or not isinstance(result_source_path, str)
        or not Path(result_source_path).is_absolute()
        or Path(result_source_path).expanduser().resolve()
        != Path(source_path_raw).expanduser().resolve()
    ):
        raise AggregateError(
            f"LongCodeQA result does not identify the pinned source file for {model}/{bracket}"
        )
    source_preflight = _longcodeqa_preflight_file(
        document,
        expected_path=Path(source_path_raw),
        label=f"{bracket} source dataset",
        expected_bytes=source_record.get("bytes"),
        expected_sha256=expected_dataset_sha256,
    )
    source_path, source_rows = _longcodeqa_json_file(
        source_preflight,
        label=f"LongCodeQA {bracket} source dataset",
        cache=json_cache,
    )
    if not isinstance(source_rows, list) or len(source_rows) != expected_size:
        raise AggregateError(
            f"Pinned LongCodeQA {bracket} source must contain {expected_size} rows"
        )
    for index, row in enumerate(source_rows):
        if (
            not isinstance(row, dict)
            or row.get("correct_letter") not in {"A", "B", "C", "D"}
            or not isinstance(row.get("repo"), str)
            or not row["repo"].strip()
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("prompt"), str)
            or not row["prompt"].strip()
        ):
            raise AggregateError(f"Malformed pinned LongCodeQA {bracket} source row {index}")

    model_key = MODEL_MANIFEST_KEYS.get(model, model)
    context_records = longcode.get("context_fit_manifests")
    context_summary = context_records.get(model_key) if isinstance(context_records, dict) else None
    manifest_record = (
        context_summary.get("manifest") if isinstance(context_summary, dict) else None
    )
    if (
        not isinstance(manifest_record, dict)
        or isinstance(manifest_record.get("bytes"), bool)
        or not isinstance(manifest_record.get("bytes"), int)
        or manifest_record["bytes"] <= 0
        or not _sha256_string(manifest_record.get("sha256"))
    ):
        raise AggregateError(
            f"Pinned paper inputs omit the {model_key} LongCodeQA context manifest"
        )
    context_path_raw = manifest_record.get("path")
    if not isinstance(context_path_raw, str) or not Path(context_path_raw).is_absolute():
        raise AggregateError(f"Malformed {model_key} LongCodeQA context-manifest path")
    context_preflight = _longcodeqa_preflight_file(
        document,
        expected_path=Path(context_path_raw),
        label=f"{model_key} context-fit manifest",
        expected_bytes=manifest_record.get("bytes"),
        expected_sha256=manifest_record.get("sha256"),
    )
    context_path, context = _longcodeqa_json_file(
        context_preflight,
        label=f"{model_key} LongCodeQA context-fit manifest",
        cache=json_cache,
    )
    context_model = context.get("model") if isinstance(context, dict) else None
    context_protocol = context.get("protocol") if isinstance(context, dict) else None
    context_brackets = context.get("brackets") if isinstance(context, dict) else None
    expected_generation = {
        "protocol": "hayatequant deterministic greedy paired comparison",
        "do_sample": False,
        "max_new_tokens": 16,
        "seed": None,
        "eos_token_ids_source": "model.generation_config.eos_token_id",
        "stop_conditions": ["any_eos_token_id", "max_new_tokens"],
        "official_reference_max_output_length": None,
        "official_reference_uniform_decoding_config": False,
        "official_reference_decoding_exact": False,
    }
    if (
        not isinstance(context, dict)
        or context.get("schema_version") != 2
        or context.get("benchmark") != "LongCodeBench/LongCodeQA context-fit validation"
        or context.get("all_rows_fit") is not True
        or not isinstance(context_model, dict)
        or not isinstance(context_model.get("path"), str)
        or Path(context_model["path"]).expanduser().resolve()
        != Path(provenance["model_path"]).expanduser().resolve()
        or context_model.get("context_limit") != expected_context_tokens
        or context_model.get("generation_eos_token_ids") != list(expected_eos_token_ids)
        or context_model.get("generation_eos_token_ids_source")
        != "model.generation_config.eos_token_id"
        or not isinstance(context_protocol, dict)
        or context_protocol.get("scope") != LONGCODE_INPUT_PROTOCOL_SCOPE
        or context_protocol.get("dataset_revision") != LONGCODE_REVISION
        or context_protocol.get("prompt") != "released prompt verbatim as raw causal text"
        or context_protocol.get("prompt_release_clarification") != LONGCODE_DISCUSSION_URL
        or context_protocol.get("chat_template") != "none"
        or context_protocol.get("truncation") != "none"
        or context_protocol.get("answer_parser_reference_commit") != LONGCODE_PROTOCOL_COMMIT
        or context_protocol.get("answer_parser") != LONGCODE_ANSWER_PARSER
        or context_protocol.get("reference_evaluator_executable_verbatim") is not False
        or context_protocol.get("reference_evaluator_known_issue")
        != "pinned evaluator stores response but reads prediction text"
        or context_protocol.get("generation") != expected_generation
        or not isinstance(context_brackets, dict)
        or set(context_brackets) != set(expected_brackets)
    ):
        raise AggregateError(
            f"Pinned LongCodeQA context-fit contract drifted for {model} in {context_path}"
        )
    context_bracket = context_brackets.get(bracket)
    if not isinstance(context_bracket, dict) or set(context_bracket) != {
        "rows",
        "dataset_path",
        "dataset_sha256",
        "input_tokens",
        "minimum_margin_tokens",
        "overflow_rows",
        "per_row",
    }:
        raise AggregateError(f"Malformed LongCodeQA context row set for {model}/{bracket}")
    per_row = context_bracket.get("per_row")
    if (
        context_bracket.get("rows") != expected_size
        or context_bracket.get("dataset_sha256") != expected_dataset_sha256
        or not isinstance(context_bracket.get("dataset_path"), str)
        or Path(context_bracket["dataset_path"]).expanduser().resolve() != source_path
        or context_bracket.get("overflow_rows") != []
        or not isinstance(per_row, list)
        or len(per_row) != expected_size
    ):
        raise AggregateError(f"LongCodeQA context-fit dataset mismatch for {model}/{bracket}")
    input_lengths = []
    margins = []
    for index, row in enumerate(per_row):
        if not isinstance(row, dict) or set(row) != {"index", "input_tokens", "margin"}:
            raise AggregateError(f"Malformed context-fit row {model}/{bracket}/{index}")
        input_tokens = _integer(
            row.get("input_tokens"),
            label=f"{model}/{bracket}/{index} context-fit input_tokens",
            minimum=1,
        )
        margin = row.get("margin")
        expected_margin = expected_context_tokens - input_tokens - 16
        if (
            row.get("index") != index
            or isinstance(margin, bool)
            or not isinstance(margin, int)
            or margin != expected_margin
            or margin < 0
        ):
            raise AggregateError(f"Invalid context-fit margin for {model}/{bracket}/{index}")
        input_lengths.append(input_tokens)
        margins.append(margin)
    expected_summary = {
        "min": min(input_lengths),
        "median": statistics.median(input_lengths),
        "max": max(input_lengths),
    }
    summary_without_rows = {
        "rows": expected_size,
        "dataset_path": str(source_path),
        "dataset_sha256": expected_dataset_sha256,
        "input_tokens": expected_summary,
        "minimum_margin_tokens": min(margins),
        "overflow_rows": [],
    }
    paper_brackets = context_summary.get("brackets")
    paper_bracket_summary = (
        paper_brackets.get(bracket) if isinstance(paper_brackets, dict) else None
    )
    if (
        context_bracket.get("input_tokens") != expected_summary
        or context_bracket.get("minimum_margin_tokens") != min(margins)
        or paper_bracket_summary != summary_without_rows
        or context_summary.get("all_rows_fit") is not True
        or context_summary.get("protocol") != context_protocol
    ):
        raise AggregateError(f"LongCodeQA context-fit summary mismatch for {model}/{bracket}")
    return {
        "rows": source_rows,
        "input_tokens": input_lengths,
        "dataset_path": str(source_path),
        "dataset_sha256": expected_dataset_sha256,
        "context_manifest_path": str(context_path),
        "context_manifest_sha256": context_preflight["sha256"],
    }


def aggregate_longcodeqa(
    root: Path,
    *,
    plans: Mapping[str, Mapping[str, Mapping[str, int]]] = LONGCODE_PLANS,
    methods: Sequence[str] = LONG_METHODS,
    expected_sha256: Mapping[str, str] = LONGCODE_SHA256,
    model_eos_token_ids: Mapping[str, Sequence[int]] = LONGCODE_MODEL_EOS_TOKEN_IDS,
) -> dict:
    root = root.expanduser().resolve()
    paths = sorted(root.rglob("results_*.json")) if root.is_dir() else []
    if not paths:
        raise AggregateError(f"No LongCodeQA shard JSON files found below {root}")
    if not set(plans).issubset(model_eos_token_ids):
        raise AggregateError("LongCodeQA model EOS-token contract is incomplete")
    groups: dict[tuple[str, str, str], list[dict]] = {}
    sources: list[dict] = []
    code_commits = set()
    input_manifests = set()
    json_cache: dict[tuple[str, int, str], tuple[Path, Any]] = {}
    for path in paths:
        document = _load_object(path)
        if document.get("schema_version") != 2 or not document.get("run_fingerprint"):
            raise AggregateError(f"Legacy/unidentified long-context result: {path}")
        provenance = _require_run_provenance(document, path=path, return_identity=True)
        code_commits.add(provenance["commit"])
        input_manifests.add(
            (provenance["input_manifest_path"], provenance["input_manifest_sha256"])
        )
        config = document.get("config")
        if not isinstance(config, dict):
            raise AggregateError(f"Missing LongCodeQA scientific config in {path}")
        if config.get("skip_outliers") is not False:
            raise AggregateError(
                f"LongCodeQA formal row used model-dependent layer skipping: {path}"
            )
        if config.get("sink_tokens") != 0 or config.get("recent_tokens") != 0:
            raise AggregateError(f"LongCodeQA formal row retained FP16 token windows: {path}")
        model = document.get("model")
        if model not in plans:
            raise AggregateError(f"Unexpected LongCodeQA model {model!r}: {path}")
        results = document.get("results")
        if not isinstance(results, dict) or not results:
            raise AggregateError(f"Missing method results in {path}")
        sources.append(_source(path))
        for method, method_result in results.items():
            if method not in methods or not isinstance(method_result, dict):
                raise AggregateError(f"Unexpected LongCodeQA method {method!r}: {path}")
            _require_formal_codec_recipe(
                method_result,
                model=str(model),
                method=str(method),
                rate=None if method == "FullKV" else 3,
                path=path,
            )
            aggregate = method_result.get("longcodeqa")
            if not isinstance(aggregate, dict):
                raise AggregateError(f"Missing longcodeqa result for {method}: {path}")
            if (
                aggregate.get("official_primary_reporting") != "per_bracket_exact_match_accuracy"
                or aggregate.get("derived_cross_bracket_summary") is not True
                or aggregate.get("protocol_scope") != LONGCODE_PROTOCOL_SCOPE
                or aggregate.get("official_reference_decoding_exact") is not False
            ):
                raise AggregateError(f"LongCodeQA primary/derived metric labels drifted in {path}")
            bracket_docs = aggregate.get("brackets")
            if not isinstance(bracket_docs, dict) or len(bracket_docs) != 1:
                raise AggregateError(
                    f"Each LongCodeQA shard must contain exactly one bracket: {path}"
                )
            for bracket, result in bracket_docs.items():
                if bracket not in plans[model] or not isinstance(result, dict):
                    raise AggregateError(
                        f"Unexpected LongCodeQA bracket {model}/{bracket}: {path}"
                    )
                plan = plans[model][bracket]
                start = _integer(result.get("start"), label=f"{path} start")
                n = _integer(result.get("n"), label=f"{path} n", minimum=1)
                limit = _integer(result.get("limit"), label=f"{path} limit", minimum=1)
                if limit != n:
                    raise AggregateError(f"LongCodeQA shard limit/n mismatch in {path}")
                correct = _integer(result.get("correct"), label=f"{path} correct")
                if correct > n:
                    raise AggregateError(f"LongCodeQA correct count exceeds n in {path}")
                if result.get("expected_bracket_n") != plan["size"]:
                    raise AggregateError(f"Wrong expected bracket count in {path}")
                if result.get("max_context_tokens") != plan["max_context_tokens"]:
                    raise AggregateError(f"Wrong LongCodeQA context limit in {path}")
                generation = result.get("generation")
                if (
                    result.get("protocol_commit") != LONGCODE_PROTOCOL_COMMIT
                    or result.get("reference_evaluator_executable_verbatim") is not False
                    or result.get("reference_evaluator_known_issue")
                    != "pinned evaluator stores response but reads prediction text"
                    or result.get("protocol_scope") != LONGCODE_PROTOCOL_SCOPE
                    or result.get("prompt_protocol") != "released_prompt_verbatim_raw_causal_text"
                    or result.get("chat_template") != "none"
                    or result.get("truncation") != "none"
                    or result.get("answer_parser") != LONGCODE_ANSWER_PARSER
                    or not isinstance(generation, dict)
                    or generation.get("protocol")
                    != "hayatequant_deterministic_greedy_paired_comparison"
                    or generation.get("do_sample") is not False
                    or generation.get("max_new_tokens") != 16
                    or generation.get("seed") is not None
                    or generation.get("eos_token_ids") != list(model_eos_token_ids[model])
                    or generation.get("eos_token_ids_source")
                    != "model.generation_config.eos_token_id"
                    or generation.get("stop_conditions") != ["any_eos_token_id", "max_new_tokens"]
                    or generation.get("official_reference_max_output_length") is not None
                    or generation.get("official_reference_uniform_decoding_config") is not False
                    or generation.get("official_reference_decoding_exact") is not False
                    or result.get("max_new_tokens") != 16
                    or config.get("longcodeqa_max_new") != 16
                    or result.get("example_cache_lifetime")
                    != PER_EXAMPLE_CACHE_LIFETIME
                    or result.get("cross_iteration_tensor_retention")
                    != PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
                ):
                    raise AggregateError(
                        f"Wrong LongCodeQA prompt/parser/generation protocol in {path}"
                    )
                dataset = result.get("dataset")
                if (
                    not isinstance(dataset, dict)
                    or dataset.get("repo_id") != "Steefano/LCB"
                    or dataset.get("revision") != LONGCODE_REVISION
                    or dataset.get("sha256") != expected_sha256.get(bracket)
                    or dataset.get("final_filtered_release") is not True
                    or dataset.get("release_clarification") != LONGCODE_DISCUSSION_URL
                ):
                    raise AggregateError(f"Wrong LongCodeQA dataset revision in {path}")
                if (
                    result.get("per_example_source_fields")
                    != ["index", "repo", "question", "correct_letter_as_gold"]
                    or result.get("per_example_prompt_retained") is not False
                ):
                    raise AggregateError(
                        f"LongCodeQA retained-source/prompt declaration drifted in {path}"
                    )
                source_contract = _longcodeqa_source_contract(
                    document,
                    provenance=provenance,
                    model=str(model),
                    bracket=str(bracket),
                    expected_size=int(plan["size"]),
                    expected_context_tokens=int(plan["max_context_tokens"]),
                    expected_dataset_sha256=str(expected_sha256[bracket]),
                    expected_brackets=tuple(plans[model]),
                    expected_eos_token_ids=model_eos_token_ids[model],
                    result_dataset=dataset,
                    json_cache=json_cache,
                )
                records = result.get("per_example")
                if not isinstance(records, list) or len(records) != n:
                    raise AggregateError(f"LongCodeQA per_example length mismatch in {path}")
                indexes = [
                    _integer(row.get("index"), label=f"{path} per_example index")
                    for row in records
                    if isinstance(row, dict)
                ]
                if indexes != list(range(start, start + n)):
                    raise AggregateError(
                        f"LongCodeQA example indexes are not exact interval order in {path}"
                    )
                record_correct = 0
                for row in records:
                    index = row["index"]
                    source_row = source_contract["rows"][index]
                    generated_text = row.get("generated_text")
                    prediction = row.get("prediction")
                    gold = row.get("gold")
                    is_correct = row.get("correct")
                    repo = row.get("repo")
                    question = row.get("question")
                    if (
                        not isinstance(generated_text, str)
                        or not isinstance(repo, str)
                        or not repo
                        or not isinstance(question, str)
                        or not question
                        or prediction is not None
                        and prediction not in {"A", "B", "C", "D"}
                        or gold not in {"A", "B", "C", "D"}
                        or not isinstance(is_correct, bool)
                        or _longcodeqa_letter(generated_text) != prediction
                        or is_correct != (prediction == gold)
                    ):
                        raise AggregateError(
                            f"LongCodeQA prediction/parser audit mismatch in {path}"
                        )
                    if (
                        gold != source_row["correct_letter"]
                        or repo != source_row["repo"]
                        or question != source_row["question"]
                    ):
                        raise AggregateError(
                            "LongCodeQA retained fields do not match the pinned source row "
                            f"{bracket}/{index} in {path}"
                        )
                    generated_tokens = _integer(
                        row.get("generated_tokens"),
                        label=f"{path} LongCodeQA generated tokens",
                        minimum=1,
                    )
                    if generated_tokens > 16:
                        raise AggregateError(f"LongCodeQA generation exceeded budget in {path}")
                    input_tokens = _integer(
                        row.get("input_tokens"),
                        label=f"{path} LongCodeQA input tokens",
                        minimum=1,
                    )
                    if input_tokens != source_contract["input_tokens"][index]:
                        raise AggregateError(
                            "LongCodeQA result token length does not match the pinned "
                            f"context-fit row {model}/{bracket}/{index} in {path}"
                        )
                    if input_tokens + 16 > plan["max_context_tokens"]:
                        raise AggregateError(
                            "LongCodeQA prompt plus registered generation budget "
                            f"exceeds model context in {path}"
                        )
                    record_correct += int(is_correct)
                if record_correct != correct:
                    raise AggregateError(f"LongCodeQA exact correct count mismatch in {path}")
                compression = _validated_long_compression_sufficient_statistics(
                    result,
                    records,
                    method=method,
                    label=f"{path} LongCodeQA/{bracket}",
                )
                if aggregate.get("compression_sufficient_statistics") != compression:
                    raise AggregateError(
                        f"LongCodeQA shard-level compression aggregate changed in {path}"
                    )
                groups.setdefault((model, method, bracket), []).append(
                    {
                        "start": start,
                        "stop": start + n,
                        "n": n,
                        "correct": correct,
                        "dataset_sha256": dataset.get("sha256"),
                        "context_manifest_sha256": source_contract["context_manifest_sha256"],
                        "input_tokens": result.get("input_tokens"),
                        **compression,
                        "compression_ledger": [
                            _long_compression_ledger_row(
                                row,
                                sample_id=f"{bracket}:{row['index']}",
                                input_tokens=int(row["input_tokens"]),
                                max_new_tokens=16,
                                generated_tokens=int(row["generated_tokens"]),
                            )
                            for row in records
                        ],
                        "source": str(path.resolve()),
                    }
                )

    expected_keys = {
        (model, method, bracket)
        for model, bracket_plans in plans.items()
        for method in methods
        for bracket in bracket_plans
    }
    if set(groups) != expected_keys:
        missing = sorted(expected_keys - set(groups))
        extra = sorted(set(groups) - expected_keys)
        raise AggregateError(
            f"LongCodeQA matrix mismatch: missing={missing[:8]} ({len(missing)} total), "
            f"extra={extra[:8]} ({len(extra)} total)"
        )
    if len(code_commits) != 1:
        raise AggregateError(f"LongCodeQA shards span multiple commits: {sorted(code_commits)}")
    if len(input_manifests) != 1:
        raise AggregateError("LongCodeQA shards span multiple paper input manifests")

    model_docs = {}
    for model, bracket_plans in plans.items():
        method_docs = {}
        for method in methods:
            bracket_docs = {}
            model_n = 0
            model_correct = 0
            for bracket, plan in bracket_plans.items():
                label = f"LongCodeQA/{model}/{method}/{bracket}"
                ordered = _validate_coverage(
                    groups[(model, method, bracket)],
                    expected_size=int(plan["size"]),
                    label=label,
                )
                hashes = {item["dataset_sha256"] for item in ordered}
                if None in hashes or len(hashes) != 1:
                    raise AggregateError(f"Dataset hash changed across {label} shards")
                context_hashes = {item["context_manifest_sha256"] for item in ordered}
                if len(context_hashes) != 1:
                    raise AggregateError(f"Context-fit manifest changed across {label} shards")
                n = sum(item["n"] for item in ordered)
                correct = sum(item["correct"] for item in ordered)
                stderr, low, high = _wilson(correct, n)
                bracket_docs[bracket] = {
                    "n": n,
                    "correct": correct,
                    "accuracy": round(correct / n, 8),
                    "accuracy_stderr": round(stderr, 8),
                    "accuracy_ci95": [round(low, 8), round(high, 8)],
                    "max_context_tokens": plan["max_context_tokens"],
                    "dataset_revision": LONGCODE_REVISION,
                    "dataset_sha256": next(iter(hashes)),
                    "context_fit_manifest_sha256": next(iter(context_hashes)),
                    "coverage": [0, n],
                    "audit_records_validated": n,
                    "released_data_complete_bracket": True,
                    **_sum_long_compression_statistics(
                        ordered,
                        method=method,
                        label=label,
                    ),
                    "compression_ledger": [
                        ledger for item in ordered for ledger in item["compression_ledger"]
                    ],
                    "shards": [item["source"] for item in ordered],
                }
                model_n += n
                model_correct += correct
            stderr, low, high = _wilson(model_correct, model_n)
            method_docs[method] = {
                "primary_reporting": "per_bracket_exact_match_accuracy",
                "derived_summary_metric": "micro_exact_match_accuracy",
                "n": model_n,
                "correct": model_correct,
                "accuracy": round(model_correct / model_n, 8),
                "accuracy_stderr": round(stderr, 8),
                "accuracy_ci95": [round(low, 8), round(high, 8)],
                "brackets": bracket_docs,
                **_sum_long_compression_statistics(
                    list(bracket_docs.values()),
                    method=method,
                    label=f"LongCodeQA/{model}/{method}",
                ),
                "compression_ledger": [
                    ledger
                    for bracket in bracket_plans
                    for ledger in bracket_docs[bracket]["compression_ledger"]
                ],
                "released_data_complete_feasible_plan": True,
            }
        for bracket in bracket_plans:
            _require_long_exact_payload_match(
                {method: method_docs[method]["brackets"][bracket] for method in methods},
                label=f"LongCodeQA/{model}/{bracket}",
            )
        _require_long_exact_payload_match(method_docs, label=f"LongCodeQA/{model}")
        model_docs[model] = {
            "methods": method_docs,
            "released_data_complete_feasible_plan": True,
        }

    return {
        "benchmark": "LongCodeBench/LongCodeQA",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in plans
        },
        "schema_version": 1,
        "aggregation": "sum_exact_counts_after_exact_interval_coverage",
        "primary_reporting": "per_bracket_exact_match_accuracy",
        "cross_bracket_summary": "derived_micro_exact_match_accuracy",
        "scope": "all released brackets that fit each unmodified model context",
        "protocol_scope": LONGCODE_PROTOCOL_SCOPE,
        "official_reference_decoding_exact": False,
        "source_row_audit": dict(LONGCODE_SOURCE_ROW_AUDIT),
        "commit": next(iter(code_commits)),
        "models": model_docs,
        "released_data_complete_feasible_plan": True,
        "sources": sources,
    }


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _lower_sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_weight_record(weights: Any, *, label: str) -> None:
    """Require a complete full-file SHA-256 checkpoint fingerprint."""
    if not isinstance(weights, dict) or weights.get("schema_version") != 1:
        raise AggregateError(f"Missing versioned model weight fingerprint in {label}")
    file_count = _integer(
        weights.get("fingerprinted_file_count"),
        label=f"{label} fingerprinted model file count",
        minimum=1,
    )
    total_bytes = _integer(
        weights.get("fingerprinted_bytes"),
        label=f"{label} fingerprinted model bytes",
        minimum=1,
    )
    layouts = weights.get("layouts")
    selected = weights.get("selected_format")
    if not isinstance(layouts, dict) or selected not in layouts:
        raise AggregateError(f"Malformed selected model weight layout in {label}")
    selected_layout = layouts[selected]
    shards = selected_layout.get("shards") if isinstance(selected_layout, dict) else None
    if not isinstance(shards, list) or len(shards) != file_count:
        raise AggregateError(f"Model weight shard count mismatch in {label}")
    seen_paths = set()
    summed_bytes = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise AggregateError(f"Malformed model weight shard in {label}")
        relative_path = shard.get("relative_path")
        nbytes = shard.get("bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path in seen_paths
            or not _sha256_string(shard.get("sha256"))
        ):
            raise AggregateError(f"Unidentified/duplicate model weight shard in {label}")
        seen_paths.add(relative_path)
        summed_bytes += _integer(nbytes, label=f"{label} model weight shard bytes", minimum=1)
    if summed_bytes != total_bytes:
        raise AggregateError(f"Model weight byte total mismatch in {label}")


def _blockquant_declared_config() -> dict:
    return {
        "name": "blockquant",
        "bits": 3,
        "value_bits": 2,
        "block_size": 3,
        "key_scale_mode_requested": "ub",
        "key_scale_mode": "ub",
        "value_scale_mode_requested": "bsm",
        "value_scale_mode": "bsm",
        "key_outlier_indices": None,
        "key_n_outlier": 32,
        "key_outlier_selection": dict(BLOCKQUANT_SELECTION_COMPLETION),
        "outlier_bits": 4,
        "remainder_policy": "lower_dimensional_tail_v1",
        "assignment_mode": "exact-full-codebook",
        "lut": None,
        "codebook_spec": "onecomp.blockquant.finite_beta_lloyd.v1",
        "rotation_seed": 0,
        "codebook_seed": 0,
        "paper_kv_recipe": True,
        "independent_baseline": True,
        "reference": "arXiv:2605.19972v1",
        "key_axis": "token",
        "value_axis": "token",
        "group_size": -1,
        "symmetric": False,
        "skip_layers": [],
        "residual_window": 128,
    }


def _blockquant_algorithm_provenance() -> dict:
    return {
        "reference": "arXiv:2605.19972v1",
        "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
        "independent_baseline": True,
        "proxy_for_another_codec": False,
        "paper_algorithm": "Block-Sphere Quantization Algorithm 1",
        "assignment": {
            "mode": "exact-full-codebook-nearest-squared-l2",
            "lut": None,
            "reason": "quality baseline; paper does not specify experimental L or k",
        },
        "codebook": {
            "paper_objective": "finite-spherical-block-marginal squared-L2 Lloyd objective",
            "published_centroids_or_training_code": False,
            "reproducibility_completion": "onecomp.blockquant.finite_beta_lloyd.v1",
            "sampling": "scrambled-Sobol exact inverse finite-spherical marginal",
            "target_samples_per_codeword": 512,
            "minimum_training_samples": 65536,
            "maximum_training_samples": 1048576,
            "restarts": 4,
            "lloyd_iterations": 25,
            "initializer": "Beta-quantile radius plus quasi-uniform angle",
            "training_assignment": "exact scipy.spatial.cKDTree query, eps=0, workers=1",
            "empty_cell_rule": "greatest-total-SSE donor, farthest unclaimed sample",
        },
        "remainder": {
            "policy": "lower_dimensional_tail_v1",
            "paper_specified": False,
            "completion": "lower_dimensional_tail_v1",
        },
        "key_outlier_selection": dict(BLOCKQUANT_SELECTION_COMPLETION),
    }


def _blockquant_part(
    ambient_dim: int,
    bits: int,
    role: str,
    scale_mode: str,
) -> dict:
    rotation = {
        "key": (104729, "edf63e9187f43a83bf6fa9ff1b40c22fc43beb9fa0e5a221435a4f4e88ac75a1"),
        "key_outlier": (
            130363,
            "935a79b095d8408209ee69fec4e49cf8bd4c06317bda345bcfcd57d61d3bea40",
        ),
        "value": (155921, "7de45e9d66b09292b1f59571b402f518f7c9d741e279dea2c65314eb9727fce8"),
    }
    full_blocks, remainder = divmod(ambient_dim, 3)
    codebooks = []
    if full_blocks:
        codebooks.append(
            {
                "kind": "full-block",
                "block_dim": 3,
                "codebook_size": 1 << (bits * 3),
                "code_bits": bits * 3,
            }
        )
    if remainder:
        codebooks.append(
            {
                "kind": "lower-dimensional-tail",
                "block_dim": remainder,
                "codebook_size": 1 << (bits * remainder),
                "code_bits": bits * remainder,
            }
        )
    seed, digest = rotation[role]
    return {
        "ambient_dim": ambient_dim,
        "bits_per_coordinate": bits,
        "block_size": 3,
        "full_blocks": full_blocks,
        "remainder": remainder,
        "remainder_policy": "lower_dimensional_tail_v1",
        "scale_mode": scale_mode,
        "assignment": "exact-full-codebook-nearest-squared-l2",
        "codebook_spec": "onecomp.blockquant.finite_beta_lloyd.v1",
        "codebook_seed": 0,
        "codebooks": codebooks,
        "rotation": {
            "kind": "dense-haar-gaussian-qr",
            "ambient_dim": ambient_dim,
            "seed": seed,
            "dtype": "float32",
            "sha256": digest,
        },
    }


def _expected_blockquant_resolved() -> dict:
    return {
        "schema_version": 1,
        "codec": "blockquant",
        "reference": "arXiv:2605.19972v1",
        "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
        "head_dim": 128,
        "paper_kv_recipe": True,
        "paper_reproduction": False,
        "random_seed_protocol": (
            "fixed rotation_seed/codebook_seed; not the paper five-random-seed aggregate"
        ),
        "config": _blockquant_declared_config(),
        "provenance": _blockquant_algorithm_provenance(),
        "roles": {
            "key": {
                "mode": "split",
                "selection": dict(BLOCKQUANT_SELECTION_COMPLETION),
                "outlier": _blockquant_part(32, 4, "key_outlier", "ub"),
                "ordinary": _blockquant_part(96, 3, "key", "ub"),
                "fp16_scale_fields_per_vector": 2,
            },
            "value": {
                "mode": "uniform",
                "part": _blockquant_part(128, 2, "value", "bsm"),
                "fp16_scale_fields_per_vector": 1,
            },
        },
    }


def _blockquant_quality_provenance_contract() -> dict:
    return {
        "method_role": "external_quality_baseline",
        "external_quality_baseline": True,
        "proposal_method": False,
        "paper_comparable": False,
        "paper_reproduction": False,
        "speed_comparison": False,
        "primary_source": "arXiv:2605.19972v1",
        "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
        "formal_rate_slot": 3,
        "paper_declared_key_bits_per_coordinate": 3.5,
        "paper_declared_value_payload_bits_per_coordinate": 2,
        "top32_completion": "per_batch_head_update_top_l2_v1",
        "remainder_completion": "lower_dimensional_tail_v1",
        "codebook_completion": "onecomp.blockquant.finite_beta_lloyd.v1",
        "random_seed_protocol": (
            "fixed rotation_seed=0 and codebook_seed=0 same-runtime quality row; "
            "not the paper five-random-seed aggregate"
        ),
        "method_definition": (
            "BlockQuant v1 exact-search fixed-seed quality baseline at the paper-declared "
            "Llama-3.1 KV point, with versioned dynamic per-head top-L2 identities and "
            "lower-dimensional p=3 tails; identity bits are counted"
        ),
        "nominal_payload_bits_per_coordinate": None,
    }


def _expected_resolved_codec_config(method: str, rate: int, head_dim: int) -> dict:
    """Reconstruct the exact role-wise formal recipe from primitive rules."""
    rate = int(rate)
    head_dim = int(head_dim)

    def _role(
        *,
        bits: int,
        block: int,
        rotation: str,
        scale: str,
        mathematical_scale: str,
        source: str,
    ) -> dict:
        return {
            "bits": int(bits),
            "block_size": int(block),
            "rotation_mode": rotation,
            "rotation_seed": 0,
            "scale_mode": scale,
            "mathematical_scale_mode": mathematical_scale,
            "source_law": source,
            "quantized_dim": head_dim,
            "outlier_channels": 0,
            "outlier_bits": None,
            "outlier_selection": "none",
            "qjl_enabled": False,
        }

    if method == "TurboQuant-paper":
        role = _role(
            bits=rate,
            block=1,
            rotation="dense",
            scale="l2",
            mathematical_scale="l2",
            source="beta",
        )
        return {
            "schema_version": 1,
            "head_dim": head_dim,
            "codec": "turboquant",
            "calibration": None,
            "roles": {"key": dict(role), "value": dict(role)},
        }
    if method == "FibQuant-paper":
        if head_dim != 128:
            raise AggregateError(
                "The frozen formal FibQuant digest contract is defined for head_dim=128"
            )
        try:
            block_size, code_bits, codebook_sha256 = FIBQUANT_PAPER_RECIPES[rate]
        except KeyError as exc:
            raise AggregateError(f"No formal FibQuant recipe at {rate} bpc") from exc
        n_codewords = 1 << code_bits
        role = {
            "payload_bits_per_coordinate": code_bits / float(block_size),
            "stored_bits_per_coordinate_including_norm": code_bits / float(block_size)
            + 16.0 / float(head_dim),
            "norm_header": "signed-fp16-per-vector",
            "block_size": block_size,
            "code_bits": code_bits,
            "codebook_size": n_codewords,
            "encoder": "exact-nearest-squared-l2",
            "source_law": "finite-spherical-beta-f_d_k",
            "normalization": "l2-before-rotation",
            "rotation": "dense-haar-gaussian-qr",
            "rotation_seed": 0,
            "codebook_seed": 0,
            "codebook_dtype": "float32",
            "codebook_sha256": codebook_sha256,
        }
        return {
            "schema_version": 1,
            "codec": "fibquant",
            "reference": "arXiv:2605.11478v1",
            "head_dim": head_dim,
            "calibration": None,
            "fixed_rate": True,
            "shared_codebook": True,
            "shared_rotation": True,
            "rotation_sha256": FIBQUANT_D128_ROTATION_SHA256,
            "paper_recipe": True,
            "codebook_training": {
                "source": "exact-spherical-beta-via-normalized-gaussian",
                "samples_per_codeword": 30,
                "sample_count": 30 * n_codewords,
                "restarts": 4,
                "lloyd_iterations": 25,
                "initializer": "beta-quantile-fibonacci-roberts",
                "empty_cell_repair": "max-total-sse-donor-farthest-unclaimed-sample",
            },
            "roles": {"key": dict(role), "value": dict(role)},
        }
    if method == "BlockQuant-paper":
        if rate != 3 or head_dim != 128:
            raise AggregateError(
                "BlockQuant-paper is defined only at formal rate slot 3 and head_dim=128"
            )
        return _expected_blockquant_resolved()
    if method in {"HayateQuant-scalar-matched", "HayateQuant-blockvq-matched"}:
        block = 1
        if method == "HayateQuant-blockvq-matched":
            for candidate in (8, 4, 2, 1):
                if rate * candidate <= 9 and head_dim % candidate == 0:
                    block = candidate
                    break
        role = _role(
            bits=rate,
            block=block,
            rotation="dense",
            scale="rms",
            mathematical_scale="rms",
            source="beta",
        )
        return {
            "schema_version": 1,
            "head_dim": head_dim,
            "codec": "hayatequant",
            "calibration": None,
            "roles": {"key": dict(role), "value": dict(role)},
        }
    if method not in PROPOSAL_METHODS:
        raise AggregateError(f"No formal resolved-codec contract for {method!r}")

    def _auto_rotation(code_bits_per_block: int) -> str:
        power_of_two = head_dim > 0 and not (head_dim & (head_dim - 1))
        if code_bits_per_block <= 4:
            return "dense"
        return "hybrid" if power_of_two else "dense"

    common = {
        "schema_version": 1,
        "head_dim": head_dim,
        "codec": "hayatequant",
        "calibration": None,
    }
    assignment = {
        "encoder": "exact",
        "lut_bins": None,
        "lut_candidates": None,
        "shape_independent": True,
        "tie_break": "smallest_global_codebook_index",
    }
    if method == "HQ":
        key_code_bits, value_code_bits = 2 * rate + 1, 2 * rate - 1

        def _fixed_role(code_bits: int, scale: str) -> dict:
            role = _role(
                bits=rate,
                block=2,
                rotation=_auto_rotation(code_bits),
                scale=scale,
                mathematical_scale=scale,
                source="gaussian",
            )
            role["code_bits_per_block"] = code_bits
            role["payload_bits_per_coordinate"] = code_bits / 2.0
            return role

        return {
            **common,
            "codebook_artifacts": json.loads(
                json.dumps(_expected_hayate_codebook_artifacts(HQ_CODEBOOK_BUNDLE))
            ),
            "assignment": assignment,
            "roles": {
                "key": _fixed_role(key_code_bits, "nc"),
                "value": _fixed_role(value_code_bits, "ls_iter"),
            },
        }

    total = 4 * rate
    candidate_pairs = [list(pair) for pair in ADAPTIVE_HQ_CANDIDATE_PAIRS[rate]]
    selector_bits = 1

    def _adaptive_role(scale: str) -> dict:
        role = _role(
            bits=rate,
            block=2,
            rotation="payload_dependent_candidate_rate",
            scale=scale,
            mathematical_scale=scale,
            source="gaussian",
        )
        role.update(
            {
                "adaptive_bit_split": "online_stats",
                "pair_code_bits_per_block": total,
            }
        )
        return role

    return {
        **common,
        "codebook_artifacts": json.loads(
            json.dumps(_expected_hayate_codebook_artifacts(ADAPTIVE_HQ_CODEBOOK_BUNDLE))
        ),
        "adaptive_allocation": {
            "block_size": 2,
            "candidate_pairs": candidate_pairs,
            "candidate_rotations": [
                {
                    "pair": pair,
                    "key": _auto_rotation(pair[0]),
                    "value": _auto_rotation(pair[1]),
                }
                for pair in candidate_pairs
            ],
            "distortion_table_version": ADAPTIVE_HQ_DISTORTION_TABLE_VERSION,
            "objective": ADAPTIVE_HQ_OBJECTIVE,
            "pair_code_bits_per_block": total,
            "policy": "cache_causal_datafree_rd",
            "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
            "prior_pair": [2 * rate + 1, 2 * rate - 1],
            "incumbent_pair": [2 * rate + 1, 2 * rate - 1],
            "switch_margin_fraction": ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION,
            "switch_margin_basis": ADAPTIVE_HQ_SWITCH_MARGIN_BASIS,
            "hysteresis": ADAPTIVE_HQ_HYSTERESIS,
            "runtime_distortion_constants": "payload_and_cache_provenance",
            "selected_pair": "payload_dependent",
            "selection_scope": "per_encode_kv_pair",
            "selector_bits_per_encode": selector_bits,
            "selector_carrier": "first_value_fp16_scale_low_fraction_bits",
            "selector_carrier_version": "value_scale_fp16_fraction_lsb_v1",
            "selector_carrier_distortion_amortization": (ADAPTIVE_HQ_CARRIER_AMORTIZATION),
            "selector_incremental_stored_bits_per_encode": 0,
            "selector_scale_reconstruction": "clear_carrier_fraction_bits",
            "state_ownership": "compressed_dynamic_cache_instance_per_layer",
        },
        "assignment": assignment,
        "roles": {
            "key": _adaptive_role("nc"),
            "value": _adaptive_role("ls_iter"),
        },
    }


def _require_formal_codec_recipe(
    row: dict, *, model: str, method: str, rate: int | None, path: Path
) -> str:
    """Validate cache policy plus declared and role-resolved codec identity."""
    if row.get("cache_policy") != {
        "sink_tokens": 0,
        "recent_tokens": 0,
        "skip_layers": [],
    }:
        raise AggregateError(f"Missing/changed formal cache policy for {model}/{method}: {path}")
    if method == "FullKV":
        if row.get("codec_config") is not None or row.get("resolved_codec_config") is not None:
            raise AggregateError(f"FullKV unexpectedly declares a compression recipe in {path}")
        return hashlib.sha256(b"FullKV:no-compression").hexdigest()
    if rate is None:
        raise AggregateError(f"Cannot resolve formal codec recipe for {model}/{method}: {path}")
    rate_i = int(rate)
    declared = row.get("codec_config")
    if not isinstance(declared, dict):
        raise AggregateError(
            f"Missing declared codec config for {model}/{rate_i}/{method}: {path}"
        )
    if method == "TurboQuant-paper":
        expected_declared = {
            "name": "turboquant",
            "bits": rate_i,
            "value_bits": None,
            "dist": "beta",
            "rotation_mode": "dense",
            "seed": 0,
            "n_outlier": 0,
            "outlier_bits": None,
            "use_qjl": False,
            "qjl_roles": "key",
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
    elif method == "FibQuant-paper":
        block_size, code_bits, _ = FIBQUANT_PAPER_RECIPES.get(rate_i, (None, None, None))
        expected_declared = {
            "name": "fibquant",
            "bits": float(rate_i),
            "block_size": None,
            "code_bits": None,
            "seed": 0,
            "codebook_seed": 0,
            "key_axis": "token",
            "value_axis": "token",
            "group_size": -1,
            "symmetric": False,
            "skip_layers": [],
            "residual_window": 128,
        }
        expected_provenance = {
            "method_role": "external_quality_baseline",
            "proposal_method": False,
            "paper_comparable": True,
            "speed_comparison": False,
            "primary_source": "arXiv:2605.11478v1",
            "nominal_payload_bits_per_coordinate": rate_i,
            "method_definition": (
                "FibQuant v1 exact spherical-Beta block-VQ quality baseline "
                f"({rate_i} payload bpc; k={block_size}, code_bits={code_bits})"
            ),
        }
        if any(row.get(key) != value for key, value in expected_provenance.items()):
            raise AggregateError(
                f"FibQuant quality-baseline provenance changed for {model}/{rate_i}: {path}"
            )
    elif method == "BlockQuant-paper":
        block_head_dim = PAPER_MODEL_HEAD_DIMS.get(model)
        candidate_resolved = row.get("resolved_codec_config")
        if block_head_dim is None and isinstance(candidate_resolved, dict):
            candidate_head_dim = candidate_resolved.get("head_dim")
            if isinstance(candidate_head_dim, int) and not isinstance(candidate_head_dim, bool):
                block_head_dim = candidate_head_dim
        if rate_i != 3 or block_head_dim != 128:
            raise AggregateError(
                f"BlockQuant-paper is outside its sole declared operating point: {path}"
            )
        expected_declared = _blockquant_declared_config()
        expected_provenance = _blockquant_quality_provenance_contract()
        if any(row.get(key) != value for key, value in expected_provenance.items()):
            raise AggregateError(
                f"BlockQuant quality-baseline provenance changed for {model}/{rate_i}: {path}"
            )
    elif method in PROPOSAL_METHODS:
        expected_declared = {
            "name": "hayatequant",
            "bits": rate_i,
            "value_bits": rate_i,
            "dist": "gaussian",
            "block_size": 2,
            "max_codebook_bits": 9,
            "codebook_init": "auto",
            "required_codebook_bundle": (
                HQ_CODEBOOK_BUNDLE if method == "HQ" else ADAPTIVE_HQ_CODEBOOK_BUNDLE
            ),
            "rotation_mode": "auto",
            "seed": 0,
            "encoder": "exact",
            "hadamard_gemm_max_dim": 4096,
            "ls_scale": True,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "n_outlier": 0,
            "key_n_outlier": None,
            "value_n_outlier": None,
            "use_qjl": False,
            "calibration": None,
            "adaptive_bit_split": (None if method == "HQ" else "online_stats"),
            "avg_bits": float(rate_i),
            "key_bit_fraction": (None if method == "HQ" else (2 * rate_i + 1) / (4 * rate_i)),
            "key_code_bits_per_block": (2 * rate_i + 1 if method == "HQ" else None),
            "value_code_bits_per_block": (2 * rate_i - 1 if method == "HQ" else None),
            "adaptive_bit_margin": 4,
            "adaptive_stats_min_tokens": 16,
            "key_error_weight": 4.0,
            "value_error_weight": 1.0,
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
        expected_provenance = HAYATE_FORMAL_PROVENANCE[method]
        if any(row.get(key) != value for key, value in expected_provenance.items()):
            raise AggregateError(
                f"Formal proposal provenance changed for {model}/{rate_i}/{method}: {path}"
            )
    elif method in {"HayateQuant-scalar-matched", "HayateQuant-blockvq-matched"}:
        expected_declared = {
            "name": "hayatequant",
            "bits": rate_i,
            "value_bits": rate_i,
            "dist": "beta",
            "block_size": (1 if method == "HayateQuant-scalar-matched" else None),
            "max_codebook_bits": 9,
            "codebook_init": "auto",
            "rotation_mode": "dense",
            "seed": 0,
            "encoder": "exact",
            "ls_scale": False,
            "key_scale_mode": None,
            "value_scale_mode": None,
            "n_outlier": 0,
            "key_n_outlier": None,
            "value_n_outlier": None,
            "use_qjl": False,
            "calibration": None,
            "adaptive_bit_split": None,
            "key_axis": "token",
            "value_axis": "token",
            "skip_layers": [],
        }
    else:
        raise AggregateError(f"Unexpected formal codec method {method!r}: {path}")
    if any(declared.get(key) != value for key, value in expected_declared.items()):
        raise AggregateError(
            f"Declared codec recipe changed for {model}/{rate_i}/{method}: {path}"
        )
    resolved = row.get("resolved_codec_config")
    head_dim = PAPER_MODEL_HEAD_DIMS.get(model)
    if head_dim is None and isinstance(resolved, dict):
        candidate = resolved.get("head_dim")
        if isinstance(candidate, int) and candidate > 0:
            # Unit/custom aggregations may deliberately pass a reduced model
            # set; production defaults remain pinned by PAPER_MODEL_HEAD_DIMS.
            head_dim = candidate
    if head_dim is None:
        raise AggregateError(f"Unknown model head dimension for {model}: {path}")
    expected_resolved = _expected_resolved_codec_config(method, rate_i, head_dim)
    if resolved != expected_resolved:
        raise AggregateError(
            f"Role-resolved codec recipe changed for {model}/{rate_i}/{method}: {path}"
        )
    return hashlib.sha256(
        json.dumps(
            {
                "cache_policy": row["cache_policy"],
                "declared": declared,
                "resolved": resolved,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _normalized_quality_runtime(record: Any, *, path: Path) -> dict:
    """Normalize result-affecting software, GPU class, and ONECOMP settings."""
    if not isinstance(record, dict):
        raise AggregateError(f"Malformed immutable runtime provenance in {path}")
    code = record.get("code")
    host = record.get("host")
    packages = record.get("packages")
    gpu = record.get("gpu")
    environment = record.get("environment")
    if not all(isinstance(value, dict) for value in (code, host, packages, gpu, environment)):
        raise AggregateError(f"Runtime software/GPU/environment provenance is missing in {path}")
    python_text = host.get("python")
    python_match = (
        re.match(r"^(\d+\.\d+\.\d+)", python_text) if isinstance(python_text, str) else None
    )
    if python_match is None:
        raise AggregateError(f"Python version provenance is missing in {path}")
    normalized_packages = {}
    for name in QUALITY_SOFTWARE_PACKAGES:
        value = packages.get(name)
        if not isinstance(value, str) or not value:
            raise AggregateError(f"Required package version {name!r} is missing in {path}")
        normalized_packages[name] = value
    if (
        gpu.get("available") is not True
        or not isinstance(gpu.get("torch_cuda"), str)
        or not gpu["torch_cuda"]
        or gpu.get("cudnn") is None
    ):
        raise AggregateError(f"CUDA software provenance is incomplete in {path}")
    devices = gpu.get("devices")
    if not isinstance(devices, list) or not devices:
        raise AggregateError(f"GPU device provenance is missing in {path}")
    device_classes = set()
    for device in devices:
        if not isinstance(device, dict):
            raise AggregateError(f"Malformed GPU device provenance in {path}")
        name = device.get("name")
        capability = device.get("compute_capability")
        if (
            not isinstance(name, str)
            or not name
            or "b200" not in name.lower()
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in capability
            )
        ):
            raise AggregateError(
                f"Formal quality run did not record an NVIDIA B200 class in {path}"
            )
        device_classes.add((" ".join(name.split()).lower(), tuple(capability)))
    semantic_environment = {key: environment.get(key) for key in QUALITY_SEMANTIC_ENVIRONMENT}
    if semantic_environment != QUALITY_SEMANTIC_ENVIRONMENT:
        raise AggregateError(f"Formal quality ONECOMP semantic environment changed in {path}")
    if environment.get("KVCOMP_CODEBOOK_DIR") is not None:
        raise AggregateError(f"Formal quality run used an external codebook in {path}")
    commit = code.get("commit")
    if not isinstance(commit, str) or not commit:
        raise AggregateError(f"onecomp source commit provenance is missing in {path}")
    return {
        "python": python_match.group(1),
        "packages": normalized_packages,
        "onecomp_commit": commit,
        "cuda": {
            "torch_cuda": gpu["torch_cuda"],
            "cudnn": gpu["cudnn"],
        },
        "gpu_classes": [
            {"name": name, "compute_capability": list(capability)}
            for name, capability in sorted(device_classes)
        ],
        "semantic_environment": semantic_environment,
    }


def _formal_smoke_file_record(
    value: Any,
    *,
    label: str,
    root: Path,
    path: Path,
) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise AggregateError(f"{label} is not an exact file record in {path}")
    raw_path = value.get("path")
    byte_count = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not Path(raw_path).is_absolute()
        or Path(raw_path).resolve(strict=False) != Path(raw_path)
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not _lower_sha256_string(digest)
    ):
        raise AggregateError(f"{label} file identity is malformed in {path}")
    try:
        Path(raw_path).relative_to(root)
    except ValueError as exc:
        raise AggregateError(f"{label} escapes its approved run-root scope in {path}") from exc
    return {"path": raw_path, "bytes": byte_count, "sha256": digest}


def _validated_formal_smoke_approval(
    record: Any,
    *,
    commit: str,
    input_manifest: Mapping[str, Any],
    path: Path,
) -> dict:
    """Validate the worker-time binding to the exact approved eight-job smoke."""

    approval = record.get("formal_smoke_approval") if isinstance(record, dict) else None
    required_keys = {
        "verification_mode",
        "marker",
        "validated_plan_sha256",
        "smoke_plan_identity",
        "code_commit",
        "input_manifest",
        "completed_job_keys",
        "results",
        "worker_preflights",
    }
    if not isinstance(approval, dict) or set(approval) != required_keys:
        raise AggregateError(f"Formal preflight has no exact smoke-approval binding in {path}")
    marker = approval.get("marker")
    if not isinstance(marker, dict) or not isinstance(marker.get("path"), str):
        raise AggregateError(f"Formal smoke marker identity is malformed in {path}")
    marker_path = Path(marker["path"])
    if marker_path.parent.name != "manifests":
        raise AggregateError(f"Formal smoke marker is outside a manifests directory in {path}")
    run_root = marker_path.parent.parent
    if not run_root.is_absolute() or run_root == marker_path.parent:
        raise AggregateError(f"Formal smoke marker has no absolute run-root identity in {path}")
    normalized_marker = _formal_smoke_file_record(
        marker,
        label="formal smoke approval marker",
        root=run_root / "manifests",
        path=path,
    )
    plan_sha256 = approval.get("validated_plan_sha256")
    smoke_plan = approval.get("smoke_plan_identity")
    if (
        approval.get("verification_mode") != FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE
        or approval.get("code_commit") != commit
        or not _lower_sha256_string(plan_sha256)
        or not isinstance(smoke_plan, dict)
        or set(smoke_plan) != {"version", "sha256"}
        or smoke_plan.get("version") != FORMAL_SMOKE_PLAN_VERSION
        or not _lower_sha256_string(smoke_plan.get("sha256"))
        or approval.get("input_manifest") != dict(input_manifest)
        or approval.get("completed_job_keys") != list(FORMAL_SMOKE_JOB_KEYS)
    ):
        raise AggregateError(f"Formal smoke approval identity changed in {path}")

    def keyed_records(value: Any, *, label: str, root: Path) -> list[dict]:
        if not isinstance(value, list) or len(value) != len(FORMAL_SMOKE_JOB_KEYS):
            raise AggregateError(f"{label} does not cover the exact smoke plan in {path}")
        normalized = []
        seen_paths: set[str] = set()
        for index, raw in enumerate(value):
            if not isinstance(raw, dict) or set(raw) != {"key", "path", "bytes", "sha256"}:
                raise AggregateError(f"{label} record {index} is malformed in {path}")
            key = raw.get("key")
            if key != FORMAL_SMOKE_JOB_KEYS[index]:
                raise AggregateError(f"{label} order/key identity changed in {path}")
            file_record = _formal_smoke_file_record(
                {name: raw.get(name) for name in ("path", "bytes", "sha256")},
                label=f"{label} {key}",
                root=root,
                path=path,
            )
            if file_record["path"] in seen_paths:
                raise AggregateError(f"{label} contains a duplicate path in {path}")
            seen_paths.add(file_record["path"])
            normalized.append({"key": key, **file_record})
        return normalized

    results = keyed_records(
        approval.get("results"),
        label="approved smoke result",
        root=run_root / "results/smoke",
    )
    worker_preflights = keyed_records(
        approval.get("worker_preflights"),
        label="approved smoke worker preflight",
        root=run_root / "provenance",
    )
    return {
        "verification_mode": FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE,
        "marker": normalized_marker,
        "validated_plan_sha256": plan_sha256,
        "smoke_plan_identity": dict(smoke_plan),
        "code_commit": commit,
        "input_manifest": dict(input_manifest),
        "completed_job_keys": list(FORMAL_SMOKE_JOB_KEYS),
        "results": results,
        "worker_preflights": worker_preflights,
    }


def _require_run_provenance(document: dict, *, path: Path, return_identity: bool = False):
    """Validate clean code, exact model, full weights, and input manifest.

    The old validator accepted a clean commit alone.  That allowed a result for
    one checkpoint or one regenerated input manifest to be filed under another
    model label while still passing every axis aggregator.
    """
    provenance = document.get("run_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise AggregateError(f"Missing run_paper_job provenance in {path}")
    model_label = document.get("model")
    config = document.get("config")
    if not isinstance(model_label, str) or not model_label:
        raise AggregateError(f"Result has no model label in {path}")
    if not isinstance(config, dict):
        raise AggregateError(f"Result has no scientific config in {path}")
    config_model_path = config.get("model_path")
    if not isinstance(config_model_path, str) or not config_model_path:
        raise AggregateError(f"Result config has no model_path in {path}")
    resolved_config_model = Path(config_model_path).expanduser().resolve()
    commits = set()
    identities = set()
    runtime_fingerprints = set()
    normalized_runtimes = {}
    smoke_approval_fingerprints = set()
    smoke_approval_records = {}
    for item in provenance:
        if not isinstance(item, dict) or "record" not in item:
            raise AggregateError(f"Malformed run provenance in {path}")
        record = item["record"]
        code = record.get("code") if isinstance(record, dict) else None
        if (
            not isinstance(code, dict)
            or code.get("tracked_dirty") is not False
            or not isinstance(code.get("commit"), str)
            or not code.get("commit")
        ):
            raise AggregateError(f"Result was not produced from a clean preflight in {path}")
        commits.add(code["commit"])
        digest = item.get("sha256")
        provenance_path = item.get("path")
        if (
            record.get("schema_version") != 1
            or not _sha256_string(digest)
            or not isinstance(provenance_path, str)
            or not Path(provenance_path).is_absolute()
        ):
            raise AggregateError(f"Malformed immutable preflight envelope in {path}")
        provenance_file = Path(provenance_path)
        if provenance_file.is_file():
            if _sha256(provenance_file).lower() != digest.lower():
                raise AggregateError(f"Immutable preflight file digest changed in {path}")
            if _load_object(provenance_file) != record:
                raise AggregateError(f"Embedded preflight differs from its source file in {path}")
        environment = record.get("environment")
        if (
            not isinstance(environment, dict)
            or "KVCOMP_CODEBOOK_DIR" not in environment
            or environment["KVCOMP_CODEBOOK_DIR"] is not None
        ):
            raise AggregateError(f"Formal result used an external KV codebook override in {path}")
        normalized_runtime = _normalized_quality_runtime(record, path=path)
        runtime_fingerprint = hashlib.sha256(
            json.dumps(normalized_runtime, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        runtime_fingerprints.add(runtime_fingerprint)
        normalized_runtimes[runtime_fingerprint] = normalized_runtime

        model = record.get("model")
        if not isinstance(model, dict):
            raise AggregateError(f"Preflight omitted the model record in {path}")
        model_path = model.get("path")
        expected_key = MODEL_MANIFEST_KEYS.get(model_label, model_label)
        if (
            not isinstance(model_path, str)
            or Path(model_path).expanduser().resolve() != resolved_config_model
            or resolved_config_model.name != model_label
            or model.get("key") != expected_key
        ):
            raise AggregateError(
                f"Result model label/config/preflight manifest key disagree in {path}"
            )
        if model.get("weight_verification_mode") != "full_sha256":
            raise AggregateError(f"Model weights were not fully SHA-256 verified in {path}")
        _validated_weight_record(model.get("weights"), label=str(path))

        input_manifest = model.get("input_manifest")
        if (
            not isinstance(input_manifest, dict)
            or not isinstance(input_manifest.get("path"), str)
            or not Path(input_manifest["path"]).is_absolute()
            or not _sha256_string(input_manifest.get("sha256"))
            or not isinstance(input_manifest.get("bytes"), int)
            or input_manifest["bytes"] <= 0
        ):
            raise AggregateError(f"Preflight omitted the pinned input-manifest digest in {path}")
        dataset_files = record.get("dataset_files")
        if not isinstance(dataset_files, list) or not dataset_files:
            raise AggregateError(f"Preflight omitted dataset/input file provenance in {path}")
        dataset_paths = set()
        for dataset in dataset_files:
            if (
                not isinstance(dataset, dict)
                or not isinstance(dataset.get("path"), str)
                or not Path(dataset["path"]).is_absolute()
                or dataset["path"] in dataset_paths
                or not _sha256_string(dataset.get("sha256"))
                or not isinstance(dataset.get("bytes"), int)
                or dataset["bytes"] <= 0
            ):
                raise AggregateError(f"Malformed/duplicate preflight dataset file in {path}")
            dataset_paths.add(dataset["path"])
        matching_inputs = [
            dataset
            for dataset in dataset_files
            if isinstance(dataset, dict) and dataset.get("path") == input_manifest["path"]
        ]
        if matching_inputs != [input_manifest]:
            raise AggregateError(
                f"Preflight dataset files do not contain the exact input manifest in {path}"
            )
        smoke_approval = _validated_formal_smoke_approval(
            record,
            commit=code["commit"],
            input_manifest=input_manifest,
            path=path,
        )
        smoke_approval_fingerprint = hashlib.sha256(
            json.dumps(
                smoke_approval,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        smoke_approval_fingerprints.add(smoke_approval_fingerprint)
        smoke_approval_records[smoke_approval_fingerprint] = smoke_approval
        identities.add(
            (
                str(resolved_config_model),
                expected_key,
                input_manifest["path"],
                input_manifest["sha256"].lower(),
            )
        )
    if len(commits) != 1:
        raise AggregateError(
            f"One result merged multiple source commits in {path}: {sorted(commits)}"
        )
    if len(identities) != 1:
        raise AggregateError(f"One result merged different model/input identities in {path}")
    if len(runtime_fingerprints) != 1:
        raise AggregateError(
            f"One result merged different software/GPU/environment strata in {path}"
        )
    if len(smoke_approval_fingerprints) != 1:
        raise AggregateError(f"One result merged different formal smoke approvals in {path}")
    identity = next(iter(identities))
    commit = next(iter(commits))
    declared_commits = [
        document.get("code_commit"),
        config.get("code_commit"),
        (
            document.get("provenance", {}).get("git", {}).get("commit")
            if isinstance(document.get("provenance"), dict)
            else None
        ),
    ]
    formal_schema = document.get("schema_version") == 2
    if formal_schema and not any(value is not None for value in declared_commits):
        raise AggregateError(f"Formal result does not declare its code commit in {path}")
    if any(value is not None and value != commit for value in declared_commits):
        raise AggregateError(f"Result/preflight code commit mismatch in {path}")
    semantic_config = document.get("semantic_config")
    run_fingerprint = document.get("run_fingerprint")
    if formal_schema and not _sha256_string(run_fingerprint):
        raise AggregateError(f"Formal result has no valid run fingerprint in {path}")
    if isinstance(semantic_config, dict) and isinstance(run_fingerprint, str):
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                {"semantic_config": semantic_config, "code_commit": commit},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if run_fingerprint != expected_fingerprint:
            raise AggregateError(f"Result fingerprint/preflight commit mismatch in {path}")
    elif config.get("schema_version") == 2 and isinstance(run_fingerprint, str):
        expected_fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if run_fingerprint != expected_fingerprint:
            raise AggregateError(f"Result fingerprint/preflight commit mismatch in {path}")
    elif formal_schema:
        raise AggregateError(
            f"Formal result fingerprint has no reproducible scientific config in {path}"
        )
    if return_identity:
        return {
            "commit": commit,
            "model_path": identity[0],
            "model_key": identity[1],
            "input_manifest_path": identity[2],
            "input_manifest_sha256": identity[3],
            "quality_runtime_fingerprint": next(iter(runtime_fingerprints)),
            "quality_runtime": normalized_runtimes[next(iter(runtime_fingerprints))],
            "formal_smoke_approval": smoke_approval_records[
                next(iter(smoke_approval_fingerprints))
            ],
        }
    return commit


def verify_ppl(
    root: Path,
    *,
    expected_models: Sequence[str] = PAPER_MODELS,
    _expected_denominators: Mapping[str, Mapping[str, int]] | None = None,
) -> dict:
    """Verify the seven-model complete WikiText-2 matrix."""
    root = root.expanduser().resolve()
    if _expected_denominators is not None:
        if set(_expected_denominators) != set(expected_models):
            raise AggregateError("Custom PPL denominator fixture does not cover every model")
        for model, denominator in _expected_denominators.items():
            if not isinstance(denominator, Mapping):
                raise AggregateError(f"Custom PPL denominator is malformed for {model}")
            n_tokens = _integer(
                denominator.get("n_tokens"), label=f"custom {model} n_tokens", minimum=1
            )
            traversed = _integer(
                denominator.get("traversed_tokens"),
                label=f"custom {model} traversed_tokens",
                minimum=2,
            )
            if traversed != n_tokens + 1:
                raise AggregateError(f"Custom PPL off-by-one contract changed for {model}")
    paths = sorted(root.rglob("results_*.json")) if root.is_dir() else []
    by_model: dict[str, dict] = {}
    sources = []
    text_hashes = set()
    code_commits = set()
    input_manifests = set()
    for path in paths:
        document = _load_object(path)
        model = document.get("model")
        if model not in expected_models:
            raise AggregateError(f"Unexpected PPL model {model!r}: {path}")
        if model in by_model:
            raise AggregateError(f"Duplicate PPL result for {model}: {path}")
        if document.get("schema_version") != 2 or not document.get("run_fingerprint"):
            raise AggregateError(f"Legacy/unidentified PPL result: {path}")
        provenance = _require_run_provenance(document, path=path, return_identity=True)
        if _expected_denominators is None:
            expected_denominator = _validated_frozen_ppl_denominator(
                str(model),
                input_manifest_path=provenance["input_manifest_path"],
                input_manifest_sha256=provenance["input_manifest_sha256"],
            )
        else:
            expected_denominator = dict(_expected_denominators[str(model)])
        code_commits.add(provenance["commit"])
        input_manifests.add(
            (provenance["input_manifest_path"], provenance["input_manifest_sha256"])
        )
        config = document.get("config")
        if not isinstance(config, dict) or config.get("ppl_windows") is not None:
            raise AggregateError(f"PPL result is not complete-text evaluation: {path}")
        if (
            config.get("run_ppl") is not True
            or config.get("dtype") != QUALITY_DTYPE
            or config.get("ppl_max_length") != PPL_MAX_LENGTH
            or config.get("ppl_stride") != PPL_STRIDE
            or config.get("sample_seed") != 2027
            or config.get("bits") != [2, 3, 4]
            or config.get("methods")
            != [
                "TurboQuant-paper",
                "FibQuant-paper",
                "BlockQuant-paper",
                "HQ",
                "AdaptiveHQ",
            ]
        ):
            raise AggregateError(f"PPL formal config contract changed in {path}")
        if config.get("skip_outliers") is not False:
            raise AggregateError(f"PPL formal row used model-dependent layer skipping: {path}")
        if config.get("sink_tokens") != 0 or config.get("recent_tokens") != 0:
            raise AggregateError(f"PPL formal row retained FP16 token windows: {path}")
        full = document.get("fullkv_ppl")
        if (
            not isinstance(full, dict)
            or full.get("ppl_complete_text_eval") is not True
            or full.get("ppl_source") != "wikitext-2-raw-v1/test"
            or full.get("ppl_n_skipped_nonfinite") != 0
            or full.get("ppl_max_windows") is not None
        ):
            raise AggregateError(f"Missing complete FullKV PPL in {path}")
        full_ppl, full_nll, full_nll_sum, full_n_tokens = _validated_ppl_numbers(
            full,
            label=f"{path} FullKV",
            expected_tokens=expected_denominator["n_tokens"],
        )
        if full.get("ppl_traversed_tokens") != expected_denominator["traversed_tokens"]:
            raise AggregateError(f"PPL frozen traversed-token count changed for {model}")
        full_accounting = _validated_ppl_protocol_and_compression(
            full, label=f"{path} FullKV", compressed=False
        )
        text_hash = full.get("ppl_text_sha256")
        if (
            full.get("ppl_dataset_revision") != WIKITEXT2_REVISION
            or not isinstance(text_hash, str)
            or len(text_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in text_hash.lower())
        ):
            raise AggregateError(f"Unpinned WikiText-2 text/revision in {path}")
        text_hashes.add(text_hash.lower())
        results = document.get("results")
        if not isinstance(results, dict) or set(results) != {"2", "3", "4"}:
            raise AggregateError(f"PPL bit matrix must be exactly 2/3/4 in {path}")
        compact = {
            "fullkv": {
                **full,
                **full_accounting,
                "ppl": full_ppl,
                "ppl_nll": full_nll,
                "ppl_nll_sum": full_nll_sum,
                "ppl_n_tokens": full_n_tokens,
                "ppl_traversed_tokens": full.get("ppl_traversed_tokens"),
            },
            "rates": {},
        }
        for rate in ("2", "3", "4"):
            methods = results[rate]
            expected = set(ZERO_RATE_METHODS[rate])
            if not isinstance(methods, dict) or set(methods) != expected:
                raise AggregateError(f"Wrong PPL methods at {rate} bit in {path}")
            compact["rates"][rate] = {}
            for method in sorted(expected):
                row = methods[method]
                if not isinstance(row, dict):
                    raise AggregateError(f"Malformed PPL row for {model}/{rate}/{method}")
                recipe_sha256 = _require_formal_codec_recipe(
                    row,
                    model=str(model),
                    method=str(method),
                    rate=int(rate),
                    path=path,
                )
                if (
                    row.get("ppl_complete_text_eval") is not True
                    or row.get("ppl_source") != "wikitext-2-raw-v1/test"
                    or row.get("ppl_n_skipped_nonfinite") != 0
                    or row.get("ppl_max_windows") is not None
                    or row.get("ppl_n_windows") != full.get("ppl_n_windows")
                ):
                    raise AggregateError(f"Capped PPL for {model}/{rate}/{method}")
                ppl, mean_nll, nll_sum, n_tokens = _validated_ppl_numbers(
                    row,
                    label=f"{model}/{rate}/{method}",
                    expected_tokens=full_n_tokens,
                )
                accounting = _validated_ppl_protocol_and_compression(
                    row,
                    label=f"{model}/{rate}/{method}",
                    compressed=True,
                    method=method,
                )
                row_text_hash = row.get("ppl_text_sha256")
                if (
                    row.get("ppl_dataset_revision") != WIKITEXT2_REVISION
                    or not isinstance(row_text_hash, str)
                    or row_text_hash.lower() != text_hash.lower()
                ):
                    raise AggregateError(
                        f"WikiText-2 text/revision changed for {model}/{rate}/{method}"
                    )
                codec_config = row.get("codec_config")
                if not isinstance(codec_config, dict):
                    raise AggregateError(f"Missing codec definition for {model}/{rate}/{method}")
                rate_i = int(rate)
                if method == "TurboQuant-paper":
                    expected_fields = {
                        "name": "turboquant",
                        "bits": rate_i,
                        "value_bits": None,
                        "dist": "beta",
                        "rotation_mode": "dense",
                        "n_outlier": 0,
                        "outlier_bits": None,
                        "use_qjl": False,
                        "qjl_roles": "key",
                    }
                elif method == "FibQuant-paper":
                    expected_fields = {
                        "name": "fibquant",
                        "bits": float(rate_i),
                        "block_size": None,
                        "code_bits": None,
                        "seed": 0,
                        "codebook_seed": 0,
                    }
                    if "ms_per_encode" in row or "throughput_Melem_s" in row:
                        raise AggregateError(
                            f"FibQuant quality row unexpectedly contains speed metrics in {path}"
                        )
                    if row.get("speed_metrics_omitted") != (
                        "quality-only external baseline; no packed FibQuant serving kernel"
                    ):
                        raise AggregateError(f"Missing FibQuant speed-omission marker in {path}")
                elif method == "BlockQuant-paper":
                    expected_fields = _blockquant_declared_config()
                    if "ms_per_encode" in row or "throughput_Melem_s" in row:
                        raise AggregateError(
                            f"BlockQuant quality row unexpectedly contains speed metrics in {path}"
                        )
                    if row.get("speed_metrics_omitted") != (
                        "quality-only external baseline; no packed BlockQuant serving kernel"
                    ):
                        raise AggregateError(f"Missing BlockQuant speed-omission marker in {path}")
                elif method in PROPOSAL_METHODS:
                    expected_fields = {
                        "name": "hayatequant",
                        "bits": rate_i,
                        "value_bits": rate_i,
                        "dist": "gaussian",
                        "block_size": 2,
                        "max_codebook_bits": 9,
                        "codebook_init": "auto",
                        "rotation_mode": "auto",
                        "encoder": "exact",
                        "ls_scale": True,
                        "key_scale_mode": "nc",
                        "value_scale_mode": "ls_iter",
                        "n_outlier": 0,
                        "outlier_bits": None,
                        "key_n_outlier": None,
                        "value_n_outlier": None,
                        "use_qjl": False,
                        "calibration": None,
                        "avg_bits": float(rate_i),
                        "adaptive_bit_split": (None if method == "HQ" else "online_stats"),
                        "key_code_bits_per_block": (2 * rate_i + 1 if method == "HQ" else None),
                        "value_code_bits_per_block": (2 * rate_i - 1 if method == "HQ" else None),
                    }
                else:  # pragma: no cover - closed by ZERO_RATE_METHODS
                    raise AggregateError(f"Unexpected PPL method {method!r}")
                if any(codec_config.get(key) != value for key, value in expected_fields.items()):
                    raise AggregateError(
                        f"Wrong codec definition for {model}/{rate}/{method}: {codec_config}"
                    )
                stored_bits, stored_coordinates, stored_rate = _validated_stored_rate(
                    row, label=f"{path} {rate}/{method}"
                )
                blockquant_stored_accounting = None
                if method == "BlockQuant-paper":
                    blockquant_stored_accounting = _validated_blockquant_stored_accounting(
                        row,
                        stored_bits=stored_bits,
                        stored_coordinates=stored_coordinates,
                        label=f"{path} {rate}/{method}",
                    )
                compact["rates"][rate][method] = {
                    "ppl": ppl,
                    "ppl_nll": mean_nll,
                    "ppl_nll_sum": nll_sum,
                    "ppl_n_tokens": n_tokens,
                    "ppl_n_windows": row.get("ppl_n_windows"),
                    "ppl_traversed_tokens": row.get("ppl_traversed_tokens"),
                    "ppl_scored_target_index_set": row.get(
                        "ppl_scored_target_index_set"
                    ),
                    "ppl_scored_target_ranges": row.get("ppl_scored_target_ranges"),
                    "ppl_scored_target_coverage": row.get("ppl_scored_target_coverage"),
                    **accounting,
                    "dataset_revision": row.get("ppl_dataset_revision"),
                    "text_sha256": row.get("ppl_text_sha256"),
                    "distortion_key": _number(
                        row.get("distortion_key"),
                        label=f"{path} {rate}/{method} key distortion",
                    ),
                    "distortion_value": _number(
                        row.get("distortion_value"),
                        label=f"{path} {rate}/{method} value distortion",
                    ),
                    "stored_bits": stored_bits,
                    "stored_coordinates": stored_coordinates,
                    "stored_bits_per_coord": stored_rate,
                    "stored_bit_accounting": blockquant_stored_accounting,
                    "codec_config": codec_config,
                    "resolved_codec_config": row["resolved_codec_config"],
                    "cache_policy": row["cache_policy"],
                    "recipe_sha256": recipe_sha256,
                    "method_role": row.get("method_role"),
                    "external_quality_baseline": row.get("external_quality_baseline"),
                    "paper_comparable": row.get("paper_comparable"),
                    "paper_reproduction": row.get("paper_reproduction"),
                    "proposal_method": row.get("proposal_method"),
                    "speed_comparison": row.get("speed_comparison"),
                    "random_seed_protocol": row.get("random_seed_protocol"),
                }
            tq_rate = compact["rates"][rate]["TurboQuant-paper"]["stored_bits_per_coord"]
            fib_rate = compact["rates"][rate]["FibQuant-paper"]["stored_bits_per_coord"]
            tq_row = compact["rates"][rate]["TurboQuant-paper"]
            fib_row = compact["rates"][rate]["FibQuant-paper"]
            if (
                tq_row["stored_coordinates"] != fib_row["stored_coordinates"]
                or tq_row["stored_bits"] != fib_row["stored_bits"]
                or tq_rate != fib_rate
                or tq_row["ppl_compression_fp16_bits"] != fib_row["ppl_compression_fp16_bits"]
                or tq_row["ppl_compression_stored_bits"] != fib_row["ppl_compression_stored_bits"]
            ):
                raise AggregateError(
                    f"Primary PPL comparison does not have exact matched bit counters "
                    f"for {model}/{rate}: "
                    f"TurboQuant={tq_rate}, FibQuant={fib_rate}"
                )
            fib_row["payload_match_vs_turboquant"] = {
                "exact_integer_counters": True,
                "ppl_compression_fp16_bits": fib_row["ppl_compression_fp16_bits"],
                "ppl_compression_stored_bits": fib_row["ppl_compression_stored_bits"],
            }
            for proposal in PROPOSAL_METHODS:
                proposal_row = compact["rates"][rate][proposal]
                proposal_rate = proposal_row["stored_bits_per_coord"]
                if (
                    tq_row["stored_coordinates"] != proposal_row["stored_coordinates"]
                    or tq_row["stored_bits"] != proposal_row["stored_bits"]
                    or tq_rate != proposal_rate
                    or tq_row["ppl_compression_fp16_bits"]
                    != proposal_row["ppl_compression_fp16_bits"]
                    or tq_row["ppl_compression_stored_bits"]
                    != proposal_row["ppl_compression_stored_bits"]
                ):
                    raise AggregateError(
                        f"Primary PPL comparison does not have exact matched bit counters for "
                        f"{model}/{rate}/{proposal}: TurboQuant={tq_rate}, "
                        f"proposal={proposal_rate}"
                    )
                proposal_row["payload_match_vs_turboquant"] = {
                    "exact_integer_counters": True,
                    "ppl_compression_fp16_bits": proposal_row["ppl_compression_fp16_bits"],
                    "ppl_compression_stored_bits": proposal_row["ppl_compression_stored_bits"],
                }
        compact["official_complete"] = True
        by_model[str(model)] = compact
        sources.append(_source(path))
    missing = sorted(set(expected_models) - set(by_model))
    if missing or len(paths) != len(expected_models):
        raise AggregateError(f"PPL matrix incomplete; missing models={missing}")
    if len(code_commits) != 1:
        raise AggregateError(f"PPL model runs span multiple commits: {sorted(code_commits)}")
    if len(input_manifests) != 1:
        raise AggregateError("PPL model runs span multiple paper input manifests")
    if len(text_hashes) != 1:
        raise AggregateError("WikiText-2 raw text hash changed across model runs")
    if next(iter(text_hashes)) != WIKITEXT2_TEXT_SHA256:
        raise AggregateError(
            "WikiText-2 raw text is not the pinned 4358-row '\\n\\n'-joined source: "
            f"{next(iter(text_hashes))}"
        )
    primary = {}
    for proposal in PROPOSAL_METHODS:
        proposal_rates = {}
        for rate in ("2", "3", "4"):
            log_ratios = [
                by_model[model]["rates"][rate][proposal]["ppl_nll"]
                - by_model[model]["rates"][rate]["TurboQuant-paper"]["ppl_nll"]
                for model in expected_models
            ]
            n_models = len(log_ratios)
            mean_log_ratio = math.fsum(log_ratios) / n_models
            if n_models > 1:
                variance = math.fsum((value - mean_log_ratio) ** 2 for value in log_ratios) / (
                    n_models - 1
                )
                stderr = math.sqrt(variance / n_models)
            else:
                stderr = 0.0
            critical = 2.4469118511449692 if n_models == 7 else 1.959963984540054
            simultaneous_critical = 3.287455157088189 if n_models == 7 else 1.959963984540054
            proposal_rates[rate] = {
                "proposal": proposal,
                "estimand": "geometric_mean_within_model_ppl_ratio_proposal_over_turboquant",
                "student_t_statistical_unit": "fixed_checkpoint",
                "checkpoint_panel": "seven_preselected_family_correlated_checkpoints_not_iid",
                "population_generalization": False,
                "n_models": n_models,
                "mean_log_ppl_ratio": round(mean_log_ratio, 12),
                "mean_log_ppl_ratio_standard_error": round(stderr, 12),
                "geometric_mean_ppl_ratio": round(math.exp(mean_log_ratio), 10),
                "relative_ppl_change_percent": round(100.0 * (math.exp(mean_log_ratio) - 1.0), 8),
                "student_t_ci95_geometric_mean_ratio": [
                    round(math.exp(mean_log_ratio - critical * stderr), 10),
                    round(math.exp(mean_log_ratio + critical * stderr), 10),
                ],
                "nominal_interval_status": "secondary_robustness_not_primary_win_gate",
                "bonferroni_simultaneous_familywise95_geometric_mean_ratio": [
                    round(math.exp(mean_log_ratio - simultaneous_critical * stderr), 10),
                    round(math.exp(mean_log_ratio + simultaneous_critical * stderr), 10),
                ],
                "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
                "familywise_alpha": 0.05,
                "per_rate_two_sided_alpha": 0.05 / 3.0,
                "per_model_ratio": {
                    model: round(
                        math.exp(
                            by_model[model]["rates"][rate][proposal]["ppl_nll"]
                            - by_model[model]["rates"][rate]["TurboQuant-paper"]["ppl_nll"]
                        ),
                        10,
                    )
                    for model in expected_models
                },
                "claim_rule": "interval_is_secondary_robustness_only",
            }
        primary[proposal] = proposal_rates
    return {
        "benchmark": "WikiText-2 perplexity",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "commit": next(iter(code_commits)),
        "models": by_model,
        "secondary_interval_robustness_vs_turboquant": primary,
        "primary_metric_note": (
            "Per-checkpoint raw PPL point estimates feed the predeclared 90% win gate; "
            "log-ratio intervals are secondary robustness summaries only."
        ),
        "inference_scope": {
            "student_t_statistical_unit": "fixed_checkpoint",
            "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
            "interval_interpretation": (
                "descriptive_uncertainty_across_the_fixed_checkpoint_panel"
            ),
            "population_generalization": False,
        },
        "compression_accounting": dict(PPL_COMPRESSION_ACCOUNTING),
        "evaluation_config": {
            "dtype": QUALITY_DTYPE,
            "run_ppl": True,
            "ppl_max_length": PPL_MAX_LENGTH,
            "ppl_stride": PPL_STRIDE,
            "ppl_windows": None,
            "sample_seed": 2027,
            "bits": [2, 3, 4],
            "methods": [
                "TurboQuant-paper",
                "FibQuant-paper",
                "BlockQuant-paper",
                "HQ",
                "AdaptiveHQ",
            ],
        },
        "teacher_forced_scoring_contract": {
            "protocol": PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": False,
            "method_overrides": {
                "BlockQuant-paper": {
                    "protocol": CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": BLOCKQUANT_TEMPORAL_DEPENDENCY,
                },
                "AdaptiveHQ": {
                    "protocol": CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL,
                    "temporal_dependency": ADAPTIVE_HQ_TEMPORAL_DEPENDENCY,
                },
            },
        },
        "dataset_revision": WIKITEXT2_REVISION,
        "text_sha256": next(iter(text_hashes)),
        "frozen_denominator_contract": {
            "kind": "repo_pinned_tokenizer_specific_wikitext2_contract",
            "path": (
                str(PPL_DENOMINATOR_CONTRACT_PATH)
                if _expected_denominators is None
                else None
            ),
            "sha256": (
                _sha256(PPL_DENOMINATOR_CONTRACT_PATH)
                if _expected_denominators is None
                else None
            ),
            "models": {
                model: {
                    "n_tokens": by_model[model]["fullkv"]["ppl_n_tokens"],
                    "traversed_tokens": by_model[model]["fullkv"][
                        "ppl_traversed_tokens"
                    ],
                }
                for model in expected_models
            },
        },
        "official_complete": True,
        "sources": sources,
    }


def verify_mechanism(
    root: Path,
    *,
    reference_ppl: Mapping[str, Any],
    expected_models: Sequence[str] = LONG_MODELS,
) -> dict:
    """Verify the exact scalar-vs-block VQ causal control on full WikiText-2."""
    root = root.expanduser().resolve()
    paths = sorted(root.rglob("results_*.json")) if root.is_dir() else []
    if not paths:
        raise AggregateError(f"No mechanism-control results found below {root}")
    controls = ("HayateQuant-scalar-matched", "HayateQuant-blockvq-matched")
    by_model = {}
    sources = []
    commits = set()
    input_manifests = set()
    for path in paths:
        document = _load_object(path)
        model = document.get("model")
        if model not in expected_models or model in by_model:
            raise AggregateError(f"Unexpected/duplicate mechanism model {model!r}: {path}")
        if document.get("schema_version") != 2 or not document.get("run_fingerprint"):
            raise AggregateError(f"Legacy mechanism-control result: {path}")
        provenance = _require_run_provenance(document, path=path, return_identity=True)
        commits.add(provenance["commit"])
        input_manifests.add(
            (provenance["input_manifest_path"], provenance["input_manifest_sha256"])
        )
        config = document.get("config")
        if (
            not isinstance(config, dict)
            or config.get("methods") != list(controls)
            or set(config.get("bits", [])) != {2, 3, 4}
            or config.get("ppl_windows") is not None
            or config.get("skip_outliers") is not False
            or config.get("sink_tokens") != 0
            or config.get("recent_tokens") != 0
        ):
            raise AggregateError(f"Wrong mechanism-control formal config in {path}")
        reference_model = reference_ppl.get("models", {}).get(str(model))
        if not isinstance(reference_model, dict):
            raise AggregateError(f"Mechanism model is absent from primary PPL: {model}")
        full = document.get("fullkv_ppl")
        if (
            not isinstance(full, dict)
            or full.get("ppl_complete_text_eval") is not True
            or full.get("ppl_source") != "wikitext-2-raw-v1/test"
            or full.get("ppl_n_skipped_nonfinite") != 0
            or full.get("ppl_max_windows") is not None
            or full.get("ppl_n_tokens") != reference_model["fullkv"].get("ppl_n_tokens")
            or full.get("ppl_n_windows") != reference_model["fullkv"].get("ppl_n_windows")
            or full.get("ppl") != reference_model["fullkv"].get("ppl")
            or full.get("ppl_dataset_revision") != WIKITEXT2_REVISION
            or full.get("ppl_text_sha256") != reference_ppl.get("text_sha256")
        ):
            raise AggregateError(f"Mechanism control changed FullKV/text denominator in {path}")
        _, mechanism_full_nll, _, _ = _validated_ppl_numbers(
            full,
            label=f"{path} mechanism FullKV",
            expected_tokens=reference_model["fullkv"]["ppl_n_tokens"],
        )
        _validated_ppl_protocol_and_compression(
            full, label=f"{path} mechanism FullKV", compressed=False
        )
        if not math.isclose(
            mechanism_full_nll,
            reference_model["fullkv"]["ppl_nll"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AggregateError(f"Mechanism control changed FullKV NLL in {path}")
        results = document.get("results")
        if not isinstance(results, dict) or set(results) != {"2", "3", "4"}:
            raise AggregateError(f"Wrong mechanism rate matrix in {path}")
        model_rates = {}
        for rate in ("2", "3", "4"):
            rate_i = int(rate)
            method_rows = results[rate]
            if not isinstance(method_rows, dict) or set(method_rows) != set(controls):
                raise AggregateError(f"Wrong mechanism methods for {model}/{rate}: {path}")
            compact_controls = {}
            for method in controls:
                row = method_rows[method]
                if not isinstance(row, dict):
                    raise AggregateError(f"Malformed mechanism row for {model}/{rate}/{method}")
                recipe_sha256 = _require_formal_codec_recipe(
                    row,
                    model=str(model),
                    method=str(method),
                    rate=rate_i,
                    path=path,
                )
                codec = row.get("codec_config") if isinstance(row, dict) else None
                expected_block = 1 if method == controls[0] else None
                expected_codec = {
                    "name": "hayatequant",
                    "bits": rate_i,
                    "value_bits": rate_i,
                    "block_size": expected_block,
                    "dist": "beta",
                    "rotation_mode": "dense",
                    "encoder": "exact",
                    "ls_scale": False,
                    "key_scale_mode": None,
                    "value_scale_mode": None,
                    "codebook_init": "auto",
                    "n_outlier": 0,
                    "key_n_outlier": None,
                    "value_n_outlier": None,
                    "use_qjl": False,
                    "calibration": None,
                }
                if not isinstance(codec, dict) or any(
                    codec.get(key) != value for key, value in expected_codec.items()
                ):
                    raise AggregateError(
                        f"Mechanism control differs outside block_size for {model}/{rate}/{method}"
                    )
                if (
                    row.get("ppl_complete_text_eval") is not True
                    or row.get("ppl_source") != "wikitext-2-raw-v1/test"
                    or row.get("ppl_n_skipped_nonfinite") != 0
                    or row.get("ppl_max_windows") is not None
                    or row.get("ppl_n_tokens") != full["ppl_n_tokens"]
                    or row.get("ppl_n_windows") != full.get("ppl_n_windows")
                    or row.get("ppl_dataset_revision") != WIKITEXT2_REVISION
                    or row.get("ppl_text_sha256") != reference_ppl.get("text_sha256")
                ):
                    raise AggregateError(
                        f"Capped/drifted mechanism PPL for {model}/{rate}/{method}"
                    )
                ppl, mean_nll, nll_sum, n_tokens = _validated_ppl_numbers(
                    row,
                    label=f"{model}/{rate}/{method}",
                    expected_tokens=full["ppl_n_tokens"],
                )
                accounting = _validated_ppl_protocol_and_compression(
                    row,
                    label=f"{model}/{rate}/{method}",
                    compressed=True,
                )
                stored_bits, stored_coordinates, stored_rate = _validated_stored_rate(
                    row, label=f"{path} {rate}/{method}"
                )
                compact_controls[method] = {
                    "ppl": ppl,
                    "ppl_nll": mean_nll,
                    "ppl_nll_sum": nll_sum,
                    "ppl_n_tokens": n_tokens,
                    **accounting,
                    "distortion_key": _number(
                        row.get("distortion_key"), label=f"{path} {rate}/{method} K distortion"
                    ),
                    "distortion_value": _number(
                        row.get("distortion_value"), label=f"{path} {rate}/{method} V distortion"
                    ),
                    "stored_bits": stored_bits,
                    "stored_coordinates": stored_coordinates,
                    "stored_bits_per_coord": stored_rate,
                    "codec_config": codec,
                    "resolved_codec_config": row["resolved_codec_config"],
                    "cache_policy": row["cache_policy"],
                    "recipe_sha256": recipe_sha256,
                }
            scalar_rate = compact_controls[controls[0]]["stored_bits_per_coord"]
            block_rate = compact_controls[controls[1]]["stored_bits_per_coord"]
            scalar_row = compact_controls[controls[0]]
            block_row = compact_controls[controls[1]]
            if (
                scalar_row["stored_coordinates"] != block_row["stored_coordinates"]
                or scalar_row["stored_bits"] != block_row["stored_bits"]
                or scalar_rate != block_rate
            ):
                raise AggregateError(f"Mechanism controls are not iso-rate for {model}/{rate}")
            scalar_codec = compact_controls[controls[0]]["codec_config"]
            block_codec = compact_controls[controls[1]]["codec_config"]
            scalar_without_block = {
                key: value for key, value in scalar_codec.items() if key != "block_size"
            }
            block_without_block = {
                key: value for key, value in block_codec.items() if key != "block_size"
            }
            if scalar_without_block != block_without_block:
                raise AggregateError(
                    f"Mechanism controls differ outside block_size for {model}/{rate}"
                )
            primary = reference_model["rates"][rate]
            turbo = primary["TurboQuant-paper"]
            turbo_rate = turbo["stored_bits_per_coord"]
            if (
                scalar_row["stored_coordinates"] != turbo.get("stored_coordinates")
                or scalar_row["stored_bits"] != turbo.get("stored_bits")
                or scalar_rate != turbo_rate
            ):
                raise AggregateError(
                    f"Mechanism controls do not match TurboQuant rate for {model}/{rate}"
                )
            combined = {
                "TurboQuant-paper": primary["TurboQuant-paper"],
                **compact_controls,
                "HQ": primary["HQ"],
                "AdaptiveHQ": primary["AdaptiveHQ"],
            }
            scalar_log_ppl = combined[controls[0]]["ppl_nll"]
            block_log_ppl = combined[controls[1]]["ppl_nll"]
            block_over_scalar = math.exp(block_log_ppl - scalar_log_ppl)
            model_rates[rate] = {
                "methods": combined,
                "causal_block_vq_ratio_block_over_scalar": round(block_over_scalar, 10),
                "causal_block_vq_relative_ppl_change_percent": round(
                    100.0 * (block_over_scalar - 1.0), 8
                ),
                "control_contract": (
                    "Beta+dense+RMS+exact encoder+K=V=r; block_size is the only "
                    "codec-config difference between HQ controls"
                ),
                "applicability": {
                    "single_lever_causal_block_vq_contrast": list(controls),
                    "HQ": "contextual formal proposal row; not part of the single-lever contrast",
                    "AdaptiveHQ": (
                        "contextual formal proposal row; causal selector changes allocation "
                        "and is not part of the single-lever contrast"
                    ),
                },
            }
        by_model[str(model)] = {"rates": model_rates, "official_complete": True}
        sources.append(_source(path))
    if set(by_model) != set(expected_models) or len(paths) != len(expected_models):
        raise AggregateError(
            f"Mechanism-control model matrix incomplete: {sorted(set(expected_models) - set(by_model))}"
        )
    if len(input_manifests) != 1:
        raise AggregateError("Mechanism runs span multiple paper input manifests")
    commits.add(reference_ppl.get("commit"))
    if None in commits or len(commits) != 1:
        raise AggregateError(
            f"Mechanism and primary PPL span commits: {sorted(map(str, commits))}"
        )
    return {
        "benchmark": "HayateQuant block-VQ mechanism ablation",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "commit": next(iter(commits)),
        "models": by_model,
        "scope": "two fixed model families, complete WikiText-2 raw test",
        "proposal_rows_in_scope": list(PROPOSAL_METHODS),
        "primary_contrast": "HayateQuant-blockvq-matched vs HayateQuant-scalar-matched",
        "inference_scope": {
            "checkpoint_panel": "two_preselected_family_correlated_checkpoints_not_iid",
            "population_generalization": False,
            "reporting": "descriptive_geometric_mean_only",
            "uncertainty_interval": None,
            "directional_claim_allowed": False,
            "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
        },
        "compression_accounting": dict(PPL_COMPRESSION_ACCOUNTING),
        "teacher_forced_scoring_contract": {
            "protocol": PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": True,
        },
        "official_complete": True,
        "sources": sources,
    }


def _load_formal_long_results(
    root: Path, *, axis: str
) -> tuple[list[tuple[Path, dict, str, dict]], str]:
    paths = sorted(root.expanduser().resolve().rglob("results_*.json")) if root.is_dir() else []
    if not paths:
        raise AggregateError(f"No {axis} result JSON files found below {root}")
    rows = []
    commits = set()
    input_manifests = set()
    for path in paths:
        document = _load_object(path)
        if document.get("schema_version") != 2 or not document.get("run_fingerprint"):
            raise AggregateError(f"Legacy/unidentified {axis} result: {path}")
        provenance = _require_run_provenance(document, path=path, return_identity=True)
        commits.add(provenance["commit"])
        input_manifests.add(
            (provenance["input_manifest_path"], provenance["input_manifest_sha256"])
        )
        config = document.get("config")
        if not isinstance(config, dict):
            raise AggregateError(f"Missing {axis} scientific config in {path}")
        if config.get("skip_outliers") is not False:
            raise AggregateError(f"{axis} formal row used model-dependent layer skipping: {path}")
        if config.get("sink_tokens") != 0 or config.get("recent_tokens") != 0:
            raise AggregateError(f"{axis} formal row retained FP16 token windows: {path}")
        results = document.get("results")
        if not isinstance(results, dict) or len(results) != 1:
            raise AggregateError(f"Each formal {axis} file must contain one method: {path}")
        method, method_result = next(iter(results.items()))
        if method not in LONG_METHODS or not isinstance(method_result, dict):
            raise AggregateError(f"Unexpected {axis} method {method!r}: {path}")
        _require_formal_codec_recipe(
            method_result,
            model=str(document.get("model")),
            method=str(method),
            rate=None if method == "FullKV" else 3,
            path=path,
        )
        rows.append((path, document, method, method_result))
    if len(commits) != 1:
        raise AggregateError(f"{axis} shards span multiple commits: {sorted(commits)}")
    if len(input_manifests) != 1:
        raise AggregateError(f"{axis} shards span multiple paper input manifests")
    return rows, next(iter(commits))


def _ruler_mapping(value: Any, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise AggregateError(f"{label} must be a JSON object")
    return value


def _ruler_digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise AggregateError(f"{label} must be a 64-hex SHA-256 digest")
    return value.lower()


def _ruler_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AggregateError(f"{label} must be a non-empty file path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise AggregateError(f"{label} is missing: {path}")
    return path


def _ruler_assert_close(
    actual: Any, expected: float, *, label: str, tolerance: float = 1e-9
) -> float:
    number = _number(actual, label=label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AggregateError(f"{label} mismatch: stored={number!r}, recomputed={expected!r}")
    return number


def _ruler_sample_credit(task: str, prediction: str, references: Sequence[str]) -> float:
    lowered = prediction.lower()
    hits = [1.0 if reference.lower() in lowered else 0.0 for reference in references]
    return max(hits) if task.startswith("qa_") else math.fsum(hits) / len(hits)


def _load_ruler_source_rows(path: Path, *, context: int, label: str) -> dict:
    """Validate one pinned NeMo-Skills RULER JSONL file and retain grading identity."""
    rows = []
    template_lengths = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise AggregateError(f"Blank RULER JSONL row in {path}:{line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AggregateError(
                        f"Invalid RULER JSONL row in {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AggregateError(f"Non-object RULER row in {path}:{line_number}")
                required = {
                    "index",
                    "input",
                    "answer_prefix",
                    "outputs",
                    "length",
                    "length_w_model_temp",
                }
                if not required.issubset(row):
                    raise AggregateError(
                        f"Raw pinned RULER row is missing {sorted(required - set(row))} "
                        f"in {path}:{line_number}"
                    )
                index = row["index"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise AggregateError(f"Invalid RULER index in {path}:{line_number}")
                if not isinstance(row["input"], str) or not row["input"]:
                    raise AggregateError(f"Invalid RULER input in {path}:{line_number}")
                if not isinstance(row["answer_prefix"], str):
                    raise AggregateError(f"Invalid RULER answer_prefix in {path}:{line_number}")
                references = row["outputs"]
                if (
                    not isinstance(references, list)
                    or not references
                    or any(not isinstance(item, str) or not item for item in references)
                ):
                    raise AggregateError(f"Invalid RULER references in {path}:{line_number}")
                released_length = row["length"]
                templated_length = row["length_w_model_temp"]
                if (
                    isinstance(released_length, bool)
                    or not isinstance(released_length, int)
                    or released_length <= 0
                    or isinstance(templated_length, bool)
                    or not isinstance(templated_length, int)
                    or templated_length <= 0
                    or templated_length > context
                ):
                    raise AggregateError(
                        f"Invalid/out-of-range RULER released length in {path}:{line_number}"
                    )
                rows.append(
                    {
                        "index": index,
                        "references": list(references),
                        "released_length": released_length,
                        "released_length_with_template": templated_length,
                    }
                )
                template_lengths.append(templated_length)
    except OSError as exc:
        raise AggregateError(f"Cannot read RULER task file {path}: {exc}") from exc
    if len(rows) != RULER_ROWS_PER_TASK:
        raise AggregateError(f"{label} has {len(rows)} rows, expected {RULER_ROWS_PER_TASK}")
    return {
        "path": path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": rows,
        "length_w_model_temp_min": min(template_lengths),
        "length_w_model_temp_max": max(template_lengths),
    }


def _ruler_model_metadata(manifest: dict, *, label: str) -> dict:
    model = _ruler_mapping(manifest.get("model"), label=f"{label} model")
    raw_model_path = model.get("path")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise AggregateError(f"{label} model.path is missing")
    model_path = Path(raw_model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise AggregateError(f"{label} model.path is not a local model directory: {model_path}")
    metadata = _ruler_mapping(model.get("metadata_files"), label=f"{label} model.metadata_files")
    required_metadata = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required_metadata.issubset(metadata):
        raise AggregateError(
            f"{label} model metadata is missing {sorted(required_metadata - set(metadata))}"
        )
    allowed_metadata = required_metadata
    if not set(metadata).issubset(allowed_metadata):
        raise AggregateError(f"{label} declares unsupported model metadata files")
    for filename, raw_record in metadata.items():
        record = _ruler_mapping(raw_record, label=f"{label} metadata {filename}")
        metadata_path = model_path / filename
        if not metadata_path.is_file():
            raise AggregateError(f"{label} model metadata is missing: {metadata_path}")
        expected_bytes = _integer(
            record.get("bytes"), label=f"{label} metadata {filename} bytes", minimum=1
        )
        expected_digest = _ruler_digest(
            record.get("sha256"), label=f"{label} metadata {filename} sha256"
        )
        if (
            metadata_path.stat().st_size != expected_bytes
            or _sha256(metadata_path) != expected_digest
        ):
            raise AggregateError(f"{label} model metadata changed: {metadata_path}")

    config = _load_object(model_path / "config.json")
    text_config = config.get("text_config")
    capacity_config = text_config if isinstance(text_config, dict) else config
    capacity = _integer(
        capacity_config.get("max_position_embeddings"),
        label=f"{label} model max_position_embeddings",
        minimum=1,
    )
    tokenizer_config = _load_object(model_path / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if chat_template is not None and not isinstance(chat_template, str):
        raise AggregateError(f"{label} tokenizer chat_template must be a string or null")
    chat_template_sha256 = (
        hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if isinstance(chat_template, str)
        else None
    )
    generation_config = _load_object(model_path / "generation_config.json")
    raw_eos = generation_config.get("eos_token_id")
    raw_eos_values = raw_eos if isinstance(raw_eos, list) else [raw_eos]
    generation_eos_token_ids = []
    for token_id in raw_eos_values:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise AggregateError(
                f"{label} generation_config.eos_token_id must contain non-negative integers"
            )
        if token_id not in generation_eos_token_ids:
            generation_eos_token_ids.append(token_id)
    if not generation_eos_token_ids:
        raise AggregateError(f"{label} generation_config.eos_token_id must not be empty")
    revision = model.get("revision")
    snapshots = model.get("snapshot_revisions")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or not isinstance(snapshots, list)
        or revision not in snapshots
        or model.get("chat_template_sha256") != chat_template_sha256
        or model.get("generation_eos_token_ids") != generation_eos_token_ids
        or model.get("generation_eos_token_ids_source") != "model.generation_config.eos_token_id"
    ):
        raise AggregateError(f"{label} model revision/chat/EOS provenance mismatch")
    return {
        "path": model_path,
        "max_position_embeddings": capacity,
        "chat_template_sha256": chat_template_sha256,
        "generation_eos_token_ids": generation_eos_token_ids,
    }


def _validated_ruler_manifest(
    ruler: dict,
    *,
    model: str,
    context: int,
    expected_contexts: Sequence[int],
    label: str,
    cache: dict,
) -> dict:
    """Validate the immutable generation manifest and current length's files."""
    envelope = _ruler_mapping(
        ruler.get("generation_manifest"), label=f"{label} generation_manifest"
    )
    manifest_path = _ruler_file(envelope.get("path"), label=f"{label} generation manifest path")
    declared_digest = _ruler_digest(
        envelope.get("sha256"), label=f"{label} generation manifest sha256"
    )
    if _sha256(manifest_path) != declared_digest:
        raise AggregateError(f"{label} generation manifest SHA-256 mismatch")
    actual_manifest = _load_object(manifest_path)
    embedded_manifest = _ruler_mapping(
        envelope.get("content"), label=f"{label} embedded generation manifest"
    )
    if embedded_manifest != actual_manifest:
        raise AggregateError(f"{label} embedded generation manifest differs from its file")

    cache_key = (model, declared_digest)
    cached = cache.get(cache_key)
    if cached is None:
        if actual_manifest.get("schema_version") != 1:
            raise AggregateError(f"{label} RULER generation manifest schema must be 1")
        if actual_manifest.get("benchmark") != RULER_PROTOCOL_LABEL:
            raise AggregateError(f"{label} generation manifest is not {RULER_PROTOCOL_LABEL}")
        sources = _ruler_mapping(actual_manifest.get("sources"), label=f"{label} manifest sources")
        if (
            sources.get("ruler_commit") != RULER_SOURCE_COMMIT
            or sources.get("nemo_skills_commit") != NEMO_SKILLS_RULER_SOURCE_COMMIT
        ):
            raise AggregateError(f"{label} generation manifest protocol commit drift")
        protocol = _ruler_mapping(
            actual_manifest.get("protocol"), label=f"{label} manifest protocol"
        )
        expected_contexts = tuple(expected_contexts)
        if (
            tuple(protocol.get("tasks") or ()) != RULER_TASKS
            or protocol.get("num_samples_per_task") != RULER_ROWS_PER_TASK
            or protocol.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL
            or protocol.get("prepare_for_nemo_skills") is not True
            or protocol.get("tokenizer_type") != "hf"
            or protocol.get("random_seed") != 42
            or protocol.get("chat_template_reserve") != 50
            or tuple(protocol.get("lengths") or ()) != expected_contexts
        ):
            raise AggregateError(
                f"{label} generation manifest is not the pinned NeMo-Skills protocol"
            )
        groups = _ruler_mapping(actual_manifest.get("files"), label=f"{label} manifest files")
        if set(groups) != {str(item) for item in expected_contexts}:
            raise AggregateError(f"{label} manifest length groups are incomplete")
        for length, group in groups.items():
            if not isinstance(group, dict) or set(group) != set(RULER_TASKS):
                raise AggregateError(f"{label} manifest {length} task group is incomplete")
        cached = {
            "path": manifest_path,
            "content": actual_manifest,
            "model": _ruler_model_metadata(actual_manifest, label=label),
            "groups": {},
        }
        cache[cache_key] = cached
    elif cached["path"] != manifest_path or cached["content"] != actual_manifest:
        raise AggregateError(f"{label} reuses a manifest digest with inconsistent identity")

    if context not in cached["groups"]:
        group = actual_manifest["files"][str(context)]
        validated = {}
        for task in RULER_TASKS:
            entry = _ruler_mapping(group[task], label=f"{label} manifest {context}/{task}")
            expected_path = (manifest_path.parent / str(context) / task / "test.jsonl").resolve()
            declared_path = _ruler_file(
                entry.get("path"), label=f"{label} manifest {context}/{task} path"
            )
            if declared_path != expected_path:
                raise AggregateError(
                    f"{label} RULER task path is not the manifest-local canonical path: "
                    f"{declared_path}"
                )
            source = _load_ruler_source_rows(
                declared_path, context=context, label=f"{label} {context}/{task}"
            )
            expected_digest = _ruler_digest(
                entry.get("sha256"), label=f"{label} manifest {context}/{task} sha256"
            )
            for field, actual in (
                ("bytes", source["bytes"]),
                ("rows", len(source["rows"])),
                ("length_w_model_temp_min", source["length_w_model_temp_min"]),
                ("length_w_model_temp_max", source["length_w_model_temp_max"]),
            ):
                if entry.get(field) != actual:
                    raise AggregateError(f"{label} manifest {context}/{task} {field} mismatch")
            if source["sha256"] != expected_digest:
                raise AggregateError(f"{label} manifest {context}/{task} data hash mismatch")
            validated[task] = source
        cached["groups"][context] = validated
    return {
        "sha256": declared_digest,
        "path": manifest_path,
        "model": cached["model"],
        "tasks": cached["groups"][context],
    }


def _validate_ruler_result(
    ruler: dict,
    *,
    path: Path,
    document: dict,
    model: str,
    method: str,
    context: int,
    expected_contexts: Sequence[int],
    manifest_cache: dict,
) -> dict:
    label = f"{path} RULER"
    if (
        ruler.get("schema_version") != RULER_RESULT_SCHEMA_VERSION
        or ruler.get("benchmark") != RULER_PROTOCOL_LABEL
        or ruler.get("metric") != "13-task macro accuracy"
        or ruler.get("evaluation_scope") != RULER_COMPLETE_EVALUATION_SCOPE
        or ruler.get("protocol_complete") is not True
        or ruler.get("canonical_nvidia_full_eval") is not False
        or ruler.get("canonical_nvidia_samples_per_task") != RULER_CANONICAL_NVIDIA_ROWS_PER_TASK
        or ruler.get("protocol_scope") != RULER_PROTOCOL_SCOPE
        or ruler.get("reference_model_wrapper_exact") is not False
        or ruler.get("samples_per_task") != RULER_ROWS_PER_TASK
        or ruler.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL
        or ruler.get("start") != 0
        or ruler.get("limit_per_task") is not None
        or ruler.get("example_cache_lifetime") != PER_EXAMPLE_CACHE_LIFETIME
        or ruler.get("cross_iteration_tensor_retention")
        != PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
    ):
        raise AggregateError(f"{label} result schema/scope is not the complete pinned protocol")
    if tuple(ruler.get("tasks") or ()) != RULER_TASKS:
        raise AggregateError(f"{label} task order/set is not the pinned 13-task protocol")
    expected_n = len(RULER_TASKS) * RULER_ROWS_PER_TASK
    if _integer(ruler.get("n"), label=f"{label} n", minimum=1) != expected_n:
        raise AggregateError(f"{label} n must be {expected_n}")

    config = _ruler_mapping(document.get("config"), label=f"{label} scientific config")
    manifest = _validated_ruler_manifest(
        ruler,
        model=model,
        context=context,
        expected_contexts=expected_contexts,
        label=label,
        cache=manifest_cache,
    )
    expected_data_root = manifest["path"].parent / str(context)
    config_manifest = _ruler_file(
        config.get("official_ruler_manifest_path"),
        label=f"{label} config manifest path",
    )
    raw_data_root = config.get("official_ruler_data_root")
    raw_model_path = config.get("model_path")
    if not isinstance(raw_data_root, str) or not isinstance(raw_model_path, str):
        raise AggregateError(f"{label} config omits model/data paths")
    if (
        config.get("run_official_ruler_v1") is not True
        or config.get("official_ruler_tasks") != []
        or config.get("official_ruler_start") != 0
        or config.get("official_ruler_limit") is not None
        or config.get("official_ruler_max_context") != context
        or Path(raw_data_root).expanduser().resolve() != expected_data_root.resolve()
        or config_manifest != manifest["path"]
        or Path(raw_model_path).expanduser().resolve() != manifest["model"]["path"]
    ):
        raise AggregateError(f"{label} scientific config does not match the evaluated protocol")

    protocol = _ruler_mapping(ruler.get("protocol"), label=f"{label} protocol")
    eos_token_ids = protocol.get("eos_token_ids")
    if (
        protocol.get("ruler_repository") != "https://github.com/NVIDIA/RULER"
        or protocol.get("ruler_source_commit") != RULER_SOURCE_COMMIT
        or protocol.get("nemo_skills_repository") != "https://github.com/NVIDIA-NeMo/Skills"
        or protocol.get("nemo_skills_source_commit") != NEMO_SKILLS_RULER_SOURCE_COMMIT
        or protocol.get("task_set") != "RULER v1 13-task synthetic setup"
        or protocol.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL
        or protocol.get("sequence_length") != context
        or protocol.get("prompt") != RULER_PROMPT_DESCRIPTION
        or protocol.get("chat_template_kwargs") != RULER_CHAT_TEMPLATE_KWARGS
        or protocol.get("generation") != RULER_GENERATION_DESCRIPTION
        or protocol.get("max_new_tokens") != RULER_MAX_NEW
        or protocol.get("eos_token_id_source") != "model.generation_config.eos_token_id"
        or eos_token_ids != manifest["model"]["generation_eos_token_ids"]
        or not isinstance(eos_token_ids, list)
        or not eos_token_ids
        or len(set(eos_token_ids)) != len(eos_token_ids)
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in eos_token_ids
        )
        or protocol.get("stop_conditions")
        != ["any_generation_config_eos_token_id", "max_new_tokens"]
        or protocol.get("scoring") != RULER_SCORING_DESCRIPTION
        or protocol.get("truncation") != "none; context overflow is fatal"
        or protocol.get("context_semantics") != RULER_CONTEXT_SEMANTICS
    ):
        raise AggregateError(f"{label} protocol metadata drifted from the pinned implementation")
    tokenizer_path = protocol.get("tokenizer_name_or_path")
    if (
        not isinstance(tokenizer_path, str)
        or Path(tokenizer_path).expanduser().resolve() != manifest["model"]["path"]
        or protocol.get("chat_template_sha256") != manifest["model"]["chat_template_sha256"]
    ):
        raise AggregateError(f"{label} tokenizer/chat-template provenance mismatch")
    wrappers = protocol.get("prompt_wrappers")
    expected_wrapper = (
        [
            "generic user message; tokenizer.apply_chat_template with "
            "enable_thinking=false when supported"
        ]
        if manifest["model"]["chat_template_sha256"] is not None
        else ["base/raw released prompt (tokenizer has no chat template)"]
    )
    if wrappers != expected_wrapper:
        raise AggregateError(f"{label} generic/default prompt wrapper mismatch")

    dataset_files = _ruler_mapping(ruler.get("dataset_files"), label=f"{label} dataset_files")
    if set(dataset_files) != set(RULER_TASKS):
        raise AggregateError(f"{label} dataset file set is incomplete")
    for task in RULER_TASKS:
        record = _ruler_mapping(dataset_files[task], label=f"{label} dataset {task}")
        source = manifest["tasks"][task]
        declared_path = _ruler_file(record.get("path"), label=f"{label} dataset {task} path")
        if (
            declared_path != source["path"]
            or record.get("sha256") != source["sha256"]
            or record.get("rows") != RULER_ROWS_PER_TASK
            or record.get("row_format") != "nvidia_ruler_input_plus_answer_prefix"
        ):
            raise AggregateError(f"{label} dataset metadata mismatch for {task}")
    combined_digest = hashlib.sha256(
        "".join(
            f"{task}:{manifest['tasks'][task]['sha256']}\n" for task in sorted(RULER_TASKS)
        ).encode("utf-8")
    ).hexdigest()
    if ruler.get("combined_dataset_sha256") != combined_digest:
        raise AggregateError(f"{label} combined dataset SHA-256 mismatch")

    per_task = _ruler_mapping(ruler.get("per_task"), label=f"{label} per_task")
    input_summaries = _ruler_mapping(ruler.get("input_tokens"), label=f"{label} input_tokens")
    examples = _ruler_mapping(ruler.get("per_example"), label=f"{label} per_example")
    for name, value in (
        ("per_task", per_task),
        ("input_tokens", input_summaries),
        ("per_example", examples),
    ):
        if set(value) != set(RULER_TASKS):
            raise AggregateError(f"{label} {name} task set is incomplete")

    model_context = _ruler_mapping(ruler.get("model_context"), label=f"{label} model_context")
    capacity = manifest["model"]["max_position_embeddings"]
    allowed_capacity_sources = {
        "model.config.max_position_embeddings",
        "model.config.get_text_config().max_position_embeddings",
    }
    if (
        model_context.get("source") not in allowed_capacity_sources
        or model_context.get("max_position_embeddings") != capacity
        or model_context.get("named_ruler_sequence_limit") != context
        or model_context.get("capacity_check") != RULER_CAPACITY_CHECK
        or context > capacity
    ):
        raise AggregateError(f"{label} effective model-context metadata mismatch")

    task_accuracies = []
    all_input_lengths = []
    all_generated_lengths = []
    all_reserved_totals = []
    task_audit = {}
    all_compression_records = []
    compression_ledger = []
    for task in RULER_TASKS:
        stored_task = _ruler_mapping(per_task[task], label=f"{label} per_task {task}")
        records = examples[task]
        if not isinstance(records, list) or len(records) != RULER_ROWS_PER_TASK:
            raise AggregateError(f"{label} {task} must retain exactly 100 per-example records")
        source_rows = manifest["tasks"][task]["rows"]
        task_compression = _validated_long_compression_sufficient_statistics(
            stored_task,
            records,
            method=method,
            label=f"{label} {task}",
        )
        all_compression_records.extend(records)
        max_new = RULER_MAX_NEW[task]
        credits = []
        task_input_lengths = []
        for row_position, (record, source_row) in enumerate(zip(records, source_rows)):
            record = _ruler_mapping(record, label=f"{label} {task} per-example row {row_position}")
            if (
                record.get("row_position") != row_position
                or record.get("index") != source_row["index"]
                or record.get("released_length") != source_row["released_length"]
                or record.get("released_length_with_template")
                != source_row["released_length_with_template"]
                or record.get("references") != source_row["references"]
            ):
                raise AggregateError(f"{label} {task} source identity drift at row {row_position}")
            prediction = record.get("prediction")
            if not isinstance(prediction, str):
                raise AggregateError(f"{label} {task} prediction is not losslessly stored")
            input_length = _integer(
                record.get("input_tokens"),
                label=f"{label} {task} input_tokens row {row_position}",
                minimum=1,
            )
            generated_length = _integer(
                record.get("generated_tokens"),
                label=f"{label} {task} generated_tokens row {row_position}",
                minimum=1,
            )
            if (
                input_length + max_new > context
                or generated_length > max_new
                or record.get("max_new_tokens") != max_new
                or record.get("stop_token_ids") != eos_token_ids
                or record.get("prompt_truncated") is not False
                or input_length + max_new > capacity
            ):
                raise AggregateError(
                    f"{label} {task} row {row_position} violates the named RULER "
                    "sequence budget or model capacity"
                )
            credit = _ruler_sample_credit(task, prediction, source_row["references"])
            _ruler_assert_close(
                record.get("credit"),
                round(credit, 6),
                label=f"{label} {task} credit row {row_position}",
                tolerance=1e-12,
            )
            credits.append(credit)
            task_input_lengths.append(input_length)
            all_input_lengths.append(input_length)
            all_generated_lengths.append(generated_length)
            all_reserved_totals.append(input_length + max_new)
            compression_ledger.append(
                _long_compression_ledger_row(
                    record,
                    sample_id=f"{task}:{source_row['index']}",
                    input_tokens=input_length,
                    max_new_tokens=max_new,
                    generated_tokens=generated_length,
                )
            )

        credit_sum = math.fsum(credits)
        accuracy = credit_sum / RULER_ROWS_PER_TASK
        stderr = statistics.stdev(credits) / math.sqrt(len(credits))
        if (
            stored_task.get("n") != RULER_ROWS_PER_TASK
            or stored_task.get("max_new_tokens") != max_new
        ):
            raise AggregateError(f"{label} {task} denominator/max generation mismatch")
        _ruler_assert_close(
            stored_task.get("credit_sum"),
            round(credit_sum, 10),
            label=f"{label} {task} credit numerator",
            tolerance=1e-12,
        )
        _ruler_assert_close(
            stored_task.get("score"),
            round(100.0 * accuracy, 4),
            label=f"{label} {task} score",
            tolerance=1e-12,
        )
        _ruler_assert_close(
            stored_task.get("accuracy"),
            round(accuracy, 6),
            label=f"{label} {task} accuracy",
            tolerance=1e-12,
        )
        _ruler_assert_close(
            stored_task.get("accuracy_stderr"),
            round(stderr, 6),
            label=f"{label} {task} accuracy stderr",
            tolerance=1e-12,
        )
        mem_fraction = _number(
            stored_task.get("mem_fraction"), label=f"{label} {task} mem_fraction"
        )
        if mem_fraction > 1.0 + 1e-9:
            raise AggregateError(f"{label} {task} mem_fraction exceeds FullKV")
        summary = _ruler_mapping(
            input_summaries[task], label=f"{label} {task} input-token summary"
        )
        expected_summary = {
            "min": min(task_input_lengths),
            "median": round(float(statistics.median(task_input_lengths)), 1),
            "max": max(task_input_lengths),
        }
        if set(summary) != set(expected_summary):
            raise AggregateError(f"{label} {task} input-token summary schema mismatch")
        for field, expected_value in expected_summary.items():
            _ruler_assert_close(
                summary[field], expected_value, label=f"{label} {task} input {field}"
            )
        task_accuracies.append(accuracy)
        task_audit[task] = {
            "n": RULER_ROWS_PER_TASK,
            "credit_sum_unrounded": credit_sum,
            "accuracy_unrounded": accuracy,
            "score": round(100.0 * accuracy, 4),
            "compression_sufficient_statistics": task_compression,
        }

    task_accuracy_sum = math.fsum(task_accuracies)
    macro_accuracy = task_accuracy_sum / len(RULER_TASKS)
    sufficient = _ruler_mapping(
        ruler.get("score_sufficient_statistics"),
        label=f"{label} score sufficient statistics",
    )
    if sufficient.get("task_count") != len(RULER_TASKS):
        raise AggregateError(f"{label} macro task denominator mismatch")
    _ruler_assert_close(
        sufficient.get("task_accuracy_sum"),
        round(task_accuracy_sum, 10),
        label=f"{label} macro numerator",
        tolerance=1e-12,
    )
    displayed_score = _ruler_assert_close(
        ruler.get("score"),
        round(100.0 * macro_accuracy, 4),
        label=f"{label} macro score",
        tolerance=1e-12,
    )
    _ruler_assert_close(
        ruler.get("accuracy"),
        round(macro_accuracy, 6),
        label=f"{label} macro accuracy",
        tolerance=1e-12,
    )
    top_mem_fraction = _number(ruler.get("mem_fraction"), label=f"{label} mem_fraction")
    if top_mem_fraction > 1.0 + 1e-9:
        raise AggregateError(f"{label} mem_fraction exceeds FullKV")

    expected_context_summary = {
        "max_observed_prompt_tokens": max(all_input_lengths),
        "max_observed_prompt_plus_max_generation_tokens": max(all_reserved_totals),
        "max_observed_generated_tokens": max(all_generated_lengths),
    }
    for field, expected_value in expected_context_summary.items():
        if model_context.get(field) != expected_value:
            raise AggregateError(f"{label} model_context.{field} mismatch")

    compression = _validated_long_compression_sufficient_statistics(
        ruler,
        all_compression_records,
        method=method,
        label=label,
    )

    return {
        "score": displayed_score,
        "score_unrounded": 100.0 * macro_accuracy,
        "task_accuracy_sum_unrounded": task_accuracy_sum,
        "n": expected_n,
        "per_task": task_audit,
        "combined_dataset_sha256": combined_digest,
        "generation_manifest_sha256": manifest["sha256"],
        "generation_manifest_path": str(manifest["path"]),
        "generation_stop_protocol": {
            "eos_token_ids": list(eos_token_ids),
            "eos_token_id_source": protocol["eos_token_id_source"],
            "stop_conditions": list(protocol["stop_conditions"]),
        },
        "model_wrapper_protocol": {
            "prompt_wrappers": list(wrappers),
            "chat_template_kwargs": dict(protocol["chat_template_kwargs"]),
            "protocol_scope": ruler["protocol_scope"],
            "reference_model_wrapper_exact": False,
        },
        "model_max_position_embeddings": capacity,
        "max_observed_prompt_tokens": max(all_input_lengths),
        "max_observed_prompt_plus_max_generation_tokens": max(all_reserved_totals),
        "input_tokens": input_summaries,
        "audit_records_validated": expected_n,
        **compression,
        "compression_ledger": compression_ledger,
        "source": str(path.resolve()),
    }


def verify_ruler(
    root: Path,
    *,
    expected_models: Sequence[str] = LONG_MODELS,
    contexts: Sequence[int] = RULER_LENGTHS,
    methods: Sequence[str] = LONG_METHODS,
) -> dict:
    """Fail-closed verification of the complete pinned 100-sample protocol.

    The displayed score is never trusted: all 1,300 predictions are regraded
    against the hashed source JSONL rows and the 13-task macro is recomputed
    from unrounded per-example credit.
    """
    expected_models = tuple(expected_models)
    contexts = tuple(contexts)
    methods = tuple(methods)
    if (
        not expected_models
        or len(set(expected_models)) != len(expected_models)
        or not contexts
        or len(set(contexts)) != len(contexts)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in contexts
        )
        or not methods
        or len(set(methods)) != len(methods)
        or not set(methods).issubset(LONG_METHODS)
    ):
        raise AggregateError("Invalid expected RULER model/context/method matrix")

    rows, code_commit = _load_formal_long_results(root, axis="RULER")
    matrix = {}
    sources = []
    manifest_cache = {}
    manifest_hashes: dict[str, set[str]] = {model: set() for model in expected_models}
    capacities: dict[str, set[int]] = {model: set() for model in expected_models}
    dataset_hashes: dict[tuple[str, int], set[str]] = {}
    input_signatures: dict[tuple[str, int], set[str]] = {}
    generation_protocols: dict[str, set[str]] = {model: set() for model in expected_models}
    wrapper_protocols: dict[str, set[str]] = {model: set() for model in expected_models}
    for path, document, method, method_result in rows:
        model = document.get("model")
        ruler = method_result.get("ruler_v1")
        if model not in expected_models or method not in methods or not isinstance(ruler, dict):
            raise AggregateError(f"Malformed/unexpected RULER result in {path}")
        context = _integer(
            ruler.get("max_context_tokens"), label=f"{path} RULER context", minimum=1
        )
        if context not in contexts:
            raise AggregateError(f"Unexpected RULER context {context} in {path}")
        key = (str(model), context, method)
        if key in matrix:
            raise AggregateError(f"Duplicate RULER matrix point {key}")
        value = _validate_ruler_result(
            ruler,
            path=path,
            document=document,
            model=str(model),
            method=method,
            context=context,
            expected_contexts=contexts,
            manifest_cache=manifest_cache,
        )
        matrix[key] = value
        manifest_hashes[str(model)].add(value["generation_manifest_sha256"])
        capacities[str(model)].add(value["model_max_position_embeddings"])
        dataset_hashes.setdefault((str(model), context), set()).add(
            value["combined_dataset_sha256"]
        )
        input_signatures.setdefault((str(model), context), set()).add(
            json.dumps(value["input_tokens"], sort_keys=True, separators=(",", ":"))
        )
        generation_protocols[str(model)].add(
            json.dumps(
                value["generation_stop_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        wrapper_protocols[str(model)].add(
            json.dumps(
                value["model_wrapper_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        sources.append(_source(path))

    expected = {
        (model, context, method)
        for model in expected_models
        for context in contexts
        for method in methods
    }
    if set(matrix) != expected:
        raise AggregateError(
            "RULER matrix mismatch: "
            f"missing={sorted(expected - set(matrix))}, "
            f"unexpected={sorted(set(matrix) - expected)}"
        )
    for model in expected_models:
        if (
            len(manifest_hashes[model]) != 1
            or len(capacities[model]) != 1
            or len(generation_protocols[model]) != 1
            or len(wrapper_protocols[model]) != 1
        ):
            raise AggregateError(f"RULER manifest/model capacity changed across {model} runs")
        for context in contexts:
            if (
                len(dataset_hashes.get((model, context), set())) != 1
                or len(input_signatures.get((model, context), set())) != 1
            ):
                raise AggregateError(
                    f"RULER dataset/tokenized prompt lengths changed across {model}/{context} methods"
                )

    for model in expected_models:
        for context in contexts:
            _require_long_exact_payload_match(
                {method: matrix[(model, context, method)] for method in methods},
                label=f"RULER/{model}/{context}",
            )

    nested = {}
    for (model, context, method), value in sorted(matrix.items()):
        nested.setdefault(model, {}).setdefault(str(context), {})[method] = value
    return {
        "benchmark": RULER_PROTOCOL_LABEL,
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "schema_version": 1,
        "source_result_schema_version": RULER_RESULT_SCHEMA_VERSION,
        "aggregation": "regrade_1300_predictions_then_recompute_13_task_macro",
        "context_semantics": RULER_CONTEXT_SEMANTICS,
        "capacity_check": RULER_CAPACITY_CHECK,
        "protocol_scope": RULER_PROTOCOL_SCOPE,
        "reference_model_wrapper_exact": False,
        "canonical_nvidia_full_eval": False,
        "canonical_nvidia_samples_per_task": RULER_CANONICAL_NVIDIA_ROWS_PER_TASK,
        "samples_per_task": RULER_ROWS_PER_TASK,
        "sampling_protocol": RULER_SAMPLING_PROTOCOL,
        "commit": code_commit,
        "models": nested,
        "generation_stop_protocols": {
            model: json.loads(next(iter(generation_protocols[model]))) for model in expected_models
        },
        "model_wrapper_protocols": {
            model: json.loads(next(iter(wrapper_protocols[model]))) for model in expected_models
        },
        "protocol_complete": True,
        "audit_records_validated": len(matrix) * len(RULER_TASKS) * RULER_ROWS_PER_TASK,
        "sources": sources,
    }


def verify_longbench(
    root: Path,
    *,
    expected_models: Sequence[str] = LONG_MODELS,
    methods: Sequence[str] = LONG_METHODS,
    task_counts: Mapping[str, int] = LONGBENCH_COUNTS,
    task_max_new: Mapping[str, int] = LONGBENCH_MAX_NEW,
    dataset_revision: str = LONGBENCH_REVISION,
    protocol_commit: str = LONGBENCH_PROTOCOL_COMMIT,
    archive_sha256: str = LONGBENCH_ARCHIVE_SHA256,
    prompt_budgets: Mapping[str, Any] = LONGBENCH_PROMPT_BUDGETS,
    model_context_limits: Mapping[str, int] = LONGBENCH_MODEL_CONTEXT_LIMITS,
    expected_truncations: Mapping[str, Mapping[str, int]] = (LONGBENCH_EXPECTED_TRUNCATIONS),
    _archive_path_override: Path | None = None,
) -> dict:
    """Verify and aggregate complete one-task LongBench-E result shards.

    Every generated prediction is regraded against the corresponding row in
    the SHA-256-pinned LongBench ``data.zip`` with the official task metric.
    Stored answers, classes, lengths, per-example scores, and aggregate
    sufficient statistics are only accepted after they match that independent
    replay.  ``_archive_path_override`` exists solely for reduced unit/custom
    protocol fixtures; the formal CLI always resolves the archive through the
    immutable paper input manifest embedded in worker provenance.
    """
    rows, code_commit = _load_formal_long_results(root, axis="LongBench-E")
    matrix = {}
    sources = []
    expected_models = tuple(expected_models)
    methods = tuple(methods)
    tasks = tuple(task_counts)
    if set(tasks) != set(task_max_new):
        raise AggregateError("LongBench task counts/max-generation maps disagree")
    if set(expected_models) != set(prompt_budgets) or set(expected_models) != set(
        model_context_limits
    ):
        raise AggregateError("LongBench model prompt-budget/context maps are incomplete")
    pinned_task_rows, dataset_provenance = _load_pinned_longbench_rows(
        rows,
        tasks=tasks,
        task_counts=task_counts,
        dataset_revision=dataset_revision,
        archive_sha256=archive_sha256,
        archive_override=_archive_path_override,
    )
    for path, document, method, method_result in rows:
        model = document.get("model")
        value = method_result.get("longbench")
        if model not in expected_models or method not in methods or not isinstance(value, dict):
            raise AggregateError(f"Malformed LongBench-E result in {path}")
        shard_tasks = value.get("tasks")
        if (
            not isinstance(shard_tasks, list)
            or len(shard_tasks) != 1
            or shard_tasks[0] not in task_counts
        ):
            raise AggregateError(f"Each LongBench-E shard must contain one pinned task: {path}")
        task = shard_tasks[0]
        if value.get("limit_per_task") is not None:
            raise AggregateError(f"LongBench-E task shard is row-capped in {path}")
        if value.get("released_data_complete") is not False:
            raise AggregateError(f"Task-subset LongBench-E shard has wrong scope label in {path}")
        per_task = value.get("per_task")
        if not isinstance(per_task, dict) or set(per_task) != {task}:
            raise AggregateError(f"Wrong LongBench-E per-task payload in {path}")
        per_task_n = value.get("per_task_n")
        if not isinstance(per_task_n, dict) or set(per_task_n) != {task}:
            raise AggregateError(f"Missing LongBench-E exact per-task denominator in {path}")
        task_dataset_sizes = value.get("task_dataset_sizes")
        if not isinstance(task_dataset_sizes, dict) or set(task_dataset_sizes) != {task}:
            raise AggregateError(f"Missing LongBench-E pinned task size in {path}")
        archive_sha = value.get("data_zip_sha256")
        if (
            value.get("dataset_revision") != dataset_revision
            or not isinstance(archive_sha, str)
            or archive_sha.lower() != archive_sha256.lower()
        ):
            raise AggregateError(f"Unpinned LongBench-E dataset in {path}")
        if value.get("protocol_commit") != protocol_commit:
            raise AggregateError(f"Wrong LongBench-E protocol commit in {path}")
        if (
            value.get("protocol_scope") != LONGBENCH_PROTOCOL_SCOPE
            or value.get("reference_model_wrapper_exact") is not False
            or value.get("example_cache_lifetime") != PER_EXAMPLE_CACHE_LIFETIME
            or value.get("cross_iteration_tensor_retention")
            != PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ):
            raise AggregateError(f"Wrong LongBench-E protocol scope in {path}")
        n = _integer(value.get("n"), label=f"{path} LongBench n", minimum=1)
        if n != task_counts[task]:
            raise AggregateError(
                f"LongBench-E {task} row count {n}, expected {task_counts[task]} in {path}"
            )
        if per_task_n[task] != n or task_dataset_sizes[task] != n:
            raise AggregateError(f"LongBench-E task denominators disagree in {path}")
        score_sums = value.get("per_task_score_sum")
        if not isinstance(score_sums, dict) or set(score_sums) != {task}:
            raise AggregateError(f"Missing LongBench-E unrounded score numerator in {path}")
        score_sum = _number(score_sums[task], label=f"{path} LongBench unrounded score sum")
        if score_sum > 100.0 * n:
            raise AggregateError(f"LongBench-E score numerator exceeds 100*n in {path}")
        raw_mean = score_sum / n
        displayed = _number(per_task[task], label=f"{path} LongBench displayed score")
        if abs(round(raw_mean, 2) - displayed) > 1e-9:
            raise AggregateError(f"LongBench-E unrounded numerator/display mismatch in {path}")
        if abs(_number(value.get("score"), label=f"{path} LongBench score") - displayed) > 1e-9:
            raise AggregateError(f"LongBench-E one-task macro/display mismatch in {path}")
        model_budget_plan = prompt_budgets[str(model)]
        expected_prompt_budget = int(
            model_budget_plan[task]
            if isinstance(model_budget_plan, Mapping)
            else model_budget_plan
        )
        config = document.get("config")
        if (
            not isinstance(config, dict)
            or config.get("longbench_max_length") != expected_prompt_budget
        ):
            raise AggregateError(f"Wrong LongBench-E prompt budget in {path}")
        if value.get("max_length") != expected_prompt_budget:
            raise AggregateError(f"LongBench-E result omitted/wrong prompt budget in {path}")
        prompt_protocol = value.get("prompt_protocol")
        generation = value.get("generation")
        eos_token_ids = generation.get("eos_token_ids") if isinstance(generation, dict) else None
        samsum_newline_token_id = (
            generation.get("samsum_newline_token_id") if isinstance(generation, dict) else None
        )
        if (
            value.get("score_protocol") != "derived_unweighted_macro_over_task_scores"
            or value.get("by_length_protocol")
            != "derived_unweighted_macro_over_official_per_task_bucket_means"
            or not isinstance(prompt_protocol, dict)
            or prompt_protocol.get("raw_prompt") != "official dataset2prompt formatting"
            or prompt_protocol.get("truncation")
            != "official middle truncation before chat wrapping"
            or prompt_protocol.get("chat_wrapper") != LONGBENCH_CHAT_WRAPPER_SCOPE
            or prompt_protocol.get("no_chat_tasks") != sorted(LONGBENCH_NO_CHAT_TASKS)
            or not isinstance(generation, dict)
            or generation.get("do_sample") is not False
            or generation.get("num_beams") != 1
            or generation.get("sampling_seed") is not None
            or generation.get("official_reference_global_seed") != 42
            or generation.get("max_new_tokens") != "official dataset2maxlen per task"
            or generation.get("eos_token_id_source") != "model.generation_config.eos_token_id"
            or not isinstance(eos_token_ids, list)
            or not eos_token_ids
            or len(set(eos_token_ids)) != len(eos_token_ids)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in eos_token_ids
            )
            or isinstance(samsum_newline_token_id, bool)
            or not isinstance(samsum_newline_token_id, int)
            or samsum_newline_token_id < 0
            or generation.get("stop_conditions")
            != [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
                "samsum_newline_token_id_for_samsum_only",
            ]
        ):
            raise AggregateError(f"Wrong LongBench-E prompt/metric/generation protocol in {path}")
        expected_stop_token_ids = list(eos_token_ids)
        if task == "samsum" and samsum_newline_token_id not in expected_stop_token_ids:
            expected_stop_token_ids.append(samsum_newline_token_id)
        records_by_task = value.get("per_example")
        if not isinstance(records_by_task, dict) or set(records_by_task) != {task}:
            raise AggregateError(f"Missing LongBench-E regradable per-example payload in {path}")
        records = records_by_task[task]
        if not isinstance(records, list) or len(records) != n:
            raise AggregateError(f"Wrong LongBench-E per-example record count in {path}")
        compression = _validated_long_compression_sufficient_statistics(
            value,
            records,
            method=method,
            label=f"{path} LongBench-E/{task}",
        )
        per_task_compression = value.get("per_task_compression_sufficient_statistics")
        if (
            not isinstance(per_task_compression, Mapping)
            or set(per_task_compression) != {task}
            or per_task_compression[task] != compression
        ):
            raise AggregateError(f"LongBench-E per-task compression aggregate changed in {path}")
        score_pct_values: list[float] = []
        bucket_scores: dict[str, list[float]] = {
            "0-4k": [],
            "4-8k": [],
            "8k+": [],
        }
        record_truncated = 0
        for expected_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise AggregateError(f"Malformed LongBench-E example {expected_index} in {path}")
            if record.get("row_index") != expected_index:
                raise AggregateError(f"LongBench-E row indexes are not exact task order in {path}")
            if not isinstance(record.get("prediction"), str):
                raise AggregateError(f"LongBench-E prediction is not losslessly stored in {path}")
            generated_text = record.get("generated_text")
            if not isinstance(generated_text, str):
                raise AggregateError(f"LongBench-E full generation is not retained in {path}")
            expected_prediction = (
                generated_text.lstrip("\n").split("\n")[0]
                if task in {"trec", "triviaqa", "samsum"}
                else generated_text
            )
            if record["prediction"] != expected_prediction:
                raise AggregateError(f"LongBench-E task post-processing mismatch in {path}")
            generated_tokens = _integer(
                record.get("generated_tokens"),
                label=f"{path} LongBench generated tokens",
                minimum=1,
            )
            if generated_tokens > task_max_new[task]:
                raise AggregateError(f"LongBench-E generation exceeded task budget in {path}")
            if record.get("stop_token_ids") != expected_stop_token_ids:
                raise AggregateError(f"LongBench-E stop-token protocol changed in {path}")
            raw_prompt_tokens = _integer(
                record.get("raw_prompt_tokens"),
                label=f"{path} LongBench raw prompt tokens",
                minimum=1,
            )
            input_tokens = _integer(
                record.get("input_tokens"),
                label=f"{path} LongBench model input tokens",
                minimum=1,
            )
            if (
                record.get("raw_prompt_token_budget") != expected_prompt_budget
                or record.get("model_input_tokens") != input_tokens
                or record.get("middle_truncated")
                is not (raw_prompt_tokens > expected_prompt_budget)
                or record.get("chat_wrapper_used") is not (task not in LONGBENCH_NO_CHAT_TASKS)
                or input_tokens + task_max_new[task] > int(model_context_limits[str(model)])
            ):
                raise AggregateError(f"LongBench-E truncation/context audit mismatch in {path}")
            record_truncated += int(record["middle_truncated"])
            source_row = pinned_task_rows[task][expected_index]
            source_answers_raw = source_row.get("answers")
            source_answers = (
                list(source_answers_raw)
                if isinstance(source_answers_raw, list)
                else [source_answers_raw]
            )
            if (
                not isinstance(source_row.get("context"), str)
                or not isinstance(source_row.get("input", ""), str)
                or not source_answers
                or not all(isinstance(answer, str) for answer in source_answers)
                or "length" not in source_row
            ):
                raise AggregateError(
                    f"Pinned LongBench-E source row is malformed for {task}/{expected_index}"
                )
            answers = record.get("answers")
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(answer, str) for answer in answers)
                or answers != source_answers
            ):
                raise AggregateError(
                    f"LongBench-E grading answers differ from pinned row in {path}"
                )
            if "all_classes" not in record or record.get("all_classes") != source_row.get(
                "all_classes"
            ):
                raise AggregateError(
                    f"LongBench-E classification metadata differs from pinned row in {path}"
                )
            original_length = _integer(
                record.get("original_length"),
                label=f"{path} LongBench original length row {expected_index}",
            )
            source_length = _integer(
                source_row.get("length"),
                label=f"{path} pinned LongBench original length row {expected_index}",
            )
            if original_length != source_length:
                raise AggregateError(f"LongBench-E length differs from pinned row in {path}")
            expected_bucket = (
                "0-4k" if original_length < 4000 else "4-8k" if original_length < 8000 else "8k+"
            )
            if record.get("length_bucket") != expected_bucket:
                raise AggregateError(f"LongBench-E length bucket mismatch in {path}")
            _integer(
                record.get("input_tokens"),
                label=f"{path} LongBench input tokens row {expected_index}",
                minimum=1,
            )
            if record.get("max_new_tokens") != task_max_new[task]:
                raise AggregateError(f"LongBench-E task max generation changed in {path}")
            score = _number(
                record.get("score"), label=f"{path} LongBench raw score row {expected_index}"
            )
            score_pct = _number(
                record.get("score_pct"),
                label=f"{path} LongBench percent score row {expected_index}",
            )
            recomputed_score = max(
                _recompute_longbench_score(
                    task,
                    record["prediction"],
                    gold,
                    all_classes=source_row.get("all_classes"),
                )
                for gold in source_answers
            )
            recomputed_pct = 100.0 * recomputed_score
            if (
                score > 1.0
                or score_pct > 100.0
                or not math.isclose(score, recomputed_score, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(score_pct, recomputed_pct, rel_tol=0.0, abs_tol=1e-10)
            ):
                raise AggregateError(
                    f"LongBench-E stored score differs from official metric replay in {path}"
                )
            score_pct_values.append(recomputed_pct)
            bucket_scores[expected_bucket].append(recomputed_pct)
        if value.get("truncated_examples") != record_truncated or value.get(
            "truncated_per_task"
        ) != {task: record_truncated}:
            raise AggregateError(f"LongBench-E truncation summary mismatch in {path}")
        record_sum = math.fsum(score_pct_values)
        if abs(record_sum - score_sum) > 1e-8:
            raise AggregateError(
                f"LongBench-E numerator does not match per-example scores in {path}"
            )
        by_length_n = value.get("by_length_n")
        by_length = value.get("by_length")
        if (
            not isinstance(by_length_n, dict)
            or set(by_length_n) != set(bucket_scores)
            or not isinstance(by_length, dict)
            or set(by_length) != set(bucket_scores)
        ):
            raise AggregateError(f"Malformed LongBench-E length breakdown in {path}")
        for bucket, bucket_values in bucket_scores.items():
            if by_length_n[bucket] != len(bucket_values):
                raise AggregateError(f"LongBench-E length denominator mismatch in {path}")
            raw_bucket = math.fsum(bucket_values) / len(bucket_values) if bucket_values else 0.0
            displayed_bucket = _number(by_length[bucket], label=f"{path} LongBench {bucket} score")
            if abs(round(raw_bucket, 2) - displayed_bucket) > 1e-9:
                raise AggregateError(f"LongBench-E length score mismatch in {path}")
        key = (str(model), method, task)
        if key in matrix:
            raise AggregateError(f"Duplicate LongBench-E point {key}")
        matrix[key] = {
            "score_sum_unrounded": score_sum,
            "score_unrounded": raw_mean,
            "score_displayed": displayed,
            "n": n,
            "by_length": value.get("by_length"),
            "by_length_n": value.get("by_length_n"),
            "data_zip_sha256": archive_sha,
            "audit_records_validated": n,
            "truncated_examples": record_truncated,
            "prompt_budget": expected_prompt_budget,
            "prompt_audit_signature": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "row_index": record["row_index"],
                            "raw_prompt_tokens": record["raw_prompt_tokens"],
                            "model_input_tokens": record["model_input_tokens"],
                            "middle_truncated": record["middle_truncated"],
                        }
                        for record in records
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "generation_stop_protocol": {
                "eos_token_ids": list(eos_token_ids),
                "eos_token_id_source": generation["eos_token_id_source"],
                "samsum_newline_token_id": samsum_newline_token_id,
                "stop_conditions": list(generation["stop_conditions"]),
            },
            **compression,
            "compression_ledger": [
                _long_compression_ledger_row(
                    record,
                    sample_id=f"{task}:{record['row_index']}",
                    input_tokens=int(record["input_tokens"]),
                    max_new_tokens=int(record["max_new_tokens"]),
                    generated_tokens=int(record["generated_tokens"]),
                )
                for record in records
            ],
            "source": str(path.resolve()),
        }
        sources.append(_source(path))
    expected = {
        (model, method, task) for model in expected_models for method in methods for task in tasks
    }
    if set(matrix) != expected:
        raise AggregateError(
            f"LongBench-E matrix incomplete: missing={sorted(expected - set(matrix))}"
        )
    for model in expected_models:
        for task in tasks:
            _require_long_exact_payload_match(
                {method: matrix[(model, method, task)] for method in methods},
                label=f"LongBench-E/{model}/{task}",
            )
    for model in expected_models:
        for task in tasks:
            signatures = {
                matrix[(model, method, task)]["prompt_audit_signature"] for method in methods
            }
            truncation_counts = {
                matrix[(model, method, task)]["truncated_examples"] for method in methods
            }
            if len(signatures) != 1 or len(truncation_counts) != 1:
                raise AggregateError(
                    f"LongBench-E tokenized inputs/truncation changed across {model}/{task}"
                )
    generation_stop_protocols = {}
    for model in expected_models:
        protocols = {
            json.dumps(
                matrix[(model, method, task)]["generation_stop_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for method in methods
            for task in tasks
        }
        if len(protocols) != 1:
            raise AggregateError(
                f"LongBench-E EOS/newline stop protocol changed across {model} shards"
            )
        generation_stop_protocols[model] = json.loads(next(iter(protocols)))
    for model in expected_models:
        if model not in expected_truncations:
            continue
        expected_model_truncations = expected_truncations[model]
        if set(expected_model_truncations) != set(tasks):
            raise AggregateError(f"Incomplete expected truncation ledger for {model}")
        for method in methods:
            for task in tasks:
                observed = matrix[(model, method, task)]["truncated_examples"]
                expected_count = int(expected_model_truncations[task])
                if observed != expected_count:
                    raise AggregateError(
                        f"LongBench-E truncation ledger changed for {model}/{method}/{task}: "
                        f"observed {observed}, expected {expected_count}"
                    )
    nested = {}
    for model in expected_models:
        nested[model] = {}
        for method in methods:
            task_rows = {task: matrix[(model, method, task)] for task in tasks}
            archives = {row["data_zip_sha256"] for row in task_rows.values()}
            if len(archives) != 1:
                raise AggregateError(
                    f"LongBench archive changed across {model}/{method} task shards"
                )
            macro = sum(row["score_unrounded"] for row in task_rows.values()) / len(task_rows)
            nested[model][method] = {
                "score": round(macro, 2),
                "score_unrounded": macro,
                "n": sum(row["n"] for row in task_rows.values()),
                "per_task": task_rows,
                "data_zip_sha256": next(iter(archives)),
                **_sum_long_compression_statistics(
                    list(task_rows.values()),
                    method=method,
                    label=f"LongBench-E/{model}/{method}",
                ),
                "compression_ledger": [
                    ledger for task in tasks for ledger in task_rows[task]["compression_ledger"]
                ],
                "released_data_complete": True,
            }
            expected_total = sum(task_counts.values())
            if nested[model][method]["n"] != expected_total:
                raise AggregateError(
                    f"LongBench-E total n is not {expected_total} for {model}/{method}"
                )
        _require_long_exact_payload_match(nested[model], label=f"LongBench-E/{model}")
    return {
        "benchmark": "LongBench-E",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "commit": code_commit,
        "models": nested,
        "official_primary_reporting": "per_task_three_length_bucket_scores",
        "reported_scalar": "derived_unweighted_macro_over_task_scores",
        "protocol_scope": LONGBENCH_PROTOCOL_SCOPE,
        "reference_model_wrapper_exact": False,
        "prompt_budgets": {
            key: (
                {task: int(budget) for task, budget in value.items()}
                if isinstance(value, Mapping)
                else int(value)
            )
            for key, value in prompt_budgets.items()
        },
        "truncation_policy": (
            "official middle truncation at an explicit model-specific raw-prompt "
            "budget; post-wrapper prompt plus task generation is capacity checked"
        ),
        "expected_truncations": {
            model: {task: int(count) for task, count in expected_truncations[model].items()}
            for model in expected_models
            if model in expected_truncations
        },
        "generation_stop_protocols": generation_stop_protocols,
        "dataset_provenance": dataset_provenance,
        "metric_replay": (
            "official task scorer rerun from stored prediction against the exact pinned "
            "data.zip row; stored scores are cross-checks only"
        ),
        "released_data_complete": True,
        "sources": sources,
    }


def verify_math500(
    root: Path,
    *,
    expected_models: Sequence[str] = LONG_MODELS,
    methods: Sequence[str] = LONG_METHODS,
    dataset_size: int = 500,
    shard_size: int = 50,
    sample_seed: int = 2027,
    dataset_revision: str = MATH500_REVISION,
    _dataset_path_override: Path | None = None,
    _dataset_sha256_override: str | None = None,
) -> dict:
    """Coverage-check and independently regrade MATH-500 generations.

    The formal path resolves the exact Arrow file through immutable paper-input
    provenance, checks every rendered query/gold against its pinned dataset
    row, and reruns Math-Verify 0.9.0 from ``generated_text``.  Stored
    ``correct`` fields and shard totals are cross-checks, never grading inputs.
    The private path/digest override is for reduced unit/custom fixtures only.
    """
    rows, code_commit = _load_formal_long_results(root, axis="MATH-500")
    pinned_rows, dataset_provenance = _load_pinned_math500_rows(
        rows,
        dataset_size=dataset_size,
        dataset_revision=dataset_revision,
        dataset_path_override=_dataset_path_override,
        dataset_sha256_override=_dataset_sha256_override,
    )
    math_verify_score = _build_pinned_math500_scorer()
    groups: dict[tuple[str, str], list[dict]] = {}
    sources = []
    for path, document, method, method_result in rows:
        model = document.get("model")
        value = method_result.get("math500")
        if model not in expected_models or method not in methods or not isinstance(value, dict):
            raise AggregateError(f"Malformed MATH-500 result in {path}")
        if value.get("complete_dataset_eval") is not False:
            raise AggregateError(f"MATH-500 shard has wrong scope label in {path}")
        n = _integer(value.get("n"), label=f"{path} MATH n", minimum=1)
        if n != shard_size or value.get("dataset_size") != dataset_size or value.get("k") != 1:
            raise AggregateError(f"Wrong MATH-500 denominator/sampling in {path}")
        if value.get("grader") != "math_verify" or value.get("grader_version") != "0.9.0":
            raise AggregateError(f"Wrong MATH-500 grader in {path}")
        grader_config = value.get("grader_config")
        if (
            not isinstance(grader_config, dict)
            or grader_config.get("dataset_revision") != dataset_revision
            or grader_config.get("prediction_extraction") != "full_generation:latex+expr"
            or grader_config.get("gold_extraction")
            != "latex(boxed_match_priority=0; inline_math_wrapped)"
            or grader_config.get("gold_serialization") != MATH500_GOLD_SERIALIZATION
            or grader_config.get("official_reference_gold_serialization_exact") is not False
            or grader_config.get("gold_serialization_adaptation_reason")
            != MATH500_GOLD_ADAPTATION_REASON
            or grader_config.get("protocol_scope") != MATH500_PROTOCOL_SCOPE
            or grader_config.get("strict") is not True
            or grader_config.get("parse")
            != {
                "fallback_mode": "first_match",
                "extraction_mode": "any_match",
                "parsing_timeout_seconds": 5,
                "raise_on_error": False,
            }
            or grader_config.get("verify")
            != {
                "float_rounding": 6,
                "numeric_precision": 15,
                "strict": True,
                "allow_set_relation_comp": False,
                "timeout_seconds": 5,
                "raise_on_error": False,
            }
        ):
            raise AggregateError(f"Wrong MATH-500 dataset revision in {path}")
        if value.get("protocol") != ("hf_math500_test_adapted_composite_four_shot"):
            raise AggregateError(f"Wrong MATH-500 generation/grading protocol in {path}")
        if (
            value.get("eviction_applicable") is not False
            or value.get("eviction_fired") is not None
            or value.get("example_cache_lifetime") != PER_EXAMPLE_CACHE_LIFETIME
            or value.get("cross_iteration_tensor_retention")
            != PER_EXAMPLE_CROSS_ITERATION_TENSOR_RETENTION
        ):
            raise AggregateError(
                f"MATH-500 non-eviction cache provenance/cache lifetime is malformed in {path}"
            )
        prompt_config = value.get("prompt_config")
        generation_config = value.get("generation_config")
        if (
            not isinstance(prompt_config, dict)
            or prompt_config.get("use_chat_template") is not False
            or prompt_config.get("system") is not None
            or prompt_config.get("query_protocol") != "Math-Verify math_500 task query"
            or prompt_config.get("query_template") != MATH500_QUERY_TEMPLATE
            or prompt_config.get("few_shot") != 4
            or prompt_config.get("few_shot_order_one_based") != MATH500_FEWSHOT_ORDER
            or prompt_config.get("few_shot_content_sha256") != MATH500_FEWSHOT_CONTENT_SHA256
            or prompt_config.get("few_shot_rendering") != MATH500_FEWSHOT_RENDERING
            or prompt_config.get("protocol_id") != MATH500_PROTOCOL_ID
            or prompt_config.get("protocol_origin") != "adapted_composite"
            or prompt_config.get("math_verify_task_source_commit")
            != MATH_VERIFY_TASK_SOURCE_COMMIT
            or prompt_config.get("lighteval_prompt_manager_reference_version")
            != MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
            or prompt_config.get("lighteval_prompt_manager_source_commit")
            != MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT
            or prompt_config.get("upstream_lighteval_builtin_task_exact") is not False
            or prompt_config.get("upstream_math_verify_custom_task_exact") is not False
            or prompt_config.get("upstream_prompt_generation_components_exact") is not True
            or prompt_config.get("upstream_component_pair_executable") is not False
            or prompt_config.get("lighteval_balanced_sampler_seed") != 0
            or not isinstance(generation_config, dict)
            or generation_config.get("max_new_tokens") != 1024
            or generation_config.get("reference_max_new_tokens") != 1024
            or generation_config.get("protocol")
            != (
                "adapted_composite_math_verify_ba3d3aa_lighteval_"
                "prompt_manager_7c1cd62_four_shot"
            )
            or generation_config.get("do_sample") is not False
            or generation_config.get("requested_do_sample") is not False
            or generation_config.get("temperature") is not None
            or generation_config.get("top_p") is not None
            or generation_config.get("stop_sequences") != list(MATH500_STOP_SEQUENCES)
            or generation_config.get("sample_seed_base") != sample_seed
            or generation_config.get("per_sample_seed_formula")
            != "base + global_question_index * 100003 + sample_index"
        ):
            raise AggregateError(f"Wrong MATH-500 prompt/generation contract in {path}")
        config = document.get("config")
        if (
            not isinstance(config, dict)
            or config.get("math_limit") != shard_size
            or config.get("max_new_math") != 1024
            or config.get("sample_seed") != sample_seed
            or config.get("math_grader") != "math_verify"
            or str(config.get("math_verify_version")) != "0.9.0"
            or config.get("math500_revision") != dataset_revision
        ):
            raise AggregateError(f"Wrong MATH-500 formal config in {path}")
        selection = value.get("selection")
        if not isinstance(selection, dict):
            raise AggregateError(f"Missing MATH-500 shard selection in {path}")
        start = _integer(selection.get("start"), label=f"{path} MATH start")
        stop = _integer(selection.get("stop_exclusive"), label=f"{path} MATH stop", minimum=1)
        if selection.get("limit") != shard_size or config.get("math_start") != start:
            raise AggregateError(f"MATH-500 selection/config mismatch in {path}")
        correct = _integer(value.get("correct"), label=f"{path} MATH correct")
        if correct > n:
            raise AggregateError(f"MATH-500 correct count exceeds n in {path}")
        pass_sum = _number(value.get("pass@1_sum"), label=f"{path} MATH pass@1 sum")
        if pass_sum != float(correct):
            raise AggregateError(f"MATH-500 correct/pass@1 numerator mismatch in {path}")
        displayed = _number(value.get("pass@1"), label=f"{path} MATH pass@1")
        if abs(displayed - round(correct / n, 4)) > 1e-9:
            raise AggregateError(f"MATH-500 shard score/count mismatch in {path}")
        questions = value.get("per_question")
        if not isinstance(questions, list) or len(questions) != n:
            raise AggregateError(f"MATH-500 regradable per-question payload is missing in {path}")
        record_correct = 0
        compression_records = []
        compression_ledger = []
        for local_index, question in enumerate(questions):
            global_index = start + local_index
            if not isinstance(question, dict) or question.get("index") != global_index:
                raise AggregateError(f"MATH-500 global question indexes are not exact in {path}")
            source_row = pinned_rows[global_index]
            source_problem = source_row.get("problem")
            source_gold = source_row.get("answer")
            if (
                not isinstance(source_problem, str)
                or not source_problem
                or not isinstance(source_gold, str)
                or not source_gold
            ):
                raise AggregateError(
                    f"Pinned MATH-500 row {global_index} has no problem/answer text"
                )
            rendered_query = question.get("question")
            main_query = MATH500_QUERY_TEMPLATE.format(problem=source_problem)
            prompt_suffix = "\n\n" + main_query
            if (
                not isinstance(rendered_query, str)
                or not rendered_query.endswith(prompt_suffix)
                or hashlib.sha256(
                    rendered_query[: -len(prompt_suffix)].encode("utf-8")
                ).hexdigest()
                != MATH500_FEWSHOT_CONTENT_SHA256
                or question.get("gold") != source_gold
            ):
                raise AggregateError(
                    f"MATH-500 question/gold differs from pinned dataset row in {path}"
                )
            input_tokens = _integer(
                question.get("input_tokens"),
                label=f"{path} MATH input tokens row {global_index}",
                minimum=1,
            )
            if question.get("k") != 1:
                raise AggregateError(f"MATH-500 per-question k changed in {path}")
            samples = question.get("samples")
            if (
                not isinstance(samples, list)
                or len(samples) != 1
                or not isinstance(samples[0], dict)
            ):
                raise AggregateError(f"MATH-500 generation audit record is missing in {path}")
            sample = samples[0]
            compression_records.append(sample)
            expected_seed = sample_seed + global_index * 100_003
            if sample.get("seed") != expected_seed:
                raise AggregateError(f"MATH-500 deterministic sample seed mismatch in {path}")
            if not isinstance(sample.get("generated_text"), str):
                raise AggregateError(f"MATH-500 full generation was not retained in {path}")
            raw_generated = sample.get("raw_generated_text")
            matched_stop = sample.get("matched_stop_sequence")
            if (
                not isinstance(raw_generated, str)
                or matched_stop is not None
                and matched_stop not in MATH500_STOP_SEQUENCES
            ):
                raise AggregateError(f"MATH-500 raw generation/stop audit is malformed in {path}")
            stop_matches = [
                (raw_generated.find(sequence), sequence)
                for sequence in MATH500_STOP_SEQUENCES
                if raw_generated.find(sequence) >= 0
            ]
            if stop_matches:
                stop_index, expected_stop = min(stop_matches, key=lambda item: item[0])
                expected_text = raw_generated[:stop_index]
            else:
                expected_stop = None
                expected_text = raw_generated
            if sample["generated_text"] != expected_text or matched_stop != expected_stop:
                raise AggregateError(f"MATH-500 stop-string processing mismatch in {path}")
            if "extracted_answer" in sample:
                raise AggregateError(f"MATH-500 ambiguous extracted_answer is forbidden in {path}")
            legacy_extracted = sample.get("legacy_regex_extracted_answer_non_grading")
            if "legacy_regex_extracted_answer_non_grading" not in sample or (
                legacy_extracted is not None and not isinstance(legacy_extracted, str)
            ):
                raise AggregateError(
                    f"MATH-500 non-grading legacy extraction is malformed in {path}"
                )
            if sample.get("grading_input") != "generated_text":
                raise AggregateError(f"MATH-500 grading input is not generated_text in {path}")
            generated_tokens = _integer(
                sample.get("generated_tokens"),
                label=f"{path} MATH generated token count row {global_index}",
                minimum=1,
            )
            if generated_tokens > 1024:
                raise AggregateError(f"MATH-500 generation exceeded 1,024 tokens in {path}")
            compression_ledger.append(
                _long_compression_ledger_row(
                    sample,
                    sample_id=f"math500:{global_index}",
                    input_tokens=input_tokens,
                    max_new_tokens=1024,
                    generated_tokens=generated_tokens,
                )
            )
            sample_correct = sample.get("correct")
            if not isinstance(sample_correct, bool):
                raise AggregateError(f"MATH-500 correctness is not boolean in {path}")
            replayed_correct = math_verify_score(sample["generated_text"], source_gold)
            if sample_correct is not replayed_correct:
                raise AggregateError(
                    f"MATH-500 stored correctness differs from Math-Verify replay in {path}"
                )
            expected_binary = int(replayed_correct)
            if question.get("pass@1") != float(expected_binary) or question.get(
                "observed_pass@1"
            ) != float(expected_binary):
                raise AggregateError(f"MATH-500 per-question score mismatch in {path}")
            record_correct += expected_binary
        if record_correct != correct:
            raise AggregateError(f"MATH-500 correct count does not match audit records in {path}")
        compression = _validated_long_compression_sufficient_statistics(
            value,
            compression_records,
            method=method,
            label=f"{path} MATH-500",
        )
        displayed_compression_ratio = _number(
            value.get("compression_ratio"), label=f"{path} MATH compression ratio"
        )
        if method == "FullKV":
            if (
                value.get("compression_ratio_aggregation") != "fullkv_no_cache"
                or displayed_compression_ratio != 1.0
            ):
                raise AggregateError(f"MATH-500 FullKV displayed accounting changed in {path}")
        else:
            fp16_bits = _integer(
                compression.get("compression_fp16_bits"),
                label=f"{path} MATH fp16 bits",
                minimum=1,
            )
            stored_bits = _integer(
                compression.get("compression_stored_bits"),
                label=f"{path} MATH stored bits",
                minimum=1,
            )
            if value.get(
                "compression_ratio_aggregation"
            ) != "exact_sum_fp16_bits_over_sum_stored_bits" or displayed_compression_ratio != round(
                fp16_bits / stored_bits, 3
            ):
                raise AggregateError(f"MATH-500 displayed exact accounting changed in {path}")
        groups.setdefault((str(model), method), []).append(
            {
                "start": start,
                "stop": stop,
                "n": n,
                "correct": correct,
                "eviction_applicable": False,
                "eviction_fired": None,
                **compression,
                "compression_ledger": compression_ledger,
                "source": str(path.resolve()),
            }
        )
        sources.append(_source(path))
    expected = {(model, method) for model in expected_models for method in methods}
    if set(groups) != expected:
        raise AggregateError(
            f"MATH-500 matrix incomplete: missing={sorted(expected - set(groups))}"
        )
    nested = {}
    for model, method in sorted(expected):
        label = f"MATH-500/{model}/{method}"
        ordered = _validate_coverage(
            groups[(model, method)], expected_size=dataset_size, label=label
        )
        correct = sum(row["correct"] for row in ordered)
        mean = correct / dataset_size
        if dataset_size > 1:
            variance = (correct * (1.0 - mean) ** 2 + (dataset_size - correct) * mean**2) / (
                dataset_size - 1
            )
            stderr = math.sqrt(variance / dataset_size)
        else:
            stderr = 0.0
        low = max(0.0, mean - 1.959963984540054 * stderr)
        high = min(1.0, mean + 1.959963984540054 * stderr)
        nested.setdefault(model, {})[method] = {
            "pass@1": round(mean, 8),
            "pass@1_sum": correct,
            "correct": correct,
            "pass@1_stderr": round(stderr, 8),
            "pass@1_ci95": [round(low, 8), round(high, 8)],
            "n": dataset_size,
            "coverage": [0, dataset_size],
            "audit_records_validated": dataset_size,
            "eviction_applicable": False,
            "eviction_fired": None,
            **_sum_long_compression_statistics(
                ordered,
                method=method,
                label=label,
            ),
            "compression_ledger": [
                ledger for row in ordered for ledger in row["compression_ledger"]
            ],
            "shards": [row["source"] for row in ordered],
        }
    for model in expected_models:
        _require_long_exact_payload_match(nested[model], label=f"MATH-500/{model}")
    return {
        "benchmark": "MATH-500",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "commit": code_commit,
        "grader": "Math-Verify 0.9.0",
        "protocol_scope": MATH500_PROTOCOL_SCOPE,
        "few_shot_protocol": {
            "protocol_id": MATH500_PROTOCOL_ID,
            "protocol_origin": "adapted_composite",
            "math_verify_task_source_commit": MATH_VERIFY_TASK_SOURCE_COMMIT,
            "lighteval_prompt_manager_reference_version": (
                MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
            ),
            "lighteval_prompt_manager_source_commit": (
                MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT
            ),
            "upstream_lighteval_builtin_task_exact": False,
            "upstream_math_verify_custom_task_exact": False,
            "upstream_prompt_generation_components_exact": True,
            "upstream_component_pair_executable": False,
            "balanced_sampler_seed": 0,
            "few_shot_count": 4,
            "few_shot_order_one_based": MATH500_FEWSHOT_ORDER,
            "few_shot_content_sha256": MATH500_FEWSHOT_CONTENT_SHA256,
            "few_shot_rendering": MATH500_FEWSHOT_RENDERING,
            "use_chat_template": False,
        },
        "official_reference_gold_serialization_exact": False,
        "gold_serialization": MATH500_GOLD_SERIALIZATION,
        "dataset_provenance": dataset_provenance,
        "metric_replay": (
            "Math-Verify 0.9.0 rerun from stored generated_text against the exact "
            "pinned dataset gold; stored correctness is a cross-check only"
        ),
        "grading_input": "generated_text",
        "aggregation": "sum_integer_correct_after_exact_interval_and_audit_validation",
        "models": nested,
        "official_complete": True,
        "sources": sources,
    }


def _speed_comparisons(protocol: dict, *, path: Path) -> dict[str, tuple[str, str]]:
    rules = protocol.get("comparison_rules")
    if not isinstance(rules, dict):
        raise AggregateError(f"Missing fair-speed comparison rules in {path}")
    if (
        rules.get("baseline_method") != "TurboQuant-paper"
        or rules.get("proposal_methods") != list(PROPOSAL_METHODS)
        or rules.get("method_set_exact") is not True
    ):
        raise AggregateError(f"Malformed fair-speed comparison taxonomy in {path}")
    if (
        rules.get("speedup_ratio") != "baseline_latency / candidate_latency"
        or rules.get("formal_inferential_unit") != "independently launched fresh process"
        or rules.get("technical_repeats_must_not_be_pooled_across_processes") is not True
        or rules.get("packed_native_hybrid_measured") is not False
        or rules.get("packed_native_hybrid_must_not_be_called_full_hayatequant") is not True
    ):
        raise AggregateError(f"Fair-speed paired inference contract is missing in {path}")
    return {
        "hq_vs_turboquant": ("TurboQuant-paper", "HQ"),
        "adaptive_hq_vs_turboquant": ("TurboQuant-paper", "AdaptiveHQ"),
    }


def _speed_recipe_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _expected_speed_labels(bits: int) -> tuple[str, str, str]:
    _ = bits
    return ("TurboQuant-paper", "HQ", "AdaptiveHQ")


def _expected_auto_block_size(bits: int, head_dim: int) -> int:
    for block_size in (8, 4, 2, 1):
        if head_dim % block_size == 0 and bits * block_size <= 9:
            return block_size
    raise AggregateError(
        f"No valid formal HayateQuant block size for bits={bits}, head_dim={head_dim}"
    )


def _expected_frontier_rotation(bits: int, block_size: int, head_dim: int) -> str:
    """Independent formal expectation for HayateQuant ``rotation_mode=auto``."""

    power_of_two = head_dim > 0 and head_dim & (head_dim - 1) == 0
    if block_size > 1:
        if bits <= 2:
            return "dense"
        return "hybrid" if power_of_two else "dense"
    if power_of_two:
        # Both formal models have d=128, below the declared default
        # hadamard_gemm_max_dim=4096.  Keep this explicit rather than deriving
        # an aggregate expectation from the implementation under test.
        if head_dim > 4096:
            return "hadamard"
        return "hadamard_gemm"
    return "dense"


def _expected_speed_resolved_config(*, codec: str, bits: int, head_dim: int) -> dict[str, Any]:
    """Build the exact per-role recipe admitted to the formal speed matrix."""
    method = {
        "turboquant": "TurboQuant-paper",
        "hq": "HQ",
        "adaptive_hq": "AdaptiveHQ",
    }.get(codec)
    if method is None:  # pragma: no cover - closed internal enum
        raise AssertionError(f"unknown speed codec role: {codec}")
    return _expected_resolved_codec_config(method, bits, head_dim)


def _validate_speed_recipe_contract(
    document: dict, *, bits: int, path: Path
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, dict[str, dict[str, int]]]]:
    """Validate shape, rate, recipe, quality/timing linkage, and E2E capacity."""

    protocol = document["protocol"]
    args = protocol["arguments"]
    model = str(document.get("model"))
    expected_evaluation_role = PAPER_MODEL_EVALUATION_ROLES.get(
        model, "custom_evaluation_role_unspecified"
    )
    if protocol.get("model_evaluation_role") != expected_evaluation_role:
        raise AggregateError(f"Formal speed model evaluation role changed for {model} in {path}")
    expected_args = {
        "mode": "both",
        "local_files_only": True,
        "device": "cuda:0",
        "dtype": "float16",
        "fused_backend": "triton",
        "batch_size": 1,
        "micro_seq_len": 4096,
        "decode_seq_len": 1,
        "quality_query_len": 8,
        "micro_warmup": 8,
        "micro_repeats": 40,
        "micro_inner_loops": 1,
        "bootstrap_resamples": 10_000,
        "e2e_prompt_len": 4096,
        "e2e_gen_len": 64,
        "e2e_warmup": 1,
        "e2e_repeats": 5,
        "attention_implementation": "sdpa",
        "sink_tokens": 0,
        "recent_tokens": 0,
    }
    drift = {
        key: (args.get(key), expected)
        for key, expected in expected_args.items()
        if args.get(key) != expected
    }
    if drift:
        raise AggregateError(f"Formal fair-speed protocol drift in {path}: {drift}")

    model_shape = protocol.get("model_kv_shape")
    if not isinstance(model_shape, dict):
        raise AggregateError(f"Missing config-derived KV shape in {path}")
    attention_heads = _integer(
        model_shape.get("num_attention_heads"),
        label=f"{path} attention heads",
        minimum=1,
    )
    kv_heads = _integer(
        model_shape.get("num_key_value_heads"),
        label=f"{path} KV heads",
        minimum=1,
    )
    head_dim = _integer(model_shape.get("head_dim"), label=f"{path} head dim", minimum=1)
    if (
        attention_heads % kv_heads
        or model_shape.get("gqa_group_size") != attention_heads // kv_heads
        or not isinstance(model_shape.get("kv_heads_derivation"), str)
        or not isinstance(model_shape.get("head_dim_derivation"), str)
    ):
        raise AggregateError(f"Malformed config-derived GQA shape in {path}")

    capacity = protocol.get("model_context_capacity")
    if (
        not isinstance(capacity, dict)
        or _integer(capacity.get("tokens"), label=f"{path} context capacity", minimum=1)
        < args["e2e_prompt_len"] + args["e2e_gen_len"]
        or not isinstance(capacity.get("derivation"), str)
    ):
        raise AggregateError(f"E2E request is not proven to fit model capacity in {path}")

    compile_audit = protocol.get("compile_runtime_audit")
    compile_mode = args["compile_mode"]
    if compile_audit != {
        "requested_mode": compile_mode,
        "compile_all_forbidden": True,
        "prefill_compile_forbidden": True,
        "decode_compiled_dispatch_created": compile_mode == "decode-only",
        "matches_requested_mode": True,
    }:
        raise AggregateError(f"Compile-mode runtime audit failed in {path}")

    labels = _expected_speed_labels(bits)
    taxonomy = protocol.get("method_taxonomy")
    micro = document.get("micro")
    e2e = document.get("e2e")
    if (
        not isinstance(taxonomy, dict)
        or set(taxonomy) != set(labels)
        or not isinstance(micro, dict)
        or not isinstance(e2e, dict)
    ):
        raise AggregateError(f"Fair-speed method taxonomy/result categories changed in {path}")
    tq, fixed, adaptive = labels
    if (
        taxonomy[tq].get("proposal_method") is not False
        or taxonomy[tq].get("proposal_variant") is not None
        or taxonomy[tq].get("fake_quant_implementation") != TURBOQUANT_FAKE_QUANT_IMPLEMENTATION
        or taxonomy[fixed].get("proposal_method") is not True
        or taxonomy[fixed].get("proposal_variant") != "fixed_global"
        or taxonomy[fixed].get("fake_quant_implementation") != HQ_FAKE_QUANT_IMPLEMENTATION
        or taxonomy[adaptive].get("proposal_method") is not True
        or taxonomy[adaptive].get("proposal_variant") != "causal_adaptive"
        or taxonomy[adaptive].get("fake_quant_implementation")
        != ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION
    ):
        raise AggregateError(f"Formal proposal taxonomy changed in {path}")

    method_configs = micro.get("method_configs")
    recipe_hashes = micro.get("recipe_sha256")
    if (
        not isinstance(method_configs, dict)
        or set(method_configs) != set(labels)
        or not isinstance(recipe_hashes, dict)
        or set(recipe_hashes) != set(labels)
        or e2e.get("method_configs") != method_configs
        or e2e.get("recipe_sha256") != recipe_hashes
    ):
        raise AggregateError(f"Micro/E2E codec recipe identity changed in {path}")
    if micro.get("public_payload_scope") != PUBLIC_LOGICAL_PAYLOAD_SCOPE:
        raise AggregateError(
            f"Fair-speed public payload is not explicitly scoped as unpacked int64 "
            f"logical storage in {path}"
        )
    payload_runtime_audit = micro.get("public_payload_runtime_audit")
    if not isinstance(payload_runtime_audit, dict) or set(payload_runtime_audit) != {
        "prefill",
        "decode",
    }:
        raise AggregateError(f"Fair-speed public-payload runtime audit is missing in {path}")
    if (
        micro.get("recipe_linkage")
        != {
            "payload_quality_and_all_micro_timings_use_same_codec_instances": True,
            "all_rows_fingerprinted_from_declared_and_role_resolved_codec_config": True,
        }
        or e2e.get("cache_factory_recipe_revalidated_on_every_timed_construction") is not True
    ):
        raise AggregateError(f"Quality/timing recipe linkage is missing in {path}")

    base_codec = {
        "key_axis": "token",
        "value_axis": "token",
        "group_size": -1,
        "symmetric": False,
        "skip_layers": [],
        "residual_window": 128,
    }
    hayate_defaults = {
        **base_codec,
        "name": "hayatequant",
        "max_codebook_bits": 9,
        "codebook_init": "auto",
        "seed": 0,
        "encoder": "exact",
        "hadamard_gemm_max_dim": 4096,
        "ls_scale": True,
        "outlier_bits": None,
        "key_n_outlier": None,
        "value_n_outlier": None,
        "key_outlier_bits": None,
        "value_outlier_bits": None,
        "avg_bits": float(bits),
        "code_bits_per_pair_block": None,
        "key_bit_fraction": None,
        "adaptive_bit_split": None,
        "adaptive_bit_margin": 4,
        "adaptive_stats_min_tokens": 16,
        "key_error_weight": 4.0,
        "value_error_weight": 1.0,
        "key_code_bits_per_block": None,
        "value_code_bits_per_block": None,
        "use_qjl": False,
    }
    expected_codec = {
        tq: {
            **base_codec,
            "name": "turboquant",
            "bits": bits,
            "dist": "beta",
            "n_outlier": 0,
            "outlier_bits": None,
            "use_qjl": False,
            "qjl_roles": "key",
            "rotation_mode": "dense",
            "seed": 0,
            "value_bits": bits,
        },
        fixed: {
            **hayate_defaults,
            "required_codebook_bundle": HQ_CODEBOOK_BUNDLE,
            "bits": bits,
            "dist": "gaussian",
            "value_bits": bits,
            "block_size": 2,
            "rotation_mode": "auto",
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_code_bits_per_block": 2 * bits + 1,
            "value_code_bits_per_block": 2 * bits - 1,
            "n_outlier": 0,
        },
        adaptive: {
            **hayate_defaults,
            "required_codebook_bundle": ADAPTIVE_HQ_CODEBOOK_BUNDLE,
            "bits": bits,
            "dist": "gaussian",
            "value_bits": bits,
            "block_size": 2,
            "rotation_mode": "auto",
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_bit_fraction": (2 * bits + 1) / (4 * bits),
            "adaptive_bit_split": "online_stats",
            "n_outlier": 0,
        },
    }
    expected_resolved = {
        tq: _expected_speed_resolved_config(codec="turboquant", bits=bits, head_dim=head_dim),
        fixed: _expected_speed_resolved_config(codec="hq", bits=bits, head_dim=head_dim),
        adaptive: _expected_speed_resolved_config(
            codec="adaptive_hq", bits=bits, head_dim=head_dim
        ),
    }
    for label in labels:
        record = method_configs[label]
        if not isinstance(record, dict) or any(
            record.get(key) != taxonomy[label].get(key)
            for key in (
                "semantic_role",
                "fake_quant_implementation",
                "proposal_method",
                "proposal_variant",
            )
        ):
            raise AggregateError(f"Method taxonomy/config mismatch for {label} in {path}")
        codec = record.get("codec")
        declared = codec.get("declared") if isinstance(codec, dict) else None
        effective = codec.get("effective") if isinstance(codec, dict) else None
        if not isinstance(declared, dict) or not isinstance(effective, dict):
            raise AggregateError(f"Missing codec config for {label} in {path}")
        if declared != expected_codec[label]:
            raise AggregateError(f"Wrong scientific codec recipe for {label} in {path}")
        if label in PROPOSAL_METHODS and effective != {
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
        }:
            raise AggregateError(f"Formal proposal scale modes changed in {path}")
        resolved = record.get("resolved_at_model_head_dim")
        if resolved != expected_resolved[label]:
            raise AggregateError(
                f"Role-resolved codec recipe changed for {label} in {path}: "
                f"{resolved!r} != {expected_resolved[label]!r}"
            )
        expected_hash = _speed_recipe_digest(record)
        if recipe_hashes.get(label) != expected_hash:
            raise AggregateError(f"Recipe fingerprint mismatch for {label} in {path}")

    regimes = micro.get("regimes")
    if not isinstance(regimes, dict) or set(regimes) != {"prefill", "decode"}:
        raise AggregateError(f"Fair-speed regimes changed in {path}")
    expected_input_elements = 2 * args["batch_size"] * kv_heads * args["micro_seq_len"] * head_dim
    public_payload_footprints: dict[str, dict[str, dict[str, int]]] = {}
    for regime, seq_len in (
        ("prefill", args["micro_seq_len"]),
        ("decode", args["decode_seq_len"]),
    ):
        row = regimes[regime]
        tensor = row.get("tensor") if isinstance(row, dict) else None
        if (
            not isinstance(tensor, dict)
            or tensor.get("shape") != [args["batch_size"], kv_heads, seq_len, head_dim]
            or tensor.get("dtype") != "float16"
            or tensor.get("generated_on_device") is not True
            or tensor.get("elements_per_key_value_pair")
            != 2 * args["batch_size"] * kv_heads * seq_len * head_dim
            or row.get("timed_recipe_sha256") != recipe_hashes
        ):
            raise AggregateError(f"Timed tensor/recipe shape changed in {path}/{regime}")
        audits = payload_runtime_audit[regime]
        if not isinstance(audits, dict) or set(audits) != set(labels):
            raise AggregateError(
                f"Public-payload representation audit coverage changed in {path}/{regime}"
            )
        regime_input_elements = 2 * args["batch_size"] * kv_heads * seq_len * head_dim
        regime_vectors_per_role = args["batch_size"] * kv_heads * seq_len
        expected_regime_logical_bits = (
            regime_input_elements * bits + 2 * regime_vectors_per_role * 16
        )
        public_payload_footprints[regime] = {}
        for label in labels:
            audit = audits[label]
            if (
                not isinstance(audit, dict)
                or audit.get("all_index_tensors_int64") is not True
                or audit.get("index_tensors_bit_packed") is not False
                or audit.get("all_scale_tensors_signed_fp16") is not True
                or audit.get("index_tensor_count") != 2
                or audit.get("scale_tensor_count") != 2
                or audit.get("scale_elements") != 2 * regime_vectors_per_role
                or audit.get("materialized_scale_tensor_bits") != 2 * regime_vectors_per_role * 16
                or audit.get("logical_total_stored_bits") != expected_regime_logical_bits
            ):
                raise AggregateError(
                    f"Public payload representation changed in {path}/{regime}/{label}"
                )
            index_elements = _integer(
                audit.get("index_elements"),
                label=f"{path}/{regime}/{label} materialized index count",
                minimum=1,
            )
            materialized_index_bits = _integer(
                audit.get("materialized_index_tensor_bits"),
                label=f"{path}/{regime}/{label} materialized index bits",
                minimum=1,
            )
            materialized_scale_bits = _integer(
                audit.get("materialized_scale_tensor_bits"),
                label=f"{path}/{regime}/{label} materialized scale bits",
                minimum=1,
            )
            materialized_payload_bytes = _integer(
                audit.get("materialized_payload_tensor_bytes"),
                label=f"{path}/{regime}/{label} materialized payload tensor bytes",
                minimum=1,
            )
            if (
                materialized_index_bits != index_elements * 64
                or materialized_index_bits <= regime_input_elements * bits
                or materialized_payload_bytes * 8
                != materialized_index_bits + materialized_scale_bits
                or audit.get("materialized_payload_tensor_scope") != PUBLIC_PAYLOAD_TENSOR_SCOPE
            ):
                raise AggregateError(
                    f"Public payload indices are not proven unpacked int64 tensors in "
                    f"{path}/{regime}/{label}"
                )
            public_payload_footprints[regime][label] = {
                "materialized_payload_tensor_bytes": materialized_payload_bytes,
                "logical_total_stored_bits": expected_regime_logical_bits,
            }

    quality = micro.get("quality")
    if not isinstance(quality, dict) or set(quality) != set(labels):
        raise AggregateError(f"Fair-speed quality rows changed in {path}")
    payload_bits = set()
    for label in labels:
        row = quality[label]
        stored = _integer(row.get("stored_bits"), label=f"{path}/{label} stored bits", minimum=1)
        if (
            row.get("recipe_sha256") != recipe_hashes[label]
            or row.get("input_kv_elements") != expected_input_elements
            or not math.isclose(
                _number(
                    row.get("stored_bits_per_kv_element"),
                    label=f"{path}/{label} stored rate",
                ),
                stored / expected_input_elements,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AggregateError(f"Quality/rate recipe mismatch for {label} in {path}")
        for metric in ("key_mse", "value_mse", "attention_output_relative_l2"):
            if _number(row.get(metric), label=f"{path}/{label}/{metric}") < 0.0:
                raise AggregateError(f"Negative quality error for {label} in {path}")
        payload_bits.add(stored)
    vectors_per_role = args["batch_size"] * kv_heads * args["micro_seq_len"]
    expected_payload_bits = expected_input_elements * bits + 2 * vectors_per_role * 16
    if payload_bits != {expected_payload_bits}:
        raise AggregateError(
            f"Fair-speed rows are not exactly equal-rate after fixed-width data-plane "
            f"accounting in {path}: "
            f"{sorted(payload_bits)} != {[expected_payload_bits]}"
        )
    rate_contract = micro.get("rate_contract")
    if (
        not isinstance(rate_contract, dict)
        or rate_contract.get("nominal_average_bits") != bits
        or rate_contract.get("equal_input_shape") is not True
        or rate_contract.get("equal_accounted_payload_bits") is not True
        or set(rate_contract.get("methods", {})) != set(labels)
    ):
        raise AggregateError(f"Equal-rate contract is missing in {path}")
    expected_allocations = {
        tq: "symmetric_scalar",
        fixed: "fixed_global",
        adaptive: "causal_adaptive",
    }
    for label in labels:
        rate = rate_contract["methods"][label]
        if (
            rate.get("declared_key_bits") != bits
            or rate.get("declared_value_bits") != bits
            or rate.get("nominal_average_code_bits_per_coordinate") != float(bits)
            or rate.get("allocation") != expected_allocations[label]
            or rate.get("accounted_payload_bits") != expected_payload_bits
            or rate.get("input_kv_elements") != expected_input_elements
            or not math.isclose(
                _number(
                    rate.get("accounted_payload_bits_per_kv_element"),
                    label=f"{path}/{label} accounted payload rate",
                ),
                expected_payload_bits / expected_input_elements,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AggregateError(f"Rate audit changed for {label} in {path}")

    expected_proposal_contract = {
        "formal_proposals": ["HQ", "AdaptiveHQ"],
        "same_total_pair_code_bit_budget": True,
        "rows": {
            "HQ": {
                "proposal_variant": "fixed_global",
                "pair_code_bits_per_block": 4 * bits,
                "exact_assignment": True,
                "selector": None,
            },
            "AdaptiveHQ": {
                "proposal_variant": "causal_adaptive",
                "pair_code_bits_per_block": 4 * bits,
                "exact_assignment": True,
                "selector": expected_resolved[adaptive]["adaptive_allocation"],
            },
        },
    }
    if micro.get("proposal_contract") != expected_proposal_contract:
        raise AggregateError(f"Formal proposal/carrier contract changed in {path}")

    parity_checks = micro.get("parity_checks")
    if not isinstance(parity_checks, dict) or set(parity_checks) != {
        "proposal_payload_vs_fake_quant_quality"
    }:
        raise AggregateError(f"Unexpected fair-speed parity contract in {path}")
    fake_quant = parity_checks.get("proposal_payload_vs_fake_quant_quality")
    if not isinstance(fake_quant, list) or {
        (row.get("method"), row.get("role")) for row in fake_quant if isinstance(row, dict)
    } != {
        (fixed, "key"),
        (fixed, "value"),
        (adaptive, "key"),
        (adaptive, "value"),
    }:
        raise AggregateError(f"Fake-quant quality parity coverage changed in {path}")
    expected_effective = {
        fixed: {
            "effective_path": "hayate_fused_roundtrip",
            "payload_materialized": False,
            "hayate_override_entered": True,
            "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
        },
        adaptive: {
            "effective_path": "inherited_encode_then_decode",
            "payload_materialized": True,
            "hayate_override_entered": False,
            "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
        },
    }
    if any(
        row.get("stored_bits_match") is not True
        or row.get("stored_bits_from_fake_quant") != expected_payload_bits
        or row.get("stored_bits_from_quality_payload") != expected_payload_bits
        or row.get("effective_implementation") != expected_effective.get(row.get("method"))
        for row in fake_quant
    ):
        raise AggregateError(
            f"Timed fake-quant path/rate differs from its declared implementation in {path}"
        )

    expected_regime_effective = {
        tq: {
            "effective_path": "inherited_encode_then_decode",
            "payload_materialized": True,
            "hayate_override_entered": False,
            "runtime_probe": "codec_has_no_hayate_rt_support_probe",
        },
        **expected_effective,
    }
    for regime_name, regime in micro["regimes"].items():
        research = regime.get("research_fake_quant") if isinstance(regime, dict) else None
        if (
            not isinstance(research, dict)
            or research.get("effective_implementations") != expected_regime_effective
        ):
            raise AggregateError(
                f"Effective fake-quant implementation changed in {path}/{regime_name}"
            )

    context = e2e.get("model_context_contract")
    required = args["e2e_prompt_len"] + args["e2e_gen_len"]
    if (
        context
        != {
            "prompt_tokens": args["e2e_prompt_len"],
            "decode_steps": args["e2e_gen_len"],
            "required_context_tokens": required,
            "model_capacity_tokens": capacity["tokens"],
            "capacity_derivation": capacity["derivation"],
            "fits": True,
        }
        or e2e.get("input", {}).get("shape") != [args["batch_size"], args["e2e_prompt_len"]]
        or e2e.get("input", {}).get("decode_steps") != args["e2e_gen_len"]
        or e2e.get("model_kv_shape_from_config") != model_shape
        or e2e.get("attention_implementation_requested") != "sdpa"
        or e2e.get("attention_implementation_effective") != "sdpa"
        or e2e.get("native_hybrid_is_full_hayatequant") is not False
        or e2e.get("timer_completion_barrier") != "end_event.synchronize before elapsed_time read"
        or e2e.get("peak_allocated_memory_scope") != SPEED_PEAK_ALLOCATED_SCOPE
    ):
        raise AggregateError(f"E2E shape/capacity/timer contract changed in {path}")
    scientific_recipe = {
        "recipe_sha256": {label: recipe_hashes[label] for label in labels},
        "method_configs": {label: method_configs[label] for label in labels},
    }
    return (
        tuple(recipe_hashes[label] for label in labels),
        scientific_recipe,
        public_payload_footprints,
    )


def _paired_ratios_from_raw(
    raw: Any,
    *,
    comparisons: Mapping[str, tuple[str, str]],
    value_key: str,
    expected_repeats: int,
    label: str,
) -> dict[str, dict]:
    """Validate counterbalancing and recompute repeatwise latency ratios."""

    if not isinstance(raw, list) or len(raw) != expected_repeats:
        raise AggregateError(
            f"{label} needs {expected_repeats} raw paired repeats, got "
            f"{len(raw) if isinstance(raw, list) else type(raw).__name__}"
        )
    methods: tuple[str, ...] | None = None
    parsed: list[dict[str, dict]] = []
    orders: list[list[str]] = []
    for repeat_index, repeat in enumerate(raw):
        if not isinstance(repeat, dict) or repeat.get("repeat") != repeat_index:
            raise AggregateError(f"{label} repeats are not contiguous and ordered")
        order = repeat.get("order")
        samples = repeat.get("samples")
        if (
            not isinstance(order, list)
            or not order
            or not all(isinstance(method, str) and method for method in order)
            or len(set(order)) != len(order)
            or not isinstance(samples, list)
            or len(samples) != len(order)
        ):
            raise AggregateError(
                f"Malformed paired order/samples in {label} repeat {repeat_index}"
            )
        current_methods = tuple(sorted(order))
        if methods is None:
            methods = current_methods
        elif current_methods != methods:
            raise AggregateError(f"Method set changed across paired repeats in {label}")
        by_method = {}
        for position, (ordered_method, sample) in enumerate(zip(order, samples)):
            if (
                not isinstance(sample, dict)
                or sample.get("method") != ordered_method
                or sample.get("order_position") != position
            ):
                raise AggregateError(
                    f"Sample/order mismatch in {label} repeat {repeat_index} position {position}"
                )
            value = _number(
                sample.get(value_key),
                label=f"{label} repeat {repeat_index} {ordered_method} {value_key}",
            )
            if value <= 0.0:
                raise AggregateError(f"Timing must be positive in {label}")
            by_method[ordered_method] = {"value": value, "position": position}
        parsed.append(by_method)
        orders.append(order)
    assert methods is not None
    method_set = set(methods)
    for name, pair in comparisons.items():
        if not set(pair).issubset(method_set):
            raise AggregateError(f"Comparison {name} methods are absent from {label}")

    # The harness emits cyclic Latin blocks. Every complete block must put
    # every method once in every position; a final incomplete block may differ
    # by at most one and may not repeat a method in the same position.
    n_methods = len(methods)
    for block_start in range(0, len(orders), n_methods):
        block = orders[block_start : block_start + n_methods]
        complete = len(block) == n_methods
        for position in range(n_methods):
            seen = [order[position] for order in block]
            if len(set(seen)) != len(seen) or (complete and set(seen) != method_set):
                raise AggregateError(
                    f"Orders are not counterbalanced in {label} block {block_start // n_methods}"
                )

    result = {}
    for name, (baseline, candidate) in comparisons.items():
        rows = []
        for repeat_index, by_method in enumerate(parsed):
            baseline_value = by_method[baseline]["value"]
            candidate_value = by_method[candidate]["value"]
            rows.append(
                {
                    "repeat": repeat_index,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "candidate_speedup_x": baseline_value / candidate_value,
                    "baseline_order_position": by_method[baseline]["position"],
                    "candidate_order_position": by_method[candidate]["position"],
                }
            )
        result[name] = {
            "baseline_method": baseline,
            "candidate_method": candidate,
            "n_paired_repeats": len(rows),
            "raw_paired_ratios": rows,
            "process_median_speedup_x": statistics.median(
                row["candidate_speedup_x"] for row in rows
            ),
        }
    return result


def _validate_recorded_speedups(
    report: Any,
    recomputed: Mapping[str, dict],
    *,
    value_key: str,
    value_unit: str,
    label: str,
) -> None:
    if (
        not isinstance(report, dict)
        or report.get("ratio_definition") != "baseline_latency / candidate_latency"
        or report.get("interpretation") != "greater_than_1_means_candidate_is_faster"
        or report.get("source_value_key") != value_key
        or report.get("source_value_unit") != value_unit
        or report.get("pairing_unit") != "same_counterbalanced_repeat_within_one_process"
        or report.get("inferential_unit") != "independently_launched_fresh_process"
    ):
        raise AggregateError(f"Malformed paired speedup contract in {label}")
    recorded = report.get("comparisons")
    if not isinstance(recorded, dict) or set(recorded) != set(recomputed):
        raise AggregateError(f"Paired comparison set changed in {label}")
    for name, expected in recomputed.items():
        actual = recorded[name]
        if (
            not isinstance(actual, dict)
            or actual.get("baseline_method") != expected["baseline_method"]
            or actual.get("candidate_method") != expected["candidate_method"]
            or actual.get("n_paired_repeats") != expected["n_paired_repeats"]
        ):
            raise AggregateError(f"Paired comparison metadata mismatch for {label}/{name}")
        rows = actual.get("raw_paired_ratios")
        if not isinstance(rows, list) or len(rows) != len(expected["raw_paired_ratios"]):
            raise AggregateError(f"Raw paired ratios missing for {label}/{name}")
        for row_index, (got, want) in enumerate(zip(rows, expected["raw_paired_ratios"])):
            if not isinstance(got, dict) or set(got) != set(want):
                raise AggregateError(f"Raw paired ratio schema mismatch for {label}/{name}")
            for key, expected_value in want.items():
                actual_value = got.get(key)
                if isinstance(expected_value, float):
                    if not isinstance(actual_value, (int, float)) or not math.isclose(
                        float(actual_value), expected_value, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        raise AggregateError(
                            f"Raw paired ratio mismatch for {label}/{name} row {row_index}"
                        )
                elif actual_value != expected_value:
                    raise AggregateError(
                        f"Raw paired ratio mismatch for {label}/{name} row {row_index}"
                    )
        summary = actual.get("within_process_summary")
        if (
            not isinstance(summary, dict)
            or summary.get("unit") != "candidate_speedup_x"
            or summary.get("n") != expected["n_paired_repeats"]
            or not math.isclose(
                _number(summary.get("median"), label=f"{label}/{name} median"),
                expected["process_median_speedup_x"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise AggregateError(f"Within-process paired summary mismatch for {label}/{name}")


def _extract_process_speedups(
    document: dict,
    *,
    comparisons: Mapping[str, tuple[str, str]],
    micro_repeats: int,
    e2e_repeats: int,
    path: Path,
) -> dict[str, dict[str, dict]]:
    """Recompute and validate every paired speedup reported by one process."""

    extracted: dict[str, dict[str, dict]] = {}
    micro = document.get("micro")
    regimes = micro.get("regimes") if isinstance(micro, dict) else None
    if not isinstance(regimes, dict) or set(regimes) != {"prefill", "decode"}:
        raise AggregateError(f"Malformed fair-speed micro regimes in {path}")
    for regime_name in ("prefill", "decode"):
        regime = regimes[regime_name]
        payload_api = regime.get("payload_api") if isinstance(regime, dict) else None
        research = regime.get("research_fake_quant") if isinstance(regime, dict) else None
        if not isinstance(payload_api, dict) or not isinstance(research, dict):
            raise AggregateError(f"Missing micro API categories in {path}/{regime_name}")
        measurements = {
            **{
                f"micro.{regime_name}.payload_api.{name}": value
                for name, value in payload_api.items()
            },
            f"micro.{regime_name}.research_fake_quant.fake_quant_roundtrip": research.get(
                "fake_quant_roundtrip"
            ),
        }
        expected_names = {
            metric
            for metric in SPEED_PAIRED_TIMING_METRICS
            if metric.startswith(f"micro.{regime_name}.")
        }
        if set(measurements) != expected_names:
            raise AggregateError(f"Micro timing operation set changed in {path}/{regime_name}")
        for metric, measurement in measurements.items():
            if not isinstance(measurement, dict):
                raise AggregateError(f"Missing timing measurement {metric} in {path}")
            if (
                measurement.get("timer") != "torch.cuda.Event"
                or measurement.get("timer_native_unit") != "milliseconds"
                or measurement.get("timer_completion_barrier")
                != "end_event.synchronize before elapsed_time read"
                or measurement.get("normalisation") != "event elapsed time / 1 inner loop(s)"
            ):
                raise AggregateError(f"Unsynchronised micro timer contract in {path}/{metric}")
            recomputed = _paired_ratios_from_raw(
                measurement.get("raw_paired_repeats"),
                comparisons=comparisons,
                value_key="elapsed_ms_per_call",
                expected_repeats=micro_repeats,
                label=f"{path}:{metric}",
            )
            _validate_recorded_speedups(
                measurement.get("paired_speedups"),
                recomputed,
                value_key="elapsed_ms_per_call",
                value_unit="ms_per_call",
                label=f"{path}:{metric}",
            )
            extracted[metric] = recomputed

    e2e = document.get("e2e")
    if not isinstance(e2e, dict):
        raise AggregateError(f"Missing fair-speed E2E result in {path}")
    raw_e2e = e2e.get("raw_paired_repeats")
    e2e_reports = e2e.get("paired_speedups")
    if not isinstance(e2e_reports, dict):
        raise AggregateError(f"Missing E2E paired speedups in {path}")
    for metric_name, value_key, value_unit in (
        ("ttft", "ttft_ms", "ms"),
        (
            "decode_steady_latency",
            "decode_steady_mean_ms_per_token",
            "ms_per_token",
        ),
    ):
        metric = f"e2e.{metric_name}"
        recomputed = _paired_ratios_from_raw(
            raw_e2e,
            comparisons=comparisons,
            value_key=value_key,
            expected_repeats=e2e_repeats,
            label=f"{path}:{metric}",
        )
        _validate_recorded_speedups(
            e2e_reports.get(metric_name),
            recomputed,
            value_key=value_key,
            value_unit=value_unit,
            label=f"{path}:{metric}",
        )
        extracted[metric] = recomputed
    return extracted


def _extract_process_peak_memory(
    document: dict,
    *,
    comparisons: Mapping[str, tuple[str, str]],
    e2e_repeats: int,
    path: Path,
) -> dict[str, dict]:
    """Recompute paired allocated-memory ratios and differences from E2E raw rows.

    This is allocator high-water memory for the dequantized research cache.  It
    is deliberately separate from both logical ``stored_bits`` and the exact
    tensor bytes of the public unpacked payload.
    """

    e2e = document.get("e2e")
    raw = e2e.get("raw_paired_repeats") if isinstance(e2e, dict) else None
    recomputed = _paired_ratios_from_raw(
        raw,
        comparisons=comparisons,
        value_key="peak_allocated_mib",
        expected_repeats=e2e_repeats,
        label=f"{path}:{SPEED_PRIMARY_MEMORY_METRIC}",
    )
    if not isinstance(raw, list):  # guarded by _paired_ratios_from_raw
        raise AssertionError("paired raw rows unexpectedly disappeared")
    by_repeat = {}
    for repeat in raw:
        samples = repeat["samples"]
        by_repeat[int(repeat["repeat"])] = {sample["method"]: sample for sample in samples}
        for sample in samples:
            start = _number(
                sample.get("starting_allocated_mib"),
                label=f"{path} synchronized starting allocated memory",
            )
            peak = _number(sample.get("peak_allocated_mib"), label=f"{path} peak allocated memory")
            incremental = _number(
                sample.get("incremental_peak_allocated_mib"),
                label=f"{path} incremental peak allocated memory",
            )
            if (
                start < 0.0
                or incremental < 0.0
                or peak <= 0.0
                or not math.isclose(peak - start, incremental, rel_tol=1e-9, abs_tol=1e-6)
            ):
                raise AggregateError(f"Inconsistent allocated-memory counters in {path}")

    result = {}
    for comparison, row in recomputed.items():
        baseline, candidate = comparisons[comparison]
        resources = []
        for repeat_index in range(e2e_repeats):
            baseline_value = _number(
                by_repeat[repeat_index][baseline]["peak_allocated_mib"],
                label=f"{path} baseline peak allocated memory",
            )
            candidate_value = _number(
                by_repeat[repeat_index][candidate]["peak_allocated_mib"],
                label=f"{path} candidate peak allocated memory",
            )
            resources.append(
                {
                    "repeat": repeat_index,
                    "baseline_peak_allocated_mib": baseline_value,
                    "candidate_peak_allocated_mib": candidate_value,
                    "candidate_memory_advantage_x": baseline_value / candidate_value,
                    "candidate_memory_reduction_mib": baseline_value - candidate_value,
                }
            )
        result[comparison] = {
            "baseline_method": row["baseline_method"],
            "candidate_method": row["candidate_method"],
            "n_paired_repeats": row["n_paired_repeats"],
            "raw_paired_memory": resources,
            "process_median_memory_advantage_x": statistics.median(
                item["candidate_memory_advantage_x"] for item in resources
            ),
            "process_median_memory_reduction_mib": statistics.median(
                item["candidate_memory_reduction_mib"] for item in resources
            ),
        }
    return result


def _process_level_ratio_summary(rows: Sequence[dict], *, label: str) -> dict:
    """Geometric-mean speedup and Student-t CI over independent processes."""

    if len(rows) < 2:
        raise AggregateError(f"{label} needs at least two independent processes for a CI")
    values = [
        _number(
            row.get("process_median_speedup_x"),
            label=f"{label} process median speedup",
        )
        for row in rows
    ]
    if any(value <= 0.0 for value in values):
        raise AggregateError(f"{label} speedup ratios must be positive")
    logs = [math.log(value) for value in values]
    log_mean = statistics.fmean(logs)
    log_sd = statistics.stdev(logs)
    log_se = log_sd / math.sqrt(len(logs))
    df = len(logs) - 1
    critical = _T975[min(df, 30)]
    assert critical is not None
    low, high = math.exp(log_mean - critical * log_se), math.exp(log_mean + critical * log_se)
    return {
        "unit": "candidate_speedup_x",
        "ratio_definition": "baseline_latency / candidate_latency",
        "n_independent_processes": len(rows),
        "process_median_paired_speedups": values,
        "geometric_mean_process_speedup_x": math.exp(log_mean),
        "median_process_speedup_x": statistics.median(values),
        "min_process_speedup_x": min(values),
        "max_process_speedup_x": max(values),
        "log_ratio_sample_sd": log_sd,
        "log_ratio_standard_error": log_se,
        "student_t_df": df,
        "student_t_critical_95": critical,
        "student_t_log_geomean_ci95": [low, high],
        "ci_assumption": (
            "two-sided Student-t interval for the mean log ratio across "
            "independently launched processes; technical repeats are paired "
            "within process and are not treated as independent"
        ),
        "critical_value_note": (
            "exact df quantile" if df <= 30 else "conservative df=30 quantile cap"
        ),
    }


def _process_level_memory_summary(rows: Sequence[dict], *, label: str) -> dict:
    """Paired peak-allocated ratio/difference intervals over fresh processes."""

    ratio_rows = [
        {"process_median_speedup_x": row["process_median_memory_advantage_x"]} for row in rows
    ]
    ratio = _process_level_ratio_summary(ratio_rows, label=f"{label}/ratio")
    ratio["unit"] = "candidate_memory_advantage_x"
    ratio["ratio_definition"] = "baseline_peak_allocated_mib / candidate_peak_allocated_mib"
    ratio["process_median_paired_memory_advantages"] = ratio.pop("process_median_paired_speedups")
    ratio["geometric_mean_process_memory_advantage_x"] = ratio.pop(
        "geometric_mean_process_speedup_x"
    )
    ratio["median_process_memory_advantage_x"] = ratio.pop("median_process_speedup_x")
    ratio["min_process_memory_advantage_x"] = ratio.pop("min_process_speedup_x")
    ratio["max_process_memory_advantage_x"] = ratio.pop("max_process_speedup_x")

    differences = [
        _number(
            row.get("process_median_memory_reduction_mib"),
            label=f"{label} process median memory reduction",
        )
        for row in rows
    ]
    if len(differences) < 2:
        raise AggregateError(f"{label} needs at least two processes for a memory CI")
    mean = statistics.fmean(differences)
    sample_sd = statistics.stdev(differences)
    standard_error = sample_sd / math.sqrt(len(differences))
    df = len(differences) - 1
    critical = _T975[min(df, 30)]
    assert critical is not None
    difference = {
        "unit": "MiB",
        "difference_definition": ("baseline_peak_allocated_mib - candidate_peak_allocated_mib"),
        "interpretation": "greater_than_0_means_candidate_used_less_peak_allocated_memory",
        "n_independent_processes": len(differences),
        "process_median_paired_memory_reductions_mib": differences,
        "mean_process_memory_reduction_mib": mean,
        "median_process_memory_reduction_mib": statistics.median(differences),
        "sample_sd_mib": sample_sd,
        "standard_error_mib": standard_error,
        "student_t_df": df,
        "student_t_critical_95": critical,
        "student_t_mean_difference_ci95_mib": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        "ci_assumption": (
            "two-sided Student-t interval for the mean paired process-level memory "
            "difference; technical repeats remain paired within process"
        ),
    }
    return {"ratio": ratio, "difference": difference}


def _student_t4_cdf(value: float) -> float:
    """Closed-form Student-t CDF for the formal five-process design (df=4)."""

    value = float(value)
    if value == 0.0:
        return 0.5
    if value < 0.0:
        return 1.0 - _student_t4_cdf(-value)
    x = 4.0 / (4.0 + value * value)
    root = math.sqrt(1.0 - x)
    regularized_beta = 1.0 - 1.5 * root + 0.5 * root**3
    return 1.0 - 0.5 * regularized_beta


def _student_t4_quantile(probability: float) -> float:
    if not 0.5 < probability < 1.0:
        raise AggregateError("Student-t4 upper quantile probability must be in (0.5, 1)")
    low, high = 0.0, 1.0
    while _student_t4_cdf(high) < probability:
        high *= 2.0
        if high > 1e8:  # pragma: no cover - protects malformed probabilities
            raise AggregateError("Student-t4 quantile search did not converge")
    for _ in range(100):
        mid = (low + high) / 2.0
        if _student_t4_cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _speed_endpoint_category(metric: str) -> str:
    if metric.startswith("micro.prefill.payload_api."):
        return "public_payload_prefill"
    if metric.startswith("micro.decode.payload_api."):
        return "public_payload_decode_step"
    if metric == "micro.prefill.research_fake_quant.fake_quant_roundtrip":
        return "research_fake_quant_prefill_roundtrip"
    if metric == "micro.decode.research_fake_quant.fake_quant_roundtrip":
        return "research_fake_quant_decode_step_roundtrip"
    if metric.startswith("e2e."):
        return "research_fake_quant_e2e"
    raise AggregateError(f"Unclassified formal speed endpoint {metric!r}")


def _build_speed_pareto_contract(
    nested: Mapping[str, Any],
    *,
    expected_models: Sequence[str],
    expected_rates: Sequence[int],
) -> dict[str, Any]:
    """Build the predeclared, simultaneous speed/measured-memory assessment.

    Per-endpoint tables retain their ordinary two-sided pointwise intervals.
    The *only* route to a complete-superiority statement is the one-sided
    Bonferroni family below, evaluated over every primary (compile=off)
    model/rate/timing and paired peak-allocated endpoint.
    """

    n_cells = len(expected_models) * len(expected_rates)
    n_timing = n_cells * len(SPEED_PAIRED_TIMING_METRICS) * len(PROPOSAL_METHODS)
    n_memory = n_cells * len(PROPOSAL_METHODS)
    family_size = n_timing + n_memory
    alpha = 0.05
    critical = _student_t4_quantile(1.0 - alpha / family_size)
    endpoints = []
    footprint_constraints = []
    logical_rate_constraints = []
    for model in expected_models:
        for rate_int in expected_rates:
            rate = str(rate_int)
            row = nested[model][rate][SPEED_PRIMARY_COMPILE_MODE]
            for proposal, comparison_name in (
                ("HQ", "hq_vs_turboquant"),
                ("AdaptiveHQ", "adaptive_hq_vs_turboquant"),
            ):
                for metric in SPEED_PAIRED_TIMING_METRICS:
                    comparison = row["paired_speedups"][metric][comparison_name]
                    values = comparison["process_median_paired_speedups"]
                    logs = [math.log(float(value)) for value in values]
                    mean_log = statistics.fmean(logs)
                    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
                    endpoints.append(
                        {
                            "proposal": proposal,
                            "comparison": comparison_name,
                            "model": model,
                            "rate": rate_int,
                            "category": _speed_endpoint_category(metric),
                            "metric": metric,
                            "unit": "candidate_speedup_x",
                            "advantage_definition": "baseline_latency / candidate_latency",
                            "geometric_mean_advantage_x": math.exp(mean_log),
                            "bonferroni_one_sided_lower95": math.exp(
                                mean_log - critical * standard_error
                            ),
                            "candidate_strictly_better": (
                                mean_log - critical * standard_error > 0.0
                            ),
                        }
                    )
                memory = row["paired_peak_allocated_memory"][comparison_name]["ratio"]
                values = memory["process_median_paired_memory_advantages"]
                logs = [math.log(float(value)) for value in values]
                mean_log = statistics.fmean(logs)
                standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
                endpoints.append(
                    {
                        "proposal": proposal,
                        "comparison": comparison_name,
                        "model": model,
                        "rate": rate_int,
                        "category": "research_fake_quant_e2e_allocator_peak",
                        "metric": SPEED_PRIMARY_MEMORY_METRIC,
                        "unit": "candidate_memory_advantage_x",
                        "advantage_definition": (
                            "baseline_peak_allocated_mib / candidate_peak_allocated_mib"
                        ),
                        "geometric_mean_advantage_x": math.exp(mean_log),
                        "bonferroni_one_sided_lower95": math.exp(
                            mean_log - critical * standard_error
                        ),
                        "candidate_strictly_better": (mean_log - critical * standard_error > 0.0),
                    }
                )
            names = _expected_speed_labels(rate_int)
            tq, fixed, adaptive = names
            for proposal in (fixed, adaptive):
                for regime in ("prefill", "decode"):
                    footprints = row["public_payload_footprints"][regime]
                    tq_physical = footprints[tq]["materialized_payload_tensor_bytes"]
                    proposal_physical = footprints[proposal]["materialized_payload_tensor_bytes"]
                    tq_logical = footprints[tq]["logical_total_stored_bits"]
                    proposal_logical = footprints[proposal]["logical_total_stored_bits"]
                    footprint_constraints.append(
                        {
                            "proposal": proposal,
                            "model": model,
                            "rate": rate_int,
                            "regime": regime,
                            "baseline_materialized_payload_tensor_bytes": tq_physical,
                            "candidate_materialized_payload_tensor_bytes": proposal_physical,
                            "candidate_not_larger": proposal_physical <= tq_physical,
                        }
                    )
                    logical_rate_constraints.append(
                        {
                            "proposal": proposal,
                            "model": model,
                            "rate": rate_int,
                            "regime": regime,
                            "baseline_logical_stored_bits": tq_logical,
                            "candidate_logical_stored_bits": proposal_logical,
                            "equal_rate": proposal_logical == tq_logical,
                        }
                    )
    if len(endpoints) != family_size:
        raise AggregateError("Internal complete-dominance endpoint coverage changed")
    all_endpoints_better = all(row["candidate_strictly_better"] for row in endpoints)
    all_footprints_not_larger = all(row["candidate_not_larger"] for row in footprint_constraints)
    all_logical_rates_equal = all(row["equal_rate"] for row in logical_rate_constraints)
    established = all_endpoints_better and all_footprints_not_larger and all_logical_rates_equal
    return {
        "scope": (
            "HQ and AdaptiveHQ versus TurboQuant Algorithm-1 reference; speed and "
            "the measured memory representations only, not downstream quality"
        ),
        "candidate_comparisons": {
            "HQ": "hq_vs_turboquant",
            "AdaptiveHQ": "adaptive_hq_vs_turboquant",
        },
        "primary_compile_mode": SPEED_PRIMARY_COMPILE_MODE,
        "predeclared_timing_metrics": list(SPEED_PAIRED_TIMING_METRICS),
        "predeclared_memory_metric": SPEED_PRIMARY_MEMORY_METRIC,
        "n_model_rate_cells": n_cells,
        "n_timing_endpoints": n_timing,
        "n_memory_endpoints": n_memory,
        "n_simultaneous_stochastic_endpoints": family_size,
        "simultaneous_inference": {
            "method": "one-sided Bonferroni Student-t intervals on process mean log ratios",
            "familywise_confidence": 0.95,
            "alpha": alpha,
            "per_endpoint_alpha": alpha / family_size,
            "student_t_df": 4,
            "student_t_critical": critical,
            "statistical_unit": "independently_launched_fresh_process",
        },
        "all_predeclared_axes_present": True,
        "stochastic_endpoints": endpoints,
        "public_unpacked_payload_tensor_constraints": footprint_constraints,
        "logical_stored_bits_equal_rate_constraints": logical_rate_constraints,
        "all_simultaneous_endpoint_lower_bounds_above_one": all_endpoints_better,
        "all_public_payload_tensor_footprints_not_larger": all_footprints_not_larger,
        "all_logical_stored_bits_equal": all_logical_rates_equal,
        "complete_strict_superiority_established": established,
        "claim": (
            "complete speed/measured-memory superiority established in the predeclared scope"
            if established
            else "complete speed/measured-memory superiority not established"
        ),
        "exclusions": (
            "public payload tensor bytes are unpacked logical-payload tensors; E2E peak "
            "allocated memory is for a dequantized research cache; neither is packed KV "
            "serving memory"
        ),
    }


def _summary_medians(value: Any, prefix: str = "") -> dict[str, dict]:
    found = {}
    if isinstance(value, dict):
        if {"unit", "n", "median"}.issubset(value):
            found[prefix] = {
                "unit": value["unit"],
                "median": _number(value["median"], label=f"speed summary {prefix}"),
            }
        else:
            for key, child in value.items():
                if key == "paired_speedups":
                    continue
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                found.update(_summary_medians(child, child_prefix))
    return found


def _speed_csv_identities(value: Any, *, label: str, path: Path) -> list[str]:
    if not isinstance(value, str):
        raise AggregateError(f"Missing {label} in formal speed preflight: {path}")
    identities = value.split(",")
    if (
        not identities
        or any(not item or re.fullmatch(r"[A-Za-z0-9._:-]+", item) is None for item in identities)
        or len(set(identities)) != len(identities)
    ):
        raise AggregateError(f"Malformed {label} in formal speed preflight: {path}")
    return identities


def _validate_speed_allocation_contract(
    *,
    preflight_record: Any,
    speed_provenance: Mapping[str, Any],
    path: Path,
) -> str:
    """Verify eight-GPU isolation and one-visible-GPU measurement provenance."""

    if not isinstance(preflight_record, Mapping):
        raise AggregateError(f"Missing formal speed preflight record in {path}")
    environment = preflight_record.get("environment")
    gpu = preflight_record.get("gpu")
    host = preflight_record.get("host")
    selected = speed_provenance.get("selected_environment")
    if not all(isinstance(value, Mapping) for value in (environment, gpu, host, selected)):
        raise AggregateError(f"Missing formal speed allocation provenance in {path}")
    allocation_devices = _speed_csv_identities(
        environment.get("PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES"),
        label="PAPER_SPEED_ALLOCATION_CUDA_VISIBLE_DEVICES",
        path=path,
    )
    job_gpus = _speed_csv_identities(
        environment.get("SLURM_JOB_GPUS"), label="SLURM_JOB_GPUS", path=path
    )
    measured_device = environment.get("PAPER_SPEED_MEASUREMENT_CUDA_VISIBLE_DEVICE")
    if (
        len(allocation_devices) != SPEED_ALLOCATION_CONTRACT["allocated_b200_gpus"]
        or len(job_gpus) != SPEED_ALLOCATION_CONTRACT["allocated_b200_gpus"]
        or measured_device != allocation_devices[0]
        or environment.get("CUDA_VISIBLE_DEVICES") != measured_device
        or selected.get("CUDA_VISIBLE_DEVICES") != measured_device
        or environment.get("SLURM_JOB_PARTITION") != SPEED_ALLOCATION_CONTRACT["partition"]
        or environment.get("SLURM_GPUS_ON_NODE") not in {"8", "gpu:nvidia_b200:8"}
        or environment.get("SLURM_CPUS_PER_TASK")
        != str(SPEED_ALLOCATION_CONTRACT["cpus_per_task"])
        or environment.get("SLURM_JOB_CPUS_PER_NODE") not in {"160", "160(x1)"}
        or environment.get("SLURM_MEM_PER_NODE")
        != str(SPEED_ALLOCATION_CONTRACT["memory_per_node_mib"])
    ):
        raise AggregateError(f"Formal speed allocation/isolation contract changed in {path}")
    devices = gpu.get("devices")
    if (
        not isinstance(devices, list)
        or len(devices) != SPEED_ALLOCATION_CONTRACT["measured_visible_gpus"]
        or not isinstance(devices[0], Mapping)
        or devices[0].get("logical_index") != 0
    ):
        raise AggregateError(f"Formal speed process did not expose exactly one GPU in {path}")
    hostname = speed_provenance.get("hostname")
    selected_node = selected.get("SLURMD_NODENAME")
    if (
        not isinstance(hostname, str)
        or not hostname
        or host.get("hostname") != hostname
        or selected_node != hostname
        or selected.get("SLURM_NODELIST") != hostname
        or environment.get("SLURM_JOB_NODELIST") != hostname
    ):
        raise AggregateError(
            f"Formal speed process/preflight Slurm node identity changed in {path}"
        )
    return hostname


def verify_speed(
    root: Path,
    *,
    expected_replicates: int = 5,
    expected_models: Sequence[str] = LONG_MODELS,
    expected_rates: Sequence[int] = (2, 3, 4),
    expected_compile_modes: Sequence[str] = ("off", "decode-only"),
) -> dict:
    if expected_replicates != 5:
        raise AggregateError(
            "Formal fair-speed verification is predeclared for exactly five "
            "independent process replicates"
        )
    root = root.expanduser().resolve()
    paths = sorted(root.rglob("rep*.json")) if root.is_dir() else []
    if not paths:
        raise AggregateError(f"No fair-speed replicate JSON files found below {root}")
    groups: dict[tuple[str, int, str], list[dict]] = {}
    sources = []
    commits = set()
    hardware_strata: set[tuple] = set()
    software_strata: set[tuple] = set()
    protocol_strata: set[tuple] = set()
    node_strata: set[str] = set()
    run_uuids: set[str] = set()
    slurm_job_ids: set[str] = set()
    model_preflight_identities: dict[str, set[tuple[str, str, str]]] = {}
    for path in paths:
        document = _load_object(path)
        schema = document.get("schema")
        if schema != {"name": "onecomp.hayatequant_fair_speed", "version": 4}:
            raise AggregateError(f"Wrong fair-speed schema in {path}")
        protocol = document.get("protocol")
        provenance = document.get("provenance")
        if not isinstance(protocol, dict) or not isinstance(provenance, dict):
            raise AggregateError(f"Missing fair-speed protocol/provenance in {path}")
        args = protocol.get("arguments")
        if not isinstance(args, dict) or args.get("mode") != "both":
            raise AggregateError(f"Formal fair-speed run must contain micro+e2e in {path}")
        model = Path(str(protocol.get("model"))).name
        bits = _integer(protocol.get("average_bits"), label=f"{path} average bits")
        compile_mode = args.get("compile_mode")
        if (
            model not in expected_models
            or bits not in expected_rates
            or compile_mode not in expected_compile_modes
            or compile_mode not in ("off", "decode-only")
        ):
            raise AggregateError(f"Unexpected fair-speed matrix point in {path}")
        if args.get("micro_repeats") != 40 or args.get("e2e_repeats") != 5:
            raise AggregateError(f"Wrong formal fair-speed repeat counts in {path}")
        if args.get("dtype") != "float16" or args.get("fused_backend") != "triton":
            raise AggregateError(f"Wrong formal fair-speed dtype/backend in {path}")
        if args.get("sink_tokens") != 0 or args.get("recent_tokens") != 0:
            raise AggregateError(f"Fair-speed formal row retained FP16 token windows: {path}")
        if "micro" not in document or "e2e" not in document:
            raise AggregateError(f"Incomplete fair-speed result in {path}")
        preflight_identity = _require_run_provenance(document, path=path, return_identity=True)
        if document.get("model") != model or Path(
            str(protocol.get("model"))
        ).expanduser().resolve() != Path(preflight_identity["model_path"]):
            raise AggregateError(f"Speed protocol/preflight model identity changed in {path}")
        commits.add(preflight_identity["commit"])
        model_preflight_identities.setdefault(model, set()).add(
            (
                preflight_identity["model_path"],
                preflight_identity["model_key"],
                preflight_identity["input_manifest_sha256"],
            )
        )
        (
            recipe_identity,
            scientific_recipe,
            public_payload_footprints,
        ) = _validate_speed_recipe_contract(document, bits=bits, path=path)
        comparisons = _speed_comparisons(protocol, path=path)
        git = provenance.get("git")
        if (
            not isinstance(git, dict)
            or git.get("dirty") is not False
            or git.get("commit") != preflight_identity["commit"]
        ):
            raise AggregateError(
                f"Fair-speed result was not produced from clean fixed source: {path}"
            )
        run_uuid = provenance.get("run_uuid")
        try:
            parsed_uuid = uuid.UUID(str(run_uuid))
        except (ValueError, AttributeError, TypeError) as exc:
            raise AggregateError(f"Malformed fair-speed process UUID in {path}") from exc
        if parsed_uuid.version != 4 or str(parsed_uuid) != run_uuid:
            raise AggregateError(f"Fair-speed process UUID is not canonical UUID4 in {path}")
        if run_uuid in run_uuids:
            raise AggregateError(f"Reused fair-speed process UUID {run_uuid} in {path}")
        run_uuids.add(run_uuid)
        pid = provenance.get("pid")
        started = provenance.get("started_at_utc")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
            raise AggregateError(f"Missing fair-speed process PID in {path}")
        try:
            parsed_started = datetime.fromisoformat(str(started))
        except ValueError as exc:
            raise AggregateError(f"Malformed fair-speed process start time in {path}") from exc
        if parsed_started.tzinfo is None:
            raise AggregateError(f"Fair-speed process start time is not timezone-aware in {path}")
        fresh = provenance.get("fresh_process_contract")
        if (
            not isinstance(fresh, dict)
            or fresh.get("required") is not True
            or fresh.get("multiple_average_rates_accepted_by_cli") is not False
            or fresh.get("compile_all_allowed") is not False
            or fresh.get("only_allowed_compile_mode") != "decode-only"
        ):
            raise AggregateError(f"Fresh-process benchmark contract is missing in {path}")
        selected_environment = provenance.get("selected_environment")
        slurm_job_id = (
            selected_environment.get("SLURM_JOB_ID")
            if isinstance(selected_environment, dict)
            else None
        )
        run_provenance = document.get("run_provenance")
        preflight_environment = (
            run_provenance[0].get("record", {}).get("environment", {})
            if isinstance(run_provenance, list) and len(run_provenance) == 1
            else {}
        )
        preflight_record = (
            run_provenance[0].get("record")
            if isinstance(run_provenance, list) and len(run_provenance) == 1
            else None
        )
        node_strata.add(
            _validate_speed_allocation_contract(
                preflight_record=preflight_record,
                speed_provenance=provenance,
                path=path,
            )
        )
        if (
            not isinstance(slurm_job_id, str)
            or not slurm_job_id
            or preflight_environment.get("SLURM_JOB_ID") != slurm_job_id
        ):
            raise AggregateError(f"Formal speed result has no matching Slurm job ID in {path}")
        if slurm_job_id in slurm_job_ids:
            raise AggregateError(
                f"Fair-speed inferential processes reused Slurm job {slurm_job_id}"
            )
        slurm_job_ids.add(slurm_job_id)
        cuda_device = provenance.get("cuda_device")
        packages = provenance.get("packages")
        smi_match = provenance.get("selected_gpu_nvidia_smi_match")
        if (
            not isinstance(cuda_device, dict)
            or not isinstance(cuda_device.get("name"), str)
            or not isinstance(cuda_device.get("compute_capability"), list)
            or len(cuda_device["compute_capability"]) != 2
            or not isinstance(cuda_device.get("total_memory_bytes"), int)
            or not isinstance(cuda_device.get("multi_processor_count"), int)
            or not isinstance(packages, dict)
            or set(packages) != set(_SPEED_REQUIRED_PACKAGES)
            or any(packages.get(name) is None for name in _SPEED_REQUIRED_PACKAGES)
            or not isinstance(provenance.get("python"), str)
            or not isinstance(provenance.get("platform"), str)
            or not isinstance(smi_match, list)
            or len(smi_match) != 1
            or not isinstance(smi_match[0], dict)
            or not smi_match[0].get("driver_version")
        ):
            raise AggregateError(f"Missing exact fair-speed GPU/software stratum in {path}")
        hardware = (
            cuda_device["name"],
            tuple(cuda_device["compute_capability"]),
            cuda_device["total_memory_bytes"],
            cuda_device["multi_processor_count"],
            smi_match[0]["driver_version"],
        )
        software = tuple(
            sorted(
                [(f"package:{key}", str(value)) for key, value in packages.items()]
                + [
                    ("python", provenance["python"]),
                    ("platform", provenance["platform"]),
                ]
            )
        )
        protocol_stratum = tuple(
            (key, json.dumps(args.get(key), sort_keys=True, separators=(",", ":")))
            for key in _SPEED_PROTOCOL_STRATUM_KEYS
        )
        effective_env = provenance.get("effective_onecomp_environment")
        expected_env = {
            "ONECOMP_COMPILE_CODEC": "1" if compile_mode == "decode-only" else "0",
            "ONECOMP_COMPILE_DECODE_ONLY": ("1" if compile_mode == "decode-only" else "0"),
            "ONECOMP_TRITON_CODEC": "1",
            "ONECOMP_TRITON_PREFILL": "0",
        }
        if effective_env != expected_env:
            raise AggregateError(f"Effective codec environment drifted in {path}")
        hardware_strata.add(hardware)
        software_strata.add(software)
        protocol_strata.add(protocol_stratum)
        seed = _integer(args.get("seed"), label=f"{path} seed")
        summaries = _summary_medians({"micro": document["micro"], "e2e": document["e2e"]})
        if not summaries:
            raise AggregateError(f"No timing summaries in {path}")
        paired_speedups = _extract_process_speedups(
            document,
            comparisons=comparisons,
            micro_repeats=int(args["micro_repeats"]),
            e2e_repeats=int(args["e2e_repeats"]),
            path=path,
        )
        paired_peak_memory = _extract_process_peak_memory(
            document,
            comparisons=comparisons,
            e2e_repeats=int(args["e2e_repeats"]),
            path=path,
        )
        groups.setdefault((model, bits, str(compile_mode)), []).append(
            {
                "seed": seed,
                "summaries": summaries,
                "paired_speedups": paired_speedups,
                "paired_peak_allocated_memory": paired_peak_memory,
                "public_payload_footprints": public_payload_footprints,
                "hardware": hardware,
                "software": software,
                "protocol_stratum": protocol_stratum,
                "run_uuid": run_uuid,
                "slurm_job_id": slurm_job_id,
                "recipe_identity": recipe_identity,
                "scientific_recipe": scientific_recipe,
                "preflight_identity": (
                    preflight_identity["model_path"],
                    preflight_identity["model_key"],
                    preflight_identity["input_manifest_sha256"],
                ),
                "source": str(path.resolve()),
            }
        )
        sources.append(_source(path))
    if len(commits) != 1:
        raise AggregateError(f"Fair-speed replicates used multiple commits: {sorted(commits)}")
    if len(hardware_strata) != 1:
        raise AggregateError(
            "Fair-speed formal matrix spans multiple GPU/driver strata; raw timings "
            f"must not be pooled: {sorted(hardware_strata)}"
        )
    if len(software_strata) != 1:
        raise AggregateError("Fair-speed formal matrix spans multiple software strata")
    if len(protocol_strata) != 1:
        raise AggregateError("Fair-speed formal matrix spans multiple timing protocol strata")
    if len(node_strata) != 1:
        raise AggregateError(
            f"Fair-speed formal matrix spans multiple physical nodes: {sorted(node_strata)}"
        )
    if any(len(identities) != 1 for identities in model_preflight_identities.values()):
        raise AggregateError("Fair-speed model/input preflight identity changed across replicates")
    expected = {
        (model, bits, compile_mode)
        for model in expected_models
        for bits in expected_rates
        for compile_mode in expected_compile_modes
    }
    if set(groups) != expected:
        raise AggregateError(
            f"Fair-speed matrix incomplete: missing={sorted(expected - set(groups))}"
        )
    nested = {}
    expected_seeds = {20260715 + index for index in range(expected_replicates)}
    for (model, bits, compile_mode), replicates in sorted(groups.items()):
        if (
            len(replicates) != expected_replicates
            or {row["seed"] for row in replicates} != expected_seeds
        ):
            raise AggregateError(
                f"Wrong fair-speed replicates for {(model, bits, compile_mode)}: "
                f"seeds={sorted(row['seed'] for row in replicates)}"
            )
        if len({row["hardware"] for row in replicates}) != 1:
            raise AggregateError(f"GPU stratum changed across {(model, bits, compile_mode)}")
        if len({row["software"] for row in replicates}) != 1:
            raise AggregateError(f"Software stratum changed across {(model, bits, compile_mode)}")
        if len({row["protocol_stratum"] for row in replicates}) != 1:
            raise AggregateError(
                f"Timing protocol stratum changed across {(model, bits, compile_mode)}"
            )
        if len({row["recipe_identity"] for row in replicates}) != 1:
            raise AggregateError(f"Codec recipe changed across {(model, bits, compile_mode)}")
        if len({_speed_recipe_digest(row["scientific_recipe"]) for row in replicates}) != 1:
            raise AggregateError(
                f"Role-resolved scientific recipe changed across " f"{(model, bits, compile_mode)}"
            )
        if (
            len({_speed_recipe_digest(row["public_payload_footprints"]) for row in replicates})
            != 1
        ):
            raise AggregateError(
                f"Public payload tensor/logical footprints changed across "
                f"{(model, bits, compile_mode)}"
            )
        if len({row["preflight_identity"] for row in replicates}) != 1:
            raise AggregateError(
                f"Model/input identity changed across {(model, bits, compile_mode)}"
            )
        path_sets = {tuple(sorted(row["summaries"])) for row in replicates}
        if len(path_sets) != 1:
            raise AggregateError(f"Fair-speed summary schema changed across replicates")
        metrics = {}
        for metric in next(iter(path_sets)):
            units = {row["summaries"][metric]["unit"] for row in replicates}
            if len(units) != 1:
                raise AggregateError(f"Fair-speed unit changed for {metric}")
            values = [row["summaries"][metric]["median"] for row in replicates]
            metrics[metric] = {
                "unit": next(iter(units)),
                "replicate_medians": values,
                "median_of_replicate_medians": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        paired_metric_sets = {tuple(sorted(row["paired_speedups"])) for row in replicates}
        if len(paired_metric_sets) != 1:
            raise AggregateError(
                f"Fair-speed paired metric schema changed across replicates for "
                f"{(model, bits, compile_mode)}"
            )
        paired_metrics = {}
        for metric in next(iter(paired_metric_sets)):
            comparison_sets = {tuple(sorted(row["paired_speedups"][metric])) for row in replicates}
            if len(comparison_sets) != 1:
                raise AggregateError(
                    f"Fair-speed comparison set changed for {metric} across replicates"
                )
            paired_metrics[metric] = {}
            for comparison in next(iter(comparison_sets)):
                process_records = []
                for replicate in sorted(replicates, key=lambda row: row["seed"]):
                    process = replicate["paired_speedups"][metric][comparison]
                    process_records.append(
                        {
                            "seed": replicate["seed"],
                            "source": replicate["source"],
                            "run_uuid": replicate["run_uuid"],
                            "slurm_job_id": replicate["slurm_job_id"],
                            "baseline_method": process["baseline_method"],
                            "candidate_method": process["candidate_method"],
                            "n_paired_repeats": process["n_paired_repeats"],
                            "raw_paired_ratios": process["raw_paired_ratios"],
                            "process_median_speedup_x": process["process_median_speedup_x"],
                        }
                    )
                if (
                    len({row["baseline_method"] for row in process_records}) != 1
                    or len({row["candidate_method"] for row in process_records}) != 1
                ):
                    raise AggregateError(
                        f"Fair-speed method labels changed for {metric}/{comparison}"
                    )
                summary = _process_level_ratio_summary(
                    process_records,
                    label=f"{model}/{bits}/{compile_mode}/{metric}/{comparison}",
                )
                summary.update(
                    {
                        "baseline_method": process_records[0]["baseline_method"],
                        "candidate_method": process_records[0]["candidate_method"],
                        "processes": process_records,
                    }
                )
                paired_metrics[metric][comparison] = summary
        memory_comparison_sets = {
            tuple(sorted(row["paired_peak_allocated_memory"])) for row in replicates
        }
        if len(memory_comparison_sets) != 1:
            raise AggregateError(
                f"Paired peak-memory comparison set changed across "
                f"{(model, bits, compile_mode)}"
            )
        paired_peak_memory = {}
        for comparison in next(iter(memory_comparison_sets)):
            process_records = []
            for replicate in sorted(replicates, key=lambda row: row["seed"]):
                process = replicate["paired_peak_allocated_memory"][comparison]
                process_records.append(
                    {
                        "seed": replicate["seed"],
                        "source": replicate["source"],
                        "run_uuid": replicate["run_uuid"],
                        "slurm_job_id": replicate["slurm_job_id"],
                        "baseline_method": process["baseline_method"],
                        "candidate_method": process["candidate_method"],
                        "n_paired_repeats": process["n_paired_repeats"],
                        "raw_paired_memory": process["raw_paired_memory"],
                        "process_median_memory_advantage_x": process[
                            "process_median_memory_advantage_x"
                        ],
                        "process_median_memory_reduction_mib": process[
                            "process_median_memory_reduction_mib"
                        ],
                    }
                )
            if (
                len({row["baseline_method"] for row in process_records}) != 1
                or len({row["candidate_method"] for row in process_records}) != 1
            ):
                raise AggregateError(f"Peak-memory method labels changed for {comparison}")
            summary = _process_level_memory_summary(
                process_records,
                label=f"{model}/{bits}/{compile_mode}/{SPEED_PRIMARY_MEMORY_METRIC}/{comparison}",
            )
            summary.update(
                {
                    "baseline_method": process_records[0]["baseline_method"],
                    "candidate_method": process_records[0]["candidate_method"],
                    "processes": process_records,
                }
            )
            paired_peak_memory[comparison] = summary
        nested.setdefault(model, {}).setdefault(str(bits), {})[compile_mode] = {
            "n_replicates": expected_replicates,
            "seeds": sorted(expected_seeds),
            "scientific_recipe": replicates[0]["scientific_recipe"],
            "metrics": metrics,
            "paired_speedups": paired_metrics,
            "paired_peak_allocated_memory": paired_peak_memory,
            "public_payload_footprints": replicates[0]["public_payload_footprints"],
            "sources": [row["source"] for row in replicates],
        }
    pareto_contract = _build_speed_pareto_contract(
        nested,
        expected_models=expected_models,
        expected_rates=expected_rates,
    )
    return {
        "benchmark": "paired HayateQuant fair speed",
        "model_evaluation_roles": {
            model: PAPER_MODEL_EVALUATION_ROLES.get(model, "custom_evaluation_role_unspecified")
            for model in expected_models
        },
        "commit": next(iter(commits)),
        "primary_compile_mode": SPEED_PRIMARY_COMPILE_MODE,
        "secondary_compile_modes": ["decode-only"],
        "primary_mode_predeclared_in_source": True,
        "hardware_stratum": {
            "gpu_name": next(iter(hardware_strata))[0],
            "compute_capability": list(next(iter(hardware_strata))[1]),
            "total_memory_bytes": next(iter(hardware_strata))[2],
            "multi_processor_count": next(iter(hardware_strata))[3],
            "driver_version": next(iter(hardware_strata))[4],
        },
        "software_stratum": dict(next(iter(software_strata))),
        "timing_protocol_stratum": {
            key: json.loads(value) for key, value in next(iter(protocol_strata))
        },
        "allocation_isolation_contract": {
            **SPEED_ALLOCATION_CONTRACT,
            "physical_node": next(iter(node_strata)),
            "allocation_cuda_visibility_recorded_before_process_narrowing": True,
        },
        "statistical_unit": "independently_launched_fresh_process",
        "independent_process_audit": {
            "n_unique_run_uuid4": len(run_uuids),
            "n_unique_slurm_job_ids": len(slurm_job_ids),
            "uuid_and_slurm_job_id_unique_for_every_source": (
                len(run_uuids) == len(sources) == len(slurm_job_ids)
            ),
        },
        "five_process_design_note": (
            "five independent processes are predeclared for a process-level "
            "Student-t interval; technical repeats remain paired within process "
            "and are never promoted to inferential replicates"
        ),
        "paired_ratio_estimator": (
            "within-process median of repeatwise baseline/candidate latency ratios, "
            "then geometric mean across independent processes"
        ),
        "confidence_interval": ("two-sided 95% Student-t interval on process-level log ratios"),
        "implementation_scope_contract": {
            "packed_native_hybrid_measured": False,
            "packed_native_hybrid_must_not_be_called_full_hayatequant": True,
            "packed_decode_claim_allowed": False,
            "measurement_scope": (
                "public unpacked logical-payload encode/decode plus research fake-quant "
                "and dequantized research-cache E2E"
            ),
            "methods": {
                "TurboQuant-paper": TURBOQUANT_FAKE_QUANT_IMPLEMENTATION,
                "HQ": HQ_FAKE_QUANT_IMPLEMENTATION,
                "AdaptiveHQ": ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION,
            },
        },
        "multiplicity_contract": {
            "interval_scope": "pointwise_nominal_95_percent",
            "multiple_comparison_adjustment": (
                "none for per-endpoint tables; one-sided Bonferroni over every "
                "predeclared primary endpoint for the complete-dominance assessment"
            ),
            "reporting_scope": (
                "per-endpoint ratios remain descriptive; a complete-superiority claim "
                "is admitted only by the simultaneous pareto contract"
            ),
            "directional_speed_claim_allowed": False,
            "blanket_or_global_speed_claim_allowed": (
                pareto_contract["complete_strict_superiority_established"]
            ),
        },
        "public_payload_scope": dict(PUBLIC_LOGICAL_PAYLOAD_SCOPE),
        "public_payload_tensor_scope": PUBLIC_PAYLOAD_TENSOR_SCOPE,
        "e2e_peak_allocated_memory_scope": dict(SPEED_PEAK_ALLOCATED_SCOPE),
        "pareto_contract": pareto_contract,
        "models": nested,
        "official_complete": True,
        "sources": sources,
    }


def _receipt_file(root: Path, relative: Any, *, label: str, receipt: Path) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AggregateError(f"{label} is not a repo-relative path in {receipt}")
    lexical = root / relative
    if lexical.is_symlink():
        raise AggregateError(f"{label} is symlinked in {receipt}: {lexical}")
    candidate = lexical.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AggregateError(f"{label} escapes RUN_ROOT in {receipt}") from exc
    if not candidate.is_file():
        raise AggregateError(f"{label} is not a retained regular file in {receipt}: {candidate}")
    return candidate


def _validate_formal_job_receipts(
    run_root: Path,
    *,
    source_paths: Sequence[Path],
    expected_commit: str,
) -> dict[str, Any]:
    """Require one successful worker/submission receipt for every source JSON.

    Failed retry attempts remain admissible evidence.  Every accepted
    submission must have exactly one terminal execution receipt: normally a
    worker receipt, or a failure-only scheduler receipt when ``sacct`` proves
    that the wrapper never started.  Scheduler receipts can never establish
    scientific success.
    """

    root = run_root.expanduser().resolve(strict=True)
    try:
        generic_receipt_validation = validate_successful_result_receipts(
            root,
            phase="formal",
            source_paths=source_paths,
            expected_commit=expected_commit,
        )
    except ReceiptValidationError as exc:
        raise AggregateError(f"Formal lifecycle receipts are not auditable: {exc}") from exc
    records_root = root / "job_records/formal"
    if not records_root.is_dir() or records_root.is_symlink():
        raise AggregateError(f"Formal repo-local job receipt tree is missing: {records_root}")
    worker_paths = sorted(records_root.glob("*/*/worker.json"))
    scheduler_paths = sorted(records_root.glob("*/*/scheduler.json"))
    submission_paths = sorted(records_root.glob("*/*/submission.json"))
    if not submission_paths:
        raise AggregateError("Formal job receipts contain no submission records")

    workers: list[tuple[Path, dict[str, Any]]] = []
    workers_by_attempt: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    schedulers_by_attempt: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    submissions: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    worker_status_counts: dict[str, int] = {}
    scheduler_status_counts: dict[str, int] = {}
    submission_status_counts: dict[str, int] = {}
    expected_results_worktree = str(root.parents[4].resolve(strict=True))
    identity_fields = (
        "schema_version",
        "phase",
        "job_key",
        "attempt_id",
        "source_commit",
        "run_id",
        "expected_result",
        "results_worktree",
        "scheduler_user",
        "scheduler_job_name",
        "submission_tag",
    )

    def expected_job_name(job_key: str) -> str:
        prefix = "formal-"
        if not job_key.startswith(prefix) or job_key == prefix:
            raise AggregateError(f"Formal job key is not phase-namespaced: {job_key!r}")
        return f"hq-f-{job_key[len(prefix):]}"

    def expected_tag(row: Mapping[str, Any]) -> str:
        fields = (
            "hayatequant-paper-submission-v1",
            row["source_commit"],
            row["run_id"],
            row["results_worktree"],
            row["phase"],
            row["job_key"],
            row["attempt_id"],
            row["expected_result"],
        )
        digest = hashlib.sha256(
            b"".join(str(value).encode("utf-8") + b"\0" for value in fields)
        ).hexdigest()
        return f"hayatequant-{digest}"

    def validate_identity(
        row: Mapping[str, Any], path: Path, *, expected_kind: str
    ) -> tuple[str, str]:
        relative = path.relative_to(records_root)
        if len(relative.parts) != 3 or relative.name != f"{expected_kind}.json":
            raise AggregateError(f"Malformed {expected_kind} receipt location: {path}")
        job_key, attempt_id = relative.parts[:2]
        expected_result = row.get("expected_result")
        if not isinstance(expected_result, str) or not expected_result.startswith(
            "results/formal/"
        ):
            raise AggregateError(f"Receipt expected result escapes results/formal: {path}")
        if (
            row.get("schema_version") != 2
            or row.get("phase") != "formal"
            or row.get("job_key") != job_key
            or row.get("attempt_id") != attempt_id
            or row.get("source_commit") != expected_commit
            or row.get("run_id") != root.name
            or row.get("results_worktree") != expected_results_worktree
            or row.get("scheduler_job_name") != expected_job_name(job_key)
            or row.get("submission_tag") != expected_tag(row)
            or not isinstance(row.get("scheduler_user"), str)
            or not row["scheduler_user"]
        ):
            raise AggregateError(f"Receipt immutable identity is malformed: {path}")
        return job_key, attempt_id

    def validate_evidence(
        evidence: Any, *, expected_path: Any, label: str, receipt: Path
    ) -> Path:
        if not isinstance(evidence, dict) or evidence.get("path") != expected_path:
            raise AggregateError(f"{label} evidence identity is malformed: {receipt}")
        retained = _receipt_file(root, expected_path, label=label, receipt=receipt)
        if (
            evidence.get("bytes") != retained.stat().st_size
            or evidence.get("sha256") != _sha256(retained)
        ):
            raise AggregateError(f"{label} evidence changed after receipt: {receipt}")
        return retained

    for path in submission_paths:
        if path.is_symlink():
            raise AggregateError(f"Symlinked submission receipt is forbidden: {path}")
        row = _load_object(path)
        job_key, attempt_id = validate_identity(row, path, expected_kind="submission")
        if (
            row.get("kind") != "hayatequant_paper_submission_receipt"
            or row.get("terminal") is not True
            or row.get("status")
            not in {"submitted", "submission_failed", "skipped_existing"}
        ):
            raise AggregateError(f"Malformed or nonterminal formal submission receipt: {path}")
        key = (job_key, attempt_id)
        if key in submissions:
            raise AggregateError(f"Duplicate formal submission receipt identity: {key}")
        submissions[key] = (path, row)
        submission_status_counts[row["status"]] = (
            submission_status_counts.get(row["status"], 0) + 1
        )
        submission_evidence = row.get("scheduler_submission_evidence")
        if not isinstance(submission_evidence, dict):
            raise AggregateError(f"Submission lacks hashed scheduler streams: {path}")
        sbatch_stdout = validate_evidence(
            submission_evidence.get("stdout"),
            expected_path=row.get("scheduler_stdout"),
            label="scheduler submission stdout",
            receipt=path,
        )
        validate_evidence(
            submission_evidence.get("stderr"),
            expected_path=row.get("scheduler_stderr"),
            label="scheduler submission stderr",
            receipt=path,
        )
        if row["status"] == "submitted":
            slurm_id = row.get("slurm_job_id")
            if not isinstance(slurm_id, str) or re.fullmatch(r"[0-9]+", slurm_id) is None:
                raise AggregateError(f"Submitted formal receipt lacks numeric Slurm ID: {path}")
            try:
                sbatch_text = sbatch_stdout.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise AggregateError(f"Cannot read retained sbatch stdout: {path}") from exc
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9._-]+)?\r?\n?", sbatch_text)
            if match is None or match.group(1) != slurm_id:
                raise AggregateError(f"Retained sbatch stdout disagrees with Slurm ID: {path}")

    for path in worker_paths:
        if path.is_symlink():
            raise AggregateError(f"Symlinked worker receipt is forbidden: {path}")
        row = _load_object(path)
        job_key, attempt_id = validate_identity(row, path, expected_kind="worker")
        if (
            row.get("kind") != "hayatequant_paper_worker_receipt"
            or row.get("terminal") is not True
            or row.get("status") not in {"succeeded", "failed"}
        ):
            raise AggregateError(f"Malformed or nonterminal formal worker receipt: {path}")
        workers.append((path, row))
        key = (job_key, attempt_id)
        if key in workers_by_attempt:
            raise AggregateError(f"Duplicate formal worker receipt identity: {key}")
        workers_by_attempt[key] = (path, row)
        worker_status_counts[row["status"]] = worker_status_counts.get(row["status"], 0) + 1
        _receipt_file(root, row.get("stdout_log"), label="worker stdout log", receipt=path)
        _receipt_file(root, row.get("stderr_log"), label="worker stderr log", receipt=path)
        submission_entry = submissions.get(key)
        if submission_entry is None:
            raise AggregateError(f"Worker has no matching submission receipt: {path}")
        submission_path, submission = submission_entry
        if submission.get("status") != "submitted":
            raise AggregateError(
                f"Worker is paired with a non-submitted attempt: {submission_path}"
            )
        if any(submission.get(field) != row.get(field) for field in identity_fields):
            raise AggregateError(
                f"Worker/submission immutable identities disagree: {path} / {submission_path}"
            )
        worker_slurm = row.get("slurm_job_id")
        submission_slurm = submission.get("slurm_job_id")
        if (
            not isinstance(worker_slurm, str)
            or not isinstance(submission_slurm, str)
            or submission_slurm.split(";", 1)[0] != worker_slurm
        ):
            raise AggregateError(
                f"Worker/submission Slurm identities disagree: {path} / {submission_path}"
            )
        if row["status"] == "failed":
            if not isinstance(row.get("exit_code"), int) or row["exit_code"] == 0:
                raise AggregateError(f"Failed worker has no nonzero exit code: {path}")
            continue
        _receipt_file(root, row.get("provenance"), label="worker provenance", receipt=path)
        if row.get("exit_code") != 0:
            raise AggregateError(f"Successful worker does not record exit code zero: {path}")
        result = row.get("result")
        if not isinstance(result, dict) or result.get("path") != row.get("expected_result"):
            raise AggregateError(f"Successful worker result identity is malformed: {path}")
        result_path = _receipt_file(
            root, result.get("path"), label="successful worker result", receipt=path
        )
        if (
            result.get("bytes") != result_path.stat().st_size
            or result.get("sha256") != _sha256(result_path)
            or result.get("json_object_validated") is not True
            or result.get("duplicate_keys_rejected") is not True
            or result.get("nonfinite_numbers_rejected") is not True
        ):
            raise AggregateError(f"Successful worker result changed after receipt: {path}")

    terminal_failure_states = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
    for path in scheduler_paths:
        if path.is_symlink():
            raise AggregateError(f"Symlinked scheduler receipt is forbidden: {path}")
        row = _load_object(path)
        job_key, attempt_id = validate_identity(row, path, expected_kind="scheduler")
        if (
            row.get("kind") != "hayatequant_paper_scheduler_receipt"
            or row.get("terminal") is not True
            or row.get("status") != "scheduler_terminal_failure"
        ):
            raise AggregateError(f"Malformed scheduler terminal-failure receipt: {path}")
        key = (job_key, attempt_id)
        if key in schedulers_by_attempt:
            raise AggregateError(f"Duplicate formal scheduler receipt identity: {key}")
        if key in workers_by_attempt:
            raise AggregateError(
                f"Worker and scheduler receipts both exist for formal attempt: {key}"
            )
        schedulers_by_attempt[key] = (path, row)
        scheduler_status_counts[row["status"]] = (
            scheduler_status_counts.get(row["status"], 0) + 1
        )
        submission_entry = submissions.get(key)
        if submission_entry is None:
            raise AggregateError(f"Scheduler receipt has no matching submission: {path}")
        submission_path, submission = submission_entry
        if submission.get("status") != "submitted" or any(
            submission.get(field) != row.get(field) for field in identity_fields
        ):
            raise AggregateError(
                f"Scheduler/submission immutable identities disagree: {path} / {submission_path}"
            )
        slurm = row.get("slurm")
        if (
            not isinstance(slurm, dict)
            or slurm.get("job_id_raw") != submission.get("slurm_job_id")
            or slurm.get("user") != row.get("scheduler_user")
            or slurm.get("job_name") != row.get("scheduler_job_name")
            or slurm.get("submission_tag") != row.get("submission_tag")
            or slurm.get("state") not in terminal_failure_states
            or slurm.get("state") == "COMPLETED"
            or not isinstance(slurm.get("exit_code"), str)
        ):
            raise AggregateError(f"Scheduler receipt is not an exact Slurm failure: {path}")
        query_evidence = row.get("scheduler_query_evidence")
        if not isinstance(query_evidence, dict):
            raise AggregateError(f"Scheduler receipt lacks raw sacct evidence: {path}")
        sacct_stdout = validate_evidence(
            query_evidence.get("stdout"),
            expected_path=(query_evidence.get("stdout") or {}).get("path"),
            label="raw sacct stdout",
            receipt=path,
        )
        validate_evidence(
            query_evidence.get("stderr"),
            expected_path=(query_evidence.get("stderr") or {}).get("path"),
            label="raw sacct stderr",
            receipt=path,
        )
        try:
            sacct_lines = [
                line for line in sacct_stdout.read_text(encoding="utf-8").splitlines() if line
            ]
        except (OSError, UnicodeError) as exc:
            raise AggregateError(f"Cannot read raw sacct stdout: {path}") from exc
        if len(sacct_lines) != 1:
            raise AggregateError(f"Raw sacct evidence is not one exact allocation row: {path}")
        sacct_fields = sacct_lines[0].split("|")
        if sacct_fields and sacct_fields[-1] == "":
            sacct_fields.pop()
        expected_sacct = [
            slurm.get("job_id_raw"),
            slurm.get("user"),
            slurm.get("job_name"),
            slurm.get("state_raw"),
            slurm.get("exit_code"),
            slurm.get("submit_time"),
            slurm.get("start_time"),
            slurm.get("end_time"),
            slurm.get("submission_tag"),
        ]
        if sacct_fields != expected_sacct:
            raise AggregateError(f"Structured scheduler receipt disagrees with raw sacct row: {path}")

    missing_terminal_execution = [
        (path, row)
        for key, (path, row) in submissions.items()
        if row["status"] == "submitted"
        and key not in workers_by_attempt
        and key not in schedulers_by_attempt
    ]
    if missing_terminal_execution:
        paths = ", ".join(str(path) for path, _row in missing_terminal_execution[:5])
        raise AggregateError(
            "Submitted formal attempts have no terminal worker or scheduler receipt: " + paths
        )
    unexpected_execution = [
        path
        for key, (path, row) in submissions.items()
        if row["status"] != "submitted"
        and (key in workers_by_attempt or key in schedulers_by_attempt)
    ]
    if unexpected_execution:
        raise AggregateError(
            "Non-accepted formal attempts have execution receipts: "
            + ", ".join(str(path) for path in unexpected_execution[:5])
        )

    source_relatives: list[str] = []
    for source_path in source_paths:
        source = source_path.expanduser().resolve(strict=True)
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise AggregateError(f"Formal source is not repo-local below RUN_ROOT: {source}") from exc
        if not relative.startswith("results/formal/"):
            raise AggregateError(f"Formal source is outside results/formal: {source}")
        source_relatives.append(relative)
        matches = [
            (path, row)
            for path, row in workers
            if row.get("status") == "succeeded" and row.get("expected_result") == relative
        ]
        if len(matches) != 1:
            raise AggregateError(
                f"Formal source requires exactly one successful worker receipt, got "
                f"{len(matches)}: {source}"
            )
    return {
        "schema_version": 1,
        "scope": (
            "repo-local submission plus exactly one worker or failure-only scheduler "
            "receipt for every accepted attempt"
        ),
        "source_results_validated": len(source_relatives),
        "worker_receipts": len(worker_paths),
        "scheduler_receipts": len(scheduler_paths),
        "submission_receipts": len(submission_paths),
        "worker_status_counts": worker_status_counts,
        "scheduler_status_counts": scheduler_status_counts,
        "submission_status_counts": submission_status_counts,
        "generic_receipt_validation": generic_receipt_validation,
    }


def _cross_axis_provenance(
    document: Mapping[str, Any],
    *,
    axes: Sequence[str],
    run_root: Path | None = None,
) -> dict:
    """Validate one immutable input/model/dataset identity across result axes."""
    input_manifests: set[tuple[str, str]] = set()
    model_identities: dict[str, set[str]] = {}
    model_records: dict[str, dict] = {}
    dataset_records: dict[str, tuple[int, str]] = {}
    axis_source_counts: dict[str, int] = {}
    validated_sources: set[str] = set()
    source_runtimes: dict[str, tuple[str, dict]] = {}
    quality_runtime_fingerprints = set()
    quality_runtime_records = {}
    smoke_approval_fingerprints = set()
    smoke_approval_records = {}
    for axis in axes:
        value = document.get(axis)
        if not isinstance(value, dict):
            raise AggregateError(f"Formal axis {axis} is not an aggregate object")
        sources = value.get("sources")
        if not isinstance(sources, list) or not sources:
            raise AggregateError(f"Formal axis {axis} has no immutable source ledger")
        axis_paths = set()
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise AggregateError(f"Formal axis {axis} source {index} is malformed")
            source_path_raw = source.get("path")
            if not isinstance(source_path_raw, str) or not Path(source_path_raw).is_absolute():
                raise AggregateError(f"Formal axis {axis} source path is not absolute")
            source_path = Path(source_path_raw).expanduser().resolve()
            if not source_path.is_file():
                raise AggregateError(f"Formal axis {axis} source disappeared: {source_path}")
            if source.get("bytes") != source_path.stat().st_size or source.get(
                "sha256"
            ) != _sha256(source_path):
                raise AggregateError(f"Formal axis {axis} source content changed: {source_path}")
            axis_paths.add(str(source_path))
            if str(source_path) in validated_sources:
                fingerprint, runtime = source_runtimes[str(source_path)]
                if axis in QUALITY_AXIS_KEYS:
                    quality_runtime_fingerprints.add(fingerprint)
                    quality_runtime_records[fingerprint] = runtime
                continue
            validated_sources.add(str(source_path))
            result = _load_object(source_path)
            identity = _require_run_provenance(result, path=source_path, return_identity=True)
            source_runtimes[str(source_path)] = (
                identity["quality_runtime_fingerprint"],
                identity["quality_runtime"],
            )
            smoke_approval = identity["formal_smoke_approval"]
            smoke_approval_fingerprint = hashlib.sha256(
                json.dumps(
                    smoke_approval,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            smoke_approval_fingerprints.add(smoke_approval_fingerprint)
            smoke_approval_records[smoke_approval_fingerprint] = smoke_approval
            if axis in QUALITY_AXIS_KEYS:
                quality_runtime_fingerprints.add(identity["quality_runtime_fingerprint"])
                quality_runtime_records[identity["quality_runtime_fingerprint"]] = identity[
                    "quality_runtime"
                ]
            input_manifests.add(
                (identity["input_manifest_path"], identity["input_manifest_sha256"])
            )
            label = str(result.get("model"))
            for envelope in result["run_provenance"]:
                record = envelope["record"]
                model = record["model"]
                canonical_model = {
                    key: model.get(key)
                    for key in (
                        "key",
                        "path",
                        "revision",
                        "snapshot_revisions",
                        "metadata_files",
                        "weights",
                        "weight_verification_mode",
                        "input_manifest",
                    )
                }
                model_digest = hashlib.sha256(
                    json.dumps(canonical_model, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                model_identities.setdefault(label, set()).add(model_digest)
                model_records.setdefault(
                    label,
                    {
                        "model_key": model.get("key"),
                        "model_path": model.get("path"),
                        "identity_sha256": model_digest,
                    },
                )
                for dataset in record["dataset_files"]:
                    dataset_path = str(Path(dataset["path"]).expanduser().resolve())
                    dataset_identity = (int(dataset["bytes"]), str(dataset["sha256"]).lower())
                    previous = dataset_records.setdefault(dataset_path, dataset_identity)
                    if previous != dataset_identity:
                        raise AggregateError(
                            "Formal axes declare conflicting provenance for dataset/input file "
                            f"{dataset_path}"
                        )
        axis_source_counts[axis] = len(axis_paths)
    if len(input_manifests) != 1:
        raise AggregateError(
            "Formal benchmark axes do not share one paper input manifest: "
            f"{sorted(input_manifests)}"
        )
    for label, digests in model_identities.items():
        if len(digests) != 1:
            raise AggregateError(f"Formal axes disagree on immutable model provenance for {label}")
    if any(axis in QUALITY_AXIS_KEYS for axis in axes) and len(quality_runtime_fingerprints) != 1:
        raise AggregateError(
            "Formal quality axes span multiple software/GPU/ONECOMP semantic strata: "
            f"{sorted(quality_runtime_fingerprints)}"
        )
    if len(smoke_approval_fingerprints) != 1:
        raise AggregateError(
            "Formal benchmark axes do not share one worker-validated smoke approval: "
            f"{sorted(smoke_approval_fingerprints)}"
        )
    input_manifest_path, input_manifest_sha256 = next(iter(input_manifests))
    input_path = Path(input_manifest_path).expanduser().resolve()
    if not input_path.is_file() or _sha256(input_path) != input_manifest_sha256:
        raise AggregateError(
            f"Cross-axis paper input manifest is missing or changed: {input_path}"
        )
    for dataset_path, (size, digest) in dataset_records.items():
        path = Path(dataset_path)
        if not path.is_file() or path.stat().st_size != size or _sha256(path).lower() != digest:
            raise AggregateError(
                f"Cross-axis dataset/input provenance no longer matches file: {path}"
            )
    result = {
        "input_manifest": {
            "path": str(input_path),
            "sha256": input_manifest_sha256,
        },
        "models": {key: model_records[key] for key in sorted(model_records)},
        "dataset_files": [
            {"path": path, "bytes": value[0], "sha256": value[1]}
            for path, value in sorted(dataset_records.items())
        ],
        "axis_source_counts": axis_source_counts,
        "validated_source_files": len(validated_sources),
        "formal_smoke_approval": smoke_approval_records[next(iter(smoke_approval_fingerprints))],
    }
    if quality_runtime_fingerprints:
        fingerprint = next(iter(quality_runtime_fingerprints))
        result["quality_runtime"] = {
            "fingerprint_sha256": fingerprint,
            "normalized": quality_runtime_records[fingerprint],
            "scope": "all quality axes; speed retains its separate timing-stratum gate",
            "excluded_nuisance_identity": ["GPU UUID", "host name", "Slurm job id"],
        }
    if run_root is not None:
        result["job_receipts"] = _validate_formal_job_receipts(
            run_root,
            source_paths=[Path(path) for path in sorted(validated_sources)],
            expected_commit=next(iter(document[axis]["commit"] for axis in axes)),
        )
    return result


def _strict_point_estimate_status(
    proposal_value: Any,
    baseline_value: Any,
    *,
    direction: str,
    matched: bool,
) -> tuple[str, bool, bool]:
    """Return ``(status, complete, win)`` for one fail-closed formal cell."""

    def finite(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    proposal_finite = finite(proposal_value)
    baseline_finite = finite(baseline_value)
    complete = proposal_finite and baseline_finite and matched
    if not proposal_finite or not baseline_finite:
        status = "missing_or_nonfinite"
    elif not matched:
        status = "comparison_mismatch"
    elif float(proposal_value) == float(baseline_value):
        status = "tie_nonwin"
    elif direction == "lower_is_better":
        status = "strict_win" if float(proposal_value) < float(baseline_value) else "strict_loss"
    elif direction == "higher_is_better":
        status = "strict_win" if float(proposal_value) > float(baseline_value) else "strict_loss"
    else:
        raise AggregateError(f"Unknown point-estimate direction {direction!r}")
    return status, complete, status == "strict_win"


def _point_estimate_pair_summary(cells: Sequence[Mapping[str, Any]], *, scope: str) -> dict:
    """Summarize one predeclared proposal/baseline universe without rounding."""

    denominator = len(cells)
    wins = sum(row.get("win") is True for row in cells)
    complete = sum(row.get("complete") is True for row in cells)
    fraction = wins / denominator if denominator else 0.0
    return {
        "scope": scope,
        "expected_cells": denominator,
        "complete_matched_cells": complete,
        "wins": wins,
        "nonwins": denominator - wins,
        "win_fraction_unrounded": fraction,
        "threshold": POINT_ESTIMATE_WIN_THRESHOLD,
        "passed": (
            denominator > 0
            and complete == denominator
            and fraction >= POINT_ESTIMATE_WIN_THRESHOLD
        ),
    }


def _point_estimate_win_gate(document: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the preregistered unrounded, strict, matched-cell win rule.

    The universe is constructed from constants in this module, never from rows
    that happen to be present in an input.  Consequently a deleted, non-finite,
    tied, or rate-mismatched cell remains in the denominator and is a non-win.
    BlockQuant's predeclared source operating point remains a descriptive
    quality comparison, but is excluded from this matched-rate denominator.
    Counting an explicitly unmatched payload as a dominance win would make the
    threshold scientifically incoherent. FibQuant and BlockQuant are
    quality-only baselines and are never speed denominators.
    """

    cells: list[dict[str, Any]] = []
    unmatched_descriptive_cells: list[dict[str, Any]] = []

    def finite(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    def nested(root: Any, *keys: str) -> Any:
        value = root
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    def exact_counter_match(proposal_row: Any, baseline_row: Any) -> bool:
        if not isinstance(proposal_row, Mapping) or not isinstance(baseline_row, Mapping):
            return False
        if "compression_ledger" in proposal_row or "compression_ledger" in baseline_row:
            proposal_match = proposal_row.get("matched_rate_vs_turboquant")
            baseline_match = baseline_row.get("matched_rate_vs_turboquant")
            baseline_reference = baseline_row.get("matched_rate_reference")
            return (
                isinstance(proposal_match, Mapping)
                and proposal_match.get("exact_integer_cross_products") is True
                and (
                    isinstance(baseline_match, Mapping)
                    and baseline_match.get("exact_integer_cross_products") is True
                    or isinstance(baseline_reference, Mapping)
                    and baseline_reference.get("method") == "TurboQuant-paper"
                )
            )
        return all(
            isinstance(proposal_row.get(key), int)
            and not isinstance(proposal_row.get(key), bool)
            and proposal_row.get(key) == baseline_row.get(key)
            for key in ("compression_fp16_bits", "compression_stored_bits")
        )

    def ppl_rate_match(proposal_row: Any, baseline_row: Any) -> bool:
        if not isinstance(proposal_row, Mapping) or not isinstance(baseline_row, Mapping):
            return False
        return all(
            isinstance(proposal_row.get(key), int)
            and not isinstance(proposal_row.get(key), bool)
            and proposal_row.get(key) == baseline_row.get(key)
            for key in (
                "stored_bits",
                "stored_coordinates",
                "ppl_compression_fp16_bits",
                "ppl_compression_stored_bits",
            )
        )

    def add_cell(
        *,
        axis: str,
        endpoint: str,
        proposal: str,
        baseline: str,
        direction: str,
        proposal_value: Any,
        baseline_value: Any,
        matched: bool,
        rate_match_required: bool = True,
        operating_point_matched: bool = True,
    ) -> None:
        proposal_finite = finite(proposal_value)
        baseline_finite = finite(baseline_value)
        status, complete, win = _strict_point_estimate_status(
            proposal_value,
            baseline_value,
            direction=direction,
            matched=matched,
        )
        if not operating_point_matched:
            unmatched_descriptive_cells.append(
                {
                    "axis": axis,
                    "endpoint": endpoint,
                    "proposal": proposal,
                    "baseline": baseline,
                    "direction": direction,
                    "proposal_value_unrounded": (
                        float(proposal_value) if proposal_finite else None
                    ),
                    "baseline_value_unrounded": (
                        float(baseline_value) if baseline_finite else None
                    ),
                    "sample_set_matched": bool(matched),
                    "matched_comparison": False,
                    "rate_match_required": rate_match_required,
                    "operating_point_matched": False,
                    "included_in_denominator": False,
                    "complete": complete,
                    "descriptive_better": win,
                    "status": f"unmatched_descriptive_{status}",
                }
            )
            return
        cells.append(
            {
                "axis": axis,
                "endpoint": endpoint,
                "proposal": proposal,
                "baseline": baseline,
                "direction": direction,
                "proposal_value_unrounded": (float(proposal_value) if proposal_finite else None),
                "baseline_value_unrounded": (float(baseline_value) if baseline_finite else None),
                "matched_comparison": bool(matched),
                "rate_match_required": rate_match_required,
                "operating_point_matched": operating_point_matched,
                "included_in_denominator": True,
                "complete": complete,
                "win": win,
                "status": status,
            }
        )

    # WikiText-2: exact NLL sums/token counts avoid rounded PPL displays.
    ppl = document.get("ppl")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model in PAPER_MODELS:
                rates = ("3",) if baseline == "BlockQuant-paper" else ("2", "3", "4")
                for rate in rates:
                    proposal_row = nested(ppl, "models", model, "rates", rate, proposal)
                    baseline_row = nested(ppl, "models", model, "rates", rate, baseline)
                    proposal_tokens = (
                        proposal_row.get("ppl_n_tokens")
                        if isinstance(proposal_row, Mapping)
                        else None
                    )
                    baseline_tokens = (
                        baseline_row.get("ppl_n_tokens")
                        if isinstance(baseline_row, Mapping)
                        else None
                    )
                    proposal_value = (
                        proposal_row.get("ppl_nll_sum") / proposal_tokens
                        if isinstance(proposal_tokens, int)
                        and proposal_tokens > 0
                        and finite(proposal_row.get("ppl_nll_sum"))
                        else None
                    )
                    baseline_value = (
                        baseline_row.get("ppl_nll_sum") / baseline_tokens
                        if isinstance(baseline_tokens, int)
                        and baseline_tokens > 0
                        and finite(baseline_row.get("ppl_nll_sum"))
                        else None
                    )
                    add_cell(
                        axis="ppl",
                        endpoint=f"{model}/rate={rate}",
                        proposal=proposal,
                        baseline=baseline,
                        direction="lower_is_better",
                        proposal_value=proposal_value,
                        baseline_value=baseline_value,
                        matched=(
                            proposal_tokens == baseline_tokens
                            and (
                                baseline == "BlockQuant-paper"
                                or ppl_rate_match(proposal_row, baseline_row)
                            )
                        ),
                        rate_match_required=baseline != "BlockQuant-paper",
                        operating_point_matched=baseline != "BlockQuant-paper",
                    )

    # Corrected zero-shot: every fixed task is one preregistered endpoint and
    # exact integer correct/n is used instead of rounded task accuracy.
    zero = document.get("zeroshot")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model in PAPER_MODELS:
                rates = ("3",) if baseline == "BlockQuant-paper" else ("2", "3", "4")
                for rate in rates:
                    for task in ZERO_TASKS:
                        proposal_row = nested(
                            zero, "models", model, "rates", rate, proposal, "tasks", task
                        )
                        baseline_row = nested(
                            zero, "models", model, "rates", rate, baseline, "tasks", task
                        )
                        primary = ZERO_TASKS[task]["primary"]
                        count_key = "correct_norm" if primary == "acc_norm" else "correct"
                        proposal_n = (
                            proposal_row.get("n") if isinstance(proposal_row, Mapping) else None
                        )
                        baseline_n = (
                            baseline_row.get("n") if isinstance(baseline_row, Mapping) else None
                        )
                        proposal_value = (
                            proposal_row.get(count_key) / proposal_n
                            if isinstance(proposal_n, int)
                            and proposal_n > 0
                            and isinstance(proposal_row.get(count_key), int)
                            else None
                        )
                        baseline_value = (
                            baseline_row.get(count_key) / baseline_n
                            if isinstance(baseline_n, int)
                            and baseline_n > 0
                            and isinstance(baseline_row.get(count_key), int)
                            else None
                        )
                        add_cell(
                            axis="zeroshot",
                            endpoint=f"{model}/rate={rate}/{task}/{primary}",
                            proposal=proposal,
                            baseline=baseline,
                            direction="higher_is_better",
                            proposal_value=proposal_value,
                            baseline_value=baseline_value,
                            matched=(
                                proposal_n == baseline_n
                                and (
                                    baseline == "BlockQuant-paper"
                                    or exact_counter_match(proposal_row, baseline_row)
                                )
                            ),
                            rate_match_required=baseline != "BlockQuant-paper",
                            operating_point_matched=baseline != "BlockQuant-paper",
                        )

    # The mechanism axis is intentionally not duplicated in a dominance
    # denominator: HQ/AdaptiveHQ are contextual rows and not members of its
    # preregistered single-lever scalar-vs-block contrast.

    ruler = document.get("ruler")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model in LONG_MODELS:
                for context in RULER_LENGTHS:
                    proposal_row = nested(ruler, "models", model, str(context), proposal)
                    baseline_row = nested(ruler, "models", model, str(context), baseline)
                    add_cell(
                        axis="ruler",
                        endpoint=f"{model}/context={context}",
                        proposal=proposal,
                        baseline=baseline,
                        direction="higher_is_better",
                        proposal_value=(
                            proposal_row.get("score_unrounded")
                            if isinstance(proposal_row, Mapping)
                            else None
                        ),
                        baseline_value=(
                            baseline_row.get("score_unrounded")
                            if isinstance(baseline_row, Mapping)
                            else None
                        ),
                        matched=(
                            baseline == "BlockQuant-paper"
                            or exact_counter_match(proposal_row, baseline_row)
                        ),
                        rate_match_required=baseline != "BlockQuant-paper",
                        operating_point_matched=baseline != "BlockQuant-paper",
                    )

    longbench = document.get("longbench")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model in LONG_MODELS:
                for task in LONGBENCH_COUNTS:
                    proposal_row = nested(longbench, "models", model, proposal, "per_task", task)
                    baseline_row = nested(longbench, "models", model, baseline, "per_task", task)
                    add_cell(
                        axis="longbench",
                        endpoint=f"{model}/{task}",
                        proposal=proposal,
                        baseline=baseline,
                        direction="higher_is_better",
                        proposal_value=(
                            proposal_row.get("score_unrounded")
                            if isinstance(proposal_row, Mapping)
                            else None
                        ),
                        baseline_value=(
                            baseline_row.get("score_unrounded")
                            if isinstance(baseline_row, Mapping)
                            else None
                        ),
                        matched=(
                            baseline == "BlockQuant-paper"
                            or exact_counter_match(proposal_row, baseline_row)
                        ),
                        rate_match_required=baseline != "BlockQuant-paper",
                        operating_point_matched=baseline != "BlockQuant-paper",
                    )

    math500 = document.get("math500")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model in LONG_MODELS:
                proposal_row = nested(math500, "models", model, proposal)
                baseline_row = nested(math500, "models", model, baseline)
                proposal_n = proposal_row.get("n") if isinstance(proposal_row, Mapping) else None
                baseline_n = baseline_row.get("n") if isinstance(baseline_row, Mapping) else None
                add_cell(
                    axis="math500",
                    endpoint=model,
                    proposal=proposal,
                    baseline=baseline,
                    direction="higher_is_better",
                    proposal_value=(
                        proposal_row.get("correct") / proposal_n
                        if isinstance(proposal_n, int)
                        and proposal_n > 0
                        and isinstance(proposal_row.get("correct"), int)
                        else None
                    ),
                    baseline_value=(
                        baseline_row.get("correct") / baseline_n
                        if isinstance(baseline_n, int)
                        and baseline_n > 0
                        and isinstance(baseline_row.get("correct"), int)
                        else None
                    ),
                    matched=(
                        proposal_n == baseline_n
                        and (
                            baseline == "BlockQuant-paper"
                            or exact_counter_match(proposal_row, baseline_row)
                        )
                    ),
                    rate_match_required=baseline != "BlockQuant-paper",
                    operating_point_matched=baseline != "BlockQuant-paper",
                )

    longcode = document.get("longcodeqa")
    for proposal in PROPOSAL_METHODS:
        for baseline in QUALITY_BASELINE_METHODS:
            for model, bracket_plans in LONGCODE_PLANS.items():
                for bracket in bracket_plans:
                    proposal_row = nested(
                        longcode,
                        "models",
                        model,
                        "methods",
                        proposal,
                        "brackets",
                        bracket,
                    )
                    baseline_row = nested(
                        longcode,
                        "models",
                        model,
                        "methods",
                        baseline,
                        "brackets",
                        bracket,
                    )
                    proposal_n = (
                        proposal_row.get("n") if isinstance(proposal_row, Mapping) else None
                    )
                    baseline_n = (
                        baseline_row.get("n") if isinstance(baseline_row, Mapping) else None
                    )
                    add_cell(
                        axis="longcodeqa",
                        endpoint=f"{model}/{bracket}",
                        proposal=proposal,
                        baseline=baseline,
                        direction="higher_is_better",
                        proposal_value=(
                            proposal_row.get("correct") / proposal_n
                            if isinstance(proposal_n, int)
                            and proposal_n > 0
                            and isinstance(proposal_row.get("correct"), int)
                            else None
                        ),
                        baseline_value=(
                            baseline_row.get("correct") / baseline_n
                            if isinstance(baseline_n, int)
                            and baseline_n > 0
                            and isinstance(baseline_row.get("correct"), int)
                            else None
                        ),
                        matched=(
                            proposal_n == baseline_n
                            and (
                                baseline == "BlockQuant-paper"
                                or exact_counter_match(proposal_row, baseline_row)
                            )
                        ),
                        rate_match_required=baseline != "BlockQuant-paper",
                        operating_point_matched=baseline != "BlockQuant-paper",
                    )

    # Speed admits only the TurboQuant baseline.  Every primary compile=off
    # timing endpoint is a cell; intervals remain secondary robustness output.
    speed = document.get("speed")
    comparisons = {
        "HQ": "hq_vs_turboquant",
        "AdaptiveHQ": "adaptive_hq_vs_turboquant",
    }
    for proposal, comparison_name in comparisons.items():
        for model in LONG_MODELS:
            for rate in (2, 3, 4):
                for metric in SPEED_PAIRED_TIMING_METRICS:
                    row = nested(
                        speed,
                        "models",
                        model,
                        str(rate),
                        SPEED_PRIMARY_COMPILE_MODE,
                        "paired_speedups",
                        metric,
                        comparison_name,
                    )
                    value = (
                        row.get("geometric_mean_process_speedup_x")
                        if isinstance(row, Mapping)
                        else None
                    )
                    add_cell(
                        axis="speed",
                        endpoint=f"{model}/rate={rate}/{metric}",
                        proposal=proposal,
                        baseline="TurboQuant-paper",
                        direction="higher_is_better",
                        proposal_value=value,
                        baseline_value=1.0,
                        matched=(
                            isinstance(row, Mapping)
                            and row.get("baseline_method") == "TurboQuant-paper"
                            and row.get("candidate_method") == proposal
                        ),
                    )

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    descriptive_summaries: dict[str, dict[str, Any]] = {}
    required_pairs = []
    for proposal in PROPOSAL_METHODS:
        summaries[proposal] = {}
        for baseline in ("TurboQuant-paper", "FibQuant-paper"):
            pair_cells = [
                row
                for row in cells
                if row["proposal"] == proposal
                and row["baseline"] == baseline
                and row["axis"] != "speed"
            ]
            summary = _point_estimate_pair_summary(pair_cells, scope="quality")
            summaries[proposal][baseline] = summary
            required_pairs.append(summary)
        descriptive_rows = [
            row
            for row in unmatched_descriptive_cells
            if row["proposal"] == proposal and row["baseline"] == "BlockQuant-paper"
        ]
        descriptive_count = len(descriptive_rows)
        descriptive_complete = sum(row.get("complete") is True for row in descriptive_rows)
        descriptive_better = sum(row.get("descriptive_better") is True for row in descriptive_rows)
        descriptive_summaries[proposal] = {
            "scope": "unmatched_source_operating_point_quality_descriptive",
            "expected_cells": descriptive_count,
            "complete_cells": descriptive_complete,
            "descriptively_better": descriptive_better,
            "not_better_or_incomplete": descriptive_count - descriptive_better,
            "descriptive_better_fraction_unrounded": (
                descriptive_better / descriptive_count if descriptive_count else 0.0
            ),
            "threshold": None,
            "passed": None,
            "included_in_point_estimate_gate": False,
        }
        speed_cells = [
            row for row in cells if row["proposal"] == proposal and row["axis"] == "speed"
        ]
        speed_summary = _point_estimate_pair_summary(speed_cells, scope="speed")
        summaries[proposal]["TurboQuant-paper-speed"] = speed_summary
        required_pairs.append(speed_summary)

    return {
        "schema_version": 1,
        "threshold": POINT_ESTIMATE_WIN_THRESHOLD,
        "threshold_comparison": "win_fraction_unrounded >= threshold",
        "cell_win_rule": "strict_unrounded_point_estimate_only",
        "nonwin_statuses": [
            "strict_loss",
            "tie_nonwin",
            "missing_or_nonfinite",
            "comparison_mismatch",
        ],
        "quality_axes": [
            "ppl",
            "zeroshot",
            "ruler",
            "longbench",
            "math500",
            "longcodeqa",
        ],
        "mechanism_axis": {
            "included_in_denominator": False,
            "reason_code": "proposal_rows_not_in_single_lever_mechanism_contrast",
        },
        "baseline_applicability": {
            "TurboQuant-paper": {
                "quality": True,
                "speed": True,
                "reason_code": "formal_quality_and_speed_baseline",
            },
            "FibQuant-paper": {
                "quality": True,
                "speed": False,
                "reason_code": "quality_only_no_packed_serving_kernel_speed_claim",
            },
            "BlockQuant-paper": {
                "quality": True,
                "speed": False,
                "included_in_denominator": False,
                "rate_match_required": False,
                "operating_point_matched": False,
                "paper_comparable": False,
                "paper_reproduction": False,
                "reason_code": "unmatched_source_operating_point_descriptive_only",
                "completion_specification": BLOCKQUANT_SELECTION_COMPLETION,
            },
        },
        "proposal_baseline_summaries": summaries,
        "unmatched_descriptive_comparisons": {
            "baseline": "BlockQuant-paper",
            "scope": "source_operating_point_quality_only",
            "included_in_point_estimate_gate": False,
            "reason_code": "payload_operating_point_not_matched_to_hq",
            "proposal_summaries": descriptive_summaries,
            "cells": unmatched_descriptive_cells,
        },
        "cells": cells,
        "passed": all(bool(summary["passed"]) for summary in required_pairs),
    }


def _finalize_formal_document(document: dict, *, run_root: Path | None = None) -> None:
    """Set commit/completeness only after all present axes share provenance."""
    present_axes = tuple(axis for axis in FORMAL_AXIS_KEYS if axis in document)
    if not present_axes:
        raise AggregateError("Aggregate contains no formal benchmark axis")
    axis_commits = {
        document[axis].get("commit")
        for axis in present_axes
        if isinstance(document[axis], dict) and isinstance(document[axis].get("commit"), str)
    }
    if len(axis_commits) != 1 or any(
        not isinstance(document[axis], dict) or not isinstance(document[axis].get("commit"), str)
        for axis in present_axes
    ):
        raise AggregateError(
            f"Formal benchmark axes do not share one fixed source commit: {sorted(axis_commits)}"
        )
    document["commit"] = next(iter(axis_commits))
    document["formal_provenance"] = _cross_axis_provenance(
        document, axes=present_axes, run_root=run_root
    )
    missing_axes = [axis for axis in FORMAL_AXIS_KEYS if axis not in document]
    incomplete_axes = []
    for axis in present_axes:
        if axis == "longbench":
            complete = document[axis].get("released_data_complete", False)
        else:
            complete = document[axis].get(
                "official_complete",
                document[axis].get("released_data_complete_feasible_plan", False),
            )
        if not complete:
            incomplete_axes.append(axis)
    document["formal_matrix_status"] = {
        "required_axes": list(FORMAL_AXIS_KEYS),
        "present_axes": list(present_axes),
        "missing_axes": missing_axes,
        "incomplete_axes": incomplete_axes,
        "one_cross_axis_provenance_identity": True,
    }
    point_gate = _point_estimate_win_gate(document)
    document["point_estimate_win_gate"] = point_gate
    coverage_complete = not missing_axes and not incomplete_axes
    document["formal_matrix_status"]["coverage_complete"] = coverage_complete
    document["formal_matrix_status"]["point_estimate_win_gate_passed"] = point_gate["passed"]
    # Scientific coverage and the predeclared directional win threshold are
    # orthogonal.  A complete experiment that misses 90% must remain
    # renderable as a complete negative result; only superiority wording is
    # disabled by ``point_estimate_win_gate.passed``.
    document["formal_matrix_complete"] = coverage_complete
    document["official_complete"] = coverage_complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--ppl-root", type=Path)
    parser.add_argument("--mechanism-root", type=Path)
    parser.add_argument("--zeroshot-root", type=Path)
    parser.add_argument("--ruler-root", type=Path)
    parser.add_argument("--longbench-root", type=Path)
    parser.add_argument("--math-root", type=Path)
    parser.add_argument("--longcodeqa-root", type=Path)
    parser.add_argument("--speed-root", type=Path)
    parser.add_argument("--speed-replicates", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "ppl",
            "mechanism",
            "zeroshot",
            "ruler",
            "longbench",
            "math500",
            "longcodeqa",
            "speed",
        ),
        default="all",
    )
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    ppl_root = args.ppl_root or run_root / "results/formal/ppl"
    mechanism_root = args.mechanism_root or run_root / "results/formal/mechanism"
    zeroshot_root = args.zeroshot_root or run_root / "results/formal/zeroshot"
    ruler_root = args.ruler_root or run_root / "results/formal/ruler"
    longbench_root = args.longbench_root or run_root / "results/formal/longbench"
    math_root = args.math_root or run_root / "results/formal/math500"
    longcodeqa_root = args.longcodeqa_root or run_root / "results/formal/longcodeqa"
    speed_root = args.speed_root or run_root / "results/formal/speed"
    output = args.output or run_root / "summaries/coverage_checked_aggregates.json"
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_scopes": {
            "TurboQuant-paper": dict(TURBOQUANT_METHOD_SCOPE),
            "FibQuant-paper": dict(FIBQUANT_METHOD_SCOPE),
            "BlockQuant-paper": dict(BLOCKQUANT_METHOD_SCOPE),
        },
    }
    if args.only in ("all", "ppl"):
        document["ppl"] = verify_ppl(ppl_root)
    if args.only in ("all", "mechanism"):
        reference_ppl = document.get("ppl") or verify_ppl(ppl_root)
        document["mechanism"] = verify_mechanism(mechanism_root, reference_ppl=reference_ppl)
    if args.only in ("all", "zeroshot"):
        document["zeroshot"] = aggregate_zeroshot(zeroshot_root)
    if args.only in ("all", "ruler"):
        document["ruler"] = verify_ruler(ruler_root)
    if args.only in ("all", "longbench"):
        document["longbench"] = verify_longbench(longbench_root)
    if args.only in ("all", "math500"):
        document["math500"] = verify_math500(math_root)
    if args.only in ("all", "longcodeqa"):
        document["longcodeqa"] = aggregate_longcodeqa(longcodeqa_root)
    if args.only in ("all", "speed"):
        document["speed"] = verify_speed(speed_root, expected_replicates=args.speed_replicates)
    _finalize_formal_document(document, run_root=run_root)
    _atomic_json(output.expanduser().resolve(), document)
    print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote {output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
