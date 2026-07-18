#!/usr/bin/env python3
"""Render the coverage-checked HayateQuant formal aggregate as LaTeX.

This is deliberately a fail-closed paper boundary.  It accepts only the
complete schema emitted by ``aggregate_hayatequant_paper_shards.py`` and never
reads raw benchmark directories, historical result bundles, or fallback
constants.  Before producing output it verifies that all formal axes are
present, complete, and tied to one clean-source commit, then re-derives the
headline PPL and corrected zero-shot point estimates, checkpoint win counts,
and robustness intervals, together with the mechanism control.

The command writes three deterministic includes and does not edit ``main.tex``::

    python benchmark/kvcomp-codec/render_hayatequant_formal_results.py \
      --aggregate /path/to/coverage_checked_aggregates.json \
      --output-dir /path/to/generated

Outputs:
  * ``hayatequant_formal_macros.tex`` -- primary point estimates, win counts,
    and interval-gated robustness statements.
  * ``hayatequant_formal_main.tex`` -- an aggregate-derived summary figure and
    compact main-paper result tables.
  * ``hayatequant_formal_tables.tex`` -- the complete formal result tables for
    the appendix.

No generation timestamp or host path is embedded, so identical aggregate bytes
produce identical LaTeX bytes on every machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

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
LONG_MODELS = ("llama-3.1-8b-instruct", "Qwen3-8B")
RATES = ("2", "3", "4")
METHODS = (
    "FullKV",
    "TurboQuant-paper",
    "FibQuant-paper",
    "BlockQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
PROPOSAL_METHODS = ("HQ", "AdaptiveHQ")
QUALITY_RATE_METHODS = {
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
ADAPTIVE_HQ_TEMPORAL_DEPENDENCY = "causal_cache_local_layer_source_rd_statistics"
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
PPL_TEACHER_FORCED_PROTOCOL = "single_forward_causal_mask_token_local_cache"
CAUSAL_INCREMENTAL_TEACHER_FORCED_PROTOCOL = "causal_prefix_prefill_then_token_step_targets"
BLOCKQUANT_TEMPORAL_DEPENDENCY = "dynamic_outlier_channels_over_update_tokens"
WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT2_TEXT_SHA256 = "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
PPL_EVALUATION_CONFIG = {
    "dtype": "bfloat16",
    "run_ppl": True,
    "ppl_max_length": 2048,
    "ppl_stride": 1024,
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
}
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
BLOCKQUANT_SOURCE_TEX_SHA256 = "dc87211edfb3e83285bc19b636d1f6ebe8b054f8c5b69ae0d5c2894b7dac721f"
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
TURBOQUANT_TABLE_LABEL = "TQ Alg.-1 MSE"
FIBQUANT_TABLE_LABEL = "FibQuant v1"
BLOCKQUANT_TABLE_LABEL = "BlockQuant v1 fixed-seed"
ZERO_TASK_PRIMARY = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
}
ZERO_TASK_SIZES = {
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "piqa": 1838,
    "hellaswag": 10042,
    "winogrande": 1267,
}
ZERO_TASK_MANIFEST_KEYS = {
    "arc_easy": "arc_easy_test",
    "arc_challenge": "arc_challenge_test",
    "piqa": "piqa_validation",
    "hellaswag": "hellaswag_validation",
    "winogrande": "winogrande_xl_validation",
}
ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA = "lm_eval_multiple_choice_prompt_v1"
ZEROSHOT_SOURCE_AUDIT_SCOPE = (
    "pinned source row, exact pre-tokenization context/continuation strings, gold, "
    "choice count, and lm-eval character-length normalization denominator; "
    "tokenizer-specific encoded boundary is not claimed by this digest"
)
RULER_CONTEXTS = ("4096", "8192", "16384")
RULER_ROWS_PER_RUN = 1300
RULER_SAMPLES_PER_TASK = 100
RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK = 500
RULER_PROTOCOL_LABEL = "NeMo-Skills RULER v1 100-sample/task protocol"
RULER_SAMPLING_PROTOCOL = (
    "pinned NeMo-Skills adapter: 100 samples/task; canonical "
    "NVIDIA/RULER v1 config: 500 samples/task"
)
RULER_AGGREGATION = "regrade_1300_predictions_then_recompute_13_task_macro"
RULER_SOURCE_RESULT_SCHEMA_VERSION = 3
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
RULER_CHAT_WRAPPER = (
    "generic user message; tokenizer.apply_chat_template with "
    "enable_thinking=false when supported"
)
RULER_MODEL_MAX_POSITION_EMBEDDINGS = {
    "llama-3.1-8b-instruct": 131072,
    "Qwen3-8B": 40960,
}
RULER_AXIS_AUDIT_RECORDS = (
    len(LONG_MODELS) * len(RULER_CONTEXTS) * len(METHODS) * RULER_ROWS_PER_RUN
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
LONGBENCH_TOTAL = sum(LONGBENCH_COUNTS.values())
LONGBENCH_ARCHIVE_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"
LONGBENCH_PROTOCOL_SCOPE = (
    "release_exact_dataset_prompts_middle_truncation_task_generation_budgets_and_scorers; "
    "model_native_chat_template_adaptation_for_new_checkpoints"
)
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
    "Qwen3-8B": {task: 40960 - LONGBENCH_MAX_NEW[task] - 16 for task in LONGBENCH_TASKS},
}
LONGBENCH_EXPECTED_TRUNCATIONS = {
    "llama-3.1-8b-instruct": {task: 0 for task in LONGBENCH_TASKS},
    "Qwen3-8B": {
        task: {"multi_news": 1, "lcc": 5, "repobench-p": 1}.get(task, 0)
        for task in LONGBENCH_TASKS
    },
}
LONGCODE_PLANS = {
    "llama-3.1-8b-instruct": {"32K": 113, "64K": 76, "128K": 92},
    "Qwen3-8B": {"32K": 113},
}
LONGCODE_PROTOCOL_SCOPE = (
    "release_exact_dataset_prompt_parser_and_per_bracket_accuracy; "
    "preregistered_paired_greedy_decoding_adaptation"
)
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
MATH500_PROTOCOL_SCOPE = (
    "adapted_composite_math_verify_ba3d3aa_lighteval_prompt_manager_7c1cd62_"
    "four_shot_query_generation_"
    "prediction_extraction_and_verify; "
    "corrected_inline_math_delimiters_for_bare_gold_answer_fragments"
)
MATH500_GOLD_SERIALIZATION = "wrap_bare_answer_as_inline_math_$...$"
MATH_VERIFY_TASK_SOURCE_COMMIT = "ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b"
MATH500_FEWSHOT_CONTENT_SHA256 = "1360154eb365a22beb1a201dd6ff159e829b6d64d59f69df29530700034c07a4"
MATH500_FEWSHOT_RENDERING = (
    'join(query + " " + answer.strip(), separator="\\n\\n") then main query'
)
MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION = "0.13.0"
MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT = "7c1cd62716b0a198a630c26d781430c54726cd02"
MATH500_PROTOCOL_ID = (
    "math500_adapted_composite_math_verify_ba3d3aa_" "lighteval_prompt_manager_7c1cd62_v1"
)
REQUIRED_AXES = (
    "ppl",
    "mechanism",
    "zeroshot",
    "ruler",
    "longbench",
    "math500",
    "longcodeqa",
    "speed",
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "created_at_utc",
    "method_scopes",
    "commit",
    "formal_provenance",
    "formal_matrix_status",
    "formal_matrix_complete",
    "point_estimate_win_gate",
    "official_complete",
    *REQUIRED_AXES,
}
QUALITY_SOFTWARE_PACKAGES = {
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
}
QUALITY_SEMANTIC_ENVIRONMENT = {
    "ONECOMP_TRITON_CODEC": "0",
    "ONECOMP_TRITON_PREFILL": "0",
    "ONECOMP_COMPILE_CODEC": "0",
    "ONECOMP_COMPILE_DECODE_ONLY": "0",
}
FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE = (
    "full_sha256_submitter_binding_plus_worker_semantic_validator_check"
)
FORMAL_SMOKE_PLAN_VERSION = 3
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
DISPLAY_MODELS = {
    "Llama-2-7b-hf": "Llama-2-7B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "DeepSeek-R1-Distill-Llama-8B": "R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B": "R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-1.5B": "R1-Distill-Qwen-1.5B",
}
RATE_WORD = {"2": "Two", "3": "Three", "4": "Four"}
SPEED_MODEL_MACRO_WORD = {
    "llama-3.1-8b-instruct": "Llama",
    "Qwen3-8B": "Qwen",
}
PAPER_MODEL_EVALUATION_ROLES = {
    model: (
        "development_and_recipe_selection"
        if model == "llama-3.1-8b-instruct"
        else "additional_development_evaluation_independence_not_claimed"
    )
    for model in PAPER_MODELS
}
SPEED_MODEL_EVALUATION_ROLES = {
    model: PAPER_MODEL_EVALUATION_ROLES[model] for model in LONG_MODELS
}
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
SAFE_PREFIX = re.compile(r"[a-z][a-z0-9_]*\Z")
FORBIDDEN_KEY_FRAGMENTS = ("fallback", "hardcoded", "placeholder", "legacy_value")
FORBIDDEN_FORMAL_LABELS = (
    "HayateQuant-frontier",
    "hayate_frontier_vs_turboquant",
    "hayate_scalar_vs_turboquant",
    "hayate_frontier_vs_hayate_scalar",
    "hayate_frontier_block_vq_",
    "hayate_scalar_tq_matched_",
)
SPEED_COMPARISONS = (
    "hq_vs_turboquant",
    "adaptive_hq_vs_turboquant",
)
SPEED_PAIRED_METRICS = (
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


class RenderError(RuntimeError):
    """The aggregate is not safe or complete enough to enter the paper."""


def _load_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise RenderError(f"Non-standard JSON numeric constant {value!r} in {path}")

    def unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RenderError(f"Duplicate JSON object key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"Cannot read aggregate JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderError("Aggregate JSON root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _expected_hayate_codebook_artifacts(bundle: str) -> dict[str, Any]:
    """Rebuild the exact manifest-pinned codebook provenance record."""

    if bundle not in {HQ_CODEBOOK_BUNDLE, ADAPTIVE_HQ_CODEBOOK_BUNDLE}:
        raise RenderError(f"Unexpected formal HayateQuant codebook bundle {bundle!r}")
    codebook_dir = REPO_ROOT / "onecomp/kvcomp/codec/hayate/codebooks"
    manifest_path = (codebook_dir / "manifest.json").resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != HQ_CODEBOOK_MANIFEST_SHA256:
        raise RenderError("Formal HayateQuant codebook manifest is missing or changed")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "hayatequant.shipped_gaussian_codebooks.v1":
        raise RenderError("Formal HayateQuant codebook manifest schema changed")
    required = manifest.get("required_bundles")
    by_name = manifest.get("artifacts")
    filenames = required.get(bundle) if isinstance(required, Mapping) else None
    if not isinstance(filenames, list) or not filenames or not isinstance(by_name, Mapping):
        raise RenderError(f"Malformed formal HayateQuant codebook bundle {bundle!r}")
    artifacts = []
    for filename in filenames:
        spec = by_name.get(filename)
        if not isinstance(filename, str) or not isinstance(spec, Mapping):
            raise RenderError(f"Malformed artifact entry in codebook bundle {bundle!r}")
        path = (codebook_dir / filename).resolve()
        if (
            path.parent != codebook_dir.resolve()
            or not path.is_file()
            or _sha256(path) != spec.get("file_sha256")
            or not _sha256_string(spec.get("tensor_sha256"))
        ):
            raise RenderError(f"Formal HayateQuant artifact is missing or changed: {filename}")
        artifacts.append(
            {
                "source": "shipped_manifest",
                "path": str(path),
                "block_size": spec.get("block_size"),
                "k": spec.get("k"),
                "file_sha256": spec.get("file_sha256"),
                "tensor_sha256": spec.get("tensor_sha256"),
                "origin": spec.get("origin"),
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


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RenderError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise RenderError(f"{label} must be an array")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RenderError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise RenderError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RenderError(f"{label} must be <= {maximum}")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RenderError(f"{label} must be an integer >= {minimum}")
    return value


def _ci(
    value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None
) -> tuple[float, float]:
    items = _sequence(value, label)
    if len(items) != 2:
        raise RenderError(f"{label} must contain exactly two endpoints")
    low = _number(items[0], f"{label}[0]", minimum=minimum, maximum=maximum)
    high = _number(items[1], f"{label}[1]", minimum=minimum, maximum=maximum)
    if low > high:
        raise RenderError(f"{label} endpoints are reversed")
    return low, high


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RenderError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _assert_close(actual: float, expected: float, label: str, *, tolerance: float = 2e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RenderError(f"{label} is stale/inconsistent: {actual} != {expected}")


def _integer_valued_number(value: Any, label: str, *, minimum: int = 0) -> int:
    number = _number(value, label, minimum=float(minimum))
    rounded = round(number)
    if not math.isclose(number, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise RenderError(f"{label} must be integer-valued")
    return int(rounded)


def _validate_ppl_sufficient_statistics(row: Mapping[str, Any], label: str) -> tuple[float, float]:
    ppl = _number(row.get("ppl"), f"{label}.ppl", minimum=0.0)
    mean_nll = _number(row.get("ppl_nll"), f"{label}.ppl_nll")
    nll_sum = _number(row.get("ppl_nll_sum"), f"{label}.ppl_nll_sum")
    n_tokens = _integer(row.get("ppl_n_tokens"), f"{label}.ppl_n_tokens", minimum=1)
    if ppl <= 0.0:
        raise RenderError(f"{label}.ppl must be positive")
    _assert_close(nll_sum / n_tokens, mean_nll, f"{label} NLL numerator", tolerance=1e-10)
    _assert_close(math.log(ppl), mean_nll, f"{label} PPL/NLL", tolerance=1e-10)
    return ppl, mean_nll


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


def _validate_ppl_protocol_and_compression(
    row: Mapping[str, Any], label: str, *, compressed: bool, method: str | None = None
) -> None:
    expected_protocol, expected_dependency = _expected_teacher_forced_contract(method)
    if (
        row.get("ppl_teacher_forced_scoring_protocol") != expected_protocol
        or row.get("ppl_teacher_forced_temporal_dependency") != expected_dependency
    ):
        raise RenderError(f"{label} teacher-forced scoring contract changed")
    ratio = _number(row.get("ppl_compression_ratio"), f"{label}.ppl_compression_ratio")
    aggregation = row.get("ppl_compression_aggregation")
    fp16_bits = row.get("ppl_compression_fp16_bits")
    stored_bits = row.get("ppl_compression_stored_bits")
    if compressed:
        if aggregation != "exact_cache_bit_counters":
            raise RenderError(f"{label} does not use exact cache-bit accounting")
        fp16_bits = _integer(fp16_bits, f"{label}.ppl_compression_fp16_bits", minimum=1)
        stored_bits = _integer(
            stored_bits,
            f"{label}.ppl_compression_stored_bits",
            minimum=1,
        )
        _assert_close(
            ratio,
            fp16_bits / stored_bits,
            f"{label} total-bit compression ratio",
            tolerance=1e-15,
        )
    elif (
        aggregation != "fullkv_no_cache"
        or fp16_bits is not None
        or stored_bits is not None
        or ratio != 1.0
    ):
        raise RenderError(f"{label} FullKV cache accounting changed")


def _validate_blockquant_stored_accounting(row: Mapping[str, Any], label: str) -> None:
    expected_provenance = {
        "method_role": "external_quality_baseline",
        "external_quality_baseline": True,
        "paper_comparable": False,
        "paper_reproduction": False,
        "proposal_method": False,
        "speed_comparison": False,
        "random_seed_protocol": (
            "fixed rotation_seed=0 and codebook_seed=0 same-runtime quality row; "
            "not the paper five-random-seed aggregate"
        ),
    }
    if any(row.get(key) != value for key, value in expected_provenance.items()):
        raise RenderError(f"{label} BlockQuant quality-only provenance changed")
    accounting = _mapping(row.get("stored_bit_accounting"), f"{label}.stored_bit_accounting")
    if (
        accounting.get("scope") != "fixed-width-data-plane-growth-only"
        or accounting.get("physical_packing_claim") is not False
        or accounting.get("head_dim") != 128
    ):
        raise RenderError(f"{label} BlockQuant logical-bit scope changed")
    vectors = _integer(accounting.get("vectors_per_role"), f"{label}.vectors", minimum=1)
    coordinates = _integer(row.get("stored_coordinates"), f"{label}.stored_coordinates", minimum=1)
    if coordinates != 2 * vectors * 128:
        raise RenderError(f"{label} BlockQuant coordinate denominator changed")
    key = _mapping(accounting.get("key"), f"{label}.key_bits")
    value = _mapping(accounting.get("value"), f"{label}.value_bits")
    identity_fields = _integer(
        key.get("dynamic_channel_identity_fields"),
        f"{label}.dynamic_identity_fields",
        minimum=1,
    )
    if identity_fields > vectors or vectors % identity_fields:
        raise RenderError(f"{label} BlockQuant identity field count changed")
    identity_bits = identity_fields * 101
    key_code = vectors * 416
    key_scales = vectors * 32
    key_total = key_code + identity_bits + key_scales
    value_payload = vectors * 256
    value_scales = vectors * 16
    value_total = value_payload + value_scales
    expected_key = {
        "code_bits": key_code,
        "payload_bits": key_code + identity_bits,
        "fp16_scale_bits": key_scales,
        "scale_fields_per_vector": 2,
        "fixed_channel_identity_bits": 0,
        "dynamic_channel_identity_bits": identity_bits,
        "dynamic_channel_identity_fields": identity_fields,
        "dynamic_channel_identity_coding": "combinadic_rank_fixed_width_v1",
        "total_bits": key_total,
        "bits_per_coordinate": key_total / float(vectors * 128),
    }
    expected_value = {
        "payload_bits": value_payload,
        "fp16_scale_bits": value_scales,
        "scale_fields_per_vector": 1,
        "total_bits": value_total,
        "bits_per_coordinate": value_total / float(vectors * 128),
    }
    total = key_total + value_total
    if (
        key != expected_key
        or value != expected_value
        or accounting.get("total_bits") != total
        or row.get("stored_bits") != total
        or accounting.get("excluded_global_state")
        != [
            "dense Haar rotations",
            "data-free codebooks",
            "Python control/container bytes",
            "allocator and unpacked research tensor bytes",
        ]
    ):
        raise RenderError(f"{label} BlockQuant logical-bit arithmetic changed")
    _assert_close(
        _number(row.get("stored_bits_per_coord"), f"{label}.stored_bits_per_coord"),
        total / coordinates,
        f"{label}.stored_bits_per_coord",
        tolerance=1e-15,
    )


def _validate_zeroshot_compression(
    row: Mapping[str, Any], label: str, *, compressed: bool
) -> tuple[int | None, int | None]:
    ratio = _number(row.get("compression_ratio"), f"{label}.compression_ratio")
    aggregation = row.get("compression_aggregation")
    fp16_bits = row.get("compression_fp16_bits")
    stored_bits = row.get("compression_stored_bits")
    if compressed:
        if aggregation != "exact_cache_bit_counters":
            raise RenderError(f"{label} does not use exact zero-shot bit totals")
        fp16_bits = _integer(fp16_bits, f"{label}.compression_fp16_bits", minimum=1)
        stored_bits = _integer(stored_bits, f"{label}.compression_stored_bits", minimum=1)
        _assert_close(
            ratio,
            fp16_bits / stored_bits,
            f"{label} exact compression ratio",
            tolerance=1e-15,
        )
        return fp16_bits, stored_bits
    if (
        aggregation != "fullkv_no_cache"
        or fp16_bits is not None
        or stored_bits is not None
        or ratio != 1.0
    ):
        raise RenderError(f"{label} FullKV zero-shot compression accounting changed")
    return None, None


def _normal_binomial_ci(estimate: float, n: int) -> tuple[float, float]:
    if n <= 1:
        stderr = 0.0
    else:
        # This mirrors the aggregate producer's sample-variance calculation.
        correct = estimate * n
        variance = (correct * (1.0 - estimate) ** 2 + (n - correct) * estimate**2) / (n - 1)
        stderr = math.sqrt(max(0.0, variance / n))
    half = 1.959963984540054 * stderr
    return max(0.0, estimate - half), min(1.0, estimate + half)


def _wilson_ci(correct: int, n: int) -> tuple[float, float]:
    estimate = correct / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    centre = (estimate + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _reject_forbidden_keys(value: Any, path: str = "aggregate") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RenderError(f"{path} contains a non-string key")
            lowered = key.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise RenderError(f"{path}.{key} is a forbidden fallback/legacy field")
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _reject_legacy_formal_labels(value: Any, path: str = "aggregate") -> None:
    """Reject legacy proposal/speed labels at the formal paper boundary."""

    if isinstance(value, dict):
        for key, child in value.items():
            if any(label in key for label in FORBIDDEN_FORMAL_LABELS):
                raise RenderError(f"{path}.{key} uses a rejected legacy formal label")
            _reject_legacy_formal_labels(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_formal_labels(child, f"{path}[{index}]")
    elif isinstance(value, str) and any(label in value for label in FORBIDDEN_FORMAL_LABELS):
        raise RenderError(f"{path} contains a rejected legacy formal label")


def _validate_sources(axis: Mapping[str, Any], label: str) -> None:
    sources = _sequence(axis.get("sources"), f"{label}.sources")
    if not sources:
        raise RenderError(f"{label}.sources must not be empty")
    for index, raw in enumerate(sources):
        source = _mapping(raw, f"{label}.sources[{index}]")
        if not isinstance(source.get("path"), str) or not source["path"]:
            raise RenderError(f"{label}.sources[{index}].path is missing")
        _integer(source.get("bytes"), f"{label}.sources[{index}].bytes", minimum=1)
        digest = source.get("sha256")
        if not isinstance(digest, str) or not SHA64.fullmatch(digest.lower()):
            raise RenderError(f"{label}.sources[{index}].sha256 is invalid")


def _validate_commit(axis: Mapping[str, Any], top_commit: str, label: str) -> None:
    if axis.get("commit") != top_commit:
        raise RenderError(f"{label} was not produced from aggregate commit {top_commit}")
    _validate_sources(axis, label)


def _validate_model_evaluation_roles(
    axis: Mapping[str, Any], models: Sequence[str], label: str
) -> None:
    expected = {model: PAPER_MODEL_EVALUATION_ROLES[model] for model in models}
    if axis.get("model_evaluation_roles") != expected:
        raise RenderError(f"{label} model evaluation/selection roles changed")


def _student_t_critical(n: int) -> float:
    # The formal schema fixes the only two panel sizes used for paper claims.
    if n == 7:
        return 2.4469118511449692  # df=6, two-sided 95%
    if n == 2:
        return 12.706204736432095  # df=1, two-sided 95%
    raise RenderError(f"No predeclared Student-t critical value for n={n}")


def _bonferroni_three_rate_t_critical(n: int) -> float:
    """Two-sided 95% FWER critical value for three predeclared rates."""
    if n == 7:
        return 3.287455157088189  # t_6(1 - 0.05 / (2 * 3))
    raise RenderError(f"No three-rate Bonferroni Student-t critical value for n={n}")


def _mean_t_ci(values: Sequence[float], label: str) -> tuple[float, float, float, float]:
    if len(values) not in (2, 7) or not all(math.isfinite(value) for value in values):
        raise RenderError(f"{label} must contain the fixed 2- or 7-model panel")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    half = _student_t_critical(len(values)) * stderr
    return mean, stderr, mean - half, mean + half


def _win_tie_loss_counts(
    values: Sequence[float], *, null: float, lower_is_better: bool
) -> tuple[int, int, int]:
    """Return HQ wins/ties/losses from unrounded checkpoint point estimates."""

    wins = ties = losses = 0
    for value in values:
        if value == null:
            ties += 1
        elif (value < null) == lower_is_better:
            wins += 1
        else:
            losses += 1
    if wins + ties + losses != len(values):  # pragma: no cover - defensive invariant
        raise RenderError("Point-estimate win accounting is incomplete")
    return wins, ties, losses


def _ppl_point_statement(ratio: float) -> str:
    if ratio < 1.0:
        return "lower point-estimate PPL on this fixed panel"
    if ratio > 1.0:
        return "higher point-estimate PPL on this fixed panel"
    return "equal point-estimate PPL on this fixed panel"


def _ppl_point_label(ratio: float) -> str:
    if ratio < 1.0:
        return "lower"
    if ratio > 1.0:
        return "higher"
    return "equal"


def _ppl_robustness_statement(ci: tuple[float, float]) -> str:
    if ci[1] < 1.0:
        return "simultaneous interval supports lower PPL on this fixed panel"
    if ci[0] > 1.0:
        return "simultaneous interval supports higher PPL on this fixed panel"
    return "simultaneous interval does not establish directional robustness on this fixed panel"


def _ppl_robustness_label(ci: tuple[float, float]) -> str:
    if ci[1] < 1.0:
        return "lower supported"
    if ci[0] > 1.0:
        return "higher supported"
    return "not established"


def _zeroshot_point_statement(delta: float, *, fixed_panel: bool = True) -> str:
    scope = " on this fixed panel" if fixed_panel else ""
    if delta > 0.0:
        return f"higher point-estimate ZS-Mean{scope}"
    if delta < 0.0:
        return f"lower point-estimate ZS-Mean{scope}"
    return f"equal point-estimate ZS-Mean{scope}"


def _zeroshot_point_label(delta: float) -> str:
    if delta > 0.0:
        return "higher"
    if delta < 0.0:
        return "lower"
    return "equal"


def _zeroshot_robustness_statement(ci: tuple[float, float]) -> str:
    if ci[0] > 0.0:
        return "simultaneous interval supports higher ZS-Mean on this fixed panel"
    if ci[1] < 0.0:
        return "simultaneous interval supports lower ZS-Mean on this fixed panel"
    return "simultaneous interval does not establish directional robustness on this fixed panel"


def _zeroshot_robustness_label(ci: tuple[float, float]) -> str:
    if ci[0] > 0.0:
        return "higher supported"
    if ci[1] < 0.0:
        return "lower supported"
    return "not established"


def _speed_claim(ci: tuple[float, float], candidate: str = "HQ") -> str:
    """Return the preregistered non-directional speed-reporting statement.

    The formal matrix reports many pointwise, unadjusted intervals across
    models, rates, compile modes, endpoints, and method comparisons.  Their
    numerical ratio and CI remain useful descriptive measurements, but no
    individual interval is promoted to a directional speed claim.
    """

    del ci, candidate
    return "descriptive ratio only; no directional speed claim"


def _validate_ppl(axis: Mapping[str, Any], commit: str) -> dict:
    _validate_commit(axis, commit, "ppl")
    _validate_model_evaluation_roles(axis, PAPER_MODELS, "ppl")
    if (
        axis.get("benchmark") != "WikiText-2 perplexity"
        or axis.get("official_complete") is not True
        or axis.get("dataset_revision") != WIKITEXT2_REVISION
        or axis.get("text_sha256") != WIKITEXT2_TEXT_SHA256
        or axis.get("evaluation_config") != PPL_EVALUATION_CONFIG
        or axis.get("compression_accounting") != PPL_COMPRESSION_ACCOUNTING
        or axis.get("teacher_forced_scoring_contract")
        != {
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
        }
    ):
        raise RenderError("ppl axis is not the complete formal WikiText-2 benchmark")
    inference_scope = _mapping(axis.get("inference_scope"), "ppl.inference_scope")
    if inference_scope != {
        "student_t_statistical_unit": "fixed_checkpoint",
        "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
        "interval_interpretation": ("descriptive_uncertainty_across_the_fixed_checkpoint_panel"),
        "population_generalization": False,
    }:
        raise RenderError("PPL fixed-checkpoint/non-population inference scope changed")
    models = _mapping(axis.get("models"), "ppl.models")
    _exact_keys(models, set(PAPER_MODELS), "ppl.models")
    primary = _mapping(
        axis.get("secondary_interval_robustness_vs_turboquant"),
        "ppl.secondary_interval_robustness_vs_turboquant",
    )
    _exact_keys(primary, set(PROPOSAL_METHODS), "ppl proposal robustness")
    ratios: dict[str, dict[str, list[float]]] = {
        proposal: {rate: [] for rate in RATES} for proposal in PROPOSAL_METHODS
    }
    for model in PAPER_MODELS:
        model_doc = _mapping(models[model], f"ppl.models.{model}")
        if model_doc.get("official_complete") is not True:
            raise RenderError(f"PPL model is incomplete: {model}")
        full = _mapping(model_doc.get("fullkv"), f"ppl.models.{model}.fullkv")
        if (
            full.get("ppl_dataset_revision") != WIKITEXT2_REVISION
            or full.get("ppl_text_sha256") != WIKITEXT2_TEXT_SHA256
        ):
            raise RenderError(f"PPL FullKV source identity changed for {model}")
        _validate_ppl_sufficient_statistics(full, f"ppl.models.{model}.fullkv")
        _validate_ppl_protocol_and_compression(
            full, f"ppl.models.{model}.fullkv", compressed=False
        )
        rates = _mapping(model_doc.get("rates"), f"ppl.models.{model}.rates")
        _exact_keys(rates, set(RATES), f"ppl.models.{model}.rates")
        for rate in RATES:
            methods = _mapping(rates[rate], f"ppl.{model}.{rate}")
            _exact_keys(
                methods,
                set(QUALITY_RATE_METHODS[rate]),
                f"ppl.{model}.{rate}",
            )
            typed_methods = {
                method: _mapping(row, f"ppl.{model}.{rate}.{method}")
                for method, row in methods.items()
            }
            tq_doc = typed_methods["TurboQuant-paper"]
            fib_doc = typed_methods["FibQuant-paper"]
            block_doc = typed_methods.get("BlockQuant-paper")
            if any(
                row.get("dataset_revision") != WIKITEXT2_REVISION
                or row.get("text_sha256") != WIKITEXT2_TEXT_SHA256
                for row in typed_methods.values()
            ):
                raise RenderError(f"PPL compressed source identity changed for {model}/{rate}")
            _, tq_nll = _validate_ppl_sufficient_statistics(tq_doc, f"ppl.{model}.{rate}.TQ")
            _validate_ppl_sufficient_statistics(fib_doc, f"ppl.{model}.{rate}.FibQuant")
            if block_doc is not None:
                _validate_ppl_sufficient_statistics(block_doc, f"ppl.{model}.{rate}.BlockQuant")
            _validate_ppl_protocol_and_compression(
                tq_doc, f"ppl.{model}.{rate}.TQ", compressed=True
            )
            _validate_ppl_protocol_and_compression(
                fib_doc, f"ppl.{model}.{rate}.FibQuant", compressed=True
            )
            if block_doc is not None:
                _validate_ppl_protocol_and_compression(
                    block_doc,
                    f"ppl.{model}.{rate}.BlockQuant",
                    compressed=True,
                    method="BlockQuant-paper",
                )
                _validate_blockquant_stored_accounting(block_doc, f"ppl.{model}.{rate}.BlockQuant")
            tq_counters = (
                tq_doc.get("ppl_compression_fp16_bits"),
                tq_doc.get("ppl_compression_stored_bits"),
            )
            for matched in ("FibQuant-paper", *PROPOSAL_METHODS):
                matched_doc = typed_methods[matched]
                _validate_ppl_protocol_and_compression(
                    matched_doc,
                    f"ppl.{model}.{rate}.{matched}",
                    compressed=True,
                    method=matched,
                )
                _, matched_nll = _validate_ppl_sufficient_statistics(
                    matched_doc, f"ppl.{model}.{rate}.{matched}"
                )
                if (
                    matched_doc.get("ppl_compression_fp16_bits"),
                    matched_doc.get("ppl_compression_stored_bits"),
                ) != tq_counters:
                    raise RenderError(
                        f"PPL exact payload counters differ for {model}/{rate}/{matched}"
                    )
                evidence = _mapping(
                    matched_doc.get("payload_match_vs_turboquant"),
                    f"ppl.{model}.{rate}.{matched}.payload_match",
                )
                if evidence != {
                    "exact_integer_counters": True,
                    "ppl_compression_fp16_bits": tq_counters[0],
                    "ppl_compression_stored_bits": tq_counters[1],
                }:
                    raise RenderError(
                        f"PPL exact payload-match evidence changed for {model}/{rate}/{matched}"
                    )
                if matched in PROPOSAL_METHODS:
                    expected_provenance = HAYATE_FORMAL_PROVENANCE[matched]
                    if any(
                        matched_doc.get(key) != value for key, value in expected_provenance.items()
                    ):
                        raise RenderError(
                            f"PPL proposal identity changed for {model}/{rate}/{matched}"
                        )
                    ratios[matched][rate].append(math.exp(matched_nll - tq_nll))

    normalized_primary: dict[str, dict[str, dict[str, Any]]] = {}
    for proposal in PROPOSAL_METHODS:
        proposal_primary = _mapping(primary[proposal], f"ppl robustness {proposal}")
        _exact_keys(proposal_primary, set(RATES), f"ppl robustness {proposal}")
        normalized_primary[proposal] = {}
        for rate in RATES:
            row = _mapping(proposal_primary[rate], f"ppl robustness {proposal}/{rate}")
            if (
                row.get("proposal") != proposal
                or row.get("estimand")
                != "geometric_mean_within_model_ppl_ratio_proposal_over_turboquant"
                or row.get("claim_rule") != "interval_is_secondary_robustness_only"
                or row.get("student_t_statistical_unit") != "fixed_checkpoint"
                or row.get("checkpoint_panel")
                != "seven_preselected_family_correlated_checkpoints_not_iid"
                or row.get("population_generalization") is not False
                or row.get("nominal_interval_status")
                != "secondary_robustness_not_primary_win_gate"
                or row.get("multiplicity_family") != "three_predeclared_average_bit_rates_2_3_4"
                or row.get("familywise_alpha") != 0.05
                or not math.isclose(
                    _number(row.get("per_rate_two_sided_alpha"), f"PPL alpha {proposal}/{rate}"),
                    0.05 / 3.0,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or row.get("n_models") != len(PAPER_MODELS)
            ):
                raise RenderError(f"PPL robustness contract changed at {proposal}/{rate}")
            log_ratios = [math.log(value) for value in ratios[proposal][rate]]
            mean_log, stderr, low_log, high_log = _mean_t_ci(
                log_ratios, f"PPL log ratios at {proposal}/{rate}"
            )
            ratio = math.exp(mean_log)
            nominal_ci = (math.exp(low_log), math.exp(high_log))
            simultaneous_half = _bonferroni_three_rate_t_critical(len(PAPER_MODELS)) * stderr
            simultaneous_ci = (
                math.exp(mean_log - simultaneous_half),
                math.exp(mean_log + simultaneous_half),
            )
            for key, expected in (
                ("mean_log_ppl_ratio", mean_log),
                ("mean_log_ppl_ratio_standard_error", stderr),
                ("geometric_mean_ppl_ratio", ratio),
                ("relative_ppl_change_percent", 100.0 * (ratio - 1.0)),
            ):
                _assert_close(
                    _number(row.get(key), f"PPL {key} {proposal}/{rate}"),
                    expected,
                    f"PPL {key} {proposal}/{rate}",
                )
            stored_nominal = _ci(
                row.get("student_t_ci95_geometric_mean_ratio"),
                f"PPL nominal CI {proposal}/{rate}",
                minimum=0.0,
            )
            stored_simultaneous = _ci(
                row.get("bonferroni_simultaneous_familywise95_geometric_mean_ratio"),
                f"PPL simultaneous CI {proposal}/{rate}",
                minimum=0.0,
            )
            for observed, expected, suffix in (
                (stored_nominal[0], nominal_ci[0], "nominal low"),
                (stored_nominal[1], nominal_ci[1], "nominal high"),
                (stored_simultaneous[0], simultaneous_ci[0], "simultaneous low"),
                (stored_simultaneous[1], simultaneous_ci[1], "simultaneous high"),
            ):
                _assert_close(observed, expected, f"PPL {proposal}/{rate} {suffix}")
            per_model = _mapping(
                row.get("per_model_ratio"), f"PPL per-model ratio {proposal}/{rate}"
            )
            _exact_keys(per_model, set(PAPER_MODELS), f"PPL per-model {proposal}/{rate}")
            for model, expected in zip(PAPER_MODELS, ratios[proposal][rate]):
                _assert_close(
                    _number(per_model[model], f"PPL ratio {proposal}/{model}/{rate}"),
                    expected,
                    f"PPL per-model ratio {proposal}/{model}/{rate}",
                )
            wins, ties, losses = _win_tie_loss_counts(
                ratios[proposal][rate], null=1.0, lower_is_better=True
            )
            normalized_primary[proposal][rate] = {
                "ratio": ratio,
                "ci": simultaneous_ci,
                "nominal_ci": nominal_ci,
                "relative_percent": 100.0 * (ratio - 1.0),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "point_statement": _ppl_point_statement(ratio),
                "robustness_statement": _ppl_robustness_statement(simultaneous_ci),
            }
    return {
        "models": models,
        "primary": normalized_primary,
        "inference_scope": dict(inference_scope),
    }


def _validate_zeroshot_source_audit(axis: Mapping[str, Any]) -> dict[str, str]:
    audit = _mapping(axis.get("pinned_source_prompt_audit"), "zeroshot source audit")
    if (
        audit.get("protocol_input_hash_schema") != ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA
        or audit.get("scope") != ZEROSHOT_SOURCE_AUDIT_SCOPE
    ):
        raise RenderError("Zero-shot pinned source/prompt audit contract changed")
    manifest = _mapping(audit.get("input_manifest"), "zeroshot source audit manifest")
    manifest_path = manifest.get("path")
    manifest_sha256 = manifest.get("sha256")
    if (
        not isinstance(manifest_path, str)
        or not Path(manifest_path).is_absolute()
        or not _sha256_string(manifest_sha256)
    ):
        raise RenderError("Zero-shot source audit has no immutable input-manifest identity")
    tasks = _mapping(audit.get("tasks"), "zeroshot source audit tasks")
    _exact_keys(tasks, set(ZERO_TASK_PRIMARY), "zeroshot source audit tasks")
    protocol_hashes = {}
    for task, expected_rows in ZERO_TASK_SIZES.items():
        task_audit = _mapping(tasks[task], f"zeroshot source audit {task}")
        protocol_hash = task_audit.get("full_split_protocol_input_sha256")
        if (
            task_audit.get("manifest_key") != ZERO_TASK_MANIFEST_KEYS[task]
            or task_audit.get("rows") != expected_rows
            or not _sha256_string(protocol_hash)
        ):
            raise RenderError(f"Zero-shot pinned source audit changed for {task}")
        source_files = _sequence(
            task_audit.get("source_files"), f"zeroshot source audit files {task}"
        )
        if not source_files:
            raise RenderError(f"Zero-shot source audit has no Arrow ledger for {task}")
        for index, raw_source in enumerate(source_files):
            source = _mapping(raw_source, f"zeroshot source audit {task} file {index}")
            path = source.get("path")
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or not isinstance(source.get("bytes"), int)
                or isinstance(source.get("bytes"), bool)
                or source["bytes"] <= 0
                or not _sha256_string(source.get("sha256"))
            ):
                raise RenderError(f"Malformed pinned Arrow content ledger for {task}")
        protocol_hashes[task] = str(protocol_hash)
    return protocol_hashes


def _validate_zeroshot_method(
    method_doc: Mapping[str, Any],
    *,
    method: str,
    label: str,
    compressed: bool,
    expected_protocol_hashes: Mapping[str, str],
) -> tuple[float, Mapping[str, Any]]:
    if method_doc.get("official_complete") is not True:
        raise RenderError(f"Incomplete zero-shot method {label}")
    tasks = _mapping(method_doc.get("tasks"), f"{label}.tasks")
    _exact_keys(tasks, set(ZERO_TASK_PRIMARY), f"{label}.tasks")
    primary_scores = []
    acc_scores = []
    acc_norm_scores = []
    task_fp16_bits = []
    task_stored_bits = []
    expected_protocol, expected_dependency = _expected_teacher_forced_contract(method)
    for task, primary_metric in ZERO_TASK_PRIMARY.items():
        task_doc = _mapping(tasks[task], f"{label}.{task}")
        n = ZERO_TASK_SIZES[task]
        if (
            task_doc.get("n") != n
            or task_doc.get("coverage") != [0, n]
            or task_doc.get("audit_records_validated") != n
            or task_doc.get("teacher_forced_scoring_protocol") != expected_protocol
            or task_doc.get("teacher_forced_temporal_dependency") != expected_dependency
            or task_doc.get("protocol_input_hash_schema") != ZEROSHOT_PROTOCOL_INPUT_HASH_SCHEMA
            or task_doc.get("protocol_input_sha256") != expected_protocol_hashes[task]
        ):
            raise RenderError(f"Incomplete zero-shot coverage for {label}/{task}")
        correct = _integer_valued_number(task_doc.get("correct"), f"{label}/{task}.correct")
        correct_norm = _integer_valued_number(
            task_doc.get("correct_norm"), f"{label}/{task}.correct_norm"
        )
        if correct > n or correct_norm > n:
            raise RenderError(f"Zero-shot correct count exceeds n for {label}/{task}")
        acc = _number(task_doc.get("acc"), f"{label}/{task}.acc", minimum=0.0, maximum=1.0)
        acc_norm = _number(
            task_doc.get("acc_norm"), f"{label}/{task}.acc_norm", minimum=0.0, maximum=1.0
        )
        _assert_close(acc, correct / n, f"{label}/{task}.acc")
        _assert_close(acc_norm, correct_norm / n, f"{label}/{task}.acc_norm")
        if task_doc.get("primary_metric") != primary_metric:
            raise RenderError(f"Wrong zero-shot primary metric for {task}")
        primary = _number(
            task_doc.get("primary_score"),
            f"{label}/{task}.primary_score",
            minimum=0.0,
            maximum=1.0,
        )
        _assert_close(
            primary,
            acc_norm if primary_metric == "acc_norm" else acc,
            f"{label}/{task}.primary_score",
        )
        fp16_bits, stored_bits = _validate_zeroshot_compression(
            task_doc, f"{label}/{task}", compressed=compressed
        )
        if compressed:
            task_fp16_bits.append(int(fp16_bits))
            task_stored_bits.append(int(stored_bits))
        primary_scores.append(primary)
        acc_scores.append(acc)
        acc_norm_scores.append(acc_norm)
    average = math.fsum(primary_scores) / len(primary_scores)
    _assert_close(
        _number(method_doc.get("avg_primary_score"), f"{label}.avg_primary_score"),
        average,
        f"{label}.avg_primary_score",
    )
    _assert_close(
        _number(method_doc.get("avg_acc"), f"{label}.avg_acc"),
        math.fsum(acc_scores) / len(acc_scores),
        f"{label}.avg_acc",
    )
    _assert_close(
        _number(method_doc.get("avg_acc_norm"), f"{label}.avg_acc_norm"),
        math.fsum(acc_norm_scores) / len(acc_norm_scores),
        f"{label}.avg_acc_norm",
    )
    if method_doc.get("n_total") != sum(ZERO_TASK_SIZES.values()):
        raise RenderError(f"Wrong zero-shot total denominator for {label}")
    method_fp16_bits, method_stored_bits = _validate_zeroshot_compression(
        method_doc, label, compressed=compressed
    )
    if compressed and (
        method_fp16_bits != sum(task_fp16_bits) or method_stored_bits != sum(task_stored_bits)
    ):
        raise RenderError(f"Zero-shot method/task bit totals disagree for {label}")
    return average, tasks


def _validate_paired_zero_task(
    comparison: Mapping[str, Any],
    *,
    hq_task: Mapping[str, Any],
    tq_task: Mapping[str, Any],
    task: str,
    label: str,
) -> tuple[float, float]:
    n = ZERO_TASK_SIZES[task]
    metric = "correct_norm" if ZERO_TASK_PRIMARY[task] == "acc_norm" else "correct"
    if (
        comparison.get("metric") != metric
        or comparison.get("n_paired") != n
        or comparison.get("claim_rule") != "claim_improvement_only_if_ci95_excludes_zero"
        or comparison.get("statistical_unit") != "example_within_fixed_task_and_checkpoint"
        or comparison.get("task_and_checkpoint_status") != "fixed_not_randomly_sampled"
        or comparison.get("population_generalization") is not False
        or comparison.get("interval_status")
        != "nominal_per_task_not_an_across_rate_familywise_interval"
    ):
        raise RenderError(f"Paired zero-shot contract changed for {label}/{task}")
    hq_correct = _integer_valued_number(hq_task.get(f"{metric}"), f"{label}/{task}.hq")
    tq_correct = _integer_valued_number(tq_task.get(f"{metric}"), f"{label}/{task}.tq")
    if (
        comparison.get("hq_correct") != hq_correct
        or comparison.get("turboquant_correct") != tq_correct
    ):
        raise RenderError(f"Paired counts are stale for {label}/{task}")
    discordant = _mapping(comparison.get("discordant"), f"{label}/{task}.discordant")
    hq_only = _integer(discordant.get("hq_only_correct"), f"{label}/{task}.hq_only")
    tq_only = _integer(discordant.get("turboquant_only_correct"), f"{label}/{task}.tq_only")
    delta = (hq_only - tq_only) / n
    _assert_close(
        _number(comparison.get("delta_accuracy_hq_minus_turboquant"), f"{label}/{task}.delta"),
        delta,
        f"{label}/{task}.delta",
    )
    _assert_close(delta, (hq_correct - tq_correct) / n, f"{label}/{task}.paired counts")
    variance = ((hq_only + tq_only) - n * delta * delta) / (n - 1)
    stderr = math.sqrt(max(0.0, variance / n))
    _assert_close(
        _number(comparison.get("paired_standard_error"), f"{label}/{task}.stderr"),
        stderr,
        f"{label}/{task}.stderr",
    )
    expected_ci = (
        max(-1.0, delta - 1.959963984540054 * stderr),
        min(1.0, delta + 1.959963984540054 * stderr),
    )
    stored_ci = _ci(
        comparison.get("paired_normal_ci95"), f"{label}/{task}.ci", minimum=-1.0, maximum=1.0
    )
    _assert_close(stored_ci[0], expected_ci[0], f"{label}/{task}.ci low")
    _assert_close(stored_ci[1], expected_ci[1], f"{label}/{task}.ci high")
    return delta, stderr


def _validate_zeroshot(axis: Mapping[str, Any], commit: str) -> dict:
    _validate_commit(axis, commit, "zeroshot")
    _validate_model_evaluation_roles(axis, PAPER_MODELS, "zeroshot")
    if (
        axis.get("benchmark") != "corrected_zero_shot_five_task"
        or axis.get("official_complete") is not True
        or axis.get("sample_seed") != 2027
        or axis.get("teacher_forced_scoring_contract")
        != {
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
        }
        or axis.get("compression_accounting") != ZEROSHOT_COMPRESSION_ACCOUNTING
    ):
        raise RenderError("zeroshot axis is not the corrected complete formal benchmark")
    inference_scope = _mapping(axis.get("inference_scope"), "zeroshot.inference_scope")
    if inference_scope != {
        "within_checkpoint_paired_unit": ("example_within_each_fixed_task_and_checkpoint"),
        "task_panel": "five_fixed_tasks_not_a_random_task_population",
        "cross_checkpoint_student_t_unit": "fixed_checkpoint",
        "checkpoint_panel": ("seven_preselected_family_correlated_checkpoints_not_iid"),
        "population_generalization": False,
    }:
        raise RenderError("Zero-shot fixed-task/checkpoint inference scope changed")
    protocol_hashes = _validate_zeroshot_source_audit(axis)
    models = _mapping(axis.get("models"), "zeroshot.models")
    _exact_keys(models, set(PAPER_MODELS), "zeroshot.models")
    stored_primary = _mapping(
        axis.get("secondary_interval_robustness_vs_turboquant"),
        "zeroshot.secondary_interval_robustness_vs_turboquant",
    )
    _exact_keys(stored_primary, set(PROPOSAL_METHODS), "zeroshot proposal robustness")
    model_deltas: dict[str, dict[str, list[float]]] = {
        proposal: {rate: [] for rate in RATES} for proposal in PROPOSAL_METHODS
    }
    for model in PAPER_MODELS:
        model_doc = _mapping(models[model], f"zeroshot.models.{model}")
        if model_doc.get("official_complete") is not True:
            raise RenderError(f"Zero-shot model is incomplete: {model}")
        rates = _mapping(model_doc.get("rates"), f"zeroshot.{model}.rates")
        _exact_keys(rates, {"full", *RATES}, f"zeroshot.{model}.rates")
        full = _mapping(rates["full"], f"zeroshot.{model}.full")
        _exact_keys(full, {"FullKV"}, f"zeroshot.{model}.full")
        _validate_zeroshot_method(
            _mapping(full["FullKV"], f"zeroshot.{model}.full.FullKV"),
            method="FullKV",
            label=f"zeroshot.{model}.full.FullKV",
            compressed=False,
            expected_protocol_hashes=protocol_hashes,
        )
        for rate in RATES:
            rate_doc = _mapping(rates[rate], f"zeroshot.{model}.{rate}")
            _exact_keys(
                rate_doc,
                {*QUALITY_RATE_METHODS[rate], "paired_proposals_vs_turboquant"},
                f"zeroshot.{model}.{rate}",
            )
            method_values = {}
            method_tasks = {}
            for method in QUALITY_RATE_METHODS[rate]:
                method_doc = _mapping(rate_doc[method], f"zeroshot.{model}.{rate}.{method}")
                method_values[method], method_tasks[method] = _validate_zeroshot_method(
                    method_doc,
                    method=method,
                    label=f"zeroshot.{model}.{rate}.{method}",
                    compressed=True,
                    expected_protocol_hashes=protocol_hashes,
                )
            tq_doc = _mapping(
                rate_doc["TurboQuant-paper"], f"zeroshot.{model}.{rate}.TurboQuant-paper"
            )
            tq_counters = (
                tq_doc.get("compression_fp16_bits"),
                tq_doc.get("compression_stored_bits"),
            )
            for matched in ("FibQuant-paper", *PROPOSAL_METHODS):
                matched_doc = _mapping(rate_doc[matched], f"zeroshot.{model}.{rate}.{matched}")
                if (
                    matched_doc.get("compression_fp16_bits"),
                    matched_doc.get("compression_stored_bits"),
                ) != tq_counters:
                    raise RenderError(
                        f"Zero-shot exact payload counters differ for {model}/{rate}/{matched}"
                    )
                if matched_doc.get("payload_match_vs_turboquant") != {
                    "exact_integer_counters": True,
                    "compression_fp16_bits": tq_counters[0],
                    "compression_stored_bits": tq_counters[1],
                }:
                    raise RenderError(
                        f"Zero-shot payload-match evidence changed for {model}/{rate}/{matched}"
                    )
                for task in ZERO_TASK_PRIMARY:
                    tq_task = method_tasks["TurboQuant-paper"][task]
                    matched_task = method_tasks[matched][task]
                    if (
                        matched_task.get("compression_fp16_bits"),
                        matched_task.get("compression_stored_bits"),
                    ) != (
                        tq_task.get("compression_fp16_bits"),
                        tq_task.get("compression_stored_bits"),
                    ):
                        raise RenderError(
                            f"Zero-shot task payload counters differ for {model}/{rate}/{task}/{matched}"
                        )
            paired_proposals = _mapping(
                rate_doc["paired_proposals_vs_turboquant"],
                f"zeroshot paired proposals {model}/{rate}",
            )
            _exact_keys(
                paired_proposals,
                set(PROPOSAL_METHODS),
                f"zeroshot paired proposals {model}/{rate}",
            )
            for proposal in PROPOSAL_METHODS:
                paired = _mapping(
                    paired_proposals[proposal],
                    f"zeroshot paired {proposal}/{model}/{rate}",
                )
                if (
                    paired.get("proposal") != proposal
                    or paired.get("claim_rule") != "interval_is_secondary_robustness_only"
                    or paired.get("statistical_unit")
                    != "paired_example_within_each_fixed_task_and_checkpoint"
                    or paired.get("task_panel") != "five_fixed_tasks_not_a_random_task_population"
                    or paired.get("population_generalization") is not False
                    or paired.get("interval_status") != "secondary_robustness_not_primary_win_gate"
                ):
                    raise RenderError(
                        f"Zero-shot robustness contract changed for {proposal}/{model}/{rate}"
                    )
                paired_tasks = _mapping(
                    paired.get("per_task"), f"paired tasks {proposal}/{model}/{rate}"
                )
                _exact_keys(
                    paired_tasks,
                    set(ZERO_TASK_PRIMARY),
                    f"paired tasks {proposal}/{model}/{rate}",
                )
                task_deltas = []
                task_stderr = []
                for task in ZERO_TASK_PRIMARY:
                    delta_task, stderr_task = _validate_paired_zero_task(
                        _mapping(paired_tasks[task], f"paired {proposal}/{model}/{rate}/{task}"),
                        hq_task=method_tasks[proposal][task],
                        tq_task=method_tasks["TurboQuant-paper"][task],
                        task=task,
                        label=f"zeroshot.{proposal}.{model}.{rate}",
                    )
                    task_deltas.append(delta_task)
                    task_stderr.append(stderr_task)
                delta = math.fsum(task_deltas) / len(task_deltas)
                _assert_close(
                    _number(
                        paired.get("delta_macro_proposal_minus_turboquant"),
                        f"zero-shot delta {proposal}/{model}/{rate}",
                        minimum=-1.0,
                        maximum=1.0,
                    ),
                    delta,
                    f"zero-shot macro delta {proposal}/{model}/{rate}",
                )
                macro_stderr = math.sqrt(math.fsum(value * value for value in task_stderr)) / len(
                    task_stderr
                )
                _assert_close(
                    _number(
                        paired.get("stratified_paired_standard_error"),
                        f"zero-shot macro stderr {proposal}/{model}/{rate}",
                    ),
                    macro_stderr,
                    f"zero-shot macro stderr {proposal}/{model}/{rate}",
                )
                paired_ci = _ci(
                    paired.get("stratified_paired_normal_ci95"),
                    f"zero-shot paired CI {proposal}/{model}/{rate}",
                    minimum=-1.0,
                    maximum=1.0,
                )
                expected_macro_ci = (
                    max(-1.0, delta - 1.959963984540054 * macro_stderr),
                    min(1.0, delta + 1.959963984540054 * macro_stderr),
                )
                _assert_close(
                    paired_ci[0],
                    expected_macro_ci[0],
                    f"zero-shot macro CI low {proposal}/{model}/{rate}",
                )
                _assert_close(
                    paired_ci[1],
                    expected_macro_ci[1],
                    f"zero-shot macro CI high {proposal}/{model}/{rate}",
                )
                _assert_close(
                    method_values[proposal] - method_values["TurboQuant-paper"],
                    delta,
                    f"zero-shot score delta {proposal}/{model}/{rate}",
                    tolerance=3e-7,
                )
                if not paired_ci[0] <= delta <= paired_ci[1]:
                    raise RenderError(
                        f"Zero-shot point estimate outside CI for {proposal}/{model}/{rate}"
                    )
                model_deltas[proposal][rate].append(delta)
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    for proposal in PROPOSAL_METHODS:
        proposal_stored = _mapping(stored_primary[proposal], f"zeroshot robustness {proposal}")
        _exact_keys(proposal_stored, set(RATES), f"zeroshot robustness {proposal}")
        primary[proposal] = {}
        for rate in RATES:
            mean, stderr, low, high = _mean_t_ci(
                model_deltas[proposal][rate],
                f"seven-model zero-shot deltas at {proposal}/{rate}",
            )
            nominal_ci = (max(-1.0, low), min(1.0, high))
            simultaneous_half = _bonferroni_three_rate_t_critical(len(PAPER_MODELS)) * stderr
            ci = (
                max(-1.0, mean - simultaneous_half),
                min(1.0, mean + simultaneous_half),
            )
            stored = _mapping(proposal_stored[rate], f"zeroshot robustness {proposal}/{rate}")
            if (
                stored.get("proposal") != proposal
                or stored.get("estimand") != "mean_fixed_checkpoint_five_task_macro_paired_delta"
                or stored.get("student_t_statistical_unit") != "fixed_checkpoint"
                or stored.get("checkpoint_panel")
                != "seven_preselected_family_correlated_checkpoints_not_iid"
                or stored.get("population_generalization") is not False
                or stored.get("n_checkpoints") != len(PAPER_MODELS)
                or stored.get("nominal_interval_status")
                != "secondary_robustness_not_primary_win_gate"
                or stored.get("multiplicity_family") != "three_predeclared_average_bit_rates_2_3_4"
                or stored.get("familywise_alpha") != 0.05
                or not math.isclose(
                    _number(
                        stored.get("per_rate_two_sided_alpha"),
                        f"zero-shot alpha {proposal}/{rate}",
                    ),
                    0.05 / 3.0,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or stored.get("claim_rule") != "interval_is_secondary_robustness_only"
            ):
                raise RenderError(f"Zero-shot robustness contract changed at {proposal}/{rate}")
            _assert_close(
                _number(stored.get("mean_delta"), f"zero-shot mean {proposal}/{rate}"),
                mean,
                f"zero-shot mean {proposal}/{rate}",
            )
            _assert_close(
                _number(
                    stored.get("mean_delta_standard_error"),
                    f"zero-shot stderr {proposal}/{rate}",
                ),
                stderr,
                f"zero-shot stderr {proposal}/{rate}",
            )
            stored_nominal = _ci(
                stored.get("nominal_student_t_ci95"),
                f"zero-shot nominal CI {proposal}/{rate}",
                minimum=-1.0,
                maximum=1.0,
            )
            stored_simultaneous = _ci(
                stored.get("bonferroni_simultaneous_familywise95_ci"),
                f"zero-shot simultaneous CI {proposal}/{rate}",
                minimum=-1.0,
                maximum=1.0,
            )
            for observed, expected, suffix in (
                (stored_nominal[0], nominal_ci[0], "nominal low"),
                (stored_nominal[1], nominal_ci[1], "nominal high"),
                (stored_simultaneous[0], ci[0], "simultaneous low"),
                (stored_simultaneous[1], ci[1], "simultaneous high"),
            ):
                _assert_close(observed, expected, f"zero-shot {proposal}/{rate} {suffix}")
            wins, ties, losses = _win_tie_loss_counts(
                model_deltas[proposal][rate], null=0.0, lower_is_better=False
            )
            primary[proposal][rate] = {
                "delta": mean,
                "stderr": stderr,
                "ci": ci,
                "nominal_ci": nominal_ci,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "point_statement": _zeroshot_point_statement(mean),
                "robustness_statement": _zeroshot_robustness_statement(ci),
            }
    return {
        "models": models,
        "primary": primary,
        "inference_scope": dict(inference_scope),
        "pinned_source_prompt_audit": dict(axis["pinned_source_prompt_audit"]),
    }


def _validate_mechanism(axis: Mapping[str, Any], commit: str) -> dict:
    _validate_commit(axis, commit, "mechanism")
    _validate_model_evaluation_roles(axis, LONG_MODELS, "mechanism")
    if (
        axis.get("benchmark") != "HayateQuant block-VQ mechanism ablation"
        or axis.get("official_complete") is not True
        or axis.get("primary_contrast")
        != "HayateQuant-blockvq-matched vs HayateQuant-scalar-matched"
        or axis.get("proposal_rows_in_scope") != list(PROPOSAL_METHODS)
        or axis.get("compression_accounting") != PPL_COMPRESSION_ACCOUNTING
        or axis.get("teacher_forced_scoring_contract")
        != {
            "protocol": PPL_TEACHER_FORCED_PROTOCOL,
            "temporal_dependency": None,
            "formal_methods_are_token_local": True,
        }
        or axis.get("inference_scope")
        != {
            "checkpoint_panel": "two_preselected_family_correlated_checkpoints_not_iid",
            "population_generalization": False,
            "reporting": "descriptive_geometric_mean_only",
            "uncertainty_interval": None,
            "directional_claim_allowed": False,
            "multiplicity_family": "three_predeclared_average_bit_rates_2_3_4",
        }
    ):
        raise RenderError("mechanism axis is not the exact formal causal control")
    models = _mapping(axis.get("models"), "mechanism.models")
    _exact_keys(models, set(LONG_MODELS), "mechanism.models")
    rate_ratios: dict[str, list[float]] = {rate: [] for rate in RATES}
    expected_methods = {
        "TurboQuant-paper",
        "HayateQuant-scalar-matched",
        "HayateQuant-blockvq-matched",
        "HQ",
        "AdaptiveHQ",
    }
    for model in LONG_MODELS:
        model_doc = _mapping(models[model], f"mechanism.{model}")
        if model_doc.get("official_complete") is not True:
            raise RenderError(f"Mechanism model is incomplete: {model}")
        rates = _mapping(model_doc.get("rates"), f"mechanism.{model}.rates")
        _exact_keys(rates, set(RATES), f"mechanism.{model}.rates")
        for rate in RATES:
            row = _mapping(rates[rate], f"mechanism.{model}.{rate}")
            methods = _mapping(row.get("methods"), f"mechanism methods {model}/{rate}")
            _exact_keys(methods, expected_methods, f"mechanism methods {model}/{rate}")
            sufficient = {}
            for method in expected_methods:
                method_doc = _mapping(methods[method], f"mechanism.{model}.{rate}.{method}")
                sufficient[method] = _validate_ppl_sufficient_statistics(
                    method_doc, f"mechanism.{model}.{rate}.{method}"
                )
                _validate_ppl_protocol_and_compression(
                    method_doc,
                    f"mechanism.{model}.{rate}.{method}",
                    compressed=True,
                    method=method,
                )
            if row.get("applicability") != {
                "single_lever_causal_block_vq_contrast": [
                    "HayateQuant-scalar-matched",
                    "HayateQuant-blockvq-matched",
                ],
                "HQ": "contextual formal proposal row; not part of the single-lever contrast",
                "AdaptiveHQ": (
                    "contextual formal proposal row; causal selector changes allocation "
                    "and is not part of the single-lever contrast"
                ),
            }:
                raise RenderError(f"Mechanism applicability changed for {model}/{rate}")
            scalar_nll = sufficient["HayateQuant-scalar-matched"][1]
            block_nll = sufficient["HayateQuant-blockvq-matched"][1]
            ratio = math.exp(block_nll - scalar_nll)
            _assert_close(
                _number(
                    row.get("causal_block_vq_ratio_block_over_scalar"),
                    f"mechanism ratio {model}/{rate}",
                ),
                ratio,
                f"mechanism ratio {model}/{rate}",
            )
            rate_ratios[rate].append(ratio)
    primary = {}
    for rate in RATES:
        log_ratios = [math.log(value) for value in rate_ratios[rate]]
        mean_log = math.fsum(log_ratios) / len(log_ratios)
        ratio = math.exp(mean_log)
        primary[rate] = {
            "ratio": ratio,
            "per_checkpoint_ratio": {
                model: value for model, value in zip(LONG_MODELS, rate_ratios[rate])
            },
            "uncertainty_interval": None,
            "claim": "descriptive only; no directional mechanism claim",
        }
    return {"models": models, "primary": primary}


def _validate_adaptive_hq_runtime(value: Any, label: str) -> Mapping[str, Any]:
    runtime = _mapping(value, f"{label}.adaptive_hq_runtime")
    if (
        runtime.get("schema_version") != 1
        or runtime.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
        or runtime.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
        or runtime.get("selector_incremental_stored_bits_total") != 0
    ):
        raise RenderError(f"{label} AdaptiveHQ policy/carrier provenance changed")
    updates = _integer(runtime.get("encoded_updates"), f"{label}.encoded_updates", minimum=1)
    tokens = _integer(runtime.get("encoded_tokens"), f"{label}.encoded_tokens", minimum=1)
    selector_bits = _integer(
        runtime.get("selector_bits_total"), f"{label}.selector_bits_total", minimum=1
    )
    if selector_bits != updates:
        raise RenderError(f"{label} AdaptiveHQ must carry one selector bit per update")
    histogram = _mapping(runtime.get("split_histogram"), f"{label}.split_histogram")
    if not histogram:
        raise RenderError(f"{label} AdaptiveHQ split histogram is empty")
    observed_pairs: set[tuple[int, int]] = set()
    histogram_updates = histogram_tokens = histogram_vectors = 0
    for key, raw in histogram.items():
        bucket = _mapping(raw, f"{label}.split_histogram.{key}")
        pair = _sequence(bucket.get("pair"), f"{label}.split_histogram.{key}.pair")
        if len(pair) != 2:
            raise RenderError(f"{label} AdaptiveHQ selector pair is malformed")
        for index, bit in enumerate(pair):
            _integer(bit, f"{label}.split_histogram.{key}.pair[{index}]", minimum=1)
        pair_tuple = (int(pair[0]), int(pair[1]))
        if key != f"{pair_tuple[0]}:{pair_tuple[1]}":
            raise RenderError(f"{label} AdaptiveHQ histogram key/pair disagree")
        observed_pairs.add(pair_tuple)
        bucket_updates = _integer(
            bucket.get("updates"),
            f"{label}.split_histogram.{key}.updates",
            minimum=1,
        )
        bucket_tokens = _integer(
            bucket.get("tokens"),
            f"{label}.split_histogram.{key}.tokens",
            minimum=1,
        )
        bucket_vectors = _integer(
            bucket.get("vectors"),
            f"{label}.split_histogram.{key}.vectors",
            minimum=1,
        )
        if not bucket_updates <= bucket_tokens <= bucket_vectors:
            raise RenderError(
                f"{label} AdaptiveHQ histogram must satisfy updates<=tokens<=vectors"
            )
        histogram_updates += bucket_updates
        histogram_tokens += bucket_tokens
        histogram_vectors += bucket_vectors
    if (
        histogram_updates != updates
        or histogram_tokens != tokens
        or histogram_vectors < tokens
        or not any(
            observed_pairs <= set(candidate_pairs)
            for candidate_pairs in ADAPTIVE_HQ_CANDIDATE_PAIRS.values()
        )
    ):
        raise RenderError(f"{label} AdaptiveHQ two-pair histogram does not replay")
    return runtime


def _aggregate_adaptive_hq_runtimes(
    values: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    histogram: dict[str, dict[str, Any]] = {}
    for index, runtime in enumerate(values):
        checked = _validate_adaptive_hq_runtime(runtime, f"{label}[{index}]")
        for key, raw in checked["split_histogram"].items():
            bucket = _mapping(raw, f"{label}[{index}].{key}")
            merged = histogram.setdefault(
                key,
                {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
            )
            if merged["pair"] != bucket["pair"]:
                raise RenderError(f"{label} AdaptiveHQ selector pair changed across samples")
            for field in ("updates", "tokens", "vectors"):
                merged[field] += int(bucket[field])
    return {
        "schema_version": 1,
        "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
        "encoded_updates": sum(int(row["encoded_updates"]) for row in values),
        "encoded_tokens": sum(int(row["encoded_tokens"]) for row in values),
        "selector_bits_total": sum(int(row["selector_bits_total"]) for row in values),
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
        "split_histogram": histogram,
    }


def _validate_long_compression_row(
    row: Mapping[str, Any], *, method: str, expected_observations: int, label: str
) -> dict[str, Mapping[str, Any]]:
    ledger = _sequence(row.get("compression_ledger"), f"{label}.compression_ledger")
    if len(ledger) != expected_observations:
        raise RenderError(f"{label} compression ledger coverage changed")
    if row.get("compression_observations") != expected_observations:
        raise RenderError(f"{label} compression observation count changed")
    by_id: dict[str, Mapping[str, Any]] = {}
    fp16_total = stored_total = 0
    adaptive_rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(ledger):
        item = _mapping(raw, f"{label}.compression_ledger[{index}]")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
            raise RenderError(f"{label} compression sample identity is malformed/duplicated")
        by_id[sample_id] = item
        _integer(item.get("input_tokens"), f"{label}.{sample_id}.input_tokens", minimum=1)
        _integer(item.get("max_new_tokens"), f"{label}.{sample_id}.max_new_tokens", minimum=1)
        _integer(item.get("generated_tokens"), f"{label}.{sample_id}.generated_tokens")
        if method == "FullKV":
            if (
                item.get("compression_aggregation") != "fullkv_no_cache"
                or item.get("compression_fp16_bits") is not None
                or item.get("compression_stored_bits") is not None
                or item.get("adaptive_hq_runtime") is not None
            ):
                raise RenderError(f"{label}/{sample_id} FullKV accounting changed")
        else:
            if item.get("compression_aggregation") != "exact_cache_bit_counters":
                raise RenderError(f"{label}/{sample_id} lacks exact cache-bit counters")
            fp16_total += _integer(
                item.get("compression_fp16_bits"),
                f"{label}.{sample_id}.compression_fp16_bits",
                minimum=1,
            )
            stored_total += _integer(
                item.get("compression_stored_bits"),
                f"{label}.{sample_id}.compression_stored_bits",
                minimum=1,
            )
            if method == "AdaptiveHQ":
                adaptive_rows.append(
                    _validate_adaptive_hq_runtime(
                        item.get("adaptive_hq_runtime"), f"{label}.{sample_id}"
                    )
                )
            elif item.get("adaptive_hq_runtime") is not None:
                raise RenderError(f"{label}/{sample_id} non-adaptive row has selector provenance")
    if method == "FullKV":
        if (
            row.get("compression_aggregation") != "fullkv_no_cache"
            or row.get("compression_fp16_bits") is not None
            or row.get("compression_stored_bits") is not None
            or row.get("adaptive_hq_runtime") is not None
        ):
            raise RenderError(f"{label} FullKV aggregate accounting changed")
    else:
        if (
            row.get("compression_aggregation") != "sum_exact_per_generation_cache_bit_counters"
            or row.get("compression_fp16_bits") != fp16_total
            or row.get("compression_stored_bits") != stored_total
        ):
            raise RenderError(f"{label} exact aggregate bit counters do not replay")
        if method == "AdaptiveHQ":
            expected_runtime = _aggregate_adaptive_hq_runtimes(adaptive_rows, label)
            if row.get("adaptive_hq_runtime") != expected_runtime:
                raise RenderError(f"{label} AdaptiveHQ aggregate provenance does not replay")
        elif row.get("adaptive_hq_runtime") is not None:
            raise RenderError(f"{label} non-adaptive aggregate has selector provenance")
    return by_id


def _validate_long_exact_rate_match(
    methods: Mapping[str, Any], *, expected_observations: int, label: str
) -> None:
    ledgers = {
        method: _validate_long_compression_row(
            _mapping(methods[method], f"{label}.{method}"),
            method=method,
            expected_observations=expected_observations,
            label=f"{label}.{method}",
        )
        for method in METHODS
    }
    reference = ledgers["TurboQuant-paper"]
    tq_row = _mapping(methods["TurboQuant-paper"], f"{label}.TurboQuant-paper")
    if tq_row.get("matched_rate_reference") != {
        "method": "TurboQuant-paper",
        "n_samples": expected_observations,
    }:
        raise RenderError(f"{label} TurboQuant exact-rate reference changed")
    for method in ("FibQuant-paper", *PROPOSAL_METHODS):
        candidate = ledgers[method]
        if set(candidate) != set(reference):
            raise RenderError(f"{label}/{method} compression sample universe changed")
        differences = 0
        for sample_id, tq in reference.items():
            row = candidate[sample_id]
            if any(
                row.get(field) != tq.get(field) for field in ("input_tokens", "max_new_tokens")
            ):
                raise RenderError(f"{label}/{method}/{sample_id} input/budget changed")
            tq_fp16 = int(tq["compression_fp16_bits"])
            tq_stored = int(tq["compression_stored_bits"])
            candidate_fp16 = int(row["compression_fp16_bits"])
            candidate_stored = int(row["compression_stored_bits"])
            if candidate_stored * tq_fp16 != tq_stored * candidate_fp16:
                raise RenderError(f"{label}/{method}/{sample_id} exact stored rate changed")
            differences += int(row.get("generated_tokens") != tq.get("generated_tokens"))
        evidence = _mapping(methods[method], f"{label}.{method}").get("matched_rate_vs_turboquant")
        if evidence != {
            "exact_integer_cross_products": True,
            "matched_sample_universe": True,
            "input_tokens_and_max_budget_identical": True,
            "generated_lengths_forced_equal": False,
            "generated_length_difference_count": differences,
            "n_samples": expected_observations,
        }:
            raise RenderError(f"{label}/{method} exact-rate evidence changed")


def _validate_ruler(axis: Mapping[str, Any], commit: str) -> Mapping[str, Any]:
    _validate_commit(axis, commit, "ruler")
    _validate_model_evaluation_roles(axis, LONG_MODELS, "ruler")
    if (
        isinstance(axis.get("schema_version"), bool)
        or axis.get("schema_version") != 1
        or axis.get("benchmark") != RULER_PROTOCOL_LABEL
        or axis.get("aggregation") != RULER_AGGREGATION
        or axis.get("source_result_schema_version") != RULER_SOURCE_RESULT_SCHEMA_VERSION
        or axis.get("context_semantics") != RULER_CONTEXT_SEMANTICS
        or axis.get("capacity_check") != RULER_CAPACITY_CHECK
        or axis.get("protocol_scope") != RULER_PROTOCOL_SCOPE
        or axis.get("reference_model_wrapper_exact") is not False
        or axis.get("protocol_complete") is not True
        or axis.get("samples_per_task") != RULER_SAMPLES_PER_TASK
        or axis.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL
        or axis.get("canonical_nvidia_samples_per_task") != RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK
        or axis.get("canonical_nvidia_full_eval") is not False
    ):
        raise RenderError("ruler axis is not the complete pinned NeMo-Skills protocol")
    axis_audit_records = _integer(
        axis.get("audit_records_validated"), "RULER axis audit_records_validated", minimum=1
    )
    if axis_audit_records != RULER_AXIS_AUDIT_RECORDS:
        raise RenderError(
            "RULER axis audit coverage must be exactly " f"{RULER_AXIS_AUDIT_RECORDS} records"
        )
    generation_protocols = _mapping(
        axis.get("generation_stop_protocols"),
        "ruler.generation_stop_protocols",
    )
    _exact_keys(generation_protocols, set(LONG_MODELS), "ruler.generation_stop_protocols")
    normalized_generation_protocols = {}
    for model in LONG_MODELS:
        protocol = _mapping(generation_protocols[model], f"RULER generation protocol {model}")
        eos_ids = protocol.get("eos_token_ids")
        if (
            protocol.get("eos_token_id_source") != "model.generation_config.eos_token_id"
            or not isinstance(eos_ids, list)
            or not eos_ids
            or len(set(eos_ids)) != len(eos_ids)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in eos_ids
            )
            or protocol.get("stop_conditions")
            != ["any_generation_config_eos_token_id", "max_new_tokens"]
        ):
            raise RenderError(f"RULER EOS/stop provenance changed for {model}")
        normalized_generation_protocols[model] = dict(protocol)
    wrapper_protocols = _mapping(
        axis.get("model_wrapper_protocols"),
        "ruler.model_wrapper_protocols",
    )
    _exact_keys(wrapper_protocols, set(LONG_MODELS), "ruler.model_wrapper_protocols")
    normalized_wrapper_protocols = {}
    for model in LONG_MODELS:
        wrapper = _mapping(wrapper_protocols[model], f"RULER wrapper protocol {model}")
        expected_wrapper = {
            "prompt_wrappers": [RULER_CHAT_WRAPPER],
            "chat_template_kwargs": RULER_CHAT_TEMPLATE_KWARGS,
            "protocol_scope": RULER_PROTOCOL_SCOPE,
            "reference_model_wrapper_exact": False,
        }
        if wrapper != expected_wrapper:
            raise RenderError(f"RULER paired non-thinking wrapper provenance changed for {model}")
        normalized_wrapper_protocols[model] = dict(wrapper)
    models = _mapping(axis.get("models"), "ruler.models")
    _exact_keys(models, set(LONG_MODELS), "ruler.models")
    for model in LONG_MODELS:
        manifest_digests = set()
        contexts = _mapping(models[model], f"ruler.{model}")
        _exact_keys(contexts, set(RULER_CONTEXTS), f"ruler.{model}")
        for context in RULER_CONTEXTS:
            dataset_digests = set()
            methods = _mapping(contexts[context], f"ruler.{model}.{context}")
            _exact_keys(methods, set(METHODS), f"ruler.{model}.{context}")
            for method in METHODS:
                row = _mapping(methods[method], f"ruler.{model}.{context}.{method}")
                _number(
                    row.get("score"),
                    f"RULER score {model}/{context}/{method}",
                    minimum=0.0,
                    maximum=100.0,
                )
                row_n = _integer(row.get("n"), f"RULER n {model}/{context}/{method}", minimum=1)
                row_audit_records = _integer(
                    row.get("audit_records_validated"),
                    f"RULER audit_records_validated {model}/{context}/{method}",
                    minimum=1,
                )
                if row_n != RULER_ROWS_PER_RUN or row_audit_records != RULER_ROWS_PER_RUN:
                    raise RenderError(
                        f"RULER row audit coverage changed for {model}/{context}/{method}"
                    )
                if row.get("generation_stop_protocol") != normalized_generation_protocols[model]:
                    raise RenderError(
                        f"RULER EOS/stop protocol changed for {model}/{context}/{method}"
                    )
                if row.get("model_wrapper_protocol") != normalized_wrapper_protocols[model]:
                    raise RenderError(
                        f"RULER paired wrapper protocol changed for {model}/{context}/{method}"
                    )
                capacity = _integer(
                    row.get("model_max_position_embeddings"),
                    f"RULER model capacity {model}/{context}/{method}",
                    minimum=1,
                )
                if capacity != RULER_MODEL_MAX_POSITION_EMBEDDINGS[model]:
                    raise RenderError(
                        f"RULER model capacity changed for {model}/{context}/{method}"
                    )
                named_limit = _integer(
                    int(context),
                    f"RULER named sequence limit {model}/{context}/{method}",
                    minimum=1,
                )
                max_reserved = _integer(
                    row.get("max_observed_prompt_plus_max_generation_tokens"),
                    f"RULER max reserved sequence {model}/{context}/{method}",
                    minimum=1,
                )
                if max_reserved > named_limit or named_limit > capacity:
                    raise RenderError(
                        f"RULER named total-sequence budget violated for "
                        f"{model}/{context}/{method}"
                    )
                manifest_digest = row.get("generation_manifest_sha256")
                if not isinstance(manifest_digest, str) or not SHA64.fullmatch(
                    manifest_digest.lower()
                ):
                    raise RenderError(
                        f"RULER generation manifest hash missing for {model}/{context}/{method}"
                    )
                manifest_digests.add(manifest_digest.lower())
                digest = row.get("combined_dataset_sha256")
                if not isinstance(digest, str) or not SHA64.fullmatch(digest.lower()):
                    raise RenderError(f"RULER dataset hash missing for {model}/{context}/{method}")
                dataset_digests.add(digest.lower())
            _validate_long_exact_rate_match(
                methods,
                expected_observations=RULER_ROWS_PER_RUN,
                label=f"ruler.{model}.{context}",
            )
            if len(dataset_digests) != 1:
                raise RenderError(f"RULER dataset changed across methods for {model}/{context}")
        if len(manifest_digests) != 1:
            raise RenderError(f"RULER generation manifest changed across {model} rows")
    return models


def _validate_longbench(axis: Mapping[str, Any], commit: str) -> Mapping[str, Any]:
    _validate_commit(axis, commit, "longbench")
    _validate_model_evaluation_roles(axis, LONG_MODELS, "longbench")
    if (
        axis.get("benchmark") != "LongBench-E"
        or axis.get("released_data_complete") is not True
        or axis.get("official_primary_reporting") != "per_task_three_length_bucket_scores"
        or axis.get("reported_scalar") != "derived_unweighted_macro_over_task_scores"
        or axis.get("protocol_scope") != LONGBENCH_PROTOCOL_SCOPE
        or axis.get("reference_model_wrapper_exact") is not False
        or axis.get("prompt_budgets") != LONGBENCH_PROMPT_BUDGETS
        or axis.get("expected_truncations") != LONGBENCH_EXPECTED_TRUNCATIONS
        or axis.get("truncation_policy")
        != (
            "official middle truncation at an explicit model-specific raw-prompt "
            "budget; post-wrapper prompt plus task generation is capacity checked"
        )
    ):
        raise RenderError("longbench axis is not complete LongBench-E")
    generation_protocols = _mapping(
        axis.get("generation_stop_protocols"),
        "longbench.generation_stop_protocols",
    )
    _exact_keys(
        generation_protocols,
        set(LONG_MODELS),
        "longbench.generation_stop_protocols",
    )
    normalized_generation_protocols = {}
    for model in LONG_MODELS:
        protocol = _mapping(generation_protocols[model], f"LongBench generation protocol {model}")
        eos_ids = protocol.get("eos_token_ids")
        newline_id = protocol.get("samsum_newline_token_id")
        if (
            protocol.get("eos_token_id_source") != "model.generation_config.eos_token_id"
            or not isinstance(eos_ids, list)
            or not eos_ids
            or len(set(eos_ids)) != len(eos_ids)
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
                for token_id in eos_ids
            )
            or isinstance(newline_id, bool)
            or not isinstance(newline_id, int)
            or newline_id < 0
            or protocol.get("stop_conditions")
            != [
                "any_generation_config_eos_token_id",
                "max_new_tokens",
                "samsum_newline_token_id_for_samsum_only",
            ]
        ):
            raise RenderError(f"LongBench EOS/stop provenance changed for {model}")
        normalized_generation_protocols[model] = dict(protocol)
    models = _mapping(axis.get("models"), "longbench.models")
    _exact_keys(models, set(LONG_MODELS), "longbench.models")
    for model in LONG_MODELS:
        methods = _mapping(models[model], f"longbench.{model}")
        _exact_keys(methods, set(METHODS), f"longbench.{model}")
        for method in METHODS:
            row = _mapping(methods[method], f"longbench.{model}.{method}")
            if (
                row.get("released_data_complete") is not True
                or row.get("n") != LONGBENCH_TOTAL
                or row.get("data_zip_sha256") != LONGBENCH_ARCHIVE_SHA256
            ):
                raise RenderError(f"LongBench-E row incomplete for {model}/{method}")
            row_score = _number(
                row.get("score_unrounded"),
                f"LongBench score {model}/{method}",
                minimum=0.0,
                maximum=100.0,
            )
            tasks = _mapping(row.get("per_task"), f"LongBench tasks {model}/{method}")
            _exact_keys(tasks, set(LONGBENCH_TASKS), f"LongBench tasks {model}/{method}")
            task_scores = []
            for task in LONGBENCH_TASKS:
                task_row = _mapping(tasks[task], f"LongBench {model}/{method}/{task}")
                expected_n = LONGBENCH_COUNTS[task]
                if (
                    task_row.get("n") != expected_n
                    or task_row.get("audit_records_validated") != expected_n
                    or task_row.get("data_zip_sha256") != LONGBENCH_ARCHIVE_SHA256
                    or task_row.get("prompt_budget") != LONGBENCH_PROMPT_BUDGETS[model][task]
                    or task_row.get("truncated_examples")
                    != LONGBENCH_EXPECTED_TRUNCATIONS[model][task]
                    or task_row.get("generation_stop_protocol")
                    != normalized_generation_protocols[model]
                ):
                    raise RenderError(
                        f"LongBench-E task denominator changed: {model}/{method}/{task}"
                    )
                prompt_signature = task_row.get("prompt_audit_signature")
                if not isinstance(prompt_signature, str) or not SHA64.fullmatch(
                    prompt_signature.lower()
                ):
                    raise RenderError(
                        f"LongBench-E prompt audit is missing: {model}/{method}/{task}"
                    )
                score_sum = _number(
                    task_row.get("score_sum_unrounded"),
                    f"LongBench task numerator {model}/{method}/{task}",
                    minimum=0.0,
                    maximum=100.0 * expected_n,
                )
                task_score = _number(
                    task_row.get("score_unrounded"),
                    f"LongBench task score {model}/{method}/{task}",
                    minimum=0.0,
                    maximum=100.0,
                )
                _assert_close(
                    task_score,
                    score_sum / expected_n,
                    f"LongBench task numerator/denominator {model}/{method}/{task}",
                )
                _assert_close(
                    _number(
                        task_row.get("score_displayed"),
                        f"LongBench displayed task score {model}/{method}/{task}",
                        minimum=0.0,
                        maximum=100.0,
                    ),
                    round(task_score, 2),
                    f"LongBench displayed task score {model}/{method}/{task}",
                )
                task_scores.append(task_score)
            macro = math.fsum(task_scores) / len(task_scores)
            _assert_close(
                row_score,
                macro,
                f"LongBench macro {model}/{method}",
            )
            _assert_close(
                _number(
                    row.get("score"),
                    f"LongBench displayed macro {model}/{method}",
                    minimum=0.0,
                    maximum=100.0,
                ),
                round(macro, 2),
                f"LongBench displayed macro {model}/{method}",
            )
        _validate_long_exact_rate_match(
            methods,
            expected_observations=LONGBENCH_TOTAL,
            label=f"longbench.{model}",
        )
    return models


def _validate_math500(axis: Mapping[str, Any], commit: str) -> Mapping[str, Any]:
    _validate_commit(axis, commit, "math500")
    _validate_model_evaluation_roles(axis, LONG_MODELS, "math500")
    if (
        axis.get("benchmark") != "MATH-500"
        or axis.get("grader") != "Math-Verify 0.9.0"
        or axis.get("protocol_scope") != MATH500_PROTOCOL_SCOPE
        or axis.get("grading_input") != "generated_text"
        or axis.get("official_reference_gold_serialization_exact") is not False
        or axis.get("gold_serialization") != MATH500_GOLD_SERIALIZATION
        or axis.get("official_complete") is not True
    ):
        raise RenderError("math500 axis is not complete pinned Math-Verify MATH-500")
    if axis.get("few_shot_protocol") != {
        "protocol_id": MATH500_PROTOCOL_ID,
        "protocol_origin": "adapted_composite",
        "math_verify_task_source_commit": MATH_VERIFY_TASK_SOURCE_COMMIT,
        "lighteval_prompt_manager_reference_version": (
            MATH500_LIGHTEVAL_PROMPT_MANAGER_REFERENCE_VERSION
        ),
        "lighteval_prompt_manager_source_commit": (MATH500_LIGHTEVAL_PROMPT_MANAGER_SOURCE_COMMIT),
        "upstream_lighteval_builtin_task_exact": False,
        "upstream_math_verify_custom_task_exact": False,
        "upstream_prompt_generation_components_exact": True,
        "upstream_component_pair_executable": False,
        "balanced_sampler_seed": 0,
        "few_shot_count": 4,
        "few_shot_order_one_based": [4, 3, 2, 1],
        "few_shot_content_sha256": MATH500_FEWSHOT_CONTENT_SHA256,
        "few_shot_rendering": MATH500_FEWSHOT_RENDERING,
        "use_chat_template": False,
    }:
        raise RenderError("MATH-500 four-shot prompt provenance changed")
    models = _mapping(axis.get("models"), "math500.models")
    _exact_keys(models, set(LONG_MODELS), "math500.models")
    for model in LONG_MODELS:
        methods = _mapping(models[model], f"math500.{model}")
        _exact_keys(methods, set(METHODS), f"math500.{model}")
        for method in METHODS:
            row = _mapping(methods[method], f"math500.{model}.{method}")
            if row.get("n") != 500 or row.get("coverage") != [0, 500]:
                raise RenderError(f"MATH-500 row incomplete for {model}/{method}")
            if row.get("audit_records_validated") != 500:
                raise RenderError(f"MATH-500 audit coverage is incomplete for {model}/{method}")
            if (
                row.get("eviction_applicable") is not False
                or row.get("eviction_fired") is not None
            ):
                raise RenderError(f"MATH-500 non-eviction provenance changed for {model}/{method}")
            correct = _integer(row.get("correct"), f"MATH correct {model}/{method}")
            if correct > 500 or row.get("pass@1_sum") != correct:
                raise RenderError(f"MATH-500 numerator is stale for {model}/{method}")
            estimate = _number(
                row.get("pass@1"), f"MATH pass@1 {model}/{method}", minimum=0.0, maximum=1.0
            )
            _assert_close(estimate, correct / 500, f"MATH pass@1 {model}/{method}")
            ci = _ci(row.get("pass@1_ci95"), f"MATH CI {model}/{method}", minimum=0.0, maximum=1.0)
            if not ci[0] <= estimate <= ci[1]:
                raise RenderError(f"MATH estimate outside CI for {model}/{method}")
            expected_ci = _normal_binomial_ci(estimate, 500)
            _assert_close(ci[0], expected_ci[0], f"MATH CI low {model}/{method}")
            _assert_close(ci[1], expected_ci[1], f"MATH CI high {model}/{method}")
        _validate_long_exact_rate_match(
            methods,
            expected_observations=500,
            label=f"math500.{model}",
        )
    return models


def _validate_longcodeqa(axis: Mapping[str, Any], commit: str) -> Mapping[str, Any]:
    _validate_commit(axis, commit, "longcodeqa")
    _validate_model_evaluation_roles(axis, LONG_MODELS, "longcodeqa")
    if (
        axis.get("schema_version") != 1
        or axis.get("benchmark") != "LongCodeBench/LongCodeQA"
        or axis.get("aggregation") != "sum_exact_counts_after_exact_interval_coverage"
        or axis.get("primary_reporting") != "per_bracket_exact_match_accuracy"
        or axis.get("cross_bracket_summary") != "derived_micro_exact_match_accuracy"
        or axis.get("released_data_complete_feasible_plan") is not True
        or axis.get("scope") != "all released brackets that fit each unmodified model context"
        or axis.get("protocol_scope") != LONGCODE_PROTOCOL_SCOPE
        or axis.get("official_reference_decoding_exact") is not False
    ):
        raise RenderError("longcodeqa axis is not the complete model-feasible plan")
    if axis.get("source_row_audit") != LONGCODE_SOURCE_ROW_AUDIT:
        raise RenderError("longcodeqa source-row/context-fit audit contract is incomplete")
    models = _mapping(axis.get("models"), "longcodeqa.models")
    _exact_keys(models, set(LONGCODE_PLANS), "longcodeqa.models")
    for model, plan in LONGCODE_PLANS.items():
        model_doc = _mapping(models[model], f"longcodeqa.{model}")
        if model_doc.get("released_data_complete_feasible_plan") is not True:
            raise RenderError(f"LongCodeQA model plan incomplete: {model}")
        methods = _mapping(model_doc.get("methods"), f"longcodeqa.{model}.methods")
        _exact_keys(methods, set(METHODS), f"longcodeqa.{model}.methods")
        for method in METHODS:
            method_doc = _mapping(methods[method], f"longcodeqa.{model}.{method}")
            if (
                method_doc.get("released_data_complete_feasible_plan") is not True
                or method_doc.get("primary_reporting") != "per_bracket_exact_match_accuracy"
                or method_doc.get("derived_summary_metric") != "micro_exact_match_accuracy"
            ):
                raise RenderError(f"LongCodeQA method incomplete: {model}/{method}")
            brackets = _mapping(
                method_doc.get("brackets"), f"longcodeqa.{model}.{method}.brackets"
            )
            _exact_keys(brackets, set(plan), f"longcodeqa.{model}.{method}.brackets")
            if method_doc.get("n") != sum(plan.values()):
                raise RenderError(f"LongCodeQA model denominator changed: {model}/{method}")
            model_correct = 0
            for bracket, expected_n in plan.items():
                row = _mapping(brackets[bracket], f"longcodeqa.{model}.{method}.{bracket}")
                if (
                    row.get("n") != expected_n
                    or row.get("coverage") != [0, expected_n]
                    or row.get("released_data_complete_bracket") is not True
                    or row.get("audit_records_validated") != expected_n
                    or not isinstance(row.get("context_fit_manifest_sha256"), str)
                    or not SHA64.fullmatch(row["context_fit_manifest_sha256"].lower())
                ):
                    raise RenderError(f"LongCodeQA bracket incomplete: {model}/{method}/{bracket}")
                correct = _integer(
                    row.get("correct"), f"LongCodeQA correct {model}/{method}/{bracket}"
                )
                if correct > expected_n:
                    raise RenderError(
                        f"LongCodeQA correct count exceeds n: {model}/{method}/{bracket}"
                    )
                model_correct += correct
                estimate = _number(
                    row.get("accuracy"),
                    f"LongCodeQA accuracy {model}/{method}/{bracket}",
                    minimum=0.0,
                    maximum=1.0,
                )
                _assert_close(
                    estimate,
                    correct / expected_n,
                    f"LongCodeQA accuracy {model}/{method}/{bracket}",
                )
                ci = _ci(
                    row.get("accuracy_ci95"),
                    f"LongCodeQA CI {model}/{method}/{bracket}",
                    minimum=0.0,
                    maximum=1.0,
                )
                if not ci[0] <= estimate <= ci[1]:
                    raise RenderError(
                        f"LongCodeQA estimate outside CI: {model}/{method}/{bracket}"
                    )
                expected_ci = _wilson_ci(correct, expected_n)
                _assert_close(
                    ci[0], expected_ci[0], f"LongCodeQA CI low {model}/{method}/{bracket}"
                )
                _assert_close(
                    ci[1], expected_ci[1], f"LongCodeQA CI high {model}/{method}/{bracket}"
                )
            if method_doc.get("correct") != model_correct:
                raise RenderError(f"LongCodeQA model numerator is stale: {model}/{method}")
            model_accuracy = _number(
                method_doc.get("accuracy"), f"LongCodeQA model accuracy {model}/{method}"
            )
            _assert_close(
                model_accuracy,
                model_correct / sum(plan.values()),
                f"LongCodeQA model accuracy {model}/{method}",
            )
            model_ci = _ci(
                method_doc.get("accuracy_ci95"),
                f"LongCodeQA model CI {model}/{method}",
                minimum=0.0,
                maximum=1.0,
            )
            expected_model_ci = _wilson_ci(model_correct, sum(plan.values()))
            _assert_close(
                model_ci[0], expected_model_ci[0], f"LongCodeQA model CI low {model}/{method}"
            )
            _assert_close(
                model_ci[1], expected_model_ci[1], f"LongCodeQA model CI high {model}/{method}"
            )
        for bracket, expected_n in plan.items():
            _validate_long_exact_rate_match(
                {
                    method: _mapping(methods[method], f"longcodeqa.{model}.{method}")["brackets"][
                        bracket
                    ]
                    for method in METHODS
                },
                expected_observations=expected_n,
                label=f"longcodeqa.{model}.{bracket}",
            )
        _validate_long_exact_rate_match(
            methods,
            expected_observations=sum(plan.values()),
            label=f"longcodeqa.{model}",
        )
    return models


def _speed_metric(row: Mapping[str, Any], path: str, unit: str) -> Mapping[str, Any]:
    metrics = _mapping(row.get("metrics"), "speed metrics")
    metric = _mapping(metrics.get(path), f"speed metric {path}")
    if metric.get("unit") != unit:
        raise RenderError(f"Speed metric {path} has unit {metric.get('unit')!r}, expected {unit}")
    values = [
        _number(value, f"speed metric {path} replicate", minimum=0.0)
        for value in _sequence(
            metric.get("replicate_medians"), f"speed metric {path} replicate_medians"
        )
    ]
    if len(values) != 5:
        raise RenderError(f"Speed metric {path} must have five process replicates")
    median = _number(
        metric.get("median_of_replicate_medians"), f"speed metric {path} median", minimum=0.0
    )
    _assert_close(median, statistics.median(values), f"speed metric {path} median")
    _assert_close(
        _number(metric.get("min"), f"speed metric {path} min", minimum=0.0),
        min(values),
        f"speed metric {path} min",
    )
    _assert_close(
        _number(metric.get("max"), f"speed metric {path} max", minimum=0.0),
        max(values),
        f"speed metric {path} max",
    )
    return metric


def _expected_speed_resolved_recipe(rate: int, method: str) -> dict[str, Any]:
    """Independent paper-boundary expectation for the d=128 speed panel."""

    def _role(
        bits: int,
        block: int,
        rotation: str,
        scale: str,
        mathematical_scale: str,
        source_law: str,
    ) -> dict[str, Any]:
        return {
            "bits": bits,
            "block_size": block,
            "rotation_mode": rotation,
            "rotation_seed": 0,
            "scale_mode": scale,
            "mathematical_scale_mode": mathematical_scale,
            "source_law": source_law,
            "quantized_dim": 128,
            "outlier_channels": 0,
            "outlier_bits": None,
            "outlier_selection": "none",
            "qjl_enabled": False,
        }

    if method == "tq":
        role = _role(rate, 1, "dense", "l2", "l2", "beta")
        return {
            "schema_version": 1,
            "head_dim": 128,
            "codec": "turboquant",
            "calibration": None,
            "roles": {"key": dict(role), "value": dict(role)},
        }
    if method not in {"hq", "adaptive"}:
        raise AssertionError(f"unknown speed method role: {method}")

    def rotation(code_bits: int) -> str:
        return "dense" if code_bits <= 4 else "hybrid"

    assignment = {
        "encoder": "exact",
        "lut_bins": None,
        "lut_candidates": None,
        "shape_independent": True,
        "tie_break": "smallest_global_codebook_index",
    }
    common = {
        "schema_version": 1,
        "head_dim": 128,
        "codec": "hayatequant",
        "calibration": None,
        "assignment": assignment,
    }
    if method == "hq":

        def fixed_role(code_bits: int, scale: str) -> dict[str, Any]:
            row = _role(rate, 2, rotation(code_bits), scale, scale, "gaussian")
            row["code_bits_per_block"] = code_bits
            row["payload_bits_per_coordinate"] = code_bits / 2.0
            return row

        return {
            **common,
            "codebook_artifacts": _expected_hayate_codebook_artifacts(HQ_CODEBOOK_BUNDLE),
            "roles": {
                "key": fixed_role(2 * rate + 1, "nc"),
                "value": fixed_role(2 * rate - 1, "ls_iter"),
            },
        }

    total = 4 * rate
    candidate_pairs = [list(pair) for pair in ADAPTIVE_HQ_CANDIDATE_PAIRS[rate]]
    selector_bits = 1

    def adaptive_role(scale: str) -> dict[str, Any]:
        row = _role(
            rate,
            2,
            "payload_dependent_candidate_rate",
            scale,
            scale,
            "gaussian",
        )
        row.update({"adaptive_bit_split": "online_stats", "pair_code_bits_per_block": total})
        return row

    return {
        **common,
        "codebook_artifacts": _expected_hayate_codebook_artifacts(ADAPTIVE_HQ_CODEBOOK_BUNDLE),
        "adaptive_allocation": {
            "block_size": 2,
            "candidate_pairs": candidate_pairs,
            "candidate_rotations": [
                {
                    "pair": pair,
                    "key": rotation(pair[0]),
                    "value": rotation(pair[1]),
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
            "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
            "selector_carrier_distortion_amortization": (ADAPTIVE_HQ_CARRIER_AMORTIZATION),
            "selector_incremental_stored_bits_per_encode": 0,
            "selector_scale_reconstruction": "clear_carrier_fraction_bits",
            "state_ownership": "compressed_dynamic_cache_instance_per_layer",
        },
        "roles": {
            "key": adaptive_role("nc"),
            "value": adaptive_role("ls_iter"),
        },
    }


def _validate_speed_scientific_recipe(
    value: Any, *, rate: int, names: Mapping[str, str], label: str
) -> str:
    """Validate and fingerprint the exact declared and role-resolved recipe."""

    recipe = _mapping(value, f"{label}.scientific_recipe")
    _exact_keys(recipe, {"recipe_sha256", "method_configs"}, f"{label}.recipe")
    hashes = _mapping(recipe["recipe_sha256"], f"{label}.recipe_sha256")
    configs = _mapping(recipe["method_configs"], f"{label}.method_configs")
    method_names = {names["tq"], names["hq"], names["adaptive"]}
    _exact_keys(hashes, method_names, f"{label}.recipe_sha256")
    _exact_keys(configs, method_names, f"{label}.method_configs")

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
        "avg_bits": float(rate),
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
    expected_declared = {
        "tq": {
            **base_codec,
            "name": "turboquant",
            "bits": rate,
            "dist": "beta",
            "n_outlier": 0,
            "outlier_bits": None,
            "use_qjl": False,
            "qjl_roles": "key",
            "rotation_mode": "dense",
            "seed": 0,
            "value_bits": rate,
        },
        "hq": {
            **hayate_defaults,
            "required_codebook_bundle": HQ_CODEBOOK_BUNDLE,
            "bits": rate,
            "value_bits": rate,
            "dist": "gaussian",
            "rotation_mode": "auto",
            "block_size": 2,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_code_bits_per_block": 2 * rate + 1,
            "value_code_bits_per_block": 2 * rate - 1,
            "n_outlier": 0,
        },
        "adaptive": {
            **hayate_defaults,
            "required_codebook_bundle": ADAPTIVE_HQ_CODEBOOK_BUNDLE,
            "bits": rate,
            "value_bits": rate,
            "dist": "gaussian",
            "rotation_mode": "auto",
            "block_size": 2,
            "key_scale_mode": "nc",
            "value_scale_mode": "ls_iter",
            "key_bit_fraction": (2 * rate + 1) / (4 * rate),
            "adaptive_bit_split": "online_stats",
            "n_outlier": 0,
        },
    }
    expected_effective = {
        "tq": {},
        "hq": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
        "adaptive": {"key_scale_mode": "nc", "value_scale_mode": "ls_iter"},
    }
    expected_taxonomy = {
        "tq": {
            "semantic_role": "existing TurboQuant class; symmetric scalar Beta/L2 reference",
            "fake_quant_implementation": TURBOQUANT_FAKE_QUANT_IMPLEMENTATION,
            "proposal_method": False,
            "proposal_variant": None,
        },
        "hq": {
            "semantic_role": "canonical globally fixed HQ paper proposal",
            "fake_quant_implementation": HQ_FAKE_QUANT_IMPLEMENTATION,
            "proposal_method": True,
            "proposal_variant": "fixed_global",
        },
        "adaptive": {
            "semantic_role": "causal online-rate-allocation HQ paper proposal",
            "fake_quant_implementation": ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION,
            "proposal_method": True,
            "proposal_variant": "causal_adaptive",
        },
    }
    for method in ("tq", "hq", "adaptive"):
        method_name = names[method]
        record = _mapping(configs[method_name], f"{label}.{method_name}")
        codec_record = _mapping(record.get("codec"), f"{label}.{method_name}.codec")
        declared = _mapping(codec_record.get("declared"), f"{label}.{method_name}.declared")
        effective = _mapping(codec_record.get("effective"), f"{label}.{method_name}.effective")
        if (
            any(record.get(key) != expected for key, expected in expected_taxonomy[method].items())
            or declared != expected_declared[method]
            or effective != expected_effective[method]
            or record.get("resolved_at_model_head_dim")
            != _expected_speed_resolved_recipe(rate, method)
        ):
            raise RenderError(
                f"Role-resolved scientific speed recipe changed for " f"{label}/{method_name}"
            )
        expected_hash = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        stored_hash = hashes[method_name]
        if not isinstance(stored_hash, str) or stored_hash != expected_hash:
            raise RenderError(f"Speed recipe fingerprint mismatch for {label}/{method_name}")
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _validate_speed_pair(
    comparison: Mapping[str, Any],
    *,
    label: str,
    baseline: str,
    candidate: str,
    repeats: int,
    sources: Sequence[str],
) -> dict:
    if (
        comparison.get("unit") != "candidate_speedup_x"
        or comparison.get("ratio_definition") != "baseline_latency / candidate_latency"
        or comparison.get("baseline_method") != baseline
        or comparison.get("candidate_method") != candidate
        or comparison.get("n_independent_processes") != 5
        or comparison.get("student_t_df") != 4
    ):
        raise RenderError(f"Fair-speed paired contract changed for {label}")
    values = [
        _number(value, f"{label}.process ratio", minimum=0.0)
        for value in _sequence(
            comparison.get("process_median_paired_speedups"),
            f"{label}.process_median_paired_speedups",
        )
    ]
    if len(values) != 5 or any(value <= 0.0 for value in values):
        raise RenderError(f"Fair-speed comparison needs five positive process ratios: {label}")
    logs = [math.log(value) for value in values]
    mean_log = statistics.fmean(logs)
    log_sd = statistics.stdev(logs)
    log_se = log_sd / math.sqrt(5.0)
    critical = 2.7764451052
    ratio = math.exp(mean_log)
    expected_ci = (
        math.exp(mean_log - critical * log_se),
        math.exp(mean_log + critical * log_se),
    )
    checks = {
        "geometric_mean_process_speedup_x": ratio,
        "median_process_speedup_x": statistics.median(values),
        "min_process_speedup_x": min(values),
        "max_process_speedup_x": max(values),
        "log_ratio_sample_sd": log_sd,
        "log_ratio_standard_error": log_se,
        "student_t_critical_95": critical,
    }
    for key, expected in checks.items():
        _assert_close(_number(comparison.get(key), f"{label}.{key}"), expected, f"{label}.{key}")
    ci = _ci(comparison.get("student_t_log_geomean_ci95"), f"{label}.ci", minimum=0.0)
    _assert_close(ci[0], expected_ci[0], f"{label}.ci low")
    _assert_close(ci[1], expected_ci[1], f"{label}.ci high")
    processes = _sequence(comparison.get("processes"), f"{label}.processes")
    if len(processes) != 5:
        raise RenderError(f"Fair-speed process audit is incomplete for {label}")
    process_values = []
    process_uuids = set()
    process_job_ids = set()
    expected_seeds = [20260715, 20260716, 20260717, 20260718, 20260719]
    for expected_seed, expected_source, process_raw in zip(expected_seeds, sources, processes):
        process = _mapping(process_raw, f"{label}.process")
        if (
            process.get("seed") != expected_seed
            or process.get("source") != expected_source
            or process.get("baseline_method") != baseline
            or process.get("candidate_method") != candidate
            or process.get("n_paired_repeats") != repeats
            or len(_sequence(process.get("raw_paired_ratios"), f"{label}.raw ratios")) != repeats
        ):
            raise RenderError(f"Fair-speed process pairing is incomplete for {label}")
        run_uuid = process.get("run_uuid")
        slurm_job_id = process.get("slurm_job_id")
        if (
            not isinstance(run_uuid, str)
            or not UUID4.fullmatch(run_uuid)
            or not isinstance(slurm_job_id, str)
            or not slurm_job_id
        ):
            raise RenderError(f"Fair-speed process identity is missing for {label}")
        process_uuids.add(run_uuid)
        process_job_ids.add(slurm_job_id)
        process_values.append(
            _number(
                process.get("process_median_speedup_x"),
                f"{label}.process_median_speedup_x",
                minimum=0.0,
            )
        )
    for actual, expected in zip(process_values, values):
        _assert_close(actual, expected, f"{label}.process ratio audit")
    if len(process_uuids) != 5 or len(process_job_ids) != 5:
        raise RenderError(f"Fair-speed process identities are reused for {label}")
    return {
        "ratio": ratio,
        "ci": ci,
        "run_uuids": process_uuids,
        "slurm_job_ids": process_job_ids,
    }


def _validate_speed_memory_pair(
    comparison: Mapping[str, Any],
    *,
    label: str,
    baseline: str,
    candidate: str,
    sources: Sequence[str],
) -> dict:
    """Recompute paired allocator-peak ratio and difference over five processes."""

    if (
        comparison.get("baseline_method") != baseline
        or comparison.get("candidate_method") != candidate
    ):
        raise RenderError(f"Peak-memory method labels changed for {label}")
    processes = _sequence(comparison.get("processes"), f"{label}.processes")
    if len(processes) != 5:
        raise RenderError(f"Peak-memory comparison needs five fresh processes: {label}")
    ratios = []
    differences = []
    run_uuids = set()
    slurm_job_ids = set()
    expected_seeds = [20260715, 20260716, 20260717, 20260718, 20260719]
    for expected_seed, expected_source, raw_process in zip(expected_seeds, sources, processes):
        process = _mapping(raw_process, f"{label}.process")
        if (
            process.get("seed") != expected_seed
            or process.get("source") != expected_source
            or process.get("baseline_method") != baseline
            or process.get("candidate_method") != candidate
            or process.get("n_paired_repeats") != 5
        ):
            raise RenderError(f"Peak-memory process pairing changed for {label}")
        raw_rows = _sequence(process.get("raw_paired_memory"), f"{label}.raw memory")
        if len(raw_rows) != 5:
            raise RenderError(f"Peak-memory technical repeat count changed for {label}")
        process_ratios = []
        process_differences = []
        for repeat_index, raw_row in enumerate(raw_rows):
            row = _mapping(raw_row, f"{label}.raw memory row")
            if row.get("repeat") != repeat_index:
                raise RenderError(f"Peak-memory repeats are not contiguous for {label}")
            baseline_value = _number(
                row.get("baseline_peak_allocated_mib"), f"{label}.baseline peak", minimum=0.0
            )
            candidate_value = _number(
                row.get("candidate_peak_allocated_mib"), f"{label}.candidate peak", minimum=0.0
            )
            if baseline_value <= 0.0 or candidate_value <= 0.0:
                raise RenderError(f"Peak allocated memory must be positive for {label}")
            ratio_value = baseline_value / candidate_value
            difference_value = baseline_value - candidate_value
            _assert_close(
                _number(row.get("candidate_memory_advantage_x"), f"{label}.raw ratio"),
                ratio_value,
                f"{label}.raw ratio",
            )
            _assert_close(
                _number(row.get("candidate_memory_reduction_mib"), f"{label}.raw difference"),
                difference_value,
                f"{label}.raw difference",
            )
            process_ratios.append(ratio_value)
            process_differences.append(difference_value)
        process_ratio = statistics.median(process_ratios)
        process_difference = statistics.median(process_differences)
        _assert_close(
            _number(
                process.get("process_median_memory_advantage_x"),
                f"{label}.process ratio",
            ),
            process_ratio,
            f"{label}.process ratio",
        )
        _assert_close(
            _number(
                process.get("process_median_memory_reduction_mib"),
                f"{label}.process difference",
            ),
            process_difference,
            f"{label}.process difference",
        )
        run_uuid = process.get("run_uuid")
        job_id = process.get("slurm_job_id")
        if (
            not isinstance(run_uuid, str)
            or not UUID4.fullmatch(run_uuid)
            or not isinstance(job_id, str)
            or not job_id
        ):
            raise RenderError(f"Peak-memory process identity is missing for {label}")
        run_uuids.add(run_uuid)
        slurm_job_ids.add(job_id)
        ratios.append(process_ratio)
        differences.append(process_difference)
    if len(run_uuids) != 5 or len(slurm_job_ids) != 5:
        raise RenderError(f"Peak-memory process identities are reused for {label}")

    ratio_record = _mapping(comparison.get("ratio"), f"{label}.ratio")
    logs = [math.log(value) for value in ratios]
    mean_log = statistics.fmean(logs)
    log_sd = statistics.stdev(logs)
    log_se = log_sd / math.sqrt(5.0)
    critical = 2.7764451052
    ratio_ci = (math.exp(mean_log - critical * log_se), math.exp(mean_log + critical * log_se))
    if (
        ratio_record.get("unit") != "candidate_memory_advantage_x"
        or ratio_record.get("ratio_definition")
        != "baseline_peak_allocated_mib / candidate_peak_allocated_mib"
        or ratio_record.get("n_independent_processes") != 5
        or ratio_record.get("student_t_df") != 4
    ):
        raise RenderError(f"Peak-memory ratio contract changed for {label}")
    stored_ratios = [
        _number(value, f"{label}.stored process memory ratio", minimum=0.0)
        for value in _sequence(
            ratio_record.get("process_median_paired_memory_advantages"),
            f"{label}.stored process memory ratios",
        )
    ]
    if len(stored_ratios) != 5:
        raise RenderError(f"Peak-memory process ratio coverage changed for {label}")
    for actual, expected in zip(stored_ratios, ratios):
        _assert_close(actual, expected, f"{label}.stored process memory ratio")
    _assert_close(
        _number(
            ratio_record.get("geometric_mean_process_memory_advantage_x"),
            f"{label}.memory ratio",
        ),
        math.exp(mean_log),
        f"{label}.memory ratio",
    )
    stored_ratio_ci = _ci(
        ratio_record.get("student_t_log_geomean_ci95"), f"{label}.memory ratio CI", minimum=0.0
    )
    _assert_close(stored_ratio_ci[0], ratio_ci[0], f"{label}.memory ratio CI low")
    _assert_close(stored_ratio_ci[1], ratio_ci[1], f"{label}.memory ratio CI high")

    difference_record = _mapping(comparison.get("difference"), f"{label}.difference")
    mean_difference = statistics.fmean(differences)
    difference_sd = statistics.stdev(differences)
    difference_se = difference_sd / math.sqrt(5.0)
    difference_ci = (
        mean_difference - critical * difference_se,
        mean_difference + critical * difference_se,
    )
    if (
        difference_record.get("unit") != "MiB"
        or difference_record.get("difference_definition")
        != "baseline_peak_allocated_mib - candidate_peak_allocated_mib"
        or difference_record.get("n_independent_processes") != 5
        or difference_record.get("student_t_df") != 4
    ):
        raise RenderError(f"Peak-memory difference contract changed for {label}")
    stored_differences = [
        _number(value, f"{label}.stored process memory difference")
        for value in _sequence(
            difference_record.get("process_median_paired_memory_reductions_mib"),
            f"{label}.stored process memory differences",
        )
    ]
    if len(stored_differences) != 5:
        raise RenderError(f"Peak-memory difference coverage changed for {label}")
    for actual, expected in zip(stored_differences, differences):
        _assert_close(actual, expected, f"{label}.stored process memory difference")
    _assert_close(
        _number(
            difference_record.get("mean_process_memory_reduction_mib"),
            f"{label}.mean memory difference",
        ),
        mean_difference,
        f"{label}.mean memory difference",
    )
    stored_difference_ci = _ci(
        difference_record.get("student_t_mean_difference_ci95_mib"),
        f"{label}.memory difference CI",
    )
    _assert_close(stored_difference_ci[0], difference_ci[0], f"{label}.difference CI low")
    _assert_close(stored_difference_ci[1], difference_ci[1], f"{label}.difference CI high")
    return {
        "ratios": ratios,
        "differences": differences,
        "run_uuids": run_uuids,
        "slurm_job_ids": slurm_job_ids,
    }


def _student_t4_cdf(value: float) -> float:
    value = float(value)
    if value == 0.0:
        return 0.5
    if value < 0.0:
        return 1.0 - _student_t4_cdf(-value)
    x = 4.0 / (4.0 + value * value)
    root = math.sqrt(1.0 - x)
    return 1.0 - 0.5 * (1.0 - 1.5 * root + 0.5 * root**3)


def _student_t4_quantile(probability: float) -> float:
    if not 0.5 < probability < 1.0:
        raise RenderError("Student-t4 upper quantile probability must be in (0.5, 1)")
    low, high = 0.0, 1.0
    while _student_t4_cdf(high) < probability:
        high *= 2.0
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
    raise RenderError(f"Unclassified speed endpoint {metric!r}")


def _validate_speed_pareto_contract(value: Any, models: Mapping[str, Any]) -> dict:
    """Recompute the only admissible complete speed/memory superiority gate."""

    contract = _mapping(value, "speed.pareto_contract")
    expected_keys = {
        "scope",
        "candidate_comparisons",
        "primary_compile_mode",
        "predeclared_timing_metrics",
        "predeclared_memory_metric",
        "n_model_rate_cells",
        "n_timing_endpoints",
        "n_memory_endpoints",
        "n_simultaneous_stochastic_endpoints",
        "simultaneous_inference",
        "all_predeclared_axes_present",
        "stochastic_endpoints",
        "public_unpacked_payload_tensor_constraints",
        "logical_stored_bits_equal_rate_constraints",
        "all_simultaneous_endpoint_lower_bounds_above_one",
        "all_public_payload_tensor_footprints_not_larger",
        "all_logical_stored_bits_equal",
        "complete_strict_superiority_established",
        "claim",
        "exclusions",
    }
    _exact_keys(contract, expected_keys, "speed.pareto_contract")
    scope = (
        "HQ and AdaptiveHQ versus TurboQuant Algorithm-1 reference; speed and "
        "the measured memory representations only, not downstream quality"
    )
    exclusions = (
        "public payload tensor bytes are unpacked logical-payload tensors; E2E peak "
        "allocated memory is for a dequantized research cache; neither is packed KV "
        "serving memory"
    )
    comparisons_by_proposal = {
        "HQ": "hq_vs_turboquant",
        "AdaptiveHQ": "adaptive_hq_vs_turboquant",
    }
    n_cells = len(LONG_MODELS) * len(RATES)
    n_timing = n_cells * len(SPEED_PAIRED_METRICS) * len(PROPOSAL_METHODS)
    n_memory = n_cells * len(PROPOSAL_METHODS)
    family_size = n_timing + n_memory
    alpha = 0.05
    critical = _student_t4_quantile(1.0 - alpha / family_size)
    if (
        contract.get("scope") != scope
        or contract.get("candidate_comparisons") != comparisons_by_proposal
        or contract.get("primary_compile_mode") != "off"
        or contract.get("predeclared_timing_metrics") != list(SPEED_PAIRED_METRICS)
        or contract.get("predeclared_memory_metric") != SPEED_PRIMARY_MEMORY_METRIC
        or contract.get("n_model_rate_cells") != n_cells
        or contract.get("n_timing_endpoints") != n_timing
        or contract.get("n_memory_endpoints") != n_memory
        or contract.get("n_simultaneous_stochastic_endpoints") != family_size
        or contract.get("all_predeclared_axes_present") is not True
        or contract.get("exclusions") != exclusions
    ):
        raise RenderError("Complete speed/memory Pareto coverage changed")
    inference = _mapping(contract.get("simultaneous_inference"), "speed.pareto inference")
    if (
        inference.get("method")
        != "one-sided Bonferroni Student-t intervals on process mean log ratios"
        or inference.get("familywise_confidence") != 0.95
        or inference.get("alpha") != alpha
        or inference.get("student_t_df") != 4
        or inference.get("statistical_unit") != "independently_launched_fresh_process"
    ):
        raise RenderError("Complete speed/memory simultaneous-inference contract changed")
    _assert_close(
        _number(inference.get("per_endpoint_alpha"), "speed Pareto per-endpoint alpha"),
        alpha / family_size,
        "speed Pareto per-endpoint alpha",
    )
    _assert_close(
        _number(inference.get("student_t_critical"), "speed Pareto critical"),
        critical,
        "speed Pareto critical",
    )

    endpoint_rows = _sequence(contract.get("stochastic_endpoints"), "speed Pareto endpoints")
    if len(endpoint_rows) != family_size:
        raise RenderError("Complete speed/memory stochastic endpoint coverage changed")
    expected_endpoints = []
    footprint_expected = []
    logical_expected = []
    for model in LONG_MODELS:
        for rate in RATES:
            row = models[model][rate]["off"]
            for proposal, comparison_name in comparisons_by_proposal.items():
                for metric in SPEED_PAIRED_METRICS:
                    comparison = row["paired_speedups"][metric][comparison_name]
                    values = [
                        _number(value, "speed Pareto source process ratio", minimum=0.0)
                        for value in _sequence(
                            comparison.get("process_median_paired_speedups"),
                            "speed Pareto source process ratios",
                        )
                    ]
                    if len(values) != 5 or any(value <= 0.0 for value in values):
                        raise RenderError("Speed Pareto source process ratio coverage changed")
                    logs = [math.log(value) for value in values]
                    mean_log = statistics.fmean(logs)
                    lower = math.exp(mean_log - critical * statistics.stdev(logs) / math.sqrt(5.0))
                    expected_endpoints.append(
                        {
                            "proposal": proposal,
                            "comparison": comparison_name,
                            "model": model,
                            "rate": int(rate),
                            "category": _speed_endpoint_category(metric),
                            "metric": metric,
                            "unit": "candidate_speedup_x",
                            "advantage_definition": "baseline_latency / candidate_latency",
                            "geometric_mean_advantage_x": math.exp(mean_log),
                            "bonferroni_one_sided_lower95": lower,
                            "candidate_strictly_better": lower > 1.0,
                        }
                    )
                memory = row["paired_peak_allocated_memory"][comparison_name]["ratio"]
                memory_values = [
                    _number(value, "speed Pareto memory process ratio", minimum=0.0)
                    for value in _sequence(
                        memory.get("process_median_paired_memory_advantages"),
                        "speed Pareto memory process ratios",
                    )
                ]
                if len(memory_values) != 5 or any(value <= 0.0 for value in memory_values):
                    raise RenderError("Speed Pareto memory process ratio coverage changed")
                logs = [math.log(value) for value in memory_values]
                mean_log = statistics.fmean(logs)
                lower = math.exp(mean_log - critical * statistics.stdev(logs) / math.sqrt(5.0))
                expected_endpoints.append(
                    {
                        "proposal": proposal,
                        "comparison": comparison_name,
                        "model": model,
                        "rate": int(rate),
                        "category": "research_fake_quant_e2e_allocator_peak",
                        "metric": SPEED_PRIMARY_MEMORY_METRIC,
                        "unit": "candidate_memory_advantage_x",
                        "advantage_definition": (
                            "baseline_peak_allocated_mib / candidate_peak_allocated_mib"
                        ),
                        "geometric_mean_advantage_x": math.exp(mean_log),
                        "bonferroni_one_sided_lower95": lower,
                        "candidate_strictly_better": lower > 1.0,
                    }
                )
            for proposal in PROPOSAL_METHODS:
                for regime in ("prefill", "decode"):
                    footprints = row["public_payload_footprints"][regime]
                    tq_footprint = footprints["TurboQuant-paper"]
                    proposal_footprint = footprints[proposal]
                    tq_bytes = _integer(
                        tq_footprint.get("materialized_payload_tensor_bytes"),
                        "TQ public payload tensor bytes",
                        minimum=1,
                    )
                    proposal_bytes = _integer(
                        proposal_footprint.get("materialized_payload_tensor_bytes"),
                        f"{proposal} public payload tensor bytes",
                        minimum=1,
                    )
                    tq_bits = _integer(
                        tq_footprint.get("logical_total_stored_bits"),
                        "TQ logical stored bits",
                        minimum=1,
                    )
                    proposal_bits = _integer(
                        proposal_footprint.get("logical_total_stored_bits"),
                        f"{proposal} logical stored bits",
                        minimum=1,
                    )
                    footprint_expected.append(
                        {
                            "proposal": proposal,
                            "model": model,
                            "rate": int(rate),
                            "regime": regime,
                            "baseline_materialized_payload_tensor_bytes": tq_bytes,
                            "candidate_materialized_payload_tensor_bytes": proposal_bytes,
                            "candidate_not_larger": proposal_bytes <= tq_bytes,
                        }
                    )
                    logical_expected.append(
                        {
                            "proposal": proposal,
                            "model": model,
                            "rate": int(rate),
                            "regime": regime,
                            "baseline_logical_stored_bits": tq_bits,
                            "candidate_logical_stored_bits": proposal_bits,
                            "equal_rate": proposal_bits == tq_bits,
                        }
                    )

    for actual_raw, expected in zip(endpoint_rows, expected_endpoints):
        actual = _mapping(actual_raw, "speed Pareto endpoint")
        _exact_keys(actual, set(expected), "speed Pareto endpoint")
        for key, expected_value in expected.items():
            if isinstance(expected_value, float):
                _assert_close(
                    _number(actual.get(key), f"speed Pareto endpoint {key}"),
                    expected_value,
                    f"speed Pareto endpoint {key}",
                )
            elif actual.get(key) != expected_value:
                raise RenderError(f"Speed Pareto endpoint {key} changed")
    footprint_rows = _sequence(
        contract.get("public_unpacked_payload_tensor_constraints"),
        "speed Pareto public payload constraints",
    )
    logical_rows = _sequence(
        contract.get("logical_stored_bits_equal_rate_constraints"),
        "speed Pareto logical stored-bit constraints",
    )
    if [
        dict(_mapping(row, "speed footprint constraint")) for row in footprint_rows
    ] != footprint_expected:
        raise RenderError("Public unpacked payload tensor-byte constraints changed")
    if [
        dict(_mapping(row, "speed logical constraint")) for row in logical_rows
    ] != logical_expected:
        raise RenderError("Logical stored-bit equal-rate constraints changed")
    all_endpoints = all(row["candidate_strictly_better"] for row in expected_endpoints)
    all_footprints = all(row["candidate_not_larger"] for row in footprint_expected)
    all_logical = all(row["equal_rate"] for row in logical_expected)
    established = all_endpoints and all_footprints and all_logical
    expected_claim = (
        "complete speed/measured-memory superiority established in the predeclared scope"
        if established
        else "complete speed/measured-memory superiority not established"
    )
    if (
        contract.get("all_simultaneous_endpoint_lower_bounds_above_one") is not all_endpoints
        or contract.get("all_public_payload_tensor_footprints_not_larger") is not all_footprints
        or contract.get("all_logical_stored_bits_equal") is not all_logical
        or contract.get("complete_strict_superiority_established") is not established
        or contract.get("claim") != expected_claim
    ):
        raise RenderError("Complete speed/memory superiority decision is stale or incomplete")
    return {
        "established": established,
        "claim": expected_claim,
        "critical": critical,
        "family_size": family_size,
        "endpoints": expected_endpoints,
    }


def _validate_speed(axis: Mapping[str, Any], commit: str) -> dict:
    _validate_commit(axis, commit, "speed")
    if (
        axis.get("benchmark") != "paired HayateQuant fair speed"
        or axis.get("model_evaluation_roles") != SPEED_MODEL_EVALUATION_ROLES
        or axis.get("official_complete") is not True
        or axis.get("primary_compile_mode") != "off"
        or axis.get("secondary_compile_modes") != ["decode-only"]
        or axis.get("primary_mode_predeclared_in_source") is not True
        or axis.get("statistical_unit") != "independently_launched_fresh_process"
        or axis.get("paired_ratio_estimator")
        != (
            "within-process median of repeatwise baseline/candidate latency ratios, "
            "then geometric mean across independent processes"
        )
        or axis.get("confidence_interval")
        != "two-sided 95% Student-t interval on process-level log ratios"
        or axis.get("public_payload_scope") != PUBLIC_LOGICAL_PAYLOAD_SCOPE
        or axis.get("public_payload_tensor_scope") != PUBLIC_PAYLOAD_TENSOR_SCOPE
        or axis.get("e2e_peak_allocated_memory_scope") != SPEED_PEAK_ALLOCATED_SCOPE
        or axis.get("implementation_scope_contract")
        != {
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
        }
    ):
        raise RenderError("speed axis is not the predeclared complete fair-speed matrix")
    expected_processes = len(LONG_MODELS) * len(RATES) * 2 * 5
    process_audit = _mapping(
        axis.get("independent_process_audit"), "speed.independent_process_audit"
    )
    if process_audit != {
        "n_unique_run_uuid4": expected_processes,
        "n_unique_slurm_job_ids": expected_processes,
        "uuid_and_slurm_job_id_unique_for_every_source": True,
    } or axis.get("five_process_design_note") != (
        "five independent processes are predeclared for a process-level "
        "Student-t interval; technical repeats remain paired within process "
        "and are never promoted to inferential replicates"
    ):
        raise RenderError("speed independent-process audit is incomplete")
    axis_sources = _sequence(axis.get("sources"), "speed.sources")
    if len(axis_sources) != expected_processes:
        raise RenderError("speed source count does not match the formal process matrix")
    axis_source_paths = {_mapping(source, "speed source").get("path") for source in axis_sources}
    if len(axis_source_paths) != expected_processes or not all(
        isinstance(path, str) and path for path in axis_source_paths
    ):
        raise RenderError("speed source ledger contains duplicate paths")
    hardware = _mapping(axis.get("hardware_stratum"), "speed.hardware_stratum")
    if not isinstance(hardware.get("gpu_name"), str) or not hardware["gpu_name"]:
        raise RenderError("Speed GPU stratum is missing")
    capability = _sequence(hardware.get("compute_capability"), "speed compute capability")
    if len(capability) != 2 or not all(isinstance(value, int) for value in capability):
        raise RenderError("Speed compute capability is malformed")
    _integer(hardware.get("total_memory_bytes"), "speed total memory", minimum=1)
    _integer(hardware.get("multi_processor_count"), "speed SM count", minimum=1)
    if not isinstance(hardware.get("driver_version"), str) or not hardware["driver_version"]:
        raise RenderError("Speed driver stratum is missing")
    software = _mapping(axis.get("software_stratum"), "speed.software_stratum")
    protocol = _mapping(axis.get("timing_protocol_stratum"), "speed.timing_protocol_stratum")
    if not software or not protocol:
        raise RenderError("Speed software/protocol stratum is missing")
    allocation = _mapping(
        axis.get("allocation_isolation_contract"), "speed.allocation_isolation_contract"
    )
    expected_allocation = {
        **SPEED_ALLOCATION_CONTRACT,
        "allocation_cuda_visibility_recorded_before_process_narrowing": True,
    }
    if any(allocation.get(key) != value for key, value in expected_allocation.items()):
        raise RenderError("Speed allocation/isolation contract changed")
    physical_node = allocation.get("physical_node")
    if (
        set(allocation) != {*expected_allocation, "physical_node"}
        or not isinstance(physical_node, str)
        or not physical_node
    ):
        raise RenderError("Speed physical-node isolation provenance is missing")
    models = _mapping(axis.get("models"), "speed.models")
    _exact_keys(models, set(LONG_MODELS), "speed.models")
    matrix_source_paths = set()
    matrix_run_uuids = set()
    matrix_slurm_job_ids = set()
    recipe_digests_by_rate: dict[str, set[str]] = {rate: set() for rate in RATES}
    footprint_digests_by_cell: dict[tuple[str, str], set[str]] = {
        (model, rate): set() for model in LONG_MODELS for rate in RATES
    }
    for model in LONG_MODELS:
        rates = _mapping(models[model], f"speed.{model}")
        _exact_keys(rates, set(RATES), f"speed.{model}")
        for rate in RATES:
            modes = _mapping(rates[rate], f"speed.{model}.{rate}")
            _exact_keys(modes, {"off", "decode-only"}, f"speed.{model}.{rate}")
            names = {
                "full": "fullkv_uncompressed",
                "tq": "TurboQuant-paper",
                "hq": "HQ",
                "adaptive": "AdaptiveHQ",
            }
            for mode in ("off", "decode-only"):
                row = _mapping(modes[mode], f"speed.{model}.{rate}.{mode}")
                if row.get("n_replicates") != 5 or row.get("seeds") != [
                    20260715,
                    20260716,
                    20260717,
                    20260718,
                    20260719,
                ]:
                    raise RenderError(f"Speed replicate count changed for {model}/{rate}/{mode}")
                row_sources = _sequence(row.get("sources"), f"speed sources {model}/{rate}/{mode}")
                if len(row_sources) != 5 or not all(
                    isinstance(value, str) and value for value in row_sources
                ):
                    raise RenderError(
                        f"Speed process source audit is incomplete for {model}/{rate}/{mode}"
                    )
                if matrix_source_paths.intersection(row_sources):
                    raise RenderError("Speed process source is reused across matrix points")
                matrix_source_paths.update(row_sources)
                recipe_digests_by_rate[rate].add(
                    _validate_speed_scientific_recipe(
                        row.get("scientific_recipe"),
                        rate=int(rate),
                        names=names,
                        label=f"speed.{model}.{rate}.{mode}",
                    )
                )
                for method in names.values():
                    _speed_metric(
                        row,
                        f"e2e.summary.{method}.decode_steady_throughput",
                        "tokens_per_second",
                    )
                for method in (names["tq"], names["hq"], names["adaptive"]):
                    _speed_metric(
                        row,
                        "micro.regimes.decode.payload_api."
                        f"roundtrip_explicit_encode_then_decode.summary.{method}",
                        "ms_per_call",
                    )
                footprints = _mapping(
                    row.get("public_payload_footprints"),
                    f"speed public payload footprints {model}/{rate}/{mode}",
                )
                _exact_keys(
                    footprints,
                    {"prefill", "decode"},
                    f"speed public payload footprint regimes {model}/{rate}/{mode}",
                )
                for regime in ("prefill", "decode"):
                    methods = _mapping(
                        footprints[regime],
                        f"speed public payload footprint {model}/{rate}/{mode}/{regime}",
                    )
                    _exact_keys(
                        methods,
                        {names["tq"], names["hq"], names["adaptive"]},
                        f"speed public payload methods {model}/{rate}/{mode}/{regime}",
                    )
                    logical_bits = set()
                    for method_name, raw_footprint in methods.items():
                        footprint = _mapping(
                            raw_footprint,
                            f"speed footprint {model}/{rate}/{mode}/{regime}/{method_name}",
                        )
                        _exact_keys(
                            footprint,
                            {"materialized_payload_tensor_bytes", "logical_total_stored_bits"},
                            f"speed footprint {model}/{rate}/{mode}/{regime}/{method_name}",
                        )
                        _integer(
                            footprint["materialized_payload_tensor_bytes"],
                            "public payload physical tensor bytes",
                            minimum=1,
                        )
                        logical_bits.add(
                            _integer(
                                footprint["logical_total_stored_bits"],
                                "public payload logical stored bits",
                                minimum=1,
                            )
                        )
                    if len(logical_bits) != 1:
                        raise RenderError(
                            f"Public payload logical rate differs across methods for "
                            f"{model}/{rate}/{mode}/{regime}"
                        )
                footprint_digests_by_cell[(model, rate)].add(
                    hashlib.sha256(
                        json.dumps(
                            footprints,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    ).hexdigest()
                )
                paired = _mapping(
                    row.get("paired_speedups"), f"speed paired {model}/{rate}/{mode}"
                )
                _exact_keys(
                    paired,
                    set(SPEED_PAIRED_METRICS),
                    f"speed paired metrics {model}/{rate}/{mode}",
                )
                expected_pairs = {
                    "hq_vs_turboquant": (names["tq"], names["hq"]),
                    "adaptive_hq_vs_turboquant": (names["tq"], names["adaptive"]),
                }
                for metric_name in SPEED_PAIRED_METRICS:
                    comparisons = _mapping(
                        paired[metric_name],
                        f"speed paired {model}/{rate}/{mode}/{metric_name}",
                    )
                    _exact_keys(
                        comparisons,
                        set(SPEED_COMPARISONS),
                        f"speed comparisons {model}/{rate}/{mode}/{metric_name}",
                    )
                    repeats = 5 if metric_name.startswith("e2e.") else 40
                    for comparison_name, (baseline, candidate) in expected_pairs.items():
                        process_audit_row = _validate_speed_pair(
                            _mapping(
                                comparisons[comparison_name],
                                f"speed pair {model}/{rate}/{mode}/{metric_name}/{comparison_name}",
                            ),
                            label=f"speed.{model}.{rate}.{mode}.{metric_name}.{comparison_name}",
                            baseline=baseline,
                            candidate=candidate,
                            repeats=repeats,
                            sources=row_sources,
                        )
                        matrix_run_uuids.update(process_audit_row["run_uuids"])
                        matrix_slurm_job_ids.update(process_audit_row["slurm_job_ids"])
                paired_memory = _mapping(
                    row.get("paired_peak_allocated_memory"),
                    f"speed paired peak memory {model}/{rate}/{mode}",
                )
                _exact_keys(
                    paired_memory,
                    set(SPEED_COMPARISONS),
                    f"speed peak-memory comparisons {model}/{rate}/{mode}",
                )
                for comparison_name, (baseline, candidate) in expected_pairs.items():
                    memory_audit = _validate_speed_memory_pair(
                        _mapping(
                            paired_memory[comparison_name],
                            f"speed memory pair {model}/{rate}/{mode}/{comparison_name}",
                        ),
                        label=f"speed.{model}.{rate}.{mode}.memory.{comparison_name}",
                        baseline=baseline,
                        candidate=candidate,
                        sources=row_sources,
                    )
                    matrix_run_uuids.update(memory_audit["run_uuids"])
                    matrix_slurm_job_ids.update(memory_audit["slurm_job_ids"])
    if matrix_source_paths != axis_source_paths:
        raise RenderError("Speed matrix sources differ from the content-hashed source ledger")
    if (
        len(matrix_run_uuids) != expected_processes
        or len(matrix_slurm_job_ids) != expected_processes
    ):
        raise RenderError("Speed process identities are reused across matrix points")
    if any(len(digests) != 1 for digests in recipe_digests_by_rate.values()):
        raise RenderError("Speed scientific recipe changed across model/compile strata")
    if any(len(digests) != 1 for digests in footprint_digests_by_cell.values()):
        raise RenderError("Public payload tensor/logical footprint changed across compile modes")
    pareto = _validate_speed_pareto_contract(axis.get("pareto_contract"), models)
    expected_multiplicity = {
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
        "blanket_or_global_speed_claim_allowed": pareto["established"],
    }
    if axis.get("multiplicity_contract") != expected_multiplicity:
        raise RenderError(
            "speed axis is not the predeclared complete fair-speed matrix: "
            "multiplicity/complete-dominance claim contract changed"
        )
    return {
        "models": models,
        "hardware": hardware,
        "public_payload_scope": dict(PUBLIC_LOGICAL_PAYLOAD_SCOPE),
        "public_payload_tensor_scope": PUBLIC_PAYLOAD_TENSOR_SCOPE,
        "peak_allocated_memory_scope": dict(SPEED_PEAK_ALLOCATED_SCOPE),
        "pareto": pareto,
    }


def _validate_formal_matrix_status(value: Any, *, point_gate_passed: bool) -> dict:
    """Require the producer's exact, complete eight-axis admission record."""
    status = _mapping(value, "formal_matrix_status")
    _exact_keys(
        status,
        {
            "required_axes",
            "present_axes",
            "missing_axes",
            "incomplete_axes",
            "one_cross_axis_provenance_identity",
            "coverage_complete",
            "point_estimate_win_gate_passed",
        },
        "formal_matrix_status",
    )
    expected = {
        "required_axes": list(REQUIRED_AXES),
        "present_axes": list(REQUIRED_AXES),
        "missing_axes": [],
        "incomplete_axes": [],
        "one_cross_axis_provenance_identity": True,
        "coverage_complete": True,
        "point_estimate_win_gate_passed": point_gate_passed,
    }
    if dict(status) != expected:
        raise RenderError("formal_matrix_status must attest the exact complete eight-axis matrix")
    return expected


def _validate_formal_smoke_approval(
    value: Any,
    *,
    input_manifest: Mapping[str, Any],
    top_commit: str,
) -> dict:
    approval = _mapping(value, "formal_provenance.formal_smoke_approval")
    _exact_keys(
        approval,
        {
            "verification_mode",
            "marker",
            "validated_plan_sha256",
            "smoke_plan_identity",
            "code_commit",
            "input_manifest",
            "completed_job_keys",
            "results",
            "worker_preflights",
        },
        "formal_provenance.formal_smoke_approval",
    )

    def file_record(raw: Any, *, label: str, root: Path | None = None) -> dict:
        record = _mapping(raw, label)
        _exact_keys(record, {"path", "bytes", "sha256"}, label)
        raw_path = record.get("path")
        digest = record.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or not Path(raw_path).is_absolute()
            or Path(raw_path).resolve(strict=False) != Path(raw_path)
            or not isinstance(digest, str)
            or not SHA64.fullmatch(digest)
        ):
            raise RenderError(f"{label} is malformed")
        _integer(record.get("bytes"), f"{label}.bytes", minimum=1)
        if root is not None:
            try:
                Path(raw_path).relative_to(root)
            except ValueError as exc:
                raise RenderError(f"{label} escapes its approved run-root scope") from exc
        return dict(record)

    marker_raw = _mapping(approval.get("marker"), "formal_provenance.formal_smoke_approval.marker")
    marker_path_raw = marker_raw.get("path")
    if not isinstance(marker_path_raw, str) or Path(marker_path_raw).parent.name != "manifests":
        raise RenderError("formal smoke approval marker is outside a manifests directory")
    run_root = Path(marker_path_raw).parent.parent
    marker = file_record(
        marker_raw,
        label="formal_provenance.formal_smoke_approval.marker",
        root=run_root / "manifests",
    )
    plan_sha256 = approval.get("validated_plan_sha256")
    plan_identity = _mapping(
        approval.get("smoke_plan_identity"),
        "formal_provenance.formal_smoke_approval.smoke_plan_identity",
    )
    _exact_keys(
        plan_identity,
        {"version", "sha256"},
        "formal_provenance.formal_smoke_approval.smoke_plan_identity",
    )
    approved_input = file_record(
        approval.get("input_manifest"),
        label="formal_provenance.formal_smoke_approval.input_manifest",
    )
    if (
        approval.get("verification_mode") != FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE
        or approval.get("code_commit") != top_commit
        or not isinstance(plan_sha256, str)
        or not SHA64.fullmatch(plan_sha256)
        or plan_identity.get("version") != FORMAL_SMOKE_PLAN_VERSION
        or not isinstance(plan_identity.get("sha256"), str)
        or not SHA64.fullmatch(plan_identity["sha256"])
        or approved_input.get("path") != input_manifest.get("path")
        or approved_input.get("sha256") != input_manifest.get("sha256")
        or approval.get("completed_job_keys") != list(FORMAL_SMOKE_JOB_KEYS)
    ):
        raise RenderError("formal smoke approval binding identity changed")

    def keyed_records(raw: Any, *, label: str, root: Path) -> list[dict]:
        records = _sequence(raw, label)
        if len(records) != len(FORMAL_SMOKE_JOB_KEYS):
            raise RenderError(f"{label} does not cover the exact smoke plan")
        normalized = []
        seen_paths: set[str] = set()
        for index, value in enumerate(records):
            record = _mapping(value, f"{label}[{index}]")
            _exact_keys(
                record,
                {"key", "path", "bytes", "sha256"},
                f"{label}[{index}]",
            )
            key = record.get("key")
            if key != FORMAL_SMOKE_JOB_KEYS[index]:
                raise RenderError(f"{label} order/key identity changed")
            normalized_record = {
                "key": key,
                **file_record(
                    {name: record.get(name) for name in ("path", "bytes", "sha256")},
                    label=f"{label}[{index}]",
                    root=root,
                ),
            }
            if normalized_record["path"] in seen_paths:
                raise RenderError(f"{label} contains a duplicate path")
            seen_paths.add(normalized_record["path"])
            normalized.append(normalized_record)
        return normalized

    results = keyed_records(
        approval.get("results"),
        label="formal_provenance.formal_smoke_approval.results",
        root=run_root / "results/smoke",
    )
    preflights = keyed_records(
        approval.get("worker_preflights"),
        label="formal_provenance.formal_smoke_approval.worker_preflights",
        root=run_root / "provenance",
    )
    return {
        "verification_mode": FORMAL_SMOKE_APPROVAL_VERIFICATION_MODE,
        "marker": marker,
        "validated_plan_sha256": plan_sha256,
        "smoke_plan_identity": dict(plan_identity),
        "code_commit": top_commit,
        "input_manifest": approved_input,
        "completed_job_keys": list(FORMAL_SMOKE_JOB_KEYS),
        "results": results,
        "worker_preflights": preflights,
    }


def _validate_formal_provenance(
    value: Any, *, document: Mapping[str, Any], top_commit: str
) -> dict:
    """Validate the aggregate producer's cross-axis scientific identity."""
    provenance = _mapping(value, "formal_provenance")
    _exact_keys(
        provenance,
        {
            "input_manifest",
            "models",
            "dataset_files",
            "axis_source_counts",
            "validated_source_files",
            "quality_runtime",
            "formal_smoke_approval",
        },
        "formal_provenance",
    )

    input_manifest = _mapping(provenance.get("input_manifest"), "formal_provenance.input_manifest")
    _exact_keys(
        input_manifest,
        {"path", "sha256"},
        "formal_provenance.input_manifest",
    )
    input_path = input_manifest.get("path")
    input_sha256 = input_manifest.get("sha256")
    if (
        not isinstance(input_path, str)
        or not input_path
        or not Path(input_path).is_absolute()
        or not isinstance(input_sha256, str)
        or not SHA64.fullmatch(input_sha256.lower())
    ):
        raise RenderError("formal_provenance.input_manifest is malformed")
    formal_smoke_approval = _validate_formal_smoke_approval(
        provenance.get("formal_smoke_approval"),
        input_manifest=input_manifest,
        top_commit=top_commit,
    )

    models_raw = _mapping(provenance.get("models"), "formal_provenance.models")
    _exact_keys(models_raw, set(PAPER_MODELS), "formal_provenance.models")
    models = {}
    for label in PAPER_MODELS:
        model = _mapping(models_raw[label], f"formal_provenance.models.{label}")
        _exact_keys(
            model,
            {"model_key", "model_path", "identity_sha256"},
            f"formal_provenance.models.{label}",
        )
        model_key = model.get("model_key")
        model_path = model.get("model_path")
        identity_sha256 = model.get("identity_sha256")
        if (
            not isinstance(model_key, str)
            or not model_key
            or not isinstance(model_path, str)
            or not model_path
            or not Path(model_path).is_absolute()
            or not isinstance(identity_sha256, str)
            or not SHA64.fullmatch(identity_sha256.lower())
        ):
            raise RenderError(f"formal_provenance model identity is malformed for {label}")
        models[label] = dict(model)

    dataset_files_raw = _sequence(
        provenance.get("dataset_files"), "formal_provenance.dataset_files"
    )
    if not dataset_files_raw:
        raise RenderError("formal_provenance.dataset_files must not be empty")
    dataset_files = []
    dataset_paths = set()
    for index, raw in enumerate(dataset_files_raw):
        dataset = _mapping(raw, f"formal_provenance.dataset_files[{index}]")
        _exact_keys(
            dataset,
            {"path", "bytes", "sha256"},
            f"formal_provenance.dataset_files[{index}]",
        )
        path = dataset.get("path")
        digest = dataset.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or not Path(path).is_absolute()
            or path in dataset_paths
            or not isinstance(digest, str)
            or not SHA64.fullmatch(digest.lower())
        ):
            raise RenderError(
                f"formal_provenance.dataset_files[{index}] is malformed or duplicated"
            )
        _integer(
            dataset.get("bytes"),
            f"formal_provenance.dataset_files[{index}].bytes",
            minimum=1,
        )
        dataset_paths.add(path)
        dataset_files.append(dict(dataset))
    matching_input_files = [
        dataset
        for dataset in dataset_files
        if dataset["path"] == input_path and dataset["sha256"].lower() == input_sha256.lower()
    ]
    if len(matching_input_files) != 1:
        raise RenderError(
            "formal_provenance dataset ledger does not contain the exact input manifest"
        )
    if formal_smoke_approval["input_manifest"] != matching_input_files[0]:
        raise RenderError("formal smoke approval input manifest differs from the dataset ledger")

    source_counts = _mapping(
        provenance.get("axis_source_counts"),
        "formal_provenance.axis_source_counts",
    )
    _exact_keys(
        source_counts,
        set(REQUIRED_AXES),
        "formal_provenance.axis_source_counts",
    )
    all_source_paths = set()
    normalized_source_counts = {}
    for axis in REQUIRED_AXES:
        axis_document = _mapping(document.get(axis), axis)
        axis_sources = _sequence(axis_document.get("sources"), f"{axis}.sources")
        axis_paths = {
            _mapping(raw, f"{axis}.sources[{index}]").get("path")
            for index, raw in enumerate(axis_sources)
        }
        if None in axis_paths or len(axis_paths) != len(axis_sources):
            raise RenderError(f"{axis}.sources contains a missing or duplicate path")
        declared_count = _integer(
            source_counts.get(axis),
            f"formal_provenance.axis_source_counts.{axis}",
            minimum=1,
        )
        if declared_count != len(axis_paths):
            raise RenderError(f"formal_provenance source count is stale for axis {axis}")
        normalized_source_counts[axis] = declared_count
        all_source_paths.update(axis_paths)
    validated_source_files = _integer(
        provenance.get("validated_source_files"),
        "formal_provenance.validated_source_files",
        minimum=1,
    )
    if validated_source_files != len(all_source_paths):
        raise RenderError(
            "formal_provenance.validated_source_files does not match the source-ledger union"
        )

    quality_runtime = _mapping(
        provenance.get("quality_runtime"), "formal_provenance.quality_runtime"
    )
    _exact_keys(
        quality_runtime,
        {
            "fingerprint_sha256",
            "normalized",
            "scope",
            "excluded_nuisance_identity",
        },
        "formal_provenance.quality_runtime",
    )
    if quality_runtime.get("scope") != (
        "all quality axes; speed retains its separate timing-stratum gate"
    ) or quality_runtime.get("excluded_nuisance_identity") != [
        "GPU UUID",
        "host name",
        "Slurm job id",
    ]:
        raise RenderError("formal_provenance quality-runtime scope changed")
    normalized = _mapping(
        quality_runtime.get("normalized"),
        "formal_provenance.quality_runtime.normalized",
    )
    _exact_keys(
        normalized,
        {
            "python",
            "packages",
            "onecomp_commit",
            "cuda",
            "gpu_classes",
            "semantic_environment",
        },
        "formal_provenance.quality_runtime.normalized",
    )
    if not isinstance(normalized.get("python"), str) or not normalized["python"]:
        raise RenderError("formal_provenance normalized Python version is missing")
    if normalized.get("onecomp_commit") != top_commit:
        raise RenderError("formal_provenance normalized onecomp commit changed")
    packages = _mapping(
        normalized.get("packages"),
        "formal_provenance.quality_runtime.normalized.packages",
    )
    _exact_keys(
        packages,
        QUALITY_SOFTWARE_PACKAGES,
        "formal_provenance.quality_runtime.normalized.packages",
    )
    if any(not isinstance(version, str) or not version for version in packages.values()):
        raise RenderError("formal_provenance contains a missing package version")
    cuda = _mapping(normalized.get("cuda"), "formal_provenance.quality_runtime.normalized.cuda")
    _exact_keys(cuda, {"torch_cuda", "cudnn"}, "formal_provenance normalized CUDA")
    if (
        not isinstance(cuda.get("torch_cuda"), str)
        or not cuda["torch_cuda"]
        or cuda.get("cudnn") is None
    ):
        raise RenderError("formal_provenance normalized CUDA software is incomplete")
    gpu_classes_raw = _sequence(
        normalized.get("gpu_classes"),
        "formal_provenance.quality_runtime.normalized.gpu_classes",
    )
    if not gpu_classes_raw:
        raise RenderError("formal_provenance normalized GPU classes are missing")
    gpu_classes = []
    for index, raw in enumerate(gpu_classes_raw):
        gpu_class = _mapping(
            raw,
            f"formal_provenance.quality_runtime.normalized.gpu_classes[{index}]",
        )
        _exact_keys(
            gpu_class,
            {"name", "compute_capability"},
            f"formal_provenance normalized GPU class {index}",
        )
        capability = _sequence(
            gpu_class.get("compute_capability"),
            f"formal_provenance normalized GPU class {index}.compute_capability",
        )
        if (
            not isinstance(gpu_class.get("name"), str)
            or "b200" not in gpu_class["name"].lower()
            or len(capability) != 2
        ):
            raise RenderError("formal quality provenance is not an NVIDIA B200 class")
        for component_index, component in enumerate(capability):
            _integer(
                component,
                "formal_provenance normalized GPU class "
                f"{index}.compute_capability[{component_index}]",
            )
        gpu_classes.append(dict(gpu_class))
    semantic_environment = _mapping(
        normalized.get("semantic_environment"),
        "formal_provenance.quality_runtime.normalized.semantic_environment",
    )
    if dict(semantic_environment) != QUALITY_SEMANTIC_ENVIRONMENT:
        raise RenderError("formal_provenance normalized ONECOMP environment changed")

    fingerprint = quality_runtime.get("fingerprint_sha256")
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if (
        not isinstance(fingerprint, str)
        or not SHA64.fullmatch(fingerprint.lower())
        or fingerprint.lower() != expected_fingerprint
    ):
        raise RenderError("formal_provenance quality-runtime fingerprint is inconsistent")
    return {
        "input_manifest": dict(input_manifest),
        "models": models,
        "dataset_files": dataset_files,
        "axis_source_counts": normalized_source_counts,
        "validated_source_files": validated_source_files,
        "formal_smoke_approval": formal_smoke_approval,
        "quality_runtime": {
            "fingerprint_sha256": fingerprint,
            "normalized": dict(normalized),
            "scope": quality_runtime["scope"],
            "excluded_nuisance_identity": list(quality_runtime["excluded_nuisance_identity"]),
        },
    }


def _validate_point_estimate_win_gate(value: Any) -> dict:
    gate = _mapping(value, "point_estimate_win_gate")
    if (
        gate.get("schema_version") != 1
        or gate.get("threshold") != 0.90
        or gate.get("threshold_comparison") != "win_fraction_unrounded >= threshold"
        or gate.get("cell_win_rule") != "strict_unrounded_point_estimate_only"
        or not isinstance(gate.get("passed"), bool)
    ):
        raise RenderError("predeclared point-estimate 90% gate schema changed")
    cells = gate.get("cells")
    if not isinstance(cells, list) or len(cells) != 776:
        raise RenderError("point-estimate gate cell ledger is missing")
    expected_quality_counts = {
        "TurboQuant-paper": 164,
        "FibQuant-paper": 164,
    }
    expected_speed_count = 60
    summaries = _mapping(
        gate.get("proposal_baseline_summaries"),
        "point_estimate_win_gate.proposal_baseline_summaries",
    )
    if set(summaries) != {"HQ", "AdaptiveHQ"}:
        raise RenderError("point-estimate gate proposal set changed")
    summary_passes: list[bool] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise RenderError("point-estimate gate contains a malformed cell")
        proposal = cell.get("proposal_value_unrounded")
        baseline = cell.get("baseline_value_unrounded")
        axis = cell.get("axis")
        expected_baselines = (
            {"TurboQuant-paper"} if axis == "speed" else {"TurboQuant-paper", "FibQuant-paper"}
        )
        if (
            cell.get("proposal") not in {"HQ", "AdaptiveHQ"}
            or cell.get("baseline") not in expected_baselines
            or isinstance(proposal, bool)
            or not isinstance(proposal, (int, float))
            or not math.isfinite(float(proposal))
            or isinstance(baseline, bool)
            or not isinstance(baseline, (int, float))
            or not math.isfinite(float(baseline))
            or cell.get("complete") is not True
            or cell.get("matched_comparison") is not True
            or cell.get("operating_point_matched") is not True
            or cell.get("included_in_denominator") is not True
        ):
            raise RenderError("point-estimate gate admitted an incomplete/nonfinite cell")
        direction = cell.get("direction")
        strict_win = (
            float(proposal) < float(baseline)
            if direction == "lower_is_better"
            else (float(proposal) > float(baseline) if direction == "higher_is_better" else None)
        )
        if strict_win is None or cell.get("win") is not strict_win:
            raise RenderError("point-estimate gate cell direction/win was tampered")
        expected_status = (
            "strict_win"
            if strict_win
            else "tie_nonwin" if float(proposal) == float(baseline) else "strict_loss"
        )
        if cell.get("status") != expected_status:
            raise RenderError("point-estimate gate cell status was tampered")
    for proposal in ("HQ", "AdaptiveHQ"):
        proposal_summaries = _mapping(
            summaries[proposal], f"point_estimate_win_gate summaries {proposal}"
        )
        expected_keys = {
            "TurboQuant-paper",
            "FibQuant-paper",
            "TurboQuant-paper-speed",
        }
        if set(proposal_summaries) != expected_keys:
            raise RenderError(f"point-estimate gate baseline set changed for {proposal}")
        for baseline, expected_count in expected_quality_counts.items():
            rows = [
                row
                for row in cells
                if row.get("proposal") == proposal
                and row.get("baseline") == baseline
                and row.get("axis") != "speed"
            ]
            summary = _mapping(
                proposal_summaries[baseline],
                f"point-estimate gate {proposal}/{baseline}",
            )
            wins = sum(row.get("win") is True for row in rows)
            if (
                len(rows) != expected_count
                or summary.get("scope") != "quality"
                or summary.get("expected_cells") != expected_count
                or summary.get("complete_matched_cells") != expected_count
                or summary.get("wins") != wins
                or summary.get("nonwins") != expected_count - wins
                or summary.get("win_fraction_unrounded") != wins / expected_count
                or summary.get("threshold") != 0.90
                or summary.get("passed") is not (wins / expected_count >= 0.90)
            ):
                raise RenderError(
                    f"point-estimate quality gate summary failed for {proposal}/{baseline}"
                )
            summary_passes.append(bool(summary["passed"]))
        speed_rows = [
            row for row in cells if row.get("proposal") == proposal and row.get("axis") == "speed"
        ]
        speed_summary = _mapping(
            proposal_summaries["TurboQuant-paper-speed"],
            f"point-estimate speed gate {proposal}",
        )
        speed_wins = sum(row.get("win") is True for row in speed_rows)
        if (
            len(speed_rows) != expected_speed_count
            or any(row.get("baseline") != "TurboQuant-paper" for row in speed_rows)
            or speed_summary.get("scope") != "speed"
            or speed_summary.get("expected_cells") != expected_speed_count
            or speed_summary.get("complete_matched_cells") != expected_speed_count
            or speed_summary.get("wins") != speed_wins
            or speed_summary.get("nonwins") != expected_speed_count - speed_wins
            or speed_summary.get("win_fraction_unrounded") != speed_wins / expected_speed_count
            or speed_summary.get("threshold") != 0.90
            or speed_summary.get("passed") is not (speed_wins / expected_speed_count >= 0.90)
        ):
            raise RenderError(f"point-estimate speed gate summary failed for {proposal}")
        summary_passes.append(bool(speed_summary["passed"]))
    applicability = _mapping(
        gate.get("baseline_applicability"),
        "point_estimate_win_gate.baseline_applicability",
    )
    block = _mapping(applicability.get("BlockQuant-paper"), "BlockQuant gate scope")
    if (
        block.get("quality") is not True
        or block.get("speed") is not False
        or block.get("included_in_denominator") is not False
        or block.get("rate_match_required") is not False
        or block.get("operating_point_matched") is not False
        or block.get("paper_reproduction") is not False
        or block.get("reason_code") != "unmatched_source_operating_point_descriptive_only"
    ):
        raise RenderError("BlockQuant point-estimate source operating-point scope changed")
    descriptive = _mapping(
        gate.get("unmatched_descriptive_comparisons"),
        "point_estimate_win_gate.unmatched_descriptive_comparisons",
    )
    if (
        descriptive.get("baseline") != "BlockQuant-paper"
        or descriptive.get("scope") != "source_operating_point_quality_only"
        or descriptive.get("included_in_point_estimate_gate") is not False
        or descriptive.get("reason_code") != "payload_operating_point_not_matched_to_hq"
    ):
        raise RenderError("BlockQuant unmatched descriptive scope changed")
    descriptive_cells = descriptive.get("cells")
    descriptive_summaries = _mapping(
        descriptive.get("proposal_summaries"), "BlockQuant descriptive summaries"
    )
    if not isinstance(descriptive_cells, list) or set(descriptive_summaries) != {
        "HQ",
        "AdaptiveHQ",
    }:
        raise RenderError("BlockQuant unmatched descriptive ledger is malformed")
    for cell in descriptive_cells:
        if not isinstance(cell, Mapping):
            raise RenderError("BlockQuant descriptive ledger contains a malformed cell")
        proposal = cell.get("proposal_value_unrounded")
        baseline = cell.get("baseline_value_unrounded")
        if (
            cell.get("baseline") != "BlockQuant-paper"
            or cell.get("sample_set_matched") is not True
            or cell.get("matched_comparison") is not False
            or cell.get("rate_match_required") is not False
            or cell.get("operating_point_matched") is not False
            or cell.get("included_in_denominator") is not False
            or cell.get("complete") is not True
            or isinstance(proposal, bool)
            or not isinstance(proposal, (int, float))
            or not math.isfinite(float(proposal))
            or isinstance(baseline, bool)
            or not isinstance(baseline, (int, float))
            or not math.isfinite(float(baseline))
        ):
            raise RenderError("BlockQuant descriptive cell scope/data is malformed")
        direction = cell.get("direction")
        better = (
            float(proposal) < float(baseline)
            if direction == "lower_is_better"
            else (float(proposal) > float(baseline) if direction == "higher_is_better" else None)
        )
        suffix = (
            "strict_win"
            if better is True
            else "tie_nonwin" if float(proposal) == float(baseline) else "strict_loss"
        )
        if (
            better is None
            or cell.get("descriptive_better") is not better
            or cell.get("status") != f"unmatched_descriptive_{suffix}"
        ):
            raise RenderError("BlockQuant descriptive direction/status was tampered")
    for proposal in ("HQ", "AdaptiveHQ"):
        rows = [cell for cell in descriptive_cells if cell.get("proposal") == proposal]
        summary = _mapping(
            descriptive_summaries[proposal], f"BlockQuant descriptive summary {proposal}"
        )
        better = sum(cell.get("descriptive_better") is True for cell in rows)
        if (
            len(rows) != 80
            or summary.get("scope") != "unmatched_source_operating_point_quality_descriptive"
            or summary.get("expected_cells") != 80
            or summary.get("complete_cells") != 80
            or summary.get("descriptively_better") != better
            or summary.get("not_better_or_incomplete") != 80 - better
            or summary.get("descriptive_better_fraction_unrounded") != better / 80
            or summary.get("threshold") is not None
            or summary.get("passed") is not None
            or summary.get("included_in_point_estimate_gate") is not False
        ):
            raise RenderError(f"BlockQuant descriptive summary failed for {proposal}")
    if gate.get("passed") is not all(summary_passes):
        raise RenderError("point-estimate global gate result does not match its summaries")
    return dict(gate)


def validate_aggregate(document: Mapping[str, Any]) -> dict:
    """Validate and normalize one complete coverage-checked aggregate."""
    _reject_forbidden_keys(document)
    _reject_legacy_formal_labels(document)
    _exact_keys(document, TOP_LEVEL_KEYS, "aggregate")
    if document.get("schema_version") != 1:
        raise RenderError("Aggregate schema_version must be 1")
    if document.get("formal_matrix_complete") is not True:
        raise RenderError("formal_matrix_complete must be true")
    if document.get("official_complete") is not True:
        raise RenderError("official_complete must be true")
    point_gate = _validate_point_estimate_win_gate(document.get("point_estimate_win_gate"))
    if not isinstance(document.get("created_at_utc"), str) or not document["created_at_utc"]:
        raise RenderError("Aggregate creation provenance is missing")
    commit = document.get("commit")
    if not isinstance(commit, str) or not SHA40.fullmatch(commit.lower()):
        raise RenderError("Aggregate commit must be one full 40-hex Git commit")
    matrix_status = _validate_formal_matrix_status(
        document.get("formal_matrix_status"), point_gate_passed=bool(point_gate["passed"])
    )
    formal_provenance = _validate_formal_provenance(
        document.get("formal_provenance"), document=document, top_commit=commit
    )
    method_scopes = _mapping(document.get("method_scopes"), "method_scopes")
    _exact_keys(
        method_scopes,
        {"TurboQuant-paper", "FibQuant-paper", "BlockQuant-paper"},
        "method_scopes",
    )
    turboquant_scope = _mapping(
        method_scopes["TurboQuant-paper"], "method_scopes.TurboQuant-paper"
    )
    if dict(turboquant_scope) != TURBOQUANT_METHOD_SCOPE:
        raise RenderError(
            "TurboQuant-paper scope must identify the Algorithm-1 MSE baseline "
            "and must not claim a paper Table 1 reproduction"
        )
    fibquant_scope = _mapping(method_scopes["FibQuant-paper"], "method_scopes.FibQuant-paper")
    if dict(fibquant_scope) != FIBQUANT_METHOD_SCOPE:
        raise RenderError(
            "FibQuant-paper scope must identify the independent v1 quality-only baseline "
            "and must exclude serving-speed claims"
        )
    blockquant_scope = _mapping(
        method_scopes["BlockQuant-paper"], "method_scopes.BlockQuant-paper"
    )
    if dict(blockquant_scope) != BLOCKQUANT_METHOD_SCOPE:
        raise RenderError(
            "BlockQuant-paper scope must identify the fixed-seed, non-reproduction "
            "quality-only row and must exclude serving-speed claims"
        )
    # Every validator below requires the same commit and a nonempty content-
    # hashed source ledger.  The aggregate producer admits a commit only after
    # checking tracked_dirty=false in every raw result preflight record.
    return {
        "commit": commit,
        "point_estimate_win_gate": point_gate,
        "formal_matrix_status": matrix_status,
        "formal_provenance": formal_provenance,
        "method_scopes": {
            "TurboQuant-paper": dict(turboquant_scope),
            "FibQuant-paper": dict(fibquant_scope),
            "BlockQuant-paper": dict(blockquant_scope),
        },
        "ppl": _validate_ppl(_mapping(document["ppl"], "ppl"), commit),
        "mechanism": _validate_mechanism(_mapping(document["mechanism"], "mechanism"), commit),
        "zeroshot": _validate_zeroshot(_mapping(document["zeroshot"], "zeroshot"), commit),
        "ruler": _validate_ruler(_mapping(document["ruler"], "ruler"), commit),
        "longbench": _validate_longbench(_mapping(document["longbench"], "longbench"), commit),
        "math500": _validate_math500(_mapping(document["math500"], "math500"), commit),
        "longcodeqa": _validate_longcodeqa(_mapping(document["longcodeqa"], "longcodeqa"), commit),
        "speed": _validate_speed(_mapping(document["speed"], "speed"), commit),
    }


def _tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        raise RenderError("Cannot render a non-finite number")
    return f"{value:.{digits}f}"


def _fmt_ci(ci: tuple[float, float], digits: int = 3) -> str:
    return f"[{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"


def _macro(name: str, value: str) -> str:
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise RenderError(f"Unsafe LaTeX macro name {name!r}")
    # newcommand is intentional: a stale hand-written definition must cause a
    # LaTeX error instead of silently overriding the formal aggregate.
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _table(
    caption: str, label: str, columns: str, header: str, rows: Sequence[str], *, wide: bool = True
) -> str:
    environment = "table*" if wide else "table"
    expected_cells = sum(character in "lcr" for character in columns)
    if expected_cells <= 0 or header.count("&") + 1 != expected_cells:
        raise RenderError(f"LaTeX column/header mismatch for {label}")
    for index, row in enumerate(rows):
        if row.count("&") + 1 != expected_cells:
            raise RenderError(f"LaTeX column mismatch for {label} row {index}")
    latex_linebreak = r" \\"
    return "\n".join(
        [
            f"\\begin{{{environment}}}[t]",
            "\\centering",
            "\\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{columns}}}",
            "\\toprule",
            header + latex_linebreak,
            "\\midrule",
            *[row + latex_linebreak for row in rows],
            "\\bottomrule",
            "\\end{tabular}",
            f"\\end{{{environment}}}",
        ]
    )


def _plot_bounds(values: Sequence[float], reference: float) -> tuple[float, float]:
    """Return deterministic linear plot bounds containing data and parity."""

    if not values or any(not math.isfinite(value) for value in values):
        raise RenderError("Formal summary figure received non-finite or empty data")
    lower = min(reference, *values)
    upper = max(reference, *values)
    span = upper - lower
    padding = max(span * 0.12, 0.02 if reference == 1.0 else 0.1)
    return lower - padding, upper + padding


def _plot_y(value: float, lower: float, upper: float, *, height: float = 3.1) -> float:
    if not lower < upper or not lower <= value <= upper:
        raise RenderError("Formal summary figure has invalid plot bounds")
    return height * (value - lower) / (upper - lower)


def _summary_panel(
    *,
    origin_x: float,
    title: str,
    y_label: str,
    values: Mapping[str, Sequence[float]],
    reference: float,
    digits: int,
) -> list[str]:
    """Render one dependency-free TikZ line panel from validated values."""

    if set(values) != set(PROPOSAL_METHODS):
        raise RenderError("Formal summary figure proposal set changed")
    if any(len(values[proposal]) != len(RATES) for proposal in PROPOSAL_METHODS):
        raise RenderError("Formal summary figure rate coverage changed")
    flattened = [float(value) for proposal in PROPOSAL_METHODS for value in values[proposal]]
    lower, upper = _plot_bounds(flattened, reference)
    x_positions = (0.8, 2.6, 4.4)
    colors = {"HQ": "blue!70!black", "AdaptiveHQ": "orange!90!black"}
    marks = {"HQ": "*", "AdaptiveHQ": "square*"}
    label_anchors = {"HQ": "south", "AdaptiveHQ": "north"}
    lines = [
        rf"\begin{{scope}}[shift={{({origin_x:.3f},0)}}]",
        r"\draw[->] (0.45,0) -- (4.85,0) node[below right] {average bits};",
        r"\draw[->] (0.45,0) -- (0.45,3.55);",
        rf"\node[font=\bfseries] at (2.65,4.05) {{{title}}};",
        rf"\node[rotate=90] at (-0.55,1.65) {{{y_label}}};",
        rf"\node[anchor=east] at (0.38,0) {{{_fmt(lower, digits)}}};",
        rf"\node[anchor=east] at (0.38,3.10) {{{_fmt(upper, digits)}}};",
    ]
    reference_y = _plot_y(reference, lower, upper)
    lines.extend(
        [
            rf"\draw[densely dashed,gray] (0.45,{reference_y:.4f}) -- (4.75,{reference_y:.4f});",
            rf"\node[anchor=east,text=gray] at (0.38,{reference_y:.4f}) {{{_fmt(reference, digits)}}};",
        ]
    )
    for x, rate in zip(x_positions, RATES):
        lines.append(rf"\draw ({x:.3f},0.06) -- ({x:.3f},-0.06) node[below] {{{rate}}};")
    for proposal in PROPOSAL_METHODS:
        points = [
            (x, _plot_y(float(value), lower, upper))
            for x, value in zip(x_positions, values[proposal])
        ]
        coordinates = " ".join(f"({x:.4f},{y:.4f})" for x, y in points)
        lines.append(
            rf"\draw[{colors[proposal]},thick] plot[mark={marks[proposal]}] coordinates {{{coordinates}}};"
        )
        for (x, y), value in zip(points, values[proposal]):
            lines.append(
                rf"\node[anchor={label_anchors[proposal]},text={colors[proposal]},"
                rf"font=\tiny] at ({x:.4f},{y:.4f}) {{{_fmt(float(value), digits)}}};"
            )
    lines.extend(
        [
            r"\draw[blue!70!black,thick] (0.70,3.58) -- (1.10,3.58) node[right,text=black] {HQ};",
            r"\draw[orange!90!black,thick] (2.45,3.58) -- (2.85,3.58) node[right,text=black] {Adaptive HQ};",
            r"\end{scope}",
        ]
    )
    return lines


def _formal_summary_figure(data: Mapping[str, Any]) -> str:
    """Build the coverage-checked quality figure without optional plot packages."""

    ppl_values = {
        proposal: [
            _number(data["ppl"]["primary"][proposal][rate]["ratio"], "summary PPL ratio")
            for rate in RATES
        ]
        for proposal in PROPOSAL_METHODS
    }
    zero_values = {
        proposal: [
            100.0
            * _number(
                data["zeroshot"]["primary"][proposal][rate]["delta"],
                "summary zero-shot delta",
            )
            for rate in RATES
        ]
        for proposal in PROPOSAL_METHODS
    }
    # This is deliberately a paired latency ratio, not a raw throughput average.
    # Each cell has already been reduced from technical repeats to one median per
    # independently launched process, then to a geometric process mean by the
    # formal speed aggregator.  Taking the geometric mean across the two
    # predeclared long-context model cells preserves that multiplicative
    # baseline-latency / candidate-latency interpretation.  The result remains
    # descriptive: the complete Pareto contract, rather than this panel, is the
    # only gate that can admit a global speed claim.
    speed_values = {}
    speed_comparisons = {
        "HQ": "hq_vs_turboquant",
        "AdaptiveHQ": "adaptive_hq_vs_turboquant",
    }
    for proposal, comparison_name in speed_comparisons.items():
        by_rate = []
        for rate in RATES:
            per_model = []
            for model in LONG_MODELS:
                comparison = data["speed"]["models"][model][rate]["off"][
                    "paired_speedups"
                ]["e2e.decode_steady_latency"][comparison_name]
                per_model.append(
                    _number(
                        comparison.get("geometric_mean_process_speedup_x"),
                        f"summary E2E decode speedup {proposal}/{model}/{rate}",
                        minimum=0.0,
                    )
                )
            if any(value <= 0.0 for value in per_model):
                raise RenderError(f"summary E2E decode speedup must be positive: {proposal}/{rate}")
            by_rate.append(math.exp(statistics.fmean(math.log(value) for value in per_model)))
        speed_values[proposal] = by_rate
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\begin{tikzpicture}[font=\scriptsize]",
    ]
    lines.extend(
        _summary_panel(
            origin_x=0.0,
            title=r"WikiText-2 quality",
            y_label=r"GM PPL ratio (proposal/TQ)",
            values=ppl_values,
            reference=1.0,
            digits=3,
        )
    )
    lines.extend(
        _summary_panel(
            origin_x=5.5,
            title=r"Five-task zero-shot quality",
            y_label=r"ZS-Mean $\Delta$ (percentage points)",
            values=zero_values,
            reference=0.0,
            digits=2,
        )
    )
    lines.extend(
        _summary_panel(
            origin_x=11.0,
            title=r"Paired E2E steady decode",
            y_label=r"speedup $=$ TQ latency / proposal latency",
            values=speed_values,
            reference=1.0,
            digits=2,
        )
    )
    lines.extend(
        [
            r"\end{tikzpicture}",
            r"\caption{\textbf{Coverage-checked fixed-panel summary.} Left: geometric mean of within-checkpoint proposal/TurboQuant Algorithm-1-MSE WikiText-2 PPL ratios; lower is better and the dashed line is parity. Middle: proposal-minus-TurboQuant ZS-Mean in percentage points; higher is better and the dashed line is parity. Right: geometric mean, over the two predeclared long-context model cells, of independently paired research fake-quant E2E steady-decode speedups (TurboQuant latency divided by proposal latency); higher is faster and the dashed line is parity. The right panel is descriptive and is neither packed KV-cache serving nor a hardware-native throughput claim. Points are descriptive estimates for fixed, family-correlated checkpoint panels. Exact per-checkpoint values, process-level speed ratios, win/tie/loss counts, and simultaneous family-wise intervals are retained in the coverage-checked audit ledger. The figure is generated from the same complete aggregate and makes no model-population claim.}",
            r"\label{fig:formal-quality-summary}",
            r"\end{figure*}",
        ]
    )
    return "\n".join(lines)


def build_main_results(data: Mapping[str, Any], aggregate_sha256: str) -> str:
    """Build compact main-paper figures/tables from the validated aggregate."""

    quality_rows = []
    for proposal in PROPOSAL_METHODS:
        for rate in RATES:
            ppl = data["ppl"]["primary"][proposal][rate]
            zero = data["zeroshot"]["primary"][proposal][rate]
            quality_rows.append(
                f"{proposal} & {rate} & {_fmt(ppl['ratio'], 4)} & "
                f"{_fmt(ppl['relative_percent'], 2)}\\% & "
                f"{ppl['wins']}/{ppl['ties']}/{ppl['losses']} & "
                f"{_fmt(100.0 * zero['delta'], 3)} & "
                f"{zero['wins']}/{zero['ties']}/{zero['losses']}"
            )

    per_model_rows = []
    for model in PAPER_MODELS:
        ppl_rates = data["ppl"]["models"][model]["rates"]
        zero_rates = data["zeroshot"]["models"][model]["rates"]
        for rate in RATES:
            ppl_methods = ppl_rates[rate]
            zero_methods = zero_rates[rate]
            per_model_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & {rate} & "
                f"{_fmt(_number(ppl_methods['TurboQuant-paper']['ppl'], 'main TQ PPL'), 2)} & "
                f"{_fmt(_number(ppl_methods['HQ']['ppl'], 'main HQ PPL'), 2)} & "
                f"{_fmt(_number(ppl_methods['AdaptiveHQ']['ppl'], 'main AdaptiveHQ PPL'), 2)} & "
                f"{_fmt(100.0 * _number(zero_methods['TurboQuant-paper']['avg_primary_score'], 'main TQ ZS'), 2)} & "
                f"{_fmt(100.0 * _number(zero_methods['HQ']['avg_primary_score'], 'main HQ ZS'), 2)} & "
                f"{_fmt(100.0 * _number(zero_methods['AdaptiveHQ']['avg_primary_score'], 'main AdaptiveHQ ZS'), 2)}"
            )
    for rate in RATES:
        ppl_values = {
            method: [
                _number(data["ppl"]["models"][model]["rates"][rate][method]["ppl"], "main aggregate PPL")
                for model in PAPER_MODELS
            ]
            for method in ("TurboQuant-paper", "HQ", "AdaptiveHQ")
        }
        zero_values = {
            method: [
                _number(
                    data["zeroshot"]["models"][model]["rates"][rate][method]["avg_primary_score"],
                    "main aggregate ZS",
                )
                for model in PAPER_MODELS
            ]
            for method in ("TurboQuant-paper", "HQ", "AdaptiveHQ")
        }
        per_model_rows.append(
            f"Aggregate & {rate} & "
            + " & ".join(
                _fmt(math.exp(math.fsum(math.log(value) for value in ppl_values[method]) / len(PAPER_MODELS)), 2)
                for method in ("TurboQuant-paper", "HQ", "AdaptiveHQ")
            )
            + " & "
            + " & ".join(
                _fmt(100.0 * math.fsum(zero_values[method]) / len(PAPER_MODELS), 2)
                for method in ("TurboQuant-paper", "HQ", "AdaptiveHQ")
            )
        )

    gate_summaries = data["point_estimate_win_gate"]["proposal_baseline_summaries"]
    gate_rows = []
    for proposal in PROPOSAL_METHODS:
        tq_quality = gate_summaries[proposal]["TurboQuant-paper"]
        fib_quality = gate_summaries[proposal]["FibQuant-paper"]
        tq_speed = gate_summaries[proposal]["TurboQuant-paper-speed"]
        passed = tq_quality["passed"] and fib_quality["passed"] and tq_speed["passed"]
        gate_rows.append(
            f"{proposal} & {tq_quality['wins']}/{tq_quality['expected_cells']} & "
            f"{fib_quality['wins']}/{fib_quality['expected_cells']} & "
            f"{tq_speed['wins']}/{tq_speed['expected_cells']} & "
            f"{'pass' if passed else 'fail'}"
        )

    pieces = [
        "% AUTO-GENERATED. DO NOT EDIT.",
        "% Compact main-paper results from coverage_checked_aggregates.json only.",
        f"% Aggregate SHA-256: {aggregate_sha256}",
        _macro("HayateFormalMainCommitHex", data["commit"]),
        _macro("HayateFormalMainAggregateSHAHex", aggregate_sha256),
        r"\HayateValidateFormalMainIdentity",
        "",
        _formal_summary_figure(data),
        "",
        _table(
            r"Seven-model quality grid in the main-paper layout. PPL is WikiText-2 "
            r"perplexity and ZS is the unweighted mean of the five declared task "
            r"metrics, both shown as point estimates. The Aggregate PPL row is a "
            r"geometric mean and the Aggregate ZS row is an arithmetic mean. HQ is "
            r"the fixed recipe and Adaptive HQ is the causal allocation option. "
            r"FibQuant and BlockQuant, including their protocol qualifications, are "
            r"retained in the audit ledger rather than crowded into the main grid.",
            "tab:formal-main-seven-model-quality",
            "lcrrrrrr",
            r"model & bits & TQ PPL & HQ PPL & Adaptive HQ PPL & TQ ZS & HQ ZS & Adaptive HQ ZS",
            per_model_rows,
        ),
        "",
        _table(
            r"Coverage-checked fixed-panel quality summary. PPL is the geometric "
            r"mean of within-checkpoint proposal/TQ Algorithm-1-MSE ratios; lower "
            r"is better. ZS-Mean is the proposal-minus-TQ paired delta in percentage "
            r"points; higher is better. W/T/L uses the seven fixed checkpoints and "
            r"unrounded values. Simultaneous intervals and all per-checkpoint values "
            r"are retained in the coverage-checked audit ledger.",
            "tab:formal-main-quality",
            "lcccccc",
            r"proposal & bits & PPL ratio & PPL change & PPL W/T/L & ZS $\Delta$ pp & ZS W/T/L",
            quality_rows,
        ),
        "",
        _table(
            r"Predeclared strict point-estimate gate. Each entry is wins/required "
            r"matched cells before display rounding; ties, losses, missing cells, and "
            r"protocol mismatches are non-wins. Pass requires at least 90\%. "
            r"BlockQuant is retained in the audit ledger but excluded here because "
            r"its identity-bit-counted source operating point is not payload-matched. "
            r"The speed column compares only like-labelled HQ or Adaptive-HQ and TQ "
            r"implementation categories; it is not a packed-serving claim.",
            "tab:formal-main-gates",
            "lcccc",
            r"proposal & quality vs TQ & quality vs FibQuant & speed vs TQ & all three",
            gate_rows,
        ),
    ]
    return "\n\n".join(pieces) + "\n"


def build_macros(data: Mapping[str, Any], aggregate_sha256: str) -> str:
    turboquant_scope = data["method_scopes"]["TurboQuant-paper"]
    fibquant_scope = data["method_scopes"]["FibQuant-paper"]
    blockquant_scope = data["method_scopes"]["BlockQuant-paper"]
    point_gate = data["point_estimate_win_gate"]
    point_gate_passed = bool(point_gate["passed"])
    gate_summaries = point_gate["proposal_baseline_summaries"]
    block_descriptive = point_gate["unmatched_descriptive_comparisons"]["proposal_summaries"]
    quality_gate_rows = [
        gate_summaries[proposal][baseline]
        for proposal in PROPOSAL_METHODS
        for baseline in ("TurboQuant-paper", "FibQuant-paper")
    ]
    speed_gate_rows = [
        gate_summaries[proposal]["TurboQuant-paper-speed"] for proposal in PROPOSAL_METHODS
    ]
    quality_passes = sum(bool(row["passed"]) for row in quality_gate_rows)
    speed_passes = sum(bool(row["passed"]) for row in speed_gate_rows)
    total_gate_passes = quality_passes + speed_passes
    total_gate_count = len(quality_gate_rows) + len(speed_gate_rows)
    quality_summary = (
        f"The complete quality matrix passed {quality_passes} of "
        f"{len(quality_gate_rows)} matched-operating-point, predeclared 90% strict "
        "point-estimate gates; "
        + (
            "directional quality superiority is allowed by this gate."
            if quality_passes == len(quality_gate_rows)
            else "directional quality superiority is not claimed."
        )
    )
    complete_speed_established = bool(data["speed"]["pareto"]["established"])
    speed_summary = (
        f"The complete fair-speed matrix passed {speed_passes} of "
        f"{len(speed_gate_rows)} predeclared 90% strict point-estimate gates; "
        + (
            "complete speed and measured-memory superiority is established."
            if complete_speed_established
            else "complete speed and measured-memory superiority is not established."
        )
    )
    intro_summary = quality_summary + " " + speed_summary
    lines = [
        "% AUTO-GENERATED. DO NOT EDIT.",
        "% Source: coverage_checked_aggregates.json content only.",
        f"% Aggregate SHA-256: {aggregate_sha256}",
        _macro("HayateFormalCommitHex", data["commit"]),
        _macro("HayateFormalAggregateSHAHex", aggregate_sha256),
        _macro("HayateFormalCommit", r"\texttt{" + data["commit"] + "}"),
        _macro("HayateFormalAggregateSHA", r"\texttt{" + aggregate_sha256 + "}"),
        _macro("HayateCodebookManifestSHA", HQ_CODEBOOK_MANIFEST_SHA256),
        _macro("HayateAbstractQualitySummary", _tex(quality_summary)),
        _macro("HayateAbstractSpeedSummary", _tex(speed_summary)),
        _macro("HayateIntroResultSummary", _tex(intro_summary)),
        _macro("HayatePointEstimateWinGatePassed", str(point_gate_passed).lower()),
        _macro("HayateDirectionalSuperiorityClaimAllowed", str(point_gate_passed).lower()),
        _macro(
            "HayatePointEstimateWinGateStatement",
            _tex(
                f"the predeclared 90% strict point-estimate gate passed all "
                f"{total_gate_count} matched-operating-point proposal/baseline scopes"
                if point_gate_passed
                else (
                    f"the predeclared 90% strict point-estimate gate passed "
                    f"{total_gate_passes} of {total_gate_count} matched-operating-point "
                    "proposal/baseline scopes; "
                    "directional superiority claims are suppressed"
                )
            ),
        ),
        _macro("HayateTurboQuantBaselineName", _tex(turboquant_scope["display_name"])),
        _macro("HayateTurboQuantBaselineScope", _tex(turboquant_scope["scope_note"])),
        _macro("HayateFibQuantBaselineName", _tex(fibquant_scope["display_name"])),
        _macro("HayateFibQuantBaselineScope", _tex(fibquant_scope["scope_note"])),
        _macro(
            "HayateFibQuantSpeedComparison",
            _tex(str(fibquant_scope["speed_comparison"]).lower()),
        ),
        _macro("HayateBlockQuantBaselineName", _tex(blockquant_scope["display_name"])),
        _macro("HayateBlockQuantBaselineScope", _tex(blockquant_scope["scope_note"])),
        _macro(
            "HayateBlockQuantSpeedComparison",
            _tex(str(blockquant_scope["speed_comparison"]).lower()),
        ),
        _macro(
            "HayateBlockQuantPaperReproduction",
            _tex(str(blockquant_scope["paper_reproduction"]).lower()),
        ),
        _macro(
            "HayateTurboQuantPaperTableOneReproduction",
            _tex(str(turboquant_scope["paper_table1_reproduction"]).lower()),
        ),
        _macro(
            "HayateQualityInferenceScope",
            _tex(
                "fixed preselected family-correlated checkpoint panel; paired examples "
                "within fixed tasks/checkpoints; no random model/task population claim"
            ),
        ),
        _macro(
            "HayateQualityMultiplicityScope",
            _tex(
                "point estimates and checkpoint win counts are primary; simultaneous "
                "95% Bonferroni family-wise intervals over three predeclared average-bit "
                "rates are a panel-level robustness supplement"
            ),
        ),
        _macro(
            "HayatePublicPayloadScope",
            _tex(
                "materialized logical payload with unpacked int64 index tensors; "
                "stored_bits counts only fixed-width KV data-plane fields; self-describing "
                "control tags, Python containers, and persistent codec state are uncharged; "
                "it is not physical tensor memory"
            ),
        ),
    ]
    proposal_infix = {"HQ": "HQ", "AdaptiveHQ": "AdaptiveHQ"}
    baseline_infix = {
        "TurboQuant-paper": "TurboQuant",
        "FibQuant-paper": "FibQuant",
        "BlockQuant-paper": "BlockQuant",
        "TurboQuant-paper-speed": "TurboQuantSpeed",
    }
    for proposal in PROPOSAL_METHODS:
        for baseline in (
            "TurboQuant-paper",
            "FibQuant-paper",
            "BlockQuant-paper",
            "TurboQuant-paper-speed",
        ):
            if baseline == "BlockQuant-paper":
                descriptive_summary = block_descriptive[proposal]
                wins = descriptive_summary["descriptively_better"]
                cell_count = descriptive_summary["expected_cells"]
                fraction = descriptive_summary["descriptive_better_fraction_unrounded"]
                passed = "not-applicable-unmatched-operating-point"
                included = "false"
            else:
                summary = gate_summaries[proposal][baseline]
                wins = summary["wins"]
                cell_count = summary["expected_cells"]
                fraction = summary["win_fraction_unrounded"]
                passed = str(bool(summary["passed"])).lower()
                included = "true"
            stem = "Hayate" + proposal_infix[proposal] + baseline_infix[baseline] + "Gate"
            lines.extend(
                [
                    _macro(stem + "Wins", str(wins)),
                    _macro(stem + "Cells", str(cell_count)),
                    _macro(
                        stem + "WinFraction",
                        _fmt(float(fraction), 6),
                    ),
                    _macro(stem + "Passed", passed),
                    _macro(stem + "IncludedInGate", included),
                ]
            )
    for rate in RATES:
        word = RATE_WORD[rate]
        for proposal, macro_infix in (("HQ", ""), ("AdaptiveHQ", "Adaptive")):
            ppl = data["ppl"]["primary"][proposal][rate]
            lines.extend(
                [
                    _macro(f"Hayate{macro_infix}PPLRatio{word}Bit", _fmt(ppl["ratio"], 4)),
                    _macro(f"Hayate{macro_infix}PPLRatioCILow{word}Bit", _fmt(ppl["ci"][0], 4)),
                    _macro(f"Hayate{macro_infix}PPLRatioCIHigh{word}Bit", _fmt(ppl["ci"][1], 4)),
                    _macro(f"Hayate{macro_infix}PPLWins{word}Bit", str(ppl["wins"])),
                    _macro(f"Hayate{macro_infix}PPLTies{word}Bit", str(ppl["ties"])),
                    _macro(f"Hayate{macro_infix}PPLLosses{word}Bit", str(ppl["losses"])),
                    _macro(
                        f"Hayate{macro_infix}PPLPointStatement{word}Bit",
                        _tex(ppl["point_statement"]),
                    ),
                    _macro(
                        f"Hayate{macro_infix}PPLRobustnessStatement{word}Bit",
                        _tex(ppl["robustness_statement"]),
                    ),
                ]
            )
            zero = data["zeroshot"]["primary"][proposal][rate]
            lines.extend(
                [
                    _macro(
                        f"Hayate{macro_infix}ZeroShotDeltaPP{word}Bit",
                        _fmt(100.0 * zero["delta"], 3),
                    ),
                    _macro(
                        f"Hayate{macro_infix}ZeroShotDeltaCILowPP{word}Bit",
                        _fmt(100.0 * zero["ci"][0], 3),
                    ),
                    _macro(
                        f"Hayate{macro_infix}ZeroShotDeltaCIHighPP{word}Bit",
                        _fmt(100.0 * zero["ci"][1], 3),
                    ),
                    _macro(f"Hayate{macro_infix}ZeroShotWins{word}Bit", str(zero["wins"])),
                    _macro(f"Hayate{macro_infix}ZeroShotTies{word}Bit", str(zero["ties"])),
                    _macro(f"Hayate{macro_infix}ZeroShotLosses{word}Bit", str(zero["losses"])),
                    _macro(
                        f"Hayate{macro_infix}ZeroShotPointStatement{word}Bit",
                        _tex(zero["point_statement"]),
                    ),
                    _macro(
                        f"Hayate{macro_infix}ZeroShotRobustnessStatement{word}Bit",
                        _tex(zero["robustness_statement"]),
                    ),
                ]
            )
        mechanism = data["mechanism"]["primary"][rate]
        lines.extend(
            [
                _macro(f"HayateMechanismRatio{word}Bit", _fmt(mechanism["ratio"], 4)),
                _macro(f"HayateMechanismRatioCILow{word}Bit", r"\mathrm{N/A}"),
                _macro(f"HayateMechanismRatioCIHigh{word}Bit", r"\mathrm{N/A}"),
                _macro(f"HayateMechanismClaim{word}Bit", _tex(mechanism["claim"])),
            ]
        )
    speed_claim_metrics = (
        ("ResearchEndToEnd", "e2e.decode_steady_latency"),
        (
            "PublicPayload",
            "micro.decode.payload_api.roundtrip_explicit_encode_then_decode",
        ),
    )
    for model in LONG_MODELS:
        model_word = SPEED_MODEL_MACRO_WORD[model]
        for rate in RATES:
            rate_word = RATE_WORD[rate]
            row = data["speed"]["models"][model][rate]["off"]
            for proposal, comparison_name, macro_infix in (
                ("HQ", "hq_vs_turboquant", ""),
                ("AdaptiveHQ", "adaptive_hq_vs_turboquant", "Adaptive"),
            ):
                for scope_word, metric in speed_claim_metrics:
                    comparison = row["paired_speedups"][metric][comparison_name]
                    ratio = _number(
                        comparison.get("geometric_mean_process_speedup_x"),
                        f"{proposal} speed macro ratio",
                    )
                    ci = _ci(
                        comparison.get("student_t_log_geomean_ci95"),
                        f"{proposal} speed macro CI",
                        minimum=0.0,
                    )
                    prefix = f"Hayate{macro_infix}{scope_word}Speed{model_word}{rate_word}Bit"
                    lines.extend(
                        [
                            _macro(f"{prefix}Ratio", _fmt(ratio, 3)),
                            _macro(f"{prefix}CILow", _fmt(ci[0], 3)),
                            _macro(f"{prefix}CIHigh", _fmt(ci[1], 3)),
                            _macro(f"{prefix}Claim", _tex(_speed_claim(ci, proposal))),
                        ]
                    )
    hardware = data["speed"]["hardware"]
    pareto = data["speed"]["pareto"]
    lines.extend(
        [
            _macro("HayateSpeedGPU", _tex(hardware["gpu_name"])),
            _macro("HayateSpeedDriver", _tex(hardware["driver_version"])),
            _macro(
                "HayateSpeedClaimScope",
                _tex(
                    "descriptive ratios with pointwise nominal unadjusted intervals only; "
                    "complete superiority is separately gated by one simultaneous "
                    "Bonferroni family over every primary speed and measured-memory endpoint"
                ),
            ),
            _macro("HayateGlobalSpeedClaim", _tex(pareto["claim"])),
            _macro("HayatePackedDecodeClaimAllowed", "false"),
            _macro("HayateCompleteDominanceEndpointCount", str(pareto["family_size"])),
            _macro("HayateCompleteDominanceCritical", _fmt(pareto["critical"], 6)),
            _macro(
                "HayateMeasuredMemoryScope",
                _tex(
                    "public unpacked payload tensor bytes and dequantized research-cache "
                    "peak allocated memory; neither is packed KV serving memory"
                ),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_tables(data: Mapping[str, Any], aggregate_sha256: str) -> str:
    turboquant_scope = data["method_scopes"]["TurboQuant-paper"]
    fibquant_scope = data["method_scopes"]["FibQuant-paper"]
    blockquant_scope = data["method_scopes"]["BlockQuant-paper"]
    tables = [
        "% AUTO-GENERATED. DO NOT EDIT.",
        "% Requires booktabs; values come only from the coverage-checked aggregate.",
        f"% Aggregate SHA-256: {aggregate_sha256}",
        "% TurboQuant-paper scope: " + turboquant_scope["scope_note"],
        "% FibQuant-paper scope: " + fibquant_scope["scope_note"],
        "% BlockQuant-paper scope: " + blockquant_scope["scope_note"],
        _macro("HayateFormalTablesCommitHex", data["commit"]),
        _macro("HayateFormalTablesAggregateSHAHex", aggregate_sha256),
        r"\HayateValidateFormalTableIdentity",
        "",
    ]

    ppl_primary_rows = []
    for proposal in PROPOSAL_METHODS:
        for rate in RATES:
            row = data["ppl"]["primary"][proposal][rate]
            ppl_primary_rows.append(
                f"{proposal} & {rate} & {_fmt(row['ratio'], 4)} & "
                f"{_fmt(row['relative_percent'], 2)}\\% & "
                f"{row['wins']}/{row['ties']}/{row['losses']} & "
                f"{_tex(_ppl_point_label(row['ratio']))} & {_fmt_ci(row['ci'], 4)} & "
                f"{_tex(_ppl_robustness_label(row['ci']))}"
            )
    tables.append(
        _table(
            r"Formal fixed-panel WikiText-2 primary comparison. The statistical unit "
            r"is each of seven preselected, family-correlated checkpoints, not a random "
            r"model population. The estimand is the geometric mean of within-checkpoint "
            r"HQ/TQ-Algorithm-1-MSE PPL ratios; its direction and checkpoint "
            r"win/tie/loss counts are the primary descriptive report. The simultaneous "
            r"95\% Bonferroni interval over the three predeclared rates is a robustness "
            r"supplement. A directional robustness statement is emitted only when that "
            r"interval excludes one. No population-level or all-checkpoint claim is made.",
            "tab:formal-ppl-primary",
            "lccrclcl",
            r"proposal & bits & proposal/TQ GM & relative change & W/T/L & point direction & simultaneous 95\% CI & robustness",
            ppl_primary_rows,
        )
    )

    ppl_model_rows = []
    for model in PAPER_MODELS:
        model_doc = data["ppl"]["models"][model]
        full = _number(model_doc["fullkv"].get("ppl"), f"render FullKV {model}")
        for rate in RATES:
            methods = model_doc["rates"][rate]
            tq = _number(methods["TurboQuant-paper"].get("ppl"), "render TQ PPL")
            fib = _number(methods["FibQuant-paper"].get("ppl"), "render FibQuant PPL")
            block = (
                _fmt(
                    _number(
                        methods["BlockQuant-paper"].get("ppl"),
                        "render BlockQuant PPL",
                    )
                )
                if rate == "3"
                else r"--"
            )
            hq = _number(methods["HQ"].get("ppl"), "render HQ PPL")
            adaptive = _number(methods["AdaptiveHQ"].get("ppl"), "render AdaptiveHQ PPL")
            ppl_model_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & {_fmt(full)} & {rate} & "
                f"{_fmt(tq)} & {_fmt(fib)} & {block} & {_fmt(hq)} & "
                f"{_fmt(adaptive)} & {_fmt(hq / tq, 4)} & {_fmt(adaptive / tq, 4)}"
            )
    tables.append(
        _table(
            r"Complete WikiText-2 PPL per checkpoint. Raw PPL is secondary and is "
            r"not compared across base, instruction, and reasoning checkpoints. "
            r"BlockQuant v1 is a supplementary fixed-seed same-runtime quality row "
            r"at its sole declared KV operating point; it is not the paper's five-seed "
            r"aggregate. Because its identity-bit-counted source operating point is not "
            r"payload-matched to HQ, it is displayed descriptively and excluded from "
            r"the strict dominance gate; paper-reproduction=false.",
            "tab:formal-ppl-models",
            "lrrrrrrrrr",
            rf"model & FullKV & bits & {TURBOQUANT_TABLE_LABEL} & {FIBQUANT_TABLE_LABEL} & {BLOCKQUANT_TABLE_LABEL} & HQ & AdaptiveHQ & HQ/TQ & AdaptiveHQ/TQ",
            ppl_model_rows,
        )
    )

    zero_primary_rows = []
    for proposal in PROPOSAL_METHODS:
        for rate in RATES:
            row = data["zeroshot"]["primary"][proposal][rate]
            zero_primary_rows.append(
                f"{proposal} & {rate} & {_fmt(100.0 * row['delta'], 3)} & "
                f"{row['wins']}/{row['ties']}/{row['losses']} & "
                f"{_tex(_zeroshot_point_label(row['delta']))} & "
                f"{_fmt_ci((100.0 * row['ci'][0], 100.0 * row['ci'][1]), 3)} & "
                f"{_tex(_zeroshot_robustness_label(row['ci']))}"
            )
    tables.append(
        _table(
            r"Corrected zero-shot fixed-panel primary comparison. The paired unit is "
            r"an example within each fixed task and checkpoint. Each model score is the "
            r"unweighted mean of ARC/PIQA/HellaSwag \texttt{acc\_norm} and WinoGrande "
            r"\texttt{acc}; the table reports the mean paired delta over seven "
            r"preselected, family-correlated checkpoints. The signed point estimate and "
            r"checkpoint win/tie/loss counts are primary. Simultaneous 95\% Bonferroni "
            r"intervals over the three predeclared rates are a robustness supplement; a "
            r"directional robustness statement requires the interval to exclude zero. "
            r"No random model/task population or all-task claim is made.",
            "tab:formal-zeroshot-primary",
            "lccclcl",
            r"proposal & bits & proposal--TQ (pp) & W/T/L & point direction & simultaneous 95\% CI (pp) & robustness",
            zero_primary_rows,
        )
    )

    zero_model_rows = []
    for model in PAPER_MODELS:
        rates = data["zeroshot"]["models"][model]["rates"]
        full = _number(rates["full"]["FullKV"].get("avg_primary_score"), "render Full score")
        for rate in RATES:
            rate_doc = rates[rate]
            tq = _number(rate_doc["TurboQuant-paper"].get("avg_primary_score"), "render TQ score")
            fib = _number(
                rate_doc["FibQuant-paper"].get("avg_primary_score"),
                "render FibQuant score",
            )
            block = (
                _fmt(
                    100.0
                    * _number(
                        rate_doc["BlockQuant-paper"].get("avg_primary_score"),
                        "render BlockQuant score",
                    ),
                    2,
                )
                if rate == "3"
                else r"--"
            )
            for proposal in PROPOSAL_METHODS:
                score = _number(
                    rate_doc[proposal].get("avg_primary_score"),
                    f"render {proposal} score",
                )
                paired = rate_doc["paired_proposals_vs_turboquant"][proposal]
                delta = _number(
                    paired.get("delta_macro_proposal_minus_turboquant"),
                    f"render {proposal} paired delta",
                )
                ci = _ci(
                    paired.get("stratified_paired_normal_ci95"),
                    f"render {proposal} paired CI",
                )
                zero_model_rows.append(
                    f"{_tex(DISPLAY_MODELS[model])} & {_fmt(100.0 * full, 2)} & {rate} & "
                    f"{_fmt(100.0 * tq, 2)} & {_fmt(100.0 * fib, 2)} & {block} & "
                    f"{proposal} & {_fmt(100.0 * score, 2)} & "
                    f"{_fmt(100.0 * delta, 2)} & "
                    f"{_fmt_ci((100.0 * ci[0], 100.0 * ci[1]), 2)} & "
                    f"{_tex(_zeroshot_point_statement(delta, fixed_panel=False))}"
                )
    tables.append(
        _table(
            r"Per-model corrected zero-shot average-primary score and paired HQ--TQ "
            r"delta. The pairing unit is an example within a fixed task/checkpoint; "
            r"the signed point-estimate direction is descriptive. Per-row intervals are "
            r"nominal robustness supplements and are not used for an across-rate claim. "
            r"ARC uses the released test split; PIQA, HellaSwag, and WinoGrande use "
            r"validation. BlockQuant is a supplementary fixed-seed row only at its "
            r"declared operating point. It is an unmatched descriptive comparison, "
            r"excluded from the strict dominance-gate denominator and not represented "
            r"as a matched-rate paper reproduction.",
            "tab:formal-zeroshot-models",
            "lrrrrrlrrrl",
            rf"model & FullKV & bits & {TURBOQUANT_TABLE_LABEL} & {FIBQUANT_TABLE_LABEL} & {BLOCKQUANT_TABLE_LABEL} & proposal & score & $\Delta$ pp & nominal paired 95\% CI & point-estimate direction",
            zero_model_rows,
        )
    )

    mechanism_rows = []
    for model in LONG_MODELS:
        for rate in RATES:
            methods = data["mechanism"]["models"][model]["rates"][rate]["methods"]
            scalar = methods["HayateQuant-scalar-matched"]
            block = methods["HayateQuant-blockvq-matched"]
            mechanism_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & {rate} & "
                f"{_fmt(_number(scalar.get('ppl'), 'scalar PPL'))} & "
                f"{_fmt(_number(block.get('ppl'), 'block PPL'))} & "
                f"{_fmt(_number(block.get('ppl'), 'block PPL') / _number(scalar.get('ppl'), 'scalar PPL'), 4)} & "
                f"{_fmt(_number(scalar.get('distortion_key'), 'scalar K distortion'), 5)} & "
                f"{_fmt(_number(block.get('distortion_key'), 'block K distortion'), 5)} & --"
            )
    for rate in RATES:
        primary = data["mechanism"]["primary"][rate]
        mechanism_rows.append(
            f"2-checkpoint descriptive GM & {rate} & -- & -- & "
            f"{_fmt(primary['ratio'], 4)} & -- & -- & {_tex(primary['claim'])}"
        )
    tables.append(
        _table(
            r"Exact mechanism control. Scalar and block rows use Beta+dense+RMS, "
            r"exact assignment, symmetric bits, and the same stored rate; block size "
            r"is the only codec difference. The two correlated checkpoints are reported "
            r"descriptively; no uncertainty interval or directional claim is made.",
            "tab:formal-mechanism",
            "lrrrrrrl",
            r"model & bits & scalar PPL & block PPL & block/scalar & scalar K-MSE & block K-MSE & scope",
            mechanism_rows,
        )
    )

    ruler_rows = []
    for model in LONG_MODELS:
        for context in RULER_CONTEXTS:
            methods = data["ruler"][model][context]
            ruler_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & {int(context) // 1024}K & "
                + " & ".join(
                    _fmt(_number(methods[method].get("score"), "RULER score"), 2)
                    for method in METHODS
                )
            )
    tables.append(
        _table(
            r"Pinned NeMo-Skills RULER-v1 protocol, 13 tasks and 100 tokenizer-specific "
            r"examples per task (not the canonical NVIDIA 500/task run). Values are "
            r"task-macro scores on a 0--100 scale; no directional "
            r"claim is inferred without a comparison interval.",
            "tab:formal-ruler",
            "llrrrrrr",
            rf"model & context & FullKV & {TURBOQUANT_TABLE_LABEL} & {FIBQUANT_TABLE_LABEL} & {BLOCKQUANT_TABLE_LABEL} & HQ & AdaptiveHQ",
            ruler_rows,
        )
    )

    longbench_rows = []
    for model in LONG_MODELS:
        methods = data["longbench"][model]
        longbench_rows.append(
            f"{_tex(DISPLAY_MODELS[model])} & "
            + " & ".join(
                _fmt(_number(methods[method].get("score_unrounded"), "LongBench score"), 2)
                for method in METHODS
            )
            + f" & {LONGBENCH_TOTAL}"
        )
    tables.append(
        _table(
            r"Complete 13-task LongBench-E. TQ, FibQuant, HQ, and AdaptiveHQ use the matched "
            r"3-bit average code budget; BlockQuant is the supplementary paper-declared "
            r"KV point with its versioned identity-bit-counted completion. "
            r"The released prompts, middle truncation, task generation budgets, and "
            r"task scorers are exact; these newer checkpoints use a preregistered "
            r"model-native chat-template adaptation. The displayed scalar is a derived "
            rf"unweighted task macro over {LONGBENCH_TOTAL:,} examples per method.",
            "tab:formal-longbench",
            "lrrrrrrr",
            rf"model & FullKV & {TURBOQUANT_TABLE_LABEL} & {FIBQUANT_TABLE_LABEL} & {BLOCKQUANT_TABLE_LABEL} & HQ & AdaptiveHQ & $n$",
            longbench_rows,
        )
    )

    math_rows = []
    for model in LONG_MODELS:
        for method in METHODS:
            row = data["math500"][model][method]
            estimate = 100.0 * _number(row.get("pass@1"), "MATH pass@1")
            ci = _ci(row.get("pass@1_ci95"), "MATH CI")
            math_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & "
                f"{_tex(turboquant_scope['display_name'] if method == 'TurboQuant-paper' else (fibquant_scope['display_name'] if method == 'FibQuant-paper' else (blockquant_scope['display_name'] if method == 'BlockQuant-paper' else method)))} & "
                f"{_fmt(estimate, 2)} & "
                f"{_fmt_ci((100.0 * ci[0], 100.0 * ci[1]), 2)} & 500"
            )
    tables.append(
        _table(
            r"Complete MATH-500 four-shot pass@1 using a vendored composite: the "
            r"Math-Verify ba3d3aa custom task plus the LightEval 7c1cd62 prompt-manager "
            r"selection/rendering algorithm. This is not LightEval's built-in MATH-500 "
            r"task or an unchanged executable upstream pair. Full-generation Math-Verify 0.9.0 "
            r"grading, with bare gold fragments explicitly wrapped in inline-math "
            r"delimiters (a corrected scorer adaptation, not byte-identical reference "
            r"gold serialization). Rows are descriptive; no cross-method directional "
            r"claim is emitted.",
            "tab:formal-math500",
            "llrrr",
            r"model & method & pass@1 (\%) & 95\% CI & $n$",
            math_rows,
        )
    )

    longcode_rows = []
    for model, plan in LONGCODE_PLANS.items():
        for bracket, expected_n in plan.items():
            cells = []
            for method in METHODS:
                row = data["longcodeqa"][model]["methods"][method]["brackets"][bracket]
                estimate = 100.0 * _number(row.get("accuracy"), "LongCodeQA accuracy")
                ci = _ci(row.get("accuracy_ci95"), "LongCodeQA CI")
                cells.append(f"{_fmt(estimate, 2)} {_fmt_ci((100.0 * ci[0], 100.0 * ci[1]), 2)}")
            longcode_rows.append(
                f"{_tex(DISPLAY_MODELS[model])} & {_tex(bracket)} & "
                + " & ".join(cells)
                + f" & {expected_n}"
            )
    tables.append(
        _table(
            r"LongCodeQA code-comprehension exact-match accuracy for every released "
            r"bracket that fits the unmodified checkpoint context. Cells show percent "
            r"[95\% Wilson CI]. This is only the LongCodeQA subtask, not a whole-"
            r"LongCodeBench score, and excludes LongSWE-Bench.",
            "tab:formal-longcodeqa",
            "llrrrrrrr",
            rf"model & bracket & FullKV & {TURBOQUANT_TABLE_LABEL} & {FIBQUANT_TABLE_LABEL} & {BLOCKQUANT_TABLE_LABEL} & HQ & AdaptiveHQ & $n$",
            longcode_rows,
        )
    )

    speed_e2e_rows = []
    speed_micro_rows = []
    for model in LONG_MODELS:
        for rate in RATES:
            names = {
                "full": "fullkv_uncompressed",
                "tq": "TurboQuant-paper",
                "hq": "HQ",
                "adaptive": "AdaptiveHQ",
            }
            for mode in ("off", "decode-only"):
                row = data["speed"]["models"][model][rate][mode]
                e2e_values = []
                for method in ("full", "tq", "hq", "adaptive"):
                    metric = _speed_metric(
                        row,
                        f"e2e.summary.{names[method]}.decode_steady_throughput",
                        "tokens_per_second",
                    )
                    e2e_values.append(
                        _fmt(_number(metric.get("median_of_replicate_medians"), "speed median"), 2)
                    )
                speed_e2e_rows.append(
                    f"{_tex(DISPLAY_MODELS[model])} & {rate} & {_tex(mode)} & "
                    + " & ".join(e2e_values)
                )
                micro_values = []
                for method in ("tq", "hq", "adaptive"):
                    metric = _speed_metric(
                        row,
                        "micro.regimes.decode.payload_api."
                        f"roundtrip_explicit_encode_then_decode.summary.{names[method]}",
                        "ms_per_call",
                    )
                    micro_values.append(
                        _fmt(_number(metric.get("median_of_replicate_medians"), "micro median"), 4)
                    )
                speed_micro_rows.append(
                    f"{_tex(DISPLAY_MODELS[model])} & {rate} & {_tex(mode)} & "
                    + " & ".join(micro_values)
                )
    tables.append(
        _table(
            r"End-to-end research fake-quant decode throughput, median of five "
            r"process medians on one measured B200 in a validated all-eight-GRES, "
            r"single-node allocation and one fixed GPU/software stratum. The cache immediately "
            r"dequantizes to floating tensors; these descriptive values are not "
            r"packed-cache serving results. No directional speed claim is emitted.",
            "tab:formal-speed-e2e",
            "lrrrrrr",
            rf"model & bits & compile & FullKV & {TURBOQUANT_TABLE_LABEL} ref. & HQ & AdaptiveHQ",
            speed_e2e_rows,
        )
    )
    tables.append(
        _table(
            r"Public logical-payload explicit encode--decode latency at the decode-step "
            r"shape, in ms/call, as the median of five process medians. The index "
            r"tensors are materialized as unpacked int64 arrays; stored bits counts only "
            r"fixed-width KV data-plane fields, excluding control tags, Python containers, "
            r"and persistent codec state, and is not their physical tensor footprint. Implementation "
            r"labels remain explicit; no directional speed claim is emitted.",
            "tab:formal-speed-micro",
            "lrrrrr",
            rf"model & bits & compile & {TURBOQUANT_TABLE_LABEL} ref. & HQ & AdaptiveHQ",
            speed_micro_rows,
        )
    )

    paired_specs = (
        (
            "e2e.decode_steady_latency",
            "tab:formal-speed-paired-e2e",
            r"Paired end-to-end steady-decode speedup (baseline latency divided by "
            r"candidate latency) for the research fake-quant generation path. Each "
            r"cell is the geometric mean over five fresh process medians with its 95\% "
            r"Student-$t$ interval. Intervals are pointwise nominal and unadjusted "
            r"across rows/comparisons; ratios and intervals are descriptive only. No "
            r"directional, blanket/global, or packed/native-hybrid serving claim is made.",
        ),
        (
            "micro.decode.payload_api.roundtrip_explicit_encode_then_decode",
            "tab:formal-speed-paired-micro",
            r"Paired public logical-payload encode--decode speedup at the decode-step "
            r"shape (unpacked int64 index tensors; not packed serving storage). "
            r"Each cell is the geometric mean over five fresh process medians with "
            r"its 95\% Student-$t$ interval. Intervals are pointwise nominal and "
            r"unadjusted across rows/comparisons; ratios and intervals are descriptive "
            r"only, and no directional or blanket/global speed claim is made. "
            r"This table measures the explicit public logical-payload API, not the "
            r"research fake-quant or key-only native path.",
        ),
    )
    for metric_name, label, caption in paired_specs:
        paired_rows = []
        for model in LONG_MODELS:
            for rate in RATES:
                for mode in ("off", "decode-only"):
                    comparisons = data["speed"]["models"][model][rate][mode]["paired_speedups"][
                        metric_name
                    ]
                    cells = []
                    for comparison_name in SPEED_COMPARISONS:
                        comparison = comparisons[comparison_name]
                        ratio = _number(
                            comparison.get("geometric_mean_process_speedup_x"),
                            "render speedup ratio",
                        )
                        ci = _ci(
                            comparison.get("student_t_log_geomean_ci95"),
                            "render speedup CI",
                            minimum=0.0,
                        )
                        cells.append(f"{_fmt(ratio, 3)} {_fmt_ci(ci, 3)}")
                    paired_rows.append(
                        f"{_tex(DISPLAY_MODELS[model])} & {rate} & {_tex(mode)} & "
                        + " & ".join(cells)
                    )
        tables.append(
            _table(
                caption,
                label,
                "lrrcc",
                r"model & bits & compile & HQ/TQ & AdaptiveHQ/TQ",
                paired_rows,
            )
        )

    public_operation_rows = []
    public_operations = (
        ("prefill", "encode", "encode"),
        ("prefill", "decode_prebuilt_payload", "decode-prebuilt"),
        (
            "prefill",
            "roundtrip_explicit_encode_then_decode",
            "explicit-roundtrip",
        ),
        ("decode", "encode", "encode"),
        ("decode", "decode_prebuilt_payload", "decode-prebuilt"),
        (
            "decode",
            "roundtrip_explicit_encode_then_decode",
            "explicit-roundtrip",
        ),
    )
    for model in LONG_MODELS:
        for rate in RATES:
            for mode in ("off", "decode-only"):
                row = data["speed"]["models"][model][rate][mode]
                for regime, operation, operation_label in public_operations:
                    metric = f"micro.{regime}.payload_api.{operation}"
                    comparisons = row["paired_speedups"][metric]
                    cells = []
                    for comparison_name in SPEED_COMPARISONS:
                        comparison = comparisons[comparison_name]
                        ratio = _number(
                            comparison.get("geometric_mean_process_speedup_x"),
                            "public operation speedup ratio",
                        )
                        ci = _ci(
                            comparison.get("student_t_log_geomean_ci95"),
                            "public operation speedup CI",
                            minimum=0.0,
                        )
                        cells.append(f"{_fmt(ratio, 3)} {_fmt_ci(ci, 3)}")
                    public_operation_rows.append(
                        f"{_tex(DISPLAY_MODELS[model])} & {rate} & {_tex(mode)} & "
                        f"{_tex(regime)} & {_tex(operation_label)} & " + " & ".join(cells)
                    )
    tables.append(
        _table(
            r"Every predeclared public logical-payload micro-operation, separated by "
            r"prefill and decode-step tensor shape. Cells are paired latency speedups "
            r"(baseline/candidate) with pointwise 95\% Student-$t$ intervals over five "
            r"fresh processes. The payload indices are unpacked int64 tensors; this is "
            r"not packed serving. Per-endpoint intervals remain descriptive.",
            "tab:formal-speed-public-operations",
            "lrrllcc",
            r"model & bits & compile & shape & operation & HQ/TQ & AdaptiveHQ/TQ",
            public_operation_rows,
        )
    )

    research_rows = []
    research_metrics = (
        (
            "micro.prefill.research_fake_quant.fake_quant_roundtrip",
            "fake-quant-prefill-roundtrip",
        ),
        (
            "micro.decode.research_fake_quant.fake_quant_roundtrip",
            "fake-quant-decode-roundtrip",
        ),
        ("e2e.ttft", "E2E-TTFT"),
        ("e2e.decode_steady_latency", "E2E-steady-decode"),
    )
    for model in LONG_MODELS:
        for rate in RATES:
            for mode in ("off", "decode-only"):
                row = data["speed"]["models"][model][rate][mode]
                for metric, category in research_metrics:
                    comparisons = row["paired_speedups"][metric]
                    cells = []
                    for comparison_name in SPEED_COMPARISONS:
                        comparison = comparisons[comparison_name]
                        ratio = _number(
                            comparison.get("geometric_mean_process_speedup_x"),
                            "research-path speedup ratio",
                        )
                        ci = _ci(
                            comparison.get("student_t_log_geomean_ci95"),
                            "research-path speedup CI",
                            minimum=0.0,
                        )
                        cells.append(f"{_fmt(ratio, 3)} {_fmt_ci(ci, 3)}")
                    research_rows.append(
                        f"{_tex(DISPLAY_MODELS[model])} & {rate} & {_tex(mode)} & "
                        f"{_tex(category)} & " + " & ".join(cells)
                    )
    tables.append(
        _table(
            r"Research fake-quant endpoints, kept separate from the public payload API. "
            r"The micro rows time ``fake\_quant\_roundtrip'': fixed HQ is fused, while "
            r"TurboQuant and AdaptiveHQ use inherited encode--decode. E2E rows use the "
            r"dequantized Transformers cache. Cells are paired "
            r"baseline/candidate latency ratios with pointwise 95\% intervals and are not "
            r"native packed-serving measurements.",
            "tab:formal-speed-research-fake-quant",
            "lrrlcc",
            r"model & bits & compile & research endpoint & HQ/TQ & AdaptiveHQ/TQ",
            research_rows,
        )
    )

    memory_rows = []
    for model in LONG_MODELS:
        for rate in RATES:
            row = data["speed"]["models"][model][rate]["off"]
            for proposal, comparison_name in (
                ("HQ", "hq_vs_turboquant"),
                ("AdaptiveHQ", "adaptive_hq_vs_turboquant"),
            ):
                memory = row["paired_peak_allocated_memory"][comparison_name]
                ratio = memory["ratio"]
                difference = memory["difference"]
                ratio_value = _number(
                    ratio.get("geometric_mean_process_memory_advantage_x"),
                    "peak allocated memory ratio",
                )
                ratio_ci = _ci(
                    ratio.get("student_t_log_geomean_ci95"),
                    "peak allocated memory ratio CI",
                    minimum=0.0,
                )
                difference_value = _number(
                    difference.get("mean_process_memory_reduction_mib"),
                    "peak allocated memory difference",
                )
                difference_ci = _ci(
                    difference.get("student_t_mean_difference_ci95_mib"),
                    "peak allocated memory difference CI",
                )
                footprints = row["public_payload_footprints"]
                prefill_tq = footprints["prefill"]["TurboQuant-paper"]
                prefill_proposal = footprints["prefill"][proposal]
                decode_tq = footprints["decode"]["TurboQuant-paper"]
                decode_proposal = footprints["decode"][proposal]
                logical_pair = (
                    f"{prefill_tq['logical_total_stored_bits']}/"
                    f"{prefill_proposal['logical_total_stored_bits']}"
                )
                memory_rows.append(
                    f"{_tex(DISPLAY_MODELS[model])} & {rate} & {proposal} & off & "
                    f"{_fmt(ratio_value, 4)} {_fmt_ci(ratio_ci, 4)} & "
                    f"{_fmt(difference_value, 2)} {_fmt_ci(difference_ci, 2)} & "
                    f"{prefill_tq['materialized_payload_tensor_bytes']} & "
                    f"{prefill_proposal['materialized_payload_tensor_bytes']} & "
                    f"{decode_tq['materialized_payload_tensor_bytes']} & "
                    f"{decode_proposal['materialized_payload_tensor_bytes']} & {logical_pair}"
                )
    tables.append(
        _table(
            r"Primary-mode memory representations. Peak is paired CUDA allocated-memory "
            r"high water for the dequantized research-cache process (resident model "
            r"included; reserved bytes excluded), shown as TQ/proposal ratio and "
            r"TQ-minus-proposal "
            r"MiB with pointwise 95\% intervals. Public columns are exact bytes of the "
            r"unpacked code-index and scale tensors only. The final TQ/proposal logical "
            r"stored-bit pair is the equal-rate data-plane count. None is packed KV "
            r"serving memory.",
            "tab:formal-speed-memory-representations",
            "lrlrccrrrrr",
            r"model & bits & proposal & compile & peak TQ/proposal & peak reduction MiB & prefill TQ B & prefill proposal B & decode TQ B & decode proposal B & logical prefill TQ/proposal bits",
            memory_rows,
        )
    )

    return "\n\n".join(tables) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render(
    aggregate: Path, output_dir: Path, *, prefix: str = "hayatequant_formal"
) -> tuple[Path, Path, Path]:
    """Validate completely, then atomically replace each deterministic include."""
    if not SAFE_PREFIX.fullmatch(prefix):
        raise RenderError("prefix must match [a-z][a-z0-9_]*")
    aggregate = aggregate.expanduser().resolve()
    document = _load_json(aggregate)
    # All validation and text construction happen before the first write.
    data = validate_aggregate(document)
    aggregate_sha = _sha256(aggregate)
    macros = build_macros(data, aggregate_sha)
    main_results = build_main_results(data, aggregate_sha)
    tables = build_tables(data, aggregate_sha)
    output_dir = output_dir.expanduser().resolve()
    macro_path = output_dir / f"{prefix}_macros.tex"
    main_path = output_dir / f"{prefix}_main.tex"
    table_path = output_dir / f"{prefix}_tables.tex"
    _atomic_text(macro_path, macros)
    _atomic_text(main_path, main_results)
    _atomic_text(table_path, tables)
    return macro_path, table_path, main_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="hayatequant_formal")
    args = parser.parse_args(argv)
    try:
        macro_path, table_path, main_path = render(
            args.aggregate, args.output_dir, prefix=args.prefix
        )
    except RenderError as exc:
        parser.error(str(exc))
    print(f"wrote {macro_path}")
    print(f"wrote {main_path}")
    print(f"wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
