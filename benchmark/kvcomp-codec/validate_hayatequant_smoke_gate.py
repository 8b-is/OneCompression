#!/usr/bin/env python3
"""Fail-closed smoke-completion and manual-approval gate for formal runs.

``approve`` validates the exact eight-job smoke plan, fingerprints every
result, and writes a marker only after the operator supplies the fixed approval
statement. ``check`` revalidates the plan and requires the marker to match the
current commit, paper-input manifest, result bytes, and native preflight files.
No formal submission path is allowed to create or repair this marker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import tempfile
from typing import Any, Mapping, Sequence

APPROVAL_STATEMENT = "I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN"
MARKER_KIND = "hayatequant_formal_smoke_approval"
MARKER_SCHEMA_VERSION = 1
SMOKE_PLAN_VERSION = 3
REPO_ROOT = Path(__file__).resolve().parents[2]
LONG_METHODS = (
    "FullKV",
    "TurboQuant-paper",
    "FibQuant-paper",
    "BlockQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
COMPRESSED_METHODS = (
    "TurboQuant-paper",
    "FibQuant-paper",
    "BlockQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
PPL_TEACHER_FORCED_PROTOCOL = "single_forward_causal_mask_token_local_cache"
BLOCKQUANT_TEACHER_FORCED_PROTOCOL = "causal_prefix_prefill_then_token_step_targets"
BLOCKQUANT_TEMPORAL_DEPENDENCY = "dynamic_outlier_channels_over_update_tokens"
ADAPTIVE_HQ_TEACHER_FORCED_PROTOCOL = "causal_prefix_prefill_then_token_step_targets"
ADAPTIVE_HQ_TEMPORAL_DEPENDENCY = "cache_local_causal_bit_allocation_statistics"
ADAPTIVE_HQ_POLICY_VERSION = "adaptive_hq_cache_causal_safe_two_pair_rd_v5"
ADAPTIVE_HQ_DISTORTION_TABLE_VERSION = (
    "source_qmc128_exact_codebook_fp16scale_safe_two_pair_update_carrier_v4"
)
ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION = "value_scale_fp16_fraction_lsb_v1"
ADAPTIVE_HQ_OBJECTIVE = "causal_raw_power_times_qmc_base_plus_one_carrier_per_update"
ADAPTIVE_HQ_CARRIER_AMORTIZATION = "signed_qmc_all_scales_delta_divided_by_current_update_vectors"
HQ_CODEBOOK_BUNDLE = "fixed_hq_b2_v1"
ADAPTIVE_HQ_CODEBOOK_BUNDLE = "adaptive_hq_b2_safe_v1"
HQ_CODEBOOK_MANIFEST_SHA256 = "df02ba5fd41fd0cc3fd2d39474fcfe7a628c15eb9ea012e8c70b1ffb3dabecde"
SPEED_METHODS = (
    "TurboQuant-paper",
    "HQ",
    "AdaptiveHQ",
)
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGCODEQA_REVISION = "989d5eff750d65a72c522e47f8f745ef2e22906b"
LONGCODEQA_PROTOCOL_COMMIT = "91db30f5144363bb809534ec7e4a84a56ea43987"
RULER_PROTOCOL_LABEL = "NeMo-Skills RULER v1 100-sample/task protocol"
RULER_SAMPLES_PER_TASK = 100
RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK = 500
RULER_SAMPLING_PROTOCOL = (
    "pinned NeMo-Skills adapter: 100 samples/task; canonical "
    "NVIDIA/RULER v1 config: 500 samples/task"
)
LONG_RUN_FLAGS = (
    "run_needle",
    "run_ruler",
    "run_official_ruler_v1",
    "run_longbench",
    "run_gsm8k",
    "run_math",
    "run_ppl",
    "run_livecodebench",
)
LONG_RECIPE_CONFIG = {
    "dtype": "bfloat16",
    "device": "cuda:0",
    "attn_implementation": "sdpa",
    "methods": list(LONG_METHODS),
    "turboquant_variants": [
        {
            "label": "TurboQuant-paper",
            "bits": 3,
            "dist": "beta",
            "rotation_mode": "dense",
        }
    ],
    "fibquant_variants": [{"label": "FibQuant-paper", "bits": 3}],
    "blockquant_variants": [{"label": "BlockQuant-paper", "bits": 3}],
    "hayatequant_variants": [
        {"label": "HQ", "bits": 3},
        {"label": "AdaptiveHQ", "bits": 3},
    ],
    "uniform_bits": [],
    "kivi_bits": [],
    "halo_variants": [],
    "codec_variants": [],
    "skip_outliers": False,
    "sink_tokens": 0,
    "recent_tokens": 0,
    "resume": False,
}
SEMANTIC_ENVIRONMENT = {
    "ONECOMP_TRITON_CODEC": "0",
    "ONECOMP_TRITON_PREFILL": "0",
    "ONECOMP_COMPILE_CODEC": "0",
    "ONECOMP_COMPILE_DECODE_ONLY": "0",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_MODE": "disabled",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}

SMOKE_PLAN = (
    {
        "key": "ppl",
        "path": "results/smoke/ppl/results_Qwen3-0.6B.json",
        "model": "Qwen3-0.6B",
        "model_key": "qwen3-0.6b-smoke",
        "kind": "ppl",
        "dataset_files": ["data/pinned_hf_datasets_manifest.json"],
        "config": {
            "methods": list(COMPRESSED_METHODS),
            "bits": [3],
            "device": "cuda:0",
            "dtype": "bfloat16",
            "extra_skip_layers": [],
            "capture_tokens": 128,
            "timing_heads": 8,
            "timing_seq": 128,
            "timing_iters": 2,
            "sample_seed": 0,
            "run_ppl": True,
            "ppl_windows": 2,
            "ppl_max_length": 512,
            "ppl_stride": 256,
            "skip_outliers": False,
            "outlier_ratio": 50.0,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "zeroshot",
        "path": "results/smoke/zeroshot/zeroshot_Qwen3-0.6B_smoke.json",
        "model": "Qwen3-0.6B",
        "model_key": "qwen3-0.6b-smoke",
        "kind": "zeroshot",
        "dataset_files": ["data/pinned_hf_datasets_manifest.json"],
        "config": {
            "bits": [3],
            "device": "cuda:0",
            "dtype": "bfloat16",
            "tasks": ["piqa"],
            "methods": list(LONG_METHODS),
            "include_fullkv": True,
            "start": 0,
            "limit": 4,
            "sample_seed": 2027,
            "sink_tokens": 0,
            "recent_tokens": 0,
            "skip_outliers": False,
            "outlier_ratio": 50.0,
            "ablation": False,
        },
    },
    {
        "key": "ruler",
        "path": "results/smoke/ruler/results_llama-3.1-8b-instruct.json",
        "model": "llama-3.1-8b-instruct",
        "model_key": "llama31-8b",
        "kind": "ruler_v1",
        "dataset_files": ["data/ruler_v1/llama31-8b/manifest.json"],
        "n": 2,
        "config": {
            "run_official_ruler_v1": True,
            "official_ruler_tasks": ["niah_single_1"],
            "official_ruler_max_context": 4096,
            "official_ruler_limit": 2,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "longbench",
        "path": "results/smoke/longbench/results_llama-3.1-8b-instruct.json",
        "model": "llama-3.1-8b-instruct",
        "model_key": "llama31-8b",
        "kind": "longbench",
        "dataset_files": ["data/pinned_hf_datasets_manifest.json"],
        "n": 1,
        "config": {
            "run_longbench": True,
            "longbench_tasks": ["passage_retrieval_en"],
            "longbench_limit": 1,
            "longbench_max_length": 4096,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "math500",
        "path": "results/smoke/math500/results_Qwen3-0.6B.json",
        "model": "Qwen3-0.6B",
        "model_key": "qwen3-0.6b-smoke",
        "kind": "math500",
        "dataset_files": ["data/pinned_hf_datasets_manifest.json"],
        "n": 2,
        "config": {
            "run_math": True,
            "math_start": 0,
            "math_limit": 2,
            "max_new_math": 256,
            "sample_seed": 2027,
            "math_grader": "math_verify",
            "math_verify_version": "0.9.0",
            "math500_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "longcode",
        "path": "results/smoke/longcodeqa/results_llama-3.1-8b-instruct.json",
        "model": "llama-3.1-8b-instruct",
        "model_key": "llama31-8b",
        "kind": "longcodeqa",
        "dataset_files": [
            "data/LongCodeQA/LQA/32K.json",
            "data/longcodeqa_context_llama31-8b.json",
        ],
        "n": 1,
        "config": {
            "run_longcodeqa": True,
            "longcodeqa_brackets": ["32K"],
            "longcodeqa_max_context": 131072,
            "longcodeqa_max_new": 16,
            "longcodeqa_start": 0,
            "longcodeqa_limit": 1,
            "skip_outliers": False,
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "speed",
        "path": "results/smoke/speed/fair_speed_llama_b3.json",
        "model": "llama-3.1-8b-instruct",
        "model_key": "llama31-8b",
        "kind": "speed",
        "dataset_files": [],
        "arguments": {
            "revision": None,
            "local_files_only": True,
            "trust_remote_code": False,
            "device": "cuda:0",
            "avg_bits": 3,
            "dtype": "float16",
            "mode": "micro",
            "batch_size": 1,
            "micro_seq_len": 512,
            "decode_seq_len": 1,
            "quality_query_len": 8,
            "micro_warmup": 2,
            "micro_repeats": 5,
            "micro_inner_loops": 1,
            "bootstrap_resamples": 100,
            "seed": 20260715,
            "e2e_prompt_len": 4096,
            "e2e_gen_len": 64,
            "e2e_warmup": 1,
            "e2e_repeats": 5,
            "attention_implementation": "sdpa",
            "compile_mode": "off",
            "fused_backend": "triton",
            "sink_tokens": 0,
            "recent_tokens": 0,
        },
    },
    {
        "key": "native-hybrid",
        "path": "results/smoke/native/hayate_native_hybrid.json",
        "model": "llama-3.1-8b-instruct",
        "model_key": "llama31-8b",
        "kind": "native",
        "dataset_files": [],
        "arguments": {
            "bits": [3, 4],
            "head_dim": 128,
            "kv_heads": 8,
            "q_heads": 32,
            "tokens": 4096,
            "key_scale_mode": "ub",
            "codebook_init": "auto",
            "bench_iters": 100,
            "input_dtype": "float16",
            "verify_optimized_store": True,
        },
    },
)


class SmokeGateError(RuntimeError):
    """The smoke plan, provenance, or explicit approval is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeGateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise SmokeGateError(f"non-finite JSON number {value!r}")


def _load_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise SmokeGateError(f"{label} is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except SmokeGateError as exc:
        raise SmokeGateError(f"{label} contains a {exc}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeGateError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SmokeGateError(f"{label} must be a JSON object: {path}")
    return value


def _expected_hayate_codebook_artifacts(bundle: str) -> dict[str, Any]:
    if bundle not in {HQ_CODEBOOK_BUNDLE, ADAPTIVE_HQ_CODEBOOK_BUNDLE}:
        raise SmokeGateError(f"Unexpected HayateQuant codebook bundle {bundle!r}")
    directory = REPO_ROOT / "onecomp/kvcomp/codec/hayate/codebooks"
    manifest_path = (directory / "manifest.json").resolve()
    if _sha256(manifest_path) != HQ_CODEBOOK_MANIFEST_SHA256:
        raise SmokeGateError("HayateQuant codebook manifest changed")
    manifest = _load_json(manifest_path, label="HayateQuant codebook manifest")
    filenames = manifest.get("required_bundles", {}).get(bundle)
    by_name = manifest.get("artifacts")
    if not isinstance(filenames, list) or not filenames or not isinstance(by_name, dict):
        raise SmokeGateError(f"Malformed HayateQuant codebook bundle {bundle!r}")
    artifacts = []
    for filename in filenames:
        spec = by_name.get(filename)
        if not isinstance(filename, str) or not isinstance(spec, dict):
            raise SmokeGateError(f"Malformed HayateQuant artifact in {bundle!r}")
        path = (directory / filename).resolve()
        if path.parent != directory.resolve() or _sha256(path) != spec.get("file_sha256"):
            raise SmokeGateError(f"HayateQuant codebook artifact changed: {filename}")
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


def _file_record(path: Path) -> dict:
    try:
        resolved = path.expanduser().resolve(strict=True)
        before = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise SmokeGateError(f"Cannot resolve smoke artifact {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SmokeGateError(f"Smoke artifact is not a regular file: {resolved}")
    digest = _sha256(resolved)
    after = resolved.stat()
    before_state = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_state = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_state != after_state:
        raise SmokeGateError(f"Smoke artifact changed while hashing: {resolved}")
    return {"path": str(resolved), "bytes": after.st_size, "sha256": digest}


def _expected_model_contract(input_manifest: Path, model_key: str) -> dict:
    document = _load_json(input_manifest, label="paper input manifest")
    models = document.get("models")
    record = models.get(model_key) if isinstance(models, dict) else None
    path = record.get("path") if isinstance(record, dict) else None
    metadata = record.get("metadata_files") if isinstance(record, dict) else None
    weights = record.get("weights") if isinstance(record, dict) else None
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(metadata, dict)
        or not isinstance(weights, dict)
        or not weights
    ):
        raise SmokeGateError(
            f"Paper input manifest has no complete model contract for smoke key {model_key!r}"
        )
    return {
        "path": str(Path(path).expanduser().resolve()),
        "metadata_files": metadata,
        "weights": weights,
    }


def _expected_model_path(input_manifest: Path, model_key: str) -> str:
    return _expected_model_contract(input_manifest, model_key)["path"]


def _expected_hf_home(input_manifest: Path) -> str:
    document = _load_json(input_manifest, label="paper input manifest")
    hf_datasets = document.get("hf_datasets")
    value = hf_datasets.get("hf_home") if isinstance(hf_datasets, dict) else None
    if not isinstance(value, str) or not value:
        raise SmokeGateError("Paper input manifest has no validated HF_HOME")
    return str(Path(value).expanduser().resolve())


def _expected_hf_packages(input_manifest: Path) -> dict:
    document = _load_json(input_manifest, label="paper input manifest")
    hf_datasets = document.get("hf_datasets")
    packages = hf_datasets.get("packages") if isinstance(hf_datasets, dict) else None
    required = ("datasets", "huggingface-hub", "pyarrow")
    if not isinstance(packages, dict) or any(
        not isinstance(packages.get(name), str) or not packages[name] for name in required
    ):
        raise SmokeGateError("Paper input manifest has no pinned HF package versions")
    return {name: packages[name] for name in required}


def _valid_full_hash_summary(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("verification_mode") == "full_sha256"
        and isinstance(value.get("file_count"), int)
        and not isinstance(value.get("file_count"), bool)
        and value["file_count"] > 0
        and isinstance(value.get("total_bytes"), int)
        and not isinstance(value.get("total_bytes"), bool)
        and value["total_bytes"] >= 0
        and isinstance(value.get("records_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["records_sha256"]) is not None
    )


def _expected_full_hash_summary(input_manifest: Path) -> dict:
    document = _load_json(input_manifest, label="paper input manifest")
    records: dict[str, dict] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"}.issubset(value):
                path = value.get("path")
                byte_count = value.get("bytes")
                digest = value.get("sha256")
                if (
                    isinstance(path, str)
                    and Path(path).is_absolute()
                    and isinstance(byte_count, int)
                    and not isinstance(byte_count, bool)
                    and byte_count >= 0
                    and isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    record = {"path": path, "bytes": byte_count, "sha256": digest}
                    if path in records and records[path] != record:
                        raise SmokeGateError(
                            f"Paper input manifest conflicts on pinned path {path}"
                        )
                    records[path] = record
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    if not records:
        raise SmokeGateError("Paper input manifest has no pinned absolute file records")
    ordered = [records[path] for path in sorted(records)]
    return {
        "verification_mode": "full_sha256",
        "file_count": len(ordered),
        "total_bytes": sum(item["bytes"] for item in ordered),
        "records_sha256": hashlib.sha256(
            json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _expect_subset(actual: Any, expected: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(actual, Mapping):
        raise SmokeGateError(f"{label} is not an object")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise SmokeGateError(f"{label} differs from exact smoke plan: {mismatches}")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_run_identity(
    document: Mapping[str, Any],
    *,
    kind: str,
    expected_commit: str,
    expected_model_path: str,
    label: str,
) -> None:
    """Recompute the benchmark's own immutable scientific-run identity."""
    config = document.get("config")
    fingerprint = document.get("run_fingerprint")
    if not isinstance(config, dict) or not isinstance(fingerprint, str):
        raise SmokeGateError(f"{label} has no reproducible scientific config")

    if kind == "zeroshot":
        if (
            config.get("code_commit") != expected_commit
            or config.get("model_path") != expected_model_path
        ):
            raise SmokeGateError(f"{label} config is not bound to the approved model/commit")
        expected = _canonical_sha256(config)
    elif kind == "speed":
        protocol = document.get("protocol")
        arguments = protocol.get("arguments") if isinstance(protocol, dict) else None
        if not isinstance(arguments, dict) or arguments.get("model") != expected_model_path:
            raise SmokeGateError(f"{label} arguments are not bound to the approved model")
        semantic = {
            key: value for key, value in arguments.items() if key not in {"output", "overwrite"}
        }
        if document.get("semantic_config") != semantic:
            raise SmokeGateError(f"{label} semantic config differs from its CLI arguments")
        if document.get("code_commit") != expected_commit:
            raise SmokeGateError(f"{label} is not bound to the approved commit")
        expected = _canonical_sha256({"semantic_config": semantic, "code_commit": expected_commit})
    else:
        if config.get("model_path") != expected_model_path:
            raise SmokeGateError(f"{label} config is not bound to the approved model")
        if document.get("code_commit") != expected_commit:
            raise SmokeGateError(f"{label} is not bound to the approved commit")
        if kind == "ppl":
            ignored = {"bits", "methods", "output_dir"}
            semantic = {key: value for key, value in config.items() if key not in ignored}
        else:
            ignored = {"methods", "output_dir", "resume"}
            semantic = {
                key: value
                for key, value in config.items()
                if key not in ignored and not key.startswith("run_")
            }
        if document.get("semantic_config") != semantic:
            raise SmokeGateError(f"{label} semantic config differs from its saved config")
        expected = _canonical_sha256({"semantic_config": semantic, "code_commit": expected_commit})
    if fingerprint != expected:
        raise SmokeGateError(f"{label} run fingerprint does not match its config/commit")


def _validate_long_recipe(config: Any, *, active_flag: str, label: str) -> None:
    _expect_subset(config, LONG_RECIPE_CONFIG, label=f"{label} common recipe")
    if not isinstance(config, Mapping):
        raise SmokeGateError(f"{label} config is not an object")
    thinkv = config.get("thinkv")
    if not isinstance(thinkv, Mapping) or thinkv.get("auto_configure") is not False:
        raise SmokeGateError(f"{label} enabled model-dependent ThinkV auto-configuration")
    for flag in LONG_RUN_FLAGS:
        expected = flag == active_flag
        if config.get(flag) is not expected:
            raise SmokeGateError(
                f"{label} must set {flag}={str(expected).lower()} for the exact smoke plan"
            )
    longcode = config.get("run_longcodeqa")
    if active_flag == "run_longcodeqa":
        if longcode is not True:
            raise SmokeGateError(f"{label} did not enable only LongCodeQA")
    elif longcode not in (None, False):
        raise SmokeGateError(f"{label} unexpectedly enabled LongCodeQA")


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _validate_preflight_record(
    record: Any,
    *,
    run_root: Path,
    expected_commit: str,
    input_manifest: Path,
    input_manifest_sha256: str,
    spec: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise SmokeGateError(f"{label} has no schema-1 worker preflight")
    code = record.get("code")
    model = record.get("model")
    environment = record.get("environment")
    host = record.get("host")
    packages = record.get("packages")
    gpu = record.get("gpu")
    gpu_devices = gpu.get("devices") if isinstance(gpu, dict) else None
    expected_model = _expected_model_contract(input_manifest, spec["model_key"])
    expected_model_path = expected_model["path"]
    expected_module = str((REPO_ROOT / "onecomp/__init__.py").resolve())
    if (
        not isinstance(code, dict)
        or code.get("commit") != expected_commit
        or code.get("tracked_dirty") is not False
        or code.get("repo") != str(REPO_ROOT)
        or code.get("onecomp_module") != expected_module
        or not isinstance(host, dict)
        or not isinstance(host.get("hostname"), str)
        or not host["hostname"]
        or not isinstance(host.get("python_executable"), str)
        or not Path(host["python_executable"]).is_absolute()
        or not isinstance(packages, dict)
        or not isinstance(model, dict)
        or model.get("key") != spec["model_key"]
        or model.get("path") != expected_model_path
        or model.get("metadata_files") != expected_model["metadata_files"]
        or model.get("weights") != expected_model["weights"]
        or model.get("weight_verification_mode") != "full_sha256"
        or not isinstance(environment, dict)
        or environment.get("KVCOMP_CODEBOOK_DIR") is not None
        or environment.get("HF_HOME") != _expected_hf_home(input_manifest)
        or environment.get("HF_DATASETS_CACHE")
        != str((Path(_expected_hf_home(input_manifest)) / "datasets").resolve())
        or environment.get("HOME") != str((run_root / "runtime/home").resolve())
        or environment.get("SLURM_JOB_NAME") != f"hq-s-{spec['key']}"
        or not isinstance(gpu, dict)
        or gpu.get("available") is not True
        or not isinstance(gpu_devices, list)
        or len(gpu_devices) != 1
        or not isinstance(gpu_devices[0], dict)
        or gpu_devices[0].get("logical_index") != 0
        or "B200" not in str(gpu_devices[0].get("name", "")).upper()
        or not _valid_full_hash_summary(record.get("pinned_input_files"))
        or record.get("pinned_input_files") != _expected_full_hash_summary(input_manifest)
    ):
        raise SmokeGateError(f"{label} has stale/dirty model, code, or GPU provenance")
    required_packages = {"torch", "transformers", "safetensors"}
    hf_package_axes = {"ppl", "zeroshot", "longbench", "math500"}
    if spec["kind"] in hf_package_axes:
        required_packages.update(("datasets", "huggingface-hub", "pyarrow"))
    if spec["kind"] in {"ppl", "ruler_v1", "longbench", "math500", "longcodeqa"}:
        required_packages.update(("hydra-core", "omegaconf", "filelock"))
    if spec["kind"] == "math500":
        required_packages.add("math-verify")
    if spec["kind"] in {"speed", "native"}:
        required_packages.add("triton")
    if spec["kind"] == "native":
        required_packages.add("vllm")
    if any(
        not isinstance(packages.get(name), str) or not packages[name] for name in required_packages
    ):
        raise SmokeGateError(f"{label} is missing required runtime package versions")
    if spec["kind"] in hf_package_axes:
        expected_hf_packages = _expected_hf_packages(input_manifest)
        if any(packages.get(name) != version for name, version in expected_hf_packages.items()):
            raise SmokeGateError(f"{label} HF runtime differs from dataset materialization")
    if spec["kind"] == "math500" and packages.get("math-verify") != "0.9.0":
        raise SmokeGateError(f"{label} does not use the pinned Math-Verify 0.9.0")
    _expect_subset(environment, SEMANTIC_ENVIRONMENT, label=f"{label} ONECOMP environment")
    manifest = model.get("input_manifest")
    expected_manifest = {
        "path": str(input_manifest),
        "bytes": input_manifest.stat().st_size,
        "sha256": input_manifest_sha256,
    }
    if manifest != expected_manifest:
        raise SmokeGateError(f"{label} used a different paper input manifest")
    dataset_files = record.get("dataset_files")
    expected_datasets = [expected_manifest]
    expected_datasets.extend(
        _file_record(run_root / relative) for relative in spec["dataset_files"]
    )
    if dataset_files != expected_datasets:
        raise SmokeGateError(f"{label} did not preflight the exact pinned dataset/protocol set")


def _validate_embedded_provenance(
    document: dict,
    *,
    run_root: Path,
    expected_commit: str,
    input_manifest: Path,
    input_manifest_sha256: str,
    spec: Mapping[str, Any],
    label: str,
) -> None:
    envelopes = document.get("run_provenance")
    if not isinstance(envelopes, list) or len(envelopes) != 1:
        raise SmokeGateError(f"{label} must contain exactly one immutable worker provenance")
    for envelope in envelopes:
        if not isinstance(envelope, dict):
            raise SmokeGateError(f"{label} has a malformed provenance envelope")
        path_value = envelope.get("path")
        digest = envelope.get("sha256")
        if not isinstance(path_value, str) or not re.fullmatch(
            r"[0-9a-f]{64}", digest if isinstance(digest, str) else ""
        ):
            raise SmokeGateError(f"{label} has an unbound provenance envelope")
        path = Path(path_value).expanduser().resolve()
        provenance_root = (run_root / "provenance").resolve()
        try:
            path.relative_to(provenance_root)
        except ValueError as exc:
            raise SmokeGateError(f"{label} provenance escaped the run root") from exc
        file_record = _file_record(path)
        if file_record["sha256"] != digest:
            raise SmokeGateError(f"{label} provenance file hash changed")
        on_disk = _load_json(path, label=f"{label} worker provenance")
        if _file_record(path) != file_record:
            raise SmokeGateError(f"{label} provenance file changed while validating")
        if on_disk != envelope.get("record"):
            raise SmokeGateError(f"{label} embedded provenance differs from its worker file")
        _validate_preflight_record(
            envelope.get("record"),
            run_root=run_root,
            expected_commit=expected_commit,
            input_manifest=input_manifest,
            input_manifest_sha256=input_manifest_sha256,
            spec=spec,
            label=label,
        )


def _unit_interval(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _validate_blockquant_identity(row: Any, *, label: str) -> None:
    if not isinstance(row, dict):
        raise SmokeGateError(f"{label} BlockQuant result is missing")
    config = row.get("codec_config")
    resolved = row.get("resolved_codec_config")
    expected_config = {
        "name": "blockquant",
        "bits": 3,
        "value_bits": 2,
        "block_size": 3,
        "key_n_outlier": 32,
        "outlier_bits": 4,
        "key_outlier_indices": None,
        "remainder_policy": "lower_dimensional_tail_v1",
        "assignment_mode": "exact-full-codebook",
        "lut": None,
        "paper_kv_recipe": True,
        "independent_baseline": True,
        "reference": "arXiv:2605.19972v1",
    }
    if (
        not isinstance(config, dict)
        or any(config.get(key) != value for key, value in expected_config.items())
        or not isinstance(resolved, dict)
        or resolved.get("codec") != "blockquant"
        or resolved.get("head_dim") != 128
        or resolved.get("paper_reproduction") is not False
        or resolved.get("config") != config
        or row.get("method_role") != "external_quality_baseline"
        or row.get("external_quality_baseline") is not True
        or row.get("proposal_method") is not False
        or row.get("paper_comparable") is not False
        or row.get("paper_reproduction") is not False
        or row.get("speed_comparison") is not False
        or "ms_per_encode" in row
        or "throughput_Melem_s" in row
    ):
        raise SmokeGateError(f"{label} BlockQuant quality-only identity changed")


def _validate_hayate_identity(row: Any, *, method: str, label: str) -> None:
    if method not in {"HQ", "AdaptiveHQ"} or not isinstance(row, dict):
        raise SmokeGateError(f"{label} formal HayateQuant result is missing")
    config = row.get("codec_config")
    resolved = row.get("resolved_codec_config")
    if not isinstance(config, dict) or not isinstance(resolved, dict):
        raise SmokeGateError(f"{label} formal HayateQuant recipe is missing")
    common = {
        "name": "hayatequant",
        "bits": 3,
        "value_bits": 3,
        "avg_bits": 3.0,
        "dist": "gaussian",
        "block_size": 2,
        "max_codebook_bits": 9,
        "rotation_mode": "auto",
        "encoder": "exact",
        "key_scale_mode": "nc",
        "value_scale_mode": "ls_iter",
        "n_outlier": 0,
        "key_error_weight": 4.0,
        "value_error_weight": 1.0,
    }
    if any(config.get(key) != value for key, value in common.items()):
        raise SmokeGateError(f"{label} formal HayateQuant common recipe changed")
    expected_bundle = HQ_CODEBOOK_BUNDLE if method == "HQ" else ADAPTIVE_HQ_CODEBOOK_BUNDLE
    if config.get("required_codebook_bundle") != expected_bundle or resolved.get(
        "codebook_artifacts"
    ) != _expected_hayate_codebook_artifacts(expected_bundle):
        raise SmokeGateError(f"{label} formal HayateQuant codebook provenance changed")
    expected_provenance = {
        "HQ": ("paper_proposal_fixed", "fixed_global", "globally_fixed_kv_code_bits"),
        "AdaptiveHQ": (
            "paper_proposal_adaptive",
            "causal_adaptive",
            "cache_local_layer_local_causal_online_stats",
        ),
    }[method]
    if (
        row.get("method_role") != expected_provenance[0]
        or row.get("proposal_method") is not True
        or row.get("proposal_family") != "HayateQuant"
        or row.get("proposal_variant") != expected_provenance[1]
        or row.get("paper_comparable") is not True
        or row.get("speed_comparison") is not True
        or row.get("allocation_scope") != expected_provenance[2]
    ):
        raise SmokeGateError(f"{label} formal HayateQuant proposal identity changed")
    if method == "HQ":
        if (
            config.get("adaptive_bit_split") is not None
            or config.get("key_code_bits_per_block") != 7
            or config.get("value_code_bits_per_block") != 5
            or resolved.get("adaptive_allocation") is not None
        ):
            raise SmokeGateError(f"{label} fixed HQ code-bit split changed")
    else:
        allocation = resolved.get("adaptive_allocation")
        if (
            config.get("adaptive_bit_split") != "online_stats"
            or config.get("key_bit_fraction") != 0.5
            or config.get("key_code_bits_per_block") is not None
            or config.get("value_code_bits_per_block") is not None
            or not isinstance(allocation, dict)
            or allocation.get("pair_code_bits_per_block") != 12
            or allocation.get("candidate_pairs") != [[6, 6], [7, 5]]
            or allocation.get("selector_bits_per_encode") != 1
            or allocation.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
            or allocation.get("distortion_table_version") != ADAPTIVE_HQ_DISTORTION_TABLE_VERSION
            or allocation.get("objective") != ADAPTIVE_HQ_OBJECTIVE
            or allocation.get("selector_carrier") != "first_value_fp16_scale_low_fraction_bits"
            or allocation.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
            or allocation.get("selector_carrier_distortion_amortization")
            != ADAPTIVE_HQ_CARRIER_AMORTIZATION
            or allocation.get("selector_incremental_stored_bits_per_encode") != 0
            or allocation.get("prior_pair") != [6, 6]
        ):
            raise SmokeGateError(f"{label} AdaptiveHQ causal carrier contract changed")


def _validate_blockquant_stored_bits(row: Mapping[str, Any], *, label: str) -> None:
    accounting = row.get("stored_bit_accounting")
    if not isinstance(accounting, dict):
        raise SmokeGateError(f"{label} BlockQuant bit decomposition is missing")
    vectors = accounting.get("vectors_per_role")
    key = accounting.get("key")
    value = accounting.get("value")
    if (
        not isinstance(vectors, int)
        or isinstance(vectors, bool)
        or vectors <= 0
        or accounting.get("head_dim") != 128
        or accounting.get("physical_packing_claim") is not False
        or not isinstance(key, dict)
        or not isinstance(value, dict)
    ):
        raise SmokeGateError(f"{label} BlockQuant bit scope changed")
    identity_fields = key.get("dynamic_channel_identity_fields")
    if (
        not isinstance(identity_fields, int)
        or isinstance(identity_fields, bool)
        or identity_fields <= 0
        or identity_fields > vectors
        or vectors % identity_fields
    ):
        raise SmokeGateError(f"{label} BlockQuant identity count changed")
    identity_bits = identity_fields * 101
    key_total = vectors * 416 + identity_bits + vectors * 32
    value_total = vectors * 256 + vectors * 16
    total = key_total + value_total
    coordinates = 2 * vectors * 128
    if (
        key.get("code_bits") != vectors * 416
        or key.get("dynamic_channel_identity_bits") != identity_bits
        or key.get("dynamic_channel_identity_coding") != "combinadic_rank_fixed_width_v1"
        or key.get("fp16_scale_bits") != vectors * 32
        or key.get("total_bits") != key_total
        or value.get("payload_bits") != vectors * 256
        or value.get("fp16_scale_bits") != vectors * 16
        or value.get("total_bits") != value_total
        or accounting.get("total_bits") != total
        or row.get("stored_bits") != total
        or row.get("stored_coordinates") != coordinates
        or not math.isclose(
            float(row.get("stored_bits_per_coord", math.nan)),
            total / coordinates,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise SmokeGateError(f"{label} BlockQuant bit arithmetic changed")


def _percentage(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 100.0


def _validate_adaptive_runtime_summary(value: Any, *, label: str) -> dict[str, Any]:
    """Validate the rate-3 two-pair runtime ledger used by every smoke axis."""

    if not isinstance(value, Mapping):
        raise SmokeGateError(f"{label} AdaptiveHQ runtime is missing")
    updates = value.get("encoded_updates")
    tokens = value.get("encoded_tokens")
    selector_bits = value.get("selector_bits_total")
    histogram = value.get("split_histogram")
    if (
        value.get("schema_version") != 1
        or value.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
        or value.get("selector_carrier_version") != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
        or value.get("selector_incremental_stored_bits_total") != 0
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 1
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < 1
        or isinstance(selector_bits, bool)
        or not isinstance(selector_bits, int)
        or selector_bits < 1
        or selector_bits != updates
        or not isinstance(histogram, Mapping)
        or not histogram
    ):
        raise SmokeGateError(f"{label} AdaptiveHQ runtime contract changed")
    checked: dict[str, dict[str, Any]] = {}
    allowed = {"6:6": [6, 6], "7:5": [7, 5]}
    for pair_key, raw in histogram.items():
        if (
            pair_key not in allowed
            or not isinstance(raw, Mapping)
            or raw.get("pair") != allowed[pair_key]
        ):
            raise SmokeGateError(f"{label} AdaptiveHQ selected an undeclared pair")
        bucket = {"pair": list(allowed[pair_key])}
        for field in ("updates", "tokens", "vectors"):
            count = raw.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise SmokeGateError(f"{label} AdaptiveHQ histogram count is malformed")
            bucket[field] = count
        if not bucket["updates"] <= bucket["tokens"] <= bucket["vectors"]:
            raise SmokeGateError(
                f"{label} AdaptiveHQ histogram must satisfy updates<=tokens<=vectors"
            )
        checked[pair_key] = bucket
    if (
        sum(row["updates"] for row in checked.values()) != updates
        or sum(row["tokens"] for row in checked.values()) != tokens
        or sum(row["vectors"] for row in checked.values()) < tokens
    ):
        raise SmokeGateError(f"{label} AdaptiveHQ histogram totals do not replay")
    return {
        "schema_version": 1,
        "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
        "encoded_updates": updates,
        "encoded_tokens": tokens,
        "selector_bits_total": selector_bits,
        "selector_incremental_stored_bits_total": 0,
        "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
        "split_histogram": checked,
    }


def _validate_speed_measurement(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or value.get("timer") != "torch.cuda.Event":
        raise SmokeGateError(f"{label} is not a CUDA-event measurement")
    raw = value.get("raw_paired_repeats")
    summaries = value.get("summary")
    if not isinstance(raw, list) or len(raw) != 5 or not isinstance(summaries, dict):
        raise SmokeGateError(f"{label} does not contain exactly five paired repeats")
    if set(summaries) != set(SPEED_METHODS):
        raise SmokeGateError(f"{label} has the wrong speed method set")
    for repeat_index, repeat in enumerate(raw):
        samples = repeat.get("samples") if isinstance(repeat, dict) else None
        if (
            not isinstance(repeat, dict)
            or repeat.get("repeat") != repeat_index
            or not isinstance(samples, list)
            or len(samples) != len(SPEED_METHODS)
            or {sample.get("method") for sample in samples if isinstance(sample, dict)}
            != set(SPEED_METHODS)
            or any(
                not isinstance(sample, dict)
                or not _finite_number(sample.get("elapsed_ms_per_call"), positive=True)
                for sample in samples
            )
        ):
            raise SmokeGateError(f"{label} has malformed paired repeat {repeat_index}")
    for method, summary in summaries.items():
        if (
            not isinstance(summary, dict)
            or summary.get("unit") != "ms_per_call"
            or not _finite_number(summary.get("median"), positive=True)
        ):
            raise SmokeGateError(f"{label} has no positive median for {method}")


def _validate_speed_payload(micro: Any, *, label: str) -> None:
    if not isinstance(micro, dict):
        raise SmokeGateError(f"{label} omitted completed micro measurements")
    method_configs = micro.get("method_configs")
    recipes = micro.get("recipe_sha256")
    quality = micro.get("quality")
    regimes = micro.get("regimes")
    if (
        any(
            not isinstance(value, dict) or set(value) != set(SPEED_METHODS)
            for value in (method_configs, recipes, quality)
        )
        or not isinstance(regimes, dict)
        or set(regimes) != {"prefill", "decode"}
    ):
        raise SmokeGateError(f"{label} has the wrong fair-speed method/regime matrix")
    for method in SPEED_METHODS:
        if recipes[method] != _canonical_sha256(method_configs[method]):
            raise SmokeGateError(f"{label} recipe digest mismatch for {method}")
        if method in {"HQ", "AdaptiveHQ"}:
            config_record = method_configs[method]
            codec_record = (
                config_record.get("codec") if isinstance(config_record, Mapping) else None
            )
            declared = codec_record.get("declared") if isinstance(codec_record, Mapping) else None
            resolved = config_record.get("resolved_at_model_head_dim")
            expected_bundle = HQ_CODEBOOK_BUNDLE if method == "HQ" else ADAPTIVE_HQ_CODEBOOK_BUNDLE
            if (
                not isinstance(declared, Mapping)
                or not isinstance(resolved, Mapping)
                or declared.get("key_error_weight") != 4.0
                or declared.get("value_error_weight") != 1.0
                or declared.get("required_codebook_bundle") != expected_bundle
                or resolved.get("codebook_artifacts")
                != _expected_hayate_codebook_artifacts(expected_bundle)
            ):
                raise SmokeGateError(
                    f"{label} scientific codebook/weight recipe changed for {method}"
                )
        row = quality[method]
        if (
            not isinstance(row, dict)
            or not _finite_number(row.get("key_mse"))
            or float(row["key_mse"]) < 0.0
            or not _finite_number(row.get("value_mse"))
            or float(row["value_mse"]) < 0.0
            or not _finite_number(row.get("attention_output_relative_l2"))
            or float(row["attention_output_relative_l2"]) < 0.0
            or not isinstance(row.get("stored_bits"), int)
            or isinstance(row.get("stored_bits"), bool)
            or row["stored_bits"] <= 0
            or row.get("recipe_sha256") != recipes[method]
        ):
            raise SmokeGateError(f"{label} has incomplete quality/accounting for {method}")
    rate = micro.get("rate_contract")
    linkage = micro.get("recipe_linkage")
    if (
        not isinstance(rate, dict)
        or rate.get("nominal_average_bits") != 3
        or rate.get("equal_input_shape") is not True
        or rate.get("equal_accounted_payload_bits") is not True
        or not isinstance(rate.get("methods"), dict)
        or set(rate["methods"]) != set(SPEED_METHODS)
        or not isinstance(linkage, dict)
        or linkage.get("payload_quality_and_all_micro_timings_use_same_codec_instances")
        is not True
    ):
        raise SmokeGateError(f"{label} failed equal-rate/recipe linkage")
    parity = micro.get("parity_checks")
    fused_parity = (
        parity.get("proposal_payload_vs_fused_quality") if isinstance(parity, dict) else None
    )
    if (
        not isinstance(parity, dict)
        or set(parity) != {"proposal_payload_vs_fused_quality"}
        or not isinstance(fused_parity, list)
        or {(row.get("method"), row.get("role")) for row in fused_parity if isinstance(row, dict)}
        != {
            ("HQ", "key"),
            ("HQ", "value"),
            ("AdaptiveHQ", "key"),
            ("AdaptiveHQ", "value"),
        }
        or any(
            not isinstance(row, dict) or row.get("stored_bits_match") is not True
            for row in fused_parity
        )
    ):
        raise SmokeGateError(f"{label} failed codec parity checks")
    proposal_contract = micro.get("proposal_contract")
    rows = proposal_contract.get("rows") if isinstance(proposal_contract, dict) else None
    if (
        not isinstance(proposal_contract, dict)
        or proposal_contract.get("formal_proposals") != ["HQ", "AdaptiveHQ"]
        or proposal_contract.get("same_total_pair_code_bit_budget") is not True
        or not isinstance(rows, dict)
        or set(rows) != {"HQ", "AdaptiveHQ"}
        or any(not isinstance(rows[name], dict) for name in ("HQ", "AdaptiveHQ"))
        or rows["HQ"].get("proposal_variant") != "fixed_global"
        or rows["HQ"].get("pair_code_bits_per_block") != 12
        or rows["HQ"].get("exact_assignment") is not True
        or rows["HQ"].get("selector") is not None
        or rows["AdaptiveHQ"].get("proposal_variant") != "causal_adaptive"
        or rows["AdaptiveHQ"].get("pair_code_bits_per_block") != 12
        or rows["AdaptiveHQ"].get("exact_assignment") is not True
        or not isinstance(rows["AdaptiveHQ"].get("selector"), dict)
        or rows["AdaptiveHQ"]["selector"].get("candidate_pairs") != [[6, 6], [7, 5]]
        or rows["AdaptiveHQ"]["selector"].get("selector_bits_per_encode") != 1
        or rows["AdaptiveHQ"]["selector"].get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
        or rows["AdaptiveHQ"]["selector"].get("distortion_table_version")
        != ADAPTIVE_HQ_DISTORTION_TABLE_VERSION
        or rows["AdaptiveHQ"]["selector"].get("objective") != ADAPTIVE_HQ_OBJECTIVE
        or rows["AdaptiveHQ"]["selector"].get("selector_carrier_version")
        != ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION
        or rows["AdaptiveHQ"]["selector"].get("selector_carrier_distortion_amortization")
        != ADAPTIVE_HQ_CARRIER_AMORTIZATION
        or rows["AdaptiveHQ"]["selector"].get("selector_incremental_stored_bits_per_encode") != 0
    ):
        raise SmokeGateError(f"{label} formal HQ/AdaptiveHQ proposal contract changed")
    for regime_name, regime in regimes.items():
        payload_api = regime.get("payload_api") if isinstance(regime, dict) else None
        fake_quant = regime.get("research_fake_quant") if isinstance(regime, dict) else None
        if (
            not isinstance(payload_api, dict)
            or set(payload_api)
            != {
                "encode",
                "decode_prebuilt_payload",
                "roundtrip_explicit_encode_then_decode",
            }
            or not isinstance(fake_quant, dict)
        ):
            raise SmokeGateError(f"{label} {regime_name} operation matrix is incomplete")
        for operation, measurement in payload_api.items():
            _validate_speed_measurement(measurement, label=f"{label} {regime_name}/{operation}")
        _validate_speed_measurement(
            fake_quant.get("fake_quant_roundtrip"),
            label=f"{label} {regime_name}/fake_quant_roundtrip",
        )


def _validate_long_axis_payload(
    axis: Any, *, kind: str, method: str, n: int, label: str
) -> list[dict[str, Any]]:
    if not isinstance(axis, dict) or axis.get("n") != n:
        raise SmokeGateError(f"{label} omitted exact {kind} output")
    accounting_records: list[tuple[Mapping[str, Any], int, int, int]] = []
    if kind == "ruler_v1":
        per_task = axis.get("per_task")
        examples = axis.get("per_example")
        task = per_task.get("niah_single_1") if isinstance(per_task, dict) else None
        rows = examples.get("niah_single_1") if isinstance(examples, dict) else None
        if (
            axis.get("schema_version") != 3
            or axis.get("benchmark") != RULER_PROTOCOL_LABEL
            or axis.get("metric") != "task-subset macro accuracy"
            or axis.get("tasks") != ["niah_single_1"]
            or axis.get("start") != 0
            or axis.get("limit_per_task") != 2
            or axis.get("samples_per_task") != RULER_SAMPLES_PER_TASK
            or axis.get("sampling_protocol") != RULER_SAMPLING_PROTOCOL
            or axis.get("protocol_complete") is not False
            or axis.get("canonical_nvidia_samples_per_task")
            != RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK
            or axis.get("canonical_nvidia_full_eval") is not False
            or axis.get("reference_model_wrapper_exact") is not False
            or not _percentage(axis.get("score"))
            or not _unit_interval(axis.get("accuracy"))
            or not isinstance(task, dict)
            or task.get("n") != 2
            or not _unit_interval(task.get("accuracy"))
            or not isinstance(rows, list)
            or len(rows) != 2
        ):
            raise SmokeGateError(f"{label} RULER smoke metric/protocol is incomplete")
        accounting_records = [
            (
                row,
                int(row.get("input_tokens", 0)),
                int(row.get("max_new_tokens", 0)),
                int(row.get("generated_tokens", 0)),
            )
            for row in rows
        ]
    elif kind == "longbench":
        rows = axis.get("per_example")
        task_rows = rows.get("passage_retrieval_en") if isinstance(rows, dict) else None
        if (
            axis.get("tasks") != ["passage_retrieval_en"]
            or axis.get("limit_per_task") != 1
            or axis.get("released_data_complete") is not False
            or axis.get("dataset_revision") != LONGBENCH_REVISION
            or axis.get("max_length") != 4096
            or not _percentage(axis.get("score"))
            or axis.get("per_task_n") != {"passage_retrieval_en": 1}
            or not isinstance(task_rows, list)
            or len(task_rows) != 1
        ):
            raise SmokeGateError(f"{label} LongBench smoke metric/protocol is incomplete")
        accounting_records = [
            (
                row,
                int(row.get("input_tokens", 0)),
                int(row.get("max_new_tokens", 0)),
                int(row.get("generated_tokens", 0)),
            )
            for row in task_rows
        ]
    elif kind == "math500":
        selection = axis.get("selection")
        per_question = axis.get("per_question")
        if (
            axis.get("task") != "math500"
            or axis.get("k") != 1
            or axis.get("grader") != "math_verify"
            or axis.get("grader_version") != "0.9.0"
            or axis.get("dataset_size") != 500
            or axis.get("complete_dataset_eval") is not False
            or not _unit_interval(axis.get("pass@1"))
            or not isinstance(axis.get("correct"), int)
            or isinstance(axis.get("correct"), bool)
            or not 0 <= axis["correct"] <= n
            or not isinstance(selection, dict)
            or selection != {"start": 0, "stop_exclusive": 2, "limit": 2}
            or not isinstance(per_question, list)
            or len(per_question) != 2
        ):
            raise SmokeGateError(f"{label} MATH-500 smoke metric/protocol is incomplete")
        for question in per_question:
            if not isinstance(question, Mapping):
                raise SmokeGateError(f"{label} MATH-500 question audit is malformed")
            samples = question.get("samples")
            if (
                not isinstance(samples, list)
                or len(samples) != 1
                or not isinstance(samples[0], Mapping)
            ):
                raise SmokeGateError(f"{label} MATH-500 sample audit is malformed")
            sample = samples[0]
            accounting_records.append(
                (
                    sample,
                    int(question.get("input_tokens", 0)),
                    256,
                    int(sample.get("generated_tokens", 0)),
                )
            )
    elif kind == "longcodeqa":
        brackets = axis.get("brackets")
        bracket = brackets.get("32K") if isinstance(brackets, dict) else None
        per_example = bracket.get("per_example") if isinstance(bracket, dict) else None
        dataset = bracket.get("dataset") if isinstance(bracket, dict) else None
        if (
            axis.get("benchmark") != "LongCodeBench/LongCodeQA"
            or axis.get("metric") != "micro_exact_match_accuracy"
            or axis.get("released_data_complete_dataset") is not False
            or not isinstance(axis.get("correct"), int)
            or isinstance(axis.get("correct"), bool)
            or not 0 <= axis["correct"] <= n
            or not _unit_interval(axis.get("accuracy"))
            or not isinstance(bracket, dict)
            or bracket.get("n") != 1
            or bracket.get("start") != 0
            or bracket.get("limit") != 1
            or bracket.get("metric") != "exact_match_accuracy"
            or bracket.get("protocol_commit") != LONGCODEQA_PROTOCOL_COMMIT
            or not isinstance(dataset, dict)
            or dataset.get("revision") != LONGCODEQA_REVISION
            or not isinstance(per_example, list)
            or len(per_example) != 1
        ):
            raise SmokeGateError(f"{label} LongCodeQA smoke metric/protocol is incomplete")
        accounting_records = [
            (
                row,
                int(row.get("input_tokens", 0)),
                16,
                int(row.get("generated_tokens", 0)),
            )
            for row in per_example
        ]
    else:  # pragma: no cover - closed by the smoke plan
        raise SmokeGateError(f"{label} unknown long-axis kind {kind!r}")

    statistics = axis.get("compression_sufficient_statistics")
    if not isinstance(statistics, Mapping) or len(accounting_records) != n:
        raise SmokeGateError(f"{label} compression sufficient statistics are missing")
    ledger = []
    if method == "FullKV":
        if (
            statistics.get("compression_aggregation") != "fullkv_no_cache"
            or statistics.get("compression_fp16_bits") is not None
            or statistics.get("compression_stored_bits") is not None
            or statistics.get("adaptive_hq_runtime") is not None
        ):
            raise SmokeGateError(f"{label} FullKV compression accounting changed")
    else:
        fp16_total = 0
        stored_total = 0
        adaptive_runtimes: list[dict[str, Any]] = []
        for index, (record, input_tokens, max_new_tokens, generated_tokens) in enumerate(
            accounting_records
        ):
            fp16_bits = record.get("compression_fp16_bits")
            stored_bits = record.get("compression_stored_bits")
            if (
                record.get("compression_aggregation") != "exact_cache_bit_counters"
                or not isinstance(fp16_bits, int)
                or isinstance(fp16_bits, bool)
                or fp16_bits <= 0
                or not isinstance(stored_bits, int)
                or isinstance(stored_bits, bool)
                or stored_bits <= 0
                or input_tokens <= 0
                or max_new_tokens <= 0
                or generated_tokens <= 0
            ):
                raise SmokeGateError(f"{label} generation {index} bit ledger is malformed")
            runtime = record.get("adaptive_hq_runtime")
            if method == "AdaptiveHQ":
                adaptive_runtimes.append(
                    _validate_adaptive_runtime_summary(
                        runtime, label=f"{label} generation {index}"
                    )
                )
            elif runtime is not None:
                raise SmokeGateError(f"{label} non-adaptive row has AdaptiveHQ provenance")
            fp16_total += fp16_bits
            stored_total += stored_bits
            ledger.append(
                {
                    "input_tokens": input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "generated_tokens": generated_tokens,
                    "compression_fp16_bits": fp16_bits,
                    "compression_stored_bits": stored_bits,
                }
            )
        if (
            statistics.get("compression_aggregation")
            != "sum_exact_per_generation_cache_bit_counters"
            or statistics.get("compression_observations") != n
            or statistics.get("compression_fp16_bits") != fp16_total
            or statistics.get("compression_stored_bits") != stored_total
        ):
            raise SmokeGateError(f"{label} compression totals do not replay")
        if method == "AdaptiveHQ":
            runtime = _validate_adaptive_runtime_summary(
                statistics.get("adaptive_hq_runtime"), label=f"{label} aggregate"
            )
            merged: dict[str, dict[str, Any]] = {}
            for source in adaptive_runtimes:
                for pair_key, bucket in source["split_histogram"].items():
                    target = merged.setdefault(
                        pair_key,
                        {"pair": list(bucket["pair"]), "updates": 0, "tokens": 0, "vectors": 0},
                    )
                    for field in ("updates", "tokens", "vectors"):
                        target[field] += bucket[field]
            expected_runtime = {
                "schema_version": 1,
                "policy_version": ADAPTIVE_HQ_POLICY_VERSION,
                "encoded_updates": sum(row["encoded_updates"] for row in adaptive_runtimes),
                "encoded_tokens": sum(row["encoded_tokens"] for row in adaptive_runtimes),
                "selector_bits_total": sum(
                    row["selector_bits_total"] for row in adaptive_runtimes
                ),
                "selector_incremental_stored_bits_total": 0,
                "selector_carrier_version": ADAPTIVE_HQ_SELECTOR_CARRIER_VERSION,
                "split_histogram": merged,
            }
            if runtime != expected_runtime:
                raise SmokeGateError(f"{label} AdaptiveHQ aggregate runtime does not replay")
    return ledger


def _validate_axis_results(
    document: dict,
    spec: Mapping[str, Any],
    *,
    run_root: Path,
    expected_commit: str,
    expected_model_path: str,
    label: str,
) -> None:
    kind = spec["kind"]
    fingerprint = document.get("run_fingerprint")
    if (
        document.get("schema_version") != 2
        or document.get("model") != spec["model"]
        or not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise SmokeGateError(f"{label} has the wrong result schema/model")
    _validate_run_identity(
        document,
        kind=kind,
        expected_commit=expected_commit,
        expected_model_path=expected_model_path,
        label=label,
    )
    if kind == "ppl":
        _expect_subset(document.get("config"), spec["config"], label=f"{label} config")
        fullkv = document.get("fullkv_ppl")
        results = document.get("results")
        rate = results.get("3") if isinstance(results, dict) else None
        if (
            not isinstance(fullkv, dict)
            or not _finite_number(fullkv.get("ppl"), positive=True)
            or fullkv.get("ppl_teacher_forced_scoring_protocol") != PPL_TEACHER_FORCED_PROTOCOL
            or fullkv.get("ppl_teacher_forced_temporal_dependency") is not None
            or not isinstance(results, dict)
            or set(results) != {"3"}
            or not isinstance(rate, dict)
            or set(rate) != set(COMPRESSED_METHODS)
        ):
            raise SmokeGateError(f"{label} omitted FullKV or 3-bit PPL results")
        for method in COMPRESSED_METHODS:
            row = rate.get(method)
            expected_protocol = (
                BLOCKQUANT_TEACHER_FORCED_PROTOCOL
                if method == "BlockQuant-paper"
                else (
                    ADAPTIVE_HQ_TEACHER_FORCED_PROTOCOL
                    if method == "AdaptiveHQ"
                    else PPL_TEACHER_FORCED_PROTOCOL
                )
            )
            expected_dependency = (
                BLOCKQUANT_TEMPORAL_DEPENDENCY
                if method == "BlockQuant-paper"
                else (ADAPTIVE_HQ_TEMPORAL_DEPENDENCY if method == "AdaptiveHQ" else None)
            )
            if (
                not isinstance(row, dict)
                or not _finite_number(row.get("ppl"), positive=True)
                or row.get("ppl_teacher_forced_scoring_protocol") != expected_protocol
                or row.get("ppl_teacher_forced_temporal_dependency") != expected_dependency
            ):
                raise SmokeGateError(f"{label} omitted {method} PPL output")
            if method == "BlockQuant-paper":
                _validate_blockquant_identity(row, label=f"{label}/{method}")
                _validate_blockquant_stored_bits(row, label=f"{label}/{method}")
            elif method in {"HQ", "AdaptiveHQ"}:
                _validate_hayate_identity(row, method=method, label=f"{label}/{method}")
        tq = rate["TurboQuant-paper"]
        expected_counters = (
            tq.get("ppl_compression_fp16_bits"),
            tq.get("ppl_compression_stored_bits"),
        )
        for method in ("FibQuant-paper", "HQ", "AdaptiveHQ"):
            row = rate[method]
            if (
                row.get("ppl_compression_fp16_bits"),
                row.get("ppl_compression_stored_bits"),
            ) != expected_counters:
                raise SmokeGateError(
                    f"{label}/{method} PPL exact counters differ from TurboQuant-paper"
                )
        return
    if kind == "zeroshot":
        _expect_subset(document.get("config"), spec["config"], label=f"{label} config")
        results = document.get("results")
        if (
            not isinstance(results, dict)
            or set(results) != {"full", "3"}
            or not isinstance(results.get("full"), dict)
            or set(results["full"]) != {"FullKV"}
            or not isinstance(results.get("3"), dict)
            or set(results["3"]) != set(LONG_METHODS)
        ):
            raise SmokeGateError(f"{label} omitted zero-shot results")
        rows = [results.get("full", {}).get("FullKV")]
        rows.extend(results.get("3", {}).get(method) for method in COMPRESSED_METHODS)
        for method, row in zip(LONG_METHODS, rows):
            task = row.get("tasks", {}).get("piqa") if isinstance(row, dict) else None
            expected_protocol = (
                BLOCKQUANT_TEACHER_FORCED_PROTOCOL
                if method == "BlockQuant-paper"
                else (
                    ADAPTIVE_HQ_TEACHER_FORCED_PROTOCOL
                    if method == "AdaptiveHQ"
                    else PPL_TEACHER_FORCED_PROTOCOL
                )
            )
            expected_dependency = (
                BLOCKQUANT_TEMPORAL_DEPENDENCY
                if method == "BlockQuant-paper"
                else (ADAPTIVE_HQ_TEMPORAL_DEPENDENCY if method == "AdaptiveHQ" else None)
            )
            if (
                not isinstance(task, dict)
                or task.get("n") != 4
                or task.get("primary_metric") != "acc_norm"
                or not _unit_interval(task.get("primary_score"))
                or not isinstance(task.get("per_example"), list)
                or len(task["per_example"]) != 4
                or task.get("teacher_forced_scoring_protocol") != expected_protocol
                or task.get("teacher_forced_temporal_dependency") != expected_dependency
            ):
                raise SmokeGateError(f"{label} omitted exact four-row piqa output for {method}")
            if method == "BlockQuant-paper":
                _validate_blockquant_identity(row, label=f"{label}/{method}")
            elif method in {"HQ", "AdaptiveHQ"}:
                _validate_hayate_identity(row, method=method, label=f"{label}/{method}")
        zero_rate = results["3"]
        tq_task = zero_rate["TurboQuant-paper"]["tasks"]["piqa"]
        expected_counters = (
            tq_task.get("compression_fp16_bits"),
            tq_task.get("compression_stored_bits"),
        )
        for method in ("FibQuant-paper", "HQ", "AdaptiveHQ"):
            task = zero_rate[method]["tasks"]["piqa"]
            if (
                task.get("compression_fp16_bits"),
                task.get("compression_stored_bits"),
            ) != expected_counters:
                raise SmokeGateError(
                    f"{label}/{method} zero-shot exact counters differ from TurboQuant-paper"
                )
        return
    if kind == "speed":
        protocol = document.get("protocol")
        arguments = protocol.get("arguments") if isinstance(protocol, dict) else None
        taxonomy = protocol.get("method_taxonomy") if isinstance(protocol, dict) else None
        if (
            document.get("schema") != {"name": "onecomp.hayatequant_fair_speed", "version": 3}
            or not isinstance(taxonomy, dict)
            or set(taxonomy) != set(SPEED_METHODS)
            or any(not isinstance(taxonomy[name], dict) for name in SPEED_METHODS)
            or taxonomy["TurboQuant-paper"].get("proposal_method") is not False
            or taxonomy["HQ"].get("proposal_variant") != "fixed_global"
            or taxonomy["AdaptiveHQ"].get("proposal_variant") != "causal_adaptive"
        ):
            raise SmokeGateError(f"{label} fair-speed formal method taxonomy changed")
        _expect_subset(arguments, spec["arguments"], label=f"{label} arguments")
        _validate_speed_payload(document.get("micro"), label=label)
        return
    config = document.get("config")
    _expect_subset(config, spec["config"], label=f"{label} config")
    active_flags = {
        "ruler_v1": "run_official_ruler_v1",
        "longbench": "run_longbench",
        "math500": "run_math",
        "longcodeqa": "run_longcodeqa",
    }
    _validate_long_recipe(config, active_flag=active_flags[kind], label=label)
    if kind == "ruler_v1":
        expected_data_root = (run_root / "data/ruler_v1/llama31-8b/4096").resolve()
        expected_manifest = (run_root / "data/ruler_v1/llama31-8b/manifest.json").resolve()
        raw_data_root = config.get("official_ruler_data_root")
        raw_manifest = config.get("official_ruler_manifest_path")
        if (
            not isinstance(raw_data_root, str)
            or Path(raw_data_root).expanduser().resolve() != expected_data_root
            or not isinstance(raw_manifest, str)
            or Path(raw_manifest).expanduser().resolve() != expected_manifest
        ):
            raise SmokeGateError(f"{label} did not execute the pinned RULER smoke data")
    if kind == "longcodeqa":
        raw_root = config.get("longcodeqa_data_root")
        if (
            not isinstance(raw_root, str)
            or Path(raw_root).expanduser().resolve() != (run_root / "data/LongCodeQA").resolve()
        ):
            raise SmokeGateError(f"{label} did not execute the pinned LongCodeQA root")
    results = document.get("results")
    if not isinstance(results, dict) or set(results) != set(LONG_METHODS):
        raise SmokeGateError(f"{label} does not contain exactly the six quality methods")
    ledgers = {}
    for method in LONG_METHODS:
        if method == "BlockQuant-paper":
            _validate_blockquant_identity(results[method], label=f"{label}/BlockQuant-paper")
        elif method in {"HQ", "AdaptiveHQ"}:
            _validate_hayate_identity(results[method], method=method, label=f"{label}/{method}")
        axis = results[method].get(kind) if isinstance(results[method], dict) else None
        ledgers[method] = _validate_long_axis_payload(
            axis,
            kind=kind,
            method=method,
            n=spec["n"],
            label=f"{label}/{method}",
        )
    tq = ledgers["TurboQuant-paper"]
    for method in ("FibQuant-paper", "HQ", "AdaptiveHQ"):
        candidate = ledgers[method]
        if len(candidate) != len(tq):
            raise SmokeGateError(f"{label}/{method} long-axis sample universe changed")
        for index, (reference, observed) in enumerate(zip(tq, candidate)):
            if (
                observed["input_tokens"] != reference["input_tokens"]
                or observed["max_new_tokens"] != reference["max_new_tokens"]
                or observed["compression_stored_bits"] * reference["compression_fp16_bits"]
                != reference["compression_stored_bits"] * observed["compression_fp16_bits"]
            ):
                raise SmokeGateError(
                    f"{label}/{method} long-axis exact matched-rate audit failed at {index}"
                )


def _validate_native(
    document: dict,
    spec: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_output: Path,
    label: str,
) -> None:
    if (
        document.get("schema")
        != {"name": "onecomp.hayatequant_vllm_native_hybrid_smoke", "version": 1}
        or document.get("ok") is not True
    ):
        raise SmokeGateError(f"{label} is not a successful native hybrid smoke")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise SmokeGateError(f"{label} omitted native provenance")
    if provenance.get("git_commit") != expected_commit or provenance.get(
        "git_status_porcelain"
    ) not in (None, ""):
        raise SmokeGateError(f"{label} was not produced from the clean approved commit")
    _expect_subset(provenance.get("arguments"), spec["arguments"], label=f"{label} arguments")
    arguments = provenance.get("arguments")
    if not isinstance(arguments, dict) or arguments.get("output") != str(expected_output):
        raise SmokeGateError(f"{label} was not written to the exact smoke output path")
    contract = document.get("implementation_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("native_hybrid_is_full_hayatequant") is not False
        or contract.get("eligible_for_full_hayatequant_serving_claim") is not False
        or contract.get("canonical_frontier_k_r_plus_1_v_r_minus_1") is not False
    ):
        raise SmokeGateError(f"{label} has a misleading native-hybrid contract")
    cases = document.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 2
        or [case.get("bits") for case in cases if isinstance(case, dict)] != [3, 4]
    ):
        raise SmokeGateError(f"{label} omitted exact native 3/4-bit cases")
    for case, bits in zip(cases, (3, 4)):
        if (
            not isinstance(case, dict)
            or case.get("bits") != bits
            or case.get("key_bits") != bits
            or case.get("native_hybrid_is_full_hayatequant") is not False
            or case.get("canonical_hayatequant_frontier_recipe") is not False
            or case.get("optimized_store_cache_bit_identical") is not True
            or case.get("optimized_store_decode_bit_identical") is not True
            or case.get("bench_iters") != spec["arguments"]["bench_iters"]
            or not _finite_number(case.get("store_ms"), positive=True)
            or not _finite_number(case.get("decode_ms"), positive=True)
            or not _finite_number(case.get("output_abs_mean"))
            or not _finite_number(case.get("output_abs_max"))
        ):
            raise SmokeGateError(f"{label} native {bits}-bit case is incomplete")


def _validate_native_preflights(
    run_root: Path,
    *,
    native_document: Mapping[str, Any],
    spec: Mapping[str, Any],
    expected_commit: str,
    input_manifest: Path,
    input_manifest_sha256: str,
) -> list[dict]:
    native_provenance = native_document.get("provenance")
    if not isinstance(native_provenance, dict):
        raise SmokeGateError("Native smoke omitted internal provenance")
    native_host = native_provenance.get("hostname")
    finished_at = native_provenance.get("finished_at_utc")
    if not isinstance(native_host, str) or not native_host or not isinstance(finished_at, str):
        raise SmokeGateError("Native smoke has no host/time identity")
    try:
        finished = datetime.fromisoformat(finished_at)
    except ValueError as exc:
        raise SmokeGateError("Native smoke completion time is not ISO-8601") from exc
    if finished.tzinfo is None:
        raise SmokeGateError("Native smoke completion time has no timezone")
    candidates: list[tuple[datetime, dict]] = []
    for path in sorted((run_root / "provenance").glob("smoke-native-hybrid_*.json")):
        initial_record = _file_record(path)
        document = _load_json(path, label="native smoke preflight")
        if _file_record(path) != initial_record:
            raise SmokeGateError("Native smoke preflight changed while validating")
        try:
            _validate_preflight_record(
                document,
                run_root=run_root,
                expected_commit=expected_commit,
                input_manifest=input_manifest,
                input_manifest_sha256=input_manifest_sha256,
                spec=spec,
                label="native smoke preflight",
            )
        except SmokeGateError:
            continue
        host = document.get("host")
        environment = document.get("environment")
        created_at = document.get("created_at_utc")
        if (
            not isinstance(host, dict)
            or host.get("hostname") != native_host
            or not isinstance(environment, dict)
            or environment.get("SLURM_JOB_NAME") != "hq-s-native-hybrid"
            or not isinstance(created_at, str)
        ):
            continue
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            continue
        if created.tzinfo is None:
            continue
        elapsed = (finished - created).total_seconds()
        if not (0.0 <= elapsed <= 6 * 60 * 60):
            continue
        candidates.append((created, initial_record))
    if not candidates:
        raise SmokeGateError(
            "No native worker preflight matches this result's commit, host, and time"
        )
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[-2][0] == candidates[-1][0]:
        raise SmokeGateError(
            "Expected exactly one native worker preflight at the latest timestamp; "
            "found an ambiguous tie"
        )
    # Failed attempts may leave older preflights without a result.  The latest
    # matching preflight before completion is the only one that can identify
    # the final output, and is the only file bound into the approval marker.
    return [candidates[-1][1]]


def validate_smoke_plan(
    *,
    run_root: Path,
    expected_commit: str,
    input_manifest: Path,
    input_manifest_sha256: str,
) -> dict:
    run_root = run_root.expanduser().resolve()
    input_manifest = input_manifest.expanduser().resolve()
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise SmokeGateError("Expected commit must be a lowercase 40-hex git commit")
    if re.fullmatch(r"[0-9a-f]{64}", input_manifest_sha256) is None:
        raise SmokeGateError("Input manifest SHA-256 must be lowercase 64-hex")
    input_manifest_record = _file_record(input_manifest)
    if input_manifest_record["sha256"] != input_manifest_sha256:
        raise SmokeGateError("Current paper input manifest is missing or has the wrong SHA-256")
    input_document = _load_json(input_manifest, label="paper input manifest")
    if (
        input_document.get("schema_version") != 2
        or input_document.get("kind") != "hayatequant_paper_input_validation"
        or input_document.get("status") != "ok"
    ):
        raise SmokeGateError("Paper input manifest is not a successful schema-2 validation")
    if _file_record(input_manifest) != input_manifest_record:
        raise SmokeGateError("Paper input manifest changed while starting smoke validation")
    result_records = []
    native_document: dict | None = None
    native_spec: Mapping[str, Any] | None = None
    for spec in SMOKE_PLAN:
        path = (run_root / spec["path"]).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise SmokeGateError(f"Smoke result {spec['key']} escaped the run root") from exc
        result_record = _file_record(path)
        document = _load_json(path, label=f"smoke result {spec['key']}")
        if _file_record(path) != result_record:
            raise SmokeGateError(f"Smoke result {spec['key']} changed while loading")
        if spec["kind"] == "native":
            _validate_native(
                document,
                spec,
                expected_commit=expected_commit,
                expected_output=path,
                label=spec["key"],
            )
            native_document = document
            native_spec = spec
        else:
            _validate_embedded_provenance(
                document,
                run_root=run_root,
                expected_commit=expected_commit,
                input_manifest=input_manifest,
                input_manifest_sha256=input_manifest_sha256,
                spec=spec,
                label=spec["key"],
            )
            _validate_axis_results(
                document,
                spec,
                run_root=run_root,
                expected_commit=expected_commit,
                expected_model_path=_expected_model_path(input_manifest, spec["model_key"]),
                label=spec["key"],
            )
        if _file_record(path) != result_record:
            raise SmokeGateError(f"Smoke result {spec['key']} changed while validating")
        result_records.append({"key": spec["key"], **result_record})
    if native_document is None or native_spec is None:
        raise SmokeGateError("Exact smoke plan omitted the native hybrid job")
    native_preflights = _validate_native_preflights(
        run_root,
        native_document=native_document,
        spec=native_spec,
        expected_commit=expected_commit,
        input_manifest=input_manifest,
        input_manifest_sha256=input_manifest_sha256,
    )
    if _file_record(input_manifest) != input_manifest_record:
        raise SmokeGateError("Paper input manifest changed during smoke validation")
    plan_sha256 = hashlib.sha256(
        json.dumps(SMOKE_PLAN, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "smoke_plan_version": SMOKE_PLAN_VERSION,
        "smoke_plan_sha256": plan_sha256,
        "code_commit": expected_commit,
        "input_manifest": {
            "path": str(input_manifest),
            "bytes": input_manifest_record["bytes"],
            "sha256": input_manifest_sha256,
        },
        "results": result_records,
        "native_worker_preflights": native_preflights,
        "completed_job_keys": [spec["key"] for spec in SMOKE_PLAN],
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def approve(marker: Path, *, validated_plan: dict, approval_statement: str) -> dict:
    if approval_statement != APPROVAL_STATEMENT:
        raise SmokeGateError(
            "Explicit approval statement is missing; smoke validation alone cannot approve formal runs"
        )
    document = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "kind": MARKER_KIND,
        "approval_statement": APPROVAL_STATEMENT,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_by": {"user": getpass.getuser(), "host": socket.gethostname()},
        "validated_plan": validated_plan,
    }
    _atomic_json(marker.expanduser().resolve(), document)
    return document


def check_marker(marker: Path, *, validated_plan: dict) -> dict:
    marker = marker.expanduser().resolve()
    document = _load_json(marker, label="formal smoke approval marker")
    approved_by = document.get("approved_by")
    if (
        document.get("schema_version") != MARKER_SCHEMA_VERSION
        or document.get("kind") != MARKER_KIND
        or document.get("approval_statement") != APPROVAL_STATEMENT
        or not isinstance(document.get("approved_at_utc"), str)
        or not isinstance(approved_by, dict)
        or not isinstance(approved_by.get("user"), str)
        or not approved_by["user"]
        or not isinstance(approved_by.get("host"), str)
        or not approved_by["host"]
        or document.get("validated_plan") != validated_plan
    ):
        raise SmokeGateError(
            "Formal smoke approval marker is stale, malformed, or does not match current results"
        )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("approve", "check"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--approval-marker", type=Path, required=True)
    parser.add_argument("--approval-statement")
    args = parser.parse_args(argv)
    validated = validate_smoke_plan(
        run_root=args.run_root,
        expected_commit=args.expected_commit,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
    )
    if args.command == "approve":
        approve(
            args.approval_marker,
            validated_plan=validated,
            approval_statement=args.approval_statement or "",
        )
        print(f"approved exact smoke plan: {args.approval_marker.resolve()}")
    else:
        check_marker(args.approval_marker, validated_plan=validated)
        print(f"validated formal smoke approval: {args.approval_marker.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
