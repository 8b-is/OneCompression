#!/usr/bin/env python3
"""Fair, paired HayateQuant/TurboQuant speed and quality benchmark.

This runner deliberately keeps three different contracts separate:

* ``payload_api`` times the public ``encode`` and ``decode`` API.  It
  materialises logical low-bit payloads as **unpacked int64 index tensors**;
  ``stored_bits`` counts fixed-width data-plane growth (codes, fp16 scales,
  and enabled mask/QJL/outlier fields), not self-describing control tags,
  Python/container bytes, persistent codec state, or the tensors' physical
  memory footprint.  The TurboQuant row in this category is the repository's
  unmodified :class:`TurboQuantCodec`; no HayateQuant fusion is credited to it.
* ``research_fake_quant`` times ``fake_quant_roundtrip``.  TurboQuant inherits
  the unfused ``encode -> decode`` implementation.  The fixed HQ, frontier,
  and dense-rotation recipes use the fused research implementation, whereas
  AdaptiveHQ currently falls back to the inherited payload chain because its
  selected K/V pair is update-dependent.  Every tensor regime persists a
  runtime probe; results are never silently presented as the same API or as
  packed serving.
* ``research_fake_quant_e2e`` measures that fake-quant path inside a real
  Transformers generation loop.  It is not a packed low-bit serving result.

The formal matrix contains ``TurboQuant-paper`` and all four public proposal
recipes: ``HQ``, ``HQ-Frontier``, ``AdaptiveHQ``, and ``RotationHQ``.  The
proposal rows use the same named recipes as every quality runner.  The
adaptive selector is carried in low fraction bits of an existing fp16 value
scale and therefore adds zero logical data-plane bits.

Important protocol constraints
------------------------------

* One invocation accepts one model, dtype, and average bit rate.  Launch a
  fresh process for every other point: codebook/device caches and optional
  CUDA-graph specialisations are process-global.
* ``torch.compile`` over prefill or the whole codec is forbidden.  The only
  accepted compile mode is ``--compile-mode decode-only``; the runner sets
  both OneComp guards and checks their effective values before importing the
  codec.
* Inputs are generated directly on the selected GPU in explicit fp16/bf16.
  KV heads and head dimension come from the model config, never from a
  hard-coded "Llama-shaped" tensor.
* Native packed key-only/hybrid kernels are outside this runner.  Such a path
  must be reported as a native hybrid, never as full HayateQuant.

Example (one fresh process per output)::

    PYTHONPATH=$PWD .venv/bin/python \
      benchmark/kvcomp-codec/hayatequant_fair_speed.py \
      --model /path/to/Llama-3.1-8B-Instruct --avg-bits 3 \
      --dtype float16 --mode both --fused-backend triton \
      --compile-mode decode-only --output results/fair_speed_r3.json

``python hayatequant_fair_speed.py --help`` and the pure statistics/order
helpers are CPU-safe; an actual benchmark intentionally requires CUDA.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

SCHEMA_NAME = "onecomp.hayatequant_fair_speed"
SCHEMA_VERSION = 4
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
_RUN_STARTED = False
_NON_SEMANTIC_ARGUMENTS = {"output", "overwrite"}
_FORMAL_MODEL_EVALUATION_ROLES = {
    "llama-3.1-8b-instruct": "development_and_recipe_selection",
    "Qwen3-8B": "additional_development_evaluation_independence_not_claimed",
}
ADAPTIVE_HQ_POLICY_VERSION = "adaptive_hq_cache_causal_fixed_incumbent_hysteresis_v6"
ADAPTIVE_HQ_DISTORTION_TABLE_VERSION = (
    "source_qmc128_exact_codebook_fp16scale_safe_two_pair_update_carrier_v4"
)
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
HQ_FRONTIER_FAKE_QUANT_IMPLEMENTATION = "HayateQuant fused research fake_quant_roundtrip"
ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION = (
    "inherited encode_then_decode fallback; AdaptiveHQ fused path unsupported"
)
ROTATION_HQ_FAKE_QUANT_IMPLEMENTATION = "HayateQuant fused research fake_quant_roundtrip"


def _model_evaluation_role(model: str) -> str:
    return _FORMAL_MODEL_EVALUATION_ROLES.get(
        pathlib.Path(model).name, "custom_evaluation_role_unspecified"
    )


def _semantic_config(args: argparse.Namespace) -> dict[str, Any]:
    """Scientific CLI fields included in immutable result identity."""

    return {key: value for key, value in vars(args).items() if key not in _NON_SEMANTIC_ARGUMENTS}


def _run_fingerprint(semantic_config: Mapping[str, Any], code_commit: str | None) -> str:
    identity = {
        "semantic_config": dict(semantic_config),
        "code_commit": code_commit,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paired CUDA-event benchmark for TurboQuant-paper and all four "
            "public HayateQuant recipes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--revision", default=None, help="optional HF model revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="CUDA device visible to this process")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--avg-bits",
        type=int,
        choices=(2, 3, 4),
        default=3,
        help="single paper average rate r (HQ uses b=2 K5/V3, K7/V5, or K9/V7)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--micro-seq-len", type=int, default=4096)
    parser.add_argument("--decode-seq-len", type=int, default=1)
    parser.add_argument("--quality-query-len", type=int, default=8)
    parser.add_argument("--micro-warmup", type=int, default=8)
    parser.add_argument("--micro-repeats", type=int, default=40)
    parser.add_argument("--micro-inner-loops", type=int, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--mode",
        choices=("micro", "e2e", "both"),
        default="micro",
        help="E2E is the Transformers research fake-quant cache, not native packed serving",
    )
    parser.add_argument("--e2e-prompt-len", type=int, default=4096)
    parser.add_argument("--e2e-gen-len", type=int, default=64)
    parser.add_argument("--e2e-warmup", type=int, default=1)
    parser.add_argument("--e2e-repeats", type=int, default=5)
    parser.add_argument("--recent-tokens", type=int, default=0)
    parser.add_argument("--sink-tokens", type=int, default=0)
    parser.add_argument(
        "--attention-implementation",
        default=None,
        help="optional Transformers attn_implementation (for example sdpa or eager)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("off", "decode-only"),
        default="off",
        help="compile-all/prefill compilation is intentionally not available",
    )
    parser.add_argument(
        "--fused-backend",
        choices=("eager", "triton"),
        default="triton",
        help="backend for HayateQuant's research fake-quant round trip",
    )
    parser.add_argument("--output", required=True, help="new JSON output (atomic write)")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _positive_ints(args: argparse.Namespace) -> None:
    fields = (
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
    )
    for name in fields:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.avg_bits not in (2, 3, 4):
        raise ValueError("--avg-bits must be one of the paper rates 2, 3, or 4")
    if args.sink_tokens < 0 or args.recent_tokens < 0:
        raise ValueError("--sink-tokens and --recent-tokens must be non-negative")


def _configure_effective_environment(args: argparse.Namespace) -> dict[str, str]:
    """Set and validate runtime switches before importing OneComp/torch.

    An inherited compile-all environment is rejected rather than silently
    contaminating a run that says ``--compile-mode off``.
    """

    inherited_compile = os.getenv("ONECOMP_COMPILE_CODEC", "0") == "1"
    inherited_decode_only = os.getenv("ONECOMP_COMPILE_DECODE_ONLY", "0") == "1"
    if args.compile_mode == "off" and inherited_compile:
        kind = "decode-only" if inherited_decode_only else "compile-all"
        raise RuntimeError(
            f"inherited ONECOMP_COMPILE_CODEC=1 ({kind}) conflicts with "
            "--compile-mode off; launch a fresh process with the variable unset"
        )
    if args.compile_mode == "decode-only":
        os.environ["ONECOMP_COMPILE_CODEC"] = "1"
        os.environ["ONECOMP_COMPILE_DECODE_ONLY"] = "1"
    else:
        os.environ["ONECOMP_COMPILE_CODEC"] = "0"
        os.environ["ONECOMP_COMPILE_DECODE_ONLY"] = "0"

    # Prefill Triton routing is a separate experimental path and is excluded
    # here so quality and eager payload measurements retain the same contract.
    if os.getenv("ONECOMP_TRITON_PREFILL", "0") == "1":
        raise RuntimeError(
            "ONECOMP_TRITON_PREFILL=1 is outside the fair-speed protocol; "
            "launch a fresh process with it unset"
        )
    os.environ["ONECOMP_TRITON_PREFILL"] = "0"
    os.environ["ONECOMP_TRITON_CODEC"] = "1" if args.fused_backend == "triton" else "0"

    if os.environ["ONECOMP_COMPILE_CODEC"] == "1" and (
        os.environ["ONECOMP_COMPILE_DECODE_ONLY"] != "1"
    ):
        raise RuntimeError("compile-all is forbidden; decode-only guard is not active")
    return {
        "ONECOMP_COMPILE_CODEC": os.environ["ONECOMP_COMPILE_CODEC"],
        "ONECOMP_COMPILE_DECODE_ONLY": os.environ["ONECOMP_COMPILE_DECODE_ONLY"],
        "ONECOMP_TRITON_CODEC": os.environ["ONECOMP_TRITON_CODEC"],
        "ONECOMP_TRITON_PREFILL": os.environ["ONECOMP_TRITON_PREFILL"],
    }


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, where ``q`` is in [0, 100]."""

    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= float(q) <= 100.0:
        raise ValueError("percentile q must be in [0, 100]")
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _bootstrap_median_ci(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Deterministic percentile-bootstrap 95% CI for the median."""

    xs = [float(x) for x in values]
    if not xs:
        raise ValueError("bootstrap needs at least one value")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if len(xs) == 1:
        return xs[0], xs[0]
    rng = random.Random(int(seed))
    n = len(xs)
    medians = [statistics.median(rng.choices(xs, k=n)) for _ in range(int(resamples))]
    return _percentile(medians, 2.5), _percentile(medians, 97.5)


def _summary(
    values: Sequence[float],
    *,
    unit: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    xs = [float(x) for x in values]
    if not xs:
        raise ValueError("summary needs at least one value")
    q1, q3 = _percentile(xs, 25), _percentile(xs, 75)
    ci_lo, ci_hi = _bootstrap_median_ci(xs, resamples=bootstrap_resamples, seed=seed)
    return {
        "unit": unit,
        "n": len(xs),
        "median": statistics.median(xs),
        "p5": _percentile(xs, 5),
        "p95": _percentile(xs, 95),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "bootstrap_median_95ci": [ci_lo, ci_hi],
        "bootstrap_resamples": int(bootstrap_resamples),
    }


def _paired_speedup_report(
    raw_paired_repeats: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, tuple[str, str]],
    *,
    value_key: str,
    value_unit: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute latency ratios within the same counterbalanced repeat.

    Each comparison maps ``name -> (baseline, candidate)`` and reports
    ``baseline_latency / candidate_latency``.  The technical repeats are
    paired observations inside one process; their bootstrap interval is kept
    only as an in-process descriptive diagnostic.  Formal inference is done
    later across independently launched processes by the aggregate verifier.
    """

    if not raw_paired_repeats:
        raise ValueError("paired speedup needs at least one repeat")
    if not comparisons:
        raise ValueError("paired speedup needs at least one comparison")
    by_comparison: dict[str, list[dict[str, Any]]] = {name: [] for name in comparisons}
    for expected_repeat, repeat in enumerate(raw_paired_repeats):
        if repeat.get("repeat") != expected_repeat:
            raise ValueError("paired repeats must be contiguous and ordered")
        samples = repeat.get("samples")
        if not isinstance(samples, list):
            raise ValueError("paired repeat samples must be a list")
        by_method: dict[str, Mapping[str, Any]] = {}
        for sample in samples:
            if not isinstance(sample, Mapping) or not isinstance(sample.get("method"), str):
                raise ValueError("paired repeat contains a malformed sample")
            method = str(sample["method"])
            if method in by_method:
                raise ValueError(f"paired repeat contains duplicate method {method!r}")
            by_method[method] = sample
        for name, (baseline, candidate) in comparisons.items():
            if baseline not in by_method or candidate not in by_method:
                raise ValueError(f"paired repeat is missing {baseline!r} or {candidate!r}")
            baseline_value = float(by_method[baseline][value_key])
            candidate_value = float(by_method[candidate][value_key])
            if (
                not math.isfinite(baseline_value)
                or not math.isfinite(candidate_value)
                or baseline_value <= 0.0
                or candidate_value <= 0.0
            ):
                raise ValueError("paired timing values must be finite and positive")
            by_comparison[name].append(
                {
                    "repeat": expected_repeat,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "candidate_speedup_x": baseline_value / candidate_value,
                    "baseline_order_position": int(by_method[baseline]["order_position"]),
                    "candidate_order_position": int(by_method[candidate]["order_position"]),
                }
            )

    rows = {}
    for name, paired_rows in by_comparison.items():
        baseline, candidate = comparisons[name]
        ratios = [row["candidate_speedup_x"] for row in paired_rows]
        rows[name] = {
            "baseline_method": baseline,
            "candidate_method": candidate,
            "n_paired_repeats": len(paired_rows),
            "raw_paired_ratios": paired_rows,
            "within_process_summary": _summary(
                ratios,
                unit="candidate_speedup_x",
                bootstrap_resamples=bootstrap_resamples,
                seed=_stable_seed(seed, f"paired:{value_key}:{name}"),
            ),
        }
    return {
        "ratio_definition": "baseline_latency / candidate_latency",
        "interpretation": "greater_than_1_means_candidate_is_faster",
        "source_value_key": value_key,
        "source_value_unit": value_unit,
        "pairing_unit": "same_counterbalanced_repeat_within_one_process",
        "inferential_unit": "independently_launched_fresh_process",
        "within_process_ci_warning": (
            "technical-repeat bootstrap intervals are descriptive only; "
            "do not pool repeats across processes as independent samples"
        ),
        "comparisons": rows,
    }


def _normalise_elapsed_ms(total_elapsed_ms: float, inner_loops: int) -> float:
    if inner_loops < 1:
        raise ValueError("inner_loops must be positive")
    return float(total_elapsed_ms) / int(inner_loops)


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "big")


def _balanced_orders(labels: Sequence[str], repeats: int, seed: int) -> list[list[str]]:
    """Randomised cyclic Latin orders, balanced in blocks of ``len(labels)``.

    Within a complete block every method appears once in every order position.
    Each block receives a fresh seeded base permutation and alternating
    direction, avoiding a fixed first/last method under thermal drift.
    """

    labels = list(labels)
    if not labels:
        raise ValueError("at least one label is required")
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    rng = random.Random(int(seed))
    n = len(labels)
    orders: list[list[str]] = []
    block = 0
    while len(orders) < repeats:
        base = labels[:]
        rng.shuffle(base)
        if block % 2:
            base.reverse()
        for offset in range(n):
            orders.append(base[offset:] + base[:offset])
            if len(orders) == repeats:
                break
        block += 1
    return orders


def _text_config(config: Any) -> Any:
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter(decoder=True)
        except TypeError:
            return getter()
    return getattr(config, "text_config", config)


def _resolve_model_shape(config: Any) -> dict[str, Any]:
    """Derive ordinary GQA KV shape solely from a Transformers config."""

    cfg = _text_config(config)
    attention_heads = getattr(cfg, "num_attention_heads", None)
    if attention_heads is None:
        raise ValueError("model config has no num_attention_heads")
    attention_heads = int(attention_heads)
    kv_heads_raw = getattr(cfg, "num_key_value_heads", None)
    if kv_heads_raw is None:
        kv_heads = attention_heads
        kv_derivation = "fallback:num_attention_heads"
    else:
        kv_heads = int(kv_heads_raw)
        kv_derivation = "config:num_key_value_heads"
    head_dim_raw = getattr(cfg, "head_dim", None)
    if head_dim_raw is not None:
        head_dim = int(head_dim_raw)
        head_dim_derivation = "config:head_dim"
    else:
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None or int(hidden_size) % attention_heads:
            raise ValueError("cannot derive head_dim from hidden_size/num_attention_heads")
        head_dim = int(hidden_size) // attention_heads
        head_dim_derivation = "config:hidden_size/num_attention_heads"
    if attention_heads < 1 or kv_heads < 1 or head_dim < 1:
        raise ValueError("model config resolved a non-positive attention shape")
    if attention_heads % kv_heads:
        raise ValueError(
            f"num_attention_heads={attention_heads} is not divisible by "
            f"num_key_value_heads={kv_heads}; ordinary GQA replication is undefined"
        )
    return {
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "gqa_group_size": attention_heads // kv_heads,
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0) or 0),
        "kv_heads_derivation": kv_derivation,
        "head_dim_derivation": head_dim_derivation,
    }


def _resolve_model_capacity(config: Any) -> dict[str, Any]:
    """Resolve the model's advertised text context without guessing.

    Formal models expose ``max_position_embeddings``.  The two legacy aliases
    are retained for exploratory configs, but the selected source is recorded
    so results cannot silently mix different notions of context capacity.
    """

    cfg = _text_config(config)
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(cfg, name, None)
        if value is not None:
            capacity = int(value)
            if capacity < 1:
                raise ValueError(f"model config {name} must be positive")
            return {
                "tokens": capacity,
                "derivation": f"config:{name}",
                "original_max_position_embeddings": getattr(
                    cfg, "original_max_position_embeddings", None
                ),
            }
    raise ValueError(
        "model config has no max_position_embeddings/n_positions/seq_length; "
        "E2E context capacity cannot be audited"
    )


def _validate_e2e_context_capacity(
    prompt_len: int, gen_len: int, capacity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail before model allocation when the timed generation cannot fit."""

    prompt_len, gen_len = int(prompt_len), int(gen_len)
    maximum = int(capacity["tokens"])
    required = prompt_len + gen_len
    if prompt_len < 1 or gen_len < 1:
        raise ValueError("E2E prompt and generation lengths must be positive")
    if required > maximum:
        raise ValueError(
            "E2E prompt+generation exceeds the model context capacity: "
            f"{prompt_len}+{gen_len}={required} > {maximum}"
        )
    return {
        "prompt_tokens": prompt_len,
        "decode_steps": gen_len,
        "required_context_tokens": required,
        "model_capacity_tokens": maximum,
        "capacity_derivation": capacity["derivation"],
        "fits": True,
    }


def _run_command(argv: Sequence[str], cwd: pathlib.Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_provenance(repo: pathlib.Path) -> dict[str, Any]:
    commit = _run_command(("git", "rev-parse", "HEAD"), repo)
    branch = _run_command(("git", "branch", "--show-current"), repo)
    status = _run_command(("git", "status", "--porcelain"), repo)
    return {
        "repository_root": str(repo),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_paths": status.splitlines() if status else [],
    }


def _nvidia_smi_snapshot() -> dict[str, Any]:
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "pstate",
        "clocks.current.graphics",
        "clocks.current.sm",
        "clocks.current.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "memory.total",
        "memory.used",
    )
    raw = _run_command(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    rows = []
    if raw:
        for line in raw.splitlines():
            values = [part.strip() for part in line.split(",")]
            rows.append(dict(zip(fields, values)))
    return {
        "sampled_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "visible_devices_env": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpus_seen_by_nvidia_smi": rows,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json_write(path: pathlib.Path, payload: dict[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite explicitly): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class _MethodSpec:
    label: str
    codec_factory: Callable[[], Any]
    semantic_role: str
    fake_quant_implementation: str
    proposal_method: bool
    proposal_variant: str | None


def _method_specs(avg_bits: int) -> list[_MethodSpec]:
    # Imports happen only after the environment and repository path preflight.
    from onecomp.kvcomp.codec.hayate.recipes import (
        build_hayate_adaptive_rate,
        build_hayate_hq,
        build_hayate_hq_frontier,
    )
    from onecomp.kvcomp.codec.turboquant import TurboQuantCodec

    r = int(avg_bits)
    return [
        _MethodSpec(
            label="TurboQuant-paper",
            codec_factory=lambda: TurboQuantCodec(
                bits=r, value_bits=r, rotation_mode="dense", dist="beta"
            ),
            semantic_role="existing TurboQuant class; symmetric scalar Beta/L2 reference",
            fake_quant_implementation=TURBOQUANT_FAKE_QUANT_IMPLEMENTATION,
            proposal_method=False,
            proposal_variant=None,
        ),
        _MethodSpec(
            label="HQ",
            codec_factory=lambda: build_hayate_hq(r),
            semantic_role="canonical globally fixed HQ paper proposal",
            fake_quant_implementation=HQ_FAKE_QUANT_IMPLEMENTATION,
            proposal_method=True,
            proposal_variant="fixed_global",
        ),
        _MethodSpec(
            label="HQ-Frontier",
            codec_factory=lambda: build_hayate_hq_frontier(r),
            semantic_role="integer K(r+1)/V(r-1) HQ frontier proposal",
            fake_quant_implementation=HQ_FRONTIER_FAKE_QUANT_IMPLEMENTATION,
            proposal_method=True,
            proposal_variant="integer_frontier",
        ),
        _MethodSpec(
            label="AdaptiveHQ",
            codec_factory=lambda: build_hayate_adaptive_rate(r),
            semantic_role="causal online-rate-allocation HQ paper proposal",
            fake_quant_implementation=ADAPTIVE_HQ_FAKE_QUANT_IMPLEMENTATION,
            proposal_method=True,
            proposal_variant="causal_adaptive",
        ),
        _MethodSpec(
            label="RotationHQ",
            codec_factory=lambda: build_hayate_hq(r, rotation_mode="dense"),
            semantic_role="fixed HQ payload with dense rotation proposal",
            fake_quant_implementation=ROTATION_HQ_FAKE_QUANT_IMPLEMENTATION,
            proposal_method=True,
            proposal_variant="rotation_dense",
        ),
    ]


def _speedup_comparisons(
    specs: Sequence[_MethodSpec],
) -> dict[str, tuple[str, str]]:
    """Predeclared paired latency comparisons used by micro and E2E runs."""

    if [spec.label for spec in specs] != [
        "TurboQuant-paper",
        "HQ",
        "HQ-Frontier",
        "AdaptiveHQ",
        "RotationHQ",
    ]:
        raise ValueError("fair-speed protocol requires the exact five-method matrix")
    return {
        "hq_vs_turboquant": ("TurboQuant-paper", "HQ"),
        "hq_frontier_vs_turboquant": ("TurboQuant-paper", "HQ-Frontier"),
        "adaptive_hq_vs_turboquant": ("TurboQuant-paper", "AdaptiveHQ"),
        "rotation_hq_vs_turboquant": ("TurboQuant-paper", "RotationHQ"),
    }


def _cuda_elapsed_ms(torch: Any, fn: Callable[[], Any], inner_loops: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = None
    for _ in range(int(inner_loops)):
        result = fn()
    end.record()
    end.synchronize()
    # Keep the result live until the end event completes.  This matters for
    # payload-producing encode calls whose temporary tensors would otherwise
    # become allocator candidates while kernels are still queued.
    _ = result
    return _normalise_elapsed_ms(start.elapsed_time(end), inner_loops)


def _warm_all(
    torch: Any,
    functions: dict[str, Callable[[], Any]],
    warmup: int,
    seed: int,
) -> list[list[str]]:
    orders = _balanced_orders(list(functions), warmup, seed)
    for order in orders:
        for label in order:
            functions[label]()
    torch.cuda.synchronize()
    return orders


def _paired_cuda_measurement(
    torch: Any,
    functions: dict[str, Callable[[], Any]],
    *,
    warmup: int,
    repeats: int,
    inner_loops: int,
    seed: int,
    bootstrap_resamples: int,
    comparisons: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    warmup_orders = _warm_all(torch, functions, warmup, seed ^ 0x5A5A5A5A)
    orders = _balanced_orders(list(functions), repeats, seed)
    raw = []
    by_method: dict[str, list[float]] = {label: [] for label in functions}
    for repeat_idx, order in enumerate(orders):
        samples = []
        for position, label in enumerate(order):
            elapsed = _cuda_elapsed_ms(torch, functions[label], inner_loops)
            by_method[label].append(elapsed)
            samples.append(
                {
                    "method": label,
                    "order_position": position,
                    "elapsed_ms_per_call": elapsed,
                    "inner_loops": int(inner_loops),
                }
            )
        raw.append({"repeat": repeat_idx, "order": order, "samples": samples})
    summaries = {
        label: _summary(
            values,
            unit="ms_per_call",
            bootstrap_resamples=bootstrap_resamples,
            seed=_stable_seed(seed, label),
        )
        for label, values in by_method.items()
    }
    return {
        "timer": "torch.cuda.Event",
        "timer_native_unit": "milliseconds",
        "timer_completion_barrier": "end_event.synchronize before elapsed_time read",
        "normalisation": f"event elapsed time / {inner_loops} inner loop(s)",
        "warmup_orders": warmup_orders,
        "raw_paired_repeats": raw,
        "summary": summaries,
        "paired_speedups": _paired_speedup_report(
            raw,
            comparisons,
            value_key="elapsed_ms_per_call",
            value_unit="ms_per_call",
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def _make_gpu_tensor(torch: Any, shape: Sequence[int], dtype: Any, device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        tuple(int(x) for x in shape), generator=generator, device=device, dtype=dtype
    )


def _codec_config(codec: Any) -> dict[str, Any]:
    cfg = codec.get_kv_config()
    # Configs are expected to be JSON-native; stringify the rare artifact-like
    # value instead of allowing a result write to fail after a long GPU run.
    clean = {}
    for key, value in cfg.items():
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            clean[key] = value
        else:
            clean[key] = str(value)
    effective = {}
    if hasattr(codec, "_scale_mode_for"):
        effective = {
            "key_scale_mode": codec._scale_mode_for("k"),  # pylint: disable=protected-access
            "value_scale_mode": codec._scale_mode_for("v"),  # pylint: disable=protected-access
        }
    return {"declared": clean, "effective": effective}


def _resolved_codec_shape(codec: Any, head_dim: int) -> dict[str, Any]:
    """Record the exact public role-resolved recipe at this model dimension.

    A block width alone is insufficient scientific provenance: ``auto`` can
    resolve to different rotations for K and V at the same average rate, and
    TurboQuant Beta uses an L2 scale while its matched Hayate control stores an
    equivalent RMS parameterization.  Keep the codec-owned record verbatim so
    the benchmark cannot independently (and incorrectly) reimplement policy.
    """

    resolver = getattr(codec, "get_resolved_kv_config", None)
    if not callable(resolver):
        raise TypeError(f"{type(codec).__name__} does not expose get_resolved_kv_config")
    resolved = resolver(int(head_dim))
    if not isinstance(resolved, dict):
        raise TypeError("get_resolved_kv_config must return a JSON object")
    # Fail before a GPU run if a codec accidentally returns an artifact-like
    # or otherwise non-serialisable value.  The canonical recipe hash below
    # must cover the exact public record, not a lossy stringification.
    json.dumps(resolved, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return resolved


def _method_config_record(spec: _MethodSpec, codec: Any, head_dim: int) -> dict[str, Any]:
    """Canonical, JSON-native description of one timed codec recipe."""

    return {
        "semantic_role": spec.semantic_role,
        "fake_quant_implementation": spec.fake_quant_implementation,
        "proposal_method": spec.proposal_method,
        "proposal_variant": spec.proposal_variant,
        "codec": _codec_config(codec),
        "resolved_at_model_head_dim": _resolved_codec_shape(codec, head_dim),
    }


def _fake_quant_effective_implementation(codec: Any, head_dim: int) -> dict[str, Any]:
    """Describe the path that ``fake_quant_roundtrip`` will actually execute.

    HayateQuant owns an override, but that override deliberately delegates to
    the base payload chain for unsupported configurations.  In particular,
    AdaptiveHQ is always delegated today.  Persisting this probe prevents a
    method-level label from turning a fallback timing into a fused timing.
    """

    probe = getattr(codec, "_rt_supported", None)
    if callable(probe):
        fused = bool(probe(int(head_dim)))
        return {
            "effective_path": (
                "hayate_fused_roundtrip" if fused else "inherited_encode_then_decode"
            ),
            "payload_materialized": not fused,
            "hayate_override_entered": fused,
            "runtime_probe": "HayateQuantCodec._rt_supported(head_dim)",
        }
    return {
        "effective_path": "inherited_encode_then_decode",
        "payload_materialized": True,
        "hayate_override_entered": False,
        "runtime_probe": "codec_has_no_hayate_rt_support_probe",
    }


def _recipe_sha256(record: Mapping[str, Any]) -> str:
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(serialised).hexdigest()


def _method_config_records(
    specs: Sequence[_MethodSpec], codecs: Mapping[str, Any], head_dim: int
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records = {
        spec.label: _method_config_record(spec, codecs[spec.label], head_dim) for spec in specs
    }
    return records, {label: _recipe_sha256(record) for label, record in records.items()}


def _validate_rate_contract(
    specs: Sequence[_MethodSpec],
    codecs: Mapping[str, Any],
    quality_rows: Mapping[str, Mapping[str, Any]],
    avg_bits: int,
) -> dict[str, Any]:
    """Prove nominal and accounted payload-rate equality for every method row."""

    nominal_rows: dict[str, dict[str, Any]] = {}
    stored_bits = set()
    input_elements = set()
    for spec in specs:
        codec = codecs[spec.label]
        key_bits = int(codec.bits)
        value_bits = int(codec._value_bits())  # pylint: disable=protected-access
        nominal = float(getattr(codec, "avg_bits", None) or (key_bits + value_bits) / 2.0)
        if nominal != float(avg_bits):
            raise AssertionError(
                f"{spec.label} nominal average rate is {nominal}, expected {avg_bits}"
            )
        row = quality_rows[spec.label]
        payload_bits = row.get("stored_bits")
        elements = row.get("input_kv_elements")
        if not isinstance(payload_bits, int) or payload_bits < 1:
            raise AssertionError(f"{spec.label} did not report positive payload bits")
        if not isinstance(elements, int) or elements < 1:
            raise AssertionError(f"{spec.label} did not report input KV elements")
        stored_bits.add(payload_bits)
        input_elements.add(elements)
        nominal_rows[spec.label] = {
            "declared_key_bits": key_bits,
            "declared_value_bits": value_bits,
            "nominal_average_code_bits_per_coordinate": nominal,
            "allocation": spec.proposal_variant or "symmetric_scalar",
            "accounted_payload_bits": payload_bits,
            "input_kv_elements": elements,
            "accounted_payload_bits_per_kv_element": payload_bits / elements,
        }
    if len(input_elements) != 1:
        raise AssertionError("fair-speed methods did not use the same quality tensor shape")
    if len(stored_bits) != 1:
        raise AssertionError(
            "fair-speed methods are not equal-rate after fixed-width data-plane accounting: "
            f"{sorted(stored_bits)}"
        )
    return {
        "comparison_basis": (
            "same input shape, same arithmetic K/V average code rate, and equal "
            "payload.numel_bits including enabled fixed-width data-plane fields"
        ),
        "shared_codebook_storage_accounting": (
            "global immutable codebooks/LUTs/rotations and Python control containers "
            "are out-of-band/cache-level state and are not charged per cache payload"
        ),
        "nominal_average_bits": int(avg_bits),
        "equal_input_shape": True,
        "equal_accounted_payload_bits": True,
        "methods": nominal_rows,
    }


def _quality_metrics(
    torch: Any,
    codecs: dict[str, Any],
    key: Any,
    value: Any,
    query: Any,
    gqa_group_size: int,
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    outputs: dict[str, tuple[Any, Any]] = {}
    rows: dict[str, Any] = {}
    key_for_attention = key.repeat_interleave(gqa_group_size, dim=1).float()
    value_for_attention = value.repeat_interleave(gqa_group_size, dim=1).float()
    scale = 1.0 / math.sqrt(key.shape[-1])
    reference_attention = (
        torch.softmax((query.float() @ key_for_attention.transpose(-1, -2)) * scale, dim=-1)
        @ value_for_attention
    )
    for label, codec in codecs.items():
        payload = codec.encode(key, value)
        key_hat, value_hat = codec.decode(payload)
        outputs[label] = (key_hat, value_hat)
        key_attn = key_hat.repeat_interleave(gqa_group_size, dim=1).float()
        value_attn = value_hat.repeat_interleave(gqa_group_size, dim=1).float()
        quant_attention = (
            torch.softmax((query.float() @ key_attn.transpose(-1, -2)) * scale, dim=-1)
            @ value_attn
        )
        denom = reference_attention.float().norm().clamp_min(1e-12)
        stored_bits = payload.numel_bits()
        rows[label] = {
            "quality_api": "decode(encode(key, value)); same codec recipe as timed row",
            "input_kv_elements": int(key.numel() + value.numel()),
            "key_mse": float((key_hat.float() - key.float()).pow(2).mean().item()),
            "value_mse": float((value_hat.float() - value.float()).pow(2).mean().item()),
            "attention_output_relative_l2": float(
                ((quant_attention - reference_attention).float().norm() / denom).item()
            ),
            "stored_bits": int(stored_bits) if stored_bits is not None else None,
            "stored_bits_per_kv_element": (
                float(stored_bits) / float(key.numel() + value.numel())
                if stored_bits is not None
                else None
            ),
        }
    return rows, outputs


def _parity_checks(
    torch: Any,
    specs: Sequence[_MethodSpec],
    codecs: dict[str, Any],
    outputs: dict[str, tuple[Any, Any]],
    key: Any,
    value: Any,
    dtype_name: str,
    quality_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _ = dtype_name
    # Verify each timed proposal fake-quant path remains quality-equivalent to
    # its payload API. Fixed recipes are fused; AdaptiveHQ is an explicitly
    # recorded inherited encode/decode fallback. This is not expected to be
    # bit-identical across Triton/eager implementations, hence MSE relative
    # deltas are kept.
    fake_quant_rows = []
    for spec in specs:
        if not spec.proposal_method:
            continue
        khat, vhat, stored = codecs[spec.label].fake_quant_roundtrip(key, value)
        payload_stored = quality_rows[spec.label].get("stored_bits")
        fake_quant_stored = int(stored) if stored is not None else None
        if fake_quant_stored != payload_stored:
            raise AssertionError(
                f"{spec.label} timed fake-quant path reports {fake_quant_stored} stored bits, "
                f"but its quality payload reports {payload_stored}"
            )
        effective = _fake_quant_effective_implementation(codecs[spec.label], int(key.shape[-1]))
        for role, source, payload_hat, fake_quant_hat in (
            ("key", key, outputs[spec.label][0], khat),
            ("value", value, outputs[spec.label][1], vhat),
        ):
            payload_mse = float((payload_hat.float() - source.float()).pow(2).mean().item())
            fake_quant_mse = float((fake_quant_hat.float() - source.float()).pow(2).mean().item())
            fake_quant_rows.append(
                {
                    "method": spec.label,
                    "role": role,
                    "effective_implementation": effective,
                    "payload_mse": payload_mse,
                    "fake_quant_mse": fake_quant_mse,
                    "relative_mse_delta": abs(payload_mse - fake_quant_mse)
                    / max(payload_mse, 1e-12),
                    "stored_bits_from_fake_quant": fake_quant_stored,
                    "stored_bits_from_quality_payload": payload_stored,
                    "stored_bits_match": True,
                }
            )
    return {
        "proposal_payload_vs_fake_quant_quality": fake_quant_rows,
    }


def _validate_proposal_contract(
    specs: Sequence[_MethodSpec], codecs: dict[str, Any], avg_bits: int
) -> dict[str, Any]:
    """Fail if any public proposal drifts from its named recipe identity."""

    rate = int(avg_bits)
    rows: dict[str, Any] = {}
    for spec in specs:
        if not spec.proposal_method:
            continue
        codec = codecs[spec.label]
        declared = codec.get_kv_config()
        resolved = codec.get_resolved_kv_config(128)
        codebook_artifacts = resolved.get("codebook_artifacts")
        if (
            declared.get("bits") != rate
            or declared.get("value_bits") != rate
            or declared.get("avg_bits") != float(rate)
            or declared.get("block_size") != 2
            or declared.get("dist") != "gaussian"
            or declared.get("encoder") != "exact"
            or not isinstance(codebook_artifacts, dict)
            or codebook_artifacts.get("development_override_dir") is not None
            or not isinstance(codebook_artifacts.get("artifacts"), list)
            or any(
                artifact.get("source") != "shipped_manifest"
                or artifact.get("manifest_pinned") is not True
                or artifact.get("manifest_sha256")
                != "818df8a17a0cac464e4b7d844e63c48ab56dabf70c6693b80af6099f5d83e573"
                for artifact in codebook_artifacts["artifacts"]
            )
        ):
            raise AssertionError(f"{spec.label} drifted from the formal HQ common contract")
        selector = None
        if spec.label in {"HQ", "RotationHQ"}:
            expected_pair = [2 * rate + 1, 2 * rate - 1]
            if (
                declared.get("max_codebook_bits") != 9
                or declared.get("key_scale_mode") != "nc"
                or declared.get("value_scale_mode") != "ls_iter"
                or declared.get("required_codebook_bundle") != "fixed_hq_b2_v1"
                or declared.get("rotation_mode")
                != ("dense" if spec.label == "RotationHQ" else "auto")
                or declared.get("adaptive_bit_split") is not None
                or [
                    declared.get("key_code_bits_per_block"),
                    declared.get("value_code_bits_per_block"),
                ]
                != expected_pair
                or codebook_artifacts.get("required") is not True
                or codebook_artifacts.get("bundle") != "fixed_hq_b2_v1"
                or codebook_artifacts.get("manifest_sha256")
                != "818df8a17a0cac464e4b7d844e63c48ab56dabf70c6693b80af6099f5d83e573"
                or len(codebook_artifacts["artifacts"]) != 4
            ):
                raise AssertionError(f"{spec.label} fixed recipe changed")
        elif spec.label == "HQ-Frontier":
            expected_pair = [2 * rate + 2, 2 * rate - 2]
            if (
                declared.get("max_codebook_bits") != 10
                or declared.get("key_scale_mode") != "ub"
                or declared.get("value_scale_mode") != "ls_iter"
                or declared.get("required_codebook_bundle") != "frontier_hq_b2_v1"
                or declared.get("rotation_mode") != "auto"
                or declared.get("adaptive_bit_split") is not None
                or [
                    declared.get("key_code_bits_per_block"),
                    declared.get("value_code_bits_per_block"),
                ]
                != expected_pair
                or codebook_artifacts.get("required") is not True
                or codebook_artifacts.get("bundle") != "frontier_hq_b2_v1"
                or codebook_artifacts.get("manifest_sha256")
                != "818df8a17a0cac464e4b7d844e63c48ab56dabf70c6693b80af6099f5d83e573"
                or len(codebook_artifacts["artifacts"]) != 5
            ):
                raise AssertionError("HQ-Frontier integer frontier recipe changed")
        elif spec.label == "AdaptiveHQ":
            expected_pair = None
            allocation = resolved.get("adaptive_allocation")
            if (
                declared.get("max_codebook_bits") != 9
                or declared.get("key_scale_mode") != "nc"
                or declared.get("value_scale_mode") != "ls_iter"
                or declared.get("required_codebook_bundle") != "adaptive_hq_b2_safe_v1"
                or declared.get("rotation_mode") != "auto"
                or declared.get("adaptive_bit_split") != "online_stats"
                or declared.get("key_bit_fraction") != (2 * rate + 1) / (4 * rate)
                or declared.get("adaptive_stats_min_tokens") != 16
                or declared.get("adaptive_bit_margin") != 4
                or declared.get("key_error_weight") != 4.0
                or declared.get("value_error_weight") != 1.0
                or declared.get("key_code_bits_per_block") is not None
                or declared.get("value_code_bits_per_block") is not None
                or not isinstance(allocation, dict)
                or codebook_artifacts.get("required") is not True
                or codebook_artifacts.get("bundle") != "adaptive_hq_b2_safe_v1"
                or codebook_artifacts.get("manifest_sha256")
                != "818df8a17a0cac464e4b7d844e63c48ab56dabf70c6693b80af6099f5d83e573"
                or len(codebook_artifacts["artifacts"]) != 7
                or allocation.get("pair_code_bits_per_block") != 4 * rate
                or allocation.get("policy_version") != ADAPTIVE_HQ_POLICY_VERSION
                or allocation.get("distortion_table_version")
                != ADAPTIVE_HQ_DISTORTION_TABLE_VERSION
                or allocation.get("objective") != ADAPTIVE_HQ_OBJECTIVE
                or allocation.get("candidate_pairs")
                != [[2 * rate, 2 * rate], [2 * rate + 1, 2 * rate - 1]]
                or allocation.get("prior_pair") != [2 * rate + 1, 2 * rate - 1]
                or allocation.get("incumbent_pair") != [2 * rate + 1, 2 * rate - 1]
                or not math.isclose(
                    float(allocation.get("switch_margin_fraction", math.nan)),
                    ADAPTIVE_HQ_SWITCH_MARGIN_FRACTION,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or allocation.get("switch_margin_basis") != ADAPTIVE_HQ_SWITCH_MARGIN_BASIS
                or allocation.get("hysteresis") != ADAPTIVE_HQ_HYSTERESIS
                or allocation.get("selector_bits_per_encode") != 1
                or allocation.get("selector_carrier") != "first_value_fp16_scale_low_fraction_bits"
                or allocation.get("selector_carrier_version") != "value_scale_fp16_fraction_lsb_v1"
                or allocation.get("selector_carrier_distortion_amortization")
                != ADAPTIVE_HQ_CARRIER_AMORTIZATION
                or allocation.get("selector_incremental_stored_bits_per_encode") != 0
                or allocation.get("selector_scale_reconstruction") != "clear_carrier_fraction_bits"
                or allocation.get("runtime_distortion_constants") != "payload_and_cache_provenance"
            ):
                raise AssertionError("AdaptiveHQ causal selector contract changed")
            selector = allocation
        else:
            raise AssertionError(f"unexpected formal proposal {spec.label}")
        rows[spec.label] = {
            "proposal_variant": spec.proposal_variant,
            "pair_code_bits_per_block": 4 * rate,
            "fixed_pair_code_bits_per_block": expected_pair,
            "exact_assignment": True,
            "selector": selector,
        }
    formal_proposals = ["HQ", "HQ-Frontier", "AdaptiveHQ", "RotationHQ"]
    if list(rows) != formal_proposals:
        raise AssertionError("formal speed runner must contain all four public HQ recipes")
    return {
        "formal_proposals": formal_proposals,
        "same_total_pair_code_bit_budget": True,
        "rows": rows,
    }


def _audit_public_payload(torch: Any, compressed: Any) -> dict[str, Any]:
    """Prove what the timed public payload actually materialises.

    The research codecs account low-bit indices as if packed, but their public
    Python payload deliberately keeps the index arrays as ordinary int64
    tensors.  A speed result must expose that distinction and fail immediately
    if a codec silently changes representation.
    """

    index_tensors = []
    scale_tensors = []
    for role_name, role_payload in (
        ("key", compressed.key_q),
        ("value", compressed.value_q),
    ):
        if not isinstance(role_payload, dict):
            raise RuntimeError(f"{role_name} public payload is not a mapping")
        mode = role_payload.get("mode")
        if mode == "single":
            subsets = (role_payload.get("main"),)
        elif mode == "outlier":
            subsets = (role_payload.get("out"), role_payload.get("in"))
        else:
            raise RuntimeError(f"{role_name} public payload has unknown mode {mode!r}")
        for subset in subsets:
            if not isinstance(subset, dict):
                raise RuntimeError(f"{role_name} public payload subset is malformed")
            codes = subset.get("codes")
            scale = subset.get("scale")
            if not isinstance(codes, torch.Tensor) or codes.dtype != torch.int64:
                raise RuntimeError(
                    f"{role_name} public payload indices must be unpacked int64 tensors"
                )
            if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16:
                raise RuntimeError(
                    f"{role_name} public payload scale metadata must be signed fp16"
                )
            index_tensors.append(codes)
            scale_tensors.append(scale)

    index_elements = sum(int(tensor.numel()) for tensor in index_tensors)
    scale_elements = sum(int(tensor.numel()) for tensor in scale_tensors)
    if index_elements <= 0 or scale_elements <= 0:
        raise RuntimeError("public logical payload contains no indices or scale metadata")
    materialized_index_tensor_bytes = sum(
        int(tensor.numel() * tensor.element_size()) for tensor in index_tensors
    )
    materialized_scale_tensor_bytes = sum(
        int(tensor.numel() * tensor.element_size()) for tensor in scale_tensors
    )
    return {
        "index_tensor_count": len(index_tensors),
        "index_elements": index_elements,
        "materialized_index_tensor_bits": materialized_index_tensor_bytes * 8,
        "scale_tensor_count": len(scale_tensors),
        "scale_elements": scale_elements,
        "materialized_scale_tensor_bits": materialized_scale_tensor_bytes * 8,
        "materialized_payload_tensor_bytes": (
            materialized_index_tensor_bytes + materialized_scale_tensor_bytes
        ),
        "materialized_payload_tensor_scope": (
            "exact storage of code-index and scale tensors in the public payload; "
            "excludes Python containers, control tags, allocator fragmentation, and "
            "persistent codec/codebook/LUT/rotation state"
        ),
        "logical_total_stored_bits": int(compressed.numel_bits()),
        "all_index_tensors_int64": True,
        "index_tensors_bit_packed": False,
        "all_scale_tensors_signed_fp16": True,
    }


def _run_micro(
    torch: Any,
    args: argparse.Namespace,
    specs: Sequence[_MethodSpec],
    model_shape: dict[str, Any],
    dtype: Any,
    device: Any,
) -> dict[str, Any]:
    codecs = {spec.label: spec.codec_factory() for spec in specs}
    labels = [spec.label for spec in specs]
    method_configs, recipe_sha256 = _method_config_records(specs, codecs, model_shape["head_dim"])
    regimes: dict[str, Any] = {}
    regime_defs = (
        ("prefill", args.micro_seq_len, args.seed + 101),
        ("decode", args.decode_seq_len, args.seed + 202),
    )
    quality_rows = None
    parity = None
    payload_audits: dict[str, Any] = {}
    speedup_comparisons = _speedup_comparisons(specs)
    proposal_contract = _validate_proposal_contract(specs, codecs, args.avg_bits)
    for regime_name, seq_len, tensor_seed in regime_defs:
        shape = (
            args.batch_size,
            model_shape["num_key_value_heads"],
            seq_len,
            model_shape["head_dim"],
        )
        key = _make_gpu_tensor(torch, shape, dtype, device, tensor_seed)
        value = _make_gpu_tensor(torch, shape, dtype, device, tensor_seed + 1)
        payloads = {label: codecs[label].encode(key, value) for label in labels}
        fake_quant_effective = {
            label: _fake_quant_effective_implementation(codecs[label], shape[-1])
            for label in labels
        }
        payload_audits[regime_name] = {
            label: _audit_public_payload(torch, payloads[label]) for label in labels
        }

        operations: dict[str, dict[str, Callable[[], Any]]] = {
            "encode": {label: (lambda c=codecs[label]: c.encode(key, value)) for label in labels},
            "decode_prebuilt_payload": {
                label: (lambda c=codecs[label], p=payloads[label]: c.decode(p)) for label in labels
            },
            "roundtrip_explicit_encode_then_decode": {
                label: (lambda c=codecs[label]: c.decode(c.encode(key, value))) for label in labels
            },
            "fake_quant_roundtrip": {
                label: (lambda c=codecs[label]: c.fake_quant_roundtrip(key, value))
                for label in labels
            },
        }
        operation_names = list(operations)
        random.Random(_stable_seed(args.seed, regime_name)).shuffle(operation_names)
        measured = {}
        for operation_index, operation in enumerate(operation_names):
            measured[operation] = _paired_cuda_measurement(
                torch,
                operations[operation],
                warmup=args.micro_warmup,
                repeats=args.micro_repeats,
                inner_loops=args.micro_inner_loops,
                seed=_stable_seed(args.seed + operation_index, f"{regime_name}:{operation}"),
                bootstrap_resamples=args.bootstrap_resamples,
                comparisons=speedup_comparisons,
            )
        regimes[regime_name] = {
            "tensor": {
                "shape": list(key.shape),
                "dtype": str(key.dtype).replace("torch.", ""),
                "device": str(key.device),
                "generated_on_device": True,
                "distribution": "independent standard normal",
                "key_seed": tensor_seed,
                "value_seed": tensor_seed + 1,
                "elements_per_key_value_pair": int(key.numel() + value.numel()),
            },
            "operation_execution_order": operation_names,
            "timed_recipe_sha256": recipe_sha256,
            "payload_api": {
                name: measured[name]
                for name in (
                    "encode",
                    "decode_prebuilt_payload",
                    "roundtrip_explicit_encode_then_decode",
                )
            },
            "research_fake_quant": {
                "warning": (
                    "TurboQuant and AdaptiveHQ use inherited encode->decode; fixed HQ "
                    "recipes are fused when their persisted runtime probe says so. Do not "
                    "treat this category as equal public-API or packed-native kernel timing."
                ),
                "effective_implementations": fake_quant_effective,
                "fake_quant_roundtrip": measured["fake_quant_roundtrip"],
            },
        }
        if regime_name == "prefill":
            query_shape = (
                args.batch_size,
                model_shape["num_attention_heads"],
                min(args.quality_query_len, seq_len),
                model_shape["head_dim"],
            )
            query = _make_gpu_tensor(torch, query_shape, dtype, device, tensor_seed + 2)
            quality_rows, outputs = _quality_metrics(
                torch, codecs, key, value, query, model_shape["gqa_group_size"]
            )
            for label, row in quality_rows.items():
                row["recipe_sha256"] = recipe_sha256[label]
            parity = _parity_checks(
                torch,
                specs,
                codecs,
                outputs,
                key,
                value,
                args.dtype,
                quality_rows,
            )
            regimes[regime_name]["quality_query"] = {
                "shape": list(query.shape),
                "dtype": str(query.dtype).replace("torch.", ""),
                "generated_on_device": True,
                "seed": tensor_seed + 2,
                "gqa_kv_repeat_factor": model_shape["gqa_group_size"],
            }
    assert quality_rows is not None and parity is not None
    rate_contract = _validate_rate_contract(specs, codecs, quality_rows, args.avg_bits)
    return {
        "category": "CUDA codec microbenchmark",
        "api_separation": {
            "payload_api": (
                "public encode/decode with a materialized logical payload; low-bit "
                "indices are unpacked int64 tensors"
            ),
            "research_fake_quant": (
                "dequantized research cache helper; fixed recipes may fuse, AdaptiveHQ "
                "currently uses the inherited payload fallback"
            ),
        },
        "public_payload_scope": dict(PUBLIC_LOGICAL_PAYLOAD_SCOPE),
        "public_payload_runtime_audit": payload_audits,
        "method_configs": method_configs,
        "recipe_sha256": recipe_sha256,
        "recipe_linkage": {
            "payload_quality_and_all_micro_timings_use_same_codec_instances": True,
            "all_rows_fingerprinted_from_declared_and_role_resolved_codec_config": True,
        },
        "proposal_contract": proposal_contract,
        "rate_contract": rate_contract,
        "quality": quality_rows,
        "parity_checks": parity,
        "regimes": regimes,
    }


def _make_cache_factory(
    spec: _MethodSpec,
    config: Any,
    args: argparse.Namespace,
    *,
    head_dim: int,
    expected_recipe_sha256: str,
) -> Callable[[], Any]:
    from onecomp.kvcomp.cache import CompressedDynamicCache

    def factory() -> Any:
        codec = spec.codec_factory()
        observed = _recipe_sha256(_method_config_record(spec, codec, head_dim))
        if observed != expected_recipe_sha256:
            raise RuntimeError(
                f"{spec.label} cache factory changed recipe within one process: "
                f"{observed} != {expected_recipe_sha256}"
            )
        return CompressedDynamicCache(
            codec=codec,
            config=config,
            skip_layers=[],
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
        )

    return factory


def _event_span(torch: Any, fn: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def _e2e_once(
    torch: Any,
    model: Any,
    input_ids: Any,
    cache_factory: Callable[[], Any],
    gen_len: int,
) -> dict[str, Any]:
    cache = cache_factory()
    torch.cuda.synchronize(input_ids.device)
    starting_allocated = float(torch.cuda.memory_allocated(input_ids.device) / (2**20))
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    prompt_len = int(input_ids.shape[1])
    cache_position = torch.arange(prompt_len, device=input_ids.device)

    def prefill() -> Any:
        output = model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        nxt = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        return output, nxt

    ttft_ms, (output, next_token) = _event_span(torch, prefill)
    del output
    generated = []
    decode_step_ms = []
    current = prompt_len
    for _ in range(int(gen_len)):
        position = torch.tensor([current], device=input_ids.device)

        def decode_step() -> Any:
            position_ids = position.unsqueeze(0).expand(input_ids.shape[0], -1)
            out = model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                cache_position=position,
                position_ids=position_ids,
            )
            nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            return out, nxt

        elapsed, (output, new_token) = _event_span(torch, decode_step)
        decode_step_ms.append(elapsed)
        generated.extend(int(token) for token in new_token.reshape(-1).tolist())
        next_token = new_token
        del output
        current += 1
    peak = float(torch.cuda.max_memory_allocated(input_ids.device) / (2**20))
    incremental_peak = peak - starting_allocated
    if incremental_peak < 0.0:
        raise RuntimeError("CUDA peak allocated memory fell below the synchronized baseline")
    compression_ratio = (
        float(cache.compression_ratio()) if hasattr(cache, "compression_ratio") else 1.0
    )
    generated_hash = hashlib.sha256(
        json.dumps(generated, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "ttft_ms": ttft_ms,
        "decode_step_ms": decode_step_ms,
        "decode_mean_ms_per_token": statistics.fmean(decode_step_ms),
        "decode_steady_mean_ms_per_token": statistics.fmean(
            decode_step_ms[1:] if len(decode_step_ms) > 1 else decode_step_ms
        ),
        "peak_allocated_mib": peak,
        "starting_allocated_mib": starting_allocated,
        "incremental_peak_allocated_mib": incremental_peak,
        "reported_compression_ratio": compression_ratio,
        "generated_token_ids_sha256": generated_hash,
    }


def _run_e2e(
    torch: Any,
    args: argparse.Namespace,
    specs: Sequence[_MethodSpec],
    config: Any,
    model_shape: dict[str, Any],
    dtype: Any,
    device: Any,
    context_contract: Mapping[str, Any],
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype,
    }
    if args.attention_implementation:
        load_kwargs["attn_implementation"] = args.attention_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device).eval()
    except torch.cuda.OutOfMemoryError as exc:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        raise RuntimeError(
            "CUDA OOM while loading the formal E2E model; this matrix point "
            f"requires a one-GPU full-model load (free={free_bytes / 2**30:.2f} GiB, "
            f"total={total_bytes / 2**30:.2f} GiB, model={args.model!r})"
        ) from exc
    actual_model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    actual_model_device = str(next(model.parameters()).device)
    actual_attention_implementation = getattr(model.config, "_attn_implementation", None)

    vocab_size = int(getattr(_text_config(config), "vocab_size", 0) or 0)
    if vocab_size < 16:
        raise ValueError(
            "model config has no usable vocab_size for deterministic synthetic tokens"
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 909)
    input_ids = torch.randint(
        low=4,
        high=vocab_size,
        size=(args.batch_size, args.e2e_prompt_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    e2e_codecs = {spec.label: spec.codec_factory() for spec in specs}
    method_configs, recipe_sha256 = _method_config_records(
        specs, e2e_codecs, model_shape["head_dim"]
    )
    factories: dict[str, Callable[[], Any]] = {
        spec.label: _make_cache_factory(
            spec,
            config,
            args,
            head_dim=model_shape["head_dim"],
            expected_recipe_sha256=recipe_sha256[spec.label],
        )
        for spec in specs
    }
    labels = list(factories)
    speedup_comparisons = _speedup_comparisons(specs)

    warmup_orders = _balanced_orders(labels, args.e2e_warmup, args.seed ^ 0x31337)
    with torch.inference_mode():
        for order in warmup_orders:
            for label in order:
                _e2e_once(torch, model, input_ids, factories[label], args.e2e_gen_len)

        orders = _balanced_orders(labels, args.e2e_repeats, args.seed ^ 0xD00D)
        raw = []
        by_method: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for repeat_idx, order in enumerate(orders):
            samples = []
            for position, label in enumerate(order):
                sample = _e2e_once(torch, model, input_ids, factories[label], args.e2e_gen_len)
                sample = {"method": label, "order_position": position, **sample}
                samples.append(sample)
                by_method[label].append(sample)
            raw.append({"repeat": repeat_idx, "order": order, "samples": samples})

    summaries = {}
    for label, samples in by_method.items():
        ttft = [row["ttft_ms"] for row in samples]
        decode = [row["decode_steady_mean_ms_per_token"] for row in samples]
        tok_s = [1000.0 / max(value, 1e-12) for value in decode]
        peak = [row["peak_allocated_mib"] for row in samples]
        incremental_peak = [row["incremental_peak_allocated_mib"] for row in samples]
        summaries[label] = {
            "ttft": _summary(
                ttft,
                unit="ms",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=_stable_seed(args.seed, f"e2e:{label}:ttft"),
            ),
            "decode_steady_latency": _summary(
                decode,
                unit="ms_per_token",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=_stable_seed(args.seed, f"e2e:{label}:decode"),
            ),
            "decode_steady_throughput": _summary(
                tok_s,
                unit="tokens_per_second",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=_stable_seed(args.seed, f"e2e:{label}:tps"),
            ),
            "peak_allocated_memory": _summary(
                peak,
                unit="MiB",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=_stable_seed(args.seed, f"e2e:{label}:peak"),
            ),
            "incremental_peak_allocated_memory": _summary(
                incremental_peak,
                unit="MiB_above_synchronized_start",
                bootstrap_resamples=args.bootstrap_resamples,
                seed=_stable_seed(args.seed, f"e2e:{label}:incremental_peak"),
            ),
            "reported_compression_ratio_samples": [
                row["reported_compression_ratio"] for row in samples
            ],
        }
    del model
    torch.cuda.synchronize()
    paired_speedups = {
        "ttft": _paired_speedup_report(
            raw,
            speedup_comparisons,
            value_key="ttft_ms",
            value_unit="ms",
            bootstrap_resamples=args.bootstrap_resamples,
            seed=_stable_seed(args.seed, "e2e:paired:ttft"),
        ),
        "decode_steady_latency": _paired_speedup_report(
            raw,
            speedup_comparisons,
            value_key="decode_steady_mean_ms_per_token",
            value_unit="ms_per_token",
            bootstrap_resamples=args.bootstrap_resamples,
            seed=_stable_seed(args.seed, "e2e:paired:decode"),
        ),
    }
    return {
        "category": "research_fake_quant_e2e",
        "warning": (
            "This cache immediately dequantizes and stores floating tensors. "
            "It measures research fake-quant overhead, not packed low-bit memory serving."
        ),
        "peak_allocated_memory_scope": {
            "counter": "torch.cuda.max_memory_allocated",
            "unit": "MiB",
            "reset": "once immediately before each timed prefill/decode generation",
            "includes": "resident model plus live tensors and research fake-quant cache",
            "incremental_definition": (
                "peak_allocated_mib minus synchronized starting_allocated_mib"
            ),
            "excludes": "CUDA reserved-but-unallocated bytes",
            "packed_kv_cache_interpretation_allowed": False,
        },
        "native_hybrid_is_full_hayatequant": False,
        "timer": "torch.cuda.Event for TTFT and each decode model step",
        "timer_completion_barrier": "end_event.synchronize before elapsed_time read",
        "model_context_contract": dict(context_contract),
        "input": {
            "source": "deterministic synthetic token ids generated on GPU",
            "shape": list(input_ids.shape),
            "dtype": str(input_ids.dtype).replace("torch.", ""),
            "seed": args.seed + 909,
            "vocab_size": vocab_size,
            "decode_steps": args.e2e_gen_len,
        },
        "model_kv_shape_from_config": model_shape,
        "method_configs": method_configs,
        "recipe_sha256": recipe_sha256,
        "cache_factory_recipe_revalidated_on_every_timed_construction": True,
        "actual_model_parameter_dtype": actual_model_dtype,
        "actual_model_parameter_device": actual_model_device,
        "attention_implementation_requested": args.attention_implementation,
        "attention_implementation_effective": actual_attention_implementation,
        "warmup_orders": warmup_orders,
        "raw_paired_repeats": raw,
        "summary": summaries,
        "paired_speedups": paired_speedups,
    }


def _runtime_provenance(
    torch: Any,
    repo: pathlib.Path,
    args: argparse.Namespace,
    effective_env: dict[str, str],
    onecomp_file: pathlib.Path,
    gpu_before: dict[str, Any],
) -> dict[str, Any]:
    prop = torch.cuda.get_device_properties(torch.device(args.device))
    uuid_value = getattr(prop, "uuid", None)
    uuid_text = str(uuid_value) if uuid_value is not None else None
    smi_rows = gpu_before.get("gpus_seen_by_nvidia_smi", [])
    smi_match = []
    if uuid_text:
        normalised_uuid = uuid_text.removeprefix("b'").removesuffix("'")
        smi_match = [row for row in smi_rows if row.get("uuid") == normalised_uuid]
    if not smi_match:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if visible.isdigit():
            smi_match = [row for row in smi_rows if row.get("index") == visible]
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURMD_NODENAME",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    )
    return {
        "run_uuid": str(uuid.uuid4()),
        "pid": os.getpid(),
        "started_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "git": _git_provenance(repo),
        "onecomp_import_file": str(onecomp_file),
        "packages": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "triton": _package_version("triton"),
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "cuda_device": {
            "requested": args.device,
            "current_logical_index": torch.cuda.current_device(),
            "name": prop.name,
            "uuid_from_torch": uuid_text,
            "total_memory_bytes": int(prop.total_memory),
            "compute_capability": [int(prop.major), int(prop.minor)],
            "multi_processor_count": int(prop.multi_processor_count),
        },
        "selected_gpu_nvidia_smi_match": smi_match,
        "nvidia_smi_before": gpu_before,
        "effective_onecomp_environment": effective_env,
        "selected_environment": {name: os.getenv(name) for name in environment_names},
        "fresh_process_contract": {
            "required": True,
            "granularity": "one {model, dtype, average bit rate, backend, compile mode} per process",
            "reason": (
                "device codebooks, torch.compile/CUDA-graph specialisations, and allocator state "
                "are process-global"
            ),
            "multiple_average_rates_accepted_by_cli": False,
            "compile_all_allowed": False,
            "only_allowed_compile_mode": "decode-only",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    global _RUN_STARTED  # pylint: disable=global-statement
    parser = _build_parser()
    args = parser.parse_args(argv)
    wall_start = time.monotonic()
    _positive_ints(args)
    if _RUN_STARTED:
        raise RuntimeError("this harness must run once per fresh Python process")
    _RUN_STARTED = True

    output_path = pathlib.Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_path}")

    effective_env = _configure_effective_environment(args)
    repo = pathlib.Path(__file__).resolve().parents[2]
    # A worktree does not have its own editable venv.  Prepending the actual
    # script repository prevents an editable install from silently importing a
    # different checkout (a previously observed source of invalid benchmarks).
    sys.path.insert(0, str(repo))

    import torch
    from transformers import AutoConfig
    import onecomp
    from onecomp.kvcomp.eval.provenance import load_run_provenance, run_code_commit

    onecomp_file = pathlib.Path(onecomp.__file__).resolve()
    try:
        onecomp_file.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError(
            f"imported onecomp from {onecomp_file}, expected current worktree {repo}"
        ) from exc
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "actual measurements require a CUDA device; --help/tests remain CPU-safe"
        )
    torch.cuda.set_device(device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model_shape = _resolve_model_shape(config)
    model_capacity = _resolve_model_capacity(config)
    e2e_context_contract = None
    if args.mode in ("e2e", "both"):
        e2e_context_contract = _validate_e2e_context_capacity(
            args.e2e_prompt_len, args.e2e_gen_len, model_capacity
        )
    specs = _method_specs(args.avg_bits)
    gpu_before = _nvidia_smi_snapshot()
    provenance = _runtime_provenance(torch, repo, args, effective_env, onecomp_file, gpu_before)
    immutable_preflight = load_run_provenance()
    code_commit = run_code_commit()
    semantic_config = _semantic_config(args)

    result: dict[str, Any] = {
        "schema_version": 2,
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        # Match the other paper axes so the common immutable-preflight
        # validator can prove model-weight and input-manifest identity.
        "model": pathlib.Path(args.model).name,
        "config": {"model_path": args.model},
        "code_commit": code_commit,
        "semantic_config": semantic_config,
        "run_fingerprint": _run_fingerprint(semantic_config, code_commit),
        "run_provenance": [immutable_preflight] if immutable_preflight else [],
        "protocol": {
            "arguments": vars(args),
            "average_bits": args.avg_bits,
            "dtype": args.dtype,
            "model": args.model,
            "model_evaluation_role": _model_evaluation_role(args.model),
            "model_revision_requested": args.revision,
            "model_config_commit_hash": getattr(config, "_commit_hash", None),
            "model_type": getattr(config, "model_type", None),
            "model_kv_shape": model_shape,
            "model_context_capacity": model_capacity,
            "method_taxonomy": {
                spec.label: {
                    "semantic_role": spec.semantic_role,
                    "fake_quant_implementation": spec.fake_quant_implementation,
                    "proposal_method": spec.proposal_method,
                    "proposal_variant": spec.proposal_variant,
                }
                for spec in specs
            },
            "comparison_rules": {
                "baseline_method": specs[0].label,
                "proposal_methods": [spec.label for spec in specs if spec.proposal_method],
                "method_set_exact": True,
                "do_not_compare_as_equal_api": [
                    "TurboQuant payload/unfused roundtrip",
                    "HQ fused fake_quant_roundtrip",
                    "HQ-Frontier fused fake_quant_roundtrip",
                    "AdaptiveHQ inherited encode_then_decode fake_quant_roundtrip",
                    "RotationHQ fused fake_quant_roundtrip",
                ],
                "packed_native_hybrid_measured": False,
                "packed_native_hybrid_must_not_be_called_full_hayatequant": True,
                "speedup_ratio": "baseline_latency / candidate_latency",
                "timing_pairing_unit": "same counterbalanced repeat",
                "formal_inferential_unit": "independently launched fresh process",
                "technical_repeats_must_not_be_pooled_across_processes": True,
                "directional_speed_claim_allowed": False,
                "speed_reporting_scope": "descriptive ratio and pointwise nominal CI only",
            },
        },
        "provenance": provenance,
    }
    if args.mode in ("micro", "both"):
        result["micro"] = _run_micro(torch, args, specs, model_shape, dtype, device)
    if args.mode in ("e2e", "both"):
        assert e2e_context_contract is not None
        result["e2e"] = _run_e2e(
            torch,
            args,
            specs,
            config,
            model_shape,
            dtype,
            device,
            e2e_context_contract,
        )
    if args.mode == "both" and (
        result["micro"]["method_configs"] != result["e2e"]["method_configs"]
        or result["micro"]["recipe_sha256"] != result["e2e"]["recipe_sha256"]
    ):
        raise RuntimeError("micro and E2E codec recipes diverged within one process")
    from onecomp.kvcomp.codec.hayate import codec as _hayate_codec

    compile_dispatch_created = (
        _hayate_codec._RT_COMPILED is not None
    )  # pylint: disable=protected-access
    if args.compile_mode == "decode-only" and not compile_dispatch_created:
        raise RuntimeError(
            "decode-only was requested, but no compiled HayateQuant dispatch was created"
        )
    if args.compile_mode == "off" and compile_dispatch_created:
        raise RuntimeError("compiled HayateQuant dispatch appeared in compile-mode=off")
    result["protocol"]["compile_runtime_audit"] = {
        "requested_mode": args.compile_mode,
        "compile_all_forbidden": True,
        "prefill_compile_forbidden": True,
        "decode_compiled_dispatch_created": compile_dispatch_created,
        "matches_requested_mode": True,
    }
    result["provenance"]["nvidia_smi_after"] = _nvidia_smi_snapshot()
    result["provenance"]["finished_at_utc"] = _datetime.datetime.now(
        _datetime.timezone.utc
    ).isoformat()
    result["provenance"]["wall_time_seconds"] = time.monotonic() - wall_start
    _atomic_json_write(output_path, result, args.overwrite)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
