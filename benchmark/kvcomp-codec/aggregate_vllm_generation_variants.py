#!/usr/bin/env python3
"""Aggregate comparable native vLLM generation receipts without overclaiming.

This accepts only the physical serving rows emitted by
``hayatequant_vllm_native_generate.py``: full-precision vLLM, upstream
TurboQuant, and faithful packed fixed HayateQuant.  TapeKV and HQ+TapeKV use
an FP16/BF16 reference cache, so they are intentionally rejected here and
must be presented in their separate semantic-generation receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class VariantAggregateError(RuntimeError):
    """A proposed native-generation comparison violates its common contract."""


_SCHEMA = {"name": "onecomp.kvcomp_vllm_generation_variants", "version": 1}
_GENERATION_SCHEMA = {
    "name": "onecomp.hayatequant_vllm_native_generation_smoke",
    "version": 1,
}
_FULL_HQ_EVENTS = {"packed_hq_store_launched", "packed_hq_decode_attention_launched"}


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantAggregateError(f"cannot read generation receipt {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise VariantAggregateError(f"generation receipt {path} is not a JSON object")
    return document


def _parse_input(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--input must be LABEL=/absolute/path.json")
    if any(char.isspace() for char in label):
        raise argparse.ArgumentTypeError("input labels must not contain whitespace")
    raw = Path(raw_path).expanduser()
    if not raw.is_absolute():
        raise argparse.ArgumentTypeError("input paths must be absolute")
    return label, raw.resolve()


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VariantAggregateError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise VariantAggregateError(f"{label} must be finite and positive")
    return value


def _generation_digest(document: dict[str, Any], label: str) -> str:
    rows = document.get("generations")
    if not isinstance(rows, list) or not rows:
        raise VariantAggregateError(f"{label} has no generations")
    normalized = []
    token_total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise VariantAggregateError(f"{label}.generations[{index}] is not an object")
        prompt, text, tokens = row.get("prompt"), row.get("text"), row.get("tokens")
        if not isinstance(prompt, str) or not isinstance(text, str):
            raise VariantAggregateError(f"{label}.generations[{index}] has non-text fields")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise VariantAggregateError(f"{label}.generations[{index}].tokens is invalid")
        normalized.append({"prompt": prompt, "text": text, "tokens": tokens})
        token_total += tokens
    if document.get("generated_tokens") != token_total:
        raise VariantAggregateError(f"{label} generated-token total does not match rows")
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_kind(document: dict[str, Any], label: str) -> str:
    if document.get("schema") != _GENERATION_SCHEMA:
        raise VariantAggregateError(f"{label} has the wrong native-generation schema")
    variant = document.get("variant")
    contract = document.get("implementation_contract")
    if not isinstance(contract, dict):
        raise VariantAggregateError(f"{label} omits its implementation contract")
    if variant == "auto":
        if contract.get("implementation_label") != "vllm_auto_full_precision_kv":
            raise VariantAggregateError(f"{label} is not the FP16/BF16 vLLM baseline")
        return "full_precision"
    if variant == "turboquant":
        if contract.get("implementation_label") != "vllm_native_turboquant_symmetric_scalar_kv":
            raise VariantAggregateError(f"{label} is not upstream native TurboQuant")
        return "turboquant"
    if variant == "hayate_full":
        evidence = document.get("runtime_evidence")
        if (
            contract.get("eligible_for_full_hayatequant_serving_claim") is not True
            or not isinstance(evidence, dict)
            or evidence.get("custom_packed_store_and_decode_execution_proved") is not True
        ):
            raise VariantAggregateError(f"{label} does not prove faithful packed HQ execution")
        events = evidence.get("events")
        if not isinstance(events, list) or {
            event.get("event") for event in events if isinstance(event, dict)
        } & _FULL_HQ_EVENTS != _FULL_HQ_EVENTS:
            raise VariantAggregateError(f"{label} is missing packed HQ runtime events")
        return "hayatequant_packed"
    raise VariantAggregateError(
        f"{label} has variant {variant!r}; Tape/reference/hybrid rows belong in a separate report"
    )


def aggregate(inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(inputs) < 2:
        raise VariantAggregateError("at least two native generation receipts are required")
    labels = [label for label, _path in inputs]
    if len(set(labels)) != len(labels):
        raise VariantAggregateError("input labels must be unique")

    rows = []
    shared: dict[str, Any] | None = None
    for label, path in inputs:
        document = _load(path)
        kind = _implementation_kind(document, label)
        if document.get("formal_quality_benchmark") is not False:
            raise VariantAggregateError(f"{label} is ambiguously labelled as a formal quality result")
        if document.get("formal_counterbalanced_speed_benchmark") is not False:
            raise VariantAggregateError(f"{label} is ambiguously labelled as formal speed")
        generation_digest = _generation_digest(document, label)
        comparison_fields = {
            "model_path": document.get("model_path"),
            "dtype": document.get("dtype"),
            "max_model_len": document.get("max_model_len"),
            "num_prompts": document.get("num_prompts"),
            "generated_tokens": document.get("generated_tokens"),
            "prompt_token_lengths": (document.get("model_context_contract") or {}).get(
                "prompt_token_lengths"
            ),
            "max_new_tokens": (document.get("model_context_contract") or {}).get(
                "max_new_tokens"
            ),
        }
        if any(value is None for value in comparison_fields.values()):
            raise VariantAggregateError(f"{label} omits a common-generation comparison field")
        if shared is None:
            shared = comparison_fields
        elif comparison_fields != shared:
            raise VariantAggregateError(
                f"{label} does not match the FP16/TQ/HQ generation workload: {comparison_fields}"
            )
        rows.append(
            {
                "label": label,
                "kind": kind,
                "variant": document["variant"],
                "bits": document.get("bits"),
                "generation_tokens_per_second": _positive_number(
                    document.get("tokens_per_second"), f"{label}.tokens_per_second"
                ),
                "generation_elapsed_seconds": _positive_number(
                    document.get("elapsed_seconds"), f"{label}.elapsed_seconds"
                ),
                "load_seconds": _positive_number(document.get("load_seconds"), f"{label}.load_seconds"),
                "cuda_memory_used_after_load_gib": _positive_number(
                    document.get("cuda_memory_used_after_load_gib"),
                    f"{label}.cuda_memory_used_after_load_gib",
                ),
                "cuda_memory_used_after_generate_gib": _positive_number(
                    document.get("cuda_memory_used_after_generate_gib"),
                    f"{label}.cuda_memory_used_after_generate_gib",
                ),
                "torch_peak_reserved_gib": float(document.get("torch_peak_reserved_gib", 0.0)),
                "generation_sha256": generation_digest,
                "receipt": str(path),
                "provenance": document.get("provenance"),
            }
        )

    fp16 = [row for row in rows if row["kind"] == "full_precision"]
    if len(fp16) != 1:
        raise VariantAggregateError("comparison requires exactly one full-precision baseline")
    baseline = fp16[0]
    for row in rows:
        row["generation_speed_relative_to_full_precision"] = round(
            row["generation_tokens_per_second"] / baseline["generation_tokens_per_second"], 5
        )
        row["post_load_memory_delta_vs_full_precision_gib"] = round(
            row["cuda_memory_used_after_load_gib"]
            - baseline["cuda_memory_used_after_load_gib"],
            5,
        )
    return {
        "schema": _SCHEMA,
        "formal_quality_benchmark": False,
        "formal_counterbalanced_speed_benchmark": False,
        "serving_comparison_scope": (
            "one warmed fresh vLLM process per physical serving variant; diagnostic generation "
            "comparison only, not an inferential speed claim"
        ),
        "excludes": [
            "TapeKV and HQ+TapeKV FP16/BF16 reference paths",
            "historical key-HQ/value-TQ hybrid",
            "formal quality conclusions",
        ],
        "shared_workload": shared,
        "full_precision_baseline_label": baseline["label"],
        "rows": rows,
    }


def _markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Native vLLM generation variants (diagnostic)",
        "",
        "This is a warmed, one-process-per-variant diagnostic. It is not a formal quality "
        "or inferential speed benchmark; TapeKV reference results are intentionally excluded.",
        "",
        "| variant | implementation | rate | token/s | vs FP16 | load s | load GPU GiB | text SHA-256 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["rows"]:
        bits = "FP16/BF16" if row["bits"] is None else str(row["bits"])
        lines.append(
            "| {label} | {kind} | {bits} | {tps:.3f} | {relative:.3f}x | {load:.3f} | "
            "{memory:.3f} | `{digest}` |".format(
                label=row["label"],
                kind=row["kind"],
                bits=bits,
                tps=row["generation_tokens_per_second"],
                relative=row["generation_speed_relative_to_full_precision"],
                load=row["load_seconds"],
                memory=row["cuda_memory_used_after_load_gib"],
                digest=row["generation_sha256"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    markdown_output = None if args.markdown_output is None else args.markdown_output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output}")
    if markdown_output is not None and markdown_output.exists():
        raise FileExistsError(f"refusing to overwrite Markdown report: {markdown_output}")
    document = aggregate(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(_markdown(document), encoding="utf-8")
    print(json.dumps(document, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
