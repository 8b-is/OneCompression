#!/usr/bin/env python3
"""Render a fail-closed LaTeX table from the three-path vLLM smoke suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class SmokeRenderError(RuntimeError):
    """The GPU smoke receipt is incomplete or internally inconsistent."""


EXPECTED_PHASES = ("taperkv", "hayatequant", "hayatequant_taperkv")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeRenderError(f"{label} must be an object")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SmokeRenderError(f"{label} must be a positive integer")
    return value


def _read_receipt(suite_path: Path, phase: dict[str, Any]) -> dict[str, Any]:
    receipt_path = Path(str(phase.get("receipt", ""))).resolve()
    if receipt_path.parent != suite_path.parent.resolve():
        raise SmokeRenderError("phase receipt must be adjacent to suite.json")
    try:
        return _object(json.loads(receipt_path.read_text(encoding="utf-8")), receipt_path.name)
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeRenderError(f"cannot read phase receipt {receipt_path}") from exc


def _validate_generation_evidence(receipt: dict[str, Any], label: str) -> None:
    num_prompts = _positive_integer(receipt.get("num_prompts"), f"{label}.num_prompts")
    generated_tokens = _positive_integer(
        receipt.get("generated_tokens"), f"{label}.generated_tokens"
    )
    generations = receipt.get("generations")
    if not isinstance(generations, list) or len(generations) != num_prompts:
        raise SmokeRenderError(f"{label} generations do not cover every submitted prompt")
    token_sum = 0
    for index, generation in enumerate(generations):
        row = _object(generation, f"{label}.generations[{index}]")
        token_sum += _positive_integer(row.get("tokens"), f"{label}.generations[{index}].tokens")
        if not isinstance(row.get("text"), str):
            raise SmokeRenderError(f"{label} generation text must be a string")
    if token_sum != generated_tokens:
        raise SmokeRenderError(f"{label} generated-token total is inconsistent")


def _validate_runtime_evidence(
    suite_path: Path,
    receipt: dict[str, Any],
    *,
    label: str,
    required_events: set[str],
    combined: bool | None = None,
) -> None:
    record = _object(receipt.get("runtime_evidence"), f"{label}.runtime_evidence")
    evidence_path = Path(str(record.get("path", ""))).resolve()
    if evidence_path.parent != suite_path.parent.resolve():
        raise SmokeRenderError("runtime evidence must be adjacent to suite.json")
    try:
        raw = evidence_path.read_bytes()
    except OSError as exc:
        raise SmokeRenderError(f"cannot read runtime evidence for {label}") from exc
    if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise SmokeRenderError(f"runtime evidence digest mismatch for {label}")
    try:
        events = [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as exc:
        raise SmokeRenderError(f"invalid runtime evidence JSON for {label}") from exc
    if not events or events != record.get("events"):
        raise SmokeRenderError(f"runtime evidence payload mismatch for {label}")
    expected_schema = {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1}
    if any(
        not isinstance(event, dict) or event.get("schema") != expected_schema for event in events
    ):
        raise SmokeRenderError(f"unexpected runtime evidence schema for {label}")
    observed = {event.get("event") for event in events}
    if not required_events.issubset(observed):
        raise SmokeRenderError(f"runtime evidence is missing custom backend events for {label}")
    if combined is not None and not any(
        event.get("event") == "logical_paged_cache_regraded"
        and event.get("hayate_then_taper") is combined
        and isinstance(event.get("maximum_pages_per_request"), int)
        and event["maximum_pages_per_request"] > 1
        for event in events
    ):
        raise SmokeRenderError(f"runtime evidence does not prove selected TapeKV path for {label}")


def validate(suite_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        raw = suite_path.read_bytes()
        suite = _object(json.loads(raw), "suite")
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeRenderError("suite is unreadable or not valid JSON") from exc
    if suite.get("schema") != {
        "name": "onecomp.kvcomp_vllm_inference_smoke_suite",
        "version": 1,
    }:
        raise SmokeRenderError("unexpected suite schema")
    if suite.get("passed") is not True:
        raise SmokeRenderError("suite did not pass all inference paths")
    phases = suite.get("phases")
    if not isinstance(phases, list) or len(phases) != len(EXPECTED_PHASES):
        raise SmokeRenderError("suite must contain the canonical three phases in order")
    phase_objects = [_object(phase, "phase") for phase in phases]
    if [phase.get("name") for phase in phase_objects] != list(EXPECTED_PHASES):
        raise SmokeRenderError("suite must contain the canonical three phases in order")
    receipts = {}
    for phase in phase_objects:
        returncode = phase.get("returncode")
        if phase.get("passed") is not True or isinstance(returncode, bool) or returncode != 0:
            raise SmokeRenderError(f"phase {phase.get('name')} is not successful")
        receipts[phase["name"]] = _read_receipt(suite_path, phase)

    taper = receipts["taperkv"]
    combo = receipts["hayatequant_taperkv"]
    for receipt, implementation in (
        (taper, "fp16_reference_fake_quant"),
        (combo, "hayatequant_then_taper_fp16_reference"),
    ):
        if receipt.get("schema_version") != 1 or receipt.get("implementation") != implementation:
            raise SmokeRenderError(f"invalid TapeKV implementation receipt: {implementation}")
        if receipt.get("eligible_for_packed_memory_or_speed_table") is not False:
            raise SmokeRenderError("TapeKV reference receipt is mislabelled as packed evidence")
        _validate_generation_evidence(receipt, implementation)
        if (
            receipt.get("exercises_batched_requests") is not True
            or receipt.get("exercises_multi_page_prompt") is not True
        ):
            raise SmokeRenderError(
                "TapeKV smoke must cover batched requests and multiple KV pages"
            )
        _validate_runtime_evidence(
            suite_path,
            receipt,
            label=implementation,
            required_events={"logical_paged_cache_regraded"},
            combined=implementation == "hayatequant_then_taper_fp16_reference",
        )

    hayate = receipts["hayatequant"]
    if (
        hayate.get("schema")
        != {
            "name": "onecomp.hayatequant_vllm_native_generation_smoke",
            "version": 1,
        }
        or hayate.get("variant") != "hayate_full"
    ):
        raise SmokeRenderError("HayateQuant receipt is not the full native variant")
    contract = _object(hayate.get("implementation_contract"), "implementation_contract")
    if (
        contract.get("implementation_label") != "native_faithful_packed_fixed_HQ"
        or contract.get("native_hybrid_is_full_hayatequant") is not True
        or contract.get("eligible_for_full_hayatequant_serving_claim") is not True
    ):
        raise SmokeRenderError("HayateQuant full-native implementation contract is incomplete")
    if (
        hayate.get("formal_quality_benchmark") is not False
        or hayate.get("formal_counterbalanced_speed_benchmark") is not False
    ):
        raise SmokeRenderError("generation smoke must not be labelled as formal evidence")
    _validate_generation_evidence(hayate, "hayatequant")
    _validate_runtime_evidence(
        suite_path,
        hayate,
        label="hayatequant",
        required_events={
            "packed_hq_store_launched",
            "packed_hq_decode_attention_launched",
        },
    )

    bits = hayate.get("bits")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (3, 4):
        raise SmokeRenderError("full-native smoke rate must be 3 or 4")

    model = suite.get("model")
    if not isinstance(model, str) or not model:
        raise SmokeRenderError("suite model must be a nonempty string")
    if (
        taper.get("model") != model
        or combo.get("model") != model
        or hayate.get("model_path") != model
    ):
        raise SmokeRenderError("phase model identities differ")
    max_model_len = _positive_integer(suite.get("max_model_len"), "max_model_len")
    max_tokens = _positive_integer(suite.get("max_tokens"), "max_tokens")
    if max_tokens >= max_model_len:
        raise SmokeRenderError("max_tokens must be smaller than max_model_len")
    packages = _object(_object(suite.get("provenance"), "provenance").get("packages"), "packages")
    if not isinstance(packages.get("vllm"), str) or not packages["vllm"]:
        raise SmokeRenderError("provenance must contain a nonempty vLLM version")
    suite["_sha256"] = hashlib.sha256(raw).hexdigest()
    return suite, receipts


def _tex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def render(suite_path: Path) -> str:
    suite, receipts = validate(suite_path)
    hayate = receipts["hayatequant"]
    package_versions = _object(
        _object(suite.get("provenance"), "provenance").get("packages"), "packages"
    )
    model_name = Path(str(suite["model"])).name
    suite_hash = suite["_sha256"]
    breakable_hash = r"\allowbreak{}".join(
        suite_hash[index : index + 8] for index in range(0, len(suite_hash), 8)
    )
    return "\n".join(
        [
            "% Generated only from a passing three-path GPU smoke receipt.",
            rf"\newcommand{{\KVCompVLLMSmokeSuiteSHA}}{{\texttt{{{breakable_hash}}}}}",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{GPU generation smoke for the three vLLM integration paths. "
            r"This is implementation evidence, not a formal quality or speed comparison.}",
            r"\label{tab:vllm-smoke}",
            r"\begin{tabular}{lll}",
            r"\toprule",
            r"path & resident KV representation & result \\",
            r"\midrule",
            r"TapeKV & model-dtype paged cache; semantic age regrading & pass \\",
            rf"faithful HQ (rate {hayate['bits']}) & packed fixed-HQ K/V blocks on eligible layers & pass \\",
            r"HQ $\rightarrow$ TapeKV & model-dtype paged cache; HQ-on-arrival then age regrading & pass \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            rf"\noindent The fail-closed smoke used \texttt{{{_tex(model_name)}}}, "
            rf"vLLM~\texttt{{{_tex(package_versions.get('vllm'))}}}, a maximum context of "
            rf"{suite['max_model_len']} tokens, and {suite['max_tokens']} generated tokens per "
            r"request. Each backend ran in a fresh process. The full-HQ row exercised packed "
            r"key and value block-VQ on every compression-eligible layer while retaining "
            r"vLLM's boundary-layer protection; the two TapeKV rows deliberately retained the "
            r"model-dtype physical cache and are ineligible for packed-memory or throughput claims.",
            "",
            rf"\noindent\textit{{Receipt identity:}} \KVCompVLLMSmokeSuiteSHA.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(args.suite.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
