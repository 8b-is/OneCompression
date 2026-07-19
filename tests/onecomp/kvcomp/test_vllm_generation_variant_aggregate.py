"""Fail-closed contracts for the native vLLM generation diagnostic aggregate."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "kvcomp-codec"
    / "aggregate_vllm_generation_variants.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("generation_variant_aggregate_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _receipt(variant: str, *, tps: float, bits: int | None = None) -> dict:
    contracts = {
        "auto": {"implementation_label": "vllm_auto_full_precision_kv"},
        "turboquant": {
            "implementation_label": "vllm_native_turboquant_symmetric_scalar_kv"
        },
        "hayate_full": {
            "implementation_label": "native_faithful_packed_fixed_HQ",
            "eligible_for_full_hayatequant_serving_claim": True,
        },
    }
    result = {
        "schema": {
            "name": "onecomp.hayatequant_vllm_native_generation_smoke",
            "version": 1,
        },
        "variant": variant,
        "implementation_contract": contracts[variant],
        "formal_quality_benchmark": False,
        "formal_counterbalanced_speed_benchmark": False,
        "bits": bits,
        "model_path": "/models/Qwen3-0.6B",
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "num_prompts": 2,
        "generated_tokens": 16,
        "elapsed_seconds": 1.0,
        "tokens_per_second": tps,
        "load_seconds": 12.0,
        "cuda_memory_used_after_load_gib": 4.0,
        "cuda_memory_used_after_generate_gib": 4.1,
        "torch_peak_reserved_gib": 5.0,
        "model_context_contract": {
            "prompt_token_lengths": [3, 5],
            "max_new_tokens": 8,
        },
        "generations": [
            {"prompt": "first", "text": "output", "tokens": 8},
            {"prompt": "second", "text": "answer", "tokens": 8},
        ],
    }
    if variant == "hayate_full":
        result["runtime_evidence"] = {
            "custom_packed_store_and_decode_execution_proved": True,
            "events": [
                {"event": "packed_hq_store_launched"},
                {"event": "packed_hq_decode_attention_launched"},
            ],
        }
    return result


def _write(tmp_path: Path, name: str, document: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(__import__("json").dumps(document), encoding="utf-8")
    return path


def test_aggregate_accepts_only_comparable_fp16_tq_and_packed_hq(tmp_path):
    module = _module()
    inputs = [
        ("fp16", _write(tmp_path, "fp16", _receipt("auto", tps=100.0))),
        ("tq3", _write(tmp_path, "tq3", _receipt("turboquant", tps=120.0, bits=3))),
        ("hq3", _write(tmp_path, "hq3", _receipt("hayate_full", tps=130.0, bits=3))),
    ]
    result = module.aggregate(inputs)
    assert result["full_precision_baseline_label"] == "fp16"
    assert [row["kind"] for row in result["rows"]] == [
        "full_precision",
        "turboquant",
        "hayatequant_packed",
    ]
    assert result["rows"][2]["generation_speed_relative_to_full_precision"] == 1.3
    assert "TapeKV" in result["excludes"][0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document.update(
                variant="hayate",
                implementation_contract={
                    "implementation_label": "native_hybrid_hayate_block_vq_keys_turboquant_scalar_values"
                },
            ),
            "Tape/reference/hybrid",
        ),
        (
            lambda document: document["runtime_evidence"].update(events=[]),
            "missing packed HQ runtime events",
        ),
        (
            lambda document: document["model_context_contract"].update(max_new_tokens=9),
            "does not match",
        ),
        (
            lambda document: document.update(formal_quality_benchmark=True),
            "ambiguously labelled as a formal quality",
        ),
    ],
)
def test_aggregate_rejects_noncomparable_or_unsupported_rows(tmp_path, mutate, message):
    module = _module()
    fp16 = _write(tmp_path, "fp16", _receipt("auto", tps=100.0))
    hq_document = _receipt("hayate_full", tps=130.0, bits=3)
    mutate(hq_document)
    hq = _write(tmp_path, "hq", hq_document)
    with pytest.raises(module.VariantAggregateError, match=message):
        module.aggregate([("fp16", fp16), ("candidate", hq)])


def test_parser_refuses_relative_input_path():
    module = _module()
    with pytest.raises(__import__("argparse").ArgumentTypeError, match="absolute"):
        module._parse_input("fp16=relative.json")


def test_aggregate_rejects_duplicate_baselines(tmp_path):
    module = _module()
    first = _write(tmp_path, "one", _receipt("auto", tps=100.0))
    second = _write(tmp_path, "two", copy.deepcopy(_receipt("auto", tps=100.0)))
    with pytest.raises(module.VariantAggregateError, match="exactly one"):
        module.aggregate([("first", first), ("second", second)])
