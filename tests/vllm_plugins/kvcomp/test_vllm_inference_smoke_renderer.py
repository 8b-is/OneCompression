"""Fail-closed contracts for rendering GPU smoke evidence into LaTeX."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "kvcomp-codec"
    / "render_vllm_inference_smoke_latex.py"
)


def _module():
    return runpy.run_path(str(SCRIPT), run_name="kvcomp_smoke_renderer_test")


def _write_receipts(tmp_path: Path) -> tuple[Path, dict]:
    model = "/models/Qwen_3%"
    generation_evidence = {
        "num_prompts": 2,
        "generated_tokens": 4,
        "generations": [
            {"prompt": "a", "text": "c", "tokens": 2},
            {"prompt": "b", "text": "d", "tokens": 2},
        ],
    }
    receipts = {
        "taperkv": {
            "schema_version": 1,
            "implementation": "fp16_reference_fake_quant",
            "eligible_for_packed_memory_or_speed_table": False,
            "model": model,
            "exercises_batched_requests": True,
            "exercises_multi_page_prompt": True,
            **generation_evidence,
        },
        "hayatequant": {
            "schema": {
                "name": "onecomp.hayatequant_vllm_native_generation_smoke",
                "version": 1,
            },
            "variant": "hayate_full",
            "bits": 3,
            "implementation_contract": {
                "implementation_label": "native_faithful_packed_fixed_HQ",
                "native_hybrid_is_full_hayatequant": True,
                "eligible_for_full_hayatequant_serving_claim": True,
            },
            "formal_quality_benchmark": False,
            "formal_counterbalanced_speed_benchmark": False,
            "model_path": model,
            **generation_evidence,
        },
        "hayatequant_taperkv": {
            "schema_version": 1,
            "implementation": "hayatequant_then_taper_fp16_reference",
            "eligible_for_packed_memory_or_speed_table": False,
            "model": model,
            "exercises_batched_requests": True,
            "exercises_multi_page_prompt": True,
            **generation_evidence,
        },
    }
    schema = {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1}
    evidence_events = {
        "taperkv": [
            {
                "schema": schema,
                "event": "logical_paged_cache_regraded",
                "hayate_then_taper": False,
                "maximum_pages_per_request": 2,
            }
        ],
        "hayatequant": [
            {"schema": schema, "event": "packed_hq_store_launched"},
            {"schema": schema, "event": "packed_hq_decode_attention_launched"},
        ],
        "hayatequant_taperkv": [
            {
                "schema": schema,
                "event": "logical_paged_cache_regraded",
                "hayate_then_taper": True,
                "maximum_pages_per_request": 2,
            }
        ],
    }
    phases = []
    for name, receipt in receipts.items():
        evidence_path = tmp_path / f"{name}.json.runtime.jsonl"
        evidence_raw = b"".join(
            (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for event in evidence_events[name]
        )
        evidence_path.write_bytes(evidence_raw)
        receipt["runtime_evidence"] = {
            "path": str(evidence_path),
            "sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "events": evidence_events[name],
        }
        receipt_path = tmp_path / f"{name}.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        phases.append(
            {
                "name": name,
                "returncode": 0,
                "passed": True,
                "receipt": str(receipt_path),
            }
        )
    suite = {
        "schema": {
            "name": "onecomp.kvcomp_vllm_inference_smoke_suite",
            "version": 1,
        },
        "passed": True,
        "model": model,
        "max_model_len": 256,
        "max_tokens": 8,
        "provenance": {"packages": {"vllm": "0.20.0"}},
        "phases": phases,
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    return suite_path, suite


def test_renderer_accepts_only_canonical_passing_suite(tmp_path):
    module = _module()
    suite_path, _suite = _write_receipts(tmp_path)
    output = module["render"](suite_path)
    assert "faithful HQ (rate 3)" in output
    assert r"Qwen\_3\%" in output
    assert "not a formal quality or speed comparison" in output


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda suite: suite.update(passed=False), "did not pass"),
        (lambda suite: suite["phases"].reverse(), "canonical three phases"),
        (
            lambda suite: suite["phases"][1].update(returncode=False),
            "not successful",
        ),
        (lambda suite: suite.update(max_tokens=256), "must be smaller"),
    ],
)
def test_renderer_rejects_invalid_suite_metadata(tmp_path, mutation, message):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    invalid = copy.deepcopy(suite)
    mutation(invalid)
    suite_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match=message):
        module["render"](suite_path)


def test_renderer_rejects_hybrid_as_full_hayatequant(tmp_path):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    hayate_path = Path(suite["phases"][1]["receipt"])
    receipt = json.loads(hayate_path.read_text(encoding="utf-8"))
    receipt["variant"] = "hayate_hybrid"
    hayate_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="full native variant"):
        module["render"](suite_path)


def test_renderer_rejects_receipt_outside_suite_directory(tmp_path):
    module = _module()
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    suite_path, suite = _write_receipts(suite_dir)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    suite["phases"][0]["receipt"] = str(external)
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="adjacent"):
        module["render"](suite_path)


def test_renderer_rejects_single_page_taper_smoke(tmp_path):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    taper_path = Path(suite["phases"][0]["receipt"])
    receipt = json.loads(taper_path.read_text(encoding="utf-8"))
    receipt["exercises_multi_page_prompt"] = False
    taper_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="multiple KV pages"):
        module["render"](suite_path)


def test_renderer_rejects_inconsistent_generated_token_total(tmp_path):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    hayate_path = Path(suite["phases"][1]["receipt"])
    receipt = json.loads(hayate_path.read_text(encoding="utf-8"))
    receipt["generated_tokens"] = 99
    hayate_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="token total"):
        module["render"](suite_path)


def test_renderer_rejects_tampered_runtime_evidence(tmp_path):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    taper_path = Path(suite["phases"][0]["receipt"])
    receipt = json.loads(taper_path.read_text(encoding="utf-8"))
    Path(receipt["runtime_evidence"]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="digest mismatch"):
        module["render"](suite_path)


def test_renderer_rejects_missing_packed_decode_evidence(tmp_path):
    module = _module()
    suite_path, suite = _write_receipts(tmp_path)
    hayate_path = Path(suite["phases"][1]["receipt"])
    receipt = json.loads(hayate_path.read_text(encoding="utf-8"))
    evidence_path = Path(receipt["runtime_evidence"]["path"])
    events = receipt["runtime_evidence"]["events"][:1]
    raw = b"".join(
        (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for event in events
    )
    evidence_path.write_bytes(raw)
    receipt["runtime_evidence"]["events"] = events
    receipt["runtime_evidence"]["sha256"] = hashlib.sha256(raw).hexdigest()
    hayate_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(module["SmokeRenderError"], match="missing custom backend events"):
        module["render"](suite_path)
