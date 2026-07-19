"""CPU contracts for the isolated vLLM inference smoke orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import subprocess

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "kvcomp-codec"
    / "run_vllm_inference_smoke_suite.py"
)


def _module():
    return runpy.run_path(str(SCRIPT), run_name="kvcomp_smoke_suite_test")


def test_suite_defines_three_fresh_process_inference_paths(tmp_path):
    module = _module()
    args = argparse.Namespace(
        model=tmp_path / "model",
        output_dir=tmp_path / "receipts",
        max_model_len=512,
        max_tokens=8,
    )
    specs = module["_command_specs"](args)
    assert [name for name, _command, _receipt in specs] == [
        "taperkv",
        "hayatequant",
        "hayatequant_taperkv",
    ]
    assert "hayate_full" in specs[1][1]
    assert "--hayate-bits" in specs[2][1]
    assert len({receipt for _name, _command, receipt in specs}) == 3


def test_suite_refuses_to_mix_stale_receipts(tmp_path):
    module = _module()
    output_dir = tmp_path / "receipts"
    output_dir.mkdir()
    (output_dir / "taperkv.json").write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        model=tmp_path / "model",
        output_dir=output_dir,
        max_model_len=512,
        max_tokens=8,
    )
    with pytest.raises(ValueError, match="new or empty"):
        module["run"](args)


def test_suite_renderer_uses_shared_fail_closed_validator(tmp_path, monkeypatch):
    module = _module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    suite = tmp_path / "suite.json"
    output = tmp_path / "vllm_inference_smoke.tex"
    result = module["_render_validated_suite"](suite, output)
    assert result.returncode == 0
    command, kwargs = calls[0]
    assert command[1] == str(module["RENDERER"])
    assert command[-4:] == ["--suite", str(suite), "--output", str(output)]
    assert kwargs["check"] is False
    assert kwargs["capture_output"] is True
