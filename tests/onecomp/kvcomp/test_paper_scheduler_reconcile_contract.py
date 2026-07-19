"""Regression tests for strict Slurm accounting parsing in paper reconciliation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


CODEC_BENCHMARK = Path(__file__).resolve().parents[3] / "benchmark" / "kvcomp-codec"
SCRIPT = CODEC_BENCHMARK / "reconcile_hayatequant_paper_jobs.py"


def _module():
    sys.path.insert(0, str(CODEC_BENCHMARK))
    try:
        spec = importlib.util.spec_from_file_location("paper_scheduler_reconcile_tested", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _row(*, comment: str) -> bytes:
    return (
        "71234|researcher|hq-f-formal-reconcile|CANCELLED by 10003|0:0|"
        f"2026-07-19T07:26:10|None|2026-07-19T10:18:02|{comment}\n"
    ).encode()


def test_sacct_parser_retains_an_empty_terminal_comment_field():
    module = _module()
    row = module._parse_sacct(_row(comment=""), expected_job_id="71234")
    assert row["Comment"] == ""
    assert row["State"] == "CANCELLED"
    assert row["StateRaw"] == "CANCELLED by 10003"


def test_sacct_parser_rejects_a_row_that_lacks_the_comment_column():
    module = _module()
    malformed = _row(comment="").rstrip(b"\n|") + b"\n"
    with pytest.raises(module.ReconcileError, match="8 fields, expected 9"):
        module._parse_sacct(malformed, expected_job_id="71234")
