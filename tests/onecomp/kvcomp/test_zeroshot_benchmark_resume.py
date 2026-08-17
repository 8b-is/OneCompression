"""Checkpoint/resume tests for the paper zero-shot driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/zeroshot_benchmark.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("kvcomp_zeroshot_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task_result(task: str, score: float) -> dict:
    return {
        "task": task,
        "n": 2,
        "acc": score,
        "acc_norm": score,
        "primary_score": score,
    }


def test_eval_method_resumes_completed_tasks_and_checkpoints_new_task(monkeypatch):
    driver = _load_driver()
    calls = []

    class Result:
        def __init__(self, task):
            self.task = task

        def as_dict(self):
            return _task_result(self.task, 0.75)

    def fake_accuracy(_model, _tok, *, task, **_kwargs):
        calls.append(task)
        return Result(task)

    monkeypatch.setattr(driver, "zeroshot_accuracy", fake_accuracy)
    checkpoints = []
    args = SimpleNamespace(
        tasks=["arc_easy", "piqa"],
        start=0,
        limit=None,
        sample_seed=2027,
    )
    result = driver._eval_method(
        object(),
        object(),
        args,
        cache_factory=lambda: None,
        existing={"tasks": {"arc_easy": _task_result("arc_easy", 0.5)}},
        on_task=checkpoints.append,
    )

    assert calls == ["piqa"]
    assert len(checkpoints) == 1
    assert result["complete"] is True
    assert result["n_total"] == 4
    assert result["avg_primary_score"] == 0.625


def test_method_summary_uses_requested_task_set_for_shards():
    driver = _load_driver()
    summary = driver._method_summary(
        {"hellaswag": _task_result("hellaswag", 0.5)},
        ["hellaswag"],
    )
    assert summary["complete"] is True
