"""CLI and scientific-identity regressions for the zero-shot driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/zeroshot_benchmark.py"
SPEC = importlib.util.spec_from_file_location("zeroshot_benchmark_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _parse(monkeypatch, tmp_path: Path, *, bits: tuple[int, ...], methods: tuple[str, ...]):
    argv = [
        str(SCRIPT),
        "--model-path",
        str(tmp_path / "model"),
        "--output-root",
        str(tmp_path / "results"),
        "--bits",
        *(str(bit) for bit in bits),
        "--methods",
        *methods,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return benchmark.parse_args()


def test_cli_preserves_canonical_blockquant_request_until_rate_filter(monkeypatch, tmp_path):
    args = _parse(
        monkeypatch,
        tmp_path,
        bits=(2, 3, 4),
        methods=("BlockQuant-paper", "HQ", "BlockQuant-paper", "TurboQuant-paper"),
    )

    assert args.methods == ["BlockQuant-paper", "HQ", "TurboQuant-paper"]
    assert "BlockQuant-paper" not in benchmark._method_list(args, bits=2)
    assert "BlockQuant-paper" in benchmark._method_list(args, bits=3)
    assert "BlockQuant-paper" not in benchmark._method_list(args, bits=4)


def test_device_is_bound_into_config_and_run_fingerprint(monkeypatch, tmp_path):
    args = _parse(monkeypatch, tmp_path, bits=(3,), methods=("BlockQuant-paper", "HQ"))
    monkeypatch.setattr(benchmark, "run_code_commit", lambda: "test-commit")

    config = benchmark._benchmark_config(args)
    assert config["device"] == "cuda:0"
    fingerprint = benchmark._fingerprint(config)
    changed = {**config, "device": "cuda:1"}
    assert benchmark._fingerprint(changed) != fingerprint


@pytest.mark.parametrize(
    ("rate", "requested"),
    [
        (2, ("TurboQuant-paper", "FibQuant-paper", "HQ", "AdaptiveHQ")),
        (
            3,
            (
                "TurboQuant-paper",
                "FibQuant-paper",
                "BlockQuant-paper",
                "HQ",
                "AdaptiveHQ",
            ),
        ),
        (4, ("TurboQuant-paper", "FibQuant-paper", "HQ", "AdaptiveHQ")),
    ],
)
def test_formal_rate_cli_saved_config_and_results_match(
    monkeypatch, tmp_path, rate: int, requested: tuple[str, ...]
):
    args = _parse(monkeypatch, tmp_path, bits=(rate,), methods=requested)
    monkeypatch.setattr(benchmark, "run_code_commit", lambda: "test-commit")
    monkeypatch.setattr(benchmark, "load_run_provenance", lambda: None)
    monkeypatch.setattr(benchmark, "_detect_skips", lambda *_args: set())
    monkeypatch.setattr(
        benchmark.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )

    class Model:
        config = SimpleNamespace(head_dim=128)
        device = "cpu"

        def eval(self):
            return self

    monkeypatch.setattr(
        benchmark.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: Model(),
    )

    class Codec:
        def __init__(self, method: str):
            self.method = method

        def get_kv_config(self):
            return {"name": self.method}

        def get_resolved_kv_config(self, head_dim: int):
            return {"name": self.method, "head_dim": head_dim}

    monkeypatch.setattr(
        benchmark,
        "_codecs",
        lambda *_args, **_kwargs: {method: Codec(method) for method in requested},
    )
    monkeypatch.setattr(
        benchmark,
        "_eval_method",
        lambda *_args, **_kwargs: {
            "avg_primary_score": 0.5,
            "avg_acc": 0.5,
            "avg_acc_norm": 0.5,
            "n_total": 1,
        },
    )
    monkeypatch.setattr(benchmark, "fibquant_quality_provenance", lambda _codec: {})
    monkeypatch.setattr(benchmark, "blockquant_quality_provenance", lambda _codec: {})
    monkeypatch.setattr(benchmark, "hayate_formal_provenance", lambda _method: {})

    result_path = benchmark._run(args)
    document = json.loads(result_path.read_text(encoding="utf-8"))

    assert document["config"]["methods"] == list(requested)
    assert list(document["results"]) == [str(rate)]
    assert list(document["results"][str(rate)]) == list(requested)
    assert ("BlockQuant-paper" in document["results"][str(rate)]) is (rate == 3)
    assert document["run_fingerprint"] == benchmark._fingerprint(document["config"])
