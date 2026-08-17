"""Dependency-light contracts for the isolated vLLM benchmark driver."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/vllm_benchmark.py"
SPEC = importlib.util.spec_from_file_location("kvcomp_vllm_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"num_hidden_layers": 2}\n', encoding="utf-8")
    values = {
        "model": model,
        "output": tmp_path / "result.json",
        "method": "taperkv",
        "rate": 3,
        "prompt_tokens": 32,
        "max_tokens": 8,
        "repeats": 1,
        "dtype": "float16",
        "gpu_memory_utilization": 0.5,
        "seed": 0,
        "trust_remote_code": False,
        "enforce_eager": False,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_taper_benchmark_keeps_graphs_and_disables_shared_prefix_pages():
    args = argparse.Namespace(
        dtype="float16",
        gpu_memory_utilization=0.5,
        prompt_tokens=32,
        max_tokens=8,
        seed=0,
        trust_remote_code=False,
        enforce_eager=False,
        method="taperkv",
        rate=3,
    )

    kwargs = benchmark._engine_kwargs(args)

    assert kwargs["enforce_eager"] is False
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["disable_log_stats"] is False
    assert "compilation_config" not in kwargs


def test_decode_measurement_uses_intertoken_intervals():
    metrics = SimpleNamespace(first_token_ts=10.0, last_token_ts=10.3)

    tokens, seconds, rate = benchmark._decode_measurement(metrics, 4)

    assert tokens == 3
    assert seconds == pytest.approx(0.3)
    assert rate == pytest.approx(10.0)
    assert benchmark._decode_measurement(None, 4) == (3, None, None)


def _evidence(event: str, **overrides) -> dict:
    row = {
        "schema": {"name": "onecomp.kvcomp_vllm_runtime_evidence", "version": 1},
        "event": event,
    }
    row.update(overrides)
    return row


def test_hayate_runtime_evidence_requires_packed_store_and_decode():
    store = _evidence(
        "packed_hayatequant_store_launched",
        implementation="native_packed_hayatequant",
        cache_dtype="uint8",
        cache_device="cuda:0",
        store_encoder="triton_fused_b2_k_nc_v_ls_iter",
    )
    decode = _evidence(
        "packed_hayatequant_decode_attention_launched",
        implementation="native_packed_hayatequant",
        cache_dtype="uint8",
        cache_device="cuda:0",
    )

    benchmark._validate_runtime_evidence("hayatequant", [store, decode])
    with pytest.raises(RuntimeError, match="missing events"):
        benchmark._validate_runtime_evidence("hayatequant", [store])
    with pytest.raises(RuntimeError, match="packed uint8"):
        benchmark._validate_runtime_evidence(
            "hayatequant", [store | {"cache_dtype": "float16"}, decode]
        )


def test_hayate_runtime_evidence_requires_every_layer_for_both_events():
    rows = []
    for layer_index in range(3):
        common = {
            "implementation": "native_packed_hayatequant",
            "cache_dtype": "uint8",
            "cache_device": "cuda:0",
            "store_encoder": "triton_fused_b2_k_nc_v_ls_iter",
            "layer_index": layer_index,
        }
        rows.extend(
            [
                _evidence("packed_hayatequant_store_launched", **common),
                _evidence("packed_hayatequant_decode_attention_launched", **common),
            ]
        )

    benchmark._validate_runtime_evidence("hayatequant", rows, 3)
    with pytest.raises(RuntimeError, match="did not cover every model layer"):
        benchmark._validate_runtime_evidence("hayatequant", rows[:-1], 3)


def test_taper_runtime_evidence_requires_cuda_regrading_backend():
    valid = _evidence(
        "gpu_paged_cache_regraded",
        backend="vllm_plugins.kvcomp.taper_backend.TaperKVBackend",
        physical_cache_device="cuda:0",
        new_token_cache_update="triton_fused_affine_quantize_and_paged_scatter",
    )

    benchmark._validate_runtime_evidence("taperkv", [valid])
    with pytest.raises(RuntimeError, match="no custom-backend runtime evidence"):
        benchmark._validate_runtime_evidence("taperkv", [])
    with pytest.raises(RuntimeError, match="GPU semantic backend"):
        benchmark._validate_runtime_evidence("taperkv", [valid | {"backend": "wrong"}])
    with pytest.raises(RuntimeError, match="fused quantize-and-store"):
        benchmark._validate_runtime_evidence(
            "taperkv", [valid | {"new_token_cache_update": "separate_kernels"}]
        )

    layered = [valid | {"layer_index": index} for index in range(2)]
    benchmark._validate_runtime_evidence("taperkv", layered, 2)
    with pytest.raises(RuntimeError, match="did not cover every model layer"):
        benchmark._validate_runtime_evidence("taperkv", layered[:1], 2)


def test_baselines_reject_custom_backend_evidence():
    benchmark._validate_runtime_evidence("fullkv", [])
    benchmark._validate_runtime_evidence("turboquant", [])
    with pytest.raises(RuntimeError, match="unexpectedly emitted"):
        benchmark._validate_runtime_evidence("fullkv", [_evidence("unexpected")])


def test_runtime_environment_is_confined_to_result_directory(tmp_path: Path, monkeypatch):
    variables = (
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "HF_HOME",
        "CUDA_CACHE_PATH",
        "TMPDIR",
        "VLLM_PLUGINS",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    output = tmp_path / "speed" / "hayatequant-r3.json"

    root = benchmark._configure_runtime_environment(output)

    assert root == (output.parent / "runtime" / output.stem).resolve()
    assert root.is_dir()
    assert benchmark.os.environ["VLLM_PLUGINS"] == "kvcomp"
    for variable in variables[:-2]:
        assert Path(benchmark.os.environ[variable]).is_relative_to(root)
        assert Path(benchmark.os.environ[variable]).is_dir()
    runtime_tmp = Path(benchmark.os.environ["TMPDIR"])
    assert runtime_tmp.is_dir()
    assert benchmark._ipc_tmpdir_fits(runtime_tmp)


def test_runtime_environment_replaces_overlong_tmpdir(tmp_path: Path, monkeypatch):
    output = tmp_path / ("long-output-component-" * 4) / "speed.json"
    overlong = output.parent / ("long-temp-component-" * 8)
    monkeypatch.setenv("TMPDIR", str(overlong))

    benchmark._configure_runtime_environment(output)

    selected = Path(benchmark.os.environ["TMPDIR"])
    assert selected != overlong.resolve()
    assert benchmark._ipc_tmpdir_fits(selected)


def test_model_overlay_changes_only_temporary_config(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    source = {"model_type": "test"}
    (model / "config.json").write_text(json.dumps(source), encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    parent = tmp_path / "work"
    parent.mkdir()

    with benchmark._model_overlay(model, {"policy": {"name": "taperkv"}}, parent) as overlay:
        config = json.loads((overlay / "config.json").read_text(encoding="utf-8"))
        assert config["kv_compression_config"]["policy"]["name"] == "taperkv"
        assert (overlay / "weights.bin").is_symlink()

    assert json.loads((model / "config.json").read_text(encoding="utf-8")) == source
    assert not overlay.exists()


def test_existing_result_is_reused_only_for_an_identical_invocation(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    existing = {
        "schema_version": 3,
        "method": "taperkv",
        "rate": None,
        "model": str(args.model.resolve()),
        "prompt_tokens": 32,
        "max_tokens": 8,
        "repeats": 1,
        "dtype": "float16",
        "gpu_memory_utilization": 0.5,
        "enforce_eager": False,
        "compilation_mode": "vllm_compile",
        "cuda_graph_mode": "vllm_default",
        "request_metrics_enabled": True,
        "model_config_sha256": benchmark._sha256(args.model / "config.json"),
        "model_checkpoint_fingerprint": None,
        "code_revision": None,
        "compression_layer_contract": {
            "expected_model_layers": 2,
            "explicit_skip_layers": [],
            "require_all_layers": True,
            "vllm_implicit_turboquant_boundary_skip_disabled": None,
        },
    }
    args.output.write_text(json.dumps(existing), encoding="utf-8")

    assert benchmark.run(args) == existing

    mismatched = _args(tmp_path, prompt_tokens=64)
    with pytest.raises(FileExistsError, match="does not match"):
        benchmark.run(mismatched)

    monkeypatch.setenv("CODE_REPO_REVISION", "different-source")
    with pytest.raises(FileExistsError, match="code_revision"):
        benchmark.run(args)
