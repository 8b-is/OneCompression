from pathlib import Path

from benchmark.kvcomp_common.paths import resolve_model_ref, resolve_output_root


def test_model_ref_prefers_external_root(tmp_path: Path) -> None:
    model = tmp_path / "weights"
    model.mkdir()
    assert resolve_model_ref("weights", tmp_path) == str(model.resolve())


def test_hf_id_is_not_rewritten(tmp_path: Path) -> None:
    assert resolve_model_ref("Qwen/Qwen3-0.6B", tmp_path) == "Qwen/Qwen3-0.6B"


def test_output_default_is_repo_relative(tmp_path: Path) -> None:
    out = resolve_output_root(None, root=tmp_path, default_relative="results")
    assert out == tmp_path / "results"
