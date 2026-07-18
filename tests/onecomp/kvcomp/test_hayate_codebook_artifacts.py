"""Fail-closed contracts for shipped HayateQuant codebook artifacts."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import torch

from onecomp.kvcomp.codec.hayate import vq
from onecomp.kvcomp.codec.hayate.recipes import (
    build_hayate_adaptive_rate,
    build_hayate_hq,
    build_hayate_hq_frontier,
)

MANIFEST_SHA256 = "818df8a17a0cac464e4b7d844e63c48ab56dabf70c6693b80af6099f5d83e573"


def _clear_codebook_caches() -> None:
    vq._gaussian_codebook_artifact.cache_clear()
    vq.lloyd_vq_gaussian_block.cache_clear()
    vq.lloyd_vq_gaussian_codebook.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_codebook_caches():
    _clear_codebook_caches()
    yield
    _clear_codebook_caches()


def _write_manifest(directory: Path, artifacts: dict, bundles: dict | None = None) -> Path:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hayatequant.shipped_gaussian_codebooks.v1",
                "artifacts": artifacts,
                "required_bundles": bundles or {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _artifact_contract(path: Path, tensor: torch.Tensor, block_size: int, k: int) -> dict:
    return {
        "block_size": block_size,
        "k": k,
        "shape": [k, block_size],
        "dtype": "torch.float32",
        "file_sha256": vq._sha256_file(str(path)),
        "tensor_sha256": vq._sha256_tensor(tensor),
        "origin": "unit_test",
    }


def _pin_test_manifest(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(vq, "_SHIPPED_CODEBOOK_MANIFEST_SHA256", vq._sha256_file(str(path)))


def test_canonical_recipes_pin_complete_shipped_bundles_and_safe_pairs():
    fixed = build_hayate_hq(2).get_resolved_kv_config(128)["codebook_artifacts"]
    assert fixed["required"] is True
    assert fixed["bundle"] == "fixed_hq_b2_v1"
    assert fixed["manifest_sha256"] == MANIFEST_SHA256
    assert len(fixed["artifacts"]) == 4
    assert {record["source"] for record in fixed["artifacts"]} == {"shipped_manifest"}

    for rate in (2, 3, 4):
        frontier = build_hayate_hq_frontier(rate).get_resolved_kv_config(128)["codebook_artifacts"]
        assert frontier["required"] is True
        assert frontier["bundle"] == "frontier_hq_b2_v1"
        assert frontier["manifest_sha256"] == MANIFEST_SHA256
        assert len(frontier["artifacts"]) == 5
        assert {record["source"] for record in frontier["artifacts"]} == {"shipped_manifest"}

    expected_pairs = {
        2: [[4, 4], [5, 3]],
        3: [[6, 6], [7, 5]],
        4: [[8, 8], [9, 7]],
    }
    for rate, pairs in expected_pairs.items():
        codec = build_hayate_adaptive_rate(rate)
        resolved = codec.get_resolved_kv_config(128)
        bundle = resolved["codebook_artifacts"]
        assert bundle["required"] is True
        assert bundle["bundle"] == "adaptive_hq_b2_safe_v1"
        assert bundle["manifest_sha256"] == MANIFEST_SHA256
        assert len(bundle["artifacts"]) == 7
        assert resolved["adaptive_allocation"]["candidate_pairs"] == pairs
        assert resolved["adaptive_allocation"]["selector_bits_per_encode"] == 1
        assert codec.key_error_weight == 4.0


def test_manifest_pinned_missing_artifact_fails_closed(tmp_path, monkeypatch):
    filename = "gauss_b2_k4.pt"
    contract = {
        "block_size": 2,
        "k": 4,
        "shape": [4, 2],
        "dtype": "torch.float32",
        "file_sha256": "0" * 64,
        "tensor_sha256": "0" * 64,
    }
    manifest = _write_manifest(tmp_path, {filename: contract}, {"test_bundle": [filename]})
    _pin_test_manifest(monkeypatch, manifest)
    monkeypatch.setattr(vq, "_CODEBOOK_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="artifact is missing"):
        vq.require_shipped_gaussian_codebook_bundle("test_bundle")


@pytest.mark.parametrize("bad_hash_field", ["file_sha256", "tensor_sha256"])
def test_manifest_hash_corruption_fails_closed(tmp_path, monkeypatch, bad_hash_field):
    filename = "gauss_b2_k4.pt"
    tensor = torch.tensor([[-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0], [2.0, 1.0]])
    artifact = tmp_path / filename
    torch.save(tensor, artifact)
    contract = _artifact_contract(artifact, tensor, 2, 4)
    contract[bad_hash_field] = "0" * 64
    manifest = _write_manifest(tmp_path, {filename: contract}, {"test_bundle": [filename]})
    _pin_test_manifest(monkeypatch, manifest)
    monkeypatch.setattr(vq, "_CODEBOOK_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        vq.require_shipped_gaussian_codebook_bundle("test_bundle")


@pytest.mark.parametrize(
    ("failure", "expected_error", "message"),
    [
        ("dtype", TypeError, "original dtype"),
        ("shape", ValueError, "has shape"),
        ("nonfinite", ValueError, "non-finite"),
        ("duplicate", ValueError, "duplicate codewords"),
    ],
)
def test_development_override_still_has_strict_tensor_validation(
    tmp_path, monkeypatch, failure, expected_error, message
):
    filename = "gauss_b2_k4.pt"
    tensor = torch.tensor([[-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0], [2.0, 1.0]])
    if failure == "dtype":
        tensor = tensor.half()
    elif failure == "shape":
        tensor = tensor[:3]
    elif failure == "nonfinite":
        tensor[0, 0] = float("nan")
    elif failure == "duplicate":
        tensor[1] = tensor[0]
    torch.save(tensor, tmp_path / filename)
    monkeypatch.setenv("KVCOMP_CODEBOOK_DIR", str(tmp_path))

    with pytest.raises(expected_error, match=message):
        vq._gaussian_codebook_artifact(2, 4)


def test_development_override_provenance_and_formal_rejection(tmp_path, monkeypatch):
    tensor = torch.tensor([[-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0], [2.0, 1.0]])
    torch.save(tensor, tmp_path / "gauss_b2_k4.pt")
    monkeypatch.setenv("KVCOMP_CODEBOOK_DIR", str(tmp_path))

    record = vq.gaussian_codebook_artifact_provenance(2, 4)
    assert record["source"] == "development_override"
    assert record["manifest_pinned"] is False
    assert record["override_dir"] == str(tmp_path.resolve())
    with pytest.raises(RuntimeError, match="forbids.*KVCOMP_CODEBOOK_DIR"):
        build_hayate_hq(2)


def test_shipped_manifest_is_itself_hash_pinned(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {})
    monkeypatch.setattr(vq, "_CODEBOOK_DIR", str(tmp_path))
    monkeypatch.setattr(vq, "_SHIPPED_CODEBOOK_MANIFEST_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="manifest SHA-256 mismatch"):
        vq._gaussian_codebook_artifact(2, 4)


def test_unshipped_development_point_retains_explicit_online_fallback(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {})
    _pin_test_manifest(monkeypatch, manifest)
    monkeypatch.setattr(vq, "_CODEBOOK_DIR", str(tmp_path))

    assert vq._gaussian_codebook_artifact(2, 4) is None
    assert vq.gaussian_codebook_artifact_provenance(2, 4) == {
        "source": "online_fallback",
        "block_size": 2,
        "k": 4,
        "development_override_dir": None,
    }


def test_wheel_package_data_contract_includes_codebooks_and_manifest():
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    entries = set(project["tool"]["setuptools"]["package-data"]["onecomp"])
    assert "kvcomp/codec/hayate/codebooks/*.pt" in entries
    assert "kvcomp/codec/hayate/codebooks/*.json" in entries
