from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "benchmark/kvcomp-codec"
sys.path.insert(0, str(SCRIPT_DIR))
import merge_hayatequant_ppl_shards as merger  # noqa: E402


COMMIT = "1" * 40
MODEL = "Llama-2-7b-hf"
MODEL_PATH = f"/models/{MODEL}"
INPUT = {"path": "/run/manifests/inputs.json", "bytes": 10, "sha256": "2" * 64}


def _document(rate: int, preflight_digest: str) -> dict:
    semantic = {
        "model_path": MODEL_PATH,
        "dtype": "bfloat16",
        "run_ppl": True,
        "ppl_windows": None,
        "ppl_max_length": 2048,
        "ppl_stride": 1024,
        "sample_seed": 2027,
        "skip_outliers": False,
        "sink_tokens": 0,
        "recent_tokens": 0,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {"semantic_config": semantic, "code_commit": COMMIT},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    methods = list(merger.EXPECTED_METHODS[str(rate)])
    record = {
        "schema_version": 1,
        "code": {"commit": COMMIT, "tracked_dirty": False},
        "model": {"path": MODEL_PATH, "input_manifest": dict(INPUT)},
    }
    return {
        "schema_version": 2,
        "run_fingerprint": fingerprint,
        "model": MODEL,
        "head_dim": 128,
        "config": {
            **semantic,
            "output_dir": f"/run/results/formal/ppl_shards/llama2/b{rate}",
            "bits": [rate],
            "methods": list(merger.FORMAL_METHODS),
        },
        "semantic_config": semantic,
        "code_commit": COMMIT,
        "skip_layers": [],
        "run_provenance": [
            {
                "path": f"/run/provenance/ppl-b{rate}.json",
                "sha256": preflight_digest,
                "record": record,
            }
        ],
        "fullkv_ppl": {"ppl": 5.0, "ppl_n_tokens": 42},
        "results": {str(rate): {method: {"ppl": rate + index / 10} for index, method in enumerate(methods)}},
    }


def _shards(tmp_path: Path) -> list[Path]:
    paths = []
    for rate in merger.FORMAL_BITS:
        path = tmp_path / f"b{rate}" / f"results_{MODEL}.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_document(rate, f"{rate}" * 64)) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def test_merge_constructs_exact_canonical_grid_and_shard_ledger(tmp_path):
    paths = _shards(tmp_path)
    output = tmp_path / "ppl" / f"results_{MODEL}.json"
    document = merger.merge(paths, output)
    assert document["config"]["bits"] == [2, 3, 4]
    assert document["config"]["methods"] == list(merger.FORMAL_METHODS)
    assert list(document["results"]) == ["2", "3", "4"]
    assert len(document["run_provenance"]) == 3
    assert [row["rate"] for row in document["ppl_shard_merge"]["shards"]] == [2, 3, 4]
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in document["ppl_shard_merge"]["shards"])


@pytest.mark.parametrize("field", ["fullkv_ppl", "semantic_config", "head_dim"])
def test_merge_rejects_changed_scientific_invariant(tmp_path, field):
    paths = _shards(tmp_path)
    document = json.loads(paths[1].read_text())
    if field == "head_dim":
        document[field] = 64
    else:
        document[field]["tampered"] = True
    paths[1].write_text(json.dumps(document) + "\n")
    with pytest.raises(merger.MergeError, match="invariant"):
        merger.merge(paths, tmp_path / "result.json")


def test_merge_rejects_missing_method_and_duplicate_rate(tmp_path):
    paths = _shards(tmp_path)
    document = json.loads(paths[1].read_text())
    document["results"]["3"].pop("RotationHQ")
    paths[1].write_text(json.dumps(document) + "\n")
    with pytest.raises(merger.MergeError, match="method coverage"):
        merger.merge(paths, tmp_path / "result.json")

    paths = _shards(tmp_path / "duplicate")
    duplicate = json.loads(paths[2].read_text())
    duplicate["config"]["bits"] = [3]
    duplicate["results"] = {"3": _document(3, "3" * 64)["results"]["3"]}
    paths[2].write_text(json.dumps(duplicate) + "\n")
    with pytest.raises(merger.MergeError, match="Duplicate PPL rate shard"):
        merger.merge(paths, tmp_path / "duplicate-result.json")


def test_merge_rejects_reused_preflight_and_existing_output(tmp_path):
    paths = _shards(tmp_path)
    document = json.loads(paths[1].read_text())
    document["run_provenance"][0]["sha256"] = "2" * 64
    paths[1].write_text(json.dumps(document) + "\n")
    with pytest.raises(merger.MergeError, match="reuse one worker preflight"):
        merger.merge(paths, tmp_path / "result.json")

    paths = _shards(tmp_path / "existing")
    output = tmp_path / "already.json"
    output.write_text("{}\n")
    with pytest.raises(merger.MergeError, match="overwrite"):
        merger.merge(paths, output)
