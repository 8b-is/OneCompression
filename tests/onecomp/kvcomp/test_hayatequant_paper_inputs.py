"""Tests for the offline HayateQuant paper-input validation manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/validate_hayatequant_paper_inputs.py"
SPEC = importlib.util.spec_from_file_location("hayatequant_paper_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _build_inputs(root: Path) -> dict:
    run_root = root / "run"
    work_root = root / "work"
    runtime_home = work_root / "runtime/home"
    runtime_home.mkdir(parents=True)
    models_root = root / "models"
    model_paths = {}
    for key in validator.RULER_MODEL_KEYS:
        path = models_root / key
        context_limit = validator.LONGCODEQA_CONTEXT_SPECS[key]["context_limit"]
        _write_json(
            path / "config.json",
            {
                "model_type": "synthetic",
                "key": key,
                "max_position_embeddings": context_limit,
            },
        )
        _write_json(
            path / "tokenizer_config.json",
            {"tokenizer_class": "Synthetic", "chat_template": f"template-{key}"},
        )
        _write_json(
            path / "generation_config.json",
            {"eos_token_id": [101, 102] if key == "llama31-8b" else [201, 202]},
        )
        _write_json(path / "tokenizer.json", {"version": "1.0", "model": {}})
        if key == "llama31-8b":
            (path / "model-00001-of-00002.safetensors").write_bytes(b"llama-shard-one")
            (path / "model-00002-of-00002.safetensors").write_bytes(b"llama-shard-two")
            _write_json(
                path / "model.safetensors.index.json",
                {
                    "metadata": {"total_size": 30},
                    "weight_map": {
                        "layer.0.weight": "model-00001-of-00002.safetensors",
                        "layer.1.weight": "model-00002-of-00002.safetensors",
                    },
                },
            )
        else:
            (path / "model.safetensors").write_bytes(b"qwen-single-file-weights")
        model_paths[key] = path

    ruler_root = run_root / "data/ruler_v1"
    for model_key in validator.RULER_MODEL_KEYS:
        model_root = ruler_root / model_key
        file_groups = {}
        for sequence_length in validator.RULER_LENGTHS:
            group = {}
            for task in validator.RULER_TASKS:
                task_path = model_root / str(sequence_length) / task / "test.jsonl"
                task_path.parent.mkdir(parents=True, exist_ok=True)
                with task_path.open("w", encoding="utf-8") as stream:
                    for index in range(validator.RULER_ROWS_PER_TASK):
                        stream.write(
                            json.dumps(
                                {
                                    "index": index,
                                    "input": f"question-{index}",
                                    "answer_prefix": " Answer:",
                                    "outputs": ["alpha"],
                                    "length": 20,
                                    "length_w_model_temp": 24,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                group[task] = {
                    **_record(task_path),
                    "rows": validator.RULER_ROWS_PER_TASK,
                    "length_w_model_temp_min": 24,
                    "length_w_model_temp_max": 24,
                }
            file_groups[str(sequence_length)] = group
        model_path = model_paths[model_key]
        metadata = {
            name: {
                "bytes": (model_path / name).stat().st_size,
                "sha256": _digest(model_path / name),
            }
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json")
        }
        _write_json(
            model_root / "manifest.json",
            {
                "schema_version": 1,
                "benchmark": validator.RULER_PROTOCOL_LABEL,
                "model": {
                    "key": model_key,
                    "path": str(model_path.resolve()),
                    "revision": "a" * 40,
                    "snapshot_revisions": ["a" * 40],
                    "metadata_files": metadata,
                },
                "protocol": {
                    "lengths": list(validator.RULER_LENGTHS),
                    "num_samples_per_task": validator.RULER_ROWS_PER_TASK,
                    "prepare_for_nemo_skills": True,
                    "sampling_protocol": validator.RULER_SAMPLING_PROTOCOL,
                    "random_seed": 42,
                    "tasks": list(validator.RULER_TASKS),
                    "chat_template_reserve": 50,
                    "tokenizer_type": "hf",
                },
                "sources": {
                    "ruler_commit": validator.RULER_SOURCE_COMMIT,
                    "nemo_skills_commit": validator.NEMO_SKILLS_SOURCE_COMMIT,
                },
                "files": file_groups,
            },
        )

    longcodeqa_root = run_root / "data/LongCodeQA"
    longcodeqa_hashes = {}
    for bracket, count in validator.LONGCODEQA_COUNTS.items():
        path = longcodeqa_root / "LQA" / f"{bracket}.json"
        _write_json(
            path,
            [
                {
                    "prompt": f"repository prompt {index}",
                    "correct_letter": "ABCD"[index % 4],
                    "repo": "owner/repository",
                    "question": f"question {index}",
                }
                for index in range(count)
            ],
        )
        longcodeqa_hashes[bracket] = _digest(path)

    context_manifests = {}
    for model_key, spec in validator.LONGCODEQA_CONTEXT_SPECS.items():
        context_limit = spec["context_limit"]
        bracket_records = {}
        for bracket in spec["brackets"]:
            count = validator.LONGCODEQA_COUNTS[bracket]
            input_lengths = [1000 + index for index in range(count)]
            per_row = [
                {
                    "index": index,
                    "input_tokens": tokens,
                    "margin": (
                        context_limit - tokens - validator.LONGCODEQA_CONTEXT_MAX_NEW_TOKENS
                    ),
                }
                for index, tokens in enumerate(input_lengths)
            ]
            margins = [row["margin"] for row in per_row]
            bracket_records[bracket] = {
                "rows": count,
                "dataset_path": str((longcodeqa_root / "LQA" / f"{bracket}.json").resolve()),
                "dataset_sha256": longcodeqa_hashes[bracket],
                "input_tokens": {
                    "min": min(input_lengths),
                    "median": statistics.median(input_lengths),
                    "max": max(input_lengths),
                },
                "minimum_margin_tokens": min(margins),
                "overflow_rows": [],
                "per_row": per_row,
            }
        model_path = model_paths[model_key]
        chat_template = f"template-{model_key}"
        context_path = run_root / "data" / f"longcodeqa_context_{model_key}.json"
        _write_json(
            context_path,
            {
                "schema_version": 2,
                "created_at_utc": "2026-07-15T00:00:00+00:00",
                "benchmark": "LongCodeBench/LongCodeQA context-fit validation",
                "model": {
                    "path": str(model_path.resolve()),
                    "revision": "a" * 40,
                    "snapshot_revisions": ["a" * 40],
                    "context_limit": context_limit,
                    "generation_eos_token_ids": (
                        [101, 102] if model_key == "llama31-8b" else [201, 202]
                    ),
                    "generation_eos_token_ids_source": ("model.generation_config.eos_token_id"),
                    "chat_template_sha256": hashlib.sha256(
                        chat_template.encode("utf-8")
                    ).hexdigest(),
                    **{
                        filename: {
                            "bytes": (model_path / filename).stat().st_size,
                            "sha256": _digest(model_path / filename),
                        }
                        for filename in (
                            "config.json",
                            "generation_config.json",
                            "tokenizer.json",
                            "tokenizer_config.json",
                        )
                    },
                },
                "protocol": {
                    "scope": validator.LONGCODEQA_PROTOCOL_SCOPE,
                    "dataset_revision": validator.LONGCODEQA_DATASET_REVISION,
                    "prompt": "released prompt verbatim as raw causal text",
                    "prompt_release_clarification": (validator.LONGCODEQA_DISCUSSION_URL),
                    "chat_template": "none",
                    "truncation": "none",
                    "answer_parser_reference_commit": (validator.LONGCODEQA_PROTOCOL_COMMIT),
                    "answer_parser": validator.LONGCODEQA_ANSWER_PARSER,
                    "reference_evaluator_executable_verbatim": False,
                    "reference_evaluator_known_issue": (
                        "pinned evaluator stores response but reads prediction text"
                    ),
                    "generation": {
                        "protocol": ("hayatequant deterministic greedy paired comparison"),
                        "do_sample": False,
                        "max_new_tokens": (validator.LONGCODEQA_CONTEXT_MAX_NEW_TOKENS),
                        "seed": None,
                        "eos_token_ids_source": ("model.generation_config.eos_token_id"),
                        "stop_conditions": [
                            "any_eos_token_id",
                            "max_new_tokens",
                        ],
                        "official_reference_max_output_length": None,
                        "official_reference_uniform_decoding_config": False,
                        "official_reference_decoding_exact": False,
                    },
                },
                "brackets": bracket_records,
                "all_rows_fit": True,
            },
        )
        context_manifests[model_key] = context_path

    hf_home = run_root / "data/hf_home"
    hf_records = {}
    for key, pin in validator.HF_DATASETS.items():
        cache_file = hf_home / "datasets" / f"{key}.arrow"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(f"cached-{key}".encode())
        rows = pin["expected_rows"] if pin["expected_rows"] is not None else 30
        hf_records[key] = {
            **pin,
            "rows": rows,
            "dataset_fingerprint": f"fingerprint-{key}",
            "features": "synthetic features",
            "cache_files": [_record(cache_file)],
            "download_checksums": {},
            "download_size": cache_file.stat().st_size,
            "dataset_size": cache_file.stat().st_size,
        }
    longbench_zip = hf_home / "hub" / "longbench-data.zip"
    longbench_zip.parent.mkdir(parents=True, exist_ok=True)
    longbench_zip.write_bytes(b"synthetic longbench archive")
    hf_manifest = run_root / "data/pinned_hf_datasets_manifest.json"
    _write_json(
        hf_manifest,
        {
            "schema_version": 1,
            "created_at_utc": "2026-07-15T00:00:00+00:00",
            "hf_home": str(hf_home.resolve()),
            "packages": {
                "datasets": "4.0.0",
                "huggingface-hub": "0.34.0",
                "pyarrow": "20.0.0",
            },
            "datasets": hf_records,
            "longbench_e": {
                "repo": "THUDM/LongBench",
                "revision": validator.LONGBENCH_REVISION,
                "filename": "data.zip",
                "artifact": _record(longbench_zip),
            },
        },
    )
    return {
        "run_root": run_root,
        "work_root": work_root,
        "runtime_home": runtime_home,
        "models_root": models_root,
        "model_paths": model_paths,
        "ruler_root": ruler_root,
        "longcodeqa_root": longcodeqa_root,
        "longcodeqa_hashes": longcodeqa_hashes,
        "context_manifests": context_manifests,
        "hf_home": hf_home,
        "hf_manifest": hf_manifest,
        "longbench_zip": longbench_zip,
    }


def _validate(paths: dict) -> dict:
    return validator.validate_inputs(
        run_root=paths["run_root"],
        work_root=paths["work_root"],
        runtime_home=paths["runtime_home"],
        models_root=paths["models_root"],
        model_paths=paths["model_paths"],
        ruler_root=paths["ruler_root"],
        longcodeqa_root=paths["longcodeqa_root"],
        hf_home=paths["hf_home"],
        hf_manifest=paths["hf_manifest"],
        require_hf_manifest=True,
        expected_longcodeqa_sha256=paths["longcodeqa_hashes"],
    )


def test_full_synthetic_input_set_validates_and_writes_atomically(tmp_path):
    paths = _build_inputs(tmp_path)
    document = _validate(paths)
    output = paths["run_root"] / "manifests/paper_inputs.json"
    validator.atomic_write_json(output, document)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["schema_version"] == 2
    assert saved["runtime_paths"] == {
        "work_root": str(paths["work_root"].resolve()),
        "home": str(paths["runtime_home"].resolve()),
    }
    assert saved["hf_datasets"]["status"] == "validated"
    assert saved["longcodeqa"]["total_rows"] == 281
    assert saved["longcodeqa"]["release_revision"] == validator.LONGCODEQA_DATASET_REVISION
    assert saved["longcodeqa"]["parser_reference_commit"] == validator.LONGCODEQA_PROTOCOL_COMMIT
    assert saved["longcodeqa"]["official_reference_decoding_exact"] is False
    assert saved["ruler_v1"]["llama31-8b"]["total_rows"] == 3900
    assert saved["ruler_v1"]["llama31-8b"]["benchmark"] == validator.RULER_PROTOCOL_LABEL
    assert (
        saved["ruler_v1"]["llama31-8b"]["sampling_protocol"] == validator.RULER_SAMPLING_PROTOCOL
    )
    assert saved["ruler_v1"]["llama31-8b"]["canonical_nvidia_full_eval"] is False
    assert (
        saved["ruler_v1"]["llama31-8b"]["canonical_nvidia_samples_per_task"]
        == validator.RULER_CANONICAL_NVIDIA_SAMPLES_PER_TASK
    )
    assert saved["ruler_v1"]["llama31-8b"]["source_commits"] == {
        "ruler": validator.RULER_SOURCE_COMMIT,
        "nemo_skills": validator.NEMO_SKILLS_SOURCE_COMMIT,
    }
    context_fit = saved["longcodeqa"]["context_fit_manifests"]
    assert set(context_fit) == {"llama31-8b", "qwen3-8b"}
    assert set(context_fit["llama31-8b"]["brackets"]) == {"32K", "64K", "128K"}
    assert set(context_fit["qwen3-8b"]["brackets"]) == {"32K"}
    assert context_fit["qwen3-8b"]["manifest"]["sha256"] == _digest(
        paths["context_manifests"]["qwen3-8b"]
    )
    llama_weights = saved["models"]["llama31-8b"]["weights"]
    assert llama_weights["selected_format"] == "safetensors"
    assert llama_weights["selected_kind"] == "sharded_index"
    assert llama_weights["layouts"]["safetensors"]["shard_count"] == 2
    assert saved["models"]["qwen3-8b"]["weights"]["selected_kind"] == "single_file"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_runtime_paths_reject_wrong_home_and_symlink_components(tmp_path):
    paths = _build_inputs(tmp_path)
    wrong_home = paths["work_root"] / "other-home"
    wrong_home.mkdir()
    paths["runtime_home"] = wrong_home
    with pytest.raises(validator.ValidationError, match="must be exactly"):
        _validate(paths)

    paths = _build_inputs(tmp_path / "symlinked")
    link = tmp_path / "work-link"
    link.symlink_to(paths["work_root"], target_is_directory=True)
    paths["work_root"] = link
    paths["runtime_home"] = link / "runtime/home"
    with pytest.raises(validator.ValidationError, match="symlink path components"):
        _validate(paths)


def test_atomic_manifest_preserves_bytes_when_only_validation_time_changes(tmp_path):
    output = tmp_path / "paper_inputs.json"
    first = {
        "schema_version": 2,
        "kind": "hayatequant_paper_input_validation",
        "status": "ok",
        "validated_at_utc": "2026-01-01T00:00:00+00:00",
        "models": {"model": {"sha256": "a" * 64}},
    }
    validator.atomic_write_json(output, first)
    original = output.read_bytes()
    second = {**first, "validated_at_utc": "2026-01-02T00:00:00+00:00"}
    validator.atomic_write_json(output, second)
    assert output.read_bytes() == original


def test_rejects_unpinned_ruler_source_commit(tmp_path):
    paths = _build_inputs(tmp_path)
    manifest_path = paths["ruler_root"] / "llama31-8b/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"]["ruler_commit"] = "0" * 40
    _write_json(manifest_path, manifest)

    with pytest.raises(validator.ValidationError, match="RULER commit"):
        _validate(paths)


def test_rejects_ruler_file_that_no_longer_matches_manifest(tmp_path):
    paths = _build_inputs(tmp_path)
    task_path = paths["ruler_root"] / "llama31-8b/4096/niah_single_1/test.jsonl"
    lines = task_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["input"] = first["input"].replace("question", "questioN")
    lines[0] = json.dumps(first, sort_keys=True)
    task_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(validator.ValidationError, match="manifest sha256 mismatch"):
        _validate(paths)


def test_accepts_upstream_ruler_source_identifiers_when_manifest_is_updated(tmp_path):
    paths = _build_inputs(tmp_path)
    task_path = paths["ruler_root"] / "llama31-8b/4096/niah_single_1/test.jsonl"
    lines = task_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["index"] = 14051
    lines[0] = json.dumps(first, sort_keys=True)
    second = json.loads(lines[1])
    second["index"] = 14051
    lines[1] = json.dumps(second, sort_keys=True)
    task_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = paths["ruler_root"] / "llama31-8b/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["4096"]["niah_single_1"].update(_record(task_path))
    _write_json(manifest_path, manifest)

    result = _validate(paths)
    assert result["status"] == "ok"


def test_rejects_unpinned_ruler_generation_protocol(tmp_path):
    paths = _build_inputs(tmp_path)
    manifest_path = paths["ruler_root"] / "llama31-8b/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["random_seed"] = 41
    _write_json(manifest_path, manifest)

    with pytest.raises(validator.ValidationError, match="random_seed must be 42"):
        _validate(paths)


def test_rejects_ruler_sampling_protocol_relabel(tmp_path):
    paths = _build_inputs(tmp_path)
    manifest_path = paths["ruler_root"] / "llama31-8b/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["sampling_protocol"] = "canonical NVIDIA/RULER 500/task"
    _write_json(manifest_path, manifest)

    with pytest.raises(validator.ValidationError, match="sampling protocol label drifted"):
        _validate(paths)


def test_rejects_invalid_longcodeqa_schema(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["longcodeqa_root"] / "LQA/32K.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["correct_letter"] = "E"
    _write_json(path, rows)

    with pytest.raises(validator.ValidationError, match="invalid correct_letter"):
        _validate(paths)


def test_duplicate_json_keys_are_rejected_in_dataset_manifests(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["longcodeqa_root"] / "LQA/32K.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    duplicate = (
        '[{"prompt":"first","prompt":"second","correct_letter":"A",'
        '"repo":"owner/repository","question":"question"},'
        + ",".join(json.dumps(row, sort_keys=True) for row in rows[1:])
        + "]\n"
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="duplicate JSON key"):
        _validate(paths)


def test_nonfinite_json_numbers_are_rejected_in_dataset_manifests(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["longcodeqa_root"] / "LQA/32K.json"
    document = path.read_text(encoding="utf-8")
    path.write_text(document.replace('"prompt":', '"unused":NaN,"prompt":', 1))
    with pytest.raises(validator.ValidationError, match="non-finite JSON number"):
        _validate(paths)


def test_required_hf_manifest_and_cached_hashes_are_enforced(tmp_path):
    paths = _build_inputs(tmp_path)
    paths["longbench_zip"].write_bytes(b"Synthetic longbench archive")
    with pytest.raises(validator.ValidationError, match="sha256 mismatch"):
        _validate(paths)

    paths = _build_inputs(tmp_path / "second")
    paths["hf_manifest"].unlink()
    with pytest.raises(validator.ValidationError, match="Required pinned HF dataset manifest"):
        _validate(paths)


def test_hf_manifest_rejects_duplicate_cache_paths(tmp_path):
    paths = _build_inputs(tmp_path)
    manifest = json.loads(paths["hf_manifest"].read_text(encoding="utf-8"))
    records = manifest["datasets"]["wikitext2_test"]["cache_files"]
    records.append(dict(records[0]))
    _write_json(paths["hf_manifest"], manifest)
    with pytest.raises(validator.ValidationError, match="repeats a cache file"):
        _validate(paths)


def test_context_fit_rejects_wrong_feasible_bracket_set(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["context_manifests"]["qwen3-8b"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["brackets"]["64K"] = document["brackets"]["32K"]
    _write_json(path, document)

    with pytest.raises(validator.ValidationError, match="brackets must be exactly"):
        _validate(paths)


def test_context_fit_rejects_overflow_even_if_manifest_claims_everything_fits(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["context_manifests"]["qwen3-8b"]
    document = json.loads(path.read_text(encoding="utf-8"))
    row = document["brackets"]["32K"]["per_row"][0]
    row["input_tokens"] = validator.LONGCODEQA_CONTEXT_SPECS["qwen3-8b"]["context_limit"]
    row["margin"] = -validator.LONGCODEQA_CONTEXT_MAX_NEW_TOKENS
    _write_json(path, document)

    with pytest.raises(validator.ValidationError, match="overflows the 40960-token"):
        _validate(paths)


def test_context_fit_rejects_changed_tokenizer_metadata(tmp_path):
    paths = _build_inputs(tmp_path)
    path = paths["context_manifests"]["llama31-8b"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["model"]["tokenizer_config.json"]["sha256"] = "0" * 64
    _write_json(path, document)

    with pytest.raises(validator.ValidationError, match="tokenizer/model metadata"):
        _validate(paths)


def test_rejects_changed_model_weight_content(tmp_path):
    paths = _build_inputs(tmp_path)
    first = validator.validate_models({"model": paths["model_paths"]["qwen3-8b"]})
    expected = first["model"]["weights"]
    weight = paths["model_paths"]["qwen3-8b"] / "model.safetensors"
    weight.write_bytes(b"Qwen-single-file-weights")
    with pytest.raises(validator.ModelWeightError, match="differ from the pinned"):
        validator.verify_model_weights(paths["model_paths"]["qwen3-8b"], expected)


@pytest.mark.parametrize("failure", ["missing", "unreferenced"])
def test_rejects_incomplete_or_unreferenced_safetensor_shards(tmp_path, failure):
    paths = _build_inputs(tmp_path)
    model = paths["model_paths"]["llama31-8b"]
    if failure == "missing":
        (model / "model-00002-of-00002.safetensors").unlink()
        match = "missing shard"
    else:
        (model / "model-00003-of-00003.safetensors").write_bytes(b"orphan")
        match = "unreferenced weight shard"
    with pytest.raises(validator.ValidationError, match=match):
        validator.validate_models({"model": model})


def test_rejects_duplicate_weight_index_json_keys(tmp_path):
    paths = _build_inputs(tmp_path)
    model = paths["model_paths"]["llama31-8b"]
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"same":"model-00001-of-00002.safetensors",'
        '"same":"model-00002-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(validator.ValidationError, match="duplicate JSON key"):
        validator.validate_models({"model": model})


def test_bin_only_layout_and_file_symlink_are_fingerprinted(tmp_path):
    model = tmp_path / "bin-model"
    model.mkdir()
    target = model / "actual.bin"
    target.write_bytes(b"bin-weights")
    (model / "pytorch_model.bin").symlink_to(target.name)

    weights = validator.fingerprint_model_weights(model)
    assert weights["selected_format"] == "pytorch_bin"
    record = weights["layouts"]["pytorch_bin"]["shards"][0]
    assert record["symlink"] is True
    assert record["symlink_target"] == "actual.bin"
    assert record["sha256"] == _digest(target)


def test_same_format_index_and_single_file_are_rejected_as_ambiguous(tmp_path):
    model = tmp_path / "ambiguous"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"single")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    _write_json(
        model / "model.safetensors.index.json",
        {"weight_map": {"weight": "model-00001-of-00001.safetensors"}},
    )
    with pytest.raises(validator.ModelWeightError, match="ambiguous safetensors"):
        validator.fingerprint_model_weights(model)
