from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/reproduce_hayatequant_paper.py"
SPEC = importlib.util.spec_from_file_location("hayatequant_reproduction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reproduction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reproduction
SPEC.loader.exec_module(reproduction)

CODEC_SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/codec_benchmark.py"
CODEC_SPEC = importlib.util.spec_from_file_location("hayatequant_codec_benchmark", CODEC_SCRIPT)
assert CODEC_SPEC is not None and CODEC_SPEC.loader is not None
codec_benchmark = importlib.util.module_from_spec(CODEC_SPEC)
sys.modules[CODEC_SPEC.name] = codec_benchmark
CODEC_SPEC.loader.exec_module(codec_benchmark)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model_manifest(tmp_path: Path) -> tuple[Path, Path]:
    models_root = tmp_path / "models"
    rows = []
    for index in range(7):
        ref = f"model-{index}"
        _write_json(models_root / ref / "config.json", {})
        (models_root / ref / "model.safetensors").write_bytes(f"weights-{index}".encode())
        rows.append({"key": f"m{index}", "display": f"Model {index}", "ref": ref})
    manifest = tmp_path / "models.json"
    _write_json(
        manifest,
        {"schema_version": 1, "speed_model": "m1", "models": rows},
    )
    return models_root, manifest


def test_quality_comparison_uses_the_paper_turboquant_recipe():
    codecs = codec_benchmark._codecs(3)

    turboquant = codecs["turboquant"]
    assert turboquant.dist == "beta"
    assert turboquant.rotation_mode == "hadamard"
    assert codecs["hayatequant"]._role_code_bits("k") == 8
    assert codecs["hayatequant"]._role_code_bits("v") == 4


def test_virtualenv_launcher_symlink_is_not_resolved(tmp_path: Path):
    launcher = tmp_path / "python"
    launcher.symlink_to(Path(sys.executable))

    resolved = reproduction._resolve_executable(str(launcher))

    assert resolved == str(launcher.absolute())
    assert Path(resolved).is_symlink()


def test_canonical_plan_has_exact_jobs_including_packed_turboquant_rate2(tmp_path: Path):
    models_root, model_manifest = _model_manifest(tmp_path)
    output_root = tmp_path / "outside-repository-results"
    args = reproduction.build_parser().parse_args(
        [
            "plan",
            "--models-root",
            str(models_root),
            "--model-manifest",
            str(model_manifest),
            "--output-root",
            str(output_root),
            "--python",
            sys.executable,
            "--vllm-python",
            sys.executable,
        ]
    )

    manifest = reproduction.build_manifest(args)

    assert len(manifest["models"]) == 7
    assert len(manifest["code"]["snapshot_sha256"]) == 64
    assert manifest["code"]["revision"]
    assert (
        manifest["protocol"]["zeroshot_token_boundary_protocol"]
        == reproduction.ZEROSHOT_TOKEN_BOUNDARY_PROTOCOL
    )
    assert manifest["protocol"]["ppl_protocol"] == reproduction.PAPER_PPL_PROTOCOL
    assert manifest["protocol"]["ppl_windows"] is None
    assert manifest["protocol"]["ppl_complete_split"] is True
    assert all(len(model["model_config_sha256"]) == 64 for model in manifest["models"])
    assert all(
        len(model["checkpoint"]["fingerprint_sha256"]) == 64 for model in manifest["models"]
    )
    assert all(
        f"PAPER_PROVENANCE_PATH={manifest['source_provenance']}" in job["command"]
        for job in manifest["jobs"]
    )
    assert all(
        f"PYTHONPATH={manifest['source_root']}" in job["command"] for job in manifest["jobs"]
    )
    expected_cache_environment = set(reproduction._runtime_cache_environment(output_root))
    assert all(expected_cache_environment.issubset(job["command"]) for job in manifest["jobs"])
    tmpdir = Path(
        next(
            item.removeprefix("TMPDIR=")
            for item in expected_cache_environment
            if item.startswith("TMPDIR=")
        )
    )
    socket_path = tmpdir / ("x" * reproduction.VLLM_IPC_NAME_LENGTH)
    assert len(reproduction.os.fsencode(socket_path)) <= reproduction.VLLM_IPC_PATH_MAX_LEN
    for job in manifest["jobs"]:
        runtime_path = next(
            item.removeprefix("PATH=") for item in job["command"] if item.startswith("PATH=")
        )
        interpreter = next(
            item for item in job["command"][1:] if "=" not in item and Path(item).is_absolute()
        )
        assert runtime_path.split(reproduction.os.pathsep)[0] == str(Path(interpreter).parent)
    assert all(
        any(
            str(item).startswith(manifest["source_root"] + "/benchmark/")
            for item in job["command"]
        )
        for job in manifest["jobs"]
    )
    assert {
        stage: sum(job["stage"] == stage for job in manifest["jobs"])
        for stage in reproduction.STAGES
    } == {
        "ppl": 7,
        "zeroshot": 7,
        "speed": 7,
    }
    ppl_jobs = [job for job in manifest["jobs"] if job["stage"] == "ppl"]
    assert all(
        not any(str(item).startswith("ppl_windows=") for item in job["command"])
        for job in ppl_jobs
    )
    speed_jobs = [job for job in manifest["jobs"] if job["stage"] == "speed"]
    speed_pairs = {
        (job["command"][job["command"].index("--method") + 1], _rate(job["command"]))
        for job in speed_jobs
    }
    assert speed_pairs == {
        ("fullkv", None),
        ("turboquant", 2),
        ("turboquant", 3),
        ("turboquant", 4),
        ("hayatequant", 2),
        ("hayatequant", 3),
        ("hayatequant", 4),
    }
    zeroshot_jobs = [job for job in manifest["jobs"] if job["stage"] == "zeroshot"]
    removed_exploration_flags = {
        "--sink-tokens",
        "--recent-tokens",
        "--skip-outliers",
        "--no-skip-outliers",
        "--outlier-ratio",
    }
    assert all(removed_exploration_flags.isdisjoint(job["command"]) for job in zeroshot_jobs)
    assert all(Path(job["result"]).is_relative_to(output_root) for job in manifest["jobs"])
    generic_files = [
        SCRIPT,
        REPO_ROOT / "benchmark/kvcomp-codec/codec_benchmark.py",
        REPO_ROOT / "benchmark/kvcomp-codec/zeroshot_benchmark.py",
        REPO_ROOT / "benchmark/kvcomp-codec/conf/paper_hayatequant.yaml",
    ]
    assert all("/lustre/" not in path.read_text(encoding="utf-8") for path in generic_files)


def test_source_archive_is_verified_materialized_and_read_only(tmp_path: Path):
    source_file = tmp_path / "source" / "module.py"
    source_file.parent.mkdir()
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    files = [Path("module.py")]
    digest = reproduction._source_snapshot_digest(files, source_file.parent)
    archive = tmp_path / "source.tar.gz"
    with reproduction.tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source_file, arcname="module.py")

    frozen = tmp_path / "frozen"
    reproduction._extract_source_archive(archive, frozen, digest, files)

    assert (frozen / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (frozen / ".onecomp_source_snapshot_sha256").read_text().strip() == digest
    assert frozen.stat().st_mode & 0o222 == 0
    assert (frozen / "module.py").stat().st_mode & 0o222 == 0

    (frozen / "module.py").chmod(0o644)
    (frozen / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="content changed"):
        reproduction._extract_source_archive(archive, frozen, digest, files)


def test_written_plan_is_self_contained_and_valid_for_submit_plan(tmp_path: Path):
    models_root, model_manifest = _model_manifest(tmp_path)
    output_root = tmp_path / "frozen-plan"
    args = reproduction.build_parser().parse_args(
        [
            "plan",
            "--models-root",
            str(models_root),
            "--model-manifest",
            str(model_manifest),
            "--output-root",
            str(output_root),
            "--python",
            sys.executable,
            "--vllm-python",
            sys.executable,
            "--stages",
            "speed",
        ]
    )

    assert reproduction.command_plan(args) == 0
    manifest_path = output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reproduction._validate_existing_plan(manifest_path, manifest)

    assert Path(manifest["model_manifest_snapshot"]).is_file()
    assert all(
        f"PAPER_PROVENANCE_SHA256={manifest['source_provenance_sha256']}" in job["command"]
        for job in manifest["jobs"]
    )

    provenance = Path(manifest["source_provenance"])
    provenance.chmod(0o644)
    provenance.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source provenance changed"):
        reproduction._validate_existing_plan(manifest_path, manifest)


def _rate(command: list[str]) -> int | None:
    return int(command[command.index("--rate") + 1]) if "--rate" in command else None


def _complete_quality_documents() -> tuple[dict, dict]:
    ppl = {
        "fullkv_ppl": {
            "ppl": 10.0,
            "ppl_complete_text_eval": True,
            "ppl_n_windows": 64,
            "ppl_max_windows": None,
        },
        "results": {},
    }
    zeroshot = {"limit": None, "complete": True, "results": {}}

    def zero_method(score: float) -> dict:
        return {
            "avg_primary_score": score,
            "complete": True,
            "tasks": {task: {} for task in reproduction.ZERO_SHOT_TASKS},
        }

    for rate in reproduction.RATES:
        ppl["results"][str(rate)] = {
            "turboquant": {
                "ppl": 10.0 + rate,
                "ppl_complete_text_eval": True,
                "ppl_n_windows": 64,
                "ppl_max_windows": None,
            },
            "hayatequant": {
                "ppl": 9.0 + rate,
                "ppl_complete_text_eval": True,
                "ppl_n_windows": 64,
                "ppl_max_windows": None,
            },
        }
        zeroshot["results"][str(rate)] = {
            "FullKV": zero_method(0.7),
            "turboquant": zero_method(0.5 + rate / 100),
            "hayatequant": zero_method(0.6 + rate / 100),
        }
    return ppl, zeroshot


def test_summary_is_complete_only_when_every_manifest_cell_exists(tmp_path: Path):
    output_root = tmp_path / "run"
    ppl_path = output_root / "ppl/model/results_model.json"
    zeroshot_path = output_root / "zeroshot/model/zeroshot_model.json"
    ppl, zeroshot = _complete_quality_documents()
    _write_json(ppl_path, ppl)
    _write_json(zeroshot_path, zeroshot)
    manifest = {
        "schema_version": 2,
        "code_commit": "test",
        "output_root": str(output_root),
        "stages": ["ppl", "zeroshot"],
        "protocol": {
            "ppl_protocol": reproduction.PAPER_PPL_PROTOCOL,
            "ppl_complete_split": True,
            "ppl_windows": None,
            "ppl_max_length": 2048,
            "ppl_stride": 1024,
            "zeroshot_limit": None,
            "zeroshot_tasks": list(reproduction.ZERO_SHOT_TASKS),
        },
        "models": [
            {
                "key": "model",
                "display": "Model",
                "ppl_result": str(ppl_path),
                "zeroshot_result": str(zeroshot_path),
            }
        ],
        "jobs": [],
    }

    complete = reproduction.summarize_manifest(manifest)

    assert complete["coverage"]["paper_complete"] is True
    assert complete["coverage"]["expected_cells_per_quality_stage"] == 6
    assert complete["aggregates"]["3"]["hayatequant_ppl_geomean"] == pytest.approx(12.0)
    assert complete["aggregates"]["4"]["turboquant_score_mean"] == pytest.approx(0.54)

    del ppl["results"]["3"]["hayatequant"]
    _write_json(ppl_path, ppl)
    incomplete = reproduction.summarize_manifest(manifest)

    assert incomplete["coverage"]["paper_complete"] is False
    assert "PPL:model:3:hayatequant" in incomplete["coverage"]["missing"]


@pytest.mark.parametrize("incomplete_scope", ["document", "method", "tasks"])
def test_summary_rejects_incomplete_zeroshot_checkpoints(
    tmp_path: Path,
    incomplete_scope: str,
):
    output_root = tmp_path / "run"
    ppl_path = output_root / "ppl/model/results_model.json"
    zeroshot_path = output_root / "zeroshot/model/zeroshot_model.json"
    ppl, zeroshot = _complete_quality_documents()
    if incomplete_scope == "document":
        zeroshot["complete"] = False
    elif incomplete_scope == "method":
        zeroshot["results"]["3"]["hayatequant"]["complete"] = False
    else:
        del zeroshot["results"]["3"]["hayatequant"]["tasks"]["hellaswag"]
    _write_json(ppl_path, ppl)
    _write_json(zeroshot_path, zeroshot)
    manifest = {
        "schema_version": 2,
        "code_commit": "test",
        "output_root": str(output_root),
        "stages": ["ppl", "zeroshot"],
        "protocol": {
            "ppl_protocol": reproduction.PAPER_PPL_PROTOCOL,
            "ppl_complete_split": True,
            "ppl_windows": None,
            "ppl_max_length": 2048,
            "ppl_stride": 1024,
            "zeroshot_limit": None,
            "zeroshot_tasks": list(reproduction.ZERO_SHOT_TASKS),
        },
        "models": [
            {
                "key": "model",
                "display": "Model",
                "ppl_result": str(ppl_path),
                "zeroshot_result": str(zeroshot_path),
            }
        ],
        "jobs": [],
    }

    summary = reproduction.summarize_manifest(manifest)

    assert summary["coverage"]["paper_complete"] is False
    assert any(
        "incomplete" in item or "coverage mismatch" in item
        for item in summary["coverage"]["missing"]
    )


def test_capped_ppl_is_explicitly_diagnostic(tmp_path: Path):
    models_root, model_manifest = _model_manifest(tmp_path)
    args = reproduction.build_parser().parse_args(
        [
            "plan",
            "--models-root",
            str(models_root),
            "--model-manifest",
            str(model_manifest),
            "--output-root",
            str(tmp_path / "diagnostic"),
            "--ppl-windows",
            "32",
        ]
    )

    manifest = reproduction.build_manifest(args)

    assert manifest["protocol"]["ppl_protocol"] == reproduction.CAPPED_PPL_PROTOCOL
    assert manifest["protocol"]["ppl_windows"] == 32
    assert manifest["protocol"]["ppl_complete_split"] is False
    assert all(
        "ppl_windows=32" in job["command"] for job in manifest["jobs"] if job["stage"] == "ppl"
    )


def test_model_manifest_requires_local_config(tmp_path: Path):
    manifest = tmp_path / "models.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "speed_model": "missing",
            "models": [{"key": "missing", "display": "Missing", "ref": "missing"}],
        },
    )

    with pytest.raises(FileNotFoundError, match="config.json"):
        reproduction.load_models(manifest, tmp_path / "models")
