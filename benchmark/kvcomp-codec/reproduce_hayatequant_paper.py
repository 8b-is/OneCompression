"""Plan, submit, and summarize the canonical HayateQuant paper experiments.

All machine-specific locations are explicit CLI inputs.  Generated manifests,
Slurm logs, runtime evidence, and results stay below ``--output-root`` and are
therefore never written into the source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_MANIFEST = REPO_ROOT / "benchmark/configs/hayatequant_models.json"
RATES = (2, 3, 4)
QUALITY_METHODS = ("turboquant", "hayatequant")
ZERO_SHOT_METHODS = ("FullKV", *QUALITY_METHODS)
ZERO_SHOT_TASKS = ("arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande")
STAGES = ("ppl", "zeroshot", "speed")
PAPER_PPL_PROTOCOL = "reference_p_main_wikitext2_raw_test_complete_split_v1"
CAPPED_PPL_PROTOCOL = "wikitext2_raw_test_capped_window_diagnostic_v1"
PAPER_PPL_MAX_LENGTH = 2048
PAPER_PPL_STRIDE = 1024
ZEROSHOT_TOKEN_BOUNDARY_PROTOCOL = (
    "lm_eval_joined_tokenization_with_longest_common_prefix_merge_fallback_v1"
)
SOURCE_ARCHIVE_NAME = "source_snapshot.tar.gz"
SOURCE_PROVENANCE_NAME = "source_provenance.json"
MODEL_MANIFEST_SNAPSHOT_NAME = "model_manifest.snapshot.json"
RUNTIME_CACHE_SUBDIRS = {
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "huggingface",
    "VLLM_CACHE_ROOT": "vllm",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "CUDA_CACHE_PATH": "cuda",
}
VLLM_IPC_PATH_MAX_LEN = 107
VLLM_IPC_NAME_LENGTH = 36


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(model_path: Path) -> dict[str, Any]:
    """Hash the files that define one local Transformers checkpoint."""

    safetensors = sorted(model_path.glob("*.safetensors"))
    pytorch_bins = sorted(model_path.glob("pytorch_model*.bin"))
    weights = safetensors or pytorch_bins
    if not weights:
        raise FileNotFoundError(f"local checkpoint has no supported weight files: {model_path}")
    weight_names = {path.name for path in safetensors + pytorch_bins}
    metadata = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file() and path.name not in weight_names and path.stat().st_size < 100 << 20
    )
    artifacts = []
    for path in [*weights, *metadata]:
        file_stat = path.stat()
        artifacts.append(
            {
                "name": path.name,
                "bytes": file_stat.st_size,
                "mtime_ns": file_stat.st_mtime_ns,
                "sha256": _sha256(path),
                "verify_on_launch": path in metadata,
            }
        )
    stable = [
        {key: artifact[key] for key in ("name", "bytes", "sha256")} for artifact in artifacts
    ]
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return {
        "format": "safetensors" if safetensors else "pytorch_bin",
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    ref: str
    path: Path
    checkpoint: dict[str, Any]

    def manifest_record(self, output_root: Path) -> dict[str, Any]:
        model_tag = self.path.name
        return {
            "key": self.key,
            "display": self.display,
            "ref": self.ref,
            "path": str(self.path),
            "model_config_sha256": _sha256(self.path / "config.json"),
            "checkpoint": self.checkpoint,
            "ppl_result": str(output_root / "ppl" / self.key / f"results_{model_tag}.json"),
            "zeroshot_result": str(
                output_root / "zeroshot" / self.key / f"zeroshot_{model_tag}.json"
            ),
        }


@dataclass
class Job:
    name: str
    stage: str
    command: list[str]
    stdout: str
    stderr: str
    result: str
    workdir: str
    job_id: str | None = None
    status: str = "planned"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON document: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON document must be an object: {path}")
    return payload


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _current_source_files() -> list[Path]:
    """List present tracked and non-ignored untracked repository files."""

    raw_paths = _git_bytes("ls-files", "-co", "--exclude-standard", "-z").split(b"\0")
    relative_paths = [Path(os.fsdecode(raw)) for raw in raw_paths if raw]
    files = [
        relative
        for relative in relative_paths
        if (REPO_ROOT / relative).is_file() or (REPO_ROOT / relative).is_symlink()
    ]
    return sorted(files, key=lambda path: os.fsencode(path.as_posix()))


def _source_snapshot_digest(
    files: list[Path],
    root: Path = REPO_ROOT,
    expected_modes: dict[Path, int] | None = None,
) -> str:
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        metadata = path.lstat()
        encoded_path = os.fsencode(relative.as_posix())
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        mode = (
            expected_modes[relative]
            if expected_modes is not None
            else stat.S_IMODE(metadata.st_mode)
        )
        digest.update(mode.to_bytes(4, "big"))
        if path.is_symlink():
            payload = os.fsencode(os.readlink(path))
            digest.update(b"L")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        else:
            digest.update(b"F")
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _source_state() -> tuple[dict[str, Any], list[Path]]:
    files = _current_source_files()
    snapshot_sha256 = _source_snapshot_digest(files)
    status = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all")
    commit = _git_commit()
    if not commit:
        raise RuntimeError(f"cannot resolve Git commit for {REPO_ROOT}")
    dirty = bool(status)
    revision = commit if not dirty else f"{commit}+dirty.{snapshot_sha256[:16]}"
    return (
        {
            "commit": commit,
            "dirty": dirty,
            "revision": revision,
            "snapshot_sha256": snapshot_sha256,
            "snapshot_file_count": len(files),
            "git_status_sha256": hashlib.sha256(status).hexdigest(),
        },
        files,
    )


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_source_archive(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for relative in files:
                        archive.add(
                            REPO_ROOT / relative,
                            arcname=relative.as_posix(),
                            recursive=False,
                            filter=_normalized_tar_info,
                        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _make_source_tree_read_only(source_root: Path) -> None:
    """Prevent benchmark jobs from accidentally mutating their source copy."""

    descendants = sorted(source_root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in descendants:
        if path.is_symlink():
            continue
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    source_root.chmod(stat.S_IMODE(source_root.stat().st_mode) & ~0o222)


def _source_archive_modes(archive_path: Path) -> dict[Path, int]:
    modes: dict[Path, int] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.issym()):
                continue
            relative = Path(member.name)
            if relative in modes:
                raise RuntimeError(f"source archive has a duplicate member: {relative}")
            modes[relative] = member.mode
    return modes


def _extract_source_archive(
    archive_path: Path,
    source_root: Path,
    snapshot_sha256: str,
    files: list[Path],
) -> None:
    """Materialize one immutable, content-addressed source directory."""

    marker_name = ".onecomp_source_snapshot_sha256"
    marker = source_root / marker_name
    archive_modes = _source_archive_modes(archive_path)
    if set(archive_modes) != set(files):
        raise RuntimeError("source archive members do not match the planned source files")
    if source_root.exists():
        try:
            recorded = marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"existing source snapshot is unverifiable: {source_root}") from exc
        if recorded != snapshot_sha256:
            raise RuntimeError(f"existing source snapshot has the wrong identity: {source_root}")
        if _source_snapshot_digest(files, source_root, archive_modes) != snapshot_sha256:
            raise RuntimeError(f"existing source snapshot content changed: {source_root}")
        return

    source_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{source_root.name}.", dir=source_root.parent))
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(temporary, filter="data")
        (temporary / marker_name).write_text(snapshot_sha256 + "\n", encoding="utf-8")
        if _source_snapshot_digest(files, temporary, archive_modes) != snapshot_sha256:
            raise RuntimeError("extracted source snapshot does not match its planned identity")
        _make_source_tree_read_only(temporary)
        os.replace(temporary, source_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _resolve_executable(value: str) -> str:
    if os.sep not in value:
        resolved = shutil.which(value)
        if resolved is None:
            raise FileNotFoundError(f"executable is not on PATH: {value}")
        return resolved
    # Keep a virtualenv launcher symlink intact. Resolving it to the base
    # interpreter can bypass that environment's pyvenv.cfg and site-packages.
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"executable does not exist or is not executable: {candidate}")
    return str(candidate)


def load_models(manifest_path: Path, models_root: Path) -> tuple[list[ModelSpec], str]:
    document = _load_json(manifest_path)
    if document.get("schema_version") != 1 or not isinstance(document.get("models"), list):
        raise RuntimeError(f"unsupported model manifest schema: {manifest_path}")

    models: list[ModelSpec] = []
    seen_keys: set[str] = set()
    seen_displays: set[str] = set()
    for row in document["models"]:
        if not isinstance(row, dict):
            raise RuntimeError(f"model entries must be objects: {manifest_path}")
        try:
            key = str(row["key"])
            display = str(row["display"])
            ref = str(row["ref"])
        except KeyError as exc:
            raise RuntimeError(f"model entry is missing {exc.args[0]!r}: {manifest_path}") from exc
        if not key or not display or not ref or key in seen_keys or display in seen_displays:
            raise RuntimeError(f"invalid or duplicate model entry {row!r}: {manifest_path}")
        unresolved = Path(ref).expanduser()
        model_path = unresolved if unresolved.is_absolute() else models_root / unresolved
        model_path = model_path.resolve()
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"model {key!r} does not resolve to a local checkpoint with config.json: "
                f"{model_path}"
            )
        models.append(
            ModelSpec(
                key=key,
                display=display,
                ref=ref,
                path=model_path,
                checkpoint=_checkpoint_identity(model_path),
            )
        )
        seen_keys.add(key)
        seen_displays.add(display)

    speed_model = str(document.get("speed_model", ""))
    if speed_model not in seen_keys:
        raise RuntimeError(f"speed_model {speed_model!r} is not one of the model keys")
    if not models:
        raise RuntimeError("model manifest is empty")
    return models, speed_model


def _job(
    *,
    output_root: Path,
    name: str,
    stage: str,
    command: Iterable[str],
    result: Path,
    workdir: Path,
) -> Job:
    logs = output_root / "logs"
    return Job(
        name=name,
        stage=stage,
        command=[str(item) for item in command],
        stdout=str(logs / f"{name}.out"),
        stderr=str(logs / f"{name}.err"),
        result=str(result),
        workdir=str(workdir),
    )


def _runtime_path_environment(executable: str) -> str:
    executable_dir = str(Path(executable).parent)
    inherited = os.environ.get("PATH", os.defpath)
    entries = [entry for entry in inherited.split(os.pathsep) if entry != executable_dir]
    return f"PATH={os.pathsep.join((executable_dir, *entries))}"


def _runtime_tmp_directory(output_root: Path) -> Path:
    """Place vLLM's Unix sockets under the nearest short writable ancestor."""

    resolved = output_root.resolve()
    identity = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:8]
    for ancestor in resolved.parents:
        candidate = ancestor / ".onecomp-tmp" / identity
        socket_path = candidate / ("x" * VLLM_IPC_NAME_LENGTH)
        if len(os.fsencode(socket_path)) > VLLM_IPC_PATH_MAX_LEN:
            continue
        if not os.access(ancestor, os.W_OK | os.X_OK):
            continue
        return candidate
    raise RuntimeError(
        "cannot place vLLM IPC sockets below a writable output ancestor; "
        "choose a shorter --output-root"
    )


def _runtime_cache_environment(output_root: Path) -> list[str]:
    cache_root = output_root.resolve() / "runtime-cache"
    environment = [
        f"{variable}={cache_root / subdirectory}"
        for variable, subdirectory in RUNTIME_CACHE_SUBDIRS.items()
    ]
    environment.append(f"TMPDIR={_runtime_tmp_directory(output_root)}")
    return environment


def _prepare_runtime_cache(output_root: Path) -> None:
    cache_root = output_root.resolve() / "runtime-cache"
    for subdirectory in RUNTIME_CACHE_SUBDIRS.values():
        (cache_root / subdirectory).mkdir(parents=True, exist_ok=True)
    runtime_tmp = _runtime_tmp_directory(output_root)
    runtime_tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
    if runtime_tmp.is_symlink() or not runtime_tmp.is_dir():
        raise RuntimeError(f"unsafe runtime TMPDIR: {runtime_tmp}")


def build_jobs(
    args: argparse.Namespace,
    models: list[ModelSpec],
    speed_key: str,
    source_provenance: Path,
    source_root: Path,
) -> list[Job]:
    output_root = args.output_root.resolve()
    python = _resolve_executable(args.python)
    vllm_python = _resolve_executable(args.vllm_python)
    codec_script = source_root / "benchmark/kvcomp-codec/codec_benchmark.py"
    zeroshot_script = source_root / "benchmark/kvcomp-codec/zeroshot_benchmark.py"
    speed_script = source_root / "benchmark/kvcomp-codec/vllm_benchmark.py"
    base_env_prefix = [
        _resolve_executable("env"),
        f"PAPER_PROVENANCE_PATH={source_provenance}",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={source_root}",
        *_runtime_cache_environment(output_root),
    ]
    quality_env_prefix = [
        *base_env_prefix,
        _runtime_path_environment(python),
    ]
    speed_env_prefix = [
        *base_env_prefix,
        _runtime_path_environment(vllm_python),
    ]
    jobs: list[Job] = []

    if "ppl" in args.stages:
        for model in models:
            result = output_root / "ppl" / model.key / f"results_{model.path.name}.json"
            command = [
                *quality_env_prefix,
                python,
                str(codec_script),
                "--config-name",
                "paper_hayatequant",
                f"model_path={model.path}",
                f"output_dir={result.parent}",
                f"dtype={args.quality_dtype}",
            ]
            if args.ppl_windows is not None:
                command.append(f"ppl_windows={args.ppl_windows}")
            jobs.append(
                _job(
                    output_root=output_root,
                    name=f"hq-ppl-{model.key}",
                    stage="ppl",
                    command=command,
                    result=result,
                    workdir=source_root,
                )
            )

    if "zeroshot" in args.stages:
        for model in models:
            result = output_root / "zeroshot" / model.key / f"zeroshot_{model.path.name}.json"
            command = [
                *quality_env_prefix,
                python,
                str(zeroshot_script),
                "--model-path",
                str(model.path),
                "--output-root",
                str(result.parent),
                "--bits",
                *(str(rate) for rate in RATES),
                "--methods",
                *ZERO_SHOT_METHODS,
                "--include-fullkv",
                "--dtype",
                args.quality_dtype,
            ]
            if args.zero_limit is not None:
                command.extend(["--limit", str(args.zero_limit)])
            jobs.append(
                _job(
                    output_root=output_root,
                    name=f"hq-zs-{model.key}",
                    stage="zeroshot",
                    command=command,
                    result=result,
                    workdir=source_root,
                )
            )

    if "speed" in args.stages:
        speed_model = next(model for model in models if model.key == speed_key)
        pairs: list[tuple[str, int | None]] = [
            ("fullkv", None),
            ("turboquant", 2),
            ("turboquant", 3),
            ("turboquant", 4),
            ("hayatequant", 2),
            ("hayatequant", 3),
            ("hayatequant", 4),
        ]
        if args.include_taperkv_speed:
            pairs.append(("taperkv", None))
        for method, rate in pairs:
            suffix = method if rate is None else f"{method}-r{rate}"
            result = output_root / "speed" / f"{suffix}.json"
            command = [
                *speed_env_prefix,
                vllm_python,
                str(speed_script),
                "--model",
                str(speed_model.path),
                "--output",
                str(result),
                "--method",
                method,
                "--prompt-tokens",
                str(args.prompt_tokens),
                "--max-tokens",
                str(args.max_tokens),
                "--repeats",
                str(args.repeats),
                "--dtype",
                args.speed_dtype,
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            if rate is not None:
                command.extend(["--rate", str(rate)])
            if args.enforce_eager:
                command.append("--enforce-eager")
            jobs.append(
                _job(
                    output_root=output_root,
                    name=f"hq-speed-{suffix}",
                    stage="speed",
                    command=command,
                    result=result,
                    workdir=source_root,
                )
            )
    return jobs


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.model_manifest.expanduser().resolve()
    models_root = args.models_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    models, speed_key = load_models(manifest_path, models_root)
    code, _ = _source_state()
    source_provenance = output_root / SOURCE_PROVENANCE_NAME
    source_root = output_root / f"source-{code['snapshot_sha256'][:16]}"
    jobs = build_jobs(args, models, speed_key, source_provenance, source_root)
    return {
        "schema_version": 2,
        "created_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "code_commit": code["commit"],
        "code": code,
        "source_provenance": str(source_provenance),
        "source_archive": str(output_root / SOURCE_ARCHIVE_NAME),
        "source_root": str(source_root),
        "model_manifest": str(manifest_path),
        "model_manifest_snapshot": str(output_root / MODEL_MANIFEST_SNAPSHOT_NAME),
        "model_manifest_sha256": _sha256(manifest_path),
        "models_root": str(models_root),
        "output_root": str(output_root),
        "stages": list(args.stages),
        "models": [model.manifest_record(output_root) for model in models],
        "speed_model": speed_key,
        "protocol": {
            "rates": list(RATES),
            "quality_methods": list(QUALITY_METHODS),
            "quality_dtype": args.quality_dtype,
            "ppl_dataset": "WikiText-2 raw test",
            "ppl_protocol": (
                PAPER_PPL_PROTOCOL if args.ppl_windows is None else CAPPED_PPL_PROTOCOL
            ),
            "ppl_complete_split": args.ppl_windows is None,
            "ppl_windows": args.ppl_windows,
            "ppl_max_length": PAPER_PPL_MAX_LENGTH,
            "ppl_stride": PAPER_PPL_STRIDE,
            "zeroshot_tasks": list(ZERO_SHOT_TASKS),
            "zeroshot_limit": args.zero_limit,
            "zeroshot_token_boundary_protocol": ZEROSHOT_TOKEN_BOUNDARY_PROTOCOL,
            "speed_dtype": args.speed_dtype,
            "speed_prompt_tokens": args.prompt_tokens,
            "speed_generated_tokens": args.max_tokens,
            "speed_repeats": args.repeats,
            "turboquant_vllm_rates": list(RATES),
            "hayatequant_vllm_rates": list(RATES),
            "taperkv_speed_included": bool(args.include_taperkv_speed),
        },
        "jobs": [asdict(job) for job in jobs],
        "slurm": None,
    }


def _write_source_provenance(manifest: dict[str, Any]) -> None:
    """Archive and attest the exact source snapshot used by planned jobs."""

    current_code, files = _source_state()
    if current_code != manifest.get("code"):
        raise RuntimeError("repository source changed while constructing the run manifest")
    archive_path = Path(manifest["source_archive"])
    _write_source_archive(archive_path, files)
    source_root = Path(manifest["source_root"])
    _extract_source_archive(
        archive_path,
        source_root,
        current_code["snapshot_sha256"],
        files,
    )
    model_manifest_snapshot = Path(manifest["model_manifest_snapshot"])
    _atomic_copy(Path(manifest["model_manifest"]), model_manifest_snapshot)
    if _sha256(model_manifest_snapshot) != manifest["model_manifest_sha256"]:
        raise RuntimeError("model manifest changed while constructing source provenance")
    record = {
        "schema_version": 1,
        "created_utc": manifest["created_utc"],
        "code": {
            **current_code,
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "source_root": str(source_root),
        },
        "model_manifest": {
            "path": manifest["model_manifest"],
            "snapshot": str(model_manifest_snapshot),
            "sha256": manifest["model_manifest_sha256"],
        },
        "models": [
            {
                "key": model["key"],
                "ref": model["ref"],
                "path": model["path"],
                "model_config_sha256": model["model_config_sha256"],
                "checkpoint": model["checkpoint"],
            }
            for model in manifest["models"]
        ],
    }
    provenance_path = Path(manifest["source_provenance"])
    _atomic_json(provenance_path, record)
    provenance_sha256 = _sha256(provenance_path)
    manifest["source_provenance_sha256"] = provenance_sha256
    manifest["source_archive_sha256"] = record["code"]["archive_sha256"]
    provenance_variable = f"PAPER_PROVENANCE_PATH={provenance_path}"
    for job in manifest["jobs"]:
        try:
            index = job["command"].index(provenance_variable)
        except ValueError as exc:
            raise RuntimeError(f"job {job['name']} has no provenance environment") from exc
        job["command"].insert(index + 1, f"PAPER_PROVENANCE_SHA256={provenance_sha256}")


def _manifest_path(output_root: Path) -> Path:
    return output_root.expanduser().resolve() / "run_manifest.json"


def _refuse_lost_submissions(path: Path, *, resubmit: bool) -> None:
    if not path.exists() or resubmit:
        return
    old = _load_json(path)
    submitted = [job for job in old.get("jobs", []) if job.get("job_id")]
    if submitted:
        raise RuntimeError(
            f"{path} already records {len(submitted)} submitted jobs; "
            "use --resubmit only after checking their scheduler state"
        )


def command_plan(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    path = _manifest_path(args.output_root)
    _refuse_lost_submissions(path, resubmit=args.overwrite_submitted_manifest)
    _write_source_provenance(manifest)
    _atomic_json(path, manifest)
    counts = {stage: sum(job["stage"] == stage for job in manifest["jobs"]) for stage in STAGES}
    print(f"wrote {path}")
    print(f"planned {len(manifest['jobs'])} jobs: {counts}")
    return 0


def _sbatch_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    command = [
        _resolve_executable(args.sbatch),
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.memory}",
        f"--time={args.time_limit}",
        f"--gres={args.gres}",
        f"--job-name={job['name']}",
        f"--output={job['stdout']}",
        f"--error={job['stderr']}",
        f"--chdir={job['workdir']}",
    ]
    if args.partition:
        command.append(f"--partition={args.partition}")
    if args.account:
        command.append(f"--account={args.account}")
    command.extend(["--wrap", shlex.join(job["command"])])
    return command


def _validate_planned_model(model: dict[str, Any]) -> None:
    root = Path(model["path"]).expanduser().resolve()
    if _sha256(root / "config.json") != model.get("model_config_sha256"):
        raise RuntimeError(f"planned model config changed: {root}")
    checkpoint = model.get("checkpoint")
    artifacts = checkpoint.get("artifacts") if isinstance(checkpoint, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"planned model has no checkpoint artifacts: {root}")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError(f"malformed planned checkpoint artifact: {root}")
        name = artifact.get("name")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise RuntimeError(f"unsafe planned checkpoint artifact: {name!r}")
        path = root / name
        try:
            file_stat = path.stat()
        except OSError as exc:
            raise RuntimeError(f"planned checkpoint artifact is unavailable: {path}") from exc
        if file_stat.st_size != artifact.get("bytes"):
            raise RuntimeError(f"planned checkpoint artifact size changed: {path}")
        if artifact.get("verify_on_launch"):
            if _sha256(path) != artifact.get("sha256"):
                raise RuntimeError(f"planned checkpoint metadata changed: {path}")
        elif file_stat.st_mtime_ns != artifact.get("mtime_ns"):
            raise RuntimeError(f"planned checkpoint weight changed: {path}")


def _validate_existing_plan(path: Path, manifest: dict[str, Any]) -> None:
    """Validate one frozen plan without consulting the mutable checkout."""

    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"unsupported run manifest schema: {path}")
    output_root = Path(manifest.get("output_root", "")).expanduser().resolve()
    if output_root != path.parent.resolve():
        raise RuntimeError("run manifest output_root does not match its directory")
    code = manifest.get("code")
    if not isinstance(code, dict) or not isinstance(code.get("snapshot_sha256"), str):
        raise RuntimeError("run manifest has no source snapshot identity")
    snapshot_sha256 = code["snapshot_sha256"]
    source_root = Path(manifest.get("source_root", "")).expanduser().resolve()
    expected_source_root = output_root / f"source-{snapshot_sha256[:16]}"
    if source_root != expected_source_root:
        raise RuntimeError("run manifest has a non-canonical frozen source root")
    archive = Path(manifest.get("source_archive", "")).expanduser().resolve()
    provenance = Path(manifest.get("source_provenance", "")).expanduser().resolve()
    if archive != output_root / SOURCE_ARCHIVE_NAME:
        raise RuntimeError("run manifest has a non-canonical source archive path")
    if provenance != output_root / SOURCE_PROVENANCE_NAME:
        raise RuntimeError("run manifest has a non-canonical source provenance path")
    if _sha256(archive) != manifest.get("source_archive_sha256"):
        raise RuntimeError("source archive changed after planning")
    if _sha256(provenance) != manifest.get("source_provenance_sha256"):
        raise RuntimeError("source provenance changed after planning")
    provenance_record = _load_json(provenance)
    if provenance_record.get("code", {}).get("snapshot_sha256") != snapshot_sha256:
        raise RuntimeError("source provenance and run manifest disagree")
    if provenance_record.get("code", {}).get("archive_sha256") != manifest.get(
        "source_archive_sha256"
    ):
        raise RuntimeError("source provenance has the wrong archive identity")
    marker = source_root / ".onecomp_source_snapshot_sha256"
    try:
        recorded_snapshot = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"frozen source root is unavailable: {source_root}") from exc
    if recorded_snapshot != snapshot_sha256:
        raise RuntimeError("frozen source marker has the wrong identity")
    archive_modes = _source_archive_modes(archive)
    source_files = sorted(archive_modes, key=lambda item: os.fsencode(item.as_posix()))
    if _source_snapshot_digest(source_files, source_root, archive_modes) != snapshot_sha256:
        raise RuntimeError("frozen source content changed after planning")
    if source_root.stat().st_mode & 0o222:
        raise RuntimeError("frozen source root is writable")
    model_manifest = Path(manifest.get("model_manifest_snapshot", "")).expanduser().resolve()
    if model_manifest != output_root / MODEL_MANIFEST_SNAPSHOT_NAME:
        raise RuntimeError("run manifest has a non-canonical model manifest snapshot")
    if _sha256(model_manifest) != manifest.get("model_manifest_sha256"):
        raise RuntimeError("model manifest snapshot changed after planning")
    for model in manifest.get("models", []):
        if not isinstance(model, dict):
            raise RuntimeError("run manifest has a malformed model entry")
        _validate_planned_model(model)
    expected_provenance_env = f"PAPER_PROVENANCE_SHA256={manifest['source_provenance_sha256']}"
    expected_provenance_path = f"PAPER_PROVENANCE_PATH={provenance}"
    expected_python_path = f"PYTHONPATH={source_root}"
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict) or job.get("workdir") != str(source_root):
            raise RuntimeError("planned job does not execute in the frozen source root")
        command = job.get("command")
        required_environment = {
            expected_provenance_path,
            expected_provenance_env,
            expected_python_path,
            "PYTHONDONTWRITEBYTECODE=1",
            *_runtime_cache_environment(output_root),
        }
        if not isinstance(command, list) or not required_environment.issubset(command):
            raise RuntimeError("planned job has no fixed provenance identity")
        runtime_paths = [item for item in command if item.startswith("PATH=")]
        if len(runtime_paths) != 1:
            raise RuntimeError("planned job has no unique runtime PATH")
        interpreter = next(
            (item for item in command[1:] if "=" not in item and Path(item).is_absolute()),
            None,
        )
        path_entries = runtime_paths[0].removeprefix("PATH=").split(os.pathsep)
        if (
            interpreter is None
            or not path_entries
            or path_entries[0] != str(Path(interpreter).parent)
        ):
            raise RuntimeError("planned job PATH does not expose its Python environment")
        for field in ("stdout", "stderr", "result"):
            target = Path(job.get(field, "")).expanduser().resolve()
            if not target.is_relative_to(output_root):
                raise RuntimeError(f"planned job {field} escapes the output root")


def _submit_jobs(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    path: Path,
) -> int:
    _prepare_runtime_cache(Path(manifest["output_root"]))
    manifest["slurm"] = {
        "partition": args.partition,
        "account": args.account,
        "gres": args.gres,
        "cpus_per_task": args.cpus_per_task,
        "memory": args.memory,
        "time_limit": args.time_limit,
    }
    Path(manifest["output_root"], "logs").mkdir(parents=True, exist_ok=True)
    if args.resubmit:
        for job in manifest["jobs"]:
            job["job_id"] = None
            job["status"] = "planned"
            job.pop("submission_error", None)
    _atomic_json(path, manifest)
    for job in manifest["jobs"]:
        submit = _sbatch_command(args, job)
        try:
            completed = subprocess.run(submit, check=True, capture_output=True, text=True)
            job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
            if not job_id:
                raise RuntimeError("sbatch returned an empty job id")
            job["job_id"] = job_id
            job["status"] = "submitted"
            print(f"submitted {job['name']}: {job_id}", flush=True)
        except Exception as exc:
            job["status"] = "submission_failed"
            job["submission_error"] = str(exc)
            _atomic_json(path, manifest)
            raise
        _atomic_json(path, manifest)
    return 0


def command_submit(args: argparse.Namespace) -> int:
    path = _manifest_path(args.output_root)
    _refuse_lost_submissions(path, resubmit=args.resubmit)
    manifest = build_manifest(args)
    _write_source_provenance(manifest)
    return _submit_jobs(args, manifest, path)


def command_submit_plan(args: argparse.Namespace) -> int:
    path = _manifest_path(args.output_root)
    _refuse_lost_submissions(path, resubmit=args.resubmit)
    manifest = _load_json(path)
    _validate_existing_plan(path, manifest)
    return _submit_jobs(args, manifest, path)


def _nested(document: dict[str, Any], *keys: str) -> Any:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _arithmetic_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _result_or_missing(path: Path, missing: list[str], label: str) -> dict[str, Any] | None:
    try:
        return _load_json(path)
    except FileNotFoundError:
        missing.append(f"{label}: missing file {path}")
    except RuntimeError as exc:
        missing.append(f"{label}: {exc}")
    return None


def summarize_manifest(run_manifest: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    stages = set(run_manifest.get("stages", []))
    protocol = run_manifest.get("protocol", {})
    limited = protocol.get("zeroshot_limit") is not None
    expected_ppl_windows = protocol.get("ppl_windows")
    paper_ppl_protocol = (
        protocol.get("ppl_protocol") == PAPER_PPL_PROTOCOL
        and expected_ppl_windows is None
        and protocol.get("ppl_complete_split") is True
        and protocol.get("ppl_max_length") == PAPER_PPL_MAX_LENGTH
        and protocol.get("ppl_stride") == PAPER_PPL_STRIDE
    )
    expected_zeroshot_tasks = tuple(protocol.get("zeroshot_tasks", ()))

    def validate_ppl_entry(entry: Any, label: str) -> None:
        if not isinstance(entry, dict):
            missing.append(f"{label}: missing result entry")
            return
        if expected_ppl_windows is None:
            if not bool(entry.get("ppl_complete_text_eval")):
                missing.append(f"{label}: incomplete full-split result")
            return
        if entry.get("ppl_n_windows") != expected_ppl_windows:
            missing.append(
                f"{label}: expected {expected_ppl_windows} windows, "
                f"got {entry.get('ppl_n_windows')!r}"
            )
        if entry.get("ppl_max_windows") != expected_ppl_windows:
            missing.append(f"{label}: max-window contract does not equal {expected_ppl_windows}")

    def validate_zeroshot_entry(entry: Any, label: str) -> None:
        if not isinstance(entry, dict):
            missing.append(f"{label}: missing result entry")
            return
        if entry.get("complete") is not True:
            missing.append(f"{label}: incomplete method checkpoint")
        tasks = entry.get("tasks")
        if not isinstance(tasks, dict):
            missing.append(f"{label}: missing task records")
            return
        expected = set(expected_zeroshot_tasks)
        actual = set(tasks)
        if actual != expected:
            missing_tasks = sorted(expected - actual)
            unexpected_tasks = sorted(actual - expected)
            missing.append(
                f"{label}: task coverage mismatch "
                f"missing={missing_tasks} unexpected={unexpected_tasks}"
            )

    for model in run_manifest.get("models", []):
        ppl = None
        zeroshot = None
        if "ppl" in stages:
            ppl = _result_or_missing(Path(model["ppl_result"]), missing, f"PPL:{model['key']}")
            if ppl is not None:
                full = _finite_number(_nested(ppl, "fullkv_ppl", "ppl"))
                if full is None:
                    missing.append(f"PPL:{model['key']}:FullKV")
                validate_ppl_entry(ppl.get("fullkv_ppl"), f"PPL:{model['key']}:FullKV")
        if "zeroshot" in stages:
            zeroshot = _result_or_missing(
                Path(model["zeroshot_result"]), missing, f"zero-shot:{model['key']}"
            )
            if zeroshot is not None:
                if zeroshot.get("complete") is not True:
                    missing.append(f"zero-shot:{model['key']}: incomplete result document")
                if zeroshot.get("limit") is not None:
                    limited = True

        for rate in RATES:
            row: dict[str, Any] = {
                "model_key": model["key"],
                "model": model["display"],
                "rate": rate,
            }
            if ppl is not None:
                row["fullkv_ppl"] = _finite_number(_nested(ppl, "fullkv_ppl", "ppl"))
            if zeroshot is not None:
                fullkv_entry = _nested(zeroshot, "results", str(rate), "FullKV")
                row["fullkv_score"] = _finite_number(
                    fullkv_entry.get("avg_primary_score")
                    if isinstance(fullkv_entry, dict)
                    else None
                )
                if row["fullkv_score"] is None:
                    missing.append(f"zero-shot:{model['key']}:{rate}:FullKV")
                validate_zeroshot_entry(
                    fullkv_entry,
                    f"zero-shot:{model['key']}:{rate}:FullKV",
                )
            for method in QUALITY_METHODS:
                if ppl is not None:
                    entry = _nested(ppl, "results", str(rate), method)
                    value = _finite_number(entry.get("ppl") if isinstance(entry, dict) else None)
                    row[f"{method}_ppl"] = value
                    if value is None:
                        missing.append(f"PPL:{model['key']}:{rate}:{method}")
                    else:
                        validate_ppl_entry(entry, f"PPL:{model['key']}:{rate}:{method}")
                if zeroshot is not None:
                    entry = _nested(zeroshot, "results", str(rate), method)
                    value = _finite_number(
                        entry.get("avg_primary_score") if isinstance(entry, dict) else None
                    )
                    row[f"{method}_score"] = value
                    if value is None:
                        missing.append(f"zero-shot:{model['key']}:{rate}:{method}")
                    validate_zeroshot_entry(
                        entry,
                        f"zero-shot:{model['key']}:{rate}:{method}",
                    )
            rows.append(row)

    aggregates: dict[str, dict[str, float | None]] = {}
    for rate in RATES:
        selected = [row for row in rows if row["rate"] == rate]
        aggregate: dict[str, float | None] = {}
        for method in QUALITY_METHODS:
            ppls = [row[f"{method}_ppl"] for row in selected if row.get(f"{method}_ppl")]
            scores = [
                row[f"{method}_score"]
                for row in selected
                if row.get(f"{method}_score") is not None
            ]
            aggregate[f"{method}_ppl_geomean"] = _geometric_mean(ppls)
            aggregate[f"{method}_score_mean"] = _arithmetic_mean(scores)
        aggregates[str(rate)] = aggregate

    speed: list[dict[str, Any]] = []
    if "speed" in stages:
        for job in run_manifest.get("jobs", []):
            if job.get("stage") != "speed":
                continue
            result = _result_or_missing(Path(job["result"]), missing, f"speed:{job['name']}")
            if result is not None:
                speed.append(
                    {
                        "job": job["name"],
                        "method": result.get("method"),
                        "rate": result.get("rate"),
                        "median_e2e_tokens_per_second": result.get("median_e2e_tokens_per_second"),
                        "median_decode_tokens_per_second": result.get(
                            "median_decode_tokens_per_second"
                        ),
                        "result": job["result"],
                    }
                )

    missing = sorted(set(missing))
    required_quality_stages = {"ppl", "zeroshot"}
    paper_complete = (
        required_quality_stages.issubset(stages)
        and not missing
        and not limited
        and paper_ppl_protocol
    )
    return {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "code_commit": run_manifest.get("code_commit"),
        "code": run_manifest.get("code"),
        "source_provenance": run_manifest.get("source_provenance"),
        "source_provenance_sha256": run_manifest.get("source_provenance_sha256"),
        "source_archive": run_manifest.get("source_archive"),
        "source_archive_sha256": run_manifest.get("source_archive_sha256"),
        "source_root": run_manifest.get("source_root"),
        "run_manifest": str(Path(run_manifest["output_root"]).resolve() / "run_manifest.json"),
        "protocol": protocol,
        "rows": rows,
        "aggregates": aggregates,
        "speed": speed,
        "coverage": {
            "models": len(run_manifest.get("models", [])),
            "rates": len(RATES),
            "quality_methods": len(QUALITY_METHODS),
            "expected_cells_per_quality_stage": (
                len(run_manifest.get("models", [])) * len(RATES) * len(QUALITY_METHODS)
            ),
            "limited_zeroshot_diagnostic": limited,
            "ppl_is_capped_subset": expected_ppl_windows is not None,
            "paper_ppl_protocol_match": paper_ppl_protocol,
            "required_quality_stages_present": required_quality_stages.issubset(stages),
            "missing": missing,
            "paper_complete": paper_complete,
        },
    }


def command_summarize(args: argparse.Namespace) -> int:
    manifest_path = _manifest_path(args.output_root)
    run_manifest = _load_json(manifest_path)
    if run_manifest.get("schema_version") != 2:
        raise RuntimeError(f"unsupported run manifest schema: {manifest_path}")
    summary = summarize_manifest(run_manifest)
    summary_path = args.output_root.expanduser().resolve() / "summary.json"
    _atomic_json(summary_path, summary)
    coverage = summary["coverage"]
    print(f"wrote {summary_path}")
    print(f"paper_complete={coverage['paper_complete']} " f"missing={len(coverage['missing'])}")
    if not coverage["paper_complete"] and not args.allow_incomplete:
        return 1
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--vllm-python", default=sys.executable)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--quality-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--speed-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--ppl-windows",
        type=int,
        help=(
            "cap WikiText-2 windows for a diagnostic run; omitted means the "
            "complete reference P-main split"
        ),
    )
    parser.add_argument("--zero-limit", type=int)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--include-taperkv-speed", action="store_true")


def _add_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sbatch", default="sbatch")
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--memory", default="128G")
    parser.add_argument("--time-limit", default="12:00:00")
    parser.add_argument("--resubmit", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate inputs and write run_manifest.json")
    _add_run_arguments(plan)
    plan.add_argument("--overwrite-submitted-manifest", action="store_true")
    plan.set_defaults(handler=command_plan)

    submit = commands.add_parser("submit", help="submit the validated plan with sbatch")
    _add_run_arguments(submit)
    _add_slurm_arguments(submit)
    submit.set_defaults(handler=command_submit)

    submit_plan = commands.add_parser(
        "submit-plan",
        help="verify and submit an existing frozen run_manifest.json",
    )
    submit_plan.add_argument("--output-root", type=Path, required=True)
    _add_slurm_arguments(submit_plan)
    submit_plan.set_defaults(handler=command_submit_plan)

    summarize = commands.add_parser("summarize", help="strictly aggregate completed results")
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.add_argument("--allow-incomplete", action="store_true")
    summarize.set_defaults(handler=command_summarize)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("ppl_windows", "zero_limit", "prompt_tokens", "max_tokens", "repeats"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    utilization = getattr(args, "gpu_memory_utilization", None)
    if utilization is not None and not 0 < utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
