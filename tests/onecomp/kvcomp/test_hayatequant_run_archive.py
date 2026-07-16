"""Security and integrity tests for the repo-local HayateQuant run archiver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmark/kvcomp-codec/archive_hayatequant_run.py"
SPEC = importlib.util.spec_from_file_location("hayatequant_run_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
archiver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = archiver
SPEC.loader.exec_module(archiver)

SOURCE_COMMIT = "a" * 40
ARCHIVER_COMMIT = "b" * 40


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, document: object) -> Path:
    return _write(
        path,
        (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_run(root: Path, *, include_optional: bool = False) -> Path:
    run = root / "external-run"
    _json(
        run / "results/formal/ppl/metrics.json",
        {
            "n_tokens": 1024,
            "input_tokens": 4096,
            "sha256": "c" * 64,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        },
    )
    input_manifest = _json(run / "manifests/paper-inputs.json", {"source_commit": SOURCE_COMMIT})
    preflight = _json(
        run / "provenance/worker-preflight.json",
        {"kind": "paper_preflight", "source_commit": SOURCE_COMMIT},
    )
    if include_optional:
        smoke = _json(run / "results/smoke/smoke.json", {"status": "ok"})
        _json(
            run / "manifests/formal_smoke_approval.json",
            {
                "schema_version": 1,
                "kind": "hayatequant_formal_smoke_approval",
                "approval_statement": "I_APPROVE_THE_VALIDATED_HAYATEQUANT_SMOKE_PLAN",
                "validated_plan": {
                    "input_manifest": {
                        "path": str(input_manifest.resolve()),
                        "bytes": input_manifest.stat().st_size,
                        "sha256": _digest(input_manifest),
                    },
                    "results": [
                        {
                            "key": "ppl",
                            "path": str(smoke.resolve()),
                            "bytes": smoke.stat().st_size,
                            "sha256": _digest(smoke),
                        }
                    ],
                    "native_worker_preflights": [
                        {
                            "path": str(preflight.resolve()),
                            "bytes": preflight.stat().st_size,
                            "sha256": _digest(preflight),
                        }
                    ],
                },
            },
        )
        _json(run / "results/dev/probe.json", {"development_only": True})
        _json(run / "summaries/coverage.json", {"complete": True})
        log = _write(run / "logs/formal/job.out", b"formal job completed\n")
        log.chmod(0o755)
    return run


def _archive(
    run: Path,
    archive_root: Path,
    *,
    run_id: str = "formal-20260716",
    required_scopes=(),
    archive_command=None,
) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    return archiver.archive_run(
        run_root=run,
        archive_root=archive_root,
        run_id=run_id,
        source_commit=SOURCE_COMMIT,
        archiver_commit=ARCHIVER_COMMIT,
        required_scopes=required_scopes,
        archive_command=archive_command,
    )


def test_archive_copies_allowlist_and_emits_verified_deterministic_inventory(tmp_path):
    run = _minimal_run(tmp_path, include_optional=True)
    # These large runtime artifacts are deliberately outside the archive allow-list.
    _write(run / "data/hf_home/hub/models--org--model/blobs/abc", b"cached model")
    _write(run / "runtime/home/.cache/model.safetensors", b"model weights")

    destination = _archive(
        run,
        tmp_path / "repo-results",
        required_scopes=(
            "results/smoke",
            "results/formal",
            "summaries",
            "manifests",
            "provenance",
            "logs",
        ),
        archive_command=("python", "archive_hayatequant_run.py", "--run-id", "formal-20260716"),
    )

    manifest = json.loads((destination / "ARCHIVE_MANIFEST.json").read_text())
    records = manifest["files"]
    paths = [record["path"] for record in records]
    assert paths == sorted(paths)
    assert paths == [
        "logs/formal/job.out",
        "manifests/formal_smoke_approval.json",
        "manifests/paper-inputs.json",
        "provenance/worker-preflight.json",
        "results/dev/probe.json",
        "results/formal/ppl/metrics.json",
        "results/smoke/smoke.json",
        "summaries/coverage.json",
    ]
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["archiver_commit"] == ARCHIVER_COMMIT
    assert manifest["missing_scopes"] == []
    assert manifest["file_count"] == len(records)
    assert manifest["total_bytes"] == sum(record["bytes"] for record in records)
    assert not (destination / "data").exists()
    assert not (destination / "runtime").exists()
    assert not (destination / ".archive-incomplete").exists()

    for record in records:
        archived = destination.joinpath(*Path(record["path"]).parts)
        assert archived.stat().st_size == record["bytes"]
        assert _digest(archived) == record["sha256"]
    assert (destination / "logs/formal/job.out").stat().st_mode & 0o111

    # The approval marker's formerly external references can be relocated to
    # the archive and revalidated using only content hashes in the repo copy.
    approval = json.loads((destination / "manifests/formal_smoke_approval.json").read_text())
    plan = approval["validated_plan"]
    references = [plan["input_manifest"], *plan["results"], *plan["native_worker_preflights"]]
    inventory = {record["path"]: record for record in records}
    for reference in references:
        relative = Path(reference["path"]).resolve().relative_to(run.resolve()).as_posix()
        assert relative in inventory
        archived = destination.joinpath(*Path(relative).parts)
        assert archived.stat().st_size == reference["bytes"] == inventory[relative]["bytes"]
        assert _digest(archived) == reference["sha256"] == inventory[relative]["sha256"]

    command = json.loads((destination / "ARCHIVE_COMMAND.json").read_text())
    assert command["command"] == [
        "python",
        "archive_hayatequant_run.py",
        "--run-id",
        "formal-20260716",
    ]
    assert command["source_commit"] == SOURCE_COMMIT
    assert command["source_run_root"] == str(run.resolve())

    checksums = {}
    for line in (destination / "SHA256SUMS").read_text().splitlines():
        digest, relative_path = line.split("  ", 1)
        checksums[relative_path] = digest
        assert _digest(destination / relative_path) == digest
    assert set(checksums) == set(paths) | {"ARCHIVE_COMMAND.json"}

    detached, name = (destination / "ARCHIVE_MANIFEST.sha256").read_text().strip().split("  ")
    assert name == "ARCHIVE_MANIFEST.json"
    assert detached == _digest(destination / name)


def test_missing_optional_scopes_are_recorded_and_required_scopes_fail_closed(tmp_path):
    run = _minimal_run(tmp_path)
    destination = _archive(run, tmp_path / "archive-one")
    manifest = json.loads((destination / "ARCHIVE_MANIFEST.json").read_text())
    assert manifest["present_scopes"] == ["results/formal", "manifests", "provenance"]
    assert manifest["missing_scopes"] == [
        "results/smoke",
        "results/dev",
        "summaries",
        "logs",
    ]

    with pytest.raises(archiver.ArchiveError, match="required source scope.*missing"):
        _archive(
            run,
            tmp_path / "archive-two",
            required_scopes=("results/formal", "logs"),
        )
    assert not (tmp_path / "archive-two/formal-20260716").exists()


def test_manifests_and_one_nonempty_result_scope_are_mandatory(tmp_path):
    no_manifest = tmp_path / "no-manifest"
    _json(no_manifest / "results/formal/result.json", {"value": 1})
    with pytest.raises(archiver.ArchiveError, match="manifests"):
        _archive(no_manifest, tmp_path / "archive-one")

    no_results = tmp_path / "no-results"
    _json(no_results / "manifests/input.json", {"value": 1})
    _json(no_results / "provenance/preflight.json", {"value": 1})
    with pytest.raises(archiver.ArchiveError, match="at least one non-empty results scope"):
        _archive(no_results, tmp_path / "archive-two")

    empty_results = tmp_path / "empty-results"
    (empty_results / "results/formal").mkdir(parents=True)
    _json(empty_results / "manifests/input.json", {"value": 1})
    _json(empty_results / "provenance/preflight.json", {"value": 1})
    with pytest.raises(archiver.ArchiveError, match="at least one non-empty results scope"):
        _archive(empty_results, tmp_path / "archive-three")


@pytest.mark.parametrize(
    "bad_run_id",
    ("../escape", "a/b", "/absolute", ".hidden", "Uppercase", "a\\b", ""),
)
def test_run_id_cannot_traverse_or_escape_archive_root(tmp_path, bad_run_id):
    run = _minimal_run(tmp_path)
    with pytest.raises(archiver.ArchiveError, match="run ID"):
        _archive(run, tmp_path / "archives", run_id=bad_run_id)
    assert not (tmp_path / "escape").exists()


def test_existing_archive_is_never_overwritten(tmp_path):
    run = _minimal_run(tmp_path)
    archive_root = tmp_path / "archives"
    existing = archive_root / "formal-20260716"
    _write(existing / "sentinel", b"owner data")

    with pytest.raises(archiver.ArchiveError, match="already exists"):
        _archive(run, archive_root)
    assert (existing / "sentinel").read_bytes() == b"owner data"
    assert not (existing / "ARCHIVE_MANIFEST.json").exists()


def test_source_and_archive_roots_cannot_overlap(tmp_path):
    run = _minimal_run(tmp_path)
    nested_archive = run / "repo-archive"
    with pytest.raises(archiver.ArchiveError, match="must not overlap"):
        archiver.archive_run(
            run_root=run,
            archive_root=nested_archive,
            run_id="formal",
            source_commit=SOURCE_COMMIT,
            archiver_commit=ARCHIVER_COMMIT,
        )
    assert not nested_archive.exists()


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_symlink_anywhere_in_allowed_tree_is_rejected_without_partial_archive(tmp_path, kind):
    run = _minimal_run(tmp_path)
    external = tmp_path / "external-target"
    if kind == "file":
        _write(external, b"not allowed")
        (run / "logs").mkdir()
        (run / "logs/link.out").symlink_to(external)
    else:
        (external / "nested").mkdir(parents=True)
        (run / "logs").symlink_to(external, target_is_directory=True)

    archive_root = tmp_path / "archives"
    with pytest.raises(archiver.ArchiveError, match="symlink"):
        _archive(run, archive_root)
    assert not (archive_root / "formal-20260716").exists()


def test_symlinked_run_root_and_parent_component_are_rejected(tmp_path):
    run = _minimal_run(tmp_path / "real")
    link = tmp_path / "run-link"
    link.symlink_to(run, target_is_directory=True)
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    with pytest.raises(archiver.ArchiveError, match="symlink component"):
        archiver.archive_run(
            run_root=link,
            archive_root=archive_root,
            run_id="formal",
            source_commit=SOURCE_COMMIT,
            archiver_commit=ARCHIVER_COMMIT,
        )


@pytest.mark.parametrize(
    "secret",
    (
        b"HF_TOKEN=" + b"hf_abcdef" + b"ghijklmnopqrstuvwxyz012345\n",
        b"Authorization: " + b"Bearer abcdef" + b"ghijklmnopqrstuvwxyz0123456789\n",
        b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----\n",
        b"Ab9Qx_2Lm7.Pv4Rk8_" + b"Ns5Tw1.Yc6Hd3Fj0ZgKu8Ee\n",
    ),
)
def test_secret_like_content_is_rejected_without_echoing_secret(tmp_path, secret):
    run = _minimal_run(tmp_path)
    _write(run / "logs/job.out", secret)
    archive_root = tmp_path / "archives"
    with pytest.raises(archiver.ArchiveError, match="secret-like content") as raised:
        _archive(run, archive_root)
    assert secret.strip().decode() not in str(raised.value)
    assert not (archive_root / "formal-20260716").exists()


def test_redacted_credentials_token_metrics_and_hashes_are_not_false_positives(tmp_path):
    run = _minimal_run(tmp_path)
    _json(
        run / "logs/safe.json",
        {
            "access_token": "<redacted>",
            "input_tokens": 12345,
            "tokenizer_parallelism": False,
            "sha256": "0123456789abcdef" * 4,
        },
    )
    destination = _archive(run, tmp_path / "archives")
    assert (destination / "logs/safe.json").exists()


def test_secret_like_archive_command_is_rejected_without_echoing_secret(tmp_path):
    run = _minimal_run(tmp_path)
    secret = "Ab9Qx_2Lm7.Pv4Rk8_Ns5Tw1.Yc6Hd3Fj0ZgKu8Ee"
    archive_root = tmp_path / "archives"
    with pytest.raises(archiver.ArchiveError, match="command contains secret-like") as raised:
        _archive(
            run,
            archive_root,
            archive_command=("python", "archiver.py", "--access-token", secret),
        )
    assert secret not in str(raised.value)
    assert not (archive_root / "formal-20260716").exists()


def test_model_or_hf_cache_inside_allowlist_is_rejected_but_outside_is_ignored(tmp_path):
    run = _minimal_run(tmp_path)
    _write(run / "data/hf_home/models--safe/blobs/one", b"outside allow-list")
    destination = _archive(run, tmp_path / "archive-one")
    assert not (destination / "data").exists()

    second = _minimal_run(tmp_path / "second")
    _write(second / "results/formal/model-00001-of-00002.safetensors", b"weights")
    with pytest.raises(archiver.ArchiveError, match="model-weight artifact"):
        _archive(second, tmp_path / "archive-two")


def test_source_mutation_during_copy_removes_partial_destination(tmp_path, monkeypatch):
    run = _minimal_run(tmp_path)
    later = _json(run / "results/formal/z-later.json", {"value": 2})
    original = archiver._copy_source_file
    mutated = False

    def mutate_after_first(source, destination):
        nonlocal mutated
        original(source, destination)
        if not mutated:
            later.write_bytes(b'{"value":3}\n')
            mutated = True

    monkeypatch.setattr(archiver, "_copy_source_file", mutate_after_first)
    archive_root = tmp_path / "archives"
    with pytest.raises(archiver.ArchiveError, match="changed"):
        _archive(run, archive_root)
    assert not (archive_root / "formal-20260716").exists()


def test_payload_inventory_is_stable_across_archive_ids(tmp_path):
    run = _minimal_run(tmp_path)
    archive_root = tmp_path / "archives"
    first = _archive(run, archive_root, run_id="formal-a")
    second = _archive(run, archive_root, run_id="formal-b")
    first_manifest = json.loads((first / "ARCHIVE_MANIFEST.json").read_text())
    second_manifest = json.loads((second / "ARCHIVE_MANIFEST.json").read_text())
    assert first_manifest["files"] == second_manifest["files"]
    assert first_manifest["inventory_sha256"] == second_manifest["inventory_sha256"]


def test_cli_verifies_commit_and_uses_fixed_repo_destination(tmp_path):
    repo = tmp_path / "repo"
    experiments = repo / "papers/hayatequant_paper/experiments"
    _write(experiments / ".keep", b"")
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Archive Test"), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "archive@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    commit = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    run = _minimal_run(tmp_path / "source")
    command = (
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(repo),
        "--run-root",
        str(run.resolve()),
        "--run-id",
        "cli-formal",
        "--source-commit",
        commit,
        "--require-scope",
        "results/formal",
    )
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    destination = repo / "papers/hayatequant_paper/experiments/results/cli-formal"
    assert completed.stdout.strip() == str(destination.resolve())
    assert (destination / "results/formal/ppl/metrics.json").is_file()
    recorded = json.loads((destination / "ARCHIVE_COMMAND.json").read_text())
    assert recorded["command"][0] == str(Path(sys.executable).resolve())
    assert recorded["command"][1] == str(SCRIPT.resolve())
    assert Path(recorded["command"][0]).is_file()

    repeated = subprocess.run(command, check=False, text=True, capture_output=True)
    assert repeated.returncode == 2
    assert "overwrite is forbidden" in repeated.stderr
