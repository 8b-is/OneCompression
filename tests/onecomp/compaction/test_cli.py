"""End-to-end standalone CLI checks."""

import json
from pathlib import Path
import re
import subprocess
import sys


def _run(config: Path, *arguments: str, input_text: str | None = None):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "onecomp.compaction.cli",
            "--config",
            str(config),
            *arguments,
        ],
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )


def test_init_record_catalog_recall_export_and_doctor(tmp_path):
    config = tmp_path / "arc" / "config.json"
    initialized = _run(config, "init", "--max-recall-units", "64")
    assert initialized.returncode == 0, initialized.stderr
    assert config.stat().st_mode & 0o777 == 0o600

    recorded = _run(
        config,
        "record",
        "--command",
        "cat result.txt",
        "--content",
        "EXACT-NEEDLE\nresult",
        "--source-key",
        "call-1",
        "--tag",
        "test",
    )
    assert recorded.returncode == 0, recorded.stderr
    obs_id = json.loads(recorded.stdout)["obs_id"]

    catalog = _run(config, "catalog", "--query", "result.txt")
    assert json.loads(catalog.stdout)["observations"][0]["obs_id"] == obs_id

    metadata = _run(config, "recall", obs_id)
    assert "EXACT-NEEDLE" not in metadata.stdout
    exact = _run(config, "recall", obs_id, "--kind", "full")
    assert exact.stdout == "EXACT-NEEDLE\nresult"

    exported = _run(config, "export")
    payload = json.loads(exported.stdout)
    assert payload["bodies_included"] is False
    assert "full_content" not in payload["observations"][0]
    assert payload["observations"][0]["body_redacted"] is True
    export_path = tmp_path / "private-export.json"
    written = _run(config, "export", "--output", str(export_path))
    assert written.returncode == 0, written.stderr
    assert export_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".private-export.json.*.tmp"))

    doctor = _run(config, "doctor")
    report = json.loads(doctor.stdout)
    assert report["ok"] is True
    assert report["verified_observations"] == 1


def test_init_refuses_overwrite_and_plugin_generation(tmp_path):
    config = tmp_path / "config.json"
    assert _run(config, "init").returncode == 0
    assert _run(config, "init").returncode == 2
    plugin = tmp_path / "plugin.ts"
    generated = _run(
        config,
        "plugin",
        "--host",
        "opencode",
        "--output",
        str(plugin),
    )
    assert generated.returncode == 0, generated.stderr
    assert '"tool.execute.after"' in plugin.read_text(encoding="utf-8")
    assert (
        _run(
            config,
            "plugin",
            "--host",
            "opencode",
            "--output",
            str(plugin),
        ).returncode
        == 2
    )
    replaced = _run(
        config,
        "plugin",
        "--host",
        "opencode",
        "--output",
        str(plugin),
        "--force",
    )
    assert replaced.returncode == 0, replaced.stderr

    guide = _run(config, "guide", "--host", "codex")
    assert guide.returncode == 0
    assert "cannot replace native tool output" in guide.stdout
    assert "onecomp-arc --config" in guide.stdout


def test_init_does_not_change_existing_parent_permissions(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)

    initialized = _run(shared / "config.json", "init")

    assert initialized.returncode == 0, initialized.stderr
    assert shared.stat().st_mode & 0o777 == 0o755


def test_run_passes_short_output_and_archives_verbose_output(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert (
        _run(
            config,
            "init",
            "--result-max-chars",
            "32",
            "--max-recall-units",
            "10000",
        ).returncode
        == 0
    )

    short = _run(config, "run", "--", sys.executable, "-c", "print('short')")
    assert short.returncode == 0
    assert short.stdout == "short\n"
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 0

    verbose = _run(
        config,
        "run",
        "--tag",
        "coding-test",
        "--",
        sys.executable,
        "-c",
        "print('NEEDLE-' + 'x' * 5000)",
    )
    assert verbose.returncode == 0, verbose.stderr
    assert "<observation_ref" in verbose.stdout
    assert "NEEDLE" in verbose.stdout
    assert len(verbose.stdout) < 2000
    match = re.search(r'id="§([0-9a-f]+)"', verbose.stdout)
    assert match is not None
    obs_id = match.group(1)
    recalled = _run(config, "recall", obs_id, "--kind", "full")
    assert recalled.stdout == "NEEDLE-" + "x" * 5000 + "\n"


def test_run_preserves_failure_status_and_fails_open_on_archive_quota(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert (
        _run(
            config,
            "init",
            "--result-max-chars",
            "8",
            "--max-archive-bytes",
            "16",
        ).returncode
        == 0
    )
    command = "import sys; print('FAIL-' + 'z' * 100); raise SystemExit(7)"
    result = _run(config, "run", "--", sys.executable, "-c", command)

    assert result.returncode == 7
    assert result.stdout == "FAIL-" + "z" * 100 + "\n"
    assert "archival failed, preserving full output" in result.stderr
    assert json.loads(_run(config, "status").stdout)["archive"]["observations"] == 0


def test_config_option_is_accepted_after_subcommand_but_not_stolen_from_child(tmp_path):
    config = tmp_path / "arc" / "config.json"
    assert _run(config, "init").returncode == 0
    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "onecomp.compaction.cli",
            "status",
            "--config",
            str(config),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    child = _run(
        config,
        "run",
        "--",
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1:])",
        "--config",
        "child-value",
    )
    assert child.returncode == 0, child.stderr
    assert "child-value" in child.stdout
