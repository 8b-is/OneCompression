"""Operational CLI for formal, standalone context compaction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

from .adapters.opencode import write_plugin
from .config import CompactionConfig
from .formal import (
    FormalArcError,
    format_citation,
    format_metadata,
    migrate_experimental_wal,
)
from .runner import CompactionRunner

DEFAULT_CONFIG = Path(".onecomp/compaction/config.json")


def main(argv: Optional[list[str]] = None) -> None:
    """Run the ``onecomp-arc`` command."""
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_config_option(arguments))
    try:
        result = args.handler(args)
    except (FormalArcError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"onecomp-arc: {type(exc).__name__}: {exc}\n")
    if result is not None:
        _print_json(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onecomp-arc",
        description="Formal addressable-recall context compaction",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="configuration path (default: .onecomp/compaction/config.json)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a private persistent archive")
    init.add_argument("--max-archive-bytes", type=int, default=0)
    init.add_argument("--max-recall-units", type=int, default=4096)
    init.add_argument("--summary-max-chars", type=int, default=24_000)
    init.add_argument("--result-max-chars", type=int, default=500)
    init.set_defaults(handler=_init)

    run = subparsers.add_parser(
        "run",
        help="run a command and replace verbose output with an exact-recall citation",
    )
    run.add_argument(
        "--threshold",
        type=int,
        help="archive output above this character count (default: configured result_max_chars)",
    )
    run.add_argument("--timeout", type=float, help="kill the command after this many seconds")
    run.add_argument(
        "--max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="fail open above this captured size (default: 64 MiB)",
    )
    run.add_argument("--tag", action="append", default=[])
    run.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- COMMAND [ARG ...]")
    run.set_defaults(handler=_run)

    guide = subparsers.add_parser(
        "guide",
        help="print a concise integration guide for a coding agent",
    )
    guide.add_argument("--host", choices=("codex", "generic"), default="generic")
    guide.set_defaults(handler=_guide)

    doctor = subparsers.add_parser("doctor", help="verify config, permissions, and checksums")
    doctor.set_defaults(handler=_doctor)

    status = subparsers.add_parser("status", help="show body-free archive accounting")
    status.set_defaults(handler=_status)

    record = subparsers.add_parser("record", help="commit one exact observation occurrence")
    record.add_argument("--command", required=True)
    content = record.add_mutually_exclusive_group(required=True)
    content.add_argument("--content")
    content.add_argument("--content-file", type=Path)
    record.add_argument("--source-key")
    record.add_argument("--returncode", type=int, default=0)
    record.add_argument("--tag", action="append", default=[])
    record.set_defaults(handler=_record)

    catalog = subparsers.add_parser("catalog", help="page through body-free metadata")
    catalog.add_argument("--cursor", type=int)
    catalog.add_argument("--limit", type=int, default=20)
    catalog.add_argument("--query", default="")
    catalog.add_argument("--tag")
    catalog.set_defaults(handler=_catalog)

    recall = subparsers.add_parser("recall", help="perform explicit bounded exact recovery")
    recall.add_argument("obs_id")
    recall.add_argument("--kind", choices=("full", "meta", "chunk", "range"), default="meta")
    recall.add_argument("--chunk-index", type=int)
    recall.add_argument("--range", dest="char_range")
    recall.set_defaults(handler=_recall)

    export = subparsers.add_parser("export", help="export JSON; bodies are redacted by default")
    export.add_argument("--output", type=Path)
    export.add_argument("--include-bodies", action="store_true")
    export.set_defaults(handler=_export)

    migrate = subparsers.add_parser("migrate", help="import an experimental JSONL WAL")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("--dry-run", action="store_true")
    migrate.add_argument("--lenient", action="store_true")
    migrate.set_defaults(handler=_migrate)

    plugin = subparsers.add_parser("plugin", help="generate an agent-host integration file")
    plugin.add_argument("--host", choices=("opencode",), required=True)
    plugin.add_argument("--output", type=Path, required=True)
    plugin.add_argument("--python", default=sys.executable)
    plugin.add_argument("--result-max-chars", type=int, default=500)
    plugin.add_argument("--max-archive-bytes", type=int, default=0)
    plugin.add_argument("--force", action="store_true", help="replace an existing output file")
    plugin.set_defaults(handler=_plugin)
    return parser


def _init(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    if config_path.exists():
        raise ValueError(f"configuration already exists: {config_path}")
    root = config_path.parent
    root_existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if not root_existed:
        _chmod(root, 0o700)
    archive_path = root / "archive.sqlite3"
    if archive_path.exists():
        raise ValueError(f"archive already exists without configuration: {archive_path}")
    config = CompactionConfig(
        profile="formal-v1",
        archive_path=archive_path,
        max_archive_bytes=args.max_archive_bytes,
        max_recall_units=args.max_recall_units,
        summary_max_chars=args.summary_max_chars,
        result_max_chars=args.result_max_chars,
    )
    with CompactionRunner(config):
        pass
    _atomic_write_text(
        config_path,
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=False,
    )
    return {
        "created": True,
        "config": str(config_path),
        "archive": str(archive_path),
        "next": (
            f"onecomp-arc --config {shlex.quote(str(config_path))} " "run -- COMMAND [ARG ...]"
        ),
    }


def _run(args: argparse.Namespace) -> None:
    """Execute an argv-safe command and fail open unless archival commits."""
    config = _load_config(args.config)
    command = list(args.argv)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("run requires COMMAND [ARG ...] after --")
    if args.threshold is not None and (not isinstance(args.threshold, int) or args.threshold < 0):
        raise ValueError("--threshold must be a non-negative integer")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.max_output_bytes <= 0:
        raise ValueError("--max-output-bytes must be a positive integer")

    timed_out = False
    with tempfile.TemporaryFile(mode="w+b") as capture:
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                stdout=capture,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.timeout,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
        capture.flush()
        size = capture.tell()
        capture.seek(0)
        raw = capture.read()

    if timed_out:
        print(f"onecomp-arc: command timed out after {args.timeout:g}s", file=sys.stderr)
    if size > args.max_output_bytes:
        print(
            "onecomp-arc: output exceeded --max-output-bytes; preserving full output",
            file=sys.stderr,
        )
        _write_raw(raw)
        _finish_command(returncode)
        return
    try:
        content = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        print(
            "onecomp-arc: output is not UTF-8; preserving original bytes without archival",
            file=sys.stderr,
        )
        _write_raw(raw)
        _finish_command(returncode)
        return
    threshold = config.result_max_chars if args.threshold is None else args.threshold
    if len(content) <= threshold:
        _write_raw(raw)
        _finish_command(returncode)
        return

    try:
        with CompactionRunner(config) as runner:
            record = runner.record_observation(
                command=shlex.join(command),
                content=content,
                source_key=f"standalone:{os.getpid()}:{time.time_ns()}",
                returncode=returncode,
                tags=["standalone-run", *args.tag],
                extra={"argv": command, "timed_out": timed_out},
            )
    except (FormalArcError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"onecomp-arc: archival failed, preserving full output: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        _write_raw(raw)
        _finish_command(returncode)
        return

    config_path = args.config.expanduser().resolve()
    citation = format_citation(record)
    recall_prefix = (
        f"onecomp-arc --config {shlex.quote(str(config_path))} " f"recall {record.obs_id}"
    )
    print(
        f"{citation}\n<standalone_recovery>Exact output: `{recall_prefix} --kind full`; "
        f"bounded recovery: `{recall_prefix} --kind chunk --chunk-index 0` or "
        f"`{recall_prefix} --kind range --range START:END`.</standalone_recovery>"
    )
    _finish_command(returncode)


def _guide(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    executable = f"onecomp-arc --config {shlex.quote(str(config_path))} run -- COMMAND [ARG ...]"
    host_note = (
        "Codex PostToolUse hooks cannot replace native tool output, so use this wrapper "
        "before verbose output enters the transcript."
        if args.host == "codex"
        else "Use this wrapper at the command-execution boundary."
    )
    sys.stdout.write(
        "# OneComp context compaction\n\n"
        f"{host_note}\n\n"
        "- Run commands expected to produce large output with:\n"
        f"  `{executable}`\n"
        "- Run interactive commands and commands requiring a TTY directly.\n"
        "- A short result passes through unchanged. A long result becomes a bounded "
        "citation only after the exact UTF-8 output commits to SQLite.\n"
        "- Follow the citation's `recall --kind chunk` or `--kind range` command when "
        "exact middle content is needed. Never treat citation previews as authoritative.\n"
        "- Preserve the command's exit status; a citation with a nonzero return code is "
        "still a failed command.\n"
    )


def _doctor(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config)
    with CompactionRunner(config) as runner:
        assert runner.archive is not None
        ids = runner.archive.known_ids()
        for obs_id in ids:
            runner.archive.get(obs_id)
        private_config = _permission_bits(args.config) in (None, 0o600)
        private_archive = _permission_bits(Path(runner.archive.path)) in (None, 0o600)
        return {
            "ok": private_config and private_archive,
            "profile": runner.profile,
            "verified_observations": len(ids),
            "config_private": private_config,
            "archive_private": private_archive,
            "status": runner.status(),
        }


def _status(args: argparse.Namespace) -> dict[str, Any]:
    with CompactionRunner(_load_config(args.config)) as runner:
        return runner.status()


def _record(args: argparse.Namespace) -> dict[str, Any]:
    if args.content_file is None:
        body = args.content
    elif str(args.content_file) == "-":
        body = sys.stdin.read()
    else:
        body = args.content_file.expanduser().read_text(encoding="utf-8")
    with CompactionRunner(_load_config(args.config)) as runner:
        record = runner.record_observation(
            command=args.command,
            content=body,
            source_key=args.source_key,
            returncode=args.returncode,
            tags=args.tag,
        )
        return {
            "obs_id": record.obs_id,
            "sequence": record.sequence,
            "char_count": record.char_count,
            "byte_count": record.byte_count,
            "content_sha256": record.content_sha256,
        }


def _catalog(args: argparse.Namespace) -> dict[str, Any]:
    with CompactionRunner(_load_config(args.config)) as runner:
        assert runner.service is not None
        page = runner.service.catalog(
            after_sequence=args.cursor,
            limit=args.limit,
            query=args.query,
            tag=args.tag,
        )
        return {
            "observations": [asdict(item) for item in page.items],
            "next_cursor": page.next_cursor,
            "has_more": page.has_more,
        }


def _recall(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    with CompactionRunner(_load_config(args.config)) as runner:
        assert runner.service is not None
        service = runner.service
        if args.kind == "meta":
            return {"kind": "meta", "content": format_metadata(service.metadata(args.obs_id))}
        if args.kind == "full":
            body = service.recall_full(args.obs_id)
        elif args.kind == "chunk":
            if args.chunk_index is None:
                raise ValueError("--kind chunk requires --chunk-index")
            body = service.recall_text_chunk(args.obs_id, args.chunk_index).text
        else:
            if args.char_range is None:
                raise ValueError("--kind range requires --range START:END")
            start, end = _parse_range(args.char_range)
            body = service.recall_range(args.obs_id, start, end)
    # Exact bodies go to stdout without JSON escaping so shell redirection is
    # lossless. Metadata remains structured and body-free.
    sys.stdout.write(body)
    return None


def _export(args: argparse.Namespace) -> dict[str, Any]:
    with CompactionRunner(_load_config(args.config)) as runner:
        assert runner.archive is not None
        rows = runner.archive.to_serializable(include_bodies=args.include_bodies)
    payload = {
        "schema": "onecomp-compaction-export-v1",
        "bodies_included": args.include_bodies,
        "observations": rows,
    }
    if args.output is None:
        return payload
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=True,
    )
    return {"output": str(destination), "observations": len(rows)}


def _migrate(args: argparse.Namespace) -> dict[str, Any]:
    with CompactionRunner(_load_config(args.config)) as runner:
        assert runner.archive is not None
        report = migrate_experimental_wal(
            args.source,
            runner.archive,
            strict=not args.lenient,
            dry_run=args.dry_run,
        )
        return asdict(report)


def _plugin(args: argparse.Namespace) -> dict[str, Any]:
    path = write_plugin(
        args.output,
        python_executable=args.python,
        result_max_chars=args.result_max_chars,
        max_archive_bytes=args.max_archive_bytes,
        overwrite=args.force,
    )
    return {"host": args.host, "output": str(path)}


def _load_config(path: Path) -> CompactionConfig:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(
            f"configuration does not exist: {config_path}; run "
            f"`onecomp-arc --config {shlex.quote(str(config_path))} init` first"
        )
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("compaction config must be a JSON object")
    if value.get("profile") != "formal-v1":
        raise ValueError("operational CLI requires profile='formal-v1'")
    archive_path = value.get("archive_path")
    if isinstance(archive_path, str) and not Path(archive_path).is_absolute():
        value["archive_path"] = config_path.parent / archive_path
    return CompactionConfig(**value)


def _parse_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError("range must be START:END") from exc
    if start < 0 or end < start:
        raise ValueError("range must satisfy 0 <= START <= END")
    return start, end


def _permission_bits(path: Path) -> Optional[int]:
    try:
        return path.expanduser().stat().st_mode & 0o777
    except OSError:
        return None


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _atomic_write_text(destination: Path, payload: str, *, overwrite: bool) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ValueError(f"configuration already exists: {destination}") from exc
            temporary.unlink()
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = -1
        if directory_descriptor >= 0:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _normalize_config_option(arguments: list[str]) -> list[str]:
    """Accept ``--config`` before or after a subcommand without ambiguity."""
    normalized = list(arguments)
    for index, value in enumerate(normalized):
        if value == "--":
            break
        if value == "--config":
            if index + 1 >= len(normalized):
                return normalized
            pair = normalized[index : index + 2]
            return pair + normalized[:index] + normalized[index + 2 :]
        if value.startswith("--config="):
            return [value] + normalized[:index] + normalized[index + 1 :]
    return normalized


def _write_raw(value: bytes) -> None:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(value.decode("utf-8", errors="replace"))
    else:
        stream.write(value)
        stream.flush()


def _finish_command(returncode: int) -> None:
    if returncode == 0:
        return
    status = 128 + abs(returncode) if returncode < 0 else returncode
    raise SystemExit(min(status, 255))


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]
