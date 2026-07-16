#!/usr/bin/env python3
"""Copy one external HayateQuant run into a content-verified repo archive.

The paper runners intentionally write large, mutable working state outside the
repository.  This command copies only the scientific result allow-list into
``papers/hayatequant_paper/experiments/results/<run-id>`` and emits a complete
byte inventory.  Model weights, Hugging Face caches, temporary files, and other
runtime state are outside the allow-list.

The implementation is deliberately fail-closed.  It rejects symlinks, special
files, unstable source trees, suspicious credentials, model/cache artifacts in
an allowed tree, unsafe run IDs, and an already existing destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_RELATIVE_ROOT = Path("papers/hayatequant_paper/experiments/results")
PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())
ALLOWED_SCOPES = (
    "results/smoke",
    "results/formal",
    "results/dev",
    "summaries",
    "manifests",
    "provenance",
    "logs",
)
RESULT_SCOPES = frozenset({"results/smoke", "results/formal", "results/dev"})
MANDATORY_SCOPES = frozenset({"manifests", "provenance"})
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CHUNK_SIZE = 1024 * 1024
SECRET_SCAN_OVERLAP = 2048

SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".netrc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
MODEL_CACHE_COMPONENTS = frozenset(
    {
        ".cache",
        "hf_home",
        "huggingface_cache",
        "model_cache",
        "transformers_cache",
    }
)
MODEL_WEIGHT_SUFFIXES = frozenset({".safetensors", ".gguf", ".ckpt", ".onnx"})

KNOWN_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")),
    ("GitLab token", re.compile(rb"glpat-[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(rb"hf_[A-Za-z0-9]{20,}")),
    ("OpenAI-style token", re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}")),
    ("Slack token", re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("bearer token", re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}")),
    (
        "JWT",
        re.compile(rb"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
)
SECRET_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?:^|[^a-z0-9_])"
    rb"(?:access[_-]?token|api[_-]?key|authorization|client[_-]?secret|"
    rb"github[_-]?token|gitlab[_-]?token|hf[_-]?token|password|passwd|private[_-]?key)"
    rb"\s*(?:=|:)\s*[\"']?([^\s,;\"']{8,})"
)
GENERIC_TOKEN_RE = re.compile(rb"[A-Za-z0-9][A-Za-z0-9._-]{39,}")
PLACEHOLDER_VALUES = frozenset(
    {
        "<omitted>",
        "<redacted>",
        "not_recorded",
        "not_set",
        "redacted",
        "removed",
        "unset",
    }
)
CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "githubtoken",
        "gitlabtoken",
        "hftoken",
        "password",
        "passwd",
        "privatekey",
    }
)
MODEL_GENERATION_TEXT_KEYS = frozenset(
    {
        "generated_text",
        "prediction",
        "raw_generated_text",
        "raw_prediction",
    }
)
STRUCTURED_PLATFORM_RE = re.compile(
    rb"^Linux-[0-9][A-Za-z0-9.+~_-]{0,127}"
    rb"-(?:x86_64|aarch64|ppc64le|s390x|riscv64)"
    rb"-with-(?:glibc|musl)[0-9][0-9.]{0,31}$"
)
STRUCTURED_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})


class ArchiveError(RuntimeError):
    """Raised when an archive cannot be produced without ambiguity or risk."""


class _DuplicateJsonKeyError(ValueError):
    """Raised internally when structured JSON contains an ambiguous object."""


StateTuple = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class DirectorySnapshot:
    path: Path
    relative_path: str | None
    state: StateTuple


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    scope: str
    state: StateTuple
    archive_mode: int
    byte_count: int
    sha256: str

    def manifest_record(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "mode": f"{self.archive_mode:04o}",
            "scope": self.scope,
        }


@dataclass(frozen=True)
class SourceInventory:
    files: tuple[SourceFile, ...]
    directories: tuple[DirectorySnapshot, ...]
    guard_directories: tuple[DirectorySnapshot, ...]
    present_scopes: tuple[str, ...]
    missing_scopes: tuple[str, ...]
    scope_file_counts: dict[str, int]


def _state_tuple(value: os.stat_result) -> StateTuple:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ArchiveError(
            "run ID must be 1-80 lowercase ASCII characters, begin with an "
            "alphanumeric character, and contain only [a-z0-9._-]"
        )
    if run_id in {".", ".."}:
        raise ArchiveError("run ID cannot be '.' or '..'")
    return run_id


def _validate_commit(value: str, *, label: str) -> str:
    normalized = value.strip().lower()
    if not COMMIT_RE.fullmatch(normalized):
        raise ArchiveError(f"{label} must be a full 40- or 64-character hexadecimal commit ID")
    return normalized


def _validate_component(name: str, *, label: str) -> None:
    if name in {"", ".", ".."}:
        raise ArchiveError(f"{label} has an unsafe path component {name!r}")
    if name != name.strip():
        raise ArchiveError(f"{label} has leading or trailing whitespace in component {name!r}")
    if "/" in name or "\\" in name:
        raise ArchiveError(f"{label} contains a path separator in component {name!r}")
    if unicodedata.normalize("NFC", name) != name:
        raise ArchiveError(f"{label} contains a non-NFC path component {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ArchiveError(f"{label} contains a control character")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in name):
        raise ArchiveError(f"{label} contains an undecodable filesystem character")


def _absolute_lexical_path(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ArchiveError(f"{label} must be an absolute path: {path}")
    if ".." in expanded.parts:
        raise ArchiveError(f"{label} must not contain '..': {path}")
    return expanded


def _reject_symlink_components(path: Path, *, label: str, must_exist: bool) -> None:
    absolute = _absolute_lexical_path(path, label=label)
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        _validate_component(component, label=label)
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist or index != len(absolute.parts[1:]) - 1:
                raise ArchiveError(f"{label} does not exist: {current}")
            return
        except OSError as exc:
            raise ArchiveError(f"cannot inspect {label}: {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArchiveError(f"{label} contains a symlink component: {current}")


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    lexical = _absolute_lexical_path(path, label=label)
    _reject_symlink_components(lexical, label=label, must_exist=True)
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ArchiveError(f"cannot resolve {label}: {lexical}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveError(f"{label} is not a directory: {lexical}")
    return resolved


def _shannon_entropy(value: bytes) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _generic_token_looks_secret(value: bytes) -> bool:
    if all(character in b"0123456789abcdefABCDEF" for character in value):
        return False
    categories = (
        any(65 <= character <= 90 for character in value),
        any(97 <= character <= 122 for character in value),
        any(48 <= character <= 57 for character in value),
        any(character in b"._-" for character in value),
    )
    return all(categories) and _shannon_entropy(value) >= 4.35


def _explicit_secret_reason(payload: bytes) -> str | None:
    for label, pattern in KNOWN_SECRET_PATTERNS:
        if pattern.search(payload):
            return label
    for match in SECRET_ASSIGNMENT_RE.finditer(payload):
        candidate = match.group(1).decode("ascii", errors="ignore").strip().lower()
        if candidate not in PLACEHOLDER_VALUES and not candidate.startswith(("${", "$")):
            return "credential assignment"
    return None


def _generic_secret_reason(payload: bytes) -> str | None:
    for match in GENERIC_TOKEN_RE.finditer(payload):
        candidate = match.group(0)
        if STRUCTURED_PLATFORM_RE.fullmatch(candidate):
            continue
        if _generic_token_looks_secret(candidate):
            return "high-entropy token"
    return None


def _secret_reason(payload: bytes) -> str | None:
    return _explicit_secret_reason(payload) or _generic_secret_reason(payload)


def _json_object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKeyError
        document[key] = value
    return document


def _structured_json_secret_reason(document: object) -> str | None:
    """Scan decoded JSON while treating only model-output text as untrusted prose.

    Known token prefixes, private-key markers, Bearer tokens, and credential
    assignments remain forbidden in every string.  The generic entropy
    heuristic is omitted only for the four lossless model-output fields.  All
    keys and all other metadata remain subject to the generic heuristic, apart
    from a narrowly validated Linux ``platform.platform()`` identifier.
    """

    pending: list[tuple[object, str | None]] = [(document, None)]
    while pending:
        value, field_name = pending.pop()
        normalized_field_name = (
            re.sub(r"[^a-z0-9]", "", field_name.lower()) if isinstance(field_name, str) else None
        )
        if normalized_field_name in CREDENTIAL_FIELD_NAMES:
            if not isinstance(value, str):
                return "credential field"
            candidate = value.strip().lower()
            if candidate not in PLACEHOLDER_VALUES and not candidate.startswith(("${", "$")):
                return "credential field"
        if isinstance(value, dict):
            for key, child in value.items():
                key_payload = key.encode("utf-8")
                reason = _secret_reason(key_payload)
                if reason is not None:
                    return reason
                pending.append((child, key))
            continue
        if isinstance(value, list):
            # A model-output exemption applies only to the direct string value
            # of the named field.  Do not let a misleading list/object under a
            # generation key turn arbitrary nested metadata into exempt prose.
            pending.extend((child, None) for child in value)
            continue
        if not isinstance(value, str):
            continue

        payload = value.encode("utf-8")
        reason = _explicit_secret_reason(payload)
        if reason is not None:
            return reason
        if field_name in MODEL_GENERATION_TEXT_KEYS:
            continue
        reason = _generic_secret_reason(payload)
        if reason is not None:
            return reason
    return None


class _SecretScanner:
    def __init__(self, *, relative_path: str) -> None:
        self.relative_path = relative_path
        self._tail = b""
        self._structured_json = (
            PurePosixPath(relative_path).suffix.lower() in STRUCTURED_JSON_SUFFIXES
        )
        self._json_payload = bytearray()

    def feed(self, chunk: bytes) -> None:
        window = self._tail + chunk
        reason = (
            _explicit_secret_reason(window) if self._structured_json else _secret_reason(window)
        )
        if reason is not None:
            raise ArchiveError(
                f"refusing to archive secret-like content ({reason}) in {self.relative_path}"
            )
        if self._structured_json:
            self._json_payload.extend(chunk)
        self._tail = window[-SECRET_SCAN_OVERLAP:]

    def finish(self) -> None:
        if not self._structured_json:
            return
        try:
            if PurePosixPath(self.relative_path).suffix.lower() == ".json":
                documents = (
                    json.loads(
                        self._json_payload,
                        object_pairs_hook=_json_object_without_duplicates,
                    ),
                )
            else:
                documents = tuple(
                    json.loads(line, object_pairs_hook=_json_object_without_duplicates)
                    for line in self._json_payload.splitlines()
                    if line.strip()
                )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            _DuplicateJsonKeyError,
        ) as exc:
            raise ArchiveError(
                f"refusing malformed structured JSON content in {self.relative_path}"
            ) from exc
        for document in documents:
            reason = _structured_json_secret_reason(document)
            if reason is not None:
                raise ArchiveError(
                    f"refusing to archive secret-like content ({reason}) in {self.relative_path}"
                )


def _path_policy_reason(relative_path: str, *, is_directory: bool) -> str | None:
    path = PurePosixPath(relative_path)
    lower_parts = tuple(part.lower() for part in path.parts)
    name = lower_parts[-1]
    if name in SENSITIVE_FILENAMES:
        return "credential-like filename"
    if any(
        part in MODEL_CACHE_COMPONENTS or part.startswith(("models--", "datasets--"))
        for part in lower_parts
    ):
        return "model/Hugging Face cache path"
    if is_directory:
        return None
    suffix = PurePosixPath(name).suffix
    if suffix in MODEL_WEIGHT_SUFFIXES:
        return "model-weight artifact"
    if suffix == ".bin" and (name.startswith("pytorch_model") or name.startswith("model")):
        return "model-weight artifact"
    return None


def _read_stable_file(
    path: Path,
    *,
    relative_path: str,
    expected_state: StateTuple | None = None,
    destination_fd: int | None = None,
) -> tuple[StateTuple, int, str]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ArchiveError(f"cannot inspect source file {relative_path}: {exc}") from exc
    before_state = _state_tuple(before)
    if expected_state is not None and before_state != expected_state:
        raise ArchiveError(f"source file changed before copy: {relative_path}")
    if not stat.S_ISREG(before.st_mode):
        raise ArchiveError(f"source entry is not a regular file: {relative_path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"cannot open source file safely: {relative_path}: {exc}") from exc
    digest = hashlib.sha256()
    scanner = _SecretScanner(relative_path=relative_path)
    byte_count = 0
    try:
        opened_state = _state_tuple(os.fstat(source_fd))
        if opened_state != before_state:
            raise ArchiveError(f"source file was replaced while opening: {relative_path}")
        while True:
            chunk = os.read(source_fd, CHUNK_SIZE)
            if not chunk:
                break
            scanner.feed(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
            if destination_fd is not None:
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise ArchiveError(f"short write while copying {relative_path}")
                    view = view[written:]
        scanner.finish()
        after_open_state = _state_tuple(os.fstat(source_fd))
    finally:
        os.close(source_fd)

    try:
        after_path_state = _state_tuple(path.lstat())
    except OSError as exc:
        raise ArchiveError(f"source file disappeared while reading: {relative_path}") from exc
    if after_open_state != before_state or after_path_state != before_state:
        raise ArchiveError(f"source file changed while reading: {relative_path}")
    if byte_count != before.st_size:
        raise ArchiveError(f"source file size changed while reading: {relative_path}")
    return before_state, byte_count, digest.hexdigest()


def _snapshot_directory(path: Path, *, relative_path: str | None) -> DirectorySnapshot:
    try:
        metadata = path.lstat()
    except OSError as exc:
        label = relative_path or str(path)
        raise ArchiveError(f"cannot inspect source directory {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ArchiveError(f"symlinked source directory is forbidden: {relative_path or path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveError(f"source scope is not a directory: {relative_path or path}")
    return DirectorySnapshot(path=path, relative_path=relative_path, state=_state_tuple(metadata))


def _walk_scope(
    scope_path: Path, *, scope: str
) -> tuple[list[SourceFile], list[DirectorySnapshot]]:
    files: list[SourceFile] = []
    directories: list[DirectorySnapshot] = []

    def visit(directory: Path, relative_directory: str) -> None:
        reason = _path_policy_reason(relative_directory, is_directory=True)
        if reason is not None:
            raise ArchiveError(f"refusing {reason} in allowed scope: {relative_directory}")
        directories.append(_snapshot_directory(directory, relative_path=relative_directory))
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
        except OSError as exc:
            raise ArchiveError(
                f"cannot scan source directory {relative_directory}: {exc}"
            ) from exc
        for entry in entries:
            _validate_component(entry.name, label=f"source path below {relative_directory}")
            relative_path = f"{relative_directory}/{entry.name}"
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArchiveError(f"cannot inspect source entry {relative_path}: {exc}") from exc
            if stat.S_ISLNK(metadata.st_mode) or entry.is_symlink():
                raise ArchiveError(f"symlinked source entry is forbidden: {relative_path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(entry.path), relative_path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ArchiveError(f"special source file is forbidden: {relative_path}")
            reason = _path_policy_reason(relative_path, is_directory=False)
            if reason is not None:
                raise ArchiveError(f"refusing {reason} in allowed scope: {relative_path}")
            path_secret = _secret_reason(relative_path.encode("utf-8"))
            if path_secret is not None:
                raise ArchiveError(
                    f"refusing secret-like source pathname ({path_secret}): {relative_path}"
                )
            state, byte_count, digest = _read_stable_file(
                Path(entry.path),
                relative_path=relative_path,
                expected_state=_state_tuple(metadata),
            )
            archive_mode = 0o755 if metadata.st_mode & 0o111 else 0o644
            files.append(
                SourceFile(
                    path=Path(entry.path),
                    relative_path=relative_path,
                    scope=scope,
                    state=state,
                    archive_mode=archive_mode,
                    byte_count=byte_count,
                    sha256=digest,
                )
            )

    visit(scope_path, scope)
    return files, directories


def _verify_directory_snapshots(snapshots: Iterable[DirectorySnapshot]) -> None:
    for snapshot in snapshots:
        label = snapshot.relative_path or str(snapshot.path)
        try:
            current = snapshot.path.lstat()
        except OSError as exc:
            raise ArchiveError(f"source directory disappeared during archive: {label}") from exc
        if _state_tuple(current) != snapshot.state:
            raise ArchiveError(f"source directory changed during archive: {label}")


def _verify_file_snapshots(files: Iterable[SourceFile]) -> None:
    for source_file in files:
        try:
            current = source_file.path.lstat()
        except OSError as exc:
            raise ArchiveError(
                f"source file disappeared during archive: {source_file.relative_path}"
            ) from exc
        if _state_tuple(current) != source_file.state:
            raise ArchiveError(f"source file changed during archive: {source_file.relative_path}")


def _normalize_required_scopes(required_scopes: Iterable[str]) -> tuple[str, ...]:
    allowed = set(ALLOWED_SCOPES)
    requested = set(required_scopes) | set(MANDATORY_SCOPES)
    unknown = sorted(requested - allowed)
    if unknown:
        raise ArchiveError(f"unknown required archive scope(s): {', '.join(unknown)}")
    return tuple(scope for scope in ALLOWED_SCOPES if scope in requested)


def build_inventory(run_root: Path, *, required_scopes: Iterable[str] = ()) -> SourceInventory:
    """Hash and security-check every file in the allowed source scopes."""

    source = _canonical_existing_directory(run_root, label="RUN_ROOT")
    required = _normalize_required_scopes(required_scopes)
    root_secret = _secret_reason(str(source).encode("utf-8"))
    if root_secret is not None:
        raise ArchiveError(f"RUN_ROOT path contains secret-like text ({root_secret})")

    guard_directories = [_snapshot_directory(source, relative_path=None)]
    results_parent = source / "results"
    if results_parent.exists() or results_parent.is_symlink():
        guard_directories.append(_snapshot_directory(results_parent, relative_path=None))

    files: list[SourceFile] = []
    directories: list[DirectorySnapshot] = []
    present: list[str] = []
    missing: list[str] = []
    counts: dict[str, int] = {}
    for scope in ALLOWED_SCOPES:
        scope_path = source.joinpath(*PurePosixPath(scope).parts)
        try:
            metadata = scope_path.lstat()
        except FileNotFoundError:
            missing.append(scope)
            counts[scope] = 0
            continue
        except OSError as exc:
            raise ArchiveError(f"cannot inspect source scope {scope}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ArchiveError(f"symlinked source scope is forbidden: {scope}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArchiveError(f"source scope is not a directory: {scope}")
        scope_files, scope_directories = _walk_scope(scope_path, scope=scope)
        files.extend(scope_files)
        directories.extend(scope_directories)
        present.append(scope)
        counts[scope] = len(scope_files)

    missing_required = [scope for scope in required if scope not in present]
    if missing_required:
        raise ArchiveError(f"required source scope(s) are missing: {', '.join(missing_required)}")
    empty_required = [scope for scope in required if counts.get(scope, 0) == 0]
    if empty_required:
        raise ArchiveError(
            f"required source scope(s) contain no files: {', '.join(empty_required)}"
        )
    if not any(counts.get(scope, 0) > 0 for scope in RESULT_SCOPES):
        raise ArchiveError("RUN_ROOT must contain at least one non-empty results scope")

    files.sort(key=lambda item: item.relative_path)
    directories.sort(key=lambda item: item.relative_path or "")
    relative_paths = [item.relative_path for item in files]
    if len(relative_paths) != len(set(relative_paths)):
        raise ArchiveError("source inventory contains duplicate relative paths")
    folded: dict[str, str] = {}
    for relative_path in relative_paths:
        previous = folded.setdefault(relative_path.casefold(), relative_path)
        if previous != relative_path:
            raise ArchiveError(
                "source paths differ only by case and are not portable: "
                f"{previous!r} and {relative_path!r}"
            )

    _verify_file_snapshots(files)
    _verify_directory_snapshots((*guard_directories, *directories))
    return SourceInventory(
        files=tuple(files),
        directories=tuple(directories),
        guard_directories=tuple(guard_directories),
        present_scopes=tuple(present),
        missing_scopes=tuple(missing),
        scope_file_counts=counts,
    )


def _safe_destination(destination: Path, *, archive_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    for component in pure.parts:
        _validate_component(component, label=f"archive path {relative_path}")
    candidate = destination.joinpath(*pure.parts)
    canonical_parent = candidate.parent.resolve(strict=False)
    if not _is_relative_to(canonical_parent, destination.resolve(strict=True)):
        raise ArchiveError(f"archive path escapes destination: {relative_path}")
    if not _is_relative_to(candidate.resolve(strict=False), archive_root.resolve(strict=True)):
        raise ArchiveError(f"archive path escapes archive root: {relative_path}")
    return candidate


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise ArchiveError(f"cannot create archive file without overwrite: {path}: {exc}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ArchiveError(f"short write while creating {path}")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_source_file(source: SourceFile, destination: Path) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(destination, flags, source.archive_mode)
    except OSError as exc:
        raise ArchiveError(
            f"cannot create archive file without overwrite: {source.relative_path}: {exc}"
        ) from exc
    try:
        state, byte_count, digest = _read_stable_file(
            source.path,
            relative_path=source.relative_path,
            expected_state=source.state,
            destination_fd=descriptor,
        )
        os.fchmod(descriptor, source.archive_mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if state != source.state or byte_count != source.byte_count or digest != source.sha256:
        raise ArchiveError(
            f"source file changed between inventory and copy: {source.relative_path}"
        )
    _, archived_size, archived_digest = _read_stable_file(
        destination,
        relative_path=f"archived/{source.relative_path}",
    )
    if archived_size != source.byte_count or archived_digest != source.sha256:
        raise ArchiveError(f"archived bytes failed verification: {source.relative_path}")


def _prepare_archive_root(archive_root: Path) -> Path:
    lexical = _absolute_lexical_path(archive_root, label="archive root")
    if lexical.exists() or lexical.is_symlink():
        return _canonical_existing_directory(lexical, label="archive root")
    parent = _canonical_existing_directory(lexical.parent, label="archive root parent")
    candidate = parent / lexical.name
    _validate_component(candidate.name, label="archive root")
    try:
        candidate.mkdir(mode=0o755)
    except OSError as exc:
        raise ArchiveError(f"cannot create archive root {candidate}: {exc}") from exc
    return candidate.resolve(strict=True)


def _generated_record(path: Path, *, relative_path: str) -> dict[str, object]:
    metadata = path.stat()
    return {
        "path": relative_path,
        "bytes": metadata.st_size,
        "sha256": _sha256_file(path),
        "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
    }


def archive_run(
    *,
    run_root: Path,
    archive_root: Path,
    run_id: str,
    source_commit: str,
    archiver_commit: str,
    required_scopes: Iterable[str] = (),
    archive_command: Sequence[str] | None = None,
) -> Path:
    """Create one immutable-by-convention run archive and return its path."""

    safe_run_id = _validate_run_id(run_id)
    safe_source_commit = _validate_commit(source_commit, label="source commit")
    safe_archiver_commit = _validate_commit(archiver_commit, label="archiver commit")
    required = _normalize_required_scopes(required_scopes)
    source = _canonical_existing_directory(run_root, label="RUN_ROOT")
    archive_lexical = _absolute_lexical_path(archive_root, label="archive root")
    if archive_lexical.exists() or archive_lexical.is_symlink():
        archive_candidate = _canonical_existing_directory(archive_lexical, label="archive root")
    else:
        archive_parent = _canonical_existing_directory(
            archive_lexical.parent, label="archive root parent"
        )
        _validate_component(archive_lexical.name, label="archive root")
        archive_candidate = archive_parent / archive_lexical.name
    if _is_relative_to(source, archive_candidate) or _is_relative_to(archive_candidate, source):
        raise ArchiveError("RUN_ROOT and repository archive root must not overlap")
    root = _prepare_archive_root(archive_root)

    destination = root / safe_run_id
    if destination.exists() or destination.is_symlink():
        raise ArchiveError(
            f"archive destination already exists; overwrite is forbidden: {destination}"
        )

    command = list(archive_command or ())
    if not command:
        command = [
            PYTHON_EXECUTABLE,
            str(Path(__file__).resolve()),
            "--run-root",
            str(source),
            "--run-id",
            safe_run_id,
            "--source-commit",
            safe_source_commit,
        ]
        for scope in required:
            command.extend(("--require-scope", scope))
    command_secret = _secret_reason("\0".join(command).encode("utf-8"))
    if command_secret is not None:
        raise ArchiveError(f"archive command contains secret-like text ({command_secret})")

    inventory = build_inventory(source, required_scopes=required)
    created = False
    try:
        try:
            destination.mkdir(mode=0o755)
        except FileExistsError as exc:
            raise ArchiveError(
                f"archive destination already exists; overwrite is forbidden: {destination}"
            ) from exc
        except OSError as exc:
            raise ArchiveError(f"cannot create archive destination {destination}: {exc}") from exc
        created = True
        incomplete = destination / ".archive-incomplete"
        _write_exclusive(incomplete, b"archive construction in progress\n")

        for directory in inventory.directories:
            assert directory.relative_path is not None
            target = _safe_destination(
                destination,
                archive_root=root,
                relative_path=directory.relative_path,
            )
            target.mkdir(parents=True, exist_ok=True, mode=0o755)
            if target.is_symlink() or not target.is_dir():
                raise ArchiveError(f"unsafe archive directory: {directory.relative_path}")

        for source_file in inventory.files:
            target = _safe_destination(
                destination,
                archive_root=root,
                relative_path=source_file.relative_path,
            )
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            _copy_source_file(source_file, target)

        _verify_file_snapshots(inventory.files)
        _verify_directory_snapshots((*inventory.guard_directories, *inventory.directories))
        payload_records = [source_file.manifest_record() for source_file in inventory.files]
        inventory_digest = _sha256_bytes(_canonical_json(payload_records))
        command_document = {
            "schema_version": 1,
            "kind": "hayatequant_run_archive_command",
            "run_id": safe_run_id,
            "source_run_root": str(source),
            "source_commit": safe_source_commit,
            "archiver_commit": safe_archiver_commit,
            "command": command,
            "allowed_scopes": list(ALLOWED_SCOPES),
            "required_scopes": list(required),
            "present_scopes": list(inventory.present_scopes),
            "missing_scopes": list(inventory.missing_scopes),
        }
        command_path = destination / "ARCHIVE_COMMAND.json"
        _write_exclusive(command_path, _canonical_json(command_document))
        command_record = _generated_record(command_path, relative_path="ARCHIVE_COMMAND.json")

        checksum_records = [*payload_records, command_record]
        checksum_records.sort(key=lambda record: str(record["path"]))
        checksum_payload = "".join(
            f"{record['sha256']}  {record['path']}\n" for record in checksum_records
        ).encode("utf-8")
        checksum_path = destination / "SHA256SUMS"
        _write_exclusive(checksum_path, checksum_payload)
        checksum_record = _generated_record(checksum_path, relative_path="SHA256SUMS")

        manifest_document = {
            "schema_version": 1,
            "kind": "hayatequant_repo_local_run_archive",
            "run_id": safe_run_id,
            "source_commit": safe_source_commit,
            "archiver_commit": safe_archiver_commit,
            "source_run_root": str(source),
            "archive_root": str(ARCHIVE_RELATIVE_ROOT / safe_run_id),
            "allowed_scopes": list(ALLOWED_SCOPES),
            "required_scopes": list(required),
            "present_scopes": list(inventory.present_scopes),
            "missing_scopes": list(inventory.missing_scopes),
            "scope_file_counts": inventory.scope_file_counts,
            "file_count": len(payload_records),
            "total_bytes": sum(int(record["bytes"]) for record in payload_records),
            "inventory_sha256": inventory_digest,
            "files": payload_records,
            "generated_files": [command_record, checksum_record],
        }
        manifest_payload = _canonical_json(manifest_document)
        manifest_path = destination / "ARCHIVE_MANIFEST.json"
        _write_exclusive(manifest_path, manifest_payload)
        detached_payload = f"{_sha256_bytes(manifest_payload)}  ARCHIVE_MANIFEST.json\n".encode(
            "utf-8"
        )
        _write_exclusive(destination / "ARCHIVE_MANIFEST.sha256", detached_payload)
        _verify_file_snapshots(inventory.files)
        _verify_directory_snapshots((*inventory.guard_directories, *inventory.directories))
        incomplete.unlink()
        return destination
    except BaseException:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        process = subprocess.run(
            ("git", "-C", str(repo_root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ArchiveError(f"cannot execute git: {exc}") from exc
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ArchiveError(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout.strip()


def _validated_repo_root(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    root = _canonical_existing_directory(path, label="repository root")
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if git_root != root:
        raise ArchiveError(f"--repo-root is not the Git worktree root: {root}")
    expected = root / "papers/hayatequant_paper/experiments"
    if not expected.is_dir():
        raise ArchiveError(f"repository has no HayateQuant experiment tree: {expected}")
    return root


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="absolute external RUN_ROOT")
    parser.add_argument("--run-id", required=True, help="new lowercase archive directory name")
    parser.add_argument(
        "--source-commit",
        required=True,
        help="full Git commit used to produce the source run",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="OneCompression Git worktree root (default: this script's worktree)",
    )
    parser.add_argument(
        "--require-scope",
        action="append",
        default=[],
        choices=ALLOWED_SCOPES,
        help="require a non-empty source scope; repeat for final-campaign coverage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        repo_root = _validated_repo_root(arguments.repo_root)
        source = _canonical_existing_directory(arguments.run_root, label="RUN_ROOT")
        if _is_relative_to(source, repo_root) or _is_relative_to(repo_root, source):
            raise ArchiveError("RUN_ROOT must be external to the repository worktree")
        source_commit = _validate_commit(arguments.source_commit, label="source commit")
        _git(repo_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
        archiver_commit = _validate_commit(
            _git(repo_root, "rev-parse", "HEAD"), label="archiver commit"
        )
        normalized_required = _normalize_required_scopes(arguments.require_scope)
        command = [
            PYTHON_EXECUTABLE,
            str(Path(__file__).resolve()),
            "--run-root",
            str(source),
            "--run-id",
            arguments.run_id,
            "--source-commit",
            source_commit,
            "--repo-root",
            str(repo_root),
        ]
        for scope in normalized_required:
            command.extend(("--require-scope", scope))
        destination = archive_run(
            run_root=source,
            archive_root=repo_root / ARCHIVE_RELATIVE_ROOT,
            run_id=arguments.run_id,
            source_commit=source_commit,
            archiver_commit=archiver_commit,
            required_scopes=normalized_required,
            archive_command=command,
        )
    except ArchiveError as exc:
        parser.exit(2, f"archive error: {exc}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
