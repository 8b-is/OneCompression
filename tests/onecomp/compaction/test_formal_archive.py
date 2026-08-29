"""Formal archive invariants and failure-injection tests."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from onecomp.compaction.formal import (
    ArchiveCapacityExceeded,
    ArchiveConfigurationMismatch,
    ArchiveIntegrityError,
    OccurrenceIdConfig,
    SQLiteObservationArchive,
    SourceOccurrenceConflict,
)


def test_identical_results_remain_distinct_occurrences(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        first = archive.append("same command", "same result", source_key="call-1")
        second = archive.append("same command", "same result", source_key="call-2")

        assert first.obs_id != second.obs_id
        assert first.sequence == 0
        assert second.sequence == 1
        assert archive.count == 2
        assert archive.verify_integrity(first.obs_id)

    assert path.stat().st_mode & 0o077 == 0


def test_source_key_retry_is_exactly_once_and_conflicts_fail_closed(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        first = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"attempt": 1},
            tags=("test", "stdout"),
        )
        retry = archive.append(
            "cmd",
            "body",
            source_key="event",
            created_step=99,
            extra={"attempt": 1},
            tags=("test", "stdout"),
        )
        assert retry.obs_id == first.obs_id
        with pytest.raises(SourceOccurrenceConflict):
            archive.append(
                "cmd",
                "body",
                source_key="event",
                extra={"attempt": 1},
                tags=("different",),
            )
        assert archive.count == 1


def test_source_key_retry_compares_canonical_json_metadata(tmp_path):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        first = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"nested": (1, 2)},
        )
        retry = archive.append(
            "cmd",
            "body",
            source_key="event",
            extra={"nested": [1, 2]},
        )
        assert retry.obs_id == first.obs_id
        with pytest.raises(ValueError, match="JSON-serializable"):
            archive.append("cmd", "body", extra={"invalid": float("nan")})


def test_transaction_rolls_back_body_and_sequence_on_metadata_failure(tmp_path, monkeypatch):
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        original = archive._meta_set

        def fail_sequence(key, value):
            if key == "next_sequence":
                raise OSError("injected durable metadata failure")
            return original(key, value)

        monkeypatch.setattr(archive, "_meta_set", fail_sequence)
        with pytest.raises(OSError, match="injected"):
            archive.append("cmd", "body")
        monkeypatch.setattr(archive, "_meta_set", original)

        record = archive.append("cmd", "body")
        assert record.sequence == 0
        assert archive.count == 1


def test_archive_quota_and_identifier_configuration_are_persistent(tmp_path):
    path = tmp_path / "formal.db"
    config = OccurrenceIdConfig(max_occurrences=4, fingerprint_bits=32)
    with SQLiteObservationArchive(
        path,
        id_config=config,
        max_archive_bytes=5,
    ) as archive:
        archive.append("cmd", "12345")
        with pytest.raises(ArchiveCapacityExceeded):
            archive.append("cmd", "6")

    with SQLiteObservationArchive(path) as reopened:
        assert reopened.id_config == config
        assert reopened.max_archive_bytes == 5

    with pytest.raises(ArchiveConfigurationMismatch):
        SQLiteObservationArchive(
            path,
            id_config=OccurrenceIdConfig(max_occurrences=8, fingerprint_bits=32),
        )


def test_archive_owner_binding_is_persistent_and_body_free(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        archive.bind_owner("tenant-a")
        archive.bind_owner("tenant-a")
    raw = sqlite3.connect(path)
    owner_digest = raw.execute("SELECT value FROM arc_meta WHERE key = 'owner_sha256'").fetchone()[
        0
    ]
    raw.close()
    assert owner_digest != "tenant-a"
    assert len(owner_digest) == 64

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveConfigurationMismatch, match="different owner"):
            archive.bind_owner("tenant-b")


def test_catalog_is_complete_body_free_filtered_and_cursor_stable(tmp_path):
    secret = "SECRET_MIDDLE_NEVER_RENDERED"
    with SQLiteObservationArchive(tmp_path / "formal.db") as archive:
        records = [
            archive.append(
                f"command {index}",
                f"body-{index}-{secret}",
                returncode=index % 2,
                created_step=index,
                tags=("even" if index % 2 == 0 else "odd",),
            )
            for index in range(7)
        ]
        pages = []
        cursor = None
        while True:
            page = archive.catalog_page(after_sequence=cursor, limit=3)
            pages.append(page)
            if not page.has_more:
                break
            cursor = page.next_cursor
        assert [item.obs_id for page in pages for item in page.items] == [
            record.obs_id for record in records
        ]
        assert secret not in repr(pages)
        assert [item.obs_id for item in archive.catalog_page(limit=10, tag="even").items] == [
            records[index].obs_id for index in (0, 2, 4, 6)
        ]
        assert [item.obs_id for item in archive.catalog_page(limit=10, returncode=1).items] == [
            records[index].obs_id for index in (1, 3, 5)
        ]
        assert archive.catalog_count() == 7
        assert archive.catalog_count(query="command 3") == 1
        assert archive.recent_ids(2) == [records[-2].obs_id, records[-1].obs_id]


def test_multiple_connections_allocate_each_occurrence_once(tmp_path):
    path = tmp_path / "concurrent.db"
    SQLiteObservationArchive(path).close()

    def write(worker):
        with SQLiteObservationArchive(path) as archive:
            return [
                archive.append(
                    "same",
                    "same",
                    source_key=f"worker-{worker}-call-{index}",
                ).obs_id
                for index in range(12)
            ]

    with ThreadPoolExecutor(max_workers=6) as pool:
        batches = list(pool.map(write, range(6)))
    obs_ids = [obs_id for batch in batches for obs_id in batch]
    assert len(obs_ids) == len(set(obs_ids)) == 72
    with SQLiteObservationArchive(path) as archive:
        assert archive.count == 72
        assert [
            archive.id_factory.sequence_from_id(obs_id) for obs_id in archive.known_ids()
        ] == list(range(72))


def test_read_verification_detects_body_tampering(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "trusted")
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE observations SET full_content = ? WHERE obs_id = ?",
        ("tampered", record.obs_id),
    )
    connection.commit()
    connection.close()

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveIntegrityError):
            archive.get(record.obs_id)


def test_every_body_returning_read_detects_tampering(tmp_path):
    path = tmp_path / "formal.db"
    with SQLiteObservationArchive(path) as archive:
        record = archive.append("cmd", "trusted needle")
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE observations SET full_content = ? WHERE obs_id = ?",
        ("tampered needle", record.obs_id),
    )
    connection.commit()
    connection.close()

    with SQLiteObservationArchive(path) as archive:
        with pytest.raises(ArchiveIntegrityError):
            archive.find_by_signature("cmd")
        with pytest.raises(ArchiveIntegrityError):
            archive.find_records_containing("needle")
        with pytest.raises(ArchiveIntegrityError):
            archive.contains_value("needle")
        with pytest.raises(ArchiveIntegrityError):
            archive.to_serializable(include_bodies=True)
        assert "full_content" not in archive.to_serializable(include_bodies=False)[0]


def test_archive_permissions_do_not_depend_on_process_umask(tmp_path):
    path = tmp_path / "private" / "formal.db"
    previous = os.umask(0)
    try:
        SQLiteObservationArchive(path).close()
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0


def test_archive_is_private_before_sqlite_opens_it(tmp_path, monkeypatch):
    path = tmp_path / "formal.db"
    original = sqlite3.connect
    observed_modes = []

    def connect(candidate, *args, **kwargs):
        observed_modes.append(Path(candidate).stat().st_mode & 0o777)
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr("onecomp.compaction.formal.archive.sqlite3.connect", connect)
    SQLiteObservationArchive(path).close()
    assert observed_modes == [0o600]


def test_archive_preserves_existing_parent_mode_and_protects_sidecars(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    path = shared / "formal.db"

    with SQLiteObservationArchive(path) as archive:
        archive.append("cmd", "body")
        assert shared.stat().st_mode & 0o777 == 0o755
        artifacts = [
            candidate
            for candidate in (
                path,
                Path(str(path) + "-wal"),
                Path(str(path) + "-shm"),
                Path(str(path) + "-journal"),
            )
            if candidate.exists()
        ]
        assert path in artifacts
        assert all(candidate.stat().st_mode & 0o077 == 0 for candidate in artifacts)


def test_committed_occurrence_survives_abrupt_process_exit(tmp_path):
    path = tmp_path / "crash.db"
    script = (
        "import os\n"
        "from onecomp.compaction.formal import SQLiteObservationArchive\n"
        f"a = SQLiteObservationArchive({str(path)!r})\n"
        "a.append('crash-command', 'committed-body', source_key='crash-call')\n"
        "os._exit(17)\n"
    )
    process = subprocess.run([sys.executable, "-c", script], check=False)
    assert process.returncode == 17
    with SQLiteObservationArchive(path) as archive:
        assert archive.n_records == 1
        committed = archive.get(archive.known_ids()[0])
        assert committed.full_content == "committed-body"
        following = archive.append("next", "body", source_key="next-call")
        assert following.sequence == 1


def test_occurrence_metadata_has_explicit_denial_of_service_bounds():
    with SQLiteObservationArchive() as archive:
        with pytest.raises(ValueError, match="cmd_signature"):
            archive.append("c" * 20_000, "body")
        with pytest.raises(ValueError, match="source_key"):
            archive.append("cmd", "body", source_key="s" * 2_000)
        with pytest.raises(ValueError, match="at most 64"):
            archive.append("cmd", "body", tags=[f"tag-{index}" for index in range(65)])
        with pytest.raises(ValueError, match="65536"):
            archive.append("cmd", "body", extra={"payload": "x" * 70_000})
