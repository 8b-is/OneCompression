"""Tests for onecomp.compaction.store."""

from onecomp.compaction.store import ObsStore


def test_put_is_idempotent_and_ids_are_stable():
    store = ObsStore()
    rec1 = store.put(cmd_signature="grep -rn foo /testbed", full_content="match1\nmatch2\n")
    assert len(rec1.obs_id) == 8
    assert rec1.line_count == 2
    assert rec1.char_count == len("match1\nmatch2\n")
    assert store.has(rec1.obs_id)
    assert store.get(rec1.obs_id) is rec1

    rec_dup = store.put(cmd_signature="grep -rn foo /testbed", full_content="match1\nmatch2\n")
    assert rec_dup.obs_id == rec1.obs_id
    assert len(store) == 1

    rec2 = store.put(cmd_signature="ls /testbed", full_content="a.py\nb.py\n")
    assert rec2.obs_id != rec1.obs_id
    assert len(store) == 2


def test_get_is_case_insensitive():
    store = ObsStore()
    rec = store.put(cmd_signature="ls", full_content="x" * 1000)
    assert store.get(rec.obs_id.upper()) is rec
    assert store.get("ffffffff") is None
    assert store.get(None) is None


def test_head_tail_previews():
    store = ObsStore()
    content = "H" * 300 + "M" * 400 + "T" * 300
    rec = store.put(cmd_signature="cat big", full_content=content, head_chars=200, tail_chars=200)
    assert rec.head == "H" * 200
    assert rec.tail == "T" * 200


def test_find_by_signature_returns_all_runs():
    store = ObsStore()
    sig = "grep -rn foo /testbed"
    rec_a = store.put(cmd_signature=sig, full_content="first run")
    rec_b = store.put(cmd_signature=sig, full_content="second run (different)")
    found = store.find_by_signature(sig)
    assert {r.obs_id for r in found} == {rec_a.obs_id, rec_b.obs_id}
    assert store.find_by_signature("nonexistent") == []


def test_contains_value_and_prefix():
    store = ObsStore()
    store.put(cmd_signature="cat secrets", full_content="noise NEEDLE=abc123xyz noise")
    assert store.contains_value("abc123xyz")
    assert not store.contains_value("missing")
    assert not store.contains_value("")
    assert store.contains_value_with_prefix("NEEDLE=", "abc123xyz")
    assert not store.contains_value_with_prefix("TOKEN=", "abc123xyz")
    assert not store.contains_value_with_prefix("", "abc123xyz")


def test_serialization_is_lossless():
    store = ObsStore()
    store.put(cmd_signature="ls", full_content="a\nb\n")
    store.put(cmd_signature="cat foo", full_content="lorem", returncode=1)
    blob = store.to_serializable()
    assert len(blob) == 2
    assert all("full_content" in entry for entry in blob)


def test_wal_roundtrip(tmp_path):
    wal = tmp_path / "obs_store.wal"
    with ObsStore(wal_path=wal) as store:
        rec1 = store.put(cmd_signature="ls /testbed", full_content="a.py\nb.py\n")
        rec2 = store.put(cmd_signature="cat README", full_content="hello world")
        # Idempotent re-store must NOT add a second WAL line.
        store.put(cmd_signature="ls /testbed", full_content="a.py\nb.py\n")
        n_before = store.n_records

    lines = [l for l in wal.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    restored = ObsStore.from_wal(wal)
    assert restored.n_records == n_before
    assert restored.get(rec1.obs_id).full_content == "a.py\nb.py\n"
    assert restored.get(rec2.obs_id).full_content == "hello world"


def test_wal_replay_skips_truncated_line(tmp_path):
    wal = tmp_path / "crash.wal"
    store = ObsStore(wal_path=wal)
    store.put(cmd_signature="echo hi", full_content="hi\n")
    store.put(cmd_signature="echo bye", full_content="bye\n")
    store.close()

    with wal.open("a", encoding="utf-8") as f:
        f.write('{"cmd_signature": "partial", "full_conte')  # simulated crash

    restored = ObsStore.from_wal(wal)
    assert restored.n_records == 2


def test_wal_replay_skips_semantically_invalid_json_row(tmp_path):
    wal = tmp_path / "bad-types.wal"
    wal.write_text(
        '{"cmd_signature": "ok", "full_content": "first"}\n'
        '{"cmd_signature": "bad", "full_content": 42}\n'
        '{"cmd_signature": "ok", "full_content": "last"}\n',
        encoding="utf-8",
    )

    restored = ObsStore.from_wal(wal)
    assert restored.n_records == 2
    assert [record["full_content"] for record in restored.to_serializable()] == ["first", "last"]


def test_wal_resume_in_place_does_not_duplicate_replayed_rows(tmp_path):
    wal = tmp_path / "resume.wal"
    with ObsStore(wal_path=wal) as store:
        first = store.put(cmd_signature="echo first", full_content="first\n")

    resumed = ObsStore.from_wal(wal, new_wal_path=wal)
    assert resumed.get(first.obs_id) is not None
    assert len(wal.read_text().splitlines()) == 1
    resumed.put(cmd_signature="echo second", full_content="second\n")
    resumed.close()
    assert len(wal.read_text().splitlines()) == 2


def test_from_wal_missing_file_returns_empty_store(tmp_path):
    restored = ObsStore.from_wal(tmp_path / "never_written.wal")
    assert restored.n_records == 0


def test_obs_id_hex_chars_validation():
    import pytest

    with pytest.raises(ValueError):
        ObsStore(obs_id_hex_chars=3)
    with pytest.raises(ValueError):
        ObsStore(obs_id_hex_chars=41)
