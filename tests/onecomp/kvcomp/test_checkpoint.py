"""Unit tests for the .okv KV-cache checkpoint/restore engine.

CPU-only; the DynamicCache test is skipped when transformers is missing.

Copyright 2025-2026 Fujitsu Ltd.
"""

import os

import pytest
import torch

from onecomp.kvcomp import (
    KVCheckpointCodec,
    KVCheckpointConfig,
    RolePrecisionPolicy,
    RoleSpan,
    extract_kv_layers,
)

NUM_LAYERS = 2
SHAPE = (1, 2, 32, 16)  # (batch, heads, seq, head_dim)


def _make_cache(seed=0):
    torch.manual_seed(seed)
    return tuple(
        (torch.randn(SHAPE, dtype=torch.float16), torch.randn(SHAPE, dtype=torch.float16))
        for _ in range(NUM_LAYERS)
    )


def _max_abs_error(cache, restored):
    err = 0.0
    for (key, value), (r_key, r_value) in zip(cache, extract_kv_layers(restored)):
        err = max(err, (key.float() - r_key.float()).abs().max().item())
        err = max(err, (value.float() - r_value.float()).abs().max().item())
    return err


@pytest.mark.parametrize("bits,tol", [(8, 0.05), (4, 0.5)])
def test_checkpoint_restore_roundtrip(tmp_path, bits, tol):
    cache = _make_cache()
    codec = KVCheckpointCodec(KVCheckpointConfig(bits=bits, store_dir=str(tmp_path)))
    info = codec.checkpoint(cache, session_id="sess")

    assert info.num_layers == NUM_LAYERS
    assert info.seq_length == SHAPE[2]
    assert os.path.exists(info.path)

    restored = codec.restore("sess")
    layers = extract_kv_layers(restored)
    assert len(layers) == NUM_LAYERS
    assert layers[0][0].shape == SHAPE
    assert _max_abs_error(cache, restored) < tol


def test_restored_cache_is_dynamic_cache(tmp_path):
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    cache = _make_cache()
    codec = KVCheckpointCodec(KVCheckpointConfig(bits=8, store_dir=str(tmp_path)))
    codec.checkpoint(cache, session_id="sess")
    restored = codec.restore("sess")
    assert isinstance(restored, DynamicCache)
    assert restored.get_seq_length() == SHAPE[2]


def test_zlib_backend_roundtrip(tmp_path):
    cache = _make_cache()
    raw_codec = KVCheckpointCodec(
        KVCheckpointConfig(bits=4, compress_backend="raw", store_dir=str(tmp_path / "raw"))
    )
    zlib_codec = KVCheckpointCodec(
        KVCheckpointConfig(bits=4, compress_backend="zlib", store_dir=str(tmp_path / "zlib"))
    )
    raw_info = raw_codec.checkpoint(cache, session_id="sess")
    zlib_info = zlib_codec.checkpoint(cache, session_id="sess")

    assert zlib_info.compressed_bytes <= raw_info.compressed_bytes
    raw_restored = extract_kv_layers(raw_codec.restore("sess"))
    zlib_restored = extract_kv_layers(zlib_codec.restore("sess"))
    for (r_key, r_value), (z_key, z_value) in zip(raw_restored, zlib_restored):
        assert torch.equal(r_key, z_key)
        assert torch.equal(r_value, z_value)


def test_list_delete_and_peek(tmp_path):
    cache = _make_cache()
    codec = KVCheckpointCodec(KVCheckpointConfig(store_dir=str(tmp_path)))
    info_a = codec.checkpoint(cache, session_id="a", metadata={"model": "demo", "turn": 3})
    codec.checkpoint(cache, session_id="b")
    assert codec.list_checkpoints() == ["a", "b"]

    # peek() reads the header only and reproduces the written info.
    peeked = codec.peek("a")
    assert peeked.metadata == {"model": "demo", "turn": 3}
    assert peeked.num_layers == NUM_LAYERS
    assert peeked.seq_length == SHAPE[2]
    assert peeked.compressed_bytes == info_a.compressed_bytes
    assert peeked.compression_ratio == pytest.approx(info_a.compression_ratio, rel=1e-9)

    assert codec.delete("a") is True
    assert codec.list_checkpoints() == ["b"]
    assert codec.delete("a") is False

    with pytest.raises(FileNotFoundError):
        codec.restore("a")
    with pytest.raises(FileNotFoundError):
        codec.peek("a")


def test_role_aware_checkpoint_roundtrip(tmp_path):
    cache = _make_cache()
    policy = RolePrecisionPolicy(bits_per_role={"system": 8, "tool": 2})
    codec = KVCheckpointCodec(KVCheckpointConfig(store_dir=str(tmp_path), role_policy=policy))
    spans = [RoleSpan(0, 16, "system"), RoleSpan(16, 32, "tool")]
    info = codec.checkpoint(cache, session_id="sess", spans=spans)
    assert info.seq_length == SHAPE[2]

    restored = codec.restore("sess")
    layers = extract_kv_layers(restored)
    assert layers[0][0].shape == SHAPE

    # The 8-bit system span must be reconstructed more accurately
    # than the 2-bit tool span.
    key, _ = cache[0]
    r_key, _ = layers[0]
    system_err = (key[:, :, :16].float() - r_key[:, :, :16].float()).abs().mean().item()
    tool_err = (key[:, :, 16:].float() - r_key[:, :, 16:].float()).abs().mean().item()
    assert system_err < tool_err


def test_checkpoint_does_not_mutate_caller_spans(tmp_path):
    cache = _make_cache()
    codec = KVCheckpointCodec(
        KVCheckpointConfig(store_dir=str(tmp_path), role_policy=RolePrecisionPolicy())
    )
    spans = [RoleSpan(16, 32, "tool"), RoleSpan(0, 16, "system")]  # unsorted on purpose
    before = [(s.start, s.end, s.role) for s in spans]
    codec.checkpoint(cache, session_id="mut", spans=spans)
    assert [(s.start, s.end, s.role) for s in spans] == before


def test_config_validates_at_construction():
    """All parameters fail fast at construction, not at checkpoint() time."""
    with pytest.raises(ValueError):
        KVCheckpointConfig(compress_backend="lz4")
    with pytest.raises(ValueError):
        KVCheckpointConfig(bits=3)
    with pytest.raises(ValueError):
        KVCheckpointConfig(key_axis="row")
    with pytest.raises(ValueError):
        KVCheckpointConfig(value_axis="row")
    with pytest.raises(ValueError):
        KVCheckpointConfig(group_size=0)


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "..", ""])
def test_session_id_with_path_separators_rejected(tmp_path, session_id):
    cache = _make_cache()
    codec = KVCheckpointCodec(KVCheckpointConfig(store_dir=str(tmp_path / "store")))
    with pytest.raises(ValueError):
        codec.checkpoint(cache, session_id=session_id)
    with pytest.raises(ValueError):
        codec.restore(session_id)
    assert not os.path.exists(str(tmp_path / "escape.okv"))


def test_corrupt_files_raise_value_error(tmp_path):
    cache = _make_cache()
    codec = KVCheckpointCodec(
        KVCheckpointConfig(bits=8, compress_backend="zlib", store_dir=str(tmp_path))
    )
    info = codec.checkpoint(cache, session_id="sess")
    with open(info.path, "rb") as f:
        blob = f.read()

    cases = {
        "badmagic": b"XXXX" + blob[4:],
        "shortlen": blob[:8],
        "shorthdr": blob[:64],
        "shortpay": blob[: len(blob) - 32],
        "badzlib": blob[:-64] + b"\x00" * 64,
    }
    for name, data in cases.items():
        with open(os.path.join(str(tmp_path), f"{name}.okv"), "wb") as f:
            f.write(data)
        with pytest.raises(ValueError):
            codec.restore(name)


def test_no_tmp_file_left_behind(tmp_path):
    cache = _make_cache()
    codec = KVCheckpointCodec(KVCheckpointConfig(store_dir=str(tmp_path)))
    codec.checkpoint(cache, session_id="sess")
    leftovers = [name for name in os.listdir(str(tmp_path)) if name.endswith(".tmp")]
    assert leftovers == []
