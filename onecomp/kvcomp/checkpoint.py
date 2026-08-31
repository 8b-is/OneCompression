"""KV-cache checkpoint/restore (session persistence) for agent workloads.

Agentic workloads are decode-dominated with very high KV-cache hit rates,
and agents blocked on tool execution keep their KV state resident in GPU
memory (KV thrashing).  This module lets an inference server quantize the
KV cache of an idle session, persist it to disk as a compact ``.okv``
file, evict the session, and later restore a transformers cache object
that seamlessly continues generation.

This is the *persistence* axis of :mod:`onecomp.kvcomp` -- orthogonal to
the live codecs/caches: quantization uses the bit-packed primitives in
:mod:`onecomp.kvcomp.codec._packing` (real storage, byte-serializable)
and the role-aware mode consumes
:class:`~onecomp.kvcomp.recipe.rolekv.RolePrecisionPolicy` /
:class:`~onecomp.kvcomp.recipe.rolekv.RoleSpan`.

The ``.okv`` container is a self-describing binary format: a magic tag,
a JSON header (layer count, shapes, bit-widths, byte counts, compression
ratio, user metadata) followed by the quantized payload, optionally
zlib-compressed.  The format is identical to the one written by the
``lab/agent-codec`` branch's ``onecomp.agent_codec`` module (okv v1);
files interchange between the two.

Classes:
    KVCheckpointConfig: Configuration for the checkpoint codec.
    KVCheckpointInfo: Metadata returned after writing a checkpoint.
    KVCheckpointCodec: Checkpoint/restore engine for transformers KV caches.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import io
import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional

import torch

from .codec._packing import (
    SUPPORTED_PACKED_BITS,
    PackedQuantizedKV,
    dequantize_packed,
    quantize_packed,
)
from .recipe.rolekv import RolePrecisionPolicy, RoleSpan, validate_spans

_MAGIC = b"OKV1"

# Sanity bound for the JSON header (metadata only, tensors live in the payload).
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_HEADER_LEN_FMT = "<Q"


def extract_kv_layers(past_key_values) -> list:
    """Extract per-layer ``(key, value)`` tensors from a transformers cache.

    Supports the transformers 5.x ``DynamicCache`` (``cache.layers[i].keys``
    / ``.values``), older caches exposing ``key_cache`` / ``value_cache``
    lists, and legacy tuples of ``(key, value)`` pairs.

    Args:
        past_key_values: A transformers cache object or a legacy tuple.

    Returns:
        List of ``(key, value)`` tensor pairs, one per layer, each of
        shape ``(batch, heads, seq, head_dim)``.
    """
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    return [(key, value) for key, value in past_key_values]


@dataclass
class KVCheckpointConfig:
    """Configuration for :class:`KVCheckpointCodec`.

    Attributes:
        bits: Uniform quantization bit-width (2, 4 or 8).  Ignored for
            spans covered by ``role_policy`` in role-aware mode.
        group_size: Quantization group size.
        compress_backend: ``"raw"`` (no compression) or ``"zlib"``.
        store_dir: Directory where ``.okv`` checkpoint files are written.
        key_axis: Quantization axis for keys (``"channel"`` or ``"token"``).
        value_axis: Quantization axis for values (``"channel"`` or ``"token"``).
        role_policy: Optional role-to-bits policy.  When set, checkpoints
            created with role spans quantize each span at its role's
            bit-width instead of the uniform ``bits``.
    """

    bits: int = 4
    group_size: int = 64
    compress_backend: str = "raw"
    store_dir: str = "./kv_checkpoints"
    key_axis: str = "channel"
    value_axis: str = "token"
    role_policy: Optional[RolePrecisionPolicy] = None

    def __post_init__(self):
        # Fail fast at construction (mirrors KVCodec.validate_params) instead
        # of erroring at checkpoint() time deep inside a long-running job.
        if self.bits not in SUPPORTED_PACKED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_PACKED_BITS}, got {self.bits!r}")
        if not (isinstance(self.group_size, int) and self.group_size > 0):
            raise ValueError(f"group_size must be a positive int, got {self.group_size!r}")
        if self.compress_backend not in ("raw", "zlib"):
            raise ValueError(
                f'compress_backend must be "raw" or "zlib", got {self.compress_backend!r}'
            )
        for attr in ("key_axis", "value_axis"):
            if getattr(self, attr) not in ("token", "channel"):
                raise ValueError(
                    f'{attr} must be "token" or "channel", got {getattr(self, attr)!r}'
                )


@dataclass
class KVCheckpointInfo:
    """Metadata describing a written checkpoint.

    Attributes:
        session_id: Session identifier (also the file stem).
        path: Absolute path of the ``.okv`` file.
        num_layers: Number of transformer layers in the cache.
        seq_length: Cached sequence length.
        original_bytes: Byte size of the raw (unquantized) KV tensors.
        compressed_bytes: Byte size of the ``.okv`` file on disk.
        compression_ratio: ``original_bytes / compressed_bytes``.
        metadata: User-supplied metadata stored in the header.
    """

    session_id: str
    path: str
    num_layers: int
    seq_length: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    metadata: dict = field(default_factory=dict)


def _pq_to_payload(pq: PackedQuantizedKV) -> dict:
    """Split a :class:`PackedQuantizedKV` into JSON-safe metadata and tensors."""
    meta = {
        "shape": list(pq.shape),
        "bits": pq.bits,
        "group_size": pq.group_size,
        "axis": pq.axis,
        "num_padded": pq.num_padded,
        "dtype": pq.dtype,
    }
    tensors = {"codes": pq.codes, "scale": pq.scale, "zero": pq.zero}
    return {"meta": meta, "tensors": tensors}


def _pq_from_payload(meta: dict, tensors: dict, device: str) -> PackedQuantizedKV:
    """Rebuild a :class:`PackedQuantizedKV` from header metadata and tensors."""
    return PackedQuantizedKV(
        codes=tensors["codes"].to(device),
        scale=tensors["scale"].to(device),
        zero=tensors["zero"].to(device),
        shape=tuple(meta["shape"]),
        bits=meta["bits"],
        group_size=meta["group_size"],
        axis=meta["axis"],
        num_padded=meta["num_padded"],
        dtype=meta["dtype"],
    )


def quantize_layers_with_roles(
    layer_tensors: list,
    spans: list,
    policy: RolePrecisionPolicy,
    group_size: int = 64,
    key_axis: str = "channel",
    value_axis: str = "token",
) -> tuple:
    """Quantize per-layer KV tensors with per-role bit-widths.

    The sequence dimension of every layer is sliced according to ``spans``
    and each slice is quantized (bit-packed) with the bit-width the policy
    assigns to its role.

    Args:
        layer_tensors: ``(key, value)`` pairs from :func:`extract_kv_layers`.
        spans: Role spans tiling the full sequence.
        policy: Role-to-bits mapping.
        group_size: Quantization group size.
        key_axis: Quantization axis for keys.
        value_axis: Quantization axis for values.

    Returns:
        ``(ordered_spans, layers)`` where ``layers[layer][span]`` is a
        ``(PackedQuantizedKV, PackedQuantizedKV)`` key/value pair.
    """
    seq_length = layer_tensors[0][0].shape[2]
    ordered = validate_spans(spans, seq_length)

    layers = []
    for key, value in layer_tensors:
        per_span = []
        for span in ordered:
            bits = policy.bits_for(span.role)
            key_slice = key[:, :, span.start : span.end, :]
            value_slice = value[:, :, span.start : span.end, :]
            qk = quantize_packed(key_slice, bits, group_size=group_size, axis=key_axis)
            qv = quantize_packed(value_slice, bits, group_size=group_size, axis=value_axis)
            per_span.append((qk, qv))
        layers.append(per_span)
    return ordered, layers


class KVCheckpointCodec:
    """Checkpoint/restore engine for transformers KV caches.

    Example:
        >>> codec = KVCheckpointCodec(KVCheckpointConfig(bits=4, store_dir="./ckpt"))
        >>> info = codec.checkpoint(outputs.past_key_values, session_id="agent-42")
        >>> cache = codec.restore("agent-42")
        >>> model.generate(..., past_key_values=cache)
    """

    def __init__(self, config: KVCheckpointConfig = None):
        """
        Args:
            config: Codec configuration.  Defaults to ``KVCheckpointConfig()``.
        """
        self.config = config if config is not None else KVCheckpointConfig()
        self.logger = getLogger(__name__)
        os.makedirs(self.config.store_dir, exist_ok=True)

    def _path_for(self, session_id: str) -> str:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if os.path.basename(session_id) != session_id or session_id in (".", ".."):
            raise ValueError(
                f"session_id must be a plain name without path separators, got {session_id!r}"
            )
        return os.path.join(self.config.store_dir, f"{session_id}.okv")

    def checkpoint(
        self,
        past_key_values,
        session_id: str,
        metadata: dict = None,
        spans: list = None,
    ) -> KVCheckpointInfo:
        """Quantize a KV cache and persist it as a ``.okv`` file.

        Args:
            past_key_values: Transformers cache (``DynamicCache`` or legacy
                tuple of ``(key, value)`` pairs).
            session_id: Identifier used as the checkpoint file stem.
            metadata: Optional JSON-serializable metadata stored in the
                file header (e.g. conversation state, model name).
            spans: Optional list of :class:`RoleSpan`.  When provided and
                ``config.role_policy`` is set, each span is quantized at
                its role's bit-width (role-aware mode).

        Returns:
            A :class:`KVCheckpointInfo` with byte counts and compression
            ratio.
        """
        layer_tensors = extract_kv_layers(past_key_values)
        if not layer_tensors:
            raise ValueError("past_key_values contains no layers; nothing to checkpoint")
        num_layers = len(layer_tensors)
        seq_lengths = {key.shape[2] for key, _ in layer_tensors}
        if len(seq_lengths) != 1:
            raise ValueError(
                "KV cache layers have differing sequence lengths "
                f"({sorted(seq_lengths)}); sliding-window / hybrid caches are "
                "not supported because cache indices no longer match token "
                "positions"
            )
        seq_length = layer_tensors[0][0].shape[2]
        original_bytes = sum(
            t.numel() * t.element_size() for key, value in layer_tensors for t in (key, value)
        )

        role_aware = spans is not None and self.config.role_policy is not None
        if (spans is not None) != (self.config.role_policy is not None):
            self.logger.warning(
                "Role-aware quantization is inactive: %s. Falling back to "
                "uniform %d-bit quantization.",
                (
                    "spans were provided but config.role_policy is not set"
                    if spans is not None
                    else "config.role_policy is set but no spans were provided"
                ),
                self.config.bits,
            )
        tensors = {}
        if role_aware:
            ordered_spans, role_layers = quantize_layers_with_roles(
                layer_tensors,
                spans,
                policy=self.config.role_policy,
                group_size=self.config.group_size,
                key_axis=self.config.key_axis,
                value_axis=self.config.value_axis,
            )
            layers_meta = []
            for layer_idx, per_span in enumerate(role_layers):
                span_meta = []
                for span_idx, (qk, qv) in enumerate(per_span):
                    key_payload = _pq_to_payload(qk)
                    value_payload = _pq_to_payload(qv)
                    prefix = f"l{layer_idx}.s{span_idx}"
                    for name, tensor in key_payload["tensors"].items():
                        tensors[f"{prefix}.k.{name}"] = tensor
                    for name, tensor in value_payload["tensors"].items():
                        tensors[f"{prefix}.v.{name}"] = tensor
                    span_meta.append({"k": key_payload["meta"], "v": value_payload["meta"]})
                layers_meta.append(span_meta)
            spans_meta = [{"start": s.start, "end": s.end, "role": s.role} for s in ordered_spans]
        else:
            layers_meta = []
            for layer_idx, (key, value) in enumerate(layer_tensors):
                qk = quantize_packed(
                    key,
                    self.config.bits,
                    group_size=self.config.group_size,
                    axis=self.config.key_axis,
                )
                qv = quantize_packed(
                    value,
                    self.config.bits,
                    group_size=self.config.group_size,
                    axis=self.config.value_axis,
                )
                key_payload = _pq_to_payload(qk)
                value_payload = _pq_to_payload(qv)
                prefix = f"l{layer_idx}"
                for name, tensor in key_payload["tensors"].items():
                    tensors[f"{prefix}.k.{name}"] = tensor
                for name, tensor in value_payload["tensors"].items():
                    tensors[f"{prefix}.v.{name}"] = tensor
                layers_meta.append({"k": key_payload["meta"], "v": value_payload["meta"]})
            spans_meta = None

        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        payload = buffer.getvalue()
        if self.config.compress_backend == "zlib":
            payload = zlib.compress(payload)

        header = {
            "format": "okv",
            "version": 1,
            "num_layers": num_layers,
            "seq_length": seq_length,
            "bits": self.config.bits,
            "group_size": self.config.group_size,
            "compress_backend": self.config.compress_backend,
            "role_aware": role_aware,
            "spans": spans_meta,
            "layers": layers_meta,
            "original_bytes": original_bytes,
            "payload_bytes": len(payload),
            "metadata": metadata or {},
        }
        header_bytes = json.dumps(header).encode("utf-8")

        path = self._path_for(session_id)
        # Atomic write: a crash mid-write must not leave a corrupt .okv
        # behind under the session's name.
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(_MAGIC)
                f.write(struct.pack(_HEADER_LEN_FMT, len(header_bytes)))
                f.write(header_bytes)
                f.write(payload)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        compressed_bytes = os.path.getsize(path)
        ratio = original_bytes / compressed_bytes if compressed_bytes else 0.0
        self.logger.info(
            "Checkpointed session %s: %d layers, seq=%d, %d -> %d bytes (x%.2f)",
            session_id,
            num_layers,
            seq_length,
            original_bytes,
            compressed_bytes,
            ratio,
        )
        return KVCheckpointInfo(
            session_id=session_id,
            path=os.path.abspath(path),
            num_layers=num_layers,
            seq_length=seq_length,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=ratio,
            metadata=metadata or {},
        )

    def _read_header(self, f, path: str) -> dict:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f"{path} is not a valid .okv checkpoint file")
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is truncated: incomplete header length field")
        (header_len,) = struct.unpack(_HEADER_LEN_FMT, length_bytes)
        if header_len > _MAX_HEADER_BYTES:
            raise ValueError(
                f"{path} has an implausible header length ({header_len} bytes); "
                "the file is corrupt or not an .okv checkpoint"
            )
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path} is truncated: incomplete header")
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path} has a corrupt header: {e}") from e
        if header.get("format") != "okv" or header.get("version") != 1:
            raise ValueError(
                f"{path} has an unsupported checkpoint format "
                f"(format={header.get('format')!r}, "
                f"version={header.get('version')!r}); this reader "
                "supports okv version 1"
            )
        missing = [
            key
            for key in ("payload_bytes", "compress_backend", "num_layers", "layers")
            if key not in header
        ]
        if missing:
            raise ValueError(f"{path} header is missing required keys: {missing}")
        return header

    def _read_file(self, session_id: str) -> tuple:
        path = self._path_for(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found for session {session_id!r} at {path}")
        with open(path, "rb") as f:
            header = self._read_header(f, path)
            payload = f.read(header["payload_bytes"])
        if len(payload) != header["payload_bytes"]:
            raise ValueError(
                f"{path} is truncated: expected {header['payload_bytes']} payload bytes, "
                f"got {len(payload)}"
            )
        if header["compress_backend"] == "zlib":
            try:
                payload = zlib.decompress(payload)
            except zlib.error as e:
                raise ValueError(f"{path} has a corrupt zlib payload: {e}") from e
        tensors = torch.load(io.BytesIO(payload), weights_only=True)
        return header, tensors

    def peek(self, session_id: str) -> KVCheckpointInfo:
        """Read a checkpoint's metadata without loading its payload.

        Args:
            session_id: Identifier used at checkpoint time.

        Returns:
            A :class:`KVCheckpointInfo` reconstructed from the file header.
        """
        path = self._path_for(session_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found for session {session_id!r} at {path}")
        with open(path, "rb") as f:
            header = self._read_header(f, path)
        compressed_bytes = os.path.getsize(path)
        original_bytes = header.get("original_bytes", 0)
        return KVCheckpointInfo(
            session_id=session_id,
            path=os.path.abspath(path),
            num_layers=header["num_layers"],
            seq_length=header.get("seq_length", 0),
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            compression_ratio=(original_bytes / compressed_bytes if compressed_bytes else 0.0),
            metadata=header.get("metadata", {}),
        )

    def restore(self, session_id: str, device: str = "cpu"):
        """Restore a checkpoint into a transformers cache object.

        Args:
            session_id: Identifier used at checkpoint time.
            device: Device for the restored KV tensors.

        Returns:
            A ``transformers.DynamicCache`` ready to be passed as
            ``past_key_values`` for continued generation.  Falls back to
            a legacy tuple if ``DynamicCache`` cannot be constructed.
        """
        header, tensors = self._read_file(session_id)

        def _load_pq(prefix, meta):
            fields = {name: tensors[f"{prefix}.{name}"] for name in ("codes", "scale", "zero")}
            return _pq_from_payload(meta, fields, device)

        layer_pairs = []
        if header["role_aware"]:
            # Validate the stored spans (raises on tampered role names).
            _ = [RoleSpan(**span) for span in header["spans"]]
            for layer_idx, spans_meta in enumerate(header["layers"]):
                keys, values = [], []
                for span_idx, span_meta in enumerate(spans_meta):
                    prefix = f"l{layer_idx}.s{span_idx}"
                    keys.append(dequantize_packed(_load_pq(f"{prefix}.k", span_meta["k"])))
                    values.append(dequantize_packed(_load_pq(f"{prefix}.v", span_meta["v"])))
                layer_pairs.append(
                    (torch.cat(keys, dim=2).to(device), torch.cat(values, dim=2).to(device))
                )
        else:
            for layer_idx, layer_meta in enumerate(header["layers"]):
                qk = _load_pq(f"l{layer_idx}.k", layer_meta["k"])
                qv = _load_pq(f"l{layer_idx}.v", layer_meta["v"])
                layer_pairs.append(
                    (dequantize_packed(qk).to(device), dequantize_packed(qv).to(device))
                )

        try:
            from transformers import DynamicCache  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.logger.warning("transformers unavailable; returning a legacy tuple cache")
            return tuple(layer_pairs)

        # Build the cache through the public update() API, which exists and
        # has the same semantics across transformers 4.36 - 5.x (constructor
        # keywords such as ddp_cache_data do not).
        cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(layer_pairs):
            cache.update(key, value, layer_idx)
        return cache

    def list_checkpoints(self) -> list:
        """List session ids of all checkpoints in ``store_dir``.

        Returns:
            Sorted list of session id strings.
        """
        sessions = [
            name[: -len(".okv")]
            for name in os.listdir(self.config.store_dir)
            if name.endswith(".okv")
        ]
        return sorted(sessions)

    def delete(self, session_id: str) -> bool:
        """Delete the checkpoint of a session.

        Args:
            session_id: Identifier used at checkpoint time.

        Returns:
            True if a file was deleted, False if none existed.
        """
        path = self._path_for(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
