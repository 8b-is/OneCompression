"""Core abstractions for KV cache compression.

This module defines the three orthogonal primitives used throughout
:mod:`onecomp.kvcomp`, mirroring the design philosophy of
:class:`onecomp.quantizer._quantizer.Quantizer` (dataclass-based, ``name``
auto-populated, ``flag_*`` capability flags, ``validate_params`` hook):

* :class:`KVCodec` -- quantizes the K/V tensors (dtype compression).
* :class:`KVSelector` -- evicts / selects tokens (token compression).
* :class:`KVPolicy` -- chooses codec/selector/budget given runtime state.
* :class:`ComposedPress` -- binds a codec + selector + policy into the single
  object consumed by the cache / attention integration.

Tensor layout convention (matching HuggingFace attention modules)::

    key, value: (batch, num_kv_heads, seq_len, head_dim)

Codecs implement *simulated* (fake) quantization by default: ``encode`` stores
a compressed representation and ``decode`` reconstructs a floating point tensor,
exactly like ``QuantizationResult.compute_dequantized_weight`` on the weight
side.  This keeps the research layer correctness-first and backend-agnostic;
Triton/CUDA kernels live in :mod:`onecomp.kvcomp.kernels`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from .config import Budget


@dataclass
class CompressedKV:
    """Container for a codec's compressed key/value representation.

    The fields are intentionally generic so that very different codecs
    (affine integer, FP8, rotated/Lloyd-Max TurboQuant, ...) can all reuse the
    same container.  ``decode`` reconstructs ``(key, value)`` from these fields.

    Attributes:
        key_q: Quantized key payload (integer codes, packed bits, or FP8).
        value_q: Quantized value payload.
        key_meta: Codec-specific metadata for the key (scale, zero, norm,
            rotation seed, codebook id, ...).
        value_meta: Codec-specific metadata for the value.
        key_residual: Recent keys kept in full precision (residual window).
        value_residual: Recent values kept in full precision.
        meta: Free-form metadata shared by key and value (layer index, dtype,
            original shape, ...).
    """

    key_q: Any = None
    value_q: Any = None
    key_meta: dict = field(default_factory=dict)
    value_meta: dict = field(default_factory=dict)
    key_residual: Optional[torch.Tensor] = None
    value_residual: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)

    def numel_bits(self) -> Optional[int]:
        """Return the codec's logical fixed-width data-plane bits, if known.

        For the rotated low-bit codecs this is the growth of the encoded KV
        data plane: codes, signed-fp16 scales, and explicitly enabled fields
        such as an outlier-channel bitmap or QJL signs/norms.  It deliberately
        excludes self-describing Python control tags, dictionary/container
        bytes, allocator overhead, and global codebook/LUT/rotation state.
        Those controls are out-of-band/cache-level configuration, so this
        value is not the resident memory of the materialized Python payload
        and is not evidence of a packed serving backend.  Returns ``None``
        when the codec does not provide this accounting contract.
        """
        stored_bits = self.meta.get("stored_bits")
        if stored_bits is None:
            return None
        if isinstance(stored_bits, bool) or not isinstance(stored_bits, int) or stored_bits < 0:
            raise ValueError(
                "CompressedKV meta['stored_bits'] must be a non-negative integer or None"
            )
        return stored_bits


@dataclass
class KVCodec:
    """Abstract base class for KV dtype quantization (encode/decode).

    A ``KVCodec`` is the KV-cache analogue of
    :class:`onecomp.quantizer._quantizer.Quantizer`.  Subclasses implement
    :meth:`encode` and :meth:`decode`; optionally :meth:`score` (for codecs
    that can compute attention scores without a full dequantization, e.g.
    TurboQuant LUT attention) and :meth:`calibrate` (for codecs that require
    offline calibration, e.g. KVQuant / KVTuner).

    The ``flag_calibration`` flag -- not the folder location -- distinguishes
    calibration-based codecs (``True``) from data-free / online codecs such as
    FP8, KIVI and TurboQuant (``False``).  This matches OneComp's existing
    convention where RTN (no calibration) and GPTQ (calibration) live side by
    side in :mod:`onecomp.quantizer`.

    Attributes:
        name (str): Codec name (defaults to the class name).
        flag_calibration (bool): Whether offline calibration is required.
        bits (int): Nominal bit-width of the quantized payload.
        key_axis (str): ``"channel"`` (per head-dim channel) or ``"token"``.
        value_axis (str): ``"token"`` (per token) or ``"channel"``.
        group_size (int): Group size along the quantization axis (-1 = whole).
        symmetric (bool): Symmetric quantization (zero point at center).
        skip_layers (list[int]): Layer indices to leave uncompressed.
        residual_window (int): Number of most-recent tokens kept in full precision
            by cache integrations (KIVI-style).
    """

    name: str = None
    flag_calibration: bool = False

    bits: int = 8
    key_axis: str = "channel"
    value_axis: str = "token"
    group_size: int = -1
    symmetric: bool = False
    skip_layers: list = field(default_factory=list)
    residual_window: int = 128

    def __post_init__(self):
        if self.name is None:
            self.name = getattr(type(self), "_registry_name", type(self).__name__)
        self.logger = getLogger(__name__)
        # Fail fast on invalid parameters at construction (mirrors the weight-side
        # Quantizer): callers that build codecs directly (e.g. the benchmark
        # harness) get the same validation the Runner performs.
        self.validate_params()

    def validate_params(self):
        """Validate codec parameters.  Override in subclasses as needed."""

    def calibrate(self, kv_stats, ctx=None):  # pylint: disable=unused-argument
        """Fit codec parameters from collected K/V statistics.

        Only meaningful when ``flag_calibration`` is True.  Default is a no-op.
        """

    def encode(self, key, value, ctx=None):  # pylint: disable=unused-argument
        """Compress ``(key, value)`` into a :class:`CompressedKV`.

        Args:
            key: Key tensor ``(batch, num_kv_heads, seq_len, head_dim)``.
            value: Value tensor of the same shape.
            ctx: Optional context dict (layer_idx, position_ids, ...).

        Returns:
            CompressedKV: Compressed representation.
        """
        raise NotImplementedError

    def decode(self, compressed: CompressedKV, ctx=None):  # pylint: disable=unused-argument
        """Reconstruct ``(key, value)`` floating point tensors."""
        raise NotImplementedError

    def score(self, query, compressed: CompressedKV, ctx=None):
        """Optionally compute attention logits without full dequantization.

        Returns ``None`` by default, signalling the caller to fall back to
        ``decode`` followed by a standard ``query @ key^T``.
        """
        _ = (query, compressed, ctx)
        return None

    def fake_quantize(self, key, value, ctx=None):
        """Convenience: ``decode(encode(...))`` -- the simulated-quant path.

        This is the KV analogue of ``compute_dequantized_weight`` and is what
        the HF research-layer cache uses to emulate the codec without custom
        kernels.
        """
        return self.decode(self.encode(key, value, ctx=ctx), ctx=ctx)

    def fake_quant_roundtrip(self, key, value, ctx=None):
        """Encode-then-decode in one call: ``(key_hat, value_hat, stored_bits)``.

        The fake-quant serving path (:class:`CompressedDynamicCache`) never
        inspects the payload -- it only needs the dequantized tensors and the
        bit accounting.  This default simply chains :meth:`encode` and
        :meth:`decode`; codecs on the per-token hot path may override it with
        a fused implementation that skips payload construction and the second
        codebook gather entirely (see ``HayateQuantCodec``), which matters
        because the per-step codec cost is dominated by kernel-launch
        overhead, not compute.

        Returns:
            ``(key_hat, value_hat, stored_bits)`` where ``stored_bits`` may be
            ``None`` when the codec does not record bit accounting.
        """
        compressed = self.encode(key, value, ctx=ctx)
        k, v = self.decode(compressed, ctx=ctx)
        return k, v, compressed.numel_bits()

    def teacher_forced_temporal_dependency(self) -> Optional[str]:
        """Describe statistics that couple one token to later tokens, if any.

        A causal mask prevents attention to future *positions*, but it cannot
        undo future dependence introduced before attention.  In particular, a
        channel-axis codec fits a scale to a sequence-axis group.  Encoding a
        complete teacher-forced window in one call can therefore change an
        early key using later tokens from that window.  PPL/log-likelihood
        evaluators use this contract to switch to causal prefix/incremental
        scoring.

        Token-axis codecs are local to each token by default.  A subclass that
        computes any other block-global statistic (for example a dynamic
        outlier-channel set) must override this method even when both declared
        axes are ``"token"``.
        """
        if "channel" in (self.key_axis, self.value_axis):
            return "sequence_axis_group_statistics"
        return None

    def get_kv_config(self) -> dict:
        """Return a JSON-serialisable config dict (for ``config.json`` save)."""
        return {
            "name": self.name,
            "bits": self.bits,
            "key_axis": self.key_axis,
            "value_axis": self.value_axis,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "skip_layers": list(self.skip_layers),
            "residual_window": self.residual_window,
        }


@dataclass
class KVSelector:
    """Abstract base class for token eviction / selection.

    Selectors decide *which tokens to keep*.  They are applied at the prefill
    boundary (the KVCache-Factory invariant: ``key_len == total_seq_len``).
    Subclasses implement :meth:`score` and :meth:`select`.

    Attributes:
        name (str): Selector name (defaults to the class name).
        flag_needs_attention (bool): Whether attention scores are required
            (e.g. SnapKV, H2O).  When True the cache integration registers an
            attention forward-hook to capture the statistics.
    """

    name: str = None
    flag_needs_attention: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = getattr(type(self), "_registry_name", type(self).__name__)
        self.logger = getLogger(__name__)
        self.validate_params()

    def validate_params(self):
        """Validate selector parameters.  Override as needed."""

    def score(self, query, key, value, attn_stats, layer_idx):
        """Return a per-token importance score ``(batch, num_kv_heads, seq)``.

        ``attn_stats`` carries optional precomputed attention information for
        selectors with ``flag_needs_attention=True``.
        """
        raise NotImplementedError

    def select(self, scores, budget: "Budget", seq_len: int):
        """Return a boolean keep-mask ``(batch, num_kv_heads, seq_len)``.

        Implementations should always preserve the sink and recent windows
        defined by ``budget`` regardless of score.
        """
        raise NotImplementedError

    def get_kv_config(self) -> dict:
        """Return a JSON-serialisable config dict of the selector's fields."""
        import dataclasses  # pylint: disable=import-outside-toplevel

        out = {"name": self.name}
        for f in dataclasses.fields(self):
            if f.name in ("name", "flag_needs_attention"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, (int, float, str, bool, list, dict, type(None))):
                out[f.name] = val
        return out


@dataclass
class KVPolicy:
    """Abstract base class for runtime compression policies.

    A policy maps the current request/system state to a
    :class:`~onecomp.kvcomp.config.CompressionPlan`.  Static recipes return a
    fixed plan; adaptive recipes (ThinKV, service-aware) switch codec/selector
    and budget dynamically.
    """

    name: str = None

    def __post_init__(self):
        if self.name is None:
            self.name = getattr(type(self), "_registry_name", type(self).__name__)
        self.logger = getLogger(__name__)

    def plan(self, request_state=None, system_state=None):
        """Return a :class:`CompressionPlan` for the given state."""
        raise NotImplementedError

    def get_kv_config(self) -> dict:
        """Return a JSON-serialisable config dict."""
        return {"name": self.name}


@dataclass
class ComposedPress:
    """Binds a codec + selector + policy into one object.

    This is the single window through which the cache / attention integration
    drives compression.  ``flag_*`` are derived from the components so the
    Runner / cache can decide whether to run calibration or register attention
    hooks.
    """

    codec: Optional[KVCodec] = None
    selector: Optional[KVSelector] = None
    policy: Optional[KVPolicy] = None
    residual_window: int = 128

    flag_calibration: bool = field(init=False, default=False)
    flag_needs_attention: bool = field(init=False, default=False)

    def __post_init__(self):
        self.flag_calibration = bool(self.codec is not None and self.codec.flag_calibration)
        self.flag_needs_attention = bool(
            self.selector is not None and self.selector.flag_needs_attention
        )
        if self.codec is not None:
            self.residual_window = self.codec.residual_window
