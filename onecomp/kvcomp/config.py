"""Configuration dataclasses for KV cache compression.

These mirror the role of :class:`onecomp.calibration.CalibrationConfig` and
:class:`onecomp.qep.QEPConfig` on the weight side, and are what users pass to
``Runner(kv_config=...)``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._base import KVCodec, KVSelector, KVPolicy, ComposedPress


@dataclass
class Budget:
    """Token budget for a selector.

    Attributes:
        keep_ratio: Fraction of prompt tokens to keep (``None`` = use
            ``keep_tokens``).
        keep_tokens: Absolute number of tokens to keep (overrides ratio when
            set).
        sink_tokens: Number of leading "attention sink" tokens always kept.
        recent_tokens: Number of most-recent tokens always kept.
        per_layer: Optional per-layer override of ``keep_tokens``.
        per_head: Optional per-(layer, head) override of ``keep_tokens``.
    """

    keep_ratio: Optional[float] = None
    keep_tokens: Optional[int] = None
    sink_tokens: int = 4
    recent_tokens: int = 256
    per_layer: Optional[dict] = None
    per_head: Optional[dict] = None

    def resolve_keep(self, seq_len: int, layer_idx: Optional[int] = None) -> int:
        """Resolve the number of tokens to keep for a given sequence length."""
        if self.per_layer is not None and layer_idx in self.per_layer:
            keep = self.per_layer[layer_idx]
        elif self.keep_tokens is not None:
            keep = self.keep_tokens
        elif self.keep_ratio is not None:
            keep = int(round(self.keep_ratio * seq_len))
        else:
            keep = seq_len
        # The sink + recent windows are always retained.
        keep = max(keep, self.sink_tokens + self.recent_tokens)
        return min(keep, seq_len)


@dataclass
class RequestState:
    """Per-request information consumed by adaptive policies."""

    prompt_len: int = 0
    generated_len: int = 0
    max_new_tokens: int = 0
    task_type: str = "chat"  # chat / rag / summarization / code / math / reasoning / agent
    quality_slo: str = "balanced"  # strict / balanced / best_effort
    latency_slo_ms: Optional[int] = None


@dataclass
class SystemState:
    """System-level information consumed by adaptive policies."""

    free_gpu_mem_gb: float = 0.0
    kv_mem_gb: float = 0.0
    decode_tokens_per_sec: float = 0.0
    network_bandwidth_gbps: Optional[float] = None
    gpu_arch: str = ""


@dataclass
class CompressionPlan:
    """Concrete plan returned by a :class:`~onecomp.kvcomp.KVPolicy`."""

    codec: Optional[KVCodec] = None
    selector: Optional[KVSelector] = None
    budget: Budget = field(default_factory=Budget)
    skip_layers: list = field(default_factory=list)
    residual_window: int = 128


@dataclass
class KVCompressionConfig:
    """Top-level KV compression recipe passed to ``Runner(kv_config=...)``.

    Attributes:
        codec: KV dtype quantizer (or ``None`` for eviction-only recipes).
        selector: Token selector (or ``None`` for quant-only recipes).
        policy: Runtime policy (defaults to a static policy when ``None``).
        residual_window: Recent tokens kept in full precision.

    Examples:
        FP8 + SnapKV recipe::

            from onecomp.kvcomp import KVCompressionConfig
            from onecomp.kvcomp.codec import FP8Codec
            from onecomp.kvcomp.select import SnapKVSelector

            kv_config = KVCompressionConfig(
                codec=FP8Codec(),
                selector=SnapKVSelector(keep_ratio=0.3),
            )
    """

    codec: Optional[KVCodec] = None
    selector: Optional[KVSelector] = None
    policy: Optional[KVPolicy] = None
    residual_window: int = 128

    def to_press(self) -> ComposedPress:
        """Build the :class:`ComposedPress` consumed by the cache integration."""
        policy = self.policy
        if policy is None:
            # Local import avoids a circular import at module load time.
            from .policy.static import StaticPolicy  # pylint: disable=import-outside-toplevel

            policy = StaticPolicy()
        return ComposedPress(
            codec=self.codec,
            selector=self.selector,
            policy=policy,
            residual_window=self.residual_window,
        )

    def to_dict(self) -> dict:
        """Serialise to the ``kv_compression_config`` block of ``config.json``."""
        return {
            "codec": self.codec.get_kv_config() if self.codec is not None else None,
            "selector": self.selector.get_kv_config() if self.selector is not None else None,
            "policy": self.policy.get_kv_config() if self.policy is not None else {"name": "static"},
            "residual_window": self.residual_window,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KVCompressionConfig":
        """Reconstruct a config from a saved ``kv_compression_config`` block."""
        from ._registry import (  # pylint: disable=import-outside-toplevel
            build_codec,
            build_selector,
            build_policy,
        )

        codec = None
        if data.get("codec"):
            cfg = dict(data["codec"])
            codec = build_codec(cfg.pop("name"), **cfg)
        selector = None
        if data.get("selector"):
            cfg = dict(data["selector"])
            selector = build_selector(cfg.pop("name"), **cfg)
        policy = None
        if data.get("policy"):
            cfg = dict(data["policy"])
            policy = build_policy(cfg.pop("name"), **cfg)
        return cls(
            codec=codec,
            selector=selector,
            policy=policy,
            residual_window=data.get("residual_window", 128),
        )
