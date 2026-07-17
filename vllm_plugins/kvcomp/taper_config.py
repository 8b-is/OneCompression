"""Validated configuration for the vLLM TapeKV reference backend.

The backend deliberately keeps vLLM's ordinary FP16/BF16 paged allocation and
applies TapeKV's affine fake-quantisation before attention.  It is therefore a
*semantic/reference* implementation: it is suitable for generation and
quality checks, but must never be reported as a packed-memory or throughput
result.  Keeping that distinction in one small, dependency-free module makes
it hard for launchers to accidentally make a physical-compression claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_SUPPORTED_POLICIES = {"taperkv", "block_taperkv"}


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be >= {minimum}{suffix}, got {result}")
    return result


@dataclass(frozen=True)
class TaperVLLMRecipe:
    """One explicit TapeKV semantic recipe accepted by the vLLM plugin."""

    policy_name: str
    key_bits: int
    value_bits: int
    b_min: int
    b_min_value: int
    sink_tokens: int = 4
    recent_tokens: int = 64
    ladder: tuple[int, ...] | None = None
    skip_layers: tuple[int, ...] = ()
    block_size: int | None = None
    residual_fp16: bool = True
    quant_granularity: str = "block"
    implementation: str = "fp16_reference_fake_quant"
    hayate_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.policy_name not in _SUPPORTED_POLICIES:
            raise ValueError(f"unsupported TapeKV policy {self.policy_name!r}")
        for name in ("key_bits", "value_bits", "b_min", "b_min_value"):
            value = _integer(getattr(self, name), name, minimum=1, maximum=16)
            object.__setattr__(self, name, value)
        if self.value_bits > self.key_bits:
            raise ValueError("value_bits must not exceed key_bits")
        if self.b_min > self.key_bits or self.b_min_value > self.value_bits:
            raise ValueError("TapeKV floor bits must not exceed their high bits")
        for name in ("sink_tokens", "recent_tokens"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        ladder = self.ladder
        if ladder is not None:
            ladder = tuple(_integer(value, "ladder entry", minimum=1, maximum=self.key_bits)
                           for value in ladder)
            if not ladder or any(left <= right for left, right in zip(ladder, ladder[1:])):
                raise ValueError("ladder must be a non-empty strictly descending sequence")
            object.__setattr__(self, "ladder", ladder)
        skip = tuple(_integer(value, "skip_layers entry", minimum=0) for value in self.skip_layers)
        object.__setattr__(self, "skip_layers", skip)
        if self.policy_name == "block_taperkv":
            object.__setattr__(self, "block_size", _integer(self.block_size, "block_size", minimum=1))
            if not isinstance(self.residual_fp16, bool):
                raise ValueError("residual_fp16 must be a bool")
            if self.quant_granularity not in {"block", "token"}:
                raise ValueError("quant_granularity must be 'block' or 'token'")
        if self.hayate_config is not None:
            if not isinstance(self.hayate_config, dict) or self.hayate_config.get("name") != "hayatequant":
                raise ValueError("hayate_config must be a serialized HayateQuant codec mapping")
            object.__setattr__(self, "implementation", "hayatequant_then_taper_fp16_reference")
        elif self.implementation != "fp16_reference_fake_quant":
            raise ValueError("only the explicit FP16 reference TapeKV backend is supported")

    @classmethod
    def from_policy(cls, policy: dict[str, Any], hayate_config: dict[str, Any] | None = None) -> "TaperVLLMRecipe":
        if not isinstance(policy, dict):
            raise ValueError("TapeKV policy must be a mapping")
        name = str(policy.get("name", ""))
        return cls(
            policy_name=name,
            key_bits=policy.get("key_bits", 8),
            value_bits=policy.get("value_bits", 4),
            b_min=policy.get("b_min", 2),
            b_min_value=policy.get("b_min_value", 2),
            sink_tokens=policy.get("sink_tokens", 4),
            recent_tokens=policy.get("recent_tokens", 64),
            ladder=None if policy.get("ladder") is None else tuple(policy["ladder"]),
            skip_layers=tuple(policy.get("skip_layers") or ()),
            block_size=policy.get("block_size"),
            residual_fp16=policy.get("residual_fp16", True),
            quant_granularity=str(policy.get("quant_granularity", "block")),
            hayate_config=hayate_config,
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TaperVLLMRecipe":
        if not isinstance(mapping, dict):
            raise ValueError("TapeKV native payload must be a JSON object")
        data = dict(mapping)
        for key in ("ladder", "skip_layers"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["ladder"] = list(self.ladder) if self.ladder is not None else None
        result["skip_layers"] = list(self.skip_layers)
        return result
