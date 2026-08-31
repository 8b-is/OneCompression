"""Validated standalone TaperKV configuration for vLLM.

The current backend applies the TaperKV schedule on GPU while retaining vLLM's
model-dtype page allocation. It is suitable for inference and quality
reproduction, but its physical memory must not be reported as packed TaperKV.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from vllm_plugins.kvcomp._validation import strict_integer

_SUPPORTED_POLICIES = {"taperkv", "block_taperkv"}
_IMPLEMENTATION = "gpu_semantic_model_dtype"
_COMMON_POLICY_FIELDS = {
    "name",
    "key_bits",
    "value_bits",
    "b_min",
    "b_min_value",
    "sink_tokens",
    "recent_tokens",
    "key_ladder",
    "value_ladder",
    "skip_layers",
    "requant_stride",
}
_BLOCK_POLICY_FIELDS = {"block_size", "residual_fp16", "quant_granularity"}


def _validate_ladder(
    values: tuple[int, ...] | list[int] | None,
    *,
    name: str,
    high: int,
    floor: int,
) -> tuple[int, ...]:
    if values is None:
        return tuple(range(high, floor - 1, -1))
    ladder = tuple(
        strict_integer(value, f"{name} entry", minimum=floor, maximum=high) for value in values
    )
    if not ladder:
        raise ValueError(f"{name} must not be empty")
    if ladder[0] != high or ladder[-1] != floor:
        raise ValueError(f"{name} must start at {high} and end at {floor}")
    if any(left <= right for left, right in zip(ladder, ladder[1:])):
        raise ValueError(f"{name} must be strictly descending")
    return ladder


@dataclass(frozen=True)
class TaperKVVLLMRecipe:
    """One validated TaperKV schedule accepted by the vLLM backend."""

    policy_name: str
    key_bits: int = 8
    value_bits: int = 4
    b_min: int = 4
    b_min_value: int = 2
    sink_tokens: int = 4
    recent_tokens: int = 128
    key_ladder: tuple[int, ...] | None = None
    value_ladder: tuple[int, ...] | None = None
    skip_layers: tuple[int, ...] = ()
    requant_stride: int = 16
    block_size: int | None = None
    residual_fp16: bool = True
    quant_granularity: str = "block"
    implementation: str = _IMPLEMENTATION

    def __post_init__(self) -> None:
        if self.policy_name not in _SUPPORTED_POLICIES:
            raise ValueError(f"unsupported TaperKV policy {self.policy_name!r}")
        for name in ("key_bits", "value_bits", "b_min", "b_min_value"):
            object.__setattr__(
                self,
                name,
                strict_integer(getattr(self, name), name, minimum=1, maximum=16),
            )
        if self.value_bits > self.key_bits:
            raise ValueError("value_bits must not exceed key_bits")
        if self.b_min > self.key_bits:
            raise ValueError("b_min must not exceed key_bits")
        if self.b_min_value > self.value_bits:
            raise ValueError("b_min_value must not exceed value_bits")
        for name in ("sink_tokens", "recent_tokens"):
            object.__setattr__(
                self,
                name,
                strict_integer(getattr(self, name), name, minimum=0),
            )
        key_ladder = _validate_ladder(
            self.key_ladder,
            name="key_ladder",
            high=self.key_bits,
            floor=self.b_min,
        )
        value_ladder = _validate_ladder(
            self.value_ladder,
            name="value_ladder",
            high=self.value_bits,
            floor=self.b_min_value,
        )
        object.__setattr__(self, "key_ladder", key_ladder)
        object.__setattr__(self, "value_ladder", value_ladder)

        boundaries = sorted(
            {0.0, 1.0}
            | {index / len(key_ladder) for index in range(1, len(key_ladder))}
            | {index / len(value_ladder) for index in range(1, len(value_ladder))}
        )
        probes = boundaries[:-1] + [
            (left + right) / 2.0 for left, right in zip(boundaries, boundaries[1:])
        ]
        for fraction in probes:
            key_width = key_ladder[min(int(fraction * len(key_ladder)), len(key_ladder) - 1)]
            value_width = value_ladder[
                min(int(fraction * len(value_ladder)), len(value_ladder) - 1)
            ]
            if key_width < value_width:
                raise ValueError("key_ladder must preserve key precision >= value precision")

        skip_layers = tuple(
            strict_integer(value, "skip_layers entry", minimum=0) for value in self.skip_layers
        )
        object.__setattr__(
            self,
            "skip_layers",
            tuple(dict.fromkeys(skip_layers)),
        )
        object.__setattr__(
            self,
            "requant_stride",
            strict_integer(self.requant_stride, "requant_stride", minimum=1),
        )
        if self.implementation != _IMPLEMENTATION:
            raise ValueError(f"implementation must be {_IMPLEMENTATION!r}")

        if self.policy_name == "block_taperkv":
            object.__setattr__(
                self,
                "block_size",
                strict_integer(self.block_size, "block_size", minimum=1),
            )
            if not isinstance(self.residual_fp16, bool):
                raise ValueError("residual_fp16 must be a bool")
            if self.quant_granularity not in {"block", "token"}:
                raise ValueError("quant_granularity must be 'block' or 'token'")
            if self.quant_granularity == "block" and not self.residual_fp16:
                raise ValueError(
                    "vLLM block TaperKV requires residual_fp16=True so a "
                    "partially written page is never repeatedly requantized"
                )
        else:
            if self.block_size is not None:
                raise ValueError("block_size is only valid for block_taperkv")
            object.__setattr__(self, "quant_granularity", "token")

    @classmethod
    def from_policy(cls, policy: dict[str, Any]) -> "TaperKVVLLMRecipe":
        if not isinstance(policy, dict):
            raise ValueError("TaperKV policy must be a mapping")
        name = str(policy.get("name", ""))
        allowed = _COMMON_POLICY_FIELDS | (
            _BLOCK_POLICY_FIELDS if name == "block_taperkv" else set()
        )
        unknown = sorted(set(policy) - allowed)
        if unknown:
            raise ValueError(f"non-canonical TaperKV policy fields: {unknown}")
        block_size = policy.get("block_size")
        requested_stride = policy.get("requant_stride")
        default_stride = block_size if name == "block_taperkv" else 16
        return cls(
            policy_name=name,
            key_bits=policy.get("key_bits", 8),
            value_bits=policy.get("value_bits", 4),
            b_min=policy.get("b_min", 4),
            b_min_value=policy.get("b_min_value", 2),
            sink_tokens=policy.get("sink_tokens", 4),
            recent_tokens=policy.get("recent_tokens", 128),
            key_ladder=(None if policy.get("key_ladder") is None else tuple(policy["key_ladder"])),
            value_ladder=(
                None if policy.get("value_ladder") is None else tuple(policy["value_ladder"])
            ),
            skip_layers=tuple(policy.get("skip_layers") or ()),
            requant_stride=(default_stride if requested_stride is None else requested_stride),
            block_size=block_size,
            residual_fp16=policy.get("residual_fp16", True),
            quant_granularity=str(policy.get("quant_granularity", "block")),
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TaperKVVLLMRecipe":
        if not isinstance(mapping, dict):
            raise ValueError("TaperKV payload must be a JSON object")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(f"non-canonical TaperKV payload fields: {unknown}")
        if "policy_name" not in mapping:
            raise ValueError("TaperKV payload requires policy_name")
        data = dict(mapping)
        for key in ("key_ladder", "value_ladder", "skip_layers"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["key_ladder"] = list(self.key_ladder)
        result["value_ladder"] = list(self.value_ladder)
        result["skip_layers"] = list(self.skip_layers)
        return result


__all__ = ["TaperKVVLLMRecipe"]
