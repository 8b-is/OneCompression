"""Validated configuration for the vLLM TaperKV reference backend.

The backend deliberately keeps vLLM's ordinary FP16/BF16 paged allocation and
applies TaperKV's affine fake-quantisation before attention.  It is therefore a
*semantic/reference* implementation: it is suitable for generation and
quality checks, but must never be reported as a packed-memory or throughput
result.  Keeping that distinction in one small, dependency-free module makes
it hard for launchers to accidentally make a physical-compression claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vllm_plugins.kvcomp._validation import strict_integer

_SUPPORTED_POLICIES = {"taperkv", "block_taperkv"}


@dataclass(frozen=True)
class TaperKVVLLMRecipe:
    """One explicit TaperKV semantic recipe accepted by the vLLM plugin."""

    policy_name: str
    key_bits: int
    value_bits: int
    b_min: int
    b_min_value: int
    sink_tokens: int = 4
    recent_tokens: int = 64
    ladder: tuple[int, ...] | None = None
    skip_layers: tuple[int, ...] = ()
    requant_stride: int = 16
    block_size: int | None = None
    residual_fp16: bool = True
    quant_granularity: str = "block"
    implementation: str = "fp16_reference_fake_quant"
    hayate_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.policy_name not in _SUPPORTED_POLICIES:
            raise ValueError(f"unsupported TaperKV policy {self.policy_name!r}")
        for name in ("key_bits", "value_bits", "b_min", "b_min_value"):
            value = strict_integer(getattr(self, name), name, minimum=1, maximum=16)
            object.__setattr__(self, name, value)
        if self.value_bits > self.key_bits:
            raise ValueError("value_bits must not exceed key_bits")
        if self.b_min > self.key_bits or self.b_min_value > self.value_bits:
            raise ValueError("TaperKV floor bits must not exceed their high bits")
        for name in ("sink_tokens", "recent_tokens"):
            object.__setattr__(self, name, strict_integer(getattr(self, name), name, minimum=0))
        ladder = self.ladder
        if ladder is not None:
            ladder = tuple(
                strict_integer(value, "ladder entry", minimum=1, maximum=self.key_bits)
                for value in ladder
            )
            if not ladder or any(left <= right for left, right in zip(ladder, ladder[1:])):
                raise ValueError("ladder must be a non-empty strictly descending sequence")
            object.__setattr__(self, "ladder", ladder)
        skip = tuple(
            strict_integer(value, "skip_layers entry", minimum=0) for value in self.skip_layers
        )
        object.__setattr__(self, "skip_layers", tuple(dict.fromkeys(skip)))
        object.__setattr__(
            self,
            "requant_stride",
            strict_integer(self.requant_stride, "requant_stride", minimum=1),
        )
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
        if self.hayate_config is not None:
            if (
                not isinstance(self.hayate_config, dict)
                or self.hayate_config.get("name") != "hayatequant"
            ):
                raise ValueError("hayate_config must be a serialized HayateQuant codec mapping")
            if self.hayate_config.get("calibration"):
                raise ValueError(
                    "calibrated HayateQuant cannot be reconstructed from the serialized "
                    "calibration digest in a vLLM worker"
                )
            if self.hayate_config.get("adaptive_bit_split") not in (
                None,
                "",
                "off",
                "none",
                False,
            ):
                raise ValueError(
                    "adaptive HayateQuant is request-local and cannot be composed "
                    "with batched vLLM TaperKV"
                )
            outlier_fields = (
                "n_outlier",
                "key_n_outlier",
                "value_n_outlier",
            )
            outlier_counts = []
            for name in outlier_fields:
                raw_value = self.hayate_config.get(name, 0)
                outlier_counts.append(
                    strict_integer(0 if raw_value is None else raw_value, name, minimum=0)
                )
            if any(value > 0 for value in outlier_counts):
                raise ValueError(
                    "dynamic HayateQuant outlier selection cannot be composed with "
                    "batched vLLM TaperKV"
                )
            if str(self.hayate_config.get("encoder") or "auto").lower() not in {
                "exact",
                "lut",
            }:
                raise ValueError("HayateQuant+TaperKV requires an explicit exact or lut encoder")
            object.__setattr__(self, "hayate_config", dict(self.hayate_config))
            object.__setattr__(self, "implementation", "hayatequant_then_taper_fp16_reference")
        elif self.implementation != "fp16_reference_fake_quant":
            raise ValueError("only the explicit FP16 reference TaperKV backend is supported")

    @classmethod
    def from_policy(
        cls, policy: dict[str, Any], hayate_config: dict[str, Any] | None = None
    ) -> "TaperKVVLLMRecipe":
        """Build a validated serving recipe from serialized OneComp policies."""

        if not isinstance(policy, dict):
            raise ValueError("TaperKV policy must be a mapping")
        name = str(policy.get("name", ""))
        block_size = policy.get("block_size")
        default_stride = block_size if name == "block_taperkv" else 16
        requested_stride = policy.get("requant_stride")
        policy_skip = tuple(policy.get("skip_layers") or ())
        codec_skip = tuple((hayate_config or {}).get("skip_layers") or ())
        return cls(
            policy_name=name,
            key_bits=policy.get("key_bits", 8),
            value_bits=policy.get("value_bits", 4),
            b_min=policy.get("b_min", 2),
            b_min_value=policy.get("b_min_value", 2),
            sink_tokens=policy.get("sink_tokens", 4),
            recent_tokens=policy.get("recent_tokens", 64),
            ladder=None if policy.get("ladder") is None else tuple(policy["ladder"]),
            skip_layers=tuple(dict.fromkeys((*policy_skip, *codec_skip))),
            requant_stride=(default_stride if requested_stride is None else requested_stride),
            block_size=block_size,
            residual_fp16=policy.get("residual_fp16", True),
            quant_granularity=str(policy.get("quant_granularity", "block")),
            hayate_config=hayate_config,
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "TaperKVVLLMRecipe":
        """Reconstruct and validate a recipe from a JSON-compatible mapping."""

        if not isinstance(mapping, dict):
            raise ValueError("TaperKV reference payload must be a JSON object")
        data = dict(mapping)
        for key in ("ladder", "skip_layers"):
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping for the worker environment."""

        result = asdict(self)
        result["ladder"] = list(self.ladder) if self.ladder is not None else None
        result["skip_layers"] = list(self.skip_layers)
        return result


# Compatibility for callers written before the reference/native naming audit.
# New code should use ``TaperKVVLLMRecipe`` so the method name and the ``KV``
# component remain explicit.
TaperVLLMRecipe = TaperKVVLLMRecipe

__all__ = ["TaperKVVLLMRecipe", "TaperVLLMRecipe"]
