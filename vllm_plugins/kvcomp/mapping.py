"""Map serialized OneComp KV compression recipes to vLLM.

The module is intentionally torch/vLLM-free.  It validates a serialized recipe,
selects a native carrier dtype, and records whether the OneComp plugin must
install a custom attention backend.  Unsupported or ambiguous recipes fail
closed instead of running under a misleading method label.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from vllm_plugins.kvcomp._validation import strict_integer
from vllm_plugins.kvcomp.native_turboquant import resolve_turboquant_codec

_NATIVE_FP8 = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}
_NATIVE_OTHER = {
    "nvfp4": "nvfp4",
    "int8": "int8_per_token_head",
}
_NEEDS_CUSTOM_BACKEND = {"int4", "int3", "int2", "kivi", "ternary"}
_HADAMARD_ROTATIONS = (None, "auto", "hadamard", "hadamard_gemm")


def _integer_field(
    config: dict,
    name: str,
    default: int | None = None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = config.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    return strict_integer(value, name, minimum=minimum, maximum=maximum)


def _integer_list(config: dict, name: str) -> list[int]:
    values = config.get(name) or []
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of integers")
    return [strict_integer(value, f"{name} entry", minimum=0) for value in values]


def _turboquant_native_requested(codec: dict) -> bool:
    raw = codec.get("vllm_mode") or os.environ.get("KVCOMP_VLLM_TURBOQUANT_MODE")
    if raw is None:
        return False
    mode = str(raw).strip().lower().replace("-", "_")
    return mode in {"native", "deployment", "vllm", "1", "true", "yes", "on"}


@dataclass
class KVCompVLLMPlan:
    """Validated vLLM execution plan for one serialized KV recipe."""

    kv_cache_dtype: str = "auto"
    native_supported: bool = False
    codec_name: Optional[str] = None
    calculate_kv_scales: bool = False
    needs_custom_backend: bool = False
    hayatequant_native: bool = False
    hayatequant_config: dict = field(default_factory=dict)
    taper_backend: bool = False
    taper_config: dict = field(default_factory=dict)
    skip_layers: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def engine_kwargs(self) -> dict:
        kwargs: dict = {"kv_cache_dtype": self.kv_cache_dtype}
        if self.kv_cache_dtype.startswith("fp8") and self.calculate_kv_scales:
            kwargs["calculate_kv_scales"] = True
        if self.native_supported and self.skip_layers:
            kwargs["kv_cache_dtype_skip_layers"] = ",".join(
                str(index) for index in self.skip_layers
            )
        return kwargs


def _plan_rolekv(
    policy: dict,
    plan: KVCompVLLMPlan,
    requested_kv_cache_dtype: str,
) -> KVCompVLLMPlan:
    bits_map = dict(policy.get("bits_per_role") or {})
    bits_map["<default>"] = _integer_field(policy, "default_bits", 4, minimum=1)
    bit_values = {
        strict_integer(value, "bits_per_role value", minimum=1) for value in bits_map.values()
    }
    sink_tokens = _integer_field(policy, "sink_tokens", 4, minimum=0)
    plan.skip_layers = _integer_list(policy, "skip_layers")
    if bit_values == {16}:
        plan.kv_cache_dtype = "auto"
        plan.native_supported = True
        plan.notes.append("RoleKV assigns model precision to every role.")
        return plan
    plan.needs_custom_backend = True
    plan.notes.append(
        "RoleKV requires the HF/custom path because vLLM cannot express its "
        f"per-role affine bit widths {sorted(bit_values)} with "
        f"sink_tokens={sink_tokens}; a uniform native dtype would change both "
        "the quantizer and realised memory rate."
    )
    if requested_kv_cache_dtype != "auto":
        plan.warnings.append(
            f"kv_cache_dtype={requested_kv_cache_dtype!r} cannot reproduce "
            "RoleKV and is ignored."
        )
    return plan


def _hayatequant_plan(
    codec: dict,
    plan: KVCompVLLMPlan,
    requested_kv_cache_dtype: str,
) -> KVCompVLLMPlan:
    from vllm_plugins.kvcomp.hayate_config import hayatequant_native_recipe

    allowed = {"name", "bits", "skip_layers", "calibration"}
    unknown = sorted(set(codec) - allowed)
    try:
        if unknown:
            raise ValueError(f"obsolete/non-canonical fields are not accepted: {unknown}")
        if codec.get("calibration"):
            raise ValueError("the packed vLLM backend does not load calibration artifacts")
        rate = _integer_field(codec, "bits", minimum=1)
        recipe = hayatequant_native_recipe(rate)
    except (TypeError, ValueError) as exc:
        plan.needs_custom_backend = True
        plan.notes.append(
            "Serialized HayateQuant is not one of the canonical 2/3/4-bit "
            f"recipes accepted by the packed backend: {exc}"
        )
        return plan

    native_config = {"rate": rate}

    # vLLM validates a built-in dtype before dispatching the plugin backend.
    # The carrier controls no HayateQuant bytes: the cache-spec hook replaces
    # its slot width and the backend replaces store/attention completely.
    carrier = "turboquant_4bit_nc" if rate == 4 else "turboquant_3bit_nc"
    plan.kv_cache_dtype = carrier
    plan.native_supported = True
    plan.hayatequant_native = True
    plan.hayatequant_config = native_config
    plan.skip_layers = _integer_list(codec, "skip_layers")
    plan.notes.append(
        f"HayateQuant rate {rate} uses its packed K/V vLLM backend "
        f"(carrier dtype {carrier!r})."
    )
    if requested_kv_cache_dtype not in ("auto", carrier):
        plan.warnings.append(
            f"kv_cache_dtype={requested_kv_cache_dtype!r} conflicts with "
            f"HayateQuant's carrier {carrier!r}; the carrier is used."
        )
    return plan


def _check_turboquant_semantics(codec: dict, plan: KVCompVLLMPlan) -> None:
    mode = codec.get("rotation_mode")
    if mode not in _HADAMARD_ROTATIONS:
        plan.warnings.append(
            f"TurboQuant rotation_mode={mode!r} differs from vLLM's randomized " "Hadamard kernel."
        )
    if codec.get("calibration"):
        plan.warnings.append("vLLM TurboQuant cannot apply a learned calibration.")
    if _integer_field(codec, "n_outlier", 0, minimum=0):
        plan.warnings.append("vLLM TurboQuant drops the requested outlier split.")
    if codec.get("use_qjl"):
        plan.warnings.append("vLLM TurboQuant does not implement the QJL residual.")


def _plan_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> KVCompVLLMPlan:
    plan = KVCompVLLMPlan()
    if not kv_config:
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        plan.notes.append("No KV compression recipe; cache uses the requested dtype.")
        return plan
    if not isinstance(kv_config, dict):
        raise TypeError("kv_compression_config must be a mapping")
    unknown_top_level = sorted(set(kv_config) - {"codec", "selector", "policy", "residual_window"})
    if unknown_top_level:
        raise ValueError(f"non-canonical kv_compression_config fields: {unknown_top_level}")
    for section in ("codec", "selector", "policy"):
        value = kv_config.get(section)
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"kv_compression_config.{section} must be a mapping")

    codec = kv_config.get("codec") or None
    selector = kv_config.get("selector") or None
    policy = kv_config.get("policy") or None
    policy_name = policy.get("name") if policy else None
    plan.codec_name = codec.get("name") if codec else None

    if codec is not None:
        for name in (
            "bits",
            "value_bits",
            "block_size",
            "max_codebook_bits",
            "seed",
            "n_outlier",
            "key_n_outlier",
            "value_n_outlier",
            "key_code_bits_per_block",
            "value_code_bits_per_block",
            "lut_bins",
            "lut_candidates",
            "residual_window",
        ):
            if codec.get(name) is not None:
                strict_integer(codec[name], f"codec.{name}")
        _integer_list(codec, "skip_layers")

    if selector is not None:
        plan.needs_custom_backend = True
        plan.notes.append(f"Selector {selector.get('name')!r} requires token-aware attention.")

    # TaperKV is a distinct proposal.  Never compose it with a codec.
    if policy_name in {"taperkv", "block_taperkv"}:
        if codec is not None or selector is not None:
            plan.needs_custom_backend = True
            plan.notes.append(
                "TaperKV must be standalone; codec/selector composition is rejected."
            )
            return plan
        try:
            taper_residual = strict_integer(
                kv_config.get("residual_window", 0),
                "kv_compression_config.residual_window",
                minimum=0,
            )
            if taper_residual != 0:
                raise ValueError(
                    "standalone TaperKV requires residual_window=0; recency is "
                    "defined only by policy.recent_tokens"
                )
            from vllm_plugins.kvcomp.taper_config import TaperKVVLLMRecipe

            recipe = TaperKVVLLMRecipe.from_policy(policy)
        except (TypeError, ValueError) as exc:
            plan.needs_custom_backend = True
            plan.notes.append(f"Invalid TaperKV policy: {exc}")
            return plan
        plan.kv_cache_dtype = "auto"
        plan.native_supported = True
        plan.taper_backend = True
        plan.taper_config = recipe.to_mapping()
        plan.skip_layers = list(recipe.skip_layers)
        plan.notes.append(
            "TaperKV uses its GPU semantic backend; physical pages remain in "
            "model dtype until a variable-width vLLM allocator is available."
        )
        if requested_kv_cache_dtype != "auto":
            plan.warnings.append(
                f"kv_cache_dtype={requested_kv_cache_dtype!r} is ignored for TaperKV."
            )
        return plan

    if policy_name == "rolekv" and not plan.needs_custom_backend:
        role_plan = _plan_rolekv(policy, plan, requested_kv_cache_dtype)
        role_residual = strict_integer(
            kv_config.get("residual_window", 0),
            "kv_compression_config.residual_window",
            minimum=0,
        )
        if role_residual > 0:
            role_plan.notes.append(
                f"residual_window={role_residual} also requires the custom " "RoleKV cache path."
            )
        return role_plan
    if policy_name not in (None, "static", "rolekv"):
        plan.needs_custom_backend = True
        plan.notes.append(f"Policy {policy_name!r} requires a token-aware custom backend.")

    residual = strict_integer(
        kv_config.get("residual_window", 128),
        "kv_compression_config.residual_window",
        minimum=0,
    )
    if residual > 0 and codec is not None:
        plan.needs_custom_backend = True
        plan.notes.append(
            f"residual_window={residual} cannot be represented by a uniform "
            "native paged cache; set it to 0 or use the HF cache."
        )

    if plan.needs_custom_backend:
        return plan
    if codec is None:
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        return plan

    name = plan.codec_name

    def set_native(dtype: str, note: str) -> None:
        plan.kv_cache_dtype = dtype
        plan.native_supported = True
        plan.notes.append(note)
        if requested_kv_cache_dtype not in ("auto", dtype):
            plan.warnings.append(
                f"kv_cache_dtype={requested_kv_cache_dtype!r} conflicts with "
                f"codec {name!r}; {dtype!r} is used."
            )

    if name in _NATIVE_FP8:
        dtype = _NATIVE_FP8[name]
        plan.calculate_kv_scales = True
        plan.skip_layers = _integer_list(codec, "skip_layers")
        set_native(dtype, f"Codec {name!r} uses native vLLM dtype {dtype!r}.")
    elif name == "hayatequant":
        return _hayatequant_plan(codec, plan, requested_kv_cache_dtype)
    elif name == "turboquant":
        if not _turboquant_native_requested(codec):
            plan.needs_custom_backend = True
            plan.notes.append(
                "OneComp TurboQuant and vLLM's deployment preset are not "
                "bit-identical. Set KVCOMP_VLLM_TURBOQUANT_MODE=native to "
                "select the vLLM implementation explicitly."
            )
        else:
            preset = resolve_turboquant_codec(codec)
            if preset is None:
                plan.needs_custom_backend = True
                plan.notes.append("This key/value width has no native vLLM TurboQuant preset.")
            else:
                plan.skip_layers = _integer_list(codec, "skip_layers")
                set_native(
                    preset.kv_cache_dtype,
                    f"TurboQuant k{preset.key_bits}v{preset.value_bits} uses "
                    f"native vLLM dtype {preset.kv_cache_dtype!r}.",
                )
                _check_turboquant_semantics(codec, plan)
    elif name in _NATIVE_OTHER:
        dtype = _NATIVE_OTHER[name]
        plan.skip_layers = _integer_list(codec, "skip_layers")
        set_native(dtype, f"Codec {name!r} uses native vLLM dtype {dtype!r}.")
    elif name in _NEEDS_CUSTOM_BACKEND:
        plan.needs_custom_backend = True
        plan.notes.append(f"Codec {name!r} requires the HF/custom backend.")
    else:
        plan.needs_custom_backend = True
        plan.warnings.append(f"Unknown KV codec {name!r}; refusing fallback.")
    return plan


def plan_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> KVCompVLLMPlan:
    """Resolve a plan and fail closed on malformed serialized configuration."""

    try:
        return _plan_from_kv_config(kv_config, requested_kv_cache_dtype)
    except (TypeError, ValueError) as exc:
        plan = KVCompVLLMPlan(needs_custom_backend=True)
        if isinstance(kv_config, dict) and isinstance(kv_config.get("codec"), dict):
            plan.codec_name = kv_config["codec"].get("name")
        plan.notes.append(
            "Invalid serialized kv_compression_config; native mapping is " f"disabled: {exc}"
        )
        return plan


def engine_kwargs_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> dict:
    """Return bare vLLM kwargs only when no OneComp backend installation is needed."""

    plan = plan_from_kv_config(kv_config, requested_kv_cache_dtype)
    if plan.needs_custom_backend or not plan.native_supported:
        detail = "; ".join(plan.notes + plan.warnings) or "no faithful native mapping"
        raise RuntimeError(f"KV recipe cannot run through native vLLM kwargs: {detail}")
    if plan.taper_backend or plan.hayatequant_native:
        raise RuntimeError(
            "This KV recipe requires the onecomp_kvcomp plugin's attention backend."
        )
    return plan.engine_kwargs()
