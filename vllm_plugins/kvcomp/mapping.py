"""Map an OneComp ``kv_compression_config`` to vLLM engine arguments.

This module is the *pure-Python* bridge between
:class:`onecomp.kvcomp.KVCompressionConfig` (written into ``config.json`` as the
``kv_compression_config`` block by :class:`onecomp.runner.Runner`) and the
arguments that configure vLLM's KV cache.

Design (see ``docs/design/kvcomp-architecture.md`` section 11).  Recent vLLM
ships native KV-cache dtypes well beyond FP8, so the bridge maps onto them:

* **FP8** -> ``"fp8_e4m3"`` / ``"fp8_e5m2"`` (data-free, ``calculate_kv_scales``).
* **TurboQuant** (and the TurboQuant-family **HALO**, **HayateQuant** at
  ``block_size=1``) -> the native ``turboquant_*`` presets
  (``turboquant_k8v4``, ``turboquant_4bit_nc``, ``turboquant_k3v4_nc``,
  ``turboquant_3bit_nc``) chosen by ``(key_bits, value_bits)`` -- vLLM runs
  Hadamard rotation + Lloyd-Max keys + uniform values with automatic
  boundary-layer protection (FlashAttention-2).  For HayateQuant block-VQ
  (``block_size>1``), the bridge keeps the byte-compatible ``turboquant_*``
  dtype but registers OneComp's native vLLM backend override for the supported
  3/4-bit ``block_size=2`` key block-VQ path.  Set
  ``KVCOMP_VLLM_HAYATEQUANT_MODE=scalar_fallback`` to force the older scalar
  TurboQuant fallback, or ``strict`` to require the HF/custom path.
* **NVFP4** -> ``"nvfp4"``; **INT8** -> ``"int8_per_token_head"``.
* Remaining low-bit/asymmetric codecs (INT4/INT3/INT2, KIVI, Ternary) and
  token-eviction selectors (SnapKV, ThinKV TBE) / adaptive per-token policies
  (TaperKV, ThinKV) are not expressible as one ``kv_cache_dtype`` and require the
  custom attention-backend / HF cache path.  For those the module returns
  ``native_supported=False`` with an explicit reason rather than silently
  downgrading.  Native dtypes added only in recent vLLM are validated at
  apply-time so an older vLLM falls back gracefully.

The module deliberately imports neither ``vllm`` nor ``torch`` so it can be
unit-tested on a CPU-only login node.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from vllm_plugins.kvcomp.native_turboquant import resolve_turboquant_codec

# OneComp codec name -> vLLM native ``kv_cache_dtype`` value.
_NATIVE_FP8 = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "fp8_e5m2": "fp8_e5m2",
}

# Other native (non-FP8) KV cache dtypes vLLM accepts.
_NATIVE_OTHER = {
    "nvfp4": "nvfp4",
    "int8": "int8_per_token_head",
}

# Codecs implemented in OneComp with no native vLLM kv_cache_dtype; they require
# the (experimental) custom attention-backend path to run *inside* vLLM, but work
# in the HF cache path today.
_NEEDS_CUSTOM_BACKEND = {
    "int4",
    "int3",
    "int2",
    "kivi",
    "ternary",
}


# Rotation modes whose transform family matches vLLM's native TurboQuant
# kernel (a randomized *Hadamard* rotation).  ``hadamard_gemm`` is the same
# signed-Hadamard transform merely folded into one GEMM;  ``auto`` resolves to
# a Hadamard variant for every power-of-two head dim (the norm for supported
# models) when ``block_size==1``, so it is treated as Hadamard-compatible.
_HADAMARD_COMPAT_ROTATIONS = (None, "auto", "hadamard", "hadamard_gemm")

_HAYATEQUANT_NATIVE_MODES = {
    "native",
    "block_vq",
    "hayate",
    "hayate_native",
}
_HAYATEQUANT_SCALAR_FALLBACK_MODES = {
    "scalar",
    "scalar_fallback",
    "fallback",
    "turboquant",
    "native_fallback",
}
_HAYATEQUANT_STRICT_MODES = {
    "strict",
    "custom",
    "custom_backend",
    "hf",
    "hf_cache",
    "faithful",
}


def _check_rotation_for_native(codec: dict, name: str, plan: "KVCompVLLMPlan"):
    """Flag rotation-semantics mismatches for the native ``turboquant_*`` path.

    vLLM's native TurboQuant KV cache always applies a randomized **Hadamard**
    rotation inside its kernel; the preset has no knob to swap it for a dense
    (QR random-orthogonal) or hybrid rotation.  OneComp codecs configured with
    ``rotation_mode='dense'`` (recommended for Qwen-family / used by the
    dense-LS variants) or ``'hybrid'`` therefore cannot be reproduced by the
    native path -- quantizing under a different rotation gives different codes
    *and* different accuracy.  Warn explicitly instead of silently swapping
    the rotation (the whole reason dense was chosen would be lost).
    """
    if codec.get("calibration"):
        plan.warnings.append(
            f"Codec '{name}' has a SpinQuant-style calibration loaded "
            f"(digest {codec['calibration']}); vLLM's native TurboQuant "
            "kernel cannot apply a learned rotation/codebook -- the native "
            "preset would quantize under the data-free transform instead "
            "(the calibration gain would be lost). Use the HF cache path "
            "for faithful calibrated results.")
        return
    mode = codec.get("rotation_mode")
    block_size = codec.get("block_size")
    if mode == "hybrid" and block_size is not None and int(block_size) <= 1:
        # Hybrid at block_size=1 degenerates to signed-Hadamard with extra
        # per-coordinate sign flips -- still the Hadamard transform family.
        plan.notes.append(
            f"Codec '{name}' rotation_mode='hybrid' with block_size=1 "
            "degenerates to a signed-Hadamard rotation (1x1 block rotations "
            "are sign flips), matching vLLM's native Hadamard kernel."
        )
        return
    if mode in _HADAMARD_COMPAT_ROTATIONS:
        if mode in ("auto", None):
            plan.notes.append(
                f"Codec '{name}' rotation_mode='auto' resolves to the "
                "(signed-)Hadamard transform for power-of-two head dims, "
                "matching vLLM's native Hadamard rotation kernel (non-power-"
                "of-two head dims resolve to dense -- check the model's "
                "head_dim if it is unusual)."
            )
        else:
            plan.notes.append(
                f"Codec '{name}' rotation_mode='{mode}' matches vLLM's native "
                "Hadamard rotation kernel (hadamard_gemm is the same transform "
                "folded into one GEMM)."
            )
        return
    plan.warnings.append(
        f"Codec '{name}' rotation_mode='{mode}' does NOT match vLLM's native "
        "TurboQuant kernel, which always applies a randomized Hadamard "
        "rotation; the native preset would quantize under a different "
        "transform (different codes, different accuracy -- e.g. the "
        "dense-rotation PPL advantage on Qwen-family models would be lost). "
        "Use the HF cache path for faithful results with this rotation."
    )


def _hayatequant_uses_block_vq(codec: dict) -> bool:
    """Best-effort, torch-free check: will HayateQuant's ``block_size`` resolve
    to ``>1`` (block-VQ) for this codec config?

    Mirrors the feasibility half of
    :func:`onecomp.kvcomp.codec._vq.resolve_block_size` (``bits * block_size
    <= max_codebook_bits``) without needing the actual head dim (not
    available at this layer, and this module intentionally stays torch-free
    -- see module docstring) -- so this can only rule out block-VQ when even
    ``block_size=2`` provably cannot fit the codebook budget, and is
    otherwise conservative (assumes the codec's default "auto" behaviour,
    which is to use the largest block size that fits).

    ``block_size`` is ``None`` (not absent) in a real ``get_kv_config()``
    dict when the codec was built with its "auto" default -- ``codec.get(
    "block_size", 1)`` would silently return ``None`` in that case (the key
    exists), so this helper must handle ``None`` explicitly rather than
    relying on ``dict.get``'s default.
    """
    block_size = codec.get("block_size")
    if block_size is not None and int(block_size) <= 1:
        return False
    bits = int(codec.get("bits", 4))
    value_bits = codec.get("value_bits")
    value_bits = int(value_bits) if value_bits is not None else bits
    max_codebook_bits = int(codec.get("max_codebook_bits") or 9)
    # Keys and values resolve block_size independently (at possibly different
    # bit-widths); block-VQ is "in play" if *either* role could use it.
    return min(bits, value_bits) * 2 <= max_codebook_bits


def _hayatequant_vllm_mode(codec: dict) -> str:
    """Resolve how block-VQ HayateQuant should be represented in vLLM.

    Upstream vLLM currently exposes TurboQuant scalar presets as dtype strings,
    not HayateQuant-specific dtype strings.  For operational usability, the
    default is OneComp's byte-compatible native backend override for supported
    HayateQuant key block-VQ presets.  Scalar fallback and strict/custom modes
    remain explicit escape hatches.
    ``KVCOMP_VLLM_SCALAR_FALLBACK=0`` is kept as a backward-compatible way to
    force strict behavior from older scripts.
    """
    raw = codec.get("vllm_mode")
    if raw is None:
        raw = codec.get("vllm_backend")
    if raw is None:
        raw = codec.get("vllm_fallback")
    if raw is None:
        raw = os.environ.get("KVCOMP_VLLM_HAYATEQUANT_MODE")
    if raw is None:
        legacy = os.environ.get("KVCOMP_VLLM_SCALAR_FALLBACK")
        if legacy is not None:
            raw = "scalar_fallback" if legacy.strip().lower() in ("1", "true", "yes", "on") else "strict"
    if raw is None:
        return "native"
    mode = str(raw).strip().lower().replace("-", "_")
    if mode in ("1", "true", "yes", "on"):
        return "native"
    if mode in ("0", "false", "no", "off"):
        return "strict"
    return mode


def _hayatequant_native_key_scale_mode(codec: dict) -> str | None:
    raw = codec.get("key_scale_mode")
    if raw is None:
        return "nc"
    mode = str(raw).strip().lower().replace("-", "_")
    if mode == "ub":
        return "ip"
    if mode in {"nc", "ip"}:
        return mode
    return None


@dataclass
class KVCompVLLMPlan:
    """Resolved vLLM KV-cache configuration derived from an OneComp config.

    Attributes:
        kv_cache_dtype: Value to pass to vLLM ``LLM(kv_cache_dtype=...)`` /
            ``EngineArgs``.  ``"auto"`` keeps the model dtype (no compression).
        native_supported: Whether the codec maps onto a native vLLM
            ``kv_cache_dtype`` (FP8).  ``False`` means a custom attention
            backend is required and ``kv_cache_dtype`` was left at ``"auto"``.
        codec_name: The OneComp codec name (``None`` for eviction-only recipes).
        calculate_kv_scales: Whether vLLM should compute FP8 scales online
            (mirrors OneComp's data-free, tuning-free FP8 path).
        needs_custom_backend: ``True`` when the recipe needs the experimental
            platform/attention-backend path to run inside vLLM.
        notes: Human-readable explanation of the mapping decision.
        warnings: Non-fatal issues (e.g. ``kv_cache_dtype`` conflict).
    """

    kv_cache_dtype: str = "auto"
    native_supported: bool = False
    codec_name: Optional[str] = None
    calculate_kv_scales: bool = False
    needs_custom_backend: bool = False
    # True when the plugin should override vLLM's TURBOQUANT backend with the
    # OneComp HayateQuant key block-VQ implementation while retaining the same
    # byte-compatible ``turboquant_*`` cache dtype.
    hayate_native: bool = False
    hayate_config: dict = field(default_factory=dict)
    # Boundary / outlier layers vLLM should keep in full precision; mapped onto
    # ``--kv-cache-dtype-skip-layers`` so the codec's ``skip_layers`` are honoured
    # instead of silently dropped.
    skip_layers: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def engine_kwargs(self) -> dict:
        """Return the subset of ``LLM`` / ``EngineArgs`` kwargs to apply."""
        kwargs: dict = {"kv_cache_dtype": self.kv_cache_dtype}
        if self.kv_cache_dtype.startswith("fp8") and self.calculate_kv_scales:
            kwargs["calculate_kv_scales"] = True
        if self.native_supported and self.skip_layers:
            # vLLM accepts a comma-separated list of layer indices to leave at the
            # model dtype (boundary-layer protection).
            kwargs["kv_cache_dtype_skip_layers"] = ",".join(str(i) for i in self.skip_layers)
        return kwargs


def _plan_rolekv(
    policy: dict,
    plan: "KVCompVLLMPlan",
    requested_kv_cache_dtype: str,
) -> "KVCompVLLMPlan":
    """Map a role-aware precision policy ("rolekv") onto vLLM engine kwargs.

    vLLM applies one ``kv_cache_dtype`` uniformly to every token of every
    sequence; per-role bit-widths are not expressible.  The mapping is
    safe-side: the most precise role decides the dtype (8-bit -> native FP8
    with online scales; >= 16 -> ``"auto"``), so no span is stored at lower
    fidelity than the policy requested.  Role-aware compression below 8 bit
    is only realised in the HF cache path (:class:`onecomp.kvcomp.cache.
    role_cache.RoleKVCache` live, or ``.okv`` checkpoints offline).
    """
    bits_map = dict(policy.get("bits_per_role") or {})
    bits_map["<default>"] = int(policy.get("default_bits", 4))
    max_bits = max(int(b) for b in bits_map.values())

    if max_bits >= 16:
        plan.kv_cache_dtype = "auto"
        plan.native_supported = True
        plan.notes.append(
            "Policy 'rolekv': most precise role is uncompressed (>= 16 bit); "
            "kv_cache_dtype='auto' keeps the model dtype."
        )
    else:
        dtype = _NATIVE_FP8["fp8"]
        plan.kv_cache_dtype = dtype
        plan.native_supported = True
        plan.calculate_kv_scales = True
        plan.notes.append(
            f"Policy 'rolekv': most precise role bits={max_bits} -> native vLLM "
            f"kv_cache_dtype='{dtype}' with online scale calculation (data-free, "
            "mirrors the codec's tuning-free path)."
        )
        if requested_kv_cache_dtype not in ("auto", dtype):
            plan.warnings.append(
                f"Requested kv_cache_dtype='{requested_kv_cache_dtype}' conflicts "
                f"with policy 'rolekv' -> '{dtype}'; using the policy's dtype."
            )
        if max_bits < 8:
            plan.warnings.append(
                f"Policy 'rolekv': most precise role bits={max_bits} is below vLLM's "
                "lowest native KV cache dtype (fp8 = 8 bit); fp8 is a safe "
                "higher-precision stand-in, so fidelity is preserved but the "
                "expected memory saving below 8 bit is not realised inside vLLM."
            )
    below = sorted(role for role, bits in bits_map.items() if int(bits) < max_bits)
    if below and plan.kv_cache_dtype != "auto":
        plan.warnings.append(
            "vLLM applies one kv_cache_dtype to every token; the per-role "
            f"bit-widths of roles {below} are collapsed to "
            f"'{plan.kv_cache_dtype}'. Role-aware compression below the deciding "
            "bit-width is only realised in the HF cache path (RoleKVCache / "
            ".okv checkpoints)."
        )
    plan.skip_layers = [int(x) for x in (policy.get("skip_layers") or [])]
    return plan


def plan_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> KVCompVLLMPlan:
    """Resolve a :class:`KVCompVLLMPlan` from a ``kv_compression_config`` dict.

    Args:
        kv_config: The ``kv_compression_config`` block (``KVCompressionConfig.
            to_dict()``), or ``None`` (no compression).
        requested_kv_cache_dtype: A ``kv_cache_dtype`` the caller already set
            (e.g. on the command line).  Used for conflict detection.

    Returns:
        A :class:`KVCompVLLMPlan`.  Never raises on an unsupported codec -- the
        caller inspects ``native_supported`` / ``needs_custom_backend``.
    """
    plan = KVCompVLLMPlan()
    if not kv_config:
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        plan.notes.append("No kv_compression_config; KV cache left uncompressed.")
        return plan

    codec = kv_config.get("codec") or None
    selector = kv_config.get("selector") or None
    policy = kv_config.get("policy") or None
    plan.codec_name = codec.get("name") if codec else None

    if selector is not None:
        plan.needs_custom_backend = True
        plan.notes.append(
            f"Selector '{selector.get('name')}' (token eviction) needs the custom "
            "attention backend; native kv_cache_dtype cannot evict tokens."
        )

    # Per-token adaptive recipes (ThinKV graded eviction, TaperKV graded
    # quantization) are not a single kv_cache_dtype and need the custom backend.
    policy_name = policy.get("name") if policy else None
    if policy_name == "rolekv" and not plan.needs_custom_backend:
        # Role-aware precision (RolePrecisionPolicy) collapses safe-side: the
        # most precise role decides the dtype, so no span is served below the
        # fidelity the policy requested.  Unlike age/thought-graded policies
        # this is a pure fidelity *upgrade* for the low-bit roles (their spans
        # were only ever quantized harder, never softer), so a compressed
        # native dtype is still honest -- the caveat is the unrealised extra
        # compression, which is warned about explicitly.
        return _plan_rolekv(policy, plan, requested_kv_cache_dtype)
    if policy_name not in (None, "static", "rolekv"):
        plan.needs_custom_backend = True
        plan.notes.append(
            f"Policy '{policy_name}' applies per-token adaptive precision; it needs "
            "the custom attention backend (use the HF cache path today)."
        )

    # If the recipe already needs the custom backend (token eviction or an
    # adaptive per-token policy), no single ``kv_cache_dtype`` can express it --
    # the codec's native dtype is irrelevant.  Finalise as custom-backend.
    if plan.needs_custom_backend:
        plan.native_supported = False
        plan.kv_cache_dtype = "auto"
        if codec is not None:
            plan.notes.append(
                f"Codec '{plan.codec_name}' is part of an eviction/adaptive recipe; "
                "the whole recipe needs the custom attention backend (HF cache path)."
            )
        return plan

    if codec is None:
        # Eviction-only recipe handled above; here: nothing to compress.
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.native_supported = True
        return plan

    name = plan.codec_name

    def _set_native(dtype: str, note: str):
        plan.kv_cache_dtype = dtype
        plan.native_supported = True
        plan.notes.append(note)
        if requested_kv_cache_dtype not in ("auto", dtype):
            plan.warnings.append(
                f"Requested kv_cache_dtype='{requested_kv_cache_dtype}' conflicts "
                f"with codec '{name}' -> '{dtype}'; using the codec's dtype."
            )

    if name in _NATIVE_FP8:
        dtype = _NATIVE_FP8[name]
        # OneComp FP8 is data-free / online -> let vLLM compute scales online.
        plan.calculate_kv_scales = True
        _set_native(dtype, f"Codec '{name}' -> native vLLM kv_cache_dtype='{dtype}'.")
        # Propagate outlier/boundary layer protection for FP8 as well.
        plan.skip_layers = [int(x) for x in (codec.get("skip_layers") or [])]
    elif name == "hayatequant" and _hayatequant_uses_block_vq(codec):
        mode = _hayatequant_vllm_mode(codec)
        tq_preset = resolve_turboquant_codec(codec)
        preset = tq_preset.kv_cache_dtype if tq_preset is not None else None
        kb = tq_preset.key_bits if tq_preset is not None else int(codec.get("bits", 4))
        raw_vb = codec.get("value_bits")
        vb = (
            tq_preset.value_bits
            if tq_preset is not None
            else int(raw_vb) if raw_vb is not None else kb
        )
        unsupported_native = []
        if codec.get("calibration"):
            unsupported_native.append("calibration")
        if int(codec.get("n_outlier") or 0) > 0:
            unsupported_native.append("n_outlier")
        if codec.get("use_qjl"):
            unsupported_native.append("use_qjl")
        if int(codec.get("max_codebook_bits") or 9) > 9:
            unsupported_native.append("max_codebook_bits>9")
        if codec.get("key_code_bits_per_block") is not None:
            unsupported_native.append("key_code_bits_per_block")
        if codec.get("value_code_bits_per_block") is not None:
            unsupported_native.append("value_code_bits_per_block")
        if codec.get("avg_bits") is not None:
            unsupported_native.append("avg_bits")
        if codec.get("code_bits_per_pair_block") is not None:
            unsupported_native.append("code_bits_per_pair_block")
        if codec.get("adaptive_bit_split") not in (None, "", "off", "none", False):
            unsupported_native.append("adaptive_bit_split")
        key_scale_mode = _hayatequant_native_key_scale_mode(codec)
        if key_scale_mode is None:
            unsupported_native.append(f"key_scale_mode={codec.get('key_scale_mode')!r}")

        if (
            mode in _HAYATEQUANT_NATIVE_MODES
            and preset is not None
            and kb in (3, 4)
            and vb in (3, 4)
            and not unsupported_native
        ):
            _set_native(
                preset,
                f"Codec 'hayatequant' (k{kb}v{vb}, block-VQ) -> byte-compatible "
                f"vLLM kv_cache_dtype='{preset}' with OneComp's native "
                "HayateQuant key block-VQ backend override.",
            )
            plan.hayate_native = True
            plan.hayate_config = {
                "bits": kb,
                "value_bits": vb,
                "dist": codec.get("dist") or "gaussian",
                "block_size": codec.get("block_size"),
                "max_codebook_bits": int(codec.get("max_codebook_bits") or 9),
                "codebook_init": codec.get("codebook_init") or "auto",
                "rotation_mode": codec.get("rotation_mode") or "auto",
                "key_scale_mode": key_scale_mode,
                "seed": int(codec.get("seed") or 0),
            }
            plan.skip_layers = [int(x) for x in (codec.get("skip_layers") or [])]
            plan.notes.append(
                "The native HayateQuant vLLM path preserves the TurboQuant byte "
                "budget and paged-cache capacity while replacing scalar key "
                "indices with block-VQ key indices (values keep vLLM's uniform "
                "quantization for compatibility)."
            )
        elif mode in _HAYATEQUANT_SCALAR_FALLBACK_MODES and preset is not None:
            _set_native(
                preset,
                f"Codec 'hayatequant' (block-VQ) served via scalar fallback "
                f"kv_cache_dtype='{preset}' (vllm_mode='{mode}'): equivalent "
                "to HayateQuant at block_size=1.",
            )
            _check_rotation_for_native(codec, name, plan)
            plan.skip_layers = [int(x) for x in (codec.get("skip_layers") or [])]
            plan.warnings.append(
                "HayateQuant block-VQ is served through vLLM's scalar "
                f"TurboQuant preset '{preset}'; the block-VQ accuracy gain is "
                "NOT reproduced here. Expect TurboQuant-level (block_size=1) "
                "accuracy and use the HF cache/custom backend path for faithful "
                "block-VQ results."
            )
        else:
            plan.needs_custom_backend = True
            plan.native_supported = False
            plan.kv_cache_dtype = "auto"
            if (
                mode not in _HAYATEQUANT_SCALAR_FALLBACK_MODES
                and mode not in _HAYATEQUANT_NATIVE_MODES
                and mode not in _HAYATEQUANT_STRICT_MODES
            ):
                plan.warnings.append(
                    f"Unknown HayateQuant vLLM mode '{mode}'; treating it as "
                    "strict/custom to avoid silently changing accuracy."
                )
            if unsupported_native and mode in _HAYATEQUANT_NATIVE_MODES:
                plan.notes.append(
                    "Codec 'hayatequant' requested native vLLM block-VQ, but "
                    f"the config uses unsupported native knobs {unsupported_native}; "
                    "use the HF cache/custom backend for faithful results or set "
                    "vllm_mode='scalar_fallback' explicitly."
                )
            elif preset is None and (
                mode in _HAYATEQUANT_SCALAR_FALLBACK_MODES
                or mode in _HAYATEQUANT_NATIVE_MODES
            ):
                plan.notes.append(
                    f"Codec 'hayatequant' k{kb}v{vb} uses block-VQ but has no "
                    "native scalar TurboQuant preset in vLLM (presets: k8v4, "
                    "4bit_nc, k3v4_nc, 3bit_nc); use the HF cache/custom backend "
                    "for faithful results or choose a supported bit-width."
                )
            else:
                plan.notes.append(
                    f"Codec 'hayatequant' with block_size={codec.get('block_size')} "
                    "> 1 (block-VQ) is in strict/custom mode "
                    f"(vllm_mode='{mode}'); use the HF cache/custom backend path "
                    "to get its accuracy gain over the scalar turboquant_* presets."
                )
    elif name in ("turboquant", "halo", "hayatequant"):
        tq_preset = resolve_turboquant_codec(codec)
        kb = tq_preset.key_bits if tq_preset is not None else int(codec.get("bits", 4))
        raw_vb = codec.get("value_bits")
        vb = (
            tq_preset.value_bits
            if tq_preset is not None
            else int(raw_vb) if raw_vb is not None else kb
        )
        if tq_preset is not None:
            preset = tq_preset.kv_cache_dtype
            _set_native(
                preset,
                f"Codec '{name}' (k{kb}v{vb}) -> native vLLM kv_cache_dtype="
                f"'{preset}' (requires a vLLM build with native TurboQuant KV "
                "cache + FlashAttention-2; boundary layers auto-protected).",
            )
            # The native kernel hard-codes a randomized *Hadamard* rotation;
            # verify the codec's rotation family matches (or warn loudly).
            _check_rotation_for_native(codec, name, plan)
            # Propagate boundary/outlier layer protection (1-line fix the OneComp
            # outlier-layer detector already produces but the plugin used to drop).
            plan.skip_layers = [int(x) for x in (codec.get("skip_layers") or [])]
            # The named presets do NOT reproduce every OneComp knob.  vLLM's
            # TurboQuant runs Lloyd-Max on *keys* and *uniform* quant on values
            # (and the "8" in k8v4 is FP8 keys, not 8-bit Lloyd-Max), so the
            # numbers differ from the HF rotated-Lloyd-Max-on-both path; warn for
            # turboquant exactly as for halo rather than swapping semantics
            # silently.
            plan.warnings.append(
                f"Codec '{name}' (k{kb}v{vb}) maps to vLLM preset '{preset}', "
                "whose keys are Lloyd-Max and values are uniform (and k8 = FP8 "
                "keys); this is not bit-identical to OneComp's rotated Lloyd-Max "
                "on both K and V (HF cache path)."
            )
            # Knobs with no preset equivalent are dropped -- say so explicitly.
            if int(codec.get("n_outlier") or 0) > 0:
                plan.warnings.append(
                    f"Codec '{name}' n_outlier={codec.get('n_outlier')} (2.5/3.5-bit "
                    "outlier split) has no preset equivalent and is dropped by the "
                    "native mapping; use the HF cache path to keep it."
                )
            if int(codec.get("key_n_outlier") or 0) > 0 or int(codec.get("value_n_outlier") or 0) > 0:
                plan.warnings.append(
                    f"Codec '{name}' role-specific outlier split "
                    f"(key_n_outlier={codec.get('key_n_outlier')}, "
                    f"value_n_outlier={codec.get('value_n_outlier')}) has no preset "
                    "equivalent and is dropped by the native mapping; use the HF "
                    "cache path to keep the BlockQuant-style recipe."
                )
            if codec.get("use_qjl"):
                plan.warnings.append(
                    f"Codec '{name}' use_qjl=True is ignored: upstream vLLM "
                    "intentionally omits the QJL residual (it amplifies variance "
                    "through softmax and harms attention quality)."
                )
            if name in ("halo", "hayatequant"):
                plan.warnings.append(
                    f"{'HALO' if name == 'halo' else 'HayateQuant'}'s Hadamard-GEMM "
                    "/ role-asymmetric (NC key, LS value) scale refinements run "
                    "only in the HF cache path; the vLLM preset uses "
                    "norm-correction on keys only."
                )
        else:
            plan.needs_custom_backend = True
            plan.kv_cache_dtype = "auto"
            plan.notes.append(
                f"Codec '{name}' k{kb}v{vb} has no native TurboQuant preset "
                "(vLLM presets: k8v4, 4bit_nc, k3v4_nc, 3bit_nc); use the HF "
                "cache path or one of those bit-widths."
            )
    elif name in _NATIVE_OTHER:
        dtype = _NATIVE_OTHER[name]
        _set_native(dtype, f"Codec '{name}' -> native vLLM kv_cache_dtype='{dtype}'.")
        # Propagate outlier/boundary layer protection for INT8/NVFP4 as well.
        plan.skip_layers = [int(x) for x in (codec.get("skip_layers") or [])]
    elif name in _NEEDS_CUSTOM_BACKEND:
        plan.needs_custom_backend = True
        plan.native_supported = False
        plan.kv_cache_dtype = "auto"
        plan.notes.append(
            f"Codec '{name}' has no native vLLM kv_cache_dtype; it requires the "
            "experimental custom attention backend (use the HF cache path for "
            "faithful results today)."
        )
    else:
        plan.native_supported = False
        plan.kv_cache_dtype = requested_kv_cache_dtype
        plan.warnings.append(f"Unknown KV codec '{name}'; leaving kv_cache_dtype unchanged.")

    return plan


def engine_kwargs_from_kv_config(
    kv_config: Optional[dict],
    requested_kv_cache_dtype: str = "auto",
) -> dict:
    """Convenience: return the vLLM engine kwargs for a ``kv_compression_config``.

    Example::

        from vllm import LLM
        from vllm_plugins.kvcomp.mapping import engine_kwargs_from_kv_config

        kwargs = engine_kwargs_from_kv_config(cfg["kv_compression_config"])
        llm = LLM(model=path, **kwargs)
    """
    return plan_from_kv_config(kv_config, requested_kv_cache_dtype).engine_kwargs()
