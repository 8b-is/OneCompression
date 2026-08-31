"""FP8 (E4M3) KV codec -- the recommended 2026 baseline.

vLLM's evaluations report FP8 KV cache as the safest default: ~2x KV capacity
with near-zero quality loss across most workloads.  This codec is therefore the
baseline against which low-bit codecs must prove themselves.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .._base import CompressedKV, KVCodec
from .._registry import register_kv_codec
from ._quant_primitives import fake_quantize_fp8


@register_kv_codec("fp8")
@dataclass
class FP8Codec(KVCodec):
    """Per-token FP8 (E4M3) KV quantization.

    Data-free and tuning-free.  On builds without ``torch.float8_e4m3fn`` this
    transparently falls back to symmetric 8-bit affine quantization.
    """

    bits: int = 8
    key_axis: str = "token"
    value_axis: str = "token"
    scale_bits: int = 32

    def validate_params(self):
        for attr in ("key_axis", "value_axis"):
            if getattr(self, attr) not in ("token", "channel"):
                raise ValueError(f"FP8Codec.{attr} must be 'token' or 'channel'")

    def encode(self, key, value, ctx=None) -> CompressedKV:
        # Store the already-fake-quantized tensors so that decode is a no-op.
        # (Real FP8 storage lives in the kernel layer; the research layer keeps
        # the dequantized tensor, matching the weight side's behaviour.)
        kq = fake_quantize_fp8(key, axis=self.key_axis)
        vq = fake_quantize_fp8(value, axis=self.value_axis)

        def n_scales(x, axis):
            reduced = x.shape[-1] if axis == "token" else x.shape[-2]
            return x.numel() // reduced

        stored = (key.numel() + value.numel()) * 8 + self.scale_bits * (
            n_scales(key, self.key_axis) + n_scales(value, self.value_axis)
        )
        return CompressedKV(
            key_q=kq,
            value_q=vq,
            meta={"dtype": key.dtype, "stored_bits": int(stored), "bits": 8, "passthrough": True},
        )

    def decode(self, compressed: CompressedKV, ctx=None):
        dtype = compressed.meta.get("dtype", torch.float16)
        return compressed.key_q.to(dtype), compressed.value_q.to(dtype)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg["scale_bits"] = self.scale_bits
        return cfg
