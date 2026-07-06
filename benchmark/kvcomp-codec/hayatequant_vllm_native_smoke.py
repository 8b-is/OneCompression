"""Smoke test for the native vLLM HayateQuant key block-VQ kernels.

This script intentionally avoids model loading.  It allocates the same
TurboQuant-layout KV cache that vLLM uses for ``turboquant_3bit_nc`` and
``turboquant_4bit_nc``, stores random K/V with the OneComp HayateQuant native
kernel, then runs one decode attention step.  It is small enough for a quick
Slurm GPU sanity check but catches the important failures: Triton compilation,
cache layout mismatches, and non-finite decode output.

Example
-------
    sbatch --gres=gpu:1 --wrap \
      "cd /path/to/onecomp-lab && .venv-vllm/bin/python \
       benchmark/kvcomp-codec/hayatequant_vllm_native_smoke.py"
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from onecomp.kvcomp.codec.hayate.codec import _block_codebook
from onecomp.kvcomp.codec.rotation import hybrid_rotation_matrix
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm_plugins.kvcomp.hayate_triton import (
    triton_hayatequant_decode_attention,
    triton_hayatequant_store,
)


def _dtype_for(bits: int) -> str:
    if bits == 3:
        return "turboquant_3bit_nc"
    if bits == 4:
        return "turboquant_4bit_nc"
    raise ValueError("native smoke supports bits in {3, 4}")


def run_case(bits: int, *, head_dim: int, kv_heads: int, q_heads: int, tokens: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native vLLM HayateQuant smoke test")
    device = torch.device("cuda")
    torch.manual_seed(0)

    dtype_name = _dtype_for(bits)
    tq = TurboQuantConfig.from_cache_dtype(dtype_name, head_dim)
    page_size = 16
    num_blocks = math.ceil(tokens / page_size)
    kv_cache = torch.empty(
        num_blocks,
        page_size,
        kv_heads,
        tq.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int32)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.float16)
    query = torch.randn(1, q_heads, head_dim, device=device, dtype=torch.float16)

    block_size = 2
    rotation = hybrid_rotation_matrix(head_dim, block_size, 0, str(device), torch.float32)
    codebook = _block_codebook(bits, block_size, "gaussian", head_dim).to(
        device=device, dtype=torch.float32
    )

    triton_hayatequant_store(
        key,
        value,
        kv_cache,
        slot_mapping,
        rotation,
        codebook,
        block_size,
        bits,
        key_packed_size=tq.key_packed_size,
        value_quant_bits=tq.effective_value_quant_bits,
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([tokens], device=device, dtype=torch.int32)
    out = triton_hayatequant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        rotation=rotation,
        codebook=codebook,
        scale=head_dim**-0.5,
        key_bits=bits,
        key_packed_size=tq.key_packed_size,
        value_quant_bits=tq.effective_value_quant_bits,
        hayate_block_size=block_size,
        norm_correction=tq.norm_correction,
        max_num_kv_splits=1,
    )
    torch.cuda.synchronize()
    if out.shape != (1, q_heads, head_dim):
        raise AssertionError(f"unexpected output shape: {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        raise AssertionError("decode output contains non-finite values")
    return {
        "bits": bits,
        "dtype": dtype_name,
        "slot_size_aligned": tq.slot_size_aligned,
        "output_abs_mean": float(out.float().abs().mean().item()),
        "output_abs_max": float(out.float().abs().max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    rows = [
        run_case(
            b,
            head_dim=args.head_dim,
            kv_heads=args.kv_heads,
            q_heads=args.q_heads,
            tokens=args.tokens,
        )
        for b in args.bits
    ]
    print(json.dumps({"ok": True, "cases": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
