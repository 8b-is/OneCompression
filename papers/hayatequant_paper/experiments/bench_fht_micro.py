"""Microbenchmark: FWHT CUDA kernel vs PyTorch butterfly vs folded Hadamard GEMM.

Shapes mirror the paper's speed harness: head_dim=128, KV cache of a
Llama-3.1-8B layer (8 KV heads), batch 1, various context lengths.

Run on a GPU node:
    .venv/bin/python bench_fht.py
"""

import math

import torch

from onecomp.kvcomp.codec import rotation as R

assert R._fht_cuda is not None, "fast_hadamard_transform kernel not installed"

DEV = "cuda"
D = 128
SEED = 0


def timeit(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'shape':>24} {'dtype':>8} | {'butterfly':>10} {'FWHT kernel':>11} "
          f"{'folded GEMM':>11} | {'kernel/GEMM':>11}")
    for tokens in (4096, 16384, 65536, 131072):
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(1, 8, tokens, D, device=DEV, dtype=dtype)
            sign = R.random_sign(D, SEED, device=DEV, dtype=dtype)
            hs = R.signed_hadamard_matrix(D, SEED, DEV, dtype)
            scale = 1.0 / math.sqrt(D)

            t_butter = timeit(lambda: R._wht_butterfly(x * sign))
            t_kernel = timeit(lambda: R._fht_cuda((x * sign).contiguous(), scale=scale))
            t_gemm = timeit(lambda: x @ hs)

            shape = f"(1,8,{tokens},{D})"
            ratio = t_kernel / t_gemm
            print(f"{shape:>24} {str(dtype).split('.')[-1]:>8} | "
                  f"{t_butter:>9.1f}u {t_kernel:>10.1f}u {t_gemm:>10.1f}u | "
                  f"{ratio:>10.2f}x")


if __name__ == "__main__":
    with torch.no_grad():
        main()
