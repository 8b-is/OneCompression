"""Break down HayateQuant's decode-step (S=1) encode latency.

The e2e A/B showed ~460-560us per encode call for a single new token, vs
TurboQuant's 158us in the same harness -- find where the fixed overhead lives.
"""

import time

import torch

from onecomp.kvcomp.codec.hayatequant import HayateQuantCodec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec

DEV = torch.device("cuda")


def med_time(fn, warm=10, reps=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e6  # us


def main():
    torch.manual_seed(0)
    k = torch.randn(1, 32, 1, 128, device=DEV)
    v = torch.randn(1, 32, 1, 128, device=DEV)

    for label, codec in [
        ("TurboQuant", TurboQuantCodec(bits=3, rotation_mode="hadamard", dist="beta")),
        ("HQ-auto", HayateQuantCodec(bits=3, block_size=None, rotation_mode="auto")),
        ("HQ-bs1", HayateQuantCodec(bits=3, block_size=1, rotation_mode="auto")),
    ]:
        t_enc = med_time(lambda: codec.encode(k, v))
        payload = codec.encode(k, v)
        t_dec = med_time(lambda: codec.decode(payload))
        print(f"{label:12s} enc {t_enc:7.1f} us   dec {t_dec:7.1f} us")

    # Torch profiler on the HQ encode to see the op-level breakdown.
    codec = HayateQuantCodec(bits=3, block_size=None, rotation_mode="auto")
    for _ in range(10):
        codec.encode(k, v)
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            codec.encode(k, v)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))


if __name__ == "__main__":
    with torch.no_grad():
        main()
