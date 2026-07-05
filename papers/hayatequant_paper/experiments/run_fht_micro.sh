#!/usr/bin/env bash
#SBATCH --job-name=fht-micro
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/%x_%j.log
# Micro-benchmark: Dao-AILab fast-hadamard-transform kernel vs dense GEMM
# rotation at KV head_dim scale (D=128) and at large D (4096), to quantify
# where the FWHT-vs-Dense speed trade-off actually lives.
set -euo pipefail
PY=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python
UV=/lustre/home/frj/yuma-ichikawa/.local/bin/uv
FHT=/lustre/home/frj/yuma-ichikawa/conference/quant_ref/fast-hadamard-transform
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
echo "host=$(hostname) start=$(date -Is) nvcc=$(which nvcc || echo MISSING)"
cd "$FHT"
$UV pip install -v . --python "$PY" --no-build-isolation 2>&1 | tail -3

$PY - <<'EOF'
import torch, time
from fast_hadamard_transform import hadamard_transform

def bench(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

for D in (128, 4096):
    N = 32 * 4096 * (128 // min(D, 128))  # keep total elements comparable
    if D == 4096: N = 4096
    x16 = torch.randn(N, D, device="cuda", dtype=torch.float16)
    x32 = x16.float()
    q16 = torch.linalg.qr(torch.randn(D, D, device="cuda"))[0].half()
    q32 = q16.float()
    t_memcpy = bench(lambda: x16.clone())
    t_fht = bench(lambda: hadamard_transform(x16, scale=D ** -0.5))
    t_gemm16 = bench(lambda: x16 @ q16)
    t_gemm32 = bench(lambda: x32 @ q32)
    print(f"D={D:5d} N={N:8d}: memcpy={t_memcpy:.4f}ms  FHT(fp16)={t_fht:.4f}ms  "
          f"denseGEMM(fp16)={t_gemm16:.4f}ms  denseGEMM(fp32)={t_gemm32:.4f}ms")
EOF
echo "done=$(date -Is)"
