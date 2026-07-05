#!/bin/bash
#SBATCH --job-name=hq-rt-speed-v11
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/hq-rt-speed-v11_%j.log

# v11: verify + measure the Triton kernel on the PREFILL regime.
#   0) full GPU parity suite (triton vs eager, roundtrip vs encode/decode)
#   1) prefill A/B: LUT path vs ONECOMP_TRITON_PREFILL=1 (exact in-kernel
#      search) -- quality can only improve (exact >= LUT); question is speed.
#   2) e2e with the full triton serving mode if (1) wins.
set -x
cd /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
RES=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results_speed
MODEL=/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct
mkdir -p "$RES"

# --- 0) GPU parity ----------------------------------------------------------
python -m pytest tests/onecomp/kvcomp/test_kvcomp.py -q -x \
    -k "triton_step or roundtrip" 2>&1 | tail -3
[ ${PIPESTATUS[0]} -ne 0 ] && exit 1

# Prefill-regime parity + quality sanity (exact must be <= LUT distortion).
python - <<'EOF'
import os, torch
from onecomp.kvcomp.codec.hayate.codec import HayateQuantCodec

torch.manual_seed(0)
for bits in (2, 3, 4):
    c = HayateQuantCodec(bits=bits)
    k = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
    os.environ.pop("ONECOMP_TRITON_CODEC", None)
    os.environ.pop("ONECOMP_TRITON_PREFILL", None)
    k1, v1, s1 = c.fake_quant_roundtrip(k, v)
    os.environ["ONECOMP_TRITON_CODEC"] = "1"
    os.environ["ONECOMP_TRITON_PREFILL"] = "1"
    k2, v2, s2 = c.fake_quant_roundtrip(k, v)
    assert s1 == s2
    m_lut = float((k1 - k).float().pow(2).mean() + (v1 - v).float().pow(2).mean())
    m_tri = float((k2 - k).float().pow(2).mean() + (v2 - v).float().pow(2).mean())
    print(f"bits={bits} prefill MSE lut={m_lut:.6f} triton={m_tri:.6f} "
          f"(triton/lut = {m_tri/m_lut:.4f})")
    assert m_tri <= m_lut * 1.001, "exact in-kernel search must not lose to LUT"
print("PREFILL_PARITY_OK")
EOF
[ $? -ne 0 ] && exit 1

cd benchmark/kvcomp-longcontext

# --- 1) codec microbench: triton incl. prefill ------------------------------
ONECOMP_TRITON_CODEC=1 ONECOMP_TRITON_PREFILL=1 python codec_speed.py \
    --device cuda --heads 32 --seq 4096 --bits 2 3 4 --repeats 20 \
    --out "$RES/codec_speed_v11_triton_prefill.json"

# --- 2) e2e with the full triton serving mode -------------------------------
ONECOMP_TRITON_CODEC=1 ONECOMP_TRITON_PREFILL=1 python e2e_speed.py \
    --model "$MODEL" --prompt-len 4096 --gen-len 128 --bits 2 3 4 --repeats 3 \
    --out "$RES/e2e_speed_v11_triton_prefill.json"

echo "ALL_DONE"
