#!/bin/bash
#SBATCH --job-name=hq-rt-speed-v8
#SBATCH --partition=batch-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/logs/hq-rt-speed-v8_%j.log

# v8 speed campaign: fused fake-quant round trip (+ optional CUDA-graph
# compile) for HayateQuant.  Single node, single session, so every A/B row
# (TurboQuant butterfly / TurboQuant+FWHT kernel / HayateQuant fused /
# HayateQuant fused+compiled) is measured under identical conditions.
set -x
cd /lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
source .venv/bin/activate
nvidia-smi --query-gpu=name,driver_version --format=csv
RES=/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_hayatequant/results_speed
MODEL=/lustre/home/frj/yuma-ichikawa/conference/models/llama-3.1-8b-instruct
mkdir -p "$RES"

# --- 0) GPU parity: fused round trip vs decode(encode(x)), eager & compiled ---
python - <<'EOF'
import os, torch
from onecomp.kvcomp.codec.hayate import HayateQuantCodec

def check(tag):
    torch.manual_seed(0)
    worst_mse, worst_frac = 0.0, 0.0
    for kw in (dict(bits=2), dict(bits=3), dict(bits=4),
               dict(bits=4, value_bits=2), dict(bits=3, block_size=1)):
        c = HayateQuantCodec(**kw)
        for s in (1, 64, 4096):
            k = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
            k1, v1 = c.decode(c.encode(k, v))
            k2, v2, _ = c.fake_quant_roundtrip(k, v)
            for a, b, x in ((k1, k2, k), (v1, v2, v)):
                m1 = float((a - x).float().pow(2).mean())
                m2 = float((b - x).float().pow(2).mean())
                rel = abs(m1 - m2) / max(m1, 1e-12)
                frac = float((a != b).float().mean())
                worst_mse, worst_frac = max(worst_mse, rel), max(worst_frac, frac)
    print(f"[parity/{tag}] worst rel-MSE diff {worst_mse:.2e}, "
          f"worst differing-element fraction {worst_frac:.2e}")
    # Tie-flips at fp GEMM noise level may swap a few nearest-codeword
    # assignments; the reconstruction *quality* must be unchanged.
    assert worst_mse < 1e-3 and worst_frac < 1e-3, (worst_mse, worst_frac)

check("eager")
os.environ["ONECOMP_COMPILE_CODEC"] = "1"
check("compiled")
os.environ.pop("ONECOMP_COMPILE_CODEC")
print("GPU parity OK")
EOF
[ $? -ne 0 ] && exit 1

cd benchmark/kvcomp-longcontext

# --- 1) codec microbench: butterfly baseline / FWHT kernel / +compiled -----
ONECOMP_DISABLE_FHT=1 python codec_speed.py --device cuda --heads 32 --seq 4096 \
    --bits 2 3 4 --repeats 20 --out "$RES/codec_speed_v8_nofht.json"
python codec_speed.py --device cuda --heads 32 --seq 4096 \
    --bits 2 3 4 --repeats 20 --out "$RES/codec_speed_v8.json"
ONECOMP_COMPILE_CODEC=1 python codec_speed.py --device cuda --heads 32 --seq 4096 \
    --bits 2 3 4 --repeats 20 --out "$RES/codec_speed_v8_compiled.json"

# --- 2) end-to-end generation: same three settings -------------------------
ONECOMP_DISABLE_FHT=1 python e2e_speed.py --model "$MODEL" --prompt-len 4096 \
    --gen-len 128 --bits 2 3 4 --repeats 3 --out "$RES/e2e_speed_v8_nofht.json"
python e2e_speed.py --model "$MODEL" --prompt-len 4096 \
    --gen-len 128 --bits 2 3 4 --repeats 3 --out "$RES/e2e_speed_v8.json"
ONECOMP_COMPILE_CODEC=1 python e2e_speed.py --model "$MODEL" --prompt-len 4096 \
    --gen-len 128 --bits 2 3 4 --repeats 3 --out "$RES/e2e_speed_v8_compiled.json"

echo "ALL_DONE"
