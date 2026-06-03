#!/usr/bin/env bash
# ============================================================
# HALO Paper — Codec Speed benchmark
# Reproduces: papers/halo_paper/results/all_results.json → speed
#
# This runs a standalone speed measurement script (no model needed,
# uses synthetic KV tensors sized to Llama-3.1-8B head config).
# ============================================================
set -euo pipefail

REPO=/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab
PY=${REPO}/.venv/bin/python
OUT_DIR=${REPO}/benchmark/kvcomp-longcontext/results_codec_speed
mkdir -p "$OUT_DIR"

cat > /tmp/run_speed.py << 'EOF'
"""Standalone codec speed measurement (GPU required)."""
import sys, json, time, torch
sys.path.insert(0, '/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab')
from onecomp.kvcomp.codec.halo import HALOCodec
from onecomp.kvcomp.codec.turboquant import TurboQuantCodec
from onecomp.kvcomp.codec.rotation import random_sign

DEVICE = "cuda:0"
N_LAYERS, N_HEADS, SEQ, HEAD_DIM = 8, 8, 2048, 128
N_WARMUP, N_RUNS = 5, 50

def measure(codec, bits, name):
    k = torch.randn(1, N_HEADS, SEQ, HEAD_DIM, device=DEVICE)
    v = torch.randn(1, N_HEADS, SEQ, HEAD_DIM, device=DEVICE)
    # warmup
    for _ in range(N_WARMUP):
        comp = codec.encode(k, v)
        codec.decode(comp)
    torch.cuda.synchronize()
    # encode
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        comp = codec.encode(k, v)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) / N_RUNS * 1000
    # decode
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        codec.decode(comp)
    torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) / N_RUNS * 1000
    # quality
    kr, vr = codec.decode(comp)
    kv_mse = ((k - kr).pow(2).mean() / k.pow(2).mean()).item()
    print(f"  {name:<40} bits={bits} enc={enc_ms:.3f}ms dec={dec_ms:.3f}ms kv_mse={kv_mse:.4f}")
    return {"enc_ms": enc_ms, "dec_ms": dec_ms, "kv_mse": kv_mse}

results = {}
for bits in [2, 3, 4]:
    results[str(bits)] = {}
    methods = {
        "TurboQuant-hadamard": TurboQuantCodec(bits=bits, rotation_mode="hadamard"),
        "HALO-hadamard-noLS":  HALOCodec(bits=bits, rotation_mode="hadamard", ls_scale=False),
        "HALO-hadamard_gemm-noLS": HALOCodec(bits=bits, rotation_mode="hadamard_gemm", ls_scale=False),
        "HALO-hadamard_gemm-LS":   HALOCodec(bits=bits, rotation_mode="hadamard_gemm", ls_scale=True),
        "HALO-dense-LS":           HALOCodec(bits=bits, rotation_mode="dense", ls_scale=True),
    }
    print(f"\n--- bits={bits} ---")
    for name, codec in methods.items():
        results[str(bits)][name] = measure(codec, bits, name)

meta = {"n_layers": N_LAYERS, "n_heads": N_HEADS, "seq": SEQ, "head_dim": HEAD_DIM,
        "n_runs": N_RUNS, "model": "llama-8b-config"}
out = {"meta": meta, "results": results}
outpath = '/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/benchmark/kvcomp-longcontext/results_codec_speed/codec_speed_llama.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {outpath}")
EOF

$PY /tmp/run_speed.py
$PY "${REPO}/papers/aggregate_results.py"
echo "Speed benchmark done."
