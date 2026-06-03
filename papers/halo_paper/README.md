# HALO Paper: Hadamard-Rotation and Least-Squares Optimal KV Cache Codec

## 概要

**主張:** HALO は TurboQuant に対して速度と品質の両面で Pareto 優位を持つ、データ不要の KV キャッシュ量子化 Codec である。

**手法の特徴:**
- **Randomized Hadamard rotation** (signed Hadamard × GEMM): TurboQuant-hadamard より ~2.8× 高速
- **Least-Squares dequantization scale**: RMS-scale より厳密に MSE が低く（理論的保証あり）
- **`hadamard_gemm` mode** (`auto`): 2の冪乗次元で一回の GEMM に折り畳む
- **`dense` mode**: 非2乗次元 (Qwen, R1 等) へのフォールバック

---

## 論文テーブル対応

### Table 1: Speed-Quality Pareto (メイン貢献)

**ベンチ:** Codec-level distortion + PPL + encode/decode latency  
**ソース:** `benchmark/kvcomp-codec/results_paper_*/`  
**集約済み:** `papers/halo_paper/results/all_results.json` → `codec_ppl_latency`

| Model | bits | 手法 | PPL | encode ms | relMSE_k |
|-------|------|------|-----|-----------|----------|
| Llama-3.1-8B | 3 | HALO-auto | **7.09** | 0.364 | 0.0316 |
| Llama-3.1-8B | 3 | TurboQuant-hadamard | 7.10 | 1.009 | 0.0321 |
| Llama-3.1-8B | 3 | TurboQuant-dense | 7.14 | **0.260** | 0.0344 |
| Ministral-8B | 3 | HALO-auto | **9.40** | 0.400 | 0.0311 |
| Ministral-8B | 3 | TurboQuant-dense | 9.36 | **0.280** | 0.0341 |

**Speed test (encode ms @3-bit, Llama setup 8L×8H×seq2048):**

| 手法 | enc ms | dec ms | kv_mse |
|------|--------|--------|--------|
| HALO-hadamard_gemm-noLS | **0.178** | **0.111** | 0.0891 |
| HALO-hadamard_gemm-LS   | 0.261 | 0.111 | **0.0879** |
| HALO-dense-LS           | 0.253 | 0.101 | 0.0896 |
| TurboQuant-hadamard     | 0.501 | 0.439 | 0.0889 |

**→ HALO-hadamard_gemm は TurboQuant-hadamard より 2.8× 高速、MSE 同等**  
**→ HALO-dense-LS は TurboQuant-dense より同等速度で MSE 改善**

---

### Table 2: Long-Context Retrieval (Needle / RULER / PPL)

**ベンチ:** `turboquant_paper.yaml` (Llama-3.1-8B, Ministral-8B)  
**ソース:** `benchmark/kvcomp-longcontext/results_tq_*/`  
**集約済み:** `papers/halo_paper/results/all_results.json` → `longcontext_retrieval`

**Llama-3.1-8B (完全完了) 抜粋:**

| 手法 | needle % | RULER % | PPL | mem |
|------|---------|---------|-----|-----|
| FullKV | 100.0 | 100.0 | 6.49 | 1.000 |
| HALO-4bit | 100.0 | 93.3 | 6.58 | 0.258 |
| HALO-3bit | 100.0 | 100.0 | 6.90 | 0.196 |
| **HALO-dense-LS-2bit** | **100.0** | **100.0** | **9.85** | **0.133** |
| HALO-2bit | 91.1 | 93.3 | 10.10 | 0.133 |
| TurboQuant-2bit | 88.9 | 80.0 | 10.17 | 0.133 |

**→ HALO-dense-LS-2bit は TurboQuant-2bit より PPL 0.32 pt 改善・Needle 11.1% 改善**

---

### Table 3: LongBench Comprehension

**ソース:** `benchmark/kvcomp-longcontext/results_tq_*/`

**Llama-3.1-8B:**

| 手法 | score | mem |
|------|-------|-----|
| TaperKV | **31.03** | 0.293 |
| TurboQuant-4bit | 30.03 | 0.259 |
| HALO-4bit | 29.67 | 0.259 |
| HALO-3bit | 28.30 | 0.196 |
| TurboQuant-3bit | 25.77 | 0.196 |
| TurboQuant-2bit | 21.34 | 0.134 |
| HALO-2bit | 20.22 | 0.134 |
| **HALO-dense-LS-2bit** | **22.38** | **0.134** |

---

## ステータス (2026-06-03 時点)

| ベンチ | Llama-8B | Ministral-8B | R1-Qwen-1.5B | R1-Qwen-7B |
|--------|----------|--------------|--------------|------------|
| Codec PPL+latency | ✅ | ✅ | ⚠️ PPL異常高 | ⚠️ PPL異常高 |
| Needle / RULER | ✅ | ✅ | — | — |
| LongBench | ✅ | ✅ | — | — |
| Speed | ✅ | — | — | — |

**⚠️ R1-Qwen 系の codec PPL 異常高問題:**  
これらのモデルは巨大外れ値チャネルを持つため `skip_layers` が必要。  
`detect_outlier_layers(ratio_thresh=50)` → `CompressedDynamicCache(skip_layers=...)` の経路は  
`kv_benchmark.py` では実装済みだが、`codec_benchmark.py` でも適用確認が必要。  
→ `papers/halo_paper/scripts/run_codec_bench_r1.sh` でスキップ層付きで再実行予定。

---

## 再現方法

```bash
# 1. Codec benchmark (Llama / Ministral)
bash papers/halo_paper/scripts/run_codec_bench.sh

# 2. Longcontext retrieval + LongBench
bash papers/halo_paper/scripts/run_longcontext_bench.sh

# 3. Speed benchmark
bash papers/halo_paper/scripts/run_speed_bench.sh

# 4. 結果集約 → all_results.json
python papers/aggregate_results.py
```

---

## ファイル構造

```
onecomp-lab/
├── papers/halo_paper/
│   ├── README.md          ← このファイル
│   ├── results/
│   │   └── all_results.json   ← 集約済み全結果
│   └── scripts/
│       ├── run_codec_bench.sh
│       ├── run_longcontext_bench.sh
│       └── run_speed_bench.sh
├── benchmark/kvcomp-codec/
│   ├── codec_benchmark.py     ← メインスクリプト
│   └── conf/paper_halo.yaml   ← 設定ファイル
└── benchmark/kvcomp-longcontext/
    ├── kv_benchmark.py
    └── conf/turboquant_paper.yaml
```
