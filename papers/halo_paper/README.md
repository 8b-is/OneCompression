# HALO Paper: Hadamard-Rotation and Least-Squares Optimal KV Cache Codec

> **⚠️ superseded by HayateQuant (2026-07-04):** HALO は
> [`papers/hayatequant_paper/`](../hayatequant_paper/README.md) の HayateQuant に
> 全面的に置き換えられた。`block_size=1` は、rotation/source-law/scale と
> metadata precision を明示的に揃えた scalar-path compatibility control であり、
> 任意の fp16 丸め順まで含む bit-for-bit strict superset の主張ではない。
> 以下の旧テーブルは HALO の比較履歴である。本 README は比較対象・再現性の
> ために保存し、各テーブルに HayateQuant との対応比較を付記する。
> 新規開発は `hayatequant` codec を使うこと。

## 概要

**主張 (歴史的):** HALO は TurboQuant に対して速度と品質の両面で Pareto 優位を持つ、データ不要の KV キャッシュ量子化 Codec である。

**手法の特徴:**
- **Randomized Hadamard rotation** (signed Hadamard × GEMM): TurboQuant-hadamard より ~2.8× 高速
- **Least-Squares dequantization scale**: RMS-scale より厳密に MSE が低く(理論的保証あり)
- **`hadamard_gemm` mode** (`auto`): 2の冪乗次元で一回の GEMM に折り畳む
- **`dense` mode**: 非2乗次元 (Qwen, R1 等) へのフォールバック

**HayateQuant が HALO に追加したもの (詳細は hayatequant_paper):**
- データ不要 **block vector quantization** (space-filling gain; HALO 分析が
  「Lever 5 / future work」として残した部分の実装)
- **Folded Hybrid Rotation** (Hadamard+block-dense を 1 GEMM に折り畳み。HALO の
  hadamard_gemm の折り畳み発想をハイブリッド回転へ一般化)
- **候補リスト Grid-LUT エンコーダ + fp16 探索** (符号帳サイズ非依存 O(T) エンコード)
- HALO の役割非対称スケール・outlier 混合精度は**そのまま継承** (`_scale.py` を共有)

---

## 論文テーブル対応 (各表に HayateQuant 対応比較を付記)

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

**→ HayateQuant による更新 (v3 実測, 同一プロトコル `results_v3/`):**

| Model | bits | 手法 | PPL | encode ms | relMSE_k |
|-------|------|------|-----|-----------|----------|
| Llama-3.1-8B | 3 | HALO-auto (本論文の最良) | 7.086 | 0.303 | 0.0319 |
| Llama-3.1-8B | 3 | **HayateQuant-auto** | **7.004** | 0.587 | **0.0290** (-9.0%) |
| Llama-3.1-8B | 3 | **HayateQuant-K12** | **6.985** | 8.33 | **0.0270** (-15.3%) |
| Llama-3.1-8B | 2 | HALO-dense-LS (本論文の最良PPL) | 9.868 | 0.252 | 0.1166 |
| Llama-3.1-8B | 2 | **HayateQuant-dense-LS** | **8.930** (-9.5%) | 3.18 | **0.0969** (-16.9%) |
| R1-Qwen-1.5B | 2 | HALO-dense-LS (本論文の最良PPL) | 159.38 | 0.248 | 0.1119 |
| R1-Qwen-1.5B | 2 | **HayateQuant-dense-LS** | **157.45** | 3.19 | **0.0992** (-11.3%) |

**Speed test (encode ms @3-bit, Llama setup 8L×8H×seq2048):**

| 手法 | enc ms | dec ms | kv_mse |
|------|--------|--------|--------|
| HALO-hadamard_gemm-noLS | **0.178** | 0.111 | 0.0891 |
| HALO-hadamard_gemm-LS   | 0.261 | 0.111 | 0.0879 |
| HALO-dense-LS           | 0.253 | 0.101 | 0.0896 |
| TurboQuant-hadamard     | 0.501 | 0.439 | 0.0889 |

**→ HALO-hadamard_gemm は TurboQuant-hadamard より 2.8× 高速、MSE 同等**  
**→ HALO-dense-LS は TurboQuant-dense より同等速度で MSE 改善**

**→ HayateQuant による更新 (v3b, 32H×seq4096 なので絶対値は非互換、比率で見る):**
HayateQuant-auto は enc 4.31ms / dec **0.479ms** / kv_mse **0.0884** —
dec は HALO (0.506ms 同設定) より速く、MSE は HALO 最良 (0.0978) 比 -9.6%。
per-layer 逐次 dequant は 56µs vs HALO 82µs vs TurboQuant 443µs。
e2e (Llama-8B, prompt=4096): HALO-NC 92.4ms TTFT / 33.0 tok/s、
HayateQuant-auto 115.3ms / 27.4 tok/s、TurboQuant 112.0ms / 19.5 tok/s —
**速度は HALO ≳ HayateQuant > TurboQuant、品質は HayateQuant > HALO > TurboQuant**
であり、速度差 (-17% tok/s) より品質差 (歪み -9〜19%, 下表の long-context) が
大きいことが HayateQuant を既定推奨とする根拠。

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

**→ HayateQuant による更新 (job 58907, `hayatequant_paper.yaml`, 同一プロトコル
1 ラン内での直接比較; 乱数・データ抽選が上表と異なるため行間比較は同ラン内で):**

| 手法 | needle % | RULER % | PPL | mem |
|------|---------|---------|-----|-----|
| FullKV | 100.0 | 93.3 | 6.49 | 1.000 |
| HALO-3bit | 100.0 | 86.7 | 6.88 | 0.196 |
| **HayateQuant-3bit** | 100.0 | 86.7 | **6.84** | 0.196 |
| HALO-2.5bit | 100.0 | 66.7 | 7.39 | 0.172 |
| **HayateQuant-2.5bit** | 100.0 | **93.3** (= FullKV) | **7.21** | 0.172 |
| HALO-2bit | 91.1 | 33.3 | 9.97 | 0.133 |
| TurboQuant-2bit | 93.3 | 40.0 | 10.17 | 0.133 |
| **HayateQuant-2bit** | **100.0** | **60.0** | **9.12** | 0.133 |

**→ 低ビット域で HALO を全指標で更新 (2-bit: Needle +8.9pt, RULER +26.7pt, PPL -0.84)。**

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

**→ HayateQuant による更新 (job 58907, 同一ラン内比較):**

| 手法 | score | mem |
|------|-------|-----|
| HALO-3bit | 31.06 | 0.196 |
| **HayateQuant-3bit** | 29.18 | 0.196 |
| **HayateQuant-2.5bit** | **30.03** | **0.172** |
| HALO-2.5bit | 28.09 | 0.172 |
| HALO-2bit | 21.14 | 0.134 |
| TurboQuant-2bit | 21.34 | 0.134 |
| **HayateQuant-2bit** | **26.87** (+5.7pt vs HALO) | 0.134 |

(3-bit の LongBench のみ HALO が上回るが、Needle/RULER/PPL/歪みでは
HayateQuant-3bit が同等以上; LongBench はタスク抽選の分散が大きい点に注意。)

---

## ステータス (2026-07-04 更新)

| ベンチ | Llama-8B | Ministral-8B | R1-Qwen-1.5B | R1-Qwen-7B |
|--------|----------|--------------|--------------|------------|
| Codec PPL+latency | ✅ (+HayateQuant v3) | ✅ | ✅ (+HayateQuant v3, skip_outliers 適用) | ⚠️ PPL異常高 |
| Needle / RULER | ✅ (+HayateQuant 58907) | ✅ | — | — |
| LongBench | ✅ (+HayateQuant 58907) | ✅ | — | — |
| Speed | ✅ (+HayateQuant v3b) | — | — | — |

**⚠️ R1-Qwen 系の codec PPL 異常高問題:**  
これらのモデルは巨大外れ値チャネルを持つため `skip_layers` が必要。  
`detect_outlier_layers(ratio_thresh=50)` → `CompressedDynamicCache(skip_layers=...)` の経路は  
`kv_benchmark.py` では実装済み。HayateQuant v3 の R1 計測は `skip_outliers: true,
outlier_ratio: 50` で実施済み (それでも PPL ~160 と高く、外れ値レイヤの影響が残る
ことは既知; 手法間の相対比較には有効)。

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

# 5. HayateQuant との比較 (v3 系ジョブ)
sbatch ichikawa_works/kvcomp_hayatequant/run_hayatequant_bench_v3.sh
sbatch ichikawa_works/kvcomp_hayatequant/run_hayatequant_speed_v3b.sh
sbatch ichikawa_works/kvcomp_hayatequant/run_hayatequant_paper_bench.sh
```

---

## ファイル構造

```
onecomp-lab/
├── papers/halo_paper/
│   ├── README.md          ← このファイル (HayateQuant 比較付き)
│   ├── results/
│   │   └── all_results.json   ← 集約済み全結果
│   └── scripts/
│       ├── run_codec_bench.sh
│       ├── run_longcontext_bench.sh
│       └── run_speed_bench.sh
├── papers/hayatequant_paper/   ← 後継論文 (推奨)
├── benchmark/kvcomp-codec/
│   ├── codec_benchmark.py     ← メインスクリプト (HayateQuant 変種も含む)
│   └── conf/paper_halo.yaml   ← 設定ファイル
└── benchmark/kvcomp-longcontext/
    ├── kv_benchmark.py
    └── conf/turboquant_paper.yaml / hayatequant_paper.yaml
```
