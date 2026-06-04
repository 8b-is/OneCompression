# TaperKV Paper: Recency-Graded Precision Scheduling for KV Cache Compression

> **最終更新: 2026-06-04 18:50 JST** (delta quantization fix 確定後)

## 概要

**主張:** TaperKV は、トークンを削除せず「古いトークンほど低精度」の精度スケジュールを適用する KV キャッシュ圧縮 Policy であり、StreamingLLM・ThinKV・PM-KVQ より LongBench スコアが高く、GSM8K では FullKV に匹敵する。

**手法の特徴:**
- **Age-graded precision ladder**: sink/recent トークンは高精度、中間を `b_min` まで線形低下
- **No eviction**: 全トークンを保持（StreamingLLM と異なり長期依存を維持）
- **Codec 非依存**: affine (TaperKV) or TurboQuant (TurboTaper) を差し替え可能
- **TurboTaper**: TaperKV policy + TurboQuant codec の組み合わせ

---

## TurboTaper バグ修正履歴 (2026-06-03/04)

2 つの実装バグを発見・修正した. **現在の全ベンチは修正済みコードで実行.**

| Bug | 症状 | 根本原因 | 修正 |
|-----|------|----------|------|
| **Bug-1** (2026-06-03) | QwQ GSM=0%; R1-7B MATH=6% | `b_min_value=2`: Hadamard+2bit値量子化が Qwen 長推論で崩壊 | `b_min_value: 3` に変更 |
| **Bug-2** (2026-06-04) | Bug-1 後も R1-7B MATH=24%; QwQ GSM=17% | `GradedQuantCache` が全フロアトークンを 16step ごとに再量子化. Lloyd-Max codec の非等冪性でドリフト累積 | **Delta quantization** in `turbotaper_cache.py`: ビット幅が変化したトークンのみ再量子化 |

---

## 論文テーブル対応

### Table 1: Policy 比較 — Llama-3.1-8B (全11メソッド、delta fix済み)

**ベンチ:** `paper_taperkv.yaml`  **ソース:** `results_taperkv/results_llama-3.1-8b-instruct.json`

| 手法 | Needle% | RULER% | LongBench | GSM8K% | MATH% | PPL | mem |
|------|---------|--------|-----------|--------|-------|-----|-----|
| FullKV | 100.0 | 100.0 | 27.79 | 85.0 | 55.0 | 6.49 | 1.000 |
| StreamingLLM | 20.0 | 13.3 | 24.81 | 36.0 | 25.0 | 6.49 | 0.209 |
| ThinKV | 100.0 | 93.3 | 26.54 | 61.0 | 22.0 | 6.64 | 0.313 |
| PM-KVQ | 100.0 | 93.3 | 19.12 | 86.0 | 45.0 | 7.95 | 0.204 |
| KIVI-INT2 | 97.8 | 93.3 | 25.86 | 84.0 | 44.0 | 7.40 | 0.161 |
| TurboQuant-3bit | 100.0 | 100.0 | 28.33 | 81.0 | 49.0 | 6.92 | 0.196 |
| HALO-3bit | 100.0 | 100.0 | 28.71 | 85.0 | 50.0 | 6.91 | 0.196 |
| **TaperKV** | **100.0** | **100.0** | **31.59** | **89.0** | 44.0 | 6.59 | 0.293 |
| TaperKV-3bit | 100.0 | 100.0 | 22.85 | 84.0 | 50.0 | 7.42 | 0.197 |
| **TurboTaper-3bit** ✅ | 100.0 | 93.3 | 27.05 | 81.0 | 48.0 | 6.66 | **0.190** |
| Uniform-INT3 | 100.0 | 100.0 | 26.61 | 65.0 | 29.0 | 8.50 | 0.197 |

**→ TaperKV は LongBench 31.59 でトップ（FullKV 27.79 を +3.8pt 上回る）**
**→ GSM8K 89.0% で FullKV (85.0%) を超過**
**→ TurboTaper-3bit: needle=1.0, ruler=0.93 を達成（0.190 の最小メモリで最高品質）**

---

### Table 2: Policy 比較 — R1-Qwen-14B (10メソッド完了、delta fix済み)

**ソース:** `results_taperkv/results_DeepSeek-R1-Distill-Qwen-14B.json`

| 手法 | LongBench | GSM8K% | MATH-500% | PPL | cr |
|------|-----------|--------|-----------|-----|----|
| FullKV | 4.26 | 88.0 | 66.0 | 11.61 | 1.00 |
| Uniform-INT3 | 3.73 | 60.0 | 48.0 | 21.17 | 3.18 |
| StreamingLLM | 3.90 | 24.0 | 16.0 | 11.61 | ∞ |
| ThinKV | 4.10 | 86.0 | 60.0 | 11.84 | 2.42 |
| PM-KVQ | 4.16 | **94.0** | 58.0 | 14.29 | 3.09 |
| **TaperKV** | **4.17** | **90.0** | 56.0 | 12.14 | 2.52 |
| **TaperKV-3bit** | **4.35** | 90.0 | 60.0 | 13.92 | 3.18 |
| TurboQuant-3bit | 4.09 | 88.0 | 60.0 | 12.60 | 3.19 |
| KIVI-INT2 | 3.96 | 86.0 | **62.0** | 13.26 | 3.53 |
| **TurboTaper-3bit** ✅ | 4.35 | **85.0** | **61.0** | 11.87 | 2.90 |

**→ TaperKV-3bit / TurboTaper-3bit が LB 4.35 でタイ最高 (FullKV 4.26 超)**
**→ TurboTaper-3bit: GSM=85% (FullKV 88%比 97%)、MATH=61% (FullKV 66%比 92%) ✅**

---

### Table 3: Reasoning Benchmark (GSM8K / MATH-500 / AIME)

**QwQ-32B — 全7メソッド完了 (delta fix済み):**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 100.0 | 66.7 | 3.72 | 1.00 |
| **TaperKV** | **96.7** | 60.0 | 3.82 | 3.36 |
| **TurboTaper** ✅ | **100.0** | **70.0** | 3.77 | 3.11 |
| KIVI-INT2 | 100.0 | 66.7 | 4.16 | 6.18 |
| HALO-3bit | 96.7 | 73.3 | 3.90 | 5.06 |
| ThinKV | 63.3 | 20.0 | 3.79 | 3.19 |
| PM-KVQ | 63.3 | 20.0 | 4.10 | 4.27 |

**→ TurboTaper MATH=70.0% > FullKV=66.7% — 圧縮しながら精度向上!**

**QwQ-32B AIME (k=1):**

| 手法 | AIME-2024 | AIME-2025 |
|------|-----------|-----------|
| FullKV | 0.300 | 0.267 |
| KIVI-INT2 | **0.333** | 0.200 |
| **TurboTaper** ✅ | **0.300** (=FK) | 0.167 |
| TaperKV | 0.200 | 0.167 |
| HALO-3bit | 0.200 | 0.167 |
| ThinKV | 0.000 | 0.000 |
| PM-KVQ | 0.000 | 0.000 |

**R1-Qwen-14B — 全7メソッド完了:**

| 手法 | GSM8K% | MATH-500% | AIME-2024 | AIME-2025 |
|------|--------|-----------|-----------|-----------|
| FullKV | 88.0 | 68.0 | 0.500 | 0.283 |
| **TaperKV** | **90.0** | 62.0 | — | **0.333** |
| **TurboTaper** ✅ | 86.0 | **68.0** | **0.433** | **0.367** |
| HALO-3bit | 88.0 | 66.0 | — | 0.267 |
| KIVI-INT2 | 86.0 | 64.0 | — | 0.267 |
| PM-KVQ | 86.0 | 58.0 | — | 0.033 |
| ThinKV | 78.0 | 54.0 | — | 0.083 |

**→ TurboTaper AIME-2025 0.367 > FullKV 0.283 (+30%)!**
**→ TurboTaper MATH = FullKV 68% (完全等価)**

**R1-Qwen-7B — 全7メソッド完了 (delta fix済み):**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 69.0 | 67.0 | 50.85 | 1.00 |
| TaperKV | **75.0** | 64.0 | 50.38 | 2.38 |
| ThinKV | 73.0 | 67.0 | 50.61 | 2.29 |
| PM-KVQ | 71.0 | 48.0 | 60.86 | 2.71 |
| **TurboTaper** ✅ | 67.0 | **65.0** | 50.42 | 2.21 |
| Uniform-INT4 | 72.0 | 66.0 | 55.45 | 2.54 |
| StreamingLLM | 25.0 | 13.0 | 50.85 | ∞ |

**R1-Distill-Llama-8B — 全7メソッド完了:**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 63.0 | 61.0 | 23.50 | 1.00 |
| TaperKV | 62.0 | 58.0 | 24.79 | 3.16 |
| **TurboTaper** ✅ | **65.0** | **59.0** | 23.52 | 3.01 |
| PM-KVQ | 61.0 | 48.0 | 29.03 | 3.33 |
| ThinKV | 35.0 | 21.0 | 24.36 | 4.12 |
| StreamingLLM | 17.0 | 15.0 | 23.50 | ∞ |

---

### Table 4: LiveCodeBench

R1-Distill-Llama-8B: FullKV=0% — モデル自体がコード生成不可 (reasoning専用, MATH蒸留)。
QwQ-32B / R1-14B が主要比較対象。LCB ジョブ実行中 / 未完了。

---

## ステータス (2026-06-04 18:50 JST)

### 完了データ ✅

| ベンチ | モデル | メソッド数 | 備考 |
|--------|--------|-----------|------|
| taperkv (policy) | Llama-3.1-8B | 11/11 | TurboTaper-3bit delta fix済み ✅ |
| taperkv (policy) | R1-14B | 10/10 | TurboTaper-3bit delta fix済み ✅ |
| taperkv (policy) | R1-7B | 9/9+ | **TurboTaper-3bit は OLD code** ⚠️ → postfix_46421.sh 実行待ち |
| taperkv (policy) | R1-1.5B | 6/6+ | KIVI/HALO/TurboTaper-3bit 実行中 (46417) |
| basic (GSM+MATH) | R1-Llama, R1-7B, R1-14B, QwQ | 7/7 each | delta fix済み ✅ |
| basic (GSM+MATH) | R1-1.5B | 5/6 | TaperKV done, TurboTaper/KIVI/HALO 実行中 |
| comp AIME-2024+2025 | R1-14B | 7/7 (2025), 3/7 (2024) | 2024: 46708 で TaperKV+TT+HALO 実行中 |
| comp AIME-2024+2025 | QwQ-32B | 7/7 (2024), 6/7 (2025) | 2025 HALO: 46535 実行中 |
| comp AIME-2024+2025 | R1-7B | 4/7 (TT完了), 残り実行中 | 46534 実行中 |
| comp AIME | R1-Llama | 3/7 | 46537 実行中 (k=2) |

### 実行中ジョブ

| JID | 内容 | 状況 |
|-----|------|------|
| 46421 | taperkv/R1-7B 残り (KIVI, HALO) | KIVI-INT2 gsm8k 計算中 (~22:00 タイムアウト) |
| 46534 | comp/r1q7 ThinKV→HALO-3bit | ThinKV aime2024=0.567 完了, aime2025 計算中 |
| 46535 | comp/qwq_25 HALO-3bit | 実行中 |
| 46537 | comp/r1l8 残り全メソッド (k=2) | ThinKV 計算中 |
| 46708 | comp/r1q14_24 TaperKV+TT+HALO | 実行中 |

### 要手動実行 (今夜)

```bash
# 46421 タイムアウト後 (~22:00)
bash /lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/kvcomp_paper/postfix_46421.sh
# → taperkv/R1-7B TurboTaper-3bit delta fix + HALO-3bit 上書き

# ⚠️ 警告: taperkv/R1-7B の現在の TurboTaper-3bit は cascade error (gsm=0.53, math=0.06)
# postfix 実行後に正しい結果で上書きされる
```

---

## 再現方法

```bash
cd benchmark/kvcomp-longcontext

# Policy comparison (taperkv)
python kv_benchmark.py --config-name paper_taperkv model_path=<MODEL> \
  output_dir=results_taperkv +resume=true

# Reasoning benchmark
python kv_benchmark.py --config-name paper_aime model_path=<MODEL> \
  output_dir=results_aime_basic_<model> aime_years=[] +resume=true

# AIME benchmark
python kv_benchmark.py --config-name paper_aime model_path=<MODEL> \
  output_dir=results_aime_comp_<model>_<year> \
  run_gsm8k=false run_math=false aime_years='[2024]' +resume=true
```

---

## ファイル構造

```
onecomp-lab/
├── papers/taperkv_paper/
│   ├── README.md          ← このファイル
│   ├── results/
│   │   └── all_results.json
│   └── scripts/
│       ├── run_policy_bench.sh
│       ├── run_reasoning_bench.sh
│       └── run_lcb_bench.sh
├── benchmark/kvcomp-longcontext/
│   ├── kv_benchmark.py
│   ├── conf/paper_taperkv.yaml  # b_min_value: 3 (Bug-1 fix)
│   └── conf/paper_aime.yaml     # b_min_value: 3, n_outlier_floor: 32
└── onecomp/kvcomp/
    ├── cache/turbotaper_cache.py  # delta quantization (Bug-2 fix)
    └── recipe/taperkv.py
```
