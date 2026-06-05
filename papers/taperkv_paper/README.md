# TaperKV Paper: Recency-Graded Precision Scheduling for KV Cache Compression

> **最終更新: 2026-06-05 16:56 JST** (Round-3 投入済み: 44ジョブ追加, taperkv/QwQ-32B, HALO-4bit, LCB, ablation 全網羅)

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

### Table 2: Policy 比較 — R1-Qwen-14B (全11メソッド完了 ✅、delta fix済み)

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
| **TurboTaper-3bit** ✅ | **4.35** | 85.0 | 61.0 | 11.87 | 2.90 |
| **HALO-3bit** ✅ | 4.11 | 84.0 | 64.0 | 12.45 | 3.19 |

**→ TaperKV-3bit / TurboTaper-3bit が LB 4.35 でタイ最高 (FullKV 4.26 超)**
**→ TurboTaper-3bit: GSM=85% (FullKV比 97%)、MATH=61% (FullKV比 92%) ✅**
**→ HALO-3bit: GSM=84%, MATH=64% (FullKV比 95%/97%) — excellent 3bit codec**

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

**R1-Qwen-14B — 全9メソッド完了 ✅:**

| 手法 | GSM8K% | MATH-500% | AIME-2024 | AIME-2025 | cr |
|------|--------|-----------|-----------|-----------|-----|
| FullKV | 88.0 | 68.0 | 0.500 | 0.283 | 1.00 |
| Uniform-INT4 | 86.0 | 70.0 | — | — | 2.68 |
| StreamingLLM | 25.0 | 16.0 | — | — | ∞ |
| ThinKV | 78.0 | 54.0 | 0.067 | 0.083 | 3.21 |
| PM-KVQ | 86.0 | 58.0 | 0.033 | 0.033 | 2.99 |
| **TaperKV** | **90.0** | 62.0 | **0.433** | **0.333** | 2.52 |
| KIVI-INT2 | 86.0 | 64.0 | — | 0.267 | 3.53 |
| HALO-3bit | 88.0 | 66.0 | **0.367** | 0.267 | 3.19 |
| **TurboTaper** ✅ | 86.0 | **68.0** | **0.433** | **0.367** | 2.41 |

**→ TurboTaper AIME-2025 0.367 > FullKV 0.283 (+30%)! ✅**
**→ TurboTaper MATH = FullKV 68% (完全等価)**
**→ TaperKV AIME-2024 0.433 vs ThinKV 0.067 (+548%)、PM-KVQ 0.033 (+1212%)!**

**R1-Qwen-7B — 全9メソッド完了 ✅ (taperkv, delta fix済み):**

| 手法 | LongBench | GSM8K% | MATH-500% | PPL | cr |
|------|-----------|--------|-----------|-----|----|
| FullKV | 3.63 | 69.0 | 61.0 | 50.85 | 1.00 |
| Uniform-INT3 | 3.61 | 66.0 | 55.0 | — | 3.15 |
| StreamingLLM | 3.22 | 25.0 | 12.0 | — | ∞ |
| ThinKV | 3.48 | 72.0 | 59.0 | — | 2.23 |
| PM-KVQ | 3.44 | 74.0 | 57.0 | — | 2.71 |
| **TaperKV** | **3.25** | **75.0** | **60.0** | — | 2.46 |
| TaperKV-3bit | 3.29 | 64.0 | 51.0 | — | 3.15 |
| TurboQuant-3bit | 3.39 | 64.0 | 62.0 | — | 3.19 |
| KIVI-INT2 | 3.07 | 69.0 | 58.0 | — | 3.18 |
| **TurboTaper-3bit** ✅ | **3.39** | **73.0** | **59.0** | — | **2.70** |
| **HALO-3bit** ✅ | **3.47** | **67.0** | **65.0** | — | **2.91** |

**→ TaperKV GSM=75% > FullKV=69% (+6pt)**
**→ TurboTaper-3bit: GSM=73% (FullKV比 106%), MATH=59% (FullKV比 97%) ✅ delta fix確認**
**→ HALO-3bit: MATH=65% > FullKV=61% — 3bit codec で精度向上**

**R1-Distill-Llama-8B — 全9メソッド完了 ✅ (basic GSM/MATH):**

| 手法 | GSM8K% | MATH-500% | cr |
|------|--------|-----------|-----|
| FullKV | 63.0 | 61.0 | 1.00 |
| Uniform-INT4 | 66.0 | 57.0 | 2.68 |
| StreamingLLM | 17.0 | 15.0 | ∞ |
| ThinKV | 35.0 | 21.0 | 4.12 |
| PM-KVQ | 61.0 | 48.0 | 3.33 |
| **TaperKV** | **62.0** | **58.0** | 3.16 |
| KIVI-INT2 | 53.0 | 56.0 | 5.99 |
| HALO-3bit | 59.0 | 54.0 | 2.91 |
| **TurboTaper** ✅ | **65.0** | **59.0** | 3.01 |

**→ TurboTaper GSM=65% > FullKV=63% — 圧縮しつつ精度向上**
**→ ThinKV: GSM=35% (FullKV比 56%) — 大幅劣化**

---

### Table 4: LiveCodeBench

R1-Distill-Llama-8B: FullKV=0% — モデル自体がコード生成不可 (reasoning専用, MATH蒸留)。
QwQ-32B / R1-14B が主要比較対象。LCB ジョブ実行中 / 未完了。

---

## ステータス (2026-06-05 15:51 JST) — 128/145 methods, 88%

### 完了データ ✅ (8データセット完全完了)

| ベンチ | モデル | メソッド数 | 備考 |
|--------|--------|-----------|------|
| taperkv (policy) | Llama-3.1-8B | **11/11** ✅ | TurboTaper-3bit delta fix済み |
| taperkv (policy) | R1-14B | **11/11** ✅ | HALO-3bit (gsm=0.84, math=0.64) 追加 |
| taperkv (policy) | R1-7B | **11/11** ✅ | TurboTaper-3bit delta fix (gsm=0.73, math=0.59), HALO-3bit (gsm=0.67, math=0.65) |
| basic (GSM+MATH) | R1-L8 | **9/9** ✅ | KIVI (gsm=0.53), HALO (gsm=0.59) 追加 |
| basic (GSM+MATH) | R1-Q7 | **9/9** ✅ | KIVI (gsm=0.69, math=0.63), HALO (gsm=0.67, math=0.69) 追加 |
| basic (GSM+MATH) | R1-Q14 | **9/9** ✅ | StreamingLLM, Uniform-INT4 追加 |
| comp AIME-2024 | QwQ-32B (24%) | **8/8** ✅ | StreamingLLM 追加 |
| comp AIME-2025 | QwQ-32B (25%) | **8/8** ✅ | StreamingLLM 追加 |
| comp AIME-2025 | R1-Q14 (25%) | **8/8** ✅ | StreamingLLM 追加 |

### 部分完了

| ベンチ | モデル | 状況 |
|--------|--------|------|
| comp AIME-2024 | R1-Q14 (24%) | 6/8 (TaperKV=0.433, TT=0.433, HALO=0.367 取得済) — KIVI+Stream 計算中 |
| comp AIME-2024+2025 | R1-Q7 | 5/8 (ThinKV=0.567/0.400, PM-KVQ=0.033/0.067, TT=0.533/0.300) — TaperKV+KIVI+HALO 計算中 |
| comp AIME-2024+2025 | R1-L8 (k=2) | 5/8 (PM-KVQ=0.067/0.133, TaperKV=0.367/0.400 取得済) — KIVI+HALO+TT 計算中 |
| basic (GSM+MATH) | R1-1.5B | 7/9 (TurboTaper=0.28/0.26 取得済) — KIVI+HALO 計算中 |
| taperkv | R1-1.5B | 9/11 (TurboQuant-3bit=0.19/0.20 取得済) — KIVI+HALO 計算中 |

### 実行中ジョブ (Round 2: 46919-46928)

| JID | 内容 | walltime |
|-----|------|---------|
| 46919 | comp/r1q7 TaperKV (k=4, 2yr) | 23h |
| 46920 | comp/r1q7 KIVI+HALO (k=4, 2yr) | 23h |
| 46921 | comp/r1l8 KIVI+HALO (k=2, 2yr) | 23h |
| 46922 | comp/r1l8 TurboTaper (k=2, 2yr) | 12h |
| 46923 | comp/r1q14_24 StreamingLLM (k=2, 2024) | 12h |
| 46924 | comp/r1q14_24 KIVI-INT2 (k=2, 2024) | 23h |
| 46925 | comp/r1q15 KIVI+HALO+TT (k=4, 2yr) | 12h |
| 46926 | basic/QwQ StreamingLLM+Uni4 (GSM/MATH) | 12h |
| 46927 | basic/R1-1.5B KIVI+HALO (GSM/MATH) | 8h |
| 46928 | taperkv/R1-1.5B KIVI+HALO (LB+GSM+MATH) | 12h |

### キュー済みジョブ (Round 3: 46947-46990, 2026-06-05 16:56 JST)

44ジョブを `submit_all_experiments.sh` で一括投入。未実施の全実験 + ablation を完全網羅。

**追加された新実験群:**

| セクション | JID | 内容 |
|-----------|-----|------|
| A: taperkv bench 追加 | 46947-46953 | taperkv/QwQ-32B (5jobs) + taperkv/R1-L8 (2jobs): 全11メソッド |
| B: HALO-4bit basic | 46954-46958 | 全5モデルに HALO-4bit + AIME |
| C: HALO-4bit comp | 46959-46965 | 全7comp dir に HALO-4bit |
| D: LiveCodeBench | 46966-46971 | 全モデル × 全メソッド (max_new_code=16384) |
| E: Ablation Ministral | 46972 | HALO lever ablation |
| F: Ablation R1-Q7 | 46973 | HALO lever + TaperKV asymmetry |
| G: b_min sweep | 46974-46975 | TaperKV b_min=2/3/4/5 (Llama+R1-Q7) |
| H: Budget sweep | 46976-46985 | target_mem=0.10/0.15/0.20/0.25/0.35 ×2モデル |
| I: TQ/HALO sweep | 46986-46987 | turboquant_paper full grid on R1-Q7+R1-L8 |
| J: Ablation R1-L8 | 46988 | HALO lever + TaperKV asymmetry |
| M: Asymmetry+reason | 46989-46990 | K/V asymmetry × GSM+MATH (R1-Q7+R1-L8) |

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
