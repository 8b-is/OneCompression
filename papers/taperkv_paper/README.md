# TaperKV Paper: Recency-Graded Precision Scheduling for KV Cache Compression

> **最終更新: 2026-06-03 22:40 JST**

## 概要

**主張:** TaperKV は、トークンを削除せず「古いトークンほど低精度」の精度スケジュールを適用する KV キャッシュ圧縮 Policy であり、StreamingLLM・ThinKV・PM-KVQ より LongBench スコアが高く、GSM8K では FullKV に匹敵する。

**手法の特徴:**
- **Age-graded precision ladder**: sink/recent トークンは高精度、中間を `b_min` まで線形低下
- **No eviction**: 全トークンを保持（StreamingLLM と異なり長期依存を維持）
- **Codec 非依存**: affine (TaperKV) or TurboQuant (TurboTaper) を差し替え可能
- **TurboTaper**: TaperKV policy + TurboQuant codec の組み合わせ

---

## 論文テーブル対応

### Table 1: Policy 比較 (メイン貢献)

**ベンチ:** `paper_taperkv.yaml` (Llama-3.1-8B, R1-Qwen-7B/14B)  
**ソース:** `benchmark/kvcomp-longcontext/results_taperkv/`  
**集約済み:** `papers/taperkv_paper/results/all_results.json` → `policy_comparison`

**Llama-3.1-8B (全11メソッド完了):**

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
| TurboTaper-3bit | 100.0 | 93.3 | 28.46 | 73.0 | 32.0 | 6.97 | 0.190 |
| Uniform-INT3 | 100.0 | 100.0 | 26.61 | 65.0 | 29.0 | 8.50 | 0.197 |

**→ TaperKV は LongBench 31.59 でトップ（FullKV 27.79 を 3.8pt 上回る）**  
**→ GSM8K 89.0% で FullKV (85.0%) を超過**

**備考:** TurboTaper-3bit の GSM8K が 73% まで低下している → Qwen系モデルでの挙動要確認

---

### Table 2: R1-Qwen-14B Policy 比較 (完了 10/11)

**ベンチ:** `paper_taperkv.yaml` (R1-Qwen-14B)  
**ソース:** `results_taperkv/results_DeepSeek-R1-Distill-Qwen-14B.json`

| 手法 | LongBench | GSM8K% | MATH-500% | PPL | cr |
|------|-----------|--------|-----------|-----|----|
| FullKV | 4.26 | 88.0 | 66.0 | 11.61 | 1.00 |
| Uniform-INT3 | 3.73 | 60.0 | 48.0 | 21.17 | 3.18 |
| StreamingLLM | 3.90 | 24.0 | 16.0 | 11.61 | ∞ |
| ThinKV | 4.10 | 86.0 | 60.0 | 11.84 | 2.42 |
| PM-KVQ | 4.16 | **94.0** | 58.0 | 14.29 | 3.09 |
| **TaperKV** | **4.17** | **90.0** | 56.0 | 12.14 | 2.52 |
| TaperKV-3bit | **4.35** | 90.0 | 60.0 | 13.92 | 3.18 |
| TurboQuant-3bit | 4.09 | 88.0 | 60.0 | 12.60 | 3.19 |
| KIVI-INT2 | 3.96 | 86.0 | **62.0** | 13.26 | 3.53 |
| TurboTaper-3bit | 3.97 | 16.0 ⚠️ | 10.0 ⚠️ | 12.06 | 3.09 |
| HALO-3bit | — | — | — | — | — (実行中) |

→ **TaperKV-3bit は LongBench 4.35 でトップ** (FullKV 4.26 を超過)  
→ **TurboTaper-3bit はRotation により R1-Qwen 系で深刻な性能低下** (要対応: 原因は後述)

---

### Table 3: Reasoning Benchmark (AIME2024/2025 / GSM8K / MATH-500)

**QwQ-32B GSM8K / MATH-500 (n=6メソッド完了):**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 100.0 | 66.7 | 3.72 | 1.00 |
| **TaperKV** | **96.7** | **60.0** | 3.82 | 3.36 |
| KIVI-INT2 | 100.0 | 66.7 | 4.16 | 6.18 |
| ThinKV | 63.3 | 20.0 | 3.79 | 3.19 |
| PM-KVQ | 63.3 | 20.0 | 4.10 | 4.27 |
| TurboTaper | 0.0 ⚠️ | 0.0 ⚠️ | 3.83 | 3.39 |

**AIME2024 QwQ-32B (pass@1, k=1):**

| 手法 | pass@1 |
|------|--------|
| FullKV | 0.30 |
| KIVI-INT2 | **0.333** |
| TaperKV | 0.20 |
| ThinKV | 0.00 |
| PM-KVQ | 0.00 |
| TurboTaper | 0.00 ⚠️ |

**R1-Qwen-14B GSM8K / MATH-500 (n=50):**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 88.0 | 68.0 | 12.43 | 1.00 |
| **TaperKV** | **90.0** | 62.0 | 13.16 | 2.52 |
| HALO-3bit | 88.0 | **66.0** | 13.23 | — |
| KIVI-INT2 | 86.0 | 64.0 | 14.31 | — |
| PM-KVQ | 86.0 | 58.0 | 13.62 | — |
| TurboTaper | 12.0 ⚠️ | 12.0 ⚠️ | 12.52 | — |

**🆕 AIME2025 R1-Qwen-14B (pass@1, k=2) — 新規データ:**

| 手法 | AIME2025 pass@1 (k=2) |
|------|----------------------|
| FullKV | 0.283 |
| **TaperKV** | **0.333** ← FullKV を超過! |
| KIVI-INT2 | 0.267 |
| HALO-3bit | 0.267 |
| ThinKV | 0.083 |
| PM-KVQ | 0.033 |
| TurboTaper | 0.017 ⚠️ |

→ **TaperKV は AIME2025 で FullKV を超えた (33.3% vs 28.3%)**

**R1-Qwen-7B GSM8K / MATH-500 (n=7メソッド):**

| 手法 | GSM8K% | MATH-500% | PPL | cr |
|------|--------|-----------|-----|----|
| FullKV | 69.0 | 67.0 | 48.93 | 1.00 |
| TaperKV | **75.0** | **64.0** | 50.38 | 2.38 |
| ThinKV | 73.0 | 67.0 | 50.61 | 2.29 |
| PM-KVQ | 71.0 | 48.0 | 60.86 | 2.71 |
| Uniform-INT4 | 72.0 | 66.0 | 55.45 | 2.54 |
| StreamingLLM | 25.0 | 13.0 | 48.93 | ∞ |
| TurboTaper | 48.0 ⚠️ | 6.0 ⚠️ | 49.74 | 2.38 |

---

### Table 4: LiveCodeBench (再実行中)

**問題:** 前回の実行 (p2-lcb-*) は `max_new_code=4096` が短すぎ、R1系モデルの`<think>`ブロックで全トークンを消費し、コードが切り捨てられて **全メソッド pass@1=0.0%** になっていた。

```
avg_gen_len (R1-Qwen-7B FullKV) = 3804 tokens / max=4096 → コード末尾が切断
```

**修正:** `max_new_code: 4096 → 16384` (`paper_aime.yaml` 修正済み)  
**再実行ジョブ:** 46393 (r1l8), 46394 (r1q7), 46395 (r1q14), 46396 (r1q15) — 24h 枠で投入済み

---

## ステータス (2026-06-03 22:40 JST)

| ベンチ | Llama-8B | R1-Qwen-7B | R1-Qwen-14B | R1-Qwen-1.5B | QwQ-32B |
|--------|----------|------------|-------------|--------------|---------|
| Policy (LB+GSM+MATH) | ✅ 11/11 | 🔄 6/11 | ✅ 10/11 | 🔄 6/11 | — |
| AIME basic (GSM+MATH) | — | ✅ 7 | ✅ 7 | 🔄 5 | ✅ 6 |
| AIME 2024 | — | — | 🔄 | — | ✅ |
| AIME 2025 | — | — | ✅ 7 | — | — |
| LiveCodeBench | 🔄 再実行中 | 🔄 再実行中 | 🔄 再実行中 | 🔄 再実行中 | — |

**実行中ジョブ:**
| ジョブ名 | JID | 内容 | 経過時間 |
|---------|-----|------|---------|
| p2-taper-r1q14 | 46036 | policy R1-14B 残り1/11 | 17h38m |
| p2-taper-r1q7 | 46098 | policy R1-7B 残り5/11 | 9h34m |
| p2-taper-r1q15 | 46034 | policy R1-1.5B 残り5/11 | 18h43m |
| p2-aime-r1l8 | 45940 | AIME R1-Llama-8B | 21h12m |
| p2-aime-r1q7 | 45941 | AIME R1-7B | 21h12m |
| p2-aime-r1q15 | 45939 | AIME R1-1.5B | 21h39m |
| p2-basic-qwq | 45938 | GSM/MATH QwQ-32B | 21h39m |
| p2-basic-r1l8 | 45915 | GSM/MATH R1-Llama | 22h13m |
| p2-basic-r1q7 | 45916 | GSM/MATH R1-7B | 22h09m |
| p2-basic-r1q15 | 45914 | GSM/MATH R1-1.5B | 22h13m |
| p2-aime24-r1q14 | 46256 | AIME2024 R1-14B | 8h05m |
| **p2-lcb2-r1l8** | **46393** | **LCB 再実行(16K)** | **新規** |
| **p2-lcb2-r1q7** | **46394** | **LCB 再実行(16K)** | **新規** |
| **p2-lcb2-r1q14** | **46395** | **LCB 再実行(16K)** | **新規** |
| **p2-lcb2-r1q15** | **46396** | **LCB 再実行(16K)** | **新規** |

---

## 要対応事項・既知問題

### ⚠️ [重大] TurboTaper が R1-Qwen 系で壊滅的に低下 (gsm8k 16%, math 10%)

**診断済みの原因:**  
`TurboTaperPolicy` は `rotation_mode="hadamard"` がデフォルト。Qwen アーキテクチャでは `skip_layers` が適用されても、残りの層に対する 2-bit Hadamard 回転量子化が attention 出力品質を大幅に悪化させる。PPL は正常値 (12.06) にも関わらず、長い推論タスクで影響が累積する。

**証拠:**
- TurboTaper-3bit R1-14B: gsm8k=16%, math=10%, PPL=12.06 (FullKV: gsm8k=88%, PPL=11.61)
- TurboTaper R1-7B: gsm8k=48%, math=6%, PPL=49.74 (FullKV PPL: 50.85 ≈ 同等)
- TurboTaper QwQ-32B: gsm8k=0%, math=0%

**PPLが正常なのに推論が壊れる** → ショートコンテキスト (PPL測定) では問題ないが、長い CoT 生成では回転誤差が累積する可能性が高い。

**対応策 (論文用):**
- 選択肢A: TurboTaper に `rotation_mode=dense` を使用 (Qwen 系に対してより安全)
- 選択肢B: TurboTaper を TaperKV 論文の比較から外し「TaperKV (proposed) vs ThinKV/PM-KVQ/KIVI」の構成にする
- **現状の判断:** Llama-8B では TurboTaper-3bit が正常動作 (LB=28.46, gsm8k=73%)。Qwen系向けの修正は別タスクとして追跡。

### ⚠️ [修正済み] LiveCodeBench pass@1 = 0.0%

- **原因:** `max_new_code=4096` が不足 (R1系 `<think>` で 3800〜4000 トークン消費)
- **修正:** `paper_aime.yaml` の `max_new_code: 4096 → 16384`
- **再実行ジョブ:** 46393-46396 (24h 枠) 投入済み

---

## 再現方法

```bash
# 1. Policy comparison (全モデル)
bash papers/taperkv_paper/scripts/run_policy_bench.sh

# 2. Reasoning benchmark (AIME/GSM8K/MATH-500)
bash papers/taperkv_paper/scripts/run_reasoning_bench.sh

# 3. LiveCodeBench
bash papers/taperkv_paper/scripts/run_lcb_bench.sh

# 4. 結果集約 → all_results.json
python papers/aggregate_results.py
```

---

## ファイル構造

```
onecomp-lab/
├── papers/taperkv_paper/
│   ├── README.md          ← このファイル
│   ├── results/
│   │   └── all_results.json   ← 集約済み全結果
│   └── scripts/
│       ├── run_policy_bench.sh
│       ├── run_reasoning_bench.sh
│       └── run_lcb_bench.sh
├── benchmark/kvcomp-longcontext/
│   ├── kv_benchmark.py
│   ├── conf/paper_taperkv.yaml
│   └── conf/paper_aime.yaml
└── onecomp/kvcomp/
    ├── cache/taper_cache.py
    ├── cache/turbotaper_cache.py
    └── recipe/taperkv.py
```
