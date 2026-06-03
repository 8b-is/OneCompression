# TaperKV Paper: Recency-Graded Precision Scheduling for KV Cache Compression

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

### Table 2: Reasoning Benchmark (AIME / GSM8K / MATH-500)

**ベンチ:** `paper_aime.yaml` (R1-Distill モデル, QwQ-32B)  
**ソース:** `benchmark/kvcomp-longcontext/results_aime_*/`

**QwQ-32B GSM8K / MATH-500 (完了):**

| 手法 | GSM8K% | MATH-500% | PPL |
|------|--------|-----------|-----|
| FullKV | 100.0 | 66.7 | 3.724 |
| TaperKV | 96.7 | 60.0 | 3.824 |
| ThinKV | 63.3 | 20.0 | 3.79 |
| PM-KVQ | — | — | — |
| TurboTaper | 0.0 | 0.0 | 3.834 |

**⚠️ TurboTaper が QwQ-32B で 0.0 になっている** — 重大バグの可能性（確認要）

**AIME2024 QwQ-32B (pass@1, k=1):**

| 手法 | pass@1 |
|------|--------|
| FullKV | 0.30 |
| KIVI-INT2 | 0.333 |
| TaperKV | 0.20 |
| ThinKV | 0.00 |
| PM-KVQ | 0.00 |
| TurboTaper | 0.00 |

**R1-Qwen-14B GSM8K / MATH-500 (完了):**

| 手法 | GSM8K% | MATH-500% |
|------|--------|-----------|
| FullKV | 88.0 | 68.0 |
| HALO-3bit | 88.0 | 66.0 |
| KIVI-INT2 | 86.0 | 64.0 |
| TaperKV | 90.0 | 62.0 |
| PM-KVQ | 86.0 | 58.0 |
| TurboTaper | 12.0 | 12.0 |

**→ HALO-3bit は R1-Qwen-14B で FullKV に匹敵** (HALO論文でも使える)  
**→ TurboTaper は R1系で深刻な性能低下** — 原因調査が必要

---

### Table 3: LiveCodeBench (進行中)

**ソース:** `results_aime_comp_*/` (現在実行中: `p2-lcb-*` ジョブ)

現時点の部分結果 (R1-Llama-8B, r1-qwen-7b):

| 手法 | LCB pass@1 |
|------|-----------|
| FullKV (r1-llama-8b) | 完了待ち |
| TaperKV (r1-llama-8b) | 実行中 |

---

## ステータス (2026-06-03 時点)

| ベンチ | Llama-8B | R1-Qwen-7B | R1-Qwen-14B | R1-Qwen-1.5B | QwQ-32B |
|--------|----------|------------|-------------|--------------|---------|
| Policy (Needle+LB+GSM) | ✅ 11/11 | 🔄 6/11 | 🔄 8/11 | 🔄 5/11 | — |
| AIME basic (GSM+MATH) | ✅ | 🔄 6/9 | ✅ 7/7 | 🔄 4/9 | 🔄 5/9 |
| AIME 2024 | — | 🔄 | 🔄 | 🔄 | ✅ |
| LiveCodeBench | 🔄 実行中 | 🔄 実行中 | 🔄 実行中 | 🔄 実行中 | — |

**実行中ジョブ (2026-06-03):**
- `p2-taper-r1q14` (13h): policy R1-Qwen-14B 残り3メソッド
- `p2-taper-r1q7` (9h): policy R1-Qwen-7B 残り5メソッド
- `p2-taper-r1q15` (14h): policy R1-Qwen-1.5B 残り6メソッド
- `p2-aime-r1l8`, `p2-aime-r1q7`, etc.: AIME補完
- `p2-lcb-r1l8`, `p2-lcb-r1q7`, `p2-lcb-r1q14`, `p2-lcb-r1q15`: LiveCodeBench

---

## 要対応事項

1. **TurboTaper が QwQ-32B / R1-Qwen で 0.0 になる問題**  
   - 推定原因: Qwen系モデルの外れ値チャネルに対して `skip_layers` が未設定  
   - 対応: `TurboTaperCache` + `detect_outlier_layers` の適用確認

2. **TaperKV R1-Qwen モデルのジョブ完了待ち**  
   - `p2-taper-*` が完了次第 `aggregate_results.py` を再実行

3. **LiveCodeBench の全結果待ち**  
   - `p2-lcb-*` ジョブ完了後に集約

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
