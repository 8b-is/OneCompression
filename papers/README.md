# KVComp 論文群 — 結果・再現スクリプト

このディレクトリには、**2本の論文**に対応するベンチマーク結果と再現スクリプトを整理しています。

---

## 論文1: HALO — Codec 論文

> **主張:** HALO は Hadamard-GEMM による高速回転 + LS-scale 最適逆量子化により、
> TurboQuant に対して Speed-Quality Pareto 優位を達成する、データ不要の KV Codec。

- **ディレクトリ:** `papers/halo_paper/`
- **現在の状態:** Llama-8B / Ministral-8B で全ベンチ完了
- **未解決:** R1-Qwen 系の codec PPL 異常高 (outlier layers 要確認)

```
papers/halo_paper/
├── README.md          ← 詳細説明・テーブル
├── results/
│   └── all_results.json   ← 集約済み全結果
└── scripts/
    ├── run_codec_bench.sh      # PPL + latency
    ├── run_longcontext_bench.sh # Needle/RULER/LongBench
    └── run_speed_bench.sh      # encode/decode ms
```

**主要結果:**

| Model | bits | 手法 | PPL | enc ms | 速度比 |
|-------|------|------|-----|--------|--------|
| Llama-8B | 3 | HALO-hadamard_gemm | 7.09 | **0.26ms** | 3.9× vs TQ-hadamard |
| Llama-8B | 2 | HALO-dense-LS | (retr) needle=100% | — | TQ-2bit needle=88.9% |
| Ministral-8B | 3 | HALO-hadamard_gemm | **9.40** | **0.28ms** | 3.8× |

---

## 論文2: TaperKV — Policy 論文

> **主張:** TaperKV の age-graded 精度スケジュールは、StreamingLLM/ThinKV/PM-KVQ に対して
> 同等メモリ予算で LongBench スコアを最大 +4pt 向上させ、GSM8K では FullKV を超える。

- **ディレクトリ:** `papers/taperkv_paper/`
- **現在の状態:** Llama-8B 完了、R1-Qwen 系・AIME・LCB は実行中

```
papers/taperkv_paper/
├── README.md          ← 詳細説明・テーブル
├── results/
│   └── all_results.json   ← 集約済み全結果
└── scripts/
    ├── run_policy_bench.sh     # Policy 比較
    ├── run_reasoning_bench.sh  # AIME/GSM8K/MATH/LCB
    └── run_lcb_bench.sh        # LiveCodeBench単独
```

**主要結果 (Llama-8B, 完了):**

| 手法 | LongBench | GSM8K% | MATH% | Needle% | mem |
|------|-----------|--------|-------|---------|-----|
| FullKV | 27.79 | 85.0 | 55.0 | 100.0 | 1.00 |
| **TaperKV** | **31.59** | **89.0** | 44.0 | **100.0** | 0.293 |
| ThinKV | 26.54 | 61.0 | 22.0 | 100.0 | 0.313 |
| PM-KVQ | 19.12 | 86.0 | 45.0 | 100.0 | 0.204 |
| StreamingLLM | 24.81 | 36.0 | 25.0 | 20.0 | 0.209 |

---

## 結果集約・更新方法

```bash
# 最新の全 JSON を読み込み all_results.json を再生成 + テーブル表示
python papers/aggregate_results.py
```

ジョブが完了するたびにこのコマンドを実行することで、最新状況を確認できます。

---

## 2本の論文の切り分け方針

| 観点 | HALO 論文 | TaperKV 論文 |
|------|-----------|--------------|
| 主題 | Codec (圧縮変換方式) | Policy (精度スケジュール) |
| 比較相手 | TurboQuant, Uniform-INT, FP8, KIVI | StreamingLLM, ThinKV, PM-KVQ |
| 評価モデル | Llama-8B, Ministral-8B (非R1) | Llama-8B + R1-Distill系 + QwQ-32B |
| 主要メトリクス | PPL, Needle, RULER, encode ms, relMSE | LongBench, GSM8K, MATH-500, AIME, LCB |
| HALO の位置づけ | 提案手法 (メイン) | 比較用 codec baseline として登場 |
| TaperKV の位置づけ | 登場しない (policy 論文扱い) | 提案手法 (メイン) |
| TurboTaper の位置づけ | 登場しない | TaperKV + TurboQuant の組み合わせ variant |

---

## 現在実行中のジョブ (2026-06-03 18:00 JST)

| ジョブ名 | 内容 | 完了時刻見込み |
|---------|------|--------------|
| p2-taper-r1q14 | policy R1-Qwen-14B (残り3/11) | 数時間後 |
| p2-taper-r1q7 | policy R1-Qwen-7B (残り5/11) | 数時間後 |
| p2-taper-r1q15 | policy R1-Qwen-1.5B (残り6/11) | 数時間後 |
| p2-aime-* | AIME/GSM/MATH 補完 | 数時間後 |
| p2-lcb-* | LiveCodeBench (r1l8/r1q7/r1q14/r1q15) | 本日中 |
| p2-aime25-r1q14 | AIME2025 R1-Qwen-14B | 数時間後 |

---

## 既知の問題

1. **TurboTaper on QwQ-32B: GSM8K/MATH-500 = 0.0**  
   - `n=30` × `k=1` のサンプル数が少なく統計的ばらつきが大きい  
   - かつ `b_min=2` (values) の2ビット量子化が 32B モデルで品質低下  
   - **対応:** TurboTaper を `b_min=3/4` または `n_outlier_floor=32` 付きで再実行予定

2. **R1-Qwen 系の codec PPL 異常高 (R1-Qwen-1.5B: PPL=17424, R1-Qwen-7B: PPL=4401)**  
   - 外れ値チャネルが原因。`kv_benchmark.py` では `skip_layers` 適用済みだが  
     `codec_benchmark.py` での確認が必要  
   - **対応:** `codec_benchmark.py` の `skip_outliers=true` を設定して再実行
