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

## 現在実行中のジョブ (2026-06-03 22:40 JST)

| ジョブ名 | JID | 内容 | 経過 | 状態 |
|---------|-----|------|------|------|
| p2-taper-r1q14 | 46036 | policy R1-14B 残り1/11 | 17h38m | 🔄 |
| p2-taper-r1q7 | 46098 | policy R1-7B 残り5/11 | 9h34m | 🔄 |
| p2-taper-r1q15 | 46034 | policy R1-1.5B 残り5/11 | 18h43m | 🔄 |
| p2-aime-r1l8/r1q7/r1q15 | 45940/41/39 | AIME/GSM/MATH 補完 | 21h+ | 🔄 |
| p2-basic-* | 45914-16,38 | GSM/MATH 補完 | 22h+ | 🔄 |
| p2-aime24-r1q14 | 46256 | AIME2024 R1-14B | 8h05m | 🔄 |
| **p2-lcb2-r1l8/r1q7/r1q14/r1q15** | **46393-96** | **LCB 再実行(16K token)** | 新規 | 🔄 |

---

## 新規・更新データ (2026-06-03 22:40 時点)

| データ | 内容 | 状態 |
|--------|------|------|
| R1-Qwen-14B policy | 10/11 メソッド完了 (HALO-3bit のみ残) | ✅ |
| AIME2025 R1-Qwen-14B | TaperKV=33.3% > FullKV=28.3% (k=2) | ✅ |
| QwQ-32B GSM/MATH | 6メソッド (TurboTaper=0.0% 問題確認) | ✅ |
| R1-Qwen-7B GSM/MATH | 7メソッド (TurboTaper=48%/6% 問題確認) | ✅ |
| LCB 全モデル | 0.0%→ **max_new_code修正後に再実行中** | 🔄 |

---

## 既知の問題

1. **TurboTaper on Qwen系: 壊滅的性能低下**  
   - 根本原因特定済み: Hadamard rotation が Qwen KV 分布に悪影響
   - Llama-8B では正常動作 (LB=28.46, GSM8K=73%)
   - **TaperKV 論文方針:** TurboTaper は Llama-8B の結果のみ示すか、`dense` rotation を使う

2. **LCB pass@1 = 0.0% (修正済み)**  
   - 原因: `max_new_code=4096` 不足 (R1 思考で全消費)
   - 修正: `paper_aime.yaml` を 16384 に更新
   - 再実行ジョブ: 46393-46396 投入済み

3. **R1-Qwen 系の codec PPL 異常高**  
   - 外れ値チャネルが原因。`codec_benchmark.py` での skip_layers 適用が必要
