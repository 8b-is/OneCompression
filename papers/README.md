# KVComp 論文群 — 結果・再現スクリプト

このディレクトリには、**3本の論文**に対応するベンチマーク結果と再現スクリプトを整理しています。

> **RULER naming boundary.** 旧 `run_ruler` の数値はローカル synthetic
> multi-key diagnostic であり、RULERベンチマークではありません。刷新後の正式軸は
> **NeMo-Skills RULER v1 100-sample/task protocol** と明記し、canonical
> NVIDIA/RULER v1 500-sample/task run とは区別します。

---

## 論文0: HayateQuant — Codec 論文

HayateQuant は data-free block-VQ、folded hybrid rotation、role-asymmetric
scale、および exact assignment を組み合わせる独立した codec recipe です。
`block_size=1` と rotation/source-law/scale を明示的に一致させた構成は
TurboQuant/HALO の選択された scalar path を検査する compatibility control
ですが、完全な上位互換や全データ・全指標での優越を意味しません。

- **ディレクトリ:** `papers/hayatequant_paper/`
- **再評価方針:** 過去の未 fingerprint の数値は、現在のコード・dataset
  fingerprint・sample denominator・hardware provenance を備えた再実行結果に
  置き換えてから引用します。この overview には旧 headline 値を残しません。
- **fixed HQ recipe:** `block_size=2`、Gaussian source codebooks、exact assignment、
  K5/V3・K7/V5・K9/V7 block-code、key NC、value LS-reassign/refit を全モデルで
  固定します。`AdaptiveHQ` だけが opt-in proposal です。
- **内部 proxy:** `HQ-Fib-init-proxy` と `HQ-Block-scale-proxy` は HQ 内部の
  ablation であり、独立した FibQuant/BlockQuant 実装ではありません。
- **HALO との関係:** HALO は backward-compatible baseline として残します。

再現手順と provenance contract は `papers/hayatequant_paper/` および
`benchmark/kvcomp-codec/README.md` を参照してください。

---

## 論文1: HALO — Codec 論文 (旧提案手法、HayateQuant に更新された)

> **主張:** HALO は Hadamard-GEMM による高速回転 + LS-scale 最適逆量子化により、
> TurboQuant に対して Speed-Quality Pareto 優位を達成する、データ不要の KV Codec。

- **ディレクトリ:** `papers/halo_paper/`
- **現在の状態:** Llama-8B / Ministral-8B で全ベンチ完了
- **未解決:** R1-Qwen 系の codec PPL 異常高 (outlier layers 要確認)
- **注記 (2026-07-04):** HALO の role-asymmetric scale (NC/LS) レバーは HayateQuant に
  引き継がれ、さらに block-VQ + hybrid rotation で歪みが更新された。本論文はアーカイブ
  として残すが、新しい主張は `papers/hayatequant_paper/` を参照。

```
papers/halo_paper/
├── README.md          ← 詳細説明・テーブル
├── results/
│   └── all_results.json   ← 集約済み全結果
└── scripts/
    ├── run_codec_bench.sh      # PPL + latency
    ├── run_longcontext_bench.sh # Needle/local synthetic multi-key/LongBench
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

> **再評価中:** 旧needleは実入力長・depth・予測を保存していなかったため、needle v2 の
> 再実行と実測メモリの照合が完了するまで、iso-memory優位・LongBench優位を主張しない。

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

**旧スナップショット（needle v2以前のため論文値として引用不可）:**

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

## 3本の論文の切り分け方針

| 観点 | HayateQuant論文 | HALO 論文 | TaperKV 論文 |
|------|-----------------|-----------|--------------|
| 主題 | Codec (圧縮変換方式、HALO後継) | Codec (圧縮変換方式、アーカイブ) | Policy (精度スケジュール) |
| 比較相手 | TurboQuant, HALO, Uniform-INT, FP8, KIVI | TurboQuant, Uniform-INT, FP8, KIVI | StreamingLLM, ThinKV, PM-KVQ |
| 評価モデル | R1-Qwen-1.5B, Llama-3.1-8B-Instruct | Llama-8B, Ministral-8B (非R1) | Llama-8B + R1-Distill系 + QwQ-32B |
| 主要メトリクス | dist(K/V), PPL, encode ms, TTFT, tok/s, peak-mem | PPL, Needle, local multi-key (legacy), encode ms, relMSE | LongBench, GSM8K, MATH-500, AIME, LCB |
| HayateQuant の位置づけ | 提案手法 (メイン、現在の推奨) | 登場しない | 登場しない |
| HALO の位置づけ | backward-compatible 比較 baseline (strict-superset 関係は主張しない) | 提案手法 (アーカイブ扱い) | 比較用 codec baseline として登場 |
| TaperKV の位置づけ | 登場しない | 登場しない | 提案手法 (メイン) |
| TurboTaper の位置づけ | 登場しない | 登場しない | TaperKV + TurboQuant の組み合わせ variant |

---

## 完了したジョブ (2026-07-04)

> **Legacy development record.** 以下の job ID、数値、v3/v5 の名称は旧
> auto/LUT recipe と旧 harness の履歴です。refreshed paper の HQ または
> AdaptiveHQ の結果として引用してはいけません。正式値は repo 内へ archive
> された coverage-complete formal run と generated table だけから更新します。

| ジョブ名 | JID | 内容 | 状態 |
|---------|-----|------|------|
| kv-hayatequant | 58907 | legacy long-context (Needle/local multi-key/LongBench-E/PPL) | ✅ 完了 (7h46m) |
| hayatequant-v3 | 58931 | v3 品質ベンチ (auto/exact/K12/dense-LS, 2モデル) + 速度 | ✅ 完了 |
| hayatequant-speed-v3b | 58938 | v3b 最終速度 (adaptive encoder + fp16 探索) + GPUパリティ | ✅ 完了 |

結果は `papers/hayatequant_paper/README.md` に反映済み。

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

4. **HayateQuant の残存コスト (v5 で解消)**  
   - v3 の Folded Rotation + 候補リスト LUT + fp16 探索で decode は TurboQuant の
     1.4 倍まで改善したが、2-bit prefill の TTFT 2.6-2.9 倍が残っていた。
   - v5 (LUT T=2 スペック + int16 LUT + `take` ギャザー経路) で 2-bit prefill
     enc 5.2 倍・dec 8.7 倍高速化し、**全ビット幅で TTFT が TurboQuant と同等
     (+2〜6%) に到達**。品質コストは 2-bit auto の PPL +0.075 (9.07→9.15) のみで、
     2-bit 推奨バリアント dense-LS (PPL 8.96) には影響なし。
