# KVComp 論文群 — 結果・再現スクリプト

このディレクトリには、**3本の論文**に対応するベンチマーク結果と再現スクリプトを整理しています。

---

## 論文0: HayateQuant — Codec 論文 (HALO の後継・現在の推奨手法)

> **主張:** HayateQuant は TurboQuant と HALO の完全な上位互換 (`block_size=1` で両者を
> bit-for-bit・速度も同等に再現) であり、(1) データ不要 block vector quantization、
> (2) 単一 GEMM に折り畳んだ Hadamard+block-dense ハイブリッド回転 (Folded Rotation)、
> (3) 候補リスト Grid-LUT による符号帳サイズ非依存 O(T) エンコーダ、により
> 同一ビット数で両者より厳密に低い歪み・PPL・下流タスク精度を達成しつつ、
> **decode スループットで TurboQuant の 1.4 倍、TTFT も 3/4-bit で同等**を実現する。

- **ディレクトリ:** `papers/hayatequant_paper/`
- **現在の状態 (v4, 2026-07-04):** 品質(歪み/PPL, 2モデル)・速度(codec/e2e)・
  paper-level ロングコンテキスト(Needle/RULER/LongBench-E/PPL, job 58907)まで**全て完了**。
  v4 監査で `dist="beta"` の正規化不一致 (CRITICAL) 等を修正し、HALO 主体だった全実験
  設定・submit スクリプトを HayateQuant 主体に世代交代 (詳細は paper README の v4 節)。
  v4 再検証ベンチ: jobs 59000-59004。
- **HALO との関係:** HALO はコードベースに残置(削除しない)。比較対象として恒久保持。
- **vLLM プラグイン:** `mapping.py` に `HayateQuantCodec` 対応 + デフォルト設定での
  `int(None)` クラッシュ修正 + 専用テスト 9 件 (`tests/vllm_plugins/kvcomp/`)。
  `vllm.general_plugins` エントリポイント登録済み(EngineArgs 自動フック)。
- **v2 での重大性能バグ修正:** block_size=1 の非効率パス + 無制限符号帳サイズによる
  TTFT 21 倍/+32GB リグレッションを発見・修正(経緯は paper README)。
- **v3 システム最適化 (ログ分析起点):** decode 0.7 倍・TTFT 2.5-4.3 倍という残存弱点を
  Folded Rotation(逆回転を 1 GEMM 化、per-layer dequant 8 倍速)+ 候補リスト LUT
  (エンコード O(K)→O(T))+ fp16 テンソルコア探索で解消。品質パリティは実測で確認
  (ドリフト <0.005%)。Qwen 系対策の HayateQuant-dense-LS、大符号帳 HayateQuant-K12 も追加。

**主要結果 (v3 実測):**

| Model | bits | 手法 | dist(K) | dist(V) | PPL |
|---|---|---|---:|---:|---:|
| Llama-3.1-8B | 3 | HALO-auto (旧最良) | 0.0319 | 0.0331 | 7.086 |
| Llama-3.1-8B | 3 | **HayateQuant-K12** | **0.0270** (-15.3%) | **0.0267** (-19.4%) | **6.985** |
| Llama-3.1-8B | 2 | HALO-dense-LS (旧最良PPL) | 0.1166 | 0.1151 | 9.868 |
| Llama-3.1-8B | 2 | **HayateQuant-dense-LS** | **0.0969** | 0.0963 | **8.930** (-9.5%) |
| R1-Qwen-1.5B | 2 | HALO-dense-LS (旧最良PPL) | 0.1119 | 0.1141 | 159.38 |
| R1-Qwen-1.5B | 2 | **HayateQuant-dense-LS** | **0.0992** | **0.0967** | **157.45** |

**推論速度 (e2e v3b, Llama-3.1-8B-Instruct, prompt=4096, gen=128):**

| 手法 | TTFT | decode tok/s | 備考 |
|---|---:|---:|---|
| FullKV | 72.7ms | 78.1 | baseline |
| TurboQuant-3bit | 112.0ms | 19.5 | — |
| HALO-NC-3bit | 92.4ms | 33.0 | — |
| **HayateQuant-3bit** | **115.3ms** | **27.4** | TTFT は TQ と同等、decode は TQ の 1.4 倍 |
| **HayateQuant-2bit** | 323.2ms | **27.5** | 2-bit のみ TTFT 2.9 倍(正直な残存コスト) |

**Paper-level ロングコンテキスト (job 58907, Llama-3.1-8B):** 低ビット域で全指標勝利 —
2-bit Needle **100%** (TQ 93.3/HALO 91.1)、2.5-bit RULER **0.933 = FullKV と同値** (TQ 0.800)、
2-bit LongBench 26.87 (TQ 21.34, +5.5pt)、2-bit PPL 9.13 (TQ 10.17)。

詳細は `papers/hayatequant_paper/README.md` を参照。

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

## 3本の論文の切り分け方針

| 観点 | HayateQuant論文 | HALO 論文 | TaperKV 論文 |
|------|-----------------|-----------|--------------|
| 主題 | Codec (圧縮変換方式、HALO後継) | Codec (圧縮変換方式、アーカイブ) | Policy (精度スケジュール) |
| 比較相手 | TurboQuant, HALO, Uniform-INT, FP8, KIVI | TurboQuant, Uniform-INT, FP8, KIVI | StreamingLLM, ThinKV, PM-KVQ |
| 評価モデル | R1-Qwen-1.5B, Llama-3.1-8B-Instruct | Llama-8B, Ministral-8B (非R1) | Llama-8B + R1-Distill系 + QwQ-32B |
| 主要メトリクス | dist(K/V), PPL, encode ms, TTFT, tok/s, peak-mem | PPL, Needle, RULER, encode ms, relMSE | LongBench, GSM8K, MATH-500, AIME, LCB |
| HayateQuant の位置づけ | 提案手法 (メイン、現在の推奨) | 登場しない | 登場しない |
| HALO の位置づけ | 比較用 baseline (superset関係) | 提案手法 (アーカイブ扱い) | 比較用 codec baseline として登場 |
| TaperKV の位置づけ | 登場しない | 登場しない | 提案手法 (メイン) |
| TurboTaper の位置づけ | 登場しない | 登場しない | TaperKV + TurboQuant の組み合わせ variant |

---

## 完了したジョブ (2026-07-04)

| ジョブ名 | JID | 内容 | 状態 |
|---------|-----|------|------|
| kv-hayatequant | 58907 | paper-level ロングコンテキスト (Needle/RULER/LongBench-E/PPL) | ✅ 完了 (7h46m) |
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

4. **HayateQuant の残存コスト (v3 で大幅縮小、2-bit prefill のみ)**  
   - v3 の Folded Rotation + 候補リスト LUT + fp16 探索により decode は TurboQuant の
     1.4 倍 (27.5 vs 19.5 tok/s)、TTFT も 3/4-bit で同等まで改善。
   - 唯一 2-bit prefill (block_size=4, K=256 の厳密探索がメモリバウンド) のみ
     TTFT 2.9 倍 (323ms vs 112ms) が残る。品質側の優位 (Needle 100%, LongBench +5.5pt,
     PPL -1.0) とのトレードオフとして正直に報告。
