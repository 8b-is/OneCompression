# HayateQuant vs TurboQuant (arXiv:2504.19874) — 精密比較

> **Status (2026-07-15): historical design note plus legacy measurements.**
> Sections 1--3 record the design lineage and contain old auto/LUT examples;
> the authoritative current HQ/AdaptiveHQ definition is the fixed recipe below.
> The numeric claims in Section 4 came
> from pre-formal runs and are invalid for the refreshed paper.  They used
> mismatched scale precision, capped/local benchmark proxies, and an unfair
> speed harness.  The coverage-checked rerun is the only admissible source of
> empirical claims.  Block-VQ is a design generalization, not a theorem that
> HayateQuant must outperform TurboQuant on every model or metric.
> The refreshed public proposal has only two rows: **HQ**, the fixed exact
> `b=2` Gaussian-codebook recipe with K5/V3, K7/V5, or K9/V7 allocation plus
> key NC and value LS reassign/refit; and **AdaptiveHQ**, which adds only the
> preregistered causal v5 opt-in selector.  Auto/LUT/K12 names below are legacy.

TurboQuant 論文 "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (Zandieh, Daliri, Hadian, Mirrokni; arXiv:2504.19874) を精読し、
主張・アルゴリズム・理論・実験プロトコルの各レベルで HayateQuant との相違点と
強化点を整理する。論文本文は `conference/quant_ref/2504.19874.pdf` /
`2504.19874v1.html`、公式実装は `quant_ref/official_impl_0xSero/`。

---

## 1. TurboQuant 論文の内容(要約)

**問題設定**: 量子化写像 Q: R^d → {0,1}^{b·d}(bit-width b)と復元写像 Q^{-1} を
設計し、worst-case 入力 x (‖x‖₂=1) に対する期待歪みを最小化する。
data-oblivious(データ依存の学習・キャリブレーションなし)かつ online
(ベクトル到着ごとに即時適用可能)であることを要求。

**アルゴリズム(2 段構成)**:

1. **TurboQuant_mse** — ランダム回転 Π(ガウス行列の QR 分解)を掛けると、
   Πx の各座標は Beta 分布(高次元では N(0,1/d) に収束)に従い、座標間は
   ほぼ独立になる。よって座標ごとに **1 次元 Lloyd-Max 最適スカラー量子化**
   (連続 k-means を数値的に解いた符号帳を事前計算・保存)を適用すれば十分、
   という設計。復元は符号帳引き + Πᵀ。
2. **TurboQuant_prod** — MSE 最適量子化器は内積推定にバイアスを持つ
   (b=1 で乗法バイアス 2/π)ため、(b−1)-bit の TurboQuant_mse を掛けた後、
   残差に **1-bit QJL**(sign(S·r) + 残差ノルム保存)を適用して不偏化する。

**理論**:

- 上界: D_mse ≤ (π√3/2)·4^{−b}(b=1..4 の精密値 0.36 / 0.117 / 0.03 / 0.009)。
  D_prod ≤ (√3π²/d)·4^{−b}‖y‖²、かつ不偏。
- 下界: Shannon Lower Bound + Yao's minimax で、**いかなる**ランダム化量子化器も
  D_mse ≥ 4^{−b}。つまり TurboQuant のスカラー Lloyd-Max は最適値の
  **高々 π√3/2 ≈ 2.72 倍**(b=1 では ≈1.45 倍)の範囲内。
- 論文 3.1 節の "Entropy Encoding Codebook Pointers" 段落は、符号語頻度の
  エントロピー符号化で最大 ~5% のビット削減が可能だが「単純さと速度のため
  採用しない」と明記(→ HayateQuant も同じ判断を踏襲)。

**実験**: (a) DBpedia/OpenAI3 埋め込みでの歪み検証(理論と一致)、
(b) Llama-3.1-8B-Instruct の Needle-In-A-Haystack(2.5-bit で recall 0.997 =
FullKV と同値)、(c) LongBench-V1(2.5/3.5-bit、outlier 32ch を +1bit で
混合精度 → "KV Size 2.5/3.5")、(d) NN 検索で PQ/RabitQ を recall で上回り
インデックス時間ほぼゼロ。

---

## 2. 相違点(アルゴリズム設計)

| 軸 | TurboQuant (論文) | HayateQuant | 差の性質 |
|---|---|---|---|
| 量子化単位 | **1 座標**(スカラー Lloyd-Max) | **b∈{1,2,4} 次元ブロック**(block-VQ; データ不要 K-means を N(0,1)^b / 厳密 Beta^b 合成サンプルで学習) | block_size=1 で scalar codes/rate/real-arithmetic reconstruction が一致。Beta の L2-vs-RMS scale parameterization は fp16 丸め後 bit-identical とは限らない |
| 理論スラック | 下界の ≤2.72 倍で "near-optimal" と主張し、**残りのギャップは放置** | そのギャップ(= 高レート理論の space-filling/shape gain, Gersho 1979)を block-VQ で**部分回収** | TurboQuant 自身の下界証明が HayateQuant の改善余地の存在証明になっている |
| 回転 | ガウス QR のランダム回転 Π(実装は Hadamard 系)1 種のみ | **Folded Hybrid Rotation**: 大域 signed-Hadamard × ブロック毎 Dense 直交回転をオフラインで 1 枚の直交行列に畳み込み。`block_size` が Hadamard↔Dense の内挿ノブ、実行時コストは常に単一 GEMM | Hadamard が苦手なモデル分布(Qwen 系で実測)への対策。実行時追加コストゼロ |
| 正規化・スケール | ‖x‖₂ を fp で保存しリスケール(1 ベクトル 1 スカラー) | 同じストレージ(1 ベクトル 1 fp16 スケール)のまま、**役割非対称スケール**: key は norm-correcting、value は least-squares / 反復 LS(HALO から継承・検証済み) | ストレージ中立の精度改善 |
| 外れ値 | outlier チャネルを +1bit(2.5/3.5-bit 実験で使用) | 同じ `n_outlier`/`outlier_bits` 機構を継承(スーパーセット) | 同等 |
| 内積不偏化 (QJL) | TurboQuant_prod の中核 | `use_qjl` オプションとして実装・保持するが、**KV キャッシュでは既定 OFF**。vLLM 上流も QJL 残差を意図的に省く(softmax を通る attention では分散増幅が品質を害する)ことを mapping が警告 | 適用領域の違いを明示(NN 検索なら prod が正当) |
| エンコーダ計算量 | 座標ごと argmin(K=2^b と線形だがスカラーなので軽い) | block-VQ の O(K) 探索を **候補リスト Grid-LUT で O(T), T=2..16 に短縮**(符号帳サイズと無関係)+ fp16 テンソルコア厳密探索 + 呼び出しサイズで exact/LUT を選ぶ adaptive encoder | block-VQ を「理論上の勝ち」から「実運用の勝ち」にした鍵。K=4096 (max_codebook_bits=12) が実用域に |

**compatibility control**: HayateQuant(block_size=1, rotation=hadamard 系,
scale=rms, dist="beta"/"gaussian")= TurboQuant。これは設計上の狙いであり、
`test_hayatequant_block_size_1_matches_turboquant`(gaussian)と
`test_hayatequant_block_size_1_matches_turboquant_beta`(L2/Beta パス、v4 で
正規化を sqrt(m) リスケールで厳密化)がコード上で固定している。
scalar codes と論理 rate は一致する。ただし fp16 metadata の丸め位置、
有限サンプルで学習した block codebook、encoder、モデルの downstream 感度があるため、
「HayateQuant は必ず負けない」または「block-VQ は必ず改善する」とは結論できない。

## 3. 理論面の位置づけ

TurboQuant の下界定理(D_mse ≥ 4^{−b})と上界(π√3/2 ≈ 2.72 倍)の間の
ギャップは、論文自身は「小さい定数」として片付けている。しかし古典的
高レート量子化理論によれば、このギャップの主成分は**次元 >1 の同時量子化で
しか得られない space-filling gain**(b→∞ で最大 πe/6 ≈ 1.53 dB)であり、
スカラー量子化に固執する限り原理的に回収できない。HayateQuant の block-VQ は
まさにこの成分を狙う:

- 2 次元ブロック(3/4-bit の auto 解決点)で理論利得の一部、
  4 次元ブロック(2-bit auto / K12)でさらに大きな割合を回収。
- 実測(合成 N(0,1), `verify_hayatequant_vq_gain.py`)で block_size 1→2→4 の
  歪み単調減少を確認。実モデルでも dist(K) が 3-bit で −9.0%(vs HALO-auto)、
  K12 で −15.3%。
- 符号帳はサンプルではなく **TurboQuant 自身の理論分布**(ガウス極限 or
  有限 d の厳密 Beta)から合成するため、**data-free / online という
  TurboQuant の土俵を一切降りていない**。

## 4. 旧実験プロトコルと結果（legacy、引用禁止）

TurboQuant 論文の KV 実験プロトコル(Needle、LongBench-E、outlier 混合精度に
よる 2.5/3.5-bit)を `benchmark/kvcomp-longcontext/` で忠実に再現した上で比較:

- **Needle**: TurboQuant-2bit 0.933 に対し HayateQuant-2bit **1.000**。
- **旧 local multi-key diagnostic (当時 RULER と誤記)**: 2.5-bit で HayateQuant
  0.933 = FullKV と同値(TurboQuant 0.800)。正式RULERスコアではない。
- **LongBench-E**: 2-bit +5.5pt、2.5-bit +3.5pt、3-bit +3.4pt。
- **WikiText-2 PPL**: 2-bit 9.125 vs 10.168(−10.3%)。
- **速度 (A100, Llama-3.1-8B, prompt 4096/gen 128, v5 コード)**: decode
  27.0-27.5 tok/s vs 19.5-20.1 tok/s(**1.34-1.38 倍**)、TTFT は全ビット幅で
  同等(113.6-117.2ms vs 110.8-111.5ms, +2〜6%)。逐次 dequant は folded
  1-GEMM 逆回転により 7 倍速(63us vs 441us/step)、decode-step の encode も
  TurboQuant より速い(437us vs 506us)。prompt 4096 + 128 生成の総時間は
  4.8s vs 6.6s(27% 短縮)。
- メモリ・圧縮率は全ビット幅で TurboQuant と同一(ストレージ中立設計)。

## 5. 現時点で言える一文サマリ

> HayateQuant は TurboQuant の scalar quantizer を matched control として持ち、
> data-free block-VQ、K/V 非対称 rate、role-aware scale、folded rotation を加えた
> 設計である。これらが同一の実効 rate で品質または速度を改善するかは、固定した
> 完全 benchmark と paired statistics によってモデル・bit幅ごとに判定する。
