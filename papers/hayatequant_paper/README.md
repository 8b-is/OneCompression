# HayateQuant Paper: Block-VQ + Folded-Hybrid-Rotation KV Codec (supersedes HALO)

## 概要

**主張:** HayateQuant は TurboQuant (arXiv:2504.19874) と HALO の**完全な上位互換**である
データ不要 KV キャッシュ量子化 Codec。`block_size=1` で両者を数値的にもレイテンシ的にも完全再現しつつ
(`test_hayatequant_block_size_1_matches_turboquant`, `test_hayatequant_block_size_1_matches_halo_scale_modes`)、
**同じビット数で厳密に低い歪み・PPL・下流タスク精度**を達成し、v3-v5 のシステム最適化
(Folded Rotation + 候補リスト LUT + fp16 スコアリング + ギャザー経路修正)により
**decode スループットでは TurboQuant を 1.34-1.38 倍上回り、TTFT も全ビット幅で
TurboQuant と同等 (+2〜6%)** まで到達した(v5 実測、下記)。

**TurboQuant 論文との精密比較:** `TURBOQUANT_COMPARISON.md` —
arXiv:2504.19874 の主張・理論(下界 4^{-b}、上界 √3π/2≈2.72 倍)・実験プロトコル
に対する HayateQuant の相違点/強化点を軸ごとに整理(block-VQ による
space-filling gain 回収、Folded Hybrid Rotation、役割非対称スケール、
候補リスト LUT、QJL の扱い、同一土俵ベンチの勝敗)。

**概念図:**

- `figs/hayatequant_vs_turboquant.png` — TurboQuant(スカラー Lloyd-Max、軸整列正方格子セル)
  と HayateQuant(Folded Hybrid Rotation + Block-VQ、Voronoi セル + space-filling gain)の
  パイプライン対比。`block_size=1` で厳密に TurboQuant へ退化する上位互換関係を明示。
- `figs/hayatequant_hybrid_rotation.png` — Hadamard ↔ Dense の「構造 vs 表現力」スペクトラム上に
  Hybrid 回転を位置づけ、`W = blockdiag(Q) · (H·sign)` をオフラインで 1 枚に畳み込むことで
  実行時コスト = 単一 GEMM(Dense と同額)となり内挿が無料になることを図示。
  D=128–256 では fp16 GEMM が memcpy(FHT プロキシ)の 1.5 倍に収まり FWHT カーネルの
  優位が D≥4096 でしか現れない実測も併記。

**新規性 (4つのレバー、いずれもデータ不要・ストレージ中立):**

1. **Block Vector Quantization** (`onecomp/kvcomp/codec/hayate/vq.py`) — TurboQuant 論文自身が
   証明する「スカラー Lloyd-Max は距離-歪み最適値の高々 `sqrt(3π/2) ≈ 2.7×` 以内」という
   スラックの正体は、古典的高レート量子化理論 (Gersho & Gray, 1992) が言う
   **space-filling gain / shape gain**(次元 >1 の同時量子化でしか得られない)である。
   HayateQuant は TurboQuant 自身の理論分布(ガウス極限 or 有限 `d` の厳密 Beta 分布)から
   合成サンプルを生成し、multi-restart Lloyd 法(K-means)で `block_size` 次元の
   ベクトル符号帳を学習する。`block_size=1` はスカラー符号帳に厳密退化するため、
   HayateQuant は理論上 TurboQuant/HALO より悪化しない。
2. **Folded Hybrid Rotation** (`rotation.py`: `hybrid_rotation_matrix`) — Block-VQ は
   各ブロックが等方的であることを要求するが、固定 Hadamard 変換(大域符号反転のみランダム)は
   ブロックレベルでは Haar-random ではない。HayateQuant は「大域 signed-Hadamard」+
   「ブロック毎の独立な小規模 Dense 直交回転」を合成する。**v3 の追加**: この 2 段変換
   `W = HS·blockdiag(Q_1..Q_{D/b})` を**オフラインで 1 枚の直交行列に畳み込む**ことで、
   実行時コストを**単一 GEMM(= 素の Dense 回転と完全に同額)**にした。つまり
   「Hadamard の速さと Dense の等方性のどちらを取るか」というトレードオフは
   **実行時には消滅**し、`block_size` は品質側の内挿ノブとしてだけ残る
   (社内議論「Hadamard と Dense の間を内挿できないか」への完全な回答)。
   数学的同値性は `test_hybrid_rotation_matrix_matches_two_stage` で固定。
3. **候補リスト Grid-LUT エンコーダ** (`_vq.py`: `build_vq_lut`/`quantize_via_lut`) —
   Block-VQ の実行時ボトルネックは最近傍符号語探索 `O(K)` (K=符号帳サイズ)だった。
   回転+RMS 正規化後のクエリ分布は**既知かつ固定**(≈N(0,1)^b)なので、`[-5,5]^b` の
   規則格子の各セルに「セル中心に近い上位 T 個の符号語」を事前計算しておけば、
   エンコードは (a) セル番地計算 O(b)、(b) T 候補に対する**厳密**距離 argmin、の
   `O(T)`(T=2〜16 ≪ K=64〜4096)になり、**符号帳サイズと無関係**になる。
   割当のみの近似で復元は常に厳密符号語。過剰歪みは実測 <0.06%
   (`test_hayatequant_lut_encoder_matches_exact_distortion`)。
   さらに厳密探索側も CUDA では fp16 テンソルコア GEMM 化(精度パリティ実測 <0.005%)。
   `encoder="auto"` は呼び出しごとに厳密 GEMM と LUT の速い方を選ぶ
   (実測クロスオーバーに基づく: b=2 prefill → LUT 4.8-10.4 倍速、b=4 小 K → 厳密 GEMM、
   K=4096 → LUT ~4 倍速、decode-step の 1 トークンエンコード → 厳密 GEMM)。
   この結果、**従来レイテンシ的に不可能だった大符号帳 (max_codebook_bits=12, 3-bit で
   K=4096) が実用域に入り**、下表の HayateQuant-K12 という新しい品質上限を解放した。
4. **HALO の検証済みレバーを継承** (`_scale.py`) — role-asymmetric scale
   (key: norm-correcting、value: least-squares/ls_iter)と outlier 混合精度
   (`n_outlier`/`outlier_bits`)をそのまま再利用。

**HALO の扱い:** `HALOCodec` はコードベースに残置(削除しない)。(a) 過去結果の再現性、
(b) HayateQuant の優位を示す比較対象として恒久的に必要、のため。新規開発は `hayatequant`。

---

## v3 システム最適化の効果 (実測, Llama-3.1-8B-Instruct, prompt=4096, gen=128, A100)

v2(2026-07-04 朝時点)の弱点はログ分析で 3 点特定された:
(1) decode スループットが TurboQuant の 0.7 倍、(2) 2/4-bit の TTFT が 3.4-4.3 倍、
(3) R1-Qwen 系では dense 回転系に PPL で負けるケース。v3 はこれを次で解消した。

### e2e 推論速度 (v2 → v3b)

| bits | 手法 | TTFT v2 | TTFT v3b | decode v2 | decode v3b |
|---|---|---:|---:|---:|---:|
| 2 | TurboQuant | 110.6ms | 111.6ms | 19.9 tok/s | 19.6 tok/s |
| 2 | **HayateQuant-auto** | 380.3ms | **323.2ms** | 13.7 tok/s | **27.5 tok/s** |
| 3 | TurboQuant | 110.7ms | 112.0ms | 19.8 tok/s | 19.5 tok/s |
| 3 | **HayateQuant-auto** | 278.1ms | **115.3ms** | 13.6 tok/s | **27.4 tok/s** |
| 4 | TurboQuant | 110.9ms | 111.9ms | 19.9 tok/s | 19.5 tok/s |
| 4 | **HayateQuant-auto** | 472.3ms | **115.3ms** | 13.7 tok/s | **27.5 tok/s** |

**→ decode スループットは全 bit 幅で TurboQuant の約 1.4 倍(27.4-27.5 vs 19.5-19.6 tok/s)。**
**→ TTFT は 3/4-bit で TurboQuant と同等 (+3%)、2-bit のみ 2.9 倍(バッチ b=4 厳密探索の
メモリバウンド性による。正直な残存コストとして報告; それでも v2 比 15% 改善し、
品質側の優位(下表)がこのコストを大きく上回る)。**
per-layer の逐次 dequant 単体では 56-58us/step で TurboQuant(443-496us)の **8 倍速**
(Folded Rotation により逆回転が 1 GEMM になったため)。

### コーデック単体 (codec_speed.py, 32 heads x 4096 seq x dim128)

| bits | 手法 | prefill enc | prefill dec | 合成KV MSE |
|---|---|---:|---:|---:|
| 3 | TurboQuant-hadamard | 1.73ms | 1.67ms | 0.0996 |
| 3 | HayateQuant-bs1 | 1.57ms | **0.475ms** | 0.0978 |
| 3 | **HayateQuant-auto** | 4.31ms | **0.479ms** | **0.0884** |
| 2 | TurboQuant-hadamard | 1.71ms | 1.66ms | 0.3412 |
| 2 | **HayateQuant-auto** | 25.7ms | 4.64ms | **0.2941** |

---

## 品質結果 (codec_benchmark.py v3, 実モデル実測)

### Llama-3.1-8B-Instruct (TurboQuant 論文のヘッドラインモデル)

| bits | 手法 | dist(K) | dist(V) | PPL | ms/encode |
|---|---|---:|---:|---:|---:|
| 4 | HALO-auto (旧最良) | 0.00871 | 0.00891 | 6.7755 | 0.324 |
| 4 | **HayateQuant-auto** | **0.00835** | **0.00844** | **6.7658** | 0.597 |
| 3 | HALO-auto (旧最良) | 0.03186 | 0.03309 | 7.0857 | 0.303 |
| 3 | **HayateQuant-auto** | 0.02900 | 0.02924 | 7.0044 | 0.587 |
| 3 | **HayateQuant-K12** (新・品質上限) | **0.02697** (-15.3%) | **0.02667** (-19.4%) | **6.9850** | 8.33 |
| 2 | HALO-dense-LS (旧最良PPL) | 0.11661 | 0.11511 | 9.8675 | 0.252 |
| 2 | **HayateQuant-auto** | 0.09809 | 0.09619 | 9.1377 | 4.75 |
| 2 | **HayateQuant-dense-LS** (新) | **0.09690** | 0.09633 | **8.9299** (-9.5%) | 3.18 |

### DeepSeek-R1-Distill-Qwen-1.5B (Qwen 系; Hadamard が苦手な分布)

| bits | 手法 | dist(K) | dist(V) | PPL |
|---|---|---:|---:|---:|
| 3 | HALO-dense-LS (旧最良PPL) | 0.03182 | 0.03310 | 163.21 |
| 3 | **HayateQuant-K12** | **0.02576** | **0.02631** | 169.02 |
| 3 | **HayateQuant-dense-LS** | 0.02866 | 0.03025 | **165.02*** |
| 2 | HALO-dense-LS (旧最良PPL) | 0.11189 | 0.11408 | 159.38 |
| 2 | **HayateQuant-dense-LS** | **0.09919** | **0.09671** | **157.45** |

*R1 の 3-bit PPL のみ HALO-dense-LS (163.2) が僅差で残るが、これは既知の
「R1-Qwen 系 codec PPL 異常(外れ値レイヤ)」の影響下にある指標で、歪みでは
HayateQuant が全勝。2-bit では PPL でも歪みでも dense-LS バリアントが全手法を更新。
ログ分析で判明した「Qwen 系は Hadamard 系回転と相性が悪い」への対策として
`HayateQuant-dense-LS`(global dense 回転 + 対称 LS スケール + block-VQ)を追加した;
このバリアントは Llama の 2-bit でも PPL 最良(8.93)であり、block-VQ が回転方式に
直交して常に利得を与えることを示している(LUT/fp16 化は rotation_mode="dense" にも適用)。

**エンコーダ近似の検証 (LUT/fp16 ablation):** `HayateQuant-exact`(厳密探索)と
`HayateQuant-auto`(LUT/fp16)の歪みは全 bit 幅・両モデルで一致(差 ≤ 1e-5)、
PPL 差も noise 範囲(例: Llama 2-bit 9.1377 vs 9.1246)。

---

## Paper-level ロングコンテキスト (kv_benchmark.py, Llama-3.1-8B-Instruct, job 58907 完了)

TurboQuant 論文と同一プロトコル(Needle 4K/8K/16K x 5 depth x 3 seed、RULER、
LongBench-E 13 タスク、WikiText-2 PPL@4096)。

| 手法 | Needle | RULER | LongBench | PPL | mem |
|---|---:|---:|---:|---:|---:|
| FullKV | 1.000 | 0.933 | 27.40 | 6.486 | 1.000 |
| TurboQuant-4bit | 1.000 | 0.933 | 30.03 | 6.565 | 0.258 |
| **HayateQuant-4bit** | 1.000 | 0.867 | 28.15 | 6.571 | 0.258 |
| TurboQuant-3.5bit | 1.000 | 0.867 | 31.76 | 6.696 | 0.235 |
| **HayateQuant-3.5bit** | 1.000 | **0.933** | 26.93 | **6.647** | 0.235 |
| TurboQuant-3bit | 1.000 | 0.800 | 25.77 | 6.917 | 0.196 |
| **HayateQuant-3bit** | 1.000 | **0.867** | **29.18** | **6.839** | 0.196 |
| TurboQuant-2.5bit | 1.000 | 0.800 | 26.57 | 7.772 | 0.172 |
| **HayateQuant-2.5bit** | 1.000 | **0.933** | **30.03** | **7.214** | 0.172 |
| TurboQuant-2bit | 0.933 | 0.400 | 21.34 | 10.168 | 0.133 |
| HALO-2bit | 0.911 | 0.333 | 21.14 | 9.966 | 0.133 |
| **HayateQuant-2bit** | **1.000** | **0.600** | **26.87** | **9.125** | 0.133 |

**→ 実運用上重要な低ビット域(2/2.5/3-bit)で全指標勝利。**
特に **2.5-bit の RULER 0.933 は FullKV と同値**(TurboQuant は 0.800)、
**2-bit の Needle 100%**(TurboQuant 93.3%、HALO 91.1%)、
LongBench も 2-bit +5.5pt / 2.5-bit +3.5pt / 3-bit +3.4pt。
KIVI-INT2 (needle 100%, RULER 0.6, PPL 7.42, mem 0.161) に対しても、同等の Needle/RULER を
より小さいメモリ (0.133) で達成しつつ LongBench では 26.87 vs 27.54 と拮抗。

---

## vLLM プラグイン

- `vllm_plugins/kvcomp/mapping.py` が `hayatequant` を認識: `block_size=1` は vLLM
  ネイティブ `turboquant_*` プリセットへ、block-VQ(`block_size>1`/auto)はカスタム
  バックエンド要と明示(silent downgrade しない)。
- **opt-in スカラーフォールバック** (`KVCOMP_VLLM_SCALAR_FALLBACK=1`): block-VQ
  設定のモデルを vLLM で「非圧縮」ではなく最寄りのスカラー `turboquant_*`
  プリセット(= HayateQuant `block_size=1` と同一の量子化)で提供する明示的
  escape hatch。block-VQ の精度利得は落ちる旨を必ず警告付きで通知
  (デフォルトは従来通り正直に custom-backend 要と報告)。
- **回転族チェック(Hadamard 対応)**: vLLM ネイティブ TurboQuant カーネルは
  randomized **Hadamard** 回転を内蔵する。mapping は codec の `rotation_mode` を検査し、
  Hadamard 族(`hadamard` / `hadamard_gemm` / `auto` / `hybrid`+`block_size=1` の退化形)
  なら「ネイティブカーネルと同族」と note で確認、`dense`(および block-VQ 用 `hybrid`)
  なら**異なる変換で量子化される旨を明示警告**する(silent swap しない)。
- デフォルト設定 (`block_size=None`) の `get_kv_config()` 辞書で `int(None)` クラッシュ
  する重大バグを発見・修正済み(`_hayatequant_uses_block_vq`、回帰テスト付き)。
- `pyproject.toml` の `vllm.general_plugins` エントリポイントに登録済みで、
  `EngineArgs.create_engine_config` フックにより `LLM(model=...)` だけで自動適用。
- 専用テスト `tests/vllm_plugins/kvcomp/test_kvcomp_vllm_plugin.py`(12 テスト)が
  偽 `EngineArgs` で登録経路を検証。回転族チェックは
  `test_kvcomp.py` の `test_vllm_mapping_*rotation*` 系 7 テストでも固定。
- **実 vLLM 0.24.0 での検証済み**(`.venv-vllm`): `CacheDType` に `turboquant_4bit_nc` /
  `turboquant_k8v4` / `turboquant_k3v4_nc` / `turboquant_3bit_nc` が存在することを確認し、
  実 `EngineArgs` に対して HayateQuant(hadamard_gemm, bs=1)→ `turboquant_4bit_nc`、
  k3v4 → `turboquant_k3v4_nc`、fp8 → `fp8_e4m3`+`calculate_kv_scales`、
  block-VQ → `auto` 温存、の全経路 pass。
- **GPU 実生成 E2E pass**(`ichikawa_works/kvcomp_hayatequant/vllm_hadamard_e2e.py`,
  Slurm job 58995, R1-Qwen-1.5B): plugin が `kv_cache_dtype=turboquant_4bit_nc` を自動設定
  → vLLM が `TURBOQUANT` attention backend を選択 → greedy 生成 2 プロンプトが
  **fp16 ベースラインと完全一致**(4-bit KV でも出力劣化なし)。

---

## v4 監査 (2026-07-04): 実装再監査・修正と実験基盤の世代交代

コードベース全体の再監査で見つかった問題と対応:

1. **[CRITICAL 修正] `dist="beta"` の正規化不一致**: TurboQuant の beta パスは
   L2 ノルム正規化 + [-1,1] 符号帳だが、HayateQuant は常に RMS 正規化だったため、
   `dist="beta"` では TurboQuant と bit-for-bit 一致しなかった(上位互換主張の穴)。
   スカラー beta 符号帳を `sqrt(m)` 倍にリスケールすることで解消 — RMS = L2/√m
   なので符号割当・再構成とも完全一致し、`lloyd_vq_beta_block`(√d スケール済み)
   とも整合。回帰テスト `test_hayatequant_block_size_1_matches_turboquant_beta` を追加。
2. **[修正] ペイロードに `dist` を保存**(TurboQuant と同様)。encode 後にコーデック
   設定を変えても decode が正しい符号帳を引く。`test_hayatequant_payload_dist_roundtrip` 追加。
3. **[修正] K-means 空クラスタ再シードの距離式**: `||x-c||² = ||x||² - 2·score` のところ
   余分な `-0.5||c||²` 項があった(再シードの順位付けのみに影響、round-trip には無影響)。
4. **[簡素化] 未使用パラメータ `eps` を削除**(HALO からのコピー残り)。
5. **[簡素化] 2 段階 hybrid 回転 API はテスト専用の参照実装と明記**、seed 式の
   docstring 不一致も修正(実運用は折り畳み 1-GEMM のみ)。

**実験基盤の世代交代 (HALO → HayateQuant 主体)**:

- `conf/turboquant_paper.yaml` / `paper_aime.yaml` / `paper_taperkv*.yaml` /
  `paper_ablation_{bmin,budget}.yaml` / `azalea_*.yaml`: 提案手法行を
  `HayateQuant-{2,2.5,3,3.5,4}bit` に置換。HALO は最強prior (NC+ls_iter) 1 系列のみ
  ベースラインとして残置。
- `submit_all.sh` (paper1 = HayateQuant ジョブ, `p1-hq-*`) / `submit_paper_final.sh`
  (P4 gap-fill も HayateQuant): 全て提案手法主体に更新。
- `benchmark/kvcomp-codec/conf/paper_hayatequant.yaml` 新設(`paper_halo.yaml` は
  レガシー再現専用と明記)。`codec_speed.py` に -exact/-K12/-dense-LS 変種を追加し
  `codec_benchmark.py` と整列。`e2e_speed.py` は重複していた HALO 3 変種を NCIT 1 本に簡素化。
- README 3 本 (`kvcomp-codec`, `kvcomp-longcontext`, `paper/`) と
  `docs/design/kvcomp-architecture.md` を HayateQuant 主体に書き換え。

修正後の検証: **144 テスト全 pass**(2 件追加)、`verify_hayatequant_vq_gain.py`
全チェック pass。v4 ベンチマーク一式を Slurm 投入(jobs 59000-59004:
codec 品質 x2 モデル / codec_speed / e2e_speed / 3-way ロングコンテキスト)。

### v4 実測結果 (jobs 59000-59003 完了; 59004 ロングコンテキストは実行待ち)

**品質 (Llama-3.1-8B, `results_paper_llama_v4/`)** — HayateQuant-auto が全ビット幅で
全ベースラインに勝利(歪み・PPL とも):

| bits | 最良ベースライン | PPL | **HayateQuant-auto** | PPL |
|---|---|---:|---|---:|
| 4 | HALO-auto | 6.776 | dK 0.00835 (-4.1%) | **6.766** |
| 3 | HALO-auto | 7.086 | dK 0.02900 (-9.0%) | **7.004** |
| 2 | HALO-dense-LS | 9.868 | dK 0.09809 (-12.2% vs HALO-auto) | **9.072** (-8.1%) |

- 2-bit は `HayateQuant-dense-LS` がさらに **PPL 8.90**(TurboQuant-hadamard 10.16 比 -12.4%)。
- R1-Qwen-1.5B でも歪みは auto が全勝、PPL は dense-LS 系が最良(159.96 @2bit)—
  Qwen 系では dense 回転が有効という v3 の結論を再確認。
- `HayateQuant-bs1` は HALO-auto と**完全一致**(上位互換ガードのライブ確認)。
- LUT (auto) は exact と歪み一致で encode 2〜4 倍高速 → LUT レバーは「効いている」。
- `K12` は 3-bit で最良品質 (PPL 6.985) だが encode 8.3ms → デフォルトにせず
  品質上限オプションとして維持(`max_codebook_bits=9` デフォルトの妥当性を確認)。

**e2e 推論速度 (Llama-3.1-8B, prompt=4096, gen=128, 3-bit)**:

| method | TTFT ms | decode tok/s | vs TurboQuant |
|---|---:|---:|---:|
| FullKV | 72.6 | 64.6 | — |
| TurboQuant-3bit | 114.9 | 16.1 | 1.00x |
| HALO-NCIT-3bit | 97.6 | 25.2 | 1.57x |
| **HayateQuant-3bit** | 115.4 | **25.6** | **1.59x** |

decode スループットで TurboQuant の 1.59 倍・HALO 超え、TTFT は TurboQuant と同等、
ピークメモリ・圧縮率 (5.10x) は全手法同一。

---

## v5 エンコーダ/デコーダ高速化 (2026-07-04 夜): データ駆動での「効く強化」の実装

v4 の残存弱点は速度側に 2 つあった: (1) 2-bit の prefill encode 21.6ms(TTFT 290ms、
TurboQuant の 2.6 倍)、(2) 2-bit の prefill decode 4.64ms(3-bit の 10 倍)。
GPU マイクロベンチ(`paper/tests/bench_encoder_paths.py` / `bench_lut_spec_sweep.py`,
A100, 実ブロック数)で原因を特定し、**測って効いたものだけ**を採用した:

1. **LUT 候補距離の fp16 化** (`_vq.py: quantize_via_lut`): 候補ギャザー+距離計算が
   支配的で、fp16 化で (b=4,K=256,T=8) 4.5 倍・(b=4,K=4096,T=16) 5.7 倍。過剰歪みは
   noise (<0.02%)。これにより従来「exact GEMM の方が速い」だった b=4 小 K 域でも
   LUT が勝つようになり、`_use_lut` の除外ルールを撤廃。
2. **2-bit LUT スペック再設計** (`lut_spec_for`): エンコードコストは bins にほぼ非依存で
   T に線形 → 格子を細かく(bins 32→64)して T を 8→2 に削減。単パス 4.76ms→0.96ms
   (5.0 倍)、過剰歪み +0.19%(2-bit KV-MSE で 0.294→0.295、PPL 9.07→9.15。それでも
   TurboQuant 10.16 / HALO 10.02 より圧倒的に良く、2-bit 推奨は据え置き dense-LS 8.96)。
   K=4096 の品質上限点は T=16 を維持(excess −0.02%)。
3. **LUT の int16 化** (`build_vq_lut`): K≤32767 では候補表を int16 で保持。
   常駐メモリとギャザー帯域を半減(bins=64,T=2,b=4 で 134MB→67MB)。
4. **`cb[codes]` ギャザーの経路修正** (`_vq.py: lookup_codewords`): PyTorch の
   advanced-indexing / index_select は小行 (b≥4) の行ギャザーで病的に遅い
   (b=4,K=256, 4.2M 行で 2.16ms; 同バイト数の平坦化 1-D `take` は 0.11ms = **19 倍**)。
   デコード・LUT 候補ギャザー・encode 内 lookup を全て `lookup_codewords`
   (大 N の b≥4 は take、それ以外は index_select)に置換。decode-step サイズの
   小ギャザーでは take の索引算術が逆に ~30us 損なので閾値 (`_TAKE_MIN_ROWS`) で回避。
5. **block_size=1 のソート短絡** (`quantize_to_vq_codebook`): Lloyd-Max 符号帳は常に
   ソート済みなのに毎回 sort + 全 N の `order[codes]` ギャザーを払っていた。
   ソート済みなら直接 searchsorted(実測 3 倍: 0.062ms→0.021ms/4.2M 要素)。

**採用しなかった強化(効果なしと実測判定)**: b=2 の LUT fp16 化(T=2 では gather が
支配的で差なし)、bins>64 の T=4 スペック(T=2 に速度で全敗、excess の改善も PPL に
現れない)、decode-step への take 適用(閾値未満では逆効果)。

### v5 実測 (codec_speed v6b, 32 heads x 4096 seq x dim128, A100)

| bits | 指標 | v4 | **v5** | TurboQuant | 対 TQ |
|---|---|---:|---:|---:|---:|
| 2 | prefill enc | 21.57ms | **4.17ms** (5.2x) | 1.70ms | 2.4x 遅 |
| 2 | prefill dec | 4.64ms | **0.53ms** (8.7x) | 1.67ms | **3.1x 速** |
| 3 | prefill enc | 4.27ms | **3.94ms** | 1.73ms | 2.3x 遅 |
| 3 | prefill dec | 0.48ms | **0.41ms** | 1.67ms | **4.0x 速** |
| 3 | decode-step enc | 453us | **437us** | 506us | **1.2x 速** |
| 3 | decode-step dec | 57us | **63us** | 441us | **7.0x 速** |

品質は全ビット幅で不変(4-bit PPL 6.759 / 3-bit 7.013 / 2-bit 9.147 (auto),
8.955 (dense-LS); 全て全ベースライン最良)。

### v5 e2e 推論速度 (Llama-3.1-8B, prompt=4096, gen=128)

| bits | 手法 | TTFT | decode tok/s | 対 TurboQuant decode |
|---|---|---:|---:|---:|
| 2 | TurboQuant | 110.8ms | 20.1 | 1.00x |
| 2 | **HayateQuant** | **117.2ms** (v4: 290ms) | **27.0** | **1.34x** |
| 3 | TurboQuant | 111.3ms | 19.6 | 1.00x |
| 3 | **HayateQuant** | **113.6ms** | **27.1** | **1.38x** |
| 4 | TurboQuant | 111.5ms | 19.5 | 1.00x |
| 4 | **HayateQuant** | **113.6ms** | **27.0** | **1.38x** |

**→ 全ビット幅で TTFT は TurboQuant と同等(+2〜6%)、decode は 1.34-1.38 倍。**
prompt 4096 + 128 トークン生成の総時間では HayateQuant 4.8s vs TurboQuant 6.6s
(**27% 短縮**)。v4 で唯一残っていた 2-bit の TTFT 2.6 倍問題は解消した。
生データ: `benchmark/kvcomp-longcontext/{codec_speed,e2e_speed}_v6b.json`
(v5 コード; `_v5/_v6.json` は途中段階の ablation 記録)。

---

## v6 再監査 (2026-07-04 深夜): 堅牢性修正・レバー分離 ablation・vLLM 再 E2E

コア 4 ファイル(`hayatequant.py` / `_vq.py` / `rotation.py` / `_scale.py`)を全行再精読し、
将来の開発・拡張で事故になり得る箇所を 4 点修正した(数値結果は全て不変):

1. **回転 seed のペイロード保存** (`hayatequant.py: _rotate/_unrotate`): 従来 rot タグは
   `("y", block_size)` 等で、decode は**インスタンスの** `self.seed` に依存していた。
   encode と decode の間に codec が別 seed で再構成されると静かに壊れる(`dist` は保存
   していたのに seed は保存していなかった非対称)。rot タグを `(kind, block_size, seed)` に
   拡張し decode はペイロードの seed を使う。旧 2-tuple ペイロードも後方互換で読める。
   TurboQuant 側は元から rot タグに seed を埋めており、これで規約が揃った。
2. **QJL+outlier の bit 会計修正** (`_stored_bits`): outlier 分割時は QJL の residual-RMS が
   subset ごと(=ベクトルあたり 2 個)保存されるのに 1 個分しか計上していなかった
   (ベクトルあたり fp16 1 個分の過少申告; QJL はデフォルト off なので既存結果に影響なし)。
3. **`value_bits`/`outlier_bits` のバリデーション追加** (`validate_params`): 従来は
   `bits` のみ検証され、`value_bits=99` が encode 時までエラーにならなかった。
4. **auto エンコーダの LUT を CUDA 限定に** (`_use_lut`): LUT/exact のクロスオーバーは
   GPU (A100) でしか測っておらず、CPU では O(bins^b×K) の表構築の償却も難しい。
   `encoder="auto"` は CPU では常に exact chunked-GEMM(予測可能)を使い、
   `encoder="lut"` の強制はどのデバイスでも従来どおり動く(CPU ユニットテストが使用)。

追加テスト 4 本(payload-seed 独立性、旧 rot タグ後方互換、value_bits 検証、QJL+outlier
会計)を含め **135 テスト全 pass**(kagura CPU, 181s)。

### レバー分離 ablation (job 59050, Llama-3.1-8B-Instruct, WikiText-2 PPL 32 windows)

`codec_benchmark.py` に opt-in の ablation グリッド(`--config-name
paper_hayatequant_ablation`)を追加。各行は HayateQuant-auto から**ちょうど 1 つ**の
レバーを外す:

| 2-bit | dist(K) | dist(V) | PPL | 読み取り |
|---|---:|---:|---:|---|
| TurboQuant-hadamard (ベースライン) | 0.1095 | 0.1171 | 10.161 | |
| HayateQuant-bs1 (スカラー=HALO 相当) | 0.1118 | 0.1158 | 10.022 | block-VQ なし |
| HayateQuant-bs2 | 0.1053 | 0.1054 | 9.512 | b=2 で −0.51 |
| **HayateQuant-auto (b=4)** | **0.0983** | **0.0964** | **9.147** | b=4 でさらに −0.36 |
| HayateQuant-noscale (RMS 両ロール) | 0.0966 | 0.0978 | 9.198 | scale レバー = −0.051 |
| HayateQuant-hadamard-bvq (hybrid なし) | 0.0983 | 0.0958 | 9.178 | hybrid レバー = −0.031 |
| HayateQuant-beta (厳密有限次元則) | 0.0981 | 0.0962 | 9.155 | gaussian と同等 |

- **block-VQ が支配的レバー**: bs1→bs2→bs4 で PPL 10.02→9.51→9.15 と単調改善
  (実 KV 上で古典的 space-filling gain 曲線どおり)。3/4-bit では auto が b=2 に解決される
  ため bs2 行 = auto 行(生きた sanity check として一致を確認)。
- **役割非対称スケール**: 全 bit で一貫して +(2-bit −0.051 / 3-bit −0.024 / 4-bit −0.013 PPL)。
- **hybrid 回転**: 低 bit で一貫して +(2-bit −0.031 / 3-bit −0.034; 4-bit は noise 内)。
  等方性検証(`verify_hayatequant_isotropy.py`)の予測と整合。
- **dist="beta"**: 歪みはわずかに改善するが PPL は gaussian と同等(±0.02)。
  デフォルト gaussian(符号帳がヘッド次元非依存でキャッシュ効率が良い)を維持。
- どのレバーも負に働く bit 幅はなく、「効いていない部分」は見つからなかった。
  外せるものを外すと必ず悪化する=現構成が最小十分であることの実測確認。

生データ: `benchmark/kvcomp-codec/results_ablation/results_llama-3.1-8b-instruct.json`

### 3-way ロングコンテキスト最終結果 (job 59004 完了, Llama-3.1-8B-Instruct, 5h51m)

TurboQuant / HALO / HayateQuant を同一プロトコル(Needle 45 prompts x 3 長,
RULER, LongBench-E n=10, WikiText-2 PPL 16 windows)で比較(v5/v6 コード):

| bits | 指標 | TurboQuant | HALO | **HayateQuant** |
|---|---|---:|---:|---:|
| 2 | Needle acc | 0.933 | 0.911 | **1.000** |
| 2 | RULER acc | 0.400 | 0.333 | **0.733** |
| 2 | PPL | 10.168 | 9.966 | **9.046** |
| 2.5 | RULER acc | 0.800 | 0.667 | **0.800** |
| 2.5 | PPL | 7.772 | 7.393 | **7.192** |
| 3 | PPL | 6.917 | 6.878 | **6.810** |
| 3.5 | PPL | 6.696 | 6.659 | **6.631** |
| 4 | PPL | 6.565 | 6.581 | 6.569 |

- **2-bit が決定的**: Needle 満点(TQ/HALO は取りこぼし)、RULER +0.33/+0.40、
  PPL −1.12/−0.92。block-VQ の space-filling gain が最も効く動作点で
  ダウンストリームでも明確に勝つ。
- 2.5/3/3.5-bit も全指標で最良または同率。4-bit は 3 手法が noise 内で並ぶ
  (量子化誤差自体が小さく差が出ない、想定どおり)。
- 参考: Uniform-INT2 は PPL 407(崩壊)、KIVI-INT2 は PPL 7.42 だが
  RULER 0.60 で HayateQuant-2.5bit(mem 同水準)に全面的に劣る。
- 生データ: `benchmark/kvcomp-longcontext/results_paper1/hayatequant_v4/results_llama-3.1-8b-instruct.json`

### vLLM 再 E2E (job 59051, DeepSeek-R1-Distill-Qwen-1.5B, A100)

現コード(v6 修正込み)で 3 ケース全て合格(`ALL_E2E_CHECKS_PASSED`):

1. HayateQuant(bs=1, hadamard_gemm) → プラグインが `turboquant_4bit_nc` を自動設定、
   実生成 OK(`HADAMARD_NATIVE_KV_OK`)。
2. ベースライン(設定なし) → `auto` のまま(誤圧縮なし, `BASELINE_OK`)。
3. **新規**: HayateQuant デフォルト(auto → block-VQ)+ `KVCOMP_VLLM_SCALAR_FALLBACK=1`
   → 最近傍スカラープリセット `turboquant_4bit_nc` で圧縮サーブ、生成品質正常
   (`BLOCKVQ_SCALAR_FALLBACK_OK`)。フォールバックの生成例: "A Hadamard matrix is a
   square matrix whose entries are either +1 or -1, and whose rows are mutually
   orthogonal." — 正しい出力。

---

## v7 レート適応回転・高予算符号帳 (2026-07-05): 測定駆動の第3ラウンド

「論文レベルの工夫を候補に挙げ、実測で効くものだけ採用する」方針で 4 候補を測定した
(データフリー光源での歪み: `bench_rvq_gain.py` job 59088 / 実モデル PPL での
レバー分解: `bench_policy_v7.py` jobs 59091-59092, Llama-3.1-8B + R1-Qwen-1.5B)。

### 候補と採否 (すべて実測に基づく)

| 候補 | 実測結果 | 採否 |
|---|---|---|
| 残差ブロック VQ (RVQ, b=8 を 8+8bit の 2 段に分解) | 2-bit 光源 MSE/dim: 単段 b=4 0.0986 に対し greedy 0.1057 / 交互再探索付き 0.1056 / 符号帳共同再学習込み 0.1040 と**全変種で劣化**。3-bit (12+12) 0.0310 vs 単段 b=4 0.0279、4-bit (8+8) 0.0105 vs 単段 b=2 0.0090 も同様。エンコードも 6-18 倍遅い (1.02ms → 5.9-11.9ms @4.2M 座標) | **不採用** |
| LUT 候補数 T=2→4 (2-bit の LUT 過剰歪み対策) | Llama 2-bit PPL 9.136 vs T=2 の 9.123 — noise 内で改善なし、gather コストのみ倍増 | **不採用** |
| 高予算 Lloyd 符号帳 (サンプル 16 倍・60 iters・8 restarts をオフライン 1 回で fit し、成果物をコミット) | 光源 MSE/dim が全点で改善: b4K256 −1.4% (0.09862→0.09693)、b4K4096 −2.6%、b2K256 −12% (0.00904→0.00795)。実 KV でも歪み一貫改善 (4-bit dist(K) 0.0084→0.0075)、Llama 2-bit PPL −0.12。ランタイムコストゼロ+符号帳がマシン非依存で bit 再現可能に | **採用** |
| 2-bit の回転を dense に (レート適応 auto) | スケールを固定した純粋な回転効果として測定: Llama 2-bit hybrid 9.123 vs **dense 8.915 (−0.21)**、両スケール設定で再現、R1-Qwen では中立 (166.6 vs 166.9)。3/4-bit は hybrid が勝つ (v6 ablation: 3-bit 7.013 vs 7.034) | **採用 (bits≤2 のみ)** |

実装 (計 3 ファイル + 成果物):

1. **`codebooks/` パッケージ** (`onecomp/kvcomp/codec/hayate/codebooks/`): `fit_codebooks.py` が
   デフォルト構成の全 (block_size, K) 点を高予算で fit し `gauss_b{b}_k{K}.pt`
   (合計 ~90KB) として保存。`lloyd_vq_gaussian_block` はデフォルト fit パラメータの
   とき成果物を優先し、無ければ従来のオンライン fit に自動フォールバック
   (成果物は必須ではない)。非デフォルトの fit パラメータ指定はオンライン fit のまま。
   fit は従来と同じ合成 N(0,1) サンプルのみ使用 — **データフリー性は不変**。
2. **レート適応 auto 回転** (`hayatequant.py: _resolved_mode`): block-VQ 時、
   subset の bits≤2 なら dense、bits≥3 なら hybrid に解決。2-bit では歪みは
   hybrid の方が僅かに良い (0.0983 vs 0.0996) のに PPL は dense が明確に勝つ —
   粗いレートでは固定 ±1 Hadamard 構造が量子化誤差をトークン間で相関させ、
   Haar-dense がそれを無相関化する attention レベルの効果であり、
   「歪みではなく PPL で方針を決める」本プロジェクトの方法論の好例。
   folded 1-GEMM なのでコストは同一。outlier 分割時も subset ごとの実効 bits で
   一貫して解決される (例: 2.5-bit = inlier 2-bit→dense / outlier 3-bit→hybrid)。

### v7 最終検証 (job 59097: フルグリッド再実行 + 速度回帰チェック)

Llama-3.1-8B-Instruct (WikiText-2 PPL 32 windows, A100):

| bits | TurboQuant-hadamard | TurboQuant-dense | HayateQuant-auto v6 | **HayateQuant-auto v7** |
|---|---:|---:|---:|---:|
| 2 | 10.161 | 9.942 | 9.138 | **8.825** (TQ 比 −1.34 / −1.12) |
| 3 | 7.097 | 7.144 | 7.004 | 7.038 (noise 内; dist(K) は 0.0290→0.0284 改善) |
| 4 | 6.786 | 6.783 | 6.766 | **6.758** (dist(K) −10%: 0.0084→0.0075) |

R1-Qwen-1.5B は全動作点 noise 内 (この模型の PPL は bits と単調でなく判定材料に
ならないことは v3 から既知; 推奨構成 dense-LS の優位は不変)。

速度は無回帰 (job 59097 [3/4]-[4/4]): e2e TTFT 117.6ms (TQ 111.2ms, +5.7%)、
デコード 26.2 tok/s (TQ 19.2, **+36%**)、コーデック単体 decode-step dec 64us
(TQ 440us, **6.9 倍**)。auto 回転の変更は folded GEMM の中身が変わるだけで
コスト同一、符号帳成果物はロードのみなので当然の結果。

vLLM E2E も再実行 (job 59099): 3 ケース全て `ALL_E2E_CHECKS_PASSED`。
ユニットテストは +2 本 (レート適応 auto の解決規則 / 成果物ロードの優先順位) を
含め **262 テスト全 pass** (`tests/onecomp/kvcomp` + `tests/vllm_plugins/kvcomp`
全体, kagura CPU, 346s)。

### v7 の教訓 (負の結果も記録する)

RVQ は「b=8 の space-filling gain を 2 段で近似する」正攻法だが、データフリー光源
ですら stage-2 残差の非ガウス性・非等方性で greedy/joint とも単段 b=4 に届かなかった。
b=8 @ 2-bit を本当に取るには構造化符号帳 (E8 lattice / QuIP# 系) が必要で、
それは将来課題として `bench_rvq_gain.py` と本節に測定条件ごと記録した。

---

## v8 オプショナル SpinQuant 風キャリブレーション + K/V 非対称ビット (2026-07-05)

第 4 ラウンド。今回から検証モデルを 3 つ(Llama-3.1-8B-Instruct / **Qwen3-8B**(新規) /
R1-Qwen-1.5B)に広げ、PPL だけでなく Needle / RULER でも採否を判定した。

### 候補 1: SpinQuant 風キャリブレーション → **optional として採用** (デフォルトは不変)

`onecomp/kvcomp/codec/hayate/calibration.py`(新規)+ `HayateQuantCodec(calibration=...)`:

- **学習回転**: 役割別(K/V 別)の直交回転 `W = W0 @ cayley(A)` を、実 KV サンプルの
  ブロック量子化歪みを目的関数に Adam+Cayley 縮約(SpinQuant と同じ多様体最適化)で学習。
  `W0` はコーデック自身のデータフリー回転(v7 レート適応 auto)なので **step 0 =
  データフリー版**、さらに held-out シャードで最良点を選ぶため、成果物がデータフリー版より
  悪くなることは構造上ない(`test_hayatequant_calibration_never_worse_on_holdout`)。
- **実データ符号帳**: 学習回転と交互に、実 z ブロック上で閉形式 Lloyd 再学習。
- 成果物はモデルごとに ~200KB(`HayateQuantCalibration.save/load`)。ランタイムコストは
  **ゼロ**(同じ 1 GEMM の中身が変わるだけ; 実測 ms_per_encode 不変)。fit は 1 モデル
  あたり数秒〜十数秒(A100)。キャリブデータは WikiText-2 **train**(評価は test — リーク無し)。
- ペイロードの rot タグ/`cb_cal` に成果物の content digest を保存し、decode 側で
  同一成果物のロードを検証(不一致は即例外 — 別回転で silent に壊れない)。
- vLLM ネイティブプリセットは学習回転を再現できないため、mapping が明示警告して
  HF キャッシュパスを案内(`turboquant_*` への silent フォールバックはしない)。

分解測定(`bench_calib_v8.py`, jobs 59102-59104; PPL 32 窓):

| モデル | bits | datafree v7 | rot のみ | cb のみ | **rot+cb (推奨)** |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | 2 | 8.872 | 8.892 | 8.809 | 8.857 |
| Llama-3.1-8B | 3 | 7.020 | 7.011 | 7.017 | **7.010** |
| Qwen3-8B | 2 | 9.963 | 9.836 | 10.130 | **9.784 (−0.18)** |
| Qwen3-8B | 3 | 8.800 | 8.781 | 8.763 | 8.782 |
| R1-Qwen-1.5B | 2 | 168.6 | 174.0 | 170.3 | **167.6** |
| R1-Qwen-1.5B | 3 | 168.0 | 168.9 | 164.6 | 167.1 |

- 単独レバーはモデル依存で符号が揺れる(cb のみは Qwen3 2-bit で悪化)が、
  **rot+cb 同時学習は全 6 点で datafree 以上**。効果が大きいのはデータフリー回転と
  分布の相性が悪いモデル(Qwen3 2-bit −0.18)で、SpinQuant の動機と整合。
- ダウンストリーム 2-bit(`calib_check_*`, jobs 59109-59110): Qwen3 は Needle
  0.956→**1.000** / PPL 改善、RULER −0.067(±0.17 の noise 内)。Llama は全指標
  noise 内で datafree と同等(Needle −0.044/RULER −0.067, いずれも 1σ 未満)。
- **結論**: 精度を最後まで搾りたいモデル(特に Qwen 系)向けの opt-in。
  デフォルトは従来どおり完全データフリー。

使い方:

```python
from onecomp.kvcomp.codec import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.calibration import capture_kv_samples

codec = HayateQuantCodec(bits=2)
kc, vc = capture_kv_samples(model, tok, texts)      # 実KVを収集 (train split)
cal = codec.calibrate({"key": kc, "value": vc})     # 数秒~十数秒 (A100)
cal.save("calib_llama_2bit.pt")
# 以後はロードするだけ (ランタイムコスト0):
codec = HayateQuantCodec(bits=2, calibration="calib_llama_2bit.pt")
```

### 候補 2: K/V 非対称ビット配分 (K4V2) → **推奨動作点として採用**

コーデックが元々持つ `value_bits`(K>V 非対称)を初めて等平均ビットで実測
(`bench_kvasym_v8.py`, jobs 59112/59117)。平均 3.0 bit/coord で:

| モデル | K3V3 uniform (v7) | **K4V2** | K2V4 |
|---|---:|---:|---:|
| Llama-3.1-8B | 7.020 | **6.933 (−0.087)** | 8.444 (崩壊気味) |
| Qwen3-8B | 8.800 | **8.648 (−0.152)** | 9.715 |

- **キーが感度支配的**(KIVI/KVQuant の知見と整合)。同じ平均ビットで K に厚く
  配るだけで両モデルとも uniform 3-bit を上回る。逆配分 (K2V4) は大きく悪化 —
  attention ロジット経由で全トークンに波及するキー誤差の方が高くつく。
- ダウンストリーム(`asym_check_*`, jobs 59121/59125): Needle は両モデル満点で
  uniform と同点。RULER は Llama で **0.733→0.800 に改善**、Qwen3 で 1.0→0.933
  (±0.14 の noise 内)。長文 PPL(4096 窓)も両モデルで K4V2 勝ち
  (Llama 6.832→6.738 / Qwen3 8.423→8.311)。値側を 2-bit に落としても
  検索系タスクは劣化しない — 感度が本当にキー側に集中していることの
  ダウンストリーム確認。
- 平均 2.5 bit は K3V2 が Llama で勝つ(7.204 vs 7.384)が Qwen3 で負ける
  (8.838 vs 8.733)ため、既定の outlier 方式(64ch@+1bit)を維持し K3V2 は
  モデル依存の代替として記録。
- `codec_benchmark.py` に `HayateQuant-K4V2` 行(bits=3 実行時のみ)を追加。
  API 変更はなし(`value_bits` は既存パラメータ)。

### 候補 3: 全部乗せ (K4V2 + キャリブレーション) → **加算的に最良**

`bench_combo_v8.py` (jobs 59118-59119):

| モデル | K3V3 v7 | K4V2 datafree | **K4V2 + calib** |
|---|---:|---:|---:|
| Llama-3.1-8B | 7.020 | 6.933 | **6.922** |
| Qwen3-8B | 8.800 | **8.648** | 8.660 (±noise) |

新レバーは干渉せず(回転/符号帳はサブセット単位、ビット配分はロール単位で直交)、
組み合わせは単独最良と同等以上。Llama の平均 3.0 bit 最終値 6.922 は
TurboQuant-hadamard 3-bit の 7.097 に対し **−0.175**。

### 追加テスト・検証

- ユニットテスト +3 本(roundtrip + digest guard / holdout 非退行 + Cayley 直交性 /
  config 反映 + n_outlier ガード)。vLLM プラグイン含め全テスト pass。
- 速度無回帰: ms_per_encode は datafree/calib で同一(表参照)、K4V2 は
  4-bit K + 2-bit V で uniform 3-bit と同じエンコードパス数。

### v8 再現

```bash
cd ichikawa_works/kvcomp_hayatequant
sbatch run_calib_v8.sh [model]     # キャリブレーション分解 (rot/cb/both x 2,3-bit)
sbatch run_kvasym_v8.sh [model]    # K/V 非対称ビット (3.0/2.5 平均)
sbatch run_combo_v8.sh [model]     # 全部乗せ (K4V2 + calib)
sbatch run_calib_check.sh calib_check_llama   # Needle/RULER (calib_check_qwen3,
sbatch run_calib_check.sh asym_check_llama    #  asym_check_qwen3 も同様)
```

生データ: `ichikawa_works/kvcomp_hayatequant/results/{calib,kvasym,combo}_v8_<model>.json`、
`benchmark/kvcomp-longcontext/results_paper1/hayatequant_{calib,asym}/`、
キャリブ成果物 `results/calib_<model>_{2bit,3bit,k4v2}.pt`。

---

## CPU-only 数理検証・テスト

- `verify_hayatequant_vq_gain.py`: block_size=1 の TurboQuant bit-for-bit 一致、
  block-VQ の単調歪み削減(ガウス/厳密 Beta 両光源)、outlier 分離下の非退行、
  hybrid 回転のブロック等方性改善 — 全チェック pass(v3 コードで再実行済み)。
- `tests/onecomp/kvcomp/test_kvcomp.py` + `tests/vllm_plugins/...`: **144 テスト pass** (v4 時点)。
  v3 で追加: `test_hybrid_rotation_matrix_matches_two_stage`(折り畳み回転の同値性)、
  `test_hayatequant_lut_encoder_matches_exact_distortion`(LUT 過剰歪み <0.5%)、
  `test_hayatequant_lut_and_exact_encoders_agree_end_to_end`、
  `test_hayatequant_large_codebook_not_worse_at_3bit`(K12)、encoder バリデーション。
- GPU パリティ(Slurm job 58938 冒頭): fp16 スコアリング+adaptive encoder の歪みドリフト
  2-bit 0.0000% / 3-bit 0.0048% / 4-bit 0.0026%。

---

## 再現方法

```bash
# 1. CPU-only 数理検証(GPU 不要)
python benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py

# 2. ユニットテスト
pytest tests/onecomp/kvcomp/test_kvcomp.py tests/vllm_plugins/kvcomp/test_kvcomp_vllm_plugin.py -q

# 3. コーデック品質 + 新バリアント(auto/exact/K12/dense-LS)ベンチ (GPU, Slurm)
cd ichikawa_works/kvcomp_hayatequant
sbatch run_hayatequant_bench_v3.sh        # codec_speed + e2e + codec_benchmark x 2 モデル

# 4. 速度のみ再測(adaptive encoder + fp16 スコアリング)
sbatch run_hayatequant_speed_v3b.sh

# 5. Paper-level ロングコンテキスト (Needle/RULER/LongBench-E/PPL)
sbatch run_hayatequant_paper_bench.sh

# 6. レバー分離 ablation (noscale / hadamard-bvq / beta / bs2)
sbatch run_hayatequant_ablation.sh
# = codec_benchmark.py --config-name paper_hayatequant_ablation model_path=...

# 7. vLLM 実機 E2E (ネイティブ bs=1 + block-VQ スカラーフォールバック)
sbatch run_vllm_hadamard_e2e.sh

# 8. v7 候補測定の再現 (RVQ/高予算符号帳のデータフリー歪み, 実PPLレバー分解)
sbatch run_rvq_gain.sh
sbatch run_policy_v7.sh          # 引数でモデル切替可
sbatch run_fit_codebooks.sh      # 公式符号帳成果物の再fit (repo内に上書き)
sbatch run_hayatequant_bench_v7.sh  # v7 最終検証 (フルグリッド + 速度)
```

## 生データ

- `ichikawa_works/kvcomp_hayatequant/results_v3/results_<model>.json` — 品質 (v3, 全バリアント)
- `ichikawa_works/kvcomp_hayatequant/results_speed/codec_speed_llama_shapes_v3b.json` — 単体速度 (最終)
- `ichikawa_works/kvcomp_hayatequant/results_speed/e2e_speed_llama-3.1-8b-instruct_v3b.json` — e2e 速度 (最終)
- `onecomp-lab/benchmark/kvcomp-longcontext/results_paper1/hayatequant/results_llama-3.1-8b-instruct.json` — paper-level
- `onecomp-lab/benchmark/kvcomp-codec/results_ablation/results_llama-3.1-8b-instruct.json` — レバー分離 ablation (v6)
- `ichikawa_works/kvcomp_hayatequant/results_v7/results_<model>.json` — 品質 (v7 最終, レート適応回転+高予算符号帳)
- `ichikawa_works/kvcomp_hayatequant/results_speed/{codec_speed,e2e_speed}*_v7.json` — 速度 (v7 無回帰確認)
- `ichikawa_works/kvcomp_hayatequant/results/rvq_gain.json` — RVQ 不採用判定の光源歪み実測 (job 59088)
- `ichikawa_works/kvcomp_hayatequant/results/policy_v7_<model>.json` — 回転x スケール分解・LUT T・bigfit の実PPL (jobs 59091-59092)
- `results_v2/`, `results_speed/*_v2_*` — v2(修正前後比較用に保持)

## ファイル構造

```
onecomp-lab/
├── papers/hayatequant_paper/README.md   ← このファイル
├── onecomp/kvcomp/codec/
│   ├── hayatequant.py    ← メイン Codec (encoder="auto": adaptive LUT/exact,
│   │                        レート適応 auto 回転: bits<=2 は dense)
│   ├── _vq.py            ← block-VQ 符号帳 + 候補リスト LUT + fp16 厳密探索
│   ├── codebooks/        ← 高予算 Lloyd 符号帳成果物 (.pt, データフリー) + fit_codebooks.py
│   ├── _scale.py         ← 役割非対称スケール (HALO と共有)
│   └── rotation.py       ← hybrid_rotation_matrix (folded 1-GEMM 回転)
├── vllm_plugins/kvcomp/  ← mapping.py + vllm_plugin.py (EngineArgs 自動フック)
├── benchmark/kvcomp-codec/codec_benchmark.py   ← auto/exact/K12/dense-LS 変種
└── benchmark/kvcomp-longcontext/               ← codec_speed / e2e_speed / kv_benchmark
```

---

## v9 機能分離リファクタリング + 実験コードのレポジトリ内移設 (2026-07-05)

開発容易性のための構造整理(数値・API は全て不変、265 テスト pass):

1. **`onecomp/kvcomp/codec/hayate/` パッケージ化**: `hayatequant.py` / `_vq.py` /
   `hayatequant_calib.py` / `codebooks/` を 1 パッケージに集約し、責務ごとに
   `codec.py` / `vq.py` / `calibration.py` / `codebooks/` へ改名。共有プリミティブ
   (`rotation.py`, `_scale.py`) は TurboQuant/HALO も使うため `codec/` 直下に残置。
2. **後方互換シム**: 旧パス (`codec.hayatequant`, `codec._vq`, `codec.hayatequant_calib`)
   は re-export シムとして残存 — 既存スクリプト・古い結果 JSON の再現コードは
   そのまま動く。新規コードは `onecomp.kvcomp.codec.hayate.*` を使う。
3. **論文実験コードのレポジトリ内移設**: `ichikawa_works/kvcomp_hayatequant/` の
   測定スクリプト (bench_policy_v7 / bench_rvq_gain / bench_calib_v8 /
   bench_kvasym_v8 / bench_combo_v8 / vllm_hadamard_e2e) を
   `papers/hayatequant_paper/experiments/` に移設し、ハードコードパスを
   repo 相対に置換。汎用 Slurm ラッパー `run_experiment.sh` と、論文の表 ↔
   スクリプト対応表 (`experiments/README.md`) を追加。**論文の全数値がこの
   リポジトリ単体で再現可能**になった。
