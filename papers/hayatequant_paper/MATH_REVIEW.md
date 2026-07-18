# HayateQuant 数理健全性レビュー (v3)

> **Legacy note.** This review predates the signed-fp16 metadata contract and
> the fixed b=2 HQ K5/V3--K9/V7 NC/LS-reassign-refit recipe. Statements of
> bit-for-bit TurboQuant equivalence below apply only to ideal real arithmetic
> (and the Gaussian/RMS parameterization); finite-Beta L2-versus-RMS fp16 scale
> rounding can change reconstructed low-precision values.  Current empirical
> claims require the formal rerun.

対象: `onecomp/kvcomp/codec/{hayatequant.py, _vq.py, _scale.py, rotation.py}` (v3,
Folded Hybrid Rotation + 候補リスト LUT + fp16 探索)。
各主張を「厳密に成り立つもの」「確率的/漸近的に成り立つもの」「近似であり実測で
統制しているもの」に分類し、既知の限界を列挙する。

---

## 1. 厳密に成り立つ性質 (証明ベース)

### 1.1 Folded Hybrid Rotation の直交性・同値性

2段変換は行ベクトル規約で `y = (x @ HS) @ B`、`B = blockdiag(Q_1,...,Q_{D/b})`。
折り畳み行列は `W = HS @ B`。

- **直交性**: `HS` は `diag(s)H/√D` (s∈{±1}^D, H は Sylvester-Hadamard) で
  `HS·HSᵀ = I`。各 `Q_i` は QR 由来の Haar 直交行列なので `B·Bᵀ = I`。
  直交行列の積は直交 ⇒ `W·Wᵀ = I`。**逆変換 `xr @ Wᵀ` は厳密**。
- **同値性**: `(x@HS)@B = x@(HS·B)` は行列積の結合律そのもの。ブロック対角性から
  `W[:, i·b:(i+1)·b] = HS[:, i·b:(i+1)·b] @ Q_i` と列ブロック毎に構成でき、
  実装(`_hybrid_matrix_cpu`)はこれを float64 で合成後 float32 に落とす。
  数値誤差は合成 1 回分の丸めのみで、テストで max|Δ| < 1e-4 を固定
  (`test_hybrid_rotation_matrix_matches_two_stage`)。
- **Q_i のシード一致**: 折り畳み側と 2 段参照側で同一シード
  (`seed*100003 + i`) を使うため、両パスは同じ変換を表す (テストで固定)。

**結論: 折り畳みは数学的に無損失。** 品質へ影響し得るのは fp32 化の丸めのみで、
これは従来の 2 段実装 (fp32 butterfly + fp32 einsum) と同水準。

### 1.2 ストレージ中立性

ブロック符号は 1 ブロックあたり `bits·b` ビット (`K = 2^{bits·b}`) なので
`(D/b)·bits·b = D·bits`。スケールは block_size に依らず 1 ベクトル 1 個 (fp16)。
⇒ **TurboQuant/HALO と同じ論理 payload rate での比較**（実 allocation は別途測定）
(`test_hayatequant_storage_neutral_across_block_size`)。

### 1.3 block_size=1 の scalar compatibility（明示的に knobs を一致させた場合）

`b=1` のとき `_block_codebook` はスカラー Lloyd-Max 符号帳を `(2^bits, 1)` に
reshape して返し、探索は TurboQuant と同じ sorted-boundary searchsorted に
ディスパッチされる。スケールモード rms/nc/ls/ls_iter は HALO と同一実装
(`_scale.py` を共有)。⇒ **同じ量子化式を再現**（Beta/L2 対 RMS の fp16
metadata 丸めは bit-for-bit 一致を保証しない）
(`test_hayatequant_block_size_1_matches_*`、数値一致 <1e-6 で固定)。
したがって「HayateQuant は理論上どの設定でも両者より悪くなり得ない」は、
*その設定を選べば常に再現できる* という意味で厳密に正しい。

### 1.4 スケールの最適性 (役割非対称スケール)

- LS スケール `s = ⟨x,ĉ⟩/‖ĉ‖²` は `min_s‖x − s·ĉ‖²` の一意解 (凸 2 次)。
  ĉ がスカラー中心の連結でもブロック符号語の連結でも、最後の次元での縮約しか
  使わないため**ブロック次元に依存しない** (次元非依存性は実装通り)。
- NC スケール `s = ‖x‖/‖ĉ‖` は `‖x̂‖=‖x‖` を厳密に保つ。恒等式
  `s_nc = √(s_ls·s_ip)` と、小角展開での MSE ペナルティ O(θ⁴) も block-VQ で
  そのまま成立 (θ は x と ĉ のなす角、という定義が次元非依存のため)。

---

## 2. 確率的・漸近的に成り立つ性質 (TurboQuant と同型の保証)

### 2.1 ソース分布の仮定と回転の役割

TurboQuant Lemma 1 (「Haar ランダム回転後の座標は Beta((d−1)/2,(d−1)/2)」) が
**厳密に**成り立つのは Haar ランダム **dense** 回転のみ。signed-Hadamard は
「±1/√D の重み和」なので座標周辺分布は CLT 的に近似ガウスに過ぎない。
これは TurboQuant/HALO/QuaRot 系すべてが暗黙に受け入れている近似で、
HayateQuant が新たに導入した仮定ではない。

### 2.2 Hybrid 回転が block-VQ に与える保証

block-VQ 符号帳は等方的な `N(0,1)^b` (または厳密 Beta ブロック周辺) に
フィットされるため、各ブロックの回転後分布が等方的である必要がある。

- **Hadamard のみ**: H は決定的行列 (ランダム性は符号 s のみ) なので、ブロック内
  共分散は入力の生共分散の構造を引きずる(非等方)。
- **Hybrid**: ブロック値 v = (HSx)_block に対し y = v @ Q_i、Q_i は Haar。
  Haar 回転の定義より、**‖v‖ を固定した条件付きで y の方向はブロック球面上一様**、
  したがって条件付き 2 次モーメントは `E[yᵀy | ‖v‖] = (‖v‖²/b)·I`。
  つまり**ブロック内の非等方性は Haar 描画に関して厳密に消える**。
  ただし Q_i はモデルごとに 1 回固定するので、これは「ランダム回転で高確率」型の
  保証 — TurboQuant 自身の保証と同じ型である。
- 残る近似: (a) ブロックの**大きさ** ‖v‖ の分布は入力依存 (ガウスなら χ 分布)、
  (b) ブロック間の独立性は Hadamard の大域ミキシングによる近似。
  どちらも `verify_hayatequant_vq_gain.py` / `verify_hayatequant_isotropy.py` で
  実測検証しており、等方性(オフ対角エネルギー)は hadamard-only 0.433 → hybrid
  0.427 と改善方向、歪みは b=1→2→4 で単調減少 (space-filling gain の理論曲線と
  整合) を確認。

### 2.3 「Hadamard < Dense (精度)」という経験則の数理的整理

同僚コメント「精度的に Hadamard < Dense なので Dense を使うのが良い」について:

- **incoherence の観点では逆向きの厳密な不等式がある**: Hadamard は
  `max_ij |H_ij| = 1/√D` (最小可能値) を厳密達成し、座標整列した外れ値の
  ℓ∞ 増幅を最小化する。Haar dense は w.h.p. `max|W_ij| ≈ √(2 ln D)/√D` で
  約 √(2 ln D) ≈ 3.1 倍 (D=128) 悪い。**worst-case 外れ値の整形では Hadamard 優位**
  (QuIP/QuaRot 系の標準的議論)。実測でも **歪み (relMSE) は Llama 系で
  Hadamard 系が優位**が多い。
- 一方 **PPL では dense が勝つ状況が実在する** (実測: R1-Qwen 全 bit 幅、
  Llama 2-bit)。歪みが同等以上でも PPL が良い、という「歪みと PPL の順位逆転」が
  観測されており、これは平均 MSE では捉えられない誤差の**構造**
  (決定的変換 H は誤差パターンがトークン間で相関しやすい、dense は誤差を
  デコリレートする) と、Qwen 系固有の巨大外れ値チャネルの相互作用と解釈できる。
  つまり「Hadamard < Dense」は**一様な真理ではなく、モデル族依存の経験則**
  (Llama 3/4-bit では hybrid/hadamard 系が PPL でも最良)。
- **HayateQuant v3 での帰結**: 回転コストは folded 化により Hadamard / Dense /
  Hybrid すべて**同一 (1 GEMM)** になったため、この選択は**純粋に品質側の
  自由度**として残せる。`rotation_mode="dense"` + block-VQ (= HayateQuant-dense-LS)
  は dense の PPL 優位と block-VQ の歪み優位を合成し、実測で両モデルの 2-bit
  PPL 最良値を更新した。**「大人しく Dense」はそのまま採用可能で、しかも
  block-VQ の上乗せがある**というのが本実装の回答。

### 2.4 fast-hadamard-transform カーネルとの関係 (トレードオフの所在)

Dao-AILab のカーネル (quant_ref/fast-hadamard-transform に取得済み) は
FWHT を fp16/bf16/fp32 で **memcpy の 1.0–1.8 倍** の時間で実行する
(= 完全にメモリ帯域律速、D ≤ 32768)。計算ノード (CUDA 13.0) と venv の
torch (CUDA 12.8) の版不一致でカーネル本体のビルドは不可だったため、
README 記載の「dim≤512 で memcpy の 1.0 倍」に基づき **memcpy をカーネル
実行時間のプロキシ**として dense GEMM 回転と比較した (job 58947, B200):

| D | N | memcpy (≈FHT kernel) | dense GEMM fp16 | dense GEMM fp32 |
|---:|---:|---:|---:|---:|
| 128 | 131072 | 0.0083ms | 0.0124ms (**1.5x**) | 0.1024ms (12x) |
| 256 | 65536 | 0.0083ms | 0.0145ms (1.8x) | 0.1801ms (22x) |
| 4096 | 4096 | 0.0083ms | 0.0955ms (**12x**) | 2.41ms (291x) |
| 8192 | 4096 | 0.0246ms | 0.3959ms (**16x**) | 8.37ms (340x) |

- **KV per-head 回転 (D=128–256) では fp16 dense GEMM は FHT カーネル比
  高々 1.5–1.8 倍、絶対値で ~0.01ms** — エンコード全体 (0.5–4.3ms) の 1% 未満
  であり、トレードオフとして意味を持たない。
- **FWHT カーネルの優位が実在するのは D ≥ 数千** (QuaRot のような hidden
  次元全体の回転) で、GEMM が計算律速に転じ 12–16 倍差になる領域。
  KV cache 量子化 (per-head D=128) はこの領域に入らない。
- したがって「速度は FWHT、精度は Dense のトレードオフ」は **D=128 の
  KV 量子化には実質存在しない**: folded W (dense でも hybrid でも) 1 GEMM で
  精度側の自由 (dense/hybrid) を速度ペナルティなしで選べる。参考: 本リポジトリの
  旧 butterfly 実装 (純 PyTorch) が 1.7ms と遅かったのは log₂D 回のメモリ往復の
  ためで、Dao カーネルはこれを 1 パスに融合するが、それは folded GEMM が既に
  達成している水準と同オーダー。
- 副次的発見: 現行実装の回転 GEMM は fp32 (0.10ms) であり、fp16 化すれば
  さらに ~8 倍 (0.012ms) 縮む。decode 側 per-layer dequant (56µs) の主成分が
  この逆回転 GEMM なので、fp16 化は decode をさらに押し上げる余地がある
  (将来最適化として記録)。

---

## 3. 近似であり実測で統制している箇所 (既知の限界)

### 3.1 候補リスト Grid-LUT エンコーダ

- **近似の所在は割当のみ**: 復元は常に厳密符号語、距離計算も候補 T 個に対しては
  厳密。誤りうるのは「真の最近傍がセルの top-T 候補に入っていない」場合のみ。
- **境界の性質**: 格子は [-5,5]^b、外側は clamp。b=1 では最外 Voronoi 領域が
  非有界なので clamp は厳密に無害。b>1 では「clamp 先セルの候補リストが真の
  最近傍を含む」は定理ではなく、外向き Voronoi 錐の幾何からの妥当な期待
  + 実測 (一致率 ≥ 99.5%、過剰歪み ≤ 0.06%) で統制。
- **理論的上界**: 過剰歪みはセル幅 h に対し O(h·√b·(境界密度)) で減衰し、
  T を増やすと指数的に減る。デフォルト (b=2: 256 bins/T=2、b=4: 32 bins/T=8–16)
  は `test_hayatequant_lut_encoder_matches_exact_distortion` (<0.5% 検定) で固定。
- **落とし穴だった点 (修正済み)**: top-1 のみの純格子 LUT は b=4/K=256 で
  過剰歪み 3.3% を出した (Voronoi 境界跨ぎ)。これが候補リスト化 (T≥8) の理由。

### 3.2 fp16 テンソルコア厳密探索

score = z·c − ½‖c‖² の比較を fp16 で行う。argmax が反転するのは 2 符号語の
score 差が fp16 分解能以下の場合のみで、その際の過剰歪みは
`Δd² = 2·Δscore ≤ 2·ulp(score)`。z, c = O(数σ) なので score = O(10)、
ulp ≈ 0.01 ⇒ ブロックあたり最大 ~0.02 の希な加算。実測ドリフト:
2-bit 0.0000% / 3-bit 0.0048% / 4-bit 0.0026% (GPU パリティ、job 58938)。
なお fp32 蓄積は torch.matmul のテンソルコア経路が保証。

### 3.3 ls_iter の単調性

古典的な交互最小化の単調性 (再量子化 → スケール再フィットで MSE 非増加) は
**割当ステップが厳密最近傍であること**を要求する。LUT/fp16 探索では割当が
ε-準最適になり得るため、単調性は「ε まで」に弱まる。実測では end-to-end
パリティ (±2% 以内、実測はほぼ 0) で統制。厳密性が必要な場合は
`encoder="exact"` (fp32 は CPU パス) が常に利用可能。

### 3.4 データ不要 K-means 符号帳のフィット品質

- 符号帳はソース法則からの合成サンプルで学習するため「データ不要」の主張は
  TurboQuant のスカラー符号帳と同じ意味で正しい (実 KV を見ない)。
- K=4096 (K12, b=4) では n_samples=100k ⇒ 1 符号語あたり ~24 サンプルと薄く、
  Lloyd 解の分散が残る (multi-restart ×4 で緩和)。それでも実測歪みは
  デフォルト比 -15〜19% 改善しており、サンプル増は更なる改善余地 (将来課題)。
- 2-bit で外れ値分割なしだと block-VQ が bs=1 に劣化し得る事象は検証済みで
  (`verify_hayatequant_vq_gain.py` Test 4)、`n_outlier` 併用で解消される。
  paper ベンチの 2.5/3.5-bit 点はこの経路を通る。

### 3.5 RMS スケールとブロック間相関

1 ベクトル 1 スケール (RMS) の規約では、有限 d でベクトルを RMS 正規化した
座標は厳密には独立でない (球面拘束による負の相関 O(1/d))。`dist="beta"` が
有限 d の厳密ブロック周辺分布を提供するが、ブロック**間**の同時法則までは
モデル化しない (これも TurboQuant のスカラー版と同型の近似)。d=128 では
補正は O(1%) 未満で、実測上 gaussian/beta の差は歪み 4 桁目。

---

## 4. 総括

| 主張 | 保証の型 | 根拠 |
|---|---|---|
| 折り畳み回転の可逆性・同値性 | 厳密 | 直交群の閉性、結合律、テスト |
| ストレージ中立 | 厳密 | ビット勘定、テスト |
| TurboQuant/HALO の scalar control (bs=1) | codes/rate/実数演算で一致 | fp16 scale parameterization の丸め差を契約テストで限定 |
| block-VQ の歪み優位 (等方ソース) | 漸近 + 実測 | space-filling gain 理論、verify スクリプト |
| hybrid 回転のブロック等方化 | Haar 描画に関して厳密 (固定後は高確率型) | §2.2、isotropy 実測 |
| LUT / fp16 の品質パリティ | 近似 (実測統制 <0.06% / <0.005%) | パリティテスト、GPU 実測 |
| dense vs hadamard の優劣 | モデル族依存の経験則 (両論あり) | §2.3、両モデル実測 |
| FWHT カーネルの適用範囲 | D≥数千で有効 (12–16x)、D=128 では GEMM 比 1.5x 以内 | §2.4、job 58947 実測 |

**数理的に致命的な誤りは見当たらない。** 非厳密な箇所はすべて (i) TurboQuant
自身が依拠しているのと同型の近似か、(ii) 実測で定量統制された工学的近似であり、
その統制値 (LUT 過剰歪み ≤0.06%、fp16 ドリフト ≤0.005%) は手法間の性能差
(歪み -9〜19%、PPL -0.1〜-1.0) より 2〜3 桁小さい。
