# HALO (enhanced) — method, memo audit, and TurboQuant positioning

This is the authoritative technical write-up for the **enhanced HALO** KV codec:
what it is, exactly how it differs from (and where it can fairly claim
superiority over) TurboQuant, the implementation status of every point in the
review memo, and the benchmark battery that backs the claims.

It is paired with:

* `README.md` — reproduction harness and the running benchmark status log.
* `onecomp/kvcomp/codec/halo.py` — the codec (the module docstring carries the
  same maths; this file is the *evaluation/claim* view).
* `ichikawa_works/kvcomp_halo_asym/` and `ichikawa_works/kvcomp_halo_research/` —
  array-level / CPU / GPU verification scripts.

---

## 1. What HALO is

HALO shares TurboQuant's data-free skeleton — orthogonal rotation → per-vector
scale → Gaussian Lloyd-Max scalar quantization — and adds a small number of
**storage-neutral, data-free** levers. The honest framing (see §3 for why this
matters): the *rotation* and the *norm-corrected key scale* are **not** novel to
HALO (upstream vLLM's TurboQuant already uses a Hadamard transform and a
norm-correction `_nc` preset). HALO's genuine contribution is to **derive the
role-asymmetric scale from a single attention-aligned distortion principle** and
to make the **value** scale MSE-optimal (least-squares), which the shipped
TurboQuant value path (uniform min-max) is not.

### The seven levers and their status

| # | Lever | What it does | Status | Where |
|---|---|---|---|---|
| 1 | **Role-asymmetric scale** | keys: norm-correcting (`nc`) so the logit shrinks by `cosθ` not `cos²θ`; values: least-squares (`ls`, MSE-optimal) | **Implemented** | `halo.py` `key_scale_mode`/`value_scale_mode`, `_fit_scale` |
| 2 | **`ls_iter`** | one Lloyd alternating-min step on values (re-quantise at fitted `s`, re-fit) — monotone-lower MSE, free | **Implemented** | `halo.py` `_fit_scale` mode `ls_iter` |
| 3 | **Outlier mixed-precision** | split `n_outlier` highest-energy channels at `outlier_bits` → true 2.5/3.5-bit operating points | **Implemented** (ported from TurboQuant) | `halo.py` `n_outlier`/`outlier_bits`, `_encode_tensor`/`_decode_tensor` |
| 4 | **Stateless block-wise preconditioning** | per-block mean-center / whiten, no EMA (fixes A-1) | **Implemented** | `halo.py` `_precondition` (EMA removed), `precond_roles` |
| 5 | **2D / lattice codebook (2-bit)** | space-filling gain ≈0.1–0.25 bit | **Future work** (documented) | — |
| 6 | **Decode-side query-only rotation** | rotate query once; aggregate inverse-rotation once per step → O(D²)/step | **Future work** (documented; needs a kernel) | — |
| 7 | **Evaluation rigor** | base-model headline, paired-PPL CIs, S=1 *and* S=1024 timing, diverse needle filler | **Implemented** | `eval/ppl.py`, `eval/longcontext.py`, `codec_speed.py` |

Default behaviour is unchanged: with `key_scale_mode=value_scale_mode=None` and
`n_outlier=0`, HALO reproduces the legacy symmetric codec bit-for-bit, so every
prior result and test still holds.

### The maths (lever 1)

For a rotated vector `x`, chosen code direction `c`, `θ = ∠(x, c)`, recon `x̂ = s·c`:

| scale | formula | ⟨x̂,x⟩ | ‖x̂‖ | MSE ‖x−sc‖² |
|---|---|---|---|---|
| LS  | ⟨x,c⟩/⟨c,c⟩ | ‖x‖²cos²θ | ‖x‖cosθ | ‖x‖²sin²θ (min) |
| NC  | ‖x‖/‖c‖     | ‖x‖²cosθ | ‖x‖ (exact) | 2‖x‖²(1−cosθ) |
| IP  | ‖x‖²/⟨x,c⟩  | ‖x‖² (exact) | ‖x‖/cosθ | (largest) |

Identity `s_NC = √(s_LS·s_IP)`. Small-angle: `2(1−cosθ)=θ²+O(θ⁴)`,
`sin²θ=θ²−O(θ⁴)`, so the **NC MSE penalty over LS is O(θ⁴)** (negligible at
3–4-bit) while the **systematic logit shrink halves from cos²θ to cosθ**. Keys
enter softmax through an inner product (a logit-faithful objective); values
enter the output linearly `Σaᵢvᵢ` (an MSE objective). Hence NC keys + LS values.
All scales store the same single fp16 per vector ⇒ storage-neutral. Verified
element-by-element in `ichikawa_works/kvcomp_halo_asym/01_verify_asym_scale.py`.

---

## 2. Review-memo audit — every item

Legend: ✅ implemented/fixed · 📝 documented (deliberate, with rationale) ·
🔭 future work.

### A. Correctness / design

| Item | Issue | Status | Resolution |
|---|---|---|---|
| **A-1** | HALO `_ema` mixed statistics across layers/requests (shared codec, no ctx) | ✅ | EMA removed; `_precondition` is exactly stateless (per-block `μ`/`w` stored in payload). No `_ema` symbol remains. Test `test_halo_stateless_precond_no_ema`. |
| **A-2** | `residual_window` is a dead parameter (no cache honours it) | ✅📝 | `_base.py` docstring now states the research caches quantise every block and do **not** honour the window; codecs are measured *without* a residual window (a strictly harder setting than KIVI). A real cache-side implementation would populate `CompressedKV.key_residual`/`value_residual`. |
| **A-3** | TurboTaper floor accounting vs codec selection mismatch (`n_outlier_floor>0`) | 📝 | Paper-2 (TaperKV/TurboTaper) scope, not HALO. Noted for the recipe paper; HALO does not use the floor codec. |
| **A-4** | Re-quantisation drift in graded caches (progressive error) | 📝 | Paper-2 scope. TurboTaper already mitigates with delta-quant (see README "TurboTaper bugs"); the "not direct-from-source" caveat is documented there. |
| **A-5** | ThinKVCache silently falls back to uncompressed for batch>1 | ✅ | A warning is now emitted in `thinkv_cache.py update()`. (transformers-internal-API dependence and `_pending_sparsity` remain documented limitations of that research cache.) |
| **A-6** | Affine-fallback bit accounting omitted scale/zero meta (over-stated compression) | ✅ | HALO + TurboQuant fallback now count `(#vectors)·2·16` for the per-token affine scale+zero. Test `test_fallback_affine_accounting_counts_scale_zero`; verified in `04_verify_outlier.py`. |
| **A-7** | TurboQuant QJL counted the 1-bit sign but not `qjl_rms` (per-vector fp16) | ✅ | `_stored_bits` now adds `#vectors·16` when `use_qjl`. Test `test_turboquant_qjl_counts_residual_rms`. |
| **A-8** | Outlier channel set is per-block dynamic (q_len=1 mask jitters) | ✅📝 | Documented as a layout caveat in both `turboquant.py` and `halo.py` docstrings (exact in fake-quant; a real kernel cannot express a per-step-dynamic mask statically). |

### B. Claims / maths (HALO)

| Item | Issue | Status | Resolution |
|---|---|---|---|
| **B-1** | "strictly higher quality" contradicted by own PPL (no error bars; R1 regime unreliable) | ✅ | Docstring no longer claims strict dominance; it claims the *speed-quality Pareto* vs the shipped default and is explicit that preconditioning is OFF by default (no robust PPL win). PPL is now paired-window with CIs (`eval/ppl.py`); the headline is base Llama. |
| **B-2** | LS scale can hurt keys (logit shrink) — the central point | ✅ | This is exactly what lever 1 fixes: NC keys. The whole role-asymmetric design is the answer to B-2; closed-form + array-verified. |
| **B-3** | HALO had no `n_outlier` → could not reach 2.5/3.5-bit | ✅ | Lever 3 implemented; HALO now produces the 2.5/3.5-bit operating points iso-memory to TurboQuant (`test_halo_outlier_matches_turboquant_storage`). |

### C. Evaluation harness

| Item | Issue | Status | Resolution |
|---|---|---|---|
| C-timing | `_ms_per_encode` is encode+decode, S=1024 only; CUDA-sync breaks CPU | ✅ | `codec_speed.py` now reports encode **and** decode separately, in **both** prefill (S=4096) and decode (S=1) regimes, with throughput (Melem/s, tok/s/layer); CPU path guarded. |
| C-needle | haystack was the same 4 sentences repeated (low-entropy, too forgiving) | ✅ | `eval/longcontext.py` now samples a 30-sentence diverse bank without immediate repetition (`_build_filler`), seeded/deterministic. Applies to both needle and RULER multi-key. |
| C-position | `greedy_answer` position/cache_position after eviction | 📝 (already correct) | Confirmed correct; kept. |

### vLLM plugin

| Item | Issue | Status | Resolution |
|---|---|---|---|
| 1 | algorithm-semantics swap silent (Lloyd-Max keys / uniform values / FP8 k8) | ✅ | `mapping.py` warns for `turboquant` *and* `halo` that the preset is not bit-identical to OneComp's rotated-Lloyd-Max-on-both path. |
| 2 | `n_outlier`/`use_qjl`/`dist`/`seed` dropped silently | ✅ | Warnings emitted when `n_outlier>0` (dropped) and `use_qjl=True` (upstream omits QJL by design). |
| 3 | `skip_layers` not propagated to `--kv-cache-dtype-skip-layers` | ✅ | Propagated in `mapping.py` (`plan.skip_layers`) and applied in `vllm_plugin.py` (`kv_cache_dtype_skip_layers`). Test `test_vllm_mapping_turboquant_warns_semantics_and_propagates_skip_layers`. |
| 4 | explicit user `kv_cache_dtype` is overridden with a warning | 📝 | **Deliberate**: the codec's dtype wins and a conflict warning is logged (tested contract). The conflict is surfaced loudly; respecting the codec keeps the config the source of truth. |
| 5 | `read_kv_compression_config` silent no-op on HF Hub ids | ✅ | Now logs an explicit info line when no local `config.json` is found. |

### Levers 5 & 6 (future work — honest scoping)

* **Lever 5 (2D/lattice codebook).** Expected gain is only 0.1–0.25 bit/coord
  (space-filling gain: 1.53 dB asymptotic, ~0.65 dB for E8). Worth it only to
  lift the 2-bit cell; documented as optional future work rather than shipped, to
  avoid over-claiming.
* **Lever 6 (decode-side query-only rotation + aggregated inverse).** A genuine
  asymptotic decode speedup (O(S·D²)→O(D²)/step) but it requires a custom
  attention kernel; the fake-quant research layer cannot express it. Documented
  as the system/kernel extension; the current `codec_speed.py` already shows the
  decode-regime (S=1) GEMM advantage of the encode path.

---

## 3. Positioning vs TurboQuant (fair, and where HALO wins)

Source: a close read of arXiv:2504.19874 (Zandieh et al., Google; ICLR 2026) and
the merged vLLM integration (PR vllm-project/vllm#38479).

**What we must NOT claim as novel** (or reviewers will reject it):

* **Norm-corrected keys.** vLLM's `turboquant_*_nc` presets already re-normalise
  the centroid to unit norm and multiply by the stored ‖k‖ — algebraically the
  same `‖k‖·(c/‖c‖)` as HALO's `nc` key scale. HALO's contribution is to *derive*
  it from the attention-logit argument and unify it with the value side, not to
  invent norm correction.
* **Hadamard rotation.** vLLM already replaced the paper's QR random orthogonal
  with a Walsh–Hadamard transform. HALO's "Hadamard-GEMM" is a speed/engineering
  framing of a standard transform (DRIVE/QuaRot lineage), not a new idea.

**Where HALO can fairly claim superiority:**

| Axis | TurboQuant (paper) | TurboQuant (vLLM) | **HALO** |
|---|---|---|---|
| Key dequant | `Q_mse` (shrinks) or +QJL (unbiased, +1 bit) | NC (`‖k‖·ĉ/‖ĉ‖`) | NC (same idea, **derived**) |
| **Value dequant** | same as key (`Q_mse`/`Q_prod`) | **uniform min-max** (scale+zero) | **least-squares (MSE-optimal)** |
| Role asymmetry | none (generic VQ) | empirical only | **first-principles (logit vs MSE)** |
| **QJL** | core (1-bit residual for unbiased IP) | omitted (variance through softmax) | **omitted + bias–variance argument** |
| Rotation | QR (O(d³) setup) | WHT | Hadamard-GEMM |
| Calibration | data-free | data-free | data-free |
| Extra storage | ‖·‖ fp16 | ‖·‖ fp16 | **identical** |

The four superiority axes, in decreasing strength:

* **(A) Role-asymmetric scale from one principle.** The TurboQuant paper offers
  `D_mse` and `D_prod` as *separate algorithms* and never assigns "keys=logit /
  values=MSE" inside one KV codec. vLLM does it empirically (and leaves values as
  uniform). HALO turns the community's empirical finding ("cosine passes but
  reasoning breaks"; K and V have different precision sensitivity) into a design
  rule. This is the headline novelty.
* **(B) MSE-optimal values.** vLLM TurboQuant values are uniform min-max, which
  does not minimise the reconstruction MSE that dominates the attention output
  `Σaᵢvᵢ`. HALO's least-squares value scale is the exact per-direction MSE
  minimiser, and `ls_iter` lowers it further (monotone). Directly comparable on
  value-relMSE.
* **(C) QJL-free logit fidelity.** The paper needs a +1-bit QJL residual to
  de-bias the inner product. HALO (and vLLM-NC) get logit fidelity from the NC
  scale at **no extra bit**, reusing the stored norm.
* **(D) Bias–variance argument for dropping QJL.** Multiple independent groups
  found QJL is unbiased but high-variance through softmax and *hurts* attention.
  HALO's low-variance, small-bias NC logit is the principled alternative — a
  direct rebuttal of the paper's "unbiased IP needs QJL" thesis.

**What we must concede** (reviewer-proofing): TurboQuant proves a near-optimal
*generic* distortion-rate (≤ √3π/2 ≈ 2.7× the Shannon lower bound, exponential
4⁻ᵇ rate). HALO does not beat that *generic* theory. Our framing: generic VQ
optimality ≠ attention-end-to-end optimality; HALO optimises a **task-aligned**
distortion (keys→logit, values→reconstruction MSE), which is the right objective
for the softmax+weighted-sum that actually consumes the KV cache.

---

## 4. Benchmark battery (what proves the claims)

All enhanced-HALO benchmarks are self-contained against the references they must
beat (FullKV, TurboQuant, legacy HALO) at **iso-bit / iso-memory**, and write to
dedicated `results_halo_enh_*` dirs so they never clobber other paper JSONs.

| Axis | Harness / config | Operating points | Result dir |
|---|---|---|---|
| Retrieval (Needle/RULER/PPL) | `kv_benchmark.py` `turboquant_paper[_ministral]` | 2 / **2.5** / 3 / **3.5** / 4-bit × {TurboQuant, HALO, HALO-NC, HALO-NCIT} | `results_halo_enh_{llama,minis}_retr` |
| LongBench-E downstream | same, `run_longbench=true` | same | `results_halo_enh_llama_lb` |
| Reasoning (GSM8K/MATH/PPL) | `paper_aime` | 3 / 4-bit × {HALO, HALO-NC, HALO-NCIT} | `results_halo_enh_basic_{r1q15,r1l8,r1q7,r1q14,qwq}` |
| Codec speed × quality | `codec_speed.py` | prefill S=4096 + decode S=1, bits 2/3/4, incl. NC/NCIT | `results_codec_speed/codec_speed_enh_*` |
| End-to-end speed | `e2e_speed.py` | TTFT / decode tok/s / ms-tok / peak mem | `results_codec_speed/e2e_*` |

The 2.5/3.5-bit points are **true** 2.5/3.5-bit: `64 ch @ b+1 + 64 @ b` on
head_dim=128 (the earlier `n_outlier=32` was an effective 2.25/3.25-bit; both
TurboQuant and HALO now use the same 64-split so the comparison is exactly
iso-memory).

Submit:

```bash
cd benchmark/kvcomp-longcontext/paper
bash submit_halo_enhanced.sh retrieval   # Llama + Ministral, full 2/2.5/3/3.5/4 grid
bash submit_halo_enhanced.sh reason      # R1/QwQ reasoning non-degradation
bash submit_halo_enhanced.sh speed       # codec + e2e latency/throughput
```

---

## 5. Verification artifacts (data-free, before any GPU run)

| Script / test | Proves |
|---|---|
| `ichikawa_works/kvcomp_halo_asym/01_verify_asym_scale.py` | lever-1 identities (`s_NC=√(s_LS·s_IP)`, MSE gaps), lever-2 monotonicity, lever-4 statelessness, storage neutrality, round-trip |
| `ichikawa_works/kvcomp_halo_asym/04_verify_outlier.py` | lever-3 round-trip on pow2 / non-pow2 dims, mask cardinality, effective bit-width, iso-storage with TurboQuant, A-6/A-7 accounting |
| `tests/onecomp/kvcomp/test_kvcomp.py` (93 tests) | scale-mode storage neutrality, NC norm preservation, `ls_iter` monotone MSE, outlier round-trip/accounting, fallback + QJL accounting, vLLM mapping warnings + skip_layers |

All 93 unit tests pass; both array-level scripts exit 0.

---

## 6. Known caveats / open items (do not hide these)

* Levers 5 and 6 are documented future work, not shipped (§2).
* `residual_window` is config-only; results are reported *without* a
  full-precision residual window (A-2).
* The outlier channel mask is per-block dynamic; faithful in fake-quant, a layout
  caveat for a real kernel (A-8). State this in the paper.
* vLLM mapping overrides an explicit `kv_cache_dtype` with a warning (a
  deliberate, tested choice — item 4).
* Whitening (`whiten=True`) lowers raw MSE but can raise PPL via the `1/w`
  un-whiten on high-variance value channels; OFF by default.
