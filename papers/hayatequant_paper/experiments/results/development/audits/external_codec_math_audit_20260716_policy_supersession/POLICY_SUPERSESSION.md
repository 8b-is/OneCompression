# External-codec formal-policy supersession (2026-07-16)

## Authority and integrity

This is the current repository policy for interpreting the immutable
`external_codec_math_audit_20260716` evidence bundle. It is a maintainer policy
attestation rooted at source commit
`8d2c073595de29a63dda4161792fad3b82c054f0`. The machine-readable statement is
`policy_supersession.json`; `SHA256SUMS` binds both files. The final Git commit
that introduces this directory supplies the repository's content-addressed
attestation. No external identity-signature claim is made.

The historical bundle is not rewritten. In particular, its original
`SHA256SUMS` has SHA-256
`c2c0a474360d316ff9c743df2954e73b51da04215661428bd0731d23cf236125`,
and the original README and JSON remain byte-identical.

## Superseded assertions

Only the following historical policy assertions are superseded:

- `external_codec_math_audit_20260716/README.md:15` and lines 154--157 say
  that BlockQuant belongs in the predeclared quality denominator;
- JSON pointer
  `/formal_contract/BlockQuant-paper/include_in_predeclared_quality_denominator`
  in `external_codec_math_audit_20260716/audit.json` has historical value
  `true`.

Those statements remain visible as historical evidence, but their current
value is `false`. The mathematical audit, source hashes, codec identities,
stored-bit examples, test records, and reproducibility limitations in that
bundle are not superseded.

## Current formal policy

`TurboQuant-paper` and `FibQuant-paper` are the matched quality baselines.
Each contributes 164 quality cells per proposal. TurboQuant additionally
contributes 60 public/research speed cells per proposal. With HQ and AdaptiveHQ,
the strict point-estimate gate therefore contains
`2 * (164 + 164 + 60) = 776` cells.

`BlockQuant-paper` remains a required independent quality baseline. Its 80
applicable cells per proposal, 160 in total, must be retained in the complete
report and descriptive ledger. Its dynamic top-32 identity-bit-counted source
payload is not rate-matched to HQ, however, so it has no dominance threshold,
pass/fail decision, or cells in the 776-cell gate. Counting an unmatched
operating point as a matched dominance win would be scientifically incoherent.

The integer labels 2/3/4 are scalar or block-code **payload-rate slots**.
For TurboQuant-paper and FibQuant-paper at head dimension 128, one signed-fp16
norm per role vector makes the exact logical stored rate `r + 0.125` bits per
coordinate. Fixed HQ has the same exact total at the corresponding slot.
BlockQuant's slot-3 label is only a matrix slot: its exact stored rate includes
dynamic identities and depends on the encode-update length.

## Packed-native comparator scope

The separate packed-native matrix compares faithful fixed HQ with the
unmodified upstream vLLM 0.20.0 native TurboQuant presets at nominal code
payload rates 3 and 4. That upstream implementation is not the public
`TurboQuant-paper` Algorithm-1 baseline: it uses a Hadamard transform,
Lloyd--Max keys, and uniform-affine values with an fp16 scale and zero point.

The comparison reports each algorithm's actual packed slot footprint; it is
not an exact-stored-bit-matched comparison. At head dimension 128, HQ/upstream
slot widths are 100/102 bytes at rate 3 and 132/134 bytes at rate 4. The vLLM
0.24.0 records in the paper README are historical plugin/mapping compatibility
smoke evidence, not part of the current packed-native formal matrix.

## Authoritative live sources

- `benchmark/kvcomp-codec/aggregate_hayatequant_paper_shards.py`
- `benchmark/kvcomp-codec/render_hayatequant_formal_results.py`
- `benchmark/kvcomp-codec/FORMAL_PAPER_RESULTS.md`
- `benchmark/kvcomp-codec/hayatequant_native_speed_formal_contract.json`
- `onecomp/kvcomp/codec/quality_baselines.py`
- `onecomp/kvcomp/codec/{turboquant,fibquant,blockquant}.py`

