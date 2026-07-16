# HQ high-budget Gaussian artifact round

Status: complete and rejected before model-quality measurement.

The six-target candidate and whole-bundle rule were frozen in
`PREREGISTRATION.md` before generation. Slurm array `69346_[0-5]` generated
all six tensors, then every task failed only while JSON-serializing PyTorch's
private `_CUuuid` object. The already-written `.pt` files were immediately
frozen in `FAILED_FIT_ARTIFACT_SHA256.txt`; none was overwritten or refit.

Metadata-only recovery array `69354_[0-5]` loaded those exact frozen tensors,
validated shape/finiteness/unique rows, and recomputed the preregistered
held-out Gaussian MSE against the current online fallback. Recovery metadata
explicitly records `artifact_refit=false` and `artifact_overwritten=false`.
Tasks 1 and 3 exit nonzero by design after writing their negative metadata.

| Target | Artifact MSE | Online MSE | Relative delta | Gate |
|---|---:|---:|---:|---:|
| b2 K8 | 0.2013945321 | 0.2016081913 | -0.1060% | pass |
| b2 K32 | 0.0577024865 | 0.0574220620 | +0.4884% | **fail** |
| b2 K128 | 0.0153720827 | 0.0155092619 | -0.8845% | pass |
| b2 K512 | 0.0042548947 | 0.0040562060 | +4.8984% | **fail** |
| b4 K64 | 0.1848507581 | 0.1860462967 | -0.6426% | pass |
| b4 K1024 | 0.0506770014 | 0.0518326801 | -2.2296% | pass |

The frozen rule requires strict source-MSE improvement for every target.
b2 K32 and K512 fail, so the whole six-file bundle is rejected. No WikiText
quality, native ABBA, production recipe, formal input, or paper value was run
or changed for this rejected bundle. In particular, the four passing tensors
cannot be cherry-picked and the previous online-fit measurements cannot be
used as a post-hoc fallback within this round.

`analysis.json` is the machine-readable decision. All failed/successful logs,
metadata, tensor/file hashes, commands, scripts, and Slurm accounting are
retained. A separate non-selection serialization round may materialize the
unchanged current online-fallback tensors for reproducibility; it is not this
rejected high-budget quality candidate.
