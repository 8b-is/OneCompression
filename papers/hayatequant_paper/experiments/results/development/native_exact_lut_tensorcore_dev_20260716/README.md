# Exact LUT tensor-core development diagnostic: retained harness failure

Status: **failed before any coverage, parity, or timing result was produced**.

The frozen source commit `f3fa1303192dbb03c972984e38f3472e4e15adfa`
was launched directly on `gpu11` GPU 2.  The first rate-2 codebook contains
fewer than 16 codewords, while the development kernel required an exactly
16-entry candidate row.  The table builder did not pad that row, so its
fail-closed shape check raised `ValueError` and the launcher exited 1.

This is a diagnostic-harness defect, not a HayateQuant quality or speed result.
No `result.json` exists and this run must not be used in any aggregate.  The
empty controller stdout, complete traceback, exit status, command, clean
detached-source status, source hash, and GPU snapshots before and after the run
are retained here.  A corrected table builder must be frozen in a new commit
before any rerun.
