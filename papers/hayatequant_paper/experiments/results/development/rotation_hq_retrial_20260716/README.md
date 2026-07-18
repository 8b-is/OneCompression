# Rotation HQ retrial (development)

This directory records a new rotation-only HQ measurement.  It is not a
formal paper aggregate and must not populate manuscript values until the
complete Rotation HQ protocol has its own coverage and provenance gate.

- The calibration source is the pinned WikiText-2 raw **train** split; its
  exact selected text and manifest are in `calibration/`.
- Evaluation uses WikiText-2 test only, so calibration and PPL evaluation are
  split-disjoint.
- Rotation HQ uses the fixed HQ `b=2` K/V block-code budgets, fixed data-free
  codebooks, exact assignment, and one fp16 scale per vector.  The artifact
  may change only the orthogonal rotation; learned codebooks are rejected.
- Each model/rate run records its source commit, command, calibration digest,
  stdout/stderr, and checksums next to its result.
