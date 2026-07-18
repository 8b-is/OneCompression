# Main-paper figure and table refresh audit

This audit records the paper-layout refresh without copying any development or
partial benchmark value into the manuscript.

- `prep.tex` is read-only and remains SHA-256
  `319a229a341e5f90bb8751bee994b77d66ef0df910ac822bbf10914e397f8d0f`.
- `main.tex` now follows the compact `prep.tex` result flow: a short abstract,
  quality figure, seven-model PPL/zero-shot table, concise aggregate/gate
  table, speed paragraph, and Rotation-HQ paragraph.
- Only a complete coverage-checked aggregate can generate numerical content.
  The complete table ledger stays alongside the aggregate in the repository;
  it is not inflated into the paper appendix.
- The current formal-result schema contains HQ and Adaptive HQ.  Rotation HQ
  is defined as a separately pre-registered opt-in learned-rotation control
  and cannot inherit either method's values.

Checks recorded in `paper_layout_check.json` use a synthetic complete fixture
solely to test the aggregate-to-LaTeX interface; it is never copied to a paper
result directory.  The full renderer suite is recorded in
`renderer-junit.xml`; it passed all 86 tests.  Verify the tracked audit files
from this directory with `sha256sum -c SHA256SUMS`.
