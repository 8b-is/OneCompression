# Direct centre-winner LUT frontier (development only)

Status: **lookup speed positive; canonical assignment accuracy negative;
candidate rejected**.

The four resolutions 256, 512, 1024, and 2048 bins per dimension were frozen
in source commit `9cb9af1878b127165fa433763f3f5b18c904cdbd` before any result was
observed.  A clean detached source was run on `gpu11` GPU 2 (NVIDIA B200).
Each table returns the canonical fp16 all-codeword winner at the spatial-cell
centre without candidate rescoring.  This is intentionally an approximate
development diagnostic, not HQ and not a formal benchmark.

Direct lookup was fast: for 2,097,152 b=2 rows, canonical exact assignment
including its fp32-to-fp16 cast was 1.89--18.11 times slower across the frozen
rate/role cells.  However, no rate/role/resolution point had zero assignment
mismatches.  At the largest 2048-bin resolution, which costs 8 MiB per
codebook and 32 MiB for the four unique fixed-HQ codebooks, Gaussian-fixture
mismatch fractions were 0.46%, 0.94%, 1.87%, and 4.08% for K=8, 32, 128, and
512 respectively.  The maximum observed Gaussian mismatch was 28.15% at a
coarser point.  These errors are too large to treat lookup as an implementation
of canonical HQ, and lookup-only timing does not show that the complete HQ
store would beat TurboQuant.  No model-quality run is promoted from this
candidate.

The first launch was submitted as Slurm job 69532 while GPU11 was occupied.
When a clean direct GPU became available, the job was cancelled; it began at
the same moment as `scancel` and was terminated before producing a result.
That cancelled attempt's submission, scheduler state, stdout/stderr, and exit
records are retained.  The subsequent direct attempt exited 0, wrote
`result.json` byte-identically to controller stdout, had empty stderr, and
used a source worktree that was clean before and after.  Commands, source
hashes, GPU snapshots, raw mismatch examples, timings, and checksums are all
stored here.
