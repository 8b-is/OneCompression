# Exact LUT tensor-core development diagnostic v2

Status: **empirical parity passed; candidate rejected on assignment latency**.

The corrected, frozen source commit
`c53cd395e024add91888a56f68d0177fab12cc99` was launched from a clean detached
worktree on `gpu11` GPU 2 (NVIDIA B200).  This is a development diagnostic and
is not a formal benchmark or a promotion result.

Across all six fixed rate/role codebooks, Gaussian and wide-range fixtures
totalled 12,000,120 query rows.  The centre-ranked T=16 tables had zero missing
canonical winners and the row-specific padded tensor-core path had zero output
mismatches.  The timing fixtures also matched both the kernel-only exact path
and the exact path including the float32-to-float16 cast.

The LUT kernel nevertheless took about 1.69 ms for 2,097,152 rows in every
cell, while canonical exact assignment including its cast took about
0.10--0.52 ms.  Thus the LUT was 3.23--16.68 times slower across the six cells.
It also requires four unique 2 MiB tables (8 MiB total) for these codebooks.
The candidate is rejected and will not replace HQ; no exhaustive coverage
certification is warranted for a speed-negative implementation.

`result.json` is byte-identical to controller stdout, controller stderr is
empty, the launcher exited 0, and source status was clean before and after.
Commands, source hashes, GPU snapshots, and checksums are retained alongside
the raw measurements.
