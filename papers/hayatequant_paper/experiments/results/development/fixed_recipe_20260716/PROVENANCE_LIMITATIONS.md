# Provenance limitations

This bundle is a development-only recipe-selection panel and is not admissible
formal paper evidence.  It records the launch HEAD, full raw result JSON,
resolved codec configurations, dataset revision/text digest, runtime scripts,
Slurm stdout/stderr, and repository-local content hashes.

The launch worktree was intentionally dirty.  The historical wrapper did not
save a byte-complete source diff, per-task environment manifest, model-weight
manifest, or immutable preflight record.  Those missing artifacts cannot be
reconstructed retrospectively and are not implied by the recorded HEAD.  The
bundle therefore supports only the documented development decision; its 21/21
short-window observation must not be substituted for, pooled with, or quoted
as the clean-commit formal run.

The refreshed formal launcher closes each limitation: it rejects a dirty tree,
pins and hashes every indexed model shard and dataset input, records software,
CUDA, Slurm, resolved role-wise codec configuration, and source commit for each
job, then requires complete coverage and one provenance stratum before results
can reach the paper renderer.
