# Commands

The candidate set and decision rule were frozen in `PREREGISTRATION.md` before
launch.  The development array is launched from the repository root with:

```bash
sbatch papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/runtime/submit_hq_quality_refinement.sh
```

After every array element completes successfully:

```bash
/lustre/home/frj/yuma-ichikawa/conference/onecomp-lab/.venv/bin/python \
  papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/runtime/analyze_hq_quality_refinement.py
```

The exact Hydra command executed by each array element is retained in the
Slurm script.  Runtime manifests record the worker, GPU, source hashes, and
exit status independently for every model/kind pair.

The separately preregistered rate-2 packed-native eligibility gate is launched
with:

```bash
sbatch papers/hayatequant_paper/experiments/results/development/hq_quality_refinement_20260716/runtime/submit_hq_rate2_native_gate.sh
```
