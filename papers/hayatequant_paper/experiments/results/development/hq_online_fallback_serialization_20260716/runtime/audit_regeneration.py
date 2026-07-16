"""Fresh-process regeneration audit; never writes or selects a codebook."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import torch

from serialize_reference import ROOT, TARGETS, construct, tensor_sha256


def main() -> None:
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    block_size, k = TARGETS[task]
    reference_meta_path = ROOT / "metadata" / f"reference_b{block_size}_k{k}.json"
    output = ROOT / "audits" / f"regeneration_b{block_size}_k{k}.json"
    if not reference_meta_path.is_file() or output.exists():
        raise RuntimeError("regeneration requires one immutable reference and absent audit")
    expected = json.loads(reference_meta_path.read_text())
    regenerated = construct(block_size, k)
    observed_sha = tensor_sha256(regenerated)
    expected_sha = expected["artifact"]["tensor_sha256"]
    match = observed_sha == expected_sha
    document = {
        "schema": "hayatequant.current_online_serialization.regeneration_audit.v1",
        "target": {"block_size": block_size, "k": k},
        "fresh_process": True,
        "reference_tensor_sha256": expected_sha,
        "regenerated_tensor_sha256": observed_sha,
        "bit_exact": match,
        "regenerated_shape": list(regenerated.shape),
        "regenerated_dtype": str(regenerated.dtype),
        "regenerated_finite": bool(torch.isfinite(regenerated).all()),
        "regenerated_unique_rows": int(torch.unique(regenerated, dim=0).shape[0]),
        "runtime": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, indent=2, sort_keys=True), flush=True)
    if not match:
        raise RuntimeError("fresh-process fallback regeneration is not bit-exact")


if __name__ == "__main__":
    main()
