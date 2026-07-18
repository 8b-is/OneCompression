"""Serialize the first output of the unchanged current HQ online fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from onecomp.kvcomp.codec.hayate import vq


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((2, 8), (2, 32), (2, 128), (2, 512))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def construct(block_size: int, k: int) -> torch.Tensor:
    if os.environ.get("KVCOMP_CODEBOOK_DIR") is not None:
        raise RuntimeError("reference constructor forbids KVCOMP_CODEBOOK_DIR")
    if vq._gaussian_codebook_artifact(block_size, k) is not None:
        raise RuntimeError("target already exists on a loader path; online fallback not isolated")
    code_bits = k.bit_length() - 1
    return vq.lloyd_vq_gaussian_codebook(
        code_bits,
        block_size,
        iters=25,
        n_samples=100_000,
        seed=0,
        restarts=4,
        init="auto",
        ambient_dim=128,
    ).detach().cpu().contiguous()


def main() -> None:
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    block_size, k = TARGETS[task]
    artifact_dir = ROOT / "artifacts"
    metadata_dir = ROOT / "metadata"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"gauss_b{block_size}_k{k}.pt"
    meta_path = metadata_dir / f"reference_b{block_size}_k{k}.json"
    if path.exists() or meta_path.exists():
        raise RuntimeError("designated first output already exists; refusing overwrite")
    started = time.perf_counter()
    tensor = construct(block_size, k)
    seconds = time.perf_counter() - started
    if (
        tensor.dtype != torch.float32
        or tuple(tensor.shape) != (k, block_size)
        or not torch.isfinite(tensor).all()
        or torch.unique(tensor, dim=0).shape[0] != k
    ):
        raise RuntimeError("online constructor tensor validation failed")
    torch.save(tensor, path)
    metadata = {
        "schema": "hayatequant.current_online_serialization.reference.v1",
        "target": {"block_size": block_size, "k": k, "code_bits": k.bit_length() - 1},
        "constructor": {
            "function": "lloyd_vq_gaussian_codebook",
            "source": "iid_standard_normal_fp32",
            "samples": 100000,
            "max_iterations": 25,
            "restarts": 4,
            "seed": 0,
            "init": "auto",
            "ambient_dim": 128,
            "elapsed_seconds": seconds,
            "quality_selection": False
        },
        "artifact": {
            "relative_path": str(path.relative_to(ROOT)),
            "shape": list(tensor.shape),
            "original_dtype": str(tensor.dtype),
            "finite": True,
            "unique_rows": k,
            "tensor_sha256": tensor_sha256(tensor),
            "file_sha256": sha256(path),
            "bytes": path.stat().st_size
        },
        "runtime": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")
        }
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
