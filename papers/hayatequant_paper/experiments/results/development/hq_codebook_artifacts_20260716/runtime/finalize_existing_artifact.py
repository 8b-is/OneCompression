"""Metadata-only recovery for immutable artifacts written by failed job 69346."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform

import torch

from fit_one_artifact import (
    EVAL_SAMPLES_BY_BLOCK,
    EVAL_SEED,
    FIT_MAX_SAMPLES,
    FIT_MIN_SAMPLES,
    FIT_SCORE_BUDGET,
    FIT_SEED,
    ITERS,
    RESTARTS,
    ROOT,
    TARGETS,
    _mse,
    _sha256,
    _tensor_sha256,
)
from onecomp.kvcomp.codec.hayate.vq import (
    _gaussian_block_samples,
    lloyd_vq_gaussian_codebook,
)


FROZEN_FILE_SHA256 = {
    (2, 8): "3bf6f9d612eb610d94cb95eafeb1a08e1baca2383465ed5193ab0c6184832fbc",
    (2, 32): "91a05e6a2c38b911df954b25ce62950d2715c348d87e8224abcab54ec4c3c6e8",
    (2, 128): "e789c4b7b7966dc3ddf55e0d1ab76f71c3b3a09e698f06a32abb588bc6ba5d3d",
    (2, 512): "3591ff90806677bcc00c50ddd44a761f11d8b845dbea1a408ce7034d6ca2d2ed",
    (4, 64): "1d46776f034453ce3d80c9a4e0404322973130cd7a2561ae552a0194af9d01bd",
    (4, 1024): "88b4b969a5c29ca70fd194bb412de72624169e3a1b76ba91b7afce8000e23cb3",
}


def main() -> None:
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= task < len(TARGETS):
        raise RuntimeError(f"invalid target task {task}")
    block_size, k = TARGETS[task]
    code_bits = k.bit_length() - 1
    n_fit = min(FIT_MAX_SAMPLES, max(FIT_MIN_SAMPLES, FIT_SCORE_BUDGET // k))
    n_eval = EVAL_SAMPLES_BY_BLOCK[block_size]
    path = ROOT / "artifacts" / f"gauss_b{block_size}_k{k}.pt"
    meta_path = ROOT / "metadata" / f"gauss_b{block_size}_k{k}.json"
    if not path.is_file() or meta_path.exists():
        raise RuntimeError("metadata recovery requires an existing artifact and absent metadata")
    frozen_sha = FROZEN_FILE_SHA256[(block_size, k)]
    if _sha256(path) != frozen_sha:
        raise RuntimeError("artifact changed after post-failure SHA freeze; refusing recovery")
    artifact = torch.load(path, map_location="cpu", weights_only=True).contiguous()
    if (
        artifact.dtype != torch.float32
        or tuple(artifact.shape) != (k, block_size)
        or not torch.isfinite(artifact).all()
        or torch.unique(artifact, dim=0).shape[0] != k
    ):
        raise RuntimeError("preserved artifact tensor validation failed")

    if not torch.cuda.is_available():
        raise RuntimeError("metadata recovery requires CUDA for the frozen held-out audit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    # The explicit non-default arguments force the current online fallback;
    # neither the preserved staging file nor a future installed artifact can
    # be silently selected here.
    online = lloyd_vq_gaussian_codebook(
        code_bits,
        block_size,
        iters=25,
        n_samples=100_000,
        seed=0,
        restarts=4,
        init="auto",
        ambient_dim=128,
    ).contiguous()
    eval_samples = _gaussian_block_samples(block_size, n_eval, EVAL_SEED).to(device)
    artifact_mse = _mse(eval_samples, artifact)
    online_mse = _mse(eval_samples, online)
    source_gate = artifact_mse < online_mse
    metadata = {
        "schema": "hayatequant.gaussian_codebook_artifact_fit.v1",
        "target": {"block_size": block_size, "k": k, "code_bits": code_bits},
        "fit": {
            "source": "iid_standard_normal_fp32",
            "samples": n_fit,
            "max_iterations": ITERS,
            "restarts": RESTARTS,
            "seed": FIT_SEED,
            "tf32_allowed": False,
            "first_restart_init": "forgy_like_all_other_restarts",
            "fit_seconds": None,
            "fit_seconds_note": "lost only from in-memory metadata after post-save JSON failure",
            "original_failed_array_job": f"69346_{task}",
            "artifact_preserved_without_refit": True,
        },
        "heldout": {
            "source": "iid_standard_normal_fp32",
            "samples": n_eval,
            "seed": EVAL_SEED,
            "distance": "explicit_fp32_squared_distance_smallest_index_argmin",
            "artifact_mse_per_coordinate": artifact_mse,
            "online_fallback_mse_per_coordinate": online_mse,
            "artifact_minus_online_mse": artifact_mse - online_mse,
            "strict_improvement_gate": source_gate,
            "metadata_only_recomputed_after_serialization_failure": True,
        },
        "artifact": {
            "relative_path": str(path.relative_to(ROOT)),
            "shape": list(artifact.shape),
            "dtype": str(artifact.dtype),
            "tensor_sha256": _tensor_sha256(artifact),
            "file_sha256": frozen_sha,
            "bytes": path.stat().st_size,
            "mtime_ns_preserved": path.stat().st_mtime_ns,
        },
        "runtime": {
            "metadata_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata_hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "metadata_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "metadata_slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "metadata_slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "recovery": {
            "reason": "all 69346 fits saved tensors then failed serializing _CUuuid",
            "artifact_refit": False,
            "artifact_overwritten": False,
            "frozen_sha_manifest": "FAILED_FIT_ARTIFACT_SHA256.txt",
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    if not source_gate:
        raise RuntimeError("preserved artifact failed strict held-out source-MSE gate")


if __name__ == "__main__":
    main()
