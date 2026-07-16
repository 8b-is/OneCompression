"""Fit one preregistered high-budget Gaussian HQ codebook artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from onecomp.kvcomp.codec.hayate.vq import (
    _gaussian_block_samples,
    _kmeans_best_of,
    fibquant_initial_codebook_k,
    lloyd_vq_gaussian_codebook,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ((2, 8), (2, 32), (2, 128), (2, 512), (4, 64), (4, 1024))
FIT_MAX_SAMPLES = 1_600_000
FIT_SCORE_BUDGET = 400_000_000
FIT_MIN_SAMPLES = 200_000
ITERS = 60
RESTARTS = 8
FIT_SEED = 0
EVAL_SEED = 123
EVAL_SAMPLES_BY_BLOCK = {2: 1_000_000, 4: 500_000}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.inference_mode()
def _mse(samples: torch.Tensor, codebook: torch.Tensor, chunk: int = 32768) -> float:
    total = torch.zeros((), device=samples.device, dtype=torch.float64)
    count = 0
    cb = codebook.to(device=samples.device, dtype=torch.float32)
    csq = cb.square().sum(dim=1)
    for start in range(0, samples.shape[0], chunk):
        x = samples[start : start + chunk]
        # Explicit fp32 squared-distance arithmetic and smallest-index argmin.
        dist = x.square().sum(dim=1, keepdim=True) - 2.0 * (x @ cb.t()) + csq
        best = dist.argmin(dim=1)
        err = (x - cb[best]).square().sum(dim=1)
        total += err.double().sum()
        count += int(x.numel())
    return float((total / count).cpu())


def main() -> None:
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= task < len(TARGETS):
        raise RuntimeError(f"invalid target task {task}")
    block_size, k = TARGETS[task]
    code_bits = k.bit_length() - 1
    if 1 << code_bits != k:
        raise RuntimeError("all targets must be powers of two")
    n_fit = min(FIT_MAX_SAMPLES, max(FIT_MIN_SAMPLES, FIT_SCORE_BUDGET // k))
    n_eval = EVAL_SAMPLES_BY_BLOCK[block_size]
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("preregistered artifact fit requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    out_dir = ROOT / "artifacts"
    meta_dir = ROOT / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gauss_b{block_size}_k{k}.pt"
    meta_path = meta_dir / f"gauss_b{block_size}_k{k}.json"
    if path.exists() or meta_path.exists():
        raise RuntimeError(f"refusing to overwrite preregistered target {path}")

    started = time.perf_counter()
    samples = _gaussian_block_samples(block_size, n_fit, FIT_SEED).to(device)
    cb = _kmeans_best_of(samples, k, ITERS, FIT_SEED, RESTARTS)
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started
    cb = cb.detach().contiguous().cpu().to(torch.float32)
    if tuple(cb.shape) != (k, block_size) or not torch.isfinite(cb).all():
        raise RuntimeError("artifact tensor shape/finiteness gate failed")
    if torch.unique(cb, dim=0).shape[0] != k:
        raise RuntimeError("artifact contains duplicate codewords")

    # Reproduce the exact missing-artifact online fallback before installing
    # the new tensor. The target directory is intentionally not on the loader
    # path in this fitting job.
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
    artifact_mse = _mse(eval_samples, cb)
    online_mse = _mse(eval_samples, online)
    source_gate = artifact_mse < online_mse

    torch.save(cb, path)
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
            "fit_seconds": fit_seconds,
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
        },
        "artifact": {
            "relative_path": str(path.relative_to(ROOT)),
            "shape": list(cb.shape),
            "dtype": str(cb.dtype),
            "tensor_sha256": _tensor_sha256(cb),
            "file_sha256": _sha256(path),
            "bytes": path.stat().st_size,
        },
        "runtime": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            # PyTorch 2.11 exposes a private _CUuuid object; convert it to a
            # stable JSON string. This line is a post-failure mechanical fix.
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT.parents[5])): _sha256(
                Path(__file__).resolve()
            ),
            "onecomp/kvcomp/codec/hayate/vq.py": _sha256(
                ROOT.parents[5] / "onecomp/kvcomp/codec/hayate/vq.py"
            ),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    if not source_gate:
        raise RuntimeError("artifact failed strict held-out source-MSE improvement gate")


if __name__ == "__main__":
    main()
