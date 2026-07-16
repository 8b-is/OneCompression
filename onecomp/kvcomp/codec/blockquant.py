"""Independent Block-Sphere Quantization quality baseline.

This module implements Algorithm 1 of *Block-Sphere Vector Quantization*
(arXiv:2605.19972v1).  It is deliberately independent from HayateQuant: it
does not import HayateQuant codebooks, recipes, rotations, scale helpers, or
payload types, and its registry identity is ``blockquant``.

Paper-defined contract
----------------------
For every vector (or fixed channel subvector), BlockQuant stores one fp16
scalar, L2-normalizes the vector, applies a seeded dense Haar rotation, splits
the rotated unit vector into blocks, and assigns every block to the exact
nearest member of a data-free block codebook.  A ``p``-dimensional, ``b``-bit
per-coordinate block has exactly ``2**(b*p)`` centroids fitted to the marginal
of the first ``p`` coordinates of ``Unif(S^(d-1))``.  Decode performs codebook
lookup, the transpose rotation, and one of the paper's three scalar rules:

* ``raw``: store the input L2 norm and apply no directional correction;
* ``bsm`` (alias ``ls``): ``<x,c> / ||c||^2``, the best-scalar MSE rule;
* ``ub`` (alias ``ip``): ``||x||^2 / <x,c>``, the ratio/IP-unbiased rule.

Reproducibility completions
---------------------------
The v1 TeX defines the continuous codebook objective but does not publish
centroids, code, a fitting seed/sample count/restart count/stopping rule, or
the experimental LUT's grid/candidate sizes.  Results must therefore not
claim byte-for-byte reproduction of an unpublished implementation.  The only
accepted codebook spec, ``onecomp.blockquant.finite_beta_lloyd.v1``, makes the
missing numerical choices immutable and records them plus every codebook SHA
in the payload.  This is a deterministic, data-free numerical realization of
the paper objective.  Quality evaluation always uses full exact codebook
search; LUT modes are rejected rather than guessed.

Algorithm 1 also assumes ``p`` divides the ambient dimension, while the KV
recipe uses ``p=3`` with 32- and 128-dimensional subvectors.  The default is
fail-closed.  The explicit ``lower_dimensional_tail_v1`` completion encodes a
remainder of size ``r`` with its own exact ``2**(b*r)`` centroid codebook.
This preserves exactly ``b`` logical code bits per coordinate and is recorded
as a non-paper-specified extension in provenance.

KV recipe
---------
Both Llama-3.1 factories fix the paper's operating point: 32 key channels at
4 bits, the other 96 at 3 bits, two fp16 key scales (448 / 128 = 3.5 logical
bits per key coordinate), and all 128 value channels at 2 bits plus one fp16
scale (2.125 logical bits per value coordinate).  The source does not specify
how a decoder learns the 32 channel identities.  ``llama31_paper_recipe``
therefore requires an explicitly pinned, immutable global channel set and
does not charge identities per vector.  The canonical cross-model quality
row instead uses ``llama31_dynamic_top_l2_completion``: it recomputes the
top-32 set per batch/head/update and charges its fixed-width combinadic
identity field.  These are distinct, provenance-marked completions; neither
is silently substituted for the other or described as a source-table
reproduction.

The source also does not explicitly assign one of Algorithm 1's ``raw``,
``BSM`` or ``UB`` decoders to each KV role.  The versioned KV completion uses
the objective-consistent mapping UB for keys (attention inner products) and
BSM for values (reconstruction).  Together with the unpublished centroids,
LUT hyperparameters, p=3 tail treatment and channel-identity lifecycle, this
means the implementation is a method-level quality baseline, not a byte-exact
reproduction of the source experiments.

``stored_bits`` is the exact logical fixed-width data-plane count (block
indices and fp16 scalars).  Research tensors use int64 indices physically;
global rotations, codebooks, the fixed channel set, Python containers, and
allocator overhead are explicitly out of scope and never described as packed
serving memory.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from .._base import CompressedKV, KVCodec
from .._registry import register_kv_codec
from .rotation import random_orthogonal

BLOCKQUANT_REFERENCE = "arXiv:2605.19972v1"
BLOCKQUANT_SOURCE_TEX_SHA256 = "dc87211edfb3e83285bc19b636d1f6ebe8b054f8c5b69ae0d5c2894b7dac721f"
BLOCKQUANT_CODEBOOK_SPEC = "onecomp.blockquant.finite_beta_lloyd.v1"
BLOCKQUANT_REMAINDER_POLICY = "lower_dimensional_tail_v1"
BLOCKQUANT_DYNAMIC_OUTLIER_POLICY = "per_batch_head_update_top_l2_v1"
BLOCKQUANT_IDENTITY_CODING = "combinadic_rank_fixed_width_v1"

_SCALE_ALIASES = {"raw": "raw", "bsm": "bsm", "ls": "bsm", "ub": "ub", "ip": "ub"}
_PAPER_BLOCK_SIZES = (2, 3)
_CODEBOOK_BLOCK_DIMS = (1, 2, 3)
_PAPER_BITS = (1, 2, 3, 4)
_TARGET_SAMPLES_PER_CODEWORD = 512
_MIN_CODEBOOK_SAMPLES = 1 << 16
_MAX_CODEBOOK_SAMPLES = 1 << 20
_RESTARTS = 4
_LLOYD_ITERATIONS = 25
_MAX_CODEWORDS = 4096
_MAX_SCORE_ELEMENTS = 4_000_000
_MIN_SEED = -(1 << 63)
_MAX_SEED = (1 << 63) - 1
_ROLE_SEED_OFFSETS = {"key": 104_729, "key_outlier": 130_363, "value": 155_921}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_seed(value: int, label: str) -> None:
    maximum_offset = max(max(_ROLE_SEED_OFFSETS.values()), _RESTARTS * 1_000_003)
    if not _is_int(value) or not _MIN_SEED <= value <= _MAX_SEED - maximum_offset:
        raise ValueError(f"{label} must be a signed 64-bit integer, got {value!r}")


def _canonical_scale_mode(value: str, label: str) -> str:
    if not isinstance(value, str) or value not in _SCALE_ALIASES:
        choices = ", ".join(sorted(_SCALE_ALIASES))
        raise ValueError(f"{label} must be one of {{{choices}}}, got {value!r}")
    return _SCALE_ALIASES[value]


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported BlockQuant floating dtype {dtype}") from exc


def _dtype_from_name(value: str) -> torch.dtype:
    names = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return names[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid BlockQuant output dtype metadata {value!r}") from exc


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"onecomp.blockquant.float32.tensor.v1\0")
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_codebook_request(ambient_dim: int, block_dim: int, bits: int) -> None:
    if not _is_int(ambient_dim) or ambient_dim < 2:
        raise ValueError(f"ambient_dim must be an integer >= 2, got {ambient_dim!r}")
    if block_dim not in _CODEBOOK_BLOCK_DIMS:
        raise ValueError(
            "BlockQuant quality baseline supports primary p=2/3 and a p=1 remainder; "
            f"got {block_dim!r}"
        )
    if block_dim > ambient_dim:
        raise ValueError(f"block_dim={block_dim} exceeds ambient_dim={ambient_dim}")
    if bits not in _PAPER_BITS:
        raise ValueError(f"bits must be one of {_PAPER_BITS}, got {bits!r}")
    codewords = 1 << (bits * block_dim)
    if codewords > _MAX_CODEWORDS:
        raise ValueError(
            f"BlockQuant codebook has {codewords} centroids, exceeding the paper-quality "
            f"limit {_MAX_CODEWORDS}"
        )


def _sample_spherical_marginal(
    ambient_dim: int, block_dim: int, count: int, seed: int
) -> torch.Tensor:
    """Integrate the exact finite-sphere marginal with scrambled Sobol nodes.

    Lemma 1 gives ``r**2 ~ Beta(p/2, (d-p)/2)`` independently of a
    uniform direction on ``S^(p-1)``.  Applying those inverse transforms to a
    scrambled Sobol net yields deterministic low-discrepancy integration
    nodes with the target marginal, without allocating ``count * d`` Gaussian
    variates.  For ``p=1`` the equivalent symmetric Beta coordinate law is
    used directly.
    """
    import numpy as np
    from scipy.special import betaincinv
    from scipy.stats import qmc

    if not _is_int(count) or count <= 0 or count & (count - 1):
        raise ValueError("BlockQuant Sobol sample count must be a positive power of two")
    _validate_seed(seed, "spherical marginal seed")
    seed_bits = seed % (1 << 64)
    seed_sequence = np.random.SeedSequence(
        [seed_bits & 0xFFFFFFFF, (seed_bits >> 32) & 0xFFFFFFFF]
    )
    generator = np.random.default_rng(seed_sequence)
    sobol_dim = 1 if block_dim == 1 else block_dim
    unit = qmc.Sobol(d=sobol_dim, scramble=True, seed=generator).random_base2(
        int(math.log2(count))
    )
    unit = np.clip(
        unit,
        np.nextafter(np.float64(0.0), np.float64(1.0)),
        np.nextafter(np.float64(1.0), np.float64(0.0)),
    )

    if block_dim == 1:
        alpha = 0.5 * (ambient_dim - 1)
        samples = (2.0 * betaincinv(alpha, alpha, unit[:, 0]) - 1.0)[:, None]
    else:
        if ambient_dim == block_dim:
            radius = np.ones(count, dtype=np.float64)
        else:
            radius = np.sqrt(
                betaincinv(
                    0.5 * block_dim,
                    0.5 * (ambient_dim - block_dim),
                    unit[:, 0],
                )
            )
        if block_dim == 2:
            angle = 2.0 * math.pi * unit[:, 1]
            direction = np.stack((np.cos(angle), np.sin(angle)), axis=1)
        elif block_dim == 3:
            z = 1.0 - 2.0 * unit[:, 1]
            angle = 2.0 * math.pi * unit[:, 2]
            radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            direction = np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1)
        else:  # guarded by _validate_codebook_request
            raise AssertionError("unreachable BlockQuant block dimension")
        samples = radius[:, None] * direction
    if not np.isfinite(samples).all():
        raise RuntimeError("BlockQuant spherical-marginal transform produced non-finite nodes")
    return torch.from_numpy(np.ascontiguousarray(samples)).to(dtype=torch.float64)


def _codebook_sample_count(n_codewords: int) -> int:
    target = max(_MIN_CODEBOOK_SAMPLES, _TARGET_SAMPLES_PER_CODEWORD * n_codewords)
    target = min(target, _MAX_CODEBOOK_SAMPLES)
    return 1 << int(math.ceil(math.log2(target)))


def _initial_codebook(ambient_dim: int, block_dim: int, count: int) -> torch.Tensor:
    """Deterministic Beta-quantile radial / quasi-uniform angular initializer."""
    import numpy as np
    from scipy.special import betaincinv

    index = np.arange(count, dtype=np.float64)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    if block_dim == 1:
        alpha = 0.5 * (ambient_dim - 1)
        quantile = (index + 0.5) / float(count)
        direction = (2.0 * betaincinv(alpha, alpha, quantile) - 1.0)[:, None]
        radius = np.ones(count, dtype=np.float64)
    elif block_dim == 2:
        angle = 2.0 * math.pi * np.mod(index * golden, 1.0)
        direction = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    elif block_dim == 3:
        z = 1.0 - 2.0 * (index + 0.5) / float(count)
        angle = 2.0 * math.pi * np.mod(index * golden, 1.0)
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        direction = np.stack((radial * np.cos(angle), radial * np.sin(angle), z), axis=1)
    else:  # guarded by _validate_codebook_request
        raise AssertionError("unreachable BlockQuant block dimension")

    if block_dim != 1:
        if ambient_dim == block_dim:
            radius = np.ones(count, dtype=np.float64)
        else:
            quantile = (index + 0.5) / float(count)
            radius = np.sqrt(
                betaincinv(0.5 * block_dim, 0.5 * (ambient_dim - block_dim), quantile)
            )
    if not np.isfinite(radius).all() or not np.isfinite(direction).all():
        raise RuntimeError("BlockQuant initializer produced non-finite values")
    return torch.from_numpy(radius[:, None] * direction).to(dtype=torch.float64)


def _assign_exact(
    samples: torch.Tensor, codebook: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest-centroid assignment with deterministic lowest-index ties."""
    flat = samples.reshape(-1, samples.shape[-1])
    count = flat.shape[0]
    n_codewords = codebook.shape[0]
    chunk = max(1, _MAX_SCORE_ELEMENTS // n_codewords)
    assignment = torch.empty(count, dtype=torch.int64, device=flat.device)
    squared_error = torch.empty(count, dtype=flat.dtype, device=flat.device)
    codeword_norm = codebook.square().sum(dim=1)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        batch = flat[start:stop]
        scores = batch @ codebook.t() - 0.5 * codeword_norm.unsqueeze(0)
        indices = scores.argmax(dim=1)
        assignment[start:stop] = indices
        squared_error[start:stop] = (batch - codebook[indices]).square().sum(dim=1)
    return assignment.reshape(samples.shape[:-1]), squared_error.reshape(samples.shape[:-1])


def _lloyd_refine(samples: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Fixed Lloyd loop with exact low-dimensional nearest-neighbour queries."""
    import numpy as np
    from scipy.spatial import cKDTree

    samples_array = np.ascontiguousarray(samples.detach().cpu().numpy(), dtype=np.float64)
    codebook = np.ascontiguousarray(initial.detach().cpu().numpy(), dtype=np.float64)
    n_codewords, block_dim = codebook.shape
    for _ in range(_LLOYD_ITERATIONS):
        distance, assignment = cKDTree(codebook).query(samples_array, k=1, eps=0.0, p=2, workers=1)
        assignment = np.asarray(assignment, dtype=np.int64)
        squared_error = np.asarray(distance, dtype=np.float64) ** 2
        counts = np.bincount(assignment, minlength=n_codewords).astype(np.int64)
        sums = np.zeros((n_codewords, block_dim), dtype=np.float64)
        np.add.at(sums, assignment, samples_array)
        occupied = counts > 0
        updated = codebook.copy()
        updated[occupied] = sums[occupied] / counts[occupied, None]

        # The paper leaves empty-cell handling unspecified.  The v1 completion
        # deterministically splits the occupied cell with greatest total SSE,
        # taking its farthest still-unclaimed sample (all ties: lowest index).
        empty = np.flatnonzero(~occupied)
        if empty.size:
            distortion = np.bincount(assignment, weights=squared_error, minlength=n_codewords)
            available = counts.copy()
            claimed = np.zeros(samples_array.shape[0], dtype=bool)
            for empty_index in empty:
                donor_score = distortion.copy()
                donor_score[available <= 1] = -math.inf
                donor = int(np.argmax(donor_score))
                if not math.isfinite(donor_score[donor]):
                    raise RuntimeError("BlockQuant cannot repair an empty Lloyd cell")
                candidates = (assignment == donor) & ~claimed
                candidate_indices = np.flatnonzero(candidates)
                if not candidate_indices.size:
                    raise RuntimeError("BlockQuant empty-cell donor has no available sample")
                sample_index = int(candidate_indices[np.argmax(squared_error[candidate_indices])])
                updated[empty_index] = samples_array[sample_index]
                claimed[sample_index] = True
                available[donor] -= 1
                distortion[donor] -= squared_error[sample_index]
        codebook = updated

    distance, _ = cKDTree(codebook).query(samples_array, k=1, eps=0.0, p=2, workers=1)
    error = float(np.square(distance, dtype=np.float64).sum())
    return torch.from_numpy(np.ascontiguousarray(codebook)), error


@lru_cache(maxsize=32)
def _codebook_result_cpu(
    ambient_dim: int,
    block_dim: int,
    bits: int,
    codebook_seed: int,
    codebook_spec: str,
) -> tuple[torch.Tensor, float]:
    _validate_codebook_request(ambient_dim, block_dim, bits)
    _validate_seed(codebook_seed, "codebook_seed")
    if codebook_spec != BLOCKQUANT_CODEBOOK_SPEC:
        raise ValueError(
            f"unsupported BlockQuant codebook spec {codebook_spec!r}; "
            f"use {BLOCKQUANT_CODEBOOK_SPEC!r}"
        )
    n_codewords = 1 << (bits * block_dim)
    sample_count = _codebook_sample_count(n_codewords)
    samples = _sample_spherical_marginal(
        ambient_dim,
        block_dim,
        sample_count,
        codebook_seed,
    )
    initial = _initial_codebook(ambient_dim, block_dim, n_codewords)
    best = None
    best_sse = math.inf
    for restart in range(_RESTARTS):
        rotation_seed = codebook_seed + 1_000_003 * (restart + 1)
        rotated = initial @ random_orthogonal(block_dim, rotation_seed, dtype=torch.float64)
        candidate, sse = _lloyd_refine(samples, rotated)
        if sse < best_sse:
            best = candidate
            best_sse = sse
    if best is None or not math.isfinite(best_sse) or not bool(torch.isfinite(best).all()):
        raise RuntimeError("BlockQuant Lloyd fitting did not produce a finite codebook")
    if bool((best.norm(dim=1) > 1.0 + 1.0e-10).any()):
        raise RuntimeError("BlockQuant Lloyd centroids escaped the spherical marginal ball")
    return best.to(dtype=torch.float32).contiguous(), best_sse


def blockquant_codebook(
    ambient_dim: int,
    block_dim: int,
    bits: int,
    *,
    codebook_seed: int = 0,
    codebook_spec: str = BLOCKQUANT_CODEBOOK_SPEC,
) -> torch.Tensor:
    """Return a defensive CPU copy of the resolved data-free codebook."""
    codebook, _ = _codebook_result_cpu(ambient_dim, block_dim, bits, codebook_seed, codebook_spec)
    return codebook.clone()


@lru_cache(maxsize=128)
def _codebook_on(
    ambient_dim: int,
    block_dim: int,
    bits: int,
    codebook_seed: int,
    codebook_spec: str,
    device_str: str,
) -> torch.Tensor:
    codebook, _ = _codebook_result_cpu(ambient_dim, block_dim, bits, codebook_seed, codebook_spec)
    return codebook.to(device=torch.device(device_str))


def _codebook_record(
    ambient_dim: int,
    block_dim: int,
    bits: int,
    codebook_seed: int,
    codebook_spec: str,
) -> dict:
    codebook, sse = _codebook_result_cpu(
        ambient_dim, block_dim, bits, codebook_seed, codebook_spec
    )
    return {
        "ambient_dim": ambient_dim,
        "block_dim": block_dim,
        "bits_per_coordinate": bits,
        "code_bits": bits * block_dim,
        "codebook_size": 1 << (bits * block_dim),
        "codebook_seed": codebook_seed,
        "codebook_spec": codebook_spec,
        "dtype": "float32",
        "training_sample_count": _codebook_sample_count(1 << (bits * block_dim)),
        "target_samples_per_codeword": _TARGET_SAMPLES_PER_CODEWORD,
        "sha256": _tensor_sha256(codebook),
        "training_sse_float64": sse,
    }


@lru_cache(maxsize=64)
def _rotation_cpu(ambient_dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(ambient_dim, seed)


@lru_cache(maxsize=128)
def _rotation_on(ambient_dim: int, seed: int, device_str: str) -> torch.Tensor:
    return _rotation_cpu(ambient_dim, seed).to(device=torch.device(device_str))


def _rotation_record(ambient_dim: int, seed: int) -> dict:
    return {
        "kind": "dense-haar-gaussian-qr",
        "ambient_dim": ambient_dim,
        "seed": seed,
        "dtype": "float32",
        "sha256": _tensor_sha256(_rotation_cpu(ambient_dim, seed)),
    }


def _paper_provenance(codebook_spec: str, remainder_policy: str, outlier_selection: dict) -> dict:
    return {
        "reference": BLOCKQUANT_REFERENCE,
        "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
        "independent_baseline": True,
        "proxy_for_another_codec": False,
        "paper_algorithm": "Block-Sphere Quantization Algorithm 1",
        "assignment": {
            "mode": "exact-full-codebook-nearest-squared-l2",
            "lut": None,
            "reason": "quality baseline; paper does not specify experimental L or k",
        },
        "codebook": {
            "paper_objective": "finite-spherical-block-marginal squared-L2 Lloyd objective",
            "published_centroids_or_training_code": False,
            "reproducibility_completion": codebook_spec,
            "sampling": "scrambled-Sobol exact inverse finite-spherical marginal",
            "target_samples_per_codeword": _TARGET_SAMPLES_PER_CODEWORD,
            "minimum_training_samples": _MIN_CODEBOOK_SAMPLES,
            "maximum_training_samples": _MAX_CODEBOOK_SAMPLES,
            "restarts": _RESTARTS,
            "lloyd_iterations": _LLOYD_ITERATIONS,
            "initializer": "Beta-quantile radius plus quasi-uniform angle",
            "training_assignment": "exact scipy.spatial.cKDTree query, eps=0, workers=1",
            "empty_cell_rule": "greatest-total-SSE donor, farthest unclaimed sample",
        },
        "remainder": {
            "policy": remainder_policy,
            "paper_specified": remainder_policy == "error",
            "completion": (None if remainder_policy == "error" else BLOCKQUANT_REMAINDER_POLICY),
        },
        "kv_recipe_under_specification": {
            "objective_scale_mode_per_key_value_published": False,
            "completion": "key-ub-value-bsm-v1",
            "p3_tail_for_32_or_128_dimensions_published": False,
            "channel_identity_lifecycle_or_coding_published": False,
            "rotation_sharing_or_role_seed_derivation_published": False,
            "centroids_or_training_code_published": False,
            "lut_grid_or_candidate_hyperparameters_published": False,
        },
        "key_outlier_selection": outlier_selection,
    }


@register_kv_codec("blockquant")
@dataclass
class BlockQuantCodec(KVCodec):
    """Independent exact-search BlockQuant quality codec.

    ``key_outlier_indices`` is optional for a general uniform-rate codec.  If
    present, it is a fixed global channel set: keys use ``outlier_bits`` on
    that set and ``bits`` on its complement, storing one fp16 scale per
    subvector.  Values always use ``value_bits`` (or ``bits`` when omitted).
    """

    flag_calibration: bool = False
    bits: int = 3
    value_bits: int | None = None
    block_size: int = 3
    key_scale_mode: str = "ub"
    value_scale_mode: str = "bsm"
    key_outlier_indices: tuple[int, ...] | list[int] | None = None
    key_n_outlier: int | None = None
    outlier_bits: int | None = None
    remainder_policy: str = "error"
    assignment_mode: str = "exact"
    codebook_spec: str = BLOCKQUANT_CODEBOOK_SPEC
    rotation_seed: int = 0
    codebook_seed: int = 0
    paper_kv_recipe: bool = False
    key_axis: str = "token"
    value_axis: str = "token"

    def validate_params(self):
        if self.bits not in _PAPER_BITS:
            raise ValueError(f"bits must be one of {_PAPER_BITS}, got {self.bits!r}")
        if self.value_bits is not None and self.value_bits not in _PAPER_BITS:
            raise ValueError(
                f"value_bits must be None or one of {_PAPER_BITS}, got {self.value_bits!r}"
            )
        if self.block_size not in _PAPER_BLOCK_SIZES:
            raise ValueError(
                f"block_size must be one of paper sizes {_PAPER_BLOCK_SIZES}, "
                f"got {self.block_size!r}"
            )
        _canonical_scale_mode(self.key_scale_mode, "key_scale_mode")
        _canonical_scale_mode(self.value_scale_mode, "value_scale_mode")
        if self.remainder_policy not in ("error", BLOCKQUANT_REMAINDER_POLICY):
            raise ValueError(
                "remainder_policy must be 'error' or " f"{BLOCKQUANT_REMAINDER_POLICY!r}"
            )
        if self.assignment_mode != "exact":
            raise ValueError(
                "BlockQuant quality baseline permits assignment_mode='exact' only; "
                "the paper does not specify LUT grid/candidate hyperparameters"
            )
        if self.codebook_spec != BLOCKQUANT_CODEBOOK_SPEC:
            raise ValueError(
                f"codebook_spec must be {BLOCKQUANT_CODEBOOK_SPEC!r}; "
                "unversioned centroid fitting is rejected"
            )
        _validate_seed(self.rotation_seed, "rotation_seed")
        _validate_seed(self.codebook_seed, "codebook_seed")

        indices = self.key_outlier_indices
        if self.key_n_outlier is not None and (
            not _is_int(self.key_n_outlier) or self.key_n_outlier <= 0
        ):
            raise ValueError("key_n_outlier must be a positive integer or None")
        if indices is not None and self.key_n_outlier is not None:
            raise ValueError(
                "key_outlier_indices and key_n_outlier are mutually exclusive; "
                "use fixed identities or the versioned dynamic top-L2 completion"
            )
        if indices is not None:
            if not isinstance(indices, (tuple, list)) or not indices:
                raise ValueError("key_outlier_indices must be a non-empty tuple/list or None")
            if any(not _is_int(index) or index < 0 for index in indices):
                raise ValueError("key_outlier_indices must contain non-negative integers")
            canonical = tuple(sorted(indices))
            if len(canonical) != len(set(canonical)):
                raise ValueError("key_outlier_indices must be unique")
            self.key_outlier_indices = canonical
            if self.outlier_bits not in _PAPER_BITS:
                raise ValueError(
                    f"outlier_bits must be one of {_PAPER_BITS} with a fixed channel set"
                )
        elif self.key_n_outlier is not None:
            if self.outlier_bits not in _PAPER_BITS:
                raise ValueError(
                    f"outlier_bits must be one of {_PAPER_BITS} with dynamic top-L2 selection"
                )
        elif self.outlier_bits is not None:
            raise ValueError("outlier_bits requires explicit fixed key_outlier_indices")

        if self.paper_kv_recipe:
            if (
                self.bits != 3
                or self._value_bits() != 2
                or self.block_size != 3
                or self.outlier_bits != 4
                or (
                    (self.key_outlier_indices is None or len(self.key_outlier_indices) != 32)
                    and self.key_n_outlier != 32
                )
                or self.remainder_policy != BLOCKQUANT_REMAINDER_POLICY
                or _canonical_scale_mode(self.key_scale_mode, "key_scale_mode") != "ub"
                or _canonical_scale_mode(self.value_scale_mode, "value_scale_mode") != "bsm"
            ):
                raise ValueError(
                    "the versioned paper_kv_recipe completion requires p=3, "
                    "K 32ch@4b + 96ch@3b, V@2b, key UB/IP, value BSM/LS, "
                    "and an explicit lower-dimensional tail"
                )

    @classmethod
    def llama31_paper_recipe(
        cls,
        key_outlier_indices: tuple[int, ...] | list[int],
        *,
        rotation_seed: int = 0,
        codebook_seed: int = 0,
        **kwargs,
    ) -> "BlockQuantCodec":
        """Build the declared Llama-3.1 rate point with pinned completions.

        The caller must obtain the 32 fixed channel identities from its
        separately pinned protocol; this factory never invents a dynamic mask.
        Objective-specific K/V scale modes and the p=3 tail are not specified
        by the paper and are therefore versioned choices, not reproduction
        claims.
        """
        forbidden = {
            "bits",
            "value_bits",
            "block_size",
            "key_scale_mode",
            "value_scale_mode",
            "outlier_bits",
            "remainder_policy",
            "paper_kv_recipe",
        }.intersection(kwargs)
        if forbidden:
            raise ValueError(f"paper recipe fields cannot be overridden: {sorted(forbidden)}")
        return cls(
            bits=3,
            value_bits=2,
            block_size=3,
            key_scale_mode="ub",
            value_scale_mode="bsm",
            key_outlier_indices=key_outlier_indices,
            outlier_bits=4,
            remainder_policy=BLOCKQUANT_REMAINDER_POLICY,
            assignment_mode="exact",
            codebook_spec=BLOCKQUANT_CODEBOOK_SPEC,
            rotation_seed=rotation_seed,
            codebook_seed=codebook_seed,
            paper_kv_recipe=True,
            **kwargs,
        )

    @classmethod
    def llama31_dynamic_top_l2_completion(
        cls,
        *,
        rotation_seed: int = 0,
        codebook_seed: int = 0,
        **kwargs,
    ) -> "BlockQuantCodec":
        """Build the paper KV point with an explicit top-32 lifecycle completion.

        The paper says to select the 32 largest-L2 key channels per attention
        head but does not define when that set is recomputed or how it reaches
        the decoder.  This versioned completion recomputes it independently
        for each ``(batch, head, encode update)`` over that update's token axis.
        The sorted identities are logically stored once per batch/head/update
        as a fixed-width combinadic rank; unlike the paper's conditional 3.5
        bpc arithmetic, these identity bits are included in ``stored_bits``.
        """
        forbidden = {
            "bits",
            "value_bits",
            "block_size",
            "key_scale_mode",
            "value_scale_mode",
            "key_outlier_indices",
            "key_n_outlier",
            "outlier_bits",
            "remainder_policy",
            "paper_kv_recipe",
        }.intersection(kwargs)
        if forbidden:
            raise ValueError(f"paper completion fields cannot be overridden: {sorted(forbidden)}")
        return cls(
            bits=3,
            value_bits=2,
            block_size=3,
            key_scale_mode="ub",
            value_scale_mode="bsm",
            key_n_outlier=32,
            outlier_bits=4,
            remainder_policy=BLOCKQUANT_REMAINDER_POLICY,
            assignment_mode="exact",
            codebook_spec=BLOCKQUANT_CODEBOOK_SPEC,
            rotation_seed=rotation_seed,
            codebook_seed=codebook_seed,
            paper_kv_recipe=True,
            **kwargs,
        )

    def _value_bits(self) -> int:
        return self.bits if self.value_bits is None else self.value_bits

    def _role_seed(self, role: str) -> int:
        return self.rotation_seed + _ROLE_SEED_OFFSETS[role]

    def _validate_pair(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("BlockQuant key and value must be torch tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("BlockQuant expects K/V layout (batch, heads, sequence, head_dim)")
        if key.shape != value.shape:
            raise ValueError(
                f"BlockQuant key/value shapes must match, got {tuple(key.shape)} and "
                f"{tuple(value.shape)}"
            )
        if key.device != value.device:
            raise ValueError("BlockQuant key and value must share a device")
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError("BlockQuant key and value must have floating-point dtypes")
        if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
            raise ValueError("BlockQuant rejects non-finite K/V inputs")
        head_dim = key.shape[-1]
        if head_dim < 2:
            raise ValueError("BlockQuant head_dim must be >= 2")
        if self.key_outlier_indices is not None:
            if self.key_outlier_indices[-1] >= head_dim:
                raise ValueError("key_outlier_indices contains a channel outside head_dim")
            if len(self.key_outlier_indices) >= head_dim:
                raise ValueError("fixed outlier set must leave at least one ordinary key channel")
        if self.key_n_outlier is not None and self.key_n_outlier >= head_dim:
            raise ValueError("dynamic key_n_outlier must leave at least one ordinary key channel")
        if self.paper_kv_recipe and head_dim != 128:
            raise ValueError("the paper Llama-3.1 BlockQuant recipe requires head_dim=128")

    def _block_layout(self, ambient_dim: int, bits: int) -> tuple[int, int]:
        full_blocks, remainder = divmod(ambient_dim, self.block_size)
        if remainder and self.remainder_policy == "error":
            raise ValueError(
                f"BlockQuant Algorithm 1 requires block_size={self.block_size} to divide "
                f"ambient_dim={ambient_dim}; select {BLOCKQUANT_REMAINDER_POLICY!r} "
                "explicitly to use the provenance-marked remainder completion"
            )
        if full_blocks:
            _validate_codebook_request(ambient_dim, self.block_size, bits)
        if remainder:
            _validate_codebook_request(ambient_dim, remainder, bits)
        return full_blocks, remainder

    @staticmethod
    def _stored_scale(source: torch.Tensor, chat: torch.Tensor, mode: str) -> torch.Tensor:
        canonical = _SCALE_ALIASES[mode]
        norm_sq = source.square().sum(dim=-1, keepdim=True)
        if canonical == "raw":
            scale = norm_sq.sqrt()
        elif canonical == "bsm":
            denominator = chat.square().sum(dim=-1, keepdim=True)
            if bool((denominator <= 0.0).any()):
                raise RuntimeError("BlockQuant BSM encountered a zero reconstructed direction")
            scale = (source * chat).sum(dim=-1, keepdim=True) / denominator
        else:
            numerator = (source * chat).sum(dim=-1, keepdim=True)
            if bool((norm_sq <= 0.0).any()):
                raise ValueError("BlockQuant UB/IP is undefined for a zero input vector")
            if bool((numerator.abs() <= 1.0e-12).any()):
                raise RuntimeError("BlockQuant UB/IP alignment is zero or numerically singular")
            scale = norm_sq / numerator
        stored = scale.to(dtype=torch.float16)
        if not bool(torch.isfinite(stored).all()):
            raise ValueError("BlockQuant scale is not representable by its fp16 data-plane field")
        return stored

    def _encode_part(self, tensor: torch.Tensor, *, bits: int, scale_mode: str, role: str) -> dict:
        ambient_dim = tensor.shape[-1]
        full_blocks, remainder = self._block_layout(ambient_dim, bits)
        rotation_seed = self._role_seed(role)
        rotation = _rotation_on(ambient_dim, rotation_seed, str(tensor.device))
        source = tensor.float()
        norm = source.norm(dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
        rotated_unit = (source / safe_norm) @ rotation

        full_codes = None
        full_record = None
        pieces = []
        if full_blocks:
            stop = full_blocks * self.block_size
            blocks = rotated_unit[..., :stop].reshape(
                *tensor.shape[:-1], full_blocks, self.block_size
            )
            codebook = _codebook_on(
                ambient_dim,
                self.block_size,
                bits,
                self.codebook_seed,
                self.codebook_spec,
                str(tensor.device),
            )
            full_codes, _ = _assign_exact(blocks, codebook)
            pieces.append(codebook[full_codes].reshape(*tensor.shape[:-1], stop))
            full_record = _codebook_record(
                ambient_dim,
                self.block_size,
                bits,
                self.codebook_seed,
                self.codebook_spec,
            )

        tail_code = None
        tail_record = None
        if remainder:
            tail = rotated_unit[..., -remainder:]
            tail_codebook = _codebook_on(
                ambient_dim,
                remainder,
                bits,
                self.codebook_seed,
                self.codebook_spec,
                str(tensor.device),
            )
            tail_code, _ = _assign_exact(tail.unsqueeze(-2), tail_codebook)
            tail_code = tail_code.squeeze(-1)
            pieces.append(tail_codebook[tail_code])
            tail_record = _codebook_record(
                ambient_dim,
                remainder,
                bits,
                self.codebook_seed,
                self.codebook_spec,
            )
        chat_unit = torch.cat(pieces, dim=-1)
        # Scale rules are defined against the unnormalised rotated input and
        # the unit-source codeword concatenation; one scalar therefore carries
        # both the original norm and the objective-dependent correction.
        scale = self._stored_scale(source @ rotation, chat_unit, scale_mode)
        return {
            "schema_version": 1,
            "ambient_dim": ambient_dim,
            "bits_per_coordinate": bits,
            "block_size": self.block_size,
            "full_blocks": full_blocks,
            "remainder": remainder,
            "remainder_policy": self.remainder_policy,
            "scale_mode_requested": scale_mode,
            "scale_mode": _SCALE_ALIASES[scale_mode],
            "scale": scale,
            "full_codes": full_codes,
            "tail_code": tail_code,
            "full_codebook": full_record,
            "tail_codebook": tail_record,
            "rotation": _rotation_record(ambient_dim, rotation_seed),
        }

    def _encode_role(self, tensor: torch.Tensor, *, role: str) -> dict:
        shape = tuple(tensor.shape)
        if role == "key" and self.key_n_outlier is not None:
            energy = tensor.float().square().sum(dim=-2)
            indices = torch.argsort(energy, dim=-1, descending=True, stable=True)[
                ..., : self.key_n_outlier
            ]
            indices = torch.sort(indices, dim=-1).values
            mask = torch.zeros(
                shape[0], shape[1], shape[-1], dtype=torch.bool, device=tensor.device
            )
            mask.scatter_(-1, indices, True)
            all_indices = torch.arange(shape[-1], device=tensor.device).expand(
                shape[0], shape[1], shape[-1]
            )
            ordinary_indices = all_indices[~mask].reshape(
                shape[0], shape[1], shape[-1] - self.key_n_outlier
            )
            outlier_gather = indices.unsqueeze(-2).expand(
                shape[0], shape[1], shape[2], self.key_n_outlier
            )
            ordinary_gather = ordinary_indices.unsqueeze(-2).expand(
                shape[0], shape[1], shape[2], shape[-1] - self.key_n_outlier
            )
            outlier = self._encode_part(
                tensor.gather(-1, outlier_gather),
                bits=int(self.outlier_bits),
                scale_mode=self.key_scale_mode,
                role="key_outlier",
            )
            ordinary = self._encode_part(
                tensor.gather(-1, ordinary_gather),
                bits=self.bits,
                scale_mode=self.key_scale_mode,
                role="key",
            )
            body = {
                "mode": "dynamic-per-batch-head-update-top-l2-split",
                "selection_policy": BLOCKQUANT_DYNAMIC_OUTLIER_POLICY,
                "identity_coding": BLOCKQUANT_IDENTITY_CODING,
                "outlier_indices": indices,
                "outlier": outlier,
                "ordinary": ordinary,
            }
        elif role == "key" and self.key_outlier_indices is not None:
            mask = torch.zeros(shape[-1], dtype=torch.bool, device=tensor.device)
            mask[list(self.key_outlier_indices)] = True
            outlier = self._encode_part(
                tensor[..., mask],
                bits=int(self.outlier_bits),
                scale_mode=self.key_scale_mode,
                role="key_outlier",
            )
            ordinary = self._encode_part(
                tensor[..., ~mask],
                bits=self.bits,
                scale_mode=self.key_scale_mode,
                role="key",
            )
            body = {
                "mode": "fixed-channel-split",
                "fixed_outlier_indices": self.key_outlier_indices,
                "outlier": outlier,
                "ordinary": ordinary,
            }
        else:
            body = {
                "mode": "uniform",
                "part": self._encode_part(
                    tensor,
                    bits=self.bits if role == "key" else self._value_bits(),
                    scale_mode=(self.key_scale_mode if role == "key" else self.value_scale_mode),
                    role=role,
                ),
            }
        return {
            "codec": "blockquant",
            "schema_version": 1,
            "role": role,
            "shape": shape,
            "dtype": _dtype_name(tensor.dtype),
            "body": body,
        }

    def logical_bits_for_shape(self, shape: tuple[int, int, int, int]) -> dict:
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(not _is_int(size) or size <= 0 for size in shape)
            or shape[-1] < 2
        ):
            raise ValueError("shape must be a positive four-integer K/V shape")
        head_dim = shape[-1]
        vectors = math.prod(shape[:-1])
        if self.paper_kv_recipe and head_dim != 128:
            raise ValueError("the paper Llama-3.1 BlockQuant recipe requires head_dim=128")
        if self.key_outlier_indices is None and self.key_n_outlier is None:
            self._block_layout(head_dim, self.bits)
        else:
            outliers = (
                len(self.key_outlier_indices)
                if self.key_outlier_indices is not None
                else int(self.key_n_outlier)
            )
            self._block_layout(outliers, int(self.outlier_bits))
            self._block_layout(head_dim - outliers, self.bits)
        self._block_layout(head_dim, self._value_bits())
        key_identity_bits = 0
        key_identity_fields = 0
        if self.key_outlier_indices is None and self.key_n_outlier is None:
            key_code_bits = vectors * head_dim * self.bits
            key_scales = vectors * 16
            key_parts = 1
        else:
            if self.key_outlier_indices is not None and self.key_outlier_indices[-1] >= head_dim:
                raise ValueError("key_outlier_indices contains a channel outside head_dim")
            outliers = (
                len(self.key_outlier_indices)
                if self.key_outlier_indices is not None
                else int(self.key_n_outlier)
            )
            if outliers >= head_dim:
                raise ValueError("key outlier split must leave ordinary channels")
            key_code_bits = vectors * (
                outliers * int(self.outlier_bits) + (head_dim - outliers) * self.bits
            )
            key_scales = vectors * 2 * 16
            key_parts = 2
            if self.key_n_outlier is not None:
                key_identity_fields = shape[0] * shape[1]
                bits_per_identity = (math.comb(head_dim, outliers) - 1).bit_length()
                key_identity_bits = key_identity_fields * bits_per_identity
        key_payload = key_code_bits + key_identity_bits
        value_payload = vectors * head_dim * self._value_bits()
        value_scales = vectors * 16
        key_total = key_payload + key_scales
        value_total = value_payload + value_scales
        return {
            "scope": "fixed-width-data-plane-growth-only",
            "physical_packing_claim": False,
            "vectors_per_role": vectors,
            "head_dim": head_dim,
            "key": {
                "code_bits": key_code_bits,
                "payload_bits": key_payload,
                "fp16_scale_bits": key_scales,
                "scale_fields_per_vector": key_parts,
                "fixed_channel_identity_bits": 0,
                "dynamic_channel_identity_bits": key_identity_bits,
                "dynamic_channel_identity_fields": key_identity_fields,
                "dynamic_channel_identity_coding": (
                    BLOCKQUANT_IDENTITY_CODING if self.key_n_outlier is not None else None
                ),
                "total_bits": key_total,
                "bits_per_coordinate": key_total / float(vectors * head_dim),
            },
            "value": {
                "payload_bits": value_payload,
                "fp16_scale_bits": value_scales,
                "scale_fields_per_vector": 1,
                "total_bits": value_total,
                "bits_per_coordinate": value_total / float(vectors * head_dim),
            },
            "total_bits": key_total + value_total,
            "excluded_global_state": [
                "dense Haar rotations",
                "data-free codebooks",
                *(
                    ["fixed key outlier channel identities"]
                    if self.key_outlier_indices is not None
                    else []
                ),
                "Python control/container bytes",
                "allocator and unpacked research tensor bytes",
            ],
        }

    def encode(self, key, value, ctx=None) -> CompressedKV:
        _ = ctx
        self._validate_pair(key, value)
        key_payload = self._encode_role(key, role="key")
        value_payload = self._encode_role(value, role="value")
        accounting = self.logical_bits_for_shape(tuple(key.shape))
        resolved = {
            "schema_version": 1,
            "codec": "blockquant",
            "paper_kv_recipe": self.paper_kv_recipe,
            "config": self.get_kv_config(),
            "provenance": _paper_provenance(
                self.codebook_spec,
                self.remainder_policy,
                self._outlier_selection_provenance(),
            ),
            "logical_bits": accounting,
        }
        return CompressedKV(
            key_q=key_payload,
            value_q=value_payload,
            meta={
                "codec": "blockquant",
                "schema_version": 1,
                "stored_bits": accounting["total_bits"],
                "stored_payload_bits": (
                    accounting["key"]["payload_bits"] + accounting["value"]["payload_bits"]
                ),
                "stored_scale_bits": (
                    accounting["key"]["fp16_scale_bits"] + accounting["value"]["fp16_scale_bits"]
                ),
                "resolved_codec_config": resolved,
            },
        )

    def _validate_part(self, part: dict, *, expected_bits: int, expected_mode: str) -> None:
        required = {
            "schema_version",
            "ambient_dim",
            "bits_per_coordinate",
            "block_size",
            "full_blocks",
            "remainder",
            "remainder_policy",
            "scale_mode_requested",
            "scale_mode",
            "scale",
            "full_codes",
            "tail_code",
            "full_codebook",
            "tail_codebook",
            "rotation",
        }
        if not isinstance(part, dict) or set(part) != required or part["schema_version"] != 1:
            raise ValueError("malformed BlockQuant part payload")
        ambient_dim = part["ambient_dim"]
        if part["bits_per_coordinate"] != expected_bits or part["block_size"] != self.block_size:
            raise ValueError("BlockQuant payload rate/block does not match decoder config")
        if part["remainder_policy"] != self.remainder_policy:
            raise ValueError("BlockQuant payload remainder policy does not match decoder config")
        full_blocks, remainder = self._block_layout(ambient_dim, expected_bits)
        if part["full_blocks"] != full_blocks or part["remainder"] != remainder:
            raise ValueError("BlockQuant payload block layout is inconsistent")
        if (
            part["scale_mode"] != _SCALE_ALIASES[expected_mode]
            or part["scale_mode_requested"] != expected_mode
        ):
            raise ValueError("BlockQuant payload scale mode does not match decoder config")

    def _decode_part(
        self,
        part: dict,
        *,
        expected_bits: int,
        expected_mode: str,
        expected_lead_shape: tuple[int, ...],
        role: str,
    ) -> torch.Tensor:
        self._validate_part(part, expected_bits=expected_bits, expected_mode=expected_mode)
        ambient_dim = part["ambient_dim"]
        full_blocks = part["full_blocks"]
        remainder = part["remainder"]
        scale = part["scale"]
        if (
            not isinstance(scale, torch.Tensor)
            or tuple(scale.shape) != expected_lead_shape + (1,)
            or scale.dtype != torch.float16
            or not bool(torch.isfinite(scale).all())
        ):
            raise ValueError("invalid BlockQuant fp16 scale field")
        pieces = []
        codes_device = scale.device
        if full_blocks:
            codes = part["full_codes"]
            expected_shape = expected_lead_shape + (full_blocks,)
            if (
                not isinstance(codes, torch.Tensor)
                or tuple(codes.shape) != expected_shape
                or codes.dtype != torch.int64
                or codes.device != scale.device
            ):
                raise ValueError("invalid BlockQuant full-block code tensor")
            size = 1 << (expected_bits * self.block_size)
            if codes.numel() and (bool((codes < 0).any()) or bool((codes >= size).any())):
                raise ValueError("BlockQuant full-block code is outside its codebook")
            record = _codebook_record(
                ambient_dim,
                self.block_size,
                expected_bits,
                self.codebook_seed,
                self.codebook_spec,
            )
            if part["full_codebook"] != record:
                raise ValueError("BlockQuant full codebook provenance/hash mismatch")
            codebook = _codebook_on(
                ambient_dim,
                self.block_size,
                expected_bits,
                self.codebook_seed,
                self.codebook_spec,
                str(codes_device),
            )
            pieces.append(codebook[codes].reshape(*expected_lead_shape, -1))
        elif part["full_codes"] is not None or part["full_codebook"] is not None:
            raise ValueError("BlockQuant payload has unexpected full-block fields")

        if remainder:
            code = part["tail_code"]
            if (
                not isinstance(code, torch.Tensor)
                or tuple(code.shape) != expected_lead_shape
                or code.dtype != torch.int64
                or code.device != scale.device
            ):
                raise ValueError("invalid BlockQuant remainder code tensor")
            size = 1 << (expected_bits * remainder)
            if code.numel() and (bool((code < 0).any()) or bool((code >= size).any())):
                raise ValueError("BlockQuant remainder code is outside its codebook")
            record = _codebook_record(
                ambient_dim,
                remainder,
                expected_bits,
                self.codebook_seed,
                self.codebook_spec,
            )
            if part["tail_codebook"] != record:
                raise ValueError("BlockQuant remainder codebook provenance/hash mismatch")
            codebook = _codebook_on(
                ambient_dim,
                remainder,
                expected_bits,
                self.codebook_seed,
                self.codebook_spec,
                str(codes_device),
            )
            pieces.append(codebook[code])
        elif part["tail_code"] is not None or part["tail_codebook"] is not None:
            raise ValueError("BlockQuant payload has unexpected remainder fields")

        rotation_seed = self._role_seed(role)
        expected_rotation = _rotation_record(ambient_dim, rotation_seed)
        if part["rotation"] != expected_rotation:
            raise ValueError("BlockQuant payload rotation provenance/hash mismatch")
        rotation = _rotation_on(ambient_dim, rotation_seed, str(codes_device))
        rotated_reconstruction = torch.cat(pieces, dim=-1)
        return (rotated_reconstruction @ rotation.t()) * scale.to(dtype=torch.float32)

    def _decode_role(self, payload: dict, *, role: str) -> torch.Tensor:
        required = {"codec", "schema_version", "role", "shape", "dtype", "body"}
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or payload["codec"] != "blockquant"
            or payload["schema_version"] != 1
            or payload["role"] != role
        ):
            raise ValueError("malformed BlockQuant role payload")
        shape = payload["shape"]
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(not _is_int(size) or size <= 0 for size in shape)
            or shape[-1] < 2
        ):
            raise ValueError("invalid BlockQuant payload shape")
        lead = shape[:-1]
        body = payload["body"]
        if not isinstance(body, dict):
            raise ValueError("malformed BlockQuant role body")
        if role == "key" and self.key_n_outlier is not None:
            required_body = {
                "mode",
                "selection_policy",
                "identity_coding",
                "outlier_indices",
                "outlier",
                "ordinary",
            }
            if set(body) != required_body:
                raise ValueError("malformed BlockQuant dynamic-channel key body")
            indices = body["outlier_indices"]
            if (
                body["mode"] != "dynamic-per-batch-head-update-top-l2-split"
                or body["selection_policy"] != BLOCKQUANT_DYNAMIC_OUTLIER_POLICY
                or body["identity_coding"] != BLOCKQUANT_IDENTITY_CODING
                or not isinstance(indices, torch.Tensor)
                or tuple(indices.shape) != (shape[0], shape[1], self.key_n_outlier)
                or indices.dtype != torch.int64
            ):
                raise ValueError("invalid BlockQuant dynamic channel identities")
            if indices.numel() and (
                bool((indices < 0).any())
                or bool((indices >= shape[-1]).any())
                or bool((indices[..., 1:] <= indices[..., :-1]).any())
            ):
                raise ValueError("BlockQuant dynamic channel identities are invalid or repeated")
            outlier = self._decode_part(
                body["outlier"],
                expected_bits=int(self.outlier_bits),
                expected_mode=self.key_scale_mode,
                expected_lead_shape=lead,
                role="key_outlier",
            )
            ordinary = self._decode_part(
                body["ordinary"],
                expected_bits=self.bits,
                expected_mode=self.key_scale_mode,
                expected_lead_shape=lead,
                role="key",
            )
            if indices.device != outlier.device or ordinary.device != outlier.device:
                raise ValueError("BlockQuant dynamic identities and payload devices differ")
            mask = torch.zeros(
                shape[0], shape[1], shape[-1], dtype=torch.bool, device=outlier.device
            )
            mask.scatter_(-1, indices, True)
            all_indices = torch.arange(shape[-1], device=outlier.device).expand(
                shape[0], shape[1], shape[-1]
            )
            ordinary_indices = all_indices[~mask].reshape(
                shape[0], shape[1], shape[-1] - self.key_n_outlier
            )
            output = torch.empty(shape, dtype=torch.float32, device=outlier.device)
            output.scatter_(
                -1,
                indices.unsqueeze(-2).expand(shape[0], shape[1], shape[2], self.key_n_outlier),
                outlier,
            )
            output.scatter_(
                -1,
                ordinary_indices.unsqueeze(-2).expand(
                    shape[0], shape[1], shape[2], shape[-1] - self.key_n_outlier
                ),
                ordinary,
            )
        elif role == "key" and self.key_outlier_indices is not None:
            if set(body) != {"mode", "fixed_outlier_indices", "outlier", "ordinary"}:
                raise ValueError("malformed BlockQuant fixed-channel key body")
            if (
                body["mode"] != "fixed-channel-split"
                or body["fixed_outlier_indices"] != self.key_outlier_indices
            ):
                raise ValueError("BlockQuant fixed channel set does not match decoder config")
            outlier = self._decode_part(
                body["outlier"],
                expected_bits=int(self.outlier_bits),
                expected_mode=self.key_scale_mode,
                expected_lead_shape=lead,
                role="key_outlier",
            )
            ordinary = self._decode_part(
                body["ordinary"],
                expected_bits=self.bits,
                expected_mode=self.key_scale_mode,
                expected_lead_shape=lead,
                role="key",
            )
            output = torch.empty(shape, dtype=torch.float32, device=outlier.device)
            mask = torch.zeros(shape[-1], dtype=torch.bool, device=outlier.device)
            mask[list(self.key_outlier_indices)] = True
            output[..., mask] = outlier
            output[..., ~mask] = ordinary
        else:
            if set(body) != {"mode", "part"} or body["mode"] != "uniform":
                raise ValueError("malformed BlockQuant uniform role body")
            output = self._decode_part(
                body["part"],
                expected_bits=self.bits if role == "key" else self._value_bits(),
                expected_mode=self.key_scale_mode if role == "key" else self.value_scale_mode,
                expected_lead_shape=lead,
                role=role,
            )
        if tuple(output.shape) != shape:
            raise ValueError("BlockQuant decoded shape differs from payload shape")
        return output.to(dtype=_dtype_from_name(payload["dtype"]))

    def decode(self, compressed: CompressedKV, ctx=None):
        _ = ctx
        if not isinstance(compressed, CompressedKV):
            raise TypeError("BlockQuant decode expects a CompressedKV")
        expected_meta_keys = {
            "codec",
            "schema_version",
            "stored_bits",
            "stored_payload_bits",
            "stored_scale_bits",
            "resolved_codec_config",
        }
        if not isinstance(compressed.meta, dict) or set(compressed.meta) != expected_meta_keys:
            raise ValueError("malformed BlockQuant shared metadata")
        if (
            compressed.meta.get("codec") != "blockquant"
            or compressed.meta.get("schema_version") != 1
        ):
            raise ValueError("compressed payload was not produced by BlockQuant")
        key = self._decode_role(compressed.key_q, role="key")
        value = self._decode_role(compressed.value_q, role="value")
        if key.shape != value.shape or key.device != value.device:
            raise ValueError("BlockQuant decoded K/V shape or device mismatch")
        accounting = self.logical_bits_for_shape(tuple(key.shape))
        expected_resolved = {
            "schema_version": 1,
            "codec": "blockquant",
            "paper_kv_recipe": self.paper_kv_recipe,
            "config": self.get_kv_config(),
            "provenance": _paper_provenance(
                self.codebook_spec,
                self.remainder_policy,
                self._outlier_selection_provenance(),
            ),
            "logical_bits": accounting,
        }
        if (
            compressed.meta.get("stored_bits") != accounting["total_bits"]
            or compressed.meta.get("stored_payload_bits")
            != accounting["key"]["payload_bits"] + accounting["value"]["payload_bits"]
            or compressed.meta.get("stored_scale_bits")
            != accounting["key"]["fp16_scale_bits"] + accounting["value"]["fp16_scale_bits"]
            or compressed.meta.get("resolved_codec_config") != expected_resolved
        ):
            raise ValueError("BlockQuant stored-bit/provenance metadata does not match payload")
        return key, value

    def get_kv_config(self) -> dict:
        config = super().get_kv_config()
        config.update(
            {
                "independent_baseline": True,
                "reference": BLOCKQUANT_REFERENCE,
                "bits": self.bits,
                "value_bits": self.value_bits,
                "block_size": self.block_size,
                "key_scale_mode_requested": self.key_scale_mode,
                "key_scale_mode": _SCALE_ALIASES[self.key_scale_mode],
                "value_scale_mode_requested": self.value_scale_mode,
                "value_scale_mode": _SCALE_ALIASES[self.value_scale_mode],
                "key_outlier_indices": (
                    None if self.key_outlier_indices is None else list(self.key_outlier_indices)
                ),
                "key_n_outlier": self.key_n_outlier,
                "key_outlier_selection": self._outlier_selection_provenance(),
                "outlier_bits": self.outlier_bits,
                "remainder_policy": self.remainder_policy,
                "assignment_mode": "exact-full-codebook",
                "lut": None,
                "codebook_spec": self.codebook_spec,
                "rotation_seed": self.rotation_seed,
                "codebook_seed": self.codebook_seed,
                "paper_kv_recipe": self.paper_kv_recipe,
            }
        )
        return config

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        """Return the role-resolved scientific recipe without fitting artifacts.

        Codebook hashes remain payload-level provenance because fitting the
        unpublished-centroid completion can take minutes at ``p=3,b=4``.  This
        declaration pins every input to that fit (source TeX hash, numerical
        spec, seeds, sizes, exact assignment and remainder policy); ``decode``
        independently recomputes and verifies the materialized SHA records.
        """
        if not _is_int(head_dim) or head_dim < 2:
            raise ValueError("head_dim must be an integer >= 2")
        if self.paper_kv_recipe and head_dim != 128:
            raise ValueError("the paper Llama-3.1 BlockQuant recipe requires head_dim=128")

        def _part(ambient_dim: int, bits: int, role: str, scale_mode: str) -> dict:
            full_blocks, remainder = self._block_layout(ambient_dim, bits)
            codebooks = []
            if full_blocks:
                codebooks.append(
                    {
                        "kind": "full-block",
                        "block_dim": self.block_size,
                        "codebook_size": 1 << (bits * self.block_size),
                        "code_bits": bits * self.block_size,
                    }
                )
            if remainder:
                codebooks.append(
                    {
                        "kind": "lower-dimensional-tail",
                        "block_dim": remainder,
                        "codebook_size": 1 << (bits * remainder),
                        "code_bits": bits * remainder,
                    }
                )
            return {
                "ambient_dim": ambient_dim,
                "bits_per_coordinate": bits,
                "block_size": self.block_size,
                "full_blocks": full_blocks,
                "remainder": remainder,
                "remainder_policy": self.remainder_policy,
                "scale_mode": _SCALE_ALIASES[scale_mode],
                "assignment": "exact-full-codebook-nearest-squared-l2",
                "codebook_spec": self.codebook_spec,
                "codebook_seed": self.codebook_seed,
                "codebooks": codebooks,
                "rotation": _rotation_record(ambient_dim, self._role_seed(role)),
            }

        if self.key_n_outlier is not None or self.key_outlier_indices is not None:
            outlier_count = (
                int(self.key_n_outlier)
                if self.key_n_outlier is not None
                else len(self.key_outlier_indices)
            )
            key = {
                "mode": "split",
                "selection": self._outlier_selection_provenance(),
                "outlier": _part(
                    outlier_count, int(self.outlier_bits), "key_outlier", self.key_scale_mode
                ),
                "ordinary": _part(head_dim - outlier_count, self.bits, "key", self.key_scale_mode),
                "fp16_scale_fields_per_vector": 2,
            }
        else:
            key = {
                "mode": "uniform",
                "part": _part(head_dim, self.bits, "key", self.key_scale_mode),
                "fp16_scale_fields_per_vector": 1,
            }
        value = {
            "mode": "uniform",
            "part": _part(head_dim, self._value_bits(), "value", self.value_scale_mode),
            "fp16_scale_fields_per_vector": 1,
        }
        return {
            "schema_version": 1,
            "codec": "blockquant",
            "reference": BLOCKQUANT_REFERENCE,
            "source_tex_sha256": BLOCKQUANT_SOURCE_TEX_SHA256,
            "head_dim": head_dim,
            "paper_kv_recipe": self.paper_kv_recipe,
            "paper_reproduction": False,
            "random_seed_protocol": (
                "fixed rotation_seed/codebook_seed; not the paper five-random-seed aggregate"
            ),
            "config": self.get_kv_config(),
            "provenance": _paper_provenance(
                self.codebook_spec,
                self.remainder_policy,
                self._outlier_selection_provenance(),
            ),
            "roles": {"key": key, "value": value},
        }

    def _outlier_selection_provenance(self) -> dict:
        if self.key_n_outlier is not None:
            return {
                "paper_statement": "per-attention-head 32 key channels with largest L2 norm",
                "paper_lifecycle_or_identity_coding_published": False,
                "mode": "versioned-reproducibility-completion",
                "completion": BLOCKQUANT_DYNAMIC_OUTLIER_POLICY,
                "energy_statistic": "sum_squares_over_current_encode_update_token_axis",
                "scope": "independent_per_batch_and_attention_head",
                "tie_break": "lowest_channel_index",
                "identity_coding": BLOCKQUANT_IDENTITY_CODING,
                "identity_bits_counted": True,
            }
        if self.key_outlier_indices is not None:
            return {
                "paper_statement": "per-attention-head 32 key channels with largest L2 norm",
                "paper_lifecycle_or_identity_coding_published": False,
                "mode": "externally-pinned-fixed-global-identities",
                "indices": list(self.key_outlier_indices),
                "identity_bits_counted": False,
                "reason": "identities are immutable out-of-band codec configuration",
            }
        return {"mode": "none"}

    def teacher_forced_temporal_dependency(self) -> str | None:
        if self.key_n_outlier is not None:
            return "dynamic_outlier_channels_over_update_tokens"
        return None


__all__ = [
    "BLOCKQUANT_CODEBOOK_SPEC",
    "BLOCKQUANT_DYNAMIC_OUTLIER_POLICY",
    "BLOCKQUANT_IDENTITY_CODING",
    "BLOCKQUANT_REFERENCE",
    "BLOCKQUANT_REMAINDER_POLICY",
    "BLOCKQUANT_SOURCE_TEX_SHA256",
    "BlockQuantCodec",
    "blockquant_codebook",
]
