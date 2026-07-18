"""Paper-faithful FibQuant fixed-rate vector quantization baseline.

This module implements the source codec described in *FibQuant: Universal
Vector Quantization for Random-Access KV-Cache Compression*
(arXiv:2605.11478v1).  It is an independent baseline, not a HayateQuant recipe
or a proxy for one:

1. store one fp16 L2 norm for every cached vector;
2. L2-normalize, then apply one seeded dense Haar orthogonal rotation shared
   by keys, values, layers, heads, prompts, and tokens;
3. split the rotated unit direction into consecutive ``block_size`` blocks;
4. encode every block by exact nearest-codeword search in a codebook for the
   finite-dimensional spherical-Beta source ``f_(d,k)``;
5. decode by table lookup, multiply by the stored fp16 norm, and apply the
   transpose of the rotation.

The integer-rate recipes used in the paper's TinyLlama experiment are
``2 bpc -> (k=4, N=256)``, ``3 bpc -> (k=2, N=64)``, and
``4 bpc -> (k=2, N=256)``.  Explicit ``block_size``/``code_bits`` pairs are
also accepted, including fractional rates, provided ``bits`` equals
``code_bits / block_size`` and the head dimension is divisible by the block
size.  ``code_bits`` is the fixed number of payload bits per block and
``N = 2**code_bits``.

For ``k > 1``, codebooks follow the paper's Appendix algorithm exactly where
specified: Beta-quantile radii, Fibonacci/Roberts--Kronecker directions,
``M=30N`` exact spherical-Beta training samples, four randomly rotated
restarts, and 25 Lloyd iterations.  The v1 paper leaves two reproducibility
details unspecified.  We complete them deterministically as follows:

* the source sample uses ``codebook_seed`` and restart ``j`` uses
  ``codebook_seed + 1_000_003 * (j + 1)``;
* an empty cell is split from the occupied cell with largest total squared
  error by placing the empty centroid at that donor's farthest unclaimed
  sample (ties always choose the lowest index).

At ``k=1`` the implementation deliberately uses the converged analytic
finite-Beta scalar Lloyd--Max codebook from TurboQuant.  This is the scalar
product specialization/control identified by the FibQuant paper and avoids
introducing finite-sample fitting noise into the equality control.

``stored_bits`` is an exact logical fixed-width data-plane count: block
indices plus one signed-fp16 norm per vector for both K and V.  Global
codebooks, the shared rotation, Python metadata, allocator overhead, and the
unpacked physical dtype of the research payload are excluded, consistently
with the other research codecs in :mod:`onecomp.kvcomp`.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from .._base import CompressedKV, KVCodec
from .._registry import register_kv_codec
from .rotation import lloyd_max_beta, random_orthogonal

FIBQUANT_REFERENCE = "arXiv:2605.11478v1"
FIBQUANT_SOURCE_TEX_SHA256 = "7e6b6a962d690a2ab42089a06e8197e08496e783126d4a01361d08f04abbe524"

_PAPER_RECIPES = {
    2.0: (4, 8),  # block size k=4, N=2**8=256
    3.0: (2, 6),  # block size k=2, N=2**6=64
    4.0: (2, 8),  # block size k=2, N=2**8=256
}
_TRAINING_SAMPLES_PER_CODEWORD = 30
_RESTARTS = 4
_LLOYD_ITERS = 25
_MAX_CODE_BITS = 14  # largest N=16384 operating point studied in the paper
_ASSIGNMENT_CHUNK = 4096
_MIN_SEED = -(1 << 63)
_MAX_SEED = (1 << 63) - 1


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_seed(value: int, label: str, *, restart_offsets: bool = False) -> None:
    maximum = _MAX_SEED - (_RESTARTS * 1_000_003 if restart_offsets else 0)
    if not _is_int(value) or not _MIN_SEED <= value <= maximum:
        raise ValueError(f"{label} must be an int in {_MIN_SEED}..{maximum}, got {value!r}")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _roberts_phi(dim: int) -> float:
    """Positive root of ``phi**(dim+1) = phi + 1`` by fixed bisection."""
    lo, hi = 1.0, 2.0
    for _ in range(128):
        mid = 0.5 * (lo + hi)
        if mid ** (dim + 1) > mid + 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _paper_directions(block_size: int, n_codewords: int) -> torch.Tensor:
    """Return the paper's quasi-uniform directions in float64 on CPU."""
    import numpy as np
    from scipy.special import ndtri

    n = np.arange(n_codewords, dtype=np.float64)
    golden_ratio = 0.5 * (1.0 + math.sqrt(5.0))
    golden_fraction = 1.0 - 1.0 / golden_ratio

    if block_size == 2:
        theta = 2.0 * math.pi * n * golden_fraction
        directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    elif block_size == 3:
        z = 1.0 - (2.0 * (n + 1.0) - 1.0) / float(n_codewords)
        theta = 2.0 * math.pi * n * golden_fraction
        radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        directions = np.stack((radial * np.cos(theta), radial * np.sin(theta), z), axis=1)
    elif block_size >= 4:
        phi = _roberts_phi(block_size)
        columns = []
        for j in range(1, block_size + 1):
            xi = np.mod((n + 0.5) * phi ** (-j), 1.0)
            # Midpoint Kronecker points are in (0,1) mathematically.  The
            # clipping only protects ndtri from floating-point endpoints.
            columns.append(ndtri(np.clip(xi, 1.0e-15, 1.0 - 1.0e-15)))
        directions = np.stack(columns, axis=1)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    else:
        raise ValueError("FibQuant quasi-uniform directions require block_size >= 2")

    return torch.from_numpy(directions).to(dtype=torch.float64)


def _paper_radii(ambient_dim: int, block_size: int, n_codewords: int) -> torch.Tensor:
    """Return the paper's Bennett--Gersho Beta-quantile radii."""
    import numpy as np
    from scipy.special import betaincinv

    if block_size == ambient_dim:
        # The full normalized vector lies on S^(d-1), not inside a ball.
        return torch.ones(n_codewords, dtype=torch.float64)

    q = (np.arange(n_codewords, dtype=np.float64) + 0.5) / float(n_codewords)
    if block_size == 2:
        radii = np.sqrt(1.0 - np.power(1.0 - q, 4.0 / float(ambient_dim)))
    else:
        alpha = 0.5 * block_size
        beta = block_size / float(block_size + 2) * (ambient_dim - block_size - 2) / 2.0 + 1.0
        if not beta > 0.0:
            raise ValueError(
                "FibQuant Beta-companding parameter must be positive; "
                f"got beta={beta} for d={ambient_dim}, k={block_size}"
            )
        radii = np.sqrt(betaincinv(alpha, beta, q))
    if not np.isfinite(radii).all():
        raise RuntimeError("FibQuant Beta-quantile initialization produced non-finite radii")
    return torch.from_numpy(radii).to(dtype=torch.float64)


def _sample_spherical_beta(
    ambient_dim: int, block_size: int, count: int, seed: int
) -> torch.Tensor:
    """Sample exactly from the first-k marginal of Unif(S^(d-1))."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(count, ambient_dim, generator=generator, dtype=torch.float64)
    unit = gaussian / gaussian.norm(dim=1, keepdim=True)
    return unit[:, :block_size].contiguous()


def _haar_cpu(dim: int, seed: int) -> torch.Tensor:
    """Float64 seeded Haar matrix for deterministic offline restarts."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.where(torch.diagonal(r) < 0.0, -1.0, 1.0)
    return q * signs.unsqueeze(0)


def _assign_exact(
    samples: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest-centroid assignment, chunked only over independent rows."""
    count = samples.shape[0]
    assignment = torch.empty(count, dtype=torch.int64)
    squared_error = torch.empty(count, dtype=samples.dtype)
    centroid_norm = centroids.square().sum(dim=1)
    for start in range(0, count, _ASSIGNMENT_CHUNK):
        stop = min(start + _ASSIGNMENT_CHUNK, count)
        batch = samples[start:stop]
        # argmax returns the first index on ties, matching Eq. (1)'s
        # deterministic smallest-codeword convention.
        scores = batch @ centroids.t() - 0.5 * centroid_norm.unsqueeze(0)
        index = scores.argmax(dim=1)
        assignment[start:stop] = index
        residual = batch - centroids[index]
        squared_error[start:stop] = residual.square().sum(dim=1)
    return assignment, squared_error


def _lloyd_refine(
    samples: torch.Tensor, initial: torch.Tensor, iterations: int
) -> tuple[torch.Tensor, float]:
    """Paper Lloyd loop with a deterministic high-distortion-cell split."""
    centroids = initial.clone()
    n_codewords, block_size = centroids.shape
    ones = torch.ones(samples.shape[0], dtype=samples.dtype)

    for _ in range(iterations):
        assignment, squared_error = _assign_exact(samples, centroids)
        counts = torch.zeros(n_codewords, dtype=samples.dtype).index_add_(0, assignment, ones)
        sums = torch.zeros(n_codewords, block_size, dtype=samples.dtype).index_add_(
            0, assignment, samples
        )
        occupied = counts > 0
        updated = centroids.clone()
        updated[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)

        empty_indices = (~occupied).nonzero(as_tuple=False).flatten()
        if empty_indices.numel():
            distortion = torch.zeros(n_codewords, dtype=samples.dtype).index_add_(
                0, assignment, squared_error
            )
            available = counts.clone()
            unclaimed = torch.ones(samples.shape[0], dtype=torch.bool)
            for empty_index in empty_indices.tolist():
                donor_score = distortion.clone()
                donor_score[available <= 1] = -math.inf
                donor = int(donor_score.argmax().item())
                if not math.isfinite(float(donor_score[donor])):
                    # This can only occur when the training set has fewer than
                    # two samples per non-empty codeword, which M=30N rules out.
                    raise RuntimeError("unable to split a non-singleton FibQuant donor cell")
                candidates = (assignment == donor) & unclaimed
                candidate_score = torch.where(
                    candidates,
                    squared_error,
                    torch.full_like(squared_error, -math.inf),
                )
                sample_index = int(candidate_score.argmax().item())
                updated[empty_index] = samples[sample_index]
                unclaimed[sample_index] = False
                available[donor] -= 1
                distortion[donor] -= squared_error[sample_index]

        centroids = updated

    _, final_squared_error = _assign_exact(samples, centroids)
    return centroids, float(final_squared_error.sum().item())


def _validate_codebook_request(ambient_dim: int, block_size: int, code_bits: int) -> None:
    if not _is_int(ambient_dim) or ambient_dim < 2:
        raise ValueError(f"ambient_dim must be an int >= 2, got {ambient_dim!r}")
    if not _is_int(block_size) or block_size < 1:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    if block_size > ambient_dim or ambient_dim % block_size:
        raise ValueError(f"block_size={block_size} must divide ambient_dim={ambient_dim}")
    if not _is_int(code_bits) or not 1 <= code_bits <= _MAX_CODE_BITS:
        raise ValueError(f"code_bits must be an int in 1..{_MAX_CODE_BITS}, got {code_bits!r}")
    if block_size == 1 and code_bits > 8:
        raise ValueError("the analytic scalar TurboQuant control supports code_bits <= 8")


@lru_cache(maxsize=32)
def _fibquant_codebook_cpu(
    ambient_dim: int, block_size: int, code_bits: int, codebook_seed: int
) -> torch.Tensor:
    """Build and cache one immutable float32 paper codebook on CPU."""
    _validate_codebook_request(ambient_dim, block_size, code_bits)
    _validate_seed(codebook_seed, "codebook_seed", restart_offsets=True)

    if block_size == 1:
        return lloyd_max_beta(code_bits, ambient_dim).reshape(-1, 1).contiguous()

    n_codewords = 1 << code_bits
    radii = _paper_radii(ambient_dim, block_size, n_codewords)
    directions = _paper_directions(block_size, n_codewords)
    initial = radii.unsqueeze(1) * directions
    samples = _sample_spherical_beta(
        ambient_dim,
        block_size,
        _TRAINING_SAMPLES_PER_CODEWORD * n_codewords,
        codebook_seed,
    )

    best_codebook = None
    best_sse = math.inf
    for restart in range(_RESTARTS):
        restart_seed = codebook_seed + 1_000_003 * (restart + 1)
        rotated = initial @ _haar_cpu(block_size, restart_seed)
        candidate, sse = _lloyd_refine(samples, rotated, _LLOYD_ITERS)
        if sse < best_sse:
            best_sse = sse
            best_codebook = candidate

    if best_codebook is None or not torch.isfinite(best_codebook).all():
        raise RuntimeError("FibQuant Lloyd fitting did not produce a finite codebook")
    if bool((best_codebook.norm(dim=1) > 1.0 + 1.0e-12).any()):
        raise RuntimeError("FibQuant Lloyd centroids escaped the unit ball")
    return best_codebook.to(dtype=torch.float32).contiguous()


def fibquant_codebook(
    ambient_dim: int, block_size: int, code_bits: int, codebook_seed: int = 0
) -> torch.Tensor:
    """Return a defensive CPU copy of the resolved FibQuant codebook."""
    return _fibquant_codebook_cpu(ambient_dim, block_size, code_bits, codebook_seed).clone()


@lru_cache(maxsize=128)
def _fibquant_codebook_on(
    ambient_dim: int,
    block_size: int,
    code_bits: int,
    codebook_seed: int,
    device_str: str,
) -> torch.Tensor:
    return _fibquant_codebook_cpu(ambient_dim, block_size, code_bits, codebook_seed).to(
        device=torch.device(device_str)
    )


@lru_cache(maxsize=64)
def _codebook_sha256(ambient_dim: int, block_size: int, code_bits: int, codebook_seed: int) -> str:
    return _tensor_sha256(
        _fibquant_codebook_cpu(ambient_dim, block_size, code_bits, codebook_seed)
    )


@lru_cache(maxsize=32)
def _rotation_cpu(ambient_dim: int, seed: int) -> torch.Tensor:
    return random_orthogonal(ambient_dim, seed)


@lru_cache(maxsize=64)
def _rotation_on(ambient_dim: int, seed: int, device_str: str) -> torch.Tensor:
    return _rotation_cpu(ambient_dim, seed).to(device=torch.device(device_str))


@lru_cache(maxsize=32)
def _rotation_sha256(ambient_dim: int, seed: int) -> str:
    return _tensor_sha256(_rotation_cpu(ambient_dim, seed))


def _resolve_recipe(
    bits: float, block_size: "int | None", code_bits: "int | None"
) -> tuple[int, int]:
    if isinstance(bits, bool) or not isinstance(bits, (int, float)):
        raise ValueError(f"bits must be a positive finite number, got {bits!r}")
    rate = float(bits)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"bits must be a positive finite number, got {bits!r}")

    if block_size is None and code_bits is None:
        if rate not in _PAPER_RECIPES:
            available = ", ".join(str(int(value)) for value in sorted(_PAPER_RECIPES))
            raise ValueError(
                f"FibQuant's implicit paper recipe supports bits in {{{available}}}; "
                "provide both block_size and code_bits for another fixed rate"
            )
        return _PAPER_RECIPES[rate]
    if block_size is None:
        raise ValueError("block_size is required when code_bits is explicit")
    if not _is_int(block_size) or block_size < 1:
        raise ValueError(f"block_size must be a positive int, got {block_size!r}")
    if code_bits is None:
        derived = rate * block_size
        if not derived.is_integer():
            raise ValueError(
                "bits * block_size must be an integer fixed block width; "
                f"got {rate} * {block_size}"
            )
        code_bits = int(derived)
    if not _is_int(code_bits):
        raise ValueError(f"code_bits must be an int, got {code_bits!r}")
    actual_rate = code_bits / float(block_size)
    if not math.isclose(rate, actual_rate, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"bits={rate} must equal code_bits/block_size={actual_rate} "
            "to prevent mislabeled fixed-rate results"
        )
    return block_size, code_bits


@register_kv_codec("fibquant")
@dataclass
class FibQuantCodec(KVCodec):
    """Independent exact-search FibQuant quality baseline.

    ``bits`` is the actual payload rate per coordinate, excluding the fp16
    norm header.  Omitting ``block_size`` and ``code_bits`` selects the
    paper's integer-rate recipe.  Supplying a custom pair requires
    ``bits == code_bits / block_size`` so results cannot silently carry a
    wrong nominal rate.
    """

    flag_calibration: bool = False
    bits: float = 4.0
    block_size: int = None
    code_bits: int = None
    seed: int = 0
    codebook_seed: int = 0
    key_axis: str = "token"
    value_axis: str = "token"

    def validate_params(self):
        block_size, code_bits = _resolve_recipe(self.bits, self.block_size, self.code_bits)
        if code_bits > _MAX_CODE_BITS:
            raise ValueError(
                f"code_bits must be <= {_MAX_CODE_BITS}, the largest paper operating point"
            )
        if block_size == 1 and code_bits > 8:
            raise ValueError("the analytic scalar TurboQuant control supports code_bits <= 8")
        _validate_seed(self.seed, "seed")
        _validate_seed(self.codebook_seed, "codebook_seed", restart_offsets=True)

    def _resolved_recipe(self) -> tuple[int, int]:
        return _resolve_recipe(self.bits, self.block_size, self.code_bits)

    def _validate_tensor_pair(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("FibQuant key and value must be torch tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("FibQuant expects K/V layout (batch, heads, sequence, head_dim)")
        if key.shape != value.shape:
            raise ValueError(
                f"FibQuant key/value shapes must match, got {tuple(key.shape)} and "
                f"{tuple(value.shape)}"
            )
        if key.device != value.device:
            raise ValueError("FibQuant key and value must be on the same device")
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError("FibQuant key and value must have floating-point dtypes")
        if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
            raise ValueError("FibQuant rejects non-finite K/V inputs")
        block_size, code_bits = self._resolved_recipe()
        _validate_codebook_request(key.shape[-1], block_size, code_bits)

    @staticmethod
    def _nearest_code(blocks: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        flat = blocks.reshape(-1, blocks.shape[-1])
        output = torch.empty(flat.shape[0], dtype=torch.int64, device=flat.device)
        codeword_norm = codebook.square().sum(dim=1)
        for start in range(0, flat.shape[0], _ASSIGNMENT_CHUNK):
            stop = min(start + _ASSIGNMENT_CHUNK, flat.shape[0])
            scores = flat[start:stop] @ codebook.t() - 0.5 * codeword_norm.unsqueeze(0)
            output[start:stop] = scores.argmax(dim=1)
        return output.reshape(blocks.shape[:-1])

    def _encode_tensor(self, tensor: torch.Tensor) -> dict:
        block_size, code_bits = self._resolved_recipe()
        ambient_dim = tensor.shape[-1]
        rotation = _rotation_on(ambient_dim, self.seed, str(tensor.device))
        codebook = _fibquant_codebook_on(
            ambient_dim,
            block_size,
            code_bits,
            self.codebook_seed,
            str(tensor.device),
        )

        source = tensor.float()
        norm = source.norm(dim=-1, keepdim=True)
        if not bool(torch.isfinite(norm).all()):
            raise ValueError("FibQuant L2 norm is not finite in float32")
        safe_norm = torch.where(norm > 0.0, norm, torch.ones_like(norm))
        rotated_unit = (source / safe_norm) @ rotation
        blocks = rotated_unit.reshape(*tensor.shape[:-1], ambient_dim // block_size, block_size)
        codes = self._nearest_code(blocks, codebook)
        stored_norm = norm.to(dtype=torch.float16)
        if not bool(torch.isfinite(stored_norm).all()):
            raise ValueError("FibQuant L2 norm is not representable by its fp16 header")
        return {
            "codec": "fibquant",
            "schema_version": 1,
            "codes": codes,
            "norm": stored_norm,
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
            "ambient_dim": ambient_dim,
            "block_size": block_size,
            "code_bits": code_bits,
            "rotation_seed": self.seed,
            "codebook_seed": self.codebook_seed,
            "codebook_sha256": _codebook_sha256(
                ambient_dim, block_size, code_bits, self.codebook_seed
            ),
            "rotation_sha256": _rotation_sha256(ambient_dim, self.seed),
        }

    @staticmethod
    def _payload_bits(tensor: torch.Tensor, block_size: int, code_bits: int) -> int:
        vectors = tensor.numel() // tensor.shape[-1]
        blocks_per_vector = tensor.shape[-1] // block_size
        return int(vectors * blocks_per_vector * code_bits)

    @staticmethod
    def _norm_bits(tensor: torch.Tensor) -> int:
        vectors = tensor.numel() // tensor.shape[-1]
        return int(vectors * 16)

    def encode(self, key, value, ctx=None) -> CompressedKV:
        _ = ctx
        self._validate_tensor_pair(key, value)
        block_size, code_bits = self._resolved_recipe()
        key_payload = self._encode_tensor(key)
        value_payload = self._encode_tensor(value)
        payload_bits = self._payload_bits(key, block_size, code_bits) + self._payload_bits(
            value, block_size, code_bits
        )
        norm_bits = self._norm_bits(key) + self._norm_bits(value)
        return CompressedKV(
            key_q=key_payload,
            value_q=value_payload,
            meta={
                "codec": "fibquant",
                "schema_version": 1,
                "stored_bits": int(payload_bits + norm_bits),
                "stored_payload_bits": int(payload_bits),
                "stored_norm_bits": int(norm_bits),
                "resolved_codec_config": self.get_resolved_kv_config(key.shape[-1]),
            },
        )

    def _decode_tensor(self, payload: dict) -> torch.Tensor:
        required = {
            "codec",
            "schema_version",
            "codes",
            "norm",
            "shape",
            "dtype",
            "ambient_dim",
            "block_size",
            "code_bits",
            "rotation_seed",
            "codebook_seed",
            "codebook_sha256",
            "rotation_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("malformed FibQuant payload fields")
        if payload["codec"] != "fibquant" or payload["schema_version"] != 1:
            raise ValueError("unsupported FibQuant payload identity/version")

        ambient_dim = payload["ambient_dim"]
        block_size = payload["block_size"]
        code_bits = payload["code_bits"]
        codebook_seed = payload["codebook_seed"]
        rotation_seed = payload["rotation_seed"]
        _validate_codebook_request(ambient_dim, block_size, code_bits)
        if (block_size, code_bits) != self._resolved_recipe():
            raise ValueError("FibQuant payload rate recipe does not match decoder configuration")
        if not _is_int(codebook_seed) or not _is_int(rotation_seed):
            raise ValueError("FibQuant payload seeds must be integers")
        if rotation_seed != self.seed or codebook_seed != self.codebook_seed:
            raise ValueError("FibQuant payload seed does not match the decoder configuration")

        shape = payload["shape"]
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(not _is_int(size) or size < 0 for size in shape)
            or shape[-1] != ambient_dim
        ):
            raise ValueError("invalid FibQuant payload shape")
        expected_code_shape = shape[:-1] + (ambient_dim // block_size,)
        codes = payload["codes"]
        norm = payload["norm"]
        if not isinstance(codes, torch.Tensor) or tuple(codes.shape) != expected_code_shape:
            raise ValueError("invalid FibQuant code tensor shape")
        if codes.dtype != torch.int64:
            raise ValueError("FibQuant research payload codes must be int64")
        if not isinstance(norm, torch.Tensor) or tuple(norm.shape) != shape[:-1] + (1,):
            raise ValueError("invalid FibQuant norm tensor shape")
        if norm.dtype != torch.float16:
            raise ValueError("FibQuant norm data-plane field must be fp16")
        if norm.device != codes.device:
            raise ValueError("FibQuant codes and norm must share a device")
        if not bool(torch.isfinite(norm).all()) or bool((norm < 0.0).any()):
            raise ValueError("FibQuant norm data-plane field must be finite and non-negative")
        n_codewords = 1 << code_bits
        if codes.numel() and (bool((codes < 0).any()) or bool((codes >= n_codewords).any())):
            raise ValueError("FibQuant code index is outside the resolved codebook")

        if payload["codebook_sha256"] != _codebook_sha256(
            ambient_dim, block_size, code_bits, codebook_seed
        ):
            raise ValueError("FibQuant payload codebook digest does not match resolved codebook")
        if payload["rotation_sha256"] != _rotation_sha256(ambient_dim, rotation_seed):
            raise ValueError("FibQuant payload rotation digest does not match resolved rotation")
        codebook = _fibquant_codebook_on(
            ambient_dim,
            block_size,
            code_bits,
            codebook_seed,
            str(codes.device),
        )
        rotation = _rotation_on(ambient_dim, rotation_seed, str(codes.device))
        reconstructed_unit = codebook[codes].reshape(shape)
        reconstructed = (reconstructed_unit @ rotation.t()) * norm.to(dtype=torch.float32)
        dtype = payload["dtype"]
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError("invalid FibQuant output dtype metadata")
        return reconstructed.to(dtype=dtype)

    def decode(self, compressed: CompressedKV, ctx=None):
        _ = ctx
        if not isinstance(compressed, CompressedKV):
            raise TypeError("FibQuant decode expects a CompressedKV")
        if compressed.meta.get("codec") != "fibquant":
            raise ValueError("compressed payload was not produced by FibQuant")
        return self._decode_tensor(compressed.key_q), self._decode_tensor(compressed.value_q)

    def get_kv_config(self) -> dict:
        cfg = super().get_kv_config()
        cfg.update(
            {
                "bits": float(self.bits),
                "block_size": self.block_size,
                "code_bits": self.code_bits,
                "seed": self.seed,
                "codebook_seed": self.codebook_seed,
            }
        )
        return cfg

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        block_size, code_bits = self._resolved_recipe()
        _validate_codebook_request(head_dim, block_size, code_bits)
        n_codewords = 1 << code_bits
        role = {
            "payload_bits_per_coordinate": code_bits / float(block_size),
            "stored_bits_per_coordinate_including_norm": code_bits / float(block_size)
            + 16.0 / float(head_dim),
            "norm_header": "signed-fp16-per-vector",
            "block_size": block_size,
            "code_bits": code_bits,
            "codebook_size": n_codewords,
            "encoder": "exact-nearest-squared-l2",
            "source_law": "finite-spherical-beta-f_d_k",
            "normalization": "l2-before-rotation",
            "rotation": "dense-haar-gaussian-qr",
            "rotation_seed": self.seed,
            "codebook_seed": self.codebook_seed,
            "codebook_dtype": "float32",
            "codebook_sha256": _codebook_sha256(
                int(head_dim), block_size, code_bits, self.codebook_seed
            ),
        }
        return {
            "schema_version": 1,
            "codec": "fibquant",
            "reference": FIBQUANT_REFERENCE,
            "source_tex_sha256": FIBQUANT_SOURCE_TEX_SHA256,
            "head_dim": int(head_dim),
            "calibration": None,
            "fixed_rate": True,
            "shared_codebook": True,
            "shared_rotation": True,
            "paper_reproduction": False,
            "reproduction_limit": (
                "the v1 source publishes no centroid artifact, implementation, "
                "training/restart seeds, or exact empty-cell tie protocol"
            ),
            "rotation_sha256": _rotation_sha256(int(head_dim), self.seed),
            "paper_recipe": float(self.bits) in _PAPER_RECIPES
            and (block_size, code_bits) == _PAPER_RECIPES[float(self.bits)],
            "codebook_training": {
                "source": (
                    "exact-spherical-beta-via-normalized-gaussian"
                    if block_size > 1
                    else "analytic-finite-beta-scalar-control"
                ),
                "samples_per_codeword": (_TRAINING_SAMPLES_PER_CODEWORD if block_size > 1 else 0),
                "sample_count": (
                    _TRAINING_SAMPLES_PER_CODEWORD * n_codewords if block_size > 1 else 0
                ),
                "restarts": _RESTARTS if block_size > 1 else 0,
                "lloyd_iterations": _LLOYD_ITERS if block_size > 1 else 0,
                "initializer": (
                    "beta-quantile-fibonacci-roberts"
                    if block_size > 1
                    else "analytic-turboquant-scalar-control"
                ),
                "empty_cell_repair": (
                    "max-total-sse-donor-farthest-unclaimed-sample"
                    if block_size > 1
                    else "not-applicable"
                ),
            },
            "roles": {"key": dict(role), "value": dict(role)},
        }


__all__ = [
    "FIBQUANT_REFERENCE",
    "FIBQUANT_SOURCE_TEX_SHA256",
    "FibQuantCodec",
    "fibquant_codebook",
]
