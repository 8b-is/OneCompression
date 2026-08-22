"""Optional SpinQuant-style calibration for :class:`HayateQuantCodec`.

HayateQuant is data-free by default (random rotation + codebooks fit to the
rotated source *law*). This module adds an **optional, offline** calibration
stage in the spirit of SpinQuant (learned orthogonal rotations, optimised on
the Cayley manifold) adapted to KV-cache block-VQ:

* **Learned per-role rotation** ``W = W0 @ cayley(A)``: initialised at the
  codec's own data-free rotation ``W0`` (so the data-free behaviour is the
  starting point, never a regression) and optimised with Adam on the
  skew-symmetric parameter ``A`` to minimise the *actual* block quantization
  distortion of real rotated-and-RMS-normalised KV vectors. Keys and values
  get separate rotations (their distributions differ; the codec already
  treats the roles asymmetrically everywhere else).
* **Real-data codebooks** (optional): closed-form Lloyd refits of the shipped
  Gaussian codebook on real z-space blocks, alternating with the rotation
  steps (joint W/codebook descent, both monotone in the same objective).

The output is a small per-model artifact (:class:`HayateQuantCalibration`,
two ``(d, d)`` fp32 matrices + two codebooks, ~200KB) that is loaded into the
codec via ``HayateQuantCodec(calibration=...)``. Per-token payload storage is
completely unchanged -- like SpinQuant's rotations, the artifact is
model-level side information, not per-tensor state. Runtime cost is likewise
unchanged: the learned rotation replaces the random one inside the same
single GEMM, and the learned codebook has the same ``K``.

Robustness: the optimisation keeps the **best** (W, codebook) pair as
measured on a held-out shard of the calibration samples, so a diverging run
can never produce an artifact worse than the data-free starting point on
that shard.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from ..rotation import (
    hybrid_rotation_matrix,
    random_orthogonal,
    is_power_of_two,
)
from .vq import quantize_to_vq_codebook, lookup_codewords, build_vq_lut

_ROLES = ("k", "v")


def _tensor_digest(tensors) -> str:
    h = hashlib.sha1()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:12]


@dataclass
class HayateQuantCalibration:
    """Per-model learned-rotation / learned-codebook artifact.

    Attributes:
        head_dim: Head dim ``d`` the artifact was fit for (rotations are
            ``(d, d)``); the codec only applies it to matching subsets.
        rot: ``role -> (d, d)`` learned orthogonal rotation (fp32), or ``{}``
            if rotations were not learned.
        codebook: ``role -> (K, block_size)`` learned codebook (fp32), or
            ``{}`` if codebooks were not learned.
        cb_key: ``role -> (bits, block_size)`` the codebook was fit for; the
            codec falls back to the data-free codebook when its resolved
            ``(bits, block_size)`` differs (e.g. the user changed ``bits``
            after calibrating), so a stale artifact can degrade gracefully
            instead of producing garbage.
        meta: Free-form fit info (model, steps, distortion before/after ...).
    """

    head_dim: int
    rot: dict = field(default_factory=dict)
    codebook: dict = field(default_factory=dict)
    cb_key: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self._dev_cache: dict = {}
        self._lut_cache: dict = {}
        self._digest = None

    @property
    def digest(self) -> str:
        """Content hash stored in payload rot tags: decode verifies the same
        artifact is loaded (same self-describing-payload convention as the
        stored rotation seed)."""
        if self._digest is None:
            ts = [self.rot[r] for r in _ROLES if r in self.rot]
            ts += [self.codebook[r] for r in _ROLES if r in self.codebook]
            self._digest = _tensor_digest(ts) if ts else "empty"
        return self._digest

    # --- device-resident accessors (cached, mirrors _*_on in hayatequant) --
    def rotation_on(self, role: str, m: int, device, dtype=torch.float32):
        w = self.rot.get(role)
        if w is None or w.shape[0] != m:
            return None
        key = ("rot", role, str(device), dtype)
        if key not in self._dev_cache:
            self._dev_cache[key] = w.to(device=device, dtype=dtype)
        return self._dev_cache[key]

    def codebook_on(self, role: str, bits: int, block_size: int, device):
        if self.cb_key.get(role) != (bits, block_size):
            return None
        cb = self.codebook.get(role)
        if cb is None:
            return None
        key = ("cb", role, str(device))
        if key not in self._dev_cache:
            self._dev_cache[key] = cb.to(device=device)
        return self._dev_cache[key]

    def lut_on(self, role: str, bits: int, block_size: int, bins: int, top_t: int, device):
        cb = self.codebook_on(role, bits, block_size, device)
        if cb is None:
            return None
        key = (role, bins, top_t, str(device))
        if key not in self._lut_cache:
            self._lut_cache[key] = build_vq_lut(cb, bins, top_t)
        return self._lut_cache[key]

    # --- persistence ---------------------------------------------------------
    def save(self, path: str):
        torch.save(
            {
                "head_dim": self.head_dim,
                "rot": self.rot,
                "codebook": self.codebook,
                "cb_key": self.cb_key,
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "HayateQuantCalibration":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            head_dim=int(d["head_dim"]),
            rot=dict(d["rot"]),
            codebook=dict(d["codebook"]),
            cb_key={k: tuple(v) for k, v in d["cb_key"].items()},
            meta=dict(d.get("meta", {})),
        )


def _initial_rotation(codec, d: int, block_size: int, bits: int, device) -> torch.Tensor:
    """The codec's own resolved data-free rotation as a (d, d) GEMM matrix --
    the Cayley optimisation starts here, so step 0 IS the data-free codec."""
    mode = codec._resolved_mode(d, block_size, bits)
    if mode == "hybrid" and is_power_of_two(d):
        return hybrid_rotation_matrix(d, block_size, codec.seed, str(device), torch.float32)
    return random_orthogonal(d, codec.seed).to(device=device, dtype=torch.float32)


def _cayley(b: torch.Tensor) -> torch.Tensor:
    """Orthogonal ``(I - A)^{-1} (I + A)`` with ``A = (B - B^T)/2`` -- the
    standard Cayley retraction used by SpinQuant for rotation learning."""
    a = (b - b.t()) / 2
    eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
    return torch.linalg.solve(eye - a, eye + a)


def _z_blocks(x: torch.Tensor, w: torch.Tensor, block_size: int) -> torch.Tensor:
    xr = x @ w
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    return (xr / rms).reshape(-1, block_size)


@torch.no_grad()
def _lloyd_refit(zb: torch.Tensor, cb: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """A few closed-form Lloyd iterations of ``cb`` on real blocks ``zb``.
    Empty clusters keep their previous centroid (they can be re-captured on a
    later refit once the rotation has moved)."""
    k = cb.shape[0]
    for _ in range(iters):
        codes = quantize_to_vq_codebook(zb, cb)
        ones = torch.ones(zb.shape[0], device=zb.device)
        counts = torch.zeros(k, device=zb.device).index_add_(0, codes, ones)
        sums = torch.zeros_like(cb).index_add_(0, codes, zb)
        keep = (counts == 0).unsqueeze(1)
        cb = torch.where(keep, cb, sums / counts.clamp_min(1.0).unsqueeze(1))
    return cb


@torch.no_grad()
def _distortion(x: torch.Tensor, w: torch.Tensor, cb: torch.Tensor, block_size: int) -> float:
    zb = _z_blocks(x, w, block_size)
    codes = quantize_to_vq_codebook(zb, cb)
    return float((zb - lookup_codewords(cb, codes)).pow(2).mean())


@torch.no_grad()
def calibrate_hayatequant(
    codec,
    key_samples: torch.Tensor,
    value_samples: torch.Tensor,
    *,
    learn_rotation: bool = True,
    learn_codebook: bool = True,
    steps: int = 300,
    lr: float = 1e-3,
    batch_vectors: int = 16384,
    refit_every: int = 50,
    max_vectors: int = 500_000,
    holdout_frac: float = 0.1,
    seed: int = 0,
    device: str = None,
    verbose: bool = False,
) -> HayateQuantCalibration:
    """Fit a :class:`HayateQuantCalibration` for ``codec`` from real KV samples.

    Args:
        codec: The canonical HayateQuant codec whose rate defines the fixed
            role codebooks and starting rotation.
        key_samples / value_samples: ``(N, head_dim)`` real K / V vectors
            (any dtype/device; flattened from ``(b, h, s, d)`` captures).
        learn_rotation: Optimise the per-role rotation (Cayley/Adam).
        learn_codebook: Refit codewords at the same fixed cardinality. This
            changes persistent model state, never the packed token layout.
        steps / lr / batch_vectors / refit_every: Optimisation budget.
        max_vectors: Sample cap per role (memory/speed).
        holdout_frac: Fraction of samples held out for best-artifact
            selection (never trained on).
        seed: Batch-sampling seed.
        device: Fit device (default: samples' device, or cuda if available).

    Returns:
        HayateQuantCalibration with per-role rotations/codebooks and fit
        diagnostics in ``meta`` (held-out distortion before/after per role).
    """
    if not (learn_rotation or learn_codebook):
        raise ValueError("nothing to learn: enable learn_rotation and/or learn_codebook")
    from .codec import (  # local import: avoid cycle at module load
        _block_codebook_by_code_bits,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cal = HayateQuantCalibration(head_dim=int(key_samples.shape[-1]))
    cal.meta = {
        "steps": steps,
        "lr": lr,
        "learn_rotation": learn_rotation,
        "learn_codebook": learn_codebook,
        "rotation_only_fixed_payload": bool(learn_rotation and not learn_codebook),
    }

    for role, samples in (("k", key_samples), ("v", value_samples)):
        bits = codec.bits
        d = int(samples.shape[-1])
        block_size = codec._resolve_block_size_for(role, bits, d)
        code_bits = codec._code_bits_for(role, bits, block_size)
        rotation_rate = codec._rotation_rate_for(role, bits, block_size)
        cal.meta[f"{role}_block_size"] = block_size
        cal.meta[f"{role}_code_bits_per_block"] = code_bits
        cal.meta[f"{role}_payload_bits_per_coordinate"] = code_bits / block_size
        cal.meta[f"{role}_base_rotation_rate"] = rotation_rate
        x = samples.reshape(-1, d).to(device=device, dtype=torch.float32)
        if x.shape[0] > max_vectors:
            sel = torch.randperm(x.shape[0], generator=gen)[:max_vectors]
            x = x[sel.to(x.device)]
        n_hold = max(1, int(x.shape[0] * holdout_frac))
        x_hold, x_train = x[:n_hold], x[n_hold:]

        w0 = _initial_rotation(codec, d, block_size, rotation_rate, device)
        cb = _block_codebook_by_code_bits(code_bits).to(device)
        base = _distortion(x_hold, w0, cb, block_size)
        best = {"w": None, "cb": None, "dist": base, "accepted": False}

        b = torch.zeros(d, d, device=device, requires_grad=learn_rotation)
        opt = torch.optim.Adam([b], lr=lr) if learn_rotation else None
        n_steps = steps if learn_rotation else (steps // refit_every + 1)
        for step in range(n_steps):
            if learn_rotation:
                idx = torch.randint(
                    0, x_train.shape[0], (min(batch_vectors, x_train.shape[0]),), generator=gen
                ).to(x_train.device)
                w = w0 @ _cayley(b)
                zb = _z_blocks(x_train[idx], w, block_size)
                with torch.no_grad():
                    codes = quantize_to_vq_codebook(zb, cb)
                loss = (zb - lookup_codewords(cb, codes)).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            do_refit = learn_codebook and (
                not learn_rotation or step % refit_every == refit_every - 1
            )
            if do_refit or step == n_steps - 1:
                with torch.no_grad():
                    w = w0 @ _cayley(b) if learn_rotation else w0
                    if learn_codebook:
                        cb = _lloyd_refit(_z_blocks(x_train, w, block_size), cb)
                    dist = _distortion(x_hold, w, cb, block_size)
                    if dist < best["dist"]:
                        best = {"w": w.clone(), "cb": cb.clone(), "dist": dist, "accepted": True}
                    if verbose:
                        print(
                            f"[calib:{role}] step {step + 1}/{n_steps} "
                            f"holdout dist {dist:.6f} (base {base:.6f})",
                            flush=True,
                        )

        if learn_rotation and best["accepted"]:
            cal.rot[role] = best["w"].detach().cpu()
        if learn_codebook and best["accepted"]:
            cal.codebook[role] = best["cb"].detach().cpu()
            cal.cb_key[role] = (bits, block_size)
        cal.meta[f"{role}_dist_before"] = base
        cal.meta[f"{role}_dist_after"] = best["dist"]
        cal.meta[f"{role}_accepted"] = best["accepted"]
    return cal


@torch.no_grad()
def capture_kv_samples(
    model, tokenizer, texts, *, max_tokens: int = 2048, layer_stride: int = 4, device=None
):
    """Capture real ``(N, head_dim)`` K / V samples from ``model`` on
    ``texts`` (list of strings), mixing every ``layer_stride``-th layer so
    one shared per-role rotation serves all layers (the codec, like
    SpinQuant's shared R1, uses one rotation model-wide)."""
    ks, vs = [], []
    for text in texts:
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        ).input_ids
        ids = ids.to(device or model.device)
        out = model(ids, use_cache=True)
        layers = out.past_key_values.layers
        for i in range(0, len(layers), layer_stride):
            ks.append(layers[i].keys.float().reshape(-1, layers[i].keys.shape[-1]).cpu())
            vs.append(layers[i].values.float().reshape(-1, layers[i].values.shape[-1]).cpu())
    return torch.cat(ks), torch.cat(vs)
