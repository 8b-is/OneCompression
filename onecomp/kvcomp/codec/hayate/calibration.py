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
import math
from dataclasses import dataclass, field

import torch

from ..rotation import (
    hybrid_rotation_matrix,
    signed_hadamard_matrix,
    random_orthogonal,
    is_power_of_two,
)
from .._scale import fit_scale
from .vq import resolve_block_size, quantize_to_vq_codebook, lookup_codewords, build_vq_lut

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
        layer_rot: ``role -> layer_idx -> (d, d)`` learned orthogonal
            rotations for layer-aware calibration.  Missing layers fall back
            to ``rot`` and then to the data-free codec rotation.
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
    layer_rot: dict = field(default_factory=dict)
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
            for r in sorted(self.layer_rot):
                for li in sorted(self.layer_rot[r]):
                    ts.append(self.layer_rot[r][li])
            ts += [self.codebook[r] for r in _ROLES if r in self.codebook]
            self._digest = _tensor_digest(ts) if ts else "empty"
        return self._digest

    # --- device-resident accessors (cached, mirrors _*_on in hayatequant) --
    def rotation_on(self, role: str, m: int, device, dtype=torch.float32, layer_idx=None):
        w = None
        if layer_idx is not None:
            w = self.layer_rot.get(role, {}).get(int(layer_idx))
        if w is None:
            w = self.rot.get(role)
        if w is None or w.shape[0] != m:
            return None
        key = ("rot", role, None if layer_idx is None else int(layer_idx), str(device), dtype)
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
        torch.save({"head_dim": self.head_dim, "rot": self.rot,
                    "layer_rot": self.layer_rot,
                    "codebook": self.codebook, "cb_key": self.cb_key,
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "HayateQuantCalibration":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(head_dim=int(d["head_dim"]), rot=dict(d["rot"]),
                   layer_rot={r: {int(li): w for li, w in dict(layers).items()}
                              for r, layers in dict(d.get("layer_rot", {})).items()},
                   codebook=dict(d["codebook"]),
                   cb_key={k: tuple(v) for k, v in d["cb_key"].items()},
                   meta=dict(d.get("meta", {})))


def _initial_rotation(codec, d: int, block_size: int, bits: int, device) -> torch.Tensor:
    """The codec's own resolved data-free rotation as a (d, d) GEMM matrix --
    the Cayley optimisation starts here, so step 0 IS the data-free codec."""
    mode = codec._resolved_mode(d, block_size, bits)
    if mode == "hybrid" and is_power_of_two(d):
        return hybrid_rotation_matrix(d, block_size, codec.seed, str(device), torch.float32)
    if mode in ("hadamard_gemm", "hadamard") and is_power_of_two(d):
        return signed_hadamard_matrix(d, codec.seed, str(device), torch.float32)
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
def _role_error(codec, role: str, x: torch.Tensor, w: torch.Tensor,
                cb: torch.Tensor, block_size: int) -> float:
    """Role-aware held-out error for choosing calibrated rotations.

    Keys are judged by angular error, a query-agnostic proxy for attention-logit
    preservation.  Values are judged by reconstruction MSE after the same
    role-specific scale fitting used by the online codec.  This better matches
    downstream use than raw z-space VQ distortion alone.
    """
    d = x.shape[-1]
    xr = x @ w
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    n_blocks = d // block_size

    def _lookup(vec_full):
        vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
        codes_ = quantize_to_vq_codebook(vb.reshape(-1, block_size), cb)
        chat_b = lookup_codewords(cb, codes_).reshape(*vb.shape)
        return codes_.reshape(vb.shape[:-1]), chat_b.reshape(*vec_full.shape[:-1], d)

    def _requant(z_full):
        codes2, chat2 = _lookup(z_full)
        return chat2, codes2

    codes, chat = _lookup(xr / rms)
    scale, codes2 = fit_scale(xr, chat, codec._scale_mode_for(role), rms,
                              requantize_fn=_requant)
    if codes2 is not None:
        chat = lookup_codewords(cb, codes2.reshape(-1)).reshape(*chat.shape)
    xhat = (chat * scale) @ w.t()
    if role == "k":
        xn = torch.nn.functional.normalize(x, dim=-1)
        hn = torch.nn.functional.normalize(xhat, dim=-1)
        return float((1.0 - (xn * hn).sum(dim=-1)).mean())
    return float((xhat - x).pow(2).sum() / x.pow(2).sum().clamp_min(1e-12))


def _reconstruct_role(codec, role: str, x: torch.Tensor, w: torch.Tensor,
                      cb: torch.Tensor, block_size: int) -> torch.Tensor:
    """Differentiable-through-rotation reconstruction with fixed assignments.

    Codeword assignments are discrete and treated as stop-gradient, matching
    the existing calibration loop.  The rotation and scale path still receive
    gradients, which is enough for small Cayley updates around the frontier
    rotation.
    """
    d = x.shape[-1]
    xr = x @ w
    rms = xr.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    n_blocks = d // block_size

    def _lookup(vec_full):
        vb = vec_full.reshape(*vec_full.shape[:-1], n_blocks, block_size)
        with torch.no_grad():
            codes_ = quantize_to_vq_codebook(vb.reshape(-1, block_size), cb)
        chat_b = lookup_codewords(cb, codes_).reshape(*vb.shape)
        return codes_.reshape(vb.shape[:-1]), chat_b.reshape(*vec_full.shape[:-1], d)

    def _requant(z_full):
        codes2, chat2 = _lookup(z_full)
        return chat2, codes2

    _, chat = _lookup(xr / rms)
    scale, codes2 = fit_scale(xr, chat, codec._scale_mode_for(role), rms,
                              requantize_fn=_requant)
    if codes2 is not None:
        chat = lookup_codewords(cb, codes2.reshape(-1)).reshape(*chat.shape)
    return (chat * scale) @ w.t()


@torch.no_grad()
def _attention_weighted_key_error(codec, x: torch.Tensor, w: torch.Tensor,
                                  cb: torch.Tensor, block_size: int,
                                  qcov: torch.Tensor) -> float:
    xhat = _reconstruct_role(codec, "k", x, w, cb, block_size)
    err = xhat - x
    return float(((err @ qcov) * err).sum(dim=-1).mean() / x.shape[-1])


@torch.no_grad()
def spectral_balanced_rotation(
    samples: torch.Tensor,
    *,
    seed: int = 0,
    center: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Closed-form covariance-balanced rotation for a single KV role.

    Let ``C = E[x x^T]`` and ``C = U diag(lambda) U^T``.  For power-of-two
    head dimensions, set ``W = U H`` with an orthonormal signed Hadamard
    matrix ``H``.  Then

        diag(W^T C W) = tr(C) / d

    because every Hadamard entry has magnitude ``1/sqrt(d)``.  This is a
    calibration-derived, fixed orthogonal rotation that spreads channel energy
    uniformly before the existing block-VQ codebook is applied.  It costs the
    same single GEMM at inference as any other calibrated HayateQuant rotation.
    For non-power-of-two dimensions, a dense Haar rotation is used as a
    deterministic near-balanced fallback.
    """
    x = samples.reshape(-1, samples.shape[-1]).to(dtype=torch.float32)
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    d = int(x.shape[-1])
    cov = (x.t() @ x) / max(1, x.shape[0])
    cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    u = eigvecs[:, order]
    if is_power_of_two(d):
        h = signed_hadamard_matrix(d, seed, str(cov.device), torch.float32)
    else:
        h = random_orthogonal(d, seed, device=cov.device, dtype=torch.float32)
    w = u @ h
    # QR polish absorbs numerical drift from eigh/Hadamard products while
    # preserving the same subspace orientation.
    q, r = torch.linalg.qr(w)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q.contiguous()


def calibrate_hayatequant_spectral(
    codec,
    key_samples: torch.Tensor,
    value_samples: torch.Tensor,
    *,
    learn_codebook: bool = True,
    refit_iters: int = 4,
    max_vectors: int = 500_000,
    holdout_frac: float = 0.1,
    seed: int = 0,
    device: str = None,
    verbose: bool = False,
) -> HayateQuantCalibration:
    """Fit the HQ-SBR artifact: spectral-balanced rotations plus optional
    codebook refits.

    This is deliberately not a family/bit selector.  The same recipe is used
    for both roles: estimate a role covariance, construct the closed-form
    balanced rotation, optionally refit the existing block-VQ codebook in the
    resulting z-space, and store the fixed artifact.  A holdout guard keeps the
    original codec rotation if the spectral rotation fails the role-aware
    objective: angular error for keys and reconstruction MSE for values.
    """
    if codec._n_outlier_for("k") or codec._n_outlier_for("v"):
        raise ValueError("spectral calibration currently requires n_outlier=0")
    from .codec import _block_codebook  # local import: avoid cycle at module load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cal = HayateQuantCalibration(head_dim=int(key_samples.shape[-1]))
    cal.meta = {
        "method": "spectral_balanced_rotation",
        "learn_codebook": learn_codebook,
        "refit_iters": refit_iters,
    }

    for role, samples in (("k", key_samples), ("v", value_samples)):
        bits = codec.bits if role == "k" else codec._value_bits()
        d = int(samples.shape[-1])
        block_size = resolve_block_size(bits, codec.block_size, codec.max_codebook_bits, d)
        x = samples.reshape(-1, d).to(device=device, dtype=torch.float32)
        if x.shape[0] > max_vectors:
            sel = torch.randperm(x.shape[0], generator=gen)[:max_vectors]
            x = x[sel.to(x.device)]
        n_hold = max(1, int(x.shape[0] * holdout_frac))
        x_hold, x_train = x[:n_hold], x[n_hold:]
        if x_train.numel() == 0:
            x_train = x_hold

        w0 = _initial_rotation(codec, d, block_size, bits, device)
        cb0 = _block_codebook(bits, block_size, codec.dist, d, codec.codebook_init).to(device)
        base = _distortion(x_hold, w0, cb0, block_size)
        base_role = _role_error(codec, role, x_hold, w0, cb0, block_size)

        ws = spectral_balanced_rotation(x_train, seed=seed + (0 if role == "k" else 17)).to(device)
        cb = cb0
        if learn_codebook:
            cb = _lloyd_refit(_z_blocks(x_train, ws, block_size), cb, iters=refit_iters)
        dist = _distortion(x_hold, ws, cb, block_size)
        role_err = _role_error(codec, role, x_hold, ws, cb, block_size)

        if role_err <= base_role:
            best_w, best_cb, best_dist, best_role, accepted = ws, cb, dist, role_err, True
        else:
            best_w, best_cb, best_dist, best_role, accepted = None, None, base, base_role, False
        if verbose:
            print(f"[sbr:{role}] holdout role_err {role_err:.6f} "
                  f"(base {base_role:.6f}); z_dist {dist:.6f} "
                  f"(base {base:.6f}, accepted={accepted})", flush=True)

        if accepted:
            cal.rot[role] = best_w.detach().cpu()
        if accepted and learn_codebook:
            cal.codebook[role] = best_cb.detach().cpu()
            cal.cb_key[role] = (bits, block_size)
        cal.meta[f"{role}_dist_before"] = base
        cal.meta[f"{role}_dist_spectral"] = dist
        cal.meta[f"{role}_dist_after"] = best_dist
        cal.meta[f"{role}_role_error_before"] = base_role
        cal.meta[f"{role}_role_error_spectral"] = role_err
        cal.meta[f"{role}_role_error_after"] = best_role
        cal.meta[f"{role}_accepted_spectral"] = accepted
    return cal


def calibrate_hayatequant_attention_weighted(
    codec,
    key_samples: torch.Tensor,
    query_samples: torch.Tensor,
    *,
    steps: int = 64,
    lr: float = 5e-4,
    batch_vectors: int = 8192,
    max_vectors: int = 200_000,
    holdout_frac: float = 0.1,
    seed: int = 0,
    device: str = None,
    verbose: bool = False,
) -> HayateQuantCalibration:
    """Fit a key-only HQ-AW artifact using query-weighted logit error.

    Frontier's bit allocation, value path, scale modes and payload format are
    left unchanged.  Only the key rotation is updated, and only if it improves
    the held-out proxy for attention-logit error,

        E[(k_hat - k)^T C_q (k_hat - k)],   C_q = E[q q^T].

    This targets what the attention score consumes while keeping inference to
    the same calibrated single-GEMM rotation path.
    """
    if codec._n_outlier_for("k") or codec._n_outlier_for("v"):
        raise ValueError("attention-weighted calibration currently requires n_outlier=0")
    from .codec import _block_codebook  # local import: avoid cycle at module load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = int(key_samples.shape[-1])
    bits = codec.bits
    block_size = resolve_block_size(bits, codec.block_size, codec.max_codebook_bits, d)
    x = key_samples.reshape(-1, d).to(device=device, dtype=torch.float32)
    if x.shape[0] > max_vectors:
        sel = torch.randperm(x.shape[0], generator=gen)[:max_vectors]
        x = x[sel.to(x.device)]
    n_hold = max(1, int(x.shape[0] * holdout_frac))
    x_hold, x_train = x[:n_hold], x[n_hold:]
    if x_train.numel() == 0:
        x_train = x_hold

    q = query_samples.reshape(-1, d).to(device=device, dtype=torch.float32)
    if q.shape[0] > max_vectors:
        sel = torch.randperm(q.shape[0], generator=gen)[:max_vectors]
        q = q[sel.to(q.device)]
    qcov = (q.t() @ q) / max(1, q.shape[0])
    trace = torch.trace(qcov).clamp_min(1e-8)
    qcov = (qcov * (d / trace)).contiguous()

    w0 = _initial_rotation(codec, d, block_size, bits, device)
    cb = _block_codebook(bits, block_size, codec.dist, d, codec.codebook_init).to(device)
    base = _attention_weighted_key_error(codec, x_hold, w0, cb, block_size, qcov)
    best = {"w": None, "loss": base, "accepted": False}

    b = torch.zeros(d, d, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for step in range(steps):
        idx = torch.randint(0, x_train.shape[0], (min(batch_vectors, x_train.shape[0]),),
                            generator=gen).to(x_train.device)
        w = w0 @ _cayley(b)
        xhat = _reconstruct_role(codec, "k", x_train[idx], w, cb, block_size)
        err = xhat - x_train[idx]
        loss = ((err @ qcov) * err).sum(dim=-1).mean() / d
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == steps - 1 or step % max(1, steps // 4) == max(1, steps // 4) - 1:
            with torch.no_grad():
                w_eval = w0 @ _cayley(b)
                hold = _attention_weighted_key_error(codec, x_hold, w_eval, cb, block_size, qcov)
                if hold < best["loss"]:
                    best = {"w": w_eval.clone(), "loss": hold, "accepted": True}
                if verbose:
                    print(f"[aw:k] step {step + 1}/{steps} holdout {hold:.6f} "
                          f"(base {base:.6f}, accepted={best['accepted']})", flush=True)

    cal = HayateQuantCalibration(head_dim=d)
    cal.meta = {
        "method": "attention_weighted_key_rotation",
        "steps": steps,
        "lr": lr,
        "k_aw_before": base,
        "k_aw_after": best["loss"],
        "k_accepted": best["accepted"],
    }
    if best["accepted"]:
        cal.rot["k"] = best["w"].detach().cpu()
    return cal


@torch.no_grad()
def _pair_logit_error(codec, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor,
                      w: torch.Tensor, cb: torch.Tensor, block_size: int) -> float:
    khat = _reconstruct_role(codec, "k", k, w, cb, block_size)
    err = ((q * (khat - k)).sum(dim=-1).pow(2) / k.shape[-1])
    weights = weights.to(device=err.device, dtype=err.dtype)
    return float((err * weights).sum() / weights.sum().clamp_min(1e-8))


def calibrate_hayatequant_pair_weighted(
    codec,
    key_pair_samples: torch.Tensor,
    query_pair_samples: torch.Tensor,
    pair_weights: torch.Tensor = None,
    *,
    steps: int = 48,
    lr: float = 2e-4,
    batch_pairs: int = 8192,
    max_pairs: int = 200_000,
    holdout_frac: float = 0.1,
    mse_lambda: float = 0.02,
    trust_lambda: float = 1e-3,
    seed: int = 0,
    device: str = None,
    verbose: bool = False,
) -> HayateQuantCalibration:
    """Fit a key rotation from actual high-attention query/key pairs.

    Unlike the marginal query-covariance objective, this optimises the score
    perturbation consumed by attention for sampled pairs,

        (q_t^T (k_hat_i - k_i))^2 / d.

    The learned artifact is still just a model-level key rotation, so runtime
    is the same calibrated HayateQuant path as Frontier.
    """
    if codec._n_outlier_for("k") or codec._n_outlier_for("v"):
        raise ValueError("pair-weighted calibration currently requires n_outlier=0")
    from .codec import _block_codebook  # local import: avoid cycle at module load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = int(key_pair_samples.shape[-1])
    bits = codec.bits
    block_size = resolve_block_size(bits, codec.block_size, codec.max_codebook_bits, d)
    k = key_pair_samples.reshape(-1, d).to(device=device, dtype=torch.float32)
    q = query_pair_samples.reshape(-1, d).to(device=device, dtype=torch.float32)
    if k.shape != q.shape:
        raise ValueError(f"paired key/query shapes must match, got {k.shape} and {q.shape}")
    if pair_weights is None:
        weights = torch.ones(k.shape[0], dtype=torch.float32)
    else:
        weights = pair_weights.reshape(-1).to(dtype=torch.float32)
    if weights.numel() != k.shape[0]:
        raise ValueError("pair_weights must have one entry per key/query pair")
    if k.shape[0] > max_pairs:
        sel = torch.randperm(k.shape[0], generator=gen)[:max_pairs]
        k = k[sel.to(k.device)]
        q = q[sel.to(q.device)]
        weights = weights[sel]
    weights = weights.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    # Normalise queries so calibration preserves logit direction rather than
    # overfitting a few unusually large activations from the calibration prompt.
    q = torch.nn.functional.normalize(q, dim=-1) * math.sqrt(d)

    n_hold = max(1, int(k.shape[0] * holdout_frac))
    k_hold, k_train = k[:n_hold], k[n_hold:]
    q_hold, q_train = q[:n_hold], q[n_hold:]
    w_hold, w_train = weights[:n_hold], weights[n_hold:]
    if k_train.numel() == 0:
        k_train, q_train, w_train = k_hold, q_hold, w_hold

    w0 = _initial_rotation(codec, d, block_size, bits, device)
    cb = _block_codebook(bits, block_size, codec.dist, d, codec.codebook_init).to(device)
    base = _pair_logit_error(codec, q_hold, k_hold, w_hold, w0, cb, block_size)
    best = {"w": None, "loss": base, "accepted": False}

    b = torch.zeros(d, d, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for step in range(steps):
        idx = torch.randint(0, k_train.shape[0], (min(batch_pairs, k_train.shape[0]),),
                            generator=gen).to(k_train.device)
        w = w0 @ _cayley(b)
        khat = _reconstruct_role(codec, "k", k_train[idx], w, cb, block_size)
        err = khat - k_train[idx]
        pair = (q_train[idx] * err).sum(dim=-1).pow(2) / d
        wb = w_train[idx].to(device=pair.device, dtype=pair.dtype)
        loss = (pair * wb).sum() / wb.sum().clamp_min(1e-8)
        if mse_lambda:
            loss = loss + mse_lambda * err.pow(2).mean()
        if trust_lambda:
            loss = loss + trust_lambda * (w - w0).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == steps - 1 or step % max(1, steps // 4) == max(1, steps // 4) - 1:
            with torch.no_grad():
                w_eval = w0 @ _cayley(b)
                hold = _pair_logit_error(codec, q_hold, k_hold, w_hold, w_eval, cb, block_size)
                if hold < best["loss"]:
                    best = {"w": w_eval.clone(), "loss": hold, "accepted": True}
                if verbose:
                    print(f"[pair:k] step {step + 1}/{steps} holdout {hold:.6f} "
                          f"(base {base:.6f}, accepted={best['accepted']})", flush=True)

    cal = HayateQuantCalibration(head_dim=d)
    cal.meta = {
        "method": "attention_pair_key_rotation",
        "steps": steps,
        "lr": lr,
        "mse_lambda": mse_lambda,
        "trust_lambda": trust_lambda,
        "k_pair_before": base,
        "k_pair_after": best["loss"],
        "k_accepted": best["accepted"],
    }
    if best["accepted"]:
        cal.rot["k"] = best["w"].detach().cpu()
    return cal


def calibrate_hayatequant_layer_pair_weighted(
    codec,
    layer_pairs: dict,
    *,
    steps: int = 32,
    lr: float = 1e-4,
    batch_pairs: int = 8192,
    max_pairs_per_layer: int = 50_000,
    holdout_frac: float = 0.1,
    mse_lambda: float = 0.05,
    trust_lambda: float = 1e-2,
    seed: int = 0,
    device: str = None,
    verbose: bool = False,
) -> HayateQuantCalibration:
    """Layer-aware attention-pair key calibration.

    The recipe is still a single codec: every layer receives the same
    attention-pair objective and the same frontier K/V bit allocation.  The
    only extra offline side information is one key rotation per layer.
    """
    if not layer_pairs:
        raise ValueError("layer_pairs is empty")
    first = next(iter(layer_pairs.values()))
    d = int(first[1].shape[-1])
    cal = HayateQuantCalibration(head_dim=d)
    cal.layer_rot["k"] = {}
    cal.meta = {
        "method": "layer_attention_pair_key_rotation",
        "steps": steps,
        "lr": lr,
        "mse_lambda": mse_lambda,
        "trust_lambda": trust_lambda,
        "layers": sorted(int(x) for x in layer_pairs),
    }
    before, after, accepted = {}, {}, {}
    for off, layer_idx in enumerate(sorted(layer_pairs)):
        q, k, w = layer_pairs[layer_idx]
        sub = calibrate_hayatequant_pair_weighted(
            codec,
            k,
            q,
            w,
            steps=steps,
            lr=lr,
            batch_pairs=batch_pairs,
            max_pairs=max_pairs_per_layer,
            holdout_frac=holdout_frac,
            mse_lambda=mse_lambda,
            trust_lambda=trust_lambda,
            seed=seed + off,
            device=device,
            verbose=verbose,
        )
        before[int(layer_idx)] = sub.meta["k_pair_before"]
        after[int(layer_idx)] = sub.meta["k_pair_after"]
        accepted[int(layer_idx)] = bool(sub.meta["k_accepted"])
        if sub.meta["k_accepted"] and "k" in sub.rot:
            cal.layer_rot["k"][int(layer_idx)] = sub.rot["k"]
        if verbose:
            print(f"[layer-pair:k] layer {layer_idx} before {before[int(layer_idx)]:.6f} "
                  f"after {after[int(layer_idx)]:.6f} accepted={accepted[int(layer_idx)]}",
                  flush=True)
    if not cal.layer_rot["k"]:
        cal.layer_rot.pop("k", None)
    cal.meta["k_pair_before"] = before
    cal.meta["k_pair_after"] = after
    cal.meta["k_accepted"] = accepted
    cal.meta["n_layers_accepted"] = len(cal.layer_rot.get("k", {}))
    return cal


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
        codec: The HayateQuantCodec whose configuration (bits / block size /
            rotation / dist) defines the starting point and the objective.
            Must have ``n_outlier == 0`` (the outlier split changes subset
            dims per tensor; calibrating it is future work).
        key_samples / value_samples: ``(N, head_dim)`` real K / V vectors
            (any dtype/device; flattened from ``(b, h, s, d)`` captures).
        learn_rotation: Optimise the per-role rotation (Cayley/Adam).
        learn_codebook: Alternate closed-form Lloyd refits of the codebook on
            real z-blocks.
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
    if codec._n_outlier_for("k") or codec._n_outlier_for("v"):
        raise ValueError("calibration currently requires n_outlier=0")
    if not (learn_rotation or learn_codebook):
        raise ValueError("nothing to learn: enable learn_rotation and/or learn_codebook")
    from .codec import _block_codebook  # local import: avoid cycle at module load

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cal = HayateQuantCalibration(head_dim=int(key_samples.shape[-1]))
    cal.meta = {"steps": steps, "lr": lr, "learn_rotation": learn_rotation,
                "learn_codebook": learn_codebook}

    for role, samples in (("k", key_samples), ("v", value_samples)):
        bits = codec.bits if role == "k" else codec._value_bits()
        d = int(samples.shape[-1])
        block_size = resolve_block_size(bits, codec.block_size, codec.max_codebook_bits, d)
        x = samples.reshape(-1, d).to(device=device, dtype=torch.float32)
        if x.shape[0] > max_vectors:
            sel = torch.randperm(x.shape[0], generator=gen)[:max_vectors]
            x = x[sel.to(x.device)]
        n_hold = max(1, int(x.shape[0] * holdout_frac))
        x_hold, x_train = x[:n_hold], x[n_hold:]

        w0 = _initial_rotation(codec, d, block_size, bits, device)
        cb = _block_codebook(bits, block_size, codec.dist, d, codec.codebook_init).to(device)
        base = _distortion(x_hold, w0, cb, block_size)
        best = {"w": None, "cb": None, "dist": base, "accepted": False}

        b = torch.zeros(d, d, device=device, requires_grad=learn_rotation)
        opt = torch.optim.Adam([b], lr=lr) if learn_rotation else None
        n_steps = steps if learn_rotation else (steps // refit_every + 1)
        for step in range(n_steps):
            if learn_rotation:
                idx = torch.randint(0, x_train.shape[0], (min(batch_vectors, x_train.shape[0]),),
                                    generator=gen).to(x_train.device)
                w = w0 @ _cayley(b)
                zb = _z_blocks(x_train[idx], w, block_size)
                with torch.no_grad():
                    codes = quantize_to_vq_codebook(zb, cb)
                loss = (zb - lookup_codewords(cb, codes)).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            do_refit = learn_codebook and (not learn_rotation or step % refit_every == refit_every - 1)
            if do_refit or step == n_steps - 1:
                with torch.no_grad():
                    w = w0 @ _cayley(b) if learn_rotation else w0
                    if learn_codebook:
                        cb = _lloyd_refit(_z_blocks(x_train, w, block_size), cb)
                    dist = _distortion(x_hold, w, cb, block_size)
                    if dist < best["dist"]:
                        best = {"w": w.clone(), "cb": cb.clone(),
                                "dist": dist, "accepted": True}
                    if verbose:
                        print(f"[calib:{role}] step {step + 1}/{n_steps} "
                              f"holdout dist {dist:.6f} (base {base:.6f})", flush=True)

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
def capture_kv_samples(model, tokenizer, texts, *, max_tokens: int = 2048,
                       layer_stride: int = 4, device=None):
    """Capture real ``(N, head_dim)`` K / V samples from ``model`` on
    ``texts`` (list of strings), mixing every ``layer_stride``-th layer so
    one shared per-role rotation serves all layers (the codec, like
    SpinQuant's shared R1, uses one rotation model-wide)."""
    ks, vs = [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).input_ids
        ids = ids.to(device or model.device)
        out = model(ids, use_cache=True)
        layers = out.past_key_values.layers
        for i in range(0, len(layers), layer_stride):
            ks.append(layers[i].keys.float().reshape(-1, layers[i].keys.shape[-1]).cpu())
            vs.append(layers[i].values.float().reshape(-1, layers[i].values.shape[-1]).cpu())
    return torch.cat(ks), torch.cat(vs)


@torch.no_grad()
def capture_qkv_samples(model, tokenizer, texts, *, max_tokens: int = 2048,
                        layer_stride: int = 4, device=None):
    """Capture query samples together with the post-RoPE KV cache samples.

    Queries are collected after RoPE inside ``self_attn`` for the same layer
    stride used for KV.  This intentionally avoids requesting attention
    weights from FlashAttention/eager attention; the offline objective only
    needs the query covariance ``E[q q^T]`` in the same coordinate system as
    the cached keys.
    """
    attn_modules = {}
    for name, mod in model.named_modules():
        if not name.endswith(".self_attn"):
            continue
        parts = name.split(".")
        try:
            li = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if li % layer_stride == 0:
            attn_modules[li] = mod
    if not attn_modules:
        raise RuntimeError("could not locate strided self_attn modules")

    qs, ks, vs = [], [], []
    current_q = []

    originals = []

    def _wrap(mod):
        orig = mod.forward
        apply_rope = orig.__func__.__globals__.get("apply_rotary_pos_emb")

        def wrapped(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            if apply_rope is not None and position_embeddings is not None:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, mod.head_dim)
                q = mod.q_proj(hidden_states).view(hidden_shape)
                k = mod.k_proj(hidden_states).view(hidden_shape)
                if hasattr(mod, "q_norm"):
                    q = mod.q_norm(q)
                if hasattr(mod, "k_norm"):
                    k = mod.k_norm(k)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                cos, sin = position_embeddings
                q, _ = apply_rope(q, k, cos, sin)
                current_q.append(q.detach().float().cpu())
            return orig(hidden_states, position_embeddings, attention_mask,
                        past_key_values, **kwargs)

        originals.append((mod, orig))
        mod.forward = wrapped

    for _, mod in sorted(attn_modules.items()):
        _wrap(mod)
    try:
        for text in texts:
            current_q.clear()
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_tokens).input_ids
            ids = ids.to(device or model.device)
            out = model(ids, use_cache=True)
            layers = out.past_key_values.layers
            head_dim = int(layers[0].keys.shape[-1])
            for q in current_q:
                if q.shape[-1] % head_dim == 0:
                    qs.append(q.reshape(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
                              .reshape(-1, head_dim))
            for i in range(0, len(layers), layer_stride):
                ks.append(layers[i].keys.float().reshape(-1, head_dim).cpu())
                vs.append(layers[i].values.float().reshape(-1, head_dim).cpu())
    finally:
        for mod, orig in originals:
            mod.forward = orig
    if not qs:
        raise RuntimeError("captured no query vectors")
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


@torch.no_grad()
def capture_attention_pair_samples_by_layer(
    model,
    tokenizer,
    texts,
    *,
    max_tokens: int = 1024,
    layer_stride: int = 4,
    topk: int = 4,
    query_stride: int = 8,
    max_pairs: int = 200_000,
    seed: int = 0,
    device=None,
):
    """Capture high-attention post-RoPE ``(q_t, k_i)`` pairs by layer.

    This is an offline calibration helper.  For every selected attention
    layer, it reconstructs post-RoPE queries and keys inside ``self_attn`` and
    keeps the top causal key positions for strided query positions.  Grouped
    query attention is handled by repeating each KV head to its query-head
    group only for the offline score computation.
    """
    attn_modules = {}
    for name, mod in model.named_modules():
        if not name.endswith(".self_attn"):
            continue
        parts = name.split(".")
        try:
            li = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if li % layer_stride == 0:
            attn_modules[li] = mod
    if not attn_modules:
        raise RuntimeError("could not locate strided self_attn modules")

    layer_pairs = {}
    originals = []

    def _wrap(layer_idx, mod):
        orig = mod.forward
        apply_rope = orig.__func__.__globals__.get("apply_rotary_pos_emb")

        def wrapped(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            if apply_rope is not None and position_embeddings is not None:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, mod.head_dim)
                q = mod.q_proj(hidden_states).view(hidden_shape)
                k = mod.k_proj(hidden_states).view(hidden_shape)
                if hasattr(mod, "q_norm"):
                    q = mod.q_norm(q)
                if hasattr(mod, "k_norm"):
                    k = mod.k_norm(k)
                q = q.transpose(1, 2).float()
                k = k.transpose(1, 2).float()
                cos, sin = position_embeddings
                q, k = apply_rope(q, k, cos, sin)

                bsz, hq, seq, d = q.shape
                hkv = k.shape[1]
                group = max(1, hq // hkv)
                k_full = k.repeat_interleave(group, dim=1)[:, :hq]
                if seq > 1:
                    q_pos = torch.arange(0, seq, max(1, query_stride), device=q.device)
                    q_sel = q[:, :, q_pos, :]
                    scores = torch.einsum("bhqd,bhkd->bhqk", q_sel, k_full) / math.sqrt(d)
                    causal = torch.arange(seq, device=q.device).view(1, 1, 1, seq) <= q_pos.view(1, 1, -1, 1)
                    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
                    k_top = min(topk, seq)
                    vals, inds = torch.topk(scores, k=k_top, dim=-1)
                    probs = torch.softmax(scores.float(), dim=-1).gather(-1, inds).clamp_min(1e-8)
                    k_src = k_full.unsqueeze(2).expand(bsz, hq, q_pos.numel(), seq, d)
                    gathered = torch.gather(k_src, 3, inds.unsqueeze(-1).expand(-1, -1, -1, -1, d))
                    q_rep = q_sel.unsqueeze(3).expand_as(gathered)
                    bucket = layer_pairs.setdefault(int(layer_idx), [[], [], []])
                    bucket[0].append(q_rep.reshape(-1, d).detach().cpu())
                    bucket[1].append(gathered.reshape(-1, d).detach().cpu())
                    # Retain the softmax mass of the sampled pair; add a tiny
                    # rank signal so tied early tokens are stable.
                    rank_scale = torch.linspace(1.0, 0.5, k_top, device=probs.device)
                    bucket[2].append((probs * rank_scale.view(1, 1, 1, -1)).reshape(-1).cpu())
            return orig(hidden_states, position_embeddings, attention_mask,
                        past_key_values, **kwargs)

        originals.append((mod, orig))
        mod.forward = wrapped

    for li, mod in sorted(attn_modules.items()):
        _wrap(li, mod)
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_tokens).input_ids
            ids = ids.to(device or model.device)
            model(ids, use_cache=True)
    finally:
        for mod, orig in originals:
            mod.forward = orig
    if not layer_pairs:
        raise RuntimeError("captured no attention pairs")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for li, (qs, ks, ws) in layer_pairs.items():
        q = torch.cat(qs)
        k = torch.cat(ks)
        w = torch.cat(ws)
        if q.shape[0] > max_pairs:
            sel = torch.randperm(q.shape[0], generator=gen)[:max_pairs]
            q, k, w = q[sel], k[sel], w[sel]
        out[int(li)] = (q, k, w)
    return out


@torch.no_grad()
def capture_attention_pair_samples(*args, **kwargs):
    """Capture high-attention post-RoPE pairs and concatenate layers."""
    layer_pairs = capture_attention_pair_samples_by_layer(*args, **kwargs)
    qs, ks, ws = [], [], []
    for li in sorted(layer_pairs):
        q, k, w = layer_pairs[li]
        qs.append(q)
        ks.append(k)
        ws.append(w)
    return torch.cat(qs), torch.cat(ks), torch.cat(ws)
