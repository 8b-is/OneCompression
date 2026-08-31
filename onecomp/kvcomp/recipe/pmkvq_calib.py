"""PM-KVQ offline calibration: block-wise IP allocation + channel reparameterization.

Implements the two offline pre-inference steps of PM-KVQ (Liu et al.,
arXiv:2505.18610) that the online :class:`~onecomp.kvcomp.cache.pmkvq_cache.PMKVQCache`
schedule alone does not capture:

1. **Block-wise memory allocation** (paper §3.2, Eq. 4-8).
   Estimate each transformer block's sensitivity to ``b``-bit KV quantization
   with a first-order Taylor approximation

       s_{i,b} = || G_{K_i} ⊙ (K_i − Q_b(K_i)) ||_1 + || G_{V_i} ⊙ (V_i − Q_b(V_i)) ||_1

   (``G`` = gradient of the LM loss w.r.t. the key/value projection), then solve
   the integer program

       min_x  Σ_i Σ_b x_{i,b} s_{i,b}
       s.t.   Σ_b x_{i,b} = 1 ∀ i,   Σ_{i,b} x_{i,b}·mem(b) ≤ M,   x_{i,b} ∈ {0,1}

   assigning a higher final bit-width ``Fbit`` to more sensitive blocks under a
   fixed average-bit budget.  Solved with :func:`scipy.optimize.milp` (cvxpy/pulp
   are not installed; scipy's HiGHS MILP backend is).

2. **Calibration with positional interpolation** (paper §3.3, Eq. 9-12).
   Reduce key quantization error by a per-channel reparameterization
   ``λ_i = (max_m |K_{m,i}|)^α`` that migrates outliers out of the quantized key
   (QServe-style), calibrated on short prompts whose RoPE positions are scaled by
   ``pos_scale`` (Eq. 12: ``s·m·θ_i``) so the low-frequency channel periodicity of
   long contexts is captured without long calibration sequences.  ``α`` is grid-
   searched per layer to minimise the key dequantization MSE.

The results attach to a :class:`~onecomp.kvcomp.recipe.pmkvq.PMKVQPolicy`
(``block_final_bits`` and ``key_channel_scale``); the cache then applies them.
This is the faithful counterpart to the policy's manual ``sensitive_layers`` knob.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

from logging import getLogger
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..codec._quant_primitives import fake_quantize_affine

logger = getLogger(__name__)

# A few reasoning-style calibration prompts (the paper uses RedPajama-arXiv; for
# a relative per-block sensitivity ranking a small reasoning set is sufficient
# and keeps calibration fast -- see tasks/lessons.md K9).
_DEFAULT_CALIB_PROMPTS = (
    "Natalia sold clips to 48 friends in April, then sold half as many in May. "
    "How many clips did she sell altogether? Reason step by step.",
    "A rectangle has length 3 more than twice its width. Its perimeter is 36. "
    "Find its area. Reason step by step.",
    "Compute the sum of all positive integers below 50 divisible by 3 or 5. " "Show your work.",
    "If 5x + 3 = 2x + 18, solve for x and verify the solution.",
)


def _attn_modules(model) -> List[Tuple[object, object]]:
    """Return ``[(k_proj, v_proj), ...]`` per decoder layer, in layer order.

    Robust to the common HF naming ``model.model.layers[i].self_attn.{k,v}_proj``
    used by Llama / Qwen2 / DeepSeek-distill architectures.
    """
    pairs = []
    by_layer: Dict[int, dict] = {}
    for name, mod in model.named_modules():
        # match '...layers.<idx>.self_attn.k_proj' / 'v_proj'
        if ".self_attn." not in name:
            continue
        if name.endswith(".k_proj") or name.endswith(".v_proj"):
            parts = name.split(".")
            try:
                li = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            slot = by_layer.setdefault(li, {})
            slot["k" if name.endswith(".k_proj") else "v"] = mod
    for li in sorted(by_layer):
        slot = by_layer[li]
        if "k" in slot and "v" in slot:
            pairs.append((slot["k"], slot["v"]))
    return pairs


def _ids(tokenizer, prompt: str, device) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"].to(device)
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def estimate_block_sensitivity(
    model,
    tokenizer,
    calib_prompts: Optional[Sequence[str]] = None,
    bit_options: Sequence[int] = (2, 4, 8),
    group_size: int = 128,
    skip_layers: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """First-order Taylor sensitivity ``s[layer, bit]`` (paper Eq. 5).

    Returns a ``(n_layers, len(bit_options))`` tensor on CPU.  Skipped (outlier)
    layers get ``+inf`` for every bit < 16 so the IP never lowers their precision.
    """
    calib_prompts = list(calib_prompts or _DEFAULT_CALIB_PROMPTS)
    skip = set(skip_layers or [])
    pairs = _attn_modules(model)
    n_layers = len(pairs)
    if n_layers == 0:
        raise RuntimeError("Could not locate k_proj/v_proj modules for sensitivity.")
    bit_options = list(bit_options)
    sens = torch.zeros(n_layers, len(bit_options), dtype=torch.float64)

    captured: Dict[int, dict] = {}

    def _mk_hook(li: int, which: str):
        def hook(_module, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            t.retain_grad()
            captured.setdefault(li, {})[which] = t

        return hook

    handles = []
    for li, (kp, vp) in enumerate(pairs):
        handles.append(kp.register_forward_hook(_mk_hook(li, "k")))
        handles.append(vp.register_forward_hook(_mk_hook(li, "v")))

    try:
        was_training = model.training
        model.eval()
        for prompt in calib_prompts:
            captured.clear()
            ids = _ids(tokenizer, prompt, model.device)
            model.zero_grad(set_to_none=True)
            out = model(ids, labels=ids, use_cache=False)
            loss = out.loss
            loss.backward()
            for li in range(n_layers):
                if li in skip:
                    sens[li, :] = float("inf")
                    continue
                slot = captured.get(li, {})
                k, v = slot.get("k"), slot.get("v")
                if k is None or v is None or k.grad is None or v.grad is None:
                    continue
                gk, gv = k.grad.detach().float(), v.grad.detach().float()
                kf, vf = k.detach().float(), v.detach().float()
                for bi, b in enumerate(bit_options):
                    if b >= 16:
                        continue
                    qk = fake_quantize_affine(kf, b, axis="token", group_size=group_size)
                    qv = fake_quantize_affine(vf, b, axis="token", group_size=group_size)
                    sk = (gk * (kf - qk)).abs().sum()
                    sv = (gv * (vf - qv)).abs().sum()
                    sens[li, bi] += float(sk + sv)
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    # Average over prompts (skipped layers stay +inf).
    finite = torch.isfinite(sens)
    sens[finite] /= max(1, len(calib_prompts))
    return sens


def solve_block_allocation(
    sensitivity: torch.Tensor,
    bit_options: Sequence[int],
    avg_bit_budget: float,
) -> Dict[int, int]:
    """Integer-program per-layer bit assignment (paper Eq. 6-8).

    Args:
        sensitivity: ``(n_layers, n_bits)`` from :func:`estimate_block_sensitivity`.
        bit_options: candidate final bit-widths (ascending), aligned with columns.
        avg_bit_budget: target *average* final bit-width across the *quantizable*
            (non-skipped) layers; the memory budget is ``avg_bit_budget * |free|``.
            Skipped (outlier) layers are kept in full precision and accounted
            separately, so they do NOT consume this budget.

    Returns:
        ``{layer_idx: final_bits}``.  Falls back to a greedy assignment if SciPy's
        MILP is unavailable or fails to converge.
    """
    n_layers, n_bits = sensitivity.shape
    bits = list(bit_options)

    # Skipped layers (inf for all low bits): pin to the highest option (they are
    # kept in FP anyway by the cache) and exclude them from the quant budget so
    # they never starve the solver.
    skipped = [i for i in range(n_layers) if not torch.isfinite(sensitivity[i]).any()]
    free = [i for i in range(n_layers) if i not in skipped]
    fixed = {i: max(bits) for i in skipped}
    # Budget applies to the quantizable layers only.
    budget_free = avg_bit_budget * len(free)

    if not free:
        return fixed

    s = sensitivity[free].clone()
    # Replace residual infs (a single skipped bit) with a large finite penalty.
    big = float(s[torch.isfinite(s)].max().item() if torch.isfinite(s).any() else 1.0) * 1e6 + 1.0
    s[~torch.isfinite(s)] = big

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        m = len(free)
        nvar = m * n_bits
        c = s.reshape(-1).double().numpy()

        # Each free layer selects exactly one bit: Σ_b x[i,b] = 1.
        A_eq = np.zeros((m, nvar))
        for r in range(m):
            A_eq[r, r * n_bits : (r + 1) * n_bits] = 1.0
        eq = LinearConstraint(A_eq, lb=np.ones(m), ub=np.ones(m))

        # Memory budget: Σ_{i,b} x[i,b]·bits[b] ≤ budget_free.
        mem = np.tile(np.array(bits, dtype=float), m)
        ub = LinearConstraint(mem.reshape(1, -1), lb=-np.inf, ub=float(budget_free))

        res = milp(
            c=c,
            constraints=[eq, ub],
            integrality=np.ones(nvar),
            bounds=Bounds(0, 1),
        )
        if res.success and res.x is not None:
            x = np.round(res.x).reshape(m, n_bits)
            alloc = dict(fixed)
            for r, li in enumerate(free):
                bi = int(x[r].argmax())
                alloc[li] = bits[bi]
            return alloc
        logger.warning("PM-KVQ block IP did not converge (%s); using greedy.", res.message)
    except Exception as e:  # pragma: no cover - scipy missing / solver error
        logger.warning("PM-KVQ block IP unavailable (%s); using greedy.", e)

    # Greedy fallback: start everyone at the lowest bit, then spend the remaining
    # budget on the layers with the largest sensitivity drop per extra bit.
    alloc = {i: bits[0] for i in free}
    alloc.update(fixed)
    spent = sum(alloc[i] for i in free)
    while True:
        best, best_gain = None, 0.0
        for i in free:
            cur = bits.index(alloc[i])
            if cur + 1 >= n_bits:
                continue
            extra = bits[cur + 1] - bits[cur]
            if spent + extra > budget_free:
                continue
            gain = float(s[free.index(i), cur] - s[free.index(i), cur + 1]) / max(1, extra)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        cur = bits.index(alloc[best])
        spent += bits[cur + 1] - bits[cur]
        alloc[best] = bits[cur + 1]
    return alloc


@torch.no_grad()
def calibrate_channel_reparam(
    model,
    tokenizer,
    calib_prompts: Optional[Sequence[str]] = None,
    final_bits_map: Optional[Dict[int, int]] = None,
    default_key_bits: int = 4,
    pos_scale: int = 4,
    alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    skip_layers: Optional[Sequence[int]] = None,
    group_size: int = 128,
) -> Dict[int, torch.Tensor]:
    """Per-channel key reparameterization factors ``λ`` (paper §3.3).

    Captures post-RoPE keys with RoPE positions scaled by ``pos_scale`` (Eq. 12
    positional interpolation), then per layer grid-searches ``α`` to minimise the
    key dequantization MSE under ``λ_i = (max_m |K_{m,i}|)^α``.

    Returns ``{layer_idx: λ}`` with ``λ`` of shape ``(n_kv_heads, head_dim)`` on
    the model device; layers in ``skip_layers`` are omitted (kept in FP).
    """
    calib_prompts = list(calib_prompts or _DEFAULT_CALIB_PROMPTS)
    skip = set(skip_layers or [])
    final_bits_map = final_bits_map or {}

    # Accumulate per-channel max|K| and the captured keys for the MSE search.
    chan_max: Dict[int, torch.Tensor] = {}
    key_samples: Dict[int, List[torch.Tensor]] = {}

    for prompt in calib_prompts:
        ids = _ids(tokenizer, prompt, model.device)
        seq = ids.shape[1]
        # Positional interpolation: scale positions by pos_scale so RoPE sees an
        # effective context pos_scale x longer (Eq. 12), capturing low-frequency
        # channel periodicity without long sequences.
        pos = (torch.arange(seq, device=model.device) * max(1, int(pos_scale))).unsqueeze(0)
        out = model(ids, position_ids=pos, use_cache=True)
        layers = out.past_key_values.layers
        for li, layer in enumerate(layers):
            if li in skip:
                continue
            k = layer.keys.detach().float()[0]  # (n_kv_heads, seq, head_dim)
            amax = k.abs().amax(dim=1)  # (n_kv_heads, head_dim)
            chan_max[li] = torch.maximum(chan_max[li], amax) if li in chan_max else amax
            # Keep a small sample of keys for the alpha MSE search (cap memory).
            key_samples.setdefault(li, [])
            if len(key_samples[li]) < 2:
                key_samples[li].append(k[:, : min(seq, 512), :].clone())

    scales: Dict[int, torch.Tensor] = {}
    for li, amax in chan_max.items():
        # Calibrate at the layer's actual KEY floor used by the cache:
        # max(block value floor, key_final_bits) -- keys never go below the key
        # floor (MKLV K>V), so the reparam must target that bit-width.
        kb = max(final_bits_map.get(li, default_key_bits), default_key_bits)
        kb = max(2, min(kb, 8))
        ks = torch.cat(key_samples[li], dim=1)  # (n_kv_heads, S, head_dim)
        best_alpha, best_mse, best_lam = 0.0, None, None
        base = amax.clamp_min(1e-6)
        for a in alpha_grid:
            lam = base.pow(a).clamp_min(1e-6)  # (n_kv_heads, head_dim)
            inv = (1.0 / lam).unsqueeze(1)  # (n_kv_heads,1,head_dim)
            recon = lam.unsqueeze(1) * fake_quantize_affine(
                ks * inv, kb, axis="token", group_size=group_size
            )
            mse = float((recon - ks).pow(2).mean())
            if best_mse is None or mse < best_mse:
                best_alpha, best_mse, best_lam = a, mse, lam
        scales[li] = best_lam.to(model.device)
        logger.debug("PM-KVQ reparam layer %d: alpha=%.2f mse=%.4g", li, best_alpha, best_mse)
    return scales


def calibrate_pmkvq(
    model,
    tokenizer,
    policy,
    *,
    block_allocation: bool = True,
    channel_reparam: bool = True,
    bit_options: Sequence[int] = (2, 4, 8),
    avg_bit_budget: Optional[float] = None,
    pos_scale: int = 4,
    calib_prompts: Optional[Sequence[str]] = None,
    group_size: int = 128,
):
    """Calibrate a :class:`PMKVQPolicy` in place (block IP + channel reparam).

    Args:
        policy: the :class:`~onecomp.kvcomp.recipe.pmkvq.PMKVQPolicy` to enrich.
        block_allocation: run the Taylor-sensitivity integer program (§3.2).
        channel_reparam: run the positional-interpolation key reparam (§3.3).
        avg_bit_budget: target average *final* bits across layers; defaults to the
            policy's ``final_bits`` (so the IP redistributes the same budget,
            giving sensitive blocks more bits and insensitive blocks fewer).

    Returns:
        The same ``policy`` (mutated) for convenience.
    """
    skip = list(getattr(policy, "skip_layers", []) or [])
    calib_prompts = list(calib_prompts or _DEFAULT_CALIB_PROMPTS)

    if block_allocation:
        budget = avg_bit_budget if avg_bit_budget is not None else float(policy.final_bits)
        sens = estimate_block_sensitivity(
            model,
            tokenizer,
            calib_prompts,
            bit_options=bit_options,
            group_size=group_size,
            skip_layers=skip,
        )
        alloc = solve_block_allocation(sens, bit_options, budget)
        policy.block_final_bits = alloc
        chosen = sorted(set(alloc.values()))
        logger.info(
            "PM-KVQ block allocation: %d layers, bits used=%s, avg=%.2f",
            len(alloc),
            chosen,
            sum(alloc.values()) / max(1, len(alloc)),
        )

    if channel_reparam:
        scales = calibrate_channel_reparam(
            model,
            tokenizer,
            calib_prompts,
            final_bits_map=getattr(policy, "block_final_bits", {}) or {},
            default_key_bits=policy.key_final_bits,
            pos_scale=pos_scale,
            skip_layers=skip,
            group_size=group_size,
        )
        policy.key_channel_scale = scales
        logger.info(
            "PM-KVQ channel reparam: calibrated %d layers (pos_scale=%d)", len(scales), pos_scale
        )

    return policy
