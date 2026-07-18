"""Build a spectral-balanced HayateQuant calibration artifact.

HQ-SBR is a single codec recipe, not a model-family selector:

    K(r+1)V(r-1) + key UB + value LS-iter + spectral-balanced artifact.

The artifact contains fixed per-role rotations and optional refit codebooks.
At inference time it is consumed by ``HayateQuantCodec(calibration=...)`` and
therefore costs the same rotation GEMM as any calibrated HayateQuant run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.kvcomp.codec import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.recipes import hayate_hq_kwargs, hayate_hq_role_rates
from onecomp.kvcomp.codec.hayate.calibration import (
    calibrate_hayatequant_spectral,
    capture_kv_samples,
)

DEFAULT_TEXT = (
    "A careful proof often starts from the quantity that downstream attention "
    "actually consumes, rather than from raw reconstruction error alone. "
    "Long-context language models store key and value vectors for every token; "
    "a robust KV quantizer should spread channel energy evenly before assigning "
    "low-bit codewords. "
) * 32


def _kv_asym(bits: int) -> tuple[float, float]:
    kv_bits = hayate_hq_role_rates(bits)
    if kv_bits is None:
        raise ValueError(f"average bits must be one of 2, 3, or 4, got {bits}")
    return kv_bits


def main() -> None:
    args = parse_args()
    key_bits, value_bits = _kv_asym(args.avg_bits)
    if args.key_bits is not None or args.value_bits is not None:
        raise ValueError(
            "fixed HQ calibration does not accept legacy integer role-bit overrides; "
            "use the canonical K/V rates recorded by hayate_hq_kwargs"
        )

    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    texts = [DEFAULT_TEXT]
    if args.text_file:
        texts = [x.strip() for x in Path(args.text_file).read_text().splitlines() if x.strip()]
    ks, vs = capture_kv_samples(
        model,
        tok,
        texts,
        max_tokens=args.max_tokens,
        layer_stride=args.layer_stride,
        device=model.device,
    )
    if args.block_size not in (None, 2):
        raise ValueError("the fixed HQ calibration recipe requires block_size=2")
    codec = HayateQuantCodec(
        **hayate_hq_kwargs(
            args.avg_bits,
            max_codebook_bits=args.max_codebook_bits,
            encoder="exact",
        )
    )
    cal = calibrate_hayatequant_spectral(
        codec,
        ks,
        vs,
        learn_codebook=not args.no_codebook_refit,
        refit_iters=args.refit_iters,
        max_vectors=args.max_vectors,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        device=args.fit_device,
        verbose=True,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.meta.update(
        {
            "model_path": str(args.model_path),
            "avg_bits": args.avg_bits,
            "key_bits": key_bits,
            "value_bits": value_bits,
            "recipe": "HQ-b2-K(r+0.5)V(r-0.5)+NC-key+LSiter-value+SBR",
        }
    )
    cal.save(str(out))
    print(f"wrote {out}")
    print(cal.meta)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--avg-bits", type=int, default=2)
    ap.add_argument("--key-bits", type=int)
    ap.add_argument("--value-bits", type=int)
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--max-codebook-bits", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--max-vectors", type=int, default=200_000)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--refit-iters", type=int, default=4)
    ap.add_argument("--no-codebook-refit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fit-device", default=None)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--text-file")
    return ap.parse_args()


if __name__ == "__main__":
    main()
