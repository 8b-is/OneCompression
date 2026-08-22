"""Build a Cayley-optimized HayateQuant calibration artifact.

This keeps the same canonical HayateQuant operating recipe at a target
average rate, but learns a fixed per-role orthogonal rotation and optional
codebook refit offline.  Inference uses the existing calibrated HayateQuant
runtime path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from onecomp.kvcomp.codec.hayate.recipes import (  # noqa: E402
    build_hayatequant,
    hayatequant_role_rates,
)
from onecomp.kvcomp.codec.hayate.calibration import (  # noqa: E402
    calibrate_hayatequant,
    capture_kv_samples,
)

DEFAULT_TEXT = (
    "A robust KV-cache quantizer should preserve attention logits and value "
    "reconstruction while keeping the same fused serving path. "
) * 64


def _kv_asym(bits: int) -> tuple[float, float]:
    return hayatequant_role_rates(bits)


def main() -> None:
    args = parse_args()
    key_bits, value_bits = _kv_asym(args.avg_bits)
    tok = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device,
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
    codec = build_hayatequant(args.avg_bits)
    cal = calibrate_hayatequant(
        codec,
        ks,
        vs,
        learn_rotation=not args.no_rotation,
        # A refit keeps exactly the same role cardinalities and packed layout.
        learn_codebook=not args.rotation_only,
        steps=args.steps,
        lr=args.lr,
        batch_vectors=args.batch_vectors,
        refit_every=args.refit_every,
        max_vectors=args.max_vectors,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        device=args.fit_device,
        verbose=True,
    )
    cal.meta.update(
        {
            "model_path": str(args.model_path),
            "avg_bits": args.avg_bits,
            "key_bits": key_bits,
            "value_bits": value_bits,
            "method": "hayatequant",
        }
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.save(str(out))
    print(f"wrote {out}")
    print(cal.meta)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--avg-bits", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--max-vectors", type=int, default=200_000)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-vectors", type=int, default=8192)
    ap.add_argument("--refit-every", type=int, default=16)
    ap.add_argument("--no-rotation", action="store_true")
    ap.add_argument("--rotation-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fit-device", default=None)
    ap.add_argument("--text-file")
    return ap.parse_args()


if __name__ == "__main__":
    main()
