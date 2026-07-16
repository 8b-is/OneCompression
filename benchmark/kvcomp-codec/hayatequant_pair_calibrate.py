"""Build an attention-pair key-rotation HayateQuant artifact.

This keeps the fixed HQ rate recipe and learns only a model-level key
rotation from high-attention post-RoPE query/key pairs. The runtime codec path
is the same calibrated HayateQuant path as HQ.
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
    calibrate_hayatequant_layer_pair_weighted,
    calibrate_hayatequant_pair_weighted,
    capture_attention_pair_samples,
    capture_attention_pair_samples_by_layer,
)

DEFAULT_TEXT = (
    "A compressed KV cache must preserve the attention scores that choose "
    "which previous tokens the decoder reads. Calibration therefore samples "
    "post-rotary query and key pairs with high causal attention and optimizes "
    "the squared score perturbation from key reconstruction error. "
) * 48


def _kv_asym(bits: int) -> tuple[float, float]:
    kv_bits = hayate_hq_role_rates(bits)
    if kv_bits is None:
        raise ValueError(f"average bits must be one of 2, 3, or 4, got {bits}")
    return kv_bits


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
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    texts = [DEFAULT_TEXT]
    if args.text_file:
        texts = [x.strip() for x in Path(args.text_file).read_text().splitlines() if x.strip()]
    if args.block_size not in (None, 2):
        raise ValueError("the fixed HQ calibration recipe requires block_size=2")
    codec = HayateQuantCodec(
        **hayate_hq_kwargs(
            args.avg_bits,
            max_codebook_bits=args.max_codebook_bits,
            encoder="exact",
        )
    )
    if args.layer_aware:
        layer_pairs = capture_attention_pair_samples_by_layer(
            model,
            tok,
            texts,
            max_tokens=args.max_tokens,
            layer_stride=args.layer_stride,
            topk=args.topk,
            query_stride=args.query_stride,
            max_pairs=args.max_pairs,
            seed=args.seed,
            device=model.device,
        )
        cal = calibrate_hayatequant_layer_pair_weighted(
            codec,
            layer_pairs,
            steps=args.steps,
            lr=args.lr,
            batch_pairs=args.batch_pairs,
            max_pairs_per_layer=args.max_pairs,
            holdout_frac=args.holdout_frac,
            mse_lambda=args.mse_lambda,
            trust_lambda=args.trust_lambda,
            seed=args.seed,
            device=args.fit_device,
            verbose=True,
        )
    else:
        qs, ks, ws = capture_attention_pair_samples(
            model,
            tok,
            texts,
            max_tokens=args.max_tokens,
            layer_stride=args.layer_stride,
            topk=args.topk,
            query_stride=args.query_stride,
            max_pairs=args.max_pairs,
            seed=args.seed,
            device=model.device,
        )
        cal = calibrate_hayatequant_pair_weighted(
            codec,
            ks,
            qs,
            ws,
            steps=args.steps,
            lr=args.lr,
            batch_pairs=args.batch_pairs,
            max_pairs=args.max_pairs,
            holdout_frac=args.holdout_frac,
            mse_lambda=args.mse_lambda,
            trust_lambda=args.trust_lambda,
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
            "topk": args.topk,
            "query_stride": args.query_stride,
            "layer_aware": args.layer_aware,
            "recipe": "HQ_b2_Kplus0.5_Vminus0.5_NCkey_LSiterValue_AttentionPairKeyRotation",
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
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--max-codebook-bits", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--query-stride", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=200_000)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-pairs", type=int, default=8192)
    ap.add_argument("--mse-lambda", type=float, default=0.02)
    ap.add_argument("--trust-lambda", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fit-device", default=None)
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--text-file")
    ap.add_argument("--layer-aware", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    main()
