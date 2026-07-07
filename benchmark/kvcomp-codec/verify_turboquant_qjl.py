"""Synthetic checks for the TurboQuant-QJL appendix baseline.

This script is deliberately model-free.  It verifies the pieces that must be
true before running expensive PPL or downstream jobs:

* ``TurboQuant-QJL`` uses one fewer scalar key bit plus a 1-bit key residual
  sketch, while values remain on the regular scalar MSE codec.
* the realised storage includes the residual norm metadata;
* the QJL row should not be interpreted as an MSE reconstruction improvement.

Run:
    python benchmark/kvcomp-codec/verify_turboquant_qjl.py

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from onecomp.kvcomp.codec.turboquant import TurboQuantCodec


def _relmse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).pow(2).sum() / x.float().pow(2).sum().clamp_min(1e-12))


def _score_error(q: torch.Tensor, k_hat: torch.Tensor, k: torch.Tensor) -> tuple[float, float]:
    err = (q.float() * (k_hat.float() - k.float())).sum(dim=-1)
    return float(err.mean()), float(err.pow(2).mean().sqrt())


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    key = torch.randn(args.batch, args.heads, args.seq, args.dim)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    rows = []
    for total_bits in args.bits:
        methods = {
            "TurboQuant-MSE": TurboQuantCodec(bits=total_bits, rotation_mode=args.rotation_mode),
            "TurboQuant-QJL": TurboQuantCodec(
                bits=total_bits - 1,
                value_bits=total_bits,
                rotation_mode=args.rotation_mode,
                use_qjl=True,
                qjl_roles="key",
            ),
        }
        for name, codec in methods.items():
            enc = codec.encode(key, value)
            key_hat, value_hat = codec.decode(enc)
            bias, rmse = _score_error(query, key_hat, key)
            rows.append({
                "target_bits": total_bits,
                "method": name,
                "key_codec_bits": codec.bits,
                "value_codec_bits": codec._value_bits(),
                "stored_bits_per_coord": enc.meta["stored_bits"] / (key.numel() + value.numel()),
                "key_relmse": _relmse(key_hat, key),
                "value_relmse": _relmse(value_hat, value),
                "score_error_mean": bias,
                "score_error_rmse": rmse,
            })
    config = vars(args).copy()
    config["output"] = None if args.output is None else str(args.output)
    return {"config": config, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rotation-mode", default="hadamard", choices=["hadamard", "dense"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data = run(args)
    text = json.dumps(data, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
