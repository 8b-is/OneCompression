"""Microbenchmark Adaptive HQ overhead against canonical HQ-frontier.

This script is intentionally small and dependency-light so the paper appendix
can reproduce the Adaptive HQ overhead table without running full PPL or
zero-shot evaluation.  It measures encode-only latency on synthetic
Llama-shaped KV tensors for a decode step (sequence length 1) and a prefill
chunk (sequence length 256).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from onecomp.kvcomp.codec.hayate.recipes import (
    build_hayate_adaptive_rate,
    build_hayate_frontier,
)


@torch.no_grad()
def _ms_per_encode(codec, x: torch.Tensor, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        codec.encode(x, x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.encode(x, x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return (time.perf_counter() - t0) * 1e3 / float(iters)


def _stored_bits_per_coord(codec, x: torch.Tensor) -> float:
    enc = codec.encode(x, x)
    return float(enc.meta["stored_bits"]) / float(2 * x.numel())


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    rows = []
    for bits in args.bits:
        codecs = {
            "HQ": build_hayate_frontier(bits),
            "Adaptive HQ": build_hayate_adaptive_rate(bits),
        }
        for seq in args.seq_lens:
            x = torch.randn(
                args.batch, args.heads, seq, args.head_dim,
                device=device, dtype=dtype,
            )
            base_ms = None
            for name, codec in codecs.items():
                ms = _ms_per_encode(codec, x, iters=args.iters, warmup=args.warmup)
                if name == "HQ":
                    base_ms = ms
                rows.append({
                    "bits": bits,
                    "seq_len": seq,
                    "method": name,
                    "ms_per_encode": round(ms, 4),
                    "relative_to_hq": None if base_ms is None else round(ms / base_ms, 4),
                    "stored_bits_per_coord": round(_stored_bits_per_coord(codec, x), 4),
                })
    return {
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "device": args.device,
        },
        "iters": args.iters,
        "warmup": args.warmup,
        "results": rows,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "| bits | seq | method | ms/encode | relative | stored b/coord |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["results"]:
        rel = "--" if row["relative_to_hq"] is None else f"{row['relative_to_hq']:.3f}x"
        lines.append(
            f"| {row['bits']} | {row['seq_len']} | {row['method']} | "
            f"{row['ms_per_encode']:.4f} | {rel} | {row['stored_bits_per_coord']:.4f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[1, 256])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(_markdown(payload))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
