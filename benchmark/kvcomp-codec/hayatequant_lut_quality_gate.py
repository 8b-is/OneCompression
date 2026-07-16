#!/usr/bin/env python3
"""Development quality gate for HQ's fixed public/native table assignment.

This deliberately compares only two assignment implementations of the same
frozen HQ recipe: the canonical all-codeword exact search and a fixed
grid-table development candidate.  It is not a new public method/variant and
must not be rendered as one in the paper.  WikiText-2 is revision and
joined-text-digest pinned by :func:`onecomp.kvcomp.eval.wikitext2_perplexity`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_weight_manifest import fingerprint_model_weights
from onecomp.kvcomp.cache import CompressedDynamicCache
from onecomp.kvcomp.codec.hayate.recipes import hayate_hq_kwargs
from onecomp.kvcomp.codec.hayate.codec import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.vq import quantize_exact_pre
from onecomp.kvcomp.eval import wikitext2_perplexity

REPO = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metadata_hashes(model_path: Path) -> dict[str, str]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    return {name: _sha(model_path / name) for name in names if (model_path / name).is_file()}


def _codec(
    rate: int,
    assignment: str,
    *,
    lut_bins: int,
    lut_candidates: int,
) -> HayateQuantCodec:
    kwargs = hayate_hq_kwargs(rate)
    if assignment == "exact":
        kwargs["encoder"] = "exact"
    elif assignment == "lut":
        kwargs.update(
            encoder="lut",
            lut_bins=int(lut_bins),
            lut_candidates=int(lut_candidates),
        )
    else:
        raise ValueError(f"unknown assignment {assignment!r}")
    return HayateQuantCodec(**kwargs)


class _AssignmentAudit:
    """Count whether LUT mismatches come from set omission or final scoring."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, int]] = {}

    def wrap(self, original):
        def audited(z, lut, bins, cb, lim=5.0, **kwargs):
            got = original(z, lut, bins, cb, lim=lim, **kwargs)
            scoring_cb = kwargs.get("scoring_cb")
            scoring_csq = kwargs.get("scoring_csq")
            if scoring_cb is None or scoring_csq is None:
                return got
            score_z = z.to(dtype=scoring_cb.dtype)
            exact = quantize_exact_pre(score_z, scoring_cb, scoring_csq)
            block_size = int(z.shape[-1])
            step = bins / (2.0 * lim)
            zi = ((z.clamp(-lim, lim) + lim) * step).long().clamp_(0, bins - 1)
            flat = zi[..., 0]
            for axis in range(1, block_size):
                flat = flat * bins + zi[..., axis]
            candidates = lut[flat].long()
            contained = (candidates == exact.unsqueeze(-1)).any(dim=-1)
            mismatch = got != exact
            outside = (z.abs() > lim).any(dim=-1)
            name = (
                f"K{int(cb.shape[0])}_b{block_size}_T{int(lut.shape[-1])}_"
                f"{str(scoring_cb.dtype).removeprefix('torch.')}"
            )
            row = self.rows.setdefault(
                name,
                {
                    "calls": 0,
                    "blocks": 0,
                    "mismatches": 0,
                    "winner_absent": 0,
                    "winner_present_but_scoring_mismatch": 0,
                    "outside_lut_box": 0,
                    "outside_and_mismatch": 0,
                },
            )
            row["calls"] += 1
            row["blocks"] += int(mismatch.numel())
            row["mismatches"] += int(mismatch.sum().item())
            row["winner_absent"] += int((~contained).sum().item())
            row["winner_present_but_scoring_mismatch"] += int((mismatch & contained).sum().item())
            row["outside_lut_box"] += int(outside.sum().item())
            row["outside_and_mismatch"] += int((outside & mismatch).sum().item())
            return got

        return audited

    def result(self) -> dict[str, dict[str, int | float]]:
        result = {}
        for name, row in self.rows.items():
            blocks = max(1, row["blocks"])
            result[name] = {
                **row,
                "mismatch_fraction": row["mismatches"] / blocks,
                "winner_absent_fraction": row["winner_absent"] / blocks,
            }
        return result


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_path = args.model_path.expanduser().resolve()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, fix_mistral_regex=True
    )
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()

    rows: dict[str, dict] = {}
    failures: list[str] = []
    for index, rate in enumerate(args.rates):
        # Alternate order across rates.  Results are deterministic point
        # estimates, but counterbalancing also exposes accidental process-state
        # dependence in the cached table/reference paths.
        order = ("exact", "lut") if index % 2 == 0 else ("lut", "exact")
        assignments = {}
        for assignment in order:
            codec = _codec(
                rate,
                assignment,
                lut_bins=args.lut_bins,
                lut_candidates=args.lut_candidates,
            )
            audit = (
                _AssignmentAudit()
                if args.audit_assignment_mismatches and assignment == "lut"
                else None
            )
            if audit is not None:
                import onecomp.kvcomp.codec.hayate.codec as codec_module

                original_assignment = codec_module.quantize_via_lut
                codec_module.quantize_via_lut = audit.wrap(original_assignment)
            try:
                ppl = wikitext2_perplexity(
                    model,
                    tokenizer,
                    cache_factory=lambda c=codec: CompressedDynamicCache(
                        codec=c,
                        config=model.config,
                        sink_tokens=0,
                        recent_tokens=0,
                    ),
                    max_length=args.max_length,
                    stride=args.stride,
                    max_windows=args.max_windows,
                )
            finally:
                if audit is not None:
                    codec_module.quantize_via_lut = original_assignment
            assignments[assignment] = {
                "codec_config": codec.get_kv_config(),
                "resolved_codec_config": codec.get_resolved_kv_config(
                    int(model.config.head_dim)
                    if getattr(model.config, "head_dim", None) is not None
                    else int(model.config.hidden_size // model.config.num_attention_heads)
                ),
                **ppl,
                **({"assignment_audit": audit.result()} if audit is not None else {}),
            }
        exact = assignments["exact"]
        lut = assignments["lut"]
        same_scientific_input = all(
            lut.get(name) == exact.get(name)
            for name in (
                "n_tokens",
                "n_windows",
                "source",
                "dataset_revision",
                "text_sha256",
                "teacher_forced_scoring_protocol",
                "teacher_forced_temporal_dependency",
                "compression_fp16_bits",
                "compression_stored_bits",
            )
        )
        relative_pct = (float(lut["ppl"]) / float(exact["ppl"]) - 1.0) * 100.0
        passed = same_scientific_input and relative_pct <= args.max_relative_ppl_degradation_pct
        rows[str(rate)] = {
            "execution_order": list(order),
            "assignments": assignments,
            "lut_minus_exact_nll": float(lut["nll"]) - float(exact["nll"]),
            "lut_relative_ppl_degradation_pct": relative_pct,
            "same_scientific_input_and_rate_accounting": same_scientific_input,
            "max_relative_ppl_degradation_pct": args.max_relative_ppl_degradation_pct,
            "passed": passed,
        }
        if not passed:
            failures.append(f"rate{rate}")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sources = (
        Path("onecomp/kvcomp/codec/hayate/recipes.py"),
        Path("onecomp/kvcomp/codec/hayate/codec.py"),
        Path("onecomp/kvcomp/codec/hayate/vq.py"),
        Path("onecomp/kvcomp/codec/hayate/triton_step.py"),
        Path("onecomp/kvcomp/eval/ppl.py"),
        Path("benchmark/kvcomp-codec/hayatequant_lut_quality_gate.py"),
    )
    return {
        "schema": {
            "name": f"onecomp.hq_lut_t{args.lut_candidates}_quality_gate",
            "version": 1,
        },
        "development_only": True,
        "public_method_variants": ["HQ"],
        "comparison": "formal HQ table assignment vs exact-search implementation reference",
        "passed": not failures,
        "failures": failures,
        "rates": rows,
        "provenance": {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": _git("status", "--porcelain"),
            "arguments": {
                **vars(args),
                "model_path": str(model_path),
            },
            "model_metadata_sha256": _metadata_hashes(model_path),
            "model_weights": fingerprint_model_weights(model_path),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": {
                "name": props.name,
                "capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
            },
            "source_sha256": {str(path): _sha(REPO / path) for path in sources},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", choices=(2, 3, 4), default=(2, 3, 4))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--max-relative-ppl-degradation-pct", type=float, default=0.25)
    parser.add_argument("--lut-bins", type=int, default=256)
    parser.add_argument("--lut-candidates", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--audit-assignment-mismatches", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
