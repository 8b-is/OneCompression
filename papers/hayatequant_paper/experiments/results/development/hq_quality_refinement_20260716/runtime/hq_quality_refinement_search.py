"""Development-only, globally predeclared HQ/AHQ quality refinement panel."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch

from onecomp.kvcomp.codec import HayateQuantCodec
from onecomp.kvcomp.codec.hayate.codec import _block_codebook_by_code_bits


REPO = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
    "onecomp-lab-kvcomp-correctness"
)
BENCHMARK = REPO / "benchmark/kvcomp-codec/codec_benchmark.py"

FIXED_METHODS = (
    "DEVONLY-HQ-b4-gaussian-auto",
    "DEVONLY-HQ-b4-gaussian-fibinit",
    "DEVONLY-HQ-b4-finite-beta",
)
ADAPTIVE_WEIGHTS = (1.0, 4.0, 16.0)
ADAPTIVE_METHODS = tuple(
    f"DEVONLY-AdaptiveHQ-safe-kw{int(weight)}" for weight in ADAPTIVE_WEIGHTS
)


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("codec_benchmark", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FixedRefinementHQ(HayateQuantCodec):
    """Attach immutable codebook identity/footprint to a development row."""

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        raw = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def get_resolved_kv_config(self, head_dim: int) -> dict:
        resolved = super().get_resolved_kv_config(head_dim)
        identities = {}
        for role, code_bits in (
            ("key", int(self.key_code_bits_per_block)),
            ("value", int(self.value_code_bits_per_block)),
        ):
            codebook = _block_codebook_by_code_bits(
                code_bits,
                int(self.block_size),
                self.dist,
                int(head_dim),
                self.codebook_init,
            ).contiguous()
            identities[role] = {
                "sha256": self._tensor_sha256(codebook),
                "shape": list(codebook.shape),
                "dtype": str(codebook.dtype),
                "footprint_bytes": int(codebook.numel() * codebook.element_size()),
            }
        resolved["development_codebook_identity"] = {
            "head_dim": int(head_dim),
            "roles": identities,
            "total_footprint_bytes": sum(
                int(row["footprint_bytes"]) for row in identities.values()
            ),
        }
        return resolved


def _fixed_rate2_codec(*, dist: str, codebook_init: str) -> HayateQuantCodec:
    return _FixedRefinementHQ(
        bits=2,
        value_bits=2,
        avg_bits=2.0,
        block_size=4,
        key_code_bits_per_block=10,
        value_code_bits_per_block=6,
        max_codebook_bits=10,
        dist=dist,
        codebook_init=codebook_init,
        rotation_mode="auto",
        encoder="exact",
        key_scale_mode="nc",
        value_scale_mode="ls_iter",
    )


class _RestrictedAdaptiveHQ(HayateQuantCodec):
    """Development probe with the preregistered safe two-pair candidate set."""

    def _adaptive_code_bit_pairs(self, block_size: int) -> list[tuple[int, int]]:
        block = int(block_size)
        if block != 2:
            raise ValueError(f"safe AdaptiveHQ development policy requires block_size=2, got {block}")
        total = self._pair_code_bits_total(block)
        if total is None or int(total) % 4:
            raise ValueError(f"safe AdaptiveHQ needs total pair bits divisible by four, got {total}")
        rate = int(total) // 4
        pairs = [(2 * rate, 2 * rate), (2 * rate + 1, 2 * rate - 1)]
        if any(min(pair) < 1 or max(pair) > int(self.max_codebook_bits) for pair in pairs):
            raise ValueError(
                "safe AdaptiveHQ candidate is infeasible under max_codebook_bits: "
                f"pairs={pairs}, cap={self.max_codebook_bits}"
            )
        return pairs

    def reset_development_trace(self) -> None:
        self._development_trace = {
            "candidate_set_version": "symmetric_plus_fixed_hq_v1",
            "candidate_pairs": [list(pair) for pair in self._adaptive_code_bit_pairs(2)],
            "encoded_updates": 0,
            "encoded_tokens": 0,
            "encoded_vectors": 0,
            "selector_bits_total": 0,
            "selector_incremental_stored_bits_total": 0,
            "split_histogram": {},
        }

    def development_trace(self) -> dict:
        if not hasattr(self, "_development_trace"):
            self.reset_development_trace()
        return copy.deepcopy(self._development_trace)

    def _commit_adaptive_state(self, state: dict, cumulative: dict, split_meta: dict) -> None:
        HayateQuantCodec._commit_adaptive_state(state, cumulative, split_meta)
        if not hasattr(self, "_development_trace"):
            self.reset_development_trace()
        trace = self._development_trace
        pair = [
            int(split_meta["key_code_bits_per_block"]),
            int(split_meta["value_code_bits_per_block"]),
        ]
        pair_key = f"{pair[0]}:{pair[1]}"
        update = split_meta["update"]
        bucket = trace["split_histogram"].setdefault(
            pair_key,
            {"pair": pair, "updates": 0, "tokens": 0, "vectors": 0},
        )
        bucket["updates"] += 1
        bucket["tokens"] += int(update["tokens"])
        bucket["vectors"] += int(update["vectors"])
        trace["encoded_updates"] += 1
        trace["encoded_tokens"] += int(update["tokens"])
        trace["encoded_vectors"] += int(update["vectors"])
        trace["selector_bits_total"] += int(split_meta["selector_bits"])
        trace["selector_incremental_stored_bits_total"] += int(
            split_meta["selector_incremental_stored_bits"]
        )


def _adaptive_codec(bits: int, weight: float) -> _RestrictedAdaptiveHQ:
    rate = int(bits)
    return _RestrictedAdaptiveHQ(
        bits=rate,
        value_bits=rate,
        avg_bits=float(rate),
        block_size=2,
        key_bit_fraction=0.5,
        adaptive_bit_split="online_stats",
        adaptive_stats_min_tokens=16,
        key_error_weight=float(weight),
        value_error_weight=1.0,
        max_codebook_bits=9,
        dist="gaussian",
        codebook_init="auto",
        rotation_mode="auto",
        encoder="exact",
        key_scale_mode="nc",
        value_scale_mode="ls_iter",
    )


def main() -> None:
    kind = os.environ.get("HQ_REFINEMENT_KIND")
    if kind not in {"fixed", "adaptive"}:
        raise RuntimeError("HQ_REFINEMENT_KIND must be 'fixed' or 'adaptive'")

    benchmark = _load_benchmark()
    benchmark.main.__wrapped__.__module__ = "__main__"
    original_codecs = benchmark._codecs
    original_ppl = benchmark.wikitext2_perplexity
    original_ppl_fields = benchmark._ppl_fields

    def development_codecs(bits, ablation=False, head_dim=128):
        codecs = original_codecs(bits, ablation=ablation, head_dim=head_dim)
        if kind == "fixed":
            if int(bits) != 2:
                raise ValueError("fixed refinement candidates are preregistered only at rate 2")
            codecs.update(
                {
                    FIXED_METHODS[0]: _fixed_rate2_codec(dist="gaussian", codebook_init="auto"),
                    FIXED_METHODS[1]: _fixed_rate2_codec(
                        dist="gaussian", codebook_init="fibquant"
                    ),
                    FIXED_METHODS[2]: _fixed_rate2_codec(dist="beta", codebook_init="auto"),
                }
            )
        else:
            codecs.update(
                {
                    name: _adaptive_codec(int(bits), weight)
                    for name, weight in zip(ADAPTIVE_METHODS, ADAPTIVE_WEIGHTS)
                }
            )
        return codecs

    def traced_ppl(*args, **kwargs):
        factory = kwargs.get("cache_factory")
        defaults = getattr(factory, "__defaults__", ()) or ()
        codec = next((item for item in defaults if isinstance(item, _RestrictedAdaptiveHQ)), None)
        if codec is not None:
            codec.reset_development_trace()
        result = original_ppl(*args, **kwargs)
        if codec is not None:
            result["development_adaptive_split_trace"] = codec.development_trace()
        return result

    def traced_ppl_fields(result):
        fields = original_ppl_fields(result)
        trace = result.get("development_adaptive_split_trace")
        if trace is not None:
            fields["ppl_development_adaptive_split_trace"] = trace
        return fields

    benchmark._codecs = development_codecs
    benchmark.wikitext2_perplexity = traced_ppl
    benchmark._ppl_fields = traced_ppl_fields
    benchmark.main()


if __name__ == "__main__":
    main()
