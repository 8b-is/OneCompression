"""Development-only fixed HayateQuant recipe search.

This wrapper deliberately keeps the paper method intact: every candidate is
the same data-free rotate -> block-VQ -> role-scale HQ codec.  It varies only
the globally fixed K/V rate offset, source-law fit, and the documented value
LS refit.  Candidate labels are never accepted by the formal-paper runners.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from onecomp.kvcomp.codec import HayateQuantCodec


REPO = Path(
    "/lustre/home/frj/yuma-ichikawa/conference/ichikawa_works/"
    "onecomp-lab-kvcomp-correctness"
)
BENCHMARK = REPO / "benchmark/kvcomp-codec/codec_benchmark.py"


def _load_benchmark():
    # Hydra resolves a relative config_path through the defining module.  The
    # module must therefore be registered in sys.modules while the decorator
    # executes; an anonymous spec makes Hydra fall back to a nonexistent
    # top-level ``conf`` package.
    spec = importlib.util.spec_from_file_location("codec_benchmark", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark module from {BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _half_bit_codec(bits: int, *, dist: str, value_scale_mode: str):
    # A common block size gives an exact fixed average rate.  At average rate 2,
    # b=4 resolves K10/V6 (2.5/1.5 bpc); at rates 3 and 4, b=2 resolves K7/V5
    # and K9/V7 respectively.  Thus delta_r=0.5 is fixed before model/task eval.
    block_size = 4 if bits == 2 else 2
    total = 2 * bits * block_size
    key_code_bits = bits * block_size + block_size // 2
    return HayateQuantCodec(
        bits=bits,
        value_bits=bits,
        avg_bits=float(bits),
        block_size=block_size,
        key_code_bits_per_block=key_code_bits,
        value_code_bits_per_block=total - key_code_bits,
        dist=dist,
        rotation_mode="auto",
        key_scale_mode="nc",
        value_scale_mode=value_scale_mode,
        # The rate-2 development candidate is K10/V6 (1024 key entries).
        # Keep the production/default fail-closed cap unchanged; opt in only
        # for this predeclared development candidate.
        max_codebook_bits=10 if key_code_bits == 10 else 9,
    )


def _half_bit_b2_codec(bits: int, *, dist: str, value_scale_mode: str):
    # The same globally fixed delta_r=0.5, with b=2 at every paper rate.
    # This resolves to K5/V3, K7/V5, and K9/V7 code bits per block.  It keeps
    # every codebook at K<=512 and is therefore the speed-conscious counterpart
    # to the b=4 K10/V6 low-rate candidate above.
    block_size = 2
    total = 2 * bits * block_size
    key_code_bits = bits * block_size + 1
    return HayateQuantCodec(
        bits=bits,
        value_bits=bits,
        avg_bits=float(bits),
        block_size=block_size,
        key_code_bits_per_block=key_code_bits,
        value_code_bits_per_block=total - key_code_bits,
        dist=dist,
        rotation_mode="auto",
        key_scale_mode="nc",
        value_scale_mode=value_scale_mode,
    )


def main():
    benchmark = _load_benchmark()
    # Imported task functions make Hydra interpret ``conf`` as a top-level
    # Python package.  Mark only the undecorated task as script-owned so Hydra
    # uses inspect.getfile(task) and resolves the real sibling ``conf/``
    # directory next to codec_benchmark.py.
    benchmark.main.__wrapped__.__module__ = "__main__"
    original_codecs = benchmark._codecs

    def development_codecs(bits, ablation=False):
        codecs = original_codecs(bits, ablation=ablation)
        codecs.update(
            {
                "HQ-dev-fixed-delta0-Gauss-NC-LSiter": HayateQuantCodec(
                    bits=bits,
                    value_bits=bits,
                    block_size=None,
                    dist="gaussian",
                    rotation_mode="auto",
                    key_scale_mode="nc",
                    value_scale_mode="ls_iter",
                ),
                "HQ-dev-fixed-delta05-Gauss-NC-LSiter": _half_bit_codec(
                    bits, dist="gaussian", value_scale_mode="ls_iter"
                ),
                "HQ-dev-fixed-delta05-Beta-NC-LSiter": _half_bit_codec(
                    bits, dist="beta", value_scale_mode="ls_iter"
                ),
                "HQ-dev-fixed-delta05-Gauss-NC-LS": _half_bit_codec(
                    bits, dist="gaussian", value_scale_mode="ls"
                ),
                "HQ-dev-fixed-delta05-b2-Gauss-NC-LSiter": _half_bit_b2_codec(
                    bits, dist="gaussian", value_scale_mode="ls_iter"
                ),
                "HQ-dev-fixed-delta05-b2-Beta-NC-LSiter": _half_bit_b2_codec(
                    bits, dist="beta", value_scale_mode="ls_iter"
                ),
                "HQ-dev-fixed-delta1-Beta-NC-LSiter": HayateQuantCodec(
                    bits=bits + 1,
                    value_bits=bits - 1,
                    block_size=None,
                    dist="beta",
                    rotation_mode="auto",
                    key_scale_mode="nc",
                    value_scale_mode="ls_iter",
                ),
            }
        )
        return codecs

    benchmark._codecs = development_codecs
    benchmark.main()


if __name__ == "__main__":
    main()
