"""KV dtype quantization codecs.

All codecs -- both calibration-free / online (FP8, INT4, KIVI, TurboQuant,
FibQuant, HayateQuant) and calibration-based (KVQuant, KVTuner; added later) -- live
here and are distinguished by ``flag_calibration``, mirroring how RTN and
GPTQ coexist in :mod:`onecomp.quantizer`.

``HALOCodec`` is kept for backward compatibility with existing configs and
result reproduction.  :class:`HayateQuantCodec` exposes scalar compatibility
controls: with ``block_size=1`` and explicitly matched rotation, source law,
and scale modes, selected TurboQuant/HALO paths can be reproduced to numerical
precision.  Its default block-VQ recipe is a distinct operating point whose
quality and latency must be measured; no universal dominance or strict-
superset claim is implied.  New HayateQuant work should use the named recipes
in :mod:`onecomp.kvcomp.codec.hayate.recipes` so those controls stay explicit.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .affine import AffineKVCodec, Int8Codec, Int4Codec, Int3Codec, Int2Codec, KIVICodec
from .fp8 import FP8Codec
from .turboquant import TurboQuantCodec
from .fibquant import FibQuantCodec
from .blockquant import BlockQuantCodec
from .halo import HALOCodec
from .hayate import HayateQuantCodec
from .lowbit_fp import NVFP4Codec, TernaryCodec
from .packed_affine import PackedAffineCodec

__all__ = [
    "AffineKVCodec",
    "Int8Codec",
    "Int4Codec",
    "Int3Codec",
    "Int2Codec",
    "KIVICodec",
    "FP8Codec",
    "TurboQuantCodec",
    "FibQuantCodec",
    "BlockQuantCodec",
    "HALOCodec",
    "HayateQuantCodec",
    "NVFP4Codec",
    "TernaryCodec",
    "PackedAffineCodec",
]
