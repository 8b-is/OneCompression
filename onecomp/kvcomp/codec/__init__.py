"""KV dtype quantization codecs.

All codecs -- both calibration-free / online (FP8, INT4, KIVI, TurboQuant,
HayateQuant) and calibration-based (KVQuant, KVTuner; added later) -- live
here and are distinguished by ``flag_calibration``, mirroring how RTN and
GPTQ coexist in :mod:`onecomp.quantizer`.

``HALOCodec`` is kept for backward compatibility with existing configs and
result reproduction, but is superseded by :class:`HayateQuantCodec`, a strict
superset: ``HayateQuantCodec(block_size=1, rotation_mode="hadamard_gemm")``
reproduces TurboQuant to numerical precision, and its role-asymmetric scale
modes (``key_scale_mode="nc"``, ``value_scale_mode="ls"``/``"ls_iter"``)
reproduce HALO's scaling exactly -- while block-VQ (``block_size>1``) and the
hybrid Hadamard+block-dense rotation give strictly lower distortion at equal
bit-rate than either predecessor (see
``benchmark/kvcomp-longcontext/paper/tests/verify_hayatequant_vq_gain.py``).
New work should use ``HayateQuantCodec``.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .affine import AffineKVCodec, Int8Codec, Int4Codec, Int3Codec, Int2Codec, KIVICodec
from .fp8 import FP8Codec
from .turboquant import TurboQuantCodec
from .halo import HALOCodec
from .hayatequant import HayateQuantCodec
from .lowbit_fp import NVFP4Codec, TernaryCodec

__all__ = [
    "AffineKVCodec",
    "Int8Codec",
    "Int4Codec",
    "Int3Codec",
    "Int2Codec",
    "KIVICodec",
    "FP8Codec",
    "TurboQuantCodec",
    "HALOCodec",
    "HayateQuantCodec",
    "NVFP4Codec",
    "TernaryCodec",
]
