"""Backward-compatibility shim.

``HayateQuantCodec`` moved to :mod:`onecomp.kvcomp.codec.hayate.codec`.
Import from :mod:`onecomp.kvcomp.codec` (or ``...codec.hayate``) in new code.

Copyright 2025-2026 Fujitsu Ltd.
"""

from .hayate.codec import (  # noqa: F401
    HayateQuantCodec,
    _scalar_codebook,
    _block_codebook,
)
