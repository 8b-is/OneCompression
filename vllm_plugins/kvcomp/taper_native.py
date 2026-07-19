"""Compatibility shim for the renamed TaperKV reference backend.

New integrations should import :mod:`vllm_plugins.kvcomp.taper_reference`.
The former module path remains available so saved launchers do not break.
"""

# pylint: disable=duplicate-code

from .taper_reference import (
    TAPER_CONFIG_ENV,
    TaperKVNativeBackend,
    TaperKVNativeImpl,
    TaperKVReferenceBackend,
    TaperKVReferenceImpl,
    taper_native_recipe,
    taper_reference_recipe,
)

__all__ = [
    "TAPER_CONFIG_ENV",
    "TaperKVReferenceBackend",
    "TaperKVReferenceImpl",
    "taper_reference_recipe",
    "TaperKVNativeBackend",
    "TaperKVNativeImpl",
    "taper_native_recipe",
]
