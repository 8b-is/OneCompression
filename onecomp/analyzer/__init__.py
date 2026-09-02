"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

# Public weight-analysis symbols are resolved through PEP 562 ``__getattr__``.
# pylint: disable=undefined-all-variable

from importlib import import_module
from typing import Any

_WEIGHT_OUTLIER_ATTRS = {
    "LayerOutlierStats",
    "WeightOutlierAnalysis",
    "WeightOutlierAnalyzer",
    "analyze_weight_outliers",
    "save_weight_distribution_plots",
}


def plot_quantization_errors(*args, **kwargs):
    """Plot quantization errors without importing matplotlib at package import time.

    Matplotlib is an optional visualization dependency.  Keeping this public
    convenience import lazy lets non-plotting analyzers (for example cumulative
    error analysis) run in minimal inference environments.
    """
    from .quantization_error import (
        plot_quantization_errors as _plot_quantization_errors,  # pylint: disable=import-outside-toplevel
    )

    return _plot_quantization_errors(*args, **kwargs)


__all__ = [
    "LayerOutlierStats",
    "WeightOutlierAnalysis",
    "WeightOutlierAnalyzer",
    "analyze_weight_outliers",
    "plot_quantization_errors",
    "save_weight_distribution_plots",
]


def __getattr__(name: str) -> Any:
    """Load torch-dependent weight analysis only when its API is requested."""
    if name not in _WEIGHT_OUTLIER_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".weight_outlier", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
