"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from .quantization_error import plot_quantization_errors
from .weight_outlier import (
    LayerOutlierStats,
    WeightOutlierAnalysis,
    WeightOutlierAnalyzer,
    analyze_weight_outliers,
    save_weight_distribution_plots,
)

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
