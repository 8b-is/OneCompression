"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from .weight_outlier import (
    LayerOutlierStats,
    WeightOutlierAnalysis,
    WeightOutlierAnalyzer,
    analyze_weight_outliers,
    save_weight_distribution_plots,
)


def plot_quantization_errors(*args, **kwargs):
    """Plot quantization errors without importing matplotlib at package import time.

    Matplotlib is an optional visualization dependency.  Keeping this public
    convenience import lazy lets non-plotting analyzers (for example cumulative
    error analysis) run in minimal inference environments.
    """
    from .quantization_error import (  # pylint: disable=import-outside-toplevel
        plot_quantization_errors as _plot_quantization_errors,
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
