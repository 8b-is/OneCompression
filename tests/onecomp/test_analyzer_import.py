"""Import-boundary tests for optional analyzer visualization dependencies."""

import importlib
import sys


def test_analyzer_package_does_not_eagerly_import_plotting_module():
    sys.modules.pop("onecomp.analyzer.quantization_error", None)
    sys.modules.pop("onecomp.analyzer", None)

    analyzer = importlib.import_module("onecomp.analyzer")

    assert callable(analyzer.plot_quantization_errors)
    assert "onecomp.analyzer.quantization_error" not in sys.modules
