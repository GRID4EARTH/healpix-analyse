"""Tests for the SciPy/HEALPix Gaussian comparison helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _benchmark_module():
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "benchmark_gaussian_filter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_gaussian_filter",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_comparison_is_zero_for_identical_common_grid():
    benchmark = _benchmark_module()
    coordinates = np.linspace(-100.0, 100.0, 21)
    x_m, y_m = np.meshgrid(coordinates, coordinates)
    result = benchmark.synthetic_scene(x_m, y_m)

    comparison = benchmark.compare_results(
        result.ravel(),
        x_m.ravel(),
        y_m.ravel(),
        result,
        coordinates,
        spacing_m=10.0,
        size_m=200.0,
        support_radius_m=20.0,
    )

    assert comparison["points"] > 0
    assert comparison["mae"] == 0.0
    assert comparison["rmse"] == 0.0
    assert comparison["max_abs"] == 0.0
    assert comparison["nrmse"] == 0.0
    assert comparison["correlation"] == 1.0
