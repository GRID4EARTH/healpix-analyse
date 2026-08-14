"""Reproducible Level-19 benchmark and profiler for Gaussian filtering."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
from healpix_geo import nested

from healpix_analyse.radial_filter import _clear_filter_caches, gaussian_filter


AUTHALIC_RADIUS_M = 6_371_007.1809


def patch(size_m: float, refinement_level: int) -> np.ndarray:
    """Return a circular patch enclosing a square of the requested size."""
    radius_m = size_m * np.sqrt(2.0) / 2.0
    radius_deg = np.rad2deg(radius_m / AUTHALIC_RADIUS_M)
    ids, _, _ = nested.cone_coverage(
        (2.0, 48.0),
        radius_deg,
        refinement_level,
        ellipsoid="WGS84",
        flat=True,
    )
    return np.asarray(ids, dtype=np.uint64)


def run(size_m: float, sigma_m: float, repeats: int, profile: bool) -> None:
    level = 19
    cell_ids = patch(size_m, level)
    values = np.sin(np.arange(cell_ids.size, dtype=np.float64) * 0.01)
    values[::401] = np.nan

    _clear_filter_caches()
    profiler = cProfile.Profile() if profile else None
    if profiler is not None:
        profiler.enable()

    started = time.perf_counter()
    result = gaussian_filter(
        values,
        cell_ids,
        level,
        sigma_m=sigma_m,
        truncate=4.0,
    )
    cold = time.perf_counter() - started

    if profiler is not None:
        profiler.disable()

    warm = []
    for offset in range(repeats):
        started = time.perf_counter()
        gaussian_filter(
            values + offset,
            cell_ids,
            level,
            sigma_m=sigma_m,
            truncate=4.0,
        )
        warm.append(time.perf_counter() - started)

    print(f"level={level} size_m={size_m:g} cells={cell_ids.size}")
    print(f"sigma_m={sigma_m:g} cold_s={cold:.6f}")
    print(f"warm_median_s={np.median(warm):.6f} repeats={repeats}")
    print(f"checksum={np.nanmean(result):.17g}")

    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(25)
        print(stream.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-m", type=float, default=600.0)
    parser.add_argument("--sigma-m", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run(args.size_m, args.sigma_m, args.repeats, args.profile)


if __name__ == "__main__":
    main()
