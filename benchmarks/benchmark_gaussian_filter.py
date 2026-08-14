"""Reproducible benchmark and profiler for Gaussian filtering."""

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


def equivalent_cartesian_grid(
    cell_count: int,
    size_m: float,
) -> tuple[np.ndarray, float]:
    """Return an area- and sample-count-matched square Cartesian grid.

    ``patch`` constructs a circle around the requested square.  This helper
    replaces that circle by a square with the same local planar area and picks
    the nearest square grid size to the number of HEALPix cells.
    """
    radius_m = size_m * np.sqrt(2.0) / 2.0
    patch_area_m2 = np.pi * radius_m**2
    side = max(1, int(np.rint(np.sqrt(cell_count))))
    spacing_m = np.sqrt(patch_area_m2) / side
    values = np.sin(np.arange(side * side, dtype=np.float64) * 0.01)
    return values.reshape(side, side), float(spacing_m)


def scipy_times(
    cell_count: int,
    size_m: float,
    sigma_m: float,
    truncate: float,
    repeats: int,
) -> tuple[tuple[int, int], float, float]:
    """Time SciPy on a comparable regular grid, importing it on demand."""
    try:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    except ImportError as error:
        raise RuntimeError(
            "SciPy comparison requires the 'benchmark' optional dependency"
        ) from error

    values, spacing_m = equivalent_cartesian_grid(cell_count, size_m)
    sigma_pixels = sigma_m / spacing_m

    # Exclude import and one-time allocation effects from the apply timing.
    scipy_gaussian_filter(
        values,
        sigma=sigma_pixels,
        truncate=truncate,
        mode="reflect",
    )
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        scipy_gaussian_filter(
            values,
            sigma=sigma_pixels,
            truncate=truncate,
            mode="reflect",
        )
        timings.append(time.perf_counter() - started)
    return values.shape, spacing_m, float(np.median(timings))


def run(
    level: int,
    size_m: float,
    sigma_m: float,
    truncate: float,
    repeats: int,
    profile: bool,
    compare_scipy: bool,
    scipy_repeats: int,
) -> None:
    cell_ids = patch(size_m, level)
    values = np.sin(np.arange(cell_ids.size, dtype=np.float64) * 0.01)

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
        truncate=truncate,
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
            truncate=truncate,
        )
        warm.append(time.perf_counter() - started)

    print(f"level={level} size_m={size_m:g} cells={cell_ids.size}")
    print(
        f"sigma_m={sigma_m:g} truncate={truncate:g} cold_s={cold:.6f}"
    )
    print(f"warm_median_s={np.median(warm):.6f} repeats={repeats}")
    print(f"checksum={np.nanmean(result):.17g}")

    if compare_scipy:
        shape, spacing_m, scipy_median = scipy_times(
            cell_ids.size,
            size_m,
            sigma_m,
            truncate,
            scipy_repeats,
        )
        warm_median = float(np.median(warm))
        print("scipy_comparison=reference_only_not_numerical_equivalence")
        print(
            f"cartesian_grid={shape[0]}x{shape[1]} "
            f"samples={shape[0] * shape[1]} spacing_m={spacing_m:.6f} "
            f"sigma_pixels={sigma_m / spacing_m:.6f}"
        )
        print(
            f"scipy_apply_median_s={scipy_median:.6f} "
            f"repeats={scipy_repeats} mode=reflect"
        )
        print(f"healpix_cold_to_scipy_ratio={cold / scipy_median:.3f}")
        print(
            f"healpix_repeat_to_scipy_ratio="
            f"{warm_median / scipy_median:.3f}"
        )

    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(25)
        print(stream.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=19)
    parser.add_argument("--size-m", type=float, default=600.0)
    parser.add_argument("--sigma-m", type=float, default=20.0)
    parser.add_argument("--truncate", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compare-scipy", action="store_true")
    parser.add_argument("--scipy-repeats", type=int, default=25)
    args = parser.parse_args()
    if args.repeats < 1 or args.scipy_repeats < 1:
        parser.error("repeat counts must be positive")
    run(
        args.level,
        args.size_m,
        args.sigma_m,
        args.truncate,
        args.repeats,
        args.profile,
        args.compare_scipy,
        args.scipy_repeats,
    )


if __name__ == "__main__":
    main()
