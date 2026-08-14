"""Reproducible benchmark and profiler for Gaussian filtering."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np
from healpix_geo import nested
from pyproj import Geod

from healpix_analyse.radial_filter import _clear_filter_caches, gaussian_filter


AUTHALIC_RADIUS_M = 6_371_007.1809
PATCH_CENTER = (2.0, 48.0)
WGS84 = Geod(ellps="WGS84")


def patch(size_m: float, refinement_level: int) -> np.ndarray:
    """Return a circular patch enclosing a square of the requested size."""
    radius_m = size_m * np.sqrt(2.0) / 2.0
    radius_deg = np.rad2deg(radius_m / AUTHALIC_RADIUS_M)
    ids, _, _ = nested.cone_coverage(
        PATCH_CENTER,
        radius_deg,
        refinement_level,
        ellipsoid="WGS84",
        flat=True,
    )
    return np.asarray(ids, dtype=np.uint64)


def equivalent_cartesian_grid(
    cell_count: int,
    size_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return an area- and sample-count-matched square Cartesian grid.

    ``patch`` constructs a circle around the requested square.  This helper
    replaces that circle by a square with the same local planar area and picks
    the nearest square grid size to the number of HEALPix cells.
    """
    radius_m = size_m * np.sqrt(2.0) / 2.0
    patch_area_m2 = np.pi * radius_m**2
    side = max(1, int(np.rint(np.sqrt(cell_count))))
    spacing_m = np.sqrt(patch_area_m2) / side
    coordinates = (
        np.arange(side, dtype=np.float64) + 0.5
    ) * spacing_m - np.sqrt(patch_area_m2) / 2.0
    x_m, y_m = np.meshgrid(coordinates, coordinates)
    values = synthetic_scene(x_m, y_m)
    return values, coordinates, float(spacing_m)


def healpix_local_coordinates(
    cell_ids: np.ndarray,
    refinement_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return east/north offsets of HEALPix centres from the patch centre."""
    longitude, latitude = nested.healpix_to_lonlat(
        cell_ids,
        refinement_level,
        ellipsoid="WGS84",
    )
    azimuth_deg, _, distance_m = WGS84.inv(
        np.full(longitude.shape, PATCH_CENTER[0]),
        np.full(latitude.shape, PATCH_CENTER[1]),
        longitude,
        latitude,
    )
    azimuth_rad = np.deg2rad(azimuth_deg)
    return (
        distance_m * np.sin(azimuth_rad),
        distance_m * np.cos(azimuth_rad),
    )


def synthetic_scene(x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
    """Evaluate a smooth, non-symmetric scene in local metric coordinates."""
    first = 0.8 * np.exp(
        -0.5 * (((x_m + 115.0) / 42.0) ** 2 + ((y_m - 55.0) / 65.0) ** 2)
    )
    second = 0.45 * np.exp(
        -0.5 * (((x_m - 130.0) / 85.0) ** 2 + ((y_m + 95.0) / 50.0) ** 2)
    )
    wave = 0.12 * np.sin(x_m / 72.0) * np.cos(y_m / 91.0)
    return 0.25 + first + second + wave


def scipy_times(
    cell_count: int,
    size_m: float,
    sigma_m: float,
    truncate: float,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Time SciPy on a comparable regular grid, importing it on demand."""
    try:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    except ImportError as error:
        raise RuntimeError(
            "SciPy comparison requires the 'benchmark' optional dependency"
        ) from error

    values, coordinates, spacing_m = equivalent_cartesian_grid(
        cell_count,
        size_m,
    )
    sigma_pixels = sigma_m / spacing_m

    # Exclude import and one-time allocation effects from the apply timing.
    result = scipy_gaussian_filter(
        values,
        sigma=sigma_pixels,
        truncate=truncate,
        mode="reflect",
    )
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = scipy_gaussian_filter(
            values,
            sigma=sigma_pixels,
            truncate=truncate,
            mode="reflect",
        )
        timings.append(time.perf_counter() - started)
    return result, coordinates, spacing_m, float(np.median(timings))


def compare_results(
    healpix_result: np.ndarray,
    healpix_x_m: np.ndarray,
    healpix_y_m: np.ndarray,
    scipy_result: np.ndarray,
    cartesian_coordinates_m: np.ndarray,
    spacing_m: float,
    size_m: float,
    support_radius_m: float,
) -> dict[str, float]:
    """Compare both outputs at common coordinates away from boundaries."""
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as error:
        raise RuntimeError(
            "SciPy comparison requires the 'benchmark' optional dependency"
        ) from error

    patch_radius_m = size_m * np.sqrt(2.0) / 2.0
    cartesian_half_width_m = (
        cartesian_coordinates_m[-1]
        + spacing_m / 2.0
    )
    # Include two grid spacings of guard space for cell-centre displacement
    # and interpolation support in addition to the truncated kernel radius.
    margin_m = support_radius_m + 2.0 * spacing_m
    common = (
        (np.hypot(healpix_x_m, healpix_y_m) <= patch_radius_m - margin_m)
        & (np.abs(healpix_x_m) <= cartesian_half_width_m - margin_m)
        & (np.abs(healpix_y_m) <= cartesian_half_width_m - margin_m)
    )
    if not np.any(common):
        raise RuntimeError("No common interior points remain for comparison")

    column = (
        healpix_x_m[common] - cartesian_coordinates_m[0]
    ) / spacing_m
    row = (
        healpix_y_m[common] - cartesian_coordinates_m[0]
    ) / spacing_m
    scipy_at_healpix = map_coordinates(
        scipy_result,
        np.vstack((row, column)),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    difference = healpix_result[common] - scipy_at_healpix
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_range = float(np.ptp(scipy_at_healpix))
    correlation = float(
        np.corrcoef(healpix_result[common], scipy_at_healpix)[0, 1]
    )
    return {
        "points": float(np.count_nonzero(common)),
        "margin_m": float(margin_m),
        "mae": float(np.mean(absolute)),
        "rmse": rmse,
        "max_abs": float(np.max(absolute)),
        "nrmse": rmse / reference_range,
        "correlation": correlation,
    }


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
    healpix_x_m, healpix_y_m = healpix_local_coordinates(cell_ids, level)
    values = synthetic_scene(healpix_x_m, healpix_y_m)

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
        scipy_result, coordinates_m, spacing_m, scipy_median = scipy_times(
            cell_ids.size,
            size_m,
            sigma_m,
            truncate,
            scipy_repeats,
        )
        comparison = compare_results(
            result,
            healpix_x_m,
            healpix_y_m,
            scipy_result,
            coordinates_m,
            spacing_m,
            size_m,
            sigma_m * truncate,
        )
        shape = scipy_result.shape
        warm_median = float(np.median(warm))
        print("scipy_comparison=same_scene_common_interior")
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
        print(
            f"result_comparison_points={int(comparison['points'])} "
            f"boundary_margin_m={comparison['margin_m']:.6f}"
        )
        print(
            f"result_mae={comparison['mae']:.9g} "
            f"result_rmse={comparison['rmse']:.9g} "
            f"result_max_abs={comparison['max_abs']:.9g}"
        )
        print(
            f"result_nrmse={comparison['nrmse']:.9g} "
            f"result_correlation={comparison['correlation']:.9g}"
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
