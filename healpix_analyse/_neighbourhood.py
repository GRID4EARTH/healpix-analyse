"""Shared HEALPix neighbourhood geometry helpers.

This module centralises the geometric neighbourhood construction used by
binary morphology and generic neighbourhood reductions so that both APIs
share exactly the same spatial semantics.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
from healpix_geo import nested
from pyproj import Geod


NeighbourhoodMethod = Literal[
    "cell_center",
    "cone_coverage",
]

_WGS84 = Geod(ellps="WGS84")
_GEOD_MAX_THREADS = 8
_GEOD_PARALLEL_MIN_PAIRS = 100_000

# Authalic radius of WGS84. Used only to convert a physical radius in metres
# to an approximate angular radius for cone_coverage candidate generation.
_WGS84_AUTHALIC_RADIUS_M = 6_371_007.1809
_WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
_WGS84_FLATTENING = 1.0 / 298.257_223_563


def _geod_thread_count(number_of_pairs: int) -> int:
    """Choose an automatic worker count capped at eight threads."""
    if number_of_pairs < _GEOD_PARALLEL_MIN_PAIRS:
        return 1

    return min(
        _GEOD_MAX_THREADS,
        os.cpu_count() or 1,
        number_of_pairs,
    )


def _wgs84_distance(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Compute exact WGS84 distances, using at most eight threads.

    ``pyproj`` releases the GIL while PROJ evaluates inverse geodesics. Large
    one-dimensional batches can therefore be split across native calls
    without changing the underlying geodesic algorithm or result ordering.
    Each worker owns its ``Geod`` instance to avoid sharing mutable wrapper
    state between threads.
    """
    first_lon = np.asarray(lon1, dtype=np.float64)
    first_lat = np.asarray(lat1, dtype=np.float64)
    second_lon = np.asarray(lon2, dtype=np.float64)
    second_lat = np.asarray(lat2, dtype=np.float64)

    number_of_pairs = first_lon.size
    workers = _geod_thread_count(number_of_pairs)
    if workers == 1:
        return _WGS84.inv(
            first_lon,
            first_lat,
            second_lon,
            second_lat,
        )[2]

    bounds = np.linspace(
        0,
        number_of_pairs,
        workers + 1,
        dtype=np.int64,
    )

    def evaluate(bounds_pair: tuple[int, int]) -> np.ndarray:
        start, stop = bounds_pair
        geod = Geod(ellps="WGS84")
        return geod.inv(
            first_lon[start:stop],
            first_lat[start:stop],
            second_lon[start:stop],
            second_lat[start:stop],
        )[2]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = executor.map(
            evaluate,
            zip(bounds[:-1], bounds[1:], strict=True),
        )
        return np.concatenate(list(chunks))


@dataclass(frozen=True)
class RelativeNeighbourhoodGeometry:
    """Relative geometry of HEALPix neighbours around target cells.

    Arrays use a padded dense representation with shape ``(N, K)`` where
    ``N`` is the number of centre cells and ``K`` is the maximum number of
    represented neighbours. Missing positions use ``neighbour_ids=-1`` and
    ``valid_mask=False``.

    Geographic directions follow the local tangent convention:

    - positive East in ``east_offset_m``
    - positive North in ``north_offset_m``
    - azimuth measured clockwise from geographic North

    The geometry is independent of data values and may therefore be reused
    across multiple variables, Sentinel-2 bands, or processing passes on the
    same HEALPix cells.
    """

    center_ids: np.ndarray
    neighbour_ids: np.ndarray
    valid_mask: np.ndarray
    distance_m: np.ndarray
    azimuth_rad: np.ndarray
    east_offset_m: np.ndarray
    north_offset_m: np.ndarray


@dataclass(frozen=True)
class MetricNeighbourhoodGeometry:
    """Minimal padded geometry for kernels that depend only on distance."""

    center_ids: np.ndarray
    neighbour_ids: np.ndarray
    valid_mask: np.ndarray
    distance_m: np.ndarray


@dataclass(frozen=True)
class CompactMetricNeighbourhoodGeometry:
    """Unpadded distance geometry with CSR-style row offsets."""

    center_ids: np.ndarray
    neighbour_indices: np.ndarray
    row_offsets: np.ndarray
    distance_m: np.ndarray


def _domain_index_dtype(number_of_centers: int) -> type[np.unsignedinteger]:
    """Return the smallest practical dtype for domain-local indices."""
    if number_of_centers <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _wgs84_ecef(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to ECEF metres."""
    longitude = np.deg2rad(longitude_deg)
    latitude = np.deg2rad(latitude_deg)
    eccentricity_squared = _WGS84_FLATTENING * (
        2.0 - _WGS84_FLATTENING
    )
    sin_latitude = np.sin(latitude)
    cos_latitude = np.cos(latitude)
    prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_M / np.sqrt(
        1.0 - eccentricity_squared * sin_latitude * sin_latitude
    )
    return (
        prime_vertical_radius * cos_latitude * np.cos(longitude),
        prime_vertical_radius * cos_latitude * np.sin(longitude),
        prime_vertical_radius
        * (1.0 - eccentricity_squared)
        * sin_latitude,
    )


def validate_neighbourhood(
    neighbourhood: NeighbourhoodMethod,
) -> None:
    """Validate a neighbourhood construction method."""
    if neighbourhood not in {
        "cell_center",
        "cone_coverage",
    }:
        raise ValueError(
            "'neighbourhood' must be either "
            "'cell_center' or 'cone_coverage'."
        )

def validate_ring(
    ring: int,
) -> int:
    """Validate and normalise a topological HEALPix ring count.

    ``ring`` is a topological distance, not a physical distance.

    ``ring=0``
        Contains only the centre cell in the raw HEALPix neighbourhood.

    ``ring=1``
        Immediate HEALPix neighbourhood.

    ``ring=2``
        Immediate neighbours plus the next topological ring.

    Notes
    -----
    The number of valid neighbours must not be assumed to be constant.
    HEALPix contains special topological locations, and at refinement
    level 0 the 12 base pixels have fewer neighbours than ordinary
    higher-resolution cells.

    The requested ring must be a non-negative integer.

    Whether a particular ring can be evaluated at a given refinement level
    is determined by ``healpix-geo``. For example, larger rings may not be
    available at refinement level 0 because they would require repeatedly
    crossing HEALPix base-cell boundaries.
    """
    if isinstance(
        ring,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'ring' must be a non-negative integer."
        )

    if not isinstance(
        ring,
        (int, np.integer),
    ):
        raise TypeError(
            "'ring' must be a non-negative integer."
        )

    ring = int(ring)

    if ring < 0:
        raise ValueError(
            "'ring' must be greater than or equal to zero."
        )

    return ring


def build_ring_neighbourhoods(
    cells: np.ndarray,
    refinement_level: int,
    *,
    ring: int = 1,
    include_self: bool = False,
    num_threads: int = 0,
) -> list[np.ndarray]:
    """Build topological NESTED HEALPix neighbourhoods.

    This helper selects cells by HEALPix topological distance rather than
    by a physical radius.

    ``ring=1`` returns the immediate HEALPix neighbourhood.

    ``ring=2`` additionally includes the next topological ring where that
    operation is supported by ``healpix-geo``.

    The centre cell is returned by ``healpix-geo`` and is removed here by
    default.

    Missing topological positions are represented by ``-1`` by
    ``healpix-geo`` and are removed.

    No fixed neighbour count is assumed.

    In particular, refinement level 0 contains the 12 HEALPix base pixels
    and may have fewer immediate neighbours than ordinary higher-resolution
    cells.

    Notes
    -----
    The positional ordering returned by ``healpix-geo`` must not be
    interpreted as geographic East/North directions.

    Geographic direction must instead be derived from the real relative
    geometry of the HEALPix cell centres.
    """
    ring = validate_ring(ring)

    raw_cells = np.asarray(cells)

    if raw_cells.ndim != 1:
        raise ValueError(
            "'cells' must be a one-dimensional array."
        )

    if raw_cells.dtype == np.bool_ or not np.issubdtype(
        raw_cells.dtype,
        np.integer,
    ):
        raise TypeError(
            "'cells' must contain integer HEALPix cell IDs."
        )

    if np.any(raw_cells < 0):
        raise ValueError(
            "'cells' must contain non-negative HEALPix cell IDs."
        )

    cells_array = raw_cells.astype(
        np.uint64,
        copy=False,
    )

    if cells_array.size == 0:
        return []

    raw = nested.kth_neighbourhood(
        cells_array,
        refinement_level,
        ring,
        num_threads=num_threads,
    )

    raw = np.asarray(
        raw,
        dtype=np.int64,
    )

    neighbourhoods = []

    for center, row in zip(
        cells_array,
        raw,
        strict=True,
    ):
        # healpix-geo uses -1 for missing topological positions.
        valid = row[
            row >= 0
        ].astype(
            np.uint64,
            copy=False,
        )

        if not include_self:
            valid = valid[
                valid != center
            ]

        # The backend ordering has topological meaning, but downstream
        # geographic operators must not interpret that ordering as
        # East/North. For this generic helper we expose only cell IDs.
        valid = np.unique(
            valid
        )

        neighbourhoods.append(
            valid
        )

    return neighbourhoods


def _pad_neighbourhoods(
    neighbourhoods: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert variable-length neighbourhoods to a dense padded matrix.

    Missing positions are represented by ``-1`` in the signed ``int64``
    neighbour matrix and by ``False`` in the accompanying validity mask.
    """
    number_of_centers = len(neighbourhoods)

    if number_of_centers == 0:
        return (
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=bool),
        )

    max_neighbours = max(
        (neighbourhood.size for neighbourhood in neighbourhoods),
        default=0,
    )

    neighbour_ids = np.full(
        (number_of_centers, max_neighbours),
        -1,
        dtype=np.int64,
    )
    valid_mask = np.zeros(
        (number_of_centers, max_neighbours),
        dtype=bool,
    )

    for row_index, neighbourhood in enumerate(neighbourhoods):
        size = neighbourhood.size
        if size == 0:
            continue

        neighbour_ids[row_index, :size] = neighbourhood.astype(
            np.int64,
            copy=False,
        )
        valid_mask[row_index, :size] = True

    return neighbour_ids, valid_mask


def relative_geometry_from_neighbours(
    center_ids: np.ndarray,
    neighbour_ids: np.ndarray,
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
) -> RelativeNeighbourhoodGeometry:
    """Compute geographic relative geometry for known HEALPix neighbours.

    ``neighbour_ids`` is a dense signed integer matrix of shape ``(N, K)``.
    Valid entries are NESTED HEALPix cell IDs and padded positions are ``-1``.

    This function deliberately separates neighbour selection from geometry.
    Neighbours may therefore originate from a topological ring, a physical
    radius, or another future candidate-selection strategy.

    For every valid centre-neighbour pair, WGS84 geodesic distance and
    forward azimuth are converted to local tangent offsets using::

        East  = distance * sin(azimuth)
        North = distance * cos(azimuth)

    HEALPix positional neighbour labels are never used to define geographic
    East or North.
    """
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Relative neighbour geometry currently supports "
            "ellipsoid='WGS84' only."
        )

    centers = np.asarray(center_ids)
    if centers.ndim != 1:
        raise ValueError("'center_ids' must be a one-dimensional array.")
    if centers.dtype == np.bool_ or not np.issubdtype(centers.dtype, np.integer):
        raise TypeError("'center_ids' must contain integer HEALPix cell IDs.")
    if np.any(centers < 0):
        raise ValueError("'center_ids' must contain non-negative HEALPix cell IDs.")
    centers = centers.astype(np.uint64, copy=False)

    neighbours = np.asarray(neighbour_ids)
    if neighbours.ndim != 2:
        raise ValueError("'neighbour_ids' must be a two-dimensional padded array.")
    if neighbours.shape[0] != centers.size:
        raise ValueError(
            "The first dimension of 'neighbour_ids' must match "
            "the number of 'center_ids'."
        )
    if neighbours.dtype == np.bool_ or not np.issubdtype(neighbours.dtype, np.integer):
        raise TypeError(
            "'neighbour_ids' must contain integer HEALPix cell IDs or -1 padding."
        )
    neighbours = neighbours.astype(np.int64, copy=False)
    if np.any(neighbours < -1):
        raise ValueError(
            "'neighbour_ids' may contain only valid cell IDs or -1 padding."
        )

    number_of_pixels = 12 * 4**refinement_level
    if np.any(centers >= number_of_pixels):
        raise ValueError(
            "'center_ids' contains cell IDs outside the requested refinement level."
        )

    valid_mask = neighbours >= 0
    if np.any(neighbours[valid_mask] >= number_of_pixels):
        raise ValueError(
            "'neighbour_ids' contains cell IDs outside the requested refinement level."
        )

    shape = neighbours.shape
    distance_m = np.full(shape, np.nan, dtype=np.float64)
    azimuth_rad = np.full(shape, np.nan, dtype=np.float64)
    east_offset_m = np.full(shape, np.nan, dtype=np.float64)
    north_offset_m = np.full(shape, np.nan, dtype=np.float64)

    if not np.any(valid_mask):
        return RelativeNeighbourhoodGeometry(
            center_ids=centers.copy(),
            neighbour_ids=neighbours.copy(),
            valid_mask=valid_mask,
            distance_m=distance_m,
            azimuth_rad=azimuth_rad,
            east_offset_m=east_offset_m,
            north_offset_m=north_offset_m,
        )

    # Vectorise geometry over every valid centre-neighbour pair. This avoids
    # a Python geodesic call for every individual HEALPix cell.
    row_indices, column_indices = np.nonzero(valid_mask)
    flat_center_ids = centers[row_indices]
    flat_neighbour_ids = neighbours[row_indices, column_indices].astype(
        np.uint64,
        copy=False,
    )

    center_lon, center_lat = nested.healpix_to_lonlat(
        flat_center_ids,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    neighbour_lon, neighbour_lat = nested.healpix_to_lonlat(
        flat_neighbour_ids,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    forward_azimuth_deg, _, flat_distance_m = _WGS84.inv(
        center_lon,
        center_lat,
        neighbour_lon,
        neighbour_lat,
    )
    flat_azimuth_rad = np.deg2rad(forward_azimuth_deg)
    flat_east_offset_m = flat_distance_m * np.sin(flat_azimuth_rad)
    flat_north_offset_m = flat_distance_m * np.cos(flat_azimuth_rad)

    distance_m[row_indices, column_indices] = flat_distance_m
    azimuth_rad[row_indices, column_indices] = flat_azimuth_rad
    east_offset_m[row_indices, column_indices] = flat_east_offset_m
    north_offset_m[row_indices, column_indices] = flat_north_offset_m

    return RelativeNeighbourhoodGeometry(
        center_ids=centers.copy(),
        neighbour_ids=neighbours.copy(),
        valid_mask=valid_mask,
        distance_m=distance_m,
        azimuth_rad=azimuth_rad,
        east_offset_m=east_offset_m,
        north_offset_m=north_offset_m,
    )


def build_relative_geometry(
    cells: np.ndarray,
    refinement_level: int,
    *,
    ring: int = 1,
    ellipsoid: str = "WGS84",
    num_threads: int = 0,
) -> RelativeNeighbourhoodGeometry:
    """Build geographic geometry for topological HEALPix neighbours.

    This convenience layer combines topological candidate discovery via
    :func:`build_ring_neighbourhoods` with WGS84 relative geometry via
    :func:`relative_geometry_from_neighbours`.

    The centre cell is excluded. For scalar-field gradients, ``ring=1`` is
    the intended default because the traced S2MSI gradient operations are
    immediate/local fixed-window operators rather than physical-radius
    operators.
    """
    neighbourhoods = build_ring_neighbourhoods(
        cells,
        refinement_level,
        ring=ring,
        include_self=False,
        num_threads=num_threads,
    )
    padded_neighbour_ids, _ = _pad_neighbourhoods(neighbourhoods)
    return relative_geometry_from_neighbours(
        cells,
        padded_neighbour_ids,
        refinement_level,
        ellipsoid=ellipsoid,
    )

def build_neighbourhoods(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    neighbourhood: NeighbourhoodMethod,
    ellipsoid: str,
) -> list[np.ndarray]:
    """Build a geometric neighbourhood for each HEALPix cell."""
    validate_neighbourhood(neighbourhood)

    cells_array = np.asarray(cells, dtype=np.uint64)
    if cells_array.size == 0:
        return []

    # Coordinate conversion has appreciable fixed overhead in healpix-geo.
    # Convert all centres in one call, then retain the exact cone-coverage
    # candidate generation used by the original implementation.
    center_lon, center_lat = nested.healpix_to_lonlat(
        cells_array,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    centers = list(zip(center_lon, center_lat, strict=True))
    candidates = [
        _cone_candidates(
            (float(lon), float(lat)),
            radius,
            refinement_level,
            ellipsoid=ellipsoid,
        )
        for lon, lat in centers
    ]

    if neighbourhood == "cone_coverage":
        return candidates

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Cell-centre geodesic filtering currently supports "
            "ellipsoid='WGS84' only."
        )

    counts = np.fromiter(
        (candidate.size for candidate in candidates),
        dtype=np.int64,
        count=cells_array.size,
    )
    if not np.any(counts):
        return candidates

    flat_candidates = np.concatenate(candidates)

    # Candidate IDs repeat heavily between adjacent centres. Resolve each
    # distinct HEALPix centre once, then expand the coordinates without
    # changing the candidate order supplied by cone_coverage.
    unique_candidates, inverse = np.unique(
        flat_candidates,
        return_inverse=True,
    )
    unique_lon, unique_lat = nested.healpix_to_lonlat(
        unique_candidates,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    candidate_lon = unique_lon[inverse]
    candidate_lat = unique_lat[inverse]
    repeated_lon = np.repeat(
        np.asarray(center_lon, dtype=np.float64),
        counts,
    )
    repeated_lat = np.repeat(
        np.asarray(center_lat, dtype=np.float64),
        counts,
    )

    distance = _wgs84_distance(
        repeated_lon,
        repeated_lat,
        candidate_lon,
        candidate_lat,
    )
    within_radius = distance <= radius

    offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(counts))
    )
    return [
        flat_candidates[start:stop][within_radius[start:stop]]
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]


def _neighbourhood(
    cell: int,
    radius: float,
    refinement_level: int,
    *,
    neighbourhood: NeighbourhoodMethod,
    ellipsoid: str,
) -> np.ndarray:
    """Return the geometric neighbourhood around one HEALPix cell."""
    lon, lat = nested.healpix_to_lonlat(
        np.asarray([cell], dtype=np.uint64),
        refinement_level,
        ellipsoid=ellipsoid,
    )

    center = (float(lon[0]), float(lat[0]))

    candidates = _cone_candidates(
        center,
        radius,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    if neighbourhood == "cone_coverage":
        return candidates

    return _filter_by_cell_center_distance(
        center,
        candidates,
        radius,
        refinement_level,
        ellipsoid=ellipsoid,
    )


def _cone_candidates(
    center: tuple[float, float],
    radius: float,
    refinement_level: int,
    *,
    ellipsoid: str,
) -> np.ndarray:
    """Find candidate cells intersecting a circular neighbourhood."""
    radius_degrees = np.rad2deg(
        radius / _WGS84_AUTHALIC_RADIUS_M
    )

    # healpix-geo currently documents this positional argument as `depth`.
    # Pass refinement_level positionally to remain compatible while exposing
    # CF-aligned terminology in healpix-analyse.
    cell_ids, _, _ = nested.cone_coverage(
        center,
        radius_degrees,
        refinement_level,
        ellipsoid=ellipsoid,
        flat=True,
    )

    return np.asarray(cell_ids, dtype=np.uint64)


def _filter_by_cell_center_distance(
    center: tuple[float, float],
    candidates: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    ellipsoid: str,
) -> np.ndarray:
    """Filter candidates using ellipsoidal centre-to-centre distance."""
    if candidates.size == 0:
        return candidates

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Cell-centre geodesic filtering currently supports "
            "ellipsoid='WGS84' only."
        )

    lon, lat = nested.healpix_to_lonlat(
        candidates,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    lon0, lat0 = center

    _, _, distance = _WGS84.inv(
        np.full(lon.shape, lon0, dtype=float),
        np.full(lat.shape, lat0, dtype=float),
        lon,
        lat,
    )

    return candidates[distance <= radius]

def relative_geometry_from_neighbourhoods(
    center_ids: np.ndarray,
    neighbourhoods: list[np.ndarray],
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
) -> RelativeNeighbourhoodGeometry:
    """Compute relative geometry from variable-length neighbour lists.

    This convenience helper converts variable-length HEALPix neighbourhoods
    to the padded representation required by
    :func:`relative_geometry_from_neighbours`.

    It deliberately keeps neighbour selection separate from geometry so that
    physical-radius, topological-ring, and future candidate-selection methods
    can all share the same WGS84 distance and azimuth implementation.
    """
    padded_neighbour_ids, _ = _pad_neighbourhoods(
        neighbourhoods,
    )

    return relative_geometry_from_neighbours(
        center_ids,
        padded_neighbour_ids,
        refinement_level,
        ellipsoid=ellipsoid,
    )


def metric_geometry_from_neighbourhoods(
    center_ids: np.ndarray,
    neighbourhoods: list[np.ndarray],
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
) -> MetricNeighbourhoodGeometry:
    """Compute only the WGS84 distances required by radial kernels."""
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Metric neighbour geometry currently supports "
            "ellipsoid='WGS84' only."
        )

    centers = np.asarray(center_ids, dtype=np.uint64)
    neighbour_ids, valid_mask = _pad_neighbourhoods(neighbourhoods)
    distance_m = np.full(neighbour_ids.shape, np.nan, dtype=np.float64)

    if not np.any(valid_mask):
        return MetricNeighbourhoodGeometry(
            centers.copy(),
            neighbour_ids,
            valid_mask,
            distance_m,
        )

    row_indices, column_indices = np.nonzero(valid_mask)
    valid_neighbours = neighbour_ids[
        row_indices,
        column_indices,
    ].astype(np.uint64, copy=False)

    # Resolve each HEALPix centre once. Both arrays contain many repeated IDs
    # on a dense patch, so this reduces conversion work and temporary memory.
    all_ids = np.concatenate((centers, valid_neighbours))
    unique_ids, inverse = np.unique(all_ids, return_inverse=True)
    unique_lon, unique_lat = nested.healpix_to_lonlat(
        unique_ids,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    center_inverse = inverse[:centers.size]
    neighbour_inverse = inverse[centers.size:]

    flat_distance_m = _wgs84_distance(
        unique_lon[center_inverse[row_indices]],
        unique_lat[center_inverse[row_indices]],
        unique_lon[neighbour_inverse],
        unique_lat[neighbour_inverse],
    )
    distance_m[row_indices, column_indices] = flat_distance_m

    return MetricNeighbourhoodGeometry(
        centers.copy(),
        neighbour_ids,
        valid_mask,
        distance_m,
    )


def build_metric_neighbourhood_geometry(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
) -> CompactMetricNeighbourhoodGeometry:
    """Build domain-restricted metric geometry with one distance pass.

    Candidate cells outside ``cells`` are discarded before coordinate lookup
    and inverse geodesy. Coordinates are looked up from the already converted
    domain centres, and the distances used for the radius cutoff are retained
    directly in the returned geometry.
    """
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Metric neighbour geometry currently supports "
            "ellipsoid='WGS84' only."
        )

    centers = np.asarray(cells, dtype=np.uint64)
    number_of_centers = centers.size
    if number_of_centers == 0:
        return CompactMetricNeighbourhoodGeometry(
            centers.copy(),
            np.empty(0, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    if radius == 0.0:
        return CompactMetricNeighbourhoodGeometry(
            centers.copy(),
            np.arange(
                number_of_centers,
                dtype=_domain_index_dtype(number_of_centers),
            ),
            np.arange(number_of_centers + 1, dtype=np.int64),
            np.zeros(number_of_centers, dtype=np.float64),
        )

    center_lon, center_lat = nested.healpix_to_lonlat(
        centers,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    candidates = [
        _cone_candidates(
            (float(lon), float(lat)),
            radius,
            refinement_level,
            ellipsoid=ellipsoid,
        )
        for lon, lat in zip(center_lon, center_lat, strict=True)
    ]
    candidate_counts = np.fromiter(
        (candidate.size for candidate in candidates),
        dtype=np.int64,
        count=number_of_centers,
    )
    if not np.any(candidate_counts):
        return CompactMetricNeighbourhoodGeometry(
            centers.copy(),
            np.empty(0, dtype=np.int64),
            np.zeros(number_of_centers + 1, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    flat_candidates = np.concatenate(candidates)
    candidate_rows = np.repeat(
        np.arange(number_of_centers, dtype=np.int64),
        candidate_counts,
    )

    # All contributing neighbours must belong to the processing domain.
    # Search the unique domain rather than sorting every repeated candidate.
    domain_order = np.argsort(centers)
    sorted_domain = centers[domain_order]
    positions = np.searchsorted(sorted_domain, flat_candidates)
    in_domain = positions < number_of_centers
    possible = np.flatnonzero(in_domain)
    in_domain[possible] = (
        sorted_domain[positions[possible]] == flat_candidates[possible]
    )

    flat_candidates = flat_candidates[in_domain]
    candidate_rows = candidate_rows[in_domain]
    domain_positions = domain_order[positions[in_domain]]

    distance_m = _wgs84_distance(
        np.asarray(center_lon)[candidate_rows],
        np.asarray(center_lat)[candidate_rows],
        np.asarray(center_lon)[domain_positions],
        np.asarray(center_lat)[domain_positions],
    )
    within_radius = distance_m <= radius
    neighbour_rows = candidate_rows[within_radius]
    neighbour_indices = domain_positions[within_radius]
    flat_distance_m = distance_m[within_radius]

    row_counts = np.bincount(
        neighbour_rows,
        minlength=number_of_centers,
    )
    row_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(row_counts))
    )

    return CompactMetricNeighbourhoodGeometry(
        centers.copy(),
        neighbour_indices.astype(
            _domain_index_dtype(number_of_centers),
            copy=False,
        ),
        row_offsets,
        flat_distance_m,
    )


def build_metric_geometry_from_vectorized_ring(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    ring: int,
    ellipsoid: str = "WGS84",
    num_threads: int = 8,
    max_candidate_pairs: int = 4_000_000,
) -> CompactMetricNeighbourhoodGeometry:
    """Experimentally build exact-cutoff geometry from batched ring candidates.

    ``kth_neighbourhood`` generates candidates for many centres in one native
    call. Domain filtering and an ECEF chord-distance lower bound remove cheap
    false positives before the surviving pairs receive the same exact WGS84
    inverse-geodesic cutoff as :func:`build_metric_neighbourhood_geometry`.

    This helper is not used by the public filter path. The caller must choose
    a ``ring`` that contains every cell centre within ``radius`` for all
    requested locations. The exact cutoff removes false positives, but it
    cannot recover a true neighbour omitted by an insufficient topological
    ring. ``max_candidate_pairs`` bounds each temporary candidate matrix.
    """
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Metric neighbour geometry currently supports "
            "ellipsoid='WGS84' only."
        )
    ring = validate_ring(ring)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("'radius' must be finite and non-negative.")
    if isinstance(num_threads, (bool, np.bool_)) or not isinstance(
        num_threads,
        (int, np.integer),
    ):
        raise TypeError("'num_threads' must be a positive integer.")
    if int(num_threads) < 1:
        raise ValueError("'num_threads' must be a positive integer.")
    if (
        isinstance(max_candidate_pairs, (bool, np.bool_))
        or not isinstance(max_candidate_pairs, (int, np.integer))
    ):
        raise TypeError("'max_candidate_pairs' must be a positive integer.")
    if int(max_candidate_pairs) < 1:
        raise ValueError("'max_candidate_pairs' must be a positive integer.")

    centers = np.asarray(cells)
    if centers.ndim != 1:
        raise ValueError("'cells' must be one-dimensional.")
    if centers.dtype == np.bool_ or not np.issubdtype(
        centers.dtype,
        np.integer,
    ):
        raise TypeError("'cells' must contain integer HEALPix IDs.")
    if np.any(centers < 0):
        raise ValueError("'cells' must contain non-negative HEALPix IDs.")
    centers = centers.astype(np.uint64, copy=False)
    if np.unique(centers).size != centers.size:
        raise ValueError("'cells' must not contain duplicate HEALPix IDs.")

    number_of_centers = centers.size
    index_dtype = _domain_index_dtype(number_of_centers)
    if number_of_centers == 0:
        return CompactMetricNeighbourhoodGeometry(
            centers.copy(),
            np.empty(0, dtype=index_dtype),
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    if radius == 0.0:
        return CompactMetricNeighbourhoodGeometry(
            centers.copy(),
            np.arange(number_of_centers, dtype=index_dtype),
            np.arange(number_of_centers + 1, dtype=np.int64),
            np.zeros(number_of_centers, dtype=np.float64),
        )

    longitude, latitude = nested.healpix_to_lonlat(
        centers,
        refinement_level,
        ellipsoid=ellipsoid,
    )
    ecef_x, ecef_y, ecef_z = _wgs84_ecef(longitude, latitude)
    domain_order = np.argsort(centers)
    sorted_domain = centers[domain_order]
    candidate_width = (2 * ring + 1) ** 2
    chunk_size = max(1, int(max_candidate_pairs) // candidate_width)
    workers = min(_GEOD_MAX_THREADS, int(num_threads))
    radius_with_roundoff = radius * (1.0 + 1.0e-12)
    chord_limit_squared = radius_with_roundoff**2

    row_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []

    for start in range(0, number_of_centers, chunk_size):
        stop = min(start + chunk_size, number_of_centers)
        raw = nested.kth_neighbourhood(
            centers[start:stop],
            refinement_level,
            ring,
            num_threads=workers,
        )
        raw = np.asarray(raw)
        flat_candidates = raw.reshape(-1)
        candidate_rows = np.repeat(
            np.arange(start, stop, dtype=np.int64),
            raw.shape[1],
        )
        valid = flat_candidates >= 0
        flat_candidates = flat_candidates[valid].astype(
            np.uint64,
            copy=False,
        )
        candidate_rows = candidate_rows[valid]

        positions = np.searchsorted(sorted_domain, flat_candidates)
        in_domain = positions < number_of_centers
        possible = np.flatnonzero(in_domain)
        in_domain[possible] = (
            sorted_domain[positions[possible]]
            == flat_candidates[possible]
        )
        candidate_rows = candidate_rows[in_domain]
        domain_positions = domain_order[positions[in_domain]]

        delta_x = ecef_x[candidate_rows] - ecef_x[domain_positions]
        delta_y = ecef_y[candidate_rows] - ecef_y[domain_positions]
        delta_z = ecef_z[candidate_rows] - ecef_z[domain_positions]
        chord_candidate = (
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
            <= chord_limit_squared
        )
        candidate_rows = candidate_rows[chord_candidate]
        domain_positions = domain_positions[chord_candidate]

        distance_m = _wgs84_distance(
            longitude[candidate_rows],
            latitude[candidate_rows],
            longitude[domain_positions],
            latitude[domain_positions],
        )
        within_radius = distance_m <= radius
        row_parts.append(candidate_rows[within_radius])
        index_parts.append(domain_positions[within_radius])
        distance_parts.append(distance_m[within_radius])

    neighbour_rows = np.concatenate(row_parts)
    neighbour_indices = np.concatenate(index_parts)
    flat_distance_m = np.concatenate(distance_parts)
    row_counts = np.bincount(
        neighbour_rows,
        minlength=number_of_centers,
    )
    row_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(row_counts))
    )
    return CompactMetricNeighbourhoodGeometry(
        centers.copy(),
        neighbour_indices.astype(index_dtype, copy=False),
        row_offsets,
        flat_distance_m,
    )

__all__ = [
    "NeighbourhoodMethod",
    "MetricNeighbourhoodGeometry",
    "CompactMetricNeighbourhoodGeometry",
    "RelativeNeighbourhoodGeometry",
    "build_neighbourhoods",
    "build_relative_geometry",
    "build_ring_neighbourhoods",
    "relative_geometry_from_neighbours",
    "validate_neighbourhood",
    "validate_ring",
    "relative_geometry_from_neighbourhoods",
    "metric_geometry_from_neighbourhoods",
    "build_metric_neighbourhood_geometry",
    "build_metric_geometry_from_vectorized_ring",
]
