"""Shared HEALPix neighbourhood geometry helpers.

This module centralises the geometric neighbourhood construction used by
binary morphology and generic neighbourhood reductions so that both APIs
share exactly the same spatial semantics.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from healpix_geo import nested
from pyproj import Geod


NeighbourhoodMethod = Literal[
    "cell_center",
    "cone_coverage",
]

_WGS84 = Geod(ellps="WGS84")

# Authalic radius of WGS84. Used only to convert a physical radius in metres
# to an approximate angular radius for cone_coverage candidate generation.
_WGS84_AUTHALIC_RADIUS_M = 6_371_007.1809


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

    return [
        _neighbourhood(
            int(cell),
            radius,
            refinement_level,
            neighbourhood=neighbourhood,
            ellipsoid=ellipsoid,
        )
        for cell in cells
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


__all__ = [
    "NeighbourhoodMethod",
    "build_neighbourhoods",
    "validate_neighbourhood",
]
