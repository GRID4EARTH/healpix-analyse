"""Binary mathematical morphology for nested HEALPix grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from healpix_geo import nested
from pyproj import Geod


NeighbourhoodMethod = Literal["cell_center", "cone_coverage"]

_WGS84 = Geod(ellps="WGS84")

# Authalic radius of WGS84, used only to convert a physical radius in metres
# into an approximate angular radius for candidate generation.
_WGS84_AUTHALIC_RADIUS_M = 6_371_007.1809


def binary_dilation(
    cells: np.ndarray,
    radius: float,
    reference_level: int,
    *,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    ellipsoid: str = "WGS84",
) -> np.ndarray:
    """Dilate a binary mask represented by active nested HEALPix cells.

    Parameters
    ----------
    cells
        Active nested HEALPix cell IDs.
    radius
        Radius of the structuring neighbourhood, in metres.
    reference_level
        HEALPix reference level.
    neighbourhood
        Definition of the structuring neighbourhood.

        ``"cell_center"``
            Include cells whose centres are within ``radius`` according
            to the ellipsoidal geodesic distance.

        ``"cone_coverage"``
            Include all cells returned by
            :func:`healpix_geo.nested.cone_coverage`.
    ellipsoid
        Reference ellipsoid. Defaults to ``"WGS84"``.

    Returns
    -------
    numpy.ndarray
        Sorted unique active HEALPix cell IDs after dilation.
    """
    cells = _validate_inputs(cells, radius, reference_level)

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = _neighbourhoods(
        cells,
        radius,
        reference_level,
        neighbourhood=neighbourhood,
        ellipsoid=ellipsoid,
    )

    return np.unique(
        np.concatenate(
            [
                cells,
                *neighbourhood_cells,
            ]
        )
    )


def binary_erosion(
    cells: np.ndarray,
    radius: float,
    reference_level: int,
    *,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    ellipsoid: str = "WGS84",
) -> np.ndarray:
    """Erode a binary mask represented by active nested HEALPix cells.

    A cell is retained only when every cell in its structuring
    neighbourhood also belongs to the input mask.

    Parameters
    ----------
    cells
        Active nested HEALPix cell IDs.
    radius
        Radius of the structuring neighbourhood, in metres.
    reference_level
        HEALPix reference level.
    neighbourhood
        Definition of the structuring neighbourhood.

        ``"cell_center"``
            Use WGS84 ellipsoidal cell-centre distances.

        ``"cone_coverage"``
            Use cells returned directly by
            :func:`healpix_geo.nested.cone_coverage`.
    ellipsoid
        Reference ellipsoid. Defaults to ``"WGS84"``.

    Returns
    -------
    numpy.ndarray
        Sorted active HEALPix cell IDs remaining after erosion.
    """
    cells = _validate_inputs(cells, radius, reference_level)

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = _neighbourhoods(
        cells,
        radius,
        reference_level,
        neighbourhood=neighbourhood,
        ellipsoid=ellipsoid,
    )

    active = set(cells.tolist())

    keep = np.fromiter(
        (
            all(int(candidate) in active for candidate in neighbours)
            for neighbours in neighbourhood_cells
        ),
        dtype=bool,
        count=cells.size,
    )

    return cells[keep]


def _neighbourhoods(
    cells: np.ndarray,
    radius: float,
    reference_level: int,
    *,
    neighbourhood: NeighbourhoodMethod,
    ellipsoid: str,
) -> list[np.ndarray]:
    """Return the structuring neighbourhood for each input cell."""
    if neighbourhood not in {"cell_center", "cone_coverage"}:
        raise ValueError(
            "'neighbourhood' must be either "
            "'cell_center' or 'cone_coverage'."
        )

    return [
        _neighbourhood(
            int(cell),
            radius,
            reference_level,
            neighbourhood=neighbourhood,
            ellipsoid=ellipsoid,
        )
        for cell in cells
    ]


def _neighbourhood(
    cell: int,
    radius: float,
    reference_level: int,
    *,
    neighbourhood: NeighbourhoodMethod,
    ellipsoid: str,
) -> np.ndarray:
    """Return the structuring neighbourhood around one HEALPix cell."""
    lon, lat = nested.healpix_to_lonlat(
        np.asarray([cell], dtype=np.uint64),
        reference_level,
        ellipsoid=ellipsoid,
    )

    center = (float(lon[0]), float(lat[0]))

    candidates = _cone_candidates(
        center,
        radius,
        reference_level,
        ellipsoid=ellipsoid,
    )

    if neighbourhood == "cone_coverage":
        return candidates

    return _filter_by_cell_center_distance(
        center,
        candidates,
        radius,
        reference_level,
        ellipsoid=ellipsoid,
    )


def _cone_candidates(
    center: tuple[float, float],
    radius: float,
    reference_level: int,
    *,
    ellipsoid: str,
) -> np.ndarray:
    """Find candidate cells intersecting a circular neighbourhood."""
    radius_degrees = np.rad2deg(
        radius / _WGS84_AUTHALIC_RADIUS_M
    )

    cell_ids, _, _ = nested.cone_coverage(
        center,
        radius_degrees,
        reference_level,
        ellipsoid=ellipsoid,
        flat=True,
    )

    return np.asarray(cell_ids, dtype=np.uint64)


def _filter_by_cell_center_distance(
    center: tuple[float, float],
    candidates: np.ndarray,
    radius: float,
    reference_level: int,
    *,
    ellipsoid: str,
) -> np.ndarray:
    """Filter candidate cells using ellipsoidal centre-to-centre distance."""
    if candidates.size == 0:
        return candidates

    lon, lat = nested.healpix_to_lonlat(
        candidates,
        reference_level,
        ellipsoid=ellipsoid,
    )

    lon0, lat0 = center

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Cell-centre geodesic filtering currently supports "
            "ellipsoid='WGS84' only."
        )

    _, _, distance = _WGS84.inv(
        np.full(lon.shape, lon0, dtype=float),
        np.full(lat.shape, lat0, dtype=float),
        lon,
        lat,
    )

    return candidates[distance <= radius]


def _validate_inputs(
    cells: np.ndarray,
    radius: float,
    reference_level: int,
) -> np.ndarray:
    """Validate and normalize morphology inputs."""
    cells = np.asarray(cells, dtype=np.uint64)

    if cells.ndim != 1:
        raise ValueError("'cells' must be a one-dimensional array.")

    if radius < 0:
        raise ValueError("'radius' must be greater than or equal to zero.")

    if not 0 <= reference_level <= 29:
        raise ValueError(
            "'reference_level' must be between 0 and 29."
        )

    return np.unique(cells)
