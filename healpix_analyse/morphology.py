"""Binary mathematical morphology for nested HEALPix grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from healpix_geo import nested
from pyproj import Geod


NeighbourhoodMethod = Literal["cell_center", "cone_coverage"]

_WGS84 = Geod(ellps="WGS84")

# Authalic radius of WGS84.
# Used only to convert a physical radius in metres to an approximate
# angular radius for cone_coverage candidate generation.
_WGS84_AUTHALIC_RADIUS_M = 6_371_007.1809


def binary_dilation(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> np.ndarray:
    """Dilate a binary mask represented by active nested HEALPix cells.

    Parameters
    ----------
    cells
        Active nested HEALPix cell IDs.
    radius
        Radius of the structuring neighbourhood, in metres.
    refinement_level
        HEALPix refinement level.
    neighbourhood
        Definition of the structuring neighbourhood.

        ``"cell_center"``
            Include cells whose centres are within ``radius`` according
            to the ellipsoidal geodesic distance.

        ``"cone_coverage"``
            Include all cells returned by
            :func:`healpix_geo.nested.cone_coverage`.

    domain
        Optional HEALPix processing domain.

        When provided, output cells are restricted to this domain.
        Cells outside the domain are considered outside the processing
        extent rather than inactive mask cells.

    ellipsoid
        Reference ellipsoid. Defaults to ``"WGS84"``.

    Returns
    -------
    numpy.ndarray
        Sorted unique active HEALPix cell IDs after dilation.
    """
    cells, domain = _validate_inputs(
        cells,
        radius,
        refinement_level,
        domain=domain,
    )

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = _neighbourhoods(
        cells,
        radius,
        refinement_level,
        neighbourhood=neighbourhood,
        ellipsoid=ellipsoid,
    )

    result = np.unique(
        np.concatenate(
            [
                cells,
                *neighbourhood_cells,
            ]
        )
    )

    if domain is not None:
        result = result[np.isin(result, domain)]

    return result


def binary_erosion(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> np.ndarray:
    """Erode a binary mask represented by active nested HEALPix cells.

    A cell is retained only when all cells in its structuring neighbourhood
    that belong to the processing domain are active.

    Parameters
    ----------
    cells
        Active nested HEALPix cell IDs.
    radius
        Radius of the structuring neighbourhood, in metres.
    refinement_level
        HEALPix refinement level.
    neighbourhood
        Definition of the structuring neighbourhood.

        ``"cell_center"``
            Use ellipsoidal cell-centre distances.

        ``"cone_coverage"``
            Use cells returned directly by
            :func:`healpix_geo.nested.cone_coverage`.

    domain
        Optional HEALPix processing domain.

        When provided, structuring-neighbourhood cells outside the domain
        are ignored rather than treated as inactive.

    ellipsoid
        Reference ellipsoid. Defaults to ``"WGS84"``.

    Returns
    -------
    numpy.ndarray
        Sorted active HEALPix cell IDs remaining after erosion.
    """
    cells, domain = _validate_inputs(
        cells,
        radius,
        refinement_level,
        domain=domain,
    )

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = _neighbourhoods(
        cells,
        radius,
        refinement_level,
        neighbourhood=neighbourhood,
        ellipsoid=ellipsoid,
    )

    active = set(cells.tolist())

    if domain is None:
        keep = np.fromiter(
            (
                all(int(candidate) in active for candidate in neighbours)
                for neighbours in neighbourhood_cells
            ),
            dtype=bool,
            count=cells.size,
        )
    else:
        domain_set = set(domain.tolist())

        keep = np.fromiter(
            (
                all(
                    int(candidate) in active
                    for candidate in neighbours
                    if int(candidate) in domain_set
                )
                for neighbours in neighbourhood_cells
            ),
            dtype=bool,
            count=cells.size,
        )

    return cells[keep]


def _neighbourhoods(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    neighbourhood: NeighbourhoodMethod,
    ellipsoid: str,
) -> list[np.ndarray]:
    """Return the structuring neighbourhood for each input cell."""
    _validate_neighbourhood(neighbourhood)

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
    """Return the structuring neighbourhood around one HEALPix cell."""
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
    # We pass refinement_level positionally to remain compatible while
    # exposing CF-aligned terminology in healpix-analyse.
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


def _validate_neighbourhood(
    neighbourhood: NeighbourhoodMethod,
) -> None:
    """Validate the neighbourhood method."""
    if neighbourhood not in {"cell_center", "cone_coverage"}:
        raise ValueError(
            "'neighbourhood' must be either "
            "'cell_center' or 'cone_coverage'."
        )


def _validate_inputs(
    cells: np.ndarray,
    radius: float,
    refinement_level: int,
    *,
    domain: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and normalize morphology inputs."""
    cells = np.asarray(cells, dtype=np.uint64)

    if cells.ndim != 1:
        raise ValueError("'cells' must be a one-dimensional array.")

    if radius < 0:
        raise ValueError("'radius' must be greater than or equal to zero.")

    if not 0 <= refinement_level <= 29:
        raise ValueError(
            "'refinement_level' must be between 0 and 29."
        )

    cells = np.unique(cells)

    if domain is None:
        return cells, None

    domain = np.asarray(domain, dtype=np.uint64)

    if domain.ndim != 1:
        raise ValueError("'domain' must be a one-dimensional array.")

    domain = np.unique(domain)

    if not np.all(np.isin(cells, domain)):
        raise ValueError(
            "All active 'cells' must belong to 'domain'."
        )

    return cells, domain
