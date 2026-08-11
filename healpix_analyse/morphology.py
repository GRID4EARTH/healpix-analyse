"""Binary mathematical morphology for nested HEALPix grids.

This module provides binary dilation and erosion for masks represented
by active nested HEALPix cell IDs.

Unlike morphology on a Cartesian raster, a HEALPix structuring
neighbourhood cannot be represented by a fixed 2-D array. Instead,
the neighbourhood is defined geometrically on the reference ellipsoid.

Binary masks are represented by the nested HEALPix cell IDs of active
cells. Structuring neighbourhoods are defined geometrically rather than
by a fixed two-dimensional kernel.

Two neighbourhood definitions are supported:

``cell_center``
    Include cells whose centres are within ``radius`` metres of the
    target-cell centre, using WGS84 ellipsoidal geodesic distance.
    This is the default and most closely reproduces the centre-distance
    semantics of a disk-shaped raster structuring element.

``cone_coverage``
    Include cells intersecting the circular region using
    :func:`healpix_geo.nested.cone_coverage`. This generally produces
    a larger neighbourhood because boundary cells may be included even
    when their centres lie outside ``radius``.

An optional processing ``domain`` distinguishes inactive cells from
cells outside the spatial processing extent.
The ``cell_center`` definition is the default because it corresponds
most closely to a classical disk-shaped raster structuring element,
where inclusion is determined from pixel-centre distances.
"""

from __future__ import annotations

import numpy as np

from ._neighbourhood import (
    NeighbourhoodMethod,
    build_neighbourhoods,
)



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
            Include a HEALPix cell when the WGS84 geodesic distance
            between its centre and the centre of the target cell is
            less than or equal to ``radius``.

            This is the default and corresponds most closely to the
            centre-distance criterion of a disk-shaped structuring
            element on a regular raster.

        ``"cone_coverage"``
            Include every HEALPix cell intersecting the circular region
            defined by ``radius``, using
            :func:`healpix_geo.nested.cone_coverage`.

            This generally produces a larger neighbourhood because cells
            whose centres lie outside the radius may still intersect the
            circular region.

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
    Examples
    --------
    Dilate a binary HEALPix mask using a 240 m cell-centre
    structuring neighbourhood:

    >>> cells = np.array([0], dtype=np.uint64)
    >>> result = binary_dilation(
    ...     cells,
    ...     radius=240.0,
    ...     refinement_level=17,
    ...     neighbourhood="cell_center",
    ... )

    The alternative coverage-based neighbourhood can be selected with:

    >>> result = binary_dilation(
    ...     cells,
    ...     radius=240.0,
    ...     refinement_level=17,
    ...     neighbourhood="cone_coverage",
    ... )


    """
    cells, domain = _validate_inputs(
        cells,
        radius,
        refinement_level,
        domain=domain,
    )

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = build_neighbourhoods(
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


    Examples
    --------
    Erode a regional binary mask while ignoring cells outside the
    processing domain:

    >>> eroded = binary_erosion(
    ...     cells,
    ...     radius=180.0,
    ...     refinement_level=17,
    ...     neighbourhood="cell_center",
    ...     domain=domain_cells,
    ... )


    """
    cells, domain = _validate_inputs(
        cells,
        radius,
        refinement_level,
        domain=domain,
    )

    if cells.size == 0 or radius == 0:
        return cells.copy()

    neighbourhood_cells = build_neighbourhoods(
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
