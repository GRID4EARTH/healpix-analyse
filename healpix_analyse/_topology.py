"""Private HEALPix topology helpers.

This module provides topology primitives used internally by
``healpix-analyse``.

Only NESTED HEALPix indexing is supported.

The current implementation uses ``healpy.get_all_neighbours`` as a
temporary compatibility backend.

This is intentionally isolated in this private module so that it can
later be replaced by direction-aware neighbour access from
``healpix-geo`` / CDSHEALPix without changing any public
``healpix-analyse`` API.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


Connectivity = Literal["edge", "edge_or_vertex"]


# healpy.get_all_neighbours() directional order:
#
#   0: SW
#   1: W
#   2: NW
#   3: N
#   4: NE
#   5: E
#   6: SE
#   7: S
#
# For a HEALPix quadrilateral, the ordinal directions
#
#   SW, NW, NE, SE
#
# correspond to edge-sharing neighbours.
_EDGE_INDICES = np.asarray([0, 2, 4, 6], dtype=np.intp)


def _validate_refinement_level(refinement_level: int) -> int:
    """Validate a NESTED HEALPix refinement level."""

    if isinstance(refinement_level, bool) or not isinstance(
        refinement_level,
        (int, np.integer),
    ):
        raise TypeError("refinement_level must be an integer")

    refinement_level = int(refinement_level)

    # HEALPix NESTED cell identifiers fit in signed 64-bit integers
    # through refinement level 29.
    if not 0 <= refinement_level <= 29:
        raise ValueError("refinement_level must be in [0, 29]")

    return refinement_level


def _as_cell_ids(cell_ids: ArrayLike) -> NDArray[np.uint64]:
    """Convert input cell identifiers to a validated 1-D uint64 array."""

    cells = np.asarray(cell_ids)

    if cells.ndim == 0:
        cells = cells.reshape(1)

    if cells.ndim != 1:
        raise ValueError("cell_ids must be a one-dimensional array")

    if not np.issubdtype(cells.dtype, np.integer):
        raise TypeError("cell_ids must contain integers")

    if np.issubdtype(cells.dtype, np.signedinteger):
        if np.any(cells < 0):
            raise ValueError("cell_ids must be non-negative")

    return cells.astype(np.uint64, copy=False)


def _npix(refinement_level: int) -> int:
    """Return the number of HEALPix cells at a refinement level."""

    nside = 1 << refinement_level
    return 12 * nside * nside


def _healpy_all_neighbours(
    cells: NDArray[np.uint64],
    refinement_level: int,
) -> NDArray[np.int64]:
    """Return healpy's eight directional NESTED neighbour positions.

    Notes
    -----
    ``healpy`` is a temporary compatibility backend only.

    It should be replaced by direction-aware neighbour access from
    ``healpix-geo`` / CDSHEALPix once that API is available.
    """

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "HEALPix topology currently requires healpy as a temporary "
            "compatibility backend. This dependency is intended to be "
            "removed once direction-aware neighbours are exposed by "
            "healpix-geo."
        ) from exc

    nside = 1 << refinement_level

    neighbours = np.asarray(
        hp.get_all_neighbours(
            nside,
            cells.astype(np.int64, copy=False),
            nest=True,
        ),
        dtype=np.int64,
    )

    # For an array input, healpy documents shape (8, N).
    #
    # Keep this scalar normalization as a defensive measure in case this
    # private helper is ever called with a single scalar-like value.
    if neighbours.ndim == 1:
        neighbours = neighbours.reshape(8, 1)

    expected_shape = (8, cells.size)

    if neighbours.shape != expected_shape:
        raise RuntimeError(
            "Unexpected shape returned by healpy.get_all_neighbours: "
            f"{neighbours.shape}; expected {expected_shape}"
        )

    return neighbours


def nested_neighbours(
    cell_ids: ArrayLike,
    refinement_level: int,
    *,
    connectivity: Connectivity = "edge_or_vertex",
) -> NDArray[np.int64]:
    """Return immediate neighbours of NESTED HEALPix cells.

    Parameters
    ----------
    cell_ids
        One-dimensional array of NESTED HEALPix cell identifiers.

    refinement_level
        HEALPix refinement level, with
        ``nside = 2**refinement_level``.

    connectivity
        Connectivity definition.

        ``"edge"``
            Return only the four edge-sharing HEALPix neighbours.

        ``"edge_or_vertex"``
            Return all immediate HEALPix neighbours that share either an
            edge or a vertex.

    Returns
    -------
    numpy.ndarray
        For ``connectivity="edge"`` the output shape is ``(N, 4)`` and
        the deterministic direction order is::

            SW, NW, NE, SE

        For ``connectivity="edge_or_vertex"`` the output shape is
        ``(N, 8)`` and the deterministic direction order is::

            SW, W, NW, N, NE, E, SE, S

        Missing topological neighbour positions are represented by ``-1``.

    Notes
    -----
    Only NESTED HEALPix indexing is supported.

    This is private implementation infrastructure.

    ``healpy`` is intentionally used only as a temporary topology backend.
    The public ``healpix-analyse`` API must not depend on healpy-specific
    implementation details.
    """

    refinement_level = _validate_refinement_level(refinement_level)
    cells = _as_cell_ids(cell_ids)

    if connectivity not in ("edge", "edge_or_vertex"):
        raise ValueError(
            "connectivity must be 'edge' or 'edge_or_vertex'"
        )

    if cells.size == 0:
        width = 4 if connectivity == "edge" else 8
        return np.empty((0, width), dtype=np.int64)

    npix = _npix(refinement_level)

    if np.any(cells >= npix):
        raise ValueError(
            "cell_ids contains an identifier outside the valid range "
            f"[0, {npix}) for refinement_level={refinement_level}"
        )

    neighbours = _healpy_all_neighbours(
        cells,
        refinement_level,
    )

    if connectivity == "edge":
        neighbours = neighbours[_EDGE_INDICES]

    # Internal representation returned to callers is one row per input
    # HEALPix cell.
    return neighbours.T.copy()


def nested_edge_neighbours(
    cell_ids: ArrayLike,
    refinement_level: int,
) -> NDArray[np.int64]:
    """Return the four edge-sharing neighbours of NESTED HEALPix cells.

    Parameters
    ----------
    cell_ids
        One-dimensional array of NESTED HEALPix cell identifiers.

    refinement_level
        HEALPix refinement level.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 4)`` with neighbour direction order::

            SW, NW, NE, SE

    Notes
    -----
    This helper is private.

    It currently uses healpy indirectly as a temporary implementation.
    """

    return nested_neighbours(
        cell_ids,
        refinement_level,
        connectivity="edge",
    )
