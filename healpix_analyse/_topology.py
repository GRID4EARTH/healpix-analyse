"""Private HEALPix topology helpers.

This module contains immediate-neighbour topology used internally by
``healpix-analyse``.

Only NESTED HEALPix indexing is currently supported.

Backend
-------
Immediate topology is provided by the direction-preserving neighbour API
in ``healpix-geo``, backed by CDSHEALPix.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from healpix_geo import nested as healpix_geo_nested
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Public type used internally by connected-component operations
# ---------------------------------------------------------------------------

Connectivity = Literal[
    "edge",
    "edge_or_vertex",
]


# ---------------------------------------------------------------------------
# HEALPix directional convention
# ---------------------------------------------------------------------------
#
# healpix_geo.nested.neighbours(..., connectivity="edge_or_vertex") returns
# directions in this order:
#
#     0: SW
#     1: W
#     2: NW
#     3: N
#     4: NE
#     5: E
#     6: SE
#     7: S
#
# A HEALPix cell is topologically a quadrilateral. The ordinal directions
#
#     SW, NW, NE, SE
#
# correspond to cells sharing one of its four edges.
#
# The remaining cardinal directions
#
#     W, N, E, S
#
# correspond to additional vertex-touching neighbours.
#
# Therefore:
#
#     edge             -> SW, NW, NE, SE
#     edge_or_vertex   -> all available immediate neighbours
#
# This provides the HEALPix analogues of Cartesian 4-connectivity and
# 8-connectivity respectively.
# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_refinement_level(
    refinement_level: int,
) -> int:
    """Validate a NESTED HEALPix refinement level."""

    if isinstance(refinement_level, bool) or not isinstance(
        refinement_level,
        (int, np.integer),
    ):
        raise TypeError(
            "refinement_level must be an integer"
        )

    refinement_level = int(
        refinement_level
    )

    # For NESTED indexing represented with signed 64-bit integers,
    # refinement levels through 29 are supported safely here.
    if not 0 <= refinement_level <= 29:
        raise ValueError(
            "refinement_level must be in [0, 29]"
        )

    return refinement_level


def _as_cell_ids(
    cell_ids: ArrayLike,
) -> NDArray[np.uint64]:
    """Convert HEALPix cell ids to a validated one-dimensional array."""

    cells = np.asarray(
        cell_ids
    )

    if cells.ndim == 0:
        cells = cells.reshape(1)

    if cells.ndim != 1:
        raise ValueError(
            "cell_ids must be a one-dimensional array"
        )

    if not np.issubdtype(
        cells.dtype,
        np.integer,
    ):
        raise TypeError(
            "cell_ids must contain integers"
        )

    if np.issubdtype(
        cells.dtype,
        np.signedinteger,
    ) and np.any(cells < 0):
        raise ValueError(
            "cell_ids must be non-negative"
        )

    return cells.astype(
        np.uint64,
        copy=False,
    )


def _npix(
    refinement_level: int,
) -> int:
    """Return the number of HEALPix cells at one refinement level."""

    nside = 1 << refinement_level

    return 12 * nside * nside


# ---------------------------------------------------------------------------
# Internal topology API
# ---------------------------------------------------------------------------


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
        One-dimensional NESTED HEALPix cell identifiers.

    refinement_level
        HEALPix refinement level, with::

            nside = 2**refinement_level

    connectivity
        ``"edge"``
            Return only the four edge-sharing neighbours.

        ``"edge_or_vertex"``
            Return all immediate neighbours sharing either an edge or a
            vertex.

    Returns
    -------
    numpy.ndarray
        One row per input cell.

        For ``connectivity="edge"`` the shape is ``(N, 4)`` and the
        deterministic direction order is::

            SW, NW, NE, SE

        For ``connectivity="edge_or_vertex"`` the shape is ``(N, 8)``
        and the deterministic direction order is::

            SW, W, NW, N, NE, E, SE, S

        Missing topological positions are represented by ``-1``.

    Notes
    -----
    Only NESTED HEALPix indexing is supported.

    This function is private implementation infrastructure and is not
    exported from the top-level ``healpix_analyse`` package.
    """

    refinement_level = (
        _validate_refinement_level(
            refinement_level
        )
    )

    cells = _as_cell_ids(
        cell_ids
    )

    if connectivity not in (
        "edge",
        "edge_or_vertex",
    ):
        raise ValueError(
            "connectivity must be 'edge' or 'edge_or_vertex'"
        )

    if cells.size == 0:
        width = (
            4
            if connectivity == "edge"
            else 8
        )

        return np.empty(
            (0, width),
            dtype=np.int64,
        )

    npix = _npix(
        refinement_level
    )

    if np.any(cells >= npix):
        raise ValueError(
            "cell_ids contains an identifier outside the valid range "
            f"[0, {npix}) for refinement_level={refinement_level}"
        )

    neighbours = np.asarray(
        healpix_geo_nested.neighbours(
            cells,
            refinement_level,
            connectivity=connectivity,
            num_threads=0,
        ),
        dtype=np.int64,
    )

    width = (
        4
        if connectivity == "edge"
        else 8
    )

    expected_shape = (
        cells.size,
        width,
    )

    if neighbours.shape != expected_shape:
        raise RuntimeError(
            "Unexpected shape returned by healpix_geo.nested.neighbours: "
            f"{neighbours.shape}; expected {expected_shape}"
        )

    return np.ascontiguousarray(
        neighbours,
        dtype=np.int64,
    )


def nested_edge_neighbours(
    cell_ids: ArrayLike,
    refinement_level: int,
) -> NDArray[np.int64]:
    """Return the four edge-sharing neighbours of NESTED HEALPix cells.

    The deterministic output direction order is::

        SW, NW, NE, SE

    Notes
    -----
    This helper is private. Immediate topology is provided by
    ``healpix-geo`` / CDSHEALPix.
    """

    return nested_neighbours(
        cell_ids,
        refinement_level,
        connectivity="edge",
    )
