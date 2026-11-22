"""Distances through water for filter operators with blocked cells (land).

A *blocked* cell neither receives nor gives weight.  Between two other cells
the distance used by the kernel is the length of the shortest path that stays
on non-blocked cells, corrected for the metrication error of the lattice.

Graph
-----
Nodes are the cells of the processing domain.  Two cells are joined by an
edge when they are immediate (8-)neighbours in the HEALPix NESTED grid, with
the geodesic distance between their centres as length.  The eight neighbours
of a cell come in two kinds: four share an edge with it, four only a vertex
(the ``W, N, E, S`` slots of the ``SW, W, NW, N, NE, E, SE, S`` order).  A
vertex edge is dropped when both cells that touch the same vertex and lie
between the two ends are blocked or outside the domain: otherwise a diagonal
chain of land cells, which is how a coast along a ring of constant latitude
looks on this grid, would leak.  Cells outside the domain cannot be crossed.

Correction
----------
A path on an 8-connected lattice is longer than the straight geodesic: by up
to 10 % in the equatorial belt of the grid and up to 40 % in the polar caps
(measured, ``benchmarks/land_barrier_lattice.json``).  The same pair of cells is therefore searched twice, once
with the blocked cells removed (``D``) and once with every domain cell
passable (``D0``; the vertex rule applies to both, so cells outside the domain
are no barrier: a hole costs the same in ``D`` and ``D0``).  The distance used is::

    d = straight * D / D0

so that ``d`` equals the straight distance whenever land does not change the
shortest lattice path (in particular for every pair that has no blocked cell
anywhere near it), and equals the lattice detour, scaled back by the
lattice's own metrication factor, otherwise.  ``D >= D0`` so ``d`` is never
shorter than the straight distance.

Search
------
Only the pairs of the straight-line disc of radius ``_SEARCH_FACTOR * radius``
are candidates: a pair with ``d <= radius`` has a lattice path of length
``D <= radius * D0 / straight``, and ``D0 / straight`` is bounded by
``_SEARCH_FACTOR`` on the HEALPix grid.  Every cell of a
shortest path of length at most ``_SEARCH_FACTOR * radius`` lies itself in the
candidate disc (triangle inequality), so the search is closed on the
candidate pairs and runs as a vectorised Bellman-Ford iteration on them,
for the source cells whose candidate disc contains a blocked cell only.
"""

from __future__ import annotations

import numpy as np

from ._neighbourhood import _wgs84_distance
from ._topology import nested_neighbours
from healpix_geo import nested

# Bound of ``D0 / straight`` over the pairs of the search disc.  Measured on
# the whole sphere, levels 3 to 7 (benchmarks/land_barrier_lattice.json): at
# most 1.10 within 30 degrees of the equator and 1.39 in the polar caps and
# just below them; an independent check found 1.422 at level 8 near lon 0,
# lat 45.  The ratio grows slowly with the level.  1.5 leaves a margin, which
# tests/test_land_barrier.py checks on a patch at level 8 and 10.
_SEARCH_FACTOR = 1.5

# Candidate pairs handled at once; bounds the temporary (pairs, 8) tables.
_CHUNK_PAIRS = 200_000

_VERTEX_SLOTS = np.array([1, 3, 5, 7])


def domain_neighbour_table(
    domain: np.ndarray, refinement_level: int
) -> np.ndarray:
    """Return, per domain cell, the domain positions of its 8 neighbours.

    Neighbours that do not exist or are not in the domain are ``-1``.  The
    slot order is ``SW, W, NW, N, NE, E, SE, S``.
    """
    global_ids = nested_neighbours(domain, refinement_level)
    order = np.argsort(domain)
    sorted_domain = domain[order]
    probe = np.where(global_ids < 0, 0, global_ids).astype(np.uint64)
    position = np.minimum(
        np.searchsorted(sorted_domain, probe), domain.size - 1
    )
    found = (global_ids >= 0) & (sorted_domain[position] == probe)
    return np.where(found, order[position], -1).astype(np.int64)


def edge_lengths(
    domain: np.ndarray, table: np.ndarray, refinement_level: int
) -> np.ndarray:
    """Geodesic centre-to-centre length of every edge of ``table``."""
    lon, lat = nested.healpix_to_lonlat(
        domain, refinement_level, ellipsoid="WGS84"
    )
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    valid = table >= 0
    rows, slots = np.nonzero(valid)
    length = np.full(table.shape, np.inf)
    length[rows, slots] = _wgs84_distance(
        lon[rows], lat[rows], lon[table[rows, slots]], lat[table[rows, slots]]
    )
    return length


def passable_edges(table: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """Boolean ``(n, 8)``: edge usable when blocked cells are removed."""
    safe = np.where(table < 0, 0, table)
    allowed = (table >= 0) & ~blocked[:, None] & ~blocked[safe]
    for slot in _VERTEX_SLOTS:
        before = table[:, (slot - 1) % 8]
        after = table[:, (slot + 1) % 8]
        free_before = (before >= 0) & ~blocked[np.where(before < 0, 0, before)]
        free_after = (after >= 0) & ~blocked[np.where(after < 0, 0, after)]
        allowed[:, slot] &= free_before | free_after
    return allowed


def _search_chunk(
    rows: np.ndarray,
    cols: np.ndarray,
    n_cells: int,
    table: np.ndarray,
    length: np.ndarray,
    allowed: np.ndarray,
    allowed_free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(D, D0)`` for candidate pairs (sources are ``rows``)."""
    n_pairs = rows.size
    key = rows * n_cells + cols
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]

    neighbour = table[cols]
    valid = neighbour >= 0
    probe = rows[:, None] * n_cells + np.where(valid, neighbour, 0)
    position = np.minimum(np.searchsorted(sorted_key, probe), n_pairs - 1)
    hit = valid & (sorted_key[position] == probe)
    source = np.where(hit, order[position], n_pairs)
    step = length[cols]
    step_free = np.where(hit & allowed_free[cols], step, np.inf)
    step_blocked = np.where(hit & allowed[cols], step, np.inf)
    del neighbour, valid, probe, position, step

    is_self = rows == cols
    results = []
    for weights in (step_blocked, step_free):
        best = np.full(n_pairs + 1, np.inf)
        best[:n_pairs][is_self] = 0.0
        while True:
            candidate = (best[source] + weights).min(axis=1)
            updated = np.minimum(best[:n_pairs], candidate)
            updated[is_self] = 0.0
            if np.array_equal(updated, best[:n_pairs]):
                break
            best[:n_pairs] = updated
        results.append(best[:n_pairs].copy())
    return results[0], results[1]


def path_distances(
    rows: np.ndarray,
    cols: np.ndarray,
    n_cells: int,
    table: np.ndarray,
    length: np.ndarray,
    allowed: np.ndarray,
    allowed_free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(D, D0)`` for candidate pairs, processed in row-aligned chunks.

    ``rows`` must be ascending (compressed-row order) and closed under the
    search: every pair of the candidate disc of a row is present.
    """
    n_pairs = rows.size
    path = np.empty(n_pairs)
    free = np.empty(n_pairs)
    start = 0
    while start < n_pairs:
        stop = min(start + _CHUNK_PAIRS, n_pairs)
        if stop < n_pairs:
            # extend to the end of the current source row
            stop = int(np.searchsorted(rows, rows[stop - 1], side="right"))
        path[start:stop], free[start:stop] = _search_chunk(
            rows[start:stop],
            cols[start:stop],
            n_cells,
            table,
            length,
            allowed,
            allowed_free,
        )
        start = stop
    return path, free


def barrier_pairs(
    geometry,
    blocked: np.ndarray,
    radius: float,
    refinement_level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairs and distances of an operator with blocked cells.

    ``geometry`` is the straight-line geometry of the domain at radius
    ``_SEARCH_FACTOR * radius``; ``blocked`` is a boolean mask over its
    centres.  Returns ``(rows, columns, distance, rows_straight,
    distance_straight)``: the pairs that carry weight with the distance
    through water (at most ``radius``), and the straight-line pairs within
    ``radius`` of every cell, blocked ones included.
    """
    n_cells = geometry.center_ids.size
    rows = np.repeat(
        np.arange(n_cells, dtype=np.int64), np.diff(geometry.row_offsets)
    )
    columns = geometry.neighbour_indices.astype(np.int64)
    straight = geometry.distance_m.copy()
    straight[rows == columns] = 0.0

    # Rows whose search disc holds a blocked cell need the path search.
    near_blocked = (
        np.bincount(rows, weights=blocked[columns], minlength=n_cells) > 0
    )
    searched = near_blocked & ~blocked
    in_search = searched[rows]
    distance = straight.copy()
    if in_search.any():
        index = np.flatnonzero(in_search)
        table = domain_neighbour_table(geometry.center_ids, refinement_level)
        length = edge_lengths(geometry.center_ids, table, refinement_level)
        allowed = passable_edges(table, blocked)
        # D0: no blocked cell, but the vertex rule still applies, so a hole
        # of the domain is never a barrier (edge or vertex contact alike).
        allowed_free = passable_edges(table, np.zeros(n_cells, dtype=bool))
        path, free = path_distances(
            rows[index],
            columns[index],
            n_cells,
            table,
            length,
            allowed,
            allowed_free,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            through_water = straight[index] * (path / free)
        through_water[rows[index] == columns[index]] = 0.0
        distance[index] = np.where(np.isfinite(path), through_water, np.inf)

    keep = (distance <= radius) & ~blocked[rows] & ~blocked[columns]
    within = straight <= radius
    return (
        rows[keep],
        columns[keep],
        distance[keep],
        rows[within],
        straight[within],
    )
