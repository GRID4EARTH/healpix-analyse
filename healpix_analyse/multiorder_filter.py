"""Gaussian filter operator on a mixed-level (multi-order) HEALPix cell set.

The cells are *leaves* of a quad-tree: every leaf has its own refinement
level, leaves do not overlap, and a place that is not covered by any leaf is
land (or simply outside the domain).  The operator spreads weight through
water: the kernel sees the length of the shortest path over neighbouring
leaves, never a straight line over land.

Graph
-----
Two leaves are neighbours when they share an edge or a corner, whatever their
levels.  For a leaf at level ``l`` the eight neighbouring cells at level ``l``
are taken from ``healpix-geo`` (via :func:`nested_neighbours`); each is then
looked up in the set: the cell itself, or its parent.  A leaf that is finer
than its neighbour finds the neighbour as the parent of one of its own
neighbours, so the coarse side needs no search of its own.

The input must be *2:1 balanced*: two leaves that touch differ by at most one
level.  A leaf whose neighbour is covered by an ancestor two or more levels
up raises ``ValueError``.

A contact through a corner only is passable when the two leaves have a common
neighbouring *leaf* that is not blocked, in both ``correction`` modes: a
diagonal chain of land would otherwise leak, the rule of
:func:`gaussian_filter_operator` with ``blocked_cells``.

Distance
--------
Edge lengths are the geodesic distance between the centres of the two leaves.
``correction="none"`` uses the shortest path as it is: absence of a leaf
and blocked leaves both force detours, and the metrication error of the
lattice (a path along cells is longer than the straight line, by up to
10 % within 30 degrees of the equator and up to 40 % elsewhere) stays.
``correction="lattice"`` is the correction of ``blocked_cells``: the
path through water ``D`` is divided by the path ``D0`` over the same graph
with the blocked leaves made passable (the corner rule still applies to
``D0``), and multiplied by the straight geodesic, so that ``d`` is exactly
the straight distance wherever blocked leaves do not matter.  Only blocked
leaves are land in that mode: a place without a leaf adds nothing to ``D`` or
to ``D0``, so the ratio ignores it and it does not lengthen ``d``, across an
edge or a corner contact alike.  **Land must therefore be given as ``blocked``
leaves in this mode.**  A hole in the set is not a barrier: the distance
across it is the straight distance.  The search stops at ``1.5 * radius``
(``D0 / straight`` is below 1.5 without holes); when a hole, or the edge of the
set, lies within reach of a source, that source is searched again up to
``3 * radius``.  A pair whose ``D`` or ``D0`` exceeds the limit is dropped, so
a hole that forces a detour longer than three radii acts as a barrier.  Use
``correction="none"`` for land made of absent leaves.
"""

from __future__ import annotations

import heapq
import warnings

import numpy as np

from healpix_geo import nested

from ._neighbourhood import _wgs84_distance
from ._topology import nested_neighbours
from .radial_filter import (
    FilterOperator,
    _normalise,
    _validate_normalization,
    _validate_positive_finite,
)

_NORMALIZATIONS = ("present", "source")
_CORRECTIONS = ("none", "lattice")
_SHORTEST_PATHS = ("builtin", "scipy")
_SCIPY_CHUNK = 256
_EDGE_SLOTS = (0, 2, 4, 6)

# Bound of D0 / straight on the HEALPix 8-neighbour lattice, as in the land
# barrier of the one-level operators (measured at one level, see
# ``healpix_analyse._land_barrier``).  Used only with ``correction="lattice"``.
_SEARCH_FACTOR = 1.5

# Search limit, in radii, for the sources that have a hole (or the edge of the
# set) within reach of the first search: the paths around a hole are longer
# than ``_SEARCH_FACTOR`` allows, in ``D`` and in ``D0``.
_HOLE_SEARCH_FACTOR = 3.0


def _validate_leaves(
    levels: np.ndarray, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(levels)
    cell_ids = np.asarray(cell_ids)
    if levels.ndim != 1 or cell_ids.shape != levels.shape:
        raise ValueError(
            "'levels' and 'cell_ids' must be one-dimensional arrays of the "
            "same length."
        )
    if levels.size == 0:
        raise ValueError("The cell set is empty.")
    if not np.issubdtype(levels.dtype, np.integer) or not np.issubdtype(
        cell_ids.dtype, np.integer
    ):
        raise ValueError("'levels' and 'cell_ids' must be integer arrays.")
    if levels.min() < 0 or levels.max() > 29:
        raise ValueError("Refinement levels must lie between 0 and 29.")
    if cell_ids.min() < 0:
        raise ValueError("Cell ids must be non-negative.")
    levels = levels.astype(np.int64)
    cell_ids = cell_ids.astype(np.int64)
    if np.any(cell_ids >= 12 * 4**levels):
        raise ValueError("A cell id is out of range for its level.")
    return levels, cell_ids


def _key(level: np.ndarray | int, ids: np.ndarray) -> np.ndarray:
    """Unique key of a cell (NUNIQ: ``4 * 4**level + id``)."""
    return 4 * np.int64(4) ** level + ids


def _check_no_overlap(levels: np.ndarray, cell_ids: np.ndarray) -> None:
    keys = _key(levels, cell_ids)
    if np.unique(keys).size != keys.size:
        raise ValueError("The cell set contains duplicate cells.")
    present = set(keys.tolist())
    for level in np.unique(levels):
        if level == 0:
            continue
        ids = cell_ids[levels == level]
        for up in range(1, int(level) + 1):
            ancestors = _key(level - up, ids >> (2 * up))
            if any(a in present for a in ancestors.tolist()):
                raise ValueError(
                    "The cell set overlaps: a leaf lies inside another leaf."
                )


def leaf_adjacency(
    levels: np.ndarray, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the neighbour pairs of a mixed-level leaf set.

    Returns ``(first, second, edge)``: positions in the input with
    ``first < second``, and a boolean that is ``True`` when the two leaves
    share (part of) an edge and ``False`` when they only touch at a corner.
    Raises ``ValueError`` when the set overlaps or is not 2:1 balanced.
    """
    return _leaf_contacts(levels, cell_ids)[:3]


def _leaf_contacts(levels: np.ndarray, cell_ids: np.ndarray):
    """``leaf_adjacency`` plus ``rim``: leaves with a neighbouring place that
    no leaf covers (a hole of the set, or its edge)."""
    levels, cell_ids = _validate_leaves(levels, cell_ids)
    _check_no_overlap(levels, cell_ids)
    position = {k: i for i, k in enumerate(_key(levels, cell_ids).tolist())}

    firsts: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    rim = np.zeros(levels.size, dtype=bool)
    for level in np.unique(levels):
        level = int(level)
        member = np.flatnonzero(levels == level)
        ids = cell_ids[member]
        neighbours = nested_neighbours(ids, level)
        for slot in range(8):
            cell = neighbours[:, slot]
            valid = cell >= 0
            if not valid.any():
                continue
            source = member[valid]
            cell = cell[valid]
            found = np.full(cell.size, -1, dtype=np.int64)
            for up in range(0, level + 1):
                key = _key(level - up, cell >> (2 * up)).tolist()
                hit = np.array([position.get(k, -1) for k in key], np.int64)
                if up >= 2 and np.any(hit >= 0):
                    bad = source[hit >= 0][0]
                    raise ValueError(
                        "The cell set is not 2:1 balanced: the leaf of level "
                        f"{level} with id {int(cell_ids[bad])} touches a leaf "
                        f"{up} levels coarser."
                    )
                found = np.where((found < 0) & (hit >= 0), hit, found)
            covered = found >= 0
            if level < 29:  # a neighbouring cell that is split into leaves
                for child in range(4):
                    key = _key(level + 1, 4 * cell + child).tolist()
                    covered |= np.array([k in position for k in key])
            rim[source[~covered]] = True
            keep = (found >= 0) & (found != source)
            if keep.any():
                firsts.append(source[keep])
                seconds.append(found[keep])
                edges.append(np.full(int(keep.sum()), slot in _EDGE_SLOTS))
    if not firsts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=bool), rim

    first = np.concatenate(firsts)
    second = np.concatenate(seconds)
    edge = np.concatenate(edges)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    n = levels.size
    pair = low * n + high
    order = np.lexsort((~edge, pair))  # edge contacts first within a pair
    pair, low, high, edge = pair[order], low[order], high[order], edge[order]
    first_of_pair = np.concatenate(([True], pair[1:] != pair[:-1]))
    return low[first_of_pair], high[first_of_pair], edge[first_of_pair], rim


def _centres(levels: np.ndarray, cell_ids: np.ndarray, ellipsoid: str):
    lon = np.empty(levels.size)
    lat = np.empty(levels.size)
    for level in np.unique(levels):
        member = levels == level
        x, y = nested.healpix_to_lonlat(
            cell_ids[member], int(level), ellipsoid=ellipsoid
        )
        lon[member] = np.asarray(x)
        lat[member] = np.asarray(y)
    return lon, lat


class _Graph:
    """Undirected graph in compressed form, with a per-edge lookup."""

    def __init__(self, n, first, second, length):
        src = np.concatenate((first, second))
        dst = np.concatenate((second, first))
        weight = np.concatenate((length, length))
        order = np.argsort(src, kind="stable")
        self.src = src[order]
        self.dst = dst[order]
        self.length = weight[order]
        self.indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(self.src, minlength=n)))
        )

    def lists(self):
        """Python adjacency lists: ``[(neighbour, length), ...]`` per node."""
        out = []
        for i in range(self.indptr.size - 1):
            a, b = self.indptr[i], self.indptr[i + 1]
            out.append(list(zip(self.dst[a:b].tolist(), self.length[a:b].tolist())))
        return out


def _dijkstra(adjacency, source: int, limit: float) -> dict[int, float]:
    """Heap Dijkstra from ``source``, cells farther than ``limit`` are left out."""
    best = {source: 0.0}
    heap = [(0.0, source)]
    done = set()
    while heap:
        dist, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        for other, step in adjacency[node]:
            new = dist + step
            if new <= limit and new < best.get(other, np.inf):
                best[other] = new
                heapq.heappush(heap, (new, other))
    return best


def _passable(first, second, edge, blocked):
    """Per pair: usable when blocked leaves are removed.

    A pair with a blocked end is not usable.  A corner-only contact needs a
    common neighbour that is not blocked.
    """
    ok = ~blocked[first] & ~blocked[second]
    free_neighbours: dict[int, set] = {}
    for a, b in zip(first[ok].tolist(), second[ok].tolist()):
        free_neighbours.setdefault(a, set()).add(b)
        free_neighbours.setdefault(b, set()).add(a)
    for k in np.flatnonzero(ok & ~edge):
        a, b = int(first[k]), int(second[k])
        if not (free_neighbours.get(a, set()) & free_neighbours.get(b, set())):
            ok[k] = False
    return ok


def _scipy_paths(n, first, second, length, targets, limit):
    """Yield ``(target, sources, distances)`` with scipy's Dijkstra."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "shortest_path='scipy' needs SciPy; install it "
            "(pip install 'healpix-analyse[scipy]') or use "
            "shortest_path='builtin'."
        ) from error
    graph = coo_matrix((length, (first, second)), shape=(n, n)).tocsr()
    for start in range(0, len(targets), _SCIPY_CHUNK):
        chunk = targets[start : start + _SCIPY_CHUNK]
        dist = dijkstra(graph, directed=False, indices=chunk, limit=limit)
        for target, row in zip(chunk, dist):
            sources = np.flatnonzero(np.isfinite(row))
            yield int(target), sources, row[sources]


def _builtin_paths(adjacency, targets, limit):
    for target in targets:
        reached = _dijkstra(adjacency, int(target), limit)
        yield (
            int(target),
            np.fromiter(reached.keys(), np.int64, len(reached)),
            np.fromiter(reached.values(), np.float64, len(reached)),
        )


def _validate_water_contacts(water_contacts, n):
    """Return the listed pairs as two int64 arrays, or raise ``ValueError``."""
    try:
        first, second = water_contacts
    except (TypeError, ValueError):
        raise ValueError(
            "'water_contacts' must be a pair (first, second) of arrays."
        ) from None
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError(
            "'water_contacts' must be two one-dimensional arrays of the "
            "same length."
        )
    if first.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if not (
        np.issubdtype(first.dtype, np.integer)
        and np.issubdtype(second.dtype, np.integer)
    ):
        raise ValueError("'water_contacts' must hold integer positions.")
    first, second = first.astype(np.int64), second.astype(np.int64)
    low = min(first.min(), second.min())
    high = max(first.max(), second.max())
    if low < 0 or high >= n:
        raise ValueError(
            "'water_contacts' positions must lie in [0, n) with n the number "
            "of leaves."
        )
    return first, second


def _water_pairs(
    levels,
    cell_ids,
    radius,
    correction,
    blocked_mask,
    water_contacts,
    ellipsoid,
    shortest_path="builtin",
):
    """Pairs ``(target, source, distance)`` within ``radius`` through water."""
    n = levels.size
    first, second, edge, rim = _leaf_contacts(levels, cell_ids)
    if water_contacts is not None:
        listed = water_contacts  # already validated: int64 arrays in [0, n)
        listed_key = np.minimum(*listed) * n + np.maximum(*listed)
        known = np.isin(listed_key, first * n + second)
        if not known.all():
            warnings.warn(
                f"{int((~known).sum())} pair(s) of 'water_contacts' are not "
                "geometric neighbours and are ignored.",
                UserWarning,
                stacklevel=3,
            )
        keep = np.isin(first * n + second, listed_key)
        first, second = first[keep], second[keep]
        edge = np.ones(first.size, dtype=bool)
    lon, lat = _centres(levels, cell_ids, ellipsoid)
    length = _wgs84_distance(lon[first], lat[first], lon[second], lat[second])
    ok = _passable(first, second, edge, blocked_mask)
    # D0: every leaf passable, but the corner rule still applies (a corner
    # contact needs a common leaf), so a hole is no barrier in lattice mode.
    ok_free = _passable(first, second, edge, np.zeros(n, dtype=bool))
    lattice = correction == "lattice"
    limit = radius * _SEARCH_FACTOR if lattice else radius
    targets = np.flatnonzero(~blocked_mask)
    if shortest_path == "builtin":
        water_lists = _Graph(n, first[ok], second[ok], length[ok]).lists()
        free_lists = (
            _Graph(n, first[ok_free], second[ok_free], length[ok_free]).lists()
            if lattice
            else None
        )

    def searches(at, at_limit):
        """Water and free runs (the free ones only with ``lattice``)."""
        if shortest_path == "scipy":
            water = _scipy_paths(n, first[ok], second[ok], length[ok], at, at_limit)
            free = (
                _scipy_paths(
                    n, first[ok_free], second[ok_free], length[ok_free], at, at_limit
                )
                if lattice
                else [None] * at.size
            )
        else:
            water = _builtin_paths(water_lists, at, at_limit)
            free = (
                _builtin_paths(free_lists, at, at_limit)
                if lattice
                else [None] * at.size
            )
        return water, free

    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    dw_l: list[np.ndarray] = []
    df_l: list[np.ndarray] = []
    retry: list[int] = []

    def collect(water_runs, free_runs, may_retry):
        for (target, src, dist), free_run in zip(water_runs, free_runs):
            if lattice:
                _, src_f, dist_f = free_run
                if may_retry and rim[src_f].any():
                    # a hole or the edge of the set is within reach: the
                    # detours around it may exceed the search limit
                    retry.append(target)
                    continue
                lookup = dict(zip(src_f.tolist(), dist_f.tolist()))
                d_free = np.array([lookup.get(x, np.nan) for x in src.tolist()])
            else:
                d_free = dist
            keep = ~blocked_mask[src] & np.isfinite(d_free)
            rows_l.append(np.full(int(keep.sum()), target, np.int64))
            cols_l.append(src[keep].astype(np.int64))
            dw_l.append(dist[keep])
            df_l.append(d_free[keep])

    collect(*searches(targets, limit), may_retry=True)
    if retry:
        again = np.asarray(retry, dtype=np.int64)
        collect(*searches(again, radius * _HOLE_SEARCH_FACTOR), may_retry=False)
    empty = np.empty(0)
    rows = np.concatenate(rows_l) if rows_l else empty
    columns = np.concatenate(cols_l) if cols_l else empty
    d_water_list = np.concatenate(dw_l) if dw_l else empty
    d_free_list = np.concatenate(df_l) if df_l else empty

    rows_a = np.asarray(rows, dtype=np.int64)
    columns_a = np.asarray(columns, dtype=np.int64)
    distance_a = np.asarray(d_water_list, dtype=np.float64)
    if lattice:
        d_free_a = np.asarray(d_free_list, dtype=np.float64)
        straight = _wgs84_distance(
            lon[rows_a], lat[rows_a], lon[columns_a], lat[columns_a]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            distance_a = np.where(
                rows_a == columns_a, 0.0, straight * distance_a / d_free_a
            )
    keep = distance_a <= radius
    rows_a, columns_a, distance_a = rows_a[keep], columns_a[keep], distance_a[keep]
    order = np.lexsort((columns_a, rows_a))
    rows_a, columns_a, distance_a = rows_a[order], columns_a[order], distance_a[order]

    return rows_a, columns_a, distance_a


def multiorder_filter_operator(
    levels: np.ndarray,
    cell_ids: np.ndarray,
    *,
    sigma_m: float,
    truncate: float = 4.0,
    normalization: str = "source",
    correction: str = "none",
    blocked: np.ndarray | None = None,
    shortest_path: str = "builtin",
    area_weights: bool = True,
    wet_fraction: np.ndarray | None = None,
    water_contacts: tuple[np.ndarray, np.ndarray] | None = None,
    ellipsoid: str = "WGS84",
) -> FilterOperator:
    """Return a Gaussian filter operator on a mixed-level set of leaves.

    Parameters
    ----------
    levels, cell_ids
        One-dimensional integer arrays: the refinement level of each leaf and
        its NESTED id at that level.  Leaves must not overlap and must be
        2:1 balanced (touching leaves differ by at most one level).  A place
        without a leaf is land.
    sigma_m
        Standard deviation of the Gaussian, in metres of distance through
        water.
    truncate
        The kernel is cut at ``truncate * sigma_m``.
    normalization
        ``"source"`` (default): the weights that leave one leaf sum to one,
        so the total of a signal of masses (one value per leaf, such as a
        probability) is conserved exactly.  The weight to a target leaf is
        proportional to its area.  ``"present"``: the weights that reach one
        leaf sum to one; with ``area_weights`` the source leaves count with
        their area, so a signal of densities is averaged over area.
    correction
        ``"none"`` (default) or ``"lattice"``; see the module description.
        With ``"lattice"`` land must be given as ``blocked`` leaves, not as
        holes in the set.
    blocked
        Optional array of dtype ``bool`` over the leaves (``TypeError``
        otherwise): blocked leaves are land inside the set.  They neither
        receive nor give weight, their row is empty, and the output there
        is 0.
    shortest_path
        ``"builtin"`` (default): the heap Dijkstra of this module, no SciPy.
        ``"scipy"``: ``scipy.sparse.csgraph.dijkstra`` on the same graph,
        imported only then (``ImportError`` if SciPy is missing; the
        ``scipy`` extra installs it).  Both give the same operator.
    area_weights
        Use the area of the leaves (``4**-level``) in the weights.  With
        ``False`` every leaf counts the same, which makes a coarse leaf as
        heavy as a fine one.
    wet_fraction
        Optional array over the leaves with the share of each leaf that is
        water, in ``(0, 1]``; the area weight of a leaf is multiplied by it
        (a leaf that is mostly land, such as a river cell, counts for the
        water it holds).  Validated whenever given, but used only with
        ``area_weights``.
    water_contacts
        Optional ``(first, second)``: two one-dimensional integer arrays of
        equal length, with values in ``[0, n)``, listing pairs of leaves that
        the caller knows to be joined by water.  Two leaves touch in the graph
        only when they touch geometrically *and* the pair is listed, and every
        listed pair counts as an edge contact.  A pair that is not a geometric
        contact is ignored with a ``UserWarning``.  Use it when the leaves are
        larger than the water they hold (a river a few cells wide), where two
        leaves that touch may hold water that does not meet.
    ellipsoid
        Ellipsoid of the centres and of the geodesic lengths.

    Returns
    -------
    FilterOperator
        Rows and columns follow the input order.  Apply with
        :func:`apply_filter_operator`.
    """
    normalization = _validate_normalization(normalization, _NORMALIZATIONS)
    if shortest_path not in _SHORTEST_PATHS:
        raise ValueError(
            f"'shortest_path' must be one of {_SHORTEST_PATHS}; "
            f"got {shortest_path!r}."
        )
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"'correction' must be one of {_CORRECTIONS}; got {correction!r}."
        )
    _validate_positive_finite(sigma_m, name="sigma_m")
    _validate_positive_finite(truncate, name="truncate")
    levels, cell_ids = _validate_leaves(levels, cell_ids)
    n = levels.size
    if blocked is None:
        blocked_mask = np.zeros(n, bool)
    else:
        blocked_mask = np.asarray(blocked)
        if blocked_mask.dtype != np.bool_:
            raise TypeError("'blocked' must be a boolean array.")
        if blocked_mask.shape != (n,):
            raise ValueError(
                "'blocked' must be a boolean array over the leaves."
            )
    wet = None
    if wet_fraction is not None:
        wet = np.asarray(wet_fraction, dtype=np.float64)
        if wet.shape != (n,) or not np.all((wet > 0.0) & (wet <= 1.0)):
            raise ValueError(
                "'wet_fraction' must be a finite array over the leaves with "
                "values in (0, 1]."
            )
    if water_contacts is not None:
        water_contacts = _validate_water_contacts(water_contacts, n)

    radius = truncate * sigma_m
    rows_a, columns_a, distance_a = _water_pairs(
        levels,
        cell_ids,
        radius,
        correction,
        blocked_mask,
        water_contacts,
        ellipsoid,
        shortest_path,
    )

    area = 4.0 ** (-levels.astype(np.float64)) if area_weights else np.ones(n)
    if wet is not None and area_weights:
        area = area * wet
    weights = np.exp(-0.5 * (distance_a / sigma_m) ** 2)
    weights = weights * (area[rows_a] if normalization == "source" else area[columns_a])
    weights = _normalise(weights, rows_a, columns_a, n, n, normalization)
    row_offsets = np.concatenate(
        ([0], np.cumsum(np.bincount(rows_a, minlength=n), dtype=np.int64))
    ).astype(np.int64)
    return FilterOperator(row_offsets, columns_a, weights, n)
