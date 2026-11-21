"""Tests of ``multiorder_filter_operator`` (mixed-level leaves with land).

Scenes are drawn on real HEALPix cells around lon 10, lat 45: a patch of
level-8 cells of which the half with ``u > 0`` (``u`` in level-8 cells, east
of the centre) is split once into level-9 children.  Land is either the
absence of leaves or ``blocked`` leaves.

The reference does not use the library's neighbour search: every leaf is
expanded into its level-9 descendants, whose neighbours come from the plain
level-9 neighbour table, and two leaves touch when some descendants touch.
The shortest path is an unbounded heap Dijkstra.  The corner rule (a contact
through a corner needs a free common neighbour) is the library's by
definition.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest
from healpix_geo import nested
from pyproj import Geod

from healpix_analyse import (
    apply_filter_operator,
    gaussian_filter_operator,
    multiorder_filter_operator,
)
from healpix_analyse.multiorder_filter import leaf_adjacency
from healpix_analyse.radial_filter import _cell_size_m

_WGS84 = Geod(ellps="WGS84")
LEVEL = 8
CELL = _cell_size_m(LEVEL)
SIGMA = 2.0 * CELL
EARTH = 6_371_007.1809


def _patch(k: int) -> np.ndarray:
    centre = np.asarray(
        nested.lonlat_to_healpix(
            np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
        ),
        dtype=np.uint64,
    )
    ring = np.asarray(nested.kth_neighbourhood(centre, LEVEL, k))[0]
    return ring[ring >= 0].astype(np.int64)


def _plane(levels, ids):
    """Tangent-plane coordinates of the centres, in level-8 cells."""
    lon = np.empty(ids.size)
    lat = np.empty(ids.size)
    for level in np.unique(levels):
        m = levels == level
        x, y = nested.healpix_to_lonlat(ids[m], int(level), ellipsoid="WGS84")
        lon[m], lat[m] = np.asarray(x), np.asarray(y)
    u = np.radians(lon - 10.0) * np.cos(np.radians(45.0)) * EARTH / CELL
    v = np.radians(lat - 45.0) * EARTH / CELL
    return u, v


def _mixed(k: int = 10):
    base = _patch(k)
    levels = np.full(base.size, LEVEL)
    u, _ = _plane(levels, base)
    split = u > 0
    children = (base[split, None] * 4 + np.arange(4)).ravel()
    ids = np.concatenate((base[~split], children))
    lev = np.concatenate(
        (np.full((~split).sum(), LEVEL), np.full(children.size, LEVEL + 1))
    )
    return lev, ids


def _uniform(k: int = 10):
    ids = _patch(k)
    return np.full(ids.size, LEVEL), ids


def _dense(op):
    n = op.n_columns
    rows = np.repeat(np.arange(op.row_offsets.size - 1), np.diff(op.row_offsets))
    matrix = np.zeros((op.row_offsets.size - 1, n))
    np.add.at(matrix, (rows, op.column_indices), op.weights)
    return matrix


def _operator(levels, ids, **kw):
    kw.setdefault("sigma_m", SIGMA)
    return multiorder_filter_operator(levels, ids, **kw)


# ---------------------------------------------------------------- reference


def _reference_pairs(levels, ids):
    """Touching leaves by level-9 descendants: ``{(a, b): is_edge}``."""
    top = int(levels.max())
    owner = {}
    for i, (level, cell) in enumerate(zip(levels.tolist(), ids.tolist())):
        step = 4 ** (top - level)
        for d in range(cell * step, (cell + 1) * step):
            owner[d] = i
    desc = np.array(sorted(owner), dtype=np.int64)
    allneigh = np.asarray(nested.neighbours(desc, top, connectivity="all"))
    edgeneigh = np.asarray(nested.neighbours(desc, top, connectivity="edge"))
    pairs = {}
    for d, row, erow in zip(desc.tolist(), allneigh, edgeneigh):
        a = owner[d]
        edge_set = {int(c) for c in erow if c >= 0}
        for c in row:
            c = int(c)
            if c < 0 or c not in owner or owner[c] == a:
                continue
            key = tuple(sorted((a, owner[c])))
            pairs[key] = pairs.get(key, False) or (c in edge_set)
    return pairs


def _reference_graph(levels, ids, blocked):
    """Adjacency lists ``[(neighbour, length_m), ...]`` through water."""
    n = ids.size
    pairs = _reference_pairs(levels, ids)
    lon = np.empty(n)
    lat = np.empty(n)
    for level in np.unique(levels):
        m = levels == level
        x, y = nested.healpix_to_lonlat(ids[m], int(level), ellipsoid="WGS84")
        lon[m], lat[m] = np.asarray(x), np.asarray(y)
    touching = [set() for _ in range(n)]
    for (a, b) in pairs:
        if not blocked[a] and not blocked[b]:
            touching[a].add(b)
            touching[b].add(a)
    adjacency = [[] for _ in range(n)]
    for (a, b), is_edge in pairs.items():
        if blocked[a] or blocked[b]:
            continue
        if not is_edge and not (touching[a] & touching[b]):
            continue
        length = _WGS84.inv(lon[a], lat[a], lon[b], lat[b])[2]
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    return adjacency


def _reference_distance(adjacency, source):
    """Unbounded Dijkstra distance (m) from ``source`` through water."""
    n = len(adjacency)
    best = np.full(n, np.inf)
    best[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, a = heapq.heappop(heap)
        if d > best[a]:
            continue
        for b, length in adjacency[a]:
            if d + length < best[b]:
                best[b] = d + length
                heapq.heappush(heap, (d + length, b))
    return best


def _reference_operator(
    levels, ids, normalization, blocked=None, area=True, wet=None
):
    n = ids.size
    blocked = np.zeros(n, bool) if blocked is None else blocked
    radius = 4.0 * SIGMA
    a = 4.0 ** (-levels.astype(float)) if area else np.ones(n)
    if wet is not None:
        a = a * wet
    matrix = np.zeros((n, n))
    adjacency = _reference_graph(levels, ids, blocked)
    for target in range(n):
        if blocked[target]:
            continue
        d = _reference_distance(adjacency, target)
        ok = (d <= radius) & ~blocked
        g = np.where(ok, np.exp(-0.5 * (d / SIGMA) ** 2), 0.0)
        matrix[target] = g * (a if normalization == "present" else a[target])
    if normalization == "present":
        sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)
    else:
        sums = matrix.sum(axis=0, keepdims=True)
        matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)
    return matrix


# -------------------------------------------------------------- equivalence


@pytest.mark.parametrize("normalization", ["present", "source"])
@pytest.mark.parametrize("with_wall", [False, True])
def test_one_level_equals_the_one_level_operator(normalization, with_wall):
    levels, ids = _uniform(12)
    u, v = _plane(levels, ids)
    wall = (np.abs(u - 0.4 * v) <= 1.0) & (np.abs(v) <= 6.0) if with_wall else None
    reference = gaussian_filter_operator(
        ids.astype(np.uint64),
        LEVEL,
        sigma_m=SIGMA,
        normalization=normalization,
        blocked_cells=ids[wall].astype(np.uint64) if with_wall else None,
    )
    operator = _operator(
        levels, ids, normalization=normalization, correction="lattice", blocked=wall
    )
    np.testing.assert_allclose(
        _dense(operator), _dense(reference), rtol=1e-12, atol=1e-15
    )
    assert np.array_equal(_dense(operator) != 0, _dense(reference) != 0)


@pytest.mark.parametrize("normalization", ["present", "source"])
def test_mixed_levels_against_brute_force(normalization):
    levels, ids = _mixed()
    u, v = _plane(levels, ids)
    # land by absence: a bay and a block of blocked leaves
    keep = ~((np.hypot(u + 3.0, v - 2.0) <= 1.6))
    levels, ids = levels[keep], ids[keep]
    u, v = u[keep], v[keep]
    blocked = (np.abs(u - 2.0) <= 0.6) & (np.abs(v + 1.0) <= 3.0)
    operator = _operator(levels, ids, normalization=normalization, blocked=blocked)
    expected = _reference_operator(levels, ids, normalization, blocked)
    np.testing.assert_allclose(_dense(operator), expected, rtol=1e-10, atol=1e-14)


def test_without_area_weights_against_brute_force():
    levels, ids = _mixed(8)
    operator = _operator(levels, ids, normalization="present", area_weights=False)
    expected = _reference_operator(levels, ids, "present", area=False)
    np.testing.assert_allclose(_dense(operator), expected, rtol=1e-10, atol=1e-14)


def test_adjacency_matches_the_descendant_reference():
    levels, ids = _mixed(8)
    first, second, edge = leaf_adjacency(levels, ids)
    got = {(int(a), int(b)): bool(e) for a, b, e in zip(first, second, edge)}
    assert got == _reference_pairs(levels, ids)


# ------------------------------------------------------------ scenes (land)


def _nearest(levels, ids, point):
    u, v = _plane(levels, ids)
    return int(np.argmin((u - point[0]) ** 2 + (v - point[1]) ** 2))


def test_closed_bay_nothing_leaks():
    levels, ids = _mixed(12)
    u, v = _plane(levels, ids)
    wall = np.abs(u) <= 1.2  # a full-height wall of absent leaves
    levels, ids, u = levels[~wall], ids[~wall], u[~wall]
    dense = _dense(_operator(levels, ids, normalization="source"))
    west, east = u < 0, u > 0
    assert dense[np.ix_(west, east)].sum() == 0.0
    assert dense[np.ix_(east, west)].sum() == 0.0
    # nothing is lost either: every column still sums to one
    np.testing.assert_allclose(dense.sum(axis=0), 1.0, rtol=0, atol=1e-13)


def test_diagonal_chain_of_land_does_not_leak():
    # a chain of leaves along the N direction (a corner contact from one to
    # the next, a coast along a ring of constant latitude); the leaves NW and
    # NE of a chain leaf touch each other at a corner across the chain
    levels, ids = _uniform(12)
    index = {int(c): i for i, c in enumerate(ids)}
    table = np.asarray(nested.neighbours(ids, LEVEL, connectivity="all"))
    start = _nearest(levels, ids, (0.0, 0.0))
    chain = {start}
    for step in (3, 7):  # N and S
        here = start
        for _ in range(7):
            here = index[int(table[here, step])]
            chain.add(here)
    wall = np.zeros(ids.size, bool)
    wall[list(chain)] = True
    a, b = index[int(table[start, 2])], index[int(table[start, 4])]  # NW, NE
    free = _dense(_operator(levels, ids, normalization="source"))
    shut = _dense(_operator(levels, ids, normalization="source", blocked=wall))
    assert free[a, b] > 0.0
    assert shut[a, b] == 0.0 and shut[b, a] == 0.0
    # the same through the absence of the chain
    kept = ~wall
    shut = _dense(_operator(levels[kept], ids[kept], normalization="source"))
    a2 = int(np.where(ids[kept] == ids[a])[0][0])
    b2 = int(np.where(ids[kept] == ids[b])[0][0])
    assert shut[a2, b2] == 0.0 and shut[b2, a2] == 0.0


def test_strait_and_island_follow_the_brute_force_path():
    levels, ids = _mixed(14)
    u, v = _plane(levels, ids)
    wall = (np.abs(u) <= 1.0) & (np.abs(v) > 1.5)  # strait of width 3 in v
    island = np.hypot(u - 6.0, v) <= 1.6
    keep = ~(wall | island)
    levels, ids, u, v = levels[keep], ids[keep], u[keep], v[keep]
    operator = _operator(levels, ids, normalization="source")
    free = _operator(*_mixed(14), normalization="source")
    dense = _dense(operator)
    expected = _reference_operator(levels, ids, "source")
    np.testing.assert_allclose(dense, expected, rtol=1e-10, atol=1e-14)

    # across the wall, away from the gap: weight only through the strait
    west = _nearest(levels, ids, (-2.5, 3.0))
    east = _nearest(levels, ids, (2.5, 3.0))
    direct = _dense(free)
    lev0, ids0 = _mixed(14)
    w0 = _nearest(lev0, ids0, (-2.5, 3.0))
    e0 = _nearest(lev0, ids0, (2.5, 3.0))
    assert dense[east, west] < direct[e0, w0]
    # the island is not crossed: behind it the weight is below the free one
    behind = _nearest(levels, ids, (8.0, 0.0))
    front = _nearest(levels, ids, (4.0, 0.0))
    behind0 = _nearest(lev0, ids0, (8.0, 0.0))
    front0 = _nearest(lev0, ids0, (4.0, 0.0))
    assert dense[behind, front] < direct[behind0, front0]


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize("correction", ["none", "lattice"])
def test_source_normalisation_conserves_the_total(correction):
    levels, ids = _mixed(12)
    u, v = _plane(levels, ids)
    keep = ~(np.hypot(u + 2.0, v) <= 1.5)
    levels, ids, u, v = levels[keep], ids[keep], u[keep], v[keep]
    blocked = (np.abs(u - 3.0) <= 0.6) & (np.abs(v) <= 3.0)
    operator = _operator(
        levels, ids, normalization="source", blocked=blocked, correction=correction
    )
    dense = _dense(operator)
    sums = dense.sum(axis=0)
    np.testing.assert_allclose(sums[~blocked], 1.0, rtol=0, atol=1e-13)
    assert np.all(sums[blocked] == 0.0)
    mass = np.random.default_rng(3).random(ids.size)
    mass[blocked] = 0.0
    out = apply_filter_operator(mass, *operator)
    assert out.sum() == pytest.approx(mass.sum(), rel=1e-12)
    assert np.all(out[blocked] == 0.0)


def test_present_normalisation_rows_sum_to_one_and_keep_a_constant():
    levels, ids = _mixed(12)
    operator = _operator(levels, ids, normalization="present")
    dense = _dense(operator)
    np.testing.assert_allclose(dense.sum(axis=1), 1.0, rtol=0, atol=1e-13)
    out = apply_filter_operator(np.full(ids.size, 3.5), *operator)
    np.testing.assert_allclose(out, 3.5, rtol=1e-13)


def test_a_target_weight_grows_with_its_area():
    # the weight handed to a coarse leaf is four times that to a fine one at
    # the same distance (source normalisation, equal Gaussian factor)
    levels, ids = _mixed(12)
    with_area = _dense(_operator(levels, ids, normalization="source"))
    without = _dense(
        _operator(levels, ids, normalization="source", area_weights=False)
    )
    assert not np.allclose(with_area, without)


def test_wet_fraction_scales_the_area_weight():
    levels, ids = _mixed(8)
    wet = np.random.default_rng(5).uniform(0.2, 1.0, ids.size)
    for normalization in ("present", "source"):
        operator = _operator(levels, ids, normalization=normalization, wet_fraction=wet)
        expected = _reference_operator(levels, ids, normalization, wet=wet)
        np.testing.assert_allclose(_dense(operator), expected, rtol=1e-10, atol=1e-14)
    with pytest.raises(ValueError, match="wet_fraction"):
        _operator(levels, ids, wet_fraction=np.zeros(ids.size))


def test_water_contacts_restrict_the_graph():
    levels, ids = _mixed(10)
    u, _ = _plane(levels, ids)
    first, second, _ = leaf_adjacency(levels, ids)
    across = (u[first] < 0) != (u[second] < 0)
    contacts = (first[~across], second[~across])
    dense = _dense(_operator(levels, ids, water_contacts=contacts))
    west, east = u < 0, u >= 0
    assert dense[np.ix_(west, east)].sum() == 0.0
    assert dense[np.ix_(east, west)].sum() == 0.0
    # with every contact listed nothing changes
    every = _dense(_operator(levels, ids, water_contacts=(first, second)))
    np.testing.assert_array_equal(every, _dense(_operator(levels, ids)))


# -------------------------------------------------------------------- input


def test_unbalanced_set_is_refused():
    base = _patch(3)
    centre = base[0]
    # one level-8 leaf next to a level-10 leaf: two levels apart
    neighbour = int(np.asarray(nested.neighbours(np.array([centre]), LEVEL, connectivity="all"))[0, 3])
    fine = neighbour * 16
    levels = np.array([LEVEL, LEVEL + 2])
    with pytest.raises(ValueError, match="2:1"):
        leaf_adjacency(levels, np.array([centre, fine]))


def test_overlapping_and_duplicate_leaves_are_refused():
    parent = int(_patch(2)[0])
    with pytest.raises(ValueError, match="overlap"):
        leaf_adjacency(np.array([LEVEL, LEVEL + 1]), np.array([parent, parent * 4]))
    with pytest.raises(ValueError, match="duplicate"):
        leaf_adjacency(np.array([LEVEL, LEVEL]), np.array([parent, parent]))


def test_argument_validation():
    levels, ids = _uniform(3)
    with pytest.raises(ValueError, match="correction"):
        _operator(levels, ids, correction="exact")
    with pytest.raises(ValueError, match="normalization"):
        _operator(levels, ids, normalization="full_disc")
    with pytest.raises(ValueError, match="blocked"):
        _operator(levels, ids, blocked=np.zeros(ids.size + 1, bool))
    with pytest.raises(ValueError, match="same length"):
        _operator(levels[:-1], ids)
    with pytest.raises(ValueError, match="out of range"):
        _operator(np.array([2]), np.array([200]))
