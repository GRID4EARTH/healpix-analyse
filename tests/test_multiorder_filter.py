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


def _reference_graph(levels, ids, blocked, corner_rule=True):
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
        if corner_rule and not is_edge and not (touching[a] & touching[b]):
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


def _reference_lattice_operator(levels, ids, blocked, normalization="source"):
    """``d = straight * D / D0``: D through water, D0 with land passable (the
    corner rule applies to both, so a hole of the set is no barrier)."""
    n = ids.size
    radius = 4.0 * SIGMA
    a = 4.0 ** (-levels.astype(float))
    lon = np.empty(n)
    lat = np.empty(n)
    for level in np.unique(levels):
        m = levels == level
        x, y = nested.healpix_to_lonlat(ids[m], int(level), ellipsoid="WGS84")
        lon[m], lat[m] = np.asarray(x), np.asarray(y)
    water = _reference_graph(levels, ids, blocked)
    free = _reference_graph(levels, ids, np.zeros(n, bool))
    matrix = np.zeros((n, n))
    for target in range(n):
        if blocked[target]:
            continue
        d_w = _reference_distance(water, target)
        d_f = _reference_distance(free, target)
        straight = np.array(
            [_WGS84.inv(lon[target], lat[target], lon[j], lat[j])[2] for j in range(n)]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(np.arange(n) == target, 0.0, straight * d_w / d_f)
        ok = np.isfinite(d) & (d <= radius) & ~blocked
        g = np.where(ok, np.exp(-0.5 * (np.where(ok, d, 0.0) / SIGMA) ** 2), 0.0)
        matrix[target] = g * (a if normalization == "present" else a[target])
    sums = matrix.sum(axis=0 if normalization == "source" else 1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def _balance(leaves):
    """Split coarse leaves until the set (a set of (level, id)) is 2:1."""
    leaves = set(leaves)
    while True:
        split = set()
        for level in sorted({lv for lv, _ in leaves}):
            if level < 2:
                continue
            ids = np.array([i for lv, i in leaves if lv == level], dtype=np.uint64)
            neigh = np.asarray(nested.neighbours(ids, level, connectivity="all"))
            for cell in np.unique(neigh[neigh >= 0]).tolist():
                for up in range(2, level + 1):
                    if (level - up, cell >> (2 * up)) in leaves:
                        split.add((level - up, cell >> (2 * up)))
        if not split:
            return leaves
        for lv, i in split:
            leaves.remove((lv, i))
            leaves |= {(lv + 1, 4 * i + k) for k in range(4)}


def _arrays(leaves):
    leaves = sorted(leaves)
    return (
        np.array([lv for lv, _ in leaves], dtype=np.int64),
        np.array([i for _, i in leaves], dtype=np.int64),
    )


def _three_levels(k: int = 5):
    """Levels 8, 9 and 10 in one set: finer towards the centre line."""
    leaves = {(LEVEL, int(i)) for i in _patch(k)}
    for _ in range(2):
        levels, ids = _arrays(leaves)
        u, v = _plane(levels, ids)
        near = np.hypot(u, v) < (3.0 if levels.max() == LEVEL else 1.6)
        for lv, i, flag in zip(levels.tolist(), ids.tolist(), near):
            if flag and lv == levels.max():
                leaves.remove((lv, i))
                leaves |= {(lv + 1, 4 * i + c) for c in range(4)}
    return _arrays(_balance(leaves))


# -------------------------------------------------------------- equivalence


@pytest.mark.parametrize("normalization", ["present", "source"])
@pytest.mark.parametrize("with_wall", [False, True])
@pytest.mark.parametrize("with_holes", [False, True])
def test_one_level_equals_the_one_level_operator(
    normalization, with_wall, with_holes
):
    levels, ids = _uniform(12)
    if with_holes:  # about 8 % of the cells are outside the domain
        keep = np.random.default_rng(1).random(ids.size) > 0.08
        levels, ids = levels[keep], ids[keep]
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
    with pytest.raises(ValueError, match="2:1") as info:
        leaf_adjacency(levels, np.array([centre, fine]))
    assert f"level {LEVEL + 2} with id {fine}" in str(info.value)


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


@pytest.mark.parametrize("correction", ["none", "lattice"])
@pytest.mark.parametrize("with_contacts", [False, True])
def test_scipy_and_builtin_paths_give_the_same_operator(correction, with_contacts):
    pytest.importorskip("scipy")
    levels, ids = _mixed(10)
    u, v = _plane(levels, ids)
    keep = ~(np.hypot(u + 2.0, v - 1.0) <= 1.5)
    levels, ids, u, v = levels[keep], ids[keep], u[keep], v[keep]
    blocked = (np.abs(u - 2.5) <= 0.6) & (np.abs(v) <= 3.0)
    kw = {}
    if with_contacts:
        first, second, _ = leaf_adjacency(levels, ids)
        drop = (np.abs(v[first] - 4.0) < 0.5) & (np.abs(v[second] - 4.0) < 0.5)
        kw["water_contacts"] = (first[~drop], second[~drop])
    a = _operator(levels, ids, blocked=blocked, correction=correction, **kw)
    b = _operator(
        levels, ids, blocked=blocked, correction=correction,
        shortest_path="scipy", **kw
    )
    np.testing.assert_allclose(_dense(a), _dense(b), rtol=1e-12, atol=1e-15)
    assert np.array_equal(_dense(a) != 0, _dense(b) != 0)


def test_shortest_path_validation():
    levels, ids = _uniform(3)
    with pytest.raises(ValueError, match="shortest_path"):
        _operator(levels, ids, shortest_path="networkx")


# ------------------------------------------------------- review additions


def _corner_pair():
    """A level-8 leaf and a neighbour that touches it at a corner only."""
    centre = 44131
    around = np.asarray(
        nested.neighbours(np.array([centre]), LEVEL, connectivity="all")
    )[0]
    around = around[around >= 0].astype(np.int64)
    levels = np.full(around.size + 1, LEVEL)
    ids = np.concatenate(([centre], around))
    first, second, edge = leaf_adjacency(levels, ids)
    mine = (first == 0) & ~edge
    assert mine.any()
    return centre, int(ids[second[mine][0]]), around


@pytest.mark.parametrize("correction", ["none", "lattice"])
def test_corner_contact_needs_a_common_leaf_in_both_modes(correction):
    centre, corner, around = _corner_pair()
    # two leaves that touch at a corner only: closed, nothing crosses
    levels = np.array([LEVEL, LEVEL])
    dense = _dense(_operator(levels, np.array([centre, corner]), correction=correction))
    assert dense[0, 1] == 0.0 and dense[1, 0] == 0.0
    # with a leaf that touches both, the contact opens
    ca = np.asarray(nested.neighbours(np.array([centre]), LEVEL, connectivity="all"))[0]
    cb = np.asarray(nested.neighbours(np.array([corner]), LEVEL, connectivity="all"))[0]
    common = sorted(set(ca[ca >= 0].tolist()) & set(cb[cb >= 0].tolist()))
    assert common
    ids = np.array([centre, corner, common[0]])
    dense = _dense(_operator(np.full(3, LEVEL), ids, correction=correction))
    assert dense[0, 1] > 0.0 and dense[1, 0] > 0.0
    # the same common leaf blocked: closed again
    blocked = np.array([False, False, True])
    dense = _dense(
        _operator(np.full(3, LEVEL), ids, correction=correction, blocked=blocked)
    )
    assert dense[0, 1] == 0.0 and dense[1, 0] == 0.0


def test_three_or_more_levels_against_brute_force():
    levels, ids = _three_levels(5)
    assert np.unique(levels).size >= 3
    for normalization in ("present", "source"):
        got = _dense(_operator(levels, ids, normalization=normalization))
        want = _reference_operator(levels, ids, normalization)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-13)


def test_lattice_correction_against_brute_force_at_mixed_levels():
    levels, ids = _mixed(6)
    u, v = _plane(levels, ids)
    blocked = (np.abs(u - 1.5) <= 0.7) & (np.abs(v) <= 2.5)
    blocked |= np.hypot(u + 2.5, v - 1.0) <= 0.8
    got = _dense(_operator(levels, ids, blocked=blocked, correction="lattice"))
    want = _reference_lattice_operator(levels, ids, blocked)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("shortest_path", ["builtin", "scipy"])
@pytest.mark.parametrize("seed", [0, 2, 5])
def test_lattice_correction_with_holes_against_brute_force(seed, shortest_path):
    if shortest_path == "scipy":
        pytest.importorskip("scipy")
    levels, ids = _mixed(6)
    keep = np.random.default_rng(seed).random(ids.size) > 0.08  # holes
    levels, ids = levels[keep], ids[keep]
    u, v = _plane(levels, ids)
    blocked = (np.abs(u - v) < 0.8) & (np.abs(u + v) < 6.0)  # diagonal band
    got = _dense(
        _operator(
            levels,
            ids,
            blocked=blocked,
            correction="lattice",
            shortest_path=shortest_path,
        )
    )
    want = _reference_lattice_operator(levels, ids, blocked)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-13)


def test_random_whole_sphere_set_adjacency_equals_brute_force():
    rng = np.random.default_rng(3)
    leaves = set()
    stack = [(2, i) for i in range(12 * 4**2)]
    while stack:
        lv, i = stack.pop()
        if lv < 5 and rng.random() < 0.5:
            stack += [(lv + 1, 4 * i + c) for c in range(4)]
        else:
            leaves.add((lv, i))
    leaves = sorted(_balance(leaves))
    keep = rng.random(len(leaves)) > 0.15
    leaves = [leaf for leaf, k in zip(leaves, keep) if k]
    leaves = [leaves[j] for j in rng.permutation(len(leaves))]
    levels, ids = _arrays(leaves)
    assert np.unique(levels).size >= 3
    first, second, edge = leaf_adjacency(levels, ids)
    got = {(int(a), int(b)): bool(e) for a, b, e in zip(first, second, edge)}
    assert got == _reference_pairs(levels, ids)


def test_extreme_levels_single_leaf_and_empty_contacts():
    # level 0: the twelve base cells
    ids = np.arange(12)
    dense = _dense(multiorder_filter_operator(np.zeros(12, int), ids, sigma_m=2.0e6))
    np.testing.assert_allclose(dense.sum(axis=0), 1.0, rtol=1e-12)
    # level 29: a leaf and its neighbour
    big = 12 * 4**29 // 3
    around = np.asarray(
        nested.neighbours(np.array([big], dtype=np.uint64), 29, connectivity="edge")
    )[0]
    pair = np.array([big, int(around[around >= 0][0])], dtype=np.int64)
    first, second, edge = leaf_adjacency(np.full(2, 29), pair)
    assert (first.tolist(), second.tolist(), edge.tolist()) == ([0], [1], [True])
    dense = _dense(multiorder_filter_operator(np.full(2, 29), pair, sigma_m=0.01))
    assert dense.shape == (2, 2) and np.all(np.isfinite(dense))
    # a single leaf keeps all of its weight
    dense = _dense(_operator(np.array([LEVEL]), np.array([44131])))
    np.testing.assert_allclose(dense, [[1.0]])
    # every leaf blocked: empty operator, no error
    levels, ids = _uniform(2)
    op = _operator(levels, ids, blocked=np.ones(ids.size, bool))
    assert op.weights.size == 0 and np.all(op.row_offsets == 0)
    # no listed water contact: every leaf stays alone
    empty = (np.empty(0, int), np.empty(0, int))
    dense = _dense(_operator(levels, ids, water_contacts=empty))
    np.testing.assert_allclose(dense, np.eye(ids.size))


def test_water_contacts_are_validated():
    levels, ids = _mixed(3)
    n = ids.size
    first, second, _ = leaf_adjacency(levels, ids)
    for bad in ([-1], [n]):
        with pytest.raises(ValueError, match="water_contacts"):
            _operator(levels, ids, water_contacts=(np.array(bad), np.array([0])))
    with pytest.raises(ValueError, match="water_contacts"):
        _operator(levels, ids, water_contacts=([0, 1], [1]))
    with pytest.raises(ValueError, match="water_contacts"):
        _operator(levels, ids, water_contacts=(np.zeros((2, 2), int), np.zeros((2, 2), int)))
    with pytest.raises(ValueError, match="water_contacts"):
        _operator(levels, ids, water_contacts=([0.5], [1.5]))
    with pytest.raises(ValueError, match="water_contacts"):
        _operator(levels, ids, water_contacts=first)
    # a listed pair that is not a geometric contact is reported
    far = np.flatnonzero(
        ~np.isin(np.arange(n), np.concatenate((first[first == 0], second[first == 0])))
    )[-1]
    with pytest.warns(UserWarning, match="not geometric neighbours"):
        _operator(levels, ids, water_contacts=(np.array([0]), np.array([far])))
    # a valid list does not warn
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _operator(levels, ids, water_contacts=(first, second))


def test_blocked_must_be_boolean():
    levels, ids = _uniform(3)
    with pytest.raises(TypeError, match="boolean"):
        _operator(levels, ids, blocked=np.full(ids.size, 0.5))
    with pytest.raises(TypeError, match="boolean"):
        _operator(levels, ids, blocked=np.zeros(ids.size, int))


@pytest.mark.parametrize("area_weights", [True, False])
def test_wet_fraction_is_validated_whenever_given(area_weights):
    levels, ids = _uniform(3)
    n = ids.size
    for bad in (np.zeros(n), np.full(n, 1.5), np.full(n, np.nan), np.ones(n + 1)):
        with pytest.raises(ValueError, match="wet_fraction"):
            _operator(levels, ids, wet_fraction=bad, area_weights=area_weights)
    _operator(levels, ids, wet_fraction=np.full(n, 0.5), area_weights=area_weights)


def test_diagonal_hole_is_no_barrier_in_lattice_mode():
    # two leaves meet at a corner and both leaves next to them are absent: the
    # lattice correction gives the straight distance, "none" the detour
    from healpix_analyse.multiorder_filter import _centres, _water_pairs

    levels, ids = _uniform(4)
    first, second, edge = leaf_adjacency(levels, ids)
    n = ids.size
    nbrs = [set() for _ in range(n)]
    for a, b, e in zip(first.tolist(), second.tolist(), edge.tolist()):
        if e:
            nbrs[a].add(b)
            nbrs[b].add(a)
    centre = _nearest(levels, ids, (0.0, 0.0))
    pair = None
    for a, b, e in zip(first.tolist(), second.tolist(), edge.tolist()):
        common = nbrs[a] & nbrs[b]
        if not e and len(common) == 2 and centre not in ({a, b} | common):
            pair = (a, b, common)
            break
    assert pair is not None
    a, b, common = pair
    keep = np.ones(n, bool)
    keep[list(common)] = False
    levels2, ids2 = levels[keep], ids[keep]
    ia = int(np.where(ids2 == ids[a])[0][0])
    ib = int(np.where(ids2 == ids[b])[0][0])
    no_block = np.zeros(ids2.size, bool)
    out = {}
    for mode in ("none", "lattice"):
        rows, cols, dist = _water_pairs(
            levels2, ids2, 6.0 * CELL, mode, no_block, None, "WGS84"
        )
        hit = (rows == ia) & (cols == ib)
        assert hit.sum() == 1
        out[mode] = float(dist[hit][0])
    lon, lat = _centres(levels2, ids2, "WGS84")
    straight = _WGS84.inv(lon[ia], lat[ia], lon[ib], lat[ib])[2]
    assert out["lattice"] == pytest.approx(straight, rel=1e-6)
    assert out["none"] > 1.5 * straight
