"""Tests of the land barrier (``blocked_cells``) and of the source
normalisation of the filter operators.

Scenes are drawn on a patch of real level-8 HEALPix cells with a local
tangent-plane coordinate system (``u``, ``v``, in cells).  The reference
distance through water is computed here with a plain heap Dijkstra on the
whole patch (no radius bound), with the edge-neighbour API used to decide
which vertex-contact edges are leaky; the library searches with a bounded
vectorised iteration.
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest
import torch
from healpix_geo import nested
from pyproj import Geod

from healpix_analyse import (
    apply_filter_operator,
    gaussian_filter,
    gaussian_filter_operator,
    radial_filter_operator,
)
from healpix_analyse._land_barrier import (
    _SEARCH_FACTOR,
    domain_neighbour_table,
    passable_edges,
)
from healpix_analyse._topology import nested_neighbours
from healpix_analyse.radial_filter import _cell_size_m

_WGS84 = Geod(ellps="WGS84")
LEVEL = 8
CELL = _cell_size_m(LEVEL)
SIGMA_CELLS = 2.0
RADIUS = 4.0 * SIGMA_CELLS * CELL
EARTH = 6_371_007.1809


def _patch(k: int) -> np.ndarray:
    centre = np.asarray(
        nested.lonlat_to_healpix(
            np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
        ),
        dtype=np.uint64,
    )
    ring = np.asarray(nested.kth_neighbourhood(centre, LEVEL, k))[0]
    return ring[ring >= 0].astype(np.uint64)


IDS = _patch(24)


def _plane(ids: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = (np.asarray(a) for a in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    x = np.radians(lon - 10.0) * np.cos(np.radians(45.0)) * EARTH
    y = np.radians(lat - 45.0) * EARTH
    angle = np.radians(angle_deg)
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    return u / CELL, v / CELL


def _strip(angle: float, width: float, half_length: float, gap: float = 0.0):
    u, v = _plane(IDS, angle)
    inside = (np.abs(u) <= width / 2) & (np.abs(v) <= half_length)
    inside &= np.abs(v) >= gap / 2
    return IDS[inside]


def _nearest(angle: float, point: tuple[float, float]) -> int:
    u, v = _plane(IDS, angle)
    return int(np.argmin((u - point[0]) ** 2 + (v - point[1]) ** 2))


def _dense(op) -> np.ndarray:
    n = op.n_columns
    rows = np.repeat(np.arange(op.row_offsets.size - 1), np.diff(op.row_offsets))
    matrix = np.zeros((op.row_offsets.size - 1, n))
    np.add.at(matrix, (rows, op.column_indices), op.weights)
    return matrix


def _operator(blocked, normalization="source", ids=IDS, sigma=SIGMA_CELLS, **kw):
    return gaussian_filter_operator(
        ids,
        LEVEL,
        sigma_m=sigma * CELL,
        normalization=normalization,
        blocked_cells=blocked,
        **kw,
    )


# ---------------------------------------------------------------- reference


def _reference_distances(ids: np.ndarray, blocked_ids: np.ndarray, source: int):
    """Unbounded heap Dijkstra: ``(D, D0)`` from ``source`` to every cell."""
    position = {int(c): i for i, c in enumerate(ids)}
    blocked = np.isin(ids, blocked_ids)
    lon, lat = (np.asarray(a) for a in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    allneigh = np.asarray(nested.neighbours(ids, LEVEL, connectivity="all"))
    edge = np.asarray(nested.neighbours(ids, LEVEL, connectivity="edge"))
    edge_sets = [set(int(c) for c in row if c >= 0) for row in edge]

    def graph(use_blocked: bool):
        adjacency = [[] for _ in ids]
        for a in range(ids.size):
            for b_id in allneigh[a]:
                if b_id < 0 or int(b_id) not in position:
                    continue
                b = position[int(b_id)]
                if use_blocked:
                    if blocked[a] or blocked[b]:
                        continue
                    if int(b_id) not in edge_sets[a]:
                        # vertex contact: the cells that touch the vertex
                        between = [
                            position[c]
                            for c in edge_sets[a] & edge_sets[b]
                            if c in position
                        ]
                        if not any(not blocked[c] for c in between):
                            continue
                length = _WGS84.inv(lon[a], lat[a], lon[b], lat[b])[2]
                adjacency[a].append((b, length))
        return adjacency

    def dijkstra(adjacency):
        best = np.full(ids.size, np.inf)
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

    return dijkstra(graph(True)), dijkstra(graph(False)), lon, lat


def _reference_column(ids, blocked_ids, source, sigma_cells=SIGMA_CELLS):
    """Source-normalised weights of one source cell, brute force."""
    path, free, lon, lat = _reference_distances(ids, blocked_ids, source)
    straight = np.array(
        [
            _WGS84.inv(lon[source], lat[source], lon[i], lat[i])[2]
            for i in range(ids.size)
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        distance = straight * path / free
    distance[source] = 0.0
    radius = 4.0 * sigma_cells * CELL
    weights = np.where(
        np.isfinite(distance) & (distance <= radius),
        np.exp(-0.5 * (distance / (sigma_cells * CELL)) ** 2),
        0.0,
    )
    return weights / weights.sum()


# -------------------------------------------------------------------- tests


@pytest.mark.parametrize(
    ("level", "lon", "lat"), [(8, 0.0, 45.0), (8, 10.0, 70.0), (10, 354.0, 48.0), (9, 0.0, 41.0)]
)
def test_search_factor_covers_the_lattice(level, lon, lat):
    """Lattice path / straight distance stays below the search factor."""
    from healpix_analyse._land_barrier import edge_lengths, path_distances
    from healpix_analyse._neighbourhood import build_metric_neighbourhood_geometry

    centre = np.asarray(
        nested.lonlat_to_healpix(np.array([lon]), np.array([lat]), level, ellipsoid="WGS84"),
        dtype=np.uint64,
    )
    ring = np.asarray(nested.kth_neighbourhood(centre, level, 30))[0]
    ids = ring[ring >= 0].astype(np.uint64)
    cell = _cell_size_m(level)
    radius = 10.0 * cell
    # a search disc well beyond the evaluated one, so that paths stay inside
    geometry = build_metric_neighbourhood_geometry(ids, 1.8 * radius, level)
    table = domain_neighbour_table(ids, level)
    length = edge_lengths(ids, table, level)
    allowed = passable_edges(table, np.zeros(ids.size, bool))
    rows = np.repeat(np.arange(ids.size), np.diff(geometry.row_offsets))
    columns = geometry.neighbour_indices.astype(np.int64)
    _, free = path_distances(
        rows, columns, ids.size, table, length, allowed, allowed
    )
    keep = (rows != columns) & (geometry.distance_m <= radius)
    # centre of the patch only: the disc of a cell near its edge is cut
    core = np.asarray(nested.kth_neighbourhood(centre, level, 14))[0]
    inner = np.isin(ids, core[core >= 0].astype(np.uint64))
    keep &= inner[rows]
    ratio = free[keep] / geometry.distance_m[keep]
    assert ratio.size > 1000
    assert ratio.max() < _SEARCH_FACTOR - 0.03
    assert _SEARCH_FACTOR >= 1.5


@pytest.mark.parametrize("blocked", [None, np.array([], dtype=np.uint64)])
@pytest.mark.parametrize("normalization", ["present", "full_disc"])
@pytest.mark.parametrize("width_rule", ["sampled", "matched"])
def test_no_blocked_cells_is_the_plain_operator(blocked, normalization, width_rule):
    plain = gaussian_filter_operator(
        IDS,
        LEVEL,
        sigma_m=SIGMA_CELLS * CELL,
        normalization=normalization,
        width_rule=width_rule,
    )
    same = gaussian_filter_operator(
        IDS,
        LEVEL,
        sigma_m=SIGMA_CELLS * CELL,
        normalization=normalization,
        width_rule=width_rule,
        blocked_cells=blocked,
    )
    for a, b in zip(plain[:3], same[:3]):
        assert np.array_equal(a, b)


def test_blocked_cells_outside_the_domain_are_ignored():
    outside = np.setdiff1d(_patch(26), IDS)
    plain = _operator(None)
    other = _operator(outside)
    for a, b in zip(plain[:3], other[:3]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("normalization", ["present", "full_disc"])
def test_open_water_far_from_land_equals_plain(normalization):
    """Rows with no blocked cell within 1.5 R are the plain rows."""
    land = _strip(0, 1.0, 3.0)
    plain = _dense(_operator(None, normalization=normalization))
    barrier = _dense(_operator(land, normalization=normalization))
    u, v = _plane(IDS, 0)
    near = np.zeros(IDS.size, bool)
    for p in np.flatnonzero(np.isin(IDS, land)):
        near |= np.hypot(u - u[p], v - v[p]) < _SEARCH_FACTOR * 4 * SIGMA_CELLS
    far = ~near
    assert far.sum() > 500
    np.testing.assert_allclose(barrier[far], plain[far], rtol=0, atol=1e-12)


@pytest.mark.parametrize("angle", [0, 45, 90, 20])
@pytest.mark.parametrize("width", [1.0, 2.0, 3.0])
def test_unbroken_strip_blocks_all_mass(angle, width):
    land = _strip(angle, width, 14.0)
    source = _nearest(angle, (-3.0, 0.0))
    u, _ = _plane(IDS, angle)
    column = _dense(_operator(land))[:, source]
    far_side = u > width / 2 + 0.5
    assert column[far_side].sum() == 0.0
    assert column[~far_side].sum() == pytest.approx(1.0, abs=1e-12)
    # without the barrier the same mass would cross
    plain = _dense(_operator(None))[:, source]
    assert plain[far_side].sum() > 1e-3


@pytest.mark.parametrize("angle", [0, 45, 90])
@pytest.mark.parametrize("width", [1.0, 2.0])
def test_strait_equals_distance_through_water(angle, width):
    land = _strip(angle, width, 14.0, gap=4.0)
    source = _nearest(angle, (-3.0, 0.0))
    u, _ = _plane(IDS, angle)
    column = _dense(_operator(land))[:, source]
    expected = _reference_column(IDS, land, source)
    assert column[u > width / 2 + 0.5].sum() > 1e-3
    np.testing.assert_allclose(column, expected, rtol=0, atol=1e-10)


def test_island_mass_goes_around():
    u, v = _plane(IDS, 0)
    land = IDS[np.hypot(u, v) <= 1.6]
    source = _nearest(0, (-3.0, 0.0))
    column = _dense(_operator(land))[:, source]
    plain = _dense(_operator(None))[:, source]
    behind = _nearest(0, (3.0, 0.0))
    assert 0.0 < column[behind] < plain[behind]
    assert column[np.isin(IDS, land)].sum() == 0.0
    np.testing.assert_allclose(
        column, _reference_column(IDS, land, source), rtol=0, atol=1e-10
    )


def test_bay_dead_end():
    """A U-shaped bay open to the east: no weight leaks through its walls."""
    u, v = _plane(IDS, 0)
    wall = (np.abs(v) <= 4.5) & (np.abs(v) >= 3.5) & (u >= -6) & (u <= 3)
    back = (np.abs(v) <= 4.5) & (u >= -6.5) & (u <= -5.5)
    land = IDS[wall | back]
    source = _nearest(0, (-3.0, 0.0))
    column = _dense(_operator(land))[:, source]
    outside_north = (v > 5.0) & (u < 0)
    assert column[outside_north].sum() == 0.0
    assert column.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        column, _reference_column(IDS, land, source), rtol=0, atol=1e-10
    )


@pytest.mark.parametrize("normalization", ["present", "full_disc", "source"])
def test_blocked_source_and_target_are_empty(normalization):
    land = _strip(0, 2.0, 10.0)
    op = _operator(land, normalization=normalization)
    blocked = np.isin(IDS, land)
    counts = np.diff(op.row_offsets)
    assert np.all(counts[blocked] == 0)
    assert np.all(counts[~blocked] > 0)
    assert not np.any(blocked[op.column_indices])
    values = np.ones(IDS.size)
    out = apply_filter_operator(values, *op)
    assert np.all(out[blocked] == 0.0)


def test_mass_per_step_lost_and_kept():
    land = _strip(0, 2.0, 14.0)
    blocked = np.isin(IDS, land)
    u, v = _plane(IDS, 0)
    source = _nearest(0, (-2.5, 0.0))
    state = np.zeros(IDS.size)
    state[source] = 1.0

    lost = _operator(land, normalization="full_disc")
    kept = _operator(land, normalization="source")
    present = _operator(land, normalization="present")

    one_lost = apply_filter_operator(state, *lost)
    one_kept = apply_filter_operator(state, *kept)
    assert one_kept.sum() == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < one_lost.sum() < 1.0 - 1e-3
    assert np.all(one_lost >= 0) and np.all(one_kept >= 0)
    assert one_lost.sum() <= one_kept.sum()

    # many steps: kept stays 1, lost keeps shrinking, never above 1
    lost_state, kept_state = state.copy(), state.copy()
    for _ in range(10):
        lost_state = apply_filter_operator(lost_state, *lost)
        kept_state = apply_filter_operator(kept_state, *kept)
        assert kept_state.sum() == pytest.approx(1.0, abs=1e-10)
        assert lost_state.sum() <= 1.0 + 1e-12
    assert lost_state.sum() < 0.9

    # row / column sums and constant field
    water = ~blocked
    cols = _dense(kept).sum(axis=0)
    np.testing.assert_allclose(cols[water], 1.0, atol=1e-12)
    assert np.all(cols[blocked] == 0.0)
    rows_lost = _dense(lost).sum(axis=1)
    assert np.all(rows_lost[water] <= 1.0 + 1e-12)
    assert rows_lost[water].min() < 0.99
    np.testing.assert_allclose(
        _dense(present).sum(axis=1)[water], 1.0, atol=1e-12
    )
    ones = np.where(blocked, 0.0, 1.0)
    const_present = apply_filter_operator(ones, *present)
    np.testing.assert_allclose(const_present[water], 1.0, atol=1e-12)
    const_kept = apply_filter_operator(ones, *kept)
    assert np.abs(const_kept[water] - 1.0).max() > 1e-3
    const_lost = apply_filter_operator(ones, *lost)
    assert const_lost[water].min() < 1.0 - 1e-3


def test_source_normalisation_without_barrier():
    op = _operator(None, normalization="source")
    cols = _dense(op).sum(axis=0)
    np.testing.assert_allclose(cols, 1.0, atol=1e-12)
    # the interior equals the "present" operator, the edge differs
    present = _dense(_operator(None, normalization="present"))
    source = _dense(op)
    u, v = _plane(IDS, 0)
    inner = np.hypot(u, v) < 8
    # the two differ only through the cell-to-cell variation (1e-4) of the
    # kernel sum over the disc
    np.testing.assert_allclose(
        source[inner][:, inner], present[inner][:, inner], rtol=5e-4, atol=0
    )


def test_open_water_symmetry():
    """Away from land and from the edge the operator is symmetric.

    The weights are symmetric up to the cell-to-cell variation (about 1e-4
    relative) of the kernel sum over the disc, which the row normalisation
    divides by.
    """
    land = _strip(0, 2.0, 5.0)
    barrier = _dense(_operator(land, normalization="present"))
    full = _dense(_operator(land, normalization="full_disc"))
    u, v = _plane(IDS, 0)
    near_land = np.zeros(IDS.size, bool)
    for p in np.flatnonzero(np.isin(IDS, land)):
        near_land |= np.hypot(u - u[p], v - v[p]) < 1.6 * 4 * SIGMA_CELLS
    interior = np.flatnonzero((full.sum(axis=1) > 1 - 1e-9) & ~near_land)
    assert interior.size > 300
    sub = barrier[np.ix_(interior, interior)]
    np.testing.assert_allclose(sub, sub.T, rtol=5e-4, atol=1e-9)


@pytest.mark.parametrize("normalization", ["present", "full_disc", "source"])
def test_matched_width_with_barrier(normalization):
    land = _strip(0, 2.0, 14.0)
    op = _operator(land, normalization=normalization, width_rule="matched")
    blocked = np.isin(IDS, land)
    if normalization == "source":
        np.testing.assert_allclose(_dense(op).sum(axis=0)[~blocked], 1.0, atol=1e-12)
    u, v = _plane(IDS, 0)
    source = _nearest(0, (-3.0, 0.0))
    assert _dense(op)[u > 1.5, source].sum() == 0.0


@pytest.mark.parametrize("sigma", [0.4, 0.55])
def test_sub_cell_width_source_conserves(sigma):
    land = _strip(0, 2.0, 14.0)
    op = _operator(land, normalization="source", width_rule="matched", sigma=sigma)
    blocked = np.isin(IDS, land)
    np.testing.assert_allclose(_dense(op).sum(axis=0)[~blocked], 1.0, atol=1e-12)


def test_radial_operator_matches_gaussian_operator():
    land = _strip(0, 2.0, 10.0, gap=3.0)
    sigma = SIGMA_CELLS * CELL
    gauss = _dense(_operator(land, normalization="present"))
    radial = _dense(
        radial_filter_operator(
            IDS,
            LEVEL,
            radius_m=4.0 * sigma,
            kernel=lambda d: np.exp(-0.5 * (d / sigma) ** 2),
            normalization="present",
            blocked_cells=land,
        )
    )
    np.testing.assert_allclose(radial, gauss, rtol=0, atol=1e-12)


@pytest.mark.parametrize("normalization", ["full_disc", "source"])
def test_radial_operator_modes(normalization):
    land = _strip(0, 2.0, 10.0, gap=3.0)
    sigma = SIGMA_CELLS * CELL
    gauss = _dense(_operator(land, normalization=normalization))
    radial = _dense(
        radial_filter_operator(
            IDS,
            LEVEL,
            radius_m=4.0 * sigma,
            kernel=lambda d: np.exp(-0.5 * (d / sigma) ** 2),
            normalization=normalization,
            blocked_cells=land,
        )
    )
    np.testing.assert_allclose(radial, gauss, rtol=0, atol=1e-12)


def test_domain_subset():
    land = _strip(0, 2.0, 14.0)
    u, v = _plane(IDS, 0)
    sub = IDS[np.hypot(u, v) <= 18][::-1].copy()  # reordered subset
    op = gaussian_filter_operator(
        IDS,
        LEVEL,
        sigma_m=SIGMA_CELLS * CELL,
        normalization="source",
        domain=sub,
        blocked_cells=land,
    )
    assert op.row_offsets.size == sub.size + 1
    assert op.n_columns == IDS.size
    blocked_sub = np.isin(sub, land)
    assert np.all(np.diff(op.row_offsets)[blocked_sub] == 0)
    # reference: the same operator built on the subset alone
    alone = gaussian_filter_operator(
        sub,
        LEVEL,
        sigma_m=SIGMA_CELLS * CELL,
        normalization="source",
        blocked_cells=land,
    )
    np.testing.assert_allclose(op.weights, alone.weights, atol=1e-12)
    np.testing.assert_array_equal(op.row_offsets, alone.row_offsets)


def test_leading_dimensions_and_torch():
    land = _strip(0, 2.0, 14.0)
    op = _operator(land)
    rng = np.random.default_rng(0)
    values = rng.random((3, 2, IDS.size))
    out = apply_filter_operator(values, *op)
    assert out.shape == values.shape
    matrix = _dense(op)
    np.testing.assert_allclose(out[1, 0], matrix @ values[1, 0], atol=1e-12)
    out_torch = apply_filter_operator(torch.from_numpy(values), *op)
    assert isinstance(out_torch, torch.Tensor)
    np.testing.assert_allclose(out_torch.numpy(), out, atol=1e-12)


def test_gaussian_filter_rejects_source():
    with pytest.raises(ValueError, match="normalization"):
        gaussian_filter(
            np.ones(IDS.size),
            IDS,
            LEVEL,
            sigma_m=CELL,
            normalization="source",
        )


@pytest.mark.parametrize(
    ("blocked", "error"),
    [
        (np.array([1.5, 2.5]), TypeError),
        (np.array([True, False]), TypeError),
        (np.array([[1, 2]]), ValueError),
        (np.array([-1, 2]), ValueError),
        (np.array([3, 3], dtype=np.uint64), ValueError),
    ],
)
def test_blocked_cells_validation(blocked, error):
    with pytest.raises(error):
        _operator(blocked)


def test_passable_edges_blocks_a_diagonal_chain():
    """Two blocked cells touching a vertex close the vertex edge."""
    table = domain_neighbour_table(IDS, LEVEL)
    centre = _nearest(0, (0.0, 0.0))
    slot = 3  # N, a vertex contact
    other = table[centre, slot]
    left, right = table[centre, 2], table[centre, 4]
    blocked = np.zeros(IDS.size, bool)
    assert passable_edges(table, blocked)[centre, slot]
    blocked[[left, right]] = True
    assert not passable_edges(table, blocked)[centre, slot]
    blocked[right] = False
    assert passable_edges(table, blocked)[centre, slot]
    blocked[other] = True
    assert not passable_edges(table, blocked)[centre, slot]


@pytest.mark.parametrize("normalization", ["present", "full_disc", "source"])
@pytest.mark.parametrize("renormalize_missing", [False, True])
def test_blocked_cells_are_zero_not_nan(normalization, renormalize_missing):
    land = _strip(0, 3.0, 10.0)
    op = _operator(land, normalization=normalization)
    blocked = np.isin(IDS, land)
    values = np.random.default_rng(1).random((2, IDS.size))
    out = apply_filter_operator(values, *op, renormalize_missing=renormalize_missing)
    assert not np.isnan(out).any()
    assert np.all(out[:, blocked] == 0.0)
    out_torch = apply_filter_operator(
        torch.from_numpy(values), *op, renormalize_missing=renormalize_missing
    )
    assert not torch.isnan(out_torch).any()
    assert torch.all(out_torch[:, torch.from_numpy(blocked)] == 0.0)


def _corner_scene():
    """A cell ``a``, its vertex neighbour ``b`` and their two common cells."""
    ids = _patch(6)
    position = {int(c): i for i, c in enumerate(ids)}
    table = nested_neighbours(ids, LEVEL)  # slots as in the library
    a = _nearest_in(ids, (0.0, 0.0))
    for slot in (1, 3, 5, 7):  # W, N, E, S: vertex contacts
        b_id = int(table[a, slot])
        common = {int(table[a, slot - 1]), int(table[a, (slot + 1) % 8])}
        if b_id in position and all(c in position for c in common):
            return ids, int(ids[a]), b_id, common
    raise AssertionError("no vertex pair with two common cells")


def _nearest_in(ids, point):
    lon, lat = (np.asarray(x) for x in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    u = np.radians(lon - 10.0) * np.cos(np.radians(45.0)) * EARTH / CELL
    v = np.radians(lat - 45.0) * EARTH / CELL
    return int(np.argmin((u - point[0]) ** 2 + (v - point[1]) ** 2))


def _barrier_distance(ids, blocked_ids, a_id, b_id):
    from healpix_analyse._land_barrier import barrier_pairs
    from healpix_analyse._neighbourhood import build_metric_neighbourhood_geometry

    radius = 4.0 * CELL
    geometry = build_metric_neighbourhood_geometry(ids, _SEARCH_FACTOR * radius, LEVEL)
    blocked = np.isin(ids, np.asarray(blocked_ids, dtype=np.uint64))
    rows, columns, distance, _, _ = barrier_pairs(geometry, blocked, radius, LEVEL)
    pa = int(np.flatnonzero(ids == a_id)[0])
    pb = int(np.flatnonzero(ids == b_id)[0])
    hit = (rows == pa) & (columns == pb)
    la = [float(np.asarray(x)[0]) for x in nested.healpix_to_lonlat(np.array([a_id], np.uint64), LEVEL, ellipsoid="WGS84")]
    lb = [float(np.asarray(x)[0]) for x in nested.healpix_to_lonlat(np.array([b_id], np.uint64), LEVEL, ellipsoid="WGS84")]
    straight = _WGS84.inv(la[0], la[1], lb[0], lb[1])[2]
    return (float(distance[hit][0]) if hit.any() else np.inf), straight


def test_diagonal_hole_of_the_domain_is_no_barrier():
    ids, a, b, common = _corner_scene()
    far = int(ids[_nearest_in(ids, (3.0, 3.0))])  # a blocked cell nearby, off the pair
    assert far not in common | {a, b}
    # the two common cells are outside the domain: no barrier
    holed = ids[~np.isin(ids, np.array(sorted(common), dtype=np.uint64))]
    d, straight = _barrier_distance(holed, [far], a, b)
    assert d == pytest.approx(straight, rel=1e-9)
    # the same two cells in the domain and blocked: still a barrier
    d, straight = _barrier_distance(ids, [far, *common], a, b)
    assert d > 1.2 * straight


def test_holes_do_not_change_open_water_paths():
    """With no blocked cell, ``D`` and ``D0`` are the same (exactly)."""
    from healpix_analyse._land_barrier import edge_lengths, path_distances
    from healpix_analyse._neighbourhood import build_metric_neighbourhood_geometry

    rng = np.random.default_rng(3)
    ids = _patch(8)
    ids = ids[rng.random(ids.size) > 0.1]
    geometry = build_metric_neighbourhood_geometry(ids, 5.0 * CELL, LEVEL)
    table = domain_neighbour_table(ids, LEVEL)
    length = edge_lengths(ids, table, LEVEL)
    free_rule = passable_edges(table, np.zeros(ids.size, bool))
    rows = np.repeat(np.arange(ids.size), np.diff(geometry.row_offsets))
    columns = geometry.neighbour_indices.astype(np.int64)
    path, free = path_distances(rows, columns, ids.size, table, length, free_rule, free_rule)
    np.testing.assert_array_equal(path, free)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_factor_is_enough_with_holes(seed, monkeypatch):
    """Random holes in the domain and a blocked band: a larger search disc
    changes nothing (the ratio ``D / D0`` cancels the detours around holes)."""
    import importlib

    radial = importlib.import_module("healpix_analyse.radial_filter")
    ids = _patch(14)
    u, v = _plane(ids, 0.0)
    keep = np.random.default_rng(seed).random(ids.size) > 0.1
    ids, u, v = ids[keep], u[keep], v[keep]
    band = ids[(np.abs(u) < 1.0) & (np.abs(v) < 12) & (np.abs(v) > 2.5)]
    result = {}
    for factor in (_SEARCH_FACTOR, 3.0):
        monkeypatch.setattr(radial, "_SEARCH_FACTOR", factor)
        result[factor] = _dense(_operator(band, ids=ids, sigma=3.0))
    np.testing.assert_allclose(result[_SEARCH_FACTOR], result[3.0], rtol=0, atol=1e-13)


def test_short_wall_of_absent_cells_matches_brute_force():
    """A short wall of absent cells (a hole of the domain) next to a small
    blocked patch: the detour stays inside the candidate disc, so the
    operator equals the unbounded brute force column by column."""
    ids = _patch(14)
    u, v = _plane(ids, 0.0)
    wall = (np.abs(u) <= 0.6) & (np.abs(v) <= 3.0)
    ids, u, v = ids[~wall], u[~wall], v[~wall]
    blocked = ids[(np.abs(u - 4.0) <= 0.6) & (np.abs(v) <= 1.0)]
    assert blocked.size
    dense = _dense(_operator(blocked, ids=ids))
    # sources on both sides of the wall and next to the blocked patch
    for point in [(-2.0, 0.0), (2.0, 0.5), (-1.5, 3.5), (3.0, -2.0)]:
        source = int(np.argmin((u - point[0]) ** 2 + (v - point[1]) ** 2))
        column = _reference_column(ids, blocked, source)
        keep = ~np.isin(ids, blocked)
        np.testing.assert_allclose(
            dense[:, source][keep], column[keep], rtol=1e-9, atol=1e-12
        )
