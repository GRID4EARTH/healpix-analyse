"""Tests for ``build_healpix_adjacency`` (healpix-geo backend, vectorised)."""

import numpy as np
import pytest
import torch

from healpix_analyse._topology import nested_neighbours
from healpix_analyse.minkowski import (
    build_healpix_adjacency,
    minkowski_functionals_healpix,
)


def _reference_loop(level, cell_ids, neighbours_of):
    """The previous per-cell Python loop, with a pluggable neighbour lookup."""

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    gid_to_lid = {int(g): l for l, g in enumerate(cell_ids)}
    edges_set = set()
    nbrs_local = []
    for lid, gid in enumerate(cell_ids):
        local = set()
        for ng in neighbours_of(int(gid)):
            if ng >= 0 and int(ng) in gid_to_lid:
                nll = gid_to_lid[int(ng)]
                edges_set.add((min(lid, nll), max(lid, nll)))
                local.add(nll)
        nbrs_local.append(local)
    triangles_set = set()
    for i, j in edges_set:
        for k in nbrs_local[i] & nbrs_local[j]:
            triangles_set.add(tuple(sorted((i, j, k))))
    edges = np.array(sorted(edges_set), dtype=np.int64).reshape(-1, 2)
    triangles = np.array(sorted(triangles_set), dtype=np.int64).reshape(-1, 3)
    return edges, triangles


def _geo_neighbours(level):
    return lambda gid: nested_neighbours(
        np.array([gid]), level, connectivity="edge_or_vertex"
    )[0]


def _regional_cells(level, seed=0):
    """A shuffled regional domain made of two disjoint NESTED blocks."""

    npix = 12 * 4**level
    rng = np.random.default_rng(seed)
    block = npix // 48
    cells = np.concatenate(
        [np.arange(0, block), np.arange(5 * block, 7 * block)]
    )
    return rng.permutation(cells)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_full_sphere_matches_reference_loop(level):
    cells = np.arange(12 * 4**level)
    edges, triangles = build_healpix_adjacency(level)
    ref_e, ref_t = _reference_loop(level, cells, _geo_neighbours(level))

    assert edges.dtype == torch.long and triangles.dtype == torch.long
    np.testing.assert_array_equal(edges.numpy(), ref_e)
    np.testing.assert_array_equal(triangles.numpy(), ref_t)


@pytest.mark.parametrize("level", [2, 4])
def test_regional_domain_matches_reference_loop(level):
    cells = _regional_cells(level)
    edges, triangles = build_healpix_adjacency(level, cells)
    ref_e, ref_t = _reference_loop(level, cells, _geo_neighbours(level))

    np.testing.assert_array_equal(edges.numpy(), ref_e)
    np.testing.assert_array_equal(triangles.numpy(), ref_t)


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize(
    "cells",
    [[4, 6, 5, 5], [5, 6, 7, 4, 6, 13, 5, 100]],
)
def test_duplicated_cells_raise(cells, nest):
    with pytest.raises(ValueError, match="duplicated"):
        build_healpix_adjacency(2, np.array(cells), nest=nest)


@pytest.mark.parametrize("level", [1, 3])
def test_full_sphere_structure(level):
    edges, triangles = build_healpix_adjacency(level)
    npix = 12 * 4**level
    e = edges.numpy()
    t = triangles.numpy()

    assert np.all(e[:, 0] < e[:, 1])
    assert np.all((t[:, 0] < t[:, 1]) & (t[:, 1] < t[:, 2]))
    # Every cell has 7 or 8 neighbours on the full sphere (level >= 1).
    degree = np.bincount(e.ravel(), minlength=npix)
    assert set(np.unique(degree)) <= {7, 8}
    # Every triangle is a 3-clique of the edge graph.
    edge_set = {tuple(p) for p in e}
    for a, b, c in t:
        assert (a, b) in edge_set and (a, c) in edge_set and (b, c) in edge_set

    x = torch.ones(1, npix, dtype=torch.float64)
    mf = minkowski_functionals_healpix(x, edges, triangles, threshold=0.5,
                                       temperature=1e3)
    assert torch.isfinite(mf["W2"]).all()


def test_ring_ordering_is_the_same_graph():
    healpix_geo = pytest.importorskip("healpix_geo")
    level = 2
    nested_cells = _regional_cells(level, seed=3)
    ring_cells = np.asarray(
        healpix_geo.nested.to_ring(nested_cells.astype(np.uint64), level),
        dtype=np.int64,
    )

    e_n, t_n = build_healpix_adjacency(level, nested_cells, nest=True)
    e_r, t_r = build_healpix_adjacency(level, ring_cells, nest=False)

    torch.testing.assert_close(e_r, e_n)
    torch.testing.assert_close(t_r, t_n)


def test_empty_and_isolated_cells_give_two_column_tensors():
    edges, triangles = build_healpix_adjacency(3, np.array([], dtype=np.int64))
    assert edges.shape == (0, 2) and triangles.shape == (0, 3)

    # Two cells far apart: no edge, no triangle.
    edges, triangles = build_healpix_adjacency(3, np.array([0, 700]))
    assert edges.shape == (0, 2) and triangles.shape == (0, 3)


@pytest.mark.parametrize("level", [-1, 1.5, True])
def test_invalid_level(level):
    with pytest.raises(ValueError):
        build_healpix_adjacency(level, np.array([0]))


def test_cell_out_of_range():
    with pytest.raises(ValueError):
        build_healpix_adjacency(1, np.array([0, 48]))
    with pytest.raises(ValueError):
        build_healpix_adjacency(1, np.array([0, 48]), nest=False)


@pytest.mark.parametrize("nest", [True, False])
@pytest.mark.parametrize("level", [0, 2])
def test_matches_healpy_loop_when_available(level, nest):
    hp = pytest.importorskip("healpy")
    nside = 2**level
    cells = (
        np.arange(12 * nside**2)
        if level == 0
        else _regional_cells(level, seed=1)
    )
    edges, triangles = build_healpix_adjacency(level, cells, nest=nest)
    ref_e, ref_t = _reference_loop(
        level,
        cells,
        lambda gid: hp.get_all_neighbours(nside, gid, nest=nest),
    )

    np.testing.assert_array_equal(edges.numpy(), ref_e)
    np.testing.assert_array_equal(triangles.numpy(), ref_t)
