"""Tests for the NESTED-block <-> square-tile reordering used by dino.py.

The DINOv3 backbone itself is not exercised here (gated weights); a tiny
stand-in ``nn.Module`` with the ``forward_features`` interface is used to
check shapes and cell-id bookkeeping of :func:`GetDINOV3SAT`.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from healpix_analyse.dino import (
    GetDINOV3SAT,
    nested_to_tiles,
    nested_to_xy,
    tile_grid_cell_ids,
    tiles_to_nested,
    xy_to_nested,
)

healpy = pytest.importorskip("healpy")


def test_xy_roundtrip_matches_healpy():
    """nested_to_xy must follow the HEALPix nest2xyf bit convention."""
    nside = 2 ** 10
    rng = np.random.default_rng(0)
    pix = rng.integers(0, 12 * nside * nside, size=5000, dtype=np.int64)
    x_ref, y_ref, face = healpy.pix2xyf(nside, pix, nest=True)
    rel = pix - face.astype(np.int64) * nside * nside
    x, y = nested_to_xy(rel)
    assert np.array_equal(x, x_ref)
    assert np.array_equal(y, y_ref)
    assert np.array_equal(xy_to_nested(x, y), rel)


def test_tiles_roundtrip_and_orientation():
    level, parent_level = 12, 8
    k = level - parent_level
    S = 2 ** k
    nside = 2 ** level
    parents = np.array([3, 7 * 4 ** 5 + 11, 12 * 4 ** parent_level - 1], dtype=np.int64)
    cell_id = (parents[:, None] << (2 * k)) + np.arange(S * S)[None]
    cell_id = cell_id.reshape(-1)
    rng = np.random.default_rng(1)
    perm = rng.permutation(cell_id.size)
    cell_id = cell_id[perm]
    data = rng.normal(size=(cell_id.size, 3)).astype(np.float32)

    tiles, pids, valid, cov = nested_to_tiles(data, cell_id, level, parent_level)
    assert tiles.shape == (3, 3, S, S)
    assert np.array_equal(pids, np.sort(parents))
    assert valid.all() and np.allclose(cov, 1.0)

    d2, id2 = tiles_to_nested(tiles, pids, parent_level, level)
    order = np.argsort(cell_id)
    assert np.array_equal(id2[np.argsort(id2)], cell_id[order])
    assert np.allclose(d2[np.argsort(id2)], data[order])

    # North corner (max x, max y) must be the top-right pixel of the tile.
    grid = tile_grid_cell_ids(pids, parent_level, level)
    lat = healpy.pix2ang(nside, grid[0].reshape(-1), nest=True, lonlat=True)[1]
    lat = lat.reshape(S, S)
    assert lat[0, S - 1] == lat.max()
    assert lat[S - 1, 0] == lat.min()


def test_missing_pixels_fill_and_coverage():
    level, parent_level = 10, 8
    S = 4
    cell_id = np.arange(S * S, dtype=np.int64)[:6]           # 6 of 16 pixels
    data = np.ones((6, 2), np.float32) * np.array([[2.0, 4.0]])
    data[0, 1] = np.nan
    tiles, pids, valid, cov = nested_to_tiles(data, cell_id, level, parent_level)
    assert pids.tolist() == [0]
    assert valid.sum() == 5
    assert np.isclose(cov[0], 5 / 16)
    assert np.allclose(tiles[0, 0], 2.0) and np.allclose(tiles[0, 1], 4.0)


class _FakeDino(nn.Module):
    """Stand-in with the torch-hub forward_features interface (patch 16)."""

    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 16, stride=16)

    def forward_features(self, x):
        p = self.proj(x).flatten(2).transpose(1, 2)              # [B, P, D]
        return {"x_norm_clstoken": p.mean(1), "x_norm_patchtokens": p}


def test_get_dinov3sat_shapes_with_fake_backbone():
    level, parent_level = 16, 10                              # 64 px tiles, 4x4 patches
    k = level - parent_level
    S = 2 ** k
    parents = np.array([5, 9], dtype=np.int64)
    cell_id = ((parents[:, None] << (2 * k)) + np.arange(S * S)[None]).reshape(-1)
    rng = np.random.default_rng(2)
    data = rng.uniform(0, 0.4, size=(cell_id.size, 4)).astype(np.float32)   # 4 bands

    res = GetDINOV3SAT(
        data, cell_id, level, parent_level, projection="nested",
        model=_FakeDino(), bands=(2, 1, 0), return_patches=True,
        pooling="cls+mean", device="cpu", batch_size=1,
    )
    assert res.embedding.shape == (2, 16)
    assert np.array_equal(res.cell_id, parents)
    assert res.patch_level == level - 4
    assert res.patch_embedding.shape == (2 * 16, 8)
    # every patch cell is an ancestor of exactly 256 input pixels
    anc = cell_id >> 8
    counts = {int(c): int((anc == c).sum()) for c in res.patch_cell_id}
    assert set(counts.values()) == {256}
    # patch cells are children of the right parent
    assert np.array_equal(res.patch_cell_id >> (2 * (res.patch_level - parent_level)),
                          np.repeat(parents, 16))


def test_get_dinov3sat_rejects_small_tiles():
    with pytest.raises(ValueError):
        GetDINOV3SAT(np.zeros((4, 3)), np.arange(4), 10, 8, model=_FakeDino())


def test_duplicate_cell_ids_are_aggregated():
    """Projected data may hit the same cell twice; duplicates are averaged, not rejected."""
    level, parent_level = 10, 8
    cell_id = np.array([0, 1, 1, 2, 3], dtype=np.int64)
    data = np.array([[1.0], [2.0], [4.0], [np.nan], [5.0]], np.float32)

    with pytest.warns(RuntimeWarning, match="duplicate"):
        tiles, pids, valid, cov = nested_to_tiles(data, cell_id, level, parent_level, fill="nan")
    assert pids.tolist() == [0]
    assert tiles.shape == (1, 1, 4, 4)
    # cell 1 -> (x, y) = (1, 0) -> row 3, col 1; mean of 2 and 4
    assert tiles[0, 0, 3, 1] == 3.0
    assert valid.sum() == 3                       # cells 0, 1, 3 (cell 2 is NaN)
    assert np.isclose(cov[0], 3 / 16)

    with pytest.warns(RuntimeWarning):
        tiles_first, _, _, _ = nested_to_tiles(
            data, cell_id, level, parent_level, fill="nan", duplicates="first")
    assert tiles_first[0, 0, 3, 1] == 2.0

    with pytest.raises(ValueError, match="duplicate"):
        nested_to_tiles(data, cell_id, level, parent_level, duplicates="error")


# ---------------------------------------------------------------------------
# Tangent-plane projection and sliding-window oversampling
# ---------------------------------------------------------------------------

def test_healpix_cells_are_not_square():
    """The premise of the tangent mode: equal area, unequal shape."""
    from healpix_analyse.dino import EARTH_RADIUS_M, nested_to_xy, xy_to_nested
    level, nside = 19, 2 ** 19
    p = healpy.ang2pix(nside, 10.55, 52.31, nest=True, lonlat=True)
    face = p // (nside * nside)
    x, y = nested_to_xy(np.array([p - face * nside * nside]))
    v = lambda dx, dy: np.array(healpy.pix2vec(
        nside, face * nside * nside + int(xy_to_nested(np.array([x[0] + dx]),
                                                       np.array([y[0] + dy]))[0]), nest=True))
    v0 = v(0, 0)
    dx_m = np.linalg.norm((v(1, 0) - v0) * EARTH_RADIUS_M)
    dy_m = np.linalg.norm((v(0, 1) - v0) * EARTH_RADIUS_M)
    assert dy_m / dx_m > 1.5          # ~1.57 at this latitude: strongly anisotropic


def test_tangent_grid_is_north_up_and_metric():
    from healpix_analyse.dino import EARTH_RADIUS_M, healpix_gsd_m, tangent_grid_lonlat
    gsd = healpix_gsd_m(19)
    lon, lat = tangent_grid_lonlat([10.55], [52.31], 9, gsd / EARTH_RADIUS_M)
    assert lat[0, 0, 4] > lat[0, 8, 4]                       # north is up
    assert lon[0, 4, 8] > lon[0, 4, 0]                       # east is right
    # the step really is the requested ground sampling, in both directions
    d_north = np.radians(lat[0, 3, 4] - lat[0, 4, 4]) * EARTH_RADIUS_M
    d_east = (np.radians(lon[0, 4, 5] - lon[0, 4, 4])
              * np.cos(np.radians(lat[0, 4, 4])) * EARTH_RADIUS_M)
    assert abs(d_north - gsd) < 0.02 * gsd
    assert abs(d_east - gsd) < 0.02 * gsd


def test_sample_healpix_recovers_a_known_field():
    """Sampling a smooth field on the tangent grid must match its analytic value."""
    from healpix_analyse.dino import healpix_gsd_m, sample_healpix, tangent_grid_lonlat
    level, nside = 8, 2 ** 8
    cells = np.arange(12 * nside * nside, dtype=np.int64)
    lon_c, lat_c = healpy.pix2ang(nside, cells, nest=True, lonlat=True)
    field = np.sin(np.radians(lat_c))[:, None].astype(np.float32)
    lon, lat = tangent_grid_lonlat([30.0], [20.0], 8, healpix_gsd_m(level) / 6371000.0)
    vals, valid = sample_healpix(field, cells, level, lon, lat)
    assert valid.all()
    assert np.allclose(vals[..., 0], np.sin(np.radians(lat)), atol=1e-5)


def test_tangent_tiles_shapes_and_partial_sky():
    from healpix_analyse.dino import tangent_tiles
    level, parent_level = 12, 8
    S = 2 ** (level - parent_level)
    parents = np.array([100, 101], dtype=np.int64)
    cell_id = ((parents[:, None] << 8) + np.arange(256)[None]).reshape(-1)
    data = np.zeros((cell_id.size, 3), np.float32)
    tiles, ids, valid, cov, lon, lat = tangent_tiles(
        data, cell_id, level, parent_level, fill="nan")
    H, W = tiles.shape[-2:]
    assert tiles.shape == (2, 3, H, W) and lon.shape == (2, H, W)
    assert H % 16 == 0 and W % 16 == 0
    # the default image is sized to contain the whole parallelogram-shaped cell,
    # so it is larger than the nested block and rectangular
    assert H >= S and W >= S and (H, W) != (S, S)
    assert np.array_equal(ids, parents)
    assert 0.0 < cov.max() < 1.0          # the cell does not fill the rectangle


def test_tangent_output_tiles_the_cells_without_holes():
    """Every cell of the token level must carry exactly one embedding."""
    level, parent_level = 16, 10
    k = level - parent_level
    S = 2 ** k
    parents = np.array([5, 6], dtype=np.int64)
    cell_id = ((parents[:, None] << (2 * k)) + np.arange(S * S)[None]).reshape(-1)
    rng = np.random.default_rng(5)
    data = rng.uniform(0, 0.4, size=(cell_id.size, 3)).astype(np.float32)

    for n in (1, 2):
        res = GetDINOV3SAT(data, cell_id, level, parent_level, over_sample=n,
                           model=_FakeDino(), return_patches=True, device="cpu")
        tl = level - 4 + int(np.log2(n))
        assert res.patch_level == tl
        # exactly the children of every tile, once each: a complete tiling
        expected = ((res.cell_id[:, None] << (2 * (tl - parent_level)))
                    + np.arange(4 ** (tl - parent_level))[None]).reshape(-1)
        assert np.array_equal(np.sort(res.patch_cell_id), np.sort(expected))
        assert np.unique(res.patch_cell_id).size == res.patch_cell_id.size
        assert res.patch_embedding.shape[0] == res.patch_cell_id.size


def test_over_sample_interleaves_tokens():
    """over_sample=2 must give a 2x denser token field, with the shifted passes
    interleaved rather than averaged."""
    level, parent_level = 16, 10
    k = level - parent_level
    S = 2 ** k
    parents = np.array([5], dtype=np.int64)
    cell_id = ((parents[:, None] << (2 * k)) + np.arange(S * S)[None]).reshape(-1)
    rng = np.random.default_rng(3)
    data = rng.uniform(0, 0.4, size=(cell_id.size, 3)).astype(np.float32)
    model = _FakeDino()

    r1 = GetDINOV3SAT(data, cell_id, level, parent_level, projection="nested",
                      model=model, return_patches=True, device="cpu")
    r2 = GetDINOV3SAT(data, cell_id, level, parent_level, projection="nested",
                      over_sample=2, model=model, return_patches=True, device="cpu")
    assert r1.patch_level == level - 4
    assert r2.patch_level == level - 3                    # one level finer
    g1 = int(np.sqrt(r1.patch_embedding.shape[0]))
    g2 = int(np.sqrt(r2.patch_embedding.shape[0]))
    assert g2 == 2 * (g1 - 1)                             # 2x denser, minus the crop border
    assert r2.patch_cell_id.size == r2.patch_embedding.shape[0]
    assert np.unique(r2.patch_cell_id).size == r2.patch_cell_id.size

    with pytest.raises(ValueError, match="power of two"):
        GetDINOV3SAT(data, cell_id, level, parent_level, over_sample=3, model=model)


def test_tangent_projection_end_to_end():
    level, parent_level = 16, 10
    k = level - parent_level
    S = 2 ** k
    parents = np.array([5, 6], dtype=np.int64)
    cell_id = ((parents[:, None] << (2 * k)) + np.arange(S * S)[None]).reshape(-1)
    rng = np.random.default_rng(4)
    data = rng.uniform(0, 0.4, size=(cell_id.size, 3)).astype(np.float32)

    res = GetDINOV3SAT(data, cell_id, level, parent_level, projection="tangent",
                       model=_FakeDino(), return_patches=True, device="cpu",
                       min_coverage=0.0)
    n_child = 4 ** (res.patch_level - parent_level)
    assert res.patch_embedding.shape[0] == res.cell_id.size * n_child
    assert res.patch_lon is not None and res.patch_lat is not None
    assert res.patch_lon.shape == res.patch_cell_id.shape
    # patch_lon / patch_lat are the centres of the cells they are reported for
    ref = healpy.ang2pix(2 ** res.patch_level, res.patch_lon, res.patch_lat,
                         nest=True, lonlat=True)
    assert np.array_equal(ref.astype(np.int64), res.patch_cell_id)
    assert res.projection == "tangent" and res.gsd_m > 0


def test_percell_one_embedding_per_cell():
    """projection='percell': one tangent plane and one embedding per output cell."""
    level, parent_level = 16, 13            # cells of 8 px: smaller than a patch
    parents = np.array([5], dtype=np.int64)
    cell_id = ((parents[:, None] << 12) + np.arange(4 ** 6)[None]).reshape(-1)
    rng = np.random.default_rng(6)
    data = rng.uniform(0, 0.4, size=(cell_id.size, 3)).astype(np.float32)

    res = GetDINOV3SAT(data, cell_id, level, parent_level, projection="percell",
                       context_px=64, model=_FakeDino(), device="cpu", batch_size=32)
    expected = np.unique(cell_id >> (2 * (level - parent_level)))
    assert np.array_equal(np.sort(res.cell_id), expected)          # complete, no holes
    assert res.embedding.shape[0] == expected.size
    assert res.parent_level == parent_level and res.projection == "percell"
    # the patch_* view mirrors the same field, for code written for the tiled modes
    assert np.array_equal(res.patch_cell_id, res.cell_id)
    assert res.patch_level == parent_level
    # centres reported are the cell centres
    ref = healpy.ang2pix(2 ** parent_level, res.patch_lon, res.patch_lat,
                         nest=True, lonlat=True)
    assert np.array_equal(ref.astype(np.int64), res.cell_id)

    # a subset of cells can be requested explicitly
    sub = expected[:7]
    r2 = GetDINOV3SAT(data, cell_id, level, parent_level, projection="percell",
                      context_px=64, out_cells=sub, model=_FakeDino(), device="cpu")
    assert np.array_equal(r2.cell_id, sub)

    # the tiled modes need level - parent_level >= 4; per-cell does not
    with pytest.raises(ValueError, match="percell"):
        GetDINOV3SAT(data, cell_id, level, parent_level, model=_FakeDino())
