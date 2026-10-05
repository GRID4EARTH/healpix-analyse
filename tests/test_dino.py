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

nested_geo = pytest.importorskip("healpix_geo.nested")

# Thin wrappers over healpix-geo, so the tests read like the module they check
# (healpy is deliberately not a dependency of this package).


def _lonlat(level, ids):
    lon, lat = nested_geo.healpix_to_lonlat(
        np.asarray(ids, dtype=np.uint64).reshape(-1), int(level))
    return np.asarray(lon), np.asarray(lat)


def _cells(level, lon, lat):
    ids = nested_geo.lonlat_to_healpix(
        np.atleast_1d(np.asarray(lon, dtype=np.float64)),
        np.atleast_1d(np.asarray(lat, dtype=np.float64)), int(level))
    ids = np.asarray(ids, dtype=np.int64)
    return ids if np.ndim(lon) else ids[0]


def _vecs(level, ids):
    """Unit vectors of the cell centres, shape [3, N]."""
    x, y, z = nested_geo.healpix_to_cartesian(
        np.atleast_1d(np.asarray(ids, dtype=np.uint64)), int(level))
    v = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    return v / np.linalg.norm(v, axis=0)


def test_xy_roundtrip_matches_healpix_geo():
    """nested_to_xy must follow the HEALPix base-cell-coordinate bit convention."""
    level = 10
    nside = 2 ** level
    rng = np.random.default_rng(0)
    pix = rng.integers(0, 12 * nside * nside, size=5000, dtype=np.int64)
    face, x_ref, y_ref = nested_geo.healpix_to_base_cell_coordinates(
        pix.astype(np.uint64), level)
    x_ref = np.asarray(x_ref, dtype=np.int64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    rel = pix - np.asarray(face, dtype=np.int64) * nside * nside
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
    lat = _lonlat(level, grid[0].reshape(-1))[1]
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
    p = _cells(level, 10.55, 52.31)
    face = p // (nside * nside)
    x, y = nested_to_xy(np.array([p - face * nside * nside]))
    def v(dx, dy):
        rel = int(xy_to_nested(np.array([x[0] + dx]), np.array([y[0] + dy]))[0])
        return _vecs(level, face * nside * nside + rel)[:, 0]

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
    lon_c, lat_c = _lonlat(level, cells)
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
    # the default image is square, a power of two, and large enough to contain
    # the whole parent cell -- so bigger than the equal-area 2**k block
    assert H == W and H > S and (H & (H - 1)) == 0
    # "exact" is the equal-area square, same shape as the nested block
    assert tangent_tiles(data, cell_id, level, parent_level,
                         tile_px="exact", fill="nan")[0].shape[-2:] == (S, S)
    # "cover" asks for the smallest image containing the whole parallelogram,
    # which is larger than the nested block and generally rectangular
    Hc, Wc = tangent_tiles(data, cell_id, level, parent_level,
                           tile_px="cover", fill="nan")[0].shape[-2:]
    assert Hc >= S and Wc >= S and (Hc, Wc) != (S, S)
    assert np.array_equal(ids, parents)
    assert 0.0 < cov.max() <= 1.0


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
    ref = _cells(res.patch_level, res.patch_lon, res.patch_lat)
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
    ref = _cells(parent_level, res.patch_lon, res.patch_lat)
    assert np.array_equal(ref.astype(np.int64), res.cell_id)

    # a subset of cells can be requested explicitly
    sub = expected[:7]
    r2 = GetDINOV3SAT(data, cell_id, level, parent_level, projection="percell",
                      context_px=64, out_cells=sub, model=_FakeDino(), device="cpu")
    assert np.array_equal(r2.cell_id, sub)

    # the tiled modes need level - parent_level >= 4; per-cell does not
    with pytest.raises(ValueError, match="percell"):
        GetDINOV3SAT(data, cell_id, level, parent_level, model=_FakeDino())


# ---------------------------------------------------------------------------
# over_sample in tangent mode: a true sliding window
# ---------------------------------------------------------------------------

class _MeanDino(nn.Module):
    """Exact stand-in: each token is the mean colour of its 16x16 patch."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward_features(self, x):
        tok = nn.functional.avg_pool2d(x, 16).flatten(2).transpose(1, 2)
        return {"x_norm_clstoken": tok.mean(1), "x_norm_patchtokens": tok}


def _smooth_field(level, block_level, lon0=2.35, lat0=48.85):
    """A smooth, anisotropic field on a block of cells around (lon0, lat0) plus its neighbours."""
    from healpix_analyse.dino import cell_neighbours, cells_at_lonlat, cell_centres_lonlat
    b = int(cells_at_lonlat(block_level, np.array([lon0]), np.array([lat0]))[0])
    nb = cell_neighbours(block_level, np.array([b]))[:, 0]
    blocks = np.r_[b, nb[nb >= 0]]
    kk = level - block_level
    ids = ((blocks[:, None] << (2 * kk)) + np.arange(4 ** kk)[None]).ravel()
    lon, lat = cell_centres_lonlat(level, ids)
    data = np.stack([np.sin(lat * 300), np.cos(lon * 250), np.sin(lat * 200 + lon * 150)], 1)
    return (0.2 + 0.1 * data).astype(np.float32), ids, b


def test_tangent_shift_moves_window_south_and_east():
    """shift=(dy, dx) must give the base grid moved dy rows south and dx columns east."""
    from healpix_analyse.dino import tangent_tiles
    level, parent_level = 17, 12
    data, ids, b = _smooth_field(level, 10)
    kw = dict(tile_px=(64, 64), fill="nan")
    base, cid = tangent_tiles(data, ids, level, parent_level, **kw)[:2]
    sel = np.isin(cid, (b << 4) + np.arange(16))
    base = base[sel]
    for dy, dx in [(8, 0), (0, 8), (4, 12)]:
        v = tangent_tiles(data, ids, level, parent_level, shift=(dy, dx), **kw)[0][sel]
        # v[r, c] == base[r + dy, c + dx]
        a = base[..., dy:, dx:]
        w = v[..., :64 - dy, :64 - dx]
        assert np.nanmax(np.abs(a - w)) < 1e-6


def test_tangent_over_sample_is_a_sliding_window():
    """With an exact backbone, a finer over_sample must get closer to the true
    16-px sliding-window mean at the position of each output cell."""
    from healpix_analyse.dino import (
        EARTH_RADIUS_M, cell_centres_lonlat, healpix_gsd_m, sample_healpix,
        tangent_grid_lonlat,
    )
    level, parent_level = 17, 12               # cells of 32 px in 64 px images
    data, ids, b = _smooth_field(level, 10)
    errs = []
    for n in (1, 2, 4):
        r = GetDINOV3SAT(data, ids, level, parent_level, projection="tangent",
                         tile_px=64, over_sample=n, model=_MeanDino(), device="cpu",
                         return_patches=True, mean=(0, 0, 0), std=(1, 1, 1),
                         input_range="unit", verbose=False)
        assert r.patch_level == level - 4 + int(np.log2(n))
        # cells of the central block: their images are entirely inside the data
        c = (r.patch_cell_id >> (2 * (r.patch_level - 10))) == b
        emb, cid = r.patch_embedding[c], r.patch_cell_id[c]
        lon, lat = cell_centres_lonlat(r.patch_level, cid)
        glon, glat = tangent_grid_lonlat(lon, lat, 16, healpix_gsd_m(level) / EARTH_RADIUS_M)
        v, ok = sample_healpix(data, ids, level, glon, glat)
        good = ok.reshape(lon.size, -1).all(1)
        truth = v.reshape(lon.size, -1, 3)[good].mean(1)
        errs.append(np.abs(emb[good] - truth).mean())
    assert errs[1] < 0.8 * errs[0] and errs[2] < 0.8 * errs[1], errs


# ---------------------------------------------------------------------------
# stride: sliding window over the images
# ---------------------------------------------------------------------------

def test_stride_centres_lattice():
    """stride=1/2: 4 centres per parent, the parent centre among them, the
    others at the midpoints / centres of four of the parent lattice."""
    from healpix_analyse.dino import cell_centres_lonlat, stride_centres, _lonlat_to_vec
    P = 12
    block = 1234
    parents = (block << 4) + np.arange(16)                       # a 4 x 4 block
    cid, lev, lon, lat, on = stride_centres(parents, P, 0.5)
    assert lev == P + 1 and cid.size == 64 and on.sum() == 16
    plon, plat = cell_centres_lonlat(P, parents)
    on_ids = cid[on] >> 2
    order = np.argsort(on_ids)
    assert np.array_equal(on_ids[order], np.sort(parents))
    vp = _lonlat_to_vec(plon, plat)[np.argsort(parents)]
    assert np.abs(_lonlat_to_vec(lon[on][order], lat[on][order]) - vp).max() < 1e-12
    # off-grid centres sit midway between two parent centres (along an axis,
    # or across a diagonal for the centre of four): check it on an 8 x 8 block,
    # away from its +x / +y edges where the neighbours are missing
    from healpix_analyse.dino import nested_to_xy
    big = (block << 6) + np.arange(64)
    cid8, _, lon8, lat8, on8 = stride_centres(big, P, 0.5)
    bx, by = nested_to_xy((cid8 >> 2) & 63)
    inner = ~on8 & (bx < 7) & (by < 7)
    blon, blat = cell_centres_lonlat(P, big)
    vb = _lonlat_to_vec(blon, blat)
    d = np.sort(np.linalg.norm(_lonlat_to_vec(lon8, lat8)[inner][:, None] - vb[None], axis=-1), axis=1)
    spacing = np.sort(np.linalg.norm(vb[:, None] - vb[None], axis=-1), axis=1)[:, 1].min()
    assert np.allclose(d[:, 0], d[:, 1], rtol=0.01)
    assert (d[:, 0] > 0.3 * spacing).all() and (d[:, 0] < 0.8 * spacing).all()
    assert stride_centres(parents, P, 1)[0].size == 16
    with pytest.raises(ValueError, match="power of two"):
        stride_centres(parents, P, 1 / 3)


def test_get_dinov3sat_stride_half():
    """stride=1/2 adds images between the stride=1 ones, which are unchanged."""
    level, parent_level = 17, 13               # cells of 16 px in 32 px images
    data, ids, b = _smooth_field(level, 11)
    sel = (ids >> (2 * (level - 11))) == b     # one block of 4 x 4 parents
    kw = dict(projection="tangent", tile_px=32, model=_MeanDino(), device="cpu",
              mean=(0, 0, 0), std=(1, 1, 1), pooling="cls")
    r1 = GetDINOV3SAT(data[sel], ids[sel], level, parent_level, **kw)
    r2 = GetDINOV3SAT(data[sel], ids[sel], level, parent_level, stride=0.5, **kw)
    assert r1.cell_id.size == 16 and r2.embedding.shape[0] == 64
    assert r2.cell_level == parent_level + 1 and r2.stride == 0.5
    assert r2.on_parent_grid.sum() == 16
    a = r1.embedding[np.argsort(r1.cell_id)]
    o = np.argsort(r2.cell_id[r2.on_parent_grid] >> 2)
    assert np.allclose(r2.embedding[r2.on_parent_grid][o], a, atol=1e-5)
    assert np.isfinite(r2.centre_lon).all() and r2.centre_lat.shape == (64,)
    with pytest.raises(ValueError, match="over_sample"):
        GetDINOV3SAT(data[sel], ids[sel], level, parent_level, stride=0.5, over_sample=2, **kw)


# ---------------------------------------------------------------------------
# input range: the network always gets [0, 1]
# ---------------------------------------------------------------------------

def test_scale_input_modes():
    from healpix_analyse.dino import scale_input, REFLECTANCE_WHITE
    x = np.array([[0.0, 0.1, 0.2], [0.3, np.nan, 0.6]], np.float32)
    out, (lo, hi), how = scale_input(x, "reflectance")
    assert how == "reflectance" and (lo, hi) == (0.0, REFLECTANCE_WHITE)
    assert np.isnan(out[1, 1]) and np.nanmax(out) == 1.0 and np.nanmin(out) == 0.0
    assert np.allclose(scale_input(x * 255 / 0.6, "uint8")[0][0], x[0] / 0.6, atol=1e-6)
    out, (lo, hi), how = scale_input(x, (0.1, 0.5))
    assert how == "fixed" and np.isclose(out[0, 2], 0.25)
    with pytest.raises(ValueError):
        scale_input(x, "kelvin")


def test_scale_input_sat493m_matches_training_statistics():
    from healpix_analyse.dino import scale_input, SAT493M_MEAN, SAT493M_STD
    rng = np.random.default_rng(0)
    x = np.stack([rng.normal(0.08, 0.02, 200000), rng.normal(0.07, 0.015, 200000),
                  rng.normal(0.05, 0.01, 200000)], 1) * 1e6           # any units
    out, (lo, hi), how = scale_input(x, "sat493m")
    assert how == "sat493m" and np.shape(lo) == (3,)
    assert np.allclose(out.mean(0), SAT493M_MEAN, atol=0.01)
    assert np.allclose(out.std(0), SAT493M_STD, atol=0.01)
    # the per-band range is reusable as is, e.g. for another date
    out2, _, how2 = scale_input(x, (lo, hi))
    assert how2 == "fixed" and np.allclose(out2, out)


def test_units_do_not_change_the_embeddings():
    """With input_range='auto' the same scene in reflectance, DN or x1e6 units
    gives the same embeddings: the network always sees the same image."""
    level, parent_level = 17, 13
    data, ids, b = _smooth_field(level, 11)
    sel = (ids >> (2 * (level - 11))) == b
    kw = dict(projection="tangent", tile_px=32, model=_MeanDino(), device="cpu", verbose=False)
    ref = GetDINOV3SAT(data[sel], ids[sel], level, parent_level, **kw)
    for k in (1e4, 1e6):
        r = GetDINOV3SAT(data[sel] * k + 3.0, ids[sel], level, parent_level, **kw)
        assert np.allclose(r.embedding, ref.embedding, atol=1e-4)
        assert r.input_scaling == "percentile"
    ref_s = GetDINOV3SAT(data[sel], ids[sel], level, parent_level, input_range="sat493m", **kw)
    r_s = GetDINOV3SAT(data[sel] * 1e6, ids[sel], level, parent_level, input_range="sat493m", **kw)
    assert np.allclose(r_s.embedding, ref_s.embedding, atol=1e-4) and r_s.input_scaling == "sat493m"
    # reusing the range of a first date gives the same mapping on a second one
    r2 = GetDINOV3SAT(data[sel], ids[sel], level, parent_level, input_range=ref.input_range, **kw)
    assert np.allclose(r2.embedding, ref.embedding, atol=1e-5) and r2.input_scaling == "fixed"
