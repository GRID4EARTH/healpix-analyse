"""DINO on HealPixPatches (projection="gauge"), and the HealPixPatches options it needs."""

import numpy as np
import pytest
import torch
import torch.nn as nn

import healpix_geo
from healpix_analyse._ellipsoid import authalic_scale
from healpix_analyse.dino import DINOV3_PATCH, GetDINOV3SAT
from healpix_analyse.patches import HealPixPatches

L = 10                      # data level
PL = 5                      # image centres: cells of 32 px
TILE = 64                   # 4 x 4 tokens per image
BLOCK_LEVEL = 3             # domain: one level-3 cell (128 x 128 cells of level 10)
CPU = dict(device="cpu")


class TokenMean(nn.Module):
    """Fake backbone: a token is the mean of its 16 x 16 px block, CLS the mean
    of the image.  Rotation-equivariant, and it keeps the images it saw."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward_features(self, x):
        self.seen.append(x.detach().clone())
        B, C, H, W = x.shape
        t = nn.functional.avg_pool2d(x, DINOV3_PATCH)            # [B, C, T, T]
        return {"x_norm_clstoken": x.mean(dim=(2, 3)),
                "x_norm_patchtokens": t.flatten(2).transpose(1, 2)}


def _domain(block=None):
    if block is None:
        block = int(healpix_geo.nested.lonlat_to_healpix(
            np.array([2.0]), np.array([46.0]), BLOCK_LEVEL, ellipsoid="sphere")[0])
    q = 4 ** (L - BLOCK_LEVEL)
    return np.arange(block * q, (block + 1) * q, dtype=np.int64)


def _run(data, ids, **kw):
    kw.setdefault("model", TokenMean())
    return GetDINOV3SAT(data, ids, L, PL, tile_px=TILE, input_range="unit",
                        mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), verbose=False,
                        device="cpu", **kw)


def test_tokens_land_on_their_own_cell():
    """A field constant on every cell of level L-4 comes back on that cell, not
    on a neighbour: no half-cell shift between the tokens and the cells."""
    ids = _domain()
    rng = np.random.default_rng(0)
    parent = ids >> 8                                             # level L - 4
    up, inv = np.unique(parent, return_inverse=True)
    f = rng.random(up.size)
    data = np.repeat(f[inv][:, None], 3, axis=1).astype(np.float32)
    res = _run(data, ids)
    assert res.cell_level == L - 4
    assert np.array_equal(res.cell_id, up)                         # every cell, once
    got = res.embedding[:, 0]
    truth = f[np.searchsorted(up, res.cell_id)]
    # interior cells (their images see only data): compare with every neighbour
    nb = healpix_geo.nested.neighbours(res.cell_id.astype(np.uint64), L - 4,
                                       connectivity="all")
    nb = np.asarray(getattr(nb, "data", nb)).astype(np.int64)
    inside = np.isin(nb, up).all(axis=1)
    c_self = np.corrcoef(got[inside], truth[inside])[0, 1]
    for j in range(nb.shape[1]):
        shifted = f[np.searchsorted(up, nb[inside, j])]
        assert c_self > np.corrcoef(got[inside], shifted)[0, 1] + 0.2
    assert c_self > 0.6


def test_constant_field_is_kept_and_coverage_is_one():
    ids = _domain()
    data = np.full((ids.size, 3), 0.25, np.float32)
    res = _run(data, ids)
    np.testing.assert_allclose(res.embedding, 0.25, atol=1e-6)
    np.testing.assert_allclose(res.image_embedding, 0.25, atol=1e-6)
    assert res.embedding.shape == (ids.size // 256, 3)


def test_images_are_north_up_east_right():
    ids = _domain()
    lon, lat = healpix_geo.nested.healpix_to_lonlat(ids.astype(np.uint64), L,
                                                    ellipsoid="sphere")
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    norm = lambda v: ((v - v.min()) / (v.max() - v.min())).astype(np.float32)
    data = np.stack([norm(lat), norm(lon), np.zeros_like(lat, np.float32)], 1)
    model = TokenMean()
    _run(data, ids, model=model, min_coverage=0.99)
    img = torch.cat(model.seen)                                   # [M, 3, N, N]
    rows = img[:, 0].mean(dim=2)                                  # latitude per row
    cols = img[:, 1].mean(dim=1)                                  # longitude per column
    assert bool((rows[:, 0] > rows[:, -1]).all())                 # north at the top
    assert bool((cols[:, -1] > cols[:, 0]).all())                 # east on the right


def test_four_rotations_match_one_for_an_equivariant_backbone():
    ids = _domain()
    data = np.random.default_rng(1).random((ids.size, 3)).astype(np.float32)
    r1 = _run(data, ids)
    r4 = _run(data, ids, rotations=4)
    np.testing.assert_allclose(r4.embedding, r1.embedding, atol=1e-5)
    np.testing.assert_allclose(r4.image_embedding, r1.image_embedding, atol=1e-5)


def test_missing_cells_and_regional_output():
    ids = _domain()
    data = np.random.default_rng(2).random((ids.size, 3)).astype(np.float32)
    data[: ids.size // 2] = np.nan                                # half the block missing
    res = _run(data, ids)
    present = np.unique(ids[ids.size // 2:] >> 8)
    assert set(res.cell_id) == set(present)
    assert np.isfinite(res.embedding).all()
    assert (res.coverage > 0).all() and (res.coverage <= 1 + 1e-6).all()


def test_centre_batches_do_not_change_the_result():
    ids = _domain()
    data = np.random.default_rng(3).random((ids.size, 3)).astype(np.float32)
    a = _run(data, ids)
    b = _run(data, ids, centre_batch=3)
    np.testing.assert_allclose(b.embedding, a.embedding, atol=1e-5)
    np.testing.assert_array_equal(b.image_cell_id, a.image_cell_id)


def test_gauge_errors():
    ids = _domain()
    data = np.random.default_rng(4).random((ids.size, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="power of two"):
        _run(data, ids, tile_px=48) if False else GetDINOV3SAT(
            data, ids, L, PL, tile_px=48, model=TokenMean(), verbose=False, device="cpu")
    with pytest.raises(ValueError, match="over_sample and stride"):
        _run(data, ids, over_sample=2)
    with pytest.raises(ValueError, match="smaller than the cell"):
        GetDINOV3SAT(data, ids, L, PL, tile_px=16, model=TokenMean(), verbose=False,
                     device="cpu")
    with pytest.raises(ValueError, match="rotations"):
        _run(data, ids, rotations=2)


# ---------------------------------------------------------------------------
# HealPixPatches: centres at any level, centre_ids, ellipsoid
# ---------------------------------------------------------------------------

def test_centres_level_matches_coarse_and_all():
    lev, N = 4, 4
    x = torch.randn(1, 2, 12 * 4**lev, dtype=torch.float64)
    F64 = dict(device="cpu", dtype=torch.float64)
    coarse = HealPixPatches(lev, N, **F64)
    as_level = HealPixPatches(lev, N, centres=lev - 2, **F64)
    torch.testing.assert_close(as_level(x), coarse(x))
    allc = HealPixPatches(lev, N, centres="all", **F64)
    torch.testing.assert_close(HealPixPatches(lev, N, centres=lev, **F64)(x), allc(x))
    mid = HealPixPatches(lev, N, centres=lev - 1, **F64)
    assert mid.M == 12 * 4 ** (lev - 1) and mid.centre_level == lev - 1


def test_centre_ids_subset_equals_slice():
    lev, N = 4, 4
    x = torch.randn(1, 1, 12 * 4**lev, dtype=torch.float64)
    F64 = dict(device="cpu", dtype=torch.float64)
    full = HealPixPatches(lev, N, centres=lev - 1, **F64)
    sub_ids = np.array([5, 17, 100, 191])
    sub = HealPixPatches(lev, N, centres=lev - 1, centre_ids=sub_ids, **F64)
    torch.testing.assert_close(sub(x), full(x)[:, sub_ids])
    with pytest.raises(ValueError, match="must lie in"):
        HealPixPatches(lev, N, centres=lev - 3, **F64)
    with pytest.raises(ValueError, match="tiles the domain"):
        HealPixPatches(lev, N, centres=lev - 1, mode="nested", **F64)


def test_authalic_scale():
    assert np.allclose(authalic_scale([0.0, 45.0, 89.0], "sphere"), 1.0)
    k = authalic_scale(np.array([0.0, 30.0, 60.0, 85.0]), "wgs84")
    # equator: N(0) / R_a = a / R_a
    assert abs(k[0] - 6378137.0 / 6371007.1809) < 1e-6
    assert np.all(np.abs(k - 1) < 3.4e-3)
    # against healpix-geo's own authalic latitude: k = N cos(phi) / (R_a cos(beta))
    ids = healpix_geo.nested.lonlat_to_healpix(np.array([3.0, 3.0]), np.array([30.0, 60.0]),
                                               20, ellipsoid="WGS84")
    _, phi = healpix_geo.nested.healpix_to_lonlat(ids, 20, ellipsoid="WGS84")
    _, beta = healpix_geo.nested.healpix_to_lonlat(ids, 20, ellipsoid="sphere")
    phi, beta = np.radians(np.asarray(phi)), np.radians(np.asarray(beta))
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    n_rad = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    ref = n_rad * np.cos(phi) / (6371007.1809 * np.cos(beta))
    np.testing.assert_allclose(authalic_scale(np.degrees(phi), "WGS84"), ref, rtol=1e-6)


def test_wgs84_patches_same_centres_small_change():
    lev, N = 8, 8
    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.random((1, 1, 12 * 4**lev)))
    F64 = dict(device="cpu", dtype=torch.float64)
    sph = HealPixPatches(lev, N, centres=lev - 2, **F64)
    wgs = HealPixPatches(lev, N, centres=lev - 2, ellipsoid="wgs84", **F64)
    assert wgs.ellipsoid == "WGS84"
    assert torch.equal(sph.centre_ids, wgs.centre_ids)
    lon_s, lat_s = sph.centre_lonlat()
    lon_w, lat_w = wgs.centre_lonlat()
    np.testing.assert_allclose(lon_s, lon_w)
    assert np.abs(lat_w - lat_s).max() > 0.1                      # geodetic vs authalic
    ys, yw = sph(x), wgs(x)
    assert not torch.equal(ys, yw)
    # a 0.3 % stretch of an 8-px patch moves the outer samples by < 0.02 cell
    assert float((ys - yw).abs().max()) < 0.05
    # the 1-px patch is the cell itself whatever the ellipsoid
    one = HealPixPatches(lev, 1, centres="all", ellipsoid="WGS84", **F64)
    torch.testing.assert_close(one(x)[0, :, 0, 0, 0, 0], x[0, 0])
