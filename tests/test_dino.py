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
        data, cell_id, level, parent_level,
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
