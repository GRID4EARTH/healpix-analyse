"""HealPixConv on an ellipsoid: the stencil sits on its cell.

A HEALPix grid on an ellipsoid is the grid of its authalic sphere.  The
stencil used to be centred at the *geodetic* latitude of each cell and then
interpolated as if it were the authalic one, which moved it by
``phi - beta`` (0.128 degree, about 14 km, at 48 N): at level 12 the centre
tap read a cell about 9 cells away.  The stencil is now built on the
authalic sphere and the ellipsoid only sets its metric.
"""

from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_analyse._ellipsoid import authalic_scale
from healpix_analyse.convol import (
    HealPixConv,
    _build_rotation_matrices,
    _local_kernel_grid,
    _stretch_to_ellipsoid,
)


def _block(level, lon, lat, side):
    """A side x side NESTED block of cells at ``level`` around (lon, lat)."""
    p = int(np.log2(side))
    c = int(np.asarray(healpix_geo.nested.lonlat_to_healpix(
        [float(lon)], [float(lat)], level - p, ellipsoid="sphere"))[0])
    return c * 4**p + np.arange(4**p, dtype=np.int64)


def _centre_tap(conv):
    """For every output cell: input position read with the largest weight
    by the centre tap, and that weight."""
    c = conv.P // 2
    idx = conv._tap_idx[c].long()            # [4, G, K]
    w = conv._tap_w[c].double()
    j = w.argmax(0, keepdim=True)
    return idx.gather(0, j)[0], w.gather(0, j)[0]


@pytest.mark.parametrize("ellipsoid", ["WGS84", "sphere"])
@pytest.mark.parametrize("gauge_type,G", [("phi", 1), ("projected_ref", 2)])
def test_centre_tap_reads_own_cell_regional(ellipsoid, gauge_type, G):
    ids = _block(12, 2.0, 48.0, 64)
    rng = np.random.default_rng(0)
    ids = ids[rng.permutation(ids.size)]     # any input order
    conv = HealPixConv(12, 1, 1, kernel_sz=3, n_gauges=G, gauge_type=gauge_type,
                       cell_ids=ids, ellipsoid=ellipsoid, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    pos, w = _centre_tap(conv)
    own = torch.arange(ids.size).expand_as(pos)
    assert torch.equal(pos, own)
    assert float(w.min()) > 1 - 1e-6


@pytest.mark.parametrize("level", [3, 5])
def test_centre_tap_reads_own_cell_full_sphere(level):
    conv = HealPixConv(level, 1, 1, kernel_sz=5, ellipsoid="WGS84", device="cpu",
                       dtype=torch.float64, cache_dir=None)
    pos, w = _centre_tap(conv)
    assert torch.equal(pos, torch.arange(12 * 4**level).expand_as(pos))
    assert float(w.min()) > 1 - 1e-6


def test_wgs84_delta_kernel_is_identity_and_follows_input():
    """The case of the report: 64 x 64 block at level 12, kernel 3."""
    ids = _block(12, 2.0, 48.0, 64)
    x = torch.randn(1, 1, ids.size, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(0))
    W = torch.zeros(1, 1, 1, 9, dtype=torch.float64)
    W[..., 4] = 1.0
    outs = {}
    for ell in ("sphere", "WGS84"):
        conv = HealPixConv(12, 1, 1, kernel_sz=3, cell_ids=ids, ellipsoid=ell,
                           device="cpu", dtype=torch.float64, cache_dir=None)
        conv.set_kernel(W)
        with torch.no_grad():
            outs[ell] = conv(x)
        torch.testing.assert_close(outs[ell], x, rtol=0, atol=1e-9)
    # a random kernel: the WGS84 output stays close to the sphere output
    # (same cells, metric changed by < 0.34 %)
    Wr = torch.randn(1, 1, 1, 9, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(1))
    ys = []
    for ell in ("sphere", "WGS84"):
        conv = HealPixConv(12, 1, 1, kernel_sz=3, cell_ids=ids, ellipsoid=ell,
                           device="cpu", dtype=torch.float64, cache_dir=None)
        conv.set_kernel(Wr)
        with torch.no_grad():
            ys.append(conv(x).ravel())
    corr = float(torch.corrcoef(torch.stack(ys))[0, 1])
    assert corr > 0.99


def test_authalic_scale():
    lat = np.array([-90.0, -48.0, 0.0, 48.0, 90.0])
    np.testing.assert_array_equal(authalic_scale(lat, "sphere"), 1.0)
    k = authalic_scale(lat, "WGS84")
    assert abs(k[2] - 1.0011) < 1e-4                     # equator
    np.testing.assert_allclose(k, k[::-1])               # symmetric
    assert np.all((k >= 1 - 1e-8) & (k < 1.0012))
    # continuous up to the poles
    near = authalic_scale(np.array([89.9, 89.999, 90.0, -90.0]), "WGS84")
    np.testing.assert_allclose(near, near[0], rtol=1e-5)


def test_stretch_scales_meridian_by_k_and_parallel_by_inverse():
    nside, k = 2**10, 1.002
    lon, lat = 2.0, 47.87
    th = np.deg2rad([90.0 - lat])
    ph = np.deg2rad([lon])
    R = _build_rotation_matrices(th, ph, 1, "phi", "cpu", torch.float64)
    vec = torch.as_tensor(_local_kernel_grid(3, nside), dtype=torch.float64)
    rot = torch.einsum("kgij,pj->kgpi", R, vec).reshape(-1, 3)
    before = rot.clone()
    out = _stretch_to_ellipsoid(rot, th, ph, np.array([k]), 1, 9)
    n = before[4]
    torch.testing.assert_close(out[4], n)                # the centre stays
    st, ct, sp, cp = np.sin(th[0]), np.cos(th[0]), np.sin(ph[0]), np.cos(ph[0])
    e_s = torch.tensor([ct * cp, ct * sp, -st], dtype=torch.float64)
    for p in range(9):
        if p == 4:
            continue
        d0 = torch.acos((before[p] @ n).clamp(-1, 1))
        d1 = torch.acos((out[p] @ n).clamp(-1, 1))
        s0 = float((before[p] @ e_s) / torch.sin(d0))     # cos of the azimuth
        # squared distance: k^2 along the meridian, 1/k^2 along the parallel
        expect = d0**2 * (k**2 * s0**2 + (1 - s0**2) / k**2)
        torch.testing.assert_close(d1**2, expect, rtol=1e-9, atol=0)
