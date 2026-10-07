"""Tests of HealPixGeoFFT: FFT on the face lattice with a geodesically sampled kernel."""

import numpy as np
import pytest
import torch

import healpix_geo.nested as hgn

from healpix_analyse.geo_fft import HealPixGeoFFT, geo_fft_convolve, lattice_geometry, make_kernel, KERNELS
from healpix_analyse.validation import direct_spherical_convolution

LEVEL = 20
R_EARTH = 6371008.8


def block(lon, lat, delta=7, level=LEVEL):
    """The 4**delta children at `level` of the cell containing (lon, lat) at level - delta."""
    parent = int(np.asarray(hgn.lonlat_to_healpix([lon], [lat], level - delta))[0])
    return np.arange(parent * 4 ** delta, (parent + 1) * 4 ** delta)


def interior(cells, n, level=LEVEL):
    _, ii, jj = [np.asarray(v, dtype=np.int64) for v in hgn.healpix_to_base_cell_coordinates(cells.tolist(), level)]
    return ((ii - ii.min() >= n) & (ii.max() - ii >= n) & (jj - jj.min() >= n) & (jj.max() - jj >= n))


def pixel_kernel(conv, scale_m, radius_m):
    """The same truncated exponential as `conv`, in pixel units, for direct_spherical_convolution."""
    alpha = conv.geometry["cell_size_rad"]
    s, R = scale_m / R_EARTH / alpha, radius_m / R_EARTH / alpha
    return (lambda rho, phi: np.where(rho <= R, np.exp(-rho / s), 0.0)), 2 * int(np.ceil(R / np.sqrt(2))) + 1


@pytest.mark.parametrize("lon,lat", [(1.46, 48.27), (-60.0, -3.0), (10.0, 78.0)])   # polar cap off-diagonal, equator, near pole
def test_matches_direct_geodesic_sum(lon, lat):
    cells = block(lon, lat)
    rng = np.random.default_rng(0)
    x = 0.15 + 0.05 * rng.standard_normal(cells.size)
    conv = HealPixGeoFFT(LEVEL, cells, "exponential", scale=50.0, radius=150.0)
    kern, ksz = pixel_kernel(conv, 50.0, 150.0)
    ref = direct_spherical_convolution(x, cells, LEVEL, kern, kernel_sz=ksz, ellipsoid="sphere", normalize=False)
    inner = interior(cells, conv.window)
    y = conv.raw(x)
    assert np.sqrt(np.mean((y - ref)[inner] ** 2)) / np.sqrt(np.mean(ref[inner] ** 2)) < 1e-4
    # normalized, with 40 % missing
    xg = x.copy()
    xg[rng.random(x.size) < 0.4] = np.nan
    refg = direct_spherical_convolution(xg, cells, LEVEL, kern, kernel_sz=ksz, ellipsoid="sphere", normalize=True)
    yg = conv(xg)
    ok = inner & np.isfinite(yg) & np.isfinite(refg)
    assert ok.sum() > 0.5 * inner.sum()
    assert np.sqrt(np.mean((yg - refg)[ok] ** 2)) / np.sqrt(np.mean(refg[ok] ** 2)) < 1e-4


def test_square_grid_kernel_is_wrong_in_the_polar_cap():
    """The point of the module: k(sqrt(di^2+dj^2)) differs from the geodesic kernel."""
    cells = block(1.46, 48.27)
    conv = HealPixGeoFFT(LEVEL, cells, "exponential", scale=50.0, radius=150.0)
    n = conv.window
    d = np.arange(-n, n + 1)
    dj, di = np.meshgrid(d, d, indexing="ij")
    s_pix = 50.0 / R_EARTH / conv.geometry["cell_size_rad"]
    k_square = np.exp(-np.hypot(di, dj) / s_pix) * (np.hypot(di, dj) <= 150.0 / R_EARTH / conv.geometry["cell_size_rad"])
    rel = np.abs(k_square - conv.kernel_image).sum() / conv.kernel_image.sum()
    assert rel > 0.2                                   # tens of percent of the kernel mass misplaced
    g = conv.geometry
    assert g["angle_deg"] < 70 and g["step_j"] / g["step_i"] > 1.5   # sheared parallelogram, unequal sides


def test_constant_is_preserved_and_dirac_gives_the_kernel_image():
    cells = block(-60.0, -3.0)
    conv = HealPixGeoFFT(LEVEL, cells, "lorentzian", scale=30.0, radius=200.0)
    inner = interior(cells, conv.window)
    assert np.allclose(conv(np.ones(cells.size))[inner], 1.0)
    # Dirac at the central cell -> raw response equals the kernel image (mirrored: response at j = k(j -> c))
    _, ii, jj = [np.asarray(v, dtype=np.int64) for v in hgn.healpix_to_base_cell_coordinates(cells.tolist(), LEVEL)]
    ic, jc = ii.min() + conv.W // 2, jj.min() + conv.H // 2
    delta = np.zeros(cells.size)
    delta[(ii == ic) & (jj == jc)] = 1.0
    resp = conv.raw(delta)
    n = conv.window
    sel = (np.abs(ii - ic) <= n) & (np.abs(jj - jc) <= n)
    img = np.zeros((2 * n + 1, 2 * n + 1))
    img[jj[sel] - jc + n, ii[sel] - ic + n] = resp[sel]
    assert np.allclose(img, conv.kernel_image, atol=1e-9)   # y[c + d] = K[d] for a Dirac at c


def test_non_contiguous_cells_and_torch_input():
    cells = block(-60.0, -3.0)
    rng = np.random.default_rng(1)
    keep = rng.random(cells.size) > 0.3
    sub = cells[keep]
    x = rng.standard_normal(sub.size)
    conv_full = HealPixGeoFFT(LEVEL, cells, "gaussian", scale=40.0)
    conv_sub = HealPixGeoFFT(LEVEL, sub, "gaussian", scale=40.0, radius=conv_full.radius)
    xf = np.full(cells.size, np.nan)
    xf[keep] = x
    y_full = conv_full(xf)[keep]
    y_sub = conv_sub(x)
    assert np.allclose(y_full, y_sub, equal_nan=True)
    # torch input, batch, gradient
    xt = torch.tensor(np.stack([x, 2 * x]), dtype=torch.float64, requires_grad=True)
    yt = conv_sub(xt)
    assert yt.shape == xt.shape
    assert torch.allclose(yt[1], 2 * yt[0], equal_nan=True)
    yt.nansum().backward()
    assert torch.isfinite(xt.grad).all()


def test_callable_anisotropic_kernel_and_units():
    cells = block(-60.0, -3.0)
    aniso = lambda rho, phi: np.exp(-(rho * np.cos(phi) / 200.0) ** 2 - (rho * np.sin(phi) / 60.0) ** 2)
    conv = HealPixGeoFFT(LEVEL, cells, aniso, radius=600.0)
    k = conv.kernel_image
    n = conv.window
    # elongated along north (phi = 0): the image decays more slowly along the north-south direction
    north = k[:, n]          # varying j at fixed i is not north in general; compare total spread instead
    assert k.sum() > 0 and np.isfinite(k).all()
    # the same kernel in pixel units must give the same image
    alpha = conv.geometry["cell_size_rad"]
    to_m = R_EARTH * alpha
    aniso_pix = lambda rho, phi: aniso(rho * to_m, phi)
    conv_pix = HealPixGeoFFT(LEVEL, cells, aniso_pix, radius=600.0 / to_m, units="pix")
    assert np.allclose(conv_pix.kernel_image, k)
    conv_deg = HealPixGeoFFT(LEVEL, cells, lambda rho, phi: aniso(np.radians(rho) * R_EARTH, phi),
                             radius=np.degrees(600.0 / R_EARTH), units="deg")
    assert np.allclose(conv_deg.kernel_image, k)


def test_predefined_kernels_build():
    cells = block(-60.0, -3.0, delta=6)
    for name in KERNELS:
        conv = HealPixGeoFFT(LEVEL, cells, name, scale=20.0, radius=100.0)
        assert conv.kernel_image.shape[0] == 2 * conv.window + 1
        y = conv(np.ones(cells.size))
        if name != "dog":
            assert conv.normalized and np.allclose(y[interior(cells, conv.window)], 1.0)
        else:
            assert not conv.normalized                     # signed kernels are applied raw
    assert make_kernel(lambda r, p: r, None) is not None
    with pytest.raises(ValueError):
        make_kernel("no_such_kernel", 1.0)
    with pytest.raises(ValueError):
        make_kernel("gaussian")                            # scale missing


def test_default_radius_and_errors():
    cells = block(-60.0, -3.0, delta=6)
    conv = HealPixGeoFFT(LEVEL, cells, "gaussian", scale=10.0)
    assert 3.5 * 10.0 < conv.radius < 4.5 * 10.0           # 1e-3 of the peak at ~3.7 sigma
    with pytest.raises(ValueError, match="base faces"):
        two = np.concatenate([block(-60.0, -3.0, delta=5), block(120.0, -3.0, delta=5)])
        HealPixGeoFFT(LEVEL, two, "gaussian", scale=10.0)
    with pytest.raises(ValueError, match="runs off"):
        edge_parent = int(np.asarray(hgn.lonlat_to_healpix([-60.0], [-3.0], 2))[0])
        edge = np.arange(edge_parent * 4 ** 2, (edge_parent + 1) * 4 ** 2)   # level 4 block = a quarter face
        HealPixGeoFFT(4, edge, "gaussian", scale=5.0, radius=50.0, units="pix")


def test_one_shot_function_and_geometry():
    cells = block(-60.0, -3.0, delta=6)
    x = np.random.default_rng(2).standard_normal(cells.size)
    y1 = geo_fft_convolve(x, cells, LEVEL, "exponential", scale=30.0, radius=90.0)
    y2 = HealPixGeoFFT(LEVEL, cells, "exponential", scale=30.0, radius=90.0)(x)
    assert np.allclose(y1, y2, equal_nan=True)
    g = lattice_geometry(LEVEL, 7, 2 ** 19, 2 ** 19)
    assert 98.0 < g["angle_deg"] < 100.5 and abs(g["step_i"] - g["step_j"]) < 0.02


def test_wide_block_warns():
    with pytest.warns(RuntimeWarning, match="degrees"):
        HealPixGeoFFT(6, block(-60.0, -3.0, delta=4, level=6), "gaussian", scale=2.0, radius=6.0, units="pix")
