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
    # Dirac at the central cell c: y[p] = sum_d K[d] x[p + d], so y[c - d] = K[d]
    _, ii, jj = [np.asarray(v, dtype=np.int64) for v in hgn.healpix_to_base_cell_coordinates(cells.tolist(), LEVEL)]
    ic, jc = ii.min() + conv.W // 2, jj.min() + conv.H // 2
    delta = np.zeros(cells.size)
    delta[(ii == ic) & (jj == jc)] = 1.0
    resp = conv.raw(delta)
    n = conv.window
    sel = (np.abs(ii - ic) <= n) & (np.abs(jj - jc) <= n)
    img = np.zeros((2 * n + 1, 2 * n + 1))
    img[jc - jj[sel] + n, ic - ii[sel] + n] = resp[sel]
    assert np.allclose(img, conv.kernel_image, atol=1e-9)


def _brute_force(x, cells, kernel, radius_m, targets, level=LEVEL):
    """sum_q k(rho(p -> q), phi(p -> q)) x_q, with rho in metres, for the cells `targets`."""
    from healpix_analyse.geo_fft import _bearing_rad, _geodesic_rad
    lon, lat = (np.asarray(v, float) for v in hgn.healpix_to_lonlat(cells.tolist(), level, ellipsoid="sphere"))
    out = []
    for p in targets:
        rho = _geodesic_rad(lon[p], lat[p], lon, lat) * R_EARTH
        phi = _bearing_rad(lon[p], lat[p], lon, lat)
        w = np.where(rho <= radius_m, kernel(rho, phi), 0.0)
        out.append(np.sum(w * x))
    return np.array(out)


@pytest.mark.parametrize("lon,lat", [(-60.0, -3.0), (1.46, 48.27)])
def test_one_sided_kernel_orientation(lon, lat):
    """A kernel that only looks north of the output cell must take its input from the north."""
    cells = block(lon, lat, delta=6)
    north = lambda rho, phi: (0.5 + 0.5 * np.cos(phi)) ** 4 * np.exp(-rho / 20.0)
    conv = HealPixGeoFFT(LEVEL, cells, north, radius=80.0, normalized=False)
    x = np.random.default_rng(3).standard_normal(cells.size)
    inner = np.flatnonzero(interior(cells, conv.window))
    targets = inner[:: max(1, inner.size // 40)]
    ref = _brute_force(x, cells, north, 80.0, targets)
    y = conv(x)[targets]
    # residual: drift of the cell geometry at the truncation rim (~1e-3); a mirrored kernel is off by ~1
    assert np.sqrt(np.mean((y - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)) < 1e-2
    mirrored = _brute_force(x, cells, lambda rho, phi: north(rho, phi + np.pi), 80.0, targets)
    assert np.sqrt(np.mean((y - mirrored) ** 2)) / np.sqrt(np.mean(ref ** 2)) > 0.3
    # a Dirac at q is seen by the output cells to its south
    lon_c, lat_c = (np.asarray(v, float) for v in hgn.healpix_to_lonlat(cells.tolist(), LEVEL, ellipsoid="sphere"))
    q = inner[inner.size // 2]
    delta = np.zeros(cells.size)
    delta[q] = 1.0
    resp = conv.raw(delta)
    hit = resp > 0.1 * resp.max()
    assert np.mean(lat_c[hit] - lat_c[q]) < 0


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


def test_default_radius_covers_every_azimuth():
    cells = block(-60.0, -3.0, delta=7)
    narrow_north = lambda rho, phi: np.exp(-0.5 * (rho * np.cos(phi) / 10.0) ** 2 - 0.5 * (rho * np.sin(phi) / 40.0) ** 2)
    conv = HealPixGeoFFT(LEVEL, cells, narrow_north)
    assert 3.5 * 40.0 < conv.radius < 4.5 * 40.0           # set by the long (east-west) axis


def test_window_contains_the_geodesic_disc():
    """The window is widened when the lattice geometry drifts across it (large window, polar cap)."""
    level = 12
    cells = block(45.0, 75.0, delta=8, level=level)
    conv = HealPixGeoFFT(level, cells, "top_hat", scale=150.0, units="km", radius=150.0)
    n = conv.window
    ic, jc = conv.i0 + conv.W // 2, conv.j0 + conv.H // 2
    big = block(45.0, 75.0, delta=10, level=level)                 # wide enough to hold the whole disc
    _, ii, jj = [np.asarray(v, dtype=np.int64) for v in hgn.healpix_to_base_cell_coordinates(big.tolist(), level)]
    lon, lat = (np.asarray(v, float) for v in hgn.healpix_to_lonlat(big.tolist(), level, ellipsoid="sphere"))
    assert ii.min() < ic - n - 8 and ii.max() > ic + n + 8 and jj.min() < jc - n - 8 and jj.max() > jc + n + 8
    c = np.flatnonzero((ii == ic) & (jj == jc))[0]
    from healpix_analyse.geo_fft import _geodesic_rad
    in_disc = _geodesic_rad(lon[c], lat[c], lon, lat) * R_EARTH / 1000.0 <= 150.0
    needed = max(np.abs(ii[in_disc] - ic).max(), np.abs(jj[in_disc] - jc).max())
    assert needed <= n
    g = conv.geometry                                  # the bound from the centre geometry alone is one short here
    centre_bound = int(np.ceil(150e3 / R_EARTH / g["cell_size_rad"]
                               / (min(g["step_i"], g["step_j"]) * np.sin(np.radians(g["angle_deg"])))))
    assert centre_bound < needed


def test_weights_and_non_finite_input():
    cells = block(-60.0, -3.0, delta=5)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, cells.size))
    w = rng.random(cells.size)
    conv = HealPixGeoFFT(LEVEL, cells, "gaussian", scale=20.0, radius=80.0)
    y = conv(x, weights=w)
    assert np.allclose(y, conv.raw(x * w) / conv.raw(np.broadcast_to(w, x.shape).copy()), equal_nan=True)
    assert np.allclose(conv(x, weights=w[None, None, :]), y, equal_nan=True)
    with pytest.raises(ValueError, match="broadcast"):
        conv(x[0, 0], weights=np.stack([w, w]))
    with pytest.raises(ValueError, match="normalized"):
        HealPixGeoFFT(LEVEL, cells, "gaussian", scale=20.0, radius=80.0, normalized=False)(x, weights=w)
    # inf counts as missing / zero in every mode
    xi = x[0, 0].copy()
    xi[10] = np.inf
    xz = xi.copy()
    xz[10] = 0.0
    assert np.isfinite(conv.raw(xi)).all()
    assert np.allclose(conv.raw(xi), conv.raw(xz))


def test_module_to_moves_the_operator():
    cells = block(-60.0, -3.0, delta=5)
    conv = HealPixGeoFFT(LEVEL, cells, "gaussian", scale=20.0, radius=80.0)
    assert {b.device for b in (conv._kfft, conv._rows, conv._cols)} == {conv.device_}
    if torch.cuda.is_available():
        conv = conv.to("cuda")
        y = conv(torch.ones(cells.size, dtype=torch.float64, device="cuda"))
        assert y.device.type == "cuda"


def test_wide_block_warns():
    with pytest.warns(RuntimeWarning, match="degrees"):
        HealPixGeoFFT(6, block(-60.0, -3.0, delta=4, level=6), "gaussian", scale=2.0, radius=6.0, units="pix")
