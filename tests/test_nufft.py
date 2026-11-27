"""Tests of healpix_analyse.nufft (needs the optional finufft package)."""

import numpy as np
import pytest

pytest.importorskip("finufft")

from healpix_analyse.nufft import (  # noqa: E402
    _is_regular,
    _unique_float,
    get_bounds,
    nufft,
    period_area,
    period_median,
    windowing_weights_bbox,
    windowing_weights_polygon,
)


def regular_grid(nx, ny, dx=1.0, dy=1.0, x0=0.0, y0=0.0):
    """Coordinates of a regular nx x ny grid, flattened like a C-ordered (ny, nx) array."""
    yy, xx = np.meshgrid(y0 + dy * np.arange(ny), x0 + dx * np.arange(nx), indexing="ij")
    return xx.ravel(), yy.ravel()


# --------------------------------------------------------------------- periods


def test_unique_float():
    rng = np.random.default_rng(0)
    Nx, Ny = 12, 18
    dx, dy = 0.5, 0.8

    x = -5 + np.arange(Nx) * dx + rng.uniform(size=Nx) * dx / 3
    y = -3 + np.arange(Ny) * dy + rng.uniform(size=Ny) * dy / 3
    xx, yy = np.meshgrid(x, y)

    np.testing.assert_allclose(x, _unique_float(xx.ravel()))
    np.testing.assert_allclose(y, _unique_float(yy.ravel()))


def test_is_regular():
    rng = np.random.default_rng(0)
    Nx, Ny = 12, 18
    dx, dy = 0.5, 0.8

    x = -5 + np.arange(Nx) * dx + rng.uniform(size=Nx) * dx / 3
    y = -3 + np.arange(Ny) * dy + rng.uniform(size=Ny) * dy / 3
    xx, yy = np.meshgrid(x, y)

    assert _is_regular(xx.ravel(), yy.ravel())

    xx[5, 5] += 0.05
    assert not _is_regular(xx.ravel(), yy.ravel())


def test_period_median():
    # regular samples, repeated: n * dx
    assert period_median(np.repeat(0.25 * np.arange(5), 3)) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="single value"):
        period_median(np.full(4, 3.0))


def test_period_area():
    """Test equivalent to regular grid."""
    dx = 0.7
    Nx, Ny = 7, 12
    xx, yy = regular_grid(Nx, Ny, dx=dx, dy=dx)

    # Because we only have the left edge of pixels (or centers same result) we are
    # underestimating to total surface area by a border of one pixel.
    expected_dx = dx / np.sqrt(Nx * Ny / (Nx - 1) / (Ny - 1))
    expected_period_x = dx * (Nx - 1) + expected_dx
    expected_period_y = dx * (Ny - 1) + expected_dx

    bounds = get_bounds(xx, yy)
    period_x, period_y = period_area(xx.size, bounds)
    assert np.isclose(period_x, expected_period_x)
    assert np.isclose(period_y, expected_period_y)


# --------------------------------------------------------------------- nufft


@pytest.mark.parametrize("nx, ny", [(16, 8), (15, 9), (12, 12)])
def test_regular_grid_equals_fft2(nx, ny):
    """Regular grid, default periods: same coefficients and frequencies as numpy."""
    x, y = regular_grid(nx, ny, dx=0.5, dy=2.0, x0=-3.0, y0=10.0)
    data = np.random.default_rng(0).standard_normal(x.size)
    fx, fy, f = nufft(x, y, data, n_frequencies=(nx, ny), rtol=1e-12)

    ref = np.fft.fftshift(np.fft.fft2(data.reshape(ny, nx)))
    np.testing.assert_allclose(f, ref, rtol=0, atol=1e-9 * np.abs(ref).max())
    np.testing.assert_allclose(fx, np.fft.fftshift(np.fft.fftfreq(nx, d=0.5)))
    np.testing.assert_allclose(fy, np.fft.fftshift(np.fft.fftfreq(ny, d=2.0)))


def test_batched_equals_unbatched():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 10, 300), rng.uniform(-2, 5, 300)
    data = rng.standard_normal((3, 300))
    _, _, batched = nufft(x, y, data, n_frequencies=(16, 10), rtol=1e-12)
    assert batched.shape == (3, 10, 16)
    for k in range(3):
        _, _, single = nufft(x, y, data[k], n_frequencies=(16, 10), rtol=1e-12)
        assert single.shape == (10, 16)
        np.testing.assert_allclose(batched[k], single, rtol=0, atol=1e-12 * np.abs(single).max())


def test_cosine_peaks_at_its_frequency():
    """3 cycles over 16 samples: all the power at fx = +-3/16, fy = 0."""
    x, y = regular_grid(16, 8)
    fx, fy, f = nufft(x, y, np.cos(2 * np.pi * 3 / 16 * x), n_frequencies=(16, 8), rtol=1e-12)
    power = np.abs(f) ** 2
    iy, ix = np.nonzero(power > 1e-6 * power.max())
    np.testing.assert_allclose(sorted(fx[ix]), [-3 / 16, 3 / 16])
    np.testing.assert_array_equal(fy[iy], 0)


def test_sign_convention_matches_numpy():
    """exp(+2i pi f0 x) peaks at +f0, as with numpy.fft."""
    x, y = regular_grid(16, 8)
    fx, fy, f = nufft(x, y, np.exp(2j * np.pi * 3 / 16 * x), n_frequencies=(16, 8), rtol=1e-12)
    iy, ix = np.unravel_index(np.argmax(np.abs(f)), f.shape)
    assert fx[ix] == pytest.approx(3 / 16)
    assert fy[iy] == 0


def test_phase_reference_is_the_smallest_coordinate():
    """A single sample at (x0 + a, y0 + b) gives exp(-2i pi (fx a + fy b))."""
    x = np.array([2.0, 2.25, 5.0])
    y = np.array([-1.0, 0.5, 3.0])
    data = np.array([0.0, 1.0, 0.0])
    fx, fy, f = nufft(x, y, data, n_frequencies=(6, 5), period=(4.0, 5.0), rtol=1e-12)
    expected = np.exp(-2j * np.pi * (fx[None, :] * 0.25 + fy[:, None] * 1.5))
    np.testing.assert_allclose(f, expected, atol=1e-10)


def test_explicit_period_sets_the_frequency_step():
    x, y = regular_grid(8, 8)
    fx, fy, _ = nufft(x, y, np.ones(x.size), n_frequencies=(8, 6), period=(16.0, 10.0))
    np.testing.assert_allclose(fx, np.arange(-4, 4) / 16)
    np.testing.assert_allclose(fy, np.arange(-3, 3) / 10)


def test_period_not_larger_than_extent_raises():
    x, y = regular_grid(8, 8)  # extent 7
    with pytest.raises(ValueError, match="period"):
        nufft(x, y, np.ones(x.size), n_frequencies=(8, 8), period=(7.0, 8.0))


@pytest.mark.parametrize("case", ["constant", "nan", "inf"])
def test_degenerate_coordinates_raise(case):
    """No NaN coordinates reach finufft (they crash the interpreter)."""
    x = np.arange(8.0)
    y = np.zeros(8) if case == "constant" else np.arange(8.0)
    if case == "nan":
        y[3] = np.nan
    if case == "inf":
        x[0] = np.inf
    with pytest.raises(ValueError):
        nufft(x, y, np.ones(8), n_frequencies=(8, 4))


def test_bad_shapes_raise():
    x, y = regular_grid(4, 4)
    with pytest.raises(ValueError):
        nufft(x, y[:-1], np.ones(x.size), n_frequencies=(4, 4))
    with pytest.raises(ValueError):
        nufft(x, y, np.ones(x.size + 1), n_frequencies=(4, 4))
    with pytest.raises(ValueError):
        nufft(x, y, np.ones((2, 2, x.size)), n_frequencies=(4, 4))


# ------------------------------------------------------------------- windows

L_SHAPE = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]


@pytest.mark.parametrize("orientation", [1, -1])
def test_concave_polygon(orientation):
    px = np.array([0.5, 1.5, 0.5, 1.5, 0.05, 3.0])
    py = np.array([0.5, 0.5, 1.5, 1.5, 0.5, 0.5])
    w = windowing_weights_polygon(px, py, L_SHAPE[::orientation], length=0.1)
    # three interior points far from the border, the notch, half-way up the
    # taper (distance 0.05 = length / 2), outside
    np.testing.assert_allclose(w, [1, 1, 1, 0, 0.5, 0], atol=1e-12)


def test_polygon_taper_profile():
    """Along a line crossing the left edge of a square: 0 outside, cosine taper, then 1."""
    d = np.linspace(-0.5, 1.0, 31)
    w = windowing_weights_polygon(d, np.full_like(d, 2.0), [[0, 0], [4, 0], [4, 4], [0, 4]], length=0.5)
    expected = np.where(d <= 0, 0.0, np.where(d < 0.5, 0.5 * (1 - np.cos(np.pi * d / 0.5)), 1.0))
    np.testing.assert_allclose(w, expected, atol=1e-12)


def test_closed_ring_and_integer_coordinates():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([2, 2, 2, 2, 2])
    w = windowing_weights_polygon(x, y, [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], length=1)
    np.testing.assert_allclose(w, [0, 1, 1, 1, 0])
    assert w.dtype == np.float64


def test_bbox_keeps_shape():
    x, y = np.meshgrid(np.linspace(-1, 3, 9), np.linspace(-1, 2, 7))
    w = windowing_weights_bbox(x, y, [0, 0, 2, 1], length=0.2)
    assert w.shape == x.shape
    assert w[(x < 0) | (x > 2) | (y < 0) | (y > 1)].max() == 0
    assert windowing_weights_bbox([1.0], [0.5], [0, 0, 2, 1], length=0.2)[0] == 1
    assert windowing_weights_bbox([0.1], [0.5], [0, 0, 2, 1], length=0.2)[0] == pytest.approx(0.5)


def test_rectangle_equivalent_tukey():
    """Test that a rectangular window is equal to a product of tukey functions."""

    def tukey(x, L):
        x = x - x[0]
        w = np.ones(x.size, dtype=np.float64)
        taper = 0.5 * (1 - np.cos(x * np.pi / L))
        w = np.where(x < L, taper, w)
        w = np.where(x > x[-1] - L, taper[::-1], w)
        return w

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-3, 3, 150)
    wx = tukey(x, 1)
    wy = tukey(y, 1)
    w_product = wx[None, :] * wy[:, None]

    xx, yy = np.meshgrid(x, y)
    w = windowing_weights_bbox(xx, yy, [-5, -3, 5, 3], length=1)

    np.testing.assert_allclose(w, w_product)
