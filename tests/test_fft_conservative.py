"""Conservative projection pair of LocalFFT / HealPixFFTConv (issue 77)."""

from __future__ import annotations

import math
import warnings

import healpix_geo
import numpy as np
import pytest
import torch
from scipy import ndimage

from healpix_analyse import HealPixFFTConv
from healpix_analyse.fft_local import (
    PROJECTION_BLUR_FLOOR_PX,
    LocalFFT,
    check_kernel_width,
)

LEVEL = 8


def _domain():
    """About 2500 cells of a round patch at mid latitude, level 8."""
    all_ids = np.arange(12 * 4**LEVEL, dtype=np.int64)
    lon, lat = healpix_geo.nested.healpix_to_lonlat(all_ids, LEVEL)
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    near = np.hypot((lon + 4.0) * math.cos(math.radians(47.0)), np.asarray(lat) - 47.0)
    return all_ids[near < 6.0]


@pytest.fixture(scope="module")
def ids():
    return _domain()


def _layer(ids, sigma, projection, kernel_sz=None):
    kernel_sz = kernel_sz or 2 * math.ceil(5 * sigma) + 1
    layer = HealPixFFTConv(
        LEVEL, 1, 1, kernel_sz, cell_ids=ids, dtype=torch.float64,
        device="cpu", projection=projection,
    ).eval()
    layer.set_gaussian(sigma, on_below="ignore")
    return layer


def _interior(layer, sigma):
    tr = layer.transform
    px = tr.pixel_size_rad
    centre = 0.5 * (tr.grid_size - 1)
    gx = np.rint(tr.projected_x.numpy() / px + centre).astype(int)
    gy = np.rint(tr.projected_y.numpy() / px + centre).astype(int)
    edt = ndimage.distance_transform_edt(tr.coverage_mask.numpy())[gy, gx]
    return edt >= math.ceil(5 * sigma) + 2


def _responses(layer, cells):
    n = layer.n_cells
    eye = torch.zeros(len(cells), n, dtype=torch.float64)
    eye[torch.arange(len(cells)), torch.as_tensor(cells)] = 1.0
    with torch.no_grad():
        return layer(eye).reshape(len(cells), n).numpy()


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_conservative_column_sums_are_one_inside(ids, sigma):
    layer = _layer(ids, sigma, "conservative")
    inside = np.flatnonzero(_interior(layer, sigma))
    assert inside.size > 100
    sums = _responses(layer, inside).sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_average_column_sums_are_not_one(ids):
    """The documented behaviour of the default convention."""
    layer = _layer(ids, 1.0, "average")
    inside = np.flatnonzero(_interior(layer, 1.0))
    sums = _responses(layer, inside).sum(axis=1)
    assert sums.max() - sums.min() > 0.05


def test_conservative_edge_mass_is_lost_not_gained(ids):
    layer = _layer(ids, 2.0, "conservative")
    tr = layer.transform
    px = tr.pixel_size_rad
    centre = 0.5 * (tr.grid_size - 1)
    gx = np.rint(tr.projected_x.numpy() / px + centre).astype(int)
    gy = np.rint(tr.projected_y.numpy() / px + centre).astype(int)
    edt = ndimage.distance_transform_edt(tr.coverage_mask.numpy())[gy, gx]
    edge = np.flatnonzero(edt <= 2)
    sums = _responses(layer, edge).sum(axis=1)
    assert sums.max() <= 1.0 + 1e-9
    assert sums.min() < 0.95  # something does leave through the edge


def test_total_of_a_field_is_kept_inside(ids):
    layer = _layer(ids, 1.0, "conservative")
    inside = np.flatnonzero(_interior(layer, 1.0))
    rng = np.random.default_rng(1)
    x = np.zeros(layer.n_cells)
    x[inside] = rng.random(inside.size)
    y = layer(torch.as_tensor(x)).detach().numpy().ravel()
    # sources inside, kernel reaches the edge only through interior cells,
    # so the whole mass is found again in the domain
    assert y.sum() == pytest.approx(x.sum(), rel=1e-3)


def test_default_is_bit_identical_to_the_previous_behaviour(ids):
    """projection='average' reproduces project/unproject of the old code."""
    tr = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    assert tr.projection == "average"
    rng = np.random.default_rng(2)
    x = torch.as_tensor(rng.random((3, ids.size)))
    projected = tr._project_tensor(x)
    # reference: the pre-change formulas written out
    sources = (x.unsqueeze(-1) * tr._grid_weights).reshape(3, -1)
    ref = torch.zeros(3, tr.grid_size**2, dtype=x.dtype)
    ref.scatter_add_(1, tr._grid_indices.reshape(1, -1).expand(3, -1), sources)
    ref = (ref * tr._grid_inverse_density).reshape(3, tr.grid_size, tr.grid_size)
    assert torch.equal(projected, ref)
    back = tr._unproject_tensor(projected)
    samples = torch.gather(
        projected.reshape(3, -1), 1, tr._grid_indices.reshape(1, -1).expand(3, -1)
    ).reshape(3, ids.size, 4)
    assert torch.equal(back, (samples * tr._grid_weights).sum(-1))
    # and the default layer equals an explicit "average" layer
    a = HealPixFFTConv(LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu")
    b = HealPixFFTConv(
        LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu",
        projection="average",
    )
    b.load_state_dict(a.state_dict())
    assert torch.equal(a(x[0]), b(x[0]))


def test_constant_kept_by_average_and_by_the_conservative_round_trip(ids):
    ones = torch.ones(ids.size, dtype=torch.float64)
    avg = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    con = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu", projection="conservative")
    for tr in (avg, con):
        out = tr.ifft(tr.fft(ones.numpy()))
        np.testing.assert_allclose(out, 1.0, atol=1e-12)
    # in the conservative mode a constant is not kept by a convolution
    layer = _layer(ids, 1.0, "conservative")
    inside = _interior(layer, 1.0)
    out = layer(ones).detach().numpy().ravel()[inside]
    assert np.abs(out - 1).max() > 1e-3
    layer = _layer(ids, 1.0, "average")
    out = layer(ones).detach().numpy().ravel()[inside]
    np.testing.assert_allclose(out, 1.0, atol=1e-9)


def test_invalid_projection_name(ids):
    with pytest.raises(ValueError, match="projection"):
        LocalFFT(ids, LEVEL, projection="mass")


def test_blur_floor_value_and_function(ids):
    tr = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    floor = tr.blur_floor_px()
    assert floor.shape == (ids.size,)
    assert abs(np.median(floor) - PROJECTION_BLUR_FLOOR_PX) < 0.03
    assert PROJECTION_BLUR_FLOOR_PX == pytest.approx(0.5774, abs=1e-4)


def test_blur_floor_matches_the_measured_response(ids):
    """The effective width of a tiny kernel is the geometric floor."""
    layer = _layer(ids, 0.05, "conservative", kernel_sz=5)
    inside = np.flatnonzero(_interior(layer, 0.2))[:40]
    tr = layer.transform
    x = tr.projected_x.numpy() / tr.pixel_size_rad
    y = tr.projected_y.numpy() / tr.pixel_size_rad
    resp = _responses(layer, inside)
    for k, j in enumerate(inside):
        d2 = (x - x[j]) ** 2 + (y - y[j]) ** 2
        width = math.sqrt(0.5 * (resp[k] * d2).sum() / resp[k].sum())
        assert width == pytest.approx(layer.blur_floor_px()[j], rel=0.02)


def test_warning_below_the_floor(ids):
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu",
        projection="conservative",
    )
    with pytest.warns(UserWarning, match="below the blur floor"):
        layer.set_gaussian(0.3)
    with pytest.raises(ValueError, match="below the blur floor"):
        layer.set_gaussian(0.3, on_below="error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer.set_gaussian(0.3, on_below="ignore")
        layer.set_gaussian(1.0)
        layer.set_gaussian(1.0 * layer.transform.pixel_size_rad, unit="rad")
    with pytest.warns(UserWarning):
        layer.set_gaussian(0.3 * layer.transform.pixel_size_rad, unit="rad")


def test_check_kernel_width_function():
    assert check_kernel_width(1.0, on_below="error") is True
    assert check_kernel_width(0.2, on_below="ignore") is False
    with pytest.raises(ValueError):
        check_kernel_width(-1.0)
    with pytest.raises(ValueError):
        check_kernel_width(1.0, on_below="x")
