"""Conservative projection pair of LocalFFT / HealPixFFTConv (issue 77)."""

from __future__ import annotations

import math
import warnings

import healpix_geo
import numpy as np
import pytest
import torch

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


def _distance_to_outside(tr):
    """Distance, in grid pixels, from each cell to the nearest uncovered grid pixel.

    This is ``scipy.ndimage.distance_transform_edt(coverage)`` read at the grid
    pixel of each cell, computed by brute force on integer coordinates (so
    exactly), which keeps these tests free of SciPy.
    """
    px = tr.pixel_size_rad
    centre = 0.5 * (tr.grid_size - 1)
    gx = np.rint(tr.projected_x.numpy() / px + centre).astype(np.int64)
    gy = np.rint(tr.projected_y.numpy() / px + centre).astype(np.int64)
    cells = np.stack((gy, gx), axis=1)
    outside = np.argwhere(~tr.coverage_mask.numpy().astype(bool)).astype(np.int64)
    d2 = (
        (cells**2).sum(axis=1)[:, None]
        + (outside**2).sum(axis=1)[None, :]
        - 2 * cells @ outside.T
    )
    return np.sqrt(d2.min(axis=1))


def _interior(layer, sigma):
    return _distance_to_outside(layer.transform) >= math.ceil(5 * sigma) + 2


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
    edge = np.flatnonzero(_distance_to_outside(layer.transform) <= 2)
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


# ---------------------------------------------------------------------------
# Review additions
# ---------------------------------------------------------------------------

BOTH = ["average", "conservative"]


def test_round_trip_is_symmetric_and_has_unit_column_sums(ids):
    """The pair is not a transpose pair; the round trip is symmetric."""
    tr = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu", projection="conservative")
    n = ids.size
    eye = torch.eye(n, dtype=torch.float64)
    # row j of `rt` is the response to cell j, i.e. column j of the operator
    rt = tr._unproject_tensor(tr._project_tensor(eye)).numpy()
    np.testing.assert_allclose(rt, rt.T, atol=1e-14)
    np.testing.assert_allclose(rt.sum(axis=1), 1.0, atol=1e-12)
    # the two steps are not transposes of one another
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal(n))
    g = torch.as_tensor(rng.standard_normal((tr.grid_size, tr.grid_size)))
    lhs = float((tr._project_tensor(x) * g).sum())
    rhs = float((x * tr._unproject_tensor(g)).sum())
    assert abs(lhs - rhs) > 1e-3
    # the round trip of the two conventions is the same operator
    avg = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    rt_avg = avg._unproject_tensor(avg._project_tensor(eye)).numpy()
    np.testing.assert_allclose(rt_avg, rt, atol=1e-14)


def test_blur_floor_is_the_same_for_both_and_not_isotropic(ids):
    floors = []
    for mode in BOTH:
        tr = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu", projection=mode)
        floors.append((tr.blur_floor_px(), tr.blur_floor_px(per_axis=True)))
    np.testing.assert_array_equal(floors[0][0], floors[1][0])
    per_axis = floors[0][1]
    assert per_axis.shape == (ids.size, 2)
    np.testing.assert_allclose(
        floors[0][0], np.sqrt(0.5 * (per_axis**2).sum(axis=1)), rtol=1e-12
    )
    assert abs(np.median(per_axis[:, 0]) - np.median(per_axis[:, 1])) > 1e-3


def test_default_path_matches_reference_made_with_main(ids):
    """Reference written by tests/data/make_fft_default_reference.py on 272df21."""
    from pathlib import Path

    ref = np.load(Path(__file__).parent / "data" / "fft_default_reference.npz")
    np.testing.assert_array_equal(ref["ids"], ids)
    x = ref["x"]
    half = 4
    axis = np.arange(-half, half + 1, dtype=np.float64)
    profile = np.exp(-0.5 * (axis / 1.5) ** 2)
    kernel = np.outer(profile, profile)
    kernel /= kernel.sum()
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu"
    ).eval()
    layer.set_kernel(kernel[None, None])
    tr = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    # identical on the machine that made the file; the tolerance (1e-10) is for
    # a different FFT backend or BLAS on another platform
    kw = dict(rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer(torch.as_tensor(x)).detach().numpy(), ref["conv"], **kw)
    np.testing.assert_allclose(tr.project(x), ref["projected"], **kw)
    np.testing.assert_allclose(tr.ifft(tr.fft(x)), ref["round_trip"], **kw)


@pytest.mark.parametrize("projection", BOTH)
def test_float32(ids, projection):
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 11, cell_ids=ids, dtype=torch.float32, device="cpu",
        projection=projection,
    ).eval()
    layer.set_gaussian(1.0)
    inside = np.flatnonzero(_interior(layer, 1.0))[:30]
    n = layer.n_cells
    eye = torch.zeros(len(inside), n)
    eye[torch.arange(len(inside)), torch.as_tensor(inside)] = 1.0
    with torch.no_grad():
        out = layer(eye)
    assert out.dtype == torch.float32
    if projection == "conservative":
        np.testing.assert_allclose(out.reshape(len(inside), n).sum(1).numpy(), 1.0, atol=1e-4)


@pytest.mark.parametrize("projection", BOTH)
def test_nan_propagates_as_on_main(ids, projection):
    layer = _layer(ids, 1.0, projection)
    x = torch.ones(ids.size, dtype=torch.float64)
    x[3] = float("nan")
    assert torch.isnan(layer(x)).all()  # same on 272df21: the FFT spreads it


@pytest.mark.parametrize("projection", BOTH)
def test_gradients_reach_input_and_weight(ids, projection):
    layer = _layer(ids, 1.0, projection).train()
    layer.weight.requires_grad_(True)
    x = torch.as_tensor(np.random.default_rng(3).random(ids.size)).requires_grad_(True)
    layer(x).square().sum().backward()
    for grad in (x.grad, layer.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0


@pytest.mark.parametrize("sigma,size", [(1.0, 11), (2.0, 5), (10.0, 5)])
def test_kernel_sums_to_one_after_truncation(ids, sigma, size):
    layer = HealPixFFTConv(
        LEVEL, 2, 2, size, cell_ids=ids, dtype=torch.float64, device="cpu"
    )
    layer.set_gaussian(sigma, on_below="ignore", on_truncated="ignore")
    w = layer.weight.detach()
    assert float(w[0, 0].sum()) == pytest.approx(1.0, abs=1e-14)
    assert float(w[1, 1].sum()) == pytest.approx(1.0, abs=1e-14)
    assert float(w[0, 1].abs().sum()) == 0.0 and float(w[1, 0].abs().sum()) == 0.0


@pytest.mark.parametrize("projection", BOTH)
def test_wide_kernel_never_wraps_around_the_domain(ids, projection):
    """Kernel 33 pixels wide: no response sum above 1 anywhere (no wrap)."""
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 33, cell_ids=ids, dtype=torch.float64, device="cpu",
        projection=projection,
    ).eval()
    layer.set_gaussian(5.0)
    cells = np.arange(0, ids.size, 7)
    sums = _responses(layer, cells).sum(axis=1)
    if projection == "conservative":
        assert sums.max() <= 1.0 + 1e-9
    assert sums.min() > 0.0
    # edge cells lose mass, far cells keep it
    assert sums.min() < 0.9 and sums.max() > 0.99


def test_truncated_kernel_warning(ids):
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 5, cell_ids=ids, dtype=torch.float64, device="cpu"
    )
    with pytest.warns(UserWarning, match="truncated"):
        layer.set_gaussian(10.0)
    with pytest.raises(ValueError, match="truncated"):
        layer.set_gaussian(10.0, on_truncated="error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer.set_gaussian(10.0, on_truncated="ignore")
        layer.set_gaussian(0.6)  # 3 sigma = 1.8 <= half-width 2
    with pytest.raises(ValueError):
        layer.set_gaussian(1.0, on_truncated="x")


def test_warnings_point_at_the_caller(ids):
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu"
    )
    for call in (
        lambda: layer.set_gaussian(0.3),
        lambda: layer.transform.check_width(0.3),
        lambda: check_kernel_width(0.3),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        assert len(caught) == 1
        assert caught[0].filename == __file__
    layer5 = HealPixFFTConv(LEVEL, 1, 1, 5, cell_ids=ids, dtype=torch.float64, device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer5.set_gaussian(10.0)
    assert len(caught) == 1 and caught[0].filename == __file__
