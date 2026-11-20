"""Width control of ``LargeConv``: explicit levels and the Down/Up blur floor."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from healpix_analyse import LargeConv, down_up_blur, gaussian_plan
from healpix_analyse.large_conv import (
    _blur_cached,
    _calibration,
    _impulse_width,
    _kernel_plan,
    _nominal_cell,
    _patch,
)

def _old_plan(kernel_sz, max_compact):
    fine_radius = (kernel_sz - 1) // 2
    max_radius = (max_compact - 1) // 2
    if fine_radius == 0:
        return 0, 1
    n, r = 0, fine_radius
    while r > max_radius:
        n += 1
        r = math.ceil(fine_radius / 2**n)
    return n, 2 * r + 1



LEVEL = 10
DEPTH = 8
CELL = _nominal_cell(LEVEL)


def _width_cells(layer, centre, dist, n_cells, cell):
    return _impulse_width(layer, centre, dist, n_cells) / cell


@pytest.mark.parametrize("kernel_sz", [1, 3, 7, 9, 15, 17, 33, 65, 129])
@pytest.mark.parametrize("max_compact", [1 + 2, 7, 11])
def test_default_plan_is_unchanged(kernel_sz, max_compact):
    assert _kernel_plan(kernel_sz, max_compact) == _old_plan(kernel_sz, max_compact)
    n, ksz = _old_plan(kernel_sz, max_compact)
    # an explicit n_levels equal to the automatic one gives the same plan
    assert _kernel_plan(kernel_sz, max_compact, n) == (n, ksz)


def test_default_output_is_bit_identical_to_explicit_automatic_levels():
    ids = np.arange(12 * 4**5, dtype=np.int64)
    kwargs = dict(cell_ids=ids, dtype=torch.float64, device="cpu", cache_dir=None)
    a = LargeConv(5, 1, 1, 17, **kwargs)
    n, _ = _kernel_plan(17, 7)
    b = LargeConv(5, 1, 1, 17, n_levels=n, **kwargs)
    assert a.n_levels == b.n_levels == n
    kernel = np.random.default_rng(1).normal(size=(1, 1, a.compact_kernel_sz**2))
    a.set_compact_kernel(kernel)
    b.set_compact_kernel(kernel)
    x = np.random.default_rng(0).normal(size=ids.size)
    assert np.array_equal(a(x), b(x))


def test_explicit_levels_set_the_compact_kernel_size():
    n, ksz = _kernel_plan(33, 7, n_levels=1)
    assert (n, ksz) == (1, 17)
    assert _kernel_plan(33, 7, n_levels=0) == (0, 33)
    assert _kernel_plan(1, 7, n_levels=2) == (2, 1)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_bad_n_levels_is_rejected(bad):
    with pytest.raises(ValueError):
        _kernel_plan(9, 7, n_levels=bad)


def test_too_many_levels_for_the_resolution_is_rejected():
    with pytest.raises(ValueError):
        LargeConv(3, 1, 1, 9, n_levels=4, device="cpu", cache_dir=None)



def test_n_levels_and_max_compact_kernel_sz_are_exclusive():
    with pytest.raises(ValueError, match="cannot both be given"):
        LargeConv(3, 1, 1, 9, n_levels=1, max_compact_kernel_sz=7, device="cpu", cache_dir=None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 1.5, True, "2"])
def test_n_levels_validation_message(bad):
    with pytest.raises(ValueError, match="n_levels must be an integer"):
        _kernel_plan(9, 7, n_levels=bad)


@pytest.mark.parametrize(
    "args",
    [(5, 1.5), (5, float("nan")), (5, -1), (5, 6), (3, 3), (10, 6), (10.5, 1)],
)
def test_down_up_blur_rejects_bad_arguments(args):
    with pytest.raises(ValueError):
        down_up_blur(*args)


@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), 0.0, -1e-3, "x", None])
def test_gaussian_plan_rejects_bad_sigma(sigma):
    with pytest.raises(ValueError, match="sigma must be"):
        gaussian_plan(LEVEL, sigma)


@pytest.mark.parametrize("max_levels", [-1, 0, 2.7, True, float("nan")])
def test_gaussian_plan_rejects_bad_max_levels(max_levels):
    with pytest.raises(ValueError, match="max_levels"):
        gaussian_plan(LEVEL, 3 * CELL, max_levels=max_levels)


def test_gaussian_plan_rejects_widths_below_the_class_and_names_the_alternative():
    with pytest.raises(ValueError, match="gaussian_filter"):
        gaussian_plan(LEVEL, 1.5 * CELL)


def test_gaussian_plan_rejects_widths_above_the_levels_available():
    with pytest.raises(ValueError, match="above the widest"):
        gaussian_plan(LEVEL, 400 * CELL, max_levels=2)


def test_gaussian_plan_rejects_a_level_that_is_too_small():
    with pytest.raises(ValueError, match="too small"):
        gaussian_plan(3, 0.1)


def test_down_up_blur_is_cached():
    down_up_blur(LEVEL, 2)
    before = _blur_cached.cache_info().hits
    down_up_blur(LEVEL, 2)
    assert _blur_cached.cache_info().hits == before + 1


def test_down_up_blur_values_and_units():
    blurs = [down_up_blur(LEVEL, n) / CELL for n in range(0, 5)]
    # radians: about 2e-3 rad for one level at level 10
    assert 1.5e-3 < down_up_blur(LEVEL, 1) < 2.6e-3
    # even zero levels blur (the compact HealPixConv is not the identity)
    assert 0.5 < blurs[0] < 1.0
    for lo, hi in zip(blurs[1:], blurs[2:]):
        assert 1.8 < hi / lo < 2.3


@pytest.fixture(scope="module")
def dense():
    """Width of real LargeConv responses over a dense sweep through 4 switches."""
    ids, centre, dist = _patch(LEVEL, DEPTH, "WGS84")
    sweep = list(np.arange(2.3, 41.0, 0.15))
    # points just below / above each switch of the number of levels
    switches = []
    prev = gaussian_plan(LEVEL, sweep[0] * CELL).n_levels
    for lo, hi in zip(sweep[:-1], sweep[1:]):
        if gaussian_plan(LEVEL, hi * CELL).n_levels != prev:
            a, b = lo, hi
            for _ in range(30):
                mid = 0.5 * (a + b)
                if gaussian_plan(LEVEL, mid * CELL).n_levels == prev:
                    a = mid
                else:
                    b = mid
            switches.append(b)
            prev = gaussian_plan(LEVEL, hi * CELL).n_levels
    sigmas = sorted(
        sweep + [w * (1 + e) for w in switches for e in (-2e-3, -1e-3, 1e-3, 2e-3)]
    )
    plans = [gaussian_plan(LEVEL, s * CELL) for s in sigmas]
    biggest = {}
    for p in plans:
        biggest[p.n_levels] = max(biggest.get(p.n_levels, 1), p.fine_kernel_sz)
    layers = {
        n: LargeConv(LEVEL, 1, 1, k, n_levels=n, cell_ids=ids, dtype=torch.float64,
                     device="cpu", cache_dir=None)
        for n, k in biggest.items()
    }
    eff = []
    for p in plans:
        layer = layers[p.n_levels]
        layer.set_gaussian_kernel(p.compact_sigma)
        eff.append(_impulse_width(layer, centre, dist, ids.size) / CELL)
    return np.array(sigmas), np.array(eff), switches, plans


def test_gaussian_plan_crosses_four_level_switches(dense):
    sigmas, eff, switches, plans = dense
    assert len(sigmas) >= 200
    assert len(switches) == 4  # about 4.7, 9.35, 18.65 and 37.3 cells
    assert [p.n_levels for p in plans[:1]] == [1]
    assert {p.n_levels for p in plans} == {1, 2, 3, 4, 5}


def test_effective_width_is_strictly_increasing_in_sigma(dense):
    sigmas, eff, switches, plans = dense
    d = np.diff(eff)
    assert np.all(d > 0.0), (sigmas[np.argmin(d)], d.min())
    # points on both sides of each switch are included and increase too
    for w in switches:
        near = (sigmas > w * 0.997) & (sigmas < w * 1.003)
        assert near.sum() >= 4
        assert np.all(np.diff(eff[near]) > 0.0)


def test_effective_width_matches_the_requested_width(dense):
    sigmas, eff, switches, plans = dense
    # measured 3.6e-4 at the calibration patch; tolerance 2e-3 relative
    assert np.max(np.abs(eff - sigmas) / sigmas) < 2e-3


def test_no_shelf_above_a_switch(dense):
    sigmas, eff, switches, plans = dense
    # the increment per unit sigma never falls below a quarter of the ideal one
    slope = np.diff(eff) / np.diff(sigmas)
    assert slope.min() > 0.25


def test_reference_cell_calibrates_on_another_patch():
    level, depth = 9, 7
    cell = _nominal_cell(level)
    parent_level = level - depth
    parent = 0 * 4**parent_level + 3 * 4 ** (parent_level - 1)  # centre of base face 0
    ids, centre, dist = _patch(level, depth, "WGS84", parent)
    ref = int(ids[centre])
    layer = None
    for sigma in (3.2 * cell, 4.0 * cell, 6.0 * cell):
        plan = gaussian_plan(level, sigma, patch_depth=depth, reference_cell=ref)
        if layer is None or layer.n_levels != plan.n_levels:
            layer = LargeConv(level, 1, 1, plan.fine_kernel_sz, n_levels=plan.n_levels,
                              cell_ids=ids, dtype=torch.float64, device="cpu", cache_dir=None)
        layer.set_gaussian_kernel(plan.compact_sigma)
        assert _impulse_width(layer, centre, dist, ids.size) == pytest.approx(sigma, rel=2e-3)


@pytest.mark.parametrize("n_levels", [1, 2])
def test_gaussian_kernel_preserves_constants(n_levels):
    ids = np.arange(12 * 4**5, dtype=np.int64)
    layer = LargeConv(
        5, 1, 1, 33, n_levels=n_levels, cell_ids=ids,
        dtype=torch.float64, device="cpu", cache_dir=None,
    )
    layer.set_gaussian_kernel(0.5)
    y = layer(np.full(ids.size, 3.0))
    assert np.allclose(y, 3.0, rtol=1e-6)


def test_set_gaussian_kernel_validation():
    layer = LargeConv(4, 1, 2, 3, n_levels=0, device="cpu", cache_dir=None)
    with pytest.raises(ValueError, match="in_channels"):
        layer.set_gaussian_kernel(1.0)
    layer = LargeConv(4, 1, 1, 3, n_levels=0, device="cpu", cache_dir=None)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="compact_sigma"):
            layer.set_gaussian_kernel(bad)


def _small_domain():
    """Level 8 cells of a 16384-cell block (patch_depth 7), a 3000-cell subset."""
    parent = 5 * 4 ** (8 - 7) + 3 * 4 ** (8 - 8)
    ids = parent * 4**7 + np.arange(4**7, dtype=np.int64)
    return ids[2000:5000]


def test_from_gaussian_equals_the_manual_construction():
    level, sigma = 8, 4.0 * _nominal_cell(8)
    ids = _small_domain()
    layer = LargeConv.from_gaussian(
        level, sigma, ids, patch_depth=7, dtype=torch.float64, device="cpu", cache_dir=None
    )
    plan = gaussian_plan(level, sigma, patch_depth=7, cell_ids=ids)
    manual = LargeConv(
        level, 1, 1, plan.fine_kernel_sz, n_levels=plan.n_levels, cell_ids=ids,
        dtype=torch.float64, device="cpu", cache_dir=None,
    )
    manual.set_gaussian_kernel(plan)
    x = np.random.default_rng(3).normal(size=ids.size)
    assert layer.gaussian_plan == plan
    assert np.array_equal(layer(x), manual(x))


def test_from_gaussian_takes_the_location_from_the_domain():
    level, sigma = 8, 4.0 * _nominal_cell(8)
    ids = _small_domain()
    a = gaussian_plan(level, sigma, patch_depth=7, cell_ids=ids)
    b = gaussian_plan(level, sigma, patch_depth=7)  # equatorial block
    c = gaussian_plan(level, sigma, patch_depth=7, reference_cell=int(ids[0]))
    # this domain lies in a block of base face 5: same calibration as the default
    assert a == b == c
    # a block elsewhere gives another calibration
    other = gaussian_plan(level, sigma, patch_depth=7, reference_cell=0)
    assert other.compact_sigma != a.compact_sigma


def test_default_location_and_explicit_reference_in_the_same_block_share_the_cache():
    ids, centre, _ = _patch(LEVEL, DEPTH, "WGS84")
    gaussian_plan(LEVEL, 5 * CELL)
    before = _calibration.cache_info()
    gaussian_plan(LEVEL, 5 * CELL, reference_cell=int(ids[centre]))
    gaussian_plan(LEVEL, 5 * CELL, reference_cell=int(ids[0]))
    after = _calibration.cache_info()
    assert after.misses == before.misses
    assert after.hits > before.hits


def test_wrong_layer_for_a_plan_is_refused():
    level = 8
    plan = gaussian_plan(level, 8.0 * _nominal_cell(level), patch_depth=7)
    ids = _small_domain()
    # forgetting n_levels: the automatic rule picks other levels
    wrong = LargeConv(level, 1, 1, plan.fine_kernel_sz, cell_ids=ids,
                      device="cpu", cache_dir=None)
    if wrong.n_levels != plan.n_levels:
        with pytest.raises(ValueError, match="n_levels"):
            wrong.set_gaussian_kernel(plan)
    # kernel_sz in place of fine_kernel_sz: compact kernel too small
    small = LargeConv(level, 1, 1, plan.kernel_sz, n_levels=plan.n_levels,
                      cell_ids=ids, device="cpu", cache_dir=None)
    with pytest.raises(ValueError, match="fine_kernel_sz"):
        small.set_gaussian_kernel(plan)


def test_range_errors_state_radians_and_cells():
    with pytest.raises(ValueError) as low:
        gaussian_plan(LEVEL, 1.5 * CELL)
    assert "rad" in str(low.value) and "cells" in str(low.value)
    with pytest.raises(ValueError) as high:
        gaussian_plan(LEVEL, 400 * CELL, max_levels=2)
    assert "rad" in str(high.value) and "cells" in str(high.value)


def test_width_spread_over_impulse_cells_inside_the_domain():
    # The accuracy figure is for the calibration (centre) cell.  Cells up to
    # 60 cells from it (patch of 256 x 256 cells) stay within the bounds
    # below (measured: 0.993-1.027 at 3 cells, 0.995-1.005 at 6, 0.998-1.002
    # at 9, 0.999-1.000 at 18).  A wider sample varies more: up to 240 cells
    # from the calibration cell the spread is 0.92-1.04 at 3 cells, and
    # impulses within 5 sigma of the patch edge lose their tail (see docs).
    ids, centre, dist0 = _patch(LEVEL, DEPTH, "WGS84")
    import healpix_geo.nested as hgn

    lon, lat = hgn.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    cand = np.where(dist0 < 60 * CELL)[0]
    imp = np.random.default_rng(0).choice(cand, 24, replace=False)
    batch = np.zeros((imp.size, ids.size))
    batch[np.arange(imp.size), imp] = 1.0
    bounds = {3.0: (0.985, 1.035), 6.0: (0.99, 1.01), 9.0: (0.99, 1.01), 18.0: (0.995, 1.005)}
    for sigma, (lo, hi) in bounds.items():
        layer = LargeConv.from_gaussian(
            LEVEL, sigma * CELL, ids, dtype=torch.float64, device="cpu", cache_dir=None
        )
        y = np.asarray(layer(batch))[:, 0]
        widths = []
        for k, c in enumerate(imp):
            d = np.arccos(np.clip(xyz @ xyz[c], -1.0, 1.0))
            widths.append(math.sqrt(np.sum(y[k] * d**2) / (2 * y[k].sum())) / CELL)
        ratio = np.array(widths) / sigma
        assert lo < ratio.min() and ratio.max() < hi, (sigma, ratio.min(), ratio.max())
        # LargeConv does not conserve the total of one cell: 0.95-1.04 here
        assert 0.9 < y.sum(axis=1).min() and y.sum(axis=1).max() < 1.1


def test_partial_domain_edge_is_a_zero_padding_effect():
    import healpix_geo.nested as hgn

    level = 8
    cell = _nominal_cell(level)
    allids = np.arange(12 * 4**level)
    lon, lat = hgn.healpix_to_lonlat(allids, level, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    c0 = np.array([math.cos(math.radians(40)) * math.cos(0.5),
                   math.cos(math.radians(40)) * math.sin(0.5), math.sin(math.radians(40))])
    radius = math.radians(8.0)
    ids = allids[np.arccos(np.clip(xyz @ c0, -1, 1)) < radius]
    pos = xyz[ids]
    r = np.arccos(np.clip(pos @ c0, -1, 1))
    layer = LargeConv.from_gaussian(
        level, 3 * cell, ids, patch_depth=7, dtype=torch.float64, device="cpu", cache_dir=None
    )

    def width_total(frac):
        k = int(np.argmin(np.abs(r - frac * radius)))
        imp = np.zeros(ids.size)
        imp[k] = 1.0
        y = np.asarray(layer(imp)).ravel()
        d = np.arccos(np.clip(pos @ pos[k], -1, 1))
        return math.sqrt(np.sum(y * d**2) / (2 * y.sum())) / cell / 3.0, y.sum()

    w_in, t_in = width_total(0.0)
    w_edge, t_edge = width_total(0.95)
    assert 0.97 < w_in < 1.03          # interior: close to the request
    assert w_edge < w_in - 0.02        # edge: narrower, zero padding removes the tail
    assert 0.85 < t_edge < 1.1         # the total of one cell is not conserved
    const = np.asarray(layer(np.ones(ids.size))).ravel()
    assert const[np.argmin(r)] == pytest.approx(1.0, abs=1e-3)
    assert 0.9 < const.min() < 1.0     # a constant drops near the edge
