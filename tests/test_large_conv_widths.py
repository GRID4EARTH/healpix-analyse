"""Width control of ``LargeConv``: explicit levels and the Down/Up blur floor."""

from __future__ import annotations

import math

import healpix_geo.nested as hgn
import numpy as np
import pytest
import torch

from healpix_analyse import LargeConv, down_up_blur, levels_for_width
from healpix_analyse.large_conv import _kernel_plan

LEVEL = 9
DEPTH = 7
CELL = math.sqrt(4 * math.pi / 12) / 2**LEVEL


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


def _patch():
    parent_level = LEVEL - DEPTH
    parent = 5 * 4**parent_level + 3 * 4 ** (parent_level - 1)
    ids = parent * 4**DEPTH + np.arange(4**DEPTH, dtype=np.int64)
    lon, lat = hgn.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    centre = int(np.argmax(xyz @ xyz.mean(axis=0)))
    d = np.arccos(np.clip(xyz @ xyz[centre], -1.0, 1.0))
    return ids, centre, d


def _gaussian(kernel_sz, sigma_coarse):
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    w = np.exp(-(xx**2 + yy**2) / (2 * max(sigma_coarse, 1e-6) ** 2)).ravel()
    return (w / w.sum()).reshape(1, 1, -1)


def _effective_width_cells(sigma_cells, n_levels=None):
    """Width (cells) of the response to one cell, caller recipe of the docs."""
    ids, centre, d = _patch()
    kernel_sz = 2 * math.ceil(3 * sigma_cells) + 1
    if n_levels is None:
        n_levels, _ = _kernel_plan(kernel_sz, 7)
        explicit = None
    else:
        explicit = n_levels
    blur = down_up_blur(LEVEL, n_levels, patch_depth=DEPTH) / CELL
    sigma_c = math.sqrt(max(sigma_cells**2 - blur**2, 0.0)) / 2**n_levels
    layer = LargeConv(
        LEVEL, 1, 1, kernel_sz, n_levels=explicit, cell_ids=ids,
        dtype=torch.float64, device="cpu", cache_dir=None,
    )
    layer.set_compact_kernel(_gaussian(layer.compact_kernel_sz, sigma_c))
    impulse = np.zeros(ids.size)
    impulse[centre] = 1.0
    w = np.asarray(layer(impulse)).ravel()
    return math.sqrt(np.sum(w * d**2) / (2 * w.sum())) / CELL


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


def test_explicit_levels_ignore_max_compact_kernel_size():
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


def test_down_up_blur_roughly_doubles_with_each_level():
    blurs = [down_up_blur(LEVEL, n, patch_depth=DEPTH) for n in range(0, 4)]
    assert blurs[0] == 0.0
    for lo, hi in zip(blurs[1:], blurs[2:]):
        assert 1.8 < hi / lo < 2.3
    # the first level blurs by about two cells
    assert 1.5 < blurs[1] / CELL < 2.6


def test_levels_for_width_picks_the_largest_level_below_the_width():
    b = [down_up_blur(LEVEL, n, patch_depth=DEPTH) for n in range(0, 4)]
    kw = dict(patch_depth=DEPTH)
    assert levels_for_width(LEVEL, 0.5 * b[1], **kw) == 0
    assert levels_for_width(LEVEL, 1.01 * b[1], **kw) == 1
    assert levels_for_width(LEVEL, 1.01 * b[2], **kw) == 2
    assert levels_for_width(LEVEL, 1.01 * b[3], **kw) == 3
    assert levels_for_width(LEVEL, 1.01 * b[3], max_levels=2, **kw) == 2


def test_effective_width_follows_the_requested_width_across_level_switches():
    # sigma in cells; the blur floors are about 2, 4.2 and 8.3 cells, so the
    # sweep crosses three switches (0 -> 1, 1 -> 2, 2 -> 3).
    b = [down_up_blur(LEVEL, n, patch_depth=DEPTH) / CELL for n in range(0, 4)]
    sweep = [2.2, 2.8, 3.6, 4.4, 5.0, 6.0, 7.5, 8.6, 9.5]
    levels_used = set()
    for sigma in sweep:
        n = levels_for_width(LEVEL, sigma * CELL, patch_depth=DEPTH)
        levels_used.add(n)
        eff = _effective_width_cells(sigma, n_levels=n)
        assert eff == pytest.approx(sigma, rel=0.06), (sigma, n, eff)
    assert {1, 2, 3} <= levels_used
    assert b[1] < sweep[1] and b[2] < sweep[4] and b[3] < sweep[7]


def test_default_rule_has_the_steps_described_in_the_issue():
    # With the default rule the width sits on the blur floor of the level
    # chosen from kernel_sz, whatever the requested width is.
    widths = [_effective_width_cells(s) for s in (2.2, 3.0, 3.9)]
    assert max(widths) - min(widths) < 1e-6 * widths[0]
    assert widths[0] > 1.5 * 2.2  # far above the requested 2.2 cells


@pytest.mark.parametrize("n_levels", [1, 2])
def test_constants_are_preserved(n_levels):
    ids = np.arange(12 * 4**5, dtype=np.int64)
    layer = LargeConv(
        5, 1, 1, 9, n_levels=n_levels, cell_ids=ids,
        dtype=torch.float64, device="cpu", cache_dir=None,
    )
    layer.set_compact_kernel(_gaussian(layer.compact_kernel_sz, 1.2))
    y = layer(np.full(ids.size, 3.0))
    assert np.allclose(y, 3.0, rtol=1e-6)
