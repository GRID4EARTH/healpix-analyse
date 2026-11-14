"""Tests for HealPixDecomp.compute_weighted / invert(HealPixWeightedPyramid)."""

import numpy as np
import pytest
import torch

from healpix_analyse.decomp import HealPixDecomp, HealPixWeightedPyramid


LEVELS = [3, 4]


@pytest.mark.parametrize("level", LEVELS)
def test_plain_compute_invert_unaffected(level):
    """New code path must not change existing exact-reconstruction behavior."""
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    rng = np.random.default_rng(0)
    x = rng.standard_normal(npix)

    pyr = decomp.compute(x)
    back = decomp.invert(pyr)
    assert np.max(np.abs(back - x)) < 1e-10


@pytest.mark.parametrize("level", LEVELS)
def test_constant_plus_mask_gives_proportional_bands(level):
    """P_q = c * P_m band by band, for a constant map behind an arbitrary mask."""
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    rng = np.random.default_rng(1)

    c = -2.5
    mask = rng.random(npix) > 0.3
    # also carve one large contiguous hole
    mask[: npix // 8] = False
    x = np.full(npix, c)
    x[~mask] = np.nan

    wpyr = decomp.compute_weighted(x)
    assert isinstance(wpyr, HealPixWeightedPyramid)
    for qb, mb in zip(wpyr.q.bands, wpyr.m.bands):
        np.testing.assert_allclose(qb, c * mb, atol=1e-9)


@pytest.mark.parametrize("level", LEVELS)
def test_constant_reconstructs_exactly_where_supported(level):
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    rng = np.random.default_rng(2)

    c = 7.25
    mask = np.ones(npix, dtype=bool)
    mask[100:300] = False
    x = np.full(npix, c)
    x[~mask] = np.nan

    wpyr = decomp.compute_weighted(x)
    y = decomp.invert(wpyr, restore_mask=True)
    finite = np.isfinite(y)
    assert finite.any()
    np.testing.assert_allclose(y[finite], c, atol=1e-6)


def test_explicit_weights_are_honored_and_zero_weight_is_missing():
    level = 3
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    rng = np.random.default_rng(3)

    x = rng.standard_normal(npix)
    weights = rng.random(npix)
    weights[0:50] = 0.0  # explicit zero-confidence region

    wpyr = decomp.compute_weighted(x, weights=weights)
    y = decomp.invert(wpyr, restore_mask=False)
    # Should not raise, should produce finite output at the fine resolution size
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_fully_masked_map_yields_all_nan_with_restore_mask():
    level = 3
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    x = np.full(npix, np.nan)

    wpyr = decomp.compute_weighted(x)
    y = decomp.invert(wpyr, restore_mask=True)
    assert np.isnan(y).all()

    y0 = decomp.invert(wpyr, restore_mask=False)
    np.testing.assert_allclose(y0, 0.0, atol=1e-12)


def test_gradient_flows_through_weighted_pipeline():
    level = 3
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    rng = np.random.default_rng(4)

    mask_np = rng.random(npix) > 0.2
    x_np = rng.standard_normal(npix)

    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    m = torch.tensor(mask_np.astype(np.float64), dtype=torch.float64, requires_grad=True)
    data = torch.where(m.bool(), x, torch.full_like(x, float("nan")))

    wpyr = decomp.compute_weighted(data, weights=m)
    y = decomp.invert(wpyr, restore_mask=False)
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # gradient wrt an always-masked-out input pixel should be exactly zero
    masked_out = ~torch.tensor(mask_np)
    if masked_out.any():
        assert torch.allclose(
            x.grad[masked_out], torch.zeros_like(x.grad[masked_out])
        )


def test_weights_shape_mismatch_raises():
    level = 3
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64)
    npix = 12 * 4 ** level
    x = np.zeros(npix)
    bad_weights = np.zeros((2, npix))
    with pytest.raises(ValueError):
        decomp.compute_weighted(x, weights=bad_weights)


def test_partial_sky_compute_weighted():
    level = 3
    rng = np.random.default_rng(5)
    npix_full = 12 * 4 ** level
    cell_ids = np.sort(rng.choice(npix_full, size=npix_full // 2, replace=False))
    decomp = HealPixDecomp(level=level, cell_ids=cell_ids, ellipsoid="sphere", dtype=torch.float64)

    x = rng.standard_normal(len(cell_ids))
    mask = rng.random(len(cell_ids)) > 0.25
    x[~mask] = np.nan

    wpyr = decomp.compute_weighted(x)
    y = decomp.invert(wpyr, restore_mask=False)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
