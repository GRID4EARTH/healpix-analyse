"""Tests for ``HealPixDown(mode="mean")``.

The mean over the 4 NESTED children is the conservative downsampling on
HEALPix (equal-area cells): the global mean of the output must equal the
global mean of the input, on every level of a hierarchy.
"""

import numpy as np
import pytest
import torch

from healpix_geo import nested

from healpix_analyse.down import HealPixDown


def _temperature_like(level: int) -> np.ndarray:
    """A field with a strong latitude gradient plus zonal structure."""
    n = 12 * 4**level
    lon, lat = nested.healpix_to_lonlat(np.arange(n), level, ellipsoid="sphere")
    lo, la = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    return (
        288.0
        - 45.0 * np.sin(la) ** 2
        + 3.0 * np.sin(3 * lo) * np.cos(la) ** 2
        + 2.0 * np.cos(2 * lo + 1) * np.sin(2 * la)
    )


def _down(level, **kwargs):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float64)
    return HealPixDown(level, mode="mean", **kwargs)


def test_full_sphere_matches_reshape_mean():
    level = 4
    x = _temperature_like(level)

    y, cell_ids_out = _down(level)(x)

    np.testing.assert_allclose(y, x.reshape(-1, 4).mean(-1), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(cell_ids_out, np.arange(12 * 4 ** (level - 1)))


def test_hierarchy_conserves_global_mean():
    level = 6
    x = _temperature_like(level)
    expected = x.mean()

    for lvl in range(level, 0, -1):
        x, _ = _down(lvl)(x)
        assert x.shape == (12 * 4 ** (lvl - 1),)
        assert x.mean() == pytest.approx(expected, rel=1e-13)


def test_smooth_does_not_conserve_global_mean():
    """Documents why ``mean`` exists: ``smooth`` only normalises rows."""
    level = 3
    x = _temperature_like(level)

    y_smooth, _ = HealPixDown(level, mode="smooth", device="cpu", dtype=torch.float64)(x)
    y_mean, _ = _down(level)(x)

    assert abs(y_mean.mean() - x.mean()) < 1e-12
    assert abs(y_smooth.mean() - x.mean()) > 1e-3


def test_constant_field_is_preserved():
    y, _ = _down(3)(np.full(12 * 4**3, 288.0))
    np.testing.assert_array_equal(y, 288.0)


def test_batch_and_types():
    level = 3
    n = 12 * 4**level
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, n))
    down = _down(level)

    y_np, _ = down(x)
    assert isinstance(y_np, np.ndarray) and y_np.shape == (5, n // 4)
    np.testing.assert_allclose(y_np, x.reshape(5, -1, 4).mean(-1), atol=1e-12)

    y_t, _ = down(torch.as_tensor(x))
    assert isinstance(y_t, torch.Tensor) and y_t.shape == (5, n // 4)
    np.testing.assert_allclose(y_t.numpy(), y_np, atol=0)


def test_is_differentiable():
    level = 2
    x = torch.randn(12 * 4**level, dtype=torch.float64, requires_grad=True)

    y, _ = _down(level)(x)
    y.sum().backward()

    # every input cell contributes exactly 1/4 to exactly one output cell
    np.testing.assert_allclose(x.grad.numpy(), 0.25, atol=1e-15)


def test_partial_sky_unsorted_complete_parents():
    level = 4
    parents = np.array([7, 100, 3, 512])
    cell_ids = (4 * parents[:, None] + np.arange(4)).ravel()
    rng = np.random.default_rng(1)
    cell_ids = rng.permutation(cell_ids)  # order must not matter
    x_full = _temperature_like(level)

    down = _down(level, cell_ids=cell_ids)
    y, cell_ids_out = down(x_full[cell_ids])

    np.testing.assert_array_equal(cell_ids_out, np.sort(parents))
    np.testing.assert_allclose(
        y, x_full.reshape(-1, 4).mean(-1)[cell_ids_out], atol=1e-12
    )
    np.testing.assert_array_equal(down.coverage, 1.0)
    assert y.mean() == pytest.approx(x_full[cell_ids].mean(), rel=1e-13)


def test_partial_sky_incomplete_parents():
    level = 4
    # parent 10: all 4 children; parent 11: 2 children; parent 300: 1 child
    cell_ids = np.array([43, 40, 41, 42, 47, 44, 1203])
    x_full = _temperature_like(level)
    x = x_full[cell_ids]

    down = _down(level, cell_ids=cell_ids)
    y, cell_ids_out = down(x)

    np.testing.assert_array_equal(cell_ids_out, [10, 11, 300])
    np.testing.assert_allclose(
        y,
        [x_full[40:44].mean(), x_full[[44, 47]].mean(), x_full[1203]],
        atol=1e-12,
    )
    np.testing.assert_array_equal(down.coverage, [1.0, 0.5, 0.25])
    # conservation holds once the output is weighted by the coverage
    weighted = (y * down.coverage).sum() / down.coverage.sum()
    assert weighted == pytest.approx(x.mean(), rel=1e-13)


def test_nan_propagates_to_its_parent_only():
    level = 3
    x = _temperature_like(level)
    x[4 * 17 + 2] = np.nan

    y, _ = _down(level)(x)

    assert np.isnan(y[17])
    assert np.isfinite(np.delete(y, 17)).all()


def test_invalid_mode_and_coverage_guard():
    with pytest.raises(ValueError, match="mean"):
        HealPixDown(3, mode="average", device="cpu")
    with pytest.raises(AttributeError):
        HealPixDown(3, mode="maxpool", device="cpu").coverage


# ---------------------------------------------------------------------------
# skipna / valid_fraction
# ---------------------------------------------------------------------------

def test_skipna_requires_mean_mode():
    with pytest.raises(ValueError, match="skipna"):
        HealPixDown(3, mode="smooth", skipna=True, device="cpu")


def test_skipna_mean_and_valid_fraction():
    level = 3
    x = _temperature_like(level)
    x[4 * 5 + 1] = np.nan                  # parent 5: 3 finite children
    x[4 * 9: 4 * 9 + 4] = np.nan           # parent 9: no finite child
    x[4 * 12 + 3] = np.inf                 # inf is skipped too

    down = _down(level, skipna=True)
    y, _ = down(x)
    frac = down.valid_fraction(x)

    ref = x.reshape(-1, 4).copy()
    ref[~np.isfinite(ref)] = np.nan
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmean(ref, axis=-1)
    np.testing.assert_allclose(y, expected, atol=1e-12)
    assert np.isnan(y[9])
    assert frac[5] == 0.75 and frac[9] == 0.0 and frac[12] == 0.75
    assert (np.delete(frac, [5, 9, 12]) == 1.0).all()

    # weighted by valid_fraction, the output mean is the mean of finite input
    ok = frac > 0
    weighted = (y[ok] * frac[ok]).sum() / frac.sum()
    assert weighted == pytest.approx(x[np.isfinite(x)].mean(), rel=1e-13)

    # without skipna the NaN still propagates; valid_fraction is the same
    plain = _down(level)
    assert np.isnan(plain(x)[0][5])
    np.testing.assert_array_equal(plain.valid_fraction(x), frac)


def test_skipna_partial_sky_combines_absent_and_nan():
    level = 4
    cell_ids = np.array([43, 40, 41, 42, 47, 44, 1203])   # parents 10, 11, 300
    x = _temperature_like(level)[cell_ids]
    x[1] = np.nan                                          # cell 40

    down = _down(level, cell_ids=cell_ids, skipna=True)
    y, _ = down(x)
    frac = down.valid_fraction(x)

    np.testing.assert_array_equal(frac, [0.75, 0.5, 0.25])
    np.testing.assert_array_equal(down.coverage, [1.0, 0.5, 0.25])
    np.testing.assert_allclose(y[0], x[[0, 2, 3]].mean(), atol=1e-12)
    weighted = (y * frac).sum() / frac.sum()
    assert weighted == pytest.approx(np.nanmean(x), rel=1e-13)


def test_skipna_hierarchy_with_carried_weights_is_conservative():
    """The recipe in the ``valid_fraction`` docstring."""
    level = 5
    x = _temperature_like(level)
    rng = np.random.default_rng(3)
    x[rng.random(x.size) < 0.3] = np.nan
    expected = np.nanmean(x)

    valid = np.isfinite(x).astype(float)
    num = np.where(np.isfinite(x), x, 0.0)
    for lvl in range(level, 0, -1):
        d = _down(lvl)
        num, _ = d(num)
        valid, _ = d(valid)
    assert num.sum() / valid.sum() == pytest.approx(expected, rel=1e-12)


def test_skipna_is_differentiable_on_finite_inputs():
    level = 2
    x = torch.randn(12 * 4**level, dtype=torch.float64)
    x[6] = torch.nan
    x.requires_grad_(True)

    y, _ = _down(level, skipna=True)(x)
    y.sum().backward()

    grad = x.grad.numpy()
    assert grad[6] == 0.0
    np.testing.assert_allclose(grad[[4, 5, 7]], 1 / 3, atol=1e-15)
    np.testing.assert_allclose(np.delete(grad, [4, 5, 6, 7]), 0.25, atol=1e-15)


def test_valid_fraction_guard():
    with pytest.raises(AttributeError):
        HealPixDown(3, mode="maxpool", device="cpu").valid_fraction(np.zeros(768))


# ---------------------------------------------------------------------------
# sparse_matrix, HealPixUp pairing
# ---------------------------------------------------------------------------

def test_sparse_matrix_matches_forward():
    level = 4
    cell_ids = np.array([43, 40, 41, 42, 47, 44, 1203])
    x = _temperature_like(level)[cell_ids]
    down = _down(level, cell_ids=cell_ids)

    M = down.sparse_matrix()
    assert M.shape == (3, 7)
    y_M = torch.sparse.mm(M, torch.as_tensor(x)[:, None])[:, 0].numpy()
    np.testing.assert_allclose(y_M, down(x)[0], atol=1e-12)


def test_up_paired_with_mean_down_copies_parent():
    from healpix_analyse.up import HealPixUp

    level = 3
    down = _down(level)
    up = HealPixUp(level - 1, paired_down=down, device="cpu", dtype=torch.float64)
    coarse = np.arange(12 * 4 ** (level - 1), dtype=float)

    fine, ids = up(coarse)
    np.testing.assert_array_equal(fine, np.repeat(coarse, 4))
    np.testing.assert_array_equal(ids, np.arange(12 * 4**level))


# ---------------------------------------------------------------------------
# HealPixResampler / HealPixDecomp wiring
# ---------------------------------------------------------------------------

def test_resampler_down_mode_mean():
    from healpix_analyse.resample import HealPixResampler, resample_healpix

    x = _temperature_like(5)
    expected = x.reshape(-1, 64).mean(-1)                  # level 5 -> 2

    op = HealPixResampler(5, 2, down_mode="mean", device="cpu", dtype=torch.float64)
    y, _ = op(x)
    np.testing.assert_allclose(y, expected, atol=1e-12)
    assert y.mean() == pytest.approx(x.mean(), rel=1e-13)
    assert "down_mode='mean'" in repr(op)

    y2, _ = resample_healpix(
        x, in_level=5, out_level=2, down_mode="mean", device="cpu",
        dtype=torch.float64,
    )
    np.testing.assert_allclose(y2, y, atol=0)

    # default unchanged
    assert HealPixResampler(5, 4, device="cpu").layers[0].mode == "smooth"
    with pytest.raises(ValueError, match="down_mode"):
        HealPixResampler(5, 4, down_mode="maxpool", device="cpu")


def test_resampler_down_mode_mean_skips_nan_per_step():
    from healpix_analyse.resample import HealPixResampler

    x = _temperature_like(3)
    x[4 * 7] = np.nan
    y, _ = HealPixResampler(
        3, 2, down_mode="mean", device="cpu", dtype=torch.float64
    )(x)
    np.testing.assert_allclose(y[7], np.nanmean(x[28:32]), atol=1e-12)


def test_decomp_down_mode_mean_reconstructs_and_conserves():
    from healpix_analyse.decomp import HealPixDecomp

    level = 4
    x = _temperature_like(level)
    dec = HealPixDecomp(level, down_mode="mean", device="cpu", dtype=torch.float64)
    pyr = dec.compute(x)

    np.testing.assert_allclose(dec.invert(pyr), x, atol=1e-10)
    assert pyr.coarse.mean() == pytest.approx(x.mean(), rel=1e-13)
    # Haar-like: each detail band sums to zero over every group of 4 children
    for band in pyr.details:
        np.testing.assert_allclose(band.reshape(-1, 4).sum(-1), 0.0, atol=1e-10)
    assert "down_mode='mean'" in repr(dec)
    assert HealPixDecomp(2, device="cpu").down_layers[0].mode == "smooth"


def test_decomp_down_mode_mean_partial_sky():
    from healpix_analyse.decomp import HealPixDecomp

    level = 4
    cell_ids = np.array([43, 40, 41, 42, 47, 44, 1203, 1202])
    x = _temperature_like(level)[cell_ids]
    dec = HealPixDecomp(
        level, cell_ids=cell_ids, Jmax=2, down_mode="mean",
        device="cpu", dtype=torch.float64,
    )
    pyr = dec.compute(x)
    np.testing.assert_allclose(dec.invert(pyr), x, atol=1e-10)
