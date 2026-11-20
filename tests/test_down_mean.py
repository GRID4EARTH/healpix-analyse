"""Tests for ``HealPixDown(mode="mean")`` and the ``down_mode`` keyword.

The mean over the 4 NESTED children is the conservative downsampling on
HEALPix (equal-area cells): the global mean of the output must equal the
global mean of a NaN-free full-sphere input, on every level of a hierarchy.

Tolerances: inputs are float64 temperature-like values (~250-300).  One
mean or sum over a few thousand such values carries a round-off of about
``n * 1e-16 * 300``, so values compared with the default ``rtol=1e-7`` need
no explicit tolerance; comparisons with zero (detail sums) use
``atol=ZERO_ATOL`` and global means use ``rel=MEAN_REL``.
"""

import numpy as np
import pytest
import torch

from healpix_geo import nested

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.down import HealPixDown
from healpix_analyse.resample import HealPixResampler, resample_healpix
from healpix_analyse.up import HealPixUp

# sums of <= 4**6 values of ~300 in float64: |error| < 4**6 * 300 * 1e-16
ZERO_ATOL = 1e-9
# relative error of a float64 mean of <= 12 * 4**6 values
MEAN_REL = 1e-12

# level-4 partial sky: parent 10 has its 4 children, parents 11 and 300 two
PARTIAL_IDS = np.array([43, 40, 41, 42, 47, 44, 1203, 1202])


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


def _sum_per_parent(values, cell_ids):
    parents, inverse = np.unique(cell_ids // 4, return_inverse=True)
    sums = np.zeros(parents.size)
    np.add.at(sums, inverse, values)
    return sums


# ---------------------------------------------------------------------------
# HealPixDown(mode="mean")
# ---------------------------------------------------------------------------

def test_full_sphere_matches_reshape_mean():
    level = 4
    x = _temperature_like(level)

    actual, cell_ids_out = _down(level)(x)

    np.testing.assert_allclose(actual, x.reshape(-1, 4).mean(-1))
    np.testing.assert_array_equal(cell_ids_out, np.arange(12 * 4 ** (level - 1)))


@pytest.mark.parametrize("level", [6, 5, 4, 3, 2, 1])
def test_each_step_conserves_global_mean(level):
    """Conservation at every step = conservation over the whole hierarchy."""
    x = _temperature_like(level)

    actual, _ = _down(level)(x)

    assert actual.shape == (12 * 4 ** (level - 1),)
    assert actual.mean() == pytest.approx(x.mean(), rel=MEAN_REL)


def test_smooth_does_not_conserve_global_mean():
    """Documents why ``mean`` exists: ``smooth`` only normalises rows."""
    level = 3
    x = _temperature_like(level)

    y_smooth, _ = HealPixDown(level, mode="smooth", device="cpu", dtype=torch.float64)(x)
    y_mean, _ = _down(level)(x)

    assert y_mean.mean() == pytest.approx(x.mean(), rel=MEAN_REL)
    # measured drift of smooth at this level is ~1e-2 K
    assert abs(y_smooth.mean() - x.mean()) > 1e-3


def test_constant_field_is_preserved():
    actual, _ = _down(3)(np.full(12 * 4**3, 288.0))
    np.testing.assert_array_equal(actual, 288.0)


def test_batch_and_types():
    level = 3
    n = 12 * 4**level
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, n)) + 288.0
    down = _down(level)

    y_np, _ = down(x)
    assert isinstance(y_np, np.ndarray) and y_np.shape == (5, n // 4)
    np.testing.assert_allclose(y_np, x.reshape(5, -1, 4).mean(-1))

    y_t, _ = down(torch.as_tensor(x))
    assert isinstance(y_t, torch.Tensor) and y_t.shape == (5, n // 4)
    np.testing.assert_array_equal(y_t.numpy(), y_np)


def test_is_differentiable():
    level = 2
    x = torch.randn(12 * 4**level, dtype=torch.float64, requires_grad=True)

    y, _ = _down(level)(x)
    y.sum().backward()

    # every input cell contributes exactly 1/4 to exactly one output cell
    np.testing.assert_allclose(x.grad.numpy(), 0.25)


def test_partial_sky_unsorted_complete_parents():
    level = 4
    parents = np.array([7, 100, 3, 512])
    cell_ids = (4 * parents[:, None] + np.arange(4)).ravel()
    rng = np.random.default_rng(1)
    cell_ids = rng.permutation(cell_ids)  # order must not matter
    x_full = _temperature_like(level)

    down = _down(level, cell_ids=cell_ids)
    actual, cell_ids_out = down(x_full[cell_ids])

    np.testing.assert_array_equal(cell_ids_out, np.sort(parents))
    np.testing.assert_allclose(actual, x_full.reshape(-1, 4).mean(-1)[cell_ids_out])
    np.testing.assert_array_equal(down.coverage, 1.0)
    assert actual.mean() == pytest.approx(x_full[cell_ids].mean(), rel=MEAN_REL)


def test_partial_sky_incomplete_parents():
    level = 4
    # parent 10: all 4 children; parent 11: 2 children; parent 300: 1 child
    cell_ids = np.array([43, 40, 41, 42, 47, 44, 1203])
    x_full = _temperature_like(level)
    x = x_full[cell_ids]

    down = _down(level, cell_ids=cell_ids)
    actual, cell_ids_out = down(x)

    np.testing.assert_array_equal(cell_ids_out, [10, 11, 300])
    expected = [x_full[40:44].mean(), x_full[[44, 47]].mean(), x_full[1203]]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(down.coverage, [1.0, 0.5, 0.25])
    # conservation holds once the output is weighted by the coverage
    weighted = (actual * down.coverage).sum() / down.coverage.sum()
    assert weighted == pytest.approx(x.mean(), rel=MEAN_REL)


def test_partial_sky_chain_with_carried_coverage_is_conservative():
    """The multi-step recipe in the ``HealPixDown`` docstring."""
    level = 5
    rng = np.random.default_rng(4)
    cell_ids = rng.choice(12 * 4**level, size=3000, replace=False)
    x = _temperature_like(level)[cell_ids]

    data, ones, ids = x, np.ones_like(x), cell_ids
    for lvl in range(level, 2, -1):     # the recipe itself is a chain
        down = _down(lvl, cell_ids=ids)
        data = down(data)[0] * down.coverage
        ones = down(ones)[0] * down.coverage
        ids = down.cell_ids_out

    # level-2 ancestors are cell_ids // 64; _sum_per_parent takes one // 4
    expected = _sum_per_parent(x, cell_ids // 16)
    count = _sum_per_parent(np.ones_like(x), cell_ids // 16)
    np.testing.assert_allclose(data / ones, expected / count)


def test_nan_propagates_to_its_parent_only():
    level = 3
    x = _temperature_like(level)
    x[4 * 17 + 2] = np.nan

    actual, _ = _down(level)(x)

    assert np.isnan(actual[17])
    assert np.isfinite(np.delete(actual, 17)).all()


@pytest.mark.parametrize(
    ["kwargs", "match"],
    (
        pytest.param({"mode": "average"}, "mode must be", id="bad-mode"),
        pytest.param({"mode": "smooth", "skipna": True}, "skipna", id="skipna-smooth"),
    ),
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HealPixDown(3, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ["call", "match"],
    (
        pytest.param(
            lambda d: d.coverage,
            "coverage is only defined for mode='mean'",
            id="coverage",
        ),
        pytest.param(
            lambda d: d.valid_fraction(np.zeros(768)),
            "valid_fraction is only defined for mode='mean'",
            id="valid_fraction",
        ),
        pytest.param(lambda d: d.sparse_matrix(), "maxpool is not linear", id="sparse_matrix"),
    ),
)
def test_mean_only_api_on_maxpool(call, match):
    # ValueError, not AttributeError: nn.Module.__getattr__ would hide the
    # message of an AttributeError raised inside the ``coverage`` property
    with pytest.raises(ValueError, match=match):
        call(HealPixDown(3, mode="maxpool", device="cpu"))


# ---------------------------------------------------------------------------
# skipna / valid_fraction
# ---------------------------------------------------------------------------

def test_skipna_mean_and_valid_fraction():
    level = 3
    x = _temperature_like(level)
    x[4 * 5 + 1] = np.nan                  # parent 5: 3 finite children
    x[4 * 9: 4 * 9 + 4] = np.nan           # parent 9: no finite child
    x[4 * 12 + 3] = np.inf                 # inf is skipped too

    down = _down(level, skipna=True)
    actual, _ = down(x)
    frac = down.valid_fraction(x)

    ref = x.reshape(-1, 4).copy()
    ref[~np.isfinite(ref)] = np.nan
    with np.errstate(invalid="ignore"), pytest.warns(RuntimeWarning):
        expected = np.nanmean(ref, axis=-1)
    np.testing.assert_allclose(actual, expected)    # NaN at parent 9 in both
    expected_frac = np.ones(actual.size)
    expected_frac[[5, 9, 12]] = [0.75, 0.0, 0.75]
    np.testing.assert_array_equal(frac, expected_frac)

    # weighted by valid_fraction, the output mean is the mean of finite input
    ok = frac > 0
    weighted = (actual[ok] * frac[ok]).sum() / frac.sum()
    assert weighted == pytest.approx(x[np.isfinite(x)].mean(), rel=MEAN_REL)

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
    actual, _ = down(x)
    frac = down.valid_fraction(x)

    np.testing.assert_array_equal(frac, [0.75, 0.5, 0.25])
    np.testing.assert_array_equal(down.coverage, [1.0, 0.5, 0.25])
    np.testing.assert_allclose(actual[0], x[[0, 2, 3]].mean())
    weighted = (actual * frac).sum() / frac.sum()
    assert weighted == pytest.approx(np.nanmean(x), rel=MEAN_REL)


def test_skipna_hierarchy_with_carried_weights_is_conservative():
    """The recipe in the ``valid_fraction`` docstring."""
    level = 5
    x = _temperature_like(level)
    rng = np.random.default_rng(3)
    x[rng.random(x.size) < 0.3] = np.nan
    expected = np.nanmean(x)

    valid = np.isfinite(x).astype(float)
    num = np.where(np.isfinite(x), x, 0.0)
    for lvl in range(level, 0, -1):     # the recipe itself is a chain
        d = _down(lvl)
        num, _ = d(num)
        valid, _ = d(valid)
    assert num.sum() / valid.sum() == pytest.approx(expected, rel=MEAN_REL)


def test_skipna_is_differentiable_on_finite_inputs():
    level = 2
    x = torch.randn(12 * 4**level, dtype=torch.float64)
    x[6] = torch.nan
    x.requires_grad_(True)

    y, _ = _down(level, skipna=True)(x)
    y.sum().backward()

    expected = np.full(x.numel(), 0.25)
    expected[[4, 5, 7]] = 1 / 3
    expected[6] = 0.0
    np.testing.assert_allclose(x.grad.numpy(), expected)


# ---------------------------------------------------------------------------
# sparse_matrix, HealPixUp pairing
# ---------------------------------------------------------------------------

def test_sparse_matrix_matches_forward():
    level = 4
    x = _temperature_like(level)[PARTIAL_IDS]
    down = _down(level, cell_ids=PARTIAL_IDS)

    M = down.sparse_matrix()
    assert M.shape == (3, 8)
    np.testing.assert_allclose(M.to_dense().numpy().sum(axis=1), 1.0)
    actual = torch.sparse.mm(M, torch.as_tensor(x)[:, None])[:, 0].numpy()
    np.testing.assert_allclose(actual, down(x)[0])


def test_up_paired_with_mean_down_copies_parent():
    level = 3
    down = _down(level)
    up = HealPixUp(level - 1, paired_down=down, device="cpu", dtype=torch.float64)
    coarse = np.arange(12 * 4 ** (level - 1), dtype=float)

    actual, ids = up(coarse)
    np.testing.assert_array_equal(actual, np.repeat(coarse, 4))
    np.testing.assert_array_equal(ids, np.arange(12 * 4**level))


@pytest.mark.parametrize(
    ["up_norm", "expected"],
    (
        # n children present: 4 for parent 10, 2 for 11 and for 300
        pytest.param("col_l1", [1, 1, 1, 1, 2, 2, 3, 3], id="col_l1-parent"),
        pytest.param("adjoint", [.25, .25, .25, .25, 1, 1, 1.5, 1.5], id="adjoint-parent-over-n"),
        pytest.param("diag_l2", [4, 4, 4, 4, 4, 4, 6, 6], id="diag_l2-n-times-parent"),
    ),
)
def test_up_paired_with_mean_down_partial_sky(up_norm, expected):
    level = 4
    down = _down(level, cell_ids=PARTIAL_IDS)
    np.testing.assert_array_equal(down.cell_ids_out, [10, 11, 300])
    up = HealPixUp(
        level - 1, paired_down=down, up_norm=up_norm,
        device="cpu", dtype=torch.float64,
    )

    actual, ids = up(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(ids, PARTIAL_IDS)       # caller's order
    np.testing.assert_allclose(actual, expected)


def test_up_rejects_maxpool_paired_down():
    down = HealPixDown(3, mode="maxpool", device="cpu")
    with pytest.raises(ValueError, match="paired_down must use mode='smooth', 'smooth_conservative' or 'mean'"):
        HealPixUp(2, paired_down=down, device="cpu")


# ---------------------------------------------------------------------------
# HealPixResampler / HealPixDecomp wiring
# ---------------------------------------------------------------------------

def test_resampler_down_mode_mean():
    x = _temperature_like(5)
    expected = x.reshape(-1, 64).mean(-1)                  # level 5 -> 2

    op = HealPixResampler(5, 2, down_mode="mean", device="cpu", dtype=torch.float64)
    actual, _ = op(x)
    np.testing.assert_allclose(actual, expected)
    assert actual.mean() == pytest.approx(x.mean(), rel=MEAN_REL)
    assert "down_mode='mean'" in repr(op)

    one_shot, _ = resample_healpix(
        x, in_level=5, out_level=2, down_mode="mean", device="cpu",
        dtype=torch.float64,
    )
    np.testing.assert_array_equal(one_shot, actual)


def test_resampler_default_is_unchanged():
    op = HealPixResampler(5, 4, device="cpu")
    assert op.layers[0].mode == "smooth"
    assert "down_mode" not in repr(op)


@pytest.mark.parametrize("cls", [HealPixResampler, HealPixDecomp])
def test_invalid_down_mode(cls):
    args = (5, 4) if cls is HealPixResampler else (5,)
    with pytest.raises(ValueError, match="down_mode must be 'smooth', 'smooth_conservative' or 'mean'"):
        cls(*args, down_mode="maxpool", device="cpu")


def test_resampler_down_mode_mean_skips_nan_per_step():
    x = _temperature_like(3)
    x[4 * 7] = np.nan
    actual, _ = HealPixResampler(
        3, 2, down_mode="mean", device="cpu", dtype=torch.float64
    )(x)
    np.testing.assert_allclose(actual[7], np.nanmean(x[28:32]))


def test_resampler_partial_sky_several_steps_is_mean_of_means():
    """Documented limit: incomplete parents are averaged step by step."""
    # level 2 -> 0: level-1 cell 0 has its 4 children (0..3), level-1 cell 1
    # only one (4); both lie in level-0 cell 0
    cell_ids = np.array([3, 0, 4, 2, 1])
    x = np.array([10.0, 10.0, 30.0, 10.0, 10.0])          # cell 4 -> 30

    actual, _ = HealPixResampler(
        2, 0, in_cell_ids=cell_ids, out_cell_ids=[0], down_mode="mean",
        device="cpu", dtype=torch.float64,
    )(x)

    np.testing.assert_allclose(actual, [(10.0 + 30.0) / 2])  # mean of means
    assert actual[0] != pytest.approx(x.mean())             # not 14.0


def test_decomp_down_mode_mean_full_sphere():
    level = 4
    x = _temperature_like(level)
    dec = HealPixDecomp(level, down_mode="mean", device="cpu", dtype=torch.float64)
    pyr = dec.compute(x)

    np.testing.assert_allclose(dec.invert(pyr), x)
    assert pyr.coarse.mean() == pytest.approx(x.mean(), rel=MEAN_REL)
    assert "filter='mean of 4 NESTED children'" in repr(dec)


@pytest.mark.parametrize("scale", [0, 1, 2, 3])
def test_decomp_down_mode_mean_detail_sums_to_zero(scale):
    """Haar-like: each detail band sums to zero over every 4 children."""
    level = 4
    x = _temperature_like(level)
    pyr = HealPixDecomp(level, down_mode="mean", device="cpu", dtype=torch.float64).compute(x)

    actual = pyr.details[scale].reshape(-1, 4).sum(-1)
    np.testing.assert_allclose(actual, 0.0, atol=ZERO_ATOL)


def test_decomp_default_is_unchanged():
    dec = HealPixDecomp(2, device="cpu")
    assert dec.down_layers[0].mode == "smooth"
    assert "filter='symmetric Gaussian'" in repr(dec)


def test_decomp_down_mode_mean_partial_sky():
    level = 4
    x = _temperature_like(level)[PARTIAL_IDS]
    dec = HealPixDecomp(
        level, cell_ids=PARTIAL_IDS, Jmax=1, down_mode="mean",
        device="cpu", dtype=torch.float64,
    )
    pyr = dec.compute(x)

    np.testing.assert_allclose(dec.invert(pyr), x)
    coverage = dec.down_layers[0].coverage
    np.testing.assert_array_equal(coverage, [1.0, 0.5, 0.5])
    # the coverage-weighted mean of the coarse band is the input mean
    weighted = (pyr.coarse * coverage).sum() / coverage.sum()
    assert weighted == pytest.approx(x.mean(), rel=MEAN_REL)
    # the detail band (caller's cell order) sums to zero over the children
    # present of every parent
    actual = _sum_per_parent(pyr.details[0], PARTIAL_IDS)
    np.testing.assert_allclose(actual, 0.0, atol=ZERO_ATOL)
