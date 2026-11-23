"""Tests for ``HealPixDown(mode="smooth_conservative")``.

The mode is ``smooth`` followed by one additive offset per batch row,
``delta = mean(x) - mean_cov(M x)``, so that the (coverage-weighted) mean of
the output equals the mean of the input, to round-off.  A constant field is
preserved because ``delta`` is then zero up to round-off.

Tolerances: float64 fields of ~300; means of a few thousand values are exact
to ~1e-13 relative, so ``rtol=1e-10`` is used for means; the stored reference
of the default mode is compared at ``rtol=1e-10`` too (machines differ at
~1e-12).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from healpix_geo import nested

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.down import HealPixDown
from healpix_analyse.resample import HealPixResampler, resample_healpix
from healpix_analyse.up import HealPixUp

MEAN_RTOL = 1e-10
REFERENCE = Path(__file__).parent / "data" / "down_smooth_reference.npz"
MODE = "smooth_conservative"


def _field(level: int) -> np.ndarray:
    n = 12 * 4**level
    lon, lat = nested.healpix_to_lonlat(np.arange(n), level, ellipsoid="sphere")
    lo, la = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    return (
        288.0
        - 45.0 * np.sin(la) ** 2
        + 3.0 * np.sin(3 * lo) * np.cos(la) ** 2
        + 2.0 * np.cos(2 * lo + 1) * np.sin(2 * la)
    )


def _down(level, mode=MODE, **kwargs):
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("dtype", torch.float64)
    kwargs.setdefault("ellipsoid", "sphere")
    return HealPixDown(level, mode=mode, **kwargs)


def test_down_module_does_not_use_scipy():
    # (the package imports scipy elsewhere, e.g. validation.py, so sys.modules
    # cannot tell: check the source of the module that implements the mode)
    import ast
    import healpix_analyse.down as down_module

    tree = ast.parse(Path(down_module.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not [n for n in names if n.split(".")[0] == "scipy"]
    assert "scipy" not in Path(down_module.__file__).read_text()


@pytest.mark.parametrize("level", [2, 4, 6])
def test_one_step_conserves_global_mean(level):
    x = _field(level)
    y, _ = _down(level)(x)
    assert y.mean() == pytest.approx(x.mean(), rel=MEAN_RTOL)
    ys, _ = _down(level, "smooth")(x)
    assert abs(ys.mean() - x.mean()) > 1e-6          # smooth does not


def test_chain_to_level_zero_conserves_global_mean():
    x = _field(6)
    m0 = x.mean()
    smooth = x.copy()
    for level in range(6, 0, -1):
        x = _down(level)(x)[0]
        assert x.mean() == pytest.approx(m0, rel=MEAN_RTOL)
        smooth = _down(level, "smooth")(smooth)[0]
    assert abs(smooth.mean() - m0) > 0.1             # smooth drifts ~0.4 K


def test_constant_field_is_preserved():
    y, _ = _down(4)(np.full(12 * 4**4, 273.15))
    np.testing.assert_allclose(y, 273.15, rtol=1e-12)


def test_equals_smooth_plus_uniform_offset():
    x = _field(4)
    y, _ = _down(4)(x)
    ys, _ = _down(4, "smooth")(x)
    d = y - ys
    np.testing.assert_allclose(d, d[0], atol=1e-12)   # a single number
    assert abs(d[0]) < 0.2                            # drift of one step


def test_information_crosses_parent_boundaries():
    level = 5
    x = np.zeros(12 * 4**level)
    x[4 * 1000 + 1] = 1.0
    y, _ = _down(level, "smooth")(x)
    yc, _ = _down(level)(x)
    assert np.count_nonzero(np.abs(y) > 1e-6) > 1
    # same smoothing: the response differs from smooth by a uniform offset
    d = yc - y
    np.testing.assert_allclose(d, d[0], atol=1e-12)


def test_batch_has_one_offset_per_row():
    x = np.stack([_field(4), _field(4) + 10.0, 2.0 * _field(4)])
    y, _ = _down(4)(x)
    assert y.shape == (3, 12 * 4**3)
    np.testing.assert_allclose(y.mean(-1), x.mean(-1), rtol=MEAN_RTOL)


def test_is_differentiable():
    x = torch.tensor(_field(3), requires_grad=True)
    y, _ = _down(3)(x)
    y.mean().backward()
    # d mean(y) / d x_j = 1 / N_in for the mean-corrected output
    np.testing.assert_allclose(x.grad.numpy(), 1.0 / x.numel(), rtol=1e-8)


@pytest.mark.parametrize("ellipsoid", ["WGS84", "sphere"])
@pytest.mark.parametrize("level", [3, 4])
def test_default_matches_stored_reference_rtol_1e10(level, ellipsoid):
    ref = np.load(REFERENCE)[f"L{level}_{ellipsoid}"]
    y, _ = HealPixDown(
        level, ellipsoid=ellipsoid, device="cpu", dtype=torch.float64
    )(_field(level))
    np.testing.assert_allclose(y, ref, rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# partial sky and NaN
# ---------------------------------------------------------------------------

def _patch(level=4):
    p = nested.cone_coverage((10.0, 20.0), 25.0, level)
    return np.unique(np.asarray(getattr(p, "data", p), dtype=np.int64).reshape(-1))


def test_partial_sky_conserves_coverage_weighted_mean():
    level = 4
    ids = _patch(level)
    assert (np.unique(ids // 4).size * 4) != ids.size   # some incomplete parents
    x = _field(level)[ids]
    op = _down(level, cell_ids=ids)
    y, out_ids = op(x)
    cov = op.coverage
    assert 0 < cov.min() < 1 and cov.max() == 1
    assert (y * cov).sum() / cov.sum() == pytest.approx(x.mean(), rel=MEAN_RTOL)
    np.testing.assert_array_equal(out_ids, np.unique(ids // 4))


def test_partial_sky_unsorted_ids_and_constant():
    ids = _patch(3)
    perm = np.random.default_rng(0).permutation(ids.size)
    ids = ids[perm]
    x = _field(3)[ids]
    op = _down(3, cell_ids=ids)
    y, _ = op(x)
    assert (y * op.coverage).sum() / op.coverage.sum() == pytest.approx(
        x.mean(), rel=MEAN_RTOL
    )
    yc, _ = op(np.full(ids.size, 5.0))
    np.testing.assert_allclose(yc, 5.0, rtol=1e-12)


def test_full_sphere_coverage_is_one():
    np.testing.assert_array_equal(_down(3).coverage, np.ones(12 * 4**2))


def test_nan_stays_local_and_finite_cells_are_corrected():
    x = _field(4)
    x[100] = np.nan
    y, _ = _down(4)(x)
    ys, _ = _down(4, "smooth")(x)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(ys))
    assert 0 < np.isnan(y).sum() < 20
    fin = ~np.isnan(y)
    d = (y - ys)[fin]
    np.testing.assert_allclose(d, d[0], atol=1e-12)    # one finite offset


def test_all_nan_input_gives_all_nan():
    y, _ = _down(2)(np.full(12 * 16, np.nan))
    assert np.isnan(y).all()


# ---------------------------------------------------------------------------
# arguments and pass-through
# ---------------------------------------------------------------------------

def test_invalid_arguments():
    with pytest.raises(ValueError, match="weight_norm='l1'"):
        _down(3, weight_norm="l2")
    with pytest.raises(ValueError, match="skipna"):
        _down(3, skipna=True)


def test_sparse_matrix_is_the_gaussian_part():
    a = _down(3).sparse_matrix().to_dense()
    b = _down(3, "smooth").sparse_matrix().to_dense()
    torch.testing.assert_close(a, b)
    x = _field(3)
    y, _ = _down(3)(x)
    ys = (b.numpy() @ x)
    d = y - ys
    np.testing.assert_allclose(d, d[0], atol=1e-12)


def test_float32():
    x = _field(4).astype(np.float32)
    y, _ = _down(4, dtype=torch.float32)(x)
    assert y.dtype == np.float32
    assert abs(float(y.astype(np.float64).mean()) - float(x.astype(np.float64).mean())) < 1e-3


def test_up_paired_with_conservative_down():
    down = _down(4)
    up = HealPixUp(3, paired_down=down, ellipsoid="sphere",
                   device="cpu", dtype=torch.float64)
    y, ids = up(np.ones(12 * 4**3))
    np.testing.assert_allclose(y, 1.0, rtol=1e-9)
    assert ids.size == 12 * 4**4


def test_resampler_and_function_conserve_mean():
    x = _field(5)
    op = HealPixResampler(5, 2, down_mode=MODE, ellipsoid="sphere",
                          device="cpu", dtype=torch.float64)
    y = op(x)[0]
    assert y.mean() == pytest.approx(x.mean(), rel=MEAN_RTOL)
    assert f"down_mode='{MODE}'" in repr(op)
    y2 = resample_healpix(x, in_level=5, out_level=2, down_mode=MODE,
                          ellipsoid="sphere", device="cpu", dtype=torch.float64)[0]
    np.testing.assert_allclose(y2, y, rtol=1e-12)


def test_decomp_down_mode_conserves_coarse_mean_and_reconstructs():
    x = _field(4)
    dec = HealPixDecomp(4, down_mode=MODE, ellipsoid="sphere",
                        device="cpu", dtype=torch.float64)
    pyr = dec.compute(x)
    np.testing.assert_allclose(dec.invert(pyr), x, rtol=1e-9)
    assert pyr.coarse.mean() == pytest.approx(x.mean(), rel=MEAN_RTOL)
    assert "mean offset" in repr(dec)


def test_invalid_down_mode_message():
    with pytest.raises(ValueError, match="down_mode must be"):
        HealPixResampler(5, 4, down_mode="nope", device="cpu")


def test_uniform_shift_does_not_keep_positivity():
    """Documented behaviour: one uniform shift, so a sparse non-negative field
    can come out slightly negative.  Measures how large the effect is."""
    level = 4
    x = np.zeros(12 * 4**level)
    x[[10, 3000]] = 1.0                      # two isolated spikes
    y, _ = _down(level)(x)
    ys, _ = _down(level, "smooth")(x)
    assert ys.min() >= 0                     # smooth keeps positivity
    delta = float((y - ys).mean())
    assert delta < 0                         # the field has lost mean in M x
    assert y.min() < 0                       # ... and the shift crosses zero
    # the effect: every cell where smooth is 0 becomes delta (~ -2.4e-5 for
    # two unit spikes at level 4, against a peak of 0.072), 752 of 768 cells
    assert y.min() == pytest.approx(delta, rel=1e-12)
    assert int((y < 0).sum()) > 0.9 * y.size
    assert abs(y.min()) < 1e-3 * ys.max()
    assert y.mean() == pytest.approx(x.mean(), rel=MEAN_RTOL)
