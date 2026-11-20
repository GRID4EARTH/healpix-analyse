"""Tests for the full-disc normalisation, the reusable filter operator and the
sub-cell Gaussian widths of ``radial_filter`` / ``gaussian_filter``.

The reference values come from a brute-force implementation that is
independent of the library geometry: a large patch of real HEALPix cells (the
"universe") with ``pyproj`` distances, of which a smaller patch is the
processing domain.  Cells of the universe outside the domain play the role of
the cells outside the domain in the full-disc normalisation.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch
from healpix_geo import nested
from pyproj import Geod

from healpix_analyse import (
    apply_filter_operator,
    gaussian_filter,
    gaussian_filter_operator,
    radial_filter,
    radial_filter_operator,
)
from healpix_analyse._neighbourhood import build_outside_domain_distances
from healpix_analyse.radial_filter import (
    _cell_size_m,
    _clear_filter_caches,
    _REFERENCE_SIGMA_CELLS,
)

_WGS84 = Geod(ellps="WGS84")
LEVEL = 8
CELL_M = _cell_size_m(LEVEL)


def _patch(k: int, level: int = LEVEL) -> np.ndarray:
    centre = nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), level, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), level, k)
    )[0]
    cells = ring[ring >= 0].astype(np.uint64)
    centre_id = np.uint64(centre[0])
    return np.concatenate(([centre_id], cells[cells != centre_id]))


def _lonlat(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = nested.healpix_to_lonlat(cells, LEVEL, ellipsoid="WGS84")
    return np.asarray(lon), np.asarray(lat)


def _distances(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Distance matrix (len(rows), len(cols)) between cell centres."""
    lon_r, lat_r = _lonlat(rows)
    lon_c, lat_c = _lonlat(cols)
    n, m = rows.size, cols.size
    d = _WGS84.inv(
        np.repeat(lon_r, m),
        np.repeat(lat_r, m),
        np.tile(lon_c, n),
        np.tile(lat_c, n),
    )[2].reshape(n, m)
    d[rows[:, None] == cols[None, :]] = 0.0
    return d


DOMAIN = _patch(9)  # 361 cells
UNIVERSE = _patch(20)  # 1681 cells; holds the discs of DOMAIN for 4 sigma <= 6 cells
UNIVERSE = np.concatenate(
    (DOMAIN, UNIVERSE[~np.isin(UNIVERSE, DOMAIN)])
)
D_DD = _distances(DOMAIN, DOMAIN)
D_DU = _distances(DOMAIN, UNIVERSE)
IN_DOMAIN = np.arange(UNIVERSE.size) < DOMAIN.size


def _reference(
    values: np.ndarray,
    sigma_m: float,
    normalization: str,
    truncate: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force sampled Gaussian: (filtered values, row sums)."""
    w = np.where(D_DU <= truncate * sigma_m, np.exp(-0.5 * (D_DU / sigma_m) ** 2), 0.0)
    w_in = w[:, IN_DOMAIN]
    total = w.sum(1) if normalization == "full_disc" else w_in.sum(1)
    return (w_in @ values) / total, w_in.sum(1) / total


def _interior(sigma_m: float, truncate: float = 4.0) -> np.ndarray:
    """Domain cells whose whole disc is inside the domain."""
    return ~np.any((D_DU <= truncate * sigma_m) & ~IN_DOMAIN[None, :], axis=1)


@pytest.fixture(scope="module")
def field() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.random(DOMAIN.size) + 1.0


# ---------------------------------------------------------------------------
# Default behaviour and normalisation
# ---------------------------------------------------------------------------


def test_default_gaussian_filter_matches_bruteforce(field):
    sigma = 1.5 * CELL_M
    expected, _ = _reference(field, sigma, "present")
    result = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=sigma)
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    explicit = gaussian_filter(
        field, DOMAIN, LEVEL, sigma_m=sigma,
        normalization="present", width_rule="sampled",
    )
    np.testing.assert_array_equal(result, explicit)


def test_present_keeps_constant_field_up_to_the_edge():
    ones = np.ones(DOMAIN.size)
    for sigma in (0.5 * CELL_M, 2.0 * CELL_M):
        out = gaussian_filter(ones, DOMAIN, LEVEL, sigma_m=sigma, normalization="present")
        np.testing.assert_allclose(out, 1.0, rtol=1e-12)


@pytest.mark.parametrize("sigma_cells", [0.8, 1.5])
def test_full_disc_row_sums_and_values(field, sigma_cells):
    sigma = sigma_cells * CELL_M
    expected, expected_sums = _reference(field, sigma, "full_disc")
    interior = _interior(sigma)
    assert interior.any() and (~interior).any()

    ones = gaussian_filter(
        np.ones(DOMAIN.size), DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc"
    )
    np.testing.assert_allclose(ones[interior], 1.0, rtol=1e-12)
    assert np.all(ones[~interior] < 1.0)
    np.testing.assert_allclose(ones, expected_sums, rtol=1e-10)

    result = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc")
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    op = gaussian_filter_operator(
        DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc"
    )
    row_sums = np.add.reduceat(op.weights, op.row_offsets[:-1])
    np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-10)


def test_full_disc_with_domain_subset_of_cell_ids(field):
    """Cells of ``cell_ids`` outside ``domain`` count as outside the domain."""
    sigma = 1.0 * CELL_M
    inner = DOMAIN[:200]
    out_a = gaussian_filter(
        field, DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc", domain=inner
    )
    out_b = gaussian_filter(
        field[:200], inner, LEVEL, sigma_m=sigma, normalization="full_disc"
    )
    np.testing.assert_allclose(out_a, out_b, rtol=1e-12)


def test_radial_filter_full_disc_matches_bruteforce(field):
    radius = 3.0 * CELL_M

    def kernel(d):
        return 1.0 / (1.0 + d / CELL_M)

    w = np.where(D_DU <= radius, 1.0 / (1.0 + D_DU / CELL_M), 0.0)
    expected = (w[:, IN_DOMAIN] @ field) / w.sum(1)
    result = radial_filter(
        field, DOMAIN, LEVEL, radius_m=radius, kernel=kernel, normalization="full_disc"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)

    present = radial_filter(
        field, DOMAIN, LEVEL, radius_m=radius, kernel=kernel, normalization="present"
    )
    np.testing.assert_allclose(
        present,
        radial_filter(field, DOMAIN, LEVEL, radius_m=radius, kernel=kernel),
        rtol=1e-12,
    )


def test_normalization_keyword_validation(field):
    with pytest.raises(ValueError, match="normalization"):
        radial_filter(
            field, DOMAIN, LEVEL, radius_m=CELL_M, kernel=lambda d: 1.0 + 0 * d,
            normalization="everything",
        )
    with pytest.raises(ValueError, match="normalize=True"):
        radial_filter(
            field, DOMAIN, LEVEL, radius_m=CELL_M, kernel=lambda d: 1.0 + 0 * d,
            normalize=False, normalization="full_disc",
        )
    with pytest.raises(ValueError, match="width_rule"):
        gaussian_filter(field, DOMAIN, LEVEL, sigma_m=CELL_M, width_rule="exact")
    with pytest.raises(ValueError, match="normalization"):
        gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M, normalization="x")


def test_full_disc_rejects_zero_weight_sum(field):
    with pytest.raises(ValueError, match="sum to zero"):
        radial_filter(
            field, DOMAIN, LEVEL, radius_m=2 * CELL_M,
            kernel=lambda d: 0.0 * d, normalization="full_disc",
        )


# ---------------------------------------------------------------------------
# NaN rule
# ---------------------------------------------------------------------------


def test_full_disc_nan_counts_as_zero(field):
    sigma = 1.2 * CELL_M
    holed = field.copy()
    holed[::7] = np.nan
    zeroed = np.where(np.isnan(holed), 0.0, holed)
    for rule in ("sampled", "matched"):
        out = gaussian_filter(
            holed, DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc", width_rule=rule
        )
        np.testing.assert_allclose(
            out,
            gaussian_filter(
                zeroed, DOMAIN, LEVEL, sigma_m=sigma,
                normalization="full_disc", width_rule=rule,
            ),
            rtol=1e-12,
        )
        assert not np.any(np.isnan(out))


def test_present_nan_is_renormalised(field):
    sigma = 1.2 * CELL_M
    holed = field.copy()
    holed[::7] = np.nan
    valid = ~np.isnan(holed)
    w = np.where(D_DD <= 4 * sigma, np.exp(-0.5 * (D_DD / sigma) ** 2), 0.0)
    w = w * valid[None, :]
    expected = (w @ np.where(valid, holed, 0.0)) / w.sum(1)
    out = gaussian_filter(holed, DOMAIN, LEVEL, sigma_m=sigma)
    np.testing.assert_allclose(out, expected, rtol=1e-10)
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=sigma)
    np.testing.assert_allclose(
        apply_filter_operator(holed, *op, renormalize_missing=True), out, rtol=1e-10
    )
    # Without renormalisation a NaN counts as zero.
    np.testing.assert_allclose(
        apply_filter_operator(holed, *op),
        apply_filter_operator(np.where(valid, holed, 0.0), *op),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Operator arrays
# ---------------------------------------------------------------------------

OPTIONS = [
    {"normalization": "present", "width_rule": "sampled"},
    {"normalization": "full_disc", "width_rule": "sampled"},
    {"normalization": "present", "width_rule": "matched"},
    {"normalization": "full_disc", "width_rule": "matched"},
]


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("sigma_cells", [0.3, 1.5])
def test_operator_equals_filter(field, options, sigma_cells):
    sigma = sigma_cells * CELL_M
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=sigma, **options)
    assert op.row_offsets.shape == (DOMAIN.size + 1,)
    assert op.column_indices.shape == op.weights.shape
    assert op.column_indices.min() >= 0 and op.column_indices.max() < DOMAIN.size

    expected = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=sigma, **options)
    np.testing.assert_allclose(apply_filter_operator(field, *op), expected, rtol=1e-12)

    stack = np.stack([field, 2.0 * field, field[::-1].copy()])
    expected_stack = gaussian_filter(stack, DOMAIN, LEVEL, sigma_m=sigma, **options)
    out = apply_filter_operator(stack, *op)
    assert out.shape == stack.shape
    np.testing.assert_allclose(out, expected_stack, rtol=1e-12)
    np.testing.assert_allclose(out[1], 2.0 * out[0], rtol=1e-12)

    cube = np.stack([stack, stack + 1.0])
    assert apply_filter_operator(cube, *op).shape == cube.shape

    tensor = torch.as_tensor(stack)
    out_t = apply_filter_operator(tensor, *op)
    assert isinstance(out_t, torch.Tensor)
    np.testing.assert_allclose(out_t.numpy(), out, rtol=1e-12)


def test_operator_with_domain_subset_and_domain_order(field):
    sigma = 1.0 * CELL_M
    domain = DOMAIN[10:150][::-1].copy()
    op = gaussian_filter_operator(
        DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc", domain=domain
    )
    assert op.row_offsets.size == domain.size + 1
    expected = gaussian_filter(
        field, DOMAIN, LEVEL, sigma_m=sigma, normalization="full_disc", domain=domain
    )
    np.testing.assert_allclose(apply_filter_operator(field, *op), expected, rtol=1e-12)


@pytest.mark.parametrize("normalization", ["present", "full_disc"])
def test_radial_operator_equals_filter(field, normalization):
    radius = 2.5 * CELL_M

    def kernel(d):
        return np.exp(-d / CELL_M)

    op = radial_filter_operator(
        DOMAIN, LEVEL, radius_m=radius, kernel=kernel, normalization=normalization
    )
    expected = radial_filter(
        field, DOMAIN, LEVEL, radius_m=radius, kernel=kernel, normalization=normalization
    )
    np.testing.assert_allclose(apply_filter_operator(field, *op), expected, rtol=1e-12)


def test_apply_filter_operator_validates_flag(field):
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M)
    with pytest.raises(TypeError, match="renormalize_missing"):
        apply_filter_operator(field, *op, renormalize_missing="yes")


# ---------------------------------------------------------------------------
# Widths below about one cell
# ---------------------------------------------------------------------------

CENTRE = 0  # _patch puts the central cell first; it is far from the edge
WIDE = _patch(14)  # 841 cells: the disc of the centre (8 cells) stays inside
D_WW = _distances(WIDE, WIDE)


def _response_width_cells(op, source: int = CENTRE) -> float:
    """One-axis RMS width of the response to one source cell, in cells."""
    impulse = np.zeros(WIDE.size)
    impulse[source] = 1.0
    r = apply_filter_operator(impulse, *op)
    return float(np.sqrt((r * D_WW[source] ** 2).sum() / r.sum() / 2.0) / CELL_M)


def _row_width_cells(op, row: int = CENTRE) -> float:
    """One-axis RMS width of one row of the operator, in cells."""
    s, e = op.row_offsets[row], op.row_offsets[row + 1]
    w = op.weights[s:e]
    d = D_WW[row, op.column_indices[s:e]]
    return float(np.sqrt((w * d**2).sum() / w.sum() / 2.0) / CELL_M)


@pytest.mark.parametrize("sigma_cells", [0.1, 0.3, 0.5, 1.0, 2.0])
def test_matched_width_equals_requested(sigma_cells):
    op = gaussian_filter_operator(
        WIDE, LEVEL, sigma_m=sigma_cells * CELL_M,
        normalization="full_disc", width_rule="matched",
    )
    # Every row has the requested second moment by construction ...
    assert _row_width_cells(op) == pytest.approx(sigma_cells, rel=1e-6)
    # ... and so does, to within a few per cent, the response to one cell.
    assert _response_width_cells(op) == pytest.approx(sigma_cells, abs=0.02 * max(1.0, sigma_cells))


def test_sampled_width_is_narrower_than_requested_below_one_cell():
    sigma = 0.3
    op = gaussian_filter_operator(WIDE, LEVEL, sigma_m=sigma * CELL_M)
    assert _response_width_cells(op) < 0.5 * sigma
    sigma = 2.0
    op = gaussian_filter_operator(WIDE, LEVEL, sigma_m=sigma * CELL_M)
    assert _response_width_cells(op) == pytest.approx(sigma, rel=0.02)


def test_matched_tends_to_identity():
    for sigma_cells in (1e-2, 1e-4):
        op = gaussian_filter_operator(
            WIDE, LEVEL, sigma_m=sigma_cells * CELL_M,
            normalization="full_disc", width_rule="matched",
        )
        self_weight = op.weights[op.column_indices == np.repeat(
            np.arange(WIDE.size), np.diff(op.row_offsets))]
        assert np.all(self_weight > 1.0 - 2.0 * sigma_cells**2 / 0.3**2)
    values = np.random.default_rng(0).random(WIDE.size)
    out = gaussian_filter(
        values, WIDE, LEVEL, sigma_m=1e-4 * CELL_M, width_rule="matched"
    )
    np.testing.assert_allclose(out, values, atol=1e-6)


def test_matched_width_is_monotone_across_the_switch():
    grid = np.concatenate((np.linspace(0.02, 1.0, 50), [1.25, 1.5, 2.0]))
    widths = []
    for sigma_cells in grid:
        op = gaussian_filter_operator(
            WIDE, LEVEL, sigma_m=sigma_cells * CELL_M,
            normalization="full_disc", width_rule="matched",
        )
        widths.append(_response_width_cells(op))
    widths = np.array(widths)
    assert np.all(np.diff(widths) > 0.0)
    # No jump at the switch: successive steps stay close to the grid step.
    assert np.all(np.abs(np.diff(widths) - np.diff(grid)) < 0.03)


def test_matched_with_large_sigma_is_close_to_sampled(field):
    """Above two cells the sampled Gaussian already has the right width."""
    sigma = 2.0 * CELL_M
    a = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=sigma)
    b = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=sigma, width_rule="matched")
    np.testing.assert_allclose(a, b, rtol=0.01)


def test_matched_rejects_short_truncation(field):
    with pytest.raises(ValueError, match="truncate"):
        gaussian_filter(
            field, DOMAIN, LEVEL, sigma_m=CELL_M, truncate=2.0, width_rule="matched"
        )


def test_matched_tiny_sigma_emits_no_warning(field):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaussian_filter(field, DOMAIN, LEVEL, sigma_m=1e-8 * CELL_M, width_rule="matched")


@pytest.mark.parametrize("options", OPTIONS)
def test_torch_autograd_through_operator(field, options):
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=0.5 * CELL_M, **options)
    x = torch.as_tensor(field, dtype=torch.float32).requires_grad_()
    out = apply_filter_operator(x, *op)
    assert out.dtype == torch.float32
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_empty_domain(field):
    empty = DOMAIN[:0]
    for options in OPTIONS:
        op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M, domain=empty, **options)
        assert op.row_offsets.tolist() == [0]
        out = gaussian_filter(field, DOMAIN, LEVEL, sigma_m=CELL_M, domain=empty, **options)
        assert out.shape == (0,)


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------

NAN_FIELD = np.random.default_rng(8).random((3, DOMAIN.size)) + 1.0
NAN_FIELD[0, ::5] = np.nan
NAN_FIELD[2, 10:40] = np.nan


def _zero_kernel(d):
    return 0.0 * d


def _exp_kernel(d):
    return np.exp(-d / CELL_M)


def test_present_operator_with_zero_weight_row_matches_the_filter(field):
    """``radial_filter(normalize=True)`` gives NaN for a zero-sum row; so does
    the operator applied with ``renormalize_missing=True`` (no error)."""
    kwargs = dict(radius_m=2 * CELL_M, kernel=_zero_kernel)
    expected = radial_filter(field, DOMAIN, LEVEL, **kwargs)
    assert np.all(np.isnan(expected))
    op = radial_filter_operator(DOMAIN, LEVEL, normalization="present", **kwargs)
    out = apply_filter_operator(field, *op, renormalize_missing=True)
    np.testing.assert_array_equal(out, expected)


def test_operators_have_no_normalize_argument(field):
    with pytest.raises(TypeError, match="normalize"):
        radial_filter_operator(
            DOMAIN, LEVEL, radius_m=CELL_M, kernel=_exp_kernel, normalize=False
        )
    with pytest.raises(TypeError, match="normalize"):
        gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M, normalize=False)


@pytest.mark.parametrize("use_domain", [False, True])
def test_operator_n_columns_is_the_length_of_cell_ids(use_domain):
    domain = DOMAIN[:50] if use_domain else None
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M, domain=domain)
    assert op.n_columns == DOMAIN.size
    assert op.row_offsets.size - 1 == (50 if use_domain else DOMAIN.size)
    rop = radial_filter_operator(
        DOMAIN, LEVEL, radius_m=CELL_M, kernel=_exp_kernel, domain=domain
    )
    assert rop.n_columns == DOMAIN.size


def test_apply_filter_operator_checks_the_signal_length(field):
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=CELL_M)
    longer = np.concatenate((field, field[:5]))
    with pytest.raises(ValueError, match="final axis"):
        apply_filter_operator(longer, *op)
    with pytest.raises(ValueError, match="final axis"):
        apply_filter_operator(longer[:-10], *op)
    with pytest.raises(ValueError, match="final axis"):
        apply_filter_operator(torch.as_tensor(longer), *op)
    with pytest.raises(ValueError, match="final axis"):
        apply_filter_operator(
            longer, op.row_offsets, op.column_indices, op.weights, n_columns=op.n_columns
        )
    # The three-array call keeps working, without the length check.
    three = apply_filter_operator(field, op.row_offsets, op.column_indices, op.weights)
    np.testing.assert_array_equal(three, apply_filter_operator(field, *op))
    unchecked = apply_filter_operator(
        longer, op.row_offsets, op.column_indices, op.weights
    )
    np.testing.assert_array_equal(unchecked, three)


# -- default path against a reference made with the code before this change --

_REFERENCE = np.load(Path(__file__).parent / "data" / "default_path_reference.npz")
_REF_DOMAIN = _REFERENCE["cells"]
_REF_VALUES = _REFERENCE["values"]  # contains NaN
_REF_SUBSET = _REF_DOMAIN[10:200][::-1].copy()


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("gauss0", lambda: gaussian_filter(_REF_VALUES, _REF_DOMAIN, LEVEL, sigma_m=0.5 * 12700.0)),
        ("gauss1", lambda: gaussian_filter(_REF_VALUES, _REF_DOMAIN, LEVEL, sigma_m=1.5 * 12700.0)),
        (
            "gauss_domain0",
            lambda: gaussian_filter(
                _REF_VALUES, _REF_DOMAIN, LEVEL, sigma_m=0.5 * 12700.0, domain=_REF_SUBSET
            ),
        ),
        (
            "gauss_domain1",
            lambda: gaussian_filter(
                _REF_VALUES, _REF_DOMAIN, LEVEL, sigma_m=1.5 * 12700.0, domain=_REF_SUBSET
            ),
        ),
        (
            "radial",
            lambda: radial_filter(
                _REF_VALUES, _REF_DOMAIN, LEVEL, radius_m=40000.0,
                kernel=lambda d: np.exp(-d / 12000.0),
            ),
        ),
        (
            "radial_raw",
            lambda: radial_filter(
                _REF_VALUES, _REF_DOMAIN, LEVEL, radius_m=40000.0,
                kernel=lambda d: np.exp(-d / 12000.0), normalize=False,
            ),
        ),
    ],
)
def test_default_path_reproduces_the_reference_with_nan(name, call):
    """``tests/data/default_path_reference.npz`` was computed with the code of
    the commit before this feature (NaN in the input, with and without a
    reordered domain subset).  The comparison is to 1e-13 rather than bitwise so
    that it does not depend on the platform's ``exp``."""
    assert np.isnan(_REF_VALUES).any()
    np.testing.assert_allclose(call(), _REFERENCE[name], rtol=1e-13, atol=0, equal_nan=True)


# -- matched widths: several rows, and the edge ------------------------------

_RANK = np.argsort(D_WW[0])  # WIDE cells by distance from the centre cell
_ROWS = [int(_RANK[i]) for i in (0, 1, 7, 25, 40)]


@pytest.mark.parametrize("sigma_cells", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("row", _ROWS)
def test_matched_row_width_for_several_rows(sigma_cells, row):
    op = gaussian_filter_operator(
        WIDE, LEVEL, sigma_m=sigma_cells * CELL_M,
        normalization="full_disc", width_rule="matched",
    )
    assert _row_width_cells(op, row) == pytest.approx(sigma_cells, rel=1e-6)


@pytest.mark.parametrize("normalization", ["present", "full_disc"])
def test_matched_edge_rows_differ_from_the_requested_width(normalization):
    """Rows whose disc leaves the domain are renormalised (present) or
    truncated (full_disc): their width is not ``sigma_m``."""
    sigma = 1.0
    interior = _interior(sigma * CELL_M)
    op = gaussian_filter_operator(
        DOMAIN, LEVEL, sigma_m=sigma * CELL_M,
        normalization=normalization, width_rule="matched",
    )
    widths = np.array([
        np.sqrt(
            (op.weights[a:b] * D_DD[i, op.column_indices[a:b]] ** 2).sum()
            / op.weights[a:b].sum() / 2.0
        ) / CELL_M
        for i, (a, b) in enumerate(zip(op.row_offsets[:-1], op.row_offsets[1:]))
    ])
    assert interior.any() and (~interior).any()
    np.testing.assert_allclose(widths[interior], sigma, rtol=1e-6)
    # Rows that lose only a thin tail are barely affected; the corner rows
    # lose a lot (measured: 0.17 cell narrower).
    assert np.all(widths[~interior] < sigma)
    assert (sigma - widths[~interior]).max() > 0.1


# -- radial_filter with full_disc: Torch, leading dimensions, NaN -------------


@pytest.mark.parametrize("kind", ["numpy", "torch"])
@pytest.mark.parametrize("shape", ["flat", "stack"])
def test_radial_full_disc_torch_leading_dims_and_nan(kind, shape):
    radius = 2.5 * CELL_M
    values = NAN_FIELD[0] if shape == "flat" else NAN_FIELD
    zeroed = np.where(np.isnan(values), 0.0, values)
    # NaN counts as zero in the full-disc rule: same as the zero-filled signal.
    expected = radial_filter(
        zeroed, DOMAIN, LEVEL, radius_m=radius, kernel=_exp_kernel,
        normalization="full_disc",
    )
    given = torch.as_tensor(values) if kind == "torch" else values
    out = radial_filter(
        given, DOMAIN, LEVEL, radius_m=radius, kernel=_exp_kernel,
        normalization="full_disc",
    )
    assert isinstance(out, torch.Tensor) == (kind == "torch")
    out = np.asarray(out)
    assert out.shape == values.shape
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# -- renormalize_missing with NaN and an all-NaN row --------------------------


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_renormalize_missing_with_nan_and_an_all_nan_row(kind):
    op = gaussian_filter_operator(DOMAIN, LEVEL, sigma_m=1.0 * CELL_M)
    row = 100
    a, b = op.row_offsets[row], op.row_offsets[row + 1]
    values = np.random.default_rng(2).random((2, DOMAIN.size)) + 1.0
    values[:, op.column_indices[a:b]] = np.nan  # every cell of the row is NaN
    values[1, ::9] = np.nan
    given = torch.as_tensor(values) if kind == "torch" else values

    renormalised = np.asarray(apply_filter_operator(given, *op, renormalize_missing=True))
    plain = np.asarray(apply_filter_operator(given, *op))
    assert np.isnan(renormalised[:, row]).all()
    assert (plain[:, row] == 0.0).all()  # a NaN counts as zero without it
    others = np.arange(DOMAIN.size) != row
    # The filter agrees with the renormalised operator, rows with no valid
    # neighbour included.
    expected = gaussian_filter(values, DOMAIN, LEVEL, sigma_m=1.0 * CELL_M)
    np.testing.assert_allclose(renormalised, expected, rtol=1e-12, equal_nan=True)
    assert np.isfinite(renormalised[0][others]).sum() > 300


# -- no stale result when the options change on the same domain ---------------

_CONFIGS = [
    {"sigma_cells": 1.0, "truncate": 4.0, "normalization": "present", "width_rule": "sampled"},
    {"sigma_cells": 1.0, "truncate": 3.0, "normalization": "present", "width_rule": "sampled"},
    {"sigma_cells": 1.0, "truncate": 4.0, "normalization": "full_disc", "width_rule": "sampled"},
    {"sigma_cells": 0.4, "truncate": 5.0, "normalization": "full_disc", "width_rule": "matched"},
    {"sigma_cells": 1.0, "truncate": 3.0, "normalization": "present", "width_rule": "matched"},
]


def _run(config, values):
    options = {k: v for k, v in config.items() if k != "sigma_cells"}
    return gaussian_filter(
        values, DOMAIN, LEVEL, sigma_m=config["sigma_cells"] * CELL_M, **options
    )


@pytest.mark.parametrize(("first", "second"), list(permutations(range(len(_CONFIGS)), 2)))
def test_changing_options_on_the_same_domain_gives_no_stale_result(field, first, second):
    _clear_filter_caches()
    fresh = _run(_CONFIGS[second], field)
    _clear_filter_caches()
    _run(_CONFIGS[first], field)
    np.testing.assert_array_equal(_run(_CONFIGS[second], field), fresh)


# -- column sums of the matched full-disc operator ----------------------------


def _interior_columns(cells, level, sigma_cells, truncate=4.0):
    """Cells whose whole kernel disc lies inside the domain."""
    cell = _cell_size_m(level)
    radius = truncate * max(sigma_cells, _REFERENCE_SIGMA_CELLS) * cell
    offsets, _ = build_outside_domain_distances(cells, radius, level)
    return np.diff(offsets) == 0


@pytest.mark.parametrize("sigma_cells", [0.3, 1.0, 2.0])
@pytest.mark.parametrize("level", [8, 10])
def test_matched_full_disc_column_sums_are_close_to_one(level, sigma_cells):
    """Interior columns (those whose disc lies inside the domain) sum to 1
    within 2e-3 (measured: up to about 1.1e-3, at 0.3 cell); they are not
    exactly 1 by construction.  Rows of the interior sum to 1 to rounding."""
    cells = _patch(30, level)
    op = gaussian_filter_operator(
        cells, level, sigma_m=sigma_cells * _cell_size_m(level),
        normalization="full_disc", width_rule="matched",
    )
    interior = _interior_columns(cells, level, sigma_cells)
    assert interior.sum() > 500
    columns = np.bincount(op.column_indices, weights=op.weights, minlength=cells.size)
    assert np.abs(columns[interior] - 1.0).max() < 2e-3
    rows = np.add.reduceat(op.weights, op.row_offsets[:-1])
    np.testing.assert_allclose(rows[interior], 1.0, rtol=0, atol=1e-12)
