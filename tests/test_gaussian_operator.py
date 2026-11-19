"""Tests for the full-disc normalisation, the reusable filter operator and the
sub-cell Gaussian widths of ``radial_filter`` / ``gaussian_filter``.

The reference values come from a brute-force implementation that is
independent of the library geometry: a large patch of real HEALPix cells (the
"universe") with ``pyproj`` distances, of which a smaller patch is the
processing domain.  Cells of the universe outside the domain play the role of
the cells outside the domain in the full-disc normalisation.
"""

from __future__ import annotations

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
from healpix_analyse.radial_filter import _cell_size_m

_WGS84 = Geod(ellps="WGS84")
LEVEL = 8
CELL_M = _cell_size_m(LEVEL)


def _patch(k: int) -> np.ndarray:
    centre = nested.lonlat_to_healpix(
        np.array([10.0]), np.array([45.0]), LEVEL, ellipsoid="WGS84"
    )
    ring = np.asarray(
        nested.kth_neighbourhood(np.asarray(centre, dtype=np.uint64), LEVEL, k)
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
