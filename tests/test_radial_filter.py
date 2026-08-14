"""Tests for metric radial filtering on NESTED HEALPix fields.

These tests focus on Issue #28-specific semantics:

- physical-distance support expressed in metres,
- isotropic ``kernel(distance_m)`` evaluation,
- normalized versus raw weighted output,
- Gaussian smoothing in physical units,
- processing-domain boundaries,
- WGS84 geometry across difficult HEALPix locations,
- NumPy/Torch consistency and an integration-level autograd check.

Generic weighted-aggregation behaviour is tested independently in
``test_weighted_neighbourhood.py``.  In particular, low-level tests for
padded positions, zero-effective-weight handling, device preservation, and
Torch gathering belong to that shared helper rather than being duplicated
here.

Important domain convention
---------------------------
``domain`` is both:

- the set of cells that participate in the spatial operation, and
- the output domain.

A cell may therefore have an input value in ``cell_ids`` while still being
completely absent from the filter because it lies outside ``domain``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from healpix_geo import nested
from pyproj import Geod

from healpix_analyse._neighbourhood import (
    CompactMetricNeighbourhoodGeometry,
    RelativeNeighbourhoodGeometry,
)
from healpix_analyse.radial_filter import gaussian_filter, radial_filter


_WGS84 = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Real-HEALPix geometry helpers
# ---------------------------------------------------------------------------


def _cell_centres(
    cells: np.ndarray,
    refinement_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 centre longitude/latitude for NESTED HEALPix cells."""
    lon, lat = nested.healpix_to_lonlat(
        np.asarray(cells, dtype=np.uint64),
        refinement_level,
        ellipsoid="WGS84",
    )

    return (
        np.asarray(lon, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
    )


def _pair_distance_m(
    center: int,
    neighbour: int,
    refinement_level: int,
) -> float:
    """Return WGS84 centre-to-centre distance for a HEALPix pair."""
    lon, lat = _cell_centres(
        np.array([center, neighbour], dtype=np.uint64),
        refinement_level,
    )

    _, _, distance_m = _WGS84.inv(
        lon[0],
        lat[0],
        lon[1],
        lat[1],
    )

    return float(distance_m)


def _immediate_neighbours(
    center: int,
    refinement_level: int,
) -> np.ndarray:
    """Return unique immediate neighbours using healpix-geo topology."""
    raw = nested.kth_neighbourhood(
        np.array([center], dtype=np.uint64),
        refinement_level,
        1,
    )

    row = np.asarray(raw, dtype=np.int64)[0]
    valid = row[
        row >= 0
    ].astype(
        np.uint64,
        copy=False,
    )

    valid = valid[
        valid != np.uint64(center)
    ]

    return np.unique(valid)


def _domain_around_center(
    center: int,
    refinement_level: int,
) -> np.ndarray:
    """Return centre + immediate neighbours as a compact real domain."""
    return np.concatenate(
        [
            np.array([center], dtype=np.uint64),
            _immediate_neighbours(center, refinement_level),
        ]
    )


def _radius_reaching_domain_from_center(
    center: int,
    domain: np.ndarray,
    refinement_level: int,
) -> float:
    """Return a radius slightly larger than all centre->domain distances."""
    distances = [
        _pair_distance_m(
            center,
            int(cell),
            refinement_level,
        )
        for cell in domain
        if int(cell) != center
    ]

    if not distances:
        return 1.0

    return max(distances) * 1.01


def _base_pixel_id(
    cell: int,
    refinement_level: int,
) -> int:
    """Return the level-0 HEALPix base-pixel ID of a NESTED cell."""
    return int(cell) >> (2 * refinement_level)


def _find_base_pixel_crossing_pair(
    refinement_level: int,
) -> tuple[int, int]:
    """Find adjacent cells belonging to different HEALPix base pixels."""
    number_of_pixels = 12 * 4**refinement_level

    for center in range(number_of_pixels):
        center_base = _base_pixel_id(
            center,
            refinement_level,
        )

        for neighbour in _immediate_neighbours(
            center,
            refinement_level,
        ):
            if _base_pixel_id(
                int(neighbour),
                refinement_level,
            ) != center_base:
                return center, int(neighbour)

    raise AssertionError(
        "Could not find a HEALPix base-pixel boundary crossing pair."
    )


def _find_longitude_wrap_pair(
    refinement_level: int,
) -> tuple[int, int]:
    """Find adjacent cells whose longitudes straddle the +/-180 degree wrap."""
    number_of_pixels = 12 * 4**refinement_level
    cells = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    lon, _ = _cell_centres(
        cells,
        refinement_level,
    )

    for center in range(number_of_pixels):
        for neighbour in _immediate_neighbours(
            center,
            refinement_level,
        ):
            if abs(
                lon[center]
                - lon[int(neighbour)]
            ) > 300.0:
                return center, int(neighbour)

    raise AssertionError(
        "Could not find an immediate HEALPix pair across longitude wrap."
    )


def _highest_latitude_cell(
    refinement_level: int,
) -> int:
    """Return a real HEALPix cell close to the north polar region."""
    number_of_pixels = 12 * 4**refinement_level
    cells = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    _, lat = _cell_centres(
        cells,
        refinement_level,
    )

    return int(
        np.argmax(lat)
    )


def _output_position(
    domain: np.ndarray,
    cell: int,
) -> int:
    """Return one output position for a cell in an explicitly ordered domain."""
    positions = np.where(
        domain == np.uint64(cell)
    )[0]

    assert positions.size == 1
    return int(positions[0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_duplicate_cell_ids():
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        radial_filter(
            np.array([1.0, 2.0]),
            np.array([10, 10], dtype=np.uint64),
            refinement_level=3,
            radius_m=1000.0,
            kernel=lambda d: np.ones_like(d),
        )


def test_rejects_domain_outside_cell_ids():
    with pytest.raises(
        ValueError,
        match="subset",
    ):
        radial_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            radius_m=1000.0,
            kernel=lambda d: np.ones_like(d),
            domain=np.array([11], dtype=np.uint64),
        )


def test_rejects_negative_radius():
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        radial_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            radius_m=-1.0,
            kernel=lambda d: np.ones_like(d),
        )


def test_rejects_boolean_radius():
    with pytest.raises(
        TypeError,
        match="radius_m",
    ):
        radial_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            radius_m=True,
            kernel=lambda d: np.ones_like(d),
        )


def test_rejects_non_callable_kernel():
    with pytest.raises(
        TypeError,
        match="kernel",
    ):
        radial_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            radius_m=1000.0,
            kernel=1.0,
        )


def test_rejects_non_finite_kernel_weight():
    with pytest.raises(
        ValueError,
        match="non-finite weights",
    ):
        radial_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            radius_m=1.0,
            kernel=lambda d: np.full_like(d, np.inf),
        )


@pytest.mark.parametrize(
    "sigma_m",
    [0.0, -1.0, np.inf, np.nan],
)
def test_gaussian_rejects_invalid_sigma(sigma_m):
    with pytest.raises(
        ValueError,
        match="sigma_m",
    ):
        gaussian_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            sigma_m=sigma_m,
        )


@pytest.mark.parametrize(
    "truncate",
    [0.0, -1.0, np.inf, np.nan],
)
def test_gaussian_rejects_invalid_truncate(truncate):
    with pytest.raises(
        ValueError,
        match="truncate",
    ):
        gaussian_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            sigma_m=100.0,
            truncate=truncate,
        )


# ---------------------------------------------------------------------------
# Core radial semantics
# ---------------------------------------------------------------------------


def test_constant_field_preserved_when_normalized():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.full(
        domain.size,
        7.25,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: 1.0 / (1.0 + d / radius_m),
        normalize=True,
        domain=domain,
    )

    np.testing.assert_allclose(
        result,
        np.full(domain.size, 7.25),
    )


def test_radius_zero_is_center_only_identity():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.arange(
        domain.size,
        dtype=np.float64,
    ) + 1.0

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=0.0,
        kernel=lambda d: np.ones_like(d),
        normalize=True,
        domain=domain,
    )

    np.testing.assert_allclose(
        result,
        values,
    )


def test_scalar_kernel_weight_is_supported():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: 2.0,
        normalize=False,
        domain=domain,
    )

    center_position = _output_position(
        domain,
        center,
    )

    # The centre output sees every cell in this compact domain.  Every valid
    # contribution has weight 2, so the raw weighted sum is 2 * cell count.
    assert result[center_position] == pytest.approx(
        2.0 * domain.size
    )


def test_custom_radial_kernel_matches_manual_weighted_sum():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.arange(
        1,
        domain.size + 1,
        dtype=np.float64,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    scale_m = radius_m / 2.0

    def kernel(distance_m):
        return 1.0 / (
            1.0 + distance_m / scale_m
        )

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )

    distances = np.array(
        [
            _pair_distance_m(
                center,
                int(cell),
                refinement_level,
            )
            for cell in domain
        ],
        dtype=np.float64,
    )

    expected = np.sum(
        kernel(distances)
        * values
    )

    np.testing.assert_allclose(
        result[
            _output_position(
                domain,
                center,
            )
        ],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_negative_finite_radial_weights_are_allowed():
    refinement_level = 5
    center = 1000
    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )
    assert neighbours.size > 0

    neighbour = int(neighbours[0])
    domain = np.array(
        [center, neighbour],
        dtype=np.uint64,
    )

    radius_m = _pair_distance_m(
        center,
        neighbour,
        refinement_level,
    ) * 1.01

    values = np.array(
        [3.0, 5.0],
        dtype=np.float64,
    )

    def signed_kernel(distance_m):
        return np.where(
            distance_m == 0.0,
            1.0,
            -1.0,
        )

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=signed_kernel,
        normalize=False,
        domain=domain,
    )

    assert result[0] == pytest.approx(
        3.0 - 5.0
    )


# ---------------------------------------------------------------------------
# Gaussian wrapper
# ---------------------------------------------------------------------------


def test_gaussian_filter_matches_explicit_radial_gaussian():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.linspace(
        1.0,
        3.0,
        domain.size,
    )

    sigma_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    ) / 2.0
    truncate = 2.0
    radius_m = sigma_m * truncate

    expected = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.exp(
            -0.5 * (d / sigma_m) ** 2
        ),
        normalize=True,
        domain=domain,
    )

    result = gaussian_filter(
        values,
        domain,
        refinement_level,
        sigma_m=sigma_m,
        truncate=truncate,
        domain=domain,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_gaussian_reuses_geometry_and_weights(monkeypatch):
    """Repeated filtering of new values must reuse the spatial plan."""
    module = importlib.import_module("healpix_analyse.radial_filter")
    module._clear_filter_caches()

    refinement_level = 7
    center = 12 * 4**refinement_level // 2
    cell_ids = _domain_around_center(center, refinement_level)
    values = np.arange(cell_ids.size, dtype=np.float64)
    sigma_m = 50_000.0

    calls = 0
    original = module.build_metric_neighbourhood_geometry

    def counted_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "build_metric_neighbourhood_geometry",
        counted_build,
    )

    first = gaussian_filter(
        values,
        cell_ids,
        refinement_level,
        sigma_m=sigma_m,
    )
    second = gaussian_filter(
        values + 1.0,
        cell_ids.copy(),
        refinement_level,
        sigma_m=sigma_m,
    )

    assert calls == 1
    np.testing.assert_allclose(second, first + 1.0)
    module._clear_filter_caches()


def test_gaussian_constant_field_preserved_at_partial_domain_boundary():
    refinement_level = 5
    center = 1000
    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )
    assert neighbours.size >= 2

    inside = int(neighbours[0])
    outside = int(neighbours[1])

    cell_ids = np.array(
        [center, inside, outside],
        dtype=np.uint64,
    )

    # The huge outside-domain value is intentionally present in the input.
    # It must remain completely absent from the filter after domain
    # restriction rather than acting as a large neighbouring sample.
    values = np.array(
        [4.0, 4.0, 1.0e9],
        dtype=np.float64,
    )

    domain = np.array(
        [center, inside],
        dtype=np.uint64,
    )

    radius_m = max(
        _pair_distance_m(
            center,
            inside,
            refinement_level,
        ),
        _pair_distance_m(
            center,
            outside,
            refinement_level,
        ),
    ) * 1.01

    result = gaussian_filter(
        values,
        cell_ids,
        refinement_level,
        sigma_m=radius_m / 2.0,
        truncate=2.0,
        domain=domain,
    )

    np.testing.assert_allclose(
        result,
        np.full(domain.size, 4.0),
    )


def test_nan_neighbour_is_excluded_and_remaining_weights_renormalized():
    refinement_level = 5
    center = 1000
    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )
    assert neighbours.size > 0

    neighbour = int(neighbours[0])
    domain = np.array(
        [center, neighbour],
        dtype=np.uint64,
    )

    values = np.array(
        [4.0, np.nan],
        dtype=np.float64,
    )

    radius_m = _pair_distance_m(
        center,
        neighbour,
        refinement_level,
    ) * 1.01

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.ones_like(d),
        normalize=True,
        domain=domain,
    )

    # For the centre cell, the NaN neighbour is unavailable and only the
    # finite self contribution remains.  The shared aggregation helper must
    # remove the missing sample from both numerator and denominator.
    assert result[
        _output_position(
            domain,
            center,
        )
    ] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Real HEALPix topology / geographical edge cases
# ---------------------------------------------------------------------------


def test_filter_crosses_healpix_base_pixel_boundary():
    refinement_level = 3
    center, neighbour = _find_base_pixel_crossing_pair(
        refinement_level
    )

    assert _base_pixel_id(
        center,
        refinement_level,
    ) != _base_pixel_id(
        neighbour,
        refinement_level,
    )

    domain = np.array(
        [center, neighbour],
        dtype=np.uint64,
    )

    values = np.array(
        [0.0, 1.0],
    )

    radius_m = _pair_distance_m(
        center,
        neighbour,
        refinement_level,
    ) * 1.01

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.ones_like(d),
        normalize=False,
        domain=domain,
    )

    # Self contributes zero and the crossing neighbour contributes one.
    assert result[
        _output_position(
            domain,
            center,
        )
    ] == pytest.approx(1.0)



def test_filter_crosses_longitude_wrap():
    refinement_level = 5
    center, neighbour = _find_longitude_wrap_pair(
        refinement_level
    )

    lon, _ = _cell_centres(
        np.array([center, neighbour], dtype=np.uint64),
        refinement_level,
    )

    assert abs(
        lon[0] - lon[1]
    ) > 300.0

    domain = np.array(
        [center, neighbour],
        dtype=np.uint64,
    )

    values = np.array(
        [0.0, 2.0],
    )

    radius_m = _pair_distance_m(
        center,
        neighbour,
        refinement_level,
    ) * 1.01

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.ones_like(d),
        normalize=False,
        domain=domain,
    )

    assert result[
        _output_position(
            domain,
            center,
        )
    ] == pytest.approx(2.0)



def test_filter_works_near_polar_region():
    refinement_level = 5
    center = _highest_latitude_cell(
        refinement_level
    )
    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )
    assert neighbours.size > 0

    neighbour = int(neighbours[0])
    domain = np.array(
        [center, neighbour],
        dtype=np.uint64,
    )

    _, lat = _cell_centres(
        np.array([center], dtype=np.uint64),
        refinement_level,
    )
    assert lat[0] > 80.0

    radius_m = _pair_distance_m(
        center,
        neighbour,
        refinement_level,
    ) * 1.01

    result = radial_filter(
        np.array([3.0, 9.0]),
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.ones_like(d),
        normalize=True,
        domain=domain,
    )

    assert np.all(
        np.isfinite(result)
    )


# ---------------------------------------------------------------------------
# Physical-scale semantics independent of refinement-level pixel units
# ---------------------------------------------------------------------------


def test_same_metric_geometry_gives_same_gaussian_response_across_levels(
    monkeypatch,
):
    """Gaussian weights depend on metres, not HEALPix pixel/ring count.

    This is a focused unit test of the public physical-scale contract.  The
    shared geometry layer is replaced with the same metric geometry at two
    refinement levels.  The filter must therefore produce identical output.

    Separate real-geometry tests above cover base-pixel, wrap, and polar
    behaviour.  This test specifically prevents accidental introduction of a
    refinement-level-dependent ``sigma_pixels`` or ring scaling inside #28.
    """
    module = importlib.import_module(
        "healpix_analyse.radial_filter"
    )

    cell_ids = np.array(
        [10, 11],
        dtype=np.uint64,
    )

    values = np.array(
        [1.0, 0.0],
        dtype=np.float64,
    )

    fake_geometry = CompactMetricNeighbourhoodGeometry(
        center_ids=cell_ids.copy(),
        neighbour_indices=np.array(
            [0, 1, 0, 1],
            dtype=np.uint32,
        ),
        row_offsets=np.array([0, 2, 4], dtype=np.int64),
        distance_m=np.array(
            [0.0, 250.0, 250.0, 0.0],
            dtype=np.float64,
        ),
    )

    def fake_metric_geometry(
        cells,
        radius,
        refinement_level,
        *,
        ellipsoid,
    ):
        del cells, radius, refinement_level, ellipsoid
        return fake_geometry

    monkeypatch.setattr(
        module,
        "build_metric_neighbourhood_geometry",
        fake_metric_geometry,
    )

    outputs = []

    for refinement_level in [4, 9]:
        outputs.append(
            gaussian_filter(
                values,
                cell_ids,
                refinement_level,
                sigma_m=200.0,
                truncate=4.0,
                domain=cell_ids,
            )
        )

    np.testing.assert_allclose(
        outputs[0],
        outputs[1],
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# NumPy / Torch integration
# ---------------------------------------------------------------------------


def test_numpy_torch_consistency():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = np.linspace(
        1.0,
        5.0,
        domain.size,
        dtype=np.float64,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    numpy_result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.exp(-d / radius_m),
        normalize=True,
        domain=domain,
    )

    torch_result = radial_filter(
        torch.tensor(
            values,
            dtype=torch.float64,
        ),
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.exp(-d / radius_m),
        normalize=True,
        domain=domain,
    )

    assert isinstance(
        torch_result,
        torch.Tensor,
    )

    np.testing.assert_allclose(
        torch_result.detach().cpu().numpy(),
        numpy_result,
        rtol=1.0e-12,
        atol=1.0e-12,
    )



def test_torch_autograd_through_radial_filter():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = torch.linspace(
        1.0,
        3.0,
        domain.size,
        dtype=torch.float64,
        requires_grad=True,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    result = radial_filter(
        values,
        domain,
        refinement_level,
        radius_m=radius_m,
        kernel=lambda d: np.exp(-d / radius_m),
        normalize=True,
        domain=domain,
    )

    loss = result.sum()
    loss.backward()

    assert values.grad is not None
    assert torch.all(
        torch.isfinite(values.grad)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_output_stays_on_cuda():
    refinement_level = 5
    center = 1000
    domain = _domain_around_center(
        center,
        refinement_level,
    )

    values = torch.ones(
        domain.size,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    radius_m = _radius_reaching_domain_from_center(
        center,
        domain,
        refinement_level,
    )

    result = gaussian_filter(
        values,
        domain,
        refinement_level,
        sigma_m=radius_m / 2.0,
        truncate=2.0,
        domain=domain,
    )

    assert result.device.type == "cuda"

    result.sum().backward()
    assert values.grad is not None
    assert values.grad.device.type == "cuda"
