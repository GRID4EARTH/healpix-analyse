"""Tests for local scalar-field gradients on NESTED HEALPix grids."""

import numpy as np
import pytest
import torch
from healpix_geo import nested

from healpix_analyse._neighbourhood import (
    build_relative_geometry,
    build_ring_neighbourhoods,
    relative_geometry_from_neighbours,
)
from healpix_analyse.gradient import (
    directional_derivative,
    gradient,
    gradient_magnitude,
)


def _linear_patch(
    center: int,
    refinement_level: int,
    *,
    gradient_east: float,
    gradient_north: float,
    constant: float = 3.0,
):
    """Create a local scalar field with an exactly known tangent gradient.

    Values are constructed directly from the same WGS84 East/North geometry
    used by the gradient definition:

        f = constant
            + gradient_east * east_offset
            + gradient_north * north_offset

    Therefore the least-squares gradient at ``center`` should recover the
    prescribed coefficients up to floating-point precision.
    """

    center_ids = np.array(
        [center],
        dtype=np.uint64,
    )

    geometry = build_relative_geometry(
        center_ids,
        refinement_level,
        ring=1,
    )

    neighbours = geometry.neighbour_ids[
        0,
        geometry.valid_mask[0],
    ].astype(
        np.uint64
    )

    cell_ids = np.concatenate(
        [
            center_ids,
            neighbours,
        ]
    )

    values = np.empty(
        cell_ids.size,
        dtype=np.float64,
    )

    values[0] = constant

    east = geometry.east_offset_m[
        0,
        geometry.valid_mask[0],
    ]

    north = geometry.north_offset_m[
        0,
        geometry.valid_mask[0],
    ]

    values[1:] = (
        constant
        + gradient_east
        * east
        + gradient_north
        * north
    )

    return (
        cell_ids,
        values,
    )


# ---------------------------------------------------------------------------
# Constant fields
# ---------------------------------------------------------------------------


def test_constant_field_has_zero_gradient():
    """A constant scalar field must have zero local derivative."""

    refinement_level = 5

    center = 42

    cell_ids, _ = _linear_patch(
        center,
        refinement_level,
        gradient_east=0.0,
        gradient_north=0.0,
    )

    values = np.full(
        cell_ids.size,
        7.5,
        dtype=np.float64,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        0.0,
        abs=1e-15,
    )

    assert north[0] == pytest.approx(
        0.0,
        abs=1e-15,
    )


# ---------------------------------------------------------------------------
# Analytical local linear fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    (
        "expected_east",
        "expected_north",
    ),
    [
        (1.0e-3, 0.0),
        (0.0, 2.5e-3),
        (1.7e-3, -8.0e-4),
        (-2.0e-4, 3.0e-4),
    ],
)
def test_local_linear_field_recovers_known_gradient(
    expected_east,
    expected_north,
):
    """Least squares must exactly recover a locally linear tangent field."""

    refinement_level = 6

    center = 1000

    cell_ids, values = _linear_patch(
        center,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-10,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-10,
        abs=1e-12,
    )


def test_gradient_magnitude_matches_known_linear_field():
    """Gradient magnitude must be hypot(East, North)."""

    refinement_level = 6

    expected_east = 3.0e-3
    expected_north = 4.0e-3

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    magnitude = gradient_magnitude(
        values,
        cell_ids,
        refinement_level,
    )

    assert magnitude[0] == pytest.approx(
        5.0e-3,
        rel=1e-10,
    )


# ---------------------------------------------------------------------------
# Geographic directional derivatives
# ---------------------------------------------------------------------------


def test_directional_derivative_north_returns_north_component():
    """Azimuth zero means geographic North."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=2.0e-3,
        gradient_north=-7.0e-4,
    )

    derivative = directional_derivative(
        values,
        cell_ids,
        refinement_level,
        azimuth_rad=0.0,
    )

    assert derivative[0] == pytest.approx(
        -7.0e-4,
        rel=1e-10,
    )


def test_directional_derivative_east_returns_east_component():
    """Azimuth pi/2 means geographic East."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=2.0e-3,
        gradient_north=-7.0e-4,
    )

    derivative = directional_derivative(
        values,
        cell_ids,
        refinement_level,
        azimuth_rad=np.pi / 2.0,
    )

    assert derivative[0] == pytest.approx(
        2.0e-3,
        rel=1e-10,
    )


def test_directional_derivative_arbitrary_azimuth():
    """Directional derivative must project the tangent gradient."""

    refinement_level = 6

    expected_east = 1.2e-3
    expected_north = -9.0e-4

    azimuth = np.deg2rad(
        37.0
    )

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    derivative = directional_derivative(
        values,
        cell_ids,
        refinement_level,
        azimuth_rad=azimuth,
    )

    expected = (
        expected_east
        * np.sin(
            azimuth
        )
        + expected_north
        * np.cos(
            azimuth
        )
    )

    assert derivative[0] == pytest.approx(
        expected,
        rel=1e-10,
    )


# ---------------------------------------------------------------------------
# Domain semantics
# ---------------------------------------------------------------------------


def test_outside_domain_neighbour_does_not_contribute():
    """A cell outside domain is absent, not a numerical padding value."""

    refinement_level = 6

    expected_east = 8.0e-4
    expected_north = -3.0e-4

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    # Put an intentionally absurd value on one neighbour.
    #
    # The cell remains present in ``cell_ids`` but is removed from the
    # processing domain.  If domain semantics are implemented correctly,
    # that value cannot affect the centre-cell gradient.
    values[-1] = 1.0e30

    domain = cell_ids[
        :-1
    ]

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-10,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-10,
        abs=1e-12,
    )


def test_insufficient_domain_neighbourhood_returns_nan():
    """A single neighbour cannot determine a 2-D tangent gradient."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.0e-3,
        gradient_north=2.0e-3,
    )

    domain = cell_ids[
        :2
    ]

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
    )

    assert np.isnan(
        east[0]
    )

    assert np.isnan(
        north[0]
    )


def test_domain_must_be_subset_of_cell_ids():
    """Unknown output cells must be rejected."""

    refinement_level = 3

    with pytest.raises(
        ValueError,
        match="subset",
    ):
        gradient(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [0, 1],
                dtype=np.uint64,
            ),
            refinement_level,
            domain=np.array(
                [0, 2],
                dtype=np.uint64,
            ),
        )


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def test_missing_neighbour_is_ignored_when_fit_remains_rank_two():
    """NaN neighbours are absent from the numerical local fit."""

    refinement_level = 6

    expected_east = 1.5e-3
    expected_north = -2.5e-4

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    values[-1] = np.nan

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-10,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-10,
        abs=1e-12,
    )


def test_missing_center_returns_nan():
    """A target cell without a finite scalar value has no gradient."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.0e-3,
        gradient_north=2.0e-3,
    )

    values[0] = np.nan

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert np.isnan(
        east[0]
    )

    assert np.isnan(
        north[0]
    )


# ---------------------------------------------------------------------------
# Multiple refinement levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refinement_level",
    [
        0,
        1,
        3,
        6,
    ],
)
def test_local_linear_gradient_multiple_refinement_levels(
    refinement_level,
):
    """The same metric-gradient semantics must hold at all tested levels."""

    expected_east = 5.0e-4
    expected_north = -2.0e-4

    center = 0

    cell_ids, values = _linear_patch(
        center,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-9,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-9,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# Latitude robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latitude",
    [
        0.0,
        80.0,
        -80.0,
    ],
)
def test_local_linear_gradient_at_different_latitudes(
    latitude,
):
    """Geographic tangent gradients must work near equator and poles."""

    refinement_level = 6

    center = int(
        nested.lonlat_to_healpix(
            np.array(
                [10.0],
                dtype=np.float64,
            ),
            np.array(
                [latitude],
                dtype=np.float64,
            ),
            refinement_level,
            ellipsoid="WGS84",
        )[0]
    )

    expected_east = 7.0e-4
    expected_north = 4.0e-4

    cell_ids, values = _linear_patch(
        center,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-9,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-9,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# Base-pixel boundary
# ---------------------------------------------------------------------------


def test_gradient_across_base_pixel_boundary():
    """Immediate neighbours across NESTED base pixels must participate."""

    refinement_level = 3

    cells_per_base_pixel = (
        4**refinement_level
    )

    number_of_pixels = (
        12
        * cells_per_base_pixel
    )

    center = None

    for candidate in range(
        number_of_pixels
    ):
        neighbours = build_ring_neighbourhoods(
            np.array(
                [candidate],
                dtype=np.uint64,
            ),
            refinement_level,
            ring=1,
        )[0]

        candidate_base = (
            candidate
            // cells_per_base_pixel
        )

        if any(
            int(neighbour)
            // cells_per_base_pixel
            != candidate_base
            for neighbour in neighbours
        ):
            center = candidate
            break

    assert center is not None

    expected_east = 9.0e-4
    expected_north = -6.0e-4

    cell_ids, values = _linear_patch(
        center,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-9,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-9,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# Longitude wrap
# ---------------------------------------------------------------------------


def test_gradient_across_longitude_wrap():
    """Longitude 0/360 must not behave like a spatial boundary."""

    refinement_level = 4

    number_of_pixels = (
        12
        * 4**refinement_level
    )

    cells = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    lon, _ = nested.healpix_to_lonlat(
        cells,
        refinement_level,
        ellipsoid="WGS84",
    )

    center = None

    for candidate in cells:
        neighbours = build_ring_neighbourhoods(
            np.array(
                [candidate],
                dtype=np.uint64,
            ),
            refinement_level,
            ring=1,
        )[0]

        if any(
            abs(
                float(
                    lon[
                        int(candidate)
                    ]
                )
                - float(
                    lon[
                        int(neighbour)
                    ]
                )
            )
            > 180.0
            for neighbour in neighbours
        ):
            center = int(
                candidate
            )
            break

    assert center is not None

    expected_east = 6.0e-4
    expected_north = 3.0e-4

    cell_ids, values = _linear_patch(
        center,
        refinement_level,
        gradient_east=expected_east,
        gradient_north=expected_north,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    assert east[0] == pytest.approx(
        expected_east,
        rel=1e-9,
        abs=1e-12,
    )

    assert north[0] == pytest.approx(
        expected_north,
        rel=1e-9,
        abs=1e-12,
    )


# ---------------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------------


def test_torch_cpu_matches_numpy():
    """Torch and NumPy must implement the same gradient semantics."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.1e-3,
        gradient_north=-4.0e-4,
    )

    numpy_east, numpy_north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    torch_east, torch_north = gradient(
        torch.tensor(
            values,
            dtype=torch.float64,
        ),
        cell_ids,
        refinement_level,
    )

    np.testing.assert_allclose(
        torch_east.detach().cpu().numpy(),
        numpy_east,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        torch_north.detach().cpu().numpy(),
        numpy_north,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-12,
    )


def test_torch_gradient_preserves_autograd():
    """Gradient computation must remain differentiable in input values."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.0e-3,
        gradient_north=2.0e-3,
    )

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
        requires_grad=True,
    )

    east, north = gradient(
        tensor,
        cell_ids,
        refinement_level,
    )

    loss = (
        east[0]
        + 2.0
        * north[0]
    )

    loss.backward()

    assert tensor.grad is not None

    assert torch.all(
        torch.isfinite(
            tensor.grad
        )
    )

    assert torch.any(
        tensor.grad
        != 0
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available.",
)
def test_torch_mps_roundtrip():
    """Gradient outputs must remain on the MPS device."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.0e-3,
        gradient_north=-5.0e-4,
    )

    tensor = torch.tensor(
        values,
        dtype=torch.float32,
        device="mps",
    )

    east, north = gradient(
        tensor,
        cell_ids,
        refinement_level,
    )

    assert east.device.type == "mps"
    assert north.device.type == "mps"

    assert float(
        east[0].cpu()
    ) == pytest.approx(
        1.0e-3,
        rel=1e-4,
    )

    assert float(
        north[0].cpu()
    ) == pytest.approx(
        -5.0e-4,
        rel=1e-4,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_torch_cuda_roundtrip():
    """Gradient outputs must remain on the CUDA device."""

    refinement_level = 6

    cell_ids, values = _linear_patch(
        1000,
        refinement_level,
        gradient_east=1.0e-3,
        gradient_north=-5.0e-4,
    )

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
        device="cuda",
    )

    east, north = gradient(
        tensor,
        cell_ids,
        refinement_level,
    )

    assert east.device.type == "cuda"
    assert north.device.type == "cuda"

# ---------------------------------------------------------------------------
# Rotationally symmetric fields
# ---------------------------------------------------------------------------


def test_rotationally_symmetric_field_has_zero_gradient_at_center():
    """A radially symmetric field must have zero gradient at its centre.

    The scalar value is defined only by the WGS84 centre-to-centre distance
    from the target cell:

        f = distance_m ** 2

    Therefore no geographic direction is preferred around the centre.

    The local East/North least-squares gradient at the centre should be
    approximately zero.

    This is a useful regression test because it checks that the gradient
    does not accidentally inherit a preferred HEALPix index direction.
    """

    refinement_level = 6

    center = 1000

    center_ids = np.array(
        [center],
        dtype=np.uint64,
    )

    geometry = build_relative_geometry(
        center_ids,
        refinement_level,
        ring=1,
    )

    valid = geometry.valid_mask[0]

    neighbours = geometry.neighbour_ids[
        0,
        valid,
    ].astype(
        np.uint64,
    )

    cell_ids = np.concatenate(
        [
            center_ids,
            neighbours,
        ]
    )

    values = np.empty(
        cell_ids.size,
        dtype=np.float64,
    )

    # Centre of the radial field.
    values[0] = 0.0

    distance = geometry.distance_m[
        0,
        valid,
    ]

    # Quadratic radial field.
    #
    # Using distance**2 avoids a cusp at the centre and gives a true
    # zero analytical gradient there.
    values[1:] = (
        distance
        * distance
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    # HEALPix neighbour locations are not perfectly rotationally symmetric,
    # so the discrete least-squares estimate need not be machine-zero.
    #
    # Compare the residual gradient with the natural scale of this local
    # quadratic field rather than demanding exact cancellation.
    characteristic_gradient = (
        2.0
        * np.max(
            distance
        )
    )

    assert abs(
        east[0]
    ) < (
        0.15
        * characteristic_gradient
    )

    assert abs(
        north[0]
    ) < (
        0.15
        * characteristic_gradient
    )
# ---------------------------------------------------------------------------
# API and domain-contract regression tests
# ---------------------------------------------------------------------------


def test_domain_output_follows_domain_order():
    """Output ordering must follow ``domain`` exactly.

    Reordering the domain must reorder the gradient outputs in exactly the
    same way without changing the underlying numerical result.

    The complete level-1 HEALPix grid is used as the domain so that changing
    its order does not change which neighbours are available to the local
    fits.
    """

    refinement_level = 1

    number_of_pixels = (
        12
        * 4**refinement_level
    )

    cell_ids = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    # An arbitrary non-constant scalar field is sufficient here.  This test
    # concerns output ordering, not analytical gradient accuracy.
    values = (
        0.25
        * cell_ids.astype(
            np.float64
        )
        + np.sin(
            cell_ids.astype(
                np.float64
            )
        )
    )

    east_reference, north_reference = gradient(
        values,
        cell_ids,
        refinement_level,
    )

    reversed_domain = cell_ids[
        ::-1
    ].copy()

    east_reversed, north_reversed = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=reversed_domain,
    )

    np.testing.assert_allclose(
        east_reversed,
        east_reference[::-1],
        equal_nan=True,
    )

    np.testing.assert_allclose(
        north_reversed,
        north_reference[::-1],
        equal_nan=True,
    )

def test_invalid_gradient_method_raises():
    """Unknown gradient-estimation methods must fail explicitly."""

    refinement_level = 1

    with pytest.raises(
        ValueError,
        match="least_squares",
    ):
        gradient(
            np.ones(
                12 * 4**refinement_level,
                dtype=np.float64,
            ),
            np.arange(
                12 * 4**refinement_level,
                dtype=np.uint64,
            ),
            refinement_level,
            method="sobel",
        )

@pytest.mark.parametrize(
    "method",
    [
        None,
        1,
        True,
    ],
)
def test_gradient_method_must_be_string(
    method,
):
    """The gradient method must be supplied as a string."""

    refinement_level = 1

    with pytest.raises(
        TypeError,
        match="string",
    ):
        gradient(
            np.ones(
                12 * 4**refinement_level,
                dtype=np.float64,
            ),
            np.arange(
                12 * 4**refinement_level,
                dtype=np.uint64,
            ),
            refinement_level,
            method=method,
        )


def test_values_and_cell_ids_must_have_same_length():
    """Each input scalar value must correspond to exactly one HEALPix cell."""

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        gradient(
            np.array(
                [
                    1.0,
                    2.0,
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    0,
                    1,
                    2,
                ],
                dtype=np.uint64,
            ),
            refinement_level=1,
        )


def test_empty_domain_returns_empty_outputs():
    """An empty processing domain must produce empty gradient arrays.

    An empty domain is a valid regional-processing case.  It must not be
    confused with an invalid neighbourhood or cause geometry construction
    to fail.
    """

    refinement_level = 1

    cell_ids = np.arange(
        12 * 4**refinement_level,
        dtype=np.uint64,
    )

    values = np.ones(
        cell_ids.size,
        dtype=np.float64,
    )

    domain = np.array(
        [],
        dtype=np.uint64,
    )

    east, north = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
    )

    assert isinstance(
        east,
        np.ndarray,
    )

    assert isinstance(
        north,
        np.ndarray,
    )

    assert east.shape == (0,)
    assert north.shape == (0,)


def test_empty_domain_propagates_to_derived_gradient_operators():
    """Derived gradient APIs must preserve empty-domain semantics."""

    refinement_level = 1

    cell_ids = np.arange(
        12 * 4**refinement_level,
        dtype=np.uint64,
    )

    values = np.ones(
        cell_ids.size,
        dtype=np.float64,
    )

    domain = np.array(
        [],
        dtype=np.uint64,
    )

    magnitude = gradient_magnitude(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
    )

    derivative = directional_derivative(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
        azimuth_rad=np.pi / 4.0,
    )

    assert magnitude.shape == (0,)
    assert derivative.shape == (0,)
