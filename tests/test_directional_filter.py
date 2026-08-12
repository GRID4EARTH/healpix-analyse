"""Tests for geographical directional filtering on NESTED HEALPix fields.

These tests focus on Issue #29-specific semantics:

- geographical azimuth,
- clockwise-from-North convention,
- forward bearing from output cell to contributing neighbour,
- physical-distance support in metres,
- HEALPix base-pixel and longitude-wrap behaviour,
- high-latitude geometry,
- partial-domain semantics,
- asymmetric directional kernels,
- Sentinel-2 shadow-direction validation.

Generic weighted aggregation details such as NaN handling, normalization,
Torch device preservation, and autograd are tested independently in
``test_weighted_neighbourhood.py``.

Important domain convention
---------------------------
``domain`` is both:
- the set of cells that participate in the operation, and
- the output domain.

Therefore a neighbour can contribute only when it belongs to ``domain``.
Tests that expect a neighbour contribution include the relevant cells in the
domain and then inspect the output corresponding to the target cell.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_geo import nested
from pyproj import Geod

from healpix_analyse.directional_filter import directional_filter

_WGS84 = Geod(ellps="WGS84")


def _cell_centres(cells: np.ndarray, refinement_level: int):
    lon, lat = nested.healpix_to_lonlat(
        np.asarray(cells, dtype=np.uint64),
        refinement_level,
        ellipsoid="WGS84",
    )
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _pair_geometry(center: int, neighbour: int, refinement_level: int):
    lon, lat = _cell_centres(
        np.array([center, neighbour], dtype=np.uint64),
        refinement_level,
    )
    azimuth_deg, _, distance_m = _WGS84.inv(
        lon[0], lat[0], lon[1], lat[1]
    )
    return float(distance_m), float(np.deg2rad(azimuth_deg) % (2.0 * np.pi))


def _immediate_neighbours(center: int, refinement_level: int) -> np.ndarray:
    raw = nested.kth_neighbourhood(
        np.array([center], dtype=np.uint64),
        refinement_level,
        1,
    )
    row = np.asarray(raw, dtype=np.int64)[0]
    valid = row[row >= 0].astype(np.uint64, copy=False)
    valid = valid[valid != np.uint64(center)]
    return np.unique(valid)


def _choose_directional_neighbour(center, refinement_level, requested_azimuth_rad):
    neighbours = _immediate_neighbours(center, refinement_level)
    assert neighbours.size > 0

    distances = []
    bearings = []
    for neighbour in neighbours:
        distance_m, bearing_rad = _pair_geometry(
            center, int(neighbour), refinement_level
        )
        distances.append(distance_m)
        bearings.append(bearing_rad)

    distances = np.asarray(distances)
    bearings = np.asarray(bearings)
    delta = (
        bearings - requested_azimuth_rad + np.pi
    ) % (2.0 * np.pi) - np.pi
    index = int(np.argmin(np.abs(delta)))
    return int(neighbours[index]), float(distances[index]), float(bearings[index])


def _make_domain_around_center(center: int, refinement_level: int) -> np.ndarray:
    neighbours = _immediate_neighbours(center, refinement_level)
    return np.concatenate(
        [np.array([center], dtype=np.uint64), neighbours]
    )


def _center_output_position(domain: np.ndarray, center: int) -> int:
    positions = np.where(domain == np.uint64(center))[0]
    assert positions.size == 1
    return int(positions[0])


def _angular_selector(half_width_rad: float):
    def kernel(distance_m, relative_bearing_rad):
        del distance_m
        return (
            np.abs(relative_bearing_rad) <= half_width_rad
        ).astype(np.float64)
    return kernel


def _distance_and_angle_selector(
    target_distance_m: float,
    distance_tolerance_m: float,
    half_width_rad: float,
):
    def kernel(distance_m, relative_bearing_rad):
        radial = (
            np.abs(distance_m - target_distance_m)
            <= distance_tolerance_m
        )
        angular = (
            np.abs(relative_bearing_rad)
            <= half_width_rad
        )
        return (radial & angular).astype(np.float64)
    return kernel


def test_rejects_duplicate_cell_ids():
    with pytest.raises(ValueError, match="duplicate"):
        directional_filter(
            np.array([1.0, 2.0]),
            np.array([10, 10], dtype=np.uint64),
            refinement_level=3,
            max_distance_m=1000.0,
            azimuth_rad=0.0,
            kernel=lambda d, a: np.ones_like(d),
        )


def test_rejects_domain_outside_cell_ids():
    with pytest.raises(ValueError, match="subset"):
        directional_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            max_distance_m=1000.0,
            azimuth_rad=0.0,
            kernel=lambda d, a: np.ones_like(d),
            domain=np.array([11], dtype=np.uint64),
        )


def test_rejects_negative_max_distance():
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        directional_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            max_distance_m=-1.0,
            azimuth_rad=0.0,
            kernel=lambda d, a: np.ones_like(d),
        )


def test_rejects_non_callable_kernel():
    with pytest.raises(TypeError, match="kernel"):
        directional_filter(
            np.array([1.0]),
            np.array([10], dtype=np.uint64),
            refinement_level=3,
            max_distance_m=1000.0,
            azimuth_rad=0.0,
            kernel=1.0,
        )


def test_constant_field_preserved_when_normalized():
    refinement_level = 5
    center = 1000
    domain = _make_domain_around_center(center, refinement_level)
    values = np.full(domain.size, 7.25)

    max_distance_m = max(
        _pair_geometry(center, int(neighbour), refinement_level)[0]
        for neighbour in domain
    ) * 1.05

    result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=max_distance_m,
        azimuth_rad=0.0,
        kernel=lambda d, a: np.ones_like(d),
        normalize=True,
        domain=domain,
    )
    np.testing.assert_allclose(result, np.full(domain.size, 7.25))


def test_output_order_follows_domain():
    refinement_level = 5
    center = 1000
    neighbours = _immediate_neighbours(center, refinement_level)
    assert neighbours.size >= 2

    cell_ids = np.concatenate(
        [np.array([center], dtype=np.uint64), neighbours]
    )
    values = np.arange(cell_ids.size, dtype=np.float64)
    domain = np.array(
        [neighbours[1], center, neighbours[0]],
        dtype=np.uint64,
    )

    result = directional_filter(
        values,
        cell_ids,
        refinement_level,
        max_distance_m=1.0,
        azimuth_rad=0.0,
        kernel=lambda d, a: np.ones_like(d),
        normalize=True,
        domain=domain,
    )

    lookup = {int(cell): values[i] for i, cell in enumerate(cell_ids)}
    expected = np.array([lookup[int(cell)] for cell in domain])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("requested_azimuth_rad", "name"),
    [
        (0.0, "north"),
        (np.pi / 2.0, "east"),
        (np.pi, "south"),
        (3.0 * np.pi / 2.0, "west"),
    ],
)
def test_geographical_cardinal_azimuth_selects_expected_direction(
    requested_azimuth_rad,
    name,
):
    del name
    refinement_level = 6
    center = 5000

    neighbour, distance_m, actual_bearing = _choose_directional_neighbour(
        center,
        refinement_level,
        requested_azimuth_rad,
    )

    angular_error = (
        actual_bearing - requested_azimuth_rad + np.pi
    ) % (2.0 * np.pi) - np.pi

    assert abs(angular_error) < np.deg2rad(50.0)

    domain = _make_domain_around_center(center, refinement_level)
    values = np.zeros(domain.size, dtype=np.float64)
    neighbour_position = np.where(domain == neighbour)[0]
    assert neighbour_position.size == 1
    values[neighbour_position[0]] = 1.0

    result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=requested_azimuth_rad,
        kernel=_angular_selector(np.deg2rad(50.0)),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert result[center_position] > 0.0


def test_clockwise_from_north_convention():
    refinement_level = 6
    center = 5000

    east_neighbour, east_distance, east_bearing = _choose_directional_neighbour(
        center,
        refinement_level,
        np.pi / 2.0,
    )

    assert abs(
        (
            east_bearing - np.pi / 2.0 + np.pi
        ) % (2.0 * np.pi) - np.pi
    ) < np.deg2rad(50.0)

    domain = _make_domain_around_center(center, refinement_level)
    values = np.zeros(domain.size, dtype=np.float64)
    values[np.where(domain == east_neighbour)[0][0]] = 1.0

    east_result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=east_distance * 1.05,
        azimuth_rad=np.pi / 2.0,
        kernel=_angular_selector(np.deg2rad(45.0)),
        normalize=False,
        domain=domain,
    )

    west_result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=east_distance * 1.05,
        azimuth_rad=3.0 * np.pi / 2.0,
        kernel=_angular_selector(np.deg2rad(45.0)),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert east_result[center_position] > west_result[center_position]


def test_azimuth_wrap_is_equivalent():
    refinement_level = 5
    center = 1000
    domain = _make_domain_around_center(center, refinement_level)
    values = np.arange(domain.size, dtype=np.float64)

    max_distance_m = max(
        _pair_geometry(center, int(neighbour), refinement_level)[0]
        for neighbour in domain
    ) * 1.05

    kernel = _angular_selector(np.deg2rad(30.0))

    first = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=max_distance_m,
        azimuth_rad=np.pi / 2.0,
        kernel=kernel,
        domain=domain,
    )
    second = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=max_distance_m,
        azimuth_rad=np.pi / 2.0 + 2.0 * np.pi,
        kernel=kernel,
        domain=domain,
    )

    np.testing.assert_allclose(first, second)


def test_forward_bearing_is_target_to_neighbour():
    refinement_level = 6
    center = 5000

    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([0.0, 1.0])
    half_width = np.deg2rad(10.0)

    along = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=_angular_selector(half_width),
        normalize=False,
        domain=domain,
    )
    opposite = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad + np.pi,
        kernel=_angular_selector(half_width),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert along[center_position] > 0.0
    assert opposite[center_position] == pytest.approx(0.0)


def test_max_distance_m_controls_support():
    refinement_level = 6
    center = 5000

    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([0.0, 5.0])
    kernel = _angular_selector(np.deg2rad(20.0))

    excluded = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 0.95,
        azimuth_rad=bearing_rad,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )
    included = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert excluded[center_position] == pytest.approx(0.0)
    assert included[center_position] == pytest.approx(5.0)


def test_across_base_pixel_boundary():
    refinement_level = 3
    npix = 12 * 4**refinement_level
    cells_per_base_pixel = 4**refinement_level
    pair = None

    for center in range(npix):
        center_base = center // cells_per_base_pixel
        for neighbour in _immediate_neighbours(center, refinement_level):
            neighbour_base = int(neighbour) // cells_per_base_pixel
            if neighbour_base != center_base:
                pair = (center, int(neighbour))
                break
        if pair is not None:
            break

    assert pair is not None
    center, neighbour = pair
    distance_m, bearing_rad = _pair_geometry(
        center,
        neighbour,
        refinement_level,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([0.0, 2.0])

    result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=_angular_selector(np.deg2rad(10.0)),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert result[center_position] == pytest.approx(2.0)


def test_longitude_wrap_direction():
    refinement_level = 6
    npix = 12 * 4**refinement_level
    pair = None

    for center in range(0, npix, max(1, npix // 20000)):
        neighbours = _immediate_neighbours(center, refinement_level)
        if neighbours.size == 0:
            continue

        center_lon, _ = _cell_centres(
            np.array([center], dtype=np.uint64),
            refinement_level,
        )
        neighbour_lon, _ = _cell_centres(
            neighbours,
            refinement_level,
        )

        difference = np.abs(neighbour_lon - center_lon[0])
        wrapped = difference > 180.0
        if np.any(wrapped):
            pair = (
                center,
                int(neighbours[np.where(wrapped)[0][0]]),
            )
            break

    if pair is None:
        pytest.skip(
            "Could not find a longitude-wrap neighbour pair "
            "with the deterministic search."
        )

    center, neighbour = pair
    distance_m, bearing_rad = _pair_geometry(
        center,
        neighbour,
        refinement_level,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)

    result = directional_filter(
        np.array([0.0, 3.0]),
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=_angular_selector(np.deg2rad(10.0)),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert result[center_position] == pytest.approx(3.0)


def test_high_latitude_directional_geometry():
    refinement_level = 6
    npix = 12 * 4**refinement_level
    candidate_cells = np.arange(npix, dtype=np.uint64)
    _, lat = _cell_centres(candidate_cells, refinement_level)

    center = int(candidate_cells[np.argmax(lat)])
    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)

    result = directional_filter(
        np.array([0.0, 4.0]),
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=_angular_selector(np.deg2rad(10.0)),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert np.isfinite(result[center_position])
    assert result[center_position] == pytest.approx(4.0)


def test_neighbour_outside_domain_does_not_contribute():
    refinement_level = 6
    center = 5000

    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    cell_ids = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([1.0, 1000.0])
    domain = np.array([center], dtype=np.uint64)

    result = directional_filter(
        values,
        cell_ids,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=lambda d, a: np.ones_like(d),
        normalize=False,
        domain=domain,
    )

    assert result[0] == pytest.approx(1.0)


def test_partial_domain_normalization_does_not_create_boundary_bias():
    refinement_level = 6
    center = 5000
    neighbours = _immediate_neighbours(center, refinement_level)
    assert neighbours.size >= 2

    cell_ids = np.concatenate(
        [np.array([center], dtype=np.uint64), neighbours]
    )
    values = np.full(cell_ids.size, 9.0)

    chosen_neighbour = int(neighbours[0])
    distance_m, bearing_rad = _pair_geometry(
        center,
        chosen_neighbour,
        refinement_level,
    )

    domain = np.array(
        [center, chosen_neighbour],
        dtype=np.uint64,
    )

    result = directional_filter(
        values,
        cell_ids,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=lambda d, a: np.ones_like(d),
        normalize=True,
        domain=domain,
    )

    np.testing.assert_allclose(result, np.array([9.0, 9.0]))


def test_opposite_azimuths_produce_different_response():
    refinement_level = 6
    center = 5000

    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([0.0, 1.0])
    kernel = _angular_selector(np.deg2rad(10.0))

    forward = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )
    backward = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad + np.pi,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert forward[center_position] == pytest.approx(1.0)
    assert backward[center_position] == pytest.approx(0.0)


@pytest.mark.parametrize("refinement_level", [4, 5, 6])
def test_physical_distance_threshold_at_multiple_levels(refinement_level):
    npix = 12 * 4**refinement_level
    center = min(1000, npix - 1)
    neighbours = _immediate_neighbours(center, refinement_level)
    assert neighbours.size > 0

    neighbour = int(neighbours[0])
    distance_m, bearing_rad = _pair_geometry(
        center,
        neighbour,
        refinement_level,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)
    values = np.array([0.0, 1.0])

    excluded = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 0.9,
        azimuth_rad=bearing_rad,
        kernel=lambda d, a: np.ones_like(d),
        normalize=False,
        domain=domain,
    )
    included = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.1,
        azimuth_rad=bearing_rad,
        kernel=lambda d, a: np.ones_like(d),
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)
    assert excluded[center_position] == pytest.approx(0.0)
    assert included[center_position] >= 1.0


def test_torch_directional_filter_preserves_autograd():
    refinement_level = 5
    center = 1000

    neighbour, distance_m, bearing_rad = _choose_directional_neighbour(
        center,
        refinement_level,
        0.0,
    )

    domain = np.array([center, neighbour], dtype=np.uint64)

    values = torch.tensor(
        [0.0, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=distance_m * 1.05,
        azimuth_rad=bearing_rad,
        kernel=_angular_selector(np.deg2rad(10.0)),
        normalize=False,
        domain=domain,
    )

    result.sum().backward()

    assert result.device == values.device
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))


def test_s2msi_style_physical_shadow_direction():
    """Validate physical displacement + geographical azimuth semantics."""
    refinement_level = 7
    center = 20000

    neighbour, shadow_distance_m, shadow_azimuth_rad = (
        _choose_directional_neighbour(
            center,
            refinement_level,
            np.deg2rad(135.0),
        )
    )

    domain = _make_domain_around_center(center, refinement_level)
    values = np.zeros(domain.size, dtype=np.float64)

    neighbour_position = np.where(domain == neighbour)[0]
    assert neighbour_position.size == 1
    values[neighbour_position[0]] = 1.0

    kernel = _distance_and_angle_selector(
        target_distance_m=shadow_distance_m,
        distance_tolerance_m=max(
            1.0,
            shadow_distance_m * 0.05,
        ),
        half_width_rad=np.deg2rad(10.0),
    )

    correct_direction = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=shadow_distance_m * 1.10,
        azimuth_rad=shadow_azimuth_rad,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )

    opposite_direction = directional_filter(
        values,
        domain,
        refinement_level,
        max_distance_m=shadow_distance_m * 1.10,
        azimuth_rad=shadow_azimuth_rad + np.pi,
        kernel=kernel,
        normalize=False,
        domain=domain,
    )

    center_position = _center_output_position(domain, center)

    assert correct_direction[center_position] == pytest.approx(1.0)
    assert opposite_direction[center_position] == pytest.approx(0.0)
