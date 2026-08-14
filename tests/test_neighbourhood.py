"""Tests for shared HEALPix neighbourhood geometry helpers.

The physical-radius and topological-ring neighbourhood definitions serve
different purposes:

- ``build_neighbourhoods`` selects cells by physical distance / coverage.
- ``build_ring_neighbourhoods`` selects cells by HEALPix topological
  distance.

The ring helper is used for local operators such as gradient estimation,
where the Cartesian source operation is based on immediate/local pixels
rather than on a prescribed physical radius.
"""

import importlib

import numpy as np
import pytest
from healpix_geo import nested

from healpix_analyse._neighbourhood import (
    build_metric_neighbourhood_geometry,
    build_neighbourhoods,
    build_relative_geometry,
    build_ring_neighbourhoods,
    metric_geometry_from_neighbourhoods,
    relative_geometry_from_neighbours,
    validate_ring,
)


def test_geod_thread_count_is_capped_at_eight(monkeypatch):
    module = importlib.import_module("healpix_analyse._neighbourhood")
    monkeypatch.setattr(module, "_GEOD_PARALLEL_MIN_PAIRS", 1)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 64)

    assert module._geod_thread_count(1_000_000) == 8


def test_parallel_wgs84_distance_is_bit_identical(monkeypatch):
    module = importlib.import_module("healpix_analyse._neighbourhood")
    longitude = np.linspace(-179.0, 179.0, 257, dtype=np.float64)
    latitude = np.linspace(-80.0, 80.0, 257, dtype=np.float64)

    monkeypatch.setattr(module, "_GEOD_PARALLEL_MIN_PAIRS", 1_000_000)
    serial = module._wgs84_distance(
        longitude,
        latitude,
        longitude[::-1].copy(),
        latitude[::-1].copy(),
    )

    monkeypatch.setattr(module, "_GEOD_PARALLEL_MIN_PAIRS", 1)
    monkeypatch.setattr(module.os, "cpu_count", lambda: 64)
    parallel = module._wgs84_distance(
        longitude,
        latitude,
        longitude[::-1].copy(),
        latitude[::-1].copy(),
    )

    np.testing.assert_array_equal(parallel, serial)


def test_fused_metric_geometry_matches_legacy_two_pass_pipeline():
    refinement_level = 5
    domain = build_ring_neighbourhoods(
        np.array([1000], dtype=np.uint64),
        refinement_level,
        ring=2,
        include_self=True,
    )[0][::-1].copy()
    radius_m = 500_000.0

    neighbourhoods = build_neighbourhoods(
        domain,
        radius_m,
        refinement_level,
        neighbourhood="cell_center",
        ellipsoid="WGS84",
    )
    legacy_neighbourhoods = [
        neighbours[np.isin(neighbours, domain)]
        for neighbours in neighbourhoods
    ]
    legacy = metric_geometry_from_neighbourhoods(
        domain,
        legacy_neighbourhoods,
        refinement_level,
    )

    fused = build_metric_neighbourhood_geometry(
        domain,
        radius_m,
        refinement_level,
    )

    legacy_counts = np.sum(legacy.valid_mask, axis=1)
    legacy_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(legacy_counts))
    )
    np.testing.assert_array_equal(fused.center_ids, legacy.center_ids)
    np.testing.assert_array_equal(
        fused.center_ids[fused.neighbour_indices],
        legacy.neighbour_ids[legacy.valid_mask],
    )
    np.testing.assert_array_equal(fused.row_offsets, legacy_offsets)
    np.testing.assert_array_equal(
        fused.distance_m,
        legacy.distance_m[legacy.valid_mask],
    )


# ---------------------------------------------------------------------------
# ring validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ring",
    [
        0,
        1,
        2,
        3,
        np.int64(1),
    ],
)
def test_validate_ring_accepts_non_negative_integers(
    ring,
):
    result = validate_ring(
        ring
    )

    assert isinstance(
        result,
        int,
    )

    assert result >= 0


@pytest.mark.parametrize(
    "ring",
    [
        -1,
        -2,
    ],
)
def test_validate_ring_rejects_negative_values(
    ring,
):
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        validate_ring(
            ring
        )


@pytest.mark.parametrize(
    "ring",
    [
        True,
        False,
        1.0,
        1.5,
        "1",
        None,
    ],
)
def test_validate_ring_rejects_non_integer_values(
    ring,
):
    with pytest.raises(
        TypeError,
        match="non-negative integer",
    ):
        validate_ring(
            ring
        )


# ---------------------------------------------------------------------------
# basic topological semantics
# ---------------------------------------------------------------------------


def test_ring_zero_without_self_is_empty():
    """ring=0 contains no neighbours after removing the centre cell."""

    result = build_ring_neighbourhoods(
        np.array(
            [0],
            dtype=np.uint64,
        ),
        refinement_level=3,
        ring=0,
        include_self=False,
    )

    assert len(result) == 1
    assert result[0].size == 0


def test_ring_zero_with_self_contains_center():
    """ring=0 represents the centre itself when include_self=True."""

    center = 42

    result = build_ring_neighbourhoods(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        refinement_level=3,
        ring=0,
        include_self=True,
    )

    np.testing.assert_array_equal(
        result[0],
        np.array(
            [center],
            dtype=np.uint64,
        ),
    )


def test_ring_one_excludes_center_by_default():
    """Immediate-neighbour output must not contain the centre cell."""

    center = 42

    result = build_ring_neighbourhoods(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        refinement_level=3,
        ring=1,
    )

    neighbours = result[0]

    assert center not in neighbours
    assert neighbours.size > 0


def test_ring_one_with_self_contains_center():
    """include_self=True explicitly restores the centre cell."""

    center = 42

    result = build_ring_neighbourhoods(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        refinement_level=3,
        ring=1,
        include_self=True,
    )

    assert center in result[0]


# ---------------------------------------------------------------------------
# Do not assume a fixed neighbour count
# ---------------------------------------------------------------------------


def test_level_zero_neighbourhood_does_not_assume_seven_or_eight():
    """Base-pixel topology must work without a fixed neighbour-count rule.

    At refinement level 0 the HEALPix grid consists of the 12 base pixels.

    These cells are a useful regression case because their topology differs
    from the common higher-resolution assumption that every cell has seven
    or eight immediate neighbours.

    The helper should simply return every valid neighbour supplied by the
    HEALPix topology backend.
    """

    result = build_ring_neighbourhoods(
        np.array(
            [0],
            dtype=np.uint64,
        ),
        refinement_level=0,
        ring=1,
    )

    neighbours = result[0]

    assert neighbours.size > 0

    assert np.all(
        neighbours < 12
    )

    assert 0 not in neighbours


# ---------------------------------------------------------------------------
# ring expansion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refinement_level",
    [
        1,
        3,
        6,
    ],
)
def test_ring_two_contains_ring_one(
    refinement_level,
):
    """Increasing topological radius must retain immediate neighbours."""

    center = 0

    ring_one = (
        build_ring_neighbourhoods(
            np.array(
                [center],
                dtype=np.uint64,
            ),
            refinement_level,
            ring=1,
        )[0]
    )

    ring_two = (
        build_ring_neighbourhoods(
            np.array(
                [center],
                dtype=np.uint64,
            ),
            refinement_level,
            ring=2,
        )[0]
    )

    assert set(
        ring_one.tolist()
    ).issubset(
        set(
            ring_two.tolist()
        )
    )

    assert ring_two.size >= ring_one.size


def test_ring_two_at_level_zero_reports_backend_limitation():
    """healpix-geo currently cannot build ring=2 at refinement level 0.

    This is not a failure of the wrapper.

    At refinement level 0 the cells are the 12 HEALPix base pixels
    themselves. Building a two-ring neighbourhood can require crossing
    base-cell boundaries more than once, which the current healpix-geo
    implementation explicitly rejects.

    Keeping this test documents that backend limitation instead of hiding
    it behind a misleading fixed-ring assumption.
    """

    with pytest.raises(
        ValueError,
        match="Crossing base cell boundaries more than once",
    ):
        build_ring_neighbourhoods(
            np.array(
                [0],
                dtype=np.uint64,
            ),
            refinement_level=0,
            ring=2,
        )

def test_level_zero_cell_zero_has_six_immediate_neighbours():
    """Document the real level-0 HEALPix base-pixel topology.

    Cell 0 has six valid immediate neighbours in the current
    healpix-geo topology.

    This protects downstream code from assuming that every HEALPix cell
    necessarily has seven or eight neighbours.
    """

    result = build_ring_neighbourhoods(
        np.array(
            [0],
            dtype=np.uint64,
        ),
        refinement_level=0,
        ring=1,
    )

    neighbours = result[0]

    assert neighbours.size == 6

    np.testing.assert_array_equal(
        neighbours,
        np.array(
            [1, 2, 3, 4, 5, 8],
            dtype=np.uint64,
        ),
    )

def test_missing_neighbour_slots_are_removed():
    """The -1 sentinel returned by healpix-geo must never leak to callers."""

    result = build_ring_neighbourhoods(
        np.array(
            [0],
            dtype=np.uint64,
        ),
        refinement_level=0,
        ring=1,
    )

    neighbours = result[0]

    assert np.all(
        neighbours >= 0
    )

    assert np.all(
        neighbours < 12
    )

# ---------------------------------------------------------------------------
# vectorisation
# ---------------------------------------------------------------------------


def test_multiple_cells_are_processed_in_one_call():
    """The helper must support vectorised HEALPix neighbourhood discovery."""

    cells = np.array(
        [
            0,
            1,
            42,
            100,
        ],
        dtype=np.uint64,
    )

    result = build_ring_neighbourhoods(
        cells,
        refinement_level=3,
        ring=1,
    )

    assert len(result) == len(
        cells
    )

    for center, neighbours in zip(
        cells,
        result,
        strict=True,
    ):
        assert center not in neighbours
        assert neighbours.size > 0
        assert neighbours.dtype == np.uint64


# ---------------------------------------------------------------------------
# Comparison with the underlying healpix-geo topology
# ---------------------------------------------------------------------------


def test_wrapper_matches_healpix_geo_valid_cell_ids():
    """Wrapper output must contain exactly the valid backend cell IDs.

    We intentionally compare sets rather than positional ordering.

    Gradient estimation needs the real neighbouring cells, not a backend
    positional convention that might be mistaken for geographic direction.
    """

    refinement_level = 3

    cells = np.array(
        [
            0,
            42,
            100,
        ],
        dtype=np.uint64,
    )

    ring = 1

    wrapped = build_ring_neighbourhoods(
        cells,
        refinement_level,
        ring=ring,
        include_self=False,
    )

    raw = np.asarray(
        nested.kth_neighbourhood(
            cells,
            refinement_level,
            ring,
        )
    )

    number_of_pixels = (
        12
        * 4**refinement_level
    )

    for center, row, actual in zip(
        cells,
        raw,
        wrapped,
        strict=True,
    ):
        expected = row[
            (row >= 0)
            & (
                row
                < number_of_pixels
            )
            & (
                row
                != center
            )
        ].astype(
            np.uint64,
            copy=False,
        )

        expected = np.unique(
            expected
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )


# ---------------------------------------------------------------------------
# HEALPix base-pixel boundaries
# ---------------------------------------------------------------------------


def test_ring_neighbourhood_can_cross_base_pixel_boundary():
    """Topological neighbours must not be restricted by NESTED base pixels."""

    refinement_level = 3

    cells_per_base_pixel = (
        4**refinement_level
    )

    number_of_pixels = (
        12
        * cells_per_base_pixel
    )

    found_crossing = False

    for center in range(
        number_of_pixels
    ):
        neighbours = (
            build_ring_neighbourhoods(
                np.array(
                    [center],
                    dtype=np.uint64,
                ),
                refinement_level,
                ring=1,
            )[0]
        )

        center_base_pixel = (
            center
            // cells_per_base_pixel
        )

        if any(
            int(neighbour)
            // cells_per_base_pixel
            != center_base_pixel
            for neighbour
            in neighbours
        ):
            found_crossing = True
            break

    assert found_crossing

# ---------------------------------------------------------------------------
# Relative geographic geometry
# ---------------------------------------------------------------------------


def test_relative_geometry_shape_matches_ring_neighbourhood():
    """Relative geometry must use one dense row per centre cell."""
    cells = np.array([0, 1, 42], dtype=np.uint64)
    geometry = build_relative_geometry(
        cells,
        refinement_level=3,
        ring=1,
    )

    assert geometry.center_ids.shape == (3,)
    assert geometry.neighbour_ids.ndim == 2
    assert (
        geometry.neighbour_ids.shape
        == geometry.valid_mask.shape
        == geometry.distance_m.shape
        == geometry.azimuth_rad.shape
        == geometry.east_offset_m.shape
        == geometry.north_offset_m.shape
    )
    np.testing.assert_array_equal(geometry.center_ids, cells)


def test_relative_geometry_valid_entries_are_finite():
    """Every real neighbour must have finite geographic geometry."""
    geometry = build_relative_geometry(
        np.array([0, 1, 42], dtype=np.uint64),
        refinement_level=3,
        ring=1,
    )
    valid = geometry.valid_mask

    assert np.all(np.isfinite(geometry.distance_m[valid]))
    assert np.all(np.isfinite(geometry.azimuth_rad[valid]))
    assert np.all(np.isfinite(geometry.east_offset_m[valid]))
    assert np.all(np.isfinite(geometry.north_offset_m[valid]))
    assert np.all(geometry.distance_m[valid] > 0.0)


def test_relative_geometry_padding_is_nan():
    """Missing neighbour positions must not acquire fake geometry."""
    # Use the low-level primitive directly so padding behaviour is tested
    # independently of the variable-length ring wrapper.
    geometry = relative_geometry_from_neighbours(
        np.array([0], dtype=np.uint64),
        np.array([[1, 2, -1, -1]], dtype=np.int64),
        refinement_level=0,
    )
    invalid = ~geometry.valid_mask

    assert np.any(invalid)
    assert np.all(geometry.neighbour_ids[invalid] == -1)
    assert np.all(np.isnan(geometry.distance_m[invalid]))
    assert np.all(np.isnan(geometry.azimuth_rad[invalid]))
    assert np.all(np.isnan(geometry.east_offset_m[invalid]))
    assert np.all(np.isnan(geometry.north_offset_m[invalid]))


def test_relative_offsets_reconstruct_geodesic_distance():
    """East/North components must reconstruct the geodesic distance."""
    geometry = build_relative_geometry(
        np.array([42], dtype=np.uint64),
        refinement_level=5,
        ring=1,
    )
    valid = geometry.valid_mask
    reconstructed_distance = np.hypot(
        geometry.east_offset_m[valid],
        geometry.north_offset_m[valid],
    )

    np.testing.assert_allclose(
        reconstructed_distance,
        geometry.distance_m[valid],
        rtol=1e-12,
        atol=1e-8,
    )


def test_relative_offsets_reconstruct_geographic_azimuth():
    """East/North offsets must preserve geographic forward azimuth."""
    geometry = build_relative_geometry(
        np.array([42], dtype=np.uint64),
        refinement_level=5,
        ring=1,
    )
    valid = geometry.valid_mask
    reconstructed_azimuth = np.arctan2(
        geometry.east_offset_m[valid],
        geometry.north_offset_m[valid],
    )

    np.testing.assert_allclose(
        reconstructed_azimuth,
        geometry.azimuth_rad[valid],
        rtol=0.0,
        atol=1e-12,
    )


def test_relative_geometry_handles_longitude_wrap():
    """Longitude 0/360 discontinuity must not affect local geometry."""
    refinement_level = 4
    number_of_pixels = 12 * 4**refinement_level
    cells = np.arange(number_of_pixels, dtype=np.uint64)
    lon, _ = nested.healpix_to_lonlat(
        cells,
        refinement_level,
        ellipsoid="WGS84",
    )

    crossing_pair = None
    for center in cells:
        neighbourhood = build_ring_neighbourhoods(
            np.array([center], dtype=np.uint64),
            refinement_level,
            ring=1,
        )[0]
        for neighbour in neighbourhood:
            if abs(float(lon[int(center)]) - float(lon[int(neighbour)])) > 180.0:
                crossing_pair = (int(center), int(neighbour))
                break
        if crossing_pair is not None:
            break

    assert crossing_pair is not None
    center, neighbour = crossing_pair
    geometry = relative_geometry_from_neighbours(
        np.array([center], dtype=np.uint64),
        np.array([[neighbour]], dtype=np.int64),
        refinement_level,
    )

    assert geometry.valid_mask[0, 0]
    assert np.isfinite(geometry.distance_m[0, 0])
    assert geometry.distance_m[0, 0] < 2_000_000.0


def test_relative_geometry_crosses_base_pixel_boundary():
    """Relative geometry must work across NESTED HEALPix base pixels."""
    refinement_level = 3
    cells_per_base_pixel = 4**refinement_level
    number_of_pixels = 12 * cells_per_base_pixel
    crossing_pair = None

    for center in range(number_of_pixels):
        neighbours = build_ring_neighbourhoods(
            np.array([center], dtype=np.uint64),
            refinement_level,
            ring=1,
        )[0]
        center_base = center // cells_per_base_pixel
        for neighbour in neighbours:
            neighbour = int(neighbour)
            if neighbour // cells_per_base_pixel != center_base:
                crossing_pair = (center, neighbour)
                break
        if crossing_pair is not None:
            break

    assert crossing_pair is not None
    center, neighbour = crossing_pair
    geometry = relative_geometry_from_neighbours(
        np.array([center], dtype=np.uint64),
        np.array([[neighbour]], dtype=np.int64),
        refinement_level,
    )

    assert geometry.valid_mask[0, 0]
    assert geometry.distance_m[0, 0] > 0.0
    assert np.isfinite(geometry.azimuth_rad[0, 0])


@pytest.mark.parametrize("latitude", [0.0, 80.0, -80.0])
def test_relative_geometry_at_different_latitudes(latitude):
    """Relative East/North geometry must remain valid at high latitude."""
    refinement_level = 6
    center = nested.lonlat_to_healpix(
        np.array([10.0], dtype=np.float64),
        np.array([latitude], dtype=np.float64),
        refinement_level,
        ellipsoid="WGS84",
    )[0]
    geometry = build_relative_geometry(
        np.array([center], dtype=np.uint64),
        refinement_level,
        ring=1,
    )
    valid = geometry.valid_mask

    assert np.any(valid)
    assert np.all(np.isfinite(geometry.distance_m[valid]))
    assert np.all(np.isfinite(geometry.east_offset_m[valid]))
    assert np.all(np.isfinite(geometry.north_offset_m[valid]))


def test_relative_geometry_accepts_explicit_neighbour_matrix():
    """The low-level primitive must not depend on ring discovery."""
    refinement_level = 3
    center = np.array([0], dtype=np.uint64)
    ring_neighbours = build_ring_neighbourhoods(
        center,
        refinement_level,
        ring=1,
    )[0]
    selected = ring_neighbours[:3]

    padded = np.full((1, 5), -1, dtype=np.int64)
    padded[0, : selected.size] = selected.astype(np.int64)
    geometry = relative_geometry_from_neighbours(
        center,
        padded,
        refinement_level,
    )

    assert geometry.valid_mask.sum() == selected.size
    np.testing.assert_array_equal(
        geometry.neighbour_ids[geometry.valid_mask],
        selected.astype(np.int64),
    )


def test_relative_geometry_rejects_invalid_padding_values():
    """Only -1 is accepted as a missing-neighbour sentinel."""
    with pytest.raises(
        ValueError,
        match="valid cell IDs or -1 padding",
    ):
        relative_geometry_from_neighbours(
            np.array([0], dtype=np.uint64),
            np.array([[1, -2]], dtype=np.int64),
            refinement_level=3,
        )


def test_relative_geometry_rejects_non_wgs84_for_now():
    """Keep ellipsoid semantics explicit until another backend is supported."""
    with pytest.raises(
        NotImplementedError,
        match="WGS84",
    ):
        relative_geometry_from_neighbours(
            np.array([0], dtype=np.uint64),
            np.array([[1]], dtype=np.int64),
            refinement_level=3,
            ellipsoid="GRS80",
        )
