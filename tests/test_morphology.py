import numpy as np
import pytest

from healpix_geo import nested
from pyproj import Geod

from healpix_analyse.morphology import (
    binary_dilation,
    binary_erosion,
)

_WGS84 = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Real-HEALPix topology helpers
# ---------------------------------------------------------------------------
#
# The tests added below intentionally use real healpix-geo topology rather
# than monkeypatched neighbourhoods.  They protect morphology behaviour at
# places where Cartesian-grid assumptions are especially dangerous:
#
# - boundaries between the 12 HEALPix base pixels,
# - high-latitude / polar regions,
# - the longitude coordinate seam,
# - several refinement levels,
# - and partial processing-domain boundaries.
#
# These helpers are test infrastructure only.  They do not duplicate the
# morphology implementation itself.


def _immediate_neighbours(
    cell: int,
    refinement_level: int,
) -> np.ndarray:
    """Return the real immediate neighbours of one NESTED HEALPix cell."""

    neighbourhood = nested.kth_neighbourhood(
        np.array([cell], dtype=np.uint64),
        refinement_level,
        ring=1,
    )[0]

    neighbourhood = np.asarray(neighbourhood, dtype=np.int64)

    # kth_neighbourhood includes the input cell itself.  At the special
    # HEALPix topological locations there can be only seven neighbours; the
    # returned array can therefore also contain a negative sentinel.  Keep
    # only real neighbouring cell ids and remove the centre cell itself.
    neighbours = neighbourhood[
        (neighbourhood >= 0)
        & (neighbourhood != cell)
    ]

    return np.unique(neighbours.astype(np.uint64))


def _cell_center(
    cell: int,
    refinement_level: int,
) -> tuple[float, float]:
    """Return WGS84 longitude/latitude for one HEALPix cell centre."""

    lon, lat = nested.healpix_to_lonlat(
        np.array([cell], dtype=np.uint64),
        refinement_level,
        ellipsoid="WGS84",
    )

    return float(lon[0]), float(lat[0])


def _center_distance_m(
    first: int,
    second: int,
    refinement_level: int,
) -> float:
    """Return WGS84 geodesic distance between two HEALPix cell centres."""

    lon1, lat1 = _cell_center(first, refinement_level)
    lon2, lat2 = _cell_center(second, refinement_level)

    _, _, distance = _WGS84.inv(
        lon1,
        lat1,
        lon2,
        lat2,
    )

    return float(distance)


def _radius_reaching_neighbour(
    first: int,
    second: int,
    refinement_level: int,
) -> float:
    """Return a radius just larger than the centre distance of a cell pair."""

    distance = _center_distance_m(
        first,
        second,
        refinement_level,
    )

    # A small margin avoids equality/round-off ambiguity at the radius
    # boundary while keeping the test local to the selected neighbour pair.
    return distance * 1.01 + 1.0


def _find_base_pixel_crossing_pair(
    refinement_level: int,
) -> tuple[int, int]:
    """Find immediate neighbours belonging to different HEALPix base pixels."""

    cells_per_base_pixel = 4**refinement_level
    npix = 12 * cells_per_base_pixel

    for cell in range(npix):
        base_pixel = cell // cells_per_base_pixel

        for neighbour in _immediate_neighbours(
            cell,
            refinement_level,
        ):
            neighbour = int(neighbour)
            neighbour_base_pixel = (
                neighbour // cells_per_base_pixel
            )

            if neighbour_base_pixel != base_pixel:
                return cell, neighbour

    raise AssertionError(
        "Could not find a HEALPix base-pixel boundary pair."
    )


def _find_longitude_wrap_pair(
    refinement_level: int,
) -> tuple[int, int, float, float]:
    """Find immediate neighbours crossing the longitude coordinate seam."""

    npix = 12 * 4**refinement_level
    cells = np.arange(npix, dtype=np.uint64)

    lon, _ = nested.healpix_to_lonlat(
        cells,
        refinement_level,
        ellipsoid="WGS84",
    )

    for cell in range(npix):
        cell_lon = float(lon[cell])

        for neighbour in _immediate_neighbours(
            cell,
            refinement_level,
        ):
            neighbour = int(neighbour)
            neighbour_lon = float(lon[neighbour])

            # Geographic neighbours can have numerically very different
            # longitudes when they lie on opposite sides of the coordinate
            # seam, for example approximately 359 degrees and 1 degree.
            if abs(cell_lon - neighbour_lon) > 180.0:
                return (
                    cell,
                    neighbour,
                    cell_lon,
                    neighbour_lon,
                )

    raise AssertionError(
        "Could not find a longitude-wrap neighbour pair."
    )


def _cell_near_lonlat(
    longitude: float,
    latitude: float,
    refinement_level: int,
) -> int:
    """Return the NESTED HEALPix cell containing a WGS84 lon/lat point."""

    cell = nested.lonlat_to_healpix(
        np.array([longitude], dtype=np.float64),
        np.array([latitude], dtype=np.float64),
        refinement_level,
        ellipsoid="WGS84",
    )

    return int(cell[0])



def test_empty_dilation():
    cells = np.array([], dtype=np.uint64)

    result = binary_dilation(
        cells,
        radius=100.0,
        refinement_level=10,
    )

    assert result.dtype == np.uint64
    assert result.size == 0


def test_empty_erosion():
    cells = np.array([], dtype=np.uint64)

    result = binary_erosion(
        cells,
        radius=100.0,
        refinement_level=10,
    )

    assert result.dtype == np.uint64
    assert result.size == 0


def test_zero_radius_returns_original_cells():
    cells = np.array([5, 3, 5], dtype=np.uint64)

    dilated = binary_dilation(
        cells,
        radius=0.0,
        refinement_level=5,
    )

    eroded = binary_erosion(
        cells,
        radius=0.0,
        refinement_level=5,
    )

    expected = np.array([3, 5], dtype=np.uint64)

    np.testing.assert_array_equal(dilated, expected)
    np.testing.assert_array_equal(eroded, expected)


def test_negative_radius_raises():
    with pytest.raises(ValueError, match="radius"):
        binary_dilation(
            np.array([1], dtype=np.uint64),
            radius=-1.0,
            refinement_level=5,
        )


def test_invalid_refinement_level_raises():
    with pytest.raises(ValueError, match="refinement_level"):
        binary_dilation(
            np.array([1], dtype=np.uint64),
            radius=10.0,
            refinement_level=30,
        )


def test_invalid_neighbourhood_raises():
    with pytest.raises(ValueError, match="neighbourhood"):
        binary_dilation(
            np.array([1], dtype=np.uint64),
            radius=10.0,
            refinement_level=5,
            neighbourhood="invalid",
        )


def test_cells_must_be_one_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        binary_dilation(
            np.array([[1, 2]], dtype=np.uint64),
            radius=10.0,
            refinement_level=5,
        )


def test_domain_must_contain_active_cells():
    cells = np.array([1, 2], dtype=np.uint64)
    domain = np.array([1], dtype=np.uint64)

    with pytest.raises(ValueError, match="must belong"):
        binary_dilation(
            cells,
            radius=10.0,
            refinement_level=5,
            domain=domain,
        )


def test_domain_restricts_dilation(monkeypatch):
    cells = np.array([10], dtype=np.uint64)
    domain = np.array([10, 11], dtype=np.uint64)

    def fake_neighbourhoods(*args, **kwargs):
        return [
            np.array([10, 11, 12], dtype=np.uint64)
        ]

    monkeypatch.setattr(
        "healpix_analyse.morphology.build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = binary_dilation(
        cells,
        radius=100.0,
        refinement_level=5,
        domain=domain,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10, 11], dtype=np.uint64),
    )


def test_erosion_without_domain_requires_full_neighbourhood(monkeypatch):
    cells = np.array([10, 11], dtype=np.uint64)

    def fake_neighbourhoods(*args, **kwargs):
        return [
            np.array([10, 11], dtype=np.uint64),
            np.array([10, 11, 12], dtype=np.uint64),
        ]

    monkeypatch.setattr(
        "healpix_analyse.morphology.build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = binary_erosion(
        cells,
        radius=100.0,
        refinement_level=5,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10], dtype=np.uint64),
    )


def test_erosion_ignores_cells_outside_domain(monkeypatch):
    cells = np.array([10, 11], dtype=np.uint64)
    domain = np.array([10, 11], dtype=np.uint64)

    def fake_neighbourhoods(*args, **kwargs):
        return [
            np.array([10, 11, 12], dtype=np.uint64),
            np.array([10, 11, 12], dtype=np.uint64),
        ]

    monkeypatch.setattr(
        "healpix_analyse.morphology.build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = binary_erosion(
        cells,
        radius=100.0,
        refinement_level=5,
        domain=domain,
    )

    np.testing.assert_array_equal(
        result,
        np.array([10, 11], dtype=np.uint64),
    )
def test_real_cone_coverage_dilation():
    """Dilation using the real healpix-geo cone coverage."""

    cells = np.array([0], dtype=np.uint64)

    result = binary_dilation(
        cells,
        radius=500_000.0,
        refinement_level=3,
        neighbourhood="cone_coverage",
    )

    # Original active cell must always remain active.
    assert 0 in result

    # A sufficiently large dilation should contain more than
    # the original cell.
    assert result.size > 1

    # Result must remain a unique uint64 array.
    assert result.dtype == np.uint64
    assert np.unique(result).size == result.size


def test_real_cell_center_dilation():
    """Dilation using real WGS84 cell-centre distances."""

    cells = np.array([0], dtype=np.uint64)

    result = binary_dilation(
        cells,
        radius=500_000.0,
        refinement_level=3,
        neighbourhood="cell_center",
    )

    assert 0 in result
    assert result.size >= 1
    assert result.dtype == np.uint64
    assert np.unique(result).size == result.size


def test_cell_center_is_subset_of_cone_coverage():
    """Cell-centre neighbourhood must be contained in cone coverage."""

    cells = np.array([0], dtype=np.uint64)

    cone = binary_dilation(
        cells,
        radius=500_000.0,
        refinement_level=3,
        neighbourhood="cone_coverage",
    )

    center = binary_dilation(
        cells,
        radius=500_000.0,
        refinement_level=3,
        neighbourhood="cell_center",
    )

    assert set(center.tolist()).issubset(
        set(cone.tolist())
    )

def test_s2msi_cell_center_exact_counts():
    """Regression test for S2MSI morphology radii at refinement level 17."""

    cells = np.array([0], dtype=np.uint64)

    expected_counts = {
        180.0: 41,
        240.0: 73,
        480.0: 295,
    }

    for radius, expected_count in expected_counts.items():
        result = binary_dilation(
            cells,
            radius=radius,
            refinement_level=17,
            neighbourhood="cell_center",
        )

        print(
            f"radius={radius:5.0f} m | "
            f"cell_center={len(result):4d} cells"
        )

        assert len(result) == expected_count

def test_s2msi_compare_neighbourhood_methods():
    """Regression test for S2MSI morphology neighbourhoods.

    Compare the two supported structuring-neighbourhood definitions
    at HEALPix refinement level 17 using the physical radii required
    by the Sentinel-2 MSI Mask S2 processing.

    The expected counts are intentionally fixed so that changes in
    healpix-geo geometry or morphology behaviour are detected.
    """

    cells = np.array([0], dtype=np.uint64)

    expected_counts = {
        180.0: {
            "cell_center": 41,
            "cone_coverage": 63,
        },
        240.0: {
            "cell_center": 73,
            "cone_coverage": 99,
        },
        480.0: {
            "cell_center": 295,
            "cone_coverage": 339,
        },
    }

    print()
    print("S2MSI morphology neighbourhood comparison")
    print("refinement_level = 17")
    print()
    print(
        f"{'radius [m]':>10} "
        f"{'cell_center':>14} "
        f"{'cone_coverage':>15} "
        f"{'difference':>12}"
    )
    print("-" * 55)

    for radius, expected in expected_counts.items():
        cell_center = binary_dilation(
            cells,
            radius=radius,
            refinement_level=17,
            neighbourhood="cell_center",
        )

        cone_coverage = binary_dilation(
            cells,
            radius=radius,
            refinement_level=17,
            neighbourhood="cone_coverage",
        )

        difference = len(cone_coverage) - len(cell_center)

        print(
            f"{radius:10.0f} "
            f"{len(cell_center):14d} "
            f"{len(cone_coverage):15d} "
            f"{difference:12d}"
        )

        # Exact regression checks.
        assert len(cell_center) == expected["cell_center"]
        assert len(cone_coverage) == expected["cone_coverage"]

        # The centre-distance neighbourhood must be fully contained
        # within the coverage-based neighbourhood.
        assert set(cell_center.tolist()).issubset(
            set(cone_coverage.tolist())
        )

# ---------------------------------------------------------------------------
# HEALPix base-pixel boundary tests
# ---------------------------------------------------------------------------
#
# HEALPix consists of 12 base pixels.  NESTED cell ids belonging to different
# base pixels are not guaranteed to be numerically adjacent, so morphology
# must use the real HEALPix topology rather than assumptions about cell-id
# continuity.
#
# These tests explicitly select two immediate neighbours that lie on opposite
# sides of a base-pixel boundary.


def test_real_dilation_crosses_base_pixel_boundary():
    """Dilation must cross a real HEALPix base-pixel boundary."""

    refinement_level = 3

    first, second = _find_base_pixel_crossing_pair(
        refinement_level
    )

    cells_per_base_pixel = 4**refinement_level

    # Confirm that the selected pair really crosses a base-pixel boundary.
    assert (
        first // cells_per_base_pixel
    ) != (
        second // cells_per_base_pixel
    )

    radius = _radius_reaching_neighbour(
        first,
        second,
        refinement_level,
    )

    result = binary_dilation(
        np.array([first], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    # The physically neighbouring cell must be reached even though its
    # NESTED id belongs to another HEALPix base pixel.
    assert second in result


def test_real_erosion_across_base_pixel_boundary():
    """Erosion must evaluate a neighbourhood across a base-pixel boundary."""

    refinement_level = 3

    center, neighbour = _find_base_pixel_crossing_pair(
        refinement_level
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    # Construct exactly the active region produced by the real physical
    # neighbourhood around the centre.  This region crosses a HEALPix
    # base-pixel boundary.
    active = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    cells_per_base_pixel = 4**refinement_level
    center_base = center // cells_per_base_pixel

    # Ensure the structuring neighbourhood really spans another base pixel.
    assert any(
        int(cell) // cells_per_base_pixel != center_base
        for cell in active
    )

    eroded = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    # All cells in the centre's structuring neighbourhood are active,
    # therefore the centre must survive erosion.
    assert center in eroded


# ---------------------------------------------------------------------------
# Latitude robustness
# ---------------------------------------------------------------------------
#
# A physical-radius structuring element must have the same semantic meaning
# at the equator and at high latitude.
#
# We deliberately do NOT assert that the number of selected HEALPix cells is
# identical at every latitude.  HEALPix is equal-area, but cell shapes and
# centre configurations vary with position.  The invariant that matters here
# is physical distance: a real immediate neighbour whose centre lies within
# the requested WGS84 radius must be included.


@pytest.mark.parametrize(
    "latitude",
    [
        0.0,    # equatorial region
        80.0,   # northern high-latitude / polar region
        -80.0,  # southern high-latitude / polar region
    ],
)
def test_real_dilation_at_different_latitudes(latitude):
    """Cell-centre dilation must work at equatorial and polar latitudes."""

    refinement_level = 5

    center = _cell_near_lonlat(
        longitude=10.0,
        latitude=latitude,
        refinement_level=refinement_level,
    )

    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )

    assert neighbours.size >= 0

    neighbour = int(neighbours[0])

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    result = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert center in result
    assert neighbour in result


@pytest.mark.parametrize(
    "latitude",
    [
        0.0,
        80.0,
        -80.0,
    ],
)
def test_real_erosion_at_different_latitudes(latitude):
    """Erosion must use the same physical-neighbourhood rule at all latitudes."""

    refinement_level = 5

    center = _cell_near_lonlat(
        longitude=10.0,
        latitude=latitude,
        refinement_level=refinement_level,
    )

    neighbour = int(
        _immediate_neighbours(
            center,
            refinement_level,
        )[0]
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    active = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    eroded = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert center in eroded


# ---------------------------------------------------------------------------
# Longitude wrap-around
# ---------------------------------------------------------------------------
#
# Longitude is only a coordinate representation.  Cells close to the two
# sides of the longitude coordinate seam can be direct geographical
# neighbours.  Morphology must therefore not accidentally split the sphere at
# that numerical discontinuity.


def test_real_dilation_crosses_longitude_wrap():
    """Dilation must cross the longitude coordinate seam."""

    refinement_level = 4

    (
        first,
        second,
        first_lon,
        second_lon,
    ) = _find_longitude_wrap_pair(
        refinement_level
    )

    # Verify that the chosen pair genuinely lies across the numeric longitude
    # discontinuity.
    assert abs(first_lon - second_lon) > 180.0

    radius = _radius_reaching_neighbour(
        first,
        second,
        refinement_level,
    )

    result = binary_dilation(
        np.array([first], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert second in result


def test_real_erosion_crosses_longitude_wrap():
    """Erosion must not treat the longitude seam as a spatial boundary."""

    refinement_level = 4

    first, second, _, _ = _find_longitude_wrap_pair(
        refinement_level
    )

    radius = _radius_reaching_neighbour(
        first,
        second,
        refinement_level,
    )

    active = binary_dilation(
        np.array([first], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert second in active

    eroded = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert first in eroded


# ---------------------------------------------------------------------------
# Multiple refinement levels
# ---------------------------------------------------------------------------
#
# Morphology is expressed in metres, not in a fixed number of HEALPix cells.
# The same physical cell-centre criterion therefore has to remain valid as the
# grid is refined.  These tests exercise both very coarse and finer NESTED
# levels without hard-coding a level-specific cell count.


@pytest.mark.parametrize(
    "refinement_level",
    [
        0,
        1,
        3,
        6,
    ],
)
def test_real_dilation_multiple_refinement_levels(
    refinement_level,
):
    """Physical-radius dilation must work across multiple HEALPix levels.

    This test deliberately includes refinement level 0.

    At level 0, the HEALPix grid consists of the 12 base pixels themselves.
    These base pixels have only six immediate neighbours, unlike most
    higher-resolution HEALPix cells, which typically have seven or eight.

    The purpose of this test is therefore NOT to assume a fixed neighbour
    count. Instead, it verifies the geometry-independent invariant:

        if a real HEALPix neighbour lies within the requested physical
        radius, dilation must include that neighbour.

    This is the behaviour that must remain valid across refinement levels.
    """

    center = 0

    neighbours = _immediate_neighbours(
        center,
        refinement_level,
    )

    # Do not assume 7 or 8 neighbours here.
    #
    # In particular, refinement level 0 consists of the 12 HEALPix base
    # pixels, and those cells have six immediate neighbours.
    #
    # What matters for this test is simply that a genuine topological
    # neighbour exists and can be reached by a physical-radius dilation.
    assert neighbours.size > 0

    neighbour = int(
        neighbours[0]
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    result = binary_dilation(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert center in result
    assert neighbour in result




@pytest.mark.parametrize(
    "refinement_level",
    [0, 1, 3, 6],
)
def test_real_erosion_multiple_refinement_levels(
    refinement_level,
):
    """Physical-radius erosion must work across multiple HEALPix levels."""

    center = 0

    neighbour = int(
        _immediate_neighbours(
            center,
            refinement_level,
        )[0]
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    active = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    eroded = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert center in eroded


# ---------------------------------------------------------------------------
# Real partial-domain boundary semantics
# ---------------------------------------------------------------------------
#
# This is an important distinction between:
#
#   1. an inactive cell inside the processing domain, and
#   2. a cell that lies outside the processing domain entirely.
#
# For erosion, cells in case (1) must remove the centre cell.
# Cells in case (2) must NOT remove it.
#
# Conceptually:
#
#                    domain outside
#                         |
#                         v
#
#                    X X X
#                    X C | outside
#                    X X |
#
# ``C`` is an active cell touching the boundary of a regional dataset.
#
# The structuring neighbourhood geometrically continues beyond the dataset,
# but those outside cells are unknown / out of scope.  They must NOT be
# silently interpreted as False.
#
# Otherwise a regional HEALPix dataset would acquire an artificial eroded rim
# merely because data were not supplied outside its spatial extent.
#
# The tests below use REAL WGS84 HEALPix neighbourhoods.  No monkeypatching is
# involved.


def test_real_domain_boundary_does_not_create_artificial_erosion():
    """Cells outside an explicit domain must not behave like inactive cells."""

    refinement_level = 6

    center = _cell_near_lonlat(
        longitude=10.0,
        latitude=45.0,
        refinement_level=refinement_level,
    )

    neighbour = int(
        _immediate_neighbours(
            center,
            refinement_level,
        )[0]
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    # Obtain the real physical structuring neighbourhood around C.
    full_neighbourhood = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert full_neighbourhood.size > 1

    # Deliberately remove one genuine neighbourhood cell from the regional
    # processing domain.
    #
    # It still exists geometrically on Earth, but for this regional dataset it
    # is OUTSIDE rather than INACTIVE.
    outside_cell = next(
        int(cell)
        for cell in full_neighbourhood
        if int(cell) != center
    )

    domain = full_neighbourhood[
        full_neighbourhood != outside_cell
    ]

    # Every cell supplied inside the domain is active.
    active = domain.copy()

    assert center in active
    assert outside_cell not in domain

    # ------------------------------------------------------------------
    # Without an explicit domain:
    #
    # The missing outside_cell is interpreted as an inactive neighbour,
    # therefore C should be eroded.
    # ------------------------------------------------------------------
    eroded_without_domain = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert center not in eroded_without_domain

    # ------------------------------------------------------------------
    # With an explicit domain:
    #
    # outside_cell is outside the processing extent and therefore must be
    # ignored.  Every neighbourhood cell that DOES belong to domain is active,
    # so C must survive.
    #
    #                    domain outside
    #                         |
    #                         v
    #
    #                    X X X
    #                    X C | outside
    #                    X X |
    #
    # This is exactly the behaviour required for regional EO products.
    # ------------------------------------------------------------------
    eroded_with_domain = binary_erosion(
        active,
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
        domain=domain,
    )

    assert center in eroded_with_domain


def test_real_domain_boundary_restricts_dilation():
    """Dilation may not create active cells outside the processing domain."""

    refinement_level = 6

    center = _cell_near_lonlat(
        longitude=10.0,
        latitude=45.0,
        refinement_level=refinement_level,
    )

    neighbour = int(
        _immediate_neighbours(
            center,
            refinement_level,
        )[0]
    )

    radius = _radius_reaching_neighbour(
        center,
        neighbour,
        refinement_level,
    )

    full_dilation = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    assert full_dilation.size > 1

    outside_cell = next(
        int(cell)
        for cell in full_dilation
        if int(cell) != center
    )

    domain = full_dilation[
        full_dilation != outside_cell
    ]

    restricted = binary_dilation(
        np.array([center], dtype=np.uint64),
        radius=radius,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
        domain=domain,
    )

    # The excluded cell is geometrically inside the dilation radius but is
    # outside the regional processing domain, so it must not appear.
    assert outside_cell not in restricted

    # Dilation output must remain completely inside the explicit domain.
    assert set(restricted.tolist()).issubset(
        set(domain.tolist())
    )

