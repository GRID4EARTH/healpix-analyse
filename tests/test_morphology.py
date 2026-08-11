import numpy as np
import pytest

from healpix_analyse.morphology import (
    binary_dilation,
    binary_erosion,
)


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
