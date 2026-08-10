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
        "healpix_analyse.morphology._neighbourhoods",
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
        "healpix_analyse.morphology._neighbourhoods",
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
        "healpix_analyse.morphology._neighbourhoods",
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
