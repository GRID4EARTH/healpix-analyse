"""Tests for generic HEALPix neighbourhood reductions."""

import importlib

import numpy as np
import pytest
import torch

from healpix_geo import nested

neighbour_reduce_module = importlib.import_module(
    "healpix_analyse.neighbour_reduce"
)
from healpix_analyse.morphology import binary_dilation
from healpix_analyse.neighbour_reduce import (
    HealPixNeighbourReducer,
    max_filter,
    mean_filter,
    median_filter,
    min_filter,
    neighbour_reduce,
)


# ---------------------------------------------------------------------------
# Helpers / deterministic fake geometry
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_geometry(
    monkeypatch,
):
    """Four-cell deterministic neighbourhood geometry.

    Neighbourhoods:

        10 -> [10, 11]
        11 -> [10, 11, 12]
        12 -> [11, 12, 13]
        13 -> [12, 13]
    """

    mapping = {
        10: [10, 11],
        11: [10, 11, 12],
        12: [11, 12, 13],
        13: [12, 13],
    }

    def fake_neighbourhoods(
        cells,
        radius,
        refinement_level,
        *,
        neighbourhood,
        ellipsoid,
    ):
        del radius
        del refinement_level
        del neighbourhood
        del ellipsoid

        return [
            np.asarray(
                mapping[int(cell)],
                dtype=np.uint64,
            )
            for cell in cells
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    return np.array(
        [10, 11, 12, 13],
        dtype=np.uint64,
    )


# ---------------------------------------------------------------------------
# Basic reductions
# ---------------------------------------------------------------------------


def test_mean(
    simple_geometry,
):
    values = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
    )

    np.testing.assert_allclose(
        result,
        [1.5, 2.0, 3.0, 3.5],
    )


def test_sum(
    simple_geometry,
):
    values = np.array(
        [1, 2, 3, 4],
        dtype=np.int16,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="sum",
    )

    np.testing.assert_array_equal(
        result,
        [3, 6, 9, 7],
    )

    assert result.dtype == np.int64


def test_count(
    simple_geometry,
):
    values = np.zeros(
        4,
        dtype=np.float32,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="count",
    )

    np.testing.assert_array_equal(
        result,
        [2, 3, 3, 2],
    )

    assert result.dtype == np.int64


def test_min_and_max(
    simple_geometry,
):
    values = np.array(
        [4, 2, 7, 1],
        dtype=np.int16,
    )

    minimum = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="min",
    )

    maximum = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="max",
    )

    np.testing.assert_array_equal(
        minimum,
        [2, 2, 1, 1],
    )

    np.testing.assert_array_equal(
        maximum,
        [4, 7, 7, 7],
    )

    assert minimum.dtype == np.int16
    assert maximum.dtype == np.int16


def test_std_population_semantics(
    simple_geometry,
):
    values = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="std",
    )

    expected = np.array(
        [
            np.std([1.0, 2.0]),
            np.std([1.0, 2.0, 3.0]),
            np.std([2.0, 3.0, 4.0]),
            np.std([3.0, 4.0]),
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


# ---------------------------------------------------------------------------
# Boolean reductions
# ---------------------------------------------------------------------------


def test_boolean_any_and_all(
    simple_geometry,
):
    values = np.array(
        [False, True, False, True],
        dtype=bool,
    )

    any_result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="any",
    )

    all_result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="all",
    )

    np.testing.assert_array_equal(
        any_result,
        [True, True, True, True],
    )

    np.testing.assert_array_equal(
        all_result,
        [False, False, False, False],
    )

    assert any_result.dtype == np.bool_
    assert all_result.dtype == np.bool_


def test_any_requires_boolean_input(
    simple_geometry,
):
    values = np.array(
        [0, 1, 0, 1],
        dtype=np.int16,
    )

    with pytest.raises(
        TypeError,
        match="boolean",
    ):
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="any",
        )


def test_all_requires_boolean_input(
    simple_geometry,
):
    values = np.array(
        [0, 1, 0, 1],
        dtype=np.int16,
    )

    with pytest.raises(
        TypeError,
        match="boolean",
    ):
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="all",
        )


# ---------------------------------------------------------------------------
# Median and categorical mode
# ---------------------------------------------------------------------------


def test_even_numerical_median_uses_mean_of_central_values(
    monkeypatch,
):
    cell_ids = np.array(
        [10, 11, 12, 13],
        dtype=np.uint64,
    )

    full_neighbourhood = np.array(
        [10, 11, 12, 13],
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            full_neighbourhood.copy()
            for _ in cells
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    values = np.array(
        [1, 1, 9, 9],
        dtype=np.int16,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="median",
    )

    # Sorted values:
    #
    #   [1, 1, 9, 9]
    #
    # Numerical median:
    #
    #   (1 + 9) / 2 = 5
    np.testing.assert_array_equal(
        result,
        [5.0, 5.0, 5.0, 5.0],
    )

    assert result.dtype == np.float64


def test_median_and_mode_are_distinct(
    monkeypatch,
):
    cell_ids = np.array(
        [10, 11, 12, 13],
        dtype=np.uint64,
    )

    full_neighbourhood = np.array(
        [10, 11, 12, 13],
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            full_neighbourhood.copy()
            for _ in cells
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    values = np.array(
        [1, 1, 9, 9],
        dtype=np.int16,
    )

    median = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="median",
    )

    mode = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="mode",
    )

    np.testing.assert_array_equal(
        median,
        [5.0, 5.0, 5.0, 5.0],
    )

    # There is a tie between categories 1 and 9.
    # The documented deterministic rule selects the smaller category.
    np.testing.assert_array_equal(
        mode,
        [1, 1, 1, 1],
    )


def test_mode_rejects_floating_input(
    simple_geometry,
):
    values = np.array(
        [1.0, 1.0, 2.0, 2.0],
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError,
        match="mode",
    ):
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="mode",
        )


def test_cartesian_3x3_numerical_median_semantics(
    monkeypatch,
):
    """Regression for a classical numerical 3x3 median.

    Cartesian source example:

        1    2   100
        3    4     5
        6    7     8

    Sorted:

        1, 2, 3, 4, 5, 6, 7, 8, 100

    Median = 5.

    The test isolates reduction semantics from HEALPix geometry.
    """

    cell_ids = np.arange(
        9,
        dtype=np.uint64,
    )

    full_neighbourhood = np.arange(
        9,
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            full_neighbourhood.copy()
            for _ in cells
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    values = np.array(
        [
            1,
            2,
            100,
            3,
            4,
            5,
            6,
            7,
            8,
        ],
        dtype=np.int16,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=1,
        radius_m=100.0,
        reduction="median",
    )

    np.testing.assert_array_equal(
        result,
        np.full(
            9,
            5.0,
        ),
    )


# ---------------------------------------------------------------------------
# Domain contract
# ---------------------------------------------------------------------------


def test_domain_must_be_subset_of_cell_ids():
    cell_ids = np.array(
        [10, 11, 12],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0, 6.0],
    )

    with pytest.raises(
        ValueError,
        match="domain",
    ):
        neighbour_reduce(
            values,
            cell_ids,
            refinement_level=5,
            radius_m=100.0,
            reduction="mean",
            domain=np.array(
                [10, 13],
                dtype=np.uint64,
            ),
        )


def test_values_outside_domain_do_not_participate(
    monkeypatch,
):
    cell_ids = np.array(
        [10, 11, 12],
        dtype=np.uint64,
    )

    # Deliberately extreme value outside the processing domain.
    values = np.array(
        [2.0, 4.0, 1000.0],
    )

    domain = np.array(
        [10, 11],
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        radius,
        refinement_level,
        *,
        neighbourhood,
        ellipsoid,
    ):
        del radius
        del refinement_level
        del neighbourhood
        del ellipsoid

        np.testing.assert_array_equal(
            cells,
            domain,
        )

        return [
            np.array(
                [10],
                dtype=np.uint64,
            ),
            np.array(
                [10, 11, 12],
                dtype=np.uint64,
            ),
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
        domain=domain,
    )

    # For target 11:
    #
    # geometrical neighbourhood = [10, 11, 12]
    # processing domain         = [10, 11]
    # effective neighbourhood   = [10, 11]
    #
    # Cell 12 and its value 1000 are absent from the reduction.
    np.testing.assert_allclose(
        result,
        [2.0, 3.0],
    )


def test_output_follows_exact_domain_order(
    monkeypatch,
):
    cell_ids = np.array(
        [10, 11, 12],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0, 8.0],
    )

    domain = np.array(
        [12, 10],
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        np.testing.assert_array_equal(
            cells,
            domain,
        )

        return [
            np.array(
                [12],
                dtype=np.uint64,
            ),
            np.array(
                [10],
                dtype=np.uint64,
            ),
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
        domain=domain,
    )

    np.testing.assert_array_equal(
        result,
        [8.0, 2.0],
    )


def test_domain_none_means_all_cell_ids_and_preserves_order(
    monkeypatch,
):
    cell_ids = np.array(
        [12, 10, 11],
        dtype=np.uint64,
    )

    values = np.array(
        [8.0, 2.0, 4.0],
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        np.testing.assert_array_equal(
            cells,
            cell_ids,
        )

        return [
            np.array(
                [12],
                dtype=np.uint64,
            ),
            np.array(
                [10],
                dtype=np.uint64,
            ),
            np.array(
                [11],
                dtype=np.uint64,
            ),
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
        domain=None,
    )

    np.testing.assert_array_equal(
        result,
        values,
    )


def test_output_cell_ids_property_preserves_domain_order(
    monkeypatch,
):
    cell_ids = np.array(
        [10, 11, 12],
        dtype=np.uint64,
    )

    domain = np.array(
        [12, 10],
        dtype=np.uint64,
    )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            np.asarray(
                [cell],
                dtype=np.uint64,
            )
            for cell in cells
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    reducer = HealPixNeighbourReducer(
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        domain=domain,
    )

    np.testing.assert_array_equal(
        reducer.output_cell_ids,
        domain,
    )


def test_empty_explicit_domain():
    cell_ids = np.array(
        [10, 11],
        dtype=np.uint64,
    )

    values = np.array(
        [1.0, 2.0],
    )

    domain = np.array(
        [],
        dtype=np.uint64,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
        domain=domain,
    )

    assert result.shape == (0,)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# include_self and empty effective neighbourhoods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("sum", 0),
        ("count", 0),
        ("any", False),
        ("all", True),
    ],
)
def test_empty_neighbourhood_identity_reductions(
    monkeypatch,
    reduction,
    expected,
):
    cell_ids = np.array(
        [10],
        dtype=np.uint64,
    )

    if reduction in {
        "any",
        "all",
    }:
        values = np.array(
            [True],
            dtype=bool,
        )
    else:
        values = np.array(
            [5],
            dtype=np.int64,
        )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            np.array(
                [10],
                dtype=np.uint64,
            )
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
        include_self=False,
    )

    assert result[0] == expected


@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "median",
        "min",
        "max",
        "std",
        "mode",
    ],
)
def test_empty_neighbourhood_raises_for_undefined_reductions(
    monkeypatch,
    reduction,
):
    cell_ids = np.array(
        [10],
        dtype=np.uint64,
    )

    if reduction == "mode":
        values = np.array(
            [5],
            dtype=np.int64,
        )
    else:
        values = np.array(
            [5.0],
        )

    def fake_neighbourhoods(
        cells,
        *args,
        **kwargs,
    ):
        return [
            np.array(
                [10],
                dtype=np.uint64,
            )
        ]

    monkeypatch.setattr(
        neighbour_reduce_module,
        "build_neighbourhoods",
        fake_neighbourhoods,
    )

    with pytest.raises(
        ValueError,
        match="empty neighbourhood",
    ):
        neighbour_reduce(
            values,
            cell_ids,
            refinement_level=5,
            radius_m=100.0,
            reduction=reduction,
            include_self=False,
        )


# ---------------------------------------------------------------------------
# Zero radius / single cells / constant fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "min",
        "max",
        "median",
    ],
)
def test_zero_radius_numerical_reductions_are_identity(
    reduction,
):
    cell_ids = np.array(
        [100, 101, 102],
        dtype=np.uint64,
    )

    values = np.array(
        [3.0, 7.0, -2.0],
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=0.0,
        reduction=reduction,
    )

    np.testing.assert_array_equal(
        result,
        values,
    )


def test_zero_radius_count_is_one():
    cell_ids = np.array(
        [100, 101, 102],
        dtype=np.uint64,
    )

    values = np.array(
        [3.0, 7.0, -2.0],
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=0.0,
        reduction="count",
    )

    np.testing.assert_array_equal(
        result,
        [1, 1, 1],
    )


def test_single_cell_domain():
    cell_ids = np.array(
        [10],
        dtype=np.uint64,
    )

    values = np.array(
        [42.0],
    )

    mean = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=0.0,
        reduction="mean",
    )

    count = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=0.0,
        reduction="count",
    )

    std = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=5,
        radius_m=0.0,
        reduction="std",
    )

    np.testing.assert_array_equal(
        mean,
        [42.0],
    )

    np.testing.assert_array_equal(
        count,
        [1],
    )

    np.testing.assert_array_equal(
        std,
        [0.0],
    )


def test_constant_field_stays_constant(
    simple_geometry,
):
    values = np.full(
        4,
        17.5,
        dtype=np.float64,
    )

    for reduction in [
        "mean",
        "min",
        "max",
        "median",
    ]:
        result = neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction=reduction,
        )

        np.testing.assert_allclose(
            result,
            17.5,
        )


# ---------------------------------------------------------------------------
# Impulse behaviour
# ---------------------------------------------------------------------------


def test_isolated_impulse_mean(
    simple_geometry,
):
    values = np.array(
        [0.0, 0.0, 1.0, 0.0],
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
    )

    np.testing.assert_allclose(
        result,
        [
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.5,
        ],
    )


def test_isolated_impulse_max(
    simple_geometry,
):
    values = np.array(
        [0.0, 0.0, 1.0, 0.0],
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="max",
    )

    np.testing.assert_array_equal(
        result,
        [0.0, 1.0, 1.0, 1.0],
    )


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_dtype",
    [
        np.int16,
        np.int32,
        np.int64,
    ],
)
@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "median",
        "std",
    ],
)
def test_integer_numerical_reductions_return_float64(
    simple_geometry,
    input_dtype,
    reduction,
):
    values = np.array(
        [1, 2, 3, 4],
        dtype=input_dtype,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "input_dtype",
    [
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "median",
        "std",
        "sum",
        "min",
        "max",
    ],
)
def test_floating_reductions_preserve_dtype(
    simple_geometry,
    input_dtype,
    reduction,
):
    values = np.array(
        [1, 2, 3, 4],
        dtype=input_dtype,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    assert result.dtype == input_dtype


# ---------------------------------------------------------------------------
# Leading/batch dimensions
# ---------------------------------------------------------------------------


def test_leading_dimensions_are_preserved(
    simple_geometry,
):
    values = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        ]
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
    )

    assert result.shape == (
        2,
        2,
        4,
    )

    np.testing.assert_allclose(
        result[0, 0],
        [1.5, 2.0, 3.0, 3.5],
    )


# ---------------------------------------------------------------------------
# NumPy / Torch consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "sum",
        "min",
        "max",
        "median",
        "count",
        "std",
    ],
)
def test_numpy_torch_consistency(
    simple_geometry,
    reduction,
):
    numpy_values = np.array(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    torch_values = torch.tensor(
        numpy_values,
        dtype=torch.float64,
    )

    numpy_result = neighbour_reduce(
        numpy_values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    torch_result = neighbour_reduce(
        torch_values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    np.testing.assert_allclose(
        numpy_result,
        torch_result.detach().cpu().numpy(),
    )


def test_numpy_torch_boolean_consistency(
    simple_geometry,
):
    numpy_values = np.array(
        [False, True, False, True],
        dtype=bool,
    )

    torch_values = torch.tensor(
        numpy_values,
        dtype=torch.bool,
    )

    for reduction in [
        "any",
        "all",
    ]:
        numpy_result = neighbour_reduce(
            numpy_values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction=reduction,
        )

        torch_result = neighbour_reduce(
            torch_values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction=reduction,
        )

        np.testing.assert_array_equal(
            numpy_result,
            torch_result.cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# PyTorch gradient behaviour
# ---------------------------------------------------------------------------


def test_mean_is_differentiable(
    simple_geometry,
):
    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="mean",
    )

    result.sum().backward()

    assert values.grad is not None

    expected_gradient = torch.tensor(
        [
            1.0 / 2.0 + 1.0 / 3.0,
            1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 3.0,
            1.0 / 3.0 + 1.0 / 3.0 + 1.0 / 2.0,
            1.0 / 3.0 + 1.0 / 2.0,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        values.grad,
        expected_gradient,
    )


def test_sum_is_differentiable(
    simple_geometry,
):
    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = neighbour_reduce(
        values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction="sum",
    )

    result.sum().backward()

    assert values.grad is not None

    expected_gradient = torch.tensor(
        [2.0, 3.0, 3.0, 2.0],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        values.grad,
        expected_gradient,
    )


# ---------------------------------------------------------------------------
# CPU / CUDA consistency
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
@pytest.mark.parametrize(
    "reduction",
    [
        "mean",
        "sum",
        "min",
        "max",
        "median",
        "count",
        "std",
    ],
)
def test_cpu_cuda_consistency(
    simple_geometry,
    reduction,
):
    cpu_values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )

    cuda_values = cpu_values.cuda()

    cpu_result = neighbour_reduce(
        cpu_values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    cuda_result = neighbour_reduce(
        cuda_values,
        simple_geometry,
        refinement_level=5,
        radius_m=100.0,
        reduction=reduction,
    )

    torch.testing.assert_close(
        cpu_result,
        cuda_result.cpu(),
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def test_convenience_wrappers(
    simple_geometry,
):
    values = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    np.testing.assert_array_equal(
        median_filter(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
        ),
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="median",
        ),
    )

    np.testing.assert_array_equal(
        mean_filter(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
        ),
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="mean",
        ),
    )

    np.testing.assert_array_equal(
        min_filter(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
        ),
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="min",
        ),
    )

    np.testing.assert_array_equal(
        max_filter(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
        ),
        neighbour_reduce(
            values,
            simple_geometry,
            refinement_level=5,
            radius_m=100.0,
            reduction="max",
        ),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_radius_raises():
    with pytest.raises(
        ValueError,
        match="radius_m",
    ):
        neighbour_reduce(
            np.array([1.0]),
            np.array(
                [1],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=-1.0,
        )


def test_nonfinite_radius_raises():
    with pytest.raises(
        ValueError,
        match="radius_m",
    ):
        neighbour_reduce(
            np.array([1.0]),
            np.array(
                [1],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=np.nan,
        )


def test_invalid_refinement_level_raises():
    with pytest.raises(
        ValueError,
        match="refinement_level",
    ):
        neighbour_reduce(
            np.array([1.0]),
            np.array(
                [1],
                dtype=np.uint64,
            ),
            refinement_level=30,
            radius_m=100.0,
        )


def test_invalid_neighbourhood_raises():
    with pytest.raises(
        ValueError,
        match="neighbourhood",
    ):
        neighbour_reduce(
            np.array([1.0]),
            np.array(
                [1],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=100.0,
            neighbourhood="invalid",
        )


def test_invalid_reduction_raises():
    with pytest.raises(
        ValueError,
        match="reduction",
    ):
        neighbour_reduce(
            np.array([1.0]),
            np.array(
                [1],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=100.0,
            reduction="invalid",
        )


def test_duplicate_cell_ids_raise():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        neighbour_reduce(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [10, 10],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=100.0,
        )


def test_duplicate_domain_ids_raise():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        neighbour_reduce(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [10, 11],
                dtype=np.uint64,
            ),
            refinement_level=5,
            radius_m=100.0,
            domain=np.array(
                [10, 10],
                dtype=np.uint64,
            ),
        )


def test_values_last_dimension_must_match_cell_ids():
    cell_ids = np.array(
        [10, 11, 12],
        dtype=np.uint64,
    )

    values = np.array(
        [1.0, 2.0],
    )

    with pytest.raises(
        ValueError,
        match="last dimension",
    ):
        neighbour_reduce(
            values,
            cell_ids,
            refinement_level=5,
            radius_m=100.0,
        )


def test_complex_values_are_rejected():
    cell_ids = np.array(
        [10],
        dtype=np.uint64,
    )

    values = np.array(
        [1.0 + 2.0j],
    )

    with pytest.raises(
        TypeError,
        match="Complex",
    ):
        neighbour_reduce(
            values,
            cell_ids,
            refinement_level=5,
            radius_m=100.0,
        )


# ---------------------------------------------------------------------------
# Real HEALPix geometry integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "refinement_level",
    [
        2,
        3,
        4,
    ],
)
def test_real_geometry_multiple_refinement_levels(
    refinement_level,
):
    center = np.array(
        [0],
        dtype=np.uint64,
    )

    radius_m = 500_000.0

    domain = binary_dilation(
        center,
        radius=radius_m,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    values = np.full(
        domain.size,
        7.25,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="mean",
        neighbourhood="cell_center",
    )

    center_position = np.flatnonzero(
        domain == 0
    )[0]

    assert result[
        center_position
    ] == pytest.approx(
        7.25
    )


def test_real_radius_smaller_than_cell_spacing():
    cell_ids = np.array(
        [0],
        dtype=np.uint64,
    )

    values = np.array(
        [17.0],
    )

    result = neighbour_reduce(
        values,
        cell_ids,
        refinement_level=3,
        radius_m=1.0,
        reduction="count",
        neighbourhood="cell_center",
    )

    np.testing.assert_array_equal(
        result,
        [1],
    )


def test_real_neighbourhood_count_matches_morphology_geometry():
    refinement_level = 3
    radius_m = 500_000.0

    center = np.array(
        [0],
        dtype=np.uint64,
    )

    domain = binary_dilation(
        center,
        radius=radius_m,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="count",
        neighbourhood="cell_center",
    )

    center_position = np.flatnonzero(
        domain == 0
    )[0]

    assert result[
        center_position
    ] == domain.size


def test_real_neighbourhood_across_base_pixel_boundary():
    refinement_level = 2
    radius_m = 2_000_000.0

    cells_per_base_pixel = (
        4**refinement_level
    )

    number_of_pixels = (
        12 * cells_per_base_pixel
    )

    found = None

    for center in range(
        number_of_pixels
    ):
        neighbourhood = binary_dilation(
            np.array(
                [center],
                dtype=np.uint64,
            ),
            radius=radius_m,
            refinement_level=refinement_level,
            neighbourhood="cell_center",
        )

        base_pixels = (
            neighbourhood
            // cells_per_base_pixel
        )

        if (
            np.unique(base_pixels).size
            > 1
        ):
            found = (
                center,
                neighbourhood,
            )
            break

    assert found is not None

    center, domain = found

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="count",
        neighbourhood="cell_center",
    )

    center_position = np.flatnonzero(
        domain == center
    )[0]

    assert result[
        center_position
    ] == domain.size


def test_real_neighbourhood_near_pole():
    refinement_level = 3
    radius_m = 1_000_000.0

    number_of_pixels = (
        12 * 4**refinement_level
    )

    all_cells = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    _, latitude = nested.healpix_to_lonlat(
        all_cells,
        refinement_level,
        ellipsoid="WGS84",
    )

    center = int(
        all_cells[
            np.argmax(
                np.abs(latitude)
            )
        ]
    )

    domain = binary_dilation(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        radius=radius_m,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="mean",
        neighbourhood="cell_center",
    )

    center_position = np.flatnonzero(
        domain == center
    )[0]

    assert result[
        center_position
    ] == pytest.approx(
        1.0
    )


def test_real_neighbourhood_across_longitude_wrap():
    refinement_level = 3
    radius_m = 1_200_000.0

    number_of_pixels = (
        12 * 4**refinement_level
    )

    all_cells = np.arange(
        number_of_pixels,
        dtype=np.uint64,
    )

    longitude, _ = nested.healpix_to_lonlat(
        all_cells,
        refinement_level,
        ellipsoid="WGS84",
    )

    longitude = np.asarray(
        longitude
    )

    # healpix-geo may expose longitude using a signed or unsigned
    # representation. Choose a cell close to the relevant numerical
    # discontinuity without assuming one particular convention.
    if np.nanmin(longitude) < 0:
        distance_to_wrap = np.abs(
            np.abs(longitude) - 180.0
        )
    else:
        distance_to_wrap = np.minimum(
            np.abs(longitude),
            np.abs(longitude - 360.0),
        )

    center = int(
        all_cells[
            np.argmin(
                distance_to_wrap
            )
        ]
    )

    domain = binary_dilation(
        np.array(
            [center],
            dtype=np.uint64,
        ),
        radius=radius_m,
        refinement_level=refinement_level,
        neighbourhood="cell_center",
    )

    neighbourhood_longitude, _ = (
        nested.healpix_to_lonlat(
            domain,
            refinement_level,
            ellipsoid="WGS84",
        )
    )

    neighbourhood_longitude = (
        np.asarray(
            neighbourhood_longitude
        )
    )

    # A neighbourhood crossing the representation discontinuity has a
    # large raw numerical longitude span although it is geographically
    # local on the sphere.
    assert (
        np.ptp(
            neighbourhood_longitude
        )
        > 180.0
    )

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="mean",
        neighbourhood="cell_center",
    )

    center_position = np.flatnonzero(
        domain == center
    )[0]

    assert result[
        center_position
    ] == pytest.approx(
        1.0
    )


def test_real_cone_coverage_reduction():
    refinement_level = 3
    radius_m = 500_000.0

    center = np.array(
        [0],
        dtype=np.uint64,
    )

    domain = binary_dilation(
        center,
        radius=radius_m,
        refinement_level=refinement_level,
        neighbourhood="cone_coverage",
    )

    values = np.ones(
        domain.size,
        dtype=np.float64,
    )

    result = neighbour_reduce(
        values,
        domain,
        refinement_level=refinement_level,
        radius_m=radius_m,
        reduction="mean",
        neighbourhood="cone_coverage",
    )

    center_position = np.flatnonzero(
        domain == 0
    )[0]

    assert result[
        center_position
    ] == pytest.approx(
        1.0
    )
