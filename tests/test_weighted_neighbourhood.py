# tests/test_weighted_neighbourhood.py

"""Tests for the shared weighted-neighbour aggregation helper.

These tests intentionally avoid any HEALPix geometry.

The helper under test is responsible only for:

- mapping neighbour cell IDs to positions in the signal array,
- respecting padded/invalid neighbour positions,
- applying supplied weights,
- excluding NaN input samples,
- optional normalization,
- empty and zero-effective-weight behaviour,
- preserving NumPy/Torch semantics and Torch autograd.

Spatial meaning such as distance, azimuth, radial kernels, and directional
kernels belongs to the caller and is tested elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_analyse._weighted_neighbourhood import (
    compact_weighted_neighbourhood_reduce,
    neighbour_positions,
    weighted_neighbourhood_reduce,
)


def test_compact_reduction_matches_padded_reduction():
    cell_ids = np.array([30, 10, 20], dtype=np.uint64)
    values = np.array(
        [[3.0, 1.0, np.nan], [6.0, 2.0, 4.0]],
        dtype=np.float64,
    )
    neighbour_ids = np.array(
        [[10, 20, 30], [30, -1, -1], [20, 10, -1]],
        dtype=np.int64,
    )
    valid_mask = neighbour_ids >= 0
    weights = np.array(
        [[0.5, 0.25, 0.125], [2.0, 0.0, 0.0], [1.5, -0.5, 0.0]],
        dtype=np.float64,
    )

    padded = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )
    compact = compact_weighted_neighbourhood_reduce(
        values,
        np.array([1, 2, 0, 0, 2, 1], dtype=np.int64),
        np.array([0, 3, 4, 6], dtype=np.int64),
        weights[valid_mask],
        normalize=True,
    )

    np.testing.assert_array_equal(compact, padded)


# ---------------------------------------------------------------------------
# neighbour_positions
# ---------------------------------------------------------------------------


def test_neighbour_positions_maps_cell_ids():
    cell_ids = np.array(
        [30, 10, 20],
        dtype=np.uint64,
    )

    neighbour_ids = np.array(
        [
            [10, 20, -1],
            [30, 10, 20],
        ],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [
            [True, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )

    result = neighbour_positions(
        cell_ids,
        neighbour_ids,
        valid_mask,
    )

    expected = np.array(
        [
            [1, 2, -1],
            [0, 1, 2],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_neighbour_positions_rejects_missing_cell():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    neighbour_ids = np.array(
        [[10, 30]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="absent from 'cell_ids'",
    ):
        neighbour_positions(
            cell_ids,
            neighbour_ids,
            valid_mask,
        )


def test_neighbour_positions_all_invalid():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    neighbour_ids = np.array(
        [[-1, -1]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[False, False]],
        dtype=bool,
    )

    result = neighbour_positions(
        cell_ids,
        neighbour_ids,
        valid_mask,
    )

    np.testing.assert_array_equal(
        result,
        np.array(
            [[-1, -1]],
            dtype=np.int64,
        ),
    )


# ---------------------------------------------------------------------------
# NumPy weighted aggregation
# ---------------------------------------------------------------------------


def test_numpy_weighted_sum():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0, 8.0],
    )

    neighbour_ids = np.array(
        [
            [10, 20],
            [20, 30],
        ],
        dtype=np.int64,
    )

    valid_mask = np.ones(
        neighbour_ids.shape,
        dtype=bool,
    )

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    expected = np.array(
        [
            1.0 * 2.0 + 2.0 * 4.0,
            3.0 * 4.0 + 4.0 * 8.0,
        ],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_numpy_normalized_weighted_mean():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0, 8.0],
    )

    neighbour_ids = np.array(
        [
            [10, 20],
            [20, 30],
        ],
        dtype=np.int64,
    )

    valid_mask = np.ones(
        neighbour_ids.shape,
        dtype=bool,
    )

    weights = np.array(
        [
            [1.0, 3.0],
            [1.0, 1.0],
        ],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    expected = np.array(
        [
            (1.0 * 2.0 + 3.0 * 4.0) / 4.0,
            (1.0 * 4.0 + 1.0 * 8.0) / 2.0,
        ],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_numpy_padding_is_ignored():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = np.array(
        [5.0, 7.0],
    )

    neighbour_ids = np.array(
        [[10, -1]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, False]],
        dtype=bool,
    )

    weights = np.array(
        [[2.0, 9999.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    np.testing.assert_allclose(
        result,
        np.array([10.0]),
    )


def test_numpy_nan_value_is_excluded():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, np.nan, 8.0],
    )

    neighbour_ids = np.array(
        [[10, 20, 30]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 100.0, 3.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    expected = np.array(
        [
            1.0 * 2.0
            + 3.0 * 8.0
        ],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_numpy_nan_value_is_removed_from_normalization():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, np.nan, 8.0],
    )

    neighbour_ids = np.array(
        [[10, 20, 30]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 100.0, 3.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    expected = np.array(
        [
            (
                1.0 * 2.0
                + 3.0 * 8.0
            )
            / (
                1.0
                + 3.0
            )
        ],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_numpy_zero_effective_weight_returns_nan_when_normalized():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0],
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[0.0, 0.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    assert result.shape == (1,)
    assert np.isnan(result[0])


def test_numpy_zero_effective_weight_returns_zero_when_not_normalized():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0],
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[0.0, 0.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0]),
    )


def test_numpy_empty_neighbour_dimension():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0],
    )

    neighbour_ids = np.empty(
        (2, 0),
        dtype=np.int64,
    )

    valid_mask = np.empty(
        (2, 0),
        dtype=bool,
    )

    weights = np.empty(
        (2, 0),
        dtype=float,
    )

    unnormalized = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    normalized = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    np.testing.assert_array_equal(
        unnormalized,
        np.zeros(2),
    )

    assert np.all(
        np.isnan(normalized)
    )


def test_numpy_multidimensional_values():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ]
    )

    neighbour_ids = np.array(
        [
            [10, 20],
            [20, 30],
        ],
        dtype=np.int64,
    )

    valid_mask = np.ones(
        neighbour_ids.shape,
        dtype=bool,
    )

    weights = np.ones(
        neighbour_ids.shape,
        dtype=float,
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    expected = np.array(
        [
            [3.0, 5.0],
            [30.0, 50.0],
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_scalar_weight_broadcasts():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = np.array(
        [2.0, 4.0],
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights=2.0,
        normalize=False,
    )

    np.testing.assert_allclose(
        result,
        np.array([12.0]),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_duplicate_cell_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="must not contain duplicate",
    ):
        weighted_neighbourhood_reduce(
            np.array([1.0, 2.0]),
            np.array(
                [10, 10],
                dtype=np.uint64,
            ),
            np.array(
                [[10]],
                dtype=np.int64,
            ),
            np.array(
                [[True]],
                dtype=bool,
            ),
            np.array(
                [[1.0]],
            ),
        )


def test_invalid_padding_marked_valid_is_rejected():
    with pytest.raises(
        ValueError,
        match="padded neighbour positions",
    ):
        weighted_neighbourhood_reduce(
            np.array([1.0]),
            np.array(
                [10],
                dtype=np.uint64,
            ),
            np.array(
                [[-1]],
                dtype=np.int64,
            ),
            np.array(
                [[True]],
                dtype=bool,
            ),
            np.array(
                [[1.0]],
            ),
        )


def test_nonfinite_valid_weight_is_rejected():
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        weighted_neighbourhood_reduce(
            np.array([1.0]),
            np.array(
                [10],
                dtype=np.uint64,
            ),
            np.array(
                [[10]],
                dtype=np.int64,
            ),
            np.array(
                [[True]],
                dtype=bool,
            ),
            np.array(
                [[np.nan]],
            ),
        )


def test_nonfinite_padding_weight_is_ignored():
    cell_ids = np.array(
        [10],
        dtype=np.uint64,
    )

    result = weighted_neighbourhood_reduce(
        np.array([3.0]),
        cell_ids,
        np.array(
            [[10, -1]],
            dtype=np.int64,
        ),
        np.array(
            [[True, False]],
            dtype=bool,
        ),
        np.array(
            [[2.0, np.nan]],
        ),
        normalize=False,
    )

    np.testing.assert_allclose(
        result,
        np.array([6.0]),
    )


def test_normalize_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="normalize",
    ):
        weighted_neighbourhood_reduce(
            np.array([1.0]),
            np.array(
                [10],
                dtype=np.uint64,
            ),
            np.array(
                [[10]],
                dtype=np.int64,
            ),
            np.array(
                [[True]],
                dtype=bool,
            ),
            np.array(
                [[1.0]],
            ),
            normalize=1,
        )


# ---------------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------------


def test_torch_weighted_sum_cpu():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [2.0, 4.0, 8.0],
        dtype=torch.float64,
    )

    neighbour_ids = np.array(
        [
            [10, 20],
            [20, 30],
        ],
        dtype=np.int64,
    )

    valid_mask = np.ones(
        neighbour_ids.shape,
        dtype=bool,
    )

    weights = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    expected = torch.tensor(
        [
            10.0,
            44.0,
        ],
        dtype=torch.float64,
    )

    assert isinstance(
        result,
        torch.Tensor,
    )

    assert result.device == values.device

    torch.testing.assert_close(
        result,
        expected,
    )


def test_torch_normalized_nan_handling():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [2.0, float("nan"), 8.0],
        dtype=torch.float64,
    )

    neighbour_ids = np.array(
        [[10, 20, 30]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 100.0, 3.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    expected = torch.tensor(
        [6.5],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        result,
        expected,
    )


def test_torch_autograd_through_values():
    cell_ids = np.array(
        [10, 20, 30],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    neighbour_ids = np.array(
        [[10, 20, 30]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 2.0, 4.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=False,
    )

    result.sum().backward()

    expected_gradient = torch.tensor(
        [1.0, 2.0, 4.0],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        values.grad,
        expected_gradient,
    )


def test_torch_autograd_with_normalization():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [1.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 3.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    result.sum().backward()

    expected_gradient = torch.tensor(
        [
            1.0 / 4.0,
            3.0 / 4.0,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        values.grad,
        expected_gradient,
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available.",
)
def test_torch_mps_device_preserved():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
        device="mps",
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 1.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    assert result.device.type == "mps"

    torch.testing.assert_close(
        result.cpu(),
        torch.tensor(
            [1.5],
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_torch_cuda_device_preserved():
    cell_ids = np.array(
        [10, 20],
        dtype=np.uint64,
    )

    values = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float32,
        device="cuda",
    )

    neighbour_ids = np.array(
        [[10, 20]],
        dtype=np.int64,
    )

    valid_mask = np.array(
        [[True, True]],
        dtype=bool,
    )

    weights = np.array(
        [[1.0, 1.0]],
    )

    result = weighted_neighbourhood_reduce(
        values,
        cell_ids,
        neighbour_ids,
        valid_mask,
        weights,
        normalize=True,
    )

    assert result.device.type == "cuda"

    torch.testing.assert_close(
        result.cpu(),
        torch.tensor(
            [1.5],
            dtype=torch.float32,
        ),
    )
