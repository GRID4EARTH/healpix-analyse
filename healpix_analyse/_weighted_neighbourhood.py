"""Shared weighted aggregation over HEALPix neighbourhoods.

This private module contains the signal-processing part shared by spatial
operators that already know:

- which HEALPix cells contribute to each output cell, and
- which weight applies to each contribution.

It deliberately contains no spatial-kernel semantics.

In particular, this module does not know whether weights were produced from:

- physical distance,
- geographical bearing,
- a Gaussian radial kernel,
- a directional kernel,
- or another future spatial model.

The intended separation is::

    spatial geometry / kernel
              |
              v
       neighbour positions
          + weights
              |
              v
    weighted_neighbourhood_reduce()
              |
              v
          output values

This helper is shared by operations such as:

- radial filtering,
- directional filtering.

The final dimension of ``values`` corresponds one-to-one with ``cell_ids``.

Cells represented by invalid/padded neighbourhood positions do not
participate.

NaN input samples are treated as unavailable observations. Their weights are
removed from the effective weighted aggregation rather than allowing one NaN
to contaminate an entire neighbourhood.

For Torch input, geometry and supplied weights are treated as constants while
the weighted signal operation remains differentiable with respect to
``values``.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_cell_ids(
    cell_ids: np.ndarray,
) -> np.ndarray:
    """Validate signal-cell IDs used for neighbour lookup."""
    raw = np.asarray(
        cell_ids
    )

    if raw.ndim != 1:
        raise ValueError(
            "'cell_ids' must be a one-dimensional array."
        )

    if raw.dtype == np.bool_ or not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise TypeError(
            "'cell_ids' must contain integer HEALPix cell IDs."
        )

    if np.any(
        raw < 0
    ):
        raise ValueError(
            "'cell_ids' must contain non-negative HEALPix cell IDs."
        )

    cells = raw.astype(
        np.uint64,
        copy=False,
    )

    if np.unique(
        cells
    ).size != cells.size:
        raise ValueError(
            "'cell_ids' must not contain duplicate HEALPix cell IDs."
        )

    return cells


def _validate_neighbour_arrays(
    neighbour_ids: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate padded neighbourhood IDs, masks, and supplied weights."""
    neighbours = np.asarray(
        neighbour_ids
    )

    if neighbours.ndim != 2:
        raise ValueError(
            "'neighbour_ids' must be a two-dimensional padded array."
        )

    if neighbours.dtype == np.bool_ or not np.issubdtype(
        neighbours.dtype,
        np.integer,
    ):
        raise TypeError(
            "'neighbour_ids' must contain integer HEALPix cell IDs "
            "or -1 padding."
        )

    neighbours = neighbours.astype(
        np.int64,
        copy=False,
    )

    if np.any(
        neighbours < -1
    ):
        raise ValueError(
            "'neighbour_ids' may contain only valid cell IDs "
            "or -1 padding."
        )

    mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if mask.shape != neighbours.shape:
        raise ValueError(
            "'valid_mask' must have the same shape as 'neighbour_ids'."
        )

    if np.any(
        mask & (neighbours < 0)
    ):
        raise ValueError(
            "'valid_mask' marks padded neighbour positions as valid."
        )

    supplied_weights = np.asarray(
        weights
    )

    if supplied_weights.ndim == 0:
        supplied_weights = np.full(
            neighbours.shape,
            supplied_weights.item(),
            dtype=np.float64,
        )
    else:
        try:
            supplied_weights = np.broadcast_to(
                supplied_weights,
                neighbours.shape,
            ).astype(
                np.float64,
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                "'weights' must be scalar or broadcastable to the "
                "shape of 'neighbour_ids'."
            ) from exc

    if np.any(
        ~np.isfinite(
            supplied_weights[
                mask
            ]
        )
    ):
        raise ValueError(
            "'weights' must be finite at valid neighbour positions."
        )

    # Padding positions never participate. Their numerical weight is made
    # explicitly zero so downstream code cannot accidentally use them.
    supplied_weights = np.where(
        mask,
        supplied_weights,
        0.0,
    )

    return (
        neighbours,
        mask,
        supplied_weights,
    )


def _validate_values(
    values: np.ndarray | torch.Tensor,
    number_of_cells: int,
) -> None:
    """Validate an input signal whose final axis corresponds to cell IDs."""
    if not isinstance(
        values,
        (np.ndarray, torch.Tensor),
    ):
        raise TypeError(
            "'values' must be a NumPy array or PyTorch tensor."
        )

    if values.ndim < 1:
        raise ValueError(
            "'values' must have at least one dimension."
        )

    if values.shape[-1] != number_of_cells:
        raise ValueError(
            "The last dimension of 'values' must match the number "
            "of entries in 'cell_ids'."
        )


def neighbour_positions(
    cell_ids: np.ndarray,
    neighbour_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Map padded HEALPix neighbour IDs to positions in ``values``.

    Parameters
    ----------
    cell_ids
        One-dimensional unique HEALPix cell IDs corresponding to the final
        dimension of the signal array.

    neighbour_ids
        Padded ``(N, K)`` matrix of HEALPix cell IDs. Padding positions must
        contain ``-1``.

    valid_mask
        Boolean ``(N, K)`` matrix indicating which neighbour positions are
        real.

    Returns
    -------
    numpy.ndarray
        Signed integer ``(N, K)`` array.

        Valid positions contain indices into ``values[..., :]``.

        Invalid/padded positions contain ``-1``.

    Notes
    -----
    Every valid neighbour must occur in ``cell_ids``. Neighbourhood/domain
    construction is responsible for enforcing this before aggregation.
    """
    cells = _validate_cell_ids(
        cell_ids
    )

    neighbours = np.asarray(
        neighbour_ids
    )

    mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if neighbours.ndim != 2:
        raise ValueError(
            "'neighbour_ids' must be a two-dimensional padded array."
        )

    if mask.shape != neighbours.shape:
        raise ValueError(
            "'valid_mask' must have the same shape as 'neighbour_ids'."
        )

    positions = np.full(
        neighbours.shape,
        -1,
        dtype=np.int64,
    )

    if not np.any(
        mask
    ):
        return positions

    sorter = np.argsort(
        cells
    )

    sorted_ids = cells[
        sorter
    ]

    valid_neighbours = neighbours[
        mask
    ].astype(
        np.uint64,
        copy=False,
    )

    sorted_positions = np.searchsorted(
        sorted_ids,
        valid_neighbours,
    )

    if np.any(
        sorted_positions
        >= sorted_ids.size
    ):
        raise ValueError(
            "A valid neighbour cell is absent from 'cell_ids'."
        )

    matched = sorted_ids[
        sorted_positions
    ]

    if not np.array_equal(
        matched,
        valid_neighbours,
    ):
        raise ValueError(
            "A valid neighbour cell is absent from 'cell_ids'."
        )

    positions[
        mask
    ] = sorter[
        sorted_positions
    ]

    return positions


def _numpy_segment_sum(
    values: np.ndarray,
    row_offsets: np.ndarray,
) -> np.ndarray:
    """Sum the final axis over contiguous CSR-style row segments."""
    number_of_rows = row_offsets.size - 1
    output_shape = (*values.shape[:-1], number_of_rows)
    if number_of_rows == 0:
        return np.empty(output_shape, dtype=values.dtype)
    if values.shape[-1] == 0:
        return np.zeros(output_shape, dtype=values.dtype)

    row_counts = np.diff(row_offsets)
    starts = np.minimum(row_offsets[:-1], values.shape[-1] - 1)
    result = np.add.reduceat(values, starts, axis=-1)
    result[..., row_counts == 0] = 0
    return result


def compact_weighted_neighbourhood_reduce(
    values: np.ndarray | torch.Tensor,
    neighbour_positions: np.ndarray,
    row_offsets: np.ndarray,
    weights: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray | torch.Tensor:
    """Apply weights over an unpadded CSR-style neighbourhood geometry."""
    if not isinstance(values, (np.ndarray, torch.Tensor)) or values.ndim < 1:
        raise TypeError("'values' must be a NumPy array or PyTorch tensor.")
    positions = np.asarray(neighbour_positions)
    if positions.ndim != 1:
        raise ValueError("'neighbour_positions' must be one-dimensional.")
    if positions.dtype == np.bool_ or not np.issubdtype(
        positions.dtype,
        np.integer,
    ):
        raise TypeError("'neighbour_positions' must contain integers.")
    if np.any(positions < 0) or np.any(positions >= values.shape[-1]):
        raise ValueError("A neighbour position is outside the input signal.")
    positions = positions.astype(np.int64, copy=False)
    offsets = np.asarray(row_offsets, dtype=np.int64)
    supplied_weights = np.asarray(weights)

    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("'row_offsets' must be a non-empty 1-D array.")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("'row_offsets' must be non-decreasing and start at 0.")
    if offsets[-1] != positions.size:
        raise ValueError("The final row offset must equal the neighbour count.")
    try:
        supplied_weights = np.broadcast_to(
            supplied_weights,
            positions.shape,
        ).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(
            "'weights' must be scalar or match the neighbour vector."
        ) from exc
    if not np.all(np.isfinite(supplied_weights)):
        raise ValueError("'weights' must be finite.")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("'normalize' must be a boolean.")

    number_of_rows = offsets.size - 1

    if isinstance(values, torch.Tensor):
        dtype = _torch_output_dtype(values)
        signal = values.to(dtype=dtype)
        output_shape = (*signal.shape[:-1], number_of_rows)
        if positions.size == 0:
            fill = float("nan") if normalize else 0.0
            return torch.full(
                output_shape,
                fill,
                dtype=dtype,
                device=signal.device,
            )

        position_tensor = torch.as_tensor(
            positions,
            dtype=torch.long,
            device=signal.device,
        )
        gathered = torch.index_select(signal, -1, position_tensor)
        weight_tensor = torch.as_tensor(
            np.array(supplied_weights, copy=True),
            dtype=dtype,
            device=signal.device,
        )
        prefix = (1,) * (signal.ndim - 1)
        weight_tensor = weight_tensor.reshape(*prefix, -1)
        sample_valid = ~torch.isnan(gathered)
        effective_weights = torch.where(
            sample_valid,
            weight_tensor,
            torch.zeros((), dtype=dtype, device=signal.device),
        )
        contributions = effective_weights * torch.where(
            sample_valid,
            gathered,
            torch.zeros((), dtype=dtype, device=signal.device),
        )
        row_ids = torch.as_tensor(
            np.repeat(
                np.arange(number_of_rows, dtype=np.int64),
                np.diff(offsets),
            ),
            dtype=torch.long,
            device=signal.device,
        )
        row_ids = row_ids.reshape(*prefix, -1).expand_as(contributions)
        numerator = torch.zeros(
            output_shape,
            dtype=dtype,
            device=signal.device,
        ).scatter_add(-1, row_ids, contributions)
        if not normalize:
            return numerator
        denominator = torch.zeros_like(numerator).scatter_add(
            -1,
            row_ids,
            effective_weights.expand_as(contributions),
        )
        nonzero = denominator != 0
        return torch.where(
            nonzero,
            numerator / torch.where(nonzero, denominator, 1),
            torch.full_like(numerator, float("nan")),
        )

    output_dtype = np.result_type(values.dtype, np.float64)
    gathered = np.asarray(values[..., positions], dtype=output_dtype)
    prefix = (1,) * (values.ndim - 1)
    broadcast_weights = supplied_weights.reshape(*prefix, -1).astype(
        output_dtype,
        copy=False,
    )
    sample_valid = ~np.isnan(gathered)
    effective_weights = np.where(sample_valid, broadcast_weights, 0.0)
    contributions = effective_weights * np.where(sample_valid, gathered, 0.0)
    numerator = _numpy_segment_sum(contributions, offsets)
    if not normalize:
        return numerator
    denominator = _numpy_segment_sum(effective_weights, offsets)
    result = np.full(numerator.shape, np.nan, dtype=output_dtype)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _torch_output_dtype(
    values: torch.Tensor,
) -> torch.dtype:
    """Return a suitable numeric output dtype for weighted filtering."""
    if (
        values.is_floating_point()
        or values.is_complex()
    ):
        return values.dtype

    return torch.get_default_dtype()


def _numpy_weighted_neighbourhood(
    values: np.ndarray,
    positions: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    """Apply supplied neighbour weights to a NumPy signal."""
    output_shape = (
        *values.shape[:-1],
        positions.shape[0],
    )

    output_dtype = np.result_type(
        values.dtype,
        np.float64,
    )

    if positions.shape[0] == 0:
        return np.empty(
            output_shape,
            dtype=output_dtype,
        )

    if positions.shape[1] == 0:
        if normalize:
            return np.full(
                output_shape,
                np.nan,
                dtype=output_dtype,
            )

        return np.zeros(
            output_shape,
            dtype=output_dtype,
        )

    safe_positions = np.where(
        valid_mask,
        positions,
        0,
    )

    gathered = np.asarray(
        values[
            ...,
            safe_positions,
        ],
        dtype=output_dtype,
    )

    prefix = (
        1,
    ) * (
        values.ndim - 1
    )

    geometry_valid = valid_mask.reshape(
        *prefix,
        *valid_mask.shape,
    )

    broadcast_weights = weights.reshape(
        *prefix,
        *weights.shape,
    ).astype(
        output_dtype,
        copy=False,
    )

    # Invalid/padded positions and NaN signal samples are both treated as
    # unavailable contributions.
    sample_valid = (
        geometry_valid
        & ~np.isnan(
            gathered
        )
    )

    effective_weights = np.where(
        sample_valid,
        broadcast_weights,
        0.0,
    )

    safe_values = np.where(
        sample_valid,
        gathered,
        0.0,
    )

    numerator = np.sum(
        effective_weights
        * safe_values,
        axis=-1,
    )

    if not normalize:
        return numerator

    denominator = np.sum(
        effective_weights,
        axis=-1,
    )

    result = np.full(
        numerator.shape,
        np.nan,
        dtype=output_dtype,
    )

    nonzero = denominator != 0.0

    np.divide(
        numerator,
        denominator,
        out=result,
        where=nonzero,
    )

    return result


def _torch_weighted_neighbourhood(
    values: torch.Tensor,
    positions: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Apply supplied neighbour weights to a Torch signal.

    Geometry and supplied weights are constants.

    Gradients are preserved with respect to ``values``.
    """
    dtype = _torch_output_dtype(
        values
    )

    signal = values.to(
        dtype=dtype
    )

    output_shape = (
        *signal.shape[:-1],
        positions.shape[0],
    )

    if positions.shape[0] == 0:
        return torch.empty(
            output_shape,
            dtype=dtype,
            device=signal.device,
        )

    if positions.shape[1] == 0:
        if normalize:
            return torch.full(
                output_shape,
                float("nan"),
                dtype=dtype,
                device=signal.device,
            )

        return torch.zeros(
            output_shape,
            dtype=dtype,
            device=signal.device,
        )

    valid_tensor = torch.as_tensor(
        valid_mask,
        dtype=torch.bool,
        device=signal.device,
    )

    position_tensor = torch.as_tensor(
        np.where(
            valid_mask,
            positions,
            0,
        ),
        dtype=torch.long,
        device=signal.device,
    )

    gathered = torch.index_select(
        signal,
        dim=-1,
        index=position_tensor.reshape(-1),
    )

    gathered = gathered.reshape(
        *signal.shape[:-1],
        *position_tensor.shape,
    )

    prefix = (
        1,
    ) * (
        signal.ndim - 1
    )

    geometry_valid = valid_tensor.reshape(
        *prefix,
        *valid_tensor.shape,
    )

    weight_tensor = torch.as_tensor(
        weights,
        dtype=dtype,
        device=signal.device,
    ).reshape(
        *prefix,
        *weights.shape,
    )

    if signal.is_complex():
        sample_valid = (
            geometry_valid
            & ~torch.isnan(
                gathered.real
            )
            & ~torch.isnan(
                gathered.imag
            )
        )
    else:
        sample_valid = (
            geometry_valid
            & ~torch.isnan(
                gathered
            )
        )

    zero = torch.zeros(
        (),
        dtype=dtype,
        device=signal.device,
    )

    effective_weights = torch.where(
        sample_valid,
        weight_tensor,
        zero,
    )

    safe_values = torch.where(
        sample_valid,
        gathered,
        zero,
    )

    numerator = torch.sum(
        effective_weights
        * safe_values,
        dim=-1,
    )

    if not normalize:
        return numerator

    denominator = torch.sum(
        effective_weights,
        dim=-1,
    )

    nonzero = denominator != 0

    # Avoid division by zero in the intermediate expression. The final
    # output is explicitly NaN wherever no effective weight is available.
    safe_denominator = torch.where(
        nonzero,
        denominator,
        torch.ones_like(
            denominator
        ),
    )

    result = (
        numerator
        / safe_denominator
    )

    return torch.where(
        nonzero,
        result,
        torch.full_like(
            result,
            float("nan"),
        ),
    )


def weighted_neighbourhood_reduce(
    values: np.ndarray | torch.Tensor,
    cell_ids: np.ndarray,
    neighbour_ids: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray | torch.Tensor:
    """Apply supplied weights over padded HEALPix neighbourhoods.

    Parameters
    ----------
    values
        Input NumPy array or PyTorch tensor.

        The final dimension corresponds one-to-one with ``cell_ids``.
        Arbitrary leading dimensions are preserved.

    cell_ids
        One-dimensional unique HEALPix cell IDs corresponding to the final
        signal dimension.

    neighbour_ids
        Signed integer array with shape ``(N, K)``.

        Each row lists cells contributing to one output location.

        Padding positions must contain ``-1``.

    valid_mask
        Boolean array with shape ``(N, K)``.

        ``True`` marks valid neighbour positions and ``False`` marks padding
        or otherwise unavailable positions.

        Domain filtering should already be represented by this neighbourhood
        structure before this helper is called.

    weights
        Supplied weights for the neighbour pairs.

        May be scalar or broadcastable to ``(N, K)``.

        The spatial interpretation of these weights is deliberately outside
        the responsibility of this function.

        For example::

            radial filtering:
                weight = kernel(distance_m)

            directional filtering:
                weight = kernel(
                    distance_m,
                    relative_bearing_rad,
                )

    normalize
        If ``False``::

            output = sum(weight * value)

        If ``True``::

                     sum(weight * value)
            output = -------------------
                         sum(weight)

        using only effective valid samples.

        NaN-valued input samples are treated as unavailable and therefore
        removed from both numerator and denominator.

        If ``normalize=True`` and the effective weight sum is zero, the
        corresponding result is NaN.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Weighted output with shape::

            values.shape[:-1] + (N,)

        where ``N`` is the first dimension of ``neighbour_ids``.

        NumPy input returns NumPy output.

        Torch input returns Torch output on the original device. Autograd is
        preserved with respect to ``values``.

    Notes
    -----
    This helper intentionally contains no HEALPix geometry calculation.

    It does not compute:

    - distances,
    - bearings,
    - physical neighbourhoods,
    - topological neighbourhoods,
    - radial kernels,
    - directional kernels.

    Those belong to the calling spatial operator.

    The processing flow is::

        spatial operator
        produces neighbour IDs + weights
                    |
                    v
        weighted_neighbourhood_reduce()
                    |
             +------+------+
             |             |
          gather        valid/NaN
          values          mask
             |             |
             +------+------+
                    |
             weighted sum
                    |
              normalization
                    |
                  output

    This separation allows the same aggregation implementation to be shared
    by several scientifically different spatial operators.
    """
    cells = _validate_cell_ids(
        cell_ids
    )

    _validate_values(
        values,
        cells.size,
    )

    if not isinstance(
        normalize,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'normalize' must be a boolean."
        )

    (
        neighbours,
        mask,
        supplied_weights,
    ) = _validate_neighbour_arrays(
        neighbour_ids,
        valid_mask,
        weights,
    )

    positions = neighbour_positions(
        cells,
        neighbours,
        mask,
    )

    if isinstance(
        values,
        torch.Tensor,
    ):
        return _torch_weighted_neighbourhood(
            values,
            positions,
            mask,
            supplied_weights,
            normalize=bool(normalize),
        )

    return _numpy_weighted_neighbourhood(
        values,
        positions,
        mask,
        supplied_weights,
        normalize=bool(normalize),
    )


__all__ = [
    "compact_weighted_neighbourhood_reduce",
    "neighbour_positions",
    "weighted_neighbourhood_reduce",
]
