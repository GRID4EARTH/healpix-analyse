"""Generic neighbourhood reductions for nested HEALPix fields.

This module provides local, unweighted reductions over geometrically
defined HEALPix neighbourhoods.

Neighbourhood geometry is shared with :mod:`healpix_analyse.morphology`.
In particular, the two supported neighbourhood definitions are:

``cell_center``
    Include cells whose centres lie within ``radius_m`` metres of the
    target-cell centre.

``cone_coverage``
    Include cells intersecting the circular coverage region returned by
    :func:`healpix_geo.nested.cone_coverage`.

The public API supports both NumPy arrays and PyTorch tensors.

Domain semantics
----------------
``cell_ids`` identifies all HEALPix cells for which input values are
available.

``domain`` identifies the valid processing/output domain.

If ``domain is None``, the processing domain is exactly ``cell_ids``.

If ``domain`` is supplied, every domain cell must occur in ``cell_ids``::

    domain subset_of cell_ids

Only cells belonging to ``domain`` participate in a reduction. A cell
that is geometrically inside a neighbourhood but outside ``domain`` is
not treated as zero, False, NaN, or any other padding value. It is
simply absent from the effective neighbourhood.

The output order follows ``domain`` exactly when it is supplied, and
``cell_ids`` otherwise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from healpix_analyse._neighbourhood import (
    NeighbourhoodMethod,
    build_neighbourhoods,
)


ArrayLike = np.ndarray | torch.Tensor

Reduction = Literal[
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "count",
    "std",
    "any",
    "all",
    "mode",
]


_VALID_REDUCTIONS = {
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "count",
    "std",
    "any",
    "all",
    "mode",
}


def _normalise_ids(
    ids,
    *,
    name: str,
    refinement_level: int,
) -> np.ndarray:
    """Validate HEALPix IDs while preserving their input order."""

    if torch.is_tensor(ids):
        ids = ids.detach().cpu().numpy()

    raw = np.asarray(ids)

    if raw.ndim != 1:
        raise ValueError(
            f"'{name}' must be a one-dimensional array."
        )

    if raw.dtype == np.bool_ or not np.issubdtype(
        raw.dtype,
        np.integer,
    ):
        raise TypeError(
            f"'{name}' must contain integer HEALPix cell IDs."
        )

    if np.any(raw < 0):
        raise ValueError(
            f"'{name}' must contain non-negative HEALPix cell IDs."
        )

    ids_array = raw.astype(
        np.uint64,
        copy=False,
    )

    if np.unique(ids_array).size != ids_array.size:
        raise ValueError(
            f"'{name}' must contain unique HEALPix cell IDs."
        )

    number_of_pixels = 12 * 4**refinement_level

    if np.any(ids_array >= number_of_pixels):
        raise ValueError(
            f"'{name}' contains HEALPix cell IDs outside "
            f"refinement_level={refinement_level}."
        )

    return ids_array


def _validate_refinement_level(
    refinement_level: int,
) -> int:
    """Validate and normalise the HEALPix refinement level."""

    if isinstance(
        refinement_level,
        (bool, np.bool_),
    ):
        raise ValueError(
            "'refinement_level' must be an integer between 0 and 29."
        )

    if not isinstance(
        refinement_level,
        (int, np.integer),
    ):
        raise ValueError(
            "'refinement_level' must be an integer between 0 and 29."
        )

    refinement_level = int(refinement_level)

    if not 0 <= refinement_level <= 29:
        raise ValueError(
            "'refinement_level' must be between 0 and 29."
        )

    return refinement_level


def _validate_radius(
    radius_m: float,
) -> float:
    """Validate and normalise a physical neighbourhood radius."""

    try:
        radius_m = float(radius_m)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "'radius_m' must be a finite numerical value."
        ) from error

    if not np.isfinite(radius_m):
        raise ValueError(
            "'radius_m' must be finite."
        )

    if radius_m < 0:
        raise ValueError(
            "'radius_m' must be greater than or equal to zero."
        )

    return radius_m


def _validate_neighbourhood_method(
    neighbourhood: str,
) -> None:
    """Validate the requested HEALPix neighbourhood definition."""

    if neighbourhood not in {
        "cell_center",
        "cone_coverage",
    }:
        raise ValueError(
            "'neighbourhood' must be either "
            "'cell_center' or 'cone_coverage'."
        )


def _validate_reduction(
    reduction: str,
) -> str:
    """Validate and normalise a reduction name."""

    if not isinstance(reduction, str):
        raise TypeError(
            "'reduction' must be a string."
        )

    reduction = reduction.strip().lower()

    if reduction not in _VALID_REDUCTIONS:
        allowed = ", ".join(
            sorted(_VALID_REDUCTIONS)
        )

        raise ValueError(
            f"Unknown reduction {reduction!r}. "
            f"Expected one of: {allowed}."
        )

    return reduction


def _output_dtype(
    input_dtype: torch.dtype,
    reduction: str,
) -> torch.dtype:
    """Return the documented output dtype for a reduction."""

    if reduction in {
        "mean",
        "median",
        "std",
    }:
        if input_dtype.is_floating_point:
            return input_dtype

        return torch.float64

    if reduction == "sum":
        if input_dtype.is_floating_point:
            return input_dtype

        return torch.int64

    if reduction == "count":
        return torch.int64

    if reduction in {
        "any",
        "all",
    }:
        return torch.bool

    # min, max and mode preserve the input dtype.
    return input_dtype


def _numerical_median(
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute a NumPy-compatible numerical median.

    PyTorch's ``torch.median`` does not have the same even-sample
    semantics as NumPy's numerical median. We therefore sort the
    samples explicitly and average the two central values when the
    neighbourhood size is even.

    NaN values propagate.
    """

    sorted_values = torch.sort(
        values,
        dim=-1,
    ).values

    number_of_values = sorted_values.shape[-1]

    if number_of_values % 2 == 1:
        result = sorted_values[
            ...,
            number_of_values // 2,
        ]

    else:
        left = sorted_values[
            ...,
            number_of_values // 2 - 1,
        ]

        right = sorted_values[
            ...,
            number_of_values // 2,
        ]

        result = (left + right) / 2

    if sorted_values.dtype.is_floating_point:
        has_nan = torch.isnan(
            sorted_values
        ).any(dim=-1)

        if torch.any(has_nan):
            result = torch.where(
                has_nan,
                torch.full_like(
                    result,
                    torch.nan,
                ),
                result,
            )

    return result


def _categorical_mode(
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute a deterministic categorical mode.

    Ties are resolved by returning the smallest category value.

    ``values`` has shape ``[batch, neighbourhood]``.
    """

    results = []

    for row in values:
        categories, counts = torch.unique(
            row,
            sorted=True,
            return_counts=True,
        )

        maximum_count = torch.max(counts)

        tied_categories = categories[
            counts == maximum_count
        ]

        results.append(
            tied_categories[0]
        )

    return torch.stack(results)


def _reduce_selected(
    selected: torch.Tensor,
    reduction: str,
    *,
    target_cell: int,
) -> torch.Tensor:
    """Apply one reduction to one effective neighbourhood.

    ``selected`` has shape::

        [flattened_batch, neighbourhood_size]
    """

    batch_size = selected.shape[0]
    neighbourhood_size = selected.shape[-1]

    if neighbourhood_size == 0:
        if reduction == "sum":
            return torch.zeros(
                batch_size,
                dtype=_output_dtype(
                    selected.dtype,
                    reduction,
                ),
                device=selected.device,
            )

        if reduction == "count":
            return torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=selected.device,
            )

        if reduction == "any":
            return torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=selected.device,
            )

        if reduction == "all":
            return torch.ones(
                batch_size,
                dtype=torch.bool,
                device=selected.device,
            )

        raise ValueError(
            f"Cell {target_cell} has an empty neighbourhood "
            f"for reduction={reduction!r}. "
            "Use include_self=True, increase radius_m, enlarge "
            "the processing domain, or select a reduction with "
            "a defined empty-neighbourhood identity."
        )

    if reduction == "count":
        return torch.full(
            (batch_size,),
            neighbourhood_size,
            dtype=torch.int64,
            device=selected.device,
        )

    if reduction == "any":
        return torch.any(
            selected,
            dim=-1,
        )

    if reduction == "all":
        return torch.all(
            selected,
            dim=-1,
        )

    if reduction == "sum":
        if not selected.dtype.is_floating_point:
            selected = selected.to(
                torch.int64
            )

        return selected.sum(
            dim=-1,
        )

    if reduction == "min":
        return selected.min(
            dim=-1
        ).values

    if reduction == "max":
        return selected.max(
            dim=-1
        ).values

    if reduction == "mode":
        return _categorical_mode(
            selected
        )

    # mean, median and std are numerical floating-point reductions.
    if not selected.dtype.is_floating_point:
        selected = selected.to(
            torch.float64
        )

    if reduction == "mean":
        return selected.mean(
            dim=-1
        )

    if reduction == "median":
        return _numerical_median(
            selected
        )

    if reduction == "std":
        # Population standard deviation, matching numpy.std default.
        return torch.std(
            selected,
            dim=-1,
            correction=0,
        )

    raise AssertionError(
        f"Unhandled reduction: {reduction}"
    )


class HealPixNeighbourReducer:
    """Reusable HEALPix neighbourhood-reduction operator.

    Geometry is constructed once during initialisation and can then be
    reused for multiple fields sharing the same ``cell_ids`` and
    processing ``domain``.

    Parameters
    ----------
    cell_ids
        One-dimensional array of unique NESTED HEALPix cell IDs for
        which input values are available.

        Input order is preserved and defines the relationship between
        ``cell_ids`` and the last dimension of ``values``.
    refinement_level
        HEALPix refinement level, between 0 and 29.
    radius_m
        Physical neighbourhood radius in metres.
    domain
        Optional processing/output domain.

        If ``None``, all cells in ``cell_ids`` form the processing
        domain.

        If supplied, every cell in ``domain`` must also occur in
        ``cell_ids``.

        Cells outside ``domain`` never participate in a reduction,
        even when they lie geometrically within ``radius_m`` and have
        an available value in ``values``.

        Output ordering follows ``domain`` exactly.
    neighbourhood
        HEALPix neighbourhood definition.

        ``"cell_center"``
            Include HEALPix cells whose centres are within
            ``radius_m`` metres of the target-cell centre.

        ``"cone_coverage"``
            Include cells intersecting the circular coverage region
            used by ``healpix_geo.nested.cone_coverage``.
    include_self
        Whether the target cell participates in its own reduction.
        Defaults to ``True``.
    ellipsoid
        Reference ellipsoid used by the shared morphology geometry.
        Defaults to ``"WGS84"``.

    Notes
    -----
    Near a partial-domain boundary, the number of contributing samples
    can decrease because cells outside ``domain`` are excluded rather
    than padded with artificial values.
    """

    def __init__(
        self,
        cell_ids,
        refinement_level: int,
        *,
        radius_m: float,
        domain=None,
        neighbourhood: NeighbourhoodMethod = "cell_center",
        include_self: bool = True,
        ellipsoid: str = "WGS84",
    ) -> None:
        refinement_level = _validate_refinement_level(
            refinement_level
        )

        radius_m = _validate_radius(
            radius_m
        )

        _validate_neighbourhood_method(
            neighbourhood
        )

        self.refinement_level = refinement_level
        self.radius_m = radius_m
        self.neighbourhood = neighbourhood
        self.include_self = bool(
            include_self
        )
        self.ellipsoid = ellipsoid

        self.cell_ids = _normalise_ids(
            cell_ids,
            name="cell_ids",
            refinement_level=refinement_level,
        )

        if domain is None:
            self.domain = self.cell_ids.copy()

        else:
            self.domain = _normalise_ids(
                domain,
                name="domain",
                refinement_level=refinement_level,
            )

            if not np.all(
                np.isin(
                    self.domain,
                    self.cell_ids,
                )
            ):
                raise ValueError(
                    "Every cell in 'domain' must occur in "
                    "'cell_ids'."
                )

        self._source_positions = {
            int(cell): position
            for position, cell
            in enumerate(
                self.cell_ids.tolist()
            )
        }

        self._domain_set = set(
            self.domain.tolist()
        )

        self._positions = (
            self._build_positions()
        )

    @property
    def output_cell_ids(
        self,
    ) -> np.ndarray:
        """Return output HEALPix IDs in exact output order."""

        return self.domain.copy()

    def _build_positions(
        self,
    ) -> tuple[np.ndarray, ...]:
        """Build input-array positions for each effective neighbourhood."""

        if self.domain.size == 0:
            return tuple()

        if self.radius_m == 0:
            geometrical_neighbourhoods = [
                np.asarray(
                    [cell],
                    dtype=np.uint64,
                )
                for cell in self.domain
            ]

        else:
            geometrical_neighbourhoods = build_neighbourhoods(
                self.domain,
                self.radius_m,
                self.refinement_level,
                neighbourhood=self.neighbourhood,
                ellipsoid=self.ellipsoid,
            )

        all_positions = []

        for target_cell, neighbours in zip(
            self.domain,
            geometrical_neighbourhoods,
            strict=True,
        ):
            target_cell_int = int(
                target_cell
            )

            positions = []

            for neighbour in neighbours:
                neighbour_int = int(
                    neighbour
                )

                # Partial-domain semantics:
                #
                # geometrically neighbouring cells outside the
                # processing domain are absent from the effective
                # neighbourhood. They are NOT zero padded.
                if neighbour_int not in self._domain_set:
                    continue

                if (
                    not self.include_self
                    and neighbour_int
                    == target_cell_int
                ):
                    continue

                source_position = (
                    self._source_positions.get(
                        neighbour_int
                    )
                )

                # This should normally be redundant because domain is
                # validated as a subset of cell_ids, but keeping the
                # check makes the geometry-to-data relationship
                # explicit and robust.
                if source_position is None:
                    continue

                positions.append(
                    source_position
                )

            if self.include_self:
                target_position = (
                    self._source_positions[
                        target_cell_int
                    ]
                )

                if (
                    target_position
                    not in positions
                ):
                    positions.append(
                        target_position
                    )

            # Reductions are order-independent. Sorting by source-array
            # position makes the internal representation deterministic.
            positions = sorted(
                set(positions)
            )

            all_positions.append(
                np.asarray(
                    positions,
                    dtype=np.int64,
                )
            )

        return tuple(
            all_positions
        )

    def __call__(
        self,
        values: ArrayLike,
        *,
        reduction: Reduction = "mean",
    ) -> ArrayLike:
        """Apply a neighbourhood reduction to a field.

        Parameters
        ----------
        values
            NumPy array or PyTorch tensor.

            The final dimension must correspond one-to-one with
            ``cell_ids`` supplied when constructing this reducer.

            Arbitrary leading dimensions are supported.
        reduction
            Reduction to apply.

            Supported reductions are:

            - ``"mean"``
            - ``"sum"``
            - ``"min"``
            - ``"max"``
            - ``"median"``
            - ``"count"``
            - ``"std"``
            - ``"any"``
            - ``"all"``
            - ``"mode"``

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Same backend as ``values``.

            The final output dimension corresponds to
            :attr:`output_cell_ids`.

        Notes
        -----
        Dtype policy:

        - ``mean``, ``median``, ``std``:
          preserve floating dtype; integer/bool input becomes float64.
        - ``sum``:
          preserve floating dtype; integer/bool input becomes int64.
        - ``count``:
          int64.
        - ``any``, ``all``:
          bool and require boolean input.
        - ``min``, ``max``, ``mode``:
          preserve input dtype.
        """

        reduction = _validate_reduction(
            reduction
        )

        is_numpy = isinstance(
            values,
            np.ndarray,
        )

        if not is_numpy and not torch.is_tensor(
            values
        ):
            raise TypeError(
                "'values' must be a numpy.ndarray "
                "or torch.Tensor."
            )

        if is_numpy:
            try:
                tensor = torch.as_tensor(
                    values
                )
            except Exception as error:
                raise TypeError(
                    "The NumPy dtype of 'values' cannot be "
                    "converted to a PyTorch tensor."
                ) from error
        else:
            tensor = values

        if torch.is_complex(
            tensor
        ):
            raise TypeError(
                "Complex-valued fields are not supported."
            )

        if tensor.ndim < 1:
            raise ValueError(
                "'values' must have at least one dimension."
            )

        if (
            tensor.shape[-1]
            != self.cell_ids.size
        ):
            raise ValueError(
                "The last dimension of 'values' must match "
                "the number of 'cell_ids'."
            )

        if (
            reduction in {"any", "all"}
            and tensor.dtype != torch.bool
        ):
            raise TypeError(
                f"reduction={reduction!r} requires boolean input."
            )

        if (
            reduction == "mode"
            and tensor.dtype.is_floating_point
        ):
            raise TypeError(
                "reduction='mode' requires integer or boolean input."
            )

        leading_shape = tuple(
            tensor.shape[:-1]
        )

        flattened = tensor.reshape(
            -1,
            tensor.shape[-1],
        )

        output_dtype = _output_dtype(
            tensor.dtype,
            reduction,
        )

        if len(self._positions) == 0:
            output = torch.empty(
                (*leading_shape, 0),
                dtype=output_dtype,
                device=tensor.device,
            )

        else:
            output_columns = []

            for target_cell, positions in zip(
                self.domain,
                self._positions,
                strict=True,
            ):
                index = torch.as_tensor(
                    positions,
                    dtype=torch.long,
                    device=tensor.device,
                )

                selected = (
                    flattened.index_select(
                        -1,
                        index,
                    )
                )

                reduced = _reduce_selected(
                    selected,
                    reduction,
                    target_cell=int(
                        target_cell
                    ),
                )

                output_columns.append(
                    reduced
                )

            output = torch.stack(
                output_columns,
                dim=-1,
            )

            output = output.reshape(
                *leading_shape,
                self.domain.size,
            )

        if is_numpy:
            return (
                output
                .detach()
                .cpu()
                .numpy()
            )

        return output


def neighbour_reduce(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    radius_m: float,
    reduction: Reduction = "mean",
    domain=None,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    include_self: bool = True,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Reduce field values over local HEALPix neighbourhoods.

    Parameters
    ----------
    values
        NumPy array or PyTorch tensor.

        The last dimension is aligned one-to-one with ``cell_ids``.
        Arbitrary leading dimensions are supported.
    cell_ids
        One-dimensional array of unique NESTED HEALPix cell IDs for
        which input values are available.

        The order of ``cell_ids`` is preserved.
    refinement_level
        HEALPix refinement level, between 0 and 29.
    radius_m
        Physical neighbourhood radius in metres.
    reduction
        Reduction applied to each effective local neighbourhood.

        Supported values are ``"mean"``, ``"sum"``, ``"min"``,
        ``"max"``, ``"median"``, ``"count"``, ``"std"``,
        ``"any"``, ``"all"``, and ``"mode"``.
    domain
        Optional processing/output domain.

        If ``None``, all cells in ``cell_ids`` form the domain.

        If provided, ``domain`` must be a subset of ``cell_ids``.
        The output contains one value per domain cell and follows the
        exact order of ``domain``.

        Cells outside ``domain`` never participate in a reduction,
        even when they lie geometrically within ``radius_m`` and have
        an input value in ``values``.

        Such cells are not zero-padded or assigned another synthetic
        value. They are simply absent from the effective neighbourhood.
    neighbourhood
        Definition of the HEALPix neighbourhood.

        ``"cell_center"``
            Include cells whose centres are within ``radius_m`` metres
            of the target-cell centre.

        ``"cone_coverage"``
            Include cells intersecting the corresponding circular
            coverage region.
    include_self
        Whether the target cell participates in its own reduction.
        Defaults to ``True``.
    ellipsoid
        Reference ellipsoid used for physical-distance calculations.
        Defaults to ``"WGS84"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Reduced values.

        The backend follows ``values``. The final dimension follows
        ``domain`` when explicitly supplied and ``cell_ids`` otherwise.

    Notes
    -----
    Near a partial-domain boundary, the effective neighbourhood can
    contain fewer samples because cells outside the processing domain
    are excluded rather than padded.

    Numerical ``median`` and categorical ``mode`` are intentionally
    different operations. For an even number of numerical samples,
    ``median`` averages the two central sorted values.

    Empty effective neighbourhoods have the following identities:

    - ``sum`` -> 0
    - ``count`` -> 0
    - ``any`` -> False
    - ``all`` -> True

    Other reductions raise ``ValueError`` on an empty effective
    neighbourhood.
    """

    reducer = HealPixNeighbourReducer(
        cell_ids,
        refinement_level,
        radius_m=radius_m,
        domain=domain,
        neighbourhood=neighbourhood,
        include_self=include_self,
        ellipsoid=ellipsoid,
    )

    return reducer(
        values,
        reduction=reduction,
    )


def median_filter(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    radius_m: float,
    domain=None,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    include_self: bool = True,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply a numerical median over HEALPix neighbourhoods."""

    return neighbour_reduce(
        values,
        cell_ids,
        refinement_level,
        radius_m=radius_m,
        reduction="median",
        domain=domain,
        neighbourhood=neighbourhood,
        include_self=include_self,
        ellipsoid=ellipsoid,
    )


def mean_filter(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    radius_m: float,
    domain=None,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    include_self: bool = True,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply a mean over HEALPix neighbourhoods."""

    return neighbour_reduce(
        values,
        cell_ids,
        refinement_level,
        radius_m=radius_m,
        reduction="mean",
        domain=domain,
        neighbourhood=neighbourhood,
        include_self=include_self,
        ellipsoid=ellipsoid,
    )


def min_filter(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    radius_m: float,
    domain=None,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    include_self: bool = True,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply a minimum over HEALPix neighbourhoods."""

    return neighbour_reduce(
        values,
        cell_ids,
        refinement_level,
        radius_m=radius_m,
        reduction="min",
        domain=domain,
        neighbourhood=neighbourhood,
        include_self=include_self,
        ellipsoid=ellipsoid,
    )


def max_filter(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    radius_m: float,
    domain=None,
    neighbourhood: NeighbourhoodMethod = "cell_center",
    include_self: bool = True,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply a maximum over HEALPix neighbourhoods."""

    return neighbour_reduce(
        values,
        cell_ids,
        refinement_level,
        radius_m=radius_m,
        reduction="max",
        domain=domain,
        neighbourhood=neighbourhood,
        include_self=include_self,
        ellipsoid=ellipsoid,
    )


__all__ = [
    "HealPixNeighbourReducer",
    "neighbour_reduce",
    "median_filter",
    "mean_filter",
    "min_filter",
    "max_filter",
]
