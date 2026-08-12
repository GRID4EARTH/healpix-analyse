"""Local scalar-field gradients on NESTED HEALPix grids.

The gradient is estimated in a local geographic tangent basis:

- positive East
- positive North

Immediate HEALPix neighbours are discovered topologically using the shared
neighbourhood infrastructure.  Their actual WGS84 geometry is then used to
fit a two-dimensional local linear model.

No Cartesian x/y HEALPix-index directions are assumed.

Domain semantics
----------------
``cell_ids`` identifies all cells for which input values are available.

``domain`` identifies the valid processing/output domain.

If ``domain is None``, the processing domain is exactly ``cell_ids``.

Only cells belonging to ``domain`` participate in a local fit.  Cells
outside ``domain`` are absent; they are not interpreted as zero, False,
NaN, or any other padding value.

If the remaining finite neighbours do not provide a rank-2 local tangent
fit, the gradient for that target cell is NaN.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from healpix_analyse._neighbourhood import (
    RelativeNeighbourhoodGeometry,
    build_relative_geometry,
)
from healpix_analyse.neighbour_reduce import (
    _normalise_ids,
    _validate_refinement_level,
)


ArrayLike = np.ndarray | torch.Tensor

GradientMethod = Literal[
    "least_squares",
]


def _validate_method(
    method: str,
) -> str:
    """Validate the gradient-estimation method."""

    if not isinstance(
        method,
        str,
    ):
        raise TypeError(
            "'method' must be a string."
        )

    method = method.strip().lower()

    if method != "least_squares":
        raise ValueError(
            "'method' must be 'least_squares'."
        )

    return method


def _validate_values(
    values: ArrayLike,
    *,
    number_of_cells: int,
) -> None:
    """Validate a one-dimensional numerical scalar field."""

    if torch.is_tensor(values):
        if values.ndim != 1:
            raise ValueError(
                "'values' must be a one-dimensional array."
            )

        if values.numel() != number_of_cells:
            raise ValueError(
                "'values' and 'cell_ids' must have the same length."
            )

        if values.dtype == torch.bool:
            raise TypeError(
                "'values' must contain numerical scalar values, not bool."
            )

        return

    array = np.asarray(
        values
    )

    if array.ndim != 1:
        raise ValueError(
            "'values' must be a one-dimensional array."
        )

    if array.size != number_of_cells:
        raise ValueError(
            "'values' and 'cell_ids' must have the same length."
        )

    if array.dtype == np.bool_ or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            "'values' must contain numerical scalar values."
        )


def _lookup_positions(
    query_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> np.ndarray:
    """Find query cell positions in an unordered unique reference array.

    Missing query IDs are returned as ``-1``.

    The implementation uses sorting + searchsorted rather than a Python
    dictionary lookup for every centre-neighbour pair.
    """

    query = np.asarray(
        query_ids
    )

    reference = np.asarray(
        reference_ids,
        dtype=np.uint64,
    )

    result = np.full(
        query.shape,
        -1,
        dtype=np.int64,
    )

    if query.size == 0 or reference.size == 0:
        return result

    valid_query = (
        query >= 0
    )

    if not np.any(
        valid_query
    ):
        return result

    query_valid = query[
        valid_query
    ].astype(
        np.uint64,
        copy=False,
    )

    order = np.argsort(
        reference
    )

    sorted_reference = reference[
        order
    ]

    insertion = np.searchsorted(
        sorted_reference,
        query_valid,
    )

    inside = (
        insertion
        < sorted_reference.size
    )

    found = np.zeros(
        query_valid.shape,
        dtype=bool,
    )

    if np.any(
        inside
    ):
        found[
            inside
        ] = (
            sorted_reference[
                insertion[
                    inside
                ]
            ]
            == query_valid[
                inside
            ]
        )

    valid_result = np.full(
        query_valid.shape,
        -1,
        dtype=np.int64,
    )

    valid_result[
        found
    ] = order[
        insertion[
            found
        ]
    ]

    result[
        valid_query
    ] = valid_result

    return result


def _prepare_gradient_geometry(
    cell_ids: np.ndarray,
    domain: np.ndarray,
    refinement_level: int,
) -> tuple[
    RelativeNeighbourhoodGeometry,
    np.ndarray,
    np.ndarray,
]:
    """Prepare immediate-neighbour geometry and value-array positions.

    Returns
    -------
    geometry
        Geographic relative geometry for every output cell.

    neighbour_positions
        Shape ``(N, K)``. Positions into the original values array.
        ``-1`` means that the geometric neighbour is outside ``domain``.

    center_positions
        Shape ``(N,)``. Positions of output cells in the original values
        array.
    """

    center_positions = _lookup_positions(
        domain,
        cell_ids,
    )

    if np.any(
        center_positions < 0
    ):
        raise ValueError(
            "'domain' must be a subset of 'cell_ids'."
        )

    geometry = build_relative_geometry(
        domain,
        refinement_level,
        ring=1,
        ellipsoid="WGS84",
    )

    # First locate neighbours within the domain itself.
    domain_positions = _lookup_positions(
        geometry.neighbour_ids,
        domain,
    )

    neighbour_positions = np.full(
        domain_positions.shape,
        -1,
        dtype=np.int64,
    )

    valid = (
        domain_positions >= 0
    )

    # Convert domain-relative positions to positions in the original
    # ``values`` / ``cell_ids`` arrays.
    neighbour_positions[
        valid
    ] = center_positions[
        domain_positions[
            valid
        ]
    ]

    return (
        geometry,
        neighbour_positions,
        center_positions,
    )


def _numpy_output_dtype(
    dtype: np.dtype,
) -> np.dtype:
    """Return the numerical dtype used for NumPy gradients."""

    if np.issubdtype(
        dtype,
        np.floating,
    ):
        return dtype

    return np.dtype(
        np.float64
    )


def _torch_output_dtype(
    values: torch.Tensor,
) -> torch.dtype:
    """Return a floating dtype suitable for Torch gradient arithmetic."""

    if values.dtype.is_floating_point:
        return values.dtype

    # MPS does not support float64 consistently.  Torch's default floating
    # dtype provides a portable promotion for integer inputs.
    return torch.get_default_dtype()


def _numpy_gradient_from_geometry(
    values: np.ndarray,
    geometry: RelativeNeighbourhoodGeometry,
    neighbour_positions: np.ndarray,
    center_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate least-squares gradients using NumPy."""

    dtype = _numpy_output_dtype(
        values.dtype
    )

    work = values.astype(
        dtype,
        copy=False,
    )

    east = geometry.east_offset_m.astype(
        dtype,
        copy=False,
    )

    north = geometry.north_offset_m.astype(
        dtype,
        copy=False,
    )

    safe_positions = np.maximum(
        neighbour_positions,
        0,
    )

    gathered = work[
        safe_positions
    ]

    center_values = work[
        center_positions
    ]

    finite_center = np.isfinite(
        center_values
    )

    finite_neighbour = np.isfinite(
        gathered
    )

    valid = (
        geometry.valid_mask
        & (neighbour_positions >= 0)
        & finite_neighbour
        & finite_center[:, None]
    )

    # Remove NaN / inf values before arithmetic.  Invalid entries receive
    # zero weight below and therefore do not contribute to the fit.
    safe_gathered = np.where(
        finite_neighbour,
        gathered,
        0.0,
    )

    safe_center = np.where(
        finite_center,
        center_values,
        0.0,
    )

    delta = (
        safe_gathered
        - safe_center[:, None]
    )

    valid_float = valid.astype(
        dtype,
        copy=False,
    )

    east_safe = np.where(
        valid,
        east,
        0.0,
    )

    north_safe = np.where(
        valid,
        north,
        0.0,
    )

    delta = (
        delta
        * valid_float
    )

    # Normal equations for
    #
    #   delta_f ~= gradient_east * delta_east
    #            + gradient_north * delta_north
    #
    # This avoids constructing a separate small least-squares object for
    # every HEALPix cell and is equivalent to the unweighted 2-D fit when
    # the local geometry has rank two.
    s_ee = np.sum(
        east_safe
        * east_safe,
        axis=1,
    )

    s_en = np.sum(
        east_safe
        * north_safe,
        axis=1,
    )

    s_nn = np.sum(
        north_safe
        * north_safe,
        axis=1,
    )

    rhs_e = np.sum(
        east_safe
        * delta,
        axis=1,
    )

    rhs_n = np.sum(
        north_safe
        * delta,
        axis=1,
    )

    determinant = (
        s_ee
        * s_nn
        - s_en
        * s_en
    )

    number_of_valid_neighbours = np.sum(
        valid,
        axis=1,
    )

    epsilon = np.finfo(
        dtype
    ).eps

    rank_scale = np.maximum(
        s_ee
        * s_nn,
        1.0,
    )

    full_rank = (
        determinant
        > (
            100.0
            * epsilon
            * rank_scale
        )
    )

    solvable = (
        finite_center
        & (number_of_valid_neighbours >= 2)
        & full_rank
    )

    safe_determinant = np.where(
        solvable,
        determinant,
        1.0,
    )

    grad_east = (
        rhs_e
        * s_nn
        - rhs_n
        * s_en
    ) / safe_determinant

    grad_north = (
        rhs_n
        * s_ee
        - rhs_e
        * s_en
    ) / safe_determinant

    grad_east = np.where(
        solvable,
        grad_east,
        np.nan,
    )

    grad_north = np.where(
        solvable,
        grad_north,
        np.nan,
    )

    return (
        grad_east,
        grad_north,
    )


def _torch_gradient_from_geometry(
    values: torch.Tensor,
    geometry: RelativeNeighbourhoodGeometry,
    neighbour_positions: np.ndarray,
    center_positions: np.ndarray,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate differentiable least-squares gradients using Torch."""

    dtype = _torch_output_dtype(
        values
    )

    work = values.to(
        dtype=dtype
    )

    device = work.device

    east = torch.as_tensor(
        geometry.east_offset_m,
        dtype=dtype,
        device=device,
    )

    north = torch.as_tensor(
        geometry.north_offset_m,
        dtype=dtype,
        device=device,
    )

    geometry_valid = torch.as_tensor(
        geometry.valid_mask,
        dtype=torch.bool,
        device=device,
    )

    positions = torch.as_tensor(
        neighbour_positions,
        dtype=torch.long,
        device=device,
    )

    centers = torch.as_tensor(
        center_positions,
        dtype=torch.long,
        device=device,
    )

    safe_positions = positions.clamp_min(
        0
    )

    gathered = torch.index_select(
        work,
        0,
        safe_positions.reshape(-1),
    ).reshape(
        safe_positions.shape
    )

    center_values = torch.index_select(
        work,
        0,
        centers,
    )

    finite_center = torch.isfinite(
        center_values
    )

    finite_neighbour = torch.isfinite(
        gathered
    )

    valid = (
        geometry_valid
        & (positions >= 0)
        & finite_neighbour
        & finite_center[:, None]
    )

    safe_gathered = torch.where(
        finite_neighbour,
        gathered,
        torch.zeros_like(
            gathered
        ),
    )

    safe_center = torch.where(
        finite_center,
        center_values,
        torch.zeros_like(
            center_values
        ),
    )

    delta = (
        safe_gathered
        - safe_center[:, None]
    )

    zeros = torch.zeros_like(
        east
    )

    east_safe = torch.where(
        valid,
        east,
        zeros,
    )

    north_safe = torch.where(
        valid,
        north,
        zeros,
    )

    delta = torch.where(
        valid,
        delta,
        torch.zeros_like(
            delta
        ),
    )

    s_ee = torch.sum(
        east_safe
        * east_safe,
        dim=1,
    )

    s_en = torch.sum(
        east_safe
        * north_safe,
        dim=1,
    )

    s_nn = torch.sum(
        north_safe
        * north_safe,
        dim=1,
    )

    rhs_e = torch.sum(
        east_safe
        * delta,
        dim=1,
    )

    rhs_n = torch.sum(
        north_safe
        * delta,
        dim=1,
    )

    determinant = (
        s_ee
        * s_nn
        - s_en
        * s_en
    )

    number_of_valid_neighbours = torch.sum(
        valid,
        dim=1,
    )

    epsilon = torch.finfo(
        dtype
    ).eps

    rank_scale = torch.maximum(
        s_ee
        * s_nn,
        torch.ones_like(
            determinant
        ),
    )

    full_rank = (
        determinant
        > (
            100.0
            * epsilon
            * rank_scale
        )
    )

    solvable = (
        finite_center
        & (number_of_valid_neighbours >= 2)
        & full_rank
    )

    safe_determinant = torch.where(
        solvable,
        determinant,
        torch.ones_like(
            determinant
        ),
    )

    grad_east = (
        rhs_e
        * s_nn
        - rhs_n
        * s_en
    ) / safe_determinant

    grad_north = (
        rhs_n
        * s_ee
        - rhs_e
        * s_en
    ) / safe_determinant

    nan = torch.full_like(
        grad_east,
        float("nan"),
    )

    grad_east = torch.where(
        solvable,
        grad_east,
        nan,
    )

    grad_north = torch.where(
        solvable,
        grad_north,
        nan,
    )

    return (
        grad_east,
        grad_north,
    )


def gradient(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    domain=None,
    method: GradientMethod = "least_squares",
) -> tuple[
    ArrayLike,
    ArrayLike,
]:
    """Estimate the local geographic gradient of a scalar HEALPix field.

    Parameters
    ----------
    values
        One-dimensional NumPy array or Torch tensor containing scalar
        values.

    cell_ids
        NESTED HEALPix cell IDs corresponding to ``values``.

    refinement_level
        HEALPix refinement level.

    domain
        Optional processing/output domain.

        Every domain cell must occur in ``cell_ids``.

        Only cells in ``domain`` participate in gradient estimation.
        Cells outside the domain are absent rather than treated as padding.

        Output ordering follows ``domain`` exactly.  If ``domain`` is
        omitted, output ordering follows ``cell_ids``.

    method
        Gradient estimation method.  Currently only
        ``"least_squares"`` is supported.

    Returns
    -------
    grad_east, grad_north
        Local tangent-gradient components.

        If input values have units ``U``, outputs have units ``U / metre``.

        A cell receives NaN if fewer than two finite domain neighbours are
        available or if the remaining neighbour geometry does not span a
        two-dimensional local tangent plane.

    Notes
    -----
    The operator uses the immediate topological HEALPix neighbourhood
    (ring=1).

    HEALPix index directions are not interpreted as geographic directions.
    East/North are derived from WGS84 centre-to-centre geodesic geometry.
    """

    refinement_level = _validate_refinement_level(
        refinement_level
    )

    _validate_method(
        method
    )

    cells = _normalise_ids(
        cell_ids,
        name="cell_ids",
        refinement_level=refinement_level,
    )

    _validate_values(
        values,
        number_of_cells=cells.size,
    )

    if domain is None:
        output_domain = cells.copy()
    else:
        output_domain = _normalise_ids(
            domain,
            name="domain",
            refinement_level=refinement_level,
        )

    (
        geometry,
        neighbour_positions,
        center_positions,
    ) = _prepare_gradient_geometry(
        cells,
        output_domain,
        refinement_level,
    )

    if torch.is_tensor(
        values
    ):
        return _torch_gradient_from_geometry(
            values,
            geometry,
            neighbour_positions,
            center_positions,
        )

    return _numpy_gradient_from_geometry(
        np.asarray(
            values
        ),
        geometry,
        neighbour_positions,
        center_positions,
    )


def gradient_magnitude(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    domain=None,
    method: GradientMethod = "least_squares",
) -> ArrayLike:
    """Return the magnitude of the local geographic gradient."""

    grad_east, grad_north = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
        method=method,
    )

    if torch.is_tensor(
        grad_east
    ):
        return torch.hypot(
            grad_east,
            grad_north,
        )

    return np.hypot(
        grad_east,
        grad_north,
    )


def directional_derivative(
    values: ArrayLike,
    cell_ids,
    refinement_level: int,
    *,
    azimuth_rad,
    domain=None,
    method: GradientMethod = "least_squares",
) -> ArrayLike:
    """Evaluate the scalar-field derivative along geographic azimuth.

    ``azimuth_rad`` is measured clockwise from geographic North:

    - 0 -> North
    - pi / 2 -> East
    - pi -> South
    - -pi / 2 -> West

    The directional derivative is derived from the local tangent gradient:

        d/ds = grad_east * sin(azimuth)
             + grad_north * cos(azimuth)
    """

    grad_east, grad_north = gradient(
        values,
        cell_ids,
        refinement_level,
        domain=domain,
        method=method,
    )

    if torch.is_tensor(
        grad_east
    ):
        azimuth = torch.as_tensor(
            azimuth_rad,
            dtype=grad_east.dtype,
            device=grad_east.device,
        )

        return (
            grad_east
            * torch.sin(
                azimuth
            )
            + grad_north
            * torch.cos(
                azimuth
            )
        )

    azimuth = np.asarray(
        azimuth_rad,
        dtype=grad_east.dtype,
    )

    return (
        grad_east
        * np.sin(
            azimuth
        )
        + grad_north
        * np.cos(
            azimuth
        )
    )


__all__ = [
    "GradientMethod",
    "directional_derivative",
    "gradient",
    "gradient_magnitude",
]
