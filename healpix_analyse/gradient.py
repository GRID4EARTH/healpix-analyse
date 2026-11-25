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
from healpix_analyse._stencil_backend import (
    BACKENDS,
    GradientStencil,
    lsq_gradient,
    triton_available,
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
    """Evaluate differentiable least-squares gradients using Torch.

    Reference implementation (materialises ``[..., N, K]``); ``values``
    may have leading dimensions.  The fast path is
    :func:`healpix_analyse._stencil_backend.lsq_gradient`.
    """

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
        -1,
        safe_positions.reshape(-1),
    ).reshape(
        *work.shape[:-1],
        *safe_positions.shape,
    )

    center_values = torch.index_select(
        work,
        -1,
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
        & finite_center[..., None]
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
        - safe_center[..., None]
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
        dim=-1,
    )

    s_en = torch.sum(
        east_safe
        * north_safe,
        dim=-1,
    )

    s_nn = torch.sum(
        north_safe
        * north_safe,
        dim=-1,
    )

    rhs_e = torch.sum(
        east_safe
        * delta,
        dim=-1,
    )

    rhs_n = torch.sum(
        north_safe
        * delta,
        dim=-1,
    )

    determinant = (
        s_ee
        * s_nn
        - s_en
        * s_en
    )

    number_of_valid_neighbours = torch.sum(
        valid,
        dim=-1,
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


class HealPixGradient:
    """Reusable least-squares gradient operator.

    The immediate-neighbour geometry (WGS84 geodesics) is built once at
    construction; every call then only gathers the values.  This is the form
    to use inside a model or a training loop: :func:`gradient` rebuilds the
    geometry at every call.

    Parameters
    ----------
    cell_ids
        NESTED HEALPix cell IDs of the input values (last dimension).
    refinement_level
        HEALPix refinement level.
    domain
        Optional processing/output domain, a subset of ``cell_ids``; see
        :func:`gradient`.
    method
        ``"least_squares"`` (the only method).

    Notes
    -----
    A call accepts values of shape ``[..., N]`` (NumPy or torch) and returns
    ``(grad_east, grad_north)`` of shape ``[..., N_domain]``.

    Torch tensors of float32 / float64 run a fused kernel: Triton on CUDA
    when it is installed, a loop over the neighbours otherwise.  On CUDA with
    Triton, float16 / bfloat16 data run the same kernels with fp32
    accumulation (the reference would compute the normal equations in the
    16-bit dtype, where squared offsets in metres overflow).  Neither
    materialises the ``[..., N, K]`` gathered tensor, and the backward pass
    is written by hand (on CUDA, a gather on the transposed neighbourhoods:
    no atomics, deterministic).  Other dtypes and NumPy arrays use the
    reference implementation.  The values agree with the reference to
    rounding (about 1e-12 relative in float64).
    """

    def __init__(
        self,
        cell_ids,
        refinement_level: int,
        *,
        domain=None,
        method: GradientMethod = "least_squares",
    ) -> None:
        self.refinement_level = _validate_refinement_level(
            refinement_level
        )
        self.method = _validate_method(
            method
        )
        self.cell_ids = _normalise_ids(
            cell_ids,
            name="cell_ids",
            refinement_level=self.refinement_level,
        )

        if domain is None:
            self.domain = self.cell_ids.copy()
        else:
            self.domain = _normalise_ids(
                domain,
                name="domain",
                refinement_level=self.refinement_level,
            )

        (
            self._geometry,
            self._neighbour_positions,
            self._center_positions,
        ) = _prepare_gradient_geometry(
            self.cell_ids,
            self.domain,
            self.refinement_level,
        )

        self._stencil = None

    @property
    def output_cell_ids(
        self,
    ) -> np.ndarray:
        """Output HEALPix IDs in exact output order."""

        return self.domain.copy()

    def _gradient_stencil(
        self,
    ) -> GradientStencil:
        if self._stencil is None:
            geometry = self._geometry
            self._stencil = GradientStencil(
                self._neighbour_positions,
                geometry.valid_mask,
                geometry.east_offset_m,
                geometry.north_offset_m,
                self._center_positions,
                self.cell_ids.size,
            )

        return self._stencil

    def __call__(
        self,
        values: ArrayLike,
        *,
        backend: str = "auto",
    ) -> tuple[
        ArrayLike,
        ArrayLike,
    ]:
        """Return ``(grad_east, grad_north)`` of ``values [..., N]``.

        ``backend`` is ``"auto"``, ``"triton"`` or ``"torch"`` (fused
        kernels only; ``"auto"`` uses Triton on CUDA when available).
        """

        if backend not in BACKENDS:
            raise ValueError(
                f"'backend' must be one of {BACKENDS}, got {backend!r}."
            )

        n_cells = self.cell_ids.size

        if torch.is_tensor(values):
            if values.ndim < 1 or values.shape[-1] != n_cells:
                raise ValueError(
                    "The last dimension of 'values' must match "
                    "the number of 'cell_ids'."
                )

            if values.dtype == torch.bool:
                raise TypeError(
                    "'values' must contain numerical scalar values, not bool."
                )

            fused = values.dtype in (
                torch.float32,
                torch.float64,
            ) or (
                # 16-bit data: fp32 accumulation in the Triton kernels; the
                # reference computes in the 16-bit dtype itself.
                values.dtype in (torch.float16, torch.bfloat16)
                and values.is_cuda
                and backend != "torch"
                and triton_available()
            )

            if not fused or self.domain.size == 0:
                return _torch_gradient_from_geometry(
                    values,
                    self._geometry,
                    self._neighbour_positions,
                    self._center_positions,
                )

            leading = tuple(values.shape[:-1])
            flat = values.reshape(-1, n_cells)
            grad_east, grad_north = lsq_gradient(
                flat,
                self._gradient_stencil(),
                backend=backend,
            )

            return (
                grad_east.reshape(*leading, self.domain.size),
                grad_north.reshape(*leading, self.domain.size),
            )

        array = np.asarray(
            values
        )

        if array.ndim < 1 or array.shape[-1] != n_cells:
            raise ValueError(
                "The last dimension of 'values' must match "
                "the number of 'cell_ids'."
            )

        if array.dtype == np.bool_ or not np.issubdtype(
            array.dtype,
            np.number,
        ):
            raise TypeError(
                "'values' must contain numerical scalar values."
            )

        if array.ndim == 1:
            return _numpy_gradient_from_geometry(
                array,
                self._geometry,
                self._neighbour_positions,
                self._center_positions,
            )

        leading = array.shape[:-1]
        rows = [
            _numpy_gradient_from_geometry(
                row,
                self._geometry,
                self._neighbour_positions,
                self._center_positions,
            )
            for row in array.reshape(-1, n_cells)
        ]
        shape = (*leading, self.domain.size)

        return (
            np.stack([row[0] for row in rows]).reshape(shape),
            np.stack([row[1] for row in rows]).reshape(shape),
        )

    def magnitude(
        self,
        values: ArrayLike,
        **kwargs,
    ) -> ArrayLike:
        """Magnitude of the local geographic gradient."""

        grad_east, grad_north = self(
            values,
            **kwargs,
        )

        return _hypot(
            grad_east,
            grad_north,
        )

    def directional_derivative(
        self,
        values: ArrayLike,
        azimuth_rad,
        **kwargs,
    ) -> ArrayLike:
        """Derivative along a geographic azimuth (clockwise from North)."""

        grad_east, grad_north = self(
            values,
            **kwargs,
        )

        return _directional(
            grad_east,
            grad_north,
            azimuth_rad,
        )


def _hypot(
    grad_east,
    grad_north,
):
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


def _directional(
    grad_east,
    grad_north,
    azimuth_rad,
):
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

    operator = HealPixGradient(
        cells,
        refinement_level,
        domain=domain,
        method=method,
    )

    return operator(
        values
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

    return _hypot(
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

    return _directional(
        grad_east,
        grad_north,
        azimuth_rad,
    )


__all__ = [
    "GradientMethod",
    "HealPixGradient",
    "directional_derivative",
    "gradient",
    "gradient_magnitude",
]
