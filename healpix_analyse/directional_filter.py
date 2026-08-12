"""Geographical directional filtering on NESTED HEALPix fields.

This module provides a deterministic directional filtering operator whose
spatial semantics are defined by:

- physical distance in metres,
- geographical forward azimuth,
- a requested geographical direction.

It is intended for Earth-observation processing where an operation has a
physical direction such as solar azimuth, illumination direction, wind
direction, or another geographical bearing.

This is deliberately different from :class:`HealPixConv`.

``HealPixConv`` represents a gauge-oriented spherical convolution on a
rotated stencil.  The operator implemented here instead works directly with
the real WGS84 geometry between HEALPix cell centres.

For an output/target cell ``i`` and a contributing neighbour/source cell
``j``::

                     geographic North
                           ^
                           |
                           |
                     j  *  |
                       /   |
                      /    |
                     /     |
                    *------+
                    i

    distance_ij
        = WGS84 geodesic distance from i to j

    bearing_ij
        = WGS84 forward azimuth from i to j

    relative_bearing_ij
        = wrap(bearing_ij - requested_azimuth)

The user-supplied kernel is evaluated as::

    weight_ij = kernel(
        distance_ij,
        relative_bearing_ij,
    )

Azimuth follows the geographical convention used by the gradient API::

    0          = North
    pi / 2     = East
    pi         = South
    3 * pi / 2 = West

Angles increase clockwise from geographic North.

Neighbour selection uses a physical radius in metres.  HEALPix ring number,
Cartesian pixel windows such as ``3x3`` or ``5x5``, and array row/column
directions are not part of the public spatial semantics.

The geometry is independent of signal values.  Neighbour selection and WGS84
relative geometry are therefore delegated to the shared neighbourhood
infrastructure in :mod:`healpix_analyse._neighbourhood`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ._neighbourhood import (
    RelativeNeighbourhoodGeometry,
    build_neighbourhoods,
    relative_geometry_from_neighbourhoods,
)

from ._weighted_neighbourhood import (
    weighted_neighbourhood_reduce,
)

Kernel = Callable[[np.ndarray, np.ndarray], Any]


def _validate_cell_ids(
    cell_ids: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate a one-dimensional array of unique HEALPix cell IDs."""
    raw = np.asarray(cell_ids)

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

    cells = raw.astype(
        np.uint64,
        copy=False,
    )

    if np.unique(cells).size != cells.size:
        raise ValueError(
            f"'{name}' must not contain duplicate HEALPix cell IDs."
        )

    return cells


def _validate_max_distance(
    max_distance_m: float,
) -> float:
    """Validate the physical support radius."""
    if isinstance(
        max_distance_m,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'max_distance_m' must be a finite non-negative number."
        )

    try:
        distance = float(max_distance_m)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "'max_distance_m' must be a finite non-negative number."
        ) from exc

    if not np.isfinite(distance):
        raise ValueError(
            "'max_distance_m' must be finite."
        )

    if distance < 0.0:
        raise ValueError(
            "'max_distance_m' must be greater than or equal to zero."
        )

    return distance


def _validate_azimuth(
    azimuth_rad: float,
) -> float:
    """Validate and wrap a geographical azimuth to ``[0, 2*pi)``."""
    if isinstance(
        azimuth_rad,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'azimuth_rad' must be a finite scalar angle in radians."
        )

    try:
        azimuth = float(azimuth_rad)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "'azimuth_rad' must be a finite scalar angle in radians."
        ) from exc

    if not np.isfinite(azimuth):
        raise ValueError(
            "'azimuth_rad' must be finite."
        )

    return azimuth % (2.0 * np.pi)


def _validate_domain(
    cell_ids: np.ndarray,
    domain: np.ndarray | None,
) -> np.ndarray:
    """Validate the processing/output domain.

    ``domain`` follows the same semantics as the existing neighbourhood
    reductions:

    - ``cell_ids`` identifies all cells for which input values exist;
    - ``domain`` identifies cells that participate in the operation and
      determines output ordering;
    - cells outside ``domain`` are absent rather than interpreted as padding.
    """
    if domain is None:
        return cell_ids.copy()

    domain_ids = _validate_cell_ids(
        domain,
        name="domain",
    )

    if domain_ids.size == 0:
        return domain_ids

    available = set(
        int(cell)
        for cell in cell_ids
    )

    missing = [
        int(cell)
        for cell in domain_ids
        if int(cell) not in available
    ]

    if missing:
        raise ValueError(
            "'domain' must be a subset of 'cell_ids'."
        )

    return domain_ids


def _validate_values(
    values: np.ndarray | torch.Tensor,
    number_of_cells: int,
) -> None:
    """Validate the signal array.

    The final axis of ``values`` corresponds one-to-one with ``cell_ids``.
    Leading dimensions are preserved.

    Examples
    --------
    A single field::

        values.shape == (N,)

    Several bands::

        values.shape == (bands, N)

    Time and bands::

        values.shape == (time, bands, N)
    """
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


def _restrict_neighbourhoods_to_domain(
    neighbourhoods: list[np.ndarray],
    domain: np.ndarray,
) -> list[np.ndarray]:
    """Remove cells outside the processing domain.

    A geometric neighbourhood can extend beyond a regional processing domain.
    Those outside cells must not contribute, even when values for them happen
    to be available in ``cell_ids``.

    In particular, cells outside ``domain`` are not interpreted as:

    - zero,
    - NaN padding,
    - periodic continuation,
    - wrapped array indices.

    They are simply absent.
    """
    if domain.size == 0:
        return []

    domain_set = {
        int(cell)
        for cell in domain
    }

    restricted: list[np.ndarray] = []

    for neighbourhood in neighbourhoods:
        kept = np.asarray(
            [
                cell
                for cell in neighbourhood
                if int(cell) in domain_set
            ],
            dtype=np.uint64,
        )

        restricted.append(
            kept
        )

    return restricted


def _wrap_relative_bearing(
    bearing_rad: np.ndarray,
    azimuth_rad: float,
    valid_mask: np.ndarray,
    distance_m: np.ndarray,
) -> np.ndarray:
    """Return bearing relative to the requested geographical direction.

    Relative bearing is wrapped to::

        [-pi, +pi)

    so that::

        0
            lies exactly along the requested direction,

        +pi / 2
            lies 90 degrees clockwise from the requested direction,

        -pi / 2
            lies 90 degrees counter-clockwise from it.

    The geographical bearing of a zero-distance pair is undefined.  For a
    centre cell contributing to itself we define ``relative_bearing = 0`` by
    convention.  This makes the self-weight well-defined while leaving the
    directional meaning of non-zero displacements unchanged.
    """
    relative = np.full(
        bearing_rad.shape,
        np.nan,
        dtype=np.float64,
    )

    if not np.any(valid_mask):
        return relative

    valid_bearing = bearing_rad[
        valid_mask
    ]

    wrapped = (
        valid_bearing
        - azimuth_rad
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    relative[
        valid_mask
    ] = wrapped

    self_mask = (
        valid_mask
        & np.isclose(
            distance_m,
            0.0,
            rtol=0.0,
            atol=0.0,
        )
    )

    relative[
        self_mask
    ] = 0.0

    return relative


def _evaluate_kernel(
    kernel: Kernel,
    geometry: RelativeNeighbourhoodGeometry,
    relative_bearing_rad: np.ndarray,
) -> np.ndarray:
    """Evaluate a directional kernel for all valid neighbour pairs.

    The kernel is called only on geometrically valid pairs.  Padding positions
    are therefore never passed to user code.

    The callable receives two one-dimensional NumPy arrays::

        kernel(
            distance_m,
            relative_bearing_rad,
        )

    It may return either:

    - one scalar weight, which is broadcast to every pair, or
    - an array broadcastable to the number of valid pairs.

    Kernel weights are treated as geometry-dependent constants.  For Torch
    signal input, autograd is preserved with respect to ``values`` but this
    function does not provide differentiation through kernel construction or
    WGS84 geometry.
    """
    if not callable(kernel):
        raise TypeError(
            "'kernel' must be callable."
        )

    weights = np.zeros(
        geometry.distance_m.shape,
        dtype=np.float64,
    )

    valid = geometry.valid_mask

    if not np.any(valid):
        return weights

    distance = geometry.distance_m[
        valid
    ]

    relative_bearing = relative_bearing_rad[
        valid
    ]

    raw_weights = kernel(
        distance,
        relative_bearing,
    )

    if isinstance(
        raw_weights,
        torch.Tensor,
    ):
        raw_weights = (
            raw_weights
            .detach()
            .cpu()
            .numpy()
        )

    raw_weights = np.asarray(
        raw_weights
    )

    if raw_weights.ndim == 0:
        valid_weights = np.full(
            distance.shape,
            raw_weights.item(),
            dtype=np.float64,
        )
    else:
        try:
            valid_weights = np.broadcast_to(
                raw_weights,
                distance.shape,
            ).astype(
                np.float64,
                copy=False,
            )
        except ValueError as exc:
            raise ValueError(
                "'kernel' must return either a scalar or values "
                "broadcastable to the number of valid neighbour pairs."
            ) from exc

    if not np.all(
        np.isfinite(valid_weights)
    ):
        raise ValueError(
            "'kernel' returned non-finite weights for valid neighbour pairs."
        )

    weights[
        valid
    ] = valid_weights

    return weights


def directional_filter(
    values: np.ndarray | torch.Tensor,
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    max_distance_m: float,
    azimuth_rad: float,
    kernel: Kernel,
    normalize: bool = False,
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> np.ndarray | torch.Tensor:
    """Apply a geographical directional filter to a HEALPix scalar field.

    Parameters
    ----------
    values
        NumPy array or PyTorch tensor containing the input signal.

        The final dimension corresponds one-to-one with ``cell_ids``.
        Leading dimensions are preserved, so input may represent one field,
        several bands, or arbitrary batches.

        Examples::

            (N,)
            (bands, N)
            (time, bands, N)

    cell_ids
        One-dimensional array of unique NESTED HEALPix cell IDs for which
        input values are available.

    refinement_level
        HEALPix refinement level.

    max_distance_m
        Maximum centre-to-centre geographical distance, in metres, over
        which cells may contribute.

        This is a physical-distance definition.  It is intentionally
        independent of HEALPix ring number, ``kernel_sz``, or Cartesian
        pixel-window dimensions.

    azimuth_rad
        Requested geographical azimuth in radians.

        The convention is::

            0          North
            pi / 2     East
            pi         South
            3*pi / 2   West

        Angles increase clockwise from geographic North.

        Values outside ``[0, 2*pi)`` are accepted and wrapped.

    kernel
        Callable defining the directional spatial weight.

        It is evaluated as::

            kernel(
                distance_m,
                relative_bearing_rad,
            )

        where ``relative_bearing_rad`` is::

            wrap(
                bearing_from_target_to_neighbour
                - azimuth_rad
            )

        in ``[-pi, +pi)``.

        Therefore::

            relative_bearing = 0
                exactly follows the requested direction.

        The callable receives one-dimensional NumPy arrays for all valid
        target-neighbour pairs and must return either one scalar weight or
        an array broadcastable to the same shape.

        Example of a simple forward angular sector::

            def forward_sector(distance_m, relative_bearing_rad):
                return (
                    np.abs(relative_bearing_rad)
                    <= np.deg2rad(20.0)
                ).astype(float)

        A distance-dependent directional kernel can combine both arguments::

            def directional_gaussian(
                distance_m,
                relative_bearing_rad,
            ):
                radial = np.exp(
                    -0.5 * (distance_m / 200.0) ** 2
                )

                angular = np.exp(
                    -0.5
                    * (
                        relative_bearing_rad
                        / np.deg2rad(15.0)
                    ) ** 2
                )

                return radial * angular

    normalize
        If ``False`` (default), return the unnormalised weighted sum::

            sum(weight * value)

        If ``True``, divide by the sum of effective weights::

            sum(weight * value)
            -------------------
                sum(weight)

        Cells outside ``domain`` and NaN-valued samples do not contribute
        to either numerator or denominator.

        If the effective weight sum is zero, the normalized result is NaN.

    domain
        Optional processing and output domain.

        ``domain`` must be a subset of ``cell_ids``.

        If omitted::

            domain == cell_ids

        Output ordering follows the exact ordering supplied by ``domain``.

        Cells outside ``domain`` are absent from the directional operation.
        They are not interpreted as zero, NaN padding, periodic continuation,
        or wrapped array indices.

    ellipsoid
        Geographical ellipsoid used by the shared neighbourhood geometry.

        The current relative-geometry implementation supports
        ``"WGS84"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Directionally filtered values.

        The output has shape::

            values.shape[:-1] + (len(domain),)

        NumPy input returns NumPy output.

        Torch input returns Torch output on the original device.  The
        weighted signal operation remains differentiable with respect to
        ``values``.

    Notes
    -----
    The processing pipeline is::

        physical neighbourhood
               |
               v
        HEALPix neighbour cells
               |
               v
        WGS84 relative geometry
        distance + forward azimuth
               |
               v
        relative bearing
        bearing - requested azimuth
               |
               v
        user directional kernel
               |
               v
        weighted aggregation

    The forward bearing is always defined from the output/target cell to the
    contributing neighbour/source cell::

        target i  ----->  source j

    Reversing this convention would change an asymmetric directional kernel
    by approximately 180 degrees and is therefore not equivalent.

    The geographical bearing of a cell to itself is undefined.  For
    zero-distance self contributions, ``relative_bearing_rad`` is defined
    as zero by convention.

    NaN input samples are treated as unavailable observations rather than
    propagating through the entire neighbourhood.  Their corresponding
    weights are removed.  With ``normalize=True`` the remaining valid weights
    are renormalized.

    This operator does not attempt to reproduce Cartesian ``roll``, shifted
    arrays, image rotation, or a gauge-oriented convolution stencil.  Its
    contract is the underlying geographical operation:

        physical distance
        +
        geographical direction
        +
        HEALPix geometry.

    Examples
    --------
    Apply a forward-looking directional mean over a 500 metre physical
    neighbourhood::

        import numpy as np

        from healpix_analyse import directional_filter

        def forward_kernel(
            distance_m,
            relative_bearing_rad,
        ):
            forward = (
                np.abs(relative_bearing_rad)
                <= np.deg2rad(20.0)
            )

            return forward.astype(float)

        filtered = directional_filter(
            values,
            cell_ids,
            refinement_level=17,
            max_distance_m=500.0,
            azimuth_rad=np.deg2rad(135.0),
            kernel=forward_kernel,
            normalize=True,
        )

    In the example above, ``135 degrees`` has its normal geographical
    meaning: southeast, measured clockwise from North.
    """
    cells = _validate_cell_ids(
        cell_ids,
        name="cell_ids",
    )

    _validate_values(
        values,
        cells.size,
    )

    output_domain = _validate_domain(
        cells,
        domain,
    )

    distance = _validate_max_distance(
        max_distance_m
    )

    azimuth = _validate_azimuth(
        azimuth_rad
    )

    if not isinstance(
        normalize,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'normalize' must be a boolean."
        )

    normalize = bool(
        normalize
    )

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Directional filtering currently supports "
            "ellipsoid='WGS84' only."
        )

    if output_domain.size == 0:
        output_shape = (
            *values.shape[:-1],
            0,
        )

        if isinstance(values, torch.Tensor):
            dtype = (
                values.dtype
                if values.is_floating_point() or values.is_complex()
                else torch.get_default_dtype()
            )

            return torch.empty(
                output_shape,
                dtype=dtype,
                device=values.device,
            )

        return np.empty(
            output_shape,
            dtype=np.result_type(
                values.dtype,
                np.float64,
            ),
        )

    # ------------------------------------------------------------------
    # Spatial support
    #
    # Public semantics are expressed in metres.  The shared neighbourhood
    # helper may use internal candidate-generation strategies, but the
    # resulting neighbourhood follows physical cell-centre distance.
    # ------------------------------------------------------------------
    neighbourhoods = build_neighbourhoods(
        output_domain,
        distance,
        refinement_level,
        neighbourhood="cell_center",
        ellipsoid=ellipsoid,
    )

    # Only cells in the processing domain may contribute.  A value may exist
    # in cell_ids while intentionally lying outside domain; such a cell must
    # still remain absent from the operation.
    neighbourhoods = _restrict_neighbourhoods_to_domain(
        neighbourhoods,
        output_domain,
    )

    # ------------------------------------------------------------------
    # Shared WGS84 geometry from Issue #27.
    #
    # No HEALPix directional-position label is interpreted as geographic
    # East/North here.  Distance and bearing come from the real relative
    # geometry of cell centres.
    # ------------------------------------------------------------------
    geometry = relative_geometry_from_neighbourhoods(
        output_domain,
        neighbourhoods,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    relative_bearing = _wrap_relative_bearing(
        geometry.azimuth_rad,
        azimuth,
        geometry.valid_mask,
        geometry.distance_m,
    )

    weights = _evaluate_kernel(
        kernel,
        geometry,
        relative_bearing,
    )
    return weighted_neighbourhood_reduce(
    values,
    cells,
    geometry.neighbour_ids,
    geometry.valid_mask,
    weights,
    normalize=normalize,
    )



__all__ = [
    "directional_filter",
]
