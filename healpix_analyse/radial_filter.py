"""Metric radial filtering on NESTED HEALPix fields.

This module provides deterministic isotropic filtering whose spatial
semantics are defined entirely by physical WGS84 centre-to-centre distance.

For an output/target cell ``i`` and a contributing neighbour/source cell
``j``::

    distance_ij = WGS84 geodesic distance from i to j
    weight_ij   = kernel(distance_ij)

The public API deliberately avoids Cartesian concepts such as ``3x3`` or
``5x5`` windows and pixel-based smoothing scales.  Distances are expressed in
metres so that the same kernel retains the same physical meaning across
HEALPix refinement levels.

Neighbour selection and WGS84 geometry are delegated to the shared
neighbourhood infrastructure.  Weighted value gathering, NaN handling,
normalization, Torch device preservation, and autograd are delegated to the
shared weighted-neighbourhood aggregation helper.

The intended separation is::

    physical neighbourhood
            |
            v
    shared WGS84 geometry
        distance_m
            |
            v
      kernel(distance_m)
            |
            v
    supplied radial weights
            |
            v
    weighted_neighbourhood_reduce()
            |
            v
          output

This keeps radial filtering separate from directional filtering.  A radial
kernel depends only on distance; geographical azimuth or relative bearing is
not part of this API.
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
from ._weighted_neighbourhood import weighted_neighbourhood_reduce


ArrayLike = np.ndarray | torch.Tensor
Kernel = Callable[[np.ndarray], Any]


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


def _validate_radius(
    radius_m: float,
) -> float:
    """Validate the physical support radius in metres."""
    if isinstance(
        radius_m,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'radius_m' must be a finite non-negative number."
        )

    try:
        radius = float(radius_m)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "'radius_m' must be a finite non-negative number."
        ) from exc

    if not np.isfinite(radius):
        raise ValueError(
            "'radius_m' must be finite."
        )

    if radius < 0.0:
        raise ValueError(
            "'radius_m' must be greater than or equal to zero."
        )

    return radius


def _validate_positive_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a strictly positive finite scalar parameter."""
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise TypeError(
            f"'{name}' must be a finite positive number."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"'{name}' must be a finite positive number."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(
            f"'{name}' must be finite."
        )

    if result <= 0.0:
        raise ValueError(
            f"'{name}' must be greater than zero."
        )

    return result


def _validate_domain(
    cell_ids: np.ndarray,
    domain: np.ndarray | None,
) -> np.ndarray:
    """Validate the processing/output domain.

    ``cell_ids`` identifies every cell for which an input value exists.
    ``domain`` identifies the cells that participate in the operation and
    determines output ordering.

    Cells outside ``domain`` are absent from the effective neighbourhood even
    when their values are available in ``cell_ids``.
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

    if any(
        int(cell) not in available
        for cell in domain_ids
    ):
        raise ValueError(
            "'domain' must be a subset of 'cell_ids'."
        )

    return domain_ids


def _validate_values(
    values: ArrayLike,
    number_of_cells: int,
) -> None:
    """Validate a signal whose final axis corresponds to ``cell_ids``."""
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
    """Remove cells outside the processing domain before aggregation.

    Domain filtering belongs to the spatial-neighbourhood stage rather than
    to ``weighted_neighbourhood_reduce``.  A cell outside ``domain`` is not a
    zero, NaN, or padded sample; it is simply absent from the operation.
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

        restricted.append(kept)

    return restricted


def _evaluate_kernel(
    kernel: Kernel,
    geometry: RelativeNeighbourhoodGeometry,
) -> np.ndarray:
    """Evaluate a radial kernel for every valid centre-neighbour pair.

    The user kernel is called only on valid physical distances.  Padded
    ``NaN`` distances are never passed to user code.

    The callable receives a one-dimensional NumPy array containing the
    physical distances of all valid pairs and may return either:

    - one scalar weight, broadcast to every valid pair; or
    - an array broadcastable to the number of valid pairs.

    Finite negative weights are allowed.  This keeps ``radial_filter`` useful
    for general radial kernels rather than restricting it to smoothing-only
    kernels.  Non-finite weights at valid positions are rejected because they
    indicate an invalid kernel definition rather than missing signal data.
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

    # The centre cell is a valid neighbour of itself for radial filtering.
    # WGS84 inverse geodesics can return tiny non-zero round-off values even
    # when centre and neighbour IDs are identical.  Canonicalize these
    # self-distances to exactly zero before user-kernel evaluation so kernels
    # may safely use conditions such as ``distance_m == 0.0``.
    distance_m = geometry.distance_m.copy()

    self_mask = (
        geometry.valid_mask
        & (
            geometry.neighbour_ids
            == geometry.center_ids[:, None]
        )
    )
    distance_m[self_mask] = 0.0

    distance = distance_m[
        valid
    ]

    raw_weights = kernel(distance)

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

    raw_weights = np.asarray(raw_weights)

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


def radial_filter(
    values: ArrayLike,
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    radius_m: float,
    kernel: Kernel,
    normalize: bool = True,
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply an isotropic physical-distance kernel to a HEALPix field.

    Parameters
    ----------
    values
        NumPy array or PyTorch tensor containing the input signal.  The final
        dimension corresponds one-to-one with ``cell_ids``.  Arbitrary
        leading dimensions are preserved.

    cell_ids
        One-dimensional array of unique NESTED HEALPix cell IDs for which
        input values are available.

    refinement_level
        HEALPix refinement level.

    radius_m
        Maximum WGS84 centre-to-centre distance, in metres, over which cells
        may contribute.  ``0`` is valid and represents centre-only physical
        support.

    kernel
        Callable defining the radial weight as a function of physical
        distance::

            weight = kernel(distance_m)

        The callable receives a one-dimensional NumPy array of valid
        centre-neighbour distances.  It may return a scalar or an array
        broadcastable to that shape.

        Finite negative weights are allowed.  Non-finite weights at valid
        neighbour positions raise ``ValueError``.

    normalize
        If ``True`` (default), return the normalized weighted result::

                  sum(weight * value)
            -----------------------------
                 sum(effective weight)

        using only effective valid samples.

        If ``False``, return the raw weighted sum::

            sum(weight * value)

        NaN-valued signal samples are excluded by the shared aggregation
        helper.  With ``normalize=True``, their weights are also removed from
        the denominator.  If the effective weight sum is zero, the normalized
        result is NaN.

    domain
        Optional processing and output domain.  It must be a subset of
        ``cell_ids``.  Output ordering follows ``domain`` exactly.

        Cells outside ``domain`` are absent from the effective neighbourhood,
        even when input values for them are present in ``cell_ids``.

    ellipsoid
        Geographical ellipsoid used by the shared neighbourhood geometry.
        The current relative-geometry implementation supports ``"WGS84"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Filtered values with shape::

            values.shape[:-1] + (len(domain),)

        when ``domain`` is supplied, or the same final-axis length as
        ``cell_ids`` otherwise.

        NumPy input returns NumPy output.  Torch input returns Torch output on
        the original device, with autograd preserved with respect to
        ``values``.

    Notes
    -----
    The spatial support uses physical WGS84 cell-centre distance.  HEALPix
    ring number and Cartesian pixel-window dimensions are deliberately not
    part of the public API.

    The implementation pipeline is::

        physical-radius neighbourhood
                    |
                    v
           processing-domain filter
                    |
                    v
       shared relative WGS84 geometry
                    |
                    v
                distance_m
                    |
                    v
           user radial kernel
                    |
                    v
       weighted_neighbourhood_reduce
                    |
                    v
                  output

    Examples
    --------
    Apply a normalized inverse-distance-like radial filter::

        import numpy as np
        from healpix_analyse import radial_filter

        def kernel(distance_m):
            return 1.0 / (1.0 + distance_m / 250.0)

        filtered = radial_filter(
            values,
            cell_ids,
            refinement_level=17,
            radius_m=1000.0,
            kernel=kernel,
            normalize=True,
        )
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

    radius = _validate_radius(radius_m)

    if not isinstance(
        normalize,
        (bool, np.bool_),
    ):
        raise TypeError(
            "'normalize' must be a boolean."
        )

    normalize = bool(normalize)

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Radial filtering currently supports ellipsoid='WGS84' only."
        )

    # A zero-radius radial neighbourhood has an exact and useful semantic:
    # each target cell contributes only to itself.  Handle this explicitly
    # instead of relying on candidate-generation / geodesic round-off at the
    # zero-distance boundary.
    if radius == 0.0:
        neighbourhoods = [
            np.asarray([cell], dtype=np.uint64)
            for cell in output_domain
        ]
    else:
        # Public radial semantics are physical centre-to-centre distance.  The
        # shared cell_center neighbourhood applies the exact WGS84 distance
        # cutoff after candidate generation.
        neighbourhoods = build_neighbourhoods(
            output_domain,
            radius,
            refinement_level,
            neighbourhood="cell_center",
            ellipsoid=ellipsoid,
        )

    # Domain is both the participating spatial set and output domain.  Cells
    # outside it are removed before geometry/aggregation rather than being
    # represented by zero-valued or NaN-valued samples.
    neighbourhoods = _restrict_neighbourhoods_to_domain(
        neighbourhoods,
        output_domain,
    )

    geometry = relative_geometry_from_neighbourhoods(
        output_domain,
        neighbourhoods,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    weights = _evaluate_kernel(
        kernel,
        geometry,
    )

    return weighted_neighbourhood_reduce(
        values,
        cells,
        geometry.neighbour_ids,
        geometry.valid_mask,
        weights,
        normalize=normalize,
    )


def gaussian_filter(
    values: ArrayLike,
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    sigma_m: float,
    truncate: float = 4.0,
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> ArrayLike:
    """Apply normalized Gaussian smoothing in physical metres.

    The Gaussian kernel is::

        weight(d) = exp(-0.5 * (d / sigma_m)**2)

    and is truncated at::

        radius_m = truncate * sigma_m

    Parameters
    ----------
    values, cell_ids, refinement_level, domain, ellipsoid
        See :func:`radial_filter`.

    sigma_m
        Gaussian standard deviation in metres.  Must be finite and strictly
        positive.

    truncate
        Gaussian support radius in units of ``sigma_m``.  Must be finite and
        strictly positive.  The default is ``4.0``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normalized Gaussian-smoothed values.

    Notes
    -----
    Gaussian smoothing is normalized by construction.  Constant finite fields
    therefore remain constant, including near partial-domain boundaries after
    unavailable samples are removed and the remaining effective weights are
    renormalized.

    The physical smoothing scale is independent of HEALPix refinement level:
    ``sigma_m=240`` always means 240 metres rather than a number of pixels.
    """
    sigma = _validate_positive_finite(
        sigma_m,
        name="sigma_m",
    )

    truncation = _validate_positive_finite(
        truncate,
        name="truncate",
    )

    radius = sigma * truncation

    # Both validated operands are finite, but their product can still
    # overflow for extreme user inputs.  Reject that explicitly rather than
    # passing an infinite physical radius into neighbourhood construction.
    if not np.isfinite(radius):
        raise ValueError(
            "'truncate * sigma_m' must be finite."
        )

    def _gaussian_kernel(
        distance_m: np.ndarray,
    ) -> np.ndarray:
        scaled = distance_m / sigma
        return np.exp(
            -0.5 * scaled * scaled
        )

    return radial_filter(
        values,
        cell_ids,
        refinement_level,
        radius_m=radius,
        kernel=_gaussian_kernel,
        normalize=True,
        domain=domain,
        ellipsoid=ellipsoid,
    )


__all__ = [
    "gaussian_filter",
    "radial_filter",
]
