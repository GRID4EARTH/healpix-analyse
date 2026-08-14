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

from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Any

import numpy as np
import torch

from ._neighbourhood import (
    CompactMetricNeighbourhoodGeometry,
    build_metric_neighbourhood_geometry,
)
from ._weighted_neighbourhood import compact_weighted_neighbourhood_reduce


ArrayLike = np.ndarray | torch.Tensor
Kernel = Callable[[np.ndarray], Any]

_GEOMETRY_CACHE_MAX_BYTES = 192 * 1024 * 1024
_WEIGHT_CACHE_MAX_BYTES = 96 * 1024 * 1024
_geometry_cache: OrderedDict[
    tuple[Any, ...],
    CompactMetricNeighbourhoodGeometry,
] = OrderedDict()
_geometry_cache_bytes = 0
_gaussian_weight_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
_gaussian_weight_cache_bytes = 0
_cache_lock = RLock()


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


def _positions_in_input(
    cell_ids: np.ndarray,
    domain: np.ndarray,
) -> np.ndarray:
    """Map the validated output domain to positions in the input signal."""
    sorter = np.argsort(cell_ids)
    positions = np.searchsorted(cell_ids[sorter], domain)
    return sorter[positions]


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


def _cache_key(
    domain: np.ndarray,
    refinement_level: int,
    radius: float,
    ellipsoid: str,
) -> tuple[Any, ...]:
    """Return an exact, mutation-safe key for reusable filter geometry."""
    contiguous = np.ascontiguousarray(domain, dtype=np.uint64)
    return (
        contiguous.size,
        contiguous.tobytes(),
        int(refinement_level),
        float(radius),
        ellipsoid,
    )


def _geometry_nbytes(geometry: CompactMetricNeighbourhoodGeometry) -> int:
    return sum(
        array.nbytes
        for array in (
            geometry.center_ids,
            geometry.neighbour_indices,
            geometry.row_offsets,
            geometry.distance_m,
        )
    )


def _filter_geometry(
    domain: np.ndarray,
    refinement_level: int,
    radius: float,
    ellipsoid: str,
) -> tuple[tuple[Any, ...], CompactMetricNeighbourhoodGeometry]:
    """Build or retrieve value-independent radial filter geometry."""
    global _geometry_cache_bytes

    key = _cache_key(domain, refinement_level, radius, ellipsoid)
    with _cache_lock:
        cached = _geometry_cache.get(key)
        if cached is not None:
            _geometry_cache.move_to_end(key)
            return key, cached

    geometry = build_metric_neighbourhood_geometry(
        domain,
        radius,
        refinement_level,
        ellipsoid=ellipsoid,
    )

    size = _geometry_nbytes(geometry)
    if size <= _GEOMETRY_CACHE_MAX_BYTES:
        with _cache_lock:
            replaced = _geometry_cache.pop(key, None)
            if replaced is not None:
                _geometry_cache_bytes -= _geometry_nbytes(replaced)
            while (
                _geometry_cache
                and _geometry_cache_bytes + size > _GEOMETRY_CACHE_MAX_BYTES
            ):
                _, evicted = _geometry_cache.popitem(last=False)
                _geometry_cache_bytes -= _geometry_nbytes(evicted)
            _geometry_cache[key] = geometry
            _geometry_cache_bytes += size

    return key, geometry


def _gaussian_weights(
    key: tuple[Any, ...],
    geometry: CompactMetricNeighbourhoodGeometry,
    sigma: float,
) -> np.ndarray:
    """Build or retrieve exact Gaussian weights for cached geometry."""
    global _gaussian_weight_cache_bytes

    weight_key = (*key, float(sigma))
    with _cache_lock:
        cached = _gaussian_weight_cache.get(weight_key)
        if cached is not None:
            _gaussian_weight_cache.move_to_end(weight_key)
            return cached

    distance = geometry.distance_m.copy()
    row_counts = np.diff(geometry.row_offsets)
    self_mask = (
        geometry.neighbour_indices
        == np.repeat(
            np.arange(geometry.center_ids.size, dtype=np.int64),
            row_counts,
        )
    )
    distance[self_mask] = 0.0
    scaled = distance / sigma
    weights = np.exp(-0.5 * scaled * scaled)

    if weights.nbytes <= _WEIGHT_CACHE_MAX_BYTES:
        with _cache_lock:
            replaced = _gaussian_weight_cache.pop(weight_key, None)
            if replaced is not None:
                _gaussian_weight_cache_bytes -= replaced.nbytes
            while (
                _gaussian_weight_cache
                and _gaussian_weight_cache_bytes + weights.nbytes
                > _WEIGHT_CACHE_MAX_BYTES
            ):
                _, evicted = _gaussian_weight_cache.popitem(last=False)
                _gaussian_weight_cache_bytes -= evicted.nbytes
            _gaussian_weight_cache[weight_key] = weights
            _gaussian_weight_cache_bytes += weights.nbytes

    return weights


def _clear_filter_caches() -> None:
    """Clear private geometry/weight caches (primarily for tests)."""
    global _geometry_cache_bytes, _gaussian_weight_cache_bytes
    with _cache_lock:
        _geometry_cache.clear()
        _gaussian_weight_cache.clear()
        _geometry_cache_bytes = 0
        _gaussian_weight_cache_bytes = 0


def _evaluate_kernel(
    kernel: Kernel,
    geometry: CompactMetricNeighbourhoodGeometry,
) -> np.ndarray:
    """Evaluate a radial kernel for every valid centre-neighbour pair.

    The user kernel is called only on valid physical distances. Compact
    geometry contains no padding positions.

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

    if geometry.distance_m.size == 0:
        return np.empty(0, dtype=np.float64)

    # The centre cell is a valid neighbour of itself for radial filtering.
    # WGS84 inverse geodesics can return tiny non-zero round-off values even
    # when centre and neighbour IDs are identical.  Canonicalize these
    # self-distances to exactly zero before user-kernel evaluation so kernels
    # may safely use conditions such as ``distance_m == 0.0``.
    distance_m = geometry.distance_m.copy()
    self_mask = geometry.neighbour_indices == np.repeat(
        np.arange(geometry.center_ids.size, dtype=np.int64),
        np.diff(geometry.row_offsets),
    )
    distance_m[self_mask] = 0.0

    raw_weights = kernel(distance_m)

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
            distance_m.shape,
            raw_weights.item(),
            dtype=np.float64,
        )
    else:
        try:
            valid_weights = np.broadcast_to(
                raw_weights,
                distance_m.shape,
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

    return valid_weights


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

    _, geometry = _filter_geometry(
        output_domain,
        refinement_level,
        radius,
        ellipsoid,
    )

    weights = _evaluate_kernel(
        kernel,
        geometry,
    )

    domain_positions = _positions_in_input(cells, geometry.center_ids)
    return compact_weighted_neighbourhood_reduce(
        values,
        domain_positions[geometry.neighbour_indices],
        geometry.row_offsets,
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

    cells = _validate_cell_ids(
        cell_ids,
        name="cell_ids",
    )
    _validate_values(values, cells.size)
    output_domain = _validate_domain(cells, domain)

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Radial filtering currently supports ellipsoid='WGS84' only."
        )

    key, geometry = _filter_geometry(
        output_domain,
        refinement_level,
        radius,
        ellipsoid,
    )
    weights = _gaussian_weights(key, geometry, sigma)

    domain_positions = _positions_in_input(cells, geometry.center_ids)
    return compact_weighted_neighbourhood_reduce(
        values,
        domain_positions[geometry.neighbour_indices],
        geometry.row_offsets,
        weights,
        normalize=True,
    )


__all__ = [
    "gaussian_filter",
    "radial_filter",
]
