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
from typing import Any, NamedTuple
import warnings

import numpy as np
import torch

from ._neighbourhood import (
    _WGS84_AUTHALIC_RADIUS_M,
    CompactMetricNeighbourhoodGeometry,
    build_metric_neighbourhood_geometry,
    build_outside_domain_distances,
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
_warned_oversize_cache_entries: set[tuple[Any, ...]] = set()


def _mib(number_of_bytes: int) -> float:
    return number_of_bytes / (1024.0**2)


def _validate_cache_limit_mib(value: float, *, name: str) -> int:
    """Validate a public cache limit and convert it to bytes."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"'{name}' must be a finite non-negative number.")
    try:
        limit_mib = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"'{name}' must be a finite non-negative number."
        ) from error
    if not np.isfinite(limit_mib) or limit_mib < 0.0:
        raise ValueError(f"'{name}' must be finite and non-negative.")
    return int(limit_mib * 1024.0**2)


def radial_filter_cache_info() -> dict[str, float | int]:
    """Return current radial/Gaussian cache limits and usage.

    Returns
    -------
    dict
        Geometry and weight limits and current usage in MiB, plus the number
        of retained entries in each cache.
    """
    with _cache_lock:
        return {
            "geometry_max_mib": _mib(_GEOMETRY_CACHE_MAX_BYTES),
            "weight_max_mib": _mib(_WEIGHT_CACHE_MAX_BYTES),
            "geometry_used_mib": _mib(_geometry_cache_bytes),
            "weight_used_mib": _mib(_gaussian_weight_cache_bytes),
            "geometry_entries": len(_geometry_cache),
            "weight_entries": len(_gaussian_weight_cache),
        }


def configure_radial_filter_cache(
    *,
    geometry_max_mib: float | None = None,
    weight_max_mib: float | None = None,
) -> dict[str, float | int]:
    """Configure bounded radial/Gaussian caches and return their status.

    ``None`` leaves a limit unchanged. A zero limit disables that cache.
    Reducing a limit immediately evicts least-recently-used entries until the
    retained data fits. Large production workloads should ensure sufficient
    process memory for both cached arrays and temporary construction arrays.

    Parameters
    ----------
    geometry_max_mib
        Maximum retained metric-geometry size in MiB.
    weight_max_mib
        Maximum retained Gaussian-weight size in MiB.

    Returns
    -------
    dict
        Updated result from :func:`radial_filter_cache_info`.
    """
    global _GEOMETRY_CACHE_MAX_BYTES, _WEIGHT_CACHE_MAX_BYTES
    global _geometry_cache_bytes, _gaussian_weight_cache_bytes

    geometry_limit = (
        None
        if geometry_max_mib is None
        else _validate_cache_limit_mib(
            geometry_max_mib,
            name="geometry_max_mib",
        )
    )
    weight_limit = (
        None
        if weight_max_mib is None
        else _validate_cache_limit_mib(
            weight_max_mib,
            name="weight_max_mib",
        )
    )

    with _cache_lock:
        if geometry_limit is not None:
            _GEOMETRY_CACHE_MAX_BYTES = geometry_limit
        if weight_limit is not None:
            _WEIGHT_CACHE_MAX_BYTES = weight_limit
        while (
            _geometry_cache
            and _geometry_cache_bytes > _GEOMETRY_CACHE_MAX_BYTES
        ):
            _, evicted = _geometry_cache.popitem(last=False)
            _geometry_cache_bytes -= _geometry_nbytes(evicted)
        while (
            _gaussian_weight_cache
            and _gaussian_weight_cache_bytes > _WEIGHT_CACHE_MAX_BYTES
        ):
            _, evicted = _gaussian_weight_cache.popitem(last=False)
            _gaussian_weight_cache_bytes -= evicted.nbytes
        _warned_oversize_cache_entries.clear()
    return radial_filter_cache_info()


def _warn_oversize_cache_entry(
    *,
    kind: str,
    required_bytes: int,
    limit_bytes: int,
    domain_size: int,
    refinement_level: int,
    radius_m: float,
    option_name: str,
) -> None:
    """Warn once when one spatial plan cannot fit its configured cache."""
    token = (
        kind,
        domain_size,
        refinement_level,
        radius_m,
        required_bytes,
        limit_bytes,
    )
    with _cache_lock:
        if token in _warned_oversize_cache_entries:
            return
        _warned_oversize_cache_entries.add(token)

    suggested_mib = int(np.ceil(_mib(required_bytes) * 1.05))
    warnings.warn(
        f"The radial-filter {kind} cache entry requires "
        f"{_mib(required_bytes):.1f} MiB, "
        f"exceeding its configured {_mib(limit_bytes):.1f} MiB cache limit. "
        f"It will not be cached, so repeated calls rebuild or recompute it. "
        f"Either call configure_radial_filter_cache({option_name}="
        f"{suggested_mib}) after checking available memory, or process tiles "
        f"with a halo of at least radius_m={radius_m:g} m and discard halo "
        f"outputs when stitching.",
        RuntimeWarning,
        stacklevel=4,
    )


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
    else:
        _warn_oversize_cache_entry(
            kind="geometry",
            required_bytes=size,
            limit_bytes=_GEOMETRY_CACHE_MAX_BYTES,
            domain_size=domain.size,
            refinement_level=refinement_level,
            radius_m=radius,
            option_name="geometry_max_mib",
        )

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
    else:
        _warn_oversize_cache_entry(
            kind="Gaussian-weight",
            required_bytes=weights.nbytes,
            limit_bytes=_WEIGHT_CACHE_MAX_BYTES,
            domain_size=geometry.center_ids.size,
            refinement_level=int(key[2]),
            radius_m=float(key[3]),
            option_name="weight_max_mib",
        )

    return weights


def _clear_filter_caches() -> None:
    """Clear private geometry/weight caches (primarily for tests)."""
    global _geometry_cache_bytes, _gaussian_weight_cache_bytes
    with _cache_lock:
        _geometry_cache.clear()
        _gaussian_weight_cache.clear()
        _warned_oversize_cache_entries.clear()
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
    # The centre cell is a valid neighbour of itself for radial filtering.
    # WGS84 inverse geodesics can return tiny non-zero round-off values even
    # when centre and neighbour IDs are identical.  Canonicalize these
    # self-distances to exactly zero before user-kernel evaluation so kernels
    # may safely use conditions such as ``distance_m == 0.0``.
    distance_m = geometry.distance_m.copy()
    distance_m[_self_mask(geometry)] = 0.0
    return _evaluate_kernel_distances(kernel, distance_m)


def _evaluate_kernel_distances(
    kernel: Kernel,
    distance_m: np.ndarray,
) -> np.ndarray:
    """Evaluate ``kernel`` on one-dimensional distances and validate it."""
    if not callable(kernel):
        raise TypeError(
            "'kernel' must be callable."
        )

    if distance_m.size == 0:
        return np.empty(0, dtype=np.float64)

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


_NORMALIZATIONS = ("present", "full_disc")
_WIDTH_RULES = ("sampled", "matched")

# Sub-cell Gaussian widths (``width_rule="matched"``).  The reference
# operator ``G0`` is the sampled Gaussian of width ``_REFERENCE_SIGMA_CELLS``
# cells; below its effective width a cell blends ``G0`` with the identity.
_REFERENCE_SIGMA_CELLS = 0.6
_MIN_MATCHED_TRUNCATE = 3.0
_MIN_MATCHED_SIGMA_CELLS = 0.5
_SHAPE_SEARCH_LOWER_CELLS = 0.02
_SHAPE_SEARCH_ITERATIONS = 60
_SHAPE_SEARCH_TOLERANCE = 1e-12


class FilterOperator(NamedTuple):
    """Weights of a linear filter in compressed-row (CSR) form.

    The operator maps a signal defined on ``cell_ids`` (final axis of length
    ``N``) to a signal defined on the output domain (``M`` rows)::

        out[..., i] = sum_k weights[k] * values[..., column_indices[k]]
                      for k in row_offsets[i]:row_offsets[i + 1]

    The first three fields are plain NumPy arrays and ``n_columns`` is the
    integer ``N``, the length of the final axis of the signal the operator
    was built for (``len(cell_ids)``).  The caller can wrap the arrays in
    the sparse-matrix type of their choice, for example
    ``scipy.sparse.csr_matrix((weights, column_indices, row_offsets),
    shape=(len(row_offsets) - 1, n_columns))``, or pass the operator to
    :func:`apply_filter_operator`.
    """

    row_offsets: np.ndarray
    column_indices: np.ndarray
    weights: np.ndarray
    n_columns: int


def _validate_normalization(normalization: str) -> str:
    if normalization not in _NORMALIZATIONS:
        raise ValueError(
            f"'normalization' must be one of {_NORMALIZATIONS}; "
            f"got {normalization!r}."
        )
    return normalization


def _validate_width_rule(width_rule: str) -> str:
    if width_rule not in _WIDTH_RULES:
        raise ValueError(
            f"'width_rule' must be one of {_WIDTH_RULES}; "
            f"got {width_rule!r}."
        )
    return width_rule


def _cell_size_m(refinement_level: int) -> float:
    """Return ``sqrt(cell area)`` in metres on the authalic sphere."""
    area = 4.0 * np.pi * _WGS84_AUTHALIC_RADIUS_M**2 / (
        12.0 * 4.0 ** int(refinement_level)
    )
    return float(np.sqrt(area))


def _self_mask(geometry: CompactMetricNeighbourhoodGeometry) -> np.ndarray:
    return geometry.neighbour_indices == np.repeat(
        np.arange(geometry.center_ids.size, dtype=np.int64),
        np.diff(geometry.row_offsets),
    )


def _row_sums(
    rows: np.ndarray,
    weights: np.ndarray,
    number_of_rows: int,
) -> np.ndarray:
    return np.bincount(rows, weights=weights, minlength=number_of_rows)


def _normalize_rows(
    weights_in: np.ndarray,
    rows_in: np.ndarray,
    weights_out: np.ndarray | None,
    rows_out: np.ndarray | None,
    number_of_rows: int,
    normalization: str,
) -> np.ndarray:
    """Divide in-domain weights by the row sum chosen by ``normalization``."""
    total = _row_sums(rows_in, weights_in, number_of_rows)
    if normalization == "full_disc":
        total = total + _row_sums(rows_out, weights_out, number_of_rows)
        if np.any(total == 0.0):
            raise ValueError(
                "The kernel weights sum to zero over the whole disc of at "
                "least one cell; the full-disc normalisation is undefined."
            )
    else:
        # As in radial_filter(normalize=True): a row whose weights sum to
        # zero is left as it is and gives NaN once applied with
        # renormalize_missing=True, instead of raising an error here.
        total = np.where(total == 0.0, 1.0, total)
    return weights_in / total[rows_in]


def _solve_gaussian_shape(
    rows: np.ndarray,
    distance: np.ndarray,
    cut: float,
    target_m2: float,
    lower: float,
    upper: float,
    number_of_rows: int,
) -> np.ndarray:
    """Per row, find ``s`` such that the sampled Gaussian has a given m2.

    The second moment is ``sum(w d^2) / sum(w)`` with
    ``w = exp(-d^2 / (2 s^2))`` on ``d <= cut``.  It increases with ``s``,
    so the root is found by bisection on ``log s`` for all rows at once.
    """
    inside = distance <= cut
    rows = rows[inside]
    d2 = distance[inside] ** 2
    low = np.full(number_of_rows, np.log(lower))
    high = np.full(number_of_rows, np.log(upper))
    for _ in range(_SHAPE_SEARCH_ITERATIONS):
        mid = 0.5 * (low + high)
        w = np.exp(-d2 / (2.0 * np.exp(2.0 * mid)[rows]))
        z = _row_sums(rows, w, number_of_rows)
        m2 = _row_sums(rows, w * d2, number_of_rows) / z
        too_wide = m2 > target_m2
        high = np.where(too_wide, mid, high)
        low = np.where(too_wide, low, mid)
        if np.max(high - low) < _SHAPE_SEARCH_TOLERANCE:
            break
    return np.exp(0.5 * (low + high))


def _second_moment(
    rows: np.ndarray,
    weights: np.ndarray,
    distance: np.ndarray,
    number_of_rows: int,
) -> np.ndarray:
    return _row_sums(rows, weights * distance**2, number_of_rows) / _row_sums(
        rows, weights, number_of_rows
    )


def _gaussian_operator_weights(
    geometry: CompactMetricNeighbourhoodGeometry,
    outside_offsets: np.ndarray | None,
    outside_distance: np.ndarray | None,
    *,
    sigma: float,
    truncate: float,
    refinement_level: int,
    normalization: str,
    width_rule: str,
) -> np.ndarray:
    """Return normalised Gaussian weights for the in-domain pairs."""
    n_rows = geometry.center_ids.size
    rows_in = np.repeat(
        np.arange(n_rows, dtype=np.int64), np.diff(geometry.row_offsets)
    )
    distance_in = geometry.distance_m.copy()
    is_self = _self_mask(geometry)
    distance_in[is_self] = 0.0

    if outside_distance is None:
        rows_out = np.empty(0, dtype=np.int64)
        distance_out = np.empty(0, dtype=np.float64)
    else:
        rows_out = np.repeat(
            np.arange(n_rows, dtype=np.int64), np.diff(outside_offsets)
        )
        distance_out = outside_distance

    if width_rule == "sampled":
        weights_in = np.exp(-0.5 * (distance_in / sigma) ** 2)
        weights_out = np.exp(-0.5 * (distance_out / sigma) ** 2)
        return _normalize_rows(
            weights_in, rows_in, weights_out, rows_out, n_rows, normalization
        )

    cell = _cell_size_m(refinement_level)
    rows_all = np.concatenate((rows_in, rows_out))
    distance_all = np.concatenate((distance_in, distance_out))

    # Reference operator G0 and its per-cell effective width.
    s0 = _REFERENCE_SIGMA_CELLS * cell
    cut0 = truncate * s0
    w0_all = np.where(
        distance_all <= cut0, np.exp(-0.5 * (distance_all / s0) ** 2), 0.0
    )
    sigma0 = np.sqrt(
        0.5 * _second_moment(rows_all, w0_all, distance_all, n_rows)
    )
    blend = sigma <= sigma0  # per cell: identity/G0 blend, else matched

    # Matched shape for the remaining cells.
    cut = truncate * max(sigma, _MIN_MATCHED_SIGMA_CELLS * cell)
    shape = np.full(n_rows, s0)
    solve = ~blend
    if np.any(solve):
        keep = solve[rows_all]
        index = np.cumsum(solve) - 1
        shape[solve] = _solve_gaussian_shape(
            index[rows_all[keep]],
            distance_all[keep],
            cut,
            2.0 * sigma**2,
            _SHAPE_SEARCH_LOWER_CELLS * cell,
            2.0 * max(sigma, _MIN_MATCHED_SIGMA_CELLS * cell),
            int(solve.sum()),
        )
    wm_all = np.where(
        distance_all <= cut,
        np.exp(-0.5 * (distance_all / shape[rows_all]) ** 2),
        0.0,
    )
    kernel_all = np.where(blend[rows_all], w0_all, wm_all)
    n_in = rows_in.size
    kernel_in, kernel_out = kernel_all[:n_in], kernel_all[n_in:]

    normalised = _normalize_rows(
        kernel_in, rows_in, kernel_out, rows_out, n_rows, normalization
    )
    mix = np.ones(n_rows)
    mix[blend] = (sigma / sigma0[blend]) ** 2
    weights = normalised * mix[rows_in]
    weights = weights + np.where(
        is_self & blend[rows_in], 1.0 - mix[rows_in], 0.0
    )
    return weights


def _validate_gaussian_parameters(
    sigma_m: float,
    truncate: float,
    *,
    width_rule: str,
    refinement_level: int,
) -> tuple[float, float, float]:
    """Validate sigma/truncate; return ``(sigma, truncate, disc radius)``."""
    sigma = _validate_positive_finite(sigma_m, name="sigma_m")
    truncation = _validate_positive_finite(truncate, name="truncate")
    _validate_width_rule(width_rule)
    scale = sigma
    if width_rule == "matched":
        if truncation < _MIN_MATCHED_TRUNCATE:
            raise ValueError(
                "width_rule='matched' needs truncate >= "
                f"{_MIN_MATCHED_TRUNCATE:g}: a shorter support cannot "
                "reach the requested width."
            )
        # Cells blended with the identity use the reference width.
        scale = max(sigma, _REFERENCE_SIGMA_CELLS * _cell_size_m(refinement_level))
    radius = scale * truncation

    # Both validated operands are finite, but their product can still
    # overflow for extreme user inputs.  Reject that explicitly rather than
    # passing an infinite physical radius into neighbourhood construction.
    if not np.isfinite(radius):
        raise ValueError("'truncate * sigma_m' must be finite.")
    return sigma, truncation, radius


def _outside_distances(
    output_domain: np.ndarray,
    refinement_level: int,
    radius: float,
    ellipsoid: str,
) -> tuple[np.ndarray, np.ndarray]:
    return build_outside_domain_distances(
        output_domain, radius, refinement_level, ellipsoid=ellipsoid
    )


def gaussian_filter_operator(
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    sigma_m: float,
    truncate: float = 4.0,
    normalization: str = "present",
    width_rule: str = "sampled",
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> FilterOperator:
    """Return the weights of :func:`gaussian_filter` as a reusable operator.

    The options are those of :func:`gaussian_filter`, and applying the result
    with :func:`apply_filter_operator` gives the same values as
    ``gaussian_filter(values, cell_ids, refinement_level, sigma_m=...,
    truncate=..., normalization=..., width_rule=..., domain=...)`` for
    values without NaN (see :func:`apply_filter_operator` for NaN).  Build
    the operator once, then apply it to as many
    signals (time steps, chunks) as needed, or hand the arrays to a
    sparse-matrix type.  Operators are always normalised.

    Parameters
    ----------
    cell_ids, refinement_level, sigma_m, truncate, normalization, \
    width_rule, domain, ellipsoid
        See :func:`gaussian_filter`.

    Returns
    -------
    FilterOperator
        ``(row_offsets, column_indices, weights, n_columns)``.  Row ``i`` is
        the output cell ``domain[i]`` (``cell_ids[i]`` without ``domain``);
        the column indices point into ``cell_ids``, i.e. into the final axis
        of the signal to filter, whose length is ``n_columns``.  The weights
        already include the normalisation.

    Notes
    -----
    With ``normalization="present"`` every row sums to one.  With
    ``"full_disc"`` the row of a cell whose kernel disc leaves the domain
    sums to the fraction of the disc weight that lies inside the domain.

    With ``width_rule="matched"`` the width of the response equals
    ``sigma_m`` for the rows whose disc lies inside the domain.  Rows near
    the edge are renormalised (``"present"``) or truncated (``"full_disc"``),
    so their effective width differs from ``sigma_m``.
    """
    sigma, truncation, radius = _validate_gaussian_parameters(
        sigma_m, truncate, width_rule=width_rule, refinement_level=refinement_level
    )
    _validate_normalization(normalization)
    cells = _validate_cell_ids(cell_ids, name="cell_ids")
    output_domain = _validate_domain(cells, domain)
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Radial filtering currently supports ellipsoid='WGS84' only."
        )
    return _gaussian_operator(
        cells,
        output_domain,
        refinement_level,
        sigma,
        truncation,
        radius,
        normalization,
        width_rule,
        ellipsoid,
    )


def _gaussian_operator(
    cells: np.ndarray,
    output_domain: np.ndarray,
    refinement_level: int,
    sigma: float,
    truncation: float,
    radius: float,
    normalization: str,
    width_rule: str,
    ellipsoid: str,
) -> FilterOperator:
    _, geometry = _filter_geometry(
        output_domain, refinement_level, radius, ellipsoid
    )
    if normalization == "full_disc" or width_rule == "matched":
        outside_offsets, outside_distance = _outside_distances(
            output_domain, refinement_level, radius, ellipsoid
        )
    else:
        outside_offsets = outside_distance = None
    weights = _gaussian_operator_weights(
        geometry,
        outside_offsets,
        outside_distance,
        sigma=sigma,
        truncate=truncation,
        refinement_level=refinement_level,
        normalization=normalization,
        width_rule=width_rule,
    )
    domain_positions = _positions_in_input(cells, geometry.center_ids)
    return FilterOperator(
        np.asarray(geometry.row_offsets, dtype=np.int64),
        domain_positions[geometry.neighbour_indices].astype(np.int64),
        weights,
        int(cells.size),
    )


def radial_filter_operator(
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    radius_m: float,
    kernel: Kernel,
    normalization: str = "present",
    domain: np.ndarray | None = None,
    ellipsoid: str = "WGS84",
) -> FilterOperator:
    """Return the weights of :func:`radial_filter` as a reusable operator.

    The options are those of :func:`radial_filter` (without ``values``) and
    ``normalization`` is described there, with one difference: there is no
    ``normalize=False``.  An operator is always normalised, so that its rows
    sum to one (``"present"``) or to the in-domain fraction of the disc
    (``"full_disc"``).  Apply the raw weighted sum with
    :func:`radial_filter` and ``normalize=False``.  See
    :class:`FilterOperator` and :func:`apply_filter_operator`.

    With ``"present"`` a row whose kernel weights sum to zero (possible only
    for a kernel with negative weights) is left as it is; it gives NaN when
    applied with ``renormalize_missing=True``, as :func:`radial_filter` does,
    and no error is raised.

    For ``normalization="full_disc"`` the kernel is evaluated on the
    distances to all cells of the disc, inside and outside the domain, in one
    call.  A ``ValueError`` is raised if the kernel weights of some disc sum
    to zero.

    The distances of the in-domain and outside cells are concatenated into
    one array for that call, so the kernel must act element by element (it
    must not normalise by a statistic of its input).  As for
    :func:`gaussian_filter_operator`, equality with the filter holds for
    values without NaN: see :func:`apply_filter_operator`.
    """
    _validate_normalization(normalization)
    cells = _validate_cell_ids(cell_ids, name="cell_ids")
    output_domain = _validate_domain(cells, domain)
    radius = _validate_radius(radius_m)
    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Radial filtering currently supports ellipsoid='WGS84' only."
        )
    _, geometry = _filter_geometry(
        output_domain, refinement_level, radius, ellipsoid
    )
    n_rows = geometry.center_ids.size
    rows_in = np.repeat(
        np.arange(n_rows, dtype=np.int64), np.diff(geometry.row_offsets)
    )
    distance_in = geometry.distance_m.copy()
    distance_in[_self_mask(geometry)] = 0.0

    if normalization == "full_disc":
        outside_offsets, outside_distance = _outside_distances(
            output_domain, refinement_level, radius, ellipsoid
        )
        rows_out = np.repeat(
            np.arange(n_rows, dtype=np.int64), np.diff(outside_offsets)
        )
        all_weights = _evaluate_kernel_distances(
            kernel,
            np.concatenate((distance_in, outside_distance)),
        )
        weights_in = all_weights[: rows_in.size]
        weights_out = all_weights[rows_in.size :]
    else:
        weights_in = _evaluate_kernel(kernel, geometry)
        rows_out = weights_out = None

    weights = _normalize_rows(
        weights_in, rows_in, weights_out, rows_out, n_rows, normalization
    )
    domain_positions = _positions_in_input(cells, geometry.center_ids)
    return FilterOperator(
        np.asarray(geometry.row_offsets, dtype=np.int64),
        domain_positions[geometry.neighbour_indices].astype(np.int64),
        weights,
        int(cells.size),
    )


def apply_filter_operator(
    values: ArrayLike,
    row_offsets: np.ndarray,
    column_indices: np.ndarray,
    weights: np.ndarray,
    n_columns: int | None = None,
    *,
    renormalize_missing: bool = False,
) -> ArrayLike:
    """Apply a :class:`FilterOperator` to a signal.

    Parameters
    ----------
    values
        NumPy array or PyTorch tensor.  The final axis is indexed by
        ``column_indices``; leading dimensions (for example time) are kept.
    row_offsets, column_indices, weights
        The first three fields of a :class:`FilterOperator`.
    n_columns
        Optional.  The fourth field of a :class:`FilterOperator`, so that
        ``apply_filter_operator(values, *operator)`` works.  When given, a
        ``ValueError`` is raised if the final axis of ``values`` does not
        have that length; without it the length is not checked, and a longer
        signal would be applied silently to its first columns.
    renormalize_missing
        NaN values in ``values`` are always left out of the weighted sum.
        With ``False`` (default) the weights are used as they are, so a NaN
        counts as zero: mass is neither added nor redistributed.  This is the
        rule of ``normalization="full_disc"``; for an operator built with
        ``"present"`` pass ``True`` to get the filter's result on data with
        NaN.  With
        ``True`` the weights of the remaining cells are divided by their sum,
        which is what :func:`gaussian_filter` and :func:`radial_filter` do
        with ``normalization="present"``; a row without any valid value gives
        NaN.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Shape ``values.shape[:-1] + (len(row_offsets) - 1,)``.  NumPy input
        gives NumPy output; Torch input stays on its device and keeps
        autograd with respect to ``values``.
    """
    if not isinstance(renormalize_missing, (bool, np.bool_)):
        raise TypeError("'renormalize_missing' must be a boolean.")
    if n_columns is not None and values.shape[-1] != n_columns:
        raise ValueError(
            "The final axis of 'values' has length "
            f"{values.shape[-1]}, but the operator was built for "
            f"{n_columns} cells."
        )
    return compact_weighted_neighbourhood_reduce(
        values,
        column_indices,
        row_offsets,
        weights,
        normalize=bool(renormalize_missing),
    )


def radial_filter(
    values: ArrayLike,
    cell_ids: np.ndarray,
    refinement_level: int,
    *,
    radius_m: float,
    kernel: Kernel,
    normalize: bool = True,
    normalization: str | None = None,
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

    normalization
        What the weights are divided by.  ``None`` (default) keeps the
        behaviour selected by ``normalize``.  Otherwise it implies
        ``normalize=True`` and must be one of:

        ``"present"``
            The sum of the weights of the cells present in the domain (and
            not NaN), as for ``normalize=True``.  A constant field stays
            constant up to the edge of the domain.

        ``"full_disc"``
            The sum of the weights over the whole kernel disc, cells outside
            the domain included.  Cells outside the domain count in the
            denominator and contribute zero to the numerator, so mass that
            diffuses out of the domain is lost: this is a raster filter with
            ``mode="constant", cval=0``.  The kernel is evaluated on the
            distances to the cells outside the domain too.  A NaN value
            inside the domain is excluded from the numerator and kept in the
            denominator, that is it counts as zero.  The result is therefore
            never NaN because of NaN input.  A ``ValueError`` is raised if
            the weights of some disc sum to zero.

        Passing ``normalization`` together with ``normalize=False`` raises
        ``ValueError``.  ``"full_disc"`` rebuilds its weights at every call
        (only the geometry is cached): to filter many time steps, build the
        weights once with :func:`radial_filter_operator`.

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

    if normalization is not None:
        _validate_normalization(normalization)
        if not normalize:
            raise ValueError(
                "'normalization' requires normalize=True."
            )

    if ellipsoid != "WGS84":
        raise NotImplementedError(
            "Radial filtering currently supports ellipsoid='WGS84' only."
        )

    if normalization == "full_disc":
        operator = radial_filter_operator(
            cells,
            refinement_level,
            radius_m=radius,
            kernel=kernel,
            normalization="full_disc",
            domain=output_domain,
            ellipsoid=ellipsoid,
        )
        return apply_filter_operator(values, *operator)

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
    normalization: str = "present",
    width_rule: str = "sampled",
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

    normalization
        ``"present"`` (default) divides the weights by their sum over the
        cells present in the domain; ``"full_disc"`` divides by their sum over
        the whole disc, cells outside the domain contributing zero, so that
        mass diffusing out of the domain is lost.  See :func:`radial_filter`
        for the rule on NaN values.

    width_rule
        ``"sampled"`` (default) uses the Gaussian above, sampled at the cell
        centres.  ``"matched"`` makes the width of the response of every cell
        equal to ``sigma_m`` also below about one cell, see Notes.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normalized Gaussian-smoothed values.

    Notes
    -----
    Gaussian smoothing is normalized by construction.  With the default
    ``normalization="present"``, constant finite fields remain constant,
    including near partial-domain boundaries after unavailable samples are
    removed and the remaining effective weights are renormalized.

    The physical smoothing scale is independent of HEALPix refinement level:
    ``sigma_m=240`` always means 240 metres rather than a number of pixels.

    **Widths below about one cell.**  With ``width_rule="sampled"`` the
    Gaussian is only sampled at the cell centres, so the effective width of
    the response to a single cell (its one-axis RMS) is smaller than
    ``sigma_m`` when ``sigma_m`` is below about one cell, and it is almost
    zero below about 0.3 cell, where the centre cell keeps nearly all the
    weight.  The output then hardly depends on ``sigma_m``.  Above about two
    cells the difference is negligible.

    ``width_rule="matched"`` removes this limit.  The width of every cell
    whose disc lies inside the domain is ``sigma_m``; at the edge of the
    domain the rows are renormalised (``"present"``) or truncated
    (``"full_disc"``), and their effective width differs.  For a cell with ``w =
    exp(-d**2 / (2 s**2))`` on the disc, the shape parameter ``s`` is solved
    per cell so that the discrete second moment ``sum(w d**2) / sum(w)``
    equals ``2 * sigma_m**2``.  Where ``sigma_m`` is below the effective width
    of the reference operator ``G0`` (the sampled Gaussian with
    ``s = 0.6`` cell), the cell uses ``I + (sigma_m / sigma_0)**2 (G0 - I)``
    instead, with ``sigma_0`` the effective width of ``G0`` there.  The result
    is continuous and monotone in ``sigma_m``, and tends to the identity when
    ``sigma_m`` tends to zero.  The support is ``truncate * max(sigma_m,
    0.6 cell)``, and the solved shape is evaluated on the whole disc,
    whichever the normalisation.

    Geometry and Gaussian weights are retained in bounded caches. If one plan
    exceeds a limit, a warning reports the measured requirement and suggests
    :func:`configure_radial_filter_cache` or halo-safe tiling. Use
    :func:`radial_filter_cache_info` to inspect current limits and usage.
    The default options use both caches; the other options use the geometry
    cache only.

    To apply the same filter many times, or from code that works chunk by
    chunk, build the weights once with :func:`gaussian_filter_operator`.
    ``normalization="full_disc"`` and ``width_rule="matched"`` rebuild their
    weights at every call, because only the geometry is cached: in a loop
    over time steps, build the operator once and apply it with
    :func:`apply_filter_operator`.
    """
    sigma, truncation, radius = _validate_gaussian_parameters(
        sigma_m,
        truncate,
        width_rule=width_rule,
        refinement_level=refinement_level,
    )
    _validate_normalization(normalization)

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

    if normalization == "present" and width_rule == "sampled":
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

    operator = _gaussian_operator(
        cells,
        output_domain,
        refinement_level,
        sigma,
        truncation,
        radius,
        normalization,
        width_rule,
        ellipsoid,
    )
    # NaN input is excluded from the sum and kept in the denominator.
    # With "present" the weights of the remaining cells are renormalised,
    # as in the default path.
    return apply_filter_operator(
        values,
        *operator,
        renormalize_missing=normalization == "present",
    )


__all__ = [
    "FilterOperator",
    "apply_filter_operator",
    "configure_radial_filter_cache",
    "gaussian_filter",
    "gaussian_filter_operator",
    "radial_filter_operator",
    "radial_filter_cache_info",
    "radial_filter",
]
