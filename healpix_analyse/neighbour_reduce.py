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

from healpix_analyse import _neighbourhood
from healpix_analyse._neighbourhood import (
    NeighbourhoodMethod,
    build_neighbourhoods,
)
from healpix_analyse._ellipsoid import canonicalize_ellipsoid
from healpix_analyse._stencil_backend import (
    BACKENDS,
    FAST_REDUCTIONS,
    Stencil,
    stencil_reduce,
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

    # A stable sort: the values are those of any sort, and on ties the
    # subgradient goes to the first sample (the rule of the fused backends).
    sorted_values = torch.sort(
        values,
        dim=-1,
        stable=True,
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


def _generic_reduce(
    work: torch.Tensor,
    stencil,
    reduction: str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorised ``std``, ``any``, ``all``, ``mode`` and integer reductions.

    The neighbourhoods are gathered in row chunks of bounded size, padded
    with the identity of the reduction (or masked).
    """

    batch = work.shape[0]
    idx = stencil.idx
    counts = stencil.cnt.long()
    n_out, kp = idx.shape
    output = torch.empty(
        (batch, n_out),
        dtype=output_dtype,
        device=work.device,
    )

    if reduction == "std" and not work.dtype.is_floating_point:
        work = work.to(torch.float64)

    if reduction == "sum" and not work.dtype.is_floating_point:
        work = work.to(torch.int64)

    if reduction in {"min", "max", "mode"} and work.dtype == torch.bool:
        work = work.to(torch.uint8)

    if reduction in {"min", "max", "mode"}:
        info = torch.iinfo(work.dtype)
        fill = info.min if reduction == "max" else info.max
    elif reduction == "all":
        fill = True
    elif reduction == "any":
        fill = False
    else:
        fill = 0

    extended = torch.cat(
        [work, work.new_full((batch, 1), fill)],
        dim=1,
    )

    step = max(1, (1 << 24) // max(1, batch * max(kp, 1)))
    taps = torch.arange(kp, device=work.device)

    for r0 in range(0, n_out, step):
        r1 = min(n_out, r0 + step)
        gathered = extended[:, idx[r0:r1].long()]
        c = counts[r0:r1]
        mask = taps[None, :] < c[:, None]

        if reduction == "sum":
            result = gathered.sum(dim=-1)
        elif reduction == "min":
            result = gathered.amin(dim=-1)
        elif reduction == "max":
            result = gathered.amax(dim=-1)
        elif reduction == "any":
            result = gathered.any(dim=-1)
        elif reduction == "all":
            result = gathered.all(dim=-1)
        elif reduction == "std":
            m = mask.to(gathered.dtype)
            mean = (gathered * m).sum(dim=-1) / c
            centred = (gathered - mean[..., None]) * m
            variance = (centred * centred).sum(dim=-1) / c
            # sqrt has an infinite derivative at 0 (NaN gradient on a
            # constant neighbourhood); torch.std, the reference, gives 0.
            # (NaN still propagates: only an exact zero is special-cased.)
            zero = variance == 0
            result = torch.where(
                zero,
                torch.zeros_like(variance),
                torch.sqrt(torch.where(zero, torch.ones_like(variance), variance)),
            )
        elif reduction == "mode":
            result = _vectorised_mode(gathered, mask)
        else:  # pragma: no cover - dispatched elsewhere
            raise AssertionError(reduction)

        output[:, r0:r1] = result.to(output_dtype)

    return output


def _vectorised_mode(
    gathered: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smallest most frequent value of every row (pads sort last)."""

    values, _ = torch.sort(gathered, dim=-1)
    batch, rows, kp = values.shape
    valid = mask.expand(batch, -1, -1)
    start = torch.ones_like(valid)
    start[..., 1:] = values[..., 1:] != values[..., :-1]
    run = torch.cumsum(start.to(torch.int64), dim=-1) - 1
    run_counts = torch.zeros(
        (batch, rows, kp),
        dtype=torch.int64,
        device=values.device,
    ).scatter_add_(-1, run, valid.to(torch.int64))
    best_run = torch.argmax(run_counts, dim=-1, keepdim=True)
    first = torch.argmax(
        ((run == best_run) & valid).to(torch.int8),
        dim=-1,
        keepdim=True,
    )
    return values.gather(-1, first)[..., 0]


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
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
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

        offsets, positions = self._build_csr()
        self._stencil = Stencil(
            offsets,
            positions,
            self.cell_ids.size,
        )
        self._counts = np.diff(offsets)
        empty = np.flatnonzero(self._counts == 0)
        # first output cell with an empty neighbourhood (checked at every
        # call for the reductions that need values; kept off the hot path)
        self._first_empty = int(empty[0]) if empty.size else -1

    @property
    def output_cell_ids(
        self,
    ) -> np.ndarray:
        """Return output HEALPix IDs in exact output order."""

        return self.domain.copy()

    def _geometric_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Geometric neighbourhoods of the domain cells in CSR form."""

        if self.radius_m == 0:
            return (
                np.arange(self.domain.size + 1, dtype=np.int64),
                self.domain.copy(),
            )

        if build_neighbourhoods is _neighbourhood.build_neighbourhoods:
            return _neighbourhood.build_neighbourhoods_csr(
                self.domain,
                self.radius_m,
                self.refinement_level,
                neighbourhood=self.neighbourhood,
                ellipsoid=self.ellipsoid,
            )

        # ``build_neighbourhoods`` was replaced (tests inject a fake
        # geometry): use the per-cell lists it returns.
        rows = build_neighbourhoods(
            self.domain,
            self.radius_m,
            self.refinement_level,
            neighbourhood=self.neighbourhood,
            ellipsoid=self.ellipsoid,
        )
        counts = np.asarray(
            [len(row) for row in rows],
            dtype=np.int64,
        )
        offsets = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        flat = (
            np.concatenate(
                [np.asarray(row, dtype=np.uint64) for row in rows]
            )
            if counts.sum()
            else np.empty(0, dtype=np.uint64)
        )
        return offsets, flat

    def _build_csr(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Input-array positions of every effective neighbourhood (CSR).

        Row ``i`` holds the sorted, unique positions in ``cell_ids`` of the
        cells of ``domain`` that lie in the neighbourhood of ``domain[i]``
        (and of ``domain[i]`` itself when ``include_self``).  Geometrically
        neighbouring cells outside the processing domain are absent from the
        effective neighbourhood: they are NOT zero padded.
        """

        n_domain = self.domain.size

        if n_domain == 0:
            return (
                np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        offsets, flat = self._geometric_csr()
        flat = np.asarray(flat, dtype=np.uint64)
        counts = np.diff(offsets)
        rows = np.repeat(
            np.arange(n_domain, dtype=np.int64),
            counts,
        )

        # Keep the neighbours that belong to the processing domain.
        domain_order = np.argsort(self.domain, kind="stable")
        sorted_domain = self.domain[domain_order]
        at = np.searchsorted(sorted_domain, flat)
        at_clipped = np.minimum(at, n_domain - 1)
        in_domain = sorted_domain[at_clipped] == flat

        if not self.include_self:
            in_domain &= flat != self.domain[rows]

        rows = rows[in_domain]
        flat = flat[in_domain]

        # Map cell IDs to positions in ``cell_ids`` (domain is a subset).
        source_order = np.argsort(self.cell_ids, kind="stable")
        sorted_sources = self.cell_ids[source_order]
        positions = source_order[
            np.searchsorted(sorted_sources, flat)
        ].astype(np.int64)

        if self.include_self:
            target_positions = source_order[
                np.searchsorted(sorted_sources, self.domain)
            ].astype(np.int64)
            rows = np.concatenate(
                [rows, np.arange(n_domain, dtype=np.int64)]
            )
            positions = np.concatenate(
                [positions, target_positions]
            )

        # Reductions are order-independent. Sorting by source-array
        # position makes the internal representation deterministic.
        order = np.lexsort((positions, rows))
        rows = rows[order]
        positions = positions[order]

        if positions.size:
            keep = np.ones(positions.size, dtype=bool)
            keep[1:] = (rows[1:] != rows[:-1]) | (
                positions[1:] != positions[:-1]
            )
            rows = rows[keep]
            positions = positions[keep]

        new_offsets = np.zeros(n_domain + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(rows, minlength=n_domain),
            out=new_offsets[1:],
        )

        return new_offsets, positions

    @property
    def _positions(
        self,
    ) -> tuple[np.ndarray, ...]:
        """Input positions of each effective neighbourhood (one array per cell)."""

        stencil = self._stencil

        return tuple(
            stencil.idx[row, : stencil.cnt[row]].astype(np.int64)
            for row in range(stencil.n_out)
        )

    @property
    def max_neighbours(
        self,
    ) -> int:
        """Largest number of cells in an effective neighbourhood."""

        return self._stencil.kp

    def _reference_reduce(
        self,
        tensor: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        """Reference implementation: one ``index_select`` per output cell.

        Kept as the ground truth of the vectorised backends (tests).
        ``tensor`` is ``[B, N_in]``; returns ``[B, N_out]``.
        """

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

            selected = tensor.index_select(
                -1,
                index,
            )

            output_columns.append(
                _reduce_selected(
                    selected,
                    reduction,
                    target_cell=int(target_cell),
                )
            )

        return torch.stack(
            output_columns,
            dim=-1,
        )

    def _check_empty_neighbourhoods(
        self,
        reduction: str,
    ) -> None:
        """Raise like the reference for an undefined empty reduction."""

        if reduction in {"sum", "count", "any", "all"}:
            return

        if self._first_empty >= 0:
            target_cell = int(self.domain[self._first_empty])

            raise ValueError(
                f"Cell {target_cell} has an empty neighbourhood "
                f"for reduction={reduction!r}. "
                "Use include_self=True, increase radius_m, enlarge "
                "the processing domain, or select a reduction with "
                "a defined empty-neighbourhood identity."
            )

    def _reduce(
        self,
        flattened: torch.Tensor,
        reduction: str,
        output_dtype: torch.dtype,
        *,
        backend: str,
    ) -> torch.Tensor:
        """Vectorised reduction of ``flattened [B, N_in]`` -> ``[B, N_out]``."""

        stencil = self._stencil
        device = flattened.device

        if reduction == "count":
            counts = stencil.on(device).cnt

            return counts.to(torch.int64).expand(
                flattened.shape[0],
                -1,
            ).clone()

        work = flattened

        if (
            reduction in {"mean", "median"}
            and not work.dtype.is_floating_point
        ):
            work = work.to(torch.float64)

        if (
            reduction in FAST_REDUCTIONS
            and work.dtype.is_floating_point
        ):
            return stencil_reduce(
                work,
                stencil,
                reduction,
                backend=backend,
            )

        return _generic_reduce(
            work,
            stencil.on(device),
            reduction,
            output_dtype,
        )

    def __call__(
        self,
        values: ArrayLike,
        *,
        reduction: Reduction = "mean",
        backend: str = "auto",
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
        backend
            ``"auto"`` (default), ``"triton"`` or ``"torch"``.  ``"auto"``
            runs the fused Triton kernels for ``sum``, ``mean``, ``min``,
            ``max`` and ``median`` of floating-point CUDA tensors when Triton
            is installed, and the PyTorch backend otherwise.  Both give the
            same values; see the notes.

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

        Gradients: ``sum``, ``mean``, ``min``, ``max`` and ``median`` have a
        hand-written backward pass (deterministic: a gather on the transposed
        neighbourhoods on CUDA, ``index_add_`` on CPU).  ``min``, ``max`` and
        ``median`` use the subgradient of torch: the whole gradient goes to
        the selected value (half to each central value of an even median),
        ties go to the first cell in the order of ``cell_ids``, a NaN row of
        ``median`` gets no gradient and ``min`` / ``max`` select the first
        NaN.
        """

        reduction = _validate_reduction(
            reduction
        )

        if backend not in BACKENDS:
            raise ValueError(
                f"'backend' must be one of {BACKENDS}, got {backend!r}."
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

        if self.domain.size == 0:
            output = torch.empty(
                (*leading_shape, 0),
                dtype=output_dtype,
                device=tensor.device,
            )

        else:
            self._check_empty_neighbourhoods(reduction)

            output = self._reduce(
                flattened,
                reduction,
                output_dtype,
                backend=backend,
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
    ellipsoid = canonicalize_ellipsoid(ellipsoid)

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
    ellipsoid = canonicalize_ellipsoid(ellipsoid)

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
    ellipsoid = canonicalize_ellipsoid(ellipsoid)

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
    ellipsoid = canonicalize_ellipsoid(ellipsoid)

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
    ellipsoid = canonicalize_ellipsoid(ellipsoid)

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
