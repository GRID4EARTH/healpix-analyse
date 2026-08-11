"""Connected-component analysis for NESTED HEALPix masks.

This module provides generic connected-component operations for binary
fields defined on HEALPix cells.

Only NESTED HEALPix indexing is currently supported.

Connectivity
------------
Two connectivity definitions are supported:

``"edge"``
    Cells are connected only when they share a HEALPix cell edge.
    This is the HEALPix analogue of Cartesian 4-connectivity.

``"edge_or_vertex"``
    Cells are connected when they share either a HEALPix cell edge or
    vertex. This is the HEALPix analogue of Cartesian 8-connectivity.

The underlying topology lookup is provided by the private
``healpix_analyse._topology`` module.

The current implementation of that private module uses ``healpy`` as a
temporary compatibility backend. Once direction-aware neighbour access
is available from ``healpix-geo`` / CDSHEALPix, only the private
topology backend needs to change.

Connected-component labeling is discrete and non-differentiable.
Torch tensors are accepted, but these operations do not participate in
autograd.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import Geod

from ._topology import Connectivity, nested_neighbours

try:
    import torch
except ImportError:  # pragma: no cover - torch is normally installed
    torch = None


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/Torch input to a CPU NumPy array."""

    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _to_numpy_cell_ids(
    cell_ids: Any,
    *,
    name: str = "cell_ids",
) -> NDArray[np.uint64]:
    """Validate and normalize HEALPix cell identifiers."""

    cells = _to_numpy(cell_ids)

    if cells.ndim == 0:
        cells = cells.reshape(1)

    if cells.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array"
        )

    if not np.issubdtype(cells.dtype, np.integer):
        raise TypeError(
            f"{name} must contain integers"
        )

    if np.issubdtype(
        cells.dtype,
        np.signedinteger,
    ) and np.any(cells < 0):
        raise ValueError(
            f"{name} must contain non-negative cell ids"
        )

    cells = cells.astype(
        np.uint64,
        copy=False,
    )

    if np.unique(cells).size != cells.size:
        raise ValueError(
            f"{name} must not contain duplicate cell ids"
        )

    return cells


def _to_numpy_mask(
    mask: Any,
) -> NDArray[np.bool_]:
    """Validate and normalize a binary mask."""

    values = _to_numpy(mask)

    if values.ndim == 0:
        values = values.reshape(1)

    if values.ndim != 1:
        raise ValueError(
            "mask must be a one-dimensional array"
        )

    if not np.issubdtype(
        values.dtype,
        np.bool_,
    ):
        raise TypeError(
            "mask must contain boolean values"
        )

    return values.astype(
        np.bool_,
        copy=False,
    )


def _to_numpy_labels(
    labels: Any,
) -> NDArray[np.int64]:
    """Validate and normalize component labels."""

    values = _to_numpy(labels)

    if values.ndim == 0:
        values = values.reshape(1)

    if values.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional array"
        )

    if not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise TypeError(
            "labels must contain integers"
        )

    values = values.astype(
        np.int64,
        copy=False,
    )

    if np.any(values < 0):
        raise ValueError(
            "labels must be non-negative"
        )

    return values


def _restore_labels_backend(
    values: NDArray[np.int64],
    reference: Any,
):
    """Return labels using the same backend/device as reference."""

    if _is_torch_tensor(reference):
        return torch.as_tensor(
            values,
            dtype=torch.int64,
            device=reference.device,
        )

    return values


def _restore_bool_backend(
    values: NDArray[np.bool_],
    reference: Any,
):
    """Return boolean values using reference backend/device."""

    if _is_torch_tensor(reference):
        return torch.as_tensor(
            values,
            dtype=torch.bool,
            device=reference.device,
        )

    return values


def _restore_numeric_backend(
    values: np.ndarray,
    reference: Any,
    *,
    integer: bool,
):
    """Return statistics using the backend of reference."""

    if _is_torch_tensor(reference):
        dtype = (
            torch.int64
            if integer
            else torch.float64
        )

        return torch.as_tensor(
            values,
            dtype=dtype,
            device=reference.device,
        )

    return values


def _validate_refinement_level(
    refinement_level: int,
) -> int:
    """Validate a NESTED HEALPix refinement level."""

    if isinstance(
        refinement_level,
        bool,
    ) or not isinstance(
        refinement_level,
        (int, np.integer),
    ):
        raise TypeError(
            "refinement_level must be an integer"
        )

    refinement_level = int(
        refinement_level
    )

    if not 0 <= refinement_level <= 29:
        raise ValueError(
            "refinement_level must be in [0, 29]"
        )

    return refinement_level


def _validate_cell_range(
    cells: NDArray[np.uint64],
    refinement_level: int,
    *,
    name: str,
) -> None:
    """Check that cell ids are valid at this refinement level."""

    nside = 1 << refinement_level
    npix = 12 * nside * nside

    if np.any(cells >= npix):
        raise ValueError(
            f"{name} contains a cell id outside "
            f"the valid range [0, {npix}) for "
            f"refinement_level={refinement_level}"
        )


def _prepare_domain(
    cell_ids: NDArray[np.uint64],
    domain: Any,
) -> NDArray[np.uint64]:
    """Normalize the processing domain.

    ``domain=None`` means that every supplied cell participates.

    If an explicit domain is supplied, every domain cell must occur in
    ``cell_ids``. Output order follows the exact domain order.
    """

    if domain is None:
        return cell_ids

    domain_cells = _to_numpy_cell_ids(
        domain,
        name="domain",
    )

    supplied = set(
        map(int, cell_ids)
    )

    missing = [
        int(cell)
        for cell in domain_cells
        if int(cell) not in supplied
    ]

    if missing:
        raise ValueError(
            "domain must be a subset of cell_ids"
        )

    return domain_cells


class _UnionFind:
    """Small deterministic union-find implementation."""

    def __init__(self, size: int) -> None:
        self.parent = np.arange(
            size,
            dtype=np.int64,
        )
        self.rank = np.zeros(
            size,
            dtype=np.uint8,
        )

    def find(self, item: int) -> int:
        parent = self.parent

        while parent[item] != item:
            parent[item] = parent[
                parent[item]
            ]
            item = int(parent[item])

        return item

    def union(
        self,
        first: int,
        second: int,
    ) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        rank_first = self.rank[root_first]
        rank_second = self.rank[root_second]

        if rank_first < rank_second:
            self.parent[root_first] = (
                root_second
            )
        elif rank_first > rank_second:
            self.parent[root_second] = (
                root_first
            )
        else:
            self.parent[root_second] = (
                root_first
            )
            self.rank[root_first] += 1


def connected_components(
    mask: ArrayLike,
    cell_ids: ArrayLike,
    refinement_level: int,
    *,
    connectivity: Connectivity = "edge",
    domain: ArrayLike | None = None,
):
    """Label connected active regions of a HEALPix mask.

    Parameters
    ----------
    mask
        One-dimensional boolean NumPy array or Torch tensor.

        ``True`` means active/foreground.

    cell_ids
        NESTED HEALPix cell id corresponding to each value in ``mask``.

    refinement_level
        HEALPix refinement level.

    connectivity
        ``"edge"``
            Cells must share a HEALPix edge.

        ``"edge_or_vertex"``
            Cells may share either an edge or a vertex.

    domain
        Optional subset of ``cell_ids`` defining the valid processing
        topology.

        ``domain=None`` means ``domain=cell_ids``.

        Cells outside ``domain`` are absent from the connectivity graph.
        Connectivity therefore cannot pass through them.

        Output order follows the exact order of ``domain``.

    Returns
    -------
    labels
        Component label for every cell in the output domain.

        ``0`` means inactive/background.

        Components are numbered ``1, 2, ...`` deterministically according
        to the first active cell encountered in input ``domain`` order.

        NumPy input returns a NumPy array. Torch mask input returns a Torch
        tensor on the same device as ``mask``.

    n_components : int
        Number of foreground components.

    Notes
    -----
    Connected-component labeling is discrete and does not participate in
    Torch autograd.
    """

    refinement_level = (
        _validate_refinement_level(
            refinement_level
        )
    )

    values = _to_numpy_mask(mask)
    cells = _to_numpy_cell_ids(
        cell_ids
    )

    if values.size != cells.size:
        raise ValueError(
            "mask and cell_ids must have the same length"
        )

    _validate_cell_range(
        cells,
        refinement_level,
        name="cell_ids",
    )

    domain_cells = _prepare_domain(
        cells,
        domain,
    )

    _validate_cell_range(
        domain_cells,
        refinement_level,
        name="domain",
    )

    if connectivity not in (
        "edge",
        "edge_or_vertex",
    ):
        raise ValueError(
            "connectivity must be "
            "'edge' or 'edge_or_vertex'"
        )

    if domain_cells.size == 0:
        labels = np.empty(
            0,
            dtype=np.int64,
        )

        return (
            _restore_labels_backend(
                labels,
                mask,
            ),
            0,
        )

    # Map cell id -> mask value.
    value_by_cell = {
        int(cell): bool(value)
        for cell, value in zip(
            cells,
            values,
            strict=True,
        )
    }

    domain_active = np.asarray(
        [
            value_by_cell[int(cell)]
            for cell in domain_cells
        ],
        dtype=np.bool_,
    )

    active_positions = np.flatnonzero(
        domain_active
    )

    if active_positions.size == 0:
        labels = np.zeros(
            domain_cells.size,
            dtype=np.int64,
        )

        return (
            _restore_labels_backend(
                labels,
                mask,
            ),
            0,
        )

    active_cells = domain_cells[
        active_positions
    ]

    # Position inside the active-cell list.
    active_index = {
        int(cell): index
        for index, cell in enumerate(
            active_cells
        )
    }

    neighbours = nested_neighbours(
        active_cells,
        refinement_level,
        connectivity=connectivity,
    )

    union_find = _UnionFind(
        active_cells.size
    )

    # Connect active cells only when the neighbour is also an active cell
    # inside the supplied processing domain.
    for index, row in enumerate(
        neighbours
    ):
        for neighbour in row:
            neighbour_id = int(neighbour)

            if neighbour_id < 0:
                continue

            other = active_index.get(
                neighbour_id
            )

            if other is None:
                continue

            union_find.union(
                index,
                other,
            )

    labels = np.zeros(
        domain_cells.size,
        dtype=np.int64,
    )

    root_to_label: dict[int, int] = {}
    next_label = 1

    # Deterministic numbering follows domain order, not union-find root
    # number and not sorted HEALPix id order.
    for domain_position in active_positions:
        cell = int(
            domain_cells[domain_position]
        )

        active_position = active_index[
            cell
        ]

        root = union_find.find(
            active_position
        )

        label = root_to_label.get(
            root
        )

        if label is None:
            label = next_label
            root_to_label[root] = label
            next_label += 1

        labels[domain_position] = label

    n_components = next_label - 1

    return (
        _restore_labels_backend(
            labels,
            mask,
        ),
        n_components,
    )


def component_size(
    labels: ArrayLike,
):
    """Return the number of cells in each connected component.

    Parameters
    ----------
    labels
        One-dimensional component labels such as the output of
        :func:`connected_components`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Array indexed by component label.

        ``sizes[0] == 0`` by definition.

        For component ``k > 0``::

            sizes[k] == number of cells labelled k

    Examples
    --------
    For::

        labels = [1, 1, 0, 2, 2, 2]

    the result is::

        [0, 2, 3]
    """

    values = _to_numpy_labels(
        labels
    )

    if values.size == 0:
        result = np.zeros(
            1,
            dtype=np.int64,
        )

        return _restore_numeric_backend(
            result,
            labels,
            integer=True,
        )

    max_label = int(
        values.max()
    )

    result = np.bincount(
        values,
        minlength=max_label + 1,
    ).astype(
        np.int64,
        copy=False,
    )

    # Background is not a component.
    result[0] = 0

    return _restore_numeric_backend(
        result,
        labels,
        integer=True,
    )


def _ellipsoid_surface_area_m2(
    ellipsoid: str,
) -> float:
    """Return total surface area of an oblate reference ellipsoid.

    The parameters are obtained from PROJ through ``pyproj.Geod``.

    The formula is the exact surface-area expression for an oblate
    spheroid. A HEALPix authalic mapping preserves this total surface
    area, so every HEALPix cell at one refinement level has exactly the
    ellipsoid surface area divided by the number of cells.
    """

    if not isinstance(
        ellipsoid,
        str,
    ):
        raise TypeError(
            "ellipsoid must be a string"
        )

    try:
        geod = Geod(
            ellps=ellipsoid
        )
    except Exception as exc:
        raise ValueError(
            f"unknown ellipsoid: {ellipsoid!r}"
        ) from exc

    a = float(geod.a)
    f = float(geod.f)

    if not math.isfinite(a) or a <= 0:
        raise ValueError(
            "invalid ellipsoid semi-major axis"
        )

    # Sphere.
    if f == 0:
        return 4.0 * math.pi * a * a

    b = a * (1.0 - f)

    eccentricity_squared = (
        1.0
        - (b * b) / (a * a)
    )

    eccentricity = math.sqrt(
        max(
            0.0,
            eccentricity_squared,
        )
    )

    if eccentricity == 0:
        return 4.0 * math.pi * a * a

    return (
        2.0
        * math.pi
        * a
        * a
        * (
            1.0
            + (
                (1.0 - eccentricity_squared)
                / eccentricity
            )
            * math.atanh(
                eccentricity
            )
        )
    )


def healpix_cell_area(
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
) -> float:
    """Return equal HEALPix cell area in square metres.

    The area is computed from the total surface area of the selected
    reference ellipsoid divided by the number of equal-area HEALPix cells
    at this refinement level.

    For the S2MSI-to-HEALPix use case, ``ellipsoid="WGS84"`` should be
    used.
    """

    refinement_level = (
        _validate_refinement_level(
            refinement_level
        )
    )

    total_area = (
        _ellipsoid_surface_area_m2(
            ellipsoid
        )
    )

    nside = 1 << refinement_level
    npix = 12 * nside * nside

    return total_area / npix


def component_area(
    labels: ArrayLike,
    refinement_level: int,
    *,
    ellipsoid: str = "WGS84",
):
    """Return physical area of each component in square metres.

    The returned array is indexed by component label.

    ``areas[0] == 0``.

    HEALPix cells at a fixed refinement level are equal-area, therefore
    component area is exactly:

        component cell count * HEALPix cell area

    under the selected ellipsoidal equal-area HEALPix model.
    """

    sizes = component_size(
        labels
    )

    if _is_torch_tensor(sizes):
        sizes_numpy = (
            sizes.detach()
            .cpu()
            .numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )
    else:
        sizes_numpy = np.asarray(
            sizes,
            dtype=np.float64,
        )

    cell_area = healpix_cell_area(
        refinement_level,
        ellipsoid=ellipsoid,
    )

    areas = sizes_numpy * cell_area
    areas[0] = 0.0

    return _restore_numeric_backend(
        areas,
        labels,
        integer=False,
    )


def remove_small_components(
    mask: ArrayLike,
    cell_ids: ArrayLike,
    refinement_level: int,
    *,
    min_cells: int | None = None,
    min_area_m2: float | None = None,
    connectivity: Connectivity = "edge",
    domain: ArrayLike | None = None,
    ellipsoid: str = "WGS84",
):
    """Remove connected components smaller than a threshold.

    Exactly one of ``min_cells`` or ``min_area_m2`` must be supplied.

    Parameters
    ----------
    mask
        Binary input mask.

    cell_ids
        NESTED HEALPix cell ids corresponding to ``mask``.

    refinement_level
        HEALPix refinement level.

    min_cells
        Minimum number of cells required to retain a component.

    min_area_m2
        Minimum physical area in square metres required to retain a
        component.

    connectivity
        ``"edge"`` or ``"edge_or_vertex"``.

    domain
        Optional processing domain.

        If supplied, the returned mask follows exact ``domain`` order.
        Cells outside the domain do not participate in connectivity.

    ellipsoid
        Reference ellipsoid used for physical-area thresholds.
        Default is ``"WGS84"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Cleaned boolean mask in output-domain order.
    """

    if (
        min_cells is None
        and min_area_m2 is None
    ) or (
        min_cells is not None
        and min_area_m2 is not None
    ):
        raise ValueError(
            "exactly one of min_cells or "
            "min_area_m2 must be provided"
        )

    if min_cells is not None:
        if isinstance(
            min_cells,
            bool,
        ) or not isinstance(
            min_cells,
            (int, np.integer),
        ):
            raise TypeError(
                "min_cells must be an integer"
            )

        if min_cells < 0:
            raise ValueError(
                "min_cells must be non-negative"
            )

    if min_area_m2 is not None:
        if isinstance(
            min_area_m2,
            bool,
        ) or not isinstance(
            min_area_m2,
            (int, float, np.number),
        ):
            raise TypeError(
                "min_area_m2 must be a number"
            )

        min_area_m2 = float(
            min_area_m2
        )

        if (
            not math.isfinite(
                min_area_m2
            )
            or min_area_m2 < 0
        ):
            raise ValueError(
                "min_area_m2 must be a finite "
                "non-negative number"
            )

    labels, _ = connected_components(
        mask,
        cell_ids,
        refinement_level,
        connectivity=connectivity,
        domain=domain,
    )

    labels_numpy = _to_numpy_labels(
        labels
    )

    if min_cells is not None:
        sizes = component_size(
            labels_numpy
        )

        keep_component = (
            sizes >= int(min_cells)
        )

    else:
        areas = component_area(
            labels_numpy,
            refinement_level,
            ellipsoid=ellipsoid,
        )

        keep_component = (
            areas >= min_area_m2
        )

    keep_component = np.asarray(
        keep_component,
        dtype=np.bool_,
    )

    # Background is always background, including threshold 0.
    keep_component[0] = False

    cleaned = keep_component[
        labels_numpy
    ]

    return _restore_bool_backend(
        cleaned,
        mask,
    )
