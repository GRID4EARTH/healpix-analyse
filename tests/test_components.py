import math

import healpy as hp
import numpy as np
import pytest

from healpix_analyse._topology import (
    nested_neighbours,
)
from healpix_analyse.components import (
    component_area,
    component_size,
    connected_components,
    healpix_cell_area,
    remove_small_components,
)


def _find_edge_pair(level=3):
    """Return two cells sharing an edge."""

    cell = np.uint64(0)

    neighbours = nested_neighbours(
        np.array([cell]),
        level,
        connectivity="edge",
    )

    return int(cell), int(neighbours[0, 0])


def _find_vertex_only_pair(level=3):
    """Return two immediate cells touching only at a vertex."""

    npix = 12 * (2**level) ** 2

    cells = np.arange(
        npix,
        dtype=np.uint64,
    )

    edge = nested_neighbours(
        cells,
        level,
        connectivity="edge",
    )

    full = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    for index, cell in enumerate(cells):
        edge_set = set(
            int(value)
            for value in edge[index]
            if value >= 0
        )

        for candidate in full[index]:
            candidate = int(candidate)

            if (
                candidate >= 0
                and candidate not in edge_set
            ):
                return int(cell), candidate

    raise AssertionError(
        "could not find a vertex-only pair"
    )


def _find_edge_chain(length=3, level=3):
    """Construct a simple path of distinct edge-connected cells."""

    start = 0

    chain = [start]
    previous = None
    current = start

    while len(chain) < length:
        neighbours = nested_neighbours(
            np.array(
                [current],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )[0]

        candidate = None

        for neighbour in neighbours:
            neighbour = int(neighbour)

            if neighbour == previous:
                continue

            if neighbour in chain:
                continue

            candidate = neighbour
            break

        if candidate is None:
            raise AssertionError(
                "could not construct edge chain"
            )

        previous = current
        current = candidate
        chain.append(current)

    return chain


def _find_base_pixel_crossing_edge_pair(
    level=3,
):
    """Find edge-sharing cells belonging to different base pixels."""

    npix = 12 * (2**level) ** 2
    cells_per_base_pixel = 4**level

    cells = np.arange(
        npix,
        dtype=np.uint64,
    )

    neighbours = nested_neighbours(
        cells,
        level,
        connectivity="edge",
    )

    for index, cell in enumerate(cells):
        cell_id = int(cell)

        base_pixel = (
            cell_id
            // cells_per_base_pixel
        )

        for neighbour in neighbours[index]:
            neighbour_id = int(neighbour)

            neighbour_base_pixel = (
                neighbour_id
                // cells_per_base_pixel
            )

            if (
                neighbour_base_pixel
                != base_pixel
            ):
                return (
                    cell_id,
                    neighbour_id,
                )

    raise AssertionError(
        "could not find an edge neighbour "
        "crossing a HEALPix base-pixel boundary"
    )


def _find_polar_edge_pair(
    level=3,
):
    """Find an edge-connected pair in a high-latitude HEALPix region."""

    nside = 2**level
    npix = 12 * nside**2

    cells = np.arange(
        npix,
        dtype=np.int64,
    )

    lon, lat = hp.pix2ang(
        nside,
        cells,
        nest=True,
        lonlat=True,
    )

    # Pick the cell centre with the largest absolute latitude.
    index = int(
        np.argmax(
            np.abs(lat)
        )
    )

    cell = int(
        cells[index]
    )

    neighbours = nested_neighbours(
        np.array(
            [cell],
            dtype=np.uint64,
        ),
        level,
        connectivity="edge",
    )[0]

    neighbour = int(
        neighbours[0]
    )

    return (
        cell,
        neighbour,
        float(lat[index]),
    )


def _find_longitude_wrap_edge_pair(
    level=3,
):
    """Find edge-sharing cells crossing the 0/360-degree longitude seam."""

    nside = 2**level
    npix = 12 * nside**2

    cells = np.arange(
        npix,
        dtype=np.int64,
    )

    lon, _ = hp.pix2ang(
        nside,
        cells,
        nest=True,
        lonlat=True,
    )

    neighbours = nested_neighbours(
        cells.astype(
            np.uint64
        ),
        level,
        connectivity="edge",
    )

    for index, cell in enumerate(cells):
        cell_lon = float(
            lon[index]
        )

        for neighbour in neighbours[index]:
            neighbour_id = int(
                neighbour
            )

            neighbour_lon = float(
                lon[neighbour_id]
            )

            # A large numeric longitude difference indicates that the
            # topological edge crosses the 0/360-degree coordinate seam.
            if abs(
                cell_lon
                - neighbour_lon
            ) > 180.0:
                return (
                    int(cell),
                    neighbour_id,
                    cell_lon,
                    neighbour_lon,
                )

    raise AssertionError(
        "could not find an edge neighbour "
        "crossing the longitude seam"
    )


def _find_complete_immediate_neighbourhood(
    level=3,
):
    """Find a cell having all eight edge-or-vertex neighbour positions."""

    npix = 12 * (2**level) ** 2

    cells = np.arange(
        npix,
        dtype=np.uint64,
    )

    neighbours = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    for cell, row in zip(
        cells,
        neighbours,
        strict=True,
    ):
        if np.all(
            row >= 0
        ):
            return (
                int(cell),
                row.astype(
                    np.uint64
                ),
            )

    raise AssertionError(
        "could not find a HEALPix cell "
        "with eight immediate neighbours"
    )


def test_single_active_cell():
    level = 3

    cell_ids = np.array(
        [0],
        dtype=np.uint64,
    )

    mask = np.array(
        [True],
        dtype=bool,
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1]),
    )

    assert n_components == 1


def test_single_inactive_cell():
    labels, n_components = (
        connected_components(
            np.array(
                [False],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([0]),
    )

    assert n_components == 0


def test_two_edge_neighbours_are_connected():
    level = 3
    first, second = _find_edge_pair(
        level
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 1]),
    )

    assert n_components == 1


def test_vertex_only_pair_is_disconnected_with_edge_connectivity():
    level = 3

    first, second = (
        _find_vertex_only_pair(
            level
        )
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 2]),
    )

    assert n_components == 2


def test_vertex_only_pair_is_connected_with_edge_or_vertex():
    level = 3

    first, second = (
        _find_vertex_only_pair(
            level
        )
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge_or_vertex",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 1]),
    )

    assert n_components == 1


def test_inactive_cell_breaks_component():
    level = 3
    chain = _find_edge_chain(
        length=3,
        level=level,
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, False, True],
                dtype=bool,
            ),
            np.array(
                chain,
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 0, 2]),
    )

    assert n_components == 2


def test_domain_breaks_connectivity():
    level = 3
    chain = _find_edge_chain(
        length=3,
        level=level,
    )

    mask = np.array(
        [True, True, True],
        dtype=bool,
    )

    cell_ids = np.array(
        chain,
        dtype=np.uint64,
    )

    domain = np.array(
        [chain[0], chain[2]],
        dtype=np.uint64,
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            connectivity="edge",
            domain=domain,
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 2]),
    )

    assert n_components == 2


def test_domain_output_follows_domain_order():
    level = 3
    chain = _find_edge_chain(
        length=3,
        level=level,
    )

    cell_ids = np.array(
        chain,
        dtype=np.uint64,
    )

    mask = np.array(
        [True, False, True],
        dtype=bool,
    )

    domain = np.array(
        [
            chain[2],
            chain[0],
        ],
        dtype=np.uint64,
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            domain=domain,
        )
    )

    # Both active, disconnected, and component numbering follows exact
    # domain order.
    np.testing.assert_array_equal(
        labels,
        np.array([1, 2]),
    )

    assert n_components == 2


def test_deterministic_component_numbering_follows_input_order():
    level = 3

    first, second = (
        _find_vertex_only_pair(
            level
        )
    )

    cell_ids = np.array(
        [second, first],
        dtype=np.uint64,
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            cell_ids,
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array([1, 2]),
    )

    assert n_components == 2


def test_component_size():
    labels = np.array(
        [1, 1, 0, 2, 2, 2],
        dtype=np.int64,
    )

    sizes = component_size(
        labels
    )

    np.testing.assert_array_equal(
        sizes,
        np.array(
            [0, 2, 3],
            dtype=np.int64,
        ),
    )


def test_component_size_empty():
    sizes = component_size(
        np.array(
            [],
            dtype=np.int64,
        )
    )

    np.testing.assert_array_equal(
        sizes,
        np.array(
            [0],
            dtype=np.int64,
        ),
    )


@pytest.mark.parametrize(
    "level",
    [0, 1, 5, 10],
)
def test_equal_area(level):
    cell_area = healpix_cell_area(
        level,
        ellipsoid="WGS84",
    )

    npix = 12 * (2**level) ** 2

    # Reconstruct the same ellipsoid surface area from every level.
    total_area = (
        cell_area
        * npix
    )

    level_zero_total = (
        healpix_cell_area(
            0,
            ellipsoid="WGS84",
        )
        * 12
    )

    assert total_area == pytest.approx(
        level_zero_total,
        rel=1e-14,
    )


def test_cell_area_decreases_by_four_each_level():
    area_level_5 = healpix_cell_area(
        5,
        ellipsoid="WGS84",
    )

    area_level_6 = healpix_cell_area(
        6,
        ellipsoid="WGS84",
    )

    assert area_level_6 == pytest.approx(
        area_level_5 / 4.0,
        rel=1e-14,
    )


def test_component_area_is_size_times_cell_area():
    level = 8

    labels = np.array(
        [1, 1, 0, 2, 2, 2],
        dtype=np.int64,
    )

    areas = component_area(
        labels,
        level,
        ellipsoid="WGS84",
    )

    cell_area = healpix_cell_area(
        level,
        ellipsoid="WGS84",
    )

    np.testing.assert_allclose(
        areas,
        np.array(
            [
                0.0,
                2.0 * cell_area,
                3.0 * cell_area,
            ]
        ),
    )


def test_remove_small_components_by_cell_count():
    level = 3

    chain = _find_edge_chain(
        length=3,
        level=level,
    )

    # Component A: two connected cells.
    first = chain[0]
    second = chain[1]

    # Find an active cell outside that component.
    edge_set = set(
        nested_neighbours(
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        ).ravel().tolist()
    )

    npix = 12 * (2**level) ** 2

    isolated = next(
        cell
        for cell in range(npix)
        if (
            cell not in (first, second)
            and cell not in edge_set
        )
    )

    cell_ids = np.array(
        [
            first,
            second,
            isolated,
        ],
        dtype=np.uint64,
    )

    mask = np.array(
        [True, True, True],
        dtype=bool,
    )

    cleaned = remove_small_components(
        mask,
        cell_ids,
        level,
        min_cells=2,
        connectivity="edge",
    )

    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [True, True, False],
            dtype=bool,
        ),
    )


def test_remove_small_components_by_area():
    level = 8

    first, second = _find_edge_pair(
        level
    )

    npix = 12 * (2**level) ** 2

    neighbours = set(
        nested_neighbours(
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        ).ravel().tolist()
    )

    isolated = next(
        cell
        for cell in range(npix)
        if (
            cell not in (first, second)
            and cell not in neighbours
        )
    )

    cell_ids = np.array(
        [
            first,
            second,
            isolated,
        ],
        dtype=np.uint64,
    )

    mask = np.array(
        [True, True, True],
        dtype=bool,
    )

    cell_area = healpix_cell_area(
        level,
        ellipsoid="WGS84",
    )

    cleaned = remove_small_components(
        mask,
        cell_ids,
        level,
        min_area_m2=1.5 * cell_area,
        connectivity="edge",
        ellipsoid="WGS84",
    )

    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [True, True, False],
            dtype=bool,
        ),
    )


def test_remove_small_components_requires_exactly_one_threshold():
    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        remove_small_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
        )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        remove_small_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
            min_cells=1,
            min_area_m2=1.0,
        )


def test_domain_must_be_subset_of_cell_ids():
    with pytest.raises(
        ValueError,
        match="subset",
    ):
        connected_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
            domain=np.array(
                [1],
                dtype=np.uint64,
            ),
        )


def test_duplicate_cell_ids_are_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [0, 0],
                dtype=np.uint64,
            ),
            3,
        )


def test_duplicate_domain_cells_are_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [0, 1],
                dtype=np.uint64,
            ),
            3,
            domain=np.array(
                [0, 0],
                dtype=np.uint64,
            ),
        )


def test_mask_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="boolean",
    ):
        connected_components(
            np.array(
                [1, 0],
                dtype=np.int64,
            ),
            np.array(
                [0, 1],
                dtype=np.uint64,
            ),
            3,
        )


def test_mask_and_cell_ids_must_have_same_length():
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        connected_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0, 1],
                dtype=np.uint64,
            ),
            3,
        )


@pytest.mark.parametrize(
    "connectivity",
    [
        "4",
        "8",
        "bad",
        "",
        None,
    ],
)
def test_invalid_connectivity(connectivity):
    with pytest.raises(
        ValueError,
        match="connectivity",
    ):
        connected_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
            connectivity=connectivity,
        )


def test_empty_input():
    labels, n_components = (
        connected_components(
            np.array(
                [],
                dtype=bool,
            ),
            np.array(
                [],
                dtype=np.uint64,
            ),
            3,
        )
    )

    assert labels.shape == (0,)
    assert n_components == 0


def test_all_background():
    labels, n_components = (
        connected_components(
            np.array(
                [False, False, False],
                dtype=bool,
            ),
            np.array(
                [0, 1, 2],
                dtype=np.uint64,
            ),
            3,
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [0, 0, 0],
            dtype=np.int64,
        ),
    )

    assert n_components == 0


def test_threshold_zero_does_not_turn_background_on():
    cleaned = remove_small_components(
        np.array(
            [False, True],
            dtype=bool,
        ),
        np.array(
            [0, 1],
            dtype=np.uint64,
        ),
        3,
        min_cells=0,
    )

    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [False, True],
            dtype=bool,
        ),
    )


def test_torch_mask_returns_torch_labels():
    torch = pytest.importorskip(
        "torch"
    )

    level = 3

    first, second = _find_edge_pair(
        level
    )

    mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
    )

    cell_ids = torch.tensor(
        [first, second],
        dtype=torch.int64,
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
        )
    )

    assert isinstance(
        labels,
        torch.Tensor,
    )

    assert labels.dtype == torch.int64
    assert labels.device == mask.device

    torch.testing.assert_close(
        labels,
        torch.tensor(
            [1, 1],
            dtype=torch.int64,
        ),
    )

    assert n_components == 1


def test_torch_component_size():
    torch = pytest.importorskip(
        "torch"
    )

    labels = torch.tensor(
        [1, 1, 0, 2, 2, 2],
        dtype=torch.int64,
    )

    sizes = component_size(
        labels
    )

    assert isinstance(
        sizes,
        torch.Tensor,
    )

    torch.testing.assert_close(
        sizes,
        torch.tensor(
            [0, 2, 3],
            dtype=torch.int64,
        ),
    )


def test_torch_component_area():
    torch = pytest.importorskip(
        "torch"
    )

    level = 8

    labels = torch.tensor(
        [1, 1, 0, 2, 2, 2],
        dtype=torch.int64,
    )

    areas = component_area(
        labels,
        level,
        ellipsoid="WGS84",
    )

    assert isinstance(
        areas,
        torch.Tensor,
    )

    assert areas.dtype == torch.float64

    cell_area = healpix_cell_area(
        level,
        ellipsoid="WGS84",
    )

    expected = torch.tensor(
        [
            0.0,
            2.0 * cell_area,
            3.0 * cell_area,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        areas,
        expected,
    )


def test_torch_remove_small_components():
    torch = pytest.importorskip(
        "torch"
    )

    level = 3
    first, second = _find_edge_pair(
        level
    )

    mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
    )

    cell_ids = torch.tensor(
        [first, second],
        dtype=torch.int64,
    )

    cleaned = remove_small_components(
        mask,
        cell_ids,
        level,
        min_cells=2,
    )

    assert isinstance(
        cleaned,
        torch.Tensor,
    )

    assert cleaned.dtype == torch.bool
    assert cleaned.device == mask.device

    torch.testing.assert_close(
        cleaned,
        torch.tensor(
            [True, True],
            dtype=torch.bool,
        ),
    )

# ---------------------------------------------------------------------------
# Real HEALPix topology integration tests
# ---------------------------------------------------------------------------


def test_component_crosses_base_pixel_boundary():
    """Connectivity must work across HEALPix base-pixel boundaries."""

    level = 3

    first, second = (
        _find_base_pixel_crossing_edge_pair(
            level
        )
    )

    cells_per_base_pixel = (
        4**level
    )

    assert (
        first
        // cells_per_base_pixel
    ) != (
        second
        // cells_per_base_pixel
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 1],
            dtype=np.int64,
        ),
    )

    assert n_components == 1


def test_component_in_polar_region():
    """Connected components must not depend on Cartesian latitude geometry."""

    level = 3

    first, second, latitude = (
        _find_polar_edge_pair(
            level
        )
    )

    # Ensure this really exercises a high-latitude HEALPix region.
    assert abs(latitude) > 60.0

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 1],
            dtype=np.int64,
        ),
    )

    assert n_components == 1


def test_component_crosses_longitude_wrap():
    """Longitude coordinate discontinuity must not break topology."""

    level = 3

    (
        first,
        second,
        longitude_first,
        longitude_second,
    ) = _find_longitude_wrap_edge_pair(
        level
    )

    # Confirm that this pair crosses the numeric longitude seam.
    assert abs(
        longitude_first
        - longitude_second
    ) > 180.0

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 1],
            dtype=np.int64,
        ),
    )

    assert n_components == 1


def test_component_can_surround_background_hole():
    """A background hole must not split a surrounding component."""

    level = 3

    center, ring = (
        _find_complete_immediate_neighbourhood(
            level
        )
    )

    cell_ids = np.concatenate(
        [
            np.array(
                [center],
                dtype=np.uint64,
            ),
            ring,
        ]
    )

    # Centre is background; its complete immediate neighbourhood is active.
    mask = np.concatenate(
        [
            np.array(
                [False],
                dtype=bool,
            ),
            np.ones(
                ring.size,
                dtype=bool,
            ),
        ]
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            connectivity="edge_or_vertex",
        )
    )

    # The central hole remains background.
    assert labels[0] == 0

    # The surrounding cells form one connected component.
    assert np.all(
        labels[1:] == 1
    )

    assert n_components == 1


def test_component_touching_domain_boundary_is_retained():
    """A component may terminate naturally at the processing-domain edge."""

    level = 3

    chain = _find_edge_chain(
        length=3,
        level=level,
    )

    cell_ids = np.array(
        chain,
        dtype=np.uint64,
    )

    mask = np.array(
        [True, True, True],
        dtype=bool,
    )

    # The third active cell exists in the supplied data but is deliberately
    # outside the processing domain.
    domain = np.array(
        chain[:2],
        dtype=np.uint64,
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            connectivity="edge",
            domain=domain,
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 1],
            dtype=np.int64,
        ),
    )

    assert n_components == 1
@pytest.mark.parametrize(
    "level",
    [
        0,
        1,
        2,
        3,
        5,
        8,
    ],
)
def test_connected_components_multiple_refinement_levels(
    level,
):
    """Edge connectivity must be stable across HEALPix refinements."""

    first, second = _find_edge_pair(
        level
    )

    labels, n_components = (
        connected_components(
            np.array(
                [True, True],
                dtype=bool,
            ),
            np.array(
                [first, second],
                dtype=np.uint64,
            ),
            level,
            connectivity="edge",
        )
    )

    np.testing.assert_array_equal(
        labels,
        np.array(
            [1, 1],
            dtype=np.int64,
        ),
    )

    assert n_components == 1

def test_wgs84_total_surface_area_matches_reference_ellipsoid():
    """Validate cell area against the defining WGS84 ellipsoid constants."""

    # WGS84 defining constants.
    semi_major = 6_378_137.0
    inverse_flattening = 298.257223563

    flattening = (
        1.0
        / inverse_flattening
    )

    semi_minor = (
        semi_major
        * (1.0 - flattening)
    )

    eccentricity_squared = (
        1.0
        - (
            semi_minor
            * semi_minor
        )
        / (
            semi_major
            * semi_major
        )
    )

    eccentricity = math.sqrt(
        eccentricity_squared
    )

    expected_surface_area = (
        2.0
        * math.pi
        * semi_major
        * semi_major
        * (
            1.0
            + (
                (
                    1.0
                    - eccentricity_squared
                )
                / eccentricity
            )
            * math.atanh(
                eccentricity
            )
        )
    )

    actual_surface_area = (
        12.0
        * healpix_cell_area(
            0,
            ellipsoid="WGS84",
        )
    )

    assert actual_surface_area == pytest.approx(
        expected_surface_area,
        rel=1e-12,
    )

def test_min_cells_threshold_is_inclusive():
    """A component exactly equal to min_cells must be retained."""

    level = 3

    first, second = _find_edge_pair(
        level
    )

    cleaned = remove_small_components(
        np.array(
            [True, True],
            dtype=bool,
        ),
        np.array(
            [first, second],
            dtype=np.uint64,
        ),
        level,
        min_cells=2,
        connectivity="edge",
    )

    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [True, True],
            dtype=bool,
        ),
    )


def test_min_area_threshold_is_inclusive():
    """A component exactly equal to min_area_m2 must be retained."""

    level = 8

    first, second = _find_edge_pair(
        level
    )

    threshold = (
        2.0
        * healpix_cell_area(
            level,
            ellipsoid="WGS84",
        )
    )

    cleaned = remove_small_components(
        np.array(
            [True, True],
            dtype=bool,
        ),
        np.array(
            [first, second],
            dtype=np.uint64,
        ),
        level,
        min_area_m2=threshold,
        connectivity="edge",
        ellipsoid="WGS84",
    )

    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [True, True],
            dtype=bool,
        ),
    )

# ---------------------------------------------------------------------------
# Additional validation
# ---------------------------------------------------------------------------


def test_component_size_rejects_negative_labels():
    with pytest.raises(
        ValueError
    ):
        component_size(
            np.array(
                [0, 1, -1],
                dtype=np.int64,
            )
        )


def test_component_size_rejects_float_labels():
    with pytest.raises(
        TypeError
    ):
        component_size(
            np.array(
                [0.0, 1.0],
                dtype=np.float64,
            )
        )


def test_component_size_rejects_multidimensional_labels():
    with pytest.raises(
        ValueError
    ):
        component_size(
            np.array(
                [[0, 1]],
                dtype=np.int64,
            )
        )


@pytest.mark.parametrize(
    "level",
    [
        -1,
        30,
    ],
)
def test_component_area_rejects_invalid_refinement_level(
    level,
):
    with pytest.raises(
        ValueError
    ):
        component_area(
            np.array(
                [0, 1],
                dtype=np.int64,
            ),
            level,
        )


def test_healpix_cell_area_rejects_unknown_ellipsoid():
    with pytest.raises(
        ValueError
    ):
        healpix_cell_area(
            3,
            ellipsoid="NOT_AN_ELLIPSOID",
        )


@pytest.mark.parametrize(
    "min_cells",
    [
        -1,
        1.5,
        True,
    ],
)
def test_remove_small_components_rejects_invalid_min_cells(
    min_cells,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        remove_small_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
            min_cells=min_cells,
        )


@pytest.mark.parametrize(
    "min_area_m2",
    [
        -1.0,
        np.nan,
        np.inf,
        True,
    ],
)
def test_remove_small_components_rejects_invalid_min_area(
    min_area_m2,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        remove_small_components(
            np.array(
                [True],
                dtype=bool,
            ),
            np.array(
                [0],
                dtype=np.uint64,
            ),
            3,
            min_area_m2=min_area_m2,
        )

# ---------------------------------------------------------------------------
# Torch device round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_connected_components_round_trip():
    import torch

    level = 3
    first, second = _find_edge_pair(
        level
    )

    mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
        device="cuda",
    )

    cell_ids = torch.tensor(
        [first, second],
        dtype=torch.int64,
        device="cuda",
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            connectivity="edge",
        )
    )

    assert labels.device.type == "cuda"

    torch.testing.assert_close(
        labels.cpu(),
        torch.tensor(
            [1, 1],
            dtype=torch.int64,
        ),
    )

    assert n_components == 1


@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="MPS is not available",
)
def test_mps_connected_components_round_trip():
    import torch

    level = 3
    first, second = _find_edge_pair(
        level
    )

    mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
        device="mps",
    )

    cell_ids = torch.tensor(
        [first, second],
        dtype=torch.int64,
        device="mps",
    )

    labels, n_components = (
        connected_components(
            mask,
            cell_ids,
            level,
            connectivity="edge",
        )
    )

    assert labels.device.type == "mps"

    torch.testing.assert_close(
        labels.cpu(),
        torch.tensor(
            [1, 1],
            dtype=torch.int64,
        ),
    )

    assert n_components == 1


