import numpy as np
import pytest

import healpix_analyse._topology as topology
from healpix_analyse._topology import (
    nested_edge_neighbours,
    nested_neighbours,
)


@pytest.mark.parametrize("level", [0, 1, 3, 6])
def test_edge_neighbours_shape(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(min(npix, 32), dtype=np.uint64)

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    assert neighbours.shape == (cells.size, 4)
    assert np.issubdtype(
        neighbours.dtype,
        np.signedinteger,
    )
    assert neighbours.dtype == np.int64
    assert neighbours.flags.c_contiguous


@pytest.mark.parametrize("level", [0, 1, 3, 6])
def test_full_neighbours_shape(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(min(npix, 32), dtype=np.uint64)

    neighbours = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    assert neighbours.shape == (cells.size, 8)
    assert neighbours.dtype == np.int64
    assert neighbours.flags.c_contiguous


@pytest.mark.parametrize(
    ("connectivity", "width"),
    [
        ("edge", 4),
        ("edge_or_vertex", 8),
    ],
)
def test_adapter_delegates_to_healpix_geo(
    monkeypatch,
    connectivity,
    width,
):
    calls = []

    def fake_neighbours(
        cells,
        depth,
        *,
        connectivity,
        num_threads,
    ):
        calls.append(
            (
                cells.copy(),
                depth,
                connectivity,
                num_threads,
            )
        )

        return np.arange(
            cells.size * width,
            dtype=np.int64,
        ).reshape(cells.size, width)

    monkeypatch.setattr(
        topology.healpix_geo_nested,
        "neighbours",
        fake_neighbours,
    )

    cells = np.array(
        [0, 1, 2],
        dtype=np.uint64,
    )

    result = nested_neighbours(
        cells,
        2,
        connectivity=connectivity,
    )

    assert result.shape == (3, width)
    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0][0],
        cells,
    )
    assert calls[0][1:] == (
        2,
        "all" if connectivity == "edge_or_vertex" else "edge",
        0,
    )


def test_adapter_rejects_unexpected_backend_shape(
    monkeypatch,
):
    monkeypatch.setattr(
        topology.healpix_geo_nested,
        "neighbours",
        lambda *args, **kwargs: np.empty(
            (8, 2),
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected shape",
    ):
        nested_neighbours(
            np.array(
                [0, 1],
                dtype=np.uint64,
            ),
            2,
        )


@pytest.mark.parametrize("level", [0, 1, 3, 6])
def test_edge_neighbours_always_exist(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    # A HEALPix cell is a quadrilateral and always has four
    # edge-sharing neighbours.
    assert np.all(neighbours >= 0)
    assert np.all(neighbours < npix)


@pytest.mark.parametrize("level", [0, 1, 3])
def test_each_cell_has_four_distinct_edge_neighbours(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    for row in neighbours:
        assert len(set(map(int, row))) == 4


@pytest.mark.parametrize("level", [0, 1, 3])
def test_edge_neighbour_relation_is_symmetric(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    adjacency = {
        int(cell): set(map(int, row))
        for cell, row in zip(
            cells,
            neighbours,
            strict=True,
        )
    }

    for cell, row in adjacency.items():
        for neighbour in row:
            assert cell in adjacency[neighbour]


@pytest.mark.parametrize("level", [0, 1, 3])
def test_edge_is_subset_of_edge_or_vertex(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

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

    for edge_row, full_row in zip(
        edge,
        full,
        strict=True,
    ):
        full_set = {
            int(value)
            for value in full_row
            if value >= 0
        }

        assert set(map(int, edge_row)).issubset(
            full_set
        )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_full_neighbourhood_has_seven_or_eight_neighbours(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

    neighbours = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    counts = np.sum(
        neighbours >= 0,
        axis=1,
    )

    assert np.all(
        (counts == 7) | (counts == 8)
    )


@pytest.mark.parametrize("level", [1, 2, 3])
def test_some_cells_have_seven_full_neighbours(level):
    npix = 12 * (2**level) ** 2
    cells = np.arange(npix, dtype=np.uint64)

    neighbours = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    counts = np.sum(
        neighbours >= 0,
        axis=1,
    )

    assert np.any(counts == 7)


# ---------------------------------------------------------------------------
# Hard-coded reference neighbours (no healpy needed)
# ---------------------------------------------------------------------------
#
# Order SW, W, NW, N, NE, E, SE, S; -1 marks a missing direction.  The
# values are the NESTED neighbours of healpy.get_all_neighbours(..., nest=True)
# (transposed), frozen here so that the test runs without healpy.

# Level 0: the twelve base cells.  Each has six neighbours: the polar base
# cells (0-3, 8-11) miss W and E, the equatorial ones (4-7) miss NW and S.
LEVEL0_NEIGHBOURS = np.array(
    [
        [4, -1, 3, 2, 1, -1, 5, 8],
        [5, -1, 0, 3, 2, -1, 6, 9],
        [6, -1, 1, 0, 3, -1, 7, 10],
        [7, -1, 2, 1, 0, -1, 4, 11],
        [11, 7, 3, -1, 0, 5, 8, -1],
        [8, 4, 0, -1, 1, 6, 9, -1],
        [9, 5, 1, -1, 2, 7, 10, -1],
        [10, 6, 2, -1, 3, 4, 11, -1],
        [11, -1, 4, 0, 5, -1, 9, 10],
        [8, -1, 5, 1, 6, -1, 10, 11],
        [9, -1, 6, 2, 7, -1, 11, 8],
        [10, -1, 7, 3, 4, -1, 8, 9],
    ],
    dtype=np.int64,
)

# Level 1: interior cells (8 neighbours) and base-cell corners where one
# direction is missing (7 neighbours).
LEVEL1_NEIGHBOURS = {
    0: [17, 19, 2, 3, 1, 23, 22, 35],
    3: [2, 13, 15, 11, 7, 6, 1, 0],
    5: [4, 6, 7, 11, 10, -1, 27, 26],
    10: [27, -1, 5, 7, 11, 9, 8, 25],
    15: [14, 9, 11, 7, 3, 2, 13, 12],
    16: [45, 47, 18, 19, 17, 35, 34, -1],
    19: [18, 12, 13, -1, 2, 0, 17, 16],
    21: [20, 22, 23, 6, 4, 26, 39, 38],
    27: [26, 4, 5, -1, 10, 8, 25, 24],
    32: [44, 45, 34, 35, 33, 38, 36, 40],
    37: [36, 38, 39, 26, 24, -1, 42, 40],
    44: [40, 41, 46, 47, 45, 34, 32, 36],
    47: [46, 28, 29, 12, 18, 16, 45, 44],
}

# Level 1 cells with exactly seven neighbours: the two side corners
# (W and E vertices) of every base cell.
LEVEL1_SEVEN_NEIGHBOUR_CELLS = np.array(
    [1, 2, 5, 6, 9, 10, 13, 14, 16, 19, 20, 23,
     24, 27, 28, 31, 33, 34, 37, 38, 41, 42, 45, 46],
)


def test_level0_neighbours_hard_coded():
    actual = nested_neighbours(
        np.arange(12),
        0,
        connectivity="edge_or_vertex",
    )

    np.testing.assert_array_equal(actual, LEVEL0_NEIGHBOURS)
    np.testing.assert_array_equal(
        nested_edge_neighbours(np.arange(12), 0),
        LEVEL0_NEIGHBOURS[:, [0, 2, 4, 6]],
    )


def test_level1_neighbours_hard_coded():
    cells = np.array(sorted(LEVEL1_NEIGHBOURS))
    expected = np.array(
        [LEVEL1_NEIGHBOURS[c] for c in cells],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        nested_neighbours(cells, 1, connectivity="edge_or_vertex"),
        expected,
    )
    np.testing.assert_array_equal(
        nested_edge_neighbours(cells, 1),
        expected[:, [0, 2, 4, 6]],
    )


def test_level1_seven_neighbour_cells_hard_coded():
    neighbours = nested_neighbours(
        np.arange(48),
        1,
        connectivity="edge_or_vertex",
    )
    counts = np.sum(neighbours >= 0, axis=1)

    np.testing.assert_array_equal(
        np.nonzero(counts == 7)[0],
        LEVEL1_SEVEN_NEIGHBOUR_CELLS,
    )
    assert np.all(counts[counts != 7] == 8)
    # Missing positions are vertex (W, N, E, S) directions only.
    assert np.all(neighbours[:, [0, 2, 4, 6]] >= 0)


def test_edge_selection_matches_healpy_documented_order():
    """Compare the healpix-geo edge directions with the healpy reference."""

    hp = pytest.importorskip("healpy")

    level = 3
    nside = 2**level
    npix = 12 * nside**2

    cells = np.arange(
        npix,
        dtype=np.int64,
    )

    # healpy order:
    #
    # SW, W, NW, N, NE, E, SE, S
    all_neighbours = hp.get_all_neighbours(
        nside,
        cells,
        nest=True,
    )

    expected = all_neighbours[
        [0, 2, 4, 6]
    ].T

    actual = nested_edge_neighbours(
        cells,
        level,
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_full_neighbour_selection_matches_healpy():
    """Ensure the private wrapper preserves healpy direction ordering."""

    hp = pytest.importorskip("healpy")

    level = 3
    nside = 2**level
    npix = 12 * nside**2

    cells = np.arange(
        npix,
        dtype=np.int64,
    )

    expected = hp.get_all_neighbours(
        nside,
        cells,
        nest=True,
    ).T

    actual = nested_neighbours(
        cells,
        level,
        connectivity="edge_or_vertex",
    )

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_scalar_like_cell_input():
    neighbours = nested_edge_neighbours(
        0,
        2,
    )

    assert neighbours.shape == (1, 4)


def test_single_element_array():
    neighbours = nested_edge_neighbours(
        np.array([0], dtype=np.uint64),
        2,
    )

    assert neighbours.shape == (1, 4)


def test_empty_input_edge():
    neighbours = nested_neighbours(
        np.array([], dtype=np.uint64),
        3,
        connectivity="edge",
    )

    assert neighbours.shape == (0, 4)
    assert neighbours.dtype == np.int64


def test_empty_input_edge_or_vertex():
    neighbours = nested_neighbours(
        np.array([], dtype=np.uint64),
        3,
        connectivity="edge_or_vertex",
    )

    assert neighbours.shape == (0, 8)
    assert neighbours.dtype == np.int64


@pytest.mark.parametrize(
    "connectivity",
    [
        "bad",
        "4",
        "8",
        "",
        None,
    ],
)
def test_invalid_connectivity(connectivity):
    with pytest.raises(
        ValueError,
        match="connectivity",
    ):
        nested_neighbours(
            np.array(
                [0],
                dtype=np.uint64,
            ),
            1,
            connectivity=connectivity,
        )


@pytest.mark.parametrize(
    "level",
    [-1, 30],
)
def test_invalid_refinement_level(level):
    with pytest.raises(
        ValueError,
        match="refinement_level",
    ):
        nested_edge_neighbours(
            np.array(
                [0],
                dtype=np.uint64,
            ),
            level,
        )


@pytest.mark.parametrize(
    "level",
    [
        1.0,
        "1",
        None,
        True,
    ],
)
def test_refinement_level_must_be_integer(level):
    with pytest.raises(
        TypeError,
        match="refinement_level",
    ):
        nested_edge_neighbours(
            np.array(
                [0],
                dtype=np.uint64,
            ),
            level,
        )


def test_cell_ids_must_be_one_dimensional():
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        nested_edge_neighbours(
            np.array(
                [[0, 1]],
                dtype=np.uint64,
            ),
            2,
        )


def test_cell_ids_must_be_integer():
    with pytest.raises(
        TypeError,
        match="integers",
    ):
        nested_edge_neighbours(
            np.array(
                [0.0, 1.0],
                dtype=np.float64,
            ),
            2,
        )


def test_negative_cell_id():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        nested_edge_neighbours(
            np.array(
                [-1],
                dtype=np.int64,
            ),
            2,
        )


def test_cell_id_above_valid_range():
    level = 2
    npix = 12 * (2**level) ** 2

    with pytest.raises(
        ValueError,
        match="valid range",
    ):
        nested_edge_neighbours(
            np.array(
                [npix],
                dtype=np.uint64,
            ),
            level,
        )


@pytest.mark.parametrize("level", [0, 1, 2, 4])
def test_all_twelve_base_pixels_are_represented(level):
    """Exercise cells belonging to every HEALPix base pixel."""

    cells_per_base_pixel = 4**level

    cells = np.array(
        [
            base_pixel * cells_per_base_pixel
            for base_pixel in range(12)
        ],
        dtype=np.uint64,
    )

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    assert neighbours.shape == (12, 4)
    assert np.all(neighbours >= 0)


@pytest.mark.parametrize("level", [2, 4])
def test_random_cells_have_valid_edge_neighbours(level):
    rng = np.random.default_rng(42)

    npix = 12 * (2**level) ** 2

    cells = rng.integers(
        0,
        npix,
        size=min(npix, 1000),
        dtype=np.int64,
    )

    neighbours = nested_edge_neighbours(
        cells,
        level,
    )

    assert np.all(neighbours >= 0)
    assert np.all(neighbours < npix)
