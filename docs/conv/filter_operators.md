# Filter operators and the land barrier

Build the weights of a metric filter once and apply them many times, keep the
total of a field (`normalization="source"`), or stop the kernel at a coast
(`blocked_cells`). The filters themselves are described in
{doc}`/conv/radial_filter`.

## Reusable weights

The weights of a filter can be built once and applied many times, for example
to every time step of a model, or chunk by chunk:

```python
from healpix_analyse import apply_filter_operator, gaussian_filter_operator

op = gaussian_filter_operator(
    cell_ids,
    refinement_level=10,
    sigma_m=6000.0,
    normalization="full_disc",
)
row_offsets, column_indices, weights, n_columns = op   # three arrays and N

out = apply_filter_operator(values, *op)         # values: (time, N) or (N,)
```

`radial_filter_operator` does the same for a kernel. The result is a
`FilterOperator`, a named tuple of three arrays in compressed-row form and the
number of columns `n_columns` (`len(cell_ids)`):

```text
out[..., i] = sum_k weights[k] * values[..., column_indices[k]]
              for k in row_offsets[i]:row_offsets[i + 1]
```

Row `i` is the output cell `domain[i]` and the column indices point into
`cell_ids`. The weights already contain the normalisation (an operator is always
normalised: there is no `normalize=False`), and
`apply_filter_operator` gives the same values as the filter with the same
options (NumPy and Torch input; leading dimensions are kept). No SciPy or
`sparse` import is needed. A caller that has one of them can wrap the arrays:

```python
import scipy.sparse as sp

matrix = sp.csr_matrix(
    (op.weights, op.column_indices, op.row_offsets),
    shape=(len(op.row_offsets) - 1, op.n_columns),   # (len(domain), len(cell_ids))
)
out = matrix @ values
```

`apply_filter_operator(values, *op)` checks that the final axis of `values` has
length `n_columns`; with the three-array call the length is not checked.

`normalization="full_disc"` and `width_rule="matched"` rebuild their weights at
every `gaussian_filter` / `radial_filter` call (only the geometry is cached), so
in a loop over time steps build the operator once.

`apply_filter_operator` leaves `NaN` values out of the sum, so a `NaN` counts as
zero, which is the rule of `"full_disc"`; the equality with the filter for an
operator built with `"present"` holds for data without `NaN`. With `renormalize_missing=True` the weights of the remaining cells are
divided by their sum, which is what the filters do with
`normalization="present"`.

xarray and dask wrappers are left to the calling package.

## Land barrier

`gaussian_filter_operator` and `radial_filter_operator` take `blocked_cells`,
an array of HEALPix cell ids (land). Without it the kernel acts on the
straight-line distance, so a kernel that is wider than a strip of land puts
weight on the far side of it, and land cells only act as zero padding. With it:

- a blocked cell **receives nothing and gives nothing**: its row is empty (the
  output there is zero) and it appears in no row (its input value is ignored); the output there is zero
  also with `renormalize_missing=True`;
- a cell reaches another one only along a path of neighbouring non-blocked
  cells, and the kernel sees the length of the **shortest such path**, not the
  straight distance. A strait lets weight through at the water distance, an
  unbroken coast lets none through, and an island is gone around.

The blocked cells must be part of the domain (`cell_ids`, or `domain`), as land
cells are when a land mask is applied to a field. Cells that are not in the
domain are never crossed, but they do not start the search by themselves: a
coast made of absent cells does not stop weight that jumps over it. Ids in
`blocked_cells` that are not in the domain are ignored.

```python
op = gaussian_filter_operator(
    cell_ids, level, sigma_m=2 * cell, width_rule="matched",
    normalization="source",            # or "full_disc", see below
    blocked_cells=cell_ids[land_mask],
)
state = apply_filter_operator(state, *op)
```

### Distance through water

The graph has the cells of the domain as nodes and the eight immediate
neighbours as edges, with the geodesic distance between the centres as length.
Of the eight neighbours of a HEALPix cell, four share an edge and four only a
vertex. A vertex edge is removed when both cells that touch that vertex between
the two ends are blocked (or outside the domain), so that a diagonal chain of
land cells, which is what a coast along a line of constant latitude looks like
on this grid, does not leak. The same rule applies to the path with every cell
passable (`D0` below), so a hole in the domain, whether the two cells meet at
an edge or at a vertex, is no barrier as long as the detour around it stays inside the candidate
disc of 1.5 radii around the source: it lengthens `D` and `D0` alike and the
correction cancels it. A longer detour leaves the disc and the pair is dropped,
so a long hole then acts as a barrier. Only blocked cells are land.

The path on such a lattice is longer than the straight geodesic. Measured over
the whole sphere (levels 3 to 7, `benchmarks/land_barrier_lattice.json`, no
land): at most **1.10** within 30 degrees of the equator and up to **1.39** in
the polar caps and just below them (above about 30 degrees), where the cells
are more distorted; the mean is 1.04. A check at levels 8 to 10 found up to
1.42 near 41 to 45 degrees of latitude, and the ratio grows slowly with the
level, so the search factor is 1.5 (`tests/test_land_barrier.py` guards the
margin). The metrication error is therefore not
the 8 % of a square lattice. It is removed instead of being tolerated: the same
pair of cells is searched twice, with the blocked cells removed (`D`) and with
every cell passable (`D0`), and the kernel sees

    d = straight * D / D0

In open water `D = D0` and `d` is the straight distance. Near land `d` is the
detour, with the lattice's own distortion divided out. `D >= D0`, so `d` is never
shorter than the straight distance. Consequences:

- A row with no blocked cell within `1.5 * truncate * sigma_m` (1.5 bounds
  `D0 / straight`, with margin) takes a fast path, is built as without the
  barrier, and is equal to it (tested at 1e-12). With no blocked cell at all,
  or an empty `blocked_cells`, the result is identical to the operator without
  the option.
- The search is restricted to the cells within `1.5 * truncate * sigma_m` of the
  row (a path that must leave this disc is not found, see the hole note above) and runs, vectorised, for the rows that have a blocked cell inside that
  disc only. The cost therefore grows with the length of the coast, not with
  the size of the domain; build time and peak memory are given below. The
  geometry is built at the 1.5 times larger radius whenever `blocked_cells` is
  given.
- **The water distance is a lattice path distance, corrected in open water; it is
not a geodesic around the coast.** For example, the detour around the tip of a
wall depends on the orientation of the wall relative to the lattice (measured
7.9 to 11.3 cells for the same wall at different angles). Near land the distance is the lattice detour divided by the lattice's own
factor for the unblocked pair. It equals the straight distance only where land
does not change the shortest lattice path; where it does, the remaining error
is the difference of the metrication of the two paths, which is not bounded
here. The tests compare the operator with an unbounded Dijkstra on the same
graph (to 1e-10), not with true geodesics around a coast.

### What happens to the mass that would go to land or behind it

The option `normalization` decides. Two operators with the same `blocked_cells`
differ only in it:

| `normalization` | mass that cannot be reached | rows | columns | constant field |
|---|---|---|---|---|
| `"full_disc"` | **lost** (zero padding: rows are divided by the open-water disc, so a likelihood built on it penalises coasts) | sum to the reachable fraction of the open-water disc, below 1 near land | sum to at most about 1 | decays next to land |
| `"source"` | **kept** (reflecting coast): every source cell divides its outgoing weights by their sum over the cells it reaches | free | sum to exactly 1 (to rounding): the total is conserved, and never above 1 for a non-negative signal | not preserved (rises or falls next to land) |
| `"present"` | renormalised per output cell over the cells that reach it | sum to 1 | free | preserved on water |

`"source"` is new and works without `blocked_cells` too, where it makes the
domain edge reflecting. In the interior it differs from `"present"` only
through the cell-to-cell variation (about 1e-4) of the kernel sum. With
`width_rule="matched"` the blend with the identity below about 0.6 cell is
applied per source cell, so that the columns still sum to 1. `gaussian_filter`
and `radial_filter` do not take `"source"` or `blocked_cells`: they are
operator options.

For `radial_filter_operator` the kernel is called once on the distances
through water, and once more on the straight distances of the whole disc when
`normalization="full_disc"` (to get the open-water normalisation).

### Measured

Synthetic coast on the 8125-cell level-10 domain of the small-sigma check
(`benchmarks/land_barrier_experiment.py`, results and figure next to it): a
strip of two cells across the whole domain with and without a four-cell strait,
and an island of radius four cells; delta 3 cells from the strip, `"matched"`.
Mass on the far side of the unbroken strip (operator without barrier, with land masked
after every step, against the barrier):

| sigma (cells) | after 1 step, no barrier | after 40 steps, no barrier | barrier |
|---|---|---|---|
| 0.7 | 0 | 0.0009 | 0 |
| 1 | 0 | 0.0062 | 0 |
| 2 | 0.0037 | 0.032 | 0 |
| 4 | 0.094 | 0.060 | 0 |

Total mass after 1 and 40 steps for sigma 2 cells (no barrier / `"full_disc"` /
`"source"`): 0.946 / 0.930 / 1.000 and 0.283 / 0.241 / 1.000; for sigma 4
cells: 0.891 / 0.757 / 1.000 and 0.178 / 0.085 / 1.000. Mass lost in one step by
a delta at 1, 3 and 6 cells from the strip (sigma 2 cells): no barrier 0.128, 0.054,
0.0005; `"full_disc"` 0.188, 0.070, 0.0005; `"source"` 0 (4e-16).

Time to build the operator (caches cleared, 8125 cells, strip of 192 land
cells, `"full_disc"`): sigma 1 cell 0.42 s (no barrier) against 0.56 s, peak memory
63 against 111 MiB; sigma 2 cells 1.4 s against 2.5 s, 194 against 366 MiB (the
memory is that of the geometry at 1.5 times the radius).

Real land mask of a fish-tracking demo domain (level 10,
39 388 cells of which 29.6 % are land, read from disk, no download; sigma 1 and
2 cells, `"matched"`, `"full_disc"`). Weight that the no barrier operator puts on
water cells the barrier does not connect, per source cell, one step, by
distance of the source to the nearest land cell:

| distance to land | sources | sigma 1: mean / max | sigma 2: mean / max |
|---|---|---|---|
| up to 1 cell | 2089 | 0.0025 / 0.28 | 0.0051 / 0.50 |
| 1 to 3 cells | 3441 | 0.0001 / 0.012 | 0.0008 / 0.082 |
| 3 to 6 cells | 4172 | 0 / 0 | 0 / 0.004 |

The maxima are sources next to a thin or narrow piece of coast (a spit, an
inlet), where up to half of the weight sits on cells that are close in a
straight line and far through water. Mean mass lost per step by sources up to
1 cell from land, no barrier / `"full_disc"` barrier: 0.287 / 0.292 (sigma 1) and
0.382 / 0.399 (sigma 2); the columns of `"source"` sum to 1 (largest deviation
0, to rounding). Build time: 1.8 s against 2.8 s (sigma 1) and 7.6 s against
15.7 s (sigma 2), 1.98 M and 7.62 M non-zeros for no barrier against 1.34 M and 4.97 M
with the barrier (blocked cells carry no weight). Sigma 4 cells was not run on
this domain (the cost grows with the square of sigma).

### Limits

- Sigma that is large compared with the domain or with the width of a channel:
  the disc then holds much of the domain, and the search and the geometry (at
  1.5 times the radius) are as large as the dense operator.
- Cost near long coasts: every row within `1.5 * truncate * sigma_m` of land is
  searched; a coast that runs through the whole domain makes this every row.
  Measured on the 8125-cell level-10 domain with a long coast (1596 land
  cells), `"source"` with `"matched"`, barrier against none: sigma 1 cell 0.61
  against 0.42 s and 585 against 438 MiB; sigma 2 cells 3.6 against 1.3 s and
  983 against 532 MiB; sigma 4 cells 25.7 against 5.2 s and 2730 against
  1352 MiB.
- The lattice metrication is divided out, but not for pairs whose shortest
  lattice path changes because of land (see above).
- Real coasts were tried on one demo domain only; GPU and dask were not tried.
