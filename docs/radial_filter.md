# Metric radial filtering on HEALPix

`radial_filter` applies a user-defined isotropic kernel to values on a NESTED HEALPix grid using **physical WGS84 centre-to-centre distance in metres**.

A convenience `gaussian_filter` wrapper provides normalized Gaussian smoothing using `sigma_m` rather than a pixel-based standard deviation.

Candidate cells and cell-centre coordinates come from healpix-geo (one batched
`healpix_geo.nested.cone_coverage` call for all centres); the distance cutoff
and the kernel weights use WGS84 geodesic distances from `pyproj.Geod`.

The central spatial contract is:

```text
weight = kernel(distance_m)
```

The operation therefore does not depend on Cartesian concepts such as:

```text
3x3
5x5
row / column
sigma_pixels
```

The same `radius_m` or `sigma_m` keeps the same physical meaning when the HEALPix refinement level changes.

## Basic radial filter

```python
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
```

For every output cell, HEALPix neighbours are selected by physical centre-to-centre distance and the kernel is evaluated using those WGS84 distances.

Conceptually:

```text
output cell
    |
    v
physical-radius HEALPix neighbourhood
    |
    v
WGS84 centre-to-centre distance
    |
    v
kernel(distance_m)
    |
    v
weighted aggregation
```

## Gaussian smoothing

For normalized Gaussian smoothing:

```python
from healpix_analyse import gaussian_filter

smoothed = gaussian_filter(
    values,
    cell_ids,
    refinement_level=17,
    sigma_m=240.0,
    truncate=4.0,
)
```

The Gaussian kernel is

```text
weight(d) = exp(-0.5 * (d / sigma_m)^2)
```

and its finite support is

```text
radius_m = truncate * sigma_m
```

With the default `truncate=4.0`, `sigma_m=240.0` therefore uses a physical support radius of:

```text
960 m
```

With the default `normalization="present"` the Gaussian wrapper is normalized by construction, so a constant finite field remains constant after filtering (with `"full_disc"` it decays near the edge of the domain, see below).

```{warning}
The default Gaussian is sampled at the cell centres. When `sigma_m` is below
about one cell, the response of the filter to a single cell is narrower than
`sigma_m`: at 0.3 cell the effective width (one-axis RMS) is about 0.11 cell
and the centre cell keeps 96 to 97 % of the weight, and below about 0.2 cell the
filter is the identity. Use `width_rule="matched"` (see
[Widths below one cell](#widths-below-one-cell)) when the width itself matters.
```

## Physical-distance semantics

`radius_m` and `sigma_m` are expressed in metres.

For example:

```python
gaussian_filter(
    values,
    cell_ids,
    refinement_level=15,
    sigma_m=500.0,
)
```

and:

```python
gaussian_filter(
    values,
    cell_ids,
    refinement_level=18,
    sigma_m=500.0,
)
```

both describe a Gaussian smoothing scale of 500 metres.

The number and geometry of contributing HEALPix cells can differ between refinement levels, but the public spatial scale does not change.

This is intentionally different from a Cartesian API such as:

```text
sigma_pixels = 3
```

whose physical meaning depends on raster resolution.

## Processing domain

`cell_ids` and `domain` have different roles.

### `cell_ids`

`cell_ids` identifies every HEALPix cell for which an input value is available.

### `domain`

`domain` identifies:

- the cells that participate in the filter, and
- the output cells and their ordering.

If `domain` is omitted:

```text
domain = cell_ids
```

If it is supplied:

```text
domain subset_of cell_ids
```

must hold.

For example:

```python
cell_ids = np.array([10, 11, 12], dtype=np.uint64)
values = np.array([2.0, 4.0, 1000.0])
domain = np.array([10, 11], dtype=np.uint64)
```

Even if cell `12` is physically inside the requested radius around cell `11`, it does not participate because it lies outside `domain`.

The effective spatial operation is therefore:

```text
geometric neighbourhood
        intersection
processing domain
```

Cells outside `domain` are not interpreted as:

- zero,
- `NaN`,
- padding,
- periodic continuation.

They are simply absent from the neighbourhood before weighted aggregation.

This prevents artificial values from being introduced at regional dataset boundaries.

## Normalization

`radial_filter` supports both normalized and unnormalized weighted operations.

### `normalize=True`

The default is:

```text
         sum(w_i * f_i)
output = ----------------
             sum(w_i)
```

using only effective valid samples.

This is the natural form for smoothing and weighted averaging.

### `normalize=False`

With:

```python
normalize=False
```

`radial_filter` returns the raw weighted sum:

```text
output = sum(w_i * f_i)
```

This is useful for general convolution-like radial kernels whose weights are not intended to form an average.

## Missing values

`NaN` signal samples are treated as unavailable observations.

Their contributions and weights are removed from the effective aggregation.

For normalized filtering, suppose the geometrical weights are:

```text
[0.2, 0.5, 0.3]
```

but the second signal value is `NaN`.

The effective normalized calculation uses only:

```text
[0.2, 0.3]
```

and renormalizes those remaining weights.

A single missing neighbour therefore does not contaminate the complete neighbourhood with `NaN`.

If the effective weight sum is zero under `normalize=True`, the result is `NaN`.

## Fixed normalisation at the edge of the domain

`normalization` chooses what the weights are divided by, in `radial_filter`
and `gaussian_filter`:

| value | divided by | at the edge of the domain |
|---|---|---|
| `"present"` (default, same as `normalize=True`) | sum of the weights of the cells present | a constant field stays constant; no mass leaves |
| `"full_disc"` | sum of the weights over the whole kernel disc, cells outside the domain included | cells outside the domain contribute zero; mass that diffuses out is lost |

`"full_disc"` is the `mode="constant", cval=0` of a raster filter. The row of
a cell whose disc crosses the edge sums to the fraction of the disc weight
that lies inside the domain, and `filter(ones)` shows that fraction.

```python
out = gaussian_filter(
    probability,
    cell_ids,
    refinement_level=10,
    sigma_m=6000.0,
    normalization="full_disc",
)
```

Cells of `cell_ids` that are not in `domain` count as outside the domain. The
kernel of `radial_filter` is also evaluated on the distances to the cells
outside the domain. A `ValueError` is raised if the weights of some disc sum to
zero. Passing `normalization` with `normalize=False` is an error.

**Missing values with `"full_disc"`.** A `NaN` inside the domain is left out of
the numerator and kept in the denominator: it counts as zero, and the result is
never `NaN` because of `NaN` input. With `"present"` the weights of the
remaining cells are renormalised, as described under *Missing values*.

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

## Widths below one cell

With `width_rule="sampled"` (default) the Gaussian is sampled at the cell
centres and a width below about one cell is not reached. With
`width_rule="matched"`:

- where `sigma_m` is above the effective width of a reference operator `G0`
  (the sampled Gaussian with a width of 0.6 cell, whose effective width is
  about 0.58 cell), the shape parameter `s` of
  `w = exp(-d^2 / (2 s^2))` is solved per cell so that the discrete second
  moment `sum(w d^2) / sum(w)` over the disc equals `2 sigma_m^2`;
- below it, the cell uses `T = I + (sigma_m / sigma_0)^2 (G0 - I)`, with
  `sigma_0` the effective width of `G0` at that cell.

The effective width of the row of every cell whose disc lies inside the domain
is then equal to `sigma_m`; rows at the edge are renormalised (`"present"`) or
truncated (`"full_disc"`), and their width differs. The
response to one cell is within a few per cent of it (one-axis RMS, the
anisotropy of the response is about 1.06), the width is continuous and monotone
in `sigma_m`, and the filter tends to the identity when `sigma_m` tends to
zero. The support is `truncate * max(sigma_m, 0.6 cell)`, and `truncate` must be at least 3 (a `ValueError` otherwise). The option works with
both normalisations and with `gaussian_filter_operator`. Rows of the matched operator sum to 1 away from the edge; its column sums,
which carry the conservation of the total, are close to 1 but not exactly 1.
Measured for `"full_disc"` over the columns whose whole disc lies inside the
domain: within 1.1e-4 of 1 on the 8125-cell level-10 domain above (0.3, 1 and 2
cells), and within 1.2e-3 on regular patches of about 7900 cells at levels 8 and
10 (worst at 0.3 cell). The tests require 2e-3 on both levels. It is not meant for
widths of several cells, where the sampled Gaussian is already accurate and the
solve costs extra time.

Measured for one real domain at level 10 (8125 cells), response to one cell, in
cells:

| requested | `"sampled"` | `"matched"` |
|---|---|---|
| 0.1 | 0.00 | 0.10 |
| 0.3 | 0.11 | 0.30 |
| 0.5 | 0.45 | 0.50 |
| 1.0 | 1.00 | 1.00 |
| 2.0 | 2.00 | 2.01 |

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
cells with a mask are in the pangeo-fish estimator. Cells that are not in the
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
an edge or at a vertex, is never a barrier: it lengthens `D` and `D0` alike and
the correction cancels it. Only blocked cells are land.

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
- The search is closed on the cells within `1.5 * truncate * sigma_m` of the
  row and runs, vectorised, for the rows that have a blocked cell inside that
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
| `"full_disc"` | **lost** (zero padding, as today in pangeo-fish: rows are divided by the open-water disc, so the likelihood penalises coasts) | sum to the reachable fraction of the open-water disc, below 1 near land | sum to at most about 1 | decays next to land |
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
Mass on the far side of the unbroken strip (PR 81 operator with land masked
after every step, against the barrier):

| sigma (cells) | after 1 step, PR 81 | after 40 steps, PR 81 | barrier |
|---|---|---|---|
| 0.7 | 0 | 0.0009 | 0 |
| 1 | 0 | 0.0062 | 0 |
| 2 | 0.0037 | 0.032 | 0 |
| 4 | 0.094 | 0.060 | 0 |

Total mass after 1 and 40 steps for sigma 2 cells (PR 81 / `"full_disc"` /
`"source"`): 0.946 / 0.930 / 1.000 and 0.283 / 0.241 / 1.000; for sigma 4
cells: 0.891 / 0.757 / 1.000 and 0.178 / 0.085 / 1.000. Mass lost in one step by
a delta at 1, 3 and 6 cells from the strip (sigma 2 cells): PR 81 0.128, 0.054,
0.0005; `"full_disc"` 0.188, 0.070, 0.0005; `"source"` 0 (4e-16).

Time to build the operator (caches cleared, 8125 cells, strip of 192 land
cells, `"full_disc"`): sigma 1 cell 0.42 s (PR 81) against 0.56 s, peak memory
63 against 111 MiB; sigma 2 cells 1.4 s against 2.5 s, 194 against 366 MiB (the
memory is that of the geometry at 1.5 times the radius).

Real land mask of the pangeo-fish demo (tag A19124 emission file, level 10,
39 388 cells of which 29.6 % are land, read from disk, no download; sigma 1 and
2 cells, `"matched"`, `"full_disc"`). Weight that the PR 81 operator puts on
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
1 cell from land, PR 81 / `"full_disc"` barrier: 0.287 / 0.292 (sigma 1) and
0.382 / 0.399 (sigma 2); the columns of `"source"` sum to 1 (largest deviation
0, to rounding). Build time: 1.8 s against 2.8 s (sigma 1) and 7.6 s against
15.7 s (sigma 2), 1.98 M and 7.62 M non-zeros for PR 81 against 1.34 M and 4.97 M
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

## Custom radial kernels

A user-defined kernel receives a one-dimensional NumPy array containing the distances of valid target-neighbour pairs:

```python
def kernel(distance_m):
    ...
```

The kernel may return:

- one scalar weight, which is broadcast to every valid pair, or
- an array broadcastable to the number of valid pairs.

For example, a compact linear kernel can be written as:

```python
def linear_kernel(distance_m):
    radius_m = 1000.0

    return np.maximum(
        0.0,
        1.0 - distance_m / radius_m,
    )
```

and used as:

```python
filtered = radial_filter(
    values,
    cell_ids,
    refinement_level=17,
    radius_m=1000.0,
    kernel=linear_kernel,
    normalize=True,
)
```

Finite negative weights are allowed by the generic radial API. This makes it possible to define signed radial kernels rather than restricting the operator to smoothing-only use cases.

Non-finite weights returned by a custom kernel at valid neighbour positions are treated as an error.

## Centre-only support

`radial_filter` accepts:

```python
radius_m=0.0
```

A zero physical radius contains only the target cell centre, so a normalized finite self-weight acts as an identity operation.

Gaussian smoothing requires:

```text
sigma_m > 0
truncate > 0
```

because a zero-width Gaussian is not represented by the convenience wrapper.

## NumPy and PyTorch

Both NumPy arrays and PyTorch tensors are supported.

The final dimension of `values` corresponds to `cell_ids`:

```text
(N,)
(bands, N)
(time, bands, N)
```

Leading dimensions are preserved.

For Torch input:

- output remains on the original device,
- the weighted signal operation remains differentiable with respect to `values`,
- geometry and kernel weights are treated as spatial constants.

Example:

```python
import torch

values = torch.tensor(
    reflectance,
    dtype=torch.float32,
    device="cuda",
    requires_grad=True,
)

smoothed = gaussian_filter(
    values,
    cell_ids,
    refinement_level=17,
    sigma_m=240.0,
)

loss = smoothed.sum()
loss.backward()
```

## Geometry and weight cache sizing

`radial_filter` and `gaussian_filter` keep bounded least-recently-used caches
for value-independent metric geometry and Gaussian weights. The defaults are:

```text
geometry: 192 MiB
weights:   96 MiB
```

If one filter plan exceeds either limit, that entry is not cached. A
`RuntimeWarning` reports the required and configured sizes, explains that a
repeat will rebuild the data, and gives two safe options:

1. increase the relevant limit after checking available process memory; or
2. process spatial tiles with a halo of at least
   `radius_m = sigma_m * truncate`, discard halo outputs, and stitch only the
   tile interiors.

Splitting without a halo changes neighbourhoods at tile boundaries and does
not preserve the filtering result.

Configure or inspect the caches with:

```python
from healpix_analyse import (
    configure_radial_filter_cache,
    radial_filter_cache_info,
)

configure_radial_filter_cache(
    geometry_max_mib=576,
    weight_max_mib=384,
)

print(radial_filter_cache_info())
```

`None` leaves a limit unchanged, while zero disables that cache. Reducing a
limit immediately evicts least-recently-used entries until retained data fits.
The limits cover retained arrays only; temporary arrays used while building
exact WGS84 geometry require additional memory.

The following measurements use a patch centred at 2 degrees East, 48 degrees
North with `sigma_m=20` and `truncate=5` (100 m support):

| Level | patch size | cells | neighbour pairs | geometry | weights | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | 600 m | 3,883 | 705,275 | 8.13 MiB | 5.38 MiB | 13.51 MiB |
| 19 | 1,200 m | 15,075 | 2,881,729 | 33.21 MiB | 21.99 MiB | 55.19 MiB |
| 20 | 600 m | 15,069 | 10,942,451 | 125.46 MiB | 83.48 MiB | 208.94 MiB |
| 20 | 1,200 m | 59,426 | 45,475,888 | 521.34 MiB | 346.95 MiB | 868.29 MiB |

These are planning estimates, not fixed formulas. Memory depends on physical
radius, domain shape, latitude, HEALPix topology, and boundary-to-interior
ratio. A larger `sigma_m` or `truncate` increases the number of neighbour
pairs approximately with the square of the support radius. Use the warning's
measured requirement for the actual workload and leave headroom for values,
outputs, coordinate arrays, and concurrent filters.

## Relationship to neighbourhood reductions

`neighbour_reduce` performs unweighted local reductions such as:

```text
mean
median
min
max
count
mode
```

over a physical HEALPix neighbourhood.

`radial_filter` adds explicit metric weights:

```text
neighbour_reduce
    neighbours
        -> unweighted reduction

radial_filter
    neighbours
        + distance_m
        -> kernel(distance_m)
        -> weighted reduction
```

The spatial neighbourhood infrastructure is shared rather than independently reimplemented.

## Relationship to directional filtering

Radial and directional filters share the same relative-neighbour geometry and weighted aggregation layers but intentionally expose different scientific kernels.

Radial filtering uses only distance:

```text
#28
weight = kernel(distance_m)
```

Directional filtering additionally uses geographical direction:

```text
#29
weight = kernel(
    distance_m,
    relative_bearing_rad,
)
```

The separation is:

```text
_neighbourhood.py
    neighbour selection + WGS84 relative geometry

_weighted_neighbourhood.py
    generic weighted aggregation

radial_filter.py
    isotropic kernel(distance_m)

directional_filter.py
    directional kernel(distance_m, relative_bearing_rad)
```

This prevents radial filtering from depending on azimuth conventions and prevents directional filtering from duplicating the radial/weighted aggregation machinery.

## HEALPix topology and geographical boundaries

The filter uses physical WGS84 geometry rather than array indexing directions.

The same semantics therefore apply across:

- HEALPix base-pixel boundaries,
- longitude wrap-around,
- high-latitude and polar regions,
- different refinement levels.

No Cartesian row/column direction is assumed.

## Sentinel-2 MSI migration examples

Metric radial filtering is useful when migrating spatial smoothing from the
projected-raster Sentinel-2 MSI processing chain to HEALPix.

The EOPF S2MSI documentation describes the Projected Geometry Processor as the
part of the L2 chain responsible for projected products and includes L2A scene
classification and atmospheric correction. The L2A outputs include spatial
fields such as cloud/snow probability layers and atmospheric quantities such
as aerosol optical thickness (AOT) and water vapour (WVP).

See:

- [EOPF S2MSI — Detailed Processing Model: Projected Geometry Processor](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html)
- [EOPF S2MSI — L2A Scene Classification Processor API](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html)
- [EOPF S2MSI — Level-2A Products](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/PDFS_ADFS/L2/PDFS_S2_MSI_L2.html)

The examples below illustrate a **migration pattern**. They do not claim that
S2MSI internally calls a particular library function such as
`scipy.ndimage.gaussian_filter`, nor do the numerical `sigma_m` values below
claim to reproduce a particular internal S2MSI parameter.

### From projected raster smoothing to metric HEALPix smoothing

In a projected UTM-style raster workflow, a local smoothing operation is
naturally expressed using a kernel or neighbourhood whose scale is tied to
raster pixels and raster resolution.

Conceptually:

```python
smoothed = raster_smoothing(
    field,
    scale_pixels=...,
)
```

A pixel-based parameter does not have a resolution-independent physical
meaning. For example, the same nominal `scale_pixels` applied to 20 m and
60 m raster products represents different physical distances.

On HEALPix, the corresponding operation can instead express the scientific
scale directly in metres:

```python
from healpix_analyse import gaussian_filter

smoothed = gaussian_filter(
    values,
    cell_ids,
    refinement_level,
    sigma_m=60.0,
)
```

The weights are evaluated from WGS84 centre-to-centre distances:

```text
weight(d) = exp(-0.5 * (d / sigma_m)^2)
```

The smoothing scale therefore remains a physical quantity rather than a
property of the raster sampling or HEALPix refinement level.

### Scene-classification and cloud-shadow fields

The S2MSI L2A scene-classification processor operates on spatial
classification and confidence/probability information, including cloud and
cloud-shadow processing.

A projected raster implementation may apply local smoothing or weighted
neighbourhood processing to such fields. In HEALPix, the same class of
operation can be expressed using a physical smoothing scale:

```python
smoothed_shadow_probability = gaussian_filter(
    cloud_shadow_probability,
    cell_ids,
    refinement_level,
    sigma_m=120.0,
)
```

Here `120.0` is only an illustrative HEALPix processing scale.

The important migration principle is:

```text
projected raster
    kernel / window scale tied to pixels
                |
                v
HEALPix
    explicit physical scale in metres
```

This keeps the interpretation of the filter independent of HEALPix base-pixel
boundaries, longitude wrap-around, and refinement level.

### Atmospheric fields

The S2MSI L2A projected processing chain also handles atmospheric quantities,
including AOT and WVP, as part of atmospheric correction.

For a projected raster workflow, local smoothing or spatial weighting may be
expressed through raster-window or pixel-scale parameters. A HEALPix
implementation can instead express the same class of spatial operation using a
physical radius or Gaussian scale:

```python
smoothed_aot = gaussian_filter(
    aot,
    cell_ids,
    refinement_level,
    sigma_m=240.0,
)
```

or with a custom physical influence kernel:

```python
import numpy as np

from healpix_analyse import radial_filter


def atmospheric_kernel(distance_m):
    return np.exp(-distance_m / 500.0)


smoothed_field = radial_filter(
    atmospheric_field,
    cell_ids,
    refinement_level,
    radius_m=2000.0,
    kernel=atmospheric_kernel,
    normalize=True,
)
```

Again, these values are illustrative. The point is that the HEALPix API makes
the spatial scale explicit in metres instead of encoding it indirectly through
the resolution of a projected raster.

### Why this matters for the S2MSI-to-HEALPix migration

The projected S2MSI workflow naturally works with two-dimensional raster
arrays whose local neighbourhoods are described in row/column or pixel terms.
HEALPix does not have a globally valid Cartesian row/column neighbourhood.

`radial_filter` therefore preserves the scientific idea of a local isotropic
filter while changing the spatial definition:

```text
projected raster
    local pixel neighbourhood
    + raster-dependent scale

            becomes

HEALPix
    physical neighbourhood
    + WGS84 distance
    + kernel(distance_m)
```

This is a migration of the **spatial semantics**, not a claim of numerical
identity with any particular internal projected-raster implementation.

## Implementation architecture

The implementation deliberately reuses the shared geometry introduced for neighbouring HEALPix operators and the shared weighted aggregation used by directional filtering.

```text
physical neighbourhood
        |
        v
processing-domain restriction
        |
        v
relative_geometry_from_neighbourhoods(...)
        |
        v
distance_m
        |
        v
kernel(distance_m)
        |
        v
weights
        |
        v
weighted_neighbourhood_reduce(...)
        |
        v
filtered values
```

`radial_filter` does not independently implement:

- HEALPix neighbourhood padding,
- WGS84 geodesic distance calculation,
- neighbour-value gathering,
- generic weighted normalization,
- Torch device/autograd handling.

Those responsibilities remain in the shared private layers so that radial and directional operators behave consistently.
