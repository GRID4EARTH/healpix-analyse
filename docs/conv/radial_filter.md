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
weight = kernel(distance_m)
```

Directional filtering additionally uses geographical direction:

```text
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
