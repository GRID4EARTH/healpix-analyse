# HEALPix neighbourhood reductions

`neighbour_reduce` applies an unweighted reduction to values in a physical neighbourhood around each HEALPix cell.

```{include} ../_generated/glance/neighbour_reduce.md
```

Unlike Cartesian image filters, neighbourhoods are specified using a physical radius rather than a fixed pixel window such as `3x3` or `5x5`. This makes the operation independent of HEALPix refinement level and avoids assumptions about Cartesian x/y directions.

## Basic usage

```python
from healpix_analyse import neighbour_reduce

filtered = neighbour_reduce(
    values,
    cell_ids,
    refinement_level=17,
    radius_m=240.0,
    reduction="median",
)
```

The last dimension of `values` corresponds one-to-one with `cell_ids`.

## Input cells and processing domain

`cell_ids` and `domain` have different roles.

### `cell_ids`

`cell_ids` identifies every HEALPix cell for which an input value is available.

For example,

```python
cell_ids = [10, 11, 12]
values   = [2.0, 4.0, 1000.0]
```

means:

```text
cell 10 ->    2.0
cell 11 ->    4.0
cell 12 -> 1000.0
```

### `domain`

`domain` defines the valid processing and output domain.

If `domain` is omitted,

```python
domain=None
```

is equivalent to using all input cells:

```text
domain = cell_ids
```

If `domain` is explicitly supplied, it must be a subset of `cell_ids`:

```text
domain ⊆ cell_ids
```

Every output value corresponds to one cell in `domain`, and output ordering follows the ordering of `domain`.

For example,

```python
domain = [11, 10]
```

produces output values ordered first for cell `11`, then for cell `10`.

## Behaviour at regional-domain boundaries

A geometric neighbourhood may extend beyond the processing domain.

Cells outside `domain` do not participate in the reduction, even if:

- they are geometrically within `radius_m`, and
- a value for that cell is present in `values`.

They are not interpreted as zero, `False`, `NaN`, or any other padding value. They are simply absent from the effective neighbourhood.

For example:

```python
cell_ids = [10, 11, 12]
values   = [2.0, 4.0, 1000.0]
domain   = [10, 11]
```

Assume that the geometrical neighbourhood around cell `11` is

```text
[10, 11, 12]
```

Because cell `12` lies outside `domain`, the effective neighbourhood is

```text
[10, 11]
```

and therefore

```text
mean = (2.0 + 4.0) / 2
     = 3.0
```

The value `1000.0` does not contribute.

This rule is important for regional or masked HEALPix datasets. The library does not invent samples beyond the valid processing domain, so local sample counts naturally decrease near a domain boundary.

## `include_self`

By default,

```python
include_self=True
```

and the target cell participates in its own neighbourhood reduction.

With

```python
include_self=False
```

the target cell is explicitly removed from the effective neighbourhood.

For sufficiently small radii, or for isolated cells in a partial domain, this can produce an empty neighbourhood.

Reductions with a natural empty-set identity use that identity:

| Reduction | Empty neighbourhood |
|---|---|
| `sum` | `0` |
| `count` | `0` |
| `any` | `False` |
| `all` | `True` |

Reductions without a useful empty-set value raise `ValueError`:

- `mean`
- `median`
- `min`
- `max`
- `std`
- `mode`

This avoids silently introducing artificial values into scientific results.

## Numerical median versus categorical mode

`median` and `mode` are intentionally distinct.

`median` is a numerical reduction. For an even number of samples, the two central sorted values are averaged.

For example:

```text
values = [1, 1, 9, 9]

median = (1 + 9) / 2 = 5
```

`mode` is a categorical reduction and returns the most frequently occurring category. Its tie-breaking rule must be deterministic.

This distinction is particularly important when integer-valued scientific classification fields are processed numerically.

## Partial-domain example

```python
import numpy as np

from healpix_analyse import neighbour_reduce

cell_ids = np.array([10, 11, 12], dtype=np.uint64)
values = np.array([2.0, 4.0, 1000.0])

result = neighbour_reduce(
    values,
    cell_ids,
    refinement_level=5,
    radius_m=100.0,
    reduction="mean",
    domain=np.array([10, 11], dtype=np.uint64),
)
```

Only cells belonging to `domain` may contribute to each output reduction.

The output contains exactly two values, corresponding to cells `10` and `11`.

## Reductions, aliases and the reusable reducer

`reduction` is one of `"mean"`, `"sum"`, `"min"`, `"max"`, `"median"`,
`"count"`, `"std"`, `"any"`, `"all"`, `"mode"`.

- `std` is the population standard deviation (no correction).
- `median` averages the two middle values for an even count.
- `mode` needs integer or boolean values; ties go to the smallest value.
- `any` and `all` need boolean values.
- Output dtype: `mean`, `median` and `std` are floating point (integer input
  becomes float64); `sum` of integers is int64; `count` is int64; `any` and
  `all` are boolean; `min`, `max` and `mode` keep the input dtype.
- An empty neighbourhood gives 0 for `sum` and `count`, `False` for `any`,
  `True` for `all`, and raises `ValueError` for the other reductions.

Four aliases fix the reduction and take the same other arguments:
`median_filter`, `mean_filter`, `min_filter`, `max_filter`.

To apply several reductions, or the same one to many fields, on the same cells,
build the neighbourhoods once with `HealPixNeighbourReducer` (`neighbour_reduce`
builds a new one at every call):

```python
from healpix_analyse import HealPixNeighbourReducer

reducer = HealPixNeighbourReducer(
    cell_ids, refinement_level=17, radius_m=240.0,
)
med = reducer(values, reduction="median")
p_max = reducer(values, reduction="max")
reducer.output_cell_ids          # the domain, in output order
```

## Differentiability, missing values and cost

- **Differentiable** (Torch input) for `mean`, `sum`, `min`, `max`, `median`
  and `std`; `count` and `mode` return integers and `any` / `all` booleans,
  which carry no gradient.
- **NaN propagates**: a NaN inside a neighbourhood makes the `mean`, `std`,
  `min`, `max` and `median` of that cell NaN. If you need NaN-aware
  averaging, use `radial_filter` with a top-hat kernel
  ({doc}`compare_neighbourhood`).
- **Cost**: the neighbourhoods are built once per call on the CPU (healpix-geo
  and pyproj); the reduction is then a Python loop over the output cells, on
  the device of the input tensor. It is therefore not the cheapest local
  operator: in one CPU measurement (20 000 cells at level 12, 3 km radius),
  `mean_filter` took about 0.5 s per call, against 0.15 s for the first call
  of `radial_filter` with a top-hat kernel and 0.012 s once its geometry is
  cached. To be confirmed by a benchmark in `benchmarks/`.

## Spatial semantics

Neighbourhood construction uses HEALPix geometry and a physical radius in metres.

The API therefore does not define neighbourhoods using Cartesian concepts such as:

```text
3x3
5x5
left/right
row/column
```

The same semantics apply across HEALPix base-pixel boundaries, near the poles, and across longitude wrap-around.

Weighted, Gaussian, and arbitrary radial-kernel filters are separate operations and are not part of `neighbour_reduce`.

## Related weighted neighbourhood operators

`neighbour_reduce()` is intended for unweighted neighbourhood reductions such as
mean, median, extrema, counts, and mask reductions.

Weighted spatial kernels are implemented as separate operators because their
semantics are different from categorical or unweighted reductions.

Use:

- radial filtering when weights depend only on physical distance,
- directional filtering when weights depend on physical distance and
  geographical bearing.

For example, directional filtering uses:

```text
weight =
    kernel(
        distance_m,
        relative_bearing_rad,
    )
```

with geographical azimuth measured clockwise from North.

See {doc}`/conv/directional_filter` for geographical directional filtering and its
Sentinel-2 MSI migration example.

All of these operators follow the same domain principle:

```text
cells outside domain
    -> absent from the effective neighbourhood
    -> not zero
    -> not NaN padding
    -> not periodic continuation
```

The weighted operators share generic neighbour-value aggregation internally,
while neighbour discovery and WGS84 geometry remain separate spatial concerns.
