# HEALPix neighbourhood reductions

`neighbour_reduce` applies an unweighted reduction to values in a physical neighbourhood around each HEALPix cell.

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

## Earth Observation migration example

A common raster-processing pattern is to apply a local median filter before
resampling or further classification.

For example, a Cartesian raster workflow may use a fixed pixel window:

```python
from scipy.ndimage import median_filter

filtered = median_filter(
    image,
    size=3,
)
```

On HEALPix, there is no fixed Cartesian `3x3` pixel window.

Instead, the equivalent operation should be expressed using a physical
neighbourhood:

```python
from healpix_analyse import median_filter

filtered = median_filter(
    values,
    cell_ids,
    refinement_level,
    radius_m=60.0,
    neighbourhood="cell_center",
)
```

The HEALPix version preserves the purpose of the raster operation while
replacing a grid-dependent pixel window with a physical-radius neighbourhood.

This pattern is useful for Earth Observation workflows migrating local raster
filters to HEALPix while keeping the processing scale expressed in physical
units.

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

See {doc}`directional_filter` for geographical directional filtering and its
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
