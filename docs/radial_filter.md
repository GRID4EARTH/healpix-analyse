# Metric radial filtering on HEALPix

`radial_filter` applies a user-defined isotropic kernel to values on a NESTED HEALPix grid using **physical WGS84 centre-to-centre distance in metres**.

A convenience `gaussian_filter` wrapper provides normalized Gaussian smoothing using `sigma_m` rather than a pixel-based standard deviation.

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

The Gaussian wrapper is normalized by construction, so a constant finite field remains constant after filtering.

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
