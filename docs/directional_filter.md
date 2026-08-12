# Geographical directional filtering

`directional_filter()` applies a directional spatial kernel to a scalar field
stored on a NESTED HEALPix grid.

Unlike a Cartesian image filter, its public spatial semantics are expressed in
physical Earth coordinates:

- distance in **metres**,
- direction as a **geographical azimuth**,
- geometry computed between HEALPix cell centres on **WGS84**.

This makes the operator suitable for Earth-observation processing where the
direction has a physical meaning, for example solar azimuth, shadow
displacement, illumination direction, wind direction, or another geographical
bearing.

## API

```python
from healpix_analyse.directional_filter import directional_filter
```

```python
directional_filter(
    values,
    cell_ids,
    refinement_level,
    *,
    max_distance_m,
    azimuth_rad,
    kernel,
    normalize=False,
    domain=None,
    ellipsoid="WGS84",
)
```

The final dimension of `values` corresponds one-to-one with `cell_ids`.
Arbitrary leading dimensions are preserved.

For example:

```text
(N,)
(bands, N)
(time, bands, N)
```

Both NumPy arrays and PyTorch tensors are supported.

For Torch input, the result remains on the same device and the weighted signal
operation remains differentiable with respect to `values`. Geometry and kernel
weights are treated as constants.

---

## Geographical direction

Azimuth follows the conventional geographical definition:

```text
             North
               0
               ^
               |
               |
West 3pi/2 <---+---> pi/2 East
               |
               |
               v
              pi
             South
```

Angles increase **clockwise from geographic North**:

```text
0           = North
pi / 2      = East
pi          = South
3 * pi / 2  = West
```

Values outside `[0, 2*pi)` are accepted and wrapped.

This convention is intentionally geographical. It is not based on array rows,
array columns, HEALPix neighbour ordering, or the orientation of a projected
raster.

---

## Target-to-neighbour bearing

For an output/target HEALPix cell `i` and a contributing source/neighbour cell
`j`, the geometry is defined as:

```text
                       geographic North
                              ^
                              |
                         j  * |
                           /  |
                          /   |
                         /    |
                        *-----+
                        i
```

The operator computes:

```text
distance_ij
    = WGS84 geodesic centre-to-centre distance from i to j

bearing_ij
    = WGS84 forward geographical azimuth from i to j

relative_bearing_ij
    = wrap(bearing_ij - requested_azimuth)
```

with `relative_bearing_ij` wrapped to:

```text
[-pi, +pi)
```

Therefore:

```text
relative_bearing = 0
    neighbour lies along the requested direction

relative_bearing = +pi/2
    neighbour lies 90 degrees clockwise from the requested direction

relative_bearing = -pi/2
    neighbour lies 90 degrees counter-clockwise from the requested direction
```

The forward-bearing direction is important: it is always from the **output
cell toward the contributing neighbour**.

---

## Directional kernel

The user supplies a callable:

```python
kernel(distance_m, relative_bearing_rad)
```

For every valid target-neighbour pair, the resulting weight is:

```text
weight_ij =
    kernel(
        distance_ij,
        relative_bearing_ij,
    )
```

The kernel receives one-dimensional NumPy arrays and may return either a scalar
weight or an array broadcastable to the number of valid pairs.

### Forward angular sector

For example, a kernel accepting cells within 20 degrees of the requested
direction is:

```python
import numpy as np


def forward_sector(distance_m, relative_bearing_rad):
    del distance_m

    return (
        np.abs(relative_bearing_rad)
        <= np.deg2rad(20.0)
    ).astype(float)
```

It can be used as:

```python
filtered = directional_filter(
    values,
    cell_ids,
    refinement_level,
    max_distance_m=500.0,
    azimuth_rad=np.deg2rad(135.0),
    kernel=forward_sector,
)
```

Here `135 degrees` has its normal geographical meaning: southeast.

### Distance-dependent directional kernel

Distance and direction can be combined in the same kernel:

```python
def directional_gaussian(
    distance_m,
    relative_bearing_rad,
):
    radial = np.exp(
        -0.5 * (distance_m / 200.0) ** 2
    )

    angular = np.exp(
        -0.5
        * (
            relative_bearing_rad
            / np.deg2rad(15.0)
        ) ** 2
    )

    return radial * angular
```

For example:

```python
filtered = directional_filter(
    values,
    cell_ids,
    refinement_level,
    max_distance_m=600.0,
    azimuth_rad=np.deg2rad(110.0),
    kernel=directional_gaussian,
    normalize=True,
)
```

The spatial support remains physically defined by `max_distance_m`; the kernel
then controls the contribution of cells within that support.

---

## Physical support in metres

`max_distance_m` is the maximum WGS84 centre-to-centre distance over which a
cell may contribute.

This is deliberately different from specifying:

```text
3 x 3 pixels
5 x 5 pixels
HEALPix ring = 1
HEALPix ring = 2
```

Those descriptions depend on raster layout or HEALPix topology. The public
meaning of `directional_filter()` instead remains:

```text
"consider cells no farther than this physical distance"
```

regardless of HEALPix base-pixel boundaries, longitude wrapping, or local
orientation.

Neighbour discovery uses the shared physical-neighbourhood infrastructure,
after which WGS84 distances and forward azimuths are computed using the shared
relative-geometry implementation.

---

## Self contribution

The geographical bearing of a zero-distance pair is undefined.

When a target cell contributes to itself, `directional_filter()` therefore
defines:

```text
distance_m = 0
relative_bearing_rad = 0
```

by convention.

The supplied kernel is still responsible for deciding the self weight.

For example, a purely angular kernel may include the centre, while a kernel
that requires a positive distance may exclude it.

---

## Domain semantics

`domain` follows the same processing-domain semantics as the neighbourhood
operators in `healpix-analyse`.

`cell_ids` identifies all cells for which input values exist.

`domain` identifies both:

- the cells that participate in the spatial operation, and
- the output cells, in the requested output order.

If `domain` is omitted:

```text
domain == cell_ids
```

If a cell exists in `cell_ids` but lies outside `domain`, it does **not**
contribute.

It is not interpreted as:

- zero,
- NaN padding,
- periodic continuation,
- a wrapped array index.

It is simply absent from the effective spatial neighbourhood.

For example:

```python
domain = np.array(
    [cell_a, cell_b, cell_c],
    dtype=np.uint64,
)

filtered = directional_filter(
    values,
    cell_ids,
    refinement_level,
    max_distance_m=300.0,
    azimuth_rad=np.deg2rad(90.0),
    kernel=forward_sector,
    domain=domain,
)
```

The result has final dimension `len(domain)` and follows the exact ordering of
`domain`.

This convention avoids artificial boundary contributions when processing a
regional subset of a HEALPix field.

---

## Normalization and missing samples

With:

```python
normalize=False
```

the result is:

```text
sum(weight * value)
```

With:

```python
normalize=True
```

the result is:

```text
sum(weight * value)
-------------------
    sum(weight)
```

using only effective valid samples.

NaN-valued signal samples are treated as unavailable observations. Their
weights are removed from both numerator and denominator.

If normalization is requested and the effective weight sum is zero, the
result for that output location is `NaN`.

This behaviour is implemented by the shared private
`weighted_neighbourhood_reduce()` helper and is common to weighted spatial
operators.

---

# Sentinel-2 MSI migration example

One motivation for `directional_filter()` is the migration of Sentinel-2 MSI
processing from projected Cartesian tiles to HEALPix.

The EOPF Sentinel-2 MSI Detailed Processing Model describes the **Projected
Geometry Processor (PGP)** as the part of the chain operating in **UTM
geometry**. PGP contains the end of L1C Mask S2 processing as well as the L2A
Scene Classification and Atmospheric Correction processing.

EOPF reference:

https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html

The corresponding L2A scene-classification API documents cloud-shadow
processing with inputs including:

- `resolution` in metres,
- `solaz` (Solar Azimuth Angle),
- `solze_noclip` (Solar Zenith Angle),
- cloud-height thresholds in metres.

EOPF reference:

https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html

These quantities have physical meanings that should be preserved when the
storage geometry changes.

## Projected raster interpretation

In a UTM raster implementation, a physical displacement is eventually turned
into movements along Cartesian image axes.

Conceptually:

```text
solar / shadow azimuth
          +
physical distance [m]
          +
projected pixel resolution [m / pixel]
          |
          v
Cartesian displacement
(dx, dy) or row/column offsets
          |
          v
sample / shift projected raster
```

The image-array displacement is an implementation of the underlying
geographical operation.

It should not become the definition of that operation.

For example, a displacement of several hundred metres toward a geographical
azimuth should not be migrated to HEALPix as:

```text
move N HEALPix indices
```

or:

```text
take the third neighbour returned by a topology routine
```

because HEALPix index ordering and neighbour-list ordering do not define
geographical East, West, North, or South.

## HEALPix interpretation

For a HEALPix-native processing chain, the same physical operation can instead
be expressed directly:

```text
solar / shadow azimuth
          +
physical distance [m]
          |
          v
physical HEALPix neighbourhood
          |
          v
WGS84 distance
+
forward geographical bearing
          |
          v
relative bearing to requested direction
          |
          v
kernel(
    distance_m,
    relative_bearing_rad,
)
          |
          v
weighted HEALPix aggregation
```

This removes the projected-raster indexing step while preserving the
scientific quantities that motivated it.

In other words:

```text
UTM implementation
------------------

physical direction/distance
        |
        v
convert to projected dx/dy
        |
        v
operate on image pixels


HEALPix implementation
----------------------

physical direction/distance
        |
        v
operate directly on
WGS84 distance + bearing
```

The HEALPix version is therefore not attempting to reproduce the UTM
row/column representation. It preserves the geographical meaning of the
operation.

---

## Example: cloud-shadow direction

The EOPF L2A scene-classification interface exposes the solar azimuth together
with a ground resolution in metres and cloud-height parameters in metres for
cloud-shadow detection.

A HEALPix migration can use these physical quantities directly.

For example, suppose an existing stage determines that the relevant shadow
search direction is `shadow_azimuth_rad` and that candidate contributions
should be considered within `shadow_distance_m`.

A simple forward shadow-sector kernel could be:

```python
def shadow_sector(
    distance_m,
    relative_bearing_rad,
):
    return (
        np.abs(relative_bearing_rad)
        <= np.deg2rad(10.0)
    ).astype(float)
```

and applied as:

```python
shadow_response = directional_filter(
    cloud_probability,
    cell_ids,
    refinement_level,
    max_distance_m=shadow_distance_m,
    azimuth_rad=shadow_azimuth_rad,
    kernel=shadow_sector,
    normalize=False,
    domain=domain,
)
```

A more selective kernel can incorporate both the expected shadow displacement
and the angular tolerance:

```python
def shadow_displacement_kernel(
    distance_m,
    relative_bearing_rad,
):
    target_distance_m = 400.0
    distance_tolerance_m = 50.0
    angular_tolerance_rad = np.deg2rad(10.0)

    radial_match = (
        np.abs(
            distance_m - target_distance_m
        )
        <= distance_tolerance_m
    )

    angular_match = (
        np.abs(relative_bearing_rad)
        <= angular_tolerance_rad
    )

    return (
        radial_match
        & angular_match
    ).astype(float)
```

This corresponds naturally to the physical statement:

```text
look about 400 m away
in the requested geographical direction
with a +/- 10 degree angular tolerance
```

rather than to a particular number of rows and columns in a projected image.

The exact cloud-shadow kernel still belongs to the migrated S2MSI processing
logic. `directional_filter()` provides the generic HEALPix operation needed to
express that logic without embedding S2MSI-specific rules in
`healpix-analyse`.

---

## Example: terrain / cast-shadow processing

The Projected Geometry Processor also carries terrain slope and cast-shadow
information through the L2A workflow.

EOPF processing overview:

https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/ppb/projected_geometry.html

For a projected UTM raster, a terrain-shadow operation may use projected
directions and raster offsets internally.

When such an operation is migrated to HEALPix, any part whose meaning is:

```text
search / propagate / weight
along a geographical azimuth
over a physical ground distance
```

can be represented using the same directional-filter abstraction.

The generic library should not encode Sentinel-2 terrain or shadow physics
itself. Instead, S2MSI supplies the physically meaningful direction,
distance, and kernel.

This separation keeps:

```text
S2MSI
    -> mission-specific shadow / illumination model

healpix-analyse
    -> generic geographical directional operator
```

---

# Relationship to other HEALPix analysis operators

The spatial infrastructure is intentionally separated into layers.

```text
_neighbourhood.py
    |
    +-- physical neighbour selection
    +-- WGS84 distance
    +-- WGS84 forward azimuth
    |
    v
directional_filter.py
    |
    +-- requested geographical azimuth
    +-- relative bearing
    +-- directional kernel evaluation
    |
    v
_weighted_neighbourhood.py
    |
    +-- gather neighbour values
    +-- validity / padding handling
    +-- NaN handling
    +-- supplied weights
    +-- normalization
    +-- NumPy / Torch aggregation
```

This also allows radial and directional filters to share the signal
aggregation machinery without sharing their spatial semantics.

Conceptually:

```text
radial filter
    kernel(distance_m)
            |
            v
weighted_neighbourhood_reduce()


directional filter
    kernel(
        distance_m,
        relative_bearing_rad,
    )
            |
            v
weighted_neighbourhood_reduce()
```

---

## Difference from neighbourhood reduction

`neighbour_reduce()` is intended for unweighted neighbourhood reductions such
as:

- mean,
- median,
- extrema,
- counts,
- mask reductions.

`directional_filter()` is for supplied spatial weights whose values depend on
both distance and geographical orientation.

A directional kernel therefore should not be represented as an artificial
categorical or unweighted neighbourhood reduction.

---

## Difference from radial filtering

A radial filter depends only on physical distance:

```text
weight = kernel(distance_m)
```

A directional filter additionally depends on geographical orientation:

```text
weight =
    kernel(
        distance_m,
        relative_bearing_rad,
    )
```

The two operations can share physical-neighbourhood geometry and generic
weighted aggregation while retaining distinct public APIs.

---

## Difference from `HealPixConv`

`directional_filter()` is deliberately different from `HealPixConv`.

`HealPixConv` represents a gauge-oriented spherical convolution using a
rotated local stencil.

`directional_filter()` instead defines weights from actual WGS84
centre-to-centre geometry:

```text
physical distance
+
geographical forward bearing
+
requested geographical azimuth
```

Therefore use `directional_filter()` when the direction itself has a physical
geographical interpretation, for example:

- solar azimuth,
- shadow direction,
- illumination direction,
- wind direction,
- transport direction.

Use a convolution operator when the intended semantics are those of a
convolutional stencil rather than a physical geographical bearing.

---

# Design summary

The key rule is:

> Do not translate a physically directional Earth-observation operation into
> HEALPix index directions.

Instead, preserve the original physical meaning.

For `directional_filter()` that means:

```text
distance
    -> metres on WGS84

direction
    -> geographical forward azimuth

0 radians
    -> North

pi / 2 radians
    -> East

relative bearing
    -> target-to-neighbour bearing minus requested azimuth

support
    -> physical radius

boundary
    -> explicit processing domain

aggregation
    -> shared weighted-neighbour reduction
```

This makes the operator independent of UTM row/column orientation and of
HEALPix topological neighbour ordering, while retaining the quantities needed
by directional Earth-observation processing.
