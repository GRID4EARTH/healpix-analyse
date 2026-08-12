# HEALPix scalar-field gradients

`healpix-analyse` provides local scalar-field gradients for NESTED HEALPix
data using the real geographic geometry of neighbouring cell centres.

The gradient is expressed in a local tangent basis:

- `grad_east`: derivative toward geographic East
- `grad_north`: derivative toward geographic North

If the input field has units `U`, the gradient components have units
`U / metre`.

## Why not use Cartesian x/y derivatives directly?

Projected raster processing commonly represents local spatial derivatives
along fixed image x/y axes.

HEALPix does not provide one globally aligned Cartesian x/y orientation over
the sphere. A HEALPix implementation should therefore not interpret cell
index directions as geographic East or North.

For each target cell, `healpix-analyse` instead uses:

```text
immediate HEALPix neighbours
        |
        v
WGS84 centre-to-centre geometry
        |
        +-- geodesic distance
        +-- geographic forward azimuth
        |
        v
local East / North offsets
        |
        v
least-squares tangent gradient
```

This defines a local geographic derivative without requiring a global raster
orientation.

## Basic usage

```python
from healpix_analyse import gradient

grad_east, grad_north = gradient(
    values,
    cell_ids,
    refinement_level,
)
```

The output follows the order of `cell_ids`.

The current gradient operator uses the immediate HEALPix neighbourhood
(`ring=1`) internally.

## Local tangent model

For a target cell with scalar value `f0`, each valid neighbouring cell
provides

```text
delta_f = f_neighbour - f0
```

and a geographic offset

```text
delta_east
delta_north
```

in metres.

The local scalar field is approximated by

```text
delta_f ~= grad_east  * delta_east
         + grad_north * delta_north
```

The two gradient components are estimated using an unweighted local
least-squares fit.

The result therefore depends on the actual geographic geometry of the
neighbouring cells rather than on their positional ordering in a HEALPix
topology array.

## Geographic direction convention

Geographic azimuth is measured clockwise from North:

```text
          North
            0
            ^
            |
West <------C------> East
-pi/2               +pi/2
            |
            v
           South
            pi
```

For a neighbour at geodesic distance `d` and forward azimuth `a`:

```text
east_offset  = d * sin(a)
north_offset = d * cos(a)
```

HEALPix topological direction labels are not used as geographic East/North
labels.

## Gradient magnitude

Use `gradient_magnitude()` when only the magnitude of the local spatial
derivative is required.

```python
from healpix_analyse import gradient_magnitude

magnitude = gradient_magnitude(
    values,
    cell_ids,
    refinement_level,
)
```

It computes

```text
sqrt(
    grad_east**2
    + grad_north**2
)
```

## Directional derivative

A derivative along a geographic azimuth can be obtained with
`directional_derivative()`.

```python
import numpy as np

from healpix_analyse import directional_derivative

eastward = directional_derivative(
    values,
    cell_ids,
    refinement_level,
    azimuth_rad=np.pi / 2,
)
```

The convention is:

```text
0        -> North
pi / 2   -> East
pi       -> South
-pi / 2  -> West
```

The directional derivative is derived from the local tangent gradient:

```text
directional_derivative
    =
grad_east  * sin(azimuth)
+
grad_north * cos(azimuth)
```

It does not use HEALPix index directions.

## Processing domains

`cell_ids` identifies cells for which input values are available.

An optional `domain` identifies the valid processing and output region.

```python
grad_east, grad_north = gradient(
    values,
    cell_ids,
    refinement_level,
    domain=domain,
)
```

Every cell in `domain` must occur in `cell_ids`.

The output follows the order of `domain`.

Cells outside the domain do not participate in the local fit. They are not
interpreted as zero, `False`, or NaN padding.

Conceptually:

```text
               outside domain
                    |
                    v

               X X X
               X C | outside
               X X |
```

For target cell `C`, only neighbours belonging to the processing domain
participate in the gradient estimate.

This avoids creating an artificial derivative solely because a regional
dataset stops at that boundary.

## Incomplete neighbourhoods

A two-dimensional tangent gradient requires enough valid neighbour geometry
to constrain both East and North components.

A gradient is returned when the remaining finite neighbours span a rank-two
local tangent plane.

If this is not possible, both gradient components are returned as NaN.

Examples include:

- fewer than two usable neighbours;
- remaining neighbours that do not span two independent local directions;
- a missing or non-finite target value.

Missing neighbour values are ignored if the remaining neighbourhood still
supports a rank-two fit.

## NumPy and PyTorch

Both NumPy arrays and PyTorch tensors are supported.

```python
import torch

values_torch = torch.tensor(
    values,
    dtype=torch.float32,
)

grad_east, grad_north = gradient(
    values_torch,
    cell_ids,
    refinement_level,
)
```

Torch results remain on the input device.

The numerical gradient operation is differentiable with respect to the input
values, so PyTorch autograd is preserved.

The geographic geometry itself depends on the HEALPix cells and refinement
level and is independent of the scalar values.

## Shared neighbourhood geometry

Gradient estimation reuses the shared HEALPix neighbourhood geometry layer.

The implementation separates:

```text
candidate neighbour discovery
        |
        v
relative geographic geometry
        |
        v
numerical operation
```

The relative geometry contains:

- neighbour cell IDs;
- valid-neighbour mask;
- WGS84 distance;
- geographic forward azimuth;
- East offset;
- North offset.

This geometry can be reused across variables or bands defined on the same
HEALPix cells.

The same generic relative-geometry primitive can also support other local
operators such as radial and directional filters.

# Sentinel-2 MSI / EOPF processing context

This section intentionally refers only to information published in the
public Sentinel-2 MSI Processor documentation.

It does **not** document or disclose non-public source-code traces, internal
implementation details, or unpublished algorithm choices.

## Public processing context

The public S2MSI documentation describes the Projected Geometry Processor as
part of the Sentinel-2 MSI processing chain and identifies Level-2A Scene
Classification and Atmospheric Correction as major processing functions.

Public documentation:

- S2MSI documentation home:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/index.html
- Detailed Processing Model — Projected Geometry Processor:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html
- All Projected Geometry Processing Unit:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/pu_all_projected_geometry.html
- Projected Geometry performance/processor description:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/ppb/projected_geometry.html

The public Projected Geometry DPM describes Scene Classification as part of
the L2A core workflow and documents a processing context that includes
projected reflectance, classification masks, atmospheric quantities, and
auxiliary geospatial information.

`healpix-analyse.gradient()` is not an S2MSI-specific algorithm. It is a
generic HEALPix building block that can be used when a processing stage
needs a local geographic spatial derivative after projected raster data are
represented on HEALPix.

## Scene Classification context

The public S2MSI DPM states that the L2A Core Processing Unit can run Scene
Classification and stores a Scene Classification Mask (SCL), Snow Confidence
Mask (SNW), and Cloud Confidence Mask (CLD).

Public documentation:

- Detailed Processing Model — Projected Geometry Processor:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html
- Scene Classification Processor API:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html
- S2MSI Context and Terminology:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/context.html

For a future HEALPix implementation of a processing step that requires a
local scalar-field derivative or local edge-strength quantity, the generic
mapping is:

```text
projected scalar field
        |
        v
HEALPix scalar field
        |
        v
gradient(...)
        |
        +-- geographic East derivative
        +-- geographic North derivative
        |
        v
gradient_magnitude(...)
        |
        v
application-specific classification logic
```

The application-specific classification rule remains outside
`healpix-analyse`.

This documentation intentionally does not claim that a particular private
S2MSI routine must be replaced by `gradient()`. It shows how the generic
operator can be used when the publicly documented processing context
requires an equivalent local spatial-derivative concept.

## DEM, slope, aspect, and hillshade context

The public Projected Geometry documentation describes a feature flag for DEM
data and its derivatives, including slope, aspect, and hillshade. The public
Scene Classification API also exposes processing interfaces involving a
`slope` field.

Public documentation:

- Detailed Processing Model — Projected Geometry Processor:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html
- Projected Geometry Interface Control Document:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/icd.html
- Scene Classification Processor API:
  https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html

For DEM values already represented on HEALPix, a generic local derivative
can be obtained as:

```python
grad_east, grad_north = gradient(
    dem_values,
    cell_ids,
    refinement_level,
)
```

These components provide geographically defined first derivatives that an
application layer can use as inputs to terrain-derived quantities.

For example, a HEALPix migration layer can conceptually use:

```text
HEALPix DEM
    |
    v
gradient(...)
    |
    +-- grad_east
    +-- grad_north
    |
    v
application-specific
slope / aspect calculation
```

`healpix-analyse` deliberately stops at the generic derivative. Definitions,
thresholds, classification rules, and product-specific terrain logic belong
to the application or S2MSI migration layer.

### Current S2MSI documentation status

The current public Projected Geometry documentation also notes limitations
for the v1 implementation concerning DEM and DEM-derivative support.

For that reason, this section should be read as a mapping of **publicly
documented processing concepts** to generic HEALPix primitives, not as a
claim that every described DEM-dependent branch is currently operational in
the released S2MSI processor.

## Example migration pattern

Suppose a projected processing stage requires the local spatial variation of
a scalar Earth Observation field.

In a projected raster representation, the result is usually expressed
relative to raster axes.

In HEALPix, use:

```python
from healpix_analyse import (
    gradient,
    gradient_magnitude,
)

grad_east, grad_north = gradient(
    values,
    cell_ids,
    refinement_level,
)

local_variation = gradient_magnitude(
    values,
    cell_ids,
    refinement_level,
)
```

The migration changes the geometric interpretation:

```text
projected raster
local x/y derivative
        |
        v
HEALPix
local geographic derivative
        |
        +-- East
        +-- North
```

The scientific/application decision that consumes the derivative remains in
the S2MSI processing layer.

## Public-documentation mapping summary

| Public S2MSI processing concept | Generic HEALPix building block | Responsibility kept outside `healpix-analyse` |
|---|---|---|
| Local spatial variation of a scalar EO field | `gradient()` | Product-specific interpretation |
| Direction-independent local derivative magnitude | `gradient_magnitude()` | Classification threshold or rule |
| Derivative along a geographic direction | `directional_derivative()` | Direction selection and product logic |
| DEM first derivatives used to construct terrain quantities | `gradient()` | Slope/aspect/hillshade definitions and S2MSI logic |
| Scene Classification processing | May consume generic derivative products where needed | SCL classes, thresholds, confidence logic, orchestration |
| Physical mask buffering | Morphology API, not gradient API | S2MSI-specific buffer parameters |
| Second derivatives | Separate mathematical operator, not `gradient()` | Product-specific second-derivative usage |

## Public references

The following public S2MSI pages are useful starting points for understanding
where generic HEALPix spatial-analysis primitives may fit into the processing
architecture:

1. Sentinel-2 MSI Processor documentation  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/index.html

2. Detailed Processing Model — Projected Geometry Processor  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html

3. All Projected Geometry Processing Unit  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/pu_all_projected_geometry.html

4. Projected Geometry Processor description  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/ppb/projected_geometry.html

5. Projected Geometry Interface Control Document  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/icd.html

6. Scene Classification Processor API  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html

7. Context and Terminology  
   https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/context.html

## Spatial edge cases

The implementation is explicitly tested for:

- HEALPix base-pixel boundaries;
- longitude wrap-around;
- equatorial regions;
- high northern latitudes;
- high southern latitudes;
- multiple HEALPix refinement levels;
- regional processing-domain boundaries;
- missing scalar values.

These cases do not require Cartesian boundary handling.

## Current scope

The current API implements an immediate-neighbour local gradient using an
unweighted least-squares fit.

The following are intentionally outside the initial gradient API:

- physical-radius gradient neighbourhoods;
- Gaussian weighting;
- anisotropic kernels;
- Cartesian raster kernel coefficients;
- second-derivative operators.

Physical-radius and weighted local operations belong to separate filtering
APIs.

Second-derivative operators should likewise be treated as distinct
mathematical operators rather than as gradient options.
