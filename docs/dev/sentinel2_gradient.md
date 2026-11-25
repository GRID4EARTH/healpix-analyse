# Sentinel-2 MSI / EOPF notes: gradients

Moved from the user guide of {doc}`/stats/gradient`.

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
