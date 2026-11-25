# Sentinel-2 MSI migration notes: radial filtering

Moved from the user guide of {doc}`/conv/radial_filter`. These notes relate the
projected-raster smoothing of the Sentinel-2 MSI processing chain to the
metric filters of this package.

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
