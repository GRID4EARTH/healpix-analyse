# Migrating a raster median filter to HEALPix

Moved from the user guide of {doc}`/stats/neighbour_reduce`.

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
