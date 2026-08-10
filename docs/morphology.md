```markdown
# Binary morphology

`healpix-analyse` provides binary mathematical morphology for masks
represented by active nested HEALPix cell IDs.

The initial API consists of:

```python
from healpix_analyse.morphology import (
    binary_dilation,
    binary_erosion,
)
```

## Structuring neighbourhoods

On a regular Cartesian raster, a disk-shaped structuring element is
usually defined by a pixel-centre distance criterion:

\[
\sqrt{\Delta x^2 + \Delta y^2} \le r.
\]

HEALPix does not have a fixed two-dimensional Cartesian neighbourhood.
Instead, `healpix-analyse` defines the structuring neighbourhood
geometrically.

Two definitions are available.

### Cell-centre neighbourhood

```python
neighbourhood="cell_center"
```

A HEALPix cell $q$ is included around a target cell $p$ when

\[
d_{\mathrm{WGS84}}
\left(
\operatorname{center}(p),
\operatorname{center}(q)
\right)
\le r.
\]

Here, `radius` is specified in metres.

This is the default neighbourhood because it most closely reproduces
the semantics of a classical disk-shaped raster structuring element:
membership is determined from the distance between cell centres.

Example:

```python
dilated = binary_dilation(
    cells,
    radius=240.0,
    refinement_level=17,
    neighbourhood="cell_center",
)
```

### Cone-coverage neighbourhood

```python
neighbourhood="cone_coverage"
```

This alternative uses:

```python
healpix_geo.nested.cone_coverage()
```

and includes HEALPix cells intersecting the requested circular region.

Therefore, a cell may be included even when its centre lies outside
the requested radius.

Conceptually:

```text
cell_center

            o   cell centre outside
        -----------------
       /                 \
      /        radius     \
     |          C          |
      \                   /
       -----------------

Only cells whose centres are inside the circle are selected.


cone_coverage

            +-------+
            |   o   |  cell centre outside
        ----|-------|----
       /    |       |    \
      /     +-------+     \
     |          C          |
      \                   /
       -----------------

The cell is also selected because it intersects the circle.
```

As a consequence,

```text
cell_center neighbourhood ⊆ cone_coverage neighbourhood
```

for the same target cell and radius.

## Sentinel-2 MSI Mask S2 example

[The Sentinel-2 MSI Mask S2 processing uses disk-shaped structuring
elements on a 60 m processing grid.](https://s2.pages.eopf.copernicus.eu/msi/s2msi/1.3.0/dpm/sensor_geometry/transition_geometry/pu_mask_s2.html?utm_source=chatgpt.com#structural-element-selection)

The original raster parameters are:

| Operation | Raster radius | Nominal physical radius |
|---|---:|---:|
| Cloud erosion | 3 pixels | 180 m |
| Snow dilation | 4 pixels | 240 m |
| Cloud dilation | 8 pixels | 480 m |

For a HEALPix implementation, the physical radii can be preserved
directly.

HEALPix refinement level 17 is close to the spatial scale of the
60 m Mask S2 processing grid.

For example:

```python
snow = binary_dilation(
    snow_cells,
    radius=240.0,
    refinement_level=17,
    neighbourhood="cell_center",
)

opaque = binary_erosion(
    opaque_cells,
    radius=180.0,
    refinement_level=17,
    neighbourhood="cell_center",
)

opaque = binary_dilation(
    opaque,
    radius=480.0,
    refinement_level=17,
    neighbourhood="cell_center",
)
```

The same applies to the cirrus mask.

## Difference between the two neighbourhood definitions

A regression test using a single HEALPix cell at refinement level 17
gives:

| Radius | `cell_center` | `cone_coverage` | Additional coverage cells |
|---:|---:|---:|---:|
| 180 m | 41 | 63 | 22 |
| 240 m | 73 | 99 | 26 |
| 480 m | 295 | 339 | 44 |

The larger values from `cone_coverage` are expected.

`cell_center` selects cells according to their centre-to-centre
distance, whereas `cone_coverage` also includes cells whose centres
are outside the circle but whose cell area intersects it.

For reproducing the S2MSI disk structuring-element semantics,
`cell_center` is therefore the recommended/default method.

## Processing domains

Binary masks are represented by their active HEALPix cell IDs.

For regional datasets such as Sentinel-2 tiles, an optional `domain`
can specify all HEALPix cells over which the mask is defined.

```python
eroded = binary_erosion(
    cells,
    radius=180.0,
    refinement_level=17,
    neighbourhood="cell_center",
    domain=domain_cells,
)
```

When a domain is supplied:

- inactive cells inside the domain are interpreted as `False`;
- cells outside the domain are outside the processing extent;
- dilation is clipped to the domain;
- erosion ignores structuring-neighbourhood cells outside the domain.

This prevents artificial erosion along the edge of a regional dataset.

## API reference

```{eval-rst}
.. autofunction:: healpix_analyse.morphology.binary_dilation
.. autofunction:: healpix_analyse.morphology.binary_erosion
```
```
