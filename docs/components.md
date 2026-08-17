# HEALPix connected components

`healpix-analyse` provides connected-component analysis for binary fields
defined on NESTED HEALPix cells.

## Basic usage

```python
from healpix_analyse import connected_components

labels, n_components = connected_components(
    mask,
    cell_ids,
    refinement_level,
    connectivity="edge",
)
```

Background cells receive label `0`.

Foreground components receive deterministic labels:

```text
1, 2, 3, ...
```

Component numbering follows the first active cell encountered in the
processing-domain order.

## Connectivity

Two HEALPix connectivity definitions are supported.

### Edge connectivity

```python
connectivity="edge"
```

Cells are connected only when they share a HEALPix cell edge.

This is the HEALPix analogue of Cartesian 4-connectivity.

For a Cartesian raster:

```text
. X .
X C X
. X .
```

For the HEALPix topology, the four edge-sharing directional neighbours
are:

```text
SW, NW, NE, SE
```

### Edge-or-vertex connectivity

```python
connectivity="edge_or_vertex"
```

Cells are connected when they share either a HEALPix edge or a HEALPix
vertex.

This is the HEALPix analogue of Cartesian 8-connectivity.

For a Cartesian raster:

```text
X X X
X C X
X X X
```

For the HEALPix topology, the full immediate neighbourhood contains
up to eight directional positions:

```text
SW, W, NW, N, NE, E, SE, S
```

At HEALPix topological special locations, one of the vertex-touching
positions may be absent, giving seven distinct immediate neighbours.

## NESTED indexing only

The current implementation supports NESTED HEALPix indexing only.

RING indexing and automatic indexing-scheme detection are intentionally
not supported.

## Processing domain

`cell_ids` defines the cells for which input values are available.

`domain` defines the cells that participate in the connected-component
graph.

```python
labels, n_components = connected_components(
    mask,
    cell_ids,
    refinement_level,
    domain=domain,
)
```

`domain` must be a subset of `cell_ids`.

Cells outside `domain` are absent from the topology.

For example:

```text
A -- X -- B
```

if `X` lies outside the domain, connectivity cannot pass through `X`,
even if its input mask value is active.

Output order follows the exact order supplied by `domain`.

## Component size

```python
from healpix_analyse import component_size

sizes = component_size(labels)
```

The returned array is indexed by component label.

For:

```text
labels = [1, 1, 0, 2, 2, 2]
```

the result is:

```text
sizes = [0, 2, 3]
```

Background label `0` is assigned size zero.

## Component area

```python
from healpix_analyse import component_area

areas = component_area(
    labels,
    refinement_level,
    ellipsoid="WGS84",
)
```

HEALPix cells at one refinement level have equal area.

The component area is therefore:

```text
component cell count × HEALPix cell area
```

For the Sentinel-2 HEALPix processing use case, WGS84 is the intended
ellipsoid.

The area computation uses the WGS84 ellipsoid surface area rather than
an independently selected spherical-Earth radius.

## Remove small components

Components can be filtered by cell count:

```python
from healpix_analyse import remove_small_components

cleaned = remove_small_components(
    mask,
    cell_ids,
    refinement_level,
    min_cells=4,
    connectivity="edge",
)
```

or by physical area:

```python
cleaned = remove_small_components(
    mask,
    cell_ids,
    refinement_level,
    min_area_m2=10_000.0,
    connectivity="edge",
    ellipsoid="WGS84",
)
```

Exactly one of `min_cells` or `min_area_m2` must be supplied.


## Threshold semantics

Component thresholds are inclusive.

```text
component size >= threshold  -> retained
component size < threshold   -> removed
```

For example, a two-cell component is retained when:

```python
cleaned = remove_small_components(
    mask,
    cell_ids,
    refinement_level,
    min_cells=2,
)
```

The same rule applies to physical-area thresholds.

Background label `0` always remains background, including when the selected threshold is zero.

## Generic Earth Observation example

Connected-component filtering is useful as a generic post-processing step for binary Earth Observation segmentations.

For example, small isolated regions can be removed from a segmentation mask using a cell-count threshold:

```python
from healpix_analyse import remove_small_components

cleaned = remove_small_components(
    segmentation_mask,
    cell_ids,
    refinement_level,
    min_cells=4,
    connectivity="edge",
)
```

The same operation can use a physical-area threshold instead:

```python
cleaned = remove_small_components(
    segmentation_mask,
    cell_ids,
    refinement_level,
    min_area_m2=100_000.0,
    connectivity="edge",
    ellipsoid="WGS84",
)
```

This pattern is generic and can be applied to cloud, snow or ice, water, land-cover, habitat, or other binary segmentation masks.

The library itself does not assign application-specific meanings or class identifiers to the foreground mask.

## Topological edge cases

Connectivity is defined by HEALPix topology rather than geographic coordinate proximity.

As a consequence, connected components work continuously across:

- HEALPix base-pixel boundaries,
- polar regions,
- the longitude coordinate wrap-around,
- and other HEALPix topological special locations.

No special treatment of the longitude seam or poles is required by the caller.

A component may also touch the boundary of a partial processing domain. Cells outside the domain are simply absent from the connectivity graph; valid foreground cells on the domain boundary remain part of their component.

Background cells may form holes inside a connected foreground region. A hole remains background and does not split the surrounding foreground when an alternative topological path connects that foreground.


## NumPy and PyTorch

NumPy arrays and PyTorch tensors are supported.

For Torch input, outputs are returned on the original device.

Connected-component operations are discrete and are therefore not
differentiable.

## Topology backend

Immediate HEALPix topology is currently obtained through a private
`healpix_analyse._topology` helper.

That helper uses the direction-preserving API:

```python
healpix_geo.nested.neighbours(...)
```

The topology architecture is:

```text
healpix-analyse
      |
      v
private topology helper
      |
      v
healpix-geo
      |
      v
CDSHEALPix
```

The private adapter preserves the connected-component topology contract while
`healpix-geo` provides deterministic directional positions and the `-1`
sentinel for missing positions. The public connected-component API is
unchanged.

## Topology performance

The `healpix-geo` backend reduces the cost of immediate-neighbour lookup for
large arrays. At depth 12 in the checked-in reference benchmark, the new
adapter is faster than the previous `healpy` adapter from 10,000 cells onward,
with the largest gains observed for one million cells.

These timings are machine-dependent reference measurements, not a performance
guarantee. The adapter uses `num_threads=0`, so `healpix-geo` may select its
automatic parallel path for sufficiently large inputs. The benchmark therefore
also reports a one-thread backend measurement to make the contribution from
parallel execution visible.

The executable benchmark, measurement procedure, exact environment, input
dtype and reference results are maintained in the repository's
[`benchmarks` directory](https://github.com/GRID4EARTH/healpix-analyse/tree/main/benchmarks).
