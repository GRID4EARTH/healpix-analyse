# Directional filtering

`directional_filter()` filters a scalar field on a NESTED HEALPix grid with a
kernel that depends on **how far** and **in which compass direction** a
neighbour lies.

Both are physical: distance in metres, direction as a geographical azimuth,
geometry computed between cell centres on WGS84. Nothing in the public API
refers to pixel offsets, HEALPix ring numbers, or neighbour-list ordering — so
the result does not change when the grid's local orientation does.

Use it when the direction itself means something on the ground: solar azimuth,
shadow displacement, illumination, wind, transport.

```python
from healpix_analyse.directional_filter import directional_filter
```

## A first example

Everything within 600 m, to the southeast, within ±20° of that bearing:

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse.directional_filter import directional_filter

level = 14                                     # ~2.4 m cells
cell_ids = nested.cone_coverage((2.35, 48.85), 0.02, level)   # a patch over Paris
cell_ids = np.asarray(getattr(cell_ids, "data", cell_ids), dtype=np.uint64).reshape(-1)
values = np.random.rand(cell_ids.size)

def forward_sector(distance_m, relative_bearing_rad):
    """1 inside a ±20° sector around the requested direction, 0 outside."""
    return (np.abs(relative_bearing_rad) <= np.deg2rad(20.0)).astype(float)

out = directional_filter(
    values, cell_ids, level,
    max_distance_m=600.0,
    azimuth_rad=np.deg2rad(135.0),             # 135° = southeast
    kernel=forward_sector,
    normalize=True,
)
print(out.shape)                               # (len(cell_ids),)
```

## The azimuth convention

Angles increase **clockwise from geographic North**, the ordinary compass
convention. Values outside `[0, 2π)` are wrapped for you.

```text
             North 0
                ^
                |
  West 3π/2 <---+---> π/2 East
                |
                v
             South π
```

This is deliberately geographical. It is not the orientation of an array's
rows and columns, nor the order in which a topology routine returns neighbours.

## What the kernel receives

For every target cell `i` and each neighbour `j` within `max_distance_m`:

```text
distance_ij          WGS84 geodesic distance, centre to centre
bearing_ij           WGS84 forward azimuth, from i towards j
relative_bearing_ij  wrap(bearing_ij - azimuth_rad), in [-π, +π)
```

so `relative_bearing = 0` means *j lies exactly in the requested direction*,
and `±π/2` means ninety degrees off it, clockwise or anticlockwise.

The bearing runs **from the output cell towards the contributing neighbour**.
Reversing it would turn an asymmetric kernel through 180°, which is not the
same operation.

Your kernel is called once, with one-dimensional NumPy arrays holding every
valid pair, and returns a scalar or an array broadcastable to them:

```python
kernel(distance_m, relative_bearing_rad) -> weight
```

A cell is its own neighbour at zero distance, where a bearing is undefined; the
convention is `distance_m = 0`, `relative_bearing_rad = 0`. Whether that centre
value counts is your kernel's decision — a purely angular kernel keeps it, one
that requires a positive distance drops it.

## Parameters

| Parameter | Meaning |
|---|---|
| `values` | The signal. Its **last** dimension matches `cell_ids`; leading dimensions are preserved, so `(N,)`, `(bands, N)` and `(time, bands, N)` all work. NumPy or Torch. |
| `cell_ids` | 1-D array of unique NESTED cell ids for which values exist. |
| `refinement_level` | HEALPix level of those cells. |
| `max_distance_m` | Physical support: a cell farther than this never contributes. |
| `azimuth_rad` | The requested direction, radians clockwise from North. |
| `kernel` | `kernel(distance_m, relative_bearing_rad) -> weight`, as above. |
| `normalize` | `False` (default) returns `Σ w·v`; `True` returns `Σ w·v / Σ w`. |
| `domain` | Optional subset of `cell_ids` — see below. Defaults to `cell_ids`. |
| `ellipsoid` | Reference ellipsoid for the geometry; `"WGS84"`. |

Returns an array of shape `values.shape[:-1] + (len(domain),)`, of the same
kind as the input. Torch input stays on its device and stays differentiable
with respect to `values`; the geometry and the kernel weights are constants.

## Domain: which cells take part, and in what order

`cell_ids` says where values exist. `domain` says which cells participate
**and** which cells come out, in that exact order.

A cell present in `cell_ids` but outside `domain` simply does not contribute.
It is not zero, not NaN padding, not a wrapped index — it is absent. That is
what lets you process a regional subset without inventing boundary values:

```python
out = directional_filter(
    values, cell_ids, level,
    max_distance_m=300.0, azimuth_rad=np.deg2rad(90.0),
    kernel=forward_sector,
    domain=domain,                 # subset of cell_ids; out has len(domain)
)
```

## Normalisation and missing data

NaN samples are treated as observations that are missing, not as poison: their
weights are dropped from both the numerator and the denominator rather than
propagating a NaN across the whole neighbourhood.

With `normalize=True`, an output cell whose effective weights all vanish is
NaN — there was nothing to average.

## Kernel recipes

An angular sector, ignoring distance:

```python
def sector(distance_m, relative_bearing_rad, half_width_deg=20.0):
    return (np.abs(relative_bearing_rad) <= np.deg2rad(half_width_deg)).astype(float)
```

A smooth cone — Gaussian in distance and in angle:

```python
def directional_gaussian(distance_m, relative_bearing_rad,
                         sigma_m=200.0, sigma_deg=15.0):
    radial = np.exp(-0.5 * (distance_m / sigma_m) ** 2)
    angular = np.exp(-0.5 * (relative_bearing_rad / np.deg2rad(sigma_deg)) ** 2)
    return radial * angular
```

An annulus at an expected displacement — "look about 400 m away, in this
direction, ±10°", which is how a cast shadow is actually specified:

```python
def displaced(distance_m, relative_bearing_rad,
              target_m=400.0, tol_m=50.0, tol_deg=10.0):
    return ((np.abs(distance_m - target_m) <= tol_m)
            & (np.abs(relative_bearing_rad) <= np.deg2rad(tol_deg))).astype(float)
```

`max_distance_m` fixes the support; the kernel shapes what happens inside it.
Keep the two consistent — a kernel peaking at 400 m under a 300 m support
returns zeros.

## Which operator do I want?

| Weight depends on | Use |
|---|---|
| nothing (plain mean, median, extrema, counts) | {doc}`neighbour_reduce` |
| distance only | {doc}`radial_filter` |
| distance **and** geographical bearing | `directional_filter()` |
| a learned stencil transported over the sphere | `HealPixConv`, {doc}`convol_doc` |

The distinction with `HealPixConv` is the one that matters. `HealPixConv` is a
gauge-equivariant convolution: a stencil rotated into a local frame, the right
tool when you want convolutional semantics. `directional_filter()` builds its
weights from real WGS84 geometry, and is the right tool when the direction has
a physical meaning you could point to on a map.

`radial_filter` and `directional_filter` share the neighbourhood search and the
weighted aggregation (`_weighted_neighbourhood.py`); only their spatial
semantics differ. Do not express a directional kernel as a categorical
neighbourhood reduction — the weights are the point.

## Why physical units rather than pixel windows

Specifying `3 × 3 pixels` or `HEALPix ring = 2` ties the meaning of a filter to
the layout of the grid. On HEALPix, cell shape varies with latitude and the
neighbour ordering carries no compass information at all, so such a filter
means something different in Brittany and in Lapland.

Stating the support as "no farther than this many metres" and the direction as
a compass bearing keeps the operation invariant to base-pixel boundaries,
longitude wrapping and local orientation — which is the whole reason this
function exists rather than a neighbour-index shift.

That argument has a concrete origin: migrating Sentinel-2 L2A cloud-shadow and
terrain processing off UTM rasters. See
{doc}`directional_filter_migration`.
