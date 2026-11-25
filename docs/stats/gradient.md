# HEALPix scalar-field gradients

`healpix-analyse` provides local scalar-field gradients for NESTED HEALPix
data using the real geographic geometry of neighbouring cell centres.

```{include} ../_generated/glance/gradient.md
```

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

`gradient` rebuilds this geometry at every call and takes one 1-D field at a
time. To keep it between calls, and to process batches, build the operator
once with `HealPixGradient`:

```python
from healpix_analyse import HealPixGradient

grad = HealPixGradient(cell_ids, refinement_level)   # geometry built once
grad_east, grad_north = grad(values)                 # values [..., N]
speed = grad.magnitude(values)
d_north = grad.directional_derivative(values, azimuth_rad=0.0)
```

A call returns arrays of shape `[..., len(domain)]` with the same semantics
as `gradient`. On CUDA, when the optional `triton` package is installed,
torch tensors run fused Triton kernels: they do not build the `[..., N, K]`
gathered tensor, float16 / bfloat16 data accumulate in fp32, and the backward
pass is written by hand (a gather on the transposed neighbourhoods, without
atomics, so the result is the same at every run). The values agree with the
reference implementation to rounding (about 1e-12 relative in float64).
Elsewhere (CPU, NumPy, `backend="torch"`) the reference implementation runs.

The same generic relative-geometry primitive can also support other local
operators such as radial and directional filters.

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
