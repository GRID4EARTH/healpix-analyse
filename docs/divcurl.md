# Multiscale divergence and curl

`HealPixDivCurl` estimates horizontal divergence and outward-normal scalar
curl from eastward/northward velocity components on a NESTED HEALPix domain.
`HealPixMultiScaleDivCurl` applies the same physical operator to every native
band returned by `HealPixDecomp`.

The implementation is local, differentiable, GPU-aware, compatible with
partial masks, and built directly on `HealPixConv`.

## Quick start

```python
from healpix_analyse import HealPixDecomp, HealPixMultiScaleDivCurl

decomp = HealPixDecomp(
    level=10,
    cell_ids=ocean_cell_ids,
    Jmax=5,
    ellipsoid="sphere",
)
velocity_pyramid = decomp.compute(velocity_uv)  # [..., 2, N]

operator = HealPixMultiScaleDivCurl(
    decomp,
    kernel_sz=3,
    n_gauges=2,
    gauge_type="projected_ref",
    singularity_lonlat=(-150.0, -20.0),
)
result = operator.compute(velocity_pyramid)

div_level_10 = result.div[0]
curl_level_10 = result.curl[0]
```

Input channel zero is eastward velocity `u`; channel one is northward
velocity `v`. If velocity is expressed in metres per second, the default
output unit is inverse seconds.

## Convolutional construction

For each target cell, `HealPixConv` provides two oriented stencil axes
$\mathbf a$ and $\mathbf b$. Fixed antisymmetric derivative-of-Gaussian
kernels estimate directional derivatives along those axes.

The input components are first embedded in a global Cartesian vector:

$$
\mathbf V = u\,\mathbf e_{\rm east} + v\,\mathbf e_{\rm north}.
$$

The three Cartesian components are globally defined scalar channels. They can
therefore be interpolated by `HealPixConv` without confusing the local vector
bases of neighbouring cells. The convolution returns
$\partial_{\mathbf a}\mathbf V$ and
$\partial_{\mathbf b}\mathbf V$, after which

$$
\operatorname{div}\mathbf V =
\mathbf a\cdot\partial_{\mathbf a}\mathbf V +
\mathbf b\cdot\partial_{\mathbf b}\mathbf V,
$$

$$
\operatorname{curl}\mathbf V =
\mathbf b\cdot\partial_{\mathbf a}\mathbf V -
\mathbf a\cdot\partial_{\mathbf b}\mathbf V.
$$

The curl sign corresponds to the outward surface normal. Embedding the vector
before convolution automatically includes the spatial change of the local
east/north basis. It also makes the final scalars invariant to a simultaneous
rotation of the convolution stencil and gauge frame, up to interpolation and
finite-difference error.

Internally, one `HealPixConv` has three input and six output channels. For
each Cartesian component it applies one derivative along each gauge axis.
The derivative kernels are fixed; gradients propagate to the input velocity.

## Scale-dependent physical distance

At HEALPix level $L_j$, the characteristic angular pixel spacing is

$$
\alpha_j = \sqrt{\frac{4\pi}{12(2^{L_j})^2}},
$$

and the physical spacing used by the derivative kernel is

$$
\Delta_j = R\alpha_j.
$$

After one Down operation, the spacing doubles approximately. The same 3x3
stencil consequently probes a physical distance twice as large, while its
weights are divided by $\Delta_j$. This is what makes divergence and curl at
different pyramid levels comparable in the same physical unit.

The exact spacings are available as:

```python
print(result.levels)
print(result.pixel_spacing_m)
```

`radius_m` defaults to the IUGG mean Earth radius, 6,371,008.8 metres. The
current physical normalisation is spherical. `ellipsoid="WGS84"` can still be
used for HEALPix cell geometry, but metre-scale derivatives on an ellipsoid
remain a spherical-radius approximation.

## Derivative kernel

For local kernel coordinates `(x, y)` in native pixels and Gaussian weights
`g`, the two kernels are

$$
D_x = \frac{xg}{\Delta_j\sum x^2g},
\qquad
D_y = \frac{yg}{\Delta_j\sum y^2g}.
$$

They have zero mean, are antisymmetric and have a unit first moment in physical
coordinates. `kernel_sz` controls the support and `sigma_pix` controls the
Gaussian width. The default `sigma_pix=kernel_sz/3` gives a compact smooth
derivative.

## Gauge handling

All `HealPixConv` gauge types are accepted:

- `"phi"` for the local meridian convention;
- `"cosmo"` for the cosmological convention;
- `"projected_ref"` to place the two gauge singularities away from a region;
- `"two_ref"` for the two-reference construction.

With `n_gauges > 1`, several rotated derivative estimates are evaluated.
`gauge_reduce="mean"` averages them and normally reduces grid anisotropy.
`gauge_reduce="none"` retains shape `[..., G, 2, N]`, which is useful for
measuring orientation sensitivity.

Every smooth tangent gauge has singularities. For a regional masked dataset,
place the singularities outside the valid domain with `projected_ref`. Multiple
gauges do not remove a singularity of the base gauge.

## Single-scale API

```python
import torch
from healpix_analyse import HealPixDivCurl

layer = HealPixDivCurl(
    level=8,
    cell_ids=cell_ids,
    kernel_sz=5,
    sigma_pix=1.5,
    n_gauges=2,
    gauge_type="phi",
    ellipsoid="sphere",
    dtype=torch.float64,
    device="cuda",
)

divcurl = layer(velocity_uv)
div = divcurl[..., 0, :]
curl = divcurl[..., 1, :]
```

Arbitrary leading dimensions are accepted. NumPy input returns NumPy output;
Torch input retains its autograd graph and executes on the configured device.

## Multiscale result

`HealPixDivCurlPyramid` is list-like. Each band retains the level and exact
NESTED identifiers of its corresponding velocity band:

| Attribute | Meaning |
|---|---|
| `bands[j]` | divergence/curl at scale `j` |
| `div[j]` | divergence at scale `j` |
| `curl[j]` | curl at scale `j` |
| `levels[j]` | Grid4Earth/HEALPix level |
| `cell_ids[j]` | identifiers matching the last data dimension |
| `pixel_spacing_m[j]` | derivative distance normalisation |

The wrapper validates pyramid levels and identifiers before computing any
derivative, preventing accidental mixing of masks or resolutions.

## Masks and limitations

`HealPixConv` only uses available input cells, so the operator remains local
on ocean or coastline masks. Derivatives close to a mask boundary are less
accurate because the stencil becomes incomplete. Scientific validation should
either exclude a narrow boundary region, retain gauge-to-gauge dispersion as
an uncertainty indicator, or compare multiple kernel sizes.

The final level-zero band contains only twelve full-sky HEALPix cells and
cannot provide a precise local derivative. It is returned for completeness,
but should normally be interpreted only as a very coarse diagnostic.

This module computes diagnostic divergence and curl. Reconstructing velocity
from them is a separate Helmholtz-Hodge inversion and requires boundary
conditions on masked domains.
