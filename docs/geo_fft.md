# FFT convolution on the face lattice with a geodesically sampled kernel

`HealPixGeoFFT` convolves a block of NESTED HEALPix cells that lies inside one
base face with a wide kernel, by a planar FFT on the face lattice, **without
reprojecting the data**. The kernel is sampled on the geodesic distances and
azimuths from the block centre, so that the distortion of the HEALPix cells is
built into the kernel image. Missing data are handled by normalized
convolution, computed as a ratio of two FFTs.

```text
HEALPix block → image on the (i, j) face lattice (zeros where missing)
             → zero-padded FFT convolution by k(geodesic distance, azimuth)
             → ratio by the convolved mask → values at the cells
```

Typical use: a Sentinel-2 tile at level 20 (6 m cells), a wide kernel in
metres (adjacency effect, exponential or Matérn covariance, Lorentzian point
spread function), clouds as NaN.

## Quick start

```python
import numpy as np
from healpix_analyse import HealPixGeoFFT

conv = HealPixGeoFFT(level=20, cell_ids=cells, kernel="exponential", scale=100.0)   # metres
y = conv(x)              # x: [N] values on `cells`, NaN where missing -> normalized convolution

print(conv)              # kernel, radius, window, lattice angle and steps, angular extent
conv.kernel_image        # the (2n+1, 2n+1) image actually applied
conv.support(x=x)        # K*w, the support map
```

`x` may be a NumPy array or a torch tensor, of shape `[N]`, `[B, N]` or
`[B, C, N]`; the output has the same shape and type. The operation is
differentiable with respect to `x`.

One-shot form:

```python
from healpix_analyse import geo_fft_convolve
y = geo_fft_convolve(x, cells, level=20, kernel="gaussian", scale=50.0)
```

## Why the face lattice is not a square grid

Inside a base face, the NESTED cells form a regular lattice indexed by
`(i, j)`, and it is tempting to reshape a block into an image and convolve it
with a kernel written as `k(sqrt(di**2 + dj**2))`. That kernel is **not** the
one asked for. The two lattice steps make an angle of 99° at the centre of an
equatorial face (the cells are rhombi, not squares), and in the polar caps
the lattice is a sheared parallelogram whose sides differ in length by up to
a factor two. A kernel isotropic on the sphere is, on the lattice, an ellipse
whose shape depends on the position.

Measured on 40 Sentinel-2 tiles at level 20 against the direct geodesic sum,
the square-grid kernel is off by 5–30 % of the standard deviation of the
exact result on complete tiles (median 9 %), and by up to 50 % under cloud
masks, because its distortion error does not cancel in the normalization
ratio. With the geodesically sampled kernel the same FFT agrees with the
direct sum to a fraction of a percent (the residual is a pixelisation bias on
complete data, which the normalization removes).

`lattice_geometry(level, face, i, j)` returns the local step lengths and
angle; `conv.geometry` holds them for the block centre.

## Kernels

### Predefined

| name | profile k(ρ) | extra parameters |
|---|---|---|
| `"gaussian"` | exp(−ρ²/2s²) | |
| `"exponential"` | exp(−ρ/s) | |
| `"matern"` | Matérn with ν | `nu` (default 1.5) |
| `"lorentzian"` | 1/(1+ρ²/s²) | |
| `"moffat"` | (1+ρ²/s²)^−β | `beta` (default 2.5) |
| `"voigt"` | Gaussian ⊛ Lorentzian | `gamma` (default s) |
| `"screened_poisson"` | K₀(ρ/s), core regularised | `r_core` (default 0.5 s) |
| `"top_hat"` | 1 for ρ ≤ s | |
| `"dog"` | difference of Gaussians (signed) | `ratio` (default 2) |

`s` is `scale`, in `units`. Extra parameters go in `kernel_kwargs`:

```python
HealPixGeoFFT(20, cells, "moffat", scale=30.0, kernel_kwargs={"beta": 3.0})
```

The signed `"dog"` kernel is applied raw (`normalized=False` is forced): a
normalized convolution is a weighted average and has no meaning for a kernel
that changes sign.

New families can be registered:

```python
from healpix_analyse.geo_fft import register_kernel
register_kernel("my_kernel", lambda scale, p=1.0: (lambda rho, phi: np.exp(-(rho / scale) ** p)))
```

### Any equation

`kernel` may be any callable `k(rho, phi)` evaluated on arrays: `rho` is the
geodesic distance from the block centre in `units`, `phi` the azimuth from
north (radians, clockwise). Anisotropic kernels in the geographic frame come
for free:

```python
aniso = lambda rho, phi: np.exp(-(rho * np.cos(phi) / 300.0) ** 2 - (rho * np.sin(phi) / 100.0) ** 2)
conv = HealPixGeoFFT(20, cells, aniso, radius=1000.0)      # 300 m north-south, 100 m east-west
```

The functions `kernel_gaussian`, `kernel_exponential`, … of
`healpix_analyse.kernel_pyramid` follow the same convention in pixel units
and can be passed with `units="pix"`.

### Units and truncation

`units` applies to `scale`, `radius` and to `rho` in a callable: `"m"`
(default), `"km"`, `"deg"` or `"pix"` (the cell size `sqrt(4π/N)`). Metres
and kilometres use the mean Earth radius 6 371 008.8 m, as elsewhere in the
package.

`radius` truncates the kernel; the default is the distance at which the
kernel falls below 10⁻³ of its value at the origin. The truncated kernel is
the operator applied, and the one to give to
`validation.direct_spherical_convolution` for a check. The kernel window is
sized from the local lattice geometry so that it covers the geodesic disc in
every direction: in the polar caps the disc can span three times more lattice
steps along one axis than its radius in cells.

## Domain and missing data

`cell_ids` are any cells of one base face, in any order, not necessarily
contiguous: the operator works on their bounding rectangle in base-face
coordinates, and lattice positions that are not in `cell_ids` count as
missing. Cells spanning two faces raise an error.

With `normalized=True` (default), NaN in `x` (or zero `weights`) are missing
data: the result is `K*(w x) / K*w`, the exact normalized convolution for the
kernel image used, NaN where the support `K*w` is below `support_eps` of its
maximum. This is the right form for gaps; it also removes the pixelisation
bias of the raw sum on complete data. `conv.raw(x)` gives the plain sum with
zeros at the missing cells.

## Limits, and when to use something else

The FFT applies one kernel image to the whole block: it is exact at the
centre and drifts with the variation of the cell geometry towards the edges.
On a tile of a few kilometres at level 20 that variation is negligible; on a
patch tens of degrees wide it is not, and a warning is raised above
`warn_extent_deg` (10°). In that regime, across base faces, or on the full
sphere, use the convolution pyramid (`HealPixWideConv`,
`HealPixPyramidConv`), whose cost is linear in the number of cells on any
domain, or `HEALPixSHT` for complete full-sky maps at levels a harmonic
transform can afford.

`HealPixFFTConv` is the other FFT of the package: it reprojects the patch on
a gnomonic tangent plane and is meant for learned kernels. `HealPixGeoFFT`
keeps the data on their own cells, which avoids the resampling and its error
at the pixel scale.

## API

```{eval-rst}
.. autoclass:: healpix_analyse.geo_fft.HealPixGeoFFT
   :members: forward, raw, support
.. autofunction:: healpix_analyse.geo_fft.geo_fft_convolve
.. autofunction:: healpix_analyse.geo_fft.make_kernel
.. autofunction:: healpix_analyse.geo_fft.register_kernel
.. autofunction:: healpix_analyse.geo_fft.lattice_geometry
```
