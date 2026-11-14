# Overview

`healpix-analyse` is a toolkit for analysing signals stored on HEALPix grids,
aimed at Earth Observation data. This page is the map: what each module is for,
and the few conventions that hold everywhere.

## What the modules do

**Spherical harmonics.** `healpix_sht` is the fast path for a full-sky HEALPix
map: a ring-based transform for spin-0, spin-1 and spin-2 fields, with power
spectra. `alm_latlon` does the same for *any* iso-latitude grid — ERA5, a
regular lat/lon grid, a Gaussian grid — so you can compare a model and an
observation without regridding either. `alm` holds the coefficients themselves
(`AlmCoeffs`); the same module also defines `AlmTransform`, a local FFT-based
SHT that `powerspectra`/`powerspectra_lonlat` build on — see {doc}`powerspectra`
for what is production-ready today and what is still in development.

**Convolution and scale.** Three convolution operators cover different regimes,
and the choice matters more than it looks — see {ref}`choosing-a-convolution`
below for the full comparison:

- `convol` (`HealPixConv`) is a gauge-equivariant convolution: a small, fixed
  stencil transported over the sphere so that the result does not depend on
  how the pixel grid happens to be oriented. Best for small learned kernels
  (3×3, 5×5) on the full sphere or a patch.
- `fft_conv` (`HealPixFFTConv`) evaluates a large learned kernel by FFT on one
  local gnomonic patch (it reuses `fft_local`). Not gauge-equivariant; the
  right tool when the kernel itself is large and the domain is one patch.
- `large_conv` (`LargeConv`) reaches a wide receptive field cheaply, by
  coarsening (`down`), convolving a small compact kernel, and refining again
  (`up`). The right tool for wide spherical context under a fixed memory
  budget.

`down` and `up` are the two halves of that coarsening on their own — smooth or
max-pool coarsening, and its adjoint — and are also useful standalone in a
U-Net encoder/decoder. `decomp` turns them into a multiscale pyramid that
reconstructs the original map exactly, masks included, and `divcurl` reads
divergence and curl off every level of that pyramid. `resample` moves data
between levels and between partial-sky domains, and to and from lat/lon.

`kernel_pyramid`/`pyramid_conv` (`HealPixKernelPyramid`, `HealPixPyramidConv`)
add a fourth, NaN/weight-aware option: one small, fixed `HealPixConv` kernel
per `decomp` band, applied identically to a data and a confidence channel and
combined with a single division after synthesis. This is a **block-diagonal**
approximation of the true (inter-band-coupled) target operator — see
{doc}`pyramid_convolution` for exactly what that means, its measured accuracy,
and what it does not do.

**Point interpolation.** `healpix_interp` bilinearly interpolates a map (or
just returns cells and weights) at arbitrary lon/lat. For `ellipsoid="sphere"`
it is a direct NumPy port of `healpy`'s own RING-scheme algorithm, validated
to return identical pixels and float64-precision-identical weights — see
{doc}`healpix_interp` for the exact-equivalence guarantee and how it was
checked. Non-spherical ellipsoids (`WGS84`, etc., no `healpy` equivalent) go
through `healpix_geo.nested.bilinear_interpolation` instead.

**Local flat-sky analysis.** Over a small patch the sphere is flat enough to use
a plain 2D FFT. `fft_local` (`LocalFFT`) builds the local tangent plane and the
FFT on top of it, with CUDA and autograd, and its own `.ps()` method reduces
the result to an isotropic 1D spectrum; `fft_conv` reuses the same projection
to convolve with kernels too large for a stencil. `powerspectra` and
`powerspectra_lonlat` offer a similar 1D spectrum through a different,
still-developing code path — see {doc}`powerspectra` before choosing between
them.

**Neighbourhood operators.** These work in *metres on WGS84*, not in pixels, so
their meaning does not drift with latitude. `neighbour_reduce` takes means,
medians, extrema and counts over a physical radius; `radial_filter` weights by
distance (including a Gaussian); `directional_filter` weights by azimuth;
`gradient` gives East/North derivatives and directional derivatives.

**Morphology and topology.** `morphology` dilates and erodes masks on the NESTED
neighbour graph; `components` labels connected regions and measures their area;
`minkowski` computes area, perimeter and Euler characteristic, differentiably,
for single or multiple thresholds.

**Foundation-model embeddings.** `dino` (`GetDINOV3SAT`) runs an unmodified
DINOv3 SAT-493M backbone on HEALPix data and maps every token back to a HEALPix
cell. See {doc}`dino`, and read its licence section before publishing results.

(choosing-a-convolution)=
## Choosing a convolution

| Need | Use | Domain | Learned weights | Gauge-equivariant | Typical cost |
|---|---|---|---|---|---|
| Small learned kernel (3×3, 5×5), e.g. one layer of a spherical U-Net | `HealPixConv` | Full sphere or patch | Yes | Yes | O(K·G·kernel²) — expensive at high nside / kernel size / gauge count |
| Large learned kernel on one local image (e.g. a wide filter on a Sentinel-2 tile) | `HealPixFFTConv` | One local patch (≤10° by default) | Yes | No | Fast for large kernels; not worth it for small ones |
| Wide receptive field under a memory budget, full sphere or a large patch | `LargeConv` | Full sphere or large patch | Yes (compact kernel) | Yes (internal `HealPixConv`) | O(K/4^L · compact_kernel²) — cheap |
| Smoothing / pooling with no learned weights, pyramid building block | `HealPixDown` / `HealPixUp` | Full sphere or patch | No | n/a | Sparse operator, cheap |
| Masked/NaN-aware multiscale filtering (block-diagonal per-band kernel) | `HealPixKernelPyramid` + `HealPixPyramidConv` | Full sphere or patch | Optional (analytic or calibrated) | Yes (per band) | O(K·compact_kernel²) per band |
| Isotropic physical filter, scale given in metres | `radial_filter` / `gaussian_filter` | Full sphere or patch | No (user kernel, fixed) | n/a | Depends on neighbourhood size |
| Directional physical filter (distance + real-world bearing) | `directional_filter` | Full sphere or patch | No | n/a | Same |
| Unweighted local reduction (mean / median / min / max / count) | `neighbour_reduce` | Full sphere or patch | No | n/a | Cheapest |

Rule of thumb: need learned weights and equivariance → `HealPixConv` (small
kernel) or `LargeConv` (wide context); need a large explicit learned kernel on
one local patch → `HealPixFFTConv`; no weights to learn, just a physical scale
→ `radial_filter` / `directional_filter` / `neighbour_reduce`.

## Conventions that hold everywhere

**Differentiable by default.** The hot paths are `torch.fft`, `einsum`,
`index_select` and sparse products — all supported by autograd. Geometry tables
(Legendre polynomials, interpolation weights, phase matrices) are computed once
and stored as non-gradient buffers.

**NumPy or Torch, single map or batch.** Every operator accepts `np.ndarray` and
`torch.Tensor` and returns the same type it was given. Shapes `[N]` and `[B, N]`
both work.

**Full-sky or partial-sky.** Operators take either a complete map or an
arbitrary set of NESTED cell ids, so you can work on one scene without padding
it out to the whole sphere.

**Geometry comes from `healpix-geo`.** Cell centres, vertices, neighbours and
ellipsoidal distances are the GRID4EARTH library's, not `healpy`'s.

**One SHT convention**, orthonormal, the same one `healpy` and the CMB
literature use:

$$a_{\ell m} = \int f(\theta, \varphi)\, Y_{\ell m}^*(\theta, \varphi)\, d\Omega$$

$$C_\ell = \frac{1}{2\ell+1} \left[ |a_{\ell 0}|^2 + 2 \sum_{m=1}^{\ell} |a_{\ell m}|^2 \right]$$

## A first example

An angular power spectrum of a full-sky map, going through `alm_latlon` so that
the same code would work on an ERA5 grid:

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

depth = 6                              # HEALPix level, nside = 64
npix  = 12 * 4 ** depth
lmax  = 3 * 2 ** depth

im = np.random.randn(npix)             # the map to analyse, NESTED order

# Where the cells are: healpix-geo returns degrees, the transform wants radians
lon, lat = nested.healpix_to_lonlat(np.arange(npix, dtype=np.uint64), depth)
theta, phi = np.radians(90.0 - lat), np.radians(lon)

# Group the cells into iso-latitude rings, then transform
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)
cl = anafast_latlon(
    im[sort_idx], ring_theta, ring_phi_list, ring_counts,
    lmax=lmax, quadrature="equal_area",
)
print(cl.shape)                        # torch.Size([193])
```

On a full-sky HEALPix map specifically, {doc}`healpix_sht` does this faster.

## Where this sits

[healpix-geo](https://healpix-geo.readthedocs.io/) locates cells;
`healpix-analyse` analyses their values;
[healpix-plot](https://github.com/GRID4EARTH/healpix-plot) draws them;
[healpix-convert](https://github.com/GRID4EARTH/healpix-convert) and
[healpix-resample](https://github.com/GRID4EARTH/healpix-resample) bring other
data onto the grid. All are part of GRID4EARTH.
