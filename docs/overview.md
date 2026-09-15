# Overview

`healpix-analyse` is a toolkit for analysing signals stored on HEALPix grids,
aimed at Earth Observation data. This page is the map: what each module is for,
and the few conventions that hold everywhere.

## What the modules do

**Spherical harmonics.** `healpix_sht` is the fast path for a full-sky HEALPix
map: a ring-based transform for spin-0, spin-1 and spin-2 fields, with power
spectra. `alm_latlon` does the same for *any* iso-latitude grid — ERA5, a
regular lat/lon grid, a Gaussian grid — so you can compare a model and an
observation without regridding either. `alm` holds the coefficients themselves.

**Convolution and scale.** `convol` (`HealPixConv`) is a gauge-equivariant
convolution: a fixed stencil transported over the sphere so that the result does
not depend on how the pixel grid happens to be oriented. `large_conv` reaches a
wide receptive field cheaply, by coarsening, convolving small, and refining
again. `down` and `up` are those two halves on their own — smooth or max-pool
coarsening, and its adjoint. `decomp` turns them into a multiscale pyramid that
reconstructs the original map exactly, masks included, and `divcurl` reads
divergence and curl off every level of that pyramid. `resample` moves data
between levels and between partial-sky domains, and to and from lat/lon.

**Local flat-sky analysis.** Over a small patch the sphere is flat enough to use
a plain 2D FFT. `fft_local` (`LocalFFT`) builds the local tangent plane and the
FFT on top of it, with CUDA and autograd; `fft_conv` uses it to convolve with
kernels too large for a stencil; `powerspectra` reduces the result to an
isotropic 1D spectrum.

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
