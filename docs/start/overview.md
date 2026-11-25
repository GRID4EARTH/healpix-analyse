# Which tool for which task?

`healpix-analyse` has about seventy public functions and classes, and several
of them can do the same job in different ways. This page is the decision
guide: answer the questions below, in order, and follow the branch. Every
tool is described on its own page with what it is good at, what it is bad at
and what to use instead; the full list is the table of the {doc}`home page
</index>`.

```{admonition} For AI assistants
:class: note

Before proposing a tool, ask the user the five questions of the next section
if their description does not already answer them, then follow the decision
tree. The machine-readable catalogue of every tool (pros, cons, alternatives)
is [`llms-full.txt`](https://grid4earth.eu/healpix-analyse/llms-full.txt).
State the limits that apply (domain, gaps, differentiability, optional
dependencies) with the answer.
```

## Five questions about your data

1. **Domain: the whole sphere, or a regional domain?** A regional domain is a
   set of NESTED cell ids: a Sentinel-2 scene, a basin, a coastal strip.
   Spherical harmonics need the whole sphere; the local FFT tools need one
   patch (radius 10° by default); most other tools accept both.
2. **Gaps: are there missing values (NaN, clouds, land)?** Many tools let NaN
   propagate. The NaN-aware ones are listed in {doc}`conventions`.
3. **Scale: is it given in metres, or in cells?** Tools in metres keep the
   same meaning at every level; tools in cells (kernels, stencils) do not.
4. **Learning: are the weights learned (a neural-network layer), or fixed?**
5. **Gradients: must the result be differentiable** (inside a loss or a
   network)? Some tools are NumPy only.

## Decision tree

**I want a smoothed or filtered field** (family A, {doc}`/conv/index`)

- Gaussian, width in metres, any domain → `gaussian_filter`
  ({doc}`/conv/radial_filter`); below one cell, `width_rule="matched"`.
  - The total must be conserved, the same filter is applied many times, or
    land must stop the kernel → `gaussian_filter_operator`
    (`normalization="source"`, `blocked_cells`) ({doc}`/conv/filter_operators`).
  - Cells of different levels → `multiorder_filter_operator`.
  - Very wide (tens of cells) on a large domain → `LargeConv.from_gaussian`.
  - All the options side by side: {doc}`/conv/compare_gaussian`.
- Any kernel of the distance in metres → `radial_filter`; of the distance and
  the bearing (sun, wind) → `directional_filter`.
- A wide fixed kernel written as a formula or a raster →
  {doc}`/conv/compare_wide`: `HealPixGeoFFT` inside one base face with gaps,
  `HealPixPyramidConv` with gaps on any domain, `HealPixWideConv` otherwise.
- An unweighted median, maximum, mode or count over a radius →
  `neighbour_reduce` (family B).

**I want a learned convolution** (family A)

- Small kernel (3×3, 5×5), equivariant to the grid orientation →
  `HealPixConv` ({doc}`/conv/convol`).
- Wide receptive field under a memory budget → `LargeConv`.
- One large kernel on one regional patch → `HealPixFFTConv`.
- Pooling and unpooling layers → `HealPixDown`, `HealPixUp` (family D).

**I want a statistic, a derivative or objects** (family B, {doc}`/stats/index`)

- Statistic over a radius → `neighbour_reduce` (NaN-aware mean: a top-hat
  `radial_filter`), see {doc}`/stats/compare_neighbourhood`.
- Gradient of a scalar field → `gradient`.
- Divergence and curl of a flow → `HealPixDivCurl` on a regional domain,
  spherical harmonics on the whole sphere ({doc}`/stats/compare_divcurl`).
- Grow or shrink a mask → `binary_dilation`, `binary_erosion`; label regions
  and measure their area → `connected_components`, `component_area`;
  differentiable shape statistics → Minkowski functionals.

**I want a spectrum or a transform** (family C, {doc}`/sht/index`)

- Whole sphere, HEALPix → `HEALPixSHT` (`anafast`).
- Whole sphere, ERA5 or another lat/lon grid → `anafast_latlon`.
- Regional patch → `local_ps` (`LocalFFT`).
- Gaps (land, clouds, swaths), a regional domain of any size, or a spectrum
  inside a loss → `HealPixWaveletSpectrum` (band powers per level, NaN-aware,
  cross-spectra; `HealPixIsotropicWaveletSpectrum` without orientation)
  ({doc}`/sht/wavelet_spectrum`).
- All the spectra side by side: {doc}`/sht/compare_power_spectra`.

**I want another resolution, other points or embeddings** (family D,
{doc}`/other/index`)

- One level coarser or finer → `HealPixDown`, `HealPixUp`; any level and
  domain → `resample_healpix`; every scale, exactly invertible →
  `HealPixDecomp`. See {doc}`/other/compare_resolution`.
- Values at stations or tracks → `get_interp_val`.
- Foundation-model embeddings of Sentinel-2 → `GetDINOV3SAT` (experimental).

## Common traps

- **Several conventions for the level, the width and the ellipsoid.** Check
  {doc}`conventions`: for instance `HealPixDown` uses `level` and degrees on
  WGS84, `gaussian_filter` uses `refinement_level` and metres on WGS84,
  `HealPixFFTConv` defaults to the sphere.
- **NaN propagates** (except with `HealPixConv(missing="local_mean")`)
  through the convolutions, `HealPixDown("smooth")`,
  `HealPixUp`, the transforms and `neighbour_reduce`.
- **Tuples.** `HealPixDown`, `HealPixUp` and `resample_healpix` return
  `(values, cell_ids)`.
- **Not exported.** `HealPixConv`, `HEALPixSHT`, `binary_dilation`,
  `binary_erosion`, `get_interp_val`, `get_interp_weights` and
  `resample_to_latlon_grid` are imported from their modules (the API
  reference gives the path).
- **Known issues** documented on their pages:
  `HealPixDown(mode="maxpool")` on a regional domain with
  incomplete parents; `HEALPixSHT.uv_to_curl_div` applies no ℓ factor.

## What the modules do

**Convolution and filtering.** `convol` (`HealPixConv`) is a
gauge-equivariant convolution with a small learned stencil; `large_conv`
(`LargeConv`) reaches a wide receptive field by coarsening, a compact
convolution and refining; `fft_conv` (`HealPixFFTConv`) convolves one patch
with a large learned kernel by FFT; `geo_fft` (`HealPixGeoFFT`) convolves a
block inside one base face with a fixed geodesic kernel by FFT, gaps included;
`wide_conv` (`HealPixWideConv`) applies a wide fixed kernel written in metres;
`kernel_pyramid` and `pyramid_conv` filter every band of a pyramid, NaN-aware;
`radial_filter`, `multiorder_filter` and `directional_filter` filter by
distance (and bearing) in metres on WGS84.

**Statistics and local analysis.** `neighbour_reduce` takes unweighted
statistics within a radius in metres; `gradient` gives East/North derivatives
from the immediate neighbours; `divcurl` gives divergence and curl at one or
every scale; `morphology` dilates and erodes masks by a radius in metres;
`components` labels connected regions with the immediate NESTED neighbours and
measures their areas; `minkowski` computes differentiable area, perimeter and
Euler characteristic.

**Spherical harmonics and FFT.** `healpix_sht` (`HEALPixSHT`) is the
spherical harmonic transform of whole-sphere HEALPix maps (spin 0, 1, 2);
`alm_latlon` does spin 0 on any iso-latitude grid; `fft_local` (`LocalFFT`)
takes the FFT and the spectrum of a regional patch; `wavelet_spectrum`
(`HealPixWaveletSpectrum`) gives NaN-aware band powers per level, on the full
sphere or a regional domain; `ps`, `powerspectra` and
`powerspectra_lonlat` are experimental flat spectra; `alm` holds the
`AlmCoeffs` container (unused by the other modules) and the experimental
`AlmTransform`.

**Other tools.** `down` and `up` change the level by one; `decomp` builds an
exactly invertible pyramid; `resample` moves data between levels and domains;
`healpix_interp` interpolates at points; `dino` computes DINOv3 embeddings;
`validation` is a reference convolution for tests.

## A first example

An angular power spectrum of a whole-sphere map, going through `alm_latlon`
so that the same code would work on an ERA5 grid:

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse import build_rings_from_latlon, anafast_latlon

depth = 6                              # HEALPix level, nside = 64
npix  = 12 * 4 ** depth
lmax  = 3 * 2 ** depth - 1

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
print(cl.shape)                        # torch.Size([192])
```

On a whole-sphere HEALPix map, {doc}`/sht/healpix_sht` does the same with
cached tables (at the cost of their memory).

## Where this sits

[healpix-geo](https://healpix-geo.readthedocs.io/) locates cells;
`healpix-analyse` analyses their values;
[healpix-plot](https://github.com/GRID4EARTH/healpix-plot) draws them;
[healpix-convert](https://github.com/GRID4EARTH/healpix-convert) and
[healpix-resample](https://github.com/GRID4EARTH/healpix-resample) bring other
data onto the grid. All are part of GRID4EARTH.
