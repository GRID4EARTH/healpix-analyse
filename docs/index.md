# healpix-analyse

Signal analysis on HEALPix grids, for Earth Observation.

`healpix-analyse` answers the question *what do you do with the values stored in
HEALPix cells*: spherical harmonic transforms and power spectra, convolutions
and multiscale pyramids, local flat-sky FFTs, neighbourhood filters, morphology,
and foundation-model embeddings. Its companion
[healpix-geo](https://healpix-geo.readthedocs.io/) answers the other question —
*where* those cells are.

Every operator is written in PyTorch, accepts NumPy arrays or tensors, works on
the full sphere or on a partial-sky set of NESTED cells, and is differentiable
through `torch.autograd`.

```bash
pip install git+https://github.com/GRID4EARTH/healpix-analyse.git
```

## Start here

1. {doc}`installation` — install it, and the extras for the examples.
2. {doc}`overview` — what each module is for, in one line each.
3. Pick your task in the table below.

## Find the page you need

| You want to… | Use | Page |
|---|---|---|
| Compute a power spectrum on the full sphere | `HEALPixSHT` | {doc}`healpix_sht` |
| …on ERA5, a lat/lon grid, or any iso-latitude grid | `alm_latlon` | {doc}`alm_latlon_1_quickstart` |
| Understand the SHT conventions and quadrature | — | {doc}`alm_latlon_2_mathematics` |
| Convolve a map, equivariantly, on the sphere | `HealPixConv` | {doc}`convol_doc` |
| Split a flow into divergence and curl | `uv_to_curl_div` | {doc}`healpix_sht` |
| Convolve with a large kernel, cheaply | `LargeConv` | {doc}`large_conv` |
| Change resolution (coarser / finer) | `HealPixDown`, `HealPixUp` | {doc}`down`, {doc}`up` |
| Build an exactly invertible multiscale pyramid | `HealPixDecomp` | {doc}`decomp` |
| Get divergence and curl at every scale | `divcurl` | {doc}`divcurl` |
| Move data between HEALPix levels or domains | `resample` | {doc}`resample_healpix` |
| FFT a local patch as if it were flat | `LocalFFT` | {doc}`fft_local` |
| Convolve a patch with a big kernel, via FFT | `HealPixFFTConv` | {doc}`fft_conv` |
| Average / take the median over a physical radius | `neighbour_reduce` | {doc}`neighbour_reduce` |
| Filter by metric distance, or with a Gaussian | `radial_filter` | {doc}`radial_filter` |
| Filter by azimuth (sun, shadow, wind) | `directional_filter` | {doc}`directional_filter` |
| Take East/North gradients | `gradient` | {doc}`gradient` |
| Dilate or erode a mask | `morphology` | {doc}`morphology` |
| Label connected regions, measure their area | `components` | {doc}`components` |
| Measure area, perimeter, Euler characteristic | `minkowski` | {doc}`minkowski` |
| Embed Sentinel-2 with DINOv3 | `GetDINOV3SAT` | {doc}`dino` |

For the signature of any function, see the {doc}`API reference <autoapi/index>`.

## Worked examples

- {doc}`Sentinel-2 local FFT <external_notebooks/fft_sentinel2_test>` — real
  B04/B08 reflectance, reconstruction metrics, local radial spectra.
- {doc}`notebooks/index` — the full notebook gallery.

```{toctree}
---
maxdepth: 1
caption: Getting Started
hidden: true
---
installation
overview
```

```{toctree}
---
maxdepth: 2
caption: Spherical harmonics
hidden: true
---
alm_latlon_1_quickstart
alm_latlon_2_mathematics
alm_latlon_3_api
healpix_sht
healpix_sht_api
healpix_sht_maths
```

```{toctree}
---
maxdepth: 2
caption: Convolution & multi-resolution
hidden: true
---
convol_doc
convol_api
convol_internals
large_conv
down
up
decomp
divcurl
resample_healpix
```

```{toctree}
---
maxdepth: 2
caption: Local flat-sky analysis
hidden: true
---
fft_local
fft_conv
```

```{toctree}
---
maxdepth: 2
caption: Filters, morphology & topology
hidden: true
---
neighbour_reduce
radial_filter
directional_filter
directional_filter_migration
gradient
morphology
components
minkowski
```

```{toctree}
---
maxdepth: 2
caption: Foundation-model embeddings
hidden: true
---
dino
```

```{toctree}
---
maxdepth: 1
caption: Notebooks
hidden: true
---
notebooks/index
```

```{toctree}
---
maxdepth: 1
caption: API Reference
hidden: true
---
autoapi/index
```

```{toctree}
---
maxdepth: 1
caption: About
hidden: true
---
changelog
license
```
