# C. Spherical harmonics & FFT

Transforms and spectra: spherical harmonics on the whole sphere (HEALPix or
lat/lon grids), the local FFT of a regional patch, and power spectra. The
FFT-based *convolutions* are in family A ({doc}`/conv/compare_fft`).

## First questions

1. **Whole sphere or a regional patch?** Spherical harmonics need the whole
   sphere; on a regional patch (radius up to 10° by default) use the local
   FFT.
2. **HEALPix or another grid?** `HEALPixSHT` for a HEALPix map;
   `alm_latlon` for ERA5, regular or Gaussian lat/lon grids (and for HEALPix
   when the memory of `HEALPixSHT` is too large).
3. **Scalar or vector field?** Spin 1 and 2 are only in `HEALPixSHT`, and
   need the optional `quaternionic` and `spherical` packages.

Comparison: {doc}`compare_power_spectra`. Divergence and curl, global or
local: {doc}`/stats/compare_divcurl`.

## I want to …

```{include} ../_generated/decisions_C.md
```

```{toctree}
---
maxdepth: 1
caption: Compare
---
compare_power_spectra
```

```{toctree}
---
maxdepth: 1
caption: Spherical harmonics
---
healpix_sht
healpix_sht_maths
alm_latlon
alm_latlon_maths
alm_coeffs
```

```{toctree}
---
maxdepth: 1
caption: Local FFT and flat spectra
---
fft_local
powerspectra
```
