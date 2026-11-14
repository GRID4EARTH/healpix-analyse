# Flat-sky power spectra: `powerspectra`, `powerspectra_lonlat`, `ps`

Three helpers compute an isotropic 1D power spectrum by binning a 2D FFT
radially. They are not spherical-harmonic transforms and should not be
confused with `sht.anafast` ({doc}`healpix_sht`) or `LocalFFT.ps`
({doc}`fft_local`), which are the production-ready spectra for, respectively,
a full-sky HEALPix map and a local gnomonic patch.

```python
from healpix_analyse.powerspectra import powerspectra
from healpix_analyse.powerspectra_lonlat import powerspectra_lonlat
from healpix_analyse.ps import ps
```

## `ps(data, data_cross=None, plot_2D_fft=False)`

A plain 2D-FFT radial power spectrum on an already-regular 2D array (`ny,
nx`). It has no notion of HEALPix, the sphere, or physical units — frequency
axes are in cycles per array index. Use it on data you have already gridded
yourself, for example the output of `LocalFFT.project`. This function is
self-contained and has no dependency on `AlmTransform`.

## `powerspectra(cell_ids, level, data, ...)` and `powerspectra_lonlat(lon, lat, data, ...)`

These compute a similar radial spectrum starting from HEALPix cells
(`powerspectra`) or from arbitrary `(lon, lat)` samples on iso-latitude rings
(`powerspectra_lonlat`). Both build an `AlmTransform` (`healpix_analyse/alm.py`)
internally and call its `.fft()` method.

### Current status: under development

`AlmTransform` is explicitly a work in progress, and `powerspectra` /
`powerspectra_lonlat` inherit its limitations:

- **`AlmTransform.ifft()` is not implemented.** It raises `NotImplementedError`
  rather than silently returning a wrong result — the Legendre-projection
  stage it needs, and the ring bookkeeping it depends on, are not built yet.
  `powerspectra`/`powerspectra_lonlat` only ever call `.fft()`, so this does
  not affect them directly, but any code trying to reconstruct a map from
  their intermediate `AlmTransform` will hit it.
- **The second transform stage is a plain FFT along latitude, not a Legendre
  sum.** `AlmTransform.fft()` does one FFT along each ring's longitude, then
  a second FFT along the ring (latitude) axis. That second step is only
  mathematically equivalent to a spherical-harmonic projection when rings are
  uniformly spaced in colatitude, which is not guaranteed in general — see the
  module's own `TODO: NUFFT or direct Legendre analysis for the second stage,
  currently only FFT stage implemented`. Treat the result as a flat-sky-style
  spectrum, not a calibrated angular power spectrum.
- **Only periodic boundary conditions are supported.** `fft(..., pbc=False)`
  raises `NotImplementedError`; `pbc=True` (the default) assumes the domain
  wraps, which is an approximation for a regional patch.
- **No handling of the 0/360° longitude wrap** in the pixel-shift correction
  (noted as a `TODO` in the source) — patches crossing the date line are not
  validated.
- **`method="alm"` is not implemented** — only `method="fft"` (the default)
  actually runs; `indexing_scheme` only accepts `"2D_array"` despite the
  docstring mentioning `"ring"` as a default.
- **Debug plotting is now gated, not removed.** Earlier versions of
  `AlmTransform.fft()` (and the constructor, through `compute_phase_shift`)
  called `matplotlib.pyplot.show()` unconditionally on every call, which
  opened blocking plot windows even in batch/headless use. These diagnostic
  plots are now only shown when the transform is constructed with
  `debug=True`; with the default `debug=False` (used internally by
  `powerspectra`/`powerspectra_lonlat`), no plot window opens.

### Recommendation

For a spectrum you can rely on today:

- full-sky HEALPix map → `sht.anafast` ({doc}`healpix_sht`);
- local HEALPix patch → `LocalFFT.ps` / `local_ps` ({doc}`fft_local`);
- an array you already gridded yourself → `ps` (above).

Use `powerspectra` / `powerspectra_lonlat` only where their specific
`AlmTransform`-based code path is what you are testing or extending, and keep
the limitations above in mind when interpreting the result.
