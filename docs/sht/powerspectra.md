# Flat power spectrum (deprecated): `powerspectra`

`powerspectra` computes an isotropic 1-D power spectrum by binning a 2-D FFT
of a rectangular block of HEALPix cells radially. It is not a
spherical-harmonic transform and should not be confused with
`HEALPixSHT.anafast` ({doc}`/sht/healpix_sht`), `anafast_latlon`
({doc}`/sht/alm_latlon`), the wavelet spectrum ({doc}`/sht/wavelet_spectrum`)
or `LocalFFT.ps` ({doc}`/sht/fft_local`).

```{include} ../_generated/glance/flat_spectra.md
```

```{warning}
**Deprecated.** `powerspectra` emits a `DeprecationWarning` and will be
removed in a future release. It is not exported from `healpix_analyse`, it
is not differentiable (it bins in NumPy), and the harmonic path of the
`AlmTransform` it is built on is not implemented. Use instead:

- the full sphere → `HEALPixSHT.anafast` ({doc}`/sht/healpix_sht`);
- a regional domain, or a map with missing data → `HealPixWaveletSpectrum` /
  `HealPixIsotropicWaveletSpectrum` ({doc}`/sht/wavelet_spectrum`), or
  `local_ps` for one gap-free patch ({doc}`/sht/fft_local`);
- a lon/lat or Gaussian grid → `build_rings_from_latlon` + `anafast_latlon`
  ({doc}`/sht/alm_latlon`).
```

```python
from healpix_analyse.powerspectra import powerspectra   # DeprecationWarning

f, pk = powerspectra(cell_ids, level, data, dx=1.0)
```

`cell_ids` is a **2-D** array of RING cell ids (`ny, nx`), one row per
iso-latitude ring, and `data` has the same shape. The function returns the
bin centres in cycles per `dx` and the spectrum averaged over
`min(nx, ny) // 2` linear radial bins, normalised by `nx · ny`.

## Limits

`powerspectra` builds an `AlmTransform` (`healpix_analyse/alm.py`) and calls
its `.fft()` method, so it inherits its limits:

- **The second stage is a plain FFT along latitude, not a Legendre sum.**
  `AlmTransform.fft()` does one FFT along each ring, then a second FFT across
  the rings. That is a spherical-harmonic projection only when the rings are
  uniformly spaced in colatitude. Treat the result as a flat spectrum, not a
  calibrated angular power spectrum.
- **`AlmTransform(method="alm")` raises `NotImplementedError`**, and so does
  `AlmTransform.ifft()`: only the FFT stage exists. `indexing_scheme` only
  accepts `"2D_array"`.
- **Only periodic boundary conditions** (`fft(..., pbc=False)` raises
  `NotImplementedError`), which is an approximation on a regional domain.
- **No handling of the 0/360° longitude wrap** in the pixel-shift
  correction.
- **Not differentiable**: the 2-D spectrum is detached and binned in NumPy.

`AlmTransform` prints diagnostics and shows debug plots (matplotlib) only
when it is built with `debug=True`.

## Removed: `powerspectra_lonlat` and `ps`

`powerspectra_lonlat` could not run (it passed 1-D cell ids to
`AlmTransform` and ignored its `lon`, `lat` and `weights`), and `ps`, a
plain 2-D FFT of a NumPy array, did not use HEALPix at all. Both were
removed. For lon/lat iso-latitude grids, build the rings with
`build_rings_from_latlon` and compute the angular power spectrum with
`anafast_latlon` ({doc}`/sht/alm_latlon`). For an array you have already
gridded yourself, `numpy.fft.fft2` followed by a radial average does what
`ps` did.
