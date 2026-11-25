# Power spectra: which tool?

Five tools return a power spectrum. They answer different questions: an
angular spectrum $C_\ell$ of a whole-sphere field, a radial spectrum of a
regional patch in cycles per unit length, or band powers of a field with
missing data, on the full sphere or a regional domain.

| Tool | Input | Output | Binning | Differentiable | Status |
|---|---|---|---|---|---|
| `HEALPixSHT.anafast` ({doc}`healpix_sht`) | a whole-sphere HEALPix map, RING (or `nest=True`) | $C_\ell$, shape `(..., lmax+1)`; spin > 0: `(..., 3, lmax+1)` = EE, BB, EB | per $\ell$ | yes | stable |
| `anafast_latlon` ({doc}`alm_latlon`) | any set of iso-latitude rings (ERA5, regular or Gaussian lat/lon grid, HEALPix) | $C_\ell$ in float64 | per $\ell$ | yes (Torch input) | stable; scale depends on the quadrature weights |
| `LocalFFT.ps` / `local_ps` ({doc}`fft_local`) | a regional patch of NESTED cells (radius 10° by default) on a gnomonic grid | 1-D radial spectrum, frequencies in cycles per tangent unit (≈ cycles per radian; divide by $R$ for cycles per metre) | linear bins up to Nyquist | yes | stable; not a $C_\ell$ without calibration |
| `HealPixWaveletSpectrum` / `HealPixIsotropicWaveletSpectrum` ({doc}`wavelet_spectrum`) | full sphere or a regional set of NESTED cells, with NaN (land, clouds) | band powers, one per level (or two per octave), with orientations; cross-spectra and coherence | one band per octave; windows measured by `windows()` | yes | new; per map, the finest band agrees with band-averaged `anafast` within about 4 %, coarser bands within 9–19 % (5 % with `down_mode="smooth"`) |
| `powerspectra` ({doc}`powerspectra`) | a 2-D array of RING cell ids | 1-D radial spectrum, cycles per `dx` | linear bins | no | deprecated (`DeprecationWarning`); not exported |

All $C_\ell$ follow the orthonormal convention of `healpy`
({doc}`/start/conventions`).

## Rules of thumb

- **Whole sphere, HEALPix** → `HEALPixSHT.anafast` (the tables cost memory:
  about 4.8 GB at level 8 in float32; see {doc}`healpix_sht`).
- **Whole sphere, lat/lon or Gaussian grid** (ERA5) → `anafast_latlon`.
  Also usable on HEALPix when `HEALPixSHT` does not fit in memory.
- **A regional patch** (one scene, one basin) → `local_ps`.
- **Missing data** (land mask, clouds, swaths), a regional domain of any
  size, or a spectrum used as a training loss → `HealPixWaveletSpectrum`
  (band powers, not a $C_\ell$ per $\ell$); `HealPixIsotropicWaveletSpectrum`
  when the orientation does not matter.
- `powerspectra` is deprecated. `powerspectra_lonlat` and `ps` were
  removed; for lon/lat grids use `build_rings_from_latlon` + `anafast_latlon`.

No tool of the package estimates a $C_\ell$ per $\ell$ on a partial domain
with a mask correction (pseudo-$C_\ell$); `anafast_latlon` and
`HEALPixSHT.anafast` assume full coverage. With missing data, the wavelet
spectrum gives band powers instead ({doc}`wavelet_spectrum`).
