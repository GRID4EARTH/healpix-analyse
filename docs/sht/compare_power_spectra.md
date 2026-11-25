# Power spectra: which tool?

Six functions return a power spectrum. They answer different questions: an
angular spectrum $C_\ell$ of a whole-sphere field, or a radial spectrum of a
regional patch in cycles per unit length.

| Tool | Input | Output | Binning | Differentiable | Status |
|---|---|---|---|---|---|
| `HEALPixSHT.anafast` ({doc}`healpix_sht`) | a whole-sphere HEALPix map, RING (or `nest=True`) | $C_\ell$, shape `(..., lmax+1)`; spin > 0: `(..., 3, lmax+1)` = EE, BB, EB | per $\ell$ | yes | stable; not exported (import from `healpix_analyse.healpix_sht`) |
| `anafast_latlon` ({doc}`alm_latlon`) | any set of iso-latitude rings (ERA5, regular or Gaussian lat/lon grid, HEALPix) | $C_\ell$ in float64 | per $\ell$ | yes (Torch input) | stable; scale depends on the quadrature weights |
| `LocalFFT.ps` / `local_ps` ({doc}`fft_local`) | a regional patch of NESTED cells (radius 10° by default) on a gnomonic grid | 1-D radial spectrum, frequencies in cycles per tangent unit (≈ cycles per radian; divide by $R$ for cycles per metre) | linear bins up to Nyquist | yes | stable; not a $C_\ell$ without calibration |
| `ps` ({doc}`powerspectra`) | a regular 2-D NumPy array | 1-D radial spectrum, cycles per array index | linear bins | no | experimental; not exported |
| `powerspectra` ({doc}`powerspectra`) | a 2-D array of RING cell ids | 1-D radial spectrum, cycles per `dx` | linear bins | no | experimental; not exported |
| `powerspectra_lonlat` ({doc}`powerspectra`) | lon/lat rings | — | — | no | does not run as written; not exported |

All $C_\ell$ follow the orthonormal convention of `healpy`
({doc}`/start/conventions`).

## Rules of thumb

- **Whole sphere, HEALPix** → `HEALPixSHT.anafast` (the tables cost memory:
  about 4.8 GB at level 8 in float32; see {doc}`healpix_sht`).
- **Whole sphere, lat/lon or Gaussian grid** (ERA5) → `anafast_latlon`.
  Also usable on HEALPix when `HEALPixSHT` does not fit in memory.
- **A regional patch** (one scene, one basin) → `local_ps`.
- **An array you already gridded yourself** → `ps`.
- `powerspectra` and `powerspectra_lonlat` are kept for development only.

No tool of the package estimates a $C_\ell$ on a partial domain with a mask
correction (pseudo-$C_\ell$); `anafast_latlon` and `HEALPixSHT.anafast`
assume full coverage.
