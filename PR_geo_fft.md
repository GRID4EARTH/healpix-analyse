# Add HealPixGeoFFT: FFT convolution on the face lattice with a geodesically sampled kernel

## Summary

Inside one base face, NESTED HEALPix cells form a regular `(i, j)` lattice on which a block can be convolved by a planar FFT. This PR adds `HealPixGeoFFT`, which does so **without reprojecting the data** and with the kernel **sampled on geodesic distances and azimuths from the block centre**, so that the distortion of the HEALPix cells is built into the kernel image. Missing data are handled by normalized convolution as a ratio of two FFTs.

The motivation is a measured error: the face lattice is not a square grid (lattice angle 99° at the equator, sheared parallelogram with sides differing by up to 2× in the polar caps). A kernel written as `k(sqrt(di² + dj²))` is off by 5–30 % of the standard deviation of the exact result on Sentinel-2 tiles at level 20, and by up to 50 % under cloud masks. With the geodesic sampling the FFT agrees with `direct_spherical_convolution` to ~1e-5 (tests).

## API

```python
from healpix_analyse import HealPixGeoFFT, geo_fft_convolve

conv = HealPixGeoFFT(level=20, cell_ids=cells, kernel="exponential", scale=100.0)  # metres
y = conv(x)                       # NaN = missing -> normalized convolution K*(wx)/K*w
conv.raw(x); conv.support(x=x); conv.kernel_image; conv.geometry; conv.extent_deg
```

- `kernel`: a predefined family (`gaussian`, `exponential`, `matern`, `lorentzian`, `moffat`, `voigt`, `screened_poisson`, `top_hat`, `dog`; `register_kernel` to add one) or **any callable** `k(rho, phi)` of the geodesic distance and the azimuth from north (anisotropic kernels in the geographic frame).
- `units`: `"m"` (default), `"km"`, `"deg"`, `"pix"` for `scale`, `radius` and `rho`.
- `radius`: truncation; default where the kernel falls below 1e-3 of its peak. The window is sized from the local lattice geometry (`R / (|e| sin θ)`), which matters in the polar caps.
- `cell_ids`: any cells of one face, in any order, not necessarily contiguous (bounding rectangle, absent cells = missing). Two faces → explicit error pointing to the pyramid.
- NumPy or torch input, `[N]`, `[B, N]`, `[B, C, N]`; differentiable w.r.t. `x`.
- Warning above `warn_extent_deg` (10°): the cell geometry then varies across the block; use `HealPixWideConv` / `HealPixPyramidConv`.

## Files

- `healpix_analyse/geo_fft.py` — the module (`HealPixGeoFFT`, `geo_fft_convolve`, `KERNELS`, `register_kernel`, `make_kernel`, `lattice_geometry`)
- `healpix_analyse/__init__.py` — exports
- `tests/test_geo_fft.py` — 11 tests: agreement with the direct geodesic sum (polar cap off-diagonal, equator, near pole; raw and normalized with 40 % gaps), square-grid kernel shown wrong in the cap, constant preserved, Dirac → kernel image, non-contiguous cells, torch batch + gradient, callable anisotropic kernel and unit consistency, all predefined kernels, default radius, errors, one-shot function, extent warning
- `docs/geo_fft.md`, `docs/index.md`, `docs/changelog.md`

## Relation to existing code

`HealPixFFTConv` reprojects on a gnomonic plane (`LocalFFT`) for learned kernels; `HealPixGeoFFT` keeps the data on their cells and targets prescribed geophysical kernels. `HealPixWideConv.lattice_offsets_m` already exposed the lattice distortion; this PR uses the same Earth radius (6 371 008.8 m) and conventions.

## Checks

`pytest tests/test_geo_fft.py tests/test_wide_conv.py` → 35 passed, 1 skipped.
