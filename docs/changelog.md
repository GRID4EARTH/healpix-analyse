# Changelog

## Unreleased

### Added
- `radial_filter` and `gaussian_filter` take `normalization="present"`
  (default, the existing behaviour) or `"full_disc"`: the weights are divided
  by their sum over the whole kernel disc, cells outside the domain included
  and contributing zero, so that mass diffusing out of the domain is lost (the
  `mode="constant", cval=0` of a raster filter). A `NaN` inside the domain
  counts as zero in that mode.
- `gaussian_filter_operator`, `radial_filter_operator` and
  `apply_filter_operator`: the weights of a filter as three plain arrays in
  compressed-row form plus the number of columns (`FilterOperator`: row
  offsets, column indices, weights, `n_columns`), built once and applied to
  values with leading dimensions (NumPy or Torch). `apply_filter_operator`
  checks the signal length when `n_columns` is given. Operators are always
  normalised. No new dependency; the result equals the filter with the same
  options.
- `gaussian_filter(..., width_rule="matched")` (and the operator): the
  requested width is reached below about one cell, by solving the Gaussian
  shape per cell above about 0.6 cell and blending with the identity below it.
  The default `"sampled"` is unchanged, and its documentation now says that its
  effective width is smaller than requested below about one cell.
- `benchmarks/benchmark_geo_release.py` and the report and timings of the
  HEALPIX-GEO 0.4.0 direct-sum benchmark
  (`benchmarks/results/geo040-performance/`), so that the first-call and
  cache timings of `gaussian_filter` can be cited and re-run.
- `LargeConv.from_gaussian`, `gaussian_plan`, `LargeConv.set_gaussian_kernel`,
  `down_up_blur` and `LargeConv(n_levels=...)`: a Gaussian smoothing of a
  given width whose effective width is continuous and strictly increasing in
  sigma. The automatic level rule gives a staircase (at level 10 the width
  sits on 2.0, 4.2, 8.3 and 16.5 cells: 592 flat steps out of 595 over 2.25
  to 32 cells). Measured over 756 widths from 2.25 to 40 cells across four
  level switches, the response to the calibration cell is strictly
  increasing and within 3.8e-4 of the requested width on the calibration
  patch (an inversion check). Widths above a quarter of the distance to the
  patch border (31.7 cells at level 10 with the default `patch_depth=8`)
  are refused: their tail is cut at the border during the calibration, and
  on a larger domain they came out up to 10.6 % too wide at 60 cells
  (within 3e-4 below the limit). `set_gaussian_kernel` refuses a plan
  measured with other options than the layer, and a domain that reaches
  more than 30 degrees from the calibration cell gives a warning. Other cells of the
  domain spread by 8 % at 3 cells, 3 % at 6 and at most 1.5 % from 9 cells (up
  to 240 cells from the calibration cell, clear of the patch edge), as much
  as the automatic rule around its own mean; the total of one cell is not
  conserved (0.92 to 1.05); near the edge of a partial domain the width is up to 12.7 % too
  narrow (zero padding); calibrated at the equator, 55 degrees latitude is
  up to 22 % too wide, so the calibration block comes from the domain by
  default. Widths are one-axis RMS (sigma, not FWHM) in radians; the
  narrowest width is about 2.2 to 2.8 cells, narrower ones are for
  `gaussian_filter`. The default behaviour of `LargeConv` is unchanged.
- `HealPixGeoFFT` / `geo_fft_convolve`: FFT convolution of a block inside one
  base face with a wide kernel sampled on geodesic distances and azimuths from
  the block centre (predefined families in metres, kilometres, degrees or
  cells, or any callable `k(rho, phi)`), normalized convolution for missing
  data as a ratio of two FFTs. The kernel window is sized from the local
  lattice geometry, which is a sheared parallelogram with unequal sides in the
  polar caps, and widened where that geometry drifts across a large window;
  treating the face lattice as a square grid is measured to be off by
  5-30 % on Sentinel-2 tiles at level 20. The kernel `k(rho, phi)` is
  evaluated from the output cell to the input cell, as in
  `directional_filter`.
- `HealPixDown(mode="mean")`: unweighted mean of the 4 NESTED children. It
  conserves the global mean of a NaN-free full-sphere map, which
  `mode="smooth"` does not (its Gaussian weights are normalised per output
  cell only). On a partial sky, `coverage` gives the fraction of children
  present per output cell.
- `skipna=True` (with `mode="mean"`): NaN-skipping mean;
  `HealPixDown.valid_fraction(x)` returns the fraction of children used for
  that input, so a weighted mean of the output stays conservative.
- `HealPixDown.sparse_matrix()`: the `smooth` or `mean` operator as a sparse
  matrix. `HealPixUp(paired_down=...)` accepts a `mean` downsampler.
- `down_mode` keyword (`"smooth"` default, or `"mean"`) on
  `HealPixResampler`, `resample_healpix` and `HealPixDecomp`. Defaults and
  reprs are unchanged.

### Fixed
- `HealPixKernelPyramid` fitting accepts operators that return torch tensors
  on a GPU or with gradients.

## 2026.11.0 (2026-11-15)

### Changed
- `healpix-geo` 0.5 or later is required: it is the first release with the
  batched `healpix_geo.nested.cone_coverage`.
- Connected-component topology now uses the direction-preserving
  `healpix_geo.nested.neighbours` backend. A reproducible benchmark records
  correctness checks, input dtype, threading mode and reference timings.
- Cone candidates for the metric neighbourhood helpers and for
  `HealPixDown` (`mode="smooth"`) come from one batched
  `healpix_geo.nested.cone_coverage` call instead of one call per cell.
  Outputs are unchanged.

### Fixed
- The `dino` sampling helpers no longer raise `OverflowError` under NumPy 2
  when masking missing bilinear stencil corners.

### Added
- `radial_filter` and `gaussian_filter` keep process-wide LRU caches of the
  filter geometry (up to 192 MiB by default) and of the Gaussian weights (up
  to 96 MiB). `configure_radial_filter_cache` changes or disables the limits
  (`0` disables a cache), and `radial_filter_cache_info` reports limits and
  current usage.

## 2026.10.2 and earlier

### Changed
- HEALPix-facing public APIs now take the Grid4Earth `level` directly and
  derive `nside = 2**level` internally. This applies to `HealPixConv`,
  `LargeConv`, `HealPixDown`, `HealPixUp`, `HEALPixSHT`, localized ALM
  transforms, and `build_healpix_adjacency`.

### Fixed
- `HealPixConv` geometry cache keys retain the input `cell_ids` order, so a
  permutation of the same partial domain cannot reuse incompatible sorting
  buffers.

### Added
- `HealPixResampler` and `resample_healpix`: reusable and one-shot local
  resampling between full or partial NESTED HEALPix levels, with NaN support.
- `HealPixDivCurl` and `HealPixMultiScaleDivCurl`: fixed gauge-aware
  derivative kernels for divergence and curl at every decomposition scale.
- `HealPixDecomp` and `HealPixPyramid`: exactly reconstructing local
  Laplacian pyramids with cell identifiers retained at every scale.
- `HealPixFFTConv`: differentiable, zero-padded FFT convolution for very large
  learned kernels on local pole-safe gnomonic HEALPix patches.
- `HEALPixSHT`: ring-based full-sky spherical harmonic transform with spin support (spin-0, spin-1, spin-2)
- `alm_latlon`: SHT for arbitrary iso-latitude grids (ERA5, regular lat/lon, HEALPix)
- `HealPixConv`: gauge-equivariant spherical convolution on HEALPix maps
- `HealPixDown` / `HealPixUp`: multi-resolution operators (smooth and max-pool modes)
- `powerspectra` / `powerspectra_lonlat`: isotropic 1D power spectrum estimation
- `LocalizedFlatSkyAlm`: flat-sky approximation for localized SHT on large patches
