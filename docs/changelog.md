# Changelog

## Unreleased

### Added
- `healpix_analyse.nufft`: two-dimensional non-uniform FFT of scattered
  samples (for instance HEALPix cell centres of a regional domain), a thin
  wrapper around `finufft.nufft2d1`, with the sign of `numpy.fft` and a phase
  reference at the smallest coordinates. The period defaults to the extent plus
  the median sample spacing (`n * dx` on a regular grid, where the result equals
  `np.fft.fftshift(np.fft.fft2(...))`) and can be set with `period=`. Batched
  data `(K, M)` gives `(K, Ny, Nx)`. `windowing_weights_polygon` and
  `windowing_weights_bbox` give Tukey tapers inside a polygon (concave allowed,
  either orientation) or a box. New optional extra `nufft` (`finufft`).

### Changed
- `LocalFFT.blur_floor_px` (called by `HealPixFFTConv.set_gaussian` and
  `LocalFFT.check_width` since 2026.11.1) no longer uses SciPy: the moments of
  the round-trip response are accumulated per grid pixel with
  `numpy.bincount` instead of building the sparse `N x N` response matrix.
  Same values to round-off, O(N), and about twice as fast.
  `tests/test_fft_conservative.py` no longer imports SciPy either.
- `import healpix_analyse` no longer needs SciPy: `validation.py` (imported by
  `kernel_pyramid`) imported `scipy.spatial.cKDTree` at module level; it is now
  imported inside `direct_spherical_convolution`, the only function that uses
  it, which raises a clear `ImportError` without SciPy. With SciPy
  installed, this also takes its import (about 0.7 s of the 5 to 7 s of
  `import healpix_analyse` on a laptop) off the package import; the first
  call of `direct_spherical_convolution` pays it instead.

## 2026.11.1 (2026-11-23)

### Added
- `HealPixDown(mode="smooth_conservative")`: the Gaussian smoothing of
  `smooth` plus one global additive offset per input map, so that the mean of
  the output equals the mean of the input (to round-off, full sphere; on a
  partial sky the `coverage`-weighted mean). Constants are preserved and
  information still crosses parent boundaries, which `mean` does not allow.
  `delta` is built from finite values only, so a NaN stays local (the finite
  output cells have the mean of the finite input cells, over different
  cells). The shift is uniform, so a negative `delta` can make a
  non-negative field slightly negative. No new dependency.
  `sparse_matrix()` returns the Gaussian part without the offset.
  Accepted as `down_mode` by `HealPixResampler`, `resample_healpix` and
  `HealPixDecomp`, and by `HealPixUp(paired_down=...)`.
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
- `gaussian_filter_operator` and `radial_filter_operator` take
  `blocked_cells` (land): a blocked cell neither receives nor gives weight, and
  the kernel acts through water, on the length of the shortest path over
  neighbouring non-blocked cells (corrected for the metrication error of the
  8-neighbour lattice), so that probability does not cross a coast. Cells far
  from land are untouched and keep the plain operator; only the rows near
  blocked cells are searched. Cells outside the domain are not land: a hole in
  the domain, at an edge or at a vertex, is no barrier as long as the detour
  around it stays inside the candidate disc of 1.5 radii around the source; a
  longer detour drops the pair, so a long hole then acts as a barrier. Without blocked cells
  the operator is the same as before.
- `multiorder_filter_operator`: a Gaussian filter operator on a mixed-level
  set of HEALPix leaves (a river at a fine level opening into a sea at a
  coarse one). Land is a place without a leaf, or a blocked leaf (with
  `correction="lattice"` only blocked leaves are land); the kernel acts on the
  shortest path through water over edge and corner contacts across levels (2:1
  balanced input, checked), with leaf areas (and an optional wet fraction) in
  the weights. `normalization="source"` conserves the total exactly; with one
  level and `correction="lattice"` the result equals the one-level operator with
  `blocked_cells` without holes, and with holes whose detours stay within both
  limits (the one-level operator limits the search by the disc of 1.5 radii,
  this one by a path length of three radii; a source with a hole or the edge of
  the set within reach is searched again up to three radii). For a long hole
  the two can differ, and a detour longer than the limit acts as a barrier. No new
  dependency: `shortest_path="scipy"` can use SciPy
  (optional extra `scipy`, imported only then).
- `normalization="source"` for the two operators: every source cell hands out
  weights that sum to one over the cells it reaches, so that the columns sum to
  one and the total of the signal is conserved (a reflecting coast and domain
  edge). With `blocked_cells`, `"full_disc"` gives the lost-mass behaviour
  (zero padding) and `"present"` renormalises per output cell. See
  "Land barrier" in `docs/radial_filter.md`.
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
- `projection="conservative"` for `LocalFFT` and `HealPixFFTConv`: the
  projection to the local grid is not divided by the accumulated weight and
  the grid is divided by it just before the back-projection, so a convolution
  with a normalised kernel keeps the total (a probability map stays
  normalised; mass leaving the domain is lost, not reflected). The default
  `"average"` is unchanged and keeps constants but not totals (response of one
  cell: 0.79 to 1.06 inside an 8125-cell domain at level 10). `HealPixFFTConv.set_gaussian` sets a
  normalised Gaussian kernel on the channel pairs `c -> c` and warns when the
  window is under 3 sigma wide.
- `LocalFFT.blur_floor_px()`, `PROJECTION_BLUR_FLOOR_PX` and
  `check_kernel_width`: the blur of one projection and back-projection
  (about 0.58 pixel, 0.23 to 0.69 per cell on the benchmark domain, the same
  for both conventions) is documented and exposed, and `set_gaussian` warns
  when the requested width is below its median.
- `benchmarks/benchmark_fft_total.py`: totals, constants and effective width of
  the local FFT convolution.

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
