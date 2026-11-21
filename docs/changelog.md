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
- `multiorder_filter_operator`: a Gaussian filter operator on a mixed-level set of
  HEALPix leaves (a river at a fine level opening into a sea at a coarse one).
  Land is a place without a leaf, or a blocked leaf; the kernel acts on the
  shortest path through water over edge and corner contacts across levels (2:1
  balanced input, checked), with leaf areas (and an optional wet fraction) in
  the weights. `normalization="source"` conserves the total exactly; with one
  level and `correction="lattice"` the result equals the one-level operator with
  `blocked_cells`. No new dependency (`shortest_path="scipy"` can use SciPy, optional extra `scipy`, imported only then).
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
