# Changelog

## Unreleased

### Changed
- `HEALPixSHT.uv_to_curl_div` now returns the divergence and the relative
  vorticity, in s⁻¹ for u, v in m/s (#94): the spin-1 E/B coefficients are
  multiplied by −√(ℓ(ℓ+1))/R before the spin-0 synthesis, with the new
  parameter `radius=` (default 6 371 008.8 m, the IUGG mean Earth radius).
  Before, it returned the E/B potential maps in the units of u and v; they
  are now `HEALPixSHT.uv_to_eb(u, v, nest=False)`. The docstring labels,
  which were inverted, are fixed. The `-map2alm` sign of
  `map2alm_spin(Q, U, spin=0)` is kept (it is the HEALPix spin convention
  and matches `healpy.map2alm_spin`) and is now documented.
- `build_healpix_adjacency` (HEALPix Minkowski functionals) no longer imports
  `healpy`, which no dependency declared: the neighbours come from
  `healpix_geo.nested.neighbours`, the backend of the connected components,
  and the edges and triangles are built with vectorised NumPy instead of a
  Python loop over the cells (#96, item 4). The output is unchanged; an input
  without any edge now returns tensors of shape `[0, 2]` and `[0, 3]`
  instead of `[0]`, and duplicated `cell_ids` raise `ValueError` (the loop
  built an asymmetric graph from them and lost triangles).
- Topology tests: hard-coded NESTED neighbours at levels 0 and 1 (base-cell
  corners with 7 neighbours included); the comparisons with `healpy` run only
  when it is installed (#41). The private topology adapter rotates the
  `healpix-geo` direction order with one copy instead of two.

### Added
- Docs: a "Basic tools for AI" section (`docs/ai/`, #109): the AI building
  blocks in one table (role, shapes, gradient, GPU kernel, deterministic
  backward), an API card per block with signatures read from the source
  code, a "which block for which architecture" table, a page on data
  (domains, NaN and masks, augmentation, batching), and runnable recipes
  (ResNet classifier, U-Net, ViT / MAE with NESTED-block tokens, training
  loop with masks and bf16 autocast) run by `tests/test_ai_recipes.py`.
  `docs/tools.yaml` gains an `ai:` field per block and an `ai:` section;
  `llms.txt` and `llms-full.txt` gain an "AI building blocks" section (API
  cards and the code of the recipes); new AI cases in the tool-choice
  evaluation set.
- `HealPixDown` / `HealPixUp(backend="auto" | "triton" | "torch" | "csr")`
  (#109, section 1). `"mean"` and `"maxpool"` Down on contiguous NESTED data
  (the full sphere, or sorted `cell_ids` whose parents are complete, detected
  at construction) reduce a `[B, N/4, 4]` view of the input, with no index
  array; an Up paired with such a `"mean"` Down is an `expand` of the
  coarse values, with no matrix. Both are bit-identical to the previous
  implementation. The Gaussian modes, and every other Up, store their matrix
  and its transpose in ELL form (fixed fan-in, int32 indices) and apply it
  with a Triton gather-reduce kernel on CUDA (float32, float16, bfloat16;
  float32 accumulation, so 16-bit data is no longer copied to float32). The
  backward is a gather with the stored transpose: no atomics,
  deterministic, twice differentiable. Where the Triton kernel does not run
  (CPU, float64, CUDA without Triton), `"auto"` keeps the CSR product,
  which is faster there; `backend="torch"` runs the ELL product with
  `embedding_bag`.
  On CUDA the other paths are fused Triton kernels too, forward and
  backward: `maxpool` (int8 argmax, one-pass backward), the expand Up, and
  `smooth_conservative` (float64 row sums in Triton; the backward is one ELL
  gather with the mean correction fused in).
  `backend="csr"` keeps the previous CSR / `torch.sparse.mm` implementation
  as the reference. `forward` now accepts `[..., N]` inputs (a `[B, C, N]`
  feature map is flattened by a view, without a copy) and returns a
  contiguous `[..., N_out]`. New benchmark `benchmarks/benchmark_updown.py`.
  Checkpoints: the child indices of `"mean"` / `"maxpool"` are no longer
  written to the `state_dict` (they are rebuilt by the constructor), so
  `"auto"` and `"csr"` checkpoints are the same. Checkpoints of earlier
  versions still load with `strict=True` (their `_children_idx` /
  `_children_valid` keys are dropped); a checkpoint written by this version
  loads into an earlier version only with `strict=False`.
- Neighbourhood reductions and the local gradient run as vectorised stencil
  operations (#109, section 4). `HealPixNeighbourReducer` no longer loops
  over the output cells in Python: its neighbourhoods are built once with
  vectorised NumPy and stored padded, and a call is a gather-reduce, a loop
  over the neighbours on CPU and fused Triton kernels on CUDA for `sum`,
  `mean`, `min`, `max` and `median` of floating-point data (the median by
  ranking the neighbours in registers). Their backward passes are written by
  hand and deterministic (a gather on the transposed neighbourhoods on
  CUDA); `min`, `max` and `median` keep the subgradient of torch, with ties
  going to the first cell in the order of `cell_ids`. `std`, `mode`, `any`,
  `all` and integer reductions are vectorised too. New keyword
  `backend="auto" | "triton" | "torch"` of `HealPixNeighbourReducer.__call__`.
  Values are unchanged (exact for `min`, `max`, `median`, `count`, `mode`,
  `any`, `all`; to rounding for the sums). One difference in the gradient:
  on ties, the median's subgradient now goes to the first cell in the order
  of `cell_ids` (a stable sort), where it depended on the unstable
  `torch.sort` before.
- `HealPixGradient`: the least-squares East/North gradient as a reusable
  operator. The WGS84 geometry is built once, a call takes values
  `[..., N]`. On CUDA with Triton, tensors run fused kernels with a
  deterministic hand-written backward, without building the `[..., N, 8]`
  gathered tensor (float16 / bfloat16 with fp32 accumulation); elsewhere the
  reference implementation runs. `gradient()` delegates to it (same values
  to about 1e-12 relative in float64).
- `HealPixDivCurl` (and `HealPixMultiScaleDivCurl`): on CUDA with Triton,
  the embedding of (u, v) in a Cartesian tangent vector and the contraction
  of the `HealPixConv` derivatives to divergence and curl (with the gauge
  mean) are fused kernels, forward and backward, instead of about ten
  PyTorch operations and their intermediate tensors. Values unchanged to
  rounding. `benchmarks/benchmark_reductions.py` measures the reductions,
  the gradient and `HealPixDivCurl` (time, peak memory, and bandwidth against
  the HBM peak of the GPU).
- `HealPixWindowAttention` and `HealPixNeighbourhoodAttention` (#109,
  section 3), new page "Attention layers". Window attention runs full
  multi-head attention inside NESTED blocks of `4**window_level` cells, by a
  reshape and `scaled_dot_product_attention` on the full sphere and on sorted
  domains of complete blocks, with padded and masked windows otherwise. The
  replacement of Swin's shifted windows is the option
  `shift="coarse_centred" | "alternate_level" | "none"`: the default,
  chosen in #109, is `"coarse_centred"` (windows centred on the cells of a
  coarser level, overlapping across block borders), set by the module
  constant `DEFAULT_WINDOW_SHIFT`. Neighbourhood attention lets each cell
  attend to the taps of the gauge-oriented `HealPixConv` stencil, with one
  relative position bias per head and tap. On CUDA both layers run fused
  Triton kernels (flash-style online softmax, 16-bit data with fp32
  accumulation, autotuned tiles, deterministic backward without atomics,
  the packed projection read and its gradient written in place); a
  PyTorch path runs elsewhere and serves as the reference. `benchmarks/benchmark_attention.py` compares them
  with dense attention and with a naive gather.
- Masked, coverage-aware normalisation and pooling for AI models (#109,
  section 5): `HealPixGroupNorm`, `HealPixLayerNorm` (over the channels of
  each cell) and `HealPixBatchNorm` (running statistics) take a validity
  mask or a coverage of shape `[N]`, `[B, N]` or `[B, C, N]`; with
  `mask=None` the NaN cells are invalid. Invalid cells stay out of the
  statistics and output 0, NaN or the input (`fill=`). With an all-valid
  mask the layers equal their `torch.nn` versions (same `state_dict`). On
  CUDA they run fused, deterministic Triton kernels (optional `triton`);
  elsewhere a few PyTorch passes with a hand-written backward. `global_pool`,
  `region_pool`, `HealPixGlobalPool` and `HealPixRegionPool`
  (`from_cell_ids` for parent cells) pool with the same masks.
- Positional encodings (#109, section 6): `HealPixPositionalEncoding` and
  `healpix_positional_encoding` give Fourier features of the cell-centre
  unit vector (healpix-geo centres, WGS84 by default), optionally the gauge
  angle of `HealPixConv` and the level; the geometry is cached per domain,
  keyed by a hash of the `cell_ids`, and the module buffer follows `.to()`.
- `HealPixConv(backend="auto" | "triton" | "torch", deterministic=True)`
  (#90). On CUDA, for float32, float16 and bfloat16, the forward and backward
  passes run as fused Triton kernels when the optional `triton` package is
  installed (new extra `triton`; the Linux CUDA wheels of PyTorch usually
  ship it already). 16-bit data accumulate in fp32, and under
  `torch.autocast` the kernels compute in the autocast dtype. Everything else
  (CPU, float64, no Triton) runs a PyTorch loop over the stencil taps with a
  hand-written backward. Neither backend materialises the gathered
  `[B, C_in, G, K, P]` tensor, so training needs a fraction of the memory
  and configurations that ran out of memory on an 80 GB GPU now fit.
  By default (`deterministic=True`), the Triton backend computes the input
  gradient without atomics, by a gather on the transposed stencil map: the
  same bits at every run, and a training step 1.6–2.2× shorter than with
  atomics on an A100. Its cost is GPU memory: the map is built at the first
  backward pass and kept, at about the size of the stencil buffers again
  (0.2–2.6 GiB in the benchmark configurations). `deterministic=False`
  scatters the gradient with fp32 atomics instead, with no extra memory and
  last bits that vary between runs. `deterministic` applies to the Triton
  backend only: on CUDA, the PyTorch backend (used without Triton) is
  deterministic only under `torch.use_deterministic_algorithms(True)`. A
  double backward (`create_graph=True`) runs the PyTorch backward on every
  device, so gradient penalties keep their second-order terms.
  `missing="local_mean"` keeps the PyTorch reference path (gathered tensor
  and matrix product) on every device.
- `benchmarks/benchmark_healpix_conv.py`: forward, forward + backward, peak
  memory and geometry build time of `HealPixConv`, on CPU or CUDA, one JSON
  line per run, and the gain table between two versions.
- Wavelet power spectrum (`healpix_analyse.wavelet_spectrum`):
  `HealPixWaveletSpectrum` (complex Morlet, 4 orientations),
  `HealPixIsotropicWaveletSpectrum` (angle-averaged Morlet, no orientation)
  and the function `wavelet_power_spectrum`, with the profiles
  `wavelet_morlet` and `wavelet_morlet_isotropic`. Band powers per HEALPix
  level (down to level 0, or two per octave with `scales_per_octave=2`), on
  the full sphere or a regional domain with NaN: the field goes down with
  `HealPixDown` on the pair (w (x − M), w) (`down_mode="mean"` by default,
  area-weighted, so that the sums are conserved; `"smooth"` as an option),
  missing stencil points are completed with the local mean, and each valid
  pixel is weighted by the coverage of its kernel. Returns the frequency 0
  (the mean of the level-0 pixels, which is the mean of the valid pixels),
  the low-pass variance, `n_eff`, optional S1 and local maps; `cross()` gives
  co-spectra and coherence of two fields on the same cells; `windows()` and
  `littlewood_paley()` measure the band responses and fit band constants on
  the full sphere. Differentiable, batched, CPU or GPU; no SciPy. Per map on
  the full sphere, the finest band agrees with the band-averaged
  `HEALPixSHT.anafast` within about 4 %; the coarser bands within 9 % one
  level down and 19 % two levels down with the default descent (within 5 %
  with `down_mode="smooth"`), measured at level 4. Page:
  `docs/sht/wavelet_spectrum.md`.
- `HealPixConv(missing="local_mean")`: NaN values (and `mask=` zeros) are
  missing exactly like pixels outside `cell_ids`; a stencil point takes the
  renormalised bilinear interpolation of its valid neighbours, or the local
  mean of the valid stencil values when it has none
  (`local_mean_weights`). Finite outputs and gradients everywhere.
  `mask=` is `[N]`, `[B, N]` or `[B, C_in, N]`.
  `completed_stencil()` returns the stencil values and their valid
  fraction. The geometry of this mode is built in float64 (then cast), so
  float32 and float64 layers agree at the domain border. The default,
  `missing="centre"`, is the existing behaviour (identical outputs). The
  geometry cache now also stores the in-domain fraction of each stencil
  point; a file without it is recomputed once when `missing="local_mean"`
  needs it.

### Changed
- `HealPixConv` stores its stencil as int32 neighbour positions in the order
  of the input (the sort permutation of a regional domain is folded in), in a
  tap-major layout: no per-call copy of the input, and half the index memory.
  The geometry is no longer part of the `state_dict` (it is rebuilt or
  reloaded from the disk cache at construction); state_dicts that hold the
  former `_pos_safe` and `_w_norm` entries still load. State dicts saved by
  this version have no `_pos_safe` / `_w_norm` entries, so an older release
  loads them only with `load_state_dict(..., strict=False)` (it builds the
  geometry at construction). `_pos_safe` and `_w_norm` remain readable as
  views built on access.
- `HealPixConv` follows `module.to(...)`, `.cuda()`, `.double()`, ...: the
  computation runs on the device and in the dtype of its parameters and
  buffers (`self.device` and `self.dtype` report them), with no per-call
  `.to()` of the buffers. With `use_norm=True` the `GroupNorm` is created in
  the layer dtype.
- `HealPixConv` geometry cache (format 3): the key hashes the raw bytes of
  `cell_ids` instead of serialising them, float64 layers keep float64
  bilinear weights in the cache (they were rounded to float32), and float16 or
  bfloat16 layers build their geometry in float32. Caches written by earlier
  versions are ignored and rebuilt once.
- `HealPixConv.set_kernel` copies the kernel in the layer dtype (it went
  through float32, which truncated the float64 kernels of `HealPixWideConv`
  and of the kernel pyramid), and accepts tensors.
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
- `AlmTransform` no longer prints in its constructor; the diagnostics are
  printed only with `debug=True`. `AlmTransform(method="alm")` now raises
  `NotImplementedError` instead of silently running the FFT path.
- `AlmCoeffs` is no longer exported from `healpix_analyse` (no function
  returns it); import it from `healpix_analyse.alm`.
- Packaging: `termcolor`, `umap-learn` and `healpix-plot` are no longer core
  dependencies (the package does not import them). `termcolor` stays in the
  `dino` and `examples` extras and `umap-learn` in the `examples` extra;
  `healpix-plot` is not on PyPI, so it is in no pyproject extra, only in the
  pixi `examples` feature (git source). New extra `sht` (`quaternionic`,
  `spherical`) for the spin ≠ 0 transforms of `HEALPixSHT`, whose
  `ImportError` now names it. `healpy`, used only by the tests, is added to
  the `test` extra and moved from the pixi core dependencies to the pixi
  `test` feature. `resample_to_latlon_grid` raises an `ImportError` naming
  the `scipy` extra when SciPy is missing.
- `HealPixConv`, `HEALPixSHT`, `binary_dilation`, `binary_erosion`,
  `get_interp_val`, `get_interp_weights` and `resample_to_latlon_grid` are
  now exported from `healpix_analyse`. `quaternionic` and `spherical` are
  imported on first spin ≠ 0 use, not when `healpix_sht` is imported.
- `alm2map_latlon`: `lmax` now defaults to `None` and is deduced from the
  length of `alm`; a value that does not match raises `ValueError`.
  `grid_summary` defaults to `lmax=64`, like `map2alm_latlon` and
  `anafast_latlon` (it was 24).

### Fixed
- `alm2map_latlon` returned the m > 0 terms at half amplitude: the synthesis
  of a real field now carries the factor 2 of the pair ±m, and
  `alm2map_latlon(map2alm_latlon(f))` returns a band-limited `f` to rounding
  error with an exact quadrature (#92).
- `alm_latlon`: uniform rings with an odd number of points lost one
  frequency in the analysis, which corrupted the aliased modes when
  `lmax >= (N_r + 1) / 2`; irregular rings failed on CUDA input (kernel
  built on the CPU). New tests in `tests/test_alm_latlon.py`.
- `HealPixDown(mode="maxpool")` on a regional domain: a child absent from
  `cell_ids` was replaced by the first input cell, which can belong to
  another parent (cells `[100, 4, 5, 6]` at level 2 with values
  `[50, 1, 2, 3]` gave `[50, 50]`); it is now replaced by a present child of
  the same parent, so a parent is the max of its children present (#93).
- `docs/other/down.md`, `docs/other/up.md`: the regional-domain examples
  read a single-centre `cone_coverage` result as one array; with
  healpix-geo 0.5 it is a tuple `(cell_ids, depths, fully_covered)`, and the
  old recipe gave repeated ids.

### Performance
- `HealPixDown` (`"smooth"`, `"smooth_conservative"`) and `HealPixUp` keep
  their matrix and its transpose as CSR buffers (int32 indices) built once in
  the constructor; they follow `.to()`. `forward` no longer assembles a COO
  tensor at each call, and the gradient applies the stored transpose (twice
  differentiable). Level 8 → 7, one CPU thread, one map: `HealPixDown`
  forward 167 → 12 ms, forward + backward 370 → 15 ms with a per-call peak
  of 151 → 6 MiB; `HealPixUp` forward 153 → 10 ms, forward + backward
  575 → 36 ms (157 → 9 MiB) (#59).
- `HealPixDown` and `HealPixUp`: one dtype rule for every mode. The output
  has the module dtype, which follows `.to(dtype)`, `.double()`, `.half()`
  (before, `"mean"` and `"maxpool"` kept the constructor dtype). After a cast
  to `bfloat16` or `float16`, the matrix and the normalisation vectors stay
  in float32 (the CPU sparse kernels have no 16-bit version), the linear
  part is computed there, and the output is cast to the 16-bit dtype. A
  `bfloat16` result for a numpy input is returned as float32.

### Compatibility
- The matrix of `HealPixDown` / `HealPixUp` is no longer in the
  `state_dict`. Loading a `state_dict` saved by an older version still works
  (its `_M_indices` / `_M_values` are ignored), but a `state_dict` saved by
  this version has no `_M_*` keys, so older versions fail to load it with
  `strict=True`. Whole modules pickled (`torch.save(module)`) by an older
  version do not unpickle with this version: rebuild them from their
  constructor arguments.

### Removed
- `down.py`: the old `_build_smooth_matrix` kept in a string, the unused
  `_normalize_weights_vec`, and the unused `pyproj` import; the `down.py` /
  `up.py` docstrings use healpix-geo instead of healpy (#61, #96).
- `healpix_analyse.powerspectra_lonlat`: it could not run (1-D cell ids
  passed to `AlmTransform`, `lon`/`lat`/`weights` ignored). For lon/lat
  iso-latitude grids, use `build_rings_from_latlon` + `anafast_latlon`.
- `healpix_analyse.ps` (a plain 2-D FFT of a NumPy array, not HEALPix).
- The legacy modules `LocalizedFlatSkyAlm`, `LocalizedAlm_optimized_v13_fixed`
  and the script `demo_ps` (they import `foscat`, which is not a dependency),
  and `make_rectangle` (it imports `cdshealpix`; no module of the package used
  it). The notebooks `Test_rectangle`, `Test_resampling_powerspectra` and
  `Test_resampling_powerspectra-s2-l2a` still import `make_rectangle` and `ps`.

### Fixed
- Ellipsoid names in any casing (`"wgs84"`, `"Wgs84"`, `"Sphere"`,
  `"SPHERE"`, ...) are accepted by the public functions and constructors
  that take `ellipsoid`, including `get_interp_val` / `get_interp_weights`,
  which rejected `"wgs84"` and `"Sphere"`, and the WGS84-only filters
  (`radial_filter`, `gaussian_filter`, `neighbour_reduce` and its
  `*_filter` shortcuts, `directional_filter`), which raised
  `NotImplementedError` for `"wgs84"`. `dino.set_ellipsoid` now also
  accepts title case. `AlmTransform` is not covered yet (#96).
- `HealPixWideConv` accepts `cell_ids` in any order: it sorts them
  internally and returns the result in the order given (it used to give
  wrong results silently on unsorted ids). A permutation of the same domain
  reuses the cached fit (#96).
- Doc paths in warnings and docstrings point to the reorganised pages
  (`docs/conv/large_conv.md`, `docs/conv/radial_filter.md`,
  `docs/conv/pyramid_conv_maths.md`, `docs/other/dino.md`); the
  `radial_filter` docstrings name the helpers the code calls
  (`build_metric_neighbourhood_geometry`,
  `compact_weighted_neighbourhood_reduce`); `nested_to_tiles` documents
  that repeated ids follow `duplicates`, and `GetDINOV3SAT` lists the
  `"percell"` projection (#96).

### Deprecated
- `healpix_analyse.powerspectra.powerspectra` emits a `DeprecationWarning`
  and will be removed. Use `HEALPixSHT.anafast` on the full sphere,
  `HealPixWaveletSpectrum` on a regional domain or with missing data, and
  `build_rings_from_latlon` + `anafast_latlon` on lon/lat grids.

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
