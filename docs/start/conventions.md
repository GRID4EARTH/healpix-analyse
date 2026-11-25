# Conventions

The modules of `healpix-analyse` were written at different times and do not
all name or order their arguments the same way. This page lists the
conventions **as they are in the code today**, so that you can call any tool
without guessing. Harmonising them is a separate piece of work; until then,
check this page when you combine two tools.

## Cells: NESTED ids, whole sphere or regional domain

A HEALPix grid at level `L` has `12 · 4**L` cells; `nside = 2**L`.

- **Ordering.** Almost every tool works on **NESTED** cell ids. The
  exceptions are the spherical-harmonic transform `HEALPixSHT`, which takes
  RING maps by default (`nest=False`, set `nest=True` for NESTED), and the
  lat/lon transforms (`alm_latlon`), which take any iso-latitude rings and no
  cell ids at all. `HealPixConv` has a `nest` argument, but its geometry is
  always NESTED (the flag has no effect on it).
- **Whole sphere or regional domain.** Most tools accept either a complete map
  or an arbitrary set of cell ids (a *regional domain*: one scene, one basin,
  one coast). Cells outside the set are absent: they are neither zero nor NaN,
  and a kernel that reaches them loses that weight. Tools that need a margin
  around the area of interest ask for a **halo** (`LargeConv`,
  `HealPixFFTConv`), see their pages.
- **Whole sphere only:** `HEALPixSHT`, `get_interp_val` / `get_interp_weights`
  (a full NESTED map), `HealPixKernelPyramid.calibrate_joint`.
- **One regional patch only:** `LocalFFT`, `HealPixFFTConv` (a patch of radius
  at most `max_patch_radius_deg`, 10 degrees by default), `HealPixGeoFFT` (cells
  inside one base face).

### Where the cell ids are given

| How | Tools |
|---|---|
| keyword `cell_ids=None` at construction; `None` = whole sphere | `HealPixDown`, `HealPixUp`, `HealPixDecomp` (2nd positional), `LargeConv`, `gaussian_plan`, `HealPixConv`, `HealPixDivCurl` (2nd positional), `build_healpix_adjacency` |
| `in_cell_ids` / `out_cell_ids` | `HealPixResampler`, `resample_healpix` |
| required, positional after the values | `neighbour_reduce` and its aliases, `radial_filter`, `gaussian_filter`, `directional_filter`, `gradient` and its variants, `connected_components`, `remove_small_components` |
| required, first positional | `LocalFFT`, `local_fft`, the filter operators (`gaussian_filter_operator`, `radial_filter_operator`); `multiorder_filter_operator(levels, cell_ids, ...)` |
| other positions | `HealPixGeoFFT(level, cell_ids, kernel)`, `geo_fft_convolve(x, cell_ids, level, kernel)`, `HealPixFFTConv(..., cell_ids=...)` (keyword, required) |
| at **call** time | `HealPixWideConv(...)(x, cell_ids)` |
| a list of active cells instead of values | `binary_dilation`, `binary_erosion` |
| `cell_id` (singular) | `GetDINOV3SAT`, `nested_to_tiles` |
| none: a full NESTED map | `get_interp_val`, `get_interp_weights` |

The neighbourhood tools (`neighbour_reduce`, the radial and directional
filters, `gradient`, the components) also take `domain`: a subset of
`cell_ids` that sets which cells are output, and in which order. Cells of
`cell_ids` outside `domain` do not take part.

## Level

| Name | Tools |
|---|---|
| `level` | `HealPixDown` (input level, at least 1), `HealPixUp` (coarse input level), `HealPixDecomp`, `LargeConv`, `gaussian_plan`, `down_up_blur`, `HealPixConv`, `HealPixFFTConv`, `HealPixGeoFFT`, `geo_fft_convolve`, `HealPixWideConv`, `LocalFFT`, `local_fft`, `HealPixDivCurl`, `HEALPixSHT`, `build_healpix_adjacency`, DINO (`level` and `parent_level`) |
| `refinement_level` | `neighbour_reduce` family, `radial_filter` family, `gaussian_filter` family, `directional_filter`, `gradient` family, components, morphology |
| `in_level` / `out_level` | `HealPixResampler`, `resample_healpix` |
| `levels` (one per cell) | `multiorder_filter_operator` |
| `depth` | `get_interp_val`, `get_interp_weights` |
| taken from a `HealPixDecomp` | `HealPixMultiScaleDivCurl`, `HealPixKernelPyramid`, `HealPixPyramidConv` |

## Default ellipsoid

Distances, cell centres and areas depend on the ellipsoid. The defaults
differ by module:

| Default | Tools |
|---|---|
| `"WGS84"` | `HealPixDown`, `HealPixUp`, `HealPixDecomp`, `HealPixResampler`, `resample_healpix`, `LargeConv`, `gaussian_plan`, `down_up_blur`, `HealPixConv`, `neighbour_reduce` family, `radial_filter` and `gaussian_filter` families, `directional_filter`, `multiorder_filter_operator`, `healpix_cell_area`, `component_area`, `remove_small_components`, `binary_dilation`, `binary_erosion` |
| `"sphere"` | `HEALPixSHT`, `LocalFFT`, `HealPixFFTConv`, `HealPixGeoFFT`, `HealPixWideConv`, `HealPixKernelPyramid`, `HealPixDivCurl`, `get_interp_val` / `get_interp_weights`, the `validation` helpers |
| inherited from the `HealPixDecomp` (so `"WGS84"` by default) | `HealPixMultiScaleDivCurl` |
| hard-coded `"WGS84"`, no argument | `gradient`, `gradient_magnitude`, `directional_derivative` |
| module setting `set_ellipsoid(name)`, `"sphere"` by default | DINO (`GetDINOV3SAT` and helpers) |
| not applicable | `alm_latlon`, components labelling, Minkowski functionals |

Things to know:

- The metric filters and `neighbour_reduce` accept **only** `"WGS84"`
  (another value raises `NotImplementedError`); their distances are WGS84
  geodesics from `pyproj`.
- Only some constructors normalise the spelling of the name (`"wgs84"` →
  `"WGS84"`): `HealPixConv`, `HealPixDecomp`, `HealPixDown`, `HealPixUp`,
  `HealPixGeoFFT` and the `validation` helpers. `get_interp_*` compare the
  string exactly: write `"sphere"` or `"WGS84"`.
- `HealPixConv` places the cell centres on its `ellipsoid`, but interpolates
  the stencil on the sphere.
- `HealPixPyramidConv` does not check that the kernel pyramid (`"sphere"` by
  default) and the decomposition (`"WGS84"` by default) use the same
  ellipsoid: pass the same value to both.

## Widths, radii and units

| Unit | Parameters |
|---|---|
| metres | `radius_m` (`neighbour_reduce`, `radial_filter`), `sigma_m` (`gaussian_filter`, operators, `multiorder_filter_operator`, and also accepted by `gaussian_plan` / `LargeConv.from_gaussian`), `max_distance_m` (`directional_filter`), `radius` (morphology), `gsd_m` (DINO); `HealPixWideConv.from_function` / `from_radial` / `from_grid` take kernels in metres |
| degrees | `radius_deg`, `sigma_deg` (the Gaussian of `HealPixDown`, `HealPixUp`, `HealPixDecomp`, `HealPixResampler`, `LargeConv`), `max_patch_radius_deg` (`LocalFFT`, `HealPixFFTConv`); lon/lat of `get_interp_*` |
| radians | `sigma` of `gaussian_plan`, `LargeConv.from_gaussian` and `down_up_blur` (one-axis RMS width); `azimuth_rad`; `pixel_size_rad` (`LocalFFT`, `HealPixFFTConv`) |
| cells / pixels | `kernel_sz` (`HealPixConv`, `LargeConv`, `HealPixFFTConv`, `HealPixDivCurl`), `compact_kernel_sz`, `sigma_pix` (`HealPixDivCurl`, `kernel_gaussian`), `scale_pix` (other `kernel_*`), `HealPixFFTConv.set_gaussian(sigma, unit="pixel")` (pixels of the tangent grid) |
| your choice | `HealPixGeoFFT` `scale` / `radius` with `units` in `"m"`, `"km"`, `"deg"`, `"pix"` (default `"m"`) |

Widths are **standard deviations** (one-axis RMS widths) of a Gaussian, not
FWHM (FWHM = 2.355 σ). "One cell" is the nominal cell size,
`sqrt(4π / (12 · 4**L))` radians, about 6.4 km at level 10 on the Earth.

### Kernel conventions

A fixed kernel is written in five ways, with different arguments, units and
angle directions; see {doc}`/conv/kernel_conventions`.

## Values: shapes, NumPy and Torch

There is no single rule. The most common one is "the last axis is the cell
axis, leading axes are a batch, NumPy in gives NumPy out, Torch in gives
Torch out on the same device", but:

| Tool | Accepted input | Output |
|---|---|---|
| `HealPixDown`, `HealPixUp` | `[N]` or `[B, N]` only | **tuple** `(y, cell_ids_out)`; cast to the operator `dtype` (float32 by default) |
| `HealPixResampler`, `resample_healpix` | `[..., N]` | tuple `(y, out_cell_ids)` |
| `HealPixDecomp` | `[..., N]` | a `HealPixPyramid` (bands in sorted cell-id order) |
| `HealPixConv`, `LargeConv`, `HealPixFFTConv` | `[N]`, `[B, N]` or `[B, C_in, N]` | `[B, C_out_total, N]` (a 2-D input gives a 3-D output); a 1-D input gives `[C_out_total, N]`, or `[N]` when there is one output channel |
| `HealPixDivCurl` | `[..., 2, N]` (u east, v north) | `[..., 2, N]` (div, curl) |
| `gradient` and variants | 1-D only | NumPy or Torch |
| radial, Gaussian and directional filters | `[..., N]` | `[..., len(domain)]`; NumPy output is at least float64 |
| `neighbour_reduce` | `[..., N]` | `[..., len(domain)]`; the dtype depends on the reduction (`count` is int64, `any`/`all` bool) |
| `HEALPixSHT` | `[..., 12·nside²]`, NumPy or Torch | always Torch, on the transform's device |
| `alm_latlon` | `[..., n_pix]` sorted by rings | Torch, float64 / complex128 |
| `binary_dilation`, `binary_erosion` | a list of active cell ids | sorted unique cell ids (NumPy) |
| `connected_components` | 1-D boolean mask | `(labels, n_components)` |
| `minkowski_functionals`, `minkowski_curves` | Torch `[B, N, N]` (planar images) | Torch |
| `get_interp_val` | full NESTED map, lon/lat in degrees | NumPy |
| DINO | `[N, C]` or `[N]` | NumPy (`DINOEmbedding`) |

## Differentiability

"Differentiable" below means differentiable with respect to the input values
through `torch.autograd` (geometry, weights and kernel shapes are constants).

- **Differentiable:** the convolutions (`HealPixConv`, `LargeConv`,
  `HealPixFFTConv`, `HealPixGeoFFT`, `HealPixPyramidConv`), `HealPixDown`
  (all modes; `maxpool` sends the gradient to the largest child),
  `HealPixUp`, `HealPixDecomp`, `HealPixResampler`, `HEALPixSHT`,
  `alm_latlon` (Torch input), `LocalFFT`, the metric filters and
  `apply_filter_operator`, `directional_filter`, `gradient`,
  `HealPixDivCurl`, the HEALPix Minkowski functionals after setup,
  `neighbour_reduce` with `mean`, `median`, `std`, `min`, `max`, `sum`.
- **Not differentiable:** `HealPixWideConv` (it detaches its input), the
  operator builders (NumPy), `neighbour_reduce` with `count`, `mode`, `any`,
  `all`, morphology, components, `get_interp_*`, DINO (inference mode),
  `powerspectra`, `ps`.
- **Not differentiable with respect to the width:** no tool gives a gradient
  with respect to `sigma_m`, `radius_m` or the parameters of a kernel callable.

## GPU

Tools that take `device=` (the convolutions, `HealPixDown`, `HealPixUp`,
`HealPixDecomp`, `HealPixResampler`, `HEALPixSHT`, `LocalFFT`,
`HealPixDivCurl`; most are `torch.nn.Module`s, `HEALPixSHT` and
`HealPixWideConv` are plain classes); most default to CUDA when it is available
(`HealPixGeoFFT` defaults to the CPU). The neighbourhood tools build their
geometry on the CPU with NumPy and run the reduction on the device of the
input tensor. No GPU timing is published for any tool yet.

## Missing values (NaN)

| Behaviour | Tools |
|---|---|
| NaN-aware: left out and the weights renormalised | `radial_filter`, `gaussian_filter` (`"present"`), `apply_filter_operator(renormalize_missing=True)`, `directional_filter` (with `normalize=True`), `HealPixGeoFFT` (normalized mode), `HealPixPyramidConv` (`"normalized"`), `HealPixDecomp.compute_weighted`, `HealPixResampler`, `HealPixDown(mode="mean", skipna=True)`, `gradient` (NaN neighbours dropped), `HealPixConv(missing="local_mean")` (completed with the valid neighbours or the local mean), `HealPixWaveletSpectrum` and `HealPixIsotropicWaveletSpectrum` |
| NaN propagates | `HealPixConv` (default `missing="centre"`), `LargeConv`, `HealPixFFTConv` (one NaN makes the whole output NaN), `HealPixWideConv`, `HealPixDivCurl`, `HealPixDown` (`smooth`, `maxpool`), `HealPixUp`, `HEALPixSHT`, `alm_latlon`, `LocalFFT`, `neighbour_reduce` (`mean`, `median`, `std`, `min`, `max`), `get_interp_val` |

## Spherical-harmonic convention

One convention, orthonormal, the same as `healpy`:

$$a_{\ell m} = \int f(\theta, \varphi)\, Y_{\ell m}^*(\theta, \varphi)\, d\Omega$$

$$C_\ell = \frac{1}{2\ell+1} \left[ |a_{\ell 0}|^2 + 2 \sum_{m=1}^{\ell} |a_{\ell m}|^2 \right]$$

`alm` arrays use the m-major layout of `healpy`, with
`(lmax+1)(lmax+2)/2` coefficients.

## Geometry

Cell centres, vertices and neighbours come from
[healpix-geo](https://healpix-geo.readthedocs.io/); ellipsoidal distances come
from `pyproj`. `healpy` is not a dependency; one exported function,
`build_healpix_adjacency`, needs it (see {doc}`installation`).
