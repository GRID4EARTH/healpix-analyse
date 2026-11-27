# Kernel conventions

A fixed kernel is written in five different ways, depending on the tool. The
arguments, their units and the direction of the angle differ: a kernel written
for one tool cannot be passed to another without care.

| Tool | Callable | Distance | Angle |
|---|---|---|---|
| `kernel_gaussian`, `kernel_exponential`, `kernel_lorentzian`, `kernel_beta`, `kernel_anisotropic_gaussian` → `HealPixKernelPyramid` ({doc}`pyramid_conv`) | `fn(rho_pix, phi_rad)` | `rho_pix`: cells of the finest band (the footprint grows with the band) | `phi_rad = atan2(y, x)` from the stencil x-axis, counter-clockwise |
| `HealPixGeoFFT` / `geo_fft_convolve` ({doc}`geo_fft`) | a name of the `KERNELS` registry with `scale=`, or `k(rho, phi)` | `rho` in `units` (`"m"` default, `"km"`, `"deg"`, `"pix"`) | `phi`: azimuth from North, clockwise, from the output cell to the input cell, radians |
| `radial_filter`, `radial_filter_operator` ({doc}`radial_filter`) | `kernel(distance_m)` | metres, WGS84 geodesic; a 1-D NumPy array | none |
| `directional_filter` ({doc}`directional_filter`) | `kernel(distance_m, relative_bearing_rad)` | metres, WGS84 geodesic | bearing relative to `azimuth_rad` (0 = North, clockwise), wrapped to [−π, π) |
| `HealPixWideConv` ({doc}`wide_conv`) | `kernel_image` on the face lattice; `from_function(fn)` with `fn(x_m, y_m)`; `from_radial(fn)` with `fn(r_m)`; `from_grid(grid, pixel_size_m, ...)` | metres: east/north offsets or great-circle distance; or lattice steps for the image (row `j`, column `i`) | east/north axes |

Learned kernels are arrays, not callables: `HealPixConv.set_kernel(W)` takes
`W` of shape `[C_in, C_out, P]` or `[G, C_in, C_out, P]` on the
`kernel_sz × kernel_sz` stencil (`P = kernel_sz²`), and `HealPixFFTConv` holds
`weight` of shape `[C_in, C_out, K, K]` on the tangent grid.

## Pitfalls

- **Same name, different profile.** The `"moffat"` kernel of `HealPixGeoFFT`
  is `(1 + (rho/s)²)^(−β)` with `beta=2.5` by default; `kernel_beta` is
  `(1 + (rho/s)²)^(−β/2)` with `beta=2.0`.
- **Angles.** `kernel_anisotropic_gaussian(..., orientation_rad)` measures the
  angle from the stencil x-axis, counter-clockwise; `HealPixGeoFFT` and
  `directional_filter` measure azimuths from North, clockwise. An anisotropic
  kernel does not carry over between them as is; an isotropic one does.
- **Units.** `kernel_*` work in cells, the filters in metres, `HealPixGeoFFT`
  in the unit you choose.
- **Normalisation.** `radial_filter` normalises by default; `HealPixGeoFFT`
  normalises in its default mode; the `kernel_*` taps and `HealPixWideConv`
  are not normalised; `directional_filter` has `normalize=False` by default.
