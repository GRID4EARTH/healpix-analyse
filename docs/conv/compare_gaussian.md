# Gaussian smoothing: the eleven ways

Eleven tools can smooth a field with a Gaussian. They differ in how the width
is given, how narrow or wide it can be, what they conserve, how they treat
gaps, and on which domain they work. Start from the first row; move down only
when one of its limits applies to you.

## Which one?

- **A width in metres, on any domain, once** → `gaussian_filter`.
- **The same smoothing many times, conserving the total, or not crossing
  land** → `gaussian_filter_operator` (`normalization="source"`,
  `blocked_cells`).
- **Cells of different levels** → `multiorder_filter_operator`.
- **A wide Gaussian (several to tens of cells) on the whole sphere or a large
  domain** → `LargeConv.from_gaussian`.
- **Smoothing while going one level coarser** → `HealPixDown(mode="smooth")`
  (or `"smooth_conservative"` to keep the mean).
- **Inside one base face, exact geodesic kernel, with gaps** →
  `HealPixGeoFFT("gaussian")`.
- **One regional patch, as part of a learned FFT model** →
  `HealPixFFTConv.set_gaussian`.
- **Gaps and confidence weights at every scale** → `kernel_gaussian` +
  `HealPixPyramidConv`.

## Comparison

| Tool | Width given as | Domain | Width range and accuracy | Conserves | Gaps (NaN) | Differentiable |
|---|---|---|---|---|---|---|
| `gaussian_filter` ({doc}`radial_filter`) | `sigma_m`, metres | whole sphere or regional | `width_rule="matched"`: requested width down to about half a cell; default `"sampled"` is too narrow below about one cell | constants (`"present"`); mass lost at the domain edge with `"full_disc"` | left out, weights renormalised | in the values |
| `gaussian_filter_operator` ({doc}`filter_operators`) | `sigma_m`, metres | whole sphere or regional, optional land barrier | as `gaussian_filter` | the **total** with `normalization="source"`; constants with `"present"` | `apply_filter_operator(renormalize_missing=True)` | application only |
| `multiorder_filter_operator` ({doc}`multiorder_filter`) | `sigma_m`, metres through water | 2:1-balanced mixed-level leaves | — | the total (`"source"`, default) | through `apply_filter_operator` | application only |
| `radial_filter` with a Gaussian `kernel` ({doc}`radial_filter`) | your `kernel(distance_m)` and `radius_m` | whole sphere or regional | sampled at cell centres, like `"sampled"` | constants (`normalize=True`) | left out | in the values |
| `HealPixDown(mode="smooth")` ({doc}`/other/down`) | `sigma_deg`, `radius_deg`, degrees | whole sphere or regional; output **one level coarser** | — | nothing (`"smooth_conservative"`: the global mean) | NaN spreads (`smooth`) | yes |
| `LargeConv.from_gaussian`, `gaussian_plan`, `down_up_blur` ({doc}`large_conv`) | `sigma` in radians or `sigma_m` in metres | whole sphere or regional with a halo | from about 2.2 cells (equator) to about 32 cells (level 10, default); within 3.8e-4 at the calibration cell, up to 8 % spread across the domain at 3 cells | constants; the total of one cell is 0.92 to 1.05 | propagates | yes (kernel frozen) |
| `HealPixFFTConv.set_gaussian` ({doc}`fft_conv`) | `sigma` in tangent-grid pixels or radians | one patch (radius 10 degrees by default) with a halo | not below the blur floor (about 0.58 pixel; a warning is given) | the total with `projection="conservative"`, constants with `"average"` | one NaN makes all NaN | yes |
| `HealPixGeoFFT("gaussian", scale=...)` ({doc}`geo_fft`) | `scale` in m, km, deg or pix | a block inside one base face | kernel sampled at the true geodesic distance; drifts near the block edges | constants (normalized mode) | ratio of two FFTs | in the values |
| `kernel_gaussian` + `HealPixPyramidConv` ({doc}`pyramid_conv`) | `sigma_pix`, cells of the finest band | whole sphere or regional | per-band approximation (22 % RMS on white noise at sigma 1.2 cells, documented) | constants (`"normalized"`) | normalised convolution | in the values |
| `HealPixWideConv.from_radial` ({doc}`wide_conv`) | your `fn(r_m)`, metres | regional, cell ids at call time | fitted per band; peak and tail approximate | no normalisation | propagates | no |
| `HealPixConv.set_kernel` by hand ({doc}`convol`) | taps on a `kernel_sz` stencil | whole sphere or regional | reach `kernel_sz // 2` cells | as your taps | propagates | yes |

All widths are standard deviations (one-axis RMS widths), not FWHM. Cost is
discussed on each page; no GPU comparison has been measured yet.

## Notes

- `gaussian_filter` and `LargeConv.from_gaussian` use the same cell size to
  convert `sigma_m`, so a width in metres means the same in both.
- `HealPixDown` changes the level: use it when you want a coarser map, not to
  smooth at the same level.
- `HealPixWideConv` is not differentiable and has a single channel; for a
  plain Gaussian, prefer one of the rows above it.
