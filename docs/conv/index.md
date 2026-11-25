# A. Convolution & filtering

Every way to convolve or filter a field on HEALPix cells, in one place:
learned convolutions for neural networks, wide fixed kernels, FFT-based
convolutions, and metric filters whose scale is given in metres.

## First questions

1. **Are the weights learned** (a layer of a neural network)? Small kernel →
   `HealPixConv`; wide receptive field → `LargeConv`; one large kernel on one
   patch → `HealPixFFTConv`.
2. **Is the scale a physical distance** (metres) and the kernel fixed? Gaussian
   or radial → `gaussian_filter` / `radial_filter`; it depends on the bearing
   → `directional_filter`; many applications, total conserved or a land
   barrier → the filter operators.
3. **Is the fixed kernel wide** (tens of cells)? See
   {doc}`compare_wide`.
4. **Do the data have gaps** (NaN, clouds, land)? Prefer the tools that say
   "NaN-aware" below: the metric filters, `HealPixGeoFFT`,
   `HealPixPyramidConv`.

Comparisons: {doc}`compare_gaussian` · {doc}`compare_wide` ·
{doc}`compare_fft` · {doc}`kernel_conventions`.

## I want to …

```{include} ../_generated/decisions_A.md
```

The unweighted neighbourhood statistics (mean, median, max over a radius) are
in {doc}`/stats/neighbour_reduce`; changing resolution is in
{doc}`/other/index`.

```{toctree}
---
maxdepth: 1
caption: Compare
---
compare_gaussian
compare_wide
compare_fft
kernel_conventions
```

```{toctree}
---
maxdepth: 1
caption: Learned convolutions
---
convol
large_conv
fft_conv
patches
```

```{toctree}
---
maxdepth: 1
caption: Wide fixed kernels
---
wide_conv
pyramid_conv
pyramid_conv_maths
geo_fft
```

```{toctree}
---
maxdepth: 1
caption: Metric filters
---
radial_filter
filter_operators
multiorder_filter
filter_cache
directional_filter
```
