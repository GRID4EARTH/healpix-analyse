# Wide receptive field: which convolution?

A compact stencil (`HealPixConv`) reaches `kernel_sz // 2` cells. Five tools
reach further. They split on two questions: **is the kernel learned or
fixed**, and **what is the domain**.

| Tool | Kernel | Domain | Gaps (NaN) | Channels | Differentiable | Accuracy and limits |
|---|---|---|---|---|---|---|
| `LargeConv` ({doc}`large_conv`) | **learned** compact kernel on a coarsened grid (Down → conv → Up) | whole sphere or regional with a halo | propagates | multi-channel, G gauges | yes | not conservative (total 0.92–1.05 for one cell); Gaussian recipe within 3.8e-4 at the calibration cell |
| `HealPixFFTConv` ({doc}`fft_conv`) | **learned** large kernel (e.g. 33×33) on a tangent grid | one patch, radius 10° by default, with a halo | one NaN makes all NaN | multi-channel | yes | approximate projection round trip; blur floor about 0.58 pixel; not gauge-equivariant |
| `HealPixWideConv` ({doc}`wide_conv`) | **fixed**, written as an image, a formula in metres or a raster | regional, refitted per domain | propagates | one | no | exponential peak about 0.82, tail floor about 1e-4 (documented) |
| `HealPixKernelPyramid` + `HealPixPyramidConv` ({doc}`pyramid_conv`) | **fixed**, one analytic or calibrated kernel per band (`calibrate_joint` fits one wide kernel across bands) | whole sphere or regional | **NaN- and weight-aware** | multi-channel | in the input | per-band approximation, 15 % to 22 % RMS in the documented tests |
| `HealPixGeoFFT` ({doc}`geo_fft`) | **fixed**, geodesic kernel (9 named or any callable) | a block inside **one base face** | **ratio of two FFTs** | one per call (any leading dims) | in the input | exact sampling of the kernel; drifts near block edges |

## Rules of thumb

- **Learned, whole sphere or large domain** → `LargeConv`.
- **Learned, one patch, very large kernel** → `HealPixFFTConv`.
- **Fixed, inside one base face, gaps** → `HealPixGeoFFT`.
- **Fixed, gaps or several channels, any domain** → `HealPixPyramidConv`.
- **Fixed, single channel, kernel in metres or a raster, no gaps** →
  `HealPixWideConv`.
- **No learning, a Gaussian or radial kernel in metres, moderate width** →
  the metric filters ({doc}`radial_filter`), whose cost grows with the number
  of cells under the kernel.

The Gaussian case is compared in detail in {doc}`compare_gaussian`; the FFT
tools in {doc}`compare_fft`.
