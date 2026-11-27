# FFT convolution: which tool?

Three ways to convolve by FFT. The transforms and spectra themselves are in
{doc}`/sht/fft_local`.

| | `HealPixFFTConv` ({doc}`fft_conv`) | `HealPixGeoFFT` ({doc}`geo_fft`) | `LocalFFT` by hand ({doc}`/sht/fft_local`) |
|---|---|---|---|
| Grid of the FFT | gnomonic tangent grid of one patch (data **reprojected** by bilinear gridding) | the (i, j) lattice of one base face (**no reprojection**) | gnomonic tangent grid of one patch |
| Domain | one patch, radius 10° by default (`max_patch_radius_deg`), with a halo | a block of cells inside one base face; warning above 10° of extent | one patch, radius 10° by default |
| Kernel | **learned**, `[C_in, C_out, K, K]`, or `set_gaussian` | **fixed**: 9 named kernels or any `k(rho, phi)` of the geodesic distance | whatever you multiply the spectrum by |
| Channels | dense channel mixing | one kernel for all leading dims | any leading dims |
| Gaps (NaN) | one NaN makes the whole output NaN | NaN-aware (ratio of two FFTs) | NaN propagates |
| Conservation | `projection="conservative"`: totals; `"average"`: constants | normalized mode: constants | depends on `projection` |
| Exactness | approximate round trip (identity kernel ≠ identity), blur floor about 0.58 pixel | kernel sampled exactly; drifts near block edges | the inverse is interpolating, not least-squares |
| Differentiable | yes (input and kernel) | yes (input) | yes (input) |
| Default ellipsoid | `"sphere"` | `"sphere"` | `"sphere"` |

## Rules of thumb

- You **learn** the kernel → `HealPixFFTConv`.
- The kernel is **known**, the block sits in one base face, and there are
  **gaps** → `HealPixGeoFFT`.
- You need the spectrum itself, or an operation in Fourier space other than a
  convolution → `LocalFFT` (`fft`, multiply, `ifft`).
- The domain spans several base faces or the whole sphere → none of these;
  see {doc}`compare_wide`.
