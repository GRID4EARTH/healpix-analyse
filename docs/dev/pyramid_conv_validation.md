# Pyramidal convolution: validation report

Moved from {doc}`/conv/pyramid_conv`.

## E. Validation results and honest limits

Numbers below are measured by the test suite (`pytest -s
tests/test_kernel_pyramid.py`) and `scripts/benchmark_pyramid_conv.py`, on
this delivery's CPU-only development machine (no GPU was available — see
the benchmark script's own docstring). They are not claims about any other
hardware.

**Per-band kernel fidelity** (`HealPixConv`'s bilinear-stencil-binding vs.
the independent brute-force oracle in `healpix_analyse.validation`, raw
unnormalized convolution, level=5, `compact_kernel_sz=5`):

| Test field | Gaussian σ=1.2px | Exponential | Lorentzian σ=1.2px | Beta(β=3) |
|---|---|---|---|---|
| Smooth (low-order spherical function) | ≲15% RMS (all pass the 15% bound) | ≲15% | ≲15% | ≲15% |
| White noise (single-pixel-scale content) | **22.3% RMS** | not separately measured | not separately measured | not separately measured |

Kernel size comparison (Lorentzian, σ=2.0px, smooth field): 5×5 = **18.4%**
RMS, 7×7 = **15.4%** RMS — a modest, not dramatic, improvement for a
kernel whose tail genuinely needs more support.

**Why the smooth-field and white-noise numbers differ so much**: this is a
real, measured property of `HealPixConv`'s own discretization, not a test
artifact (see `test_single_band_white_noise_sensitivity_is_characterized`
and the module docstring of `healpix_analyse.validation`). `HealPixConv`
binds data to its fixed stencil by *bilinear interpolation* at rotated
positions that rarely land exactly on neighbouring pixel centres, which
closely tracks a continuous kernel's action on spatially smooth content but
departs substantially from a nearest-pixel quadrature reference on
single-pixel-scale content. **Practical implication**: treat this
convolution as accurate for band-limited/smooth fields (a few percent RMS);
do not extrapolate that accuracy to per-pixel claims on rough,
noise-dominated data without checking on data of the relevant roughness.

**Calibration**: see [Section D](../conv/pyramid_conv_maths.md#d-calibration-how-it-works-and-its-limits)
above for the two headline numbers (≈5% on a representable target, ≈76%
residual on a target far wider than one band's own support).

**Masking/normalization identities** (exact, machine-precision, all
in `tests/test_decomp_weighted.py` and `tests/test_pyramid_conv.py`):

- Plain `compute`/`invert` reconstruction is unaffected by this feature
  (≤1e-10 max abs error at level 3–4, float64).
- Constant map behind an arbitrary/contiguous mask: `q = c · m` band by
  band, to ≤1e-9; `HealPixPyramidConv` reconstructs exactly `c` at every
  supported pixel, to ≤1e-4 (kernel-normalization floating-point level, not
  the `1e-9` of the pure-decomposition identity — the kernel pyramid's
  raw, unnormalized taps make this a slightly less tight identity than the
  decomposition-only one).
- A fully masked map reconstructs to all-`NaN` (`restore_mask=True`) or
  all-`0` (`restore_mask=False`).
- Gradients flow through the full `compute_weighted → apply → invert`
  pipeline and through `HealPixPyramidConv`; gradient at an
  always-masked-out input pixel is exactly zero.

**CPU benchmark** (2 physical cores available in this environment, `float32`,
`compact_kernel_sz=5`, `Gaussian(sigma_pix=1.2)`, `Jmax` chosen per level;
full `compute_weighted → apply_pyramid → invert` forward pass, geometry
cache warm after the first build):

| level | npix | Jmax | one-time build (s) | forward pass (ms) | forward / pixel (×1e-6 ms) |
|---:|---:|---:|---:|---:|---:|
| 5 | 12,288 | 2 | 0.02 | 33.5 | 2.73 |
| 6 | 49,152 | 3 | 8.08 | 122.9 | 2.50 |
| 7 | 196,608 | 3 | 27.1 | 492.4 | 2.50 |

The near-constant per-pixel forward cost across a 16× increase in pixel
count is consistent with the claimed `O(N)` complexity. **No GPU was
available to produce a GPU number**; do not assume these CPU figures
translate directly. The one-time geometry build cost (per `HealPixConv`,
cached to disk after the first run at a given configuration — see
{doc}`/dev/convol_internals`) is non-trivial at higher levels and should be
amortized across many forward calls, not repeated per call.

**Known, currently unaddressed limitations** (stated plainly, not silently
worked around):

1. **Block-diagonal only.** No inter-band coupling is modeled or corrected
   for; see [Section A.1](../conv/pyramid_conv_maths.md#pyr-a1)
   and [Section D](../conv/pyramid_conv_maths.md#d-calibration-how-it-works-and-its-limits).
2. **`calibrate` is single-band, not joint** — it cannot make a compact
   per-band kernel reproduce an arbitrarily wide target on its own. Use
   `calibrate_joint` ([Section D.1](../conv/pyramid_conv_maths.md#pyr-d1))
   for that: a genuine multi-band joint least-squares fit against one wide
   target, measured to cut the residual from ≈87% to ≈3% on the same
   target and per-band kernel size. `calibrate_joint` itself only supports
   `channels=1` and a full-sphere `decomp` for the fit — both currently
   unaddressed limitations, not silent approximations.
3. **No automated anisotropic/gauge validation oracle.** The independent
   reference in `healpix_analyse.validation` only supports isotropic
   kernels (see its module docstring for why); anisotropic kernels are
   checked qualitatively (`test_anisotropic_kernel_runs_and_is_direction_sensitive`),
   not against an independent numeric oracle.
4. **Signed-kernel normalized mode is not fully characterized.** Nothing
   prevents a signed kernel from producing a negative synthesized "weight"
   `S(B_K m̃)` in `mode="normalized"`, which is not a meaningful confidence
   value; use `mode="signed"` for signed kernels on fully finite data
   instead.
5. **`ellipsoid` consistency across `decomp`/kernel pyramid is not
   auto-checked** (using `"sphere"` for one and `"WGS84"` for the other is
   still a silent mislabeling) — see [Section A.5](../conv/pyramid_conv_maths.md#pyr-a5).
   The *casing* of a given ellipsoid name is handled automatically as of
   this revision (`"wgs84"`/`"WGS84"`/`"Wgs84"` all resolve the same way);
   only mixing genuinely different ellipsoids remains unchecked.
6. **No memory-chunked/streaming internals beyond per-band processing.**
   Each band is processed as one dense tensor; very large single bands are
   not further chunked internally.
7. **No GPU benchmark**, for the reason stated above.
8. **`HealPixKernelPyramid.calibrate` is comparatively slow** (a Python
   loop over `P` taps × `n_excitations` per band, each a full
   `HealPixConv` forward pass) — fine for offline calibration at moderate
   `compact_kernel_sz`, not intended as a hot path.
9. **Multi-channel (`channels>1`) requires `n_gauges=1`.** The general
   multi-gauge, multi-channel case (`[G, C, C, P]` with `G>1`) is not
   supported — see [Section A.6](../conv/pyramid_conv_maths.md#pyr-a6). Every
   channel also shares the *same* kernel profile (block-diagonal, identical
   diagonal blocks); per-channel-distinct profiles are not supported.
