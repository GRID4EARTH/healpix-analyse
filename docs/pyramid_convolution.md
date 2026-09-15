# Pyramidal convolution — compact multiscale spherical filtering

`healpix_analyse.kernel_pyramid` and `healpix_analyse.pyramid_conv` combine
{doc}`decomp` (the exact-reconstruction Laplacian pyramid) with
{doc}`convol_doc` (`HealPixConv`'s gauge-equivariant compact stencil) to
build masked, NaN-aware, multiscale spherical filters that stay `O(N)` in
the number of pixels.

This page is the guide, the math, the API reference, the implementation
notes, and the validation/limits summary in one place. It does not claim to
reproduce the specific construction of Farbman, Fattal & Lischinski,
*"Convolution Pyramids"*, ACM TOG 30(4), 2011 — see
[Relation to "Convolution Pyramids" and honest scope](#relation-to-convolution-pyramids-and-honest-scope)
below.

---

## Quick start

```python
import numpy as np
import torch
from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.kernel_pyramid import HealPixKernelPyramid, kernel_gaussian
from healpix_analyse.pyramid_conv import HealPixPyramidConv

level = 8
decomp = HealPixDecomp(level=level, ellipsoid="sphere", Jmax=3)

# One fixed 5x5 Gaussian-shaped kernel per pyramid band, sharing decomp's
# exact geometry, gauge convention and cell-id domain.
kernel_pyramid = HealPixKernelPyramid.from_kernel(
    decomp, kernel_gaussian(sigma_pix=1.2), compact_kernel_sz=5,
)

pconv = HealPixPyramidConv(decomp, kernel_pyramid, mode="normalized")

npix = 12 * 4 ** level
x = np.random.randn(npix)
x[np.random.rand(npix) > 0.9] = np.nan   # missing data / land mask / etc.

y, support = pconv(x, return_support=True)
```

`y` is finite wherever the convolution had any support at all, and `NaN`
(the default; pass `restore_mask=False` for `0`) elsewhere. See
[examples/pyramid_conv_quickstart.py](../examples/pyramid_conv_quickstart.py)
for a runnable, slightly more complete version of this example.

---

## A. Mathematical foundations

### A.1 The target operator, and what a pyramid can and cannot give you for free

Let `W` be the analysis operator of a `HealPixDecomp` (its stacked
`Down`/`Up` chain) and `S` its synthesis operator, so `S W = I` exactly —
this is the exact-reconstruction identity `decomp.invert(decomp.compute(x))
== x` already documented in {doc}`decomp`. Let `K` be a target convolution
kernel. The composite pipeline this module implements is

```text
y = S · B · W · x
```

for some operator `B` acting on pyramid coefficients. This equals plain
convolution, `y = K x`, only for the specific choice `B = W K S` — and that
operator is, in general, **dense across bands**: `B`'s off-diagonal blocks
couple detail at one scale to neighbouring scales. **Decomposition is not
diagonalization**: nothing about `S W = I` implies that a per-band-only `B`
(no inter-band terms) reproduces `K x` well.

`HealPixKernelPyramid` builds exactly that: a **block-diagonal**
approximation of `B`, one small compact `HealPixConv` kernel acting
independently on each band, with **no cross-band terms**. This is a
deliberate simplification whose approximation error must be *measured*, not
assumed — see [Section E](#e-validation-results-and-honest-limits).

### A.2 NaN/weight propagation and normalized convolution

For missing data, `HealPixDecomp.compute_weighted(x, weights=None)` analyzes
two channels through the *identical* linear operator `W`:

```text
q = W(m ⊙ x_safe)          (data channel: missing values zeroed)
m̃ = W(m)                   (weight channel: same operator, weights only)
```

where `m` is the (0/1 or continuous) confidence weight and `x_safe` replaces
missing values with `0`. Because `W` is data-independent (built once from
geometry alone), running it on `m ⊙ x_safe` and on `m` is well-defined and
requires no new machinery.

A kernel pyramid `B_K` (per-band `HealPixConv`s) is then applied
*identically* to both channels — `HealPixPyramidConv.apply_pyramid` never
lets `q` and `m` see different kernels — and the result is combined with a
**single division after synthesis**:

```text
y = S(B_K q) / S(B_K m̃)
```

never per band. This is the standard normalized-convolution convention: a
constant field `x ≡ c` behind an arbitrary mask satisfies `q = c · m`
*band by band* (tested in `tests/test_decomp_weighted.py::test_constant_plus_mask_gives_proportional_bands`),
so `y` reconstructs back to exactly `c` wherever the synthesized support
`S(B_K m̃)` is non-zero, and is set to `NaN` (or `0`) exactly where it is
not. The detail bands of `m̃` are **signed correction terms**, not per-band
confidences in `[0, 1]` — never clip or threshold them individually; only
the fully-synthesized `S(B_K m̃)` is a meaningful support/confidence map.

### A.3 Signed kernels and positivity

Nothing in `HealPixKernelPyramid`/`HealPixPyramidConv` requires kernel
weights to be non-negative. A signed kernel (e.g. a Mexican-hat/Laplacian
edge detector) is fully supported in `mode="signed"` (no masking, no
division — see [Section C](#c-api-reference)); in `mode="normalized"`, a
signed kernel is applied identically to the data and weight channels, but
note that a *negative* synthesized weight `S(B_K m̃)` at some pixel makes the
division well-defined but not a meaningful "confidence" any more — this is
an open point, see [Section E](#e-validation-results-and-honest-limits).

### A.4 Anisotropy and gauges

A kernel profile is a function `K(rho_pix, phi_rad)` evaluated once, on the
fixed North-Pole stencil, at each band's own resolution (see
`_stencil_pixel_polar`, which reuses `HealPixConv`'s own
`_local_kernel_grid` so the evaluation points are guaranteed identical to
the runtime stencil). `HealPixConv`'s per-pixel gauge rotation
(`gauge_type`, `n_gauges`, `singularity_lonlat`/`ref_direction` — see
{doc}`convol_doc`) then carries that fixed-frame anisotropy into every
output pixel's own local tangent frame consistently, the same way
`HealPixConv` does for a learned kernel. `HealPixKernelPyramid.from_kernel`
forwards `gauge_type`/`n_gauges`/`singularity_lonlat`/`ref_direction`
unchanged to every band's `HealPixConv`.

### A.5 True sphere vs. ellipsoid

Every geometry call in this module (`HealPixConv`, and the independent
validation oracle) defaults to `ellipsoid="sphere"`, not the codebase-wide
`"WGS84"` default used elsewhere (e.g. `HealPixDecomp`'s own default). Pass
a `HealPixDecomp` explicitly constructed with `ellipsoid="sphere"` if you
want the kernel pyramid's geometry assumption to match the decomposition's
own `Down`/`Up` geometry; mixing `ellipsoid="WGS84"` in `decomp` with the
kernel pyramid's `"sphere"` stencils silently mislabels an ellipsoidal
distance as a spherical one and is not currently checked automatically.

---

## B. Implementation notes

- **`healpix_analyse.decomp.HealPixWeightedPyramid`** (new): pairs a `q` and
  an `m̃` `HealPixPyramid`, both produced by `HealPixDecomp.compute_weighted`.
  `HealPixDecomp.invert` dispatches on its argument type: a plain
  `HealPixPyramid` still takes the pre-existing exact-inverse path
  unchanged; a `HealPixWeightedPyramid` takes the new
  synthesize-both-then-divide-once path (`_invert_weighted`).
  `compute`/`invert`'s pre-existing behavior on finite data is unchanged —
  covered by `tests/test_decomp_weighted.py::test_plain_compute_invert_unaffected`.
- **`healpix_analyse.kernel_pyramid.HealPixKernelPyramid`**: one
  `HealPixConv(in_channels=1, out_channels=1, ...)` per band, with a fixed
  (`requires_grad=False`) kernel set via `HealPixConv.set_kernel`. Built
  either analytically (`from_kernel`, evaluating a Python callable on the
  exact stencil geometry) or by least-squares calibration against a
  reference operator (`calibrate`, see [Section D](#d-calibration-how-it-works-and-its-limits)).
- **`healpix_analyse.pyramid_conv.HealPixPyramidConv`**: an `nn.Module`
  wrapping a `(decomp, kernel_pyramid)` pair. `apply_pyramid` applies the
  per-band kernels without synthesizing; `forward` adds the
  compute_weighted → apply → invert(divide-once) pipeline (`"normalized"`
  mode, default) or the plain compute → apply → invert pipeline
  (`"signed"` mode, for fully finite data and/or signed kernels where
  dividing by a weight channel would be the wrong operation).
- **`healpix_analyse.validation`**: an *independently implemented* (not
  reusing `HealPixConv`'s bilinear-stencil-binding machinery) brute-force
  reference convolution, `direct_spherical_convolution`, built from true
  pixel centres (`healpix_geo.nested.healpix_to_lonlat`) and a `scipy`
  KD-tree neighbour search on unit vectors, evaluating the kernel against
  exact great-circle angular distance. Used only for testing/calibration,
  not as a fast path — see its module docstring for the explicit
  isotropic-only scope.
- **Complexity**: every step (`HealPixDown`/`HealPixUp`'s sparse matrices,
  `HealPixConv`'s fixed `P`-tap gather-and-multiply) is `O(N)` in total
  pixel count for a fixed `compact_kernel_sz`, with no dense `N×N`
  matrices and no full-resolution `expand()`+`stack()` anywhere in the
  `HealPixPyramidConv` forward path. Empirically confirmed near-linear
  scaling is reported in [Section E](#e-validation-results-and-honest-limits).
- **PyTorch conventions**: shapes follow `HealPixDecomp`/`HealPixConv`
  conventions (`[N]`, `[B, N]`, `[B, C, N]`); `float64` is used in the test
  suite for identity/gradient checks, `float32` for the benchmark. The
  `isfinite`-based missing-value mask itself carries no gradient (as with
  any boolean masking decision), but surviving finite values of `data` and
  `weights` stay fully differentiable — confirmed in
  `tests/test_decomp_weighted.py::test_gradient_flows_through_weighted_pipeline`
  and `tests/test_pyramid_conv.py::test_gradient_flows_through_pyramid_conv`.

---

## C. API reference

```python
# healpix_analyse.decomp (extended)
pyramid = decomp.compute_weighted(data, weights=None)      # -> HealPixWeightedPyramid
data    = decomp.invert(pyramid_or_weighted_pyramid,
                         restore_mask=True, eps=1e-8)       # dispatches on type

# healpix_analyse.kernel_pyramid
from healpix_analyse.kernel_pyramid import (
    HealPixKernelPyramid,
    kernel_gaussian, kernel_exponential, kernel_lorentzian, kernel_beta,
    kernel_anisotropic_gaussian,
)

kp = HealPixKernelPyramid.from_kernel(
    decomp, kernel, compact_kernel_sz=5, gauge_type="phi", n_gauges=1,
    singularity_lonlat=None, ref_direction=None, bands=None,
    dtype=None, device=None,
)
kp = HealPixKernelPyramid.calibrate(
    decomp, reference_fn, compact_kernel_sz=5, gauge_type="phi",
    n_probes=128, n_excitations=4, seed=0, ridge=1e-6,
)
conv_bands = kp.apply(bands)   # apply each band's kernel; no synthesis

# healpix_analyse.pyramid_conv
from healpix_analyse.pyramid_conv import HealPixPyramidConv

pconv = HealPixPyramidConv(decomp, kp, mode="normalized")  # or mode="signed"
y = pconv(x, weights=None, return_support=False, restore_mask=True, eps=1e-8)
y, support = pconv(x, return_support=True)                  # normalized mode only
conv_pyramid = pconv.apply_pyramid(pyramid_or_weighted_pyramid)

# healpix_analyse.validation
from healpix_analyse.validation import (
    direct_spherical_convolution, direct_reference_operator_factory,
    smooth_test_field,
)
y_ref = direct_spherical_convolution(
    x, cell_ids, level, kernel_iso, weights=None, kernel_sz=5,
    ellipsoid="sphere", eps=1e-8, restore_mask=True, normalize=True,
)
```

Kernel profiles are `fn(rho_pix, phi_rad) -> weight`, in *pixel units*
(`rho_pix = angular_distance / alpha_pix`, so the same callable keeps the
same shape in pixels at every band, and its physical footprint grows
automatically as the pyramid coarsens). `normalize=False` on
`direct_spherical_convolution` matches `HealPixConv`'s own raw
(unnormalized) tap-weighted-sum convention — use this when comparing a
single kernel-pyramid band directly, as the tests do; `normalize=True`
(default) matches the masked/weighted `HealPixPyramidConv` convention.

---

## D. Calibration: how it works and its limits

`HealPixKernelPyramid.calibrate` fits one band's `P` kernel taps by least
squares against a `reference_fn(cell_ids, level) -> operator` (typically
built with `direct_reference_operator_factory`). For each band, it draws
`n_excitations` independent random excitation fields, evaluates
`HealPixConv`'s own bilinear-interpolated per-tap response
(`x_interp[i, p]`, obtained by running the *same* `HealPixConv` with a
one-hot kernel — this guarantees the fit is against the exact runtime
stencil, not an approximation of it) at `n_probes` random output pixels,
and solves a small ridge-regularised least-squares problem.

**This is single-band-at-a-time calibration** — it does not jointly
optimize across bands, and it fits one band's target *at that band's own
resolution*. Two things follow, both confirmed by
`tests/test_kernel_pyramid.py`:

- When the reference target's own support is comparable to the band's
  compact stencil (e.g. calibrating a 5×5 band against a `sigma_pix≈1.2`
  Gaussian), calibration recovers essentially the same accuracy as directly
  evaluating the analytic kernel (`test_calibrate_recovers_a_representable_target`,
  measured ≈5.1% vs ≈4.8% RMS — see Section E).
- Calibrating a single band's compact 5×5 kernel against a target many
  times wider than its own support (e.g. `sigma_pix=6`) does **not**
  meaningfully close the gap (`test_calibrate_on_a_much_wider_target_does_not_silently_claim_success`,
  measured ≈76% RMS residual — this is printed, not asserted tight, on
  purpose). Reproducing a wide target well from a cascade of small per-band
  kernels requires *joint*, cross-band least-squares optimization (this is
  the actual contribution of Farbman/Fattal/Lischinski 2011) — that is
  **not implemented here**; see
  [Relation to "Convolution Pyramids" and honest scope](#relation-to-convolution-pyramids-and-honest-scope).

---

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

**Calibration**: see [Section D](#d-calibration-how-it-works-and-its-limits)
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
{doc}`convol_internals`) is non-trivial at higher levels and should be
amortized across many forward calls, not repeated per call.

**Known, currently unaddressed limitations** (stated plainly, not silently
worked around):

1. **Block-diagonal only.** No inter-band coupling is modeled or corrected
   for; see [Section A.1](#a-1-the-target-operator-and-what-a-pyramid-can-and-cannot-give-you-for-free)
   and [Section D](#d-calibration-how-it-works-and-its-limits).
2. **`calibrate` is single-band, not joint.** It cannot make a compact
   per-band kernel reproduce an arbitrarily wide target; a genuine
   multi-level joint calibration (the actual contribution of the cited
   1  2011 paper) is not implemented.
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
5. **`ellipsoid` consistency is not auto-checked.** See
   [Section A.5](#a-5-true-sphere-vs-ellipsoid).
6. **No memory-chunked/streaming internals beyond per-band processing.**
   Each band is processed as one dense tensor; very large single bands are
   not further chunked internally.
7. **No GPU benchmark**, for the reason stated above.
8. **`HealPixKernelPyramid.calibrate` is comparatively slow** (a Python
   loop over `P` taps × `n_excitations` per band, each a full
   `HealPixConv` forward pass) — fine for offline calibration at moderate
   `compact_kernel_sz`, not intended as a hot path.

---

## Relation to "Convolution Pyramids" and honest scope

This module is **inspired by**, but does not claim to reproduce, Z.
Farbman, R. Fattal, D. Lischinski, *"Convolution Pyramids"*, ACM
Transactions on Graphics 30(4), 2011. The cited paper's core technique —
jointly optimizing every pyramid level's kernel by least squares so that
the *cascade* reproduces one global target operator (their examples include
large-radius blurs and gradient-domain/Poisson-type operators) — is
explicitly **not implemented** here. What this module provides instead:

- an exact-reconstruction pyramid (`HealPixDecomp`, pre-existing) reused
  as-is;
- a **block-diagonal**, per-band-only kernel pyramid, built either
  analytically from a chosen kernel family or by **independent per-band**
  least-squares calibration (not the joint, cross-band optimization of the
  cited paper);
- correct NaN/weight propagation through that block-diagonal pyramid with
  the standard single-division-after-synthesis normalized-convolution
  convention.

No numerical claim in this page compares against the cited paper's own
published results, and no claim of exact reproduction of its method is
made anywhere in this codebase.
