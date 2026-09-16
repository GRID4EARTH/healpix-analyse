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

> **Looking for “I have one wide kernel and I want to convolve with it”?**
> That is [`docs/wide_convolution.md`](wide_convolution.md) and
> `HealPixWideConv`: you give the kernel as an image, a formula in metres or
> a raster, and it derives the per-band kernels itself. This page documents
> the lower-level pair it is built on — analytic per-band profiles,
> multi-channel kernels, and NaN/weight-aware filtering.

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

### Where do I define my convolution?

Everything about *what the convolution does* is set in the two lines that
build `kernel_pyramid` above — nowhere else:

| I want to change... | Set this | Where |
|---|---|---|
| The kernel's **shape** (Gaussian, exponential, ...) | which `kernel_*` factory you pass, e.g. `kernel_gaussian` vs. `kernel_exponential`/`kernel_lorentzian`/`kernel_beta`/`kernel_anisotropic_gaussian` (all in `healpix_analyse.kernel_pyramid`), or your own `kernel(rho_pix, phi) -> weight` callable | 1st argument of `HealPixKernelPyramid.from_kernel(decomp, KERNEL, ...)` |
| The kernel's **width/scale** | the factory's own scale parameter — usually `sigma_pix`, but check the factory's docstring, it isn't always named that | inside the `KERNEL` call, e.g. `kernel_gaussian(sigma_pix=1.2)` |
| The kernel's **footprint size** (how many taps) | `compact_kernel_sz` (odd; 5 is the usual default) | `from_kernel(..., compact_kernel_sz=5)` |
| **How far** the filter can reach (into a hole, or a domain's own edge — see §A.2) | `Jmax` on the `HealPixDecomp` the kernel pyramid is built from | `HealPixDecomp(..., Jmax=...)` |
| **Multiple co-registered channels at once** (e.g. RGB) | `channels=C` | `from_kernel(..., channels=C)` — see [§A.6](#a-6-multiple-channels-eg-rgb) |
| **Anisotropy** (direction-dependent response) | a `kernel(rho_pix, phi)` that actually depends on `phi`, plus `gauge_type` | `from_kernel(..., gauge_type=...)` — see [§A.4](#a-4-anisotropy-and-gauges) |
| A kernel **fit to data** instead of an analytic formula | `HealPixKernelPyramid.calibrate(...)` instead of `.from_kernel(...)` | see [§D](#d-calibration-how-it-works-and-its-limits) |
| A **wide, slowly-decaying analytic target** (Lorentzian/power-law tail) decomposed into small kernels across all bands, defined once at J=0 | `HealPixKernelPyramid.calibrate_joint(decomp, target_kernel, ...)` | see [§D.1](#d-1-calibrate-joint-decomposing-one-wide-kernel-across-bands) |
| `mode="normalized"` (NaN/weight-aware) vs. `"signed"` (no masking, allows negative kernels) | `mode=` | `HealPixPyramidConv(decomp, kernel_pyramid, mode=...)` |
| **Where** the kernel is defined (once, at the finest band, vs. re-derived per band) | `weights_from_finest_band` (default `True`) | `from_kernel(..., weights_from_finest_band=True)` — see [§A.1bis](#a-1bis-one-kernel-at-j-0-not-one-per-band) |

None of this lives in `HealPixPyramidConv` itself — it only *applies* the
kernel pyramid it is given (`decomp.compute_weighted` → per-band kernel →
`decomp.invert`); it has no parameters of its own beyond `mode`.

In `Notebooks/pyramid_conv_sentinel2_test.ipynb`, all of the above are
collected as plain variables at the top (§1: `KERNEL_SHAPE`, `SIGMA_PIX`,
`COMPACT_KERNEL_SZ`, `JMAX`, `N_CHANNELS`, `GAUGE_TYPE`) precisely so they
don't need to be hunted down inside §3's actual construction code.

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

### A.1bis One kernel, at J=0 — not one per band

`from_kernel` takes a single continuous profile, `kernel(rho_pix, phi_rad)`,
and by default (`weights_from_finest_band=True`) samples it **exactly once**,
on band 0's (the finest band's) own discrete stencil — then reuses that
identical tap vector, unchanged, to build every coarser band's `HealPixConv`.
The pyramid is *constructed from* one kernel given at the finest resolution;
it is not `n_bands` independent recomputations of "the same" kernel on each
band's own resolution.

This matters because real HEALPix pixel geometry is not perfectly
self-similar across `nside` the way an idealized flat/Cartesian grid's
would be (nearest-neighbour angular spacing varies slightly with position
and with resolution). Re-evaluating the profile fresh at each band's own
`nside` — the previous default, still available as
`weights_from_finest_band=False` for comparison — means two bands
nominally carrying "the same" kernel can end up with slightly different
discrete taps for reasons that have nothing to do with the kernel itself.
Sharing one realization removes that spurious source of band-to-band
drift.

Note this is deliberately *not* the literal `B = W K S` composition from
§A.1 pushed down to "compute `K` once, then derive every band from it via
the decomp's own analysis/synthesis chain". That composition, worked
through with `S W = I`, collapses completely:

```text
y = S B W x = S (W K S) W x = (S W) K (S W) x = K x
```

i.e. the multiscale structure cancels out and the pipeline degenerates to
one ordinary, single-scale convolution with `K` at the finest resolution —
useless for the hole-filling use case this pyramid exists for, since only
`K`'s own compact support could then reach across a hole (see §A.2). The
block-diagonal, per-band structure (§A.1) is what keeps the bands' own
information distinct so a coarser band can actually reach further; sharing
its weights across bands (this section) only removes an *incidental*
source of band-to-band inconsistency in how that per-band kernel gets
discretized, without reintroducing the collapse above.

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

The ellipsoid name is resolved case-insensitively: `healpix_geo`'s own
registry is case-sensitive (it accepts `"WGS84"` and `"sphere"` but rejects
`"wgs84"` or `"SPHERE"` outright), and real data sources do not all agree on
a casing — EOPF/GRID4EARTH Sentinel-2 products, for example, declare
`ellipsoid="wgs84"` (lowercase). Every constructor here that takes an
`ellipsoid` argument (`HealPixConv`, `HealPixDown`/`HealPixUp`,
`HealPixDecomp`, `HealPixKernelPyramid.from_kernel`/`calibrate`, and the
`validation` module) canonicalizes it once at construction time
(`healpix_analyse._ellipsoid.canonicalize_ellipsoid`), so `"wgs84"`,
`"WGS84"`, and `"Wgs84"` all resolve to the same geometry — you no longer
need to re-case a store's own reported ellipsoid string by hand.

**This canonicalization is scoped to this package's own constructors.** A
bare `healpix_geo` call written directly in your own code (e.g.
`healpix_geo.nested.healpix_to_lonlat(..., ellipsoid=src.ellipsoid)`), or a
call into a *different* package that does its own ellipsoid resolution
(e.g. `healpix_plot.HealpixGrid(..., ellipsoid=src.ellipsoid)`), does **not**
go through `healpix_analyse`'s canonicalization and will still reject a
lowercase `"wgs84"` from a store. Wrap the value yourself in that case:
`healpix_analyse._ellipsoid.canonicalize_ellipsoid(src.ellipsoid)` — see
`Notebooks/pyramid_conv_sentinel2_test.ipynb` §6 and §8 for two real
examples of exactly this.

### A.6 Multiple channels (e.g. RGB)

`HealPixKernelPyramid.from_kernel`/`calibrate` accept a `channels` argument
(default 1) to filter several co-registered scalar fields — e.g. R, G, B —
through the same fixed kernel pyramid in a single call. This builds one
`in_channels=out_channels=channels` `HealPixConv` per band whose
`[channels, channels, P]` kernel is zero off the diagonal: every channel is
filtered independently with the *identical* profile, with **no**
cross-channel mixing (confirmed in
`tests/test_kernel_pyramid.py::test_channels_are_independent_no_cross_channel_mixing`).
It requires `n_gauges=1` — the general multi-gauge, multi-channel case is
not supported.

This exists to fix a real shape pitfall, not just for convenience. Passing
a `[C, N]` array (e.g. `rgb.T`) straight to a *`channels=1`* kernel pyramid
relies on `HealPixConv`'s ambiguous 2-D input convention, which treats a
`[B, N]` array as `B` independent 1-channel samples — so it silently works
numerically (each of the `C` "batch" rows is filtered independently, which
happens to be exactly what you want for independent channels), but
`HealPixConv` never squeezes a 2-D input's output back to 2-D, so the
result comes back `[C, 1, N]`, not `[C, N]` — breaking a plain `.T` round
trip downstream (see `tests/test_kernel_pyramid.py::test_channels_output_shape_matches_input_no_spurious_axis`
for the regression test, and the `HealPixKernelPyramid.apply` docstring for
the exact mechanism). Building the kernel pyramid with `channels=C` instead
gives you the numerically identical result (see
`tests/test_pyramid_conv.py::test_multichannel_matches_running_channels_separately`)
with the shape you actually asked for. `HealPixPyramidConv.forward` needs
no changes for this — it passes `x` straight through to
`HealPixDecomp.compute_weighted`/`invert` (which already support arbitrary
leading dimensions), and only `HealPixKernelPyramid.apply` needed the fix.

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
  `HealPixConv(in_channels=out_channels=channels, ...)` per band (`channels`
  defaults to 1), with a fixed (`requires_grad=False`) kernel set via
  `HealPixConv.set_kernel`, block-diagonal across channels when
  `channels>1` — see [Section A.6](#a-6-multiple-channels-eg-rgb). Built
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
    channels=1, ellipsoid="sphere", dtype=None, device=None,
)
kp = HealPixKernelPyramid.calibrate(
    decomp, reference_fn, compact_kernel_sz=5, gauge_type="phi",
    n_probes=128, n_excitations=4, seed=0, ridge=1e-6,
    channels=1, ellipsoid="sphere",
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
  kernels requires *joint*, cross-band least-squares optimization — see
  `calibrate_joint`, next.

### D.1 `calibrate_joint`: decomposing one wide kernel across bands

`HealPixKernelPyramid.calibrate_joint(decomp, target_kernel, ...)` fits
*every* band's `P` taps **together**, in one least-squares solve, against a
single wide target kernel defined once at the finest resolution
(`decomp.levels[0]`) — not `calibrate`'s independent per-band fits. The
design matrix is built by probing the real, composed pipeline: for each
band `j` and tap `p`, a one-hot kernel is placed on that tap alone (every
other band's contribution held at zero), the map is reconstructed through
the *actual* `HealPixDecomp.invert`, and the response at a set of random
probe pixels becomes that column of the design matrix — so the fit
"knows" about the true inter-band coupling the block-diagonal `apply()`
itself will later exploit (each band's kernel only has to supply what the
pyramid's own geometric coarsening does not already contribute).

Measured (`test_calibrate_joint_decomposes_a_wide_kernel_across_bands`,
level=5, `Jmax=3`, `compact_kernel_sz=5` throughout, target =
`kernel_lorentzian(scale_pix=6)`, brute-force reference `kernel_sz=25`):

| Method | Same target | rel RMS (smooth field) |
|---|---|---|
| `calibrate` (single band, band 0 alone) | Lorentzian σ=6px | **≈87%** |
| `calibrate_joint` (4 bands, jointly) | Lorentzian σ=6px | **≈3.2%** |

Same wide target, same per-band kernel size — the only difference is
fitting all bands together against the real reconstructed output instead
of each band alone against the target evaluated at its own resolution.
White-noise excitation (harder, see the smooth-vs-noise discussion in
Section E) measured ≈10% rel RMS for the same fit.

**Requires a full-sphere `HealPixDecomp`** (`cell_ids=None`) for the fit
itself (an unambiguous "external" pixel order is needed to compare against
the direct reference on the same footprint); the resulting
`HealPixKernelPyramid` can still be used afterwards on a partial-domain
decomposition, same as `from_kernel`'s. Reach is still bounded by
`decomp.Jmax` — a target wider than the coarsest band's own reach cannot
be represented regardless of how well the taps are fit; see
[Relation to "Convolution Pyramids" and honest scope](#relation-to-convolution-pyramids-and-honest-scope)
for how this relates to (and differs from) the cited paper's own
construction.

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
2. **`calibrate` is single-band, not joint** — it cannot make a compact
   per-band kernel reproduce an arbitrarily wide target on its own. Use
   `calibrate_joint` ([Section D.1](#d-1-calibrate-joint-decomposing-one-wide-kernel-across-bands))
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
   still a silent mislabeling) — see [Section A.5](#a-5-true-sphere-vs-ellipsoid).
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
   supported — see [Section A.6](#a-6-multiple-channels-eg-rgb). Every
   channel also shares the *same* kernel profile (block-diagonal, identical
   diagonal blocks); per-channel-distinct profiles are not supported.

---

## Relation to "Convolution Pyramids" and honest scope

This module is **inspired by**, but does not claim to reproduce exactly, Z.
Farbman, R. Fattal, D. Lischinski, *"Convolution Pyramids"*, ACM
Transactions on Graphics 30(4), 2011. The cited paper's core technique —
jointly optimizing every pyramid level's kernel so that the *cascade*
reproduces one global target operator — is what `calibrate_joint`
([Section D.1](#d-1-calibrate-joint-decomposing-one-wide-kernel-across-bands))
now does too, but by a different, more general and less precise route:
direct numerical least-squares against a brute-force reference, evaluated
by probing the real `HealPixDecomp`/`HealPixConv` machinery, rather than
the paper's closed-form, boundary-matched derivations for specific kernel
families (large-radius blurs, gradient-domain/Poisson-type operators).
`calibrate_joint` can in principle fit *any* profile the direct oracle can
evaluate, at the cost of no closed-form accuracy guarantee — every claim
about it in this document is an empirical measurement on a specific
target/level/`Jmax`, not a general bound, and should be re-measured for a
materially different kernel family or pyramid depth. What this module
provides in total:

- an exact-reconstruction pyramid (`HealPixDecomp`, pre-existing) reused
  as-is;
- a **block-diagonal**, per-band-only kernel pyramid, built analytically
  from a chosen kernel family (`from_kernel`), by **independent per-band**
  least-squares calibration (`calibrate`), or by **joint, cross-band**
  least-squares calibration against one wide target (`calibrate_joint`);
- correct NaN/weight propagation through that block-diagonal pyramid with
  the standard single-division-after-synthesis normalized-convolution
  convention.

No numerical claim in this page compares against the cited paper's own
published results, and no claim of exact reproduction of its method is
made anywhere in this codebase.
