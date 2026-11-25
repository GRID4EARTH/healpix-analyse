# Pyramidal convolution: mathematics, implementation and calibration

The guide is {doc}`/conv/pyramid_conv`; the validation report is
{doc}`/dev/pyramid_conv_validation`.

## A. Mathematical foundations

(pyr-a1)=
### A.1 The target operator, and what a pyramid can and cannot give you for free

Let `W` be the analysis operator of a `HealPixDecomp` (its stacked
`Down`/`Up` chain) and `S` its synthesis operator, so `S W = I` exactly —
this is the exact-reconstruction identity `decomp.invert(decomp.compute(x))
== x` already documented in {doc}`/other/decomp`. Let `K` be a target convolution
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
assumed — see [Section E](../dev/pyramid_conv_validation.md#e-validation-results-and-honest-limits).

(pyr-a1bis)=
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
division — see [Section C](pyramid_conv.md#calls-at-a-glance)); in `mode="normalized"`, a
signed kernel is applied identically to the data and weight channels, but
note that a *negative* synthesized weight `S(B_K m̃)` at some pixel makes the
division well-defined but not a meaningful "confidence" any more — this is
an open point, see [Section E](../dev/pyramid_conv_validation.md#e-validation-results-and-honest-limits).

(pyr-a4)=
### A.4 Anisotropy and gauges

A kernel profile is a function `K(rho_pix, phi_rad)` evaluated once, on the
fixed North-Pole stencil, at each band's own resolution (see
`_stencil_pixel_polar`, which reuses `HealPixConv`'s own
`_local_kernel_grid` so the evaluation points are guaranteed identical to
the runtime stencil). `HealPixConv`'s per-pixel gauge rotation
(`gauge_type`, `n_gauges`, `singularity_lonlat`/`ref_direction` — see
{doc}`/conv/convol`) then carries that fixed-frame anisotropy into every
output pixel's own local tangent frame consistently, the same way
`HealPixConv` does for a learned kernel. `HealPixKernelPyramid.from_kernel`
forwards `gauge_type`/`n_gauges`/`singularity_lonlat`/`ref_direction`
unchanged to every band's `HealPixConv`.

(pyr-a5)=
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

(pyr-a6)=
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
  `channels>1` — see [Section A.6](#pyr-a6). Built
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
  scaling is reported in [Section E](../dev/pyramid_conv_validation.md#e-validation-results-and-honest-limits).
- **PyTorch conventions**: shapes follow `HealPixDecomp`/`HealPixConv`
  conventions (`[N]`, `[B, N]`, `[B, C, N]`); `float64` is used in the test
  suite for identity/gradient checks, `float32` for the benchmark. The
  `isfinite`-based missing-value mask itself carries no gradient (as with
  any boolean masking decision), but surviving finite values of `data` and
  `weights` stay fully differentiable — confirmed in
  `tests/test_decomp_weighted.py::test_gradient_flows_through_weighted_pipeline`
  and `tests/test_pyramid_conv.py::test_gradient_flows_through_pyramid_conv`.

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

(pyr-d1)=
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
[Relation to "Convolution Pyramids" and honest scope](pyramid_conv.md#relation-to-convolution-pyramids-and-honest-scope)
for how this relates to (and differs from) the cited paper's own
construction.
