# Wide-kernel convolution — one kernel image, one call

`HealPixWideConv` convolves a HEALPix field by a kernel that is **too wide for
a compact stencil**: an exponential with a 500 m scale, a Lorentzian, a
top-hat a kilometre across. You hand it the kernel the way you naturally hold
it — an image, a formula in metres, a raster — and it works out by itself the
small per-band kernels that reproduce it through a scale pyramid.

```python
import numpy as np
from healpix_analyse import HealPixWideConv

conv = HealPixWideConv.from_radial(
    lambda r: np.exp(-r / 500.0),        # r in metres
    level=17, n=64, lon=2.3198, lat=48.8704,
    Jmax=6, compact_kernel_sz=5,
)
y = conv(x, cell_ids)                    # x: [N] or [..., N] -> same shape
```

That is the whole API. `x` may be numpy or torch, on CPU or GPU; the output
comes back the same type, on the same device, with the same shape.

This document covers: why the pyramid is needed at all (§A), the four ways of
specifying the kernel and when each matters (§B), what each tuning knob
actually buys, measured (§C), the API (§D), and the limits (§E).

Related: [`docs/pyramid_convolution.md`](pyramid_convolution.md) documents the
lower-level `HealPixKernelPyramid` / `HealPixPyramidConv` pair (analytic
per-band profiles, multi-channel, NaN/weight-aware filtering).
`HealPixWideConv` is the single-call front end for the "one wide kernel"
problem and is deliberately narrower in scope — see §F.

---

## Which constructor?

| you have | use | see |
|---|---|---|
| the kernel already sampled on the HEALPix lattice, as a `(2n+1, 2n+1)` image | `HealPixWideConv(kernel_image, level, ...)` | [§B.1](#b1-mode-1--an-image-on-the-healpix-lattice) |
| a formula in metres, possibly anisotropic | `HealPixWideConv.from_function(fn, level, n, lon, lat, ...)` | [§B.2](#b2-mode-2--a-function-of-x-y-in-metres) |
| a raster with a fixed pixel size in metres | `HealPixWideConv.from_grid(grid, pixel_size_m, level, n, lon, lat, ...)` | [§B.3](#b3-mode-3--a-raster-in-metres-bilinearly-resampled) |
| a formula in `r` only (isotropic — the common case) | `HealPixWideConv.from_radial(fn, level, n, lon, lat, ...)` | [§B.4](#b4-mode-4--a-function-of-r-in-metres) |

All four produce the same kind of object and are equally accurate; the choice
is about which description is natural for your problem. §B.6 measures that
claim.

---

## A. What problem this solves

### A.1 Why a compact stencil cannot do it

`HealPixConv` convolves with a `kernel_sz x kernel_sz` neighbourhood, so its
reach is `kernel_sz // 2` pixels — two, for the usual `5x5`. A kernel whose
scale is tens of pixels simply does not fit: the stencil would have to be as
wide as the kernel, and its cost grows as the square of that width.

Measured, on a 256×256 HEALPix patch at level 17 (pixel 49.7 m), against an
exponential kernel `exp(-r/R0)` — a single `5x5` band (`Jmax=0`), fitted as
well as least squares allows:

| kernel scale `R0` | single `5x5` band, rel RMS |
|---|---|
| 5 px (249 m) | 0.79 |
| 10 px (497 m) | 0.93 |
| 20 px (995 m) | 0.98 |

It gets *worse* as the kernel widens, and it cannot do otherwise: there is no
`5x5` stencil that looks like a 20-pixel-wide kernel.

### A.2 The pyramid idea

This is the "convolution pyramids" construction (Farbman, Fattal &
Lischinski, ACM TOG 2011): decompose the signal into a Laplacian pyramid,
convolve **each band with its own small kernel**, and synthesize. A wide
kernel is wide only in the finest band's units — at band `j` the pixels are
`2**j` times larger, so the same physical width is `2**j` times fewer pixels.
After a few stages, a very wide kernel is a handful of pixels across, and a
`5x5` stencil is enough *at that band*.

`HealPixDecomp` is a Laplacian pyramid, so synthesis is the exact algebraic
inverse of analysis (`S W = I`, verified to ~1e-16 — see
[`docs/decomp.md`](decomp.md)). That matters here: if every band's kernel were
exactly right, the result would be the target kernel **exactly**. The per-band
fit residuals are therefore the method's entire error budget, and
`fit_residuals()` reports them.

Same patch, same kernels, with the pyramid (`Jmax=6`, `5x5` per band):

| kernel scale `R0` | single `5x5` band | pyramid | ratio |
|---|---|---|---|
| 5 px (249 m) | 0.79 | 0.132 | 6.0× |
| 10 px (497 m) | 0.93 | 0.123 | 7.6× |
| 20 px (995 m) | 0.98 | 0.124 | 7.9× |

The pyramid's error is **flat in the kernel's width** while the single band's
degrades. That is the entire point of the construction.

### A.3 Why the per-band kernels are fitted, not sampled

The obvious recipe — decompose the kernel image `K` into the same pyramid and
use `band_j(K)`'s own values around the centre as band `j`'s taps — does not
work, and it is worth saying why, because it looks right.

The input to band `j` is not a Dirac: it is `band_j(δ)`, a small Laplacian
bump with a negative surround. Convolving that by `band_j(K)` does not give
`band_j(K)` back. Measured on the level-17 patch, that recipe lands at **~99%
relative error** — no better than doing nothing.

What the class does instead: for each band, solve for the
`compact_kernel_sz**2` taps `w_j` that best satisfy

```
band_j(δ)  ∗  w_j   ≈   band_j(K)
```

in the least-squares sense, with the design matrix built by probing the real
`HealPixConv` with one-hot kernels — so the stencil geometry used to *fit* is
exactly the one used to *apply*. Because `S W = I`, if each band's fit were
exact the synthesized result would be `K` exactly; the residuals are the gap.

Typical residuals, fine band first (level 17, 256×256, `R0=10 px`, `5x5`):

```
0.82  0.59  0.34  0.17  0.10  0.16  0.01
```

The finest bands are the worst-fitted — they carry the kernel's sharpest
structure — but they also carry the least of a wide kernel's energy, which is
why the overall error lands near 0.12 rather than near 0.8.

The fit needs a domain, so it happens on the **first call**, on that call's
own `cell_ids`, and is cached per domain afterwards. Fitting on the data's own
patch also keeps the geometry honest: HEALPix pixel shape depends on latitude
(§B.5), so a fit made somewhere else would be the wrong fit.

Cost, level 17, 65 536 cells, `5x5`, `Jmax=6`: a few seconds for the first
call (the fit), then **~0.1 s** per call on the same domain.

---

## B. Specifying the kernel — the four modes

### B.1 Mode 1 — an image on the HEALPix lattice

```python
conv = HealPixWideConv(kernel_image, level, Jmax=6, compact_kernel_sz=5)
```

`kernel_image` is a square, **odd-sided** `(2n+1, 2n+1)` array laid on the
base face's integer `(i, j)` lattice: `kernel_image[a, b]` is the weight at
offset `(di, dj) = (b - n, a - n)` from the centre — rows are `j`, columns are
`i`, the same convention as unfolding a NESTED tile into a square image.

The side must be odd because there has to be a centre pixel; a 256×256 domain
therefore takes a 255×255 kernel at most.

Use this when you already hold the kernel on the lattice — for example when it
came out of another HEALPix computation. If your kernel is defined in metres,
prefer modes 2–4: on the lattice, "one pixel right" is not the same distance
everywhere (§B.5).

### B.2 Mode 2 — a function of `x, y` in metres

```python
conv = HealPixWideConv.from_function(
    lambda x_m, y_m: np.exp(-np.hypot(x_m / 1500.0, y_m / 400.0)),
    level=17, n=64, lon=2.3198, lat=48.8704, Jmax=6,
)
```

`fn(x_m, y_m)` is evaluated at each lattice cell's true **east/north** offset
in metres from the centre. This is what makes an anisotropic kernel
meaningful: "1500 m east-west by 400 m north-south" is a statement about the
ground, not about pixel indices.

### B.3 Mode 3 — a raster in metres, bilinearly resampled

```python
conv = HealPixWideConv.from_grid(
    raster, pixel_size_m=25.0,           # or (sx, sy)
    level=17, n=32, lon=2.3198, lat=48.8704, Jmax=6,
)
```

`raster` is a regular grid whose pixels are `pixel_size_m` apart, centred on
its own middle pixel — the shape you would read from a file or draw on graph
paper. It is **bilinearly interpolated** at the lattice cells' true metric
positions. That resampling is what accounts for the HEALPix deformation:
dropping the raster straight onto `(i, j)` indices would shear it.

Measured: a 1200 × 1200 m square top-hat given on a 25 m raster comes back, on
the ground, **1194 m east by 1187 m north** at half maximum — square, as it
must be, which means sheared in the `(i, j)` image.

The resampling itself is accurate: against the same kernel evaluated exactly,
the bilinear reconstruction differs by `1.8e-4` relative (level 17, 20 m
raster) — far below the method's own error.

Cells falling outside the raster are set to `fill` (default 0) and a
`RuntimeWarning` names how many. Make the raster cover more ground than the
`(2n+1)` lattice, whose corners reach `n·√2` pixels out.

### B.4 Mode 4 — a function of `r` in metres

```python
conv = HealPixWideConv.from_radial(
    lambda r_m: np.exp(-r_m / 500.0),
    level=17, n=64, lon=2.3198, lat=48.8704, Jmax=6,
)
```

The common case, and the shortest path when the kernel is isotropic. `r` is
the exact great-circle distance in metres from the centre cell, so the kernel
is isotropic **on the ground**. Verified directly: `max |kernel_image −
exp(-r/R0)|` is exactly `0` — the weight depends on `r` and on nothing else.

"Isotropic on the ground" is not the same as isotropic in `(i, j)`. In the
polar caps, cells at the same distance in metres sit at quite different index
offsets, and only this construction gives them the same weight.

### B.5 The HEALPix lattice is not a square metric grid

This is the fact that makes modes 2–4 necessary, and it is worth knowing
independently of this module. The primitive is public:

```python
x_m, y_m = HealPixWideConv.lattice_offsets_m(level, lon, lat, n)
```

It returns the east/north position in metres of every cell of the
`(2n+1, 2n+1)` lattice around a point, in an azimuthal-equidistant projection,
so `hypot(x_m, y_m)` is the exact great-circle distance.

Measured at level 17:

| location | one `i`-step | one `j`-step | ratio |
|---|---|---|---|
| equatorial belt (0°, −20°) | 50 m | 50 m | **1.00** |
| polar cap (Paris, 48.87°) | 42 m | 71 m | **1.67** |

In the equatorial belt (`|lat| < 41.8 deg`) the lattice *is* a square metric
grid — rotated 45°, but square — so mode 1 is already correct there. In the
polar caps the two axes are neither equal nor orthogonal, and the cost of
ignoring it is large. Taking the same `exp(-r/497 m)` kernel and indexing it
by `(i, j)` instead of by true distance:

| location | `(i, j)`-indexed vs metric kernel |
|---|---|
| equatorial belt | 1.7% |
| polar cap (Paris) | **43.9%** |

A 44% error in the kernel, silently, with no exception and no warning — the
convolution would run perfectly and filter with the wrong thing.

### B.6 The recipe is free; the shape is not

The four modes change **what the kernel is**, not how well the pyramid applies
it. Running the *same* `exp(-r/497 m)` kernel through all four, on the same
level-17 patch (`Jmax=6`, `5x5`):

| mode | kernel image vs mode 1 | rel RMS |
|---|---|---|
| 1 — image | 0 | 0.1239 |
| 2 — `fn(x, y)` | 0 | 0.1239 |
| 3 — raster, bilinear | 1.8e-4 | 0.1239 |
| 4 — `fn(r)` | 0 | 0.1239 |

Identical, to the bilinear resampling's own accuracy.

What *does* move the error is the kernel's shape:

| kernel shape | rel RMS |
|---|---|
| smooth isotropic exponential | 0.123 |
| square top-hat, sharp edges | 0.238 |
| strongly anisotropic, 1500 × 400 m | 0.319 |

Sharp edges and strong anisotropy are what a `5x5` per-band stencil struggles
with — visible directly in the fit residuals (~0.8 at the fine bands for the
exponential, ~1.0 for the top-hat's edge). Those are the cases where raising
`compact_kernel_sz` earns its cost.

---

## C. Tuning — what each knob buys, measured

All measurements below: level 17, 256×256 patch centred on (2.3198, 48.8704),
`exp(-r/R0)` kernel, error = relative RMS between `conv(δ)` and the kernel
laid on the same domain. "peak" is the reconstructed peak relative to the
kernel's own.

### C.1 `compact_kernel_sz` — the accuracy knob

| `compact_kernel_sz` | taps/band | rel RMS | peak | fit time |
|---|---|---|---|---|
| 3 | 9 | 0.196 | 0.77 | 1.3 s |
| 5 | 25 | 0.128 | 0.82 | 3.0 s |
| 7 | 49 | 0.094 | 0.85 | 6.1 s |
| 9 | 81 | 0.074 | 0.87 | 12.9 s |
| 11 | 121 | 0.058 | 0.89 | 19.4 s |

*(128×128 patch, `Jmax=5`, so the fit times stay comparable; the error trend
matches the 256×256 case, where `3/5/7` give 0.195 / 0.124 / 0.089.)*

Monotonic, with no plateau in this range — this is the knob to reach for when
accuracy matters. The cost grows roughly as `compact_kernel_sz**2`, since the
fit probes one one-hot kernel per tap per band. At 65 536 cells,
`compact_kernel_sz=9` was heavy enough to be worth splitting or running on a
smaller patch.

### C.2 `Jmax` — how deep the pyramid must go

| `Jmax` | bands | rel RMS | peak |
|---|---|---|---|
| 0 | 1 | 0.927 | 1.11 |
| 1 | 2 | 0.702 | 0.90 |
| 2 | 3 | 0.368 | 0.86 |
| 4 | 5 | **0.120** | 0.82 |
| 6 | 7 | 0.124 | 0.82 |
| 8 | 9 | 0.124 | 0.82 |

*(`R0 = 10 px`, `5x5`.)*

It **saturates**. Below the saturation point every extra stage helps a lot;
above it, extra bands cost a little time and change nothing. Here saturation
is at `Jmax=4`, where the stencils' combined reach
(`compact_kernel_sz//2 · 2**Jmax` = 32 fine pixels) is comparable to the
kernel's 1% radius (46 pixels).

There is no formula worth trusting here: raise `Jmax` until the error stops
moving, then stop. A too-small `Jmax` is the one setting that fails loudly
(0.93 at `Jmax=0`); a too-large one is merely slightly wasteful.

### C.3 Kernel width — the method is width-independent

| `R0` | rel RMS (pyramid) | rel RMS (`Jmax=0`) |
|---|---|---|
| 2.5 px (124 m) | 0.179 | — |
| 5 px (249 m) | 0.132 | 0.79 |
| 10 px (497 m) | 0.123 | 0.93 |
| 20 px (995 m) | 0.124 | 0.98 |
| 40 px (1990 m) | 0.125 | — |

Flat from ~5 pixels up, while a single compact band degrades steadily. Note
the other end: a **narrow** kernel (2.5 px) is slightly *worse*, because it
lives almost entirely in the finest band, where the `5x5` fit is weakest. For
kernels of a few pixels, a plain `HealPixConv` is the better tool — the
pyramid is for wide kernels.

### C.4 `n` — where you truncate the kernel image

| `n` | image | value at the image edge | rel RMS |
|---|---|---|---|
| 16 | 33×33 | 2.0e-1 | 0.256 |
| 32 | 65×65 | 4.1e-2 | 0.160 |
| 64 | 129×129 | 1.7e-3 | 0.124 |
| 127 | 255×255 | 3.1e-6 | 0.123 |

*(`R0 = 10 px`, `5x5`, `Jmax=6`.)*

A truncated kernel has a cliff at its edge, and the pyramid then has to
reproduce that cliff. Rule of thumb: choose `n` so the kernel has fallen to
**≲1e-3** of its peak — past that there is nothing left to gain (1.7e-3 and
3.1e-6 give the same answer), and each extra ring costs memory in the kernel
image only, not in the convolution.

---

## D. API reference

```python
class HealPixWideConv:

    def __init__(self, kernel_image, level, *,
                 Jmax=6, compact_kernel_sz=5, gauge_type="phi",
                 ellipsoid="sphere", ridge=1e-12,
                 dtype=torch.float64, device=None)
        """kernel_image: (2n+1, 2n+1) on the (i, j) lattice at `level`."""

    # --- alternative constructors (§B) ---
    @classmethod
    def from_function(cls, fn, level, n, lon, lat, *, r_earth=6371008.8, **kw)
        """fn(x_m, y_m) -> weight, at true east/north offsets in metres."""

    @classmethod
    def from_radial(cls, fn, level, n, lon, lat, *, r_earth=6371008.8, **kw)
        """fn(r_m) -> weight, at the true great-circle distance in metres."""

    @classmethod
    def from_grid(cls, grid, pixel_size_m, level, n, lon, lat, *,
                  fill=0.0, r_earth=6371008.8, **kw)
        """A metric raster, bilinearly resampled onto the lattice."""

    @staticmethod
    def lattice_offsets_m(level, lon, lat, n, *, r_earth=6371008.8)
        """-> (x_m, y_m), each (2n+1, 2n+1): east/north offsets in metres."""

    # --- use ---
    def __call__(self, x, cell_ids)
        """x: [N] or [..., N] -> same shape, same type, same device."""

    # --- inspection ---
    def fit_residuals(self, cell_ids=None) -> tuple   # per band, fine first
    def band_kernels(self, cell_ids=None) -> tuple    # each (KSZ, KSZ)
    def kernel_as_field(self, cell_ids, centre_cell=None) -> np.ndarray
    def reference_centre(self, cell_ids) -> int
    decomp                                            # the pyramid, last domain
```

Notes on the parameters that are easy to get wrong:

- **`level`** — `cell_ids` passed to `__call__` must be at this level.
- **`n`** — half-size of the kernel image; see §C.4.
- **`ridge`** — Tikhonov term on the per-band least squares. The default
  (`1e-12`) is essentially "none"; raise it only if a band's fit is visibly
  unstable.
- **`dtype`** — `float64` is the default and is recommended. The coarse bands'
  taps are large (the Dirac's coarse-band content is tiny, so the taps that
  amplify it are not), and the fit is a normal-equations solve.
- **`gauge_type`, `ellipsoid`** — passed through to each band's `HealPixConv`;
  see [`docs/convol_doc.md`](convol_doc.md) and
  [`docs/pyramid_convolution.md`](pyramid_convolution.md) §A.5.

### Validation helpers

`kernel_as_field` and `reference_centre` exist for one specific reason: to get
the centring right when checking the operator. The operator is
translation-invariant — the fitted kernel applies at every pixel — but the
*fit* is centred on one cell, and comparing `conv(δ)` against a kernel laid on
a **different** cell inflates the error badly. Measured: the same setup scored
0.124 correctly centred and 0.176 with a one-pixel offset. So:

```python
c = conv.reference_centre(cell_ids)
x = np.zeros(cell_ids.size); x[np.searchsorted(cell_ids, c)] = 1.0
y = conv(x, cell_ids)
K = conv.kernel_as_field(cell_ids, centre_cell=c)     # the right reference
```

---

## E. Validation and honest limits

### E.1 What is verified

`tests/test_wide_conv.py` (24 tests) pins down: convolving a Dirac returns the
kernel; the pyramid beats a single compact band by more than 2×; shape and
device round-tripping for `[N]`, `[B, N]`, `[..., N]`, numpy and torch;
linearity; per-domain caching; the metric constructors against exact
evaluation; that the lattice is square in the belt and sheared in the cap; and
the two clipping warnings.

The step-by-step validation, with plots, is in
`Notebooks/pyramid_conv_single_test.ipynb`; the four kernel modes are
demonstrated in `Notebooks/wide_conv_kernel_recipes.ipynb`. Both are fully
synthetic and run without network access.

### E.2 The error is concentrated on sharp features

At the reference setting the profiles overlay everywhere except within a few
pixels of `r = 0`, where the result reaches **~0.82** instead of 1.0.
`exp(-r)` is not differentiable at the origin, which is the hardest possible
feature for a compact stencil; a smooth kernel (a Gaussian) has no such
corner. Total mass is better preserved than the peak: 3.3% low at the same
setting (peak 18.3% low, mass 3.3% low, same run).

If the peak matters more than the bulk, raise `compact_kernel_sz` — it moves
the peak from 0.77 (`3x3`) to 0.89 (`11x11`), see §C.1.

### E.3 The far tail does not follow

On a log radial profile, the result tracks the kernel down to about **1e-3 of
the peak** and then flattens onto a floor around `1e-4`, changing sign further
out. Below ~1e-3 relative, what you are looking at is the pyramid's own
reconstruction residual, not the kernel.

For a kernel used as a smoother this is irrelevant. If your application needs
several decades of dynamic range in the kernel's tail, this method is not the
right one.

### E.4 Domain and face edges

Two clipping situations, both warned about rather than silently accepted:

- **Domain edge.** Kernel pixels falling outside `cell_ids` are dropped
  (`"... fall outside the domain"`). The fit then sees a clipped kernel and is
  biased. Use a domain comfortably larger than the kernel.
- **Base-face edge.** A kernel pixel stepping off the base face would land on
  a neighbouring face, where `(i, j)` no longer means the same direction, so
  those are dropped (`"... fall off base face"`). `lattice_offsets_m` raises
  outright in that situation rather than returning meaningless coordinates.

### E.5 Scope

Deliberately **single-channel** and **not mask-aware**:

- one kernel, applied independently to every leading dimension of `x`; there
  is no channel mixing;
- NaN in `x` propagates. For hole-filling on real, gappy data use
  `HealPixPyramidConv` in `mode="normalized"` — see
  [`docs/pyramid_convolution.md`](pyramid_convolution.md) §A.2.

### E.6 The fit is per domain

The per-band kernels are fitted on the first call and cached against the
`cell_ids` array's contents. A different domain triggers a new fit — which is
correct (pixel geometry is latitude-dependent) but worth knowing if you loop
over many small patches: batch them, or reuse one domain.

---

## F. Relation to the other convolution modules

| module | kernel | reach | masks | channels |
|---|---|---|---|---|
| `HealPixConv` | one compact stencil | `kernel_sz // 2` px | no | yes, dense |
| `HealPixKernelPyramid` + `HealPixPyramidConv` | an analytic profile per band | multiscale | yes (`"normalized"`) | yes, block-diagonal |
| **`HealPixWideConv`** | **one wide kernel, given as an image** | **multiscale** | **no** | **no** |
| `LargeConv` / `HealPixFFTConv` | arbitrary, via FFT on a local grid | grid-limited | see their docs | see their docs |

`HealPixKernelPyramid.calibrate_joint` solves a related problem — fitting all
bands jointly against one wide target operator, probed through the full
pipeline. `HealPixWideConv` fits each band separately against that band's own
share of the kernel, which is cheaper, better conditioned, and exact in the
limit (because `S W = I`). Use `calibrate_joint` when the target is an
*operator* you can only probe; use `HealPixWideConv` when the target is a
*kernel* you can write down.

---

## G. Reproducing the numbers in this document

Every figure quoted here was measured on this package, at level 17, on a
256×256 HEALPix patch centred on (lon 2.3198, lat 48.8704) unless stated
otherwise. The two notebooks reproduce the headline results end to end:

```bash
jupyter nbconvert --to notebook --execute Notebooks/pyramid_conv_single_test.ipynb
jupyter nbconvert --to notebook --execute Notebooks/wide_conv_kernel_recipes.ipynb
pytest tests/test_wide_conv.py -q -s        # prints its own measured values
```

The sweeps of §C are a few lines on top of the same setup: build one
`HealPixWideConv` per parameter value, put a Dirac on
`conv.reference_centre(cell_ids)`, and compare against
`conv.kernel_as_field(cell_ids, centre_cell=...)`.
