# Pyramidal convolution — compact multiscale spherical filtering

`healpix_analyse.kernel_pyramid` and `healpix_analyse.pyramid_conv` combine
{doc}`/other/decomp` (the exact-reconstruction Laplacian pyramid) with
{doc}`/conv/convol` (`HealPixConv`'s gauge-equivariant compact stencil) to
build masked, NaN-aware, multiscale spherical filters that stay `O(N)` in
the number of pixels.

```{include} ../_generated/glance/kernel_pyramid.md
```

```{include} ../_generated/glance/pyramid_conv.md
```

This page is the guide. The mathematics, implementation notes and calibration
are in {doc}`/conv/pyramid_conv_maths`; the validation report is in the
developer notes ({doc}`/dev/pyramid_conv_validation`). It does not claim to
reproduce the specific construction of Farbman, Fattal & Lischinski,
*"Convolution Pyramids"*, ACM TOG 30(4), 2011 — see
[Relation to "Convolution Pyramids" and honest scope](#relation-to-convolution-pyramids-and-honest-scope)
below.

---

> **Looking for “I have one wide kernel and I want to convolve with it”?**
> That is [Wide-kernel convolution](wide_conv.md) and
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
[examples/pyramid_conv_quickstart.py](https://github.com/GRID4EARTH/healpix-analyse/blob/main/examples/pyramid_conv_quickstart.py)
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
| **Multiple co-registered channels at once** (e.g. RGB) | `channels=C` | `from_kernel(..., channels=C)` — see [§A.6](pyramid_conv_maths.md#pyr-a6) |
| **Anisotropy** (direction-dependent response) | a `kernel(rho_pix, phi)` that actually depends on `phi`, plus `gauge_type` | `from_kernel(..., gauge_type=...)` — see [§A.4](pyramid_conv_maths.md#pyr-a4) |
| A kernel **fit to data** instead of an analytic formula | `HealPixKernelPyramid.calibrate(...)` instead of `.from_kernel(...)` | see [§D](pyramid_conv_maths.md#d-calibration-how-it-works-and-its-limits) |
| A **wide, slowly-decaying analytic target** (Lorentzian/power-law tail) decomposed into small kernels across all bands, defined once at J=0 | `HealPixKernelPyramid.calibrate_joint(decomp, target_kernel, ...)` | see [§D.1](pyramid_conv_maths.md#pyr-d1) |
| `mode="normalized"` (NaN/weight-aware) vs. `"signed"` (no masking, allows negative kernels) | `mode=` | `HealPixPyramidConv(decomp, kernel_pyramid, mode=...)` |
| **Where** the kernel is defined (once, at the finest band, vs. re-derived per band) | `weights_from_finest_band` (default `True`) | `from_kernel(..., weights_from_finest_band=True)` — see [§A.1bis](pyramid_conv_maths.md#pyr-a1bis) |

None of this lives in `HealPixPyramidConv` itself — it only *applies* the
kernel pyramid it is given (`decomp.compute_weighted` → per-band kernel →
`decomp.invert`); it has no parameters of its own beyond `mode`.

In `Notebooks/pyramid_conv_sentinel2_test.ipynb`, all of the above are
collected as plain variables at the top (§1: `KERNEL_SHAPE`, `SIGMA_PIX`,
`COMPACT_KERNEL_SZ`, `JMAX`, `N_CHANNELS`, `GAUGE_TYPE`) precisely so they
don't need to be hunted down inside §3's actual construction code.

---

## Calls at a glance

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

## Relation to "Convolution Pyramids" and honest scope

This module is **inspired by**, but does not claim to reproduce exactly, Z.
Farbman, R. Fattal, D. Lischinski, *"Convolution Pyramids"*, ACM
Transactions on Graphics 30(4), 2011. The cited paper's core technique —
jointly optimizing every pyramid level's kernel so that the *cascade*
reproduces one global target operator — is what `calibrate_joint`
([Section D.1](pyramid_conv_maths.md#pyr-d1))
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
