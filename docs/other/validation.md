# Validation tools

`healpix_analyse.validation` holds a direct, non-pyramidal spherical
convolution used as an independent reference ("oracle") to test and calibrate
the pyramidal convolutions ({doc}`/conv/pyramid_conv`). It is not exported
from `healpix_analyse`.

```{include} ../_generated/glance/validation.md
```

```python
from healpix_analyse.validation import (
    direct_spherical_convolution,
    direct_reference_operator_factory,
    smooth_test_field,
)

y_ref = direct_spherical_convolution(
    x, cell_ids, level, kernel_iso,        # kernel_iso(rho_pix, phi) -> weight
    weights=None, kernel_sz=5, ellipsoid="sphere",
    eps=1e-8, restore_mask=True, normalize=True,
)
```

- It locates the true cell centres with healpix-geo, finds the neighbours
  with a SciPy KD-tree on unit vectors, and evaluates the kernel at the exact
  great-circle distance, in the same cell units (`rho_pix`) as the
  `kernel_*` profiles, so the same callables can be passed to both.
- It is independent of `HealPixConv` on purpose: it checks the pyramidal path
  rather than a copy of its own machinery.
- Only **isotropic** kernels (`phi` is passed as 0).
- `normalize=True` (default) matches the masked, weighted
  `HealPixPyramidConv`; `normalize=False` matches the raw tap-weighted sum of
  `HealPixConv`, for comparing one band.
- A plain Python loop over the cells: meant for the modest map sizes of tests
  and calibration, not for production.
- Needs SciPy (`pip install "healpix-analyse[scipy]"`); it is also what makes
  `HealPixKernelPyramid.calibrate_joint` need SciPy.

`direct_reference_operator_factory(kernel, ...)` wraps it as a reference
operator for the calibration, and `smooth_test_field(cell_ids, level)` gives
a smooth field for tests.
