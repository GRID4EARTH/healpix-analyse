# Divergence and curl: which tool?

Three tools split a horizontal flow `(u east, v north)` into a divergent and a
rotational part.

| | `HEALPixSHT.uv_to_curl_div` ({doc}`/sht/healpix_sht`) | `HealPixDivCurl` ({doc}`divcurl`) | `HealPixMultiScaleDivCurl` ({doc}`divcurl`) |
|---|---|---|---|
| Method | global spin-1 spherical harmonics | derivative-of-Gaussian kernels on the cells (a fixed `HealPixConv` stencil) | the same kernels on every band of a `HealPixDecomp` pyramid |
| Domain | **whole sphere** only | whole sphere or a regional set of cell ids | the domain of the decomposition |
| Ordering | RING by default (`nest=True` for NESTED) | NESTED | NESTED |
| Input | `u` and `v`, each `(..., N)` | `[..., 2, N]` (channel 0 = u, 1 = v) | a `HealPixPyramid` of such bands |
| Output | `(div, curl)` Torch tensors | `[..., 2, N]` (0 = div, 1 = curl); NumPy in → NumPy out | a `HealPixDivCurlPyramid` (`.div`, `.curl`, fine to coarse) |
| Units | none applied: the E and B parts synthesised as scalar maps; multiply the alm by $\sqrt{\ell(\ell+1)}/R$ for s⁻¹ | s⁻¹ for u, v in m/s (spacing from `radius_m`, 6 371 008.8 m by default) | s⁻¹, each band at its own spacing |
| Default ellipsoid | `"sphere"` (transform) | `"sphere"` | inherited from the decomposition (`"WGS84"` by default) |
| Gaps and masks | none (whole sphere) | NaN propagates; near a mask edge missing taps take the centre value, so the derivative is one-sided | same, per band; the coarsest band can hold very few cells |
| Optional dependencies | `quaternionic`, `spherical` | none | none |
| Differentiable | yes | yes | yes |

## Rules of thumb

- **Global field (reanalysis, model output on the whole sphere)** → the
  spectral route, as in the validation pipeline of {doc}`divcurl`: it has no
  edges and no mask, and is exact up to `lmax` once the
  $\sqrt{\ell(\ell+1)}/R$ factor is applied.
- **Regional domain, coastline or mask** → `HealPixDivCurl` at one scale, or
  `HealPixMultiScaleDivCurl` to see each scale separately.
- Pass the same ellipsoid to `HealPixDivCurl` and to the decomposition you
  compare it with: their defaults differ.
