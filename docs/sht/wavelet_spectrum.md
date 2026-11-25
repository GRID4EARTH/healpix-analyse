# Wavelet power spectrum (NaN- and mask-aware)

**Module** `healpix_analyse.wavelet_spectrum`
**Classes** `HealPixWaveletSpectrum`, `HealPixIsotropicWaveletSpectrum`
**Function** `wavelet_power_spectrum`

A band power spectrum for the data we actually have: a regional domain or the
full sphere, with NaN for land, clouds, swaths or missing tiles. At each
HEALPix level, a wavelet is applied with `HealPixConv` and the field goes one
level down with `HealPixDown`; the spectrum is the mean of `|y|²` at each
level, corrected for the part of the wavelet that fell on missing pixels.
Everything is PyTorch: differentiable, batched, CPU or GPU, usable as a loss.

```{include} ../_generated/glance/wavelet_spectrum.md
```

## Quick start

```python
import numpy as np
import torch
from healpix_analyse import HealPixWaveletSpectrum, HealPixIsotropicWaveletSpectrum

level = 8                                   # about 25 km cells
sst = torch.as_tensor(np.load("sst_level8.npy"))   # [12 * 4**8], land = NaN

ws = HealPixWaveletSpectrum(level)          # complex Morlet, 4 orientations
out = ws(sst)
out.power_total     # [S] band powers, fine to coarse (one per level, 8 .. 0)
out.power           # [S, 4] per orientation
out.monopole        # mean of the valid pixels (the frequency 0)
out.lowpass_var     # variance of the 12 level-0 pixels around it
out.n_eff           # [S] sum of the weights of each band; 0 -> NaN band
out.scale_km        # [S] nominal wavelength of each band

# isotropic: one real radial wavelet, no orientations, cheaper
iso = HealPixIsotropicWaveletSpectrum(level)(sst)

# cross-spectrum of two fields on the same cells (their NaN may differ)
t2m = torch.as_tensor(np.load("t2m_level8.npy"))
cr = ws.cross(sst, t2m)
cr.co_total, cr.coherence
```

A regional domain is given by its NESTED cell ids, as for the other operators:
`HealPixWaveletSpectrum(level, cell_ids=ids)`. Inputs are `[..., N]` (NumPy or
Torch); outputs are Torch tensors with the same leading dimensions.

## What it computes

Let `x` be the map at level J, `w` its weight (0 where `x` is NaN or masked,
optionally a `weights=` map in [0, 1]).

1. **Descent.** The pair `(q, m) = (w x, w)` goes down to level 0 with
   `HealPixDown(mode="mean")` (`down_mode="mean"`, the default), made
   area-weighted: a coarse value is the sum of its 4 children divided by 4, a
   child outside `cell_ids` counting 0. (`HealPixDown` divides by the number
   `n_P` of children present in `cell_ids`; the spectrum multiplies by
   `n_P / 4`.) The sums are then conserved,
   `Σ_k q_{j,k} = Σ_k q_{J,k} / 4^{J−j}`, on a regional domain as on the full
   sphere; `m_j` stays in [0, 1], `q_j / m_j` is the mean of the valid
   descendants, and a cell outside `cell_ids` counts exactly like a NaN, so
   a regional domain gives the same spectrum as the full sphere with NaN
   outside it.
   The fraction `f_j` of descendants with `w > 0` goes down the same way, and
   a level-`j` pixel is valid when `f_j ≥ min_valid` (0.5); at J, every pixel
   with `w > 0` is valid. A `weights=` taper therefore never drops a pixel
   from a band: it only sets its share of the averages (`u = m · e`, below).
2. **Frequency 0.** `M = Σ_k q_{0,k} / Σ_k m_{0,k}`, the `m`-weighted mean
   of the level-0 pixels, which by the conservation above is
   `Σ w x / Σ w`, the mean of the valid pixels (within 1e-12 in float64 in
   the tests). In practice `q` is first centred on the direct
   mean `M̂ = Σ w x / Σ w`, `q = w (x − M̂)` (see below), and
   `M = M̂ + Σ q_0 / Σ m_0`, the same number.
   `down_mode="smooth"` (the Gaussian descent of `HealPixDown`) is the other
   accepted descent: it aliases less (next section) but does not conserve
   the mean (see {doc}`/other/down`), so `M` is then the direct mean `M̂`
   and the level-0 pixels do not average to it. `maxpool` and
   `smooth_conservative` raise, because they do not keep `q / m` a weighted
   mean of the valid pixels.
3. **Bands.** At every level `j` from J down to 0, the field
   `x_j = q_j / m_j` is convolved with the wavelet
   (`HealPixConv(missing="local_mean")`, below). For orientation `g` and the
   weights `u` of the next section:

$$
P_j^{(g)} = \frac{\sum_k u_{j,k}\,|y^{(g)}_{j,k}|^2 / e_{j,k}}{\sum_k u_{j,k}},
\qquad
S1_j^{(g)} = \frac{\sum_k u_{j,k}\,|y^{(g)}_{j,k}| / \sqrt{e_{j,k}}}{\sum_k u_{j,k}},
\qquad P_j = \sum_g P_j^{(g)}
$$

4. **Low-pass.** `P_low`, the `m`-weighted variance of the 12 level-0 pixels
   around `M` (multipoles ℓ of about 1 and 2 with `"smooth"`; with `"mean"`
   it also picks up aliased power from all scales, next section).

The wavelets have zero sum, so they do not see `M`. Centring before the
descent (the descent is linear, so it changes nothing else) avoids the
float32 cancellation of a field such as SST in kelvin at the coarse levels:
in float32 (the default) the band powers agree with float64 within the
rounding of the input. `S1` (the mean `|y|`)
is optional (`s1=True`); higher-order scattering statistics are not part of
this module.

## Missing pixels: completion inside `HealPixConv`

A stencil value of `HealPixConv` is a bilinear interpolation of 4 pixels. The
spectrum uses the option `missing="local_mean"` of `HealPixConv`
({doc}`/conv/convol`), which treats a NaN pixel exactly like a pixel outside
`cell_ids`:

- a stencil point with at least one valid neighbour takes the renormalised
  interpolation of its valid neighbours;
- a point with no valid neighbour takes the local mean of the valid stencil
  values of the same output pixel (Gaussian weights of the wavelet envelope);
- the valid fraction of every stencil point gives the coverage
  `e = Σ |ψ_p|² cov_p / Σ |ψ_p|²` of the kernel at each pixel.

Every valid pixel therefore has an output, at the coast as at the border of a
regional domain, with no halo. Since `Σ ψ = 0`, completing with the local mean
removes the jump that a constant fill would put at the coast. `|y|²` is
divided by `e` and, by default, each pixel enters the mean with the weight
`u = m · e` (`coverage_weight=True`): no valid pixel is left out, and those
whose kernel is mostly on land count less.

| `edge=` | Missing stencil points | Pixels in the mean |
|---|---|---|
| `"mean_corrected"` (default) | completed with the local mean; the squared output is divided by `e` | every valid pixel, weight `m · e`; `min_coverage=τ` drops those with `e < τ` |
| `"complete"` | not completed | only the pixels whose whole kernel is valid |
| `"none"` | set to the frequency 0 | every valid pixel |

On a toy (flat grid, 35 % land with an irregular coast, red field
`P(k) ∝ k⁻³` like SST, Morlet G = 4, 5×5), keeping every valid pixel, the bias
of the band power from the finest band to four octaves down is:

| Completion | Bias |
|---|---|
| frequency 0 (`"none"`) | +53 to +220 % |
| centre value (the `"centre"` rule of `HealPixConv`) | −1.6 to −5.7 % |
| true mirroring across the coast | +1.1 to +4.7 % |
| local mean, weight `e` (default) | −0.3 to −1.1 % |
| local mean, τ = 0.9 (drops 5–18 % of the pixels) | −0.1 to 0.0 % |

These numbers come from a toy run during the design of this feature; the
same comparison on real ERA5 coastlines at level 8 is still to be done (see
the last section).

## The wavelets

- **Oriented** (`HealPixWaveletSpectrum`): the complex Morlet
  `ψ(ρ, φ) = e^{−ρ²/2σ²} (e^{i k₀ ρ cos φ} − e^{−k₀²σ²/2})`
  (`wavelet_morlet`), sampled on the stencil with zero
  sum and unit L2 norm, in `G = 4` orientations spaced π/4 (the gauges of
  `HealPixConv`). Defaults: `k₀ = 2.4` rad/pixel, `σ = 0.8` pixel, 5×5.
- **Isotropic** (`HealPixIsotropicWaveletSpectrum`): the angle average of the
  same Morlet, `e^{−ρ²/2σ²} (J₀(k₀ρ) − e^{−k₀²σ²/2})`, computed as a mean of
  the oriented profile over 64 angles (no Bessel function, no SciPy)
  (`wavelet_morlet_isotropic`). No orientation, a real
  output, one gauge: the gather of `HealPixConv` is 4 times smaller at the
  same kernel size. Defaults: `k₀ = 2.0`, `σ = 1.0`, 7×7 (with the oriented
  defaults on a 5×5 stencil the response still varies by about 15 % with
  direction; with these, about 2 %).
- **Scales per octave.** The HEALPix descent is dyadic. `scales_per_octave=2`
  applies two wavelets per level, dilated by 1 and √2, and makes each band
  half as wide (`σ k₀ ≈ 1.9 Q`). The kernels must then be larger, so their
  size is derived and checked by the code: `kernel_sz = 2⌈2.2 σ_max⌉ + 1`
  (11 for Q = 2), and a value that cuts more than 0.1 % of the envelope energy
  raises an error. Narrower bands are also narrower in angle, so the oriented
  class uses `G = 4 Q` orientations. Q ≥ 3 is not available.
- **Coarse levels.** Bands go down to level 0. A stencil whose corners would
  be more than `max_stencil_deg` (180°) from its centre falls back to the 5×5
  dyadic wavelet: for 11×11 at levels 1 and 0, for 7×7 at level 0. At levels
  0 and 1 the stencil is no longer local (166° and 83° for 5×5): these bands
  are global filters at ℓ of a few units, read through their measured windows.

## Cross-spectra

`ws.cross(x, y)` takes two fields on the same cells (same shape, same order).
Each keeps its own frequency 0, descent and completion; the joint weight is
`√(u_x u_y)` (the intersection of the two masks for binary weights).

$$
C_j^{xy,(g)} = \frac{\sum_k u^{xy}_k\, y^{x,(g)}_k\, \overline{y^{y,(g)}_k}\,/\sqrt{e^x_k e^y_k}}{\sum_k u^{xy}_k},
\qquad
r_j = \frac{\sum_g \mathrm{Re}\, C_j^{xy,(g)}}{\sqrt{P_j^{x|xy}\, P_j^{y|xy}}}
$$

`co` is the real part (co-spectrum) and `quad` the imaginary part (oriented
class only), non-zero when `y` is shifted relative to `x` along the
orientation. The coherence uses auto spectra computed on the same joint
weights, so `|r_j| ≤ 1`. With `ws.cross(xs)` and `xs` of shape `[..., C, N]`,
every pair is returned (`[..., C, C, S]`). `cross(x, x)` is the auto spectrum.

## Windows and Littlewood–Paley

`ws.windows()` measures, on the full sphere, the response of every band to
unit-power multipoles: for each ℓ, the `2ℓ + 1` real orthonormal harmonics
are synthesised with `HEALPixSHT.alm2map` and passed through the spectrum.
For an isotropic field, `E[power_total] = Σ_ℓ C_ℓ response[s, ℓ]`; `ell_eff`
is the centroid of each band. It costs `(lmax + 1)²` maps and the tables of
`HEALPixSHT` ({doc}`healpix_sht`): use levels up to about 5.

`ws.littlewood_paley()` fits one constant per band and one for the low-pass
(least squares, `torch.linalg.lstsq`), so that
`(constants * power_total).sum(-1) + lowpass * lowpass_var` estimates the
variance. The band powers themselves are returned unnormalised (unit-L2
taps).

Both are measured through the operator itself, so they follow the descent.
`ell_nominal` (`k₀` over the pixel size of the level) and `scale_km` do not
depend on it. Measured at level 4 (full sphere, float64):

| | `down_mode="mean"` (default) | `down_mode="smooth"` |
|---|---|---|
| `ell_eff / ell_nominal`, bands below J | 1.03 to 1.26 | 0.49 to 0.71 |
| share of the band from ℓ > 3 nside_j, white noise | 44 to 55 % | ≤ 1.1 % |
| same, `C_ℓ ∝ (ℓ+1)⁻³` | 5 to 13 % | ≤ 0.05 % |
| `lowpass` window summed over ℓ (white noise) | 0.88 | 0.09 |
| Littlewood–Paley ripple, oriented / isotropic | 0.21 / 0.53 | 0.39 / 0.47 |

The finest band does not go through the descent and is the same for both.
The boxcar of `"mean"` barely attenuates the top of each band, so the bands
sit at their nominal ℓ; but it does not cut the multipoles that the grid of
level `j` cannot represent (ℓ > 3 nside_j), and folds them into the band and
into the low-pass. For a red field such as SST this is 5 to 13 % of a
coarse band; for white noise, about half. The Gaussian of `"smooth"` cuts
them, at the price of bands centred at about 0.6 times their nominal ℓ and
of a mean that is not conserved. The windows include the aliased part, so
the comparison of a band with the `anafast` of the same map stays exact on
average; map to map it scatters more (next section).

## Validation

Measured with the tests of `tests/test_wavelet_spectrum.py` and
`tests/test_convol_missing.py` (CPU, float64):

- **Against `anafast`** (full sphere, `ellipsoid="sphere"`): each map's band
  power against `Σ_ℓ Ĉ_ℓ response[s, ℓ]`, with `Ĉ_ℓ` the `anafast` of the same
  map. At level 4, over 60 Gaussian maps with `C_ℓ ∝ (ℓ+1)^{-α}`,
  α = 0, 2, 3, the largest deviation per band (oriented / isotropic):

  | Band level | 4 (finest) | 3 | 2 | 1 | 0 |
  |---|---|---|---|---|---|
  | `"mean"` (default) | 4.1 / 2.7 % | 8.5 / 7.5 % | 19 / 15 % | 53 / 48 % | 132 / 152 % |
  | `"smooth"` | 4.1 / 2.7 % | 1.1 / 1.6 % | 3.9 / 3.6 % | 5.3 / 6.8 % | 48 / 51 % |

  At level 0 the 5×5 stencil covers the sphere. The residual comes from the
  pixelisation, the finite number of orientations and, below the finest
  band, the descent: the operator is not exactly rotation-invariant, and the
  power that the boxcar folds into a band depends on how the field sits on
  the pixels. The test runs at level 3, with tolerances per descent (6 % for
  the finest band; with `"mean"` 25 % one level down and 40 % two levels
  down, with `"smooth"` 6 % and 20 %).
- **Littlewood–Paley** at level 4: with the fitted constants, the weighted
  band powers match the variance of power-law maps within 9 % with either
  descent (`"mean"`: at most 6.3 % oriented and 8.0 % isotropic; `"smooth"`:
  6.9 % and 8.4 %); the fitted
  sum of windows still varies by 21 % (oriented) and 53 % (isotropic) over
  `1 ≤ ℓ ≤ 3 nside − 1` with `"mean"` (39 % and 47 % with `"smooth"`),
  because no band reaches the highest multipoles and the level-0 band
  overlaps the low-pass.
- **Frequency 0**: with `"mean"`, `monopole` equals `Σ w x / Σ w` (the
  `nanmean` without `weights=`) within 1e-12 on random fields with NaN, on
  the full sphere and on an unsorted regional domain whose parents miss
  some of their children at every level, and so does `Σ q_0 / Σ m_0` of the
  descent itself; without the `n_P / 4` factor the latter is off. A regional
  domain gives the same spectrum as the full sphere with NaN outside it.
- **Constants and NaN**: a constant map with 20 % NaN, on the full sphere or a
  regional domain, gives zero band powers and the right frequency 0 for every
  edge mode; NaN inputs give finite outputs and gradients, and the gradient at
  a NaN pixel is 0; `gradcheck` passes in float64.
- **`HealPixConv` default unchanged**: identical outputs to the previous
  implementation (full sphere and unsorted regional domain); a NaN pixel and
  a pixel outside `cell_ids` give the same result with `missing="local_mean"`.

## Cost

The cost is that of `HealPixConv` at the finest level: its gather holds
`B · C · G · K · P` values (K cells, P = `kernel_sz²`), twice with
`missing="local_mean"` (values and mask); each coarser level costs 4 times
less. At level 8 on the full sphere (K = 786 432), one map, float32: about
0.3 GB per gathered tensor for the oriented default (G = 4, 5×5), 0.15 GB for
the isotropic default (7×7), 3 GB for the oriented class with Q = 2 (G = 8,
11×11). GPU timing: to be measured.

## Differentiability and dependencies

Differentiable with respect to the data and the weights (no NumPy in the
data path, NaN removed with `torch.where`); the mask itself only selects.
Batched over any leading dimensions. Dependencies: PyTorch and healpix-geo,
as `HealPixConv` and `HealPixDown`; no SciPy.

## Limitations and what remains

- The band powers are not yet normalised by default: the Littlewood–Paley
  constants are fitted on demand on the full sphere at small levels; tuning
  `σ`, `k₀` and the coarse bands to reduce the ripple, and shipping constants
  for large levels, remain to be done.
- With the default `"mean"` descent the bands below the input level alias
  (see the windows above); `"smooth"` does not, but does not conserve the
  mean. Taking the bands from the Gaussian descent and only the frequency 0
  and the low-pass from the mean descent would give both, for one more
  descent of `(q, m)`; not done here.
- Validation on ERA5 SST at level 8 (real coastlines), GPU timing, and a
  performance path for levels ≥ 10 (with the gather optimisation of
  `HealPixConv`) are still to do.
- `scales_per_octave` is 1 or 2.
- `windows()` is full sphere only; on a regional domain, use the windows of
  the full sphere as an approximation.
