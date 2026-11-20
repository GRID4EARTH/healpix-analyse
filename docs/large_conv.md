# Large receptive-field convolution

`LargeConv` provides large effective convolution kernels without constructing
the expensive fine-resolution interpolation stencil required by a direct
`HealPixConv`.

It applies a matched multiresolution operator:

$$
y = U_1 U_2 \cdots U_L\; C_{W_c}\;D_L\cdots D_2D_1 x + b,
$$

where:

- $D_i$ is a smooth `HealPixDown`;
- $C_{W_c}$ is a compact trainable `HealPixConv`;
- $U_i$ is the `HealPixUp` constructed from the same smoothing parameters;
- $b$ is added after returning to the fine resolution.

The compact kernel is learned end-to-end through the complete chain. Its
effective fine-resolution kernel therefore includes the exact Down/Up
operators used at inference.

## Quick start

```python
import torch
from healpix_analyse import LargeConv

layer = LargeConv(
    level=8,  # nside = 256
    in_channels=8,
    out_channels=16,
    kernel_sz=33,
    max_compact_kernel_sz=7,
    cell_ids=cell_ids,
    ellipsoid="sphere",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

x = torch.randn(4, 8, len(cell_ids), device=layer.device)
y = layer(x)
print(y.shape)  # [4, 16, len(cell_ids)] for one gauge
```

## Selecting the resolution and compact kernel

Kernel size is converted using its radius rather than its diameter. For a
requested fine kernel $K_f$:

$$
r_f = \frac{K_f-1}{2}, \qquad
r_c(L) = \left\lceil\frac{r_f}{2^L}\right\rceil, \qquad
K_c(L)=2r_c(L)+1.
$$

The smallest $L$ satisfying

$$K_c(L) \leq K_{\max}$$

is selected automatically.

For `kernel_sz=33` and `max_compact_kernel_sz=7`:

| Down levels | Coarse radius | Compact kernel | Accepted? |
|---:|---:|---:|---:|
| 0 | 16 | 33 | no |
| 1 | 8 | 17 | no |
| 2 | 4 | 9 | no |
| 3 | 2 | 5 | yes |

Therefore a requested `33×33` receptive field uses three Down operations, a
`5×5` compact convolution, and three Up operations. A `7×7` compact kernel at
that level would have a nominal fine diameter of `49×49`.

The selected plan is exposed through:

```python
print(layer.n_levels)                 # 3
print(layer.compact_kernel_sz)        # 5
print(layer.coarse_nside)             # nside // 8
print(layer.effective_kernel_sz)      # 33
```

When the requested kernel already fits within the configured maximum, no
resolution change is performed and `LargeConv` reduces to one `HealPixConv`
plus its fine-resolution bias and optional normalisation.

## Gaussian smoothing of a given width

### The problem: the width is a staircase

Every level adds a blur of its own, and so does the compact `HealPixConv`
(it is not the identity: with a point-like kernel it already blurs by about
0.74 cell). `down_up_blur(level, n)` measures the narrowest response of a
`LargeConv` with `n` levels: the response to one cell with a point-like
compact kernel.

Conventions: all widths are **one-axis RMS widths**,
`sqrt(sum(w d**2) / (2 sum(w)))` for the response `w` and the angular
distance `d` to the impulse cell. This is the standard deviation of a
Gaussian, the same quantity as `sigma_m` of {doc}`gaussian_filter
<radial_filter>`, not its FWHM (2.355 sigma). Functions take and return
**radians**; `gaussian_plan` and `LargeConv.from_gaussian` also accept
`sigma_m` in metres, converted with the cell size of `gaussian_filter`
(square root of the cell area on the authalic sphere), so that a width in
metres means the same in both. "Cell" means the nominal cell size
`sqrt(4 pi / 12) / 2**level` (0.00100 rad at level 10), which is that cell
size divided by the authalic radius.

| Levels | `down_up_blur`, level 10 (cells) | (rad) |
|---:|---:|---:|
| 0 | 0.74 | 0.00074 |
| 1 | 2.01 | 0.00201 |
| 2 | 4.20 | 0.00420 |
| 3 | 8.33 | 0.00833 |
| 4 | 16.5 | 0.0165 |

With the automatic level rule the number of levels follows from `kernel_sz`
and the width of a fitted Gaussian sits on these values. Measured at
level 10 over 756 requested widths from 2.25 to 40 cells
(`benchmarks/benchmark_largeconv_widths.py`), the effective width has 592
flat steps out of 595 (up to 32 cells) and is wrong by up to 106 %. A cost
function of the width is constant on those plateaus, which stalls an
optimiser.

A first attempt, choosing the number of levels from the blur floor and
subtracting the floor from the width, is not enough either. Just above a
switch the compact Gaussian is much narrower than one coarse cell, the
sampled Gaussian is a point, and the width goes through a shelf; at the
first switch it even steps down (2.11 to 2.05 cells). The recipe below
replaces it.

### The recipe: `LargeConv.from_gaussian`

```python
from healpix_analyse import LargeConv

layer = LargeConv.from_gaussian(level, sigma_rad, cell_ids)
y = layer(x)
```

One call: it computes `gaussian_plan`, builds the layer with the same
options and sets the kernel. The calibration location is the block that
contains `reference_cell`; by default it is a cell at the centre of
`cell_ids` (the equatorial block when `cell_ids` is `None`). The plan is
kept in `layer.gaussian_plan`. `gaussian_plan(level, sigma, cell_ids=...)`
is public for inspection (levels, compact sigma, kernel sizes); the manual
construction is

```python
plan = gaussian_plan(level, sigma_rad, cell_ids=cell_ids)
layer = LargeConv(level, 1, 1, plan.fine_kernel_sz,
                  n_levels=plan.n_levels, cell_ids=cell_ids)
layer.set_gaussian_kernel(plan)
```

`set_gaussian_kernel` refuses a layer that has other levels than the plan
(forgotten `n_levels`) or a compact kernel too small for the Gaussian
(`plan.kernel_sz` passed instead of `plan.fine_kernel_sz`).

For each number of levels `n`, `gaussian_plan` measures the real
`LargeConv` response to the centre cell of the calibration block for compact
Gaussians of width 0, 0.05, ..., 2.25 coarse cells. The table includes
everything: the Down/Up blur, the compact convolution, and the sampling of
the Gaussian on the coarse grid. The number of levels goes from `n - 1` to
`n` where level `n` reaches the requested width with a compact Gaussian of
0.5 coarse cell, and the compact width is the one for which the table gives
the requested width. Both branches give exactly the requested width at the
switch, so the width is continuous, and strictly increasing, in sigma.

Measured at level 10 on the calibration patch (256 x 256 cells near the
equator), 756 widths from 2.25 to 40 cells including the four switches
(4.7, 9.35, 18.65 and 36.9 cells):

- the effective width of the response to the calibration cell is strictly
  increasing (smallest step 0.048 cell for a sigma step of 0.05 cell; no
  shelf, no step down);
- it equals the requested width within 3.8e-4 relative (median 2.3e-5).

**This accuracy is for the calibration cell.** Other cells and other places
differ, as follows.

- **Other cells of the domain.** Measured on a 512 x 512 cell patch
  (level 10, near the equator, calibrated on its centre cell), response to 96
  random cells (fixed seed) drawn among those at least `5 sigma + 2**n` fine
  cells from the patch edge, so that the tail and one coarse block are
  inside; the sampling windows cover 7.5 or more coarse blocks of the plan
  in each direction (the automatic rule has coarser blocks: 1.9 at sigma 36,
  10 at 18, 26 at 9, more below). Ratio to the requested width (plan) and
  to its own mean (automatic rule, which does not aim at the requested
  width), min / mean / max:

  | sigma (cells) | plan levels | plan width / sigma | plan total | automatic rule: width / own mean | automatic rule: total |
  |---:|---:|:---|:---|:---|:---|
  | 3 | 1 | 0.921 / 0.986 / 1.043 | 0.948 / 0.999 / 1.046 | 0.971 / 1.000 / 1.021 | 0.932 / 0.998 / 1.045 |
  | 6 | 2 | 0.971 / 0.991 / 1.009 | 0.945 / 0.999 / 1.047 | 0.984 / 1.000 / 1.016 | 0.945 / 0.999 / 1.048 |
  | 9 | 2 | 0.988 / 0.996 / 1.003 | 0.943 / 1.001 / 1.047 | 0.981 / 1.000 / 1.018 | 0.941 / 1.001 / 1.047 |
  | 18 | 3 | 0.992 / 0.997 / 1.001 | 0.931 / 0.994 / 1.046 | 0.980 / 1.000 / 1.014 | 0.920 / 0.994 / 1.044 |
  | 36 | 4 | 0.993 / 0.999 / 1.001 | 0.940 / 0.996 / 1.040 | 0.987 / 1.000 / 1.005 | 0.917 / 0.990 / 1.024 |

  The width spread of the plan is 8 % wide at 3 cells (0.92 to 1.04, up to
  240 cells from the calibration cell, which lies at 0.9999), 3 % at 6 cells and
  at most 1.5 % from 9 cells on. The automatic rule, which has the same
  Down/Up chain, spreads as much around its own mean, so this is a property
  of the chain and not of the recipe. The width ratio varies with the
  distance from the calibration cell for narrow widths (within 16 cells of
  it the spread at 3 cells is 0.997 to 1.019); the total of one cell is 0.93
  to 1.05 everywhere. Calibrating on the cell-averaged width was not
  adopted: the mean is already within 1.4 % of the request at 3 cells
  (0.986) and within 1 % above, and averaging would not reduce the spread.
- **The total of one cell is not conserved.** `LargeConv` is not a
  conservative operator: the sum of the response to one cell is 0.92 to
  1.05 of the input in the table above, for the plan and for the automatic
  rule alike. Constants are preserved when the kernel sums to one, but the
  integral of a field is not. A conservative smoothing would need another
  operator, such as a conservative FFT projection; `gaussian_filter` with
  `normalization="present"` normalises each row to one over the cells present
  (and `"full_disc"` loses the mass that leaves the domain), see
  {doc}`radial_filter`.
- **Tail near the patch edge (measurement artefact and domain edge).** On
  the same patch, cells up to 20 cells from the edge (sampling window of
  236 cells half-width, 29 to 118 plan blocks) give width / sigma 0.985 to
  1.003 at 9 cells, 0.925 to 1.001 at 18 and 0.851 to 1.001 at 36, and the
  total 0.896 to 1.064 (plan) and up to 1.18 (automatic rule at 36 cells):
  the tail that falls outside the patch is lost. Wide spreads seen with
  impulses that close to an edge are this effect; with the margin of the
  table above they disappear (0.988 or better at 9 cells and above). The
  next bullet gives the same effect at the edge of a real partial domain.
- **Domain edge (partial sky).** Down and Up use zero padding, so the tail
  that falls outside the domain is lost. In a 15-degree disc at level 8
  (13 411 cells), the response to a cell at 0.92 of the radius is
  narrower than requested by 6.4 % (sigma 3 cells) and 12.7 % (sigma 6),
  with a total of 0.91 and 0.92; at 0.8 of the radius 0.6 % and 3.0 %; at
  the centre 0.5 % and 0.04 %. A constant field stays 1 in the interior and
  drops to 0.98 at the edge.
- **Other places.** The compact convolution samples the Gaussian on a
  lattice whose shape depends on the position. With the equatorial
  calibration, the response at 55 degrees latitude is up to 21.8 % wider
  than requested (22.4 % in an independent sweep; median 0.4 %): 13 to 22 %
  below 3 cells, 5 to 12 % from 3 to 4.7 cells, 1 to 7 % from 4.7 to 9.35,
  at most 3.6 % above, and still strictly increasing. That is why
  `from_gaussian` and `gaussian_plan(cell_ids=...)` calibrate on the block
  of the domain by default. At 55 degrees that gives an error of 2.9e-4 and
  a narrowest width of 2.75 cells.

Limits:

- **Narrowest width.** Level 1 with a compact Gaussian of 0.5 coarse cell:
  2.21 cells (0.0022 rad) at level 10 near the equator, 2.75 cells at 55
  degrees latitude. Below it `gaussian_plan` raises `ValueError` (the message
  gives radians and cells). Narrower widths are not in this class: use
  `gaussian_filter` ({doc}`radial_filter`), which with `width_rule="matched"`
  reaches the requested width down to about half a cell (its default `"sampled"` is narrower than requested
  below about one cell).
- **Widest width.** Set by `max_levels` and the calibration patch
  (`min(level, patch_depth) - 3` levels, 5 at level 10) and the largest
  calibrated compact width (2.25 coarse cells); `gaussian_plan` raises
  `ValueError` above it.
- **Cost and cache.** The first call for a block builds 65 536-cell
  `LargeConv` layers and evaluates 46 kernels per level count: 16 to 19 s
  for the first call (it calibrates one and two levels), 1 to 2 s for each
  further level count, then cached. A new block (a domain elsewhere) costs
  about 16 to 19 s again. The cache holds 32 entries, one per (level, number
  of levels, options, block), so about six blocks; the default location and
  an explicit `reference_cell` in the same block share their entries.
  `down_up_blur` is cached the same way. Later calls are table look-ups.
  The compact kernel can be as large as 25 x 25.
- **Not measured.** Other levels than 8, 9 and 10, real data, GPU,
  ellipsoids other than WGS84, other `radius_deg`, `sigma_deg`,
  `weight_norm` and `up_norm`.

### Choosing the levels yourself

`LargeConv(..., n_levels=n)` fixes the number of levels; the compact kernel
size then follows from `kernel_sz` (`2 ceil(r_f / 2**n) + 1`). It cannot be
combined with `max_compact_kernel_sz`. Without `n_levels` nothing changes:
the automatic rule above is used and the results are bit-identical to
before.

## Why the compact kernel is learned directly

Naively resizing a `33×33` array to `5×5` would ignore:

- the smoothing transfer function of `HealPixDown`;
- the transpose and normalisation used by `HealPixUp`;
- HEALPix geometry and bilinear stencil interpolation;
- gauge orientation;
- partial-patch boundaries.

Learning $W_c$ through the whole chain optimises the actual operator
$U^LC_{W_c}D^L$. Gradients pass through every sparse matrix multiplication,
the compact convolution, and each interpolation operation.

The representation is intentionally lower rank than an arbitrary direct
large kernel. Downsampling removes high-frequency information, so an
oscillatory fine kernel cannot in general be reproduced exactly. Large smooth
kernels and learned contextual filters are the intended use cases.

## Matched Down and Up operators

Every pair receives the same:

- `ellipsoid`;
- `radius_deg`;
- `sigma_deg`;
- `weight_norm`;
- floating-point dtype and device.

In addition, every `HealPixUp` is constructed with `paired_down=down_layer`.
It transposes the already stored sparse Down matrix instead of rebuilding an
operator on a potentially different set of fine children. The pairing is
therefore exact even for irregular partial patches.

Only `HealPixDown(mode="smooth")` is used. Max pooling is nonlinear and cannot
define a linear effective convolution kernel.

`up_norm="col_l1"` with `weight_norm="l1"` is the default combination. It
preserves constant fields and is a suitable default for learned features.

```python
layer = LargeConv(
    level=7,
    in_channels=4,
    out_channels=8,
    kernel_sz=65,
    weight_norm="l1",
    up_norm="col_l1",
)
```

Other supported combinations are inherited from `HealPixDown` and
`HealPixUp`:

- `up_norm="adjoint"` gives the raw transpose of Down;
- `weight_norm="l2", up_norm="diag_l2"` favours local energy consistency.

## Bias, normalisation and activation

The internal compact convolution has a fixed zero bias. `LargeConv.bias` is
added only after the final Up operation, so it represents a true
fine-resolution bias even when partial-patch boundary normalisation is not
uniform.

When `use_norm=True`, GroupNorm and ReLU are also applied at the final fine
resolution:

```text
Down × L → compact convolution → Up × L → bias → GroupNorm → ReLU
```

Applying these nonlinear operations at the coarse level would define a
different operator and would prevent interpretation as one effective large
convolution.

## Input and output shapes

| Input | Interpretation | Output |
|---|---|---|
| `[N]` | one single-channel map | `[N]` when total output channels = 1, otherwise `[C_out, N]` |
| `[B, N]` | batch of single-channel maps | `[B, G*C_out, N]` |
| `[B, C_in, N]` | general batch | `[B, G*C_out, N]` |

NumPy input returns NumPy output. Torch input returns Torch output and retains
autograd connectivity.

## Partial HEALPix patches

`cell_ids` must be unique NESTED identifiers at the fine level. `LargeConv`:

1. sorts identifiers internally to match the sparse Down matrices;
2. builds the parent cell set at every coarse level;
3. builds each Up operator from the exact corresponding parent set;
4. crops extra children produced by partial-patch upsampling;
5. restores the original `cell_ids` order before returning.

```python
layer = LargeConv(
    level=6,
    in_channels=1,
    out_channels=2,
    kernel_sz=33,
    cell_ids=cell_ids,
)
```

As with every local convolution, values close to the patch boundary have less
context. Supply a halo around the requested output region and crop the final
result when accurate boundary responses are important.

## Gauges

The internal convolution supports the same gauge options as `HealPixConv`:

- `"phi"`;
- `"cosmo"`;
- `"projected_ref"`;
- `"two_ref"`.

`n_gauges`, `singularity_lonlat` and `ref_direction` are forwarded unchanged.
The output contains `n_gauges * out_channels` channels.

## Kernel access and initialisation

The compact trainable parameter is available as `layer.weight` and has shape

```text
[n_gauges, in_channels, out_channels, compact_kernel_sz**2]
```

A fixed or custom initial compact kernel can be installed with:

```python
layer.set_compact_kernel(weight, bias=bias, requires_grad=True)
```

The provided array describes the compact coarse kernel, not a fine `33×33`
kernel. Fitting a compact kernel from a pre-existing large fine kernel requires
an explicit operator-matching optimisation and is outside the current API.

## Computational benefit

A direct convolution stores and gathers values proportional to

$$K_{\text{fine pixels}}\,K_f^2.$$

After $L$ levels, the dominant compact convolution scales approximately as

$$\frac{K_{\text{fine pixels}}}{4^L}\,K_c^2,$$

plus sparse Down/Up operations. For `33×33 → 5×5` with three levels, the
convolution gather term is reduced nominally by

$$
4^3\frac{33^2}{5^2} \approx 2788.
$$

Actual speedup depends on batch size, channels, patch shape, device and the
sparse-matrix overhead.

## Limitations

- The effective kernel is approximate in the sense of representational
  bandwidth; it is exactly the operator learned through the selected chain.
- `nside` must remain at least one after all selected downsamplings.
- Only NESTED ordering is currently supported.
- Geometry construction can be expensive; the internal `HealPixConv` cache is
  enabled by default.
- Full-sphere operation at very high `nside` still requires substantial memory
  for Down/Up sparse matrices.
- Partial patches should include a halo for reliable edge values.

See `Notebooks/large_conv_test.ipynb` for configuration, constant-field,
ordering, gradient, receptive-field and training tests.
