# `HealPixDown` — HEALPix resolution reduction

**Module** `healpix_analyse.down`  
**Class** `HealPixDown`  
**Inherits** `torch.nn.Module`

```{include} ../_generated/glance/down.md
```

---

## Overview

`HealPixDown` reduces the HEALPix resolution by a factor of 2:

$$n_{\text{side,out}} = n_{\text{side,in}} \mathbin{/} 2, \qquad N_{\text{out}} = N_{\text{in}} \mathbin{/} 4$$

Four strategies are available:

- **`"smooth"`** — Gaussian-weighted average over a disc.  Implemented as
  a sparse linear operator `M @ x` (differentiable, usable in autograd).
- **`"smooth_conservative"`** — the same Gaussian smoothing, plus one number
  added to every output cell so that the mean of the output equals the mean
  of the input.
- **`"mean"`** — unweighted mean of the 4 direct NESTED children; conserves
  the global mean exactly, but nothing crosses parent boundaries.
- **`"maxpool"`** — Non-linear maximum over the 4 direct NESTED children
  of each coarse pixel. Differentiable almost everywhere: the gradient goes
  to the largest child.

All modes work on the whole sphere or on a regional set of cells defined by
`cell_ids` (but see the partial-domain issue of `"maxpool"` below).

Input shapes `[N]` and `[B, N]` only (a `[B, C, N]` tensor raises
`ValueError`: reshape it to `[B·C, N]`). NumPy arrays and torch tensors are
both accepted and the return type mirrors the input type, cast to the
operator `dtype` (float32 by default). `forward` returns a tuple
`(y, cell_ids_out)`.

---

## Constructor

```python
HealPixDown(
    level,
    mode        = "smooth",
    ellipsoid   = "WGS84",       # third positional argument
    radius_deg  = None,
    sigma_deg   = None,
    weight_norm = "l1",
    cell_ids    = None,
    device      = None,
    dtype       = torch.float32,
    skipna      = False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `int` | — | Input Grid4Earth/HEALPix level, integer ≥ 1. Internally, `nside_in = 2**level`. |
| `mode` | `str` | `"smooth"` | Downsampling strategy: `"smooth"`, `"smooth_conservative"`, `"maxpool"` or `"mean"`. |
| `ellipsoid` | `str` | `"WGS84"` | Ellipsoid of the cell centres (third positional argument). |
| `radius_deg` | `float` or `None` | auto | Angular radius of the Gaussian disc (degrees). Default ≈ 3 × pixel size. Used with `"smooth"` and `"smooth_conservative"`. |
| `sigma_deg` | `float` or `None` | auto | Gaussian sigma (degrees). Default = `radius_deg / 2`. Used with `"smooth"` and `"smooth_conservative"`. |
| `weight_norm` | `str` | `"l1"` | Per-output-pixel weight normalisation in `"smooth"` mode: `"l1"` (sum=1, preserves constants), `"l2"` (sum of squares=1, preserves energy), `"none"` (raw Gaussian). |
| `cell_ids` | array-like or `None` | `None` | Input pixel indices (NESTED ordering) for a regional domain. `None` = full sphere. |
| `device` | device or `None` | auto | Torch device. Defaults to CUDA if available, else CPU. |
| `dtype` | `torch.dtype` | `float32` | Dtype for the sparse matrix values; the input is cast to it. |
| `skipna` | `bool` | `False` | Only with `mode="mean"`: average the finite children only (otherwise `ValueError`). |

---

## Forward

```python
y, cell_ids_out = down(x)
```

| Argument | Type | Shape | Description |
|---|---|---|---|
| `x` | numpy or torch | `[N]` or `[B, N]` | Input map(s) at fine resolution `nside_in`. `N` must equal `len(cell_ids)` on a regional domain, or `12 * nside_in²` otherwise. |

| Return | Type | Shape | Description |
|---|---|---|---|
| `y` | same as `x` | `[N_out]` or `[B, N_out]` | Downsampled map(s) at coarse resolution `nside_out = nside_in // 2`. |
| `cell_ids_out` | `np.ndarray` | `[N_out]` | NESTED pixel indices of the output pixels at `nside_out`. |

---

## Modes

### `"smooth"` — Gaussian-weighted average

The output for coarse pixel `p_out` is:

$$y_{p_{\text{out}}} = \sum_{p_{\text{in}} \in \text{disc}} M_{p_{\text{out}}, p_{\text{in}}} \cdot x_{p_{\text{in}}}$$

where the weights are:

$$M_{p_{\text{out}}, p_{\text{in}}} \propto \exp\!\left(-\frac{\gamma^2}{2\sigma^2}\right) \cdot \mathbb{1}[\gamma \leq r]$$

and `γ` is the haversine angular distance between the centers of the
fine and coarse pixels.

The sparse matrix `M` is built once at construction and stored as a
buffer — `forward` only applies it.

**`weight_norm` options:**

| Value | Effect |
|---|---|
| `"l1"` *(default)* | Sum of weights = 1. A constant input map returns a constant output. Recommended for standard downsampling. |
| `"l2"` | Sum of squares = 1. Preserves local signal energy rather than amplitude. |
| `"none"` | Raw Gaussian weights (sum < 1 near boundaries). |

### `"smooth_conservative"` — Gaussian average with the mean conserved

The rows of the `"smooth"` matrix sum to 1 (a constant field is preserved),
but its column sums are not uniform (4 × column sum is between 0.90 and
1.08), so every step moves the global mean and the drift accumulates over a
chain of levels. HEALPix cells have equal area, so conserving the global mean
means conserving the plain mean. `"smooth_conservative"` does it in the
simplest way:

$$y = M x + \delta, \qquad \delta = \overline{x} - \overline{M x}$$

with one $\delta$ per input map (per row of a batch), added to every output
cell. The smoothing, the weights and the way information crosses parent
boundaries are those of `"smooth"`; the output differs from `"smooth"` by a
uniform offset (−1e-4 K at level 8 → 7 and −0.30 K at level 1 → 0 on the test
field below). The offset is linear in `x`, and a constant field is preserved
(`δ` is zero up to round-off). It is not local: a cell far from where the
drift comes from is shifted as much as any other. `weight_norm` must be
`"l1"`.

- **Regional domain:** the means are weighted by `coverage` (the fraction of the 4
  children present in `cell_ids`), as for `"mean"`: the coverage-weighted mean
  of the output equals the mean of the input. As for `"mean"`, this holds for
  one step; chaining operators over incomplete parents gives a mean of means.
- **NaN:** `δ` is built from finite values only, so a NaN stays where
  `"smooth"` puts it (it reaches the output cells whose disc contains it) and
  does not spread through `δ`. The finite output cells then have the same
  mean as the finite input cells (to round-off, coverage-weighted on a regional
  domain), but they are not the same cells; the conservation of the mean of the
  whole map is a statement for NaN-free input. `skipna` is not available.
- **Positivity:** the correction is one uniform shift. A negative `δ` can make
  a non-negative field (a probability, say) slightly negative, so positivity
  is not kept.
- `sparse_matrix()` returns the Gaussian matrix `M`, without the offset (the
  offset is a dense rank-one term). `HealPixUp(paired_down=...)` therefore
  pairs with `M`.
- With `HealPixResampler` and a NaN-free map the mean is conserved. With
  missing values the resampler renormalises by the coverage and the mean is
  not conserved.

### Which mode conserves what

Synthetic temperature-like field (288 − 45 sin²(lat) + zonal terms), level 8
to level 0, float64, `ellipsoid="sphere"`, one CPU thread
(`benchmarks/benchmark_smooth_conservative.py`):

| | `"smooth"` | `"smooth_conservative"` | `"mean"` |
|---|---|---|---|
| Drift of the global mean, level 8 → 0 | +0.48 K | 6e-14 K | 0 |
| Constant field preserved | yes | yes | yes |
| Coarse cells reached by a fine cell (level 5 → 4; min / median / max over all fine cells) | 5 / 7 / 9 | 5 / 7 / 9 | 1 |
| Regional domain | yes | yes (coverage-weighted mean) | yes |
| Build the level-8 operator | 1.5 s | 1.6 s | 0.003 s |
| Apply, level 8 → 7 | 56 ms | 54 ms | a reshape and a mean |

The extra work in `forward` is two sums over the input and the output.
Real data were not tested.

### `"maxpool"` — Maximum over NESTED children

In NESTED ordering, the 4 children of coarse pixel `p` are always
`4p, 4p+1, 4p+2, 4p+3`.  The output is:

$$y_{p_{\text{out}}} = \max(x_{4p},\; x_{4p+1},\; x_{4p+2},\; x_{4p+3})$$

The gradient goes to the largest child (differentiable almost everywhere).
NaN propagates.

```{warning}
**Known issue on regional domains.** When a parent has fewer than four
children in `cell_ids`, the missing children are replaced by the value of the
first input cell (`cell_ids[0]`), which may not be a child of that parent at
all. Measured: cells `[100, 4, 5, 6]` at level 2 with values `[50, 1, 2, 3]`
give 50 for parent 1 instead of 3. Until this is fixed, use `maxpool` only
when every parent has its four children, or pad the domain to whole parents.
```

---

## Regional domains

When `cell_ids` is provided, the operator works on a subset of the
sphere.  The coarse output cell ids are automatically derived as the
unique parents of the input pixels in NESTED ordering:

```
cell_ids_out = unique(cell_ids_in // 4)
```

Some coarse pixels at the patch boundary may have fewer than 4 children
available.  In `"smooth"` mode, the Gaussian weights of absent fine
pixels are set to zero and the remaining weights are renormalized.

---

## Examples

### Full sphere

```python
import numpy as np
from healpix_analyse.down import HealPixDown

level = 6
nside = 2**level
down  = HealPixDown(level=level)                         # "smooth", l1

# Single map
x = np.random.randn(12 * nside**2)                      # [N]
y, ids_out = down(x)
print(y.shape)       # (12288,) = 12 * 32²

# Batch
x_batch = np.random.randn(8, 12 * nside**2)             # [B, N]
y_batch, _ = down(x_batch)
print(y_batch.shape) # (8, 12288)
```

### Regional domain

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse.down import HealPixDown

level = 6                       # nside = 2**6
# the NESTED cells within 20 degrees of (lon, lat) = (0, 0)
patch = nested.cone_coverage((0.0, 0.0), 20.0, level)
patch = np.asarray(getattr(patch, "data", patch), dtype=np.int64).reshape(-1)

down = HealPixDown(level=level, cell_ids=patch)

x_patch = np.random.randn(len(patch))
y_patch, coarse_ids = down(x_patch)
print(f"Fine   pixels: {len(patch)}")
print(f"Coarse pixels: {len(coarse_ids)}")    # ≈ len(patch) / 4
```

### Torch tensor, GPU

```python
import torch
from healpix_analyse.down import HealPixDown

level = 7
nside = 2**level
down  = HealPixDown(level=level, device="cuda", dtype=torch.float32)

x = torch.randn(4, 12 * nside**2, device="cuda")        # [B, N]
y, ids_out = down(x)
print(y.shape, y.device)   # (4, 49152) cuda:0
```

### Maxpool mode

```python
from healpix_analyse.down import HealPixDown
import numpy as np

down_max = HealPixDown(level=6, mode="maxpool")
x = np.random.randn(12 * 64**2)
y, _ = down_max(x)
```

### Custom Gaussian scale

```python
# Wider disc, sharper Gaussian — more blurring
down_wide = HealPixDown(
    level=6,
    mode="smooth",
    radius_deg=2.0,
    sigma_deg=0.5,
    weight_norm="l1",
)
```

---

## Use in a U-Net encoder

```python
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown
import torch

level = 6
nside = 2**level
conv1 = HealPixConv(level,     in_channels=1,  out_channels=32, use_norm=True)
down1 = HealPixDown(level,     mode="smooth")
conv2 = HealPixConv(level - 1, in_channels=32, out_channels=64, use_norm=True)
down2 = HealPixDown(level - 1, mode="smooth")

def down_channels(down, f):
    # HealPixDown takes [N] or [B, N]: fold the channels into the batch
    B, C, N = f.shape
    y, _ = down(f.reshape(B * C, N))
    return y.reshape(B, C, -1)

x = torch.randn(4, 1, 12 * nside**2)        # [B, 1, N]
f1 = conv1(x)                               # [B, 32, N]
f1_down = down_channels(down1, f1)          # [B, 32, N/4]
f2 = conv2(f1_down)                         # [B, 64, N/4]
f2_down = down_channels(down2, f2)          # [B, 64, N/16]
```

---

## Properties

| Attribute | Description |
|---|---|
| `level_in` | Fine input Grid4Earth/HEALPix level. |
| `level_out` | Coarse output level (`level_in - 1`). |
| `nside_in` | Fine input resolution. |
| `nside_out` | Coarse output resolution (`nside_in // 2`). |
| `N_in` | Number of input pixels (full sphere or `len(cell_ids)`). |
| `N_out` | Number of output pixels. |
| `mode` | `"smooth"`, `"smooth_conservative"`, `"maxpool"` or `"mean"`. |
| `partial` | `True` when `cell_ids` was provided. |
| `cell_ids_out` | Output pixel indices (coarse NESTED ids). |
| `ellipsoid`, `skipna` | As given to the constructor. |
| `coverage` | Modes `"mean"` and `"smooth_conservative"`: fraction of each parent covered by the domain. |
| `valid_fraction(x)` | Mode `"mean"`: fraction of finite children per parent. |
| `sparse_matrix()` | The sparse matrix of the `"smooth"`, `"smooth_conservative"` (Gaussian part) and `"mean"` modes; `"maxpool"` raises. |

---

## Pairing with `HealPixUp`

`HealPixDown(mode="smooth")` and `HealPixUp` are designed as a matched
pair.  The upsampling operator is the **adjoint (transpose)** of the
smooth downsampling matrix, with the same `weight_norm`, `radius_deg`,
and `sigma_deg`.  Use the same values in both constructors:

```python
from healpix_analyse.down import HealPixDown
from healpix_analyse.up   import HealPixUp

level = 6
nside = 2**level
down = HealPixDown(level=level, weight_norm="l1")
up   = HealPixUp(level=level - 1, weight_norm="l1", up_norm="col_l1")

x = torch.randn(4, 12 * nside**2)
y, _  = down(x)          # [4, N/4]
x_up, _ = up(y)          # [4, N]  ← approximate round-trip
```
