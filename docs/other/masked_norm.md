# Masked normalisation and pooling for AI models

**Module** `healpix_analyse.masked_norm`  
**Layers** `HealPixGroupNorm`, `HealPixLayerNorm`, `HealPixBatchNorm`,
`HealPixGlobalPool`, `HealPixRegionPool`; **functions** `global_pool`,
`region_pool`  
**Status** experimental

```{include} ../_generated/glance/masked_norm.md
```

## Why

Earth-observation fields have holes: clouds, land in an ocean product,
cells outside the swath, the padding of a regional domain. `torch.nn`
normalisation and pooling layers treat a NaN or a zero-filled cell as an
ordinary value: one NaN turns a whole group into NaN, and zero padding
biases the mean and the variance. The layers here take a **validity mask or
a coverage** and leave the invalid cells out of the statistics.

HEALPix cells have equal area, so an area-weighted statistic is a plain
weighted statistic whose weight is the mask value of the cell:

```text
S0   = Σ w_i
mean = Σ w_i x_i / S0
var  = Σ w_i (x_i − mean)² / S0                 (biased, as torch.nn)
y_i  = (x_i − mean) / √(var + eps) · γ + β      where w_i > 0
```

## The mask

Every layer takes `forward(x, mask=None)` with `x` of shape `[B, C, N]` (the
full sphere or a regional domain, cells in any order) and a mask of shape

- `[N]`: the same cells for every sample and channel;
- `[B, N]`: one mask per sample;
- `[B, C, N]`: one mask per channel.

The mask is bool (valid / invalid) or a float **coverage** in [0, 1], for
instance the fraction of a coarse cell covered by valid fine cells after a
`mean` Down. A cell of weight 0 < w < 1 is normalised like a valid cell;
its weight only sets its share in the statistics (scaling every weight by
the same factor changes nothing).

- `mask=None`: the invalid cells are the NaN (and ±inf) values of `x`.
- With a mask, non-finite values of `x` are invalid as well.
- The value of an invalid cell is never read: a NaN there reaches neither
  the output nor the gradients.
- The mask is not differentiated.

## What a masked cell outputs: `fill`

| `fill` | Output at a cell of weight 0 | Gradient there | When |
|---|---|---|---|
| `"zero"` (default) | 0 | 0 | before a convolution or a loss: safe, no NaN |
| `"nan"` | NaN | 0 | to hand the hole to the next masked layer, e.g. `HealPixConv(missing="local_mean")` |
| `"passthrough"` | the input value (NaN included) | 1 | residual branches that must keep the input untouched |

A whole group (or channel, or cell for LayerNorm) without any valid cell
outputs `fill` everywhere and never NaN.

## The layers

```python
import torch
from healpix_analyse import HealPixGroupNorm, HealPixLayerNorm, HealPixBatchNorm

x = torch.randn(8, 64, 12 * 4**7)          # full sphere, level 7
valid = torch.rand(8, 12 * 4**7) > 0.3     # e.g. cloud-free cells, per sample

gn = HealPixGroupNorm(32, 64)              # as torch.nn.GroupNorm(32, 64)
y = gn(x, valid)

ln = HealPixLayerNorm(64)                  # over the channels of each cell
bn = HealPixBatchNorm(64)                  # running statistics, as BatchNorm1d
```

`HealPixGroupNorm`
: Statistics per sample and group of `C / num_groups` channels, over the
  valid cells. Same parameters and `state_dict` as `torch.nn.GroupNorm`.

`HealPixLayerNorm`
: Each cell (token) of each sample is normalised over its `C` channels, as
  `torch.nn.LayerNorm(C)` on `x.transpose(1, 2)` (the transformer / ConvNeXt
  convention). An `[N]` or `[B, N]` mask makes whole cells valid or not; a
  `[B, C, N]` mask, or NaN in some channels only, weights the channels
  inside a cell. For a normalisation over the channels *and* the cells of a
  sample use `HealPixGroupNorm(1, C)`.

`HealPixBatchNorm`
: Statistics per channel over the samples and the valid cells. The running
  mean and variance are updated with the weighted mean and the unbiased
  variance `var · S0 / (S0 − 1)` (`S0` = number of valid cells for a binary
  mask); a channel with `S0 ≤ 1` in a batch keeps its running statistics.
  `momentum=None` (cumulative average), `track_running_stats` and the
  `state_dict` behave as in `torch.nn.BatchNorm1d`. In evaluation the running
  statistics are applied cell by cell.

With an all-valid mask (or `mask=None` and no NaN), each layer gives the
output and the gradients of its `torch.nn` equivalent; the tests check it to
1e-12 in float64 and 1e-5 in float32.

## Backends and gradients

`backend="auto"` (default) picks:

- **`"triton"`** on CUDA for float32, float16 and bfloat16, when the
  optional `triton` package is installed. Fused kernels, forward and
  backward; 16-bit data accumulate in float32.
- **`"torch"`** everywhere else (CPU, float64, CUDA without Triton): a few
  PyTorch passes (validity, row sums, one fused sum of squares, the output)
  and the same hand-written backward. Nothing of size `[B, C, N]` is saved
  beyond the input.

The Triton kernels and their memory traffic (T = one pass over `x`):

| Layer | Forward | Backward |
|---|---|---|
| GroupNorm, BatchNorm (training) | statistics in **one read** (a weighted Welford update per lane, partial sums of a fixed chunking merged with Chan's formula), then normalisation, affine step and `fill` in one pass: 3 T | sums of `dy` and `dy·x̂` per chunk, then `dx`: 5 T |
| LayerNorm, C ≤ 512 | one `[C, BLOCK_N]` tile per program, held in registers: **2 T** | one tile of `x` and `dy`: **3 T** |
| LayerNorm, C > 512 | loop over the channels: 4 T | 5 T |
| BatchNorm (evaluation) | one element-wise pass: 2 T | 3 T (sums only for `γ`, `β`) |

`torch.nn.GroupNorm` and `BatchNorm` also read `x` twice in the forward.
The mask is read through broadcast strides: an `[N]` or `[B, N]` mask is
never expanded to `[B, C, N]`. No kernel uses atomics; the reductions keep
a fixed tile and a fixed chunking (only the element-wise kernels autotune
their tile), so outputs and gradients are bitwise reproducible.

The backward formula, for the cells of weight w > 0 and `g = ∂L/∂y · γ`:

```text
∂L/∂x_i = rstd · (g_i − w_i (A + x̂_i B)),   A = Σ g / S0,   B = Σ g x̂ / S0
```

The gradients are exact: `torch.autograd.gradcheck` passes for every layer
with a fractional coverage. A second derivative (`create_graph=True`, e.g.
a gradient penalty) runs through `masked_norm.masked_norm_reference`, the
plain differentiable PyTorch reference of the tests, so `gradgradcheck`
passes as well.

## Pooling

```python
from healpix_analyse import global_pool, region_pool, HealPixRegionPool

feat = global_pool(x, valid)                       # [B, C], area-weighted mean
peak = global_pool(x, valid, mode="max")           # max over the valid cells
y, n = global_pool(x, valid, return_weight=True)   # n = covered cells per (b, c)

# regions = parent cells two levels up (any NESTED domain, sorted or not)
pool = HealPixRegionPool.from_cell_ids(cell_ids, level=7, region_level=5)
z = pool(x, valid)                                 # [B, C, R]; pool.region_ids: parent ids
```

`mode` is `"mean"` (`Σ w x / Σ w`), `"sum"` (`Σ w x`) or `"max"` (over the
cells of weight > 0). A sample, channel or region without any valid cell
gets `fill="zero"` (default) or `"nan"`.

`region_pool` takes any integer labels (`-1` leaves a cell out): basins,
countries, land-cover classes. Neither backend uses atomics, so the
results are deterministic.

- **Triton** (CUDA): `global_pool` reads `x` once (partial sums per chunk,
  as the normalisations); its backward is one element-wise pass. Regions are
  stored in CSR form (int32 cell positions sorted by region, int32
  offsets); one program per (region, block of rows) gathers the cells of its
  region, and the backward gathers the `[B, C, R]` gradient by the label of
  each cell. `"max"` stores the index of the maximum (the first one on ties)
  and its backward writes one value per (b, c, region).
- **PyTorch**: the cells of each region are gathered into a padded
  `[B, C, R, L]` tensor (L = size of the largest region) and reduced;
  regions of very different sizes waste memory in the padding. When the
  regions are contiguous, sorted and of equal size (`from_cell_ids` on the
  full sphere or on a complete sorted domain) the gather is a free view.

For one level coarser on every cell, keeping a HEALPix grid, use
{doc}`HealPixDown <down>` instead.

## Cost

`benchmarks/benchmark_layers.py` measures forward, forward + backward and
peak memory against the unmasked `torch.nn` layers; the tables of the pull
request that added the module (#109, section 5) give A100 and CPU numbers.
