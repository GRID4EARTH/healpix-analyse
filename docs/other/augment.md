# Batched data augmentation on the GPU

**Module** `healpix_analyse.augment`
**Class** `HealPixAugment` (a `torch.nn.Module`, Compose-like)

```{include} ../_generated/glance/augment.md
```

`HealPixAugment` applies a list of transforms to a **whole batch at once**,
with **its own random parameters for every sample**, drawn from a seeded
`torch.Generator`. Everything runs on the device of the data and is
differentiable with respect to the data. It works on a regional domain (a set
of NESTED `cell_ids`, in any order) or on the full sphere (`cell_ids=None`).

```python
import torch
from healpix_analyse import (HealPixAugment, RandomShift, RandomRotation,
                             RandomZoom, RandomFlip, AdditiveNoise, RandomMask)

aug = HealPixAugment(
    [RandomShift(0.02),            # move the domain by up to 0.02 rad (~127 km)
     RandomRotation(),             # any angle about the domain centre
     RandomZoom((0.8, 1.25)),      # homothety, log-uniform per sample
     RandomFlip(),                 # east-west mirror, p = 0.5
     AdditiveNoise((0.0, 0.05)),   # per-sample noise level
     RandomMask(0.75, block=64)],  # MAE: mask 75 % of the 64-cell NESTED tokens
    level=10, cell_ids=cell_ids,
).to("cuda")

g = torch.Generator().manual_seed(0)
out = aug(x, labels=land_mask, generator=g)   # x: [B, C, N]
out.x, out.valid, out.mask, out.labels, out.ids_keep, out.ids_restore
```

## Transforms

| Transform | What it does | Kind |
|---|---|---|
| `RandomRotation(angles=pi)` | rotation of a regional domain about its centre, any angle | tangent plane |
| `RandomZoom(scales=(0.8, 1.25))` | homothety of factor *s* about the domain centre (*s* > 1 magnifies) | tangent plane |
| `RandomFlip(p=0.5, axis="ew")` | mirror of the local frame (full sphere: longitude → −longitude) | tangent plane / 3-D |
| `TangentLinear(A)` | fixed 2×2 map of the tangent plane | tangent plane |
| `RandomShift(max_distance)` | translation on the sphere: a rotation about an axis through the centre of the Earth | 3-D |
| `RandomSO3()` | uniform global rotation of a full-sphere field | 3-D |
| `Rotation3D(R)` | fixed 3×3 orthogonal map | 3-D |
| `RandomGaugeRotation(n_gauges, flip=0)` | cyclic shift (and reflection) of the G gauge axis of `HealPixConv` outputs | channels |
| `AdditiveNoise(std)` | Gaussian noise, per-sample level | values |
| `RandomMask(ratio, block=1, fill=0)` | random cell (`block=1`) or NESTED-token masking, MAE style | values |
| `RandomCrop(n_cells)` | random sub-domain of `4**k` cells: the children of one cell of level L−k, a contiguous NESTED slice | support |

`gauge_rotate(x, shifts, n_gauges, flip=None)` is the functional form of the
gauge shift.

## How it works: one gather per fused group

Each geometric transform is a *source map*: the output at cell *p* reads the
input at *S(p)*. 3-D maps are `S(p) = R p`. Tangent-plane maps are
`S(p) = lift(A proj(p))` in the gnomonic projection about the domain centre.
A run of geometric transforms whose 3-D maps come before its tangent-plane
maps (shift, then rotation, zoom and flip, for instance) composes **exactly**
into `S(p) = R lift(A proj(p))`, one `R` and one `A` per sample. It is then
applied as **one gather**: 4 bilinear weights per output cell and per sampling
tap.

- **On CUDA** (with the optional `triton` package, float32 / float16 /
  bfloat16 data), one Triton kernel does all of it in a single pass, for each
  output cell of each sample:
  - the source point from the per-sample `R` and `A`;
  - the HEALPix bilinear interpolation (the `healpy` algorithm of
    `get_interp_weights`, transcribed);
  - the binary search of the 4 cells among the cells of the domain;
  - the weighted gather of the channels, accumulated in fp32.

  No index or weight map is stored, and no geometry is rebuilt on the CPU for
  a new draw. The backward pass is a second kernel that recomputes the same
  weights and scatters the gradient into an **int64 fixed-point**
  accumulator. Integer atomics give the same sum in any order, so the
  gradient is deterministic, without sorting a transposed map. Tiles are
  autotuned.
- **On the CPU** (and for float64), the same weights come from torch ops,
  followed by a `torch.gather` loop. This is also the reference that the
  tests compare the kernels to.

A 3-D map after a tangent-plane map starts a new gather. The channel, value
and support transforms run as cheap tensor operations between the gathers.
`RandomGaugeRotation` commutes with the gathers.

## Zooming out: anti-aliasing

For *s* < 1 the input must be low-passed first. Two filters are available
with `antialias=`:

- `"supersample"` (default): every output cell averages *n*×*n* bilinear
  samples spread over its footprint in the input, with *n* = ⌈1/*s*min⌉ for
  the batch. This is a box filter the size of the output cell, exact for any
  *s*. The gather then has 4*n*² taps.
- `"pyramid"`: the input is averaged over NESTED children into levels L−1 …
  L−k (a change of level, like `HealPixDown` `mean`). Each sample then reads
  level L − round(log2(1/*s*)) with 4 taps (mip-mapping, exact for
  *s* = 2^−k). It is cheaper for strong zoom-outs.

## Outside the domain

An output cell whose source point falls outside the regional domain gets
`fill` (NaN by default) and `valid = False` in the result. By default
(`min_support=1`), a cell counts as inside when every bilinear neighbour with
a non-zero weight is a cell of the domain. `min_support=0.5` keeps cells with
at least half their weight in the domain and renormalises their weights. The
valid area then grows by about half a cell, with a first-order error at the
border. A NaN in the input spreads to every output cell that gives it a
non-zero weight.

## Labels and masks

`labels=` (a tensor, a dict, a list or a tuple of tensors) is transformed with
the **same draw**, using **nearest neighbour**. Outside the domain, integer
labels get `label_fill` (−1 by default), boolean masks get `False` and float
labels get NaN. To transform a target field bilinearly with the same draw,
draw the plan once and apply it to both:

```python
plan = aug.sample(B, generator=g)
xa = plan.apply(x).x
ya = plan.apply(y).x          # same rotation, shift, zoom, noise and mask
```

## Determinism and accuracy

- The same seed gives the same plan on any device. The draws use the
  generator; the noise is drawn on the device of the data, from a seed taken
  from the generator.
- With the float64 geometry (`geom_dtype`, the default), the identity
  transform is exact, bit for bit, and the fused CUDA kernel gives exactly the
  same output as the torch reference.
- `geom_dtype=torch.float32` is opt-in. Near a cell or face boundary, its
  rounding (about 1e-7 rad) can pick a different bilinear neighbour than
  float64 does, or than the same transform does on another device. The
  swapped neighbour carries a small weight, but that weight grows with the
  level, because the cells shrink while the rounding stays the same: about
  1.5×10⁻⁴ at level 8, 2×10⁻⁴ at level 10 and up to 2.2×10⁻³ at level 12.
  In the benchmark cases, 0.36–0.68 % of the output cells are concerned, and
  the validity mask differs at 0–4 cells in 3×10⁵. From level 10 on,
  float32 also makes the identity inexact: rounding can flag border cells as
  outside the domain. `neighbour_mismatch(map_a, map_b)` measures this for a
  given draw.
- A rotation by 0.7 rad followed by its inverse (two gathers) brings back a
  field that varies over the scale of the domain to 6×10⁻⁵ of its range.
- An SO(3) rotation of a full-sphere ℓ ≤ 3 field at level 6 preserves its
  mean and power to 10⁻³ and matches the analytic rotation to 2×10⁻³.
- The fused Triton kernels match the torch reference to 10⁻⁵ in float32
  (`tests/test_augment.py`).

## Performance

`benchmarks/benchmark_augment.py` measures the throughput in samples per
second for a batch of 64 regional domains of 65 536 cells. It reports the
geometry alone, the forward, the forward + backward, and the bandwidth of the
fused forward against the GPU's peak. The numbers are in the pull request
that added the module.
