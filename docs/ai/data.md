# Data for AI models

How HEALPix data reach the blocks of {doc}`index`: the domain, missing
values, augmentation and batching.

## The domain is a set of cell ids, fixed at construction

- **Full sphere**: `cell_ids=None`; the input has `N = 12 * 4**level` cells
  in NESTED order.
- **Regional domain**: pass the NESTED `cell_ids` of the region (a Sentinel-2
  scene, a basin, a coastal strip) to every layer. The geometry of
  `HealPixConv`, `HealPixDown` and `HealPixUp` is built for those cells only;
  the input must have `N = len(cell_ids)` cells, row `k` being `cell_ids[k]`.
- **Levels of a multiscale model.** One level coarser, the cells are
  `np.unique(cell_ids // 4)`: build the layers of each level with the cell
  ids of that level. Build the regional domain from **complete parents**:
  every descendant of a set of coarse cells,

  ```python
  k = 4 ** (level - coarse_level)
  cell_ids = (coarse_ids[:, None] * k + np.arange(k)).ravel()
  ```

  Then `HealPixDown(mode="mean")` is the exact mean of 4 children at every
  level, `HealPixUp` gives back the same cells, and `HealPixPatches(mode=
  "nested")` is a view of the data (see {doc}`recipes/mae`).
- **Edges.** `HealPixConv` completes stencil points outside the domain with
  the value of the centre (`missing="centre"`, the default) or with the local
  mean (`missing="local_mean"`); `HealPixDown` / `HealPixUp` in the Gaussian
  modes see only the cells of the domain. Add a halo of cells around the region of
  interest when the edge matters, and leave it out of the loss.
- **Stencil geometry.** `HealPixConv` (with its strided and transposed
  forms and `missing="local_mean"`), `HealPixNeighbourhoodAttention` and the
  gauge mode of `HealPixPatches` build their stencil in float64 whatever the
  dtype of the layer. A bilinear neighbour inside the domain counts only if
  its weight is at least 1e-6 of the full weight of its stencil point;
  below that the weight is rounding noise. A point left with no neighbour
  that counts falls back to the centre (0 for `HealPixPatches`), as when all
  its neighbours are outside the domain. So the cells a layer reads do not
  depend on rounding.

## Missing values (NaN) and masks

Clouds, land, outside the swath: most blocks let NaN propagate.

1. Build a validity mask, `mask = torch.isfinite(x).all(dim=1)` (`[B, N]`).
2. Zero-fill the input and give the mask to the model as an extra channel:
   `torch.cat([torch.nan_to_num(x), mask[:, None].float()], dim=1)`.
3. Pass the mask to the masked norms (`HealPixGroupNorm(...)(h, mask)`) and
   to the pooling (`global_pool(h, mask)`): invalid cells do not enter the
   statistics, and the norms write 0 there (`fill="zero"`).
4. One level coarser, the mask is `HealPixDown(mode="mean")` of the mask
   (the valid fraction of the 4 children), thresholded or used as a coverage
   in [0, 1] — the masked norms and the pooling accept both.
5. Compute the loss on valid cells only (a weighted mean with the mask).

`HealPixConv(missing="local_mean")` and `HealPixWaveletSpectrum` complete
the missing cells by the local mean instead; this runs the PyTorch
reference path, which keeps the gathered stencil in memory.

## Augmentation

`HealPixAugment` ({doc}`/other/augment`) augments a whole batch at once,
with its own random parameters per sample from a seeded `torch.Generator`,
on the device of the data: `RandomShift`, `RandomRotation`, `RandomZoom`,
`RandomFlip`, `RandomSO3` (full sphere), `AdditiveNoise`, `RandomMask`
(cells or NESTED tokens, MAE style) and `RandomCrop`. Consecutive geometric
transforms are fused into one gather (one Triton kernel on CUDA).

- **Input and target together.** Draw the plan once and apply it to both:
  `plan = aug.sample(B, generator=g)`, then `plan.apply(x).x` and
  `plan.apply(y).x` (bilinear). `aug(x, labels=...)` transforms labels and
  masks with the same draw, by nearest neighbour.
- **Outside the domain.** On a regional domain, an output cell whose source
  falls outside the domain gets `fill` (NaN by default) and `valid=False`:
  it is a gap like any other, left out of the input mask and of the loss.
- **Synthetic gaps.** `RandomMask(ratio, block=..., fill=float("nan"))` on
  the input only turns masked cells or blocks into NaN; the model handles
  them as clouds.
- **Precision of the geometry.** The augmentation geometry is float64 by
  default (`geom_dtype`): the identity is exact, and the CUDA kernel matches
  the CPU reference bit for bit. `geom_dtype=torch.float32` is opt-in:
  faster, but near a cell or face boundary its rounding can pick another
  bilinear neighbour, more often at high levels.

See the {doc}`training recipe <recipes/training>`.

## Batching

- Every sample of a batch must live on the **same cell ids**: a batch is a
  stacked tensor `[B, C, N]`, and a `torch.utils.data.DataLoader` with the
  default collate works.
- **Variable-size batching is not available yet**: there is no padded or
  packed batch of domains of different sizes. Group the samples by domain
  and build the layers of each domain (the learned weights can be shared:
  the weight shape of `HealPixConv` does not depend on the domain), or
  train on the full sphere and mask what is outside the region.
- The geometry is built at construction and cached (`HealPixConv` on disk,
  `HealPixPositionalEncoding` in memory), so building layers for a few
  domains is cheap after the first time.
- Mixed precision: inside `torch.autocast`, `torch.nn.Linear`,
  `HealPixWindowAttention` and `HealPixNeighbourhoodAttention` run in
  bfloat16; `HealPixConv` casts its input to its own dtype (build it
  with `dtype=torch.bfloat16` for 16-bit kernels); the masked norms keep the
  dtype of their input. See {doc}`recipes/training`.
