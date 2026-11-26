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
  level, `HealPixUp` gives back the same cells, and a reshape groups the
  cells of one parent into a token (see {doc}`recipes/mae`).
- **Edges.** `HealPixConv` completes stencil points outside the domain with
  the value of the centre (`missing="centre"`, the default) or with the local
  mean (`missing="local_mean"`); `HealPixDown` / `HealPixUp` in the Gaussian
  modes see only the cells of the domain. Add a halo of cells around the region of
  interest when the edge matters, and leave it out of the loss.

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

No HEALPix augmentation module is merged yet (batched geometric
augmentations on the GPU are in review,
[#111](https://github.com/GRID4EARTH/healpix-analyse/pull/111)). Meanwhile,
augment in value space (noise, channel dropout, random NaN masks such as
synthetic clouds), or draw random token masks as in the
{doc}`MAE recipe <recipes/mae>`.

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
