# Recipe: U-Net on a regional domain with gaps

Encoder levels `level, level-1, level-2`, each a `HealPixConv` +
`HealPixGroupNorm` + GELU. Between levels the encoder goes one level coarser
with a learned strided convolution, `HealPixConv(stride=2)`, and the decoder
comes back with `HealPixConvTranspose`, the adjoint of that strided layer,
then concatenates the skip connection of the same level and applies a
`HealPixConv` + norm. With `sampling="fixed"` the resampling is
`HealPixDown(mode="mean")` and `HealPixUp` paired with it (every child gets
its parent's value) instead. The validity mask is pooled with
`HealPixDown` on the way down and taken back from the encoder on the way up.

Points to copy:

- one set of cell ids per level, `ids[i + 1] = np.unique(ids[i] // 4)`, and
  every layer of a level built with the ids of that level; a strided layer
  writes on `conv.cell_ids_out`, the same sorted parents;
- `HealPixConv(level, c, c, stride=2, cell_ids=fine ids)` keeps the stencil
  and spacing of level `level`, evaluated at the parent centres; it replaces
  a convolution followed by `HealPixDown`;
- `HealPixConvTranspose(level, c_in, c_out, cell_ids=fine ids)` reads the
  parents and writes the fine cells; with `n_gauges=G` it reads `G*c_in`
  channels and sums over the gauges;
- `HealPixDown` / `HealPixUp` take `[B, C, N]` and return
  `(values, cell_ids_out)`; `HealPixUp(level, cell_ids=coarse ids)`
  interpolates with the Gaussian transpose instead of copying the parent;
- for a wide receptive field at one level, `LargeConv` replaces a stack of
  3×3 layers.

```{literalinclude} unet.py
:language: python
```
