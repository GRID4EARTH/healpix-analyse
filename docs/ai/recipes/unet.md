# Recipe: U-Net on a regional domain with gaps

Encoder levels `level, level-1, level-2`, each a `HealPixConv` +
`HealPixGroupNorm` + GELU, joined by `HealPixDown(mode="mean")`; the decoder
goes back up with `HealPixUp`, concatenates the skip connection of the same
level and applies a `HealPixConv` + norm. The validity mask follows the
same path: pooled with `HealPixDown` on the way down, taken back from the
encoder on the way up.

Points to copy:

- one set of cell ids per level, `ids[i + 1] = np.unique(ids[i] // 4)`, and
  every layer of a level built with the ids of that level;
- `HealPixUp(level=coarse level, cell_ids=coarse ids)` returns the cells of
  the finer level (the same cells as the encoder, because the domain is made
  of complete parents);
- for a wide receptive field at one level, `LargeConv` replaces a stack of
  3×3 layers.

```{literalinclude} unet.py
:language: python
```
