# Recipe: U-Net on a regional domain with gaps

Encoder levels `level, level-1, level-2`, each a `HealPixConv` +
`HealPixGroupNorm` + GELU, joined by `HealPixDown(mode="mean")`; the decoder
goes back up with `HealPixUp` paired with the Down of the same level
(every child gets its parent's value), concatenates the skip connection of the same
level and applies a `HealPixConv` + norm. The validity mask follows the
same path: pooled with `HealPixDown` on the way down, taken back from the
encoder on the way up.

Points to copy:

- one set of cell ids per level, `ids[i + 1] = np.unique(ids[i] // 4)`, and
  every layer of a level built with the ids of that level;
- `HealPixUp(level=coarse level, paired_down=down)` returns the cells of the
  finer level, the same cells as the encoder; `HealPixUp(level, cell_ids=
  coarse ids)` interpolates with the Gaussian transpose instead (smoother
  upsampling, same cells because the domain is made of complete parents);
- `HealPixDown` and `HealPixUp` take `[B, C, N]` and return
  `(values, cell_ids_out)`;
- for a wide receptive field at one level, `LargeConv` replaces a stack of
  3×3 layers.

```{literalinclude} unet.py
:language: python
```
