# Recipe: ResNet block and a small classifier

A pre-activation residual block, `x + conv(act(norm(conv(act(norm(x))))))`,
where the convolutions are `HealPixConv` (3×3 stencil, one gauge, so
`out_channels` channels) and the norms are masked `HealPixGroupNorm`. A
classifier stacks a stem, a block, a `HealPixDown(mode="mean")` pooling to
the next level, a second block and a masked `global_pool` readout.

Points to copy:

- the regional domain is built from complete parents, so the pooled level
  has exactly `N / 4` cells, `np.unique(cell_ids // 4)`;
- `HealPixDown` takes `[B, C, N]` (any leading dimensions) and returns
  `(values, cell_ids_out)`; the same operator pools the validity mask;
- NaN never enters a layer: zero-fill plus a mask channel, then the mask goes
  to every norm and to the readout.

```{literalinclude} resnet.py
:language: python
```
