# Recipe: neighbourhood-attention transformer for per-cell segmentation

A transformer that stays at the input level and predicts a class per cell:
a `HealPixConv` embedding, pre-norm blocks with
`HealPixNeighbourhoodAttention` (each cell attends to the 3×3 taps of its
gauge-oriented `HealPixConv` stencil, with a relative position bias shared
by the whole domain), and a per-cell linear head. Cells under the gap get no
label (`ignore_index`).

Points to copy:

- the layer is the attention only; with `n_gauges=G` its output has `G*C`
  channels, as `HealPixConv` (the recipe keeps `G = 1`);
- on a regional domain, taps fully outside the domain are left out of the
  softmax (`outside="mask"`, the default);
- for a wider context, stack more blocks, raise `kernel_sz`, or combine
  with the window attention of the {doc}`Swin-like recipe <swin>`.

```{literalinclude} neighbourhood_attention.py
:language: python
```
