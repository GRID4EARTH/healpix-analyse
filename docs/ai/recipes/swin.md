# Recipe: Swin-like hierarchical transformer with window attention

A Swin-style model on a regional domain: a `HealPixConv` embedding, then
stages of pre-norm transformer blocks, `x + attn(norm(x))` and
`x + mlp(norm(x))`, where `attn` is `HealPixWindowAttention`. Between
stages, `HealPixDown(mode="mean")` goes one level coarser and a pointwise
linear layer widens the channels (Swin's patch merging). A masked
`global_pool` gives one vector per sample for the head.

Points to copy:

- `HealPixWindowAttention` is the attention only (qkv and output
  projections, `[B, C, N] -> [B, C, N]`): the residuals, the norms and the
  MLP are yours, as in any transformer block;
- with the default `shift="coarse_centred"` the keys of a window are the
  cells within a disc around its block, so windows overlap across block
  borders and the receptive field grows at every layer: no shifted-window
  alternation is needed (`shift="alternate_level"` with `shifted=True` on
  every second block is the closest to Swin; see {doc}`/conv/attention`);
- windows are NESTED blocks of `4**window_level` cells: use the full sphere
  or a sorted domain made of complete blocks, so no padding is needed;
- each stage builds its layers with the cell ids of its level,
  `np.unique(ids // 4)`.

```{literalinclude} swin.py
:language: python
```
