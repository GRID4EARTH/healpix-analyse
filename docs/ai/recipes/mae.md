# Recipe: masked autoencoder (ViT / MAE) with patch tokens

`HealPixPatches` cuts a field `[B, C, K]` into `M` square `N × N` patches
centred on the cells of level `level − log2(N)`, oriented by the gauge, so a
shared patch embedding sees the same orientation everywhere on the sphere.
Each token gets the `HealPixPositionalEncoding` of its centre cell
(`patches.centre_ids` at `patches.centre_level`). The encoder sees the kept
tokens only, gathered with `patches(x, index=keep)` (one draw per sample,
nothing computed for the masked patches). The decoder puts a learned mask
token at the masked places, predicts every patch, and `patches.fold` brings
the prediction back onto the cells. The loss counts the masked tokens and
the valid pixels only.

Points to copy:

- the output `[B, M, G, C, N, N]` is contiguous: `y.flatten(2)` is one token
  of `G·C·N²` values per centre, a free view;
- NaN propagates through the extraction: zero-fill, extract a validity
  channel with the values, and weight the loss with the patches of the mask;
- `mode="nested"` (NESTED tiles, face-oriented) is the cheap alternative:
  on the full sphere or a sorted domain of complete coarse cells it is a view
  plus one gather, and `fold` inverts it exactly;
- with `centres="coarse"` and `mode="gauge"` the square grids do not tile the
  diamond-shaped cells: `fold` misses up to 16 cells next to the poles at
  `N = 4`, and the tips of the diamonds from `N = 8` (see
  {doc}`/conv/patches`);
- `patches.fold(y, index=...)` folds only the patches listed, for instance
  the 25 % of kept tokens of an MAE; on CUDA, with patches of at least 64
  pixels (`N ≥ 8`) and at most 37.5 % of the centres selected, the Triton
  fold then visits only the cells that those patches read, with the same
  result bit for bit (see {doc}`/conv/patches`). Above 37.5 % (the 75 % of
  masked tokens, say) it walks every cell. The recipe folds every
  predicted patch;
- for a ViT classifier, drop the masking and the decoder and pool the
  encoded tokens (mean over `M`);
- for a JEPA-style model, keep this encoder as the context encoder, an EMA
  copy as the target encoder, and query a predictor with the positional
  encoding of the masked centres.

```{literalinclude} mae.py
:language: python
```
