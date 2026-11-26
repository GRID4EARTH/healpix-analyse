# Recipe: masked autoencoder (ViT / MAE) with NESTED-block tokens

In NESTED order, the `4**k` cells of one parent cell at level `level - k`
are consecutive. A reshape turns a field `[B, C, N]` into `T = N / 4**k`
tokens of `C * 4**k` values, and back: no gather, no geometry. Each token
gets the `HealPixPositionalEncoding` of its parent cell (`pe().T`,
`[T, F]`), projected by a linear layer. The encoder sees the visible tokens
only; the decoder gets the encoded tokens back at their place, a learned
mask token elsewhere, and predicts every cell of every token. The loss
counts the masked tokens and the valid cells only.

Points to copy:

- tokens need a sorted domain of complete parents (the full sphere, or every
  descendant of a set of coarse cells);
- NaN cells are zero-filled, a validity channel goes into the token, and
  the loss weight excludes them;
- for a ViT classifier, drop the masking and the decoder and pool the
  encoded tokens (mean over `T`);
- the same encoder, an EMA copy as target and a predictor queried with the
  positional encoding of the masked tokens give a JEPA-style model.

NESTED blocks are diamonds of the HEALPix grid, not squares in a local
frame; square, oriented patches (`HealPixPatches`) are in review
([#113](https://github.com/GRID4EARTH/healpix-analyse/pull/113)).

```{literalinclude} mae.py
:language: python
```
