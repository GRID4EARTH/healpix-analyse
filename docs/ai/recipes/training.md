# Recipe: training loop with augmentation, NaN masks and bf16 autocast

A small denoiser trained on synthetic data: smooth fields (two `HealPixUp`
steps of coarse noise) observed with noise and NaN clouds. The loop is plain
PyTorch: a `DataLoader` over stacked `[B, C, N]` tensors (every sample on
the same cells), batched augmentation with `HealPixAugment`,
`torch.autocast` in bfloat16, AdamW, a masked mean squared error.

Points to copy:

- one geometric plan per batch (`geom.sample(B, generator=g)`: shift,
  rotation, zoom and mirror, own parameters per sample), applied to the
  input and to the target with `plan.apply`, so both move together;
- `RandomMask(0.2, block=16, fill=float("nan"))` on the input only adds
  synthetic gaps; cells read outside the domain are NaN with
  `valid=False`; both become missing values for the model and the loss;
- the augmentation geometry is float64 by default; `geom_dtype=torch.float32`
  is opt-in (faster, can pick another bilinear neighbour near cell
  boundaries);

- the model zero-fills NaN and adds a validity channel, and the masked norm
  gets the mask;
- the loss is a weighted mean over valid target cells;
- inside `torch.autocast`, `torch.nn.Linear` runs in bfloat16, while
  `HealPixConv` casts its input to its own dtype (float32 here) and the
  masked norm keeps the dtype of its input; bfloat16 needs no `GradScaler`.

```{literalinclude} training.py
:language: python
```
