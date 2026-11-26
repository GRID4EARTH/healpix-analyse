# Recipe: training loop with NaN masks and bf16 autocast

A small denoiser trained on synthetic data: smooth fields (two `HealPixUp`
steps of coarse noise) observed with noise and NaN clouds. The loop is plain
PyTorch: a `DataLoader` over stacked `[B, C, N]` tensors (every sample on
the same cells), `torch.autocast` in bfloat16, AdamW, a masked mean squared
error.

Points to copy:

- the model zero-fills NaN and adds a validity channel, and the masked norm
  gets the mask;
- the loss is a weighted mean over valid target cells;
- inside `torch.autocast`, `torch.nn.Linear` runs in bfloat16, while
  `HealPixConv` casts its input to its own dtype (float32 here) and the
  masked norm keeps the dtype of its input; bfloat16 needs no `GradScaler`.

```{literalinclude} training.py
:language: python
```
