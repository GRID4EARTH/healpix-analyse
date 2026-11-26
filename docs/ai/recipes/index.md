# Recipes

Short, complete PyTorch models built only from the {doc}`blocks <../index>`
of `healpix-analyse` and `torch.nn`. Each page shows one script; the same
scripts run in the test suite (`tests/test_ai_recipes.py`) on the CPU at a
small level, in a few seconds each, so the code shown here runs as it is.
Copy one, then raise `level` and the widths, and move to the GPU (`DEVICE`
picks CUDA when it is available).

| Recipe | Model | Blocks |
|---|---|---|
| {doc}`resnet` | pre-activation ResNet block, small classifier with a masked readout | `HealPixConv`, `HealPixGroupNorm`, `HealPixDown`, `global_pool` |
| {doc}`unet` | U-Net with skip connections on a regional domain with gaps | `HealPixConv`, `HealPixGroupNorm`, `HealPixDown`, `HealPixUp` |
| {doc}`mae` | masked autoencoder (ViT / MAE): patch tokens, masked token sampling, fold decoder | `HealPixPatches`, `HealPixPositionalEncoding`, `torch.nn.TransformerEncoder` |
| {doc}`training` | training loop: NaN masks, masked loss, bf16 autocast, `DataLoader` | `HealPixConv`, `HealPixGroupNorm`, `HealPixUp` |
| {doc}`swin` | Swin-like hierarchical transformer (window attention, patch merging, masked readout) | `HealPixWindowAttention`, `HealPixLayerNorm`, `HealPixDown`, `global_pool` |
| {doc}`neighbourhood_attention` | neighbourhood-attention transformer for per-cell segmentation | `HealPixNeighbourhoodAttention`, `HealPixLayerNorm`, `HealPixConv` |

Recipes for the blocks still in review (strided and transposed convolutions,
augmentations) are added when they are merged.

```{toctree}
---
maxdepth: 1
hidden: true
---
resnet
unet
mae
training
swin
neighbourhood_attention
```
