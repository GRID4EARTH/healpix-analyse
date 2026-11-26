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
| {doc}`unet` | U-Net with skip connections on a regional domain with gaps; strided conv / transpose or Down / Up | `HealPixConv(stride=2)`, `HealPixConvTranspose`, `HealPixGroupNorm`, `HealPixDown`, `HealPixUp` |
| {doc}`mae` | masked autoencoder (ViT / MAE): patch tokens, masked token sampling, fold decoder | `HealPixPatches`, `HealPixPositionalEncoding`, `torch.nn.TransformerEncoder` |
| {doc}`training` | training loop: batched augmentation (shift, rotation, zoom, flip, masking), NaN masks, masked loss, bf16 autocast | `HealPixAugment`, `HealPixConv`, `HealPixGroupNorm`, `HealPixUp` |
| {doc}`gradient_penalty` | GAN critic with a gradient penalty (double backward) | `HealPixConv`, `HealPixConv(stride=2)`, `global_pool` |
| {doc}`swin` | Swin-like hierarchical transformer (window attention, patch merging, masked readout) | `HealPixWindowAttention`, `HealPixLayerNorm`, `HealPixDown`, `global_pool` |
| {doc}`neighbourhood_attention` | neighbourhood-attention transformer for per-cell segmentation | `HealPixNeighbourhoodAttention`, `HealPixLayerNorm`, `HealPixConv` |

```{toctree}
---
maxdepth: 1
hidden: true
---
resnet
unet
mae
training
gradient_penalty
swin
neighbourhood_attention
```
