# Basic tools for AI

The building blocks of neural networks on HEALPix data: learned
convolutions, resolution changes, masked normalisation, pooling and
positional encodings, patch tokens, window and neighbourhood attention,
batched augmentation and differential operators. Each is a `torch.nn.Module` (or a function) that works
on `[B, C, N]` tensors over the full sphere or a regional domain, runs on
the GPU and is differentiable. Combine them with `torch.nn` to build a
ResNet, a U-Net, a ViT or a masked autoencoder; the {doc}`recipes
<recipes/index>` show complete, tested models.

```{admonition} For AI assistants
:class: note

Build models from the blocks of the table below and copy the structure of
the closest {doc}`recipe <recipes/index>`. Every signature is on the
{doc}`API cards <api_cards>` page, read from the source code; do not guess
arguments. [`llms-full.txt`](https://grid4earth.eu/healpix-analyse/llms-full.txt)
has the same cards and the full code of the recipes. Read {doc}`data` before
writing the data pipeline: the domain is fixed at construction, NaN must be
masked, and every sample of a batch shares the same cells.
```

## The building blocks

One line per block. *Deterministic backward* means the same gradient bits
at every run on the GPU. The full description of each block (pros, cons,
alternatives) is on its page.

```{include} ../_generated/ai_blocks.md
```

Not on this list but often useful in a model:
{doc}`HealPixPyramidConv </conv/pyramid_conv>` (NaN-aware multiscale
filtering), {doc}`HealPixFFTConv </conv/fft_conv>` (one large learned kernel
on one regional patch), and {doc}`GetDINOV3SAT </other/dino>` (frozen foundation-model embeddings).

## Which block for which architecture

```{include} ../_generated/ai_architectures.md
```

```{toctree}
---
maxdepth: 1
---
data
api_cards
recipes/index
```
