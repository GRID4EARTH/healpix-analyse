# healpix-analyse

Signal analysis on HEALPix grids, for Earth observation.

`healpix-analyse` answers the question *what do you do with the values stored in
HEALPix cells*: convolutions and filters in metres, neighbourhood statistics,
spherical harmonics and local FFT spectra, resolution changes, interpolation
and foundation-model embeddings. Its companion
[healpix-geo](https://healpix-geo.readthedocs.io/) answers the other question —
*where* those cells are.

The tools are written in PyTorch and work on the whole sphere or on a regional
set of NESTED cells; most of them accept NumPy arrays or tensors and are
differentiable. The exceptions are listed on the {doc}`start/conventions` page.

```bash
pip install git+https://github.com/GRID4EARTH/healpix-analyse.git
```

## Find the tool for your task

Start from the **{doc}`decision guide <start/overview>`**: a few questions
about your data and your goal lead to the right tool. Or pick the family of
your task:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} A. Convolution & filtering
:link: conv/index
:link-type: doc

Smooth or filter in metres, learn a convolution, wide kernels, FFT
convolutions, land barrier.
:::

:::{grid-item-card} B. Statistics & local analysis
:link: stats/index
:link-type: doc

Neighbourhood mean / median / max, gradients, divergence and curl, masks,
connected components, Minkowski functionals.
:::

:::{grid-item-card} C. Spherical harmonics & FFT
:link: sht/index
:link-type: doc

Spherical-harmonic transforms on HEALPix and lat/lon grids, local FFT,
power spectra.
:::

:::{grid-item-card} D. Other tools
:link: other/index
:link-type: doc

Coarser and finer resolutions, multiscale pyramids, resampling,
interpolation at points, tiles, DINOv3 embeddings.
:::

:::{grid-item-card} Basic tools for AI
:link: ai/index
:link-type: doc

The building blocks of neural networks on HEALPix data (convolutions,
resolution changes, masked norms, pooling, positional encodings,
attention, patch tokens) and runnable ResNet, U-Net, MAE and Swin-like
recipes.
:::
::::

## Every tool, one line each

What each tool is for, with its page. The same catalogue, with the pros, cons
and alternatives of every tool, is published for AI assistants as
[`llms.txt`](https://grid4earth.eu/healpix-analyse/llms.txt) and
[`llms-full.txt`](https://grid4earth.eu/healpix-analyse/llms-full.txt).

```{include} _generated/decisions_all.md
```

For the signature of any function, see the {doc}`API reference <api/index>`.

```{toctree}
---
maxdepth: 1
caption: Getting started
hidden: true
---
start/installation
start/overview
start/conventions
```

```{toctree}
---
maxdepth: 2
caption: A. Convolution & filtering
hidden: true
---
conv/index
```

```{toctree}
---
maxdepth: 2
caption: B. Statistics & local analysis
hidden: true
---
stats/index
```

```{toctree}
---
maxdepth: 2
caption: C. Spherical harmonics & FFT
hidden: true
---
sht/index
```

```{toctree}
---
maxdepth: 2
caption: D. Other tools
hidden: true
---
other/index
```

```{toctree}
---
maxdepth: 2
caption: Basic tools for AI
hidden: true
---
ai/index
```

```{toctree}
---
maxdepth: 1
caption: Examples
hidden: true
---
notebooks/index
```

```{toctree}
---
maxdepth: 2
caption: Reference
hidden: true
---
api/index
```

```{toctree}
---
maxdepth: 2
caption: Developer notes
hidden: true
---
dev/index
```

```{toctree}
---
maxdepth: 1
caption: About
hidden: true
---
about/changelog
about/license
```
