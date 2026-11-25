# D. Other tools

Changing resolution, multiscale decompositions, resampling between levels and
domains, interpolation at points, lat/lon grids, tiles and foundation-model
embeddings, and the validation helpers.

## First questions

1. **Coarser, finer, or both?** One level coarser → `HealPixDown`; finer →
   `HealPixUp`; any level to any level, or another domain →
   `resample_healpix`; every scale at once → `HealPixDecomp`
   ({doc}`compare_resolution`).
2. **Values somewhere else than at cell centres?** → `get_interp_val`.
3. **Embeddings for clustering or classification?** → `GetDINOV3SAT`
   (experimental, gated weights).
4. **Random augmentation of training batches?** → `HealPixAugment`
   ({doc}`augment`).

## I want to …

```{include} ../_generated/decisions_D.md
```

```{toctree}
---
maxdepth: 1
caption: Compare
---
compare_resolution
```

```{toctree}
---
maxdepth: 1
caption: Tools
---
down
up
decomp
resample
interpolation
augment
dino
validation
```
