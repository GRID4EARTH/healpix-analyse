# B. Statistics & local analysis

Local statistics and local analysis without spherical harmonics or FFT:
statistics over a radius in metres, gradients, divergence and curl, masks,
connected regions and Minkowski functionals.

## First questions

1. **A statistic over a radius?** Unweighted (median, max, mode, count) →
   `neighbour_reduce`; weighted or with gaps → `radial_filter`
   ({doc}`compare_neighbourhood`).
2. **A derivative?** Of a scalar field → `gradient`; divergence and curl of a
   flow → `HealPixDivCurl` locally, or spherical harmonics on the whole sphere
   ({doc}`compare_divcurl`).
3. **A mask?** Grow or shrink it → morphology; label its regions and measure
   them → components; a differentiable shape statistic → Minkowski
   functionals.

Comparisons: {doc}`compare_neighbourhood` · {doc}`compare_divcurl`.

## I want to …

```{include} ../_generated/decisions_B.md
```

```{toctree}
---
maxdepth: 1
caption: Compare
---
compare_neighbourhood
compare_divcurl
```

```{toctree}
---
maxdepth: 1
caption: Tools
---
neighbour_reduce
gradient
divcurl
morphology
components
minkowski
```
