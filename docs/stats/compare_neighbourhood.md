# Neighbourhood statistics: which tool?

Several tools take a statistic over the cells within a radius. They share the
same neighbourhoods (cells whose centres lie within `radius` metres on WGS84,
found with healpix-geo), but differ in weights, missing values and outputs.

| Need | Tool | Weights | NaN | Output | Differentiable |
|---|---|---|---|---|---|
| mean, median, min, max, std, count, mode, any, all | `neighbour_reduce` and its aliases ({doc}`neighbour_reduce`) | none | propagates | values on `domain` | mean, sum, min, max, median, std |
| the same reductions many times on the same cells | `HealPixNeighbourReducer` ({doc}`neighbour_reduce`) | none | propagates | values on `domain` | as above |
| a weighted mean (Gaussian, top hat, any kernel of the distance) | `radial_filter`, `gaussian_filter` ({doc}`/conv/radial_filter`) | `kernel(distance_m)` | **left out, weights renormalised** | values on `domain` | in the values |
| grow a mask by a radius | `binary_dilation` ({doc}`morphology`) | none | — | cell ids | no |
| shrink a mask by a radius | `binary_erosion` ({doc}`morphology`) | none | — | cell ids | no |

## Mean: `mean_filter` or a top-hat `radial_filter`?

They use the same cell-centre neighbourhood and give the same result on
complete data (agreement to 3e-16 in one test). Prefer `radial_filter` with a
top-hat kernel when:

- the data have gaps: `radial_filter` leaves NaN out and renormalises, while
  `mean_filter` returns NaN;
- you filter several times on the same cells: `radial_filter` caches its
  geometry and was faster in a CPU measurement (see {doc}`neighbour_reduce`).

```python
from healpix_analyse import radial_filter

mean = radial_filter(values, cell_ids, refinement_level,
                     radius_m=3000.0, kernel=lambda d: 1.0)
```

Neither weights by cell area.

## Dilating a mask: `max_filter` or `binary_dilation`?

On a boolean field, `max_filter` (with `domain` equal to the cells) and
`binary_dilation` give the same cells; this was checked for both
neighbourhood types on one test domain. Choose by the representation you have:

- a boolean value per cell → `max_filter` (and `min_filter` for an erosion);
  the output is a boolean array on `domain`, and `include_self` is an option;
- a list of active cells → `binary_dilation` / `binary_erosion`; the output is
  a list of cell ids, unbounded without `domain`.

## Connected regions

To label regions or measure their area, use the components
({doc}`components`): they use the immediate edge or edge-and-vertex
neighbours of the NESTED scheme, not a metric radius.
