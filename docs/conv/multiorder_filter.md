# Multi-order (mixed-level) filtering

`multiorder_filter_operator(levels, cell_ids, sigma_m=...)` builds the same kind
of operator on a set of *leaves* of different refinement levels, such as a
river refined to level 15 that opens into a sea held at level 11. `levels` and
`cell_ids` are two integer arrays (the level of each leaf and its NESTED id at
that level); the result is a `FilterOperator` over the leaves in input order, to
be applied with `apply_filter_operator`. **Land is a place without a leaf**,
and leaves listed in the optional boolean `blocked` are land inside the set.

```{include} ../_generated/glance/multiorder_filter.md
```

```python
op = multiorder_filter_operator(
    levels, cell_ids, sigma_m=5_000.0, normalization="source",
)
state = apply_filter_operator(state, *op)   # probability per leaf, conserved
```

- **Graph.** Two leaves are neighbours when they share an edge or a corner,
  at any levels: the eight neighbours of a leaf at its own level come from
  healpix-geo, and each is looked up in the set (the cell itself, or its
  parent).
  The input must be **2:1 balanced** (touching leaves differ by at most one
  level) and must not overlap; a clear `ValueError` is raised otherwise. A
  contact through a corner only is usable when the two leaves have a common
  neighbouring leaf that is not blocked, in both `correction` modes, so a
  diagonal chain of land does not leak (the rule of the one-level barrier).
- **Distance.** Edge lengths are geodesic distances between leaf centres, and
  the kernel sees the length of the shortest path (a heap Dijkstra).
  With `correction="none"` (default) the path is used as it is: it is longer
  than the straight distance by the usual lattice metrication, up to 10 %
  within 30 degrees of the equator and up to 40 % elsewhere, and the width of
  the kernel is smaller by that factor.
  With `correction="lattice"` the correction of `blocked_cells` is applied,
  `d = straight * D / D0`, where `D0` is the path over the same graph with the
  blocked leaves passable (the corner rule applies to `D0` too). In open water
  `d` is exactly the straight distance, at any mix of levels. A place without
  a leaf adds nothing to `D` or `D0`, so the ratio ignores it and it does not
  lengthen `d`, across an edge or a corner contact alike: in this mode **only
  blocked leaves are land**; supply the dry leaves of a full tessellation as
  `blocked`. A hole in the set is **not** a barrier with
  `correction="lattice"` as long as the detour around it stays within the
  limits below: the distance across it is then the straight distance.
  The search is sized with `D0 / straight <= 1.5`, which holds without holes:
  on a mixed set at levels 11 to 15 along a wavy coast (22765 leaves, 400
  sources, 2.6 million pairs, no hole) the largest ratio measured was 1.38.
  Paths around a hole are longer, in `D` and in `D0`, so a source that has a
  hole or the edge of the set within reach of that search is searched again up
  to `3 * radius`; a pair whose `D` or `D0` is beyond the limit is dropped, so
  a hole that forces a detour longer than three radii acts as a barrier. On
  the 22765-leaf coast (3175 blocked leaves, sigma 1.2 km, same operator) the
  repeat costs 8 % (SciPy, 12.0 to 12.9 s) and 12 % (built-in, 44 to 50 s);
  searching every source up to `3 * radius` would cost 2.2 and 2.7 times. For
  land made of absent leaves use `correction="none"`. With one level,
  `correction="lattice"` gives the operator of
  `gaussian_filter_operator(..., blocked_cells=...)` to rounding (tested at
  `rtol=1e-12`) without holes, and with holes whose detours stay within both
  limits: the one-level operator limits the search by the candidate disc of
  1.5 radii around the source, this one by a path length of three radii, so
  for a long hole (a detour of about two to three radii or more) they can
  differ, and each acts as a barrier beyond its own limit.
- **SciPy is optional.** `shortest_path="scipy"` runs
  `scipy.sparse.csgraph.dijkstra` on the same graph (install the `scipy`
  extra); it is only imported then, and the default `"builtin"` needs no
  SciPy.
- **Weights.** A leaf of level `l` has the area `4**-l`, and `wet_fraction`
  (optional, per leaf, in `(0, 1]`) multiplies it for leaves that are mostly
  land. `normalization="source"` makes the weights that leave one leaf sum to
  one, so the total of a signal of masses (one value per leaf) is conserved
  exactly; the weight to a target is proportional to the target's area, so a
  coarse leaf receives more than a fine one at the same distance.
  `normalization="present"` makes the weights that reach a leaf sum to one, with
  the source areas as weights: it averages a signal of densities over area. With
  `area_weights=False` every leaf counts the same.
- **Leaves larger than the water.** Two leaves may touch while the water they
  hold does not meet (a river a few hundred metres wide in leaves of 200 m,
  with an embankment between two meanders). The leaf graph alone cannot know,
  and it shortens the path across such a spot. `water_contacts=(first, second)`
  restricts the graph to pairs that the caller knows to be joined by water.
- Tested against a brute-force reference that expands every leaf to its finest
  descendants and runs an unbounded Dijkstra: strait, island, closed bay,
  diagonal chain, a diagonal hole, conservation with `"source"`, and the
  one-level case. The cost is a Python heap search per leaf, fine for
  thousands of leaves and not meant for millions.
- Not done: a width rule below one leaf (the Gaussian is evaluated as it is, so
  a `sigma_m` below the distance to the nearest neighbour leaves each leaf
  nearly alone), a `domain` argument, and a proof of the
  `D0 / straight <= 1.5` bound across level steps for `correction="lattice"`
  (it is measured on sets without holes, see above, not proved).
