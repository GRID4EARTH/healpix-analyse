# Changing resolution: which tool?

Ten ways to go from one sampling to another. They differ in direction, in what
they conserve and in how they treat gaps.

| Tool | Direction | Conserves | Gaps (NaN) | Differentiable | Domain |
|---|---|---|---|---|---|
| `HealPixDown(mode="smooth")` ({doc}`down`) | one level coarser | nothing (rows normalised: constants kept) | spreads over each disc | yes | whole sphere or regional |
| `HealPixDown(mode="smooth_conservative")` ({doc}`down`) | one level coarser | the global mean (coverage-weighted on a regional domain) | offset from finite values; NaN stays local | yes | whole sphere or regional |
| `HealPixDown(mode="mean")` ({doc}`down`) | one level coarser | the global mean (four children, no overlap) | to the parent only; `skipna=True` ignores them | yes | whole sphere or regional |
| `HealPixDown(mode="maxpool")` ({doc}`down`) | one level coarser | — (maximum) | propagates | almost everywhere | whole sphere; **known issue** on regional domains with incomplete parents |
| `HealPixUp` ({doc}`up`) | one level finer | constants with `col_l1` and `weight_norm="l1"` | not handled | yes | whole sphere or regional |
| `HealPixResampler`, `resample_healpix` ({doc}`resample`) | any level to any level, and one domain to another | depends on `down_mode` | normalised by the coverage; NaN where there is no support | yes | input and output domains chosen independently |
| `HealPixDecomp` ({doc}`decomp`) | all coarser levels at once, as bands | exact synthesis (sum of the bands); the coarse band keeps the mean with `down_mode="mean"` or `"smooth_conservative"` | `compute_weighted` is NaN-aware | yes | whole sphere or regional |
| `nested_to_tiles` / `tiles_to_nested` ({doc}`dino`) | re-layout of NESTED blocks as square images, no resampling | everything (a permutation) | `fill` = mean, zero or NaN | no (NumPy) | the blocks present |
| `resample_to_latlon_grid` ({doc}`resample`) | scattered samples to a regular lat/lon raster with the shape of `data` (2-D) | — | points outside the convex hull of the samples get the mean of `data` | no (NumPy, SciPy) | any points, not tied to HEALPix ids |
| `get_interp_val` ({doc}`interpolation`) | a whole-sphere map to arbitrary points (bilinear) | — | NaN if one of the four cells is NaN | no (NumPy) | whole-sphere map only |

## Rules of thumb

- **A coarser map for analysis, keeping the mean** → `HealPixDown` with
  `"mean"` (no overlap) or `"smooth_conservative"` (smooth).
- **A pooling layer in a network** → `HealPixDown` `"smooth"` or `"maxpool"`
  (with complete parents), and `HealPixUp` in the decoder.
- **Several levels at once, from one domain to another, with gaps** →
  `resample_healpix` (or one `HealPixResampler` reused).
- **Every scale, exactly invertible** → `HealPixDecomp`.
- **Values at stations, tracks or profiles** → `get_interp_val`.
- **A picture for a CNN that expects square images** → `nested_to_tiles`.

`HealPixDown` and `HealPixUp` take `[N]` or `[B, N]` only and return a tuple
`(y, cell_ids_out)`; see {doc}`/start/conventions`.
