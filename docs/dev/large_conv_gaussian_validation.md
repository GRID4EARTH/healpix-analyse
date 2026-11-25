# LargeConv Gaussian recipe: validation measurements

Moved from {doc}`/conv/large_conv` (section *The recipe: `LargeConv.from_gaussian`*).
The benchmark is `benchmarks/benchmark_largeconv_widths.py`.

Measured at level 10 on the calibration patch (256 x 256 cells near the
equator), 756 widths from 2.25 to 40 cells including the four switches
(4.7, 9.35, 18.65 and 36.9 cells; the benchmark was run before the limit at
31.7 cells of the next paragraph, which `gaussian_plan` now enforces with
the default `patch_depth=8`):

- the effective width of the response to the calibration cell is strictly
  increasing (smallest step 0.048 cell for a sigma step of 0.05 cell; no
  shelf, no step down);
- it equals the requested width within 3.8e-4 relative (median 2.3e-5).

**This accuracy is for the calibration cell, and it is measured on the
patch used for the calibration**, so it checks the inversion of the table.
The patch has zero padding at its border, so a wide response loses its tail
there during the calibration, and the plan then sets a compact Gaussian that
is too wide on a larger domain. Re-measured at the same cell on the
512 x 512 block that contains the patch, the width / sigma is 1.0000 to
1.0003 up to 32 cells (border at 4 sigmas or more), 1.0011 at 36, 1.0044 at
40, 1.0126 at 48, 1.047 at 56 and 1.106 at 60 cells. `gaussian_plan`
therefore refuses widths above a quarter of the distance from the
calibration cell to the patch border (31.7 cells at level 10 with
`patch_depth=8`; `patch_depth=9` doubles it, for a calibration on four
times more cells). Other cells and other places differ, as follows.

- **Other cells of the domain.** Measured on a 512 x 512 cell patch
  (level 10, near the equator, calibrated on its centre cell), response to 96
  random cells (fixed seed) drawn among those at least `5 sigma + 2**n` fine
  cells from the patch edge, so that the tail and one coarse block are
  inside; the sampling windows cover 7.5 or more coarse blocks of the plan
  in each direction (the automatic rule has coarser blocks: 1.9 at sigma 36,
  10 at 18, 26 at 9, more below). Ratio to the requested width (plan) and
  to its own mean (automatic rule, which does not aim at the requested
  width), min / mean / max:

  | sigma (cells) | plan levels | plan width / sigma | plan total | automatic rule: width / own mean | automatic rule: total |
  |---:|---:|:---|:---|:---|:---|
  | 3 | 1 | 0.921 / 0.986 / 1.043 | 0.948 / 0.999 / 1.046 | 0.971 / 1.000 / 1.021 | 0.932 / 0.998 / 1.045 |
  | 6 | 2 | 0.971 / 0.991 / 1.009 | 0.945 / 0.999 / 1.047 | 0.984 / 1.000 / 1.016 | 0.945 / 0.999 / 1.048 |
  | 9 | 2 | 0.988 / 0.996 / 1.003 | 0.943 / 1.001 / 1.047 | 0.981 / 1.000 / 1.018 | 0.941 / 1.001 / 1.047 |
  | 18 | 3 | 0.992 / 0.997 / 1.001 | 0.931 / 0.994 / 1.046 | 0.980 / 1.000 / 1.014 | 0.920 / 0.994 / 1.044 |
  | 36 | 4 | 0.993 / 0.999 / 1.001 | 0.940 / 0.996 / 1.040 | 0.987 / 1.000 / 1.005 | 0.917 / 0.990 / 1.024 |

  The width spread of the plan is 8 % wide at 3 cells (0.92 to 1.04, up to
  240 cells from the calibration cell, which lies at 0.9999), 3 % at 6 cells and
  at most 1.5 % from 9 cells on. The automatic rule, which has the same
  Down/Up chain, spreads as much around its own mean, so this is a property
  of the chain and not of the recipe. The width ratio varies with the
  distance from the calibration cell for narrow widths (within 16 cells of
  it the spread at 3 cells is 0.997 to 1.019); the total of one cell is 0.93
  to 1.05 everywhere. Calibrating on the cell-averaged width was not
  adopted: the mean is already within 1.4 % of the request at 3 cells
  (0.986) and within 1 % above, and averaging would not reduce the spread.
- **The total of one cell is not conserved.** `LargeConv` is not a
  conservative operator: the sum of the response to one cell is 0.92 to
  1.05 of the input in the table above, for the plan and for the automatic
  rule alike. Constants are preserved when the kernel sums to one, but the
  integral of a field is not. A conservative smoothing would need another
  operator, such as a conservative FFT projection; `gaussian_filter` with
  `normalization="present"` normalises each row to one over the cells present
  (and `"full_disc"` loses the mass that leaves the domain), see
  {doc}`/conv/radial_filter`.
- **Tail near the patch edge (measurement artefact and domain edge).** On
  the same patch, cells up to 20 cells from the edge (sampling window of
  236 cells half-width, 29 to 118 plan blocks) give width / sigma 0.985 to
  1.003 at 9 cells, 0.925 to 1.001 at 18 and 0.851 to 1.001 at 36, and the
  total 0.896 to 1.064 (plan) and up to 1.18 (automatic rule at 36 cells):
  the tail that falls outside the patch is lost. Wide spreads seen with
  impulses that close to an edge are this effect; with the margin of the
  table above they disappear (0.988 or better at 9 cells and above). The
  next bullet gives the same effect at the edge of a real partial domain.
- **Domain edge (regional domain).** Down and Up use zero padding, so the tail
  that falls outside the domain is lost. In a 15-degree disc at level 8
  (13 411 cells), the response to a cell at 0.92 of the radius is
  narrower than requested by 6.4 % (sigma 3 cells) and 12.7 % (sigma 6),
  with a total of 0.91 and 0.92; at 0.8 of the radius 0.6 % and 3.0 %; at
  the centre 0.5 % and 0.04 %. A constant field stays 1 in the interior and
  drops to 0.98 at the edge.
- **Other places.** The compact convolution samples the Gaussian on a
  lattice whose shape depends on the position. With the equatorial
  calibration, the response at 55 degrees latitude is up to 21.8 % wider
  than requested (22.4 % in an independent sweep; median 0.4 %): 13 to 22 %
  below 3 cells, 5 to 12 % from 3 to 4.7 cells, 1 to 7 % from 4.7 to 9.35,
  at most 3.6 % above, and still strictly increasing. That is why
  `from_gaussian` and `gaussian_plan(cell_ids=...)` calibrate on the block
  of the domain by default. At 55 degrees that gives an error of 2.9e-4 and
  a narrowest width of 2.75 cells.
