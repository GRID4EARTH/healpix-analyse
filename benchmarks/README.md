# Gaussian filter performance

Run the representative Level-19 benchmark from the repository root:

```console
python benchmarks/benchmark_gaussian_filter.py --profile
```

The default 600 m patch contains 3,883 cells at the fixed test location and
uses `sigma_m=20`, `truncate=4`. It is intentionally small enough for routine
development while exercising the same high-resolution path as a 3,600 m
patch. Use `--size-m 3600` for the larger observed workload (the enclosing
circular fixture contains 133,002 cells).

On an Apple Silicon development machine, commit `b7234ec` and the optimized
working tree produced these cold timings for the default fixture:

| implementation | cold time |
| --- | ---: |
| `main` (`b7234ec`) | 2.50 s |
| vectorized, serial geodesy | 0.32 s |
| vectorized, up to 8 geodesy threads | 0.16 s |
| cached repeat | 0.0055 s |

The observed 3,600 m fixture (133,002 cells) measured 109.56 s on the same
`main` commit and 11.20 s after optimization: a 9.8x cold-start speedup.

The exact timing is machine-dependent. The checksum and tests are the semantic
guards. The initial padded-vectorized implementation was bit-for-bit equal to
`main`, including NaNs. The later compact reduction changes floating-point
summation grouping; a separate-checkout Level-20 comparison against the padded
implementation measured `max_abs_error=6.66e-16` and
`mean_abs_error=7.18e-17`.

Before optimization, `cProfile` attributed 2.33 s of 2.50 s to
`build_neighbourhoods`; 7,768 small `healpix_to_lonlat` calls consumed 1.98 s.
`line_profiler` located 85.3% of `_filter_by_cell_center_distance` in its
per-cell coordinate conversion. The optimized code batches centre conversion,
deduplicates and batches candidate conversion, and runs the exact WGS84 cutoff
as one vector operation.

Before geodesy multithreading, the two batched `Geod.inv` calls accounted for
about 0.22 s of a 0.36 s profiled run. Geometry construction now retains the
distances used for candidate selection, eliminating the second inverse-
geodesic pass. Repeated calls with identical domain, level, radius, and
Gaussian sigma reuse bounded compact geometry and weight caches. No
approximate Gaussian or planar-distance algorithm is used.

Large WGS84 inverse-geodesic batches are split across at most eight threads.
Small batches remain serial to avoid thread-pool overhead. PROJ still performs
every inverse calculation, and tests require parallel distances to be
bit-for-bit identical to the serial result.

The requested Level-20 profile can be reproduced with:

```console
python benchmarks/benchmark_gaussian_filter.py \
  --level 20 --size-m 600 --sigma-m 20 --truncate 5 --profile \
  --compare-scipy
```

That fixture contains 15,069 cells. Serial two-pass WGS84 geodesy measured
5.98 s cold, and the padded eight-thread implementation measured 2.253 s cold
with a 1.989 s repeat. Fusing distance selection with metric geometry and
storing unpadded domain-local neighbour indices measured 1.676 s cold and
0.0448 s cached-repeat median. The compact geometry is about 125.5 MiB rather
than 196.8 MiB, so it fits the bounded 192 MiB geometry cache.
Three additional unprofiled cold runs measured 1.254, 1.442, and 1.600 s.

With the SciPy comparison enabled on the same Apple Silicon machine, the
area-matched grid was 123 x 123 (15,129 samples), with 6.114 m spacing and a
3.271-pixel sigma. The final measurement gave 1.676 s for the exact HEALPix
cold path, 0.0448 s for a cached repeat, and 0.000138 s median for SciPy over
100 applications. The cold-time ratio is specific to this small fixture and
machine; it primarily
shows the algorithmic advantage of a separable Cartesian Gaussian, rather
than an implementation target attainable without changing HEALPix semantics.

The final `cProfile` run attributed 1.574 s of 1.676 s to filter geometry.
Within it, 15,069 `cone_coverage` calls used 0.514 s, the single multithreaded
WGS84 distance pass used 0.486 s, and domain lookup used 0.131 s. Gaussian
weight construction used 0.050 s. Threaded cumulative times overlap, so these
numbers are not additive. The remaining cold bottleneck is split between
exact inverse geodesy and per-centre candidate construction, rather than
Gaussian weight application.

`line_profiler` gives the same conclusion inside the fused geometry builder:
candidate `cone_coverage` uses 41.5%, the one WGS84 distance pass 29.5%, and
domain `searchsorted` plus matching about 12.7%. Compact NumPy reduction takes
about 0.034 s in the profiled cold call; its largest individual operations are
value gathering and construction of weighted contributions.

An experimental helper,
`build_metric_geometry_from_vectorized_ring`, is available in the private
`healpix_analyse._neighbourhood` module for further candidate-generation
work. It batches `kth_neighbourhood`, filters to the domain, applies a cheap
WGS84 ECEF chord-distance lower bound, and then uses the normal exact inverse-
geodesic cutoff. It is deliberately not connected to the public filter path:
the caller must supply a topological ring proven to cover the requested
physical radius at every relevant location.

For the complete Level-20, 600 m fixture, `ring=22` reproduced all 10,942,451
cone-derived pairs with no missing or extra neighbours and bit-identical
distances. In that run cone geometry took 1.871 s and vectorized-ring geometry
took 1.931 s. Eliminating 15,069 scalar cone calls was offset by processing
the larger ring candidate set, so the experimental helper is not currently a
performance replacement. A separate global sample needed `ring=24` rather
than 22, which reinforces why no general automatic ring is assumed.

`--compare-scipy` also times `scipy.ndimage.gaussian_filter` on a regular
two-dimensional grid with approximately the same sample count and the same
local planar area as the circular HEALPix fixture. The Cartesian pixel sigma
is derived as `sigma_m / spacing_m`, and the same `truncate` is used. Install
the benchmark dependency first with `pip install -e '.[benchmark]'`.

The comparison now evaluates the same smooth, non-symmetric analytic scene at
the HEALPix and Cartesian cell centres. After filtering, the Cartesian result
is bilinearly sampled at HEALPix centre coordinates. Error metrics use only
the common interior where the full truncated kernel is separated from both
domain boundaries by two additional grid spacings.

For the requested Level-20 parameters, a new run used 6,821 common interior
points and measured:

| result metric | value |
| --- | ---: |
| mean absolute error | 0.000141410 |
| root mean square error | 0.000247732 |
| maximum absolute error | 0.00189367 |
| normalized RMSE | 0.000350969 |
| correlation | 0.999999158 |

The same run measured 1.724 s HEALPix cold, 0.0541 s cached-repeat median,
and 0.000203 s SciPy apply median. Timing variation is expected at this scale.

This remains a reference comparison, not a numerical-equivalence test.
SciPy uses a separable Gaussian on a square Cartesian grid with reflected
boundaries. The HEALPix implementation evaluates a finite radial kernel using
exact WGS84 inverse-geodesic distances on an irregular spherical grid and
renormalizes at the supplied-domain boundary. The reported numerical errors
therefore include discretization and interpolation differences between the
grids. Those semantics and asymptotic costs are intentionally not changed
merely to match the SciPy timing.
