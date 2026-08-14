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
| vectorized | 0.32 s |
| cached repeat | 0.0055 s |

The observed 3,600 m fixture (133,002 cells) measured 109.56 s on the same
`main` commit and 11.20 s after optimization: a 9.8x cold-start speedup.

The exact timing is machine-dependent. The checksum and tests are the semantic
guards. A separate-checkout comparison on the fixture, including NaNs, was
bit-for-bit equal (`max_abs_error=0`).

Before optimization, `cProfile` attributed 2.33 s of 2.50 s to
`build_neighbourhoods`; 7,768 small `healpix_to_lonlat` calls consumed 1.98 s.
`line_profiler` located 85.3% of `_filter_by_cell_center_distance` in its
per-cell coordinate conversion. The optimized code batches centre conversion,
deduplicates and batches candidate conversion, and runs the exact WGS84 cutoff
as one vector operation.

The remaining cold-path bottleneck is exact WGS84 inverse geodesy: the two
batched `Geod.inv` calls account for about 0.22 s of a 0.36 s profiled run.
The first call selects candidates and the second constructs reusable metric
geometry. Repeated calls with identical domain, level, radius, and Gaussian
sigma reuse bounded geometry and weight caches. No approximate Gaussian or
planar-distance algorithm is used.
