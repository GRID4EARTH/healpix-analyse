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
guards. A separate-checkout comparison on the fixture, including NaNs, was
bit-for-bit equal (`max_abs_error=0`).

Before optimization, `cProfile` attributed 2.33 s of 2.50 s to
`build_neighbourhoods`; 7,768 small `healpix_to_lonlat` calls consumed 1.98 s.
`line_profiler` located 85.3% of `_filter_by_cell_center_distance` in its
per-cell coordinate conversion. The optimized code batches centre conversion,
deduplicates and batches candidate conversion, and runs the exact WGS84 cutoff
as one vector operation.

Before geodesy multithreading, the two batched `Geod.inv` calls accounted for
about 0.22 s of a 0.36 s profiled run. The first call selects candidates and
the second constructs reusable metric geometry. Repeated calls with identical
domain, level, radius, and Gaussian sigma reuse bounded geometry and weight
caches. No approximate Gaussian or planar-distance algorithm is used.

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

That fixture contains 15,069 cells. Serial WGS84 geodesy measured 5.98 s cold;
the eight-thread-capped implementation measured 2.15 s cold, a 2.8x overall
speedup. Its geometry exceeds the 192 MiB cache limit, so the measured 1.95 s
repeat rebuilds geometry rather than representing a cache hit.

With the SciPy comparison enabled on the same Apple Silicon machine, the
area-matched grid was 123 x 123 (15,129 samples), with 6.114 m spacing and a
3.271-pixel sigma. A new measurement gave 2.253 s for the exact HEALPix cold
path and 0.000167 s median for SciPy over 50 applications. The roughly 13,500x
cold-time ratio is specific to this small fixture and machine; it primarily
shows the algorithmic advantage of a separable Cartesian Gaussian, rather
than an implementation target attainable without changing HEALPix semantics.

The accompanying `cProfile` run attributed 1.984 s of 2.253 s to filter
geometry: 1.230 s in neighbourhood construction and 0.689 s in metric
geometry. The two multithreaded WGS84 distance batches had 0.672 s cumulative
wrapper time, while candidate deduplication used 0.627 s (`argsort` alone used
0.486 s). Threaded cumulative times overlap, so these numbers are not
additive. At this scale the remaining cold bottleneck is therefore split
between exact inverse geodesy and candidate construction/deduplication, rather
than being attributable to Gaussian weight application.

`--compare-scipy` also times `scipy.ndimage.gaussian_filter` on a regular
two-dimensional grid with approximately the same sample count and the same
local planar area as the circular HEALPix fixture. The Cartesian pixel sigma
is derived as `sigma_m / spacing_m`, and the same `truncate` is used. Install
the benchmark dependency first with `pip install -e '.[benchmark]'`.

This is a reference throughput comparison, not a numerical-equivalence test.
SciPy uses a separable Gaussian on a square Cartesian grid with reflected
boundaries. The HEALPix implementation evaluates a finite radial kernel using
exact WGS84 inverse-geodesic distances on an irregular spherical grid and
renormalizes at the supplied-domain boundary. Those semantics and asymptotic
costs are intentionally not changed merely to match the SciPy timing.
