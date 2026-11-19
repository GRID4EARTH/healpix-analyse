# HEALPIX-GEO 0.4.0: performance impact of the unpublished batch API

Measured on 2026-09-08.

Most of the ANALYSE integration speed-up remains without the batch API. In the four cases measured here, the first-call time with the published release is about 1.41 to 1.87 times the batch time. Against the then-current main, the three cases compared were about 13 to 30 times faster. The Gaussian output of the published release (sequential) and of the batch path is identical.

## First Gaussian call, total

Median of 3 runs each, in seconds.

| level | size (m) | cells | main + GEO 0.4.0 | integration + GEO 0.4.0 | integration + GEO batch | published / batch |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 600 | 3,883 | 2.4181 | 0.0794 | 0.0431 | 1.84× |
| 20 | 600 | 15,069 | 14.9453 | 1.1130 | 0.7752 | 1.44× |
| 19 | 3,600 | 133,002 | 66.1529 | 2.7849 | 1.4904 | 1.87× |
| 20 | 1,200 | 59,426 | not measured | 4.6338 | 3.2948 | 1.41× |

The comparison with main was also measured newly for this report. Even with the published release, speed-ups over main of 30.4×, 13.4× and 23.8× remain, in table order.

## Control experiment that changes only the batch

Within the same GEO integration build, only `cone_coverage_many` was enabled or disabled. The execution order in each repetition was shuffled with a fixed seed. Times in seconds.

| level / size | total scalar | total batch | total ratio | search scalar | search batch | search ratio |
|---|---|---|---|---|---|---|
| 19 / 600 m | 0.0776 | 0.0431 | 1.80× | 0.0399 | 0.0053 | 7.52× |
| 20 / 600 m | 1.0769 | 0.7752 | 1.39× | 0.3655 | 0.0607 | 6.02× |
| 19 / 3,600 m | 2.7045 | 1.4904 | 1.81× | 1.4094 | 0.1781 | 7.91× |
| 20 / 1,200 m | 4.5489 | 3.2948 | 1.38× | 1.5139 | 0.2533 | 5.98× |

The search time is the whole of `_cone_candidate_csr` (including the assembly of the CSR arrays), not only the time of the native call. Geodesic distance computation, domain matching, weight construction and aggregation remain after the search is accelerated. For that reason, a 6 to 8 times speed-up of the search alone does not become a 6 to 8 times speed-up of the whole Gaussian.

The comparison between the published release and the integration build also contains the difference between GEO versions. To judge the effect of the batch alone, use this control experiment.

## Re-runs and cache

The same cells and parameters were re-run with the input values changed to `values + 1`. Times in seconds.

| level / size | published, default capacity | batch, default capacity | cache hit | published, enlarged capacity |
|---|---:|---:|---|---:|
| 19 / 600 m | 0.0031 | 0.0029 | yes | not measured |
| 20 / 600 m | 0.0610 | 0.0588 | yes | not measured |
| 19 / 3,600 m | 2.7790 | 1.4968 | no | 0.1100 |
| 20 / 1,200 m | 4.6255 | 3.2878 | no | 0.2543 |

The default limits are 192 MiB for geometry and 96 MiB for weights. The two large cases exceed both limits and rebuild every time. In the enlarged-capacity experiment both limits were set to 1024 MiB. This is not a setting that reserves 2 GiB permanently.

- Memory actually held, level 19 / 3,600 m: geometry 199.2 MiB + weights 131.5 MiB.
- Memory actually held, level 20 / 1,200 m: geometry 521.3 MiB + weights 347.0 MiB.

The held amounts are not the peak memory including temporary arrays during processing. A change of input domain or sigma requires a rebuild. This experiment covers the case where the same domain and parameters are used repeatedly. A cache hit was confirmed by the number of search calls being zero and by `radial_filter_cache_info()`.

## Correctness and neighbour API

- In all four cases, the input cell IDs and values are identical in both GEO environments.
- The Gaussian outputs of integration scalar, integration batch and published scalar are also bit-identical.
- The comparison with main shows a maximum absolute difference of 4.44e-16 in all three cases. This is of the size expected from a different floating-point aggregation order.
- In a temporary copy, the connectivity passed to GEO was converted from `edge_or_vertex` to `all`, and the 8 columns were rotated one column to the left. ANALYSE's previous direction order and public API names are kept; the four edge directions are not changed.
- Levels 0/1/3/6 (all cells) and levels 16/20/29 (10,000 fixed-seed samples each) agree exactly with healpy. Missing positions, int64 and C-contiguity were also checked.
- With the temporary copy containing the correction above and the test-side adaptations (mock expectations, and `raising=False` in the monkeypatch that imitates the unpublished API), the result is **408 passed, 21 skipped**.
- This Gaussian path does not call the `_topology` adapter for immediate neighbours, so the direction-order correction does not affect the Gaussian timings above.

The code, dependency settings and PRs of the main project were not changed. Documentation build and CI verification were not part of this performance check.

## Measurement conditions and limits

- ANALYSE integration: `3ba9c25be4bd2ed5e2856b7e3c1eebc7bb65b955`.
- ANALYSE main: `b7234ecd0b4be49657e5f1e5d765740a4deee747`.
- GEO integration: `e170f4afb7f1c01add0d51e88de0c6fd8b9dd9c4` (the installed version string reads 0.2.1). Published release: PyPI 0.4.0.
- macOS 26.4 arm64, 18 logical CPUs, Python 3.13.15, NumPy 2.5.2, pyproj 3.7.2, healpy 1.20.0.
- Centre (2°E, 48°N), WGS84, the asymmetric synthetic field of the existing `benchmark_gaussian_filter`.
- `size_m` is a generation parameter for the circle enclosing a square; the actual input domain is circular.
- sigma = 20 m. Level 19 uses truncate = 4 (radius 80 m), level 20 uses truncate = 5 (radius 100 m).
- Median of 3 runs per condition. Benchmark processes were not run concurrently.
- Before each first-call measurement, the ANALYSE geometry/weight caches were cleared. Imports, input creation, initialization and saving of results are outside the timing. This is not a process start-up time with the OS cache and the allocator also reset.
- The 0.4.0 environment shares NumPy and other packages with the existing environment; only GEO was installed in a temporary venv.

This is a synthetic input at one location with sigma = 20 m. It does not guarantee performance at other latitudes, with very large sigma, on real Sentinel-2 tiles, or on different hardware. Peak memory and the comparison with SciPy were not measured here.

## Assessment

From these performance results, there is little need to make the publication of the batch API a mandatory condition for promoting to main. One option is to state explicitly the sequential fallback of the published 0.4.0, and to proceed with the neighbour-API correction, the dependency-setting update and the documentation/CI verification. The batch API is tracked as an additional speed-up.

If the real workload has a strict upper bound on first-call latency, decide separately for that workload.

## Files for reproduction

- `../../benchmark_geo_release.py`: control comparison, published-release comparison, cache experiment.
- `integration/timings.json`, `release040/timings.json`, `release040-large-cache/timings.json`: all measured values and environment information.
- `main-timings.json`: measurements of main, run with PYTHONPATH pointing to a main exported with git archive.
- `output-validation.json`: recorded result of the comparison between saved inputs and filter outputs.
- `verify_topology.py`, `topology-validation.json`: neighbour verification.

From the repository root, use the Python of the matching GEO environment:

```sh
PYTHONPATH=. python benchmarks/benchmark_geo_release.py --output /tmp/geo-benchmark
PYTHONPATH=. python benchmarks/benchmark_geo_release.py --cases 19:3600:4 20:1200:5 --cache-mib 1024 --output /tmp/geo-cache-benchmark
```

## Note added when the files were added to the repository

- This note and the changes to the list of files above are not part of the original report; the measured values and tables are unchanged. The original report also listed `run_main_baseline.py`, `verify_outputs.py`, `topology-adaptation.patch` and `adapted-tests.log`, and the saved `.npy` / `.npz` fixtures. They are not included here: the first two read the saved fixtures, and the patch and the log refer to a temporary copy of the integration branch.
- `benchmark_geo_release.py` was run against main at 272df21 with `healpix-geo` 0.5.0 (`--cases 19:600:4 20:600:5 --repeats 1`) and works unchanged. `healpix-geo` 0.5.0 has no `nested.cone_coverage_many` (the batched search is `nested.cone_coverage` itself), so the script then runs only the `scalar` mode, which already uses the batched `cone_coverage`.
- `verify_topology.py` was run against main at 272df21 and reproduces `topology-validation.json` byte for byte.
