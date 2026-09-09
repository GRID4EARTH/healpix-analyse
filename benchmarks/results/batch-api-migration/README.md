# Batched coverage API migration

The revised API preserves native batching. It does not replace the batch
operation with a loop over scalar queries. GEO now accepts `(N, 2)` centers
through `nested.cone_coverage` and returns three `RaggedArray` objects sharing
one offsets array. ANALYSE consumes the compact data directly.

Three alternating old/new process pairs per case, same saved inputs,
macOS arm64 / 18 logical CPUs, Python 3.13.15. Each timed cold call clears
filter caches; imports, initialization and fixture loading are excluded.
The radius is 80 m at level 19 and 100 m at level 20 (`sigma_m=20`).

| Level / nominal patch size | Old batch cold (s) | New batch cold (s) | Old repeat (s) | New repeat (s) |
|---|---:|---:|---:|---:|
| 19 / 600 m | 0.04478 | 0.04568 | 0.00317 | 0.00311 |
| 20 / 600 m | 0.79107 | 0.79580 | 0.06020 | 0.05959 |
| 19 / 3600 m | 1.60426 | 1.57734 | 1.54158 | 1.55429 |
| 20 / 1200 m | 3.44068 | 3.44049 | 3.39003 | 3.42375 |

Values are medians. Cold differences range from about -2% to +2%; this small
experiment finds no material regression, but does not establish an exact
zero-overhead guarantee. All 12 paired output arrays are bit-identical.
The large cases exceed default cache limits, so repeats rebuild geometry.
The fixture is a circular domain around (2 E, 48 N) enclosing the nominal
square patch; it is a synthetic scene, not a complete Sentinel-2 tile.

Baseline: ANALYSE `3ba9c25`, GEO integration `e170f4a`.
Candidate: GEO `0495c3b` on main `7659474`, with the ANALYSE API adapter in
this branch. GEO's development version is 0.4.1; this is not a released API.
WGS84 geodesic distances still use `pyproj.Geod`; neither spherical angular
distances nor a healpy backend were substituted.

`timings.json` records individual timings and loaded module paths.
`correctness.json` records exact comparisons. Use
`benchmarks/benchmark_batch_migration.py --fixture INPUT.npz --output OUTPUT.npy`
in each environment, with the same `ids` and `values` arrays; the filename
encodes `level-size-truncate-input.npz`. Fixtures come from
`benchmark_gaussian_filter.patch` and `synthetic_scene`.

Validation: ANALYSE 408 passed / 21 skipped; GEO Rust 89 passed; GEO Python
567 passed / 8 skipped / 1 pre-existing failure. The failure is an existing
reference test calling nonexistent `healpy.healpix_to_base_cell_coordinates`;
it reproduces against released GEO 0.4.0. The new batched tests pass. Rust
Clippy passes with warnings denied, and Rust formatting/Python lint pass.

The dependency pin must remain until a GEO release contains native batched
coverage. Do not replace it with GEO 0.4.0 to make a promotion PR mergeable.
