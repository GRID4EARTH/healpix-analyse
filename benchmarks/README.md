# Topology adapter performance

`benchmark_topology.py` compares the connected-component topology paths:

- the previous `healpy` adapter, including direction selection, transpose and
  C-contiguous conversion;
- the current `healpix-analyse` adapter;
- `healpix_geo.nested.neighbours()` with automatic threading;
- the same backend restricted to one thread.

Run the benchmark from the repository root in the Pixi development
environment:

```console
pixi run python benchmarks/benchmark_topology.py
```

The benchmark uses deterministic random NESTED cell IDs, verifies exact output
equality before every timed comparison, warms each implementation, shuffles
measurement order and reports the median. Small calls are repeated within each
sample, and every sample contains at least ten calls, so that timer resolution
and thread scheduling do not dominate a single call.

Use `--dtype int64` to measure conversion overhead for NumPy's default random
integer dtype. Use `--json topology-benchmark.json` to retain all samples and
environment metadata.

## Reference result

Measured on 2026-08-16 with:

- Apple M5 Max, 18 cores, 128 GB memory;
- macOS 26.4 arm64;
- Python 3.12.13;
- NumPy 2.5.2;
- healpy 1.20.0;
- `healpix-geo` 0.2.1 built from commit
  `3115982b6719b2b44ea447eb035e493b7a2707c9`;
- depth 12, `numpy.uint64` input, seed `20260816`;
- 3 warmups and 9 shuffled timing repetitions;
- at least 10 calls and 1,000,000 processed cells per timing sample.

| cells | connectivity | previous (ms) | new (ms) | speedup | backend auto (ms) | backend 1 thread (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | edge | 0.024 | 0.019 | 1.24x | 0.016 | 0.016 |
| 1,000 | edge_or_vertex | 0.024 | 0.021 | 1.12x | 0.019 | 0.018 |
| 10,000 | edge | 0.218 | 0.123 | 1.77x | 0.117 | 0.116 |
| 10,000 | edge_or_vertex | 0.212 | 0.137 | 1.55x | 0.131 | 0.132 |
| 100,000 | edge | 3.179 | 0.894 | 3.55x | 0.818 | 1.491 |
| 100,000 | edge_or_vertex | 3.969 | 1.461 | 2.72x | 1.263 | 2.149 |
| 1,000,000 | edge | 31.787 | 5.827 | 5.46x | 5.683 | 17.217 |
| 1,000,000 | edge_or_vertex | 38.350 | 7.951 | 4.82x | 7.983 | 20.835 |

Timings are machine-dependent. They are reference measurements rather than a
performance guarantee. In particular, `num_threads=0` allows `healpix-geo` to
use its automatic parallel path for sufficiently large inputs. The one-thread
backend column separates that effect from the backend implementation itself.
The small difference between the adapter and the automatic backend is within
normal run-to-run variation here, so this table does not claim a precise
adapter-overhead percentage.

For the NumPy-default `int64` input, the same benchmark at one million cells
measured 28.622 ms versus 6.162 ms (4.65x) for edge connectivity and
32.434 ms versus 8.940 ms (3.63x) for edge-or-vertex connectivity. Recording
the dtype matters because both adapters preserve their input-validation and
conversion costs.
