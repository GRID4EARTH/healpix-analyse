# Migration to the batch API based on Justus's implementation: final performance check

Baseline: the previous ANALYSE `integration` with GEO `integration`.
Candidate: GEO PR #252 at `42d3bfb6919bbfe33ded96bb56642faeb186e1d5` plus the
local Python adapter and compatibility changes, with this branch of ANALYSE.
These are not results of PR #252 unmodified.

## Method

The same saved inputs were used for four cases, five pairs each, 40 runs in
total, each in a separate process. The order of old and new within each pair
was shuffled with a fixed seed. Each run first initialises the libraries on
10 cells and clears the filter caches, then times the first (cold) call. It
then reruns with the same cell ids and the values shifted by 1, and checks the
shifted output with a tolerance of 1e-12. No other tests or builds ran during
the measurements. macOS arm64, same machine, Python 3.13, sigma = 20 m,
default cache sizes. Gaussian distances still use WGS84 `pyproj.Geod`.

`run_comparison.py` reproduces the runs; it takes the two checkouts, their
Python interpreters and the fixture directory as arguments. The fixtures are
not part of the repository.

## Medians

A positive change means the candidate took longer.

| level / size (m) / truncate | cold: old (s) | cold: new (s) | change | repeat: old (s) | repeat: new (s) | change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 / 600 / 4 | 0.043135 | 0.043506 | +0.86% | 0.003121 | 0.002997 | -3.99% |
| 20 / 600 / 5 | 0.776757 | 0.771537 | -0.67% | 0.061775 | 0.060960 | -1.32% |
| 19 / 3600 / 4 | 1.561686 | 1.566667 | +0.32% | 1.569193 | 1.563751 | -0.35% |
| 20 / 1200 / 5 | 3.433233 | 3.444451 | +0.33% | 3.430040 | 3.423830 | -0.18% |

Cold medians change by -0.67 % to +0.86 %: no material regression in these
cases, but this does not guarantee zero overhead for every machine or input.
The 19/3600 and 20/1200 cases exceed the caches, so repeats recompute.
All 20 cold outputs are identical, including NaN positions (maximum absolute
difference 0). Per-run values and paired differences are in `summary.json`;
all runs are in `timings.json`.

## Final code check

After the measurements, Clippy flagged an `if let` that always matches; it was
replaced by the equivalent direct destructuring of the two elements. The
algorithm, thread count and return values did not change, and the timings
above precede this cleanup. After it, the release build was rebuilt: 66 GEO
cone coverage tests and 408 ANALYSE tests passed (474 in total, 21 skipped),
Rust Clippy passed for the workspace and all targets with warnings denied, and
`git diff --check` passed.

The same file contents were then brought into GEO PR #292, and the ANALYSE
pixi pin was updated to `508ef39850a7bb0e8e07b0360ca7a9a5326d69d5`, which has
PR #252 as an ancestor; the follow-up commit is `2209051`.

A last check before publishing added the eight-thread limit to GEO's
`num_threads=0` as well. The measured ANALYSE path requests eight threads
explicitly and is not affected. The published GEO commit was
`5a43c343d2f5940ae87a730fbedb856e0256ef4a`; its release build passed 69 GEO and
408 ANALYSE tests (477 in total, 21 skipped).

Later changes to #292 (no thread cap, always ragged results) are validated in
GRID4EARTH/healpix-analyse#51, not by these timings.
