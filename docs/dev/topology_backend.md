# Topology backend of the connected components

Moved from the user guide of {doc}`/stats/components`.

Immediate HEALPix topology is currently obtained through a private
`healpix_analyse._topology` helper.

That helper uses the direction-preserving API:

```python
healpix_geo.nested.neighbours(...)
```

The topology architecture is:

```text
healpix-analyse
      |
      v
private topology helper
      |
      v
healpix-geo
      |
      v
CDSHEALPix
```

The private adapter preserves the connected-component topology contract while
`healpix-geo` provides deterministic directional positions and the `-1`
sentinel for missing positions. The public connected-component API is
unchanged.

## Topology performance

The `healpix-geo` backend reduces the cost of immediate-neighbour lookup for
large arrays. At depth 12 in the checked-in reference benchmark, the new
adapter is faster than the previous `healpy` adapter from 10,000 cells onward,
with the largest gains observed for one million cells.

These timings are machine-dependent reference measurements, not a performance
guarantee. The adapter uses `num_threads=0`, so `healpix-geo` may select its
automatic parallel path for sufficiently large inputs. The benchmark therefore
also reports a one-thread backend measurement to make the contribution from
parallel execution visible.

The executable benchmark, measurement procedure, exact environment, input
dtype and reference results are maintained in the repository's
[`benchmarks` directory](https://github.com/GRID4EARTH/healpix-analyse/tree/main/benchmarks).
