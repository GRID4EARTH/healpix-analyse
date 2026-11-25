# Filter caches

`radial_filter` and `gaussian_filter` keep bounded least-recently-used caches
for value-independent metric geometry and Gaussian weights. The defaults are:

```text
geometry: 192 MiB
weights:   96 MiB
```

If one filter plan exceeds either limit, that entry is not cached. A
`RuntimeWarning` reports the required and configured sizes, explains that a
repeat will rebuild the data, and gives two safe options:

1. increase the relevant limit after checking available process memory; or
2. process spatial tiles with a halo of at least
   `radius_m = sigma_m * truncate`, discard halo outputs, and stitch only the
   tile interiors.

Splitting without a halo changes neighbourhoods at tile boundaries and does
not preserve the filtering result.

Configure or inspect the caches with:

```python
from healpix_analyse import (
    configure_radial_filter_cache,
    radial_filter_cache_info,
)

configure_radial_filter_cache(
    geometry_max_mib=576,
    weight_max_mib=384,
)

print(radial_filter_cache_info())
```

`None` leaves a limit unchanged, while zero disables that cache. Reducing a
limit immediately evicts least-recently-used entries until retained data fits.
The limits cover retained arrays only; temporary arrays used while building
exact WGS84 geometry require additional memory.

The following measurements use a patch centred at 2 degrees East, 48 degrees
North with `sigma_m=20` and `truncate=5` (100 m support):

| Level | patch size | cells | neighbour pairs | geometry | weights | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 19 | 600 m | 3,883 | 705,275 | 8.13 MiB | 5.38 MiB | 13.51 MiB |
| 19 | 1,200 m | 15,075 | 2,881,729 | 33.21 MiB | 21.99 MiB | 55.19 MiB |
| 20 | 600 m | 15,069 | 10,942,451 | 125.46 MiB | 83.48 MiB | 208.94 MiB |
| 20 | 1,200 m | 59,426 | 45,475,888 | 521.34 MiB | 346.95 MiB | 868.29 MiB |

These are planning estimates, not fixed formulas. Memory depends on physical
radius, domain shape, latitude, HEALPix topology, and boundary-to-interior
ratio. A larger `sigma_m` or `truncate` increases the number of neighbour
pairs approximately with the square of the support radius. Use the warning's
measured requirement for the actual workload and leave headroom for values,
outputs, coordinate arrays, and concurrent filters.
