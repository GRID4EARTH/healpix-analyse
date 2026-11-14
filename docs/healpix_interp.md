# Point interpolation

`healpix_interp` bilinearly interpolates a HEALPix map (or just returns the 4
surrounding cells and their weights) at arbitrary longitude/latitude points.
It is the HEALPix equivalent of `scipy.ndimage.map_coordinates` for a regular
grid: given a point that does not sit exactly on a cell centre, it tells you
which cells to blend, and by how much.

```python
from healpix_analyse.healpix_interp import get_interp_val, get_interp_weights

depth = 10  # nside = 2**depth
val = get_interp_val(hpx_map, lon=45.03, lat=30.03, depth=depth)

pixels, weights = get_interp_weights(45.03, 30.03, depth=depth)
# pixels : (N, 4) uint64, NESTED scheme
# weights: (N, 4) float64, each row sums to 1
```

`hpx_map` must be in NESTED order (`12 * 4**depth` values); `lon`/`lat` are in
degrees and accept scalars, 1D arrays, or arrays of any shape.

## Exact equivalence with `healpy`, for `ellipsoid="sphere"`

`ellipsoid="sphere"` is the default, and for that case `get_interp_weights` is
**not** a geometric reinvention of bilinear interpolation on HEALPix — it is a
direct NumPy port of the reference algorithm used by `healpy` itself
(`T_Healpix_Base::get_interpol`, RING scheme, transcribed from
[`healpix_base.cc`](https://github.com/healpy/healpixmirror/blob/main/src/cxx/Healpix_cxx/healpix_base.cc)).
The RING-scheme result is converted to NESTED with `healpix_geo.ring.to_nested`
(still GRID4EARTH, not `healpy`, see the {doc}`overview` conventions).

Concretely, for `ellipsoid="sphere"`:

```python
pixels, weights = get_interp_weights(lon, lat, depth=depth)
# is guaranteed identical to
hp_pixels, hp_weights = healpy.get_interp_weights(
    2**depth, lon, lat, lonlat=True, nest=True
)
```

up to floating-point rounding (the two implementations do not perform the
arithmetic in the same order, since one runs in NumPy and the other in
compiled C++).

**Validated on:**

- the 4 selected pixel ids, across ~130 000 points: 20 000 points drawn
  uniformly on the sphere at each of `depth = 3, 4, 6, 8, 10, 12`, plus a
  dedicated stress test with points within ~0.2° of both poles, plus points
  sitting exactly on a ring/φ boundary (`lon = 0°, 360°, 180°`, `lat = 0°`) —
  **zero mismatches** against `healpy.get_interp_weights(..., nest=True)`.
- the corresponding weights: identical to within `1e-15`–`4e-12`
  (`depth = 3` to `depth = 12`), i.e. pure float64 rounding noise growing with
  the number of intermediate operations at high resolution — far below any
  scientifically meaningful threshold.

This does *not* mean any two HEALPix bilinear-interpolation implementations
agree with each other in general — they don't. `healpix_geo`'s own
`nested.bilinear_interpolation` (a tangent-plane construction, correct in its
own right) picks a geometrically different 4th corner near ring/φ boundaries
and disagrees with `healpy` there by construction, not by bug. `get_interp_weights`
deliberately reproduces `healpy`'s specific RING-based convention bit for bit,
because that convention is the de facto standard other tools compare against.

## Non-spherical ellipsoids

`healpy` has no notion of a reference ellipsoid — it only ever works on a unit
sphere. So "identical to `healpy`" is not a meaningful target for
`ellipsoid="WGS84"` (or any other ellipsoid supported by `healpix-geo`): there
is no `healpy` result to match. For any `ellipsoid != "sphere"`,
`get_interp_weights` delegates directly to `healpix_geo.nested.bilinear_interpolation`,
which computes a geometrically correct bilinear interpolation on that
ellipsoid natively.

## Implementation note

The RING-scheme port (`_ring_above`, `_get_ring_info2`, `_get_interpol_ring` in
`healpix_interp.py`) is a line-by-line transcription of the C++ reference and
should not be simplified or "cleaned up" without re-running the validation
above — several constants (`fact1_`, `fact2_`, the half-pixel φ shift that
depends on ring parity) are easy to get subtly, silently wrong, and a wrong
version still returns plausible-looking pixels and weights.
