# `healpix_interp` — API reference

> **Module** `healpix_analyse.healpix_interp`  
> **Parts** [1 · Quickstart](healpix_interp_1_quickstart.md) · [2 · Mathematics](healpix_interp_2_mathematics.md) · **3 · API reference**

---

## Overview

| Function | Purpose |
|----------|---------|
| [`get_interp_weights`](#get_interp_weights) | Find the 4 nearest NESTED cells and bilinear weights for arbitrary (lon, lat) |
| [`get_interp_val`](#get_interp_val) | Bilinearly interpolate a NESTED map at arbitrary (lon, lat) |

---

## `get_interp_weights`

```python
pixels, weights = get_interp_weights(
    lon, lat,
    depth,
    ellipsoid="sphere"
)
```

Return the 4 nearest HEALPix cells (NESTED) and their bilinear interpolation
weights for each query position.

The algorithm locates the containing cell via `healpix_geo.nested.lonlat_to_healpix`,
gathers its 8 immediate neighbours, projects the 9 candidate cell centres onto
the local tangent plane at each query point (gnomonic projection), keeps the
4 closest, and derives area-proportional bilinear weights from their tangent-plane
positions.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lon` | array-like `[N]` | — | Longitude in degrees. |
| `lat` | array-like `[N]` | — | Latitude in degrees. Must have the same shape as `lon`. |
| `depth` | `int` | — | HEALPix depth (`nside = 2**depth`). |
| `ellipsoid` | `str` | `"sphere"` | Reference ellipsoid passed to `healpix-geo` (e.g. `"sphere"`, `"WGS84"`). |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `pixels` | `np.ndarray [N, 4]`, uint64 | Indices of the 4 selected cells, NESTED scheme. |
| `weights` | `np.ndarray [N, 4]`, float64 | Bilinear weights. Each row sums to 1. |

### Raises

`ValueError` — if `lon` and `lat` do not have the same shape.

### Notes

- With `ellipsoid="sphere"`, results closely match `healpy.get_interp_weights`
  (NESTED scheme); small differences come from using the local tangent plane
  rather than healpy's internal RING-based scheme.
- Non-spherical ellipsoids (e.g. `"WGS84"`) fold the authalic latitude
  correction into the lon/lat → cell conversion, which `healpy` cannot do.

---

## `get_interp_val`

```python
vals = get_interp_val(
    hpx_map,
    lon, lat,
    depth,
    ellipsoid="sphere"
)
```

Bilinearly interpolate a HEALPix map (NESTED) at arbitrary (lon, lat) positions.

Internally calls `get_interp_weights` and combines the 4 neighbouring cell
values with their weights:

$$
f(\text{lon}, \text{lat}) = \sum_{i=1}^{4} w_i \, m[p_i]
$$

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `hpx_map` | `np.ndarray [12 * 4**depth]` | — | Map values, NESTED order. |
| `lon` | `float` or array-like | — | Longitude(s) in degrees. |
| `lat` | `float` or array-like | — | Latitude(s) in degrees, same shape as `lon`. |
| `depth` | `int` | — | HEALPix depth (`nside = 2**depth`). |
| `ellipsoid` | `str` | `"sphere"` | Reference ellipsoid, see `get_interp_weights`. |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `vals` | `float` or `np.ndarray` | Interpolated value(s). Scalar if `lon`/`lat` are scalar, otherwise same shape as `lon`. |

### Notes

- `hpx_map` must be in **NESTED** order and contain exactly `12 * 4**depth`
  elements. Convert a RING map first: `hp.reorder(m, r2n=True)`.
- Equivalent to `healpy.get_interp_val(m, theta, phi, nest=True, lonlat=True)`
  for `ellipsoid="sphere"`.

---
## Internal functions (not public API)

These functions are implementation details.  Their signatures may change between
versions.

| Function | Description |
|----------|-------------|
| `_gnomonic_project(lon_ref_deg, lat_ref_deg, lon_deg, lat_deg)` | Gnomonic (tangent-plane) projection of points relative to a reference point. Returns `(x, y)` in radians. |
| `_bilinear_weights_from_tangent_plane(sel_px, sel_py)` | Compute area-proportional bilinear weights from 4 tangent-plane cell positions, query at the origin. Returns shape `[N, 4]`, rows summing to 1. |

---

## Error handling

| Situation | Behaviour |
|-----------|-----------|
| `lon.shape != lat.shape` in `get_interp_weights` | `ValueError` |
| Query point falls exactly on a cell boundary (degenerate tangent-plane extent) | Weights default to `0.25` each (uniform fallback) |
| `hpx_map` size does not match `12 * 4**depth` | `IndexError` when indexing `hpx_map[pixels]` |
| `lon`/`lat` passed as scalars to `get_interp_val` | Returns a Python `float` instead of an array |

---

## Type reference

````python
# Input coordinates accept any of:
ArrayLike = Union[np.ndarray, float, List[float]]

# All returns:
pixels        : np.ndarray  uint64   [N, 4]
weights       : np.ndarray  float64  [N, 4]
vals          : Union[float, np.ndarray]  float64
````
