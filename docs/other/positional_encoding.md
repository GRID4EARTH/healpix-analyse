# Positional encodings of HEALPix cells

**Module** `healpix_analyse.positional_encoding`  
**Layer** `HealPixPositionalEncoding`; **functions**
`healpix_positional_encoding`, `positional_encoding_cache_info`,
`clear_positional_encoding_cache`  
**Status** experimental

```{include} ../_generated/glance/positional_encoding.md
```

## What it computes

A transformer on HEALPix tokens, or a CNN that should know where it is,
needs the position of each cell. Every model used to rebuild its own; this
module computes one standard encoding per domain and caches it.

The cell centres come from `healpix-geo`, on the ellipsoid you choose
(`"WGS84"` by default, as everywhere in the package). On an ellipsoid
`healpix-geo` returns the geodetic latitude, so the unit vector is the
normal to the ellipsoid at the cell centre; on `"sphere"` it is the
direction of the cell centre:

```text
(x, y, z) = (cos lat · cos lon,  cos lat · sin lon,  sin lat)
```

The features, in this order (`feature_names` lists them):

| Features | Count | Option |
|---|---|---|
| `x, y, z` | 3 | `include_xyz=True` (default) |
| `sin(π f x), sin(π f y), sin(π f z), cos(π f x), cos(π f y), cos(π f z)` for each frequency f | 6 per frequency | `n_frequencies` or `frequencies` |
| `cos α, sin α`: the gauge angle | 2 | `gauge_type=` |
| `level / 29` | 1 | `include_level=True` |

- **Frequencies.** By default `1, 2, 4, …, 2**level` (`level + 1` of them,
  at most 16). The shortest wavelength, `2 / 2**level` in unit-vector units,
  is then about two cells. Pass `frequencies=[...]` for any other set.
- **Gauge angle.** `α` is the angle, in the tangent plane, from the local
  north `e_θ` to the first axis of the frame that
  {doc}`HealPixConv <../conv/convol>` uses with the same `gauge_type`
  (`"phi"`, `"cosmo"`, `"projected_ref"`, `"two_ref"`; `ref_direction` as
  there). It tells a model how its convolution kernels are oriented. It is
  0 everywhere for `"phi"`.
- **Level.** A constant channel, to tell the levels of a multi-level model
  apart.

## Usage

```python
import numpy as np
import torch
from healpix_analyse import HealPixPositionalEncoding, healpix_positional_encoding

# Full sphere, level 8: [F, N] with F = 3 + 6 * 9
enc = healpix_positional_encoding(8)

# Regional domain, cells in any order: row k is cell_ids[k]
pe = HealPixPositionalEncoding(10, cell_ids, gauge_type="projected_ref",
                               include_level=True).to("cuda")
x = torch.randn(4, 32, len(cell_ids), device="cuda")
x_in = pe(x)                      # [4, 32 + pe.num_features, N]: x and the encoding
tokens = pe().T                   # [N, F], e.g. to add to a projection of the tokens
```

`HealPixPositionalEncoding` stores the encoding as a non-persistent buffer:
it follows `.to()`, `.cuda()` and `.half()`, and it is not written to the
`state_dict` (it is rebuilt from the parameters).

## Cache

The geometry of a domain (unit vectors and gauge angle, float64, on the
CPU) is cached in memory, keyed by a SHA-256 hash of the `cell_ids` (their
order included), the level, the ellipsoid and the gauge. The cache keeps the
8 most recently used domains. The features themselves are a few
element-wise operations on the requested device.
`positional_encoding_cache_info()` reports hits and misses;
`clear_positional_encoding_cache()` empties it.

Memory: the encoding holds F × N values on the device, e.g. 57 × 786 432
float32 values (0.17 GiB) for the full sphere at level 8 with the default
frequencies.
