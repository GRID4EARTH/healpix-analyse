# HEALPix-to-HEALPix resampling

`resample_healpix` converts NESTED HEALPix data between Grid4Earth levels and
between full or partial cell domains. It is implemented as a sequence of the
package's local `HealPixDown` or `HealPixUp` operators.

`HealPixResampler` provides the same operation as a reusable PyTorch module.
Use the class when several maps share the same input and output geometries, so
the sparse operators are constructed only once.

## Quick start

```python
from healpix_analyse import resample_healpix

out_data, returned_ids = resample_healpix(
    in_data,
    in_level=10,
    out_level=7,
    in_cell_ids=in_cell_ids,
    out_cell_ids=out_cell_ids,
    ellipsoid="sphere",
)
```

`returned_ids` always has the same order as `out_cell_ids`. If
`out_cell_ids=None`, it contains every output cell in canonical NESTED order.

## Full and partial domains

The two cell-ID arguments are independent:

| Argument | `None` | Array provided |
|---|---|---|
| `in_cell_ids` | Input is the complete sphere at `in_level` | Last input dimension matches these IDs |
| `out_cell_ids` | Return the complete sphere at `out_level` | Return exactly these cells in this order |

`in_level` and `out_level` are always required. A full-sphere input is selected
with `in_cell_ids=None`, not with `in_level=None`, because the resolution must
remain known.

Input IDs may be in any order. Output IDs may also be in any order, and that
order is preserved exactly. Both arrays must contain unique valid NESTED IDs.

## The three resolution cases

### Same level

When `in_level == out_level`, no interpolation is performed. Values are joined
directly by NESTED identifier. Requested cells absent from a partial input are
filled with `NaN`.

### Resolution reduction

When `in_level > out_level`, the operator constructs

```text
level L -> Down -> level L-1 -> ... -> requested output level
```

Every step uses `HealPixDown(mode="smooth")`. For partial maps, the exact
coarse IDs are derived from the available fine cells. The final result is then
joined to the requested output domain.

### Resolution increase

When `in_level < out_level`, every requested output ID is first mapped to its
input-level NESTED ancestor:

```python
ancestor_ids = out_cell_ids // 4 ** (out_level - in_level)
```

Only ancestors present in the input are retained. `HealPixUp` is then applied
successively, keeping the complete descendant tree of every retained ancestor.
The final data are gathered using the exact requested IDs and order. If an
ancestor is absent, all requested descendants relying on it remain `NaN`.

## Missing data and coastlines

Non-finite input values are treated as missing samples. At every Up or Down
step, the implementation applies the same sparse operator to:

1. the data with missing samples replaced by zero;
2. a finite-data mask;
3. an all-valid reference mask, precomputed when the resampler is built.

The result is renormalised by the available weighted support. Consequently:

- a coastal cell remains calculable from the available ocean samples;
- a channel may have a different validity mask from another channel;
- a target with no non-zero support is `NaN`;
- when all inputs are finite, the result is the unchanged output of the
  underlying `HealPixDown` or `HealPixUp` operator.

This behaviour follows the local masked-map philosophy of `healpix-analyse`.

## Reusable operator

```python
import torch
from healpix_analyse import HealPixResampler

resampler = HealPixResampler(
    in_level=11,
    out_level=8,
    in_cell_ids=ocean_ids_level_11,
    out_cell_ids=comparison_ids_level_8,
    ellipsoid="sphere",
    dtype=torch.float64,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

temperature_8, ids_8 = resampler(temperature_11)
velocity_8, ids_8 = resampler(velocity_11)
```

The geometry, sparse matrices, ID joins and full-support normalisations are
reused by both calls.

## Shapes, NumPy and Torch

The last dimension is always the HEALPix pixel dimension. Arbitrary leading
dimensions are retained:

```text
[N]       -> [N_out]
[C, N]    -> [C, N_out]
[B, C, N] -> [B, C, N_out]
```

NumPy input returns a NumPy array. Torch input returns a differentiable tensor
on the configured device. Output values use the configured floating-point
`dtype`, since integer arrays cannot represent `NaN`.

## Filtering options

The following parameters are forwarded to the underlying operators:

| Parameter | Default | Meaning |
|---|---:|---|
| `ellipsoid` | `"WGS84"` | HEALPix geometry |
| `radius_deg` | level-dependent | Gaussian support radius |
| `sigma_deg` | level-dependent | Gaussian width |
| `weight_norm` | `"l1"` | Down normalisation |
| `up_norm` | `"col_l1"` | Up normalisation |

Leaving `radius_deg` and `sigma_deg` unset lets every step select a filter
appropriate to its own native level. Supplying them uses the same angular
filter at every step.

## Function signature

```python
out_data, out_ids = resample_healpix(
    in_data,
    in_level=...,
    out_level=...,
    in_cell_ids=None,
    out_cell_ids=None,
    ellipsoid="WGS84",
    radius_deg=None,
    sigma_deg=None,
    weight_norm="l1",
    up_norm="col_l1",
    dtype=torch.float32,
    device=None,
)
```

The arguments after `in_data` are keyword-only to prevent accidental exchange
of input/output levels or cell-ID arrays.

## Performance considerations

Requesting the complete output sphere necessarily allocates
`12 * 4**out_level` identifiers and output values. Full-sphere Down operations
also construct full-sphere sparse operators. For repeated comparisons, build a
single `HealPixResampler` and reuse it.

For partial upsampling, only the input-level ancestors required by the output
cells and their descendant trees are propagated. This follows NESTED hierarchy
semantics and avoids constructing unrelated output regions.
