# HealPixConv — API reference

Parameters, methods, tensor shapes and cost of `HealPixConv`. The guide is
{doc}`convol_doc`; the geometry behind it is {doc}`convol_internals`.

---

## The layer

### Constructor parameters

```python
HealPixConv(
    level,
    in_channels,
    out_channels,
    kernel_sz          = 3,
    n_gauges           = 1,
    gauge_type         = "phi",
    singularity_lonlat = None,
    ref_direction      = None,
    cell_ids           = None,
    nest               = True,
    use_norm           = False,
    device             = None,
    ellipsoid          = "WGS84",
    dtype              = torch.float32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `int` | — | Grid4Earth/HEALPix level, integer ≥ 0. Internally, `nside = 2**level`. |
| `in_channels` | `int` | — | Number of input feature channels C_in. |
| `out_channels` | `int` | — | Output channels per gauge C_out. Total output channels = G × C_out. |
| `kernel_sz` | `int` | `3` | Stencil side length. Must be a positive odd integer. P = kernel_sz². |
| `n_gauges` | `int` | `1` | Number of gauge orientations G. Gauge g rotates the stencil by g·π/G. |
| `gauge_type` | `str` | `"phi"` | One of `"phi"`, `"cosmo"`, `"projected_ref"`, `"two_ref"`. See the section. |
| `singularity_lonlat` | `(lon, lat)` or `[(lon₁,lat₁),(lon₂,lat₂)]` or `None` | `None` | Geographic coordinates of the desired singularity point(s) in degrees. For `"projected_ref"`: one pair. For `"two_ref"`: a list of two pairs. Overrides `ref_direction`. |
| `ref_direction` | `array (3,)` or `(2, 3)` or `None` | `None` | Low-level alternative: raw unit reference vector(s). Shape `(3,)` for `"projected_ref"`, `(2, 3)` for `"two_ref"`. Ignored when `singularity_lonlat` is provided. |
| `cell_ids` | `array-like` or `None` | `None` | NESTED pixel indices for a partial-sky patch. `None` = full sphere. |
| `nest` | `bool` | `True` | Pixel ordering of the input map. `True` = NESTED, `False` = RING. |
| `use_norm` | `bool` | `False` | Apply GroupNorm + ReLU after the convolution. |
| `device` | `str` or `torch.device` or `None` | `None` | Target device. Defaults to CUDA if available, else CPU. |
| `ellipsoid` | `str` | `"WGS84"` | Reference ellipsoid for healpix_geo coordinate conversions. |
| `dtype` | `torch.dtype` | `torch.float32` | Floating-point precision for all buffers and parameters. |

**Raises:**

- `ValueError` — if `level` is not a non-negative integer.
- `ValueError` — if `kernel_sz` is not a positive odd integer.
- `ValueError` — if `gauge_type` is not one of the four valid strings.
- `ValueError` — if `cell_ids` contains identifiers outside the selected
  level's valid pixel range.
- `ValueError` — if `singularity_lonlat` is used with a gauge type that
  does not support it, or if the wrong number of pairs is provided for
  `"two_ref"`.

---

### `forward(x)`

Apply the gauge-equivariant spherical convolution.

```python
y = conv(x)
```

**Parameters:**

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `x` | `np.ndarray` or `torch.Tensor` | `[N]`, `[B, N]`, or `[B, C_in, N]` | Input map(s) at the HEALPix resolution of this layer. N must equal `len(cell_ids)` for partial-sky, or `12 · nside²` for full sphere. |

**Returns:**

| Name | Type | Shape | Description |
|---|---|---|---|
| `y` | same type as `x` | `[G·C_out, N]` or `[B, G·C_out, N]` | Convolved output. Output type and number of dimensions match the input. |

**Raises:**

- `ValueError` — if `x` has the wrong number of channels or pixels.

---

### `set_kernel(W, bias=None, requires_grad=False)`

Replace the learnable kernel with a fixed (or re-initialised) array.
Returns `self` for chaining.

```python
conv.set_kernel(W, bias=None, requires_grad=False)
```

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `W` | `array-like` | `[C_in, C_out, P]` or `[G, C_in, C_out, P]` | Kernel weights. The first form broadcasts the same kernel over all G gauges. |
| `bias` | `array-like` or `None` | `[G · C_out]` | Bias vector. `None` resets the bias to zero. |
| `requires_grad` | `bool` | — | If `True`, the kernel and bias remain learnable after this call. |

**Example — isotropic Gaussian smoothing (kernel_sz=3):**

```python
W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]              = 0.5          # centre point (index 4 in a 3×3 grid)
W[0, 0, [0,1,2,3,5,6,7,8]] = 0.5 / 8  # 8 neighbours equally weighted
conv.set_kernel(W)
```

---

### `singularity_info()`

Return a human-readable string describing where the gauge singularities are
placed.

```python
print(conv.singularity_info())
```

Example output for `"two_ref"`:

```
gauge_type='two_ref':
  singularity 1  : lon=-55.00°  lat=-10.00°  (index +1, user-defined)
  singularity 1b : lon=+125.00°  lat=+10.00°  (index +1, antipode of 1)
  singularity 2  : lon=+20.00°  lat=+10.00°  (index +1, user-defined)
  singularity 2b : lon=+200.00°  lat=-10.00°  (index +1, antipode of 2)
  N/S poles      : index -1 each (side-effect, keep outside domain of interest)
  ref_directions : r1=[...]
                   r2=[...]
```

---

## Tensor shapes

| Symbol | Meaning |
|---|---|
| B | Batch size |
| C_in | Input channels |
| C_out | Output channels per gauge |
| G | Number of gauges (`n_gauges`) |
| K | Number of pixels (`len(cell_ids)` or `12·nside²`) |
| P | Stencil points (`kernel_sz²`) |

| Buffer / tensor | Shape | Description |
|---|---|---|
| `_pos_safe` | `[G, 4, K·P]` | Column indices into the sorted input for each bilinear neighbour |
| `_w_norm` | `[G, 4, K·P]` | Renormalised bilinear weights |
| `_sort_order` | `[K]` | Index that sorts pixel ids ascending |
| `_inv_order` | `[K]` | Inverse permutation to restore original order |
| `weight` | `[G, C_in, C_out, P]` | Learned kernel (nn.Parameter) |
| `bias` | `[G · C_out]` | Learned bias (nn.Parameter) |

---

## Cost

### Constructor (one-time cost)

The geometry precomputation scales as O(K · G · P) and is dominated by the
`healpix_geo` coordinate lookup and the `get_interp_weights` call. For
typical configurations (nside=64, G=4, kernel_sz=3):

| Operation | K·G·P entries | Approx. time |
|---|---|---|
| `healpix_to_lonlat` | 49 152 | < 0.1 s |
| `get_interp_weights` | 49 152 × 4 × 9 = 1.77 M | ~0.5 s |
| `_bind_support_batched` | 1.77 M searchsorted | ~0.3 s |

### Forward pass (per batch)

The forward is a single `index_select` over `[G · 4 · K · P]` entries,
followed by a weighted sum and one `einsum`. On GPU (A100) with batch size
8, nside=64, G=4, C_in=16, C_out=32: typically under 5 ms.

### Memory

The dominant buffers are `_pos_safe` and `_w_norm`, each of shape
`[G, 4, K·P]`. For nside=128, G=4, kernel_sz=5 (P=25):

```
K = 12 × 128² = 196 608
Buffer size = 4 × 4 × 196 608 × 25 × 4 bytes (float32) ≈ 750 MB
```

For very large nside, consider reducing G or kernel_sz, or using
`torch.float16` for the buffers.

