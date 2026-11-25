# HealPixConv — gauge-equivariant convolution

`HealPixConv` is a PyTorch `nn.Module` that convolves a signal on a HEALPix
sphere with a learned kernel, equivariantly: the same stencil is applied at
every cell, transported into a local frame, so the result does not depend on
how the pixel grid happens to be oriented there.

It works on full-sky maps and on partial-sky patches, takes NumPy arrays or
Torch tensors, and is differentiable end to end.

This page is the guide and the reference (parameters, shapes, cost). How the
geometry is built is in the developer notes: {doc}`/dev/convol_internals`.

---

## What it does

`HealPixConv` implements the spherical convolution described in the
*gauge-equivariant* framework: a fixed `kernel_sz × kernel_sz` stencil is
defined at the North Pole, then **rotated** to each target pixel by a
composition of spherical rotations. The same learned kernel is applied at
every pixel, in every gauge orientation, making the layer equivariant to
the chosen gauge group.

Key design choices:

- **The kernel never moves.** It is the *data* that is pulled toward the
  fixed kernel via bilinear interpolation of the input map, not the kernel
  that is warped per pixel.
- **All heavy geometry is precomputed once** at construction time and stored
  as registered buffers. The forward pass is purely index-and-multiply.
- **Fully vectorised**: no Python loops over pixels or gauges in either the
  constructor or the forward pass.
- **Differentiable**: every operation in the hot path (`index_select`,
  `einsum`, arithmetic) is supported by PyTorch autograd.

---

## Quick start

```python
import numpy as np
import torch
from healpix_analyse.convol import HealPixConv

level      = 6                         # Grid4Earth resolution level
nside      = 2**level                  # internal HEALPix resolution
npix       = 12 * nside ** 2           # 49 152 pixels
in_ch, out_ch = 3, 16

# --- build the layer (geometry precomputed here) ---
conv = HealPixConv(
    level       = level,
    in_channels = in_ch,
    out_channels= out_ch,
    kernel_sz   = 3,          # 3×3 = 9 stencil points
    n_gauges    = 4,          # G=4 gauge orientations
    gauge_type  = "projected_ref",
    singularity_lonlat = (84.0, 28.0),   # Himalayas + antipodal Pacific
)

# --- forward pass (differentiable) ---
x = torch.randn(8, in_ch, npix)         # batch of 8 maps
y = conv(x)                             # [8, G*out_ch, npix] = [8, 64, 49152]

# --- inspect singularity placement ---
print(conv.singularity_info())
```

---

## Gauges and singularities

The **hairy-ball theorem** states that every smooth tangent vector field on
S² must have at least one zero. Equivalently, the total index of all
singularities of a gauge field on S² must equal exactly 2.

Each built-in gauge places those unavoidable bad points differently.

### `"phi"` — meridian-aligned

```
α_base(k) = 0
```

The kernel is always aligned with the local meridian. Singularities are
fixed at the **geographic North and South Poles** (index +1 each, total = 2).

Best for data that is well-behaved away from the poles and where
interpretability matters. Computationally cheapest.

### `"cosmo"` — cosmological convention

```
α_base(k) = -φ_k   (Northern hemisphere, θ ≤ π/2)
α_base(k) = +φ_k   (Southern hemisphere, θ > π/2)
```

Same singularity locations as `"phi"` but the gauge angle flips sign
across the equator to match the CMB/cosmological convention used by
`healpy`. Useful when comparing results with legacy healpy-based pipelines.

### `"projected_ref"` — one freely placed antipodal pair

A reference vector **r** ∈ ℝ³ is projected onto the tangent plane at each
pixel `k`:

```
r_proj(k) = r − (r · n_k) · n_k
α_base(k) = atan2( r_proj · e_φ,  r_proj · e_θ )
```

The gauge is undefined exactly where `r_proj = 0`, i.e. where `r ∥ n_k`.
This occurs at the two **antipodal** points:

```
singularity₁ = (lon_s,        lat_s)       ← direction of r
singularity₂ = (lon_s + 180°, −lat_s)      ← antipode, forced
```

To place the first singularity at a desired location, pass:

```python
singularity_lonlat = (lon_s, lat_s)
```

The reference vector is computed automatically:

```
r = [cos(lat_s)·cos(lon_s),  cos(lat_s)·sin(lon_s),  sin(lat_s)]
```

**Constraint:** the second singularity is always the antipode of the first.
You cannot move them independently with this gauge type — use `"two_ref"`
for that.

### `"two_ref"` — two freely placed singularity pairs

Two independent reference vectors **r₁** and **r₂** define the gauge angle
via the **complex product** of their tangent-plane projections:

```
z_j(k) = (r_j_proj · e_θ)  +  i · (r_j_proj · e_φ)      j = 1, 2

α_base(k) = arg( z₁(k) · z₂(k) )
           = atan2( Re(z₁)·Im(z₂) + Im(z₁)·Re(z₂),
                    Re(z₁)·Re(z₂) − Im(z₁)·Im(z₂) )
```

Using the complex product avoids two independent `atan2` calls (which would
each wrap and accumulate phase jumps) and computes `arg(z₁) + arg(z₂)` in
a single numerically stable operation.

**Singularity structure (Poincaré–Hopf budget):**

| Location | Count | Index | Origin |
|---|---|---|---|
| `+r₁`, `−r₁` | 2 | +1 each | zeros of z₁ |
| `+r₂`, `−r₂` | 2 | +1 each | zeros of z₂ |
| N-Pole, S-Pole | 2 | −1 each | base-frame side-effect |
| **Total** | | **4−2 = 2** | ✓ |

The four user-controlled points are **index +1** (vortex-like), which is the
well-behaved type to place over unimportant regions. The geographic poles
become **index −1** (hyperbolic saddle), a sharper singularity — keep them
outside the domain of interest or over regions where accuracy is not
required.

To place singularities, pass a list of two `(lon, lat)` pairs:

```python
singularity_lonlat = [(lon_1, lat_1), (lon_2, lat_2)]
```

Each pair controls one **user-specified** singularity; its antipodal point
appears automatically. The full set of four bad points is thus:

```
{(lon₁, lat₁),  (lon₁+180°, −lat₁),  (lon₂, lat₂),  (lon₂+180°, −lat₂)}
```

### Choosing where the singularities go

| Domain | Strategy |
|---|---|
| **Ocean model** | Place all 4 bad points over land masses. Example: Amazon + Borneo (antipodal pair) and Africa + central Pacific (second pair). |
| **Atmosphere** | Place all 4 bad points over open ocean. Example: central Pacific + Indian Ocean (pair 1) and South Atlantic + Maritime Continent (pair 2). |
| **Full sphere / neutral** | Use `"phi"` (poles). For `"projected_ref"`, passing `singularity_lonlat=(0, 90)` reproduces the pole placement with a smoother field off the poles. |
| **Poles outside domain** | Any `"two_ref"` config naturally moves the geographic-pole index-−1 singularities to a fixed location — confirm they are harmless for your domain. |

---

## Recipes

### Full-sphere convolution, `"phi"` gauge

```python
conv = HealPixConv(level=6, in_channels=1, out_channels=32,
                   kernel_sz=3, n_gauges=1, gauge_type="phi")
y = conv(x)   # x: [B, 1, 49152]  →  y: [B, 32, 49152]
```

### Multi-gauge equivariant layer

```python
conv = HealPixConv(level=6, in_channels=16, out_channels=16,
                   kernel_sz=3, n_gauges=4, gauge_type="phi")
# output has G * C_out = 64 channels
```

### Ocean model — singularities over two land masses

```python
# Singularity pair 1: Amazon basin + Borneo (its antipode)
# Singularity pair 2: Africa + central Pacific (its antipode)
conv = HealPixConv(
    level=6, in_channels=3, out_channels=16,
    gauge_type="two_ref",
    singularity_lonlat=[(-55.0, -10.0), (20.0, 5.0)],
)
print(conv.singularity_info())
```

### Atmospheric model — singularities over open ocean

```python
conv = HealPixConv(
    level=6, in_channels=5, out_channels=32,
    gauge_type="projected_ref",
    singularity_lonlat=(-160.0, 0.0),   # central Pacific + Indian Ocean antipode
)
```

### Partial-sky patch

```python
import healpix_geo

level = 6
nside = 2**level
cell_ids, _, _ = healpix_geo.nested.cone_coverage(
    (0.0, 45.0), 20.0, level, ellipsoid="WGS84"
)
conv = HealPixConv(
    level=level, in_channels=1, out_channels=8,
    cell_ids=cell_ids,
)
x_patch = torch.randn(4, 1, len(cell_ids))
y_patch = conv(x_patch)
```

### Fixed (non-learnable) isotropic kernel

```python
conv = HealPixConv(level=5, in_channels=1, out_channels=1, kernel_sz=3)
W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]              = 0.5       # centre
W[0, 0, [0,1,2,3,5,6,7,8]] = 0.5/8 # ring
conv.set_kernel(W, requires_grad=False)
```

### With GroupNorm + ReLU

```python
conv = HealPixConv(
    level=6, in_channels=16, out_channels=16,
    n_gauges=4, use_norm=True,
)
# The output is already passed through GroupNorm then ReLU inside forward()
```

### NumPy in / NumPy out

```python
x_np = np.random.randn(49152).astype(np.float32)
y_np = conv(x_np)   # returns np.ndarray, same shape policy as input
```

---

## Reference: parameters, methods and shapes

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

### Tensor shapes

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

### Cost

#### Constructor (one-time cost)

The geometry precomputation scales as O(K · G · P) and is dominated by the
`healpix_geo` coordinate lookup and the `get_interp_weights` call. For
typical configurations (nside=64, G=4, kernel_sz=3):

| Operation | K·G·P entries | Approx. time |
|---|---|---|
| `healpix_to_lonlat` | 49 152 | < 0.1 s |
| `get_interp_weights` | 49 152 × 4 × 9 = 1.77 M | ~0.5 s |
| `_bind_support_batched` | 1.77 M searchsorted | ~0.3 s |

#### Forward pass (per batch)

The forward is a single `index_select` over `[G · 4 · K · P]` entries,
followed by a weighted sum and one `einsum`. On GPU (A100) with batch size
8, nside=64, G=4, C_in=16, C_out=32: typically under 5 ms.

#### Memory

The dominant buffers are `_pos_safe` and `_w_norm`, each of shape
`[G, 4, K·P]`. For nside=128, G=4, kernel_sz=5 (P=25):

```
K = 12 × 128² = 196 608
Buffer size = 4 × 4 × 196 608 × 25 × 4 bytes (float32) ≈ 750 MB
```

For very large nside, consider reducing G or kernel_sz, or using
`torch.float16` for the buffers.

