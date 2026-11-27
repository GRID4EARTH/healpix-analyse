# HealPixConv — gauge-equivariant convolution

`HealPixConv` is a PyTorch `nn.Module` that convolves a signal on a HEALPix
sphere with a learned kernel, equivariantly: the same stencil is applied at
every cell, transported into a local frame, so the result does not depend on
how the pixel grid happens to be oriented there.

```{include} ../_generated/glance/healpix_conv.md
```

It works on whole-sphere maps and on regional patches, takes NumPy arrays or
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
- **Differentiable**, with a hand-written backward that never stores the
  interpolated stencil values; fused Triton kernels on CUDA (see *Backends*
  below). Second-order gradients (`create_graph=True`) run on the same
  kernels, and `torch.func` (`vmap`, `jacrev`, `jacfwd`, `jvp`) works.
- **Strided and transposed**: `stride=2` evaluates the stencil only at the
  parent cells (the encoder step of a U-Net), and `HealPixConvTranspose` is
  its adjoint (the decoder step). See *Strided and transposed convolutions*
  below.

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

### Regional patch

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

### Missing pixels: `missing="local_mean"`

Each stencil value is a bilinear interpolation of 4 pixels. By default
(`missing="centre"`), neighbours outside `cell_ids` are dropped and the
others renormalised, and a stencil point with no neighbour in the domain
takes the value of the centre pixel; NaN values propagate.

With `missing="local_mean"`, a NaN pixel (or a pixel with `mask == 0`) is
missing exactly like a pixel outside `cell_ids`: a stencil point with at
least one valid neighbour takes the renormalised interpolation of its valid
neighbours, and a point with none takes the mean of the valid stencil values
of the same output pixel (weights `local_mean_weights`, uniform by default).
The geometry of this mode is always built in float64 and then cast to
`dtype`, because the choice between the two rules depends on the valid
fraction of each stencil point: a float32 build gives points just outside the
domain spurious fractions (up to 5e-4 at level 6, near the poles), enough to
switch a stencil value from the local mean to the value of one neighbour. Its
cache entry is the one of float64 layers.
Every pixel gets a finite output and finite gradients; the gradient at a NaN
pixel is 0. It costs a second gather (the mask), and it runs the PyTorch
reference path (the gathered `[B, C_in, G, K, P]` tensor and one matrix
product) on every device, not the fused kernels of *Backends* below. After
its first forward pass, the layer keeps the stencil in the layout of that
gather: int64 positions and the weights, `[4, G, K·P]` each, as the buffers of
earlier versions did. Without missing pixels, the result is the same as the
default.

```python
conv = HealPixConv(level=6, in_channels=1, out_channels=8, kernel_sz=5,
                   cell_ids=cell_ids, missing="local_mean")
y = conv(x_with_nan)                     # NaN completed, finite output
y = conv(x, mask=valid)                  # or an explicit mask in [0, 1]
z, cov = conv.completed_stencil(x_with_nan)   # [B, C, G, N, P] values and valid fraction
```

`completed_stencil` returns the values the kernel multiplies and the valid
fraction of every stencil point, from which the coverage of any kernel
follows; the wavelet power spectrum ({doc}`/sht/wavelet_spectrum`) uses it.

### Strided and transposed convolutions

An encoder block of a U-Net applies a convolution then a downsampling.
`HealPixConv(stride=2)` does both in one step: it evaluates the stencil only
at the centres of the parent cells (level `level - 1`), so it computes N/4
outputs instead of N and needs no `HealPixDown`. The output lies on the
parent cells, in the order of `conv.cell_ids_out` (sorted, as
`HealPixDown.cell_ids_out`; on a regional domain, every parent with at least
one child in `cell_ids`).

`HealPixConvTranspose` is its adjoint, for the decoder: with the same
geometry arguments and the same weights,
`<conv(x) - conv.bias, y> = <x, convT(y) - convT.bias>` holds exactly. It
reads gauge-major channels on the parent cells and writes `out_channels` on
the fine cells, in the order of `cell_ids`.

```python
from healpix_analyse.convol import HealPixConv, HealPixConvTranspose

level, G = 7, 4
enc = HealPixConv(level, 16, 32, kernel_sz=3, n_gauges=G, stride=2)       # level 7 -> 6
dec = HealPixConvTranspose(level, 32, 16, kernel_sz=3, n_gauges=G)        # level 6 -> 7

x = torch.randn(8, 16, 12 * 4**level)
h = enc(x)                      # [8, G*32, 12 * 4**6], on enc.cell_ids_out
z = dec(h)                      # [8, 16, 12 * 4**7]
dec.weight = enc.weight         # optional: tie the decoder to the encoder
```

**Definition of the strided output.** Output `k` is the stride-1
convolution evaluated at the centre of parent `k`, with the same stencil and
kernel: the stencil keeps the fine-level spacing (one level-`level` cell), it
is centred on the parent centre (the vertex shared by its 4 children) with
the gauge of that point, and its points are interpolated bilinearly from the
fine cells. The receptive field and the learned kernel are therefore those of
the stride-1 layer, and weights move between `stride=1` and `stride=2` layers
unchanged. Two other definitions were rejected:

- a spacing of two fine cells would skip fine cells between the stencil
  points (aliasing);
- a stencil centred on one of the 4 children would shift each output by half
  a fine cell, in a direction that changes from one base face to the next.

`stride=2` needs `kernel_sz ≥ 3`: with `kernel_sz=1` the only stencil point
is the parent centre, which lies on the ring of its East and West children, so
its bilinear weights are about 0.5 / 0.5 on those two children and ignore the
other two. That is not a mean of the 4 children, so the combination is
rejected; `HealPixDown(mode="mean")` is the mean.

`HealPixDown` has no mode that picks one value per parent, so the strided
layer is not bit-identical to a convolution followed by a downsampling. On a
smooth field it differs from the stride-1 layer followed by
`HealPixDown(mode="mean")` (the mean of the 4 children) at first order in the
cell size: 3 % of the amplitude at level 4, halved at each finer level
(`tests/test_healpix_conv_strided.py`).

Both layers reuse the geometry, the disk cache and the kernels of the stride-1
layer: the forward pass of `HealPixConvTranspose` is the input gradient of the
strided layer, and its input gradient is the strided forward pass. Both are
deterministic by default.

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
    cache_dir          = <default>,   # $HEALPIXCONV_CACHE or ~/.cache/healpixconv
    missing            = "centre",
    local_mean_weights = None,
    backend            = "auto",
    deterministic      = True,
    stride             = 1,
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
| `cell_ids` | `array-like` or `None` | `None` | NESTED pixel indices for a regional patch. `None` = full sphere. |
| `nest` | `bool` | `True` | Kept for compatibility. The geometry is always built on NESTED ids: the flag only enters the cache key and has no effect on the result. Pass NESTED data. |
| `use_norm` | `bool` | `False` | Apply GroupNorm + ReLU after the convolution. |
| `device` | `str` or `torch.device` or `None` | `None` | Target device. Defaults to CUDA if available, else CPU. |
| `ellipsoid` | `str` or `dict` | `"WGS84"` | Reference ellipsoid of the HEALPix grid. The grid of an ellipsoid is the grid of its authalic sphere: the stencil is centred on each cell and interpolated there for every ellipsoid; the ellipsoid only makes the stencil square in metres on the ground (offsets change by at most 0.11 % on WGS84). `"sphere"`: isotropic on the authalic sphere. |
| `dtype` | `torch.dtype` | `torch.float32` | Floating-point precision for all buffers and parameters. `module.to(...)` moves and casts the layer afterwards. |
| `backend` | `str` | `"auto"` | `"auto"`, `"triton"` or `"torch"`; see *Backends* below. |
| `deterministic` | `bool` | `True` | On CUDA: input gradient by a gather on the transposed stencil map, without atomics, identical at every run (Triton kernels, and the PyTorch backend). `False`: fp32 atomics (Triton) or `index_add_` (PyTorch), no extra memory. |
| `stride` | `int` | `1` | `2`: strided convolution, output on the parent cells (`cell_ids_out`); see *Strided and transposed convolutions*. Needs `level ≥ 1`, `kernel_sz ≥ 3` and `missing="centre"`. |
| `missing` | `str` | `"centre"` | `"centre"`: today's behaviour (points outside the domain take the centre value; NaN propagates). `"local_mean"`: NaN and points outside the domain are completed by the renormalised interpolation of the valid neighbours, or the local mean of the valid stencil values. |
| `local_mean_weights` | `array (P,)` or `None` | `None` | Non-negative weights of the stencil points in the local mean of `"local_mean"`; `None` = uniform. |

**Raises:**

- `ValueError` — if `level` is not a non-negative integer.
- `ValueError` — if `kernel_sz` is not a positive odd integer.
- `ValueError` — if `gauge_type` is not one of the four valid strings.
- `ValueError` — if `cell_ids` contains identifiers outside the selected
  level's valid pixel range.
- `ValueError` — if `singularity_lonlat` is used with a gauge type that
  does not support it, or if the wrong number of pairs is provided for
  `"two_ref"`.

### Backends

With `missing="centre"` (the default), the forward and backward passes never
build the gathered `[B, C_in, G, K, P]` tensor. With `backend="auto"`, CUDA tensors in float32,
float16 or bfloat16 run fused Triton kernels when the optional `triton`
package is installed (16-bit data accumulate in fp32; under `torch.autocast`
the kernels compute in the autocast dtype). Everything else (CPU, float64, no
Triton) runs a PyTorch loop over the P taps with a hand-written backward.

By default (`deterministic=True`), the Triton backend computes the input
gradient by a gather on the transposed stencil map: no atomics, the same bits
at every run, and, on an A100, a training step 1.6–2.2× shorter than with
atomics. The cost is memory. The map is built at the first backward pass
(0.02–0.2 s on an A100) and stays on the GPU, at about the size of the stencil
buffers again: 0.2–2.6 GiB for the configurations of
`benchmarks/benchmark_healpix_conv.py`. `module.to(...)` drops it, and the next
backward pass rebuilds it.

`deterministic=False` scatters the gradient with fp32 atomics instead. It
needs no extra memory, but the summation order, and so the last bits, vary
between runs. `torch.use_deterministic_algorithms(True)` selects the gather in
either case.

The PyTorch backend scatters the input gradient with `index_add_` on CPU,
which is deterministic there. On CUDA (without Triton) `index_add_` runs with
atomics, so with `deterministic=True` the PyTorch backend gathers on the
transposed map instead: the same bits at every run, with no
`torch.use_deterministic_algorithms` needed. That map is also built at the
first backward pass and kept (`nnz` entries of 4 + 4 bytes and two int32 per
cell and tap, about the size of the stencil buffers); `deterministic=False`
keeps `index_add_`.

The forward pass, the input gradient and the weight gradient are three
operators of the same geometry (the convolution, its adjoint, and a bilinear
weight gradient), each an autograd Function whose backward calls the others.
So a double backward (`create_graph=True`, for example a gradient penalty)
runs on the same kernels as the first one, Triton included, with the same
determinism. The Functions also define `jvp` and `vmap`: `torch.func.vmap`,
`jacrev`, `jacfwd`, `jvp` and `functional_call` work on the layer. A
`vmap` over the input folds the mapped dimension into the batch; any other
`vmap` (over the weights, say) runs one call per mapped element.

---

### `forward(x, mask=None)`

Apply the gauge-equivariant spherical convolution.

```python
y = conv(x)
y = conv(x, mask=valid)        # missing="local_mean" only
```

**Parameters:**

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `x` | `np.ndarray` or `torch.Tensor` | `[N]`, `[B, N]`, or `[B, C_in, N]` | Input map(s) at the HEALPix resolution of this layer. N must equal `len(cell_ids)` for a regional domain, or `12 · nside²` for the whole sphere. |
| `mask` | array-like or `None` | `[N]`, `[B, N]` or `[B, C_in, N]` | Only with `missing="local_mean"`: pixel weights in [0, 1] (0 = missing). `[N]` applies to every sample and channel, `[B, N]` to every channel of each sample; a per-channel mask must be `[B, C_in, N]` (B and C_in may be 1). Other shapes raise. NaN values of `x` are always missing. |

**Returns:**

| Name | Type | Shape | Description |
|---|---|---|---|
| `y` | same type as `x` | `[G·C_out, N_out]` or `[B, G·C_out, N_out]` | Convolved output. Output type and number of dimensions match the input. `N_out = N` for `stride=1`; for `stride=2`, the parent cells of `cell_ids_out`. |

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
| `_tap_idx` | `[P, 4, G, K]` (int32) | Position in the input of each bilinear neighbour of each tap (not in the `state_dict`); for `stride=2`, `K` is the number of parent cells `K_out` |
| `_tap_w` | `[P, 4, G, K]` (layer dtype) | Renormalised bilinear weights (not in the `state_dict`) |
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

See *Backends* above: one tap at a time, so the memory of a training step
is about the input and the output plus one tap, never the
`B · C_in · G · K · P` gathered values. `benchmarks/benchmark_healpix_conv.py`
measures the forward and backward passes and their peak memory on CPU or
CUDA.

#### Memory

The dominant buffers are `_tap_idx` (int32) and `_tap_w` (float32 for a
float32 layer), each of shape `[P, 4, G, K]`. For nside=128, G=4,
kernel_sz=5 (P=25):

```
K = 12 × 128² = 196 608
entries per buffer = 25 × 4 × 4 × 196 608 = 78 643 200
_tap_idx : 78 643 200 × 4 bytes ≈ 315 MB
_tap_w   : 78 643 200 × 4 bytes ≈ 315 MB
total                           ≈ 629 MB   (944 MB for a float64 layer)
```

On CUDA, the backward pass of the default `deterministic=True` adds the
transposed map, about the same size again; `deterministic=False` avoids it.
For very large nside, consider reducing G or kernel_sz.

#### Geometry cache on disk

The geometry (stencil rotation, bilinear binding) is saved after the first
construction and reloaded afterwards: about 0.8 s to build at nside 64,
G = 4, against about 5 ms to reload (figures from the code comments). The
cache directory is `cache_dir`, by default `$HEALPIXCONV_CACHE` or
`~/.cache/healpixconv`; `cache_dir=None` disables it. The key holds nside,
`kernel_sz`, G, the gauge type and reference directions, the ellipsoid, the
cell ids (their raw bytes, in order) and the precision of the build. The
geometry is always built in float64, whatever the layer dtype, and only the
final weights are cast to it: a float32 build lets a stencil point near a cell
or face boundary pick other neighbours on another device (CPU and CUDA round
`cos` and `matmul` differently), which changed the output of a regional layer
by tens of percent between CPU and CUDA. Layers of every dtype therefore share
one cache entry. `set_kernel` copies the kernel in the layer dtype.

#### Edges of a regional domain

A stencil point whose bilinear neighbours are not in `cell_ids` takes the
neighbours of the centre point of the same stencil: at the edge of a regional
domain the missing samples are silently replaced by the centre value. Include a
margin around the area of interest.

