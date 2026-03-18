# `HealPixConv` — Gauge-equivariant spherical convolution

**Module** `healpix_analyse.convol`  
**Class** `HealPixConv(nside, in_channels, out_channels, ...)`  
**Inherits from** `torch.nn.Module`

---

## Principle

`HealPixConv` is a **gauge-equivariant** spherical convolution.  The key
idea is that the kernel is defined **once at the North Pole** as a regular
grid of `P = kernel_sz²` points, then this grid is **rotated** to each
target pixel by a gauge rotation matrix.  The result is that the
convolution kernel has a consistent orientation at every pixel of the
sphere — it is not based on HEALPix's own topology (which has no
canonical N/E/S/W).

### The three stages (precomputed at construction)

**A — Geometry**

A `kernel_sz × kernel_sz` grid of unit vectors is placed at the North
Pole, spaced by one HEALPix pixel size (`hp.nside2resol`).  For each
target pixel `k` at colatitude `θ_k`, longitude `φ_k`, the full
rotation matrix is:

$$
R_{\text{total}}(k, g) = R_{\text{gauge}}(\alpha_g) \cdot R_z(\phi_k) \cdot R_y(\theta_k)
$$

where the gauge angle for orientation `g` is:

$$
\alpha_g = \alpha_{\text{base}}(k) + g \cdot \frac{\pi}{G}
$$

and the base angle depends on the `gauge_type`:

| `gauge_type` | `alpha_base` | Best for |
|---|---|---|
| `"phi"` | 0 (meridian-aligned) | Earth-observation, NWP |
| `"cosmo"` | $2 \cdot \text{sign}(\theta - \pi/2) \cdot \phi$ | CMB, cosmology |

For `n_gauges = G > 1`, `G` evenly-spaced orientations are computed per
pixel.  The kernel is **shared** across gauges; the G responses are
concatenated along the channel dimension.

**B — Binding**

For each rotated stencil point, `healpy.get_interp_weights` returns 4
bilinear-interpolation neighbors and weights.  When the convolution
covers a **partial-sky patch** (`cell_ids` given), out-of-patch neighbors
are zeroed and the remaining weights are renormalized to sum to 1.
Stencil points with zero total weight fall back to the center pixel.

**C — Convolution**

The gathered, interpolated values form a tensor `[B, C_in, K, P]`.  The
kernel `W[G, C_in, C_out, P]` is applied via:

$$
y_{b, g \cdot C_{out} + o,\, k}
= \text{bias}_{g \cdot C_{out}+o}
+ \sum_{c=0}^{C_{in}-1} \sum_{p=0}^{P-1} W_{g,c,o,p} \cdot x^{\text{interp}}_{b,c,k,p}
$$

---

## Input / output shapes

| Input | Output | Notes |
|---|---|---|
| `[N]` | `[G*C_out, N]` (or `[N]` if `G=C_out=1`) | numpy or torch |
| `[B, N]` | `[B, G*C_out, N]` | single-channel batch |
| `[B, C_in, N]` | `[B, G*C_out, N]` | multi-channel batch |

The return type mirrors the input type (numpy → numpy, torch → torch).

---

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside` | `int` | — | HEALPix resolution (power of 2). |
| `in_channels` | `int` | — | Input channels `C_in`. |
| `out_channels` | `int` | — | Output channels per gauge `C_out`. Total = `G * C_out`. |
| `kernel_sz` | `int` | `3` | Odd integer ≥ 1.  `P = kernel_sz²` stencil points. |
| `n_gauges` | `int` | `1` | Number of gauge orientations `G`. |
| `gauge_type` | `{"phi","cosmo"}` | `"phi"` | Gauge convention. |
| `cell_ids` | array-like or `None` | `None` | Pixel indices (NESTED) for partial sky.  `None` = full sphere. |
| `level` | `int` or `None` | `None` | `nside = 2**level`.  Required with `cell_ids`. |
| `nest` | `bool` | `True` | NESTED ordering. |
| `use_norm` | `bool` | `False` | GroupNorm + ReLU after convolution. |
| `device` | device or `None` | auto | Torch device. |
| `dtype` | `torch.dtype` | `float32` | Dtype for kernel parameters. |

---

## Learnable attributes

| Attribute | Shape | Description |
|---|---|---|
| `weight` | `[G, C_in, C_out, P]` | Kernel shared across pixels, applied per gauge. Initialised with Kaiming uniform. |
| `bias` | `[G * C_out]` | Per-output-channel bias. Initialised to zero. |

---

## `set_kernel` method

```python
conv.set_kernel(W, bias=None, requires_grad=False)
```

Replace the learnable kernel with a fixed (or re-initialised) array.

**Parameters**

| | Type | Description |
|---|---|---|
| `W` | array-like `[C_in, C_out, P]` or `[G, C_in, C_out, P]` | Kernel values.  `[C_in, C_out, P]` is broadcast over all G gauges. |
| `bias` | array-like `[G*C_out]` or `None` | If `None`, bias is reset to zero. |
| `requires_grad` | `bool`, default `False` | Set to `True` to fine-tune from this initialisation. |

**Returns** `self` (chainable).

---

## Stencil point indexing

The `P = kernel_sz²` stencil positions are laid out row-major from the
top-left of the local grid.  For `kernel_sz=3`:

```
p =  0   1   2
     3   4   5     ← p=4 is the CENTER (kernel_sz//2 * (kernel_sz+1))
     6   7   8
```

This indexing is used when setting fixed kernel weights manually.

---

# Case 1 — Learned kernels: U-Net on the sphere

## Architecture

```
Input [B, C_in, N]
  │
  ▼  HealPixConv  nside=64, 1→32 ch, use_norm=True
  ▼
[B, 32, N]  ─────────────────────────────────┐  skip s1
  │                                          │
  ▼  HealPixDown nside=64                    │
[B, 32, N/4]                                 │
  │                                          │
  ▼  HealPixConv  nside=32, 32→64 ch         │
[B, 64, N/4]  ───────────────────┐ skip s2   │
  │                              │           │
  ▼  HealPixDown nside=32        │           │
[B, 64, N/16]  ← bottleneck      │           │
  │                              │           │
  ▼  HealPixConv  nside=16, 64→128 ch        │
  │                              │           │
  ▼  HealPixUp   nside=16        │           │
[B, 128, N/4]                    │           │
  │  cat(s2) ────────────────────┘           │
[B, 192, N/4]                                │
  │                                          │
  ▼  HealPixConv  nside=32, 192→64 ch        │
  │                                          │
  ▼  HealPixUp   nside=32                    │
[B, 64, N]                                   │
  │  cat(s1) ───────────────────────────────-┘
[B, 96, N]
  │
  ▼  HealPixConv  nside=64, 96→32 ch
  │
  ▼  HealPixConv  nside=64, 32→C_out ch, kernel_sz=1  (1×1 output head)
  │
Output [B, C_out, N]
```

## Complete example

```python
import torch
import torch.nn as nn
import numpy as np
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown
from healpix_analyse.up     import HealPixUp


class SphereUNet(nn.Module):
    """2-level spherical U-Net on the full HEALPix sphere."""

    def __init__(self, nside: int, in_channels: int, out_channels: int):
        super().__init__()

        # Encoder
        self.enc1  = HealPixConv(nside,    in_channels,  32, kernel_sz=3, use_norm=True)
        self.down1 = HealPixDown(nside,    mode="smooth")
        self.enc2  = HealPixConv(nside//2, 32,           64, kernel_sz=3, use_norm=True)
        self.down2 = HealPixDown(nside//2, mode="smooth")

        # Bottleneck
        self.bottle = HealPixConv(nside//4, 64, 128, kernel_sz=3, use_norm=True)

        # Decoder
        self.up2   = HealPixUp(nside//4)
        self.dec2  = HealPixConv(nside//2, 128+64, 64, kernel_sz=3, use_norm=True)
        self.up1   = HealPixUp(nside//2)
        self.dec1  = HealPixConv(nside,    64+32,  32, kernel_sz=3, use_norm=True)

        # Output head: 1×1 convolution = channel projection, no spatial mixing
        self.head  = HealPixConv(nside, 32, out_channels, kernel_sz=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)                      # [B, 32, N]
        x2, _ = self.down1(s1)                 # [B, 32, N/4]
        s2 = self.enc2(x2)                     # [B, 64, N/4]
        x3, _ = self.down2(s2)                 # [B, 64, N/16]

        # Bottleneck
        xb = self.bottle(x3)                   # [B, 128, N/16]

        # Decoder with skip connections
        xu2, _ = self.up2(xb)                  # [B, 128, N/4]
        xd2 = self.dec2(torch.cat([xu2, s2], dim=1))   # [B, 64, N/4]
        xu1, _ = self.up1(xd2)                 # [B, 64, N]
        xd1 = self.dec1(torch.cat([xu1, s1], dim=1))   # [B, 32, N]

        return self.head(xd1)                  # [B, out_channels, N]


# ---- instantiate ----
nside = 64
model = SphereUNet(nside=nside, in_channels=2, out_channels=1)

N = 12 * nside**2
x = torch.randn(4, 2, N)
y = model(x)
print(f"Input  {tuple(x.shape)}")   # (4, 2, 49152)
print(f"Output {tuple(y.shape)}")   # (4, 1, 49152)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

## Multi-gauge variant

Setting `n_gauges=G` produces G rotated copies of the same kernel,
concatenated as additional output channels.  This is useful to build
rotation-equivariant features.

```python
# 4 gauges: 0°, 45°, 90°, 135° rotations of the same 3×3 kernel
conv_g4 = HealPixConv(
    nside=64, in_channels=2, out_channels=8,
    kernel_sz=3, n_gauges=4, use_norm=True,
)
x = torch.randn(4, 2, 12 * 64**2)
y = conv_g4(x)
print(y.shape)    # (4, 32, 49152)  — G * C_out = 4 * 8
```

## Training loop

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    pred = model(x_batch)        # [B, C_out, N]
    loss = criterion(pred, y_true)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}   loss={loss.item():.5f}")
```

## Inspecting learned kernels

After training, kernel weights are standard `nn.Parameter` tensors.

```python
# weight shape: [G, C_in, C_out, P]
W = model.enc1.weight.detach().cpu().numpy()   # [1, 1, 32, 9]

# For gauge 0, input ch 0, output ch 3: show the 9 spatial weights
print(W[0, 0, 3, :])
# [ p0  p1  p2 ]
# [ p3  p4  p5 ]    ← p4 is the center
# [ p6  p7  p8 ]
```

---

# Case 2 — Fixed kernels: hand-crafted spherical filters

`HealPixConv` doubles as a **fixed spatial filter** by calling
`conv.set_kernel(W)` after construction.  The kernel is then frozen
(`requires_grad=False` by default) and no longer updated during training.

This is useful for:
- classical filtering (smoothing, sharpening, edge detection),
- physics-motivated priors (Laplacian, gradient, divergence),
- analysis pipelines that compute local statistics.

## Stencil point reference

For `kernel_sz=3`, `P=9`:

```
p =  0   1   2
     3   4   5     p=4 is the CENTER
     6   7   8
```

> **Important**: the geometric meaning of `p=0..8` depends on the gauge.
> With `gauge_type="phi"` and the default North-Pole grid, the layout
> is approximately:
>
> ```
>  NW   N   NE
>   W  ctr   E
>  SW   S   SE
> ```
>
> but this orientation rotates smoothly across the sphere following the
> meridian.  For a cosmological survey use `gauge_type="cosmo"`.

## Example A — Isotropic Gaussian smoothing

```python
import numpy as np
from healpix_analyse.convol import HealPixConv

nside = 64
conv = HealPixConv(nside=nside, in_channels=1, out_channels=1, kernel_sz=3)

# 9-point Gaussian: center gets more weight than the 8 neighbours
sigma     = 1.0   # in pixel-size units
w_center  = np.exp(-0.0 / (2 * sigma**2))   # distance 0
w_ring    = np.exp(-1.0 / (2 * sigma**2))   # distance ~1 pixel

W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]  = w_center          # center (p=4)
W[0, 0, [0,1,2,3,5,6,7,8]] = w_ring   # 8 neighbours
W /= W.sum()                     # normalise so output = weighted mean

conv.set_kernel(W)

import numpy as np
sky = np.random.randn(12 * nside**2).astype(np.float32)
sky_smooth = conv(sky)     # returns np.ndarray [N]
print(sky_smooth.shape)    # (49152,)
```

## Example B — Discrete Laplacian (edge detection)

With the isotropic 9-point stencil, the discrete Laplacian weights the
8 neighbours equally with -1 and the center with +8:

```python
conv_lap = HealPixConv(nside=nside, in_channels=1, out_channels=1, kernel_sz=3)

W_lap = np.zeros((1, 1, 9), dtype=np.float32)
W_lap[0, 0, 4]  =  8.0    # center
W_lap[0, 0, [0,1,2,3,5,6,7,8]] = -1.0   # neighbours

conv_lap.set_kernel(W_lap)

edges = conv_lap(sky)      # highlights spatial gradients
```

## Example C — Approximate directional gradients

Using `gauge_type="phi"` with `kernel_sz=3`, stencil points approximate:

```
p : 0=NW  1=N  2=NE
    3=W    4=ctr  5=E
    6=SW  7=S  8=SE
```

```python
conv_grad = HealPixConv(
    nside=nside, in_channels=1, out_channels=2,
    kernel_sz=3, gauge_type="phi",
)

W_grad = np.zeros((1, 2, 9), dtype=np.float32)
# Output channel 0: approximate N–S gradient  (N minus S)
W_grad[0, 0, 1] = +1.0   # N
W_grad[0, 0, 7] = -1.0   # S
# Output channel 1: approximate E–W gradient  (E minus W)
W_grad[0, 1, 5] = +1.0   # E
W_grad[0, 1, 3] = -1.0   # W

conv_grad.set_kernel(W_grad)

# sky [N] → grad [2, N]
grad = conv_grad(sky)
grad_NS = grad[0]   # dT/dlat
grad_EW = grad[1]   # dT/dlon
```

## Example D — Multi-channel co-convolution

Apply several fixed filters in one pass to a multi-channel input.

```python
C = 4   # e.g. T, U, V, Q

conv_multi = HealPixConv(
    nside=nside, in_channels=C, out_channels=2 * C,
    kernel_sz=3,
)

# W shape: [G=1, C_in=4, C_out=8, P=9]
W_multi = np.zeros((1, C, 2 * C, 9), dtype=np.float32)

for c in range(C):
    # First C output channels: Gaussian smoothing of input channel c
    W_multi[0, c, c, 4]  = w_center          # center
    W_multi[0, c, c, [0,1,2,3,5,6,7,8]] = w_ring
    W_multi[0, c, c, :]  /= W_multi[0, c, c, :].sum()

    # Next C output channels: Laplacian of input channel c
    W_multi[0, c, c + C, 4]  =  8.0
    W_multi[0, c, c + C, [0,1,2,3,5,6,7,8]] = -1.0

conv_multi.set_kernel(W_multi)

# [B, 4, N] → [B, 8, N]
maps = np.random.randn(8, C, 12 * nside**2).astype(np.float32)
result = conv_multi(maps)
print(result.shape)           # (8, 8, 49152)

smooth  = result[:, :C,  :]  # smoothed channels
laplace = result[:, C:,  :]  # Laplacian channels
```

## Example E — Partial-sky patch

Everything above works identically on a sky patch.

```python
import healpy as hp

nside = 128
level = 7       # nside = 2**7 = 128

# 15° disc around the Galactic centre
vec   = hp.ang2vec(np.pi / 2, 0.0)
patch = hp.query_disc(nside, vec, np.radians(15.0), nest=True)

conv_patch = HealPixConv(
    nside=nside, in_channels=1, out_channels=1,
    kernel_sz=3, cell_ids=patch, level=level,
)

# Gaussian smoothing, fixed
conv_patch.set_kernel(W)    # same W as Example A

sky_patch = np.random.randn(len(patch)).astype(np.float32)
sky_smooth = conv_patch(sky_patch)
print(sky_smooth.shape)     # (len(patch),)
```

## Example F — Combining fixed pre-processing with learned features

A fixed gradient filter followed by a learned block:

```python
class PhysicsGradientBlock(nn.Module):
    """
    Fixed gauge-equivariant gradient → learned channel extractor.
    """

    def __init__(self, nside: int, out_channels: int):
        super().__init__()

        # Fixed: 1 input → 2 gradient channels (N-S and E-W)
        self.grad = HealPixConv(nside=nside, in_channels=1, out_channels=2,
                                kernel_sz=3, gauge_type="phi")
        self.grad.set_kernel(W_grad)      # from Example C, requires_grad=False

        # Learned: 2 gradient channels → learned features
        self.feat = HealPixConv(nside=nside, in_channels=2, out_channels=out_channels,
                                kernel_sz=3, use_norm=True)

    def forward(self, x):
        g = self.grad(x)     # [B, 2, N]  fixed, no gradient
        return self.feat(g)  # [B, out_channels, N]  learned


block = PhysicsGradientBlock(nside=64, out_channels=16)
y = block(torch.randn(4, 1, 12 * 64**2))
print(y.shape)   # (4, 16, 49152)

# Only the learned part has parameters:
trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")
```

---

## Multi-gauge fixed filters

With `n_gauges=G`, `set_kernel` accepts shape `[C_in, C_out, P]` (same kernel
broadcast to all gauges) or `[G, C_in, C_out, P]` (different kernel per gauge).

```python
# Two-gauge Laplacian: isotropic (g=0) and rotated-45° (g=1)
conv_2g = HealPixConv(nside=64, in_channels=1, out_channels=1, n_gauges=2)

W_2g = np.zeros((2, 1, 1, 9), dtype=np.float32)
# Both gauges: same isotropic Laplacian (rotation-invariant)
for g in range(2):
    W_2g[g, 0, 0, 4]  =  8.0
    W_2g[g, 0, 0, [0,1,2,3,5,6,7,8]] = -1.0

conv_2g.set_kernel(W_2g)

y = conv_2g(sky)
print(y.shape)   # (2, 49152)  — G * C_out = 2 * 1
```

---

## Summary: kernel shape reference

| `n_gauges` | `set_kernel(W)` shape | Meaning |
|---|---|---|
| 1 | `[C_in, C_out, P]` | Single gauge, broadcast. |
| G | `[C_in, C_out, P]` | Same kernel, broadcast to all G gauges. |
| G | `[G, C_in, C_out, P]` | Different kernel per gauge. |

The `weight` parameter always has shape `[G, C_in, C_out, P]` internally.
