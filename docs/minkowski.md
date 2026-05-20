# `minkowski` — Differentiable Minkowski Functionals

**Module** `healpix_analyse.minkowski`  
**Functions** `minkowski_functionals`, `minkowski_curves`

---

## Overview

Minkowski functionals are morphological descriptors from integral geometry
that capture the **geometry** and **topology** of a field at different
intensity levels.  In 2D, there are three:

| Functional | Symbol | Physical meaning |
|---|---|---|
| Area | W0 | Fraction of pixels above the threshold |
| Perimeter | W1 | Normalised boundary length |
| Euler characteristic | W2 | Connected components minus holes (χ) |

This module provides fully differentiable PyTorch implementations for
batched 2D images of shape ``[B, N, N]``.  Gradients flow back through
``img`` **and** through ``threshold`` (when it is a tensor), making both
functions suitable as loss terms or feature extractors in end-to-end
deep learning pipelines.

---

## Mathematical background

For a 2D field $f : \mathbb{R}^2 \to [0,1]$ and a threshold $t$, the
excursion set $A_t = \{x \mid f(x) \geq t\}$ has three Minkowski
functionals.  The soft (differentiable) approximation replaces the
indicator $\mathbf{1}[f \geq t]$ with a sigmoid
$\sigma(\tau(f - t))$ (temperature $\tau$).

### W0 — Area

$$W_0 = \frac{1}{N^2} \sum_{i,j} f_{ij}$$

### W1 — Perimeter

Approximated by the $L_1$ gradient norm (sum of absolute first-order
finite differences):

$$W_1 = \frac{1}{N^2} \sum_{i,j} \bigl(|f_{i,j+1} - f_{i,j}| + |f_{i+1,j} - f_{i,j}|\bigr)$$

For a hard binary image this equals the boundary length divided by $N^2$.

### W2 — Euler characteristic

Computed via the **pixel-complex formula** (4-connectivity, Hadwiger 1957).
Logical AND is approximated by the product of soft pixel values:

$$W_2 = \frac{Q_1 - Q_h - Q_v + Q_f}{N^2}$$

where:

| Term | Definition |
|---|---|
| $Q_1$ | $\sum_{i,j} f_{ij}$ (vertices) |
| $Q_h$ | $\sum_{i,j} f_{i,j} \cdot f_{i,j+1}$ (horizontal edge pairs) |
| $Q_v$ | $\sum_{i,j} f_{i,j} \cdot f_{i+1,j}$ (vertical edge pairs) |
| $Q_f$ | $\sum_{i,j} f_{i,j} \cdot f_{i,j+1} \cdot f_{i+1,j} \cdot f_{i+1,j+1}$ (2×2 faces) |

---

## `minkowski_functionals`

```python
healpix_analyse.minkowski.minkowski_functionals(
    img,
    threshold  = None,
    temperature = 20.0,
)
```

Compute W0, W1, W2 at a **single threshold level** (or with no thresholding).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img` | `Tensor [B, N, N]` | — | Batch of square 2D images, values in [0, 1]. |
| `threshold` | see below | `None` | Soft threshold applied via sigmoid before computation. `None` = use pixel values directly as soft membership. |
| `temperature` | `float` or `Tensor` | `20.0` | Sigmoid sharpness. Higher → closer to hard binary thresholding. |

**Accepted shapes for `threshold`:**

| Shape | Behaviour |
|---|---|
| `None` | No thresholding |
| `float` | Scalar — same threshold for the entire batch |
| `Tensor []` | Scalar tensor (may carry `requires_grad=True`) |
| `Tensor [B]` | One threshold per image in the batch |
| `Tensor [B, N, N]` | Per-pixel spatial threshold (e.g. learned) |
| Any shape broadcastable onto `[B, N, N]` | Accepted directly |

### Returns

`dict[str, Tensor]` with keys `'W0'`, `'W1'`, `'W2'`, each of shape `[B]`.

All outputs are differentiable w.r.t. `img` and (when applicable) `threshold`.

### Examples

```python
import torch
from healpix_analyse import minkowski_functionals

B, N = 8, 64
img = torch.rand(B, N, N, requires_grad=True)

# ── No threshold (continuous soft field) ──────────────────────────────────
mf = minkowski_functionals(img)
# mf['W0'].shape == mf['W1'].shape == mf['W2'].shape == [8]

# Backpropagate
mf['W2'].sum().backward()

# ── Scalar threshold ──────────────────────────────────────────────────────
mf = minkowski_functionals(img, threshold=0.5, temperature=30.0)

# ── Per-image threshold [B] ───────────────────────────────────────────────
t = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8])  # [B]
mf = minkowski_functionals(img, threshold=t)

# ── Spatial threshold [B, N, N] (learned) ────────────────────────────────
t_spatial = torch.rand(B, N, N, requires_grad=True)
mf = minkowski_functionals(img, threshold=t_spatial)
mf['W0'].sum().backward()
print(t_spatial.grad is not None)   # True — gradient through threshold
```

---

## `minkowski_curves`

```python
healpix_analyse.minkowski.minkowski_curves(
    img,
    thresholds,
    temperature = 20.0,
)
```

Compute W0, W1, W2 at **multiple threshold levels**, returning a curve
for each functional.  This is the standard form used in cosmological
field analysis and topological texture descriptors.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img` | `Tensor [B, N, N]` | — | Batch of square 2D images. |
| `thresholds` | `Tensor [T]` or `[B, T]` | — | Threshold levels. `[T]` = same grid for all images; `[B, T]` = a different grid per image (e.g. per-sample learned thresholds). |
| `temperature` | `float` or `Tensor` | `20.0` | Sigmoid sharpness. |

### Returns

`dict[str, Tensor]` with keys `'W0'`, `'W1'`, `'W2'`, each of shape `[B, T]`.

Fully differentiable w.r.t. `img` and `thresholds`.

### Examples

```python
import torch
from healpix_analyse import minkowski_curves

B, N, T = 4, 64, 16
img = torch.rand(B, N, N, requires_grad=True)

# ── Shared threshold grid [T] ─────────────────────────────────────────────
thresholds = torch.linspace(0.05, 0.95, T)
curves = minkowski_curves(img, thresholds)
# curves['W0'].shape == [4, 16]

# Use as a feature vector in a loss
loss = curves['W2'].pow(2).mean()
loss.backward()

# ── Per-image threshold grid [B, T] ──────────────────────────────────────
thresholds_bt = torch.rand(B, T, requires_grad=True)
curves = minkowski_curves(img, thresholds_bt)
curves['W1'].sum().backward()
print(thresholds_bt.grad is not None)  # True
```

---

## Use as loss / feature in a training loop

```python
import torch
from healpix_analyse import minkowski_curves

def minkowski_loss(pred, target, T=16, temperature=25.0):
    """
    MSE between the Minkowski curves of pred and target.
    Encourages the network to match morphological structure,
    not just pixel-wise intensities.
    """
    thresholds = torch.linspace(0.05, 0.95, T,
                                device=pred.device, dtype=pred.dtype)
    c_pred   = minkowski_curves(pred,   thresholds, temperature)
    c_target = minkowski_curves(target, thresholds, temperature)

    loss = sum(
        (c_pred[k] - c_target[k]).pow(2).mean()
        for k in ('W0', 'W1', 'W2')
    )
    return loss


# In your training loop:
pred   = model(x)                        # [B, N, N]
target = y                               # [B, N, N]
loss   = minkowski_loss(pred, target)
loss.backward()
```

---

## Sanity checks (2D planar)

| Input | Expected output |
|---|---|
| All-ones image | W0=1, W1=0, W2≈1/N² |
| All-zeros image | W0=0, W1=0, W2=0 |
| Checkerboard | W0=0.5, W1 near maximum, W2=0.5 |
| Single square blob | W0 ∝ blob area, W2 ≈ 1/N² |

---

## HEALPix version

For spherical HEALPix maps (`[..., Npix]`) the formulas are adapted to the
pixel graph: vertices are pixel centres, edges connect adjacent pixels (up to
8 neighbours each), and faces are the triangles formed by mutually adjacent
triples.  The Euler characteristic formula then reads:

$$W_2 = \frac{Q_1 - Q_E + Q_F}{N_{\text{pix}}}$$

where the three terms count (soft) active vertices, edges, and triangular
faces.  For a full sphere with all pixels active this recovers
$\chi(\mathbb{S}^2) = 2 / N_{\text{pix}}$ (the Euler characteristic of the
sphere, properly normalised).

The adjacency graph is **precomputed once** with `build_healpix_adjacency`
(requires healpy) and then reused across all differentiable forward passes.

### `build_healpix_adjacency`

```python
healpix_analyse.minkowski.build_healpix_adjacency(
    nside,
    cell_ids  = None,
    nest      = True,
    device    = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside` | `int` | — | HEALPix resolution parameter. |
| `cell_ids` | array-like or `None` | `None` | Global pixel indices of the map. `None` = full sky. |
| `nest` | `bool` | `True` | Pixel ordering (`True` = NESTED, `False` = RING). |
| `device` | device or `None` | `None` | Target device for the returned tensors. |

Returns `(edges, triangles)`:

| Return | Shape | Description |
|---|---|---|
| `edges` | `LongTensor [E, 2]` | Unique adjacent local-index pairs `(i < j)`. |
| `triangles` | `LongTensor [F, 3]` | Unique mutually adjacent triples `(i < j < k)`. |

### `minkowski_functionals_healpix`

```python
healpix_analyse.minkowski.minkowski_functionals_healpix(
    img,
    edges,
    triangles,
    threshold   = None,
    temperature = 20.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img` | `Tensor [..., Npix]` | — | Batch of HEALPix maps, any leading dims. |
| `edges` | `LongTensor [E, 2]` | — | From `build_healpix_adjacency`. |
| `triangles` | `LongTensor [F, 3]` | — | From `build_healpix_adjacency`. |
| `threshold` | same cases as 2D | `None` | Accepts `float`, `Tensor[B]`, `Tensor[B, Npix]`, etc. |
| `temperature` | `float` | `20.0` | Sigmoid sharpness. |

Returns `dict` with `'W0'`, `'W1'`, `'W2'`, each of shape `[...]`.

### `minkowski_curves_healpix`

```python
healpix_analyse.minkowski.minkowski_curves_healpix(
    img,
    edges,
    triangles,
    thresholds,
    temperature = 20.0,
)
```

`thresholds` can be `[T]` (shared) or `[B, T]` (per-sample).
Returns `dict` with `'W0'`, `'W1'`, `'W2'`, each of shape `[..., T]`.

### Example

```python
import torch
import numpy as np
from healpix_analyse import (
    build_healpix_adjacency,
    minkowski_functionals_healpix,
    minkowski_curves_healpix,
)

nside = 64
B     = 4

# ── One-time setup (uses healpy, not differentiable) ─────────────────────
edges, triangles = build_healpix_adjacency(nside, nest=True, device="cpu")
print(f"Edges: {edges.shape}, Triangles: {triangles.shape}")

# ── Forward pass (fully differentiable) ──────────────────────────────────
Npix = 12 * nside**2
img  = torch.rand(B, Npix, requires_grad=True)

mf = minkowski_functionals_healpix(img, edges, triangles, threshold=0.5)
# mf['W0'].shape == mf['W1'].shape == mf['W2'].shape == [4]

mf['W2'].sum().backward()
print(img.grad is not None)   # True

# ── Minkowski curves ──────────────────────────────────────────────────────
T = 16
thresholds = torch.linspace(0.05, 0.95, T)
curves = minkowski_curves_healpix(img, edges, triangles, thresholds)
# curves['W0'].shape == [4, 16]

# ── Partial sky ───────────────────────────────────────────────────────────
import healpy as hp
patch = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0.), np.radians(20.), nest=True)
edges_p, tri_p = build_healpix_adjacency(nside, cell_ids=patch, nest=True)

img_patch = torch.rand(B, len(patch), requires_grad=True)
mf_patch  = minkowski_functionals_healpix(img_patch, edges_p, tri_p)
```

### Sanity checks (HEALPix)

| Input | Expected output |
|---|---|
| All-ones map | W0=1, W1=0, W2≈2/Npix (Euler char. of sphere) |
| All-zeros map | W0=0, W1=0, W2=0 |
| Half-sky step | W0≈0.5, W1 large (great-circle boundary), W2≈0 |
