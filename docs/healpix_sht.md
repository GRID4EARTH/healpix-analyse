# `healpix_sht` — Ring-based Spherical Harmonic Transform for HEALPix

`healpix_sht` provides a fast, fully differentiable Spherical Harmonic Transform (SHT)
for full-sky HEALPix maps.  It supports scalar fields (spin-0), CMB polarisation (spin-2),
and general vector fields (spin-1), including a one-call decomposition of any (u, v) wind
or flow field into its **curl** and **divergence** components.

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart — scalar field (spin-0)](#quickstart--scalar-field-spin-0)
3. [Spin-2: CMB polarisation Q/U → E/B](#spin-2-cmb-polarisation-qu--eb)
4. [Spin-1: curl and divergence decomposition](#spin-1-curl-and-divergence-decomposition)
5. [Power spectra](#power-spectra)
6. [Gradient-based learning](#gradient-based-learning)
7. [API reference](#api-reference)
8. [Algorithm](#algorithm)
9. [Mathematical conventions](#mathematical-conventions)
10. [Performance notes](#performance-notes)

---

## Installation

`healpix_sht` is part of the `healpix-analyse` package.  Place `healpix_sht.py`
inside `healpix_analyse/` and import it as:

```python
from healpix_analyse.healpix_sht import HEALPixSHT
```

**Required dependencies**

| Package | Purpose |
|---|---|
| `torch` | all computation, autograd |
| `numpy` | geometry precomputation |
| `healpix-geo` | pixel → (lon, lat) conversion |

**Optional dependencies** (required only for spin ≥ 1)

| Package | Purpose |
|---|---|
| `quaternionic` | rotation quaternions for Wigner D-matrices |
| `spherical` | spin-weighted spherical harmonics |

```bash
pip install healpix-geo
pip install quaternionic spherical   # only needed for spin > 0
```

---

## Quickstart — scalar field (spin-0)

```python
import numpy as np
import torch
import healpy as hp
from healpix_analyse.healpix_sht import HEALPixSHT

nside = 64
t_map = np.random.randn(12 * nside**2).astype(np.float32)

# --- create the transform object (precomputes geometry once) ---
sht = HEALPixSHT(nside=nside)
print(sht)
# HEALPixSHT(nside=64, lmax=191, n_rings=255, n_alm=18528, dtype=torch.float32, device=cpu)

# --- analysis: map → alm ---
alm = sht.map2alm(t_map)           # torch.Tensor, shape (18528,), complex64
alm_hp = hp.map2alm(t_map, iter=0) # healpy reference

# --- synthesis: alm → map ---
t_rec = sht.alm2map(alm)           # torch.Tensor, shape (49152,), float32

# --- power spectrum ---
cl     = sht.anafast(t_map)        # shape (192,)
cl_hp  = hp.anafast(t_map, iter=0)

# --- verify against healpy ---
print(np.max(np.abs(alm.numpy() - alm_hp)))  # < 1e-6 (float32)
print(np.max(np.abs(cl.numpy()  - cl_hp)))   # < 1e-10
```

### NESTED ordering

Both analysis and synthesis accept a `nest=True` keyword for NESTED-ordered maps:

```python
alm   = sht.map2alm(t_map_nested, nest=True)
t_rec = sht.alm2map(alm, nest=True)
```

### Batched maps

The leading dimensions are arbitrary, so you can process a whole batch at once:

```python
batch = torch.randn(16, 12 * nside**2)   # 16 maps
alm   = sht.map2alm(batch)               # shape (16, 18528)
t_rec = sht.alm2map(alm)                 # shape (16, 49152)
```

---

## Spin-2: CMB polarisation Q/U → E/B

Spin-2 decomposition separates a polarisation field (Q, U) into its
**E-mode** (gradient, curl-free) and **B-mode** (divergence-free) components —
the standard decomposition used throughout CMB cosmology.

```python
import healpy as hp
from healpix_analyse.healpix_sht import HEALPixSHT

nside = 64
sht   = HEALPixSHT(nside=nside)

Q = np.random.randn(12 * nside**2).astype(np.float32)
U = np.random.randn(12 * nside**2).astype(np.float32)

# --- analysis: (Q, U) → (almE, almB) ---
almE, almB = sht.map2alm_spin(Q, U, spin=2)
# almE, almB: complex Tensors of shape (n_alm,)

# --- synthesis: (almE, almB) → (Q_rec, U_rec) ---
Q_rec, U_rec = sht.alm2map_spin(almE, almB, spin=2)

# --- cross-check against healpy ---
almE_hp, almB_hp = hp.map2alm_spin(np.vstack([Q, U]), 2)
print(np.max(np.abs(almE.numpy() - almE_hp)))  # < 1e-5

# --- EE / BB / EB power spectra ---
maps_QU = torch.stack([
    torch.as_tensor(Q),
    torch.as_tensor(U),
], dim=-2)   # shape (2, N)

cl_EEB = sht.anafast(maps_QU, spin=2)   # shape (3, lmax+1)
cl_EE  = cl_EEB[0]
cl_BB  = cl_EEB[1]
cl_EB  = cl_EEB[2]
```

---

## Spin-1: curl and divergence decomposition

This is perhaps the most powerful feature of `healpix_sht` for Earth-observation
and fluid-dynamics applications.

### The problem

Any smooth vector field **v** = (u, v) defined on the sphere can be uniquely
decomposed into two scalar potentials:

```
v = ∇Φ  +  ∇×Ψ
```

where **∇Φ** is the **divergent** (irrotational) part and **∇×Ψ** is the
**rotational** (non-divergent) part.  This is the Helmholtz–Hodge decomposition
on the sphere.

The two scalar maps **div** and **curl** (or equivalently **Φ** and **Ψ**)
are reconstructed via spin-1 spherical harmonics.  In terms of the E/B
decomposition of a spin-1 field:

- **almE** encodes the **divergent** component (gradient mode)
- **almB** encodes the **rotational** component (curl mode)

### One-call interface: `uv_to_curl_div`

```python
from healpix_analyse.healpix_sht import HEALPixSHT
import numpy as np

nside = 64
sht   = HEALPixSHT(nside=nside)

# Wind or ocean current field: u = east component, v = north component
u = np.random.randn(12 * nside**2).astype(np.float32)
v = np.random.randn(12 * nside**2).astype(np.float32)

# --- one call: vector field → divergence and curl maps ---
div, curl = sht.uv_to_curl_div(u, v)
# div  : divergence map  (∇·v),  shape (N,)  real Tensor
# curl : vorticity map   (∇×v),  shape (N,)  real Tensor
```

### What each output represents

| Output | Physical meaning | Zero when… |
|--------|-----------------|------------|
| `div`  | Divergence ∇·**v** — sources and sinks | the field is purely rotational (e.g. geostrophic flow) |
| `curl` | Vorticity ∇×**v** — rotation intensity | the field is purely potential (e.g. gravity waves) |

### Step-by-step version

If you need the intermediate harmonic coefficients (e.g. for filtering
or cross-spectra), use the two underlying calls directly:

```python
# decompose (u, v) → E-mode (divergent) and B-mode (rotational) alm
almE, almB = sht.map2alm_spin(u, v, spin=1)

# optional: filter in harmonic space (e.g. low-pass at l < 50)
almE_filtered = almE.clone()
almB_filtered = almB.clone()
# ... set high-l coefficients to zero ...

# reconstruct
div_filtered, curl_filtered = sht.alm2map_spin(almE_filtered, almB_filtered, spin=1)
```

### Application: separating wind regimes

```python
# Decompose ERA5-style wind into rotational and divergent parts
nside = 128
sht   = HEALPixSHT(nside=nside)

# u10, v10: 10-metre wind components on a HEALPix grid, ring-ordered
div, curl = sht.uv_to_curl_div(u10, v10)

# div  captures convergence zones (precipitation, fronts)
# curl captures cyclones, anticyclones, jet streams
```

### Application: ocean surface currents

```python
u_ssh, v_ssh = compute_geostrophic_currents(ssh_map)  # from SSH gradients

div, curl = sht.uv_to_curl_div(u_ssh, v_ssh)
# div  ≈ 0 for geostrophic flow (purely rotational)
# curl ≠ 0 shows the eddy field
```

---

## Power spectra

### Scalar auto-spectrum

```python
cl = sht.anafast(t_map)                     # shape (lmax+1,)
```

### Scalar cross-spectrum

```python
cl_cross = sht.anafast(t_map1, map2=t_map2) # shape (lmax+1,)
```

### Polarisation (spin-2) spectra: EE, BB, EB

```python
maps_QU  = torch.stack([torch.as_tensor(Q), torch.as_tensor(U)], dim=-2)  # (2, N)
cl3      = sht.anafast(maps_QU, spin=2)   # shape (3, lmax+1)
cl_EE, cl_BB, cl_EB = cl3[0], cl3[1], cl3[2]
```

### Batched spectra

```python
maps_batch = torch.randn(32, 12 * nside**2)
cl_batch   = sht.anafast(maps_batch)        # shape (32, lmax+1)
```

---

## Gradient-based learning

All operations in the forward pass — `torch.fft.fft`, `torch.fft.ifft`,
`torch.einsum` — are fully supported by `torch.autograd`.  Precomputed geometry
(Legendre tables, phase matrices, permutation indices) is stored as plain NumPy
arrays or non-grad tensors and does not participate in the gradient graph.

This means you can use `map2alm`, `alm2map`, `map2alm_spin`, and `alm2map_spin`
directly inside a neural network or an optimisation loop:

```python
import torch
from healpix_analyse.healpix_sht import HEALPixSHT

nside  = 64
sht    = HEALPixSHT(nside=nside, dtype=torch.float64)
target = torch.randn(12 * nside**2, dtype=torch.float64)

# Learnable map
x = torch.randn(12 * nside**2, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-2)

for step in range(200):
    optimizer.zero_grad()

    alm  = sht.map2alm(x)
    cl   = sht.anafast(x)
    loss = ((cl - sht.anafast(target)) ** 2).mean()

    loss.backward()   # gradients flow through fft and einsum
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step:4d}  loss = {loss.item():.6e}")
```

### Inside a neural network

```python
class SpectralLayer(torch.nn.Module):
    def __init__(self, nside, lmax_out):
        super().__init__()
        self.sht  = HEALPixSHT(nside=nside)
        self.weight = torch.nn.Parameter(
            torch.ones(self.sht.n_alm, dtype=torch.complex64)
        )

    def forward(self, x):
        alm = self.sht.map2alm(x)           # analysis
        alm = alm * self.weight             # learnable spectral filter
        return self.sht.alm2map(alm)        # synthesis
```

---

## API reference

### `HEALPixSHT(nside, lmax, dtype, device, ellipsoid)`

Instantiate the transform and precompute all geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nside` | `int` | — | HEALPix resolution parameter.  Must be a power of 2. |
| `lmax` | `int` or `None` | `3·nside - 1` | Maximum multipole ℓ. |
| `dtype` | `torch.dtype` | `torch.float32` | Real dtype for maps.  Use `torch.float64` for high-precision work. |
| `device` | `str` or `torch.device` or `None` | auto | Computation device.  Defaults to CUDA when available. |
| `ellipsoid` | `str` | `"sphere"` | Geometry model: `"sphere"` or `"WGS84"`. |

**Read-only attributes**

| Attribute | Description |
|-----------|-------------|
| `sht.nside` | HEALPix nside |
| `sht.lmax` | Maximum multipole |
| `sht.n_pix` | Total pixels: 12·nside² |
| `sht.n_rings` | Number of iso-latitude rings: 4·nside − 1 |
| `sht.n_alm` | Number of alm coefficients: (lmax+1)·(lmax+2)//2 |

---

### `sht.map2alm(im, nest=False)` → `Tensor[..., K]`

Analysis transform: HEALPix map → spherical harmonic coefficients.

```
a_lm = ∫ f(θ,φ) · Y_lm*(θ,φ) dΩ
```

| Parameter | Description |
|-----------|-------------|
| `im` | `(..., N)` real array-like or Tensor.  RING ordering by default. |
| `nest` | `bool`. Set `True` for NESTED input. |

Returns a complex Tensor of shape `(..., K)` where `K = (lmax+1)·(lmax+2)//2`.

**alm layout** (same as healpy):
```
[m=0: l=0..lmax | m=1: l=1..lmax | … | m=lmax: l=lmax]
```

---

### `sht.alm2map(alm, nest=False)` → `Tensor[..., N]`

Synthesis transform: a_lm → HEALPix map.

```
f(θ,φ) = Σ_{l,m} a_lm · Y_lm(θ,φ)
```

| Parameter | Description |
|-----------|-------------|
| `alm` | `(..., K)` complex array-like or Tensor. |
| `nest` | `bool`. Set `True` for NESTED output. |

Returns a real Tensor of shape `(..., N)`.

---

### `sht.map2alm_spin(Q, U, spin, nest=False)` → `(almE, almB)`

Spin-s analysis: (Q, U) → E-mode and B-mode spherical harmonic coefficients.

| `spin` | Physical interpretation |
|--------|------------------------|
| `0` | Scalar pair: `almE = map2alm(Q)`, `almB = map2alm(U)` |
| `1` | Vector field: `almE` = divergent part, `almB` = rotational part |
| `2` | CMB polarisation: `almE` = E-modes, `almB` = B-modes |

| Parameter | Description |
|-----------|-------------|
| `Q` | `(..., N)` real Tensor — first component (east / Stokes Q). |
| `U` | `(..., N)` real Tensor — second component (north / Stokes U). |
| `spin` | `int`. Spin weight: `0`, `1`, or `2`. |
| `nest` | `bool`. NESTED ordering. |

Returns `(almE, almB)`, each a complex Tensor of shape `(..., K)`.

> **Requires** `quaternionic` and `spherical` packages for `spin > 0`.

---

### `sht.alm2map_spin(almE, almB, spin, nest=False)` → `(Q, U)`

Spin-s synthesis: adjoint of `map2alm_spin`.

| Parameter | Description |
|-----------|-------------|
| `almE` | `(..., K)` complex Tensor. |
| `almB` | `(..., K)` complex Tensor. |
| `spin` | `int`. Spin weight: `0`, `1`, or `2`. |
| `nest` | `bool`. NESTED output. |

Returns `(Q, U)`, each a real Tensor of shape `(..., N)`.

---

### `sht.uv_to_curl_div(u, v, nest=False)` → `(div, curl)`

One-call Helmholtz–Hodge decomposition of a tangent-plane vector field.

Internally calls `map2alm_spin(u, v, spin=1)` and
`alm2map_spin(almE, almB, spin=1)`.

| Parameter | Description |
|-----------|-------------|
| `u` | `(..., N)` real Tensor — east component. |
| `v` | `(..., N)` real Tensor — north component. |
| `nest` | `bool`. NESTED ordering. |

Returns:

| Output | Description |
|--------|-------------|
| `div`  | Divergence map ∇·**v** — sources and sinks of the flow. |
| `curl` | Vorticity map ∇×**v** — rotation intensity of the flow. |

---

### `sht.anafast(im, map2=None, spin=0, nest=False)` → `Tensor`

Angular power spectrum C_ℓ.

| Parameter | Description |
|-----------|-------------|
| `im` | `(..., N)` for spin=0, or `(..., 2, N)` for spin>0 with `[Q, U]` on axis -2. |
| `map2` | Same shape as `im`. If given, computes the cross-spectrum. |
| `spin` | `int`. `0` for scalar; `1` or `2` for E/B modes. |
| `nest` | `bool`. NESTED ordering. |

**Returns**

- `spin=0` → `(..., lmax+1)` real Tensor.
- `spin>0` → `(..., 3, lmax+1)` real Tensor with `[C_l^EE, C_l^BB, C_l^EB]`.

**Formula** (spin=0):

```
C_l = 1/(2l+1) × [ |a_{l0}|² + 2 Σ_{m=1}^{l} |a_{lm}|² ]
```

---

## Algorithm

### Ring-FFT decomposition

HEALPix maps have `4·nside − 1` iso-latitude rings.  Within each ring the
pixels are **uniformly spaced in longitude**, which allows the longitude integral
to be computed exactly with a standard 1-D FFT.

**Analysis pipeline (map → alm)**

```
1. For each ring r:
      F_r(m) = FFT_m( f_ring_r )  ×  exp(-i·m·φ_0^r)

2. For each m = 0..lmax:
      a_lm = sqrt(2l+1)/N  ×  Σ_r  √4π·P̃_lm(cos θ_r)  ×  F_r(m)
```

**Synthesis pipeline (alm → map)**

```
1. For each m = 0..lmax, for each ring r:
      G_r(m) = Σ_l  a_lm  ×  sqrt((2l+1)/4π) · P̃_lm(cos θ_r)

2. For each ring r:
      f_ring_r = IFFT[ H_r ]   where H_r[k] includes both positive and
                                conjugate-negative frequency aliases
```

### Speed comparison

| Method | Legendre sums | Complexity |
|--------|--------------|-----------|
| `alm_latlon` (pixel-by-pixel) | 12·nside² | O(lmax² · nside²) |
| `healpix_sht` (ring-by-ring) | 4·nside − 1 | O(lmax² · nside) |
| Speed-up | ~3·nside | e.g. ×192 at nside=64 |

At `nside=64` (lmax=191), the ring-based approach performs ~255 Legendre
summations instead of ~49 000 — a factor of ~192 reduction.

---

## Mathematical conventions

### Spherical harmonics (spin-0)

The normalisation follows healpy and the standard CMB literature:

```
Y_lm(θ,φ) = sqrt((2l+1)/(4π)) · P̃_lm(cos θ) · exp(i·m·φ)

a_lm = ∫ f(θ,φ) · Y_lm*(θ,φ) dΩ

C_l  = 1/(2l+1) · [ |a_{l0}|² + 2 Σ_{m=1}^{l} |a_{lm}|² ]
```

where P̃_lm are the **normalised** associated Legendre polynomials satisfying:

```
∫₋₁¹ P̃_lm(x)² dx = 1
```

Only m ≥ 0 coefficients are stored (reality condition: a_{l,−m} = (−1)^m · ā_{lm}).

### Spin-weighted harmonics

For spin weight s, the decomposition is:

```
(Q ± iU)(θ,φ) = Σ_{lm} (±almE ∓ i·almB) · ±sY_lm(θ,φ)
```

giving:

```
almE − i·almB = ∫ (Q+iU) · (+sY_lm)* dΩ   [libsharp / healpy convention]
```

The spin harmonics are evaluated at the ring colatitudes using the
`spherical` package (Boyle convention) with sign corrections applied to
match the healpy/libsharp convention.

### alm storage layout

Coefficients are stored in a flat 1-D complex Tensor, ordered identically
to `healpy.map2alm`:

```
index  0          → (l=0, m=0)
index  1          → (l=1, m=0)
...
index  lmax       → (l=lmax, m=0)
index  lmax+1     → (l=1,    m=1)
index  lmax+2     → (l=2,    m=1)
...
index  K-1        → (l=lmax, m=lmax)
```

Total: `K = (lmax+1)·(lmax+2)//2` complex coefficients.

---

## Performance notes

- **Precomputation**: the Legendre tables and phase matrix are computed once
  at `HEALPixSHT(nside=...)` time.  At nside=64 this takes ~0.5 s; subsequent
  calls to `map2alm` / `alm2map` are fast.

- **Spin harmonics** are computed lazily on the first call to `map2alm_spin`
  for a given spin value and cached for all subsequent calls.

- **dtype**: use `torch.float32` for speed-critical applications (training,
  large batches).  Use `torch.float64` when comparing against healpy or when
  high numerical accuracy matters.

- **GPU**: pass `device="cuda"` at construction time.  All hot-path operations
  (`fft`, `einsum`) run natively on GPU.

- **Batch size**: process many maps at once with a leading batch dimension
  `im.shape = (B, N)` for an effective B× throughput increase on GPU.
