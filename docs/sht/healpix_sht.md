# HEALPixSHT — spherical harmonics on the whole sphere

`healpix_sht` is the spherical harmonic transform for a **whole-sphere HEALPix
map**: analysis, synthesis and power spectra, differentiable throughout. It
handles scalar fields (spin-0), CMB polarisation (spin-2) and vector fields
(spin-1), including a one-call split of any `(u, v)` flow into its divergence
and relative vorticity.

```{include} ../_generated/glance/healpix_sht.md
```

```python
from healpix_analyse.healpix_sht import HEALPixSHT
```

Spin-0 needs only `torch`, `numpy` and `healpix-geo`. Spin ≥ 1 additionally
needs `quaternionic` and `spherical`:

```bash
pip install "healpix-analyse[sht]"      # quaternionic + spherical, only for spin > 0
```

For a grid that is *not* a whole-sphere HEALPix map — ERA5, a regular lat/lon grid, a
Gaussian grid — use {doc}`alm_latlon </sht/alm_latlon>` instead.

This page is the guide; the methods, shapes and costs are in the reference
section at the end, and the conventions and the ring-FFT algorithm are in {doc}`/sht/healpix_sht_maths`.

---

## Quickstart — scalar field (spin-0)

```python
import numpy as np
import torch
import healpy as hp
from healpix_analyse.healpix_sht import HEALPixSHT

level = 6
nside = 2**level
t_map = np.random.randn(12 * nside**2).astype(np.float32)

# --- create the transform object (precomputes geometry once) ---
sht = HEALPixSHT(level=level)
print(sht)
# HEALPixSHT(level=6, nside=64, lmax=191, n_rings=255, n_alm=18528, dtype=torch.float32, device=cpu)

# --- analysis: map → alm ---
alm = sht.map2alm(t_map)           # torch.Tensor, shape (18528,), complex64
alm_hp = hp.map2alm(t_map, iter=0) # healpy reference

# --- synthesis: alm → map ---
t_rec = sht.alm2map(alm)           # torch.Tensor, shape (49152,), float32

# --- power spectrum ---
cl     = sht.anafast(t_map)        # shape (192,)
cl_hp  = hp.anafast(t_map, iter=0)

# --- verify against healpy ---
print(np.max(np.abs(alm.cpu().numpy() - alm_hp)))  # < 1e-6 (float32)
print(np.max(np.abs(cl.cpu().numpy()  - cl_hp)))   # < 1e-10
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
sht   = HEALPixSHT(level=6)

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

## Spin-1: divergence and vorticity

This is perhaps the most powerful feature of `healpix_sht` for Earth-observation
and fluid-dynamics applications.

### The problem

Any smooth tangent vector field **u** = (u, v) on the sphere splits into a
gradient and a rotational part:

```
u = ∇χ  +  k × ∇ψ
```

where χ is the **velocity potential** (the divergent, irrotational part) and
ψ the **streamfunction** (the rotational, non-divergent part); **k** is the
local upward unit vector. This is the Helmholtz–Hodge decomposition on the
sphere. The divergence and the relative vorticity follow from the two
potentials:

```
div  = ∇·u        = ∇²χ
curl = k·(∇ × u)  = ∇²ψ
```

In terms of the spin-1 E/B decomposition of `(Q, U) = (-v, u)`:

- **almE** is the **gradient** (curl-free) part, carried by χ;
- **almB** is the **rotational** (divergence-free) part, carried by ψ.

On a sphere of radius R, for each multipole ℓ:

```
almE_ℓm = √(ℓ(ℓ+1))/R · χ_ℓm     div_ℓm  = -√(ℓ(ℓ+1))/R · almE_ℓm = -ℓ(ℓ+1)/R² · χ_ℓm
almB_ℓm = √(ℓ(ℓ+1))/R · ψ_ℓm     curl_ℓm = -√(ℓ(ℓ+1))/R · almB_ℓm = -ℓ(ℓ+1)/R² · ψ_ℓm
```

### One-call interface: `uv_to_curl_div`

```python
from healpix_analyse.healpix_sht import HEALPixSHT
import numpy as np

nside = 64
sht   = HEALPixSHT(level=6)

# Wind or ocean current field in m/s: u = east component, v = north component
u = np.random.randn(12 * nside**2).astype(np.float32)
v = np.random.randn(12 * nside**2).astype(np.float32)

# --- one call: vector field → divergence and relative vorticity ---
div, curl = sht.uv_to_curl_div(u, v)          # radius defaults to 6 371 008.8 m
# div  : divergence,          shape (N,) real Tensor, in s⁻¹
# curl : relative vorticity,  shape (N,) real Tensor, in s⁻¹
```

`radius=` sets the sphere radius in the length unit of `u` and `v`; the
default is the IUGG mean Earth radius, the same as the `radius_m` default of
`HealPixDivCurl` ({doc}`/stats/divcurl`).

### What each output represents

| Output | What it is | Zero when… |
|--------|-----------------|------------|
| `div`  | divergence ∇·u, in s⁻¹ for u, v in m/s; positive where the flow spreads out (sources) | the field is purely rotational (e.g. geostrophic flow) |
| `curl` | relative vorticity k·(∇ × u), in s⁻¹; positive for counter-clockwise rotation seen from above (cyclonic in the northern hemisphere) | the field is purely potential (irrotational) |

### E/B potential maps: `uv_to_eb`

`uv_to_eb(u, v)` returns the E and B parts synthesised as scalar maps without
any ℓ factor: `almE, almB = map2alm_spin(-v, u, spin=1)`, then
`alm2map(almE)` and `alm2map(almB)`. Both keep the units of `u` and `v`.
These are the maps that `uv_to_curl_div` returned before #94.

```python
E, B = sht.uv_to_eb(u, v)   # gradient (E) and rotational (B) parts, in m/s
```

### Step-by-step version

If you need the intermediate harmonic coefficients (e.g. for filtering
or cross-spectra), use the calls that `uv_to_curl_div` makes itself:

```python
import torch

R = 6371008.8
almE, almB = sht.map2alm_spin(-v, u, spin=1)

# optional: filter in harmonic space (e.g. low-pass at l < 50)
# ... set high-l coefficients of almE and almB to zero ...

# the factor -sqrt(l(l+1))/R in the flat alm layout
fac = torch.empty(sht.n_alm, dtype=sht.dtype, device=sht.device)
for m in range(sht.lmax + 1):
    ell = torch.arange(m, sht.lmax + 1, dtype=sht.dtype, device=sht.device)
    fac[sht._alm_slice(m)] = -torch.sqrt(ell * (ell + 1.0)) / R

div  = sht.alm2map(almE * fac)   # same as uv_to_curl_div
curl = sht.alm2map(almB * fac)
```

```{note}
The factor √(ℓ(ℓ+1)) grows with ℓ, so noise near `lmax` (including the
quadrature error of the HEALPix analysis) is amplified. For smooth fields, a
smaller `lmax` (e.g. `2 * nside`) gives cleaner derivatives. For a regional
domain or a mask, see {doc}`/stats/compare_divcurl`.
```

### Application: separating wind regimes

```python
# Decompose ERA5-style wind into rotational and divergent parts
nside = 128
sht   = HEALPixSHT(level=7)

# u10, v10: 10-metre wind components on a HEALPix grid, ring-ordered
div, curl = sht.uv_to_curl_div(u10, v10)

# div  (s⁻¹) captures convergence zones (precipitation, fronts)
# curl (s⁻¹) captures cyclones, anticyclones, jet streams
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

`sht.anafast` below is the production-ready angular power spectrum for a
whole-sphere HEALPix map, validated against `healpy`. The separate
`powerspectra` function in `powerspectra.py` computes a different, flat
spectrum through `AlmTransform` and is deprecated — see
{doc}`/sht/powerspectra`.

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
sht    = HEALPixSHT(level=6, dtype=torch.float64)
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
    def __init__(self, level, lmax_out):
        super().__init__()
        self.sht  = HEALPixSHT(level=level)
        self.weight = torch.nn.Parameter(
            torch.ones(self.sht.n_alm, dtype=torch.complex64)
        )

    def forward(self, x):
        alm = self.sht.map2alm(x)           # analysis
        alm = alm * self.weight             # learnable spectral filter
        return self.sht.alm2map(alm)        # synthesis
```

---

## Reference: methods, shapes and cost

The docstrings are in the API reference
({py:class}`~healpix_analyse.healpix_sht.HEALPixSHT`); this section adds the
conventions and costs that are not in them.

#### `HEALPixSHT(level, lmax=None, dtype=torch.float32, device=None, ellipsoid="sphere")`

Instantiate the transform and precompute all geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `int` | — | Grid4Earth/HEALPix level, integer ≥ 0. Internally, `nside = 2**level`. |
| `lmax` | `int` or `None` | `3·2**level - 1` | Maximum multipole ℓ. |
| `dtype` | `torch.dtype` | `torch.float32` | Real dtype for maps.  Use `torch.float64` for high-precision work. |
| `device` | `str` or `torch.device` or `None` | auto | Computation device.  Defaults to CUDA when available. |
| `ellipsoid` | `str` | `"sphere"` | Passed to `healpix_geo` to place the ring centres (not validated). |

Errors: `ValueError` for a `dtype` other than float32/float64, and for a map
or alm array whose last dimension is not `12·nside²` or `K`.

**Read-only attributes**

| Attribute | Description |
|-----------|-------------|
| `sht.nside` | HEALPix nside |
| `sht.lmax` | Maximum multipole |
| `sht.n_pix` | Total pixels: 12·nside² |
| `sht.n_rings` | Number of iso-latitude rings: 4·nside − 1 |
| `sht.n_alm` | Number of alm coefficients: (lmax+1)·(lmax+2)//2 |

---

#### `sht.map2alm(im, nest=False)` → `Tensor[..., K]`

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

#### `sht.alm2map(alm, nest=False)` → `Tensor[..., N]`

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

#### `sht.map2alm_spin(Q, U, spin=2, nest=False)` → `(almE, almB)`

Spin-s analysis: (Q, U) → E-mode and B-mode spherical harmonic coefficients.

| `spin` | Physical interpretation |
|--------|------------------------|
| `0` | Scalar pair: returns `almE = -map2alm(Q)`, `almB = -map2alm(U)`. The minus sign is the HEALPix spin convention `Q + iU = -Σ (almE + i almB) sY_ℓm` at s = 0, the same as `healpy.map2alm_spin(..., 0)`; call `map2alm` for plain scalar coefficients |
| `1` | Vector field: `almE` = gradient (curl-free) part, `almB` = rotational (divergence-free) part |
| `2` | CMB polarisation: `almE` = E-modes, `almB` = B-modes |

| Parameter | Description |
|-----------|-------------|
| `Q` | `(..., N)` real Tensor — first component (Stokes Q). |
| `U` | `(..., N)` real Tensor — second component (Stokes U). For a vector field `(u east, v north)`, `uv_to_eb` and `uv_to_curl_div` pass `Q = -v`, `U = u`. |
| `spin` | `int`, default `2`. The examples use `0`, `1` and `2`; the code does not restrict it. |
| `nest` | `bool`. NESTED ordering. |

Returns `(almE, almB)`, each a complex Tensor of shape `(..., K)`.

> **Requires** the `quaternionic` and `spherical` packages for `spin > 0`.
> Install them with the `sht` extra, `pip install 'healpix-analyse[sht]'`
> (see {doc}`/start/installation`).

---

#### `sht.alm2map_spin(almE, almB, spin=2, nest=False)` → `(Q, U)`

Spin-s synthesis: adjoint of `map2alm_spin`.

| Parameter | Description |
|-----------|-------------|
| `almE` | `(..., K)` complex Tensor. |
| `almB` | `(..., K)` complex Tensor. |
| `spin` | `int`, default `2`. |
| `nest` | `bool`. NESTED output. |

Returns `(Q, U)`, each a real Tensor of shape `(..., N)`.

---

#### `sht.uv_to_curl_div(u, v, nest=False, radius=6371008.8)` → `(div, curl)`

Divergence and relative vorticity of a tangent vector field on the whole
sphere.

What the code does: `almE, almB = map2alm_spin(-v, u, spin=1)`, then
`div = alm2map(-√(ℓ(ℓ+1))/radius · almE)` and
`curl = alm2map(-√(ℓ(ℓ+1))/radius · almB)` (spin-0 synthesis). The
div/curl validation pipeline ({doc}`/stats/divcurl`) applies the same
factor and signs. Needs `quaternionic` and `spherical`.

| Parameter | Description |
|-----------|-------------|
| `u` | `(..., N)` real Tensor — east component. |
| `v` | `(..., N)` real Tensor — north component. |
| `nest` | `bool`. NESTED ordering. |
| `radius` | `float`, default `6371008.8` (IUGG mean Earth radius, m). In the length unit of `u` and `v`. |

Returns:

| Output | Description |
|--------|-------------|
| `div`  | Divergence ∇·u = ∇²χ, in s⁻¹ for `u`, `v` in m/s. |
| `curl` | Relative vorticity k·(∇ × u) = ∇²ψ, in s⁻¹; positive counter-clockwise seen from above. |

---

#### `sht.uv_to_eb(u, v, nest=False)` → `(E, B)`

E/B potential maps of a tangent vector field: `almE, almB =
map2alm_spin(-v, u, spin=1)`, then `E = alm2map(almE)`, `B = alm2map(almB)`,
with no ℓ factor. `E` is the gradient (curl-free) part, `B` the rotational
(divergence-free) part, both in the units of `u` and `v`. In harmonic space,
`div_ℓm = -√(ℓ(ℓ+1))/R · E_ℓm` and `curl_ℓm = -√(ℓ(ℓ+1))/R · B_ℓm`. Needs
`quaternionic` and `spherical`.

| Parameter | Description |
|-----------|-------------|
| `u` | `(..., N)` real Tensor — east component. |
| `v` | `(..., N)` real Tensor — north component. |
| `nest` | `bool`. NESTED ordering. |

---

#### `sht.anafast(im, map2=None, spin=0, nest=False)` → `Tensor`

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

### Cost and tuning

- **Precomputation**: the Legendre tables and phase matrix are computed once
  at `HEALPixSHT(level=...)` time; subsequent calls to `map2alm` /
  `alm2map` reuse them. At level 6 (nside 64) the precomputation takes about
  0.5 s according to a notebook comment (not benchmarked).

- **Memory**: the two Legendre tables hold about `36·nside³` complex values
  with the default `lmax` (computed from the code): 76 MB at level 6, 0.60 GB
  at level 7 and 4.8 GB at level 8 in complex64 (twice that in complex128).
  Each spin value used adds four more tables of the same size. This, not the
  speed, limits the level you can use.

- **Spin harmonics** are computed lazily on the first call to `map2alm_spin`
  for a given spin value and cached for all subsequent calls.

- **dtype**: use `torch.float32` for speed-critical applications (training,
  large batches).  Use `torch.float64` when comparing against healpy or when
  high numerical accuracy matters.

- **GPU**: pass `device="cuda"` at construction time.  All hot-path operations
  (`fft`, `einsum`) run natively on GPU.

- **Batch size**: process many maps at once with a leading batch dimension
  `im.shape = (B, N)` for an effective B× throughput increase on GPU.
