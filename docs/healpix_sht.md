# HEALPixSHT — spherical harmonics on the full sky

`healpix_sht` is the fast spherical harmonic transform for a **full-sky HEALPix
map**: analysis, synthesis and power spectra, differentiable throughout. It
handles scalar fields (spin-0), CMB polarisation (spin-2) and vector fields
(spin-1), including a one-call split of any `(u, v)` flow into its divergence
and curl.

```python
from healpix_analyse.healpix_sht import HEALPixSHT
```

Spin-0 needs only `torch`, `numpy` and `healpix-geo`. Spin ≥ 1 additionally
needs `quaternionic` and `spherical`:

```bash
pip install quaternionic spherical      # only for spin > 0
```

For a grid that is *not* full-sky HEALPix — ERA5, a regular lat/lon grid, a
Gaussian grid — use {doc}`alm_latlon <alm_latlon_1_quickstart>` instead.

This page is the guide. The methods are listed in {doc}`healpix_sht_api`, and
the conventions and the ring-FFT algorithm are in {doc}`healpix_sht_maths`.

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
sht   = HEALPixSHT(level=6)

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
sht   = HEALPixSHT(level=7)

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

`sht.anafast` below is the production-ready angular power spectrum for a
full-sky HEALPix map, validated against `healpy`. The separate
`powerspectra`/`powerspectra_lonlat` functions in `powerspectra.py` compute a
different, flat-sky-style spectrum through `AlmTransform` and are still under
development — see {doc}`powerspectra` for their current status before using
them instead of `anafast`.

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

