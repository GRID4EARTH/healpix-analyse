# HEALPixSHT — API reference

Every public method of `HEALPixSHT`, and what the transform costs. The guide is
{doc}`healpix_sht`; the conventions are {doc}`healpix_sht_maths`.

---

## Methods

### `HEALPixSHT(level, lmax, dtype, device, ellipsoid)`

Instantiate the transform and precompute all geometry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `int` | — | Grid4Earth/HEALPix level, integer ≥ 0. Internally, `nside = 2**level`. |
| `lmax` | `int` or `None` | `3·2**level - 1` | Maximum multipole ℓ. |
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

## Cost and tuning

- **Precomputation**: the Legendre tables and phase matrix are computed once
  at `HEALPixSHT(level=...)` time.  At level=6 (nside=64) this takes ~0.5 s; subsequent
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
