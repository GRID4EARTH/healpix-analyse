# HEALPixSHT — conventions and algorithm

The normalisation this module follows, how coefficients are laid out, and why
the ring-FFT decomposition is fast. Read it to check a result against another
implementation, or before changing the transform.

Usage is in {doc}`/sht/healpix_sht`, the methods in {doc}`/sht/healpix_sht`.

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

### Cost compared with `alm_latlon`

Both transforms are ring-based: a per-ring FFT in longitude, then a Legendre
projection over the `4·nside − 1` rings for every `m`. An earlier version of
this page counted `alm_latlon` as "pixel-by-pixel" (12·nside² Legendre sums)
and concluded on a speed-up of about 3·nside; that premise is wrong, and the
same claim in the module docstring of `healpix_sht` should be read with this
in mind.

The real differences are:

- `HEALPixSHT` computes its Legendre tables once at construction and keeps
  them (memory: about `36·nside³` complex values, see the guide), while
  `alm_latlon` recomputes them in float64 at every call;
- `alm2map_latlon` synthesises each ring with a direct DFT, `HEALPixSHT`
  with an FFT;
- `HEALPixSHT` runs in float32 or float64 on the GPU.

No benchmark of the two is in the repository yet: the speed ratio is to be
measured.
