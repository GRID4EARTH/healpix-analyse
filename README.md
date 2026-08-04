# healpix-analyse

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://eopf-dggs.github.io/healpix-analyse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

A Python toolkit for analysing signals defined on HEALPix spherical grids,
with a focus on Earth Observation data. All operators are implemented in PyTorch
and are fully differentiable through `torch.autograd`.

**[Read the full documentation] (https://grid4earth.github.io/healpix-analyse/)**

---

## Features

- **Spherical harmonic transforms** — local ALM coefficients, ring-based full-sky SHT (spin-0, 1, 2), power spectra
- **Local 2D FFT** — pole-safe gnomonic projection, fast FFT/IFFT, CUDA and autograd
- **FFT large-kernel convolution** — zero-padded local `HealPixFFTConv` with CUDA and autograd
- **Gauge-equivariant convolution** — `HealPixConv` with configurable kernel size, gauge types, and number of gauges
- **Large-kernel convolution** — matched Down/Up hierarchy with a compact learned kernel
- **Multi-resolution operators** — `HealPixDown` (smooth / max-pool) and `HealPixUp` (adjoint upsampling), NESTED ordering
- **Masked multiscale decomposition** — exactly reconstructing local `HealPixDecomp` pyramids
- **Multiscale divergence and curl** — gauge-aware local derivatives of HEALPix velocity fields
- **HEALPix resampling** — local Up/Down conversion between full or partial NESTED domains
- **Differentiable by default** — all hot-path operations are autograd-compatible
- **NumPy and Torch interoperability** — accepts both array types, returns the same type

## Package map

```
healpix_analyse/
├── alm.py               # Local complex spherical harmonic coefficients
├── alm_latlon.py         # SHT for arbitrary iso-latitude grids
├── healpix_sht.py        # Ring-based full-sky SHT for HEALPix
├── fft_local.py          # Gnomonic 2D FFT for local HEALPix patches
├── fft_conv.py           # FFT-accelerated large-kernel local convolution
├── convol.py             # Gauge-equivariant spherical convolution (HealPixConv)
├── large_conv.py         # Multiresolution large-receptive-field convolution
├── down.py               # Resolution reduction (HealPixDown)
├── up.py                 # Resolution increase (HealPixUp)
├── decomp.py             # Exact local multiscale pyramid (HealPixDecomp)
├── divcurl.py            # Gauge-aware divergence/curl at every pyramid scale
├── powerspectra.py        # Isotropic power spectrum on HEALPix patches
├── powerspectra_lonlat.py # Power spectrum on irregular lon/lat grids
├── healpix_interp.py      # Bilinear interpolation on HEALPix (NESTED)
├── make_rectangle.py      # Rectangular HEALPix patches from bounding boxes
├── resample.py            # HEALPix level/domain resampling and regular lat/lon conversion
└── ps.py                  # Power spectrum utilities
```

---

## Quick start

All HEALPix-facing public interfaces use the Grid4Earth `level` convention;
the internal HEALPix resolution is always `nside = 2**level`.

```python
import numpy as np
import healpy as hp
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

nside = 64
npix  = 12 * nside**2
lmax  = 3 * nside

# Random test map
im = np.random.randn(npix)

# Build ring structure from HEALPix coordinates
theta, phi = hp.pix2ang(nside, np.arange(npix))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

# Compute angular power spectrum
cl = anafast_latlon(
    im[sort_idx], ring_theta, ring_phi_list, ring_counts,
    lmax=lmax, quadrature="equal_area",
)
print(cl.shape)   # torch.Size([193])
```

### Local flat-sky FFT

```python
from healpix_analyse import LocalFFT

transform = LocalFFT(cell_ids, level, device="cuda")
spectrum = transform.fft(data)
reconstructed = transform.ifft(spectrum)
frequency, power = transform.ps(spectrum)
```

The transform accepts local NESTED HEALPix patches up to a configurable
angular radius (10 degrees by default). Its three-dimensional tangent-frame
construction works at the poles and across 0/360 degrees. See the
[detailed local FFT documentation](docs/fft_local.md) for geometry,
normalisation, reconstruction accuracy and Sentinel-2 examples.

### FFT-accelerated large kernels

```python
from healpix_analyse import HealPixFFTConv

layer = HealPixFFTConv(
    level=12,
    in_channels=4,
    out_channels=8,
    kernel_sz=65,
    cell_ids=cell_ids,
    device="cuda",
)
y = layer(x)
```

The layer performs a zero-padded linear convolution on the same pole-safe
gnomonic grid used by `LocalFFT`. See the [FFT convolution documentation](docs/fft_conv.md).

### Large receptive-field convolution

```python
from healpix_analyse import LargeConv

layer = LargeConv(
    level=8,  # nside = 2**level = 256
    in_channels=8,
    out_channels=16,
    kernel_sz=33,
    max_compact_kernel_sz=7,
)
y = layer(x)
```

This example automatically uses three smooth Down operations, a compact
`5×5` convolution, and the three exactly paired Up operations. See the
[LargeConv documentation](docs/large_conv.md) for kernel planning, partial
patches, gradients and limitations.

### Exactly reconstructing multiscale decomposition

```python
from healpix_analyse import HealPixDecomp

decomp = HealPixDecomp(level=10, cell_ids=ocean_cell_ids, Jmax=5)
pyramid = decomp.compute(u_v)
u_v_reconstructed = decomp.invert(pyramid)
```

The pyramid retains the NESTED cell identifiers at every scale and works on
irregular masked domains such as ocean fields bounded by coastlines. See the
[multiscale decomposition documentation](docs/decomp.md).

### Multiscale divergence and curl

```python
from healpix_analyse import HealPixMultiScaleDivCurl

divcurl = HealPixMultiScaleDivCurl(decomp, kernel_sz=3, n_gauges=2)
diagnostics = divcurl(pyramid)

divergence = diagnostics.div
curl = diagnostics.curl
```

Each scale uses a fixed derivative-of-Gaussian `HealPixConv` kernel normalised
by that level's physical pixel spacing. See the
[divergence and curl documentation](docs/divcurl.md).

### HEALPix-to-HEALPix resampling

```python
from healpix_analyse import resample_healpix

out_data, out_ids = resample_healpix(
    in_data,
    in_level=11,
    out_level=8,
    in_cell_ids=in_cell_ids,
    out_cell_ids=out_cell_ids,
)
```

The output order follows `out_cell_ids`; unavailable cells are filled with
`NaN`. See the [HEALPix resampling documentation](docs/resample_healpix.md).

---

## Installation

```bash
pip install git+https://github.com/EOPF-DGGS/healpix-analyse.git
```

### From source (development)

```bash
git clone git@github.com:EOPF-DGGS/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

## Documentation

Full documentation is available at **[eopf-dggs.github.io/healpix-analyse](https://grid4earth.github.io/healpix-analyse/)**.

To build locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

## Relationship to healpix-geo and healpix-ai

- [healpix-geo](https://healpix-geo.readthedocs.io/) — HEALPix geometry: pixel coordinates, ellipsoids, coverage queries
- **healpix-analyse** — signal analysis: SHT, convolutions, power spectra, multi-resolution operators
- [healpix-ai](https://iaocea.github.io/healpix-ai/) — deep learning: autoencoders, U-Nets, forecasters built on top of `healpix-analyse`

## License

Apache 2.0 — see [LICENSE](LICENSE).
