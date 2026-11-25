# healpix-analyse

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://grid4earth.eu/healpix-analyse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1176905310.svg)](https://zenodo.org/badge/latestdoi/1176905310)

Signal analysis on HEALPix grids, for Earth observation.

`healpix-analyse` works on the values stored in HEALPix cells: filters and
convolutions with widths in metres, gradients and neighbourhood statistics,
connected regions of a mask, resolution changes and resampling, spherical
harmonics and local FFT spectra, and foundation-model embeddings. It runs on a
regional domain (a Sentinel-2 scene, a basin, a coastal strip) or on the full
sphere. The tools are written in PyTorch; most accept NumPy arrays or tensors
and are differentiable through `torch.autograd`.

**[Read the documentation](https://grid4earth.eu/healpix-analyse/)** — start
with the **[decision guide](https://grid4earth.eu/healpix-analyse/start/overview.html)**
to find the tool for your task.

---

## Quick start

A field on a regional domain is two arrays: the values, and the NESTED
`cell_ids` of the cells that hold them, at a refinement `level`. Here the
domain is every level-10 cell (about 6.4 km) within one degree of a point in
the Bay of Biscay, computed on the WGS84 ellipsoid by
[healpix-geo](https://healpix-geo.readthedocs.io/):

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse import gaussian_filter, gradient, resample_healpix

level = 10
cell_ids, _, _ = nested.cone_coverage(
    (-4.5, 45.5), 1.0, level, ellipsoid="WGS84", flat=True
)
cell_ids = np.asarray(cell_ids, dtype=np.uint64)

# A field on those cells, e.g. a sea-surface temperature (degC).
lon, lat = nested.healpix_to_lonlat(cell_ids, level, ellipsoid="WGS84")
sst = 15.0 + 0.5 * (lat - 45.5) + 0.3 * np.sin(np.deg2rad(8 * lon))
sst += 0.2 * np.random.default_rng(0).standard_normal(sst.shape)

# Gaussian smoothing, width in metres (WGS84 geodesic distances).
sst_smooth = gaussian_filter(sst, cell_ids, level, sigma_m=15_000.0)

# Geographic gradient, east and north components, in degC per metre.
d_east, d_north = gradient(sst_smooth, cell_ids, level)

# Coarsen the regional field to level 8 (about 25 km): the output cells are
# the parents of the domain (in NESTED ordering, cell >> 2 per level).
cell_ids_l8 = np.unique(cell_ids >> np.uint64(2 * (level - 8)))
sst_l8, cell_ids_l8 = resample_healpix(
    sst_smooth, in_level=level, out_level=8,
    in_cell_ids=cell_ids, out_cell_ids=cell_ids_l8,
)

print(len(cell_ids), "cells at level", level, "->", len(cell_ids_l8), "at level 8")
print(f"mean northward gradient: {np.nanmean(d_north) * 1e5:.2f} degC per 100 km")
```

To build the domain from your own data instead, convert its longitudes and
latitudes with `nested.lonlat_to_healpix(lon, lat, level, ellipsoid="WGS84")`
and keep the unique cell ids. For the full sphere, use every cell:
`cell_ids = np.arange(12 * 4**level, dtype=np.uint64)`.

All public interfaces take a `level` (some name it `refinement_level`, or
`in_level` / `out_level`). A grid at level `L` has `12 · 4**L` cells. Where a
third-party HEALPix library asks for `nside`, it is `2**level`. The
[conventions page](https://grid4earth.eu/healpix-analyse/start/conventions.html)
lists argument names, units, default ellipsoids and array shapes tool by tool.

## What is in the package

The documentation is organised by task. Each family page compares the tools
that do the same job and gives the pros, cons and alternatives of each one.

| Family | For example |
|---|---|
| [A. Convolution & filtering](https://grid4earth.eu/healpix-analyse/conv/index.html) | Gaussian, radial and directional filters in metres (`gaussian_filter`, `radial_filter`, `directional_filter`); learned convolutions (`HealPixConv`, `LargeConv`, `HealPixFFTConv`); wide fixed kernels (`HealPixWideConv`, `HealPixPyramidConv`, `HealPixGeoFFT`) |
| [B. Statistics & local analysis](https://grid4earth.eu/healpix-analyse/stats/index.html) | neighbourhood median / max / count (`neighbour_reduce`), `gradient`, divergence and curl of vector fields, `connected_components`, morphology, Minkowski functionals |
| [C. Spherical harmonics & FFT](https://grid4earth.eu/healpix-analyse/sht/index.html) | full-sphere spherical-harmonic transforms (`HEALPixSHT`, lat/lon grids), local FFT of a regional patch (`LocalFFT`), power and wavelet spectra |
| [D. Other tools](https://grid4earth.eu/healpix-analyse/other/index.html) | coarser and finer levels (`HealPixDown`, `HealPixUp`, `resample_healpix`), exact multiscale pyramids (`HealPixDecomp`), interpolation at points, DINOv3 embeddings of Sentinel-2 tiles |

The same catalogue is published for AI assistants as
[`llms.txt`](https://grid4earth.eu/healpix-analyse/llms.txt) and
[`llms-full.txt`](https://grid4earth.eu/healpix-analyse/llms-full.txt); its
source is [`docs/tools.yaml`](docs/tools.yaml).

## Relationship with healpix-geo and the GRID4EARTH suite

- **[healpix-geo](https://healpix-geo.readthedocs.io/)** answers *where* the
  cells are: cell ids from longitudes and latitudes, cell centres and
  boundaries, coverage of a cone, a box or a polygon, neighbours, on the sphere
  or on an ellipsoid such as WGS84. It is a required dependency:
  `healpix-analyse` takes all its HEALPix geometry from it, and you use it to
  build the `cell_ids` of your domain.
- **healpix-analyse** answers *what to do with the values* in those cells:
  filtering, statistics, spectra, resolution changes.
- **healpix-ai** builds deep-learning
  models (autoencoders, U-Nets, forecasters) on top of `healpix-analyse`.
- **[healpix-plot](https://github.com/GRID4EARTH/healpix-plot)** draws HEALPix
  fields on maps.

## Installation

```bash
pip install git+https://github.com/GRID4EARTH/healpix-analyse.git
```

`healpix-plot`, listed as a dependency, is not on PyPI; if pip cannot resolve
it, install it first:

```bash
pip install git+https://github.com/GRID4EARTH/healpix-plot
```

From source, for development:

```bash
git clone https://github.com/GRID4EARTH/healpix-analyse.git
cd healpix-analyse
pip install -e ".[test]"
```

or with [Pixi](https://pixi.sh/), which pins a full reproducible environment
(Linux x86-64 and macOS arm64):

```bash
pixi install
pixi run python -c "import healpix_analyse; print('ok')"
```

Optional extras: `scipy` (a few optional features), `examples` (the
Sentinel-2 example scripts), `notebooks` (JupyterLab), `dino` / `dino-hf`
(DINOv3 embeddings), `docs`, `test`, `benchmark`. The
[installation page](https://grid4earth.eu/healpix-analyse/start/installation.html)
says which feature needs which extra.

To build the documentation locally:

```bash
pixi run -e docs build-docs
```

or `pip install -e ".[docs]"` and `python -m sphinx -b html docs docs/_build`.

## Development and contributing

Bug reports, feature requests, and contributions are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution process and
[`DEVELOPMENT.md`](DEVELOPMENT.md) for the lightweight branch and cross-repository development
workflow.

## Citation

GitHub releases are archived by Zenodo. Use the DOI badge at the top of this page to cite the
latest archived release, or cite the DOI for the exact version used in your work.

## Funding and acknowledgements

**Funded by ESA, built by a European consortium.**

ESA Contract: `4000147951/25/I-NS` — Technical Officer: Vincent Dumoulin.

We also thank all external contributors and the open-source projects on which `healpix-analyse`
depends.

## License

Apache 2.0 — see [LICENSE](LICENSE).
