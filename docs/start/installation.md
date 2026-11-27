# Installation

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) (CPU or GPU)
- [healpix-geo](https://healpix-geo.readthedocs.io/) ≥ 0.5 — all HEALPix
  geometry comes from here
- `numpy`, and `pyproj` ≥ 3.6 — ellipsoidal distances

These are installed for you by any of the commands below.

## Install

```bash
pip install git+https://github.com/GRID4EARTH/healpix-analyse.git
```

From source, for development:

```bash
git clone git@github.com:GRID4EARTH/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

Or with [Pixi](https://pixi.sh/), which pins a full reproducible environment
(Linux x86-64 and macOS arm64):

```bash
pixi install
pixi run python -c "import healpix_analyse; print('ok')"
```

## Optional dependencies

Nothing below is needed for the core tools. Each package is imported only
when the feature that needs it is used.

| Package | Needed by | How to get it |
|---|---|---|
| SciPy | `multiorder_filter_operator(shortest_path="scipy")` ({doc}`/conv/multiorder_filter`); the `"matern"`, `"voigt"` and `"screened_poisson"` kernels of `HealPixGeoFFT` ({doc}`/conv/geo_fft`); `HealPixKernelPyramid.calibrate_joint` and the `validation` helpers ({doc}`/other/validation`); `resample_to_latlon_grid` ({doc}`/other/resample`) | extra `scipy` |
| `quaternionic` + `spherical` | `HEALPixSHT` with spin ≠ 0, including `uv_to_curl_div` ({doc}`/sht/healpix_sht`) | extra `sht` |
| `transformers` | `load_dinov3_sat(source="hf")` ({doc}`/other/dino`) | extra `dino-hf` |
| `termcolor` | the torch-hub code of DINOv3 ({doc}`/other/dino`) | extra `dino` |
| `matplotlib` | the plotting option of the deprecated `powerspectra` and the debug plots of `AlmTransform` | extra `examples` |
| `umap-learn`, `healpix-plot` | the Sentinel-2 example scripts and notebooks only; the package does not import them | extra `examples` (`healpix-plot` is not on PyPI: `pip install git+https://github.com/GRID4EARTH/healpix-plot`; the pixi `examples` feature has it) |

SciPy is never a core dependency, and new features do not rely on it.

## Extras

| Extra | `pip install -e ".[…]"` | What it is for |
|---|---|---|
| `scipy` | `scipy` | the SciPy features listed above |
| `sht` | `sht` | `quaternionic` + `spherical`, for the spin ≠ 0 transforms of `HEALPixSHT` |
| `benchmark` | `benchmark` | SciPy, for the scripts in `benchmarks/` |
| `triton` | `triton` | the fused CUDA kernels of `HealPixConv` ({doc}`/conv/convol`); the Linux CUDA wheels of PyTorch usually include it already |
| `dino` | `dino` | DINOv3 embeddings via torch-hub ({doc}`/other/dino`) |
| `dino-hf` | `dino-hf` | the same, loading the weights from Hugging Face instead |
| `examples` | `examples` | the Sentinel-2 example scripts: xarray, zarr, obstore, UMAP, scikit-learn, matplotlib, cartopy |
| `notebooks` | `notebooks` | JupyterLab, to run the notebooks |
| `docs` | `docs` | Sphinx, to build this documentation |
| `test` | `test` | pytest, and `healpy` for reference values |

The DINOv3 **weights** are a separate matter: they are gated by Meta and this
package never downloads or redistributes them. See {doc}`/other/dino`.

## Check that it worked

```python
import numpy as np
from healpix_geo import nested
from healpix_analyse.down import HealPixDown

depth = 4
ids = np.arange(12 * 4 ** depth, dtype=np.int64)
print(len(ids), "cells at level", depth)
print(nested.healpix_to_lonlat(ids.astype(np.uint64), depth)[0][:3], "…")
```

## Building this documentation

The documentation builds with Sphinx, warnings as errors, like the CI:

```bash
pixi run -e docs build-docs
```

or, without pixi, `pip install -e . -r docs/requirements.txt` and then
`python -m sphinx -b html -W -n docs docs/_build`. The tables of the family
pages, the "At a glance" boxes, the API reference by family and `llms.txt`
are generated from `docs/tools.yaml`; the build fails if a public symbol is
missing from it.
