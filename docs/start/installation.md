# Installation

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) (CPU or GPU)
- [healpix-geo](https://healpix-geo.readthedocs.io/) ≥ 0.5 — all HEALPix
  geometry comes from here
- `numpy`, and `pyproj` ≥ 3.6 — ellipsoidal distances

These are installed for you by any of the commands below. `pyproject.toml`
also lists `termcolor`, `umap-learn` and `healpix-plot` as core dependencies,
although the package itself does not import them (see *Dependencies that
need attention* below).

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
| `healpy` | `build_healpix_adjacency` and therefore the HEALPix Minkowski functionals ({doc}`/stats/minkowski`) | not in any extra: `pip install healpy` (the default pixi environment has it) |
| `quaternionic` + `spherical` | `HEALPixSHT` with spin ≠ 0, including `uv_to_curl_div` ({doc}`/sht/healpix_sht`) | not in any extra: `pip install quaternionic spherical` |
| `transformers` | `load_dinov3_sat(source="hf")` ({doc}`/other/dino`) | extra `dino-hf` |
| `termcolor` | the torch-hub code of DINOv3 ({doc}`/other/dino`) | extra `dino` (also a core dependency today) |
| `matplotlib` | the plotting options of `ps`, `powerspectra`, `powerspectra_lonlat` and the debug plots of `AlmTransform` | extra `examples` |
| `cdshealpix` | the legacy helper module `healpix_analyse.make_rectangle` (used by some notebooks; not part of the documented API) | not in any extra |
| `foscat` | the legacy modules `LocalizedFlatSkyAlm`, `LocalizedAlm_optimized_v13_fixed` and the script `demo_ps` (excluded from the API reference) | not in any extra |

SciPy is never a core dependency, and new features do not rely on it.

## Extras

| Extra | `pip install -e ".[…]"` | What it is for |
|---|---|---|
| `scipy` | `scipy` | the SciPy features listed above |
| `benchmark` | `benchmark` | SciPy, for the scripts in `benchmarks/` |
| `dino` | `dino` | DINOv3 embeddings via torch-hub ({doc}`/other/dino`) |
| `dino-hf` | `dino-hf` | the same, loading the weights from Hugging Face instead |
| `examples` | `examples` | the Sentinel-2 example scripts: xarray, zarr, obstore, UMAP, scikit-learn, matplotlib, cartopy |
| `notebooks` | `notebooks` | JupyterLab, to run the notebooks |
| `docs` | `docs` | Sphinx, to build this documentation |
| `test` | `test` | pytest |

The DINOv3 **weights** are a separate matter: they are gated by Meta and this
package never downloads or redistributes them. See {doc}`/other/dino`.

## Dependencies that need attention

These are inconsistencies of the current packaging, recorded here so that
nobody is surprised; they are open questions of the documentation overhaul.

- **`healpix-plot`** is listed in the core `dependencies` of
  `pyproject.toml`, but it is not published on PyPI, and a comment in the
  `examples` extra says that it cannot be listed for that reason. The pixi
  environments pull it from git; otherwise install it yourself:

  ```bash
  pip install git+https://github.com/GRID4EARTH/healpix-plot
  ```

- **`umap-learn`** and **`termcolor`** are core dependencies but are only used
  by the examples and the DINOv3 hub code.
- **`healpy`**, **`quaternionic`**, **`spherical`**, **`cdshealpix`** and
  **`foscat`** are imported by some modules (table above) but declared
  nowhere in `pyproject.toml`.

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
