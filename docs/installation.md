# Installation

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) (CPU or GPU)
- [healpix-geo](https://healpix-geo.readthedocs.io/) — all HEALPix geometry
  comes from here
- `pyproj` — ellipsoidal distances

Those four are installed for you by any of the commands below.

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

Or with [Pixi](https://pixi.sh/), which pins a full reproducible environment:

```bash
pixi install
pixi run python -c "import healpix_analyse; print('ok')"
```

## Extras

Nothing below is needed for the core operators; install an extra only for what
you actually want to run.

| Extra | `pip install -e ".[…]"` | What it is for |
|---|---|---|
| `dino` | `dino` | DINOv3 embeddings via torch-hub ({doc}`dino`) |
| `dino-hf` | `dino-hf` | the same, loading the weights from Hugging Face instead |
| `examples` | `examples` | the Sentinel-2 example scripts: xarray, zarr, obstore, UMAP, scikit-learn, matplotlib, cartopy |
| `notebooks` | `notebooks` | JupyterLab, to run the notebooks |
| `docs` | `docs` | Sphinx, to build this documentation |
| `test` | `test` | pytest |

Two packages are deliberately *not* dependencies:

- **`healpy`** is not used by `healpix-analyse`. It appears in this
  documentation only where an example cross-checks a result against it, and in
  a few legacy pixel-query helpers; the geometry itself is `healpix-geo`'s.
- **`healpix-plot`** is not on PyPI, so it cannot be listed as a dependency.
  The `pixi` environments pull it from git; otherwise install it yourself:

  ```bash
  pip install git+https://github.com/GRID4EARTH/healpix-plot
  ```

The DINOv3 **weights** are a separate matter: they are gated by Meta and this
package never downloads or redistributes them. See {doc}`dino`.

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
