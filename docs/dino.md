# `GetDINOV3SAT` — DINOv3 embeddings of HEALPix data

**Module** `healpix_analyse.dino`  
**Function** `GetDINOV3SAT`  
**Status** experimental

All HEALPix geometry — cell centres, vertices, bilinear stencils, neighbours,
base-cell coordinates — comes from **`healpix-geo`**, the GRID4EARTH library.
`healpy` is not a dependency of this package and is not used anywhere in it,
including in the tests and the notebooks. The four primitives the examples
need are re-exported so that code built on this module does not have to reach
for another HEALPix library either: `cell_centres_lonlat`, `cells_at_lonlat`,
`cell_vectors` and `cell_neighbours`.

---

## Overview

`GetDINOV3SAT` computes DINOv3 (SAT-493M) embeddings of Sentinel-2-like data
stored on a NESTED HEALPix grid, **without modifying the network**.

The trick is purely combinatorial. In the NESTED scheme a cell of
`parent_level` contains exactly `4**(level - parent_level)` cells of `level`,
stored contiguously and ordered by a Morton (Z-order) code of their local
face coordinates `(x, y)`. A NESTED block is therefore an *exact* square image
of side `S = 2**(level - parent_level)` pixels, with no resampling. The image
is fed to DINOv3 as is, and every 16 × 16 patch token maps back to one
HEALPix cell of `level - 4`.

```
level 19 cells  ──nested_to_tiles──▶  tiles [M, 3, 256, 256]  ──DINOv3──▶  CLS  [M, 1024]           (one per level-11 cell)
                                                                          patches [M·256, 1024]     (one per level-15 cell)
```

Compared with running DINOv3 on UTM tiles, embeddings live directly on the
sphere: they are seamless across tiles of the same face, hierarchical
(`level - 4`, `parent_level`) for free, and comparable across dates and
orbits because the grid never changes.

---

## Signature

```python
GetDINOV3SAT(
    data,                     # [N, C] reflectances in [0, 1], NaN = missing
    cell_id,                  # [N] NESTED ids at `level`
    level,                    # resolution of `data`
    parent_level,             # resolution of the DINO tiles, level - parent_level >= 4
    model          = None,    # backbone from load_dinov3_sat(); loaded if None
    weights        = None,    # .pth path (torch-hub) or HF id
    model_name     = "dinov3_vitl16",
    bands          = (0, 1, 2),          # (R, G, B) columns of data — S2: B04, B03, B02
    mean           = SAT493M_MEAN,
    std            = SAT493M_STD,
    pooling        = "cls",              # "cls" | "mean" | "cls+mean"
    return_patches = False,
    min_coverage   = 0.0,
    fill           = "mean",             # missing pixels: per-tile mean | "zero"
    batch_size     = 16,
    device         = None,
    autocast       = True,
) -> DINOEmbedding
```

| Field of `DINOEmbedding` | Shape | Meaning |
|---|---|---|
| `embedding` | `[M, Ndino]` | one vector per tile of `parent_level` present in `cell_id` |
| `cell_id` | `[M]` | NESTED ids of those tiles |
| `coverage` | `[M]` | fraction of `level` pixels available in each tile |
| `patch_embedding` | `[M·P, Ndino]` | patch tokens (`return_patches=True`) |
| `patch_cell_id` | `[M·P]` | NESTED ids of the patches at `patch_level = level - 4` |

`Ndino` is 1024 for ViT-L/16 (`pooling="cls+mean"` doubles it).

---

## Choosing `parent_level`

| `level - parent_level` | tile side | patch tokens | comment |
|---|---|---|---|
| 4 | 16 px | 1 | one patch per tile — no context, avoid |
| 6 | 64 px | 4 × 4 | small context, many tiles |
| **8** | **256 px** | **16 × 16** | close to the 224–256 px training regime, recommended |
| 10 | 1024 px | 64 × 64 | 4096 tokens, quadratic attention cost |

For Sentinel-2 at level 19 (≈ 10 m), `parent_level = 11` gives 256 px tiles
of ≈ 2.5 km and patch embeddings on level-15 cells (≈ 160 m).

---

## Input range — what the network expects

DINOv3 SAT-493M was trained on 8-bit RGB imagery divided by 255, then
normalised with `SAT493M_MEAN` / `SAT493M_STD`. It expects values in [0, 1]
with the brightness of an 8-bit visual product. `GetDINOV3SAT` therefore maps
its input through `input_range` before anything else, and prints the ranges
before and after (`verbose=True`):

| `input_range` | mapping to [0, 1], then clipped |
|---|---|
| `"auto"` (default), `"percentile"` | 1st-99th percentiles of all the input values, all bands together |
| `"sat493m"` | per band: the scene gets the mean / std of the SAT-493M training images (about 110 / 105 / 75 ± 54 / 40 / 36 in 0-255 levels) |
| `"reflectance"` | `(0, REFLECTANCE_WHITE)` = `(0, 0.3)` |
| `"uint8"` | `(0, 255)` |
| `"unit"` | `(0, 1)`: the data are already in range |
| `(lo, hi)` | explicit |

Raw units fed unchanged (Sentinel-2 DN, scaled floats of order 1e6, ...) give
a white or black image and tokens that carry almost nothing. With `"auto"`
the embeddings no longer depend on the units. For a time series, pass the
`res.input_range` of the first date to the others so that every date shares
one mapping. `scale_input(data, input_range, verbose=True)` does the same
mapping on its own, to check a store before running the network; it prints
the per-band mean / std in 0-255 levels next to the SAT-493M ones.

The constants are those of Meta's transform for the SAT-493M weights
(`ToTensor()` then `Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156,
0.143))`). Note that SAT-493M was trained on sub-metre imagery: at Sentinel-2's
10 m a 16-px patch covers ~160 m, and objects look 15-20 times smaller than in
training -- a gap no radiometric mapping removes.

---

## One vector per image: `pooling`

DINOv3 returns one token per 16 x 16 px patch plus a global CLS token: a 64 px
image gives 16 patch tokens and one CLS. `pooling` chooses the vector kept:

| `pooling` | vector | describes |
|---|---|---|
| `"cls"` (package default) | CLS token | the whole image |
| `"mean"` | mean of all patch tokens | the whole image, every patch equal |
| `"cls+mean"` | both, concatenated (2 x Ndino) | global and local |
| `"centre"` | mean of the patch tokens covering the parent cell at the image centre | the cell itself, the rest of the image serving as context |

With images larger than their cell (`TILE_PX = 64` for 32 px cells), only
`"centre"` describes the cell rather than its surroundings: it averages the
central 2 x 2 tokens. The window is a centred square of the cell's size,
widened when needed to stay centred on the token grid. The notebooks use it.

---

## Sliding window over the images: `stride`

In tangent mode, `stride` sets the step between image centres, in parent
cells. `stride=1` (default) gives one image per parent cell. `stride=1/2`
adds the images centred between two neighbouring cells and between four, with
the same projection, resolution and size: a block of `G x G` cells gives
`(2G - 1)**2` images inside it, e.g. 31 x 31 for 16 x 16.

```python
res = GetDINOV3SAT(rgb, cell_id, level=20, parent_level=15, projection="tangent",
                   tile_px=64, stride=0.5, model=model)
res.embedding        # [4 M, Ndino]  four images per parent cell
res.cell_id          # [4 M]  cells of res.cell_level = parent_level + 1
res.centre_lon, res.centre_lat   # image centres
res.on_parent_grid   # True for the images that stride=1 gives
```

The centres are exact HEALPix points, defined across face boundaries: the
north corners (largest face `x` and `y`) of the cells of
`parent_level + log2(1/stride)` (`stride_centres`). The images centred on a
parent cell are exactly those of `stride=1`.

`stride` is not `over_sample`: `over_sample=n` keeps one image per cell and
shifts the windows by `16/n` px *inside* it to get a denser token field;
`stride` moves the images themselves. The two cannot be combined, and
`stride < 1` does not return patch tokens.

`Notebooks/dino_debug_petite_zone.ipynb` checks all this on a 512 x 512 px
zone: the images DINO receives, the classes (UMAP + k-means) with
`stride=1` and `stride=1/2`.

---

## Orientation and geometry

Inside a HEALPix base face the local axes are rotated by 45° with respect to
East/North: the north corner of a face is at `(x, y) = (max, max)`. Tiles are
built with `row = S - 1 - y`, `col = x`, so North points to the top-right
corner. The orientation is the same for all tiles of one face and rotated on
the polar faces. For a satellite backbone (no gravity direction) this is a
mild effect; when fine-tuning, use rotation augmentations rather than image
flips. Pixel-shape distortion of HEALPix cells (equal area, variable shape)
is negligible at the scale of a 16 px patch.

---

## Example

### The data: GRID4EARTH Sentinel-2

The public part of the OVH `grid4earth` bucket is served at
<https://data.grid4earth.eu> (see `docs/bucket-layout.md` of
[GRID4EARTH/project-guidelines](https://github.com/GRID4EARTH/project-guidelines)).
Only the `public` directory is readable without credentials, and the proxy
maps it to the root of the domain, so every URL starts with one of
`converted/`, `auxiliary/`, `eopf-mirror/`, `reprocessed/`, `legacy/`:

```
https://data.grid4earth.eu/converted/sentinel-2-l2a/<PRODUCT_ID>.zarr
```

The prefix is `converted/`, not `legacy/`: `legacy/` holds the untouched
originals and, as of September 2026, serves nothing for Sentinel-2. The
HEALPix conversions live under `converted/`.

Two things differ from a classic data cube:

- **the HEALPix level is a zarr group**, not a dimension —
  `<PRODUCT_ID>.zarr/measurements/reflectance/<level>` — so one store holds
  one acquisition at several levels;
- **one product = one acquisition**: a time series is a *list of products*,
  not a `time` axis;
- the stores are **zarr v3** (`zarr.json`, no `.zgroup`), publish levels 17,
  19 and 20, hold `b02`/`b03`/`b04` on a `cells` dimension with a `cell_ids`
  coordinate, and declare their **reference ellipsoid as WGS84**. That last
  point is not cosmetic: reading those cell ids as if they were on a sphere
  displaces every centre by up to ~0.2° of latitude. `ProductSeries` reads the
  declaration and the examples pass it to `set_ellipsoid` before computing
  anything.

`Notebooks/g4e_source.py` wraps this: `ProductSeries` opens the products
(obstore + `zarr.storage.ObjectStore` when installed, fsspec otherwise) and
exposes `src.level`, `src.cell_id`, `src.dates`, `src.rgb(i)` — reflectances
in [0, 1], NaN where missing, with retries and an optional on-disk cache.

```python
from g4e_source import ProductSeries, G4E_PRODUCTS
from healpix_analyse.dino import GetDINOV3SAT, load_dinov3_sat

src = ProductSeries(G4E_PRODUCTS, level=20)   # measurements/reflectance/20
rgb = src.rgb(0)                              # [N, 3] float32, R,G,B = b04,b03,b02

model = load_dinov3_sat("dinov3_vitl16", weights="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
res = GetDINOV3SAT(rgb, src.cell_id, level=src.level, parent_level=12,
                   projection="percell", model=model, min_coverage=1.0)

res.embedding.shape        # (M, 1024)   exactly one per level-12 cell, no hole
res.cell_id.shape          # (M,)        the level-12 NESTED ids
```

`Notebooks/dino_umap_sentinel2.py` runs this over the products, projects the
tokens with UMAP, clusters the UMAP coordinates with k-means and draws the
cluster maps next to the RGB tiles (unsupervised classification test):

```
python Notebooks/dino_umap_sentinel2.py --source g4e --level 20 \
       --parent-level 12 --projection percell --clusters 16
```

`--fake` runs the whole pipeline with a random backbone when the weights are
not available; `--cache DIR` keeps the downloaded reflectances on disk.

---

## Licence and provenance — please read

`healpix-analyse` is released under the **Apache License 2.0**. This module
is an independent, HEALPix-only adapter and is deliberately kept separate
from the model it calls:

- **No DINOv3 code is copied or vendored** in this package. The backbone is
  loaded at run time from Meta's own distribution — the
  [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3)
  torch-hub repository or the `facebook/dinov3-*` Hugging Face models — via
  `torch.hub.load` / `transformers.AutoModel`.
- **No DINOv3 weights are redistributed.** The SAT-493M checkpoints are
  gated: you must request them from Meta and accept the **DINOv3 License**
  (see `LICENSE.md` in the DINOv3 repository and the model cards on Hugging
  Face) before downloading. `healpix-analyse` never downloads them for you
  and never ships them.
- **The network is not modified** — no architecture change, no fine-tuning,
  no re-implementation. This module only reorders HEALPix cells into images
  (`nested_to_tiles`) and maps the tokens back to cells (`tile_grid_cell_ids`);
  these helpers are original code, useful with any 16-px-patch ViT.
- The use you make of the DINOv3 model and of its outputs is governed by the
  DINOv3 License, not by the Apache licence of this package. Check that your
  use case (research, commercial, redistribution of derived products) is
  permitted by that licence, and cite the DINOv3 paper
  (Siméoni et al., 2025, *DINOv3*, arXiv:2508.10104) when publishing results
  obtained with it.

The normalisation constants `SAT493M_MEAN` / `SAT493M_STD` are the values
published in the DINOv3 model card for the satellite models; verify them
against the version of the model you download.
