# `GetDINOV3SAT` — DINOv3 embeddings of HEALPix data

**Module** `healpix_analyse.dino`  
**Function** `GetDINOV3SAT`  
**Status** experimental

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

## Getting the weights (required step)

The SAT-493M checkpoints are **gated**: `healpix-analyse` never downloads
them, and calling `GetDINOV3SAT` / `load_dinov3_sat` without `weights`
raises an error on purpose. Without it, torch-hub would silently fall back to
the *web* checkpoint (`lvd1689m`, a different model) and fail with
`HTTP Error 403: Forbidden` anyway.

**Route A — torch-hub (recommended)**

1. Request access on Meta's DINOv3 download page,
   <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>,
   and accept the **DINOv3 License**. You receive personalised download URLs
   by e-mail (they expire after a few days).
2. Download the satellite ViT-L/16 checkpoint,
   `dinov3_vitl16_pretrain_sat493m-<hash>.pth` (≈ 1.2 GB), and copy it to the
   machine where you run the code (e.g. Datarmor).
3. Pass its path as `weights`:

   ```python
   model = load_dinov3_sat("dinov3_vitl16",
                           weights="/path/to/dinov3_vitl16_pretrain_sat493m-<hash>.pth")
   ```

   or, for the example script,
   `python Notebooks/dino_umap_sentinel2.py --weights /path/to/dinov3_vitl16_pretrain_sat493m-<hash>.pth`.

   `weights` also accepts the personalised URL directly. The DINOv3 code is
   fetched by `torch.hub` from `facebookresearch/dinov3` on first use (cached in
   `~/.cache/torch/hub/`); on a machine without internet access, clone that
   repository and pass its path as `repo=` / `--repo`.

**Route B — Hugging Face**

1. Accept the licence on the model page
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m>.
2. `pip install transformers` and `huggingface-cli login`.
3. `load_dinov3_sat("dinov3_vitl16", source="hf")` or
   `python Notebooks/dino_umap_sentinel2.py --hf`.

To test the pipeline without any weights, use `--fake` (random backbone: the
embeddings are meaningless, only the HEALPix plumbing is exercised).

---

## Example

```python
import xarray as xr
from healpix_analyse.dino import GetDINOV3SAT, load_dinov3_sat

ds = xr.open_zarr("https://data-taos.ifremer.fr/EGU25_CFOSAT/Sentinel2_test.zarr")   # level 19
rgb = ds.Sentinel2.isel(time=0).sel(bands=["b04", "b03", "b02"]).transpose("cells", "bands").values / 1e4

model = load_dinov3_sat("dinov3_vitl16", weights="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
res = GetDINOV3SAT(rgb, ds.cell_ids.values, level=19, parent_level=11,
                   model=model, return_patches=True, min_coverage=0.5)

res.embedding.shape        # (M, 1024)   one per level-11 cell
res.patch_embedding.shape  # (M*256, 1024)  one per level-15 cell -> res.patch_cell_id
```

`Notebooks/dino_umap_sentinel2.py` runs this on all 88 dates of the demo
store, projects the patch tokens with UMAP, clusters them with k-means and
draws the cluster maps next to the RGB scene in lon/lat with `healpix_plot`
(unsupervised classification test; needs `healpix-plot`, `cartopy`, `umap-learn`,
`scikit-learn`). `--fake` runs the whole pipeline with a random backbone when the
weights are not available.

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
