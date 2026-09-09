"""
DINOv3 SAT-493M embeddings of Sentinel-2 HEALPix data + UMAP / k-means
======================================================================

Unsupervised-classification smoke test of :func:`healpix_analyse.dino.GetDINOV3SAT`
on the GRID4EARTH Sentinel-2 HEALPix demo store (level 19, NESTED,
bands b02/b03/b04/b08, 88 dates, tile T32UPC).

Pipeline
--------
1. open the zarr store, take the RGB bands (b04, b03, b02), scale DN/10000;
2. for every selected date, cut the NESTED domain into DINO tiles of
   ``2**tile_levels`` px and run DINOv3; keep the **patch tokens**, i.e. one
   1024-d vector per HEALPix cell of level ``level - 4`` (16 px = 160 m);
3. UMAP (2-d) of the L2-normalised patch tokens, k-means in embedding space;
4. figures: UMAP scatter coloured by cluster, and for a few dates the RGB
   tile next to the cluster map (both reordered with ``nested_to_tiles``);
5. a temporal-consistency score: fraction of dates on which a cell keeps its
   majority label (a purely unsupervised sanity check).

Usage
-----
    python dino_umap_sentinel2.py --weights /path/to/dinov3_vitl16_pretrain_sat493m-*.pth
    python dino_umap_sentinel2.py --fake              # no weights: random backbone, checks the plumbing only
    python dino_umap_sentinel2.py --hf                # weights from Hugging Face (needs `huggingface-cli login`)

Requirements: healpix_analyse, xarray, zarr, umap-learn, scikit-learn,
matplotlib, and the DINOv3 backbone (torch-hub clone of
facebookresearch/dinov3 or `transformers`).  DINOv3 code and weights are
distributed by Meta under the DINOv3 License (gated download); see
docs/dino.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from healpix_analyse.dino import (
    GetDINOV3SAT,
    load_dinov3_sat,
    nested_to_tiles,
)

DEFAULT_ZARR = "https://data-taos.ifremer.fr/EGU25_CFOSAT/Sentinel2_test.zarr"
RGB = ("b04", "b03", "b02")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FakeDino(nn.Module):
    """Random ViT-like stand-in (patch 16, dim 64) to test the plumbing."""

    def __init__(self, dim: int = 64):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Conv2d(3, dim, 16, stride=16)
        self.mix = nn.Linear(dim, dim)

    def forward_features(self, x):
        p = self.proj(x).flatten(2).transpose(1, 2)
        p = torch.tanh(self.mix(p))
        return {"x_norm_clstoken": p.mean(1), "x_norm_patchtokens": p}


def open_store(url: str) -> xr.Dataset:
    """Open the (zarr v2) store with either zarr 2.x or zarr 3.x."""
    try:
        return xr.open_zarr(url, zarr_format=2)
    except TypeError:
        return xr.open_zarr(url)


def healpix_level(ds: xr.Dataset, default: int = 19) -> int:
    a = ds["cell_ids"].attrs
    if "level" in a:
        return int(a["level"])
    if "resolution" in a:
        return int(a["resolution"])
    if "nside" in a:
        return int(np.log2(int(a["nside"])))
    print(f"[warn] no level attribute on cell_ids, assuming level {default}")
    return default


def rgb_at(ds: xr.Dataset, t: int) -> np.ndarray:
    """[N, 3] reflectances in [0, 1] (NaN kept) for date index ``t``."""
    x = ds["Sentinel2"].isel(time=t).sel(bands=list(RGB)).transpose("cells", "bands").values
    x = x.astype(np.float32) / 10000.0
    x[np.isfinite(x)] = np.clip(x[np.isfinite(x)], 0.0, 1.0)
    return x


def scene_tiles(values: np.ndarray, cell_id: np.ndarray, level: int, tile_levels: int):
    """Reorder a scene into a few display tiles of ``2**tile_levels`` px."""
    tiles, pids, valid, cov = nested_to_tiles(values, cell_id, level, level - tile_levels, fill="nan")
    keep = cov > 0.2
    return tiles[keep], pids[keep], valid[keep]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--weights", default=None, help="DINOv3 SAT-493M .pth (torch-hub route)")
    ap.add_argument("--repo", default="facebookresearch/dinov3",
                    help="torch-hub repo or local clone of facebookresearch/dinov3")
    ap.add_argument("--model-name", default="dinov3_vitl16")
    ap.add_argument("--hf", action="store_true", help="load weights from Hugging Face instead")
    ap.add_argument("--fake", action="store_true", help="random backbone (no weights needed)")
    ap.add_argument("--tile-levels", type=int, default=8,
                    help="DINO tile side = 2**tile_levels px (8 -> 256 px, 16x16 patches)")
    ap.add_argument("--times", default="all", help="'all', an int (first n dates) or 'a:b'")
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--umap-max", type=int, default=60000, help="max vectors for UMAP fit")
    ap.add_argument("--n-show", type=int, default=4, help="dates shown as RGB/cluster maps")
    ap.add_argument("--out", default="dino_umap_out")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ---- data -------------------------------------------------------------
    ds = open_store(args.zarr)
    level = healpix_level(ds)
    cell_id = ds["cell_ids"].values.astype(np.int64)
    n_time = ds.sizes["time"]
    if args.times == "all":
        times = list(range(n_time))
    elif ":" in args.times:
        a, b = args.times.split(":")
        times = list(range(int(a or 0), int(b or n_time)))
    else:
        times = list(range(min(int(args.times), n_time)))
    print(f"store: {args.zarr}\n  level {level}, {cell_id.size} cells, {len(times)}/{n_time} dates")

    # ---- model ------------------------------------------------------------
    if args.fake:
        model = FakeDino()
        print("  backbone: FakeDino (random weights; plumbing test only)")
    else:
        model = load_dinov3_sat(
            args.model_name, args.weights, source="hf" if args.hf else "hub",
            repo=args.repo, device=args.device,
        )
        print(f"  backbone: {args.model_name} ({'HF' if args.hf else 'torch-hub'})")

    # ---- embeddings -------------------------------------------------------
    parent_level = level - args.tile_levels
    emb, pid, tid, cov_all = [], [], [], []
    for t in times:
        rgb = rgb_at(ds, t)
        res = GetDINOV3SAT(
            rgb, cell_id, level, parent_level,
            model=model, return_patches=True, min_coverage=args.min_coverage,
            device=args.device,
        )
        if res.patch_embedding.shape[0] == 0:
            continue
        emb.append(res.patch_embedding.astype(np.float32))
        pid.append(res.patch_cell_id)
        tid.append(np.full(res.patch_cell_id.size, t, dtype=np.int32))
        cov_all.append(res.coverage)
        print(f"  t={t:3d}  tiles={res.cell_id.size:3d}  patches={res.patch_cell_id.size:6d}"
              f"  coverage={res.coverage.mean():.2f}  [{time.time() - t0:.0f}s]")

    emb = np.concatenate(emb)
    pid = np.concatenate(pid)
    tid = np.concatenate(tid)
    patch_level = level - 4
    print(f"embeddings: {emb.shape}  (one per level-{patch_level} cell and date)")
    np.savez_compressed(os.path.join(args.out, "patch_embeddings.npz"),
                        embedding=emb, cell_id=pid, time=tid, patch_level=patch_level)

    # ---- UMAP + k-means ---------------------------------------------------
    import umap
    from sklearn.cluster import KMeans

    z = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(z.shape[0], size=min(args.umap_max, z.shape[0]), replace=False)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)
    u_fit = reducer.fit_transform(z[fit_idx])
    u = reducer.transform(z) if z.shape[0] > fit_idx.size else u_fit

    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=0).fit(z[fit_idx])
    label = km.predict(z)
    np.savez_compressed(os.path.join(args.out, "umap_labels.npz"), umap=u, label=label,
                        cell_id=pid, time=tid)

    # temporal consistency: does a cell keep its majority label across dates?
    uniq, inv = np.unique(pid, return_inverse=True)
    counts = np.zeros((uniq.size, args.clusters), np.int64)
    np.add.at(counts, (inv, label), 1)
    consistency = counts.max(1) / counts.sum(1)
    print(f"temporal consistency (majority-label fraction per cell): "
          f"mean {consistency.mean():.3f}, median {np.median(consistency):.3f}")

    # ---- figures ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10" if args.clusters <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    sub = rng.choice(u.shape[0], size=min(40000, u.shape[0]), replace=False)
    ax.scatter(u[sub, 0], u[sub, 1], c=label[sub], cmap=cmap, s=1.5, alpha=0.6,
               vmin=-0.5, vmax=cmap.N - 0.5)
    ax.set_title(f"UMAP of DINOv3 SAT patch tokens (level {patch_level}), k-means k={args.clusters}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "umap.png"), dpi=150)
    plt.close(fig)

    show = times[:: max(1, len(times) // args.n_show)][: args.n_show]
    disp_levels = args.tile_levels + 1           # display tiles = 2 DINO tiles wide
    rows = []
    for t in show:
        rgb = rgb_at(ds, t)
        rgb_tiles, disp_pids, _ = scene_tiles(rgb, cell_id, level, disp_levels)
        m = tid == t
        lab_tiles, lab_pids, lab_valid = scene_tiles(
            label[m].astype(np.float32), pid[m], patch_level, disp_levels - 4)
        rows.append((t, rgb_tiles, disp_pids, lab_tiles, lab_pids, lab_valid))

    n_tiles = max(len(r[2]) for r in rows)
    fig, axes = plt.subplots(2 * len(rows), n_tiles, figsize=(3.2 * n_tiles, 6.4 * len(rows)),
                             squeeze=False)
    for r, (t, rgb_tiles, disp_pids, lab_tiles, lab_pids, lab_valid) in enumerate(rows):
        date = str(ds["time"].values[t])[:10]
        for j in range(n_tiles):
            a_rgb, a_lab = axes[2 * r, j], axes[2 * r + 1, j]
            a_rgb.set_axis_off(); a_lab.set_axis_off()
            if j >= len(disp_pids):
                continue
            img = np.transpose(rgb_tiles[j], (1, 2, 0))
            p = np.nanpercentile(img, 98)
            a_rgb.imshow(np.clip(np.nan_to_num(img) / max(p, 1e-3), 0, 1))
            a_rgb.set_title(f"{date}  tile {disp_pids[j]} (L{level - disp_levels})", fontsize=8)
            jj = np.where(lab_pids == disp_pids[j])[0]
            if jj.size:
                lab = lab_tiles[jj[0], 0]
                lab = np.ma.masked_where(~lab_valid[jj[0]], lab)
                a_lab.imshow(lab, cmap=cmap, vmin=-0.5, vmax=cmap.N - 0.5, interpolation="nearest")
            a_lab.set_title(f"k-means labels (L{patch_level})", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "cluster_maps.png"), dpi=150)
    plt.close(fig)

    print(f"done in {time.time() - t0:.0f}s -> {args.out}/ (umap.png, cluster_maps.png, *.npz)")


if __name__ == "__main__":
    sys.exit(main())
