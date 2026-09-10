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
2b. by default every tile is resampled onto a local tangent plane, so the
   shapes are right (--projection nested for the raw index reshaping);
3. UMAP of the L2-normalised patch tokens, then k-means **on the UMAP
   coordinates** (--cluster-space dino to cluster the raw embeddings instead);
4. figures: UMAP scatter coloured by cluster, and for a few dates the RGB
   scene next to the cluster map, drawn in lon/lat with ``healpix_plot``;
5. a temporal-consistency score: fraction of dates on which a cell keeps its
   majority label (a purely unsupervised sanity check).

Usage
-----
    python dino_umap_sentinel2.py --weights /path/to/dinov3_vitl16_pretrain_sat493m-*.pth
    python dino_umap_sentinel2.py --fake              # no weights: random backbone, checks the plumbing only
    python dino_umap_sentinel2.py --hf                # weights from Hugging Face (needs `huggingface-cli login`)

Requirements: healpix_analyse, healpix_plot, xarray, zarr, umap-learn,
scikit-learn, matplotlib, cartopy, and the DINOv3 backbone (torch-hub clone of
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

from healpix_analyse.dino import GetDINOV3SAT, load_dinov3_sat

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


def cell_dim(ds: xr.Dataset) -> str:
    """Name of the cell dimension: 'cell_ids' in the raw store, 'cells' after xdggs.decode."""
    return ds["cell_ids"].dims[0]


def rgb_at(ds: xr.Dataset, t: int, *, cache: str | None = None, retries: int = 5) -> np.ndarray:
    """
    [N, 3] reflectances in [0, 1] (NaN kept) for date index ``t``.

    One date is a single zarr chunk of a few hundred MB pulled over HTTP, and
    the server does drop connections, so the read is retried with an
    exponential back-off.  With ``cache``, each date is kept locally as float16
    so that a re-run (or the figures) never downloads it twice.
    """
    fname = os.path.join(cache, f"rgb_{t:04d}.npy") if cache else None
    if fname is not None and os.path.exists(fname):
        return np.load(fname).astype(np.float32)

    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            x = (ds["Sentinel2"].isel(time=t).sel(bands=list(RGB))
                 .transpose(cell_dim(ds), "bands").values)
            break
        except Exception as exc:                                    # noqa: BLE001
            if attempt == retries:
                raise
            print(f"  [retry {attempt}/{retries - 1}] t={t}: "
                  f"{type(exc).__name__}: {exc}; waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2

    x = x.astype(np.float32) / 10000.0
    finite = np.isfinite(x)
    x[finite] = np.clip(x[finite], 0.0, 1.0)
    if fname is not None:
        os.makedirs(cache, exist_ok=True)
        np.save(fname, x.astype(np.float16))
    return x


def plot_maps(ds, cell_id, level, show, pid, tid, label, patch_level, consistency, uniq, cmap,
              cache=None):
    """RGB scene, k-means labels and temporal consistency, drawn in lon/lat with healpix_plot."""
    import cartopy.crs as ccrs
    import healpix_plot
    import matplotlib.pyplot as plt

    grid_px = healpix_plot.HealpixGrid(level=level, indexing_scheme="nested", ellipsoid="WGS84")
    grid_patch = healpix_plot.HealpixGrid(level=patch_level, indexing_scheme="nested", ellipsoid="WGS84")
    n = len(show)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.6 * n), squeeze=False,
                             subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained")
    for r, t in enumerate(show):
        date = str(ds["time"].values[t])[:10]
        rgb = rgb_at(ds, t, cache=cache)
        hi = np.nanpercentile(rgb, 98)
        healpix_plot.plot(cell_id, rgb, healpix_grid=grid_px, sampling_grid={"shape": 768},
                          ax=axes[r, 0], rgb_clip=(0.0, float(max(hi, 1e-3))), axis_labels="none",
                          title=f"{date}  RGB (level {level})")
        m = tid == t
        healpix_plot.plot(pid[m], label[m].astype(np.float32), healpix_grid=grid_patch,
                          sampling_grid={"shape": 768}, ax=axes[r, 1], cmap=cmap,
                          vmin=-0.5, vmax=cmap.N - 0.5, axis_labels="none",
                          title=f"k-means labels (level {patch_level})")
        if r == 0:
            mp = healpix_plot.plot(uniq, consistency.astype(np.float32), healpix_grid=grid_patch,
                                   sampling_grid={"shape": 768}, ax=axes[r, 2], cmap="magma",
                                   vmin=0, vmax=1, axis_labels="none",
                                   title="temporal consistency of the label")
            fig.colorbar(mp, ax=axes[r, 2], shrink=0.8)
        else:
            axes[r, 2].set_axis_off()
        for a in axes[r, :2]:
            a.gridlines(draw_labels=False, linewidth=0.3)
    return fig


def categorical_cmap(n: int):
    """A discrete colormap with exactly ``n`` distinguishable colours."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if n <= 10:
        colors = list(plt.get_cmap("tab10").colors)
    elif n <= 20:
        colors = list(plt.get_cmap("tab20").colors)
    else:
        colors = (list(plt.get_cmap("tab20").colors)
                  + list(plt.get_cmap("tab20b").colors)
                  + list(plt.get_cmap("tab20c").colors))
        if n > len(colors):
            colors = [tuple(c) for c in plt.get_cmap("gist_ncar")(np.linspace(0.02, 0.98, n))]
    return ListedColormap(colors[:n])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--weights", default=None,
                    help="DINOv3 SAT-493M .pth (local path or personalised download URL). "
                         "Required unless --fake/--hf: the weights are gated, request them on "
                         "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
    ap.add_argument("--repo", default="facebookresearch/dinov3",
                    help="torch-hub repo or local clone of facebookresearch/dinov3")
    ap.add_argument("--model-name", default="dinov3_vitl16")
    ap.add_argument("--hf", action="store_true", help="load weights from Hugging Face instead")
    ap.add_argument("--fake", action="store_true", help="random backbone (no weights needed)")
    ap.add_argument("--projection", default="tangent",
                    choices=["tangent", "nested", "percell"],
                    help="'tangent' resamples every tile onto a north-up local tangent plane "
                         "(HEALPix cells have equal area but not equal shape: at 52 deg N the "
                         "nested image is stretched by 2 and sheared by 30 deg). 'nested' is "
                         "the exact index reshaping, geometrically wrong away from the equator). "
                         "'percell' builds one tangent plane per output cell and one embedding "
                         "each -- exact, and one forward pass per cell instead of per tile")
    ap.add_argument("--context-px", type=int, default=None,
                    help="window side in percell mode, a multiple of 16. Default: the cell's "
                         "own footprint, which costs one token per output cell -- the same "
                         "count as the tiled modes. Larger windows buy context at (m**2)")
    ap.add_argument("--over-sample", type=int, default=1,
                    help="sliding-window factor, a power of two: the network runs n**2 times on "
                         "windows shifted by 16/n px, giving a token field n times denser")
    ap.add_argument("--gsd-m", type=float, default=None,
                    help="ground sampling of the tangent grid in metres (default: HEALPix native)")
    ap.add_argument("--interpolation", default="bilinear", choices=["bilinear", "nearest"])
    ap.add_argument("--scan-dates", action="store_true",
                    help="only rank the dates by cloudiness and exit, so a clear scene can be "
                         "picked; each date is still downloaded once (use --cache)")
    ap.add_argument("--tile-levels", type=int, default=8,
                    help="DINO tile side = 2**tile_levels px (8 -> 256 px, 16x16 patches)")
    ap.add_argument("--times", default="all", help="'all', an int (first n dates) or 'a:b'")
    ap.add_argument("--clusters", type=int, default=16,
                    help="number of k-means groups")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="minimum fraction of usable pixels per parent tile. 1.0 (default) "
                         "keeps only COMPLETE tiles: no missing cell and no NaN, so no filler "
                         "value ever reaches the network. Lower it (0.9, 0.5) if too few tiles "
                         "survive because of clouds")
    ap.add_argument("--duplicates", default="mean", choices=["mean", "first", "error"],
                    help="how to aggregate repeated cell ids in the store")
    ap.add_argument("--float16", action="store_true",
                    help="store the patch embeddings as float16 (halves the memory)")
    ap.add_argument("--cache", default=None,
                    help="directory where each date is kept locally (float16), so an "
                         "interrupted run resumes without downloading it again")
    ap.add_argument("--retries", type=int, default=5,
                    help="attempts per date before giving up (the store is served over HTTP)")
    ap.add_argument("--skip-failed", action="store_true",
                    help="skip a date that still fails after --retries instead of stopping")
    ap.add_argument("--umap-max", type=int, default=60000, help="max vectors for UMAP fit")
    ap.add_argument("--token-norm", default="tile", choices=["none", "tile", "pc", "tile+pc"],
                    help="a ViT sees the whole tile through global attention, so a tile-wide "
                         "radiometric offset (haze, illumination) ends up in every one of its "
                         "tokens and the clusters follow the tiles. 'tile' subtracts each "
                         "tile's mean token (the fix, at the price of making the description "
                         "relative to each tile), 'pc' drops the leading principal components")
    ap.add_argument("--drop-pc", type=int, default=1,
                    help="principal components dropped when --token-norm contains 'pc'")
    ap.add_argument("--cluster-space", default="umap", choices=["umap", "dino"],
                    help="run k-means on the UMAP coordinates (default) or directly on the "
                         "1024-d DINOv3 embeddings")
    ap.add_argument("--umap-dim", type=int, default=2,
                    help="UMAP components. 2 is what the scatter shows; 5-10 often clusters "
                         "better while keeping the first two for display")
    ap.add_argument("--umap-neighbors", type=int, default=30,
                    help="UMAP n_neighbors: small = local detail, large = global structure")
    ap.add_argument("--umap-min-dist", type=float, default=0.0,
                    help="UMAP min_dist: 0.0 packs the groups tightly, which suits clustering")
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

    # ---- optional: rank the dates by cloudiness and stop --------------------
    if args.scan_dates:
        rows = []
        for t in times:
            try:
                x = rgb_at(ds, t, cache=args.cache, retries=args.retries)
            except Exception as exc:                                # noqa: BLE001
                print(f"  t={t:3d}  unreadable: {type(exc).__name__}", flush=True)
                continue
            ok = np.isfinite(x).all(axis=1)
            v = x[ok]
            # thin cloud and haze are bright AND grey; blue is raised the most
            bright_grey = float((v.min(axis=1) > 0.18).mean())
            blue_excess = float(v[:, 2].mean() - v[:, 0].mean())
            rows.append((t, str(ds["time"].values[t])[:10], v[:, 0].mean(),
                         bright_grey, blue_excess, 1.0 - ok.mean()))
            print(f"  t={t:3d} {rows[-1][1]}  bright-grey {bright_grey:.3f}  "
                  f"blue-excess {blue_excess:+.4f}  nan {rows[-1][5]:.3f}", flush=True)
        rows.sort(key=lambda r: (r[3], r[4]))
        print("\nclearest first (bright-grey fraction, then blue excess):")
        print(f"{'t':>4} {'date':>12} {'bright-grey':>12} {'blue-excess':>12} {'nan':>7}")
        for t, dt, _m, bg, be, nn in rows[:15]:
            print(f"{t:4d} {dt:>12} {bg:12.3f} {be:+12.4f} {nn:7.3f}")
        print("\nThese are proxies, not a cloud mask: bright-grey counts pixels that are both "
              "bright and unsaturated, blue-excess is raised by haze. Look at the best few "
              "with the notebook before trusting them.")
        return 0

    # ---- model ------------------------------------------------------------
    if not args.fake and not args.hf and args.weights is None:
        ap.error("--weights is required (gated DINOv3 SAT-493M checkpoint); use --fake to test without it")
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
    n_tiles_total = np.unique(cell_id >> (2 * args.tile_levels)).size
    if args.projection != "percell":
        n_patch = (n_tiles_total * 4 ** (args.tile_levels - 4)
                   * args.over_sample ** 2 * len(times))
        gib = n_patch * 1024 * (2 if args.float16 else 4) / 2 ** 30
        print(f"  tiles of {2 ** args.tile_levels} px at level {parent_level}, "
              f"embeddings at level {level - 4 + int(np.log2(args.over_sample))}: "
              f"{n_patch} vectors (~{gib:.1f} GiB)")
        if gib > 4:
            print("  [warn] that is a lot of memory; restrict the dates with --times a:b, "
                  "use --float16, or take larger tiles with --tile-levels")
    if args.projection == "percell":
        cell_px = 16 * max(1, 2 ** (args.tile_levels - 4))
        ctx = args.context_px if args.context_px is not None else cell_px
        g = max(1, ctx // 16)
        n_tok = n_tiles_total * g * g * len(times)
        ref = n_tiles_total * (cell_px // 16) ** 2 * len(times)
        print(f"  percell: {n_tiles_total} cells at level {level - args.tile_levels} "
              f"(footprint {cell_px} px), window {ctx} px = {g}x{g} patches")
        print(f"  {n_tok:,} tokens in total ({n_tok / max(ref, 1):.0f}x the tiled modes), "
              f"{n_tiles_total * len(times):,} forward passes")
    emb, pid, tid, cov_all = [], [], [], []
    failed = []
    for t in times:
        try:
            rgb = rgb_at(ds, t, cache=args.cache, retries=args.retries)
        except Exception as exc:                                    # noqa: BLE001
            if not args.skip_failed:
                raise
            print(f"  [skip] t={t}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(t)
            continue
        res = GetDINOV3SAT(
            rgb, cell_id, level, parent_level,
            model=model, return_patches=True, min_coverage=args.min_coverage,
            duplicates=args.duplicates, device=args.device,
            projection=args.projection, over_sample=args.over_sample,
            gsd_m=args.gsd_m, interpolation=args.interpolation,
            context_px=args.context_px,
        )
        if res.patch_embedding.shape[0] == 0:
            print(f"  t={t:3d}  no tile reaches min_coverage={args.min_coverage}"
                  + ("  [in tangent mode the square grid overruns the diamond-shaped block, "
                     "so tiles at the edge of the domain never reach 1.0]"
                     if args.projection == "tangent" else ""), flush=True)
            continue
        pe = res.patch_embedding.astype(np.float32)
        n_tile = res.cell_id.size
        P = pe.shape[0] // n_tile
        blk = pe.reshape(n_tile, P, -1)
        if P == 1:
            # percell mode: one token per tile, so there is no tile effect to
            # measure and no per-tile mean to remove (it would zero everything)
            if t == times[0] and "tile" in args.token_norm:
                print("  [note] --token-norm 'tile' does not apply in percell mode "
                      "(one token per cell); only the 'pc' part is used")
        else:
            if t == times[0]:
                grand = pe.mean(0)
                frac = ((((blk.mean(1) - grand) ** 2).sum() * P)
                        / (((pe - grand) ** 2).sum() + 1e-12))
                print(f"  variance carried by the per-tile mean token: {frac:.3f}"
                      + ("  [the clusters would follow the tiles]" if frac > 0.5 else ""))
            if "tile" in args.token_norm:
                pe = (blk - blk.mean(axis=1, keepdims=True)).reshape(pe.shape)
        emb.append(pe.astype(np.float16 if args.float16 else np.float32))
        pid.append(res.patch_cell_id)
        tid.append(np.full(res.patch_cell_id.size, t, dtype=np.int32))
        cov_all.append(res.coverage)
        print(f"  t={t:3d}  tiles={res.cell_id.size:3d}/{n_tiles_total}"
              f"  patches={res.patch_cell_id.size:6d}"
              f"  coverage={res.coverage.mean():.3f}  [{time.time() - t0:.0f}s]")

    if failed:
        print(f"[warn] {len(failed)} date(s) could not be read and were skipped: {failed}")
        times = [t for t in times if t not in failed]
    if not emb:
        raise SystemExit(
            f"no embedding was produced: no parent tile reached min_coverage="
            f"{args.min_coverage} (lower it, e.g. --min-coverage 0.9), or no date could be read")

    emb = np.concatenate(emb)
    pid = np.concatenate(pid)
    tid = np.concatenate(tid)
    patch_level = res.patch_level          # level - 4 + log2(over_sample)
    print(f"embeddings: {emb.shape}  (one per level-{patch_level} cell and date)")
    np.savez_compressed(os.path.join(args.out, "patch_embeddings.npz"),
                        embedding=emb, cell_id=pid, time=tid, patch_level=patch_level)

    # ---- UMAP + k-means ---------------------------------------------------
    import umap
    from sklearn.cluster import KMeans

    z = emb.astype(np.float32)
    if "pc" in args.token_norm:
        from sklearn.decomposition import PCA
        pc = PCA(n_components=args.drop_pc, random_state=0).fit(z)
        z = z - pc.inverse_transform(pc.transform(z))
        print(f"dropped the first {args.drop_pc} principal component(s)")
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(z.shape[0], size=min(args.umap_max, z.shape[0]), replace=False)
    reducer = umap.UMAP(n_components=max(2, args.umap_dim), n_neighbors=args.umap_neighbors,
                        min_dist=args.umap_min_dist, metric="cosine", random_state=0)
    u_fit = reducer.fit_transform(z[fit_idx])
    u = reducer.transform(z) if z.shape[0] > fit_idx.size else u_fit

    # k-means on the UMAP coordinates by default: UMAP pulls apart the manifold,
    # which k-means (spherical, equal-variance groups) cannot do on its own in
    # the raw 1024-d space.  The price is that UMAP distorts global distances
    # and is stochastic, so the groups depend on n_neighbors / min_dist / seed.
    space = u if args.cluster_space == "umap" else z
    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=0).fit(space[fit_idx])
    label = km.predict(space)
    print(f"k-means in the {args.cluster_space} space "
          f"({space.shape[1]}-d), K={args.clusters}")
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

    cmap = categorical_cmap(args.clusters)
    fig, ax = plt.subplots(figsize=(7, 6))
    sub = rng.choice(u.shape[0], size=min(40000, u.shape[0]), replace=False)
    ax.scatter(u[sub, 0], u[sub, 1], c=label[sub], cmap=cmap, s=1.5, alpha=0.6,
               vmin=-0.5, vmax=cmap.N - 0.5)
    ax.set_title(f"UMAP of DINOv3 SAT patch tokens (level {patch_level})\n"
                 f"k-means K={args.clusters} in the {args.cluster_space} space")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "umap.png"), dpi=150)
    plt.close(fig)

    show = times[:: max(1, len(times) // args.n_show)][: args.n_show]
    fig = plot_maps(ds, cell_id, level, show, pid, tid, label, patch_level, consistency, uniq, cmap,
                    cache=args.cache)
    fig.savefig(os.path.join(args.out, "cluster_maps.png"), dpi=150)
    plt.close(fig)

    print(f"done in {time.time() - t0:.0f}s -> {args.out}/ (umap.png, cluster_maps.png, *.npz)")


if __name__ == "__main__":
    sys.exit(main())
