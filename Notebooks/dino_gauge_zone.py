"""
DINOv3 on a small zone with ``GetDINOV3SAT(projection="gauge")``, and unsupervised classes.

The zone is the HEALPix cell of ``level - log2(zone_px)`` holding ``(lon, lat)``
(Paris by default) on a GRID4EARTH Sentinel-2 product.  DINOv3 gives one
vector per cell of ``level - 4`` (the tokens folded back on the cells), and
those vectors are classified without supervision in several ways, side by
side, so that the methods can be compared on the same embeddings:

``pca3-kmeans``   k-means on the first 3 principal components (what a PCA-RGB
                  view shows, turned into classes)
``cos-kmeans``    k-means on the L2-normalised vectors (cosine geometry, the
                  one DINO features are trained with)
``umap-kmeans``   UMAP (cosine, 10-D) then k-means
``umap-hdbscan``  UMAP then HDBSCAN: finds the number of classes, leaves the
                  ambiguous cells out
``spectral``      spectral clustering on the cosine k-nearest-neighbour graph
``colour``        k-means on the mean colour of the cell: the baseline that
                  says what DINO adds

Every map is drawn north up on the zone.  Each title gives the spatial
coherence (fraction of the 8 neighbours with the same class; chance = 1/K)
and the adjusted Rand index against the colour baseline (0 = DINO separates
something else than colour).

Run on the GRID4EARTH JupyterHub, where the weights are::

    PYTHONPATH=<clone> python Notebooks/dino_gauge_zone.py --out ~/dino_gauge_paris

``--fake`` replaces DINOv3 by a mean-colour backbone and ``--synthetic`` the
product by a synthetic scene, to check the chain without weights or data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import healpix_analyse.dino as D  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lon", type=float, default=2.35)
    p.add_argument("--lat", type=float, default=48.85)
    p.add_argument("--level", type=int, default=20)
    p.add_argument("--zone-px", type=int, default=1024, help="side of the zone, px (power of 2)")
    p.add_argument("--parent-level", type=int, default=15, help="image centres (cells of 32 px at 20)")
    p.add_argument("--tile-px", type=int, default=64)
    p.add_argument("--product", type=int, default=0)
    p.add_argument("--weights", default="/home/jovyan/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    p.add_argument("--cache", default="/home/jovyan/s2_cache")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rotations", type=int, default=1, choices=(1, 4))
    p.add_argument("--input-range", default="sat493m")
    p.add_argument("--compare-tangent", action="store_true",
                   help="also run the previous tangent mode (stride 1/2, centre pooling)")
    p.add_argument("--out", default="dino_gauge_zone_out")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- data

def synthetic(level, zone_level, lon0, lat0):
    """Fields, a wood, a town and a river, in metres east / north of the zone centre."""
    zone = int(D.cells_at_lonlat(zone_level, np.array([lon0]), np.array([lat0]))[0])
    nb = D.cell_neighbours(zone_level, np.array([zone]))[:, 0]
    kk = level - zone_level
    blocks = np.r_[zone, nb[nb >= 0]]
    ids = ((blocks[:, None] << (2 * kk)) + np.arange(4 ** kk)[None]).ravel()
    lon, lat = D.cell_centres_lonlat(level, ids)
    zl, za = D.cell_centres_lonlat(zone_level, np.array([zone]))
    xi, eta = D.lonlat_to_offsets(zl, za, lon[None], lat[None])
    X, Y = xi[0] * D.EARTH_RADIUS_M, eta[0] * D.EARTH_RADIUS_M
    L = D.healpix_gsd_m(zone_level)
    rng = np.random.default_rng(0)
    rgb = np.empty((ids.size, 3), np.float32)
    field = ((X // (0.11 * L)).astype(int) + 3 * (Y // (0.08 * L)).astype(int)) % 4
    pal = np.array([(0.10, 0.16, 0.06), (0.18, 0.15, 0.08), (0.08, 0.12, 0.05), (0.20, 0.20, 0.12)])
    rgb[:] = pal[field]
    wood = np.hypot(X + 0.25 * L, Y - 0.2 * L) < 0.18 * L
    rgb[wood] = (0.04, 0.08, 0.03)
    rgb[wood] += 0.02 * rng.standard_normal((wood.sum(), 3))
    town = (np.abs(X - 0.2 * L) < 0.2 * L) & (np.abs(Y + 0.15 * L) < 0.15 * L)
    grid = ((X % (0.03 * L)) < 0.006 * L) | ((Y % (0.03 * L)) < 0.006 * L)
    rgb[town] = np.where(grid[town, None], (0.30, 0.30, 0.30), (0.20, 0.17, 0.16))
    river = np.abs(Y - 0.12 * L * np.sin(X / (0.2 * L))) < 0.02 * L
    rgb[river] = (0.03, 0.05, 0.08)
    return ids, rgb, "sphere"


# --------------------------------------------------------------------------- classes

def classify(E, C, K, seed=0):
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import PCA

    out = {}
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    pca = PCA(n_components=3, random_state=seed).fit(E)
    P3 = pca.transform(E)
    out["pca3-kmeans"] = KMeans(K, n_init=10, random_state=seed).fit_predict(P3)
    out["cos-kmeans"] = KMeans(K, n_init=10, random_state=seed).fit_predict(En)
    try:
        import umap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0, metric="cosine",
                          random_state=seed).fit_transform(E)
        out["umap-kmeans"] = KMeans(K, n_init=10, random_state=seed).fit_predict(U)
        from sklearn.cluster import HDBSCAN
        out["umap-hdbscan"] = HDBSCAN(min_cluster_size=max(15, len(E) // 100)).fit_predict(U)
    except ImportError:
        print("umap-learn missing: UMAP classes skipped")
    out["spectral"] = SpectralClustering(
        K, affinity="nearest_neighbors", n_neighbors=15, assign_labels="cluster_qr",
        random_state=seed).fit_predict(En)
    out["colour"] = KMeans(K, n_init=10, random_state=seed).fit_predict(C)
    return out, P3, pca.explained_variance_ratio_


def coherence(cells, labels, level):
    """Fraction of the 8 neighbours (present, classified) with the same label."""
    o = np.argsort(cells)
    sc, sl = cells[o], labels[o]
    nb = D.cell_neighbours(level, sc)                                  # [8, N]
    pos = np.clip(np.searchsorted(sc, nb), 0, sc.size - 1)
    ok = (nb >= 0) & (sc[pos] == nb) & (sl[pos] >= 0) & (sl[None] >= 0)
    return float(((sl[pos] == sl[None]) & ok).sum() / max(ok.sum(), 1))


# --------------------------------------------------------------------------- main

def main():
    a = parse()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import adjusted_rand_score

    from dino_umap_sentinel2 import categorical_cmap

    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out_dir = os.path.expanduser(a.out)
    L, PL = a.level, a.parent_level
    zone_level = L - int(np.log2(a.zone_px))
    TL = L - D.DINOV3_PATCH_LEVELS

    # ---- data: the zone and its 8 neighbours (the images overflow the zone)
    t0 = time.time()
    if a.synthetic:
        ids, rgb, ell = synthetic(L, zone_level, a.lon, a.lat)
        date = "synthetic"
    else:
        from g4e_source import G4E_PRODUCTS, ProductSeries
        src = ProductSeries(G4E_PRODUCTS, level=L)
        ell = getattr(src, "ellipsoid", "sphere")
        D.set_ellipsoid(ell)
        rgb_all = src.rgb(a.product, cache=a.cache)
        ids_all = np.asarray(src.cell_id, dtype=np.int64)
        date = str(src.dates[a.product])
        zk = 2 * (L - zone_level)
        zone = int(D.cells_at_lonlat(zone_level, np.array([a.lon]), np.array([a.lat]))[0])
        znb = D.cell_neighbours(zone_level, np.array([zone]))[:, 0]
        sel = np.isin(ids_all >> zk, np.r_[zone, znb[znb >= 0]])
        ids, rgb = ids_all[sel], rgb_all[sel]
        del rgb_all, ids_all
    D.set_ellipsoid(ell)
    zone = int(D.cells_at_lonlat(zone_level, np.array([a.lon]), np.array([a.lat]))[0])
    zk = 2 * (L - zone_level)
    # keep the parent cells of the zone and enough rings around it for the
    # images that reach the zone to see their data: no image is wasted on the
    # far side of the neighbour blocks
    kp = 2 * (L - PL)
    ring = np.unique(ids[(ids >> zk) == zone] >> kp)
    for _ in range(int(np.ceil(a.tile_px / 2 ** (L - PL))) + 1):
        nbr = D.cell_neighbours(PL, ring)
        ring = np.unique(np.r_[ring, nbr[nbr >= 0].ravel()])
    m = np.isin(ids >> kp, ring)
    ids, rgb = ids[m], rgb[m]
    in_zone = (ids >> zk) == zone
    print(f"{date}: {ids.size:,} cells around the zone {zone} (level {zone_level}, "
          f"{a.zone_px} px), ellipsoid {ell}, read in {time.time() - t0:.0f} s")

    # ---- model
    if a.fake or not os.path.exists(a.weights):
        import torch.nn as nn

        class MeanDino(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward_features(self, x):
                tok = nn.functional.avg_pool2d(x, D.DINOV3_PATCH).flatten(2).transpose(1, 2)
                return {"x_norm_clstoken": tok.mean(1), "x_norm_patchtokens": tok}
        model, label = MeanDino(), "FAKE backbone (mean colour), not DINOv3"
        print("#" * 70 + f"\n{label}\n" + "#" * 70)
    else:
        model, label = D.load_dinov3_sat("dinov3_vitl16", weights=a.weights,
                                         device=a.device), "DINOv3 ViT-L/16 SAT-493M"

    # ---- embeddings: one vector per cell of level - 4
    lo, hi, how = D.input_range_of(rgb[in_zone], a.input_range)
    t0 = time.time()
    res = D.GetDINOV3SAT(rgb, ids, L, PL, model=model, tile_px=a.tile_px, ellipsoid=ell,
                         rotations=a.rotations, input_range=(lo, hi), min_coverage=0.0,
                         batch_size=64, device=a.device)
    t_dino = time.time() - t0
    keep = (res.cell_id >> (2 * (TL - zone_level))) == zone
    cells, E = res.cell_id[keep], res.embedding[keep]
    print(f"{label}: {res.image_cell_id.size} images of {a.tile_px} px -> {cells.size} cells of "
          f"level {TL} in the zone, vectors {E.shape[1]}-D, {t_dino:.0f} s")

    # mean colour of every cell (the baseline)
    par = ids[in_zone] >> (2 * (L - TL))
    up, inv = np.unique(par, return_inverse=True)
    C = np.zeros((up.size, 3))
    np.add.at(C, inv, np.nan_to_num(rgb[in_zone]))
    C /= np.bincount(inv, minlength=up.size)[:, None]
    C = C[np.searchsorted(up, cells)]

    labs, P3, evr = classify(E, C, a.k)
    stats = {}
    for name, lab in labs.items():
        k_eff = int(lab.max()) + 1
        stats[name] = dict(k=k_eff, coherence=coherence(cells, lab, TL),
                           ari_colour=float(adjusted_rand_score(lab, labs["colour"])),
                           unclassified=float((lab < 0).mean()))
        print(f"{name:13s} K={k_eff:2d}  coherence {stats[name]['coherence']:.3f} "
              f"(chance {1 / max(k_eff, 1):.3f})  ARI vs colour {stats[name]['ari_colour']:.3f}")

    # ---- north-up view of the zone
    gsd = D.healpix_gsd_m(L)
    zl, za = D.cell_centres_lonlat(zone_level, np.array([zone]))
    OH, OW = D.parent_cover_px(np.array([zone]), zone_level, gsd, margin_px=0)
    olon, olat = D.tangent_grid_lonlat(zl, za, (OH, OW), gsd / D.EARTH_RADIUS_M)
    olon, olat = olon[0], olat[0]
    view, _ = D.sample_healpix(rgb, ids, L, olon, olat)
    inside = D.cells_at_lonlat(zone_level, olon, olat) == zone
    pix_cell = D.cells_at_lonlat(TL, olon, olat)
    o = np.argsort(cells)
    p = np.clip(np.searchsorted(cells[o], pix_cell), 0, cells.size - 1)
    pix_row = np.where((cells[o][p] == pix_cell) & inside, o[p], -1)

    def raster(v):
        img = np.full(pix_row.shape + np.shape(v)[1:], np.nan)
        img[pix_row >= 0] = np.asarray(v)[pix_row[pix_row >= 0]]
        return img

    lo_v, hi_v = np.nanpercentile(view[inside], [2, 98])
    rgb_img = np.clip((view - lo_v) / (hi_v - lo_v), 0, 1)
    rgb_img[~inside] = 1.0
    p_lo, p_hi = np.percentile(P3, [2, 98], axis=0)
    pca_img = np.clip((raster(P3) - p_lo) / (p_hi - p_lo), 0, 1)
    pca_img[~inside] = 1.0

    def show_classes(ax, lab, title):
        k_eff = max(int(lab.max()) + 1, 1)
        img = raster(np.where(lab < 0, np.nan, lab).astype(float))
        ax.imshow(rgb_img, alpha=0.35, interpolation="nearest")
        ax.imshow(img, cmap=categorical_cmap(k_eff), vmin=-0.5, vmax=k_eff - 0.5,
                  interpolation="nearest", alpha=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    km = a.zone_px * gsd / 1e3
    head = (f"{label} — {date} — zone {a.zone_px} px (~{km:.1f} km) at {a.lon:.2f}E {a.lat:.2f}N — "
            f"images {a.tile_px} px on cells of level {PL}, one vector per cell of level {TL} "
            f"(~{gsd * 16:.0f} m)")

    fig, ax = plt.subplots(1, 2, figsize=(16, 8.5))
    ax[0].imshow(rgb_img, interpolation="nearest"); ax[0].set_title("RGB, north up")
    ax[1].imshow(pca_img, interpolation="nearest")
    ax[1].set_title("first 3 principal components of the DINO vectors as RGB "
                    f"({100 * evr.sum():.0f} % of the variance)")
    for x in ax:
        x.set_axis_off()
    fig.suptitle(head, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_rgb_pca.png"), dpi=110)

    names = [n for n in ("pca3-kmeans", "cos-kmeans", "umap-kmeans", "umap-hdbscan",
                         "spectral", "colour") if n in labs]
    fig, axes = plt.subplots(2, 3, figsize=(21, 14.5))
    for x, n in zip(axes.ravel(), names):
        s = stats[n]
        extra = f", {100 * s['unclassified']:.0f} % left out" if s["unclassified"] > 0 else ""
        show_classes(x, labs[n], f"{n}  K={s['k']}{extra}\ncoherence {s['coherence']:.2f} "
                     f"(chance {1 / max(s['k'], 1):.2f}), ARI vs colour {s['ari_colour']:.2f}")
    for x in axes.ravel()[len(names):]:
        x.set_axis_off()
    fig.suptitle(head, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_classes.png"), dpi=110)

    # ---- the previous mode, for comparison
    if a.compare_tangent:
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        t0 = time.time()
        r_old = D.GetDINOV3SAT(rgb, ids, L, PL, model=model, projection="tangent", stride=0.5,
                               tile_px=a.tile_px, pooling="centre", input_range=(lo, hi),
                               min_coverage=0.0, batch_size=64, device=a.device)
        v_old = D._lonlat_to_vec(r_old.centre_lon, r_old.centre_lat)
        sel_old = (D.cells_at_lonlat(zone_level, r_old.centre_lon, r_old.centre_lat) == zone)
        Eo = r_old.embedding[sel_old]
        Eon = Eo / np.linalg.norm(Eo, axis=1, keepdims=True)
        lab_old = KMeans(a.k, n_init=10, random_state=0).fit_predict(Eon)
        nn_ = NearestNeighbors(n_neighbors=1).fit(v_old[sel_old])
        pv = D._lonlat_to_vec(olon, olat).reshape(-1, 3)
        img_old = lab_old[nn_.kneighbors(pv, return_distance=False)[:, 0]].reshape(olon.shape)
        img_old = np.where(inside, img_old, np.nan)
        print(f"tangent, stride 1/2, centre pooling: {sel_old.sum()} images in the zone, "
              f"{time.time() - t0:.0f} s")
        fig, ax = plt.subplots(1, 3, figsize=(24, 8.5))
        ax[0].imshow(rgb_img, interpolation="nearest"); ax[0].set_title("RGB")
        cm = categorical_cmap(a.k)
        ax[1].imshow(rgb_img, alpha=0.35); ax[1].imshow(img_old, cmap=cm, vmin=-.5,
                                                        vmax=a.k - .5, alpha=.9)
        ax[1].set_title(f"before: tangent, stride 1/2, pooling 'centre' — cos-kmeans K={a.k}")
        show_classes(ax[2], labs["cos-kmeans"],
                     f"now: gauge, one vector per cell of level {TL} — cos-kmeans K={a.k}")
        for x in ax:
            x.set_axis_off()
        fig.suptitle(head, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "3_before_after.png"), dpi=110)

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(dict(label=label, date=date, zone=zone, zone_level=zone_level,
                       n_cells=int(cells.size), n_images=int(res.image_cell_id.size),
                       seconds_dino=t_dino, pca_variance=evr.tolist(), classes=stats), f, indent=1)
    print("figures in", out_dir)


if __name__ == "__main__":
    main()
