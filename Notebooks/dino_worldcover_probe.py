"""
Downstream test of the UTM fingerprint: a linear probe for ESA WorldCover classes.

Reads the embeddings saved by ``dino_utm_vs_healpix.py --save-emb`` (mean of patch
tokens, one vector per level-14 ground point, three views: ``utm`` UTM grid,
``utm_c`` UTM resampled to true north, ``hpx`` HEALPix patch) and labels every
ground point with the majority ESA WorldCover 2021 (v200, 10 m) class in a
9 x 9 grid of samples within +-150 m of its centre.

Two content-controlled transfers, each with 5-fold spatial cross-validation
(folds = level-11 parents of the ground points, so that train and test never
share a neighbourhood):

``site``   on every site: a probe trained on the true-north view (what the model
           sees on the central meridian, where UTM is true north), tested on the
           UTM-grid view and on the HEALPix view of held-out points of the same site.
           The drop from the true-north score is the downstream cost of the
           distortion at that site.
``pair``   on the cross-zone pair: a probe trained on one zone's view, tested on
           the other zone's view of held-out points, in UTM and in HEALPix: a
           model calibrated on one side of a zone boundary used on the other side.

    PYTHONPATH=<clone> python Notebooks/dino_worldcover_probe.py --runs ~/dino_utm_bands_bl \
        --configs Notebooks/utm_configs --out ~/dino_worldcover_probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
WC_NAMES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
            50: "built-up", 60: "bare / sparse", 70: "snow and ice", 80: "water",
            90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}


def wc_tile(lon, lat):
    la = int(np.floor(lat / 3.0) * 3)
    lo = int(np.floor(lon / 3.0) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def worldcover_labels(lon, lat, half_m=150.0, n=9):
    """Majority WorldCover class in an n x n grid within +-half_m of each point."""
    import rasterio
    lon = (np.asarray(lon, float) + 180.0) % 360.0 - 180.0     # healpix-geo gives [0, 360)
    u = np.linspace(-half_m, half_m, n)
    dy, dx = np.meshgrid(u, u, indexing="ij")
    dlat = dy.ravel() / 111320.0
    labels = np.zeros(len(lon), dtype=np.int64)
    pts_lon = lon[:, None] + dx.ravel()[None] / (111320.0 * np.cos(np.radians(lat[:, None])))
    pts_lat = lat[:, None] + dlat[None]
    tiles = np.array([[wc_tile(a, b) for a, b in zip(r1, r2)] for r1, r2 in zip(pts_lon, pts_lat)])
    vals = np.zeros(pts_lon.shape, dtype=np.int64)
    for t in np.unique(tiles):
        m = tiles == t
        try:
            with rasterio.open(WC_URL.format(tile=t)) as src:
                vals[m] = np.array([v[0] for v in src.sample(zip(pts_lon[m], pts_lat[m]))])
        except Exception as e:                                  # noqa: BLE001
            print(f"WorldCover tile {t} not readable ({e}); its samples left unlabelled")
            vals[m] = 0
    for i in range(len(lon)):
        labels[i] = Counter(vals[i]).most_common(1)[0][0]
    return labels


def folds_of(cells, k=5, seed=0):
    """Spatial folds: level-11 parents of the level-14 cells, assigned at random."""
    par = cells >> 6
    up, inv = np.unique(par, return_inverse=True)
    f = np.random.default_rng(seed).integers(0, k, up.size)
    return f[inv]


def probe(Xtr, ytr, Xte_list, yte, folds, k=5):
    """Cross-validated accuracy and macro-F1 of a probe trained on Xtr, tested on each
    of Xte_list (same points, other views), held-out folds only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    norm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
    Xtr = norm(Xtr.astype(np.float32))
    Xte_list = [norm(X.astype(np.float32)) for X in Xte_list]
    preds = [np.full(len(yte), -1) for _ in Xte_list]
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(np.unique(ytr[tr])) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr[tr], ytr[tr])
        for p, X in zip(preds, Xte_list):
            p[te] = clf.predict(X[te])
    out = []
    for p in preds:
        ok = p >= 0
        out.append(dict(acc=float(accuracy_score(yte[ok], p[ok])),
                        f1=float(f1_score(yte[ok], p[ok], average="macro"))))
    return out


def keep_classes(y, min_n=15):
    c = Counter(y)
    return np.array([v > 0 and c[v] >= min_n for v in y])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.expanduser("~/dino_utm_bands_bl"))
    p.add_argument("--configs", default="Notebooks/utm_configs")
    p.add_argument("--bands", nargs="+", default=["equator_highlands:equator_kenya_highlands",
                                                  "europe_france:europe_france",
                                                  "high_norway:high_norway"])
    p.add_argument("--out", default=os.path.expanduser("~/dino_worldcover_probe.json"))
    a = p.parse_args()

    result = {}
    for spec in a.bands:
        band, cfgname = spec.split(":")
        cfg = json.load(open(os.path.join(a.configs, cfgname + ".json")))
        res = {"sites": {}, "pairs": {}}
        data = {}
        for f in sorted(glob.glob(os.path.join(a.runs, band, "emb_*.npz"))):
            name = os.path.basename(f)[4:-4]
            z = np.load(f)
            y = worldcover_labels(z["lon"].astype(float), z["lat"].astype(float))
            data[name] = dict(z=z, y=y)
        for name, d in data.items():
            if name.startswith("null"):
                continue
            z, y = d["z"], d["y"]
            m = keep_classes(y)
            if len(np.unique(y[m])) < 2:
                continue
            fo = folds_of(z["cells"][m])
            base, utm, hpx = probe(z["utm_c"][m], y[m], [z["utm_c"][m], z["utm"][m], z["hpx"][m]],
                                   y[m], fo)
            res["sites"][name] = dict(gamma=float(np.median(np.abs(z["gamma"]))), n=int(m.sum()),
                                      classes={WC_NAMES.get(int(c), str(c)): int(v)
                                               for c, v in Counter(y[m]).items()},
                                      true_north=base, utm=utm, hpx=hpx)
            print(band, name, f"|g|={res['sites'][name]['gamma']:.2f}",
                  f"acc trueN {base['acc']:.3f} -> UTM {utm['acc']:.3f}, HEALPix {hpx['acc']:.3f}",
                  flush=True)
        for n1, n2, label in cfg["pairs"]:
            if label != "cross-zone" or n1 not in data or n2 not in data:
                continue
            z1, z2 = data[n1]["z"], data[n2]["z"]
            common, i1, i2 = np.intersect1d(z1["cells"], z2["cells"], return_indices=True)
            y = data[n1]["y"][i1]
            m = keep_classes(y)
            fo = folds_of(common[m])
            out = {}
            for kind in ("utm", "utm_c", "hpx"):
                A, B = z1[kind][i1][m], z2[kind][i2][m]
                same, cross = probe(A, y[m], [A, B], y[m], fo)
                same2, cross2 = probe(B, y[m], [B, A], y[m], fo)
                out[kind] = dict(same=dict(acc=(same["acc"] + same2["acc"]) / 2,
                                           f1=(same["f1"] + same2["f1"]) / 2),
                                 cross=dict(acc=(cross["acc"] + cross2["acc"]) / 2,
                                            f1=(cross["f1"] + cross2["f1"]) / 2))
            res["pairs"][label] = dict(n=int(m.sum()), classes={WC_NAMES.get(int(c), str(c)): int(v)
                                                                 for c, v in Counter(y[m]).items()},
                                       **out)
            print(band, label, {k: (round(v["same"]["acc"], 3), round(v["cross"]["acc"], 3))
                                for k, v in out.items()}, flush=True)
        result[band] = res
    json.dump(result, open(a.out, "w"), indent=1)
    print("written", a.out)


if __name__ == "__main__":
    main()
