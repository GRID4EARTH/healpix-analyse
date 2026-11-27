"""
Is the UTM map distortion visible in DINOv3 embeddings?  UTM tiles against HEALPix.

For every ground point (a cell of level ``CENTRE_LEVEL``) three images of
``N`` px at the HEALPix level-``LEVEL`` spacing (6.1 m at level 20) are fed to
DINOv3 SAT-493M:

``utm``     the UTM tile resampled on its own grid axes (grid north up, UTM
            metres): what a model sees when it is run on the native product
``utm_c``   the same tile resampled on a true-north, ground-metre grid, built
            from the numerical Jacobian of the projection at the point: UTM
            with its distortion (scale factor k, meridian convergence gamma)
            removed
``hpx``     the HEALPix conversion of the tile (``healpix_resample``
            ``PSFResampler``, level ``LEVEL``) cut by
            ``HealPixPatches(mode="gauge", ellipsoid="WGS84")``, north up

The distances between the image vectors (1 - cosine) answer three questions:

1. *sensitivity*: how much DINO moves for a known rotation, shift or scale
   of the same image (on the site at the central meridian);
2. *zone gradient*: ``d(utm, hpx)`` and ``d(utm_c, hpx)`` on sites at
   increasing distance from the central meridian of UTM zone 30;
3. *same ground, two zones*: on the overlap of tiles 30UYU and 31UCP
   (same datatake), ``d(utm_30, utm_31)`` against ``d(hpx_30, hpx_31)``.
   A same-zone overlap (30UXU / 30UYU) is the null test.

Sentinel-2 L2A true-colour (TCI) COGs are read from the public Earth Search
catalogue (AWS).  Run on a GPU node with the DINOv3 weights::

    PYTHONPATH=<healpix-analyse clone> python Notebooks/dino_utm_vs_healpix.py \
        --weights ~/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth --out ~/dino_utm_hpx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAC = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/"

# (name, STAC item, lon, lat): 2025-07-11, S2A and S2B datatakes, UTM 30 / 31
SITES = [
    ("z30_dl0.0", "S2A_30UVU_20250711_0_L2A", -3.00, 48.25),
    ("z30_dl1.0", "S2A_30UWU_20250711_1_L2A", -2.00, 48.25),
    ("z30_dl2.0", "S2B_30UXU_20250711_0_L2A", -1.00, 48.25),
    ("z30_dl3.0", "S2B_30UYU_20250711_0_L2A", 0.00, 48.25),
    ("z30_dl3.75", "S2B_30UYU_20250711_0_L2A", 0.75, 48.25),     # overlap with 31UCP
    ("z31_dl-2.25", "S2B_31UCP_20250711_0_L2A", 0.75, 48.25),
    ("null_30UXU", "S2B_30UXU_20250711_0_L2A", -0.24, 48.30),    # same-zone overlap
    ("null_30UYU", "S2B_30UYU_20250711_0_L2A", -0.24, 48.30),
]
PAIRS = [("z30_dl3.75", "z31_dl-2.25", "cross-zone"), ("null_30UXU", "null_30UYU", "same-zone")]

A_WGS84, F_WGS84 = 6378137.0, 1 / 298.257223563
R_AUTHALIC = 6371007.1809


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.expanduser(
        "~/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"))
    p.add_argument("--out", default=os.path.expanduser("~/dino_utm_hpx"))
    p.add_argument("--level", type=int, default=20)
    p.add_argument("--centre-level", type=int, default=14)
    p.add_argument("--n", type=int, default=128, help="image side, px")
    p.add_argument("--crop", type=int, default=1100, help="UTM crop side, 10 m px")
    p.add_argument("--backend", default="torch")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--config", default=None,
                   help="JSON with 'sites' [[name, item, lon, lat], ...], 'pairs' "
                        "[[a, b, label], ...] and 'sensitivity_site'; default: the "
                        "48 deg N set of this file")
    p.add_argument("--k", type=int, default=8, help="k-means classes for the pair agreement")
    p.add_argument("--model", default="dinov3", choices=("dinov3", "prithvi"),
                   help="dinov3: SAT-493M on the RGB TCI; prithvi: Prithvi-EO-2.0-300M on six "
                        "L2A bands (B02 B03 B04 B8A B11 B12, the 20 m ones on their own grid); "
                        "prithvi takes the bilinear conversion only")
    p.add_argument("--save-emb", action="store_true",
                   help="save the embeddings (mean of patch tokens, float16) per site, for "
                        "downstream probes (Notebooks/dino_worldcover_probe.py)")
    p.add_argument("--resampler", default="psf", choices=("psf", "ct", "bilinear"),
                   help="UTM -> HEALPix: 'psf' (PSFResampler, damped fit: sharpens small "
                        "scales) or 'ct' (CloughTocherResampler, C1 interpolation, no "
                        "deconvolution) or 'bilinear' (the tile sampled bilinearly at the "
                        "HEALPix cell centres: the same interpolation as the UTM images)")
    p.add_argument("--max-centres", type=int, default=800)
    p.add_argument("--lam", type=float, default=0.1,
                   help="PSFResampler damping: level 20 (6.2 m) is finer than the 10 m "
                        "samples, the undamped fit amplifies the noise")
    return p.parse_args()


# --------------------------------------------------------------------------- data

def read_crop(item_id, lon, lat, size):
    """TCI crop of ``size`` px centred on (lon, lat): uint8 [3, H, W], transform, CRS."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    req = urllib.request.Request(STAC + item_id, headers={"User-Agent": "healpix-analyse-test"})
    item = json.load(urllib.request.urlopen(req, timeout=120))
    href = item["assets"]["visual"]["href"]
    with rasterio.open(href) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        e, n = tr.transform(lon, lat)
        row, col = src.index(e, n)
        win = Window(col - size // 2, row - size // 2, size, size)
        # boundless: a window crossing the tile edge keeps its size and origin
        # (a clipped read would shift the whole crop); outside = 0 = TCI nodata
        arr = src.read(window=win, boundless=True, fill_value=0)      # [3, H, W]
        transform = src.window_transform(win)
        crs = src.crs
    return arr, transform, crs, item["properties"]


PRITHVI_ASSETS = ("blue", "green", "red", "nir08", "swir16", "swir22")


def read_source(item_id, lon, lat, size, model):
    """The crop as a list of (float32 [C, h, w], transform), NaN = nodata, plus CRS and
    properties.  ``size`` is in 10 m pixels; a 20 m band gets the same ground window.
    dinov3: the TCI (0-255).  prithvi: six L2A bands as reflectance x 10000."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    req = urllib.request.Request(STAC + item_id, headers={"User-Agent": "healpix-analyse-test"})
    item = json.load(urllib.request.urlopen(req, timeout=120))
    keys = ("visual",) if model == "dinov3" else PRITHVI_ASSETS
    bands, crs = [], None
    for key in keys:
        asset = item["assets"][key]
        with rasterio.open(asset["href"]) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            e, n = tr.transform(lon, lat)
            row, col = src.index(e, n)
            sz = int(round(size * 10.0 / src.res[0]))
            win = Window(col - sz // 2, row - sz // 2, sz, sz)
            arr = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
            transform = src.window_transform(win)
            crs = src.crs
        nod = (arr == 0).all(0)
        if key != "visual":
            rb = (asset.get("raster:bands") or [{}])[0]
            arr = (arr * rb.get("scale", 1e-4) + rb.get("offset", 0.0)) * 1e4
        arr[:, nod] = np.nan
        bands.append((arr, transform))
    return bands, crs, item["properties"]


def sample_multi(bands, centres_en, offsets_en):
    """``sample_utm`` on every band, each on its own grid: [M, C_total, P]."""
    return np.concatenate([sample_utm(arr, tr, centres_en, offsets_en) for arr, tr in bands],
                          axis=1)


def utm_lonlat(transform, crs, H, W):
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    e = transform.c + transform.a * jj
    n = transform.f + transform.e * ii
    lon, lat = tr.transform(e, n)
    return np.asarray(lon), np.asarray(lat)


def jacobian(crs, lon, lat):
    """UTM metres per ground metre east / north at (lon, lat): columns of J [M, 2, 2],
    plus the convergence gamma (deg, grid north from true north) and scale k."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    phi = np.radians(lat)
    e2 = F_WGS84 * (2 - F_WGS84)
    w = np.sqrt(1 - e2 * np.sin(phi) ** 2)
    n_rad = A_WGS84 / w                                   # prime vertical
    m_rad = A_WGS84 * (1 - e2) / w ** 3                   # meridian
    d = 1e-5                                              # degrees
    e0, n0 = tr.transform(lon, lat)
    ee, ne = tr.transform(lon + d, lat)
    en, nn = tr.transform(lon, lat + d)
    g_east = n_rad * np.cos(phi) * np.radians(d)          # ground metres
    g_north = m_rad * np.radians(d)
    east = np.stack([(ee - e0) / g_east, (ne - n0) / g_east], -1)
    north = np.stack([(en - e0) / g_north, (nn - n0) / g_north], -1)
    J = np.stack([east, north], -1)                       # [M, 2(E,N), 2(east,north)]
    gamma = np.degrees(np.arctan2(north[:, 0], north[:, 1]))
    k = np.hypot(north[:, 0], north[:, 1])
    return J, gamma, k


def sample_utm(arr, transform, centres_en, offsets_en):
    """Bilinear samples of the crop at centre + offset (UTM metres).

    centres_en [M, 2], offsets_en [M, P, 2] -> float [M, 3, P]."""
    C, H, W = arr.shape
    img = torch.as_tensor(arr, dtype=torch.float32)[None]          # [1, 3, H, W]
    pts = centres_en[:, None, :] + offsets_en                      # [M, P, 2]
    col = (pts[..., 0] - transform.c) / transform.a                # pixel units, edges at ints
    row = (pts[..., 1] - transform.f) / transform.e
    gx = 2 * col / W - 1
    gy = 2 * row / H - 1
    grid = torch.as_tensor(np.stack([gx, gy], -1), dtype=torch.float32)[None]   # [1, M, P, 2]
    out = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)
    return out[0].permute(1, 0, 2).numpy()                         # [M, 3, P]


def grid_offsets(n, s):
    """Image pixel offsets (east, north) in ground metres, row 0 = north, col 0 = west."""
    u = (np.arange(n) - (n - 1) / 2.0) * s
    east = np.tile(u, (n, 1))
    north = -np.repeat(u[:, None], n, 1)
    return np.stack([east.ravel(), north.ravel()], -1)             # [P, 2]


# --------------------------------------------------------------------------- DINO

def make_prithvi(n):
    """Prithvi-EO-2.0-300M encoder for single-frame images of n px (sin-cos positions)."""
    import importlib.util
    from huggingface_hub import hf_hub_download
    repo = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    d = "/tmp/prithvi"
    code = hf_hub_download(repo, "prithvi_mae.py", local_dir=d)
    ckpt = hf_hub_download(repo, "Prithvi_EO_V2_300M.pt", local_dir=d)
    cfg = json.load(open(hf_hub_download(repo, "config.json", local_dir=d)))["pretrained_cfg"]
    spec = importlib.util.spec_from_file_location("prithvi_mae", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    enc = mod.PrithviViT(img_size=n, patch_size=(1, 16, 16), num_frames=1, in_chans=6,
                         embed_dim=cfg["embed_dim"], depth=cfg["depth"],
                         num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"])
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    sd = {k: v for k, v in sd.items() if k != "pos_embed"}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print("Prithvi: missing", missing, "unexpected", len(unexpected))
    enc.prithvi_mean = torch.tensor(cfg["mean"]).view(1, 6, 1, 1)
    enc.prithvi_std = torch.tensor(cfg["std"]).view(1, 6, 1, 1)
    enc.kind = "prithvi"
    return enc.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def make_model(a):
    import healpix_analyse.dino as D
    if a.model == "prithvi":
        return make_prithvi(a.n)
    if a.fake:
        import torch.nn as nn

        class MeanDino(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward_features(self, x):
                tok = nn.functional.avg_pool2d(x, 16).flatten(2).transpose(1, 2)
                return {"x_norm_clstoken": tok.mean(1), "x_norm_patchtokens": tok}
        return MeanDino().cuda() if torch.cuda.is_available() else MeanDino()
    return D.load_dinov3_sat("dinov3_vitl16", weights=a.weights)


def embed(model, imgs, bs=64):
    """imgs [M, C, n, n] (TCI 0-255, or Prithvi reflectance x 1e4) -> dict of [M, D]."""
    import healpix_analyse.dino as D
    dev = next(model.parameters()).device
    prithvi = getattr(model, "kind", "") == "prithvi"
    if prithvi:
        mean, std = model.prithvi_mean.to(dev), model.prithvi_std.to(dev)
    else:
        mean = torch.tensor(D.SAT493M_MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(D.SAT493M_STD, device=dev).view(1, 3, 1, 1)
    out = {"cls": [], "mean": [], "centre": []}
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            x = torch.as_tensor(imgs[i:i + bs], dtype=torch.float32, device=dev)
            if prithvi:
                x = (x - mean) / std
            else:
                x = (x.clamp(0, 255) / 255.0 - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                if prithvi:
                    h = model.forward_features(x)[-1]
                    cls, pt = h[:, 0], h[:, 1:]
                else:
                    cls, pt = D._forward(model, x)
            cls, pt = cls.float(), pt.float()
            T = int(round(pt.shape[1] ** 0.5))
            g = pt.reshape(pt.shape[0], T, T, -1)
            c0 = T // 2 - 1
            out["cls"].append(cls.cpu())
            out["mean"].append(pt.mean(1).cpu())
            out["centre"].append(g[:, c0:c0 + 2, c0:c0 + 2].mean((1, 2)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def cosd(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - (a * b).sum(1)


# --------------------------------------------------------------------------- main

def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    import healpix_geo
    from healpix_resample.psf import PSFResampler
    from pyproj import Transformer
    from healpix_analyse.patches import HealPixPatches

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sites, pairs_cfg, sens_site = SITES, PAIRS, SITES[0][0]
    if a.config:
        cfg = json.load(open(os.path.expanduser(a.config)))
        sites = [tuple(x) for x in cfg["sites"]]
        pairs_cfg = [tuple(x) for x in cfg["pairs"]]
        sens_site = cfg.get("sensitivity_site", sites[0][0])
    model = make_model(a)
    s = R_AUTHALIC * np.sqrt(np.pi / 3) / 2 ** a.level            # image spacing, m
    offs = grid_offsets(a.n, s)
    print(f"images {a.n} px at {s:.3f} m = {a.n * s:.0f} m, centres level {a.centre_level}")

    res, site_info, examples = {}, {}, {}
    for name, item, lon0, lat0 in sites:
        t0 = time.time()
        bands, crs, props = read_source(item, lon0, lat0, a.crop, a.model)
        arr, transform = bands[0]                                   # the 10 m reference grid
        C = sum(b[0].shape[0] for b in bands)
        H, W = arr.shape[1:]
        lon, lat = utm_lonlat(transform, crs, H, W)
        ok = np.isfinite(arr).all(0).ravel()
        # ---- HEALPix conversion of the crop (PSF resampler, WGS84)
        if a.resampler != "bilinear" and len(bands) > 1:
            raise SystemExit("multi-resolution bands: use --resampler bilinear")
        vals = torch.as_tensor(arr.reshape(C, -1)[:, ok], dtype=torch.float32, device=dev)
        if a.resampler == "bilinear":
            # every level-20 cell of the crop: the children of the level-(L-2) cells
            # holding a pixel, sampled bilinearly at their centres
            up = a.level - 2
            par = np.unique(np.asarray(healpix_geo.nested.lonlat_to_healpix(
                lon.ravel()[ok], lat.ravel()[ok], up, ellipsoid="WGS84"), dtype=np.int64))
            hid = ((par[:, None] << 4) + np.arange(16)[None]).ravel()
            hl, ha = healpix_geo.nested.healpix_to_lonlat(hid.astype(np.uint64), a.level,
                                                          ellipsoid="WGS84")
            he, hn = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
                np.asarray(hl, float), np.asarray(ha, float))
            colf = (he - transform.c) / transform.a
            rowf = (hn - transform.f) / transform.e
            inside_c = (colf > 0.5) & (colf < W - 0.5) & (rowf > 0.5) & (rowf < H - 0.5)
            hid = hid[inside_c]
            v = sample_multi(bands, np.zeros((1, 2)),
                             np.stack([he[inside_c], hn[inside_c]], -1)[None])[0]   # [C, K]
            good_c = np.isfinite(v).all(0)
            hid, v = hid[good_c], v[:, good_c]
            o = np.argsort(hid)

            class _R:
                pass
            r = _R()
            r.cell_ids, r.cell_data = hid[o], torch.as_tensor(v[:, o])
            op = None
        elif a.resampler == "ct":
            from healpix_resample.clough_tocher import CloughTocherResampler
            op = CloughTocherResampler(lon.ravel()[ok], lat.ravel()[ok], a.level, device=dev,
                                       ellipsoid="WGS84", dtype=torch.float32, verbose=False)
            r = op.resample(vals)
        else:
            op = PSFResampler(lon.ravel()[ok], lat.ravel()[ok], a.level, device=dev,
                              ellipsoid="WGS84")
            r = op.resample(vals, lam=a.lam, clip=(0.0, 255.0))
        hid = np.asarray(torch.as_tensor(r.cell_ids).cpu()).astype(np.int64)
        hval = torch.as_tensor(r.cell_data).float().cpu()
        o = np.argsort(hid)
        hid, hval = hid[o], hval[:, torch.as_tensor(o)]
        good = torch.isfinite(hval).all(0).cpu().numpy()
        hid, hval = hid[good], hval[:, torch.as_tensor(good)]
        # ---- centres: level-c cells whose whole image lies inside the crop
        to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        lev_c = a.centre_level
        cand = np.unique(hid >> (2 * (a.level - lev_c)))
        clon, clat = healpix_geo.nested.healpix_to_lonlat(cand.astype(np.uint64), lev_c,
                                                          ellipsoid="WGS84")
        clon, clat = np.asarray(clon, float), np.asarray(clat, float)
        ce, cn = to_utm.transform(clon, clat)
        margin = a.n * s * 0.75 + 200.0
        x0, y0 = transform.c, transform.f
        inside = ((ce - x0 > margin) & (x0 + W * 10 - ce > margin)
                  & (y0 - cn > margin) & (cn - (y0 - H * 10) > margin))
        cand, clon, clat, ce, cn = cand[inside], clon[inside], clat[inside], ce[inside], cn[inside]
        if cand.size > a.max_centres:
            pick = np.random.default_rng(0).choice(cand.size, a.max_centres, replace=False)
            pick.sort()
            cand, clon, clat, ce, cn = cand[pick], clon[pick], clat[pick], ce[pick], cn[pick]
        cen = np.stack([ce, cn], -1)
        J, gamma, k = jacobian(crs, clon, clat)
        # ---- the three images
        utm = sample_multi(bands, cen, np.broadcast_to(offs, (cand.size,) + offs.shape)
                           ).reshape(-1, C, a.n, a.n)
        utm_c = sample_multi(bands, cen, np.einsum("mij,pj->mpi", J, offs)
                             ).reshape(-1, C, a.n, a.n)
        P = HealPixPatches(a.level, a.n, cell_ids=hid, centres=lev_c, centre_ids=cand,
                           ellipsoid="WGS84", device=dev, backend=a.backend)
        hpx = P(hval.to(dev)[None])[0, :, 0].transpose(-1, -2).float().cpu().numpy()
        cov = P.coverage[:, 0].float().mean((1, 2)).cpu().numpy()
        del P, op
        sel = ((cov > 0.999) & np.isfinite(utm).all((1, 2, 3)) & np.isfinite(utm_c).all((1, 2, 3))
               & np.isfinite(hpx).all((1, 2, 3)))
        if sel.sum() < 30:
            print(f"{name}: only {int(sel.sum())} complete images (nodata or edge), skipped",
                  flush=True)
            continue
        E = {kind: embed(model, im[sel]) for kind, im in
             (("utm", utm), ("utm_c", utm_c), ("hpx", hpx))}
        res[name] = dict(cells=cand[sel], gamma=gamma[sel], k=k[sel], E=E)
        if a.save_emb:
            np.savez_compressed(os.path.join(a.out, f"emb_{name}.npz"), cells=cand[sel],
                                lon=clon[sel], lat=clat[sel], gamma=gamma[sel],
                                **{kind: E[kind]["mean"].astype(np.float16) for kind in E})
        perm = np.random.default_rng(1).permutation(int(sel.sum()))
        site_info[name] = dict(item=item, platform=props.get("platform"),
                               datatake=props.get("s2:datatake_id"), lon=lon0, lat=lat0,
                               dlon_cm=round(lon0 - (3.0 if crs.to_epsg() == 32631 else -3.0), 3),
                               epsg=crs.to_epsg(), n=int(sel.sum()),
                               gamma_med=float(np.median(gamma)), k_med=float(np.median(k)),
                               content={kk: float(np.median(cosd(E["hpx"][kk], E["hpx"][kk][perm])))
                                        for kk in E["hpx"]})
        mid = int(np.argmin(np.hypot(ce - ce.mean(), cn - cn.mean())[sel]))
        rgb = [0, 1, 2] if C == 3 else [2, 1, 0]
        examples[name] = dict(utm=utm[sel][mid][rgb], utm_c=utm_c[sel][mid][rgb],
                              hpx=hpx[sel][mid][rgb])
        print(f"{name}: {item}, epsg {crs.to_epsg()}, gamma {np.median(gamma):+.3f} deg, "
              f"k {np.median(k):.5f}, {int(sel.sum())} centres, {time.time() - t0:.0f} s",
              flush=True)

    # ---- 1. sensitivity on the central-meridian site
    name0, item0, lon0, lat0 = next((x for x in sites if x[0] == sens_site and x[0] in res),
                                    next(x for x in sites if x[0] in res))
    bands, crs, _ = read_source(item0, lon0, lat0, a.crop, a.model)
    C = sum(b[0].shape[0] for b in bands)
    cells = res[name0]["cells"]
    clon, clat = healpix_geo.nested.healpix_to_lonlat(cells.astype(np.uint64), a.centre_level,
                                                      ellipsoid="WGS84")
    ce, cn = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        np.asarray(clon, float), np.asarray(clat, float))
    cen = np.stack([ce, cn], -1)
    M = cen.shape[0]
    ref = embed(model, sample_multi(bands, cen, np.broadcast_to(offs, (M,) + offs.shape)
                                  ).reshape(-1, C, a.n, a.n))
    sens = {"rotation_deg": {}, "shift_px": {}, "scale_pct": {}}
    for th in (0.25, 0.5, 1, 2, 3, 4, 6, 10, 20, 45, 90):
        t = np.radians(th)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        e = embed(model, sample_multi(bands, cen, np.broadcast_to(offs @ R.T, (M,) + offs.shape)
                                    ).reshape(-1, C, a.n, a.n))
        sens["rotation_deg"][th] = {kk: float(np.median(cosd(ref[kk], e[kk]))) for kk in ref}
    for sh in (0.25, 0.5, 1, 2, 4):
        e = embed(model, sample_multi(bands, cen,
                                    np.broadcast_to(offs + np.array([sh * s, 0.0]), (M,) + offs.shape)
                                    ).reshape(-1, C, a.n, a.n))
        sens["shift_px"][sh] = {kk: float(np.median(cosd(ref[kk], e[kk]))) for kk in ref}
    for sc in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10):
        e = embed(model, sample_multi(bands, cen,
                                    np.broadcast_to(offs * (1 + sc / 100), (M,) + offs.shape)
                                    ).reshape(-1, C, a.n, a.n))
        sens["scale_pct"][sc] = {kk: float(np.median(cosd(ref[kk], e[kk]))) for kk in ref}

    # ---- 2. zone gradient and 3. pairs
    grad = {}
    for name in res:
        E = res[name]["E"]
        grad[name] = {kk: dict(utm_hpx=float(np.median(cosd(E["utm"][kk], E["hpx"][kk]))),
                               utmc_hpx=float(np.median(cosd(E["utm_c"][kk], E["hpx"][kk]))),
                               utm_utmc=float(np.median(cosd(E["utm"][kk], E["utm_c"][kk]))),
                               utm_hpx_q=np.percentile(cosd(E["utm"][kk], E["hpx"][kk]),
                                                       [25, 75]).tolist(),
                               utmc_hpx_q=np.percentile(cosd(E["utm_c"][kk], E["hpx"][kk]),
                                                        [25, 75]).tolist())
                      for kk in E["utm"]}
    pairs = {}
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    for n1, n2, label in [p_ for p_ in pairs_cfg if p_[0] in res and p_[1] in res]:
        c1, c2 = res[n1]["cells"], res[n2]["cells"]
        common, i1, i2 = np.intersect1d(c1, c2, return_indices=True)
        pr = {"n": int(common.size)}
        for kk in res[n1]["E"]["utm"]:
            for kind in ("utm", "utm_c", "hpx"):
                d = cosd(res[n1]["E"][kind][kk][i1], res[n2]["E"][kind][kk][i2])
                a1 = res[n1]["E"][kind][kk][i1]
                a2 = res[n2]["E"][kind][kk][i2]
                u1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
                u2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
                # retrieval: is the same ground point the nearest neighbour in the other tile?
                top1 = float((np.argmax(u1 @ u2.T, axis=1) == np.arange(len(u1))).mean())
                # classification agreement: one k-means on both views, labels compared
                km = KMeans(a.k, n_init=10, random_state=0).fit(np.r_[u1, u2])
                ari = float(adjusted_rand_score(km.labels_[:len(u1)], km.labels_[len(u1):]))
                pr[f"{kind}/{kk}"] = dict(median=float(np.median(d)),
                                          q=np.percentile(d, [25, 75]).tolist(),
                                          top1=top1, kmeans_ari=ari,
                                          kmeans_agree=float((km.labels_[:len(u1)]
                                                              == km.labels_[len(u1):]).mean()))
        pairs[label] = pr

    # ---- save numbers, distributions and examples
    out = dict(spacing_m=s, n_px=a.n, level=a.level, centre_level=a.centre_level,
               fake=a.fake, lam=a.lam, resampler=a.resampler, sensitivity_site=sens_site,
               sites=site_info,
               sensitivity=sens, gradient=grad, pairs=pairs)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(a.out, "distances.npz"), **{
        f"{name}__{pair}__{kk}": cosd(res[name]["E"][p1][kk], res[name]["E"][p2][kk])
        for name in res for kk in res[name]["E"]["utm"]
        for pair, (p1, p2) in {"utm_hpx": ("utm", "hpx"), "utmc_hpx": ("utm_c", "hpx"),
                               "utm_utmc": ("utm", "utm_c")}.items()})
    np.savez_compressed(os.path.join(a.out, "examples.npz"),
                        **{f"{n}__{k}": v for n, d in examples.items() for k, v in d.items()})
    print(json.dumps(dict(sites=site_info, gradient={n: grad[n]["mean"] for n in grad},
                          pairs={l: {k: v for k, v in p.items() if "mean" in k or k == "n"}
                                 for l, p in pairs.items()}), indent=1))
    print("results in", a.out)


if __name__ == "__main__":
    main()
