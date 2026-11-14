"""
g4e_source.py
=============
Reading Sentinel-2 HEALPix data from the GRID4EARTH bucket.

Layout (docs/bucket-layout.md of GRID4EARTH/project-guidelines).  The `public`
directory of the `grid4earth` OVH bucket is served at
``https://data.grid4earth.eu``::

    converted/sentinel-2-l2a/<PRODUCT_ID>.zarr     <- the HEALPix products
    auxiliary/...   eopf-mirror/...   reprocessed/...   legacy/...

Note the ``converted/`` prefix: ``legacy/`` holds the untouched originals and,
as of September 2026, serves nothing for Sentinel-2 (every key 404s).  The
HEALPix conversions are under ``converted/``.

Inside a product the HEALPix level is a **group**, not a dimension::

    <PRODUCT_ID>.zarr/measurements/reflectance/<level>

so one store holds one acquisition at several levels, and a time series is a
list of products rather than a `time` dimension.  That is the shape assumed by
:class:`ProductSeries` below; :class:`TimeSeriesStore` keeps the older form (a
single store with a `time` dimension) working.

Both expose the same three things to the rest of the pipeline::

    src.level          # HEALPix level of the data
    src.cell_id        # [N] NESTED cell ids
    src.dates          # one label per time step
    src.rgb(i)         # [N, 3] reflectances in [0, 1], NaN where missing

Opening follows GRID4EARTH/egu26-demos (`poster/healpix-geo/xdggs.ipynb`):
obstore + zarr's ObjectStore when they are installed, plain fsspec otherwise.
xdggs is deliberately not used -- the cell ids are read from the store.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np
import xarray as xr

# The public directory of the grid4earth bucket
G4E_BASE = "https://data.grid4earth.eu"

# Sentinel-2 L2A HEALPix conversions
G4E_L2A = f"{G4E_BASE}/converted/sentinel-2-l2a"
G4E_PRODUCTS = (
    "S2B_MSIL2A_20250522T105619_N0511_R094_20250522T121018",
    "S2C_MSIL2A_20250527T105641_N0511_R094_20250527T165313",
)

RGB = ("b04", "b03", "b02")          # Sentinel-2 red, green, blue at 10 m

# How raw band values are turned into something a ViT can eat.
#
#   "reflectance"  the L2A convention: divide by REFLECTANCE_SCALE.  Fixed, so
#                  the same physical value maps to the same number in every
#                  product, every tile and every date -- which is what the
#                  classification needs, since it compares patches across the
#                  whole dataset.
#   "percentile"   linear rescale by the 2-98 percentile spread, computed ONCE
#                  on the first product read and then reused unchanged for every
#                  other product.  Use it when the store's units are unknown.
#   "none"         raw values.
#
# There is deliberately no per-tile and no per-product normalisation: any
# statistic recomputed per tile makes two identical patches in different tiles
# land on different numbers, and the clustering that follows compares them.
SCALING = "reflectance"
REFLECTANCE_SCALE = 10000.0

# Levels published per product (the `multiscales` convention lists them in
# `measurements/reflectance/zarr.json`); 17 is the coarsest and the cheapest.
G4E_LEVELS = (17, 19, 20)


# ---------------------------------------------------------------------------
# opening
# ---------------------------------------------------------------------------

def open_zarr_group(url: str, group: Optional[str] = None) -> xr.Dataset:
    """
    Open a zarr store, with or without a group, over HTTP or from disk.

    Tries obstore first -- what the GRID4EARTH demos use, and much faster on a
    plain HTTP store -- then falls back to fsspec, then to a v2 store.
    """
    kw = {"engine": "zarr"}
    if group:
        kw["group"] = group

    if "://" in url and not url.startswith("file://"):
        try:
            from obstore.store import HTTPStore
            from zarr.storage import ObjectStore

            return xr.open_dataset(ObjectStore(HTTPStore(url)), **kw)
        except Exception:
            pass          # obstore not installed, or not an HTTP store

    try:
        return xr.open_dataset(url, **kw)
    except Exception:
        return xr.open_dataset(url, zarr_format=2, **kw)   # older stores are v2


def cell_dim(ds: xr.Dataset) -> str:
    """Name of the cell dimension: 'cell_ids' in a raw store, 'cells' once decoded."""
    for name in ("cell_ids", "cells"):
        if name in ds.coords:
            return ds[name].dims[0]
    for d in ds.sizes:
        if d not in ("time", "bands", "band"):
            return d
    raise KeyError("no cell dimension found")


def cell_ids_of(ds: xr.Dataset) -> np.ndarray:
    for name in ("cell_ids", "cells", "cell_id"):
        if name in ds.coords or name in ds.variables:
            return np.asarray(ds[name].values).astype(np.int64)
    raise KeyError(f"no cell id coordinate in {list(ds.coords)}")


def ellipsoid_of(ds: xr.Dataset, default: str = "sphere") -> str:
    """
    Reference ellipsoid declared by the store, from the ``dggs`` attributes.

    EOPF HEALPix products say ``wgs84``; assuming a sphere instead displaces
    every cell centre by up to ~0.2 degrees of latitude, so this is worth
    reading rather than guessing.
    """
    for attrs in (ds.attrs, *(ds[n].attrs for n in ("cell_ids", "cells", "cell_id")
                              if n in ds.coords or n in ds.variables)):
        dggs = attrs.get("dggs")
        if isinstance(dggs, dict):
            ell = dggs.get("ellipsoid")
            if isinstance(ell, dict) and ell.get("name"):
                return str(ell["name"])
            if isinstance(ell, str):
                return ell
    return default


def level_of(ds: xr.Dataset, fallback: Optional[int] = None) -> int:
    """HEALPix level, from the cell-id attributes or from the dataset attributes."""
    sources = []
    for attrs in (ds.attrs, *(ds[n].attrs for n in ("cell_ids", "cells", "cell_id")
                              if n in ds.coords or n in ds.variables)):
        sources.append(dict(attrs))
        dggs = attrs.get("dggs")                # the EOPF `dggs` convention nests it
        if isinstance(dggs, dict):
            sources.append(dggs)

    for a in sources:
        # `refinement_level` is what the zarr `dggs` convention calls it
        for key in ("refinement_level", "level", "resolution", "depth"):
            if key in a:
                return int(a[key])
        if "nside" in a:
            return int(np.log2(int(a["nside"])))
    if fallback is None:
        raise KeyError("no HEALPix level in the store; pass level=")
    print(f"[warn] no level attribute found, assuming level {fallback}")
    return int(fallback)


# Variables that live on the cell dimension but are not measurements.
_NOT_A_BAND = {"cell_ids", "cells", "cell_id", "spatial_ref", "crs"}


def band_names(ds: xr.Dataset) -> list:
    """
    Data variables defined on the cell dimension alone -- the bands.

    The cell-id array is often stored as a plain variable rather than a
    coordinate, so it has to be excluded explicitly or it is reported as a band.
    """
    d = cell_dim(ds)
    return [str(v) for v in ds.data_vars
            if ds[v].dims == (d,) and str(v) not in _NOT_A_BAND]


def band_stats(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 98.0):
    """Per-band (offset, scale) from the finite values -- the "percentile" mode."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    off, sca = [], []
    for c in range(x.shape[1]):
        good = x[:, c][np.isfinite(x[:, c])]
        if good.size == 0:
            off.append(0.0); sca.append(1.0); continue
        lo, hi = np.percentile(good, [lo_p, hi_p])
        off.append(float(lo)); sca.append(max(float(hi - lo), 1e-6))
    return np.array(off, np.float32), np.array(sca, np.float32)


def normalise(x: np.ndarray, how: str = SCALING, *, stats=None) -> np.ndarray:
    """
    Turn raw band values ``[N, C]`` into model input, NaN preserved.

    The transform is **affine and fixed**: the same raw value always gives the
    same number, whatever tile or product it came from.  Nothing is squashed and
    nothing is clipped -- a bright roof stays brighter than a bright field
    instead of being flattened onto the same 1.0, and DINOv3's own mean/std
    normalisation handles the rest.

    ``stats`` is the ``(offset, scale)`` pair for the "percentile" mode; pass the
    one computed on the first product so every product shares it.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if how == "none":
        return x
    if how == "reflectance":
        return x / REFLECTANCE_SCALE
    if how == "percentile":
        off, sca = band_stats(x) if stats is None else stats
        return (x - off[None, :]) / sca[None, :]
    raise ValueError("scaling must be 'reflectance', 'percentile' or 'none'")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

class _Source:
    """Common part: retries, on-disk cache, scaling to reflectance."""

    dates: list
    level: int
    cell_id: np.ndarray

    def _read(self, i: int) -> np.ndarray:              # pragma: no cover
        raise NotImplementedError

    def cache_name(self, i: int) -> str:
        """
        File name for the cached reflectances of step ``i``.

        The HEALPix level and the band list are part of the name: without them a
        cache written at one level is silently reused at another, and the arrays
        no longer match ``cell_id``.
        """
        bands = "-".join(str(b) for b in getattr(self, "bands", ()))
        how = getattr(self, "scaling", SCALING)
        return f"{self.tag}_l{int(self.level)}_{bands}_{how}_{i:04d}.npy"

    def rgb(self, i: int, *, cache: Optional[str] = None, retries: int = 5) -> np.ndarray:
        """[N, 3] reflectances in [0, 1] for time step ``i``; NaN where missing."""
        fname = os.path.join(cache, self.cache_name(i)) if cache else None
        if fname is not None and os.path.exists(fname):
            x = np.load(fname).astype(np.float32)
            if x.shape[0] == self.cell_id.size:
                return x
            print(f"  [cache] {fname} a {x.shape[0]} lignes pour {self.cell_id.size} "
                  "cellules : ignore et relu depuis le store")

        delay = 2.0
        for attempt in range(1, retries + 1):
            try:
                x = self._read(i)
                break
            except Exception as exc:                    # noqa: BLE001
                if attempt == retries:
                    raise
                print(f"  [retry {attempt}/{retries - 1}] {i}: "
                      f"{type(exc).__name__}: {exc}; waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay *= 2

        x = np.asarray(x, dtype=np.float32)
        how = getattr(self, "scaling", SCALING)
        # Les unites du store ne sont declarees nulle part : on les montre.  Une
        # echelle fausse ne se voit pas sur un affichage (qui etire en
        # percentiles) mais aplatit l'image vue par le reseau, et les tokens ne
        # portent alors plus que leur position.
        _raw = np.percentile(x[np.isfinite(x)], [2, 50, 98]) if np.isfinite(x).any() else [np.nan]*3
        print(f"  valeurs brutes   p2 {_raw[0]:.4g}  mediane {_raw[1]:.4g}  p98 {_raw[2]:.4g}")
        if how == "percentile" and getattr(self, "_stats", None) is None:
            # computed once, on the first product read, then reused for all the
            # others: the embeddings of two dates have to live on the same scale
            self._stats = band_stats(x)
            print(f"  [percentile] offset {np.round(self._stats[0], 1)} "
                  f"echelle {np.round(self._stats[1], 1)} (fige pour toute la serie)")
        x = normalise(x, how, stats=getattr(self, "_stats", None))
        _n = np.percentile(x[np.isfinite(x)], [2, 50, 98]) if np.isfinite(x).any() else [np.nan]*3
        print(f"  apres '{how}'     p2 {_n[0]:.4g}  mediane {_n[1]:.4g}  p98 {_n[2]:.4g}")
        if np.isfinite(_n).all() and (_n[2] - _n[0]) < 0.02:
            print("  /!\\ dynamique quasi nulle : l'image vue par DINOv3 est plate, "
                  "ses tokens ne porteront que leur position.\n"
                  "      SCALING est probablement inadapte aux unites du store "
                  "(essaie 'percentile', ou 'none' si les donnees sont deja en reflectance).")
        if fname is not None:
            os.makedirs(cache, exist_ok=True)
            np.save(fname, x.astype(np.float16))
        return x


class ProductSeries(_Source):
    """
    GRID4EARTH layout: one zarr per acquisition, HEALPix level as a group.

    Parameters
    ----------
    products : sequence of str
        Product ids, without the ``.zarr`` suffix.
    level : int
        HEALPix level, i.e. the group under ``measurements/reflectance``.
    base : str
        Collection directory; defaults to the public Sentinel-2 L2A one.
    bands : sequence of str
        Band variables to read, in (R, G, B) order.
    """

    tag = "g4e"

    def __init__(self, products: Sequence[str] = G4E_PRODUCTS, level: int = 20,
                 base: str = G4E_L2A, bands: Sequence[str] = RGB,
                 group_fmt: str = "measurements/reflectance/{level}",
                 scaling: str = SCALING):
        self.products = [str(p).removesuffix(".zarr") for p in products]
        self.base = base.rstrip("/")
        self.bands = list(bands)
        self.scaling = scaling
        self._stats = None
        self.group = group_fmt.format(level=level)
        self.dates = [self._date(p) for p in self.products]

        ds = self._open(0)
        self.level = level_of(ds, fallback=level)
        if self.level != level:
            print(f"[warn] group says level {level}, the store says {self.level}")
        self.cell_id = cell_ids_of(ds)
        self.ellipsoid = ellipsoid_of(ds)
        available = band_names(ds)
        missing = [b for b in self.bands if b not in available]
        if missing:
            raise KeyError(f"bands {missing} are not in the store; it has {available}")
        print(f"{len(self.products)} products, level {self.level}, "
              f"{self.cell_id.size} cells, bands {available}, "
              f"ellipsoid {self.ellipsoid}")

    @staticmethod
    def _date(product: str) -> str:
        """Sensing date from the product id: S2B_MSIL2A_<YYYYMMDD>T<hhmmss>_..."""
        for part in product.split("_"):
            if len(part) == 15 and part[8] == "T" and part[:8].isdigit():
                return f"{part[:4]}-{part[4:6]}-{part[6:8]}"
        return product

    def url(self, i: int) -> str:
        return f"{self.base}/{self.products[i]}.zarr"

    def _open(self, i: int) -> xr.Dataset:
        return open_zarr_group(self.url(i), self.group)

    def _read(self, i: int) -> np.ndarray:
        ds = self._open(i)
        ids = cell_ids_of(ds)
        if not np.array_equal(ids, self.cell_id):
            raise ValueError(
                f"product {self.products[i]} covers different cells than the first one; "
                "embeddings can still be computed per product, but the temporal "
                "consistency score assumes a common grid")
        return np.stack([ds[b].values for b in self.bands], axis=-1)


class TimeSeriesStore(_Source):
    """A single store with a ``time`` dimension and a ``bands`` coordinate."""

    tag = "rgb"

    def __init__(self, url: str, bands: Sequence[str] = RGB,
                 var: str = "Sentinel2", level: Optional[int] = None):
        self.ds = open_zarr_group(url)
        self.var, self.bands = var, list(bands)
        self.scaling = SCALING
        self._stats = None
        self.level = level_of(self.ds, fallback=level)
        self.cell_id = cell_ids_of(self.ds)
        self.ellipsoid = ellipsoid_of(self.ds)
        self.dates = [str(d)[:10] for d in self.ds["time"].values]

    def _read(self, i: int) -> np.ndarray:
        return (self.ds[self.var].isel(time=i).sel(bands=self.bands)
                .transpose(cell_dim(self.ds), "bands").values)


def make_source(spec: str, *, level: int = 20, products: Sequence[str] = G4E_PRODUCTS,
                bands: Sequence[str] = RGB, base: str = G4E_L2A) -> _Source:
    """``spec`` is "g4e" for the GRID4EARTH products, or the URL of a single store."""
    if spec == "g4e":
        return ProductSeries(products, level=level, base=base, bands=bands)
    return TimeSeriesStore(spec, bands=bands)
