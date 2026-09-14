"""
g4e_source.py
=============
Reading Sentinel-2 HEALPix data from the GRID4EARTH bucket.

Layout (docs/bucket-layout.md of GRID4EARTH/project-guidelines).  The `public`
directory of the `grid4earth` OVH bucket is served at
``https://data.grid4earth.eu``::

    legacy/sentinel-2-l2a/<PRODUCT_ID>.zarr
    legacy/sentinel-2-l1c/<PRODUCT_ID>.zarr
    eopf-mirror/...            reprocessed/...            auxiliary/...

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

# Sentinel-2 L2A products listed in the bucket layout
G4E_L2A = f"{G4E_BASE}/legacy/sentinel-2-l2a"
G4E_PRODUCTS = (
    "S2B_MSIL2A_20250522T105619_N0511_R094_20250522T121018",
    "S2C_MSIL2A_20250527T105641_N0511_R094_20250527T165313",
)

RGB = ("b04", "b03", "b02")          # Sentinel-2 red, green, blue at 10 m
REFLECTANCE_SCALE = 10000.0          # L2A digital numbers -> reflectance


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


def level_of(ds: xr.Dataset, fallback: Optional[int] = None) -> int:
    """HEALPix level, from the cell-id attributes or from the dataset attributes."""
    sources = []
    for name in ("cell_ids", "cells", "cell_id"):
        if name in ds.coords or name in ds.variables:
            sources.append(dict(ds[name].attrs))
    sources.append(dict(ds.attrs))
    dggs = ds.attrs.get("dggs")
    if isinstance(dggs, dict):
        sources.append(dggs)

    for a in sources:
        for key in ("level", "resolution", "depth"):
            if key in a:
                return int(a[key])
        if "nside" in a:
            return int(np.log2(int(a["nside"])))
    if fallback is None:
        raise KeyError("no HEALPix level in the store; pass level=")
    print(f"[warn] no level attribute found, assuming level {fallback}")
    return int(fallback)


def band_names(ds: xr.Dataset) -> list:
    """Data variables defined on the cell dimension alone -- the bands."""
    d = cell_dim(ds)
    return [str(v) for v in ds.data_vars if ds[v].dims == (d,)]


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

    def rgb(self, i: int, *, cache: Optional[str] = None, retries: int = 5) -> np.ndarray:
        """[N, 3] reflectances in [0, 1] for time step ``i``; NaN where missing."""
        fname = os.path.join(cache, f"{self.tag}_{i:04d}.npy") if cache else None
        if fname is not None and os.path.exists(fname):
            return np.load(fname).astype(np.float32)

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
        if np.nanmax(x) > 1.5:                          # digital numbers, not reflectance
            x = x / REFLECTANCE_SCALE
        finite = np.isfinite(x)
        x[finite] = np.clip(x[finite], 0.0, 1.0)
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

    def __init__(self, products: Sequence[str] = G4E_PRODUCTS, level: int = 17,
                 base: str = G4E_L2A, bands: Sequence[str] = RGB,
                 group_fmt: str = "measurements/reflectance/{level}"):
        self.products = [str(p).removesuffix(".zarr") for p in products]
        self.base = base.rstrip("/")
        self.bands = list(bands)
        self.group = group_fmt.format(level=level)
        self.dates = [self._date(p) for p in self.products]

        ds = self._open(0)
        self.level = level_of(ds, fallback=level)
        if self.level != level:
            print(f"[warn] group says level {level}, the store says {self.level}")
        self.cell_id = cell_ids_of(ds)
        available = band_names(ds)
        missing = [b for b in self.bands if b not in available]
        if missing:
            raise KeyError(f"bands {missing} are not in the store; it has {available}")
        print(f"{len(self.products)} products, level {self.level}, "
              f"{self.cell_id.size} cells, bands {available}")

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
        self.level = level_of(self.ds, fallback=level)
        self.cell_id = cell_ids_of(self.ds)
        self.dates = [str(d)[:10] for d in self.ds["time"].values]

    def _read(self, i: int) -> np.ndarray:
        return (self.ds[self.var].isel(time=i).sel(bands=self.bands)
                .transpose(cell_dim(self.ds), "bands").values)


def make_source(spec: str, *, level: int = 17, products: Sequence[str] = G4E_PRODUCTS,
                bands: Sequence[str] = RGB, base: str = G4E_L2A) -> _Source:
    """``spec`` is "g4e" for the GRID4EARTH products, or the URL of a single store."""
    if spec == "g4e":
        return ProductSeries(products, level=level, base=base, bands=bands)
    return TimeSeriesStore(spec, bands=bands)
