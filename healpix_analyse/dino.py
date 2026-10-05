"""
dino.py
=======
DINOv3 (SAT-493M) embeddings of NESTED HEALPix data for the
``healpix_analyse`` package.

The idea
--------
A ViT is not a "grid" architecture: after the patch embedding it only sees a
set of tokens.  In the NESTED HEALPix scheme every cell at ``parent_level``
contains exactly ``4**(level - parent_level)`` cells of ``level``, stored
contiguously and ordered by a Morton (Z-order) code of their local ``(x, y)``
face coordinates.  A NESTED block can therefore be reshaped *exactly* into a
square image of side ``2**(level - parent_level)`` pixels, fed to an
unmodified DINOv3 network, and every 16x16 patch token maps back to one
HEALPix cell at ``level - 4``.

This module provides

- :func:`nested_to_tiles`   NESTED block  -> square tiles ``[M, C, S, S]``
- :func:`tiles_to_nested`   the exact inverse (for debugging / round trips)
- :func:`load_dinov3_sat`   load a DINOv3 SAT-493M backbone (hub or HF)
- :func:`GetDINOV3SAT`      the user-facing function

Orientation
-----------
Inside a HEALPix base face the local axes ``(x, y)`` are rotated by 45
degrees with respect to the local East/North directions: the "north" corner
of a face is at ``(x, y) = (max, max)``.  Tiles are built with
``row = S - 1 - y`` and ``col = x`` so that North points to the top-right
corner of the image.  The orientation is consistent for all tiles of the
same base face and rotated for the polar faces, which is a mild nuisance for
a satellite backbone (no gravity direction) and should be handled by
rotation augmentations when fine-tuning.

Dependencies: numpy, torch, ``healpix-geo`` (the GRID4EARTH HEALPix library --
``healpy`` is deliberately not used anywhere in this package), and either the
``dinov3`` torch-hub repo (``facebookresearch/dinov3``) or ``transformers``
(HF weights).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, torch.Tensor]

# DINOv3 SAT-493M normalisation constants (from the DINOv3 model card;
# they differ from the ImageNet/LVD-1689M values used by the web models).
SAT493M_MEAN: Tuple[float, float, float] = (0.430, 0.411, 0.296)
SAT493M_STD: Tuple[float, float, float] = (0.213, 0.156, 0.143)

# DINOv3 patch size (all released models).
DINOV3_PATCH: int = 16
DINOV3_PATCH_LEVELS: int = 4          # 16 px  == 4 HEALPix levels

_HF_SAT_IDS = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-sat493m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-sat493m",
}


# ---------------------------------------------------------------------------
# HEALPix primitives -- healpix-geo (GRID4EARTH), never healpy
# ---------------------------------------------------------------------------
#
# ``healpix_geo.nested`` is the reference implementation used across the
# GRID4EARTH suite.  It takes a *depth* (what we call ``level``) rather than an
# ``nside``, wants ``uint64`` cell ids, and returns degrees.  The thin wrappers
# below are the only place where that convention is spelled out, so the rest of
# the module stays readable.

from healpix_geo import nested as _hpx          # noqa: E402

# Reference ellipsoid used for every cell <-> lon/lat conversion.
#
# It is NOT cosmetic: a HEALPix cell id means a different patch of ground on the
# sphere and on WGS84, and the difference reaches ~0.2 degrees of latitude in
# mid-latitudes -- tens of kilometres, far more than a tile.  EOPF HEALPix
# products declare their ellipsoid in the `dggs` attributes of the level group
# (GRID4EARTH Sentinel-2 says "wgs84"); read it from the store and set it here
# before computing anything, or pass it to :func:`set_ellipsoid`.
HEALPIX_ELLIPSOID: str = "sphere"


def set_ellipsoid(name: str) -> str:
    """
    Choose the reference ellipsoid ("sphere", "wgs84", ...) and return the old one.

    Call it once, with whatever the store's ``dggs.ellipsoid.name`` says, before
    any embedding is computed.
    """
    from healpix_geo import ellipsoid as _ell

    # healpix-geo's names are case-sensitive ("WGS84"), while stores spell the
    # same thing in lower case ("wgs84"); accept either.
    for candidate in (str(name), str(name).upper(), str(name).lower()):
        try:
            _ell.resolve(candidate)
        except Exception:                      # noqa: BLE001
            continue
        global HEALPIX_ELLIPSOID
        previous, HEALPIX_ELLIPSOID = HEALPIX_ELLIPSOID, candidate
        return previous
    raise ValueError(f"unknown ellipsoid {name!r}; try 'sphere' or 'WGS84'")


def _as_ids(ids: np.ndarray) -> np.ndarray:
    """Cell ids in the ``uint64`` form healpix-geo expects."""
    return np.ascontiguousarray(np.asarray(ids), dtype=np.uint64)


def _unmask(a, fill):
    """
    Plain ndarray out of whatever healpix-geo returned.

    Several healpix-geo functions return a masked array (``marray``) so that
    absent neighbours or stencil corners can be flagged; ``fill`` is what those
    entries become here -- ``-1`` for an id, ``0.0`` for a weight, which is what
    the rest of this module already treats as "nothing there".
    """
    mask = getattr(a, "mask", None)
    data = np.asarray(getattr(a, "data", a))
    if data.dtype.kind == "u":
        # recent healpix-geo returns uint64 ids, which cannot hold the -1 fill
        data = data.astype(np.int64)
    if mask is None:
        return data
    return np.where(np.asarray(mask), fill, data)


def _pix2ang(level: int, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cell centres as (lon, lat) in degrees."""
    lon, lat = _hpx.healpix_to_lonlat(_as_ids(ids), int(level),
                                      ellipsoid=HEALPIX_ELLIPSOID)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _ang2pix(level: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """NESTED cell id containing each (lon, lat), in degrees."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    shape = np.broadcast(lon, lat).shape
    ids = _hpx.lonlat_to_healpix(
        np.ascontiguousarray(np.broadcast_to(lon, shape).reshape(-1)),
        np.ascontiguousarray(np.broadcast_to(lat, shape).reshape(-1)),
        int(level),
        ellipsoid=HEALPIX_ELLIPSOID,
    )
    return np.asarray(ids, dtype=np.int64).reshape(shape)


def _pix2vec(level: int, ids: np.ndarray) -> np.ndarray:
    """Cell centres as **unit** 3-D vectors, shape ``[3, ...]``."""
    x, y, z = _hpx.healpix_to_cartesian(_as_ids(ids), int(level),
                                        ellipsoid=HEALPIX_ELLIPSOID)
    v = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    # healpix-geo returns metres on the chosen ellipsoid, whose radius varies
    # with latitude; normalise rather than dividing by a constant.
    return v / np.linalg.norm(v, axis=0)


def _boundaries_lonlat(level: int, ids: np.ndarray, step: int = 4
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Cell outlines as (lon, lat) in degrees, shape ``[M, 4 * step]``."""
    lon, lat = _hpx.vertices(_as_ids(ids), int(level), step=int(step),
                             ellipsoid=HEALPIX_ELLIPSOID)
    return (_unmask(lon, np.nan).astype(np.float64),
            _unmask(lat, np.nan).astype(np.float64))


def _interp_weights(level: int, lon: np.ndarray, lat: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation stencil: ids and weights, both shape ``[4, P]``.

    The transpose is the only difference from healpix-geo's ``[P, 4]``; it keeps
    the "one row per stencil corner" layout the sampling code is written around.
    """
    ids, wgt = _hpx.bilinear_interpolation(
        np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64), int(level),
        ellipsoid=HEALPIX_ELLIPSOID,
    )
    return (_unmask(ids, -1).astype(np.int64).T,
            _unmask(wgt, 0.0).astype(np.float64).T)


def _neighbours(level: int, ids: np.ndarray) -> np.ndarray:
    """The (up to) eight neighbours of each cell, shape ``[8, N]``, -1 if absent."""
    nb = _hpx.neighbours(_as_ids(ids), int(level), connectivity="all")
    return _unmask(nb, -1).astype(np.int64).T


# Public names for the HEALPix primitives, so that notebooks and scripts built
# on this module never have to reach for healpy either.

def cell_centres_lonlat(level: int, cell_id: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Centres of NESTED cells at ``level``, as (lon, lat) in degrees."""
    return _pix2ang(level, cell_id)


def cells_at_lonlat(level: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """NESTED cell ids at ``level`` containing the given (lon, lat), in degrees."""
    return _ang2pix(level, lon, lat)


def cell_vectors(level: int, cell_id: np.ndarray) -> np.ndarray:
    """Unit vectors of the cell centres, shape ``[3, N]``."""
    return _pix2vec(level, cell_id)


def cell_neighbours(level: int, cell_id: np.ndarray) -> np.ndarray:
    """The eight NESTED neighbours of each cell, shape ``[8, N]``; -1 where absent."""
    return _neighbours(level, cell_id)


# ---------------------------------------------------------------------------
# Morton (Z-order) helpers -- pure integer arithmetic
# ---------------------------------------------------------------------------

_M1 = np.int64(0x5555555555555555)
_M2 = np.int64(0x3333333333333333)
_M4 = np.int64(0x0F0F0F0F0F0F0F0F)
_M8 = np.int64(0x00FF00FF00FF00FF)
_M16 = np.int64(0x0000FFFF0000FFFF)
_M32 = np.int64(0x00000000FFFFFFFF)


def _compact_bits(v: np.ndarray) -> np.ndarray:
    """Keep the even bits of ``v`` and pack them (inverse of :func:`_spread_bits`)."""
    v = v & _M1
    v = (v | (v >> 1)) & _M2
    v = (v | (v >> 2)) & _M4
    v = (v | (v >> 4)) & _M8
    v = (v | (v >> 8)) & _M16
    v = (v | (v >> 16)) & _M32
    return v


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Spread the 32 low bits of ``v`` onto the even bit positions."""
    v = v & _M32
    v = (v | (v << 16)) & _M16
    v = (v | (v << 8)) & _M8
    v = (v | (v << 4)) & _M4
    v = (v | (v << 2)) & _M2
    v = (v | (v << 1)) & _M1
    return v


def nested_to_xy(rel_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a *relative* NESTED index into local ``(x, y)`` coordinates.

    ``rel_id`` is the index of a cell inside its ancestor block
    (``cell_id - parent_id * 4**k``); the result satisfies
    ``0 <= x, y < 2**k``.  This is the HEALPix ``nest2xyf`` convention:
    ``x`` occupies the even bits, ``y`` the odd bits.
    """
    rel_id = np.asarray(rel_id, dtype=np.int64)
    return _compact_bits(rel_id), _compact_bits(rel_id >> 1)


def xy_to_nested(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`nested_to_xy`."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    return _spread_bits(x) | (_spread_bits(y) << 1)


# ---------------------------------------------------------------------------
# NESTED block <-> square tile
# ---------------------------------------------------------------------------

def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _deduplicate(
    d: np.ndarray,
    ids: np.ndarray,
    how: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse repeated cell ids.

    Data projected from another grid (UTM, swath, ...) onto HEALPix regularly
    contains a few cells hit by more than one source pixel.  Rather than
    rejecting such an input, the duplicated rows are averaged (NaN-aware), or
    the first occurrence is kept.
    """
    uniq_ids, inv = np.unique(ids, return_inverse=True)
    inv = np.asarray(inv).ravel()      # numpy >= 2 may return the input shape
    n_dup = ids.shape[0] - uniq_ids.shape[0]
    if n_dup == 0:
        return d, ids
    if how == "error":
        raise ValueError(
            f"cell_id contains {n_dup} duplicate ids "
            "(pass duplicates='mean' or 'first' to aggregate them)"
        )
    warnings.warn(
        f"cell_id contains {n_dup} duplicate ids out of {ids.shape[0]}; "
        f"aggregating with duplicates={how!r}",
        RuntimeWarning,
        stacklevel=3,
    )
    if how == "first":
        _, first = np.unique(ids, return_index=True)
        return d[first], uniq_ids
    if how != "mean":
        raise ValueError("duplicates must be 'mean', 'first' or 'error'")

    finite = np.isfinite(d)
    acc = np.zeros((uniq_ids.shape[0], d.shape[1]), dtype=np.float64)
    cnt = np.zeros((uniq_ids.shape[0], d.shape[1]), dtype=np.int64)
    np.add.at(acc, inv, np.where(finite, d, 0.0))
    np.add.at(cnt, inv, finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / cnt, np.nan).astype(np.float32)
    return out, uniq_ids


def nested_to_tiles(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    fill: str = "mean",
    duplicates: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder a NESTED HEALPix dataset into square tiles, one per
    ``parent_level`` cell present in ``cell_id``.

    Parameters
    ----------
    data : array [N, C] or [N]
        Pixel values at HEALPix ``level``.  NaN marks a missing pixel.
    cell_id : int array [N]
        NESTED cell ids at ``level``.  Duplicates are not allowed.
    level : int
        Resolution of ``data`` (``nside = 2**level``).
    parent_level : int
        Resolution of the tiles.  ``k = level - parent_level >= 1``; the
        tiles have side ``S = 2**k``.
    fill : {"mean", "zero", "nan"}
        Value given to the pixels of a tile that are absent from ``cell_id``
        (or NaN): the per-tile, per-band mean of the available pixels,
        zero, or NaN.
    duplicates : {"mean", "first", "error"}
        What to do when the same cell id appears several times, which happens
        when data projected from another grid puts two source pixels in the
        same HEALPix cell: average them (NaN-aware), keep the first one, or
        raise.

    Returns
    -------
    tiles : float32 array [M, C, S, S]
    parent_ids : int64 array [M]
        NESTED ids of the tiles at ``parent_level`` (sorted).
    valid : bool array [M, S, S]
        Availability mask of every pixel.
    coverage : float array [M]
        Fraction of available pixels per tile.
    """
    k = int(level) - int(parent_level)
    if k < 1:
        raise ValueError(
            f"level ({level}) must be larger than parent_level ({parent_level})"
        )
    if k > 15:
        raise ValueError("level - parent_level > 15 is not supported")
    S = 1 << k

    d = _as_numpy(data)
    squeeze = d.ndim == 1
    if squeeze:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError(f"data must have shape [N, C] or [N], got {d.shape}")
    d = d.astype(np.float32, copy=False)

    ids = _as_numpy(cell_id).astype(np.int64)
    if ids.shape != (d.shape[0],):
        raise ValueError("cell_id must have shape [N] matching data")

    d, ids = _deduplicate(d, ids, duplicates)

    parent = ids >> (2 * k)
    parent_ids, tile_idx = np.unique(parent, return_inverse=True)
    tile_idx = np.asarray(tile_idx).ravel()      # numpy >= 2 may return (N, 1)
    rel = ids - (parent_ids[tile_idx] << (2 * k))
    x, y = nested_to_xy(rel)
    row = (S - 1) - y
    col = x

    M, C = parent_ids.shape[0], d.shape[1]
    tiles = np.full((M, S, S, C), np.nan, dtype=np.float32)
    valid = np.zeros((M, S, S), dtype=bool)

    tiles[tile_idx, row, col] = d
    valid[tile_idx, row, col] = np.isfinite(d).all(axis=1)

    coverage = valid.reshape(M, -1).mean(axis=1)

    if fill == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(tiles.reshape(M, S * S, C), axis=1)      # [M, C]
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        bad = ~np.isfinite(tiles)                                   # [M,S,S,C]
        tiles = np.where(bad, np.broadcast_to(m[:, None, None, :], tiles.shape), tiles)
    elif fill == "zero":
        tiles = np.nan_to_num(tiles, nan=0.0)
    elif fill != "nan":
        raise ValueError("fill must be 'mean', 'zero' or 'nan'")

    tiles = np.ascontiguousarray(np.transpose(tiles, (0, 3, 1, 2)))   # [M,C,S,S]
    return tiles, parent_ids, valid, coverage


def tiles_to_nested(
    tiles: np.ndarray,
    parent_ids: np.ndarray,
    parent_level: int,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of :func:`nested_to_tiles` (full tiles, no mask).

    Parameters
    ----------
    tiles : array [M, C, S, S]
    parent_ids : int array [M]
    parent_level, level : int

    Returns
    -------
    data : array [M * S * S, C]
    cell_id : int64 array [M * S * S]
        NESTED ids at ``level``, in tile-major order.
    """
    k = int(level) - int(parent_level)
    M, C, S, S2 = tiles.shape
    if S != S2 or S != (1 << k):
        raise ValueError("tile side does not match level - parent_level")
    cell_id = tile_grid_cell_ids(np.asarray(parent_ids), parent_level, level)
    data = np.transpose(tiles, (0, 2, 3, 1)).reshape(M * S * S, C)
    return data, cell_id.reshape(-1)


def tile_grid_cell_ids(
    parent_ids: np.ndarray,
    parent_level: int,
    sub_level: int,
) -> np.ndarray:
    """
    NESTED ids (at ``sub_level``) of the ``[S, S]`` image grid of every tile.

    Returns an int64 array ``[M, S, S]`` indexed as ``[tile, row, col]`` with
    the same orientation convention as :func:`nested_to_tiles`.
    """
    k = int(sub_level) - int(parent_level)
    if k < 0:
        raise ValueError("sub_level must be >= parent_level")
    S = 1 << k
    row, col = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    rel = xy_to_nested(col, (S - 1) - row)                          # [S, S]
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    return (parent_ids[:, None, None] << (2 * k)) + rel[None]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local tangent plane
#
# HEALPix cells have equal area but not equal shape: inside a base face the two
# NESTED axes are neither orthogonal nor of equal length, and the distortion
# grows with latitude.  At 52 deg N and level 19 the two steps measure 10.6 m
# and 16.7 m with a 60.5 deg angle between them -- a square in NESTED index
# space is a parallelogram on the ground, stretched by a factor 2 and sheared by
# 30 deg.  Feeding that to a network trained on ordinary map-projected imagery
# costs more than the resampling needed to avoid it, hence the tangent-plane
# mode: every tile is resampled onto a north-up gnomonic grid of constant
# ground sampling, exactly as ``fft_local.LocalFFT`` does for the local FFT.
# ---------------------------------------------------------------------------

EARTH_RADIUS_M: float = 6371000.0


def healpix_gsd_m(level: int, radius: float = EARTH_RADIUS_M) -> float:
    """Square root of the HEALPix cell area at ``level``, in metres."""
    return float(radius * np.sqrt(np.pi / 3.0) / (2 ** int(level)))


def _lonlat_to_vec(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _vec_to_lonlat(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0, np.degrees(
        np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    )


def tangent_grid_lonlat(
    centre_lon: np.ndarray,
    centre_lat: np.ndarray,
    size: int,
    gsd_rad: float,
    *,
    shift: Tuple[float, float] = (0.0, 0.0),
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Longitude / latitude of north-up gnomonic grids centred on given points.

    Parameters
    ----------
    centre_lon, centre_lat : array [M]
        Tangent points, in degrees.
    size : int or (int, int)
        Side of the grid in pixels; a pair gives (rows, columns).
    gsd_rad : float
        Pixel spacing in tangent units (ground sampling / Earth radius).
    shift : (float, float)
        Extra offset of the grid origin, in pixels (row, column), with the
        image conventions: a positive row shift moves the window **south**
        (down the image), a positive column shift moves it **east**.  Row
        ``r`` of the shifted grid is then row ``r + shift[0]`` of the
        unshifted one -- what the token interleaving of the sliding-window
        oversampling in :func:`GetDINOV3SAT` relies on.

    Returns
    -------
    lon, lat : array [M, rows, cols]
        Row 0 is the northernmost, column 0 the westernmost: the images are
        north-up and east-right, whatever the HEALPix face.
    """
    rows, cols = (size, size) if np.isscalar(size) else (int(size[0]), int(size[1]))
    xi = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0 + shift[1]) * gsd_rad
    # rows grow southwards, so a positive row shift lowers eta
    eta = ((rows - 1) / 2.0 - np.arange(rows, dtype=np.float64) - shift[0]) * gsd_rad
    ETA, XI = np.meshgrid(eta, xi, indexing="ij")                     # [rows, cols]
    return offsets_to_lonlat(centre_lon, centre_lat, XI, ETA)


def offsets_to_lonlat(
    centre_lon: np.ndarray,
    centre_lat: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gnomonic offsets (east ``xi``, north ``eta``, in tangent units) to lon/lat.

    ``xi`` and ``eta`` are broadcast against the ``[M]`` tangent points, giving
    ``[M, *xi.shape]`` outputs.
    """
    centre_lon = np.atleast_1d(np.asarray(centre_lon, dtype=np.float64))
    centre_lat = np.atleast_1d(np.asarray(centre_lat, dtype=np.float64))
    v0 = _lonlat_to_vec(centre_lon, centre_lat)
    east = np.stack([-v0[:, 1], v0[:, 0], np.zeros_like(v0[:, 0])], axis=1)
    small = np.linalg.norm(east, axis=1) < 1e-12
    east[small] = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(v0, east)
    north /= np.linalg.norm(north, axis=1, keepdims=True)

    extra = (1,) * np.ndim(xi)
    v0 = v0.reshape(v0.shape[0], *extra, 3)
    east = east.reshape(east.shape[0], *extra, 3)
    north = north.reshape(north.shape[0], *extra, 3)
    d = v0 + np.asarray(xi)[None, ..., None] * east + np.asarray(eta)[None, ..., None] * north
    return _vec_to_lonlat(d)


def lonlat_to_offsets(
    centre_lon: np.ndarray,
    centre_lat: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of :func:`offsets_to_lonlat`: lon/lat to gnomonic offsets.

    ``centre_*`` has shape ``[M]`` and ``lon`` / ``lat`` shape ``[M, ...]``;
    the returned ``xi`` (east) and ``eta`` (north) have the shape of ``lon``.
    """
    centre_lon = np.atleast_1d(np.asarray(centre_lon, dtype=np.float64))
    centre_lat = np.atleast_1d(np.asarray(centre_lat, dtype=np.float64))
    v0 = _lonlat_to_vec(centre_lon, centre_lat)
    east = np.stack([-v0[:, 1], v0[:, 0], np.zeros_like(v0[:, 0])], axis=1)
    small = np.linalg.norm(east, axis=1) < 1e-12
    east[small] = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(v0, east)
    north /= np.linalg.norm(north, axis=1, keepdims=True)

    d = _lonlat_to_vec(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    extra = (1,) * (d.ndim - 2)
    v0 = v0.reshape(v0.shape[0], *extra, 3)
    east = east.reshape(east.shape[0], *extra, 3)
    north = north.reshape(north.shape[0], *extra, 3)
    denom = (d * v0).sum(-1)
    return (d * east).sum(-1) / denom, (d * north).sum(-1) / denom


def parent_cover_px(
    centre_ids: np.ndarray,
    parent_level: int,
    gsd_m: float,
    *,
    margin_px: int = 8,
    radius: float = EARTH_RADIUS_M,
) -> Tuple[int, int]:
    """
    Size (rows, columns) of the smallest tangent image covering every parent cell.

    A HEALPix cell is a parallelogram, not a square, so a north-up image that
    must contain it entirely is larger than the cell -- and rectangular.  Both
    sides are rounded up to a multiple of the DINOv3 patch.
    """
    clon, clat = _pix2ang(parent_level, centre_ids)
    blon, blat = _boundaries_lonlat(parent_level, centre_ids, step=4)   # [M, 16]
    xi, eta = lonlat_to_offsets(clon, clat, blon, blat)
    px = gsd_m / radius
    w = 2 * np.abs(xi).max() / px + margin_px
    h = 2 * np.abs(eta).max() / px + margin_px
    up = lambda v: int(DINOV3_PATCH * np.ceil(v / DINOV3_PATCH))
    return up(h), up(w)


def sample_healpix(
    data: np.ndarray,
    cell_id: np.ndarray,
    level: int,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    interpolation: str = "bilinear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a partial-sky NESTED HEALPix map at arbitrary lon/lat.

    Cells absent from ``cell_id`` and non-finite values are treated as missing:
    the interpolation weights are renormalised over the available neighbours,
    and the returned mask is False where nothing was available.

    Returns
    -------
    values : float32 array [..., C]
    valid : bool array [...]
    """
    shape = np.shape(lon)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)

    if interpolation == "nearest":
        pix = _ang2pix(level, lon, lat)[None]                              # [1, P]
        wgt = np.ones_like(pix, dtype=np.float64)
    elif interpolation == "bilinear":
        pix, wgt = _interp_weights(level, lon, lat)                        # [4, P]
    else:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    order = np.argsort(cell_id)
    sorted_ids = cell_id[order]
    pos = np.searchsorted(sorted_ids, pix)
    pos = np.clip(pos, 0, sorted_ids.size - 1)
    present = sorted_ids[pos] == pix
    src = order[pos]                                                   # [K, P]

    vals = np.where(present[..., None], data[src], 0.0)                # [K, P, C]
    finite = present[..., None] & np.isfinite(data[src])
    w = np.where(finite, wgt[..., None], 0.0)
    tot = w.sum(axis=0)                                                # [P, C]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(tot > 0, (w * np.nan_to_num(vals)).sum(axis=0) / tot, np.nan)

    valid = (tot > 0).all(axis=-1)
    return (out.astype(np.float32).reshape(*shape, data.shape[1]),
            valid.reshape(shape))


def pow2_cover_px(centre_ids: np.ndarray, parent_level: int, gsd_m: float,
                  *, margin_px: int = 8, radius: float = EARTH_RADIUS_M) -> int:
    """
    Side of the smallest **square power-of-two** image containing every parent cell.

    A HEALPix cell and a square of the same area cannot contain one another: a
    square of side ``2**k`` px has exactly the cell's area, so roughly a fifth of
    the cell always falls outside it.  Those cells then have no token of their
    own.  Doubling the side fixes it and keeps the shape a ViT expects.
    """
    H, W = parent_cover_px(centre_ids, parent_level, gsd_m, margin_px=margin_px,
                           radius=radius)
    need = max(int(H), int(W))
    side = DINOV3_PATCH
    while side < need:
        side *= 2
    return side


def tangent_tiles(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    tile_px: Optional[Union[int, str, Tuple[int, int]]] = None,
    gsd_m: Optional[float] = None,
    shift: Tuple[float, float] = (0.0, 0.0),
    interpolation: str = "bilinear",
    fill: str = "mean",
    duplicates: str = "mean",
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample the data onto north-up tangent-plane images, one per tile centre.

    Tile centres are the HEALPix cells of ``parent_level`` present in the data;
    each image is a square of ``tile_px`` pixels at a constant ground sampling,
    so its shape is right whatever the local HEALPix distortion.  Unlike
    :func:`nested_to_tiles` the images may overlap and are not restricted to
    the content of one NESTED block.

    Returns
    -------
    tiles : float32 [M, C, S, S]
    centre_ids : int64 [M]        NESTED ids of the tile centres
    valid : bool [M, S, S]
    coverage : float [M]
    lon, lat : float64 [M, S, S]  geographic position of every pixel
    """
    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d.astype(np.float32, copy=False)
    ids = _as_numpy(cell_id).astype(np.int64)
    d, ids = _deduplicate(d, ids, duplicates)

    k = int(level) - int(parent_level)
    if k < 1:
        # this is a resampler, not a DINO entry point: the 4-level minimum is
        # GetDINOV3SAT's business, not this function's
        raise ValueError("parent_level must be coarser than level")
    gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level, radius)
    centre_ids = np.unique(ids >> (2 * k))
    if tile_px is None:
        # Square and a power of two -- the shape a ViT expects -- and large
        # enough to contain the whole parent cell, so that every cell of the
        # block has a token of its own instead of borrowing its neighbour's.
        # "exact" gives 2**k px (equal area, ~20% of the cell falls outside),
        # "cover" the smallest rectangle containing it.
        H = W = pow2_cover_px(centre_ids, parent_level, gsd, radius=radius)
    elif isinstance(tile_px, str):
        if tile_px == "exact":
            H = W = 2 ** k
        elif tile_px == "cover":
            H, W = parent_cover_px(centre_ids, parent_level, gsd, radius=radius)
        else:
            raise ValueError("tile_px must be an int, a (rows, cols) pair, "
                             "'exact' or 'cover'")
    elif np.isscalar(tile_px):
        H = W = int(tile_px)
    else:
        H, W = (int(v) for v in tile_px)

    clon, clat = _pix2ang(parent_level, centre_ids)
    tiles, valid, coverage, lon, lat = _tangent_images(
        d, ids, level, clon, clat, H, W, gsd / radius, shift=shift,
        interpolation=interpolation, fill=fill)
    return tiles, centre_ids, valid, coverage, lon, lat



def _tangent_images(
    d: np.ndarray,
    ids: np.ndarray,
    level: int,
    clon: np.ndarray,
    clat: np.ndarray,
    H: int,
    W: int,
    gsd_rad: float,
    *,
    shift: Tuple[float, float] = (0.0, 0.0),
    interpolation: str = "bilinear",
    fill: str = "mean",
    keep_lonlat: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    North-up tangent images ``[M, C, H, W]`` centred on arbitrary points.

    ``keep_lonlat=False`` does not keep the pixel positions (``lon`` and
    ``lat`` come back as ``None``): they weigh twice as much as the images.

    ``d`` / ``ids`` must already be deduplicated.  Shared by
    :func:`tangent_tiles` (centres = parent cells) and :func:`tangent_images`
    (centres = anything, e.g. the :func:`stride_centres` lattice).
    """
    clon = np.atleast_1d(np.asarray(clon, dtype=np.float64))
    clat = np.atleast_1d(np.asarray(clat, dtype=np.float64))
    M, C = clon.size, d.shape[1]

    # Sample image by image rather than in one call: the bilinear stencil
    # costs about 64 bytes per point (4 ids + 4 weights) before the values are
    # even gathered, so ~2e6 points per pass keeps the temporaries near 100 MB
    # instead of several GB on a full level-20 scene.
    chunk = max(1, 2_000_000 // max(1, H * W))
    vals = np.empty((M, H, W, C), dtype=np.float32)
    valid = np.empty((M, H, W), dtype=bool)
    lon = np.empty((M, H, W)) if keep_lonlat else None
    lat = np.empty((M, H, W)) if keep_lonlat else None
    for a in range(0, M, chunk):
        b_ = min(a + chunk, M)
        lo_, la_ = tangent_grid_lonlat(clon[a:b_], clat[a:b_], (H, W), gsd_rad, shift=shift)
        vals[a:b_], valid[a:b_] = sample_healpix(
            d, ids, level, lo_, la_, interpolation=interpolation)
        if keep_lonlat:
            lon[a:b_], lat[a:b_] = lo_, la_

    coverage = valid.reshape(M, -1).mean(axis=1) if M else np.zeros(0)

    if fill == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(vals.reshape(M, H * W, C), axis=1)
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        vals = np.where(np.isfinite(vals), vals, m[:, None, None, :])
    elif fill == "zero":
        vals = np.nan_to_num(vals, nan=0.0)
    elif fill != "nan":
        raise ValueError("fill must be 'mean', 'zero' or 'nan'")

    tiles = np.ascontiguousarray(np.transpose(vals, (0, 3, 1, 2)))     # [M, C, H, W]
    return tiles, valid, coverage, lon, lat


def tangent_images(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    centre_lon: ArrayLike,
    centre_lat: ArrayLike,
    tile_px: Union[int, Tuple[int, int]],
    *,
    gsd_m: Optional[float] = None,
    shift: Tuple[float, float] = (0.0, 0.0),
    interpolation: str = "bilinear",
    fill: str = "mean",
    duplicates: str = "mean",
    radius: float = EARTH_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    North-up tangent images of ``tile_px`` pixels centred on given points.

    Same projection, sampling and filling as :func:`tangent_tiles`, but the
    centres are free -- the points of :func:`stride_centres`, for instance.

    Returns
    -------
    tiles : float32 [M, C, H, W]
    valid : bool [M, H, W]
    coverage : float [M]
    lon, lat : float64 [M, H, W]
    """
    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d.astype(np.float32, copy=False)
    ids = _as_numpy(cell_id).astype(np.int64)
    d, ids = _deduplicate(d, ids, duplicates)
    H, W = (int(tile_px), int(tile_px)) if np.isscalar(tile_px) else (int(v) for v in tile_px)
    gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level, radius)
    return _tangent_images(d, ids, level, centre_lon, centre_lat, H, W, gsd / radius,
                           shift=shift, interpolation=interpolation, fill=fill)


def _stride_n(stride: float) -> int:
    """``n`` such that ``stride = 1/n``, checking that ``n`` is a power of two."""
    stride = float(stride)
    n = int(round(1.0 / stride)) if stride > 0 else 0
    if n < 1 or (n & (n - 1)) or abs(1.0 / n - stride) > 1e-9:
        raise ValueError("stride must be 1/n with n a power of two (1, 0.5, 0.25, ...)")
    return n


def stride_centres(
    parent_ids: ArrayLike,
    parent_level: int,
    stride: float = 0.5,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Image centres of a sliding window of step ``stride`` cell over the parent cells.

    With ``stride = 1`` the centres are the parent cells' centres.  With
    ``stride = 1/n`` (``n`` a power of two) the lattice is ``n`` times denser
    along both face axes: between two neighbouring centres come ``n - 1``
    evenly spaced points -- the midpoint for ``n = 2`` -- and in between four of
    them, the corresponding interior points (their centre for ``n = 2``).

    Those points are exact HEALPix points, defined across face boundaries: in
    face coordinates a parent cell ``(x, y)`` of size 1 has its centre at
    ``(x + 1/2, y + 1/2)``, and the lattice points belonging to it are
    ``(x + i/n, y + j/n)`` for ``i, j = 1..n`` -- the north corners (largest
    ``x`` and ``y``) of its ``n**2`` children at ``parent_level + log2(n)``.
    The child ``(n/2 - 1, n/2 - 1)`` has the parent centre as north corner, so
    the ``stride = 1`` images are a subset of the ``stride = 1/n`` ones.

    Returns
    -------
    cell_id : int64 [M * n**2]
        The children whose north corner is the image centre (the parent cells
        themselves when ``stride = 1``).
    cell_level : int
        ``parent_level + log2(n)``.
    lon, lat : float64 [M * n**2]
        The image centres, in degrees.
    on_parent_grid : bool [M * n**2]
        True for the centres of the parent cells, i.e. the ``stride = 1`` images.
    """
    parents = np.unique(_as_numpy(parent_ids).astype(np.int64))
    n = _stride_n(stride)
    if n == 1:
        lon, lat = _pix2ang(parent_level, parents)
        return parents, int(parent_level), lon, lat, np.ones(parents.size, bool)
    m = int(np.log2(n))
    child = ((parents[:, None] << (2 * m)) + np.arange(4 ** m, dtype=np.int64)[None]).reshape(-1)
    blon, blat = _boundaries_lonlat(parent_level + m, child, step=1)    # [M, 4]
    # corner 2 of healpix-geo's outline is the north corner (largest x and y)
    lx, ly = nested_to_xy(child & (4 ** m - 1))
    on_grid = (lx == n // 2 - 1) & (ly == n // 2 - 1)
    return child, int(parent_level) + m, blon[:, 2], blat[:, 2], on_grid



# ---------------------------------------------------------------------------
# Input range: what the network expects
# ---------------------------------------------------------------------------
#
# DINOv3 SAT-493M was trained on 8-bit RGB imagery divided by 255, then
# normalised with SAT493M_MEAN / SAT493M_STD.  The network therefore expects
# values in [0, 1] with the brightness of an 8-bit visual product.  Raw stores
# come in anything -- reflectance (0-1), Sentinel-2 DN (0-10000), 8-bit, or
# arbitrary scaled units -- and feeding them unchanged gives a saturated or
# black image whose tokens carry almost nothing but position.  Every call to
# GetDINOV3SAT therefore maps its input through ``input_range`` first.

REFLECTANCE_WHITE: float = 0.3      # reflectance shown as white in "reflectance" mode


def input_range_of(
    data: ArrayLike,
    input_range: Union[str, Tuple[float, float]] = "auto",
    *,
    percentiles: Tuple[float, float] = (1.0, 99.0),
    sample: int = 2_000_000,
) -> Tuple[float, float, str]:
    """
    ``(lo, hi, how)`` such that ``clip((x - lo) / (hi - lo), 0, 1)`` is what DINOv3 SAT expects.

    input_range
        ``"auto"`` / ``"percentile"``: the ``percentiles`` of all the finite
        values passed (all bands together, so colours are kept) -- works
        whatever the units; ``"reflectance"``: ``(0, REFLECTANCE_WHITE)``;
        ``"uint8"``: ``(0, 255)``; ``"unit"``: ``(0, 1)``, the data are already
        in the network's range; or an explicit ``(lo, hi)`` pair -- the way to
        apply the range of a first date unchanged to the next ones.
    """
    if not isinstance(input_range, str):
        lo, hi = (float(v) for v in input_range)
        how = "fixed"
    elif input_range == "unit":
        lo, hi, how = 0.0, 1.0, "unit"
    elif input_range == "uint8":
        lo, hi, how = 0.0, 255.0, "uint8"
    elif input_range == "reflectance":
        lo, hi, how = 0.0, REFLECTANCE_WHITE, "reflectance"
    elif input_range in ("auto", "percentile"):
        x = np.asarray(_as_numpy(data), dtype=np.float64).reshape(-1)
        if x.size > sample:
            x = x[np.random.default_rng(0).integers(0, x.size, sample)]
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("no finite value in the input")
        lo, hi = (float(v) for v in np.percentile(x, percentiles))
        how = "percentile"
    else:
        raise ValueError("input_range must be 'auto', 'percentile', 'reflectance', "
                         "'uint8', 'unit' or a (lo, hi) pair")
    if not hi > lo:
        raise ValueError(f"empty input range ({lo}, {hi}): constant data?")
    return lo, hi, how


def scale_input(
    data: ArrayLike,
    input_range: Union[str, Tuple[float, float]] = "auto",
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], str]:
    """
    Map the data to the [0, 1] range DINOv3 SAT expects (NaN kept).

    Returns the scaled float32 array, the ``(lo, hi)`` used and how it was
    chosen.  ``verbose`` prints the raw and scaled ranges and the saturated
    fraction -- the check to make before trusting any embedding.
    """
    d = np.asarray(_as_numpy(data), dtype=np.float32)
    lo, hi, how = input_range_of(d, input_range)
    out = (d - np.float32(lo)) / np.float32(hi - lo)
    fin = np.isfinite(out)
    sat = float(((out[fin] < 0) | (out[fin] > 1)).mean()) if fin.any() else 0.0   # strictly outside
    out = np.where(fin, np.clip(out, 0.0, 1.0), np.nan).astype(np.float32)
    if verbose:
        raw = d[np.isfinite(d)]
        q = np.percentile(raw, [1, 50, 99]) if raw.size else [np.nan] * 3
        m = np.nanmedian(out) if fin.any() else np.nan
        print(f"DINO input: raw p1 {q[0]:.4g}  median {q[1]:.4g}  p99 {q[2]:.4g} -> "
              f"'{how}' [{lo:.4g}, {hi:.4g}] -> [0, 1] (x255: median {255 * m:.0f}), "
              f"{100 * sat:.1f}% clipped")
        if sat > 0.2:
            print("  /!\\ more than 20% of the values clipped: input_range is probably wrong "
                  "for these units")
        if fin.any() and (m < 0.05 or m > 0.95):
            print("  /!\\ the scaled image is nearly black or white: check input_range")
    return out, (lo, hi), how

def load_dinov3_sat(
    model_name: str = "dinov3_vitl16",
    weights: Optional[str] = None,
    *,
    source: str = "auto",
    repo: str = "facebookresearch/dinov3",
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Load a DINOv3 SAT-493M backbone in evaluation mode.

    Parameters
    ----------
    model_name : {"dinov3_vitl16", "dinov3_vit7b16"}
        SAT-493M is released for ViT-L/16 and ViT-7B/16 only.
    weights : str, optional
        Path to the ``.pth`` checkpoint (torch-hub route; DINOv3 weights are
        gated and must be downloaded after accepting the licence) or a
        Hugging Face model id / local directory (HF route).
    source : {"auto", "hub", "hf"}
        "hub" uses ``torch.hub.load(repo, model_name, weights=weights)``;
        "hf" uses ``transformers.AutoModel``; "auto" tries hub then hf.
    repo : str
        Torch-hub repo or a local clone of ``facebookresearch/dinov3``.
    device : str or torch.device, optional
    """
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if weights is None and source != "hf":
        raise ValueError("DINOv3 SAT-493M weights are gated; pass `weights`.")
    # A path that looks like a checkpoint but is not there fails deep inside the
    # hub code with an unhelpful message; say it plainly instead.
    if isinstance(weights, str) and weights.endswith(".pth"):
        import os
        if not os.path.exists(os.path.expanduser(weights)):
            raise FileNotFoundError(f"checkpoint not found: {weights}")
    errors = []
    if source in ("auto", "hub"):
        try:
            src = "local" if (repo.startswith((".", "/", "~")) or ":" in repo[:3]) else "github"
            model = torch.hub.load(repo, model_name, source=src, weights=weights)
            # torch.hub prints "Using cache found in .../facebookresearch_dinov3_main"
            # about the *source code* it cached, never about the weights; say which
            # checkpoint was actually loaded so the two are not confused.
            print(f"DINOv3 {model_name}: poids charges depuis {weights}")
            return model.eval().to(device)
        except Exception as e:                         # noqa: BLE001
            errors.append(f"hub: {e!r}")
            if source == "hub":
                raise
    if source in ("auto", "hf"):
        try:
            from transformers import AutoModel
            hf_id = weights if weights is not None else _HF_SAT_IDS[model_name]
            model = AutoModel.from_pretrained(hf_id)
            return model.eval().to(device)
        except Exception as e:                         # noqa: BLE001
            errors.append(f"hf: {e!r}")
    # A missing ancillary package is by far the most common failure: the DINOv3
    # hub code imports a few helpers of its own.  Say which one, and how to fix
    # it, rather than leaving the caller with a generic message.
    missing = []
    for e in errors:
        m = re.search(r"No module named '([^']+)'", e)
        if m and m.group(1) not in missing:
            missing.append(m.group(1))
    hint = ""
    if missing:
        deps = [m for m in missing if m != "transformers"]
        if deps:
            hint = ("\nThe DINOv3 code needs a package that is not installed here: "
                    f"pip install {' '.join(deps)}  (then restart the kernel)")
        elif missing == ["transformers"]:
            hint = ("\nThe Hugging Face route needs: pip install transformers  "
                    "(or use the torch-hub route with `weights=<path to the .pth>`)")
    raise RuntimeError("Could not load DINOv3: " + " | ".join(errors) + hint)


def _forward(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the backbone and return ``(cls [B, D], patches [B, P, D])`` for both
    the torch-hub (``forward_features``) and the HF (``AutoModel``) APIs.
    """
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    out = model(pixel_values=x)
    h = out.last_hidden_state                                       # [B, 1+R+P, D]
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return h[:, 0], h[:, 1 + n_reg:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DINOEmbedding:
    """
    Result of :func:`GetDINOV3SAT`.

    Attributes
    ----------
    embedding : array [M, Ndino]
        One vector per tile of ``parent_level`` (CLS token by default).
    cell_id : int64 array [M]
        NESTED ids of the tiles at ``parent_level``.
    parent_level : int
    coverage : float array [M]
        Fraction of ``level`` pixels available in each tile.
    patch_embedding : array [M * P, Ndino] or None
        Patch tokens (``return_patches=True``); one per HEALPix cell at
        ``patch_level = level - 4``.
    patch_cell_id : int64 array [M * P] or None
    patch_level : int or None
    stride : float
        Step between image centres, in parent cells (tangent mode).
    cell_level : int
        Level of ``cell_id``: ``parent_level`` when ``stride = 1``,
        ``parent_level + log2(1 / stride)`` otherwise, the image centre being
        then the north corner of the cell (see :func:`stride_centres`).
    centre_lon, centre_lat : float arrays [M]
        Image centres, in degrees (tangent mode).
    on_parent_grid : bool array [M]
        True for the images centred on a parent cell, i.e. those that
        ``stride = 1`` produces.
    input_range : (float, float)
        The ``(lo, hi)`` mapped to [0, 1] before the network; pass it as
        ``input_range`` to process other dates with the same mapping.
    input_scaling : str
        How it was chosen ("percentile", "reflectance", "fixed", ...).
    """

    embedding: np.ndarray
    cell_id: np.ndarray
    parent_level: int
    coverage: np.ndarray
    patch_embedding: Optional[np.ndarray] = None
    patch_cell_id: Optional[np.ndarray] = None
    patch_level: Optional[int] = None
    patch_lon: Optional[np.ndarray] = None
    patch_lat: Optional[np.ndarray] = None
    projection: str = "tangent"
    over_sample: int = 1
    gsd_m: Optional[float] = None
    tile_px: Optional[int] = None
    stride: float = 1.0
    cell_level: Optional[int] = None
    centre_lon: Optional[np.ndarray] = None
    centre_lat: Optional[np.ndarray] = None
    on_parent_grid: Optional[np.ndarray] = None
    input_range: Optional[Tuple[float, float]] = None
    input_scaling: Optional[str] = None


def _percell_embeddings(
    d: np.ndarray,
    ids: np.ndarray,
    level: int,
    out_cells: np.ndarray,
    *,
    token_level: int,
    context_px: int,
    gsd: float,
    interpolation: str,
    fill: str,
    model: nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    pooling: str,
    batch_size: int,
    dev: torch.device,
    autocast: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One tangent plane, and one embedding, per output cell.

    For every cell of ``token_level`` a north-up image of ``context_px`` pixels
    is built, tangent at that cell's centre, and passed through the backbone.
    Nothing is interleaved and nothing is resampled back: the window is centred
    exactly on the cell, in the cell's own frame.  The cost is one forward pass
    per cell, against one per tile for the sliding-window mode -- three orders
    of magnitude more on a typical scene -- so this is the reference to
    validate against on a small area, not the production path.

    Returns
    -------
    emb : float32 [Ncell, D]
    coverage : float [Ncell]
    """
    lon0, lat0 = _pix2ang(token_level, out_cells)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    use_ac = autocast and dev.type == "cuda"
    g = context_px // DINOV3_PATCH                       # tokens per axis
    centre_token = (g // 2) * g + (g // 2)               # patch holding the centre

    out, cov = [], []
    with torch.inference_mode():
        for i in range(0, out_cells.size, batch_size):
            lo, la = lon0[i:i + batch_size], lat0[i:i + batch_size]
            lon, lat = tangent_grid_lonlat(lo, la, context_px, gsd / EARTH_RADIUS_M)
            vals, valid = sample_healpix(d, ids, level, lon, lat,
                                         interpolation=interpolation)
            b, C = vals.shape[0], vals.shape[-1]
            cov.append(valid.reshape(b, -1).mean(axis=1))
            if fill == "mean":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    m = np.nanmean(vals.reshape(b, -1, C), axis=1)
                m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
                vals = np.where(~np.isfinite(vals),
                                np.broadcast_to(m[:, None, None, :], vals.shape), vals)
            else:
                vals = np.nan_to_num(vals, nan=0.0)

            x = torch.from_numpy(np.ascontiguousarray(
                np.transpose(vals, (0, 3, 1, 2)))).to(dev)
            x = (x - mean_t) / std_t
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_ac):
                cls, patches = _forward(model, x)
            cls, patches = cls.float(), patches.float()
            if pooling == "centre":
                v = patches[:, centre_token]
            elif pooling == "cls":
                v = cls
            elif pooling == "mean":
                v = patches.mean(dim=1)
            elif pooling == "cls+mean":
                v = torch.cat([cls, patches.mean(dim=1)], dim=1)
            else:
                raise ValueError("pooling must be 'centre', 'cls', 'mean' or 'cls+mean'")
            out.append(v.cpu().numpy())

    if not out:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(out).astype(np.float32), np.concatenate(cov)


def GetDINOV3SAT(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    projection: str = "tangent",
    over_sample: int = 1,
    stride: float = 1.0,
    context_px: Optional[int] = None,
    out_cells: Optional[ArrayLike] = None,
    gsd_m: Optional[float] = None,
    tile_px: Optional[int] = None,
    interpolation: str = "bilinear",
    model: Optional[nn.Module] = None,
    weights: Optional[str] = None,
    model_name: str = "dinov3_vitl16",
    bands: Sequence[int] = (0, 1, 2),
    input_range: Union[str, Tuple[float, float]] = "auto",
    verbose: bool = True,
    mean: Sequence[float] = SAT493M_MEAN,
    std: Sequence[float] = SAT493M_STD,
    pooling: str = "cls",
    return_patches: bool = False,
    min_coverage: float = 0.0,
    fill: str = "mean",
    duplicates: str = "mean",
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    autocast: bool = True,
) -> DINOEmbedding:
    """
    DINOv3 SAT-493M embeddings of NESTED HEALPix data.

    Every HEALPix cell of ``parent_level`` present in ``cell_id`` gives one
    square image, which is passed to an unmodified DINOv3 backbone.  How that
    image is built is the important choice:

    ``projection="tangent"`` (default)
        The image is resampled onto a north-up gnomonic plane tangent at the
        cell centre, with a constant ground sampling.  Shapes are then correct:
        HEALPix cells have equal area but not equal shape, and at 52 deg N the
        two NESTED axes measure 10.6 m and 16.7 m with a 60.5 deg angle between
        them, so the "square" of the ``nested`` mode is really a parallelogram
        stretched by a factor 2.  A network trained on map-projected imagery
        cannot be expected to see through that.  The image is sized to contain
        the whole (parallelogram-shaped) cell, so it is rectangular and a
        little larger than the cell.

        The tokens sit on a square metric grid while the cells are
        parallelograms, so the output is *not* built by assigning each token to
        the cell it falls in -- that would leave some cells with two tokens and
        others with none, a moire of holes on a map.  Instead the cells of the
        token level inside each tile are enumerated and the token field is read
        at their positions: every cell carries exactly one embedding, and the
        output tiles the sphere without holes.  ``patch_lon`` / ``patch_lat``
        are then the cell centres.  The price is a resampling, twice.

    ``projection="percell"``
        One tangent plane **per output cell**: for every cell of
        ``parent_level`` a north-up image of ``context_px`` pixels is built,
        tangent at that cell's centre, and the embedding is the patch token
        holding the centre (or ``pooling="cls"`` for the whole window).  The
        window is centred exactly on the cell, in the cell's own frame, so
        nothing is interleaved and nothing is resampled back -- this is the
        exact sliding window, and ``level - parent_level >= 4`` no longer
        applies since the cell may be finer than a patch.

        It costs one forward pass per cell, but each window is small: with the
        default ``context_px`` -- the cell's own footprint -- the total number
        of tokens equals the number of output cells, exactly what the tiled
        modes produce, so this is affordable.  What is expensive is context: a
        window ``m`` patches wide costs ``m**2`` tokens per cell.

    ``projection="nested"``
        The historical mode: the NESTED block is reshaped into an image, which
        is exact in index space and free of resampling, and every token maps
        onto exactly one cell of ``level - 4``.  Geometrically wrong away from
        the equator; kept for comparison.

    Parameters
    ----------
    data : array [N, C]
        Values at ``level``, in any units: they are mapped to the network's
        range by ``input_range``.  NaN marks a missing pixel.
    cell_id : int array [N]
        NESTED cell ids at ``level``.
    level : int
        Resolution of ``data``.
    parent_level : int
        Resolution of the tile centres; ``level - parent_level >= 4`` is
        required (one DINO patch is 16 px = 4 HEALPix levels).  ``level - 8``
        (256 px tiles, 16x16 patch tokens) is the natural choice.
    projection : {"tangent", "nested"}
        See above.
    context_px : int, optional
        Side of the window in ``percell`` mode, a multiple of 16.  The default
        is the cell's own footprint, so the total number of tokens equals the
        number of output cells -- the same count as the tiled modes.  A larger
        window gives the network context, at a cost growing as
        ``(context_px / cell)**2``: for cells of 16 px, 48 px costs 9 times
        more and 224 px 196 times.  With a centred pooling an even number of
        patches per axis leaves no token on the centre, so the value is raised
        by one patch.
    out_cells : int array, optional
        Restrict ``percell`` mode to these cells instead of every cell of
        ``parent_level`` present in the data.  The way to try it on a small
        area before paying for the whole scene.
    over_sample : int
        Sliding-window factor, a power of two.  ``1`` (default) runs the
        patches side by side, exactly as the network does.  ``n`` runs the
        network ``n**2`` times on windows shifted by ``16 / n`` pixels and
        interleaves the tokens, giving a token field ``n`` times denser in each
        direction, at ``level - 4 + log2(n)``.  Cost grows as ``n**2``.
    stride : float
        Tangent mode only: step between image centres, in parent cells.
        ``1`` (default) gives one image per parent cell.  ``1/2`` adds the
        images centred between two neighbouring cells and between four, with
        the same projection, resolution and size -- a sliding window of half a
        cell over the images: a block of ``G x G`` cells gives ``(2G - 1)**2``
        images inside it (``2G`` per side once the half steps towards the next
        block are counted).  ``1/n`` generalises, ``n`` a power of two.  The
        images are identified by ``cell_id`` at ``cell_level`` and located by
        ``centre_lon`` / ``centre_lat``; see :func:`stride_centres`.  Not
        compatible with ``over_sample > 1`` nor ``return_patches``.
    gsd_m : float, optional
        Ground sampling of the tangent grid, in metres.  Defaults to the
        HEALPix native value ``sqrt(cell area)`` at ``level``.
    tile_px : int or (int, int), optional
        Size of the images, ``(rows, columns)``.  In tangent mode the default
        is the smallest multiple of 16 containing every parent cell (see
        :func:`parent_cover_px`); in nested mode it is
        ``2**(level - parent_level)``.
    interpolation : {"bilinear", "nearest"}
        Resampling onto the tangent grid.
    model : nn.Module, optional
        A backbone already loaded with :func:`load_dinov3_sat`.  When
        ``None`` the model is loaded from ``weights`` / ``model_name``.
    weights, model_name :
        Passed to :func:`load_dinov3_sat` when ``model`` is ``None``.
    bands : sequence of 3 int
        Indices of the (R, G, B) bands in ``data`` -- Sentinel-2 B04, B03, B02.
    input_range : str or (float, float)
        How the data are mapped to the [0, 1] range the network expects (an
        8-bit RGB image divided by 255): see :func:`input_range_of`.  The
        default ``"auto"`` stretches the 1-99 percentiles of the whole input
        (all bands together) to [0, 1] and clips, whatever the units.  For a
        time series, pass the ``res.input_range`` of the first date to the
        others so that they share one mapping.
    verbose : bool
        Print the input ranges before and after scaling.
    mean, std : sequence of 3 float
        Normalisation constants (SAT-493M defaults).
    pooling : {"cls", "mean", "cls+mean", "centre"}
        Tile vector: CLS token, mean of the patch tokens, or their
        concatenation (``2 * Ndino``).  In ``percell`` mode the default is the
        patch token holding the cell centre (``"centre"``), which is what
        describes the cell itself rather than its whole window.
    return_patches : bool
        Also return the patch tokens.
    min_coverage : float
        Drop tiles whose fraction of usable pixels is below this value; 1.0
        keeps only the complete ones.
    fill : {"mean", "zero"}
        How missing pixels are filled before normalisation.
    duplicates : {"mean", "first", "error"}
        How repeated cell ids are aggregated (see :func:`nested_to_tiles`).
    batch_size : int
    device : str or torch.device, optional
    autocast : bool
        Use bfloat16 autocast on CUDA.

    Returns
    -------
    DINOEmbedding
    """
    k = int(level) - int(parent_level)
    if k < 1:
        raise ValueError("parent_level must be coarser than level")
    if k < DINOV3_PATCH_LEVELS and projection != "percell":
        raise ValueError(
            f"level - parent_level must be >= {DINOV3_PATCH_LEVELS} "
            f"(one DINOv3 patch = {DINOV3_PATCH} px), got {k}. "
            "projection='percell' has no such constraint: it centres a window "
            "on every cell instead of tiling."
        )
    if len(bands) != 3:
        raise ValueError("DINOv3 SAT expects 3 bands (R, G, B)")
    if fill == "nan":
        raise ValueError("fill='nan' cannot be fed to the network")
    n = int(over_sample)
    if n < 1 or (n & (n - 1)):
        raise ValueError("over_sample must be a power of two (1, 2, 4, ...)")
    if n > DINOV3_PATCH:
        raise ValueError(f"over_sample must be <= {DINOV3_PATCH}")
    stride = float(stride)
    if stride != 1.0:
        if projection != "tangent":
            raise ValueError("stride is only available with projection='tangent'")
        if n > 1 or return_patches:
            raise ValueError("stride < 1 cannot be combined with over_sample > 1 "
                             "or return_patches")
        _stride_n(stride)                                       # validates the value
    if projection not in ("tangent", "nested", "percell"):
        raise ValueError("projection must be 'tangent', 'nested' or 'percell'")

    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d[:, list(bands)]
    ids = _as_numpy(cell_id).astype(np.int64)
    d, ids = _deduplicate(d, ids, duplicates)
    # the network expects an 8-bit-like RGB image in [0, 1]; never feed raw units
    d, in_range, in_how = scale_input(d, input_range, verbose=verbose)

    # ---- one tangent plane per output cell --------------------------------
    if projection == "percell":
        # Default window: the cell's own footprint.  The total number of tokens
        # is then the number of output cells -- the same count the tiled modes
        # produce -- which is what makes per-cell inference affordable.
        cell_px = DINOV3_PATCH * max(1, int(2 ** (k - DINOV3_PATCH_LEVELS)))
        ctx = int(context_px) if context_px is not None else cell_px
        if ctx % DINOV3_PATCH:
            raise ValueError(f"context_px must be a multiple of {DINOV3_PATCH}")
        if pooling != "mean" and (ctx // DINOV3_PATCH) % 2 == 0:
            # with an even number of patches per axis no token is centred on
            # the cell; one more patch puts the centre token exactly on it
            ctx += DINOV3_PATCH
        context_px = ctx
        gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level)
        tl = int(parent_level)                      # the output level, one embedding each
        cells = (np.unique(ids >> (2 * k)) if out_cells is None
                 else np.unique(_as_numpy(out_cells).astype(np.int64)))

        if model is None:
            model = load_dinov3_sat(model_name, weights, device=device)
        dev = next(model.parameters()).device if device is None else torch.device(device)
        model = model.to(dev).eval()

        emb, cov = _percell_embeddings(
            d, ids, level, cells, token_level=tl, context_px=context_px, gsd=gsd,
            interpolation=interpolation, fill=fill, model=model, mean=mean, std=std,
            pooling=("centre" if pooling == "cls" else pooling),
            batch_size=batch_size, dev=dev, autocast=autocast,
        )
        keep = cov >= float(min_coverage)
        res = DINOEmbedding(
            embedding=emb[keep], cell_id=cells[keep], parent_level=tl,
            coverage=cov[keep], projection="percell", over_sample=1,
            gsd_m=gsd, tile_px=(context_px, context_px),
            input_range=in_range, input_scaling=in_how,
        )
        # the same field is exposed through the patch_* names so that code
        # written for the tiled modes keeps working unchanged
        res.patch_embedding, res.patch_cell_id, res.patch_level = (
            res.embedding, res.cell_id, tl)
        res.patch_lon, res.patch_lat = _pix2ang(tl, res.cell_id)
        return res

    step = DINOV3_PATCH // n                       # window stride, in pixels
    shifts = [(a * step, b * step) for a in range(n) for b in range(n)]

    # ---- build the images ------------------------------------------------
    if projection == "tangent":
        gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level)
        all_centres = np.unique(ids >> (2 * k))
        if tile_px is None:
            H = W = pow2_cover_px(all_centres, parent_level, gsd)
        elif isinstance(tile_px, str):
            if tile_px == "exact":
                H = W = 2 ** k
            elif tile_px == "cover":
                H, W = parent_cover_px(all_centres, parent_level, gsd)
            else:
                raise ValueError("tile_px must be an int, a (rows, cols) pair, "
                                 "'exact' or 'cover'")
        elif np.isscalar(tile_px):
            H = W = int(tile_px)
        else:
            H, W = (int(v) for v in tile_px)
        if H % DINOV3_PATCH or W % DINOV3_PATCH:
            raise ValueError(f"tile_px must be a multiple of {DINOV3_PATCH}")

        if stride != 1.0:
            # sliding window over the images: centres every `stride` cell
            centre_ids, cell_level, c_lon, c_lat, on_grid = stride_centres(
                all_centres, parent_level, stride)
            tiles0, _valid, coverage, _lon, _lat = _tangent_images(
                d, ids, level, c_lon, c_lat, H, W, gsd / EARTH_RADIUS_M,
                interpolation=interpolation, fill=fill, keep_lonlat=False)
        else:
            tiles0, centre_ids, _valid, coverage, _lon, _lat = tangent_tiles(
                d, ids, level, parent_level, tile_px=(H, W), gsd_m=gsd,
                interpolation=interpolation, fill=fill, duplicates=duplicates,
            )
            cell_level, on_grid = int(parent_level), np.ones(centre_ids.size, bool)
            c_lon, c_lat = _pix2ang(parent_level, centre_ids)
        keep = coverage >= float(min_coverage)
        centre_ids, coverage = centre_ids[keep], coverage[keep]
        c_lon, c_lat, on_grid = c_lon[keep], c_lat[keep], on_grid[keep]
        views = []
        for sh in shifts:
            if sh == (0.0, 0.0):
                views.append(tiles0[keep])
            else:
                views.append(tangent_tiles(
                    d, ids, level, parent_level, tile_px=(H, W), gsd_m=gsd, shift=sh,
                    interpolation=interpolation, fill=fill, duplicates=duplicates,
                )[0][keep])
        tok_h, tok_w = H // DINOV3_PATCH, W // DINOV3_PATCH
    else:
        S = int(tile_px) if tile_px is not None else (1 << k)
        if S % DINOV3_PATCH:
            raise ValueError(f"tile_px must be a multiple of {DINOV3_PATCH}")
        tiles0, centre_ids, _valid, coverage = nested_to_tiles(
            d, ids, level, parent_level, fill=fill, duplicates=duplicates
        )
        keep = coverage >= float(min_coverage)
        tiles0, centre_ids, coverage = tiles0[keep], centre_ids[keep], coverage[keep]
        tok_h = tok_w = S // DINOV3_PATCH - (1 if n > 1 else 0)
        views = []
        for (dy, dx) in shifts:
            h, w = DINOV3_PATCH * tok_h, DINOV3_PATCH * tok_w
            views.append(np.ascontiguousarray(tiles0[:, :, dy:dy + h, dx:dx + w]))
        H, W = S, S

    M = views[0].shape[0]

    # ---- forward ---------------------------------------------------------
    if model is None:
        model = load_dinov3_sat(model_name, weights, device=device)
    dev = next(model.parameters()).device if device is None else torch.device(device)
    model = model.to(dev).eval()

    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    use_ac = autocast and dev.type == "cuda"

    cls_sum, D = None, 0
    tok = None                                     # [M, n*tok_h, n*tok_w, D]
    with torch.inference_mode():
        for v, (a, b) in zip(views, [(a, b) for a in range(n) for b in range(n)]):
            cls_out, patch_out = [], []
            for i in range(0, M, batch_size):
                x = torch.from_numpy(v[i:i + batch_size]).to(dev)
                x = (x - mean_t) / std_t
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_ac):
                    cls, patches = _forward(model, x)
                cls, patches = cls.float(), patches.float()
                if pooling == "cls":
                    val = cls
                elif pooling == "mean":
                    val = patches.mean(dim=1)
                elif pooling == "cls+mean":
                    val = torch.cat([cls, patches.mean(dim=1)], dim=1)
                else:
                    raise ValueError("pooling must be 'cls', 'mean' or 'cls+mean'")
                cls_out.append(val.cpu())
                if return_patches:
                    patch_out.append(patches.cpu())
            c = torch.cat(cls_out).numpy() if cls_out else np.zeros((0, 0), np.float32)
            cls_sum = c if cls_sum is None else cls_sum + c
            if return_patches and patch_out:
                pt = torch.cat(patch_out).numpy()
                if pt.shape[1] != tok_h * tok_w:
                    raise RuntimeError(
                        f"got {pt.shape[1]} patch tokens for a {tok_h}x{tok_w} grid; "
                        "is the model patch size 16?"
                    )
                D = pt.shape[-1]
                if tok is None:
                    tok = np.zeros((M, tok_h * n, tok_w * n, D), np.float32)
                tok[:, a::n, b::n] = pt.reshape(M, tok_h, tok_w, D)

    embedding = (cls_sum / len(views)) if cls_sum is not None else np.zeros((0, 0), np.float32)

    res = DINOEmbedding(
        embedding=embedding,
        cell_id=centre_ids,
        parent_level=int(parent_level),
        coverage=coverage,
        projection=projection,
        over_sample=n,
        gsd_m=(gsd if projection == "tangent" else None),
        tile_px=(H, W),
        input_range=in_range,
        input_scaling=in_how,
    )
    if projection == "tangent":
        res.stride, res.cell_level = stride, cell_level
        res.centre_lon, res.centre_lat, res.on_parent_grid = c_lon, c_lat, on_grid
    if not return_patches:
        return res

    token_level = int(level) - DINOV3_PATCH_LEVELS + int(np.log2(n))
    res.patch_level = token_level

    if projection == "nested":
        Fh, Fw = tok_h * n, tok_w * n
        full = S * n // DINOV3_PATCH               # cells per axis at token_level
        off = int((DINOV3_PATCH - 1) / 2.0 // step)
        f = np.arange(Fh) + off
        col = np.tile(f, (Fh, 1))
        row = np.repeat(f[:, None], Fw, axis=1)
        rel = xy_to_nested(col, (full - 1) - row)             # [Fh, Fw]
        shift_bits = 2 * (token_level - int(parent_level))
        res.patch_cell_id = ((centre_ids[:, None, None] << shift_bits) + rel[None]).reshape(-1)
        res.patch_embedding = (tok.reshape(-1, D) if tok is not None
                               else np.zeros((0, 0), np.float32))
        return res

    # Tangent mode: the tokens sit on a square metric grid while the HEALPix
    # cells are parallelograms, so assigning each token to the cell it falls in
    # gives some cells two tokens and others none -- a moire of holes on a map.
    # Go the other way instead: enumerate the cells of ``token_level`` inside
    # each tile (they tile it exactly, by construction) and read the token
    # field at their positions.  Every cell then carries exactly one embedding
    # and the output has no holes.
    c_lev = token_level - int(parent_level)
    child = ((centre_ids[:, None] << (2 * c_lev))
             + np.arange(4 ** c_lev, dtype=np.int64)[None])          # [M, Nc]
    lon, lat = _pix2ang(token_level, child.reshape(-1))
    lon, lat = lon.reshape(child.shape), lat.reshape(child.shape)

    clon, clat = _pix2ang(parent_level, centre_ids)
    xi, eta = lonlat_to_offsets(clon, clat, lon, lat)                # [M, Nc]

    px = gsd / EARTH_RADIUS_M
    half = (DINOV3_PATCH - 1) / 2.0
    col = (xi / px + (W - 1) / 2.0 - half) / step
    row = ((H - 1) / 2.0 - eta / px - half) / step
    outside = ((col < -0.5) | (col > tok_w * n - 0.5)
               | (row < -0.5) | (row > tok_h * n - 0.5))
    if outside.any():
        warnings.warn(
            f"{int(outside.sum())} of {outside.size} cells fall outside the tangent "
            "image and take the nearest token; use a larger tile_px",
            RuntimeWarning, stacklevel=2,
        )
    ci = np.clip(np.rint(col).astype(np.int64), 0, tok_w * n - 1)
    ri = np.clip(np.rint(row).astype(np.int64), 0, tok_h * n - 1)

    m = np.arange(child.shape[0])[:, None]
    res.patch_embedding = (tok[m, ri, ci].reshape(-1, D) if tok is not None
                           else np.zeros((0, 0), np.float32))
    res.patch_cell_id = child.reshape(-1)
    res.patch_lon, res.patch_lat = lon.reshape(-1), lat.reshape(-1)
    return res



__all__ = [
    "GetDINOV3SAT",
    "DINOEmbedding",
    "load_dinov3_sat",
    "nested_to_tiles",
    "tiles_to_nested",
    "tile_grid_cell_ids",
    "tangent_tiles",
    "tangent_images",
    "stride_centres",
    "tangent_grid_lonlat",
    "offsets_to_lonlat",
    "sample_healpix",
    "healpix_gsd_m",
    "lonlat_to_offsets",
    "parent_cover_px",
    "pow2_cover_px",
    "nested_to_xy",
    "xy_to_nested",
    "cell_centres_lonlat",
    "cells_at_lonlat",
    "cell_vectors",
    "cell_neighbours",
    "set_ellipsoid",
    "scale_input",
    "input_range_of",
    "REFLECTANCE_WHITE",
    "SAT493M_MEAN",
    "SAT493M_STD",
]
