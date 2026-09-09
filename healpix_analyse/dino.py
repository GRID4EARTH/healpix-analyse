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

Dependencies: numpy, torch, and either the ``dinov3`` torch-hub repo
(``facebookresearch/dinov3``) or ``transformers`` (HF weights).
"""

from __future__ import annotations

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
# Morton (Z-order) helpers -- pure integer arithmetic, no healpy needed
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
        Extra offset of the grid origin, in pixels (row, column).  Used by the
        sliding-window oversampling.

    Returns
    -------
    lon, lat : array [M, rows, cols]
        Row 0 is the northernmost, column 0 the westernmost: the images are
        north-up and east-right, whatever the HEALPix face.
    """
    rows, cols = (size, size) if np.isscalar(size) else (int(size[0]), int(size[1]))
    xi = (np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0 + shift[1]) * gsd_rad
    eta = ((rows - 1) / 2.0 - np.arange(rows, dtype=np.float64) + shift[0]) * gsd_rad
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
    import healpy as hp

    nside = 2 ** int(parent_level)
    clon, clat = hp.pix2ang(nside, centre_ids, nest=True, lonlat=True)
    corners = hp.boundaries(nside, centre_ids, step=4, nest=True)      # [M, 3, 16]
    blon, blat = _vec_to_lonlat(np.transpose(corners, (0, 2, 1)))      # [M, 16]
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
    import healpy as hp

    nside = 2 ** int(level)
    shape = np.shape(lon)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)

    if interpolation == "nearest":
        pix = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)[None]    # [1, P]
        wgt = np.ones_like(pix, dtype=np.float64)
    elif interpolation == "bilinear":
        pix, wgt = hp.get_interp_weights(nside, lon, lat, nest=True, lonlat=True)
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


def tangent_tiles(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    tile_px: Optional[Union[int, Tuple[int, int]]] = None,
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
    import healpy as hp

    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d.astype(np.float32, copy=False)
    ids = _as_numpy(cell_id).astype(np.int64)
    d, ids = _deduplicate(d, ids, duplicates)

    k = int(level) - int(parent_level)
    if k < DINOV3_PATCH_LEVELS:
        raise ValueError(
            f"level - parent_level must be >= {DINOV3_PATCH_LEVELS}, got {k}")
    gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level, radius)
    centre_ids = np.unique(ids >> (2 * k))
    if tile_px is None:
        H, W = parent_cover_px(centre_ids, parent_level, gsd, radius=radius)
    elif np.isscalar(tile_px):
        H = W = int(tile_px)
    else:
        H, W = (int(v) for v in tile_px)

    clon, clat = hp.pix2ang(2 ** int(parent_level), centre_ids, nest=True, lonlat=True)
    lon, lat = tangent_grid_lonlat(clon, clat, (H, W), gsd / radius, shift=shift, radius=radius)

    vals, valid = sample_healpix(d, ids, level, lon, lat, interpolation=interpolation)
    M, C = centre_ids.size, d.shape[1]
    coverage = valid.reshape(M, -1).mean(axis=1)

    if fill == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(vals.reshape(M, H * W, C), axis=1)
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        bad = ~np.isfinite(vals)
        vals = np.where(bad, np.broadcast_to(m[:, None, None, :], vals.shape), vals)
    elif fill == "zero":
        vals = np.nan_to_num(vals, nan=0.0)
    elif fill != "nan":
        raise ValueError("fill must be 'mean', 'zero' or 'nan'")

    tiles = np.ascontiguousarray(np.transpose(vals, (0, 3, 1, 2)))     # [M, C, S, S]
    return tiles, centre_ids, valid, coverage, lon, lat


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
    errors = []
    if source in ("auto", "hub"):
        try:
            src = "local" if (repo.startswith((".", "/", "~")) or ":" in repo[:3]) else "github"
            model = torch.hub.load(repo, model_name, source=src, weights=weights)
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
    raise RuntimeError("Could not load DINOv3: " + " | ".join(errors))


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
    import healpy as hp

    lon0, lat0 = hp.pix2ang(2 ** int(token_level), out_cells, nest=True, lonlat=True)
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
    context_px: int = 224,
    out_cells: Optional[ArrayLike] = None,
    gsd_m: Optional[float] = None,
    tile_px: Optional[int] = None,
    interpolation: str = "bilinear",
    model: Optional[nn.Module] = None,
    weights: Optional[str] = None,
    model_name: str = "dinov3_vitl16",
    bands: Sequence[int] = (0, 1, 2),
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
        applies since the cell may be finer than a patch.  It costs one forward
        pass per cell instead of one per tile, three orders of magnitude more on
        a typical scene, so it is the reference to validate on a small area
        (``out_cells``) rather than the production path.

    ``projection="nested"``
        The historical mode: the NESTED block is reshaped into an image, which
        is exact in index space and free of resampling, and every token maps
        onto exactly one cell of ``level - 4``.  Geometrically wrong away from
        the equator; kept for comparison.

    Parameters
    ----------
    data : array [N, C]
        Reflectances at ``level``, expected in ``[0, 1]`` (Sentinel-2
        ``DN / 10000`` for instance).  NaN marks a missing pixel.
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
    context_px : int
        Side of the window in ``percell`` mode; a multiple of 16.  224 matches
        the DINOv3 training size; smaller is cheaper but gives less context.
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
    if projection not in ("tangent", "nested", "percell"):
        raise ValueError("projection must be 'tangent', 'nested' or 'percell'")

    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d[:, list(bands)]
    ids = _as_numpy(cell_id).astype(np.int64)
    d, ids = _deduplicate(d, ids, duplicates)

    # ---- one tangent plane per output cell --------------------------------
    if projection == "percell":
        if context_px % DINOV3_PATCH:
            raise ValueError(f"context_px must be a multiple of {DINOV3_PATCH}")
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
        )
        # the same field is exposed through the patch_* names so that code
        # written for the tiled modes keeps working unchanged
        res.patch_embedding, res.patch_cell_id, res.patch_level = (
            res.embedding, res.cell_id, tl)
        import healpy as hp
        res.patch_lon, res.patch_lat = hp.pix2ang(
            2 ** tl, res.cell_id, nest=True, lonlat=True)
        return res

    step = DINOV3_PATCH // n                       # window stride, in pixels
    shifts = [(a * step, b * step) for a in range(n) for b in range(n)]

    # ---- build the images ------------------------------------------------
    if projection == "tangent":
        gsd = float(gsd_m) if gsd_m is not None else healpix_gsd_m(level)
        all_centres = np.unique(ids >> (2 * k))
        if tile_px is None:
            H, W = parent_cover_px(all_centres, parent_level, gsd)
        elif np.isscalar(tile_px):
            H = W = int(tile_px)
        else:
            H, W = (int(v) for v in tile_px)
        if H % DINOV3_PATCH or W % DINOV3_PATCH:
            raise ValueError(f"tile_px must be a multiple of {DINOV3_PATCH}")

        tiles0, centre_ids, _valid, coverage, _lon, _lat = tangent_tiles(
            d, ids, level, parent_level, tile_px=(H, W), gsd_m=gsd,
            interpolation=interpolation, fill=fill, duplicates=duplicates,
        )
        keep = coverage >= float(min_coverage)
        centre_ids, coverage = centre_ids[keep], coverage[keep]
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
    )
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
    import healpy as hp

    c_lev = token_level - int(parent_level)
    child = ((centre_ids[:, None] << (2 * c_lev))
             + np.arange(4 ** c_lev, dtype=np.int64)[None])          # [M, Nc]
    lon, lat = hp.pix2ang(2 ** token_level, child.reshape(-1), nest=True, lonlat=True)
    lon, lat = lon.reshape(child.shape), lat.reshape(child.shape)

    clon, clat = hp.pix2ang(2 ** int(parent_level), centre_ids, nest=True, lonlat=True)
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
    "tangent_grid_lonlat",
    "offsets_to_lonlat",
    "sample_healpix",
    "healpix_gsd_m",
    "lonlat_to_offsets",
    "parent_cover_px",
    "nested_to_xy",
    "xy_to_nested",
    "SAT493M_MEAN",
    "SAT493M_STD",
]
