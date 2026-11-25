"""Positional encodings of HEALPix cells for AI models (issue #109, section 6).

Fourier features of the unit vector of each cell centre, optionally with the
gauge angle of :class:`healpix_analyse.HealPixConv` and the level, computed
once per domain and cached.

The cell centres come from ``healpix_geo`` on the chosen ellipsoid (WGS84 by
default, as everywhere in the package).  On an ellipsoid ``healpix_geo``
returns the geodetic latitude, so the unit vector is the ellipsoid normal at
the cell centre::

    (x, y, z) = (cos(lat) cos(lon), cos(lat) sin(lon), sin(lat))

On ``"sphere"`` it is the direction of the cell centre.

Features, in this order (``feature_names`` lists them):

* ``x, y, z`` (``include_xyz=True``);
* for each frequency ``f``: ``sin(pi f x), sin(pi f y), sin(pi f z),
  cos(pi f x), cos(pi f y), cos(pi f z)``;
* ``cos(alpha), sin(alpha)`` with ``gauge_type`` set: ``alpha`` is the angle,
  in the tangent plane, from the local north ``e_theta`` to the first axis of
  the gauge frame of ``HealPixConv(gauge_type=...)`` (gauge 0); it is 0
  everywhere for ``"phi"``;
* ``level / 29`` (``include_level=True``), a constant channel that tells the
  levels of a multi-level model apart.

The default frequencies are ``1, 2, 4, ..., 2**level`` (``level + 1`` of them,
at most 16): the shortest wavelength, ``2 / 2**level`` in unit-vector units,
is about two cells.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import healpix_geo

from healpix_analyse._ellipsoid import canonicalize_ellipsoid

__all__ = [
    "HealPixPositionalEncoding",
    "healpix_positional_encoding",
    "positional_encoding_cache_info",
    "clear_positional_encoding_cache",
]

GAUGE_TYPES = ("phi", "cosmo", "projected_ref", "two_ref")
_MAX_LEVEL = 29
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 8
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cell_ids_array(level: int, cell_ids) -> np.ndarray:
    npix = 12 * 4 ** int(level)
    if cell_ids is None:
        return np.arange(npix, dtype=np.int64)
    if isinstance(cell_ids, torch.Tensor):
        cell_ids = cell_ids.detach().cpu().numpy()
    ids = np.asarray(cell_ids).astype(np.int64, copy=False).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= npix):
        raise ValueError(f"cell_ids must be in [0, {npix}) at level {level}.")
    return ids


def _domain_key(ids: np.ndarray) -> str:
    """Hash of the cell ids (order included)."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def _gauge_angle(th, ph, gauge_type, ref_direction, chunk=1 << 20):
    """Angle from e_theta to the first gauge axis, in [-pi, pi]."""
    from healpix_analyse.convol import _build_rotation_matrices

    out = np.empty(th.shape[0], dtype=np.float64)
    for s in range(0, th.shape[0], chunk):
        t = th[s:s + chunk]
        p = ph[s:s + chunk]
        R = _build_rotation_matrices(t, p, 1, gauge_type, "cpu", torch.float64,
                                     ref_direction=ref_direction)[:, 0]   # [k, 3, 3]
        v = R[:, :, 0].numpy()                     # image of the north-pole x axis
        ct, st, cp, sp = np.cos(t), np.sin(t), np.cos(p), np.sin(p)
        e_th = np.stack([ct * cp, ct * sp, -st], axis=1)
        e_ph = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
        out[s:s + chunk] = np.arctan2((v * e_ph).sum(1), (v * e_th).sum(1))
    return out


def _geometry(level, ids, ellipsoid, gauge_type, ref_direction):
    """Unit vectors [3, N] (float64) and gauge angle [N] or None, cached."""
    ref = None if ref_direction is None else tuple(np.asarray(ref_direction, float).ravel())
    key = (_domain_key(ids), int(level), str(ellipsoid), gauge_type, ref)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        _CACHE_STATS["hits"] += 1
        return hit
    _CACHE_STATS["misses"] += 1
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        ids.astype(np.uint64), int(level), ellipsoid=ellipsoid)
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    cl = np.cos(lat)
    xyz = np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)])
    alpha = None
    if gauge_type is not None:
        alpha = _gauge_angle(np.pi / 2 - lat, lon, gauge_type, ref_direction)
    entry = {"xyz": torch.from_numpy(xyz), "alpha": None if alpha is None
             else torch.from_numpy(alpha)}
    _CACHE[key] = entry
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return entry


def positional_encoding_cache_info() -> dict:
    """Hits, misses and number of domains of the geometry cache (in memory,
    per process, at most 8 domains, least recently used dropped first)."""
    return {"hits": _CACHE_STATS["hits"], "misses": _CACHE_STATS["misses"],
            "size": len(_CACHE), "maxsize": _CACHE_MAX}


def clear_positional_encoding_cache() -> None:
    """Empty the geometry cache of the positional encodings."""
    _CACHE.clear()
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0


def _frequencies(level, n_frequencies, frequencies):
    if frequencies is not None:
        f = [float(v) for v in np.asarray(frequencies, dtype=float).ravel()]
        if not f:
            raise ValueError("frequencies must not be empty.")
        return f
    n = min(int(level) + 1, 16) if n_frequencies is None else int(n_frequencies)
    if n < 0:
        raise ValueError("n_frequencies must be >= 0.")
    return [float(2 ** k) for k in range(n)]


def _feature_names(freqs, include_xyz, gauge_type, include_level):
    names = ["x", "y", "z"] if include_xyz else []
    for f in freqs:
        fs = f"{f:g}"
        names += [f"sin({fs}*pi*{a})" for a in "xyz"]
        names += [f"cos({fs}*pi*{a})" for a in "xyz"]
    if gauge_type is not None:
        names += ["cos(alpha)", "sin(alpha)"]
    if include_level:
        names.append("level/29")
    return names


def healpix_positional_encoding(
    level: int,
    cell_ids=None,
    *,
    n_frequencies: Optional[int] = None,
    frequencies: Optional[Sequence[float]] = None,
    include_xyz: bool = True,
    gauge_type: Optional[str] = None,
    ref_direction=None,
    include_level: bool = False,
    ellipsoid: str = "WGS84",
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Positional encoding of HEALPix cells, shape ``[F, N]``.

    Parameters
    ----------
    level : int
        HEALPix level (depth) of the cells.
    cell_ids : array-like of int, optional
        NESTED ids of the cells, in the order of the data (a regional domain
        need not be sorted).  None = the full sphere, ``arange(12 * 4**level)``.
    n_frequencies : int, optional
        Number of octave frequencies ``1, 2, ..., 2**(n-1)``; default
        ``min(level + 1, 16)``.  0 gives no Fourier feature.
    frequencies : sequence of float, optional
        Explicit frequencies (overrides ``n_frequencies``).
    include_xyz : bool, default True
        Add the unit vector itself.
    gauge_type : {"phi", "cosmo", "projected_ref", "two_ref"} or None
        Add ``cos(alpha), sin(alpha)`` of the gauge of
        ``HealPixConv(gauge_type=...)``; ``ref_direction`` as there.
    include_level : bool, default False
        Add the constant channel ``level / 29``.
    ellipsoid : str, default "WGS84"
        Ellipsoid of the cell centres (``healpix_geo`` name).
    device, dtype
        Of the result (float32 by default).

    Returns
    -------
    torch.Tensor, shape [F, N]
        Row k of the encoding is cell ``cell_ids[k]``.  The geometry (unit
        vectors and gauge angle, float64 on the CPU) is cached per domain,
        keyed by a hash of ``cell_ids``; the features themselves are computed
        on ``device`` (a few element-wise operations).
    """
    if gauge_type is not None and gauge_type not in GAUGE_TYPES:
        raise ValueError(f"gauge_type must be one of {GAUGE_TYPES} or None.")
    if not 0 <= int(level) <= _MAX_LEVEL:
        raise ValueError(f"level must be in [0, {_MAX_LEVEL}].")
    ellipsoid = canonicalize_ellipsoid(ellipsoid)
    ids = _cell_ids_array(level, cell_ids)
    geom = _geometry(level, ids, ellipsoid, gauge_type, ref_direction)
    freqs = _frequencies(level, n_frequencies, frequencies)

    xyz = geom["xyz"].to(device=device)                       # [3, N] float64
    feats = [xyz] if include_xyz else []
    if freqs:
        f = torch.tensor(freqs, dtype=torch.float64, device=xyz.device) * math.pi
        arg = f.view(-1, 1, 1) * xyz.unsqueeze(0)               # [nf, 3, N]
        feats.append(torch.cat([arg.sin(), arg.cos()], dim=1).reshape(-1, xyz.shape[1]))
    if gauge_type is not None:
        a = geom["alpha"].to(device=xyz.device)
        feats.append(torch.stack([a.cos(), a.sin()]))
    if include_level:
        feats.append(torch.full((1, xyz.shape[1]), level / _MAX_LEVEL,
                                dtype=torch.float64, device=xyz.device))
    if not feats:
        return torch.empty(0, xyz.shape[1], dtype=dtype, device=xyz.device)
    return torch.cat(feats, dim=0).to(dtype)


class HealPixPositionalEncoding(nn.Module):
    """Cached positional encoding of a HEALPix domain, as a module.

    The ``[F, N]`` encoding is computed at construction (see
    :func:`healpix_positional_encoding` for the parameters and the
    features) and stored as the non-persistent buffer ``encoding``: it
    follows ``.to()`` / ``.cuda()`` / ``.half()`` and is not saved in the
    ``state_dict`` (it is rebuilt from the parameters).

    ``forward(x=None)`` returns the encoding ``[F, N]``; with ``x`` of shape
    ``[B, C, N]`` it returns ``torch.cat([x, encoding], dim=1)`` of shape
    ``[B, C + F, N]`` (the encoding cast to the dtype of ``x``).

    Attributes
    ----------
    num_features : int
        F.
    feature_names : list of str
    cell_ids : numpy.ndarray
    """

    def __init__(self, level: int, cell_ids=None, *, n_frequencies: Optional[int] = None,
                 frequencies: Optional[Sequence[float]] = None, include_xyz: bool = True,
                 gauge_type: Optional[str] = None, ref_direction=None,
                 include_level: bool = False, ellipsoid: str = "WGS84",
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.level = int(level)
        self.cell_ids = _cell_ids_array(level, cell_ids)
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)
        self.gauge_type = gauge_type
        self.include_level = bool(include_level)
        self.include_xyz = bool(include_xyz)
        self.frequencies = _frequencies(level, n_frequencies, frequencies)
        enc = healpix_positional_encoding(
            level, self.cell_ids, frequencies=self.frequencies or None,
            n_frequencies=0 if not self.frequencies else None,
            include_xyz=include_xyz, gauge_type=gauge_type, ref_direction=ref_direction,
            include_level=include_level, ellipsoid=self.ellipsoid, device=device, dtype=dtype)
        self.register_buffer("encoding", enc, persistent=False)
        self.feature_names = _feature_names(self.frequencies, include_xyz, gauge_type,
                                            include_level)
        self.num_features = enc.shape[0]

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"level={self.level}, n_cells={self.cell_ids.size}, "
                f"num_features={self.num_features}, ellipsoid={self.ellipsoid!r}")

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            return self.encoding
        if x.dim() != 3 or x.shape[-1] != self.encoding.shape[1]:
            raise ValueError(
                f"x must have shape [B, C, {self.encoding.shape[1]}]; got {tuple(x.shape)}.")
        enc = self.encoding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([x, enc], dim=1)
