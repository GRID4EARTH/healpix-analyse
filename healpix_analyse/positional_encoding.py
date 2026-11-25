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
import os
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

import healpix_geo

from healpix_analyse._ellipsoid import canonicalize_ellipsoid

try:  # Triton is an optional extra
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _fourier_axis(v, f, out_ptr, row, N, offs, m):
        # sin / cos(pi f v): the argument is reduced modulo 2 in float64
        # (f v is exact for octave frequencies), then evaluated in float32.
        t = f * v
        r = t - 2.0 * tl.floor(t * 0.5)
        a = (r * 3.141592653589793).to(tl.float32)
        tl.store(out_ptr + row * N + offs, tl.sin(a).to(out_ptr.dtype.element_ty), mask=m)
        tl.store(out_ptr + (row + 3) * N + offs, tl.cos(a).to(out_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _posenc_kernel(xyz_ptr, alpha_ptr, f_ptr, out_ptr, N, NF, ROW0, GROW, LROW, level_val,
                       HAS_XYZ: tl.constexpr, HAS_GAUGE: tl.constexpr,
                       HAS_LEVEL: tl.constexpr, BLOCK: tl.constexpr):
        """All the features of BLOCK cells in one pass: reads 3 float64 (+1),
        writes F values per cell."""
        offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        m = offs < N
        x = tl.load(xyz_ptr + offs, mask=m, other=0.0)
        y = tl.load(xyz_ptr + N + offs, mask=m, other=0.0)
        z = tl.load(xyz_ptr + 2 * N + offs, mask=m, other=0.0)
        ty = out_ptr.dtype.element_ty
        if HAS_XYZ:
            tl.store(out_ptr + offs, x.to(ty), mask=m)
            tl.store(out_ptr + N + offs, y.to(ty), mask=m)
            tl.store(out_ptr + 2 * N + offs, z.to(ty), mask=m)
        for k in range(0, NF):
            f = tl.load(f_ptr + k)
            row = ROW0 + 6 * k
            _fourier_axis(x, f, out_ptr, row, N, offs, m)
            _fourier_axis(y, f, out_ptr, row + 1, N, offs, m)
            _fourier_axis(z, f, out_ptr, row + 2, N, offs, m)
        if HAS_GAUGE:
            al = tl.load(alpha_ptr + offs, mask=m, other=0.0)
            tl.store(out_ptr + GROW * N + offs, tl.cos(al).to(ty), mask=m)
            tl.store(out_ptr + (GROW + 1) * N + offs, tl.sin(al).to(ty), mask=m)
        if HAS_LEVEL:
            tl.store(out_ptr + LROW * N + offs,
                     (tl.zeros([BLOCK], tl.float32) + level_val).to(ty), mask=m)



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


def _gauge_angle(th, ph, gauge_type, ref_direction):
    """Angle from e_theta to the first gauge axis of ``HealPixConv``.

    ``HealPixConv`` rotates its frame about the normal n by ``alpha_base``,
    and ``n x e_theta = e_phi``, so the first axis is
    ``cos(alpha) e_theta + sin(alpha) e_phi`` with ``alpha = alpha_base``:
    the closed forms of ``convol._build_rotation_matrices`` (checked against
    its rotation matrices in the tests), without building them.
    """
    if gauge_type == "phi":
        return np.zeros_like(th)
    if gauge_type == "cosmo":
        return np.where(th > np.pi / 2, ph, -ph)
    ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)

    def tangent(r):
        r = np.asarray(r, dtype=np.float64).reshape(3)
        r = r / max(np.linalg.norm(r), 1e-12)
        r_eth = r[0] * ct * cp + r[1] * ct * sp - r[2] * st
        r_eph = -r[0] * sp + r[1] * cp
        return r_eth, r_eph

    if gauge_type == "projected_ref":
        eth, eph = tangent([1.0, 0.0, 0.0] if ref_direction is None else ref_direction)
        return np.arctan2(eph, eth)
    if ref_direction is None or np.asarray(ref_direction).size != 6:
        raise ValueError("gauge_type='two_ref' needs ref_direction of shape (2, 3).")
    ref = np.asarray(ref_direction, dtype=np.float64).reshape(2, 3)
    e1, p1 = tangent(ref[0])
    e2, p2 = tangent(ref[1])
    return np.arctan2(e1 * p2 + p1 * e2, e1 * e2 - p1 * p2)


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
    backend: str = "auto",
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
    backend : {"auto", "triton", "torch"}, default "auto"
        ``"auto"``: on CUDA (float32 / 16-bit), one Triton kernel writes all
        the features (one read of the float64 geometry, one write of the
        ``[F, N]`` result); elsewhere PyTorch element-wise operations in
        float64.  Both reduce the Fourier arguments modulo 2 in float64.

    Returns
    -------
    torch.Tensor, shape [F, N]
        Row k of the encoding is cell ``cell_ids[k]``.  The geometry (unit
        vectors and gauge angle, float64 on the CPU) is cached per domain,
        keyed by a hash of ``cell_ids``, with one copy per device it was
        used on; the features themselves are computed on ``device``.
    """
    if gauge_type is not None and gauge_type not in GAUGE_TYPES:
        raise ValueError(f"gauge_type must be one of {GAUGE_TYPES} or None.")
    if not 0 <= int(level) <= _MAX_LEVEL:
        raise ValueError(f"level must be in [0, {_MAX_LEVEL}].")
    ellipsoid = canonicalize_ellipsoid(ellipsoid)
    ids = _cell_ids_array(level, cell_ids)
    geom = _geometry(level, ids, ellipsoid, gauge_type, ref_direction)
    freqs = _frequencies(level, n_frequencies, frequencies)

    dev = torch.device(device) if device is not None else torch.device("cpu")
    interp = os.environ.get("TRITON_INTERPRET") == "1" and backend == "triton"
    use_triton = (_HAS_TRITON and (dev.type == "cuda" or interp)
                  and backend in ("auto", "triton")
                  and dtype in (torch.float32, torch.float16, torch.bfloat16))
    if backend == "triton" and not use_triton:
        raise RuntimeError("backend='triton' needs triton, a CUDA device and a 16/32-bit dtype.")
    xyz = _on_device(geom, "xyz", dev)                          # [3, N] float64
    N = xyz.shape[1]
    F = (3 if include_xyz else 0) + 6 * len(freqs) + (2 if gauge_type else 0) + int(include_level)
    if use_triton:
        out = torch.empty(F, N, dtype=dtype, device=dev)
        if F == 0 or N == 0:
            return out
        fr = torch.tensor(freqs or [0.0], dtype=torch.float64, device=dev)
        alpha = _on_device(geom, "alpha", dev) if gauge_type is not None else xyz
        row0 = 3 if include_xyz else 0
        grow = row0 + 6 * len(freqs)
        lrow = grow + (2 if gauge_type else 0)
        BLOCK = 512
        _posenc_kernel[(triton.cdiv(N, BLOCK),)](
            xyz, alpha, fr, out, N, len(freqs), row0, grow, lrow, float(level) / _MAX_LEVEL,
            HAS_XYZ=include_xyz, HAS_GAUGE=gauge_type is not None,
            HAS_LEVEL=include_level, BLOCK=BLOCK, num_warps=4)
        return out
    feats = [xyz] if include_xyz else []
    if freqs:
        f = torch.tensor(freqs, dtype=torch.float64, device=xyz.device)
        t = f.view(-1, 1, 1) * xyz.unsqueeze(0)                  # [nf, 3, N]
        arg = (t - 2.0 * torch.floor(0.5 * t)) * math.pi         # reduced modulo 2 pi
        feats.append(torch.cat([arg.sin(), arg.cos()], dim=1).reshape(-1, N))
    if gauge_type is not None:
        a = _on_device(geom, "alpha", dev)
        feats.append(torch.stack([a.cos(), a.sin()]))
    if include_level:
        feats.append(torch.full((1, N), level / _MAX_LEVEL, dtype=torch.float64, device=dev))
    if not feats:
        return torch.empty(0, N, dtype=dtype, device=dev)
    return torch.cat(feats, dim=0).to(dtype)


def _on_device(geom, name, dev):
    """The cached geometry tensor on ``dev`` (copied once per device)."""
    if dev.type == "cpu":
        return geom[name]
    per = geom.setdefault("device", {})
    key = (name, str(dev))
    if key not in per:
        per[key] = geom[name].to(dev)
    return per[key]


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
                 device=None, dtype: torch.dtype = torch.float32, backend: str = "auto"):
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
            include_level=include_level, ellipsoid=self.ellipsoid, device=device, dtype=dtype,
            backend=backend)
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
