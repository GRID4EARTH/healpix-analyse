"""
patches.py
==========
Patch extraction on HEALPix cells: the tokens of ViT, MAE, JEPA and other
token-based models, and its adjoint ``fold``.

:class:`HealPixPatches` cuts a field at level ``L`` into square patches of
``N x N = 2**p x 2**p`` pixels, centred on the cells of level ``L - p``
(``centres="coarse"``, a tiling aligned with NESTED) or on every cell
(``centres="all"``, overlapping patches), or on the cells of any level in
between (``centres=<level>``, e.g. images larger than their cell for a
pre-trained ViT).  Two modes:

``mode="gauge"`` (default)
    An ``N x N`` grid in the local tangent plane of each centre, oriented by
    the gauge exactly like a :class:`~healpix_analyse.convol.HealPixConv`
    kernel built with ``ellipsoid="sphere"`` (same rotation matrices, same
    bilinear interpolation) with the spacing of one cell of level ``L``.  A shared patch embedding then sees
    the same orientation everywhere on the sphere.
``mode="nested"``
    The ``4**p`` descendants of each level ``L - p`` cell, as a
    ``2**p x 2**p`` image in the layout of
    :func:`~healpix_analyse.dino.nested_to_tiles`.  Face-oriented: the cheap
    option, a reshape (or a single gather) of the data.

The compute backends are in ``healpix_analyse._patches_backend``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import healpix_geo
from healpix_analyse.convol import (
    _build_rotation_matrices,
    _get_interp_weights,
    _significant,
)
from healpix_analyse._ellipsoid import authalic_scale, canonicalize_ellipsoid
from healpix_analyse.dino import xy_to_nested
from healpix_analyse._patches_backend import (
    BACKENDS,
    FOLD_NORMS,
    PatchExtractFunction,
    PatchFoldFunction,
    PatchGeometry,
    _Options,
    use_triton,
)

__all__ = ["HealPixPatches"]

_GAUGE_TYPES = ("phi", "cosmo", "projected_ref", "two_ref")


def _patch_grid(N: int, nside: int) -> np.ndarray:
    """``N x N`` grid of unit vectors around the North Pole, ``[N*N, 3]``.

    Same construction as ``convol._local_kernel_grid`` (azimuthal equidistant
    offsets of one cell width), with the offsets ``i - (N - 1) / 2`` so that
    an even ``N`` is centred on the cell centre.  Pixel ``p = a*N + b`` is the
    point ``(x, y) = (u[b], u[a])``.
    """
    u = np.arange(N, dtype=np.float64) - (N - 1) / 2.0
    xx, yy = np.meshgrid(u, u)
    alpha_pix = np.sqrt(4 * np.pi / (12 * nside**2))
    dtheta = np.sqrt(xx**2 + yy**2).ravel() * alpha_pix
    dphi = np.arctan2(yy, xx).ravel()
    return np.stack([np.sin(dtheta) * np.cos(dphi),
                     np.sin(dtheta) * np.sin(dphi),
                     np.cos(dtheta)], axis=-1)


def _stretch_to_ellipsoid(rot, th, ph, k, G, P):
    """Map a grid that is isotropic on the ground onto the authalic sphere.

    ``rot`` [M*G*P, 3] are the grid points rotated to the centres (colatitude
    ``th``, longitude ``ph`` on the authalic sphere); ``k`` [M] is the scale
    of :func:`~healpix_analyse._ellipsoid.authalic_scale`.  Every offset from
    the centre (azimuthal equidistant) is multiplied by ``k`` along the
    meridian and divided by ``k`` along the parallel, so that the patch keeps
    its shape on the ellipsoid.  Modified in place and returned.
    """
    M = th.shape[0]
    v = rot.view(M, G * P, 3)
    th_t = torch.as_tensor(th, dtype=v.dtype)[:, None]
    ph_t = torch.as_tensor(ph, dtype=v.dtype)[:, None]
    ct, st, cp, sp = th_t.cos(), th_t.sin(), ph_t.cos(), ph_t.sin()
    n = torch.stack([st * cp, st * sp, ct], -1)                    # [M, 1, 3]
    e_s = torch.stack([ct * cp, ct * sp, -st], -1)                 # south
    e_e = torch.stack([-sp, cp, torch.zeros_like(st)], -1)         # east
    t_s = (v * e_s).sum(-1)
    t_e = (v * e_e).sum(-1)
    up = (v * n).sum(-1)
    t = torch.hypot(t_s, t_e)
    d = torch.atan2(t, up)
    one = torch.ones_like(t)
    scale = torch.where(t > 0, d / torch.where(t > 0, t, one), one)
    kk = torch.as_tensor(k, dtype=v.dtype)[:, None]
    o_s, o_e = t_s * scale * kk, t_e * scale / kk
    r = torch.hypot(o_s, o_e)
    sinc = torch.where(r > 0, r.sin() / torch.where(r > 0, r, one), one)
    v.copy_(r.cos()[..., None] * n
            + sinc[..., None] * (o_s[..., None] * e_s + o_e[..., None] * e_e))
    return rot


def _lonlat_to_vec(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _ref_direction(gauge_type, singularity_lonlat, ref_direction):
    """Reference vector(s) of the ``projected_ref`` / ``two_ref`` gauges, as in
    :class:`~healpix_analyse.convol.HealPixConv`."""
    if gauge_type not in ("projected_ref", "two_ref"):
        if singularity_lonlat is not None:
            raise ValueError("singularity_lonlat is only valid for "
                             "gauge_type='projected_ref' or 'two_ref'.")
        return None
    if singularity_lonlat is not None:
        if gauge_type == "projected_ref":
            rd = _lonlat_to_vec(*singularity_lonlat)
        else:
            if len(singularity_lonlat) != 2:
                raise ValueError("For gauge_type='two_ref', singularity_lonlat must be "
                                 "two (lon, lat) pairs.")
            rd = np.stack([_lonlat_to_vec(*ll) for ll in singularity_lonlat])
    elif ref_direction is not None:
        rd = np.asarray(ref_direction, dtype=np.float64)
        if gauge_type == "two_ref" and rd.shape != (2, 3):
            raise ValueError("For gauge_type='two_ref', ref_direction must have shape (2, 3).")
        if gauge_type == "projected_ref":
            rd = rd.ravel()
    else:
        rd = (np.array([1.0, 0.0, 0.0]) if gauge_type == "projected_ref"
              else np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    return rd / np.linalg.norm(rd, axis=-1, keepdims=True)


def _bind(cells, ids_np, full, is_sorted=False):
    """Positions of the NESTED ``cells`` in the domain ``ids_np`` (input order)
    and their presence.  ``is_sorted``: ``ids_np`` is strictly increasing, no
    sort needed (the batches of centres of a large domain then cost O(K) each
    instead of O(K log K))."""
    if full:
        return cells, np.ones(cells.shape, dtype=bool)
    if is_sorted:
        srt = ids_np
        pos = np.searchsorted(srt, cells)
        pos_c = np.minimum(pos, srt.size - 1)
        present = srt[pos_c] == cells
        return np.where(present, pos_c, 0), present
    order = np.argsort(ids_np, kind="stable")
    srt = ids_np[order]
    pos = np.searchsorted(srt, cells)
    pos_c = np.minimum(pos, srt.size - 1)
    present = srt[pos_c] == cells
    return np.where(present, order[pos_c], 0), present


def _nested_tiles(x, perm, M):
    """[B, C, M*Q] -> [B, M, C, Q], pixels of each block permuted: one gather."""
    B, C, _ = x.shape
    return x.view(B, C, M, perm.numel()).permute(0, 2, 1, 3).index_select(3, perm)


def _nested_untiles(y, iperm):
    """Inverse of :func:`_nested_tiles`: [B, M, C, Q] -> [B, C, M*Q], one gather."""
    B, M, C, Q = y.shape
    return y.permute(0, 2, 1, 3).index_select(3, iperm).reshape(B, C, M * Q)


class _NestedTiles(torch.autograd.Function):
    """Contiguous NESTED fast path.  Both directions are permutations, so the
    backward of each is the other one: a gather, not the atomic ``index_add_``
    of the backward of ``index_select``."""

    @staticmethod
    def forward(ctx, x, perm, iperm, inverse):
        ctx.save_for_backward(perm, iperm)
        ctx.inverse = inverse
        if inverse:
            return _nested_untiles(x, iperm)
        return _nested_tiles(x, perm, x.shape[2] // perm.numel())

    @staticmethod
    def backward(ctx, g):
        perm, iperm = ctx.saved_tensors
        return _NestedTiles.apply(g.contiguous(), perm, iperm, not ctx.inverse), None, None, None


class HealPixPatches(nn.Module):
    """
    Square patches (tokens) of a HEALPix field, gauge-oriented or NESTED.

    Parameters
    ----------
    level : int
        Level ``L`` of the data (``nside = 2**L``).
    patch_size : int, default 4
        Side ``N = 2**p`` of the patches, a power of 2 with ``p <= L``.
    cell_ids : array-like or None
        NESTED cell ids of a regional domain at level ``L``, in the order of
        the data (any order).  None = the full sphere in NESTED order.
    centres : {"coarse", "all"} or int, default "coarse"
        ``"coarse"``: one patch per cell of level ``L - p`` (the parents of
        the cells of the domain), a tiling aligned with NESTED.  ``"all"``:
        one overlapping patch centred on every cell of the domain, in the
        order of ``cell_ids`` (``mode="gauge"`` only).  An integer ``c``
        with ``L - p <= c <= L``: one patch per cell of level ``c`` holding
        cells of the domain, overlapping as soon as ``c > L - p``
        (``mode="gauge"`` only unless ``c = L - p``): images larger than
        their cell, as a pre-trained ViT wants.  ``"coarse"`` is ``L - p``,
        ``"all"`` is ``L`` in the order of ``cell_ids``.
    centre_ids : array-like or None
        NESTED ids of the patch centres at the centre level, instead of the
        cells derived from the domain: a subset, to build the geometry by
        batches of centres, or centres outside the domain (their patches
        read what the domain holds; ``coverage`` says how much).  Unique.
    ellipsoid : str or dict, default "sphere"
        Reference ellipsoid of the HEALPix grid, e.g. ``"WGS84"`` for the
        GRID4EARTH Sentinel-2 stores.  The cells read and the patch centres
        do not depend on it (a HEALPix grid on an ellipsoid is the grid of
        its authalic sphere, so ids are ids); it sets the shape of the patch
        on the ground: with an ellipsoid the grid is square in metres on the
        ellipsoid instead of on the authalic sphere, a correction below
        0.34 % on WGS84.  :meth:`centre_lonlat` returns geodetic coordinates
        on it.
    mode : {"gauge", "nested"}, default "gauge"
        ``"gauge"``: an ``N x N`` grid in the tangent plane of each centre,
        rotated by the gauge and sampled by bilinear interpolation, exactly
        like a :class:`~healpix_analyse.convol.HealPixConv` kernel with
        ``ellipsoid="sphere"``, with the spacing of one cell of level ``L``.
        With ``gauge_type="phi"``, gauge 0, row ``a`` of the patch runs
        towards the east and column ``b`` towards the south: the patch is
        the **transpose** of a north-up map (a mirror image, not a
        rotation); ``y.transpose(-1, -2)`` gives north up, east right, which
        is what a network pre-trained on map imagery expects.  ``"nested"``: the ``4**p``
        descendants of each centre as a ``2**p x 2**p`` image (row ``N-1-y``,
        column ``x`` in the face coordinates of HEALPix, the layout of
        :func:`~healpix_analyse.dino.nested_to_tiles`).
    n_gauges : int, default 1
        Number of gauge orientations G (``mode="gauge"``); gauge ``g`` rotates
        the grid by ``g * pi / G``: with ``G = 4``, gauge ``g + 2`` is
        ``torch.rot90(patch_g, 1, (-2, -1))``.  With ``gauge_type="cosmo"`` the
        rotation changes sign at southern centres (``k=3`` instead of 1).
        ``mode="nested"`` has ``G = 1``.
    gauge_type : {"phi", "cosmo", "projected_ref", "two_ref"}, default "phi"
        Gauge convention, as in :class:`~healpix_analyse.convol.HealPixConv`.
    singularity_lonlat, ref_direction
        Position of the gauge singularities for ``projected_ref`` and
        ``two_ref``, as in :class:`~healpix_analyse.convol.HealPixConv`.
    fold_norm : {"weight", "count", "none"}, default "weight"
        Default normalisation of :meth:`fold`.  ``"weight"`` divides each
        cell by the sum of the weights with which the patches read it, so
        that ``fold(forward(x))`` keeps a constant field; ``"count"`` by the
        number of patch samples that read it (the overlap count); ``"none"``
        is the plain adjoint.  The three agree for ``mode="nested"`` on a
        complete domain.
    backend : {"auto", "triton", "torch"}, default "auto"
        ``"auto"`` runs the Triton kernels on CUDA for float32, float16 and
        bfloat16 when the optional ``triton`` package is installed, PyTorch
        otherwise.  ``mode="nested"`` on the full sphere or a complete,
        sorted domain uses neither: it is a view plus one gather.
    device : torch device or None
        Device of the geometry (CUDA when available by default).
    dtype : torch.dtype, default torch.float32
        Precision of the geometry: float64 builds and keeps it in float64,
        any other dtype in float32.

    Notes
    -----
    Output layout: ``[B, M, G, C, N, N]`` for ``M`` patches.  The tensor is
    contiguous, so ``y.flatten(2)`` (one token of ``G*C*N*N`` values per
    centre) or ``y.flatten(3)`` (``G`` tokens per centre) are free views.

    In ``mode="gauge"``, a grid point whose 4 bilinear neighbours are not all
    in the domain uses the present ones, renormalised; a point with none is
    0.  ``coverage`` holds, per pixel, the fraction of the bilinear weight
    inside the domain (1 everywhere on the full sphere).  In
    ``mode="nested"``, the descendants absent from the domain are 0 and
    their coverage is 0.

    The grid is placed on the sphere of the HEALPix grid (the authalic sphere
    for an ellipsoid): a patch of size 1 returns the cells themselves.

    The backward pass of :meth:`forward` is :meth:`fold` without
    normalisation, and the backward of :meth:`fold` is :meth:`forward`; both
    run on the selected backend and are differentiable again.  On CUDA the
    Triton fold is a gather on the transposed map, so it is deterministic.
    """

    backend = "auto"

    def __init__(
        self,
        level: int,
        patch_size: int = 4,
        cell_ids=None,
        centres="coarse",
        mode: str = "gauge",
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat=None,
        ref_direction=None,
        fold_norm: str = "weight",
        backend: str = "auto",
        device=None,
        dtype: torch.dtype = torch.float32,
        centre_ids=None,
        ellipsoid="sphere",
    ):
        super().__init__()
        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0.")
        self.level = int(level)
        self.nside = 2**self.level
        N = int(patch_size)
        if N < 1 or N & (N - 1):
            raise ValueError("patch_size must be a power of 2.")
        self.patch_size = N
        self.p = N.bit_length() - 1
        if self.p > self.level:
            raise ValueError(f"patch_size 2**{self.p} needs level >= {self.p}.")
        if isinstance(centres, str):
            if centres not in ("coarse", "all"):
                raise ValueError("centres must be 'coarse', 'all' or a level.")
            centre_level = self.level - self.p if centres == "coarse" else self.level
        else:
            if isinstance(centres, bool) or int(centres) != centres:
                raise ValueError("centres must be 'coarse', 'all' or a level.")
            centre_level = int(centres)
            if not self.level - self.p <= centre_level <= self.level:
                raise ValueError(f"centres={centre_level} must lie in "
                                 f"[{self.level - self.p}, {self.level}] for "
                                 f"patch_size={N} at level {self.level}.")
        if mode not in ("gauge", "nested"):
            raise ValueError("mode must be 'gauge' or 'nested'.")
        if mode == "nested" and centre_level != self.level - self.p:
            raise ValueError("mode='nested' tiles the domain: use centres='coarse'.")
        if gauge_type not in _GAUGE_TYPES:
            raise ValueError(f"gauge_type must be one of {_GAUGE_TYPES}.")
        if fold_norm not in FOLD_NORMS:
            raise ValueError(f"fold_norm must be one of {FOLD_NORMS}.")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
        self.centres = centres
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)
        self.mode = mode
        self.G = int(max(1, n_gauges)) if mode == "gauge" else 1
        self.gauge_type = gauge_type
        self.fold_norm = fold_norm
        self.backend = backend
        self.ref_direction = _ref_direction(gauge_type, singularity_lonlat, ref_direction)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        gdtype = torch.float64 if dtype == torch.float64 else torch.float32

        # ---- domain and centres ----
        self.partial = cell_ids is not None
        npix = 12 * self.nside**2
        if self.partial:
            ids_np = np.asarray(cell_ids, dtype=np.int64).ravel()
            if ids_np.size == 0:
                raise ValueError("cell_ids must not be empty.")
            if np.any(ids_np < 0) or np.any(ids_np >= npix):
                raise ValueError(f"cell_ids contain identifiers outside level={self.level}.")
            # strictly increasing ids (the usual NESTED order) are unique and
            # need no sort: an O(K) check instead of an O(K log K) unique
            self._sorted = bool(ids_np.size < 2 or np.all(ids_np[1:] > ids_np[:-1]))
            if not self._sorted and np.unique(ids_np).size != ids_np.size:
                raise ValueError("cell_ids must contain unique identifiers.")
        else:
            ids_np = np.arange(npix, dtype=np.int64)
            self._sorted = True
        self.K = ids_np.size
        Q = 4**self.p
        self.centre_level = centre_level
        explicit_centres = centre_ids is not None
        if explicit_centres:
            centre_ids = np.asarray(centre_ids, dtype=np.int64).ravel()
            if centre_ids.size == 0:
                raise ValueError("centre_ids must not be empty.")
            if np.any(centre_ids < 0) or np.any(centre_ids >= 12 * 4**centre_level):
                raise ValueError(f"centre_ids contain identifiers outside level={centre_level}.")
            if np.unique(centre_ids).size != centre_ids.size:
                raise ValueError("centre_ids must contain unique identifiers.")
        elif centres == "all":
            centre_ids = ids_np.copy()
        else:
            centre_ids = (np.unique(ids_np >> (2 * (self.level - centre_level)))
                          if self.partial
                          else np.arange(12 * 4**centre_level, dtype=np.int64))
        self.M = centre_ids.size
        self.P = N * N

        # ---- geometry [J, M*G*P] ----
        if mode == "nested":
            a, b = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")   # row, col
            perm = xy_to_nested(b.ravel(), (N - 1) - a.ravel())             # [P] child index
            cells = (centre_ids[:, None] * Q + perm[None, :]).reshape(1, -1)
            pos, present = _bind(cells, ids_np, not self.partial, self._sorted)
            w = present.astype(np.float64)
            coverage = w[0]
            inv = np.empty_like(perm)
            inv[perm] = np.arange(perm.size)
            self.register_buffer("_perm", torch.as_tensor(perm, device=device), persistent=False)
            self.register_buffer("_iperm", torch.as_tensor(inv, device=device), persistent=False)
            # contiguous fast path: the full sphere with all its centres, or a
            # sorted domain made of complete level L - p cells (explicit
            # centre_ids are checked by the second test)
            self.contiguous = (not self.partial and not explicit_centres) or (
                self.K == self.M * Q
                and np.array_equal(ids_np, (centre_ids[:, None] * Q + np.arange(Q)).ravel()))
            idx_t = torch.as_tensor(pos, dtype=torch.int32)
            w_t = torch.as_tensor(w, dtype=gdtype)
        else:
            self.contiguous = False
            lon, lat = healpix_geo.nested.healpix_to_lonlat(
                centre_ids, self.centre_level, ellipsoid="sphere")
            k_ell = None
            if self.ellipsoid != "sphere" and N > 1:
                _, lat_g = healpix_geo.nested.healpix_to_lonlat(
                    centre_ids, self.centre_level, ellipsoid=self.ellipsoid)
                k_ell = authalic_scale(np.asarray(lat_g, dtype=np.float64), self.ellipsoid)
            th = np.deg2rad(90.0 - np.asarray(lat, dtype=np.float64))
            ph = np.deg2rad(np.asarray(lon, dtype=np.float64))
            # built in float64 whatever the dtype, as HealPixConv: which cells a
            # grid point reads must not depend on rounding
            R = _build_rotation_matrices(th, ph, self.G, gauge_type, "cpu", torch.float64,
                                         ref_direction=self.ref_direction)   # [M, G, 3, 3]
            vec = torch.as_tensor(_patch_grid(N, self.nside), dtype=torch.float64)
            rot = torch.einsum("mgij,pj->mgpi", R, vec).reshape(-1, 3)
            del R
            if k_ell is not None:
                rot = _stretch_to_ellipsoid(rot, th, ph, k_ell, self.G, self.P)
            i4, w4 = _get_interp_weights(self.nside, rot, True, "cpu", torch.float64)   # [4, R]
            del rot
            pos, present = _bind(i4.numpy(), ids_np, not self.partial, self._sorted)
            w4 = w4.numpy()
            # the rule of HealPixConv: a neighbour with less than 1e-6 of the
            # full bilinear weight does not count
            present = _significant(w4, present, 0)
            pos = np.where(present, pos, 0)
            w = w4 * present
            coverage = w.sum(0)
            w = np.where(coverage > 0, w / np.where(coverage > 0, coverage, 1.0), 0.0)
            if not self.partial:
                coverage = np.ones_like(coverage)   # not 1 - 2e-16
            idx_t = torch.as_tensor(pos, dtype=torch.int32)
            w_t = torch.as_tensor(w, dtype=gdtype)

        self.register_buffer("_idx", idx_t.contiguous().to(device), persistent=False)
        self.register_buffer("_w", w_t.contiguous().to(device), persistent=False)
        self.register_buffer(
            "coverage",
            torch.as_tensor(coverage, dtype=gdtype).view(self.M, self.G, N, N).to(device),
            persistent=False,
        )
        self.register_buffer("centre_ids", torch.as_tensor(centre_ids, device=device),
                             persistent=False)
        self._geom = None

    # ------------------------------------------------------------------
    # Geometry, device and dtype
    # ------------------------------------------------------------------

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        self._geom = None
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_geom"] = None
        return state

    def _geometry(self) -> PatchGeometry:
        """The sparse map, with its transposed map cached across calls."""
        if self._geom is None:
            w = self._w if self._w.dtype == torch.float64 else self._w.float()
            self._geom = PatchGeometry(self._idx, w.contiguous(), self.K, self.M, self.G, self.P)
        return self._geom

    @property
    def device(self) -> torch.device:
        return self._idx.device

    def centre_lonlat(self):
        """Longitudes and latitudes of the patch centres, in degrees, on
        ``ellipsoid`` (geodetic latitudes on an ellipsoid)."""
        ids = self.centre_ids.detach().cpu().numpy()
        lon, lat = healpix_geo.nested.healpix_to_lonlat(
            ids, self.centre_level, ellipsoid=self.ellipsoid)
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _as_bck(self, x):
        is_numpy = not torch.is_tensor(x)
        t = torch.as_tensor(np.asarray(x)) if is_numpy else x
        t = t.to(self.device)
        nd = t.dim()
        if nd == 1:
            t = t.view(1, 1, -1)
        elif nd == 2:
            t = t.unsqueeze(1)
        elif nd != 3:
            raise ValueError(f"x must have shape [K], [B, K] or [B, C, K]; got {tuple(t.shape)}.")
        if t.shape[-1] != self.K:
            raise ValueError(f"x has {t.shape[-1]} cells, the domain has {self.K}.")
        if not t.is_floating_point():
            t = t.to(self._w.dtype)
        return t, is_numpy, nd

    def _selection(self, index, B):
        """``index`` [Ms] or [B, Ms] -> int32 [1 or B, Ms] on the device, or None."""
        if index is None:
            return None
        sel = torch.as_tensor(index, device=self.device)
        if sel.dtype in (torch.bool,) or sel.is_floating_point():
            raise ValueError("index must hold integer positions in centre_ids.")
        if sel.dim() == 1:
            sel = sel.unsqueeze(0)
        if sel.dim() != 2 or sel.shape[0] not in (1, B):
            raise ValueError(f"index must have shape [Ms] or [B={B}, Ms]; got {tuple(sel.shape)}.")
        if sel.numel():
            lo, hi = torch.aminmax(sel)
            if int(lo) < 0 or int(hi) >= self.M:      # one device sync
                raise ValueError(f"index must lie in [0, {self.M}).")
        return sel.to(torch.int32).contiguous()

    def _opts(self, t):
        return _Options(use_triton(t, self.backend))

    # ------------------------------------------------------------------
    # Extraction and fold
    # ------------------------------------------------------------------

    def forward(self, x, index=None, zorder: bool = False):
        """
        Extract the patches.

        Parameters
        ----------
        x : array-like, shape [K], [B, K] or [B, C, K]
            Values of the domain cells, in the order of ``cell_ids``.
        index : array-like of int or None
            Masked token sampling: positions in ``centre_ids`` of the
            patches to extract, shape ``[Ms]`` (the same for every sample) or
            ``[B, Ms]`` (one draw per sample, e.g. the tokens an MAE keeps).
            Only those patches are gathered.  None = all ``M`` patches.
        zorder : bool, default False
            ``mode="nested"`` on the full sphere or a complete sorted domain
            only: return the zero-copy view ``[B, M, 1, C, 4**p]`` of the
            data, the pixels of each patch in NESTED order instead of the
            ``N x N`` image (a linear patch embedding does not depend on the
            order of the pixels).

        Returns
        -------
        y : Tensor [B, Ms, G, C, N, N]
            ``[B, Ms, G, 1, N, N]`` for a ``[B, K]`` input and
            ``[Ms, G, 1, N, N]`` for a ``[K]`` input.  NumPy in, NumPy out.
        """
        t, is_numpy, nd = self._as_bck(x)
        B, C, _ = t.shape
        N = self.patch_size
        sel = self._selection(index, B)
        if zorder:
            if self.mode != "nested" or not self.contiguous or sel is not None:
                raise ValueError("zorder=True needs mode='nested' on the full sphere or a "
                                 "complete sorted domain, without index.")
            y = t.view(B, C, self.M, 1, self.P).permute(0, 2, 3, 1, 4)
            return self._restore(y, is_numpy, nd)
        if self.mode == "nested" and self.contiguous and sel is None:
            # one gather: [B, C, M, Q] viewed as [B, M, C, Q], pixels permuted
            y = _NestedTiles.apply(t, self._perm, self._iperm, False).view(
                B, self.M, 1, C, N, N)
        else:
            y = PatchExtractFunction.apply(t, self._geometry(), sel, self._opts(t))
            y = y.view(B, y.shape[1], self.G, C, N, N)
        return self._restore(y, is_numpy, nd)

    def fold(self, y, index=None, norm: Optional[str] = None):
        """
        Adjoint of :meth:`forward`: scatter-add the patches back to the cells.

        Parameters
        ----------
        y : array-like [B, Ms, G, C, N, N] (or [Ms, G, C, N, N])
            Patches in the layout of :meth:`forward`.
        index : array-like of int or None
            The positions in ``centre_ids`` of the ``Ms`` patches, as
            given to :meth:`forward`.  None = all ``M`` patches.
        norm : {"weight", "count", "none"} or None
            Normalisation (None = ``fold_norm`` of the constructor).  With
            ``index``, it counts only the patches given, per sample.  A cell
            that no patch reads is 0.

        Returns
        -------
        x : Tensor [B, C, K] ([C, K] for a 5-D input); NumPy in, NumPy out.
        """
        norm = self.fold_norm if norm is None else norm
        if norm not in FOLD_NORMS:
            raise ValueError(f"norm must be one of {FOLD_NORMS}.")
        is_numpy = not torch.is_tensor(y)
        t = torch.as_tensor(np.asarray(y)) if is_numpy else y
        t = t.to(self.device)
        squeeze = t.dim() == 5
        if squeeze:
            t = t.unsqueeze(0)
        N = self.patch_size
        if t.dim() != 6 or t.shape[2] != self.G or tuple(t.shape[-2:]) != (N, N):
            raise ValueError(f"y must have shape [B, Ms, {self.G}, C, {N}, {N}]; "
                             f"got {tuple(t.shape)}.")
        B, Ms, G, C = t.shape[:4]
        sel = self._selection(index, B)
        if sel is None and Ms != self.M:
            raise ValueError(f"y holds {Ms} patches; pass index for a subset of the {self.M}.")
        if sel is not None and sel.shape[1] != Ms:
            raise ValueError("index and y disagree on the number of patches.")
        t5 = t.reshape(B, Ms, G, C, self.P)
        if sel is not None and norm != "none" and sel.shape[1] > 1:
            srt = torch.sort(sel, dim=1).values
            if bool((srt[:, 1:] == srt[:, :-1]).any()):
                raise ValueError("a normalised fold needs distinct centres in each row of "
                                 "index; use norm='none' to sum repeated patches.")
        if self.mode == "nested" and self.contiguous and sel is None:
            # every cell is read once, with weight 1: the three norms agree
            out = _NestedTiles.apply(t5.reshape(B, self.M, C, self.P), self._perm,
                                     self._iperm, True)
        else:
            if not t5.is_floating_point():
                t5 = t5.to(self._w.dtype)
            out = PatchFoldFunction.apply(t5, self._geometry(), sel, norm, self._opts(t5))
        if squeeze:
            out = out[0]
        return out.detach().cpu().numpy() if is_numpy else out

    @staticmethod
    def _restore(y, is_numpy, nd):
        if nd == 1:
            y = y[0]
        return y.detach().cpu().numpy() if is_numpy else y

    def extra_repr(self):
        return (f"level={self.level}, patch_size={self.patch_size}, mode={self.mode!r}, "
                f"centres={self.centres!r}, centre_level={self.centre_level}, "
                f"ellipsoid={self.ellipsoid!r}, n_centres={self.M}, n_gauges={self.G}, "
                f"gauge_type={self.gauge_type!r}, partial={self.partial}")
