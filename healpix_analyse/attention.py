"""
attention.py
============
Local self-attention layers on HEALPix maps (issue #109, section 3).

``HealPixWindowAttention``
    Full multi-head attention inside the NESTED blocks of ``4**w`` cells
    (Swin-like windows).  A block is a contiguous slice of a sorted domain, so
    on the full sphere and on sorted domains of complete blocks the windows
    are a reshape followed by
    :func:`torch.nn.functional.scaled_dot_product_attention` (flash or
    memory-efficient kernels).  Other domains are padded and masked.  The
    replacement of Swin's shifted windows is selected by ``shift``.

``HealPixNeighbourhoodAttention``
    Each cell attends to the P taps of its gauge-oriented stencil, the
    stencil of :class:`~healpix_analyse.convol.HealPixConv`, with one relative
    position bias per head and tap shared over the sphere.  Fused Triton
    kernels on CUDA, a PyTorch tap loop otherwise
    (``healpix_analyse._attention_backend``).

Global attention on patch tokens (ViT, MAE) needs no layer of its own: it is
standard attention on the tokens of the patch extraction.

Tensors are ``[B, C, N]`` (batch, channels, cells), as in the other layers of
the package; ``N`` follows the order of ``cell_ids`` (full sphere: NESTED).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import healpix_geo

from healpix_analyse._attention_backend import (
    BACKENDS,
    WindowPlan,
    neighbourhood_attention,
    neighbourhood_attention_qkv,
    window_attention,
)
from healpix_analyse._conv_backend import build_transposed_map
from healpix_analyse._ellipsoid import canonicalize_ellipsoid
from healpix_analyse._topology import nested_neighbours
from healpix_analyse.convol import _DEFAULT_CACHE_DIR, HealPixConv

__all__ = [
    "WINDOW_SHIFTS",
    "DEFAULT_WINDOW_SHIFT",
    "HealPixWindowAttention",
    "HealPixNeighbourhoodAttention",
]

#: Replacements of Swin's shifted windows (``HealPixWindowAttention(shift=...)``).
WINDOW_SHIFTS = ("none", "alternate_level", "coarse_centred")

#: Default of ``HealPixWindowAttention(shift=None)``: windows centred on the
#: cells of a coarser level, overlapping across the block borders (decision of
#: the maintainer on the open question of issue #109).  Change it here.
DEFAULT_WINDOW_SHIFT = "coarse_centred"

_PAD = 8        # window lengths are padded to a multiple of 8 (SDPA alignment)


def _round_up(n, m=_PAD):
    return -(-int(n) // m) * m


def _check_level(level):
    if isinstance(level, bool) or int(level) != level or int(level) < 0:
        raise ValueError("level must be an integer >= 0.")
    return int(level)


def _domain_ids(level, cell_ids):
    npix = 12 * 4 ** level
    if cell_ids is None:
        return np.arange(npix, dtype=np.int64), False
    ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError("cell_ids must not be empty.")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must contain unique identifiers.")
    if np.any(ids < 0) or np.any(ids >= npix):
        raise ValueError(f"cell_ids contain identifiers outside level={level}.")
    return ids, True


def _unit_vectors(ids, level, ellipsoid):
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        np.asarray(ids, dtype=np.uint64), level, ellipsoid=ellipsoid)
    lon = np.deg2rad(np.asarray(lon, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


# ===========================================================================
# Window plans
# ===========================================================================

def _block_plan(ids, wl):
    """Queries grouped by the NESTED blocks of ``4**wl`` cells.

    Returns ``(blocks, q_index, q_inv)``: the block ids ``[nB]``; ``None,
    None`` for a sorted domain of complete blocks (a reshape), else the
    padded positions ``[nB, Tq]`` (padding = N) and, for each cell, its
    position in the flattened ``[nB * Tq]`` windows.
    """
    N = ids.size
    T = 4 ** wl
    if N % T == 0:
        heads = ids[::T]
        if np.all(heads % T == 0) and np.array_equal(
                ids, (heads[:, None] + np.arange(T)).ravel()):
            return heads >> (2 * wl), None, None
    order = np.argsort(ids, kind="stable")
    parent = ids[order] >> (2 * wl)
    blocks, start, counts = np.unique(parent, return_index=True, return_counts=True)
    Tq = _round_up(counts.max())
    blk = np.repeat(np.arange(blocks.size), counts)
    slot = np.arange(N) - start[blk]
    q_index = np.full((blocks.size, Tq), N, dtype=np.int64)
    q_index[blk, slot] = order
    q_inv = np.empty(N, dtype=np.int64)
    q_inv[order] = blk * Tq + slot
    return blocks, q_index, q_inv


def _centred_keys(ids, level, wl, blocks, halo, ellipsoid, chunk_cells=1 << 22):
    """Keys of the windows centred on the level-(L-wl) cells ``blocks``.

    The window of block ``c`` holds the domain cells whose angular distance
    to the centre of ``c`` is at most ``(1 + halo) * r_c``, with ``r_c`` the
    largest distance from that centre to a cell of the complete block ``c``.
    The candidates are the cells of ``c`` and of its (up to 8) neighbours at
    the coarse level.  Returns positions ``[nB, Tk]`` (padding = N).
    """
    N = ids.size
    T = 4 ** wl
    lc = level - wl
    sorted_ids = np.sort(ids)
    sort_order = np.argsort(ids, kind="stable")
    nb = nested_neighbours(blocks, lc)                                   # [nB, 8], -1 = none
    cand_blocks = np.concatenate([blocks[:, None], nb], axis=1)          # [nB, 9]
    centres = _unit_vectors(blocks, lc, ellipsoid)                       # [nB, 3]
    per = max(1, chunk_cells // (9 * T))
    rows = []
    for a in range(0, blocks.size, per):
        cb = cand_blocks[a:a + per]                                      # [m, 9]
        m = cb.shape[0]
        cells = (np.where(cb >= 0, cb, 0)[:, :, None] * T + np.arange(T)).reshape(m, -1)
        ok = np.repeat(cb >= 0, T, axis=1)
        vec = _unit_vectors(cells.ravel(), level, ellipsoid).reshape(m, 9 * T, 3)
        cosang = np.einsum("mkd,md->mk", vec, centres[a:a + per])
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        r_c = ang[:, :T].max(axis=1)
        keep = ok & (ang <= (1.0 + halo) * r_c[:, None] * (1 + 1e-9) + 1e-15)
        keep[:, :T] = True                                               # the block itself
        pos = np.searchsorted(sorted_ids, cells)
        pos_c = np.minimum(pos, N - 1)
        inside = sorted_ids[pos_c] == cells
        keep &= inside
        for r in range(m):
            rows.append(sort_order[pos_c[r][keep[r]]])
    Tk = _round_up(max(r.size for r in rows))
    kv = np.full((len(rows), Tk), N, dtype=np.int64)
    for i, r in enumerate(rows):
        kv[i, :r.size] = r
    return kv


def _colour_windows(kv_index, N):
    """Greedy colouring of overlapping windows: no two windows of a colour share a key.

    Returns the window ids sorted by colour (int32) and the start of each
    colour in that list.  With windows that overlap only their neighbours,
    a handful of colours suffice (4 to 6 on HEALPix).
    """
    NW, TK = kv_index.shape
    flat = kv_index.ravel()
    ok = flat < N
    cell = flat[ok]
    win = np.repeat(np.arange(NW), TK)[ok]
    order = np.lexsort((win, cell))
    cell, win = cell[order], win[order]
    # pairs of windows that share a cell: consecutive entries of the same cell,
    # up to the largest number of windows per cell
    pairs = []
    k = 1
    while True:
        same = cell[k:] == cell[:-k]
        if not same.any():
            break
        pairs.append(np.stack([win[:-k][same], win[k:][same]], axis=1))
        k += 1
    pairs = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    start = np.searchsorted(pairs[:, 0], np.arange(NW + 1))
    colour = np.full(NW, -1, dtype=np.int64)
    for w in range(NW):
        nb = colour[pairs[start[w]:start[w + 1], 1]]
        used = set(nb[nb >= 0].tolist())
        c = 0
        while c in used:
            c += 1
        colour[w] = c
    order = np.argsort(colour, kind="stable")
    bounds = tuple(int(b) for b in np.searchsorted(colour[order], np.arange(colour.max() + 2)))
    return order.astype(np.int32), bounds


class HealPixWindowAttention(nn.Module):
    """Multi-head self-attention inside NESTED windows (Swin-like).

    The queries of a window are a NESTED block of ``4**window_level`` cells,
    the descendants of one cell of level ``level - window_level``; each cell
    attends to the keys of its window with standard scaled dot-product
    attention.

    On CUDA (float32, float16, bfloat16, autocast) fused Triton kernels in
    the style of flash attention run on every kind of window: they read the
    packed projection in place, gather the rows of padded or overlapping
    windows in registers and mask the padding there, and their backward is
    deterministic (no atomics).  On the full sphere and on regional domains
    whose ``cell_ids`` are sorted and made of complete blocks, the query
    windows are consecutive cells (no index at all).  Elsewhere (CPU,
    float64, attention dropout) the windows are gathered and passed to
    :func:`torch.nn.functional.scaled_dot_product_attention`.

    Swin's shifted windows have no direct equivalent on HEALPix.  ``shift``
    selects the replacement:

    ``"none"``
        Every layer uses the blocks of ``window_level``: cells never see
        across a block border.
    ``"alternate_level"``
        Layers with ``shifted=True`` use the blocks of ``window_level + 1``
        (4 times larger).  Stacking ``shifted=False, True, False, True, ...``
        mixes the 4 sibling blocks; the receptive field stops at the block of
        ``window_level + 1``, and grows further only through Down/Up, as in
        the hierarchical stages of Swin.
    ``"coarse_centred"``
        The queries are the cells of a block; the keys are the cells within
        an angular radius ``(1 + halo) r_c`` of the centre of the block (its
        cell of level ``level - window_level``), with ``r_c`` the radius of
        the block itself.  The windows overlap across the block borders, so
        the receptive field grows at every layer, at the price of about
        ``1.6 (1 + halo)**2`` times more keys and a key mask.  ``shifted`` is
        ignored.

    ``shift=None`` takes :data:`DEFAULT_WINDOW_SHIFT`, ``"coarse_centred"``:
    the maintainer chose the overlapping windows centred on the cells of a
    coarser level as the replacement of Swin's shifted windows (issue #109).
    ``"alternate_level"`` and ``"none"`` stay available.

    Parameters
    ----------
    level : int
        HEALPix level of the input (``nside = 2**level``).
    embed_dim : int
        Number of channels C (input and output).
    num_heads : int
        Number of heads; must divide ``embed_dim``.
    window_level : int
        Windows of ``4**window_level`` cells, ``1 <= window_level <= level``
        (``<= level - 1`` with ``shift="alternate_level"``).
    cell_ids : array-like or None
        NESTED identifiers of a regional domain, in the order of the input.
        None = full sphere.
    shift : {"none", "alternate_level", "coarse_centred"} or None
    shifted : bool, default False
        With ``shift="alternate_level"``: this layer is the shifted one of a
        pair (windows of ``window_level + 1``).
    halo : float, default 0.25
        With ``shift="coarse_centred"``: relative overlap of the windows, in
        ``[0, 1]``.
    qkv_bias : bool, default True
    attn_drop : float, default 0.0
        Dropout on the attention weights, in training mode (runs the PyTorch
        path).
    backend : {"auto", "triton", "torch"}, default "auto"
    ellipsoid : str, default "WGS84"
        Ellipsoid of the cell centres (``shift="coarse_centred"`` only).
    device, dtype : torch device and dtype of the parameters.

    Notes
    -----
    The Triton backward is deterministic: the key and value gradients of
    overlapping windows are summed per cell in a fixed order.  On the
    PyTorch path, the backward of the gathers accumulates with
    ``index_put_``, whose order on CUDA is fixed only under
    ``torch.use_deterministic_algorithms(True)``.

    Examples
    --------
    >>> blocks = nn.Sequential(*[
    ...     HealPixWindowAttention(level=6, embed_dim=64, num_heads=4,
    ...                            window_level=3, shifted=bool(i % 2))
    ...     for i in range(4)])
    >>> y = blocks(torch.randn(2, 64, 12 * 4**6))
    """

    def __init__(
        self,
        level: int,
        embed_dim: int,
        num_heads: int,
        window_level: int,
        *,
        cell_ids=None,
        shift: Optional[str] = None,
        shifted: bool = False,
        halo: float = 0.25,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        backend: str = "auto",
        ellipsoid: str = "WGS84",
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.level = _check_level(level)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError("num_heads must divide embed_dim.")
        self.head_dim = self.embed_dim // self.num_heads
        self.shift = DEFAULT_WINDOW_SHIFT if shift is None else shift
        if self.shift not in WINDOW_SHIFTS:
            raise ValueError(f"shift must be one of {WINDOW_SHIFTS} or None, got {shift!r}.")
        self.shifted = bool(shifted)
        self.window_level = _check_level(window_level)
        wl = self.window_level + (1 if (self.shift == "alternate_level" and self.shifted) else 0)
        if self.window_level < 1 or wl > self.level:
            raise ValueError(
                f"window_level must be in [1, level] (and window_level + 1 <= level for "
                f"the shifted layer of shift='alternate_level'); got {window_level} "
                f"at level {level}.")
        self.effective_window_level = wl
        self.halo = float(halo)
        if not 0.0 <= self.halo <= 1.0:
            raise ValueError("halo must be in [0, 1].")
        self.attn_drop = float(attn_drop)
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
        self.backend = backend
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)

        ids, self.partial = _domain_ids(self.level, cell_ids)
        self.N = ids.size
        blocks, q_index, q_inv = _block_plan(ids, wl)
        self.n_windows = blocks.size
        self.fast = q_index is None
        if self.shift == "coarse_centred":
            kv_index = _centred_keys(ids, self.level, wl, blocks, self.halo, self.ellipsoid)
        else:
            kv_index = q_index
        self.window_len = 4 ** wl if q_index is None else q_index.shape[1]
        self.key_len = self.window_len if kv_index is None else kv_index.shape[1]

        def buf(name, a, dt=torch.int32):
            t = None if a is None else torch.as_tensor(a, dtype=dt, device=device)
            self.register_buffer(name, t, persistent=False)

        kv_len = colours = None
        self._colour_bounds = ()
        if kv_index is not None:
            # the keys of a window fill its first slots: loop over them only
            lens = (kv_index < self.N).sum(axis=1)
            if lens.min() != kv_index.shape[1]:
                kv_len = lens
        if kv_index is not None and kv_index is not q_index:
            colours, self._colour_bounds = _colour_windows(kv_index, self.N)
        self.n_colours = max(0, len(self._colour_bounds) - 1)
        self.mean_keys = float(kv_len.mean()) if kv_len is not None else float(self.key_len)
        buf("_q_index", q_index)
        buf("_q_inv", q_inv, torch.long)
        buf("_kv_index", kv_index)
        buf("_kv_len", kv_len)
        buf("_colours", colours)
        self.register_buffer(
            "_kv_mask",
            None if kv_index is None or not bool((kv_index >= self.N).any())
            else torch.as_tensor(kv_index < self.N, device=device),
            persistent=False)
        self.register_buffer("_blocks", torch.as_tensor(blocks, device=device), persistent=False)

        C = self.embed_dim
        self.qkv = nn.Linear(C, 3 * C, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = nn.Linear(C, C, device=device, dtype=dtype)

    @property
    def window_cells(self) -> torch.Tensor:
        """Identifiers of the window centres: cells of level ``level - effective_window_level``."""
        return self._blocks

    def plan(self) -> WindowPlan:
        """The windows as a ``WindowPlan`` (``healpix_analyse._attention_backend``)."""
        return WindowPlan(self.N, self.n_windows, self.window_len, self.key_len,
                          self._q_index, self._kv_index, self._kv_len, self._q_inv,
                          self._kv_mask, self._colours, self._colour_bounds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x [B, C, N] -> [B, C, N]``."""
        if x.dim() != 3 or x.shape[1] != self.embed_dim or x.shape[2] != self.N:
            raise ValueError(
                f"expected input [B, {self.embed_dim}, {self.N}], got {tuple(x.shape)}.")
        B, C, N = x.shape
        qkv = self.qkv(x.transpose(1, 2).to(self.qkv.weight.dtype))       # [B, N, 3C]
        o = window_attention(
            qkv.view(B, N, 3, self.num_heads, self.head_dim), self.plan(),
            backend=self.backend, dropout_p=self.attn_drop if self.training else 0.0)
        o = self.proj(o.reshape(B, N, C).to(self.proj.weight.dtype))
        return o.transpose(1, 2)

    def extra_repr(self):
        return (f"level={self.level}, embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"window_level={self.window_level}, shift={self.shift!r}, "
                f"shifted={self.shifted}, windows={self.n_windows}x{self.window_len}"
                f"{'' if self.key_len == self.window_len else f'->{self.key_len}'}, "
                f"fast={self.fast}")


# ===========================================================================
# Neighbourhood attention
# ===========================================================================

class HealPixNeighbourhoodAttention(nn.Module):
    """Multi-head attention of each cell on its gauge-oriented stencil.

    The HEALPix analogue of neighbourhood attention (NATTEN): cell ``n``
    attends to the ``P = kernel_sz**2`` taps of the stencil of
    :class:`~healpix_analyse.convol.HealPixConv` around it.  The key and the
    value of a tap are the bilinear interpolations of the keys and values of
    the 4 cells around the tap, with the weights of the convolution::

        s[h, g, n, p] = <q[h, n], k_p[h, g, n]> / sqrt(D) + bias[h, p]
        out[g, n]     = proj( sum_p softmax_p(s) v_p[g, n] )

    The stencil is oriented by the gauge, so one relative position bias
    table ``[num_heads, P]`` is shared over the whole sphere.  With
    ``n_gauges = G`` the stencil is rotated by ``pi / G`` from one gauge to
    the next and the output stacks the G gauges, ``[B, G*C, N]``, as
    ``HealPixConv`` does.  The projections are shared by the gauges.

    On CUDA (float32, float16, bfloat16, and under ``torch.autocast``) a
    fused Triton kernel gathers the taps in registers and runs an online
    softmax: the ``[B, H, G, N, P, D]`` gathered keys and values are never
    materialised, and the backward is deterministic (no atomics).  On CPU and
    in float64 a PyTorch loop over the taps runs instead (see
    ``healpix_analyse._attention_backend``).

    Parameters
    ----------
    level : int
        HEALPix level of the input.
    embed_dim : int
        Number of channels C.
    num_heads : int
        Number of heads; must divide ``embed_dim``.
    kernel_sz : int, default 3
        Odd stencil width (P = kernel_sz**2 taps), as in ``HealPixConv``.
    n_gauges, gauge_type, singularity_lonlat, ref_direction
        Gauge of the stencil, as in ``HealPixConv``.
    cell_ids : array-like or None
        NESTED identifiers of a regional domain, in the order of the input
        (any order).  None = full sphere.
    outside : {"mask", "centre"}, default "mask"
        Taps whose 4 neighbours are all outside a regional domain:
        ``"mask"`` leaves them out of the softmax, ``"centre"`` keeps them
        with the key and value of the centre cell (the rule of
        ``HealPixConv``).  Taps with some neighbours inside use the
        renormalised bilinear weights of those neighbours in both cases.
    qkv_bias : bool, default True
    pos_bias : bool, default True
        Learn the relative position bias ``[num_heads, P]``.
    backend : {"auto", "triton", "torch"}, default "auto"
    ellipsoid, cache_dir
        As in ``HealPixConv`` (the geometry is built by it, and shares its
        disk cache).
    device, dtype : torch device and dtype of the parameters.

    Examples
    --------
    >>> na = HealPixNeighbourhoodAttention(level=6, embed_dim=64, num_heads=4)
    >>> y = na(torch.randn(2, 64, 12 * 4**6))          # [2, 64, N]
    """

    backend = "auto"
    _tmap = None

    def __init__(
        self,
        level: int,
        embed_dim: int,
        num_heads: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat=None,
        ref_direction=None,
        *,
        cell_ids=None,
        outside: str = "mask",
        qkv_bias: bool = True,
        pos_bias: bool = True,
        backend: str = "auto",
        ellipsoid: str = "WGS84",
        cache_dir=_DEFAULT_CACHE_DIR,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if outside not in ("mask", "centre"):
            raise ValueError("outside must be 'mask' or 'centre'.")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
        self.level = _check_level(level)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        if self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError("num_heads must divide embed_dim.")
        self.head_dim = self.embed_dim // self.num_heads
        self.outside = outside
        self.backend = backend
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The geometry of HealPixConv, built in float64 on the CPU with its
        # domain support (missing="local_mean"), and kept as tap geometry.
        geo = HealPixConv(
            self.level, 1, 1, kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ref_direction=ref_direction, cell_ids=cell_ids, ellipsoid=ellipsoid,
            device="cpu", dtype=torch.float64, cache_dir=cache_dir, missing="local_mean",
        )
        self.kernel_sz = geo.kernel_sz
        self.P, self.G, self.N = geo.P, geo.G, geo.K
        self.gauge_type = geo.gauge_type
        self.partial = geo.partial
        valid = None
        if outside == "mask":
            v = (geo._support.view(self.G, self.N, self.P) > 0).permute(2, 0, 1).contiguous()
            if not bool(v.all()):
                valid = v
        self.register_buffer("_tap_idx", geo._tap_idx.to(device).contiguous(), persistent=False)
        self.register_buffer("_tap_w", geo._tap_w.to(device=device, dtype=dtype).contiguous(),
                             persistent=False)
        self.register_buffer("_valid", None if valid is None else valid.to(device),
                             persistent=False)
        del geo

        C = self.embed_dim
        self.qkv = nn.Linear(C, 3 * C, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = nn.Linear(C, C, device=device, dtype=dtype)
        if pos_bias:
            self.pos_bias = nn.Parameter(
                torch.empty(self.num_heads, self.P, device=device, dtype=dtype))
            nn.init.trunc_normal_(self.pos_bias, std=0.02)
        else:
            self.register_parameter("pos_bias", None)

    def _apply(self, fn, *args, **kwargs):
        result = super()._apply(fn, *args, **kwargs)
        self._tmap = None
        return result

    def _transposed_map(self):
        if self._tmap is None or self._tmap[0].device != self._tap_idx.device:
            self._tmap = build_transposed_map(self._tap_idx, self._tap_w.float(), self.N)
        return self._tmap

    def attend(self, q, k, v):
        """The attention core: ``q, k, v [B, H, N, D] -> [B, G, H, N, D]`` (no projections)."""
        w = self._tap_w
        if w.dtype != q.dtype and q.dtype == torch.float64:
            w = w.to(q.dtype)
        return neighbourhood_attention(
            q, k, v, self.pos_bias, self._tap_idx, w, self._valid,
            backend=self.backend, transposed=self._transposed_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x [B, C, N] -> [B, G*C, N]``."""
        if x.dim() != 3 or x.shape[1] != self.embed_dim or x.shape[2] != self.N:
            raise ValueError(
                f"expected input [B, {self.embed_dim}, {self.N}], got {tuple(x.shape)}.")
        B, C, N = x.shape
        H, D, G = self.num_heads, self.head_dim, self.G
        qkv = self.qkv(x.transpose(1, 2).to(self.qkv.weight.dtype))       # [B, N, 3C]
        w = self._tap_w
        if w.dtype != qkv.dtype and qkv.dtype == torch.float64:
            w = w.to(qkv.dtype)
        o = neighbourhood_attention_qkv(
            qkv.view(B, N, 3, H, D), self.pos_bias, self._tap_idx, w, self._valid,
            backend=self.backend, transposed=self._transposed_map)        # [B, G, N, H, D]
        o = o.reshape(B, G, N, C)
        o = self.proj(o.to(self.proj.weight.dtype))                       # [B, G, N, C]
        return o.permute(0, 1, 3, 2).reshape(B, G * C, N)

    def extra_repr(self):
        return (f"level={self.level}, embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"kernel_sz={self.kernel_sz}, n_gauges={self.G}, gauge_type={self.gauge_type!r}, "
                f"cells={self.N}, outside={self.outside!r}, backend={self.backend!r}")
