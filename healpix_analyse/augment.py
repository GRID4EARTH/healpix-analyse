"""Batched data augmentation on HEALPix grids, on the GPU.

:class:`HealPixAugment` applies a list of transforms to a whole batch at once,
with its own random parameters for every sample, drawn from a seeded
``torch.Generator``.  Every transform runs on the device of the data and is
differentiable with respect to the data.

Geometric transforms
    Each one is a *source map*: the output at cell ``p`` reads the input at
    ``S(p)``.  Two families compose exactly:

    * 3-D maps ``S(p) = R p`` (``R`` 3x3, orthogonal): :class:`RandomSO3`
      (global rotation of a full-sphere field), :class:`RandomShift` (move a
      regional domain on the sphere, a rotation about an axis through the
      centre of the Earth), :class:`Rotation3D`, and :class:`RandomFlip` on
      the full sphere (longitude -> -longitude);
    * tangent-plane maps ``S(p) = lift(A proj(p))`` (``A`` 2x2) in the
      gnomonic projection about the centre of a regional domain:
      :class:`RandomRotation` (any angle about the domain centre),
      :class:`RandomZoom` (homothety of any scale factor), :class:`RandomFlip`
      (mirror of the local frame), :class:`TangentLinear`.

    A run of consecutive geometric transforms whose 3-D maps come before its
    tangent-plane maps (for example shift, then rotation, zoom and flip) is
    fused into **one** source map ``S(p) = R lift(A proj(p))`` and **one**
    gather: 4 bilinear weights per output cell (per sampling tap, see
    ``antialias``), computed on the device from the per-sample ``R`` and
    ``A``.  The HEALPix bilinear interpolation is
    :func:`healpix_analyse.get_interp_weights` (the ``healpy`` algorithm)
    transcribed to torch, so no geometry is rebuilt on the CPU for a new
    draw.  A 3-D map that follows a tangent-plane map starts a new gather.

Channel, value and support transforms
    :class:`RandomGaugeRotation` (cyclic shift, and optionally reflection, of
    the G gauge axis of :class:`~healpix_analyse.HealPixConv` outputs; free,
    and it commutes with the gathers), :class:`AdditiveNoise`,
    :class:`RandomMask` (random cell or NESTED-token masking, MAE style) and
    :class:`RandomCrop` (random sub-domains through NESTED prefixes, which are
    contiguous slices).

Outside the domain
    An output cell whose source point falls outside the domain gets ``fill``
    (NaN by default) and ``valid = False`` in the result.  By default
    (``min_support=1``) a cell is inside when every bilinear neighbour with a
    non-zero weight is a cell of the domain; a lower ``min_support`` also
    keeps cells with that fraction of their weight in the domain and
    renormalises their weights (a first-order, extrapolated value at the
    border).
    Labels (nearest neighbour) get ``label_fill`` there.  A NaN in the input
    propagates to every output cell that gives it a non-zero weight.

Examples
--------
>>> import torch
>>> from healpix_analyse.augment import (HealPixAugment, RandomShift,
...     RandomRotation, RandomZoom, RandomFlip, AdditiveNoise, RandomMask)
>>> cell_ids = torch.arange(4**6) + 3 * 4**6     # a base-cell block, level 6
>>> aug = HealPixAugment(
...     [RandomShift(0.02), RandomRotation(), RandomZoom((0.8, 1.25)),
...      RandomFlip(), AdditiveNoise(0.01), RandomMask(0.5, block=16)],
...     level=6, cell_ids=cell_ids)
>>> g = torch.Generator().manual_seed(0)
>>> x = torch.randn(8, 2, cell_ids.numel())
>>> out = aug(x, generator=g)
>>> out.x.shape, out.valid.shape, out.mask.shape
(torch.Size([8, 2, 4096]), torch.Size([8, 4096]), torch.Size([8, 4096]))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import healpix_geo
import numpy as np
import torch
from torch import nn

from healpix_analyse._augment_backend import (
    _TRITON_DTYPES,
    GatherMap,
    apply_gather,
    fused_augment_triton,
    gather_backward,
    ring_theta_table,
    triton_on,
    bilinear_ring,
    ring_to_nest,
)

__all__ = [
    "HealPixAugment",
    "AugmentPlan",
    "AugmentResult",
    "RandomRotation",
    "RandomSO3",
    "RandomShift",
    "RandomZoom",
    "RandomFlip",
    "Rotation3D",
    "TangentLinear",
    "RandomGaugeRotation",
    "AdditiveNoise",
    "RandomMask",
    "RandomCrop",
    "gauge_rotate",
]

_F64 = torch.float64
# Geometry points per chunk: bounds the transient memory on CUDA, and keeps
# the temporaries in cache on the CPU (large fresh allocations page-fault).
_MAX_POINTS = {"cuda": 1 << 22, "cpu": 1 << 15}
# Dense RING -> position table for a regional domain when its RING range is at
# most _RINGTAB_FACTOR times its number of cells (or _RINGTAB_MIN entries).
_RINGTAB_FACTOR = 64
_RINGTAB_MIN = 1 << 22


# ===========================================================================
# Random draws
# ===========================================================================

def _rand(B, gen):
    return torch.rand(B, generator=gen, device=gen.device, dtype=_F64).cpu()


def _uniform(B, gen, lo, hi):
    return lo + (hi - lo) * _rand(B, gen)


def _as_range(v, name, symmetric=False):
    if isinstance(v, (tuple, list)):
        lo, hi = float(v[0]), float(v[1])
    else:
        lo, hi = (-float(v), float(v)) if symmetric else (float(v), float(v))
    if hi < lo:
        raise ValueError(f"{name}: empty range ({lo}, {hi})")
    return lo, hi


def _rot2(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rodrigues(axis, angle):
    """Rotation matrices [B, 3, 3] about unit ``axis`` [B, 3] by ``angle`` [B]."""
    B = axis.shape[0]
    K = torch.zeros(B, 3, 3, dtype=axis.dtype)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s = torch.sin(angle).view(B, 1, 1)
    c = torch.cos(angle).view(B, 1, 1)
    eye = torch.eye(3, dtype=axis.dtype).expand(B, 3, 3)
    return eye + s * K + (1 - c) * (K @ K)


# ===========================================================================
# Transforms
# ===========================================================================

class Transform:
    """Base class of the transforms understood by :class:`HealPixAugment`."""

    #: ``"R"`` (3-D map), ``"A"`` (tangent-plane map), ``"gauge"``,
    #: ``"noise"``, ``"mask"`` or ``"crop"``.
    kind = None

    def kind_for(self, grid) -> str:
        return self.kind

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


class RandomRotation(Transform):
    """Rotation of a regional domain about its centre by a random angle.

    Parameters
    ----------
    angles : float or (float, float), default pi
        Angle range in radians; a float ``a`` means ``(-a, a)``.  Drawn
        uniformly per sample.  ``(a, a)`` gives a fixed rotation.
    """

    kind = "A"

    def __init__(self, angles: Union[float, Tuple[float, float]] = math.pi):
        self.angles = _as_range(angles, "angles", symmetric=True)

    def draw(self, B, gen, grid):
        a = _uniform(B, gen, *self.angles)
        return _rot2(a), {"angle": a}


class RandomZoom(Transform):
    """Homothety of factor ``s`` about the centre of a regional domain.

    ``s > 1`` magnifies (zoom in), ``s < 1`` shrinks (zoom out); the output
    at tangent-plane position ``u`` reads the input at ``u / s``.  For
    ``s < 1`` the input is low-pass filtered first, see ``antialias`` in
    :class:`HealPixAugment`.

    Parameters
    ----------
    scales : float or (float, float), default (0.8, 1.25)
        Scale range, drawn log-uniformly per sample.
    """

    kind = "A"

    def __init__(self, scales: Union[float, Tuple[float, float]] = (0.8, 1.25)):
        lo, hi = _as_range(scales, "scales")
        if lo <= 0:
            raise ValueError("scales must be positive")
        self.scales = (lo, hi)

    def draw(self, B, gen, grid):
        s = torch.exp(_uniform(B, gen, math.log(self.scales[0]), math.log(self.scales[1])))
        A = torch.zeros(B, 2, 2, dtype=_F64)
        A[:, 0, 0] = 1.0 / s
        A[:, 1, 1] = 1.0 / s
        return A, {"scale": s}


class RandomFlip(Transform):
    """Mirror of the local frame with probability ``p``.

    On a regional domain: east-west mirror about the north-south line through
    the domain centre (``axis="ew"``) or north-south mirror (``axis="ns"``).
    On the full sphere: longitude -> -longitude (``"ew"``) or latitude ->
    -latitude (``"ns"``).  Combine with :class:`RandomRotation` for a mirror
    about a random line.  For gauge channels, see :class:`RandomGaugeRotation`
    (``flip``).
    """

    def __init__(self, p: float = 0.5, axis: str = "ew"):
        if axis not in ("ew", "ns"):
            raise ValueError("axis must be 'ew' or 'ns'")
        self.p = float(p)
        self.axis = axis

    def kind_for(self, grid):
        return "R" if grid.full else "A"

    def draw(self, B, gen, grid):
        flip = _rand(B, gen) < self.p
        sgn = torch.where(flip, -1.0, 1.0).to(_F64)
        if grid.full:
            M = torch.eye(3, dtype=_F64).repeat(B, 1, 1)
            M[:, 1 if self.axis == "ew" else 2, 1 if self.axis == "ew" else 2] = sgn
        else:
            M = torch.eye(2, dtype=_F64).repeat(B, 1, 1)
            M[:, 0 if self.axis == "ew" else 1, 0 if self.axis == "ew" else 1] = sgn
        return M, {"flip": flip}


class TangentLinear(Transform):
    """Fixed 2x2 map ``A`` of the tangent plane of a regional domain.

    The output at tangent-plane position ``u`` (east, north, gnomonic
    projection about the domain centre) reads the input at ``A u``.  ``A`` is
    ``[2, 2]`` or ``[B, 2, 2]`` (one per sample).
    """

    kind = "A"

    def __init__(self, A):
        self.A = torch.as_tensor(A, dtype=_F64)

    def draw(self, B, gen, grid):
        A = self.A.cpu()
        return (A.expand(B, 2, 2) if A.dim() == 2 else A).clone(), {}


class Rotation3D(Transform):
    """Fixed 3x3 orthogonal map ``R``: the output at ``p`` reads the input at ``R p``.

    ``R`` is ``[3, 3]`` or ``[B, 3, 3]``.  A reflection (``det R = -1``) is
    allowed.
    """

    kind = "R"

    def __init__(self, R):
        self.R = torch.as_tensor(R, dtype=_F64)

    def draw(self, B, gen, grid):
        R = self.R.cpu()
        return (R.expand(B, 3, 3) if R.dim() == 2 else R).clone(), {}


class RandomSO3(Transform):
    """Uniformly random global rotation (Haar measure on SO(3)).

    Meant for full-sphere fields; on a regional domain most cells would read
    outside the domain (use :class:`RandomShift` and :class:`RandomRotation`).
    """

    kind = "R"

    def draw(self, B, gen, grid):
        q = torch.randn(B, 4, generator=gen, device=gen.device, dtype=_F64).cpu()
        q = q / q.norm(dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        R = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ], -2)
        return R, {"quaternion": q}


class RandomShift(Transform):
    """Translation of a regional domain on the sphere.

    The content moves by a great-circle distance ``d`` in a random direction:
    a rotation about the axis through the centre of the Earth perpendicular
    to the domain centre and to the direction of motion.  The data is
    resampled on the cells of the domain; cells that read outside it get the
    fill value.

    Parameters
    ----------
    max_distance : float or (float, float)
        Distance in radians (multiply by the Earth radius for metres).  A
        float draws the displacement uniformly in the disc of that radius;
        a pair ``(lo, hi)`` draws the distance uniformly in ``[lo, hi]``.
    """

    kind = "R"

    def __init__(self, max_distance: Union[float, Tuple[float, float]]):
        self.disc = not isinstance(max_distance, (tuple, list))
        self.max_distance = _as_range(max_distance, "max_distance") if not self.disc else (0.0, float(max_distance))

    def draw(self, B, gen, grid):
        if grid.full:
            raise ValueError("RandomShift needs a regional domain; use RandomSO3 on the full sphere")
        psi = _uniform(B, gen, 0.0, 2 * math.pi)
        u = _rand(B, gen)
        lo, hi = self.max_distance
        d = hi * torch.sqrt(u) if self.disc else lo + (hi - lo) * u
        c, e1, e2 = (grid.frame[k].cpu().to(_F64) for k in range(3))
        t = torch.cos(psi)[:, None] * e1 + torch.sin(psi)[:, None] * e2
        axis = torch.cross(c.expand(B, 3), t, dim=1)
        axis = axis / axis.norm(dim=1, keepdim=True)
        return _rodrigues(axis, d), {"distance": d, "direction": psi}


class RandomGaugeRotation(Transform):
    """Cyclic shift (and optional reflection) of the gauge axis.

    For the outputs of :class:`~healpix_analyse.HealPixConv` with
    ``n_gauges=G`` (channels ``[B, G*C, N]``, gauge-major), whose gauge ``g``
    is the orientation ``g * pi / G``.  Gauge ``g`` of the output is gauge
    ``(s*g - k) mod G`` of the input, with ``k`` uniform in ``[0, G)`` and
    ``s = -1`` with probability ``flip`` (mirror of the local frame).  Free
    (a channel permutation) and it commutes with the geometric gathers.
    """

    kind = "gauge"

    def __init__(self, n_gauges: int, flip: float = 0.0):
        self.n_gauges = int(n_gauges)
        self.flip = float(flip)

    def draw(self, B, gen, grid):
        G = self.n_gauges
        k = torch.floor(_rand(B, gen) * G).long().clamp_max(G - 1)
        s = torch.where(_rand(B, gen) < self.flip, -1, 1).long()
        g = torch.arange(G)
        perm = (s[:, None] * g[None, :] - k[:, None]) % G
        return perm, {"gauge_shift": k, "gauge_flip": s < 0}


class AdditiveNoise(Transform):
    """Additive Gaussian noise, cell by cell.

    ``std`` is a float or a range ``(lo, hi)`` drawn uniformly per sample.
    The noise is drawn on the device of the data from a seed taken from the
    generator, so re-applying a plan gives the same noise.
    """

    kind = "noise"

    def __init__(self, std: Union[float, Tuple[float, float]] = 0.1):
        self.std = _as_range(std, "std")

    def draw(self, B, gen, grid):
        sig = _uniform(B, gen, *self.std)
        seed = int(torch.randint(0, 2 ** 62, (1,), generator=gen, device=gen.device).item())
        return (sig, seed), {"noise_std": sig}


class RandomMask(Transform):
    """Random cell or token masking, MAE style.

    A fraction ``ratio`` of the tokens of every sample is masked, each sample
    with its own random set.  A token is ``block`` consecutive cells of the
    data (``block = 4**p`` gives the NESTED blocks of the patches of level
    ``L - p`` when the data is in NESTED order and the domain is made of whole
    blocks); ``block=1`` masks single cells.  Masked values are set to
    ``fill``.  The result carries ``mask`` ([B, N], True = masked) and the MAE
    index tensors ``ids_keep`` [B, n_keep] and ``ids_restore`` [B, n_tokens].
    """

    kind = "mask"

    def __init__(self, ratio: float = 0.75, block: int = 1, fill: float = 0.0):
        if not 0.0 <= ratio < 1.0:
            raise ValueError("ratio must be in [0, 1)")
        self.ratio = float(ratio)
        self.block = int(block)
        self.fill = float(fill)

    def draw(self, B, gen, grid, n_cells):
        if n_cells % self.block:
            raise ValueError(f"RandomMask: {n_cells} cells is not a multiple of block={self.block}")
        n_tok = n_cells // self.block
        n_keep = int(round(n_tok * (1.0 - self.ratio)))
        noise = torch.rand(B, n_tok, generator=gen, device=gen.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        tok_mask = torch.ones(B, n_tok, dtype=torch.bool, device=gen.device)
        tok_mask[:, :n_keep] = False
        tok_mask = torch.gather(tok_mask, 1, ids_restore)
        return (tok_mask, ids_shuffle[:, :n_keep], ids_restore), {}


class RandomCrop(Transform):
    """Random sub-domain of ``n_cells`` cells through a NESTED prefix.

    ``n_cells = 4**k``: every sample keeps the ``4**k`` children of one
    random cell of level ``L - k`` whose children are all in the domain, a
    contiguous slice of the sorted cells.  Must come after the geometric
    transforms.  The result carries the ``cell_ids`` [B, n_cells] of each
    crop.
    """

    kind = "crop"

    def __init__(self, n_cells: int):
        k = int(round(math.log(n_cells, 4)))
        if 4 ** k != n_cells:
            raise ValueError("n_cells must be a power of 4")
        self.n_cells = int(n_cells)

    def draw(self, B, gen, grid):
        starts = grid.block_starts(self.n_cells)
        if starts.numel() == 0:
            raise ValueError(f"no complete NESTED block of {self.n_cells} cells in the domain")
        r = torch.floor(_rand(B, gen) * starts.numel()).long().clamp_max(starts.numel() - 1)
        return starts[r], {"crop_start": starts[r]}


def gauge_rotate(x: torch.Tensor, shifts, n_gauges: int, flip=None) -> torch.Tensor:
    """Per-sample cyclic shift (and reflection) of the gauge axis.

    ``x`` is ``[B, G*C, N]``; gauge ``g`` of sample ``b`` of the output is
    gauge ``(s_b * g - shifts[b]) mod G`` of the input, ``s_b = -1`` where
    ``flip[b]`` is true.
    """
    B = x.shape[0]
    G = int(n_gauges)
    shifts = torch.as_tensor(shifts, device=x.device).long().reshape(-1).expand(B)
    s = torch.ones(B, dtype=torch.long, device=x.device)
    if flip is not None:
        s = torch.where(torch.as_tensor(flip, device=x.device).reshape(-1).expand(B), -1, 1)
    perm = (s[:, None] * torch.arange(G, device=x.device)[None, :] - shifts[:, None]) % G
    return _permute_gauges(x, perm, G)


def _permute_gauges(x, perm, G):
    B, CG = x.shape[0], x.shape[1]
    if CG % G:
        raise ValueError(f"{CG} channels is not a multiple of n_gauges={G}")
    rest = x.shape[2:]
    xv = x.reshape(B, G, CG // G, *rest)
    idx = perm.to(x.device).view(B, G, *([1] * (xv.dim() - 2))).expand_as(xv)
    return torch.gather(xv, 1, idx).reshape(x.shape)


# ===========================================================================
# Domain geometry (built once, on the CPU, with healpix-geo)
# ===========================================================================

class _Grid(nn.Module):
    def __init__(self, level, cell_ids, geom_dtype):
        super().__init__()
        self.level = int(level)
        self.full = cell_ids is None
        npix = 12 * 4 ** self.level
        if self.full:
            ids = np.arange(npix, dtype=np.int64)
        else:
            ids = np.asarray(
                cell_ids.detach().cpu().numpy() if torch.is_tensor(cell_ids) else cell_ids
            ).astype(np.int64).reshape(-1)
            if ids.size == 0 or ids.min() < 0 or ids.max() >= npix:
                raise ValueError(f"cell_ids must be NESTED ids of level {self.level}")
        perm = np.argsort(ids, kind="stable")
        ids_sorted = ids[perm]
        if not self.full and np.any(np.diff(ids_sorted) == 0):
            raise ValueError("cell_ids contains duplicates")
        self.N = ids.size
        self.sorted_is_identity = bool(np.all(perm == np.arange(ids.size)))
        lon, lat = healpix_geo.nested.healpix_to_lonlat(ids.astype(np.uint64), self.level)
        lon = np.radians(np.asarray(lon, dtype=np.float64))
        lat = np.radians(np.asarray(lat, dtype=np.float64))
        xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
        self.register_buffer("xyz", torch.as_tensor(xyz, dtype=geom_dtype), persistent=False)
        self.register_buffer("ids_sorted", torch.as_tensor(ids_sorted), persistent=False)
        self.register_buffer("perm", torch.as_tensor(perm), persistent=False)
        self.h = math.sqrt(4 * math.pi / npix)  # mean cell size (radians)
        if self.full:
            frame = np.eye(3)
            uv = np.zeros((self.N, 2))
        else:
            c = xyz.mean(0)
            c /= np.linalg.norm(c)
            ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(ref, c)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(c, e1)
            dot = xyz @ c
            if dot.min() <= 0.05:
                raise ValueError(
                    "the regional domain must lie within a hemisphere (cells up to "
                    "~87 degrees from its centre) for the tangent-plane transforms"
                )
            uv = np.stack([xyz @ e1, xyz @ e2], -1) / dot[:, None]
            frame = np.stack([c, e1, e2])
        self.register_buffer("frame", torch.as_tensor(frame, dtype=geom_dtype), persistent=False)
        self.register_buffer("uv", torch.as_tensor(uv, dtype=geom_dtype), persistent=False)
        self._ids_np = ids_sorted
        self._perm_np = perm
        self._pyr = {}
        self._blocks = {}

    def theta_table(self, nside, dtype, device):
        key = ("theta", int(nside), str(dtype), str(device))
        if key not in self._pyr:
            self._pyr[key] = ring_theta_table(int(nside), dtype, device)
        return self._pyr[key]

    # -- NESTED -> position tables (fused kernel) ----------------------------
    def nest_table(self, k, device):
        """Cell -> position lookup of level ``L - k`` for the fused kernel.

        Built once per domain; no search in the kernel.  Full sphere:
        ``("r2n", table)``, the RING -> NESTED table (int32), or ``("full",)``
        (closed-form RING -> NESTED) above 2**28 cells.  Compact regional
        domain (RING range at most 64 times the number of cells, or 4M
        entries): ``("ringtab", table, rmin)``, a dense RING -> position
        table.  Otherwise ``("nest2", parent, block, 2*j)``: a two-level table built
        once per domain.  ``parent`` [12 * 4**(L-k-j)] holds the slot of each
        level-(L-k-j) parent that has cells in the domain (-1 otherwise), and
        ``block`` [n_slots * 4**j] the position of each of its 4**j children
        in the source array of that level (-1 when absent).  ``j`` minimises
        the size of the two arrays (dense over the id range for a compact
        domain, block-sparse for a scattered one).
        """
        key = ("nest2", int(k), str(device))
        if key not in self._pyr:
            order = self.level - k
            npix = 12 * 4 ** order
            if self.full and k == 0:
                if npix <= (1 << 28):
                    r2n = np.asarray(healpix_geo.ring.to_nested(
                        np.arange(npix, dtype=np.uint64), order)).astype(np.int32)
                    self._pyr[key] = ("r2n", torch.as_tensor(r2n, device=device))
                else:
                    self._pyr[key] = ("full",)
            else:
                ids_k = self._ids_np if k == 0 else self.pyramid(k, "cpu")[k - 1][0].numpy()
                ring = np.asarray(healpix_geo.nested.to_ring(ids_k.astype(np.uint64), order)).astype(np.int64)
                rmin, rmax = int(ring.min()), int(ring.max())
                span = rmax - rmin + 1
                if span <= max(_RINGTAB_FACTOR * ids_k.size, _RINGTAB_MIN) and span < (1 << 31):
                    tab = np.full(span, -1, dtype=np.int32)
                    tab[ring - rmin] = self._perm_np if k == 0 else np.arange(ids_k.size)
                    self._pyr[key] = ("ringtab", torch.as_tensor(tab, device=device), rmin)
                    return self._pyr[key]
                if k == 0:
                    ids, pos = self._ids_np, self._perm_np
                else:
                    ids = self.pyramid(k, "cpu")[k - 1][0].numpy()
                    pos = np.arange(ids.size)
                best = None
                for j in range(order + 1):
                    par = np.unique(ids >> (2 * j))
                    size = 12 * 4 ** (order - j) + par.size * 4 ** j
                    if best is None or size < best[0]:
                        best = (size, j, par)
                _, j, par = best
                parent = np.full(12 * 4 ** (order - j), -1, dtype=np.int32)
                parent[par] = np.arange(par.size, dtype=np.int32)
                block = np.full(par.size * 4 ** j, -1, dtype=np.int32)
                block[parent[ids >> (2 * j)].astype(np.int64) * 4 ** j + (ids & (4 ** j - 1))] = pos
                self._pyr[key] = ("nest2", torch.as_tensor(parent, device=device),
                                  torch.as_tensor(block, device=device), 2 * j)
        return self._pyr[key]

    # -- RING lookup tables --------------------------------------------------
    def ring_index(self, k, device):
        """Lookup of RING ids of level ``L - k`` into positions of the source.

        Returns ``("table", r2n)`` (full sphere: position = r2n[ring]) or
        ``("sorted", ring_sorted, pos)`` (sorted RING ids of the cells of
        level ``L - k`` of the domain and their positions in the source array
        of that level).  Built once with healpix-geo.
        """
        key = ("ring", int(k), str(device))
        if key not in self._pyr:
            order = self.level - k
            if k == 0:
                ids, pos = self._ids_np, self._perm_np
            else:
                ids = self.pyramid(k, "cpu")[k - 1][0].numpy()
                pos = np.arange(ids.size)
            if self.full and k == 0:
                npix = 12 * 4 ** order
                if npix <= (1 << 27):
                    r2n = np.asarray(
                        healpix_geo.ring.to_nested(np.arange(npix, dtype=np.uint64), order)
                    ).astype(np.int64)
                    self._pyr[key] = ("table", torch.as_tensor(r2n, device=device))
                else:
                    self._pyr[key] = ("bits",)
            else:
                ring = np.asarray(
                    healpix_geo.nested.to_ring(ids.astype(np.uint64), order)
                ).astype(np.int64)
                o = np.argsort(ring, kind="stable")
                self._pyr[key] = (
                    "sorted",
                    torch.as_tensor(ring[o], device=device),
                    torch.as_tensor(pos[o], device=device),
                )
        return self._pyr[key]

    # -- crops ---------------------------------------------------------------
    def block_starts(self, n):
        if n not in self._blocks:
            s = self._ids_np
            if n > s.size:
                st = np.zeros(0, dtype=np.int64)
            else:
                first = s[: s.size - n + 1]
                last = s[n - 1:]
                st = np.nonzero((first % n == 0) & (last - first == n - 1))[0]
            self._blocks[n] = torch.as_tensor(st, dtype=torch.long)
        return self._blocks[n]

    # -- mean pyramid for antialias="pyramid" -------------------------------
    def pyramid(self, kmax, device):
        """Per level k = 1..kmax: (sorted ids, child table [N_k, 4], offset)."""
        key = (int(kmax), str(device))
        if key not in self._pyr:
            levels = []
            ids_prev = self._ids_np
            pos_prev = self._perm_np  # level-0 positions in input order
            off = self.N
            for _ in range(int(kmax)):
                parent = ids_prev >> 2
                ids_k = np.unique(parent)
                row = np.searchsorted(ids_k, parent)
                table = np.full((ids_k.size, 4), -1, dtype=np.int64)
                table[row, ids_prev & 3] = pos_prev
                levels.append((
                    torch.as_tensor(ids_k, device=device),
                    torch.as_tensor(table, device=device),
                    off,
                ))
                off += ids_k.size
                ids_prev = ids_k
                pos_prev = np.arange(ids_k.size)
            self._pyr[key] = levels
        return self._pyr[key]


# ===========================================================================
# Plan and result
# ===========================================================================

@dataclass
class AugmentResult:
    """Output of :class:`HealPixAugment`.

    Attributes
    ----------
    x : Tensor
        Augmented data, same layout as the input (``[B, N]``, ``[B, C, N]``
        or ``[B, ..., N]``; ``N`` changes after a :class:`RandomCrop`).
    labels : Tensor, dict of Tensors, or None
        Labels / masks transformed with the same draw (nearest neighbour).
    valid : BoolTensor [B, N]
        False where a geometric transform read outside the domain (``x`` holds
        ``fill`` there).
    mask : BoolTensor [B, N] or None
        Cells masked by :class:`RandomMask` (True = masked).
    ids_keep, ids_restore : LongTensor or None
        MAE indices of the last :class:`RandomMask`.
    cell_ids : LongTensor [B, N] or None
        Cell ids of each sample after a :class:`RandomCrop`.
    """

    x: torch.Tensor
    labels: Any = None
    valid: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    ids_keep: Optional[torch.Tensor] = None
    ids_restore: Optional[torch.Tensor] = None
    cell_ids: Optional[torch.Tensor] = None


class AugmentPlan:
    """One random draw of a :class:`HealPixAugment`, ready to apply.

    Holds the fused gather maps and the per-sample parameters; apply it to
    several tensors (inputs, targets, labels) for the same transformation.
    ``params`` maps parameter names (``"angle"``, ``"scale"``, ...) to
    per-sample tensors.
    """

    def __init__(self, aug, steps, params, batch_size, device):
        self.aug = aug
        self.steps = steps
        self.params = params
        self.batch_size = batch_size
        self.device = device

    @property
    def gather_maps(self) -> List[GatherMap]:
        """The gather maps of the fused geometric groups (built on demand)."""
        return [s[1].gather_map() for s in self.steps if s[0] == "gather"]

    @property
    def groups(self) -> List["GeometricGroup"]:
        """The fused geometric groups of the plan, in order."""
        return [s[1] for s in self.steps if s[0] == "gather"]

    def apply(self, x: torch.Tensor, labels=None) -> AugmentResult:
        """Apply the plan to data ``x`` (and optionally labels / masks)."""
        aug = self.aug
        B = self.batch_size
        if x.shape[0] != B:
            raise ValueError(f"plan drawn for batch {B}, got data of batch {x.shape[0]}")
        lead = x.shape[1:-1]
        x3 = x.reshape(B, -1, x.shape[-1])
        if x3.shape[-1] != aug.grid.N:
            raise ValueError(f"expected {aug.grid.N} cells on the last axis, got {x3.shape[-1]}")
        lab = labels
        valid = torch.ones(B, x3.shape[-1], dtype=torch.bool, device=x.device)
        valid_all = True
        mask = ids_keep = ids_restore = cell_ids = None
        for step in self.steps:
            kind = step[0]
            if kind == "gather":
                grp = step[1]
                src = aug._source(x3, grp.kmax)
                need_nn = lab is not None or not valid_all or mask is not None
                x3, valid_g, gmap = grp.run(src, need_nn)
                if need_nn:
                    nn_c = gmap.nn.clamp_min(0)
                    valid = valid_g & torch.gather(valid, 1, nn_c)
                    if mask is not None:
                        mask = torch.gather(mask, 1, nn_c) & valid_g
                    lab = _map_labels(lab, lambda t, nn=gmap.nn: _nearest(t, nn, aug.label_fill))
                else:
                    valid = valid_g
                valid_all = False
            elif kind == "gauge":
                x3 = _permute_gauges(x3, step[1], step[2])
            elif kind == "noise":
                sig, seed = step[1]
                g = torch.Generator(device=x3.device).manual_seed(seed)
                noise = torch.randn(x3.shape, generator=g, device=x3.device, dtype=x3.dtype)
                x3 = x3 + sig.to(x3.device, x3.dtype).view(B, 1, 1) * noise
            elif kind == "mask":
                (tok_mask, ids_keep, ids_restore), block, fill = step[1], step[2], step[3]
                m = tok_mask.to(x3.device).repeat_interleave(block, dim=1)
                x3 = torch.where(m.unsqueeze(1), torch.full((), fill, dtype=x3.dtype, device=x3.device), x3)
                mask = m if mask is None else (mask | m)
                ids_keep, ids_restore = ids_keep.to(x3.device), ids_restore.to(x3.device)
            elif kind == "crop":
                start, n = step[1].to(x3.device), step[2]
                pos = aug.grid.perm[start[:, None] + torch.arange(n, device=x3.device)[None, :]]
                cell_ids = aug.grid.ids_sorted[start[:, None] + torch.arange(n, device=x3.device)[None, :]]
                x3 = torch.gather(x3, 2, pos.unsqueeze(1).expand(B, x3.shape[1], n))
                valid = torch.gather(valid, 1, pos)
                if mask is not None:
                    mask = torch.gather(mask, 1, pos)
                lab = _map_labels(lab, lambda t: _take(t, pos))
        out = x3.reshape(B, *lead, x3.shape[-1])
        return AugmentResult(out, lab, valid, mask, ids_keep, ids_restore, cell_ids)


def _map_labels(lab, fn):
    if lab is None:
        return None
    if isinstance(lab, dict):
        return {k: fn(v) for k, v in lab.items()}
    if isinstance(lab, (list, tuple)):
        return type(lab)(fn(v) for v in lab)
    return fn(lab)


def _take(t, pos):
    B = t.shape[0]
    t3 = t.reshape(B, -1, t.shape[-1])
    out = torch.gather(t3, 2, pos.unsqueeze(1).expand(B, t3.shape[1], pos.shape[1]))
    return out.reshape(*t.shape[:-1], pos.shape[1])


def _nearest(t, nn_idx, label_fill):
    B = t.shape[0]
    t3 = t.reshape(B, -1, t.shape[-1])
    nn_c = nn_idx.clamp_min(0).unsqueeze(1).expand(B, t3.shape[1], nn_idx.shape[1])
    out = torch.gather(t3, 2, nn_c)
    if t.dtype == torch.bool:
        fill = torch.zeros((), dtype=torch.bool, device=t.device)
    elif t.is_floating_point():
        fill = torch.full((), float("nan"), dtype=t.dtype, device=t.device)
    else:
        fill = torch.full((), label_fill, dtype=t.dtype, device=t.device)
    out = torch.where((nn_idx >= 0).unsqueeze(1), out, fill)
    return out.reshape(*t.shape[:-1], nn_idx.shape[1])


# ===========================================================================
# Compose
# ===========================================================================

class HealPixAugment(nn.Module):
    """Batched, per-sample random augmentation of HEALPix data (Compose).

    Parameters
    ----------
    transforms : sequence of transforms
        Applied in order.  Consecutive geometric transforms are fused into
        one gather when their 3-D maps (:class:`RandomShift`,
        :class:`RandomSO3`, :class:`Rotation3D`) come before their
        tangent-plane maps (:class:`RandomRotation`, :class:`RandomZoom`,
        :class:`RandomFlip`, :class:`TangentLinear`).
    level : int
        HEALPix level of the data.
    cell_ids : array or tensor of NESTED ids, optional
        The cells of a regional domain, in the order of the data (any order).
        ``None`` means the full sphere in NESTED order.
    fill : float, default NaN
        Value of output cells that read outside the domain.
    label_fill : int, default -1
        Their value for integer labels (False for boolean masks, NaN for
        float labels).
    min_support : float, default 1.0
        An output cell is inside the domain when at least this fraction of
        its bilinear weight falls on cells of the domain; its weights are
        then renormalised over those cells.  The default keeps only cells
        whose neighbours are all in the domain (second-order accurate); 0.5
        widens the valid area by about half a cell at the border, with a
        first-order error there.
    antialias : {"supersample", "pyramid", None}, default "supersample"
        Low-pass filter for zooms out (s < 1).  ``"supersample"``: every
        output cell averages ``n x n`` bilinear samples spread over its
        footprint in the input, ``n = ceil(1 / s_min)`` for the batch (a box
        filter of the size of the output cell, exact for any s; the gather
        then has ``4 n^2`` taps).  ``"pyramid"``: the input is averaged over
        NESTED children into levels ``L-1 ... L-k`` (a change of level, as
        :class:`~healpix_analyse.HealPixDown` ``mean``), and each sample reads
        level ``L - round(log2(1/s))`` with 4 bilinear taps (mip-mapping;
        exact for ``s = 2^-k``).  ``None``: no filter.
    max_supersample : int, default 4
        Cap on ``n`` for ``"supersample"``.
    max_pyramid : int, default 4
        Number of coarser levels available to ``"pyramid"``.
    geom_dtype : torch.dtype, default torch.float64
        Dtype of the geometry (positions and weights).  float64 makes the
        identity transform exact; float32 is faster on consumer GPUs.
    backend : {"auto", "triton", "torch"}
        Gather backend: Triton on CUDA for float32 / float16 / bfloat16 data
        when available, the PyTorch reference otherwise.
    deterministic : bool, default True
        Triton backward without atomics (transposed map); see
        ``healpix_analyse._augment_backend``.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        level: int,
        cell_ids=None,
        *,
        fill: float = float("nan"),
        label_fill: int = -1,
        min_support: float = 1.0,
        antialias: Optional[str] = "supersample",
        max_supersample: int = 4,
        max_pyramid: int = 4,
        geom_dtype: torch.dtype = torch.float64,
        backend: str = "auto",
        deterministic: bool = True,
    ):
        super().__init__()
        if antialias not in ("supersample", "pyramid", None):
            raise ValueError("antialias must be 'supersample', 'pyramid' or None")
        self.transforms = list(transforms)
        self.grid = _Grid(level, cell_ids, geom_dtype)
        self.fill = float(fill)
        self.label_fill = int(label_fill)
        self.min_support = float(min_support)
        self.antialias = antialias
        self.max_supersample = int(max_supersample)
        self.max_pyramid = int(max_pyramid)
        self.geom_dtype = geom_dtype
        self.backend = backend
        self.deterministic = bool(deterministic)
        self._snap = 1e-10 if geom_dtype == torch.float64 else 1e-4
        seen_crop = False
        for t in self.transforms:
            k = t.kind_for(self.grid)
            if k in ("R", "A") and seen_crop:
                raise ValueError("geometric transforms must come before RandomCrop")
            if k == "A" and self.grid.full:
                raise ValueError(f"{type(t).__name__} needs a regional domain (cell_ids)")
            seen_crop |= k == "crop"

    @property
    def device(self):
        return self.grid.xyz.device

    # -- drawing ------------------------------------------------------------
    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> AugmentPlan:
        """Draw the per-sample parameters and build the gather maps.

        The parameters are drawn from ``generator`` (the global CPU generator
        when None) and the maps are built on the device of the module (move
        it with ``.to(device)``).  The same seed gives the same plan on any
        device.
        """
        B = int(batch_size)
        gen = generator if generator is not None else torch.default_generator
        steps, params = [], {}
        grp = None
        n_cells = self.grid.N

        def flush():
            nonlocal grp
            if grp is not None:
                steps.append(("gather", self._build_map(*grp)))
                grp = None

        for i, t in enumerate(self.transforms):
            k = t.kind_for(self.grid)
            name = f"{i}:{type(t).__name__}"
            if k in ("R", "A"):
                M, p = t.draw(B, gen, self.grid)
                if grp is None:
                    grp = [torch.eye(3, dtype=_F64).repeat(B, 1, 1), torch.eye(2, dtype=_F64).repeat(B, 1, 1), False, False]
                if k == "R":
                    if grp[3]:
                        flush()
                        grp = [torch.eye(3, dtype=_F64).repeat(B, 1, 1), torch.eye(2, dtype=_F64).repeat(B, 1, 1), False, False]
                    grp[0] = grp[0] @ M
                    grp[2] = True
                else:
                    grp[1] = grp[1] @ M
                    grp[3] = True
            elif k == "gauge":
                perm, p = t.draw(B, gen, self.grid)
                steps.append(("gauge", perm.to(self.device), t.n_gauges))
            elif k == "noise":
                flush()
                v, p = t.draw(B, gen, self.grid)
                steps.append(("noise", v))
            elif k == "mask":
                flush()
                v, p = t.draw(B, gen, self.grid, n_cells)
                steps.append(("mask", v, t.block, t.fill))
            elif k == "crop":
                flush()
                v, p = t.draw(B, gen, self.grid)
                steps.append(("crop", v, t.n_cells))
                n_cells = t.n_cells
            else:
                raise TypeError(f"unknown transform {t!r}")
            for key, val in p.items():
                params[f"{name}.{key}"] = val
        flush()
        return AugmentPlan(self, steps, params, B, self.device)

    def forward(self, x: torch.Tensor, labels=None, generator: Optional[torch.Generator] = None) -> AugmentResult:
        """Draw a plan for the batch ``x`` and apply it (see :class:`AugmentResult`)."""
        if x.device != self.device:
            self.to(x.device)
        return self.sample(x.shape[0], generator).apply(x, labels)

    # -- geometry -----------------------------------------------------------
    def _lookup(self, q, k, offset=0):
        """Source positions, weights and presence of points q [..., 3].

        Bilinear lookup at level ``L - k`` (RING scheme, torch), then the
        RING ids are found among the cells of the domain at that level.  On
        the CPU the points are processed in cache-sized chunks.
        """
        order = self.grid.level - k
        index = self.grid.ring_index(k, q.device)
        shape = q.shape[:-1]
        qf = q.reshape(-1, 3)
        P = qf.shape[0]
        pos = torch.empty(P, 4, dtype=torch.long, device=q.device)
        w = torch.empty(P, 4, dtype=q.dtype, device=q.device)
        present = torch.empty(P, 4, dtype=torch.bool, device=q.device)
        step = _MAX_POINTS.get(q.device.type, 1 << 22)
        for a in range(0, P, step):
            b = min(P, a + step)
            pix, wa = bilinear_ring(qf[a:b], order)
            if index[0] == "table":
                pos[a:b] = index[1][pix]
                present[a:b] = True
            elif index[0] == "bits":
                pos[a:b] = ring_to_nest(pix, order)
                present[a:b] = True
            else:
                rs, rp = index[1], index[2]
                p = torch.searchsorted(rs, pix).clamp_max(rs.numel() - 1)
                present[a:b] = rs[p] == pix
                pos[a:b] = rp[p]
            w[a:b] = wa
        return (pos + offset).view(*shape, 4), w.view(*shape, 4), present.view(*shape, 4)

    def _build_map(self, R, A, has_r, has_a) -> "GeometricGroup":
        g = self.grid
        dev, gd = self.device, self.geom_dtype
        B = R.shape[0]
        # Footprint growth of the tangent map (largest singular value of A).
        a2 = (A ** 2).sum((1, 2))
        det = torch.linalg.det(A)
        smax = torch.sqrt((a2 + torch.sqrt((a2 ** 2 - 4 * det ** 2).clamp_min(0))) / 2)
        n_ss, levels = 1, torch.zeros(B, dtype=torch.long)
        if has_a and self.antialias == "supersample":
            m = float(smax.max())
            if m > 1 + 1e-6:  # rotations and flips: smax = 1 up to rounding
                n_ss = min(self.max_supersample, math.ceil(m - 1e-6))
        elif has_a and self.antialias == "pyramid":
            levels = torch.round(torch.log2(smax.clamp_min(1.0)) - 1e-9).long().clamp(0, self.max_pyramid)
        T = n_ss * n_ss
        kmax = int(levels.max())
        pyr = g.pyramid(kmax, dev) if kmax > 0 else []
        n_src = g.N + sum(lv[0].numel() for lv in pyr)
        if n_ss > 1:
            o = (torch.arange(n_ss, dtype=gd, device=dev) + 0.5) / n_ss - 0.5
            off = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(T, 2) * g.h
        else:
            off = torch.zeros(1, 2, dtype=gd, device=dev)
        return GeometricGroup(
            self, R.to(dev, gd).contiguous(), A.to(dev, gd).contiguous(), has_r, has_a,
            n_ss, levels, kmax, n_src, off.contiguous())

    def _torch_map(self, grp) -> GatherMap:
        """Gather map of a group, computed with torch ops (reference, CPU)."""
        g = self.grid
        dev, gd = self.device, self.geom_dtype
        B, N = grp.B, g.N
        Rd, Ad, off, has_a, has_r = grp.R, grp.A, grp.off, grp.has_a, grp.has_r
        n_ss, T, kmax = grp.n_ss, grp.T, grp.kmax
        pyr = g.pyramid(kmax, dev) if kmax > 0 else []
        n_src = grp.n_src
        c, e1, e2 = g.frame[0], g.frame[1], g.frame[2]
        lev_d = grp.levels.to(dev)

        idx = torch.empty(B, 4 * T, N, dtype=torch.int32, device=dev)
        wts = torch.empty(B, 4 * T, N, dtype=gd, device=dev)
        valid = torch.empty(B, N, dtype=torch.bool, device=dev)
        nn_idx = torch.empty(B, N, dtype=torch.long, device=dev)
        chunk = max(1, _MAX_POINTS.get(dev.type, 1 << 22) // (T * N))
        for b0 in range(0, B, chunk):
            b1 = min(B, b0 + chunk)
            bs = b1 - b0
            if has_a:
                wv = g.uv[None, None] + off[None, :, None, :]                     # [1,T,N,2]
                uvp = torch.einsum("bij,xtnj->btni", Ad[b0:b1], wv)              # [bs,T,N,2]
                q = c + uvp[..., :1] * e1 + uvp[..., 1:] * e2                    # [bs,T,N,3]
                if has_r:
                    q = torch.einsum("bij,btnj->btni", Rd[b0:b1], q)
            else:
                q = torch.einsum("bij,nj->bni", Rd[b0:b1], g.xyz).unsqueeze(1)   # [bs,1,N,3]
            # level-0 lookup (always: also gives the nearest cell for labels)
            pos, w, present = self._lookup(q, 0)
            if kmax > 0:
                lb = lev_d[b0:b1]
                for k in range(1, kmax + 1):
                    sel = lb == k
                    if bool(sel.any()):
                        pk, wk, prk = self._lookup(q[sel], k, pyr[k - 1][2])
                        pos[sel], w[sel], present[sel] = pk, wk, prk
            w = torch.where(w < self._snap, torch.zeros((), dtype=gd, device=dev), w)
            total = w.sum(-1).sum(1)                                              # [bs,N] (= T)
            wp = torch.where(present, w, torch.zeros((), dtype=gd, device=dev))
            wp = wp.permute(0, 1, 3, 2).reshape(bs, 4 * T, N)
            pp = pos.permute(0, 1, 3, 2).reshape(bs, 4 * T, N)
            s_in = wp.sum(1)
            ok = s_in >= (self.min_support - 0.1 * self._snap) * total  # relative tolerance
            ok &= s_in > 0
            wn = torch.where(ok.unsqueeze(1), wp / s_in.clamp_min(1e-300).unsqueeze(1), torch.zeros((), dtype=gd, device=dev))
            idx[b0:b1] = torch.where(wn > 0, pp, torch.full((), -1, dtype=pp.dtype, device=dev)).int()
            wts[b0:b1] = wn
            valid[b0:b1] = ok
            # nearest neighbour (labels): level-0 lookup of the centre of the cell
            if T == 1 and kmax == 0:
                p0, w0, pr0 = pos[:, 0], w[:, 0], present[:, 0]
            else:
                if n_ss % 2 == 1:
                    qc = q[:, T // 2]                       # the centre tap
                elif has_a:
                    uvc = torch.einsum("bij,nj->bni", Ad[b0:b1], g.uv)
                    qc = c + uvc[..., :1] * e1 + uvc[..., 1:] * e2
                    if has_r:
                        qc = torch.einsum("bij,bnj->bni", Rd[b0:b1], qc)
                else:
                    qc = torch.einsum("bij,nj->bni", Rd[b0:b1], g.xyz)
                p0, w0, pr0 = self._lookup(qc, 0)
            w0 = torch.where(pr0, w0, torch.full((), -1.0, dtype=gd, device=dev))
            best = w0.argmax(-1, keepdim=True)
            nn_b = torch.gather(p0, -1, best).squeeze(-1)
            found = torch.gather(w0, -1, best).squeeze(-1) >= 0
            nn_idx[b0:b1] = torch.where(found & ok, nn_b, torch.full((), -1, dtype=nn_b.dtype, device=dev))
        return GatherMap(idx, wts, valid, nn_idx, n_src)

    def _source(self, x3, kmax):
        """The input, followed by its mean pyramid when the map reads it."""
        if kmax == 0:
            return x3
        parts = [x3]
        prev = x3
        for ids_k, table, _ in self.grid.pyramid(kmax, self.device):
            present = table >= 0                                   # [N_k, 4]
            ch = prev[:, :, table.clamp_min(0)]                    # [B, C, N_k, 4]
            zero = torch.zeros((), dtype=prev.dtype, device=prev.device)
            cur = torch.where(present, ch, zero).sum(-1) / present.sum(-1).to(prev.dtype)
            parts.append(cur)
            prev = cur
        return torch.cat(parts, dim=2)


# ===========================================================================
# One fused geometric group
# ===========================================================================

class _FusedAugment(torch.autograd.Function):
    """One Triton kernel for the forward (bilinear weights from the
    per-sample parameters + gather of all channels) and one for the backward
    (the same weights recomputed + deterministic fixed-point scatter).  No
    map is kept between the two: only the per-cell scale [B, N]."""

    @staticmethod
    def forward(ctx, src, grp, need_nn):
        aug = grp.aug
        out, scale, nn, _ = fused_augment_triton(
            grp.geo(), src.contiguous(), store_nn=need_nn, fill=aug.fill)
        grp._last = (scale, grp._nn_from(nn, scale) if need_nn else None)
        ctx.grp = grp
        ctx.scale = scale
        ctx.src_dtype = src.dtype
        ctx.n_src = src.shape[2]
        return out

    @staticmethod
    def backward(ctx, gout):
        grp = ctx.grp
        if torch.is_grad_enabled():  # create_graph=True: differentiable scatter
            gm = grp.gather_map("triton")
            return gather_backward(gout, gm, True, ctx.src_dtype), None, None
        g = fused_augment_triton(grp.geo(), gout=gout, scale=ctx.scale,
                                 deterministic=grp.aug.deterministic, n_src=ctx.n_src)
        return g.to(ctx.src_dtype), None, None


class GeometricGroup:
    """Consecutive geometric transforms fused into one source map
    ``S(p) = R lift(A proj(p))`` per sample, and one gather.

    On CUDA (and ``backend != "torch"``), :meth:`run` launches one Triton
    kernel that computes the bilinear weights of each output cell from the
    per-sample ``R`` and ``A`` and gathers the channels in the same pass; the
    map (indices and weights) is written only when the backward pass, the
    labels or a composition with an earlier step need it.  Elsewhere the map
    is built with torch ops.
    """

    def __init__(self, aug, R, A, has_r, has_a, n_ss, levels, kmax, n_src, off):
        self.aug = aug
        self.R, self.A = R, A
        self.has_r, self.has_a = bool(has_r), bool(has_a)
        self.n_ss, self.T = int(n_ss), int(n_ss) ** 2
        self.levels, self.kmax, self.n_src = levels, int(kmax), int(n_src)
        self.off = off
        self.B = R.shape[0]
        self._geo = None
        self._maps = {}
        self._last = None

    @property
    def n_taps(self) -> int:
        """Entries per output cell of the gather (4 per sampling tap)."""
        return 4 * self.T

    def triton_ok(self) -> bool:
        return self.aug.backend != "torch" and triton_on(self.aug.device)

    def geo(self) -> dict:
        if self._geo is None:
            aug, g = self.aug, self.aug.grid
            dev, gd = aug.device, aug.geom_dtype
            frame_h = torch.cat([g.frame.reshape(-1),
                                 torch.tensor([g.h], dtype=gd, device=dev)]).contiguous()
            pyr = g.pyramid(self.kmax, dev) if self.kmax > 0 else []
            launches = []
            for k in range(self.kmax + 1):
                bsel = torch.nonzero(self.levels == k).reshape(-1).to(torch.int32).to(dev)
                nside = 1 << (g.level - k)
                launches.append((bsel, nside, g.theta_table(nside, gd, dev), g.nest_table(k, dev),
                                 0 if k == 0 else pyr[k - 1][2]))
            self._geo = dict(
                uv=g.uv.contiguous(), xyz=g.xyz.contiguous(), A=self.A, R=self.R, off=self.off,
                frame_h=frame_h, launches=launches, T=self.T,
                t_nn=(self.T // 2) if self.n_ss % 2 == 1 else self.T,
                has_a=self.has_a, has_r=self.has_r, min_support=aug.min_support,
                snap=aug._snap, B=self.B, N=g.N)
        return self._geo

    def _nn_from(self, nn, scale):
        """Nearest cell for labels (level 0, cell centre)."""
        if self.kmax == 0:
            return nn
        g, dev = self.aug.grid, self.aug.device
        nside = 1 << g.level
        geo0 = dict(self.geo())
        geo0.update(T=1, t_nn=0, off=torch.zeros(1, 2, dtype=self.off.dtype, device=dev),
                    launches=[(torch.arange(self.B, dtype=torch.int32, device=dev), nside,
                               g.theta_table(nside, self.aug.geom_dtype, dev),
                               g.nest_table(0, dev), 0)])
        _, _, nn0, _ = fused_augment_triton(geo0, None, store_nn=True)
        return torch.where(scale > 0, nn0, torch.full((), -1, dtype=torch.long, device=dev))

    def gather_map(self, kind: Optional[str] = None) -> GatherMap:
        """The map of the group: ``"triton"`` (fused kernel) or ``"torch"``."""
        kind = kind or ("triton" if self.triton_ok() else "torch")
        if kind not in self._maps:
            if kind == "triton":
                _, scale, nn, (idx, w) = fused_augment_triton(
                    self.geo(), None, store_map=True, store_nn=True)
                self._maps["triton"] = GatherMap(
                    idx, w * scale.unsqueeze(1), scale > 0, self._nn_from(nn, scale), self.n_src)
            else:
                self._maps["torch"] = self.aug._torch_map(self)
        return self._maps[kind]

    def run(self, src, need_nn):
        """Gather ``src`` [B, C, n_src]; returns (out, valid, map or None)."""
        aug = self.aug
        if self.triton_ok() and src.dtype in _TRITON_DTYPES and triton_on(src.device):
            out = _FusedAugment.apply(src, self, need_nn)
            scale, nn = self._last
            gm = _NNOnly(nn) if need_nn else None
            return out, scale > 0, gm
        gm = self.gather_map("triton" if self.triton_ok() else "torch")
        out = apply_gather(src, gm, aug.fill, aug.backend, aug.deterministic)
        return out, gm.valid, gm


class _NNOnly:
    """The nearest-cell part of a map (labels), from the fused kernel."""

    def __init__(self, nn):
        self.nn = nn
