"""
convol.py
=========
Gauge-equivariant spherical convolution on HEALPix maps.

The stencil geometry (rotation of the kernel grid, bilinear neighbours and
their weights) is computed once at construction, cached on disk, and stored as
the *tap geometry* ``[P, 4, G, K]``: int32 neighbour positions in the order of
the input, and their weights.  The forward and backward passes run in
``healpix_analyse._conv_backend``: fused Triton kernels on CUDA (float32,
float16, bfloat16) when Triton is installed, a PyTorch loop over the taps
otherwise.  Neither materialises the gathered ``[B, C_in, G, K, P]`` tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import healpix_geo
from healpix_analyse.healpix_interp import get_interp_weights
from healpix_analyse._ellipsoid import canonicalize_ellipsoid
from healpix_analyse._conv_backend import (
    BACKENDS,
    build_gather_map,
    build_transposed_map,
    gauge_conv,
    gauge_conv_transpose,
    tap_geometry,
)

ArrayLike = Union[np.ndarray, torch.Tensor]

# Default cache directory: $HEALPIXCONV_CACHE or ~/.cache/healpixconv
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("HEALPIXCONV_CACHE", Path.home() / ".cache" / "healpixconv")
)


# ===========================================================================
# Geometry cache helpers
# ===========================================================================

# Bump when the meaning or layout of any cached buffer changes.
_CACHE_FORMAT = 3

# Stencil points (output cells x gauges x taps) per chunk of the geometry
# build: bounds its float64 temporaries to a few hundred MB.
_GEOMETRY_CHUNK_POINTS = 1 << 21


def _geometry_dtype(dtype: torch.dtype) -> torch.dtype:
    """Precision of the geometry build: float64, whatever the layer dtype.

    The rotations, the stencil projection, ``get_interp_weights`` and the
    binding decide which 4 cells each stencil point reads.  Built in float32,
    a point that lies near a cell or face boundary can pick other cells on
    another device or with another float32 implementation (CPU and CUDA
    differ in the last bits of ``cos`` / ``matmul``), which changes the
    output by tens of percent on a regional domain.  In float64 the choice is
    stable; only the final weights are cast to the layer dtype.  The cost is
    one-off (construction, or once per disk-cache entry).
    """
    return torch.float64


def _geometry_cache_key(
    nside: int,
    kernel_sz: int,
    G: int,
    gauge_type: str,
    ref_direction,        # np.ndarray or None
    nest: bool,
    cell_ids,             # np.ndarray or None
    ellipsoid: str,
    geom_dtype: torch.dtype = torch.float32,
    stride: int = 1,
) -> str:
    """
    Return a hex digest that uniquely identifies a set of geometry parameters.

    Only the fields that affect _get_interp_weights / _bind_support_batched
    are included, plus the precision of the build (float64 since #109), which
    is also the dtype of the cached weights.  The device is excluded.  The
    ``cell_ids`` enter as their raw int64 bytes, in the input order: the
    cached geometry depends on that order, and hashing the bytes costs far
    less than serialising a list.  ``stride`` enters only when it is not 1,
    so that the keys of stride-1 layers are those of earlier versions.
    """
    ids = None if cell_ids is None else np.ascontiguousarray(
        np.asarray(cell_ids, dtype=np.int64).ravel(), dtype="<i8"
    )
    state: dict = {
        "cache_format": _CACHE_FORMAT,
        "nside":      nside,
        "kernel_sz":  kernel_sz,
        "G":          G,
        "gauge_type": gauge_type,
        "nest":       nest,
        "ellipsoid":  ellipsoid,
        "geom_dtype": str(geom_dtype).replace("torch.", ""),
        # round to 9 decimal places to avoid float noise from different
        # construction paths that produce the same effective direction
        "ref_direction": (
            None if ref_direction is None
            else np.round(np.asarray(ref_direction, dtype=np.float64), 9).tolist()
        ),
        "n_cells": None if ids is None else int(ids.size),
    }
    if stride != 1:
        state["stride"] = int(stride)
    h = hashlib.sha256(json.dumps(state, sort_keys=True).encode())
    if ids is not None:
        h.update(memoryview(ids).cast("B"))
    return h.hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"healpixconv_{key}.pt"


def _load_geometry_cache(
    cache_dir: Path,
    key: str,
    device: torch.device,
) -> "dict | None":
    """
    Try to load precomputed geometry buffers from disk.

    Returns a dict with keys tap_idx, tap_w, sort_order, inv_order and
    support (None when absent), or None if the cache does not exist or is
    corrupt.  ``tap_w`` and ``support`` keep the dtype they were built in
    (part of the key).
    """
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return {
            "tap_idx":    data["tap_idx"].to(device=device, dtype=torch.int32),  # [P, 4, G, K]
            "tap_w":      data["tap_w"].to(device=device),                       # [P, 4, G, K]
            "sort_order": data["sort_order"].to(device=device),                  # [K] long
            "inv_order":  data["inv_order"].to(device=device),                   # [K] long
            # [G, K*P], the in-domain fraction of each stencil point
            "support":    (data["support"].to(device=device)
                           if "support" in data else None),
        }
    except Exception:
        # Corrupt or outdated cache — ignore and recompute
        return None


def _save_geometry_cache(
    cache_dir: Path,
    key: str,
    tap_idx:    torch.Tensor,   # [P, 4, G, K] int32
    tap_w:      torch.Tensor,   # [P, 4, G, K]
    sort_order: torch.Tensor,   # [K]
    inv_order:  torch.Tensor,   # [K]
    support:    "torch.Tensor | None" = None,   # [G, K*P]
) -> None:
    """Save geometry buffers to disk as CPU tensors (weights in their own dtype)."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = _cache_path(cache_dir, key).with_suffix(".tmp")
        data = {
            "tap_idx":    tap_idx.cpu().to(dtype=torch.int32),
            "tap_w":      tap_w.cpu(),
            "sort_order": sort_order.cpu(),
            "inv_order":  inv_order.cpu(),
        }
        if support is not None:
            data["support"] = support.cpu()
        torch.save(data, tmp)
        tmp.replace(_cache_path(cache_dir, key))   # atomic rename
    except Exception:
        # Non-fatal: cache write failure just means the next run recomputes
        pass


# ===========================================================================
# I/O helpers  (unchanged)
# ===========================================================================

def _prepare_input_conv(x, device, dtype):
    """Normalise input to [B, C, N]. Returns (tensor, is_numpy, was_1d, was_2d)."""
    is_numpy = isinstance(x, np.ndarray)
    t = torch.as_tensor(x, dtype=dtype, device=device) if is_numpy \
        else x.to(device=device, dtype=dtype)

    was_1d = (t.ndim == 1)
    was_2d = (t.ndim <= 2)

    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 2:
        t = t.unsqueeze(1)
    elif t.ndim != 3:
        raise ValueError(
            f"Input must have shape [N], [B, N] or [B, C, N]; got {tuple(t.shape)}"
        )
    return t, is_numpy, was_1d, was_2d


def _mask_as_bcn(mask, t):
    """Pixel weights of ``missing="local_mean"`` as ``[B, C, N]``, like ``t``.

    Accepts ``[N]`` (every sample and channel), ``[B, N]`` (per sample, every
    channel) or ``[B, C, N]``; B and C may be 1.  Any other shape raises, so
    that a per-channel ``[C, N]`` mask is never read as per-sample.  Non-finite
    values count as 0 and the weights are clamped to [0, 1].
    """
    B, C, N = t.shape
    m = torch.as_tensor(mask, device=t.device, dtype=t.dtype)
    shape = tuple(m.shape)
    if m.dim() == 1:
        m = m.view(1, 1, -1)
    elif m.dim() == 2:
        m = m.unsqueeze(1)
    if (m.dim() != 3 or m.shape[-1] != N or m.shape[0] not in (1, B)
            or m.shape[1] not in (1, C)):
        raise ValueError(
            f"mask must have shape [N], [B, N] or [B, C, N] with N={N}, B={B} "
            f"(or 1), C={C} (or 1); got {shape}."
        )
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).clamp(0.0, 1.0)
    return m.expand(B, C, N)


def _restore_output_conv(t, is_numpy, was_1d):
    """Convert [B, C_out, N] back to the original shape / type."""
    if was_1d:
        t = t.squeeze(0)
        if t.shape[0] == 1:
            t = t.squeeze(0)
    if is_numpy:
        return t.detach().cpu().numpy()
    return t


# ===========================================================================
# Geometry helpers  (unchanged)
# ===========================================================================

def _build_rotation_matrices(th, ph, G, gauge_type, device, dtype, ref_direction=None):
    """
    Build rotation matrices [K, G, 3, 3] carrying the North-Pole kernel
    grid to each of the K target pixels with G gauge angles.

    R_total = R_gauge(alpha_g) @ Rz(phi) @ Ry(theta)

    Gauge types::

        "phi"
            alpha_base = 0 everywhere.  Singularities at the geographic poles.

        "cosmo"
            Same singularities as "phi" but the gauge angle flips sign across
            the equator to match the cosmological convention.

        "projected_ref"
            ref_direction is a single unit vector r (shape (3,)).
            alpha_base = atan2(r_proj·e_φ,  r_proj·e_θ)
            where r_proj = r - (r·n)·n  is the projection of r onto the
            tangent plane at n.
            Singularities at {+r, −r}  (antipodal pair), total index = 2.

        "two_ref"
            ref_direction is TWO unit vectors (shape (2, 3)): r1 and r2.
            The gauge angle is defined as the argument of the complex PRODUCT
            of the two projected tangent vectors:

                z_j(n) = (r_j_proj·e_θ) + i·(r_j_proj·e_φ)   j = 1, 2
                alpha_base(n) = arg(z1(n) · z2(n))
                              = atan2(Re(z1)·Im(z2) + Im(z1)·Re(z2),
                                      Re(z1)·Re(z2) − Im(z1)·Im(z2))

            Using the complex product is equivalent to arg(z1) + arg(z2), but
            avoids two separate atan2 calls and never wraps independently.

            Singularity structure (Poincaré-Hopf, total index must = 2):
              - +r1, −r1 : index +1 each   (zeros of z1)
              - +r2, −r2 : index +1 each   (zeros of z2)
              - N-Pole, S-Pole : index −1 each  (base-frame singularities
                                 absorbed by z1·z2, but each of z1 and z2 winds
                                 −1 there, so the poles get net index 1 − 2 = −1)
              Total: 4×(+1) + 2×(−1) = 2  (OK)

            In practice the four user-chosen bad points {+r1, −r1, +r2, −r2}
            are index +1 (well-behaved to avoid); the poles become index −1
            (hyperbolic singularity — keep them away from the domain of interest
            or over regions the network need not be accurate in).
    """
    th = np.asarray(th, dtype=np.float64).reshape(-1)
    ph = np.asarray(ph, dtype=np.float64).reshape(-1)
    K  = th.shape[0]

    th_t = torch.as_tensor(th, device=device, dtype=dtype)
    ph_t = torch.as_tensor(ph, device=device, dtype=dtype)
    ct, st = torch.cos(th_t), torch.sin(th_t)
    cp, sp = torch.cos(ph_t), torch.sin(ph_t)

    R_base = torch.zeros(K, 3, 3, device=device, dtype=dtype)
    R_base[:, 0, 0] =  cp * ct;  R_base[:, 0, 1] = -sp;  R_base[:, 0, 2] =  cp * st
    R_base[:, 1, 0] =  sp * ct;  R_base[:, 1, 1] =  cp;  R_base[:, 1, 2] =  sp * st
    R_base[:, 2, 0] = -st;       R_base[:, 2, 1] = 0.;   R_base[:, 2, 2] =  ct

    n = R_base[:, :, 2]
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if gauge_type == "cosmo":
        is_south   = th_t > math.pi / 2
        alpha_base = torch.where(is_south,  ph_t, -ph_t)
        sign_g     = torch.where(is_south,
                                 -torch.ones_like(th_t),
                                  torch.ones_like(th_t))

    elif gauge_type in ("projected_ref", "two_ref"):
        # Local orthonormal tangent frame at each pixel
        e_th  = torch.stack([ ct * cp,  ct * sp, -st],               dim=1)  # [K, 3]
        e_ph  = torch.stack([-sp,        cp,      torch.zeros_like(st)], dim=1)  # [K, 3]
        n_pix = torch.stack([ st * cp,  st * sp,  ct],               dim=1)  # [K, 3]

        def _tangent_complex(r_vec):
            """Project unit vector r onto the tangent plane → (r_eth, r_eph)."""
            r  = torch.as_tensor(r_vec, device=device, dtype=dtype)
            r  = r / r.norm().clamp_min(1e-12)
            r_dot_n  = (r[None, :] * n_pix).sum(dim=1, keepdim=True)   # [K, 1]
            r_proj   = r[None, :] - r_dot_n * n_pix                     # [K, 3]
            r_eth    = (r_proj * e_th).sum(dim=1)                        # [K]
            r_eph    = (r_proj * e_ph).sum(dim=1)                        # [K]
            return r_eth, r_eph                                           # Re(z), Im(z)

        if gauge_type == "projected_ref":
            # Single reference vector → singularities at {+r, −r}
            if ref_direction is None:
                ref_direction = [1.0, 0.0, 0.0]
            eth, eph   = _tangent_complex(ref_direction)
            alpha_base = torch.atan2(eph, eth)

        else:  # "two_ref"
            # Two reference vectors → singularities at {+r1,−r1,+r2,−r2}
            # alpha = arg(z1 · z2)  via complex product, never wraps
            eth1, eph1 = _tangent_complex(ref_direction[0])   # Re(z1), Im(z1)
            eth2, eph2 = _tangent_complex(ref_direction[1])   # Re(z2), Im(z2)
            re_prod    = eth1 * eth2 - eph1 * eph2            # Re(z1·z2)
            im_prod    = eth1 * eph2 + eph1 * eth2            # Im(z1·z2)
            alpha_base = torch.atan2(im_prod, re_prod)

        sign_g = torch.ones_like(th_t)

    else:  # "phi"
        alpha_base = torch.zeros_like(th_t)
        sign_g     = torch.ones_like(th_t)

    g_shifts = torch.arange(G, device=device, dtype=dtype) * (math.pi / G)
    alpha_g  = alpha_base[:, None] + sign_g[:, None] * g_shifts[None, :]
    ca = torch.cos(alpha_g);  sa = torch.sin(alpha_g)

    n_g  = n[:, None, :].expand(K, G, 3)
    nxg, nyg, nzg = n_g[..., 0], n_g[..., 1], n_g[..., 2]

    K_skew = torch.zeros(K, G, 3, 3, device=device, dtype=dtype)
    K_skew[..., 0, 1] = -nzg;  K_skew[..., 0, 2] =  nyg
    K_skew[..., 1, 0] =  nzg;  K_skew[..., 1, 2] = -nxg
    K_skew[..., 2, 0] = -nyg;  K_skew[..., 2, 1] =  nxg

    outer  = n_g.unsqueeze(-1) * n_g.unsqueeze(-2)
    I      = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    R_gauge = (
        I      * ca.view(K, G, 1, 1)
        + K_skew * sa.view(K, G, 1, 1)
        + outer  * (1.0 - ca).view(K, G, 1, 1)
    )

    R_tot = torch.matmul(R_gauge, R_base[:, None, :, :].expand(K, G, 3, 3))
    return R_tot   # [K, G, 3, 3]


def _local_kernel_grid(kernel_sz, nside):
    """
    Build a kernel_sz × kernel_sz grid of unit vectors at the North Pole.
    Returns np.ndarray [P=kernel_sz^2, 3].
    """
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    alpha_pix = np.sqrt(4 * np.pi / (12 * nside ** 2))

    dtheta = np.sqrt(xx ** 2 + yy ** 2).ravel() * alpha_pix
    dphi   = np.arctan2(yy, xx).ravel()

    x = np.sin(dtheta) * np.cos(dphi)
    y = np.sin(dtheta) * np.sin(dphi)
    z = np.cos(dtheta)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


# ===========================================================================
# Helper 1 — _get_interp_weights  (unchanged)
# ===========================================================================

def _get_interp_weights(nside, vecs, nest, device, dtype):
    """
    Compute bilinear-interpolation neighbours for M direction vectors.

    Parameters
    ----------
    vecs : torch.Tensor [M, 3]

    Returns
    -------
    idx_t : LongTensor [4, M]
    w_t   : Tensor     [4, M]
    """
    vn    = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(vn[:, 2].clamp(-1., 1.))
    phi   = torch.atan2(vn[:, 1], vn[:, 0]) % (2.0 * math.pi)

    lon_np = np.rad2deg(phi.detach().cpu().numpy())
    lat_np = 90.0 - np.rad2deg(theta.detach().cpu().numpy())

    depth = int(math.log2(nside))
    i_np, w_np = get_interp_weights(lon_np, lat_np, depth)

    return (
        torch.as_tensor(i_np.T.copy(), device=device, dtype=torch.long),
        torch.as_tensor(w_np.T.copy(), device=device, dtype=dtype),
    )


# ===========================================================================
# Helper 2 — _bind_support_batched  (unchanged)
# ===========================================================================

def _bind_support_batched(
    idx_t:       torch.Tensor,   # [G, 4, K*P]
    w_t:         torch.Tensor,   # [G, 4, K*P]
    ids_sorted:  torch.Tensor,   # [N_in] sorted pixel ids
    kernel_sz:   int,
    K:           int,
    P:           int,
    device:      torch.device,
    return_support: bool = False,
):
    """
    Vectorised binding of stencil neighbours for ALL G gauges at once.

    Returns
    -------
    pos_safe : [G, 4, K*P]   column indices in ids_sorted (0 for absent)
    w_norm   : [G, 4, K*P]   renormalised weights
    support  : [G, K*P]      only with ``return_support=True``: the fraction
                             of the bilinear weight of each stencil point
                             that falls inside the domain, computed before
                             a fully absent point is bound to the centre
                             (0 for such a point, 1 inside the domain).
    """
    G = idx_t.shape[0]
    M = K * P
    N_sorted = ids_sorted.numel()

    pos = torch.searchsorted(ids_sorted, idx_t.reshape(-1)).view(G, 4, M)

    in_range = pos < N_sorted
    cmp_vals = torch.full_like(idx_t, -1)
    cmp_vals[in_range] = ids_sorted[pos[in_range]]
    present  = cmp_vals == idx_t          # [G, 4, M]
    support  = (w_t * present).sum(dim=1) if return_support else None

    p_ref  = (kernel_sz // 2) * (kernel_sz + 1)
    empty  = ~present.any(dim=1)          # [G, M]

    if empty.any():
        k_id     = torch.div(
            torch.arange(M, device=device), P, rounding_mode="floor"
        )
        ref_cols = (k_id * P + p_ref)

        for g in range(G):
            empty_g = empty[g]
            if not empty_g.any():
                continue

            ref_g = ref_cols[empty_g]
            idx_t[g, :, empty_g] = idx_t[g, :, ref_g]
            w_t  [g, :, empty_g] = w_t  [g, :, ref_g]

            idx_e  = idx_t[g, :, empty_g].reshape(-1)
            pos_e  = torch.searchsorted(ids_sorted, idx_e)
            valid_e = pos_e < N_sorted
            pos_e_c = pos_e.clamp(0, max(N_sorted - 1, 0))
            pres_e  = valid_e & (ids_sorted[pos_e_c] == idx_e)
            present[g, :, empty_g] = pres_e.view(4, -1)
            pos    [g, :, empty_g] = pos_e_c.view(4, -1)

    w      = w_t * present
    colsum = w.sum(dim=1, keepdim=True)
    zero_c = colsum == 0

    if zero_c.any():
        zero_mask = zero_c.squeeze(1)
        w[:, 0, :] = torch.where(
            zero_mask, present[:, 0, :].to(w.dtype), w[:, 0, :]
        )
        colsum = w.sum(dim=1, keepdim=True)

    w_norm   = w / colsum.clamp_min(1e-12)
    pos_safe = torch.where(present, pos, torch.zeros_like(pos))
    if return_support:
        return pos_safe, w_norm, support               # [G,4,K*P] x2, [G,K*P]
    return pos_safe, w_norm                             # [G, 4, K*P] each


def _stencil_major(t: torch.Tensor) -> torch.Tensor:
    """Tap-major ``[P, 4, G, K]`` -> stencil-major ``[4, G, K, P]`` (a view)."""
    return t.permute(1, 2, 3, 0)


# ===========================================================================
# HealPixConv
# ===========================================================================

class HealPixConv(nn.Module):
    """
    Gauge-equivariant spherical convolution on HEALPix maps.

    Algorithm
    ---------
    The convolution is precomputed at construction time in three stages.

    **Stage A — Kernel + rotation**

    A ``kernel_sz × kernel_sz`` grid of P unit vectors is defined at the
    **North Pole** (z = 1) with angular spacing equal to one HEALPix pixel
    width.  This is the stencil template; it never changes.

    For every target pixel k (colatitude θ_k, longitude φ_k) and every
    gauge g, a rotation matrix is built::

        R_total[k,g] = R_gauge(α_g)  @  Rz(φ_k)  @  Ry(θ_k)
                       └─ gauge roll ┘  └────── carry N.Pole → pixel k ──┘

    Each of the P kernel vectors is then rotated into its position on the
    sphere around pixel k::

        rotated[k, g, p] = R_total[k, g] @ vec_pole[p]   ∈ ℝ³

    **Stage B — Bilinear binding**

    For each of the K × G × P rotated directions, ``get_interp_weights``
    returns the 4 nearest HEALPix neighbours and their bilinear weights.
    These indices and weights are stored as buffers (precomputed once).

    **Stage C — Forward pass (data → kernel)**

    *It is the data that is brought to the (fixed) kernel, not the kernel
    that moves.*  At inference::

        x_interp[b, c, g, k, p] = Σ_{j=0}^{3}  w[g,j,k,p] · x[b, c, nbr[g,j,k,p]]

    i.e. the signal value at rotated stencil point p of pixel k (gauge g)
    is obtained by bilinear interpolation of the input map.  Then::

        y[b, g·C_out + o, k] = Σ_{c,p}  W[g, c, o, p] · x_interp[b, c, g, k, p]

    The learned kernel ``W[G, C_in, C_out, P]`` is a fixed set of weights;
    its P positions implicitly correspond to the stencil template at the
    North Pole.  Gauge equivariance is achieved because the *same* kernel
    is applied after the stencil has been rotated by the gauge.

    Gauge types and singularities
    ------------------------------
    Every smooth gauge on S² has exactly two singular points (hairy-ball
    theorem).  The three built-in gauges place them differently:

    ``"phi"``
        Singularities at the **geographic poles** (θ = 0, π).
        ``α_base = 0`` everywhere → kernel always meridian-aligned.
        Simple and fast; bad only at the poles.

    ``"cosmo"``
        Singularities also at the poles, but the gauge flips sign across
        the equator to match the cosmological convention.

    ``"projected_ref"``
        A reference vector **r** is projected onto the tangent plane; the
        gauge angle is ``atan2(r·e_φ, r·e_θ)``.  The two singularities are
        the antipodal points where **r** is parallel to the surface normal,
        i.e. where the map pixel lies exactly on the direction of **r**::

            singularity₁ = (lon_s, lat_s)
            singularity₂ = (lon_s + 180°, -lat_s)   ← antipode

        You control where those two points land by choosing **r**.
        Use ``singularity_lonlat=(lon_s, lat_s)`` to specify the first
        singularity in geographic coordinates; the reference vector is
        computed automatically::

            r = [cos(lat_s)·cos(lon_s),
                 cos(lat_s)·sin(lon_s),
                 sin(lat_s)]

        Practical choices:
          - Ocean model  → place singularities over land (e.g. Amazon basin
            and its antipode in the Indian Ocean).
          - Atmosphere   → place singularities over open ocean (e.g. central
            Pacific and its antipode in the Atlantic).
          - Full sphere  → place singularities at the geographic poles
            (``singularity_lonlat=(0, 90)``), which reproduces the ``"phi"``
            gauge but with the smooth-everywhere property off the poles.

    Parameters
    ----------
    level : int
        Grid4Earth/HEALPix level (integer >= 0).  The internal HEALPix
        resolution is ``nside = 2**level``.
    in_channels : int
        Number of input channels C_in.
    out_channels : int
        Number of output channels per gauge C_out.
        Total output channels = n_gauges × out_channels.
    kernel_sz : int, default 3
        Odd integer ≥ 1.  P = kernel_sz² stencil points.
    n_gauges : int, default 1
        Number of gauge orientations G.  Each gauge rotates the stencil
        by an additional angle of π/G relative to the previous one.
    gauge_type : {"phi", "cosmo", "projected_ref"}, default "phi"
        Gauge convention.  See *Gauge types and singularities* above.
    singularity_lonlat : (float, float) or None
        Only used when ``gauge_type="projected_ref"``.
        Geographic coordinates ``(longitude_deg, latitude_deg)`` of the
        **first** singularity point.  The second singularity is placed
        automatically at the antipodal point.
        Overrides ``ref_direction`` when both are provided.
        Example — singularities over the Himalayas and the central Pacific::

            singularity_lonlat=(84.0, 28.0)   # Kathmandu region

    ref_direction : array-like (3,) or None
        Low-level alternative to ``singularity_lonlat``: pass the reference
        3-D unit vector **r** directly.  Ignored when ``singularity_lonlat``
        is provided.  Default ``[1, 0, 0]`` → singularities at
        (lon=0°, lat=0°) and (lon=180°, lat=0°).
    cell_ids : array-like or None
        Pixel indices (NESTED) for partial-sky.  None = full sphere.
    nest : bool, default True
        NESTED pixel ordering.
    use_norm : bool, default False
        Apply GroupNorm + ReLU after convolution.
    ellipsoid : str, default "WGS84"
        Reference ellipsoid for healpix_geo coordinate conversions.
    device, dtype : torch device / dtype.
        Initial placement; ``module.to(...)`` moves and casts the layer
        afterwards like any other module.
    backend : {"auto", "triton", "torch"}, default "auto"
        Compute backend of the forward and backward passes.  ``"auto"``
        runs the fused Triton kernels on CUDA for float32, float16 and
        bfloat16 when the optional ``triton`` package is installed, and the
        PyTorch tap loop otherwise (CPU, float64).  Can be changed after
        construction (``conv.backend = "torch"``).  ``missing="local_mean"``
        always runs the PyTorch reference path (gathered stencil + matrix
        product) and ignores this setting.  A double backward
        (``create_graph=True``, e.g. a gradient penalty) runs the backward
        of the PyTorch tap loop on every device: the Triton kernels are not
        differentiable themselves.
    deterministic : bool, default True
        Applies to the Triton backend only.  ``True`` computes the gradient with respect to
        the input by a gather on the transposed stencil map: no atomics, the
        same bits at every run, and a training step 1.6-2.2x shorter than
        with atomics on an A100.  The cost is memory: the map is built at the
        first backward pass (0.02-0.2 s on an A100) and kept on the GPU,
        about the size of the geometry buffers again (0.2-2.6 GiB in the
        configurations of ``benchmarks/benchmark_healpix_conv.py``).
        ``False`` scatters the gradient with fp32 atomics instead: no extra
        memory, but the summation order, so the last bits, vary between
        runs.  ``torch.use_deterministic_algorithms(True)`` selects the
        gather even with ``False``.  The PyTorch backend ignores this
        option.  On CPU it is deterministic.  On CUDA without Triton, and for
        a double backward, its ``index_add_`` is deterministic only under
        ``torch.use_deterministic_algorithms(True)``.
    missing : {"centre", "local_mean"}, default "centre"
        How a stencil value is formed when its pixels are missing.  Each
        stencil value is a bilinear interpolation of 4 pixels.

        ``"centre"`` (the existing behaviour): neighbours outside
        ``cell_ids`` are dropped and the others renormalised; a stencil
        point whose 4 neighbours are all outside takes the value of the
        centre pixel.  NaN values are not handled and propagate.

        ``"local_mean"``: NaN values (and pixels with ``mask == 0``) are
        missing exactly like pixels outside ``cell_ids``.  A stencil point
        with at least one valid neighbour takes the renormalised bilinear
        interpolation of its valid neighbours; a point with none takes the
        local mean of the valid stencil values of the same output pixel
        (weighted by ``local_mean_weights``).  Every pixel gets a finite
        output, and so do the gradients.  Costs a second gather (the mask).
        The geometry of this mode is always built in float64, then cast to
        ``dtype``: the choice between the two rules depends on the valid
        fraction of each stencil point, and a float32 build (``acos`` near
        the poles) gives points just outside the domain spurious fractions
        (5e-4 at level 6).  Every layer builds its geometry in float64
        (see :func:`_geometry_dtype`), so the cache entry is shared.
    local_mean_weights : array-like [P] or None
        Non-negative weights of the stencil points in the local mean of
        ``missing="local_mean"``.  None = uniform.
    stride : {1, 2}, default 1
        ``2`` gives the strided convolution of an encoder: the stencil is
        evaluated only at the centres of the parent cells (level
        ``level - 1``), so the layer computes N/4 outputs instead of N and
        replaces a convolution followed by :class:`~healpix_analyse.down.HealPixDown`.
        The output lies on the parent cells, in the order of
        :attr:`cell_ids_out` (sorted, as for ``HealPixDown``; on a regional
        domain, every parent with at least one child in ``cell_ids``).

        Definition: output ``k`` is the stride-1 convolution evaluated at the
        centre of parent ``k``, with the same stencil and kernel.  The
        stencil keeps the fine-level spacing (one level-``level`` cell) and
        is centred on the parent centre, the vertex shared by its 4
        children, with the gauge of that point; its points are interpolated
        bilinearly from the fine cells as at stride 1.  The receptive field
        and the learned kernel are therefore those of the stride-1 layer at
        level ``level``, and weights move between ``stride=1`` and
        ``stride=2`` layers unchanged.  For a smooth
        field it differs from the stride-1 layer followed by
        ``HealPixDown(mode="mean")`` (the mean of the 4 children) at first
        order in the cell size: 3e-2 of the amplitude at level 4, halved at
        each finer level.  A
        spacing of two fine cells would skip fine cells between the stencil
        points (aliasing), and a stencil centred on one child would shift
        each output by half a fine cell in a direction that changes from
        one base face to the next.  Needs ``level >= 1``, ``kernel_sz >= 3``
        and ``missing="centre"``: with ``kernel_sz=1`` the only stencil
        point is the parent centre, which lies on the ring of its East and
        West children, so its bilinear weights are about 0.5 / 0.5 on those
        two and the layer would not be the mean of the 4 children
        (``HealPixDown(mode="mean")`` is).

    Examples
    --------
    Ocean model — singularities over Africa and its antipode (Pacific):

    >>> conv = HealPixConv(
    ...     level=6, in_channels=1, out_channels=16,
    ...     gauge_type="projected_ref",
    ...     singularity_lonlat=(20.0, 5.0),   # Gulf of Guinea coast
    ... )

    Atmosphere — singularities over the Pacific and Indian oceans:

    >>> conv = HealPixConv(
    ...     level=6, in_channels=1, out_channels=16,
    ...     gauge_type="projected_ref",
    ...     singularity_lonlat=(-160.0, 0.0),  # central Pacific
    ... )
    """

    # Class defaults, so that a layer pickled whole before these options
    # existed still runs forward() and repr() after unpickling.
    missing = "centre"
    backend = "auto"
    deterministic = True
    stride = 1
    _cell_ids_in = None
    _tmap = None
    _gmap = None
    _lm_stencil = None

    def __init__(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[tuple[float, float]] = None,
        ref_direction=None,
        cell_ids=None,
        nest: bool = True,
        use_norm: bool = False,
        device=None,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float32,
        cache_dir: "Path | str | None" = _DEFAULT_CACHE_DIR,
        missing: str = "centre",
        local_mean_weights=None,
        backend: str = "auto",
        deterministic: bool = True,
        stride: int = 1,
    ):
        super().__init__()

        if missing not in ("centre", "local_mean"):
            raise ValueError("missing must be 'centre' or 'local_mean'.")
        self.missing = missing
        if isinstance(stride, bool) or stride not in (1, 2):
            raise ValueError("stride must be 1 or 2.")
        self.stride = int(stride)
        if self.stride == 2 and missing != "centre":
            raise ValueError("stride=2 supports missing='centre' only.")
        if self.stride == 2 and int(kernel_sz) == 1:
            raise ValueError(
                "stride=2 needs kernel_sz >= 3: with kernel_sz=1 the only "
                "stencil point is the parent centre, which lies on the ring of "
                "its East and West children, so the bilinear weights are about "
                "0.5 / 0.5 on those two children and ignore the other two. Use "
                "HealPixDown(mode='mean') for the mean of the 4 children."
            )
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
        self.backend = backend
        self.deterministic = bool(deterministic)
        self._tmap = None   # transposed stencil maps, built on demand
        self._gmap = None

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype  = dtype
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)

        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0.")
        if self.stride == 2 and int(level) < 1:
            raise ValueError("stride=2 needs level >= 1.")
        self.level        = int(level)
        self.level_out    = self.level - (self.stride - 1)
        self.nside        = 2 ** self.level
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_sz    = int(kernel_sz)
        self.G            = int(max(1, n_gauges))
        self.P            = self.kernel_sz * self.kernel_sz
        self.nest         = bool(nest)

        if self.kernel_sz < 1 or self.kernel_sz % 2 == 0:
            raise ValueError("kernel_sz must be a positive odd integer.")
        if gauge_type not in ("phi", "cosmo", "projected_ref", "two_ref"):
            raise ValueError(
                "gauge_type must be 'phi', 'cosmo', 'projected_ref', or 'two_ref'."
            )
        self.gauge_type = gauge_type

        # ------------------------------------------------------------------
        # Resolve reference direction(s).
        # ------------------------------------------------------------------
        if singularity_lonlat is not None:
            if gauge_type == "projected_ref":
                lon_s_deg, lat_s_deg = float(singularity_lonlat[0]), float(singularity_lonlat[1])
                lon_s = np.radians(lon_s_deg)
                lat_s = np.radians(lat_s_deg)
                rd = np.array([
                    np.cos(lat_s) * np.cos(lon_s),
                    np.cos(lat_s) * np.sin(lon_s),
                    np.sin(lat_s),
                ], dtype=np.float64)
                self.ref_direction = rd / np.linalg.norm(rd)
                self.singularity_1 = (lon_s_deg % 360.0, lat_s_deg)
                self.singularity_2 = ((lon_s_deg + 180.0) % 360.0, -lat_s_deg)

            elif gauge_type == "two_ref":
                if len(singularity_lonlat) != 2:
                    raise ValueError(
                        "For gauge_type='two_ref', singularity_lonlat must be a "
                        "sequence of exactly two (lon, lat) pairs, e.g. "
                        "[(lon1, lat1), (lon2, lat2)]."
                    )

                def _lonlat_to_vec(lon_deg, lat_deg):
                    lon_r, lat_r = np.radians(lon_deg), np.radians(lat_deg)
                    v = np.array([
                        np.cos(lat_r) * np.cos(lon_r),
                        np.cos(lat_r) * np.sin(lon_r),
                        np.sin(lat_r),
                    ], dtype=np.float64)
                    return v / np.linalg.norm(v)

                def _vec_to_lonlat(v):
                    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
                    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
                    return lon, lat

                r1 = _lonlat_to_vec(*singularity_lonlat[0])
                r2 = _lonlat_to_vec(*singularity_lonlat[1])
                self.ref_direction = np.stack([r1, r2], axis=0)

                self.singularity_1  = _vec_to_lonlat(r1)
                self.singularity_1b = _vec_to_lonlat(-r1)
                self.singularity_2  = _vec_to_lonlat(r2)
                self.singularity_2b = _vec_to_lonlat(-r2)

            else:
                raise ValueError(
                    "singularity_lonlat is only valid for "
                    "gauge_type='projected_ref' or 'two_ref'."
                )

        elif ref_direction is not None:
            if gauge_type == "two_ref":
                rd = np.asarray(ref_direction, dtype=np.float64)
                if rd.shape != (2, 3):
                    raise ValueError(
                        "For gauge_type='two_ref', ref_direction must have "
                        f"shape (2, 3); got {rd.shape}."
                    )
                rd[0] /= np.linalg.norm(rd[0])
                rd[1] /= np.linalg.norm(rd[1])
                self.ref_direction = rd
                def _v2ll(v):
                    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
                    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
                    return lon, lat
                self.singularity_1  = _v2ll(rd[0])
                self.singularity_1b = _v2ll(-rd[0])
                self.singularity_2  = _v2ll(rd[1])
                self.singularity_2b = _v2ll(-rd[1])
            else:
                rd = np.asarray(ref_direction, dtype=np.float64).ravel()
                self.ref_direction = rd / np.linalg.norm(rd)
                r = self.ref_direction
                lat_s = np.degrees(np.arcsin(np.clip(r[2], -1.0, 1.0)))
                lon_s = np.degrees(np.arctan2(r[1], r[0])) % 360.0
                self.singularity_1 = (lon_s, lat_s)
                self.singularity_2 = ((lon_s + 180.0) % 360.0, -lat_s)
        else:
            if gauge_type == "two_ref":
                self.ref_direction = np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
                )
                self.singularity_1  = (0.0,  0.0)
                self.singularity_1b = (180.0, 0.0)
                self.singularity_2  = (90.0,  0.0)
                self.singularity_2b = (270.0, 0.0)
            else:
                self.ref_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                self.singularity_1 = (0.0,   0.0)
                self.singularity_2 = (180.0, 0.0)

        # ---- pixel domain ----
        self.partial = cell_ids is not None
        if self.partial:
            ids_np = np.asarray(cell_ids, dtype=np.int64).ravel()
            if ids_np.size == 0:
                raise ValueError("cell_ids must not be empty.")
            if np.unique(ids_np).size != ids_np.size:
                raise ValueError("cell_ids must contain unique identifiers.")
            npix = 12 * self.nside**2
            if np.any(ids_np < 0) or np.any(ids_np >= npix):
                raise ValueError(
                    f"cell_ids contain identifiers outside level={self.level}."
                )
        else:
            ids_np = np.arange(12 * self.nside ** 2, dtype=np.int64)
        self.K = len(ids_np)            # input cells
        self._cell_ids_in = ids_np if self.partial else None

        # ---- stencil centres (output cells) ----
        if self.stride == 1:
            ids_out = ids_np
        elif self.partial:
            ids_out = np.unique(ids_np // 4).astype(np.int64)   # as HealPixDown
        else:
            ids_out = np.arange(12 * 4 ** self.level_out, dtype=np.int64)
        self._cell_ids_out = ids_out
        self.K_out = len(ids_out)       # output cells (= K for stride 1)

        # ------------------------------------------------------------------
        # OPT 4 — Geometry cache
        # _get_interp_weights + _bind_support_batched are the dominant costs
        # at init time (~0.8 s for nside=64, G=4).  Their result depends only
        # on the constructor parameters, never on the data.  We therefore
        # compute them once, serialise to disk, and reload on subsequent runs
        # (~5 ms).
        #
        # Cache key = SHA-256 of all geometry-affecting parameters.
        # Cache file = <cache_dir>/healpixconv_<key>.pt
        # Written atomically (tmp → rename) so a crash cannot corrupt the cache.
        # ------------------------------------------------------------------
        _cache_dir_resolved = (
            Path(cache_dir) if cache_dir is not None else None
        )
        # missing="local_mean" switches rules on the valid fraction of each
        # stencil point, which a float32 build gets wrong just outside the
        # domain (acos near the poles): that geometry is built in float64
        geom_dtype = (torch.float64 if self.missing == "local_mean"
                      else _geometry_dtype(self.dtype))
        _cache_key = _geometry_cache_key(
            nside      = self.nside,
            kernel_sz  = self.kernel_sz,
            G          = self.G,
            gauge_type = self.gauge_type,
            ref_direction = getattr(self, "ref_direction", None),
            nest       = self.nest,
            cell_ids   = cell_ids,
            ellipsoid  = self.ellipsoid,
            geom_dtype = geom_dtype,
            stride     = self.stride,
        )
        _cached = (
            _load_geometry_cache(_cache_dir_resolved, _cache_key, self.device)
            if _cache_dir_resolved is not None else None
        )

        if (_cached is not None and self.missing == "local_mean"
                and _cached.get("support") is None):
            _cached = None   # cache without the support: recompute once

        if _cached is not None:
            # ---- Fast path: restore buffers from disk ----
            _support     = _cached["support"]
            sort_order_t = _cached["sort_order"]
            inv_order_t  = _cached["inv_order"]
            tap_idx      = _cached["tap_idx"]
            tap_w        = _cached["tap_w"]

        else:
            # ---- Slow path: full geometry computation ----

            # Stage A: rotation matrices + stencil projection, centred on
            # the output cells (the parents for stride 2); the stencil keeps
            # the spacing of the input level
            lon, lat = healpix_geo.nested.healpix_to_lonlat(
                ids_out.tolist(), self.level_out,
                ellipsoid=self.ellipsoid,
            )
            th = np.deg2rad(90.0 - np.asarray(lat, dtype=np.float64))
            ph = np.deg2rad(np.asarray(lon, dtype=np.float64))
            vec_t = torch.as_tensor(
                _local_kernel_grid(self.kernel_sz, self.nside),
                device=self.device, dtype=geom_dtype,
            )  # [P, 3]
            K_out, G, P = self.K_out, self.G, self.P

            # Stage B: binding
            ids_sorted   = np.sort(ids_np)
            ids_sorted_t = torch.as_tensor(
                ids_sorted, device=self.device, dtype=torch.long
            )

            sort_order = np.argsort(ids_np)
            inv_order  = np.empty_like(sort_order)
            inv_order[sort_order] = np.arange(len(sort_order))
            sort_order_t = torch.as_tensor(
                sort_order, dtype=torch.long, device=self.device
            )
            inv_order_t = torch.as_tensor(
                inv_order, dtype=torch.long, device=self.device
            )

            # Stages A and B by chunks of output cells, written into the
            # final tap-major buffers: the float64 temporaries of the build
            # (rotations, rotated stencil, interpolation, binding) are
            # bounded by the chunk, so the peak memory is the final geometry
            # plus one chunk.  Every stencil point depends only on its own
            # centre (the fallback of an empty point reads the centre tap of
            # the same cell), so the chunking changes the result only by the
            # rounding of the batched rotation matmul (dense stencil to 1e-13).
            tap_idx = torch.empty(P, 4, G, K_out, dtype=torch.int32, device=self.device)
            tap_w = torch.empty(P, 4, G, K_out, dtype=geom_dtype, device=self.device)
            _support = torch.empty(G, K_out, P, dtype=geom_dtype, device=self.device)
            chunk = max(1, _GEOMETRY_CHUNK_POINTS // (G * P))
            for k0 in range(0, K_out, chunk):
                k1 = min(K_out, k0 + chunk)
                kc = k1 - k0
                # Stage A: rotation matrices + stencil projection
                R_tot = _build_rotation_matrices(
                    th[k0:k1], ph[k0:k1], G, self.gauge_type, self.device, geom_dtype,
                    ref_direction=self.ref_direction,
                )  # [kc, G, 3, 3]
                flat = torch.einsum("kgij,pj->kgpi", R_tot, vec_t).reshape(-1, 3)
                del R_tot
                idx_flat, w_flat = _get_interp_weights(
                    self.nside, flat, self.nest, self.device, geom_dtype
                )  # [4, kc*G*P]
                del flat
                idx_c = idx_flat.view(4, kc, G, P).permute(2, 0, 1, 3).reshape(G, 4, kc * P)
                w_c = w_flat.view(4, kc, G, P).permute(2, 0, 1, 3).reshape(G, 4, kc * P)
                del idx_flat, w_flat
                # Stage B: binding (idx_c / w_c are fresh copies, modified in place)
                pos_c, wn_c, sup_c = _bind_support_batched(
                    idx_c, w_c, ids_sorted_t, self.kernel_sz, kc, P, self.device,
                    return_support=True,
                )  # [G, 4, kc*P] each, [G, kc*P]
                del idx_c, w_c
                # Tap-major layout [P, 4, G, K], positions in the input
                # order: the sort permutation is folded into the indices once
                # here, so the forward gathers the input as it comes.
                ti, tw = tap_geometry(
                    pos_c.permute(1, 0, 2), wn_c.permute(1, 0, 2),
                    sort_order_t, G, kc, P,
                )
                tap_idx[..., k0:k1] = ti
                tap_w[..., k0:k1] = tw
                _support[:, k0:k1] = sup_c.view(G, kc, P)
                del pos_c, wn_c, sup_c, ti, tw
            _support = _support.view(G, K_out * P)

            # Persist to disk for the next run
            if _cache_dir_resolved is not None:
                _save_geometry_cache(
                    _cache_dir_resolved, _cache_key,
                    tap_idx, tap_w, sort_order_t, inv_order_t,
                    support=_support,
                )

        # The geometry is a function of the constructor arguments, rebuilt
        # (or reloaded from the disk cache) at construction: it is not part
        # of the state_dict.  Only the two [K] permutations are, as before.
        self.register_buffer("_sort_order", sort_order_t)
        self.register_buffer("_inv_order",  inv_order_t)
        self.register_buffer("_tap_idx", tap_idx.contiguous(), persistent=False)
        self.register_buffer(
            "_tap_w", tap_w.to(dtype=self.dtype).contiguous(), persistent=False
        )

        # ---- missing="local_mean": domain fraction and local-mean weights ----
        # Not persistent, so the state_dict of the default mode is unchanged.
        if self.missing == "local_mean":
            self.register_buffer(
                "_support",
                _support.to(device=self.device, dtype=self.dtype).contiguous(),
                persistent=False,
            )
            if local_mean_weights is None:
                lmw = torch.ones(self.P, device=self.device, dtype=self.dtype)
            else:
                lmw = torch.as_tensor(
                    np.asarray(local_mean_weights, dtype=np.float64).ravel(),
                    device=self.device, dtype=self.dtype,
                )
                if lmw.numel() != self.P or bool((lmw < 0).any()) or float(lmw.sum()) <= 0:
                    raise ValueError(
                        f"local_mean_weights must hold {self.P} non-negative "
                        "values with a positive sum."
                    )
            self.register_buffer("_local_mean_w", lmw, persistent=False)
        elif local_mean_weights is not None:
            raise ValueError("local_mean_weights needs missing='local_mean'.")

        # ---- learnable kernel and bias ----
        self.weight = nn.Parameter(
            torch.empty(
                self.G, self.in_channels, self.out_channels, self.P,
                device=self.device, dtype=self.dtype,
            )
        )
        nn.init.kaiming_uniform_(
            self.weight.view(
                self.G * self.in_channels, self.out_channels * self.P
            ),
            a=0., mode="fan_in", nonlinearity="relu",
        )
        self.bias = nn.Parameter(
            torch.zeros(
                self.G * self.out_channels,
                device=self.device, dtype=self.dtype,
            )
        )

        # ---- optional GroupNorm + ReLU ----
        self.use_norm = bool(use_norm)
        if self.use_norm:
            C_tot = self.G * self.out_channels
            g = min(8, C_tot)
            while C_tot % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, C_tot, device=self.device, dtype=self.dtype)
        else:
            self.norm = None

        self.to(self.device)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Cell ids of the output, at level :attr:`level_out` (NESTED).

        The input cells (``cell_ids`` in their order, or every cell) for
        ``stride=1``; the parent cells, sorted, for ``stride=2``.
        """
        if self._cell_ids_out is None:
            # not AttributeError: nn.Module.__getattr__ would hide the message
            raise RuntimeError(
                "this regional layer was pickled before #109 and does not hold "
                "its cell ids (earlier versions kept only their sort "
                "permutation). Rebuild it with the same arguments, or load its "
                "state_dict into a new layer, to get cell_ids_out."
            )
        return self._cell_ids_out

    # ------------------------------------------------------------------
    # Device, dtype and state_dict
    # ------------------------------------------------------------------

    def _apply(self, fn, *args, **kwargs):
        # module.to(...), .cuda(), .double(), ...: the layer follows its
        # parameters and buffers; self.device / self.dtype only report them.
        result = super()._apply(fn, *args, **kwargs)
        self.device = self.weight.device
        self.dtype = self.weight.dtype
        self._tmap = None
        self._gmap = None
        self._lm_stencil = None
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # state_dicts written before the tap geometry hold the stencil as
        # _pos_safe / _w_norm.  The geometry is rebuilt from the constructor
        # arguments, so those entries are dropped rather than reported as
        # unexpected keys.
        for name in ("_pos_safe", "_w_norm"):
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def _pos_safe(self) -> torch.Tensor:
        """Legacy view ``[4, G, K*P]``: neighbour positions in the sorted cell
        order (built on access from the tap geometry)."""
        pos = self._inv_order[self._tap_idx.long()]              # [P, 4, G, K]
        return _stencil_major(pos).reshape(4, self.G, -1)

    @property
    def _w_norm(self) -> torch.Tensor:
        """Legacy view ``[4, G, K*P]`` of the renormalised bilinear weights."""
        return _stencil_major(self._tap_w).reshape(4, self.G, -1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tmap"] = None          # derived caches, rebuilt on demand
        state["_gmap"] = None
        state["_lm_stencil"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # a layer pickled whole before the tap geometry holds the stencil as
        # the buffers _pos_safe / _w_norm: convert them
        bufs = self._buffers
        if "_pos_safe" in bufs and "_tap_idx" not in bufs:
            pos, w = bufs.pop("_pos_safe"), bufs.pop("_w_norm")
            idx, tw = tap_geometry(pos, w, bufs["_sort_order"], self.G, self.K, self.P)
            self.register_buffer("_tap_idx", idx, persistent=False)
            self.register_buffer("_tap_w", tw, persistent=False)
        # a layer pickled whole before stride existed is a stride-1 layer.
        # A regional one did not store its cell ids (only their sort
        # permutation), so they cannot be rebuilt here: cell_ids_out then
        # raises with an explanation; everything else works as before.
        if "K_out" not in self.__dict__:
            self.K_out = self.K
            self.level_out = self.level
            self._cell_ids_out = None if self.partial else np.arange(self.K, dtype=np.int64)

    def _local_mean_taps(self, device, dtype):
        """Stencil of ``missing="local_mean"`` in the layout of its gather (cached).

        Returns int64 positions ``[4, G*K*P]`` in the input order, (g, k, p)
        with p fastest, and the weights ``[4, G, K, P]``.  Built at the first
        call and kept, as the ``_pos_safe`` / ``_w_norm`` buffers were before
        the tap layout, so that the forward pass does not convert the indices
        again; ``module.to(...)`` and pickling drop it.
        """
        c = self._lm_stencil
        if c is None or c[0].device != device or c[1].dtype != dtype:
            idx = _stencil_major(self._tap_idx.to(device=device))             # [4, G, K, P]
            w = _stencil_major(self._tap_w.to(device=device, dtype=dtype))
            c = (idx.reshape(4, -1).long(), w.contiguous())
            self._lm_stencil = c
        return c

    def _transposed_map(self):
        """Transposed stencil map of the deterministic Triton backward (cached)."""
        if self._tmap is None or self._tmap[0].device != self._tap_idx.device:
            self._tmap = build_transposed_map(self._tap_idx, self._tap_w, self.K)
        return self._tmap

    def _gather_map(self):
        """Transposed stencil map of the deterministic PyTorch backward on CUDA (cached)."""
        if self._gmap is None or self._gmap[0][0].device != self._tap_idx.device:
            self._gmap = build_gather_map(self._tap_idx, self._tap_w, self.K)
        return self._gmap

    def _backend_kwargs(self):
        return dict(backend=self.backend, deterministic=self.deterministic,
                    transposed=self._transposed_map, gather_map=self._gather_map)

    # ------------------------------------------------------------------
    # Kernel management
    # ------------------------------------------------------------------

    def set_kernel(self, W, bias=None, requires_grad=False):
        """
        Replace the learnable kernel with a fixed (or re-initialised) array.

        The values are copied in the dtype of the layer: a float64 layer keeps
        float64 kernels.

        Parameters
        ----------
        W : array-like
            Shape [C_in, C_out, P]       — same kernel broadcast over all G.
            Shape [G, C_in, C_out, P]    — per-gauge kernels.
        bias : array-like or None
            Shape [G * C_out].  None resets bias to zero.
        requires_grad : bool, default False
        """
        dev, dt = self.weight.device, self.weight.dtype
        if torch.is_tensor(W):
            W_t = W.detach().to(device=dev, dtype=dt)
        else:
            W_t = torch.as_tensor(np.asarray(W, dtype=np.float64), device=dev).to(dt)
        if W_t.ndim == 3:
            W_t = W_t.unsqueeze(0).expand((self.G,) + tuple(W_t.shape))
        expected = tuple(self.weight.shape)
        if tuple(W_t.shape) != expected:
            raise ValueError(
                f"W must have shape {expected} or {expected[1:]}; "
                f"got {tuple(W_t.shape)}."
            )
        with torch.no_grad():
            self.weight.copy_(W_t)
            if bias is not None:
                if torch.is_tensor(bias):
                    b_t = bias.detach().reshape(-1)
                else:
                    b_t = torch.as_tensor(np.asarray(bias, dtype=np.float64).ravel())
                self.bias.copy_(b_t.to(device=dev, dtype=dt))
            else:
                self.bias.zero_()
        self.weight.requires_grad_(requires_grad)
        self.bias.requires_grad_(requires_grad)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, mask=None):
        """
        Apply gauge-equivariant spherical convolution.

        Parameters
        ----------
        x : array-like, shape [N], [B, N] or [B, C_in, N]
        mask : array-like or None
            Only with ``missing="local_mean"``: weights in [0, 1] of the
            pixels (0 = missing).  Shape ``[N]`` (every sample and channel),
            ``[B, N]`` (one mask per sample, for every channel) or
            ``[B, C_in, N]`` (one mask per channel); B and C_in may be 1.
            Any other shape raises.  NaN values of ``x`` are always
            missing.  None = ``isfinite(x)``.

        Returns
        -------
        y : same type, shape [G*C_out, N_out] or [B, G*C_out, N_out]
            ``N_out = N`` for ``stride=1``; for ``stride=2`` the parent
            cells, in the order of :attr:`cell_ids_out`.

        Notes
        -----
        The input is moved to the device of the layer and cast to its dtype.
        The computation runs on the backend selected by ``self.backend``
        (see the class docstring); neither backend materialises the gathered
        ``[B, C_in, G, K, P]`` tensor.  ``missing="local_mean"`` runs the
        PyTorch reference path instead, which does.  Row k of the geometry is
        pixel ``cell_ids[k]``, so the output follows the input order.
        """
        t, is_numpy, was_1d, _ = _prepare_input_conv(
            x, self._tap_w.device, self.weight.dtype
        )
        B, C_in, N = t.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {C_in}."
            )
        if N != self.K:
            raise ValueError(f"Expected {self.K} pixels, got {N}.")

        if self.missing == "local_mean":
            # NaN / masked pixels completed like pixels outside cell_ids
            gathered, _ = self._local_mean_stencil(t, mask)
            y = self._contract_gathered(gathered)
        else:
            if mask is not None:
                raise ValueError("mask= needs missing='local_mean'.")
            y = gauge_conv(
                t, self.weight, self.bias, self._tap_idx, self._tap_w,
                **self._backend_kwargs(),
            )

        if self.use_norm and self.norm is not None:
            y = F.relu(self.norm(y), inplace=True)

        return _restore_output_conv(y, is_numpy, was_1d)

    def _contract_gathered(self, gathered):
        """Reference contraction of the gathered stencil values ``[B, C_in, G, K, P]``.

        ``y[b, g*C_out + o, k] = sum_{c,p} W[g, c, o, p] gathered[b, c, g, k, p]
        + bias`` by one batched matrix product, as before the fused backends;
        used by ``missing="local_mean"``.
        """
        B, C_in, G, K, P = gathered.shape
        C_out = self.out_channels
        g_mat = gathered.permute(2, 0, 3, 1, 4).reshape(G, B * K, C_in * P)
        W_mat = self.weight.permute(0, 1, 3, 2).reshape(G, C_in * P, C_out)
        y = torch.bmm(g_mat, W_mat)                                 # [G, B*K, C_out]
        y = y.view(G, B, K, C_out).permute(1, 0, 3, 2).reshape(B, G * C_out, K)
        return y + self.bias.view(1, -1, 1)

    def _local_mean_stencil(self, t, mask=None):
        """Completed stencil values for ``missing="local_mean"``.

        ``t`` is ``[B, C, K]`` (already prepared).  Returns ``(z, cov)``, both
        ``[B, C, G, K, P]``: the stencil values, and the valid fraction of
        each stencil point (its bilinear weight on valid pixels inside the
        domain, in [0, 1]).
        """
        if self.missing != "local_mean":
            raise RuntimeError("needs missing='local_mean'.")
        B, C, N = t.shape
        G, P, K = self.G, self.P, self.K
        # positions in the input order of t: the same elements and weights as
        # the sorted-order stencil of the reference implementation
        pos, wn = self._local_mean_taps(t.device, t.dtype)
        sup = self._support.to(device=t.device, dtype=t.dtype).view(1, 1, G, K, P)

        v = torch.isfinite(t).to(t.dtype)
        if mask is not None:
            v = v * _mask_as_bcn(mask, t)
        xz = torch.where(v > 0, t, torch.zeros_like(t))
        vx = xz * v

        ivx = t.new_zeros(B, C, G, K, P)
        iv  = t.new_zeros(B, C, G, K, P)
        for j in range(4):
            pj = pos[j]
            wj = wn[j].view(1, 1, G, K, P)
            ivx = ivx + vx.index_select(2, pj).view(B, C, G, K, P) * wj
            iv  = iv + v.index_select(2, pj).view(B, C, G, K, P) * wj

        # wn is renormalised over the neighbours inside the domain; sup puts
        # back the fraction outside it (0 for a point with no neighbour in it)
        cov = iv * sup
        a = self._local_mean_w.to(device=t.device, dtype=t.dtype).view(1, 1, 1, 1, P)
        num = (a * ivx * sup).sum(-1, keepdim=True)
        den = (a * cov).sum(-1, keepdim=True)
        has_den = den > 1e-6
        xbar = torch.where(has_den, num / torch.where(has_den, den, torch.ones_like(den)),
                           torch.zeros_like(den))

        # renormalised interpolation where a valid neighbour has weight,
        # else the local mean (sup is exact: the geometry of this mode is
        # built in float64, so no spurious fraction crosses the threshold)
        has = cov > 1e-6
        z_direct = ivx / torch.where(has, iv, torch.ones_like(iv))
        z = torch.where(has, z_direct, xbar.expand_as(z_direct))
        return z, cov

    def completed_stencil(self, x, mask=None):
        """Stencil values and their valid fraction (``missing="local_mean"``).

        Parameters
        ----------
        x : array-like, shape [N], [B, N] or [B, C, N]
        mask : array-like or None
            Weights in [0, 1] of the pixels, shape ``[N]``, ``[B, N]`` or
            ``[B, C, N]`` as in :meth:`forward`; NaN values of ``x`` are
            always missing.

        Returns
        -------
        z : torch.Tensor [B, C, G, N, P]
            The values the kernel weights multiply: renormalised bilinear
            interpolation of the valid pixels, or the local mean for a
            stencil point with no valid neighbour.
        cov : torch.Tensor [B, C, G, N, P]
            The valid fraction of each stencil point, in [0, 1].  With tap
            weights t_p (real or complex),
            ``(cov * abs(t)**2).sum(-1) / (abs(t)**2).sum()`` is the
            coverage e of the kernel at each pixel, in [0, 1], as used by
            :class:`~healpix_analyse.wavelet_spectrum.HealPixWaveletSpectrum`.
            The ``abs(t)**2`` weights keep it defined for a zero-sum
            (band-pass) kernel.
        """
        t, _, _, _ = _prepare_input_conv(x, self.device, self.dtype)
        if t.shape[-1] != self.K:
            raise ValueError(f"Expected {self.K} pixels, got {t.shape[-1]}.")
        return self._local_mean_stencil(t, mask)

    def singularity_info(self) -> str:
        """
        Return a human-readable description of the gauge singularities.

        For "phi" and "cosmo": singularities are always at the geographic poles.
        For "projected_ref": two antipodal singularities at plus/minus r.
        For "two_ref": four singularities at {plus/minus r1, plus/minus r2}
        (index +1 each) plus index-(-1) singularities at the geographic poles.
        """
        if self.gauge_type in ("phi", "cosmo"):
            return (
                f"gauge_type='{self.gauge_type}': singularities fixed at "
                "the geographic poles (lat=+90° and lat=-90°)."
            )
        if self.gauge_type == "projected_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            return (
                f"gauge_type='projected_ref':\n"
                f"  singularity 1 : lon={s1[0]:+.2f}°  lat={s1[1]:+.2f}°  (index +1)\n"
                f"  singularity 2 : lon={s2[0]:+.2f}°  lat={s2[1]:+.2f}°  (index +1, antipode)\n"
                f"  ref_direction : {self.ref_direction.tolist()}"
            )
        # "two_ref"
        s1, s1b = self.singularity_1, self.singularity_1b
        s2, s2b = self.singularity_2, self.singularity_2b
        return (
            f"gauge_type='two_ref':\n"
            f"  singularity 1  : lon={s1[0]:+.2f}°  lat={s1[1]:+.2f}°  (index +1, user-defined)\n"
            f"  singularity 1b : lon={s1b[0]:+.2f}°  lat={s1b[1]:+.2f}°  (index +1, antipode of 1)\n"
            f"  singularity 2  : lon={s2[0]:+.2f}°  lat={s2[1]:+.2f}°  (index +1, user-defined)\n"
            f"  singularity 2b : lon={s2b[0]:+.2f}°  lat={s2b[1]:+.2f}°  (index +1, antipode of 2)\n"
            f"  N/S poles      : index -1 each (side-effect, keep outside domain of interest)\n"
            f"  ref_directions : r1={self.ref_direction[0].tolist()}\n"
            f"                   r2={self.ref_direction[1].tolist()}"
        )

    def extra_repr(self):
        base = (
            f"level={self.level}, nside={self.nside}, in={self.in_channels}, out={self.out_channels}, "
            f"kernel_sz={self.kernel_sz}, P={self.P}, G={self.G}, "
            f"gauge={self.gauge_type!r}, partial={self.partial}"
        )
        if self.stride != 1:
            base += f", stride={self.stride}"
        if self.missing != "centre":
            base += f", missing={self.missing!r}"
        if self.gauge_type == "projected_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            base += (
                f", sing1=({s1[0]:.1f}°,{s1[1]:.1f}°)"
                f", sing2=({s2[0]:.1f}°,{s2[1]:.1f}°)"
            )
        elif self.gauge_type == "two_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            base += (
                f", sing1=({s1[0]:.1f}°,{s1[1]:.1f}°)"
                f", sing2=({s2[0]:.1f}°,{s2[1]:.1f}°)"
            )
        return base


# ===========================================================================
# HealPixConvTranspose
# ===========================================================================

class HealPixConvTranspose(HealPixConv):
    """Transposed gauge-equivariant convolution: the adjoint of a strided :class:`HealPixConv`.

    The decoder counterpart of ``HealPixConv(stride=2)``.  With the same
    geometry arguments (``level``, ``kernel_sz``, ``n_gauges``, the gauge,
    ``cell_ids``, ``stride``) and the same weights, it is the exact adjoint
    of that layer::

        <conv(x) - conv.bias, y> = <x, convT(y) - convT.bias>

    It reads data on the parent cells (level ``level - 1``, in the order of
    :attr:`cell_ids_in`, the ``cell_ids_out`` of the strided layer) and
    writes on the fine cells (level ``level``, in the order of ``cell_ids``):
    each parent spreads its value over the fine cells its stencil reads,
    with the bilinear weights of that stencil, weighted by the kernel.  It
    reuses the geometry, the disk cache and the kernels of
    :class:`HealPixConv`: its forward pass is the input gradient of the
    strided layer, its input gradient the strided forward, and its weight
    gradient the weight gradient kernel with the roles of the two tensors
    exchanged.  Deterministic by default, like :class:`HealPixConv`.

    Parameters
    ----------
    level : int
        The fine (output) level, as for the strided layer it mirrors.
    in_channels : int
        Input channels *per gauge*: the input has ``n_gauges * in_channels``
        channels, gauge-major, as a :class:`HealPixConv` with
        ``out_channels=in_channels`` writes them.
    out_channels : int
        Output channels (the adjoint sums over the gauges).
    stride : {2, 1}, default 2
        ``1`` gives the adjoint of a stride-1 :class:`HealPixConv` (input and
        output on the same cells).
    kernel_sz, n_gauges, gauge_type, singularity_lonlat, ref_direction, cell_ids, nest, use_norm, device, ellipsoid, dtype, cache_dir, backend, deterministic
        As in :class:`HealPixConv`; ``cell_ids`` are the fine cells.
        ``use_norm`` applies GroupNorm + ReLU over the ``out_channels``.

    Attributes
    ----------
    weight : Parameter [G, out_channels, in_channels, P]
        The layout of the weight of ``HealPixConv(in_channels=out_channels,
        out_channels=in_channels)``: ``convT.weight = conv.weight`` ties the
        two layers.
    bias : Parameter [out_channels]

    Examples
    --------
    >>> down = HealPixConv(6, 8, 16, n_gauges=2, stride=2)
    >>> up = HealPixConvTranspose(6, 16, 8, n_gauges=2)
    >>> x = torch.randn(4, 8, 12 * 4**6)
    >>> h = down(x)                       # [4, 32, 12 * 4**5]
    >>> up(h).shape                       # [4, 8, 12 * 4**6]
    """

    def __init__(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[tuple[float, float]] = None,
        ref_direction=None,
        cell_ids=None,
        nest: bool = True,
        use_norm: bool = False,
        device=None,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float32,
        cache_dir: "Path | str | None" = _DEFAULT_CACHE_DIR,
        backend: str = "auto",
        deterministic: bool = True,
        stride: int = 2,
    ):
        # the geometry and the weight [G, out, in, P] of the layer it is the
        # adjoint of: HealPixConv(in=out_channels, out=in_channels)
        super().__init__(
            level, out_channels, in_channels, kernel_sz=kernel_sz, n_gauges=n_gauges,
            gauge_type=gauge_type, singularity_lonlat=singularity_lonlat,
            ref_direction=ref_direction, cell_ids=cell_ids, nest=nest, use_norm=False,
            device=device, ellipsoid=ellipsoid, dtype=dtype, cache_dir=cache_dir,
            backend=backend, deterministic=deterministic, stride=stride,
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bias = nn.Parameter(
            torch.zeros(self.out_channels, device=self.device, dtype=self.dtype)
        )
        self.use_norm = bool(use_norm)
        if self.use_norm:
            C_tot = self.out_channels
            g = min(8, C_tot)
            while C_tot % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, C_tot, device=self.device, dtype=self.dtype)

    @property
    def cell_ids_in(self) -> np.ndarray:
        """Cell ids of the input: the parent cells (sorted) for ``stride=2``."""
        return self._cell_ids_out

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Cell ids of the output, at level ``level`` (the ``cell_ids`` given)."""
        if not self.partial:
            return np.arange(self.K, dtype=np.int64)
        return self._cell_ids_in

    def forward(self, x, mask=None):
        """
        Apply the transposed convolution.

        Parameters
        ----------
        x : array-like, shape [N_in], [B, N_in] or [B, G*C_in, N_in]
            Gauge-major channels on the cells :attr:`cell_ids_in` (the 1-D
            and 2-D forms hold one channel, as for :class:`HealPixConv`).

        Returns
        -------
        y : same type, shape [C_out, N] or [B, C_out, N]
            On the cells ``cell_ids``, in their order.
        """
        if mask is not None:
            raise ValueError("HealPixConvTranspose has no mask argument.")
        t, is_numpy, was_1d, _ = _prepare_input_conv(
            x, self._tap_w.device, self.weight.dtype
        )
        B, C, N = t.shape
        if C != self.G * self.in_channels:
            raise ValueError(
                f"Expected {self.G * self.in_channels} input channels "
                f"(n_gauges * in_channels), got {C}."
            )
        if N != self.K_out:
            raise ValueError(f"Expected {self.K_out} pixels, got {N}.")
        y = gauge_conv_transpose(
            t, self.weight, self.bias, self._tap_idx, self._tap_w, k_in=self.K,
            **self._backend_kwargs(),
        )
        if self.use_norm and self.norm is not None:
            y = F.relu(self.norm(y), inplace=True)
        return _restore_output_conv(y, is_numpy, was_1d)

    def completed_stencil(self, x, mask=None):
        raise NotImplementedError("completed_stencil is a HealPixConv(missing='local_mean') method.")

    def extra_repr(self):
        return "transposed, " + super().extra_repr()
