"""Compute backends of :mod:`healpix_analyse.augment`.

Every geometric augmentation is one gather: each output cell reads J source
cells of the input (4 bilinear neighbours per sampling tap) and sums them
with weights::

    out[b, c, i] = sum_j w[b, j, i] * src[b, c, idx[b, j, i]]

This module holds the two halves of that operation.

Geometry on the device
    :func:`bilinear_nested` returns, for unit vectors given as a torch tensor
    (on any device), the 4 NESTED cells and the 4 bilinear weights of the
    HEALPix interpolation.  It is a line-by-line torch transcription of the
    NumPy port of ``T_Healpix_Base::get_interpol`` in
    :func:`healpix_analyse.healpix_interp.get_interp_weights` (RING scheme),
    followed by a torch transcription of ``ring2xyf`` / ``xyf2nest`` for the
    RING -> NESTED conversion (``healpix_geo.ring.to_nested`` on the CPU).
    The tests check both against the NumPy / healpix-geo versions.  No
    geometry is rebuilt on the CPU for a new random draw.

Gather
    ``"torch"`` (CPU, float64, and the reference everywhere): a loop over
    the J taps with ``torch.gather``; autograd gives the backward.
    ``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation):
    one fused kernel for the forward.  The backward with respect to the data
    is, by default (``deterministic=True``), a gather on the transposed map
    (for each source cell, its incoming (output cell, weight) entries in CSR
    form, built once per map with a stable sort): no atomics and the same
    result at every run.  ``deterministic=False`` scatters with fp32
    atomics.  A backward with ``create_graph=True`` runs the torch path.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Optional

import torch

try:  # Triton is an optional extra
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = [
    "BACKENDS",
    "triton_available",
    "ring_interp",
    "ring_to_nest",
    "bilinear_ring",
    "bilinear_nested",
    "triton_on",
    "ring_theta_table",
    "fused_augment_triton",
    "GatherMap",
    "apply_gather",
    "gather_torch",
]

BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def triton_available() -> bool:
    """True when the Triton gather can run (Triton importable and CUDA up)."""
    return _HAS_TRITON and torch.cuda.is_available()


# ===========================================================================
# HEALPix bilinear interpolation in torch (RING get_interpol + RING -> NEST)
# ===========================================================================

def _ring_above(z: torch.Tensor, nside: int) -> torch.Tensor:
    az = z.abs()
    r_eq = torch.trunc(nside * (2.0 - 1.5 * z))
    r_pol = torch.trunc(nside * torch.sqrt((3.0 * (1.0 - az)).clamp_min(0.0)))
    r_pol = torch.where(z > 0, r_pol, 4 * nside - r_pol - 1)
    return torch.where(az <= 2.0 / 3.0, r_eq, r_pol).long()


def _ring_info(ring: torch.Tensor, nside: int, fdtype: torch.dtype):
    """startpix, ringpix, theta, shifted of RING rings ``1 <= ring < 4*nside``."""
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    fact1 = 2.0 / (3.0 * nside)
    fact2 = 1.0 / (3.0 * nside * nside)

    northring = torch.where(ring > 2 * nside, 4 * nside - ring, ring)
    cap = northring < nside
    nr = northring.to(fdtype)
    tmp = nr * nr * fact2
    th_cap = torch.atan2(torch.sqrt((tmp * (2.0 - tmp)).clamp_min(0.0)), 1.0 - tmp)
    th_eq = torch.acos(((2 * nside - nr) * fact1).clamp(-1.0, 1.0))
    theta = torch.where(cap, th_cap, th_eq)
    ringpix = torch.where(cap, 4 * northring, torch.full_like(northring, 4 * nside))
    shifted = cap | (((northring - nside) & 1) == 0)
    startpix = torch.where(
        cap, 2 * northring * (northring - 1), ncap + (northring - nside) * 4 * nside
    )
    south = northring != ring
    theta = torch.where(south, math.pi - theta, theta)
    startpix = torch.where(south, npix - startpix - ringpix, startpix)
    return startpix, ringpix, theta, shifted


def _ring_row(phi, sp, nr, shifted):
    dphi = (2.0 * math.pi) / nr.to(phi.dtype)
    hs = shifted.to(phi.dtype) * 0.5
    tmp = phi / dphi - hs
    # tmp >= -0.5, so floor() equals the C++ "tmp<0 ? I(tmp)-1 : I(tmp)".
    i1f = torch.floor(tmp)
    w1 = (phi - (i1f + hs) * dphi) / dphi
    i1 = i1f.long()
    i2 = i1 + 1
    i1 = torch.where(i1 < 0, i1 + nr, i1)
    i2 = torch.where(i2 >= nr, i2 - nr, i2)
    return sp + i1, sp + i2, 1.0 - w1, w1


def ring_interp(z: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor, nside: int):
    """Torch port of ``T_Healpix_Base::get_interpol`` (RING scheme).

    Parameters
    ----------
    z, theta, phi : torch.Tensor, same shape
        ``cos(theta)``, colatitude in radians, longitude in ``[0, 2 pi)``.
        ``z`` is passed separately so that it can come straight from a unit
        vector (more accurate near the poles than ``cos(acos(z))``).
    nside : int

    Returns
    -------
    pix : LongTensor [..., 4]  RING pixels.
    wgt : Tensor [..., 4]      bilinear weights (rows sum to 1).
    """
    npix = 12 * nside * nside
    fdtype = theta.dtype
    ir1 = _ring_above(z, nside)
    ir2 = ir1 + 1
    lo, hi = 1, 4 * nside - 1
    sp1, nr1, th1, sh1 = _ring_info(ir1.clamp(lo, hi), nside, fdtype)
    sp2, nr2, th2, sh2 = _ring_info(ir2.clamp(lo, hi), nside, fdtype)
    p0, p1, w0, w1 = _ring_row(phi, sp1, nr1, sh1)
    p2, p3, w2, w3 = _ring_row(phi, sp2, nr2, sh2)

    north = ir1 == 0
    south = (ir2 == 4 * nside) & ~north

    # Interior: linear in theta between the two rings.
    wt_mid = (theta - th1) / (th2 - th1)
    # North cap: the "upper ring" is the 4 polar pixels of ring 1, opposite.
    wt_n = theta / th2
    fac_n = (1.0 - wt_n) * 0.25
    # South cap: symmetric.
    wt_s = (theta - th1) / (math.pi - th1)
    fac_s = wt_s * 0.25

    a0 = torch.where(north, fac_n, torch.where(south, w0 * (1 - wt_s) + fac_s, w0 * (1 - wt_mid)))
    a1 = torch.where(north, fac_n, torch.where(south, w1 * (1 - wt_s) + fac_s, w1 * (1 - wt_mid)))
    a2 = torch.where(north, w2 * wt_n + fac_n, torch.where(south, fac_s, w2 * wt_mid))
    a3 = torch.where(north, w3 * wt_n + fac_n, torch.where(south, fac_s, w3 * wt_mid))

    q0 = torch.where(north, (p2 + 2) & 3, p0)
    q1 = torch.where(north, (p3 + 2) & 3, p1)
    q2 = torch.where(south, ((p0 + 2) & 3) + npix - 4, p2)
    q3 = torch.where(south, ((p1 + 2) & 3) + npix - 4, p3)
    return torch.stack([q0, q1, q2, q3], dim=-1), torch.stack([a0, a1, a2, a3], dim=-1)


def _isqrt(v: torch.Tensor) -> torch.Tensor:
    v = v.clamp_min(0)
    s = torch.floor(torch.sqrt(v.to(torch.float64))).long()
    s = s - (s * s > v).long()
    s = s + ((s + 1) * (s + 1) <= v).long()
    return s


def _spread_bits(x: torch.Tensor) -> torch.Tensor:
    x = x & 0x00000000FFFFFFFF
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    return x


_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)


def ring_to_nest(pix: torch.Tensor, order: int) -> torch.Tensor:
    """Torch port of HEALPix ``ring2xyf`` + ``xyf2nest`` (int64 in, int64 out)."""
    n = 1 << order
    ncap = 2 * n * (n - 1)
    npix = 12 * n * n
    nl2 = 2 * n
    pix = pix.long()

    # North polar cap
    ir_n = (1 + _isqrt(1 + 2 * pix)) >> 1
    ir_n_safe = ir_n.clamp_min(1)
    iphi_n = (pix + 1) - 2 * ir_n * (ir_n - 1)
    face_n = torch.div(iphi_n - 1, ir_n_safe, rounding_mode="floor")
    # Equatorial belt
    ip = (pix - ncap).clamp_min(0)
    tmp = ip >> (order + 2)
    ir_e = tmp + n
    iphi_e = ip - tmp * 4 * n + 1
    ks_e = (ir_e + n) & 1
    ire = tmp + 1
    irm = nl2 + 1 - tmp
    ifm = (iphi_e - (ire >> 1) + n - 1) >> order
    ifp = (iphi_e - (irm >> 1) + n - 1) >> order
    face_e = torch.where(ifp == ifm, ifp | 4, torch.where(ifp < ifm, ifp, ifm + 8))
    # South polar cap
    ips = (npix - pix).clamp_min(1)
    ir_s = (1 + _isqrt(2 * ips - 1)) >> 1
    ir_s_safe = ir_s.clamp_min(1)
    iphi_s = 4 * ir_s + 1 - (ips - 2 * ir_s * (ir_s - 1))
    face_s = torch.div(iphi_s - 1, ir_s_safe, rounding_mode="floor") + 8

    north = pix < ncap
    eq = (pix >= ncap) & (pix < npix - ncap)
    iring = torch.where(north, ir_n, torch.where(eq, ir_e, 2 * nl2 - ir_s))
    iphi = torch.where(north, iphi_n, torch.where(eq, iphi_e, iphi_s))
    nr = torch.where(north, ir_n, torch.where(eq, torch.full_like(pix, n), ir_s))
    kshift = torch.where(eq, ks_e, torch.zeros_like(pix))
    face = torch.where(north, face_n, torch.where(eq, face_e, face_s)).clamp(0, 11)

    jpll = torch.tensor(_JPLL, dtype=torch.long, device=pix.device)[face]
    irt = iring - (2 + (face >> 2)) * n + 1
    ipt = 2 * iphi - jpll * nr - kshift - 1
    ipt = torch.where(ipt >= nl2, ipt - 8 * n, ipt)
    ix = (ipt - irt) >> 1
    iy = (-ipt - irt) >> 1
    return (face << (2 * order)) + _spread_bits(ix) + (_spread_bits(iy) << 1)


def bilinear_ring(vec: torch.Tensor, order: int):
    """4 RING cells and bilinear weights at the points ``vec`` [..., 3]."""
    x, y, z = vec.unbind(-1)
    rho = torch.sqrt(x * x + y * y)
    r = torch.sqrt(rho * rho + z * z)
    zc = (z / r).clamp(-1.0, 1.0)
    theta = torch.atan2(rho, z)
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2.0 * math.pi, phi)
    phi = torch.where(phi >= 2.0 * math.pi, phi - 2.0 * math.pi, phi)
    return ring_interp(zc, theta, phi, 1 << order)


def bilinear_nested(vec: torch.Tensor, order: int):
    """4 NESTED cells and bilinear weights at unit vectors ``vec`` [..., 3].

    Same cells and weights as
    ``healpix_analyse.get_interp_weights(lon, lat, order)`` (sphere), computed
    with torch ops on the device of ``vec``.  ``vec`` need not be normalised.
    """
    pix, w = bilinear_ring(vec, order)
    return ring_to_nest(pix, order), w


# ===========================================================================
# Fused plan kernel: positions -> bilinear lookup -> domain search -> weights
# ===========================================================================

def triton_on(device: torch.device) -> bool:
    """Triton kernels usable for tensors on ``device`` (CUDA, or the Triton
    interpreter on the CPU when ``TRITON_INTERPRET=1``, for tests)."""
    if not _HAS_TRITON:
        return False
    if device.type == "cuda":
        return True
    return os.environ.get("TRITON_INTERPRET", "0") == "1" and device.type == "cpu"


if _HAS_TRITON:
    # Cephes atan on [0, 1] (rational approximation, ~1 ulp in float64).
    # Written out instead of libdevice so that the Triton interpreter runs it.
    @triton.jit
    def _tl_atan01(t):
        big = t > 0.66
        x = tl.where(big, (t - 1.0) / (t + 1.0), t)
        z = x * x
        p = ((((-8.750608600031904122785e-1 * z - 1.615753718733365076637e1) * z
               - 7.500855792314704667340e1) * z - 1.228866684490136173410e2) * z
             - 6.485021904942025371773e1)
        q = (((((z + 2.485846490142306297962e1) * z + 1.650270098316988542046e2) * z
               + 4.328810604912902668951e2) * z + 4.853903996359136964868e2) * z
             + 1.945506571482613964425e2)
        r = x + x * z * p / q
        return tl.where(big, 0.7853981633974483 + (r + 3.061616997868383e-17), r)

    @triton.jit
    def _tl_atan2(y, x):
        ax = tl.abs(x)
        ay = tl.abs(y)
        mx = tl.maximum(ax, ay)
        mn = tl.minimum(ax, ay)
        t = tl.where(mx > 0, mn / tl.where(mx > 0, mx, 1.0), 0.0 * mx)
        a = _tl_atan01(t)
        a = tl.where(ay > ax, (1.5707963267948966 - a) + 6.123233995736766e-17, a)
        a = tl.where(x < 0, (3.141592653589793 - a) + 1.2246467991473532e-16, a)
        return tl.where(y < 0, -a, a)

    @triton.jit
    def _tl_ring_int(ring, nside):
        """startpix, ringpix, shifted of RING rings (int64)."""
        ncap = 2 * nside * (nside - 1)
        npix = 12 * nside * nside
        northring = tl.where(ring > 2 * nside, 4 * nside - ring, ring)
        cap = northring < nside
        ringpix = tl.where(cap, 4 * northring, 4 * nside)
        shifted = cap | (((northring - nside) & 1) == 0)
        startpix = tl.where(cap, 2 * northring * (northring - 1),
                            ncap + (northring - nside) * 4 * nside)
        south = northring != ring
        startpix = tl.where(south, npix - startpix - ringpix, startpix)
        return startpix, ringpix, shifted

    @triton.jit
    def _tl_ring_row(phi, sp, nr, shifted):
        dphi = 6.283185307179586 / nr.to(phi.dtype)
        hs = tl.where(shifted, 0.5, 0.0).to(phi.dtype)
        tmp = phi / dphi - hs
        i1f = tl.floor(tmp)
        w1 = (phi - (i1f + hs) * dphi) / dphi
        i1 = i1f.to(tl.int64)
        i2 = i1 + 1
        i1 = tl.where(i1 < 0, i1 + nr, i1)
        i2 = tl.where(i2 >= nr, i2 - nr, i2)
        return sp + i1, sp + i2, 1.0 - w1, w1

    @triton.jit
    def _tl_find(key, rs_ptr, rp_ptr, n_rs, MODE: tl.constexpr, LOG2: tl.constexpr):
        if MODE == 1:
            pos = tl.load(rs_ptr + key)
            present = key >= 0
        else:
            lo = tl.zeros(key.shape, dtype=tl.int64)
            hi = tl.zeros(key.shape, dtype=tl.int64) + n_rs
            for _ in tl.static_range(LOG2):
                mid = (lo + hi) >> 1
                v = tl.load(rs_ptr + tl.minimum(mid, n_rs - 1))
                go = (v < key) & (mid < n_rs)
                lo = tl.where(go, mid + 1, lo)
                hi = tl.where(go, hi, mid)
            p = tl.minimum(lo, n_rs - 1)
            present = tl.load(rs_ptr + p) == key
            pos = tl.load(rp_ptr + p)
        return pos, present

    @triton.jit
    def _tl_scatter(ptrs, val, mask, FIXED: tl.constexpr):
        if FIXED:
            tl.atomic_add(ptrs, tl.floor(val + 0.5).to(tl.int64), mask=mask)
        else:
            tl.atomic_add(ptrs, val.to(tl.float32), mask=mask)

    def _aug_configs():
        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return [triton.Config({"BLOCK": 64}, num_warps=4)]
        return [
            triton.Config({"BLOCK": bl}, num_warps=nw)
            for bl in (64, 128, 256) for nw in (2, 4, 8)
            if not (bl == 64 and nw == 8)
        ]

    @triton.autotune(configs=_aug_configs(),
                     key=["N", "C", "T", "GATHER", "SCATTER", "STORE_MAP", "HAS_A", "HAS_R"],
                     reset_to_zero=["gsrc_ptr"])
    @triton.jit
    def _aug_kernel(
        uv_ptr, xyz_ptr, a_ptr, r_ptr, off_ptr, fr_ptr, thr_ptr, rs_ptr, rp_ptr, bsel_ptr,
        src_ptr, out_ptr, idx_ptr, w_ptr, scale_ptr, nn_ptr, gout_ptr, gsrc_ptr, fsc_ptr,
        N, NS, C, n_rs, nside_in, pos_off, min_support, snap, fill,
        T: tl.constexpr, T_NN: tl.constexpr, HAS_A: tl.constexpr, HAS_R: tl.constexpr,
        MODE: tl.constexpr, LOG2: tl.constexpr, GATHER: tl.constexpr, SCATTER: tl.constexpr,
        STORE_MAP: tl.constexpr, STORE_NN: tl.constexpr, FIXED: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """One augmented batch in one pass.

        For every output cell of sample ``b`` and every sampling tap: the
        source point (``R lift(A (uv + offset))`` or ``R xyz``), the HEALPix
        bilinear lookup (``get_interpol``, RING), the binary search of the 4
        RING cells among the cells of the domain, and, with ``GATHER``, the
        weighted gather of ``BLOCK_C`` channels accumulated in fp32.  The
        weights are renormalised once at the end (one scale per cell, written
        to ``scale``), so nothing is stored between the taps.

        ``SCATTER`` is the backward pass: the same geometry is recomputed and
        ``scale * w * grad_out`` is scattered into ``gsrc``; with ``FIXED``
        (deterministic) into an int64 fixed-point accumulator (``fsc`` =
        2^k), whose integer atomics give the same sum in any order.
        ``STORE_NN`` writes the nearest cell (labels); ``STORE_MAP`` the raw
        map (idx, weight) for inspection and the torch fallback.
        """
        pid = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = tl.load(bsel_ptr + tl.program_id(2)).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        o64 = offs.to(tl.int64)
        nside = nside_in.to(tl.int64)
        npix = 12 * nside * nside
        h = tl.load(fr_ptr + 9)
        if HAS_A:
            u0 = tl.load(uv_ptr + 2 * o64, mask=m, other=0.0)
            v0 = tl.load(uv_ptr + 2 * o64 + 1, mask=m, other=0.0)
            a00 = tl.load(a_ptr + b * 4)
            a01 = tl.load(a_ptr + b * 4 + 1)
            a10 = tl.load(a_ptr + b * 4 + 2)
            a11 = tl.load(a_ptr + b * 4 + 3)
            c0 = tl.load(fr_ptr + 0)
            c1 = tl.load(fr_ptr + 1)
            c2 = tl.load(fr_ptr + 2)
            e10 = tl.load(fr_ptr + 3)
            e11 = tl.load(fr_ptr + 4)
            e12 = tl.load(fr_ptr + 5)
            e20 = tl.load(fr_ptr + 6)
            e21 = tl.load(fr_ptr + 7)
            e22 = tl.load(fr_ptr + 8)
        else:
            x0 = tl.load(xyz_ptr + 3 * o64, mask=m, other=1.0)
            y0 = tl.load(xyz_ptr + 3 * o64 + 1, mask=m, other=0.0)
            z0 = tl.load(xyz_ptr + 3 * o64 + 2, mask=m, other=0.0)
        if HAS_R:
            r00 = tl.load(r_ptr + b * 9)
            r01 = tl.load(r_ptr + b * 9 + 1)
            r02 = tl.load(r_ptr + b * 9 + 2)
            r10 = tl.load(r_ptr + b * 9 + 3)
            r11 = tl.load(r_ptr + b * 9 + 4)
            r12 = tl.load(r_ptr + b * 9 + 5)
            r20 = tl.load(r_ptr + b * 9 + 6)
            r21 = tl.load(r_ptr + b * 9 + 7)
            r22 = tl.load(r_ptr + b * 9 + 8)
        zero = tl.zeros((BLOCK,), dtype=tl.float32).to(h.dtype)
        s_in = zero
        tot = zero
        nn_pos = tl.zeros((BLOCK,), dtype=tl.int64) - 1
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mc = offs_c < C
        src_b = src_ptr + b * C * NS + offs_c.to(tl.int64)[:, None] * NS
        acc = tl.zeros((BLOCK_C, BLOCK), dtype=tl.float32)
        if SCATTER:
            g_in = tl.load(gout_ptr + b * C * N + offs_c.to(tl.int64)[:, None] * N + o64[None, :],
                           mask=mc[:, None] & m[None, :], other=0.0)
            sc_in = tl.load(scale_ptr + b * N + o64, mask=m, other=0.0)
            coef = sc_in[None, :] * g_in.to(sc_in.dtype)
            if FIXED:
                coef = coef * tl.load(fsc_ptr)
            gs_b = gsrc_ptr + b * C * NS + offs_c.to(tl.int64)[:, None] * NS
        J: tl.constexpr = 4 * T
        out_b = b * J * N
        do_store = pid_c == 0
        NT: tl.constexpr = T + 1 if T_NN == T else T
        for t in tl.static_range(NT):
            if HAS_A:
                if t < T:
                    u = u0 + tl.load(off_ptr + 2 * t)
                    v = v0 + tl.load(off_ptr + 2 * t + 1)
                else:
                    u = u0
                    v = v0
                up = a00 * u + a01 * v
                vp = a10 * u + a11 * v
                qx = c0 + up * e10 + vp * e20
                qy = c1 + up * e11 + vp * e21
                qz = c2 + up * e12 + vp * e22
            else:
                qx = x0
                qy = y0
                qz = z0
            if HAS_R:
                px = r00 * qx + r01 * qy + r02 * qz
                py = r10 * qx + r11 * qy + r12 * qz
                pz = r20 * qx + r21 * qy + r22 * qz
                qx = px
                qy = py
                qz = pz
            # --- get_interpol (RING) ---------------------------------------
            rho = tl.sqrt(qx * qx + qy * qy)
            rr = tl.sqrt(rho * rho + qz * qz)
            z = tl.minimum(tl.maximum(qz / rr, -1.0), 1.0)
            theta = _tl_atan2(rho, qz)
            phi = _tl_atan2(qy, qx)
            phi = tl.where(phi < 0, phi + 6.283185307179586, phi)
            phi = tl.where(phi >= 6.283185307179586, phi - 6.283185307179586, phi)
            az = tl.abs(z)
            nsf = nside.to(z.dtype)
            r_eq = tl.floor(nsf * (2.0 - 1.5 * z))
            r_pol = tl.floor(nsf * tl.sqrt(tl.maximum(3.0 * (1.0 - az), 0.0)))
            r_pol = tl.where(z > 0, r_pol, 4.0 * nsf - r_pol - 1.0)
            ir1 = tl.where(az <= 2.0 / 3.0, r_eq, r_pol).to(tl.int64)
            ir2 = ir1 + 1
            ra = tl.minimum(tl.maximum(ir1, 1), 4 * nside - 1)
            rb = tl.minimum(tl.maximum(ir2, 1), 4 * nside - 1)
            sp1, nr1, sh1 = _tl_ring_int(ra, nside)
            sp2, nr2, sh2 = _tl_ring_int(rb, nside)
            th1 = tl.load(thr_ptr + ra)
            th2 = tl.load(thr_ptr + rb)
            p0, p1, w0, w1 = _tl_ring_row(phi, sp1, nr1, sh1)
            p2, p3, w2, w3 = _tl_ring_row(phi, sp2, nr2, sh2)
            north = ir1 == 0
            south = (ir2 == 4 * nside) & (ir1 != 0)
            wt_mid = (theta - th1) / (th2 - th1)
            wt_n = theta / th2
            fac_n = (1.0 - wt_n) * 0.25
            wt_s = (theta - th1) / (3.141592653589793 - th1)
            fac_s = wt_s * 0.25
            g0 = tl.where(north, fac_n, tl.where(south, w0 * (1 - wt_s) + fac_s, w0 * (1 - wt_mid)))
            g1 = tl.where(north, fac_n, tl.where(south, w1 * (1 - wt_s) + fac_s, w1 * (1 - wt_mid)))
            g2 = tl.where(north, w2 * wt_n + fac_n, tl.where(south, fac_s, w2 * wt_mid))
            g3 = tl.where(north, w3 * wt_n + fac_n, tl.where(south, fac_s, w3 * wt_mid))
            k0 = tl.where(north, (p2 + 2) & 3, p0)
            k1 = tl.where(north, (p3 + 2) & 3, p1)
            k2 = tl.where(south, ((p0 + 2) & 3) + npix - 4, p2)
            k3 = tl.where(south, ((p1 + 2) & 3) + npix - 4, p3)
            g0 = tl.where(g0 < snap, 0.0, g0)
            g1 = tl.where(g1 < snap, 0.0, g1)
            g2 = tl.where(g2 < snap, 0.0, g2)
            g3 = tl.where(g3 < snap, 0.0, g3)
            # --- domain lookup ---------------------------------------------
            q0, f0 = _tl_find(k0, rs_ptr, rp_ptr, n_rs, MODE, LOG2)
            q1, f1 = _tl_find(k1, rs_ptr, rp_ptr, n_rs, MODE, LOG2)
            q2, f2 = _tl_find(k2, rs_ptr, rp_ptr, n_rs, MODE, LOG2)
            q3, f3 = _tl_find(k3, rs_ptr, rp_ptr, n_rs, MODE, LOG2)
            q0 = q0 + pos_off
            q1 = q1 + pos_off
            q2 = q2 + pos_off
            q3 = q3 + pos_off
            if t < T:
                h0 = tl.where(f0, g0, 0.0)
                h1 = tl.where(f1, g1, 0.0)
                h2 = tl.where(f2, g2, 0.0)
                h3 = tl.where(f3, g3, 0.0)
                s_in += h0 + h1 + h2 + h3
                tot += g0 + g1 + g2 + g3
                if GATHER:
                    # raw (unnormalised) weights; one scale per cell at the end
                    v = tl.load(src_b + q0[None, :], mask=mc[:, None] & (m & (h0 > 0))[None, :], other=0.0)
                    acc += h0.to(tl.float32)[None, :] * v.to(tl.float32)
                    v = tl.load(src_b + q1[None, :], mask=mc[:, None] & (m & (h1 > 0))[None, :], other=0.0)
                    acc += h1.to(tl.float32)[None, :] * v.to(tl.float32)
                    v = tl.load(src_b + q2[None, :], mask=mc[:, None] & (m & (h2 > 0))[None, :], other=0.0)
                    acc += h2.to(tl.float32)[None, :] * v.to(tl.float32)
                    v = tl.load(src_b + q3[None, :], mask=mc[:, None] & (m & (h3 > 0))[None, :], other=0.0)
                    acc += h3.to(tl.float32)[None, :] * v.to(tl.float32)
                if SCATTER:
                    ma = mc[:, None] & (m & (h0 > 0))[None, :]
                    _tl_scatter(gs_b + q0[None, :], h0[None, :] * coef, ma, FIXED)
                    ma = mc[:, None] & (m & (h1 > 0))[None, :]
                    _tl_scatter(gs_b + q1[None, :], h1[None, :] * coef, ma, FIXED)
                    ma = mc[:, None] & (m & (h2 > 0))[None, :]
                    _tl_scatter(gs_b + q2[None, :], h2[None, :] * coef, ma, FIXED)
                    ma = mc[:, None] & (m & (h3 > 0))[None, :]
                    _tl_scatter(gs_b + q3[None, :], h3[None, :] * coef, ma, FIXED)
                if STORE_MAP:
                    sm = m & do_store
                    base = out_b + (4 * t) * N + o64
                    tl.store(idx_ptr + base, tl.where(h0 > 0, q0, -1).to(tl.int32), mask=sm)
                    tl.store(idx_ptr + base + N, tl.where(h1 > 0, q1, -1).to(tl.int32), mask=sm)
                    tl.store(idx_ptr + base + 2 * N, tl.where(h2 > 0, q2, -1).to(tl.int32), mask=sm)
                    tl.store(idx_ptr + base + 3 * N, tl.where(h3 > 0, q3, -1).to(tl.int32), mask=sm)
                    tl.store(w_ptr + base, h0, mask=sm)
                    tl.store(w_ptr + base + N, h1, mask=sm)
                    tl.store(w_ptr + base + 2 * N, h2, mask=sm)
                    tl.store(w_ptr + base + 3 * N, h3, mask=sm)
            if STORE_NN:
                if t == T_NN:
                    ww0 = tl.where(f0, g0, -1.0)
                    ww1 = tl.where(f1, g1, -1.0)
                    ww2 = tl.where(f2, g2, -1.0)
                    ww3 = tl.where(f3, g3, -1.0)
                    best = ww0
                    nn_pos = tl.where(ww0 >= 0, q0 - pos_off, -1)
                    nn_pos = tl.where(ww1 > best, q1 - pos_off, nn_pos)
                    best = tl.maximum(best, ww1)
                    nn_pos = tl.where(ww2 > best, q2 - pos_off, nn_pos)
                    best = tl.maximum(best, ww2)
                    nn_pos = tl.where(ww3 > best, q3 - pos_off, nn_pos)
        ok = (s_in >= min_support * tot - 1e-12) & (s_in > 0)
        scale = tl.where(ok, 1.0 / tl.where(ok, s_in, 1.0), 0.0)
        if GATHER:
            res = tl.where(ok[None, :], acc * scale.to(tl.float32)[None, :], fill)
            out = out_ptr + b * C * N + offs_c.to(tl.int64)[:, None] * N + o64[None, :]
            tl.store(out, res.to(out_ptr.dtype.element_ty), mask=mc[:, None] & m[None, :])
        sm = m & do_store
        if not SCATTER:
            tl.store(scale_ptr + b * N + o64, scale, mask=sm)
        if STORE_NN:
            tl.store(nn_ptr + b * N + o64, tl.where(ok, nn_pos, -1), mask=sm)


def ring_theta_table(nside: int, dtype, device) -> torch.Tensor:
    """Colatitude of every RING ring (index 0 unused), as in :func:`ring_interp`."""
    ring = torch.arange(4 * nside, dtype=torch.long, device=device).clamp_min(1)
    return _ring_info(ring, nside, dtype)[2].contiguous()


# Fixed-point backward: gradients are accumulated as int64 multiples of
# 2^-k * max|grad_out|, k = _FIXED_BITS.  Exact and order-independent as long
# as the weights received by one source cell sum to less than
# 2^(62 - _FIXED_BITS) (= 4096: a zoom-in by a factor of ~30); the rounding of
# each contribution is below 2^-_FIXED_BITS * max|grad_out| (~1e-15).
_FIXED_BITS = 50


def fused_augment_triton(geo, src=None, *, gout=None, scale=None, store_map=False,
                         store_nn=False, fill=float("nan"), deterministic=True, n_src=None):
    """Run the fused augmentation kernel for one geometric group.

    ``geo`` is a dict with the per-launch geometry: ``uv``, ``xyz``, ``A``
    [B, 2, 2], ``R`` [B, 3, 3], ``off`` [T, 2], ``frame_h`` (c, e1, e2, h),
    ``launches`` = list of (bsel, nside, thr, lookup, pos_off), ``T``,
    ``t_nn``, ``has_a``, ``has_r``, ``min_support``, ``snap``, ``B``, ``N``.

    Forward (``gout is None``): with ``src`` [B, C, NS] (float32 / float16 /
    bfloat16) gathers into ``out`` [B, C, N]; always writes ``scale`` [B, N]
    (the renormalisation of each cell, 0 outside the domain).  Returns
    ``(out, scale, nn, maps)``: ``nn`` [B, N] with ``store_nn``, ``maps`` =
    (idx [B, 4T, N] int32, raw w [B, 4T, N]) with ``store_map`` (entry weight
    = w * scale).

    Backward (``gout`` [B, C, N] and the forward ``scale``): returns the
    gradient [B, C, n_src] in float32, deterministic with
    ``deterministic=True`` (int64 fixed point), fp32 atomics otherwise.
    """
    B, N, T = geo["B"], geo["N"], geo["T"]
    gd = geo["frame_h"].dtype
    dev = geo["frame_h"].device
    J = 4 * T
    dummy = geo["frame_h"]
    scatter = gout is not None
    gather = src is not None and not scatter
    maps = nn = out = gsrc = None
    fsc = dummy
    fixed = False
    if scatter:
        gout = gout.contiguous()
        _, C, _ = gout.shape
        NS = int(n_src)
        gmax = float(gout.abs().max()) if gout.numel() else 0.0
        fixed = bool(deterministic) and math.isfinite(gmax)
        if fixed:
            k = _FIXED_BITS - (math.frexp(gmax)[1] if gmax > 0 else 0)
            fsc = torch.tensor([2.0 ** k], dtype=torch.float64, device=dev)
            gsrc = torch.zeros(B, C, NS, dtype=torch.int64, device=dev)
        else:
            gsrc = torch.zeros(B, C, NS, dtype=torch.float32, device=dev)
    elif gather:
        _, C, NS = src.shape
        out = torch.empty(B, C, N, dtype=src.dtype, device=dev)
    else:
        C, NS = 1, 1
    if scatter or gather:
        BLOCK_C = max(1, min(16, triton.next_power_of_2(C)))
        n_cb = triton.cdiv(C, BLOCK_C)
    else:
        BLOCK_C, n_cb = 1, 1
    if not scatter:
        scale = torch.empty(B, N, dtype=gd, device=dev)
    if store_map:
        maps = (torch.empty(B, J, N, dtype=torch.int32, device=dev),
                torch.empty(B, J, N, dtype=gd, device=dev))
    if store_nn:
        nn = torch.empty(B, N, dtype=torch.long, device=dev)
    idx, w = maps if store_map else (dummy, dummy)
    has_a, has_r = geo["has_a"], geo["has_r"]
    for bsel, nside, thr, lookup, pos_off in geo["launches"]:
        nb = bsel.numel()
        if nb == 0:
            continue
        if lookup[0] == "table":
            mode, rs, rp, n_rs, log2 = 1, lookup[1], lookup[1], lookup[1].numel(), 1
        else:
            rs, rp = lookup[1], lookup[2]
            mode, n_rs = 0, rs.numel()
            log2 = max(1, int(math.ceil(math.log2(n_rs + 1))))
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), n_cb, nb)  # noqa: E731
        _aug_kernel[grid](
            geo["uv"] if has_a else dummy, geo["xyz"] if not has_a else dummy,
            geo["A"] if has_a else dummy, geo["R"] if has_r else dummy,
            geo["off"] if has_a else dummy, geo["frame_h"], thr, rs, rp, bsel,
            src if gather else dummy, out if gather else dummy, idx, w, scale,
            nn if store_nn else dummy, gout if scatter else dummy,
            gsrc if scatter else torch.zeros(1, device=dev), fsc,
            N, NS, C, n_rs, int(nside), int(pos_off),
            float(geo["min_support"]), float(geo["snap"]), float(fill),
            T=T, T_NN=geo["t_nn"], HAS_A=bool(has_a), HAS_R=bool(has_r), MODE=mode, LOG2=log2,
            GATHER=gather, SCATTER=scatter, STORE_MAP=bool(store_map), STORE_NN=bool(store_nn),
            FIXED=fixed, BLOCK_C=BLOCK_C,
        )
    if scatter:
        if fixed:
            return (gsrc.to(torch.float64) / fsc).to(torch.float32)
        return gsrc
    return out, scale, nn, maps


# ===========================================================================
# Gather map
# ===========================================================================

class GatherMap:
    """Indices and weights of one fused geometric gather.

    Attributes
    ----------
    idx : IntTensor [B, J, N]
        Positions in the source array (``-1`` where unused).
    w : Tensor [B, J, N], float32 (float64 for a float64 geometry)
        Renormalised weights, 0 where unused.  A row whose weights are all 0
        is outside the domain and receives the fill value.
    valid : BoolTensor [B, N]
    nn : LongTensor [B, N]
        Nearest source cell (for labels), ``-1`` outside the domain.
    n_src : int
        Length of the source axis (the input, or the input pyramid).
    """

    def __init__(self, idx, w, valid, nn, n_src):
        self.idx = idx
        self.w = w
        self.valid = valid
        self.nn = nn
        self.n_src = int(n_src)
        self._csr = None
        self._w32 = None

    @property
    def w32(self):
        """float32 copy of the weights, read by the Triton kernels."""
        if self._w32 is None:
            self._w32 = self.w if self.w.dtype == torch.float32 else self.w.float().contiguous()
        return self._w32

    @property
    def batch_size(self) -> int:
        return self.idx.shape[0]

    def csr(self):
        """Transposed map (rowptr [B*n_src+1], entries [E]) for the backward."""
        if self._csr is None:
            B, J, N = self.idx.shape
            ns = self.n_src
            ok = (self.idx >= 0) & (self.w != 0)
            b_off = (torch.arange(B, device=self.idx.device) * ns).view(B, 1, 1)
            key = torch.where(ok, self.idx.long() + b_off, torch.full_like(b_off, B * ns))
            key = key.reshape(-1)
            if B * ns < 2 ** 31 - 1:
                key = key.int()
            skey, order = torch.sort(key, stable=True)
            counts = torch.bincount(skey.long(), minlength=B * ns + 1)[: B * ns]
            rowptr = torch.zeros(B * ns + 1, dtype=torch.long, device=key.device)
            rowptr[1:] = torch.cumsum(counts, 0)
            n_valid = int(rowptr[-1].item())
            entries = order[:n_valid]
            max_cnt = int(counts.max().item()) if counts.numel() else 0
            if B * J * N < 2 ** 31 - 1:
                entries = entries.int()
            self._csr = (rowptr, entries, max_cnt)
        return self._csr


# ===========================================================================
# Torch reference gather
# ===========================================================================

def gather_torch(src: torch.Tensor, gmap: GatherMap, fill: float) -> torch.Tensor:
    """Reference gather ``[B, C, n_src] -> [B, C, N]`` (autograd backward)."""
    B, C, _ = src.shape
    _, J, N = gmap.idx.shape
    acc_dtype = torch.float32 if src.dtype in (torch.float16, torch.bfloat16) else src.dtype
    s = src.to(acc_dtype)
    out = torch.zeros(B, C, N, dtype=acc_dtype, device=src.device)
    for j in range(J):
        ij = gmap.idx[:, j].clamp_min(0).long()
        v = torch.gather(s, 2, ij.unsqueeze(1).expand(B, C, N))
        wj = gmap.w[:, j].to(acc_dtype).unsqueeze(1)
        out = out + torch.where(wj != 0, wj * v, torch.zeros((), dtype=acc_dtype, device=src.device))
    out = torch.where(gmap.valid.unsqueeze(1), out, torch.full((), fill, dtype=acc_dtype, device=src.device))
    return out.to(src.dtype)


# ===========================================================================
# Triton kernels
# ===========================================================================

if _HAS_TRITON:

    def _cfg(name, sizes=(128, 256, 512), warps=(2, 4, 8)):
        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return [triton.Config({name: 64}, num_warps=4)]
        return [triton.Config({name: s}, num_warps=w) for s in sizes for w in warps]

    @triton.autotune(configs=_cfg("BLOCK_N"), key=["C", "N", "J"])
    @triton.jit
    def _gather_fwd_kernel(
        src_ptr, idx_ptr, w_ptr, out_ptr,
        C, NS, N, fill,
        J: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mn = offs_n < N
        mc = offs_c < C
        src_b = src_ptr + b * C * NS + offs_c.to(tl.int64)[:, None] * NS
        acc = tl.zeros((BLOCK_C, BLOCK_N), dtype=tl.float32)
        wsum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for j in tl.static_range(J):
            base = (b * J + j) * N
            ij = tl.load(idx_ptr + base + offs_n, mask=mn, other=-1)
            wj = tl.load(w_ptr + base + offs_n, mask=mn, other=0.0).to(tl.float32)
            ok = (ij >= 0) & (wj != 0.0)
            v = tl.load(src_b + ij.to(tl.int64)[None, :], mask=mc[:, None] & ok[None, :], other=0.0)
            acc += wj[None, :] * v.to(tl.float32)
            wsum += tl.abs(wj)
        res = tl.where((wsum > 0.0)[None, :], acc, fill)
        out_b = out_ptr + b * C * N + offs_c.to(tl.int64)[:, None] * N + offs_n[None, :]
        tl.store(out_b, res.to(out_ptr.dtype.element_ty), mask=mc[:, None] & mn[None, :])

    @triton.autotune(configs=_cfg("BLOCK_N"), key=["C", "N", "J"], reset_to_zero=["gsrc_ptr"])
    @triton.jit
    def _gather_bwd_atomic_kernel(
        gout_ptr, idx_ptr, w_ptr, gsrc_ptr,
        C, NS, N,
        J: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mn = offs_n < N
        mc = offs_c < C
        g = tl.load(
            gout_ptr + b * C * N + offs_c.to(tl.int64)[:, None] * N + offs_n[None, :],
            mask=mc[:, None] & mn[None, :], other=0.0,
        ).to(tl.float32)
        gs_b = gsrc_ptr + b * C * NS + offs_c.to(tl.int64)[:, None] * NS
        for j in tl.static_range(J):
            base = (b * J + j) * N
            ij = tl.load(idx_ptr + base + offs_n, mask=mn, other=-1)
            wj = tl.load(w_ptr + base + offs_n, mask=mn, other=0.0).to(tl.float32)
            ok = (ij >= 0) & (wj != 0.0)
            tl.atomic_add(gs_b + ij.to(tl.int64)[None, :], wj[None, :] * g,
                          mask=mc[:, None] & ok[None, :])

    @triton.autotune(configs=_cfg("BLOCK_R", sizes=(64, 128, 256)), key=["C", "NS"])
    @triton.jit
    def _gather_bwd_csr_kernel(
        gout_ptr, rowptr_ptr, ent_ptr, w_ptr, gsrc_ptr,
        C, NS, N,
        BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = tl.program_id(2).to(tl.int64)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mr = offs_r < NS
        mc = offs_c < C
        rows = b * NS + offs_r
        start = tl.load(rowptr_ptr + rows, mask=mr, other=0)
        end = tl.load(rowptr_ptr + rows + 1, mask=mr, other=0)
        cnt = end - start
        max_cnt = tl.max(cnt, axis=0)
        acc = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
        gout_b = gout_ptr + b * C * N + offs_c.to(tl.int64)[:, None] * N
        for k in range(0, max_cnt):
            mk = mr & (k < cnt)
            e = tl.load(ent_ptr + start + k, mask=mk, other=0).to(tl.int64)
            we = tl.load(w_ptr + e, mask=mk, other=0.0).to(tl.float32)
            n = e % N
            g = tl.load(gout_b + n[None, :], mask=mc[:, None] & mk[None, :], other=0.0)
            acc += we[None, :] * g.to(tl.float32)
        out = gsrc_ptr + b * C * NS + offs_c.to(tl.int64)[:, None] * NS + offs_r[None, :]
        tl.store(out, acc.to(gsrc_ptr.dtype.element_ty), mask=mc[:, None] & mr[None, :])


def _block_c(C: int) -> int:
    return max(1, min(8, triton.next_power_of_2(C)))


def _gather_triton(src, gmap, fill):
    B, C, NS = src.shape
    _, J, N = gmap.idx.shape
    out = torch.empty(B, C, N, dtype=src.dtype, device=src.device)
    BLOCK_C = _block_c(C)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(C, BLOCK_C), B)  # noqa: E731
    _gather_fwd_kernel[grid](
        src, gmap.idx, gmap.w32, out, C, NS, N, float(fill), J=J, BLOCK_C=BLOCK_C,
    )
    return out


def _scatter_torch(gout, gmap):
    """Differentiable adjoint of the gather (index_add over the taps)."""
    B, C, N = gout.shape
    ns = gmap.n_src
    acc_dtype = torch.float32 if gout.dtype in (torch.float16, torch.bfloat16) else gout.dtype
    g = gout.to(acc_dtype)
    gsrc = torch.zeros(B, C, ns, dtype=acc_dtype, device=gout.device)
    for j in range(gmap.idx.shape[1]):
        ij = gmap.idx[:, j].clamp_min(0).long()
        wj = gmap.w[:, j].to(acc_dtype).unsqueeze(1)
        contrib = torch.where(wj != 0, wj * g, torch.zeros((), dtype=acc_dtype, device=g.device))
        gsrc = gsrc.scatter_add(2, ij.unsqueeze(1).expand(B, C, N), contrib)
    return gsrc


class _TritonGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, gmap, fill, deterministic):
        ctx.gmap = gmap
        ctx.deterministic = deterministic
        ctx.src_dtype = src.dtype
        return _gather_triton(src.contiguous(), gmap, fill)

    @staticmethod
    def backward(ctx, gout):
        return gather_backward(gout, ctx.gmap, ctx.deterministic, ctx.src_dtype), None, None, None


def gather_backward(gout, gmap, deterministic, src_dtype):
    """Gradient of a gather with respect to its source [B, C, n_src]."""
    if torch.is_grad_enabled():  # create_graph=True: differentiable path
        return _scatter_torch(gout, gmap).to(src_dtype)
    if not triton_on(gout.device):
        return _scatter_torch(gout, gmap).to(src_dtype)
    gout = gout.contiguous()
    B, C, N = gout.shape
    ns = gmap.n_src
    J = gmap.idx.shape[1]
    BLOCK_C = _block_c(C)
    if deterministic:
        rowptr, entries, _ = gmap.csr()
        gsrc = torch.empty(B, C, ns, dtype=src_dtype, device=gout.device)
        grid = lambda meta: (triton.cdiv(ns, meta["BLOCK_R"]), triton.cdiv(C, BLOCK_C), B)  # noqa: E731
        _gather_bwd_csr_kernel[grid](
            gout, rowptr, entries, gmap.w32, gsrc, C, ns, N, BLOCK_C=BLOCK_C,
        )
        return gsrc
    gsrc = torch.zeros(B, C, ns, dtype=torch.float32, device=gout.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(C, BLOCK_C), B)  # noqa: E731
    _gather_bwd_atomic_kernel[grid](
        gout, gmap.idx, gmap.w32, gsrc, C, ns, N, J=J, BLOCK_C=BLOCK_C,
    )
    return gsrc.to(src_dtype)


_WARNED_TORCH_CUDA = False


def resolve_backend(backend: str, src: torch.Tensor) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    can = triton_on(src.device) and src.dtype in _TRITON_DTYPES
    if backend == "triton":
        if not can:
            raise RuntimeError(
                "backend='triton' needs the triton package, a CUDA tensor and "
                "float32 / float16 / bfloat16 data"
            )
        return "triton"
    if backend == "auto":
        return "triton" if can else "torch"
    return "torch"


def apply_gather(
    src: torch.Tensor,
    gmap: GatherMap,
    fill: float = float("nan"),
    backend: str = "auto",
    deterministic: bool = True,
) -> torch.Tensor:
    """Apply a :class:`GatherMap` to ``src`` [B, C, n_src] -> [B, C, N]."""
    if src.dim() != 3 or src.shape[0] != gmap.batch_size or src.shape[2] != gmap.n_src:
        raise ValueError(
            f"expected data [B={gmap.batch_size}, C, {gmap.n_src}], got {tuple(src.shape)}"
        )
    be = resolve_backend(backend, src)
    if be == "triton":
        return _TritonGather.apply(src, gmap, float(fill), bool(deterministic))
    global _WARNED_TORCH_CUDA
    if src.is_cuda and src.requires_grad and not _WARNED_TORCH_CUDA and backend == "auto":
        _WARNED_TORCH_CUDA = True
        warnings.warn(
            "healpix_analyse.augment: the torch gather backward on CUDA uses "
            "scatter_add (atomics), so it is not deterministic unless "
            "torch.use_deterministic_algorithms(True); install triton for the "
            "deterministic fused backward.",
            stacklevel=3,
        )
    return gather_torch(src, gmap, fill)
