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
    BLOCK_N = 256
    BLOCK_C = _block_c(C)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C), B)
    _gather_fwd_kernel[grid](
        src, gmap.idx, gmap.w32, out, C, NS, N, float(fill),
        J=J, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, num_warps=4,
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
        gmap = ctx.gmap
        if torch.is_grad_enabled():  # create_graph=True: differentiable path
            return _scatter_torch(gout, gmap).to(ctx.src_dtype), None, None, None
        gout = gout.contiguous()
        B, C, N = gout.shape
        ns = gmap.n_src
        J = gmap.idx.shape[1]
        BLOCK_C = _block_c(C)
        if ctx.deterministic:
            rowptr, entries, _ = gmap.csr()
            gsrc = torch.empty(B, C, ns, dtype=ctx.src_dtype, device=gout.device)
            BLOCK_R = 128
            grid = (triton.cdiv(ns, BLOCK_R), triton.cdiv(C, BLOCK_C), B)
            _gather_bwd_csr_kernel[grid](
                gout, rowptr, entries, gmap.w32, gsrc, C, ns, N,
                BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=4,
            )
            return gsrc, None, None, None
        gsrc = torch.zeros(B, C, ns, dtype=torch.float32, device=gout.device)
        BLOCK_N = 256
        grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(C, BLOCK_C), B)
        _gather_bwd_atomic_kernel[grid](
            gout, gmap.idx, gmap.w32, gsrc, C, ns, N,
            J=J, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, num_warps=4,
        )
        return gsrc.to(ctx.src_dtype), None, None, None


_WARNED_TORCH_CUDA = False


def resolve_backend(backend: str, src: torch.Tensor) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    can = _HAS_TRITON and src.is_cuda and src.dtype in _TRITON_DTYPES
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
