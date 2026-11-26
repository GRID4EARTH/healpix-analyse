"""Compute backends of :class:`healpix_analyse.patches.HealPixPatches`.

A patch extraction is a sparse linear map from the cells to the patch pixels.
Each patch pixel is a weighted sum of ``J`` input cells (``J = 4`` bilinear
neighbours in ``mode="gauge"``, ``J = 1`` in ``mode="nested"``)::

    y[b, i, g, c, p] = sum_j w[j, r] x[b, c, idx[j, r]],   r = (m*G + g)*P + p

with ``m = sel[b, i]`` (the selected centres) or ``m = i``.  ``idx`` and ``w``
are the *patch geometry*, both ``[J, M*G*P]``: ``idx`` holds int32 positions
in the order of the input and ``w`` the weights (0 for a neighbour outside
the domain).

The adjoint, ``fold``, is the transposed map::

    x[b, c, s] = scale[b, s] * sum_{(j, r): idx[j, r] = s} w[j, r] y[b, m(r), g(r), c, p(r)]

It runs as a *gather* on the transposed map (for each cell, its incoming
(row, weight) entries in CSR form), so it needs no atomics and gives the same
bits at every run.  The backward of the extraction is the raw fold, and the
backward of the fold is an extraction: both autograd Functions call each
other, so the gradients of any order are available on both backends.

``"torch"`` (CPU, float64, and the reference everywhere)
    ``index_select`` of the J neighbours and a weighted sum; the fold uses
    ``index_add_`` (deterministic on CPU).
``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation)
    One gather kernel and one CSR gather kernel.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
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

BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FOLD_NORMS = ("weight", "count", "none")


def triton_available() -> bool:
    """True when the Triton kernels can run (Triton importable and CUDA present)."""
    return _HAS_TRITON and torch.cuda.is_available()


# ===========================================================================
# Geometry
# ===========================================================================

@dataclass(eq=False)
class PatchGeometry:
    """Sparse map from ``K`` cells to ``M`` patches of ``G`` gauges and ``P`` pixels.

    ``idx`` int32 and ``w`` (float32 or float64) are ``[J, M*G*P]``.  The
    transposed map and the normalisations of ``fold`` are built on demand
    and cached here.
    """

    idx: torch.Tensor
    w: torch.Tensor
    K: int
    M: int
    G: int
    P: int
    _tmap: Optional[tuple] = field(default=None, repr=False)
    _norms: dict = field(default_factory=dict, repr=False)
    _idx64: Optional[torch.Tensor] = field(default=None, repr=False)
    _wcast: dict = field(default_factory=dict, repr=False)
    _footprint: Optional[torch.Tensor] = field(default=None, repr=False)

    def footprint(self):
        """``(fp [M, F] int32, shift)``: the blocks of ``2**shift`` consecutive
        cells that the patch of each centre reads (entries with a non-zero
        weight), padded with the sentinel ``n_mark_blocks(K, shift)``.  Built
        once, on the device of the geometry; the masked fold uses it to skip
        the cells that no selected patch reads.  ``shift`` is ``MARK_SHIFT``
        when set, else the smallest one whose footprint has at most
        ``MARK_MAX_F`` blocks per centre.
        """
        fp = self._footprint
        if fp is None or fp[0].device != self.idx.device or (
                MARK_SHIFT is not None and fp[1] != MARK_SHIFT):
            if MARK_SHIFT is not None:
                fp = (build_footprint(self.idx, self.w, self.K, self.M, MARK_SHIFT),
                      MARK_SHIFT)
            else:
                for shift in range(0, 2 * 16):
                    f = build_footprint(self.idx, self.w, self.K, self.M, shift)
                    if f.shape[1] <= MARK_MAX_F or (1 << shift) >= self.K:
                        break
                fp = (f, shift)
            self._footprint = fp
        return self._footprint

    def idx64(self):
        """int64 copy of ``idx`` for the PyTorch backend (``index_select`` and
        ``index_add_`` take a slow path with int32 indices on CPU)."""
        if self._idx64 is None:
            self._idx64 = self.idx.long()
        return self._idx64

    def w_as(self, dtype):
        """``w`` cast to ``dtype`` (cached)."""
        if dtype == self.w.dtype:
            return self.w
        if dtype not in self._wcast:
            self._wcast[dtype] = self.w.to(dtype)
        return self._wcast[dtype]

    @property
    def J(self) -> int:
        return self.idx.shape[0]

    def to(self, device=None, dtype=None):
        """Copy on ``device``; ``dtype`` casts the weights (float64 or float32)."""
        w = self.w.to(device=device, dtype=dtype if dtype is not None else self.w.dtype)
        return PatchGeometry(self.idx.to(device=device), w, self.K, self.M, self.G, self.P)

    def transposed(self):
        """``(ptr [K+1] int64, rows [nnz] int32, tw [nnz])``, zero weights dropped.

        Entries of cell ``s`` are ``ptr[s] <= e < ptr[s+1]``, in increasing
        row order (a fixed order, hence a deterministic sum).
        """
        if self._tmap is None or self._tmap[0].device != self.idx.device:
            self._tmap = build_transposed_map(self.idx, self.w, self.K)
        return self._tmap

    def norm(self, kind):
        """``[K]`` normalisation of the full fold: sum of the weights
        (``"weight"``) or number of entries (``"count"``) that read each cell."""
        key = (kind, self.w.device)
        if key not in self._norms:
            if kind == "count":
                ptr = self.transposed()[0]
                n = (ptr[1:] - ptr[:-1]).to(self.w.dtype)
            else:
                # summed in float64 on the CPU: deterministic, once per geometry
                n = torch.zeros(self.K, dtype=torch.float64)
                n.index_add_(0, self.idx.reshape(-1).cpu().long(),
                             self.w.reshape(-1).cpu().to(torch.float64))
                n = n.to(device=self.w.device, dtype=self.w.dtype)
            self._norms[key] = n
        return self._norms[key]


# Granularity of the masked fold: a cell is visited when a selected patch reads
# a cell of its block of 2**shift consecutive (NESTED) cells.  With
# MARK_SHIFT = None the shift is the smallest one whose footprint has at most
# MARK_MAX_F blocks per centre: single cells for N = 4, blocks of 4 cells for
# N = 8, of 16 cells for N = 16.  On an A100 (benchmarks/profile_masked_fold.py)
# the fold kernel took the same time with single cells (312 per centre at
# N = 16) and blocks of 16 cells (32 per centre), while building the mark of
# the single cells cost 0.15-0.45 ms more per call.
MARK_SHIFT = None
MARK_MAX_F = 48


def n_mark_blocks(K, shift):
    return (K + (1 << shift) - 1) >> shift


def build_footprint(idx, w, K, M, shift, chunk_entries=1 << 24):
    """Per-centre list of the blocks of ``2**shift`` cells its patch reads (see
    :meth:`PatchGeometry.footprint`).  Processed in chunks of centres to
    bound the temporary memory."""
    J, R = idx.shape
    GP = R // M
    nblk = n_mark_blocks(K, shift)
    per = max(1, chunk_entries // max(1, J * GP))
    parts, F = [], 1
    for m0 in range(0, M, per):
        m1 = min(M, m0 + per)
        v = (idx[:, m0 * GP:m1 * GP].long() >> shift)
        v = torch.where(w[:, m0 * GP:m1 * GP] != 0, v, torch.full_like(v, nblk))
        v = v.view(J, m1 - m0, GP).permute(1, 0, 2).reshape(m1 - m0, J * GP)
        v = torch.sort(v, dim=1).values
        first = torch.ones_like(v, dtype=torch.bool)
        first[:, 1:] = v[:, 1:] != v[:, :-1]
        first &= v != nblk
        cnt = first.sum(1)
        f = int(cnt.max()) if cnt.numel() else 0
        F = max(F, f)
        pos = torch.where(first, torch.cumsum(first, 1) - 1, torch.full_like(v, f))
        out = torch.full((m1 - m0, f + 1), nblk, dtype=torch.int64, device=idx.device)
        out.scatter_(1, pos, torch.where(first, v, torch.full_like(v, nblk)))
        parts.append(out[:, :f])
    fp = torch.full((M, F), nblk, dtype=torch.int32, device=idx.device)
    m0 = 0
    for part in parts:
        fp[m0:m0 + part.shape[0], :part.shape[1]] = part.to(torch.int32)
        m0 += part.shape[0]
    return fp


def selection_mark(geom, sel):
    """``([Bs, n_blocks + 1] uint8, shift)``: 1 for the blocks of ``2**shift``
    cells that a selected patch of row b of ``sel`` reads.  Every block
    receives the same value, so the scatter is deterministic."""
    fp, shift = geom.footprint()
    nblk = n_mark_blocks(geom.K, shift)
    blocks = fp[sel.long()].reshape(sel.shape[0], -1).long()          # [Bs, Ms*F]
    mark = torch.zeros(sel.shape[0], nblk + 1, dtype=torch.uint8, device=sel.device)
    mark.scatter_(1, blocks, 1)
    return mark, shift


def build_transposed_map(idx, w, K):
    """CSR transpose of the patch geometry (see :meth:`PatchGeometry.transposed`)."""
    J, R = idx.shape
    if R >= 2**31:
        raise ValueError("patch geometry too large for int32 row numbers.")
    nz = (w != 0).reshape(-1)
    src = idx.reshape(-1).long()[nz]
    rows = torch.arange(R, device=idx.device, dtype=torch.int64).repeat(J)[nz]
    tw = w.reshape(-1)[nz]
    # sort by (source cell, row): a fixed order of the entries of each cell
    key = src * R + rows
    order = torch.argsort(key)
    src, rows, tw = src[order], rows[order], tw[order]
    ptr = torch.searchsorted(src, torch.arange(K + 1, device=idx.device, dtype=torch.int64))
    return ptr, rows.to(torch.int32), tw.contiguous()


# ===========================================================================
# PyTorch backend
# ===========================================================================

def _torch_extract(x, geom, sel):
    """x [B, C, K] -> y [B, Msel, G, C, P]."""
    B, C, K = x.shape
    J, R = geom.idx.shape
    G, P, GP = geom.G, geom.P, geom.G * geom.P
    w = geom.w_as(x.dtype)
    idx = geom.idx64()
    # channels last: one row of B*C contiguous values per cell, so each
    # neighbour is a contiguous row copy (20-37x faster than gathering along
    # the last axis of [B, C, K] on a 16-core CPU)
    xt = x.reshape(B * C, K).t().contiguous()                     # [K, B*C]
    if sel is None:
        Ms = geom.M
        y = xt[idx[0]] * w[0, :, None]                             # [R, B*C]
        for j in range(1, J):
            y.addcmul_(xt[idx[j]], w[j, :, None])
    else:
        Ms = sel.shape[1]
        rows = (sel.long().unsqueeze(-1) * GP
                + torch.arange(GP, device=x.device)).reshape(sel.shape[0], -1)   # [Bs, Rs]
        if sel.shape[0] == 1:
            r = rows[0]
            y = xt[idx[0][r]] * w[0][r, None]                      # [Rs, B*C]
            for j in range(1, J):
                y.addcmul_(xt[idx[j][r]], w[j][r, None])
        else:                                                      # one draw per sample
            x3 = xt.view(K, B, C)
            bb = torch.arange(B, device=x.device).view(B, 1)
            y = x3[idx[0][rows], bb] * w[0][rows, None]            # [B, Rs, C]
            for j in range(1, J):
                y.addcmul_(x3[idx[j][rows], bb], w[j][rows, None])
            return y.view(B, Ms, G, P, C).transpose(3, 4).contiguous()
    return y.view(Ms, G, P, B, C).permute(3, 0, 1, 4, 2).contiguous()


def _torch_fold(y, geom, ones=False):
    """Raw adjoint: y [B, M, G, C, P] (all centres) -> x [B, C, K]."""
    B, M, G, C, P = y.shape
    J = geom.J
    w = geom.w_as(y.dtype)
    if ones:
        w = (w != 0).to(y.dtype)
    idx = geom.idx64()
    yt = y.permute(1, 2, 4, 0, 3).reshape(M * G * P, B * C)       # [R, B*C]
    out = y.new_zeros(geom.K, B * C)
    for j in range(J):
        out.index_add_(0, idx[j], yt * w[j, :, None])
    return out.t().reshape(B, C, geom.K)


# ===========================================================================
# Triton backend
# ===========================================================================

def _next_pow2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def _block_c(C):
    return min(32, max(2, _next_pow2(C)))


if _HAS_TRITON:

    _GATHER_CONFIGS = [
        triton.Config({"BLOCK_R": br}, num_warps=nw)
        for br, nw in ((128, 4), (256, 8))
    ]

    @triton.autotune(configs=_GATHER_CONFIGS, key=["C", "J", "P", "CL", "SIZE"])
    @triton.jit
    def _gather_kernel(x_ptr, idx_ptr, w_ptr, sel_ptr, y_ptr,
                       C, K, R_FULL, G, P, NSEL, SEL_SB, SIZE,
                       J: tl.constexpr, HAS_SEL: tl.constexpr, CL: tl.constexpr,
                       BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        """y[b, i, g, c, p] = sum_j w[j, r] x[b, c, idx[j, r]], r = (sel[b, i]*G + g)*P + p."""
        pid_r = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        pid_c = tl.program_id(2)
        GP = G * P
        r = pid_r.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        r_ok = r < NSEL.to(tl.int64) * GP
        i = r // GP
        q = r - i * GP
        if HAS_SEL:
            m = tl.load(sel_ptr + b * SEL_SB + i, mask=r_ok, other=0).to(tl.int64)
        else:
            m = i
        row = m * GP + q
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        ok = r_ok[:, None] & c_ok[None, :]
        if CL:   # x is [B, K, C]: one contiguous run of C values per neighbour
            x_base = x_ptr + b * K * C + c[None, :]
        else:    # x is [B, C, K]
            x_base = x_ptr + (b * C + c.to(tl.int64))[None, :] * K
        acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(J):
            ix = tl.load(idx_ptr + j * R_FULL + row, mask=r_ok, other=0).to(tl.int64)
            wv = tl.load(w_ptr + j * R_FULL + row, mask=r_ok, other=0.0).to(tl.float32)
            if CL:
                v = tl.load(x_base + ix[:, None] * C, mask=ok, other=0.0).to(tl.float32)
            else:
                v = tl.load(x_base + ix[:, None], mask=ok, other=0.0).to(tl.float32)
            acc += wv[:, None] * v
        g = q // P
        p = q - g * P
        out = (((b * NSEL + i) * G + g) * C)[:, None] + c[None, :]
        tl.store(y_ptr + out * P + p[:, None], acc.to(y_ptr.dtype.element_ty), mask=ok)

    _FOLD_CONFIGS = [
        triton.Config({"BLOCK_S": bs}, num_warps=nw)
        for bs, nw in ((64, 4), (128, 4))
    ]

    @triton.autotune(configs=_FOLD_CONFIGS, key=["C", "P", "HAS_SEL", "HAS_MARK", "CL", "SIZE"])
    @triton.jit
    def _fold_kernel(y_ptr, sel_ptr, ptr_ptr, rows_ptr, tw_ptr, x_ptr, scale_ptr, mark_ptr,
                     C, K, NY, G, P, SEL_SB, MARK_SB, SIZE,
                     HAS_SEL: tl.constexpr, NORM: tl.constexpr, WRITE_SCALE: tl.constexpr,
                     CL: tl.constexpr, HAS_MARK: tl.constexpr, MARK_SHIFT: tl.constexpr,
                     BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
        """x[b, c, s] = sum_e tw[e] y[b, m(e), g(e), c, p(e)] / norm[b, s], no atomics.

        ``y`` is ``[B, NY, G, C, P]`` (``[B, NY, G, P, C]`` with ``CL``), and row
        ``r = (m*G + g)*P + p`` of the map reads patch ``m``.  ``HAS_SEL``:
        ``sel[b, m]`` is 1 for the centres selected in sample ``b``; the entries
        of the others count as 0.  The selection byte and the patch value are
        loaded side by side (the value of a centre not selected is read and
        discarded), so no load waits on the selection.  ``NORM``: 0 none, 1
        sum of the weights read, 2 their number; computed on the fly from the
        same entries (written to ``scale`` with ``WRITE_SCALE``, for the
        backward pass).

        ``HAS_MARK``: ``mark[b, s >> MARK_SHIFT]`` is 0 for the cells that no
        selected patch reads; their entries are not visited (they would all
        count as 0), and they get 0.  The cells that are visited run the same
        loop in the same order, so the result is bitwise the same with and
        without the mark.
        """
        pid_s = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        pid_c = tl.program_id(2)
        s = pid_s.to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S)
        s_ok = s < K
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        if HAS_MARK:
            mk = tl.load(mark_ptr + b * MARK_SB + (s >> MARK_SHIFT), mask=s_ok, other=0)
            act = s_ok & (mk != 0)
        else:
            act = s_ok
        start = tl.load(ptr_ptr + s, mask=act, other=0)
        n = tl.load(ptr_ptr + s + 1, mask=act, other=0) - start
        n_max = tl.max(n, axis=0)
        y_base = y_ptr + b * NY * G * C * P
        acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        nacc = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for jj in range(0, n_max):
            e_ok = s_ok & (jj < n)
            e = start + jj
            r = tl.load(rows_ptr + e, mask=e_ok, other=0).to(tl.int64)
            mg = r // P
            p = r - mg * P
            yrow = mg
            wv = tl.load(tw_ptr + e, mask=e_ok, other=0.0).to(tl.float32)
            if CL:   # y is [B, NY, G, P, C]
                off = ((yrow * P + p) * C)[:, None] + c[None, :]
            else:
                off = (yrow * C)[:, None] * P + c[None, :] * P + p[:, None]
            v = tl.load(y_base + off, mask=e_ok[:, None] & c_ok[None, :], other=0.0)
            if HAS_SEL:
                sb = tl.load(sel_ptr + b * SEL_SB + mg // G, mask=e_ok, other=0)
                ok = e_ok & (sb != 0)
                wv = tl.where(ok, wv, 0.0)
                v = tl.where(ok[:, None], v, 0.0)
            else:
                ok = e_ok
            if NORM == 1:
                nacc += wv
            if NORM == 2:
                nacc += tl.where(ok, 1.0, 0.0)
            acc += wv[:, None] * v.to(tl.float32)
        if NORM != 0:
            sc = tl.where(nacc > 0, 1.0 / tl.where(nacc > 0, nacc, 1.0), 0.0)
            acc = acc * sc[:, None]
            if WRITE_SCALE:
                tl.store(scale_ptr + b * K + s, sc, mask=s_ok & (pid_c == 0))
        tl.store(x_ptr + (b * C + c.to(tl.int64))[None, :] * K + s[:, None],
                 acc.to(x_ptr.dtype.element_ty), mask=s_ok[:, None] & c_ok[None, :])


def _device_ctx(t):
    # a CPU tensor reaches the kernels only under TRITON_INTERPRET=1 (tests)
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()


def _size_class(n):
    """Autotune key: one bucket per factor of 4 of the problem size."""
    return max(0, int(n).bit_length() // 2)


# Channels-last staging: copy the input to [B, K, C] (gather) or the patches to
# [B, NY, G, P, C] (fold) when C >= these thresholds, so that every neighbour
# read is one contiguous run of C values instead of C scattered words.
# A100, float32 (benchmarks/benchmark_patches.py, configurations gpu_short):
# the staged fold is 1.3-3.4x faster for every C (3 and 16 measured); the
# staged gather is 1.2-1.6x faster for C = 16 and 1.15x slower for C = 3.
GATHER_CL_MIN_C = 8
FOLD_CL_MIN_C = 1


def _triton_extract(x, geom, sel):
    B, C, K = x.shape
    cl = C >= GATHER_CL_MIN_C
    x = x.transpose(1, 2).contiguous() if cl else x.contiguous()
    G, P = geom.G, geom.P
    if sel is None:
        nsel, sel_t, sel_sb = geom.M, geom.idx, 0      # sel_ptr unused
    else:
        nsel = sel.shape[1]
        sel_t = sel
        sel_sb = sel.stride(0) if sel.shape[0] > 1 else 0
    y = torch.empty(B, nsel, G, C, P, device=x.device, dtype=x.dtype)
    if y.numel() == 0:
        return y
    BLOCK_C = _block_c(C)
    R_full = geom.idx.shape[1]

    def grid(meta):
        return (triton.cdiv(nsel * G * P, meta["BLOCK_R"]), B, triton.cdiv(C, BLOCK_C))

    with _device_ctx(x):
        _gather_kernel[grid](
            x, geom.idx, geom.w, sel_t, y,
            C, K, R_full, G, P, nsel, sel_sb, _size_class(nsel * G * P),
            J=geom.J, HAS_SEL=sel is not None, CL=cl, BLOCK_C=BLOCK_C,
        )
    return y


def _triton_fold(y, geom, sel=None, norm="none", want_scale=False, mark=None):
    """y [B, NY, G, C, P] -> (x [B, C, K], scale [B, K] or None).

    ``sel`` [Bs, Ms]: the distinct centres of the ``NY = Ms`` patches of ``y``
    (None when ``y`` holds all ``M`` centres).  The patches are copied, in
    the channels-last layout, into a buffer of all ``M`` centres that is not
    initialised for the others; a ``[Bs, M]`` selection byte tells the kernel
    which entries count.  ``mark`` (from :func:`selection_mark`) restricts
    the fold to the cells the selection reads.  The normalisation is
    computed in the kernel, from the entries it reads.
    """
    B, NY, G, C, P = y.shape
    K = geom.K
    ptr, rows, tw = geom.transposed()
    x = torch.empty(B, C, K, device=y.device, dtype=y.dtype)
    mode = {"none": 0, "weight": 1, "count": 2}[norm]
    scale = (torch.empty(B, K, device=y.device, dtype=torch.float32)
             if (want_scale and mode) else None)
    if x.numel() == 0:
        return x, scale
    mark_t, shift = mark if mark is not None else (None, 0)
    mark_sb = mark_t.stride(0) if (mark_t is not None and mark_t.shape[0] > 1) else 0
    sbit, sel_sb = None, 0
    if sel is not None:
        # the selected patches at the position of their centre, channels-last
        # (one copy, as the channels-last staging), the others not initialised
        cl = True
        M = geom.M
        full = torch.empty(B, M, G, P, C, device=y.device, dtype=y.dtype)
        src = y.transpose(3, 4)
        if sel.shape[0] == 1:
            full[:, sel[0].long()] = src
        else:
            bb = torch.arange(B, device=y.device).view(B, 1)
            full[bb, sel.long()] = src
        y, NY = full, M
        sbit = torch.zeros(sel.shape[0], M, dtype=torch.uint8, device=y.device)
        sbit.scatter_(1, sel.long(), 1)
        sel_sb = M if sel.shape[0] > 1 else 0
    else:
        cl = C >= FOLD_CL_MIN_C
        y = y.transpose(3, 4).contiguous() if cl else y.contiguous()
    BLOCK_C = _block_c(C)

    def grid(meta):
        return (triton.cdiv(K, meta["BLOCK_S"]), B, triton.cdiv(C, BLOCK_C))

    with _device_ctx(y):
        _fold_kernel[grid](
            y, sbit if sbit is not None else ptr, ptr, rows, tw, x,
            scale if scale is not None else tw, mark_t if mark_t is not None else ptr,
            C, K, NY, G, P, sel_sb, mark_sb, _size_class(K),
            HAS_SEL=sbit is not None, NORM=mode, WRITE_SCALE=scale is not None, CL=cl,
            HAS_MARK=mark_t is not None, MARK_SHIFT=shift, BLOCK_C=BLOCK_C,
        )
    return x, scale


# ===========================================================================
# Dispatch
# ===========================================================================

def use_triton(x, backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == "torch":
        return False
    ok = _HAS_TRITON and x.is_cuda and x.dtype in _TRITON_DTYPES
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, a CUDA tensor and float32, float16 "
            "or bfloat16 data (float64 runs on backend='torch')."
        )
    return ok


def _expand_selected(y, sel, M):
    """[B, Ms, G, C, P] -> [B, M, G, C, P], zeros for the centres not selected
    (and sums for a centre selected twice)."""
    B = y.shape[0]
    full = y.new_zeros((B, M) + tuple(y.shape[2:]))
    b_ids = torch.arange(B, device=y.device).view(B, 1).expand(B, sel.shape[1])
    full.index_put_((b_ids, sel.long().expand(B, -1)), y, accumulate=True)
    return full


def has_repeats(sel):
    """True when a row of ``sel`` repeats a centre (one device sync)."""
    if sel.shape[1] < 2:
        return False
    srt = torch.sort(sel, dim=1).values
    return bool((srt[:, 1:] == srt[:, :-1]).any())


def _extract(x, geom, sel, use_tri):
    if use_tri:
        return _triton_extract(x, geom, sel)
    return _torch_extract(x, geom, sel)


# Masked fold (Triton): visit only the cell blocks that the selected patches
# read, instead of every cell of the transposed map.  False restores the walk
# over the whole map (same result, bit for bit; kept for tests and benchmarks).
MASKED_FOLD_SKIP = True


def _fold(y, geom, sel, norm, use_tri, want_scale):
    """Normalised fold; returns (x, scale) with scale [K] or [B, K] or None."""
    if use_tri:
        y = y.contiguous()
        mark = None
        if sel is not None:
            if has_repeats(sel):
                # the copies are summed in a full tensor of all M centres
                y, sel = _expand_selected(y, sel, geom.M), None
            elif MASKED_FOLD_SKIP:
                mark = selection_mark(geom, sel)
        return _triton_fold(y, geom, sel=sel, norm=norm, want_scale=want_scale, mark=mark)
    scale = fold_scale(geom, norm, sel, y.shape[0], y.device, y.dtype)
    yy = _expand_selected(y, sel, geom.M) if sel is not None else y
    out = _torch_fold(yy, geom)
    if scale is not None:
        out = out * scale.view(-1, 1, geom.K)
    return out, scale


@dataclass(frozen=True)
class _Options:
    use_triton: bool


class PatchExtractFunction(torch.autograd.Function):
    """``x [B, C, K] -> y [B, Msel, G, C, P]``; its backward is the raw fold."""

    @staticmethod
    def forward(ctx, x, geom, sel, opts):
        ctx.geom, ctx.opts = geom, opts
        ctx.save_for_backward(*([sel] if sel is not None else []))
        ctx.has_sel = sel is not None
        return _extract(x, geom, sel, opts.use_triton)

    @staticmethod
    def backward(ctx, gy):
        sel = ctx.saved_tensors[0] if ctx.has_sel else None
        gx = None
        if ctx.needs_input_grad[0]:
            gx = PatchFoldFunction.apply(gy, ctx.geom, sel, "none", ctx.opts)
        return gx, None, None, None


class PatchFoldFunction(torch.autograd.Function):
    """``y [B, Msel, G, C, P] -> x [B, C, K] = scale * P^T y``; its backward is
    an extraction of ``scale * gx``."""

    @staticmethod
    def forward(ctx, y, geom, sel, norm, opts):
        ctx.geom, ctx.opts = geom, opts
        x, scale = _fold(y, geom, sel, norm, opts.use_triton, ctx.needs_input_grad[0])
        ctx.has_sel = sel is not None
        ctx.has_scale = scale is not None
        ctx.save_for_backward(*[t for t in (sel, scale) if t is not None])
        return x

    @staticmethod
    def backward(ctx, gx):
        saved = list(ctx.saved_tensors)
        sel = saved.pop(0) if ctx.has_sel else None
        scale = saved.pop(0) if ctx.has_scale else None
        gy = None
        if ctx.needs_input_grad[0]:
            g = gx if scale is None else gx * scale.to(gx.dtype).view(-1, 1, ctx.geom.K)
            gy = PatchExtractFunction.apply(g, ctx.geom, sel, ctx.opts)
        return gy, None, None, None, None


def fold_scale(geom, norm, sel, B, device, dtype):
    """PyTorch backend: ``1 / norm`` of the fold (0 where no patch reads a
    cell), [K] or [B, K]."""
    if norm == "none":
        return None
    if sel is None:
        n = geom.norm(norm).to(device=device, dtype=dtype)
    else:
        ones = torch.ones((B, sel.shape[1], geom.G, 1, geom.P), device=device, dtype=dtype)
        n = _torch_fold(_expand_selected(ones, sel, geom.M), geom,
                        ones=(norm == "count")).view(B, geom.K)
    return torch.where(n > 0, 1.0 / n.clamp_min(torch.finfo(n.dtype).tiny), torch.zeros_like(n))
