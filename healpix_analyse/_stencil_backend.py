"""Compute backends of the neighbourhood reductions and of the local gradient.

Both operators are stencil operations: each output cell gathers the values of
a small set of input cells (its neighbourhood, at most ``KP`` of them) and
reduces them.  The neighbourhood is stored once, padded::

    idx[n, k]   int32 position in the input of the k-th neighbour of cell n
    cnt[n]      number of valid neighbours of cell n (k < cnt[n])

Neither backend materialises the gathered tensor ``[B, N, KP]`` for the
reductions that do not need all the values at once.

``"torch"`` (CPU, integer data, the reference everywhere)
    A loop over the KP taps: ``index_select`` of one tap ``[B, N]`` at a
    time, combined into running sums / extrema.  The median needs all the
    values of a row: it sorts rows in chunks of bounded size.  The backward
    passes are written by hand (``index_add_`` / ``scatter_add_``), and are
    deterministic on CPU.

``"triton"`` (CUDA, float16 / bfloat16 / float32 / float64)
    One program per (batch row, block of output cells).  The tap tile
    ``[BLOCK_N, KP]`` is gathered in registers and reduced there: sums with
    fp32 accumulation (fp64 for float64), extrema with the index of the
    selected tap, the median by ranking the KP values of each row against one
    another (a comparison network in registers, ``KP <= 64``).  Backward: a
    gather on the transposed stencil (for each input cell, its incoming
    (output cell, tap) entries in CSR form), with no atomics, so the result
    is the same at every run.

Subgradients of min, max and median follow torch: the whole gradient goes to
the selected value, half to each of the two central values for an even
median, nothing for a row that contains NaN (median) and the first NaN for
min/max.  Ties go to the first neighbour in the order of the input.

Triton is optional: without it (or without CUDA) everything runs on the
``"torch"`` backend.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from healpix_analyse._conv_backend import _HAS_TRITON, triton, tl

__all__ = [
    "BACKENDS",
    "triton_available",
    "Stencil",
    "stencil_reduce",
    "GradientStencil",
    "lsq_gradient",
]

BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_GRAD_TRITON_DTYPES = (torch.float32, torch.float64)
FAST_REDUCTIONS = ("sum", "mean", "min", "max", "median")
_OPS = {"sum": 0, "mean": 1, "max": 2, "min": 3, "median": 4}

# Triton tile limits: wider neighbourhoods run on the torch backend.
_MAX_KP_TRITON = 256
_MAX_KP_MEDIAN_TRITON = 64
# Elements of one chunk of the materialised median / generic path.
_CHUNK_ELEMENTS = 1 << 24
# Longest rows of the rank-based median of the torch backend (KP^2 per cell).
_MAX_KP_RANK = 32


def triton_available() -> bool:
    """True when the Triton kernels can run (Triton importable and CUDA present)."""
    return _HAS_TRITON and torch.cuda.is_available()


def _device_ctx(device):
    """CUDA device guard (a no-op on CPU, for the Triton interpreter)."""
    if device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _next_pow2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def _acc_dtype(dtype):
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


_WARNED_NO_TRITON = False


def _warn_no_triton():
    global _WARNED_NO_TRITON
    if not _WARNED_NO_TRITON:
        _WARNED_NO_TRITON = True
        warnings.warn(
            "Neighbourhood reductions run on CUDA without Triton: the PyTorch "
            "backend is correct but slower on GPUs. Install the optional "
            "'triton' package (pip install healpix-analyse[triton]) for the "
            "fused kernels.",
            RuntimeWarning,
            stacklevel=4,
        )


def _use_triton(x, backend, dtypes, kp_ok=True):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == "torch":
        return False
    if x.is_cuda and not _HAS_TRITON and x.dtype in dtypes:
        _warn_no_triton()
    ok = _HAS_TRITON and x.is_cuda and x.dtype in dtypes and kp_ok
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, a CUDA tensor, a supported floating "
            "dtype and a neighbourhood small enough for the register tiles."
        )
    return ok


# ===========================================================================
# Stencil geometry
# ===========================================================================

def _transposed(src, n_of, n_in):
    """CSR of the entries keyed by their source: ``(ptr, order)``.

    ``ptr`` is int64 ``[n_in + 1]``; ``order`` sorts the entries by source,
    stably, so that the entries of one source keep their (cell, tap) order.
    """
    order = torch.argsort(src, stable=True)
    counts = torch.bincount(src, minlength=n_in)
    ptr = torch.zeros(n_in + 1, dtype=torch.int64, device=src.device)
    torch.cumsum(counts, 0, out=ptr[1:])
    return ptr, order


@dataclass
class _StencilOnDevice:
    idx: torch.Tensor          # int32 [N, KP], pads -> n_in
    cnt: torch.Tensor          # int32 [N]
    tmap: tuple | None = None  # (ptr int64, tn int32, tk int16)


class Stencil:
    """Padded neighbourhood positions, cached per device.

    Parameters
    ----------
    offsets : int64 array [N + 1]
        CSR row offsets.
    positions : int array [nnz]
        Input positions of the neighbours, row after row.
    n_in : int
        Size of the input (last dimension of the values).
    """

    def __init__(self, offsets, positions, n_in):
        offsets = np.asarray(offsets, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        self.n_out = int(offsets.size - 1)
        self.n_in = int(n_in)
        if self.n_in >= 2**31 - 1:
            raise ValueError("too many input cells for int32 positions.")
        cnt = np.diff(offsets)
        self.kp = int(cnt.max()) if cnt.size else 0
        idx = np.full((self.n_out, self.kp), self.n_in, dtype=np.int32)
        if positions.size:
            rows = np.repeat(np.arange(self.n_out), cnt)
            cols = np.arange(positions.size) - np.repeat(offsets[:-1], cnt)
            idx[rows, cols] = positions
        self.idx = idx
        self.cnt = cnt.astype(np.int32)
        self._dev: dict = {}

    def on(self, device) -> _StencilOnDevice:
        device = torch.device(device)
        key = str(device)
        if key not in self._dev:
            self._dev[key] = _StencilOnDevice(
                idx=torch.as_tensor(self.idx, device=device),
                cnt=torch.as_tensor(self.cnt, device=device),
            )
        return self._dev[key]

    def transposed(self, device):
        """``(ptr, tn, tk)``: for input cell j, the entries ``ptr[j]:ptr[j+1]``
        hold the output cells ``tn`` and taps ``tk`` that read j."""
        sd = self.on(device)
        if sd.tmap is None:
            kp = self.kp
            k = torch.arange(kp, device=sd.idx.device, dtype=torch.int32)
            valid = k[None, :] < sd.cnt[:, None]
            src = sd.idx[valid].long()
            n_of = torch.arange(self.n_out, device=src.device, dtype=torch.int32)
            n_of = n_of[:, None].expand(-1, kp)[valid]
            k_of = k[None, :].expand(self.n_out, -1)[valid]
            ptr, order = _transposed(src, n_of, self.n_in)
            dt = torch.int16 if kp < 2**15 else torch.int32
            sd.tmap = (ptr, n_of[order].contiguous(), k_of[order].to(dt).contiguous())
        return sd.tmap


# ===========================================================================
# Reductions: torch backend
# ===========================================================================

def _extend(x, fill):
    """Append one column of ``fill``: the target of the padded taps."""
    return torch.cat([x, x.new_full((x.shape[0], 1), fill)], dim=1)


def _torch_sum(x, sd, mean):
    acc = _acc_dtype(x.dtype)
    B = x.shape[0]
    N, KP = sd.idx.shape
    if KP == 0:
        return torch.zeros(B, N, dtype=x.dtype, device=x.device)
    xe = _extend(x.to(acc), 0.0)
    out = xe.index_select(1, sd.idx[:, 0].long())
    for k in range(1, KP):
        out += xe.index_select(1, sd.idx[:, k].long())
    if mean:
        out = out / sd.cnt.to(acc)
    return out.to(x.dtype)


def _torch_extremum(x, sd, is_max):
    """Running extremum with the selected tap (first NaN, else first extremum)."""
    xe = _extend(x, float("-inf") if is_max else float("inf"))
    KP = sd.idx.shape[1]
    best = xe.index_select(1, sd.idx[:, 0].long())
    sel = torch.zeros(best.shape, dtype=torch.int16, device=x.device)
    best_nan = torch.isnan(best)
    for k in range(1, KP):
        v = xe.index_select(1, sd.idx[:, k].long())
        v_nan = torch.isnan(v)
        better = (v > best) if is_max else (v < best)
        upd = better | (v_nan & ~best_nan)
        best = torch.where(upd, v, best)
        sel.masked_fill_(upd, k)
        best_nan |= v_nan
    return best, sel


def _row_chunks(n_rows, width):
    step = max(1, _CHUNK_ELEMENTS // max(1, width))
    for r0 in range(0, n_rows, step):
        yield r0, min(n_rows, r0 + step)


def _tap_order(kp, device):
    """``[KP(i), KP(j)]`` bool: ``j < i``."""
    k = torch.arange(kp, device=device)
    return k[None, :] < k[:, None]


def _torch_median(x, sd, need_sel):
    B = x.shape[0]
    N, KP = sd.idx.shape
    xe = _extend(x, float("inf"))
    out = torch.empty(B, N, dtype=x.dtype, device=x.device)
    sel = (torch.empty(2, B, N, dtype=torch.int16, device=x.device)
           if need_sel else None)
    cnt = sd.cnt.long()
    rank_path = KP <= _MAX_KP_RANK
    for r0, r1 in _row_chunks(N, B * KP * (KP if rank_path else 1)):
        g = xe[:, sd.idx[r0:r1].long()]                     # [B, R, KP]
        c = cnt[r0:r1]
        i_lo = ((c - 1) // 2).expand(B, -1)[..., None]
        i_hi = (c // 2).expand(B, -1)[..., None]
        if rank_path:
            # rank of every value in its row, ties broken by tap (= a stable
            # sort); the +inf pads sit after the valid taps, so the valid
            # values take the ranks 0..cnt-1.  Faster than torch.sort on CPU
            # for short rows.
            less = (g[..., None, :] < g[..., :, None]) | (
                (g[..., None, :] == g[..., :, None]) & _tap_order(KP, g.device))
            rank = less.sum(dim=-1)                         # [B, R, KP]
            s_lo = torch.argmax((rank == i_lo).to(torch.uint8), dim=-1, keepdim=True)
            s_hi = torch.argmax((rank == i_hi).to(torch.uint8), dim=-1, keepdim=True)
        else:
            _, si = torch.sort(g, dim=-1, stable=True)
            s_lo = si.gather(-1, i_lo)
            s_hi = si.gather(-1, i_hi)
        lo = g.gather(-1, s_lo)[..., 0]
        hi = g.gather(-1, s_hi)[..., 0]
        odd = (c % 2 == 1)
        val = torch.where(odd, hi, (lo + hi) / 2)
        has_nan = torch.isnan(g).any(dim=-1)
        out[:, r0:r1] = torch.where(has_nan, torch.full_like(val, float("nan")), val)
        if need_sel:
            s_lo = s_lo[..., 0].to(torch.int16)
            s_hi = s_hi[..., 0].to(torch.int16)
            s_lo = torch.where(odd | has_nan, torch.full_like(s_lo, -1), s_lo)
            s_hi = torch.where(has_nan, torch.full_like(s_hi, -1), s_hi)
            sel[0, :, r0:r1] = s_lo
            sel[1, :, r0:r1] = s_hi
    return out, sel


def _torch_reduce(x, sd, op, need_sel):
    if op in ("sum", "mean"):
        return _torch_sum(x, sd, op == "mean"), None
    if op in ("max", "min"):
        best, sel = _torch_extremum(x, sd, op == "max")
        return best, (sel if need_sel else None)
    if op == "median":
        return _torch_median(x, sd, need_sel)
    raise AssertionError(op)


def _torch_reduce_backward(gy, sd, op, sel, n_in, dtype):
    """Hand-written backward (differentiable itself: double backward works)."""
    acc = _acc_dtype(dtype)
    B, N = gy.shape
    KP = sd.idx.shape[1]
    gy = gy.to(acc)
    gx = torch.zeros(B, n_in + 1, dtype=acc, device=gy.device)
    if op in ("sum", "mean"):
        w = gy / sd.cnt.to(acc) if op == "mean" else gy
        for k in range(KP):
            gx = gx.index_add(1, sd.idx[:, k].long(), w)
    else:
        idx = sd.idx.long()[None].expand(B, -1, -1)
        if op in ("max", "min"):
            pos = idx.gather(2, sel.long()[..., None])[..., 0]
            gx = gx.scatter_add(1, pos, gy)
        else:  # median
            s_lo, s_hi = sel[0].long(), sel[1].long()
            pos_lo = idx.gather(2, s_lo.clamp_min(0)[..., None])[..., 0]
            pos_hi = idx.gather(2, s_hi.clamp_min(0)[..., None])[..., 0]
            even = s_lo >= 0
            w_hi = torch.where(s_hi < 0, torch.zeros_like(gy),
                               torch.where(even, gy / 2, gy))
            w_lo = torch.where(even, gy / 2, torch.zeros_like(gy))
            gx = gx.scatter_add(1, pos_hi, w_hi).scatter_add(1, pos_lo, w_lo)
    return gx[:, :n_in].to(dtype)


# ===========================================================================
# Triton kernels
# ===========================================================================

if _HAS_TRITON:

    @triton.jit
    def _reduce_fwd_kernel(x_ptr, idx_ptr, cnt_ptr, y_ptr, sel_ptr,
                           N_IN, N_OUT, KP, B, NBLK,
                           OP: tl.constexpr, NEED_SEL: tl.constexpr, ACC64: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """y[b, n] = reduce_k x[b, idx[n, k]] over k < cnt[n]."""
        pid = tl.program_id(0)
        b = (pid // NBLK).to(tl.int64)
        n = (pid % NBLK) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < N_OUT
        cnt = tl.load(cnt_ptr + n, mask=n_ok, other=0)
        k = tl.arange(0, BLOCK_K)
        m = n_ok[:, None] & (k[None, :] < cnt[:, None])
        src = tl.load(idx_ptr + n.to(tl.int64)[:, None] * KP + k[None, :], mask=m, other=0)
        v = tl.load(x_ptr + b * N_IN + src.to(tl.int64), mask=m, other=0.0)
        y_off = b * N_OUT + n
        if OP <= 1:
            if ACC64:
                vf = v.to(tl.float64)
            else:
                vf = v.to(tl.float32)
            s = tl.sum(tl.where(m, vf, 0.0), axis=1)
            if OP == 1:
                s = s / cnt.to(s.dtype)
            tl.store(y_ptr + y_off, s.to(y_ptr.dtype.element_ty), mask=n_ok)
        else:
            isn = m & (v != v)
            has_nan = tl.max(isn.to(tl.int32), axis=1) > 0
            nan_v = tl.sum(tl.where(isn, v, 0.0), axis=1)          # NaN iff has_nan
            kk = k[None, :]
            if OP == 4:
                # rank of each value among the valid ones of its row, ties
                # broken by tap: a strict total order, so ranks are 0..cnt-1
                vi = v[:, :, None]
                vj = v[:, None, :]
                ki = kk[:, :, None]
                kj = kk[:, None, :]
                less = (vj < vi) | ((vj == vi) & (kj < ki))
                less = less & m[:, None, :]
                rank = tl.sum(less.to(tl.int32), axis=2)
                r_lo = (cnt - 1) // 2
                r_hi = cnt // 2
                hit_lo = m & (rank == r_lo[:, None])
                hit_hi = m & (rank == r_hi[:, None])
                lo = tl.max(tl.where(hit_lo, v, float("-inf")), axis=1)
                hi = tl.max(tl.where(hit_hi, v, float("-inf")), axis=1)
                odd = (cnt % 2) == 1
                val = tl.where(odd, hi, (lo + hi) / 2)
                out = tl.where(has_nan, nan_v, val)
                tl.store(y_ptr + y_off, out.to(y_ptr.dtype.element_ty), mask=n_ok)
                if NEED_SEL:
                    s_lo = tl.min(tl.where(hit_lo, kk, BLOCK_K), axis=1)
                    s_hi = tl.min(tl.where(hit_hi, kk, BLOCK_K), axis=1)
                    s_lo = tl.where(odd | has_nan, -1, s_lo)
                    s_hi = tl.where(has_nan, -1, s_hi)
                    tl.store(sel_ptr + y_off, s_lo.to(tl.int16), mask=n_ok)
                    tl.store(sel_ptr + B * N_OUT + y_off, s_hi.to(tl.int16), mask=n_ok)
            else:
                ok = m & (v == v)
                if OP == 2:
                    vv = tl.where(ok, v, float("-inf"))
                    best = tl.max(vv, axis=1)
                else:
                    vv = tl.where(ok, v, float("inf"))
                    best = tl.min(vv, axis=1)
                cand = tl.where(has_nan[:, None], isn, m & (vv == best[:, None]))
                sel = tl.min(tl.where(cand, kk, BLOCK_K), axis=1)
                # the value at the selected tap, exactly (sign of zero included)
                picked = tl.max(tl.where(kk == sel[:, None], v, float("-inf")), axis=1)
                out = tl.where(has_nan, nan_v, picked)
                tl.store(y_ptr + y_off, out.to(y_ptr.dtype.element_ty), mask=n_ok)
                if NEED_SEL:
                    tl.store(sel_ptr + y_off, sel.to(tl.int16), mask=n_ok)

    @triton.jit
    def _reduce_bwd_kernel(gy_ptr, ptr_ptr, tn_ptr, tk_ptr, cnt_ptr, sel_ptr, gx_ptr,
                           N_IN, N_OUT, B, NBLK,
                           OP: tl.constexpr, ACC64: tl.constexpr, BLOCK: tl.constexpr):
        """gx[b, j] = sum over the entries (n, k) that read j of gy[b, n] * w(n, k)."""
        pid = tl.program_id(0)
        b = (pid // NBLK).to(tl.int64)
        j = (pid % NBLK) * BLOCK + tl.arange(0, BLOCK)
        j_ok = j < N_IN
        start = tl.load(ptr_ptr + j, mask=j_ok, other=0)
        n_e = tl.load(ptr_ptr + j + 1, mask=j_ok, other=0) - start
        n_max = tl.max(n_e, axis=0)
        if ACC64:
            acc = tl.zeros((BLOCK,), dtype=tl.float64)
        else:
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for e in range(0, n_max):
            mm = j_ok & (e < n_e)
            ent = start + e
            nn = tl.load(tn_ptr + ent, mask=mm, other=0).to(tl.int64)
            g = tl.load(gy_ptr + b * N_OUT + nn, mask=mm, other=0.0).to(acc.dtype)
            if OP == 0:
                acc += g
            elif OP == 1:
                c = tl.load(cnt_ptr + nn, mask=mm, other=1).to(acc.dtype)
                acc += g / c
            elif OP == 4:
                kk = tl.load(tk_ptr + ent, mask=mm, other=-2).to(tl.int32)
                s_lo = tl.load(sel_ptr + b * N_OUT + nn, mask=mm, other=-1).to(tl.int32)
                s_hi = tl.load(sel_ptr + B * N_OUT + b * N_OUT + nn, mask=mm, other=-1).to(tl.int32)
                half = g / 2
                w = tl.where(s_lo >= 0,
                             tl.where(s_lo == kk, half, 0.0) + tl.where(s_hi == kk, half, 0.0),
                             tl.where(s_hi == kk, g, 0.0))
                acc += w
            else:
                kk = tl.load(tk_ptr + ent, mask=mm, other=-2).to(tl.int32)
                s = tl.load(sel_ptr + b * N_OUT + nn, mask=mm, other=-1).to(tl.int32)
                acc += tl.where(s == kk, g, 0.0)
        tl.store(gx_ptr + b * N_IN + j, acc.to(gx_ptr.dtype.element_ty), mask=j_ok)


def _fwd_blocks(op, kp):
    bk = max(2, _next_pow2(kp))
    if op == "median":
        bn = max(2, min(128, 8192 // (bk * bk)))
    else:
        bn = max(16, min(256, 4096 // bk))
    return bn, bk


def _triton_reduce(x, sd, op, need_sel):
    B, n_in = x.shape
    N, KP = sd.idx.shape
    y = torch.empty(B, N, dtype=x.dtype, device=x.device)
    if N == 0 or B == 0:
        sel = None
        if need_sel and op in ("min", "max", "median"):
            shape = (2, B, N) if op == "median" else (B, N)
            sel = torch.empty(shape, dtype=torch.int16, device=x.device)
        return y, sel
    sel = None
    if op in ("min", "max"):
        sel = torch.empty(B, N, dtype=torch.int16, device=x.device) if need_sel else y
    elif op == "median":
        sel = torch.empty(2, B, N, dtype=torch.int16, device=x.device) if need_sel else y
    bn, bk = _fwd_blocks(op, KP)
    nblk = triton.cdiv(N, bn)
    with _device_ctx(x.device):
        _reduce_fwd_kernel[(B * nblk,)](
            x, sd.idx, sd.cnt, y, sel if sel is not None else y,
            n_in, N, KP, B, nblk,
            OP=_OPS[op], NEED_SEL=bool(need_sel and sel is not None),
            ACC64=x.dtype == torch.float64, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=4,
        )
    return y, (sel if need_sel and op in ("min", "max", "median") else None)


def _triton_reduce_backward(gy, stencil, sd, op, sel, n_in, dtype):
    B, N = gy.shape
    ptr, tn, tk = stencil.transposed(gy.device)
    gy = gy.to(dtype).contiguous()
    gx = torch.empty(B, n_in, dtype=dtype, device=gy.device)
    if B == 0 or n_in == 0:
        return gx
    block = 128
    nblk = triton.cdiv(n_in, block)
    with _device_ctx(gy.device):
        _reduce_bwd_kernel[(B * nblk,)](
            gy, ptr, tn, tk, sd.cnt, sel if sel is not None else gy, gx,
            n_in, N, B, nblk,
            OP=_OPS[op], ACC64=dtype == torch.float64, BLOCK=block, num_warps=4,
        )
    return gx


# ===========================================================================
# Reductions: autograd Function and entry point
# ===========================================================================

@dataclass(frozen=True)
class _Opts:
    op: str
    use_triton: bool
    stencil: object = field(compare=False)


class StencilReduceFunction(torch.autograd.Function):
    """``x [B, n_in] -> y [B, N]`` for sum, mean, min, max and median."""

    @staticmethod
    def forward(ctx, x, opts):
        sd = opts.stencil.on(x.device)
        need_sel = ctx.needs_input_grad[0]
        if opts.use_triton:
            y, sel = _triton_reduce(x, sd, opts.op, need_sel)
        else:
            y, sel = _torch_reduce(x, sd, opts.op, need_sel)
        ctx.opts = opts
        ctx.n_in = x.shape[1]
        ctx.x_dtype = x.dtype
        if sel is not None:
            ctx.save_for_backward(sel)
        ctx.has_sel = sel is not None
        return y

    @staticmethod
    def backward(ctx, gy):
        opts = ctx.opts
        sel = ctx.saved_tensors[0] if ctx.has_sel else None
        sd = opts.stencil.on(gy.device)
        if opts.use_triton and not torch.is_grad_enabled():
            gx = _triton_reduce_backward(gy, opts.stencil, sd, opts.op, sel,
                                         ctx.n_in, ctx.x_dtype)
        else:
            # Grad mode is on in a backward pass only with create_graph=True:
            # the torch backward is recorded by autograd (double backward).
            gx = _torch_reduce_backward(gy, sd, opts.op, sel, ctx.n_in, ctx.x_dtype)
        return gx, None


def stencil_reduce(x, stencil, op, *, backend="auto"):
    """Reduce ``x [B, n_in]`` (floating) over the neighbourhoods of ``stencil``.

    ``op`` is one of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``,
    ``"median"``.  Returns ``[B, N]`` in the dtype of ``x``.  Rows with no
    neighbour must have been rejected by the caller for every op but sum.
    """
    if op not in FAST_REDUCTIONS:
        raise ValueError(f"op must be one of {FAST_REDUCTIONS}, got {op!r}.")
    kp_limit = _MAX_KP_MEDIAN_TRITON if op == "median" else _MAX_KP_TRITON
    use_triton = _use_triton(x, backend, _TRITON_DTYPES,
                             kp_ok=0 < stencil.kp <= kp_limit)
    if use_triton:
        x = x.contiguous()
    opts = _Opts(op, use_triton, stencil)
    return StencilReduceFunction.apply(x, opts)


# ===========================================================================
# Least-squares gradient
# ===========================================================================
#
# For output cell n with centre position c(n) and neighbours j = idx[n, k]
# (tangent offsets e_k, n_k), the valid taps are those with a valid geometry
# and finite x_j and x_c.  With s_ee = sum e^2, s_en = sum e n, s_nn = sum n^2,
# r_e = sum e (x_j - x_c), r_n = sum n (x_j - x_c) and det = s_ee s_nn - s_en^2:
#
#     g_east  = (r_e s_nn - r_n s_en) / det
#     g_north = (r_n s_ee - r_e s_en) / det
#
# (NaN where fewer than two valid taps, a non-finite centre or a rank-deficient
# fit).  The output is linear in x for a fixed validity pattern, so with the
# upstream gradients (u, w) and P = (u s_nn - w s_en) / det,
# Q = (w s_ee - u s_en) / det (zero for unsolvable cells):
#
#     d/dx_j += e_k P + n_k Q   for each valid tap (n, k) reading j
#     d/dx_c -= sum_k valid (e_k P + n_k Q)


@dataclass
class _GradOnDevice:
    idx: torch.Tensor        # int32 [N, K]; invalid taps -> n_in (a NaN column)
    ctr: torch.Tensor        # int32 [N]
    east: dict               # dtype -> [N, K] (0 on invalid taps)
    north: dict
    tmap: dict               # dtype -> (ptr, tn, te, tno)


class GradientStencil:
    """Geometry of the least-squares gradient, cached per device and dtype.

    Parameters
    ----------
    neighbour_positions : int array [N, K]
        Input positions of the immediate neighbours, ``-1`` when absent.
    valid_mask : bool array [N, K]
        Validity of the geometry.
    east, north : float64 arrays [N, K]
        Tangent offsets in metres.
    center_positions : int array [N]
    n_in : int
    """

    def __init__(self, neighbour_positions, valid_mask, east, north,
                 center_positions, n_in):
        pos = np.asarray(neighbour_positions, dtype=np.int64)
        valid = np.asarray(valid_mask, dtype=bool) & (pos >= 0)
        self.n_in = int(n_in)
        if self.n_in >= 2**31 - 1:
            raise ValueError("too many input cells for int32 positions.")
        self.n_out, self.k = pos.shape
        self.idx = np.where(valid, pos, self.n_in).astype(np.int32)
        self.valid = valid
        self.east = np.where(valid, east, 0.0)
        self.north = np.where(valid, north, 0.0)
        self.ctr = np.asarray(center_positions, dtype=np.int32)
        self._dev: dict = {}

    def on(self, device, dtype) -> _GradOnDevice:
        device = torch.device(device)
        key = str(device)
        if key not in self._dev:
            self._dev[key] = _GradOnDevice(
                idx=torch.as_tensor(self.idx, device=device),
                ctr=torch.as_tensor(self.ctr, device=device),
                east={}, north={}, tmap={},
            )
        gd = self._dev[key]
        if dtype not in gd.east:
            gd.east[dtype] = torch.as_tensor(self.east, dtype=dtype, device=device)
            gd.north[dtype] = torch.as_tensor(self.north, dtype=dtype, device=device)
        return gd

    def transposed(self, device, dtype):
        gd = self.on(device, dtype)
        if dtype not in gd.tmap:
            valid = torch.as_tensor(self.valid, device=gd.idx.device)
            src = gd.idx[valid].long()
            n_of = torch.arange(self.n_out, device=src.device, dtype=torch.int32)
            n_of = n_of[:, None].expand(-1, self.k)[valid]
            ptr, order = _transposed(src, n_of, self.n_in)
            gd.tmap[dtype] = (ptr, n_of[order].contiguous(),
                              gd.east[dtype][valid][order].contiguous(),
                              gd.north[dtype][valid][order].contiguous())
        return gd.tmap[dtype]


def _lsq_sums(x, gd, dtype):
    """Running normal-equation sums over the K taps (no ``[B, N, K]`` tensor)."""
    xe = _extend(x, float("nan"))
    xc = x.index_select(1, gd.ctr.long())
    fc = torch.isfinite(xc)
    xc0 = torch.where(fc, xc, torch.zeros_like(xc))
    E, Nn = gd.east[dtype], gd.north[dtype]
    zero = torch.zeros((), dtype=dtype, device=x.device)
    s_ee = s_en = s_nn = r_e = r_n = sum_e = sum_n = 0
    n_valid = 0
    for k in range(gd.idx.shape[1]):
        v = xe.index_select(1, gd.idx[:, k].long())
        valid = torch.isfinite(v) & fc
        e = torch.where(valid, E[:, k], zero)
        n = torch.where(valid, Nn[:, k], zero)
        d = torch.where(valid, v - xc0, zero)
        s_ee = s_ee + e * e
        s_en = s_en + e * n
        s_nn = s_nn + n * n
        r_e = r_e + e * d
        r_n = r_n + n * d
        sum_e = sum_e + e
        sum_n = sum_n + n
        n_valid = n_valid + valid.to(torch.int32)
    det = s_ee * s_nn - s_en * s_en
    eps = torch.finfo(dtype).eps
    rank_scale = torch.maximum(s_ee * s_nn, torch.ones_like(det))
    solvable = fc & (n_valid >= 2) & (det > 100.0 * eps * rank_scale)
    safe_det = torch.where(solvable, det, torch.ones_like(det))
    return dict(s_ee=s_ee, s_en=s_en, s_nn=s_nn, r_e=r_e, r_n=r_n,
                sum_e=sum_e, sum_n=sum_n, solvable=solvable, det=safe_det)


def _torch_lsq_forward(x, gd, dtype):
    if gd.idx.shape[1] == 0:
        nan = torch.full((x.shape[0], gd.idx.shape[0]), float("nan"),
                         dtype=dtype, device=x.device)
        return nan, nan.clone()
    s = _lsq_sums(x, gd, dtype)
    ge = (s["r_e"] * s["s_nn"] - s["r_n"] * s["s_en"]) / s["det"]
    gn = (s["r_n"] * s["s_ee"] - s["r_e"] * s["s_en"]) / s["det"]
    nan = torch.full_like(ge, float("nan"))
    return torch.where(s["solvable"], ge, nan), torch.where(s["solvable"], gn, nan)


def _torch_lsq_backward(x, gd, dtype, gu, gw, n_in):
    B = x.shape[0]
    gx = torch.zeros(B, n_in + 1, dtype=dtype, device=x.device)
    if gd.idx.shape[1] == 0:
        return gx[:, :n_in]
    with torch.no_grad():
        s = _lsq_sums(x, gd, dtype)
    zero = torch.zeros((), dtype=dtype, device=x.device)
    gu = torch.where(s["solvable"], gu, zero)
    gw = torch.where(s["solvable"], gw, zero)
    P = torch.where(s["solvable"], (gu * s["s_nn"] - gw * s["s_en"]) / s["det"], zero)
    Q = torch.where(s["solvable"], (gw * s["s_ee"] - gu * s["s_en"]) / s["det"], zero)
    gx = gx.index_add(1, gd.ctr.long(), -(s["sum_e"] * P + s["sum_n"] * Q))
    xe = _extend(x, float("nan"))
    xc = x.index_select(1, gd.ctr.long())
    fc = torch.isfinite(xc)
    E, Nn = gd.east[dtype], gd.north[dtype]
    for k in range(gd.idx.shape[1]):
        col = gd.idx[:, k].long()
        valid = torch.isfinite(xe.index_select(1, col)) & fc
        contrib = torch.where(valid, E[:, k] * P + Nn[:, k] * Q, zero)
        gx = gx.index_add(1, col, contrib)
    return gx[:, :n_in]


if _HAS_TRITON:

    @triton.jit
    def _lsq_sums_tile(x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, b, n, n_ok, N_IN, K,
                       BLOCK_K: tl.constexpr, ACC64: tl.constexpr):
        k = tl.arange(0, BLOCK_K)
        m = n_ok[:, None] & (k[None, :] < K)
        off = n.to(tl.int64)[:, None] * K + k[None, :]
        src = tl.load(idx_ptr + off, mask=m, other=0)
        geo = m & (src < N_IN)
        xb = x_ptr + b * N_IN
        v = tl.load(xb + src.to(tl.int64), mask=geo, other=0.0)
        c = tl.load(ctr_ptr + n, mask=n_ok, other=0).to(tl.int64)
        xc = tl.load(xb + c, mask=n_ok, other=0.0)
        if ACC64:
            v = v.to(tl.float64)
            xc = xc.to(tl.float64)
        else:
            v = v.to(tl.float32)
            xc = xc.to(tl.float32)
        fc = n_ok & (tl.abs(xc) < float("inf"))
        valid = geo & (tl.abs(v) < float("inf")) & fc[:, None]
        E = tl.load(e_ptr + off, mask=m, other=0.0).to(v.dtype)
        Nn = tl.load(no_ptr + off, mask=m, other=0.0).to(v.dtype)
        e = tl.where(valid, E, 0.0)
        nn = tl.where(valid, Nn, 0.0)
        d = tl.where(valid, v - tl.where(fc, xc, 0.0)[:, None], 0.0)
        s_ee = tl.sum(e * e, axis=1)
        s_en = tl.sum(e * nn, axis=1)
        s_nn = tl.sum(nn * nn, axis=1)
        r_e = tl.sum(e * d, axis=1)
        r_n = tl.sum(nn * d, axis=1)
        sum_e = tl.sum(e, axis=1)
        sum_n = tl.sum(nn, axis=1)
        n_valid = tl.sum(valid.to(tl.int32), axis=1)
        return s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, n_valid, fc, c

    @triton.jit
    def _lsq_fwd_kernel(x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, ge_ptr, gn_ptr,
                        N_IN, N_OUT, K, NBLK, EPS100,
                        ACC64: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0)
        b = (pid // NBLK).to(tl.int64)
        n = (pid % NBLK) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < N_OUT
        s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, n_valid, fc, c = _lsq_sums_tile(
            x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, b, n, n_ok, N_IN, K, BLOCK_K, ACC64)
        det = s_ee * s_nn - s_en * s_en
        scale = tl.maximum(s_ee * s_nn, 1.0)
        solv = fc & (n_valid >= 2) & (det > EPS100 * scale)
        sd = tl.where(solv, det, 1.0)
        ge = (r_e * s_nn - r_n * s_en) / sd
        gn = (r_n * s_ee - r_e * s_en) / sd
        nan = float("nan")
        ge = tl.where(solv, ge, nan)
        gn = tl.where(solv, gn, nan)
        tl.store(ge_ptr + b * N_OUT + n, ge.to(ge_ptr.dtype.element_ty), mask=n_ok)
        tl.store(gn_ptr + b * N_OUT + n, gn.to(gn_ptr.dtype.element_ty), mask=n_ok)

    @triton.jit
    def _lsq_bwd1_kernel(x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, gu_ptr, gw_ptr,
                         p_ptr, q_ptr, gx_ptr,
                         N_IN, N_OUT, K, NBLK, EPS100,
                         ACC64: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """P, Q per output cell, and the centre term written at gx[b, c(n)]."""
        pid = tl.program_id(0)
        b = (pid // NBLK).to(tl.int64)
        n = (pid % NBLK) * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = n < N_OUT
        s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, n_valid, fc, c = _lsq_sums_tile(
            x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, b, n, n_ok, N_IN, K, BLOCK_K, ACC64)
        det = s_ee * s_nn - s_en * s_en
        scale = tl.maximum(s_ee * s_nn, 1.0)
        solv = fc & (n_valid >= 2) & (det > EPS100 * scale)
        sd = tl.where(solv, det, 1.0)
        gu = tl.load(gu_ptr + b * N_OUT + n, mask=n_ok, other=0.0).to(s_ee.dtype)
        gw = tl.load(gw_ptr + b * N_OUT + n, mask=n_ok, other=0.0).to(s_ee.dtype)
        gu = tl.where(solv, gu, 0.0)
        gw = tl.where(solv, gw, 0.0)
        P = tl.where(solv, (gu * s_nn - gw * s_en) / sd, 0.0)
        Q = tl.where(solv, (gw * s_ee - gu * s_en) / sd, 0.0)
        tl.store(p_ptr + b * N_OUT + n, P, mask=n_ok)
        tl.store(q_ptr + b * N_OUT + n, Q, mask=n_ok)
        ctr_term = -(sum_e * P + sum_n * Q)
        tl.store(gx_ptr + b * N_IN + c, ctr_term.to(gx_ptr.dtype.element_ty), mask=n_ok)

    @triton.jit
    def _lsq_bwd2_kernel(x_ptr, p_ptr, q_ptr, ptr_ptr, tn_ptr, te_ptr, tno_ptr, gx_ptr,
                         N_IN, N_OUT, NBLK,
                         ACC64: tl.constexpr, BLOCK: tl.constexpr):
        """gx[b, j] += finite(x_j) * sum over entries (n, k) reading j of e P + n Q."""
        pid = tl.program_id(0)
        b = (pid // NBLK).to(tl.int64)
        j = (pid % NBLK) * BLOCK + tl.arange(0, BLOCK)
        j_ok = j < N_IN
        xj = tl.load(x_ptr + b * N_IN + j, mask=j_ok, other=0.0)
        acc = tl.load(gx_ptr + b * N_IN + j, mask=j_ok, other=0.0)
        if ACC64:
            acc = acc.to(tl.float64)
        else:
            acc = acc.to(tl.float32)
        start = tl.load(ptr_ptr + j, mask=j_ok, other=0)
        n_e = tl.load(ptr_ptr + j + 1, mask=j_ok, other=0) - start
        n_max = tl.max(n_e, axis=0)
        part = tl.zeros((BLOCK,), dtype=acc.dtype)
        for e in range(0, n_max):
            mm = j_ok & (e < n_e)
            ent = start + e
            nn = tl.load(tn_ptr + ent, mask=mm, other=0).to(tl.int64)
            te = tl.load(te_ptr + ent, mask=mm, other=0.0).to(acc.dtype)
            tno = tl.load(tno_ptr + ent, mask=mm, other=0.0).to(acc.dtype)
            P = tl.load(p_ptr + b * N_OUT + nn, mask=mm, other=0.0)
            Q = tl.load(q_ptr + b * N_OUT + nn, mask=mm, other=0.0)
            part += te * P + tno * Q
        fin = tl.abs(xj.to(acc.dtype)) < float("inf")
        acc += tl.where(fin, part, 0.0)
        tl.store(gx_ptr + b * N_IN + j, acc.to(gx_ptr.dtype.element_ty), mask=j_ok)


def _grad_blocks(k):
    bk = max(2, _next_pow2(k))
    return max(16, min(256, 2048 // bk)), bk


def _triton_lsq_forward(x, gd, dtype):
    B, n_in = x.shape
    N, K = gd.idx.shape
    ge = torch.empty(B, N, dtype=dtype, device=x.device)
    gn = torch.empty(B, N, dtype=dtype, device=x.device)
    if B * N == 0:
        return ge, gn
    bn, bk = _grad_blocks(K)
    nblk = triton.cdiv(N, bn)
    with _device_ctx(x.device):
        _lsq_fwd_kernel[(B * nblk,)](
            x, gd.idx, gd.east[dtype], gd.north[dtype], gd.ctr, ge, gn,
            n_in, N, K, nblk, 100.0 * torch.finfo(dtype).eps,
            ACC64=dtype == torch.float64, BLOCK_N=bn, BLOCK_K=bk, num_warps=4,
        )
    return ge, gn


def _triton_lsq_backward(x, gstencil, gd, dtype, gu, gw):
    B, n_in = x.shape
    N, K = gd.idx.shape
    gx = torch.zeros(B, n_in, dtype=dtype, device=x.device)
    if B * N == 0:
        return gx
    P = torch.empty(B, N, dtype=dtype, device=x.device)
    Q = torch.empty(B, N, dtype=dtype, device=x.device)
    bn, bk = _grad_blocks(K)
    nblk = triton.cdiv(N, bn)
    eps100 = 100.0 * torch.finfo(dtype).eps
    acc64 = dtype == torch.float64
    ptr, tn, te, tno = gstencil.transposed(x.device, dtype)
    with _device_ctx(x.device):
        _lsq_bwd1_kernel[(B * nblk,)](
            x, gd.idx, gd.east[dtype], gd.north[dtype], gd.ctr,
            gu.to(dtype).contiguous(), gw.to(dtype).contiguous(), P, Q, gx,
            n_in, N, K, nblk, eps100,
            ACC64=acc64, BLOCK_N=bn, BLOCK_K=bk, num_warps=4,
        )
        block = 128
        nb2 = triton.cdiv(n_in, block)
        _lsq_bwd2_kernel[(B * nb2,)](
            x, P, Q, ptr, tn, te, tno, gx, n_in, N, nb2,
            ACC64=acc64, BLOCK=block, num_warps=4,
        )
    return gx


@dataclass(frozen=True)
class _GradOpts:
    use_triton: bool
    stencil: object = field(compare=False)


class LsqGradientFunction(torch.autograd.Function):
    """``x [B, n_in] -> (g_east [B, N], g_north [B, N])``."""

    @staticmethod
    def forward(ctx, x, opts):
        dtype = x.dtype
        gd = opts.stencil.on(x.device, dtype)
        if opts.use_triton:
            ge, gn = _triton_lsq_forward(x, gd, dtype)
        else:
            ge, gn = _torch_lsq_forward(x, gd, dtype)
        ctx.opts = opts
        ctx.save_for_backward(x)
        return ge, gn

    @staticmethod
    def backward(ctx, gu, gw):
        (x,) = ctx.saved_tensors
        opts = ctx.opts
        dtype = x.dtype
        gd = opts.stencil.on(x.device, dtype)
        if gu is None:
            gu = torch.zeros(x.shape[0], gd.idx.shape[0], dtype=dtype, device=x.device)
        if gw is None:
            gw = torch.zeros_like(gu)
        if opts.use_triton and not torch.is_grad_enabled():
            gx = _triton_lsq_backward(x, opts.stencil, gd, dtype, gu, gw)
        else:
            gx = _torch_lsq_backward(x.detach(), gd, dtype, gu.to(dtype), gw.to(dtype),
                                     x.shape[1])
        return gx, None


def lsq_gradient(x, stencil, *, backend="auto"):
    """Least-squares tangent gradient of ``x [B, n_in]`` (float32 / float64)."""
    use_triton = _use_triton(x, backend, _GRAD_TRITON_DTYPES, kp_ok=stencil.k <= 64)
    if use_triton:
        x = x.contiguous()
    return LsqGradientFunction.apply(x, _GradOpts(use_triton, stencil))
