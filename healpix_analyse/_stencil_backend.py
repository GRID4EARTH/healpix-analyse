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
import math
import os
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
    "divcurl_embed",
    "divcurl_contract",
]

BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_GRAD_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
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
#
# Every kernel handles a block of cells for BB batch rows: the neighbour
# indices (and, for the gradient, the geometry) are loaded once per program
# and reused for the BB rows, so the stencil is read ~B/BB times instead of B
# times.  Tiles (BLOCK_N / BLOCK, BB, warps) are autotuned per problem shape.


def _size_class(n):
    """Coarse size bucket, part of the autotune keys."""
    return int(math.log2(max(int(n), 1))) // 2


if _HAS_TRITON:

    def _cfgs(spec):
        configs = [triton.Config(kw, num_warps=w) for kw, w in spec]
        if os.environ.get("TRITON_INTERPRET") == "1":
            # one config in the interpreter, with several batch rows per program
            configs = [c for c in configs if c.kwargs["BB"] == 4][:1] or configs[:1]
        return configs

    _RED_FWD_CFGS = _cfgs([({"BLOCK_N": bn, "BB": bb}, w)
                           for bn in (16, 32, 64, 128, 256)
                           for bb in (1, 2, 4, 8) for w in (4, 8)])
    _BWD_CFGS = _cfgs([({"BLOCK": bl, "BB": bb}, w)
                       for bl in (64, 128, 256) for bb in (2, 4, 8) for w in (4, 8)])
    _LSQ_CFGS = _cfgs([({"BLOCK_N": bn, "BB": bb}, w)
                       for bn in (32, 64, 128, 256) for bb in (1, 2, 4, 8) for w in (4, 8)])

    def _meta(named_args, kwargs):
        meta = dict(named_args)
        meta.update(kwargs)
        return meta

    def _prune_bb(configs, B):
        bb_max = max(2, _next_pow2(B))
        kept = [c for c in configs if c.kwargs["BB"] <= bb_max]
        return kept or configs[:1]

    def _prune_red_fwd(configs, named_args, **kwargs):
        meta = _meta(named_args, kwargs)
        bk = meta["BLOCK_K"]
        if meta["OP"] == 4:   # [BLOCK_N, BK, BK] comparison tile
            kept = [c for c in configs if c.kwargs["BLOCK_N"] * bk * bk <= 16384
                    and c.kwargs["BLOCK_N"] <= 64]
        else:
            kept = [c for c in configs if c.kwargs["BLOCK_N"] * bk <= 8192]
        if not kept:
            kept = [min(configs, key=lambda c: c.kwargs["BLOCK_N"])]
        return _prune_bb(kept, meta["B"])

    def _prune_bwd(configs, named_args, **kwargs):
        return _prune_bb(configs, _meta(named_args, kwargs)["B"])

    def _prune_lsq(configs, named_args, **kwargs):
        meta = _meta(named_args, kwargs)
        bk = meta["BLOCK_K"]
        kept = [c for c in configs if c.kwargs["BLOCK_N"] * bk <= 4096] or configs[:1]
        return _prune_bb(kept, meta["B"])

    @triton.autotune(configs=_RED_FWD_CFGS, key=["KP", "SIZE", "OP", "NEED_SEL", "ACC64"],
                     prune_configs_by={"early_config_prune": _prune_red_fwd})
    @triton.jit
    def _reduce_fwd_kernel(x_ptr, idx_ptr, cnt_ptr, y_ptr, sel_ptr,
                           N_IN, N_OUT, KP, B, SIZE,
                           OP: tl.constexpr, NEED_SEL: tl.constexpr, ACC64: tl.constexpr,
                           BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BB: tl.constexpr):
        """y[b, n] = reduce_k x[b, idx[n, k]] over k < cnt[n], for BB rows b."""
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        b0 = tl.program_id(1) * BB
        n_ok = n < N_OUT
        cnt = tl.load(cnt_ptr + n, mask=n_ok, other=0)
        k = tl.arange(0, BLOCK_K)
        kk = k[None, :]
        m0 = n_ok[:, None] & (kk < cnt[:, None])
        src = tl.load(idx_ptr + n.to(tl.int64)[:, None] * KP + kk, mask=m0, other=0).to(tl.int64)
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            row_ok = b < B
            m = m0 & row_ok
            out_ok = n_ok & row_ok
            v = tl.load(x_ptr + b * N_IN + src, mask=m, other=0.0)
            y_off = b * N_OUT + n
            if OP <= 1:
                if ACC64:
                    vf = v.to(tl.float64)
                else:
                    vf = v.to(tl.float32)
                s = tl.sum(tl.where(m, vf, 0.0), axis=1)
                if OP == 1:
                    s = s / cnt.to(s.dtype)
                tl.store(y_ptr + y_off, s.to(y_ptr.dtype.element_ty), mask=out_ok)
            else:
                isn = m & (v != v)
                has_nan = tl.max(isn.to(tl.int32), axis=1) > 0
                nan_v = tl.sum(tl.where(isn, v, 0.0), axis=1)      # NaN iff has_nan
                if OP == 4:
                    # rank of each value among the valid ones of its row, ties
                    # broken by tap: a strict total order, ranks are 0..cnt-1
                    vi = v[:, :, None]
                    vj = v[:, None, :]
                    less = (vj < vi) | ((vj == vi) & (kk[:, None, :] < kk[:, :, None]))
                    less = less & m[:, None, :]
                    rank = tl.sum(less.to(tl.int32), axis=2)
                    hit_lo = m & (rank == ((cnt - 1) // 2)[:, None])
                    hit_hi = m & (rank == (cnt // 2)[:, None])
                    lo = tl.max(tl.where(hit_lo, v, float("-inf")), axis=1)
                    hi = tl.max(tl.where(hit_hi, v, float("-inf")), axis=1)
                    odd = (cnt % 2) == 1
                    val = tl.where(odd, hi, (lo + hi) / 2)
                    out = tl.where(has_nan, nan_v, val)
                    tl.store(y_ptr + y_off, out.to(y_ptr.dtype.element_ty), mask=out_ok)
                    if NEED_SEL:
                        s_lo = tl.min(tl.where(hit_lo, kk, BLOCK_K), axis=1)
                        s_hi = tl.min(tl.where(hit_hi, kk, BLOCK_K), axis=1)
                        s_lo = tl.where(odd | has_nan, -1, s_lo)
                        s_hi = tl.where(has_nan, -1, s_hi)
                        tl.store(sel_ptr + y_off, s_lo.to(tl.int16), mask=out_ok)
                        tl.store(sel_ptr + B * N_OUT + y_off, s_hi.to(tl.int16), mask=out_ok)
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
                    tl.store(y_ptr + y_off, out.to(y_ptr.dtype.element_ty), mask=out_ok)
                    if NEED_SEL:
                        tl.store(sel_ptr + y_off, sel.to(tl.int16), mask=out_ok)

    @triton.autotune(configs=_BWD_CFGS, key=["SIZE", "OP", "ACC64"],
                     prune_configs_by={"early_config_prune": _prune_bwd})
    @triton.jit
    def _reduce_bwd_kernel(gy_ptr, ptr_ptr, tn_ptr, tk_ptr, cnt_ptr, sel_ptr, gx_ptr,
                           N_IN, N_OUT, B, SIZE,
                           OP: tl.constexpr, ACC64: tl.constexpr,
                           BLOCK: tl.constexpr, BB: tl.constexpr):
        """gx[b, j] = sum over the entries (n, k) that read j of gy[b, n] * w(n, k).

        A gather on the transposed stencil: no atomics, deterministic.  The
        entries of a block of input cells are read once for BB batch rows.
        """
        j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        bv = tl.program_id(1) * BB + tl.arange(0, BB)
        j_ok = j < N_IN
        b_ok = bv < B
        brow = bv.to(tl.int64)[None, :] * N_OUT
        start = tl.load(ptr_ptr + j, mask=j_ok, other=0)
        n_e = tl.load(ptr_ptr + j + 1, mask=j_ok, other=0) - start
        n_max = tl.max(n_e, axis=0)
        if ACC64:
            acc = tl.zeros((BLOCK, BB), dtype=tl.float64)
        else:
            acc = tl.zeros((BLOCK, BB), dtype=tl.float32)
        for e in range(0, n_max):
            mm = j_ok & (e < n_e)
            m2 = mm[:, None] & b_ok[None, :]
            ent = start + e
            nn = tl.load(tn_ptr + ent, mask=mm, other=0).to(tl.int64)
            goff = brow + nn[:, None]
            g = tl.load(gy_ptr + goff, mask=m2, other=0.0).to(acc.dtype)
            if OP == 0:
                acc += g
            elif OP == 1:
                c = tl.load(cnt_ptr + nn, mask=mm, other=1).to(acc.dtype)
                acc += g / c[:, None]
            elif OP == 4:
                kk = tl.load(tk_ptr + ent, mask=mm, other=-2).to(tl.int32)[:, None]
                s_lo = tl.load(sel_ptr + goff, mask=m2, other=-1).to(tl.int32)
                s_hi = tl.load(sel_ptr + B * N_OUT + goff, mask=m2, other=-1).to(tl.int32)
                half = g / 2
                acc += tl.where(s_lo >= 0,
                                tl.where(s_lo == kk, half, 0.0) + tl.where(s_hi == kk, half, 0.0),
                                tl.where(s_hi == kk, g, 0.0))
            else:
                kk = tl.load(tk_ptr + ent, mask=mm, other=-2).to(tl.int32)[:, None]
                s = tl.load(sel_ptr + goff, mask=m2, other=-1).to(tl.int32)
                acc += tl.where(s == kk, g, 0.0)
        tl.store(gx_ptr + bv.to(tl.int64)[None, :] * N_IN + j[:, None],
                 acc.to(gx_ptr.dtype.element_ty), mask=j_ok[:, None] & b_ok[None, :])


def _triton_reduce(x, sd, op, need_sel):
    B, n_in = x.shape
    N, KP = sd.idx.shape
    y = torch.empty(B, N, dtype=x.dtype, device=x.device)
    sel = None
    if need_sel and op in ("min", "max"):
        sel = torch.empty(B, N, dtype=torch.int16, device=x.device)
    elif need_sel and op == "median":
        sel = torch.empty(2, B, N, dtype=torch.int16, device=x.device)
    if N == 0 or B == 0:
        return y, sel
    bk = max(2, _next_pow2(KP))

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(B, meta["BB"]))

    with _device_ctx(x.device):
        _reduce_fwd_kernel[grid](
            x, sd.idx, sd.cnt, y, sel if sel is not None else y,
            n_in, N, KP, B, _size_class(B * N),
            OP=_OPS[op], NEED_SEL=sel is not None,
            ACC64=x.dtype == torch.float64, BLOCK_K=bk,
        )
    return y, sel


def _triton_reduce_backward(gy, stencil, sd, op, sel, n_in, dtype):
    B, N = gy.shape
    ptr, tn, tk = stencil.transposed(gy.device)
    gy = gy.to(dtype).contiguous()
    gx = torch.empty(B, n_in, dtype=dtype, device=gy.device)
    if B == 0 or n_in == 0:
        return gx

    def grid(meta):
        return (triton.cdiv(n_in, meta["BLOCK"]), triton.cdiv(B, meta["BB"]))

    with _device_ctx(gy.device):
        _reduce_bwd_kernel[grid](
            gy, ptr, tn, tk, sd.cnt, sel if sel is not None else gy, gx,
            n_in, N, B, _size_class(B * n_in),
            OP=_OPS[op], ACC64=dtype == torch.float64,
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
    def _lsq_geometry(idx_ptr, e_ptr, no_ptr, ctr_ptr, n, n_ok, N_IN, K,
                      BLOCK_K: tl.constexpr, ACC64: tl.constexpr):
        """Stencil of a block of cells, loaded once for all the batch rows."""
        k = tl.arange(0, BLOCK_K)
        m = n_ok[:, None] & (k[None, :] < K)
        off = n.to(tl.int64)[:, None] * K + k[None, :]
        src = tl.load(idx_ptr + off, mask=m, other=0)
        geo = m & (src < N_IN)
        c = tl.load(ctr_ptr + n, mask=n_ok, other=0).to(tl.int64)
        if ACC64:
            E = tl.load(e_ptr + off, mask=m, other=0.0).to(tl.float64)
            Nn = tl.load(no_ptr + off, mask=m, other=0.0).to(tl.float64)
        else:
            E = tl.load(e_ptr + off, mask=m, other=0.0).to(tl.float32)
            Nn = tl.load(no_ptr + off, mask=m, other=0.0).to(tl.float32)
        return src.to(tl.int64), geo, c, E, Nn

    @triton.jit
    def _lsq_row(x_ptr, b, row_ok, src, geo, c, n_ok, E, Nn, N_IN, EPS100,
                 ACC64: tl.constexpr):
        """Normal-equation sums of one batch row (registers only)."""
        xb = x_ptr + b * N_IN
        v = tl.load(xb + src, mask=geo & row_ok, other=0.0)
        xc = tl.load(xb + c, mask=n_ok & row_ok, other=0.0)
        if ACC64:
            v = v.to(tl.float64)
            xc = xc.to(tl.float64)
        else:
            v = v.to(tl.float32)
            xc = xc.to(tl.float32)
        fc = n_ok & row_ok & (tl.abs(xc) < float("inf"))
        valid = geo & (tl.abs(v) < float("inf")) & fc[:, None]
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
        det = s_ee * s_nn - s_en * s_en
        scale = tl.maximum(s_ee * s_nn, 1.0)
        solv = fc & (n_valid >= 2) & (det > EPS100 * scale)
        sd = tl.where(solv, det, 1.0)
        return s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, solv, sd

    @triton.autotune(configs=_LSQ_CFGS, key=["K", "SIZE", "ACC64"],
                     prune_configs_by={"early_config_prune": _prune_lsq})
    @triton.jit
    def _lsq_fwd_kernel(x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, ge_ptr, gn_ptr,
                        N_IN, N_OUT, K, B, SIZE, EPS100,
                        ACC64: tl.constexpr, BLOCK_K: tl.constexpr,
                        BLOCK_N: tl.constexpr, BB: tl.constexpr):
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        b0 = tl.program_id(1) * BB
        n_ok = n < N_OUT
        src, geo, c, E, Nn = _lsq_geometry(idx_ptr, e_ptr, no_ptr, ctr_ptr, n, n_ok,
                                           N_IN, K, BLOCK_K, ACC64)
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            row_ok = b < B
            s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, solv, sd = _lsq_row(
                x_ptr, b, row_ok, src, geo, c, n_ok, E, Nn, N_IN, EPS100, ACC64)
            ge = tl.where(solv, (r_e * s_nn - r_n * s_en) / sd, float("nan"))
            gn = tl.where(solv, (r_n * s_ee - r_e * s_en) / sd, float("nan"))
            out_ok = n_ok & row_ok
            tl.store(ge_ptr + b * N_OUT + n, ge.to(ge_ptr.dtype.element_ty), mask=out_ok)
            tl.store(gn_ptr + b * N_OUT + n, gn.to(gn_ptr.dtype.element_ty), mask=out_ok)

    @triton.autotune(configs=_LSQ_CFGS, key=["K", "SIZE", "ACC64"],
                     prune_configs_by={"early_config_prune": _prune_lsq})
    @triton.jit
    def _lsq_bwd1_kernel(x_ptr, idx_ptr, e_ptr, no_ptr, ctr_ptr, gu_ptr, gw_ptr,
                         p_ptr, q_ptr, gx_ptr,
                         N_IN, N_OUT, K, B, SIZE, EPS100,
                         ACC64: tl.constexpr, BLOCK_K: tl.constexpr,
                         BLOCK_N: tl.constexpr, BB: tl.constexpr):
        """P, Q per output cell, and the centre term written at gx[b, c(n)]."""
        n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        b0 = tl.program_id(1) * BB
        n_ok = n < N_OUT
        src, geo, c, E, Nn = _lsq_geometry(idx_ptr, e_ptr, no_ptr, ctr_ptr, n, n_ok,
                                           N_IN, K, BLOCK_K, ACC64)
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            row_ok = b < B
            out_ok = n_ok & row_ok
            s_ee, s_en, s_nn, r_e, r_n, sum_e, sum_n, solv, sd = _lsq_row(
                x_ptr, b, row_ok, src, geo, c, n_ok, E, Nn, N_IN, EPS100, ACC64)
            gu = tl.load(gu_ptr + b * N_OUT + n, mask=out_ok, other=0.0).to(s_ee.dtype)
            gw = tl.load(gw_ptr + b * N_OUT + n, mask=out_ok, other=0.0).to(s_ee.dtype)
            gu = tl.where(solv, gu, 0.0)
            gw = tl.where(solv, gw, 0.0)
            P = tl.where(solv, (gu * s_nn - gw * s_en) / sd, 0.0)
            Q = tl.where(solv, (gw * s_ee - gu * s_en) / sd, 0.0)
            tl.store(p_ptr + b * N_OUT + n, P, mask=out_ok)
            tl.store(q_ptr + b * N_OUT + n, Q, mask=out_ok)
            ctr_term = -(sum_e * P + sum_n * Q)
            tl.store(gx_ptr + b * N_IN + c, ctr_term.to(gx_ptr.dtype.element_ty), mask=out_ok)

    @triton.autotune(configs=_BWD_CFGS, key=["SIZE", "ACC64"],
                     prune_configs_by={"early_config_prune": _prune_bwd},
                     restore_value=["gx_ptr"])
    @triton.jit
    def _lsq_bwd2_kernel(x_ptr, p_ptr, q_ptr, ptr_ptr, tn_ptr, te_ptr, tno_ptr, gx_ptr,
                         N_IN, N_OUT, B, SIZE,
                         ACC64: tl.constexpr, BLOCK: tl.constexpr, BB: tl.constexpr):
        """gx[b, j] += finite(x_j) * sum over entries (n, k) reading j of e P + n Q."""
        j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        bv = tl.program_id(1) * BB + tl.arange(0, BB)
        j_ok = j < N_IN
        b_ok = bv < B
        mo = j_ok[:, None] & b_ok[None, :]
        xoff = bv.to(tl.int64)[None, :] * N_IN + j[:, None]
        prow = bv.to(tl.int64)[None, :] * N_OUT
        start = tl.load(ptr_ptr + j, mask=j_ok, other=0)
        n_e = tl.load(ptr_ptr + j + 1, mask=j_ok, other=0) - start
        n_max = tl.max(n_e, axis=0)
        if ACC64:
            part = tl.zeros((BLOCK, BB), dtype=tl.float64)
        else:
            part = tl.zeros((BLOCK, BB), dtype=tl.float32)
        for e in range(0, n_max):
            mm = j_ok & (e < n_e)
            m2 = mm[:, None] & b_ok[None, :]
            ent = start + e
            nn = tl.load(tn_ptr + ent, mask=mm, other=0).to(tl.int64)
            te = tl.load(te_ptr + ent, mask=mm, other=0.0).to(part.dtype)
            tno = tl.load(tno_ptr + ent, mask=mm, other=0.0).to(part.dtype)
            P = tl.load(p_ptr + prow + nn[:, None], mask=m2, other=0.0)
            Q = tl.load(q_ptr + prow + nn[:, None], mask=m2, other=0.0)
            part += te[:, None] * P + tno[:, None] * Q
        xj = tl.load(x_ptr + xoff, mask=mo, other=0.0).to(part.dtype)
        acc = tl.load(gx_ptr + xoff, mask=mo, other=0.0).to(part.dtype)
        acc += tl.where(tl.abs(xj) < float("inf"), part, 0.0)
        tl.store(gx_ptr + xoff, acc.to(gx_ptr.dtype.element_ty), mask=mo)


def _geo_dtype(dtype):
    """dtype of the geometry and of the accumulators (fp32 for 16-bit data)."""
    return _acc_dtype(dtype)


def _triton_lsq_forward(x, gd, dtype):
    B, n_in = x.shape
    N, K = gd.idx.shape
    gdt = _geo_dtype(dtype)
    ge = torch.empty(B, N, dtype=dtype, device=x.device)
    gn = torch.empty(B, N, dtype=dtype, device=x.device)
    if B * N == 0:
        return ge, gn

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(B, meta["BB"]))

    with _device_ctx(x.device):
        _lsq_fwd_kernel[grid](
            x, gd.idx, gd.east[gdt], gd.north[gdt], gd.ctr, ge, gn,
            n_in, N, K, B, _size_class(B * N), 100.0 * torch.finfo(gdt).eps,
            ACC64=gdt == torch.float64, BLOCK_K=max(2, _next_pow2(K)),
        )
    return ge, gn


def _triton_lsq_backward(x, gstencil, gd, dtype, gu, gw):
    B, n_in = x.shape
    N, K = gd.idx.shape
    gdt = _geo_dtype(dtype)
    gx = torch.zeros(B, n_in, dtype=gdt, device=x.device)
    if B * N == 0:
        return gx.to(dtype)
    P = torch.empty(B, N, dtype=gdt, device=x.device)
    Q = torch.empty(B, N, dtype=gdt, device=x.device)
    acc64 = gdt == torch.float64
    ptr, tn, te, tno = gstencil.transposed(x.device, gdt)

    def grid1(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(B, meta["BB"]))

    def grid2(meta):
        return (triton.cdiv(n_in, meta["BLOCK"]), triton.cdiv(B, meta["BB"]))

    with _device_ctx(x.device):
        _lsq_bwd1_kernel[grid1](
            x, gd.idx, gd.east[gdt], gd.north[gdt], gd.ctr,
            gu.contiguous(), gw.contiguous(), P, Q, gx,
            n_in, N, K, B, _size_class(B * N), 100.0 * torch.finfo(gdt).eps,
            ACC64=acc64, BLOCK_K=max(2, _next_pow2(K)),
        )
        _lsq_bwd2_kernel[grid2](
            x, P, Q, ptr, tn, te, tno, gx, n_in, N, B, _size_class(B * n_in),
            ACC64=acc64,
        )
    return gx.to(dtype)


@dataclass(frozen=True)
class _GradOpts:
    use_triton: bool
    stencil: object = field(compare=False)


class LsqGradientFunction(torch.autograd.Function):
    """``x [B, n_in] -> (g_east [B, N], g_north [B, N])``.

    16-bit data compute in fp32 (geometry, sums and gradients) and return
    the input dtype.
    """

    @staticmethod
    def forward(ctx, x, opts):
        dtype = x.dtype
        gdt = _geo_dtype(dtype)
        gd = opts.stencil.on(x.device, gdt)
        if opts.use_triton:
            ge, gn = _triton_lsq_forward(x, gd, dtype)
        else:
            ge, gn = _torch_lsq_forward(x.to(gdt), gd, gdt)
            ge, gn = ge.to(dtype), gn.to(dtype)
        ctx.opts = opts
        ctx.save_for_backward(x)
        return ge, gn

    @staticmethod
    def backward(ctx, gu, gw):
        (x,) = ctx.saved_tensors
        opts = ctx.opts
        dtype = x.dtype
        gdt = _geo_dtype(dtype)
        gd = opts.stencil.on(x.device, gdt)
        if gu is None:
            gu = torch.zeros(x.shape[0], gd.idx.shape[0], dtype=dtype, device=x.device)
        if gw is None:
            gw = torch.zeros_like(gu)
        if opts.use_triton and not torch.is_grad_enabled():
            gx = _triton_lsq_backward(x, opts.stencil, gd, dtype, gu, gw)
        else:
            gx = _torch_lsq_backward(x.detach().to(gdt), gd, gdt, gu.to(gdt), gw.to(gdt),
                                     x.shape[1]).to(dtype)
        return gx, None


def lsq_gradient(x, stencil, *, backend="auto"):
    """Least-squares tangent gradient of ``x [B, n_in]``.

    float32 / float64 everywhere; float16 / bfloat16 on the Triton backend
    (fp32 accumulation).
    """
    use_triton = _use_triton(x, backend, _GRAD_TRITON_DTYPES, kp_ok=stencil.k <= 64)
    if use_triton:
        x = x.contiguous()
    return LsqGradientFunction.apply(x, _GradOpts(use_triton, stencil))


# ===========================================================================
# Divergence / curl: the embedding and the contraction around HealPixConv
# ===========================================================================
#
# HealPixDivCurl = embed -> HealPixConv (3 -> 6 channels per gauge) -> contract.
#
#   embed:    V[b, c, n] = u[b, n] east[c, n] + v[b, n] north[c, n]      c < 3
#   contract: div[b, n]  = s sum_g sum_c a[g,c,n] Da[b,g,c,n] + bx[g,c,n] Db[b,g,c,n]
#             curl[b, n] = s sum_g sum_c bx[g,c,n] Da[b,g,c,n] - a[g,c,n] Db[b,g,c,n]
#
# (s = 1/G with the gauge mean, else one output per gauge).  Each is one
# elementwise pass, forward and backward, instead of ~10 PyTorch kernels and
# their intermediate tensors; no atomics.

if _HAS_TRITON:

    _EW_CFGS = _cfgs([({"BLOCK": bl, "BB": bb}, w)
                      for bl in (256, 512, 1024) for bb in (1, 2, 4) for w in (4, 8)])

    def _prune_ew(configs, named_args, **kwargs):
        return _prune_bb([c for c in configs if c.kwargs["BB"] >= 1], _meta(named_args, kwargs)["B"])

    @triton.autotune(configs=_EW_CFGS, key=["SIZE", "BWD"],
                     prune_configs_by={"early_config_prune": _prune_ew})
    @triton.jit
    def _embed_kernel(a_ptr, e_ptr, no_ptr, out_ptr, N, B, SIZE,
                      BWD: tl.constexpr, BLOCK: tl.constexpr, BB: tl.constexpr):
        """Forward: a = uv [B, 2, N] -> out = V [B, 3, N].
        Backward: a = gV [B, 3, N] -> out = g_uv [B, 2, N]."""
        n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        b0 = tl.program_id(1) * BB
        ok = n < N
        e0 = tl.load(e_ptr + n, mask=ok, other=0.0).to(tl.float64 if a_ptr.dtype.element_ty == tl.float64 else tl.float32)
        e1 = tl.load(e_ptr + N + n, mask=ok, other=0.0).to(e0.dtype)
        e2 = tl.load(e_ptr + 2 * N + n, mask=ok, other=0.0).to(e0.dtype)
        n0 = tl.load(no_ptr + n, mask=ok, other=0.0).to(e0.dtype)
        n1 = tl.load(no_ptr + N + n, mask=ok, other=0.0).to(e0.dtype)
        n2 = tl.load(no_ptr + 2 * N + n, mask=ok, other=0.0).to(e0.dtype)
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            m = ok & (b < B)
            if BWD:
                base = a_ptr + b * 3 * N
                g0 = tl.load(base + n, mask=m, other=0.0).to(e0.dtype)
                g1 = tl.load(base + N + n, mask=m, other=0.0).to(e0.dtype)
                g2 = tl.load(base + 2 * N + n, mask=m, other=0.0).to(e0.dtype)
                ob = out_ptr + b * 2 * N
                tl.store(ob + n, (g0 * e0 + g1 * e1 + g2 * e2).to(out_ptr.dtype.element_ty), mask=m)
                tl.store(ob + N + n, (g0 * n0 + g1 * n1 + g2 * n2).to(out_ptr.dtype.element_ty), mask=m)
            else:
                base = a_ptr + b * 2 * N
                u = tl.load(base + n, mask=m, other=0.0).to(e0.dtype)
                v = tl.load(base + N + n, mask=m, other=0.0).to(e0.dtype)
                ob = out_ptr + b * 3 * N
                tl.store(ob + n, (u * e0 + v * n0).to(out_ptr.dtype.element_ty), mask=m)
                tl.store(ob + N + n, (u * e1 + v * n1).to(out_ptr.dtype.element_ty), mask=m)
                tl.store(ob + 2 * N + n, (u * e2 + v * n2).to(out_ptr.dtype.element_ty), mask=m)

    @triton.autotune(configs=_EW_CFGS, key=["SIZE", "G", "MEAN"],
                     prune_configs_by={"early_config_prune": _prune_ew})
    @triton.jit
    def _contract_fwd_kernel(d_ptr, ax_ptr, bx_ptr, out_ptr, N, G, B, SIZE, SCALE,
                             MEAN: tl.constexpr, BLOCK: tl.constexpr, BB: tl.constexpr):
        """d [B, G, 6, N] -> out [B, 2, N] (MEAN) or [B, G, 2, N]."""
        n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        b0 = tl.program_id(1) * BB
        ok = n < N
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            m = ok & (b < B)
            if d_ptr.dtype.element_ty == tl.float64:
                div_acc = tl.zeros((BLOCK,), dtype=tl.float64)
            else:
                div_acc = tl.zeros((BLOCK,), dtype=tl.float32)
            curl_acc = tl.zeros_like(div_acc)
            for g in range(0, G):
                dbase = d_ptr + ((b * G + g) * 6) * N
                div = tl.zeros_like(div_acc)
                curl = tl.zeros_like(div_acc)
                for c in tl.static_range(3):
                    a = tl.load(ax_ptr + (g * 3 + c) * N + n, mask=ok, other=0.0).to(div.dtype)
                    bb = tl.load(bx_ptr + (g * 3 + c) * N + n, mask=ok, other=0.0).to(div.dtype)
                    da = tl.load(dbase + c * N + n, mask=m, other=0.0).to(div.dtype)
                    db = tl.load(dbase + (3 + c) * N + n, mask=m, other=0.0).to(div.dtype)
                    div += a * da + bb * db
                    curl += bb * da - a * db
                if MEAN:
                    div_acc += div
                    curl_acc += curl
                else:
                    obase = out_ptr + ((b * G + g) * 2) * N
                    tl.store(obase + n, div.to(out_ptr.dtype.element_ty), mask=m)
                    tl.store(obase + N + n, curl.to(out_ptr.dtype.element_ty), mask=m)
            if MEAN:
                obase = out_ptr + b * 2 * N
                tl.store(obase + n, (div_acc * SCALE).to(out_ptr.dtype.element_ty), mask=m)
                tl.store(obase + N + n, (curl_acc * SCALE).to(out_ptr.dtype.element_ty), mask=m)

    @triton.autotune(configs=_EW_CFGS, key=["SIZE", "G", "MEAN"],
                     prune_configs_by={"early_config_prune": _prune_ew})
    @triton.jit
    def _contract_bwd_kernel(go_ptr, ax_ptr, bx_ptr, gd_ptr, N, G, B, SIZE, SCALE,
                             MEAN: tl.constexpr, BLOCK: tl.constexpr, BB: tl.constexpr):
        """g_out [B, 2, N] or [B, G, 2, N] -> g_d [B, G, 6, N]."""
        n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        b0 = tl.program_id(1) * BB
        ok = n < N
        for i in tl.static_range(BB):
            b = (b0 + i).to(tl.int64)
            m = ok & (b < B)
            if MEAN:
                gdiv = tl.load(go_ptr + b * 2 * N + n, mask=m, other=0.0)
                gcurl = tl.load(go_ptr + b * 2 * N + N + n, mask=m, other=0.0)
                if gd_ptr.dtype.element_ty == tl.float64:
                    gdiv = gdiv.to(tl.float64) * SCALE
                    gcurl = gcurl.to(tl.float64) * SCALE
                else:
                    gdiv = gdiv.to(tl.float32) * SCALE
                    gcurl = gcurl.to(tl.float32) * SCALE
            for g in range(0, G):
                if not MEAN:
                    obase = go_ptr + ((b * G + g) * 2) * N
                    gdiv = tl.load(obase + n, mask=m, other=0.0)
                    gcurl = tl.load(obase + N + n, mask=m, other=0.0)
                    if gd_ptr.dtype.element_ty == tl.float64:
                        gdiv = gdiv.to(tl.float64)
                        gcurl = gcurl.to(tl.float64)
                    else:
                        gdiv = gdiv.to(tl.float32)
                        gcurl = gcurl.to(tl.float32)
                dbase = gd_ptr + ((b * G + g) * 6) * N
                for c in tl.static_range(3):
                    a = tl.load(ax_ptr + (g * 3 + c) * N + n, mask=ok, other=0.0).to(gdiv.dtype)
                    bb = tl.load(bx_ptr + (g * 3 + c) * N + n, mask=ok, other=0.0).to(gdiv.dtype)
                    tl.store(dbase + c * N + n,
                             (gdiv * a + gcurl * bb).to(gd_ptr.dtype.element_ty), mask=m)
                    tl.store(dbase + (3 + c) * N + n,
                             (gdiv * bb - gcurl * a).to(gd_ptr.dtype.element_ty), mask=m)


def _torch_embed(uv, east, north):
    u = uv[:, 0, :]
    v = uv[:, 1, :]
    return u[:, None, :] * east[None, :, :] + v[:, None, :] * north[None, :, :]


def _torch_contract(d, axis_a, axis_b, mean):
    """Reference contraction (the formulas of ``HealPixDivCurl.forward``)."""
    da = d[:, :, :3, :]
    db = d[:, :, 3:, :]
    a = axis_a[None]
    b = axis_b[None]
    div = (a * da).sum(dim=2) + (b * db).sum(dim=2)
    curl = (b * da).sum(dim=2) - (a * db).sum(dim=2)
    out = torch.stack((div, curl), dim=2)
    return out.mean(dim=1) if mean else out


def _ew_grid(n, B):
    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]), triton.cdiv(B, meta["BB"]))
    return grid


class _EmbedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uv, east, north):
        B, _, N = uv.shape
        out = torch.empty(B, 3, N, dtype=uv.dtype, device=uv.device)
        with _device_ctx(uv.device):
            _embed_kernel[_ew_grid(N, B)](uv, east, north, out, N, B, _size_class(B * N), BWD=False)
        ctx.save_for_backward(east, north)
        return out

    @staticmethod
    def backward(ctx, g):
        east, north = ctx.saved_tensors
        if torch.is_grad_enabled():      # double backward: recorded torch ops
            return torch.stack(((g * east).sum(1), (g * north).sum(1)), 1), None, None
        g = g.contiguous()
        B, _, N = g.shape
        out = torch.empty(B, 2, N, dtype=g.dtype, device=g.device)
        with _device_ctx(g.device):
            _embed_kernel[_ew_grid(N, B)](g, east, north, out, N, B, _size_class(B * N), BWD=True)
        return out, None, None


class _ContractFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, axis_a, axis_b, mean):
        B, G, _, N = d.shape
        shape = (B, 2, N) if mean else (B, G, 2, N)
        out = torch.empty(shape, dtype=d.dtype, device=d.device)
        with _device_ctx(d.device):
            _contract_fwd_kernel[_ew_grid(N, B)](d, axis_a, axis_b, out, N, G, B,
                                                 _size_class(B * N * G), 1.0 / G, MEAN=mean)
        ctx.save_for_backward(axis_a, axis_b)
        ctx.mean = mean
        ctx.G = G
        return out

    @staticmethod
    def backward(ctx, g):
        axis_a, axis_b = ctx.saved_tensors
        mean, G = ctx.mean, ctx.G
        if torch.is_grad_enabled():      # double backward: recorded torch ops
            gg = (g[:, None] / G).expand(-1, G, -1, -1) if mean else g
            gdiv, gcurl = gg[:, :, 0:1, :], gg[:, :, 1:2, :]
            a, b = axis_a[None], axis_b[None]
            return torch.cat((gdiv * a + gcurl * b, gdiv * b - gcurl * a), 2), None, None, None
        g = g.contiguous()
        B, N = g.shape[0], g.shape[-1]
        gd = torch.empty(B, G, 6, N, dtype=g.dtype, device=g.device)
        with _device_ctx(g.device):
            _contract_bwd_kernel[_ew_grid(N, B)](g, axis_a, axis_b, gd, N, G, B,
                                                 _size_class(B * N * G), 1.0 / G, MEAN=mean)
        return gd, None, None, None


def divcurl_embed(uv, east, north, *, backend="auto"):
    """``uv [B, 2, N]`` (eastward, northward) -> Cartesian tangent vector ``[B, 3, N]``."""
    if _use_triton(uv, backend, (torch.float32, torch.float64)):
        return _EmbedFunction.apply(uv.contiguous(), east.contiguous(), north.contiguous())
    return _torch_embed(uv, east, north)


def divcurl_contract(d, axis_a, axis_b, *, mean, backend="auto"):
    """``d [B, G, 6, N]`` directional derivatives -> div / curl, ``[B, 2, N]``
    with the gauge mean, else ``[B, G, 2, N]``."""
    if _use_triton(d, backend, (torch.float32, torch.float64)):
        return _ContractFunction.apply(d.contiguous(), axis_a.contiguous(),
                                       axis_b.contiguous(), bool(mean))
    return _torch_contract(d, axis_a, axis_b, mean)
