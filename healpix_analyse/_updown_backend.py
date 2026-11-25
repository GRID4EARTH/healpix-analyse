"""Compute backends of :class:`~healpix_analyse.down.HealPixDown` and
:class:`~healpix_analyse.up.HealPixUp` (issue #109, section 1).

Two fast paths replace the sparse CSR product and the child gather of the
reference implementation (``backend="csr"``):

**Contiguous NESTED blocks.**  On the full sphere, the 4 children of parent
``p`` are cells ``4p .. 4p+3``, contiguous in memory.  The same holds on a
regional domain whose ``cell_ids`` are sorted and whose parents are complete
(:func:`nested_blocks`).  Down ``mean`` / ``maxpool`` is then a reduction
over a ``[B, N/4, 4]`` view of the input, and an Up that copies each parent
to its 4 children (``paired_down`` a ``mean`` Down) is an ``expand``: no
index array, no sparse matrix, and the gradient is the standard one.  These
paths do the same floating-point operations as the reference, in the same
order, so their results are bit-identical to it.

**ELL fixed fan-in gather.**  The rows of the ``smooth`` matrix have a nearly
constant number of non-zeros, so the matrix is stored in ELL form: ``idx``
(int32) and ``w``, both ``[N_out, R]``; the entries of row ``i`` are
``(idx[i, r], w[i, r])``, in increasing column order, and the padding of a
short row has ``idx = N_in`` (one past the last column) and ``w = 0``.  The
product::

    y[b, i] = (sum_r w[i, r] * x[b, idx[i, r]]) / div[i]

runs as:

* one Triton gather-reduce kernel on CUDA (float32 / float16 / bfloat16
  data, fp32 accumulation, so 16-bit data is never copied to float32); the
  padding is masked out of the loads;
* ``torch.nn.functional.embedding_bag`` (weighted sum of gathered rows)
  elsewhere: CPU, float64, CUDA without Triton.  It gathers rows, so it reads
  a transposed copy of ``x`` with one extra row of zeros at index ``N_in``,
  where the padding points (a NaN in ``x`` therefore never meets a padding
  weight), in at least float32.

``div`` is the optional ``up_norm`` normalisation of :class:`HealPixUp`.  The
backward pass is the same product with the transposed matrix, stored in ELL
form too (with ``div`` folded into its weights): deterministic, no atomics,
and itself differentiable.

The Triton path keeps the ``[B*C, N]`` row-major layout end to end: the
output is a new contiguous ``[B, N_out]`` tensor, with no permute or
transpose copy.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # Triton is an optional extra
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAS_TRITON = False

BACKENDS = ("auto", "triton", "torch", "csr")
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def triton_available() -> bool:
    """True when the Triton kernel can run (Triton importable and CUDA present)."""
    return _HAS_TRITON and torch.cuda.is_available()


def check_backend(backend: str) -> str:
    """Validate and normalise a ``backend=`` argument."""
    backend = str(backend).lower().strip()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    return backend


# ===========================================================================
# Contiguous NESTED blocks
# ===========================================================================

def nested_blocks(cell_ids: Optional[np.ndarray]) -> bool:
    """True when every group of 4 consecutive cells holds the 4 children of one
    parent, in order, and the parents increase.

    ``None`` (the full sphere) is always contiguous.  A regional domain is
    when its ``cell_ids`` are sorted and its parents complete.
    """
    if cell_ids is None:
        return True
    ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    if ids.size == 0 or ids.size % 4:
        return False
    blk = ids.reshape(-1, 4)
    if not np.array_equal(blk, blk[:, :1] + np.arange(4, dtype=np.int64)):
        return False
    first = blk[:, 0]
    return bool(np.all(first % 4 == 0) and np.all(np.diff(first) > 0))


# ===========================================================================
# ELL storage
# ===========================================================================

class Ell(NamedTuple):
    """One sparse operator in ELL form (see the module docstring)."""

    idx: torch.Tensor        # int32 [N_out, R]; padding = n_cols
    w: torch.Tensor          # [N_out, R]; padding = 0
    n_cols: int              # N_in


def build_ell(rows, cols, vals, n_rows: int, n_cols: int):
    """ELL arrays ``(idx, w)``, both ``[n_rows, R]``, of a COO matrix.

    Entries of a row keep the order in which they are given (a coalesced COO:
    increasing columns), so the sum order is fixed.  Explicit zeros are kept.
    """
    rows = torch.as_tensor(rows, dtype=torch.int64)
    cols = torch.as_tensor(cols, dtype=torch.int64, device=rows.device)
    vals = torch.as_tensor(vals, device=rows.device)
    if n_rows >= 2**31 or n_cols >= 2**31 - 1:
        raise ValueError("ELL operator too large for int32 indices.")
    order = torch.argsort(rows, stable=True)
    rows, cols, vals = rows[order], cols[order], vals[order]
    counts = torch.bincount(rows, minlength=n_rows)
    R = max(int(counts.max()) if rows.numel() else 0, 1)
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=rows.device) - start[rows]
    idx = torch.full((n_rows, R), n_cols, dtype=torch.int32, device=rows.device)
    w = torch.zeros((n_rows, R), dtype=vals.dtype, device=rows.device)
    idx[rows, pos] = cols.to(torch.int32)
    w[rows, pos] = vals
    return idx, w


def ell_to_coo(idx: torch.Tensor, w: torch.Tensor, n_cols: int) -> torch.Tensor:
    """The coalesced COO ``[N_out, n_cols]`` tensor of an ELL operator."""
    n, R = idx.shape
    valid = idx < n_cols
    rows = torch.arange(n, device=idx.device)[:, None].expand(n, R)[valid]
    return torch.sparse_coo_tensor(
        torch.stack([rows, idx[valid].long()]), w[valid], size=(n, n_cols),
        device=idx.device, dtype=w.dtype,
    ).coalesce()


# ===========================================================================
# PyTorch backend: embedding_bag on a transposed copy
# ===========================================================================

def _ell_torch(x, idx, w, div, out_dtype):
    """``x [B, N_in] -> [B, N_out]``, accumulation in ``promote(x, w, float32)``.

    ``embedding_bag`` sums weighted rows, so it reads ``x^T`` (a copy with a
    zero row at ``N_in`` for the padding) and returns ``y^T``.
    """
    B, n_in = x.shape
    acc = torch.promote_types(torch.promote_types(x.dtype, w.dtype), torch.float32)
    xt = x.new_empty((n_in + 1, B), dtype=acc)
    xt[:n_in].copy_(x.T)
    xt[n_in].zero_()
    yt = F.embedding_bag(idx, xt, per_sample_weights=w.to(acc), mode="sum")
    if div is not None:
        yt = yt / div.to(acc)[:, None]
    return yt.T.to(out_dtype).contiguous()


# ===========================================================================
# Triton backend
# ===========================================================================

def _size_class(n):
    return int(math.log2(max(int(n), 1))) // 2


_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

if _HAS_TRITON:

    if _INTERPRET:   # the CPU interpreter: one config per kernel
        _ELL_CONFIGS = [triton.Config({"BLOCK_I": 64, "BLOCK_B": 2}, num_warps=4)]
        _EW_CONFIGS = [triton.Config({"BLOCK": 128}, num_warps=4)]
    else:
        # a small space of likely winners (gather-bound kernels: wide row
        # tiles, a few batch rows per program to reuse the operator entries)
        _ELL_CONFIGS = [
            triton.Config({"BLOCK_I": bi, "BLOCK_B": bb}, num_warps=nw)
            for bi, bb, nw in ((256, 1, 4), (512, 2, 4), (256, 4, 4), (512, 4, 8))
        ]
        _EW_CONFIGS = [
            triton.Config({"BLOCK": bl}, num_warps=nw) for bl, nw in ((1024, 4), (2048, 8))
        ]

    @triton.jit
    def _finite(v):
        w = v.to(tl.float32)         # exact for 16-bit values
        return (w - w) == 0          # False for NaN and +-inf

    # ---- ELL gather-reduce ------------------------------------------------

    @triton.autotune(configs=_ELL_CONFIGS, key=["R", "B_CLASS", "SIZE", "HAS_PRO", "HAS_EPI"])
    @triton.jit
    def _ell_kernel(x_ptr, idx_ptr, w_ptr, div_ptr, out_ptr,
                    pro_y_ptr, pro_c_ptr, pro_s_ptr, epi_x_ptr, epi_s_ptr,
                    B, N_in, N_out, R, B_CLASS, SIZE,
                    HAS_DIV: tl.constexpr, HAS_PRO: tl.constexpr, HAS_EPI: tl.constexpr,
                    BLOCK_I: tl.constexpr, BLOCK_B: tl.constexpr):
        """out[b, i] = (sum_r w[i, r] * x'[b, idx[i, r]]) / div[i] + epi[b, i].

        fp32 accumulation; the slots are summed in order (deterministic).
        One program: BLOCK_I rows of the operator for BLOCK_B rows of x, so
        each (idx, w) entry is read once for BLOCK_B rows.

        Optional prologue (backward of ``smooth_conservative``):
        ``x'[b, k] = x[b, k] - s[b] * c[k] * finite(y[b, k])``.
        Optional epilogue: ``epi[b, i] = s2[b] * finite(z[b, i])``.
        """
        pid_i = tl.program_id(0)
        pid_b = tl.program_id(1)
        i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        i_ok = i < N_out
        b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_ok = b < B
        xrow = b.to(tl.int64) * N_in
        erow = i.to(tl.int64) * R
        if HAS_PRO:
            s_pro = tl.load(pro_s_ptr + b, mask=b_ok, other=0.0)
        acc = tl.zeros((BLOCK_B, BLOCK_I), dtype=tl.float32)
        for r in range(0, R):
            col = tl.load(idx_ptr + erow + r, mask=i_ok, other=N_in)
            wr = tl.load(w_ptr + erow + r, mask=i_ok, other=0.0).to(tl.float32)
            ok = col < N_in
            m2 = b_ok[:, None] & ok[None, :]
            off = xrow[:, None] + col.to(tl.int64)[None, :]
            xv = tl.load(x_ptr + off, mask=m2, other=0.0).to(tl.float32)
            if HAS_PRO:
                yv = tl.load(pro_y_ptr + off, mask=m2, other=0.0)
                cv = tl.load(pro_c_ptr + col, mask=ok, other=0.0).to(tl.float32)
                xv = xv - tl.where(_finite(yv), s_pro[:, None] * cv[None, :], 0.0)
            acc += xv * wr[None, :]
        if HAS_DIV:
            d = tl.load(div_ptr + i, mask=i_ok, other=1.0).to(tl.float32)
            acc = tl.math.div_rn(acc, d[None, :] + tl.zeros_like(acc))
        yoff = b.to(tl.int64)[:, None] * N_out + i[None, :]
        m_out = b_ok[:, None] & i_ok[None, :]
        if HAS_EPI:
            zv = tl.load(epi_x_ptr + yoff, mask=m_out, other=0.0)
            s_epi = tl.load(epi_s_ptr + b, mask=b_ok, other=0.0)
            acc += tl.where(_finite(zv), s_epi[:, None] + tl.zeros_like(acc), 0.0)
        tl.store(out_ptr + yoff, acc.to(out_ptr.dtype.element_ty), mask=m_out)

    # ---- max over blocks of 4 (contiguous NESTED maxpool) -----------------

    @triton.jit
    def _quad(ptr, p, ok):
        """The 4 values ptr[4p + k], k = 0..3, from one coalesced [BLOCK, 4] load."""
        k = tl.arange(0, 4)
        v = tl.load(ptr + p.to(tl.int64)[:, None] * 4 + k[None, :],
                    mask=ok[:, None], other=0.0)
        v = tl.reshape(v, (v.shape[0], 2, 2))
        v02, v13 = tl.split(v)                   # [BLOCK, 2]: (v0, v2), (v1, v3)
        v0, v2 = tl.split(v02)
        v1, v3 = tl.split(v13)
        return v0, v1, v2, v3

    @triton.jit
    def _take(m, a, v, k):
        """Running max that propagates the first NaN, as torch.max."""
        t = (v > m) | ((v != v) & (m == m))
        return tl.where(t, v, m), tl.where(t, k, a)

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE"])
    @triton.jit
    def _max4_fwd_kernel(x_ptr, y_ptr, arg_ptr, M, SIZE, BLOCK: tl.constexpr):
        """y[p] = max(x[4p:4p+4]), arg[p] = its position (int8)."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < M
        v0, v1, v2, v3 = _quad(x_ptr, p, ok)
        # compared in fp32 (exact for 16-bit values), stored back unchanged
        v0, v1 = v0.to(tl.float32), v1.to(tl.float32)
        v2, v3 = v2.to(tl.float32), v3.to(tl.float32)
        a = tl.zeros((BLOCK,), dtype=tl.int32)
        m, a = _take(v0, a, v1, 1)
        m, a = _take(m, a, v2, 2)
        m, a = _take(m, a, v3, 3)
        tl.store(y_ptr + p, m.to(y_ptr.dtype.element_ty), mask=ok)
        tl.store(arg_ptr + p, a.to(tl.int8), mask=ok)

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE"])
    @triton.jit
    def _max4_bwd_kernel(g_ptr, arg_ptr, gx_ptr, M, SIZE, BLOCK: tl.constexpr):
        """gx[4p + k] = g[p] if k == arg[p] else 0 (one coalesced [BLOCK, 4] store)."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < M
        g = tl.load(g_ptr + p, mask=ok, other=0.0)
        a = tl.load(arg_ptr + p, mask=ok, other=0).to(tl.int32)
        k = tl.arange(0, 4)
        v = tl.where(k[None, :] == a[:, None], g[:, None], 0.0)
        tl.store(gx_ptr + p.to(tl.int64)[:, None] * 4 + k[None, :],
                 v.to(gx_ptr.dtype.element_ty), mask=ok[:, None])

    # ---- copy each parent to its 4 children (contiguous NESTED Up) --------

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE"])
    @triton.jit
    def _expand4_fwd_kernel(x_ptr, v_ptr, d_ptr, y_ptr, M, NC, eps, SIZE,
                            HAS_D: tl.constexpr, BLOCK: tl.constexpr):
        """y[4p + k] = (x[p] * v[c]) / max(d[c], eps), c = p mod NC (fp32)."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < M
        c = p % NC
        x = tl.load(x_ptr + p, mask=ok, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + c, mask=ok, other=0.0).to(tl.float32)
        y = x * v
        if HAS_D:
            d = tl.load(d_ptr + c, mask=ok, other=1.0).to(tl.float32)
            y = tl.math.div_rn(y, tl.maximum(d, eps))
        y = y.to(y_ptr.dtype.element_ty)
        k = tl.arange(0, 4)
        tl.store(y_ptr + p.to(tl.int64)[:, None] * 4 + k[None, :],
                 y[:, None] + tl.zeros((BLOCK, 4), dtype=y.dtype), mask=ok[:, None])

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE"])
    @triton.jit
    def _expand4_bwd_kernel(g_ptr, v_ptr, d_ptr, gx_ptr, M, NC, eps, SIZE,
                            HAS_D: tl.constexpr, BLOCK: tl.constexpr):
        """gx[p] = (sum_k g[4p + k]) * v[c] / max(d[c], eps) (fp32)."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < M
        c = p % NC
        g0, g1, g2, g3 = _quad(g_ptr, p, ok)
        s = ((g0.to(tl.float32) + g1.to(tl.float32)) + g2.to(tl.float32)) + g3.to(tl.float32)
        s = s * tl.load(v_ptr + c, mask=ok, other=0.0).to(tl.float32)
        if HAS_D:
            d = tl.load(d_ptr + c, mask=ok, other=1.0).to(tl.float32)
            s = tl.math.div_rn(s, tl.maximum(d, eps))
        tl.store(gx_ptr + p, s.to(gx_ptr.dtype.element_ty), mask=ok)

    # ---- per-row finite sums (smooth_conservative) ------------------------

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE", "HAS_W"])
    @triton.jit
    def _row_stats_kernel(x_ptr, w_ptr, sum_ptr, cnt_ptr, N, CHUNK, N_CHUNK, SIZE,
                          HAS_W: tl.constexpr, BLOCK: tl.constexpr):
        """Partial sums, in float64, of w * x and of w over the finite x of one
        chunk of one row (w = 1 without HAS_W).  Written per (row, chunk): the
        caller adds the chunks in a fixed order (deterministic, no atomics)."""
        row = tl.program_id(0)
        ch = tl.program_id(1)
        lo = ch * CHUNK
        hi = tl.minimum(lo + CHUNK, N)
        base = row.to(tl.int64) * N
        acc = tl.zeros((BLOCK,), dtype=tl.float64)
        cnt = tl.zeros((BLOCK,), dtype=tl.float64)
        for start in range(lo, hi, BLOCK):
            j = start + tl.arange(0, BLOCK)
            ok = j < hi
            v = tl.load(x_ptr + base + j, mask=ok, other=0.0)
            f = ok & _finite(v)
            v64 = tl.where(f, v.to(tl.float64), 0.0)
            if HAS_W:
                w = tl.load(w_ptr + j, mask=ok, other=0.0).to(tl.float64)
                acc += v64 * w
                cnt += tl.where(f, w, 0.0)
            else:
                acc += v64
                cnt += tl.where(f, 1.0, 0.0).to(tl.float64)
        tl.store(sum_ptr + row * N_CHUNK + ch, tl.sum(acc, axis=0))
        tl.store(cnt_ptr + row * N_CHUNK + ch, tl.sum(cnt, axis=0))

    @triton.autotune(configs=_EW_CONFIGS, key=["SIZE"])
    @triton.jit
    def _add_rows_kernel(y_ptr, d_ptr, out_ptr, N, M, SIZE, BLOCK: tl.constexpr):
        """out[b, i] = y[b, i] + d[b] in the dtype of y, then cast to out."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < M
        y = tl.load(y_ptr + p, mask=ok, other=0.0)
        d = tl.load(d_ptr + p // N, mask=ok, other=0.0)
        tl.store(out_ptr + p, (y + d.to(y.dtype)).to(out_ptr.dtype.element_ty), mask=ok)


def _ell_triton(x, idx, w, div, out_dtype, pro=None, epi=None):
    """ELL product on CUDA; ``pro = (y, c, s)`` and ``epi = (z, s2)`` are the
    optional prologue / epilogue of :func:`_ell_kernel`."""
    B, n_in = x.shape
    n_out, R = idx.shape
    y = torch.empty((B, n_out), dtype=out_dtype, device=x.device)
    if B == 0 or n_out == 0:
        return y
    x = x.contiguous()
    if div is not None:
        div = div.contiguous()
    dummy = w
    py, pc, ps = pro if pro is not None else (dummy, dummy, dummy)
    ez, es = epi if epi is not None else (dummy, dummy)

    def grid(meta):
        return (triton.cdiv(n_out, meta["BLOCK_I"]), triton.cdiv(B, meta["BLOCK_B"]))

    _ell_kernel[grid](x, idx, w, div if div is not None else dummy, y,
                      py, pc, ps, ez, es,
                      B, n_in, n_out, R, _size_class(B), _size_class(n_out),
                      HAS_DIV=div is not None, HAS_PRO=pro is not None,
                      HAS_EPI=epi is not None)
    return y


def _ew_grid(m):
    return lambda meta: (triton.cdiv(m, meta["BLOCK"]),)


def _row_stats(x, w=None):
    """``(sum of w*x, sum of w)`` over the finite entries of each row, float64."""
    B, N = x.shape
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count if x.is_cuda else 4
    n_chunk = max(1, min(triton.cdiv(N, 4096), triton.cdiv(4 * sms, max(B, 1))))
    chunk = triton.cdiv(N, n_chunk)
    s = torch.empty((B, n_chunk), dtype=torch.float64, device=x.device)
    c = torch.empty_like(s)
    _row_stats_kernel[(B, n_chunk)](x, w if w is not None else x, s, c, N, chunk, n_chunk,
                                    _size_class(N), HAS_W=w is not None)
    return s.sum(1), c.sum(1)


# ===========================================================================
# Autograd function and dispatch
# ===========================================================================

def use_triton(x: torch.Tensor, backend: str) -> bool:
    """Whether :func:`ell_matmul` runs the Triton kernel for ``x``."""
    ok = _HAS_TRITON and x.is_cuda and x.dtype in _TRITON_DTYPES
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, a CUDA tensor and float32, float16 "
            "or bfloat16 data (float64 runs on backend='torch')."
        )
    return ok and backend in ("auto", "triton")


def _apply(x, op: Ell, div, out_dtype, triton_ok):
    if triton_ok:
        return _ell_triton(x, op.idx, op.w, div, out_dtype)
    return _ell_torch(x, op.idx, op.w, div, out_dtype)


class EllMatmul(torch.autograd.Function):
    """``y = D^-1 A x`` for an ELL operator ``A`` (``fwd``) and a row
    normalisation ``D`` (``div``, or None).

    The backward is ``(D^-1 A)^T g``: the ELL operator ``bwd``, with ``D^-1``
    folded in its weights, so it needs no normalisation (``bdiv`` None).  It
    is itself an ``EllMatmul`` with the two operators swapped, hence twice
    differentiable.  The operators are constants.
    """

    @staticmethod
    def forward(ctx, x, fwd, bwd, div, bdiv, out_dtype, triton_ok):  # noqa: D102
        ctx.fwd, ctx.bwd, ctx.div, ctx.bdiv = fwd, bwd, div, bdiv
        ctx.in_dtype, ctx.triton_ok = x.dtype, triton_ok
        return _apply(x, fwd, div, out_dtype, triton_ok)

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        triton_ok = ctx.triton_ok and g.dtype in _TRITON_DTYPES
        gx = EllMatmul.apply(g, ctx.bwd, ctx.fwd, ctx.bdiv, ctx.div,
                             ctx.in_dtype, triton_ok)
        return gx, None, None, None, None, None, None


def ell_matmul(x, fwd: Ell, bwd: Ell, *, div=None, out_dtype=None, backend="auto"):
    """``[B, N_in] -> [B, N_out]`` product with an ELL operator.

    Parameters
    ----------
    x : Tensor [B, N_in]
    fwd, bwd : Ell
        The operator ``A`` and the transpose of ``D^-1 A`` (for the backward).
    div : Tensor [N_out] or None
        Row normalisation ``D`` (divides the output).
    out_dtype : torch.dtype or None
        Output dtype (``x.dtype`` by default).  The accumulation is at least
        float32.
    backend : {"auto", "triton", "torch"}
    """
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return EllMatmul.apply(x, fwd, bwd, div, None, out_dtype, use_triton(x, backend))


# ===========================================================================
# Fused CUDA paths of the other modes (Triton), with a PyTorch reference
# ===========================================================================

def _fused_ok(x, backend):
    return (backend in ("auto", "triton") and _HAS_TRITON and x.is_cuda
            and x.dtype in _TRITON_DTYPES)


class _Max4(torch.autograd.Function):
    """Max over contiguous blocks of 4 (``maxpool`` on contiguous NESTED data).

    The forward stores the position of the maximum in int8 (1 byte per output
    instead of the 8 of ``torch.max``); the backward writes each gradient to
    its position and zeros elsewhere, in one pass, without a separate zero
    fill.  Same values, NaN rule and gradient routing as ``torch.max(dim=-1)``.
    """

    @staticmethod
    def forward(ctx, x):  # noqa: D102
        B, N = x.shape
        m = B * (N // 4)
        y = torch.empty((B, N // 4), dtype=x.dtype, device=x.device)
        arg = torch.empty((B, N // 4), dtype=torch.int8, device=x.device)
        if m:
            _max4_fwd_kernel[_ew_grid(m)](x, y, arg, m, _size_class(m))
        ctx.save_for_backward(arg)
        ctx.n = N
        return y

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        (arg,) = ctx.saved_tensors
        if torch.is_grad_enabled():            # double backward: differentiable form
            onehot = F.one_hot(arg.long(), 4).to(g.dtype)
            return (g.unsqueeze(-1) * onehot).reshape(g.shape[0], ctx.n)
        g = g.contiguous()
        gx = torch.empty((g.shape[0], ctx.n), dtype=g.dtype, device=g.device)
        m = g.numel()
        if m:
            _max4_bwd_kernel[_ew_grid(m)](g, arg, gx, m, _size_class(m))
        return gx


def max4(x, backend="auto"):
    """``x [B, 4M] -> [B, M]``: max over contiguous blocks of 4.

    Fused Triton kernels on CUDA; ``torch.max`` elsewhere (bit-identical).
    """
    if _fused_ok(x, backend):
        return _Max4.apply(x.contiguous())
    return x.reshape(x.shape[0], -1, 4).max(dim=-1).values


class _Expand4(torch.autograd.Function):
    """``y[b, 4c + k] = (x[b, c] * v[c]) / max(d[c], eps)`` in float32, cast to
    ``out_dtype``: an Up that copies each parent to its 4 children.  The
    backward sums the 4 children and applies the same scale, in one pass."""

    @staticmethod
    def forward(ctx, x, v, d, eps, out_dtype):  # noqa: D102
        B, nc = x.shape
        y = torch.empty((B, 4 * nc), dtype=out_dtype, device=x.device)
        m = B * nc
        if m:
            _expand4_fwd_kernel[_ew_grid(m)](x, v, d if d is not None else v, y, m, nc,
                                             eps, _size_class(m), HAS_D=d is not None)
        ctx.save_for_backward(v, d if d is not None else v)
        ctx.has_d, ctx.eps, ctx.in_dtype, ctx.shape = d is not None, eps, x.dtype, (B, nc)
        return y

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        v, d = ctx.saved_tensors
        B, nc = ctx.shape
        if torch.is_grad_enabled():            # double backward: differentiable form
            s = g.reshape(B, nc, 4).sum(-1, dtype=torch.float32) * v.float()
            if ctx.has_d:
                s = s / d.float().clamp_min(ctx.eps)
            return s.to(ctx.in_dtype), None, None, None, None
        g = g.contiguous()
        gx = torch.empty((B, nc), dtype=ctx.in_dtype, device=g.device)
        m = B * nc
        if m:
            _expand4_bwd_kernel[_ew_grid(m)](g, v, d, gx, m, nc, ctx.eps, _size_class(m),
                                             HAS_D=ctx.has_d)
        return gx, None, None, None, None


def expand4(x, v, d, eps, out_dtype, backend="auto"):
    """Copy each scaled coarse value to its 4 contiguous children.

    ``x [B, M] -> [B, 4M]``, ``(x * v) / d.clamp_min(eps)`` computed in
    ``v``'s precision (float32 for 16-bit data) and cast to ``out_dtype``: the
    operations of the CSR reference, in the same order (bit-identical).  One
    fused Triton kernel on CUDA (forward and backward), PyTorch elsewhere.
    """
    if _fused_ok(x, backend) and v.dtype == torch.float32:
        return _Expand4.apply(x.contiguous(), v.contiguous(),
                              None if d is None else d.to(torch.float32).contiguous(),
                              float(eps), out_dtype)
    t = x.to(v.dtype) if x.dtype != v.dtype else x
    y = t * v
    if d is not None:
        y = y / d.to(t.dtype).clamp_min(eps)
    B, nc = y.shape
    return y.to(out_dtype).unsqueeze(-1).expand(B, nc, 4).reshape(B, 4 * nc)


class _ConservativeEll(torch.autograd.Function):
    """``smooth_conservative`` on CUDA:
    ``out = M x + delta``, ``delta = mean_fin(x) - mean_cov,fin(M x)`` per row.

    Forward: the ELL kernel (float32 output), two Triton row reductions in
    float64 (finite values only, coverage weights) and one add-and-cast pass.
    Backward, with ``S = sum(g)`` per row and ``delta`` finite:
    ``grad x = M^T (g - S/W_y * cov * finite(y)) + S/n_x * finite(x)``: one ELL
    gather of ``M^T`` with the correction fused as prologue and epilogue.
    """

    @staticmethod
    def forward(ctx, x, fwd, bwd, cov, out_dtype):  # noqa: D102
        y = _ell_triton(x, fwd.idx, fwd.w, None, torch.float32)
        sx, nx = _row_stats(x)
        sy, wy = _row_stats(y, cov)
        delta = sx / nx - sy / wy
        ok = torch.isfinite(delta)
        delta = torch.where(ok, delta, torch.zeros_like(delta))
        out = torch.empty((y.shape[0], y.shape[1]), dtype=out_dtype, device=y.device)
        m = y.numel()
        if m:
            _add_rows_kernel[_ew_grid(m)](y, delta.to(torch.float32), out, y.shape[1], m,
                                          _size_class(m))
        ctx.save_for_backward(x, y, cov, nx, wy, ok)
        ctx.fwd, ctx.bwd, ctx.in_dtype = fwd, bwd, x.dtype
        return out

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        x, y, cov, nx, wy, ok = ctx.saved_tensors
        if torch.is_grad_enabled():            # double backward: the reference
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                out = _conservative_reference(xr, ctx.fwd, ctx.bwd, cov, torch.float32)
                (gx,) = torch.autograd.grad(out, xr, g, create_graph=True)
            return gx.to(ctx.in_dtype), None, None, None, None
        g = g.contiguous()
        S = g.sum(-1, dtype=torch.float64)
        zero = torch.zeros_like(S)
        s_y = torch.where(ok, S / wy, zero).to(torch.float32)
        s_x = torch.where(ok, S / nx, zero).to(torch.float32)
        gx = _ell_triton(g, ctx.bwd.idx, ctx.bwd.w, None, ctx.in_dtype,
                         pro=(y, cov.to(torch.float32), s_y), epi=(x, s_x))
        return gx, None, None, None, None


def _conservative_reference(t, fwd, bwd, cov, comp_dtype, backend="torch"):
    """The PyTorch form of :class:`_ConservativeEll` (``HealPixDown._conserve_mean``)."""
    y = ell_matmul(t, fwd, bwd, out_dtype=comp_dtype, backend=backend)
    cov = cov.to(device=y.device)
    fin_x = torch.isfinite(t)
    fin_y = torch.isfinite(y)
    sum_x = torch.where(fin_x, t, torch.zeros_like(t)).double().sum(-1)
    mean_x = sum_x / fin_x.sum(-1)
    wy = cov * fin_y
    sum_y = (torch.where(fin_y, y, torch.zeros_like(y)).double() * cov).sum(-1)
    mean_y = sum_y / wy.sum(-1)
    delta = mean_x - mean_y
    delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
    return y + delta.to(y.dtype)[:, None]


def conservative_matmul(x, fwd, bwd, cov, out_dtype, comp_dtype, backend="auto"):
    """``smooth_conservative`` product: fused on CUDA, PyTorch elsewhere."""
    if _fused_ok(x, backend):
        return _ConservativeEll.apply(x.contiguous(), fwd, bwd, cov, out_dtype)
    return _conservative_reference(x, fwd, bwd, cov, comp_dtype, backend).to(out_dtype)
