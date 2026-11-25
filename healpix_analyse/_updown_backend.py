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
# Triton backend: one gather-reduce kernel
# ===========================================================================

def _size_class(n):
    return int(math.log2(max(int(n), 1))) // 2


if _HAS_TRITON:

    if os.environ.get("TRITON_INTERPRET") == "1":   # the CPU interpreter: one config
        _ELL_CONFIGS = [triton.Config({"BLOCK_I": 64, "BLOCK_B": 2}, num_warps=4)]
    else:
        _ELL_CONFIGS = [
            triton.Config({"BLOCK_I": bi, "BLOCK_B": bb}, num_warps=nw)
            for bi, bb, nw in (
                (128, 1, 4), (256, 1, 4), (512, 1, 4), (1024, 1, 8),
                (256, 2, 4), (512, 2, 8), (256, 4, 4), (512, 4, 8),
                (128, 8, 4), (256, 8, 8),
            )
        ]

    @triton.autotune(configs=_ELL_CONFIGS, key=["R", "B_CLASS", "SIZE"])
    @triton.jit
    def _ell_kernel(x_ptr, idx_ptr, w_ptr, div_ptr, y_ptr,
                    B, N_in, N_out, R, B_CLASS, SIZE,
                    HAS_DIV: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_B: tl.constexpr):
        """y[b, i] = (sum_r w[i, r] * x[b, idx[i, r]]) / div[i], fp32 accumulation.

        One program: BLOCK_I rows of the operator for BLOCK_B rows of x, so
        each (idx, w) entry is read once for BLOCK_B rows of x.  The slots are
        summed in order: the result is the same at every run.
        """
        pid_i = tl.program_id(0)
        pid_b = tl.program_id(1)
        i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        i_ok = i < N_out
        b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        b_ok = b < B
        xrow = b.to(tl.int64) * N_in
        erow = i.to(tl.int64) * R
        acc = tl.zeros((BLOCK_B, BLOCK_I), dtype=tl.float32)
        for r in range(0, R):
            col = tl.load(idx_ptr + erow + r, mask=i_ok, other=N_in)
            wr = tl.load(w_ptr + erow + r, mask=i_ok, other=0.0).to(tl.float32)
            ok = col < N_in
            xv = tl.load(x_ptr + xrow[:, None] + col.to(tl.int64)[None, :],
                         mask=b_ok[:, None] & ok[None, :], other=0.0)
            acc += xv.to(tl.float32) * wr[None, :]
        if HAS_DIV:
            d = tl.load(div_ptr + i, mask=i_ok, other=1.0).to(tl.float32)
            acc = acc / d[None, :]
        yoff = b.to(tl.int64)[:, None] * N_out + i[None, :]
        tl.store(y_ptr + yoff, acc.to(y_ptr.dtype.element_ty),
                 mask=b_ok[:, None] & i_ok[None, :])


def _ell_triton(x, idx, w, div, out_dtype):
    B, n_in = x.shape
    n_out, R = idx.shape
    y = torch.empty((B, n_out), dtype=out_dtype, device=x.device)
    if B == 0 or n_out == 0:
        return y
    x = x.contiguous()
    if div is not None:
        div = div.contiguous()

    def grid(meta):
        return (triton.cdiv(n_out, meta["BLOCK_I"]), triton.cdiv(B, meta["BLOCK_B"]))

    _ell_kernel[grid](x, idx, w, div if div is not None else w, y,
                      B, n_in, n_out, R, _size_class(B), _size_class(n_out),
                      HAS_DIV=div is not None)
    return y


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
