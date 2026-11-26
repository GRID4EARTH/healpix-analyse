"""Masked, coverage-aware normalisation and pooling layers for HEALPix data.

Issue #109, section 5.  ``torch.nn.GroupNorm``, ``LayerNorm`` and
``BatchNorm1d`` see NaN, masked cells and padding as ordinary values.  The
layers of this module take a **validity mask or a coverage** instead:

* HEALPix cells have equal area, so the weight of a cell is just its mask
  value ``w`` in [0, 1] (1 = fully valid, 0 = missing, a fraction = partial
  coverage, e.g. after a ``mean`` Down of a partly covered domain);
* with ``mask=None``, NaN (and +-inf) values of ``x`` are the invalid cells;
  with a mask, non-finite values of ``x`` are invalid as well, whatever the
  mask says;
* cells of weight 0 contribute nothing to the statistics, and their value is
  never read, so a NaN there does not propagate to the output or to the
  gradients;
* with an all-valid mask the layers match their ``torch.nn`` equivalents.

Weighted statistics of a set of cells ``i`` with weights ``w_i``::

    S0   = sum_i w_i
    mean = sum_i w_i x_i / S0
    var  = sum_i w_i (x_i - mean)**2 / S0          (biased, as torch.nn)
    y_i  = (x_i - mean) / sqrt(var + eps) * gamma + beta    where w_i > 0

The output at a cell of weight 0 is chosen by ``fill`` (see :data:`FILLS`).
A cell of weight 0 < w < 1 is normalised like a valid cell; its weight only
changes its share in the statistics.

Backends
--------
``"torch"``  (CPU, float64, CUDA without Triton)
    A few PyTorch passes (validity, row sums, one fused sum of squares, the
    output) and a hand-written backward that saves nothing of size
    ``[B, C, N]`` beyond the input.  :func:`masked_norm_reference`, plain
    differentiable PyTorch, is the ground truth of the tests and serves the
    second derivatives.
``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation)
    Fused kernels for ``HealPixGroupNorm``, ``HealPixBatchNorm`` (training)
    and ``HealPixLayerNorm``: the mask is read through broadcast strides
    (never expanded), the statistics are reduced in fixed-order partial sums
    (no atomics: deterministic), and the backward is written by hand from
    the saved per-row mean and 1/std.  A second derivative
    (``create_graph=True``) falls back to the PyTorch reference.

``backend="auto"`` (default) uses Triton when it can and PyTorch otherwise.
The pooling functions follow the same rule: Triton kernels on CUDA (one read
of ``x``, no atomics), plain PyTorch elsewhere; both are deterministic.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Triton is an optional extra
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = [
    "FILLS",
    "HealPixGroupNorm",
    "HealPixLayerNorm",
    "HealPixBatchNorm",
    "HealPixGlobalPool",
    "HealPixRegionPool",
    "global_pool",
    "region_pool",
]

#: Output at a cell of weight 0: ``"zero"`` (0, the default: safe for the
#: next convolution and for the loss), ``"nan"`` (marks the cell as missing
#: for the next masked layer, e.g. ``HealPixConv(missing="local_mean")``),
#: ``"passthrough"`` (the input value, NaN included; gradient 1 there).
FILLS = ("zero", "nan", "passthrough")
_FILL_CODE = {"zero": 0, "nan": 1, "passthrough": 2}
BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# ===========================================================================
# Inputs, masks and weights
# ===========================================================================

def _check_x(x: torch.Tensor, C: Optional[int] = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor of shape [B, C, N].")
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B, C, N]; got {tuple(x.shape)}.")
    if C is not None and x.shape[1] != C:
        raise ValueError(f"Expected {C} channels, got {x.shape[1]}.")
    return x


def _check_fill(fill: str) -> str:
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}; got {fill!r}.")
    return fill


def _mask_3d(mask, x: torch.Tensor) -> Optional[torch.Tensor]:
    """The mask as a 3-D tensor ``[B|1, C|1, N]`` on the device of ``x``.

    Accepts ``[N]`` (every sample and channel), ``[B, N]`` (one mask per
    sample, every channel) or ``[B, C, N]``; B and C may be 1.  Bool or
    numeric; it is not expanded.  None stays None.
    """
    if mask is None:
        return None
    B, C, N = x.shape
    m = torch.as_tensor(mask, device=x.device)
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
    return m.detach()


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _weights(x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
    """Cell weights ``[B, C, N]`` in the compute dtype: the mask clamped to
    [0, 1] (non-finite mask values count as 0) times ``isfinite(x)``."""
    cdt = _compute_dtype(x.dtype)
    finite = torch.isfinite(x)
    if m is None:
        return finite.to(cdt)
    mw = m.to(cdt)
    mw = torch.where(torch.isfinite(mw), mw, torch.zeros_like(mw)).clamp(0.0, 1.0)
    return mw * finite.to(cdt)


def _validity(x: torch.Tensor, m: Optional[torch.Tensor]):
    """``(valid, cov)``: ``valid`` the bool ``[B, C, N]`` of the cells of
    weight > 0; ``cov`` None for a binary mask (None or bool: the weight is
    ``valid``), else the clamped coverage ``[B|1, C|1, N]`` in the compute
    dtype (the weight is ``where(valid, cov, 0)``)."""
    valid = torch.isfinite(x)
    if m is None:
        return valid, None
    if m.dtype == torch.bool:
        return valid & m, None
    cov = m.to(_compute_dtype(x.dtype))
    cov = torch.where(torch.isfinite(cov), cov, torch.zeros_like(cov)).clamp(0.0, 1.0)
    return valid & (cov > 0), cov


def _apply_fill(y, x, valid, fill):
    if fill == "zero":
        return torch.where(valid, y, torch.zeros_like(y))
    if fill == "nan":
        return torch.where(valid, y, torch.full_like(y, float("nan")))
    return torch.where(valid, y, x.to(y.dtype))


# ===========================================================================
# PyTorch reference
# ===========================================================================

def _weighted_moments(xz, w, dims):
    """S0, mean and biased variance over ``dims`` (keepdim), two-pass."""
    s0 = w.sum(dim=dims, keepdim=True)
    s0c = s0.clamp_min(torch.finfo(s0.dtype).tiny)
    mean = (w * xz).sum(dim=dims, keepdim=True) / s0c
    var = (w * (xz - mean).square()).sum(dim=dims, keepdim=True) / s0c
    return s0, mean, var


def masked_norm_reference(x, mask, kind, num_groups=1, weight=None, bias=None,
                          eps=1e-5, fill="zero", mean=None, var=None):
    """PyTorch reference of the masked normalisations (differentiable).

    ``kind`` is ``"group"`` (statistics per sample and group of channels),
    ``"batch"`` (per channel, over the samples and the cells) or ``"layer"``
    (per sample and cell, over the channels).  With ``mean`` and ``var``
    given (``[C]``, batch kind only), they replace the batch statistics
    (evaluation mode).  Returns ``(y, mean, var, S0)`` with the statistics
    in the shape of their reduction.
    """
    B, C, N = x.shape
    cdt = _compute_dtype(x.dtype)
    m = _mask_3d(mask, x)
    w = _weights(x, m).expand(B, C, N)
    valid = w > 0
    xc = x.to(cdt)
    xz = torch.where(valid, xc, torch.zeros_like(xc))

    if kind == "group":
        G = num_groups
        xv = xz.reshape(B, G, -1)
        wv = w.reshape(B, G, -1)
        s0, mu, va = _weighted_moments(xv, wv, (2,))
        y = ((xv - mu) * torch.rsqrt(va + eps)).reshape(B, C, N)
    elif kind == "batch":
        if mean is not None:
            mu = mean.to(cdt).view(1, C, 1)
            va = var.to(cdt).view(1, C, 1)
            s0 = w.sum(dim=(0, 2), keepdim=True)
        else:
            s0, mu, va = _weighted_moments(xz, w, (0, 2))
        y = (xz - mu) * torch.rsqrt(va + eps)
    elif kind == "layer":
        s0, mu, va = _weighted_moments(xz, w, (1,))
        y = (xz - mu) * torch.rsqrt(va + eps)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    if weight is not None:
        y = y * weight.to(cdt).view(1, C, 1)
    if bias is not None:
        y = y + bias.to(cdt).view(1, C, 1)
    y = _apply_fill(y, x, valid, fill).to(x.dtype)
    return y, mu, va, s0


# ===========================================================================
# PyTorch backend: few passes, hand-written backward
# ===========================================================================

def _row_sum(t, kind, G):
    """Sums over the cells of each statistics row: ``[B, G]`` (group),
    ``[C]`` (batch) or ``[B, N]`` (layer)."""
    B, C, N = t.shape
    if kind == "group":
        return t.reshape(B, G, -1).sum(-1)
    if kind == "batch":
        return t.sum((0, 2))
    return t.sum(1)


def _row_sq(d, kind, G):
    """Sum of squares over each statistics row (one fused reduction)."""
    B, C, N = d.shape
    if kind == "group":
        return torch.linalg.vector_norm(d.reshape(B, G, -1), dim=-1).square()
    if kind == "batch":
        return torch.linalg.vector_norm(d, dim=(0, 2)).square()
    return torch.linalg.vector_norm(d, dim=1).square()


def _bc(s, kind, C, G):
    """Row statistics broadcastable to ``[B, C, N]``."""
    if kind == "group":
        return s.repeat_interleave(C // G, dim=1).unsqueeze(-1)
    if kind == "batch":
        return s.view(1, C, 1)
    return s.unsqueeze(1)


def _masked_dev(x, m, kind, G, mu=None):
    """valid, cov, xz (x at the valid cells, 0 elsewhere), and with ``mu``
    the masked deviation ``where(valid, x - mu, 0)``."""
    cdt = _compute_dtype(x.dtype)
    valid, cov = _validity(x, m)
    xc = x.to(cdt)
    zero = torch.zeros((), dtype=cdt, device=x.device)
    xz = torch.where(valid, xc, zero)
    if mu is None:
        return valid, cov, xz, None
    C = x.shape[1]
    d = (xz - _bc(mu, kind, C, G)).mul_(valid)
    return valid, cov, xz, d


class _TorchNormFunction(torch.autograd.Function):
    """Masked group / batch / layer normalisation in a few PyTorch passes.

    Forward: the validity, ``xz``, the row sums, the masked deviation and
    one fused sum of squares, then ``y = d * scale + shift``.  Backward,
    by hand from ``x`` and the saved row mean and 1/std (nothing of size
    ``[B, C, N]`` is saved beyond ``x``)::

        dx = rstd * (gamma * g - w * (A + xhat * Bv))
        A  = sum(gamma g) / S0,   Bv = sum(gamma g xhat) / S0   (per row)

    with ``g`` the output gradient at the valid cells.  A second derivative
    goes through :func:`masked_norm_reference`.
    """

    @staticmethod
    def forward(ctx, x, m, weight, bias, kind, G, eps, fill):
        B, C, N = x.shape
        cdt = _compute_dtype(x.dtype)
        valid, cov, xz, _ = _masked_dev(x, m, kind, G)
        if cov is None:
            s0 = _row_sum(valid, kind, G).to(cdt)
            s1 = _row_sum(xz, kind, G)
        else:
            w = torch.where(valid, cov, torch.zeros((), dtype=cdt, device=x.device))
            s0 = _row_sum(w.expand(B, C, N), kind, G)
            s1 = _row_sum(xz * w, kind, G)
        s0c = s0.clamp_min(torch.finfo(cdt).tiny)
        mu = s1 / s0c
        d = (xz - _bc(mu, kind, C, G)).mul_(valid)
        if cov is None:
            m2 = _row_sq(d, kind, G)
        else:
            m2 = _row_sum(d.square() * w, kind, G)
        var = m2 / s0c
        rstd = torch.rsqrt(var + eps)
        affine = weight is not None
        if kind == "layer":
            y = d.mul_(_bc(rstd, kind, C, G))
            if affine:
                y = torch.addcmul(bias.to(cdt).view(1, C, 1), y, weight.to(cdt).view(1, C, 1))
        else:
            scale = _bc(rstd, kind, C, G)
            if affine:
                scale = scale * weight.to(cdt).view(1, C, 1)
                y = torch.addcmul(bias.to(cdt).view(1, C, 1), d, scale)
            else:
                y = d.mul_(scale)
        if fill == "zero":
            if affine:
                y = y.mul_(valid)
        elif fill == "nan":
            y = torch.where(valid, y, torch.full((), float("nan"), dtype=cdt, device=x.device))
        else:
            y = torch.where(valid, y, x.to(cdt))
        ctx.save_for_backward(x, m if m is not None else torch.empty(0),
                              weight if affine else torch.empty(0), mu, rstd, s0)
        ctx.meta = (kind, G, eps, fill, m is not None, affine)
        ctx.mark_non_differentiable(mu, var, s0)
        return y.to(x.dtype), mu, var, s0

    @staticmethod
    def backward(ctx, dy, _dmu, _dvar, _ds0):
        x, m, weight, mu, rstd, s0 = ctx.saved_tensors
        kind, G, eps, fill, has_m, affine = ctx.meta
        m = m if has_m else None
        if torch.is_grad_enabled():
            return _reference_backward(ctx, x, m, weight if affine else None, dy,
                                       kind, G, eps, fill)
        B, C, N = x.shape
        cdt = _compute_dtype(x.dtype)
        zero = torch.zeros((), dtype=cdt, device=x.device)
        valid, cov, _, d = _masked_dev(x, m, kind, G, mu)
        rb = _bc(rstd, kind, C, G)
        xh = d.mul_(rb)
        dyc = dy.to(cdt)
        g = torch.where(valid, dyc, zero)
        gam = weight.to(cdt) if affine else torch.ones(C, dtype=cdt, device=x.device)
        s0c = s0.clamp_min(torch.finfo(cdt).tiny)
        if kind == "layer":
            gx = g * xh
            dw = gx.sum((0, 2)) if affine else None
            db = g.sum((0, 2)) if affine else None
            sg = torch.einsum("bcn,c->bn", g, gam)
            sgx = torch.einsum("bcn,c->bn", gx, gam)
            del gx
        else:
            P1 = g.sum(-1)                                   # [B, C]
            P2 = torch.linalg.vecdot(g, xh)                  # [B, C]
            dw = P2.sum(0) if affine else None
            db = P1.sum(0) if affine else None
            g1, g2 = P1 * gam, P2 * gam
            if kind == "group":
                sg, sgx = g1.view(B, G, -1).sum(-1), g2.view(B, G, -1).sum(-1)
            else:
                sg, sgx = g1.sum(0), g2.sum(0)
        A = _bc(sg / s0c, kind, C, G)
        Bv = _bc(sgx / s0c, kind, C, G)
        t = torch.addcmul(A, xh, Bv)
        if cov is None:
            t = t.mul_(valid)
        else:
            t = t.mul_(torch.where(valid, cov, zero))
        gam3 = gam.view(1, C, 1)
        if kind == "layer":
            dx = (g * gam3).sub_(t).mul_(rb)
        else:
            dx = (g * (rb * gam3)).sub_(t.mul_(rb))
        if fill == "passthrough":
            dx = torch.where(valid, dx, dyc)
        dx = dx.to(x.dtype)
        if dw is not None:
            dw = dw.to(weight.dtype) if ctx.needs_input_grad[2] else None
            db = db.to(weight.dtype) if ctx.needs_input_grad[3] else None
        return (dx if ctx.needs_input_grad[0] else None), None, dw, db, None, None, None, None


def _batch_norm_eval(x, m, rm, rv, weight, bias, eps, fill):
    """Element-wise masked batch normalisation with given statistics."""
    C = x.shape[1]
    cdt = _compute_dtype(x.dtype)
    valid, _ = _validity(x, m)
    scale = torch.rsqrt(rv.to(cdt) + eps)
    if weight is not None:
        scale = scale * weight.to(cdt)
    shift = -rm.to(cdt) * scale
    if bias is not None:
        shift = shift + bias.to(cdt)
    y = torch.addcmul(shift.view(1, C, 1), x.to(cdt), scale.view(1, C, 1))
    return _apply_fill(y, x, valid, fill).to(x.dtype)


# ===========================================================================
# Triton kernels
# ===========================================================================
#
# Memory traffic (the kernels are bandwidth-bound; T = one pass over x):
#
#   group / batch, forward   stats: read x (one-pass Welford)   normalise: read x, write y   = 3 T
#   group / batch, backward  sums:  read x, dy                  dx: read x, dy, write dx     = 5 T
#   layer (C <= 512), fwd    one tile [C, BLOCK_N] in registers: read x, write y            = 2 T
#   layer (C <= 512), bwd    read x, dy, write dx                                           = 3 T
#
# torch.nn.GroupNorm / BatchNorm also make two passes (3 T forward).  The
# mask is read through broadcast strides, so an [N] or [B, N] mask adds
# almost nothing.  Every kernel that reduces runs with a fixed tile, a fixed
# chunking and a fixed num_warps, so the order of its sums does not depend
# on the run: the outputs and gradients are bitwise reproducible, across
# processes too.  Only the element-wise kernels (no reduction) autotune.

def triton_available() -> bool:
    """True when Triton is installed and a CUDA device is visible."""
    return _HAS_TRITON and torch.cuda.is_available()


_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
# HEALPIX_AUTOTUNE=full widens the autotune space (slower first call).
_FULL_TUNE = os.environ.get("HEALPIX_AUTOTUNE") == "full"
_RED_BLOCK = 1024           # tile of the reductions (fixed: sets the summation order)
_LN_BWD_BLOCK = 64          # cells per program of the LayerNorm backward (fixed)
_LN_TILE_MAX_C = 512        # widest C served by the one-tile LayerNorm kernels


def _ew_configs():
    if _INTERPRET:
        return [triton.Config({"BLOCK": 1024}, num_warps=4)]
    if _FULL_TUNE:
        return [triton.Config({"BLOCK": b}, num_warps=w)
                for b in (1024, 2048, 4096) for w in (4, 8)]
    return [triton.Config({"BLOCK": 1024}, num_warps=4),
            triton.Config({"BLOCK": 4096}, num_warps=8)]


# Kernels that reduce run with a fixed tile and a fixed num_warps: the order
# of every sum, hence each bit of the result, is the same in every process.
_RED_WARPS = 4
_TILE_WARPS = 8             # [rows, cells] tiles of up to 16 K elements
_LN_FWD_BLOCK_N = 32        # cells per program of the LayerNorm forward (fixed)


if _HAS_TRITON:

    @triton.jit
    def _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M: tl.constexpr):
        """Cell weight: the clamped mask (1 without mask) where x is finite."""
        finite = (x == x) & (tl.abs(x) < float("inf"))
        if HAS_M:
            w = tl.load(m_ptr + b * smb + c * smc + offs, mask=inb, other=0).to(tl.float32)
            w = tl.where(w == w, w, 0.0)
            w = tl.minimum(tl.maximum(w, 0.0), 1.0)
        else:
            w = tl.full(x.shape, 1.0, tl.float32)
        return tl.where(finite & inb, w, 0.0)

    @triton.jit
    def _fill_out(y, x, w, FILL: tl.constexpr):
        if FILL == 0:
            y = tl.where(w > 0, y, 0.0)
        elif FILL == 1:
            y = tl.where(w > 0, y, float("nan"))
        else:
            y = tl.where(w > 0, y, x)
        return y

    # ---------------------------------------------------------------- group / batch

    @triton.jit
    def _rows_stats_kernel(
        x_ptr, m_ptr, part_ptr,
        C, N, CHUNK, NCH, smb, smc,
        HAS_M: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Partial (S0, mean, M2) of a chunk of one (b, c) row, in ONE read:
        a weighted Welford update per lane, the lanes merged by Chan's rule."""
        bc = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        b = bc // C
        c = bc % C
        start = j * CHUNK
        row = x_ptr + bc * N
        ar = tl.arange(0, BLOCK)
        n = tl.zeros([BLOCK], tl.float32)
        mean = tl.zeros([BLOCK], tl.float32)
        m2 = tl.zeros([BLOCK], tl.float32)
        for k in range(0, CHUNK, BLOCK):
            offs = start + k + ar
            inb = (offs < N) & (k + ar < CHUNK)
            x = tl.load(row + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            x = tl.where(w > 0, x, 0.0)
            nn_ = n + w
            delta = x - mean
            mean = mean + tl.where(nn_ > 0, w * delta / tl.maximum(nn_, 1e-30), 0.0)
            m2 = m2 + w * delta * (x - mean)
            n = nn_
        s0 = tl.sum(n, 0)
        mu = tl.sum(n * mean, 0) / tl.maximum(s0, 1e-30)
        d = mean - mu
        m2t = tl.sum(m2, 0) + tl.sum(n * d * d, 0)
        out = part_ptr + (bc * NCH + j) * 3
        tl.store(out, s0)
        tl.store(out + 1, mu)
        tl.store(out + 2, m2t)

    @triton.autotune(configs=_ew_configs(), key=["N", "C"])
    @triton.jit
    def _rows_norm_fwd_kernel(
        x_ptr, m_ptr, y_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr,
        C, N, smb, smc, SB, CG,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """y = (x - mean) * rstd * gamma + beta at the valid cells, fill
        elsewhere: normalisation, affine step and fill in one pass."""
        pid = tl.program_id(0).to(tl.int64)
        nblk = tl.cdiv(N, BLOCK)
        bc = pid // nblk
        b = bc // C
        c = bc % C
        s = b * SB + c // CG
        offs = (pid % nblk) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        base = bc * N
        x = tl.load(x_ptr + base + offs, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
        scale = tl.load(rstd_ptr + s)
        shift = -tl.load(mean_ptr + s) * scale
        if AFFINE:
            gam = tl.load(g_ptr + c).to(tl.float32)
            scale = scale * gam
            shift = shift * gam + tl.load(b_ptr + c).to(tl.float32)
        y = _fill_out(x * scale + shift, x, w, FILL)
        tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=inb)

    @triton.jit
    def _rows_bwd_part_kernel(
        x_ptr, m_ptr, dy_ptr, mean_ptr, rstd_ptr, part_ptr,
        C, N, CHUNK, NCH, smb, smc, SB, CG,
        HAS_M: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Partial sums of dy and dy * xhat over the valid cells of a chunk."""
        bc = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        b = bc // C
        c = bc % C
        s = b * SB + c // CG
        mu = tl.load(mean_ptr + s)
        r = tl.load(rstd_ptr + s)
        start = j * CHUNK
        base = bc * N
        ar = tl.arange(0, BLOCK)
        p1 = tl.zeros([BLOCK], tl.float32)
        p2 = tl.zeros([BLOCK], tl.float32)
        for k in range(0, CHUNK, BLOCK):
            offs = start + k + ar
            inb = (offs < N) & (k + ar < CHUNK)
            x = tl.load(x_ptr + base + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            dy = tl.load(dy_ptr + base + offs, mask=inb, other=0.0).to(tl.float32)
            dy = tl.where(w > 0, dy, 0.0)
            p1 += dy
            p2 += dy * tl.where(w > 0, (x - mu) * r, 0.0)
        out = part_ptr + (bc * NCH + j) * 2
        tl.store(out, tl.sum(p1, 0))
        tl.store(out + 1, tl.sum(p2, 0))

    @triton.autotune(configs=_ew_configs(), key=["N", "C"])
    @triton.jit
    def _rows_bwd_dx_kernel(
        x_ptr, m_ptr, dy_ptr, dx_ptr, mean_ptr, rstd_ptr, a_ptr, bb_ptr, g_ptr,
        C, N, smb, smc, SB, CG,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """dx = rstd * (gamma * dy - w * (A + xhat * B)) at the valid cells."""
        pid = tl.program_id(0).to(tl.int64)
        nblk = tl.cdiv(N, BLOCK)
        bc = pid // nblk
        b = bc // C
        c = bc % C
        s = b * SB + c // CG
        offs = (pid % nblk) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        base = bc * N
        x = tl.load(x_ptr + base + offs, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
        dy = tl.load(dy_ptr + base + offs, mask=inb, other=0.0).to(tl.float32)
        mu = tl.load(mean_ptr + s)
        r = tl.load(rstd_ptr + s)
        A = tl.load(a_ptr + s)
        Bv = tl.load(bb_ptr + s)
        gam = 1.0
        if AFFINE:
            gam = tl.load(g_ptr + c).to(tl.float32)
        xh = (x - mu) * r
        dx = r * (gam * dy - w * (A + xh * Bv))
        if FILL == 2:
            dx = tl.where(w > 0, dx, dy)
        else:
            dx = tl.where(w > 0, dx, 0.0)
        tl.store(dx_ptr + base + offs, dx.to(dx_ptr.dtype.element_ty), mask=inb)

    # ---------------------------------------------------------------- layer, one tile

    @triton.jit
    def _ln_tile_fwd_kernel(
        x_ptr, m_ptr, y_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr,
        C, N, smb, smc, eps,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """LayerNorm of BLOCK_N cells: the [C, BLOCK_N] tile is read once,
        reduced over C in registers, and written once."""
        pid = tl.program_id(0).to(tl.int64)
        nblk = tl.cdiv(N, BLOCK_N)
        b = pid // nblk
        offs = (pid % nblk) * BLOCK_N + tl.arange(0, BLOCK_N)
        cc = tl.arange(0, BLOCK_C)
        inb = (cc[:, None] < C) & (offs[None, :] < N)
        base = b * C * N
        ptr = base + cc[:, None] * N + offs[None, :]
        x = tl.load(x_ptr + ptr, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, cc[:, None], offs[None, :], inb, smb, smc, HAS_M)
        xz = tl.where(w > 0, x, 0.0)
        s0 = tl.sum(w, 0)
        s0c = tl.maximum(s0, 1e-30)
        mu = tl.sum(w * xz, 0) / s0c
        d = tl.where(w > 0, xz - mu[None, :], 0.0)
        r = 1.0 / tl.sqrt(tl.sum(w * d * d, 0) / s0c + eps)
        y = d * r[None, :]
        if AFFINE:
            gam = tl.load(g_ptr + cc, mask=cc < C, other=0.0).to(tl.float32)
            bet = tl.load(b_ptr + cc, mask=cc < C, other=0.0).to(tl.float32)
            y = y * gam[:, None] + bet[:, None]
        y = _fill_out(y, x, w, FILL)
        tl.store(y_ptr + ptr, y.to(y_ptr.dtype.element_ty), mask=inb)
        nm = offs < N
        tl.store(mean_ptr + b * N + offs, mu, mask=nm)
        tl.store(rstd_ptr + b * N + offs, r, mask=nm)

    @triton.jit
    def _ln_tile_bwd_kernel(
        x_ptr, m_ptr, dy_ptr, dx_ptr, g_ptr, mean_ptr, rstd_ptr, pg_ptr, pb_ptr,
        C, N, NBLK, smb, smc,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """dx of BLOCK_N cells from one read of the x and dy tiles; partial
        dgamma / dbeta of the program (fixed BLOCK_N: deterministic)."""
        pid = tl.program_id(0).to(tl.int64)
        b = pid // NBLK
        pb = pid % NBLK
        offs = pb * BLOCK_N + tl.arange(0, BLOCK_N)
        cc = tl.arange(0, BLOCK_C)
        cm = cc < C
        inb = cm[:, None] & (offs[None, :] < N)
        base = b * C * N
        ptr = base + cc[:, None] * N + offs[None, :]
        x = tl.load(x_ptr + ptr, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, cc[:, None], offs[None, :], inb, smb, smc, HAS_M)
        dyr = tl.load(dy_ptr + ptr, mask=inb, other=0.0).to(tl.float32)
        nm = offs < N
        mu = tl.load(mean_ptr + b * N + offs, mask=nm, other=0.0)
        r = tl.load(rstd_ptr + b * N + offs, mask=nm, other=0.0)
        xh = tl.where(w > 0, (x - mu[None, :]) * r[None, :], 0.0)
        dy = tl.where(w > 0, dyr, 0.0)
        if AFFINE:
            gam = tl.load(g_ptr + cc, mask=cm, other=0.0).to(tl.float32)
            prow = (b * NBLK + pb) * C
            tl.store(pg_ptr + prow + cc, tl.sum(dy * xh, 1), mask=cm)
            tl.store(pb_ptr + prow + cc, tl.sum(dy, 1), mask=cm)
            g = dy * gam[:, None]
        else:
            g = dy
        s0c = tl.maximum(tl.sum(w, 0), 1e-30)
        A = tl.sum(g, 0) / s0c
        Bv = tl.sum(g * xh, 0) / s0c
        dx = r[None, :] * (g - w * (A[None, :] + xh * Bv[None, :]))
        if FILL == 2:
            dx = tl.where(w > 0, dx, dyr)
        else:
            dx = tl.where(w > 0, dx, 0.0)
        tl.store(dx_ptr + ptr, dx.to(dx_ptr.dtype.element_ty), mask=inb)

    # ---------------------------------------------------------------- layer, channel loop (C > 512)

    @triton.jit
    def _ln_fwd_kernel(
        x_ptr, m_ptr, y_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr,
        C, N, smb, smc, eps,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """LayerNorm for wide C: loops over the channels (3 reads of x)."""
        pid = tl.program_id(0).to(tl.int64)
        nblk = tl.cdiv(N, BLOCK)
        b = pid // nblk
        offs = (pid % nblk) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        base = b * C * N
        s0 = tl.zeros([BLOCK], tl.float32)
        s1 = tl.zeros([BLOCK], tl.float32)
        for c in range(0, C):
            x = tl.load(x_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            s0 += w
            s1 += w * tl.where(w > 0, x, 0.0)
        mu = s1 / tl.maximum(s0, 1e-30)
        m2 = tl.zeros([BLOCK], tl.float32)
        for c in range(0, C):
            x = tl.load(x_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            d = tl.where(w > 0, x - mu, 0.0)
            m2 += w * d * d
        r = 1.0 / tl.sqrt(m2 / tl.maximum(s0, 1e-30) + eps)
        tl.store(mean_ptr + b * N + offs, mu, mask=inb)
        tl.store(rstd_ptr + b * N + offs, r, mask=inb)
        for c in range(0, C):
            x = tl.load(x_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            y = (x - mu) * r
            if AFFINE:
                y = y * tl.load(g_ptr + c).to(tl.float32) + tl.load(b_ptr + c).to(tl.float32)
            y = _fill_out(y, x, w, FILL)
            tl.store(y_ptr + base + c * N + offs, y.to(y_ptr.dtype.element_ty), mask=inb)

    @triton.jit
    def _ln_bwd_kernel(
        x_ptr, m_ptr, dy_ptr, dx_ptr, g_ptr, mean_ptr, rstd_ptr, pg_ptr, pb_ptr,
        C, N, NBLK, smb, smc,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        b = pid // NBLK
        pb = pid % NBLK
        offs = pb * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        base = b * C * N
        mu = tl.load(mean_ptr + b * N + offs, mask=inb, other=0.0)
        r = tl.load(rstd_ptr + b * N + offs, mask=inb, other=0.0)
        s0 = tl.zeros([BLOCK], tl.float32)
        sg = tl.zeros([BLOCK], tl.float32)
        sgx = tl.zeros([BLOCK], tl.float32)
        prow = (b * NBLK + pb) * C
        for c in range(0, C):
            x = tl.load(x_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            dy = tl.load(dy_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            dy = tl.where(w > 0, dy, 0.0)
            xh = tl.where(w > 0, (x - mu) * r, 0.0)
            gam = 1.0
            if AFFINE:
                gam = tl.load(g_ptr + c).to(tl.float32)
                tl.store(pg_ptr + prow + c, tl.sum(dy * xh, 0))
                tl.store(pb_ptr + prow + c, tl.sum(dy, 0))
            s0 += w
            sg += gam * dy
            sgx += gam * dy * xh
        s0 = tl.maximum(s0, 1e-30)
        A = sg / s0
        Bv = sgx / s0
        for c in range(0, C):
            x = tl.load(x_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            dyr = tl.load(dy_ptr + base + c * N + offs, mask=inb, other=0.0).to(tl.float32)
            gam = 1.0
            if AFFINE:
                gam = tl.load(g_ptr + c).to(tl.float32)
            xh = (x - mu) * r
            dx = r * (gam * dyr - w * (A + xh * Bv))
            if FILL == 2:
                dx = tl.where(w > 0, dx, dyr)
            else:
                dx = tl.where(w > 0, dx, 0.0)
            tl.store(dx_ptr + base + c * N + offs, dx.to(dx_ptr.dtype.element_ty), mask=inb)

    # ---------------------------------------------------------------- pooling

    @triton.jit
    def _pool_part_kernel(
        x_ptr, m_ptr, part_ptr, arg_ptr,
        C, N, CHUNK, NCH, smb, smc,
        HAS_M: tl.constexpr, IS_MAX: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Partial (S0, sum w x) or (S0, max, first argmax) of a chunk."""
        bc = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        b = bc // C
        c = bc % C
        start = j * CHUNK
        row = x_ptr + bc * N
        ar = tl.arange(0, BLOCK)
        s0 = tl.zeros([BLOCK], tl.float32)
        acc = tl.zeros([BLOCK], tl.float32)
        best = tl.full([BLOCK], -float("inf"), tl.float32)
        bidx = tl.zeros([BLOCK], tl.int32)
        for k in range(0, CHUNK, BLOCK):
            offs = start + k + ar
            inb = (offs < N) & (k + ar < CHUNK)
            x = tl.load(row + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            s0 += w
            if IS_MAX:
                xv = tl.where(w > 0, x, -float("inf"))
                upd = xv > best
                best = tl.where(upd, xv, best)
                bidx = tl.where(upd, offs, bidx)
            else:
                acc += w * tl.where(w > 0, x, 0.0)
        out = part_ptr + (bc * NCH + j) * 2
        tl.store(out, tl.sum(s0, 0))
        if IS_MAX:
            mx = tl.max(best, 0)
            first = tl.min(tl.where(best == mx, bidx, 2147483647), 0)
            tl.store(out + 1, mx)
            tl.store(arg_ptr + bc * NCH + j, first)
        else:
            tl.store(out + 1, tl.sum(acc, 0))

    @triton.autotune(configs=_ew_configs(), key=["N", "C"])
    @triton.jit
    def _pool_bwd_kernel(
        x_ptr, m_ptr, lab_ptr, g_ptr, dx_ptr,
        C, N, R, smb, smc,
        HAS_M: tl.constexpr, HAS_LAB: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """dx = w * G[bc, region(n)] at the valid cells (G = dy / S0 for a
        mean, dy for a sum): a gather from the [B*C, R] gradient, no atomics."""
        pid = tl.program_id(0).to(tl.int64)
        nblk = tl.cdiv(N, BLOCK)
        bc = pid // nblk
        b = bc // C
        c = bc % C
        offs = (pid % nblk) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        x = tl.load(x_ptr + bc * N + offs, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
        if HAS_LAB:
            lab = tl.load(lab_ptr + offs, mask=inb, other=-1)
            ok = lab >= 0
            g = tl.load(g_ptr + bc * R + tl.maximum(lab, 0), mask=inb & ok, other=0.0)
            w = tl.where(ok, w, 0.0)
        else:
            g = tl.load(g_ptr + bc)
        dx = w * g
        tl.store(dx_ptr + bc * N + offs, dx.to(dx_ptr.dtype.element_ty), mask=inb)

    @triton.jit
    def _region_ident_kernel(
        x_ptr, m_ptr, s0_ptr, y_ptr, arg_ptr,
        C, N, R, L, smb, smc,
        HAS_M: tl.constexpr, IS_MAX: tl.constexpr,
        BLOCK_G: tl.constexpr, BLOCK_L: tl.constexpr,
    ):
        """Sorted, contiguous regions of L cells: region r of row bc is the
        slice x[bc, r*L:(r+1)*L].  One program reads a contiguous
        [BLOCK_G, L] tile (coalesced, no index list)."""
        pid = tl.program_id(0).to(tl.int64)
        ng = tl.cdiv(R, BLOCK_G)
        bc = pid // ng
        g0 = (pid % ng) * BLOCK_G
        b = bc // C
        c = bc % C
        rg = g0 + tl.arange(0, BLOCK_G)
        kk = tl.arange(0, BLOCK_L)
        cell = rg[:, None] * L + kk[None, :]
        inb = (rg[:, None] < R) & (kk[None, :] < L)
        x = tl.load(x_ptr + bc * N + cell, mask=inb, other=0.0).to(tl.float32)
        w = _load_w(x, m_ptr, b, c, cell, inb, smb, smc, HAS_M)
        out = bc * R + rg
        om = rg < R
        tl.store(s0_ptr + out, tl.sum(w, 1), mask=om)
        if IS_MAX:
            xv = tl.where(w > 0, x, -float("inf"))
            mx = tl.max(xv, 1)
            first = tl.min(tl.where(xv == mx[:, None], cell, 2147483647), 1)
            tl.store(y_ptr + out, mx, mask=om)
            tl.store(arg_ptr + out, first.to(tl.int32), mask=om)
        else:
            tl.store(y_ptr + out, tl.sum(w * tl.where(w > 0, x, 0.0), 1), mask=om)

    @triton.jit
    def _region_part_kernel(
        x_ptr, m_ptr, perm_ptr, offs_ptr, s0_ptr, y_ptr, arg_ptr,
        C, N, R, BC, smb, smc,
        HAS_M: tl.constexpr, IS_MAX: tl.constexpr,
        BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """One region, BLOCK_R (b, c) rows: gather the region's cells (CSR,
        int32) and reduce them in a fixed order."""
        reg = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        rm = rows < BC
        rows64 = rows.to(tl.int64)
        b = rows64 // C
        c = rows64 % C
        k0 = tl.load(offs_ptr + reg)
        k1 = tl.load(offs_ptr + reg + 1)
        ar = tl.arange(0, BLOCK_K)
        s0 = tl.zeros([BLOCK_R, BLOCK_K], tl.float32)
        acc = tl.zeros([BLOCK_R, BLOCK_K], tl.float32)
        best = tl.full([BLOCK_R, BLOCK_K], -float("inf"), tl.float32)
        bidx = tl.zeros([BLOCK_R, BLOCK_K], tl.int32)
        for k in range(k0, k1, BLOCK_K):
            km = (k + ar) < k1
            cell = tl.load(perm_ptr + k + ar, mask=km, other=0)
            inb = rm[:, None] & km[None, :]
            x = tl.load(x_ptr + rows64[:, None] * N + cell[None, :], mask=inb,
                        other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b[:, None], c[:, None], cell[None, :], inb, smb, smc, HAS_M)
            s0 += w
            if IS_MAX:
                xv = tl.where(w > 0, x, -float("inf"))
                upd = xv > best
                best = tl.where(upd, xv, best)
                bidx = tl.where(upd, cell[None, :] + tl.zeros([BLOCK_R, BLOCK_K], tl.int32), bidx)
            else:
                acc += w * tl.where(w > 0, x, 0.0)
        out = rows64 * R + reg
        tl.store(s0_ptr + out, tl.sum(s0, 1), mask=rm)
        if IS_MAX:
            mx = tl.max(best, 1)
            first = tl.min(tl.where(best == mx[:, None], bidx, 2147483647), 1)
            tl.store(y_ptr + out, mx, mask=rm)
            tl.store(arg_ptr + out, first, mask=rm)
        else:
            tl.store(y_ptr + out, tl.sum(acc, 1), mask=rm)


def _mask_args(m, x):
    """Pointer tensor and broadcast strides (sample, channel) of the mask."""
    if m is None:
        return x, 0, 0, False
    B, C, N = x.shape
    if m.dtype == torch.bool:
        m = m.view(torch.uint8)
    m = m.contiguous()
    smb = m.shape[1] * N if m.shape[0] > 1 else 0
    smc = N if m.shape[1] > 1 else 0
    return m, smb, smc, True


def _chunks(rows: int, N: int, device) -> tuple[int, int]:
    """(CHUNK, NCH): enough programs to fill the GPU.  Depends only on the
    shape and the device, never on a tuned configuration."""
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # pragma: no cover - CPU under the interpreter
        sms = 108
    target = 8 * sms
    nch = max(1, min(-(-target // max(rows, 1)), -(-N // _RED_BLOCK)))
    chunk = -(-N // nch)
    chunk = -(-chunk // _RED_BLOCK) * _RED_BLOCK
    nch = -(-N // chunk)
    return chunk, nch


def _ew_grid(rows, N):
    """Flat 1-D grid (rows x blocks of cells): no 65 535 limit on an axis."""
    return lambda meta: (rows * triton.cdiv(N, meta["BLOCK"]),)


def _rows_layout(kind, B, C, G):
    """(SB, CG) of the (b, c) -> statistics-row map: row = b * SB + c // CG."""
    if kind == "group":
        return G, C // G
    return 0, 1  # batch


def _combine_parts(parts, kind, B, C, G):
    """Chan's merge of per-chunk (S0, mean, M2) into per-row S0, mean, var."""
    nch = parts.shape[1]
    p = parts.view(B, C, nch, 3)
    if kind == "group":
        p = p.reshape(B * G, -1, 3)
    else:
        p = p.permute(1, 0, 2, 3).reshape(C, -1, 3)
    s0k, muk, m2k = p.unbind(-1)
    s0 = s0k.sum(1)
    s0c = s0.clamp_min(1e-30)
    mu = (s0k * muk).sum(1) / s0c
    m2 = m2k.sum(1) + (s0k * (muk - mu[:, None]).square()).sum(1)
    return s0, mu, m2 / s0c


def _rows_backward_kernels(x, mt, smb, smc, has_m, dy, mu, rstd, s0, weight, affine,
                           kind, G, fill, with_stats):
    """dx (and the per-channel sums of dy, dy * xhat) of the group / batch
    normalisation; ``with_stats=False`` for given statistics (evaluation)."""
    B, C, N = x.shape
    SB, CG = _rows_layout(kind, B, C, G)
    P1 = P2 = None
    if with_stats or affine:
        chunk, nch = _chunks(B * C, N, x.device)
        parts = torch.empty(B * C, nch, 2, device=x.device, dtype=torch.float32)
        _rows_bwd_part_kernel[(B * C, nch)](
            x, mt, dy, mu, rstd, parts, C, N, chunk, nch, smb, smc, SB, CG,
            HAS_M=has_m, BLOCK=_RED_BLOCK, num_warps=_RED_WARPS)
        P = parts.sum(1).view(B, C, 2)
        P1, P2 = P[..., 0], P[..., 1]
    if with_stats:
        gam = weight.float().view(1, C) if affine else torch.ones(1, C, device=x.device)
        g1, g2 = gam * P1, gam * P2
        if kind == "group":
            sg = g1.view(B, G, -1).sum(-1).reshape(-1)
            sgx = g2.view(B, G, -1).sum(-1).reshape(-1)
        else:
            sg, sgx = g1.sum(0), g2.sum(0)
        s0c = s0.clamp_min(1e-30)
        A = (sg / s0c).contiguous()
        Bv = (sgx / s0c).contiguous()
    else:
        A = Bv = torch.zeros_like(mu)
    dx = torch.empty_like(x)
    _rows_bwd_dx_kernel[_ew_grid(B * C, N)](
        x, mt, dy, dx, mu, rstd, A, Bv, weight if affine else x,
        C, N, smb, smc, SB, CG,
        HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill])
    return dx, P1, P2


class _RowsNormFunction(torch.autograd.Function):
    """Triton group / batch normalisation.  With ``stats=(mean, var)`` given
    ([C], batch kind) the statistics are not computed (evaluation)."""

    @staticmethod
    def forward(ctx, x, m, weight, bias, kind, G, eps, fill, stats=None):
        B, C, N = x.shape
        x_in = x
        x = x.contiguous()
        mt, smb, smc, has_m = _mask_args(m, x)
        SB, CG = _rows_layout(kind, B, C, G)
        if stats is None:
            chunk, nch = _chunks(B * C, N, x.device)
            parts = torch.empty(B * C, nch, 3, device=x.device, dtype=torch.float32)
            _rows_stats_kernel[(B * C, nch)](
                x, mt, parts, C, N, chunk, nch, smb, smc,
                HAS_M=has_m, BLOCK=_RED_BLOCK, num_warps=_RED_WARPS)
            s0, mu, var = _combine_parts(parts, kind, B, C, G)
        else:
            mu = stats[0].float().contiguous()
            var = stats[1].float().contiguous()
            s0 = torch.zeros_like(mu)
        rstd = torch.rsqrt(var + eps)
        affine = weight is not None
        y = torch.empty_like(x)
        _rows_norm_fwd_kernel[_ew_grid(B * C, N)](
            x, mt, y, mu, rstd,
            weight if affine else x, bias if affine else x,
            C, N, smb, smc, SB, CG,
            HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill])
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0),
                              weight if affine else torch.empty(0), mu, rstd, s0)
        ctx.meta = (kind, G, eps, fill, has_m, affine, stats is None)
        ctx.mark_non_differentiable(mu, var, s0)
        return y, mu, var, s0

    @staticmethod
    def backward(ctx, dy, _dmu, _dvar, _ds0):
        x, m, weight, mu, rstd, s0 = ctx.saved_tensors
        kind, G, eps, fill, has_m, affine, with_stats = ctx.meta
        m = m if has_m else None
        w_in = weight if affine else None
        if torch.is_grad_enabled():  # double backward: reference with autograd
            if not with_stats:
                return _eval_reference_backward(ctx, x, m, w_in, mu, rstd, dy, fill)
            return _reference_backward(ctx, x, m, w_in, dy, kind, G, eps, fill) + (None,)
        x = x.contiguous()
        dy = dy.contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        dx, P1, P2 = _rows_backward_kernels(
            x, mt, smb, smc, has_m, dy, mu, rstd, s0, weight, affine, kind, G, fill,
            with_stats)
        dw = P2.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[2] else None
        db = P1.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[3] else None
        return dx, None, dw, db, None, None, None, None, None


def _eval_reference_backward(ctx, x, m, weight, mu, rstd, dy, fill):
    """Graph-keeping backward of the evaluation-mode batch normalisation."""
    C = x.shape[1]
    cdt = _compute_dtype(x.dtype)
    with torch.enable_grad():
        br = None
        if weight is not None and ctx.needs_input_grad[3]:
            br = torch.zeros_like(weight).requires_grad_(True)
        valid, _ = _validity(x, m)
        scale = rstd.to(cdt)
        if weight is not None:
            scale = scale * weight.to(cdt)
        shift = -mu.to(cdt) * scale
        if br is not None:
            shift = shift + br.to(cdt)
        y = torch.addcmul(shift.view(1, C, 1), x.to(cdt), scale.view(1, C, 1))
        y = _apply_fill(y, x, valid, fill).to(x.dtype)
        ins = [(i, t) for i, t in ((0, x), (2, weight), (3, br))
               if t is not None and ctx.needs_input_grad[i] and t.requires_grad]
        grads = torch.autograd.grad(y, [t for _, t in ins], dy,
                                    create_graph=True, allow_unused=True)
    out = [None] * 9
    for (i, _), g in zip(ins, grads):
        out[i] = g
    return tuple(out)


class _LayerNormFunction(torch.autograd.Function):
    """Triton per-cell LayerNorm over the channels."""

    @staticmethod
    def forward(ctx, x, m, weight, bias, eps, fill):
        B, C, N = x.shape
        x_in = x
        x = x.contiguous()
        mt, smb, smc, has_m = _mask_args(m, x)
        affine = weight is not None
        y = torch.empty_like(x)
        mu = torch.empty(B, N, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mu)
        args = (x, mt, y, weight if affine else x, bias if affine else x, mu, rstd,
                C, N, smb, smc, eps)
        if C <= _LN_TILE_MAX_C:
            bc = max(16, triton.next_power_of_2(C))
            bn = max(8, min(_LN_FWD_BLOCK_N, 16384 // bc))
            _ln_tile_fwd_kernel[(B * triton.cdiv(N, bn),)](
                *args, HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill], BLOCK_C=bc,
                BLOCK_N=bn, num_warps=_TILE_WARPS if bc * bn > 2048 else _RED_WARPS)
        else:
            _ln_fwd_kernel[(B * triton.cdiv(N, 128),)](
                *args, HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
                BLOCK=128, num_warps=4)
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0),
                              weight if affine else torch.empty(0), mu, rstd)
        ctx.meta = (eps, fill, has_m, affine)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, m, weight, mu, rstd = ctx.saved_tensors
        eps, fill, has_m, affine = ctx.meta
        m = m if has_m else None
        w_in = weight if affine else None
        if torch.is_grad_enabled():
            dx, _, dw, db, *_ = _reference_backward(ctx, x, m, w_in, dy, "layer", 1, eps, fill)
            return dx, None, dw, db, None, None
        B, C, N = x.shape
        x = x.contiguous()
        dy = dy.contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        tile = C <= _LN_TILE_MAX_C
        if tile:
            bc = max(16, triton.next_power_of_2(C))
            bn = max(8, min(_LN_BWD_BLOCK, 16384 // bc))
        else:
            bn = 128
        nblk = triton.cdiv(N, bn)
        dx = torch.empty_like(x)
        if affine:
            pg = torch.empty(B * nblk, C, device=x.device, dtype=torch.float32)
            pb = torch.empty_like(pg)
        else:
            pg = pb = mu
        args = (x, mt, dy, dx, weight if affine else x, mu, rstd, pg, pb, C, N, nblk, smb, smc)
        if tile:
            _ln_tile_bwd_kernel[(B * nblk,)](
                *args, HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
                BLOCK_C=bc, BLOCK_N=bn, num_warps=_TILE_WARPS)
        else:
            _ln_bwd_kernel[(B * nblk,)](
                *args, HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
                BLOCK=bn, num_warps=4)
        dw = pg.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[2] else None
        db = pb.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[3] else None
        return dx, None, dw, db, None, None


# ---------------------------------------------------------------------------
# Pooling on Triton
# ---------------------------------------------------------------------------

class _GlobalPoolFunction(torch.autograd.Function):
    """Masked global pooling: one read of x (partial sums per chunk, fixed
    order), backward one gather-free element-wise pass (max: a scatter of
    the B*C gradients to their argmax, unique positions)."""

    @staticmethod
    def forward(ctx, x, m, mode):
        B, C, N = x.shape
        x_in = x
        x = x.contiguous()
        mt, smb, smc, has_m = _mask_args(m, x)
        chunk, nch = _chunks(B * C, N, x.device)
        parts = torch.empty(B * C, nch, 2, device=x.device, dtype=torch.float32)
        is_max = mode == "max"
        arg = torch.empty(B * C, nch, device=x.device, dtype=torch.int32) if is_max else parts
        _pool_part_kernel[(B * C, nch)](
            x, mt, parts, arg, C, N, chunk, nch, smb, smc,
            HAS_M=has_m, IS_MAX=is_max, BLOCK=_RED_BLOCK, num_warps=_RED_WARPS)
        s0 = parts[..., 0].sum(1)
        if is_max:
            y, j = parts[..., 1].max(1)
            idx = arg.gather(1, j[:, None]).squeeze(1).long()
        else:
            y = parts[..., 1].sum(1)
            idx = None
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0), s0,
                              idx if idx is not None else torch.empty(0))
        ctx.meta = (mode, has_m)
        ctx.mark_non_differentiable(s0)
        return y.view(B, C), s0.view(B, C)

    @staticmethod
    def backward(ctx, dy, _ds0):
        x, m, s0, idx = ctx.saved_tensors
        mode, has_m = ctx.meta
        m = m if has_m else None
        B, C, N = x.shape
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = _torch_global_pool(x, m, mode)[0]
                (dx,) = torch.autograd.grad(y, x, dy, create_graph=True)
            return dx, None, None
        x = x.contiguous()
        dyf = dy.reshape(-1).float()
        if mode == "max":
            dx = torch.zeros(B * C, N, device=x.device, dtype=torch.float32)
            ok = s0 > 0
            rows = torch.arange(B * C, device=x.device)
            dx[rows[ok], idx[ok]] = dyf[ok]
            return dx.view(B, C, N).to(x.dtype), None, None
        g = torch.where(s0 > 0, dyf, torch.zeros_like(dyf)).contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        dx = torch.empty_like(x)
        _pool_bwd_kernel[_ew_grid(B * C, N)](
            x, mt, x, g, dx, C, N, 1, smb, smc, HAS_M=has_m, HAS_LAB=False)
        return dx, None, None


_MAX_REGION_CACHE: dict = {}


def _max_region(offs):
    """Size of the largest region (cached per offsets tensor)."""
    key = (offs.data_ptr(), offs.numel(), str(offs.device))
    if key not in _MAX_REGION_CACHE:
        if len(_MAX_REGION_CACHE) > 64:
            _MAX_REGION_CACHE.clear()
        _MAX_REGION_CACHE[key] = int((offs[1:] - offs[:-1]).max()) if offs.numel() > 1 else 1
    return _MAX_REGION_CACHE[key]


class _RegionPoolFunction(torch.autograd.Function):
    """Masked pooling over CSR regions: one program per (region, block of
    rows) gathers the region's cells (int32 indices) in a fixed order;
    backward: a gather of the [B*C, R] gradient by the region label of each
    cell (no atomics)."""

    @staticmethod
    def forward(ctx, x, m, mode, perm, offs, labels, ident_len=0):
        B, C, N = x.shape
        R = offs.numel() - 1
        x_in = x
        x = x.contiguous()
        mt, smb, smc, has_m = _mask_args(m, x)
        BC = B * C
        is_max = mode == "max"
        s0 = torch.empty(BC, R, device=x.device, dtype=torch.float32)
        y = torch.empty_like(s0)
        arg = torch.empty(BC, R, device=x.device, dtype=torch.int32) if is_max else s0
        # BLOCK_K covers a typical region (no masked lanes for regions of
        # 4**k cells); BLOCK_R fills a 4 K-element tile with rows.
        if ident_len:
            L = int(ident_len)
            block_l = max(1, triton.next_power_of_2(L))
            block_g = max(1, min(256, 4096 // block_l))
            _region_ident_kernel[(BC * triton.cdiv(R, block_g),)](
                x, mt, s0, y, arg, C, N, R, L, smb, smc,
                HAS_M=has_m, IS_MAX=is_max, BLOCK_G=block_g, BLOCK_L=block_l,
                num_warps=_TILE_WARPS)
        else:
            lmax = _max_region(offs)
            block_k = min(128, max(16, triton.next_power_of_2(lmax)))
            block_r = min(max(1, triton.next_power_of_2(BC)), max(1, 4096 // block_k))
            _region_part_kernel[(R, triton.cdiv(BC, block_r))](
                x, mt, perm, offs, s0, y, arg, C, N, R, BC, smb, smc,
                HAS_M=has_m, IS_MAX=is_max, BLOCK_R=block_r, BLOCK_K=block_k,
                num_warps=_TILE_WARPS)
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0), s0, labels,
                              arg if is_max else torch.empty(0))
        ctx.meta = (mode, has_m, R)
        ctx.mark_non_differentiable(s0)
        return y.view(B, C, R), s0.view(B, C, R)

    @staticmethod
    def backward(ctx, dy, _ds0):
        x, m, s0, labels, arg = ctx.saved_tensors
        mode, has_m, R = ctx.meta
        m = m if has_m else None
        B, C, N = x.shape
        dyf = dy.reshape(B * C, R).float()
        if torch.is_grad_enabled():
            with torch.enable_grad():
                y = _torch_region_pool(x, m, mode, labels.long(), R)[0]
                (dx,) = torch.autograd.grad(y, x, dy, create_graph=True)
            return dx, None, None, None, None, None, None
        if mode == "max":
            dx = torch.zeros(B * C, N, device=x.device, dtype=torch.float32)
            ok = s0 > 0
            rows = torch.arange(B * C, device=x.device)[:, None].expand(B * C, R)
            dx[rows[ok], arg.long()[ok]] = dyf[ok]
            return dx.view(B, C, N).to(x.dtype), None, None, None, None, None, None
        g = torch.where(s0 > 0, dyf, torch.zeros_like(dyf)).contiguous()
        x = x.contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        dx = torch.empty_like(x)
        _pool_bwd_kernel[_ew_grid(B * C, N)](
            x, mt, labels, g, dx, C, N, R, smb, smc, HAS_M=has_m, HAS_LAB=True)
        return dx, None, None, None, None, None, None


def _reference_backward(ctx, x, m, weight, dy, kind, G, eps, fill):
    """Backward through the PyTorch reference, keeping the graph
    (``create_graph=True``): ``x`` and ``weight`` are the saved inputs, still
    connected to the graph of the caller.  Returns the gradients of the
    ``_RowsNormFunction`` inputs (x, mask, weight, bias, ...)."""
    with torch.enable_grad():
        br = None
        if weight is not None and ctx.needs_input_grad[3]:
            br = torch.zeros_like(weight).requires_grad_(True)
        y, *_ = masked_norm_reference(x, m, kind, G, weight, br, eps, fill)
        ins = [(i, t) for i, t in ((0, x), (2, weight), (3, br))
               if t is not None and ctx.needs_input_grad[i] and t.requires_grad]
        grads = torch.autograd.grad(y, [t for _, t in ins], dy,
                                    create_graph=True, allow_unused=True)
    out = [None] * 8
    for (i, _), g in zip(ins, grads):
        out[i] = g
    return tuple(out)


def _use_triton(x: torch.Tensor, backend: str) -> bool:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}.")
    ok = (_HAS_TRITON and (x.is_cuda or (_INTERPRET and backend == "triton"))
          and x.dtype in _TRITON_DTYPES)
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs the triton package, a CUDA tensor and a "
            "float32 / float16 / bfloat16 input."
        )
    return ok and backend in ("auto", "triton")


# ===========================================================================
# Layers
# ===========================================================================

class _MaskedNormMixin:
    """Shared ``fill`` / ``backend`` handling."""

    def _init_masked(self, fill, backend):
        self.fill = _check_fill(fill)
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}.")
        self.backend = backend

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{super().extra_repr()}, fill={self.fill!r}, backend={self.backend!r}"


class HealPixGroupNorm(_MaskedNormMixin, nn.GroupNorm):
    """Group normalisation of ``[B, C, N]`` HEALPix data with a validity mask
    or a coverage.

    The statistics of each sample and group of ``C / num_groups`` channels
    are the weighted mean and (biased) variance over the cells of that group,
    with the cell weights given by ``mask`` (cells have equal area).  With an
    all-valid mask the output equals ``torch.nn.GroupNorm`` (same parameters,
    same ``state_dict``).

    Parameters
    ----------
    num_groups, num_channels, eps, affine, device, dtype
        As ``torch.nn.GroupNorm``.
    fill : {"zero", "nan", "passthrough"}, default "zero"
        Output at the cells of weight 0 (see :data:`FILLS`).
    backend : {"auto", "triton", "torch"}, default "auto"
        ``"auto"``: fused Triton kernels on CUDA for float32 / float16 /
        bfloat16 when Triton is installed, the PyTorch reference otherwise.

    Notes
    -----
    ``forward(x, mask=None)`` takes ``x`` of shape ``[B, C, N]`` and ``mask``
    of shape ``[N]``, ``[B, N]`` or ``[B, C, N]`` (B and C may be 1), bool or
    float in [0, 1].  ``mask=None`` means ``isfinite(x)``; with a mask,
    non-finite values of ``x`` are invalid too.  The mask is not
    differentiated.  Returns ``y`` of the shape and dtype of ``x``; 16-bit
    inputs are reduced in float32.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                 affine: bool = True, fill: str = "zero", backend: str = "auto",
                 device=None, dtype=None):
        super().__init__(num_groups, num_channels, eps, affine, device=device, dtype=dtype)
        self._init_masked(fill, backend)

    def forward(self, x, mask=None):
        _check_x(x, self.num_channels)
        m = _mask_3d(mask, x)
        if _use_triton(x, self.backend):
            y, *_ = _RowsNormFunction.apply(
                x, m, self.weight, self.bias, "group", self.num_groups,
                self.eps, self.fill)
            return y
        y, *_ = _TorchNormFunction.apply(
            x, m, self.weight, self.bias, "group", self.num_groups, self.eps, self.fill)
        return y


class HealPixLayerNorm(_MaskedNormMixin, nn.Module):
    """Layer normalisation over the channels of each cell, for ``[B, C, N]``
    HEALPix data with a validity mask or a coverage.

    Each cell (token) of each sample is normalised over its ``C`` channels,
    as ``torch.nn.LayerNorm(C)`` applied to ``x.transpose(1, 2)`` (the
    ConvNeXt / transformer convention).  For a normalisation over the
    channels *and* the cells of a sample, use ``HealPixGroupNorm(1, C)``.

    A mask of shape ``[N]`` or ``[B, N]`` makes whole cells valid or not;
    a ``[B, C, N]`` mask (or NaN in some channels only, with ``mask=None``)
    weights the channels inside a cell.  A cell whose channels all have
    weight 0 gets ``fill``.

    Parameters
    ----------
    num_channels : int
    eps : float, default 1e-5
    elementwise_affine : bool, default True
        Learnable per-channel ``weight`` and ``bias`` (as ``nn.LayerNorm``).
    fill, backend
        As :class:`HealPixGroupNorm`.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5,
                 elementwise_affine: bool = True, fill: str = "zero",
                 backend: str = "auto", device=None, dtype=None):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        fk = {"device": device, "dtype": dtype}
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.num_channels, **fk))
            self.bias = nn.Parameter(torch.zeros(self.num_channels, **fk))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._init_masked(fill, backend)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"{self.num_channels}, eps={self.eps}, elementwise_affine="
                f"{self.elementwise_affine}, fill={self.fill!r}, backend={self.backend!r}")

    def forward(self, x, mask=None):
        _check_x(x, self.num_channels)
        m = _mask_3d(mask, x)
        if _use_triton(x, self.backend):
            return _LayerNormFunction.apply(x, m, self.weight, self.bias, self.eps, self.fill)
        y, *_ = _TorchNormFunction.apply(
            x, m, self.weight, self.bias, "layer", 1, self.eps, self.fill)
        return y


class HealPixBatchNorm(_MaskedNormMixin, nn.BatchNorm1d):
    """Batch normalisation of ``[B, C, N]`` HEALPix data with a validity mask
    or a coverage, with running statistics.

    In training, the statistics of each channel are the weighted mean and
    (biased) variance over the samples and the cells; the running mean and
    variance are updated with the weighted mean and the unbiased variance
    ``var * S0 / (S0 - 1)``, ``S0`` being the sum of the weights of the
    channel (the number of valid cells for a binary mask).  A channel with
    ``S0 <= 1`` in a batch keeps its running statistics, and that batch does
    not count for it: with ``momentum=None`` (cumulative average) each
    channel averages the batches in which it had valid cells (a per-channel
    count, the non-persistent buffer ``num_batches_tracked_per_channel``,
    rebuilt from ``num_batches_tracked`` when a ``state_dict`` is loaded).
    ``num_batches_tracked`` counts the batches in which at least one channel
    had valid cells.  In evaluation the running statistics are used, unless
    ``track_running_stats=False`` (then the batch statistics, as
    ``torch.nn``).  With an all-valid mask the layer matches
    ``torch.nn.BatchNorm1d`` on ``[B, C, N]``: outputs, running statistics,
    ``momentum=None`` and ``state_dict``.

    Parameters
    ----------
    num_features, eps, momentum, affine, track_running_stats, device, dtype
        As ``torch.nn.BatchNorm1d``.
    fill, backend
        As :class:`HealPixGroupNorm`.  On Triton, evaluation runs one fused
        element-wise kernel (forward) and the matching backward.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: Optional[float] = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 fill: str = "zero", backend: str = "auto", device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats,
                         device=device, dtype=dtype)
        self._init_masked(fill, backend)
        if self.track_running_stats:
            self.register_buffer(
                "num_batches_tracked_per_channel",
                torch.zeros(num_features, dtype=torch.long, device=device), persistent=False)
        else:
            self.num_batches_tracked_per_channel = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.num_batches_tracked_per_channel is not None and self.num_batches_tracked is not None:
            self.num_batches_tracked_per_channel.fill_(int(self.num_batches_tracked))

    def forward(self, x, mask=None):
        _check_x(x, self.num_features)
        m = _mask_3d(mask, x)
        C = self.num_features
        track = (self.training and self.track_running_stats
                 and self.num_batches_tracked is not None)
        use_batch = self.training or (self.running_mean is None and self.running_var is None)

        if not use_batch:
            if _use_triton(x, self.backend):
                y, *_ = _RowsNormFunction.apply(
                    x, m, self.weight, self.bias, "batch", 1, self.eps, self.fill,
                    (self.running_mean, self.running_var))
                return y
            return _batch_norm_eval(x, m, self.running_mean, self.running_var,
                                    self.weight, self.bias, self.eps, self.fill)

        if _use_triton(x, self.backend):
            y, mu, var, s0 = _RowsNormFunction.apply(
                x, m, self.weight, self.bias, "batch", 1, self.eps, self.fill)
        else:
            y, mu, var, s0 = _TorchNormFunction.apply(
                x, m, self.weight, self.bias, "batch", 1, self.eps, self.fill)

        if self.training and self.running_mean is not None:
            with torch.no_grad():
                rm = self.running_mean
                rv = self.running_var
                sdt = _compute_dtype(rm.dtype)
                mu = mu.reshape(C).to(sdt)
                var = var.reshape(C).to(sdt)
                s0 = s0.reshape(C).to(sdt)
                ok = s0 > 1
                # As torch.nn.modules.batchnorm._BatchNorm.forward, but a batch
                # counts only for the channels that had valid cells (no sync).
                if track:
                    self.num_batches_tracked.add_(ok.any().to(self.num_batches_tracked.dtype))
                    cnt = self.num_batches_tracked_per_channel
                    cnt.add_(ok.to(cnt.dtype))
                if self.momentum is None:
                    if track:
                        factor = 1.0 / cnt.clamp_min(1).to(sdt)
                    else:
                        factor = 0.0
                else:
                    factor = self.momentum
                unbiased = var * s0 / (s0 - 1).clamp_min(1e-30)
                new_m = (1 - factor) * rm.to(sdt) + factor * mu
                new_v = (1 - factor) * rv.to(sdt) + factor * unbiased
                rm.copy_(torch.where(ok, new_m, rm.to(sdt)).to(rm.dtype))
                rv.copy_(torch.where(ok, new_v, rv.to(sdt)).to(rv.dtype))
        return y


# ===========================================================================
# Pooling
# ===========================================================================

POOL_MODES = ("mean", "max", "sum")


def _empty_value(fill: str) -> float:
    if fill == "zero":
        return 0.0
    if fill == "nan":
        return float("nan")
    raise ValueError(f"fill must be 'zero' or 'nan' for pooling; got {fill!r}.")


def global_pool(x, mask=None, mode: str = "mean", fill: str = "zero",
                return_weight: bool = False, backend: str = "auto"):
    """Masked global pooling of ``[B, C, N]`` HEALPix data over the cells.

    Parameters
    ----------
    x : torch.Tensor, shape [B, C, N]
        The full sphere or a regional domain (any order of the cells).
    mask : array-like or None
        Cell weights, shape ``[N]``, ``[B, N]`` or ``[B, C, N]``, bool or
        float in [0, 1].  None = ``isfinite(x)``; non-finite values are
        always invalid.
    mode : {"mean", "max", "sum"}, default "mean"
        ``"mean"``: ``sum w x / sum w`` (area-weighted mean, cells have equal
        area); ``"sum"``: ``sum w x``; ``"max"``: maximum over the cells of
        weight > 0 (the weight value does not scale it).
    fill : {"zero", "nan"}, default "zero"
        Result for a (sample, channel) without any valid cell.
    return_weight : bool, default False
        Also return ``sum w`` (``[B, C]``), the covered number of cells.
    backend : {"auto", "triton", "torch"}, default "auto"
        ``"auto"``: a Triton kernel on CUDA (one read of ``x``) when Triton
        is installed.

    Returns
    -------
    y : torch.Tensor, shape [B, C] (and the weight sum if requested)

    Notes
    -----
    The gradient of ``"max"`` goes to the maximum; with ties, the PyTorch
    backend shares it equally (as ``torch.amax``) and the Triton backend
    gives it to the first cell.
    """
    _check_x(x)
    if mode not in POOL_MODES:
        raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
    empty = _empty_value(fill)
    m = _mask_3d(mask, x)
    if _use_triton(x, backend):
        y, s0 = _GlobalPoolFunction.apply(x, m, "max" if mode == "max" else "sum")
    else:
        y, s0 = _torch_global_pool(x, m, mode)
    return _pool_finish(y, s0, mode, empty, x.dtype, return_weight)


def _torch_global_pool(x, m, mode):
    """PyTorch global pooling: (sum or max, S0), shape [B, C]."""
    src, wt = _pool_sources(x, m, mode)
    s0 = wt.sum(-1, dtype=src.dtype).expand(x.shape[0], x.shape[1])
    y = src.amax(-1) if mode == "max" else src.sum(-1)
    return y, s0


def _torch_region_pool(x, m, mode, lab, R):
    """PyTorch region pooling by scatter (reference for double backward)."""
    B, C, N = x.shape
    src, wt = _pool_sources(x, m, mode)
    keep = lab >= 0
    pos = torch.nonzero(keep).reshape(-1)
    li = lab[keep]
    sv = src.index_select(-1, pos)
    if mode == "max":
        y = torch.full((B, C, R), -float("inf"), dtype=src.dtype, device=x.device)
        y = y.scatter_reduce(-1, li.view(1, 1, -1).expand(B, C, -1), sv, "amax")
    else:
        y = torch.zeros(B, C, R, dtype=src.dtype, device=x.device).index_add(-1, li, sv)
    wv = wt.index_select(-1, pos).to(src.dtype)
    s0 = torch.zeros(*wv.shape[:2], R, dtype=src.dtype, device=x.device).index_add(-1, li, wv)
    return y, s0.expand(B, C, R)


def _pool_sources(x, m, mode):
    """The values to reduce (``[B, C, N]``, compute dtype: x at the valid
    cells, times the coverage for mean / sum, -inf / 0 elsewhere) and the
    cell weights (bool ``valid`` or the coverage)."""
    cdt = _compute_dtype(x.dtype)
    valid, cov = _validity(x, m)
    xc = x.to(cdt)
    if mode == "max":
        src = torch.where(valid, xc, torch.full((), -float("inf"), dtype=cdt, device=x.device))
    else:
        src = torch.where(valid, xc if cov is None else xc * cov,
                          torch.zeros((), dtype=cdt, device=x.device))
    wt = valid if cov is None else torch.where(valid, cov, torch.zeros((), dtype=cdt,
                                                                      device=x.device))
    return src, wt


def _pool_finish(y, s0, mode, empty, dtype, return_weight):
    has = s0 > 0
    if mode == "mean":
        y = y / s0.clamp_min(torch.finfo(s0.dtype).tiny)
    y = torch.where(has, y, torch.full((), empty, dtype=y.dtype, device=y.device)).to(dtype)
    return (y, s0) if return_weight else y


class HealPixGlobalPool(nn.Module):
    """Module form of :func:`global_pool`: ``[B, C, N]`` -> ``[B, C]``."""

    def __init__(self, mode: str = "mean", fill: str = "zero", backend: str = "auto"):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
        _empty_value(fill)
        self.mode = mode
        self.fill = fill
        self.backend = backend

    def extra_repr(self) -> str:  # pragma: no cover
        return f"mode={self.mode!r}, fill={self.fill!r}"

    def forward(self, x, mask=None, return_weight: bool = False):
        return global_pool(x, mask, self.mode, self.fill, return_weight, self.backend)


def _region_layout(regions, n_regions=None):
    """Layouts of the regions, as a dict of CPU tensors and flags:

    * ELL (PyTorch backend): ``idx [R, L]`` (padded with 0) and ``pad
      [R, L]``, L being the size of the largest region; ``ident`` when the
      layout is the identity (a free view), ``full`` without padding;
    * CSR (Triton backend): ``perm`` (int32 cell positions sorted by region),
      ``offs`` (int32 ``[R + 1]``) and ``labels`` (int32 ``[N]``, -1 = none).
    """
    if isinstance(regions, torch.Tensor):
        r = regions.detach().to("cpu", torch.int64).reshape(-1)
    else:
        r = torch.as_tensor(np.asarray(regions, dtype=np.int64)).reshape(-1)
    keep = r >= 0
    R = int(n_regions) if n_regions is not None else (int(r.max()) + 1 if keep.any() else 0)
    if keep.any() and int(r[keep].max()) >= R:
        raise ValueError(f"region labels must be < n_regions={R}.")
    pos = torch.nonzero(keep).reshape(-1)
    lab = r[keep]
    order = torch.argsort(lab, stable=True)
    pos, lab = pos[order], lab[order]
    counts = torch.bincount(lab, minlength=R)
    L = int(counts.max()) if R > 0 and counts.numel() else 0
    L = max(L, 1)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(lab.numel()) - starts[lab]
    idx = torch.zeros(R, L, dtype=torch.int64)
    pad = torch.zeros(R, L, dtype=torch.bool)
    idx[lab, rank] = pos
    pad[lab, rank] = True
    n = r.numel()
    full = bool(pad.all())
    ident = full and idx.numel() == n and bool(torch.equal(idx.reshape(-1), torch.arange(n)))
    offs = torch.zeros(R + 1, dtype=torch.int32)
    offs[1:] = torch.cumsum(counts, 0).to(torch.int32)
    labels = torch.where(keep, r, torch.full_like(r, -1)).to(torch.int32)
    return dict(idx=idx, pad=pad, counts=counts, n=n, ident=ident, full=full,
                perm=pos.to(torch.int32), offs=offs, labels=labels)


def _region_reduce(t, idx, pad, ident, full, op, dtype):
    """Reduce ``t [B', C', N]`` over the cells of each region -> ``[B', C', R]``."""
    R, L = idx.shape
    lead = t.shape[:-1]
    if ident:                       # sorted, equal regions: a view
        g = t.reshape(*lead, R, L)
    else:
        g = t.index_select(-1, idx.reshape(-1)).view(*lead, R, L)
    if not full:
        neutral = -float("inf") if op == "max" else 0
        g = torch.where(pad, g, torch.full((), neutral, dtype=g.dtype, device=g.device))
    if op == "max":
        return g.amax(-1)
    return g.sum(-1, dtype=dtype)


def _region_pool_run(x, lay, mask, mode, fill, return_weight, backend):
    _check_x(x)
    if mode not in POOL_MODES:
        raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
    empty = _empty_value(fill)
    B, C, N = x.shape
    m = _mask_3d(mask, x)
    R = lay["idx"].shape[0]
    if _use_triton(x, backend) and R > 0:
        ident_len = lay["idx"].shape[1] if (lay["ident"] and lay["idx"].shape[1] <= 4096) else 0
        y, s0 = _RegionPoolFunction.apply(x, m, "max" if mode == "max" else "sum",
                                          lay["perm"], lay["offs"], lay["labels"], ident_len)
    else:
        src, wt = _pool_sources(x, m, mode)
        cdt = src.dtype
        y = _region_reduce(src, lay["idx"], lay["pad"], lay["ident"], lay["full"],
                           "max" if mode == "max" else "sum", cdt)
        s0 = _region_reduce(wt, lay["idx"], lay["pad"], lay["ident"], lay["full"],
                            "sum", cdt).expand(B, C, R)
    return _pool_finish(y, s0, mode, empty, x.dtype, return_weight)


def _layout_to(lay, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in lay.items()}


def region_pool(x, regions, mask=None, mode: str = "mean", fill: str = "zero",
                n_regions: Optional[int] = None, return_weight: bool = False,
                backend: str = "auto"):
    """Masked pooling of ``[B, C, N]`` HEALPix data over labelled regions.

    Parameters
    ----------
    x : torch.Tensor, shape [B, C, N]
    regions : array-like of int, shape [N]
        Region label of each cell (in the order of ``x``), in
        ``[0, n_regions)``; ``-1`` excludes the cell.  Labels of coarser
        parents: :meth:`HealPixRegionPool.from_cell_ids`.
    mask, mode, fill, return_weight, backend
        As :func:`global_pool`; a region without any valid cell gets
        ``fill``.
    n_regions : int, optional
        Number of regions R (default ``max(regions) + 1``).

    Returns
    -------
    y : torch.Tensor, shape [B, C, R]

    Notes
    -----
    No atomics, deterministic forward and backward.  Triton backend: one
    program per (region, block of rows) gathers the cells of the region
    from a CSR layout (int32), and the backward is a gather of the
    ``[B, C, R]`` gradient by the label of each cell.  PyTorch backend: the
    cells are gathered into a padded ``[B, C, R, L]`` tensor (L = size of
    the largest region; a free view when the regions are contiguous, sorted
    and of equal size, as ``from_cell_ids`` gives on the full sphere).  For
    many calls with the same regions, build a :class:`HealPixRegionPool`
    once.
    """
    lay = _region_layout(regions, n_regions)
    if lay["n"] != x.shape[-1]:
        raise ValueError(f"regions has {lay['n']} labels for {x.shape[-1]} cells.")
    return _region_pool_run(x, _layout_to(lay, x.device), mask, mode, fill,
                            return_weight, backend)


class HealPixRegionPool(nn.Module):
    """Masked pooling over fixed regions: ``[B, C, N]`` -> ``[B, C, R]``.

    The layouts of the regions are built once and stored as non-persistent
    buffers, so they follow ``.to()``.  See :func:`region_pool`.

    Parameters
    ----------
    regions : array-like of int, shape [N]
        Region label of each cell, ``-1`` to exclude a cell.
    mode : {"mean", "max", "sum"}, default "mean"
    fill : {"zero", "nan"}, default "zero"
    n_regions : int, optional
    backend : {"auto", "triton", "torch"}, default "auto"

    Attributes
    ----------
    n_regions : int
    counts : torch.Tensor, shape [R]
        Number of cells of each region.
    region_ids : numpy.ndarray or None
        With :meth:`from_cell_ids`, the parent cell id of each region.
    """

    _LAYOUT_TENSORS = ("idx", "pad", "perm", "offs", "labels")

    def __init__(self, regions, mode: str = "mean", fill: str = "zero",
                 n_regions: Optional[int] = None, backend: str = "auto"):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
        _empty_value(fill)
        lay = _region_layout(regions, n_regions)
        self.mode = mode
        self.fill = fill
        self.backend = backend
        self.n_regions = lay["idx"].shape[0]
        self.n_cells = lay["n"]
        self._ident = lay["ident"]
        self._full = lay["full"]
        self.region_ids = None
        for k in self._LAYOUT_TENSORS:
            self.register_buffer("_" + k, lay[k], persistent=False)
        self.register_buffer("counts", lay["counts"], persistent=False)

    @classmethod
    def from_cell_ids(cls, cell_ids, level: int, region_level: int, **kwargs):
        """Regions = the parent cells at ``region_level`` of the NESTED
        ``cell_ids`` (any order, full sphere or regional domain).  The
        regions are numbered in the increasing order of their parent id,
        stored in ``region_ids``."""
        if not 0 <= region_level <= level:
            raise ValueError("region_level must be in [0, level].")
        ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
        parents = ids >> (2 * (int(level) - int(region_level)))
        uniq, inv = np.unique(parents, return_inverse=True)
        obj = cls(inv.reshape(-1), n_regions=len(uniq), **kwargs)
        obj.region_ids = uniq
        return obj

    def extra_repr(self) -> str:  # pragma: no cover
        return f"n_regions={self.n_regions}, mode={self.mode!r}, fill={self.fill!r}"

    def forward(self, x, mask=None, return_weight: bool = False):
        if x.shape[-1] != self.n_cells:
            raise ValueError(f"Expected {self.n_cells} cells, got {x.shape[-1]}.")
        lay = {k: getattr(self, "_" + k) for k in self._LAYOUT_TENSORS}
        lay.update(ident=self._ident, full=self._full)
        return _region_pool_run(x, lay, mask, self.mode, self.fill, return_weight,
                                self.backend)
