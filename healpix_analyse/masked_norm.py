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
``"torch"``  (CPU, float64, and the reference everywhere)
    Plain PyTorch operations; autograd computes the gradients.
``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation)
    Fused kernels for ``HealPixGroupNorm``, ``HealPixBatchNorm`` (training)
    and ``HealPixLayerNorm``: the mask is read through broadcast strides
    (never expanded), the statistics are reduced in fixed-order partial sums
    (no atomics: deterministic), and the backward is written by hand from
    the saved per-row mean and 1/std.  A second derivative
    (``create_graph=True``) falls back to the PyTorch reference.

``backend="auto"`` (default) uses Triton when it can and PyTorch otherwise.
The pooling functions are plain PyTorch (one reduction each, deterministic).
"""

from __future__ import annotations

import math
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
# Triton kernels
# ===========================================================================

def triton_available() -> bool:
    """True when Triton is installed and a CUDA device is visible."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M: tl.constexpr):
        finite = (x == x) & (tl.abs(x) < float("inf"))
        if HAS_M:
            w = tl.load(m_ptr + b * smb + c * smc + offs, mask=inb, other=0).to(tl.float32)
            w = tl.where(w == w, w, 0.0)
            w = tl.minimum(tl.maximum(w, 0.0), 1.0)
        else:
            w = tl.full(x.shape, 1.0, tl.float32)
        w = tl.where(finite & inb, w, 0.0)
        return w

    @triton.jit
    def _rows_stats_kernel(
        x_ptr, m_ptr, part_ptr,
        C, N, CHUNK, NCH, smb, smc,
        HAS_M: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Partial (S0, mean, M2) of a chunk of one (b, c) row."""
        bc = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1)
        b = bc // C
        c = bc % C
        start = j * CHUNK
        row = x_ptr + bc * N
        ar = tl.arange(0, BLOCK)
        s0v = tl.zeros([BLOCK], tl.float32)
        s1v = tl.zeros([BLOCK], tl.float32)
        for k in range(0, CHUNK, BLOCK):
            offs = start + k + ar
            inb = (offs < N) & (k + ar < CHUNK)
            x = tl.load(row + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            x = tl.where(w > 0, x, 0.0)
            s0v += w
            s1v += w * x
        s0 = tl.sum(s0v, 0)
        mu = tl.sum(s1v, 0) / tl.maximum(s0, 1e-30)
        m2v = tl.zeros([BLOCK], tl.float32)
        for k in range(0, CHUNK, BLOCK):
            offs = start + k + ar
            inb = (offs < N) & (k + ar < CHUNK)
            x = tl.load(row + offs, mask=inb, other=0.0).to(tl.float32)
            w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
            d = tl.where(w > 0, x - mu, 0.0)
            m2v += w * d * d
        out = part_ptr + (bc * NCH + j) * 3
        tl.store(out, s0)
        tl.store(out + 1, mu)
        tl.store(out + 2, tl.sum(m2v, 0))

    @triton.jit
    def _rows_norm_fwd_kernel(
        x_ptr, m_ptr, y_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr,
        C, N, smb, smc, SB, CG,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        b = bc // C
        c = bc % C
        s = b * SB + c // CG
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inb = offs < N
        base = bc * N
        xr = tl.load(x_ptr + base + offs, mask=inb, other=0.0)
        x = xr.to(tl.float32)
        w = _load_w(x, m_ptr, b, c, offs, inb, smb, smc, HAS_M)
        mu = tl.load(mean_ptr + s)
        r = tl.load(rstd_ptr + s)
        y = (x - mu) * r
        if AFFINE:
            y = y * tl.load(g_ptr + c).to(tl.float32) + tl.load(b_ptr + c).to(tl.float32)
        if FILL == 0:
            y = tl.where(w > 0, y, 0.0)
        elif FILL == 1:
            y = tl.where(w > 0, y, float("nan"))
        else:
            y = tl.where(w > 0, y, x)
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
            xh = tl.where(w > 0, (x - mu) * r, 0.0)
            p1 += dy
            p2 += dy * xh
        out = part_ptr + (bc * NCH + j) * 2
        tl.store(out, tl.sum(p1, 0))
        tl.store(out + 1, tl.sum(p2, 0))

    @triton.jit
    def _rows_bwd_dx_kernel(
        x_ptr, m_ptr, dy_ptr, dx_ptr, mean_ptr, rstd_ptr, a_ptr, bb_ptr, g_ptr,
        C, N, smb, smc, SB, CG,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bc = tl.program_id(0).to(tl.int64)
        b = bc // C
        c = bc % C
        s = b * SB + c // CG
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
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

    @triton.jit
    def _ln_fwd_kernel(
        x_ptr, m_ptr, y_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr,
        C, N, smb, smc, eps,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """LayerNorm over the channels of each cell: one program per
        (sample, block of cells), loops over the channels."""
        b = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
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
            if FILL == 0:
                y = tl.where(w > 0, y, 0.0)
            elif FILL == 1:
                y = tl.where(w > 0, y, float("nan"))
            else:
                y = tl.where(w > 0, y, x)
            tl.store(y_ptr + base + c * N + offs, y.to(y_ptr.dtype.element_ty), mask=inb)

    @triton.jit
    def _ln_bwd_kernel(
        x_ptr, m_ptr, dy_ptr, dx_ptr, g_ptr, mean_ptr, rstd_ptr, pg_ptr, pb_ptr,
        C, N, NBLK, smb, smc,
        HAS_M: tl.constexpr, AFFINE: tl.constexpr, FILL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """dx of the per-cell LayerNorm, and partial sums of dgamma / dbeta
        per (program, channel) (no atomics)."""
        b = tl.program_id(0).to(tl.int64)
        pb = tl.program_id(1)
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


_BLOCK = 1024
_LN_BLOCK = 128


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
    """(CHUNK, NCH): enough programs to fill the GPU, fixed for a shape."""
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # pragma: no cover
        sms = 108
    target = 8 * sms
    nch = max(1, min(-(-target // max(rows, 1)), -(-N // _BLOCK)))
    chunk = -(-N // nch)
    chunk = -(-chunk // _BLOCK) * _BLOCK
    nch = -(-N // chunk)
    return chunk, nch


def _rows_layout(kind, B, C, G):
    """(SB, CG, number of stat rows) of the (b, c) -> stat-row map."""
    if kind == "group":
        return G, C // G, B * G
    return 0, 1, C  # batch


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


class _RowsNormFunction(torch.autograd.Function):
    """Triton group / batch normalisation (training statistics)."""

    @staticmethod
    def forward(ctx, x, m, weight, bias, kind, G, eps, fill):
        B, C, N = x.shape
        x_in = x
        x = x.contiguous()
        mt, smb, smc, has_m = _mask_args(m, x)
        SB, CG, R = _rows_layout(kind, B, C, G)
        chunk, nch = _chunks(B * C, N, x.device)
        parts = torch.empty(B * C, nch, 3, device=x.device, dtype=torch.float32)
        _rows_stats_kernel[(B * C, nch)](
            x, mt, parts, C, N, chunk, nch, smb, smc,
            HAS_M=has_m, BLOCK=_BLOCK, num_warps=4)
        s0, mu, var = _combine_parts(parts, kind, B, C, G)
        rstd = torch.rsqrt(var + eps)
        affine = weight is not None
        y = torch.empty_like(x)
        _rows_norm_fwd_kernel[(B * C, triton.cdiv(N, _BLOCK))](
            x, mt, y, mu, rstd,
            weight if affine else x, bias if affine else x,
            C, N, smb, smc, SB, CG,
            HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
            BLOCK=_BLOCK, num_warps=4)
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0), weight
                              if affine else torch.empty(0), mu, rstd, s0)
        ctx.meta = (kind, G, eps, fill, has_m, affine, bias is not None)
        ctx.mark_non_differentiable(mu, var, s0)
        return y, mu, var, s0

    @staticmethod
    def backward(ctx, dy, _dmu, _dvar, _ds0):
        x, m, weight, mu, rstd, s0 = ctx.saved_tensors
        kind, G, eps, fill, has_m, affine, has_bias = ctx.meta
        m = m if has_m else None
        w_in = weight if affine else None
        if torch.is_grad_enabled():  # double backward: reference with autograd
            return _reference_backward(ctx, x, m, w_in, dy, kind, G, eps, fill)
        B, C, N = x.shape
        x = x.contiguous()
        dy = dy.contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        SB, CG, R = _rows_layout(kind, B, C, G)
        chunk, nch = _chunks(B * C, N, x.device)
        parts = torch.empty(B * C, nch, 2, device=x.device, dtype=torch.float32)
        _rows_bwd_part_kernel[(B * C, nch)](
            x, mt, dy, mu, rstd, parts, C, N, chunk, nch, smb, smc, SB, CG,
            HAS_M=has_m, BLOCK=_BLOCK, num_warps=4)
        P = parts.sum(1).view(B, C, 2)
        P1, P2 = P[..., 0], P[..., 1]
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
        dx = torch.empty_like(x)
        _rows_bwd_dx_kernel[(B * C, triton.cdiv(N, _BLOCK))](
            x, mt, dy, dx, mu, rstd, A, Bv, weight if affine else x,
            C, N, smb, smc, SB, CG,
            HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
            BLOCK=_BLOCK, num_warps=4)
        dw = P2.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[2] else None
        db = P1.sum(0).to(weight.dtype) if has_bias and ctx.needs_input_grad[3] else None
        return dx, None, dw, db, None, None, None, None


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
        _ln_fwd_kernel[(B, triton.cdiv(N, _LN_BLOCK))](
            x, mt, y, weight if affine else x, bias if affine else x, mu, rstd,
            C, N, smb, smc, eps,
            HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
            BLOCK=_LN_BLOCK, num_warps=4)
        ctx.save_for_backward(x_in, m if m is not None else torch.empty(0),
                              weight if affine else torch.empty(0), mu, rstd)
        ctx.meta = (eps, fill, has_m, affine, bias is not None)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, m, weight, mu, rstd = ctx.saved_tensors
        eps, fill, has_m, affine, has_bias = ctx.meta
        m = m if has_m else None
        w_in = weight if affine else None
        if torch.is_grad_enabled():
            dx, _, dw, db, *_ = _reference_backward(
                ctx, x, m, w_in, dy, "layer", 1, eps, fill)
            return dx, None, dw, db, None, None
        B, C, N = x.shape
        x = x.contiguous()
        dy = dy.contiguous()
        mt, smb, smc, _ = _mask_args(m, x)
        nblk = triton.cdiv(N, _LN_BLOCK)
        dx = torch.empty_like(x)
        if affine:
            pg = torch.empty(B * nblk, C, device=x.device, dtype=torch.float32)
            pb = torch.empty_like(pg)
        else:
            pg = pb = mu
        _ln_bwd_kernel[(B, nblk)](
            x, mt, dy, dx, weight if affine else x, mu, rstd, pg, pb,
            C, N, nblk, smb, smc,
            HAS_M=has_m, AFFINE=affine, FILL=_FILL_CODE[fill],
            BLOCK=_LN_BLOCK, num_warps=4)
        dw = pg.sum(0).to(weight.dtype) if affine and ctx.needs_input_grad[2] else None
        db = pb.sum(0).to(weight.dtype) if has_bias and ctx.needs_input_grad[3] else None
        return dx, None, dw, db, None, None


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
    ok = _HAS_TRITON and x.is_cuda and x.dtype in _TRITON_DTYPES
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

    Forward
    -------
    ``forward(x, mask=None)`` with ``x`` of shape ``[B, C, N]`` and ``mask``
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
        y, *_ = masked_norm_reference(
            x, m, "group", self.num_groups, self.weight, self.bias, self.eps, self.fill)
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
        y, *_ = masked_norm_reference(
            x, m, "layer", 1, self.weight, self.bias, self.eps, self.fill)
        return y


class HealPixBatchNorm(_MaskedNormMixin, nn.BatchNorm1d):
    """Batch normalisation of ``[B, C, N]`` HEALPix data with a validity mask
    or a coverage, with running statistics.

    In training, the statistics of each channel are the weighted mean and
    (biased) variance over the samples and the cells; the running mean and
    variance are updated with the weighted mean and the unbiased variance
    ``var * S0 / (S0 - 1)``, ``S0`` being the sum of the weights of the
    channel (the number of valid cells for a binary mask).  A channel with
    ``S0 <= 1`` in a batch keeps its running statistics.  In evaluation the
    running statistics are used, unless ``track_running_stats=False`` (then
    the batch statistics, as ``torch.nn``).  With an all-valid mask
    the layer matches ``torch.nn.BatchNorm1d`` on ``[B, C, N]``: outputs,
    running statistics, ``momentum=None`` (cumulative average) and
    ``state_dict``.

    Parameters
    ----------
    num_features, eps, momentum, affine, track_running_stats, device, dtype
        As ``torch.nn.BatchNorm1d``.
    fill, backend
        As :class:`HealPixGroupNorm`.  The Triton kernels serve the training
        statistics; evaluation is an element-wise PyTorch expression.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: Optional[float] = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 fill: str = "zero", backend: str = "auto", device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats,
                         device=device, dtype=dtype)
        self._init_masked(fill, backend)

    def forward(self, x, mask=None):
        _check_x(x, self.num_features)
        m = _mask_3d(mask, x)
        C = self.num_features

        # Same decision as torch.nn.modules.batchnorm._BatchNorm.forward.
        if self.momentum is None:
            factor = 0.0
        else:
            factor = self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                factor = 1.0 / float(self.num_batches_tracked)
        use_batch = self.training or (self.running_mean is None and self.running_var is None)

        if not use_batch:
            y, *_ = masked_norm_reference(
                x, m, "batch", 1, self.weight, self.bias, self.eps, self.fill,
                mean=self.running_mean, var=self.running_var)
            return y

        if _use_triton(x, self.backend):
            y, mu, var, s0 = _RowsNormFunction.apply(
                x, m, self.weight, self.bias, "batch", 1, self.eps, self.fill)
        else:
            y, mu, var, s0 = masked_norm_reference(
                x, m, "batch", 1, self.weight, self.bias, self.eps, self.fill)

        if self.training and self.running_mean is not None:
            with torch.no_grad():
                rm = self.running_mean
                rv = self.running_var
                sdt = _compute_dtype(rm.dtype)
                mu = mu.reshape(C).to(sdt)
                var = var.reshape(C).to(sdt)
                s0 = s0.reshape(C).to(sdt)
                ok = s0 > 1
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
                return_weight: bool = False):
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

    Returns
    -------
    y : torch.Tensor, shape [B, C] (and the weight sum if requested)
    """
    _check_x(x)
    if mode not in POOL_MODES:
        raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
    empty = _empty_value(fill)
    B, C, N = x.shape
    cdt = _compute_dtype(x.dtype)
    w = _weights(x, _mask_3d(mask, x)).expand(B, C, N)
    valid = w > 0
    xc = x.to(cdt)
    s0 = w.sum(-1)
    has = s0 > 0
    if mode == "max":
        y = torch.where(valid, xc, torch.full_like(xc, -float("inf"))).amax(-1)
    else:
        xz = torch.where(valid, xc, torch.zeros_like(xc))
        y = (xz * w).sum(-1)
        if mode == "mean":
            y = y / s0.clamp_min(torch.finfo(cdt).tiny)
    y = torch.where(has, y, torch.full_like(y, empty)).to(x.dtype)
    return (y, s0) if return_weight else y


class HealPixGlobalPool(nn.Module):
    """Module form of :func:`global_pool`: ``[B, C, N]`` -> ``[B, C]``."""

    def __init__(self, mode: str = "mean", fill: str = "zero"):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
        _empty_value(fill)
        self.mode = mode
        self.fill = fill

    def extra_repr(self) -> str:  # pragma: no cover
        return f"mode={self.mode!r}, fill={self.fill!r}"

    def forward(self, x, mask=None):
        return global_pool(x, mask, self.mode, self.fill)


def _region_layout(regions, n_regions=None):
    """ELL layout of the regions: indices ``[R, L]`` (padded with 0) and the
    padding mask ``[R, L]``, L being the size of the largest region."""
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
    return idx, pad, counts, r.numel()


def _region_pool_layout(x, idx, pad, mask, mode, fill, return_weight):
    _check_x(x)
    if mode not in POOL_MODES:
        raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
    empty = _empty_value(fill)
    B, C, N = x.shape
    R, L = idx.shape
    cdt = _compute_dtype(x.dtype)
    w = _weights(x, _mask_3d(mask, x))
    xc = x.to(cdt)
    flat = idx.reshape(-1)
    wg = w.index_select(-1, flat).view(w.shape[0], w.shape[1], R, L) * pad.to(cdt)
    xg = xc.index_select(-1, flat).view(B, C, R, L)
    valid = wg > 0
    s0 = wg.sum(-1).expand(B, C, R)
    has = s0 > 0
    if mode == "max":
        y = torch.where(valid, xg, torch.full_like(xg, -float("inf"))).amax(-1)
    else:
        xz = torch.where(valid, xg, torch.zeros_like(xg))
        y = (xz * wg).sum(-1)
        if mode == "mean":
            y = y / s0.clamp_min(torch.finfo(cdt).tiny)
    y = torch.where(has, y, torch.full_like(y, empty)).to(x.dtype)
    return (y, s0) if return_weight else y


def region_pool(x, regions, mask=None, mode: str = "mean", fill: str = "zero",
                n_regions: Optional[int] = None, return_weight: bool = False):
    """Masked pooling of ``[B, C, N]`` HEALPix data over labelled regions.

    Parameters
    ----------
    x : torch.Tensor, shape [B, C, N]
    regions : array-like of int, shape [N]
        Region label of each cell (in the order of ``x``), in
        ``[0, n_regions)``; ``-1`` excludes the cell.  Labels of coarser
        parents: :meth:`HealPixRegionPool.from_cell_ids`.
    mask, mode, fill, return_weight
        As :func:`global_pool`; a region without any valid cell gets
        ``fill``.
    n_regions : int, optional
        Number of regions R (default ``max(regions) + 1``).

    Returns
    -------
    y : torch.Tensor, shape [B, C, R]

    Notes
    -----
    The cells of each region are gathered into a padded ``[B, C, R, L]``
    tensor, L being the size of the largest region, then reduced: no
    atomics, deterministic forward and backward.  For many calls with the
    same regions, build a :class:`HealPixRegionPool` once.
    """
    idx, pad, _, n = _region_layout(regions, n_regions)
    if n != x.shape[-1]:
        raise ValueError(f"regions has {n} labels for {x.shape[-1]} cells.")
    return _region_pool_layout(x, idx.to(x.device), pad.to(x.device), mask, mode,
                               fill, return_weight)


class HealPixRegionPool(nn.Module):
    """Masked pooling over fixed regions: ``[B, C, N]`` -> ``[B, C, R]``.

    The layout of the regions is built once and stored as non-persistent
    buffers, so it follows ``.to()``.  See :func:`region_pool`.

    Parameters
    ----------
    regions : array-like of int, shape [N]
        Region label of each cell, ``-1`` to exclude a cell.
    mode : {"mean", "max", "sum"}, default "mean"
    fill : {"zero", "nan"}, default "zero"
    n_regions : int, optional

    Attributes
    ----------
    n_regions : int
    counts : torch.Tensor, shape [R]
        Number of cells of each region.
    region_ids : numpy.ndarray or None
        With :meth:`from_cell_ids`, the parent cell id of each region.
    """

    def __init__(self, regions, mode: str = "mean", fill: str = "zero",
                 n_regions: Optional[int] = None):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError(f"mode must be one of {POOL_MODES}; got {mode!r}.")
        _empty_value(fill)
        idx, pad, counts, n = _region_layout(regions, n_regions)
        self.mode = mode
        self.fill = fill
        self.n_regions = idx.shape[0]
        self.n_cells = n
        self.region_ids = None
        self.register_buffer("_idx", idx, persistent=False)
        self.register_buffer("_pad", pad, persistent=False)
        self.register_buffer("counts", counts, persistent=False)

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
        return _region_pool_layout(x, self._idx, self._pad, mask, self.mode,
                                   self.fill, return_weight)
