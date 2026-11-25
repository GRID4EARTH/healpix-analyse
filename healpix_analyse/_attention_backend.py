"""Compute backends of :class:`healpix_analyse.attention.HealPixNeighbourhoodAttention`.

Each cell attends to the P taps of its gauge-oriented stencil, the stencil of
:class:`~healpix_analyse.convol.HealPixConv`.  The key and the value of a tap
are the bilinear interpolations of the keys and values of 4 cells::

    k_p[g, n] = sum_j w[p, j, g, n] k[idx[p, j, g, n]]        (same for v)
    s_p       = <q[n], k_p[g, n]> / sqrt(D) + bias[h, p]      (-inf for a masked tap)
    out[g, n] = sum_p softmax_p(s)_p v_p[g, n]

``idx`` (int32, positions in the input order) and ``w`` are the tap geometry
``[P, 4, G, N]`` of :func:`healpix_analyse._conv_backend.tap_geometry`, and
``valid`` (``[P, G, N]`` bool, or None) excludes from the softmax the taps that
fall outside a regional domain.

Shapes: ``q, k, v [B, H, N, D]`` (shared by the G gauges), ``bias [H, P]``,
output ``[B, G, H, N, D]``.

Neither backend materialises the gathered ``[B, H, G, N, P, D]`` keys and
values.

``"torch"`` (CPU, float64, and the reference everywhere)
    A loop over the taps (one gathered tap at a time) with a hand-written
    backward: the scores are recomputed, ``dq`` is a gather, ``dk`` and ``dv``
    are scattered with ``index_add_``.  The backward is written with
    differentiable operations, so a double backward works.

``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation)
    Forward: one program per (block of cells, batch, gauge, head) gathers the
    4 neighbours of each tap in registers and runs an online softmax over the
    taps (flash-attention style); it stores the output and the log-sum-exp.
    Backward, deterministic, no atomics: a first kernel recomputes the scores
    and writes ``dq`` (summed over the gauges in registers) together with the
    probabilities ``a`` and the score gradients ``ds`` (``[B, G, H, P, N]``,
    P values per cell and head: the size of the input for P = D); a second
    kernel gathers ``dk`` and ``dv`` for each source cell from the transposed
    stencil map (:func:`~healpix_analyse._conv_backend.build_transposed_map`).
    The bias gradient is a PyTorch reduction of ``ds``.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from healpix_analyse._conv_backend import (
    BACKENDS,
    _HAS_TRITON,
    build_transposed_map,
    triton_available,
)

if _HAS_TRITON:
    import triton
    import triton.language as tl

__all__ = [
    "BACKENDS",
    "triton_available",
    "neighbourhood_attention",
    "neighbourhood_attention_naive",
    "NeighbourhoodAttentionFunction",
]

_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# ===========================================================================
# Reference: the gathered keys and values materialised (autograd)
# ===========================================================================

def _gathered(t, idx, w):
    """t [B, H, N, D] -> [B, H, G, N, P, D]: the bilinear taps, materialised."""
    P, _, G, N = idx.shape
    B, H, _, D = t.shape
    acc = None
    for j in range(4):
        sel = t.index_select(2, idx[:, j].permute(1, 2, 0).reshape(-1).long())
        sel = sel.view(B, H, G, N, P, D) * w[:, j].permute(1, 2, 0).to(t.dtype)[..., None]
        acc = sel if acc is None else acc + sel
    return acc


def neighbourhood_attention_naive(q, k, v, bias, idx, w, valid=None):
    """Neighbourhood attention with the gathered taps materialised.

    The straightforward implementation (``[B, H, G, N, P, D]`` keys and
    values, then a softmax over the taps), differentiated by autograd.  Used
    as a reference in the tests and as the baseline of the benchmark.
    Returns ``[B, G, H, N, D]``.
    """
    D = q.shape[-1]
    kt = _gathered(k, idx, w)
    vt = _gathered(v, idx, w)
    s = torch.einsum("bhnd,bhgnpd->bhgnp", q, kt) / math.sqrt(D)
    if bias is not None:
        s = s + bias.to(s.dtype)[:, None, None, :].unsqueeze(0)
    if valid is not None:
        s = s.masked_fill(~valid.bool().permute(1, 2, 0)[None, None], float("-inf"))
    a = torch.softmax(s, dim=-1)
    out = torch.einsum("bhgnp,bhgnpd->bhgnd", a, vt)
    return out.permute(0, 2, 1, 3, 4).contiguous()


# ===========================================================================
# PyTorch backend: loop over the taps, hand-written backward
# ===========================================================================

def _cells_first(t):
    """[B, H, N, D] -> [N, B*H, D] (contiguous): the gathers read whole rows."""
    B, H, N, D = t.shape
    return t.permute(2, 0, 1, 3).reshape(N, B * H, D)


def _gather_tap(t, idx_p, w_p):
    """t [N, BH, D]; idx_p, w_p [4, G, N] -> [G, N, BH, D] = sum_j w * t[idx].

    The 4 neighbours are gathered by one call (int64 indices: index_select is
    much slower with int32 on CPU), then weighted and summed.
    """
    _, BH, D = t.shape
    G, N = idx_p.shape[1], idx_p.shape[2]
    sel = t.index_select(0, idx_p.reshape(-1).long()).view(4, G * N, BH, D)
    wj = w_p.to(t.dtype).reshape(4, G * N, 1, 1)
    if torch.is_grad_enabled():
        return (sel * wj).sum(0).view(G, N, BH, D)
    return sel.mul_(wj).sum(0).view(G, N, BH, D)


def _scatter_tap(dst, idx_p, w_p, h):
    """dst [N, BH, D] + the scatter of sum_j w_p[j] * h (h [G, N, BH, D])."""
    _, BH, D = dst.shape
    inplace = not torch.is_grad_enabled()
    for j in range(4):
        src = (h * w_p[j].to(h.dtype).view(*w_p[j].shape, 1, 1)).reshape(-1, BH, D)
        idx = idx_p[j].reshape(-1).long()
        dst = dst.index_add_(0, idx, src) if inplace else dst.index_add(0, idx, src)
    return dst


def _torch_scores(qt, kt, bias, idx, w, valid, scale, B):
    """Scores [G, N, BH, P] (cells-first inputs), masked taps at -inf."""
    cols = [(qt * _gather_tap(kt, idx[p], w[p])).sum(-1) for p in range(idx.shape[0])]
    s = torch.stack(cols, dim=-1) * scale
    if bias is not None:
        s = s + bias.to(s.dtype).repeat(B, 1)                     # [BH, P]
    if valid is not None:
        s = s.masked_fill(~valid.bool().permute(1, 2, 0)[:, :, None, :], float("-inf"))
    return s


def _torch_forward(q, k, v, bias, idx, w, valid, scale):
    """Returns out [B, G, H, N, D] and the probabilities a [G, N, BH, P]."""
    B, H, N, D = q.shape
    G = idx.shape[2]
    qt, kt, vt = _cells_first(q), _cells_first(k), _cells_first(v)
    a = torch.softmax(_torch_scores(qt, kt, bias, idx, w, valid, scale, B), dim=-1)
    out = None
    for p in range(idx.shape[0]):
        vp = _gather_tap(vt, idx[p], w[p])
        if out is None:
            out = vp.mul_(a[..., p, None]) if not torch.is_grad_enabled() else vp * a[..., p, None]
        else:
            out = out.addcmul(a[..., p, None], vp)
    out = out.view(G, N, B, H, D).permute(2, 0, 3, 1, 4).contiguous()
    return out, a


def _torch_backward(q, k, v, bias, idx, w, valid, scale, out, a, gout, needs):
    """Gradients of q, k, v, bias.  ``a`` None: recomputed (double backward)."""
    need_q, need_k, need_v, need_b = needs
    B, H, N, D = q.shape
    G = idx.shape[2]
    qt, kt, vt = _cells_first(q), _cells_first(k), _cells_first(v)
    if a is None:
        a = torch.softmax(_torch_scores(qt, kt, bias, idx, w, valid, scale, B), dim=-1)
    go = gout.to(q.dtype).permute(1, 3, 0, 2, 4).reshape(G, N, B * H, D)
    o = out.permute(1, 3, 0, 2, 4).reshape(G, N, B * H, D)
    delta = (go * o).sum(-1)                                    # [G, N, BH]
    dq = torch.zeros_like(qt) if need_q else None
    dk = torch.zeros_like(kt) if need_k else None
    dv = torch.zeros_like(vt) if need_v else None
    ds_cols = []
    for p in range(idx.shape[0]):
        a_p = a[..., p]
        ds_p = a_p * ((go * _gather_tap(vt, idx[p], w[p])).sum(-1) - delta)
        ds_cols.append(ds_p)
        if need_q:
            dq = dq + (ds_p[..., None] * _gather_tap(kt, idx[p], w[p])).sum(0) * scale
        if need_k:
            dk = _scatter_tap(dk, idx[p], w[p], ds_p[..., None] * qt * scale)
        if need_v:
            dv = _scatter_tap(dv, idx[p], w[p], a_p[..., None] * go)

    def back(t):
        return None if t is None else t.view(N, B, H, D).permute(1, 2, 0, 3)

    db = None
    if need_b:
        db = torch.stack(ds_cols, dim=-1).view(G, N, B, H, -1).sum(dim=(0, 1, 2)).to(bias.dtype)
    return back(dq), back(dk), back(dv), db


# ===========================================================================
# Triton backend
# ===========================================================================

def _block_d(D):
    return max(16, 1 << max(0, int(D) - 1).bit_length())


if _HAS_TRITON:

    def _configs(spec):
        if os.environ.get("TRITON_INTERPRET") == "1":   # the interpreter: one config
            spec = spec[:1]
        return [triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=1) for bm, nw in spec]

    _NA_CONFIGS = _configs(((32, 4), (32, 2), (64, 4), (64, 8), (128, 8)))

    @triton.jit
    def _tap_kv(k_base, v_base, idx_ptr, w_ptr, p, g, n, n_ok, d, d_ok, G, N, D,
                NEED_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """Bilinear key (and value) of tap p for the cells n: [BLOCK_M, BLOCK_D] fp32."""
        kt = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        vt = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        md = n_ok[:, None] & d_ok[None, :]
        for j in tl.static_range(4):
            off = ((p * 4 + j) * G + g).to(tl.int64) * N + n
            src = tl.load(idx_ptr + off, mask=n_ok, other=0).to(tl.int64)
            ww = tl.load(w_ptr + off, mask=n_ok, other=0.0).to(tl.float32)
            rows = src[:, None] * D + d[None, :]
            kt += ww[:, None] * tl.load(k_base + rows, mask=md, other=0.0).to(tl.float32)
            if NEED_V:
                vt += ww[:, None] * tl.load(v_base + rows, mask=md, other=0.0).to(tl.float32)
        return kt, vt

    @triton.jit
    def _tap_score(qv, kt, bias_ptr, valid_ptr, h, p, g, n, n_ok, G, N, P, scale,
                   HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr):
        s = tl.sum(qv * kt, axis=1) * scale
        if HAS_BIAS:
            s += tl.load(bias_ptr + h * P + p).to(tl.float32)
        if HAS_VALID:
            ok = tl.load(valid_ptr + (p * G + g).to(tl.int64) * N + n, mask=n_ok, other=1)
            s = tl.where(ok != 0, s, float("-inf"))
        return s

    @triton.autotune(configs=_NA_CONFIGS, key=["D", "P", "G", "SIZE"])
    @triton.jit
    def _na_fwd_kernel(q_ptr, k_ptr, v_ptr, idx_ptr, w_ptr, bias_ptr, valid_ptr,
                       o_ptr, lse_ptr, B, H, G, N, D, P, scale, SIZE,
                       HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """o[b, g, h, n] = sum_p softmax_p(s) v_p; lse[b, g, h, n] = logsumexp_p(s)."""
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)           # head fastest, then batch: the same idx / w
        h = pid % H                      # tiles are reused from L2
        rest = pid // H
        b = rest % B
        g = rest // B
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        md = n_ok[:, None] & d_ok[None, :]
        bh = (b * H + h).to(tl.int64) * N * D
        qv = tl.load(q_ptr + bh + n[:, None] * D + d[None, :], mask=md, other=0.0).to(tl.float32)
        m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)   # finite: no inf - inf
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for p in range(0, P):
            kt, vt = _tap_kv(k_ptr + bh, v_ptr + bh, idx_ptr, w_ptr, p, g, n, n_ok, d, d_ok,
                             G, N, D, True, BLOCK_M, BLOCK_D)
            s = _tap_score(qv, kt, bias_ptr, valid_ptr, h, p, g, n, n_ok, G, N, P, scale,
                           HAS_BIAS, HAS_VALID)
            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            e = tl.exp(s - m_new)
            l_i = l_i * alpha + e
            acc = acc * alpha[:, None] + e[:, None] * vt
            m_i = m_new
        acc = acc / l_i[:, None]
        bgh = ((b * G + g) * H + h).to(tl.int64) * N
        tl.store(o_ptr + (bgh + n)[:, None] * D + d[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=md)
        tl.store(lse_ptr + bgh + n, m_i + tl.log(l_i), mask=n_ok)

    @triton.autotune(configs=_NA_CONFIGS, key=["D", "P", "G", "SIZE"])
    @triton.jit
    def _na_bwd_q_kernel(q_ptr, k_ptr, v_ptr, idx_ptr, w_ptr, bias_ptr, valid_ptr,
                         go_ptr, lse_ptr, delta_ptr, dq_ptr, a_ptr, ds_ptr,
                         B, H, G, N, D, P, scale, SIZE,
                         HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """dq (summed over the gauges in registers), a and ds [B, G, H, P, N]."""
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)
        h = pid % H
        b = pid // H
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        md = n_ok[:, None] & d_ok[None, :]
        bh = (b * H + h).to(tl.int64) * N * D
        qv = tl.load(q_ptr + bh + n[:, None] * D + d[None, :], mask=md, other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bgh = ((b * G + g) * H + h).to(tl.int64)
            go = tl.load(go_ptr + (bgh * N + n)[:, None] * D + d[None, :],
                         mask=md, other=0.0).to(tl.float32)
            lse = tl.load(lse_ptr + bgh * N + n, mask=n_ok, other=0.0)
            delta = tl.load(delta_ptr + bgh * N + n, mask=n_ok, other=0.0)
            for p in range(0, P):
                kt, vt = _tap_kv(k_ptr + bh, v_ptr + bh, idx_ptr, w_ptr, p, g, n, n_ok, d, d_ok,
                                 G, N, D, True, BLOCK_M, BLOCK_D)
                s = _tap_score(qv, kt, bias_ptr, valid_ptr, h, p, g, n, n_ok, G, N, P, scale,
                               HAS_BIAS, HAS_VALID)
                a = tl.exp(s - lse)
                ds = a * (tl.sum(go * vt, axis=1) - delta)
                dq += ds[:, None] * kt
                row = (bgh * P + p) * N + n
                tl.store(a_ptr + row, a, mask=n_ok)
                tl.store(ds_ptr + row, ds, mask=n_ok)
        tl.store(dq_ptr + bh + n[:, None] * D + d[None, :],
                 (dq * scale).to(dq_ptr.dtype.element_ty), mask=md)

    @triton.autotune(configs=_NA_CONFIGS, key=["D", "P", "G", "SIZE"])
    @triton.jit
    def _na_bwd_kv_kernel(q_ptr, go_ptr, a_ptr, ds_ptr, tptr_ptr, tbase_ptr, tk_ptr, tw_ptr,
                          dk_ptr, dv_ptr, B, H, G, N, D, P, scale, SIZE,
                          BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """dk[s] = scale sum tw ds q[t], dv[s] = sum tw a go[t] over the entries of s.

        One program per (block of source cells, batch, head); the entries of
        each source cell come from the transposed map, in a fixed order: no
        atomics.
        """
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)
        h = pid % H
        b = pid // H
        s_ = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        s_ok = s_ < N
        s64 = s_.to(tl.int64)
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        bh = (b * H + h).to(tl.int64) * N * D
        GN1 = (G * N + 1).to(tl.int64)       # row stride of ptr [P, G*N + 1]
        dk = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bgh = ((b * G + g) * H + h).to(tl.int64)
            for p in range(0, P):
                base = tl.load(tbase_ptr + p)
                row = s64 + g * N + p * GN1
                start = tl.load(tptr_ptr + row, mask=s_ok, other=0)
                cnt = tl.load(tptr_ptr + row + 1, mask=s_ok, other=0) - start
                n_max = tl.max(cnt, axis=0)
                for jj in range(0, n_max):
                    m = s_ok & (jj < cnt)
                    e = base + start.to(tl.int64) + jj
                    t = tl.load(tk_ptr + e, mask=m, other=0).to(tl.int64)
                    ww = tl.load(tw_ptr + e, mask=m, other=0.0).to(tl.float32)
                    pr = (bgh * P + p) * N + t
                    av = tl.load(a_ptr + pr, mask=m, other=0.0)
                    dsv = tl.load(ds_ptr + pr, mask=m, other=0.0)
                    mm = m[:, None] & d_ok[None, :]
                    qv = tl.load(q_ptr + bh + t[:, None] * D + d[None, :], mask=mm, other=0.0)
                    gov = tl.load(go_ptr + (bgh * N + t)[:, None] * D + d[None, :],
                                  mask=mm, other=0.0)
                    dk += (ww * dsv)[:, None] * qv.to(tl.float32)
                    dv += (ww * av)[:, None] * gov.to(tl.float32)
        md = s_ok[:, None] & d_ok[None, :]
        out = bh + s64[:, None] * D + d[None, :]
        tl.store(dk_ptr + out, (dk * scale).to(dk_ptr.dtype.element_ty), mask=md)
        tl.store(dv_ptr + out, dv.to(dv_ptr.dtype.element_ty), mask=md)


def _size_class(n):
    return int(math.log2(max(int(n), 1))) // 2


def _device_ctx(device):
    # CPU tensors reach the kernels only under the Triton interpreter (tests)
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def _triton_forward(q, k, v, bias, idx, w, valid, scale):
    B, H, N, D = q.shape
    P, _, G, _ = idx.shape
    out = torch.empty(B, G, H, N, D, device=q.device, dtype=q.dtype)
    lse = torch.empty(B, G, H, N, device=q.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), B * G * H)

    with _device_ctx(q.device):
        _na_fwd_kernel[grid](
            q, k, v, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx, out, lse,
            B, H, G, N, D, P, scale, _size_class(B * N),
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None,
            BLOCK_D=_block_d(D),
        )
    return out, lse


def _triton_backward(q, k, v, bias, idx, w, valid, scale, out, lse, gout, transposed):
    B, H, N, D = q.shape
    P, _, G, _ = idx.shape
    gout = gout.to(q.dtype).contiguous()
    delta = (gout.float() * out.float()).sum(-1).contiguous()          # [B, G, H, N]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    a = torch.empty(B, G, H, P, N, device=q.device, dtype=torch.float32)
    ds = torch.empty_like(a)
    tptr, tbase, tk, tw = transposed() if transposed is not None \
        else build_transposed_map(idx, w, N)
    size = _size_class(B * N)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    with _device_ctx(q.device):
        _na_bwd_q_kernel[grid](
            q, k, v, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx,
            gout, lse, delta, dq, a, ds,
            B, H, G, N, D, P, scale, size,
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None,
            BLOCK_D=_block_d(D),
        )
        _na_bwd_kv_kernel[grid](
            q, gout, a, ds, tptr, tbase, tk, tw, dk, dv,
            B, H, G, N, D, P, scale, size, BLOCK_D=_block_d(D),
        )
    db = ds.sum(dim=(0, 1, 4)).to(bias.dtype) if bias is not None else None   # [H, P]
    return dq, dk, dv, db


# ===========================================================================
# One autograd Function, two backends
# ===========================================================================

@dataclass(frozen=True)
class _Options:
    use_triton: bool
    scale: float
    transposed: Optional[Callable[[], tuple]]


class NeighbourhoodAttentionFunction(torch.autograd.Function):
    """``q, k, v [B, H, N, D] -> [B, G, H, N, D]``; dispatches to Triton or PyTorch."""

    @staticmethod
    def forward(ctx, q, k, v, bias, idx, w, valid, opts):
        if opts.use_triton:
            out, stat = _triton_forward(q, k, v, bias, idx, w, valid, opts.scale)
        else:
            out, stat = _torch_forward(q, k, v, bias, idx, w, valid, opts.scale)
        ctx.opts = opts
        ctx.save_for_backward(q, k, v, bias, idx, w, valid, out, stat)
        return out

    @staticmethod
    def backward(ctx, gout):
        q, k, v, bias, idx, w, valid, out, stat = ctx.saved_tensors
        opts = ctx.opts
        needs = ctx.needs_input_grad[:4]
        if opts.use_triton and not torch.is_grad_enabled():
            dq, dk, dv, db = _triton_backward(q, k, v, bias, idx, w, valid, opts.scale,
                                              out, stat, gout, opts.transposed)
            dq = dq if needs[0] else None
            dk = dk if needs[1] else None
            dv = dv if needs[2] else None
            db = db if needs[3] else None
        else:
            # Grad mode is on in a backward only with create_graph=True (double
            # backward): the probabilities are recomputed with operations that
            # autograd records, and the output is recomputed so that its
            # dependence on q, k, v is part of the graph.
            if torch.is_grad_enabled():
                out, _ = _torch_forward(q, k, v, bias, idx, w, valid, opts.scale)
                a = None
            else:
                a = stat if not opts.use_triton else None
            dq, dk, dv, db = _torch_backward(q, k, v, bias, idx, w, valid, opts.scale,
                                             out, a, gout, needs)
        return dq, dk, dv, db, None, None, None, None


def _autocast_dtype():
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:  # torch < 2.4
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None


def _use_triton(x, backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == "torch":
        return False
    ok = _HAS_TRITON and x.is_cuda and x.dtype in _TRITON_DTYPES
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, CUDA tensors and float32, float16 "
            "or bfloat16 data (float64 runs on backend='torch')."
        )
    return ok


def neighbourhood_attention(q, k, v, bias, idx, w, valid=None, *, backend="auto",
                            scale=None, transposed=None):
    """Neighbourhood attention on precomputed tap geometry.

    Parameters
    ----------
    q, k, v : Tensor [B, H, N, D]
    bias : Tensor [H, P] or None
        Relative position bias, one value per head and tap.
    idx, w : Tensor [P, 4, G, N]
        Tap geometry; ``idx`` int32 on the device of ``q``.
    valid : BoolTensor [P, G, N] or None
        Taps that take part in the softmax (None: all).  Every cell needs at
        least one valid tap.
    backend : {"auto", "triton", "torch"}
    scale : float or None
        Score scale, ``1 / sqrt(D)`` by default.
    transposed : callable or None
        Returns the transposed map of ``idx, w`` (cached by the caller); used
        by the Triton backward only.

    Returns
    -------
    Tensor [B, G, H, N, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    dt = _autocast_dtype() if q.is_cuda else None
    if dt is not None and q.dtype == torch.float32:
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
    use_triton = _use_triton(q, backend)
    if use_triton:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if valid is not None:
            valid = valid.to(torch.int8).contiguous()
        if bias is not None:
            bias = bias.contiguous()
    elif valid is not None:
        valid = valid.bool()
    if not use_triton and bias is not None and bias.dtype != q.dtype:
        bias = bias.to(q.dtype)
    opts = _Options(use_triton, float(scale), transposed)
    return NeighbourhoodAttentionFunction.apply(q, k, v, bias, idx, w, valid, opts)
