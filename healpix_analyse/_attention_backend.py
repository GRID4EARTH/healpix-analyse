"""Compute backends of the attention layers of :mod:`healpix_analyse.attention`.

Neighbourhood attention (:class:`~healpix_analyse.attention.HealPixNeighbourhoodAttention`)
---------------------------------------------------------------------------------------

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

Both backends read the packed projection ``qkv [B, N, 3, H, D]``; the Triton
kernels read it with strides and write its gradient directly, so no permuted
copy of q, k, v (or of their gradients) is made.

Window attention (:class:`~healpix_analyse.attention.HealPixWindowAttention`)
-------------------------------------------------------------------------
``"triton"``: flash-attention kernels on gathered windows.  Forward: one
program per (window, block of queries, batch, head) gathers its query rows
and loops over the key blocks of the window (``tl.dot`` for q k^T and p v,
online softmax); padding keys are masked in registers, so a padded or
overlapping window costs no gathered copy and no mask tensor.  Backward,
deterministic, no atomics: ``dq`` per block of queries; ``dk``, ``dv`` per
block of keys, written to the cells directly when each cell is the key of
one window slot, else to per-slot partial sums reduced per cell in a fixed
order (CSR).  ``"torch"``: gather, ``scaled_dot_product_attention``, scatter.
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
    "neighbourhood_attention_qkv",
    "WindowPlan",
    "window_attention",
    "window_attention_torch",
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
    """dst [N, BH, D] + the scatter of sum_j w_p[j] * h (h [G, N, BH, D]), one index_add."""
    _, BH, D = dst.shape
    src = (h.unsqueeze(0) * w_p.to(h.dtype)[..., None, None]).reshape(-1, BH, D)
    idx = idx_p.reshape(-1).long()
    if torch.is_grad_enabled():
        return dst.index_add(0, idx, src)
    return dst.index_add_(0, idx, src)


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
# Triton kernels
# ===========================================================================
#
# Layouts (no permute copy on the way in or out):
#   qkv   [B, N, 3, H, D]  the output of the qkv projection, read with strides
#                          (batch stride SB, cell stride SN, in elements);
#   dqkv  same layout: the kernels write the gradient of the projection output.
#   neighbourhood: o [B, G, N, H, D], lse [B, G, H, N] (fp32);
#   window:        o [B, N, H, D],    lse [B, N, H]    (fp32).

def _block_d(D):
    return max(16, 1 << max(0, int(D) - 1).bit_length())


def _precision(dtype):
    if dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return "ieee"
    return "tf32"


if _HAS_TRITON:

    _INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

    def _configs(spec):
        if _INTERPRET:                          # the interpreter: one config
            spec = spec[:1]
        return [triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=1) for bm, nw in spec]

    # a small autotune space (the likely winners): gather-bound kernels
    _NA_CONFIGS = _configs(((64, 4), (32, 4), (128, 8)))

    def _win_configs(spec):
        if _INTERPRET:
            spec = spec[:1]
        return [triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
                for bq, bk, nw, ns in spec]

    # a small autotune space (the likely winners of flash attention at D <= 64)
    _WIN_CONFIGS = _win_configs((
        (64, 64, 4, 2), (64, 32, 4, 2), (128, 64, 8, 2)))

    # -----------------------------------------------------------------------
    # Neighbourhood attention
    # -----------------------------------------------------------------------

    @triton.jit
    def _tap_kv(k_base, v_base, idx_ptr, w_ptr, p, g, n, n_ok, d, d_ok, G, N, SN,
                BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """Bilinear key and value of tap p for the cells n: [BLOCK_M, BLOCK_D] fp32."""
        kt = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        vt = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        md = n_ok[:, None] & d_ok[None, :]
        for j in tl.static_range(4):
            off = ((p * 4 + j) * G + g).to(tl.int64) * N + n
            src = tl.load(idx_ptr + off, mask=n_ok, other=0).to(tl.int64)
            ww = tl.load(w_ptr + off, mask=n_ok, other=0.0).to(tl.float32)
            rows = src[:, None] * SN + d[None, :]
            kt += ww[:, None] * tl.load(k_base + rows, mask=md, other=0.0).to(tl.float32)
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
    def _na_fwd_kernel(qkv_ptr, SB, SN, idx_ptr, w_ptr, bias_ptr, valid_ptr, o_ptr, lse_ptr,
                       B, H, G, N, D, scale, SIZE, P: tl.constexpr,
                       HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """o[b, g, n, h] = sum_p softmax_p(s) v_p; lse[b, g, h, n] = logsumexp_p(s)."""
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)           # head fastest, then batch: the same idx / w
        h = pid % H                      # tiles are reused from L2
        rest = pid // H
        b = rest % B
        g = rest // B
        C = H * D
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        md = n_ok[:, None] & d_ok[None, :]
        q_base = qkv_ptr + b.to(tl.int64) * SB + h * D
        qv = tl.load(q_base + n.to(tl.int64)[:, None] * SN + d[None, :], mask=md,
                     other=0.0).to(tl.float32)
        m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)   # finite: no inf - inf
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for p in tl.static_range(P):
            kt, vt = _tap_kv(q_base + C, q_base + 2 * C, idx_ptr, w_ptr, p, g, n, n_ok, d, d_ok,
                             G, N, SN, BLOCK_M, BLOCK_D)
            s = _tap_score(qv, kt, bias_ptr, valid_ptr, h, p, g, n, n_ok, G, N, P, scale,
                           HAS_BIAS, HAS_VALID)
            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            e = tl.exp(s - m_new)
            l_i = l_i * alpha + e
            acc = acc * alpha[:, None] + e[:, None] * vt
            m_i = m_new
        acc = acc / l_i[:, None]
        orow = ((b * G + g).to(tl.int64) * N + n) * C + h * D
        tl.store(o_ptr + orow[:, None] + d[None, :], acc.to(o_ptr.dtype.element_ty), mask=md)
        tl.store(lse_ptr + ((b * G + g) * H + h).to(tl.int64) * N + n, m_i + tl.log(l_i),
                 mask=n_ok)

    @triton.autotune(configs=_NA_CONFIGS, key=["D", "P", "G", "SIZE"])
    @triton.jit
    def _na_bwd_q_kernel(qkv_ptr, SB, SN, idx_ptr, w_ptr, bias_ptr, valid_ptr,
                         go_ptr, lse_ptr, delta_ptr, dqkv_ptr, a_ptr, ds_ptr,
                         B, H, G, N, D, scale, SIZE, P: tl.constexpr,
                         HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
        """dq (summed over the gauges in registers), a and ds [B, G, H, P, N]."""
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)
        h = pid % H
        b = pid // H
        C = H * D
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        md = n_ok[:, None] & d_ok[None, :]
        q_base = qkv_ptr + b.to(tl.int64) * SB + h * D
        nrow = n.to(tl.int64)[:, None] * SN + d[None, :]
        qv = tl.load(q_base + nrow, mask=md, other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bgh = ((b * G + g) * H + h).to(tl.int64)
            orow = ((b * G + g).to(tl.int64) * N + n) * C + h * D
            go = tl.load(go_ptr + orow[:, None] + d[None, :], mask=md, other=0.0).to(tl.float32)
            lse = tl.load(lse_ptr + bgh * N + n, mask=n_ok, other=0.0)
            delta = tl.load(delta_ptr + bgh * N + n, mask=n_ok, other=0.0)
            for p in tl.static_range(P):
                kt, vt = _tap_kv(q_base + C, q_base + 2 * C, idx_ptr, w_ptr, p, g, n, n_ok,
                                 d, d_ok, G, N, SN, BLOCK_M, BLOCK_D)
                s = _tap_score(qv, kt, bias_ptr, valid_ptr, h, p, g, n, n_ok, G, N, P, scale,
                               HAS_BIAS, HAS_VALID)
                a = tl.exp(s - lse)
                ds = a * (tl.sum(go * vt, axis=1) - delta)
                dq += ds[:, None] * kt
                row = (bgh * P + p) * N + n
                tl.store(a_ptr + row, a, mask=n_ok)
                tl.store(ds_ptr + row, ds, mask=n_ok)
        tl.store(dqkv_ptr + b.to(tl.int64) * SB + h * D + nrow,
                 (dq * scale).to(dqkv_ptr.dtype.element_ty), mask=md)

    @triton.autotune(configs=_NA_CONFIGS, key=["D", "P", "G", "SIZE"])
    @triton.jit
    def _na_bwd_kv_kernel(qkv_ptr, SB, SN, go_ptr, a_ptr, ds_ptr, tptr_ptr, tbase_ptr, tk_ptr,
                          tw_ptr, dqkv_ptr, B, H, G, N, D, scale, SIZE, P: tl.constexpr,
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
        C = H * D
        s_ = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        s_ok = s_ < N
        s64 = s_.to(tl.int64)
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        q_base = qkv_ptr + b.to(tl.int64) * SB + h * D
        GN1 = (G * N + 1).to(tl.int64)       # row stride of ptr [P, G*N + 1]
        dk = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bgh = ((b * G + g) * H + h).to(tl.int64)
            go_base = go_ptr + (b * G + g).to(tl.int64) * N * C + h * D
            for p in tl.static_range(P):
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
                    qv = tl.load(q_base + t[:, None] * SN + d[None, :], mask=mm, other=0.0)
                    gov = tl.load(go_base + t[:, None] * C + d[None, :], mask=mm, other=0.0)
                    dk += (ww * dsv)[:, None] * qv.to(tl.float32)
                    dv += (ww * av)[:, None] * gov.to(tl.float32)
        md = s_ok[:, None] & d_ok[None, :]
        out = dqkv_ptr + b.to(tl.int64) * SB + h * D + s64[:, None] * SN + d[None, :]
        tl.store(out + C, (dk * scale).to(dqkv_ptr.dtype.element_ty), mask=md)
        tl.store(out + 2 * C, dv.to(dqkv_ptr.dtype.element_ty), mask=md)

    # -----------------------------------------------------------------------
    # Window attention (flash-attention style, gathered windows)
    # -----------------------------------------------------------------------

    @triton.jit
    def _win_rows(idx_ptr, wi, slot, T, N, CONTIG: tl.constexpr):
        """Cell positions of the window slots (padding -> N) and their mask."""
        s_ok = slot < T
        if CONTIG:
            pos = (wi * T + slot).to(tl.int64)
        else:
            pos = tl.load(idx_ptr + wi.to(tl.int64) * T + slot, mask=s_ok, other=N).to(tl.int64)
        ok = s_ok & (pos < N)
        return tl.where(ok, pos, 0), ok

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D"])
    @triton.jit
    def _win_fwd_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, o_ptr, lse_ptr,
                        H, N, D, TQ, TK, scale,
                        Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr, PRECISION: tl.constexpr,
                        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        """o[b, n, h] = softmax(q k^T * scale) v over the keys of the window of n."""
        n_qb = tl.cdiv(TQ, BLOCK_Q)
        pid0 = tl.program_id(0)
        wi = pid0 // n_qb
        qb = pid0 % n_qb
        pid1 = tl.program_id(1)
        h = pid1 % H
        b = pid1 // H
        C = H * D
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        base = qkv_ptr + b.to(tl.int64) * SB + h * D
        qpos, q_ok = _win_rows(qidx_ptr, wi, qb * BLOCK_Q + tl.arange(0, BLOCK_Q), TQ, N, Q_CONTIG)
        q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=q_ok[:, None] & d_ok[None, :],
                    other=0.0)
        m_i = tl.full((BLOCK_Q,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, TK, BLOCK_K):
            kpos, k_ok = _win_rows(kvidx_ptr, wi, k0 + tl.arange(0, BLOCK_K), TK, N, KV_CONTIG)
            mk = k_ok[:, None] & d_ok[None, :]
            k = tl.load(base + C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            v = tl.load(base + 2 * C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
            s = tl.where(k_ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
            m_i = m_new
        acc = acc / l_i[:, None]
        rows = (b.to(tl.int64) * N + qpos)
        tl.store(o_ptr + rows[:, None] * C + h * D + d[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=q_ok[:, None] & d_ok[None, :])
        tl.store(lse_ptr + rows * H + h, m_i + tl.log(l_i), mask=q_ok)

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D"])
    @triton.jit
    def _win_bwd_q_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, go_ptr, lse_ptr, delta_ptr,
                          dqkv_ptr, H, N, D, TQ, TK, scale,
                          Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr,
                          PRECISION: tl.constexpr,
                          BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        """dq of the queries of one block of the window, written into dqkv."""
        n_qb = tl.cdiv(TQ, BLOCK_Q)
        pid0 = tl.program_id(0)
        wi = pid0 // n_qb
        qb = pid0 % n_qb
        pid1 = tl.program_id(1)
        h = pid1 % H
        b = pid1 // H
        C = H * D
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        base = qkv_ptr + b.to(tl.int64) * SB + h * D
        qpos, q_ok = _win_rows(qidx_ptr, wi, qb * BLOCK_Q + tl.arange(0, BLOCK_Q), TQ, N, Q_CONTIG)
        mq = q_ok[:, None] & d_ok[None, :]
        q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=mq, other=0.0)
        rows = b.to(tl.int64) * N + qpos
        go = tl.load(go_ptr + rows[:, None] * C + h * D + d[None, :], mask=mq, other=0.0)
        lse = tl.load(lse_ptr + rows * H + h, mask=q_ok, other=0.0)
        delta = tl.load(delta_ptr + rows * H + h, mask=q_ok, other=0.0)
        dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, TK, BLOCK_K):
            kpos, k_ok = _win_rows(kvidx_ptr, wi, k0 + tl.arange(0, BLOCK_K), TK, N, KV_CONTIG)
            mk = k_ok[:, None] & d_ok[None, :]
            k = tl.load(base + C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            v = tl.load(base + 2 * C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
            p = tl.where(k_ok[None, :], tl.exp(s - lse[:, None]), 0.0)
            dp = tl.dot(go, tl.trans(v), input_precision=PRECISION)
            ds = p * (dp - delta[:, None])
            dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=PRECISION)
        tl.store(dqkv_ptr + b.to(tl.int64) * SB + h * D + qpos[:, None] * SN + d[None, :],
                 (dq * scale).to(dqkv_ptr.dtype.element_ty), mask=mq)

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D"])
    @triton.jit
    def _win_bwd_kv_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, go_ptr, lse_ptr, delta_ptr,
                           dqkv_ptr, part_ptr, H, N, D, TQ, TK, NW, scale,
                           Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr,
                           DIRECT: tl.constexpr, PRECISION: tl.constexpr,
                           BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        """dk, dv of one block of keys of one window.

        ``DIRECT`` (every cell is the key of one window slot): written into
        dqkv.  Otherwise (overlapping windows): written into the partial
        buffer ``part [B, NW*TK, 2, H, D]`` (fp32), reduced per cell by
        :func:`_win_reduce_kernel` in a fixed order (no atomics).
        """
        n_kb = tl.cdiv(TK, BLOCK_K)
        pid0 = tl.program_id(0)
        wi = pid0 // n_kb
        kb = pid0 % n_kb
        pid1 = tl.program_id(1)
        h = pid1 % H
        b = pid1 // H
        C = H * D
        d = tl.arange(0, BLOCK_D)
        d_ok = d < D
        base = qkv_ptr + b.to(tl.int64) * SB + h * D
        kslot = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        kpos, k_ok = _win_rows(kvidx_ptr, wi, kslot, TK, N, KV_CONTIG)
        mk = k_ok[:, None] & d_ok[None, :]
        k = tl.load(base + C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
        v = tl.load(base + 2 * C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
        dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        for q0 in range(0, TQ, BLOCK_Q):
            qpos, q_ok = _win_rows(qidx_ptr, wi, q0 + tl.arange(0, BLOCK_Q), TQ, N, Q_CONTIG)
            mq = q_ok[:, None] & d_ok[None, :]
            q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=mq, other=0.0)
            rows = b.to(tl.int64) * N + qpos
            go = tl.load(go_ptr + rows[:, None] * C + h * D + d[None, :], mask=mq, other=0.0)
            lse = tl.load(lse_ptr + rows * H + h, mask=q_ok, other=0.0)
            delta = tl.load(delta_ptr + rows * H + h, mask=q_ok, other=0.0)
            sT = tl.dot(k, tl.trans(q), input_precision=PRECISION) * scale       # [BK, BQ]
            pT = tl.where(k_ok[:, None] & q_ok[None, :], tl.exp(sT - lse[None, :]), 0.0)
            dv = tl.dot(pT.to(go.dtype), go, dv, input_precision=PRECISION)
            dpT = tl.dot(v, tl.trans(go), input_precision=PRECISION)
            dsT = pT * (dpT - delta[None, :])
            dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=PRECISION)
        if DIRECT:
            out = dqkv_ptr + b.to(tl.int64) * SB + h * D + kpos[:, None] * SN + d[None, :]
            tl.store(out + C, (dk * scale).to(dqkv_ptr.dtype.element_ty), mask=mk)
            tl.store(out + 2 * C, dv.to(dqkv_ptr.dtype.element_ty), mask=mk)
        else:
            prow = (b.to(tl.int64) * NW + wi) * TK + kslot
            out = part_ptr + prow[:, None] * (2 * C) + h * D + d[None, :]
            ms = (kslot < TK)[:, None] & d_ok[None, :]
            tl.store(out, dk * scale, mask=ms)
            tl.store(out + C, dv, mask=ms)

    @triton.jit
    def _win_reduce_kernel(part_ptr, kptr_ptr, kent_ptr, dqkv_ptr, SB, SN, N, C2, R,
                           BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr):
        """dqkv[b, m, C:3C] = sum over the window slots of m of part[b, slot] (fixed order)."""
        pid = tl.program_id(0)
        b = tl.program_id(1)
        pf = tl.program_id(2)
        m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        m_ok = m < N
        f = pf * BLOCK_F + tl.arange(0, BLOCK_F)
        f_ok = f < C2
        start = tl.load(kptr_ptr + m, mask=m_ok, other=0)
        cnt = tl.load(kptr_ptr + m + 1, mask=m_ok, other=0) - start
        n_max = tl.max(cnt, axis=0)
        acc = tl.zeros((BLOCK_M, BLOCK_F), dtype=tl.float32)
        pb = part_ptr + b.to(tl.int64) * R * C2
        for jj in range(0, n_max):
            ok = m_ok & (jj < cnt)
            e = tl.load(kent_ptr + start + jj, mask=ok, other=0).to(tl.int64)
            acc += tl.load(pb + e[:, None] * C2 + f[None, :], mask=ok[:, None] & f_ok[None, :],
                           other=0.0)
        out = dqkv_ptr + b.to(tl.int64) * SB + m.to(tl.int64)[:, None] * SN + (C2 // 2) + f[None, :]
        tl.store(out, acc.to(dqkv_ptr.dtype.element_ty), mask=m_ok[:, None] & f_ok[None, :])


def _size_class(n):
    return int(math.log2(max(int(n), 1))) // 2


def _device_ctx(device):
    # CPU tensors reach the kernels only under the Triton interpreter (tests)
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def _qkv_strides(qkv):
    """qkv [B, N, 3, H, D] with unit strides over (3, H, D): batch and cell strides."""
    B, N, three, H, D = qkv.shape
    if qkv.stride(4) != 1 or qkv.stride(3) != D or qkv.stride(2) != H * D:
        raise ValueError("qkv must be contiguous over its last three dimensions.")
    return qkv.stride(0), qkv.stride(1)


# ---------------------------------------------------------------------------
# Neighbourhood attention: launchers
# ---------------------------------------------------------------------------

def _na_triton_forward(qkv, bias, idx, w, valid, scale):
    B, N, _, H, D = qkv.shape
    P, _, G, _ = idx.shape
    SB, SN = _qkv_strides(qkv)
    out = torch.empty(B, G, N, H, D, device=qkv.device, dtype=qkv.dtype)
    lse = torch.empty(B, G, H, N, device=qkv.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), B * G * H)

    with _device_ctx(qkv.device):
        _na_fwd_kernel[grid](
            qkv, SB, SN, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx, out, lse,
            B, H, G, N, D, scale, _size_class(B * N), P=P,
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None, BLOCK_D=_block_d(D),
        )
    return out, lse


def _na_triton_backward(qkv, bias, idx, w, valid, scale, out, lse, gout, transposed):
    B, N, _, H, D = qkv.shape
    P, _, G, _ = idx.shape
    SB, SN = _qkv_strides(qkv)
    gout = gout.to(qkv.dtype).contiguous()                              # [B, G, N, H, D]
    delta = (gout.float() * out.float()).sum(-1).permute(0, 1, 3, 2).contiguous()  # [B,G,H,N]
    dqkv = torch.empty(B, N, 3, H, D, device=qkv.device, dtype=qkv.dtype)
    a = torch.empty(B, G, H, P, N, device=qkv.device, dtype=torch.float32)
    ds = torch.empty_like(a)
    tptr, tbase, tk, tw = transposed() if transposed is not None \
        else build_transposed_map(idx, w, N)
    size = _size_class(B * N)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    with _device_ctx(qkv.device):
        _na_bwd_q_kernel[grid](
            qkv, SB, SN, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx,
            gout, lse, delta, dqkv, a, ds, B, H, G, N, D, scale, size, P=P,
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None, BLOCK_D=_block_d(D),
        )
        _na_bwd_kv_kernel[grid](
            qkv, SB, SN, gout, a, ds, tptr, tbase, tk, tw, dqkv,
            B, H, G, N, D, scale, size, P=P, BLOCK_D=_block_d(D),
        )
    db = ds.sum(dim=(0, 1, 4)).to(bias.dtype) if bias is not None else None   # [H, P]
    return dqkv, db


# ===========================================================================
# Neighbourhood attention: one autograd Function, two backends
# ===========================================================================

@dataclass(frozen=True)
class _Options:
    use_triton: bool
    scale: float
    transposed: Optional[Callable[[], tuple]]


def _split(qkv):
    """qkv [B, N, 3, H, D] -> q, k, v views [B, H, N, D]."""
    return (qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2),
            qkv[:, :, 2].transpose(1, 2))


class NeighbourhoodAttentionFunction(torch.autograd.Function):
    """``qkv [B, N, 3, H, D] -> [B, G, N, H, D]``; dispatches to Triton or PyTorch."""

    @staticmethod
    def forward(ctx, qkv, bias, idx, w, valid, opts):
        if opts.use_triton:
            out, stat = _na_triton_forward(qkv, bias, idx, w, valid, opts.scale)
        else:
            out, stat = _torch_forward(*_split(qkv), bias, idx, w, valid, opts.scale)
            out = out.permute(0, 1, 3, 2, 4).contiguous()                # [B, G, N, H, D]
        ctx.opts = opts
        ctx.save_for_backward(qkv, bias, idx, w, valid, out, stat)
        return out

    @staticmethod
    def backward(ctx, gout):
        qkv, bias, idx, w, valid, out, stat = ctx.saved_tensors
        opts = ctx.opts
        need_qkv, need_b = ctx.needs_input_grad[:2]
        if opts.use_triton and not torch.is_grad_enabled():
            dqkv, db = _na_triton_backward(qkv, bias, idx, w, valid, opts.scale,
                                           out, stat, gout, opts.transposed)
        else:
            # Grad mode is on in a backward only with create_graph=True (double
            # backward): the probabilities and the output are recomputed with
            # operations that autograd records.
            q, k, v = _split(qkv)
            if torch.is_grad_enabled():
                o5, _ = _torch_forward(q, k, v, bias, idx, w, valid, opts.scale)
                a = None
            else:
                o5 = out.permute(0, 1, 3, 2, 4)
                a = stat if not opts.use_triton else None
            dq, dk, dv, db = _torch_backward(
                q, k, v, bias, idx, w, valid, opts.scale, o5, a,
                gout.permute(0, 1, 3, 2, 4), (need_qkv, need_qkv, need_qkv, need_b))
            dqkv = torch.stack([t.transpose(1, 2) for t in (dq, dk, dv)], dim=2) \
                if need_qkv else None
        return (dqkv if need_qkv else None, db if need_b else None, None, None, None, None)


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
    ok = _HAS_TRITON and (x.is_cuda or _INTERPRET_CPU) and x.dtype in _TRITON_DTYPES
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, CUDA tensors and float32, float16 "
            "or bfloat16 data (float64 runs on backend='torch')."
        )
    return ok


# Tests only: TRITON_INTERPRET=1 runs the kernels on CPU tensors with backend="triton".
_INTERPRET_CPU = _HAS_TRITON and os.environ.get("TRITON_INTERPRET") == "1"


def neighbourhood_attention_qkv(qkv, bias, idx, w, valid=None, *, backend="auto",
                                scale=None, transposed=None):
    """Neighbourhood attention on the packed projection ``qkv [B, N, 3, H, D]``.

    Returns ``[B, G, N, H, D]``.  The Triton kernels read ``qkv`` in place and
    write its gradient directly (no permute copy); see
    :func:`neighbourhood_attention` for the other arguments.
    """
    D = qkv.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    dt = _autocast_dtype() if qkv.is_cuda else None
    if dt is not None and qkv.dtype == torch.float32:
        qkv = qkv.to(dt)
    use_triton = _use_triton(qkv, backend) if backend != "auto" else \
        (_HAS_TRITON and qkv.is_cuda and qkv.dtype in _TRITON_DTYPES)
    if use_triton:
        if qkv.stride(4) != 1 or qkv.stride(3) != D or qkv.stride(2) != qkv.shape[3] * D:
            qkv = qkv.contiguous()
        if valid is not None:
            valid = valid.to(torch.int8).contiguous()
        if bias is not None:
            bias = bias.contiguous()
    else:
        if valid is not None:
            valid = valid.bool()
        if bias is not None and bias.dtype != qkv.dtype:
            bias = bias.to(qkv.dtype)
    opts = _Options(use_triton, float(scale), transposed)
    return NeighbourhoodAttentionFunction.apply(qkv, bias, idx, w, valid, opts)


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
    qkv = torch.stack([q, k, v], dim=1).permute(0, 3, 1, 2, 4)        # [B, N, 3, H, D] view
    if q.dtype != torch.float64 and (q.is_cuda or _INTERPRET_CPU) and backend != "torch":
        qkv = qkv.contiguous()
    out = neighbourhood_attention_qkv(qkv, bias, idx, w, valid, backend=backend, scale=scale,
                                      transposed=transposed)
    return out.permute(0, 1, 3, 2, 4)


# ===========================================================================
# Window attention
# ===========================================================================

@dataclass(frozen=True)
class WindowPlan:
    """Windows of :class:`~healpix_analyse.attention.HealPixWindowAttention` (device tensors).

    ``q_index [NW, TQ]`` / ``kv_index [NW, TK]`` (int32, padding = N) or None
    when the windows are consecutive cells; ``q_inv [N]`` (int64): the flat
    query slot of each cell (None with consecutive windows); ``kv_mask
    [NW, TK]`` (bool) or None when no key slot is padding; ``kptr [N+1]`` /
    ``kent`` (int32): for each cell, the flat key slots ``w * TK + s`` that
    hold it (CSR, fixed order), None when every cell is the key of one slot.
    """
    N: int
    NW: int
    TQ: int
    TK: int
    q_index: Optional[torch.Tensor] = None
    kv_index: Optional[torch.Tensor] = None
    q_inv: Optional[torch.Tensor] = None
    kv_mask: Optional[torch.Tensor] = None
    kptr: Optional[torch.Tensor] = None
    kent: Optional[torch.Tensor] = None


def _win_triton_forward(qkv, plan, scale):
    B, N, _, H, D = qkv.shape
    SB, SN = _qkv_strides(qkv)
    out = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)
    lse = torch.empty(B, N, H, device=qkv.device, dtype=torch.float32)
    qi = plan.q_index if plan.q_index is not None else qkv
    ki = plan.kv_index if plan.kv_index is not None else qkv

    def grid(meta):
        return (plan.NW * triton.cdiv(plan.TQ, meta["BLOCK_Q"]), B * H)

    with _device_ctx(qkv.device):
        _win_fwd_kernel[grid](
            qkv, SB, SN, qi, ki, out, lse, H, N, D, plan.TQ, plan.TK, scale,
            Q_CONTIG=plan.q_index is None, KV_CONTIG=plan.kv_index is None,
            PRECISION=_precision(qkv.dtype), BLOCK_D=_block_d(D),
        )
    return out, lse


def _win_triton_backward(qkv, plan, scale, out, lse, gout):
    B, N, _, H, D = qkv.shape
    SB, SN = _qkv_strides(qkv)
    gout = gout.to(qkv.dtype).contiguous()                               # [B, N, H, D]
    delta = (gout.float() * out.float()).sum(-1).contiguous()           # [B, N, H]
    dqkv = torch.empty(B, N, 3, H, D, device=qkv.device, dtype=qkv.dtype)
    qi = plan.q_index if plan.q_index is not None else qkv
    ki = plan.kv_index if plan.kv_index is not None else qkv
    direct = plan.kptr is None
    C = H * D
    part = qkv if direct else torch.empty(B, plan.NW * plan.TK, 2, H, D, device=qkv.device,
                                          dtype=torch.float32)
    prec = _precision(qkv.dtype)
    common = dict(Q_CONTIG=plan.q_index is None, KV_CONTIG=plan.kv_index is None,
                  PRECISION=prec, BLOCK_D=_block_d(D))
    with _device_ctx(qkv.device):
        _win_bwd_q_kernel[lambda m: (plan.NW * triton.cdiv(plan.TQ, m["BLOCK_Q"]), B * H)](
            qkv, SB, SN, qi, ki, gout, lse, delta, dqkv, H, N, D, plan.TQ, plan.TK, scale,
            **common)
        _win_bwd_kv_kernel[lambda m: (plan.NW * triton.cdiv(plan.TK, m["BLOCK_K"]), B * H)](
            qkv, SB, SN, qi, ki, gout, lse, delta, dqkv, part, H, N, D, plan.TQ, plan.TK,
            plan.NW, scale, DIRECT=direct, **common)
        if not direct:
            BF = min(128, triton.next_power_of_2(2 * C))
            _win_reduce_kernel[(triton.cdiv(N, 64), B, triton.cdiv(2 * C, BF))](
                part, plan.kptr, plan.kent, dqkv, SB, SN, N, 2 * C, plan.NW * plan.TK,
                BLOCK_M=64, BLOCK_F=BF, num_warps=4)
    return dqkv


def _gather_rows(t, index):
    """t [B, N, ...] -> [B, NW, T, ...] at ``index`` (padding N -> zeros)."""
    pad = torch.cat([t, t.new_zeros(t.shape[0], 1, *t.shape[2:])], dim=1)
    return pad[:, index.long()]


def window_attention_torch(qkv, plan, scale, dropout_p=0.0):
    """Reference: gather the windows, ``scaled_dot_product_attention``, scatter back.

    ``qkv [B, N, 3, H, D] -> [B, N, H, D]``; differentiable by autograd.
    """
    B, N, _, H, D = qkv.shape
    NW = plan.NW
    if plan.q_index is None:
        q = qkv[:, :, 0].reshape(B, NW, plan.TQ, H, D)
    else:
        q = _gather_rows(qkv[:, :, 0], plan.q_index)
    if plan.kv_index is None:
        kv = qkv[:, :, 1:].reshape(B, NW, plan.TK, 2, H, D)
    else:
        kv = _gather_rows(qkv[:, :, 1:], plan.kv_index)
    Tq, Tk = q.shape[2], kv.shape[2]
    q = q.permute(0, 1, 3, 2, 4).reshape(B * NW, H, Tq, D)
    k = kv[:, :, :, 0].permute(0, 1, 3, 2, 4).reshape(B * NW, H, Tk, D)
    v = kv[:, :, :, 1].permute(0, 1, 3, 2, 4).reshape(B * NW, H, Tk, D)
    mask = None
    if plan.kv_mask is not None:
        mask = plan.kv_mask.view(1, NW, 1, 1, Tk).expand(B, NW, 1, 1, Tk)
        mask = mask.reshape(B * NW, 1, 1, Tk)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                         dropout_p=dropout_p, scale=scale)
    o = o.view(B, NW, H, Tq, D).permute(0, 1, 3, 2, 4).reshape(B, NW * Tq, H, D)
    return o if plan.q_inv is None else o[:, plan.q_inv]


class WindowAttentionFunction(torch.autograd.Function):
    """``qkv [B, N, 3, H, D] -> [B, N, H, D]`` with the fused Triton kernels."""

    @staticmethod
    def forward(ctx, qkv, plan, scale):
        out, lse = _win_triton_forward(qkv, plan, scale)
        ctx.plan, ctx.scale = plan, scale
        ctx.save_for_backward(qkv, out, lse)
        return out

    @staticmethod
    def backward(ctx, gout):
        qkv, out, lse = ctx.saved_tensors
        if torch.is_grad_enabled():
            # double backward: differentiate the PyTorch reference
            with torch.enable_grad():
                ref = window_attention_torch(qkv, ctx.plan, ctx.scale)
                (g,) = torch.autograd.grad(ref, qkv, gout, create_graph=True)
            return g, None, None
        return _win_triton_backward(qkv, ctx.plan, ctx.scale, out, lse, gout), None, None


def window_attention(qkv, plan, *, scale=None, backend="auto", dropout_p=0.0):
    """Attention inside windows: ``qkv [B, N, 3, H, D] -> [B, N, H, D]``.

    ``backend="auto"``: fused Triton kernels on CUDA (float32, float16,
    bfloat16, autocast) when Triton is installed and there is no dropout;
    the PyTorch reference (gather + ``scaled_dot_product_attention``)
    otherwise.
    """
    D = qkv.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    dt = _autocast_dtype() if qkv.is_cuda else None
    if dt is not None and qkv.dtype == torch.float32:
        qkv = qkv.to(dt)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    ok = (_HAS_TRITON and (qkv.is_cuda or _INTERPRET_CPU) and qkv.dtype in _TRITON_DTYPES
          and dropout_p == 0.0)
    if backend == "triton" and not ok:
        raise RuntimeError("backend='triton' needs Triton, CUDA, float32/16-bit data and "
                           "no attention dropout.")
    if backend == "torch" or not ok or (backend == "auto" and not qkv.is_cuda):
        return window_attention_torch(qkv, plan, scale, dropout_p)
    if qkv.stride(4) != 1 or qkv.stride(3) != D or qkv.stride(2) != qkv.shape[3] * D:
        qkv = qkv.contiguous()
    return WindowAttentionFunction.apply(qkv, plan, float(scale))
