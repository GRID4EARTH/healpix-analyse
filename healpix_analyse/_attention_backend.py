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
    Forward: one program per (block of cells, batch, gauge), all heads at
    once (the stencil indices and weights are read once, and each gathered
    row holds the H*D values of a cell), gathers the 4 neighbours of each tap
    in registers and runs an online softmax over the taps (flash-attention
    style); it stores the output and the log-sum-exp.
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
online softmax); each window loops over its own number of keys only, and
padding keys are masked in registers, so a padded or overlapping window
costs no gathered copy and no mask tensor.  Backward, deterministic, no
atomics: ``dq`` per block of queries; ``dk``, ``dv`` per block of keys,
written to the cells directly when each cell is the key of one window slot.
Overlapping windows are coloured so that no two windows of a colour share a
key cell; the colours run one after the other and add their contribution in
place (fixed order, fp32).  ``"torch"``: gather,
``scaled_dot_product_attention``, scatter.
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
#   neighbourhood: o [B, G, N, H, D], lse and delta [B, G, H, N] (fp32);
#   window:        o [B, N, H, D],    lse and delta [B, N, H]    (fp32).
# The neighbourhood kernels process all the heads of a cell in one program:
# the stencil indices and weights are read once, and each gathered row is the
# H*D values of a cell (wider memory transactions than one head).

def _pow2(n, lo=1):
    return max(lo, 1 << max(0, int(n) - 1).bit_length())


def _block_d(D):
    return _pow2(D, 16)       # tl.dot needs >= 16


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

    # small autotune spaces (the likely winners)
    _NA_CONFIGS = _configs(((32, 4), (16, 4), (64, 8)))

    def _win_configs(spec):
        if _INTERPRET:
            spec = spec[:1]
        return [triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
                for bq, bk, nw, ns in spec]

    _WIN_CONFIGS = _win_configs(((64, 64, 4, 2), (64, 32, 4, 2), (128, 64, 8, 2)))

    # -----------------------------------------------------------------------
    # delta[row, h] = sum_d go[row, h, d] * o[row, h, d]  (fp32, both layouts)
    # -----------------------------------------------------------------------

    @triton.jit
    def _delta_kernel(go_ptr, o_ptr, delta_ptr, R, NC, H, D,
                      HN_LAYOUT: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        """Rows ``r`` of ``[R, H, D]``; HN_LAYOUT: row r = bg*NC + n -> delta[(bg*H + h)*NC + n],
        else delta[r*H + h]."""
        r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        r_ok = r < R
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_D)
        hd = h[:, None] * D + d[None, :]
        m3 = r_ok[:, None, None] & ((h < H)[:, None] & (d < D)[None, :])[None, :, :]
        off = r.to(tl.int64)[:, None, None] * (H * D) + hd[None, :, :]
        x = tl.load(go_ptr + off, mask=m3, other=0.0).to(tl.float32)
        y = tl.load(o_ptr + off, mask=m3, other=0.0).to(tl.float32)
        s = tl.sum(x * y, axis=2)                                         # [BM, BH]
        r64 = r.to(tl.int64)[:, None]
        if HN_LAYOUT:
            dst = ((r64 // NC) * H + h[None, :]) * NC + (r64 % NC)
        else:
            dst = r64 * H + h[None, :]
        tl.store(delta_ptr + dst, s, mask=r_ok[:, None] & (h < H)[None, :])

    # -----------------------------------------------------------------------
    # Neighbourhood attention (all heads per program)
    # -----------------------------------------------------------------------

    @triton.jit
    def _na_tap_kv(k_base, v_base, idx_ptr, w_ptr, p, g, n, n_ok, hd, m3, G, N, SN,
                   BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        """Bilinear key and value of tap p for the cells n, all heads: [BM, BH, BD] fp32."""
        kt = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        vt = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        for j in tl.static_range(4):
            off = ((p * 4 + j) * G + g).to(tl.int64) * N + n
            src = tl.load(idx_ptr + off, mask=n_ok, other=0).to(tl.int64)
            ww = tl.load(w_ptr + off, mask=n_ok, other=0.0).to(tl.float32)
            rows = src[:, None, None] * SN + hd[None, :, :]
            kt += ww[:, None, None] * tl.load(k_base + rows, mask=m3, other=0.0).to(tl.float32)
            vt += ww[:, None, None] * tl.load(v_base + rows, mask=m3, other=0.0).to(tl.float32)
        return kt, vt

    @triton.jit
    def _na_tap_score(qv, kt, bias_ptr, valid_ptr, h, h_ok, p, g, n, n_ok, G, N, P, scale,
                      HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr):
        s = tl.sum(qv * kt, axis=2) * scale                                # [BM, BH]
        if HAS_BIAS:
            s += tl.load(bias_ptr + h * P + p, mask=h_ok, other=0.0).to(tl.float32)[None, :]
        if HAS_VALID:
            ok = tl.load(valid_ptr + (p * G + g).to(tl.int64) * N + n, mask=n_ok, other=1)
            s = tl.where((ok != 0)[:, None], s, float("-inf"))
        return s

    @triton.autotune(configs=_NA_CONFIGS, key=["H", "D", "P", "G", "SIZE", "HAS_VALID"])
    @triton.jit
    def _na_fwd_kernel(qkv_ptr, SB, SN, idx_ptr, w_ptr, bias_ptr, valid_ptr, o_ptr, lse_ptr,
                       B, H, G, N, D, scale, SIZE, P: tl.constexpr,
                       HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        """o[b, g, n] = sum_p softmax_p(s) v_p; lse[b, g, h, n] = logsumexp_p(s)."""
        pid_n = tl.program_id(0)
        pid = tl.program_id(1)           # batch fastest: the same idx / w tiles
        b = pid % B                      # are reused from L2
        g = pid // B
        C = H * D
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        h = tl.arange(0, BLOCK_H)
        h_ok = h < H
        d = tl.arange(0, BLOCK_D)
        hd = h[:, None] * D + d[None, :]
        m3 = n_ok[:, None, None] & (h_ok[:, None] & (d < D)[None, :])[None, :, :]
        q_base = qkv_ptr + b.to(tl.int64) * SB
        qv = tl.load(q_base + n.to(tl.int64)[:, None, None] * SN + hd[None, :, :], mask=m3,
                     other=0.0).to(tl.float32)
        m_i = tl.full((BLOCK_M, BLOCK_H), -1.0e30, dtype=tl.float32)   # finite: no inf - inf
        l_i = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        for p in tl.static_range(P):
            kt, vt = _na_tap_kv(q_base + C, q_base + 2 * C, idx_ptr, w_ptr, p, g, n, n_ok, hd,
                                m3, G, N, SN, BLOCK_M, BLOCK_H, BLOCK_D)
            s = _na_tap_score(qv, kt, bias_ptr, valid_ptr, h, h_ok, p, g, n, n_ok, G, N, P,
                              scale, HAS_BIAS, HAS_VALID)
            m_new = tl.maximum(m_i, s)
            alpha = tl.exp(m_i - m_new)
            e = tl.exp(s - m_new)
            l_i = l_i * alpha + e
            acc = acc * alpha[:, :, None] + e[:, :, None] * vt
            m_i = m_new
        acc = acc / l_i[:, :, None]
        bg = (b * G + g).to(tl.int64)
        orow = (bg * N + n)[:, None, None] * C + hd[None, :, :]
        tl.store(o_ptr + orow, acc.to(o_ptr.dtype.element_ty), mask=m3)
        lrow = (bg * H + h[None, :]) * N + n[:, None]
        tl.store(lse_ptr + lrow, m_i + tl.log(l_i), mask=n_ok[:, None] & h_ok[None, :])

    @triton.autotune(configs=_NA_CONFIGS, key=["H", "D", "P", "G", "SIZE", "HAS_VALID"])
    @triton.jit
    def _na_bwd_q_kernel(qkv_ptr, SB, SN, idx_ptr, w_ptr, bias_ptr, valid_ptr,
                         go_ptr, lse_ptr, delta_ptr, dqkv_ptr, a_ptr, ds_ptr,
                         B, H, G, N, D, scale, SIZE, P: tl.constexpr,
                         HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        """dq (summed over the gauges in registers), a and ds [B, G, H, P, N]."""
        pid_n = tl.program_id(0)
        b = tl.program_id(1)
        C = H * D
        n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        n_ok = n < N
        h = tl.arange(0, BLOCK_H)
        h_ok = h < H
        d = tl.arange(0, BLOCK_D)
        hd = h[:, None] * D + d[None, :]
        m3 = n_ok[:, None, None] & (h_ok[:, None] & (d < D)[None, :])[None, :, :]
        mh = n_ok[:, None] & h_ok[None, :]
        q_base = qkv_ptr + b.to(tl.int64) * SB
        nrow = n.to(tl.int64)[:, None, None] * SN + hd[None, :, :]
        qv = tl.load(q_base + nrow, mask=m3, other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bg = (b * G + g).to(tl.int64)
            go = tl.load(go_ptr + (bg * N + n)[:, None, None] * C + hd[None, :, :], mask=m3,
                         other=0.0).to(tl.float32)
            lrow = (bg * H + h[None, :]) * N + n[:, None]
            lse = tl.load(lse_ptr + lrow, mask=mh, other=0.0)
            delta = tl.load(delta_ptr + lrow, mask=mh, other=0.0)
            for p in tl.static_range(P):
                kt, vt = _na_tap_kv(q_base + C, q_base + 2 * C, idx_ptr, w_ptr, p, g, n, n_ok,
                                    hd, m3, G, N, SN, BLOCK_M, BLOCK_H, BLOCK_D)
                s = _na_tap_score(qv, kt, bias_ptr, valid_ptr, h, h_ok, p, g, n, n_ok, G, N, P,
                                  scale, HAS_BIAS, HAS_VALID)
                a = tl.exp(s - lse)
                ds = a * (tl.sum(go * vt, axis=2) - delta)
                dq += ds[:, :, None] * kt
                row = ((bg * H + h[None, :]) * P + p) * N + n[:, None]
                tl.store(a_ptr + row, a, mask=mh)
                tl.store(ds_ptr + row, ds, mask=mh)
        tl.store(dqkv_ptr + b.to(tl.int64) * SB + nrow,
                 (dq * scale).to(dqkv_ptr.dtype.element_ty), mask=m3)

    @triton.autotune(configs=_NA_CONFIGS, key=["H", "D", "P", "G", "SIZE"])
    @triton.jit
    def _na_bwd_kv_kernel(qkv_ptr, SB, SN, go_ptr, a_ptr, ds_ptr, tptr_ptr, tbase_ptr, tk_ptr,
                          tw_ptr, dqkv_ptr, B, H, G, N, D, scale, SIZE, P: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
        """dk[s] = scale sum tw ds q[t], dv[s] = sum tw a go[t] over the entries of s.

        One program per (block of source cells, batch); the entries of each
        source cell come from the transposed map, in a fixed order: no atomics.
        """
        pid_n = tl.program_id(0)
        b = tl.program_id(1)
        C = H * D
        s_ = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        s_ok = s_ < N
        s64 = s_.to(tl.int64)
        h = tl.arange(0, BLOCK_H)
        h_ok = h < H
        d = tl.arange(0, BLOCK_D)
        hd = h[:, None] * D + d[None, :]
        hd_ok = h_ok[:, None] & (d < D)[None, :]
        q_base = qkv_ptr + b.to(tl.int64) * SB
        GN1 = (G * N + 1).to(tl.int64)       # row stride of ptr [P, G*N + 1]
        dk = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        dv = tl.zeros((BLOCK_M, BLOCK_H, BLOCK_D), dtype=tl.float32)
        for g in range(0, G):
            bg = (b * G + g).to(tl.int64)
            go_base = go_ptr + bg * N * C
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
                    mh = m[:, None] & h_ok[None, :]
                    pr = ((bg * H + h[None, :]) * P + p) * N + t[:, None]
                    av = tl.load(a_ptr + pr, mask=mh, other=0.0)
                    dsv = tl.load(ds_ptr + pr, mask=mh, other=0.0)
                    m3 = m[:, None, None] & hd_ok[None, :, :]
                    qv = tl.load(q_base + t[:, None, None] * SN + hd[None, :, :], mask=m3,
                                 other=0.0)
                    gov = tl.load(go_base + t[:, None, None] * C + hd[None, :, :], mask=m3,
                                  other=0.0)
                    dk += (ww[:, None] * dsv)[:, :, None] * qv.to(tl.float32)
                    dv += (ww[:, None] * av)[:, :, None] * gov.to(tl.float32)
        m3 = s_ok[:, None, None] & hd_ok[None, :, :]
        out = dqkv_ptr + b.to(tl.int64) * SB + s64[:, None, None] * SN + hd[None, :, :]
        tl.store(out + C, (dk * scale).to(dqkv_ptr.dtype.element_ty), mask=m3)
        tl.store(out + 2 * C, dv.to(dqkv_ptr.dtype.element_ty), mask=m3)

    # -----------------------------------------------------------------------
    # Window attention (flash-attention style, gathered windows)
    # -----------------------------------------------------------------------

    @triton.jit
    def _win_rows(idx_ptr, wi, slot, T, n_valid, N, CONTIG: tl.constexpr):
        """Cell positions of the window slots (padding -> N) and their mask."""
        s_ok = slot < n_valid
        if CONTIG:
            pos = (wi * T + slot).to(tl.int64)
        else:
            pos = tl.load(idx_ptr + wi.to(tl.int64) * T + slot, mask=s_ok, other=N).to(tl.int64)
        ok = s_ok & (pos < N)
        return tl.where(ok, pos, 0), ok

    @triton.jit
    def _win_nkeys(kvlen_ptr, wi, TK, HAS_LEN: tl.constexpr):
        if HAS_LEN:
            return tl.load(kvlen_ptr + wi)
        return TK

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D", "Q_CONTIG", "KV_CONTIG", "HAS_LEN"])
    @triton.jit
    def _win_fwd_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, kvlen_ptr, o_ptr, lse_ptr,
                        H, N, D, TQ, TK, scale,
                        Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr, HAS_LEN: tl.constexpr,
                        PRECISION: tl.constexpr,
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
        qpos, q_ok = _win_rows(qidx_ptr, wi, qb * BLOCK_Q + tl.arange(0, BLOCK_Q), TQ, TQ, N,
                               Q_CONTIG)
        q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=q_ok[:, None] & d_ok[None, :],
                    other=0.0)
        n_k = _win_nkeys(kvlen_ptr, wi, TK, HAS_LEN)
        m_i = tl.full((BLOCK_Q,), -1.0e30, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, n_k, BLOCK_K):
            kpos, k_ok = _win_rows(kvidx_ptr, wi, k0 + tl.arange(0, BLOCK_K), TK, n_k, N,
                                   KV_CONTIG)
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

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D", "Q_CONTIG", "KV_CONTIG", "HAS_LEN"])
    @triton.jit
    def _win_bwd_q_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, kvlen_ptr, go_ptr, lse_ptr,
                          delta_ptr, dqkv_ptr, H, N, D, TQ, TK, scale,
                          Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr,
                          HAS_LEN: tl.constexpr, PRECISION: tl.constexpr,
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
        qpos, q_ok = _win_rows(qidx_ptr, wi, qb * BLOCK_Q + tl.arange(0, BLOCK_Q), TQ, TQ, N,
                               Q_CONTIG)
        mq = q_ok[:, None] & d_ok[None, :]
        q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=mq, other=0.0)
        rows = b.to(tl.int64) * N + qpos
        go = tl.load(go_ptr + rows[:, None] * C + h * D + d[None, :], mask=mq, other=0.0)
        lse = tl.load(lse_ptr + rows * H + h, mask=q_ok, other=0.0)
        delta = tl.load(delta_ptr + rows * H + h, mask=q_ok, other=0.0)
        n_k = _win_nkeys(kvlen_ptr, wi, TK, HAS_LEN)
        dq = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        for k0 in range(0, n_k, BLOCK_K):
            kpos, k_ok = _win_rows(kvidx_ptr, wi, k0 + tl.arange(0, BLOCK_K), TK, n_k, N,
                                   KV_CONTIG)
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

    @triton.autotune(configs=_WIN_CONFIGS, key=["TQ", "TK", "D", "Q_CONTIG", "KV_CONTIG", "HAS_LEN"])
    @triton.jit
    def _win_bwd_kv_kernel(qkv_ptr, SB, SN, qidx_ptr, kvidx_ptr, kvlen_ptr, wlist_ptr,
                           go_ptr, lse_ptr, delta_ptr, acc_ptr, ASB, ASN, H, N, D, TQ, TK,
                           scale,
                           Q_CONTIG: tl.constexpr, KV_CONTIG: tl.constexpr,
                           HAS_LEN: tl.constexpr, ACCUMULATE: tl.constexpr,
                           PRECISION: tl.constexpr,
                           BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        """dk, dv of one block of keys of one window.

        Written to ``acc`` (k at offset 0, v at offset C of each row, strides
        ASB / ASN).  ``ACCUMULATE`` (overlapping windows): the windows come
        from ``wlist``, one colour at a time, and no two windows of a colour
        share a key cell, so the read-add-write below never races and the
        summation order is fixed by the colour order: deterministic, no
        atomics.
        """
        n_kb = tl.cdiv(TK, BLOCK_K)
        pid0 = tl.program_id(0)
        kb = pid0 % n_kb
        if ACCUMULATE:
            wi = tl.load(wlist_ptr + pid0 // n_kb)
        else:
            wi = pid0 // n_kb
        n_k = _win_nkeys(kvlen_ptr, wi, TK, HAS_LEN)
        if kb * BLOCK_K < n_k:
            pid1 = tl.program_id(1)
            h = pid1 % H
            b = pid1 // H
            C = H * D
            d = tl.arange(0, BLOCK_D)
            d_ok = d < D
            base = qkv_ptr + b.to(tl.int64) * SB + h * D
            kslot = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            kpos, k_ok = _win_rows(kvidx_ptr, wi, kslot, TK, n_k, N, KV_CONTIG)
            mk = k_ok[:, None] & d_ok[None, :]
            k = tl.load(base + C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            v = tl.load(base + 2 * C + kpos[:, None] * SN + d[None, :], mask=mk, other=0.0)
            dk = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
            dv = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
            for q0 in range(0, TQ, BLOCK_Q):
                qpos, q_ok = _win_rows(qidx_ptr, wi, q0 + tl.arange(0, BLOCK_Q), TQ, TQ, N,
                                       Q_CONTIG)
                mq = q_ok[:, None] & d_ok[None, :]
                q = tl.load(base + qpos[:, None] * SN + d[None, :], mask=mq, other=0.0)
                rows = b.to(tl.int64) * N + qpos
                go = tl.load(go_ptr + rows[:, None] * C + h * D + d[None, :], mask=mq, other=0.0)
                lse = tl.load(lse_ptr + rows * H + h, mask=q_ok, other=0.0)
                delta = tl.load(delta_ptr + rows * H + h, mask=q_ok, other=0.0)
                sT = tl.dot(k, tl.trans(q), input_precision=PRECISION) * scale     # [BK, BQ]
                pT = tl.where(k_ok[:, None] & q_ok[None, :], tl.exp(sT - lse[None, :]), 0.0)
                dv = tl.dot(pT.to(go.dtype), go, dv, input_precision=PRECISION)
                dpT = tl.dot(v, tl.trans(go), input_precision=PRECISION)
                dsT = pT * (dpT - delta[None, :])
                dk = tl.dot(dsT.to(q.dtype), q, dk, input_precision=PRECISION)
            out = acc_ptr + b.to(tl.int64) * ASB + h * D + kpos[:, None] * ASN + d[None, :]
            dk = dk * scale
            if ACCUMULATE:
                dk += tl.load(out, mask=mk, other=0.0).to(tl.float32)
                dv += tl.load(out + C, mask=mk, other=0.0).to(tl.float32)
            tl.store(out, dk.to(acc_ptr.dtype.element_ty), mask=mk)
            tl.store(out + C, dv.to(acc_ptr.dtype.element_ty), mask=mk)


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


def _delta(gout, out, nc, hn_layout):
    """sum_d gout * out in fp32, without fp32 copies of the two tensors."""
    H, D = out.shape[-2], out.shape[-1]
    R = out.numel() // (H * D)
    if hn_layout:      # neighbourhood: rows (b, g, n) -> [B, G, H, N]
        delta = torch.empty(out.shape[0], out.shape[1], H, nc, device=out.device,
                            dtype=torch.float32)
    else:              # window: rows (b, n) -> [B, N, H]
        delta = torch.empty(out.shape[:-1], device=out.device, dtype=torch.float32)
    BH, BD = _pow2(H), _pow2(D)
    BM = max(1, min(64, 4096 // (BH * BD)))
    with _device_ctx(out.device):
        _delta_kernel[(triton.cdiv(R, BM),)](gout, out, delta, R, nc, H, D,
                                             HN_LAYOUT=hn_layout, BLOCK_M=BM,
                                             BLOCK_H=BH, BLOCK_D=BD)
    return delta


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
        return (triton.cdiv(N, meta["BLOCK_M"]), B * G)

    with _device_ctx(qkv.device):
        _na_fwd_kernel[grid](
            qkv, SB, SN, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx, out, lse,
            B, H, G, N, D, scale, _size_class(B * N), P=P,
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None,
            BLOCK_H=_pow2(H), BLOCK_D=_pow2(D),
        )
    return out, lse


def _na_triton_backward(qkv, bias, idx, w, valid, scale, out, lse, gout, transposed):
    B, N, _, H, D = qkv.shape
    P, _, G, _ = idx.shape
    SB, SN = _qkv_strides(qkv)
    gout = gout.to(qkv.dtype).contiguous()                              # [B, G, N, H, D]
    delta = _delta(gout, out, N, True)                                  # [B, G, H, N]
    dqkv = torch.empty(B, N, 3, H, D, device=qkv.device, dtype=qkv.dtype)
    a = torch.empty(B, G, H, P, N, device=qkv.device, dtype=torch.float32)
    ds = torch.empty_like(a)
    tptr, tbase, tk, tw = transposed() if transposed is not None \
        else build_transposed_map(idx, w, N)
    size = _size_class(B * N)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_M"]), B)

    with _device_ctx(qkv.device):
        _na_bwd_q_kernel[grid](
            qkv, SB, SN, idx, w, bias if bias is not None else w,
            valid if valid is not None else idx,
            gout, lse, delta, dqkv, a, ds, B, H, G, N, D, scale, size, P=P,
            HAS_BIAS=bias is not None, HAS_VALID=valid is not None,
            BLOCK_H=_pow2(H), BLOCK_D=_pow2(D),
        )
        _na_bwd_kv_kernel[grid](
            qkv, SB, SN, gout, a, ds, tptr, tbase, tk, tw, dqkv,
            B, H, G, N, D, scale, size, P=P, BLOCK_H=_pow2(H), BLOCK_D=_pow2(D),
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
        qkv = qkv.contiguous()      # the gradient is written with the strides of qkv
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
    when the windows are consecutive cells; ``kv_len [NW]`` (int32): the
    number of keys of each window (its first slots), None when all have TK;
    ``q_inv [N]`` (int64): the flat query slot of each cell (None with
    consecutive windows); ``kv_mask [NW, TK]`` (bool) or None when no key slot
    is padding; ``colours`` (int32, the window ids sorted by colour) and
    ``colour_bounds`` (the start of each colour in it): set when windows
    overlap; no two windows of a colour share a key cell.
    """
    N: int
    NW: int
    TQ: int
    TK: int
    q_index: Optional[torch.Tensor] = None
    kv_index: Optional[torch.Tensor] = None
    kv_len: Optional[torch.Tensor] = None
    q_inv: Optional[torch.Tensor] = None
    kv_mask: Optional[torch.Tensor] = None
    colours: Optional[torch.Tensor] = None
    colour_bounds: tuple = ()


def _win_triton_forward(qkv, plan, scale):
    B, N, _, H, D = qkv.shape
    SB, SN = _qkv_strides(qkv)
    out = torch.empty(B, N, H, D, device=qkv.device, dtype=qkv.dtype)
    lse = torch.empty(B, N, H, device=qkv.device, dtype=torch.float32)
    qi = plan.q_index if plan.q_index is not None else qkv
    ki = plan.kv_index if plan.kv_index is not None else qkv
    kl = plan.kv_len if plan.kv_len is not None else qkv

    def grid(meta):
        return (plan.NW * triton.cdiv(plan.TQ, meta["BLOCK_Q"]), B * H)

    with _device_ctx(qkv.device):
        _win_fwd_kernel[grid](
            qkv, SB, SN, qi, ki, kl, out, lse, H, N, D, plan.TQ, plan.TK, scale,
            Q_CONTIG=plan.q_index is None, KV_CONTIG=plan.kv_index is None,
            HAS_LEN=plan.kv_len is not None,
            PRECISION=_precision(qkv.dtype), BLOCK_D=_block_d(D),
        )
    return out, lse


_ACC_CONFIG = (64, 64)       # BLOCK_Q, BLOCK_K of the accumulating (coloured) launches


def _win_triton_backward(qkv, plan, scale, out, lse, gout):
    B, N, _, H, D = qkv.shape
    SB, SN = _qkv_strides(qkv)
    gout = gout.to(qkv.dtype).contiguous()                               # [B, N, H, D]
    delta = _delta(gout, out, N, False)                                  # [B, N, H]
    dqkv = torch.empty(B, N, 3, H, D, device=qkv.device, dtype=qkv.dtype)
    qi = plan.q_index if plan.q_index is not None else qkv
    ki = plan.kv_index if plan.kv_index is not None else qkv
    kl = plan.kv_len if plan.kv_len is not None else qkv
    flags = dict(Q_CONTIG=plan.q_index is None, KV_CONTIG=plan.kv_index is None,
                 HAS_LEN=plan.kv_len is not None, PRECISION=_precision(qkv.dtype),
                 BLOCK_D=_block_d(D))
    overlap = plan.colours is not None
    with _device_ctx(qkv.device):
        _win_bwd_q_kernel[lambda m: (plan.NW * triton.cdiv(plan.TQ, m["BLOCK_Q"]), B * H)](
            qkv, SB, SN, qi, ki, kl, gout, lse, delta, dqkv, H, N, D, plan.TQ, plan.TK,
            scale, **flags)
        dkv = dqkv[:, :, 1:]
        if not overlap:
            # each cell is the key of one window slot: written in place
            _win_bwd_kv_kernel[lambda m: (plan.NW * triton.cdiv(plan.TK, m["BLOCK_K"]), B * H)](
                qkv, SB, SN, qi, ki, kl, qkv, gout, lse, delta, dkv, SB, SN, H, N, D,
                plan.TQ, plan.TK, scale, ACCUMULATE=False, **flags)
        else:
            # overlapping windows: accumulate colour by colour (fp32)
            if qkv.dtype == torch.float32:
                dkv.zero_()
                acc, ASB, ASN = dkv, SB, SN
            else:
                acc = torch.zeros(B, N, 2, H, D, device=qkv.device, dtype=torch.float32)
                ASB, ASN = acc.stride(0), acc.stride(1)
            # The read-add-write kernel must run exactly once per colour: no
            # autotune (its trial runs would add the contribution again).
            bounds = plan.colour_bounds
            BQ, BK = _ACC_CONFIG
            for c in range(len(bounds) - 1):
                lo, hi = bounds[c], bounds[c + 1]
                wl = plan.colours[lo:hi]
                _win_bwd_kv_kernel.fn[((hi - lo) * triton.cdiv(plan.TK, BK), B * H)](
                    qkv, SB, SN, qi, ki, kl, wl, gout, lse, delta, acc, ASB, ASN, H, N, D,
                    plan.TQ, plan.TK, scale, ACCUMULATE=True, BLOCK_Q=BQ, BLOCK_K=BK,
                    num_warps=4, num_stages=2, **flags)
            if acc is not dkv:
                dkv.copy_(acc)
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


def _sdpa_math():
    """Context: SDPA on its math kernel (the fused ones have no double backward)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except ImportError:  # pragma: no cover - torch < 2.3
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                              enable_math=True)


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
            with torch.enable_grad(), _sdpa_math():
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
    qkv = qkv.contiguous()          # the gradient is written with the strides of qkv
    return WindowAttentionFunction.apply(qkv, plan, float(scale))
