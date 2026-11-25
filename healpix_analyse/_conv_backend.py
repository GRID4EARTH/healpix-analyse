"""Compute backends of :class:`healpix_analyse.convol.HealPixConv`.

The layer is a sum, over the P stencil taps and the 4 bilinear neighbours of
each tap, of gathered input rows times one weight matrix per gauge and tap::

    y[b, g*Co + o, k] = bias[g*Co + o] + sum_p sum_c A_p[g, b, k, c] W[g, c, o, p]
    A_p[g, b, k, c]   = sum_j w[p, j, g, k] x[b, c, idx[p, j, g, k]]

``idx`` and ``w`` are the *tap geometry*, both ``[P, 4, G, K]``: ``idx`` holds
int32 positions in the order of the input (the sort permutation of the
partial domain is folded in) and ``w`` the renormalised bilinear weights.

Neither backend ever materialises the gathered tensor ``[B, C, G, K, P]``
(the E elements of issue #90).

``"torch"`` (CPU, float64, and the reference everywhere)
    A loop over the P taps: ``index_select`` of the 4 neighbours into a
    ``[G, K*B, C]`` buffer, then ``baddbmm``.  The backward is written by
    hand: it recomputes the gather of each tap for grad W and scatters grad x
    with ``index_add_``.  Memory: input + output + one tap.

``"triton"`` (CUDA, float32 / float16 / bfloat16, fp32 accumulation)
    Fused kernels.  Forward: a gather-GEMM, one program per (batch, gauge,
    block of cells, block of output channels); the tap tile ``A_p`` is built
    in registers and multiplied by ``W[g, p]``.  grad W: the same gather,
    reduced over the rows; each program sums a contiguous chunk of rows for a
    block of taps, contracted by one product over (tap, channel), and writes
    its partial sum (no atomics, deterministic).  grad x: by default
    (``deterministic=True``), a gather on the transposed map of the stencil
    (for each source cell, its incoming (target cell, weight) entries, in CSR
    form per gauge and tap): no atomics, and the same result at every run.
    With ``deterministic=False``, a GEMM-scatter with fp32 atomics.

Extension points
----------------
The kernels are written so that the two variants needed by
``HealPixConv(missing="local_mean")`` and the wavelet power spectrum can plug
in without a rewrite:

* a **masked gather** (value and validity in one pass, renormalisation and the
  local-mean fallback in registers) replaces ``_gather_taps``, the only place
  that reads the input through the stencil;
* a **contraction with arbitrary taps** (complex taps, several voices, the
  coverage contraction) replaces ``_weight_taps``: the weight tile is read
  from a ``[G, P, C, Co]`` array whose meaning the kernels do not assume.

Triton is optional: without it (or without CUDA) everything runs on the
``"torch"`` backend.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import torch

try:  # Triton is an optional extra
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = [
    "BACKENDS",
    "triton_available",
    "tap_geometry",
    "build_transposed_map",
    "gauge_conv",
    "GaugeConvFunction",
]

BACKENDS = ("auto", "triton", "torch")
_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def triton_available() -> bool:
    """True when the Triton kernels can run (Triton importable and CUDA present)."""
    return _HAS_TRITON and torch.cuda.is_available()


# ===========================================================================
# Geometry helpers
# ===========================================================================

def tap_geometry(pos_safe, w_norm, sort_order, G, K, P):
    """Tap-major geometry from the bound stencil.

    Parameters
    ----------
    pos_safe : LongTensor [4, G, K*P]
        Positions of the 4 bilinear neighbours in the *sorted* cell order.
    w_norm : Tensor [4, G, K*P]
        Renormalised bilinear weights.
    sort_order : LongTensor [K]
        ``argsort(cell_ids)``.

    Returns
    -------
    idx : IntTensor [P, 4, G, K]
        Positions in the order of the input: ``sort_order[pos_safe]``, so that
        the input is gathered directly, with no per-call reordering copy.
    w : Tensor [P, 4, G, K]
    """
    pos = pos_safe.reshape(4, G, K, P)
    idx = sort_order.to(torch.int32)[pos].permute(3, 0, 1, 2).contiguous()
    w = w_norm.reshape(4, G, K, P).permute(3, 0, 1, 2).contiguous()
    return idx, w


def build_transposed_map(idx, w, K):
    """Transposed stencil map, in CSR form per tap and gauge.

    For tap ``p``, gauge ``g`` and source cell ``s``, the entries
    ``e in [base[p] + ptr[p, g*K + s], base[p] + ptr[p, g*K + s + 1])`` hold
    the target cells ``tk[e]`` that read ``s`` through that tap and gauge and
    the bilinear weights ``tw[e]``.  Zero weights are dropped.  The order of
    the entries is fixed, so a gather on this map is deterministic.

    Returns
    -------
    ptr : IntTensor [P, G*K + 1]
        Row offsets relative to ``base[p]``.
    base : LongTensor [P + 1]
    tk : IntTensor [nnz]
    tw : Tensor [nnz]  (same dtype as ``w``)
    """
    P, J, G, K_ = idx.shape
    if K_ != K:
        raise ValueError(f"idx has {K_} cells, expected {K}.")
    GK = G * K
    if J * GK >= 2**31:
        raise ValueError("transposed map too large for int32 row offsets.")
    dev = idx.device
    nz_counts = [(w[p] != 0).sum().item() for p in range(P)]
    base_list = [0]
    for n in nz_counts:
        base_list.append(base_list[-1] + n)
    nnz = base_list[-1]
    tk = torch.empty(nnz, dtype=torch.int32, device=dev)
    tw = torch.empty(nnz, dtype=w.dtype, device=dev)
    ptr = torch.empty(P, GK + 1, dtype=torch.int32, device=dev)
    rows = torch.arange(GK + 1, device=dev, dtype=torch.int64)
    gofs = (torch.arange(G, device=dev, dtype=torch.int64) * K).view(1, G, 1)
    k_ids = torch.arange(K, device=dev, dtype=torch.int32).view(1, 1, K).expand(J, G, K)
    for p in range(P):
        wp = w[p]
        nz = wp != 0
        key = (idx[p].to(torch.int64) + gofs)[nz]
        order = torch.argsort(key, stable=True)
        skey = key[order]
        a, e = base_list[p], base_list[p + 1]
        tk[a:e] = k_ids[nz][order]
        tw[a:e] = wp[nz][order]
        ptr[p] = torch.searchsorted(skey, rows).to(torch.int32)
    base = torch.tensor(base_list, dtype=torch.int64, device=dev)
    return ptr, base, tk, tw


# ===========================================================================
# PyTorch backend: tap loop with a hand-written backward
# ===========================================================================

def _gather_tap(xt, idx_p, w_p):
    """xt [K, B, C]; idx_p, w_p [4, G, K] -> [G, K*B, C] = sum_j w * xt[idx]."""
    G, K = idx_p.shape[1], idx_p.shape[2]
    B, C = xt.shape[1], xt.shape[2]
    acc = None
    for j in range(4):
        # int64 per tap: index_select / index_add_ take a slow path with int32
        # indices on CPU (index_add_ is ~10x slower), and the copy is small
        v = xt.index_select(0, idx_p[j].reshape(-1).long()).view(G, K, B, C)
        v = v * w_p[j].view(G, K, 1, 1)
        acc = v if acc is None else acc.add_(v)
    return acc.view(G, K * B, C)


def _torch_forward(x, weight, bias, idx, w):
    B, C, K = x.shape
    G, _, CO, P = weight.shape
    w = w.to(x.dtype)
    xt = x.permute(2, 0, 1).contiguous()                       # [K, B, C]
    y = x.new_zeros(G, K * B, CO)
    for p in range(P):
        y.baddbmm_(_gather_tap(xt, idx[p], w[p]), weight[..., p])
    out = y.view(G, K, B, CO).permute(2, 0, 3, 1).reshape(B, G * CO, K).contiguous()
    if bias is not None:
        out.add_(bias.to(out.dtype).view(1, -1, 1))
    return out


def _torch_backward(x, weight, idx, w, gy, need_x, need_w):
    B, C, K = x.shape
    G, _, CO, P = weight.shape
    w = w.to(x.dtype)
    gyt = gy.to(x.dtype).view(B, G, CO, K).permute(1, 3, 0, 2).reshape(G, K * B, CO)
    xt = x.permute(2, 0, 1).contiguous() if need_w else None
    gx_flat = x.new_zeros(K, B * C) if need_x else None
    gW = torch.empty_like(weight) if need_w else None
    for p in range(P):
        if need_w:
            # recomputed, never stored
            gW[..., p] = torch.bmm(_gather_tap(xt, idx[p], w[p]).transpose(1, 2), gyt)
        if need_x:
            h = torch.bmm(gyt, weight[..., p].transpose(1, 2)).view(G, K, B * C)
            for j in range(4):
                gx_flat.index_add_(
                    0, idx[p, j].reshape(-1).long(),
                    (h * w[p, j].view(G, K, 1)).view(G * K, B * C),
                )
    gx = gx_flat.view(K, B, C).permute(1, 2, 0) if need_x else None
    return gx, gW


# ===========================================================================
# Triton backend
# ===========================================================================

def _next_pow2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def _block_nc(C, CO):
    """BLOCK_C, BLOCK_N: the channel tiles (tl.dot needs >= 16)."""
    return (min(64, max(16, _next_pow2(C))), min(64, max(16, _next_pow2(CO))))


def _precision(dtype):
    if dtype == torch.float32 and not torch.backends.cuda.matmul.allow_tf32:
        return "ieee"
    return "tf32"   # TF32 for float32 when allowed; ignored for 16-bit inputs


# programs of the grad-W kernel per streaming multiprocessor (row chunks)
_SPLIT_PROGRAMS_PER_SM = 32


def _size_class(n):
    """Coarse size bucket, part of the autotune key."""
    return int(math.log2(max(int(n), 1))) // 2


def _taps_per_block(P, width):
    """Taps contracted together by one ``tl.dot`` in the grad-W kernel (TP, a power of two).

    grad W reduces over the rows: with one tap its products are
    ``[C, rows] @ [rows, Co]``, only 16 x 16 outputs for 16 channels, which
    leave the GPU idle (3.5x the forward time on an A100).  A block of taps
    makes them ``[TP*C, rows] @ [rows, Co]``.  TP is the largest power of two
    with ``TP * width <= 128`` whose padding of the last block stays within a
    quarter.  The forward and the deterministic grad x keep one tap per
    product (TP = 1): there the products are already ``[rows, C] @ [C, Co]``,
    and blocks of taps measured 1.1-2x slower (register pressure).
    """
    tp = 1
    while tp * 2 * width <= 128 and tp * 2 <= P:
        cand = tp * 2
        slots = -(-P // cand) * cand
        if (slots - P) > slots // 4:
            break
        tp = cand
    return tp


if _HAS_TRITON:

    def _configs(spec):
        return [triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns) for bm, nw, ns in spec]

    _FWD_CONFIGS = _configs(((32, 2, 1), (32, 4, 1), (64, 2, 1), (64, 4, 1), (128, 4, 1), (128, 8, 1)))
    _BWD_W_CONFIGS = _configs(((32, 2, 1), (32, 4, 1), (64, 4, 1), (64, 8, 1), (128, 8, 1)))
    _BWD_X_CONFIGS = _configs(((32, 2, 1), (64, 2, 1), (64, 4, 1), (128, 4, 1)))
    _BWD_X_CSR_CONFIGS = _configs(((32, 2, 1), (32, 4, 1), (64, 2, 1), (64, 4, 1), (128, 4, 1), (128, 8, 1)))

    @triton.jit
    def _gather_taps(xt_ptr, idx_ptr, w_ptr, p0, g, b, k, c, k_ok, c_ok, K, C, G, P,
                     BLOCK_M: tl.constexpr, TP: tl.constexpr, BLOCK_C: tl.constexpr):
        """A[m, t, c] = sum_j w[p0+t, j, g, k_m] * xt[b, idx[p0+t, j, g, k_m], c], in fp32.

        ``xt`` is ``[B, K, C]``, ``idx`` and ``w`` are ``[P, 4, G, K]``; taps
        beyond P read nothing and give 0.  This is the only place that reads
        the input through the stencil: a masked variant (value and validity in
        one pass) replaces it.
        """
        p = p0 + tl.arange(0, TP)
        m2 = k_ok[:, None] & (p < P)[None, :]                          # [BM, TP]
        a = tl.zeros((BLOCK_M, TP, BLOCK_C), dtype=tl.float32)
        x_base = xt_ptr + b.to(tl.int64) * K * C
        for j in tl.static_range(4):
            off = ((p * 4 + j) * G + g).to(tl.int64)[None, :] * K + k[:, None]
            src = tl.load(idx_ptr + off, mask=m2, other=0).to(tl.int64)
            ww = tl.load(w_ptr + off, mask=m2, other=0.0).to(tl.float32)
            xv = tl.load(x_base + src[:, :, None] * C + c[None, None, :],
                         mask=m2[:, :, None] & c_ok[None, None, :], other=0.0)
            a += ww[:, :, None] * xv.to(tl.float32)
        return a

    @triton.jit
    def _weight_taps(wt_ptr, p0, g, c, o, c_ok, o_ok, C, CO, P,
                     TP: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_N: tl.constexpr):
        """W2[(t, c), o] = W[g, p0+t, c, o] as a [TP*BLOCK_C, BLOCK_N] tile.

        ``wt`` is ``[G, P, C, CO]``.  The kernels do not assume what these taps
        are: a contraction with other taps (complex, several voices, the
        coverage) reads its own array here.
        """
        p = p0 + tl.arange(0, TP)
        rows = ((g * P + p)[:, None] * C + c[None, :]).to(tl.int64)     # [TP, BC]
        m = ((p < P)[:, None] & c_ok[None, :])[:, :, None] & o_ok[None, None, :]
        w3 = tl.load(wt_ptr + rows[:, :, None] * CO + o[None, None, :], mask=m, other=0.0)
        return tl.reshape(w3, (TP * BLOCK_C, BLOCK_N))

    @triton.autotune(configs=_FWD_CONFIGS, key=["C", "CO", "P", "SIZE"])
    @triton.jit
    def _fwd_kernel(xt_ptr, idx_ptr, w_ptr, wt_ptr, bias_ptr, y_ptr,
                    K, C, CO, G, P, B, SIZE,
                    HAS_BIAS: tl.constexpr, PRECISION: tl.constexpr, TP: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
        """y[b, g*CO + o, k] = bias + sum_p A_p[g, b, k, :] @ W[g, p]; y is [B, G*CO, K].

        TP taps are contracted by one ``tl.dot`` (reduction over (tap, channel)).
        """
        pid = tl.program_id(0)
        pid_n = tl.program_id(1)
        b = pid % B                      # batch fastest: the same idx / w tiles
        rest = pid // B                  # are reused from L2 across the batch
        g = rest % G
        kb = rest // G
        k = kb * BLOCK_M + tl.arange(0, BLOCK_M)
        k_ok = k < K
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_ok = offs_n < CO
        offs_c = tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for p0 in range(0, P, TP):
            for c0 in range(0, C, BLOCK_C):
                c = c0 + offs_c
                c_ok = c < C
                a = _gather_taps(xt_ptr, idx_ptr, w_ptr, p0, g, b, k, c, k_ok, c_ok, K, C, G, P,
                                 BLOCK_M, TP, BLOCK_C)
                w2 = _weight_taps(wt_ptr, p0, g, c, offs_n, c_ok, n_ok, C, CO, P,
                                  TP, BLOCK_C, BLOCK_N)
                a2 = tl.reshape(a, (BLOCK_M, TP * BLOCK_C))
                acc = tl.dot(a2.to(w2.dtype), w2, acc, input_precision=PRECISION)
        if HAS_BIAS:
            bv = tl.load(bias_ptr + g * CO + offs_n, mask=n_ok, other=0.0).to(tl.float32)
            acc += bv[None, :]
        ch = (b.to(tl.int64) * G + g) * CO + offs_n
        tl.store(y_ptr + ch[None, :] * K + k[:, None], acc.to(y_ptr.dtype.element_ty),
                 mask=k_ok[:, None] & n_ok[None, :])

    @triton.autotune(configs=_BWD_W_CONFIGS, key=["C", "CO", "P", "SIZE"])
    @triton.jit
    def _bwd_w_kernel(xt_ptr, idx_ptr, w_ptr, gy_ptr, part_ptr,
                      K, C, CO, G, P, B, N_SPLIT, SIZE,
                      PRECISION: tl.constexpr, TP: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
        """part[s, g, p, c, o] = sum over the rows of chunk s of A_p[g, r, c] * gy[g, r, o].

        One program: TP taps x one channel tile x one output tile, for gauge g
        and row chunk s; the product is [TP*BLOCK_C, BLOCK_M] @ [BLOCK_M, BLOCK_N].
        """
        pid0 = tl.program_id(0)          # tap block fastest: the programs of one
        pid1 = tl.program_id(1)          # row chunk run together and share gy in L2
        n_tb = tl.cdiv(P, TP)
        tb = pid0 % n_tb
        cn = pid0 // n_tb
        g = pid1 % G
        s = pid1 // G
        n_bn = tl.cdiv(CO, BLOCK_N)
        c = (cn // n_bn) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        o = (cn % n_bn) * BLOCK_N + tl.arange(0, BLOCK_N)
        o_ok = o < CO
        p0 = tb * TP
        n_kb = tl.cdiv(K, BLOCK_M)
        n_tiles = B * n_kb
        tps = tl.cdiv(n_tiles, N_SPLIT)
        t_lo = s * tps
        t_hi = tl.minimum(t_lo + tps, n_tiles)
        acc = tl.zeros((TP * BLOCK_C, BLOCK_N), dtype=tl.float32)
        for t in range(t_lo, t_hi):
            b = t % B
            kb = t // B
            k = kb * BLOCK_M + tl.arange(0, BLOCK_M)
            k_ok = k < K
            a = _gather_taps(xt_ptr, idx_ptr, w_ptr, p0, g, b, k, c, k_ok, c_ok, K, C, G, P,
                             BLOCK_M, TP, BLOCK_C)
            a2 = tl.reshape(a, (BLOCK_M, TP * BLOCK_C))
            ch = (b.to(tl.int64) * G + g) * CO + o
            gy = tl.load(gy_ptr + ch[None, :] * K + k[:, None],
                         mask=k_ok[:, None] & o_ok[None, :], other=0.0)
            acc = tl.dot(tl.trans(a2.to(gy.dtype)), gy, acc, input_precision=PRECISION)
        acc3 = tl.reshape(acc, (TP, BLOCK_C, BLOCK_N))
        p = p0 + tl.arange(0, TP)
        rows = (((s * G + g) * P + p).to(tl.int64)[:, None] * C + c[None, :])    # [TP, BC]
        m = ((p < P)[:, None] & c_ok[None, :])[:, :, None] & o_ok[None, None, :]
        tl.store(part_ptr + rows[:, :, None] * CO + o[None, None, :], acc3, mask=m)

    @triton.autotune(configs=_BWD_X_CONFIGS, key=["C", "CO", "P", "SIZE"],
                     reset_to_zero=["gx_ptr"])
    @triton.jit
    def _bwd_x_atomic_kernel(gy_ptr, idx_ptr, w_ptr, wt_ptr, gx_ptr,
                             K, C, CO, G, P, B, SIZE,
                             PRECISION: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
        """gx[b, idx_j, c] += w_j * (gy[g, b, k, :] @ W[g, p]^T); gx is fp32 [B, K, C]."""
        pid = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = pid % B
        rest = pid // B
        g = rest % G
        kb = rest // G
        k = kb * BLOCK_M + tl.arange(0, BLOCK_M)
        k_ok = k < K
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        offs_n = tl.arange(0, BLOCK_N)
        gx_base = gx_ptr + b.to(tl.int64) * K * C
        for p in range(0, P):
            h = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
            for n0 in range(0, CO, BLOCK_N):
                o = n0 + offs_n
                o_ok = o < CO
                ch = (b.to(tl.int64) * G + g) * CO + o
                gy = tl.load(gy_ptr + ch[None, :] * K + k[:, None],
                             mask=k_ok[:, None] & o_ok[None, :], other=0.0)
                wT = tl.load(wt_ptr + ((g * P + p) * C + c)[None, :] * CO + o[:, None],
                             mask=o_ok[:, None] & c_ok[None, :], other=0.0)
                h = tl.dot(gy, wT, h, input_precision=PRECISION)
            for j in tl.static_range(4):
                off = ((p * 4 + j) * G + g).to(tl.int64) * K + k
                src = tl.load(idx_ptr + off, mask=k_ok, other=0).to(tl.int64)
                ww = tl.load(w_ptr + off, mask=k_ok, other=0.0).to(tl.float32)
                tl.atomic_add(gx_base + src[:, None] * C + c[None, :], ww[:, None] * h,
                              mask=k_ok[:, None] & c_ok[None, :], sem="relaxed")

    @triton.autotune(configs=_BWD_X_CSR_CONFIGS, key=["C", "CO", "P", "SIZE"])
    @triton.jit
    def _bwd_x_csr_kernel(gyt_ptr, tptr_ptr, tbase_ptr, tk_ptr, tw_ptr, wt_ptr, gx_ptr,
                          K, C, CO, G, P, B, SIZE,
                          PRECISION: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
        """gx[b, c, s] = sum_{g,p} (sum_e tw_e gy[b, g, tk_e, :]) @ W[g, p]^T, no atomics.

        ``gyt`` is ``[B, G, K, CO]``; gx is written as ``[B, C, K]``.
        """
        pid = tl.program_id(0)
        pid_c = tl.program_id(1)
        b = pid % B
        sb = pid // B
        s = sb * BLOCK_M + tl.arange(0, BLOCK_M)
        s_ok = s < K
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C
        offs_n = tl.arange(0, BLOCK_N)
        GK1 = (G * K + 1).to(tl.int64)       # row stride of ptr [P, G*K + 1]
        s64 = s.to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
        for g in range(0, G):
            gy_base = gyt_ptr + (b.to(tl.int64) * G + g) * K * CO
            for p in range(0, P):
                base = tl.load(tbase_ptr + p)
                row = s64 + g * K + p * GK1
                start = tl.load(tptr_ptr + row, mask=s_ok, other=0)
                n = tl.load(tptr_ptr + row + 1, mask=s_ok, other=0) - start
                n_max = tl.max(n, axis=0)
                for n0 in range(0, CO, BLOCK_N):
                    o = n0 + offs_n
                    o_ok = o < CO
                    at = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                    for jj in range(0, n_max):
                        m = s_ok & (jj < n)
                        e = base + start.to(tl.int64) + jj
                        kk = tl.load(tk_ptr + e, mask=m, other=0).to(tl.int64)
                        ww = tl.load(tw_ptr + e, mask=m, other=0.0).to(tl.float32)
                        v = tl.load(gy_base + kk[:, None] * CO + o[None, :],
                                    mask=m[:, None] & o_ok[None, :], other=0.0)
                        at += ww[:, None] * v.to(tl.float32)
                    wT = tl.load(wt_ptr + ((g * P + p) * C + c)[None, :] * CO + o[:, None],
                                 mask=o_ok[:, None] & c_ok[None, :], other=0.0)
                    acc = tl.dot(at.to(wT.dtype), wT, acc, input_precision=PRECISION)
        tl.store(gx_ptr + (b.to(tl.int64) * C + c)[None, :] * K + s[:, None],
                 acc.to(gx_ptr.dtype.element_ty), mask=s_ok[:, None] & c_ok[None, :])


def _check_triton_inputs(x, weight, idx, w):
    if not x.is_cuda:
        raise ValueError("the Triton backend needs CUDA tensors.")
    if x.dtype not in _TRITON_DTYPES:
        raise ValueError(f"the Triton backend does not support {x.dtype}; use backend='torch'.")
    if idx.dtype != torch.int32 or not idx.is_contiguous() or not w.is_contiguous():
        raise ValueError("idx must be a contiguous int32 tensor and w contiguous.")


def _triton_forward(x, weight, bias, idx, w):
    B, C, K = x.shape
    G, _, CO, P = weight.shape
    xt = x.transpose(1, 2).contiguous()                       # [B, K, C]
    wt = weight.permute(0, 3, 1, 2).contiguous()              # [G, P, C, CO]
    y = torch.empty(B, G * CO, K, device=x.device, dtype=x.dtype)
    BLOCK_C, BLOCK_N = _block_nc(C, CO)

    def grid(meta):
        return (B * G * triton.cdiv(K, meta["BLOCK_M"]), triton.cdiv(CO, BLOCK_N))

    with torch.cuda.device(x.device):
        _fwd_kernel[grid](
            xt, idx, w, wt, bias if bias is not None else wt, y,
            K, C, CO, G, P, B, _size_class(B * K),
            HAS_BIAS=bias is not None, PRECISION=_precision(x.dtype),
            TP=1, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,     # see _taps_per_block
        )
    return y


def _n_split(x, n_rows, G, P, n_cn, C, CO):
    """Number of row chunks of the grad-W kernel: enough programs to fill the GPU.

    ``P`` is the number of tap blocks of the kernel."""
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    n = triton.cdiv(_SPLIT_PROGRAMS_PER_SM * sms, max(1, G * P * n_cn))
    n = max(1, min(n, triton.cdiv(n_rows, 8 * 128)))      # >= ~8 row tiles each
    # bound the buffer of partial sums to ~256 MB
    n = max(1, min(n, (2**26) // max(1, G * P * C * CO)))
    return int(n)


def _triton_backward(x, weight, idx, w, gy, need_x, need_w, deterministic, transposed):
    B, C, K = x.shape
    G, _, CO, P = weight.shape
    gy = gy.to(x.dtype).contiguous()
    wt = weight.permute(0, 3, 1, 2).contiguous()              # [G, P, C, CO]
    BLOCK_C, BLOCK_N = _block_nc(C, CO)
    prec = _precision(x.dtype)
    size = _size_class(B * K)
    gx = gW = None
    with torch.cuda.device(x.device):
        if need_w:
            xt = x.transpose(1, 2).contiguous()
            n_cn = triton.cdiv(C, BLOCK_C) * triton.cdiv(CO, BLOCK_N)
            tp = _taps_per_block(P, BLOCK_C)
            n_tb = triton.cdiv(P, tp)
            n_split = _n_split(x, B * K, G, n_tb, n_cn, C, CO)
            part = torch.empty(n_split, G, P, C, CO, device=x.device, dtype=torch.float32)
            _bwd_w_kernel[(n_tb * n_cn, G * n_split)](
                xt, idx, w, gy, part, K, C, CO, G, P, B, n_split, size,
                PRECISION=prec, TP=tp, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
            )
            del xt
            gW = part.sum(0).permute(0, 2, 3, 1).to(weight.dtype)   # [G, C, CO, P]
            del part
        if need_x:
            if deterministic:
                tmap = (transposed() if transposed is not None
                        else build_transposed_map(idx, w, K))
                tptr, tbase, tk, tw = tmap
                gyt = gy.view(B, G, CO, K).transpose(2, 3).contiguous()   # [B, G, K, CO]
                gx = torch.empty(B, C, K, device=x.device, dtype=x.dtype)

                def grid(meta):
                    return (B * triton.cdiv(K, meta["BLOCK_M"]), triton.cdiv(C, BLOCK_C))

                _bwd_x_csr_kernel[grid](
                    gyt, tptr, tbase, tk, tw, wt, gx, K, C, CO, G, P, B, size,
                    PRECISION=prec, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
                )
            else:
                gxt = torch.zeros(B, K, C, device=x.device, dtype=torch.float32)

                def grid(meta):
                    return (B * G * triton.cdiv(K, meta["BLOCK_M"]), triton.cdiv(C, BLOCK_C))

                _bwd_x_atomic_kernel[grid](
                    gy, idx, w, wt, gxt, K, C, CO, G, P, B, size,
                    PRECISION=prec, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
                )
                gx = gxt.transpose(1, 2).to(x.dtype)
    return gx, gW


# ===========================================================================
# One autograd Function, two backends
# ===========================================================================

@dataclass(frozen=True)
class _Options:
    use_triton: bool
    deterministic: bool
    transposed: Optional[Callable[[], tuple]]


class GaugeConvFunction(torch.autograd.Function):
    """``x [B, C, K] -> y [B, G*Co, K]``; dispatches to Triton or to PyTorch.

    ``idx`` and ``w`` are the tap geometry of :func:`tap_geometry`.  The
    choice of backend is made by :func:`gauge_conv` and carried in ``opts``.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, idx, w, opts):
        if opts.use_triton:
            y = _triton_forward(x, weight, bias, idx, w)
        else:
            y = _torch_forward(x, weight, bias, idx, w)
        ctx.opts = opts
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(x, weight, idx, w)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, weight, idx, w = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        opts = ctx.opts
        gx = gW = gb = None
        if need_x or need_w:
            if opts.use_triton and not torch.is_grad_enabled():
                gx, gW = _triton_backward(x, weight, idx, w, gy, need_x, need_w,
                                          opts.deterministic, opts.transposed)
            else:
                # Grad mode is on in a backward pass only with create_graph=True
                # (double backward, e.g. a gradient penalty).  The Triton
                # kernels write into buffers outside the graph, so that case
                # runs the PyTorch backward, whose operations autograd records.
                gx, gW = _torch_backward(x, weight, idx, w, gy, need_x, need_w)
        if need_b:
            acc = torch.promote_types(gy.dtype, torch.float32)   # fp32 for 16-bit
            gb = gy.sum(dim=(0, 2), dtype=acc).to(ctx.bias_dtype)
        return gx, gW, gb, None, None, None


def _autocast_dtype(device_type):
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except TypeError:  # torch < 2.4
        if device_type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None


_WARNED_NO_TRITON = False


def _warn_no_triton():
    global _WARNED_NO_TRITON
    if not _WARNED_NO_TRITON:
        _WARNED_NO_TRITON = True
        warnings.warn(
            "HealPixConv runs on CUDA without Triton: the PyTorch backend keeps "
            "the memory low but its backward pass is slow on GPUs. Install the "
            "optional 'triton' package (pip install healpix-analyse[triton]) "
            "for the fused kernels.",
            RuntimeWarning,
            stacklevel=5,       # the caller of HealPixConv.forward
        )


def _compute_dtype(x):
    """dtype the Triton kernels run in: the autocast dtype for float32 data."""
    dt = _autocast_dtype("cuda") if x.is_cuda else None
    return dt if (dt is not None and x.dtype == torch.float32) else x.dtype


def _use_triton(x, backend):
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == "torch":
        return False
    if x.is_cuda and not _HAS_TRITON and _compute_dtype(x) in _TRITON_DTYPES:
        _warn_no_triton()
    ok = _HAS_TRITON and x.is_cuda and _compute_dtype(x) in _TRITON_DTYPES
    if backend == "triton" and not ok:
        raise RuntimeError(
            "backend='triton' needs Triton, a CUDA tensor and float32, float16 "
            "or bfloat16 data (float64 runs on backend='torch')."
        )
    return ok


def gauge_conv(x, weight, bias, idx, w, *, backend="auto", deterministic=True,
               transposed=None):
    """Gauge-equivariant convolution on precomputed tap geometry.

    Parameters
    ----------
    x : Tensor [B, C, K]
    weight : Tensor [G, C, Co, P]
    bias : Tensor [G*Co] or None
    idx, w : Tensor [P, 4, G, K]
        Tap geometry (:func:`tap_geometry`); ``idx`` int32 on the device of ``x``.
    backend : {"auto", "triton", "torch"}
        ``"auto"`` runs Triton on CUDA for float32/float16/bfloat16 when Triton
        is installed, and PyTorch otherwise (CPU, float64).
    deterministic : bool, default True
        Triton only: compute grad x by a gather on the transposed map (no
        atomics, identical results at every run); ``False`` uses fp32
        atomics.  ``torch.are_deterministic_algorithms_enabled()`` selects
        the gather even with ``False``.  The PyTorch backend ignores it: on
        CUDA its ``index_add_`` is deterministic only under
        ``torch.use_deterministic_algorithms(True)``.  A double backward
        (``create_graph=True``) always runs the PyTorch backward.
    transposed : callable or None
        Returns the transposed map (:func:`build_transposed_map`) of
        ``idx, w``, so that the caller can cache it; called only by a
        deterministic Triton backward.  None builds the map at every
        backward pass.

    Returns
    -------
    Tensor [B, G*Co, K]
    """
    use_triton = _use_triton(x, backend)
    if use_triton:
        x = x.to(_compute_dtype(x))           # autocast: 16-bit inputs, fp32 accumulation
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        x = x.contiguous()
    elif weight.dtype != x.dtype:
        x = x.to(weight.dtype)
    deterministic = bool(deterministic) or torch.are_deterministic_algorithms_enabled()
    opts = _Options(use_triton, deterministic, transposed)
    return GaugeConvFunction.apply(x, weight, bias, idx, w, opts)
