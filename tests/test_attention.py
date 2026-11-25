"""Tests of healpix_analyse.attention (issue #109, section 3)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import healpix_analyse.attention as att
from healpix_analyse.attention import (
    DEFAULT_WINDOW_SHIFT,
    WINDOW_SHIFTS,
    HealPixNeighbourhoodAttention,
    HealPixWindowAttention,
)
from healpix_analyse._attention_backend import (
    _HAS_TRITON,
    neighbourhood_attention,
    neighbourhood_attention_naive,
)

HAS_CUDA = torch.cuda.is_available()
needs_triton = pytest.mark.skipif(not (_HAS_TRITON and HAS_CUDA), reason="needs Triton and CUDA")
F64 = torch.float64


def _regional_ids(level=3, seed=0):
    """An unsorted regional domain with incomplete NESTED blocks."""
    rng = np.random.default_rng(seed)
    ids = np.arange(37, 37 + 12 * 4 ** level // 3)
    ids = ids[rng.random(ids.size) > 0.1]
    return rng.permutation(ids)


def _na(level=3, cell_ids=None, G=1, kernel_sz=3, dtype=F64, device="cpu", **kw):
    return HealPixNeighbourhoodAttention(
        level, 16, 2, kernel_sz=kernel_sz, n_gauges=G, cell_ids=cell_ids,
        device=device, dtype=dtype, cache_dir=None, **kw)


def _qkvb(na, B=2, D=8, dtype=F64, device="cpu", seed=1):
    g = torch.Generator().manual_seed(seed)
    H, N = na.num_heads, na.N
    t = [torch.randn(B, H, N, D, generator=g, dtype=dtype).to(device).requires_grad_()
         for _ in range(3)]
    b = torch.randn(H, na.P, generator=g, dtype=dtype).to(device).requires_grad_()
    return (*t, b)


def _rot90_taps(k):
    """Tap permutation of a 90 deg rotation of the k x k template (as in the conv tests)."""
    xx, yy = np.meshgrid(np.arange(k) - k // 2, np.arange(k) - k // 2)
    xx, yy = xx.ravel(), yy.ravel()
    where = {(x, y): p for p, (x, y) in enumerate(zip(xx, yy))}
    return np.array([where[(-yy[p], xx[p])] for p in range(xx.size)])


def _dense_reference(q, k, v, bias, idx, w, valid):
    """Dense masked attention: each query against all P*N interpolated keys.

    The interpolation of each (gauge, tap) is a dense [N, N] matrix; the
    attention matrix [N, P*N] is masked to the P taps of the query.
    """
    P, _, G, N = idx.shape
    B, H, _, D = q.shape
    outs = []
    for g in range(G):
        M = torch.zeros(P, N, N, dtype=q.dtype)
        for p in range(P):
            for j in range(4):
                M[p].index_put_((torch.arange(N), idx[p, j, g].long()),
                                w[p, j, g].to(q.dtype), accumulate=True)
        ks = torch.einsum("pnm,bhmd->bhpnd", M, k).reshape(B, H, P * N, D)
        vs = torch.einsum("pnm,bhmd->bhpnd", M, v).reshape(B, H, P * N, D)
        allowed = torch.zeros(N, P, N, dtype=torch.bool)
        allowed[torch.arange(N), :, torch.arange(N)] = True if valid is None \
            else valid[:, g].T.bool()
        mask = torch.full((H, N, P * N), float("-inf"), dtype=q.dtype)
        bias_cols = bias[:, :, None].expand(H, P, N).reshape(H, 1, P * N).expand(H, N, P * N)
        mask = torch.where(allowed.reshape(1, N, P * N), bias_cols, mask)
        outs.append(F.scaled_dot_product_attention(q, ks, vs, attn_mask=mask))
    return torch.stack(outs, dim=1)                       # [B, G, H, N, D]


# ---------------------------------------------------------------------------
# Neighbourhood attention: values and gradients
# ---------------------------------------------------------------------------

CASES = {
    "full_G1": dict(level=2, cell_ids=None, G=1),
    "full_G2_k5": dict(level=2, cell_ids=None, G=2, kernel_sz=5),
    "regional_unsorted": dict(level=3, cell_ids="regional", G=1),
    "regional_unsorted_G2": dict(level=3, cell_ids="regional", G=2),
}


def _make(case):
    kw = dict(CASES[case])
    if kw["cell_ids"] == "regional":
        kw["cell_ids"] = _regional_ids(kw["level"])
    return _na(**kw)


@pytest.mark.parametrize("case", list(CASES))
def test_neighbourhood_matches_dense_masked_attention(case):
    na = _make(case)
    if CASES[case]["cell_ids"] is not None:
        assert na._valid is not None and not bool(na._valid.all())
    q, k, v, b = _qkvb(na)
    out = neighbourhood_attention(q, k, v, b, na._tap_idx, na._tap_w, na._valid, backend="torch")
    ref = _dense_reference(q, k, v, b, na._tap_idx, na._tap_w, na._valid)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)
    g = torch.randn_like(out)
    ga = torch.autograd.grad(out, (q, k, v, b), g)
    gr = torch.autograd.grad(ref, (q, k, v, b), g)
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r, rtol=0, atol=1e-11)


@pytest.mark.parametrize("case", ["full_G2_k5", "regional_unsorted"])
def test_neighbourhood_matches_naive_gather(case):
    na = _make(case)
    q, k, v, b = _qkvb(na)
    out = neighbourhood_attention(q, k, v, b, na._tap_idx, na._tap_w, na._valid, backend="torch")
    ref = neighbourhood_attention_naive(q, k, v, b, na._tap_idx, na._tap_w, na._valid)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)


def test_neighbourhood_gradcheck():
    na = _na(level=1, cell_ids=np.array([5, 0, 1, 2, 3, 7, 12, 13, 14, 40]), G=2)
    q, k, v, b = _qkvb(na, B=1, D=2)

    def f(q, k, v, b):
        return neighbourhood_attention(q, k, v, b, na._tap_idx, na._tap_w, na._valid,
                                       backend="torch")

    assert torch.autograd.gradcheck(f, (q, k, v, b))
    assert torch.autograd.gradgradcheck(f, (q, k, v, b), fast_mode=True)


def test_module_gradcheck_full_sphere():
    na = HealPixNeighbourhoodAttention(1, 4, 2, device="cpu", dtype=F64, cache_dir=None)
    x = torch.randn(1, 4, na.N, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: na(x), (x,), fast_mode=True)


def test_neighbourhood_gauge_equivariance():
    """Rotating the gauge by pi/2 is a permutation of the taps.

    With G = 4 the stencil of gauge g + 2 is the stencil of gauge g rotated by
    pi/2: the output of gauge g + 2 equals the output of gauge g with the
    bias table permuted accordingly.
    """
    na = _na(level=2, G=4, gauge_type="projected_ref", singularity_lonlat=(20.0, 5.0))
    idx, w = na._tap_idx, na._tap_w
    P = na.P
    perm = torch.as_tensor(_rot90_taps(3))
    q, k, v, b = _qkvb(na)
    out = neighbourhood_attention(q, k, v, b, idx, w, None, backend="torch")
    b2 = torch.empty_like(b)
    b2[:, perm] = b
    out2 = neighbourhood_attention(q, k, v, b2, idx, w, None, backend="torch")
    for g in (0, 1):
        torch.testing.assert_close(out[:, g + 2], out2[:, g], rtol=0, atol=1e-9)
    # the module: one gauge block of the output per gauge
    x = torch.randn(2, 16, na.N, dtype=F64)
    y = na(x).view(2, 4, 16, na.N)
    with torch.no_grad():
        na.pos_bias.copy_(na.pos_bias[:, torch.argsort(perm)])
    y2 = na(x).view(2, 4, 16, na.N)
    torch.testing.assert_close(y[:, 2:], y2[:, :2], rtol=0, atol=1e-9)


def test_regional_order_equivariance():
    """The same domain in two orders gives the same field."""
    ids = _regional_ids(3)
    perm = np.random.default_rng(3).permutation(ids.size)
    a = _na(level=3, cell_ids=ids)
    bmod = _na(level=3, cell_ids=ids[perm])
    bmod.load_state_dict(a.state_dict())
    x = torch.randn(2, 16, ids.size, dtype=F64)
    torch.testing.assert_close(a(x)[..., perm], bmod(x[..., perm]), rtol=0, atol=1e-12)


def test_outside_options():
    ids = _regional_ids(3)
    m = _na(level=3, cell_ids=ids)
    c = _na(level=3, cell_ids=ids, outside="centre")
    assert m._valid is not None and c._valid is None
    c.load_state_dict(m.state_dict())
    x = torch.randn(1, 16, ids.size, dtype=F64)
    inner = m._valid.all(dim=0).all(dim=0)                # cells with no masked tap
    ym, yc = m(x), c(x)
    torch.testing.assert_close(ym[..., inner], yc[..., inner], rtol=0, atol=1e-12)
    assert not torch.allclose(ym[..., ~inner], yc[..., ~inner])
    # full sphere: nothing to mask
    assert _na(level=2)._valid is None
    with pytest.raises(ValueError):
        _na(level=2, outside="nan")


def test_module_float32_and_shapes():
    na = _na(level=2, G=2, dtype=torch.float32)
    y = na(torch.randn(3, 16, na.N))
    assert y.shape == (3, 32, na.N) and y.dtype == torch.float32
    with pytest.raises(ValueError):
        na(torch.randn(3, 8, na.N))
    with pytest.raises(ValueError):
        HealPixNeighbourhoodAttention(2, 16, 3, device="cpu", cache_dir=None)


# ---------------------------------------------------------------------------
# Neighbourhood attention: Triton (CUDA)
# ---------------------------------------------------------------------------

@needs_triton
@pytest.mark.parametrize("case", list(CASES))
def test_triton_matches_torch(case):
    na = _make(case).to(device="cuda", dtype=torch.float32)
    q, k, v, b = _qkvb(na, D=16, dtype=torch.float32, device="cuda")
    out = neighbourhood_attention(q, k, v, b, na._tap_idx, na._tap_w, na._valid, backend="triton",
                                  transposed=na._transposed_map)
    ref = neighbourhood_attention(*(t.double() for t in (q, k, v, b)), na._tap_idx,
                                  na._tap_w.double(), na._valid, backend="torch")
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-5)
    g = torch.randn_like(out)
    ga = torch.autograd.grad(out, (q, k, v, b), g)
    qd, kd, vd, bd = (t.detach().double().requires_grad_() for t in (q, k, v, b))
    ref = neighbourhood_attention(qd, kd, vd, bd, na._tap_idx, na._tap_w.double(), na._valid,
                                  backend="torch")
    gr = torch.autograd.grad(ref, (qd, kd, vd, bd), g.double())
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a.double(), r, rtol=0, atol=2e-5 * max(1.0, r.abs().max().item()))


@needs_triton
def test_triton_backward_is_bitwise_reproducible():
    na = _make("regional_unsorted_G2").to(device="cuda", dtype=torch.float32)
    x = torch.randn(2, 16, na.N, device="cuda", requires_grad=True)
    grads = []
    for _ in range(2):
        na.zero_grad()
        x.grad = None
        na(x).square().sum().backward()
        grads.append([x.grad.clone()] + [p.grad.clone() for p in na.parameters()])
    for a, r in zip(*grads):
        assert torch.equal(a, r)


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_16bit_autocast(dtype):
    na = _make("full_G1").to(device="cuda", dtype=torch.float32)
    x = torch.randn(2, 16, na.N, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        y = na(x)
    ref = na.to(F64)(x.double())
    torch.testing.assert_close(y.double(), ref, rtol=0, atol=5e-2)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


@needs_triton
def test_triton_double_backward_matches_torch():
    na = _make("full_G1").to(device="cuda", dtype=torch.float32)
    x = torch.randn(1, 16, na.N, device="cuda", requires_grad=True)
    pens = []
    for backend in ("triton", "torch"):
        na.backend = backend
        (gx,) = torch.autograd.grad(na(x).square().sum(), x, create_graph=True)
        (ggx,) = torch.autograd.grad(gx.square().sum(), x)
        pens.append(ggx)
    torch.testing.assert_close(pens[0], pens[1], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Window attention
# ---------------------------------------------------------------------------

def _window_dense_reference(m, x):
    """Full attention over the N cells with a dense mask built from the plan."""
    N, C = m.N, m.embed_dim
    H, D = m.num_heads, m.head_dim
    if m._kv_index is None:      # reshape path: blocks of consecutive positions
        T = m.window_len
        win = torch.arange(N) // T
        allowed = win[:, None] == win[None, :]
    else:
        allowed = torch.zeros(N, N, dtype=torch.bool)
        if m.fast:
            q_win = torch.arange(N) // m.window_len
        else:
            q_win = m._q_inv // m._q_index.shape[1]
        kv = m._kv_index
        for i in range(N):
            keys = kv[q_win[i]]
            allowed[i, keys[keys < N]] = True
    qkv = m.qkv(x.transpose(1, 2)).view(x.shape[0], N, 3, H, D).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=allowed)
    return m.proj(o.transpose(1, 2).reshape(x.shape[0], N, C)).transpose(1, 2)


@pytest.mark.parametrize("shift", WINDOW_SHIFTS)
@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("domain", ["full", "regional"])
def test_window_matches_dense_masked_attention(shift, shifted, domain):
    ids = None if domain == "full" else _regional_ids(3)
    m = HealPixWindowAttention(3, 16, 2, 1, cell_ids=ids, shift=shift, shifted=shifted,
                               device="cpu", dtype=F64)
    assert m.fast == (domain == "full")
    x = torch.randn(2, 16, m.N, dtype=F64, requires_grad=True)
    y = m(x)
    ref = _window_dense_reference(m, x)
    torch.testing.assert_close(y, ref, rtol=0, atol=1e-12)
    g = torch.randn_like(y)
    torch.testing.assert_close(torch.autograd.grad(y, x, g)[0],
                               torch.autograd.grad(ref, x, g)[0], rtol=0, atol=1e-12)


def test_window_levels_and_default_shift():
    assert DEFAULT_WINDOW_SHIFT in WINDOW_SHIFTS
    assert HealPixWindowAttention(3, 8, 2, 1, device="cpu").shift == DEFAULT_WINDOW_SHIFT
    a = HealPixWindowAttention(3, 8, 2, 1, shift="alternate_level", device="cpu")
    b = HealPixWindowAttention(3, 8, 2, 1, shift="alternate_level", shifted=True, device="cpu")
    n = HealPixWindowAttention(3, 8, 2, 1, shift="none", shifted=True, device="cpu")
    assert (a.window_len, b.window_len, n.window_len) == (4, 16, 4)
    assert (a.n_windows, b.n_windows) == (192, 48)
    c = HealPixWindowAttention(4, 8, 2, 2, shift="coarse_centred", device="cpu")
    assert c.window_len == 16 and c.key_len > 16 and c._kv_mask is not None
    # every window holds its own block, and the windows overlap
    blocks = torch.arange(c.N).view(-1, 16)
    for i in range(0, c.n_windows, 97):
        assert set(blocks[i].tolist()) <= set(c._kv_index[i].tolist())
    with pytest.raises(ValueError):
        HealPixWindowAttention(3, 8, 2, 3, shift="alternate_level", shifted=True, device="cpu")
    with pytest.raises(ValueError):
        HealPixWindowAttention(3, 8, 2, 1, shift="shifted", device="cpu")
    with pytest.raises(ValueError):
        HealPixWindowAttention(3, 8, 2, 0, device="cpu")


def test_window_sorted_complete_domain_takes_the_reshape():
    ids = np.arange(64, 64 + 16 * 10)           # 10 complete level-2 blocks, sorted
    m = HealPixWindowAttention(3, 8, 2, 2, cell_ids=ids, shift="none", device="cpu", dtype=F64)
    assert m.fast and m._q_index is None and m._kv_index is None
    perm = np.random.default_rng(0).permutation(ids.size)
    u = HealPixWindowAttention(3, 8, 2, 2, cell_ids=ids[perm], shift="none", device="cpu",
                               dtype=F64)
    assert not u.fast
    u.load_state_dict(m.state_dict())
    x = torch.randn(1, 8, ids.size, dtype=F64)
    torch.testing.assert_close(m(x)[..., perm], u(x[..., perm]), rtol=0, atol=1e-12)


def test_window_default_is_a_module_constant(monkeypatch):
    monkeypatch.setattr(att, "DEFAULT_WINDOW_SHIFT", "coarse_centred")
    assert HealPixWindowAttention(3, 8, 2, 1, device="cpu").shift == "coarse_centred"


# ---------------------------------------------------------------------------
# Window attention: Triton (CUDA)
# ---------------------------------------------------------------------------

def _window(shift, domain, shifted=False, device="cpu", dtype=F64, level=4, wl=2):
    ids = None if domain == "full" else _regional_ids(level)
    return HealPixWindowAttention(level, 32, 2, wl, cell_ids=ids, shift=shift, shifted=shifted,
                                  device=device, dtype=dtype)


@needs_triton
@pytest.mark.parametrize("shift,shifted", [("none", False), ("alternate_level", True),
                                           ("coarse_centred", False)])
@pytest.mark.parametrize("domain", ["full", "regional"])
def test_window_triton_matches_torch(shift, shifted, domain):
    from healpix_analyse._attention_backend import window_attention
    m = _window(shift, domain, shifted, device="cuda", dtype=torch.float32)
    plan = m.plan()
    qkv = torch.randn(2, m.N, 3, 2, 16, device="cuda", requires_grad=True)
    out = window_attention(qkv, plan, backend="triton")
    q64 = qkv.detach().double().requires_grad_()
    ref = window_attention(q64, plan, backend="torch")
    torch.testing.assert_close(out.double(), ref, rtol=0, atol=1e-5)
    g = torch.randn_like(out)
    (ga,) = torch.autograd.grad(out, qkv, g)
    (gr,) = torch.autograd.grad(ref, q64, g.double())
    torch.testing.assert_close(ga.double(), gr, rtol=0, atol=2e-5 * max(1.0, gr.abs().max().item()))


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_window_triton_16bit(dtype):
    m = _window("coarse_centred", "regional", device="cuda", dtype=torch.float32)
    x = torch.randn(2, 32, m.N, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        y = m(x)
    ref = m.to(F64)(x.double())
    torch.testing.assert_close(y.double(), ref, rtol=0, atol=5e-2)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


@needs_triton
def test_window_triton_backward_is_bitwise_reproducible():
    m = _window("coarse_centred", "full", device="cuda", dtype=torch.float32)
    x = torch.randn(2, 32, m.N, device="cuda", requires_grad=True)
    grads = []
    for _ in range(2):
        m.zero_grad()
        x.grad = None
        m(x).square().sum().backward()
        grads.append([x.grad.clone()] + [p.grad.clone() for p in m.parameters()])
    for a, r in zip(*grads):
        assert torch.equal(a, r)


@needs_triton
def test_window_triton_double_backward():
    m = _window("coarse_centred", "regional", device="cuda", dtype=torch.float32)
    x = torch.randn(1, 32, m.N, device="cuda", requires_grad=True)
    pens = []
    for backend in ("triton", "torch"):
        m.backend = backend
        (gx,) = torch.autograd.grad(m(x).square().sum(), x, create_graph=True)
        (ggx,) = torch.autograd.grad(gx.square().sum(), x)
        pens.append(ggx)
    torch.testing.assert_close(pens[0], pens[1], rtol=1e-3, atol=1e-4)
