"""Strided and transposed gauge convolutions, and torch.func support (issue #109).

Reference: the dense stencil operator ``A[g, p, k, s]`` (stencil centre k,
input cell s), assembled independently of the tap layout from the geometry
helpers of ``convol.py``, with the centres on the parent cells for
``stride=2``.  ``S x = sum_{g,p} W A x`` and ``S^T y = sum A^T W^T y`` are
computed from it with ``einsum`` and autograd.

* PyTorch backend: float64, outputs and gradients to 1e-12.
* Triton backend (CUDA): float32 to 1e-5; skipped without CUDA or Triton.
"""

from __future__ import annotations

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_analyse import _conv_backend as cb
from healpix_analyse.convol import (
    HealPixConv,
    HealPixConvTranspose,
    _bind_support_batched,
    _build_rotation_matrices,
    _geometry_cache_key,
    _get_interp_weights,
    _local_kernel_grid,
)
from healpix_analyse.down import HealPixDown

needs_triton = pytest.mark.skipif(
    not cb.triton_available(), reason="needs CUDA and the optional triton package"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ragged_patch(level, m, drop=0.15, seed=3):
    """A NESTED patch of 4**m cells with a fraction dropped (incomplete
    parents), in a shuffled order."""
    n = 4**m
    start = 4 * 4**level + (4**level // 2 // n) * n
    ids = np.arange(start, start + n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ids = ids[rng.random(n) > drop]
    return ids[rng.permutation(ids.size)]


CASES = {
    "full": dict(level=3, cell_ids=None, G=4, k=3, B=2, C=3, Co=2, gauge="phi"),
    "regional": dict(level=5, cell_ids=ragged_patch(5, 4), G=2, k=5, B=3, C=2, Co=3,
                     gauge="projected_ref"),
}


def make(cs, stride, dtype=torch.float64, device="cpu", transpose=False, **kw):
    torch.manual_seed(0)
    cls = HealPixConvTranspose if transpose else HealPixConv
    c_in, c_out = (cs["Co"], cs["C"]) if transpose else (cs["C"], cs["Co"])
    conv = cls(cs["level"], c_in, c_out, kernel_sz=cs["k"], n_gauges=cs["G"],
               gauge_type=cs["gauge"], cell_ids=cs["cell_ids"], device=device,
               dtype=dtype, cache_dir=None, stride=stride, **kw)
    with torch.no_grad():
        conv.weight.normal_()
        conv.bias.normal_()
    return conv


def dense_operator(conv, cell_ids):
    """A [G, P, K_out, K_in] (input order) and the centres, built from scratch."""
    level, G, P, k = conv.level, conv.G, conv.P, conv.kernel_sz
    ids_in = (np.arange(12 * 4**level, dtype=np.int64) if cell_ids is None
              else np.asarray(cell_ids, dtype=np.int64))
    if conv.stride == 2:
        centres, lev_c = np.unique(ids_in // 4), level - 1
    else:
        centres, lev_c = ids_in, level
    K_out, K_in = centres.size, ids_in.size
    lon, lat = healpix_geo.nested.healpix_to_lonlat(centres.tolist(), lev_c,
                                                   ellipsoid=conv.ellipsoid)
    th = np.deg2rad(90.0 - np.asarray(lat, dtype=np.float64))
    ph = np.deg2rad(np.asarray(lon, dtype=np.float64))
    R = _build_rotation_matrices(th, ph, G, conv.gauge_type, "cpu", torch.float64,
                                 ref_direction=conv.ref_direction)
    vec = torch.as_tensor(_local_kernel_grid(k, 2**level), dtype=torch.float64)
    flat = torch.einsum("kgij,pj->kgpi", R, vec).reshape(-1, 3)
    idx, w = _get_interp_weights(2**level, flat, True, "cpu", torch.float64)
    idx = idx.view(4, K_out, G, P).permute(2, 0, 1, 3).reshape(G, 4, K_out * P)
    w = w.view(4, K_out, G, P).permute(2, 0, 1, 3).reshape(G, 4, K_out * P)
    pos, wn = _bind_support_batched(idx.clone(), w.clone(),
                                    torch.as_tensor(np.sort(ids_in)), k, K_out, P, "cpu")
    order = torch.as_tensor(np.argsort(ids_in))          # sorted position -> input index
    A = torch.zeros(G, P, K_out, K_in, dtype=torch.float64)
    rows = torch.arange(K_out).repeat_interleave(P)
    taps = torch.arange(P).repeat(K_out)
    for g in range(G):
        for j in range(4):
            A[g].index_put_((taps, rows, order[pos[g, j]]), wn[g, j], accumulate=True)
    return A, centres


def ref_s(A, W, b, x):
    y = torch.einsum("gpks,bcs,gcop->bgok", A, x, W).flatten(1, 2)
    return y if b is None else y + b.view(1, -1, 1)


def ref_st(A, W, b, y):
    G, CO = W.shape[0], W.shape[2]
    yy = y.view(y.shape[0], G, CO, -1)
    x = torch.einsum("gpks,bgok,gcop->bcs", A, yy, W)
    return x if b is None else x + b.view(1, -1, 1)


def grads(fn, conv, x0):
    """Output and the gradients (input, weight, bias) of sum(out * sin(arange))."""
    x = x0.clone().requires_grad_(True)
    conv.zero_grad(set_to_none=True)
    y = fn(x)
    gy = torch.sin(torch.arange(y.numel(), dtype=torch.float64, device=y.device)).view_as(y)
    (y * gy.to(y.dtype)).sum().backward()
    return (y.detach(), x.grad.detach(), conv.weight.grad.detach().clone(),
            conv.bias.grad.detach().clone())


def rel_errs(got, ref):
    return [float((a.double().cpu() - b.double().cpu()).abs().max() / b.double().cpu().abs().max())
            for a, b in zip(got, ref)]


# ---------------------------------------------------------------------------
# Strided convolution: definition, cells, consistency with conv + Down
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_strided_matches_dense_reference(case):
    cs = CASES[case]
    conv = make(cs, 2)
    A, centres = dense_operator(conv, cs["cell_ids"])
    np.testing.assert_array_equal(conv.cell_ids_out, centres)
    assert conv.K_out == centres.size and conv.level_out == cs["level"] - 1
    x0 = torch.randn(cs["B"], cs["C"], conv.K, dtype=torch.float64)
    got = grads(conv, conv, x0)
    ref = grads(lambda x: ref_s(A, conv.weight, conv.bias, x), conv, x0)
    assert got[0].shape == (cs["B"], cs["G"] * cs["Co"], centres.size)
    assert max(rel_errs(got, ref)) < 1e-12


def test_strided_full_sphere_cells_and_down_order():
    conv = HealPixConv(3, 1, 1, stride=2, device="cpu", cache_dir=None)
    np.testing.assert_array_equal(conv.cell_ids_out, np.arange(12 * 4**2))
    ids = ragged_patch(5, 4)
    conv = HealPixConv(5, 1, 1, stride=2, cell_ids=ids, device="cpu", cache_dir=None)
    down = HealPixDown(5, mode="mean", cell_ids=ids, device="cpu")
    np.testing.assert_array_equal(conv.cell_ids_out, down.cell_ids_out)


@pytest.mark.parametrize("k", [1, 3])
def test_strided_converges_to_conv_then_down(k):
    """On a smooth field the strided layer and conv + HealPixDown(mean) agree
    at first order in the cell size (stencil at the parent centre vs the mean
    of the 4 children)."""
    errs = []
    for level in (4, 5, 6):
        ids = np.arange(12 * 4**level)
        lon, lat = healpix_geo.nested.healpix_to_lonlat(ids.tolist(), level, ellipsoid="WGS84")
        lon, lat = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
        x = torch.as_tensor(np.cos(lat) ** 2 * np.sin(3 * lon) + 0.5 * np.sin(2 * lat)).view(1, 1, -1)
        s2 = HealPixConv(level, 1, 1, kernel_sz=k, stride=2, device="cpu", dtype=torch.float64,
                         cache_dir=None)
        s1 = HealPixConv(level, 1, 1, kernel_sz=k, device="cpu", dtype=torch.float64,
                         cache_dir=None)
        s1.weight.data.copy_(s2.weight.data)
        down = HealPixDown(level, mode="mean", device="cpu", dtype=torch.float64)
        with torch.no_grad():
            a = s2(x)[0, 0]
            b = down(s1(x)[0])[0][0]
        errs.append(float((a - b).abs().max() / b.abs().max()))
    assert errs[0] < 3e-2
    assert errs[1] < 0.6 * errs[0] and errs[2] < 0.6 * errs[1], errs


def test_strided_arguments():
    with pytest.raises(ValueError, match="stride"):
        HealPixConv(2, 1, 1, stride=3, device="cpu", cache_dir=None)
    with pytest.raises(ValueError, match="level >= 1"):
        HealPixConv(0, 1, 1, stride=2, device="cpu", cache_dir=None)
    with pytest.raises(ValueError, match="missing"):
        HealPixConv(2, 1, 1, stride=2, missing="local_mean", device="cpu", cache_dir=None)
    conv = HealPixConv(2, 1, 1, stride=2, device="cpu", cache_dir=None)
    assert "stride=2" in repr(conv)
    with pytest.raises(ValueError, match="pixels"):
        conv(torch.zeros(1, 1, conv.K_out))


def test_strided_cache(tmp_path):
    ids = ragged_patch(4, 3)
    key1 = _geometry_cache_key(16, 3, 2, "phi", None, True, ids, "WGS84")
    assert key1 == _geometry_cache_key(16, 3, 2, "phi", None, True, ids, "WGS84", stride=1)
    assert key1 != _geometry_cache_key(16, 3, 2, "phi", None, True, ids, "WGS84", stride=2)
    kw = dict(n_gauges=2, cell_ids=ids, device="cpu", dtype=torch.float64, cache_dir=tmp_path)
    a = HealPixConv(4, 1, 1, stride=2, **kw)
    b = HealPixConv(4, 1, 1, stride=2, **kw)            # from the disk cache
    c = HealPixConv(4, 1, 1, stride=1, **kw)            # its own entry
    assert len(list(tmp_path.glob("*.pt"))) == 2
    assert torch.equal(a._tap_idx, b._tap_idx) and torch.equal(a._tap_w, b._tap_w)
    assert c._tap_idx.shape[-1] == c.K and a._tap_idx.shape[-1] == a.K_out


# ---------------------------------------------------------------------------
# Transposed convolution: the adjoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_transpose_is_the_adjoint(case, stride):
    """<S x, y> = <x, S^T y> with the same weights, and S^T matches the dense A^T."""
    cs = CASES[case]
    conv = make(cs, stride)
    convT = make(cs, stride, transpose=True)
    with torch.no_grad():
        convT.weight.copy_(conv.weight)
    assert convT.weight.shape == (cs["G"], cs["C"], cs["Co"], conv.P)
    assert convT.bias.shape == (cs["C"],)
    np.testing.assert_array_equal(convT.cell_ids_in, conv.cell_ids_out)
    x = torch.randn(cs["B"], cs["C"], conv.K, dtype=torch.float64)
    y = torch.randn(cs["B"], cs["G"] * cs["Co"], conv.K_out, dtype=torch.float64)
    with torch.no_grad():
        lhs = ((conv(x) - conv.bias.view(1, -1, 1)) * y).sum()
        rhs = (x * (convT(y) - convT.bias.view(1, -1, 1))).sum()
    assert abs(float(lhs - rhs)) < 1e-12 * float(lhs.abs())
    A, _ = dense_operator(conv, cs["cell_ids"])
    got = grads(convT, convT, y)
    ref = grads(lambda t: ref_st(A, convT.weight, convT.bias, t), convT, y)
    assert got[0].shape == (cs["B"], cs["C"], conv.K)
    assert max(rel_errs(got, ref)) < 1e-12


def test_transpose_cell_order_and_shapes():
    ids = ragged_patch(4, 3)
    convT = HealPixConvTranspose(4, 3, 2, n_gauges=2, cell_ids=ids, device="cpu", cache_dir=None)
    np.testing.assert_array_equal(convT.cell_ids_out, ids)
    y = torch.randn(5, 6, convT.K_out)
    assert convT(y).shape == (5, 2, ids.size)
    with pytest.raises(ValueError, match="input channels"):
        convT(torch.randn(5, 3, convT.K_out))
    full = HealPixConvTranspose(2, 1, 1, device="cpu", cache_dir=None, use_norm=True)
    assert full(torch.randn(12 * 4)).shape == (12 * 16,)
    with pytest.raises(ValueError, match="shape"):
        full.set_kernel(np.zeros((1, 1, 2, 9)))
    full.set_kernel(np.ones((1, 1, 9)), bias=[0.5])


def test_gather_map_matches_index_add():
    """The PyTorch gather of S^T (used on CUDA) equals the index_add_ scatter."""
    for stride in (1, 2):
        cs = CASES["regional"]
        conv = make(cs, stride)
        gm = cb.build_gather_map(conv._tap_idx, conv._tap_w, conv.K)
        assert sum(len(g[1]) for g in gm) == int((conv._tap_w != 0).sum())
        y = torch.randn(2, cs["G"] * cs["Co"], conv.K_out, dtype=torch.float64)
        bT = torch.randn(cs["C"], dtype=torch.float64)
        a = cb._torch_transpose(y, conv.weight, bT, conv._tap_idx, conv._tap_w, conv.K)
        b = cb._torch_transpose(y, conv.weight, bT, conv._tap_idx, conv._tap_w, conv.K, gm)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-13)


def test_transposed_map_with_strided_geometry():
    conv = make(CASES["regional"], 2)
    ptr, base, tk, tw = cb.build_transposed_map(conv._tap_idx, conv._tap_w, conv.K)
    P, _, G, K_out = conv._tap_idx.shape
    assert ptr.shape == (P, G * conv.K + 1) and int(tk.max()) < K_out
    assert int(base[-1]) == int((conv._tap_w != 0).sum())


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _small(stride, transpose=False):
    cs = dict(level=2, cell_ids=None, G=2, k=3, B=2, C=2, Co=2, gauge="phi")
    return make(cs, stride, transpose=transpose)


@pytest.mark.parametrize("transpose", [False, True])
def test_gradcheck_strided(transpose):
    conv = _small(2, transpose)
    n_in = conv.G * conv.in_channels if transpose else conv.in_channels
    K = conv.K_out if transpose else conv.K
    x = torch.randn(2, n_in, K, dtype=torch.float64, requires_grad=True)
    W = conv.weight.detach().clone().requires_grad_(True)
    b = conv.bias.detach().clone().requires_grad_(True)
    kw = dict(backend="torch")
    if transpose:
        def f(x, W, b):
            return cb.gauge_conv_transpose(x, W, b, conv._tap_idx, conv._tap_w, k_in=conv.K, **kw)
    else:
        def f(x, W, b):
            return cb.gauge_conv(x, W, b, conv._tap_idx, conv._tap_w, **kw)
    assert torch.autograd.gradcheck(f, (x, W, b), eps=1e-6, atol=1e-8)
    assert torch.autograd.gradgradcheck(f, (x, W, b), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# torch.func: vmap, jacrev, jacfwd, jvp
# ---------------------------------------------------------------------------

def _func_case(stride, transpose, backend="torch", device="cpu", dtype=torch.float64):
    conv = HealPixConv(1, 2, 2, kernel_sz=3, n_gauges=2, stride=stride, device=device,
                       dtype=dtype, cache_dir=None)
    idx, w, K = conv._tap_idx, conv._tap_w, conv.K
    if transpose:
        def f(x, W, b):
            return cb.gauge_conv_transpose(x, W, b, idx, w, k_in=K, backend=backend)
        x = torch.randn(3, 4, conv.K_out, device=device, dtype=dtype)
        b = torch.randn(2, device=device, dtype=dtype)
    else:
        def f(x, W, b):
            return cb.gauge_conv(x, W, b, idx, w, backend=backend)
        x = torch.randn(3, 2, K, device=device, dtype=dtype)
        b = torch.randn(4, device=device, dtype=dtype)
    W = torch.randn_like(conv.weight)
    return f, x, W, b


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_torch_func(stride, transpose):
    from torch.func import jacfwd, jacrev, jvp, vmap

    f, x, W, b = _func_case(stride, transpose)
    tol = dict(rtol=1e-10, atol=1e-10)
    # vmap over the input (folded into the batch) and over the weights (loop)
    xs = torch.randn(4, *x.shape, dtype=x.dtype)
    torch.testing.assert_close(vmap(f, in_dims=(0, None, None))(xs, W, b),
                               torch.stack([f(xi, W, b) for xi in xs]), **tol)
    torch.testing.assert_close(vmap(f, in_dims=(1, None, None), out_dims=1)(xs, W, b),
                               torch.stack([f(xs[:, i], W, b) for i in range(xs.shape[1])], 1),
                               **tol)
    Ws = torch.randn(3, *W.shape, dtype=W.dtype)
    torch.testing.assert_close(vmap(f, in_dims=(None, 0, None))(x, Ws, b),
                               torch.stack([f(x, Wi, b) for Wi in Ws]), **tol)
    # Jacobians against torch.autograd.functional (plain reverse mode)
    x1 = x[:1]
    for arg in (0, 1, 2):
        ref = torch.autograd.functional.jacobian(lambda *a: f(*a), (x1, W, b))[arg]
        torch.testing.assert_close(jacrev(f, argnums=arg)(x1, W, b), ref, **tol)
        torch.testing.assert_close(jacfwd(f, argnums=arg)(x1, W, b), ref, **tol)
    # forward mode: the map is bilinear in (x, W)
    tx, tW, tb = torch.randn_like(x), torch.randn_like(W), torch.randn_like(b)
    out, tout = jvp(f, (x, W, b), (tx, tW, tb))
    torch.testing.assert_close(out, f(x, W, b), **tol)
    torch.testing.assert_close(tout, f(tx, W, None) + f(x, tW, None) + tb.view(1, -1, 1), **tol)
    # Hessian-vector products through the double backward (gradient penalty)
    def loss(W):
        return f(x, W, b).square().sum()
    hvp = jvp(jacrev(loss), (W,), (tW,))[1]
    hvp_ref = torch.autograd.functional.hvp(loss, W, tW)[1]
    torch.testing.assert_close(hvp, hvp_ref, **tol)


def test_torch_func_on_the_module():
    from torch.func import functional_call, jacrev, vmap

    conv = HealPixConv(1, 1, 2, n_gauges=2, stride=2, device="cpu", dtype=torch.float64,
                       cache_dir=None)
    params = {k: v.detach() for k, v in conv.named_parameters()}
    x = torch.randn(5, 1, conv.K, dtype=torch.float64)
    jac = jacrev(lambda p, x: functional_call(conv, p, (x,)))(params, x[:1])
    assert jac["weight"].shape == (1, 4, conv.K_out) + tuple(conv.weight.shape)
    per_sample = vmap(lambda xi: conv(xi.unsqueeze(0)).squeeze(0))(x)
    torch.testing.assert_close(per_sample, conv(x), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# CUDA: Triton kernels, PyTorch gather
# ---------------------------------------------------------------------------

@needs_triton
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_triton_strided_and_transpose(case, stride, deterministic):
    cs = dict(CASES[case], C=16, Co=24)
    dev = torch.device("cuda")
    ref_conv = make(cs, stride)
    A, _ = dense_operator(ref_conv, cs["cell_ids"])
    A = A.float().to(dev)
    conv = make(cs, stride, torch.float32, dev, backend="triton", deterministic=deterministic)
    convT = make(cs, stride, torch.float32, dev, transpose=True, backend="triton",
                 deterministic=deterministic)
    x0 = torch.randn(cs["B"], cs["C"], conv.K, device=dev)
    errs = rel_errs(grads(conv, conv, x0),
                    grads(lambda x: ref_s(A, conv.weight, conv.bias, x), conv, x0))
    assert max(errs) < 1e-5, errs
    y0 = torch.randn(cs["B"], cs["G"] * cs["Co"], conv.K_out, device=dev)
    errs = rel_errs(grads(convT, convT, y0),
                    grads(lambda y: ref_st(A, convT.weight, convT.bias, y), convT, y0))
    assert max(errs) < 1e-5, errs


@needs_triton
@pytest.mark.parametrize("transpose", [False, True])
def test_triton_strided_bitwise_reproducible(transpose):
    cs = dict(CASES["full"], C=16, Co=16, B=4)
    dev = torch.device("cuda")
    m = make(cs, 2, torch.float32, dev, transpose=transpose, backend="triton")
    n_in = m.G * m.in_channels if transpose else m.in_channels
    x0 = torch.randn(4, n_in, m.K_out if transpose else m.K, device=dev)
    a, b = grads(m, m, x0), grads(m, m, x0)
    for u, v in zip(a, b):
        assert torch.equal(u, v)


@needs_cuda
@pytest.mark.parametrize("stride", [1, 2])
def test_torch_backend_on_cuda_is_deterministic_by_default(stride):
    """No use_deterministic_algorithms needed: S^T is a gather on CUDA."""
    cs = dict(CASES["regional"], C=16, Co=16)
    dev = torch.device("cuda")
    conv = make(cs, stride, torch.float32, dev, backend="torch")
    assert not torch.are_deterministic_algorithms_enabled()
    x0 = torch.randn(cs["B"], 16, conv.K, device=dev)
    a, b = grads(conv, conv, x0), grads(conv, conv, x0)
    for u, v in zip(a, b):
        assert torch.equal(u, v)
    ref = make(cs, stride)
    with torch.no_grad():
        ref.weight.copy_(conv.weight)
        ref.bias.copy_(conv.bias)
    errs = rel_errs(a, grads(ref, ref, x0.double().cpu()))
    assert max(errs) < 1e-5, errs


@needs_triton
@pytest.mark.parametrize("transpose", [False, True])
def test_triton_torch_func(transpose):
    """vmap, jacrev, jacfwd and jvp on the Triton kernels, against the PyTorch backend."""
    from torch.func import jacfwd, jacrev, jvp, vmap

    f, x, W, b = _func_case(2, transpose, "triton", "cuda", torch.float32)
    g, _, _, _ = _func_case(2, transpose, "torch", "cuda", torch.float32)
    tol = dict(rtol=1e-4, atol=1e-4)
    xs = torch.randn(4, *x.shape, device="cuda")
    torch.testing.assert_close(vmap(f, in_dims=(0, None, None))(xs, W, b),
                               vmap(g, in_dims=(0, None, None))(xs, W, b), **tol)
    for arg in (0, 1):
        torch.testing.assert_close(jacrev(f, argnums=arg)(x[:1], W, b),
                                   jacrev(g, argnums=arg)(x[:1], W, b), **tol)
        torch.testing.assert_close(jacfwd(f, argnums=arg)(x[:1], W, b),
                                   jacfwd(g, argnums=arg)(x[:1], W, b), **tol)
    t = (torch.randn_like(x), torch.randn_like(W), torch.randn_like(b))
    torch.testing.assert_close(jvp(f, (x, W, b), t)[1], jvp(g, (x, W, b), t)[1], **tol)


@needs_cuda
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_float32_geometry_is_the_same_on_cpu_and_cuda(stride, transpose):
    """A float32 layer built on CPU and on CUDA reads the same cells with the
    same weights (to float32 rounding), and gives the same output: the
    geometry is built in float64 on both.  Built in float32, one stencil point
    of this ragged domain (cell 4694), whose neighbours are outside the
    domain, kept a present neighbour with a rounding-size weight that the
    renormalisation raised to 1, where the float64 build falls back to the
    centre: ~20 % output differences."""
    cs = CASES["regional"]
    a = make(cs, stride, torch.float32, "cpu", transpose=transpose, backend="torch")
    b = make(cs, stride, torch.float32, "cuda", transpose=transpose)
    # the same cells with the same weights for every stencil point; the 4
    # neighbour slots may come in another order, so compare the dense stencil
    def dense(c):
        P, _, G, K = c._tap_idx.shape
        D = torch.zeros(P, G, K, c.K, dtype=torch.float64)
        D.scatter_add_(3, c._tap_idx.cpu().long().permute(0, 2, 3, 1),
                       c._tap_w.cpu().double().permute(0, 2, 3, 1))
        return D
    torch.testing.assert_close(dense(a), dense(b), rtol=0, atol=1e-6)
    assert torch.equal(a._sort_order, b._sort_order.cpu())
    np.testing.assert_array_equal(a.cell_ids_out, b.cell_ids_out)
    with torch.no_grad():
        b.weight.copy_(a.weight)
        b.bias.copy_(a.bias)
    n_in = a.G * a.in_channels if transpose else a.in_channels
    x = torch.randn(cs["B"], n_in, a.K_out if transpose else a.K)
    with torch.no_grad():
        ya, yb = a(x), b(x.cuda()).cpu()
    assert float((ya - yb).abs().max() / ya.abs().max()) < 1e-5
